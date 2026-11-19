"""Python implementation of MRBEE (Lorincz-Comi et al., AJHG 2024).

Implements Mendelian Randomization with Bias-Corrected Estimating Equations, subtracting
exposure measurement-error variance and sample-overlap covariance (`rho_e`), applying an
exact truncated-normal Mills-ratio Winner's Curse factor (`gamma_wc`), and filtering
horizontal-pleiotropy outliers via iterative IMRP Sidak testing.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _estimate_overlap_rho_e(zx: np.ndarray, zy: np.ndarray) -> float:
    ok = np.isfinite(zx) & np.isfinite(zy) & (np.abs(zx) < 1.96) & (np.abs(zy) < 1.96)
    if ok.sum() < 30:
        return 0.0
    r = float(np.corrcoef(zx[ok], zy[ok])[0, 1])
    return float(np.clip(r, -0.95, 0.95)) if np.isfinite(r) else 0.0


def _wc_factor(zx_sel: np.ndarray, c_thresh: float) -> float:
    if c_thresh <= 1e-6 or len(zx_sel) == 0:
        return 1.0
    m2_obs = float(np.mean(zx_sel ** 2))

    def _mill_ratio(v):
        t = c_thresh / np.sqrt(max(v, 1.0))
        return 1.0 + t * np.exp(stats.norm.logpdf(t) - stats.norm.logcdf(-t))

    def _obj(v):
        return v * _mill_ratio(v) - m2_obs

    if _obj(1.0) >= 0:
        return float(_mill_ratio(1.001))
    hi = max(m2_obs, 2.0)
    while _obj(hi) < 0 and hi < 1e6:
        hi *= 2.0
    v_hat = float(optimize.brentq(_obj, 1.0, hi, xtol=1e-6))
    return float(_mill_ratio(v_hat))


def fit_mrbee(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    prune: np.ndarray | None = None,
    inst_p: float | None = None,
    rho_e: float | None = None,
    pleio_p_thresh: float = 0.05,
    max_iter: int = 30,
    tol: float = 1e-6,
    wc_correct: bool = True,
) -> dict:
    """Fit MRBEE on summary statistics `(xbeta, xse, ybeta, yse)`."""
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=np.float64) for v in (xbeta, xse, ybeta, yse))
    n_all = len(xbeta)
    zx_all = xbeta / xse
    zy_all = ybeta / yse
    rho_e_hat = _estimate_overlap_rho_e(zx_all, zy_all) if rho_e is None else float(rho_e)

    idx = np.arange(n_all) if prune is None else np.asarray(prune, dtype=int)
    if inst_p is not None and inst_p < 1.0:
        c_thresh = float(stats.norm.isf(inst_p / 2.0))
        idx = idx[np.abs(zx_all[idx]) >= c_thresh]
    else:
        min_abs_z = float(np.min(np.abs(zx_all[idx]))) if len(idx) > 0 else 0.0
        c_thresh = min_abs_z if (wc_correct and min_abs_z >= 3.0 and len(idx) < 0.8 * n_all) else 0.0

    if len(idx) < 3:
        return dict(
            b=np.nan,
            b_se=np.nan,
            pval=np.nan,
            b_ci=[np.nan, np.nan],
            n_snps_used=len(idx),
            n_outliers=0,
            rho_e=rho_e_hat,
            gamma_wc=1.0,
            tau2=0.0,
            converged=False,
        )

    xb, xs, yb, ys = xbeta[idx], xse[idx], ybeta[idx], yse[idx]
    gamma_wc = _wc_factor(xb / xs, c_thresh) if wc_correct else 1.0
    sx2_eff = gamma_wc * (xs ** 2)
    sxy_eff = gamma_wc * rho_e_hat * xs * ys
    sy2 = ys ** 2

    m_iv = len(xb)
    valid = np.ones(m_iv, dtype=bool)
    b_hat = 0.0
    tau2 = 0.0
    converged = False

    den0 = float(np.sum((xb ** 2 - sx2_eff) / sy2))
    if den0 > 0:
        b_hat = float(np.clip(np.sum((xb * yb - sxy_eff) / sy2) / den0, -3.0, 3.0))

    it = 0
    for it in range(max_iter):
        b_old = b_hat
        v_res = np.maximum(sy2 + (b_hat ** 2) * (xs ** 2) - 2.0 * b_hat * rho_e_hat * xs * ys + tau2, 1e-300)
        w = np.where(valid, 1.0 / v_res, 0.0)

        den = float(np.sum(w * (xb ** 2 - sx2_eff)))
        if den <= 1e-12:
            sx2_cap = np.minimum(sx2_eff, 0.85 * (xb ** 2))
            sxy_cap = (sx2_cap / np.maximum(xs ** 2, 1e-300)) * rho_e_hat * xs * ys
            den = float(np.sum(w * (xb ** 2 - sx2_cap)))
            num = float(np.sum(w * (xb * yb - sxy_cap)))
        else:
            num = float(np.sum(w * (xb * yb - sxy_eff)))

        if den <= 1e-12:
            break
        b_hat = float(np.clip(num / den, -5.0, 5.0))

        v_null = np.maximum(sy2 + (b_hat ** 2) * (xs ** 2) - 2.0 * b_hat * rho_e_hat * xs * ys + tau2, 1e-300)
        q_j = ((yb - b_hat * xb) ** 2) / v_null
        if pleio_p_thresh is not None and pleio_p_thresh > 0 and m_iv >= 5:
            p_cut = 1.0 - (1.0 - min(pleio_p_thresh, 0.5)) ** (1.0 / m_iv)
            chi_cut = stats.chi2.isf(max(p_cut, 1e-12), 1)
            new_valid = q_j <= chi_cut
            if new_valid.sum() >= max(3, int(0.5 * m_iv)):
                valid = new_valid

        v0 = np.maximum(sy2 + (b_hat ** 2) * (xs ** 2) - 2.0 * b_hat * rho_e_hat * xs * ys, 1e-300)
        w0 = np.where(valid, 1.0 / v0, 0.0)
        n_val = int(valid.sum())
        if n_val > 2 and w0.sum() > 0:
            q_tot = float(np.sum(w0 * ((yb - b_hat * xb) ** 2)))
            c_dl = float(w0.sum() - (w0 ** 2).sum() / w0.sum())
            tau2 = max(0.0, (q_tot - (n_val - 1)) / max(c_dl, 1e-12))

        if abs(b_hat - b_old) < tol:
            converged = True
            break

    v_res = np.maximum(sy2 + (b_hat ** 2) * (xs ** 2) - 2.0 * b_hat * rho_e_hat * xs * ys + tau2, 1e-300)
    w = np.where(valid, 1.0 / v_res, 0.0)
    sx2_use = sx2_eff if np.sum(w * (xb ** 2 - sx2_eff)) > 1e-12 else np.minimum(sx2_eff, 0.85 * (xb ** 2))
    sxy_use = (sx2_use / np.maximum(xs ** 2, 1e-300)) * rho_e_hat * xs * ys
    den = float(np.sum(w * (xb ** 2 - sx2_use)))
    psi = w * ((xb * yb - sxy_use) - b_hat * (xb ** 2 - sx2_use))
    n_val = max(int(valid.sum()), 2)
    dof_scale = n_val / max(n_val - 1, 1)
    b_se = float(np.sqrt(dof_scale * np.sum(psi ** 2)) / max(abs(den), 1e-12))

    z975 = float(stats.norm.ppf(0.975))
    pval = float(min(1.0, 2.0 * stats.norm.sf(abs(b_hat) / max(b_se, 1e-300))))
    return dict(
        b=b_hat,
        b_se=b_se,
        pval=pval,
        b_ci=[b_hat - z975 * b_se, b_hat + z975 * b_se],
        n_snps_used=int(valid.sum()),
        n_outliers=int((~valid).sum()),
        rho_e=rho_e_hat,
        gamma_wc=gamma_wc,
        tau2=tau2,
        converged=bool(converged or it < max_iter),
    )
