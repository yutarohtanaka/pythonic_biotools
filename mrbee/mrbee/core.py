"""Python implementation of MRBEE and MVMR-BEE (Lorincz-Comi et al., AJHG 2024).

Implements univariable and multivariable Mendelian Randomization with Bias-Corrected Estimating
Equations, subtracting exposure measurement-error variance and sample-overlap covariance (`rho_e` /
`Rxy`), applying an exact truncated-normal Mills-ratio Winner's Curse factor (`gamma_wc`), and
filtering horizontal-pleiotropy outliers via iterative IMRP Sidak testing.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import optimize, stats


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")


def _huber_rlm_no_intercept(
    X: np.ndarray,
    y: np.ndarray,
    k: float = 1.345,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """Huber M-estimation without intercept (`MASS::rlm(y ~ X - 1)` IRLS)."""
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(max_iter):
        r = y - X @ theta
        scale = float(np.median(np.abs(r))) / 0.6744897501960817
        if scale <= 1e-12:
            break
        u = np.abs(r) / scale
        w = np.where(u <= k, 1.0, k / u)
        sw = np.sqrt(w)
        theta_new, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        denom = max(float(np.linalg.norm(theta)), 1e-12)
        if float(np.linalg.norm(theta_new - theta)) / denom < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


def _fit_bee_r_default(
    bx_mat: np.ndarray,
    sx_mat: np.ndarray,
    by_vec: np.ndarray,
    sy_vec: np.ndarray,
    r_xx: np.ndarray,
    r_xy: np.ndarray,
    pleio_p_thresh: float | None,
    max_iter: int,
    tol: float,
) -> dict:
    """Upstream `noahlorinczcomi/MRBEE` v1.0.0 (`MRBEE.IMRP.R` / `MRBEE.IMRP.UV.R`) routine."""
    m, p = bx_mat.shape
    sy2 = sy_vec ** 2

    # Pre-standardize once up front (fixed weights w_j = 1 / sey_j^2, no overdispersion tau^2)
    by_std = by_vec / sy_vec
    bx_std = bx_mat / sy_vec[:, None]
    sigma_xx = sx_mat[:, :, None] * r_xx[None, :, :] * sx_mat[:, None, :]
    sigma_xy = sy_vec[:, None] * sx_mat * r_xy[None, :]
    sigma_xx_std = sigma_xx / sy2[:, None, None]
    sigma_xy_std = sigma_xy / sy2[:, None]

    def _solve_std(valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        bx_v = bx_std[valid_mask]
        by_v = by_std[valid_mask]
        xtx = bx_v.T @ bx_v
        sum_sxx = np.sum(sigma_xx_std[valid_mask], axis=0)
        sum_sxy = np.sum(sigma_xy_std[valid_mask], axis=0)
        xty = bx_v.T @ by_v

        shrink = 1.0
        h_mat = 0.5 * ((xtx - sum_sxx) + (xtx - sum_sxx).T)
        ref_scale = max(float(np.trace(xtx)) / max(p, 1), 1e-30)
        if float(np.linalg.eigvalsh(h_mat).min()) <= 1e-8 * ref_scale:
            for cand in (0.85, 0.70, 0.50, 0.25, 0.0):
                h_try = 0.5 * ((xtx - cand * sum_sxx) + (xtx - cand * sum_sxx).T)
                if float(np.linalg.eigvalsh(h_try).min()) > 1e-8 * ref_scale:
                    shrink = cand
                    h_mat = h_try
                    break
            else:
                shrink = 0.0
                h_mat = xtx + 1e-8 * ref_scale * np.eye(p)

        g_vec = xty - shrink * sum_sxy
        theta_sol = np.linalg.solve(h_mat, g_vec)
        return theta_sol, h_mat, shrink

    # Huber RLM initialization + 3 * MAD residual trimming + ||theta||_2 <= 3 ||theta_init||_2 cap
    theta_init = _huber_rlm_no_intercept(bx_std, by_std)
    norm_init = float(np.linalg.norm(theta_init))
    max_norm = 3.0 * norm_init

    def _cap_theta(th: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(th))
        if norm_init > 0.0 and nrm > max_norm:
            return th * (max_norm / nrm)
        return th

    r_init = by_std - bx_std @ theta_init
    mad_r = 1.4826 * float(np.median(np.abs(r_init - np.median(r_init))))
    if mad_r > 1e-12:
        init_valid = np.abs(r_init) <= 3.0 * mad_r
        valid = init_valid if int(init_valid.sum()) > p + 1 else np.ones(m, dtype=bool)
    else:
        valid = np.ones(m, dtype=bool)

    theta = theta_init.copy()
    converged = False

    for it in range(max_iter):
        theta_old = theta.copy()
        valid_old = valid.copy()
        if int(valid.sum()) <= p:
            break

        theta, _, _ = _solve_std(valid)
        theta = _cap_theta(theta)

        resid_std = by_std - bx_std @ theta
        if pleio_p_thresh is not None and pleio_p_thresh > 0.0 and int(valid.sum()) > p + 1:
            vare = max(float(np.var(resid_std[valid], ddof=1)), 1e-30)
            q_j = (resid_std ** 2) / vare
            p_raw = np.clip(stats.chi2.sf(q_j, df=1), 0.0, 1.0)
            p_adj = -np.expm1(m * np.log1p(-p_raw))
            new_valid = p_adj >= pleio_p_thresh
            if int(new_valid.sum()) > p + 1:
                valid = new_valid

        if it >= 1 and np.array_equal(valid, valid_old) and float(np.max(np.abs(theta - theta_old))) < tol:
            converged = True
            break

    if int(valid.sum()) > p:
        theta, h_mat, shrink = _solve_std(valid)
        theta = _cap_theta(theta)
    else:
        _, h_mat, shrink = _solve_std(np.ones(m, dtype=bool))

    h_inv = np.linalg.pinv(h_mat)
    bx_v = bx_std[valid]
    by_v = by_std[valid]
    sxx_v = sigma_xx_std[valid]
    sxy_v = sigma_xy_std[valid]

    h_ii = np.einsum("ip,pq,iq->i", bx_v, h_inv, bx_v)
    lev_denom = np.maximum(1.0 - h_ii, 0.5)

    bx_by_v = bx_v * by_v[:, None] - shrink * sxy_v
    bx_bx_th_v = bx_v * (bx_v @ theta)[:, None] - shrink * np.einsum("ikl,l->ik", sxx_v, theta)
    psi_v = (bx_by_v - bx_bx_th_v) / lev_denom[:, None]

    n_val = int(valid.sum())
    dof_scale = m / max(n_val - p, 1)
    meat = dof_scale * (psi_v.T @ psi_v)
    cov_theta = h_inv @ meat @ h_inv.T
    se = np.sqrt(np.maximum(np.diag(cov_theta), 1e-30))
    z975 = float(stats.norm.ppf(0.975))
    pvals = 2.0 * stats.norm.sf(np.abs(theta) / se)

    return dict(
        b=theta,
        b_se=se,
        cov=cov_theta,
        pval=pvals,
        b_ci=np.column_stack([theta - z975 * se, theta + z975 * se]),
        n_snps_used=n_val,
        n_outliers=int((~valid).sum()),
        valid_mask=valid,
        tau2=0.0,
        converged=bool(converged),
    )


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
    wc_correct: bool | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Fit univariable MRBEE on summary statistics `(xbeta, xse, ybeta, yse)`."""
    _validate_settings(settings)
    wc_use = (settings == "default") if wc_correct is None else bool(wc_correct)

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
        c_thresh = min_abs_z if (wc_use and min_abs_z >= 3.0 and len(idx) < 0.8 * n_all) else 0.0

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
    gamma_wc = _wc_factor(xb / xs, c_thresh) if wc_use else 1.0

    if settings == "R_default":
        mv_res = _fit_bee_r_default(
            xb[:, None],
            (np.sqrt(gamma_wc) * xs)[:, None],
            yb,
            ys,
            np.ones((1, 1), dtype=np.float64),
            np.array([rho_e_hat], dtype=np.float64),
            pleio_p_thresh,
            max_iter,
            tol,
        )
        b_hat = float(mv_res["b"][0])
        b_se = float(mv_res["b_se"][0])
        z975 = float(stats.norm.ppf(0.975))
        pval = float(min(1.0, 2.0 * stats.norm.sf(abs(b_hat) / max(b_se, 1e-300))))
        return dict(
            b=b_hat,
            b_se=b_se,
            pval=pval,
            b_ci=[b_hat - z975 * b_se, b_hat + z975 * b_se],
            n_snps_used=mv_res["n_snps_used"],
            n_outliers=mv_res["n_outliers"],
            rho_e=rho_e_hat,
            gamma_wc=gamma_wc,
            tau2=0.0,
            converged=mv_res["converged"],
        )

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
        b_hat = float(np.sum((xb * yb - sxy_eff) / sy2) / den0)

    for it in range(max_iter):
        b_old = b_hat
        valid_old = valid.copy()
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
        b_hat = float(num / den)

        # Do not inflate v_null with tau2 on early iterations before outliers are masked
        tau2_null = 0.0 if it < 2 else tau2
        v_null = np.maximum(sy2 + (b_hat ** 2) * (xs ** 2) - 2.0 * b_hat * rho_e_hat * xs * ys + tau2_null, 1e-300)
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

        if it >= 1 and np.array_equal(valid, valid_old) and abs(b_hat - b_old) < tol:
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
        converged=bool(converged),
    )


def fit_mvmr_bee(
    bX: np.ndarray,
    seX: np.ndarray,
    by: np.ndarray,
    sey: np.ndarray,
    Rxy: np.ndarray | None = None,
    pleio_p_thresh: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-6,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Fit Multivariable MRBEE (`MRBEE.IMRP`, Lorincz-Comi et al., AJHG 2024).

    Solves bias-corrected multivariable estimating equations
    `(B_X^T W B_X - sum_j w_j Sigma_XX,j)^{-1} (B_X^T W b_y - sum_j w_j sigma_Xy,j)`
    with iterative IMRP Sidak horizontal-pleiotropy outlier removal and sandwich covariance.

    Parameters
    ----------
    bX, seX : (M, P) array_like
        Instrument-exposure effect sizes and standard errors across `M` SNPs and `P` exposures.
    by, sey : (M,) array_like
        Instrument-outcome effect sizes and standard errors.
    Rxy : optional (P + 1, P + 1) array_like
        Estimation-error correlation matrix ordered `[X_1, ..., X_P, y]`. Defaults to `I_{P+1}`.
    settings : {"default", "R_default"}, default "default"
        `"default"` uses iterative theta-dependent weights with DerSimonian-Laird `tau2`.
        `"R_default"` matches upstream `noahlorinczcomi/MRBEE` v1.0.0 (`MRBEE.IMRP.R`).
    """
    _validate_settings(settings)
    bx_mat = np.asarray(bX, dtype=np.float64)
    if bx_mat.ndim == 1:
        bx_mat = bx_mat[:, None]
    sx_mat = np.asarray(seX, dtype=np.float64).reshape(bx_mat.shape)
    by_vec = np.asarray(by, dtype=np.float64).ravel()
    sy_vec = np.asarray(sey, dtype=np.float64).ravel()

    m, p = bx_mat.shape
    if by_vec.size != m or sy_vec.size != m:
        raise ValueError("bX, seX, by, sey must have matching instrument count M.")
    if m <= p + 1:
        raise ValueError(f"fit_mvmr_bee requires M > P + 1 instruments (got M={m}, P={p}).")
    if np.any(sx_mat <= 0.0) or np.any(sy_vec <= 0.0):
        raise ValueError("seX and sey must be strictly positive.")

    if Rxy is None:
        r_full = np.eye(p + 1, dtype=np.float64)
    else:
        r_full = np.asarray(Rxy, dtype=np.float64)
        if r_full.shape != (p + 1, p + 1):
            raise ValueError(f"Rxy must have shape ({p + 1}, {p + 1}), got {r_full.shape}.")
        r_full = 0.5 * (r_full + r_full.T)

    r_xx = r_full[:p, :p]
    r_xy = r_full[:p, p]

    if settings == "R_default":
        return _fit_bee_r_default(bx_mat, sx_mat, by_vec, sy_vec, r_xx, r_xy, pleio_p_thresh, max_iter, tol)

    # Per-SNP error covariances Sigma_XX[j] (m, p, p) and sigma_Xy[j] (m, p)
    sigma_xx = sx_mat[:, :, None] * r_xx[None, :, :] * sx_mat[:, None, :]
    sigma_xy = sy_vec[:, None] * sx_mat * r_xy[None, :]
    sy2 = sy_vec ** 2

    def _solve_bee(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        xtwx = bx_mat.T @ (bx_mat * weights[:, None])
        sum_sxx = np.einsum("j,jkl->kl", weights, sigma_xx)
        sum_sxy = np.einsum("j,jk->k", weights, sigma_xy)
        xtwy = bx_mat.T @ (weights * by_vec)

        shrink = 1.0
        h_mat = 0.5 * ((xtwx - sum_sxx) + (xtwx - sum_sxx).T)
        ref_scale = max(float(np.trace(xtwx)) / max(p, 1), 1e-30)
        min_eig = float(np.linalg.eigvalsh(h_mat).min())
        if min_eig <= 1e-8 * ref_scale:
            # Cap measurement-error subtraction on ultra-weak exposures so H stays positive-definite
            for cand in (0.85, 0.70, 0.50, 0.25, 0.0):
                h_try = 0.5 * ((xtwx - cand * sum_sxx) + (xtwx - cand * sum_sxx).T)
                if float(np.linalg.eigvalsh(h_try).min()) > 1e-8 * ref_scale:
                    shrink = cand
                    h_mat = h_try
                    break
            else:
                shrink = 0.0
                h_mat = xtwx + 1e-8 * ref_scale * np.eye(p)

        g_vec = xtwy - shrink * sum_sxy
        theta_sol = np.linalg.solve(h_mat, g_vec)
        return theta_sol, h_mat, shrink

    valid = np.ones(m, dtype=bool)
    w_init = 1.0 / sy2
    theta, _, _ = _solve_bee(w_init)
    tau2 = 0.0
    converged = False

    for it in range(max_iter):
        theta_old = theta.copy()
        valid_old = valid.copy()

        quad_xx = np.einsum("k,jkl,l->j", theta, sigma_xx, theta)
        cross_xy = sigma_xy @ theta
        v_res = np.maximum(sy2 + quad_xx - 2.0 * cross_xy + tau2, 1e-300)
        w = np.where(valid, 1.0 / v_res, 0.0)
        if np.sum(w > 0) <= p:
            break
        theta, _, _ = _solve_bee(w)

        quad_xx = np.einsum("k,jkl,l->j", theta, sigma_xx, theta)
        cross_xy = sigma_xy @ theta
        tau2_null = 0.0 if it < 2 else tau2
        v_null = np.maximum(sy2 + quad_xx - 2.0 * cross_xy + tau2_null, 1e-300)
        resid = by_vec - bx_mat @ theta
        q_j = (resid ** 2) / v_null

        if pleio_p_thresh is not None and pleio_p_thresh > 0.0 and m >= p + 4:
            p_cut = 1.0 - (1.0 - min(pleio_p_thresh, 0.5)) ** (1.0 / m)
            chi_cut = float(stats.chi2.isf(max(p_cut, 1e-12), 1))
            new_valid = q_j <= chi_cut
            if int(new_valid.sum()) >= max(p + 2, int(0.5 * m)):
                valid = new_valid

        v0 = np.maximum(sy2 + quad_xx - 2.0 * cross_xy, 1e-300)
        w0 = np.where(valid, 1.0 / v0, 0.0)
        n_val = int(valid.sum())
        if n_val > p + 1 and w0.sum() > 0.0:
            q_tot = float(np.sum(w0 * (resid ** 2)))
            c_dl = float(w0.sum() - (w0 ** 2).sum() / w0.sum())
            tau2 = max(0.0, (q_tot - (n_val - p)) / max(c_dl, 1e-12))

        if it >= 1 and np.array_equal(valid, valid_old) and float(np.max(np.abs(theta - theta_old))) < tol:
            converged = True
            break

    quad_xx = np.einsum("k,jkl,l->j", theta, sigma_xx, theta)
    cross_xy = sigma_xy @ theta
    v_res = np.maximum(sy2 + quad_xx - 2.0 * cross_xy + tau2, 1e-300)
    w = np.where(valid, 1.0 / v_res, 0.0)
    theta, h_mat, shrink = _solve_bee(w)

    # Per-SNP bias-corrected score vectors psi_j (m, p)
    bx_by = bx_mat * by_vec[:, None] - shrink * sigma_xy
    bx_bx_theta = bx_mat * (bx_mat @ theta)[:, None] - shrink * np.einsum("jkl,l->jk", sigma_xx, theta)
    psi = w[:, None] * (bx_by - bx_bx_theta)

    n_val = max(int(valid.sum()), p + 1)
    dof_scale = n_val / max(n_val - p, 1)
    meat = dof_scale * (psi.T @ psi)
    h_inv = np.linalg.pinv(h_mat)
    cov_theta = h_inv @ meat @ h_inv.T
    se = np.sqrt(np.maximum(np.diag(cov_theta), 1e-30))
    z975 = float(stats.norm.ppf(0.975))
    pvals = 2.0 * stats.norm.sf(np.abs(theta) / se)

    return dict(
        b=theta,
        b_se=se,
        cov=cov_theta,
        pval=pvals,
        b_ci=np.column_stack([theta - z975 * se, theta + z975 * se]),
        n_snps_used=int(valid.sum()),
        n_outliers=int((~valid).sum()),
        valid_mask=valid,
        tau2=float(tau2),
        converged=bool(converged),
    )
