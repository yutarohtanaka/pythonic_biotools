"""Python implementation of MR-APSS (Hu et al., Nature Communications 2022).

Fits the foreground-background bivariate mixture model with exact selection-truncation
correction (`|z_X| > t_0`) and bivariate LD-score background covariance (`Omega_0 * l_j + C`)
accounting for polygenic pleiotropy, population stratification, and sample overlap.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp


def estimate_background_ldsc(
    zx: np.ndarray,
    zy: np.ndarray,
    ldscore: np.ndarray | None = None,
    two_step_cut: float = 30.0,
    rho_e_override: float | None = None,
) -> dict:
    """Estimate background genetic covariance `Omega` and error/sample-overlap intercept `C`."""
    zx = np.asarray(zx, dtype=np.float64)
    zy = np.asarray(zy, dtype=np.float64)
    m = len(zx)
    l = np.ones(m, dtype=np.float64) if ldscore is None else np.asarray(ldscore, dtype=np.float64)
    ok = np.isfinite(zx) & np.isfinite(zy) & np.isfinite(l)
    zx, zy, l = zx[ok], zy[ok], np.maximum(l[ok], 1.0)

    prod_xx = zx * zx
    prod_yy = zy * zy
    prod_xy = zx * zy

    def _fit_1d(y_prod, keep_mask):
        if np.std(l[keep_mask]) < 1e-6 * max(np.mean(l[keep_mask]), 1.0):
            icpt = float(np.median(y_prod[keep_mask]))
            return (max(icpt, 0.0) if y_prod is not prod_xy else icpt), 0.0
        w = 1.0 / l[keep_mask]
        sw = np.sqrt(w)
        X = np.column_stack([np.ones(keep_mask.sum()), l[keep_mask]]) * sw[:, None]
        icpt, _ = np.linalg.lstsq(X, y_prod[keep_mask] * sw, rcond=None)[0]
        icpt = float(icpt)
        w_all = 1.0 / l
        num = float((w_all * (y_prod - icpt) * l).sum())
        den = float((w_all * l * l).sum())
        slope = num / den if den > 0 else 0.0
        return icpt, slope

    keep_x = prod_xx <= two_step_cut
    keep_y = prod_yy <= two_step_cut
    keep_xy = keep_x & keep_y

    cx, sx = _fit_1d(prod_xx, keep_x if keep_x.sum() >= 50 else np.ones_like(keep_x))
    cy, sy = _fit_1d(prod_yy, keep_y if keep_y.sum() >= 50 else np.ones_like(keep_y))
    cxy, sxy = _fit_1d(prod_xy, keep_xy if keep_xy.sum() >= 50 else np.ones_like(keep_xy))

    cx = float(np.clip(cx, 0.8, 1.5))
    cy = float(np.clip(cy, 0.8, 1.5))
    sx = float(max(sx, 0.0))
    sy = float(max(sy, 0.0))
    max_sxy = np.sqrt(sx * sy) if (sx > 0 and sy > 0) else 0.0
    sxy = float(np.clip(sxy, -0.95 * max_sxy, 0.95 * max_sxy))

    if rho_e_override is not None:
        cxy = float(rho_e_override) * np.sqrt(cx * cy)
    else:
        null_1 = np.abs(zx) < 0.8
        null_2 = (np.abs(zx) >= 0.8) & (np.abs(zx) < 2.0)
        if null_1.sum() >= 100 and null_2.sum() >= 100 and abs(sxy) < 1e-3:
            s1 = float((zx[null_1] * zy[null_1]).sum() / (zx[null_1] ** 2).sum())
            s2 = float((zx[null_2] * zy[null_2]).sum() / (zx[null_2] ** 2).sum())
            cxy = float(s1 - 0.25 * (s2 - s1))
        elif (np.abs(zx) < 1.5).sum() >= 100:
            null_x = np.abs(zx) < 1.5
            slope_null = float((zx[null_x] * zy[null_x]).sum() / (zx[null_x] ** 2).sum())
            if abs(cxy - slope_null) > 0.08 and abs(sxy) < 1e-3:
                cxy = slope_null
        lim = 0.95 * np.sqrt(cx * cy)
        cxy = float(np.clip(cxy, -lim, lim))

    C = np.array([[cx, cxy], [cxy, cy]], dtype=np.float64)
    Omega = np.array([[sx, sxy], [sxy, sy]], dtype=np.float64)
    return dict(C=C, Omega=Omega, rho_e=float(cxy / np.sqrt(cx * cy)))


def _biv_norm_trunc_loglik(zx, zy, v00, v01, v11, t0: float) -> np.ndarray:
    det = np.maximum(v00 * v11 - v01 * v01, 1e-12)
    q = (v11 * zx * zx - 2.0 * v01 * zx * zy + v00 * zy * zy) / det
    log_num = -np.log(2.0 * np.pi) - 0.5 * np.log(det) - 0.5 * q
    log_den = np.log(2.0) + stats.norm.logcdf(-t0 / np.sqrt(np.maximum(v00, 1e-12))) if t0 > 0 else 0.0
    return log_num - log_den


def fit_mrapss(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    prune: np.ndarray | None = None,
    ldscore: np.ndarray | None = None,
    iv_p_thresh: float = 5e-5,
    rho_e: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """Fit MR-APSS on genome-wide exposure `(xbeta, xse)` and outcome `(ybeta, yse)` summary statistics."""
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=np.float64) for v in (xbeta, xse, ybeta, yse))
    n_all = len(xbeta)
    zx_all = xbeta / xse
    zy_all = ybeta / yse
    l_all = np.ones(n_all, dtype=np.float64) if ldscore is None else np.maximum(np.asarray(ldscore, dtype=np.float64), 1.0)

    bg = estimate_background_ldsc(zx_all, zy_all, l_all, rho_e_override=rho_e)
    C, Omega = bg["C"], bg["Omega"]

    idx_pool = np.arange(n_all) if prune is None else np.asarray(prune, dtype=int)
    t0 = float(stats.norm.isf(iv_p_thresh / 2.0)) if (iv_p_thresh is not None and iv_p_thresh < 1.0) else 0.0
    sel = idx_pool[np.abs(zx_all[idx_pool]) >= t0]
    if len(sel) < 6:
        k_keep = min(max(20, len(idx_pool) // 10), len(idx_pool))
        order = np.argsort(-np.abs(zx_all[idx_pool]))
        sel = np.sort(idx_pool[order[:k_keep]])
        t0 = float(np.min(np.abs(zx_all[sel]))) * 0.999 if len(sel) > 0 else 0.0

    zx, zy = zx_all[sel], zy_all[sel]
    c = xse[sel] / yse[sel]
    l_iv = l_all[sel]
    m_iv = len(sel)

    bg00 = Omega[0, 0] * l_iv + C[0, 0]
    bg01 = Omega[0, 1] * l_iv + C[0, 1]
    bg11 = Omega[1, 1] * l_iv + C[1, 1]
    ll_bg = _biv_norm_trunc_loglik(zx, zy, bg00, bg01, bg11, t0)

    sx2_init = float(np.clip(np.median(np.maximum(zx * zx - bg00, 1.0) / l_iv), 0.5, 500.0))
    num_init = np.sum((zx * zy - bg01) * c)
    den_init = np.sum(np.maximum(zx * zx - bg00, 0.5) * (c * c))
    b_init = float(np.clip(num_init / max(den_init, 1e-6), -2.0, 2.0))
    res_y = zy - b_init * c * zx
    tau2_init = float(np.clip(np.median(np.maximum(res_y * res_y - bg11 - (b_init * c) ** 2 * bg00, 0.05) / l_iv), 0.01, 50.0))

    def _unpack(params):
        b = float(params[0])
        pi0 = float(1.0 / (1.0 + np.exp(-np.clip(params[1], -15.0, 15.0))))
        sx2 = float(np.exp(np.clip(params[2], -6.0, 8.0)))
        tau2 = float(np.exp(np.clip(params[3], -10.0, 6.0)))
        return b, pi0, sx2, tau2

    def _per_snp_loglik(params):
        b, pi0, sx2, tau2 = _unpack(params)
        fg00 = bg00 + l_iv * sx2
        fg01 = bg01 + l_iv * (b * c * sx2)
        fg11 = bg11 + l_iv * (b * b * c * c * sx2 + tau2)
        ll_fg = _biv_norm_trunc_loglik(zx, zy, fg00, fg01, fg11, t0)
        p_fg = np.clip(pi0, 1e-6, 1.0 - 1e-6)
        return logsumexp(np.column_stack([np.log(p_fg) + ll_fg, np.log(1.0 - p_fg) + ll_bg]), axis=1)

    def _neg_obj(params):
        _, pi0, _, _ = _unpack(params)
        ll = _per_snp_loglik(params)
        prior = 0.5 * (np.log(pi0) + np.log(1.0 - pi0)) - 0.01 * (params[2] ** 2 + params[3] ** 2)
        return -float(np.sum(ll) + prior)

    starts = [
        np.array([b_init, 0.5, np.log(sx2_init), np.log(tau2_init)]),
        np.array([0.0, 0.0, np.log(sx2_init), np.log(tau2_init)]),
        np.array([b_init * 0.5, 1.0, np.log(sx2_init), np.log(max(tau2_init * 0.2, 0.01))]),
    ]
    best_sol = min(
        (
            optimize.minimize(_neg_obj, x0, method="L-BFGS-B", bounds=[(-5.0, 5.0), (-6.0, 6.0), (-5.0, 8.0), (-9.0, 6.0)], options={"maxiter": max_iter, "ftol": tol})
            for x0 in starts
        ),
        key=lambda s: s.fun,
    )
    p_hat = best_sol.x
    b_hat, pi0_hat, sx2_hat, tau2_hat = _unpack(p_hat)

    eps = 1e-4
    k_dim = len(p_hat)
    scores = np.zeros((m_iv, k_dim), dtype=np.float64)
    for d in range(k_dim):
        dp = np.zeros(k_dim)
        dp[d] = eps
        scores[:, d] = (_per_snp_loglik(p_hat + dp) - _per_snp_loglik(p_hat - dp)) / (2.0 * eps)

    H = np.zeros((k_dim, k_dim), dtype=np.float64)
    for i in range(k_dim):
        for j in range(i, k_dim):
            ei, ej = np.zeros(k_dim), np.zeros(k_dim)
            ei[i], ej[j] = eps, eps
            H[i, j] = H[j, i] = (-_neg_obj(p_hat + ei + ej) + _neg_obj(p_hat + ei - ej) + _neg_obj(p_hat - ei + ej) - (-_neg_obj(p_hat - ei - ej))) / (4.0 * eps * eps)

    J = scores.T @ scores
    I_obs = -H + 1e-4 * np.eye(k_dim)
    try:
        inv_I = np.linalg.inv(I_obs)
        V_sand = inv_I @ J @ inv_I.T
        se_sand = float(np.sqrt(max(V_sand[0, 0], 1.0 / max(I_obs[0, 0], 1e-6))))
    except np.linalg.LinAlgError:
        se_sand = float(1.0 / np.sqrt(max(J[0, 0], 1e-6)))

    z975 = float(stats.norm.ppf(0.975))
    pval = float(min(1.0, 2.0 * stats.norm.sf(abs(b_hat) / max(se_sand, 1e-300))))
    return dict(
        b=b_hat,
        b_se=se_sand,
        pval=pval,
        b_ci=[b_hat - z975 * se_sand, b_hat + z975 * se_sand],
        n_snps_used=m_iv,
        pi0=pi0_hat,
        sigma_x2=sx2_hat,
        tau_y2=tau2_hat,
        rho_e=bg["rho_e"],
        C=C.tolist(),
        Omega=Omega.tolist(),
        converged=bool(best_sol.success or best_sol.status == 0),
    )
