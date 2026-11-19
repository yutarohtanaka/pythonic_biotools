"""Python implementation of MR-APSS (Hu et al., Nature Communications 2022).

Fits the foreground-background bivariate mixture model with exact selection-truncation
correction (`|z_X| > t_0`) and bivariate LD-score background covariance (`Omega_0 * l_j + C`)
accounting for polygenic pleiotropy, population stratification, and sample overlap.
"""
from __future__ import annotations

from typing import Literal
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

_CHI2_DF1_MEDIAN = float(stats.chi2.ppf(0.5, df=1))  # 0.454936423119572


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")


def estimate_background_ldsc(
    zx: np.ndarray,
    zy: np.ndarray,
    ldscore: np.ndarray | None = None,
    two_step_cut: float = 30.0,
    rho_e_override: float | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Estimate background genetic covariance `Omega` and error/sample-overlap intercept `C`."""
    _validate_settings(settings)
    zx = np.asarray(zx, dtype=np.float64)
    zy = np.asarray(zy, dtype=np.float64)
    m = len(zx)
    l = np.ones(m, dtype=np.float64) if ldscore is None else np.asarray(ldscore, dtype=np.float64)
    ok = np.isfinite(zx) & np.isfinite(zy) & np.isfinite(l)
    zx, zy, l = zx[ok], zy[ok], np.maximum(l[ok], 1.0)

    prod_xx = zx * zx
    prod_yy = zy * zy
    prod_xy = zx * zy

    keep_x = prod_xx <= two_step_cut
    keep_y = prod_yy <= two_step_cut
    keep_xy = keep_x & keep_y
    mask_x = keep_x if keep_x.sum() >= 50 else np.ones_like(keep_x)
    mask_y = keep_y if keep_y.sum() >= 50 else np.ones_like(keep_y)
    mask_xy = keep_xy if keep_xy.sum() >= 50 else np.ones_like(keep_xy)

    if settings == "R_default":
        is_const_l = np.std(l) < 1e-6 * max(float(np.mean(l)), 1.0)
        if is_const_l:
            cx = max(float(np.median(prod_xx) / _CHI2_DF1_MEDIAN), 1e-4)
            cy = max(float(np.median(prod_yy) / _CHI2_DF1_MEDIAN), 1e-4)
            sx = 0.0
            sy = 0.0
            cxy = float(np.mean(prod_xy[mask_xy]))
            sxy = 0.0
        else:
            def _irwls_marginal(y_prod: np.ndarray, k_mask: np.ndarray) -> tuple[float, float]:
                lk = l[k_mask]
                yk = y_prod[k_mask]
                icpt = 1.0
                slope = max(float((np.mean(yk) - 1.0) / np.mean(lk)), 0.0)
                for _ in range(3):
                    a_k = np.maximum(max(icpt, 1e-4) + max(slope, 0.0) * lk, 1e-6)
                    w_k = 1.0 / (lk * np.maximum(a_k * a_k + a_k * a_k, 1e-12))
                    sw = np.sqrt(w_k)
                    X = np.column_stack([np.ones(lk.size), lk]) * sw[:, None]
                    sol = np.linalg.lstsq(X, yk * sw, rcond=None)[0]
                    icpt, slope = float(sol[0]), float(sol[1])
                for _ in range(3):
                    a_all = np.maximum(max(icpt, 1e-4) + max(slope, 0.0) * l, 1e-6)
                    w_all = 1.0 / (l * np.maximum(a_all * a_all + a_all * a_all, 1e-12))
                    den = float(np.sum(w_all * l * l))
                    slope = float(np.sum(w_all * l * (y_prod - icpt))) / den if den > 0.0 else 0.0
                return icpt, slope

            cx, sx = _irwls_marginal(prod_xx, mask_x)
            cy, sy = _irwls_marginal(prod_yy, mask_y)

            lk = l[mask_xy]
            yk = prod_xy[mask_xy]
            ax_k = np.maximum(max(cx, 1e-4) + max(sx, 0.0) * lk, 1e-6)
            by_k = np.maximum(max(cy, 1e-4) + max(sy, 0.0) * lk, 1e-6)
            cxy = 0.0
            sxy = float(np.mean(yk) / np.mean(lk))
            for _ in range(3):
                c_k = cxy + sxy * lk
                w_k = 1.0 / (lk * np.maximum(ax_k * by_k + c_k * c_k, 1e-12))
                sw = np.sqrt(w_k)
                X = np.column_stack([np.ones(lk.size), lk]) * sw[:, None]
                sol = np.linalg.lstsq(X, yk * sw, rcond=None)[0]
                cxy, sxy = float(sol[0]), float(sol[1])

            ax_all = np.maximum(max(cx, 1e-4) + max(sx, 0.0) * l, 1e-6)
            by_all = np.maximum(max(cy, 1e-4) + max(sy, 0.0) * l, 1e-6)
            for _ in range(3):
                c_all = cxy + sxy * l
                w_all = 1.0 / (l * np.maximum(ax_all * by_all + c_all * c_all, 1e-12))
                den = float(np.sum(w_all * l * l))
                sxy = float(np.sum(w_all * l * (prod_xy - cxy))) / den if den > 0.0 else 0.0

        cx = float(max(cx, 1e-4))
        cy = float(max(cy, 1e-4))
        sx = float(max(sx, 0.0))
        sy = float(max(sy, 0.0))
        max_sxy = np.sqrt(sx * sy) if (sx > 0.0 and sy > 0.0) else 0.0
        sxy = float(np.clip(sxy, -0.999 * max_sxy, 0.999 * max_sxy))
        if rho_e_override is not None:
            cxy = float(rho_e_override) * np.sqrt(cx * cy)
        else:
            lim = 0.999 * np.sqrt(cx * cy)
            cxy = float(np.clip(cxy, -lim, lim))
        C = np.array([[cx, cxy], [cxy, cy]], dtype=np.float64)
        Omega = np.array([[sx, sxy], [sxy, sy]], dtype=np.float64)
        return dict(C=C, Omega=Omega, rho_e=float(cxy / np.sqrt(cx * cy)))

    def _fit_1d(y_prod, keep_mask):
        if np.std(l[keep_mask]) < 1e-6 * max(np.mean(l[keep_mask]), 1.0):
            if y_prod is prod_xy:
                return float(np.mean(y_prod[keep_mask])), 0.0
            icpt = float(np.median(y_prod[keep_mask]) / _CHI2_DF1_MEDIAN)
            return max(icpt, 0.0), 0.0
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

    cx, sx = _fit_1d(prod_xx, mask_x)
    cy, sy = _fit_1d(prod_yy, mask_y)
    cxy, sxy = _fit_1d(prod_xy, mask_xy)

    cx = float(np.clip(cx, 0.8, 1.5))
    cy = float(np.clip(cy, 0.8, 1.5))
    sx = float(max(sx, 0.0))
    sy = float(max(sy, 0.0))
    max_sxy = np.sqrt(sx * sy) if (sx > 0 and sy > 0) else 0.0
    sxy = float(np.clip(sxy, -0.95 * max_sxy, 0.95 * max_sxy))

    if rho_e_override is not None:
        cxy = float(rho_e_override) * np.sqrt(cx * cy)
    else:
        if abs(sxy) < 1e-6 and (np.abs(zx) <= 2.0).sum() >= 100:
            null_x = np.abs(zx) <= 2.0
            denom_nx = float((zx[null_x] ** 2).sum())
            if denom_nx > 0.0:
                cxy = float(cx * (zx[null_x] * zy[null_x]).sum() / denom_nx)
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
    allow_threshold_relaxation: bool = False,
    max_iter: int = 200,
    tol: float | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Fit MR-APSS on genome-wide exposure `(xbeta, xse)` and outcome `(ybeta, yse)` summary statistics."""
    _validate_settings(settings)
    tol_use = (1e-6 if settings == "default" else 1e-8) if tol is None else float(tol)

    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=np.float64) for v in (xbeta, xse, ybeta, yse))
    n_all = len(xbeta)
    if n_all == 0:
        raise ValueError("Empty summary statistics passed to fit_mrapss.")
    zx_all = xbeta / xse
    zy_all = ybeta / yse
    l_all = np.ones(n_all, dtype=np.float64) if ldscore is None else np.maximum(np.asarray(ldscore, dtype=np.float64), 1.0)

    bg = estimate_background_ldsc(
        zx_all,
        zy_all,
        None if ldscore is None else l_all,
        rho_e_override=rho_e,
        settings=settings,
    )
    C, Omega = bg["C"], bg["Omega"]

    idx_pool = np.arange(n_all) if prune is None else np.asarray(prune, dtype=int)
    if len(idx_pool) == 0:
        raise ValueError("No candidate SNPs available in prune index pool.")

    t0 = float(stats.norm.isf(iv_p_thresh / 2.0)) if (iv_p_thresh is not None and iv_p_thresh < 1.0) else 0.0
    sel = idx_pool[np.abs(zx_all[idx_pool]) >= t0]
    if len(sel) < 6:
        warnings.warn(
            f"Only {len(sel)} instrument(s) pass iv_p_thresh={iv_p_thresh} (< 6 recommended).",
            UserWarning,
        )
        if allow_threshold_relaxation:
            k_keep = min(max(20, len(idx_pool) // 10), len(idx_pool))
            order = np.argsort(-np.abs(zx_all[idx_pool]))
            sel = np.sort(idx_pool[order[:k_keep]])
            if len(sel) == 0:
                raise ValueError("0 instruments available after threshold relaxation.")
            t0 = float(np.min(np.abs(zx_all[sel]))) * 0.999
        elif len(sel) == 0:
            raise ValueError(
                f"0 instruments pass iv_p_thresh={iv_p_thresh}; relax iv_p_thresh or set allow_threshold_relaxation=True."
            )

    zx, zy = zx_all[sel], zy_all[sel]
    c = xse[sel] / yse[sel]
    l_iv = l_all[sel]
    m_iv = len(sel)

    bg00 = Omega[0, 0] * l_iv + C[0, 0]
    bg01 = Omega[0, 1] * l_iv + C[0, 1]
    bg11 = Omega[1, 1] * l_iv + C[1, 1]
    ll_bg = _biv_norm_trunc_loglik(zx, zy, bg00, bg01, bg11, t0)

    sx2_init = float(np.clip(np.median(np.maximum(zx * zx - bg00, 1.0) / l_iv), 0.5, 500.0))
    num_init = float(np.sum((zx * zy - bg01) * c))
    den_init = float(np.sum(np.maximum(zx * zx - bg00, 0.5) * (c * c)))
    raw_b_init = num_init / max(den_init, 1e-6)
    b_bound = max(50.0, 10.0 * abs(raw_b_init))
    b_init = float(np.clip(raw_b_init, -0.8 * b_bound, 0.8 * b_bound))
    res_y = zy - b_init * c * zx
    tau2_init = float(np.clip(np.median(np.maximum(res_y * res_y - bg11 - (b_init * c) ** 2 * bg00, 0.05) / l_iv), 0.01, 50.0))

    is_r_default = settings == "R_default"

    def _unpack(params):
        b = float(params[0])
        if is_r_default:
            pi0 = float(1.0 / (1.0 + np.exp(-np.clip(params[1], -20.0, 20.0))))
            sx2 = float(np.exp(np.clip(params[2], -16.0, 12.0)))
            tau2 = float(np.exp(np.clip(params[3], -20.0, 10.0)))
        else:
            pi0 = float(1.0 / (1.0 + np.exp(-np.clip(params[1], -15.0, 15.0))))
            sx2 = float(np.exp(np.clip(params[2], -6.0, 8.0)))
            tau2 = float(np.exp(np.clip(params[3], -10.0, 6.0)))
        return b, pi0, sx2, tau2

    def _fg_loglik(b: float, sx2: float, tau2: float) -> np.ndarray:
        fg00 = bg00 + l_iv * sx2
        fg01 = bg01 + l_iv * (b * c * sx2)
        fg11 = bg11 + l_iv * (b * b * c * c * sx2 + tau2)
        return _biv_norm_trunc_loglik(zx, zy, fg00, fg01, fg11, t0)

    def _per_snp_loglik(params):
        b, pi0, sx2, tau2 = _unpack(params)
        ll_fg = _fg_loglik(b, sx2, tau2)
        eps_pi = 1e-10 if is_r_default else 1e-6
        p_fg = np.clip(pi0, eps_pi, 1.0 - eps_pi)
        return logsumexp(np.column_stack([np.log(p_fg) + ll_fg, np.log(1.0 - p_fg) + ll_bg]), axis=1)

    def _neg_obj(params):
        ll = _per_snp_loglik(params)
        if is_r_default:
            return -float(np.sum(ll))
        _, pi0, _, _ = _unpack(params)
        prior = 0.5 * (np.log(pi0) + np.log(1.0 - pi0)) - 0.01 * (params[2] ** 2 + params[3] ** 2)
        return -float(np.sum(ll) + prior)

    if is_r_default:
        full_bounds = [(-b_bound, b_bound), (-18.0, 18.0), (-15.0, 12.0), (-18.0, 10.0)]
        opt_opts = {
            "maxiter": max(max_iter, 500),
            "ftol": min(tol_use * 1e-4, 1e-13),
            "gtol": min(tol_use, 1e-8),
        }

        def _em_warmstart(x_start: np.ndarray, fix_b_zero: bool = False) -> np.ndarray:
            cur = x_start.copy()
            if fix_b_zero:
                cur[0] = 0.0
            ll_prev = -np.inf
            for _ in range(min(max_iter, 60)):
                b_c, pi_c, sx2_c, tau2_c = _unpack(cur)
                ll_fg = _fg_loglik(0.0 if fix_b_zero else b_c, sx2_c, tau2_c)
                p_fg = np.clip(pi_c, 1e-10, 1.0 - 1e-10)
                la = np.log(p_fg) + ll_fg
                lb = np.log(1.0 - p_fg) + ll_bg
                l_mix = np.logaddexp(la, lb)
                ll_cur = float(np.sum(l_mix))
                if abs(ll_cur - ll_prev) < tol_use * max(1.0, abs(ll_cur)):
                    break
                ll_prev = ll_cur
                gamma = np.exp(la - l_mix)
                pi_new = float(np.clip(np.mean(gamma), 1e-8, 1.0 - 1e-8))
                cur[1] = float(np.clip(np.log(pi_new / (1.0 - pi_new)), -18.0, 18.0))
                if fix_b_zero:
                    res_m = optimize.minimize(
                        lambda v: -float(np.sum(gamma * _fg_loglik(0.0, float(np.exp(v[0])), float(np.exp(v[1]))))),
                        cur[2:],
                        method="L-BFGS-B",
                        bounds=full_bounds[2:],
                        options={"maxiter": 15, "ftol": 1e-11},
                    )
                    cur[2:] = res_m.x
                else:
                    res_m = optimize.minimize(
                        lambda v: -float(np.sum(gamma * _fg_loglik(float(v[0]), float(np.exp(v[1])), float(np.exp(v[2]))))),
                        np.array([cur[0], cur[2], cur[3]]),
                        method="L-BFGS-B",
                        bounds=[full_bounds[0], full_bounds[2], full_bounds[3]],
                        options={"maxiter": 15, "ftol": 1e-11},
                    )
                    cur[0], cur[2], cur[3] = res_m.x[0], res_m.x[1], res_m.x[2]
            return cur

        raw_starts = [
            np.array([b_init, 0.0, np.log(sx2_init), np.log(tau2_init)]),
            np.array([0.0, 0.0, np.log(sx2_init), np.log(tau2_init)]),
            np.array([b_init * 0.5, 1.0, np.log(sx2_init), np.log(max(tau2_init * 0.1, 1e-4))]),
        ]
        starts = [_em_warmstart(raw_starts[0], fix_b_zero=False)] + raw_starts
        best_sol = min(
            (optimize.minimize(_neg_obj, x0, method="L-BFGS-B", bounds=full_bounds, options=opt_opts) for x0 in starts),
            key=lambda s: s.fun,
        )

        null_bounds = full_bounds[1:]
        em_null = _em_warmstart(np.array([0.0, 0.0, np.log(sx2_init), np.log(tau2_init)]), fix_b_zero=True)[1:]
        null_starts = [
            em_null,
            best_sol.x[1:].copy(),
            np.array([0.0, np.log(sx2_init), np.log(tau2_init)]),
        ]
        best_null = min(
            (
                optimize.minimize(
                    lambda u: _neg_obj(np.array([0.0, u[0], u[1], u[2]])),
                    u0,
                    method="L-BFGS-B",
                    bounds=null_bounds,
                    options=opt_opts,
                )
                for u0 in null_starts
            ),
            key=lambda s: s.fun,
        )
        p_null = np.array([0.0, best_null.x[0], best_null.x[1], best_null.x[2]])
        polished = optimize.minimize(_neg_obj, p_null, method="L-BFGS-B", bounds=full_bounds, options=opt_opts)
        if polished.fun < best_sol.fun:
            best_sol = polished
    else:
        full_bounds = [(-b_bound, b_bound), (-6.0, 6.0), (-5.0, 8.0), (-9.0, 6.0)]
        starts = [
            np.array([b_init, 0.5, np.log(sx2_init), np.log(tau2_init)]),
            np.array([0.0, 0.0, np.log(sx2_init), np.log(tau2_init)]),
            np.array([b_init * 0.5, 1.0, np.log(sx2_init), np.log(max(tau2_init * 0.2, 0.01))]),
        ]
        best_sol = min(
            (
                optimize.minimize(_neg_obj, x0, method="L-BFGS-B", bounds=full_bounds, options={"maxiter": max_iter, "ftol": tol_use})
                for x0 in starts
            ),
            key=lambda s: s.fun,
        )

        # Profile null model (b = 0 fixed) for exact Likelihood Ratio Test (Hu et al. 2022)
        null_bounds = full_bounds[1:]
        null_starts = [
            best_sol.x[1:].copy(),
            np.array([0.0, np.log(sx2_init), np.log(tau2_init)]),
        ]
        best_null = min(
            (
                optimize.minimize(
                    lambda u: _neg_obj(np.array([0.0, u[0], u[1], u[2]])),
                    u0,
                    method="L-BFGS-B",
                    bounds=null_bounds,
                    options={"maxiter": max_iter, "ftol": tol_use},
                )
                for u0 in null_starts
            ),
            key=lambda s: s.fun,
        )
        p_null = np.array([0.0, best_null.x[0], best_null.x[1], best_null.x[2]])
        if best_null.fun < best_sol.fun:
            polished = optimize.minimize(_neg_obj, p_null, method="L-BFGS-B", bounds=full_bounds, options={"maxiter": max_iter, "ftol": tol_use})
            if polished.fun < best_sol.fun:
                best_sol = polished

    p_hat = best_sol.x
    b_hat, pi0_hat, sx2_hat, tau2_hat = _unpack(p_hat)

    ll_full = float(np.sum(_per_snp_loglik(p_hat)))
    ll_null = float(np.sum(_per_snp_loglik(p_null)))
    lrt_stat = max(0.0, 2.0 * (ll_full - ll_null))
    p_lrt = float(stats.chi2.sf(lrt_stat, df=1))

    eps = 1e-4
    k_dim = len(p_hat)
    scores = np.zeros((m_iv, k_dim), dtype=np.float64)
    for d in range(k_dim):
        dp = np.zeros(k_dim)
        dp[d] = eps
        scores[:, d] = (_per_snp_loglik(p_hat + dp) - _per_snp_loglik(p_hat - dp)) / (2.0 * eps)

    I_obs = np.zeros((k_dim, k_dim), dtype=np.float64)
    f0 = _neg_obj(p_hat)
    for i in range(k_dim):
        ei = np.zeros(k_dim)
        ei[i] = eps
        I_obs[i, i] = (_neg_obj(p_hat + ei) - 2.0 * f0 + _neg_obj(p_hat - ei)) / (eps * eps)
        for j in range(i + 1, k_dim):
            ej = np.zeros(k_dim)
            ej[j] = eps
            I_obs[i, j] = I_obs[j, i] = (
                _neg_obj(p_hat + ei + ej)
                - _neg_obj(p_hat + ei - ej)
                - _neg_obj(p_hat - ei + ej)
                + _neg_obj(p_hat - ei - ej)
            ) / (4.0 * eps * eps)

    J = scores.T @ scores
    eig_min = float(np.linalg.eigvalsh(0.5 * (I_obs + I_obs.T)).min())
    reg = max(1e-4, -eig_min + 1e-4) if eig_min <= 0.0 else 1e-4
    I_reg = 0.5 * (I_obs + I_obs.T) + reg * np.eye(k_dim)
    try:
        inv_I = np.linalg.inv(I_reg)
        V_sand = inv_I @ J @ inv_I.T
        se_sand = float(np.sqrt(max(V_sand[0, 0], 1.0 / max(I_reg[0, 0], 1e-6))))
        se_hess = float(np.sqrt(max(inv_I[0, 0], 1.0 / max(I_reg[0, 0], 1e-6))))
    except np.linalg.LinAlgError:
        se_sand = float(1.0 / np.sqrt(max(J[0, 0], 1e-6)))
        se_hess = se_sand

    z975 = float(stats.norm.ppf(0.975))
    if is_r_default:
        b_se_out = float(abs(b_hat) / np.sqrt(max(lrt_stat, 1e-12))) if abs(b_hat) > 0.0 else se_hess
        pval_out = p_lrt
    else:
        b_se_out = se_sand
        pval_out = float(min(1.0, 2.0 * stats.norm.sf(abs(b_hat) / max(se_sand, 1e-300))))

    return dict(
        b=b_hat,
        b_se=b_se_out,
        pval=pval_out,
        p=pval_out,
        p_lrt=p_lrt,
        lrt_stat=lrt_stat,
        b_ci=[b_hat - z975 * b_se_out, b_hat + z975 * b_se_out],
        n_snps_used=m_iv,
        pi0=pi0_hat,
        sigma_x2=sx2_hat,
        tau_y2=tau2_hat,
        rho_e=bg["rho_e"],
        C=C.tolist(),
        Omega=Omega.tolist(),
        converged=bool(best_sol.success or best_sol.status == 0),
    )
