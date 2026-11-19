"""
Empirical Bayes variance moderation (`fit_f_dist`, `squeeze_var`, `trigamma_inverse`)
and NaN-preserving Benjamini-Hochberg FDR + z_std transformation
(Smyth 2004; Phipson et al. 2016; Hoffman & Roussos 2021).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import digamma, ndtri_exp, polygamma
from scipy.stats import chi2, f as f_dist, false_discovery_control, norm
from scipy.stats import t as t_dist


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Invert the trigamma function ``y -> polygamma(1, y)`` on ``(0, inf)``.

    Returns ``0.0`` for ``+np.inf``, ``np.nan`` for ``np.nan`` or ``<= 0``, preserves
    input shape, and uses 4 vectorized Newton-Raphson iterations on convex
    ``1 / psi_1(y)`` (Smyth 2004).
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)

    pos_inf = arr == np.inf
    if np.any(pos_inf):
        out[pos_inf] = 0.0

    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return out

    xm = arr[valid]
    ym = np.empty_like(xm)
    large = xm > 1e7
    small = xm < 1e-6
    mid = ~(large | small)

    ym[large] = 1.0 / np.sqrt(xm[large])
    ym[small] = 1.0 / xm[small] + 0.5
    if np.any(mid):
        x_mid = xm[mid]
        y = np.where(x_mid > 10.0, 1.0 / np.sqrt(x_mid), 0.5 + 1.0 / x_mid)
        for _ in range(4):
            tri = polygamma(1, y)
            y = y + tri * (1.0 - tri / x_mid) / polygamma(2, y)
        ym[mid] = y

    out[valid] = ym
    return out


def _lowess_tricube_1d(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_eval: np.ndarray,
    span: float = 0.5,
    n_anchors: int = 50,
) -> np.ndarray:
    """Cleveland tricube locally-linear trend evaluated on quantile anchors."""
    n = x_obs.size
    if n < 4 or np.nanmax(x_obs) <= np.nanmin(x_obs):
        return np.full(x_eval.shape, float(np.mean(y_obs)), dtype=np.float64)

    m_anc = min(n, n_anchors)
    qs = np.linspace(0.0, 1.0, m_anc)
    anchors = np.unique(np.quantile(x_obs, qs))
    if anchors.size < 2:
        return np.full(x_eval.shape, float(np.mean(y_obs)), dtype=np.float64)

    k = max(4, min(n, int(np.ceil(span * n))))
    anchor_fits = np.empty_like(anchors, dtype=np.float64)

    for idx, a0 in enumerate(anchors):
        dx = x_obs - a0
        adx = np.abs(dx)
        h = float(np.partition(adx, k - 1)[k - 1])
        if h <= 0.0:
            h = float(np.max(adx))
        if h <= 0.0:
            anchor_fits[idx] = float(np.mean(y_obs))
            continue
        u = np.clip(adx / (h * 1.0001), 0.0, 1.0)
        w = (1.0 - u**3) ** 3
        sw = float(np.sum(w))
        sx = float(np.sum(w * dx))
        sy = float(np.sum(w * y_obs))
        sxx = float(np.sum(w * dx * dx))
        sxy = float(np.sum(w * dx * y_obs))
        denom = sw * sxx - sx * sx
        if denom > 1e-12 * sw * sxx and sw > 0.0:
            anchor_fits[idx] = (sxx * sy - sx * sxy) / denom
        elif sw > 0.0:
            anchor_fits[idx] = sy / sw
        else:
            anchor_fits[idx] = float(np.mean(y_obs))

    x_fill = np.where(np.isfinite(x_eval), x_eval, float(np.median(x_obs)))
    return np.interp(x_fill, anchors, anchor_fits)


def _winsorized_gaussian_moments(
    r: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[float, float]:
    """Consistent mean and variance from asymmetric Winsorized residuals (Phipson et al. 2016)."""
    p_lo, p_hi = float(winsor_tail_p[0]), float(winsor_tail_p[1])
    if not (0.0 <= p_lo < 0.5 and 0.0 <= p_hi < 0.5 and (p_lo + p_hi) < 0.9):
        raise ValueError("Invalid winsor_tail_p; require 0 <= p_lo, p_hi < 0.5 and p_lo + p_hi < 0.9.")

    q_lo, q_hi = np.quantile(r, [p_lo, 1.0 - p_hi])
    r_win = np.clip(r, q_lo, q_hi)
    mean_win = float(np.mean(r_win))
    var_win = float(np.sum((r_win - mean_win) ** 2) / max(r.size - 1, 1))

    z_lo = float(norm.ppf(p_lo)) if p_lo > 0.0 else -np.inf
    z_hi = float(norm.ppf(1.0 - p_hi)) if p_hi > 0.0 else np.inf
    phi_lo = float(norm.pdf(z_lo)) if np.isfinite(z_lo) else 0.0
    phi_hi = float(norm.pdf(z_hi)) if np.isfinite(z_hi) else 0.0

    t1_lo = z_lo * p_lo if p_lo > 0.0 else 0.0
    t1_hi = z_hi * p_hi if p_hi > 0.0 else 0.0
    mu_w = t1_lo + t1_hi + phi_lo - phi_hi

    t2_lo = (z_lo**2) * p_lo + z_lo * phi_lo if p_lo > 0.0 else 0.0
    t2_hi = (z_hi**2) * p_hi - z_hi * phi_hi if p_hi > 0.0 else 0.0
    ez2_w = t2_lo + t2_hi + (1.0 - p_lo - p_hi)
    var_w = max(ez2_w - mu_w**2, 1e-8)

    sigma_hat = np.sqrt(var_win / var_w)
    mu_hat = mean_win - sigma_hat * mu_w
    return float(mu_hat), float(sigma_hat**2)


def fit_f_dist(
    var_g: np.ndarray,
    df1: int | float | np.ndarray,
    covariate: Sequence[float] | np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Moment-match a scaled-inverse-chi-squared prior to per-feature sample variances.

    Supports scalar or gene-specific ``df1`` (such as Satterthwaite degrees of freedom),
    mean-variance trend (``covariate``), and Phipson et al. (2016) ``robust=True``
    winsorized prior estimation with per-gene outlier ``df_prior`` downweighting.
    """
    v = np.asarray(var_g, dtype=np.float64).ravel()
    n_genes = v.size
    df1_arr = np.broadcast_to(np.asarray(df1, dtype=np.float64), (n_genes,)).copy()

    has_cov = covariate is not None
    cov_arr = np.asarray(covariate, dtype=np.float64).ravel() if has_cov else None
    if cov_arr is not None and cov_arr.size != n_genes:
        raise ValueError(f"covariate length ({cov_arr.size}) must match var_g length ({n_genes}).")

    valid = np.isfinite(v) & (v > 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0)
    if cov_arr is not None:
        valid = valid & np.isfinite(cov_arr)
    if np.any(valid):
        v_ref = float(np.quantile(v[valid], 0.90))
        if v_ref > 0.0:
            valid = valid & (v > 1e-24 * v_ref)

    n_val = int(np.sum(valid))
    if n_val < 2:
        s20_fallback = float(v[valid][0]) if n_val == 1 else 1.0
        s20_ret: float | np.ndarray = np.full(n_genes, s20_fallback, dtype=np.float64) if has_cov else s20_fallback
        df0_ret: float | np.ndarray = np.full(n_genes, np.inf, dtype=np.float64) if robust else float("inf")
        return s20_ret, df0_ret

    v_val = v[valid]
    d_val = df1_arr[valid]
    z = np.log(v_val)
    e = z - digamma(d_val / 2.0) + np.log(d_val / 2.0)
    mean_trigamma_d1 = float(np.mean(polygamma(1, d_val / 2.0)))

    if cov_arr is not None:
        e_trend_all = _lowess_tricube_1d(cov_arr[valid], e, cov_arr, span=0.5)
        resid_e = e - e_trend_all[valid]
        if robust and n_val >= 4:
            mu_r, var_r = _winsorized_gaussian_moments(resid_e, winsor_tail_p=winsor_tail_p)
            e_prior_all = e_trend_all + mu_r
            total_var_e = var_r * (n_val / max(n_val - 2, 1))
        else:
            e_prior_all = e_trend_all
            total_var_e = float(np.sum(resid_e**2) / max(n_val - 2, 1))
        evar = total_var_e - mean_trigamma_d1
    else:
        if robust and n_val >= 4:
            emean, total_var_e = _winsorized_gaussian_moments(e, winsor_tail_p=winsor_tail_p)
        else:
            emean = float(np.mean(e))
            total_var_e = float(np.sum((e - emean) ** 2) / (n_val - 1))
        evar = total_var_e - mean_trigamma_d1

    if evar > 0.0:
        df0_scalar = float(2.0 * trigamma_inverse(evar))
        shift = float(digamma(df0_scalar / 2.0) - np.log(df0_scalar / 2.0))
    else:
        df0_scalar = float("inf")
        shift = 0.0

    if cov_arr is not None:
        s20_out: float | np.ndarray = np.exp(e_prior_all + shift)
        s20_vec = s20_out
    else:
        s20_scalar = float(np.exp(emean + shift))
        s20_out = s20_scalar
        s20_vec = np.full(n_genes, s20_scalar, dtype=np.float64)

    if not robust:
        return s20_out, df0_scalar

    df0_vec = np.full(n_genes, df0_scalar, dtype=np.float64)
    check_mask = valid & (s20_vec > 0.0)
    m_chk = int(np.sum(check_mask))
    if m_chk >= 2:
        f_ratio = v[check_mask] / s20_vec[check_mask]
        d_chk = df1_arr[check_mask]
        if np.isfinite(df0_scalar):
            p_upper = f_dist.sf(f_ratio, d_chk, df0_scalar)
        else:
            p_upper = chi2.sf(d_chk * f_ratio, d_chk)

        asc_order = np.argsort(f_ratio, kind="mergesort")
        f_sorted = f_ratio[asc_order]
        p_sorted = p_upper[asc_order]
        rank_desc = np.arange(m_chk, 0, -1, dtype=np.float64)
        u_exp = rank_desc / float(m_chk)
        w_sorted = np.where(f_sorted > 1.0, np.clip(p_sorted / u_exp, 0.0, 1.0), 1.0)
        w_sorted = np.minimum.accumulate(w_sorted)
        weights = np.empty_like(w_sorted)
        weights[asc_order] = w_sorted

        if np.isfinite(df0_scalar):
            df0_chk = df0_scalar * weights
        else:
            df0_chk = np.where(weights >= 1.0 - 1e-6, np.inf, 100.0 * weights)
        df0_vec[check_mask] = df0_chk

    return s20_out, df0_vec


def squeeze_var(
    var_g: np.ndarray,
    df1: int | float | np.ndarray,
    covariate: Sequence[float] | np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Shrink per-feature residual variances toward a fitted Empirical Bayes prior."""
    v = np.asarray(var_g, dtype=np.float64)
    orig_shape = v.shape
    v_flat = v.ravel()
    n_genes = v_flat.size

    scalar_df1 = np.ndim(df1) == 0
    df1_arr = np.broadcast_to(np.asarray(df1, dtype=np.float64), (n_genes,)).copy()
    valid_gene = np.isfinite(v_flat) & (v_flat >= 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0)

    s20, df0 = fit_f_dist(
        v_flat,
        df1_arr if not scalar_df1 else float(df1),
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    s20_arr = np.broadcast_to(np.asarray(s20, dtype=np.float64), (n_genes,))
    df0_arr = np.broadcast_to(np.asarray(df0, dtype=np.float64), (n_genes,))

    is_inf_df0 = np.isinf(df0_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        shrunk_finite = (df0_arr * s20_arr + df1_arr * v_flat) / (df0_arr + df1_arr)
    var_post_flat = np.where(is_inf_df0, s20_arr, shrunk_finite)
    var_post_flat = np.where(valid_gene, var_post_flat, np.nan)

    if scalar_df1 and np.ndim(df0) == 0:
        df_total: float | np.ndarray = float("inf") if np.isinf(df0) else float(df0) + float(df1)
    else:
        df_total_arr = np.where(is_inf_df0, np.inf, df0_arr + df1_arr)
        df_total = np.where(valid_gene, df_total_arr, np.nan)

    return var_post_flat.reshape(orig_shape), s20, df0, df_total


def _moderated_p_and_fdr(
    t_stat: np.ndarray,
    df_total: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact t / normal two-sided p-values, NaN-preserving BH FDR, and z_std."""
    n = t_stat.size
    df_arr = np.broadcast_to(np.asarray(df_total, dtype=np.float64), (n,))
    valid = np.isfinite(t_stat) & (~np.isnan(df_arr)) & (df_arr > 0.0)

    p_val = np.full(n, np.nan, dtype=np.float64)
    z_std = np.full(n, np.nan, dtype=np.float64)
    if np.any(valid):
        t_v = t_stat[valid]
        abs_t = np.abs(t_v)
        df_v = df_arr[valid]
        inf_m = np.isinf(df_v)
        pv = np.empty_like(abs_t)
        zv = np.empty_like(abs_t)
        if np.any(inf_m):
            pv[inf_m] = 2.0 * norm.sf(abs_t[inf_m])
            zv[inf_m] = t_v[inf_m]
        if np.any(~inf_m):
            pv[~inf_m] = 2.0 * t_dist.sf(abs_t[~inf_m], df_v[~inf_m])
            log_p_half = t_dist.logsf(abs_t[~inf_m], df_v[~inf_m])
            z_abs = np.maximum(-ndtri_exp(log_p_half), 0.0)
            zv[~inf_m] = np.sign(t_v[~inf_m]) * z_abs
        p_val[valid] = np.clip(pv, 0.0, 1.0)
        z_std[valid] = zv

    adj_p = np.full(n, np.nan, dtype=np.float64)
    valid_p = np.isfinite(p_val)
    if np.any(valid_p):
        adj_p[valid_p] = false_discovery_control(p_val[valid_p], method="bh")

    return p_val, adj_p, z_std
