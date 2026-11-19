"""Empirical Bayes variance moderation (squeezeVar) and Efron locfdr empirical null."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, ndtr, polygamma
from scipy.stats import f as f_dist
from scipy.stats import norm

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate the ``settings`` parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def trigamma_inverse(x: float | np.ndarray) -> np.ndarray:
    """Invert ``trigamma(y) = x`` via Smyth (2004) Newton iteration."""
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    v = np.atleast_1d(arr).copy()
    out = np.full_like(v, np.nan, dtype=np.float64)

    inf_mask = v <= 1e-12
    out[inf_mask] = np.inf
    large_mask = v > 1e7
    out[large_mask] = 1.0 / np.sqrt(np.maximum(v[large_mask], 1e-30))

    work = np.isfinite(v) & (~inf_mask) & (~large_mask)
    if np.any(work):
        vw = v[work]
        y = np.where(vw > 4.0, 1.0 / np.sqrt(vw), 0.5 + 1.0 / vw)
        for _ in range(8):
            tri = polygamma(1, y)
            tetra = polygamma(2, y)
            # Newton step on g(y) = 1/psi_1(y) - 1/v = 0 (note tetra < 0)
            step = tri * (1.0 - tri / vw) / tetra
            y = np.maximum(y + step, 1e-12)
        out[work] = y

    return out[0] if scalar else out  # type: ignore[return-value]


@dataclass
class SqueezeVarResult:
    """Output of Empirical Bayes quasi-dispersion moderation."""

    df_prior: float | np.ndarray
    s2_prior: float
    s2_post: np.ndarray
    df_total: np.ndarray


def _fit_f_dist_standard(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Smyth (2004) non-robust moment estimation of ``(df_prior, s2_prior)``."""
    ok = np.isfinite(s2) & np.isfinite(df) & (df > 0.0)
    if int(np.sum(ok)) < 2:
        med = float(np.nanmedian(s2)) if np.any(np.isfinite(s2)) else 1.0
        return np.inf, max(med, 1e-8)

    s2_ok = s2[ok].copy()
    df_ok = df[ok]
    pos = s2_ok > 1e-15
    floor_val = 0.5 * float(np.min(s2_ok[pos])) if np.any(pos) else 1e-8
    s2_ok = np.maximum(s2_ok, max(floor_val, 1e-15))

    half_df = 0.5 * df_ok
    e = np.log(s2_ok) - digamma(half_df) + np.log(half_df)
    emean = float(np.mean(e))
    evar = float(np.var(e, ddof=1)) - float(np.mean(polygamma(1, half_df)))

    if evar <= 0.0:
        return np.inf, float(np.exp(emean))

    df_prior = float(2.0 * trigamma_inverse(evar))
    if not np.isfinite(df_prior):
        return np.inf, float(np.exp(emean))

    s2_prior = float(np.exp(emean + digamma(0.5 * df_prior) - np.log(0.5 * df_prior)))
    return df_prior, max(s2_prior, 1e-12)


def _winsorized_gaussian_factors(p_lo: float, p_hi: float) -> tuple[float, float]:
    """Theoretical mean and variance of a standard normal Winsorized at ``(p_lo, 1-p_hi)``."""
    c_lo = float(norm.ppf(p_lo)) if p_lo > 0.0 else -np.inf
    c_hi = float(norm.ppf(1.0 - p_hi)) if p_hi > 0.0 else np.inf
    # E[W] and E[W^2] for Z ~ N(0, 1) clamped to [c_lo, c_hi]
    phi_lo = float(norm.pdf(c_lo)) if np.isfinite(c_lo) else 0.0
    phi_hi = float(norm.pdf(c_hi)) if np.isfinite(c_hi) else 0.0
    ez = (
        (p_lo * c_lo if np.isfinite(c_lo) else 0.0)
        + (p_hi * c_hi if np.isfinite(c_hi) else 0.0)
        + (phi_lo - phi_hi)
    )
    mid_prob = 1.0 - p_lo - p_hi
    int_z2 = (
        mid_prob
        + (c_lo * phi_lo if np.isfinite(c_lo) else 0.0)
        - (c_hi * phi_hi if np.isfinite(c_hi) else 0.0)
    )
    ez2 = (
        (p_lo * (c_lo * c_lo) if np.isfinite(c_lo) else 0.0)
        + (p_hi * (c_hi * c_hi) if np.isfinite(c_hi) else 0.0)
        + int_z2
    )
    return ez, max(ez2 - ez * ez, 1e-8)


def _fit_f_dist_robust(
    s2: np.ndarray,
    df: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[np.ndarray, float]:
    """Phipson et al. (2016) robust ``fitFDistRobustly`` with outlier df reduction."""
    n = len(s2)
    ok = np.isfinite(s2) & np.isfinite(df) & (df > 0.0)
    if int(np.sum(ok)) < 5:
        d0, s02 = _fit_f_dist_standard(s2, df)
        return np.full(n, d0, dtype=np.float64), s02

    s2_ok = s2[ok].copy()
    df_ok = df[ok]
    pos = s2_ok > 1e-12
    floor_val = 0.5 * float(np.min(s2_ok[pos])) if np.any(pos) else 1e-8
    s2_ok = np.maximum(s2_ok, max(floor_val, 1e-12))

    half_df = 0.5 * df_ok
    e = np.log(s2_ok) - digamma(half_df) + np.log(half_df)

    p_lo, p_hi = winsor_tail_p
    q_lo, q_hi = np.quantile(e, [p_lo, 1.0 - p_hi])
    ew = np.clip(e, q_lo, q_hi)

    w_mean_std, w_var_std = _winsorized_gaussian_factors(p_lo, p_hi)
    win_var = float(np.var(ew, ddof=1)) / w_var_std
    win_sd = float(np.sqrt(max(win_var, 1e-12)))
    emean = float(np.mean(ew)) - w_mean_std * win_sd

    evar = win_var - float(np.mean(polygamma(1, half_df)))
    if evar <= 0.0:
        d0_scalar = np.inf
        s2_prior = float(np.exp(emean))
        return np.full(n, d0_scalar, dtype=np.float64), max(s2_prior, 1e-12)

    d0_scalar = float(2.0 * trigamma_inverse(evar))
    if not np.isfinite(d0_scalar):
        s2_prior = float(np.exp(emean))
        return np.full(n, np.inf, dtype=np.float64), max(s2_prior, 1e-12)

    s2_prior = max(
        float(np.exp(emean + digamma(0.5 * d0_scalar) - np.log(0.5 * d0_scalar))),
        1e-12,
    )

    # Identify hypervariable outlier transcripts & reduce their individual df_prior
    df_prior_vec = np.full(n, d0_scalar, dtype=np.float64)
    f_stat = s2_ok / s2_prior
    p_upper = f_dist.sf(f_stat, dfn=df_ok, dfd=d0_scalar)
    # Compare upper tail against uniform order-statistic / Holm bound
    m_ok = len(s2_ok)
    rank_desc = np.argsort(np.argsort(p_upper)) + 1.0
    p_adj = np.minimum(1.0, p_upper * (m_ok / rank_desc))
    outlier = (f_stat > 1.0) & (p_adj < 0.05)
    if np.any(outlier):
        # Match moment variance for extreme outliers so they are shrunk less aggressively
        e_out = e[outlier] - emean
        v_ind = np.maximum(e_out * e_out - polygamma(1, half_df[outlier]), evar)
        d0_ind = np.minimum(d0_scalar, 2.0 * trigamma_inverse(v_ind))
        ok_idx = np.flatnonzero(ok)
        df_prior_vec[ok_idx[outlier]] = d0_ind

    return df_prior_vec, s2_prior


def squeeze_var(
    s2: np.ndarray,
    df: np.ndarray,
    robust: bool | None = None,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
    settings: SettingsMode = "default",
) -> SqueezeVarResult:
    """Empirical Bayes moderation of transcript quasi-dispersions (`limma::squeezeVar`)."""
    validate_settings(settings)
    s2_arr = np.asarray(s2, dtype=np.float64)
    df_arr = np.asarray(df, dtype=np.float64)
    if robust is None:
        robust = settings == "default"

    if robust:
        df_prior_vec, s2_prior = _fit_f_dist_robust(
            s2_arr, df_arr, winsor_tail_p=winsor_tail_p
        )
        fin_d0 = np.isfinite(df_prior_vec)
        d0_safe = np.where(fin_d0, df_prior_vec, 1.0)
        s2_post = np.where(
            fin_d0,
            (d0_safe * s2_prior + df_arr * s2_arr) / (d0_safe + df_arr),
            s2_prior,
        )
        df_total = df_arr + df_prior_vec
        df_prior_out: float | np.ndarray = df_prior_vec
    else:
        df_prior_scalar, s2_prior = _fit_f_dist_standard(s2_arr, df_arr)
        if np.isinf(df_prior_scalar):
            s2_post = np.where(np.isfinite(s2_arr), s2_prior, np.nan)
            df_total = np.where(np.isfinite(s2_arr), np.inf, np.nan)
        else:
            s2_post = (df_prior_scalar * s2_prior + df_arr * s2_arr) / (
                df_prior_scalar + df_arr
            )
            df_total = df_arr + df_prior_scalar
        df_prior_out = df_prior_scalar

    bad = (~np.isfinite(s2_arr)) | (~np.isfinite(df_arr)) | (df_arr <= 0.0)
    s2_post = np.where(bad, np.nan, s2_post)
    df_total = np.where(bad, np.nan, df_total)
    return SqueezeVarResult(
        df_prior=df_prior_out,
        s2_prior=s2_prior,
        s2_post=s2_post,
        df_total=df_total,
    )


def _truncated_normal_mle(
    z_sub: np.ndarray, a: float, b: float, mu_init: float, sig_init: float
) -> tuple[float, float]:
    """Conditional MLE of N(mu0, sigma0^2) truncated to [a, b] (Efron 2004)."""
    if len(z_sub) < 10 or b <= a:
        return mu_init, sig_init

    def neg_cond_ll(params: np.ndarray) -> float:
        mu, log_sig = float(params[0]), float(params[1])
        sig = float(np.exp(log_sig))
        za = (a - mu) / sig
        zb = (b - mu) / sig
        prob = float(ndtr(zb) - ndtr(za))
        if prob <= 1e-12:
            return 1e12
        resid = (z_sub - mu) / sig
        ll = -len(z_sub) * (log_sig + np.log(prob)) - 0.5 * float(np.sum(resid * resid))
        return -ll

    x0 = np.array([np.clip(mu_init, a, b), np.log(np.clip(sig_init, 0.25, 4.0))])
    bounds = [(a - 0.5, b + 0.5), (np.log(0.25), np.log(4.5))]
    opt = minimize(neg_cond_ll, x0, method="L-BFGS-B", bounds=bounds)
    if opt.success and np.all(np.isfinite(opt.x)):
        return float(opt.x[0]), float(np.exp(opt.x[1]))
    return mu_init, sig_init


def _poisson_glm_histogram_peak(z_valid: np.ndarray) -> tuple[float, float]:
    """Fit Efron's locfdr Poisson-GLM polynomial density to histogram bin counts."""
    n = len(z_valid)
    z_min, z_max = float(np.min(z_valid)), float(np.max(z_valid))
    if z_max - z_min < 1e-6:
        return 0.0, 1.0

    n_bins = int(np.clip(n // 2, 25, 120))
    counts, edges = np.histogram(z_valid, bins=n_bins, range=(z_min, z_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Standardized Chebyshev/Legendre-like basis up to degree df_poly
    df_poly = 7 if n >= 120 else 5
    z_mid = 0.5 * (z_min + z_max)
    z_half = 0.5 * (z_max - z_min)
    u = (centers - z_mid) / z_half
    basis_cols = [np.ones_like(u)] + [u**k for k in range(1, df_poly + 1)]
    q_mat, _ = np.linalg.qr(np.column_stack(basis_cols), mode="reduced")
    # Ensure column 0 is constant
    x_glm = np.column_stack([np.ones_like(u), q_mat[:, 1:]])

    y_bin = counts.astype(np.float64)
    beta = np.zeros(x_glm.shape[1], dtype=np.float64)
    beta[0] = np.log(max(float(np.mean(y_bin)), 0.5))

    for _ in range(25):
        eta = np.clip(x_glm @ beta, -20.0, 20.0)
        mu = np.exp(eta)
        xtwx = (x_glm * mu[:, None]).T @ x_glm + 1e-6 * np.eye(x_glm.shape[1])
        grad = x_glm.T @ (y_bin - mu)
        step = np.linalg.solve(xtwx, grad)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-7:
            break

    log_f = x_glm @ beta
    # Restrict mode search to central quantiles [10%, 90%] so extreme tail spikes don't hijack null
    q10, q90 = np.quantile(z_valid, [0.10, 0.90])
    central_bins = (centers >= q10) & (centers <= q90)
    if not np.any(central_bins):
        central_bins = np.ones_like(centers, dtype=bool)
    mode_idx = int(np.flatnonzero(central_bins)[np.argmax(log_f[central_bins])])
    mu_peak = float(centers[mode_idx])

    # Local weighted quadratic fit log_f(x) ~ c0 + c1*(x - mu_peak) + c2*(x - mu_peak)^2
    mad_sd = float(np.median(np.abs(z_valid - np.median(z_valid))) * 1.4826)
    win = max(1.2 * max(mad_sd, 0.5), 1.0)
    loc_mask = np.abs(centers - mu_peak) <= win
    if int(np.sum(loc_mask)) >= 4:
        dx = centers[loc_mask] - mu_peak
        w_loc = np.exp(log_f[loc_mask])
        design_q = np.column_stack([np.ones_like(dx), dx, dx * dx])
        coef_q = np.linalg.lstsq(
            design_q * np.sqrt(w_loc[:, None]),
            log_f[loc_mask] * np.sqrt(w_loc),
            rcond=None,
        )[0]
        c1, c2 = float(coef_q[1]), float(coef_q[2])
        if c2 < -1e-4:
            sig_cm = float(np.clip(np.sqrt(-1.0 / (2.0 * c2)), 0.25, 4.0))
            mu_cm = float(np.clip(mu_peak - c1 / (2.0 * c2), q10, q90))
            return mu_cm, sig_cm

    return mu_peak, max(mad_sd, 0.5)


def estimate_empirical_null(
    z_scores: np.ndarray,
    settings: SettingsMode = "default",
) -> tuple[float, float]:
    """Estimate Efron's empirical null ``(mu0, sigma0)`` on central z-scores."""
    validate_settings(settings)
    z_arr = np.asarray(z_scores, dtype=np.float64)
    z_valid = z_arr[np.isfinite(z_arr)]
    n_valid = len(z_valid)

    if settings == "default":
        if n_valid < 50:
            return 0.0, 1.0
        med = float(np.median(z_valid))
        mad_sd = float(np.median(np.abs(z_valid - med)) * 1.4826)
        if mad_sd < 1e-6:
            return 0.0, 1.0
        half_w = float(np.clip(1.8 * mad_sd, 1.0, 3.5))
        a, b = med - half_w, med + half_w
        z_in = z_valid[(z_valid >= a) & (z_valid <= b)]
        return _truncated_normal_mle(z_in, a, b, med, mad_sd)

    # R_default: Poisson-GLM histogram regression + locfdr mlests truncated-normal MLE
    if n_valid < 20:
        return 0.0, 1.0
    mu_cm, sig_cm = _poisson_glm_histogram_peak(z_valid)
    half_w = float(np.clip(1.6 * sig_cm, 1.0, 3.5))
    a, b = mu_cm - half_w, mu_cm + half_w
    z_in = z_valid[(z_valid >= a) & (z_valid <= b)]
    if len(z_in) >= 15:
        return _truncated_normal_mle(z_in, a, b, mu_cm, sig_cm)
    return mu_cm, sig_cm
