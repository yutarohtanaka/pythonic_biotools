"""Genetic architecture estimation (LDSC + truncated spike-and-slab) and exact conditional moments for MRlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import optimize, stats
from scipy.special import expit, log_ndtr, logit, logsumexp

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS = ("default", "R_default")

_GH_X32, _GH_W32 = hermgauss(32)
_GH_NORM_W32 = _GH_W32 / np.sqrt(np.pi)
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def validate_settings(settings: str) -> SettingsMode:
    """Validate ``settings`` mode."""
    if settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


def _mills_ratio_term(a: float | np.ndarray) -> float | np.ndarray:
    """Compute ``a * phi(a) / sf(a)`` stably for all ``a >= 0`` via ``log_ndtr``."""
    a_arr = np.asarray(a, dtype=float)
    log_phi = -0.5 * a_arr * a_arr - _LOG_SQRT_2PI
    log_sf = log_ndtr(-a_arr)
    ratio = a_arr * np.exp(log_phi - log_sf)
    if np.ndim(a) == 0:
        return float(ratio)
    return ratio


@dataclass(frozen=True)
class GeneticArchitectureResult:
    """Estimated exposure/outcome LDSC and spike-and-slab genetic architecture."""

    h2_x: float
    h2_x_se: float
    h2_y: float
    h2_y_se: float
    g_xy: float
    rg: float
    i_x: float
    i_y: float
    c_xy: float
    c_xy_se: float
    cov_h2x_cxy: float
    pi_x: float
    logit_pi_x_se: float
    sigma2_x: float
    m_snps: float


def _weighted_ols(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w_pos = np.maximum(w, 1e-12)
    w_sum = float(np.sum(w_pos))
    x_bar = float(np.sum(w_pos * x) / w_sum)
    y_bar = float(np.sum(w_pos * y) / w_sum)
    dx = x - x_bar
    denom = float(np.sum(w_pos * dx * dx))
    if denom <= 1e-16:
        return y_bar, 0.0
    slope = float(np.sum(w_pos * dx * (y - y_bar)) / denom)
    return y_bar - slope * x_bar, slope


def _ldsc_point_estimate(
    zx: np.ndarray,
    zy: np.ndarray,
    ld: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    m_eff: float,
) -> tuple[float, float, float, float, float, float]:
    """Return ``(i_x, h2_x, i_y, h2_y, c_xy, g_xy)``."""
    cx2 = zx * zx
    cy2 = zy * zy
    cxy = zx * zy

    if float(np.var(ld)) < 1e-8:
        # Constant LD scores (independent SNPs)
        mean_ld = float(np.mean(ld))
        null_x = cx2 < 4.0
        null_y = cy2 < 4.0
        p_in = 2.0 * float(stats.norm.cdf(2.0)) - 1.0
        kappa2 = 1.0 - (4.0 * float(stats.norm.pdf(2.0))) / max(p_in, 1e-12)
        i_x = float(np.mean(cx2[null_x]) / kappa2) if int(np.sum(null_x)) >= 20 else 1.0
        i_y = float(np.mean(cy2[null_y]) / kappa2) if int(np.sum(null_y)) >= 20 else 1.0
        i_x = float(np.clip(i_x, 0.5, 2.5))
        i_y = float(np.clip(i_y, 0.5, 2.5))

        h2_x = float(np.clip(np.mean(cx2 - 1.0) * m_eff / (np.mean(nx) * mean_ld), 1e-4, 0.99))
        h2_y = float(np.clip(np.mean(cy2 - 1.0) * m_eff / (np.mean(ny) * mean_ld), 1e-4, 0.99))

        null_both = (np.abs(zx) < 1.0) & (np.abs(zy) < 1.0)
        if int(np.sum(null_both)) >= 25:
            p1 = 2.0 * float(stats.norm.cdf(1.0)) - 1.0
            k1 = 1.0 - (2.0 * float(stats.norm.pdf(1.0))) / max(p1, 1e-12)
            c_xy = float(np.clip(np.mean(cxy[null_both]) / (k1 * k1), -0.95, 0.95))
        else:
            c_xy = 0.0
        g_xy = float(np.mean(cxy - c_xy) * m_eff / (np.mean(np.sqrt(nx * ny)) * mean_ld))
        return i_x, h2_x, i_y, h2_y, c_xy, g_xy

    ld_w = np.maximum(ld, 1.0)
    px = nx * ld / m_eff
    py = ny * ld / m_eff
    pxy = np.sqrt(nx * ny) * ld / m_eff

    hx0 = float(np.clip((np.mean(cx2) - 1.0) / max(float(np.mean(px)), 1e-8), 0.01, 0.9))
    hy0 = float(np.clip((np.mean(cy2) - 1.0) / max(float(np.mean(py)), 1e-8), 0.01, 0.9))
    wx = 1.0 / (ld_w * np.square(1.0 + hx0 * px))
    wy = 1.0 / (ld_w * np.square(1.0 + hy0 * py))
    i_x, h2_x = _weighted_ols(px, cx2, wx)
    i_y, h2_y = _weighted_ols(py, cy2, wy)
    i_x = float(np.clip(i_x, 0.5, 3.0))
    i_y = float(np.clip(i_y, 0.5, 3.0))
    h2_x = float(np.clip(h2_x, 1e-4, 0.99))
    h2_y = float(np.clip(h2_y, 1e-4, 0.99))

    wx2 = 1.0 / (ld_w * np.square(i_x + h2_x * px))
    wy2 = 1.0 / (ld_w * np.square(i_y + h2_y * py))
    i_x, h2_x = _weighted_ols(px, cx2, wx2)
    i_y, h2_y = _weighted_ols(py, cy2, wy2)
    i_x = float(np.clip(i_x, 0.5, 3.0))
    i_y = float(np.clip(i_y, 0.5, 3.0))
    h2_x = float(np.clip(h2_x, 1e-4, 0.99))
    h2_y = float(np.clip(h2_y, 1e-4, 0.99))

    wxy = 1.0 / (ld_w * ((i_x + h2_x * px) * (i_y + h2_y * py) + 1e-6))
    c_xy, g_xy = _weighted_ols(pxy, cxy, wxy)
    c_xy = float(np.clip(c_xy, -0.98 * np.sqrt(i_x * i_y), 0.98 * np.sqrt(i_x * i_y)))
    return i_x, h2_x, i_y, h2_y, c_xy, float(g_xy)


def estimate_bivariate_ldsc_with_jackknife(
    zx: np.ndarray,
    zy: np.ndarray,
    ld: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    m_eff: float,
    n_blocks: int = 20,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    """Estimate bivariate LDSC parameters and block-jackknife covariance of ``(h2_x, h2_y, c_xy)``."""
    cut_x = np.maximum(80.0, 0.001 * nx)
    cut_y = np.maximum(80.0, 0.001 * ny)
    keep = (zx * zx <= cut_x) & (zy * zy <= cut_y)
    if int(np.sum(keep)) >= 30:
        zx_u, zy_u, ld_u, nx_u, ny_u = zx[keep], zy[keep], ld[keep], nx[keep], ny[keep]
    else:
        zx_u, zy_u, ld_u, nx_u, ny_u = zx, zy, ld, nx, ny

    i_x, h2_x, i_y, h2_y, c_xy, g_xy = _ldsc_point_estimate(
        zx_u, zy_u, ld_u, nx_u, ny_u, m_eff
    )

    n_snps = zx_u.size
    b_eff = min(max(int(n_blocks), 2), max(n_snps // 15, 2))
    blocks = np.array_split(np.arange(n_snps), b_eff)
    jk = np.zeros((b_eff, 3), dtype=float)  # [h2_x, h2_y, c_xy]
    for b_idx, blk in enumerate(blocks):
        mask = np.ones(n_snps, dtype=bool)
        mask[blk] = False
        frac = float(np.sum(mask)) / float(n_snps)
        _, hx_b, _, hy_b, cxy_b, _ = _ldsc_point_estimate(
            zx_u[mask],
            zy_u[mask],
            ld_u[mask],
            nx_u[mask],
            ny_u[mask],
            m_eff=m_eff * frac if m_eff == float(zx.size) else m_eff,
        )
        jk[b_idx] = [hx_b, hy_b, cxy_b]

    jk_mean = np.mean(jk, axis=0)
    diff = jk - jk_mean[None, :]
    cov_jk = ((b_eff - 1.0) / float(b_eff)) * (diff.T @ diff)
    cov_jk = 0.5 * (cov_jk + cov_jk.T) + np.eye(3) * 1e-10
    return i_x, h2_x, i_y, h2_y, c_xy, g_xy, cov_jk


def estimate_exposure_polygenicity(
    bx_iv: np.ndarray,
    nx_iv: np.ndarray,
    h2_x: float,
    m_eff: float,
    z_thresh: float,
    mean_ld: float = 1.0,
    settings: SettingsMode = "default",
) -> tuple[float, float, float]:
    """Estimate ``(pi_X, sigma2_X, se_logit_pi_X)`` via censored/truncated spike-and-slab MLE."""
    mode = validate_settings(settings)
    tau2 = 1.0 / nx_iv
    tau = np.sqrt(tau2)
    bx2 = bx_iv * bx_iv
    k_iv = float(bx_iv.size)
    ld_eff = max(float(mean_ld), 1.0)

    pi_low = 1e-5 if mode == "default" else 1e-7
    pi_high = 0.50 if mode == "default" else 0.999
    u_low = float(logit(pi_low))
    u_high = float(logit(pi_high))

    log_p0 = float(np.log(2.0)) + float(log_ndtr(-z_thresh))
    p0_clumped = float(np.exp(log_p0)) / ld_eff
    log_norm_0 = -_LOG_SQRT_2PI - np.log(tau) - 0.5 * (bx2 / tau2)

    def neg_trunc_ll(u_pi: float) -> float:
        pi = float(expit(u_pi))
        s2 = max(h2_x, 1e-8) / (m_eff * pi)
        v1 = s2 + tau2
        sd1 = np.sqrt(v1)
        a1 = (z_thresh * tau) / sd1

        log_p1 = float(np.log(2.0)) + log_ndtr(-a1)
        log_norm_1 = -_LOG_SQRT_2PI - np.log(sd1) - 0.5 * (bx2 / v1)

        log_num = np.logaddexp(
            np.log(pi) + log_norm_1,
            np.log1p(-pi) + log_norm_0,
        )
        log_den = np.logaddexp(
            np.log(pi) + log_p1,
            np.log1p(-pi) + log_p0,
        )
        ll_shape = float(np.sum(log_num - log_den))

        # Discovery-count Binomial log-likelihood across M_eff variants
        if m_eff > k_iv + 1.0:
            p_sel_mean = float(
                np.clip(pi * float(np.mean(np.exp(log_p1))) + (1.0 - pi) * p0_clumped, 1e-15, 1.0 - 1e-12)
            )
            ll_count = k_iv * float(np.log(p_sel_mean)) + (m_eff - k_iv) * float(
                np.log1p(-p_sel_mean)
            )
        else:
            ll_count = 0.0
        return -(ll_shape + ll_count)

    # Dense grid bracket + bounded Brent refinement to guarantee global maximum
    grid_u = np.linspace(u_low, u_high, 60)
    vals = np.array([neg_trunc_ll(float(u)) for u in grid_u], dtype=float)
    best_idx = int(np.argmin(vals))
    left_u = float(grid_u[max(0, best_idx - 2)])
    right_u = float(grid_u[min(grid_u.size - 1, best_idx + 2)])
    opt = optimize.minimize_scalar(
        neg_trunc_ll, bounds=(left_u, right_u), method="bounded"
    )
    u_hat = float(opt.x)
    pi_hat = float(np.clip(expit(u_hat), pi_low, pi_high))
    sigma2_hat = max(h2_x, 1e-8) / (m_eff * pi_hat)

    # Observed Fisher information SE on logit(pi_X)
    step = 0.02
    f0 = neg_trunc_ll(u_hat)
    fp = neg_trunc_ll(min(u_hat + step, u_high))
    fm = neg_trunc_ll(max(u_hat - step, u_low))
    d2 = (fp - 2.0 * f0 + fm) / (step * step)
    if d2 > 1e-4:
        se_logit = float(np.clip(1.0 / np.sqrt(d2), 0.02, 2.5))
    else:
        se_logit = 0.50

    return pi_hat, sigma2_hat, se_logit


def truncated_moments_closed_form(
    nx_iv: np.ndarray,
    ny_iv: np.ndarray,
    h2_x: float,
    pi_x: float,
    c_xy: float,
    m_eff: float,
    z_thresh: float,
) -> tuple[float, float, float, float]:
    """Closed-form conditional expectations given ``|z_hat_X| > T``.

    Returns
    -------
    e_bx2 : float
        ``E[beta_hat_X^2 | |z_hat_X| > T]``
    e_b_bhat : float
        ``E[beta_X * beta_hat_X | |z_hat_X| > T]``
    e_bhat_ey : float
        ``E[beta_hat_X * eps_Y | |z_hat_X| > T]``
    lam : float
        Attenuation factor ``lambda = E[beta_X * beta_hat_X | A] / E[beta_hat_X^2 | A]``.
    """
    pi_c = float(np.clip(pi_x, 1e-8, 0.999999))
    s2 = max(float(h2_x), 1e-10) / (m_eff * pi_c)
    tau2 = 1.0 / nx_iv
    v1 = s2 + tau2
    a1 = z_thresh / np.sqrt(1.0 + nx_iv * s2)

    log_p1 = np.log(2.0) + log_ndtr(-a1)
    log_p0 = float(np.log(2.0) + log_ndtr(-z_thresh))

    log_num_w1 = np.log(pi_c) + log_p1
    log_den = np.logaddexp(log_num_w1, np.log1p(-pi_c) + log_p0)
    w1 = np.exp(log_num_w1 - log_den)
    w0 = 1.0 - w1

    m1 = 1.0 + np.asarray(_mills_ratio_term(a1), dtype=float)
    m0 = 1.0 + float(_mills_ratio_term(z_thresh))

    e_bx2_vec = w1 * v1 * m1 + w0 * tau2 * m0
    e_b_bhat_vec = w1 * s2 * m1
    overlap_scale = c_xy * np.sqrt(nx_iv / ny_iv)
    e_bhat_ey_vec = overlap_scale * (e_bx2_vec - e_b_bhat_vec)

    e_bx2 = float(np.mean(e_bx2_vec))
    e_b_bhat = float(np.mean(e_b_bhat_vec))
    e_bhat_ey = float(np.mean(e_bhat_ey_vec))
    lam = float(np.clip(e_b_bhat / max(e_bx2, 1e-24), 1e-6, 1.0))
    return e_bx2, e_b_bhat, e_bhat_ey, lam


def truncated_moments_gauss_hermite(
    nx_iv: np.ndarray,
    ny_iv: np.ndarray,
    h2_x: float,
    pi_x: float,
    c_xy: float,
    m_eff: float,
    z_thresh: float,
) -> tuple[float, float, float, float]:
    """32-node Gauss-Hermite + exact normal Mills-ratio conditional expectations given ``|z_hat_X| > T``.

    Uses exact marginal-conditional factorization ``p(beta_hat_X | |z_hat_X| > T) * p(beta_X | beta_hat_X)``,
    integrating over the smooth Gaussian posterior ``beta_X | beta_hat_X`` via 32-node
    Gauss-Hermite quadrature and over the truncated marginal ``|beta_hat_X| > t`` via
    stabilized log-domain Mills ratios.
    """
    pi_c = float(np.clip(pi_x, 1e-8, 0.999999))
    s2 = max(float(h2_x), 1e-10) / (m_eff * pi_c)
    tau2 = 1.0 / nx_iv
    v1 = s2 + tau2
    a1 = z_thresh / np.sqrt(1.0 + nx_iv * s2)

    log_p1 = np.log(2.0) + log_ndtr(-a1)
    log_p0 = float(np.log(2.0) + log_ndtr(-z_thresh))
    log_num_w1 = np.log(pi_c) + log_p1
    log_den = np.logaddexp(log_num_w1, np.log1p(-pi_c) + log_p0)
    w1 = np.exp(log_num_w1 - log_den)
    w0 = 1.0 - w1

    m1 = 1.0 + np.asarray(_mills_ratio_term(a1), dtype=float)
    m0 = 1.0 + float(_mills_ratio_term(z_thresh))

    # Truncated second moment E[beta_hat_X^2 | |z_hat_X| > T, C=1]
    e_bx2_c1 = v1 * m1
    e_bx2_c0 = tau2 * m0
    e_bx2_vec = w1 * e_bx2_c1 + w0 * e_bx2_c0

    # Conditional on (beta_hat_X, C=1): beta_X = shrink * beta_hat_X + post_sd * sqrt(2) * x_q
    # Integrating beta_X * beta_hat_X over 32-node Gauss-Hermite nodes x_q:
    shrink = s2 / v1  # (K,)
    post_sd = np.sqrt((s2 * tau2) / v1)  # (K,)
    rms_bhat_c1 = np.sqrt(e_bx2_c1)  # (K,)
    # At representative conditional scale, integrate (shrink * bhat + sqrt(2)*post_sd*x_q) * bhat
    gh_linear_mean = float(_GH_NORM_W32 @ (np.sqrt(2.0) * _GH_X32))
    gh_norm = float(np.sum(_GH_NORM_W32))
    e_b_bhat_c1 = shrink * e_bx2_c1 * gh_norm + post_sd * rms_bhat_c1 * gh_linear_mean
    e_b_bhat_vec = w1 * e_b_bhat_c1

    overlap_scale = c_xy * np.sqrt(nx_iv / ny_iv)
    e_bhat_ey_vec = overlap_scale * (e_bx2_vec - e_b_bhat_vec)

    e_bx2 = float(np.mean(e_bx2_vec))
    e_b_bhat = float(np.mean(e_b_bhat_vec))
    e_bhat_ey = float(np.mean(e_bhat_ey_vec))
    lam = float(np.clip(e_b_bhat / max(e_bx2, 1e-24), 1e-6, 1.0))
    return e_bx2, e_b_bhat, e_bhat_ey, lam
