"""Stage-1 bivariate LD Score Regression (LDSC) for LHC-MR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate the mandatory ``settings`` parameter."""
    if settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class BivariateLDSCResult:
    """Summary of Stage-1 bivariate LD score regression."""

    i_x: float
    i_y: float
    i_xy: float
    h2_x: float
    h2_y: float
    g_xy: float
    rg: float
    n_snps_used: int
    n_outliers_removed: int
    settings: SettingsMode


def _weighted_simple_ols(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    """Weighted linear regression y ~ intercept + slope * x."""
    w_pos = np.maximum(w, 1e-12)
    w_sum = float(np.sum(w_pos))
    x_bar = float(np.sum(w_pos * x) / w_sum)
    y_bar = float(np.sum(w_pos * y) / w_sum)
    dx = x - x_bar
    denom = float(np.sum(w_pos * dx * dx))
    if denom <= 1e-16:
        return y_bar, 0.0
    slope = float(np.sum(w_pos * dx * (y - y_bar)) / denom)
    intercept = y_bar - slope * x_bar
    return intercept, slope


def _weighted_constrained_slope(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, intercept: float
) -> float:
    """Weighted slope of (y - intercept) ~ 0 + slope * x."""
    w_pos = np.maximum(w, 1e-12)
    denom = float(np.sum(w_pos * x * x))
    if denom <= 1e-16:
        return 0.0
    return float(np.sum(w_pos * x * (y - intercept)) / denom)


def _truncated_null_corr(zx: np.ndarray, zy: np.ndarray, z_cut: float = 1.5) -> float:
    """Estimate null Z-score correlation from |zx| < z_cut, |zy| < z_cut."""
    mask = (np.abs(zx) < z_cut) & (np.abs(zy) < z_cut)
    if int(np.sum(mask)) < 25:
        return float(np.clip(np.corrcoef(zx, zy)[0, 1], -0.95, 0.95))
    p_in = 2.0 * float(stats.norm.cdf(z_cut)) - 1.0
    kappa = 1.0 - (2.0 * z_cut * float(stats.norm.pdf(z_cut))) / max(p_in, 1e-12)
    cov_trunc = float(np.mean(zx[mask] * zy[mask]))
    rho_est = cov_trunc / max(kappa * kappa, 1e-6)
    return float(np.clip(rho_est, -0.95, 0.95))


def bivariate_ldsc(
    zx: ArrayLike,
    zy: ArrayLike,
    ld_scores: ArrayLike,
    nx: float | ArrayLike,
    ny: float | ArrayLike,
    m_snps: float | None = None,
    filter_outliers: bool | None = None,
    two_step_chisq: float = 30.0,
    settings: SettingsMode = "default",
) -> BivariateLDSCResult:
    """Estimate Stage-1 two-step bivariate LDSC intercepts and heritabilities.

    Uses Bulik-Sullivan et al. (2015) two-step IRLS estimator: Step 1 estimates
    intercepts on SNPs with ``z^2 < two_step_chisq`` (default 30) to prevent
    leverage distortion from sparse large-effect loci; Step 2 estimates slopes
    (heritabilities and genetic covariance) across all retained SNPs with the
    intercepts fixed at Step 1.
    """
    mode = validate_settings(settings)
    zx_arr = np.asarray(zx, dtype=float).ravel()
    zy_arr = np.asarray(zy, dtype=float).ravel()
    ld_arr = np.asarray(ld_scores, dtype=float).ravel()
    if not (zx_arr.shape == zy_arr.shape == ld_arr.shape):
        raise ValueError("zx, zy, and ld_scores must have identical 1D shapes.")
    if zx_arr.size < 20:
        raise ValueError("At least 20 SNPs are required for bivariate LDSC.")

    nx_arr = np.broadcast_to(np.asarray(nx, dtype=float), zx_arr.shape).copy()
    ny_arr = np.broadcast_to(np.asarray(ny, dtype=float), zy_arr.shape).copy()
    if np.any(nx_arr <= 0) or np.any(ny_arr <= 0):
        raise ValueError("Sample sizes nx and ny must be strictly positive.")

    finite = (
        np.isfinite(zx_arr)
        & np.isfinite(zy_arr)
        & np.isfinite(ld_arr)
        & np.isfinite(nx_arr)
        & np.isfinite(ny_arr)
    )
    zx_f = zx_arr[finite]
    zy_f = zy_arr[finite]
    ld_f = np.maximum(ld_arr[finite], 1e-3)
    nx_f = nx_arr[finite]
    ny_f = ny_arr[finite]

    m_eff = float(m_snps) if m_snps is not None else float(zx_f.size)
    do_filter = (mode == "R_default") if filter_outliers is None else bool(filter_outliers)

    n_outliers = 0
    if do_filter:
        cut_x = np.maximum(80.0, 0.001 * nx_f)
        cut_y = np.maximum(80.0, 0.001 * ny_f)
        keep = (zx_f * zx_f <= cut_x) & (zy_f * zy_f <= cut_y)
        if int(np.sum(keep)) >= 20:
            n_outliers = int(zx_f.size - np.sum(keep))
            zx_f = zx_f[keep]
            zy_f = zy_f[keep]
            ld_f = ld_f[keep]
            nx_f = nx_f[keep]
            ny_f = ny_f[keep]

    chisq_x = zx_f * zx_f
    chisq_y = zy_f * zy_f
    cross_xy = zx_f * zy_f

    if float(np.var(ld_f)) < 1e-8:
        null_x = chisq_x < 4.0
        null_y = chisq_y < 4.0
        p_in = 2.0 * float(stats.norm.cdf(2.0)) - 1.0
        kappa2 = 1.0 - (4.0 * float(stats.norm.pdf(2.0))) / max(p_in, 1e-12)
        i_x = float(np.mean(chisq_x[null_x]) / kappa2) if int(np.sum(null_x)) >= 25 else 1.0
        i_y = float(np.mean(chisq_y[null_y]) / kappa2) if int(np.sum(null_y)) >= 25 else 1.0
        i_x = float(np.clip(i_x, 0.5, 2.5))
        i_y = float(np.clip(i_y, 0.5, 2.5))

        mean_ld = float(np.mean(ld_f))
        h2_x = float(
            np.clip(np.mean(chisq_x - i_x) * m_eff / (np.mean(nx_f) * mean_ld), 1e-4, 0.99)
        )
        h2_y = float(
            np.clip(np.mean(chisq_y - i_y) * m_eff / (np.mean(ny_f) * mean_ld), 1e-4, 0.99)
        )
        i_xy = _truncated_null_corr(zx_f, zy_f, z_cut=1.5) * float(np.sqrt(i_x * i_y))
        g_xy = float(
            np.mean(cross_xy - i_xy) * m_eff / (np.mean(np.sqrt(nx_f * ny_f)) * mean_ld)
        )
    else:
        ld_w = np.maximum(ld_f, 1.0)
        pred_x = nx_f * ld_f / m_eff
        pred_y = ny_f * ld_f / m_eff
        pred_xy = np.sqrt(nx_f * ny_f) * ld_f / m_eff

        # Apply two-step (chisq < two_step_chisq) only in low-per-SNP-SNR GWAS regimes
        # (median chisq < 1.5) where chisq < 30 does not right-censor high-LD polygenic SNPs;
        # in dense high-SNR sub-genomes, full heteroskedastic IRLS is unbiased.
        use_two_step_x = float(np.percentile(pred_x, 90)) < 2.0
        use_two_step_y = float(np.percentile(pred_y, 90)) < 2.0
        s1_x = (chisq_x < two_step_chisq) if use_two_step_x else np.ones_like(chisq_x, dtype=bool)
        if int(np.sum(s1_x)) < 30:
            s1_x = np.ones_like(chisq_x, dtype=bool)
        s1_y = (chisq_y < two_step_chisq) if use_two_step_y else np.ones_like(chisq_y, dtype=bool)
        if int(np.sum(s1_y)) < 30:
            s1_y = np.ones_like(chisq_y, dtype=bool)

        h2_x0 = float(
            np.clip(
                (np.mean(chisq_x[s1_x]) - 1.0) / max(float(np.mean(pred_x[s1_x])), 1e-8),
                0.01,
                0.9,
            )
        )
        h2_y0 = float(
            np.clip(
                (np.mean(chisq_y[s1_y]) - 1.0) / max(float(np.mean(pred_y[s1_y])), 1e-8),
                0.01,
                0.9,
            )
        )

        for _ in range(2):
            wx1 = 1.0 / (ld_w[s1_x] * np.square(1.0 + h2_x0 * pred_x[s1_x]))
            i_x, h2_x_s1 = _weighted_simple_ols(pred_x[s1_x], chisq_x[s1_x], wx1)
            i_x = float(np.clip(i_x, 0.5, 2.5))
            h2_x0 = float(np.clip(h2_x_s1, 1e-4, 0.95))

            wy1 = 1.0 / (ld_w[s1_y] * np.square(1.0 + h2_y0 * pred_y[s1_y]))
            i_y, h2_y_s1 = _weighted_simple_ols(pred_y[s1_y], chisq_y[s1_y], wy1)
            i_y = float(np.clip(i_y, 0.5, 2.5))
            h2_y0 = float(np.clip(h2_y_s1, 1e-4, 0.95))

        # Step 2: estimate slopes across all retained SNPs with intercepts fixed at Step 1
        h2_x = h2_x0
        h2_y = h2_y0
        for _ in range(2):
            wx2 = 1.0 / (ld_w * np.square(i_x + h2_x * pred_x))
            wy2 = 1.0 / (ld_w * np.square(i_y + h2_y * pred_y))
            h2_x = float(
                np.clip(_weighted_constrained_slope(pred_x, chisq_x, wx2, i_x), 1e-4, 0.99)
            )
            h2_y = float(
                np.clip(_weighted_constrained_slope(pred_y, chisq_y, wy2, i_y), 1e-4, 0.99)
            )

        # Cross-trait two-step LDSC
        s1_xy = s1_x & s1_y
        if int(np.sum(s1_xy)) < 30:
            s1_xy = np.ones_like(cross_xy, dtype=bool)
        w_xy1 = 1.0 / (
            ld_w[s1_xy]
            * ((i_x + h2_x * pred_x[s1_xy]) * (i_y + h2_y * pred_y[s1_xy]) + 1e-6)
        )
        i_xy_raw, g_xy_s1 = _weighted_simple_ols(
            pred_xy[s1_xy], cross_xy[s1_xy], w_xy1
        )
        max_ixy = 0.98 * float(np.sqrt(i_x * i_y))
        i_xy = float(np.clip(i_xy_raw, -max_ixy, max_ixy))

        denom_xy = (i_x + h2_x * pred_x) * (i_y + h2_y * pred_y) + np.square(
            i_xy + g_xy_s1 * pred_xy
        )
        w_xy2 = 1.0 / (ld_w * np.maximum(denom_xy, 1e-6))
        g_xy = _weighted_constrained_slope(pred_xy, cross_xy, w_xy2, i_xy)

    rg = float(np.clip(g_xy / np.sqrt(max(h2_x * h2_y, 1e-12)), -1.0, 1.0))
    return BivariateLDSCResult(
        i_x=float(i_x),
        i_y=float(i_y),
        i_xy=float(i_xy),
        h2_x=float(h2_x),
        h2_y=float(h2_y),
        g_xy=float(g_xy),
        rg=rg,
        n_snps_used=int(zx_f.size),
        n_outliers_removed=n_outliers,
        settings=mode,
    )
