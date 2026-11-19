"""BSmooth local tricube weighted moving-average methylation smoothing."""

from __future__ import annotations

from typing import Literal

import numpy as np

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate ``settings`` parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def _tricube_window_neighbors(
    pos_sorted: np.ndarray, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded neighbor index matrix and tricube kernel weights on sorted positions."""
    n = len(pos_sorted)
    if n == 0:
        return np.zeros((0, 1), dtype=np.int64), np.zeros((0, 1), dtype=np.float64)

    half_span = max(0.5 * float(span), 0.5)
    left = np.searchsorted(pos_sorted, pos_sorted - half_span, side="left")
    right = np.searchsorted(pos_sorted, pos_sorted + half_span, side="right")
    widths = np.maximum(right - left, 1)
    max_w = int(np.max(widths))

    offsets = np.arange(max_w, dtype=np.int64)[None, :]
    raw_idx = left[:, None] + offsets
    valid = raw_idx < right[:, None]
    j_idx = np.minimum(raw_idx, n - 1)

    dist = np.abs(pos_sorted[j_idx] - pos_sorted[:, None])
    u = dist / half_span
    k_mat = np.where(valid & (u <= 1.0), (1.0 - np.minimum(u, 1.0) ** 3) ** 3, 0.0)
    # Guarantee self-inclusion if u == 1 on zero-span edge cases
    row_sums = np.sum(k_mat, axis=1)
    zero_rows = row_sums <= 0.0
    if np.any(zero_rows):
        j_idx[zero_rows, 0] = np.flatnonzero(zero_rows)
        k_mat[zero_rows, 0] = 1.0
    return j_idx, k_mat


def smooth_counts_tricube(
    chrom: np.ndarray,
    pos: np.ndarray,
    x_mat: np.ndarray,
    n_mat: np.ndarray,
    smoothing_span: float = 500.0,
) -> np.ndarray:
    """Compute BSmooth tricube local-coverage-weighted methylation proportions.

    Parameters
    ----------
    chrom
        Chromosome labels of shape ``(n_cpgs,)``.
    pos
        Genomic coordinates of shape ``(n_cpgs,)``.
    x_mat
        Methylated counts of shape ``(n_cpgs, n_samples)``.
    n_mat
        Total coverage counts of shape ``(n_cpgs, n_samples)``.
    smoothing_span
        Total window width in base pairs (default ``500.0`` bp, i.e. ``+/- 250`` bp).

    Returns
    -------
    p_smooth
        Smoothed methylation proportions of shape ``(n_cpgs, n_samples)``.
    """
    n_cpgs, n_samples = x_mat.shape
    out = np.zeros((n_cpgs, n_samples), dtype=np.float64)
    if n_cpgs == 0:
        return out

    chrom_str = np.asarray(chrom, dtype=str)
    pos_f = np.asarray(pos, dtype=np.float64)

    for ch in np.unique(chrom_str):
        mask = chrom_str == ch
        idx_ch = np.flatnonzero(mask)
        order = np.argsort(pos_f[idx_ch], kind="mergesort")
        sorted_idx = idx_ch[order]

        pos_s = pos_f[sorted_idx]
        x_s = x_mat[sorted_idx]
        n_s = n_mat[sorted_idx]

        j_idx, k_mat = _tricube_window_neighbors(pos_s, span=smoothing_span)
        # Aggregate numerator sum_m K[i,m] * X[j,r] and denominator sum_m K[i,m] * N[j,r]
        num = np.einsum("im,imr->ir", k_mat, x_s[j_idx])
        den = np.einsum("im,imr->ir", k_mat, n_s[j_idx])

        # Fallback to unweighted sample mean on chromosome where local window has 0 coverage
        has_cov = den > 0.0
        safe_den = np.where(has_cov, den, 1.0)
        p_ch = np.where(has_cov, num / safe_den, np.nan)
        for r in range(n_samples):
            nan_r = ~np.isfinite(p_ch[:, r])
            if np.any(nan_r):
                tot_n = float(np.sum(n_s[:, r]))
                fallback = float(np.sum(x_s[:, r])) / tot_n if tot_n > 0.0 else 0.5
                p_ch[nan_r, r] = fallback
        out[sorted_idx] = np.clip(p_ch, 0.0, 1.0)

    return out


def compute_group_mean_and_var(
    chrom: np.ndarray,
    pos: np.ndarray,
    x_group: np.ndarray,
    n_group: np.ndarray,
    phi: np.ndarray,
    smoothing: bool,
    smoothing_span: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Beta-Binomial precision-weighted group mean and Wald variance.

    Implements Feng et al. (*NAR* 2014) exact variance equations for both
    unsmoothed (`smoothing=False`) and tricube-smoothed (`smoothing=True`) CpG
    methylation estimates.
    """
    n_cpgs = len(pos)
    if n_cpgs == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    phi_col = np.clip(np.asarray(phi, dtype=np.float64), 1e-5, 0.99)[:, None]
    pos_cov = n_group > 0.0
    denom_w = 1.0 + np.maximum(n_group - 1.0, 0.0) * phi_col
    w_ir = np.where(pos_cov, n_group / denom_w, 0.0)
    n_safe = np.where(pos_cov, n_group, 1.0)
    p_ir = np.where(pos_cov, x_group / n_safe, 0.0)

    w_sum = np.sum(w_ir, axis=1)
    wp_sum = np.sum(w_ir * p_ir, axis=1)

    if not smoothing:
        has_w = w_sum > 0.0
        w_safe = np.where(has_w, w_sum, 1.0)
        mu = np.where(has_w, wp_sum / w_safe, 0.5)
        mu_var = np.clip(mu, 1e-3, 1.0 - 1e-3)
        var = np.where(has_w, mu_var * (1.0 - mu_var) / w_safe, np.nan)
        return np.clip(mu, 0.0, 1.0), var

    mu_out = np.zeros(n_cpgs, dtype=np.float64)
    var_out = np.full(n_cpgs, np.nan, dtype=np.float64)
    chrom_str = np.asarray(chrom, dtype=str)
    pos_f = np.asarray(pos, dtype=np.float64)

    for ch in np.unique(chrom_str):
        idx_ch = np.flatnonzero(chrom_str == ch)
        order = np.argsort(pos_f[idx_ch], kind="mergesort")
        s_idx = idx_ch[order]

        j_idx, k_mat = _tricube_window_neighbors(pos_f[s_idx], span=smoothing_span)
        wp_s = wp_sum[s_idx]
        w_s = w_sum[s_idx]

        num_mu = np.sum(k_mat * wp_s[j_idx], axis=1)
        den_mu = np.sum(k_mat * w_s[j_idx], axis=1)
        ok_den = den_mu > 0.0
        den_safe = np.where(ok_den, den_mu, 1.0)
        mu_s = np.clip(np.where(ok_den, num_mu / den_safe, 0.5), 0.0, 1.0)

        mu_clamped = np.clip(mu_s, 1e-3, 1.0 - 1e-3)
        v_unit = mu_clamped * (1.0 - mu_clamped) * w_s
        num_var = np.sum((k_mat * k_mat) * v_unit[j_idx], axis=1)
        var_s = np.where(ok_den, num_var / (den_safe * den_safe), np.nan)

        mu_out[s_idx] = mu_s
        var_out[s_idx] = var_s

    return mu_out, var_out
