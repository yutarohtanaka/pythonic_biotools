"""Shared utilities, settings validation, and abundance helpers for csaw."""

from __future__ import annotations

from typing import Literal

import numpy as np

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate the mandatory ``settings`` mode parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings '{settings}'. Must be one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR adjusted p-values, preserving NaNs."""
    p = np.asarray(pvals, dtype=float)
    fdr = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    m = int(np.sum(valid))
    if m == 0:
        return fdr
    pv = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)
    adj = ranked * (m / ranks)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out_valid = np.empty(m, dtype=float)
    out_valid[order] = adj
    fdr[valid] = out_valid
    return fdr


def ave_log_cpm(
    counts: np.ndarray,
    lib_sizes: np.ndarray | None = None,
    prior_count: float = 2.0,
) -> np.ndarray:
    """Compute average log2 counts-per-million across libraries (edgeR::aveLogCPM style).

    Parameters
    ----------
    counts : np.ndarray
        Count matrix of shape ``(n_features, n_samples)``.
    lib_sizes : np.ndarray | None
        Library sizes of shape ``(n_samples,)``. Defaults to column sums.
    prior_count : float
        Prior count added proportionally to library size to damp low-count variance.

    Returns
    -------
    np.ndarray
        1D array of ``log2`` average CPM values of shape ``(n_features,)``.
    """
    y = np.asarray(counts, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n_features, n_samples = y.shape
    if n_features == 0:
        return np.empty(0, dtype=float)
    if lib_sizes is None:
        libs = np.sum(y, axis=0)
    else:
        libs = np.asarray(lib_sizes, dtype=float).ravel()
        if libs.shape[0] != n_samples:
            raise ValueError(
                f"lib_sizes length ({libs.shape[0]}) does not match n_samples ({n_samples})."
            )
    libs = np.maximum(libs, 1.0)
    mean_lib = float(np.mean(libs))
    prior = prior_count * (libs / mean_lib)
    aug_libs = libs + 2.0 * prior
    cpm = (y + prior[None, :]) / aug_libs[None, :] * 1e6
    return np.log2(np.maximum(np.mean(cpm, axis=1), 1e-12))


def scaled_average(
    counts: np.ndarray,
    lib_sizes: np.ndarray | None = None,
    scale: float = 1.0,
    prior_count: float = 2.0,
) -> np.ndarray:
    """Compute width-scaled average log2 CPM (csaw::scaledAverage).

    Matches ``aveLogCPM(y, lib.size=lib_sizes, prior.count=prior_count * scale) - log2(scale)``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    return ave_log_cpm(counts, lib_sizes=lib_sizes, prior_count=prior_count * scale) - np.log2(scale)


def loess_fit_1d(
    x: np.ndarray,
    y: np.ndarray,
    span: float = 0.3,
    n_anchors: int = 200,
) -> np.ndarray:
    """Fast 1D Cleveland tricube local linear regression (limma::loessFit equivalent).

    Evaluates local weighted linear fits on up to ``n_anchors`` quantile-spaced anchor
    points and linearly interpolates back to all observations, handling exact ties cleanly.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = xv.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2 or np.ptp(xv) <= 1e-12:
        return np.full(n, float(np.mean(yv)), dtype=float)

    span = float(np.clip(span, 0.05, 1.0))
    k = max(3, min(n, int(np.ceil(span * n))))

    order = np.argsort(xv, kind="mergesort")
    xs = xv[order]
    ys = yv[order]

    # Choose anchor x-values covering [xs[0], xs[-1]]
    unique_x = np.unique(xs)
    if unique_x.shape[0] <= n_anchors:
        anchors = unique_x
    else:
        q_idx = np.linspace(0, n - 1, n_anchors, dtype=int)
        anchors = np.unique(np.concatenate([[xs[0]], xs[q_idx], [xs[-1]]]))

    fitted_anchors = np.empty(anchors.shape[0], dtype=float)
    for idx, x0 in enumerate(anchors):
        # Find k nearest neighbors in sorted xs using binary search + two-pointer window
        pos = int(np.searchsorted(xs, x0))
        left = max(0, pos - k)
        right = min(n, pos + k)
        sub_x = xs[left:right]
        sub_y = ys[left:right]
        dists = np.abs(sub_x - x0)
        if dists.shape[0] > k:
            part = np.argpartition(dists, k - 1)[:k]
            sub_x = sub_x[part]
            sub_y = sub_y[part]
            dists = dists[part]
        max_d = float(np.max(dists))
        if max_d <= 1e-12:
            # Expand to include all exact ties at x0
            tie_lo = int(np.searchsorted(xs, x0, side="left"))
            tie_hi = int(np.searchsorted(xs, x0, side="right"))
            fitted_anchors[idx] = float(np.mean(ys[tie_lo:tie_hi]))
            continue
        u = dists / (max_d * 1.0001)
        w = (1.0 - u ** 3) ** 3
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            fitted_anchors[idx] = float(np.mean(sub_y))
            continue
        dx = sub_x - x0
        mx = float(np.sum(w * dx) / w_sum)
        my = float(np.sum(w * sub_y) / w_sum)
        sxx = float(np.sum(w * (dx - mx) ** 2))
        if sxx <= 1e-14:
            fitted_anchors[idx] = my
        else:
            slope = float(np.sum(w * (dx - mx) * (sub_y - my)) / sxx)
            fitted_anchors[idx] = my - slope * mx

    fitted_sorted = np.interp(xs, anchors, fitted_anchors)
    out = np.empty(n, dtype=float)
    out[order] = fitted_sorted
    return out
