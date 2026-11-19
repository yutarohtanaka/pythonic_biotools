"""Composition-bias TMM normalization and non-linear efficiency loess offsets."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .utils import SettingsMode, ave_log_cpm, loess_fit_1d, validate_settings
from .windows import WindowData


def _tmm_pair(
    obs: np.ndarray,
    ref: np.ndarray,
    lib_obs: float,
    lib_ref: float,
    logratio_trim: float,
    sum_trim: float,
    acutoff: float,
    do_weighting: bool,
) -> float:
    """Compute Robinson & Oshlack (2010) TMM factor for one library vs reference."""
    if np.array_equal(obs, ref) and abs(lib_obs - lib_ref) <= 1e-12:
        return 1.0

    pos = (obs > 0) & (ref > 0)
    y_k = obs[pos]
    y_r = ref[pos]
    if y_k.size == 0:
        return 1.0

    p_k = y_k / lib_obs
    p_r = y_r / lib_ref
    log_r = np.log2(p_k / p_r)
    abs_e = 0.5 * (np.log2(p_k) + np.log2(p_r))
    v = (lib_obs - y_k) / (lib_obs * y_k) + (lib_ref - y_r) / (lib_ref * y_r)

    fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > acutoff) & (v > 0)
    log_r = log_r[fin]
    abs_e = abs_e[fin]
    v = v[fin]

    n = int(log_r.size)
    if n == 0:
        return 1.0

    lo_l = np.floor(n * logratio_trim) + 1.0
    hi_l = n + 1.0 - lo_l
    lo_s = np.floor(n * sum_trim) + 1.0
    hi_s = n + 1.0 - lo_s

    ranks_l = rankdata(log_r, method="average")
    ranks_s = rankdata(abs_e, method="average")
    keep = (ranks_l >= lo_l) & (ranks_l <= hi_l) & (ranks_s >= lo_s) & (ranks_s <= hi_s)
    if not np.any(keep):
        return float(2.0 ** np.median(log_r))

    if do_weighting:
        weights = 1.0 / v[keep]
        f_log2 = float(np.sum(log_r[keep] * weights) / np.sum(weights))
    else:
        f_log2 = float(np.mean(log_r[keep]))
    return float(2.0 ** f_log2) if np.isfinite(f_log2) else 1.0


def norm_factors(
    counts: WindowData | np.ndarray,
    totals: np.ndarray | None = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    acutoff: float = -1e10,
    do_weighting: bool = True,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Compute TMM library scaling factors on background bins (csaw::normFactors).

    Implements Robinson & Oshlack (*Genome Biology* 2010) / ``edgeR::calcNormFactors``
    recommended by Lun & Smyth (*NAR* 2016) on 10 kb genomic bins to eliminate ChIP
    composition bias without treating true differential binding as bias.

    Parameters
    ----------
    counts : WindowData | np.ndarray
        Count matrix of shape ``(n_bins, n_samples)`` or ``WindowData``.
    totals : np.ndarray | None
        Total library sizes of shape ``(n_samples,)``. Extracted from ``counts.totals`` if
        ``WindowData`` is given, else defaults to column sums.
    logratio_trim : float
        Fraction trimmed from each tail of M-values (log2 fold ratios), default ``0.3``.
    sum_trim : float
        Fraction trimmed from each tail of A-values (average log2 expression), default ``0.05``.
    acutoff : float
        Minimum A-value threshold.
    do_weighting : bool
        Use inverse delta-method binomial variance weights.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    np.ndarray
        Geometric-mean-1 TMM normalization factors of shape ``(n_samples,)``.
    """
    validate_settings(settings)
    if not (0.0 <= logratio_trim < 0.5):
        raise ValueError(f"logratio_trim must lie in [0, 0.5), got {logratio_trim}.")
    if not (0.0 <= sum_trim < 0.5):
        raise ValueError(f"sum_trim must lie in [0, 0.5), got {sum_trim}.")

    if isinstance(counts, WindowData):
        y = counts.counts.astype(float)
        libs = counts.totals.astype(float) if totals is None else np.asarray(totals, dtype=float)
    else:
        y = np.asarray(counts, dtype=float)
        libs = np.sum(y, axis=0) if totals is None else np.asarray(totals, dtype=float)

    if y.ndim != 2:
        raise ValueError("counts must be a 2D matrix of shape (n_features, n_samples).")
    n_features, n_samples = y.shape
    if n_samples == 0:
        return np.empty(0, dtype=float)
    if n_samples == 1 or n_features == 0:
        return np.ones(n_samples, dtype=float)

    libs = np.maximum(libs.ravel(), 1.0)
    nonzero_rows = np.sum(y, axis=1) > 0
    y = y[nonzero_rows, :]
    if y.shape[0] == 0:
        return np.ones(n_samples, dtype=float)

    # Select reference column via 75th percentile closest to mean 75th percentile (edgeR exact)
    q75 = np.quantile(y / libs[None, :], 0.75, axis=0)
    if np.all(q75 <= 0):
        ref_idx = 0
    else:
        ref_idx = int(np.argmin(np.abs(q75 - np.mean(q75))))

    factors = np.ones(n_samples, dtype=float)
    for j in range(n_samples):
        if j == ref_idx:
            factors[j] = 1.0
        else:
            factors[j] = _tmm_pair(
                obs=y[:, j],
                ref=y[:, ref_idx],
                lib_obs=float(libs[j]),
                lib_ref=float(libs[ref_idx]),
                logratio_trim=logratio_trim,
                sum_trim=sum_trim,
                acutoff=acutoff,
                do_weighting=do_weighting,
            )

    # Geometric mean normalization to 1.0
    factors /= np.exp(np.mean(np.log(np.maximum(factors, 1e-12))))
    return factors


def norm_offsets(
    counts: WindowData | np.ndarray,
    totals: np.ndarray | None = None,
    span: float = 0.3,
    prior_count: float = 2.0,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Compute trended loess offset matrix for non-linear efficiency bias (csaw::normOffsets).

    Fits sample-specific local linear loess curves of log-counts against average window
    abundance ``aveLogCPM`` across filtered peaks/windows, removing row means and scaling
    to mean log library size on the natural-log scale expected by NB GLMs.

    Parameters
    ----------
    counts : WindowData | np.ndarray
        Filtered window count matrix of shape ``(n_windows, n_samples)`` or ``WindowData``.
    totals : np.ndarray | None
        Total library sizes of shape ``(n_samples,)``.
    span : float
        Loess smoothing span in ``(0, 1]``, default ``0.3``.
    prior_count : float
        Prior count scaled by library size to stabilize low counts.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    np.ndarray
        Natural-log GLM offset matrix of shape ``(n_windows, n_samples)``.
    """
    validate_settings(settings)
    if not (0.0 < span <= 1.0):
        raise ValueError(f"span must lie in (0, 1], got {span}.")

    if isinstance(counts, WindowData):
        y = counts.counts.astype(float)
        libs = counts.totals.astype(float) if totals is None else np.asarray(totals, dtype=float)
    else:
        y = np.asarray(counts, dtype=float)
        libs = np.sum(y, axis=0) if totals is None else np.asarray(totals, dtype=float)

    if y.ndim != 2:
        raise ValueError("counts must be a 2D matrix of shape (n_windows, n_samples).")
    n_windows, n_samples = y.shape
    libs = np.maximum(libs.ravel(), 1.0)
    mean_log_lib = float(np.mean(np.log(libs)))

    if n_windows == 0 or n_samples == 0:
        return np.zeros((n_windows, n_samples), dtype=float)

    ab = ave_log_cpm(y, lib_sizes=libs, prior_count=prior_count)
    mean_lib = float(np.mean(libs))
    prior = prior_count * (libs / mean_lib)
    log_y = np.log(np.maximum(y + prior[None, :], 1e-12))
    ref_log_y = np.mean(log_y, axis=1)

    offsets = np.empty((n_windows, n_samples), dtype=float)
    for j in range(n_samples):
        diff_j = log_y[:, j] - ref_log_y
        offsets[:, j] = loess_fit_1d(ab, diff_j, span=span)

    offsets -= np.mean(offsets, axis=1, keepdims=True)
    offsets += mean_log_lib
    return offsets
