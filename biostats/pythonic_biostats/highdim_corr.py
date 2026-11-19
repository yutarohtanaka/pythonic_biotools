"""
Vectorized high-dimensional correlation scans, memory-bounded pooled-null
permutation empirical FDR, bootstrap CIs, partial correlations, and NaN-safe
statistical test wrappers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as _st


def _require_finite(*arrays: Sequence[float] | np.ndarray) -> list[np.ndarray]:
    out = [np.asarray(a, dtype=np.float64) for a in arrays]
    for a in out:
        if not np.all(np.isfinite(a)):
            raise ValueError("Non-finite values (NaN or Inf) passed to statistical test.")
    return out


def _centered_matrix_corr(Xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Correlation of each column of `Xc` against `yc`, both already mean-centered."""
    num = Xc.T @ yc
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / denom


def spearman_matrix(X: np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray:
    """Vectorized Spearman rank correlation of `y` against every column of `X` (`n_samples x n_features`)."""
    mat = np.asarray(X, dtype=np.float64)
    vec = np.asarray(y, dtype=np.float64).ravel()
    if mat.shape[0] != vec.size:
        raise ValueError(f"X has {mat.shape[0]} rows, but y has length {vec.size}.")
    Xr = _st.rankdata(mat, axis=0)
    yr = _st.rankdata(vec)
    return _centered_matrix_corr(Xr - Xr.mean(axis=0), yr - yr.mean())


def pearson_matrix(X: np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation of `y` against every column of `X` (`n_samples x n_features`)."""
    mat = np.asarray(X, dtype=np.float64)
    vec = np.asarray(y, dtype=np.float64).ravel()
    if mat.shape[0] != vec.size:
        raise ValueError(f"X has {mat.shape[0]} rows, but y has length {vec.size}.")
    return _centered_matrix_corr(mat - mat.mean(axis=0), vec - vec.mean())


def bh_fdr(pvals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values via `scipy.stats.false_discovery_control`."""
    p = np.asarray(pvals, dtype=np.float64).ravel()
    if p.size == 0:
        return np.array([], dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise ValueError("Non-finite p-values passed to bh_fdr.")
    return _st.false_discovery_control(np.clip(p, 0.0, 1.0), method="bh")


def permutation_empirical_fdr(
    X: np.ndarray,
    y: Sequence[float] | np.ndarray,
    observed_rho: np.ndarray | None = None,
    method: str = "spearman",
    n_perm: int = 10_000,
    n_bins: int = 4000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled-null empirical p-values and BH-FDR for genome-wide correlation scans.

    Permutes `y` across samples `n_perm` times, streaming `|rho|` into a fixed-width
    histogram so memory stays O(`n_bins`) even for 50,000 features x 10,000 permutations.

    Returns `(observed_rho, empirical_p, empirical_fdr)`.
    """
    mat = np.asarray(X, dtype=np.float64)
    vec = np.asarray(y, dtype=np.float64).ravel()
    if method == "spearman":
        Xr = _st.rankdata(mat, axis=0)
        yr = _st.rankdata(vec)
    elif method == "pearson":
        Xr = mat
        yr = vec
    else:
        raise ValueError("method must be 'spearman' or 'pearson'.")

    Xc = Xr - Xr.mean(axis=0)
    yc = yr - yr.mean()
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    if observed_rho is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            obs_rho = (Xc.T @ yc) / denom
    else:
        obs_rho = np.asarray(observed_rho, dtype=np.float64).ravel()

    rng = np.random.default_rng(random_state)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    pooled_counts = np.zeros(n_bins, dtype=np.int64)

    for _ in range(n_perm):
        yc_perm = rng.permutation(yc)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_rho = np.abs((Xc.T @ yc_perm) / denom)
        null_rho = null_rho[np.isfinite(null_rho)]
        counts, _ = np.histogram(np.clip(null_rho, 0.0, 1.0), bins=bin_edges)
        pooled_counts += counts

    total_null = int(pooled_counts.sum())
    cum_from_right = np.cumsum(pooled_counts[::-1])[::-1]

    obs_abs = np.where(np.isfinite(obs_rho), np.abs(obs_rho), 0.0)
    bin_idx = np.clip(np.searchsorted(bin_edges, obs_abs, side="left") - 1, 0, n_bins - 1)
    exceed = np.where(bin_idx + 1 < n_bins, cum_from_right[np.minimum(bin_idx + 1, n_bins - 1)], 0)
    empirical_p = (exceed + 1.0) / (total_null + 1.0)
    fdr = bh_fdr(empirical_p)
    return obs_rho, empirical_p, fdr


def bootstrap_spearman_ci(
    X: np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized non-parametric bootstrap `(1 - alpha)` confidence intervals for Spearman rho."""
    mat = np.asarray(X, dtype=np.float64)
    vec = np.asarray(y, dtype=np.float64).ravel()
    rng = np.random.default_rng(random_state)
    n, n_features = mat.shape
    boot_rho = np.empty((n_boot, n_features), dtype=np.float32)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rho[b] = spearman_matrix(mat[idx], vec[idx])

    lo = np.nanpercentile(boot_rho, 100.0 * (alpha / 2.0), axis=0)
    hi = np.nanpercentile(boot_rho, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def partial_corr_pval(
    x: Sequence[float],
    y: Sequence[float],
    covariates: np.ndarray,
) -> tuple[float, float]:
    """Partial Pearson correlation of `x` and `y` controlling for `covariates` (`df = n - 2 - k`)."""
    xv, yv = _require_finite(x, y)
    cov = np.asarray(covariates, dtype=np.float64).reshape(xv.size, -1)
    if not np.all(np.isfinite(cov)):
        raise ValueError("Non-finite values in covariates.")
    C = np.column_stack([np.ones(xv.size, dtype=np.float64), cov])
    rx = xv - C @ np.linalg.lstsq(C, xv, rcond=None)[0]
    ry = yv - C @ np.linalg.lstsq(C, yv, rcond=None)[0]
    df = xv.size - C.shape[1] - 1
    if df < 1 or np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan"), float("nan")
    r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -1.0, 1.0))
    t_val = r * np.sqrt(df / max(1e-300, 1.0 - r * r))
    return r, float(2.0 * _st.t.sf(abs(t_val), df))


def ttest_ind_safe(x: Sequence[float], y: Sequence[float], equal_var: bool = False) -> tuple[float, float]:
    """Two-sample Welch/Student t-test raising explicitly on NaN/Inf inputs."""
    xv, yv = _require_finite(x, y)
    res = _st.ttest_ind(xv, yv, equal_var=equal_var)
    return float(res.statistic), float(res.pvalue)


def mannwhitneyu_safe(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test raising explicitly on NaN/Inf inputs."""
    xv, yv = _require_finite(x, y)
    res = _st.mannwhitneyu(xv, yv, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def fisher_exact_two_sided(table: Sequence[Sequence[int]]) -> float:
    """Exact two-sided Fisher's exact test p-value without floating-point underflow floors."""
    arr = np.asarray(table, dtype=np.int64)
    return float(_st.fisher_exact(arr, alternative="two-sided").pvalue)
