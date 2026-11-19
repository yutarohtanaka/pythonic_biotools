"""Python implementation of chromVAR (Schep et al., Nature Methods 2017).

Implements the exact published chromVAR statistical workflow on raw fragment counts
(supporting both dense `DataFrame`/`ndarray` and `scipy.sparse` matrices without dense blowup):
  1. `compute_expectations`: Expected fraction of fragments per peak across cells/samples.
  2. `get_background_peaks`: Mahalanobis-whitened 2D Gaussian-kernel background peak sampling
     matched on log10 total accessibility and peak GC content.
  3. `compute_deviations`: Raw motif accessibility deviations `(O - E) / E` and background
     bias-corrected deviation Z-scores `(dev - mean(bg_dev)) / sd(bg_dev)`.
  4. `compute_variability`: Across-sample motif variability, bootstrap SDs, and chi-square p-values.
  5. `run_raw_count_pychromvar`: Optional convenience wrapper around `pychromvar` + `pyjaspar`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats


@dataclass(frozen=True)
class ChromVARResult:
    """Output of `compute_deviations` (`motifs x samples` raw deviations and bias-corrected Z-scores)."""

    deviations: pd.DataFrame
    z_scores: pd.DataFrame


def _validate_raw_counts_array(vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0) or not np.allclose(arr, np.round(arr), atol=1e-4):
        raise ValueError(
            "chromVAR requires raw non-negative integer fragment counts; do not pass TF-IDF or log-normalized values."
        )
    return arr


def _coerce_counts(
    counts_peaks_by_samples: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    sample_names: Sequence[str] | None = None,
) -> tuple[np.ndarray | sparse.csr_matrix, list[str]]:
    """Validate raw integer counts and preserve `scipy.sparse` CSR representation when sparse."""
    if sparse.issparse(counts_peaks_by_samples):
        sp = counts_peaks_by_samples.tocsr().astype(np.float64)
        _validate_raw_counts_array(sp.data)
        names = (
            [str(s) for s in sample_names]
            if sample_names is not None
            else [f"sample_{j}" for j in range(sp.shape[1])]
        )
        return sp, names

    if isinstance(counts_peaks_by_samples, pd.DataFrame):
        names = (
            [str(s) for s in sample_names]
            if sample_names is not None
            else [str(c) for c in counts_peaks_by_samples.columns]
        )
        if len(counts_peaks_by_samples.dtypes) > 0 and all(
            isinstance(dt, pd.SparseDtype) for dt in counts_peaks_by_samples.dtypes
        ):
            sp = counts_peaks_by_samples.sparse.to_coo().tocsr().astype(np.float64)
            _validate_raw_counts_array(sp.data)
            return sp, names
        arr = _validate_raw_counts_array(counts_peaks_by_samples.to_numpy())
        return arr, names

    arr = _validate_raw_counts_array(counts_peaks_by_samples)
    names = (
        [str(s) for s in sample_names]
        if sample_names is not None
        else [f"sample_{j}" for j in range(arr.shape[1])]
    )
    return arr, names


def compute_expectations(
    counts_peaks_by_samples: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
) -> np.ndarray:
    """Compute expected fragment fraction per peak `e_i = sum_j X_ij / sum_{i,j} X_ij` (Schep et al., 2017)."""
    X, _ = _coerce_counts(counts_peaks_by_samples)
    peak_totals = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    grand_total = float(peak_totals.sum())
    if grand_total <= 0.0:
        raise ValueError("Count matrix has zero total fragments.")
    return peak_totals / grand_total


def get_background_peaks(
    counts_peaks_by_samples: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    gc_content: np.ndarray | pd.Series,
    n_iterations: int = 50,
    w: float = 0.1,
    bs: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Sample `n_iterations` GC- and accessibility-matched background peaks per peak (Schep et al., 2017).

    Matches peaks in Mahalanobis-whitened `(log10(peak_fragments), gc_content)` space using a 2D
    grid of `bs x bs` bins smoothed with a Gaussian kernel of bandwidth `w`.
    """
    X, _ = _coerce_counts(counts_peaks_by_samples)
    n_peaks = X.shape[0]
    gc = np.asarray(gc_content, dtype=np.float64).ravel()
    if len(gc) != n_peaks:
        raise ValueError(f"gc_content length ({len(gc)}) must match number of peaks ({n_peaks}).")

    peak_sums = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    log_frags = np.log10(np.maximum(peak_sums, 1.0))
    feats = np.column_stack([log_frags, gc])
    cov = np.cov(feats.T) + 1e-6 * np.eye(2)
    L = np.linalg.cholesky(cov)
    norm_data = np.linalg.solve(L, (feats - feats.mean(axis=0)).T).T  # (n_peaks, 2)

    n_bins = min(bs, max(5, n_peaks // 4))
    edges0 = np.linspace(norm_data[:, 0].min() - 1e-6, norm_data[:, 0].max() + 1e-6, n_bins + 1)
    edges1 = np.linspace(norm_data[:, 1].min() - 1e-6, norm_data[:, 1].max() + 1e-6, n_bins + 1)
    centers0 = 0.5 * (edges0[:-1] + edges0[1:])
    centers1 = 0.5 * (edges1[:-1] + edges1[1:])

    bin0 = np.clip(np.digitize(norm_data[:, 0], edges0) - 1, 0, n_bins - 1)
    bin1 = np.clip(np.digitize(norm_data[:, 1], edges1) - 1, 0, n_bins - 1)
    flat_bin = bin0 * n_bins + bin1

    grid_c0, grid_c1 = np.meshgrid(centers0, centers1, indexing="ij")
    grid_pts = np.column_stack([grid_c0.ravel(), grid_c1.ravel()])  # (B^2, 2)
    bin_counts = np.bincount(flat_bin, minlength=n_bins * n_bins).astype(np.float64)

    dist2 = np.sum((grid_pts[:, None, :] - grid_pts[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * dist2 / (w ** 2))  # (B^2, B^2)

    smoothed_density = np.maximum(kernel @ bin_counts, 1e-12)
    bin_to_bin_prob = kernel * (bin_counts / smoothed_density)[None, :]
    row_sums = bin_to_bin_prob.sum(axis=1, keepdims=True)
    bin_to_bin_prob = np.divide(
        bin_to_bin_prob, row_sums, out=np.full_like(bin_to_bin_prob, 1.0 / len(grid_pts)), where=row_sums > 0
    )

    rng = np.random.default_rng(seed)
    bg_peaks = np.empty((n_peaks, n_iterations), dtype=np.int64)
    peaks_in_bin = [np.flatnonzero(flat_bin == b) for b in range(n_bins * n_bins)]
    non_empty_bins = np.flatnonzero(bin_counts > 0)

    for b in non_empty_bins:
        members = peaks_in_bin[b]
        probs = bin_to_bin_prob[b, non_empty_bins]
        probs = probs / probs.sum()
        sampled_bins = rng.choice(non_empty_bins, size=(len(members), n_iterations), p=probs)
        for idx_m, p_idx in enumerate(members):
            for it in range(n_iterations):
                cand = peaks_in_bin[int(sampled_bins[idx_m, it])]
                bg_peaks[p_idx, it] = int(cand[rng.integers(0, len(cand))])
    return bg_peaks


def compute_deviations(
    counts_peaks_by_samples: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    motif_matches_peaks_by_motifs: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    background_peaks: np.ndarray,
    expectations: np.ndarray | None = None,
    sample_names: Sequence[str] | None = None,
    motif_names: Sequence[str] | None = None,
) -> ChromVARResult:
    """Compute raw chromVAR deviations and bias-corrected Z-scores (`motifs x samples`)."""
    X, samples = _coerce_counts(counts_peaks_by_samples, sample_names=sample_names)

    is_sparse_m = sparse.issparse(motif_matches_peaks_by_motifs)
    if is_sparse_m:
        M_sp = motif_matches_peaks_by_motifs.tocsc().astype(np.float64)
        n_peaks_m, k_motifs = M_sp.shape
        motifs = (
            [str(m) for m in motif_names]
            if motif_names is not None
            else [f"motif_{k}" for k in range(k_motifs)]
        )
        M_dense = None
    elif isinstance(motif_matches_peaks_by_motifs, pd.DataFrame):
        motifs = (
            [str(m) for m in motif_names]
            if motif_names is not None
            else [str(c) for c in motif_matches_peaks_by_motifs.columns]
        )
        if len(motif_matches_peaks_by_motifs.dtypes) > 0 and all(
            isinstance(dt, pd.SparseDtype) for dt in motif_matches_peaks_by_motifs.dtypes
        ):
            M_sp = motif_matches_peaks_by_motifs.sparse.to_coo().tocsc().astype(np.float64)
            n_peaks_m, k_motifs = M_sp.shape
            is_sparse_m = True
            M_dense = None
        else:
            M_dense = np.asarray(motif_matches_peaks_by_motifs.to_numpy(), dtype=np.float64)
            n_peaks_m, k_motifs = M_dense.shape
            M_sp = None
    else:
        M_dense = np.asarray(motif_matches_peaks_by_motifs, dtype=np.float64)
        n_peaks_m, k_motifs = M_dense.shape
        motifs = (
            [str(m) for m in motif_names]
            if motif_names is not None
            else [f"motif_{k}" for k in range(k_motifs)]
        )
        M_sp = None

    n_peaks, n_samples = X.shape
    if n_peaks != n_peaks_m:
        raise ValueError("Peak dimension of counts and motif_matches must match.")

    sample_depths = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    if np.any(sample_depths <= 0):
        raise ValueError("All samples must have > 0 total fragments.")

    if expectations is None:
        peak_totals = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        exp_vec = peak_totals / float(peak_totals.sum())
    else:
        exp_vec = np.asarray(expectations, dtype=np.float64).ravel()

    use_sparse = sparse.issparse(X) or is_sparse_m
    n_iter = background_peaks.shape[1]
    bg_devs = np.empty((n_iter, k_motifs, n_samples), dtype=np.float64)

    if use_sparse:
        X_csr = X if sparse.issparse(X) else sparse.csr_matrix(X)
        M_t = (M_sp.T if M_sp is not None else sparse.csr_matrix(M_dense.T)).tocsr()
        obs_m = np.asarray((M_t @ X_csr).toarray(), dtype=np.float64)
        exp_m = np.asarray(M_t @ exp_vec, dtype=np.float64).ravel()[:, None] * sample_depths[None, :]
        raw_dev = np.divide(obs_m - exp_m, exp_m, out=np.zeros_like(obs_m), where=exp_m > 0)

        ones_p = np.ones(n_peaks, dtype=np.float64)
        rows_p = np.arange(n_peaks, dtype=np.int64)
        for b in range(n_iter):
            idx = background_peaks[:, b]
            P_b = sparse.csr_matrix((ones_p, (rows_p, idx)), shape=(n_peaks, n_peaks))
            M_bg_t = M_t @ P_b
            obs_bg = np.asarray((M_bg_t @ X_csr).toarray(), dtype=np.float64)
            exp_bg = np.asarray(M_bg_t @ exp_vec, dtype=np.float64).ravel()[:, None] * sample_depths[None, :]
            bg_devs[b] = np.divide(obs_bg - exp_bg, exp_bg, out=np.zeros_like(obs_bg), where=exp_bg > 0)
    else:
        assert M_dense is not None and isinstance(X, np.ndarray)
        obs_m = M_dense.T @ X
        exp_m = (M_dense.T @ exp_vec)[:, None] * sample_depths[None, :]
        raw_dev = np.divide(obs_m - exp_m, exp_m, out=np.zeros_like(obs_m), where=exp_m > 0)

        for b in range(n_iter):
            idx = background_peaks[:, b]
            obs_bg = M_dense.T @ X[idx, :]
            exp_bg = (M_dense.T @ exp_vec[idx])[:, None] * sample_depths[None, :]
            bg_devs[b] = np.divide(obs_bg - exp_bg, exp_bg, out=np.zeros_like(obs_bg), where=exp_bg > 0)

    bg_mean = bg_devs.mean(axis=0)
    bg_sd = np.maximum(bg_devs.std(axis=0, ddof=1), 1e-8)
    bias_corrected_dev = raw_dev - bg_mean
    z_mat = bias_corrected_dev / bg_sd

    return ChromVARResult(
        deviations=pd.DataFrame(bias_corrected_dev, index=motifs, columns=samples),
        z_scores=pd.DataFrame(z_mat, index=motifs, columns=samples),
    )


def compute_variability(
    z_scores: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute chromVAR motif variability (`SD(z)` across samples), bootstrap CIs, and chi-square p-values."""
    Z = z_scores.to_numpy(dtype=np.float64)
    k_motifs, n_samples = Z.shape
    n_eff = np.sum(np.isfinite(Z), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        variability = np.where(n_eff >= 2, np.nanstd(Z, axis=1, ddof=1), np.nan)

        rng = np.random.default_rng(seed)
        boot_sd = np.empty((n_boot, k_motifs), dtype=np.float64)
        for b in range(n_boot):
            cols = rng.integers(0, n_samples, size=n_samples)
            Z_b = Z[:, cols]
            n_b = np.sum(np.isfinite(Z_b), axis=1)
            boot_sd[b] = np.where(n_b >= 2, np.nanstd(Z_b, axis=1, ddof=1), np.nan)

        ci_lo = np.nanpercentile(boot_sd, 2.5, axis=0)
        ci_hi = np.nanpercentile(boot_sd, 97.5, axis=0)

    pvals = np.full(k_motifs, np.nan, dtype=np.float64)
    valid_m = (n_eff >= 2) & np.isfinite(variability)
    if np.any(valid_m):
        df_m = (n_eff[valid_m] - 1).astype(np.float64)
        pvals[valid_m] = stats.chi2.sf(df_m * (variability[valid_m] ** 2), df=df_m)

    q = np.full(k_motifs, np.nan, dtype=np.float64)
    valid_p = np.isfinite(pvals)
    if np.any(valid_p):
        pv = pvals[valid_p]
        order = np.argsort(pv, kind="mergesort")
        ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1, dtype=np.float64)
        q_sub = np.empty_like(pv)
        q_sub[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
        q[valid_p] = q_sub

    return pd.DataFrame(
        {
            "variability": variability,
            "bootstrap_lower_bound": ci_lo,
            "bootstrap_upper_bound": ci_hi,
            "p_value": pvals,
            "p_value_adj": q,
        },
        index=z_scores.index,
    )


def run_raw_count_pychromvar(
    counts_peaks_by_samples: pd.DataFrame,
    genome_fasta: str | Path,
    jaspar_release: str = "JASPAR2020",
    tax_group: str = "vertebrates",
    peak_delimiter: str = ":|(-)",
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience wrapper executing `pychromvar` + `pyjaspar` after validating raw integer counts."""
    vals = _validate_raw_counts_array(counts_peaks_by_samples.to_numpy())
    import anndata as ad
    import pychromvar as pc
    from pyjaspar import jaspardb

    adata = ad.AnnData(X=vals.T.astype(np.float32))
    adata.obs_names = counts_peaks_by_samples.columns.astype(str)
    adata.var_names = counts_peaks_by_samples.index.astype(str)

    np.random.seed(seed)
    pc.add_peak_seq(adata, genome_file=str(genome_fasta), delimiter=peak_delimiter)
    pc.add_gc_bias(adata)
    pc.get_bg_peaks(adata)

    jdb = jaspardb(release=jaspar_release)
    motifs = jdb.fetch_motifs(collection="CORE", tax_group=[tax_group])
    pc.match_motif(adata, motifs=motifs)
    dev = pc.compute_deviations(adata)
    return pd.DataFrame(dev.X.T, index=dev.var_names, columns=dev.obs_names)
