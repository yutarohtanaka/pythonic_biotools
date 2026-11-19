"""Python implementation of chromVAR (Schep et al., Nature Methods 2017).

Implements the exact published chromVAR statistical workflow on raw fragment counts:
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
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ChromVARResult:
    """Output of `compute_deviations` (`motifs x samples` raw deviations and bias-corrected Z-scores)."""

    deviations: pd.DataFrame
    z_scores: pd.DataFrame


def _validate_raw_counts(counts_peaks_by_samples: np.ndarray) -> np.ndarray:
    vals = np.asarray(counts_peaks_by_samples, dtype=np.float64)
    if np.any(vals < 0.0) or not np.allclose(vals, np.round(vals), atol=1e-4):
        raise ValueError(
            "chromVAR requires raw non-negative integer fragment counts; do not pass TF-IDF or log-normalized values."
        )
    return vals


def compute_expectations(counts_peaks_by_samples: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Compute expected fragment fraction per peak `e_i = sum_j X_ij / sum_{i,j} X_ij` (Schep et al., 2017)."""
    X = _validate_raw_counts(counts_peaks_by_samples.to_numpy() if isinstance(counts_peaks_by_samples, pd.DataFrame) else counts_peaks_by_samples)
    peak_totals = X.sum(axis=1)
    grand_total = float(peak_totals.sum())
    if grand_total <= 0.0:
        raise ValueError("Count matrix has zero total fragments.")
    return peak_totals / grand_total


def get_background_peaks(
    counts_peaks_by_samples: pd.DataFrame | np.ndarray,
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
    X = _validate_raw_counts(counts_peaks_by_samples.to_numpy() if isinstance(counts_peaks_by_samples, pd.DataFrame) else counts_peaks_by_samples)
    n_peaks = X.shape[0]
    gc = np.asarray(gc_content, dtype=np.float64).ravel()
    if len(gc) != n_peaks:
        raise ValueError(f"gc_content length ({len(gc)}) must match number of peaks ({n_peaks}).")

    log_frags = np.log10(np.maximum(X.sum(axis=1), 1.0))
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

    # Schep et al. (2017): weight each candidate peak q by K(bin(p), bin(q)) / smoothed_density(bin(q)),
    # so summing over the bin_counts[q] peaks in bin q gives bin weight K(b, q) * bin_counts[q] / density[q].
    smoothed_density = np.maximum(kernel @ bin_counts, 1e-12)
    bin_to_bin_prob = kernel * (bin_counts / smoothed_density)[None, :]
    row_sums = bin_to_bin_prob.sum(axis=1, keepdims=True)
    bin_to_bin_prob = np.divide(bin_to_bin_prob, row_sums, out=np.full_like(bin_to_bin_prob, 1.0 / len(grid_pts)), where=row_sums > 0)

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
    counts_peaks_by_samples: pd.DataFrame,
    motif_matches_peaks_by_motifs: pd.DataFrame,
    background_peaks: np.ndarray,
    expectations: np.ndarray | None = None,
) -> ChromVARResult:
    """Compute raw chromVAR deviations and bias-corrected Z-scores (`motifs x samples`)."""
    X = _validate_raw_counts(counts_peaks_by_samples.to_numpy())  # (P, S)
    M = np.asarray(motif_matches_peaks_by_motifs.to_numpy(), dtype=np.float64)  # (P, K)
    if X.shape[0] != M.shape[0]:
        raise ValueError("Peak dimension of counts and motif_matches must match.")

    exp_vec = compute_expectations(X) if expectations is None else np.asarray(expectations, dtype=np.float64)
    sample_depths = X.sum(axis=0)  # (S,)
    if np.any(sample_depths <= 0):
        raise ValueError("All samples must have > 0 total fragments.")

    obs_m = M.T @ X  # (K, S)
    exp_m = (M.T @ exp_vec)[:, None] * sample_depths[None, :]  # (K, S)
    raw_dev = np.divide(obs_m - exp_m, exp_m, out=np.zeros_like(obs_m), where=exp_m > 0)

    n_iter = background_peaks.shape[1]
    bg_devs = np.empty((n_iter, M.shape[1], X.shape[1]), dtype=np.float64)
    for b in range(n_iter):
        idx = background_peaks[:, b]
        M_bg = M[idx, :]  # wait: peak i with motif k replaced by background peak bg[i, b]
        # Under chromVAR, for motif k present in peaks P_k, its background set in iteration b is {bg[i, b] : i in P_k}
        obs_bg = M.T @ X[idx, :]
        exp_bg = (M.T @ exp_vec[idx])[:, None] * sample_depths[None, :]
        bg_devs[b] = np.divide(obs_bg - exp_bg, exp_bg, out=np.zeros_like(obs_bg), where=exp_bg > 0)

    bg_mean = bg_devs.mean(axis=0)
    bg_sd = np.maximum(bg_devs.std(axis=0, ddof=1), 1e-8)
    bias_corrected_dev = raw_dev - bg_mean
    z_mat = bias_corrected_dev / bg_sd

    motifs = list(motif_matches_peaks_by_motifs.columns)
    samples = list(counts_peaks_by_samples.columns)
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
    variability = np.nanstd(Z, axis=1, ddof=1)

    rng = np.random.default_rng(seed)
    boot_sd = np.empty((n_boot, k_motifs), dtype=np.float64)
    for b in range(n_boot):
        cols = rng.integers(0, n_samples, size=n_samples)
        boot_sd[b] = np.nanstd(Z[:, cols], axis=1, ddof=1)

    ci_lo = np.percentile(boot_sd, 2.5, axis=0)
    ci_hi = np.percentile(boot_sd, 97.5, axis=0)
    pvals = stats.chi2.sf((n_samples - 1) * (variability ** 2), df=max(n_samples - 1, 1))
    order = np.argsort(pvals)
    q = np.empty_like(pvals)
    ranked = pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1)
    q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)

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
    vals = _validate_raw_counts(counts_peaks_by_samples.to_numpy())
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
