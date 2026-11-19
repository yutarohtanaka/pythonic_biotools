"""GC-content and mean-accessibility matched background peak sampling for `figr`."""

from __future__ import annotations

from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .io import extract_matrix


def get_background_peaks(
    atac: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    peak_gc: np.ndarray | pd.Series | Sequence[float] | str,
    *,
    n_bg: int = 100,
    n_bins: int = 50,
    bandwidth: float = 0.1,
    method: Literal["chromvar_kernel", "knn_mahalanobis"] = "chromvar_kernel",
    exclude_self: bool = True,
    layer: str | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
    seed: int = 42,
) -> np.ndarray:
    """Sample `n_bg` GC- and accessibility-matched background peak indices per target peak.

    Matches peaks in Mahalanobis-whitened `(log10(mean_accessibility), peak_gc)` space
    using either a chromVAR-style 2D Gaussian-kernel density grid (`"chromvar_kernel"`)
    or k-nearest neighbor sampling in whitened Mahalanobis space (`"knn_mahalanobis"`).

    Parameters
    ----------
    atac
        scATAC accessibility matrix (`cells x peaks` by default, or `AnnData` `obs x var`).
    peak_gc
        Per-peak GC fraction vector of length `n_peaks`, or column name in `atac.var` when
        `atac` is an `AnnData`.
    n_bg
        Number of background peaks to sample per peak.
    n_bins
        Maximum number of bins per dimension for `"chromvar_kernel"`.
    bandwidth
        Gaussian kernel bandwidth in whitened standard-deviation units (`"chromvar_kernel"`).
    method
        Background matching algorithm (`"chromvar_kernel"` or `"knn_mahalanobis"`).
    exclude_self
        If `True`, never select peak `p` as its own background peak (when `n_peaks >= 2`).
    layer
        Optional `AnnData.layers` key.
    orientation
        Orientation for `DataFrame` / `ndarray` / sparse inputs (`"cells_by_features"` or
        `"features_by_cells"`). Ignored for `AnnData`.
    seed
        Random seed for `np.random.default_rng`.

    Returns
    -------
    np.ndarray
        Integer index array of shape `(n_peaks, n_bg)` (`int64`).
    """
    if n_bg < 1:
        raise ValueError(f"`n_bg` must be >= 1, got {n_bg}.")
    if n_bins < 2:
        raise ValueError(f"`n_bins` must be >= 2, got {n_bins}.")
    if bandwidth <= 0.0:
        raise ValueError(f"`bandwidth` must be > 0, got {bandwidth}.")
    if method not in ("chromvar_kernel", "knn_mahalanobis"):
        raise ValueError(f"Unknown `method` {method!r}; expected 'chromvar_kernel' or 'knn_mahalanobis'.")

    X, _, _, _, _ = extract_matrix(
        atac,
        layer=layer,
        orientation=orientation,
        feature_prefix="peak",
    )
    _, n_peaks = X.shape
    if n_peaks < 2:
        raise ValueError(f"At least 2 peaks are required for background peak matching, got {n_peaks}.")

    if isinstance(peak_gc, str):
        if not isinstance(atac, ad.AnnData) or peak_gc not in atac.var.columns:
            raise ValueError(
                f"String `peak_gc` ({peak_gc!r}) requires an `AnnData` input with `atac.var[{peak_gc!r}]`."
            )
        gc = pd.to_numeric(atac.var[peak_gc], errors="raise").to_numpy(dtype=np.float64).ravel()
    else:
        gc = np.asarray(peak_gc, dtype=np.float64).ravel()

    if len(gc) != n_peaks:
        raise ValueError(f"`peak_gc` length ({len(gc)}) must match number of peaks ({n_peaks}).")
    if not np.all(np.isfinite(gc)):
        raise ValueError("`peak_gc` must contain only finite values.")

    mean_acc = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    if not np.all(np.isfinite(mean_acc)):
        raise ValueError("`atac` contains non-finite values.")
    log_acc = np.log10(np.maximum(mean_acc, 1e-8))

    feats = np.column_stack([log_acc, gc])
    cov = np.cov(feats.T) + 1e-6 * np.eye(2, dtype=np.float64)
    L = np.linalg.cholesky(cov)
    norm_data = np.linalg.solve(L, (feats - feats.mean(axis=0)).T).T  # (n_peaks, 2)

    rng = np.random.default_rng(seed)

    if method == "knn_mahalanobis":
        n_avail = n_peaks - (1 if exclude_self else 0)
        n_pool = min(n_avail, max(20, min(n_bg, max(20, n_peaks // 5))))
        n_query = min(n_peaks, n_pool + (1 if exclude_self else 0))
        nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
        nn.fit(norm_data)
        _, nbr_indices = nn.kneighbors(norm_data)
        bg_peaks = np.empty((n_peaks, n_bg), dtype=np.int64)
        for p_idx in range(n_peaks):
            cand = nbr_indices[p_idx]
            if exclude_self:
                cand = cand[cand != p_idx]
            cand = cand[:n_pool]
            if len(cand) == 0:
                cand = np.delete(np.arange(n_peaks, dtype=np.int64), p_idx) if exclude_self else np.arange(n_peaks, dtype=np.int64)
            replace = len(cand) < n_bg
            bg_peaks[p_idx] = rng.choice(cand, size=n_bg, replace=replace)
        return bg_peaks

    # Method: "chromvar_kernel" on occupied 2D grid bins with local ESS floor
    b_grid = min(n_bins, max(5, int(np.sqrt(n_peaks))))
    edges0 = np.linspace(norm_data[:, 0].min() - 1e-6, norm_data[:, 0].max() + 1e-6, b_grid + 1)
    edges1 = np.linspace(norm_data[:, 1].min() - 1e-6, norm_data[:, 1].max() + 1e-6, b_grid + 1)
    centers0 = 0.5 * (edges0[:-1] + edges0[1:])
    centers1 = 0.5 * (edges1[:-1] + edges1[1:])

    bin0 = np.clip(np.digitize(norm_data[:, 0], edges0) - 1, 0, b_grid - 1)
    bin1 = np.clip(np.digitize(norm_data[:, 1], edges1) - 1, 0, b_grid - 1)
    flat_bin = bin0 * b_grid + bin1

    grid_c0, grid_c1 = np.meshgrid(centers0, centers1, indexing="ij")
    grid_pts = np.column_stack([grid_c0.ravel(), grid_c1.ravel()])
    bin_counts = np.bincount(flat_bin, minlength=b_grid * b_grid).astype(np.float64)

    non_empty_bins = np.flatnonzero(bin_counts > 0)
    ne_pts = grid_pts[non_empty_bins]
    ne_counts = bin_counts[non_empty_bins]
    ne_dist2 = np.sum((ne_pts[:, None, :] - ne_pts[None, :, :]) ** 2, axis=2)

    kernel = np.exp(-0.5 * ne_dist2 / (bandwidth ** 2))
    smoothed_density = np.maximum(kernel @ ne_counts, 1e-12)
    density_inv = ne_counts / smoothed_density
    bin_to_bin_prob = kernel * density_inv[None, :]

    # Prevent degenerate 1-peak sampling on sparse/outlier bins by enforcing a local ESS floor
    target_ess = min(max(20.0, 0.5 * n_bg), float(n_peaks - (1 if exclude_self else 0)))
    for i in range(len(non_empty_bins)):
        row = bin_to_bin_prob[i]
        s = float(row.sum())
        ess = (s * s) / float(np.sum((row ** 2) / ne_counts)) if s > 0.0 else 0.0
        if ess < target_ess:
            order = np.argsort(ne_dist2[i], kind="mergesort")
            cum = np.cumsum(ne_counts[order])
            k_idx = min(len(order) - 1, int(np.searchsorted(cum, target_ess)))
            w_eff = max(bandwidth, float(np.sqrt(ne_dist2[i, order[k_idx]])) * 0.75)
            bin_to_bin_prob[i] = np.exp(-0.5 * ne_dist2[i] / (w_eff ** 2)) * density_inv

    bg_peaks = np.empty((n_peaks, n_bg), dtype=np.int64)
    peaks_in_bin = [np.flatnonzero(flat_bin == b) for b in non_empty_bins]

    for i_b, b in enumerate(non_empty_bins):
        members = peaks_in_bin[i_b]
        if exclude_self and len(members) == 1 and len(non_empty_bins) > 1:
            p_idx = int(members[0])
            valid_pos = np.flatnonzero(non_empty_bins != b)
            probs = bin_to_bin_prob[i_b, valid_pos]
            prob_sum = float(probs.sum())
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                probs = np.full(len(valid_pos), 1.0 / len(valid_pos), dtype=np.float64)
            else:
                probs = probs / prob_sum
            chosen_pos = rng.choice(valid_pos, size=n_bg, p=probs)
            for it in range(n_bg):
                cand = peaks_in_bin[int(chosen_pos[it])]
                bg_peaks[p_idx, it] = int(cand[rng.integers(0, len(cand))])
        else:
            probs = bin_to_bin_prob[i_b]
            prob_sum = float(probs.sum())
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                probs = np.full(len(non_empty_bins), 1.0 / len(non_empty_bins), dtype=np.float64)
            else:
                probs = probs / prob_sum
            chosen_pos = rng.choice(len(non_empty_bins), size=(len(members), n_bg), p=probs)
            for idx_m, p_idx in enumerate(members):
                for it in range(n_bg):
                    cp = int(chosen_pos[idx_m, it])
                    if exclude_self and cp == i_b and len(members) > 1:
                        r_idx = int(rng.integers(0, len(members) - 1))
                        if r_idx >= idx_m:
                            r_idx += 1
                        bg_peaks[int(p_idx), it] = int(members[r_idx])
                    else:
                        cand = peaks_in_bin[cp]
                        bg_peaks[int(p_idx), it] = int(cand[rng.integers(0, len(cand))])

    return bg_peaks
