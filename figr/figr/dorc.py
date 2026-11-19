"""DORC calling (`call_dorcs`), cell kNN smoothing (`build_knn_graph`, `smooth_scores_knn`), and DORC scoring (`get_dorc_scores`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .corr import GenePeakCorrResult, _normalize_cells_matrix
from .io import SettingsMode, extract_matrix, validate_settings


@dataclass(frozen=True)
class DORCCallResult:
    """Typed result container for `call_dorcs`."""

    gene: np.ndarray
    n_peaks: np.ndarray
    mean_r: np.ndarray
    mean_z: np.ndarray
    min_p: np.ndarray
    rank: np.ndarray
    is_dorc: np.ndarray
    dorc_genes: list[str]
    dorc_k: int
    filtered_links: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return ranked gene-level DORC calling summary as a pandas DataFrame."""
        cols = ["gene", "n_peaks", "mean_r", "mean_z", "min_p", "rank", "is_dorc"]
        if len(self.gene) == 0:
            return pd.DataFrame(
                {
                    "gene": pd.Series(dtype=object),
                    "n_peaks": pd.Series(dtype=np.int64),
                    "mean_r": pd.Series(dtype=np.float64),
                    "mean_z": pd.Series(dtype=np.float64),
                    "min_p": pd.Series(dtype=np.float64),
                    "rank": pd.Series(dtype=np.int64),
                    "is_dorc": pd.Series(dtype=bool),
                },
                columns=cols,
            )
        return pd.DataFrame(
            {
                "gene": self.gene,
                "n_peaks": self.n_peaks,
                "mean_r": self.mean_r,
                "mean_z": self.mean_z,
                "min_p": self.min_p,
                "rank": self.rank,
                "is_dorc": self.is_dorc,
            }
        )

    def dorc_table(self) -> pd.DataFrame:
        """Return only genes meeting `n_peaks >= dorc_k`."""
        df = self.to_dataframe()
        return df.loc[df["is_dorc"]].reset_index(drop=True)

    def links_dataframe(self) -> pd.DataFrame:
        """Return the filtered, deduplicated peak-gene links used to call DORCs."""
        return self.filtered_links.copy()

    def summary(self) -> str:
        """Return a human-readable summary of DORC calls."""
        return (
            f"DORCCallResult(n_linked_genes={len(self.gene)}, "
            f"n_dorcs={len(self.dorc_genes)}, dorc_k={self.dorc_k}, "
            f"n_links={len(self.filtered_links)})"
        )


@dataclass(frozen=True)
class DORCScoresResult:
    """Typed result container for `get_dorc_scores`."""

    cell_names: list[str]
    dorc_genes: list[str]
    n_peaks_per_dorc: np.ndarray
    raw_scores: np.ndarray
    smoothed_scores: np.ndarray | None
    agg: str
    normalized: bool
    scale_factor: float

    def to_dataframe(self, smoothed: bool = True) -> pd.DataFrame:
        """Return per-cell DORC activity matrix (`cells x dorc_genes`) as a DataFrame."""
        mat = self.smoothed_scores if (smoothed and self.smoothed_scores is not None) else self.raw_scores
        return pd.DataFrame(mat, index=self.cell_names, columns=self.dorc_genes)

    def raw_dataframe(self) -> pd.DataFrame:
        """Return unsmoothed per-cell DORC scores (`cells x dorc_genes`)."""
        return pd.DataFrame(self.raw_scores, index=self.cell_names, columns=self.dorc_genes)

    def smoothed_dataframe(self) -> pd.DataFrame | None:
        """Return kNN-smoothed per-cell DORC scores if computed, else `None`."""
        if self.smoothed_scores is None:
            return None
        return pd.DataFrame(self.smoothed_scores, index=self.cell_names, columns=self.dorc_genes)

    def summary(self) -> str:
        """Return a human-readable summary of DORC scoring."""
        return (
            f"DORCScoresResult(n_cells={len(self.cell_names)}, "
            f"n_dorcs={len(self.dorc_genes)}, agg={self.agg!r}, "
            f"normalized={self.normalized}, smoothed={self.smoothed_scores is not None})"
        )


def call_dorcs(
    corr_result: GenePeakCorrResult | pd.DataFrame,
    *,
    dorc_k: int | None = None,
    p_cut: float | None = 0.05,
    fdr_cut: float | None = None,
    min_r: float | None = None,
    settings: SettingsMode = "default",
) -> DORCCallResult:
    """Call Domains of Regulatory Chromatin (DORCs) from cis peak-gene correlation results."""
    validate_settings(settings)
    eff_dorc_k = (7 if settings == "R_default" else 5) if dorc_k is None else int(dorc_k)
    if eff_dorc_k < 1:
        raise ValueError(f"`dorc_k` must be >= 1, got {eff_dorc_k}.")

    df = corr_result.to_dataframe() if isinstance(corr_result, GenePeakCorrResult) else corr_result.copy()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`corr_result` must be a `GenePeakCorrResult` or `pd.DataFrame`.")

    required_cols = {"gene", "peak"}
    if p_cut is not None:
        required_cols.add("p_value")
    if fdr_cut is not None:
        required_cols.add("fdr")
    if min_r is not None:
        required_cols.add("r_obs")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in `corr_result`: {sorted(missing)}.")

    mask = np.ones(len(df), dtype=bool)
    if p_cut is not None:
        pv = pd.to_numeric(df["p_value"], errors="coerce").to_numpy(dtype=np.float64)
        mask &= np.isfinite(pv) & (pv <= p_cut)
    if fdr_cut is not None:
        fv = pd.to_numeric(df["fdr"], errors="coerce").to_numpy(dtype=np.float64)
        mask &= np.isfinite(fv) & (fv <= fdr_cut)
    if min_r is not None:
        rv = pd.to_numeric(df["r_obs"], errors="coerce").to_numpy(dtype=np.float64)
        mask &= np.isfinite(rv) & (rv >= min_r)

    filtered = df.loc[mask].copy()
    if filtered.empty:
        return DORCCallResult(
            gene=np.empty(0, dtype=object),
            n_peaks=np.empty(0, dtype=np.int64),
            mean_r=np.empty(0, dtype=np.float64),
            mean_z=np.empty(0, dtype=np.float64),
            min_p=np.empty(0, dtype=np.float64),
            rank=np.empty(0, dtype=np.int64),
            is_dorc=np.empty(0, dtype=bool),
            dorc_genes=[],
            dorc_k=eff_dorc_k,
            filtered_links=filtered.reset_index(drop=True),
        )

    if "p_value" in filtered.columns:
        filtered = filtered.sort_values("p_value", kind="mergesort", na_position="last")
    filtered = filtered.drop_duplicates(subset=["gene", "peak"], keep="first").reset_index(drop=True)

    if "r_obs" not in filtered.columns:
        filtered["r_obs"] = np.nan
    if "z_score" not in filtered.columns:
        filtered["z_score"] = np.nan
    if "p_value" not in filtered.columns:
        filtered["p_value"] = np.nan

    grouped = (
        filtered.groupby("gene", sort=False, as_index=False)
        .agg(
            n_peaks=("peak", "count"),
            mean_r=("r_obs", "mean"),
            mean_z=("z_score", "mean"),
            min_p=("p_value", "min"),
        )
    )
    grouped = grouped.sort_values(
        by=["n_peaks", "mean_z", "gene"],
        ascending=[False, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    n_genes = len(grouped)
    rank = np.arange(1, n_genes + 1, dtype=np.int64)
    n_peaks_arr = grouped["n_peaks"].to_numpy(dtype=np.int64)
    is_dorc = n_peaks_arr >= eff_dorc_k
    gene_arr = grouped["gene"].astype(str).to_numpy(dtype=object)
    dorc_genes = [str(g) for g in gene_arr[is_dorc]]

    return DORCCallResult(
        gene=gene_arr,
        n_peaks=n_peaks_arr,
        mean_r=grouped["mean_r"].to_numpy(dtype=np.float64),
        mean_z=grouped["mean_z"].to_numpy(dtype=np.float64),
        min_p=grouped["min_p"].to_numpy(dtype=np.float64),
        rank=rank,
        is_dorc=is_dorc,
        dorc_genes=dorc_genes,
        dorc_k=eff_dorc_k,
        filtered_links=filtered,
    )


def _is_adjacency_matrix(arr: np.ndarray) -> bool:
    """Disambiguate `(N, N)` adjacency/weight matrix vs `(N, k)` integer neighbor index matrix."""
    if arr.shape[0] != arr.shape[1]:
        return False
    if not np.issubdtype(arr.dtype, np.integer):
        return True
    n = arr.shape[0]
    if n == 1 and arr[0, 0] == 1:
        return True
    if n == 2 and np.all((arr == 0) | (arr == 1)) and np.all(np.diag(arr) == 1):
        return True
    if n > 2 and np.all((arr == 0) | (arr == 1)):
        # A valid (N, N) neighbor index matrix for N > 2 must contain indices >= 2
        return True
    return False


def build_knn_graph(
    knn: np.ndarray | sparse.spmatrix | sparse.sparray | None = None,
    *,
    embedding: np.ndarray | pd.DataFrame | None = None,
    k: int = 30,
    include_self: bool = True,
    weights: Literal["uniform", "distance", "gaussian"] = "uniform",
    bandwidth: float | None = None,
    knn_type: Literal["auto", "indices", "adjacency"] = "auto",
) -> sparse.csr_matrix:
    """Build a row-stochastic `(N, N)` CSR kNN smoothing matrix `W` (every row sums to 1.0)."""
    if knn is None and embedding is None:
        raise ValueError("Provide either `knn` (neighbor indices or adjacency matrix) or `embedding`.")
    if weights not in ("uniform", "distance", "gaussian"):
        raise ValueError(f"Invalid `weights` {weights!r}; expected 'uniform', 'distance', or 'gaussian'.")
    if bandwidth is not None and bandwidth <= 0.0:
        raise ValueError(f"`bandwidth` must be > 0, got {bandwidth}.")
    if knn_type not in ("auto", "indices", "adjacency"):
        raise ValueError(f"Invalid `knn_type` {knn_type!r}; expected 'auto', 'indices', or 'adjacency'.")

    if knn is not None:
        if sparse.issparse(knn):
            if knn_type == "indices":
                raise ValueError("Sparse `knn` matrix cannot be interpreted as neighbor `indices`.")
            W = knn.tocsr().astype(np.float64)
            if W.ndim != 2 or W.shape[0] != W.shape[1]:
                raise ValueError(f"Sparse `knn` adjacency matrix must be square, got {W.shape}.")
            if W.shape[0] == 0:
                raise ValueError("`knn` matrix cannot be empty.")
            if np.any(~np.isfinite(W.data)) or np.any(W.data < 0.0):
                raise ValueError("`knn` weights must be finite and non-negative.")
            return _finalize_row_stochastic(W, include_self=include_self)

        arr = np.asarray(knn)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"`knn` must be a non-empty 2D array, got shape {arr.shape}.")

        if knn_type == "adjacency":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(f"`knn` adjacency matrix must be square, got shape {arr.shape}.")
            treat_as_adjacency = True
        elif knn_type == "indices":
            treat_as_adjacency = False
        else:
            treat_as_adjacency = _is_adjacency_matrix(arr)

        if treat_as_adjacency:
            arr_f = np.asarray(arr, dtype=np.float64)
            if np.any(~np.isfinite(arr_f)) or np.any(arr_f < 0.0):
                raise ValueError("`knn` adjacency/weight matrix must be finite and non-negative.")
            return _finalize_row_stochastic(sparse.csr_matrix(arr_f), include_self=include_self)

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("Neighbor index `knn` array must have an integer dtype.")
        n_cells, n_nbrs = arr.shape
        if np.any(arr < 0) or np.any(arr >= n_cells):
            raise ValueError(f"`knn` neighbor indices out of range [0, {n_cells}).")

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n_cells):
            seen = set(int(x) for x in arr[i])
            if include_self:
                seen.add(i)
            if not seen:
                seen.add(i)
            w = 1.0 / len(seen)
            for j in sorted(seen):
                rows.append(i)
                cols.append(j)
                vals.append(w)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float64)

    # Build from embedding
    if isinstance(embedding, pd.DataFrame):
        emb = embedding.to_numpy(dtype=np.float64)
    else:
        emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
        raise ValueError(f"`embedding` must be a non-empty 2D array, got shape {emb.shape}.")
    if not np.all(np.isfinite(emb)):
        raise ValueError("`embedding` must contain only finite values.")
    if k < 1:
        raise ValueError(f"`k` must be >= 1, got {k}.")

    n_cells = emb.shape[0]
    if n_cells == 1:
        return sparse.identity(1, dtype=np.float64, format="csr")

    n_query = min(n_cells, k if include_self else k + 1)
    nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
    nn.fit(emb)
    dists, inds = nn.kneighbors(emb)

    if not include_self:
        filt_inds = np.empty((n_cells, min(k, n_cells - 1)), dtype=np.int64)
        filt_dists = np.empty((n_cells, min(k, n_cells - 1)), dtype=np.float64)
        target_k = filt_inds.shape[1]
        for i in range(n_cells):
            mask = inds[i] != i
            row_i = inds[i][mask][:target_k]
            row_d = dists[i][mask][:target_k]
            if len(row_i) < target_k:
                # Duplicate-point tie fallback: fill remaining slots with non-self points
                others = [j for j in inds[i] if j != i]
                while len(others) < target_k:
                    others.append((i + 1) % n_cells)
                row_i = np.asarray(others[:target_k], dtype=np.int64)
                row_d = np.linalg.norm(emb[row_i] - emb[i], axis=1)
            filt_inds[i] = row_i
            filt_dists[i] = row_d
        inds = filt_inds
        dists = filt_dists

    rows_out: list[int] = []
    cols_out: list[int] = []
    vals_out: list[float] = []
    eps = 1e-12

    global_pos_dists = dists[dists > 0.0]
    default_bw = float(np.median(global_pos_dists)) if len(global_pos_dists) > 0 else 1.0
    default_bw = max(default_bw, eps)

    for i in range(n_cells):
        row_i = inds[i]
        row_d = dists[i]
        if include_self and i not in row_i:
            # Replace furthest neighbor with self if duplicate coordinates displaced self
            row_i = np.concatenate([[i], row_i[:-1]])
            row_d = np.concatenate([[0.0], row_d[:-1]])

        if weights == "uniform":
            w_row = np.full(len(row_i), 1.0 / len(row_i), dtype=np.float64)
        elif weights == "distance":
            pos_d = row_d[row_d > eps]
            d_floor = float(np.min(pos_d)) * 0.5 if len(pos_d) > 0 else 1.0
            raw_w = 1.0 / np.maximum(row_d, d_floor)
            w_row = raw_w / float(np.sum(raw_w))
        else:  # gaussian
            if bandwidth is not None:
                sigma = bandwidth
            else:
                pos_d = row_d[row_d > 0.0]
                sigma = max(float(np.median(pos_d)) if len(pos_d) > 0 else default_bw, eps)
            raw_w = np.exp(-0.5 * (row_d ** 2) / (sigma ** 2))
            w_sum = float(np.sum(raw_w))
            w_row = raw_w / w_sum if w_sum > 0.0 else np.full(len(row_i), 1.0 / len(row_i), dtype=np.float64)

        rows_out.extend([i] * len(row_i))
        cols_out.extend(int(j) for j in row_i)
        vals_out.extend(float(w) for w in w_row)

    return sparse.csr_matrix((vals_out, (rows_out, cols_out)), shape=(n_cells, n_cells), dtype=np.float64)


def _finalize_row_stochastic(W: sparse.csr_matrix, include_self: bool) -> sparse.csr_matrix:
    """Ensure diagonal self-loops for zero-sum/missing-self rows and row-normalize CSR matrix `W`."""
    n = W.shape[0]
    W_lil = W.tolil(copy=True)
    row_sums = np.asarray(W.sum(axis=1), dtype=np.float64).ravel()
    diag = W.diagonal()

    if include_self:
        # Add unit self-loop when diagonal is zero and row is binary or empty
        for i in range(n):
            if diag[i] == 0.0:
                if row_sums[i] == 0.0:
                    W_lil[i, i] = 1.0
                else:
                    row_vals = W_lil.data[i]
                    W_lil[i, i] = float(np.mean(row_vals)) if len(row_vals) > 0 else 1.0
    else:
        for i in range(n):
            if row_sums[i] == 0.0:
                W_lil[i, i] = 1.0

    W_csr = W_lil.tocsr()
    new_sums = np.asarray(W_csr.sum(axis=1), dtype=np.float64).ravel()
    inv = np.where(new_sums > 0.0, 1.0 / np.maximum(new_sums, 1e-12), 1.0)
    return (sparse.diags(inv) @ W_csr).tocsr()


def smooth_scores_knn(
    scores: pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    knn: np.ndarray | sparse.spmatrix | sparse.sparray | None = None,
    *,
    embedding: np.ndarray | pd.DataFrame | None = None,
    k: int = 30,
    include_self: bool = True,
    weights: Literal["uniform", "distance", "gaussian"] = "uniform",
    bandwidth: float | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
) -> pd.DataFrame | np.ndarray:
    """Smooth per-cell score profiles across a cell kNN graph (`S_smooth = W @ S_raw`)."""
    if orientation not in ("cells_by_features", "features_by_cells"):
        raise ValueError(f"Invalid `orientation` {orientation!r}.")

    is_df = isinstance(scores, pd.DataFrame)
    if is_df:
        arr = scores.to_numpy(dtype=np.float64)
        mat = arr.T if orientation == "features_by_cells" else arr
    elif sparse.issparse(scores):
        mat = scores.tocsr().astype(np.float64)
        if orientation == "features_by_cells":
            mat = mat.T.tocsr()
    else:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"`scores` must be 2D, got shape {arr.shape}.")
        mat = arr.T if orientation == "features_by_cells" else arr

    W = build_knn_graph(
        knn=knn,
        embedding=embedding,
        k=k,
        include_self=include_self,
        weights=weights,
        bandwidth=bandwidth,
    )
    if W.shape[0] != mat.shape[0]:
        raise ValueError(
            f"kNN graph cell count ({W.shape[0]}) must match `scores` cell count ({mat.shape[0]})."
        )

    smoothed = W @ mat
    if sparse.issparse(smoothed):
        smoothed_arr = smoothed.toarray()
    else:
        smoothed_arr = np.asarray(smoothed, dtype=np.float64)

    if orientation == "features_by_cells":
        smoothed_out = smoothed_arr.T
    else:
        smoothed_out = smoothed_arr

    if is_df:
        return pd.DataFrame(smoothed_out, index=scores.index, columns=scores.columns)
    return smoothed_out


def get_dorc_scores(
    atac: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    dorcs: DORCCallResult | GenePeakCorrResult | pd.DataFrame,
    *,
    gene_list: Sequence[str] | None = None,
    normalize_atac: bool = True,
    scale_factor: float = 1e4,
    agg: Literal["sum", "mean"] = "sum",
    knn: np.ndarray | sparse.spmatrix | sparse.sparray | None = None,
    embedding: np.ndarray | pd.DataFrame | None = None,
    k: int = 30,
    include_self: bool = True,
    weights: Literal["uniform", "distance", "gaussian"] = "uniform",
    bandwidth: float | None = None,
    layer: str | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
) -> DORCScoresResult:
    """Compute per-cell DORC accessibility scores and optionally smooth over a cell kNN graph."""
    if agg not in ("sum", "mean"):
        raise ValueError(f"Invalid `agg` {agg!r}; expected 'sum' or 'mean'.")

    X_atac, cell_names, peak_names, _, _ = extract_matrix(
        atac,
        layer=layer,
        orientation=orientation,
        feature_prefix="peak",
    )
    n_cells, n_peaks = X_atac.shape

    if isinstance(dorcs, DORCCallResult):
        links_df = dorcs.filtered_links
        target_genes = list(gene_list) if gene_list is not None else list(dorcs.dorc_genes)
    elif isinstance(dorcs, GenePeakCorrResult):
        links_df = dorcs.to_dataframe()
        target_genes = (
            list(gene_list)
            if gene_list is not None
            else sorted({str(g) for g in links_df["gene"]})
        )
    elif isinstance(dorcs, pd.DataFrame):
        links_df = dorcs
        if not {"gene", "peak"}.issubset(links_df.columns):
            raise ValueError("`dorcs` DataFrame must contain 'gene' and 'peak' columns.")
        target_genes = (
            list(gene_list)
            if gene_list is not None
            else sorted({str(g) for g in links_df["gene"]})
        )
    else:
        raise TypeError("`dorcs` must be a `DORCCallResult`, `GenePeakCorrResult`, or `pd.DataFrame`.")

    # Preserve order while deduplicating target_genes
    seen_g: set[str] = set()
    ordered_genes: list[str] = []
    for g in target_genes:
        gs = str(g)
        if gs not in seen_g:
            seen_g.add(gs)
            ordered_genes.append(gs)

    if len(ordered_genes) == 0:
        raw_empty = np.zeros((n_cells, 0), dtype=np.float64)
        smoothed_empty = (
            np.zeros((n_cells, 0), dtype=np.float64) if (knn is not None or embedding is not None) else None
        )
        return DORCScoresResult(
            cell_names=cell_names,
            dorc_genes=[],
            n_peaks_per_dorc=np.zeros(0, dtype=np.int64),
            raw_scores=raw_empty,
            smoothed_scores=smoothed_empty,
            agg=agg,
            normalized=normalize_atac,
            scale_factor=scale_factor,
        )

    peak_to_idx = {str(p): idx for idx, p in enumerate(peak_names)}
    gene_to_row = {g: i for i, g in enumerate(ordered_genes)}

    link_pairs = links_df[["gene", "peak"]].drop_duplicates()
    ind_rows: list[int] = []
    ind_cols: list[int] = []
    counts_per_dorc = np.zeros(len(ordered_genes), dtype=np.int64)

    for g_val, p_val in zip(link_pairs["gene"].astype(str), link_pairs["peak"].astype(str)):
        d_idx = gene_to_row.get(g_val)
        p_idx = peak_to_idx.get(p_val)
        if d_idx is not None and p_idx is not None:
            ind_rows.append(d_idx)
            ind_cols.append(p_idx)
            counts_per_dorc[d_idx] += 1

    if len(ind_rows) == 0:
        raise ValueError("Zero peaks overlap between `dorcs` links and `atac` peak names.")

    if normalize_atac:
        X_norm = _normalize_cells_matrix(X_atac, scale_factor=scale_factor, log1p=False)
    else:
        X_norm = X_atac

    G = sparse.csc_matrix(
        (np.ones(len(ind_rows), dtype=np.float64), (ind_rows, ind_cols)),
        shape=(len(ordered_genes), n_peaks),
        dtype=np.float64,
    )
    proj = X_norm @ G.T
    raw_scores = np.asarray(proj.toarray() if sparse.issparse(proj) else proj, dtype=np.float64)

    if agg == "mean":
        denom = np.maximum(counts_per_dorc, 1).astype(np.float64)
        raw_scores = raw_scores / denom[None, :]

    smoothed_scores: np.ndarray | None = None
    if knn is not None or embedding is not None:
        smoothed_scores = np.asarray(
            smooth_scores_knn(
                raw_scores,
                knn=knn,
                embedding=embedding,
                k=k,
                include_self=include_self,
                weights=weights,
                bandwidth=bandwidth,
                orientation="cells_by_features",
            ),
            dtype=np.float64,
        )

    return DORCScoresResult(
        cell_names=cell_names,
        dorc_genes=ordered_genes,
        n_peaks_per_dorc=counts_per_dorc,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        agg=agg,
        normalized=normalize_atac,
        scale_factor=scale_factor,
    )
