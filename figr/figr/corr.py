"""Cis-window peak-to-gene Spearman correlation and background null Z-test (`run_gene_peak_corr`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .background import get_background_peaks
from .io import (
    SettingsMode,
    align_cells,
    bh_fdr,
    coerce_peaks,
    coerce_tss_table,
    extract_matrix,
    validate_settings,
)


@dataclass(frozen=True)
class GenePeakCorrResult:
    """Typed result container for `run_gene_peak_corr`."""

    gene: np.ndarray
    peak: np.ndarray
    chrom: np.ndarray
    tss: np.ndarray
    peak_start: np.ndarray
    peak_end: np.ndarray
    tss_distance: np.ndarray
    r_obs: np.ndarray
    bg_mean: np.ndarray
    bg_std: np.ndarray
    z_score: np.ndarray
    p_value: np.ndarray
    fdr: np.ndarray
    n_cells: int
    n_bg: int
    window_pad: int
    alternative: str

    def to_dataframe(self, sort_by_pvalue: bool = False) -> pd.DataFrame:
        """Return tidy per-(gene, peak) correlation results as a pandas DataFrame."""
        cols = [
            "gene",
            "peak",
            "chrom",
            "tss",
            "peak_start",
            "peak_end",
            "tss_distance",
            "r_obs",
            "bg_mean",
            "bg_std",
            "z_score",
            "p_value",
            "fdr",
        ]
        if len(self.gene) == 0:
            return pd.DataFrame(
                {
                    "gene": pd.Series(dtype=object),
                    "peak": pd.Series(dtype=object),
                    "chrom": pd.Series(dtype=object),
                    "tss": pd.Series(dtype=np.int64),
                    "peak_start": pd.Series(dtype=np.int64),
                    "peak_end": pd.Series(dtype=np.int64),
                    "tss_distance": pd.Series(dtype=np.int64),
                    "r_obs": pd.Series(dtype=np.float64),
                    "bg_mean": pd.Series(dtype=np.float64),
                    "bg_std": pd.Series(dtype=np.float64),
                    "z_score": pd.Series(dtype=np.float64),
                    "p_value": pd.Series(dtype=np.float64),
                    "fdr": pd.Series(dtype=np.float64),
                },
                columns=cols,
            )
        df = pd.DataFrame(
            {
                "gene": self.gene,
                "peak": self.peak,
                "chrom": self.chrom,
                "tss": self.tss,
                "peak_start": self.peak_start,
                "peak_end": self.peak_end,
                "tss_distance": self.tss_distance,
                "r_obs": self.r_obs,
                "bg_mean": self.bg_mean,
                "bg_std": self.bg_std,
                "z_score": self.z_score,
                "p_value": self.p_value,
                "fdr": self.fdr,
            }
        )
        if sort_by_pvalue:
            df = df.sort_values("p_value", kind="mergesort", na_position="last").reset_index(drop=True)
        return df

    def significant(
        self,
        p_cut: float | None = 0.05,
        fdr_cut: float | None = None,
        min_r: float | None = None,
    ) -> pd.DataFrame:
        """Filter cis peak-gene links by p-value, BH-FDR, and/or minimum observed correlation."""
        df = self.to_dataframe(sort_by_pvalue=True)
        mask = np.ones(len(df), dtype=bool)
        if p_cut is not None:
            p_arr = df["p_value"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(p_arr) & (p_arr <= p_cut)
        if fdr_cut is not None:
            f_arr = df["fdr"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(f_arr) & (f_arr <= fdr_cut)
        if min_r is not None:
            r_arr = df["r_obs"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(r_arr) & (r_arr >= min_r)
        return df.loc[mask].reset_index(drop=True)

    def summary(self) -> str:
        """Return a human-readable summary of peak-gene correlation results."""
        n_total = len(self.gene)
        valid = np.isfinite(self.p_value)
        n_valid = int(np.sum(valid))
        n_p05 = int(np.sum(self.p_value[valid] <= 0.05)) if n_valid else 0
        n_fdr05 = int(np.sum(self.fdr[valid] <= 0.05)) if n_valid else 0
        return (
            f"GenePeakCorrResult(n_pairs={n_total}, n_valid={n_valid}, "
            f"n_cells={self.n_cells}, n_bg={self.n_bg}, window_pad={self.window_pad}, "
            f"alternative={self.alternative!r}, sig_p05={n_p05}, sig_fdr05={n_fdr05})"
        )


def _normalize_cells_matrix(
    mat: np.ndarray | sparse.csr_matrix,
    scale_factor: float,
    log1p: bool = False,
) -> np.ndarray | sparse.csr_matrix:
    """Library-size normalize each cell row without unnecessary densification."""
    depths = np.asarray(mat.sum(axis=1), dtype=np.float64).ravel()
    inv_depths = np.where(depths > 0.0, scale_factor / np.maximum(depths, 1e-12), 0.0)
    if sparse.issparse(mat):
        scaled = (sparse.diags(inv_depths) @ mat).tocsr()
        if log1p:
            scaled = scaled.copy()
            scaled.data = np.log1p(scaled.data)
        return scaled
    scaled_dense = mat * inv_depths[:, None]
    if log1p:
        scaled_dense = np.log1p(scaled_dense)
    return scaled_dense


def _unit_rank_dense(v: np.ndarray) -> tuple[bool, np.ndarray]:
    """Compute centered, unit-L2-normalized average-tie Spearman rank vector for a 1D array."""
    n = v.shape[0]
    if n < 2:
        return False, np.zeros(n, dtype=np.float64)
    ranks = stats.rankdata(v, method="average")
    centered = ranks - 0.5 * (n + 1)
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12 or not np.isfinite(norm):
        return False, np.zeros(n, dtype=np.float64)
    return True, centered / norm


def _unit_rank_sparse_col(X_csc: sparse.csc_matrix, col_idx: int) -> tuple[bool, np.ndarray]:
    """Compute centered, unit-L2-normalized average-tie Spearman rank vector for CSC column `col_idx`."""
    n = X_csc.shape[0]
    start, end = int(X_csc.indptr[col_idx]), int(X_csc.indptr[col_idx + 1])
    nnz = end - start
    if nnz == 0:
        return False, np.zeros(n, dtype=np.float64)

    idx = X_csc.indices[start:end]
    vals = X_csc.data[start:end]

    if not X_csc.has_canonical_format and nnz > 1 and not np.all(idx[:-1] < idx[1:]):
        order = np.argsort(idx, kind="mergesort")
        idx_s = idx[order]
        vals_s = vals[order]
        if np.any(idx_s[:-1] == idx_s[1:]):
            uniq_idx, inv = np.unique(idx_s, return_inverse=True)
            sum_vals = np.zeros(len(uniq_idx), dtype=np.float64)
            np.add.at(sum_vals, inv, vals_s)
            idx = uniq_idx
            vals = sum_vals
        else:
            idx = idx_s
            vals = vals_s
        nnz = len(idx)

    if nnz == n:
        return _unit_rank_dense(vals)

    n_implicit_zero = n - nnz

    if np.all(vals > 0.0):
        zero_rank = 0.5 * (n_implicit_zero + 1)
        ranks = np.full(n, zero_rank, dtype=np.float64)
        ranks[idx] = n_implicit_zero + stats.rankdata(vals, method="average")
    else:
        neg_mask = vals < 0.0
        zero_mask = vals == 0.0
        pos_mask = vals > 0.0
        n_neg = int(np.sum(neg_mask))
        n_zero = n_implicit_zero + int(np.sum(zero_mask))
        zero_rank = n_neg + 0.5 * (n_zero + 1)
        ranks = np.full(n, zero_rank, dtype=np.float64)
        if n_neg > 0:
            ranks[idx[neg_mask]] = stats.rankdata(vals[neg_mask], method="average")
        if np.any(pos_mask):
            ranks[idx[pos_mask]] = n_neg + n_zero + stats.rankdata(vals[pos_mask], method="average")

    centered = ranks - 0.5 * (n + 1)
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12 or not np.isfinite(norm):
        return False, np.zeros(n, dtype=np.float64)
    return True, centered / norm


def compute_unit_rank_matrix(
    mat: np.ndarray | sparse.csr_matrix | sparse.csc_matrix,
    col_indices: Sequence[int] | np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(n_cells, len(col_indices))` unit-normalized rank (or centered Pearson) vectors.

    Returns `(U, valid_mask)` where `U.T @ V` yields exact correlations.
    """
    cols = np.asarray(col_indices, dtype=np.int64)
    n_cells = mat.shape[0]
    n_cols = len(cols)
    U = np.zeros((n_cells, n_cols), dtype=np.float64)
    valid = np.zeros(n_cols, dtype=bool)
    if n_cols == 0:
        return U, valid

    is_sp = sparse.issparse(mat)
    if is_sp:
        mat_csc = mat.tocsc()
        if not mat_csc.has_canonical_format:
            mat_csc = mat_csc.copy()
            mat_csc.sum_duplicates()
            mat_csc.sort_indices()
    else:
        mat_csc = None

    for k, j in enumerate(cols):
        if method == "spearman":
            if is_sp:
                ok, vec = _unit_rank_sparse_col(mat_csc, int(j))  # type: ignore[arg-type]
            else:
                ok, vec = _unit_rank_dense(mat[:, int(j)])
        else:
            if is_sp:
                v = mat_csc[:, int(j)].toarray().ravel()  # type: ignore[index]
            else:
                v = np.asarray(mat[:, int(j)], dtype=np.float64).ravel()
            centered = v - float(np.mean(v))
            norm = float(np.linalg.norm(centered))
            if norm <= 1e-12 or not np.isfinite(norm):
                ok, vec = False, np.zeros(n_cells, dtype=np.float64)
            else:
                ok, vec = True, centered / norm
        valid[k] = ok
        if ok:
            U[:, k] = vec

    return U, valid


def run_gene_peak_corr(
    rna: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    atac: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    tss: pd.DataFrame | str | Path,
    *,
    peaks: pd.DataFrame | Sequence[str] | None = None,
    peak_gc: np.ndarray | pd.Series | Sequence[float] | str | None = None,
    window_pad: int = 50_000,
    n_bg: int = 100,
    bg_peaks: np.ndarray | None = None,
    bg_method: Literal["chromvar_kernel", "knn_mahalanobis"] = "chromvar_kernel",
    alternative: Literal["greater", "two-sided", "less"] = "greater",
    normalize_atac: bool = True,
    scale_factor: float = 1e4,
    normalize_rna: bool = False,
    rna_layer: str | None = None,
    atac_layer: str | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
    keep_multi_mapping_peaks: bool | None = None,
    p_cut: float | None = None,
    fdr_cut: float | None = None,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> GenePeakCorrResult:
    """Test cis peak-to-gene correlations against GC/accessibility-matched background peaks.

    Maps ATAC peaks within `[tss - window_pad, tss + window_pad]` on each gene's chromosome,
    computes exact tie-averaged Spearman correlations across aligned cells, evaluates `n_bg`
    matched background correlations per cis pair, and computes empirical null Z-scores,
    p-values, and `NaN`-aware Benjamini–Hochberg FDR.
    """
    validate_settings(settings)
    eff_keep_multi = (
        (settings != "R_default")
        if keep_multi_mapping_peaks is None
        else bool(keep_multi_mapping_peaks)
    )
    if window_pad <= 0:
        raise ValueError(f"`window_pad` must be > 0, got {window_pad}.")
    if alternative not in ("greater", "two-sided", "less"):
        raise ValueError(f"Invalid `alternative` {alternative!r}; expected 'greater', 'two-sided', or 'less'.")
    if bg_peaks is None and n_bg < 2:
        raise ValueError(f"`n_bg` must be >= 2 to estimate background SD, got {n_bg}.")

    X_rna, rna_cells, gene_names, rna_has_cells, _ = extract_matrix(
        rna,
        layer=rna_layer,
        orientation=orientation,
        feature_prefix="gene",
    )
    X_atac, atac_cells, peak_names, atac_has_cells, _ = extract_matrix(
        atac,
        layer=atac_layer,
        orientation=orientation,
        feature_prefix="peak",
    )

    X_rna, X_atac, shared_cells = align_cells(
        X_rna,
        rna_cells,
        rna_has_cells,
        X_atac,
        atac_cells,
        atac_has_cells,
        min_cells=3,
    )
    n_cells = len(shared_cells)
    n_peaks = X_atac.shape[1]

    tss_df = coerce_tss_table(tss)
    peak_df = coerce_peaks(peaks, peak_names, atac_data=atac)

    if normalize_atac:
        X_atac = _normalize_cells_matrix(X_atac, scale_factor=scale_factor, log1p=False)
    if normalize_rna:
        X_rna = _normalize_cells_matrix(X_rna, scale_factor=scale_factor, log1p=True)

    if bg_peaks is None:
        if peak_gc is None:
            if isinstance(atac, ad.AnnData) and "gc_content" in atac.var.columns:
                peak_gc = "gc_content"
            elif isinstance(atac, ad.AnnData) and "GC" in atac.var.columns:
                peak_gc = "GC"
            else:
                raise ValueError("`peak_gc` is required when `bg_peaks` is not provided.")
        if isinstance(peak_gc, str):
            if not isinstance(atac, ad.AnnData) or peak_gc not in atac.var.columns:
                raise ValueError(f"String `peak_gc` ({peak_gc!r}) not found in `atac.var`.")
            gc_arr = pd.to_numeric(atac.var[peak_gc], errors="raise").to_numpy(dtype=np.float64).ravel()
        else:
            gc_arr = np.asarray(peak_gc, dtype=np.float64).ravel()

        bg_mat = get_background_peaks(
            X_atac,
            gc_arr,
            n_bg=n_bg,
            method=bg_method,
            exclude_self=True,
            seed=seed,
        )
    else:
        bg_mat = np.asarray(bg_peaks, dtype=np.int64)
        if bg_mat.ndim != 2 or bg_mat.shape[0] != n_peaks or bg_mat.shape[1] < 2:
            raise ValueError(
                f"`bg_peaks` must have shape ({n_peaks}, n_bg >= 2), got {bg_mat.shape}."
            )
        if np.any(bg_mat < 0) or np.any(bg_mat >= n_peaks):
            raise ValueError("`bg_peaks` contains out-of-range peak indices.")

    n_bg_eff = int(bg_mat.shape[1])

    # Build chromosome-partitioned sorted peak lookup for O(log P) cis-window intersection
    gene_to_idx: dict[str, int] = {}
    for idx, g in enumerate(gene_names):
        if g not in gene_to_idx:
            gene_to_idx[g] = idx

    peak_chroms = peak_df["chrom"].astype(str).to_numpy()
    peak_starts = peak_df["start"].to_numpy(dtype=np.int64)
    peak_ends = peak_df["end"].to_numpy(dtype=np.int64)
    peak_mids = peak_df["mid"].to_numpy(dtype=np.int64)
    peak_ids = peak_df["peak"].astype(str).to_numpy()

    chrom_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for ch in np.unique(peak_chroms):
        idx_ch = np.flatnonzero(peak_chroms == ch)
        order = idx_ch[np.argsort(peak_starts[idx_ch], kind="mergesort")]
        s_ch = peak_starts[order]
        e_ch = peak_ends[order]
        max_len = int(np.max(e_ch - s_ch))
        chrom_index[str(ch)] = (order, s_ch, e_ch, max_len)

    # Enumerate candidate cis (gene, peak) pairs
    cand_g_idx: list[int] = []
    cand_p_idx: list[int] = []
    cand_tss: list[int] = []
    cand_dist: list[int] = []

    for row in tss_df.itertuples(index=False):
        g_name = str(row.gene_name)
        g_idx = gene_to_idx.get(g_name)
        if g_idx is None:
            continue
        ch = str(row.chrom)
        ch_entry = chrom_index.get(ch)
        if ch_entry is None:
            continue
        order, s_ch, e_ch, max_len = ch_entry
        tss_pos = int(row.tss)
        strand = str(row.strand)
        w_lo = tss_pos - window_pad
        w_hi = tss_pos + window_pad
        lo = int(np.searchsorted(s_ch, w_lo - max_len, side="left"))
        hi = int(np.searchsorted(s_ch, w_hi, side="right"))
        if hi <= lo:
            continue
        sub_mask = e_ch[lo:hi] >= w_lo
        hits = order[lo:hi][sub_mask]
        if len(hits) == 0:
            continue
        sign = -1 if strand == "-" else 1
        for p_idx in hits:
            cand_g_idx.append(g_idx)
            cand_p_idx.append(int(p_idx))
            cand_tss.append(tss_pos)
            cand_dist.append(int((peak_mids[p_idx] - tss_pos) * sign))

    if len(cand_g_idx) == 0:
        empty_str = np.empty(0, dtype=object)
        empty_int = np.empty(0, dtype=np.int64)
        empty_flt = np.empty(0, dtype=np.float64)
        return GenePeakCorrResult(
            gene=empty_str,
            peak=empty_str,
            chrom=empty_str,
            tss=empty_int,
            peak_start=empty_int,
            peak_end=empty_int,
            tss_distance=empty_int,
            r_obs=empty_flt,
            bg_mean=empty_flt,
            bg_std=empty_flt,
            z_score=empty_flt,
            p_value=empty_flt,
            fdr=empty_flt,
            n_cells=n_cells,
            n_bg=n_bg_eff,
            window_pad=window_pad,
            alternative=alternative,
        )

    cand_g_arr = np.asarray(cand_g_idx, dtype=np.int64)
    cand_p_arr = np.asarray(cand_p_idx, dtype=np.int64)
    cand_tss_arr = np.asarray(cand_tss, dtype=np.int64)
    cand_dist_arr = np.asarray(cand_dist, dtype=np.int64)

    # Pre-rank only unique active genes and unique active (cis + background) peaks once
    unique_genes = np.unique(cand_g_arr)
    g_local_map = {int(g): i for i, g in enumerate(unique_genes)}
    U_gene, valid_gene = compute_unit_rank_matrix(X_rna, unique_genes, method="spearman")

    unique_cis_peaks = np.unique(cand_p_arr)
    needed_peaks = np.unique(np.concatenate([unique_cis_peaks, bg_mat[unique_cis_peaks].ravel()]))
    p_local_map = {int(p): i for i, p in enumerate(needed_peaks)}
    U_peak, valid_peak = compute_unit_rank_matrix(X_atac, needed_peaks, method="spearman")

    m_pairs = len(cand_g_arr)
    r_obs = np.full(m_pairs, np.nan, dtype=np.float64)
    bg_mean = np.full(m_pairs, np.nan, dtype=np.float64)
    bg_std = np.full(m_pairs, np.nan, dtype=np.float64)
    z_score = np.full(m_pairs, np.nan, dtype=np.float64)
    p_value = np.full(m_pairs, np.nan, dtype=np.float64)

    # Group cis pairs by gene so each gene computes one vectorized dot product across its cis & bg peaks
    order_by_gene = np.argsort(cand_g_arr, kind="mergesort")
    sorted_g = cand_g_arr[order_by_gene]
    split_pts = np.flatnonzero(np.diff(sorted_g)) + 1
    groups = np.split(order_by_gene, split_pts)

    for grp in groups:
        g_idx = int(cand_g_arr[grp[0]])
        gl = g_local_map[g_idx]
        if not valid_gene[gl]:
            continue
        u_g = U_gene[:, gl]

        p_globals = cand_p_arr[grp]
        p_locals = np.array([p_local_map[int(p)] for p in p_globals], dtype=np.int64)
        bg_globals = bg_mat[p_globals]  # (len(grp), n_bg_eff)
        bg_locals = np.vectorize(p_local_map.__getitem__, otypes=[np.int64])(bg_globals)

        # Compute observed correlations
        r_g_obs = u_g @ U_peak[:, p_locals]
        r_g_bg = u_g @ U_peak[:, bg_locals.ravel()]
        r_g_bg = r_g_bg.reshape(len(grp), n_bg_eff)

        p_ok = valid_peak[p_locals]
        bg_ok = valid_peak[bg_locals]

        for idx_in_grp, pair_i in enumerate(grp):
            if not p_ok[idx_in_grp]:
                continue
            obs_val = float(np.clip(r_g_obs[idx_in_grp], -1.0, 1.0))
            r_obs[pair_i] = obs_val

            bg_vals = r_g_bg[idx_in_grp]
            bg_mask = bg_ok[idx_in_grp] & np.isfinite(bg_vals)
            if int(np.sum(bg_mask)) < 2:
                continue
            bg_valid = np.clip(bg_vals[bg_mask], -1.0, 1.0)
            mu = float(np.mean(bg_valid))
            sd = float(np.std(bg_valid, ddof=1))
            bg_mean[pair_i] = mu
            bg_std[pair_i] = sd

            if sd <= 1e-12:
                if abs(obs_val - mu) <= 1e-12:
                    z = 0.0
                    z_score[pair_i] = z
                    p_value[pair_i] = 1.0 if alternative == "two-sided" else 0.5
                else:
                    # Undefined non-zero discrepancy with zero background variance
                    z_score[pair_i] = np.nan
                    p_value[pair_i] = np.nan
            else:
                z = (obs_val - mu) / sd
                z_score[pair_i] = z
                if alternative == "greater":
                    p = float(stats.norm.sf(z))
                elif alternative == "less":
                    p = float(stats.norm.cdf(z))
                else:
                    p = float(2.0 * stats.norm.sf(abs(z)))
                p_value[pair_i] = float(np.clip(p, 0.0, 1.0))

    gene_arr = np.asarray([gene_names[g] for g in cand_g_arr], dtype=object)
    peak_arr = peak_ids[cand_p_arr].astype(object)
    chrom_arr = peak_chroms[cand_p_arr].astype(object)
    pstart_arr = peak_starts[cand_p_arr]
    pend_arr = peak_ends[cand_p_arr]

    # Deduplicate multi-TSS hits for identical (gene, peak) deterministically before BH-FDR
    df_all = pd.DataFrame(
        {
            "gene": gene_arr,
            "peak": peak_arr,
            "chrom": chrom_arr,
            "tss": cand_tss_arr,
            "peak_start": pstart_arr,
            "peak_end": pend_arr,
            "tss_distance": cand_dist_arr,
            "r_obs": r_obs,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "z_score": z_score,
            "p_value": p_value,
            "_abs_z": np.abs(z_score),
            "_abs_dist": np.abs(cand_dist_arr),
            "_orig": np.arange(m_pairs, dtype=np.int64),
        }
    )
    if df_all.duplicated(subset=["gene", "peak"]).any():
        df_all = (
            df_all.sort_values(
                by=["p_value", "_abs_z", "_abs_dist", "_orig"],
                ascending=[True, False, True, True],
                kind="mergesort",
                na_position="last",
            )
            .drop_duplicates(subset=["gene", "peak"], keep="first")
            .sort_values(by="_orig", kind="mergesort")
            .reset_index(drop=True)
        )

    if not eff_keep_multi and len(df_all) > 0 and df_all.duplicated(subset=["peak"]).any():
        # Match buenrostrolab/FigR cisCor.R keepMultiMappingPeaks=FALSE: keep max r_obs gene per peak
        df_all = (
            df_all.sort_values(
                by=["r_obs", "p_value", "_abs_dist", "_orig"],
                ascending=[False, True, True, True],
                kind="mergesort",
                na_position="last",
            )
            .drop_duplicates(subset=["peak"], keep="first")
            .sort_values(by="_orig", kind="mergesort")
            .reset_index(drop=True)
        )

    fdr_arr = bh_fdr(df_all["p_value"].to_numpy(dtype=np.float64))
    df_all["fdr"] = fdr_arr

    # Optional post-FDR filtering
    if p_cut is not None or fdr_cut is not None:
        keep = np.ones(len(df_all), dtype=bool)
        if p_cut is not None:
            pv = df_all["p_value"].to_numpy(dtype=np.float64)
            keep &= np.isfinite(pv) & (pv <= p_cut)
        if fdr_cut is not None:
            fv = df_all["fdr"].to_numpy(dtype=np.float64)
            keep &= np.isfinite(fv) & (fv <= fdr_cut)
        df_all = df_all.loc[keep].reset_index(drop=True)

    return GenePeakCorrResult(
        gene=df_all["gene"].to_numpy(dtype=object),
        peak=df_all["peak"].to_numpy(dtype=object),
        chrom=df_all["chrom"].to_numpy(dtype=object),
        tss=df_all["tss"].to_numpy(dtype=np.int64),
        peak_start=df_all["peak_start"].to_numpy(dtype=np.int64),
        peak_end=df_all["peak_end"].to_numpy(dtype=np.int64),
        tss_distance=df_all["tss_distance"].to_numpy(dtype=np.int64),
        r_obs=df_all["r_obs"].to_numpy(dtype=np.float64),
        bg_mean=df_all["bg_mean"].to_numpy(dtype=np.float64),
        bg_std=df_all["bg_std"].to_numpy(dtype=np.float64),
        z_score=df_all["z_score"].to_numpy(dtype=np.float64),
        p_value=df_all["p_value"].to_numpy(dtype=np.float64),
        fdr=df_all["fdr"].to_numpy(dtype=np.float64),
        n_cells=n_cells,
        n_bg=n_bg_eff,
        window_pad=window_pad,
        alternative=alternative,
    )
