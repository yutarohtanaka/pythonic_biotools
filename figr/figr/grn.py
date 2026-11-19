"""TF-motif enrichment x TF-RNA correlation gene regulatory network inference (`run_figr_grn`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .background import get_background_peaks
from .corr import GenePeakCorrResult, compute_unit_rank_matrix
from .dorc import DORCCallResult, DORCScoresResult
from .io import SettingsMode, align_cells, bh_fdr, extract_matrix, validate_settings


@dataclass(frozen=True)
class FigRGRNResult:
    """Typed result container for `run_figr_grn`."""

    dorc_gene: np.ndarray
    motif: np.ndarray
    tf_gene: np.ndarray
    n_dorc_peaks: np.ndarray
    motif_obs_count: np.ndarray
    motif_bg_mean: np.ndarray
    motif_enrichment_z: np.ndarray
    p_motif: np.ndarray
    fdr_motif: np.ndarray
    r_tf_dorc: np.ndarray
    p_corr: np.ndarray
    fdr_corr: np.ndarray
    dorc_reg_score: np.ndarray
    role: np.ndarray
    n_cells: int
    enrichment_method: str
    corr_method: str
    score_mode: str

    def to_dataframe(self, sort_by_abs_score: bool = True) -> pd.DataFrame:
        """Return tidy `(TF, DORC)` regulatory edges as a pandas DataFrame."""
        cols = [
            "dorc_gene",
            "motif",
            "tf_gene",
            "n_dorc_peaks",
            "motif_obs_count",
            "motif_bg_mean",
            "motif_enrichment_z",
            "p_motif",
            "fdr_motif",
            "r_tf_dorc",
            "p_corr",
            "fdr_corr",
            "dorc_reg_score",
            "role",
        ]
        if len(self.dorc_gene) == 0:
            return pd.DataFrame(
                {
                    "dorc_gene": pd.Series(dtype=object),
                    "motif": pd.Series(dtype=object),
                    "tf_gene": pd.Series(dtype=object),
                    "n_dorc_peaks": pd.Series(dtype=np.int64),
                    "motif_obs_count": pd.Series(dtype=np.float64),
                    "motif_bg_mean": pd.Series(dtype=np.float64),
                    "motif_enrichment_z": pd.Series(dtype=np.float64),
                    "p_motif": pd.Series(dtype=np.float64),
                    "fdr_motif": pd.Series(dtype=np.float64),
                    "r_tf_dorc": pd.Series(dtype=np.float64),
                    "p_corr": pd.Series(dtype=np.float64),
                    "fdr_corr": pd.Series(dtype=np.float64),
                    "dorc_reg_score": pd.Series(dtype=np.float64),
                    "role": pd.Series(dtype=object),
                },
                columns=cols,
            )

        df = pd.DataFrame(
            {
                "dorc_gene": self.dorc_gene,
                "motif": self.motif,
                "tf_gene": self.tf_gene,
                "n_dorc_peaks": self.n_dorc_peaks,
                "motif_obs_count": self.motif_obs_count,
                "motif_bg_mean": self.motif_bg_mean,
                "motif_enrichment_z": self.motif_enrichment_z,
                "p_motif": self.p_motif,
                "fdr_motif": self.fdr_motif,
                "r_tf_dorc": self.r_tf_dorc,
                "p_corr": self.p_corr,
                "fdr_corr": self.fdr_corr,
                "dorc_reg_score": self.dorc_reg_score,
                "role": self.role,
            }
        )
        if sort_by_abs_score:
            df["_abs_score"] = np.abs(df["dorc_reg_score"].to_numpy(dtype=np.float64))
            df = (
                df.sort_values(
                    by=["_abs_score", "dorc_gene", "tf_gene", "motif"],
                    ascending=[False, True, True, True],
                    kind="mergesort",
                    na_position="last",
                )
                .drop(columns=["_abs_score"])
                .reset_index(drop=True)
            )
        return df

    def significant(
        self,
        min_abs_score: float = 1.0,
        fdr_motif_cut: float | None = None,
        fdr_corr_cut: float | None = None,
    ) -> pd.DataFrame:
        """Return regulatory interactions passing regulation score and optional FDR thresholds."""
        df = self.to_dataframe(sort_by_abs_score=True)
        if df.empty:
            return df
        s_arr = df["dorc_reg_score"].to_numpy(dtype=np.float64)
        mask = np.isfinite(s_arr) & (np.abs(s_arr) >= min_abs_score)
        if fdr_motif_cut is not None:
            fm = df["fdr_motif"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(fm) & (fm <= fdr_motif_cut)
        if fdr_corr_cut is not None:
            fc = df["fdr_corr"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(fc) & (fc <= fdr_corr_cut)
        return df.loc[mask].reset_index(drop=True)

    def drivers(self, dorc_gene: str, top_n: int = 10) -> pd.DataFrame:
        """Return top `top_n` candidate TF drivers for a specific `dorc_gene` ranked by `|dorc_reg_score|`."""
        df = self.to_dataframe(sort_by_abs_score=True)
        sub = df.loc[df["dorc_gene"] == str(dorc_gene)].reset_index(drop=True)
        return sub.head(top_n)

    def to_adjacency(
        self,
        value: Literal["dorc_reg_score", "r_tf_dorc", "motif_enrichment_z"] = "dorc_reg_score",
    ) -> pd.DataFrame:
        """Pivot GRN results into a `(tf_gene x dorc_gene)` adjacency matrix."""
        if value not in ("dorc_reg_score", "r_tf_dorc", "motif_enrichment_z"):
            raise ValueError(f"Unsupported `value` column {value!r}.")
        df = self.to_dataframe(sort_by_abs_score=True)
        if df.empty:
            return pd.DataFrame()
        dedup = df.drop_duplicates(subset=["tf_gene", "dorc_gene"], keep="first")
        return dedup.pivot(index="tf_gene", columns="dorc_gene", values=value).fillna(0.0)

    def summary(self) -> str:
        """Return a human-readable summary of inferred TF-DORC regulatory interactions."""
        n_edges = len(self.dorc_gene)
        valid = np.isfinite(self.dorc_reg_score)
        n_act = int(np.sum(self.role == "activator"))
        n_rep = int(np.sum(self.role == "repressor"))
        n_sig = int(np.sum(np.abs(self.dorc_reg_score[valid]) >= 1.0)) if np.any(valid) else 0
        return (
            f"FigRGRNResult(n_edges={n_edges}, n_cells={self.n_cells}, "
            f"enrichment={self.enrichment_method!r}, corr={self.corr_method!r}, "
            f"score_mode={self.score_mode!r}, activators={n_act}, repressors={n_rep}, "
            f"abs_score_ge_1={n_sig})"
        )


def _coerce_peak_motifs(
    peak_motifs: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    peak_names: Sequence[str] | None,
    motif_names: Sequence[str] | None,
    motif_threshold: float,
) -> tuple[np.ndarray | sparse.csr_matrix, list[str], list[str], bool]:
    """Coerce `(peaks x motifs)` matrix to binary float64 CSR or ndarray."""
    if isinstance(peak_motifs, ad.AnnData):
        raw = peak_motifs.X
        p_names = [str(p) for p in peak_motifs.obs_names]
        m_names = [str(m) for m in peak_motifs.var_names]
        has_explicit_peaks = True
    elif isinstance(peak_motifs, pd.DataFrame):
        if len(peak_motifs.dtypes) > 0 and all(isinstance(dt, pd.SparseDtype) for dt in peak_motifs.dtypes):
            raw = peak_motifs.sparse.to_coo().tocsr()
        else:
            raw = peak_motifs.to_numpy(dtype=np.float64)
        p_names = [str(p) for p in peak_motifs.index]
        m_names = [str(m) for m in peak_motifs.columns]
        has_explicit_peaks = True
    elif sparse.issparse(peak_motifs):
        raw = peak_motifs.tocsr()
        if raw.ndim != 2:
            raise ValueError(f"`peak_motifs` must be 2D, got {raw.shape}.")
        p_names = [f"peak_{i}" for i in range(raw.shape[0])]
        m_names = [f"motif_{j}" for j in range(raw.shape[1])]
        has_explicit_peaks = False
    else:
        raw = np.asarray(peak_motifs, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError(f"`peak_motifs` must be 2D, got {raw.shape}.")
        p_names = [f"peak_{i}" for i in range(raw.shape[0])]
        m_names = [f"motif_{j}" for j in range(raw.shape[1])]
        has_explicit_peaks = False

    n_peaks, n_motifs = raw.shape
    if n_peaks == 0 or n_motifs == 0:
        raise ValueError(f"`peak_motifs` cannot be empty; got shape {raw.shape}.")

    if not has_explicit_peaks and peak_names is not None:
        p_names = [str(p) for p in peak_names]
        if len(p_names) != n_peaks:
            raise ValueError(f"`peak_names` length ({len(p_names)}) must match `peak_motifs` rows ({n_peaks}).")
        has_explicit_peaks = True
    if motif_names is not None:
        m_names = [str(m) for m in motif_names]
        if len(m_names) != n_motifs:
            raise ValueError(f"`motif_names` length ({len(m_names)}) must match `peak_motifs` cols ({n_motifs}).")

    if sparse.issparse(raw):
        if motif_threshold >= 0.0:
            sp = raw.tocsr().astype(np.float64)
            mask = sp.data > motif_threshold
            # Keep only entries strictly greater than threshold as 1.0
            new_sp = sp.copy()
            new_sp.data = mask.astype(np.float64)
            new_sp.eliminate_zeros()
            return new_sp, p_names, m_names, has_explicit_peaks
        dense_bin = (raw.toarray() > motif_threshold).astype(np.float64)
        return dense_bin, p_names, m_names, has_explicit_peaks

    bin_mat = (np.asarray(raw, dtype=np.float64) > motif_threshold).astype(np.float64)
    return bin_mat, p_names, m_names, has_explicit_peaks


def _resolve_motif_to_tf(
    motif_names: list[str],
    motif_to_tf: dict[str, str] | pd.DataFrame | None,
) -> list[str]:
    if motif_to_tf is None:
        return list(motif_names)
    if isinstance(motif_to_tf, dict):
        return [str(motif_to_tf.get(m, m)) for m in motif_names]
    if isinstance(motif_to_tf, pd.DataFrame):
        if "motif" not in motif_to_tf.columns:
            raise ValueError("`motif_to_tf` DataFrame must contain a 'motif' column.")
        tf_col = next(
            (c for c in ("tf_gene", "tf", "gene", "gene_name", "symbol") if c in motif_to_tf.columns),
            None,
        )
        if tf_col is None:
            raise ValueError("`motif_to_tf` DataFrame must contain a TF column ('tf_gene', 'tf', 'gene', etc.).")
        mapping = dict(zip(motif_to_tf["motif"].astype(str), motif_to_tf[tf_col].astype(str)))
        return [mapping.get(m, m) for m in motif_names]
    raise TypeError("`motif_to_tf` must be a dict, pd.DataFrame, or None.")


def run_figr_grn(
    dorc_scores: DORCScoresResult | ad.AnnData | pd.DataFrame | np.ndarray,
    rna: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    dorc_links: DORCCallResult | GenePeakCorrResult | pd.DataFrame,
    peak_motifs: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    *,
    atac: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray | None = None,
    peak_gc: np.ndarray | pd.Series | Sequence[float] | str | None = None,
    background_peaks: np.ndarray | None = None,
    motif_to_tf: dict[str, str] | pd.DataFrame | None = None,
    peak_names: Sequence[str] | None = None,
    motif_names: Sequence[str] | None = None,
    motif_threshold: float = 0.0,
    dorc_k_pool: int | None = None,
    n_bg: int = 50,
    bg_method: Literal["chromvar_kernel", "knn_mahalanobis"] = "chromvar_kernel",
    enrichment_method: Literal["background_z", "hypergeometric"] = "background_z",
    corr_method: Literal["spearman", "pearson"] = "spearman",
    corr_p_method: Literal["ttest", "z_across_tfs"] | None = None,
    score_mode: Literal["product", "kartha_union"] | None = None,
    clamp_negative_enrichment: bool = True,
    min_p: float = 1e-300,
    rna_layer: str | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
    seed: int = 42,
    settings: SettingsMode = "default",
) -> FigRGRNResult:
    """Infer TF-DORC regulatory networks by combining TF motif enrichment with TF-RNA correlation."""
    validate_settings(settings)
    eff_dorc_k_pool = (
        (30 if settings == "R_default" else 0)
        if dorc_k_pool is None
        else int(dorc_k_pool)
    )
    eff_corr_p_method: Literal["ttest", "z_across_tfs"] = (
        ("z_across_tfs" if settings == "R_default" else "ttest")
        if corr_p_method is None
        else corr_p_method
    )
    eff_score_mode: Literal["product", "kartha_union"] = (
        ("kartha_union" if settings == "R_default" else "product")
        if score_mode is None
        else score_mode
    )

    if enrichment_method not in ("background_z", "hypergeometric"):
        raise ValueError(f"Invalid `enrichment_method` {enrichment_method!r}.")
    if corr_method not in ("spearman", "pearson"):
        raise ValueError(f"Invalid `corr_method` {corr_method!r}.")
    if eff_corr_p_method not in ("ttest", "z_across_tfs"):
        raise ValueError(f"Invalid `corr_p_method` {eff_corr_p_method!r}.")
    if eff_score_mode not in ("product", "kartha_union"):
        raise ValueError(f"Invalid `score_mode` {eff_score_mode!r}.")
    if eff_dorc_k_pool < 0:
        raise ValueError(f"`dorc_k_pool` must be >= 0, got {eff_dorc_k_pool}.")
    if min_p <= 0.0 or min_p > 1.0:
        raise ValueError(f"`min_p` must be in (0, 1], got {min_p}.")

    # 1. Extract DORC scores & RNA matrices and align cells
    if isinstance(dorc_scores, DORCScoresResult):
        mat_dorc_raw = dorc_scores.smoothed_scores if dorc_scores.smoothed_scores is not None else dorc_scores.raw_scores
        dorc_cells = list(dorc_scores.cell_names)
        dorc_genes = list(dorc_scores.dorc_genes)
        dorc_has_cells = True
        if len(dorc_genes) == 0:
            X_rna_tmp, rna_cells_tmp, _, rna_has_cells_tmp, _ = extract_matrix(
                rna, layer=rna_layer, orientation=orientation, feature_prefix="gene"
            )
            _, _, shared_tmp = align_cells(
                mat_dorc_raw, dorc_cells, dorc_has_cells, X_rna_tmp, rna_cells_tmp, rna_has_cells_tmp, min_cells=3
            )
            return FigRGRNResult(
                dorc_gene=np.empty(0, dtype=object),
                motif=np.empty(0, dtype=object),
                tf_gene=np.empty(0, dtype=object),
                n_dorc_peaks=np.empty(0, dtype=np.int64),
                motif_obs_count=np.empty(0, dtype=np.float64),
                motif_bg_mean=np.empty(0, dtype=np.float64),
                motif_enrichment_z=np.empty(0, dtype=np.float64),
                p_motif=np.empty(0, dtype=np.float64),
                fdr_motif=np.empty(0, dtype=np.float64),
                r_tf_dorc=np.empty(0, dtype=np.float64),
                p_corr=np.empty(0, dtype=np.float64),
                fdr_corr=np.empty(0, dtype=np.float64),
                dorc_reg_score=np.empty(0, dtype=np.float64),
                role=np.empty(0, dtype=object),
                n_cells=len(shared_tmp),
                enrichment_method=enrichment_method,
                corr_method=corr_method,
                score_mode=eff_score_mode,
            )
    else:
        mat_dorc_raw, dorc_cells, dorc_genes, dorc_has_cells, _ = extract_matrix(
            dorc_scores, orientation=orientation, feature_prefix="dorc"
        )

    X_rna, rna_cells, rna_gene_names, rna_has_cells, _ = extract_matrix(
        rna, layer=rna_layer, orientation=orientation, feature_prefix="gene"
    )
    mat_dorc, X_rna_aligned, shared_cells = align_cells(
        mat_dorc_raw, dorc_cells, dorc_has_cells, X_rna, rna_cells, rna_has_cells, min_cells=3
    )
    if sparse.issparse(mat_dorc):
        mat_dorc = mat_dorc.toarray()
    else:
        mat_dorc = np.asarray(mat_dorc, dtype=np.float64)
    n_cells = len(shared_cells)

    # 2. Extract and binarize peak-motif matrix, aligning rows to `atac` / `peak_names` order
    atac_peak_names: list[str] | None = None
    if atac is not None:
        if isinstance(atac, ad.AnnData):
            atac_peak_names = atac.var_names.astype(str).tolist()
        elif isinstance(atac, pd.DataFrame):
            atac_peak_names = (
                atac.columns.astype(str).tolist()
                if orientation == "cells_by_features"
                else atac.index.astype(str).tolist()
            )

    if peak_names is not None:
        inferred_peak_names: Sequence[str] | None = peak_names
    elif not isinstance(peak_motifs, (ad.AnnData, pd.DataFrame)) and atac_peak_names is not None:
        inferred_peak_names = atac_peak_names
    else:
        inferred_peak_names = None

    M_bin, resolved_peak_names, resolved_motif_names, pm_has_explicit_peaks = _coerce_peak_motifs(
        peak_motifs, inferred_peak_names, motif_names, motif_threshold
    )

    target_peak_names: list[str] | None = None
    target_source = ""
    if atac_peak_names is not None:
        target_peak_names = atac_peak_names
        target_source = "atac"
    elif peak_names is not None and isinstance(peak_motifs, (ad.AnnData, pd.DataFrame)):
        target_peak_names = [str(p) for p in peak_names]
        target_source = "peak_names"

    if target_peak_names is not None:
        if len(set(target_peak_names)) != len(target_peak_names):
            raise ValueError(f"Duplicate peak identifiers detected in `{target_source}`.")
        if pm_has_explicit_peaks:
            if len(set(resolved_peak_names)) != len(resolved_peak_names):
                raise ValueError("Duplicate peak identifiers detected in `peak_motifs`.")
            pm_idx_map = {p: i for i, p in enumerate(resolved_peak_names)}
            missing_peaks = [p for p in target_peak_names if p not in pm_idx_map]
            if missing_peaks:
                raise ValueError(
                    f"`peak_motifs` is missing {len(missing_peaks)} peak(s) required by `{target_source}` "
                    f"(e.g., {missing_peaks[:5]})."
                )
            if resolved_peak_names != target_peak_names:
                row_order = np.asarray([pm_idx_map[p] for p in target_peak_names], dtype=np.int64)
                M_bin = M_bin[row_order, :]
                resolved_peak_names = list(target_peak_names)

    n_peaks = M_bin.shape[0]
    peak_to_idx = {p: i for i, p in enumerate(resolved_peak_names)}

    # 3. Map motifs to TF genes present in RNA
    tf_for_motif = _resolve_motif_to_tf(resolved_motif_names, motif_to_tf)
    rna_gene_to_idx: dict[str, int] = {}
    for idx, g in enumerate(rna_gene_names):
        if g not in rna_gene_to_idx:
            rna_gene_to_idx[g] = idx

    active_motif_indices: list[int] = []
    active_motif_names: list[str] = []
    active_tf_names: list[str] = []
    active_rna_indices: list[int] = []
    for m_idx, (m_name, tf_name) in enumerate(zip(resolved_motif_names, tf_for_motif)):
        r_idx = rna_gene_to_idx.get(tf_name)
        if r_idx is not None:
            active_motif_indices.append(m_idx)
            active_motif_names.append(m_name)
            active_tf_names.append(tf_name)
            active_rna_indices.append(r_idx)

    if len(active_motif_indices) == 0:
        raise ValueError("Zero overlap between motif TF gene symbols and `rna` gene names.")

    # Slice M_bin to active motifs only
    M_active = M_bin[:, active_motif_indices]
    if sparse.issparse(M_active):
        M_active = M_active.tocsr()

    # 4. Resolve DORC peak sets (with optional k-NN DORC pooling)
    if isinstance(dorc_links, DORCCallResult):
        links_df = dorc_links.filtered_links
    elif isinstance(dorc_links, GenePeakCorrResult):
        links_df = dorc_links.to_dataframe()
    elif isinstance(dorc_links, pd.DataFrame):
        links_df = dorc_links
    else:
        raise TypeError("`dorc_links` must be a `DORCCallResult`, `GenePeakCorrResult`, or `pd.DataFrame`.")

    if not {"gene", "peak"}.issubset(links_df.columns):
        raise ValueError("`dorc_links` must contain 'gene' and 'peak' columns.")

    raw_dorc_peaks: dict[str, list[int]] = {g: [] for g in dorc_genes}
    for g_val, p_val in zip(links_df["gene"].astype(str), links_df["peak"].astype(str)):
        if g_val in raw_dorc_peaks:
            p_idx = peak_to_idx.get(p_val)
            if p_idx is not None:
                raw_dorc_peaks[g_val].append(p_idx)
    for g in dorc_genes:
        raw_dorc_peaks[g] = sorted(set(raw_dorc_peaks[g]))

    pooled_dorc_peaks: list[np.ndarray] = []
    if eff_dorc_k_pool > 0 and len(dorc_genes) > 1:
        k_eff = min(eff_dorc_k_pool, len(dorc_genes))
        means = mat_dorc.mean(axis=0, keepdims=True)
        sds = mat_dorc.std(axis=0, keepdims=True)
        z_dorc = np.where(sds > 1e-12, (mat_dorc - means) / np.maximum(sds, 1e-12), 0.0)
        # Pairwise Euclidean distance between DORC score profiles across cells
        diff = z_dorc.T[:, None, :] - z_dorc.T[None, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        for d_idx in range(len(dorc_genes)):
            nbrs = np.argsort(dist_mat[d_idx], kind="mergesort")[:k_eff]
            union_idx: set[int] = set()
            for nb in nbrs:
                union_idx.update(raw_dorc_peaks[dorc_genes[int(nb)]])
            pooled_dorc_peaks.append(np.asarray(sorted(union_idx), dtype=np.int64))
    else:
        for g in dorc_genes:
            pooled_dorc_peaks.append(np.asarray(raw_dorc_peaks[g], dtype=np.int64))

    active_dorc_indices = [i for i, p_arr in enumerate(pooled_dorc_peaks) if len(p_arr) > 0]
    if len(active_dorc_indices) == 0:
        raise ValueError("None of the DORC genes in `dorc_scores` have mapped peaks in `peak_motifs`.")

    # 5. Resolve background peaks or hypergeometric fallback
    eff_enrichment_method = enrichment_method
    bg_mat: np.ndarray | None = None
    if eff_enrichment_method == "background_z":
        if background_peaks is not None:
            bg_mat = np.asarray(background_peaks, dtype=np.int64)
            if bg_mat.ndim != 2 or bg_mat.shape[0] != n_peaks or bg_mat.shape[1] < 2:
                raise ValueError(
                    f"`background_peaks` must have shape ({n_peaks}, n_bg >= 2), got {bg_mat.shape}."
                )
            if np.any(bg_mat < 0) or np.any(bg_mat >= n_peaks):
                raise ValueError("`background_peaks` contains out-of-range peak indices.")
        elif atac is not None:
            gc_arg = peak_gc
            if gc_arg is None and isinstance(atac, ad.AnnData):
                if "gc_content" in atac.var.columns:
                    gc_arg = "gc_content"
                elif "GC" in atac.var.columns:
                    gc_arg = "GC"
            if isinstance(gc_arg, pd.Series) and atac_peak_names is not None:
                gc_idx = gc_arg.index.astype(str).tolist()
                if len(set(gc_idx)) != len(gc_idx):
                    raise ValueError("Duplicate peak identifiers detected in `peak_gc` Series index.")
                gc_idx_set = set(gc_idx)
                if any(p in gc_idx_set for p in atac_peak_names):
                    missing_gc = [p for p in atac_peak_names if p not in gc_idx_set]
                    if missing_gc:
                        raise ValueError(
                            f"`peak_gc` Series is missing {len(missing_gc)} peak(s) from `atac` "
                            f"(e.g., {missing_gc[:5]})."
                        )
                    gc_series = gc_arg.copy()
                    gc_series.index = gc_idx
                    gc_arg = gc_series.reindex(atac_peak_names).to_numpy(dtype=np.float64)
            if gc_arg is not None:
                bg_mat = get_background_peaks(
                    atac,
                    gc_arg,
                    n_bg=n_bg,
                    method=bg_method,
                    exclude_self=True,
                    orientation=orientation,
                    seed=seed,
                )
                if bg_mat.shape[0] != n_peaks:
                    raise ValueError(
                        f"`atac` peak count ({bg_mat.shape[0]}) must match `peak_motifs` rows ({n_peaks})."
                    )
            else:
                eff_enrichment_method = "hypergeometric"
        else:
            eff_enrichment_method = "hypergeometric"

    # 6. Precompute TF-RNA vs DORC correlations across aligned cells
    U_tf, valid_tf = compute_unit_rank_matrix(X_rna_aligned, active_rna_indices, method=corr_method)
    U_dorc, valid_dorc = compute_unit_rank_matrix(mat_dorc, active_dorc_indices, method=corr_method)
    # Shape: (n_active_dorcs, n_active_motifs)
    corr_mat = U_dorc.T @ U_tf
    valid_pair_mat = valid_dorc[:, None] & valid_tf[None, :]
    corr_mat = np.where(valid_pair_mat, np.clip(corr_mat, -1.0, 1.0), np.nan)

    if eff_corr_p_method == "ttest":
        t_stats = corr_mat * np.sqrt((n_cells - 2) / np.maximum(1.0 - corr_mat ** 2, 1e-15))
        p_corr_mat = np.where(valid_pair_mat, 2.0 * stats.t.sf(np.abs(t_stats), df=n_cells - 2), np.nan)
        p_corr_mat = np.clip(p_corr_mat, 0.0, 1.0)
    else:
        p_corr_mat = np.full_like(corr_mat, np.nan)
        for d_pos in range(corr_mat.shape[0]):
            row = corr_mat[d_pos]
            fin = np.isfinite(row)
            if np.sum(fin) >= 2:
                mu_r = float(np.mean(row[fin]))
                sd_r = float(np.std(row[fin], ddof=1))
                if sd_r > 1e-12:
                    z_r = (row[fin] - mu_r) / sd_r
                    p_corr_mat[d_pos, fin] = np.clip(2.0 * stats.norm.sf(np.abs(z_r)), 0.0, 1.0)
                else:
                    p_corr_mat[d_pos, fin] = 1.0
            elif np.sum(fin) == 1:
                p_corr_mat[d_pos, fin] = 1.0

    # 7. Evaluate motif enrichment per active DORC across active motifs
    n_act_dorcs = len(active_dorc_indices)
    n_act_motifs = len(active_motif_indices)

    if eff_enrichment_method == "hypergeometric":
        total_motif_hits = np.asarray(M_active.sum(axis=0), dtype=np.float64).ravel()

    out_dorc: list[str] = []
    out_motif: list[str] = []
    out_tf: list[str] = []
    out_npeaks: list[int] = []
    out_obs: list[float] = []
    out_bg_mean: list[float] = []
    out_z: list[float] = []
    out_pmotif: list[float] = []
    out_r: list[float] = []
    out_pcorr: list[float] = []

    for d_pos, d_idx in enumerate(active_dorc_indices):
        d_name = dorc_genes[d_idx]
        p_set = pooled_dorc_peaks[d_idx]
        m_d = len(p_set)

        c_obs = np.asarray(M_active[p_set, :].sum(axis=0), dtype=np.float64).ravel()

        if eff_enrichment_method == "background_z":
            assert bg_mat is not None
            n_bg_cols = bg_mat.shape[1]
            c_bg = np.empty((n_bg_cols, n_act_motifs), dtype=np.float64)
            for b in range(n_bg_cols):
                bg_idx = bg_mat[p_set, b]
                c_bg[b, :] = np.asarray(M_active[bg_idx, :].sum(axis=0), dtype=np.float64).ravel()
            mu_bg = c_bg.mean(axis=0)
            sd_bg = c_bg.std(axis=0, ddof=1)

            z_vec = np.empty(n_act_motifs, dtype=np.float64)
            for t_pos in range(n_act_motifs):
                if sd_bg[t_pos] > 1e-12:
                    z_vec[t_pos] = (c_obs[t_pos] - mu_bg[t_pos]) / sd_bg[t_pos]
                elif abs(c_obs[t_pos] - mu_bg[t_pos]) <= 1e-12:
                    z_vec[t_pos] = 0.0
                else:
                    q_hat = float(
                        np.clip(
                            mu_bg[t_pos] / m_d,
                            1.0 / (2.0 * m_d * n_bg_cols),
                            1.0 - 1.0 / (2.0 * m_d * n_bg_cols),
                        )
                    )
                    sd_eff = float(np.sqrt(m_d * q_hat * (1.0 - q_hat)))
                    z_vec[t_pos] = (c_obs[t_pos] - mu_bg[t_pos]) / max(sd_eff, 1e-12)
            if settings == "R_default":
                p_m_vec = np.clip(2.0 * stats.norm.sf(np.abs(z_vec)), 0.0, 1.0)
            else:
                p_m_vec = np.clip(stats.norm.sf(z_vec), 0.0, 1.0)
        else:
            mu_bg = m_d * total_motif_hits / n_peaks
            var_h = m_d * (total_motif_hits / n_peaks) * (1.0 - total_motif_hits / n_peaks) * (n_peaks - m_d) / max(n_peaks - 1, 1)
            sd_h = np.sqrt(np.maximum(var_h, 0.0))
            z_vec = np.where(sd_h > 1e-12, (c_obs - mu_bg) / np.maximum(sd_h, 1e-12), 0.0)
            if settings == "R_default":
                p_m_vec = np.clip(2.0 * stats.norm.sf(np.abs(z_vec)), 0.0, 1.0)
            else:
                p_m_vec = np.clip(stats.hypergeom.sf(c_obs - 1, n_peaks, total_motif_hits, m_d), 0.0, 1.0)

        for t_pos in range(n_act_motifs):
            out_dorc.append(d_name)
            out_motif.append(active_motif_names[t_pos])
            out_tf.append(active_tf_names[t_pos])
            out_npeaks.append(m_d)
            out_obs.append(float(c_obs[t_pos]))
            out_bg_mean.append(float(mu_bg[t_pos]))
            out_z.append(float(z_vec[t_pos]))
            out_pmotif.append(float(p_m_vec[t_pos]))
            out_r.append(float(corr_mat[d_pos, t_pos]))
            out_pcorr.append(float(p_corr_mat[d_pos, t_pos]))

    z_arr = np.asarray(out_z, dtype=np.float64)
    p_motif_arr = np.asarray(out_pmotif, dtype=np.float64)
    r_arr = np.asarray(out_r, dtype=np.float64)
    p_corr_arr = np.asarray(out_pcorr, dtype=np.float64)

    fdr_motif_arr = bh_fdr(p_motif_arr)
    fdr_corr_arr = bh_fdr(p_corr_arr)

    p_m_clip = np.clip(p_motif_arr, min_p, 1.0)
    p_c_clip = np.clip(p_corr_arr, min_p, 1.0)

    if settings == "R_default":
        # FigR.R: Enrichment.log10P[z < 0] <- 0 (equivalently p_enrich = 1 when z < 0)
        enrichment_ok = z_arr >= 0.0 if clamp_negative_enrichment else np.ones_like(z_arr, dtype=bool)
    elif clamp_negative_enrichment:
        enrichment_ok = z_arr > 0.0
    else:
        enrichment_ok = np.ones_like(z_arr, dtype=bool)

    E = np.where(enrichment_ok, -np.log10(p_m_clip), 0.0)
    C = -np.log10(p_c_clip)

    if eff_score_mode == "product":
        score_arr = np.where(np.isfinite(r_arr) & np.isfinite(p_corr_arr), np.sign(r_arr) * E * C, np.nan)
    else:
        p_m_eff = np.where(enrichment_ok, p_m_clip, 1.0)
        p_union = np.clip(p_m_eff + p_c_clip - p_m_eff * p_c_clip, min_p, 1.0)
        raw_union = np.where(
            enrichment_ok & (p_union < 1.0),
            np.sign(r_arr) * (-np.log10(p_union)),
            0.0,
        )
        score_arr = np.where(np.isfinite(r_arr) & np.isfinite(p_corr_arr), raw_union, np.nan)

    role_list = [
        "activator" if (np.isfinite(s) and s > 0.0) else ("repressor" if (np.isfinite(s) and s < 0.0) else "neutral")
        for s in score_arr
    ]

    return FigRGRNResult(
        dorc_gene=np.asarray(out_dorc, dtype=object),
        motif=np.asarray(out_motif, dtype=object),
        tf_gene=np.asarray(out_tf, dtype=object),
        n_dorc_peaks=np.asarray(out_npeaks, dtype=np.int64),
        motif_obs_count=np.asarray(out_obs, dtype=np.float64),
        motif_bg_mean=np.asarray(out_bg_mean, dtype=np.float64),
        motif_enrichment_z=z_arr,
        p_motif=p_motif_arr,
        fdr_motif=fdr_motif_arr,
        r_tf_dorc=r_arr,
        p_corr=p_corr_arr,
        fdr_corr=fdr_corr_arr,
        dorc_reg_score=score_arr,
        role=np.asarray(role_list, dtype=object),
        n_cells=n_cells,
        enrichment_method=eff_enrichment_method,
        corr_method=corr_method,
        score_mode=eff_score_mode,
    )
