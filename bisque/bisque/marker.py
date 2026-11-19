"""Marker-based PCA bulk RNA-seq abundance scoring via Bisque (Jew et al., 2020)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from ._io import (
    ExpressionInput,
    SettingsMode,
    cpm_scale_factors,
    parse_expression_input,
    subset_columns_dense,
    validate_settings,
)


@dataclass(frozen=True)
class MarkerDeconvolutionResult:
    """Typed result container for `marker_based_decomposition`."""

    bulk_props: np.ndarray  # (n_bulk, n_cell_types)
    sample_names: list[str]
    cell_types: list[str]
    markers_used: dict[str, list[str]]
    variance_explained: dict[str, float]
    weights_used: dict[str, np.ndarray] | None
    normalization: str
    weighted_pca: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Return estimated marker-based abundance scores as ``samples x cell_types``."""
        return pd.DataFrame(
            self.bulk_props.copy(),
            index=self.sample_names,
            columns=self.cell_types,
        )

    def variance_explained_series(self) -> pd.Series:
        """Return proportion of marker variance explained by PC1 per cell type."""
        return pd.Series(
            [self.variance_explained[ct] for ct in self.cell_types],
            index=self.cell_types,
            name="variance_explained",
            dtype=np.float64,
        )

    def summary(self) -> str:
        """Return a concise multi-line summary of marker-based decomposition."""
        valid_cts = [
            ct for ct in self.cell_types if np.isfinite(self.variance_explained[ct])
        ]
        pve_str = ", ".join(
            f"{ct}={self.variance_explained[ct]:.3f} ({len(self.markers_used[ct])} genes)"
            for ct in valid_cts
        )
        return "\n".join(
            [
                "MarkerDeconvolutionResult",
                f"  Bulk samples:       {len(self.sample_names)}",
                f"  Cell types:         {len(self.cell_types)} ({len(valid_cts)} valid)",
                f"  Weighted PCA:       {self.weighted_pca}",
                f"  Normalization:      {self.normalization}",
                f"  PC1 PVE per type:   {pve_str if pve_str else 'none'}",
            ]
        )


def _parse_markers_and_weights(
    markers: Mapping[str, Sequence[str]] | pd.DataFrame,
    *,
    weights: Mapping[str, Mapping[str, float]] | pd.DataFrame | None,
    weighted_pca: bool,
    ct_col: str,
    gene_col: str,
    weight_col: str,
) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, float]] | None]:
    ct_order: list[str] = []
    ct_genes: dict[str, list[str]] = {}
    inferred_weights: dict[str, dict[str, float]] = {}
    has_df_weights = False

    if isinstance(markers, pd.DataFrame):
        if markers.empty:
            raise ValueError("`markers` DataFrame must not be empty.")
        if ct_col not in markers.columns or gene_col not in markers.columns:
            raise ValueError(
                f"`markers` DataFrame must contain columns '{ct_col}' and '{gene_col}'."
            )
        if weighted_pca and weights is None and weight_col not in markers.columns:
            raise ValueError(
                f"weighted_pca=True requires `{weight_col}` column in `markers` or explicit `weights`."
            )
        has_df_weights = weight_col in markers.columns and weights is None

        ct_vals = markers[ct_col].astype(str)
        gene_vals = markers[gene_col].astype(str)
        if has_df_weights:
            w_vals = pd.to_numeric(markers[weight_col], errors="raise").to_numpy(
                dtype=np.float64
            )
        else:
            w_vals = None

        seen_pairs: set[tuple[str, str]] = set()
        for idx in range(len(markers)):
            ct = ct_vals.iloc[idx]
            gene = gene_vals.iloc[idx]
            if ct not in ct_genes:
                ct_order.append(ct)
                ct_genes[ct] = []
                if has_df_weights:
                    inferred_weights[ct] = {}
            if (ct, gene) in seen_pairs:
                continue
            seen_pairs.add((ct, gene))
            ct_genes[ct].append(gene)
            if has_df_weights and w_vals is not None:
                inferred_weights[ct][gene] = float(w_vals[idx])
    elif isinstance(markers, Mapping):
        if len(markers) == 0:
            raise ValueError("`markers` mapping must not be empty.")
        for ct_raw, genes_raw in markers.items():
            ct = str(ct_raw)
            ct_order.append(ct)
            ct_genes[ct] = list(dict.fromkeys(str(g) for g in genes_raw))
    else:
        raise ValueError("`markers` must be a mapping {cell_type: [genes]} or a DataFrame.")

    weight_lookup: dict[str, dict[str, float]] | None = None
    if weights is not None:
        weight_lookup = {}
        if isinstance(weights, pd.DataFrame):
            if (
                ct_col not in weights.columns
                or gene_col not in weights.columns
                or weight_col not in weights.columns
            ):
                raise ValueError(
                    f"`weights` DataFrame must contain columns '{ct_col}', '{gene_col}', and '{weight_col}'."
                )
            for idx in range(len(weights)):
                ct = str(weights[ct_col].iloc[idx])
                gene = str(weights[gene_col].iloc[idx])
                w_val = float(weights[weight_col].iloc[idx])
                weight_lookup.setdefault(ct, {})[gene] = w_val
        elif isinstance(weights, Mapping):
            for ct_raw, g_map in weights.items():
                ct = str(ct_raw)
                weight_lookup[ct] = {str(g): float(w) for g, w in g_map.items()}
        else:
            raise ValueError("`weights` must be a nested mapping or a DataFrame.")
    elif has_df_weights:
        weight_lookup = inferred_weights
    elif weighted_pca:
        raise ValueError(
            "weighted_pca=True requires either `weights` or a weight column in `markers`."
        )

    return ct_order, ct_genes, weight_lookup


def marker_based_decomposition(
    bulk: ExpressionInput,
    markers: Mapping[str, Sequence[str]] | pd.DataFrame,
    *,
    weights: Mapping[str, Mapping[str, float]] | pd.DataFrame | None = None,
    weighted_pca: bool | None = None,
    ct_col: str = "cluster",
    gene_col: str = "gene",
    weight_col: str = "avg_logFC",
    min_gene: int = 5,
    max_gene: int | None = 200,
    unique_markers: bool = True,
    normalize_cpm: bool = True,
    normalization: Literal["none", "minmax", "proportional"] | None = None,
    bulk_layer: str | None = None,
    bulk_sample_names: Sequence[str] | None = None,
    bulk_gene_names: Sequence[str] | None = None,
    settings: SettingsMode = "default",
) -> MarkerDeconvolutionResult:
    """
    Estimate relative cell-type abundance from bulk expression using marker-gene PCA.
    """
    eff_mode = validate_settings(settings)
    if normalization is None:
        eff_norm: Literal["none", "minmax", "proportional"] = (
            "minmax" if eff_mode == "R_default" else "none"
        )
    else:
        eff_norm = normalization
    if weighted_pca is None:
        if eff_mode == "R_default":
            eff_weighted = bool(
                weights is not None
                or (isinstance(markers, pd.DataFrame) and weight_col in markers.columns)
            )
        else:
            eff_weighted = False
    else:
        eff_weighted = bool(weighted_pca)

    if min_gene < 1:
        raise ValueError(f"`min_gene` must be >= 1; got {min_gene}.")
    if max_gene is not None and max_gene < min_gene:
        raise ValueError(
            f"`max_gene` ({max_gene}) must be >= `min_gene` ({min_gene})."
        )
    if eff_norm not in ("none", "minmax", "proportional"):
        raise ValueError(
            f"`normalization` must be 'none', 'minmax', or 'proportional'; got {eff_norm!r}."
        )

    parsed_bulk = parse_expression_input(
        bulk,
        name="bulk",
        layer=bulk_layer,
        sample_names=bulk_sample_names,
        gene_names=bulk_gene_names,
        obs_prefix="sample",
        allow_1d=False,
        require_nonnegative=normalize_cpm,
    )
    n_bulk = parsed_bulk.matrix.shape[0]
    if n_bulk < 2:
        raise ValueError(
            f"marker_based_decomposition requires at least 2 bulk samples; got {n_bulk}."
        )

    ct_order, ct_genes, weight_lookup = _parse_markers_and_weights(
        markers,
        weights=weights,
        weighted_pca=eff_weighted,
        ct_col=ct_col,
        gene_col=gene_col,
        weight_col=weight_col,
    )

    if unique_markers:
        gene_ct_counts: dict[str, int] = {}
        for ct in ct_order:
            for g in ct_genes[ct]:
                gene_ct_counts[g] = gene_ct_counts.get(g, 0) + 1
        shared_genes = {g for g, cnt in gene_ct_counts.items() if cnt > 1}
        if shared_genes:
            ct_genes = {
                ct: [g for g in ct_genes[ct] if g not in shared_genes]
                for ct in ct_order
            }

    bulk_gene_index = pd.Index(parsed_bulk.var_names)
    union_genes_ordered: list[str] = []
    seen_union: set[str] = set()
    for ct in ct_order:
        for g in ct_genes[ct]:
            if g in bulk_gene_index and g not in seen_union:
                seen_union.add(g)
                union_genes_ordered.append(g)

    if union_genes_ordered:
        col_idx = bulk_gene_index.get_indexer(union_genes_ordered)
        X_markers_all = subset_columns_dense(parsed_bulk.matrix, col_idx)
        if normalize_cpm:
            scale = cpm_scale_factors(parsed_bulk.matrix)
            X_markers_all = X_markers_all * scale[:, None]
        marker_col_map = {g: j for j, g in enumerate(union_genes_ordered)}
        marker_sd = np.std(X_markers_all, axis=0, ddof=1)
    else:
        X_markers_all = np.empty((n_bulk, 0), dtype=np.float64)
        marker_col_map = {}
        marker_sd = np.empty(0, dtype=np.float64)

    n_ct = len(ct_order)
    raw_scores = np.full((n_bulk, n_ct), np.nan, dtype=np.float64)
    markers_used: dict[str, list[str]] = {}
    variance_explained: dict[str, float] = {}
    weights_used: dict[str, np.ndarray] | None = {} if eff_weighted else None

    for k, ct in enumerate(ct_order):
        cand_genes = [
            g
            for g in ct_genes[ct]
            if g in marker_col_map and marker_sd[marker_col_map[g]] > 1e-12
        ]

        if eff_weighted:
            assert weight_lookup is not None
            ct_w_map = weight_lookup.get(ct, {})
            cand_weights = []
            for g in cand_genes:
                if g not in ct_w_map:
                    raise ValueError(
                        f"Missing weight for marker gene '{g}' in cell type '{ct}'."
                    )
                w_val = float(ct_w_map[g])
                if not np.isfinite(w_val) or w_val <= 0.0:
                    raise ValueError(
                        f"Marker weight for '{g}' in cell type '{ct}' must be finite and > 0; got {w_val}."
                    )
                cand_weights.append(w_val)

            if max_gene is not None and len(cand_genes) > max_gene:
                order = np.argsort(-np.asarray(cand_weights, dtype=np.float64), kind="mergesort")
                top_idx = order[:max_gene]
                cand_genes = [cand_genes[i] for i in top_idx]
                cand_weights = [cand_weights[i] for i in top_idx]
        else:
            if max_gene is not None and len(cand_genes) > max_gene:
                cand_genes = cand_genes[:max_gene]
            cand_weights = [1.0] * len(cand_genes)

        markers_used[ct] = cand_genes

        if len(cand_genes) < min_gene:
            warnings.warn(
                f"Cell type '{ct}' has only {len(cand_genes)} informative markers "
                f"(min_gene={min_gene}); setting abundance scores to NaN.",
                UserWarning,
                stacklevel=2,
            )
            variance_explained[ct] = float("nan")
            if weights_used is not None:
                weights_used[ct] = np.asarray(cand_weights, dtype=np.float64)
            continue

        idx_cols = [marker_col_map[g] for g in cand_genes]
        X_k = X_markers_all[:, idx_cols]
        mu_k = np.mean(X_k, axis=0)
        sd_k = np.std(X_k, axis=0, ddof=1)
        Z_k = (X_k - mu_k) / sd_k

        w_arr = np.asarray(cand_weights, dtype=np.float64)
        if weights_used is not None:
            weights_used[ct] = w_arr.copy()

        Z_w = Z_k * w_arr[None, :] if eff_weighted else Z_k

        U, S, Vt = np.linalg.svd(Z_w, full_matrices=False)
        score = U[:, 0] * S[0]
        total_var = float(np.sum(S * S))
        pve = float((S[0] * S[0]) / total_var) if total_var > 0.0 else 0.0

        # Deterministic sign alignment against weighted mean standardized marker expression
        z_bar = np.sum(Z_k * w_arr[None, :], axis=1) / np.sum(w_arr)
        align = float(np.dot(score, z_bar))
        if align < 0.0 or (abs(align) <= 1e-15 and float(np.sum(Vt[0, :])) < 0.0):
            score = -score

        raw_scores[:, k] = score
        variance_explained[ct] = pve

    if eff_norm == "none":
        bulk_props = raw_scores
    else:
        minmax = np.full_like(raw_scores, np.nan, dtype=np.float64)
        valid_cols = np.isfinite(raw_scores).all(axis=0)
        for k in np.where(valid_cols)[0]:
            col = raw_scores[:, k]
            c_min = float(np.min(col))
            c_max = float(np.max(col))
            span = c_max - c_min
            if span > 1e-15:
                minmax[:, k] = (col - c_min) / span
            else:
                minmax[:, k] = 0.5

        if eff_norm == "minmax":
            bulk_props = minmax
        else:
            bulk_props = np.full_like(minmax, np.nan, dtype=np.float64)
            n_valid_ct = int(np.sum(valid_cols))
            if n_valid_ct > 0:
                valid_sub = minmax[:, valid_cols]
                row_sums = np.sum(valid_sub, axis=1, keepdims=True)
                pos_rows = (row_sums > 1e-15).ravel()
                scaled = np.empty_like(valid_sub, dtype=np.float64)
                scaled[pos_rows, :] = valid_sub[pos_rows, :] / row_sums[pos_rows, :]
                scaled[~pos_rows, :] = 1.0 / float(n_valid_ct)
                bulk_props[:, valid_cols] = scaled

    return MarkerDeconvolutionResult(
        bulk_props=bulk_props,
        sample_names=parsed_bulk.obs_names,
        cell_types=ct_order,
        markers_used=markers_used,
        variance_explained=variance_explained,
        weights_used=weights_used,
        normalization=eff_norm,
        weighted_pca=eff_weighted,
    )
