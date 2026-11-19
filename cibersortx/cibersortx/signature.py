"""
Clean-room signature matrix construction from scRNA-seq or replicate reference
profiles via one-vs-rest differential expression and condition-number sweep
(Newman et al., Nature Methods 2015; Nature Biotechnology 2019).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats

from ._io import (
    ExpressionInput,
    SettingsInput,
    bh_fdr,
    parse_label_vector,
    parse_sc_reference_sparse_aware,
    validate_settings,
)


@dataclass(frozen=True)
class SignatureMatrixResult:
    """Typed result container for `build_signature_matrix`."""

    signature: np.ndarray
    gene_names: list[str]
    cell_types: list[str]
    condition_number: float
    selected_genes_per_type: int
    condition_curve: dict[int, float]
    deg_table: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return `(G, K)` signature matrix as a pandas DataFrame (`genes x cell_types`)."""
        return pd.DataFrame(
            self.signature,
            index=self.gene_names,
            columns=self.cell_types,
        )

    def deg_dataframe(self) -> pd.DataFrame:
        """Return differential expression screening table across cell types."""
        return self.deg_table.copy()

    def summary(self) -> str:
        return (
            f"SignatureMatrixResult(genes={len(self.gene_names)}, "
            f"cell_types={len(self.cell_types)}, "
            f"g_top={self.selected_genes_per_type}, "
            f"cond={self.condition_number:.3f})"
        )


def _compute_one_vs_rest_pvalues(
    x_k: np.ndarray,
    x_rest: np.ndarray,
    test_method: Literal["mannwhitney", "ttest"],
) -> np.ndarray:
    n_genes = x_k.shape[0]
    if test_method == "mannwhitney":
        res = stats.mannwhitneyu(x_k, x_rest, axis=1, alternative="two-sided")
        p_vals = np.asarray(res.pvalue, dtype=np.float64)
        p_vals = np.where(np.isfinite(p_vals), p_vals, 1.0)
        return p_vals
    if test_method == "ttest":
        var_k = np.var(x_k, axis=1, ddof=1) if x_k.shape[1] > 1 else np.zeros(n_genes)
        var_r = np.var(x_rest, axis=1, ddof=1) if x_rest.shape[1] > 1 else np.zeros(n_genes)
        active = (var_k > 1e-12) | (var_r > 1e-12)
        p_vals = np.ones(n_genes, dtype=np.float64)
        if np.any(active):
            res = stats.ttest_ind(x_k[active], x_rest[active], axis=1, equal_var=False)
            pv = np.asarray(res.pvalue, dtype=np.float64)
            p_vals[active] = np.where(np.isfinite(pv), pv, 1.0)
        return p_vals
    raise ValueError(f"Unsupported test_method '{test_method}'. Expected 'mannwhitney' or 'ttest'.")


def build_signature_matrix(
    reference: ExpressionInput,
    cell_type_labels: str | Sequence[Any] | np.ndarray | pd.Series,
    *,
    layer: str | None = None,
    cells_as_rows: bool = False,
    donor_labels: str | Sequence[Any] | np.ndarray | pd.Series | None = None,
    replicates_per_type: int | None = None,
    normalize_cells: bool = True,
    target_sum: float = 1e4,
    test_method: Literal["mannwhitney", "ttest"] | None = None,
    q_value: float = 0.01,
    min_fc: float = 1.5,
    min_expr: float = 0.0,
    min_cells_per_gene: int = 3,
    min_genes: int = 50,
    max_genes: int = 150,
    step: int = 5,
    summary_stat: Literal["mean", "median"] = "mean",
    is_log2: bool = False,
    log2_offset: float = 0.0,
    seed: int = 0,
    settings: SettingsInput = "default",
) -> SignatureMatrixResult:
    """
    Construct a cell-type signature matrix from single-cell or replicate
    reference profiles via one-vs-rest DEG selection and condition-number
    minimization (Newman et al., Nat Biotechnol 2019).

    When `settings="R_default"` (or `"cibersortx_default"`), applies the exact
    Newman et al. (2019) single-cell signature protocol:
      - `> 50%` within-phenotype non-zero single-cell expression fraction gate
      - Random single-cell mini-bulk replicate pooling (default 5 replicates/type)
      - Welch two-sample `ttest` (`q <= 0.01`, `FC >= 1.5`)
    """
    eff_mode = validate_settings(settings)
    eff_test_method: Literal["mannwhitney", "ttest"] = (
        test_method
        if test_method is not None
        else ("ttest" if eff_mode == "R_default" else "mannwhitney")
    )
    if target_sum <= 0:
        raise ValueError("`target_sum` must be positive.")
    if min_fc <= 0:
        raise ValueError("`min_fc` must be positive.")
    if min_genes < 1 or max_genes < min_genes:
        raise ValueError("Require 1 <= min_genes <= max_genes.")
    if step < 1:
        raise ValueError("`step` must be >= 1.")
    if min_cells_per_gene < 1:
        raise ValueError("`min_cells_per_gene` must be >= 1.")
    if summary_stat not in ("mean", "median"):
        raise ValueError(f"Unsupported summary_stat '{summary_stat}'.")
    if replicates_per_type is not None and replicates_per_type < 2:
        raise ValueError("`replicates_per_type` must be >= 2 when provided.")

    X_raw, gene_names, _cell_names, is_sparse = parse_sc_reference_sparse_aware(
        reference,
        layer=layer,
        cells_as_rows=cells_as_rows,
        is_log2=is_log2,
        log2_offset=log2_offset,
    )
    n_cells_orig, n_genes_orig = X_raw.shape
    labels = parse_label_vector(reference, cell_type_labels, n_obs=n_cells_orig, kind="cell_type_labels")

    donor_vec: list[str] | None = None
    if donor_labels is not None:
        donor_vec = parse_label_vector(reference, donor_labels, n_obs=n_cells_orig, kind="donor_labels")

    # 1. Cell library-size normalization and zero-sum cell removal
    if is_sparse:
        cell_sums = np.asarray(X_raw.sum(axis=1)).ravel()
    else:
        cell_sums = np.sum(X_raw, axis=1)

    nonzero_cells = cell_sums > 0.0
    if not np.any(nonzero_cells):
        raise ValueError("All reference cells have zero total expression.")
    if not np.all(nonzero_cells):
        n_dropped = int(np.sum(~nonzero_cells))
        warnings.warn(
            f"Excluded {n_dropped} zero-count cells from reference.",
            UserWarning,
            stacklevel=2,
        )
        X_raw = X_raw[nonzero_cells]
        labels = [lbl for lbl, keep in zip(labels, nonzero_cells) if keep]
        if donor_vec is not None:
            donor_vec = [d for d, keep in zip(donor_vec, nonzero_cells) if keep]
        cell_sums = cell_sums[nonzero_cells]

    if normalize_cells:
        scale = float(target_sum) / cell_sums
        if is_sparse:
            X_norm = sparse.diags(scale) @ X_raw
        else:
            X_norm = X_raw * scale[:, None]
    else:
        X_norm = X_raw

    # Preserve non-alphabetical first-seen cell type order
    cell_types = list(dict.fromkeys(labels))
    if len(cell_types) < 2:
        raise ValueError(f"At least 2 cell types are required, found {len(cell_types)}.")

    # Filter genes with zero total expression or insufficient detected cells
    n_cells_rem = X_norm.shape[0]
    min_cells_eff = min(int(min_cells_per_gene), n_cells_rem)
    if is_sparse:
        X_csc = X_norm.tocsc()
        gene_totals = np.asarray(X_csc.sum(axis=0)).ravel()
        gene_detected = np.diff(X_csc.indptr)
    else:
        gene_totals = np.sum(X_norm, axis=0)
        gene_detected = np.sum(X_norm > 0.0, axis=0)

    keep_genes = (gene_totals > 0.0) & (gene_detected >= min_cells_eff)
    if int(np.sum(keep_genes)) < len(cell_types):
        raise ValueError(
            "Too few expressed genes remain after filtering "
            f"({int(np.sum(keep_genes))} genes for {len(cell_types)} cell types)."
        )

    active_genes = [g for g, keep in zip(gene_names, keep_genes) if keep]
    if is_sparse:
        X_active = X_norm[:, keep_genes].toarray()  # (C_rem, G_active)
    else:
        X_active = np.asarray(X_norm[:, keep_genes], dtype=np.float64)

    # Transpose to (G_active, C_rem)
    prof_mat = np.ascontiguousarray(X_active.T, dtype=np.float64)
    prof_labels = np.asarray(labels, dtype=object)

    # Newman et al. (2019) > 50% within-phenotype non-zero expression fraction gate
    nonzero_frac_by_type: dict[str, np.ndarray] = {}
    if eff_mode == "R_default":
        for ct in cell_types:
            ct_sc_mask = prof_labels == ct
            cols_sc = prof_mat[:, ct_sc_mask]
            nz_frac = np.mean(cols_sc > 0.0, axis=1)
            nonzero_frac_by_type[ct] = nz_frac
            if np.any(nz_frac > 0.5):
                low_det = nz_frac <= 0.5
                if np.any(low_det):
                    prof_mat[np.ix_(low_det, np.flatnonzero(ct_sc_mask))] = 0.0

    # 2. Optional replicate pseudobulk partitioning (automatic mini-bulks in R_default)
    eff_reps = (
        replicates_per_type
        if replicates_per_type is not None
        else (5 if (eff_mode == "R_default" and donor_vec is None) else None)
    )
    if donor_vec is not None:
        donor_arr = np.asarray(donor_vec, dtype=object)
        rep_cols: list[np.ndarray] = []
        rep_types: list[str] = []
        for ct in cell_types:
            ct_mask = prof_labels == ct
            donors_ct = list(dict.fromkeys(donor_arr[ct_mask].tolist()))
            for d in donors_ct:
                idx = np.flatnonzero(ct_mask & (donor_arr == d))
                rep_cols.append(np.mean(prof_mat[:, idx], axis=1))
                rep_types.append(ct)
        prof_mat = np.column_stack(rep_cols)
        prof_labels = np.asarray(rep_types, dtype=object)
    elif eff_reps is not None:
        rng = np.random.default_rng(seed)
        rep_cols = []
        rep_types = []
        for ct in cell_types:
            idx = np.flatnonzero(prof_labels == ct)
            perm_idx = idx[rng.permutation(len(idx))]
            n_buckets = min(int(eff_reps), len(perm_idx))
            for bucket in np.array_split(perm_idx, n_buckets):
                rep_cols.append(np.mean(prof_mat[:, bucket], axis=1))
                rep_types.append(ct)
        prof_mat = np.column_stack(rep_cols)
        prof_labels = np.asarray(rep_types, dtype=object)

    n_genes_active = len(active_genes)
    n_types = len(cell_types)

    # Full cell-type expression summary (G_active, K)
    B_full = np.empty((n_genes_active, n_types), dtype=np.float64)
    for k, ct in enumerate(cell_types):
        cols_k = prof_mat[:, prof_labels == ct]
        if summary_stat == "mean":
            B_full[:, k] = np.mean(cols_k, axis=1)
        else:
            B_full[:, k] = np.median(cols_k, axis=1)

    # 3. One-vs-rest Differential Expression Screening
    log2_min_fc = float(np.log2(min_fc))
    markers_by_type: dict[str, list[int]] = {}
    deg_rows: list[pd.DataFrame] = []

    for k, ct in enumerate(cell_types):
        mask_k = prof_labels == ct
        x_k = prof_mat[:, mask_k]
        x_rest = prof_mat[:, ~mask_k]
        mu_k = np.mean(x_k, axis=1)
        mu_rest = np.mean(x_rest, axis=1)
        log2_fc = np.log2((mu_k + 1.0) / (mu_rest + 1.0))

        p_vals = _compute_one_vs_rest_pvalues(x_k, x_rest, test_method=eff_test_method)
        q_vals = bh_fdr(p_vals)

        strict_mask = (q_vals <= q_value) & (log2_fc >= log2_min_fc) & (mu_k >= min_expr)
        if eff_mode == "R_default" and ct in nonzero_frac_by_type:
            nz_f = nonzero_frac_by_type[ct]
            if np.any(nz_f > 0.5):
                strict_mask = strict_mask & (nz_f > 0.5)
        strict_idx = np.flatnonzero(strict_mask)

        # Sort deterministically by (-log2_fc, q_val, gene_name)
        def _rank_indices(indices: np.ndarray) -> list[int]:
            return sorted(
                indices.tolist(),
                key=lambda g_i: (-float(log2_fc[g_i]), float(q_vals[g_i]), active_genes[g_i]),
            )

        ranked = _rank_indices(strict_idx)
        if len(ranked) < min_genes:
            warnings.warn(
                f"Cell type '{ct}' had only {len(ranked)} genes passing DEG cutoffs "
                f"(min_genes={min_genes}); supplementing with top upregulated genes.",
                UserWarning,
                stacklevel=2,
            )
            used = set(ranked)
            tier1 = np.array(
                [i for i in range(n_genes_active) if i not in used and log2_fc[i] > 0.0 and q_vals[i] <= 0.25],
                dtype=int,
            )
            ranked.extend(_rank_indices(tier1))
            used.update(ranked)
            if len(ranked) < min_genes:
                tier2 = np.array(
                    [i for i in range(n_genes_active) if i not in used and log2_fc[i] > 0.0],
                    dtype=int,
                )
                ranked.extend(_rank_indices(tier2))
                used.update(ranked)
            if len(ranked) < min_genes:
                tier3 = np.array([i for i in range(n_genes_active) if i not in used], dtype=int)
                ranked.extend(_rank_indices(tier3))

        markers_by_type[ct] = ranked
        deg_rows.append(
            pd.DataFrame(
                {
                    "gene": active_genes,
                    "cell_type": ct,
                    "mean_target": mu_k,
                    "mean_rest": mu_rest,
                    "log2_fc": log2_fc,
                    "p_value": p_vals,
                    "q_value": q_vals,
                    "passed_strict": strict_mask,
                }
            )
        )

    # 4. Condition-Number Optimization Sweep
    max_avail = min(len(markers_by_type[ct]) for ct in cell_types)
    g_start = min(int(min_genes), max_avail)
    g_stop = min(int(max_genes), max_avail)
    grid = list(range(g_start, g_stop + 1, int(step)))
    if not grid:
        grid = [g_start]

    condition_curve: dict[int, float] = {}
    best_g_top: int | None = None
    best_cond = np.inf
    best_union_mask: np.ndarray | None = None

    for g_top in grid:
        union_idx: set[int] = set()
        for ct in cell_types:
            union_idx.update(markers_by_type[ct][:g_top])
        if len(union_idx) < n_types:
            continue
        mask = np.zeros(n_genes_active, dtype=bool)
        mask[list(union_idx)] = True
        B_sub = B_full[mask, :]
        cond = float(np.linalg.cond(B_sub))
        condition_curve[int(g_top)] = cond
        if cond < best_cond - 1e-12:
            best_cond = cond
            best_g_top = int(g_top)
            best_union_mask = mask

    if best_g_top is None or best_union_mask is None:
        raise ValueError("Unable to construct a valid signature matrix with >= K genes.")

    sig_genes = [g for g, keep in zip(active_genes, best_union_mask) if keep]
    sig_matrix = np.ascontiguousarray(B_full[best_union_mask, :], dtype=np.float64)

    sig_gene_set = set(sig_genes)
    deg_table = pd.concat(deg_rows, ignore_index=True)
    deg_table["selected"] = deg_table["gene"].isin(sig_gene_set)

    return SignatureMatrixResult(
        signature=sig_matrix,
        gene_names=sig_genes,
        cell_types=cell_types,
        condition_number=float(best_cond),
        selected_genes_per_type=int(best_g_top),
        condition_curve=condition_curve,
        deg_table=deg_table,
    )
