"""Reference-based bulk RNA-seq deconvolution via Bisque (Jew et al., 2020)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, nnls

from ._io import (
    ExpressionInput,
    ParsedExpression,
    SettingsMode,
    cpm_normalize,
    parse_expression_input,
    resolve_sc_annotations,
    subset_columns_dense,
    validate_settings,
)


def _solve_simplex_qp(
    G_mat: np.ndarray,
    Z: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Solve the exact equality-and-inequality-constrained simplex quadratic program
    (`limSolve::lsei` in `BisqueRNA::Deconvolve`):
        min_{p in R^K} ||Z p - b||_2^2   subject to  sum_k p_k = 1,  p_k >= 0.
    """
    n_ct = Z.shape[1]
    if n_ct == 1:
        p_one = np.ones(1, dtype=np.float64)
        return p_one, float(np.linalg.norm(Z @ p_one - b))

    c_vec = Z.T @ b
    scale = max(float(np.trace(G_mat)), 1.0)
    G_sc = G_mat / scale
    c_sc = c_vec / scale

    def _obj_and_grad(p: np.ndarray) -> tuple[float, np.ndarray]:
        gp = G_sc @ p
        val = 0.5 * float(np.dot(p, gp)) - float(np.dot(c_sc, p))
        grad = gp - c_sc
        return val, grad

    p0 = np.full(n_ct, 1.0 / float(n_ct), dtype=np.float64)
    bounds = [(0.0, 1.0)] * n_ct
    constraints = {
        "type": "eq",
        "fun": lambda p: float(np.sum(p) - 1.0),
        "jac": lambda p: np.ones(n_ct, dtype=np.float64),
    }
    res = minimize(
        _obj_and_grad,
        p0,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    p_hat = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    p_sum = float(np.sum(p_hat))
    if p_sum > 1e-15:
        p_hat /= p_sum
    else:
        p_hat = p0
    rnorm = float(np.linalg.norm(Z @ p_hat - b))
    return p_hat, rnorm


@dataclass(frozen=True)
class SCReference:
    """Single-cell pseudobulk and cell-type reference summary."""

    signature_matrix: np.ndarray  # (n_genes, n_cell_types)
    sc_props: np.ndarray  # (n_subjects, n_cell_types)
    pseudobulk: np.ndarray  # (n_subjects, n_genes)
    gene_names: list[str]
    cell_types: list[str]
    subject_names: list[str]

    def signature_dataframe(self) -> pd.DataFrame:
        """Return cell-type signature matrix ``Z`` as ``genes x cell_types``."""
        return pd.DataFrame(
            self.signature_matrix.copy(),
            index=self.gene_names,
            columns=self.cell_types,
        )

    def props_dataframe(self) -> pd.DataFrame:
        """Return observed single-cell donor proportions as ``subjects x cell_types``."""
        return pd.DataFrame(
            self.sc_props.copy(),
            index=self.subject_names,
            columns=self.cell_types,
        )

    def pseudobulk_dataframe(self) -> pd.DataFrame:
        """Return synthetic pseudobulk matrix ``Y_sc = P_sc @ Z.T`` as ``subjects x genes``."""
        return pd.DataFrame(
            self.pseudobulk.copy(),
            index=self.subject_names,
            columns=self.gene_names,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the signature matrix ``Z`` as a ``pandas.DataFrame``."""
        return self.signature_dataframe()

    def summary(self) -> str:
        """Return a concise human-readable summary of the single-cell reference."""
        return "\n".join(
            [
                "SCReference",
                f"  Subjects:   {len(self.subject_names)}",
                f"  Cell types: {len(self.cell_types)} ({', '.join(self.cell_types)})",
                f"  Genes:      {len(self.gene_names)}",
            ]
        )


@dataclass(frozen=True)
class ReferenceDeconvolutionResult:
    """Typed result container for `reference_based_decomposition`."""

    bulk_props: np.ndarray  # (n_bulk, n_cell_types)
    sc_props: np.ndarray  # (n_sc_donors, n_cell_types)
    sample_names: list[str]
    sc_subject_names: list[str]
    cell_types: list[str]
    genes_used: list[str]
    rnorm: np.ndarray  # (n_bulk,)
    signature_matrix: np.ndarray  # (n_genes_used, n_cell_types)
    transformed_bulk: np.ndarray  # (n_bulk, n_genes_used)
    mode: str  # "overlapping" | "non_overlapping" | "untransformed_fallback"
    overlapping_subjects: list[str]

    @property
    def used_fallback(self) -> bool:
        """Whether deconvolution fell back to untransformed NNLS."""
        return self.mode == "untransformed_fallback"

    def to_dataframe(self) -> pd.DataFrame:
        """Return estimated bulk cell-type proportions as ``samples x cell_types``."""
        return pd.DataFrame(
            self.bulk_props.copy(),
            index=self.sample_names,
            columns=self.cell_types,
        )

    def sc_dataframe(self) -> pd.DataFrame:
        """Return observed single-cell donor proportions as ``subjects x cell_types``."""
        return pd.DataFrame(
            self.sc_props.copy(),
            index=self.sc_subject_names,
            columns=self.cell_types,
        )

    def signature_dataframe(self) -> pd.DataFrame:
        """Return cell-type signature matrix as ``genes_used x cell_types``."""
        return pd.DataFrame(
            self.signature_matrix.copy(),
            index=self.genes_used,
            columns=self.cell_types,
        )

    def transformed_bulk_dataframe(self) -> pd.DataFrame:
        """Return domain-aligned bulk expression as ``samples x genes_used``."""
        return pd.DataFrame(
            self.transformed_bulk.copy(),
            index=self.sample_names,
            columns=self.genes_used,
        )

    def summary(self) -> str:
        """Return a concise multi-line summary of the deconvolution run."""
        mean_props = np.nanmean(self.bulk_props, axis=0)
        prop_str = ", ".join(
            f"{ct}={val:.3f}" for ct, val in zip(self.cell_types, mean_props)
        )
        valid_rnorm = self.rnorm[np.isfinite(self.rnorm)]
        mean_rnorm = float(np.mean(valid_rnorm)) if valid_rnorm.size > 0 else float("nan")
        return "\n".join(
            [
                "ReferenceDeconvolutionResult",
                f"  Mode:                 {self.mode}",
                f"  Bulk samples:         {len(self.sample_names)}",
                f"  SC donors:            {len(self.sc_subject_names)}",
                f"  Overlapping subjects: {len(self.overlapping_subjects)}",
                f"  Cell types ({len(self.cell_types)}):      {', '.join(self.cell_types)}",
                f"  Genes used:           {len(self.genes_used)}",
                f"  Mean NNLS residual:   {mean_rnorm:.4g}",
                f"  Mean bulk props:      {prop_str}",
            ]
        )


def _compute_sc_reference_core(
    sc_matrix: np.ndarray | sparse.csr_matrix,
    gene_names: list[str],
    ct_labels: np.ndarray,
    subj_labels: np.ndarray,
    select_ct: Sequence[str] | None = None,
) -> SCReference:
    """Compute signature matrix Z, donor proportions P_sc, and pseudobulk Y_sc."""
    available_ct = set(ct_labels.tolist())
    if select_ct is not None:
        ordered_ct = list(dict.fromkeys(str(c) for c in select_ct))
        if len(ordered_ct) == 0:
            raise ValueError("`select_ct` must contain at least one cell type.")
        missing_ct = [c for c in ordered_ct if c not in available_ct]
        if missing_ct:
            raise ValueError(
                f"Requested cell types in `select_ct` not found in single-cell data: {missing_ct}"
            )
        keep_cells = np.isin(ct_labels, ordered_ct)
        sc_matrix = sc_matrix[keep_cells, :]
        ct_labels = ct_labels[keep_cells]
        subj_labels = subj_labels[keep_cells]
    else:
        ordered_ct = sorted(available_ct)

    ordered_subjects = sorted(set(subj_labels.tolist()))
    n_cells = len(ct_labels)
    n_ct = len(ordered_ct)
    n_subj = len(ordered_subjects)
    if n_cells == 0 or n_ct == 0 or n_subj == 0:
        raise ValueError("No valid cells remain after cell-type filtering.")

    ct_to_idx = {ct: k for k, ct in enumerate(ordered_ct)}
    subj_to_idx = {s: j for j, s in enumerate(ordered_subjects)}
    ct_idx = np.array([ct_to_idx[c] for c in ct_labels], dtype=np.intp)
    subj_idx = np.array([subj_to_idx[s] for s in subj_labels], dtype=np.intp)

    # Signature matrix Z (G x K): mean expression per cell type (sparse-safe)
    counts_per_ct = np.bincount(ct_idx, minlength=n_ct).astype(np.float64)
    ind = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float64), (ct_idx, np.arange(n_cells, dtype=np.intp))),
        shape=(n_ct, n_cells),
    )
    ct_sums = ind @ sc_matrix
    if sparse.issparse(ct_sums):
        ct_sums = ct_sums.toarray()
    ct_means = np.asarray(ct_sums, dtype=np.float64) / counts_per_ct[:, None]
    Z = ct_means.T  # (n_genes, n_ct)

    # Donor proportions P_sc (D x K)
    p_counts = np.zeros((n_subj, n_ct), dtype=np.float64)
    np.add.at(p_counts, (subj_idx, ct_idx), 1.0)
    P_sc = p_counts / p_counts.sum(axis=1, keepdims=True)

    # Synthetic single-cell pseudobulk Y_sc (D x G)
    Y_sc = P_sc @ Z.T

    return SCReference(
        signature_matrix=Z,
        sc_props=P_sc,
        pseudobulk=Y_sc,
        gene_names=list(gene_names),
        cell_types=ordered_ct,
        subject_names=ordered_subjects,
    )


def generate_sc_reference(
    sc: ExpressionInput,
    *,
    cell_types: str | Sequence[str] | np.ndarray = "cell_type",
    subject_names: str | Sequence[str] | np.ndarray | None = "subject_id",
    select_ct: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    normalize_cpm: bool = True,
    old_cpm: bool | None = None,
    sc_layer: str | None = None,
    sc_gene_names: Sequence[str] | None = None,
    settings: SettingsMode = "default",
) -> SCReference:
    """
    Build single-cell signature matrix, donor cell proportions, and pseudobulk.
    """
    eff_mode = validate_settings(settings)
    eff_old_cpm = (True if eff_mode == "R_default" else False) if old_cpm is None else bool(old_cpm)
    parsed_sc: ParsedExpression = parse_expression_input(
        sc,
        name="sc",
        layer=sc_layer,
        gene_names=sc_gene_names,
        obs_prefix="cell",
        allow_1d=False,
        require_nonnegative=normalize_cpm,
    )
    valid_mask, ct_valid, subj_valid = resolve_sc_annotations(
        parsed_sc, cell_types=cell_types, subject_names=subject_names
    )

    sc_mat = parsed_sc.matrix[valid_mask, :]
    sc_genes = pd.Index(parsed_sc.var_names)

    if markers is not None:
        marker_idx = pd.Index(dict.fromkeys(str(m) for m in markers))
        target_genes = sc_genes.intersection(marker_idx, sort=False)
        if len(target_genes) == 0:
            raise ValueError("No markers found in single-cell gene names.")
    else:
        target_genes = sc_genes

    col_idx = sc_genes.get_indexer(target_genes)

    if normalize_cpm and not eff_old_cpm:
        sc_mat = cpm_normalize(sc_mat)
        sc_mat = sc_mat[:, col_idx]
    elif normalize_cpm and eff_old_cpm:
        sc_mat = cpm_normalize(sc_mat[:, col_idx])
    else:
        sc_mat = sc_mat[:, col_idx]

    return _compute_sc_reference_core(
        sc_matrix=sc_mat,
        gene_names=list(target_genes),
        ct_labels=ct_valid,
        subj_labels=subj_valid,
        select_ct=select_ct,
    )


def reference_based_decomposition(
    bulk: ExpressionInput,
    sc: ExpressionInput,
    *,
    cell_types: str | Sequence[str] | np.ndarray = "cell_type",
    subject_names: str | Sequence[str] | np.ndarray | None = "subject_id",
    markers: Sequence[str] | None = None,
    select_ct: Sequence[str] | None = None,
    use_overlap: bool | None = None,
    old_cpm: bool | None = None,
    normalize_cpm: bool = True,
    bulk_layer: str | None = None,
    sc_layer: str | None = None,
    bulk_sample_names: Sequence[str] | None = None,
    bulk_gene_names: Sequence[str] | None = None,
    sc_gene_names: Sequence[str] | None = None,
    settings: SettingsMode = "default",
) -> ReferenceDeconvolutionResult:
    """
    Estimate cell-type proportions in bulk expression via Bisque reference alignment.
    """
    eff_mode = validate_settings(settings)
    eff_old_cpm = (True if eff_mode == "R_default" else False) if old_cpm is None else bool(old_cpm)
    parsed_bulk = parse_expression_input(
        bulk,
        name="bulk",
        layer=bulk_layer,
        sample_names=bulk_sample_names,
        gene_names=bulk_gene_names,
        obs_prefix="sample",
        allow_1d=True,
        require_nonnegative=normalize_cpm,
    )
    parsed_sc = parse_expression_input(
        sc,
        name="sc",
        layer=sc_layer,
        gene_names=sc_gene_names,
        obs_prefix="cell",
        allow_1d=False,
        require_nonnegative=normalize_cpm,
    )

    valid_mask, ct_valid, subj_valid = resolve_sc_annotations(
        parsed_sc, cell_types=cell_types, subject_names=subject_names
    )
    sc_mat = parsed_sc.matrix[valid_mask, :]
    bulk_mat = parsed_bulk.matrix

    bulk_idx_genes = pd.Index(parsed_bulk.var_names)
    sc_idx_genes = pd.Index(parsed_sc.var_names)
    common_genes = bulk_idx_genes.intersection(sc_idx_genes, sort=False)

    if markers is not None:
        marker_idx = pd.Index(dict.fromkeys(str(m) for m in markers))
        common_genes = common_genes.intersection(marker_idx, sort=False)

    if len(common_genes) == 0:
        raise ValueError(
            "No common genes found between bulk and single-cell expression matrices"
            + (" (after marker filtering)." if markers is not None else ".")
        )

    bulk_col_idx = bulk_idx_genes.get_indexer(common_genes)
    sc_col_idx = sc_idx_genes.get_indexer(common_genes)

    if normalize_cpm and not eff_old_cpm:
        bulk_norm = cpm_normalize(bulk_mat)
        sc_norm = cpm_normalize(sc_mat)
        B = subset_columns_dense(bulk_norm, bulk_col_idx)
        sc_sub = sc_norm[:, sc_col_idx]
    elif normalize_cpm and eff_old_cpm:
        B_sub = subset_columns_dense(bulk_mat, bulk_col_idx)
        B = np.asarray(cpm_normalize(B_sub), dtype=np.float64)
        sc_sub = cpm_normalize(sc_mat[:, sc_col_idx])
    else:
        B = subset_columns_dense(bulk_mat, bulk_col_idx)
        sc_sub = sc_mat[:, sc_col_idx]

    sc_ref = _compute_sc_reference_core(
        sc_matrix=sc_sub,
        gene_names=list(common_genes),
        ct_labels=ct_valid,
        subj_labels=subj_valid,
        select_ct=select_ct,
    )

    Z = sc_ref.signature_matrix  # (G_common, K)
    P_sc = sc_ref.sc_props  # (D, K)
    Y_sc = sc_ref.pseudobulk  # (D, G_common)

    n_bulk = B.shape[0]
    n_donors = Y_sc.shape[0]

    # Zero-variance / zero-expression gene filtering
    keep = np.sum(Z, axis=1) > 0.0
    if n_bulk >= 2:
        bulk_sd_all = np.std(B, axis=0, ddof=1)
        keep &= bulk_sd_all > 1e-12
    else:
        keep &= B[0, :] > 0.0

    if n_donors >= 2:
        sc_sd_all = np.std(Y_sc, axis=0, ddof=1)
        keep &= sc_sd_all > 1e-12

    if not np.any(keep):
        raise ValueError("No informative common genes with non-zero variance remain.")

    genes_used = [g for g, flag in zip(common_genes, keep) if flag]
    Z = Z[keep, :]
    B = B[:, keep]
    Y_sc = Y_sc[:, keep]

    sc_subj_to_idx = {s: j for j, s in enumerate(sc_ref.subject_names)}
    overlap_pairs = [
        (i, sc_subj_to_idx[s], s)
        for i, s in enumerate(parsed_bulk.obs_names)
        if s in sc_subj_to_idx
    ]
    if use_overlap is None:
        eff_use_overlap = (eff_mode == "R_default") and (len(overlap_pairs) > 0)
    else:
        eff_use_overlap = bool(use_overlap)

    # Gene-specific affine domain transformation
    overlapping_subjects: list[str] = []
    run_non_overlap = not eff_use_overlap
    sample_names_out = list(parsed_bulk.obs_names)

    if eff_use_overlap:
        overlapping_subjects = [s for _, _, s in overlap_pairs]
        if len(overlap_pairs) == 0:
            raise ValueError(
                "use_overlap=True requires overlapping subject IDs between bulk sample names "
                "and single-cell subject names, but found 0 overlapping subjects."
            )
        if len(overlap_pairs) == 1:
            warnings.warn(
                "Only 1 overlapping subject found between bulk and single-cell data; "
                "falling back to non-overlapping alignment.",
                UserWarning,
                stacklevel=2,
            )
            run_non_overlap = True
        else:
            mode = "overlapping"
            b_idx = np.array([i for i, _, _ in overlap_pairs], dtype=np.intp)
            s_idx = np.array([j for _, j, _ in overlap_pairs], dtype=np.intp)
            X_train = B[b_idx, :]
            Y_train = Y_sc[s_idx, :]

            mu_x = np.mean(X_train, axis=0)
            sd_x = np.std(X_train, axis=0, ddof=1)
            mu_y = np.mean(Y_train, axis=0)
            sd_y = np.std(Y_train, axis=0, ddof=1)

            valid_paired = (sd_x > 1e-12) & (sd_y > 1e-12)

            if eff_mode == "R_default":
                # BisqueRNA v1.0.5 SupervisedTransformBulk: drop anyNA zero-variance training genes
                # and deconvolve remaining non-overlapping bulk samples (when remaining is non-empty)
                if not np.any(valid_paired):
                    raise ValueError(
                        "No informative common genes remain after dropping zero-variance overlapping training genes."
                    )
                b_set = set(int(i) for i in b_idx)
                rem_list = [i for i in range(n_bulk) if i not in b_set]
                eval_idx = (
                    np.asarray(rem_list, dtype=np.intp)
                    if len(rem_list) > 0
                    else np.arange(n_bulk, dtype=np.intp)
                )
                sample_names_out = [parsed_bulk.obs_names[int(i)] for i in eval_idx]
                genes_used = [g for g, flag in zip(genes_used, valid_paired) if flag]
                Z = Z[valid_paired, :]
                x_z = (X_train[:, valid_paired] - mu_x[valid_paired]) / sd_x[valid_paired]
                y_z = (Y_train[:, valid_paired] - mu_y[valid_paired]) / sd_y[valid_paired]
                denom = np.sum(x_z * x_z, axis=0)
                beta_hat = np.sum(x_z * y_z, axis=0) / denom
                B_eval = B[eval_idx, :][:, valid_paired]
                B_tilde = (
                    mu_y[valid_paired]
                    + sd_y[valid_paired]
                    * beta_hat
                    * ((B_eval - mu_x[valid_paired]) / sd_x[valid_paired])
                )
            else:
                B_tilde = np.empty_like(B, dtype=np.float64)
                if np.any(valid_paired):
                    x_z = (X_train[:, valid_paired] - mu_x[valid_paired]) / sd_x[valid_paired]
                    y_z = (Y_train[:, valid_paired] - mu_y[valid_paired]) / sd_y[valid_paired]
                    denom = np.sum(x_z * x_z, axis=0)
                    beta_hat = np.sum(x_z * y_z, axis=0) / denom
                    B_tilde[:, valid_paired] = (
                        mu_y[valid_paired]
                        + sd_y[valid_paired]
                        * beta_hat
                        * ((B[:, valid_paired] - mu_x[valid_paired]) / sd_x[valid_paired])
                    )

                if np.any(~valid_paired):
                    inv = ~valid_paired
                    if n_donors >= 2 and n_bulk >= 2:
                        mu_sc_inv = np.mean(Y_sc[:, inv], axis=0)
                        sig_sc_inv = np.sqrt(
                            np.sum((Y_sc[:, inv] - mu_sc_inv) ** 2, axis=0) / (n_donors + 1.0)
                        )
                        mu_b_inv = np.mean(B[:, inv], axis=0)
                        sd_b_inv = np.std(B[:, inv], axis=0, ddof=1)
                        B_tilde[:, inv] = mu_sc_inv + sig_sc_inv * (
                            (B[:, inv] - mu_b_inv) / sd_b_inv
                        )
                    else:
                        B_tilde[:, inv] = mu_y[inv]

    if run_non_overlap:
        if n_donors < 2 or n_bulk < 2:
            warnings.warn(
                f"Cannot perform affine variance alignment with n_donors={n_donors} and "
                f"n_bulk={n_bulk} (requires >= 2 for both); using untransformed NNLS fallback.",
                UserWarning,
                stacklevel=2,
            )
            mode = "untransformed_fallback"
            B_tilde = B.copy()
        else:
            mode = "non_overlapping"
            mu_sc = np.mean(Y_sc, axis=0)
            sigma_sc_tilde = np.sqrt(
                np.sum((Y_sc - mu_sc) ** 2, axis=0) / (n_donors + 1.0)
            )
            mu_bulk = np.mean(B, axis=0)
            sd_bulk = np.std(B, axis=0, ddof=1)
            B_tilde = mu_sc + sigma_sc_tilde * ((B - mu_bulk) / sd_bulk)

    n_eval = B_tilde.shape[0]
    n_ct = len(sc_ref.cell_types)
    bulk_props = np.empty((n_eval, n_ct), dtype=np.float64)
    rnorm = np.empty(n_eval, dtype=np.float64)

    if eff_mode == "R_default":
        # Exact constrained simplex QP matching limSolve::lsei in BisqueRNA::Deconvolve
        G_mat = Z.T @ Z
        for i in range(n_eval):
            p_hat, res_norm = _solve_simplex_qp(G_mat, Z, B_tilde[i, :])
            bulk_props[i, :] = p_hat
            rnorm[i] = res_norm
    else:
        for i in range(n_eval):
            q_hat, res_norm = nnls(Z, B_tilde[i, :])
            rnorm[i] = float(res_norm)
            q_sum = float(np.sum(q_hat))
            if q_sum > 1e-15:
                bulk_props[i, :] = q_hat / q_sum
            else:
                warnings.warn(
                    f"NNLS returned zero-sum weights for bulk sample '{sample_names_out[i]}'; "
                    "setting estimated proportions to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                bulk_props[i, :] = np.nan

    return ReferenceDeconvolutionResult(
        bulk_props=bulk_props,
        sc_props=P_sc,
        sample_names=sample_names_out,
        sc_subject_names=sc_ref.subject_names,
        cell_types=sc_ref.cell_types,
        genes_used=genes_used,
        rnorm=rnorm,
        signature_matrix=Z,
        transformed_bulk=B_tilde,
        mode=mode,
        overlapping_subjects=overlapping_subjects,
    )
