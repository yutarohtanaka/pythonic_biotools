"""
Cross-platform Empirical Bayes ComBat batch alignment (`combat_align`),
Bulk-mode (`B-mode`), and Single-cell-mode (`S-mode`) for `cibersortx`
(Johnson et al., Biostatistics 2007; Newman et al., Nature Biotechnology 2019).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
import warnings

import numpy as np
import pandas as pd

from ._io import (
    ExpressionInput,
    align_genes,
    parse_expression_matrix,
    parse_label_vector,
    parse_sc_reference_sparse_aware,
    parse_signature_matrix,
)

if TYPE_CHECKING:
    from .deconvolve import DeconvolutionResult


@dataclass(frozen=True)
class BatchAlignmentResult:
    """Typed result container for `combat_align`."""

    adjusted_target: np.ndarray
    adjusted_reference: np.ndarray | None
    gene_names: list[str]
    gamma_target: np.ndarray
    delta2_target: np.ndarray
    gamma_ref: np.ndarray
    delta2_ref: np.ndarray
    space: str
    target_names: list[str] | None = None
    reference_names: list[str] | None = None
    alpha_hat: np.ndarray | None = None
    sigma_hat: np.ndarray | None = None
    zero_var_mask: np.ndarray | None = None
    mean_target: np.ndarray | None = None
    mean_ref: np.ndarray | None = None

    def to_dataframe(self, which: Literal["target", "reference"] = "target") -> pd.DataFrame:
        """Return aligned `(G, N)` expression matrix as a pandas DataFrame."""
        if which == "target":
            return pd.DataFrame(
                self.adjusted_target,
                index=self.gene_names,
                columns=self.target_names,
            )
        if which == "reference":
            if self.adjusted_reference is None:
                raise ValueError(
                    "`adjusted_reference` is only populated when `space='common'`."
                )
            return pd.DataFrame(
                self.adjusted_reference,
                index=self.gene_names,
                columns=self.reference_names,
            )
        raise ValueError(f"Unsupported `which='{which}'`. Expected 'target' or 'reference'.")


def _fit_eb_batch_params(
    z_std: np.ndarray,
    *,
    mean_only: bool,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Empirical Bayes Normal / Inverse-Gamma shrinkage parameters for one batch."""
    n_genes, n_samples = z_std.shape
    gamma_hat = np.mean(z_std, axis=1)
    gamma_bar = float(np.mean(gamma_hat))
    tau2 = float(np.var(gamma_hat, ddof=1)) if n_genes > 1 else 1.0
    tau2 = max(tau2, 1e-8)

    if n_samples < 2 or mean_only:
        delta2_star = np.ones(n_genes, dtype=np.float64)
        gamma_star = (n_samples * tau2 * gamma_hat + delta2_star * gamma_bar) / (
            n_samples * tau2 + delta2_star
        )
        return gamma_star, delta2_star

    delta2_hat = np.maximum(np.var(z_std, axis=1, ddof=1), 1e-8)
    v_bar = float(np.mean(delta2_hat))
    s2 = float(np.var(delta2_hat, ddof=1)) if n_genes > 1 else 0.0

    if s2 < 1e-10 or v_bar <= 1e-12:
        delta2_star = delta2_hat
        gamma_star = (n_samples * tau2 * gamma_hat + delta2_star * gamma_bar) / (
            n_samples * tau2 + delta2_star
        )
        return gamma_star, delta2_star

    lam = (v_bar ** 2 + 2.0 * s2) / s2
    theta = (v_bar ** 3 + v_bar * s2) / s2

    gamma_star = gamma_hat.copy()
    delta2_star = delta2_hat.copy()
    denom_scale = 0.5 * n_samples + lam - 1.0

    for _ in range(max_iter):
        gamma_new = (n_samples * tau2 * gamma_hat + delta2_star * gamma_bar) / (
            n_samples * tau2 + delta2_star
        )
        sum_sq = (n_samples - 1) * delta2_hat + n_samples * (gamma_hat - gamma_new) ** 2
        delta2_new = np.maximum((theta + 0.5 * sum_sq) / denom_scale, 1e-8)

        diff_g = np.max(np.abs(gamma_new - gamma_star) / np.maximum(np.abs(gamma_star), 1e-8))
        diff_d = np.max(np.abs(delta2_new - delta2_star) / np.maximum(np.abs(delta2_star), 1e-8))
        gamma_star = gamma_new
        delta2_star = delta2_new
        if max(diff_g, diff_d) < tol:
            break

    return gamma_star, delta2_star


def _apply_combat_transform_to_target(
    mat: np.ndarray,
    align: BatchAlignmentResult,
    *,
    is_reference_batch: bool = False,
) -> np.ndarray:
    """Apply learned ComBat transform to an arbitrary `(G, M)` matrix from target or ref batch."""
    assert align.alpha_hat is not None and align.sigma_hat is not None
    assert align.zero_var_mask is not None
    assert align.mean_target is not None and align.mean_ref is not None

    z = np.log2(np.maximum(mat, 0.0) + 1.0)
    z_adj = z.copy()
    active = ~align.zero_var_mask
    zero_var = align.zero_var_mask

    if np.any(active):
        alpha = align.alpha_hat[active, None]
        sigma = align.sigma_hat[active, None]
        z_std = (z[active] - alpha) / sigma
        if is_reference_batch:
            g = align.gamma_ref[active, None]
            d2 = align.delta2_ref[active, None]
        else:
            g = align.gamma_target[active, None]
            d2 = align.delta2_target[active, None]

        if align.space == "reference":
            if is_reference_batch:
                z_adj[active] = z[active]
            else:
                scale = np.sqrt(align.delta2_ref[active, None] / d2)
                z_adj[active] = alpha + sigma * (scale * (z_std - g) + align.gamma_ref[active, None])
        else:  # space == "common"
            z_adj[active] = alpha + (sigma / np.sqrt(d2)) * (z_std - g)

    if np.any(zero_var):
        if align.space == "reference":
            if not is_reference_batch:
                z_adj[zero_var] = (
                    z[zero_var] - align.mean_target[zero_var, None] + align.mean_ref[zero_var, None]
                )
        else:
            z_adj[zero_var] = align.alpha_hat[zero_var, None]

    return np.maximum(np.power(2.0, z_adj) - 1.0, 0.0)


def combat_align(
    target: pd.DataFrame | np.ndarray,
    reference: pd.DataFrame | np.ndarray,
    *,
    space: Literal["reference", "common"] = "reference",
    mean_only: bool = False,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> BatchAlignmentResult:
    """
    Two-batch Empirical Bayes ComBat location-scale alignment in `log2(x + 1)`
    space mapping `target` `(G, N1)` onto `reference` `(G, N2)` (`space='reference'`)
    or both batches onto their pooled grand-mean space (`space='common'`).
    """
    if space not in ("reference", "common"):
        raise ValueError(f"Unsupported ComBat `space='{space}'`. Expected 'reference' or 'common'.")
    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1.")

    y1, genes1, names1 = parse_expression_matrix(
        target,
        default_row_prefix="gene",
        default_col_prefix="target",
        name="target",
    )
    y2, genes2, names2 = parse_expression_matrix(
        reference,
        default_row_prefix="gene",
        default_col_prefix="reference",
        name="reference",
    )

    if isinstance(target, pd.DataFrame) and isinstance(reference, pd.DataFrame):
        shared_genes, idx1, idx2 = align_genes(genes1, genes2)
        if not shared_genes:
            raise ValueError("`target` and `reference` share zero genes.")
        y1 = y1[idx1, :]
        y2 = y2[idx2, :]
        gene_names = shared_genes
    else:
        if y1.shape[0] != y2.shape[0]:
            raise ValueError(
                f"Gene count mismatch between target ({y1.shape[0]}) and reference ({y2.shape[0]})."
            )
        gene_names = genes1 if isinstance(target, pd.DataFrame) else genes2

    n_genes, n1 = y1.shape
    n2 = y2.shape[1]
    n_tot = n1 + n2

    if (n1 < 2 or n2 < 2) and not mean_only:
        warnings.warn(
            f"Batch size < 2 detected (n1={n1}, n2={n2}); using mean-only ComBat adjustment.",
            UserWarning,
            stacklevel=2,
        )

    z1 = np.log2(y1 + 1.0)
    z2 = np.log2(y2 + 1.0)

    mean1 = np.mean(z1, axis=1)
    mean2 = np.mean(z2, axis=1)
    alpha_hat = (n1 * mean1 + n2 * mean2) / n_tot

    ss1 = np.sum((z1 - mean1[:, None]) ** 2, axis=1)
    ss2 = np.sum((z2 - mean2[:, None]) ** 2, axis=1)
    var_pool = (ss1 + ss2) / n_tot
    zero_var = var_pool <= 1e-12
    sigma_hat = np.sqrt(np.maximum(var_pool, 1e-8))

    gamma_target = np.zeros(n_genes, dtype=np.float64)
    delta2_target = np.ones(n_genes, dtype=np.float64)
    gamma_ref = np.zeros(n_genes, dtype=np.float64)
    delta2_ref = np.ones(n_genes, dtype=np.float64)

    active = ~zero_var
    if np.any(active):
        z1_std = (z1[active] - alpha_hat[active, None]) / sigma_hat[active, None]
        z2_std = (z2[active] - alpha_hat[active, None]) / sigma_hat[active, None]
        use_mean_only = bool(mean_only or n1 < 2 or n2 < 2)

        gt, d2t = _fit_eb_batch_params(
            z1_std, mean_only=use_mean_only or (n1 < 2), max_iter=max_iter, tol=tol
        )
        gr, d2r = _fit_eb_batch_params(
            z2_std, mean_only=use_mean_only or (n2 < 2), max_iter=max_iter, tol=tol
        )
        gamma_target[active] = gt
        delta2_target[active] = d2t
        gamma_ref[active] = gr
        delta2_ref[active] = d2r

    res_meta = BatchAlignmentResult(
        adjusted_target=y1,
        adjusted_reference=None,
        gene_names=gene_names,
        gamma_target=gamma_target,
        delta2_target=delta2_target,
        gamma_ref=gamma_ref,
        delta2_ref=delta2_ref,
        space=space,
        target_names=names1,
        reference_names=names2,
        alpha_hat=alpha_hat,
        sigma_hat=sigma_hat,
        zero_var_mask=zero_var,
        mean_target=mean1,
        mean_ref=mean2,
    )

    adj_target = _apply_combat_transform_to_target(y1, res_meta, is_reference_batch=False)
    adj_ref = (
        _apply_combat_transform_to_target(y2, res_meta, is_reference_batch=True)
        if space == "common"
        else None
    )

    return BatchAlignmentResult(
        adjusted_target=adj_target,
        adjusted_reference=adj_ref,
        gene_names=gene_names,
        gamma_target=gamma_target,
        delta2_target=delta2_target,
        gamma_ref=gamma_ref,
        delta2_ref=delta2_ref,
        space=space,
        target_names=names1,
        reference_names=names2,
        alpha_hat=alpha_hat,
        sigma_hat=sigma_hat,
        zero_var_mask=zero_var,
        mean_target=mean1,
        mean_ref=mean2,
    )


def b_mode_adjust(
    mixture: ExpressionInput,
    signature: Any,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    nu_grid: Sequence[float] = (0.25, 0.5, 0.75),
    C: float = 1.0,
    n_permutations: int = 100,
    permutation_mode: Literal["per_sample", "pooled"] = "per_sample",
    batch_space: Literal["reference", "common"] = "reference",
    is_log2: bool = False,
    log2_offset: float = 0.0,
    svr_tol: float = 1e-3,
    svr_max_iter: int = -1,
    absolute_mode: bool = False,
    seed: int = 0,
) -> DeconvolutionResult:
    """
    CIBERSORTx `B-mode` (Bulk-mode) cross-platform batch adjustment:
    1. Initial nu-SVR deconvolution -> fractions F0.
    2. Construct signature-space synthetic mixtures M_tilde = B @ F0.T.
    3. ComBat align observed bulk M -> M_tilde.
    4. Re-deconvolve adjusted bulk M_adj.
    """
    from .deconvolve import DeconvolutionResult, _deconvolve_core

    B_raw, sig_genes, cell_types = parse_signature_matrix(
        signature, is_log2=is_log2, log2_offset=log2_offset
    )
    M_raw, mix_genes, sample_names = parse_expression_matrix(
        mixture,
        layer=layer,
        samples_as_rows=samples_as_rows,
        default_row_prefix="gene",
        default_col_prefix="sample",
        allow_1d=True,
        name="mixture",
        is_log2=is_log2,
        log2_offset=log2_offset,
    )

    shared_genes, idx_sig, idx_mix = align_genes(sig_genes, mix_genes)
    n_types = len(cell_types)
    if len(shared_genes) < max(n_types, 3):
        raise ValueError(
            f"Shared gene count ({len(shared_genes)}) must be >= max(K={n_types}, 3)."
        )

    B_0 = B_raw[idx_sig, :]
    M_0 = M_raw[idx_mix, :]
    n_samples = M_0.shape[1]

    init_res = _deconvolve_core(
        M_0,
        B_0,
        gene_names=shared_genes,
        sample_names=sample_names,
        cell_types=cell_types,
        nu_grid=nu_grid,
        C=C,
        n_permutations=0,
        permutation_mode=permutation_mode,
        svr_tol=svr_tol,
        svr_max_iter=svr_max_iter,
        absolute_mode=absolute_mode,
        seed=seed,
        batch_mode="B",
    )

    valid_samples = np.all(np.isfinite(init_res.fractions), axis=1)
    if int(np.sum(valid_samples)) < 2:
        warnings.warn(
            "B-mode requires at least 2 valid bulk samples; returning unadjusted deconvolution.",
            UserWarning,
            stacklevel=2,
        )
        final_unadj = _deconvolve_core(
            M_0,
            B_0,
            gene_names=shared_genes,
            sample_names=sample_names,
            cell_types=cell_types,
            nu_grid=nu_grid,
            C=C,
            n_permutations=n_permutations,
            permutation_mode=permutation_mode,
            svr_tol=svr_tol,
            svr_max_iter=svr_max_iter,
            absolute_mode=absolute_mode,
            seed=seed,
            batch_mode="B",
        )
        return final_unadj

    M_valid = M_0[:, valid_samples]
    F_valid = init_res.fractions[valid_samples, :]
    M_tilde = B_0 @ F_valid.T

    align_res = combat_align(
        pd.DataFrame(M_valid, index=shared_genes),
        pd.DataFrame(M_tilde, index=shared_genes),
        space=batch_space,
    )

    M_adj = M_0.copy()
    M_adj[:, valid_samples] = align_res.adjusted_target

    if batch_space == "common":
        B_used = _apply_combat_transform_to_target(B_0, align_res, is_reference_batch=True)
        adj_sig: np.ndarray | None = B_used
    else:
        B_used = B_0
        adj_sig = None

    final_res = _deconvolve_core(
        M_adj,
        B_used,
        gene_names=shared_genes,
        sample_names=sample_names,
        cell_types=cell_types,
        nu_grid=nu_grid,
        C=C,
        n_permutations=n_permutations,
        permutation_mode=permutation_mode,
        svr_tol=svr_tol,
        svr_max_iter=svr_max_iter,
        absolute_mode=absolute_mode,
        seed=seed,
        batch_mode="B",
    )

    return DeconvolutionResult(
        fractions=final_res.fractions,
        sample_names=final_res.sample_names,
        cell_types=final_res.cell_types,
        p_value=final_res.p_value,
        pearson_r=final_res.pearson_r,
        rmse=final_res.rmse,
        best_nu=final_res.best_nu,
        n_genes_used=final_res.n_genes_used,
        batch_mode="B",
        adjusted_mixture=M_adj,
        adjusted_signature=adj_sig,
        gene_names_used=final_res.gene_names_used,
    )


def s_mode_adjust(
    mixture: ExpressionInput,
    signature: Any,
    sc_ref: ExpressionInput,
    sc_cell_type_labels: str | Sequence[Any] | np.ndarray | pd.Series,
    *,
    layer: str | None = None,
    sc_layer: str | None = None,
    samples_as_rows: bool = False,
    sc_cells_as_rows: bool = False,
    normalize_sc_cells: bool = True,
    n_synthetic: int = 100,
    cells_per_mixture: int = 100,
    nu_grid: Sequence[float] = (0.25, 0.5, 0.75),
    C: float = 1.0,
    n_permutations: int = 100,
    permutation_mode: Literal["per_sample", "pooled"] = "per_sample",
    batch_space: Literal["reference", "common"] = "reference",
    is_log2: bool = False,
    log2_offset: float = 0.0,
    svr_tol: float = 1e-3,
    svr_max_iter: int = -1,
    absolute_mode: bool = False,
    seed: int = 0,
) -> DeconvolutionResult:
    """
    CIBERSORTx `S-mode` (Single-cell mode) cross-platform batch adjustment:
    1. Generate `n_synthetic` Dirichlet pseudobulk mixtures from `sc_ref`.
    2. Rescale synthetic mixtures to match library size of `signature` B_S.
    3. ComBat align synthetic mixtures (sc batch) -> bulk mixture M.
    4. Transform signature B_S using learned ComBat parameters and deconvolve.
    """
    from .deconvolve import DeconvolutionResult, _deconvolve_core

    if n_synthetic < 2:
        raise ValueError("`n_synthetic` must be >= 2.")
    if cells_per_mixture < 1:
        raise ValueError("`cells_per_mixture` must be >= 1.")

    B_raw, sig_genes, cell_types = parse_signature_matrix(
        signature, is_log2=is_log2, log2_offset=log2_offset
    )
    M_raw, mix_genes, sample_names = parse_expression_matrix(
        mixture,
        layer=layer,
        samples_as_rows=samples_as_rows,
        default_row_prefix="gene",
        default_col_prefix="sample",
        allow_1d=True,
        name="mixture",
        is_log2=is_log2,
        log2_offset=log2_offset,
    )
    S_raw, sc_genes, _sc_cell_names, is_sparse = parse_sc_reference_sparse_aware(
        sc_ref,
        layer=sc_layer,
        cells_as_rows=sc_cells_as_rows,
        is_log2=is_log2,
        log2_offset=log2_offset,
    )
    sc_labels = parse_label_vector(
        sc_ref, sc_cell_type_labels, n_obs=S_raw.shape[0], kind="sc_cell_type_labels"
    )

    # Intersect genes across B_S, M, and S preserving B_S gene order
    shared_bm, _, _ = align_genes(sig_genes, mix_genes)
    shared_genes, idx_sig, idx_sc = align_genes(shared_bm, sc_genes)
    if len(shared_genes) < max(len(cell_types), 3):
        raise ValueError(
            f"Shared gene count across signature, mixture, and sc_ref ({len(shared_genes)}) "
            f"must be >= max(K={len(cell_types)}, 3)."
        )
    sig_idx_map = {g: i for i, g in enumerate(sig_genes)}
    mix_idx_map = {g: i for i, g in enumerate(mix_genes)}
    idx_sig_final = np.array([sig_idx_map[g] for g in shared_genes], dtype=int)
    idx_mix_final = np.array([mix_idx_map[g] for g in shared_genes], dtype=int)

    B_0 = B_raw[idx_sig_final, :]
    M_0 = M_raw[idx_mix_final, :]

    # Normalize sc_ref cells on full gene set before subsetting to shared genes (if requested)
    if is_sparse:
        cell_sums = np.asarray(S_raw.sum(axis=1)).ravel()
    else:
        cell_sums = np.sum(S_raw, axis=1)
    nonzero_cells = cell_sums > 0.0
    if not np.any(nonzero_cells):
        raise ValueError("All `sc_ref` cells have zero total expression.")
    if not np.all(nonzero_cells):
        S_raw = S_raw[nonzero_cells]
        sc_labels = [lbl for lbl, keep in zip(sc_labels, nonzero_cells) if keep]
        cell_sums = cell_sums[nonzero_cells]

    S_sub = S_raw[:, idx_sc]
    S_dense = S_sub.toarray() if is_sparse else np.asarray(S_sub, dtype=np.float64)
    if normalize_sc_cells:
        S_dense = S_dense * (1e4 / cell_sums[:, None])

    sc_label_arr = np.asarray(sc_labels, dtype=object)
    ct_indices: list[np.ndarray] = []
    for ct in cell_types:
        idx = np.flatnonzero(sc_label_arr == ct)
        if len(idx) == 0:
            raise ValueError(
                f"Signature cell type '{ct}' is absent from `sc_cell_type_labels`."
            )
        ct_indices.append(idx)

    # Generate Dirichlet-weighted synthetic pseudobulk mixtures
    rng = np.random.default_rng(seed)
    n_genes = len(shared_genes)
    n_types = len(cell_types)
    M_synth = np.empty((n_genes, n_synthetic), dtype=np.float64)

    for m in range(n_synthetic):
        pi = rng.dirichlet(np.ones(n_types, dtype=np.float64))
        counts = rng.multinomial(cells_per_mixture, pi)
        # Ensure at least 1 cell for positive Dirichlet weights when pool >= K
        if cells_per_mixture >= n_types and np.any(counts == 0):
            zero_k = np.flatnonzero(counts == 0)
            for k_z in zero_k:
                donor_k = int(np.argmax(counts))
                if counts[donor_k] > 1:
                    counts[donor_k] -= 1
                    counts[k_z] += 1
        chosen_cells: list[np.ndarray] = []
        for k_idx, cnt in enumerate(counts):
            if cnt > 0:
                pool = ct_indices[k_idx]
                chosen_cells.append(pool[rng.integers(0, len(pool), size=int(cnt))])
        all_chosen = np.concatenate(chosen_cells)
        M_synth[:, m] = np.mean(S_dense[all_chosen, :], axis=0)

    # Rescale synthetic mixtures to match library sum of signature matrix B_0
    med_b = float(np.median(np.sum(B_0, axis=0)))
    med_synth = float(np.median(np.sum(M_synth, axis=0)))
    if med_b > 0.0 and med_synth > 0.0:
        M_synth *= med_b / med_synth

    align_res = combat_align(
        pd.DataFrame(M_synth, index=shared_genes),
        pd.DataFrame(M_0, index=shared_genes),
        space=batch_space,
    )

    B_adj = _apply_combat_transform_to_target(B_0, align_res, is_reference_batch=False)
    if batch_space == "common":
        assert align_res.adjusted_reference is not None
        M_used = align_res.adjusted_reference
        adj_mix: np.ndarray | None = M_used
    else:
        M_used = M_0
        adj_mix = None

    final_res = _deconvolve_core(
        M_used,
        B_adj,
        gene_names=shared_genes,
        sample_names=sample_names,
        cell_types=cell_types,
        nu_grid=nu_grid,
        C=C,
        n_permutations=n_permutations,
        permutation_mode=permutation_mode,
        svr_tol=svr_tol,
        svr_max_iter=svr_max_iter,
        absolute_mode=absolute_mode,
        seed=seed,
        batch_mode="S",
    )

    return DeconvolutionResult(
        fractions=final_res.fractions,
        sample_names=final_res.sample_names,
        cell_types=final_res.cell_types,
        p_value=final_res.p_value,
        pearson_r=final_res.pearson_r,
        rmse=final_res.rmse,
        best_nu=final_res.best_nu,
        n_genes_used=final_res.n_genes_used,
        batch_mode="S",
        adjusted_mixture=adj_mix,
        adjusted_signature=B_adj,
        gene_names_used=final_res.gene_names_used,
    )
