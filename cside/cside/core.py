"""Core pipeline, result containers, and convenience wrappers for CSIDE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2, norm

from .likelihood import fit_gene_cside
from .splines import SettingsMode, build_spline_design, validate_settings


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values while preserving NaNs."""
    pvals = np.asarray(pvals, dtype=np.float64)
    qvals = np.full_like(pvals, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)
    m = int(np.sum(valid))
    if m == 0:
        return qvals
    pv = np.clip(pvals[valid], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = ranked * (m / ranks)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    qvals[valid] = q_sorted[inv_order]
    return qvals


@dataclass
class CSIDEConfig:
    """Resolved parameter configuration for CSIDE differential expression."""

    settings: SettingsMode = "default"
    doublet_mode: bool = False
    weight_threshold: float | None = None
    cell_type_threshold: float | None = None
    gene_threshold: float | None = None
    fdr: float | None = None
    log_fc_thresh: float | None = None

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if self.settings == "default":
            if self.cell_type_threshold is None:
                self.cell_type_threshold = 15.0
            if self.weight_threshold is None:
                self.weight_threshold = 0.75
            if self.gene_threshold is None:
                self.gene_threshold = 1e-5
            if self.fdr is None:
                self.fdr = 0.05
            if self.log_fc_thresh is None:
                self.log_fc_thresh = 0.25
        else:
            # spacexr::run.CSIDE R parity defaults
            if self.cell_type_threshold is None:
                self.cell_type_threshold = 125.0
            if self.weight_threshold is None:
                self.weight_threshold = 0.99 if self.doublet_mode else 0.8
            if self.gene_threshold is None:
                self.gene_threshold = 5e-5
            if self.fdr is None:
                self.fdr = 0.01
            if self.log_fc_thresh is None:
                self.log_fc_thresh = 0.4


@dataclass
class CSIDEResult:
    """Container for CSIDE cell-type-specific spatial DE results."""

    all_gene_list: dict[str, pd.DataFrame]
    sig_gene_list: dict[str, pd.DataFrame]
    sigma_gene: pd.Series
    design_matrix: pd.DataFrame
    cell_types: list[str]
    coefficients: dict[str, pd.DataFrame] = field(default_factory=dict)
    settings: SettingsMode = "default"
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise text summary of the CSIDE model fit."""
        n_spots, n_cov = self.design_matrix.shape
        n_genes = len(self.sigma_gene)
        med_sig = float(self.sigma_gene.median()) if n_genes > 0 else float("nan")
        lines = [
            f"CSIDEResult (settings={self.settings!r})",
            f"  Spots retained : {n_spots}",
            f"  Design columns : {n_cov} ({', '.join(map(str, self.design_matrix.columns[:6]))}"
            + (", ...)" if n_cov > 6 else ")"),
            f"  Genes tested   : {n_genes} (median sigma_g = {med_sig:.3f})",
            f"  Cell types ({len(self.cell_types)}):",
        ]
        for ct in self.cell_types:
            n_sig = len(self.sig_gene_list.get(ct, ()))
            n_tot = len(self.all_gene_list.get(ct, ()))
            lines.append(f"    - {ct}: {n_sig} / {n_tot} significant DE genes")
        return "\n".join(lines)

    def to_dataframe(self, significant_only: bool = False) -> pd.DataFrame:
        """Combine per-cell-type results into a tidy long-form DataFrame."""
        source = self.sig_gene_list if significant_only else self.all_gene_list
        frames: list[pd.DataFrame] = []
        for ct in self.cell_types:
            df_ct = source.get(ct)
            if df_ct is None or df_ct.empty:
                continue
            sub = df_ct.copy()
            sub.insert(0, "gene", sub.index.astype(str))
            sub.insert(0, "cell_type", ct)
            sub["sigma_gene"] = self.sigma_gene.reindex(sub["gene"]).to_numpy()
            frames.append(sub.reset_index(drop=True))
        if not frames:
            return pd.DataFrame(
                columns=[
                    "cell_type",
                    "gene",
                    "log_fc",
                    "se",
                    "z_stat",
                    "p_val",
                    "q_val",
                    "significant",
                    "converged",
                    "sigma_gene",
                ]
            )
        return pd.concat(frames, ignore_index=True)


def _extract_inputs(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    cell_type_weights: pd.DataFrame | np.ndarray | None,
    n_umi: pd.Series | np.ndarray | None,
) -> tuple[
    np.ndarray | sparse.csr_matrix,
    pd.Index,
    pd.Index,
    pd.DataFrame,
    np.ndarray,
]:
    """Normalize counts, weights, and library sizes across input types."""
    if isinstance(counts, ad.AnnData):
        adata = counts
        spot_index = pd.Index(adata.obs_names, name="spot")
        gene_index = pd.Index(adata.var_names, name="gene")
        x_raw = adata.X
        if sparse.issparse(x_raw):
            x_mat: np.ndarray | sparse.csr_matrix = x_raw.tocsr()
        else:
            x_mat = np.asarray(x_raw, dtype=np.float64)

        if cell_type_weights is None:
            for key in ("cell_type_weights", "rctd_weights", "weights"):
                if key in adata.obsm:
                    cell_type_weights = adata.obsm[key]
                    break
            if cell_type_weights is None:
                raise ValueError(
                    "cell_type_weights was None and not found in adata.obsm['cell_type_weights']."
                )
        if isinstance(cell_type_weights, pd.DataFrame):
            w_df = cell_type_weights.reindex(spot_index)
        else:
            w_arr = np.asarray(cell_type_weights, dtype=np.float64)
            ct_names = list(
                adata.uns.get(
                    "cell_type_names",
                    adata.uns.get(
                        "cell_types",
                        [f"cell_type_{k}" for k in range(w_arr.shape[1])],
                    ),
                )
            )
            w_df = pd.DataFrame(w_arr, index=spot_index, columns=ct_names)

        if n_umi is None:
            if "n_umi" in adata.obs:
                umi_vec = np.asarray(adata.obs["n_umi"], dtype=np.float64)
            elif "total_counts" in adata.obs:
                umi_vec = np.asarray(adata.obs["total_counts"], dtype=np.float64)
            else:
                umi_vec = np.asarray(x_mat.sum(axis=1)).ravel().astype(np.float64)
        elif isinstance(n_umi, pd.Series):
            umi_vec = n_umi.reindex(spot_index).to_numpy(dtype=np.float64)
        else:
            umi_vec = np.asarray(n_umi, dtype=np.float64).ravel()

        return x_mat, spot_index, gene_index, w_df, umi_vec

    # Non-AnnData inputs
    if cell_type_weights is None:
        raise ValueError("cell_type_weights must be provided when counts is not AnnData.")

    if isinstance(cell_type_weights, pd.DataFrame):
        w_df = cell_type_weights.copy()
        spot_index = w_df.index
    else:
        w_arr = np.asarray(cell_type_weights, dtype=np.float64)
        if w_arr.ndim != 2:
            raise ValueError(f"cell_type_weights must be 2D; got {w_arr.shape}.")
        spot_index = pd.RangeIndex(w_arr.shape[0], name="spot")
        w_df = pd.DataFrame(
            w_arr,
            index=spot_index,
            columns=[f"cell_type_{k}" for k in range(w_arr.shape[1])],
        )

    n_spots_w = len(w_df)
    if isinstance(counts, pd.DataFrame):
        # Support either (spots x genes) or R-style (genes x spots) DataFrame
        if counts.shape[0] != n_spots_w and counts.shape[1] == n_spots_w:
            counts = counts.T
        if counts.shape[0] != n_spots_w:
            raise ValueError(
                f"counts shape {counts.shape} is incompatible with {n_spots_w} spots."
            )
        if counts.index.equals(w_df.index):
            spot_index = counts.index
        else:
            w_df.index = counts.index
            spot_index = counts.index
        gene_index = pd.Index(counts.columns, name="gene")
        x_mat = counts.to_numpy(dtype=np.float64)
    elif sparse.issparse(counts):
        if counts.shape[0] != n_spots_w and counts.shape[1] == n_spots_w:
            counts = counts.T
        if counts.shape[0] != n_spots_w:
            raise ValueError(
                f"counts shape {counts.shape} is incompatible with {n_spots_w} spots."
            )
        x_mat = counts.tocsr()
        gene_index = pd.Index([f"gene_{g}" for g in range(x_mat.shape[1])], name="gene")
    else:
        arr = np.asarray(counts, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"counts must be 2D; got {arr.shape}.")
        if arr.shape[0] != n_spots_w and arr.shape[1] == n_spots_w:
            arr = arr.T
        if arr.shape[0] != n_spots_w:
            raise ValueError(
                f"counts shape {arr.shape} is incompatible with {n_spots_w} spots."
            )
        x_mat = arr
        gene_index = pd.Index([f"gene_{g}" for g in range(arr.shape[1])], name="gene")

    if n_umi is None:
        umi_vec = np.asarray(x_mat.sum(axis=1)).ravel().astype(np.float64)
    elif isinstance(n_umi, pd.Series):
        umi_vec = n_umi.reindex(spot_index).to_numpy(dtype=np.float64)
    else:
        umi_vec = np.asarray(n_umi, dtype=np.float64).ravel()

    return x_mat, spot_index, gene_index, w_df, umi_vec


def _prepare_design_matrix(
    explanatory_variable: pd.DataFrame | pd.Series | np.ndarray | dict[Any, float],
    spot_index: pd.Index,
    add_intercept: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert ``explanatory_variable`` into a spot-aligned design DataFrame."""
    n_spots = len(spot_index)
    if isinstance(explanatory_variable, dict):
        s = pd.Series(explanatory_variable).reindex(spot_index)
        vals = s.to_numpy(dtype=np.float64)
        design_df = pd.DataFrame(
            {"Intercept": np.ones(n_spots, dtype=np.float64), "explanatory": vals},
            index=spot_index,
        )
        return design_df, np.array([True, False], dtype=bool)

    if isinstance(explanatory_variable, pd.Series):
        if explanatory_variable.index.isin(spot_index).all() and len(explanatory_variable) == n_spots:
            s = explanatory_variable.reindex(spot_index)
        else:
            s = pd.Series(explanatory_variable.to_numpy(dtype=np.float64), index=spot_index)
        col_name = str(explanatory_variable.name or "explanatory")
        design_df = pd.DataFrame(
            {"Intercept": np.ones(n_spots, dtype=np.float64), col_name: s.to_numpy(dtype=np.float64)},
            index=spot_index,
        )
        return design_df, np.array([True, False], dtype=bool)

    if isinstance(explanatory_variable, pd.DataFrame):
        if len(explanatory_variable) != n_spots:
            raise ValueError(
                f"Design matrix has {len(explanatory_variable)} rows; expected {n_spots}."
            )
        if explanatory_variable.index.equals(spot_index):
            df = explanatory_variable.astype(np.float64).copy()
        else:
            df = pd.DataFrame(
                explanatory_variable.to_numpy(dtype=np.float64),
                index=spot_index,
                columns=explanatory_variable.columns,
            )
    else:
        arr = np.asarray(explanatory_variable, dtype=np.float64)
        if arr.ndim == 1:
            if len(arr) != n_spots:
                raise ValueError(
                    f"explanatory_variable has length {len(arr)}; expected {n_spots}."
                )
            design_df = pd.DataFrame(
                {"Intercept": np.ones(n_spots, dtype=np.float64), "explanatory": arr},
                index=spot_index,
            )
            return design_df, np.array([True, False], dtype=bool)
        if arr.ndim != 2 or arr.shape[0] != n_spots:
            raise ValueError(
                f"explanatory_variable must have {n_spots} rows; got {arr.shape}."
            )
        cols = [f"X{j}" for j in range(arr.shape[1])]
        df = pd.DataFrame(arr, index=spot_index, columns=cols)

    mat = df.to_numpy(dtype=np.float64)
    # Check whether constant 1 vector already lies in the column space of mat
    ones = np.ones(n_spots, dtype=np.float64)
    coef_1, _, _, _ = np.linalg.lstsq(mat, ones, rcond=None)
    spans_ones = float(np.max(np.abs(mat @ coef_1 - ones))) < 1e-6

    if add_intercept and not spans_ones:
        df.insert(0, "Intercept", ones)
        mat = df.to_numpy(dtype=np.float64)

    # Identify constant/baseline columns
    is_const = np.ptp(mat, axis=0) < 1e-10
    if not np.any(is_const) and spans_ones:
        # E.g. categorical one-hot region matrix where sum_r X_ir = 1
        intercept_mask = np.ones(mat.shape[1], dtype=bool)
    else:
        intercept_mask = is_const
    return df, intercept_mask


def run_cside(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    cell_type_weights: pd.DataFrame | np.ndarray | None = None,
    explanatory_variable: pd.DataFrame | pd.Series | np.ndarray | dict[Any, float] | None = None,
    n_umi: pd.Series | np.ndarray | None = None,
    cell_types: list[str] | None = None,
    doublet_mode: bool = False,
    weight_threshold: float | None = None,
    cell_type_threshold: float | None = None,
    gene_threshold: float | None = None,
    fdr: float | None = None,
    log_fc_thresh: float | None = None,
    test_mode: Literal["individual", "nonparam", "categorical"] = "individual",
    params_to_test: list[int] | int | None = None,
    estimate_sigma: bool = True,
    settings: SettingsMode = "default",
) -> CSIDEResult:
    """Run CSIDE cell-type-specific differential expression in spatial transcriptomics.

    Fits the overdispersed Poisson-lognormal cell-type mixture GLM (Cable et al.,
    *Nature Methods* 2022) per gene, estimating gene-specific overdispersion
    ``sigma_g`` via Gauss-Hermite quadrature and cell-type x covariate effects
    via damped Fisher scoring with analytical Fisher information covariance.

    Parameters
    ----------
    counts
        ``AnnData`` (spot x gene counts in ``adata.X``) or ``(n_spots, n_genes)``
        array / sparse matrix / DataFrame.
    cell_type_weights
        ``(n_spots, n_cell_types)`` deconvolved cell-type proportion weights.
        Automatically read from ``adata.obsm["cell_type_weights"]`` when
        ``counts`` is an ``AnnData`` and ``cell_type_weights`` is ``None``.
    explanatory_variable
        1D spot covariate vector/Series or 2D spot x covariate design matrix.
    n_umi
        Optional total UMI counts per spot (defaults to row sums of ``counts``).
    cell_types
        Optional subset of cell types to include in the mixture model.
    doublet_mode
        Whether weights originate from RCTD doublet mode (adjusts
        ``weight_threshold`` to ``0.99`` under ``settings="R_default"``).
    weight_threshold
        Minimum sum of modeled cell-type weights required to retain a spot.
    cell_type_threshold
        Minimum aggregate cell-type weight across spots required to model a
        cell type.
    gene_threshold
        Minimum mean normalized expression ``mean(Y_ig / N_i)`` to test a gene.
    fdr
        Benjamini-Hochberg FDR significance threshold.
    log_fc_thresh
        Minimum absolute natural-log fold change threshold for significance.
    test_mode
        ``"individual"`` (single covariate or multi-parameter test),
        ``"nonparam"`` (smooth spatial spline basis test), or ``"categorical"``
        (pairwise multi-region contrast test).
    params_to_test
        0-based column index (or list of indices) in ``design_matrix`` to test.
    estimate_sigma
        Whether to estimate gene-specific overdispersion ``sigma_g`` per gene.
    settings
        ``"default"`` (Pythonic defaults with mild ``1e-4`` spline Hessian
        stabilization) or ``"R_default"`` (``spacexr::run.CSIDE`` parity).

    Returns
    -------
    CSIDEResult
    """
    cfg = CSIDEConfig(
        settings=settings,
        doublet_mode=doublet_mode,
        weight_threshold=weight_threshold,
        cell_type_threshold=cell_type_threshold,
        gene_threshold=gene_threshold,
        fdr=fdr,
        log_fc_thresh=log_fc_thresh,
    )
    if explanatory_variable is None:
        raise ValueError("explanatory_variable must be provided to run_cside.")

    x_mat, spot_index, gene_index, w_df, umi_vec = _extract_inputs(
        counts, cell_type_weights, n_umi
    )
    design_df, intercept_mask = _prepare_design_matrix(
        explanatory_variable,
        spot_index,
        add_intercept=(test_mode != "categorical"),
    )

    if not np.all(np.isfinite(design_df.to_numpy())):
        raise ValueError("explanatory_variable contains non-finite values.")

    w_raw = np.maximum(w_df.to_numpy(dtype=np.float64), 0.0)
    row_totals_all = np.sum(w_raw, axis=1)
    # Normalize unnormalized density weights (e.g. cell2location) where row sum > 1
    over_one = row_totals_all > (1.0 + 1e-5)
    if np.any(over_one):
        pos_rows = row_totals_all > 0.0
        w_raw[pos_rows] = w_raw[pos_rows] / row_totals_all[pos_rows, None]

    all_ct_names = list(map(str, w_df.columns))
    if cell_types is not None:
        missing = [ct for ct in cell_types if ct not in all_ct_names]
        if missing:
            raise ValueError(f"Requested cell_types not found in weights: {missing}")
        cand_indices = [all_ct_names.index(ct) for ct in cell_types]
    else:
        cand_indices = list(range(len(all_ct_names)))

    assert cfg.cell_type_threshold is not None
    assert cfg.weight_threshold is not None
    assert cfg.gene_threshold is not None
    assert cfg.fdr is not None
    assert cfg.log_fc_thresh is not None

    ct_agg = np.sum(w_raw[:, cand_indices], axis=0)
    kept_local = [i for i, tot in enumerate(ct_agg) if tot >= cfg.cell_type_threshold]
    if not kept_local:
        raise ValueError(
            f"No cell types met cell_type_threshold={cfg.cell_type_threshold} "
            f"(max aggregate weight = {float(np.max(ct_agg)) if len(ct_agg) else 0.0:.2f})."
        )
    kept_ct_idx = [cand_indices[i] for i in kept_local]
    modeled_cell_types = [all_ct_names[idx] for idx in kept_ct_idx]

    w_sub = w_raw[:, kept_ct_idx]
    spot_weight_sum = np.sum(w_sub, axis=1)
    spot_mask = (spot_weight_sum >= cfg.weight_threshold) & (umi_vec > 0)
    n_kept_spots = int(np.sum(spot_mask))
    d_dim = design_df.shape[1]
    if n_kept_spots <= d_dim:
        raise ValueError(
            f"Only {n_kept_spots} spots passed weight_threshold={cfg.weight_threshold}; "
            f"need > {d_dim} spots for design matrix rank."
        )

    w_kept = w_sub[spot_mask] / np.maximum(spot_weight_sum[spot_mask, None], 1e-12)
    umi_kept = umi_vec[spot_mask]
    design_kept_df = design_df.loc[spot_mask].copy()
    design_kept = design_kept_df.to_numpy(dtype=np.float64)

    x_kept = x_mat[spot_mask]
    # Gene filtering: mean(Y_ig / N_i) >= gene_threshold and total counts > 0
    inv_umi = 1.0 / umi_kept
    if sparse.issparse(x_kept):
        mean_norm = np.asarray(x_kept.multiply(inv_umi[:, None]).mean(axis=0)).ravel()
        gene_totals = np.asarray(x_kept.sum(axis=0)).ravel()
    else:
        mean_norm = np.mean(x_kept * inv_umi[:, None], axis=0)
        gene_totals = np.sum(x_kept, axis=0)

    gene_mask = (mean_norm >= cfg.gene_threshold) & (gene_totals > 0)
    kept_gene_indices = np.flatnonzero(gene_mask)
    kept_gene_names = [str(gene_index[g]) for g in kept_gene_indices]

    if params_to_test is None:
        if test_mode == "categorical":
            test_indices = list(range(d_dim))
        elif test_mode == "nonparam":
            test_indices = [j for j in range(d_dim) if not intercept_mask[j]]
            if not test_indices:
                test_indices = list(range(1, d_dim))
        else:
            non_int = [j for j in range(d_dim) if not intercept_mask[j]]
            test_indices = [non_int[-1]] if non_int else [d_dim - 1]
    elif isinstance(params_to_test, int):
        test_indices = [params_to_test]
    else:
        test_indices = list(params_to_test)

    k_types = len(modeled_cell_types)
    n_genes_tested = len(kept_gene_indices)

    log_fc_mat = np.full((n_genes_tested, k_types), np.nan, dtype=np.float64)
    se_mat = np.full((n_genes_tested, k_types), np.nan, dtype=np.float64)
    z_mat = np.full((n_genes_tested, k_types), np.nan, dtype=np.float64)
    pval_mat = np.full((n_genes_tested, k_types), np.nan, dtype=np.float64)
    conv_mat = np.zeros((n_genes_tested, k_types), dtype=bool)
    sigma_arr = np.full(n_genes_tested, np.nan, dtype=np.float64)
    coef_tensor = np.full((n_genes_tested, d_dim, k_types), np.nan, dtype=np.float64)

    for g_out, g_in in enumerate(kept_gene_indices):
        if sparse.issparse(x_kept):
            y_g = x_kept[:, g_in].toarray().ravel().astype(np.float64)
        else:
            y_g = np.asarray(x_kept[:, g_in], dtype=np.float64)

        alpha, cov, sig_g, conv_mask, _ = fit_gene_cside(
            y=y_g,
            n_umi=umi_kept,
            weights=w_kept,
            design=design_kept,
            intercept_mask=intercept_mask,
            settings=cfg.settings,
            estimate_sigma=estimate_sigma,
        )
        sigma_arr[g_out] = sig_g
        coef_tensor[g_out] = alpha

        for k in range(k_types):
            if test_mode == "categorical":
                # Pairwise contrasts across all region pairs (r1 < r2)
                valid_regions = [r for r in test_indices if conv_mask[r, k]]
                if len(valid_regions) < 2:
                    continue
                best_p = 2.0
                best_lfc = np.nan
                best_se = np.nan
                best_z = np.nan
                n_pairs = 0
                for ii in range(len(valid_regions)):
                    r1 = valid_regions[ii]
                    idx1 = r1 * k_types + k
                    for jj in range(ii + 1, len(valid_regions)):
                        r2 = valid_regions[jj]
                        idx2 = r2 * k_types + k
                        var_pair = (
                            cov[idx2, idx2] + cov[idx1, idx1] - 2.0 * cov[idx2, idx1]
                        )
                        if not np.isfinite(var_pair) or var_pair <= 1e-16:
                            continue
                        se_pair = float(np.sqrt(var_pair))
                        lfc_pair = float(alpha[r2, k] - alpha[r1, k])
                        z_pair = lfc_pair / se_pair
                        p_pair = float(2.0 * norm.sf(abs(z_pair)))
                        n_pairs += 1
                        if p_pair < best_p:
                            best_p = p_pair
                            best_lfc = lfc_pair
                            best_se = se_pair
                            best_z = z_pair
                if n_pairs > 0 and np.isfinite(best_p):
                    log_fc_mat[g_out, k] = best_lfc
                    se_mat[g_out, k] = best_se
                    z_mat[g_out, k] = best_z
                    pval_mat[g_out, k] = min(1.0, best_p * n_pairs)
                    conv_mat[g_out, k] = True

            elif len(test_indices) == 1:
                d_idx = test_indices[0]
                if not conv_mask[d_idx, k]:
                    continue
                p_idx = d_idx * k_types + k
                var_dk = float(cov[p_idx, p_idx])
                if not np.isfinite(var_dk) or var_dk <= 0.0:
                    continue
                se_dk = float(np.sqrt(var_dk))
                lfc_dk = float(alpha[d_idx, k])
                z_dk = lfc_dk / se_dk
                log_fc_mat[g_out, k] = lfc_dk
                se_mat[g_out, k] = se_dk
                z_mat[g_out, k] = z_dk
                pval_mat[g_out, k] = float(2.0 * norm.sf(abs(z_dk)))
                conv_mat[g_out, k] = True

            else:
                valid_d = [d_idx for d_idx in test_indices if conv_mask[d_idx, k]]
                if not valid_d:
                    continue
                if cfg.settings == "R_default" or test_mode == "individual":
                    # spacexr Bonferroni-minimum over basis parameters
                    best_p = 2.0
                    best_lfc = np.nan
                    best_se = np.nan
                    best_z = np.nan
                    n_ok = 0
                    for d_idx in valid_d:
                        p_idx = d_idx * k_types + k
                        var_dk = float(cov[p_idx, p_idx])
                        if not np.isfinite(var_dk) or var_dk <= 0.0:
                            continue
                        se_dk = float(np.sqrt(var_dk))
                        lfc_dk = float(alpha[d_idx, k])
                        z_dk = lfc_dk / se_dk
                        p_dk = float(2.0 * norm.sf(abs(z_dk)))
                        n_ok += 1
                        if p_dk < best_p:
                            best_p = p_dk
                            best_lfc = lfc_dk
                            best_se = se_dk
                            best_z = z_dk
                    if n_ok > 0 and np.isfinite(best_p):
                        log_fc_mat[g_out, k] = best_lfc
                        se_mat[g_out, k] = best_se
                        z_mat[g_out, k] = best_z
                        pval_mat[g_out, k] = min(1.0, best_p * n_ok)
                        conv_mat[g_out, k] = True
                else:
                    # Pythonic default for nonparam: rotation-invariant joint
                    # spline Wald chi^2 test + fitted spatial log-FC range
                    flat_idx = np.array([d_idx * k_types + k for d_idx in valid_d])
                    beta_sub = alpha[valid_d, k]
                    cov_sub = cov[np.ix_(flat_idx, flat_idx)]
                    try:
                        stat_chi2 = float(beta_sub @ np.linalg.solve(cov_sub, beta_sub))
                    except np.linalg.LinAlgError:
                        stat_chi2 = float(
                            beta_sub @ (np.linalg.pinv(cov_sub, rcond=1e-10) @ beta_sub)
                        )
                    if not np.isfinite(stat_chi2) or stat_chi2 < 0.0:
                        continue
                    p_joint = float(chi2.sf(stat_chi2, df=len(valid_d)))
                    eta_spatial = design_kept[:, valid_d] @ beta_sub
                    if len(eta_spatial) >= 20:
                        span_lfc = float(
                            np.percentile(eta_spatial, 97.5)
                            - np.percentile(eta_spatial, 2.5)
                        )
                    else:
                        span_lfc = float(np.ptp(eta_spatial))
                    dom_sign = float(np.sign(beta_sub[int(np.argmax(np.abs(beta_sub)))]))
                    if dom_sign == 0.0:
                        dom_sign = 1.0
                    signed_lfc = dom_sign * span_lfc
                    z_equiv = dom_sign * float(
                        norm.isf(np.clip(0.5 * p_joint, 1e-300, 0.5))
                    )
                    se_equiv = (
                        abs(signed_lfc / z_equiv)
                        if abs(z_equiv) > 1e-8
                        else float(np.sqrt(max(float(np.max(np.diag(cov_sub))), 1e-8)))
                    )
                    log_fc_mat[g_out, k] = signed_lfc
                    se_mat[g_out, k] = se_equiv
                    z_mat[g_out, k] = z_equiv
                    pval_mat[g_out, k] = p_joint
                    conv_mat[g_out, k] = True

    all_gene_list: dict[str, pd.DataFrame] = {}
    sig_gene_list: dict[str, pd.DataFrame] = {}
    coefficients: dict[str, pd.DataFrame] = {}

    for k, ct in enumerate(modeled_cell_types):
        qvals = _bh_fdr(pval_mat[:, k])
        sig_mask = (
            conv_mat[:, k]
            & np.isfinite(qvals)
            & (qvals < cfg.fdr)
            & (np.abs(log_fc_mat[:, k]) >= cfg.log_fc_thresh)
        )
        df_ct = pd.DataFrame(
            {
                "log_fc": log_fc_mat[:, k],
                "se": se_mat[:, k],
                "z_stat": z_mat[:, k],
                "p_val": pval_mat[:, k],
                "q_val": qvals,
                "significant": sig_mask,
                "converged": conv_mat[:, k],
            },
            index=pd.Index(kept_gene_names, name="gene"),
        )
        all_gene_list[ct] = df_ct
        sig_gene_list[ct] = df_ct.loc[df_ct["significant"]].sort_values("p_val")
        coefficients[ct] = pd.DataFrame(
            coef_tensor[:, :, k],
            index=pd.Index(kept_gene_names, name="gene"),
            columns=design_kept_df.columns,
        )

    sigma_series = pd.Series(
        sigma_arr, index=pd.Index(kept_gene_names, name="gene"), name="sigma_gene"
    )

    return CSIDEResult(
        all_gene_list=all_gene_list,
        sig_gene_list=sig_gene_list,
        sigma_gene=sigma_series,
        design_matrix=design_kept_df,
        cell_types=modeled_cell_types,
        coefficients=coefficients,
        settings=cfg.settings,
        params={
            "doublet_mode": cfg.doublet_mode,
            "weight_threshold": cfg.weight_threshold,
            "cell_type_threshold": cfg.cell_type_threshold,
            "gene_threshold": cfg.gene_threshold,
            "fdr": cfg.fdr,
            "log_fc_thresh": cfg.log_fc_thresh,
            "test_mode": test_mode,
        },
    )


def run_cside_single(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    cell_type_weights: pd.DataFrame | np.ndarray | None = None,
    explanatory_variable: pd.Series | np.ndarray | dict[Any, float] | None = None,
    n_umi: pd.Series | np.ndarray | None = None,
    cell_types: list[str] | None = None,
    doublet_mode: bool = False,
    weight_threshold: float | None = None,
    cell_type_threshold: float | None = None,
    gene_threshold: float | None = None,
    fdr: float | None = None,
    log_fc_thresh: float | None = None,
    standardize_range: bool = True,
    settings: SettingsMode = "default",
) -> CSIDEResult:
    """Convenience wrapper for single-covariate CSIDE spatial DE (`run.CSIDE.single`)."""
    validate_settings(settings)
    if explanatory_variable is None:
        raise ValueError("explanatory_variable must be provided to run_cside_single.")

    if isinstance(explanatory_variable, dict):
        cov_vals: pd.Series | np.ndarray = pd.Series(explanatory_variable, dtype=np.float64)
    elif isinstance(explanatory_variable, pd.Series):
        cov_vals = explanatory_variable.astype(np.float64).copy()
    else:
        arr = np.asarray(explanatory_variable, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"run_cside_single expects a 1D covariate; got shape {arr.shape}."
            )
        cov_vals = arr.copy()

    if standardize_range:
        vmin = float(np.min(cov_vals))
        vmax = float(np.max(cov_vals))
        if vmax - vmin <= 1e-12:
            raise ValueError("explanatory_variable is constant across all spots.")
        cov_vals = (cov_vals - vmin) / (vmax - vmin)

    return run_cside(
        counts=counts,
        cell_type_weights=cell_type_weights,
        explanatory_variable=cov_vals,
        n_umi=n_umi,
        cell_types=cell_types,
        doublet_mode=doublet_mode,
        weight_threshold=weight_threshold,
        cell_type_threshold=cell_type_threshold,
        gene_threshold=gene_threshold,
        fdr=fdr,
        log_fc_thresh=log_fc_thresh,
        test_mode="individual",
        params_to_test=1,
        settings=settings,
    )


def run_cside_regions(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    cell_type_weights: pd.DataFrame | np.ndarray | None = None,
    region_labels: pd.Series | np.ndarray | list[Any] | None = None,
    n_umi: pd.Series | np.ndarray | None = None,
    cell_types: list[str] | None = None,
    doublet_mode: bool = False,
    weight_threshold: float | None = None,
    cell_type_threshold: float | None = None,
    gene_threshold: float | None = None,
    fdr: float | None = None,
    log_fc_thresh: float | None = None,
    settings: SettingsMode = "default",
) -> CSIDEResult:
    """Discrete multi-region CSIDE spatial DE (`run.CSIDE.regions`)."""
    validate_settings(settings)
    if region_labels is None:
        if isinstance(counts, ad.AnnData) and "region" in counts.obs:
            region_labels = counts.obs["region"]
        else:
            raise ValueError("region_labels must be provided to run_cside_regions.")

    s_reg = pd.Series(region_labels)
    design_reg = pd.get_dummies(s_reg, prefix="region", dtype=np.float64)
    if design_reg.shape[1] < 2:
        raise ValueError("run_cside_regions requires at least 2 distinct regions.")
    if isinstance(region_labels, pd.Series):
        design_reg.index = region_labels.index

    return run_cside(
        counts=counts,
        cell_type_weights=cell_type_weights,
        explanatory_variable=design_reg,
        n_umi=n_umi,
        cell_types=cell_types,
        doublet_mode=doublet_mode,
        weight_threshold=weight_threshold,
        cell_type_threshold=cell_type_threshold,
        gene_threshold=gene_threshold,
        fdr=fdr,
        log_fc_thresh=log_fc_thresh,
        test_mode="categorical",
        settings=settings,
    )


def run_cside_nonparam(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    cell_type_weights: pd.DataFrame | np.ndarray | None = None,
    coords: pd.DataFrame | np.ndarray | None = None,
    df: int = 15,
    basis: Literal["thin_plate", "bspline"] = "thin_plate",
    n_umi: pd.Series | np.ndarray | None = None,
    cell_types: list[str] | None = None,
    doublet_mode: bool = False,
    weight_threshold: float | None = None,
    cell_type_threshold: float | None = None,
    gene_threshold: float | None = None,
    fdr: float | None = None,
    log_fc_thresh: float | None = None,
    settings: SettingsMode = "default",
) -> CSIDEResult:
    """Non-parametric smooth spatial spline CSIDE (`run.CSIDE.nonparam`)."""
    validate_settings(settings)
    if coords is None:
        if isinstance(counts, ad.AnnData) and "spatial" in counts.obsm:
            coords = pd.DataFrame(
                np.asarray(counts.obsm["spatial"], dtype=np.float64)[:, :2],
                index=counts.obs_names,
                columns=["x", "y"],
            )
        else:
            raise ValueError(
                "coords must be provided (or stored in adata.obsm['spatial']) for run_cside_nonparam."
            )

    spline_design = build_spline_design(
        coords=coords, df=df, basis=basis, settings=settings
    )
    return run_cside(
        counts=counts,
        cell_type_weights=cell_type_weights,
        explanatory_variable=spline_design,
        n_umi=n_umi,
        cell_types=cell_types,
        doublet_mode=doublet_mode,
        weight_threshold=weight_threshold,
        cell_type_threshold=cell_type_threshold,
        gene_threshold=gene_threshold,
        fdr=fdr,
        log_fc_thresh=log_fc_thresh,
        test_mode="nonparam",
        settings=settings,
    )
