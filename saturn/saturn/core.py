"""Core pipeline for satuRn quasi-binomial Empirical Bayes DTU analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from scipy.stats import t as t_dist

from .eb import (
    SettingsMode,
    SqueezeVarResult,
    estimate_empirical_null,
    squeeze_var,
    validate_settings,
)
from .glm import fit_quasibinomial_batch


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment preserving NaNs."""
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
    q_sorted = np.minimum.accumulate((ranked * (m / ranks))[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    qvals[valid] = q_sorted[inv]
    return qvals


@dataclass
class FilteredDTUData:
    """Filtered transcript counts and parent-gene binomial trial totals."""

    counts: np.ndarray | sparse.csc_matrix
    tx2gene: pd.DataFrame
    gene_totals: np.ndarray | sparse.csc_matrix
    sample_names: pd.Index
    transcript_ids: pd.Index
    obs: pd.DataFrame | None = None
    settings: SettingsMode = "default"

    @property
    def other_counts(self) -> np.ndarray | sparse.csc_matrix:
        """Compute ``n_gi - y_ti`` binomial complement counts per transcript."""
        gene_codes, _ = pd.factorize(self.tx2gene["gene_id"], sort=False)
        if sparse.issparse(self.counts):
            gt_tx = self.gene_totals[:, gene_codes]
            return (gt_tx - self.counts).tocsc()
        return self.gene_totals[:, gene_codes] - self.counts

    def to_anndata(self) -> ad.AnnData:
        """Convert filtered DTU data into an ``AnnData`` object."""
        obs_df = (
            self.obs.copy()
            if self.obs is not None
            else pd.DataFrame(index=self.sample_names)
        )
        var_df = self.tx2gene.set_index("transcript_id").reindex(self.transcript_ids)
        x_val = self.counts.tocsr() if sparse.issparse(self.counts) else self.counts.copy()
        return ad.AnnData(X=x_val, obs=obs_df, var=var_df)


@dataclass
class DTUFit:
    """Fitted quasi-binomial GLM results across transcripts."""

    coefficients: pd.DataFrame
    cov_unscaled: np.ndarray
    dispersion: pd.Series
    df_residual: pd.Series
    deviance: pd.Series
    converged: pd.Series
    boundary: pd.Series
    tx2gene: pd.DataFrame
    design: pd.DataFrame
    settings: SettingsMode = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a readable summary of fitted quasi-binomial DTU models."""
        n_tx = len(self.coefficients)
        n_genes = int(self.tx2gene["gene_id"].nunique())
        n_conv = int(self.converged.sum())
        n_bnd = int(self.boundary.sum())
        med_disp = float(self.dispersion.dropna().median()) if n_tx > 0 else float("nan")
        return (
            f"DTUFit (settings={self.settings!r})\n"
            f"  Samples x Covariates : {self.design.shape[0]} x {self.design.shape[1]}\n"
            f"  Transcripts (Genes)  : {n_tx} ({n_genes} genes)\n"
            f"  Converged / Boundary : {n_conv}/{n_tx} converged, {n_bnd} boundary\n"
            f"  Median Dispersion    : {med_disp:.4f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export per-transcript model diagnostics and coefficients."""
        out = self.tx2gene.set_index("transcript_id").copy()
        out["dispersion"] = self.dispersion
        out["df_residual"] = self.df_residual
        out["deviance"] = self.deviance
        out["converged"] = self.converged
        out["boundary"] = self.boundary
        for col in self.coefficients.columns:
            out[f"coef_{col}"] = self.coefficients[col]
        return out.reset_index()


def _parse_tx2gene(
    tx2gene: pd.DataFrame | pd.Series | dict[str, str] | np.ndarray | list[str],
    default_tx_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Normalize ``tx2gene`` into a 2-column DataFrame ``['transcript_id', 'gene_id']``."""
    if isinstance(tx2gene, dict):
        df = pd.DataFrame(
            list(tx2gene.items()), columns=["transcript_id", "gene_id"]
        )
    elif isinstance(tx2gene, pd.Series):
        tx_idx = tx2gene.index if default_tx_ids is None else tx2gene.index
        df = pd.DataFrame(
            {
                "transcript_id": tx_idx.astype(str),
                "gene_id": tx2gene.astype(str).to_numpy(),
            }
        )
    elif isinstance(tx2gene, pd.DataFrame):
        cols_lower = {c.lower(): c for c in tx2gene.columns}
        tx_col = None
        for cand in ("transcript_id", "isoform_id", "tx_id", "feature_id", "transcript"):
            if cand in cols_lower:
                tx_col = cols_lower[cand]
                break
        gene_col = None
        for cand in ("gene_id", "group_id", "gene", "parent_gene"):
            if cand in cols_lower:
                gene_col = cols_lower[cand]
                break

        if tx_col is not None and gene_col is not None:
            df = pd.DataFrame(
                {
                    "transcript_id": tx2gene[tx_col].astype(str).to_numpy(),
                    "gene_id": tx2gene[gene_col].astype(str).to_numpy(),
                }
            )
        elif gene_col is not None and tx_col is None:
            df = pd.DataFrame(
                {
                    "transcript_id": tx2gene.index.astype(str),
                    "gene_id": tx2gene[gene_col].astype(str).to_numpy(),
                }
            )
        elif tx2gene.shape[1] >= 2:
            df = pd.DataFrame(
                {
                    "transcript_id": tx2gene.iloc[:, 0].astype(str).to_numpy(),
                    "gene_id": tx2gene.iloc[:, 1].astype(str).to_numpy(),
                }
            )
        else:
            raise ValueError(
                "tx2gene DataFrame must have ('transcript_id', 'gene_id') or >=2 columns."
            )
    else:
        arr = np.asarray(tx2gene)
        if arr.ndim == 1:
            if default_tx_ids is None or len(default_tx_ids) != len(arr):
                tx_ids = [f"tx_{i}" for i in range(len(arr))]
            else:
                tx_ids = list(map(str, default_tx_ids))
            df = pd.DataFrame({"transcript_id": tx_ids, "gene_id": arr.astype(str)})
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            df = pd.DataFrame(
                {"transcript_id": arr[:, 0].astype(str), "gene_id": arr[:, 1].astype(str)}
            )
        else:
            raise ValueError(f"Unsupported tx2gene shape {arr.shape}.")

    return df.reset_index(drop=True)


def _extract_counts_and_tx2gene(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    tx2gene: pd.DataFrame | pd.Series | dict[str, str] | np.ndarray | list[str] | None,
) -> tuple[
    np.ndarray | sparse.csc_matrix,
    pd.Index,
    pd.Index,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    """Orient counts to ``(n_samples, n_transcripts)`` and align ``tx2gene``."""
    if isinstance(counts, ad.AnnData):
        adata = counts
        sample_names = pd.Index(adata.obs_names.astype(str), name="sample")
        tx_ids = pd.Index(adata.var_names.astype(str), name="transcript_id")
        if sparse.issparse(adata.X):
            x_mat: np.ndarray | sparse.csc_matrix = adata.X.tocsc()
        else:
            x_mat = np.asarray(adata.X, dtype=np.float64)

        if tx2gene is None:
            for gcol in ("gene_id", "gene", "group_id"):
                if gcol in adata.var.columns:
                    tx2gene = pd.DataFrame(
                        {
                            "transcript_id": tx_ids,
                            "gene_id": adata.var[gcol].astype(str).to_numpy(),
                        }
                    )
                    break
            if tx2gene is None:
                raise ValueError(
                    "tx2gene was None and 'gene_id' column was not found in adata.var."
                )
        t2g = _parse_tx2gene(tx2gene, default_tx_ids=tx_ids)
        t2g = t2g.set_index("transcript_id").reindex(tx_ids).reset_index()
        if t2g["gene_id"].isna().any():
            raise ValueError("tx2gene is missing gene_id mappings for some transcripts.")
        return x_mat, sample_names, tx_ids, t2g, adata.obs.copy()

    if tx2gene is None:
        raise ValueError("tx2gene must be provided when counts is not an AnnData.")

    if isinstance(counts, pd.DataFrame):
        t2g_raw = _parse_tx2gene(tx2gene)
        tx_set = set(t2g_raw["transcript_id"])
        # Support both (samples x transcripts) and R-style (transcripts x samples)
        cols_match = len(set(counts.columns.astype(str)) & tx_set)
        idx_match = len(set(counts.index.astype(str)) & tx_set)
        if idx_match > cols_match and counts.shape[0] == len(t2g_raw):
            counts = counts.T
        elif counts.shape[1] != len(t2g_raw) and counts.shape[0] == len(t2g_raw):
            counts = counts.T

        sample_names = pd.Index(counts.index.astype(str), name="sample")
        tx_ids = pd.Index(counts.columns.astype(str), name="transcript_id")
        x_mat = counts.to_numpy(dtype=np.float64)
        t2g = t2g_raw.set_index("transcript_id").reindex(tx_ids).reset_index()
        if t2g["gene_id"].isna().any():
            raise ValueError("tx2gene transcript_id values do not match counts columns/index.")
        return x_mat, sample_names, tx_ids, t2g, None

    t2g = _parse_tx2gene(tx2gene)
    n_tx = len(t2g)
    if sparse.issparse(counts):
        if counts.shape[1] != n_tx and counts.shape[0] == n_tx:
            counts = counts.T
        if counts.shape[1] != n_tx:
            raise ValueError(
                f"counts shape {counts.shape} incompatible with {n_tx} transcripts in tx2gene."
            )
        x_mat = counts.tocsc()
    else:
        arr = np.asarray(counts, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"counts must be 2D; got {arr.shape}.")
        if arr.shape[1] != n_tx and arr.shape[0] == n_tx:
            arr = arr.T
        if arr.shape[1] != n_tx:
            raise ValueError(
                f"counts shape {arr.shape} incompatible with {n_tx} transcripts in tx2gene."
            )
        x_mat = arr

    n_samples = x_mat.shape[0]
    sample_names = pd.Index([f"sample_{i}" for i in range(n_samples)], name="sample")
    tx_ids = pd.Index(t2g["transcript_id"].astype(str), name="transcript_id")
    return x_mat, sample_names, tx_ids, t2g, None


def _compute_gene_totals(
    x_mat: np.ndarray | sparse.csc_matrix, gene_codes: np.ndarray, n_genes: int
) -> np.ndarray | sparse.csc_matrix:
    """Compute ``(n_samples, n_genes)`` parent-gene count sums efficiently."""
    n_tx = len(gene_codes)
    tx_to_gene = sparse.csc_matrix(
        (np.ones(n_tx, dtype=np.float64), (np.arange(n_tx), gene_codes)),
        shape=(n_tx, n_genes),
    )
    if sparse.issparse(x_mat):
        return (x_mat @ tx_to_gene).tocsc()
    return np.asarray(x_mat @ tx_to_gene, dtype=np.float64)


def filter_dtu(
    counts: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    tx2gene: pd.DataFrame | pd.Series | dict[str, str] | np.ndarray | list[str] | None = None,
    min_count: float = 10.0,
    min_samples: int = 3,
    remove_single_isoform: bool = True,
    settings: SettingsMode = "default",
) -> FilteredDTUData:
    """Filter low-expression transcripts and single-isoform genes for DTU analysis.

    Parameters
    ----------
    counts
        ``AnnData`` (cells/samples x transcripts) or 2D DataFrame / ndarray /
        sparse matrix of transcript counts.
    tx2gene
        Mapping from ``transcript_id`` to ``gene_id`` (optional if ``counts`` is
        an ``AnnData`` with ``adata.var["gene_id"]``).
    min_count
        Minimum transcript count threshold per sample/cell.
    min_samples
        Minimum number of samples/cells meeting ``min_count``.
    remove_single_isoform
        Whether to drop genes that retain fewer than 2 expressed isoforms.
    settings
        ``"default"`` or ``"R_default"``.

    Returns
    -------
    FilteredDTUData
    """
    validate_settings(settings)
    x_mat, sample_names, tx_ids, t2g, obs_df = _extract_counts_and_tx2gene(
        counts, tx2gene
    )

    if sparse.issparse(x_mat):
        if min_count <= 0:
            n_pass = np.full(x_mat.shape[1], x_mat.shape[0], dtype=int)
        else:
            # Count entries >= min_count per column cleanly via CSC data
            csc = x_mat.tocsc()
            ge_data = (csc.data >= min_count).astype(np.int64)
            ge_mat = sparse.csc_matrix((ge_data, csc.indices, csc.indptr), shape=csc.shape)
            n_pass = np.asarray(ge_mat.sum(axis=0)).ravel()
    else:
        n_pass = np.sum(x_mat >= min_count, axis=0)

    keep_mask = n_pass >= int(min_samples)
    t2g_kept = t2g.loc[keep_mask].reset_index(drop=True)

    if remove_single_isoform and not t2g_kept.empty:
        gene_counts = t2g_kept["gene_id"].value_counts()
        multi_genes = set(gene_counts[gene_counts >= 2].index)
        multi_mask = t2g_kept["gene_id"].isin(multi_genes).to_numpy()
        orig_indices = np.flatnonzero(keep_mask)[multi_mask]
        t2g_kept = t2g_kept.loc[multi_mask].reset_index(drop=True)
    else:
        orig_indices = np.flatnonzero(keep_mask)

    x_filt = x_mat[:, orig_indices]
    if sparse.issparse(x_filt):
        x_filt = x_filt.tocsc()
    tx_filt = pd.Index(t2g_kept["transcript_id"].astype(str), name="transcript_id")

    gene_codes, uniques = pd.factorize(t2g_kept["gene_id"], sort=False)
    gene_totals = _compute_gene_totals(x_filt, gene_codes, len(uniques))

    return FilteredDTUData(
        counts=x_filt,
        tx2gene=t2g_kept,
        gene_totals=gene_totals,
        sample_names=sample_names,
        transcript_ids=tx_filt,
        obs=obs_df,
        settings=settings,
    )


def _prepare_design(
    design: pd.DataFrame | pd.Series | np.ndarray | list[Any] | str | None,
    obs_df: pd.DataFrame | None,
    sample_names: pd.Index,
) -> pd.DataFrame:
    """Construct numeric full-rank design matrix aligned to ``sample_names``."""
    n_samples = len(sample_names)
    if design is None:
        if obs_df is None or obs_df.empty:
            raise ValueError("design must be provided when counts has no obs table.")
        design_src: Any = obs_df
    elif isinstance(design, str):
        if obs_df is None or design not in obs_df.columns:
            raise ValueError(f"Column {design!r} not found in adata.obs.")
        design_src = obs_df[[design]]
    else:
        design_src = design

    if isinstance(design_src, pd.Series):
        design_src = design_src.to_frame(name=str(design_src.name or "group"))
    elif not isinstance(design_src, pd.DataFrame):
        arr = np.asarray(design_src)
        if arr.ndim == 1:
            design_src = pd.DataFrame({"group": arr}, index=sample_names)
        elif arr.ndim == 2:
            cols = [f"X{j}" for j in range(arr.shape[1])]
            design_src = pd.DataFrame(arr, index=sample_names, columns=cols)
        else:
            raise ValueError(f"Unsupported design shape {arr.shape}.")

    if len(design_src) != n_samples:
        raise ValueError(
            f"design has {len(design_src)} rows, expected {n_samples} samples."
        )
    if design_src.index.equals(sample_names):
        df = design_src.copy()
    elif set(sample_names).issubset(set(design_src.index.astype(str))):
        df = design_src.copy()
        df.index = df.index.astype(str)
        df = df.reindex(sample_names)
    else:
        df = design_src.copy()
        df.index = sample_names

    # Encode any categorical / object / boolean columns via dummy variables
    has_non_num = any(
        not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        for c in df.columns
    )
    if has_non_num:
        df = pd.get_dummies(df, drop_first=True, dtype=np.float64)
    else:
        df = df.astype(np.float64)

    mat = df.to_numpy(dtype=np.float64)
    ones = np.ones(n_samples, dtype=np.float64)
    if mat.shape[1] == 0:
        return pd.DataFrame({"Intercept": ones}, index=sample_names)

    coef_1, _, _, _ = np.linalg.lstsq(mat, ones, rcond=None)
    spans_ones = float(np.max(np.abs(mat @ coef_1 - ones))) < 1e-6
    if not spans_ones:
        df.insert(0, "Intercept", ones)

    return df


def fit_dtu(
    counts: FilteredDTUData | ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    tx2gene: pd.DataFrame | pd.Series | dict[str, str] | np.ndarray | list[str] | None = None,
    design: pd.DataFrame | pd.Series | np.ndarray | list[Any] | str | None = None,
    max_abs_beta: float = 15.0,
    settings: SettingsMode = "default",
) -> DTUFit:
    """Fit per-transcript quasi-binomial logit GLMs (`satuRn::fitDTU`).

    Fits across active samples where parent-gene total counts ``n_gi > 0``,
    cleanly supporting both dense bulk RNA-seq and sparse single-cell isoform or
    exon count matrices.
    """
    validate_settings(settings)
    if isinstance(counts, FilteredDTUData):
        x_mat = counts.counts
        t2g = counts.tx2gene.copy()
        gene_totals = counts.gene_totals
        sample_names = counts.sample_names
        tx_ids = counts.transcript_ids
        obs_df = counts.obs
        gene_codes, _ = pd.factorize(t2g["gene_id"], sort=False)
    else:
        x_mat, sample_names, tx_ids, t2g, obs_df = _extract_counts_and_tx2gene(
            counts, tx2gene
        )
        gene_codes, uniques = pd.factorize(t2g["gene_id"], sort=False)
        gene_totals = _compute_gene_totals(x_mat, gene_codes, len(uniques))

    design_df = _prepare_design(design, obs_df, sample_names)
    design_mat = design_df.to_numpy(dtype=np.float64)
    n_samples, p_cov = design_mat.shape
    n_tx = len(tx_ids)
    n_genes = int(gene_totals.shape[1]) if n_tx > 0 else 0

    beta_all = np.full((n_tx, p_cov), np.nan, dtype=np.float64)
    cov_all = np.full((n_tx, p_cov, p_cov), np.nan, dtype=np.float64)
    disp_all = np.full(n_tx, np.nan, dtype=np.float64)
    dev_all = np.full(n_tx, np.nan, dtype=np.float64)
    df_res_all = np.zeros(n_tx, dtype=np.float64)
    conv_all = np.zeros(n_tx, dtype=bool)
    bnd_all = np.zeros(n_tx, dtype=bool)

    if n_tx == 0:
        return DTUFit(
            coefficients=pd.DataFrame(beta_all, index=tx_ids, columns=design_df.columns),
            cov_unscaled=cov_all,
            dispersion=pd.Series(disp_all, index=tx_ids, name="dispersion"),
            df_residual=pd.Series(df_res_all, index=tx_ids, name="df_residual"),
            deviance=pd.Series(dev_all, index=tx_ids, name="deviance"),
            converged=pd.Series(conv_all, index=tx_ids, name="converged"),
            boundary=pd.Series(bnd_all, index=tx_ids, name="boundary"),
            tx2gene=t2g,
            design=design_df,
            settings=settings,
        )

    # Group transcripts by parent gene; fast-batch all genes expressed across all samples
    gene_to_tx: list[list[int]] = [[] for _ in range(n_genes)]
    for t_idx, g_idx in enumerate(gene_codes):
        gene_to_tx[int(g_idx)].append(t_idx)

    dense_gt = (
        gene_totals.toarray()
        if sparse.issparse(gene_totals)
        else np.asarray(gene_totals, dtype=np.float64)
    )
    full_genes = np.flatnonzero(np.all(dense_gt > 0.0, axis=0))
    sparse_genes = np.flatnonzero(~np.all(dense_gt > 0.0, axis=0))

    if len(full_genes) > 0:
        full_tx = [t for g in full_genes for t in gene_to_tx[int(g)]]
        full_tx_arr = np.array(full_tx, dtype=np.int64)
        full_g_arr = gene_codes[full_tx_arr]
        if sparse.issparse(x_mat):
            y_full = x_mat[:, full_tx_arr].toarray().T.astype(np.float64)
        else:
            y_full = np.asarray(x_mat[:, full_tx_arr], dtype=np.float64).T
        n_full = dense_gt[:, full_g_arr].T

        b_f, c_f, d_f, dev_f, cv_f, bd_f, dfr_f = fit_quasibinomial_batch(
            y_batch=y_full,
            n_batch=n_full,
            design_sub=design_mat,
            settings=settings,
            max_abs_beta=max_abs_beta,
        )
        beta_all[full_tx_arr] = b_f
        cov_all[full_tx_arr] = c_f
        disp_all[full_tx_arr] = d_f
        dev_all[full_tx_arr] = dev_f
        df_res_all[full_tx_arr] = float(dfr_f)
        conv_all[full_tx_arr] = cv_f
        bnd_all[full_tx_arr] = bd_f

    for g_idx in sparse_genes:
        n_vec = dense_gt[:, int(g_idx)]
        active_samples = np.flatnonzero(n_vec > 0.0)
        tx_list = gene_to_tx[int(g_idx)]
        tx_arr = np.array(tx_list, dtype=np.int64)
        m_act = len(active_samples)
        if m_act <= p_cov:
            df_res_all[tx_arr] = float(max(0, m_act - p_cov))
            continue

        if sparse.issparse(x_mat):
            y_sub = x_mat[active_samples[:, None], tx_arr[None, :]].toarray().T.astype(np.float64)
        else:
            y_sub = np.asarray(x_mat[np.ix_(active_samples, tx_arr)], dtype=np.float64).T
        n_sub = np.broadcast_to(n_vec[active_samples][None, :], y_sub.shape).copy()
        x_sub = design_mat[active_samples]

        b_s, c_s, d_s, dev_s, cv_s, bd_s, dfr_s = fit_quasibinomial_batch(
            y_batch=y_sub,
            n_batch=n_sub,
            design_sub=x_sub,
            settings=settings,
            max_abs_beta=max_abs_beta,
        )
        beta_all[tx_arr] = b_s
        cov_all[tx_arr] = c_s
        disp_all[tx_arr] = d_s
        dev_all[tx_arr] = dev_s
        df_res_all[tx_arr] = float(dfr_s)
        conv_all[tx_arr] = cv_s
        bnd_all[tx_arr] = bd_s

    return DTUFit(
        coefficients=pd.DataFrame(beta_all, index=tx_ids, columns=design_df.columns),
        cov_unscaled=cov_all,
        dispersion=pd.Series(disp_all, index=tx_ids, name="dispersion"),
        df_residual=pd.Series(df_res_all, index=tx_ids, name="df_residual"),
        deviance=pd.Series(dev_all, index=tx_ids, name="deviance"),
        converged=pd.Series(conv_all, index=tx_ids, name="converged"),
        boundary=pd.Series(bnd_all, index=tx_ids, name="boundary"),
        tx2gene=t2g,
        design=design_df,
        settings=settings,
    )


def _test_single_contrast(
    fit: DTUFit,
    c_vec: np.ndarray,
    eb_res: SqueezeVarResult,
    diag_plot: bool,
    settings: SettingsMode,
) -> pd.DataFrame:
    """Evaluate moderated quasi-t statistics and empirical null for one contrast."""
    beta = fit.coefficients.to_numpy(dtype=np.float64)
    cov_u = fit.cov_unscaled
    estimates = beta @ c_vec
    v_unscaled = np.einsum("p,tpq,q->t", c_vec, cov_u, c_vec)

    valid_var = np.isfinite(v_unscaled) & (v_unscaled > 0.0) & np.isfinite(eb_res.s2_post)
    se = np.where(valid_var, np.sqrt(np.maximum(eb_res.s2_post * v_unscaled, 0.0)), np.nan)
    tstat = np.where(valid_var & (se > 0.0), estimates / se, np.nan)

    df_tot = eb_res.df_total
    pval_reg = np.full_like(tstat, np.nan, dtype=np.float64)
    fin_mask = np.isfinite(tstat) & np.isfinite(df_tot) & (df_tot > 0.0)
    inf_mask = np.isfinite(tstat) & np.isinf(df_tot)

    if np.any(fin_mask):
        pval_reg[fin_mask] = 2.0 * t_dist.sf(np.abs(tstat[fin_mask]), df=df_tot[fin_mask])
    if np.any(inf_mask):
        pval_reg[inf_mask] = 2.0 * norm.sf(np.abs(tstat[inf_mask]))

    fdr_reg = _bh_fdr(pval_reg)

    # Exact tail-preserving z-score conversion z_t = Phi^{-1}(F_{t, df_total}(t_t))
    z_scores = np.full_like(tstat, np.nan, dtype=np.float64)
    ok_z = np.isfinite(pval_reg) & np.isfinite(tstat)
    if np.any(ok_z):
        half_p = np.clip(0.5 * pval_reg[ok_z], 1e-300, 0.5)
        abs_z = norm.isf(half_p)
        z_scores[ok_z] = np.sign(tstat[ok_z]) * abs_z

    mu0, sigma0 = estimate_empirical_null(z_scores, settings=settings)
    z_emp = (z_scores - mu0) / sigma0
    pval_emp = np.where(np.isfinite(z_emp), 2.0 * norm.sf(np.abs(z_emp)), np.nan)
    fdr_emp = _bh_fdr(pval_emp)

    tx_ids = fit.coefficients.index
    df_out = pd.DataFrame(
        {
            "transcript_id": tx_ids.astype(str),
            "gene_id": fit.tx2gene.set_index("transcript_id")
            .reindex(tx_ids)["gene_id"]
            .to_numpy(),
            "estimates": estimates,
            "se": se,
            "s2_post": eb_res.s2_post,
            "df_total": df_tot,
            "tstat": tstat,
            "z_score": z_scores,
            "pval_regular": pval_reg,
            "fdr_regular": fdr_reg,
            "pval_empirical": pval_emp,
            "fdr_empirical": fdr_emp,
            "converged": fit.converged.to_numpy(),
            "boundary": fit.boundary.to_numpy(),
        },
        index=tx_ids,
    )
    df_out.attrs["mu0"] = mu0
    df_out.attrs["sigma0"] = sigma0
    df_out.attrs["s2_prior"] = eb_res.s2_prior
    df_out.attrs["df_prior"] = eb_res.df_prior
    if diag_plot:
        z_fin = z_scores[np.isfinite(z_scores)]
        grid_x = np.linspace(-5.0, 5.0, 201)
        df_out.attrs["diag_data"] = {
            "z_grid": grid_x,
            "theoretical_null_pdf": norm.pdf(grid_x, loc=0.0, scale=1.0),
            "empirical_null_pdf": norm.pdf(grid_x, loc=mu0, scale=sigma0),
            "n_valid_z": int(len(z_fin)),
            "mu0": mu0,
            "sigma0": sigma0,
        }
    return df_out


def test_dtu(
    fit: DTUFit,
    contrast: str | list[float] | np.ndarray | pd.DataFrame | dict[str, list[float] | np.ndarray],
    diag_plot: bool = False,
    sort: bool = False,
    settings: SettingsMode | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Perform moderated quasi-t DTU testing with empirical null recalibration (`satuRn::testDTU`).

    Parameters
    ----------
    fit
        Fitted ``DTUFit`` returned by :func:`fit_dtu`.
    contrast
        Contrast specification: design column name ``str``, 1D numeric contrast
        vector of length ``p``, ``dict[name, vector]``, or 2D contrast matrix
        ``(p, n_contrasts)``.
    diag_plot
        If ``True``, attaches theoretical vs empirical null density curve data
        to ``result.attrs["diag_data"]``.
    sort
        Whether to sort results ascending by ``pval_empirical``.
    settings
        ``"default"`` or ``"R_default"`` (defaults to ``fit.settings`` if ``None``,
        or validated when explicitly provided).

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
    """
    eff_settings = fit.settings if settings is None else validate_settings(settings)
    eb_res = squeeze_var(
        fit.dispersion.to_numpy(dtype=np.float64),
        fit.df_residual.to_numpy(dtype=np.float64),
        settings=eff_settings,
    )

    p_cov = fit.design.shape[1]
    col_names = list(map(str, fit.design.columns))

    if isinstance(contrast, str):
        if contrast not in col_names:
            raise ValueError(
                f"Contrast column {contrast!r} not in design columns {col_names}."
            )
        c_vec = np.zeros(p_cov, dtype=np.float64)
        c_vec[col_names.index(contrast)] = 1.0
        res = _test_single_contrast(fit, c_vec, eb_res, diag_plot, eff_settings)
        return res.sort_values("pval_empirical") if sort else res

    if isinstance(contrast, dict):
        out_dict: dict[str, pd.DataFrame] = {}
        for name, vec in contrast.items():
            c_arr = np.asarray(vec, dtype=np.float64).ravel()
            if len(c_arr) != p_cov:
                raise ValueError(
                    f"Contrast {name!r} has length {len(c_arr)}; expected {p_cov}."
                )
            df_c = _test_single_contrast(fit, c_arr, eb_res, diag_plot, eff_settings)
            out_dict[name] = df_c.sort_values("pval_empirical") if sort else df_c
        return out_dict

    if isinstance(contrast, pd.DataFrame):
        c_mat = contrast.to_numpy(dtype=np.float64)
        if c_mat.shape[0] != p_cov and c_mat.shape[1] == p_cov:
            c_mat = c_mat.T
            c_names = list(map(str, contrast.index))
        else:
            c_names = list(map(str, contrast.columns))
        if c_mat.shape[0] != p_cov:
            raise ValueError(
                f"Contrast matrix shape {contrast.shape} incompatible with {p_cov} covariates."
            )
        if c_mat.shape[1] == 1:
            res = _test_single_contrast(fit, c_mat[:, 0], eb_res, diag_plot, eff_settings)
            return res.sort_values("pval_empirical") if sort else res
        return {
            c_names[j]: (
                _test_single_contrast(fit, c_mat[:, j], eb_res, diag_plot, eff_settings).sort_values(
                    "pval_empirical"
                )
                if sort
                else _test_single_contrast(fit, c_mat[:, j], eb_res, diag_plot, eff_settings)
            )
            for j in range(c_mat.shape[1])
        }

    c_arr = np.asarray(contrast, dtype=np.float64)
    if c_arr.ndim == 1:
        if len(c_arr) != p_cov:
            raise ValueError(
                f"Contrast vector has length {len(c_arr)}; expected {p_cov}."
            )
        res = _test_single_contrast(fit, c_arr, eb_res, diag_plot, eff_settings)
        return res.sort_values("pval_empirical") if sort else res

    if c_arr.ndim == 2:
        if c_arr.shape[0] != p_cov and c_arr.shape[1] == p_cov:
            c_arr = c_arr.T
        if c_arr.shape[0] != p_cov:
            raise ValueError(
                f"Contrast matrix shape {c_arr.shape} incompatible with {p_cov} covariates."
            )
        if c_arr.shape[1] == 1:
            res = _test_single_contrast(fit, c_arr[:, 0], eb_res, diag_plot, eff_settings)
            return res.sort_values("pval_empirical") if sort else res
        return {
            f"contrast_{j}": (
                _test_single_contrast(fit, c_arr[:, j], eb_res, diag_plot, eff_settings).sort_values(
                    "pval_empirical"
                )
                if sort
                else _test_single_contrast(fit, c_arr[:, j], eb_res, diag_plot, eff_settings)
            )
            for j in range(c_arr.shape[1])
        }

    raise ValueError(f"Unsupported contrast specification: {type(contrast)}.")


# Prevent pytest from collecting `test_dtu` as a test function when imported into test modules
test_dtu.__test__ = False  # type: ignore[attr-defined]


def stagewise_dtu(
    test_res_df: pd.DataFrame,
    alpha: float = 0.05,
    pval_col: str = "pval_empirical",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Two-stage gene Simes screening + Shaffer DTU transcript confirmation (`stageR`).

    Controls the gene-level overall false discovery rate (OFDR) at level ``alpha``
    following Van den Berge et al. (*Genome Biology* 2017).
    """
    validate_settings(settings)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1); got {alpha}.")
    if pval_col not in test_res_df.columns:
        if "pval_empirical" in test_res_df.columns:
            pval_col = "pval_empirical"
        elif "pval_regular" in test_res_df.columns:
            pval_col = "pval_regular"
        else:
            raise ValueError(f"Column {pval_col!r} not found in test_res_df.")

    df = test_res_df.copy()
    if "transcript_id" not in df.columns:
        df.insert(0, "transcript_id", df.index.astype(str))
    if "gene_id" not in df.columns:
        raise ValueError("test_res_df must contain a 'gene_id' column.")

    # Stage 1: Simes p-value per gene
    p_raw = df[pval_col].to_numpy(dtype=np.float64)
    gene_ids = df["gene_id"].astype(str).to_numpy()
    unique_genes, inv_gene = np.unique(gene_ids, return_inverse=True)
    n_genes_total = len(unique_genes)

    gene_simes = np.full(n_genes_total, np.nan, dtype=np.float64)
    gene_to_rows: list[np.ndarray] = []
    for g_i in range(n_genes_total):
        rows = np.flatnonzero(inv_gene == g_i)
        gene_to_rows.append(rows)
        pg = p_raw[rows]
        pg_valid = pg[np.isfinite(pg)]
        m_g = len(pg_valid)
        if m_g == 0:
            continue
        pg_sorted = np.sort(np.clip(pg_valid, 0.0, 1.0))
        ranks = np.arange(1, m_g + 1, dtype=np.float64)
        gene_simes[g_i] = float(np.min(np.minimum(1.0, pg_sorted * (m_g / ranks))))

    padj_gene_unique = _bh_fdr(gene_simes)
    n_tested_genes = int(np.sum(np.isfinite(padj_gene_unique)))
    screened_genes = np.isfinite(padj_gene_unique) & (padj_gene_unique <= alpha)
    n_screened = int(np.sum(screened_genes))

    pval_gene_col = gene_simes[inv_gene]
    padj_gene_col = padj_gene_unique[inv_gene]
    padj_tx_col = np.full(len(df), np.nan, dtype=np.float64)
    tx_sig_col = np.zeros(len(df), dtype=bool)

    if n_screened > 0 and n_tested_genes > 0:
        alpha_stage2 = alpha * (float(n_screened) / float(n_tested_genes))
        scale_back = float(n_tested_genes) / float(n_screened)
    else:
        alpha_stage2 = 0.0
        scale_back = np.inf

    for g_i in range(n_genes_total):
        rows = gene_to_rows[g_i]
        pg = p_raw[rows]
        valid_local = np.isfinite(pg)
        if not np.any(valid_local):
            continue

        if not screened_genes[g_i]:
            padj_tx_col[rows[valid_local]] = 1.0
            continue

        valid_rows = rows[valid_local]
        pv = np.clip(pg[valid_local], 0.0, 1.0)
        m_g = len(pv)

        if m_g <= 2:
            # Under DTU sum-to-1 logic, rejecting a 2-isoform gene implies both shift
            padj_tx_col[valid_rows] = padj_gene_unique[g_i]
            tx_sig_col[valid_rows] = True
        else:
            order = np.argsort(pv)
            pv_sorted = pv[order]
            ranks = np.arange(1, m_g + 1, dtype=np.int64)
            # Shaffer DTU step-down multipliers: max(1, min(m_g - 2, m_g - k + 1))
            mult = np.maximum(1, np.minimum(m_g - 2, m_g - ranks + 1)).astype(np.float64)
            shaffer_p = np.minimum(1.0, np.maximum.accumulate(pv_sorted * mult))
            confirmed_sorted = shaffer_p <= alpha_stage2
            # Enforce strict step-down stopping rule
            if not np.all(confirmed_sorted):
                first_fail = int(np.argmax(~confirmed_sorted))
                confirmed_sorted[first_fail:] = False

            rescaled = np.minimum(1.0, np.maximum(padj_gene_unique[g_i], shaffer_p * scale_back))
            inv_ord = np.empty_like(order)
            inv_ord[order] = np.arange(m_g)
            padj_tx_col[valid_rows] = rescaled[inv_ord]
            tx_sig_col[valid_rows] = confirmed_sorted[inv_ord]

    gene_sig_col = np.isfinite(padj_gene_col) & (padj_gene_col <= alpha)
    out = pd.DataFrame(
        {
            "transcript_id": df["transcript_id"].to_numpy(),
            "gene_id": gene_ids,
            "pval_transcript": p_raw,
            "pval_gene": pval_gene_col,
            "padj_gene": padj_gene_col,
            "padj_transcript": padj_tx_col,
            "gene_significant": gene_sig_col,
            "tx_significant": tx_sig_col,
        },
        index=df.index,
    )
    return out
