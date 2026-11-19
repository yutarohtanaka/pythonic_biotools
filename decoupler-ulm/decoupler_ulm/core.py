"""
Bias-corrected projection of single-cell cNMF / NMF Gene Expression Programs (GEPs)
onto bulk or pseudobulk RNA-seq matrices.

Replaces un-logged linear TPM dot products (which suffer extreme distortion from top
housekeeping/collagen outliers) with Univariate Linear Model (`ulm_tstat`, equivalent
to `decoupler.run_ulm`), `log2_dot`, and `top_k_zscore` projections.
"""

from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy import stats

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(f"settings must be 'default' or 'R_default', got {settings!r}.")
    return settings  # type: ignore[return-value]


def _compute_gene_zscores(X_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe gene-wise Z-score across samples (or within-sample across genes if n_samples == 1)."""
    n_genes, n_samples = X_log.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if n_samples == 1:
            finite = np.isfinite(X_log[:, 0])
            if int(finite.sum()) < 2:
                return np.full_like(X_log, np.nan), np.zeros(n_genes, dtype=bool)
            mu = float(np.nanmean(X_log[:, 0]))
            sd = float(np.nanstd(X_log[:, 0], ddof=1))
            if not np.isfinite(sd) or sd <= 1e-12:
                Z = np.where(np.isfinite(X_log), 0.0, np.nan)
                return Z, np.zeros(n_genes, dtype=bool)
            Z = np.where(np.isfinite(X_log), (X_log - mu) / sd, np.nan)
            return Z, finite

        n_obs = np.sum(np.isfinite(X_log), axis=1)
        g_mean = np.nanmean(X_log, axis=1, keepdims=True)
        g_std = np.nanstd(X_log, axis=1, ddof=1, keepdims=True)
        valid_g = (n_obs >= 2) & np.isfinite(g_std.ravel()) & (g_std.ravel() > 1e-12)

        Z = np.where(np.isfinite(X_log), 0.0, np.nan)
        if np.any(valid_g):
            Z[valid_g] = (X_log[valid_g] - g_mean[valid_g]) / g_std[valid_g]
        return Z, valid_g


def _ols_ulm_tstat_and_pvals(
    Y: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-(sample, program) OLS t-statistics and two-sided Student-t p-values."""
    n_genes, n_samples = Y.shape
    _, n_programs = W.shape

    if np.all(np.isfinite(Y)) and np.all(np.isfinite(W)):
        if n_genes < 3:
            raise ValueError("Need at least 3 shared genes for ULM regression.")
        df = float(n_genes - 2)
        yc = Y - Y.mean(axis=0, keepdims=True)
        wc = W - W.mean(axis=0, keepdims=True)
        y_ss = (yc**2).sum(axis=0)
        w_ss = (wc**2).sum(axis=0)
        denom = np.sqrt(np.maximum(y_ss[:, None] * w_ss[None, :], 1e-300))
        zero_var = (y_ss[:, None] <= 1e-24) | (w_ss[None, :] <= 1e-24)
        r_mat = np.where(zero_var, 0.0, (yc.T @ wc) / denom)
        r_mat = np.clip(r_mat, -1.0 + 1e-12, 1.0 - 1e-12)
        t_mat = r_mat * np.sqrt(df / np.maximum(1.0 - r_mat**2, 1e-24))
        p_mat = 2.0 * stats.t.sf(np.abs(t_mat), df=df)
        return t_mat, p_mat

    t_mat = np.full((n_samples, n_programs), np.nan, dtype=np.float64)
    p_mat = np.full((n_samples, n_programs), np.nan, dtype=np.float64)
    fin_y = np.isfinite(Y)
    fin_w = np.isfinite(W)

    for s_idx in range(n_samples):
        ys = Y[:, s_idx]
        fy = fin_y[:, s_idx]
        for p_idx in range(n_programs):
            mask = fy & fin_w[:, p_idx]
            n_m = int(mask.sum())
            if n_m < 3:
                continue
            yv = ys[mask]
            wv = W[mask, p_idx]
            yc = yv - float(yv.mean())
            wc = wv - float(wv.mean())
            y_ss = float(np.sum(yc * yc))
            w_ss = float(np.sum(wc * wc))
            if y_ss <= 1e-24 or w_ss <= 1e-24:
                t_mat[s_idx, p_idx] = 0.0
                p_mat[s_idx, p_idx] = 1.0
                continue
            r = float(np.clip(np.sum(yc * wc) / np.sqrt(y_ss * w_ss), -1.0 + 1e-12, 1.0 - 1e-12))
            df = float(n_m - 2)
            t_val = r * np.sqrt(df / max(1.0 - r * r, 1e-24))
            t_mat[s_idx, p_idx] = t_val
            p_mat[s_idx, p_idx] = float(2.0 * stats.t.sf(abs(t_val), df=df))

    return t_mat, p_mat


def project_gep_programs(
    expr_genes_by_samples: pd.DataFrame,
    program_weights_genes_by_programs: pd.DataFrame,
    is_log_transformed: bool = False,
    top_k_markers: int = 50,
    method: str = "ulm_tstat",
    center_across_samples: bool = False,
    zscore_genes: bool | None = None,
    return_pvalues: bool = False,
    *,
    settings: SettingsMode = "default",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Project continuous gene-program weights (`genes x programs`) onto an expression matrix (`genes x samples`).

    Parameters
    ----------
    expr_genes_by_samples : DataFrame (genes x samples)
        Bulk or pseudobulk RNA-seq matrix (raw TPM/CPM if `is_log_transformed=False`, or log2(TPM+1)).
    program_weights_genes_by_programs : DataFrame (genes x programs)
        Continuous cNMF / NMF gene spectra / meta-gene weights.
    method : {'ulm_tstat', 'log2_dot', 'top_k_zscore'}
        - `'ulm_tstat'`: Regresses sample log2 expression (exact single-sample `decoupler.run_ulm` mode
          when `center_across_samples=False`, or gene-wise Z-scored expression across samples when
          `center_across_samples=True`) on continuous program weights and returns the exact OLS t-statistic.
        - `'log2_dot'`: NaN-safe L1-renormalized matrix product of log2 expression with positive weights.
        - `'top_k_zscore'`: NaN-safe mean gene-wise Z-score across the top-`top_k_markers` weighted genes.
    center_across_samples : bool, default False
        When `False` (default), `'ulm_tstat'` runs directly on each sample's log-expression vector
        independently (`decoupler.run_ulm`-exact, supporting `n_samples == 1`). When `True`, genes are
        first Z-scored across samples (`ddof=1`).
    zscore_genes : bool, optional
        Backward-compatible alias for `center_across_samples` if explicitly supplied.
    return_pvalues : bool, default False
        When `True`, returns `(scores_df, pvalues_df)`.
    settings : {'default', 'R_default'}, default 'default'
        When `'R_default'`, matches `saezlab/decoupleR::run_ulm` exactly: uses `expr_genes_by_samples`
        directly on its input scale (no auto `log2(X + 1)`), pads all features in `expr_genes_by_samples`
        absent from `program_weights_genes_by_programs` with `0.0` weights (requiring `minsize >= 5`
        shared targets rather than `>= 10` subsetting), and forces `center_across_samples = False`.
    """
    _validate_settings(settings)
    if zscore_genes is not None:
        center_across_samples = bool(zscore_genes)
    if settings == "R_default":
        center_across_samples = False

    expr = expr_genes_by_samples.copy().astype(np.float64)
    weights = program_weights_genes_by_programs.copy().astype(np.float64)
    expr.index = expr.index.astype(str).str.upper()
    weights.index = weights.index.astype(str).str.upper()

    expr = expr.groupby(level=0).mean()
    weights = weights.groupby(level=0).mean()

    shared_genes = [g for g in weights.index if g in expr.index]
    if settings == "R_default":
        if len(shared_genes) < 5:
            raise ValueError(
                f"Fewer than 5 shared genes ({len(shared_genes)}) between expression and program weights."
            )
        X_log = expr.to_numpy(dtype=np.float64)
        W = weights.reindex(expr.index, fill_value=0.0).to_numpy(dtype=np.float64)
    else:
        if len(shared_genes) < 10:
            raise ValueError(
                f"Fewer than 10 shared genes ({len(shared_genes)}) between expression and program weights."
            )
        X_raw = expr.loc[shared_genes].to_numpy(dtype=np.float64)
        if is_log_transformed:
            X_log = X_raw
        else:
            X_log = np.where(np.isfinite(X_raw), np.log2(np.maximum(X_raw, 0.0) + 1.0), np.nan)
        W = weights.loc[shared_genes].to_numpy(dtype=np.float64)  # (G, P)
    samples = list(expr.columns)
    programs = list(weights.columns)

    if method == "log2_dot":
        if return_pvalues:
            raise ValueError("return_pvalues=True is not supported for method='log2_dot'.")
        w_pos = np.where(np.isfinite(W) & (W > 0.0), W, 0.0)
        finite_x = np.isfinite(X_log)
        X_zero = np.where(finite_x, X_log, 0.0)
        num = X_zero.T @ w_pos  # (S, P)
        den = finite_x.astype(np.float64).T @ w_pos  # (S, P)
        any_obs_sample = finite_x.any(axis=0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.where(any_obs_sample, 0.0, np.nan))
        return pd.DataFrame(scores, index=samples, columns=programs)

    if method == "top_k_zscore":
        Z, _ = _compute_gene_zscores(X_log)
        scores = np.full((len(samples), len(programs)), np.nan, dtype=np.float64)
        pvals = np.full((len(samples), len(programs)), np.nan, dtype=np.float64)
        for p_i in range(len(programs)):
            w_col = W[:, p_i]
            fin_idx = np.flatnonzero(np.isfinite(w_col))
            if fin_idx.size == 0:
                continue
            k_eff = min(top_k_markers, fin_idx.size)
            top_idx = fin_idx[np.argsort(-w_col[fin_idx], kind="mergesort")[:k_eff]]
            Z_top = Z[top_idx, :]
            n_k = np.sum(np.isfinite(Z_top), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_z = np.nanmean(Z_top, axis=0)
            scores[:, p_i] = np.where(n_k >= 1, mean_z, np.nan)
            if return_pvalues:
                pvals[:, p_i] = np.where(n_k >= 1, 2.0 * stats.norm.sf(np.abs(mean_z) * np.sqrt(n_k)), np.nan)
        scores_df = pd.DataFrame(scores, index=samples, columns=programs)
        if return_pvalues:
            return scores_df, pd.DataFrame(pvals, index=samples, columns=programs)
        return scores_df

    if method == "ulm_tstat":
        if center_across_samples:
            if X_log.shape[1] < 2:
                raise ValueError("center_across_samples=True requires at least 2 samples.")
            Z, valid_g = _compute_gene_zscores(X_log)
            if int(valid_g.sum()) < 5:
                raise ValueError("Fewer than 5 variable genes across samples for centered ULM regression.")
            Y_reg = Z[valid_g, :]
            W_reg = W[valid_g, :]
        else:
            Y_reg = X_log
            W_reg = W
            if Y_reg.shape[0] < 5:
                raise ValueError("Fewer than 5 shared genes for ULM regression.")

        t_mat, p_mat = _ols_ulm_tstat_and_pvals(Y_reg, W_reg)
        scores_df = pd.DataFrame(t_mat, index=samples, columns=programs)
        if return_pvalues:
            return scores_df, pd.DataFrame(p_mat, index=samples, columns=programs)
        return scores_df

    raise ValueError(f"Unsupported method {method!r}; choose from 'ulm_tstat', 'log2_dot', 'top_k_zscore'.")
