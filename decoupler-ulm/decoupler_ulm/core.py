"""
Bias-corrected projection of single-cell cNMF / NMF Gene Expression Programs (GEPs)
onto bulk or pseudobulk RNA-seq matrices.

Replaces un-logged linear TPM dot products (which suffer extreme distortion from top
housekeeping/collagen outliers) with Univariate Linear Model (`ulm_tstat`, equivalent
to `decoupler.run_ulm`), `log2_dot`, and `top_k_zscore` projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def project_gep_programs(
    expr_genes_by_samples: pd.DataFrame,
    program_weights_genes_by_programs: pd.DataFrame,
    is_log_transformed: bool = False,
    top_k_markers: int = 50,
    method: str = "ulm_tstat",
) -> pd.DataFrame:
    """Project continuous gene-program weights (`genes x programs`) onto an expression matrix (`genes x samples`).

    Parameters
    ----------
    expr_genes_by_samples : DataFrame (genes x samples)
        Bulk or pseudobulk RNA-seq matrix (raw TPM/CPM if `is_log_transformed=False`, or log2(TPM+1)).
    program_weights_genes_by_programs : DataFrame (genes x programs)
        Continuous cNMF / NMF gene spectra / meta-gene weights.
    method : {'ulm_tstat', 'log2_dot', 'top_k_zscore'}
        - `'ulm_tstat'`: Regresses gene-wise Z-scored log2 expression on continuous program weights
          and returns the exact ordinary-least-squares t-statistic (`samples x programs`).
        - `'log2_dot'`: Matrix product of log2 expression with L1-normalized positive weights.
        - `'top_k_zscore'`: Mean gene-wise Z-score across the top-`top_k_markers` weighted genes per program.
    """
    expr = expr_genes_by_samples.copy().astype(np.float64)
    weights = program_weights_genes_by_programs.copy().astype(np.float64)
    expr.index = expr.index.astype(str).str.upper()
    weights.index = weights.index.astype(str).str.upper()

    expr = expr.groupby(level=0).mean()
    weights = weights.groupby(level=0).mean()

    shared_genes = [g for g in weights.index if g in expr.index]
    if len(shared_genes) < 10:
        raise ValueError(f"Fewer than 10 shared genes ({len(shared_genes)}) between expression and program weights.")

    X_log = expr.loc[shared_genes].to_numpy(dtype=np.float64)
    if not is_log_transformed:
        X_log = np.log2(np.maximum(X_log, 0.0) + 1.0)
    W = weights.loc[shared_genes].to_numpy(dtype=np.float64)  # (G, P)
    samples = list(expr.columns)
    programs = list(weights.columns)

    if method == "log2_dot":
        w_pos = np.maximum(W, 0.0)
        w_norm = w_pos / np.maximum(w_pos.sum(axis=0, keepdims=True), 1e-12)
        scores = X_log.T @ w_norm  # (S, P)
        return pd.DataFrame(scores, index=samples, columns=programs)

    # Gene-wise Z-score across samples: (G, S)
    g_mean = X_log.mean(axis=1, keepdims=True)
    g_std = X_log.std(axis=1, ddof=1, keepdims=True)
    valid_g = (g_std.ravel() > 1e-12)
    Z = np.zeros_like(X_log)
    Z[valid_g] = (X_log[valid_g] - g_mean[valid_g]) / g_std[valid_g]

    if method == "top_k_zscore":
        scores = np.zeros((len(samples), len(programs)), dtype=np.float64)
        k_eff = min(top_k_markers, len(shared_genes))
        for p_i in range(len(programs)):
            top_idx = np.argsort(-W[:, p_i])[:k_eff]
            scores[:, p_i] = Z[top_idx, :].mean(axis=0)
        return pd.DataFrame(scores, index=samples, columns=programs)

    if method == "ulm_tstat":
        Z_sub = Z[valid_g, :]  # (G_val, S)
        W_sub = W[valid_g, :]  # (G_val, P)
        g_cnt = Z_sub.shape[0]
        if g_cnt < 5:
            raise ValueError("Fewer than 5 variable genes across samples for ULM regression.")
        df = g_cnt - 2
        zc = Z_sub - Z_sub.mean(axis=0, keepdims=True)
        wc = W_sub - W_sub.mean(axis=0, keepdims=True)
        z_norm = np.sqrt(np.maximum((zc**2).sum(axis=0), 1e-300))  # (S,)
        w_norm = np.sqrt(np.maximum((wc**2).sum(axis=0), 1e-300))  # (P,)
        r_mat = (zc.T @ wc) / (z_norm[:, None] * w_norm[None, :])   # (S, P)
        r_mat = np.clip(r_mat, -0.9999999, 0.9999999)
        t_mat = r_mat * np.sqrt(df / (1.0 - r_mat**2))
        return pd.DataFrame(t_mat, index=samples, columns=programs)

    raise ValueError(f"Unsupported method {method!r}; choose from 'ulm_tstat', 'log2_dot', 'top_k_zscore'.")
