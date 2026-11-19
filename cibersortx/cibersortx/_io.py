"""
Input parsing, orientation normalization, sparse extraction, gene alignment,
and NaN-aware multiple-testing utilities for `cibersortx`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

ExpressionInput = ad.AnnData | pd.DataFrame | np.ndarray


def _validate_unique_names(names: Sequence[str], kind: str) -> list[str]:
    out = [str(x) for x in names]
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate {kind} names detected; {kind} names must be unique.")
    return out


def _extract_anndata_matrix(
    adata: ad.AnnData,
    layer: str | None = None,
) -> tuple[np.ndarray | sparse.spmatrix, list[str], list[str]]:
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData.layers.")
        mat = adata.layers[layer]
    else:
        mat = adata.X
    if mat is None:
        raise ValueError("AnnData expression matrix is None; provide non-empty `adata.X` or a valid `layer`.")
    obs_names = _validate_unique_names(list(adata.obs_names), "sample/cell")
    gene_names = _validate_unique_names(list(adata.var_names), "gene")
    return mat, gene_names, obs_names


def _to_dense_float64(mat: Any) -> np.ndarray:
    if sparse.issparse(mat):
        return np.asarray(mat.toarray(), dtype=np.float64)
    return np.asarray(mat, dtype=np.float64)


def parse_expression_matrix(
    data: ExpressionInput,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    default_row_prefix: str = "gene",
    default_col_prefix: str = "sample",
    allow_1d: bool = False,
    name: str = "expression",
    is_log2: bool = False,
    log2_offset: float = 0.0,
    allow_negative: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Parse `AnnData` (`obs x var`), `pd.DataFrame`, or `np.ndarray` into a
    `(genes, obs)` dense `float64` matrix with validated gene and obs names.
    """
    if isinstance(data, ad.AnnData):
        raw, gene_names, obs_names = _extract_anndata_matrix(data, layer=layer)
        arr = _to_dense_float64(raw).T  # (var=genes, obs=samples/cells)
    elif isinstance(data, pd.DataFrame):
        if layer is not None:
            raise ValueError("`layer` is only supported when passing an AnnData object.")
        if any(isinstance(dt, pd.SparseDtype) for dt in data.dtypes):
            arr = data.sparse.to_dense().to_numpy(dtype=np.float64)
        else:
            arr = data.to_numpy(dtype=np.float64)
        if samples_as_rows:
            arr = arr.T
            gene_names = _validate_unique_names(list(data.columns), "gene")
            obs_names = _validate_unique_names(list(data.index), default_col_prefix)
        else:
            gene_names = _validate_unique_names(list(data.index), "gene")
            obs_names = _validate_unique_names(list(data.columns), default_col_prefix)
    else:
        if layer is not None:
            raise ValueError("`layer` is only supported when passing an AnnData object.")
        arr = _to_dense_float64(data)
        if arr.ndim == 1:
            if not allow_1d:
                raise ValueError(f"{name} must be a 2D array, got 1D shape {arr.shape}.")
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
        elif samples_as_rows:
            arr = arr.T
        n_genes, n_obs = arr.shape
        gene_names = [f"{default_row_prefix}_{i}" for i in range(n_genes)]
        obs_names = [f"{default_col_prefix}_{j}" for j in range(n_obs)]

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    if is_log2:
        arr = np.maximum(np.power(2.0, arr) - float(log2_offset), 0.0)
    elif not allow_negative and np.min(arr) < 0.0:
        raise ValueError(
            f"{name} contains negative values (min={float(np.min(arr)):.4g}). "
            "Set `is_log2=True` if values are log2-transformed."
        )

    return np.ascontiguousarray(arr, dtype=np.float64), gene_names, obs_names


def parse_sc_reference_sparse_aware(
    reference: ExpressionInput,
    *,
    layer: str | None = None,
    cells_as_rows: bool = False,
    is_log2: bool = False,
    log2_offset: float = 0.0,
) -> tuple[np.ndarray | sparse.csr_matrix, list[str], list[str], bool]:
    """
    Parse single-cell reference into `(cells, genes)` orientation, preserving
    CSR sparsity when given a sparse `AnnData` in linear space.

    Returns `(mat_cells_by_genes, gene_names, cell_names, is_sparse)`.
    """
    if isinstance(reference, ad.AnnData):
        raw, gene_names, cell_names = _extract_anndata_matrix(reference, layer=layer)
        if sparse.issparse(raw):
            csr = raw.tocsr().astype(np.float64, copy=True)
            if csr.shape[0] == 0 or csr.shape[1] == 0:
                raise ValueError(f"reference must be non-empty, got shape {csr.shape}.")
            if csr.nnz > 0:
                if not np.all(np.isfinite(csr.data)):
                    raise ValueError("reference contains NaN or Inf values.")
                if is_log2:
                    # Log2 conversion of sparse matrix shifts zeros to max(1 - offset, 0);
                    # if offset != 1, convert to dense cleanly.
                    if abs(log2_offset - 1.0) <= 1e-12:
                        csr.data = np.maximum(np.power(2.0, csr.data) - 1.0, 0.0)
                    else:
                        arr = np.maximum(np.power(2.0, csr.toarray()) - float(log2_offset), 0.0)
                        return arr, gene_names, cell_names, False
                elif np.min(csr.data) < 0.0:
                    raise ValueError(
                        f"reference contains negative values (min={float(np.min(csr.data)):.4g})."
                    )
            return csr, gene_names, cell_names, True

    mat_g_by_c, gene_names, cell_names = parse_expression_matrix(
        reference,
        layer=layer,
        samples_as_rows=cells_as_rows,
        default_row_prefix="gene",
        default_col_prefix="cell",
        allow_1d=False,
        name="reference",
        is_log2=is_log2,
        log2_offset=log2_offset,
        allow_negative=False,
    )
    return np.ascontiguousarray(mat_g_by_c.T), gene_names, cell_names, False


def parse_label_vector(
    data: Any,
    labels: str | Sequence[Any] | np.ndarray | pd.Series,
    *,
    n_obs: int,
    kind: str = "cell_type_labels",
) -> list[str]:
    """Resolve `obs` column name (for AnnData) or 1D label sequence into `list[str]`."""
    if isinstance(labels, str):
        if not isinstance(data, ad.AnnData):
            raise ValueError(
                f"`{kind}` was passed as a column name '{labels}', which requires an AnnData input."
            )
        if labels not in data.obs.columns:
            raise ValueError(f"Column '{labels}' not found in AnnData.obs.")
        series = data.obs[labels]
        if series.isna().any():
            raise ValueError(f"AnnData.obs['{labels}'] contains missing values.")
        out = [str(x) for x in series.tolist()]
    else:
        if isinstance(labels, pd.Series):
            if labels.isna().any():
                raise ValueError(f"`{kind}` contains missing values.")
            vals = labels.tolist()
        else:
            arr = np.asarray(labels, dtype=object)
            if arr.ndim != 1:
                raise ValueError(f"`{kind}` must be 1D, got shape {arr.shape}.")
            vals = arr.tolist()
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            raise ValueError(f"`{kind}` contains NaN or None values.")
        out = [str(v) for v in vals]

    if len(out) != n_obs:
        raise ValueError(
            f"Length of `{kind}` ({len(out)}) does not match number of observations ({n_obs})."
        )
    return out


def parse_signature_matrix(
    signature: Any,
    *,
    is_log2: bool = False,
    log2_offset: float = 0.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Parse signature matrix from `SignatureMatrixResult`, `pd.DataFrame`
    (`genes x cell_types`), `np.ndarray` (`genes x cell_types`), or `AnnData`
    (`cell_types x genes` in `obs x var`).
    """
    if hasattr(signature, "signature") and hasattr(signature, "gene_names") and hasattr(signature, "cell_types"):
        mat = np.asarray(signature.signature, dtype=np.float64)
        gene_names = _validate_unique_names(list(signature.gene_names), "gene")
        cell_types = _validate_unique_names(list(signature.cell_types), "cell_type")
        if mat.shape != (len(gene_names), len(cell_types)):
            raise ValueError("SignatureMatrixResult has inconsistent matrix shape.")
        if not np.all(np.isfinite(mat)):
            raise ValueError("Signature matrix contains NaN or Inf values.")
        if is_log2:
            mat = np.maximum(np.power(2.0, mat) - float(log2_offset), 0.0)
        elif np.min(mat) < 0.0:
            raise ValueError("Signature matrix contains negative values.")
        return np.ascontiguousarray(mat, dtype=np.float64), gene_names, cell_types

    return parse_expression_matrix(
        signature,
        samples_as_rows=False,
        default_row_prefix="gene",
        default_col_prefix="cell_type",
        allow_1d=False,
        name="signature",
        is_log2=is_log2,
        log2_offset=log2_offset,
        allow_negative=False,
    )


def align_genes(
    genes_primary: Sequence[str],
    genes_secondary: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Intersect two unique gene lists while deterministically preserving
    `genes_primary` order.
    """
    idx_a = pd.Index(genes_primary)
    idx_b = pd.Index(genes_secondary)
    common = idx_a.intersection(idx_b, sort=False)
    shared_genes = [str(g) for g in common]
    pos_a = idx_a.get_indexer(common)
    pos_b = idx_b.get_indexer(common)
    return shared_genes, pos_a, pos_b


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    NaN-aware Benjamini-Hochberg FDR correction across finite p-values only.
    Non-finite entries remain `np.nan` and do not inflate the test count `m`.
    """
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(p)
    if np.any(finite_mask):
        p_valid = np.clip(p[finite_mask], 0.0, 1.0)
        q[finite_mask] = stats.false_discovery_control(p_valid, method="bh")
    return q
