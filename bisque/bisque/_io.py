"""Unified input parsing, validation, and sparse-safe CPM utilities for `bisque`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ExpressionInput = ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix


@dataclass(frozen=True)
class ParsedExpression:
    """Validated expression matrix in (observations x genes) orientation."""

    matrix: np.ndarray | sparse.csr_matrix
    obs_names: list[str]
    var_names: list[str]
    is_anndata: bool
    obs_df: pd.DataFrame | None


def _validate_numeric_values(
    matrix: np.ndarray | sparse.csr_matrix,
    *,
    name: str,
    require_nonnegative: bool,
) -> None:
    if sparse.issparse(matrix):
        data = matrix.data
        if data.size > 0:
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name} contains NaN or Inf values.")
            if require_nonnegative and np.any(data < 0):
                raise ValueError(
                    f"{name} contains negative values, which are invalid when normalize_cpm=True."
                )
    else:
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains NaN or Inf values.")
        if require_nonnegative and np.any(matrix < 0):
            raise ValueError(
                f"{name} contains negative values, which are invalid when normalize_cpm=True."
            )


def _coerce_name_list(
    names: Sequence[str] | None,
    expected_len: int,
    default_prefix: str,
    kind: str,
) -> list[str]:
    if names is None:
        return [f"{default_prefix}_{i}" for i in range(expected_len)]
    out = [str(x) for x in names]
    if len(out) != expected_len:
        raise ValueError(
            f"Length of {kind} ({len(out)}) does not match matrix dimension ({expected_len})."
        )
    return out


def parse_expression_input(
    data: ExpressionInput,
    *,
    name: str,
    layer: str | None = None,
    sample_names: Sequence[str] | None = None,
    gene_names: Sequence[str] | None = None,
    obs_prefix: str = "sample",
    allow_1d: bool = False,
    require_nonnegative: bool = True,
) -> ParsedExpression:
    """
    Parse AnnData, DataFrame, ndarray, or scipy.sparse matrix into (obs x var).

    Parameters
    ----------
    data
        Expression matrix in ``obs x var`` (samples/cells x genes) orientation.
    name
        Human-readable label for error messages (e.g. ``"bulk"`` or ``"sc"``).
    layer
        Optional layer name when ``data`` is an ``AnnData`` object.
    sample_names
        Optional override for observation names.
    gene_names
        Optional override for gene names.
    obs_prefix
        Prefix used when synthesizing default observation names for raw arrays.
    allow_1d
        If True, a 1D ``np.ndarray`` of shape ``(n_genes,)`` is reshaped to ``(1, n_genes)``.
    require_nonnegative
        If True, reject negative entries (required for CPM normalization).
    """
    is_anndata = isinstance(data, ad.AnnData)
    obs_df: pd.DataFrame | None = None

    if is_anndata:
        if layer is not None:
            if layer not in data.layers:
                raise ValueError(f"Layer '{layer}' not found in {name}.layers.")
            raw_X = data.layers[layer]
        else:
            raw_X = data.X
        if raw_X is None:
            raise ValueError(f"{name} expression matrix is None.")
        obs_df = data.obs
        inferred_obs = [str(x) for x in data.obs_names]
        inferred_var = [str(x) for x in data.var_names]
        if sparse.issparse(raw_X):
            matrix: np.ndarray | sparse.csr_matrix = raw_X.tocsr().astype(
                np.float64, copy=True
            )
        else:
            matrix = np.asarray(raw_X, dtype=np.float64)
    elif isinstance(data, pd.DataFrame):
        if layer is not None:
            raise ValueError(f"`layer` is only supported for AnnData inputs ({name}).")
        inferred_obs = [str(x) for x in data.index]
        inferred_var = [str(x) for x in data.columns]
        if any(isinstance(dt, pd.SparseDtype) for dt in data.dtypes):
            matrix = data.sparse.to_coo().tocsr().astype(np.float64)
        else:
            matrix = np.asarray(data.to_numpy(dtype=np.float64), dtype=np.float64)
    elif sparse.issparse(data):
        if layer is not None:
            raise ValueError(f"`layer` is only supported for AnnData inputs ({name}).")
        if data.ndim != 2:
            raise ValueError(f"{name} sparse matrix must be 2D; got {data.ndim}D.")
        matrix = data.tocsr().astype(np.float64, copy=True)
        inferred_obs = None
        inferred_var = None
    else:
        if layer is not None:
            raise ValueError(f"`layer` is only supported for AnnData inputs ({name}).")
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1 and allow_1d:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"{name} array must be 2D; got shape {arr.shape}.")
        matrix = arr
        inferred_obs = None
        inferred_var = None

    n_obs, n_var = matrix.shape
    if n_obs == 0 or n_var == 0:
        raise ValueError(f"{name} matrix must be non-empty; got shape {matrix.shape}.")

    _validate_numeric_values(
        matrix, name=name, require_nonnegative=require_nonnegative
    )

    if sample_names is not None:
        obs_names = _coerce_name_list(sample_names, n_obs, obs_prefix, f"{name}_sample_names")
    elif inferred_obs is not None:
        obs_names = inferred_obs
    else:
        obs_names = [f"{obs_prefix}_{i}" for i in range(n_obs)]

    if gene_names is not None:
        var_names = _coerce_name_list(gene_names, n_var, "gene", f"{name}_gene_names")
    elif inferred_var is not None:
        var_names = inferred_var
    else:
        var_names = [f"gene_{j}" for j in range(n_var)]

    if pd.Index(var_names).has_duplicates:
        raise ValueError(f"{name} contains duplicate gene names.")

    return ParsedExpression(
        matrix=matrix,
        obs_names=obs_names,
        var_names=var_names,
        is_anndata=is_anndata,
        obs_df=obs_df,
    )


def cpm_scale_factors(matrix: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    """Compute per-observation CPM multipliers (1e6 / row_sum, 0 when row_sum <= 0)."""
    if sparse.issparse(matrix):
        row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    else:
        row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    scale = np.zeros_like(row_sums, dtype=np.float64)
    pos = row_sums > 0.0
    scale[pos] = 1e6 / row_sums[pos]
    return scale


def cpm_normalize(matrix: np.ndarray | sparse.csr_matrix) -> np.ndarray | sparse.csr_matrix:
    """Row-wise CPM normalization preserving sparse structure when input is sparse."""
    scale = cpm_scale_factors(matrix)
    if sparse.issparse(matrix):
        return (sparse.diags(scale) @ matrix).tocsr()
    return matrix * scale[:, None]


def subset_columns_dense(
    matrix: np.ndarray | sparse.csr_matrix, col_indices: np.ndarray
) -> np.ndarray:
    """Extract selected columns as a dense float64 ndarray."""
    sub = matrix[:, col_indices]
    if sparse.issparse(sub):
        return sub.toarray().astype(np.float64, copy=False)
    return np.asarray(sub, dtype=np.float64).copy()


def resolve_sc_annotations(
    parsed_sc: ParsedExpression,
    cell_types: str | Sequence[str] | np.ndarray,
    subject_names: str | Sequence[str] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve cell-type and subject labels for single-cell observations.

    Returns
    -------
    valid_mask : np.ndarray of bool
        Mask of cells with valid (non-null) cell type and subject labels.
    ct_labels : np.ndarray of str
        Cell-type labels for valid cells.
    subj_labels : np.ndarray of str
        Subject/donor labels for valid cells.
    """
    n_cells = parsed_sc.matrix.shape[0]

    def _extract_series(
        spec: str | Sequence[str] | np.ndarray | None,
        arg_name: str,
        default_single: str | None = None,
    ) -> pd.Series:
        if spec is None:
            if default_single is None:
                raise ValueError(f"`{arg_name}` cannot be None.")
            return pd.Series([default_single] * n_cells, dtype="object")
        if isinstance(spec, str):
            if not parsed_sc.is_anndata or parsed_sc.obs_df is None:
                raise ValueError(
                    f"`{arg_name}='{spec}'` is a column name, which requires `sc` to be an AnnData object; "
                    f"pass an explicit sequence/array of labels for DataFrame/ndarray inputs."
                )
            if spec not in parsed_sc.obs_df.columns:
                raise ValueError(f"Column '{spec}' not found in sc.obs.")
            return parsed_sc.obs_df[spec].reset_index(drop=True)
        arr = list(spec)
        if len(arr) != n_cells:
            raise ValueError(
                f"Length of `{arg_name}` ({len(arr)}) does not match number of cells ({n_cells})."
            )
        return pd.Series(arr)

    ct_series = _extract_series(cell_types, "cell_types")
    subj_series = _extract_series(
        subject_names, "subject_names", default_single="subject_0"
    )

    invalid_mask = ct_series.isna().to_numpy() | subj_series.isna().to_numpy()
    if np.any(invalid_mask):
        n_dropped = int(np.sum(invalid_mask))
        warnings.warn(
            f"Dropping {n_dropped} single cells with missing (NaN/None) cell_type or subject_names.",
            UserWarning,
            stacklevel=3,
        )

    valid_mask = ~invalid_mask
    if not np.any(valid_mask):
        raise ValueError("No valid cells remain after removing missing annotations.")

    ct_valid = ct_series[valid_mask].astype(str).to_numpy()
    subj_valid = subj_series[valid_mask].astype(str).to_numpy()
    return valid_mask, ct_valid, subj_valid
