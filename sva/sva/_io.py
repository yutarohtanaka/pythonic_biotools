"""
Input coercion, matrix orientation handling, and linear-algebra design helpers for `sva`.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` parameter strictly (`'default'` or `'R_default'`)."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unsupported settings={settings!r}; expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def is_anndata(obj: Any) -> bool:
    """Return True if `obj` behaves like an `anndata.AnnData` container."""
    return (
        hasattr(obj, "X")
        and hasattr(obj, "obs")
        and hasattr(obj, "var")
        and hasattr(obj, "obs_names")
        and hasattr(obj, "var_names")
    )


def coerce_expression_matrix(
    dat: Any,
    *,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], Any | None]:
    """Coerce `dat` to an internal `(n_features, n_samples)` float64 matrix.

    Orientation rules:
    - `AnnData`: always `obs x var` (`n_samples x n_features`), transposed internally
      to `(n_features, n_samples)`.
    - `DataFrame`, `scipy.sparse`, `ndarray`: `(n_features, n_samples)` when
      `features_as_rows=True` (Bioconductor default), or `(n_samples, n_features)`
      when `features_as_rows=False`.
    """
    adata_ref = None
    if is_anndata(dat):
        adata_ref = dat
        if layer is not None:
            if not hasattr(dat, "layers") or layer not in dat.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData object.")
            raw = dat.layers[layer]
        else:
            raw = dat.X

        if sparse.issparse(raw):
            arr = np.asarray(raw.toarray(), dtype=np.float64)
        else:
            arr = np.asarray(raw, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix in AnnData, got shape {arr.shape}.")

        Y = np.array(arr.T, dtype=np.float64, copy=True)
        feature_names = [str(x) for x in dat.var_names]
        sample_names = [str(x) for x in dat.obs_names]

    elif isinstance(dat, pd.DataFrame):
        if any(isinstance(dt, pd.SparseDtype) for dt in dat.dtypes):
            dense_df = dat.sparse.to_dense()
            arr = dense_df.to_numpy(dtype=np.float64)
        else:
            arr = dat.to_numpy(dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D DataFrame, got shape {arr.shape}.")

        if features_as_rows:
            Y = np.array(arr, dtype=np.float64, copy=True)
            feature_names = [str(x) for x in dat.index]
            sample_names = [str(x) for x in dat.columns]
        else:
            Y = np.array(arr.T, dtype=np.float64, copy=True)
            feature_names = [str(x) for x in dat.columns]
            sample_names = [str(x) for x in dat.index]

    elif sparse.issparse(dat):
        arr = np.asarray(dat.toarray(), dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D sparse matrix, got shape {arr.shape}.")
        Y = np.array(arr if features_as_rows else arr.T, dtype=np.float64, copy=True)
        feature_names = [f"feature_{i}" for i in range(Y.shape[0])]
        sample_names = [f"sample_{j}" for j in range(Y.shape[1])]

    else:
        arr = np.asarray(dat, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D expression matrix, got array with ndim={arr.ndim} and shape={arr.shape}."
            )
        Y = np.array(arr if features_as_rows else arr.T, dtype=np.float64, copy=True)
        feature_names = [f"feature_{i}" for i in range(Y.shape[0])]
        sample_names = [f"sample_{j}" for j in range(Y.shape[1])]

    n_features, n_samples = Y.shape
    if n_features < 1 or n_samples < 2:
        raise ValueError(
            f"Expression matrix must have at least 1 feature and 2 samples, got {Y.shape}."
        )
    if len(feature_names) != n_features or len(sample_names) != n_samples:
        raise ValueError("Feature/sample names length mismatch with matrix dimensions.")

    return Y, feature_names, sample_names, adata_ref


def _design_from_obs(obs: pd.DataFrame, cols: Sequence[str], n_samples: int) -> np.ndarray:
    """Build an intercept + covariates float64 design matrix from `AnnData.obs` columns."""
    parts: list[np.ndarray] = [np.ones((n_samples, 1), dtype=np.float64)]
    for col in cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in AnnData.obs.")
        s = obs[col]
        if s.isna().any():
            raise ValueError(f"AnnData.obs column '{col}' contains missing (NaN) values.")
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            col_arr = s.to_numpy(dtype=np.float64).reshape(n_samples, 1)
            if not np.all(np.isfinite(col_arr)):
                raise ValueError(f"AnnData.obs column '{col}' contains non-finite values.")
            parts.append(col_arr)
        else:
            dummies = pd.get_dummies(s, prefix=str(col), drop_first=True, dtype=np.float64)
            if dummies.shape[1] > 0:
                parts.append(dummies.to_numpy(dtype=np.float64))
    return np.hstack(parts)


def coerce_design_matrix(
    design: Any,
    n_samples: int,
    *,
    adata_ref: Any | None = None,
    name: str = "mod",
) -> np.ndarray:
    """Coerce `mod` or `mod0` into a finite 2D `(n_samples, p)` float64 design matrix."""
    if isinstance(design, str):
        if adata_ref is None:
            raise ValueError(
                f"String column specification for '{name}' is only supported when `dat` is an AnnData object."
            )
        X = _design_from_obs(adata_ref.obs, [design], n_samples)
    elif (
        isinstance(design, (list, tuple))
        and (len(design) == 0 or all(isinstance(item, str) for item in design))
    ):
        if adata_ref is None:
            raise ValueError(
                f"Sequence of column names for '{name}' is only supported when `dat` is an AnnData object."
            )
        X = _design_from_obs(adata_ref.obs, list(design), n_samples)
    elif isinstance(design, pd.DataFrame):
        if len(design) != n_samples:
            raise ValueError(
                f"Design matrix '{name}' has {len(design)} rows, expected {n_samples} samples."
            )
        parts: list[np.ndarray] = []
        for col in design.columns:
            s = design[col]
            if s.isna().any():
                raise ValueError(f"Design matrix '{name}' column '{col}' contains NaN values.")
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                parts.append(s.to_numpy(dtype=np.float64).reshape(n_samples, 1))
            else:
                dummies = pd.get_dummies(s, prefix=str(col), drop_first=True, dtype=np.float64)
                if dummies.shape[1] > 0:
                    parts.append(dummies.to_numpy(dtype=np.float64))
        if not parts:
            raise ValueError(f"Design matrix '{name}' has no usable columns.")
        X = np.hstack(parts)
    else:
        X = np.asarray(design, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            f"Design matrix '{name}' must be 2D of shape (n_samples, p), got shape {X.shape}."
        )
    if X.shape[0] != n_samples:
        raise ValueError(
            f"Design matrix '{name}' has {X.shape[0]} rows, expected {n_samples} samples."
        )
    if X.shape[1] < 1:
        raise ValueError(f"Design matrix '{name}' must have at least 1 column.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Design matrix '{name}' contains NaN or Inf values.")

    return np.array(X, dtype=np.float64, copy=True)


def orthonormal_basis(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute an orthonormal column-space basis `Q` and numerical rank via SVD."""
    if X.size == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64), 0
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.empty((X.shape[0], 0), dtype=np.float64), 0
    tol = max(X.shape) * np.finfo(np.float64).eps * float(s[0])
    rank = int(np.sum(s > tol))
    return U[:, :rank], rank


def is_nested_subspace(X0: np.ndarray, Q1: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True iff `col(X0)` lies within `col(Q1)` up to relative Frobenius tolerance."""
    if X0.shape[1] == 0:
        return True
    if Q1.shape[1] == 0:
        return float(np.linalg.norm(X0)) <= tol
    proj = Q1 @ (Q1.T @ X0)
    resid_norm = float(np.linalg.norm(X0 - proj))
    ref_norm = max(float(np.linalg.norm(X0)), 1.0)
    return resid_norm <= tol * ref_norm


def validate_nested_models(
    X1: np.ndarray,
    X0: np.ndarray | None,
    n_samples: int,
    *,
    require_strict_nesting: bool = True,
    min_resid_df: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Validate `mod` and `mod0`, returning `(X1, X0, Q1, r1, Q0, r0)`."""
    if X0 is None:
        X0_arr = np.ones((n_samples, 1), dtype=np.float64)
    else:
        X0_arr = X0

    Q1, r1 = orthonormal_basis(X1)
    Q0, r0 = orthonormal_basis(X0_arr)

    if r1 < 1:
        raise ValueError("Design matrix 'mod' has zero numerical rank.")
    if n_samples - r1 < min_resid_df:
        raise ValueError(
            f"Insufficient residual degrees of freedom: n - rank(mod) = {n_samples - r1} < {min_resid_df}."
        )
    if not is_nested_subspace(X0_arr, Q1):
        raise ValueError("mod0 must be nested in the column space of mod.")
    if require_strict_nesting and (r1 - r0) < 1:
        raise ValueError(
            f"Full model 'mod' (rank={r1}) must have strictly greater rank than null model 'mod0' (rank={r0})."
        )

    return X1, X0_arr, Q1, r1, Q0, r0
