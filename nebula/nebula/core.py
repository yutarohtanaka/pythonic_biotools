"""
Core public API for `nebula`: `group_cell`, `nebula`, `GroupedCellData`,
`GroupedData`, and `NebulaResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln
from scipy.stats import norm

from .hl import compute_schur_covariance, fit_hl_gene, fit_hl_sigma2_1d
from .ln import fit_ln_gene, initial_beta_guess

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unsupported settings={settings!r}; expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class GroupedCellData:
    """
    Validated, subject-grouped single-cell inputs returned by :func:`group_cell`.
    """

    count: Any
    id: np.ndarray
    subject_codes: np.ndarray
    subject_names: list[str]
    pred: np.ndarray | pd.DataFrame | None
    offset: np.ndarray | None
    order: np.ndarray
    was_already_grouped: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Return tidy cell-level grouping metadata as a pandas DataFrame."""
        n_cells = len(self.id)
        data: dict[str, Any] = {
            "cell_index": np.arange(n_cells, dtype=np.int64),
            "original_index": self.order,
            "subject_id": self.id,
            "subject_code": self.subject_codes,
        }
        if self.offset is not None:
            data["offset"] = self.offset
        if self.pred is not None:
            if isinstance(self.pred, pd.DataFrame):
                for col in self.pred.columns:
                    data[str(col)] = self.pred[col].to_numpy()
            else:
                pred_arr = np.asarray(self.pred)
                if pred_arr.ndim == 1:
                    data["pred_0"] = pred_arr
                else:
                    for k in range(pred_arr.shape[1]):
                        data[f"pred_{k}"] = pred_arr[:, k]
        return pd.DataFrame(data)


GroupedData = GroupedCellData


@dataclass(frozen=True)
class NebulaResult:
    """
    Immutable result container returned by :func:`nebula`.
    """

    gene_names: list[str]
    predictor_names: list[str]
    log_fc: np.ndarray
    se: np.ndarray
    z_stat: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    overdispersion_subject: np.ndarray
    overdispersion_cell: np.ndarray
    convergence: np.ndarray
    converged: np.ndarray
    algorithm: list[str]
    loglik: np.ndarray
    n_iter: np.ndarray
    kept_mask: np.ndarray
    cpc: np.ndarray
    nonzero_cells: np.ndarray
    model: str
    method_requested: str
    n_cells: int
    n_subjects: int
    mean_cells_per_subject: float
    subject_names: list[str]
    covariance_matrices: np.ndarray | None = None
    random_effects: np.ndarray | None = None
    laplace_orders: np.ndarray | None = None

    @property
    def sigma2(self) -> np.ndarray:
        """Alias for ``overdispersion_subject``."""
        return self.overdispersion_subject

    @property
    def phi(self) -> np.ndarray:
        """Alias for ``overdispersion_cell``."""
        return self.overdispersion_cell

    @property
    def method_used(self) -> list[str]:
        """Alias for ``algorithm``."""
        return self.algorithm

    def to_dataframe(
        self,
        include_filtered: bool = False,
        sort_by: str | None = None,
    ) -> pd.DataFrame:
        """
        Return tidy per-gene inference and overdispersion results as a DataFrame.
        """
        data: dict[str, Any] = {"gene": np.asarray(self.gene_names, dtype=object)}
        for k, name in enumerate(self.predictor_names):
            data[f"logFC_{name}"] = self.log_fc[:, k]
            data[f"se_{name}"] = self.se[:, k]
            data[f"z_{name}"] = self.z_stat[:, k]
            data[f"p_{name}"] = self.p_value[:, k]
            data[f"q_{name}"] = self.q_value[:, k]

        data["overdispersion_subject"] = self.overdispersion_subject
        data["overdispersion_cell"] = self.overdispersion_cell
        data["algorithm"] = np.asarray(self.algorithm, dtype=object)
        data["convergence"] = self.convergence
        data["converged"] = self.converged
        data["loglik"] = self.loglik
        data["cpc"] = self.cpc
        data["nonzero_cells"] = self.nonzero_cells

        df = pd.DataFrame(data)
        if not include_filtered:
            df = df.loc[self.kept_mask].reset_index(drop=True)

        if sort_by is not None:
            if sort_by not in df.columns:
                raise ValueError(
                    f"Column {sort_by!r} not found in NebulaResult DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
            df = df.sort_values(
                sort_by, kind="mergesort", na_position="last"
            ).reset_index(drop=True)
        return df

    def overdispersion_dataframe(self) -> pd.DataFrame:
        """Return subject and cell overdispersion estimates per gene."""
        return pd.DataFrame(
            {
                "gene": self.gene_names,
                "Subject": self.overdispersion_subject,
                "Cell": self.overdispersion_cell,
                "algorithm": self.algorithm,
                "convergence": self.convergence,
                "converged": self.converged,
            }
        )

    def significant(
        self,
        predictor: str | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Return kept genes with ``q_{predictor} <= alpha``, sorted by p-value.
        """
        if predictor is None:
            non_intercept = [
                p for p in self.predictor_names if p not in ("(Intercept)", "Intercept")
            ]
            predictor = non_intercept[0] if non_intercept else self.predictor_names[0]
        if predictor not in self.predictor_names:
            raise ValueError(
                f"Predictor {predictor!r} not found. Available: {self.predictor_names}"
            )
        df = self.to_dataframe(include_filtered=False)
        q_col = f"q_{predictor}"
        p_col = f"p_{predictor}"
        sig = df.loc[df[q_col].notna() & (df[q_col] <= alpha)].copy()
        return sig.sort_values(
            p_col, kind="mergesort", na_position="last"
        ).reset_index(drop=True)

    def summary(self) -> str:
        """Return a human-readable multi-line diagnostic summary."""
        n_total = len(self.gene_names)
        n_kept = int(np.sum(self.kept_mask))
        n_filt = n_total - n_kept
        n_conv = int(np.sum(self.converged & self.kept_mask))

        algo_counts: dict[str, int] = {}
        for g in range(n_total):
            if self.kept_mask[g]:
                algo_counts[self.algorithm[g]] = algo_counts.get(self.algorithm[g], 0) + 1

        kept_s2 = self.overdispersion_subject[self.kept_mask]
        kept_phi = self.overdispersion_cell[self.kept_mask]
        med_s2 = float(np.nanmedian(kept_s2)) if np.any(np.isfinite(kept_s2)) else float("nan")
        med_phi = float(np.nanmedian(kept_phi)) if np.any(np.isfinite(kept_phi)) else float("nan")

        algo_str = ", ".join(f"{k}={v}" for k, v in sorted(algo_counts.items())) or "none"
        lines = [
            f"NEBULA Result (model={self.model}, method_requested={self.method_requested})",
            f"  Cells: {self.n_cells}, Subjects: {self.n_subjects}, "
            f"Mean cells/subject (kappa_cps): {self.mean_cells_per_subject:.2f}",
            f"  Genes: {n_total} total ({n_kept} kept, {n_filt} filtered)",
            f"  Algorithms: {algo_str}; Converged: {n_conv}/{n_kept}",
            f"  Predictors ({len(self.predictor_names)}): {', '.join(self.predictor_names)}",
            f"  Median overdispersion: subject sigma^2={med_s2:.4g}, cell phi={med_phi:.4g}",
        ]
        return "\n".join(lines)


def _bh_fdr_nan_aware(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values while ignoring NaN entries."""
    q = np.full(p_values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p_values)
    m = int(np.sum(valid))
    if m == 0:
        return q
    pv = np.asarray(p_values[valid], dtype=np.float64)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * float(m) / np.arange(1, m + 1, dtype=np.float64)
    q_sub = np.empty(m, dtype=np.float64)
    q_sub[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    q[valid] = q_sub
    return q


def _validate_id_array(id_vals: Sequence[Any] | np.ndarray | pd.Series, n_cells: int) -> np.ndarray:
    if isinstance(id_vals, pd.Series):
        arr = id_vals.to_numpy()
    else:
        arr = np.asarray(id_vals)
    if arr.ndim != 1:
        raise ValueError(f"`id` must be 1-dimensional, got shape {arr.shape}.")
    if arr.shape[0] != n_cells:
        raise ValueError(
            f"Length of `id` ({arr.shape[0]}) does not match number of cells ({n_cells})."
        )
    if pd.isna(arr).any():
        raise ValueError("Subject `id` cannot contain NaN or missing values.")
    return arr


def _normalize_sparse_matrix(mat: Any, *, prefer_csr: bool) -> sparse.csr_matrix | sparse.csc_matrix:
    """Convert any scipy sparse matrix/array to canonical CSR or CSC without densifying."""
    if prefer_csr:
        sp = sparse.csr_matrix(mat) if not isinstance(mat, sparse.csr_matrix) else mat
    else:
        sp = sparse.csc_matrix(mat) if not isinstance(mat, sparse.csc_matrix) else mat
    return sp


def group_cell(
    count: np.ndarray | pd.DataFrame | sparse.spmatrix | sparse.sparray,
    id: Sequence[Any] | np.ndarray | pd.Series,
    pred: np.ndarray | pd.DataFrame | None = None,
    offset: Sequence[float] | np.ndarray | pd.Series | None = None,
    *,
    genes_as_rows: bool = True,
) -> GroupedCellData:
    """
    Validate and stably reorder cells into contiguous subject blocks.

    When cells are already grouped into contiguous subject blocks, no cell
    permutation is performed (``was_already_grouped=True``).
    """
    if sparse.issparse(count):
        if count.ndim != 2:
            raise ValueError(f"`count` must be 2D, got {count.ndim}D.")
        n_cells = count.shape[1] if genes_as_rows else count.shape[0]
    elif isinstance(count, pd.DataFrame):
        if count.ndim != 2:
            raise ValueError(f"`count` must be 2D, got {count.ndim}D.")
        n_cells = count.shape[1] if genes_as_rows else count.shape[0]
    else:
        count_arr = np.asarray(count)
        if count_arr.ndim != 2:
            raise ValueError(f"`count` must be 2D, got shape {count_arr.shape}.")
        count = count_arr
        n_cells = count_arr.shape[1] if genes_as_rows else count_arr.shape[0]

    if n_cells == 0:
        raise ValueError("`count` contains zero cells.")

    id_arr = _validate_id_array(id, n_cells)
    codes_raw, uniques = pd.factorize(id_arr, sort=False)
    codes = np.asarray(codes_raw, dtype=np.int64)
    subject_names = [str(u) for u in uniques]

    if pred is not None:
        if isinstance(pred, pd.DataFrame):
            if pred.shape[0] != n_cells:
                raise ValueError(
                    f"`pred` row count ({pred.shape[0]}) does not match cells ({n_cells})."
                )
        else:
            pred_arr = np.asarray(pred)
            if pred_arr.shape[0] != n_cells:
                raise ValueError(
                    f"`pred` row count ({pred_arr.shape[0]}) does not match cells ({n_cells})."
                )
            pred = pred_arr

    offset_arr: np.ndarray | None = None
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64)
        if offset_arr.ndim != 1 or offset_arr.shape[0] != n_cells:
            raise ValueError(
                f"`offset` must be a 1D vector of length {n_cells}, got shape {offset_arr.shape}."
            )
        if not np.all(np.isfinite(offset_arr)) or np.any(offset_arr <= 0.0):
            raise ValueError("`offset` must contain strictly positive finite values.")

    was_already_grouped = bool(np.all(codes[1:] >= codes[:-1]))
    if was_already_grouped:
        order = np.arange(n_cells, dtype=np.int64)
        count_out = (
            _normalize_sparse_matrix(count, prefer_csr=genes_as_rows)
            if sparse.issparse(count)
            else count
        )
        return GroupedCellData(
            count=count_out,
            id=id_arr,
            subject_codes=codes,
            subject_names=subject_names,
            pred=pred,
            offset=offset_arr,
            order=order,
            was_already_grouped=True,
        )

    order = np.argsort(codes, kind="mergesort").astype(np.int64)
    codes_sorted = codes[order]
    id_sorted = id_arr[order]

    if sparse.issparse(count):
        sp = _normalize_sparse_matrix(count, prefer_csr=genes_as_rows)
        count_sorted = sp[:, order] if genes_as_rows else sp[order, :]
        count_sorted = _normalize_sparse_matrix(count_sorted, prefer_csr=genes_as_rows)
    elif isinstance(count, pd.DataFrame):
        count_sorted = count.iloc[:, order] if genes_as_rows else count.iloc[order, :]
    else:
        count_sorted = count[:, order] if genes_as_rows else count[order, :]

    pred_sorted: np.ndarray | pd.DataFrame | None
    if pred is None:
        pred_sorted = None
    elif isinstance(pred, pd.DataFrame):
        pred_sorted = pred.iloc[order].reset_index(drop=True)
    else:
        pred_sorted = pred[order]

    offset_sorted = offset_arr[order] if offset_arr is not None else None

    return GroupedCellData(
        count=count_sorted,
        id=id_sorted,
        subject_codes=codes_sorted,
        subject_names=subject_names,
        pred=pred_sorted,
        offset=offset_sorted,
        order=order,
        was_already_grouped=False,
    )


def _encode_series(series: pd.Series, col_name: str) -> tuple[np.ndarray, list[str]]:
    """Encode a single Series (numeric, bool, or categorical/string) into float64 columns."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=np.float64)[:, None], [col_name]
    if pd.api.types.is_numeric_dtype(series):
        vals = series.to_numpy(dtype=np.float64)
        return vals[:, None], [col_name]
    if series.isna().any():
        raise ValueError(f"Predictor column {col_name!r} contains missing values.")
    dummies = pd.get_dummies(series, prefix=col_name, drop_first=True, dtype=np.float64)
    if dummies.shape[1] == 0:
        raise ValueError(
            f"Categorical predictor {col_name!r} has only one unique level and is collinear with intercept."
        )
    return dummies.to_numpy(dtype=np.float64), [str(c) for c in dummies.columns]


def _dataframe_to_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    for col in df.columns:
        mat, col_names = _encode_series(df[col], str(col))
        cols.append(mat)
        names.extend(col_names)
    if not cols:
        return np.empty((df.shape[0], 0), dtype=np.float64), []
    return np.hstack(cols), names


def _parse_formula(formula: str, obs_df: pd.DataFrame) -> tuple[np.ndarray, list[str], bool]:
    """
    Pure-Python/pandas parser for right-hand formulas such as
    ``"~ x + group"``, ``"~ x * group"``, ``"~ x + group - 1"``.
    Returns ``(X_no_intercept, col_names, want_intercept)``.
    """
    rhs = formula.strip()
    if rhs.startswith("~"):
        rhs = rhs[1:].strip()
    if not rhs:
        raise ValueError(f"Empty formula: {formula!r}")

    want_intercept = True
    # Detect explicit - 1 or + 0 / leading 0 +
    rhs_clean = re.sub(r"\s+", " ", rhs)
    tokens = [t.strip() for t in re.split(r"(?=[+-])", rhs_clean) if t.strip()]

    raw_terms: list[str] = []
    for tok in tokens:
        if tok in ("- 1", "-1", "+ 0", "+0", "0"):
            want_intercept = False
        elif tok in ("+ 1", "+1", "1"):
            want_intercept = True
        else:
            if tok.startswith("-"):
                raise ValueError(f"Unsupported negative formula term {tok!r} in {formula!r}.")
            term = tok.lstrip("+").strip()
            if term:
                raw_terms.append(term)

    expanded_terms: list[str] = []
    for term in raw_terms:
        if "*" in term:
            parts = [p.strip() for p in term.split("*") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Only 2-way `*` interactions are supported, got {term!r}.")
            a, b = parts
            for sub in (a, b, f"{a}:{b}"):
                if sub not in expanded_terms:
                    expanded_terms.append(sub)
        else:
            if term not in expanded_terms:
                expanded_terms.append(term)

    cols: list[np.ndarray] = []
    names: list[str] = []
    encoded_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def get_encoded(var: str) -> tuple[np.ndarray, list[str]]:
        if var not in obs_df.columns:
            raise ValueError(f"Formula variable {var!r} not found in `.obs`.")
        if var not in encoded_cache:
            encoded_cache[var] = _encode_series(obs_df[var], var)
        return encoded_cache[var]

    for term in expanded_terms:
        if ":" in term:
            parts = [p.strip() for p in term.split(":")]
            if len(parts) != 2:
                raise ValueError(f"Only 2-way `:` interactions are supported, got {term!r}.")
            m1, n1 = get_encoded(parts[0])
            m2, n2 = get_encoded(parts[1])
            for i1, name1 in enumerate(n1):
                for i2, name2 in enumerate(n2):
                    cols.append((m1[:, i1] * m2[:, i2])[:, None])
                    names.append(f"{name1}:{name2}")
        else:
            mat, col_names = get_encoded(term)
            cols.append(mat)
            names.extend(col_names)

    if not cols:
        return np.empty((obs_df.shape[0], 0), dtype=np.float64), [], want_intercept
    return np.hstack(cols), names, want_intercept


def _build_design_matrix(
    pred: Any,
    n_cells: int,
    obs_df: pd.DataFrame | None,
    add_intercept: bool,
) -> tuple[np.ndarray, list[str], int | None]:
    """Construct validated full-rank float64 design matrix and intercept index."""
    want_intercept = add_intercept

    if pred is None:
        X = np.ones((n_cells, 1), dtype=np.float64)
        return X, ["(Intercept)"], 0

    if isinstance(pred, str):
        if obs_df is None:
            raise ValueError("String `pred` requires an AnnData object with `.obs`.")
        if pred.strip().startswith("~"):
            X_raw, col_names, formula_intercept = _parse_formula(pred, obs_df)
            want_intercept = add_intercept and formula_intercept
        else:
            if pred not in obs_df.columns:
                raise ValueError(f"Predictor column {pred!r} not found in `adata.obs`.")
            X_raw, col_names = _dataframe_to_design(obs_df[[pred]])
    elif isinstance(pred, (list, tuple, pd.Index)) and obs_df is not None and all(
        isinstance(x, str) for x in pred
    ):
        missing = [x for x in pred if x not in obs_df.columns]
        if missing:
            raise ValueError(f"Predictor columns not found in `adata.obs`: {missing}")
        X_raw, col_names = _dataframe_to_design(obs_df[list(pred)])
    elif isinstance(pred, pd.DataFrame):
        if pred.shape[0] != n_cells:
            raise ValueError(
                f"`pred` row count ({pred.shape[0]}) does not match cells ({n_cells})."
            )
        X_raw, col_names = _dataframe_to_design(pred)
    else:
        arr = np.asarray(pred, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"`pred` length ({arr.shape[0]}) does not match cells ({n_cells})."
                )
            X_raw = arr[:, None]
            col_names = ["x1"]
        elif arr.ndim == 2:
            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"`pred` row count ({arr.shape[0]}) does not match cells ({n_cells})."
                )
            X_raw = arr
            col_names = [
                "(Intercept)"
                if (k == 0 and np.allclose(X_raw[:, 0], 1.0, rtol=1e-12, atol=0.0))
                else f"x{k + 1}"
                for k in range(X_raw.shape[1])
            ]
        else:
            raise ValueError(f"`pred` must be 1D or 2D, got shape {arr.shape}.")

    if X_raw.shape[1] == 0:
        if not want_intercept:
            raise ValueError("Design matrix has 0 columns.")
        return np.ones((n_cells, 1), dtype=np.float64), ["(Intercept)"], 0

    if not np.all(np.isfinite(X_raw)):
        raise ValueError("Design matrix `pred` contains NaN or Inf values.")

    const_cols: list[int] = []
    for k in range(X_raw.shape[1]):
        col = X_raw[:, k]
        col_mean = float(np.mean(col))
        if abs(col_mean) > 1e-300 and float(np.ptp(col)) <= 1e-12 * abs(col_mean):
            const_cols.append(k)
    if const_cols:
        X = np.asarray(X_raw, dtype=np.float64)
        intercept_col = const_cols[0]
    elif want_intercept:
        X = np.column_stack([np.ones(n_cells, dtype=np.float64), X_raw])
        col_names = ["(Intercept)", *col_names]
        intercept_col = 0
    else:
        X = np.asarray(X_raw, dtype=np.float64)
        intercept_col = None

    col_norms = np.linalg.norm(X, axis=0)
    if np.any(col_norms <= 1e-300):
        raise ValueError("Design matrix `pred` is rank-deficient / singular.")
    X_eq = X / col_norms[None, :]
    s = np.linalg.svd(X_eq, compute_uv=False)
    if s.size == 0 or s[-1] <= 1e-10 * float(s[0]):
        raise ValueError("Design matrix `pred` is rank-deficient / singular.")

    return X, col_names, intercept_col


def _validate_counts_values(vals: np.ndarray) -> None:
    if vals.size == 0:
        return
    if not np.all(np.isfinite(vals)):
        raise ValueError("Count matrix contains NaN or infinite values.")
    if np.any(vals < 0.0):
        raise ValueError("Count matrix contains negative values.")
    if not np.allclose(vals, np.round(vals), atol=1e-6):
        raise ValueError("Count matrix must contain non-negative integer counts.")


def _is_anndata(obj: Any) -> bool:
    return (
        hasattr(obj, "X")
        and hasattr(obj, "obs")
        and hasattr(obj, "var")
        and hasattr(obj, "obs_names")
        and hasattr(obj, "var_names")
    )


def nebula(
    data: Any,
    id: str | Sequence[Any] | np.ndarray | pd.Series,
    pred: np.ndarray | pd.DataFrame | Sequence[str] | str | None = None,
    offset: str | Sequence[float] | np.ndarray | pd.Series | None = None,
    *,
    layer: str | None = None,
    genes_as_rows: bool = True,
    add_intercept: bool = True,
    model: Literal["NBGMM", "PMM", "NBLMM"] = "NBGMM",
    method: Literal["auto", "LN", "HL"] = "auto",
    cpc: float = 0.005,
    mincp: int = 5,
    cutoff_cell: float = 20.0,
    kappa: float = 800.0,
    laplace_order: Literal[1, 2] | None = None,
    covariance: bool = False,
    output_re: bool = False,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_overdispersion: float = 1e-6,
    max_overdispersion: float = 1e4,
    settings: SettingsMode = "default",
) -> NebulaResult:
    """
    Fit multi-subject negative binomial / Poisson-Gamma generalized linear mixed
    models across genes using NEBULA-LN and/or NEBULA-HL.
    """
    eff_mode = _validate_settings(settings)
    if model not in ("NBGMM", "PMM", "NBLMM"):
        raise ValueError(f"Unsupported model {model!r}; expected 'NBGMM', 'PMM', or 'NBLMM'.")
    if method not in ("auto", "LN", "HL"):
        raise ValueError(f"Unsupported method {method!r}; expected 'auto', 'LN', or 'HL'.")
    if laplace_order is not None and laplace_order not in (1, 2):
        raise ValueError(f"Unsupported laplace_order {laplace_order!r}; expected 1 or 2.")
    if min_overdispersion <= 0 or min_overdispersion >= max_overdispersion:
        raise ValueError("Require 0 < min_overdispersion < max_overdispersion.")
    eff_method_req = ("LN" if method == "auto" else method) if eff_mode == "R_default" else method

    obs_df: pd.DataFrame | None = None
    if _is_anndata(data):
        obs_df = data.obs
        raw_mat = data.layers[layer] if layer is not None else data.X
        if raw_mat is None:
            raise ValueError("AnnData count matrix is None.")
        gene_names = [str(g) for g in data.var_names]
        eff_genes_as_rows = False  # AnnData is obs x var (cells x genes)
        n_cells, n_genes = raw_mat.shape

        if isinstance(id, str):
            if id not in obs_df.columns:
                raise ValueError(f"Subject column {id!r} not found in `adata.obs`.")
            id_input = obs_df[id]
        else:
            id_input = id

        if offset is None:
            offset_input = np.ones(n_cells, dtype=np.float64)
        elif isinstance(offset, str):
            if offset in obs_df.columns:
                offset_input = obs_df[offset].to_numpy(dtype=np.float64)
            elif offset == "library_size":
                if sparse.issparse(raw_mat):
                    lib = np.asarray(raw_mat.sum(axis=1), dtype=np.float64).ravel()
                else:
                    lib = np.sum(np.asarray(raw_mat, dtype=np.float64), axis=1)
                lib = np.maximum(lib, 1.0)
                offset_input = lib / float(np.mean(lib))
            else:
                raise ValueError(f"Offset column {offset!r} not found in `adata.obs`.")
        else:
            offset_input = offset
    else:
        eff_genes_as_rows = genes_as_rows
        if isinstance(data, pd.DataFrame):
            if any(isinstance(dt, pd.SparseDtype) for dt in data.dtypes):
                raw_mat = data.sparse.to_coo().tocsr()
            else:
                raw_mat = data.to_numpy(dtype=np.float64)
            gene_names = [str(g) for g in (data.index if eff_genes_as_rows else data.columns)]
        elif sparse.issparse(data):
            raw_mat = data
            n_genes = raw_mat.shape[0] if eff_genes_as_rows else raw_mat.shape[1]
            gene_names = [f"gene_{g}" for g in range(n_genes)]
        else:
            raw_mat = np.asarray(data, dtype=np.float64)
            if raw_mat.ndim != 2:
                raise ValueError(f"`data` must be 2D, got shape {raw_mat.shape}.")
            n_genes = raw_mat.shape[0] if eff_genes_as_rows else raw_mat.shape[1]
            gene_names = [f"gene_{g}" for g in range(n_genes)]

        n_genes = raw_mat.shape[0] if eff_genes_as_rows else raw_mat.shape[1]
        n_cells = raw_mat.shape[1] if eff_genes_as_rows else raw_mat.shape[0]
        id_input = id
        if offset is None:
            offset_input = np.ones(n_cells, dtype=np.float64)
        elif isinstance(offset, str):
            raise ValueError("String `offset` requires an AnnData input.")
        else:
            offset_input = offset

    if n_genes == 0 or n_cells == 0:
        raise ValueError(f"Input count matrix cannot be empty, got shape {raw_mat.shape}.")

    # Validate raw counts before sorting/fitting without densifying sparse matrices
    if sparse.issparse(raw_mat):
        _validate_counts_values(np.asarray(raw_mat.data, dtype=np.float64))
    else:
        _validate_counts_values(np.asarray(raw_mat, dtype=np.float64))

    X_raw, predictor_names, intercept_col = _build_design_matrix(
        pred, n_cells, obs_df, add_intercept
    )

    grouped = group_cell(
        raw_mat,
        id_input,
        pred=X_raw,
        offset=offset_input,
        genes_as_rows=eff_genes_as_rows,
    )

    X = np.asarray(grouped.pred, dtype=np.float64)
    log_offset = np.log(np.asarray(grouped.offset, dtype=np.float64))
    subj = grouped.subject_codes
    subject_names = grouped.subject_names
    m = len(subject_names)
    p = X.shape[1]
    cps = float(n_cells) / float(m)

    n_per_subj = np.bincount(subj, minlength=m)
    if m == 1:
        warnings.warn(
            "Only 1 subject present (m = 1); subject-level variance sigma^2 is unidentifiable.",
            UserWarning,
            stacklevel=2,
        )
    elif int(np.max(n_per_subj)) == 1:
        warnings.warn(
            "All subjects have only 1 cell (max n_i = 1); subject variance sigma^2 "
            "and cell overdispersion phi are confounded.",
            UserWarning,
            stacklevel=2,
        )

    is_sp = sparse.issparse(grouped.count)
    if is_sp:
        sp = _normalize_sparse_matrix(grouped.count, prefer_csr=eff_genes_as_rows)
        # Copy before sum_duplicates / eliminate_zeros only if needed
        if not sp.has_canonical_format or np.any(sp.data == 0):
            sp = sp.copy()
            sp.sum_duplicates()
            sp.eliminate_zeros()
        sp_data = np.asarray(sp.data, dtype=np.float64)
        sp_indices = np.asarray(sp.indices, dtype=np.int64)
        sp_indptr = np.asarray(sp.indptr, dtype=np.int64)
        sum_axis = 1 if eff_genes_as_rows else 0
        gene_totals = np.asarray(sp.sum(axis=sum_axis), dtype=np.float64).ravel()
        nonzero_cells = np.diff(sp_indptr).astype(np.int64)
    else:
        dense_mat = np.asarray(grouped.count, dtype=np.float64)
        sum_axis = 1 if eff_genes_as_rows else 0
        gene_totals = np.sum(dense_mat, axis=sum_axis)
        nonzero_cells = np.sum(dense_mat > 0.0, axis=sum_axis).astype(np.int64)

    cpc_vec = gene_totals / float(n_cells)
    kept_mask = (cpc_vec >= cpc) & (nonzero_cells >= mincp) & (gene_totals > 0.0)

    log_fc = np.full((n_genes, p), np.nan, dtype=np.float64)
    se_arr = np.full((n_genes, p), np.nan, dtype=np.float64)
    z_stat = np.full((n_genes, p), np.nan, dtype=np.float64)
    p_value = np.full((n_genes, p), np.nan, dtype=np.float64)
    overdisp_s = np.full(n_genes, np.nan, dtype=np.float64)
    overdisp_c = np.full(n_genes, np.nan, dtype=np.float64)
    convergence = np.zeros(n_genes, dtype=np.int64)
    algorithm: list[str] = ["filtered"] * n_genes
    loglik = np.full(n_genes, np.nan, dtype=np.float64)
    n_iter_arr = np.zeros(n_genes, dtype=np.int64)
    laplace_orders_arr = np.zeros(n_genes, dtype=np.int64)

    cov_mats = (
        np.full((n_genes, p, p), np.nan, dtype=np.float64) if covariance else None
    )
    re_mat = (
        np.full((n_genes, m), np.nan, dtype=np.float64) if output_re else None
    )

    for g in range(n_genes):
        if not kept_mask[g]:
            continue

        if is_sp:
            s, e = int(sp_indptr[g]), int(sp_indptr[g + 1])
            cell_idx = sp_indices[s:e]
            y_nz = sp_data[s:e]
        else:
            y_g = dense_mat[g, :] if eff_genes_as_rows else dense_mat[:, g]
            cell_idx = np.flatnonzero(y_g)
            y_nz = y_g[cell_idx]

        subj_nz = subj[cell_idx]
        Y_subj = np.bincount(subj_nz, weights=y_nz, minlength=m)
        Y_nz = Y_subj[Y_subj > 0.0]
        const_y_fact = float(np.sum(gammaln(y_nz + 1.0))) if y_nz.size > 0 else 0.0
        beta0 = initial_beta_guess(X, log_offset, y_nz, cell_idx, n_cells, intercept_col)

        if eff_mode == "R_default":
            gene_laplace_order = (
                (2 if (float(cpc_vec[g]) * cps < 3.0) else 1)
                if laplace_order is None
                else int(laplace_order)
            )
        else:
            gene_laplace_order = 2 if laplace_order is None else int(laplace_order)

        if model == "PMM":
            beta_hat, s2_hat, phi_hat, ll, conv, n_it = fit_ln_gene(
                X,
                log_offset,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y_fact,
                beta0,
                min_overdispersion,
                max_overdispersion,
                max_iter,
                tol,
                pmm_only=True,
            )
            algo_g = "PMM"
        elif model == "NBLMM":
            use_nblmm_ln = (
                (eff_method_req == "LN" and cps >= 30.0)
                if eff_mode == "R_default"
                else (method == "LN")
            )
            if use_nblmm_ln:
                beta_hat, s2_ln, phi_hat, ll, conv, n_it = fit_ln_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta0,
                    min_overdispersion,
                    max_overdispersion,
                    max_iter,
                    tol,
                    pmm_only=False,
                )
                s2_hat = float(np.clip(np.log1p(s2_ln), min_overdispersion, max_overdispersion))
                if intercept_col is not None:
                    beta_hat = beta_hat.copy()
                    beta_hat[intercept_col] -= 0.5 * s2_hat / float(X[0, intercept_col])
                algo_g = "LN"
            else:
                beta_ws, s2_ws, phi_ws, _, _, _ = fit_ln_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta0,
                    min_overdispersion,
                    max_overdispersion,
                    min(40, max_iter),
                    max(tol, 1e-4),
                    pmm_only=False,
                )
                s2_nblmm_ws = float(np.clip(np.log1p(s2_ws), min_overdispersion, max_overdispersion))
                if intercept_col is not None:
                    beta_ws = beta_ws.copy()
                    beta_ws[intercept_col] -= 0.5 * s2_nblmm_ws / float(X[0, intercept_col])
                beta_hat, s2_hat, phi_hat, ll, conv, n_it = fit_hl_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta_ws,
                    min_overdispersion,
                    max_overdispersion,
                    max_iter,
                    tol,
                    gaussian_prior=True,
                    laplace_order=1,
                    sigma2_init=s2_nblmm_ws,
                    phi_init=float(np.clip(phi_ws, min_overdispersion, max_overdispersion)),
                )
                algo_g = "NBLMM"
                laplace_orders_arr[g] = 1
        else:
            # NBGMM
            if eff_mode == "R_default":
                use_hl_direct = eff_method_req == "HL" or cps < 30.0
            else:
                use_hl_direct = method == "HL" or (method == "auto" and cps < cutoff_cell)

            if use_hl_direct:
                beta_ws, s2_ws, phi_ws, _, _, _ = fit_ln_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta0,
                    min_overdispersion,
                    max_overdispersion,
                    min(40, max_iter),
                    max(tol, 1e-4),
                    pmm_only=False,
                )
                beta_hat, s2_hat, phi_hat, ll, conv, n_it = fit_hl_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta_ws,
                    min_overdispersion,
                    max_overdispersion,
                    max_iter,
                    tol,
                    gaussian_prior=False,
                    laplace_order=gene_laplace_order,
                    sigma2_init=float(np.clip(s2_ws, min_overdispersion, max_overdispersion)),
                    phi_init=float(np.clip(phi_ws, min_overdispersion, max_overdispersion)),
                )
                algo_g = "HL"
                laplace_orders_arr[g] = gene_laplace_order
            else:
                beta_hat, s2_hat, phi_hat, ll, conv, n_it = fit_ln_gene(
                    X,
                    log_offset,
                    subj,
                    m,
                    y_nz,
                    cell_idx,
                    subj_nz,
                    Y_subj,
                    Y_nz,
                    const_y_fact,
                    beta0,
                    min_overdispersion,
                    max_overdispersion,
                    max_iter,
                    tol,
                    pmm_only=False,
                )
                algo_g = "LN"
                kappa_obs = cps / max(phi_hat, 1e-12)
                if eff_mode == "R_default":
                    if (cutoff_cell > 0.0 and kappa_obs < cutoff_cell) or conv == -40:
                        beta_hat, s2_hat, phi_hat, ll, conv, n_it2 = fit_hl_gene(
                            X,
                            log_offset,
                            subj,
                            m,
                            y_nz,
                            cell_idx,
                            subj_nz,
                            Y_subj,
                            Y_nz,
                            const_y_fact,
                            beta_hat,
                            min_overdispersion,
                            max_overdispersion,
                            max_iter,
                            tol,
                            gaussian_prior=False,
                            laplace_order=gene_laplace_order,
                            sigma2_init=float(np.clip(s2_hat, min_overdispersion, max_overdispersion)),
                            phi_init=float(np.clip(phi_hat, min_overdispersion, max_overdispersion)),
                        )
                        n_it += n_it2
                        algo_g = "HL"
                        laplace_orders_arr[g] = gene_laplace_order
                    elif (
                        cutoff_cell <= kappa_obs < kappa
                        and s2_hat < (8.0 / kappa_obs)
                    ):
                        s2_1d, ll_1d, conv_1d, n_it_1d = fit_hl_sigma2_1d(
                            X,
                            log_offset,
                            subj,
                            m,
                            y_nz,
                            cell_idx,
                            subj_nz,
                            Y_subj,
                            Y_nz,
                            const_y_fact,
                            beta_hat,
                            phi_hat,
                            min_overdispersion,
                            max_overdispersion,
                            max_iter,
                            tol,
                            gaussian_prior=False,
                            laplace_order=gene_laplace_order,
                            sigma2_init=float(np.clip(s2_hat, min_overdispersion, max_overdispersion)),
                            kappa_obs=kappa_obs,
                        )
                        s2_hat = s2_1d
                        ll = ll_1d
                        conv = conv_1d
                        n_it += n_it_1d
                        algo_g = "NBGMM (LN+HL)"
                        laplace_orders_arr[g] = gene_laplace_order
                else:
                    if method == "auto" and (
                        (cutoff_cell > 0.0 and kappa_obs < cutoff_cell)
                        or conv == -40
                    ):
                        beta_hat, s2_hat, phi_hat, ll, conv, n_it2 = fit_hl_gene(
                            X,
                            log_offset,
                            subj,
                            m,
                            y_nz,
                            cell_idx,
                            subj_nz,
                            Y_subj,
                            Y_nz,
                            const_y_fact,
                            beta_hat,
                            min_overdispersion,
                            max_overdispersion,
                            max_iter,
                            tol,
                            gaussian_prior=False,
                            laplace_order=gene_laplace_order,
                            sigma2_init=float(np.clip(s2_hat, min_overdispersion, max_overdispersion)),
                            phi_init=float(np.clip(phi_hat, min_overdispersion, max_overdispersion)),
                        )
                        n_it += n_it2
                        algo_g = "HL"
                        laplace_orders_arr[g] = gene_laplace_order

        cov_g, se_g, omega_g = compute_schur_covariance(
            beta_hat,
            s2_hat,
            phi_hat,
            X,
            log_offset,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            model=model,
        )

        z_g = beta_hat / se_g
        p_g = 2.0 * norm.sf(np.abs(z_g))

        log_fc[g] = beta_hat
        se_arr[g] = se_g
        z_stat[g] = z_g
        p_value[g] = p_g
        overdisp_s[g] = s2_hat
        overdisp_c[g] = phi_hat
        convergence[g] = conv
        algorithm[g] = algo_g
        loglik[g] = ll
        n_iter_arr[g] = n_it
        if cov_mats is not None:
            cov_mats[g] = cov_g
        if re_mat is not None:
            re_mat[g] = omega_g

    q_value = np.full((n_genes, p), np.nan, dtype=np.float64)
    for k in range(p):
        q_value[:, k] = _bh_fdr_nan_aware(p_value[:, k])

    converged = np.isin(convergence, [1, -10])

    return NebulaResult(
        gene_names=gene_names,
        predictor_names=predictor_names,
        log_fc=log_fc,
        se=se_arr,
        z_stat=z_stat,
        p_value=p_value,
        q_value=q_value,
        overdispersion_subject=overdisp_s,
        overdispersion_cell=overdisp_c,
        convergence=convergence,
        converged=converged,
        algorithm=algorithm,
        loglik=loglik,
        n_iter=n_iter_arr,
        kept_mask=kept_mask,
        cpc=cpc_vec,
        nonzero_cells=nonzero_cells,
        model=model,
        method_requested=eff_method_req,
        n_cells=n_cells,
        n_subjects=m,
        mean_cells_per_subject=cps,
        subject_names=subject_names,
        covariance_matrices=cov_mats,
        random_effects=re_mat,
        laplace_orders=laplace_orders_arr,
    )
