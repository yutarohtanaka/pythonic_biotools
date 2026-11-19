"""
Mixed-effects formula parsing, programmatic design specification, contrast builder,
and unified expression/weight input coercion (AnnData, DataFrame, ndarray, sparse).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class DesignSpec:
    """Immutable mixed-effects model design specification.

    Attributes
    ----------
    X : np.ndarray
        Fixed-effects design matrix of shape ``(N, p)`` (float64).
    fixed_names : list[str]
        Column names of ``X`` (length ``p``).
    fixed_terms : dict[str, list[int]]
        Mapping from logical non-intercept fixed-effect term name to column indices in ``X``.
    has_intercept : bool
        Whether the model includes an intercept term.
    intercept_cols : list[int]
        Column indices in ``X`` corresponding to the intercept.
    random_matrices : list[np.ndarray]
        List of ``R`` random-effect design matrices, each of shape ``(N, m_r)`` (float64).
    random_names : list[str]
        Names of the ``R`` random-effect terms.
    random_level_names : list[list[str]]
        Level names for each random-effect matrix (length ``m_r`` per term).
    sample_names : list[str]
        Sample identifiers of length ``N``.
    """

    X: np.ndarray
    fixed_names: list[str]
    fixed_terms: dict[str, list[int]]
    has_intercept: bool
    intercept_cols: list[int]
    random_matrices: list[np.ndarray]
    random_names: list[str]
    random_level_names: list[list[str]]
    sample_names: list[str]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_fixed(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_random_terms(self) -> int:
        return len(self.random_matrices)

    @property
    def total_random_levels(self) -> int:
        return int(sum(z.shape[1] for z in self.random_matrices))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the fixed-effects design matrix ``X`` as a labeled DataFrame."""
        return pd.DataFrame(self.X, index=self.sample_names, columns=self.fixed_names)

    def summary(self) -> str:
        """Return a concise text summary of the mixed-effects design."""
        re_desc = ", ".join(
            f"{name}({z.shape[1]})" for name, z in zip(self.random_names, self.random_matrices)
        )
        return (
            f"DesignSpec(n_samples={self.n_samples}, n_fixed={self.n_fixed}, "
            f"fixed_terms={list(self.fixed_terms.keys())}, "
            f"random_terms=[{re_desc}])"
        )

    @classmethod
    def from_matrices(
        cls,
        X: np.ndarray | pd.DataFrame,
        random_effects: (
            Sequence[np.ndarray | pd.Series | pd.DataFrame]
            | Mapping[str, np.ndarray | pd.Series | pd.DataFrame]
            | None
        ) = None,
        *,
        fixed_names: Sequence[str] | None = None,
        fixed_terms: Mapping[str, Sequence[int | str]] | None = None,
        random_names: Sequence[str] | None = None,
        sample_names: Sequence[str] | None = None,
    ) -> DesignSpec:
        """Construct a ``DesignSpec`` programmatically without a formula string."""
        if isinstance(X, pd.DataFrame):
            n_samples = X.shape[0]
            inferred_sample_names = [str(s) for s in X.index]
            inferred_fixed_names = [str(c) for c in X.columns]
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            if X_arr.ndim != 2:
                raise ValueError(f"Design matrix X must be 2D, got shape {X_arr.shape}.")
            n_samples = X_arr.shape[0]
            inferred_sample_names = [f"sample_{i}" for i in range(n_samples)]
            inferred_fixed_names = []
            for j in range(X_arr.shape[1]):
                if np.allclose(X_arr[:, j], 1.0, atol=1e-12):
                    inferred_fixed_names.append("Intercept")
                else:
                    inferred_fixed_names.append(f"x{j}")

        n_fixed = X_arr.shape[1]
        if n_samples < 1 or n_fixed < 1:
            raise ValueError(f"Design matrix X must be non-empty, got shape {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Design matrix X must not contain NaN or Inf values.")

        f_names = (
            inferred_fixed_names if fixed_names is None else [str(c) for c in fixed_names]
        )
        if len(f_names) != n_fixed:
            raise ValueError(
                f"fixed_names length ({len(f_names)}) must match X columns ({n_fixed})."
            )

        s_names = (
            inferred_sample_names if sample_names is None else [str(s) for s in sample_names]
        )
        if len(s_names) != n_samples:
            raise ValueError(
                f"sample_names length ({len(s_names)}) must match X rows ({n_samples})."
            )

        rank_x = int(np.linalg.matrix_rank(X_arr))
        if n_samples - rank_x < 1:
            raise ValueError(
                f"Design matrix X leaves no residual degrees of freedom "
                f"(n_samples={n_samples}, rank={rank_x})."
            )

        intercept_cols = [
            j
            for j in range(n_fixed)
            if np.allclose(X_arr[:, j], 1.0, atol=1e-12) or f_names[j] in ("Intercept", "(Intercept)")
        ]
        # Verify intercept column actually constant if matched by name
        intercept_cols = [
            j for j in intercept_cols if np.allclose(X_arr[:, j], X_arr[0, j], atol=1e-12) and abs(X_arr[0, j]) > 1e-12
        ]
        has_intercept = len(intercept_cols) > 0

        name_to_idx = {name: idx for idx, name in enumerate(f_names)}
        if fixed_terms is not None:
            f_terms: dict[str, list[int]] = {}
            for term, cols in fixed_terms.items():
                idx_list: list[int] = []
                for c in cols:
                    if isinstance(c, (int, np.integer)):
                        ci = int(c)
                        if ci < 0 or ci >= n_fixed:
                            raise ValueError(f"Column index {ci} out of bounds for n_fixed={n_fixed}.")
                        idx_list.append(ci)
                    else:
                        if str(c) not in name_to_idx:
                            raise ValueError(f"Fixed term column '{c}' not found in fixed_names.")
                        idx_list.append(name_to_idx[str(c)])
                if idx_list:
                    f_terms[str(term)] = idx_list
        else:
            intercept_set = set(intercept_cols)
            f_terms = {f_names[j]: [j] for j in range(n_fixed) if j not in intercept_set}

        r_matrices: list[np.ndarray] = []
        r_names: list[str] = []
        r_level_names: list[list[str]] = []

        if random_effects is not None:
            if isinstance(random_effects, Mapping):
                items = list(random_effects.items())
                default_names = [str(k) for k, _ in items]
                vals = [v for _, v in items]
            else:
                vals = list(random_effects)
                default_names = [f"re_{r}" for r in range(len(vals))]

            if random_names is not None:
                if len(random_names) != len(vals):
                    raise ValueError(
                        f"random_names length ({len(random_names)}) must match "
                        f"number of random effects ({len(vals)})."
                    )
                r_names = [str(n) for n in random_names]
            else:
                r_names = default_names

            for rname, val in zip(r_names, vals):
                if isinstance(val, str):
                    raise ValueError(
                        f"String random effect '{val}' requires `data` DataFrame; "
                        "pass a grouping array/Series or 2D Z matrix to DesignSpec.from_matrices."
                    )
                if isinstance(val, pd.DataFrame):
                    z_mat = val.to_numpy(dtype=np.float64)
                    if z_mat.shape[0] != n_samples or z_mat.shape[1] < 1:
                        raise ValueError(
                            f"Random effect '{rname}' has shape {z_mat.shape}, expected ({n_samples}, >=1)."
                        )
                    if not np.all(np.isfinite(z_mat)):
                        raise ValueError(f"Random effect '{rname}' contains NaN or Inf.")
                    r_matrices.append(np.ascontiguousarray(z_mat, dtype=np.float64))
                    r_level_names.append([str(c) for c in val.columns])
                else:
                    arr = np.asarray(val)
                    if arr.ndim == 1 or isinstance(val, pd.Series):
                        series = val if isinstance(val, pd.Series) else pd.Series(arr)
                        if series.shape[0] != n_samples:
                            raise ValueError(
                                f"Random effect '{rname}' length ({series.shape[0]}) != n_samples ({n_samples})."
                            )
                        z_mat, lvl_names = _encode_group_indicator(
                            series, rname, drop_unused_levels=True, min_levels=2
                        )
                        r_matrices.append(z_mat)
                        r_level_names.append(lvl_names)
                    elif arr.ndim == 2:
                        z_mat = np.asarray(arr, dtype=np.float64)
                        if z_mat.shape[0] != n_samples or z_mat.shape[1] < 1:
                            raise ValueError(
                                f"Random effect '{rname}' has shape {z_mat.shape}, expected ({n_samples}, >=1)."
                            )
                        if not np.all(np.isfinite(z_mat)):
                            raise ValueError(f"Random effect '{rname}' contains NaN or Inf.")
                        r_matrices.append(np.ascontiguousarray(z_mat, dtype=np.float64))
                        r_level_names.append([f"{rname}_{j}" for j in range(z_mat.shape[1])])
                    else:
                        raise ValueError(
                            f"Random effect '{rname}' must be 1D grouping labels or 2D matrix, got ndim={arr.ndim}."
                        )

        return cls(
            X=np.ascontiguousarray(X_arr, dtype=np.float64),
            fixed_names=f_names,
            fixed_terms=f_terms,
            has_intercept=has_intercept,
            intercept_cols=intercept_cols,
            random_matrices=r_matrices,
            random_names=r_names,
            random_level_names=r_level_names,
            sample_names=s_names,
        )


def _encode_group_indicator(
    series: pd.Series,
    name: str,
    *,
    drop_unused_levels: bool = True,
    min_levels: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Convert a 1D grouping Series into a one-hot ``(N, m_r)`` float64 matrix."""
    if series.isna().any():
        raise ValueError(f"Random effect grouping column '{name}' contains missing (NaN/NA) values.")

    if isinstance(series.dtype, pd.CategoricalDtype):
        s_cat = series.cat.remove_unused_categories() if drop_unused_levels else series
        categories = [str(c) for c in s_cat.cat.categories]
        codes = s_cat.cat.codes.to_numpy(dtype=np.int64)
    else:
        str_vals = series.astype(str)
        unique_vals = sorted(str_vals.unique().tolist())
        cat = pd.Categorical(str_vals, categories=unique_vals, ordered=True)
        categories = unique_vals
        codes = cat.codes.astype(np.int64)

    if len(categories) < min_levels:
        raise ValueError(
            f"Random effect grouping factor '{name}' must have at least {min_levels} distinct levels "
            f"(got {len(categories)})."
        )

    n = len(codes)
    m_r = len(categories)
    Z = np.zeros((n, m_r), dtype=np.float64)
    valid = codes >= 0
    Z[np.nonzero(valid)[0], codes[valid]] = 1.0
    return Z, categories


def _split_top_level(expr: str, sep_chars: str) -> list[tuple[str, str]]:
    """Split ``expr`` on top-level operators in ``sep_chars`` outside parentheses.

    Returns a list of ``(sign, token)`` tuples where ``sign`` is ``'+'`` or ``'-'``.
    """
    tokens: list[tuple[str, str]] = []
    depth = 0
    current: list[str] = []
    cur_sign = "+"
    i = 0
    s = expr.strip()
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unmatched ')' in formula expression: '{expr}'.")
            current.append(ch)
        elif depth == 0 and ch in sep_chars:
            tok = "".join(current).strip()
            if tok:
                tokens.append((cur_sign, tok))
            elif ch == "+" and not tokens:
                pass
            elif ch == "-" and not tokens:
                pass
            else:
                raise ValueError(f"Malformed operator '{ch}' in formula expression: '{expr}'.")
            cur_sign = ch
            current = []
        else:
            current.append(ch)
        i += 1
    if depth != 0:
        raise ValueError(f"Unmatched '(' in formula expression: '{expr}'.")
    tok = "".join(current).strip()
    if tok:
        tokens.append((cur_sign, tok))
    elif tokens:
        raise ValueError(f"Trailing operator '{cur_sign}' in formula expression: '{expr}'.")
    return tokens


def _split_char_top_level(expr: str, sep: str) -> list[str]:
    """Split ``expr`` by single character ``sep`` outside parentheses."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in expr:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unmatched ')' in formula: '{expr}'.")
            buf.append(ch)
        elif depth == 0 and ch == sep:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise ValueError(f"Unmatched '(' in formula: '{expr}'.")
    parts.append("".join(buf).strip())
    return parts


def _expand_fixed_token(token: str) -> list[str]:
    """Expand a fixed-effect term containing ``*``, ``/``, ``:``, and parentheses."""
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        # Check if outer parens wrap the entire token
        depth = 0
        wraps_all = True
        for idx, ch in enumerate(token):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(token) - 1:
                    wraps_all = False
                    break
        if wraps_all:
            inner = token[1:-1].strip()
            sub_tokens = _split_top_level(inner, "+-")
            out: list[str] = []
            for sign, sub in sub_tokens:
                if sign == "-":
                    raise ValueError(f"Unsupported '-' inside fixed-effect group '{token}'.")
                out.extend(_expand_fixed_token(sub))
            return out

    # Check top-level '*'
    star_parts = _split_char_top_level(token, "*")
    if len(star_parts) > 1:
        acc = _expand_fixed_token(star_parts[0])
        for nxt in star_parts[1:]:
            rhs = _expand_fixed_token(nxt)
            combined: list[str] = list(acc)
            for r in rhs:
                if r not in combined:
                    combined.append(r)
            for a in acc:
                for r in rhs:
                    inter = _canonical_interaction(a, r)
                    if inter not in combined:
                        combined.append(inter)
            acc = combined
        return acc

    # Check top-level '/' (nesting a/b -> a + a:b)
    slash_parts = _split_char_top_level(token, "/")
    if len(slash_parts) > 1:
        acc = _expand_fixed_token(slash_parts[0])
        prefix = list(acc)
        for nxt in slash_parts[1:]:
            rhs = _expand_fixed_token(nxt)
            new_inters: list[str] = []
            for p in prefix:
                for r in rhs:
                    inter = _canonical_interaction(p, r)
                    if inter not in acc:
                        acc.append(inter)
                    new_inters.append(inter)
            prefix = new_inters
        return acc

    # Check top-level ':'
    colon_parts = _split_char_top_level(token, ":")
    if len(colon_parts) > 1:
        acc = _expand_fixed_token(colon_parts[0])
        for nxt in colon_parts[1:]:
            rhs = _expand_fixed_token(nxt)
            acc = [_canonical_interaction(a, r) for a in acc for r in rhs]
        return acc

    if not token:
        raise ValueError("Empty term encountered in formula.")
    return [token]


def _canonical_interaction(a: str, b: str) -> str:
    """Combine two factor/interaction strings with ':'."""
    if a == "1":
        return b
    if b == "1":
        return a
    parts_a = a.split(":")
    parts_b = b.split(":")
    merged = list(parts_a)
    for pb in parts_b:
        if pb not in merged:
            merged.append(pb)
    return ":".join(merged)


def _expand_random_group(rhs: str, data: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Expand a random-effect grouping expression (``g``, ``a:b``, or ``a/b``)."""
    rhs = rhs.strip()
    if not rhs:
        raise ValueError("Empty random-effect grouping expression after '|'.")

    if "/" in rhs:
        parts = [p.strip() for p in rhs.split("/")]
        if any(not p for p in parts):
            raise ValueError(f"Malformed nested random effect '{rhs}'.")
        out: list[tuple[str, pd.Series]] = []
        cum_cols: list[str] = []
        for p in parts:
            sub_cols = [c.strip() for c in p.split(":")]
            cum_cols.extend(sub_cols)
            term_name = ":".join(cum_cols)
            series = _combine_group_columns(cum_cols, data, term_name)
            out.append((term_name, series))
        return out

    cols = [c.strip() for c in rhs.split(":")]
    if any(not c for c in cols):
        raise ValueError(f"Malformed random effect grouping '{rhs}'.")
    term_name = ":".join(cols)
    return [(term_name, _combine_group_columns(cols, data, term_name))]


def _combine_group_columns(cols: list[str], data: pd.DataFrame, term_name: str) -> pd.Series:
    """Combine one or more grouping columns into a single Series."""
    for c in cols:
        if c not in data.columns:
            raise ValueError(
                f"Random-effect column '{c}' (in '{term_name}') not found in data columns: "
                f"{list(data.columns)}."
            )
        if data[c].isna().any():
            raise ValueError(f"Random-effect column '{c}' contains missing (NaN/NA) values.")
    if len(cols) == 1:
        return data[cols[0]]
    combined = data[cols[0]].astype(str)
    for c in cols[1:]:
        combined = combined + ":" + data[c].astype(str)
    return combined


def _is_categorical_series(s: pd.Series) -> bool:
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(s.dtype)
        or pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
    )


def _encode_fixed_variable(
    col: str,
    data: pd.DataFrame,
    *,
    drop_first: bool,
    drop_unused_levels: bool = True,
) -> tuple[np.ndarray, list[str], bool]:
    """Encode a single fixed-effect variable into columns and names."""
    if col not in data.columns:
        raise ValueError(
            f"Fixed-effect variable '{col}' not found in data columns: {list(data.columns)}."
        )
    s = data[col]
    if s.isna().any():
        raise ValueError(f"Fixed-effect variable '{col}' contains missing (NaN/NA) values.")

    if _is_categorical_series(s):
        if isinstance(s.dtype, pd.CategoricalDtype):
            s_cat = s.cat.remove_unused_categories() if drop_unused_levels else s
            levels = list(s_cat.cat.categories)
            codes = s_cat.cat.codes.to_numpy(dtype=np.int64)
        elif pd.api.types.is_bool_dtype(s.dtype):
            obs = set(bool(v) for v in s.unique())
            levels = [lvl for lvl in (False, True) if lvl in obs] if drop_unused_levels else [False, True]
            cat = pd.Categorical(s.astype(bool), categories=levels, ordered=True)
            codes = cat.codes.astype(np.int64)
        else:
            str_s = s.astype(str)
            levels = sorted(str_s.unique().tolist())
            cat = pd.Categorical(str_s, categories=levels, ordered=True)
            codes = cat.codes.astype(np.int64)

        if len(levels) < 2:
            raise ValueError(
                f"Categorical fixed factor '{col}' must have at least 2 observed levels (got {levels})."
            )
        start_idx = 1 if drop_first else 0
        cols_list: list[np.ndarray] = []
        names_list: list[str] = []
        for k in range(start_idx, len(levels)):
            ind = (codes == k).astype(np.float64)[:, None]
            cols_list.append(ind)
            names_list.append(f"{col}_{levels[k]}")
        return np.hstack(cols_list), names_list, True

    arr = s.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Numeric fixed-effect variable '{col}' contains Inf or NaN values.")
    return arr[:, None], [col], False


def build_design(
    formula: str,
    data: pd.DataFrame,
    drop_unused_levels: bool = True,
) -> DesignSpec:
    """Parse an lme4/variancePartition-style mixed-effects formula into a ``DesignSpec``.

    Supports fixed effects (``1``, ``0``/``-1``, numeric, categorical, ``:``, ``*``, ``/``,
    and parenthesized groups), random intercepts ``(1 | grp)``, nested ``(1 | a/b)``,
    crossed ``(1 | a:b)``, pure random slopes ``(0 + x | grp)``, and uncorrelated
    intercept+slope ``(1 + x || grp)``. Correlated random slopes ``(1 + x | grp)``
    raise ``ValueError``.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula must be a non-empty string (e.g. '~ condition + (1 | subject)').")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if data.shape[0] < 1:
        raise ValueError("`data` must contain at least 1 sample.")

    if "~" not in formula:
        raise ValueError(f"Formula must contain '~' (got '{formula}').")
    _, rhs = formula.split("~", 1)
    rhs = rhs.strip()
    if not rhs:
        raise ValueError(f"Formula right-hand side is empty: '{formula}'.")

    top_tokens = _split_top_level(rhs, "+-")

    has_intercept = True
    raw_fixed_tokens: list[tuple[str, str]] = []
    random_specs: list[tuple[str, str, str | None]] = []  # (term_name, grp_expr, slope_var_or_None)

    for sign, tok in top_tokens:
        tok_s = tok.strip()
        if tok_s.startswith("(") and tok_s.endswith(")") and "|" in tok_s:
            if sign == "-":
                raise ValueError(f"Removing random-effect term '- {tok_s}' is not supported.")
            inner = tok_s[1:-1].strip()
            if "||" in inner:
                parts = inner.split("||")
                if len(parts) != 2:
                    raise ValueError(f"Malformed random-effect term '{tok_s}'.")
                re_lhs, re_rhs = parts[0].strip(), parts[1].strip()
                lhs_items = _split_top_level(re_lhs, "+-")
                include_re_intercept = True
                slope_vars: list[str] = []
                for lsign, ltok in lhs_items:
                    if ltok in ("0",) or (lsign == "-" and ltok == "1"):
                        include_re_intercept = False
                    elif ltok == "1" and lsign == "+":
                        include_re_intercept = True
                    elif lsign == "+":
                        slope_vars.append(ltok)
                    else:
                        raise ValueError(f"Unsupported term '{lsign} {ltok}' in random effect '{tok_s}'.")
                if include_re_intercept:
                    random_specs.append(("intercept", re_rhs, None))
                for sv in slope_vars:
                    random_specs.append(("slope", re_rhs, sv))
            else:
                parts = inner.split("|")
                if len(parts) != 2:
                    raise ValueError(f"Malformed random-effect term '{tok_s}'.")
                re_lhs, re_rhs = parts[0].strip(), parts[1].strip()
                lhs_items = _split_top_level(re_lhs, "+-")
                if len(lhs_items) == 1 and lhs_items[0] == ("+", "1"):
                    random_specs.append(("intercept", re_rhs, None))
                elif (
                    len(lhs_items) == 2
                    and (lhs_items[0] == ("+", "0") or lhs_items[0] == ("-", "1"))
                    and lhs_items[1][0] == "+"
                    and lhs_items[1][1] not in ("0", "1")
                ):
                    random_specs.append(("slope", re_rhs, lhs_items[1][1]))
                else:
                    raise ValueError(
                        f"Correlated or multi-term single-bar random effect '{tok_s}' is not supported. "
                        "Use uncorrelated '(1 + x || group)' or '(1 | group) + (0 + x | group)' instead."
                    )
        else:
            raw_fixed_tokens.append((sign, tok_s))

    # Process fixed tokens
    fixed_term_order: list[str] = []
    explicit_one = False
    for sign, tok_s in raw_fixed_tokens:
        if tok_s == "0" and sign == "+":
            has_intercept = False
            continue
        if tok_s == "1" and sign == "-":
            has_intercept = False
            continue
        if tok_s == "1" and sign == "+":
            explicit_one = True
            continue
        expanded = _expand_fixed_token(tok_s)
        if sign == "+":
            for t in expanded:
                if t == "1":
                    explicit_one = True
                elif t == "0":
                    has_intercept = False
                elif t not in fixed_term_order:
                    fixed_term_order.append(t)
        else:
            for t in expanded:
                if t in fixed_term_order:
                    fixed_term_order.remove(t)

    n_samples = data.shape[0]
    X_blocks: list[np.ndarray] = []
    fixed_names: list[str] = []
    fixed_terms: dict[str, list[int]] = {}
    intercept_cols: list[int] = []
    cur_col = 0

    if has_intercept or (explicit_one and not fixed_term_order):
        has_intercept = True
        X_blocks.append(np.ones((n_samples, 1), dtype=np.float64))
        fixed_names.append("Intercept")
        intercept_cols.append(cur_col)
        cur_col += 1

    # Encode each non-intercept fixed term
    encoded_main_cache: dict[tuple[str, bool], tuple[np.ndarray, list[str], bool]] = {}
    used_full_factor = has_intercept

    for term in fixed_term_order:
        factors = term.split(":")
        if len(factors) == 1:
            f = factors[0]
            drop_first = used_full_factor
            mat, names, is_cat = _encode_fixed_variable(
                f, data, drop_first=drop_first, drop_unused_levels=drop_unused_levels
            )
            encoded_main_cache[(f, drop_first)] = (mat, names, is_cat)
            if is_cat and not drop_first:
                used_full_factor = True
        else:
            # Interaction term
            factor_mats: list[tuple[np.ndarray, list[str]]] = []
            for idx_f, f in enumerate(factors):
                # Use reduced coding if intercept/full-factor already present or main effect exists
                if used_full_factor or f in fixed_term_order:
                    drop_first = True
                else:
                    drop_first = idx_f > 0
                key = (f, drop_first)
                if key not in encoded_main_cache:
                    encoded_main_cache[key] = _encode_fixed_variable(
                        f, data, drop_first=drop_first, drop_unused_levels=drop_unused_levels
                    )
                mat_f, names_f, is_cat_f = encoded_main_cache[key]
                if is_cat_f and not drop_first:
                    used_full_factor = True
                factor_mats.append((mat_f, names_f))

            mat, names = factor_mats[0]
            for mat_next, names_next in factor_mats[1:]:
                prod_cols = []
                prod_names = []
                for i_a in range(mat.shape[1]):
                    for i_b in range(mat_next.shape[1]):
                        prod_cols.append((mat[:, i_a] * mat_next[:, i_b])[:, None])
                        prod_names.append(f"{names[i_a]}:{names_next[i_b]}")
                mat = np.hstack(prod_cols)
                names = prod_names

        cols_idx = list(range(cur_col, cur_col + mat.shape[1]))
        X_blocks.append(mat)
        fixed_names.extend(names)
        fixed_terms[term] = cols_idx
        cur_col += mat.shape[1]

    if not X_blocks:
        raise ValueError(f"Formula '{formula}' produced an empty fixed-effects design matrix.")

    X = np.ascontiguousarray(np.hstack(X_blocks), dtype=np.float64)
    rank_x = int(np.linalg.matrix_rank(X))
    if n_samples - rank_x < 1:
        raise ValueError(
            f"Fixed-effects design leaves no residual degrees of freedom "
            f"(n_samples={n_samples}, rank={rank_x})."
        )

    # Encode random effects
    random_matrices: list[np.ndarray] = []
    random_names: list[str] = []
    random_level_names: list[list[str]] = []
    seen_random_names: dict[str, int] = {}

    for kind, re_rhs, slope_var in random_specs:
        groups = _expand_random_group(re_rhs, data)
        for grp_name, grp_series in groups:
            Z_base, lvl_names = _encode_group_indicator(
                grp_series, grp_name, drop_unused_levels=drop_unused_levels, min_levels=2
            )
            if kind == "intercept":
                Z_mat = Z_base
                rname = grp_name
                r_lvls = lvl_names
            else:
                assert slope_var is not None
                if slope_var not in data.columns:
                    raise ValueError(
                        f"Random-slope covariate '{slope_var}' not found in data columns."
                    )
                s_col = data[slope_var]
                if s_col.isna().any():
                    raise ValueError(f"Random-slope covariate '{slope_var}' contains NaN/NA.")
                if _is_categorical_series(s_col):
                    raise ValueError(
                        f"Random-slope covariate '{slope_var}' must be numeric (got {s_col.dtype})."
                    )
                x_slope = s_col.to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(x_slope)):
                    raise ValueError(f"Random-slope covariate '{slope_var}' contains Inf/NaN.")
                Z_mat = Z_base * x_slope[:, None]
                rname = f"{grp_name}:{slope_var}"
                r_lvls = [f"{lvl}:{slope_var}" for lvl in lvl_names]

            if rname in seen_random_names:
                seen_random_names[rname] += 1
                rname_unique = f"{rname}_{seen_random_names[rname]}"
            else:
                seen_random_names[rname] = 1
                rname_unique = rname

            random_matrices.append(np.ascontiguousarray(Z_mat, dtype=np.float64))
            random_names.append(rname_unique)
            random_level_names.append(r_lvls)

    return DesignSpec(
        X=X,
        fixed_names=fixed_names,
        fixed_terms=fixed_terms,
        has_intercept=has_intercept,
        intercept_cols=intercept_cols,
        random_matrices=random_matrices,
        random_names=random_names,
        random_level_names=random_level_names,
        sample_names=[str(s) for s in data.index],
    )


def _eval_linear_contrast_ast(expr: str, fixed_names: Sequence[str]) -> np.ndarray:
    """Safely evaluate a linear combination string over ``fixed_names`` via Python AST."""
    name_to_idx = {name: i for i, name in enumerate(fixed_names)}
    p = len(fixed_names)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid contrast expression '{expr}': {exc}") from exc

    def _eval(node: ast.AST) -> tuple[np.ndarray | None, float | None]:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Name):
            if node.id not in name_to_idx:
                raise ValueError(
                    f"Unknown coefficient '{node.id}' in contrast '{expr}'. "
                    f"Available coefficients: {list(fixed_names)}."
                )
            vec = np.zeros(p, dtype=np.float64)
            vec[name_to_idx[node.id]] = 1.0
            return vec, None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return None, float(node.value)
        if isinstance(node, ast.UnaryOp):
            vec, scal = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return vec, scal
            if isinstance(node.op, ast.USub):
                return (-vec if vec is not None else None), (-scal if scal is not None else None)
            raise ValueError(f"Unsupported unary operator in contrast '{expr}'.")
        if isinstance(node, ast.BinOp):
            lv, ls = _eval(node.left)
            rv, rs = _eval(node.right)
            if isinstance(node.op, ast.Add):
                if lv is not None and rv is not None:
                    return lv + rv, None
                if ls is not None and rs is not None:
                    return None, ls + rs
                raise ValueError(f"Cannot add scalar and coefficient vector in contrast '{expr}'.")
            if isinstance(node.op, ast.Sub):
                if lv is not None and rv is not None:
                    return lv - rv, None
                if ls is not None and rs is not None:
                    return None, ls - rs
                raise ValueError(f"Cannot subtract scalar and coefficient vector in contrast '{expr}'.")
            if isinstance(node.op, ast.Mult):
                if lv is not None and rs is not None:
                    return lv * rs, None
                if ls is not None and rv is not None:
                    return ls * rv, None
                if ls is not None and rs is not None:
                    return None, ls * rs
                raise ValueError(f"Nonlinear coefficient multiplication is not allowed in contrast '{expr}'.")
            if isinstance(node.op, ast.Div):
                if rs is not None:
                    if rs == 0.0:
                        raise ValueError(f"Division by zero in contrast '{expr}'.")
                    if lv is not None:
                        return lv / rs, None
                    if ls is not None:
                        return None, ls / rs
                raise ValueError(f"Division by a coefficient is not allowed in contrast '{expr}'.")
            raise ValueError(f"Unsupported operator in contrast '{expr}'.")
        raise ValueError(f"Unsupported syntax in contrast expression '{expr}'.")

    vec, _ = _eval(tree)
    if vec is None:
        raise ValueError(f"Contrast expression '{expr}' evaluated to a scalar instead of a coefficient contrast.")
    if np.allclose(vec, 0.0, atol=1e-15):
        raise ValueError(f"Contrast expression '{expr}' evaluated to the zero vector.")
    return vec


def make_contrasts(
    contrasts: Mapping[str, str | Sequence[float] | np.ndarray] | Sequence[str] | np.ndarray | pd.DataFrame | str,
    fixed_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Build a ``(p, L)`` contrast matrix and contrast names from strings, dicts, or arrays."""
    p = len(fixed_names)
    if p < 1:
        raise ValueError("fixed_names must be non-empty.")

    if isinstance(contrasts, str):
        if contrasts in fixed_names:
            vec = np.zeros(p, dtype=np.float64)
            vec[list(fixed_names).index(contrasts)] = 1.0
        else:
            vec = _eval_linear_contrast_ast(contrasts, fixed_names)
        return vec[:, None], [contrasts]

    if isinstance(contrasts, Mapping):
        if not contrasts:
            raise ValueError("contrasts mapping must not be empty.")
        cols: list[np.ndarray] = []
        names: list[str] = []
        for cname, spec in contrasts.items():
            if isinstance(spec, str):
                vec = (
                    _one_hot(spec, fixed_names)
                    if spec in fixed_names
                    else _eval_linear_contrast_ast(spec, fixed_names)
                )
            else:
                vec = np.asarray(spec, dtype=np.float64).ravel()
                if vec.size != p:
                    raise ValueError(
                        f"Contrast '{cname}' has length {vec.size}, expected {p}."
                    )
                if not np.all(np.isfinite(vec)) or np.allclose(vec, 0.0, atol=1e-15):
                    raise ValueError(f"Contrast '{cname}' must be finite and non-zero.")
            cols.append(vec[:, None])
            names.append(str(cname))
        return np.hstack(cols), names

    if isinstance(contrasts, pd.DataFrame):
        mat = contrasts.to_numpy(dtype=np.float64)
        if mat.shape[0] != p:
            raise ValueError(f"Contrast DataFrame has {mat.shape[0]} rows, expected {p}.")
        if mat.shape[1] < 1 or not np.all(np.isfinite(mat)):
            raise ValueError("Contrast DataFrame must be finite and have >= 1 column.")
        return mat, [str(c) for c in contrasts.columns]

    if isinstance(contrasts, Sequence) and len(contrasts) > 0 and isinstance(contrasts[0], str):
        cols = []
        names = []
        for s in contrasts:
            if not isinstance(s, str):
                raise ValueError("Mixed string/non-string entries in contrasts sequence.")
            vec = _one_hot(s, fixed_names) if s in fixed_names else _eval_linear_contrast_ast(s, fixed_names)
            cols.append(vec[:, None])
            names.append(s)
        return np.hstack(cols), names

    arr = np.asarray(contrasts, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != p:
            raise ValueError(f"Contrast vector has length {arr.size}, expected {p}.")
        if not np.all(np.isfinite(arr)) or np.allclose(arr, 0.0, atol=1e-15):
            raise ValueError("Contrast vector must be finite and non-zero.")
        return arr[:, None], ["contrast_0"]
    if arr.ndim == 2:
        if arr.shape[0] != p or arr.shape[1] < 1:
            raise ValueError(f"Contrast matrix has shape {arr.shape}, expected ({p}, >=1).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Contrast matrix contains NaN or Inf.")
        for l_idx in range(arr.shape[1]):
            if np.allclose(arr[:, l_idx], 0.0, atol=1e-15):
                raise ValueError(f"Contrast column {l_idx} is the zero vector.")
        return np.ascontiguousarray(arr, dtype=np.float64), [f"contrast_{l}" for l in range(arr.shape[1])]

    raise ValueError(f"Unsupported contrasts specification of type {type(contrasts).__name__}.")


def _one_hot(name: str, fixed_names: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(fixed_names), dtype=np.float64)
    vec[list(fixed_names).index(name)] = 1.0
    return vec


def _is_anndata(obj: Any) -> bool:
    return (
        hasattr(obj, "X")
        and hasattr(obj, "obs")
        and hasattr(obj, "var")
        and hasattr(obj, "obs_names")
        and hasattr(obj, "var_names")
        and hasattr(obj, "layers")
    )


def _to_dense_float64(mat: Any) -> np.ndarray:
    if sparse.issparse(mat):
        return np.asarray(mat.toarray(), dtype=np.float64)
    return np.asarray(mat, dtype=np.float64)


def coerce_expression_and_weights(
    expr: Any,
    formula: str | DesignSpec | None = None,
    data: pd.DataFrame | None = None,
    *,
    design: np.ndarray | pd.DataFrame | None = None,
    random_effects: (
        Sequence[np.ndarray | pd.Series | pd.DataFrame | str]
        | Mapping[str, np.ndarray | pd.Series | pd.DataFrame]
        | None
    ) = None,
    weights: np.ndarray | pd.DataFrame | str | None = None,
    layer: str | None = None,
    genes_as_rows: bool = False,
    gene_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, bool, DesignSpec, list[str]]:
    """Coerce expression matrix, optional precision weights, and design into canonical ``(N, G)`` form.

    Returns
    -------
    Y : np.ndarray
        Expression matrix of shape ``(N_samples, G_genes)`` (float64).
    W : np.ndarray | None
        Precision weights: ``None`` (uniform), shape ``(N,)`` (shared), or shape ``(N, G)`` (per-gene).
    shared_weights : bool
        True if ``W`` is ``None`` or 1D ``(N,)`` (or identical across all genes).
    design_spec : DesignSpec
        Validated ``DesignSpec`` matching ``N_samples``.
    resolved_gene_names : list[str]
        Length-``G`` list of gene names.
    """
    is_adata = _is_anndata(expr)
    adata_obs: pd.DataFrame | None = None
    inferred_sample_names: list[str] | None = None
    inferred_gene_names: list[str] | None = None

    if is_adata:
        if layer is not None:
            if layer not in expr.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData.layers.")
            raw_x = expr.layers[layer]
        else:
            raw_x = expr.X
        Y = _to_dense_float64(raw_x)
        inferred_sample_names = [str(s) for s in expr.obs_names]
        inferred_gene_names = [str(g) for g in expr.var_names]
        adata_obs = expr.obs
        if data is None:
            data = adata_obs
    elif isinstance(expr, pd.DataFrame):
        mat = expr.to_numpy(dtype=np.float64)
        if genes_as_rows:
            Y = np.ascontiguousarray(mat.T, dtype=np.float64)
            inferred_sample_names = [str(c) for c in expr.columns]
            inferred_gene_names = [str(idx) for idx in expr.index]
        else:
            Y = np.ascontiguousarray(mat, dtype=np.float64)
            inferred_sample_names = [str(idx) for idx in expr.index]
            inferred_gene_names = [str(c) for c in expr.columns]
    else:
        mat = _to_dense_float64(expr)
        if mat.ndim != 2:
            raise ValueError(f"Expression matrix must be 2D, got shape {mat.shape}.")
        Y = np.ascontiguousarray(mat.T if genes_as_rows else mat, dtype=np.float64)
        inferred_sample_names = [f"sample_{i}" for i in range(Y.shape[0])]
        inferred_gene_names = [f"gene_{g}" for g in range(Y.shape[1])]

    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError(f"Expression matrix must be non-empty 2D, got shape {Y.shape}.")

    n_samples, n_genes = Y.shape
    resolved_gene_names = (
        inferred_gene_names if gene_names is None else [str(g) for g in gene_names]
    )
    if len(resolved_gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(resolved_gene_names)}) must match number of genes ({n_genes})."
        )

    # Resolve DesignSpec
    if isinstance(formula, DesignSpec):
        design_spec = formula
    elif isinstance(formula, str):
        if data is None:
            raise ValueError("`data` DataFrame (or AnnData with .obs) is required when passing a formula string.")
        design_spec = build_design(formula, data)
    elif formula is None and design is not None:
        resolved_re: Any = random_effects
        if random_effects is not None and not isinstance(random_effects, Mapping):
            re_list = []
            re_names_auto: list[str] = []
            for idx, item in enumerate(random_effects):
                if isinstance(item, str):
                    if data is None or item not in data.columns:
                        raise ValueError(f"Random-effect column '{item}' not found in `data`.")
                    re_list.append(data[item])
                    re_names_auto.append(item)
                else:
                    re_list.append(item)
                    re_names_auto.append(getattr(item, "name", None) or f"re_{idx}")
            resolved_re = dict(zip(re_names_auto, re_list))
        design_spec = DesignSpec.from_matrices(
            design,
            resolved_re,
            sample_names=inferred_sample_names,
        )
    else:
        raise ValueError("Must provide either `formula` (with `data`), a `DesignSpec`, or `design`.")

    if design_spec.n_samples != n_samples:
        hint = (
            " If genes are rows and samples are columns, pass `genes_as_rows=True`."
            if Y.shape[1] == design_spec.n_samples and not genes_as_rows and not is_adata
            else ""
        )
        raise ValueError(
            f"Expression matrix has {n_samples} samples, which does not match design samples "
            f"({design_spec.n_samples}).{hint}"
        )

    # Resolve weights
    W_out: np.ndarray | None = None
    shared_weights = True
    if weights is not None:
        if isinstance(weights, str):
            if not is_adata:
                raise ValueError("String `weights` layer name is only supported when `expr` is an AnnData.")
            if weights not in expr.layers:
                raise ValueError(f"Weight layer '{weights}' not found in AnnData.layers.")
            w_raw = _to_dense_float64(expr.layers[weights])
        elif isinstance(weights, pd.DataFrame):
            w_arr = weights.to_numpy(dtype=np.float64)
            w_raw = np.ascontiguousarray(w_arr.T if genes_as_rows else w_arr, dtype=np.float64)
        else:
            w_arr = _to_dense_float64(weights)
            if w_arr.ndim == 1:
                w_raw = w_arr
            elif w_arr.ndim == 2:
                w_raw = np.ascontiguousarray(w_arr.T if genes_as_rows else w_arr, dtype=np.float64)
            else:
                raise ValueError(f"weights must be 1D or 2D, got shape {w_arr.shape}.")

        if w_raw.ndim == 1:
            if w_raw.shape[0] != n_samples:
                raise ValueError(f"1D weights length ({w_raw.shape[0]}) != n_samples ({n_samples}).")
            if not np.all(np.isfinite(w_raw)) or np.any(w_raw <= 0.0):
                raise ValueError("Precision weights must be strictly positive and finite.")
            W_out = np.ascontiguousarray(w_raw, dtype=np.float64)
            shared_weights = True
        elif w_raw.ndim == 2:
            if w_raw.shape != (n_samples, n_genes):
                raise ValueError(
                    f"2D weights shape {w_raw.shape} does not match expression shape ({n_samples}, {n_genes})."
                )
            # Validate finite & positive on observed entries (or all entries if complete)
            obs_mask = np.isfinite(Y)
            w_obs = w_raw[obs_mask]
            if not np.all(np.isfinite(w_obs)) or np.any(w_obs <= 0.0):
                raise ValueError("Precision weights must be strictly positive and finite.")
            # Check if weights are actually identical across all genes
            if n_genes == 1 or (np.all(np.isfinite(w_raw)) and np.allclose(w_raw, w_raw[:, :1], atol=1e-14, rtol=1e-14)):
                W_out = np.ascontiguousarray(w_raw[:, 0], dtype=np.float64)
                shared_weights = True
            else:
                W_out = np.ascontiguousarray(w_raw, dtype=np.float64)
                shared_weights = False
        else:
            raise ValueError(f"weights must be 1D or 2D, got ndim={w_raw.ndim}.")

    return Y, W_out, shared_weights, design_spec, resolved_gene_names
