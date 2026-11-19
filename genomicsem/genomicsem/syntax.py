"""
`lavaan`-style structural equation syntax parser (`parse_lavaan_syntax`) and
Reticular Action Model (`RAMSpec`) matrix compiler for `genomicsem`
(Grotzinger et al., Nature Human Behaviour 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .ldsc import SettingsMode, validate_settings

_TERM_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?\s*\*?\s*(.+)$")


@dataclass
class ModelTermSpec:
    """Single parsed structural or covariance relationship in a `lavaan` model."""

    lhs: str
    op: str  # '=~', '~', or '~~'
    rhs: str
    free: bool
    fixed_val: float
    param_idx: int  # index in free theta vector (-1 if fixed)
    label: str
    is_variance: bool


@dataclass
class RAMSpec:
    """
    Compiled Reticular Action Model (`RAM`) representation:
    $\\Sigma(\\boldsymbol{\\theta}) = F (I_m - A(\\boldsymbol{\\theta}))^{-1} S_{\\text{ram}}(\\boldsymbol{\\theta}) (I_m - A(\\boldsymbol{\\theta}))^{-\\top} F^\\top$.
    """

    observed_traits: list[str]
    latent_vars: list[str]
    all_vars: list[str]
    F: np.ndarray  # (k, m)
    A_fixed: np.ndarray  # (m, m)
    S_fixed: np.ndarray  # (m, m)
    # Per free parameter r in 0..q-1: list of (row, col) entries in A or S_ram
    A_free_entries: list[list[tuple[int, int]]]
    S_free_entries: list[list[tuple[int, int]]]
    is_free_variance: np.ndarray  # bool array of length q
    terms: list[ModelTermSpec]
    n_free: int
    std_lv: bool
    settings: SettingsMode = "default"

    def build_matrices(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Populate `(A(theta), S_ram(theta))` from free parameter vector `theta`."""
        A = self.A_fixed.copy()
        S = self.S_fixed.copy()
        for r, val in enumerate(theta):
            fval = float(val)
            for u, v in self.A_free_entries[r]:
                A[u, v] = fval
            for u, v in self.S_free_entries[r]:
                S[u, v] = fval
                S[v, u] = fval
        return A, S


def _parse_single_rhs_token(token: str, sign_prefix: float = 1.0) -> tuple[str | None, float | None, str]:
    """
    Parse a single RHS token like `'T1'`, `'1*T1'`, `'0*T2'`, `'NA*T1'`, `'l1*T3'`.

    Returns `(modifier_label_or_NA, fixed_float_or_None, var_name)`.
    """
    tok = token.strip()
    if not tok:
        raise ValueError("Empty RHS term in model syntax.")

    if "*" in tok:
        left, right = tok.split("*", 1)
        mod = left.strip()
        var = right.strip()
        if not var:
            raise ValueError(f"Missing variable name after '*' in {token!r}.")
        if mod.upper() == "NA":
            return ("NA", None, var)
        try:
            fval = float(mod) * sign_prefix
            return (None, fval, var)
        except ValueError:
            # Parameter equality-constraint label
            if sign_prefix < 0:
                raise ValueError(f"Negative sign not supported on labeled parameter {token!r}.")
            return (mod, None, var)

    return (None, None, tok)


def _split_rhs_terms(rhs_str: str) -> list[tuple[float, str]]:
    """Split `rhs_str` on top-level `+` / `-` signs (preserving scientific notation like `1e-3*T1`)."""
    items: list[tuple[float, str]] = []
    cur: list[str] = []
    cur_sign = 1.0
    i = 0
    s = rhs_str.strip()
    while i < len(s):
        ch = s[i]
        if ch in "+-":
            # Check if part of scientific notation e.g. 1e-3*T1
            prev_nonspace = "".join(cur).rstrip()
            if prev_nonspace and prev_nonspace[-1] in ("e", "E"):
                cur.append(ch)
                i += 1
                continue
            if not prev_nonspace:
                if ch == "-":
                    cur_sign *= -1.0
                i += 1
                continue
            items.append((cur_sign, "".join(cur).strip()))
            cur = []
            cur_sign = -1.0 if ch == "-" else 1.0
            i += 1
        else:
            cur.append(ch)
            i += 1
    tail = "".join(cur).strip()
    if tail:
        items.append((cur_sign, tail))
    return items


def parse_lavaan_syntax(
    model_str: str,
    observed_traits: Sequence[str],
    std_lv: bool = True,
    settings: SettingsMode = "default",
) -> RAMSpec:
    """
    Parse `lavaan`-compatible structural equation syntax into a compiled `RAMSpec`.

    Supports:
    - Latent measurement equations: `'F1 =~ T1 + T2 + T3'`, `'F1 =~ 1*T1 + 0*T2 + NA*T3'`
    - Structural regressions: `'F2 ~ F1'`, `'T3 ~ F1 + T1'`
    - Residual variances & covariances: `'T1 ~~ T2'`, `'T1 ~~ 0*T1'`, `'F1 ~~ F2'`
    - Equality constraints via parameter labels: `'F1 =~ l1*T1 + l1*T2'`
    """
    validate_settings(settings)
    obs_list = [str(t) for t in observed_traits]
    if len(obs_list) < 2:
        raise ValueError("observed_traits must contain at least 2 traits.")
    obs_set = set(obs_list)

    raw_statements: list[str] = []
    for line in model_str.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        for part in clean.split(";"):
            p = part.strip()
            if p:
                raw_statements.append(p)

    if not raw_statements:
        raise ValueError("model_str contains no valid model statements.")

    # Parse raw tuples: (lhs, op, rhs, mod_label, fixed_val)
    parsed_raw: list[tuple[str, str, str, str | None, float | None]] = []
    latent_order: list[str] = []

    def _register_latent_if_needed(vname: str) -> None:
        if vname not in obs_set and vname not in latent_order:
            latent_order.append(vname)

    for stmt in raw_statements:
        if "=~" in stmt:
            op = "=~"
            lhs, rhs_all = stmt.split("=~", 1)
        elif "~~" in stmt:
            op = "~~"
            lhs, rhs_all = stmt.split("~~", 1)
        elif "~" in stmt:
            op = "~"
            lhs, rhs_all = stmt.split("~", 1)
        else:
            raise ValueError(f"Unrecognized operator in statement: {stmt!r}")

        lhs_v = lhs.strip()
        if not lhs_v:
            raise ValueError(f"Empty LHS in statement: {stmt!r}")
        if op == "=~" and lhs_v in obs_set:
            raise ValueError(
                f"Observed trait {lhs_v!r} cannot be on LHS of '=~' latent definition."
            )
        _register_latent_if_needed(lhs_v)

        rhs_tokens = _split_rhs_terms(rhs_all)
        if not rhs_tokens:
            raise ValueError(f"Empty RHS in statement: {stmt!r}")

        for sgn, tok in rhs_tokens:
            mod_lbl, fixed_val, rhs_v = _parse_single_rhs_token(tok, sign_prefix=sgn)
            _register_latent_if_needed(rhs_v)
            parsed_raw.append((lhs_v, op, rhs_v, mod_lbl, fixed_val))

    # Handle std_lv=False unit-loading identification on first indicator of each latent factor
    explicit_var_specs: set[tuple[str, str]] = set()
    for lhs_v, op, rhs_v, _, _ in parsed_raw:
        if op == "~~":
            explicit_var_specs.add((lhs_v, rhs_v))
            explicit_var_specs.add((rhs_v, lhs_v))

    first_loading_seen: set[str] = set()
    normalized_entries: list[tuple[str, str, str, str | None, float | None]] = []
    for lhs_v, op, rhs_v, mod_lbl, fixed_val in parsed_raw:
        if op == "=~":
            if lhs_v not in first_loading_seen:
                first_loading_seen.add(lhs_v)
                if not std_lv and mod_lbl is None and fixed_val is None:
                    fixed_val = 1.0
        normalized_entries.append((lhs_v, op, rhs_v, mod_lbl, fixed_val))

    # Add default observed residual variances T_i ~~ T_i if not explicitly specified
    for t in obs_list:
        if (t, t) not in explicit_var_specs:
            normalized_entries.append((t, "~~", t, None, None))
            explicit_var_specs.add((t, t))

    # Add default latent variances F_j ~~ F_j if not explicitly specified
    for lat in latent_order:
        if (lat, lat) not in explicit_var_specs:
            if std_lv:
                normalized_entries.append((lat, "~~", lat, None, 1.0))
            else:
                normalized_entries.append((lat, "~~", lat, None, None))
            explicit_var_specs.add((lat, lat))

    all_vars = obs_list + latent_order
    var_to_idx = {v: idx for idx, v in enumerate(all_vars)}
    k = len(obs_list)
    m = len(all_vars)

    F = np.zeros((k, m), dtype=float)
    F[np.arange(k), np.arange(k)] = 1.0
    A_fixed = np.zeros((m, m), dtype=float)
    S_fixed = np.zeros((m, m), dtype=float)

    A_free_entries: list[list[tuple[int, int]]] = []
    S_free_entries: list[list[tuple[int, int]]] = []
    is_free_var_list: list[bool] = []
    label_to_param_idx: dict[str, int] = {}
    terms: list[ModelTermSpec] = []

    for lhs_v, op, rhs_v, mod_lbl, fixed_val in normalized_entries:
        u_lhs = var_to_idx[lhs_v]
        u_rhs = var_to_idx[rhs_v]
        is_var = op == "~~" and lhs_v == rhs_v

        if op == "=~":
            # Factor lhs_v -> indicator rhs_v => directed edge A[rhs_v, lhs_v]
            target_matrix = "A"
            cell = (u_rhs, u_lhs)
        elif op == "~":
            # Regression lhs_v ~ rhs_v => directed edge A[lhs_v, rhs_v]
            target_matrix = "A"
            cell = (u_lhs, u_rhs)
        else:
            target_matrix = "S"
            cell = (u_lhs, u_rhs)

        if fixed_val is not None:
            if target_matrix == "A":
                A_fixed[cell[0], cell[1]] = fixed_val
            else:
                S_fixed[cell[0], cell[1]] = fixed_val
                S_fixed[cell[1], cell[0]] = fixed_val
            terms.append(
                ModelTermSpec(
                    lhs=lhs_v,
                    op=op,
                    rhs=rhs_v,
                    free=False,
                    fixed_val=fixed_val,
                    param_idx=-1,
                    label="",
                    is_variance=is_var,
                )
            )
        else:
            eff_label = "" if (mod_lbl is None or mod_lbl == "NA") else mod_lbl
            if eff_label and eff_label in label_to_param_idx:
                p_idx = label_to_param_idx[eff_label]
                is_free_var_list[p_idx] = is_free_var_list[p_idx] and is_var
            else:
                p_idx = len(A_free_entries)
                A_free_entries.append([])
                S_free_entries.append([])
                is_free_var_list.append(is_var)
                if eff_label:
                    label_to_param_idx[eff_label] = p_idx

            if target_matrix == "A":
                A_free_entries[p_idx].append(cell)
            else:
                S_free_entries[p_idx].append(cell)

            terms.append(
                ModelTermSpec(
                    lhs=lhs_v,
                    op=op,
                    rhs=rhs_v,
                    free=True,
                    fixed_val=float("nan"),
                    param_idx=p_idx,
                    label=eff_label,
                    is_variance=is_var,
                )
            )

    return RAMSpec(
        observed_traits=obs_list,
        latent_vars=latent_order,
        all_vars=all_vars,
        F=F,
        A_fixed=A_fixed,
        S_fixed=S_fixed,
        A_free_entries=A_free_entries,
        S_free_entries=S_free_entries,
        is_free_variance=np.array(is_free_var_list, dtype=bool),
        terms=terms,
        n_free=len(A_free_entries),
        std_lv=std_lv,
        settings=settings,
    )
