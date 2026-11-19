"""
Core DISCOVER models and statistical tests (Canisius, Martens & Wessels, 2016).

Reference
---------
Canisius, S., Martens, J. W. M., & Wessels, L. F. A. (2016).
A novel independence test for somatic alterations in cancer shows that biology
drives mutual exclusivity but chance explains most co-occurrence.
Genome Biology, 17, 261. https://doi.org/10.1186/s13059-016-1114-x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special

from .poisson_binomial import (
    _tail_from_pmf,
    bh_fdr,
    dbh_fdr,
    discover_fdr,
    poisson_binomial_pmf,
    poisson_binomial_pval,
)

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class DiscoverMatrix:
    """
    Fitted DISCOVER somatic alteration matrix and tumor-specific background rates.

    Attributes
    ----------
    events : np.ndarray of shape (M, N)
        Binary alteration matrix in {0.0, 1.0, NaN} (genes x tumors).
    bg_probs : np.ndarray of shape (M, N)
        Maximum-entropy background probabilities P_{ij} = sigma(mu_i + lambda_j)
        on observed entries and NaN on unprofiled (masked) entries.
    mask : np.ndarray of shape (M, N) of bool
        True where gene i was profiled in tumor j (`~np.isnan(events)`).
    gene_names : list[str]
        Gene identifiers of length M.
    sample_names : list[str]
        Tumor/sample identifiers of length N.
    strata : np.ndarray | None
        Optional stratum label per tumor of length N.
    mu : np.ndarray
        Fitted gene-level logit intercepts (shape (M,) or (M, n_strata)).
    lam : np.ndarray
        Fitted tumor-level mutation-burden logit parameters of shape (N,).
    converged : bool
        Whether optimization achieved target marginal tolerance across all strata.
    max_marginal_error : float
        Maximum absolute discrepancy between observed and expected row/column sums.
    """

    events: np.ndarray
    bg_probs: np.ndarray
    mask: np.ndarray
    gene_names: list[str]
    sample_names: list[str]
    strata: np.ndarray | None
    mu: np.ndarray
    lam: np.ndarray
    converged: bool
    max_marginal_error: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.events.shape

    @property
    def fully_observed(self) -> bool:
        return bool(np.all(self.mask))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the M x N background probability matrix as a DataFrame."""
        return pd.DataFrame(
            self.bg_probs,
            index=self.gene_names,
            columns=self.sample_names,
        )

    def events_dataframe(self) -> pd.DataFrame:
        """Return the M x N somatic alteration matrix as a DataFrame."""
        return pd.DataFrame(
            self.events,
            index=self.gene_names,
            columns=self.sample_names,
        )

    def subset(
        self,
        genes: Sequence[str | int] | None = None,
        samples: Sequence[str | int] | None = None,
    ) -> DiscoverMatrix:
        """
        Subset genes and/or tumors while preserving pre-fitted genome-wide
        background probabilities (recommended by Canisius et al., 2016).
        """
        row_idx = _resolve_indices(genes, self.gene_names, axis_name="gene")
        col_idx = _resolve_indices(samples, self.sample_names, axis_name="sample")
        ix = np.ix_(row_idx, col_idx)
        new_mu = self.mu[row_idx] if self.mu.ndim == 1 else self.mu[row_idx, :]
        new_strata = None if self.strata is None else self.strata[col_idx]
        return DiscoverMatrix(
            events=self.events[ix].copy(),
            bg_probs=self.bg_probs[ix].copy(),
            mask=self.mask[ix].copy(),
            gene_names=[self.gene_names[i] for i in row_idx],
            sample_names=[self.sample_names[j] for j in col_idx],
            strata=new_strata,
            mu=new_mu.copy(),
            lam=self.lam[col_idx].copy(),
            converged=self.converged,
            max_marginal_error=self.max_marginal_error,
        )


@dataclass(frozen=True)
class PairwiseDiscoverResult:
    """Pairwise DISCOVER mutual exclusivity or co-occurrence test results."""

    gene1: list[str]
    gene2: list[str]
    n_valid_tumors: np.ndarray
    n_mut_gene1: np.ndarray
    n_mut_gene2: np.ndarray
    obs_co_mutations: np.ndarray
    exp_co_mutations: np.ndarray
    p_value: np.ndarray
    p_min: np.ndarray
    q_value: np.ndarray
    alternative: str
    mid_p: bool
    fdr_method: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gene1": self.gene1,
                "gene2": self.gene2,
                "n_valid_tumors": self.n_valid_tumors,
                "n_mut_gene1": self.n_mut_gene1,
                "n_mut_gene2": self.n_mut_gene2,
                "obs_co_mutations": self.obs_co_mutations,
                "exp_co_mutations": self.exp_co_mutations,
                "p_value": self.p_value,
                "p_min": self.p_min,
                "q_value": self.q_value,
            }
        )

    def significant(self, alpha: float = 0.05) -> pd.DataFrame:
        df = self.to_dataframe()
        return df.loc[df["q_value"] <= alpha].sort_values("p_value").reset_index(drop=True)


@dataclass(frozen=True)
class GroupwiseDiscoverResult:
    """Groupwise DISCOVER pathway test results ('impurity', 'coverage', 'exclusivity')."""

    group_names: list[str]
    genes: list[tuple[str, ...]]
    n_genes: np.ndarray
    n_valid_tumors: np.ndarray
    method: str
    direction: str
    obs_stat: np.ndarray
    exp_stat: np.ndarray
    p_value: np.ndarray
    p_min: np.ndarray
    q_value: np.ndarray
    mid_p: bool
    fdr_method: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": self.group_names,
                "genes": [",".join(g) for g in self.genes],
                "n_genes": self.n_genes,
                "n_valid_tumors": self.n_valid_tumors,
                "method": self.method,
                "direction": self.direction,
                "obs_stat": self.obs_stat,
                "exp_stat": self.exp_stat,
                "p_value": self.p_value,
                "p_min": self.p_min,
                "q_value": self.q_value,
            }
        )


def _resolve_indices(
    selection: Sequence[str | int] | None,
    names: list[str],
    *,
    axis_name: str,
) -> np.ndarray:
    if selection is None:
        return np.arange(len(names), dtype=np.int64)
    name_to_idx = {name: i for i, name in enumerate(names)}
    out: list[int] = []
    for item in selection:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idx = int(item)
            if idx < 0 or idx >= len(names):
                raise IndexError(f"{axis_name} index {idx} out of range [0, {len(names)}).")
            out.append(idx)
        else:
            key = str(item)
            if key not in name_to_idx:
                raise KeyError(f"Unknown {axis_name} '{key}'.")
            out.append(name_to_idx[key])
    return np.asarray(out, dtype=np.int64)


def _parse_events_input(
    events: np.ndarray | pd.DataFrame,
    gene_names: Sequence[str] | None,
    sample_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if isinstance(events, pd.DataFrame):
        arr = events.to_numpy(dtype=np.float64, copy=True)
        g_names = [str(x) for x in events.index] if gene_names is None else [str(x) for x in gene_names]
        s_names = [str(x) for x in events.columns] if sample_names is None else [str(x) for x in sample_names]
    else:
        arr = np.asarray(events, dtype=np.float64).copy()
        if arr.ndim != 2:
            raise ValueError(f"events must be a 2D (genes x tumors) matrix, got {arr.ndim}D.")
        g_names = [f"G{i}" for i in range(arr.shape[0])] if gene_names is None else [str(x) for x in gene_names]
        s_names = [f"S{j}" for j in range(arr.shape[1])] if sample_names is None else [str(x) for x in sample_names]

    m, n = arr.shape
    if m == 0 or n == 0:
        raise ValueError(f"events matrix must be non-empty, got shape ({m}, {n}).")
    if len(g_names) != m:
        raise ValueError(f"len(gene_names)={len(g_names)} does not match {m} rows.")
    if len(s_names) != n:
        raise ValueError(f"len(sample_names)={len(s_names)} does not match {n} columns.")
    if len(set(g_names)) != m:
        raise ValueError("gene_names must be unique.")
    if len(set(s_names)) != n:
        raise ValueError("sample_names must be unique.")

    mask = ~np.isnan(arr)
    if not np.any(mask):
        raise ValueError("events matrix contains only NaN entries.")
    obs_vals = arr[mask]
    if not np.all((obs_vals == 0.0) | (obs_vals == 1.0)):
        raise ValueError("Observed entries of events must be binary {0, 1} (or NaN for unprofiled).")

    return arr, g_names, s_names


def _fit_single_stratum(
    a_mat: np.ndarray,
    obs_mask: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Fit P_{ij} = sigma(mu_i + lambda_j) on a single stratum with NaN support."""
    m, n = a_mat.shape
    probs = np.full((m, n), np.nan, dtype=np.float64)
    mu_full = np.zeros(m, dtype=np.float64)
    lam_full = np.zeros(n, dtype=np.float64)

    active = obs_mask.copy()
    a_zero = np.where(obs_mask, a_mat, 0.0)
    r_rem = np.sum(a_zero, axis=1)
    c_rem = np.sum(a_zero, axis=0)

    # Exact Rasch/MaxEnt boundary peeling for 0-sum or saturated rows/columns
    while True:
        changed = False
        r_cap = np.sum(active, axis=1)
        zero_r = (r_cap > 0) & (r_rem <= 0.0)
        if np.any(zero_r):
            for i in np.where(zero_r)[0]:
                cols = active[i]
                probs[i, cols] = 0.0
                mu_full[i] = -35.0
                active[i, :] = False
            changed = True

        r_cap = np.sum(active, axis=1)
        full_r = (r_cap > 0) & (r_rem >= r_cap)
        if np.any(full_r):
            for i in np.where(full_r)[0]:
                cols = active[i]
                probs[i, cols] = 1.0
                mu_full[i] = 35.0
                c_rem[cols] -= 1.0
                r_rem[i] = 0.0
                active[i, :] = False
            changed = True

        c_cap = np.sum(active, axis=0)
        zero_c = (c_cap > 0) & (c_rem <= 0.0)
        if np.any(zero_c):
            for j in np.where(zero_c)[0]:
                rows = active[:, j]
                probs[rows, j] = 0.0
                lam_full[j] = -35.0
                active[:, j] = False
            changed = True

        c_cap = np.sum(active, axis=0)
        full_c = (c_cap > 0) & (c_rem >= c_cap)
        if np.any(full_c):
            for j in np.where(full_c)[0]:
                rows = active[:, j]
                probs[rows, j] = 1.0
                lam_full[j] = 35.0
                r_rem[rows] -= 1.0
                c_rem[j] = 0.0
                active[:, j] = False
            changed = True

        if not changed:
            break

    act_rows = np.where(np.any(active, axis=1))[0]
    act_cols = np.where(np.any(active, axis=0))[0]

    if act_rows.size > 0 and act_cols.size > 0:
        ix = np.ix_(act_rows, act_cols)
        m_sub = active[ix]
        a_sub = np.where(m_sub, a_mat[ix], 0.0)
        ma, na = m_sub.shape

        r_obs = np.sum(a_sub, axis=1)
        c_obs = np.sum(a_sub, axis=0)
        r_n = np.sum(m_sub, axis=1)
        c_n = np.sum(m_sub, axis=0)

        p_global = float(np.clip(np.sum(r_obs) / np.sum(r_n), 1e-4, 1.0 - 1e-4))
        g_logit = float(special.logit(p_global))
        mu0 = special.logit(np.clip(r_obs / r_n, 1e-4, 1.0 - 1e-4)) - 0.5 * g_logit
        lam0 = special.logit(np.clip(c_obs / c_n, 1e-4, 1.0 - 1e-4)) - 0.5 * g_logit
        lam0 -= np.mean(lam0)
        mu0 += np.mean(lam0)
        x0 = np.concatenate([mu0, lam0])

        # Tiny gauge-fixing penalty 0.5 * eps * (sum_j lambda_j)^2 has identically
        # zero gradient at optimum because sum_i g_mu_i == sum_j g_lam_j == 0.
        gauge_eps = 1e-6

        def _obj_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            mu_v = x[:ma]
            lam_v = x[ma:]
            eta = mu_v[:, None] + lam_v[None, :]
            # Numerically stable softplus: log(1 + exp(eta)) = logaddexp(0, eta)
            loss_mat = np.logaddexp(0.0, eta) - a_sub * eta
            nll = float(np.sum(np.where(m_sub, loss_mat, 0.0)))
            lam_sum = float(np.sum(lam_v))
            nll += 0.5 * gauge_eps * (lam_sum ** 2)

            p_mat = special.expit(eta)
            diff = np.where(m_sub, p_mat - a_sub, 0.0)
            g_mu = np.sum(diff, axis=1)
            g_lam = np.sum(diff, axis=0) + gauge_eps * lam_sum
            return nll, np.concatenate([g_mu, g_lam])

        bounds = [(-35.0, 35.0)] * (ma + na)
        opt = optimize.minimize(
            _obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": min(tol * 0.1, 1e-10), "maxiter": max_iter},
        )
        mu_v = opt.x[:ma].copy()
        lam_v = opt.x[ma:].copy()

        # Exact unpenalized coordinate-block Newton polishing (handles multiple
        # disconnected observation blocks without any regularization bias)
        for _ in range(300):
            eta = np.clip(mu_v[:, None] + lam_v[None, :], -40.0, 40.0)
            p_mat = special.expit(eta)
            diff = np.where(m_sub, p_mat - a_sub, 0.0)
            err = max(
                float(np.max(np.abs(np.sum(diff, axis=1)))),
                float(np.max(np.abs(np.sum(diff, axis=0)))),
            )
            if err <= tol:
                break

            w_mat = np.where(m_sub, p_mat * (1.0 - p_mat), 0.0)
            h_mu = np.sum(w_mat, axis=1)
            step_mu = np.where(h_mu > 1e-15, np.sum(diff, axis=1) / np.maximum(h_mu, 1e-15), 0.0)
            mu_v = np.clip(mu_v - np.clip(step_mu, -2.0, 2.0), -35.0, 35.0)

            eta = np.clip(mu_v[:, None] + lam_v[None, :], -40.0, 40.0)
            p_mat = special.expit(eta)
            diff = np.where(m_sub, p_mat - a_sub, 0.0)
            w_mat = np.where(m_sub, p_mat * (1.0 - p_mat), 0.0)
            h_lam = np.sum(w_mat, axis=0)
            step_lam = np.where(h_lam > 1e-15, np.sum(diff, axis=0) / np.maximum(h_lam, 1e-15), 0.0)
            lam_v = np.clip(lam_v - np.clip(step_lam, -2.0, 2.0), -35.0, 35.0)

            # Keep centered gauge
            shift = float(np.mean(lam_v))
            lam_v -= shift
            mu_v += shift

        p_sub = special.expit(np.clip(mu_v[:, None] + lam_v[None, :], -40.0, 40.0))
        probs[ix] = np.where(m_sub, p_sub, probs[ix])
        mu_full[act_rows] = mu_v
        lam_full[act_cols] = lam_v

    # Compute exact marginal error over all observed entries in stratum
    obs_row_sums = np.sum(np.where(obs_mask, a_mat, 0.0), axis=1)
    exp_row_sums = np.sum(np.where(obs_mask, probs, 0.0), axis=1)
    obs_col_sums = np.sum(np.where(obs_mask, a_mat, 0.0), axis=0)
    exp_col_sums = np.sum(np.where(obs_mask, probs, 0.0), axis=0)

    max_err = max(
        float(np.max(np.abs(exp_row_sums - obs_row_sums))),
        float(np.max(np.abs(exp_col_sums - obs_col_sums))),
    )
    converged = max_err <= max(tol * 10.0, 1e-6)
    return probs, mu_full, lam_full, converged, max_err


def fit_discover_background(
    events: np.ndarray | pd.DataFrame,
    *,
    strata: Sequence[str | int] | np.ndarray | pd.Series | None = None,
    gene_names: Sequence[str] | None = None,
    sample_names: Sequence[str] | None = None,
    tol: float = 1e-9,
    max_iter: int = 2000,
) -> DiscoverMatrix:
    """
    Estimate the DISCOVER maximum-entropy background alteration probability matrix.

    Fits Pr(A_{ij} = 1) = sigma(mu_i + lambda_j) on observed panel entries so that
    expected row sums equal observed gene alteration counts and expected column sums
    equal tumor-specific mutation burdens (within each stratum if `strata` is given).
    """
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    arr, g_names, s_names = _parse_events_input(events, gene_names, sample_names)
    mask = ~np.isnan(arr)
    m, n = arr.shape

    if strata is None:
        bg_probs, mu_vec, lam_vec, converged, max_err = _fit_single_stratum(
            arr, mask, tol=tol, max_iter=max_iter
        )
        return DiscoverMatrix(
            events=arr,
            bg_probs=bg_probs,
            mask=mask,
            gene_names=g_names,
            sample_names=s_names,
            strata=None,
            mu=mu_vec,
            lam=lam_vec,
            converged=converged,
            max_marginal_error=max_err,
        )

    strata_arr = np.asarray(strata)
    if strata_arr.ndim != 1 or strata_arr.size != n:
        raise ValueError(f"strata must be a 1D array of length {n}, got shape {strata_arr.shape}.")
    if pd.isna(strata_arr).any():
        raise ValueError("strata must not contain missing/NaN labels.")

    unique_strata = list(dict.fromkeys(strata_arr.tolist()))
    bg_probs = np.full((m, n), np.nan, dtype=np.float64)
    mu_mat = np.zeros((m, len(unique_strata)), dtype=np.float64)
    lam_vec = np.zeros(n, dtype=np.float64)
    all_converged = True
    max_err = 0.0

    for s_idx, s_label in enumerate(unique_strata):
        cols = np.where(strata_arr == s_label)[0]
        sub_mask = mask[:, cols]
        if not np.any(sub_mask):
            continue
        p_sub, mu_sub, lam_sub, conv_sub, err_sub = _fit_single_stratum(
            arr[:, cols], sub_mask, tol=tol, max_iter=max_iter
        )
        bg_probs[:, cols] = p_sub
        mu_mat[:, s_idx] = mu_sub
        lam_vec[cols] = lam_sub
        all_converged = all_converged and conv_sub
        max_err = max(max_err, err_sub)

    return DiscoverMatrix(
        events=arr,
        bg_probs=bg_probs,
        mask=mask,
        gene_names=g_names,
        sample_names=s_names,
        strata=strata_arr.copy(),
        mu=mu_mat,
        lam=lam_vec,
        converged=all_converged,
        max_marginal_error=max_err,
    )


def pairwise_discover_test(
    data: DiscoverMatrix | np.ndarray | pd.DataFrame,
    *,
    g1: Sequence[str | int] | None = None,
    g2: Sequence[str | int] | None = None,
    alternative: Literal["exclusivity", "co-occurrence", "co_occurrence", "less", "greater"] = "exclusivity",
    mid_p: bool = True,
    fdr_method: Literal["discover_fdr", "bh"] = "discover_fdr",
    min_mutations: int = 1,
    strata: Sequence[str | int] | np.ndarray | pd.Series | None = None,
    settings: SettingsMode = "default",
) -> PairwiseDiscoverResult:
    """
    Pairwise DISCOVER independence test for mutual exclusivity or co-occurrence.

    Parameters
    ----------
    data : DiscoverMatrix | np.ndarray | pd.DataFrame
        Pre-fitted `DiscoverMatrix` or binary alteration matrix (M genes x N tumors).
    g1, g2 : optional gene subsets
        Gene names or row indices to test. When omitted, all unordered pairs (i < j)
        meeting `min_mutations` are tested.
    alternative : {"exclusivity", "co-occurrence", "less", "greater"}
        "exclusivity" / "less" tests lower tail (fewer co-mutations than expected);
        "co-occurrence" / "greater" tests upper tail (more co-mutations than expected).
    mid_p : bool, default=True
        Apply Lancaster (1961) mid-p correction (default in Canisius et al., 2016;
        forced to `False` when `settings="R_default"`).
    fdr_method : {"discover_fdr", "bh"}, default="discover_fdr"
        Multiple-testing correction method.
    min_mutations : int, default=1
        Minimum observed alterations required for each gene in a tested pair.
    strata : optional 1D sequence
        Tumor stratum labels (used only if `data` is not already a `DiscoverMatrix`).
    settings : {"default", "R_default"}, default="default"
        `"default"` uses Lancaster mid-p and minimum-p adaptive discrete FDR;
        `"R_default"` forces `mid_p=False` and computes exact `DBH` discrete
        Benjamini-Hochberg FDR matching `NKI-CCB/DISCOVER` (`fdr.f90`).
    """
    _validate_settings(settings)
    eff_mid_p = False if settings == "R_default" else bool(mid_p)
    alt_map = {
        "exclusivity": "less",
        "less": "less",
        "co-occurrence": "greater",
        "co_occurrence": "greater",
        "greater": "greater",
    }
    if alternative not in alt_map:
        raise ValueError(
            f"Invalid alternative '{alternative}'. Choose 'exclusivity'/'less' "
            "or 'co-occurrence'/'greater'."
        )
    tail = alt_map[alternative]
    canonical_alt = "exclusivity" if tail == "less" else "co-occurrence"

    if fdr_method not in ("discover_fdr", "bh"):
        raise ValueError(f"Invalid fdr_method '{fdr_method}'. Expected 'discover_fdr' or 'bh'.")
    if min_mutations < 0:
        raise ValueError("min_mutations must be >= 0.")

    bg = (
        data
        if isinstance(data, DiscoverMatrix)
        else fit_discover_background(data, strata=strata)
    )

    idx1 = _resolve_indices(g1, bg.gene_names, axis_name="gene")
    idx2 = _resolve_indices(g2, bg.gene_names, axis_name="gene")

    total_gene_muts = np.sum(np.where(bg.mask, bg.events, 0.0), axis=1).astype(np.int64)

    # Enumerate unique unordered gene pairs
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if g1 is None and g2 is None:
        eligible_idx = [int(i) for i in idx1 if total_gene_muts[i] >= min_mutations]
        for a_pos in range(len(eligible_idx)):
            i = eligible_idx[a_pos]
            for b_pos in range(a_pos + 1, len(eligible_idx)):
                j = eligible_idx[b_pos]
                pairs.append((i, j))
    else:
        for i_raw in idx1:
            i = int(i_raw)
            if total_gene_muts[i] < min_mutations:
                continue
            for j_raw in idx2:
                j = int(j_raw)
                if i == j or total_gene_muts[j] < min_mutations:
                    continue
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    pairs.append((i, j))

    n_pairs = len(pairs)
    if n_pairs == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return PairwiseDiscoverResult(
            gene1=[],
            gene2=[],
            n_valid_tumors=empty_i,
            n_mut_gene1=empty_i,
            n_mut_gene2=empty_i,
            obs_co_mutations=empty_i,
            exp_co_mutations=empty_f,
            p_value=empty_f,
            p_min=empty_f,
            q_value=empty_f,
            alternative=canonical_alt,
            mid_p=eff_mid_p,
            fdr_method=fdr_method,
        )

    # Vectorized BLAS summary matrices across observed masks
    o_f = bg.mask.astype(np.float64)
    a0 = np.where(bg.mask, bg.events, 0.0)
    p0 = np.where(bg.mask, bg.bg_probs, 0.0)

    n_valid_mat = np.rint(o_f @ o_f.T).astype(np.int64)
    n_mut_on_valid = np.rint(a0 @ o_f.T).astype(np.int64)
    obs_co_mat = np.rint(a0 @ a0.T).astype(np.int64)
    exp_co_mat = p0 @ p0.T

    p_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    q_arr = np.array([p[1] for p in pairs], dtype=np.int64)

    n_valid = n_valid_mat[p_arr, q_arr]
    n_mut1 = n_mut_on_valid[p_arr, q_arr]
    n_mut2 = n_mut_on_valid[q_arr, p_arr]
    obs_co = obs_co_mat[p_arr, q_arr]
    exp_co = exp_co_mat[p_arr, q_arr]

    p_vals = np.ones(n_pairs, dtype=np.float64)
    p_mins = np.ones(n_pairs, dtype=np.float64)
    dbh_supports: list[np.ndarray] = []
    dbh_pmfs: list[np.ndarray] = []
    use_dbh = settings == "R_default" and fdr_method == "discover_fdr"

    fully_obs = bg.fully_observed
    bg_p = bg.bg_probs
    bg_m = bg.mask

    for idx in range(n_pairs):
        i = int(p_arr[idx])
        j = int(q_arr[idx])
        if int(n_valid[idx]) == 0:
            p_vals[idx] = 1.0
            p_mins[idx] = 1.0
            if use_dbh:
                dbh_supports.append(np.array([1.0], dtype=np.float64))
                dbh_pmfs.append(np.array([1.0], dtype=np.float64))
            continue

        if fully_obs:
            pk = bg_p[i] * bg_p[j]
        else:
            valid = bg_m[i] & bg_m[j]
            pk = bg_p[i, valid] * bg_p[j, valid]

        k_obs = int(obs_co[idx])
        if use_dbh:
            pmf = poisson_binomial_pmf(pk, method="auto")
            pv_check = _tail_from_pmf(k_obs, pmf, tail, mid_p=False)
            if pk.size > 64 and pv_check < 1e-12:
                pmf = poisson_binomial_pmf(pk, method="dp")
            if tail == "less":
                s_vec = np.clip(np.cumsum(pmf), 0.0, 1.0)
                p_vals[idx] = float(s_vec[k_obs])
                p_mins[idx] = float(s_vec[0])
                dbh_supports.append(s_vec)
                dbh_pmfs.append(pmf)
            else:
                k_max = int(min(n_mut1[idx], n_mut2[idx]))
                pmf_rev = pmf[::-1]
                s_vec = np.clip(np.cumsum(pmf_rev), 0.0, 1.0)
                p_vals[idx] = float(s_vec[pmf.size - 1 - k_obs])
                p_mins[idx] = float(s_vec[pmf.size - 1 - k_max])
                dbh_supports.append(s_vec)
                dbh_pmfs.append(pmf_rev)
        elif tail == "less":
            # Exact Pr(X = 0) in O(N) without FFT when k_obs == 0
            if np.any(pk >= 1.0):
                pr_zero = 0.0
            else:
                pr_zero = float(np.exp(np.sum(np.log1p(-pk))))
            p_min_val = (0.5 * pr_zero) if eff_mid_p else pr_zero
            p_mins[idx] = np.clip(p_min_val, 0.0, 1.0)
            if k_obs == 0:
                p_vals[idx] = p_mins[idx]
            else:
                p_vals[idx] = poisson_binomial_pval(
                    k_obs, pk, alternative="less", mid_p=eff_mid_p
                )
        else:
            k_max = int(min(n_mut1[idx], n_mut2[idx]))
            pmf = poisson_binomial_pmf(pk, method="auto")
            pv = _tail_from_pmf(k_obs, pmf, "greater", mid_p=eff_mid_p)
            pm = _tail_from_pmf(k_max, pmf, "greater", mid_p=eff_mid_p)
            if pk.size > 64 and (pv < 1e-12 or pm < 1e-12):
                pmf_dp = poisson_binomial_pmf(pk, method="dp")
                pv = _tail_from_pmf(k_obs, pmf_dp, "greater", mid_p=eff_mid_p)
                pm = _tail_from_pmf(k_max, pmf_dp, "greater", mid_p=eff_mid_p)
            p_vals[idx] = float(np.clip(pv, 0.0, 1.0))
            p_mins[idx] = float(np.clip(pm, 0.0, 1.0))

    if use_dbh:
        q_vals = dbh_fdr(p_vals, dbh_supports, dbh_pmfs)
    elif fdr_method == "discover_fdr":
        q_vals = discover_fdr(p_vals, p_mins)
    else:
        q_vals = bh_fdr(p_vals)

    return PairwiseDiscoverResult(
        gene1=[bg.gene_names[i] for i in p_arr],
        gene2=[bg.gene_names[j] for j in q_arr],
        n_valid_tumors=n_valid,
        n_mut_gene1=n_mut1,
        n_mut_gene2=n_mut2,
        obs_co_mutations=obs_co,
        exp_co_mutations=exp_co,
        p_value=p_vals,
        p_min=p_mins,
        q_value=q_vals,
        alternative=canonical_alt,
        mid_p=eff_mid_p,
        fdr_method=fdr_method,
    )


def _group_event_probs_and_obs(
    a_sub: np.ndarray,
    p_sub: np.ndarray,
    m_sub: np.ndarray,
    method: str,
) -> tuple[np.ndarray, int]:
    """
    Compute per-tumor Poisson-Binomial event probabilities and observed group
    statistic across tumors where at least 2 genes in the group are observed
    (or all genes in the group are observed when fully profiled).
    """
    g_size, _ = a_sub.shape
    if g_size < 2:
        raise ValueError(f"Each gene group must contain at least 2 genes, got {g_size}.")

    # Use tumors where all genes in the pathway are profiled; if none exist, fall back
    # to tumors where at least 2 pathway genes are profiled.
    valid_cols = np.all(m_sub, axis=0)
    if not np.any(valid_cols):
        valid_cols = np.sum(m_sub, axis=0) >= 2
    if not np.any(valid_cols):
        return np.array([], dtype=np.float64), 0

    a_v = a_sub[:, valid_cols]
    p_v = p_sub[:, valid_cols]
    m_v = m_sub[:, valid_cols]

    z_obs = np.sum(np.where(m_v, a_v, 0.0), axis=0)

    # Exact cancellation-free 3-state DP across G genes (vectorized over all valid tumors):
    # Tracks s0 = Pr(Z_k = 0), s1 = Pr(Z_k = 1), s2plus = Pr(Z_k >= 2) using pure
    # non-negative additions/multiplications so tiny P_{i,k} never suffer 1 - p0 - p1 cancellation.
    n_v = a_v.shape[1]
    s0 = np.ones(n_v, dtype=np.float64)
    s1 = np.zeros(n_v, dtype=np.float64)
    s2plus = np.zeros(n_v, dtype=np.float64)

    for g_idx in range(g_size):
        p_row = np.where(m_v[g_idx], p_v[g_idx], 0.0)
        q_row = 1.0 - p_row
        s2plus = s2plus + s1 * p_row
        s1 = s1 * q_row + s0 * p_row
        s0 = s0 * q_row

    if method == "coverage":
        pk = np.clip(s1 + s2plus, 0.0, 1.0)
        obs = int(np.sum(z_obs >= 1.0))
    elif method == "impurity":
        pk = np.clip(s2plus, 0.0, 1.0)
        obs = int(np.sum(z_obs >= 2.0))
    elif method == "exclusivity":
        pk = np.clip(s1, 0.0, 1.0)
        obs = int(np.sum(z_obs == 1.0))
    else:
        raise ValueError(
            f"Invalid groupwise method '{method}'. Expected 'impurity', 'coverage', or 'exclusivity'."
        )

    return pk, obs


def groupwise_discover_test(
    data: DiscoverMatrix | np.ndarray | pd.DataFrame,
    genes: Sequence[str | int] | None = None,
    *,
    gene_sets: Mapping[str, Sequence[str | int]] | Sequence[Sequence[str | int]] | None = None,
    method: Literal["impurity", "coverage", "exclusivity"] = "impurity",
    direction: Literal["exclusivity", "co-occurrence", "co_occurrence"] = "exclusivity",
    mid_p: bool = True,
    fdr_method: Literal["discover_fdr", "bh"] = "discover_fdr",
    strata: Sequence[str | int] | np.ndarray | pd.Series | None = None,
    settings: SettingsMode = "default",
) -> GroupwiseDiscoverResult:
    """
    Groupwise DISCOVER pathway test (Canisius, Martens & Wessels, 2016).

    Supports all three set statistics defined in Canisius et al. (2016):
    - `"impurity"`: Tumors with >1 altered gene in S (`p_k = 1 - p_{0,k} - p_{1,k}`).
      Lower than expected under mutual exclusivity; higher under co-occurrence.
    - `"coverage"`: Tumors with >=1 altered gene in S (`p_k = 1 - prod_{i in S}(1 - P_{i,k})`).
      Higher than expected under mutual exclusivity; lower under co-occurrence.
    - `"exclusivity"`: Tumors with ==1 altered gene in S (`p_k = sum_i P_{i,k} prod_{j!=i}(1 - P_{j,k})`).
      Higher than expected under mutual exclusivity; lower under co-occurrence.
    """
    _validate_settings(settings)
    eff_mid_p = False if settings == "R_default" else bool(mid_p)
    if method not in ("impurity", "coverage", "exclusivity"):
        raise ValueError(
            f"Invalid method '{method}'. Expected 'impurity', 'coverage', or 'exclusivity'."
        )
    if direction not in ("exclusivity", "co-occurrence", "co_occurrence"):
        raise ValueError(
            f"Invalid direction '{direction}'. Expected 'exclusivity' or 'co-occurrence'."
        )
    if fdr_method not in ("discover_fdr", "bh"):
        raise ValueError(f"Invalid fdr_method '{fdr_method}'. Expected 'discover_fdr' or 'bh'.")
    if genes is not None and gene_sets is not None:
        raise ValueError("Specify either `genes` (single pathway) or `gene_sets`, not both.")

    canonical_dir = "exclusivity" if direction == "exclusivity" else "co-occurrence"

    bg = (
        data
        if isinstance(data, DiscoverMatrix)
        else fit_discover_background(data, strata=strata)
    )

    groups_dict: dict[str, list[int]] = {}
    if gene_sets is not None:
        if isinstance(gene_sets, Mapping):
            for name, g_list in gene_sets.items():
                idx = _resolve_indices(g_list, bg.gene_names, axis_name="gene")
                uniq = list(dict.fromkeys(int(i) for i in idx))
                if len(uniq) < 2:
                    raise ValueError(f"Gene set '{name}' must have at least 2 distinct genes.")
                groups_dict[str(name)] = uniq
        else:
            for g_pos, g_list in enumerate(gene_sets):
                idx = _resolve_indices(g_list, bg.gene_names, axis_name="gene")
                uniq = list(dict.fromkeys(int(i) for i in idx))
                if len(uniq) < 2:
                    raise ValueError(f"Gene set {g_pos} must have at least 2 distinct genes.")
                groups_dict[f"Group_{g_pos}"] = uniq
    else:
        idx = _resolve_indices(genes, bg.gene_names, axis_name="gene")
        uniq = list(dict.fromkeys(int(i) for i in idx))
        if len(uniq) < 2:
            raise ValueError("Groupwise test requires at least 2 distinct genes.")
        groups_dict["Group_0"] = uniq

    # Tail orientation:
    # Under mutual exclusivity: impurity is lower ("less"), coverage & exclusivity are higher ("greater").
    # Under co-occurrence: impurity is higher ("greater"), coverage & exclusivity are lower ("less").
    if canonical_dir == "exclusivity":
        tail = "less" if method == "impurity" else "greater"
    else:
        tail = "greater" if method == "impurity" else "less"

    g_names_list = list(groups_dict.keys())
    n_groups = len(g_names_list)
    genes_tuples: list[tuple[str, ...]] = []
    n_genes_arr = np.zeros(n_groups, dtype=np.int64)
    n_valid_arr = np.zeros(n_groups, dtype=np.int64)
    obs_arr = np.zeros(n_groups, dtype=np.int64)
    exp_arr = np.zeros(n_groups, dtype=np.float64)
    p_vals = np.ones(n_groups, dtype=np.float64)
    p_mins = np.ones(n_groups, dtype=np.float64)
    dbh_supports: list[np.ndarray] = []
    dbh_pmfs: list[np.ndarray] = []
    use_dbh = settings == "R_default" and fdr_method == "discover_fdr"

    for g_i, g_label in enumerate(g_names_list):
        r_idx = np.asarray(groups_dict[g_label], dtype=np.int64)
        genes_tuples.append(tuple(bg.gene_names[i] for i in r_idx))
        n_genes_arr[g_i] = r_idx.size

        pk, obs_val = _group_event_probs_and_obs(
            bg.events[r_idx], bg.bg_probs[r_idx], bg.mask[r_idx], method
        )
        n_valid_arr[g_i] = pk.size
        obs_arr[g_i] = obs_val
        exp_arr[g_i] = float(np.sum(pk))

        if pk.size == 0:
            p_vals[g_i] = 1.0
            p_mins[g_i] = 1.0
            if use_dbh:
                dbh_supports.append(np.array([1.0], dtype=np.float64))
                dbh_pmfs.append(np.array([1.0], dtype=np.float64))
            continue

        total_alts = int(
            np.rint(np.sum(np.where(bg.mask[r_idx], bg.events[r_idx], 0.0)))
        )
        if method in ("coverage", "exclusivity"):
            k_ext = min(pk.size, total_alts)
        else:
            k_ext = min(pk.size, total_alts // 2)

        if use_dbh:
            pmf = poisson_binomial_pmf(pk, method="auto")
            pv_check = _tail_from_pmf(obs_val, pmf, tail, mid_p=False)
            if pk.size > 64 and pv_check < 1e-12:
                pmf = poisson_binomial_pmf(pk, method="dp")
            if tail == "less":
                s_vec = np.clip(np.cumsum(pmf), 0.0, 1.0)
                pv = float(s_vec[obs_val])
                pm = float(s_vec[0])
                dbh_supports.append(s_vec)
                dbh_pmfs.append(pmf)
            else:
                pmf_rev = pmf[::-1]
                s_vec = np.clip(np.cumsum(pmf_rev), 0.0, 1.0)
                pv = float(s_vec[pmf.size - 1 - obs_val])
                pm = float(s_vec[pmf.size - 1 - k_ext])
                dbh_supports.append(s_vec)
                dbh_pmfs.append(pmf_rev)
        elif tail == "less":
            if np.any(pk >= 1.0):
                pr_zero = 0.0
            else:
                pr_zero = float(np.exp(np.sum(np.log1p(-pk))))
            pm = (0.5 * pr_zero) if eff_mid_p else pr_zero
            if obs_val == 0:
                pv = pm
            else:
                pv = poisson_binomial_pval(obs_val, pk, alternative="less", mid_p=eff_mid_p)
        else:
            pmf = poisson_binomial_pmf(pk, method="auto")
            pv = _tail_from_pmf(obs_val, pmf, tail, mid_p=eff_mid_p)
            pm = _tail_from_pmf(k_ext, pmf, tail, mid_p=eff_mid_p)
            if pk.size > 64 and (pv < 1e-12 or pm < 1e-12):
                pmf_dp = poisson_binomial_pmf(pk, method="dp")
                pv = _tail_from_pmf(obs_val, pmf_dp, tail, mid_p=eff_mid_p)
                pm = _tail_from_pmf(k_ext, pmf_dp, tail, mid_p=eff_mid_p)

        p_vals[g_i] = float(np.clip(pv, 0.0, 1.0))
        p_mins[g_i] = float(np.clip(pm, 0.0, 1.0))

    if use_dbh:
        q_vals = dbh_fdr(p_vals, dbh_supports, dbh_pmfs)
    elif fdr_method == "discover_fdr":
        q_vals = discover_fdr(p_vals, p_mins)
    else:
        q_vals = bh_fdr(p_vals)

    return GroupwiseDiscoverResult(
        group_names=g_names_list,
        genes=genes_tuples,
        n_genes=n_genes_arr,
        n_valid_tumors=n_valid_arr,
        method=method,
        direction=canonical_dir,
        obs_stat=obs_arr,
        exp_stat=exp_arr,
        p_value=p_vals,
        p_min=p_mins,
        q_value=q_vals,
        mid_p=eff_mid_p,
        fdr_method=fdr_method,
    )
