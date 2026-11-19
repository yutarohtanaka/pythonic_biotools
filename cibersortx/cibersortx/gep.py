"""
Group-Mode adaptive noise-filtered NNLS GEP estimation (`impute_gep_group`)
and High-Resolution Mode sliding-window sample-specific purification
(`impute_gep_highres`) (Newman et al., Nature Biotechnology 2019).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from ._io import ExpressionInput, bh_fdr, parse_expression_matrix


@dataclass(frozen=True)
class GroupGEPResult:
    """Typed result container for `impute_gep_group`."""

    gep_raw: np.ndarray
    gep_filtered: np.ndarray
    se: np.ndarray
    cv: np.ndarray
    z_score: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    passed_filter: np.ndarray
    r_squared: np.ndarray
    condition_number: float
    vif: np.ndarray
    gene_names: list[str]
    cell_types: list[str]

    def gep_dataframe(self, filtered: bool = True) -> pd.DataFrame:
        """Return `(G, K)` cell-type expression matrix (`genes x cell_types`)."""
        mat = self.gep_filtered if filtered else self.gep_raw
        return pd.DataFrame(mat, index=self.gene_names, columns=self.cell_types)

    def to_dataframe(self, filtered: bool = False) -> pd.DataFrame:
        """
        Return tidy long-format DataFrame (`gene`, `cell_type`, `expression`,
        `se`, `cv`, `z_score`, `p_value`, `q_value`, `passed_filter`).
        """
        n_genes, n_types = self.gep_raw.shape
        genes_col = np.repeat(self.gene_names, n_types)
        ct_col = np.tile(self.cell_types, n_genes)
        expr_mat = self.gep_filtered if filtered else self.gep_raw
        df = pd.DataFrame(
            {
                "gene": genes_col,
                "cell_type": ct_col,
                "expression": expr_mat.ravel(),
                "se": self.se.ravel(),
                "cv": self.cv.ravel(),
                "z_score": self.z_score.ravel(),
                "p_value": self.p_value.ravel(),
                "q_value": self.q_value.ravel(),
                "passed_filter": self.passed_filter.ravel(),
            }
        )
        if filtered:
            df = df.loc[df["passed_filter"]].reset_index(drop=True)
        return df

    def summary(self) -> str:
        n_pass = int(np.sum(self.passed_filter))
        total = int(self.passed_filter.size)
        return (
            f"GroupGEPResult(genes={len(self.gene_names)}, cell_types={len(self.cell_types)}, "
            f"cond={self.condition_number:.2f}, passed={n_pass}/{total})"
        )


@dataclass(frozen=True)
class HighResGEPResult:
    """Typed result container for `impute_gep_highres`."""

    gep_raw: np.ndarray
    gep_filtered: np.ndarray
    group_gep: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    is_variable: np.ndarray
    window_size: int
    gene_names: list[str]
    sample_names: list[str]
    cell_types: list[str]

    def gep_dataframe(self, cell_type: str, filtered: bool = False) -> pd.DataFrame:
        """Return `(G, N)` purified expression matrix for `cell_type`."""
        if cell_type not in self.cell_types:
            raise KeyError(
                f"Unknown cell_type '{cell_type}'. Available: {self.cell_types}."
            )
        k = self.cell_types.index(cell_type)
        mat = self.gep_filtered[:, :, k] if filtered else self.gep_raw[:, :, k]
        return pd.DataFrame(mat, index=self.gene_names, columns=self.sample_names)

    def to_dataframe(self, only_variable: bool = False) -> pd.DataFrame:
        """
        Return tidy summary DataFrame across `(gene, cell_type)` pairs with
        `mean_expr`, `sd_expr`, `group_expr`, `p_value`, `q_value`, `is_variable`.
        """
        n_genes, _n_samples, n_types = self.gep_raw.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_expr = np.nanmean(self.gep_raw, axis=1)
            sd_expr = np.nanstd(self.gep_raw, axis=1, ddof=0)

        df = pd.DataFrame(
            {
                "gene": np.repeat(self.gene_names, n_types),
                "cell_type": np.tile(self.cell_types, n_genes),
                "mean_expr": mean_expr.ravel(),
                "sd_expr": sd_expr.ravel(),
                "group_expr": self.group_gep.ravel(),
                "p_value": self.p_value.ravel(),
                "q_value": self.q_value.ravel(),
                "is_variable": self.is_variable.ravel(),
            }
        )
        if only_variable:
            df = df.loc[df["is_variable"]].reset_index(drop=True)
        return df

    def summary(self) -> str:
        n_var = int(np.sum(self.is_variable))
        total = int(self.is_variable.size)
        return (
            f"HighResGEPResult(genes={len(self.gene_names)}, samples={len(self.sample_names)}, "
            f"cell_types={len(self.cell_types)}, window_size={self.window_size}, "
            f"variable={n_var}/{total})"
        )


def _parse_fractions_and_align(
    mixture: ExpressionInput,
    fractions: Any,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    is_log2: bool = False,
    log2_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """
    Parse `mixture` `(G, N)` and `fractions` `(N, K)`, aligning samples by name
    when both expose sample identifiers.
    """
    M_raw, gene_names, mix_samples = parse_expression_matrix(
        mixture,
        layer=layer,
        samples_as_rows=samples_as_rows,
        default_row_prefix="gene",
        default_col_prefix="sample",
        allow_1d=False,
        name="mixture",
        is_log2=is_log2,
        log2_offset=log2_offset,
    )

    has_frac_names = False
    if hasattr(fractions, "fractions") and hasattr(fractions, "sample_names") and hasattr(fractions, "cell_types"):
        F_raw = np.asarray(fractions.fractions, dtype=np.float64)
        frac_samples = [str(s) for s in fractions.sample_names]
        cell_types = [str(ct) for ct in fractions.cell_types]
        has_frac_names = True
    elif isinstance(fractions, pd.DataFrame):
        F_raw = fractions.to_numpy(dtype=np.float64)
        frac_samples = [str(s) for s in fractions.index]
        cell_types = [str(ct) for ct in fractions.columns]
        has_frac_names = True
    else:
        F_raw = np.asarray(fractions, dtype=np.float64)
        if F_raw.ndim != 2:
            raise ValueError(f"`fractions` must be 2D (samples x cell_types), got shape {F_raw.shape}.")
        frac_samples = mix_samples
        cell_types = [f"cell_type_{k}" for k in range(F_raw.shape[1])]

    if F_raw.ndim != 2 or F_raw.shape[0] == 0 or F_raw.shape[1] == 0:
        raise ValueError(f"`fractions` must be non-empty 2D, got shape {F_raw.shape}.")
    if not np.all(np.isfinite(F_raw)):
        raise ValueError("`fractions` contains NaN or Inf values.")
    if np.min(F_raw) < 0.0:
        raise ValueError("`fractions` contains negative values.")
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("Duplicate cell-type names in `fractions`.")
    if len(set(frac_samples)) != len(frac_samples):
        raise ValueError("Duplicate sample names in `fractions`.")

    has_mix_names = isinstance(mixture, (pd.DataFrame, )) or hasattr(mixture, "obs_names")
    if has_frac_names and has_mix_names:
        mix_idx = pd.Index(mix_samples)
        frac_idx = pd.Index(frac_samples)
        common_samples = frac_idx.intersection(mix_idx, sort=False)
        if len(common_samples) == 0:
            raise ValueError("`mixture` and `fractions` share zero sample names.")
        pos_f = frac_idx.get_indexer(common_samples)
        pos_m = mix_idx.get_indexer(common_samples)
        F_aligned = np.ascontiguousarray(F_raw[pos_f, :], dtype=np.float64)
        M_aligned = np.ascontiguousarray(M_raw[:, pos_m], dtype=np.float64)
        sample_names = [str(s) for s in common_samples]
    else:
        if F_raw.shape[0] != M_raw.shape[1]:
            raise ValueError(
                f"Sample count mismatch: mixture has {M_raw.shape[1]} samples, "
                f"fractions has {F_raw.shape[0]} rows."
            )
        F_aligned = np.ascontiguousarray(F_raw, dtype=np.float64)
        M_aligned = np.ascontiguousarray(M_raw, dtype=np.float64)
        sample_names = frac_samples if has_frac_names else mix_samples

    return M_aligned, F_aligned, gene_names, sample_names, cell_types


def _solve_batch_qr_nnls(
    F_mat: np.ndarray,
    M_mat: np.ndarray,
    *,
    ridge_alpha: float = 0.0,
) -> np.ndarray:
    """
    Solve `min_{x >= 0} ||F_mat @ x - m_g||_2^2 + ridge_alpha * ||x||_2^2`
    across all genes `g` using thin QR reduction + OLS-KKT fast-path.
    """
    n_samples, n_types = F_mat.shape
    n_genes = M_mat.shape[0]

    if ridge_alpha > 0.0:
        F_aug = np.vstack([F_mat, np.sqrt(ridge_alpha) * np.eye(n_types, dtype=np.float64)])
        Q, R = np.linalg.qr(F_aug, mode="reduced")
        Q_top = Q[:n_samples, :]
    else:
        Q, R = np.linalg.qr(F_mat, mode="reduced")
        Q_top = Q

    rhs = M_mat @ Q_top  # (G, K)
    X_out = np.empty((n_genes, n_types), dtype=np.float64)

    diag_r = np.abs(np.diag(R))
    if np.all(diag_r > 1e-12):
        X_ols = linalg.solve_triangular(R, rhs.T, lower=False, check_finite=False).T
        kkt_ok = np.all(X_ols >= 0.0, axis=1)
        X_out[kkt_ok, :] = X_ols[kkt_ok, :]
        fallback_idx = np.flatnonzero(~kkt_ok)
    else:
        fallback_idx = np.arange(n_genes, dtype=int)

    for g in fallback_idx:
        X_out[g, :] = optimize.nnls(R, rhs[g, :])[0]

    return X_out


def impute_gep_group(
    mixture: ExpressionInput,
    fractions: Any,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    n_bootstrap: int = 50,
    ridge_alpha: float = 0.0,
    cond_threshold: float = 1e4,
    alpha: float = 0.05,
    cv_max: float = 1.0,
    adaptive_cv: bool = False,
    min_mean_fraction: float = 0.01,
    min_samples_present: int = 3,
    is_log2: bool = False,
    log2_offset: float = 0.0,
    seed: int = 0,
) -> GroupGEPResult:
    """
    CIBERSORTx Group-Mode cell-type gene expression estimation using QR-NNLS,
    sample-level bootstrap standard errors, one-sided z-tests with BH-FDR,
    and adaptive noise filtering (Newman et al., Nat Biotechnol 2019).
    """
    if n_bootstrap < 0:
        raise ValueError("`n_bootstrap` must be >= 0.")
    if ridge_alpha < 0.0:
        raise ValueError("`ridge_alpha` must be >= 0.")
    if cv_max <= 0.0:
        raise ValueError("`cv_max` must be positive.")

    M, F, gene_names, _sample_names, cell_types = _parse_fractions_and_align(
        mixture,
        fractions,
        layer=layer,
        samples_as_rows=samples_as_rows,
        is_log2=is_log2,
        log2_offset=log2_offset,
    )
    n_genes, n_samples = M.shape
    n_types = F.shape[1]

    if n_samples <= n_types:
        raise ValueError(
            f"Group-Mode requires n_samples > n_cell_types ({n_samples} <= {n_types})."
        )

    active_cols = np.max(F, axis=0) > 0.0
    if not np.any(active_cols):
        raise ValueError("All cell-type fractions are identically zero.")

    F_act = F[:, active_cols]
    cond_num = float(np.linalg.cond(F_act))

    gram = F_act.T @ F_act
    inv_gram = np.linalg.pinv(gram)
    col_norm2 = np.sum(F_act ** 2, axis=0)
    vif_act = np.maximum(np.diag(inv_gram) * col_norm2, 1.0)
    vif = np.full(n_types, np.inf, dtype=np.float64)
    vif[active_cols] = vif_act

    eff_ridge = float(ridge_alpha)
    if eff_ridge == 0.0 and cond_num > cond_threshold:
        warnings.warn(
            f"Fraction matrix is ill-conditioned (cond={cond_num:.2e} > {cond_threshold:.2e}); "
            "engaging adaptive ridge=1e-4 for stabilization.",
            UserWarning,
            stacklevel=2,
        )
        eff_ridge = 1e-4

    gep_act = _solve_batch_qr_nnls(F_act, M, ridge_alpha=eff_ridge)
    gep_raw = np.zeros((n_genes, n_types), dtype=np.float64)
    gep_raw[:, active_cols] = gep_act

    M_hat = gep_raw @ F.T
    ss_res = np.sum((M - M_hat) ** 2, axis=1)
    ss_tot = np.sum((M - np.mean(M, axis=1, keepdims=True)) ** 2, axis=1)
    r_squared = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)

    mean_frac = np.mean(F, axis=0)
    n_present = np.sum(F > 0.01, axis=0)
    estimable = active_cols & (mean_frac >= min_mean_fraction) & (n_present >= min_samples_present)

    se = np.full((n_genes, n_types), np.nan, dtype=np.float64)
    if n_bootstrap >= 2:
        rng = np.random.default_rng(seed)
        X_boot = np.full((n_bootstrap, n_genes, n_types), np.nan, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_samples, size=n_samples)
            F_b = F[idx, :]
            M_b = M[:, idx]
            act_b = np.max(F_b, axis=0) > 0.0
            if not np.any(act_b):
                continue
            F_b_act = F_b[:, act_b]
            cond_b = float(np.linalg.cond(F_b_act))
            ridge_b = eff_ridge if (eff_ridge > 0.0 or cond_b <= cond_threshold) else 1e-4
            X_b_act = _solve_batch_qr_nnls(F_b_act, M_b, ridge_alpha=ridge_b)
            tmp = np.full((n_genes, n_types), np.nan, dtype=np.float64)
            tmp[:, act_b] = X_b_act
            X_boot[b] = tmp

        finite_counts = np.sum(np.isfinite(X_boot), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            boot_sd = np.nanstd(X_boot, axis=0, ddof=1)
        se = np.where(finite_counts >= 2, boot_sd, np.nan)

    cv = np.where(np.isfinite(se), se / np.maximum(gep_raw, 1e-12), np.nan)
    z_score = np.full((n_genes, n_types), np.nan, dtype=np.float64)
    p_value = np.full((n_genes, n_types), np.nan, dtype=np.float64)

    for k in range(n_types):
        if not estimable[k]:
            continue
        se_k = se[:, k]
        est_k = gep_raw[:, k]
        valid_se = np.isfinite(se_k)
        pos_se = valid_se & (se_k > 0.0)
        zero_se = valid_se & (se_k <= 0.0)

        z_k = np.full(n_genes, np.nan, dtype=np.float64)
        z_k[pos_se] = est_k[pos_se] / se_k[pos_se]
        z_k[zero_se & (est_k > 0.0)] = np.inf
        z_k[zero_se & (est_k <= 0.0)] = 0.0
        z_score[:, k] = z_k

        fin_z = np.isfinite(z_k)
        p_k = np.full(n_genes, np.nan, dtype=np.float64)
        p_k[fin_z] = stats.norm.sf(z_k[fin_z])
        p_k[zero_se & (est_k > 0.0)] = 0.0
        p_value[:, k] = p_k

    q_value = np.full((n_genes, n_types), np.nan, dtype=np.float64)
    for k in range(n_types):
        q_value[:, k] = bh_fdr(p_value[:, k])

    cv_thresh = np.array(
        [
            cv_max * (1.0 + np.log1p(max(float(vif[k]) - 1.0, 0.0)))
            if (adaptive_cv and np.isfinite(vif[k]))
            else cv_max
            for k in range(n_types)
        ],
        dtype=np.float64,
    )

    passed_filter = (
        estimable[None, :]
        & (gep_raw > 0.0)
        & np.isfinite(se)
        & np.isfinite(q_value)
        & (q_value <= alpha)
        & (cv <= cv_thresh[None, :])
    )
    gep_filtered = np.where(passed_filter, gep_raw, np.nan)

    return GroupGEPResult(
        gep_raw=gep_raw,
        gep_filtered=gep_filtered,
        se=se,
        cv=cv,
        z_score=z_score,
        p_value=p_value,
        q_value=q_value,
        passed_filter=passed_filter,
        r_squared=r_squared,
        condition_number=cond_num,
        vif=vif,
        gene_names=gene_names,
        cell_types=cell_types,
    )


def _build_window_schedule(
    pos_window_bounds: list[tuple[int, int]],
) -> list[tuple[int, list[int], list[int], list[int], bool]]:
    """Group window positions by window length `(t1 - t0)` for vectorized median aggregation."""
    by_len: dict[int, list[tuple[int, int, int]]] = {}
    for p, (t0, t1) in enumerate(pos_window_bounds):
        by_len.setdefault(t1 - t0, []).append((p, t0, t1))

    schedule: list[tuple[int, list[int], list[int], list[int], bool]] = []
    for length, items in by_len.items():
        ps = [it[0] for it in items]
        t0s = [it[1] for it in items]
        t1s = [it[2] for it in items]
        contiguous = len(t0s) > 1 and t0s == list(range(t0s[0], t0s[0] + len(t0s)))
        schedule.append((length, ps, t0s, t1s, contiguous))
    return schedule


def _batched_spd_nnls_small_k(G: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Exact vectorized active-set NNLS for SPD Gram matrices `G` `(M, K, K)` with
    `2 <= K <= 5` on fallback windows where unconstrained OLS had negative entries.
    """
    n_prob, n_types, _ = G.shape
    best_obj = np.zeros(n_prob, dtype=np.float64)
    best_x = np.zeros((n_prob, n_types), dtype=np.float64)
    prob_idx = np.arange(n_prob)

    for s in range(1, n_types):
        subsets = list(itertools.combinations(range(n_types), s))
        idx = np.asarray(subsets, dtype=int)  # (C, s)
        G_sub = G[:, idx[:, :, None], idx[:, None, :]]  # (M, C, s, s)
        rhs_sub = rhs[:, idx]  # (M, C, s)

        if s == 1:
            diag = G_sub[:, :, 0, 0][..., None]
            x_sub = np.where(diag > 1e-24, rhs_sub / np.maximum(diag, 1e-24), np.nan)
        elif s == 2:
            a = G_sub[:, :, 0, 0]
            b = G_sub[:, :, 0, 1]
            d = G_sub[:, :, 1, 1]
            det = a * d - b * b
            safe_det = np.where(det > 1e-24, det, 1.0)
            r0 = rhs_sub[:, :, 0]
            r1 = rhs_sub[:, :, 1]
            x0 = np.where(det > 1e-24, (d * r0 - b * r1) / safe_det, np.nan)
            x1 = np.where(det > 1e-24, (-b * r0 + a * r1) / safe_det, np.nan)
            x_sub = np.stack([x0, x1], axis=-1)
        else:
            x_sub = np.linalg.solve(G_sub, rhs_sub[..., None])[..., 0]

        feasible = np.all(np.isfinite(x_sub), axis=2) & np.all(x_sub >= 0.0, axis=2)
        obj = np.where(feasible, -0.5 * np.sum(x_sub * rhs_sub, axis=2), np.inf)
        min_c = np.argmin(obj, axis=1)
        min_obj = obj[prob_idx, min_c]
        improve = min_obj < best_obj
        if np.any(improve):
            best_obj[improve] = min_obj[improve]
            best_x[improve] = 0.0
            m_imp = np.flatnonzero(improve)
            c_imp = min_c[improve]
            best_x[m_imp[:, None], idx[c_imp]] = x_sub[m_imp, c_imp]

    return best_x


def _solve_sliding_windows_batch(
    F: np.ndarray,
    Y: np.ndarray,
    k: int,
    other_contrib: np.ndarray,
    *,
    denom_k: np.ndarray,
    win_grid: np.ndarray,
    pos_window_bounds: list[tuple[int, int]],
    window_schedule: list[tuple[int, list[int], list[int], list[int], bool]],
    ridge_alpha: float,
    aggregation: Literal["median", "mean"],
    sort_by: Literal["residual", "bulk"],
    smooth_window: int,
) -> np.ndarray:
    """
    Sort samples and solve sliding-window non-negative ridge/NNLS across a batch
    `Y` `(B, N)` of bulk/permuted profiles, returning `(B, N)` purified profiles.
    """
    n_batch, n_samples = Y.shape
    n_types = F.shape[1]
    n_windows = win_grid.shape[0]

    if sort_by == "residual":
        score = (Y - other_contrib[None, :]) / denom_k[None, :]
    else:
        score = Y

    pi = np.argsort(score, axis=1, kind="mergesort")
    idx_windows = pi[:, win_grid]  # (B, T, w_eff)
    F_wins = F[idx_windows, :]  # (B, T, w_eff, K)
    b_idx = np.arange(n_batch)[:, None, None]
    Y_wins = Y[b_idx, idx_windows]  # (B, T, w_eff)

    eye_k = np.eye(n_types, dtype=np.float64)
    grams = np.einsum("btwi,btwj->btij", F_wins, F_wins) + ridge_alpha * eye_k[None, None, :, :]
    rhs = np.einsum("btwi,btw->bti", F_wins, Y_wins)

    x_sol = np.empty((n_batch, n_windows, n_types), dtype=np.float64)
    if n_types == 1:
        x_sol = np.maximum(rhs / np.maximum(grams[..., 0], 1e-30), 0.0)
    else:
        try:
            x_ols = np.linalg.solve(grams, rhs[..., None])[..., 0]
            kkt_ok = np.all(np.isfinite(x_ols), axis=2) & np.all(x_ols >= 0.0, axis=2)
            x_sol[kkt_ok] = x_ols[kkt_ok]
            fallback_mask = ~kkt_ok
        except np.linalg.LinAlgError:
            fallback_mask = np.ones((n_batch, n_windows), dtype=bool)

        if np.any(fallback_mask):
            if ridge_alpha >= 1e-8 and n_types <= 5:
                x_sol[fallback_mask] = _batched_spd_nnls_small_k(
                    grams[fallback_mask],
                    rhs[fallback_mask],
                )
            else:
                fallback_bt = np.argwhere(fallback_mask)
                sqrt_lam = np.sqrt(ridge_alpha) * eye_k
                zeros_k = np.zeros(n_types, dtype=np.float64)
                for b, t in fallback_bt:
                    if ridge_alpha > 0.0:
                        F_aug = np.vstack([F_wins[b, t], sqrt_lam])
                        y_aug = np.concatenate([Y_wins[b, t], zeros_k])
                        x_sol[b, t, :] = optimize.nnls(F_aug, y_aug)[0]
                    else:
                        x_sol[b, t, :] = optimize.nnls(F_wins[b, t], Y_wins[b, t])[0]

    x_win_k = x_sol[:, :, k]  # (B, T)
    if aggregation == "mean":
        cs = np.concatenate(
            [np.zeros((n_batch, 1), dtype=np.float64), np.cumsum(x_win_k, axis=1)],
            axis=1,
        )
        t0_arr = np.asarray([b[0] for b in pos_window_bounds], dtype=int)
        t1_arr = np.asarray([b[1] for b in pos_window_bounds], dtype=int)
        sorted_vals = (cs[:, t1_arr] - cs[:, t0_arr]) / (t1_arr - t0_arr)[None, :]
    else:
        sorted_vals = np.empty((n_batch, n_samples), dtype=np.float64)
        for length, ps, t0s, t1s, contiguous in window_schedule:
            if length == 1:
                sorted_vals[:, ps] = x_win_k[:, t0s]
            elif length == 2:
                t0_next = [t + 1 for t in t0s]
                sorted_vals[:, ps] = 0.5 * (x_win_k[:, t0s] + x_win_k[:, t0_next])
            elif contiguous:
                view = np.lib.stride_tricks.sliding_window_view(
                    x_win_k[:, t0s[0] : t0s[-1] + length],
                    length,
                    axis=1,
                )
                sorted_vals[:, ps] = np.median(view, axis=2)
            else:
                stacked = np.stack(
                    [x_win_k[:, t0:t1] for t0, t1 in zip(t0s, t1s)],
                    axis=1,
                )
                sorted_vals[:, ps] = np.median(stacked, axis=2)

    if smooth_window > 1:
        half = smooth_window // 2
        smoothed = np.empty_like(sorted_vals)
        for p in range(n_samples):
            lo = max(0, p - half)
            hi = min(n_samples, p + half + 1)
            smoothed[:, p] = np.median(sorted_vals[:, lo:hi], axis=1)
        sorted_vals = smoothed

    x_hat = np.empty_like(sorted_vals)
    np.put_along_axis(x_hat, pi, sorted_vals, axis=1)
    return x_hat


def _solve_sliding_windows_gene_ct(
    F: np.ndarray,
    m_g: np.ndarray,
    k: int,
    x_grp_g: np.ndarray,
    *,
    win_grid: np.ndarray,
    pos_window_bounds: list[tuple[int, int]],
    ridge_alpha: float,
    aggregation: Literal["median", "mean"],
    sort_by: Literal["residual", "bulk"],
    smooth_window: int,
) -> np.ndarray:
    """
    Sort samples for `(g, k)` and solve sliding-window non-negative ridge/NNLS
    across `win_grid` `(T, w_eff)`, returning purified profile `(N,)`.
    """
    other_contrib = (F @ x_grp_g) - F[:, k] * x_grp_g[k]
    denom_k = np.maximum(F[:, k], 0.05)
    schedule = _build_window_schedule(pos_window_bounds)
    return _solve_sliding_windows_batch(
        F,
        m_g[None, :],
        k,
        other_contrib,
        denom_k=denom_k,
        win_grid=win_grid,
        pos_window_bounds=pos_window_bounds,
        window_schedule=schedule,
        ridge_alpha=ridge_alpha,
        aggregation=aggregation,
        sort_by=sort_by,
        smooth_window=smooth_window,
    )[0]


def impute_gep_highres(
    mixture: ExpressionInput,
    fractions: Any,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    cell_types: Sequence[str] | None = None,
    window_size: int | None = None,
    ridge_alpha: float = 0.01,
    aggregation: Literal["median", "mean"] = "median",
    sort_by: Literal["residual", "bulk"] = "residual",
    smooth_window: int = 1,
    n_permutations: int = 199,
    alpha: float = 0.05,
    min_expr: float = 0.0,
    collapse_invariant: bool = False,
    is_log2: bool = False,
    log2_offset: float = 0.0,
    seed: int = 0,
) -> HighResGEPResult:
    """
    CIBERSORTx High-Resolution Mode sample-specific cell-type expression
    purification via residual-sorted sliding-window non-negative ridge/NNLS
    and residual-permutation significance testing (Newman et al., Nat Biotechnol 2019).
    """
    if ridge_alpha < 0.0:
        raise ValueError("`ridge_alpha` must be >= 0.")
    if aggregation not in ("median", "mean"):
        raise ValueError(f"Unsupported aggregation '{aggregation}'. Expected 'median' or 'mean'.")
    if sort_by not in ("residual", "bulk"):
        raise ValueError(f"Unsupported sort_by '{sort_by}'. Expected 'residual' or 'bulk'.")
    if smooth_window < 1:
        raise ValueError("`smooth_window` must be >= 1.")
    if n_permutations < 0:
        raise ValueError("`n_permutations` must be >= 0.")

    M, F, gene_names, sample_names, all_cell_types = _parse_fractions_and_align(
        mixture,
        fractions,
        layer=layer,
        samples_as_rows=samples_as_rows,
        is_log2=is_log2,
        log2_offset=log2_offset,
    )
    n_genes, n_samples = M.shape
    n_types = F.shape[1]

    if n_samples <= n_types + 2:
        raise ValueError(
            f"High-Resolution Mode requires n_samples > n_cell_types + 2 ({n_samples} <= {n_types + 2})."
        )

    if cell_types is None:
        target_indices = list(range(n_types))
        target_names = all_cell_types
    else:
        target_names = [str(ct) for ct in cell_types]
        target_indices = []
        for ct in target_names:
            if ct not in all_cell_types:
                raise ValueError(f"Unknown cell_type '{ct}' requested in `cell_types`.")
            target_indices.append(all_cell_types.index(ct))

    if window_size is None:
        w_eff = min(n_samples - 1, max(n_types + 3, min(4 * n_types, n_samples // 2)))
    else:
        if int(window_size) < 1:
            raise ValueError("`window_size` must be >= 1.")
        w_eff = int(window_size)
        if w_eff > n_samples:
            warnings.warn(
                f"window_size ({w_eff}) exceeds n_samples ({n_samples}); clamping to {n_samples}.",
                UserWarning,
                stacklevel=2,
            )
            w_eff = n_samples
        if w_eff <= n_types:
            warnings.warn(
                f"window_size ({w_eff}) <= n_cell_types ({n_types}); engaging ridge regularization.",
                UserWarning,
                stacklevel=2,
            )

    eff_ridge = float(ridge_alpha)
    if w_eff <= n_types and eff_ridge <= 0.0:
        eff_ridge = 0.01

    active_cols = np.max(F, axis=0) > 0.0
    if not np.any(active_cols):
        raise ValueError("All cell-type fractions are identically zero.")

    F_act = F[:, active_cols]
    grp_ridge = 0.0 if float(np.linalg.cond(F_act)) <= 1e4 else 1e-4
    grp_act = _solve_batch_qr_nnls(F_act, M, ridge_alpha=grp_ridge)
    group_gep_all = np.zeros((n_genes, n_types), dtype=np.float64)
    group_gep_all[:, active_cols] = grp_act

    n_target = len(target_indices)
    gep_raw = np.full((n_genes, n_samples, n_target), np.nan, dtype=np.float64)
    p_value = np.full((n_genes, n_target), np.nan, dtype=np.float64)

    n_windows = n_samples - w_eff + 1
    win_grid = np.arange(n_windows, dtype=int)[:, None] + np.arange(w_eff, dtype=int)[None, :]
    pos_window_bounds = [
        (max(0, p - w_eff + 1), min(n_windows, p + 1)) for p in range(n_samples)
    ]
    window_schedule = _build_window_schedule(pos_window_bounds)
    denom_by_k = [np.maximum(F[:, k], 0.05) for k in range(n_types)]
    perm_chunk_size = (
        max(1, min(n_permutations, 2_000_000 // max(1, n_windows * w_eff * n_types)))
        if n_permutations > 0
        else 0
    )

    rng = np.random.default_rng(seed)

    for g in range(n_genes):
        m_g = M[g, :]
        if np.max(m_g) == 0.0 or np.var(m_g) <= 1e-12:
            for out_k, k in enumerate(target_indices):
                if active_cols[k]:
                    gep_raw[g, :, out_k] = group_gep_all[g, k]
                    if n_permutations > 0:
                        p_value[g, out_k] = 1.0
            continue

        x_grp_g = group_gep_all[g, :]
        m_fit = F @ x_grp_g
        e_g = m_g - m_fit

        for out_k, k in enumerate(target_indices):
            if not active_cols[k]:
                continue
            other_contrib = m_fit - F[:, k] * x_grp_g[k]
            denom_k = denom_by_k[k]
            x_hat = _solve_sliding_windows_batch(
                F,
                m_g[None, :],
                k,
                other_contrib,
                denom_k=denom_k,
                win_grid=win_grid,
                pos_window_bounds=pos_window_bounds,
                window_schedule=window_schedule,
                ridge_alpha=eff_ridge,
                aggregation=aggregation,
                sort_by=sort_by,
                smooth_window=smooth_window,
            )[0]
            gep_raw[g, :, out_k] = x_hat

            if n_permutations > 0:
                t_obs = float(np.std(x_hat, ddof=0))
                ge_count = 0
                remaining = n_permutations
                while remaining > 0:
                    cur_b = min(perm_chunk_size, remaining)
                    perms = np.asarray(
                        [rng.permutation(n_samples) for _ in range(cur_b)],
                        dtype=np.intp,
                    )
                    y_perm = np.maximum(m_fit[None, :] + e_g[perms], 0.0)
                    x_perm = _solve_sliding_windows_batch(
                        F,
                        y_perm,
                        k,
                        other_contrib,
                        denom_k=denom_k,
                        win_grid=win_grid,
                        pos_window_bounds=pos_window_bounds,
                        window_schedule=window_schedule,
                        ridge_alpha=eff_ridge,
                        aggregation=aggregation,
                        sort_by=sort_by,
                        smooth_window=smooth_window,
                    )
                    t_perm = np.std(x_perm, axis=1, ddof=0)
                    ge_count += int(np.sum(t_perm >= t_obs))
                    remaining -= cur_b
                p_value[g, out_k] = (1.0 + ge_count) / (n_permutations + 1.0)

    q_value = np.full((n_genes, n_target), np.nan, dtype=np.float64)
    for out_k in range(n_target):
        q_value[:, out_k] = bh_fdr(p_value[:, out_k])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_gep = np.nanmean(gep_raw, axis=1)

    if n_permutations > 0:
        is_variable = np.isfinite(q_value) & (q_value <= alpha) & (mean_gep > min_expr)
    else:
        is_variable = np.isfinite(mean_gep) & (mean_gep > min_expr)

    group_sub = group_gep_all[:, target_indices]
    gep_filtered = np.full_like(gep_raw, np.nan)
    for g in range(n_genes):
        for out_k in range(n_target):
            if is_variable[g, out_k]:
                gep_filtered[g, :, out_k] = gep_raw[g, :, out_k]
            elif collapse_invariant and active_cols[target_indices[out_k]]:
                gep_filtered[g, :, out_k] = group_sub[g, out_k]

    return HighResGEPResult(
        gep_raw=gep_raw,
        gep_filtered=gep_filtered,
        group_gep=group_sub,
        p_value=p_value,
        q_value=q_value,
        is_variable=is_variable,
        window_size=w_eff,
        gene_names=gene_names,
        sample_names=sample_names,
        cell_types=target_names,
    )
