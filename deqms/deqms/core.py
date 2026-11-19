"""Core linear modeling and count-dependent Empirical Bayes for DEqMS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats

from ._validation import SettingsMode, validate_settings
from .preprocessing import ProteinSummarizationResult


def _trigamma_inverse(y: float) -> float:
    """Invert ``polygamma(1, x) = y`` for ``y > 0`` via Smyth's Newton method."""
    if not np.isfinite(y) or y <= 0.0:
        return 1e10
    if y > 1e7:
        return float(1.0 / np.sqrt(y))
    if y < 1e-6:
        return float(1.0 / y + 0.5)

    x = 0.5 + 1.0 / y
    for _ in range(25):
        tri = float(special.polygamma(1, x))
        tetra = float(special.polygamma(2, x))
        step = tri * (1.0 - tri / y) / tetra
        x += step
        if abs(step) <= 1e-11 * x:
            break
    return float(max(x, 1e-8))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment preserving NaN entries."""
    out = np.full(pvals.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)
    n = int(np.sum(valid))
    if n == 0:
        return out
    pv = np.clip(pvals[valid], 0.0, 1.0)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * (n / np.arange(1, n + 1, dtype=np.float64))
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n)
    out[valid] = adj_sorted[inv_order]
    return out


def _compute_b_lods(
    t_stat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df_total: np.ndarray,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
) -> np.ndarray:
    """Compute Smyth (2004) Empirical Bayes log-odds (B-statistic)."""
    out = np.full(t_stat.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(t_stat)
        & np.isfinite(stdev_unscaled)
        & (stdev_unscaled > 0.0)
        & np.isfinite(df_total)
        & (df_total > 0.0)
    )
    n = int(np.sum(valid))
    if n == 0:
        return out

    tv = np.abs(t_stat[valid])
    uv = stdev_unscaled[valid]
    dfv = df_total[valid]

    # Smyth's tmixture estimator for prior variance v0 of true non-null beta
    v0_min = stdev_coef_lim[0] ** 2
    v0_max = stdev_coef_lim[1] ** 2
    n_top = max(1, int(np.ceil(0.5 * proportion * n)))
    top_idx = np.argsort(tv)[::-1][:n_top]

    v0_est_list: list[float] = []
    for rank_0, idx_i in enumerate(top_idx):
        t_i = float(tv[idx_i])
        df_i = float(dfv[idx_i])
        u_i = float(uv[idx_i])
        p0 = 2.0 * float(stats.t.sf(t_i, df=min(df_i, 1e8)))
        p1 = (rank_0 + 0.5) / n
        tail_de = (p1 - (1.0 - proportion) * p0) / proportion
        if 0.0 < tail_de < 1.0:
            t1 = float(stats.t.isf(0.5 * tail_de, df=min(df_i, 1e8)))
            if t1 > 0.0 and t_i > t1:
                v0_est_list.append((u_i**2) * ((t_i / t1) ** 2 - 1.0))

    if v0_est_list:
        v0 = float(np.clip(np.median(v0_est_list), v0_min, v0_max))
    else:
        v0 = float(np.sqrt(v0_min * v0_max))

    r = (uv**2) / (uv**2 + v0)
    log_prior_odds = float(np.log(proportion / (1.0 - proportion)))

    finite_df = dfv < 1e7
    kern = np.empty(n, dtype=np.float64)
    if np.any(finite_df):
        d_f = dfv[finite_df]
        t_f = tv[finite_df]
        r_f = r[finite_df]
        kern[finite_df] = (
            0.5
            * (d_f + 1.0)
            * (np.log1p((t_f**2) / d_f) - np.log1p(r_f * (t_f**2) / d_f))
        )
    if np.any(~finite_df):
        t_inf = tv[~finite_df]
        r_inf = r[~finite_df]
        kern[~finite_df] = 0.5 * (1.0 - r_inf) * (t_inf**2)

    out[valid] = log_prior_odds + 0.5 * np.log(r) + kern
    return out


def _fit_f_dist_nonrobust(
    log_var_residual: np.ndarray, df_resid: np.ndarray
) -> tuple[float, float]:
    """Moment estimator for ``(df_prior, emean)`` on detrended log-variances."""
    e = (
        log_var_residual
        - special.digamma(0.5 * df_resid)
        + np.log(0.5 * df_resid)
    )
    emean = float(np.mean(e))
    n = e.size
    if n <= 1:
        return 1e10, emean

    sample_var = float(np.sum((e - emean) ** 2) / (n - 1))
    mean_tri = float(np.mean(special.polygamma(1, 0.5 * df_resid)))
    evar = sample_var - mean_tri
    if evar <= 0.0:
        return 1e10, emean
    df_prior = 2.0 * _trigamma_inverse(evar)
    return df_prior, emean


def _winsorized_log_f_moments(
    df1: float, df2: float, p_lo: float, p_hi: float, u_grid: np.ndarray
) -> tuple[float, float]:
    """Exact midpoint-quadrature Winsorized mean & variance of ``ln F(df1, df2)``."""
    # Log of F(df1, df2) = ln(chi2_df1 / df1) - ln(chi2_df2 / df2)
    if df2 >= 1e7:
        q = stats.chi2.ppf(u_grid, df=df1) / df1
        vals = np.log(np.maximum(q, 1e-300))
    else:
        q = stats.f.ppf(u_grid, dfn=df1, dfd=df2)
        vals = np.log(np.maximum(q, 1e-300))
    lo, hi = np.quantile(vals, [p_lo, 1.0 - p_hi])
    w = np.clip(vals, lo, hi)
    mean_w = float(np.mean(w))
    var_w = float(np.mean((w - mean_w) ** 2))
    return mean_w, var_w


def _fit_f_dist_robust(
    log_var_residual: np.ndarray,
    df_resid: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[float, float]:
    """Winsorized moment estimator for ``(df_prior, emean)`` (Phipson et al. 2016)."""
    p_lo, p_hi = winsor_tail_p
    e = (
        log_var_residual
        - special.digamma(0.5 * df_resid)
        + np.log(0.5 * df_resid)
    )
    n = e.size
    if n <= 3:
        return _fit_f_dist_nonrobust(log_var_residual, df_resid)

    q_lo, q_hi = np.quantile(e, [p_lo, 1.0 - p_hi])
    e_win = np.clip(e, q_lo, q_hi)
    win_mean = float(np.mean(e_win))
    win_var = float(np.sum((e_win - win_mean) ** 2) / (n - 1))

    df1_rep = float(np.median(df_resid))
    u_grid = (np.arange(256, dtype=np.float64) + 0.5) / 256.0

    _, var_inf = _winsorized_log_f_moments(df1_rep, 1e8, p_lo, p_hi, u_grid)
    if win_var <= var_inf:
        df_prior = 1e10
        mean_shift, _ = _winsorized_log_f_moments(df1_rep, 1e8, p_lo, p_hi, u_grid)
    else:
        _, var_tiny = _winsorized_log_f_moments(df1_rep, 0.05, p_lo, p_hi, u_grid)
        if win_var >= var_tiny:
            df_prior = 0.05
        else:
            def _obj(log_d0: float) -> float:
                _, vw = _winsorized_log_f_moments(
                    df1_rep, float(np.exp(log_d0)), p_lo, p_hi, u_grid
                )
                return vw - win_var

            sol_log_d0 = float(
                optimize.brentq(_obj, np.log(0.05), np.log(1e5), xtol=1e-5)
            )
            df_prior = float(np.exp(sol_log_d0))
        mean_shift, _ = _winsorized_log_f_moments(
            df1_rep, df_prior, p_lo, p_hi, u_grid
        )

    # Recall ln F(df1, df2) has theoretical un-Winsorized mean:
    # [digamma(df1/2) - ln(df1/2)] - [digamma(df2/2) - ln(df2/2)]
    unwin_theory_mean = float(
        special.digamma(0.5 * df1_rep) - np.log(0.5 * df1_rep)
    ) - (
        0.0
        if df_prior >= 1e7
        else float(special.digamma(0.5 * df_prior) - np.log(0.5 * df_prior))
    )
    winsor_bias = mean_shift - unwin_theory_mean
    emean = win_mean - winsor_bias
    return df_prior, emean


def _loess_degree1_tricube(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    span: float = 0.75,
) -> np.ndarray:
    """Exact local-linear (degree 1) tricube LOESS evaluated at ``x_eval``."""
    n = x_train.size
    if n == 0:
        raise ValueError("Cannot fit LOESS on empty arrays.")
    if span <= 0.0:
        raise ValueError(f"LOESS span must be > 0, got {span}.")
    if n == 1:
        return np.full(x_eval.shape, float(y_train[0]), dtype=np.float64)

    uniq_x, inv_idx = np.unique(x_eval, return_inverse=True)
    u_preds = np.empty(uniq_x.size, dtype=np.float64)

    q = max(2, min(n, int(np.floor(span * n)))) if span <= 1.0 else n

    for idx_u, x0 in enumerate(uniq_x):
        dists = np.abs(x_train - x0)
        if span <= 1.0:
            d_max = float(np.partition(dists, q - 1)[q - 1])
        else:
            d_max = float(np.max(dists) * span)

        if d_max <= 0.0:
            same = dists == 0.0
            u_preds[idx_u] = float(np.mean(y_train[same]))
            continue

        in_win = dists < d_max
        if not np.any(in_win):
            same = dists <= d_max
            u_preds[idx_u] = float(np.mean(y_train[same]))
            continue

        xw = x_train[in_win]
        yw = y_train[in_win]
        u = dists[in_win] / d_max
        w = (1.0 - u**3) ** 3
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            u_preds[idx_u] = float(np.mean(yw))
            continue

        x_bar = float(np.sum(w * xw) / w_sum)
        y_bar = float(np.sum(w * yw) / w_sum)
        dx = xw - x_bar
        s_xx = float(np.sum(w * dx * dx))
        if s_xx <= 1e-15:
            u_preds[idx_u] = y_bar
        else:
            s_xy = float(np.sum(w * dx * (yw - y_bar)))
            slope = s_xy / s_xx
            u_preds[idx_u] = y_bar + slope * (float(x0) - x_bar)

    return u_preds[inv_idx]


def _natural_cubic_spline_basis(
    x_train: np.ndarray, x_eval: np.ndarray, df: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Construct natural cubic spline basis (intercept + ``df`` degrees of freedom)."""
    x_min = float(np.min(x_train))
    x_max = float(np.max(x_train))
    if x_max <= x_min or df <= 1:
        b_tr = np.column_stack([np.ones_like(x_train), x_train])
        b_ev = np.column_stack([np.ones_like(x_eval), x_eval])
        return b_tr, b_ev

    n_interior = max(0, df - 1)
    if n_interior > 0:
        probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        raw_knots = np.quantile(x_train, probs)
        interior = np.unique(raw_knots[(raw_knots > x_min) & (raw_knots < x_max)])
    else:
        interior = np.array([], dtype=np.float64)

    knots = np.concatenate([[x_min], interior, [x_max]])
    k_len = knots.size
    if k_len <= 2:
        b_tr = np.column_stack([np.ones_like(x_train), x_train])
        b_ev = np.column_stack([np.ones_like(x_eval), x_eval])
        return b_tr, b_ev

    def _eval_basis(xv: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(xv), xv]
        xi_last = knots[-1]
        xi_prev = knots[-2]
        d_prev = (
            np.maximum(xv - xi_prev, 0.0) ** 3 - np.maximum(xv - xi_last, 0.0) ** 3
        ) / (xi_last - xi_prev)
        for k_idx in range(k_len - 2):
            xi_k = knots[k_idx]
            d_k = (
                np.maximum(xv - xi_k, 0.0) ** 3 - np.maximum(xv - xi_last, 0.0) ** 3
            ) / (xi_last - xi_k)
            cols.append(d_k - d_prev)
        return np.column_stack(cols)

    return _eval_basis(x_train), _eval_basis(x_eval)


def _fit_variance_count_trend(
    log2_count: np.ndarray,
    log_var: np.ndarray,
    fit_mask: np.ndarray,
    fit_method: str,
    span: float,
) -> np.ndarray:
    """Fit ``f_hat(log2(count))`` on valid proteins and predict for all proteins."""
    valid_methods = ("loess", "spline", "nls")
    if fit_method not in valid_methods:
        raise ValueError(
            f"Unknown fit_method={fit_method!r}. Expected one of {valid_methods}."
        )

    x_tr = log2_count[fit_mask]
    y_tr = log_var[fit_mask]
    if x_tr.size == 0:
        raise ValueError(
            "No proteins with positive residual variance available to fit variance trend."
        )

    if fit_method == "loess":
        return _loess_degree1_tricube(x_tr, y_tr, log2_count, span=span)

    if fit_method == "spline":
        b_tr, b_ev = _natural_cubic_spline_basis(x_tr, log2_count, df=4)
        coef, _, _, _ = linalg.lstsq(b_tr, y_tr)
        return np.asarray(b_ev @ coef, dtype=np.float64)

    # fit_method == "nls": a + b / count
    inv_c_tr = np.exp2(-x_tr)
    inv_c_ev = np.exp2(-log2_count)
    design_tr = np.column_stack([np.ones_like(inv_c_tr), inv_c_tr])
    design_ev = np.column_stack([np.ones_like(inv_c_ev), inv_c_ev])
    coef, _, _, _ = linalg.lstsq(design_tr, y_tr)
    return np.asarray(design_ev @ coef, dtype=np.float64)


def _prepare_design_and_contrast(
    design: Any,
    contrast: Any,
    n_samples: int,
) -> tuple[np.ndarray, list[str], np.ndarray | None, list[str] | None, int]:
    """Normalize design and optional contrast into 2D matrices + default coef index."""
    if design is None:
        raise ValueError("design must be provided when expr_or_fit is an expression matrix.")

    if isinstance(design, pd.DataFrame):
        x_mat = design.to_numpy(dtype=np.float64, copy=True)
        coef_names = [str(c) for c in design.columns]
        default_coef = 1 if x_mat.shape[1] >= 2 else 0
    else:
        raw = np.asarray(design)
        if raw.ndim == 1:
            if raw.shape[0] != n_samples:
                raise ValueError(
                    f"1D design length ({raw.shape[0]}) != n_samples ({n_samples})."
                )
            # Group labels -> build indicator or intercept+treatment design
            uniq_groups: list[Any] = []
            for g in raw:
                if g not in uniq_groups:
                    uniq_groups.append(g)
            if len(uniq_groups) < 2:
                raise ValueError("1D group design must contain at least 2 distinct groups.")
            if contrast is not None:
                # One-hot group indicator matrix so contrast vector applies directly to group means
                x_mat = np.column_stack(
                    [(raw == g).astype(np.float64) for g in uniq_groups]
                )
                coef_names = [str(g) for g in uniq_groups]
                default_coef = 0
            else:
                # Intercept + treatment effects relative to first group (e.g. group1 - group0)
                cols = [np.ones(n_samples, dtype=np.float64)]
                coef_names = ["Intercept"]
                for g in uniq_groups[1:]:
                    cols.append((raw == g).astype(np.float64))
                    coef_names.append(f"{g}_vs_{uniq_groups[0]}")
                x_mat = np.column_stack(cols)
                default_coef = 1
        elif raw.ndim == 2:
            x_mat = raw.astype(np.float64, copy=True)
            coef_names = [f"coef_{j}" for j in range(x_mat.shape[1])]
            # If first column is an all-ones intercept and >= 2 columns, default to col 1
            if x_mat.shape[1] >= 2 and np.allclose(x_mat[:, 0], 1.0):
                default_coef = 1
            else:
                default_coef = 0
        else:
            raise ValueError("design must be a 1D group vector or 2D design matrix.")

    if x_mat.shape[0] != n_samples:
        raise ValueError(
            f"design rows ({x_mat.shape[0]}) must match expression samples ({n_samples})."
        )
    if not np.all(np.isfinite(x_mat)):
        raise ValueError("design matrix must contain only finite values.")

    if contrast is None:
        return x_mat, coef_names, None, None, default_coef

    if isinstance(contrast, pd.DataFrame):
        c_mat = contrast.to_numpy(dtype=np.float64, copy=True)
        c_names = [str(c) for c in contrast.columns]
    elif isinstance(contrast, pd.Series):
        c_mat = contrast.to_numpy(dtype=np.float64, copy=True)[:, np.newaxis]
        c_names = [str(contrast.name) if contrast.name is not None else "contrast_0"]
    else:
        c_arr = np.asarray(contrast, dtype=np.float64)
        if c_arr.ndim == 1:
            c_mat = c_arr[:, np.newaxis]
            c_names = ["contrast_0"]
        elif c_arr.ndim == 2:
            c_mat = c_arr.copy()
            c_names = [f"contrast_{j}" for j in range(c_arr.shape[1])]
        else:
            raise ValueError("contrast must be a 1D vector or 2D contrast matrix.")

    if c_mat.shape[0] != x_mat.shape[1]:
        raise ValueError(
            f"contrast length ({c_mat.shape[0]}) must match design columns ({x_mat.shape[1]})."
        )
    return x_mat, coef_names, c_mat, c_names, 0


@dataclass
class LimmaFit:
    """Per-protein linear model fit (analogous to ``limma::MArrayLM``)."""

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    s2: np.ndarray
    df_residual: np.ndarray
    ave_expr: np.ndarray
    protein_names: list[str]
    coef_names: list[str]
    default_coef: int
    count: np.ndarray | None = None
    settings: SettingsMode = "default"


@dataclass
class DEqMSFit:
    """Full DEqMS + standard limma Empirical Bayes result container."""

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    s2: np.ndarray
    df_residual: np.ndarray
    ave_expr: np.ndarray
    protein_names: list[str]
    coef_names: list[str]
    default_coef: int
    count: np.ndarray
    # Standard limma eBayes statistics
    df_prior: float
    s2_prior: float
    s2_post: np.ndarray
    t: np.ndarray
    p_value: np.ndarray
    adj_p_val: np.ndarray
    b: np.ndarray
    # DEqMS count-dependent eBayes statistics
    sca_f_hat: np.ndarray
    sca_dfprior: float
    sca_dfprior_vec: np.ndarray
    sca_priorvar: np.ndarray
    sca_postvar: np.ndarray
    sca_t: np.ndarray
    sca_p: np.ndarray
    sca_adj_pval: np.ndarray
    sca_b: np.ndarray
    fit_method: str
    span: float
    robust: bool
    settings: SettingsMode

    def to_dataframe(
        self, coef: int | str | None = None, sort_by: str | None = "sca.P.Value"
    ) -> pd.DataFrame:
        """Format DEqMS results as a DataFrame via ``output_result``."""
        return output_result(
            self, coef=coef, sort_by=sort_by, settings=self.settings
        )


def lm_fit(
    expr: pd.DataFrame | np.ndarray | ProteinSummarizationResult,
    design: Any,
    contrast: Any = None,
    count: np.ndarray | pd.Series | list[int] | None = None,
    settings: SettingsMode = "default",
) -> LimmaFit:
    """Fit per-protein OLS models supporting both complete and partially missing rows."""
    validate_settings(settings)

    extracted_count = count
    if isinstance(expr, ProteinSummarizationResult):
        if extracted_count is None:
            extracted_count = expr.count
        expr_df_or_arr: pd.DataFrame | np.ndarray = expr.abundance
    else:
        expr_df_or_arr = expr

    if isinstance(expr_df_or_arr, pd.DataFrame):
        protein_names = [str(idx) for idx in expr_df_or_arr.index]
        y_mat = expr_df_or_arr.to_numpy(dtype=np.float64, copy=True)
    else:
        y_mat = np.asarray(expr_df_or_arr, dtype=np.float64).copy()
        if y_mat.ndim != 2:
            raise ValueError("Expression matrix must be 2D (n_proteins, n_samples).")
        protein_names = [f"protein_{i}" for i in range(y_mat.shape[0])]

    n_prot, n_samp = y_mat.shape
    if n_prot == 0 or n_samp == 0:
        raise ValueError("Expression matrix must be non-empty.")

    y_mat[~np.isfinite(y_mat)] = np.nan
    x_mat, raw_coef_names, c_mat, c_names, default_coef = _prepare_design_and_contrast(
        design, contrast, n_samp
    )

    out_names = c_names if c_names is not None else raw_coef_names
    n_out_coefs = len(out_names)

    coefs = np.full((n_prot, n_out_coefs), np.nan, dtype=np.float64)
    stdev_unscaled = np.full((n_prot, n_out_coefs), np.nan, dtype=np.float64)
    s2 = np.full(n_prot, np.nan, dtype=np.float64)
    df_resid = np.zeros(n_prot, dtype=np.float64)

    row_has_obs = np.any(np.isfinite(y_mat), axis=1)
    ave_expr = np.full(n_prot, np.nan, dtype=np.float64)
    if np.any(row_has_obs):
        ave_expr[row_has_obs] = np.nanmean(y_mat[row_has_obs], axis=1)

    complete_rows = np.all(np.isfinite(y_mat), axis=1)

    # Fast vectorized path for all complete rows
    if np.any(complete_rows):
        y_comp = y_mat[complete_rows]  # (G_comp, N)
        beta_raw, _, rank_x, _ = linalg.lstsq(x_mat, y_comp.T)  # (K, G_comp)
        df_full = n_samp - int(rank_x)
        xtx_pinv = linalg.pinv(x_mat.T @ x_mat)
        proj = xtx_pinv @ (x_mat.T @ x_mat)

        if c_mat is not None:
            beta_out = (c_mat.T @ beta_raw).T
            cov_c = c_mat.T @ xtx_pinv @ c_mat
            u_vec = np.sqrt(np.maximum(np.diag(cov_c), 0.0))
            # Check estimability of each contrast column
            for m_idx in range(n_out_coefs):
                c_col = c_mat[:, m_idx]
                c_norm = float(linalg.norm(c_col))
                if c_norm > 0 and float(linalg.norm(c_col - proj.T @ c_col)) > 1e-7 * c_norm:
                    beta_out[:, m_idx] = np.nan
                    u_vec[m_idx] = np.nan
        else:
            beta_out = beta_raw.T
            u_vec = np.sqrt(np.maximum(np.diag(xtx_pinv), 0.0))
            if int(rank_x) < x_mat.shape[1]:
                for k_idx in range(x_mat.shape[1]):
                    e_k = np.zeros(x_mat.shape[1], dtype=np.float64)
                    e_k[k_idx] = 1.0
                    if float(linalg.norm(e_k - proj.T @ e_k)) > 1e-7:
                        beta_out[:, k_idx] = np.nan
                        u_vec[k_idx] = np.nan

        coefs[complete_rows] = beta_out
        stdev_unscaled[complete_rows] = u_vec[np.newaxis, :]
        df_resid[complete_rows] = float(max(df_full, 0))
        if df_full >= 1:
            fitted = (x_mat @ beta_raw).T
            rss = np.sum((y_comp - fitted) ** 2, axis=1)
            s2[complete_rows] = rss / df_full

    # Exact per-protein QR/SVD OLS for partially observed rows
    incomplete_indices = np.flatnonzero(~complete_rows)
    for g_idx in incomplete_indices:
        obs = np.isfinite(y_mat[g_idx])
        n_obs = int(np.sum(obs))
        if n_obs == 0:
            continue
        x_g = x_mat[obs]
        y_g = y_mat[g_idx, obs]
        beta_g, _, rank_g, _ = linalg.lstsq(x_g, y_g)
        d_g = n_obs - int(rank_g)
        df_resid[g_idx] = float(max(d_g, 0))

        xtx_pinv_g = linalg.pinv(x_g.T @ x_g)
        proj_g = xtx_pinv_g @ (x_g.T @ x_g)

        if c_mat is not None:
            for m_idx in range(n_out_coefs):
                c_col = c_mat[:, m_idx]
                c_norm = float(linalg.norm(c_col))
                if c_norm > 0 and float(linalg.norm(c_col - proj_g.T @ c_col)) <= 1e-7 * c_norm:
                    coefs[g_idx, m_idx] = float(c_col @ beta_g)
                    var_factor = float(c_col @ xtx_pinv_g @ c_col)
                    if var_factor > 0.0:
                        stdev_unscaled[g_idx, m_idx] = float(np.sqrt(var_factor))
        else:
            for k_idx in range(n_out_coefs):
                e_k = np.zeros(x_g.shape[1], dtype=np.float64)
                e_k[k_idx] = 1.0
                if float(linalg.norm(e_k - proj_g.T @ e_k)) <= 1e-7:
                    coefs[g_idx, k_idx] = float(beta_g[k_idx])
                    var_factor = float(xtx_pinv_g[k_idx, k_idx])
                    if var_factor > 0.0:
                        stdev_unscaled[g_idx, k_idx] = float(np.sqrt(var_factor))

        if d_g >= 1:
            rss_g = float(np.sum((y_g - x_g @ beta_g) ** 2))
            s2[g_idx] = rss_g / d_g

    sigma = np.where(np.isfinite(s2) & (s2 >= 0.0), np.sqrt(np.maximum(s2, 0.0)), np.nan)

    count_arr: np.ndarray | None = None
    if extracted_count is not None:
        if isinstance(extracted_count, pd.Series):
            if set(protein_names).issubset(set(extracted_count.index.astype(str))):
                reindexed = extracted_count.copy()
                reindexed.index = reindexed.index.astype(str)
                count_arr = reindexed.loc[protein_names].to_numpy(dtype=np.float64)
            else:
                count_arr = extracted_count.to_numpy(dtype=np.float64)
        else:
            count_arr = np.asarray(extracted_count, dtype=np.float64)

    return LimmaFit(
        coefficients=coefs,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        s2=s2,
        df_residual=df_resid,
        ave_expr=ave_expr,
        protein_names=protein_names,
        coef_names=out_names,
        default_coef=default_coef,
        count=count_arr,
        settings=settings,
    )


def _coerce_to_limma_fit(
    expr_or_fit: Any,
    design: Any,
    contrast: Any,
    count: Any,
    settings: SettingsMode,
) -> LimmaFit:
    """Convert expression matrix, summarization result, dict, or LimmaFit into LimmaFit."""
    if isinstance(expr_or_fit, LimmaFit):
        fit = expr_or_fit
        if count is not None:
            if isinstance(count, pd.Series) and set(fit.protein_names).issubset(
                set(count.index.astype(str))
            ):
                s_cnt = count.copy()
                s_cnt.index = s_cnt.index.astype(str)
                cnt_arr = s_cnt.loc[fit.protein_names].to_numpy(dtype=np.float64)
            else:
                cnt_arr = np.asarray(count, dtype=np.float64)
            return LimmaFit(
                coefficients=fit.coefficients,
                stdev_unscaled=fit.stdev_unscaled,
                sigma=fit.sigma,
                s2=fit.s2,
                df_residual=fit.df_residual,
                ave_expr=fit.ave_expr,
                protein_names=fit.protein_names,
                coef_names=fit.coef_names,
                default_coef=fit.default_coef,
                count=cnt_arr,
                settings=settings,
            )
        return fit

    if isinstance(expr_or_fit, dict):
        coefs = np.asarray(expr_or_fit["coefficients"], dtype=np.float64)
        if coefs.ndim == 1:
            coefs = coefs[:, np.newaxis]
        u = np.asarray(expr_or_fit["stdev_unscaled"], dtype=np.float64)
        if u.ndim == 1:
            u = u[:, np.newaxis]
        if "s2" in expr_or_fit:
            s2 = np.asarray(expr_or_fit["s2"], dtype=np.float64)
        elif "sigma" in expr_or_fit:
            s2 = np.asarray(expr_or_fit["sigma"], dtype=np.float64) ** 2
        else:
            raise ValueError("Fit dict must include 's2' or 'sigma'.")
        df_r = np.asarray(expr_or_fit["df_residual"], dtype=np.float64)
        if df_r.ndim == 0:
            df_r = np.full(s2.shape[0], float(df_r), dtype=np.float64)
        ave = np.asarray(
            expr_or_fit.get("AveExpr", expr_or_fit.get("ave_expr", np.zeros_like(s2))),
            dtype=np.float64,
        )
        p_names = [
            str(p)
            for p in expr_or_fit.get(
                "protein_names", [f"protein_{i}" for i in range(s2.shape[0])]
            )
        ]
        c_names = [
            str(c)
            for c in expr_or_fit.get(
                "coef_names", [f"coef_{j}" for j in range(coefs.shape[1])]
            )
        ]
        raw_cnt = count if count is not None else expr_or_fit.get("count")
        cnt_arr = np.asarray(raw_cnt, dtype=np.float64) if raw_cnt is not None else None
        return LimmaFit(
            coefficients=coefs,
            stdev_unscaled=u,
            sigma=np.sqrt(np.maximum(s2, 0.0)),
            s2=s2,
            df_residual=df_r,
            ave_expr=ave,
            protein_names=p_names,
            coef_names=c_names,
            default_coef=int(expr_or_fit.get("default_coef", 0)),
            count=cnt_arr,
            settings=settings,
        )

    return lm_fit(
        expr=expr_or_fit,
        design=design,
        contrast=contrast,
        count=count,
        settings=settings,
    )


def spectra_count_ebayes(
    expr_or_fit: pd.DataFrame | np.ndarray | ProteinSummarizationResult | LimmaFit | dict[str, Any],
    design: Any = None,
    contrast: Any = None,
    count: np.ndarray | pd.Series | list[int] | None = None,
    fit_method: str = "loess",
    span: float = 0.75,
    robust: bool | None = None,
    settings: SettingsMode = "default",
) -> DEqMSFit:
    """Run DEqMS peptide/PSM-count-dependent Empirical Bayes moderation.

    Implements ``DEqMS::spectraCounteBayes`` (Zhu et al., *MCP* 2020) alongside
    standard ``limma::eBayes`` for direct comparison:
    - ``settings="default"``: ``robust=True`` (Winsorized prior-df estimation +
      per-protein outlier prior degrees of freedom ``sca_dfprior_vec``), trend
      fitted on proteins with ``df_g >= 2``.
    - ``settings="R_default"``: ``robust=False`` (scalar ``sca_dfprior``),
      local-linear degree-1 tricube ``loess`` (``span=0.75``), strict
      ``min(count) >= 1`` validation, exact ``DEqMS`` v1.22 parity.
    """
    validate_settings(settings)
    use_robust = (settings == "default") if robust is None else bool(robust)

    base_fit = _coerce_to_limma_fit(
        expr_or_fit=expr_or_fit,
        design=design,
        contrast=contrast,
        count=count,
        settings=settings,
    )

    if base_fit.count is None:
        raise ValueError(
            "Peptide/PSM count vector 'count' must be provided (or included in expr_or_fit)."
        )

    cnt = np.asarray(base_fit.count, dtype=np.float64)
    n_prot, n_coefs = base_fit.coefficients.shape
    if cnt.ndim != 1 or cnt.shape[0] != n_prot:
        raise ValueError(
            f"count length ({cnt.shape}) must match number of proteins ({n_prot})."
        )
    if not np.all(np.isfinite(cnt)) or np.min(cnt) < 1.0:
        raise ValueError("All protein PSM/peptide counts must be finite and >= 1.")

    s2_raw = base_fit.s2.copy()
    df_r = base_fit.df_residual.copy()

    # Clamp exact machine-zero variances on constant non-missing rows to tiny positive floor
    pos_s2 = s2_raw[np.isfinite(s2_raw) & (s2_raw > 0.0)]
    if pos_s2.size == 0:
        raise ValueError("No proteins have positive residual variance s2 > 0.")
    s2_floor = float(np.min(pos_s2) * 1e-4)
    s2 = np.where(
        np.isfinite(s2_raw) & (df_r >= 1.0),
        np.maximum(s2_raw, s2_floor),
        np.nan,
    )

    # Trend-fitting mask: df_g >= 2 in "default" (with fallback to >= 1 if tiny sample size)
    min_df_trend = 2.0 if settings == "default" else 1.0
    fit_mask = np.isfinite(s2) & (df_r >= min_df_trend)
    if np.sum(fit_mask) < 3:
        fit_mask = np.isfinite(s2) & (df_r >= 1.0)
    eval_mask = np.isfinite(s2) & (df_r >= 1.0)

    # --- 1. Standard limma Empirical Bayes (untrended) ---
    log_s2 = np.where(eval_mask, np.log(np.where(eval_mask, s2, 1.0)), np.nan)
    if use_robust:
        limma_d0, limma_emean = _fit_f_dist_robust(log_s2[fit_mask], df_r[fit_mask])
    else:
        limma_d0, limma_emean = _fit_f_dist_nonrobust(log_s2[fit_mask], df_r[fit_mask])

    limma_scale_shift = (
        0.0
        if limma_d0 >= 1e7
        else float(special.digamma(0.5 * limma_d0) - np.log(0.5 * limma_d0))
    )
    s2_prior_limma = float(np.exp(limma_emean + limma_scale_shift))

    s2_post_limma = np.full(n_prot, np.nan, dtype=np.float64)
    df_tot_limma = np.full(n_prot, np.nan, dtype=np.float64)
    if limma_d0 >= 1e7:
        s2_post_limma[eval_mask] = s2_prior_limma
        df_tot_limma[eval_mask] = 1e10
    else:
        s2_post_limma[eval_mask] = (
            limma_d0 * s2_prior_limma + df_r[eval_mask] * s2[eval_mask]
        ) / (limma_d0 + df_r[eval_mask])
        df_tot_limma[eval_mask] = limma_d0 + df_r[eval_mask]

    limma_t = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    limma_p = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    limma_adj_p = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    limma_b = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)

    for k_idx in range(n_coefs):
        denom = np.sqrt(s2_post_limma) * base_fit.stdev_unscaled[:, k_idx]
        valid_k = eval_mask & np.isfinite(base_fit.coefficients[:, k_idx]) & np.isfinite(denom) & (denom > 0.0)
        if np.any(valid_k):
            tk = base_fit.coefficients[valid_k, k_idx] / denom[valid_k]
            limma_t[valid_k, k_idx] = tk
            df_cap = np.minimum(df_tot_limma[valid_k], 1e8)
            limma_p[valid_k, k_idx] = 2.0 * stats.t.sf(np.abs(tk), df=df_cap)
            limma_adj_p[:, k_idx] = _bh_fdr(limma_p[:, k_idx])
            limma_b[:, k_idx] = _compute_b_lods(
                limma_t[:, k_idx], base_fit.stdev_unscaled[:, k_idx], df_tot_limma
            )

    # --- 2. DEqMS Count-Dependent Empirical Bayes ---
    log2_cnt = np.log2(cnt)
    f_hat = _fit_variance_count_trend(
        log2_count=log2_cnt,
        log_var=log_s2,
        fit_mask=fit_mask,
        fit_method=fit_method,
        span=span,
    )

    detrended_res = log_s2[fit_mask] - f_hat[fit_mask]
    if use_robust:
        sca_d0, sca_emean = _fit_f_dist_robust(detrended_res, df_r[fit_mask])
    else:
        sca_d0, sca_emean = _fit_f_dist_nonrobust(detrended_res, df_r[fit_mask])

    sca_scale_shift = (
        0.0
        if sca_d0 >= 1e7
        else float(special.digamma(0.5 * sca_d0) - np.log(0.5 * sca_d0))
    )
    sca_priorvar = np.exp(f_hat + sca_emean + sca_scale_shift)

    sca_d0_vec = np.full(n_prot, sca_d0, dtype=np.float64)
    if use_robust:
        # Phipson et al. (2016) per-protein outlier prior degrees of freedom
        f_ratio = s2[eval_mask] / sca_priorvar[eval_mask]
        if sca_d0 >= 1e7:
            p_upper = stats.chi2.sf(f_ratio * df_r[eval_mask], df=df_r[eval_mask])
        else:
            p_upper = stats.f.sf(f_ratio, dfn=df_r[eval_mask], dfd=sca_d0)
        r_outlier = _bh_fdr(p_upper)
        high_var = (f_ratio > 1.0) & (r_outlier < 0.5)
        d0_eval = np.full(np.sum(eval_mask), min(sca_d0, 1e6), dtype=np.float64)
        if np.any(high_var):
            weight = np.clip(2.0 * r_outlier[high_var], 0.05 / max(sca_d0, 1.0), 1.0)
            d0_eval[high_var] = np.maximum(0.05, d0_eval[high_var] * weight)
        if sca_d0 < 1e7 or np.any(high_var):
            sca_d0_vec[eval_mask] = np.where(high_var, d0_eval, sca_d0)

    sca_postvar = np.full(n_prot, np.nan, dtype=np.float64)
    df_tot_sca = np.full(n_prot, np.nan, dtype=np.float64)
    inf_d0_mask = eval_mask & (sca_d0_vec >= 1e7)
    fin_d0_mask = eval_mask & (sca_d0_vec < 1e7)

    if np.any(inf_d0_mask):
        sca_postvar[inf_d0_mask] = sca_priorvar[inf_d0_mask]
        df_tot_sca[inf_d0_mask] = 1e10
    if np.any(fin_d0_mask):
        d0_f = sca_d0_vec[fin_d0_mask]
        dg_f = df_r[fin_d0_mask]
        sca_postvar[fin_d0_mask] = (
            d0_f * sca_priorvar[fin_d0_mask] + dg_f * s2[fin_d0_mask]
        ) / (d0_f + dg_f)
        df_tot_sca[fin_d0_mask] = d0_f + dg_f

    sca_t = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    sca_p = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    sca_adj_pval = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)
    sca_b = np.full((n_prot, n_coefs), np.nan, dtype=np.float64)

    for k_idx in range(n_coefs):
        denom_sca = np.sqrt(sca_postvar) * base_fit.stdev_unscaled[:, k_idx]
        valid_k = (
            eval_mask
            & np.isfinite(base_fit.coefficients[:, k_idx])
            & np.isfinite(denom_sca)
            & (denom_sca > 0.0)
        )
        if np.any(valid_k):
            tk = base_fit.coefficients[valid_k, k_idx] / denom_sca[valid_k]
            sca_t[valid_k, k_idx] = tk
            df_cap = np.minimum(df_tot_sca[valid_k], 1e8)
            sca_p[valid_k, k_idx] = 2.0 * stats.t.sf(np.abs(tk), df=df_cap)
            sca_adj_pval[:, k_idx] = _bh_fdr(sca_p[:, k_idx])
            sca_b[:, k_idx] = _compute_b_lods(
                sca_t[:, k_idx], base_fit.stdev_unscaled[:, k_idx], df_tot_sca
            )

    return DEqMSFit(
        coefficients=base_fit.coefficients,
        stdev_unscaled=base_fit.stdev_unscaled,
        sigma=base_fit.sigma,
        s2=base_fit.s2,
        df_residual=base_fit.df_residual,
        ave_expr=base_fit.ave_expr,
        protein_names=base_fit.protein_names,
        coef_names=base_fit.coef_names,
        default_coef=base_fit.default_coef,
        count=cnt,
        df_prior=limma_d0,
        s2_prior=s2_prior_limma,
        s2_post=s2_post_limma,
        t=limma_t,
        p_value=limma_p,
        adj_p_val=limma_adj_p,
        b=limma_b,
        sca_f_hat=f_hat,
        sca_dfprior=sca_d0,
        sca_dfprior_vec=sca_d0_vec,
        sca_priorvar=sca_priorvar,
        sca_postvar=sca_postvar,
        sca_t=sca_t,
        sca_p=sca_p,
        sca_adj_pval=sca_adj_pval,
        sca_b=sca_b,
        fit_method=fit_method,
        span=span,
        robust=use_robust,
        settings=settings,
    )


def output_result(
    fit: DEqMSFit,
    coef: int | str | None = None,
    sort_by: str | None = "sca.P.Value",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Extract a ``DEqMS::outputResult`` differential expression table.

    Columns match upstream Bioconductor ``DEqMS::outputResult``:
    ``['logFC', 'AveExpr', 't', 'P.Value', 'adj.P.Val', 'B', 'gene', 'count',
    'sca.t', 'sca.P.Value', 'sca.adj.pval']``.
    """
    validate_settings(settings)
    if not isinstance(fit, DEqMSFit):
        raise TypeError("fit must be a DEqMSFit returned by spectra_count_ebayes.")

    n_coefs = fit.coefficients.shape[1]
    if coef is None:
        col_idx = fit.default_coef
    elif isinstance(coef, str):
        if coef not in fit.coef_names:
            raise ValueError(
                f"coef={coef!r} not found in fit.coef_names={fit.coef_names}."
            )
        col_idx = fit.coef_names.index(coef)
    elif isinstance(coef, (int, np.integer)):
        col_idx = int(coef)
        if col_idx < -n_coefs or col_idx >= n_coefs:
            raise ValueError(
                f"coef index {col_idx} out of bounds for {n_coefs} coefficients."
            )
        if col_idx < 0:
            col_idx += n_coefs
    else:
        raise TypeError("coef must be None, an int column index, or a str column name.")

    count_col = (
        fit.count.astype(np.int64)
        if np.all(np.floor(fit.count) == fit.count)
        else fit.count.copy()
    )

    df = pd.DataFrame(
        {
            "logFC": fit.coefficients[:, col_idx],
            "AveExpr": fit.ave_expr,
            "t": fit.t[:, col_idx],
            "P.Value": fit.p_value[:, col_idx],
            "adj.P.Val": fit.adj_p_val[:, col_idx],
            "B": fit.b[:, col_idx],
            "gene": fit.protein_names,
            "count": count_col,
            "sca.t": fit.sca_t[:, col_idx],
            "sca.P.Value": fit.sca_p[:, col_idx],
            "sca.adj.pval": fit.sca_adj_pval[:, col_idx],
        },
        index=pd.Index(fit.protein_names, name="gene"),
    )

    if sort_by is None or sort_by == "none":
        return df

    valid_sorts = {
        "sca.P.Value": ("sca.P.Value", True),
        "sca.adj.pval": ("sca.adj.pval", True),
        "P.Value": ("P.Value", True),
        "adj.P.Val": ("adj.P.Val", True),
        "B": ("B", False),
        "logFC": ("logFC", False),
    }
    if sort_by not in valid_sorts:
        raise ValueError(
            f"Unknown sort_by={sort_by!r}. Expected None, 'none', or one of {list(valid_sorts)}."
        )

    sort_col, ascending = valid_sorts[sort_by]
    return df.sort_values(
        by=sort_col, ascending=ascending, na_position="last", kind="mergesort"
    )


class DEqMS:
    """Object-oriented DEqMS estimator for quantitative proteomics."""

    def __init__(
        self,
        fit_method: str = "loess",
        span: float = 0.75,
        robust: bool | None = None,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        if fit_method not in ("loess", "spline", "nls"):
            raise ValueError(
                f"Unknown fit_method={fit_method!r}. Expected 'loess', 'spline', or 'nls'."
            )
        if span <= 0.0:
            raise ValueError(f"span must be > 0, got {span}.")

        self.fit_method = fit_method
        self.span = span
        self.robust = robust
        self.settings: SettingsMode = settings
        self.fit_: DEqMSFit | None = None

    def fit(
        self,
        expr_or_fit: pd.DataFrame | np.ndarray | ProteinSummarizationResult | LimmaFit | dict[str, Any],
        design: Any = None,
        contrast: Any = None,
        count: np.ndarray | pd.Series | list[int] | None = None,
    ) -> DEqMS:
        """Fit per-protein linear models and count-dependent Empirical Bayes."""
        self.fit_ = spectra_count_ebayes(
            expr_or_fit=expr_or_fit,
            design=design,
            contrast=contrast,
            count=count,
            fit_method=self.fit_method,
            span=self.span,
            robust=self.robust,
            settings=self.settings,
        )
        return self

    def output_result(
        self,
        coef: int | str | None = None,
        sort_by: str | None = "sca.P.Value",
    ) -> pd.DataFrame:
        """Return formatted differential protein expression table."""
        if self.fit_ is None:
            raise RuntimeError("DEqMS.fit() must be called before output_result().")
        return output_result(
            self.fit_, coef=coef, sort_by=sort_by, settings=self.settings
        )

    def run(
        self,
        expr_or_fit: pd.DataFrame | np.ndarray | ProteinSummarizationResult | LimmaFit | dict[str, Any],
        design: Any = None,
        contrast: Any = None,
        count: np.ndarray | pd.Series | list[int] | None = None,
        coef: int | str | None = None,
        sort_by: str | None = "sca.P.Value",
    ) -> pd.DataFrame:
        """Fit DEqMS and immediately return ``output_result`` DataFrame."""
        self.fit(expr_or_fit=expr_or_fit, design=design, contrast=contrast, count=count)
        return self.output_result(coef=coef, sort_by=sort_by)

    @property
    def sca_dfprior(self) -> float:
        if self.fit_ is None:
            raise RuntimeError("DEqMS model is not fitted yet.")
        return self.fit_.sca_dfprior

    @property
    def sca_priorvar(self) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("DEqMS model is not fitted yet.")
        return self.fit_.sca_priorvar

    @property
    def sca_postvar(self) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("DEqMS model is not fitted yet.")
        return self.fit_.sca_postvar
