"""
Hypothesis testing and empirical-Bayes local FDR routines for `sva`.

Implements:
- `FTestResult` & `f_pvalue`: vectorized nested-OLS F-test across features with
  per-feature missing-value fallback and `NaN`-aware Benjamini–Hochberg FDR.
- `edge_lfdr`: Storey (2003) pi0 estimation and probit-space Gaussian KDE local
  false discovery rate (`sva:::edge.lfdr`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.stats import f as f_dist
from scipy.stats import false_discovery_control, gaussian_kde, norm

from ._io import (
    SettingsMode,
    coerce_design_matrix,
    coerce_expression_matrix,
    is_nested_subspace,
    orthonormal_basis,
    validate_nested_models,
    validate_settings,
)


@dataclass(frozen=True)
class FTestResult:
    """Typed result container for nested-model feature-wise F-tests (`f_pvalue`)."""

    feature_names: list[str]
    f_stat: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    df1: int | np.ndarray
    df2: int | np.ndarray

    def to_dataframe(self, sort_by_pvalue: bool = False) -> pd.DataFrame:
        """Return per-feature F-test statistics, p-values, and BH q-values."""
        m = len(self.feature_names)
        df1_arr = np.broadcast_to(np.asarray(self.df1), (m,)).copy()
        df2_arr = np.broadcast_to(np.asarray(self.df2), (m,)).copy()
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "f_stat": self.f_stat,
                "p_value": self.p_value,
                "q_value": self.q_value,
                "df1": df1_arr,
                "df2": df2_arr,
            }
        )
        if sort_by_pvalue:
            df = df.sort_values("p_value", kind="mergesort", na_position="last").reset_index(
                drop=True
            )
        return df

    def significant(self, alpha: float = 0.05, use_qvalue: bool = True) -> pd.DataFrame:
        """Return rows with `q_value <= alpha` (or `p_value <= alpha`), sorted by p-value."""
        df = self.to_dataframe(sort_by_pvalue=True)
        col = "q_value" if use_qvalue else "p_value"
        mask = df[col].notna() & (df[col] <= alpha)
        return df.loc[mask].reset_index(drop=True)

    def summary(self) -> str:
        """Return a human-readable summary of the nested F-test results."""
        m = len(self.feature_names)
        valid = np.isfinite(self.p_value)
        n_valid = int(np.sum(valid))
        n_sig_q = int(np.sum(np.isfinite(self.q_value) & (self.q_value <= 0.05)))
        n_sig_p = int(np.sum(valid & (self.p_value <= 0.05)))
        min_p = float(np.nanmin(self.p_value)) if n_valid > 0 else float("nan")
        min_q = float(np.nanmin(self.q_value)) if n_valid > 0 else float("nan")
        df1_disp = int(self.df1) if np.ndim(self.df1) == 0 else "varying"
        df2_disp = int(self.df2) if np.ndim(self.df2) == 0 else "varying"
        return (
            f"FTestResult(features={m}, valid={n_valid}, df=({df1_disp}, {df2_disp}), "
            f"p<=0.05: {n_sig_p}, q<=0.05: {n_sig_q}, min_p={min_p:.3e}, min_q={min_q:.3e})"
        )


def f_pvalue_arrays(
    Y: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray | None,
    feature_names: list[str],
    *,
    Q1: np.ndarray | None = None,
    r1: int | None = None,
    Q0: np.ndarray | None = None,
    r0: int | None = None,
) -> FTestResult:
    """Internal vectorized nested-model F-test operating directly on validated arrays."""
    m, n = Y.shape
    if Q1 is None or r1 is None or Q0 is None or r0 is None or X0 is None:
        X1, X0_arr, Q1_u, r1_u, Q0_u, r0_u = validate_nested_models(
            X1, X0, n, require_strict_nesting=True, min_resid_df=1
        )
    else:
        X0_arr = X0
        Q1_u, r1_u, Q0_u, r0_u = Q1, r1, Q0, r0

    df1_full = r1_u - r0_u
    df2_full = n - r1_u

    f_stat = np.full(m, np.nan, dtype=np.float64)
    p_value = np.full(m, np.nan, dtype=np.float64)

    all_finite = np.all(np.isfinite(Y), axis=1)
    if np.any(all_finite):
        Y_c = Y[all_finite]
        R1 = Y_c - (Y_c @ Q1_u) @ Q1_u.T
        R0 = Y_c - (Y_c @ Q0_u) @ Q0_u.T

        rss1 = np.einsum("ij,ij->i", R1, R1)
        rss0 = np.einsum("ij,ij->i", R0, R0)
        row_norm2 = np.einsum("ij,ij->i", Y_c, Y_c)

        degen = (row_norm2 <= 0.0) | (
            (rss1 <= 1e-24 * row_norm2) & (rss0 <= 1e-24 * row_norm2)
        )
        exact_fit = (~degen) & (rss1 <= 1e-28 * row_norm2)
        regular = (~degen) & (~exact_fit)

        f_c = np.full(Y_c.shape[0], np.nan, dtype=np.float64)
        p_c = np.full(Y_c.shape[0], np.nan, dtype=np.float64)

        if np.any(exact_fit):
            f_c[exact_fit] = np.inf
            p_c[exact_fit] = 0.0

        if np.any(regular):
            drss = np.maximum(rss0[regular] - rss1[regular], 0.0)
            f_reg = (drss / df1_full) / (rss1[regular] / df2_full)
            f_c[regular] = f_reg
            p_c[regular] = np.clip(f_dist.sf(f_reg, df1_full, df2_full), 0.0, 1.0)

        f_stat[all_finite] = f_c
        p_value[all_finite] = p_c

    has_incomplete = bool(np.any(~all_finite))
    df1_out: int | np.ndarray
    df2_out: int | np.ndarray

    if not has_incomplete:
        df1_out = int(df1_full)
        df2_out = int(df2_full)
    else:
        df1_arr = np.full(m, float(df1_full), dtype=np.float64)
        df2_arr = np.full(m, float(df2_full), dtype=np.float64)
        for g in np.flatnonzero(~all_finite):
            mask = np.isfinite(Y[g])
            n_g = int(np.sum(mask))
            if n_g < 2:
                df1_arr[g] = np.nan
                df2_arr[g] = np.nan
                continue
            X1_g = X1[mask]
            X0_g = X0_arr[mask]
            Q1_g, r1_g = orthonormal_basis(X1_g)
            Q0_g, r0_g = orthonormal_basis(X0_g)
            df1_g = r1_g - r0_g
            df2_g = n_g - r1_g
            if df1_g < 1 or df2_g < 1 or not is_nested_subspace(X0_g, Q1_g):
                df1_arr[g] = np.nan
                df2_arr[g] = np.nan
                continue

            y_g = Y[g, mask]
            r1_vec = y_g - Q1_g @ (Q1_g.T @ y_g)
            r0_vec = y_g - Q0_g @ (Q0_g.T @ y_g)
            rss1_g = float(np.dot(r1_vec, r1_vec))
            rss0_g = float(np.dot(r0_vec, r0_vec))
            row_norm2_g = float(np.dot(y_g, y_g))

            df1_arr[g] = float(df1_g)
            df2_arr[g] = float(df2_g)

            if row_norm2_g <= 0.0 or (
                rss1_g <= 1e-24 * row_norm2_g and rss0_g <= 1e-24 * row_norm2_g
            ):
                continue
            if rss1_g <= 1e-28 * row_norm2_g:
                f_stat[g] = np.inf
                p_value[g] = 0.0
                continue
            drss_g = max(rss0_g - rss1_g, 0.0)
            f_g = (drss_g / df1_g) / (rss1_g / df2_g)
            f_stat[g] = f_g
            p_value[g] = float(np.clip(f_dist.sf(f_g, df1_g, df2_g), 0.0, 1.0))

        df1_out = df1_arr
        df2_out = df2_arr

    q_value = np.full(m, np.nan, dtype=np.float64)
    valid = np.isfinite(p_value)
    if np.any(valid):
        q_value[valid] = false_discovery_control(np.clip(p_value[valid], 0.0, 1.0), method="bh")

    return FTestResult(
        feature_names=list(feature_names),
        f_stat=f_stat,
        p_value=p_value,
        q_value=q_value,
        df1=df1_out,
        df2=df2_out,
    )


def f_pvalue(
    dat: Any,
    mod: Any,
    mod0: Any | None = None,
    *,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> FTestResult:
    """Compute feature-wise nested-OLS F-tests comparing full model `mod` vs null `mod0`.

    Parameters
    ----------
    dat:
        Expression matrix (`ndarray`, `pd.DataFrame`, `scipy.sparse`, or `AnnData`).
        `AnnData` is always `obs x var` (`n_samples x n_features`); `DataFrame`/`ndarray`
        default to `(n_features, n_samples)` when `features_as_rows=True`.
    mod:
        Full design matrix of shape `(n_samples, p1)`, or column name(s) in `AnnData.obs`.
    mod0:
        Nested null design matrix of shape `(n_samples, p0)`, `AnnData.obs` column(s),
        or `None` (defaults to an intercept-only column of ones).
    features_as_rows:
        Orientation flag for `ndarray`/`DataFrame`/`sparse` inputs (ignored for `AnnData`).
    layer:
        Optional `AnnData.layers[layer]` name when `dat` is an `AnnData` object.
    """
    Y, feature_names, _, adata_ref = coerce_expression_matrix(
        dat, features_as_rows=features_as_rows, layer=layer
    )
    n_samples = Y.shape[1]
    X1 = coerce_design_matrix(mod, n_samples, adata_ref=adata_ref, name="mod")
    X0 = (
        coerce_design_matrix(mod0, n_samples, adata_ref=adata_ref, name="mod0")
        if mod0 is not None
        else None
    )
    return f_pvalue_arrays(Y, X1, X0, feature_names)


def estimate_pi0(
    p_values: np.ndarray,
    *,
    settings: SettingsMode = "default",
) -> float:
    """Estimate the proportion of true null hypotheses `pi0` (Storey & Tibshirani 2003)."""
    eff_mode = validate_settings(settings)
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    G = p.size
    if G == 0:
        return 1.0
    if eff_mode == "R_default":
        return float(np.clip(np.mean(p >= 0.8) / 0.2, 0.0, 1.0))
    if G < 20:
        return float(np.clip(2.0 * np.mean(p), 0.0, 1.0))

    lambdas = np.arange(0.05, 0.91, 0.05, dtype=np.float64)
    pi0_lam = np.array([np.mean(p > lam) / (1.0 - lam) for lam in lambdas], dtype=np.float64)

    try:
        coefs = np.polyfit(lambdas, pi0_lam, deg=3)
        pi0_est = float(np.polyval(coefs, 0.90))
    except (np.linalg.LinAlgError, ValueError):
        pi0_est = float(np.median(pi0_lam[-3:]))

    if not np.isfinite(pi0_est):
        pi0_est = float(pi0_lam[len(lambdas) // 2])

    return float(np.clip(pi0_est, 0.0, 1.0))


def edge_lfdr(
    p_values: np.ndarray,
    *,
    trunc: bool = True,
    monotone: bool = True,
    adj: float = 1.5,
    eps: float = 1e-8,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Compute empirical-Bayes local false discovery rates via probit-space KDE.

    When `settings="default"`, uses Storey's cubic-polynomial `pi0` estimator over
    `lambda in [0.05..0.90]` and direct probit-space Gaussian KDE evaluation.
    When `settings="R_default"`, matches Bioconductor `sva:::edge.lfdr` (`helper.R`):
    estimates `pi0 = min(1.0, mean(p >= 0.8) / 0.2)` (fixed `lambda = 0.8`) and
    smooths the probit log-density ratio `log(pi0 * phi(z) / g(z))` via GCV cubic
    smoothing spline interpolation over the KDE grid.
    """
    eff_mode = validate_settings(settings)
    p_arr = np.asarray(p_values, dtype=np.float64)
    out = np.full(p_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p_arr)
    if not np.any(valid):
        return out

    p_valid = p_arr[valid]
    if eff_mode == "R_default":
        pi0 = float(min(1.0, max(0.0, float(np.mean(p_valid >= 0.8)) / 0.2)))
    else:
        pi0 = estimate_pi0(np.clip(p_valid, eps, 1.0 - eps), settings="default")

    if pi0 <= 0.0:
        out[valid] = 0.0
        return out

    p = np.clip(p_valid, eps, 1.0 - eps)
    G = p.size
    z = norm.ppf(p)
    std_z = float(np.std(z, ddof=1)) if G > 1 else 0.0
    if G < 3 or std_z < 1e-10:
        lfdr = np.full(G, pi0, dtype=np.float64)
    elif eff_mode == "R_default":
        try:
            q75, q25 = np.percentile(z, [75.0, 25.0])
            iqr_z = float(q75 - q25)
            lo = min(std_z, iqr_z / 1.34) if iqr_z > 0.0 else std_z
            bw = max(0.9 * lo * (G ** (-0.2)) * adj, 1e-6 * max(std_z, 1.0))
            bw_factor = max(bw / std_z, 1e-6)
            kde = gaussian_kde(z, bw_method=bw_factor)
            x_grid = np.linspace(float(np.min(z)) - 3.0 * bw, float(np.max(z)) + 3.0 * bw, 512)
            g_grid = np.maximum(kde.evaluate(x_grid), 1e-300)
            f0_grid = np.maximum(pi0 * norm.pdf(x_grid), 1e-300)
            log_ratio_grid = np.log(f0_grid) - np.log(g_grid)
            spline = make_smoothing_spline(x_grid, log_ratio_grid)
            log_ratio_eval = np.clip(spline(z), -700.0, 700.0)
            lfdr = np.exp(log_ratio_eval)
        except (np.linalg.LinAlgError, ValueError):
            lfdr = np.full(G, pi0, dtype=np.float64)
    else:
        try:
            kde = gaussian_kde(z, bw_method=lambda k: max(float(k.scotts_factor()) * adj, 1e-6))
            g_z = kde.evaluate(z)
            lfdr = (pi0 * norm.pdf(z)) / np.maximum(g_z, 1e-300)
        except (np.linalg.LinAlgError, ValueError):
            lfdr = np.full(G, pi0, dtype=np.float64)

    if trunc:
        lfdr = np.clip(lfdr, 0.0, 1.0)

    if monotone and G > 1:
        order = np.argsort(p, kind="mergesort")
        lfdr_sorted = np.maximum.accumulate(lfdr[order])
        lfdr_mono = np.empty_like(lfdr_sorted)
        lfdr_mono[order] = lfdr_sorted
        lfdr = lfdr_mono

    out[valid] = lfdr
    return out
