"""
Pythonic reimplementation of limma's empirical Bayes variance moderation
(Smyth 2004, Statistical Applications in Genetics and Molecular Biology).

Implements exact moment-matching (`fitFDist` / `squeezeVar`) on a scaled
inverse-chi-squared prior via trigamma inversion, providing:
  - `moderated_ttest`: fast two-group differential analysis across features
  - `limma_top_table`: tidy DataFrame output mirroring R's `limma::topTable`
  - `moderated_lm_contrast`: general linear model design + contrast moderation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, polygamma
from scipy.stats import false_discovery_control
from scipy.stats import t as t_dist


@dataclass(frozen=True)
class ModeratedTestResult:
    """Result container for limma-style empirical Bayes moderated hypothesis testing."""

    feature_names: list[str]
    log_fc: np.ndarray
    ave_expr: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    adj_p_val: np.ndarray
    var_prior: float
    df_prior: float
    var_post: np.ndarray

    def to_dataframe(self, sort_by_pvalue: bool = True) -> pd.DataFrame:
        """Format results as a `limma::topTable`-compatible DataFrame."""
        df = pd.DataFrame(
            {
                "logFC": self.log_fc,
                "AveExpr": self.ave_expr,
                "t": self.t_stat,
                "P.Value": self.p_value,
                "adj.P.Val": self.adj_p_val,
                "s2_post": self.var_post,
            },
            index=self.feature_names,
        )
        if sort_by_pvalue:
            df = df.sort_values("P.Value", kind="mergesort")
        return df


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Invert the trigamma function y -> polygamma(1, y) on (0, inf)."""
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(arr)
    for i, xi in enumerate(arr):
        if not np.isfinite(xi) or xi <= 0.0:
            out[i] = np.inf
            continue
        if xi > 1e7:
            out[i] = 1.0 / np.sqrt(xi)
            continue
        if xi < 1e-6:
            out[i] = 1.0 / xi + 0.5
            continue
        out[i] = brentq(lambda y: polygamma(1, y) - xi, 1e-8, 1e8, xtol=1e-12, rtol=1e-10)
    return out


def fit_f_dist(var_g: np.ndarray, df1: int | float) -> tuple[float, float]:
    """Moment-match a scaled-inverse-chi-squared prior to per-feature sample variances."""
    v = np.asarray(var_g, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size < 2:
        s20 = float(v[0]) if v.size == 1 else 1.0
        return s20, float("inf")

    z = np.log(v)
    e = z - digamma(df1 / 2.0) + np.log(df1 / 2.0)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / (v.size - 1) - polygamma(1, df1 / 2.0))
    if evar > 0.0:
        df0 = float(2.0 * trigamma_inverse(evar)[0])
        s20 = float(np.exp(emean + digamma(df0 / 2.0) - np.log(df0 / 2.0)))
    else:
        df0 = float("inf")
        s20 = float(np.exp(emean))
    return s20, df0


def squeeze_var(var_g: np.ndarray, df1: int | float) -> tuple[np.ndarray, float, float, float]:
    """Shrink per-feature sample variances toward a fitted empirical Bayes prior."""
    v = np.asarray(var_g, dtype=np.float64)
    s20, df0 = fit_f_dist(v, df1)
    if np.isfinite(df0):
        var_post = (df0 * s20 + df1 * v) / (df0 + df1)
        df_total = float(df0 + df1)
    else:
        var_post = np.full_like(v, s20, dtype=np.float64)
        df_total = float("inf")
    return var_post, s20, df0, df_total


def moderated_ttest(
    X: np.ndarray | pd.DataFrame,
    group: Sequence[bool | int],
    feature_names: Sequence[str] | None = None,
) -> ModeratedTestResult:
    """Two-group empirical Bayes moderated t-test (`limma::eBayes`) across columns of `X`."""
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns] if feature_names is None else [str(f) for f in feature_names]
        mat = X.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(X, dtype=np.float64)
        names = [f"feature_{i}" for i in range(mat.shape[1])] if feature_names is None else [str(f) for f in feature_names]

    grp = np.asarray(group, dtype=bool)
    if mat.shape[0] != grp.size:
        raise ValueError(f"X has {mat.shape[0]} rows, but group has length {grp.size}.")

    n1 = int((~grp).sum())
    n2 = int(grp.sum())
    df1 = n1 + n2 - 2
    if n1 < 1 or n2 < 1 or df1 < 1:
        raise ValueError("Need at least 2 total degrees of freedom and >=1 sample per group.")

    x1, x2 = mat[~grp], mat[grp]
    mean1, mean2 = x1.mean(axis=0), x2.mean(axis=0)
    ave_expr = mat.mean(axis=0)
    ss1 = ((x1 - mean1) ** 2).sum(axis=0)
    ss2 = ((x2 - mean2) ** 2).sum(axis=0)
    var_g = (ss1 + ss2) / df1

    s20, df0 = fit_f_dist(var_g, df1)
    if np.isfinite(df0):
        var_post = (df0 * s20 + df1 * var_g) / (df0 + df1)
        df_total = df0 + df1
    else:
        var_post = var_g
        df_total = float(df1)

    se = np.sqrt(np.maximum(var_post * (1.0 / n1 + 1.0 / n2), 1e-300))
    diff = mean2 - mean1
    t_stat = diff / se
    p_val = 2.0 * t_dist.sf(np.abs(t_stat), df_total)
    p_clean = np.where(np.isfinite(p_val), np.clip(p_val, 0.0, 1.0), 1.0)
    adj_p = false_discovery_control(p_clean, method="bh") if p_clean.size > 0 else np.array([])

    return ModeratedTestResult(
        feature_names=names,
        log_fc=diff,
        ave_expr=ave_expr,
        t_stat=t_stat,
        p_value=p_clean,
        adj_p_val=adj_p,
        var_prior=s20,
        df_prior=df0,
        var_post=var_post,
    )


def moderated_lm_contrast(
    Y: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
    contrast: Sequence[float],
    feature_names: Sequence[str] | None = None,
) -> ModeratedTestResult:
    """General linear model `limma::lmFit` + `contrasts.fit` + `eBayes` across features."""
    if isinstance(Y, pd.DataFrame):
        names = [str(c) for c in Y.columns] if feature_names is None else [str(f) for f in feature_names]
        ymat = Y.to_numpy(dtype=np.float64)
    else:
        ymat = np.asarray(Y, dtype=np.float64)
        names = [f"feature_{i}" for i in range(ymat.shape[1])] if feature_names is None else [str(f) for f in feature_names]

    X = design.to_numpy(dtype=np.float64) if isinstance(design, pd.DataFrame) else np.asarray(design, dtype=np.float64)
    c = np.asarray(contrast, dtype=np.float64).ravel()
    n, p = X.shape
    if ymat.shape[0] != n:
        raise ValueError(f"Y has {ymat.shape[0]} samples, but design has {n} rows.")
    if c.size != p:
        raise ValueError(f"Contrast length ({c.size}) must match design columns ({p}).")

    rank = int(np.linalg.matrix_rank(X))
    df1 = n - rank
    if df1 < 1:
        raise ValueError(f"Residual degrees of freedom must be >= 1 (got n={n}, rank={rank}).")

    xtx_inv = np.linalg.pinv(X.T @ X)
    betas = xtx_inv @ (X.T @ ymat)
    fitted = X @ betas
    resid = ymat - fitted
    var_g = np.sum(resid**2, axis=0) / df1

    s20, df0 = fit_f_dist(var_g, df1)
    if np.isfinite(df0):
        var_post = (df0 * s20 + df1 * var_g) / (df0 + df1)
        df_total = df0 + df1
    else:
        var_post = var_g
        df_total = float(df1)

    unscaled_var = float(c @ xtx_inv @ c)
    if unscaled_var <= 0.0:
        raise ValueError("Contrast vector has zero variance under the supplied design matrix.")

    est = (c @ betas).ravel()
    se = np.sqrt(np.maximum(var_post * unscaled_var, 1e-300))
    t_stat = est / se
    p_val = 2.0 * t_dist.sf(np.abs(t_stat), df_total)
    p_clean = np.where(np.isfinite(p_val), np.clip(p_val, 0.0, 1.0), 1.0)
    adj_p = false_discovery_control(p_clean, method="bh") if p_clean.size > 0 else np.array([])

    return ModeratedTestResult(
        feature_names=names,
        log_fc=est,
        ave_expr=ymat.mean(axis=0),
        t_stat=t_stat,
        p_value=p_clean,
        adj_p_val=adj_p,
        var_prior=s20,
        df_prior=df0,
        var_post=var_post,
    )


def limma_top_table(
    X: np.ndarray | pd.DataFrame,
    group: Sequence[bool | int],
    feature_names: Sequence[str] | None = None,
    sort_by_pvalue: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper returning a `limma::topTable` DataFrame for a two-group comparison."""
    return moderated_ttest(X, group, feature_names=feature_names).to_dataframe(sort_by_pvalue=sort_by_pvalue)
