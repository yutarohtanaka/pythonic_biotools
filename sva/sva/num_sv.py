"""
Dimensionality estimation (`num_sv`) for latent surrogate variables.

Implements:
- `method="be"`: Buja & Eyuboglu (1992) permutation parallel analysis on
  `mod`-residualized data with post-permutation orthogonal re-projection.
- `method="leek"`: Leek (2011) asymptotic conditional SVD / Marchenko–Pastur +
  Tracy–Widom rank estimation adjusted for gene-wise variance heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ._io import coerce_design_matrix, coerce_expression_matrix, orthonormal_basis


@dataclass(frozen=True)
class NumSVResult:
    """Typed result container for surrogate variable rank estimation (`num_sv`)."""

    n_sv: int
    method: str
    singular_values: np.ndarray
    variance_fractions: np.ndarray
    p_values: np.ndarray | None
    thresholds: np.ndarray | None
    residual_df: int
    n_features_used: int

    def __int__(self) -> int:
        return self.n_sv

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-component singular values, variance fractions, and selection status."""
        k_max = len(self.singular_values)
        components = np.arange(1, k_max + 1, dtype=int)
        p_col = (
            self.p_values
            if self.p_values is not None
            else np.full(k_max, np.nan, dtype=np.float64)
        )
        if self.thresholds is None:
            t_col = np.full(k_max, np.nan, dtype=np.float64)
        else:
            t_col = np.broadcast_to(np.asarray(self.thresholds, dtype=np.float64), (k_max,)).copy()
        selected = components <= self.n_sv
        return pd.DataFrame(
            {
                "component": components,
                "singular_value": self.singular_values,
                "variance_fraction": self.variance_fractions,
                "p_value": p_col,
                "threshold": t_col,
                "selected": selected,
            }
        )

    def summary(self) -> str:
        """Return a human-readable summary of the `num_sv` rank estimate."""
        top_vf = (
            ", ".join(f"{vf:.3f}" for vf in self.variance_fractions[: min(5, len(self.variance_fractions))])
            if len(self.variance_fractions) > 0
            else "none"
        )
        return (
            f"NumSVResult(n_sv={self.n_sv}, method={self.method!r}, "
            f"features_used={self.n_features_used}, residual_df={self.residual_df}, "
            f"top_var_frac=[{top_vf}])"
        )


def _tw1_quantile(sig_level: float) -> float:
    """Return GOE Tracy–Widom (beta=1) upper-tail quantile for significance level `sig_level`."""
    if sig_level <= 0.01:
        return 2.023 + 1.2 * max(0.0, float(-np.log10(sig_level / 0.01)))
    if sig_level < 0.05:
        return 0.979 + (2.023 - 0.979) * ((0.05 - sig_level) / 0.04)
    if sig_level <= 0.15:
        return 0.979
    return 0.450


def _silverstein_mp_tw_edge(v_row: np.ndarray, d_res: int, t_alpha: float) -> float:
    """Compute generalized Marchenko–Pastur + Tracy–Widom upper edge for `s_1^2`.

    For an `(m_u, d_res)` residual matrix with independent Gaussian rows of
    variances `v_row = (v_1, ..., v_{m_u})`, solves the Silverstein (1995)
    fixed-point equation for the upper spectral edge `z_0` of `R^T R / d_res`
    and computes the El Karoui (2007) Tracy–Widom (beta=1) fluctuation scale
    from `(1/2) z''(m_0)`. Returns `d_res * (z_0 + t_alpha * sigma_TW)`.
    """
    v = np.asarray(v_row, dtype=np.float64)
    v = v[v > 0.0]
    if v.size == 0 or d_res < 1:
        return 0.0

    vmax = float(np.max(v))
    vn = v / vmax
    one_minus_vn = 1.0 - vn
    inv_n = 1.0 / float(d_res)

    def _f(u: float) -> float:
        eu = float(np.exp(u))
        w = vn / (one_minus_vn + eu)
        return 1.0 - inv_n * float(np.dot(w, w))

    eu_init = max(np.sqrt(float(np.sum(vn)) * inv_n), 1e-6)
    u_guess = float(np.log(eu_init))
    ua = min(u_guess - 2.0, -5.0)
    while _f(ua) >= 0.0 and ua > -45.0:
        ua -= 5.0
    ub = max(u_guess + 2.0, 5.0)
    while _f(ub) <= 0.0 and ub < 45.0:
        ub += 5.0

    if _f(ua) >= 0.0:
        u0 = ua
    elif _f(ub) <= 0.0:
        u0 = ub
    else:
        u0 = float(brentq(_f, ua, ub, xtol=1e-10, rtol=1e-10, maxiter=100))

    eu0 = float(np.exp(u0))
    w0 = vn / (one_minus_vn + eu0)
    one_plus_eu = 1.0 + eu0
    z0 = one_plus_eu * (1.0 + inv_n * float(np.sum(w0)))
    sigma_tw = (one_plus_eu / (float(d_res) ** (2.0 / 3.0))) * (
        (1.0 + inv_n * float(np.sum(w0 ** 3))) ** (1.0 / 3.0)
    )
    return float(d_res) * vmax * (z0 + t_alpha * sigma_tw)


def num_sv_arrays(
    Y: np.ndarray,
    X1: np.ndarray,
    *,
    method: str = "be",
    vfilter: int | None = None,
    B: int = 20,
    sig_level: float = 0.10,
    seed: int | np.random.Generator | None = None,
) -> NumSVResult:
    """Internal `num_sv` implementation operating on validated `(m, n)` and `(n, p)` arrays."""
    if method not in {"be", "leek"}:
        raise ValueError(f"Unsupported num_sv method {method!r}; expected 'be' or 'leek'.")
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Input matrix contains NaN or Inf values; filter or impute incomplete features before SVA."
        )
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}.")
    if not (0.0 < sig_level < 1.0):
        raise ValueError(f"sig_level must be in (0, 1), got {sig_level}.")

    m, n = Y.shape
    Q1, r1 = orthonormal_basis(X1)
    if r1 < 1:
        raise ValueError("Design matrix 'mod' has zero numerical rank.")
    d_e = n - r1
    if d_e < 2:
        raise ValueError(
            f"Insufficient residual degrees of freedom for num_sv: n - rank(mod) = {d_e} < 2."
        )

    Y_use = Y
    if vfilter is not None:
        if vfilter < 2:
            raise ValueError(f"vfilter must be >= 2, got {vfilter}.")
        k_keep = min(int(vfilter), m)
        row_var = np.var(Y_use, axis=1, ddof=1)
        top_idx = np.sort(np.argsort(-row_var, kind="mergesort")[:k_keep])
        Y_use = Y_use[top_idx]

    R = Y_use - (Y_use @ Q1) @ Q1.T
    y_row_ss = np.einsum("ij,ij->i", Y_use, Y_use)
    row_ss = np.einsum("ij,ij->i", R, R)
    keep_rows = (y_row_ss > 0.0) & (row_ss > 1e-24 * y_row_ss)
    R = R[keep_rows]

    m_u = R.shape[0]
    if m_u < 2:
        return NumSVResult(
            n_sv=0,
            method=method,
            singular_values=np.empty(0, dtype=np.float64),
            variance_fractions=np.empty(0, dtype=np.float64),
            p_values=np.empty(0, dtype=np.float64) if method == "be" else None,
            thresholds=np.empty(0, dtype=np.float64),
            residual_df=d_e,
            n_features_used=m_u,
        )

    k_max = min(m_u, d_e)
    if method == "be":
        s = np.linalg.svd(R, full_matrices=False, compute_uv=False)[:k_max]
        U_svd = None
    else:
        U_full, s_full, _ = np.linalg.svd(R, full_matrices=False)
        s = s_full[:k_max]
        U_svd = U_full[:, :k_max]

    s2 = s ** 2
    ss_total = float(np.sum(s2))
    if ss_total <= 0.0:
        return NumSVResult(
            n_sv=0,
            method=method,
            singular_values=s,
            variance_fractions=np.zeros_like(s),
            p_values=np.ones_like(s) if method == "be" else None,
            thresholds=np.full(k_max, sig_level if method == "be" else np.nan, dtype=np.float64),
            residual_df=d_e,
            n_features_used=m_u,
        )

    var_frac = s2 / ss_total

    if method == "be":
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        null_frac = np.empty((B, k_max), dtype=np.float64)
        for b in range(B):
            perm_idx = np.argsort(rng.random(R.shape), axis=1)
            R_perm = np.take_along_axis(R, perm_idx, axis=1)
            R_res = R_perm - (R_perm @ Q1) @ Q1.T
            s_b = np.linalg.svd(R_res, full_matrices=False, compute_uv=False)[:k_max]
            s_b2 = s_b ** 2
            ss_b = float(np.sum(s_b2))
            null_frac[b] = s_b2 / ss_b if ss_b > 0.0 else 0.0

        p_values = np.mean(null_frac >= var_frac[None, :], axis=0)
        n_sv = 0
        for k in range(k_max - 1):
            if p_values[k] <= sig_level:
                n_sv += 1
            else:
                break

        return NumSVResult(
            n_sv=n_sv,
            method="be",
            singular_values=s,
            variance_fractions=var_frac,
            p_values=p_values,
            thresholds=np.full(k_max, float(sig_level), dtype=np.float64),
            residual_df=d_e,
            n_features_used=m_u,
        )

    # method == "leek": deflated generalized Marchenko–Pastur + Tracy–Widom upper edge
    assert U_svd is not None
    lam = s2 / float(m_u)
    thresholds = np.full(k_max, np.nan, dtype=np.float64)
    t_alpha = _tw1_quantile(sig_level)

    n_sv = 0
    accepting = True
    for k in range(k_max - 1):
        d_ek = d_e - k
        v_gk = ((U_svd[:, k:] ** 2) @ s2[k:]) / float(d_ek)
        edge_s2 = _silverstein_mp_tw_edge(v_gk, d_ek, t_alpha)
        lam_edge = edge_s2 / float(m_u)
        thresholds[k] = lam_edge

        if accepting:
            if lam[k] > lam_edge:
                n_sv += 1
            else:
                accepting = False

    return NumSVResult(
        n_sv=n_sv,
        method="leek",
        singular_values=s,
        variance_fractions=var_frac,
        p_values=None,
        thresholds=thresholds,
        residual_df=d_e,
        n_features_used=m_u,
    )


def num_sv(
    dat: Any,
    mod: Any,
    method: str = "be",
    vfilter: int | None = None,
    B: int = 20,
    sig_level: float = 0.10,
    seed: int | np.random.Generator | None = None,
    *,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> NumSVResult:
    """Estimate the number of latent surrogate variables orthogonal to `mod`.

    Parameters
    ----------
    dat:
        Expression matrix (`ndarray`, `pd.DataFrame`, `scipy.sparse`, or `AnnData`).
    mod:
        Primary model design matrix `(n_samples, p)` or `AnnData.obs` column(s).
    method:
        `"be"` (Buja–Eyuboglu permutation parallel analysis) or `"leek"`
        (Leek 2011 asymptotic conditional SVD / RMT rank estimator).
    vfilter:
        Optional number of highest-variance features to retain prior to SVD.
    B:
        Number of permutations for `method="be"`.
    sig_level:
        Significance level for sequential component selection.
    seed:
        Optional RNG seed or `np.random.Generator` for reproducible `"be"` permutations.
    features_as_rows:
        Orientation flag for `ndarray`/`DataFrame`/`sparse` inputs (ignored for `AnnData`).
    layer:
        Optional `AnnData.layers[layer]` name when `dat` is an `AnnData` object.
    """
    Y, _, _, adata_ref = coerce_expression_matrix(
        dat, features_as_rows=features_as_rows, layer=layer
    )
    X1 = coerce_design_matrix(mod, Y.shape[1], adata_ref=adata_ref, name="mod")
    return num_sv_arrays(
        Y,
        X1,
        method=method,
        vfilter=vfilter,
        B=B,
        sig_level=sig_level,
        seed=seed,
    )
