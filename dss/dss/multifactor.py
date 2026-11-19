"""Park & Wu (2016) DSS-general multifactor differential methylation (GLS/WLS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .dispersion import shrink_dispersion_bb
from .smoothing import SettingsMode, smooth_counts_tricube, validate_settings


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment preserving NaNs."""
    pvals = np.asarray(pvals, dtype=np.float64)
    qvals = np.full_like(pvals, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)
    m = int(np.sum(valid))
    if m == 0:
        return qvals
    pv = np.clip(pvals[valid], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate((ranked * (m / ranks))[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    qvals[valid] = q_sorted[inv]
    return qvals


@dataclass
class MultifactorFit:
    """Fitted Park & Wu (2016) arcsine Beta-Binomial GLS model across CpGs."""

    coords: pd.DataFrame
    design: pd.DataFrame
    coefficients: np.ndarray
    cov_beta: np.ndarray
    phi: np.ndarray
    weights: np.ndarray
    valid_cpg: np.ndarray
    settings: SettingsMode = "default"

    def summary(self) -> str:
        """Return a concise text summary of the multifactor methylation fit."""
        n_cpgs = len(self.coords)
        n_ok = int(np.sum(self.valid_cpg))
        med_phi = float(np.nanmedian(self.phi)) if n_cpgs > 0 else float("nan")
        return (
            f"MultifactorFit (settings={self.settings!r})\n"
            f"  CpGs modeled         : {n_ok} / {n_cpgs}\n"
            f"  Samples x Covariates : {self.design.shape[0]} x {self.design.shape[1]} "
            f"({', '.join(map(str, self.design.columns))})\n"
            f"  Median shrunken phi  : {med_phi:.4f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export coordinates, shrunken dispersions, and fitted coefficients."""
        df = self.coords[["chr", "pos"]].copy().reset_index(drop=True)
        df["phi"] = self.phi
        for j, col in enumerate(self.design.columns):
            df[f"beta_{col}"] = self.coefficients[:, j]
        return df


def _prepare_multifactor_design(
    design: pd.DataFrame | np.ndarray | pd.Series, n_samples: int
) -> pd.DataFrame:
    """Build a numeric full-rank design DataFrame of shape ``(n_samples, p)``."""
    if isinstance(design, pd.Series):
        df = design.to_frame(name=str(design.name or "X1")).copy()
    elif isinstance(design, pd.DataFrame):
        df = design.copy()
    else:
        arr = np.asarray(design)
        if arr.ndim == 1:
            df = pd.DataFrame({"X1": arr})
        elif arr.ndim == 2:
            df = pd.DataFrame(arr, columns=[f"X{j}" for j in range(arr.shape[1])])
        else:
            raise ValueError(f"Unsupported design shape {arr.shape}.")

    if len(df) != n_samples:
        raise ValueError(
            f"design has {len(df)} rows; expected {n_samples} samples."
        )

    has_cat = any(
        not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        for c in df.columns
    )
    if has_cat:
        df = pd.get_dummies(df, drop_first=True, dtype=np.float64)
    else:
        df = df.astype(np.float64)

    mat = df.to_numpy(dtype=np.float64)
    ones = np.ones(n_samples, dtype=np.float64)
    if mat.shape[1] == 0:
        return pd.DataFrame({"Intercept": ones}, index=df.index)

    c1, _, _, _ = np.linalg.lstsq(mat, ones, rcond=None)
    spans_ones = float(np.max(np.abs(mat @ c1 - ones))) < 1e-6
    if not spans_ones:
        df.insert(0, "Intercept", ones)
    return df


def dml_fit_multifactor(
    N_mat: np.ndarray | pd.DataFrame,
    X_mat: np.ndarray | pd.DataFrame,
    coords_df: pd.DataFrame,
    design: pd.DataFrame | np.ndarray | pd.Series,
    smoothing: bool = False,
    smoothing_span: float = 500.0,
    settings: SettingsMode = "default",
) -> MultifactorFit:
    """Fit Park & Wu (2016) `DSS-general` multifactor Beta-Binomial GLS model.

    Applies the variance-stabilizing ``arcsin(2 * p_hat - 1)`` link transformation
    with Empirical Bayes shrunken Beta-Binomial precision weights
    ``w_ij = N_ij / (1 + (N_ij - 1) * phi_i)``.
    """
    validate_settings(settings)
    if not {"chr", "pos"}.issubset(coords_df.columns):
        raise ValueError("coords_df must contain 'chr' and 'pos' columns.")

    n_cpgs = len(coords_df)
    n_arr = np.asarray(N_mat, dtype=np.float64)
    x_arr = np.asarray(X_mat, dtype=np.float64)
    if n_arr.shape != x_arr.shape or n_arr.ndim != 2:
        raise ValueError(
            f"N_mat shape {n_arr.shape} and X_mat shape {x_arr.shape} must match and be 2D."
        )

    n_design_rows = len(design) if hasattr(design, "__len__") else n_arr.shape[1]
    if n_arr.shape[0] != n_cpgs and n_arr.shape[1] == n_cpgs and n_arr.shape[0] == n_design_rows:
        n_arr = n_arr.T
        x_arr = x_arr.T
    if n_arr.shape[0] != n_cpgs:
        raise ValueError(
            f"N_mat shape {n_arr.shape} incompatible with {n_cpgs} CpG coordinates."
        )

    n_samples = n_arr.shape[1]
    design_df = _prepare_multifactor_design(design, n_samples=n_samples)
    z_mat = design_df.to_numpy(dtype=np.float64)
    p_cov = z_mat.shape[1]
    if n_samples <= p_cov:
        raise ValueError(
            f"Number of samples ({n_samples}) must exceed design columns ({p_cov})."
        )

    coords_clean = coords_df[["chr", "pos"]].copy().reset_index(drop=True)
    chrom = coords_clean["chr"].astype(str).to_numpy()
    pos = coords_clean["pos"].to_numpy(dtype=np.float64)

    # Methylation proportions & arcsin(2p - 1) transformation
    if smoothing:
        p_obs = smooth_counts_tricube(
            chrom, pos, x_arr, n_arr, smoothing_span=smoothing_span
        )
        p_obs = np.clip(p_obs, 1e-3, 1.0 - 1e-3)
    else:
        pos_n = n_arr > 0.0
        n_safe = np.where(pos_n, n_arr, 1.0)
        p_obs = np.where(pos_n, (x_arr + 0.01) / (n_safe + 0.02), 0.5)
        p_obs = np.clip(p_obs, 1e-4, 1.0 - 1e-4)

    y_trans = np.arcsin(np.clip(2.0 * p_obs - 1.0, -1.0, 1.0))

    # Step 1: initial coverage-weighted WLS to estimate sample-specific means p_hat0
    eye_p = np.eye(p_cov, dtype=np.float64)
    w0 = np.maximum(n_arr, 0.0)
    ztw0z = np.einsum("ij,jp,jq->ipq", w0, z_mat, z_mat) + 1e-6 * eye_p[None, :, :]
    ztw0y = (w0 * y_trans) @ z_mat
    beta0 = np.linalg.solve(ztw0z, ztw0y[..., None])[..., 0]
    y_hat0 = np.clip(beta0 @ z_mat.T, -0.5 * np.pi, 0.5 * np.pi)
    p_hat0 = np.clip(0.5 * (np.sin(y_hat0) + 1.0), 1e-4, 1.0 - 1e-4)

    # Step 2: Empirical Bayes Beta-Binomial dispersion shrinkage across samples
    sample_groups = [
        (x_arr[:, j : j + 1], n_arr[:, j : j + 1], p_hat0[:, j])
        for j in range(n_samples)
    ]
    phi_post, _, _ = shrink_dispersion_bb(sample_groups, settings=settings)

    # Step 3: Analytical Beta-Binomial precision weights & vectorized GLS
    pos_cov = n_arr > 0.0
    denom_w = 1.0 + np.maximum(n_arr - 1.0, 0.0) * phi_post[:, None]
    w_gls = np.where(pos_cov, n_arr / denom_w, 0.0)

    ztwz = np.einsum("ij,jp,jq->ipq", w_gls, z_mat, z_mat)
    ztwy = (w_gls * y_trans) @ z_mat

    n_active = np.sum(pos_cov, axis=1)
    det_vals = np.linalg.det(ztwz)
    valid_cpg = (n_active > p_cov) & np.isfinite(det_vals) & (det_vals > 1e-12)

    coefficients = np.full((n_cpgs, p_cov), np.nan, dtype=np.float64)
    cov_beta = np.full((n_cpgs, p_cov, p_cov), np.nan, dtype=np.float64)

    if np.any(valid_cpg):
        ztwz_ok = ztwz[valid_cpg]
        cov_ok = np.linalg.inv(ztwz_ok)
        beta_ok = np.einsum("ipq,iq->ip", cov_ok, ztwy[valid_cpg])
        coefficients[valid_cpg] = beta_ok
        cov_beta[valid_cpg] = cov_ok

    return MultifactorFit(
        coords=coords_clean,
        design=design_df,
        coefficients=coefficients,
        cov_beta=cov_beta,
        phi=phi_post,
        weights=w_gls,
        valid_cpg=valid_cpg,
        settings=settings,
    )


def dml_test_multifactor(
    fit: MultifactorFit,
    coef: int | str | None = None,
    contrast: list[float] | np.ndarray | pd.DataFrame | None = None,
    settings: SettingsMode | None = None,
) -> pd.DataFrame:
    """Hypothesis testing for `DSS-general` multifactor fits (`DSS::DMLtest.multiFactor`).

    Supports testing a single design column (`coef`), a 1D linear contrast vector
    (`contrast`), or a multi-degree-of-freedom 2D contrast matrix (`contrast`).
    """
    if settings is not None:
        validate_settings(settings)
    if coef is not None and contrast is not None:
        raise ValueError("Specify at most one of `coef` or `contrast`, not both.")

    p_cov = fit.design.shape[1]
    col_names = list(map(str, fit.design.columns))

    if coef is None and contrast is None:
        if p_cov == 2:
            coef = 1
        else:
            raise ValueError(
                "Please specify `coef` (column index or name) or `contrast`."
            )

    if coef is not None:
        if isinstance(coef, str):
            if coef not in col_names:
                raise ValueError(
                    f"coef={coef!r} not found in design columns {col_names}."
                )
            c_idx = col_names.index(coef)
        else:
            c_idx = int(coef)
            if c_idx < 0 or c_idx >= p_cov:
                raise ValueError(
                    f"coef index {c_idx} out of bounds for {p_cov} design columns."
                )
        c_vec = np.zeros(p_cov, dtype=np.float64)
        c_vec[c_idx] = 1.0
        return _eval_1d_contrast(fit, c_vec)

    # contrast is provided
    c_arr = (
        contrast.to_numpy(dtype=np.float64)
        if isinstance(contrast, pd.DataFrame)
        else np.asarray(contrast, dtype=np.float64)
    )
    if c_arr.ndim == 1:
        if len(c_arr) != p_cov:
            raise ValueError(
                f"1D contrast has length {len(c_arr)}; expected {p_cov}."
            )
        return _eval_1d_contrast(fit, c_arr)

    if c_arr.ndim == 2:
        # Accept either (q, p) or (p, q)
        if c_arr.shape[1] == p_cov:
            l_mat = c_arr
        elif c_arr.shape[0] == p_cov:
            l_mat = c_arr.T
        else:
            raise ValueError(
                f"2D contrast shape {c_arr.shape} incompatible with {p_cov} covariates."
            )
        if l_mat.shape[0] == 1:
            return _eval_1d_contrast(fit, l_mat[0])
        return _eval_multidf_contrast(fit, l_mat)

    raise ValueError(f"Unsupported contrast shape {c_arr.shape}.")


def _eval_1d_contrast(fit: MultifactorFit, c_vec: np.ndarray) -> pd.DataFrame:
    """Evaluate 1-df Wald z-test for a linear contrast on arcsine scale."""
    est = fit.coefficients @ c_vec
    var = np.einsum("p,ipq,q->i", c_vec, fit.cov_beta, c_vec)
    ok = fit.valid_cpg & np.isfinite(est) & np.isfinite(var) & (var > 0.0)
    se = np.where(ok, np.sqrt(np.maximum(var, 0.0)), np.nan)
    stat = np.where(ok, est / se, np.nan)
    pval = np.where(ok, 2.0 * norm.sf(np.abs(stat)), np.nan)
    fdr = _bh_fdr(pval)

    out = fit.coords[["chr", "pos"]].copy().reset_index(drop=True)
    out["diff"] = est
    out["diff_se"] = se
    out["stat"] = stat
    out["phi"] = fit.phi
    out["pval"] = pval
    out["fdr"] = fdr
    return out


def _eval_multidf_contrast(fit: MultifactorFit, l_mat: np.ndarray) -> pd.DataFrame:
    """Evaluate q-df Wald chi^2 test for a matrix contrast L @ beta = 0."""
    q_df = int(np.linalg.matrix_rank(l_mat))
    if q_df == 0:
        raise ValueError("Contrast matrix has rank 0.")
    # Reduce L to full row rank if redundant rows were supplied
    if q_df < l_mat.shape[0]:
        u, s, vt = np.linalg.svd(l_mat, full_matrices=False)
        keep = s > 1e-10
        l_mat = vt[keep]
        q_df = l_mat.shape[0]

    n_cpgs = len(fit.coords)
    stat = np.full(n_cpgs, np.nan, dtype=np.float64)
    pval = np.full(n_cpgs, np.nan, dtype=np.float64)

    ok_idx = np.flatnonzero(fit.valid_cpg)
    if len(ok_idx) > 0:
        lb = fit.coefficients[ok_idx] @ l_mat.T  # (m, q)
        lcl = np.einsum("qp,ipr,sr->iqs", l_mat, fit.cov_beta[ok_idx], l_mat)
        inv_lcl = np.linalg.inv(lcl)
        chi2_vals = np.einsum("iq,iqs,is->i", lb, inv_lcl, lb)
        chi2_vals = np.maximum(chi2_vals, 0.0)
        stat[ok_idx] = chi2_vals
        pval[ok_idx] = chi2.sf(chi2_vals, df=q_df)

    fdr = _bh_fdr(pval)
    out = fit.coords[["chr", "pos"]].copy().reset_index(drop=True)
    out["stat"] = stat
    out["df"] = q_df
    out["pval"] = pval
    out["fdr"] = fdr
    return out
