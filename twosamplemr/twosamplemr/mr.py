"""
Summary-statistics Univariable & Multivariable Mendelian Randomization (`TwoSampleMR` / `MVMR`).

Implements:
  - Multiplicative random-effects Inverse-Variance Weighted MR (`ivw_mr`) + Cochran's Q
  - Coding-invariant MR-Egger regression (`mr_egger`, orienting beta_exp >= 0)
  - Bowden et al. (2016) Weighted Median MR (`weighted_median_mr`)
  - Multivariable MR (`multivariable_mr`) with Sanderson-Windmeijer conditional F-statistics
  - Full multi-estimator summary runner (`run_mr_suite`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _validate_settings(settings: str) -> Literal["default", "R_default"]:
    if not isinstance(settings, str) or settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings {settings!r}; expected 'default' or 'R_default'."
        )
    return "R_default" if settings == "R_default" else "default"


@dataclass(frozen=True)
class MRSummaryResult:
    """Comprehensive univariable & multivariable Mendelian Randomization output."""

    exposure_name: str
    outcome_name: str
    n_instruments: int
    mean_f_stat: float
    conditional_f_stat: float
    beta_ivw: float
    se_ivw: float
    pval_ivw: float
    cochran_q: float
    cochran_q_pval: float
    beta_egger: float
    se_egger: float
    pval_egger: float
    egger_intercept: float
    se_egger_intercept: float
    pval_egger_intercept: float
    beta_weighted_median: float
    se_weighted_median: float
    pval_weighted_median: float
    beta_mvmr: float | None = None
    se_mvmr: float | None = None
    pval_mvmr: float | None = None

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"method": "IVW (random-effects)", "beta": self.beta_ivw, "se": self.se_ivw, "p_value": self.pval_ivw},
            {"method": "Weighted Median", "beta": self.beta_weighted_median, "se": self.se_weighted_median, "p_value": self.pval_weighted_median},
            {"method": "MR-Egger (slope)", "beta": self.beta_egger, "se": self.se_egger, "p_value": self.pval_egger},
            {"method": "MR-Egger (intercept)", "beta": self.egger_intercept, "se": self.se_egger_intercept, "p_value": self.pval_egger_intercept},
        ]
        if self.beta_mvmr is not None:
            rows.append({"method": "Multivariable MR (MVMR)", "beta": self.beta_mvmr, "se": self.se_mvmr, "p_value": self.pval_mvmr})
        return pd.DataFrame(rows)


def ivw_mr(
    beta_exp: Sequence[float] | np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    settings: Literal["default", "R_default"] = "default",
) -> dict[str, float]:
    """Multiplicative random-effects Inverse-Variance Weighted (IVW) MR + Cochran's Q."""
    _validate_settings(settings)
    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    k = bx.size
    if k < 1 or by.size != k or sy.size != k:
        raise ValueError("beta_exp, beta_out, se_out must share the same length >= 1.")

    w = 1.0 / (sy**2)
    denom = float(np.sum(w * (bx**2)))
    if denom <= 0.0:
        raise ValueError("Exposure effects are identically zero.")
    beta = float(np.sum(w * bx * by) / denom)
    resid = by - beta * bx
    cochran_q = float(np.sum(w * (resid**2)))
    df = max(k - 1, 1)
    q_pval = float(stats.chi2.sf(cochran_q, df=df)) if k > 1 else 1.0
    sigma2_re = max(1.0, cochran_q / df) if k > 1 else 1.0
    se = float(np.sqrt(sigma2_re / denom))
    pval = float(2.0 * stats.norm.sf(abs(beta / se)))
    return {
        "beta": beta,
        "se": se,
        "pval": pval,
        "cochran_q": cochran_q,
        "cochran_q_pval": q_pval,
    }


def mr_egger(
    beta_exp: Sequence[float] | np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    settings: Literal["default", "R_default"] = "default",
) -> dict[str, float]:
    """Coding-invariant MR-Egger regression (orients all `beta_exp >= 0` prior to WLS)."""
    _validate_settings(settings)
    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    k = bx.size
    if k < 3:
        raise ValueError("MR-Egger requires at least K >= 3 genetic instruments.")

    signs = np.where(bx >= 0.0, 1.0, -1.0)
    bx_pos = bx * signs
    by_pos = by * signs
    w = 1.0 / (sy**2)

    x_mat = np.column_stack([np.ones(k, dtype=np.float64), bx_pos])
    xtwx_inv = np.linalg.pinv(x_mat.T @ (x_mat * w[:, None]))
    theta = xtwx_inv @ (x_mat.T @ (w * by_pos))
    intercept = float(theta[0])
    slope = float(theta[1])

    resid = by_pos - x_mat @ theta
    df = k - 2
    s2 = max(1.0, float(np.sum(w * (resid**2)) / df))
    se_int = float(np.sqrt(max(s2 * xtwx_inv[0, 0], 1e-12)))
    se_slope = float(np.sqrt(max(s2 * xtwx_inv[1, 1], 1e-12)))
    pval_int = float(2.0 * stats.t.sf(abs(intercept / se_int), df=df))
    pval_slope = float(2.0 * stats.t.sf(abs(slope / se_slope), df=df))

    return {
        "beta": slope,
        "se": se_slope,
        "pval": pval_slope,
        "intercept": intercept,
        "se_intercept": se_int,
        "pval_intercept": pval_int,
    }


def weighted_median_mr(
    beta_exp: Sequence[float] | np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    se_exp: Sequence[float] | np.ndarray | None = None,
    n_boot: int | None = None,
    seed: int = 42,
    settings: Literal["default", "R_default"] = "default",
) -> dict[str, float]:
    """Bowden et al. (2016) Weighted Median MR estimator with parametric bootstrap SE."""
    eff_settings = _validate_settings(settings)
    n_boot_val = 1000 if n_boot is None else int(n_boot)
    if n_boot_val < 2:
        raise ValueError(f"n_boot must be >= 2; got {n_boot!r}.")

    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    k = bx.size
    if k < 3 or by.size != k or sy.size != k:
        raise ValueError("Weighted Median MR requires matching arrays of length K >= 3.")
    if np.any(bx == 0.0):
        raise ValueError("Instrument exposure effects beta_exp must be non-zero.")
    if np.any(sy <= 0.0):
        raise ValueError("Outcome standard errors se_out must be strictly positive.")

    sx: np.ndarray | None = None
    if se_exp is not None:
        sx = np.asarray(se_exp, dtype=np.float64).ravel()
        if sx.size != k or np.any(sx <= 0.0):
            raise ValueError("se_exp must match beta_exp length and be strictly positive.")

    signs = np.where(bx >= 0.0, 1.0, -1.0)
    bx_pos = bx * signs
    by_pos = by * signs

    ratios = by_pos / bx_pos
    if eff_settings == "R_default" and sx is not None:
        # 2nd-order delta-method inverse-variance weights (MRCIEU/TwoSampleMR::mr_weighted_median)
        vbj = (sy / bx_pos) ** 2 + (ratios * (sx / bx_pos)) ** 2
        weights = 1.0 / vbj
    else:
        weights = (bx_pos**2) / (sy**2)
    weights = weights / np.sum(weights)

    order = np.argsort(ratios, kind="mergesort")
    cum_w = np.cumsum(weights[order]) - 0.5 * weights[order]
    beta_wm = float(np.interp(0.5, cum_w, ratios[order]))

    rng = np.random.default_rng(seed)
    boot_ests = np.empty(n_boot_val, dtype=np.float64)
    for b in range(n_boot_val):
        if sx is not None:
            bx_sim = rng.normal(bx_pos, sx)
            bx_sim = np.where(bx_sim == 0.0, 1e-15, bx_sim)
        else:
            bx_sim = bx_pos
        by_sim = rng.normal(by_pos, sy)
        r_sim = by_sim / bx_sim
        ord_sim = np.argsort(r_sim, kind="mergesort")
        cw_sim = np.cumsum(weights[ord_sim]) - 0.5 * weights[ord_sim]
        boot_ests[b] = float(np.interp(0.5, cw_sim, r_sim[ord_sim]))

    if eff_settings == "R_default":
        se_wm = max(float(np.std(boot_ests, ddof=1)), 1e-300)
    else:
        se_wm = float(np.std(boot_ests, ddof=1) + 1e-12)
    pval_wm = float(2.0 * stats.norm.sf(abs(beta_wm / se_wm)))
    return {"beta": beta_wm, "se": se_wm, "pval": pval_wm}


def multivariable_mr(
    beta_exp_primary: Sequence[float] | np.ndarray,
    se_exp_primary: Sequence[float] | np.ndarray,
    beta_exp_covariates: np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    se_exp_covariates: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> dict[str, float | np.ndarray]:
    """Multivariable MR (`MVMR`) + Sanderson-Windmeijer conditional F-statistic for primary exposure."""
    eff_settings = _validate_settings(settings)
    bx = np.asarray(beta_exp_primary, dtype=np.float64).ravel()
    sx = np.asarray(se_exp_primary, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    c_mat = np.asarray(beta_exp_covariates, dtype=np.float64)
    if c_mat.ndim == 1:
        c_mat = c_mat[:, None]

    k = bx.size
    if sx.size != k or by.size != k or sy.size != k or c_mat.shape[0] != k:
        raise ValueError("All MVMR input arrays must have matching instrument dimension K.")
    if np.any(sx <= 0.0) or np.any(sy <= 0.0):
        raise ValueError("Exposure and outcome standard errors must be strictly positive.")

    x_mvmr = np.column_stack([bx, c_mat])
    p_exp = x_mvmr.shape[1]
    if k <= p_exp:
        raise ValueError(f"MVMR requires K > number of exposures ({k} <= {p_exp}).")

    sc_mat: np.ndarray | None = None
    if se_exp_covariates is not None:
        sc_mat = np.asarray(se_exp_covariates, dtype=np.float64)
        if sc_mat.ndim == 1:
            sc_mat = sc_mat[:, None]
        if sc_mat.shape != c_mat.shape or np.any(sc_mat < 0.0):
            raise ValueError(f"se_exp_covariates must be non-negative with shape {c_mat.shape}.")

    w = 1.0 / (sy**2)
    xtwx_inv = np.linalg.pinv(x_mvmr.T @ (x_mvmr * w[:, None]))
    theta = xtwx_inv @ (x_mvmr.T @ (w * by))
    resid = by - x_mvmr @ theta
    df_mvmr = k - p_exp
    raw_s2 = float(np.sum(w * (resid**2)) / df_mvmr)
    if eff_settings == "R_default":
        # Upstream WSpiller/MVMR::ivw_mvmr uses un-clamped residual variance and Student-t_{K-P} p-values
        s2 = raw_s2
        se_vec = np.sqrt(np.maximum(s2 * np.diag(xtwx_inv), 1e-300))
    else:
        s2 = max(1.0, raw_s2)
        se_vec = np.sqrt(np.maximum(s2 * np.diag(xtwx_inv), 1e-12))
    all_pvals = 2.0 * stats.t.sf(np.abs(theta / se_vec), df=df_mvmr)

    if eff_settings == "R_default":
        # Upstream WSpiller/MVMR::strength_mvmr estimates delta via unweighted OLS
        delta_hat = np.linalg.pinv(c_mat.T @ c_mat) @ (c_mat.T @ bx)
        if sc_mat is not None:
            var_cond = sx**2 + (sc_mat**2) @ (delta_hat**2)
        else:
            var_cond = sx**2
    else:
        # Sanderson-Windmeijer (2019/2021) conditional F for primary exposure given secondary exposures,
        # weighting by exposure measurement-error variance sx_1^2 + sum_{m=2}^P delta_m^2 sx_m^2
        w_x0 = 1.0 / (sx**2)
        delta_hat = np.linalg.pinv(c_mat.T @ (c_mat * w_x0[:, None])) @ (c_mat.T @ (w_x0 * bx))
        if sc_mat is not None:
            for _ in range(25):
                var_cond = sx**2 + (sc_mat**2) @ (delta_hat**2)
                w_cond = 1.0 / np.maximum(var_cond, 1e-30)
                delta_new = np.linalg.pinv(c_mat.T @ (c_mat * w_cond[:, None])) @ (c_mat.T @ (w_cond * bx))
                if np.max(np.abs(delta_new - delta_hat)) < 1e-10:
                    delta_hat = delta_new
                    break
                delta_hat = delta_new
            var_cond = sx**2 + (sc_mat**2) @ (delta_hat**2)
        else:
            var_cond = sx**2

    bx_resid = bx - c_mat @ delta_hat
    df_cond = max(k - c_mat.shape[1], 1)
    f_cond = float(np.sum((bx_resid**2) / np.maximum(var_cond, 1e-30)) / df_cond)

    beta_primary = float(theta[0])
    se_primary = float(se_vec[0])
    pval_primary = float(all_pvals[0])
    return {
        "beta_primary": beta_primary,
        "se_primary": se_primary,
        "pval_primary": pval_primary,
        "conditional_f_stat": f_cond,
        "all_betas": theta,
        "all_ses": se_vec,
        "all_pvals": all_pvals,
    }


def run_mr_suite(
    beta_exp: Sequence[float] | np.ndarray,
    se_exp: Sequence[float] | np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    covariate_exp_betas: np.ndarray | None = None,
    covariate_exp_ses: np.ndarray | None = None,
    exposure_name: str = "exposure",
    outcome_name: str = "outcome",
    n_boot: int | None = None,
    seed: int = 42,
    settings: Literal["default", "R_default"] = "default",
) -> MRSummaryResult:
    """Run IVW, coding-invariant MR-Egger, Weighted Median, and optional MVMR in one call."""
    eff_settings = _validate_settings(settings)
    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    sx = np.asarray(se_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    if bx.size < 3:
        raise ValueError("run_mr_suite requires at least K >= 3 instruments.")

    mean_f = float(np.mean((bx / sx) ** 2))
    ivw = ivw_mr(bx, by, sy, settings=eff_settings)
    egger = mr_egger(bx, by, sy, settings=eff_settings)
    wm = weighted_median_mr(bx, by, sy, se_exp=sx, n_boot=n_boot, seed=seed, settings=eff_settings)

    if covariate_exp_betas is not None:
        mv = multivariable_mr(
            bx,
            sx,
            covariate_exp_betas,
            by,
            sy,
            se_exp_covariates=covariate_exp_ses,
            settings=eff_settings,
        )
        b_mv, s_mv, p_mv = float(mv["beta_primary"]), float(mv["se_primary"]), float(mv["pval_primary"])
        f_cond = float(mv["conditional_f_stat"])
    else:
        b_mv, s_mv, p_mv = None, None, None
        f_cond = mean_f

    return MRSummaryResult(
        exposure_name=exposure_name,
        outcome_name=outcome_name,
        n_instruments=bx.size,
        mean_f_stat=mean_f,
        conditional_f_stat=f_cond,
        beta_ivw=ivw["beta"],
        se_ivw=ivw["se"],
        pval_ivw=ivw["pval"],
        cochran_q=ivw["cochran_q"],
        cochran_q_pval=ivw["cochran_q_pval"],
        beta_egger=egger["beta"],
        se_egger=egger["se"],
        pval_egger=egger["pval"],
        egger_intercept=egger["intercept"],
        se_egger_intercept=egger["se_intercept"],
        pval_egger_intercept=egger["pval_intercept"],
        beta_weighted_median=wm["beta"],
        se_weighted_median=wm["se"],
        pval_weighted_median=wm["pval"],
        beta_mvmr=b_mv,
        se_mvmr=s_mv,
        pval_mvmr=p_mv,
    )
