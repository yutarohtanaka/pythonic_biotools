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
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


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
) -> dict[str, float]:
    """Multiplicative random-effects Inverse-Variance Weighted (IVW) MR + Cochran's Q."""
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
) -> dict[str, float]:
    """Coding-invariant MR-Egger regression (orients all `beta_exp >= 0` prior to WLS)."""
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
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Bowden et al. (2016) Weighted Median MR estimator with parametric bootstrap SE."""
    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    if bx.size < 3:
        raise ValueError("Weighted Median MR requires at least K >= 3 instruments.")
    if np.any(bx == 0.0):
        raise ValueError("Instrument exposure effects beta_exp must be non-zero.")

    ratios = by / bx
    weights = (bx**2) / (sy**2)
    weights = weights / np.sum(weights)

    order = np.argsort(ratios, kind="mergesort")
    cum_w = np.cumsum(weights[order]) - 0.5 * weights[order]
    beta_wm = float(np.interp(0.5, cum_w, ratios[order]))

    rng = np.random.default_rng(seed)
    boot_ests = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        by_sim = rng.normal(by, sy)
        r_sim = by_sim / bx
        ord_sim = np.argsort(r_sim, kind="mergesort")
        cw_sim = np.cumsum(weights[ord_sim]) - 0.5 * weights[ord_sim]
        boot_ests[b] = float(np.interp(0.5, cw_sim, r_sim[ord_sim]))

    se_wm = float(np.std(boot_ests, ddof=1) + 1e-12)
    pval_wm = float(2.0 * stats.norm.sf(abs(beta_wm / se_wm)))
    return {"beta": beta_wm, "se": se_wm, "pval": pval_wm}


def multivariable_mr(
    beta_exp_primary: Sequence[float] | np.ndarray,
    se_exp_primary: Sequence[float] | np.ndarray,
    beta_exp_covariates: np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Multivariable MR (`MVMR`) + Sanderson-Windmeijer conditional F-statistic for primary exposure."""
    bx = np.asarray(beta_exp_primary, dtype=np.float64).ravel()
    sx = np.asarray(se_exp_primary, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    c_mat = np.asarray(beta_exp_covariates, dtype=np.float64)
    if c_mat.ndim == 1:
        c_mat = c_mat[:, None]

    k = bx.size
    x_mvmr = np.column_stack([bx, c_mat])
    p_exp = x_mvmr.shape[1]
    if k <= p_exp:
        raise ValueError(f"MVMR requires K > number of exposures ({k} <= {p_exp}).")

    w = 1.0 / (sy**2)
    xtwx_inv = np.linalg.pinv(x_mvmr.T @ (x_mvmr * w[:, None]))
    theta = xtwx_inv @ (x_mvmr.T @ (w * by))
    resid = by - x_mvmr @ theta
    s2 = max(1.0, float(np.sum(w * (resid**2)) / (k - p_exp)))
    se_vec = np.sqrt(np.maximum(s2 * np.diag(xtwx_inv), 1e-12))

    # Sanderson-Windmeijer conditional F for primary exposure given secondary exposures
    gamma_hat = np.linalg.pinv(c_mat.T @ (c_mat * w[:, None])) @ (c_mat.T @ (w * bx))
    bx_resid = bx - c_mat @ gamma_hat
    f_cond = float(np.sum((bx_resid / sx) ** 2) / max(k - c_mat.shape[1], 1))

    beta_primary = float(theta[0])
    se_primary = float(se_vec[0])
    pval_primary = float(2.0 * stats.norm.sf(abs(beta_primary / se_primary)))
    return {
        "beta_primary": beta_primary,
        "se_primary": se_primary,
        "pval_primary": pval_primary,
        "conditional_f_stat": f_cond,
        "all_betas": theta,
        "all_ses": se_vec,
    }


def run_mr_suite(
    beta_exp: Sequence[float] | np.ndarray,
    se_exp: Sequence[float] | np.ndarray,
    beta_out: Sequence[float] | np.ndarray,
    se_out: Sequence[float] | np.ndarray,
    covariate_exp_betas: np.ndarray | None = None,
    exposure_name: str = "exposure",
    outcome_name: str = "outcome",
    n_boot: int = 500,
    seed: int = 42,
) -> MRSummaryResult:
    """Run IVW, coding-invariant MR-Egger, Weighted Median, and optional MVMR in one call."""
    bx = np.asarray(beta_exp, dtype=np.float64).ravel()
    sx = np.asarray(se_exp, dtype=np.float64).ravel()
    by = np.asarray(beta_out, dtype=np.float64).ravel()
    sy = np.asarray(se_out, dtype=np.float64).ravel()
    if bx.size < 3:
        raise ValueError("run_mr_suite requires at least K >= 3 instruments.")

    mean_f = float(np.mean((bx / sx) ** 2))
    ivw = ivw_mr(bx, by, sy)
    egger = mr_egger(bx, by, sy)
    wm = weighted_median_mr(bx, by, sy, n_boot=n_boot, seed=seed)

    if covariate_exp_betas is not None:
        mv = multivariable_mr(bx, sx, covariate_exp_betas, by, sy)
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
