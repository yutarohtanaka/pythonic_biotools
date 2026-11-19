"""
Survival analysis, competing-risks regression, and heteroskedasticity-robust WLS.

Implements Pythonic equivalents of:
  - `survival::coxph(..., ties="efron")`: Efron-tie Cause-Specific Cox Proportional Hazards
  - `cmprsk::crr`: IPCW Fine-Gray subdistribution hazard regression for competing risks
  - `sandwich::vcovHC(..., type="HC3")`: White-Huber HC3 leverage-corrected weighted regression
  - `compute_mdes_interaction_hr`: Analytical minimum detectable interaction hazard ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class HazardModelResult:
    """Result container for Cause-Specific Cox PH and Fine-Gray competing-risks models."""

    model_type: str
    coef_names: list[str]
    beta: np.ndarray
    se: np.ndarray
    z_stat: np.ndarray
    p_value: np.ndarray
    hazard_ratio: np.ndarray
    ci_low_hr: np.ndarray
    ci_high_hr: np.ndarray
    partial_loglik: float
    n_samples: int
    n_target_events: int
    converged: bool

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": self.beta,
                "se": self.se,
                "z": self.z_stat,
                "p_value": self.p_value,
                "HR": self.hazard_ratio,
                "HR_95CI_low": self.ci_low_hr,
                "HR_95CI_high": self.ci_high_hr,
            },
            index=self.coef_names,
        )


@dataclass(frozen=True)
class RobustWLSResult:
    """Result container for White-Huber HC3 heteroskedasticity-robust weighted least squares."""

    coef_names: list[str]
    beta: np.ndarray
    se_hc3: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    df_resid: int
    cov_hc3: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": self.beta,
                "se_HC3": self.se_hc3,
                "t": self.t_stat,
                "p_value": self.p_value,
            },
            index=self.coef_names,
        )


def kaplan_meier_censoring_survival(
    times: Sequence[float],
    event_types: Sequence[int],
    eps_g: float = 0.05,
) -> np.ndarray:
    r"""Estimate Kaplan-Meier censoring survival $\hat{G}(t)$ (`event_type == 0` = censored)
    with lower clamp $\epsilon_G$ to prevent inverse-probability weight explosion.
    """
    t = np.asarray(times, dtype=np.float64)
    is_censored = (np.asarray(event_types, dtype=int) == 0).astype(np.float64)
    uniq_t = np.unique(t)
    g_surv = np.ones_like(t, dtype=np.float64)
    curr_g = 1.0
    for ut in uniq_t:
        at_risk = float(np.sum(t >= ut))
        c_events = float(np.sum((t == ut) & (is_censored == 1.0)))
        if at_risk > 0.0 and c_events > 0.0:
            curr_g *= 1.0 - (c_events / at_risk)
        curr_g = max(curr_g, eps_g)
        g_surv[t >= ut] = curr_g
    return np.maximum(g_surv, eps_g)


def _efron_partial_loglik_and_derivatives(
    beta: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    is_event: np.ndarray,
    sample_weights: np.ndarray | None = None,
    ridge_lambda: float = 1e-4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Efron-tie Cox partial log-likelihood, gradient, and negative Hessian."""
    n, p = x.shape
    w_samp = np.ones(n, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)

    eta = np.clip(x @ beta, -25.0, 25.0)
    theta = w_samp * np.exp(eta)
    tx = x * theta[:, None]

    order = np.argsort(-times, kind="mergesort")
    t_sorted = times[order]
    ev_sorted = is_event[order]
    w_sorted = w_samp[order]
    eta_sorted = eta[order]
    theta_sorted = theta[order]
    x_sorted = x[order]
    tx_sorted = tx[order]

    loglik = 0.0
    grad = np.zeros(p, dtype=np.float64)
    info = np.zeros((p, p), dtype=np.float64)

    uniq_event_times = np.unique(t_sorted[ev_sorted == 1])
    for ut in uniq_event_times:
        in_risk = t_sorted >= ut
        at_time_event = (t_sorted == ut) & (ev_sorted == 1)
        d = int(np.sum(at_time_event))
        if d == 0:
            continue

        s0_risk = float(np.sum(theta_sorted[in_risk]))
        s1_risk = np.sum(tx_sorted[in_risk], axis=0)
        x_risk = x_sorted[in_risk]
        th_risk = theta_sorted[in_risk]
        s2_risk = (x_risk * th_risk[:, None]).T @ x_risk

        s0_tie = float(np.sum(theta_sorted[at_time_event]))
        s1_tie = np.sum(tx_sorted[at_time_event], axis=0)
        x_tie = x_sorted[at_time_event]
        th_tie = theta_sorted[at_time_event]
        s2_tie = (x_tie * th_tie[:, None]).T @ x_tie

        w_tie_mean = float(np.mean(w_sorted[at_time_event]))
        loglik += float(np.sum(w_sorted[at_time_event] * eta_sorted[at_time_event]))
        grad += np.sum(x_sorted[at_time_event] * w_sorted[at_time_event, None], axis=0)

        for l_idx in range(d):
            frac = l_idx / float(d)
            denom = max(s0_risk - frac * s0_tie, 1e-12)
            v1 = (s1_risk - frac * s1_tie) / denom
            v2 = (s2_risk - frac * s2_tie) / denom
            loglik -= w_tie_mean * np.log(denom)
            grad -= w_tie_mean * v1
            info += w_tie_mean * (v2 - np.outer(v1, v1))

    loglik -= 0.5 * ridge_lambda * float(np.sum(beta**2))
    grad -= ridge_lambda * beta
    info += ridge_lambda * np.eye(p)
    return loglik, grad, info


def _extract_design(
    X: np.ndarray | pd.DataFrame,
    coef_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns] if coef_names is None else [str(c) for c in coef_names]
        x = X.to_numpy(dtype=np.float64)
    else:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        names = [f"x{i}" for i in range(x.shape[1])] if coef_names is None else [str(c) for c in coef_names]
    return x, names


def fit_cox_ph_efron(
    X: np.ndarray | pd.DataFrame,
    times: Sequence[float],
    event_types: Sequence[int],
    target_event: int = 1,
    coef_names: Sequence[str] | None = None,
    ridge_lambda: float = 1e-4,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> HazardModelResult:
    """Fit an Efron-tie Cause-Specific Cox Proportional Hazards model (`survival::coxph`).

    Parameters
    ----------
    X : (n_samples, n_covariates) array or DataFrame
        Covariate matrix (do NOT include an intercept column; baseline hazard absorbs it).
    times : length-`n_samples` sequence of float
        Follow-up / event times.
    event_types : length-`n_samples` sequence of int
        `0` = right-censored, `target_event` = event of interest, other non-zero ints = competing events
        (treated as censored at their event time in cause-specific Cox PH).
    """
    x, names = _extract_design(X, coef_names)
    t = np.asarray(times, dtype=np.float64).ravel()
    ev = np.asarray(event_types, dtype=int).ravel()
    if x.shape[0] != t.size or t.size != ev.size:
        raise ValueError("X, times, and event_types must share the same sample count.")

    is_event = (ev == target_event).astype(int)
    p = x.shape[1]
    beta = np.zeros(p, dtype=np.float64)
    converged = False

    for _ in range(max_iter):
        _, grad, info = _efron_partial_loglik_and_derivatives(
            beta, x, t, is_event, sample_weights=None, ridge_lambda=ridge_lambda
        )
        step = np.linalg.pinv(info) @ grad
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 2.0:
            step = step * (2.0 / step_norm)
        beta_new = beta + step
        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    ll, _, info = _efron_partial_loglik_and_derivatives(
        beta, x, t, is_event, sample_weights=None, ridge_lambda=ridge_lambda
    )
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    z_stat = beta / se
    p_val = 2.0 * stats.norm.sf(np.abs(z_stat))
    hr = np.exp(np.clip(beta, -50.0, 50.0))
    ci_low = np.exp(np.clip(beta - 1.959964 * se, -50.0, 50.0))
    ci_high = np.exp(np.clip(beta + 1.959964 * se, -50.0, 50.0))

    return HazardModelResult(
        model_type="CoxPH_Efron",
        coef_names=names,
        beta=beta,
        se=se,
        z_stat=z_stat,
        p_value=p_val,
        hazard_ratio=hr,
        ci_low_hr=ci_low,
        ci_high_hr=ci_high,
        partial_loglik=ll,
        n_samples=x.shape[0],
        n_target_events=int(is_event.sum()),
        converged=converged,
    )


def fit_fine_gray_ipcw(
    X: np.ndarray | pd.DataFrame,
    times: Sequence[float],
    event_types: Sequence[int],
    target_event: int = 1,
    coef_names: Sequence[str] | None = None,
    eps_g: float = 0.05,
    ridge_lambda: float = 1e-4,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> HazardModelResult:
    r"""Fit an IPCW Fine-Gray subdistribution hazard model (`cmprsk::crr`).

    Subjects experiencing a competing event ($E_i \notin \{0, \text{target\_event}\}$) remain in
    the subdistribution risk set with Kaplan-Meier inverse-probability-of-censoring weights.
    """
    x, names = _extract_design(X, coef_names)
    t = np.asarray(times, dtype=np.float64).ravel().copy()
    ev = np.asarray(event_types, dtype=int).ravel()
    if x.shape[0] != t.size or t.size != ev.size:
        raise ValueError("X, times, and event_types must share the same sample count.")

    g_hat = kaplan_meier_censoring_survival(t, ev, eps_g=eps_g)
    is_target = (ev == target_event).astype(int)
    is_competing = (ev != 0) & (ev != target_event)

    t_sub = t.copy()
    t_max = float(np.max(t)) + 0.01
    t_sub[is_competing] = t_max

    g_tmax = float(np.min(g_hat))
    weights = np.ones_like(t, dtype=np.float64)
    weights[is_competing] = np.clip(g_tmax / g_hat[is_competing], eps_g, 1.0)

    p = x.shape[1]
    beta = np.zeros(p, dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        _, grad, info = _efron_partial_loglik_and_derivatives(
            beta, x, t_sub, is_target, sample_weights=weights, ridge_lambda=ridge_lambda
        )
        step = np.linalg.pinv(info) @ grad
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 2.0:
            step = step * (2.0 / step_norm)
        beta_new = beta + step
        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    ll, _, info = _efron_partial_loglik_and_derivatives(
        beta, x, t_sub, is_target, sample_weights=weights, ridge_lambda=ridge_lambda
    )
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    z_stat = beta / se
    p_val = 2.0 * stats.norm.sf(np.abs(z_stat))
    shr = np.exp(np.clip(beta, -50.0, 50.0))
    ci_low = np.exp(np.clip(beta - 1.959964 * se, -50.0, 50.0))
    ci_high = np.exp(np.clip(beta + 1.959964 * se, -50.0, 50.0))

    return HazardModelResult(
        model_type="FineGray_IPCW",
        coef_names=names,
        beta=beta,
        se=se,
        z_stat=z_stat,
        p_value=p_val,
        hazard_ratio=shr,
        ci_low_hr=ci_low,
        ci_high_hr=ci_high,
        partial_loglik=ll,
        n_samples=x.shape[0],
        n_target_events=int(is_target.sum()),
        converged=converged,
    )


def fit_wls_hc3(
    X: np.ndarray | pd.DataFrame,
    y: Sequence[float],
    weights: Sequence[float] | None = None,
    add_intercept: bool = False,
    coef_names: Sequence[str] | None = None,
) -> RobustWLSResult:
    """Fit Weighted Least Squares with White-Huber HC3 leverage-corrected covariance (`vcovHC(..., 'HC3')`).

    Ideal for heteroskedastic regression, deconvolved cell-type eQTLs (pass `weights = cell_fraction**2`),
    and small-sample regression with influential observations.
    """
    x, names = _extract_design(X, coef_names)
    if add_intercept:
        x = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
        names = ["(Intercept)", *names]

    y_vec = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y_vec.size:
        raise ValueError("X and y must share the same sample count.")

    if weights is None:
        w_sqrt = np.ones(x.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if np.any(w < 0.0):
            raise ValueError("Weights must be non-negative.")
        w_sqrt = np.sqrt(np.maximum(w, 1e-6))

    xw = x * w_sqrt[:, None]
    yw = y_vec * w_sqrt
    xtx_inv = np.linalg.pinv(xw.T @ xw)
    beta = xtx_inv @ (xw.T @ yw)
    resid = yw - xw @ beta

    h_diag = np.clip(np.sum((xw @ xtx_inv) * xw, axis=1), 0.0, 0.99)
    hc3_scale = (resid / (1.0 - h_diag)) ** 2
    meat = (xw * hc3_scale[:, None]).T @ xw
    cov_hc3 = xtx_inv @ meat @ xtx_inv

    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-12))
    df_resid = max(int(x.shape[0] - x.shape[1]), 1)
    t_stat = beta / se
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid)

    return RobustWLSResult(
        coef_names=names,
        beta=beta,
        se_hc3=se,
        t_stat=t_stat,
        p_value=p_val,
        df_resid=df_resid,
        cov_hc3=cov_hc3,
    )


def compute_mdes_interaction_hr(
    n_events: int,
    prop_stratum: float = 0.5,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
) -> float:
    """Analytical Minimum Detectable Effect Size (MDES) interaction Hazard Ratio at target power

    Computes the smallest detectable multiplicative interaction HR between a standardized
    continuous score ($SD = 1$) and a binary subgroup indicator of prevalence `prop_stratum`.
    """
    if n_events < 1 or not (0.0 < prop_stratum < 1.0):
        raise ValueError("Require n_events >= 1 and 0 < prop_stratum < 1.")
    z_alpha = float(stats.norm.isf(alpha_two_sided / 2.0))
    z_beta = float(stats.norm.ppf(power))
    eff_n = max(n_events * prop_stratum * (1.0 - prop_stratum), 1.0)
    min_beta_int = (z_alpha + z_beta) / np.sqrt(eff_n)
    return float(np.exp(min_beta_int))
