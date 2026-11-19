"""
Survival analysis, competing-risks regression, and heteroskedasticity-robust WLS.

Implements Pythonic equivalents of:
  - `survival::coxph(..., ties="efron")`: Efron-tie Cause-Specific Cox Proportional Hazards
  - `cmprsk::crr`: IPCW Fine-Gray subdistribution hazard regression for competing risks
    (Fine & Gray, JASA 1999, Eqs. 6-10: exact time-varying risk-set weights and
    robust score + censoring-martingale sandwich covariance)
  - `sandwich::vcovHC(..., type="HC3")`: MacKinnon-White HC3 leverage-corrected weighted regression
  - `compute_mdes_interaction_hr`: Analytical minimum detectable interaction hazard ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(f"settings must be 'default' or 'R_default', got {settings!r}.")
    return settings  # type: ignore[return-value]


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


def _km_censoring_curves(
    times: np.ndarray,
    event_types: np.ndarray,
    eps_g: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute both right-continuous G(t) and left-continuous G(t-) KM censoring curves.

    Returns `(uniq_t, g_left_uniq, g_right_uniq, dc_uniq, yc_uniq)` over sorted unique times.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    is_cens = np.asarray(event_types, dtype=int).ravel() == 0
    uniq_t = np.unique(t)
    m = uniq_t.size
    g_left = np.ones(m, dtype=np.float64)
    g_right = np.ones(m, dtype=np.float64)
    dc_arr = np.zeros(m, dtype=np.float64)
    yc_arr = np.zeros(m, dtype=np.float64)

    curr_g = 1.0
    for idx, ut in enumerate(uniq_t):
        g_left[idx] = max(curr_g, eps_g)
        at_risk = float(np.sum(t >= ut))
        c_events = float(np.sum((t == ut) & is_cens))
        yc_arr[idx] = at_risk
        dc_arr[idx] = c_events
        if at_risk > 0.0 and c_events > 0.0:
            curr_g = max(curr_g * (1.0 - c_events / at_risk), eps_g)
        g_right[idx] = max(curr_g, eps_g)
    return uniq_t, g_left, g_right, dc_arr, yc_arr


def kaplan_meier_censoring_survival(
    times: Sequence[float],
    event_types: Sequence[int],
    eps_g: float = 0.05,
    left_continuous: bool = False,
) -> np.ndarray:
    r"""Estimate Kaplan-Meier censoring survival $\hat{G}(t)$ (or left-continuous $\hat{G}(t-)$
    when `left_continuous=True`, `event_type == 0` = censored) with lower clamp $\epsilon_G$
    to prevent inverse-probability weight explosion.
    """
    t = np.asarray(times, dtype=np.float64)
    ev = np.asarray(event_types, dtype=int)
    if t.size == 0:
        return np.array([], dtype=np.float64)
    uniq_t, g_left, g_right, _, _ = _km_censoring_curves(t.ravel(), ev.ravel(), eps_g=eps_g)
    pos = np.searchsorted(uniq_t, t)
    curve = g_left if left_continuous else g_right
    return np.maximum(curve[pos], eps_g)


def _efron_partial_loglik_and_derivatives(
    beta: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    is_event: np.ndarray,
    sample_weights: np.ndarray | None = None,
    ridge_lambda: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Efron-tie Cox partial log-likelihood, gradient, and negative Hessian."""
    n, p = x.shape
    w_samp = np.ones(n, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)

    eta = np.clip(x @ beta, -35.0, 35.0)
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

    if ridge_lambda > 0.0:
        loglik -= 0.5 * ridge_lambda * float(np.sum(beta**2))
        grad -= ridge_lambda * beta
        info += ridge_lambda * np.eye(p)
    return loglik, grad, info


def _fine_gray_loglik_and_derivatives(
    beta: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    is_target: np.ndarray,
    is_competing: np.ndarray,
    g_minus_subj: np.ndarray,
    uniq_target_times: np.ndarray,
    g_minus_target_times: np.ndarray,
    ridge_lambda: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Compute Fine-Gray (1999, Eq. 6-7) IPCW log-pseudo-likelihood, score, and observed information.

    At each unique target failure time $t_k$, subject $i$ belongs to the subdistribution risk set
    with time-varying weight:
      - $w_i(t_k) = 1$ if $T_i \ge t_k$,
      - $w_i(t_k) = \hat{G}(t_k-) / \hat{G}(T_i-)$ if $T_i < t_k$ and subject $i$ had a competing event,
      - $w_i(t_k) = 0$ otherwise.
    """
    n, p = x.shape
    eta = np.clip(x @ beta, -35.0, 35.0)
    exp_eta = np.exp(eta)

    loglik = 0.0
    grad = np.zeros(p, dtype=np.float64)
    info = np.zeros((p, p), dtype=np.float64)

    for k_idx, tk in enumerate(uniq_target_times):
        at_event = (times == tk) & (is_target == 1)
        d_k = int(np.sum(at_event))
        if d_k == 0:
            continue

        gk = g_minus_target_times[k_idx]
        w_k = np.zeros(n, dtype=np.float64)
        w_k[times >= tk] = 1.0
        comp_mask = is_competing & (times < tk)
        if np.any(comp_mask):
            w_k[comp_mask] = np.clip(gk / np.maximum(g_minus_subj[comp_mask], 1e-300), 0.0, 1.0)

        theta_k = w_k * exp_eta
        s0 = max(float(np.sum(theta_k)), 1e-12)
        tx_k = x * theta_k[:, None]
        s1 = np.sum(tx_k, axis=0)
        s2 = tx_k.T @ x
        x_bar = s1 / s0

        loglik += float(np.sum(eta[at_event])) - d_k * np.log(s0)
        grad += np.sum(x[at_event], axis=0) - d_k * x_bar
        info += d_k * ((s2 / s0) - np.outer(x_bar, x_bar))

    if ridge_lambda > 0.0:
        loglik -= 0.5 * ridge_lambda * float(np.sum(beta**2))
        grad -= ridge_lambda * beta
        info += ridge_lambda * np.eye(p)
    return loglik, grad, info


def _fine_gray_sandwich_covariance(
    beta: np.ndarray,
    info: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    is_target: np.ndarray,
    is_competing: np.ndarray,
    is_censored: np.ndarray,
    g_minus_subj: np.ndarray,
    uniq_target_times: np.ndarray,
    g_minus_target_times: np.ndarray,
    uniq_t: np.ndarray,
    dc_uniq: np.ndarray,
    yc_uniq: np.ndarray,
    censoring_correction: bool = True,
    strict_comp_tie: bool = False,
) -> np.ndarray:
    r"""Compute Fine-Gray (1999, Eqs. 8-10) robust sandwich covariance $I(\hat{\beta})^{-1} \Omega I(\hat{\beta})^{-1}$.

    Decomposes per-subject asymptotic score influence into:
      - $\hat{\eta}_i$: subdistribution counting-process score residual
      - $\hat{\psi}_i$: IPCW Kaplan-Meier censoring-martingale residual correction
    """
    n, p = x.shape
    eta = np.clip(x @ beta, -35.0, 35.0)
    exp_eta = np.exp(eta)

    eta_subj = np.zeros((n, p), dtype=np.float64)
    k_count = uniq_target_times.size
    d_lam_vec = np.zeros(k_count, dtype=np.float64)
    x_bar_mat = np.zeros((k_count, p), dtype=np.float64)
    theta_mat = np.zeros((k_count, n), dtype=np.float64)

    for k_idx, tk in enumerate(uniq_target_times):
        at_event = (times == tk) & (is_target == 1)
        d_k = int(np.sum(at_event))
        if d_k == 0:
            continue

        gk = g_minus_target_times[k_idx]
        w_k = np.zeros(n, dtype=np.float64)
        w_k[times >= tk] = 1.0
        comp_mask = is_competing & (times < tk)
        if np.any(comp_mask):
            w_k[comp_mask] = np.clip(gk / np.maximum(g_minus_subj[comp_mask], 1e-300), 0.0, 1.0)

        theta_k = w_k * exp_eta
        s0 = max(float(np.sum(theta_k)), 1e-12)
        x_bar = np.sum(x * theta_k[:, None], axis=0) / s0
        d_lam = d_k / s0

        d_lam_vec[k_idx] = d_lam
        x_bar_mat[k_idx] = x_bar
        theta_mat[k_idx] = theta_k

        x_centered = x - x_bar[None, :]
        eta_subj[at_event] += x_centered[at_event]
        eta_subj -= x_centered * (theta_k * d_lam)[:, None]

    psi_subj = np.zeros((n, p), dtype=np.float64)
    if censoring_correction and np.any(is_competing) and np.any(dc_uniq > 0.0):
        cens_indices = np.flatnonzero((dc_uniq > 0.0) & (yc_uniq > 0.0))
        for c_idx in cens_indices:
            u_c = uniq_t[c_idx]
            dc = dc_uniq[c_idx]
            yc = yc_uniq[c_idx]
            later_k = np.flatnonzero(uniq_target_times > u_c)
            if later_k.size == 0:
                continue
            # `crr.f:347`: strict `times < u_c` in settings='R_default'
            comp_le_u = is_competing & ((times < u_c) if strict_comp_tie else (times <= u_c))
            if not np.any(comp_le_u):
                continue

            x_comp = x[comp_le_u]
            q_u = np.zeros(p, dtype=np.float64)
            for k_idx in later_k:
                d_lam = d_lam_vec[k_idx]
                if d_lam <= 0.0:
                    continue
                th_comp = theta_mat[k_idx, comp_le_u]
                q_u += d_lam * np.sum((x_comp - x_bar_mat[k_idx][None, :]) * th_comp[:, None], axis=0)

            at_cens = (times == u_c) & is_censored
            in_cens_risk = times >= u_c
            psi_subj[at_cens] += q_u / yc
            psi_subj[in_cens_risk] -= (dc / (yc * yc)) * q_u

    total_scores = eta_subj + psi_subj
    meat = total_scores.T @ total_scores
    h_inv = np.linalg.pinv(info)
    return h_inv @ meat @ h_inv


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


def _newton_maximize_with_backtracking(
    beta_init: np.ndarray,
    eval_fn,
    max_iter: int = 60,
    max_step_halving: int = 15,
    max_step_norm: float = 2.5,
    tol: float = 1e-7,
) -> tuple[np.ndarray, bool]:
    """Maximize a concave log-(pseudo-)likelihood via Newton-Raphson with monotone step-halving."""
    beta = beta_init.copy()
    converged = False

    for _ in range(max_iter):
        ll, grad, info = eval_fn(beta)
        if float(np.max(np.abs(grad))) < tol * 1e-2:
            converged = True
            break

        step = np.linalg.pinv(info) @ grad
        step_norm = float(np.max(np.abs(step)))
        if step_norm < tol:
            beta = beta + step
            converged = True
            break
        if step_norm > max_step_norm:
            step = step * (max_step_norm / step_norm)

        alpha = 1.0
        beta_new = beta + step
        for _ in range(max_step_halving):
            cand = beta + alpha * step
            ll_cand, _, _ = eval_fn(cand)
            if np.isfinite(ll_cand) and ll_cand >= ll - 1e-12:
                beta_new = cand
                break
            alpha *= 0.5
        else:
            beta_new = beta + alpha * step

        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    return beta, converged


def fit_cox_ph_efron(
    X: np.ndarray | pd.DataFrame,
    times: Sequence[float],
    event_types: Sequence[int],
    target_event: int = 1,
    coef_names: Sequence[str] | None = None,
    ridge_lambda: float = 0.0,
    max_iter: int = 60,
    max_step_halving: int = 15,
    tol: float = 1e-7,
    *,
    settings: SettingsMode = "default",
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
    _validate_settings(settings)
    x, names = _extract_design(X, coef_names)
    t = np.asarray(times, dtype=np.float64).ravel()
    ev = np.asarray(event_types, dtype=int).ravel()
    if x.shape[0] != t.size or t.size != ev.size:
        raise ValueError("X, times, and event_types must share the same sample count.")

    is_event = (ev == target_event).astype(int)
    p = x.shape[1]
    beta0 = np.zeros(p, dtype=np.float64)

    def _eval(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return _efron_partial_loglik_and_derivatives(
            b, x, t, is_event, sample_weights=None, ridge_lambda=ridge_lambda
        )

    if int(is_event.sum()) == 0:
        beta, converged = beta0, True
    else:
        beta, converged = _newton_maximize_with_backtracking(
            beta0, _eval, max_iter=max_iter, max_step_halving=max_step_halving, tol=tol
        )

    ll, _, info = _eval(beta)
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    z_stat = beta / se
    p_val = 2.0 * stats.norm.sf(np.abs(z_stat))
    hr = np.exp(np.clip(beta, -50.0, 50.0))
    z_crit = float(stats.norm.isf(0.025)) if settings == "R_default" else 1.959964
    ci_low = np.exp(np.clip(beta - z_crit * se, -50.0, 50.0))
    ci_high = np.exp(np.clip(beta + z_crit * se, -50.0, 50.0))

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
    eps_g: float | None = None,
    ridge_lambda: float = 0.0,
    censoring_correction: bool = True,
    max_iter: int = 60,
    max_step_halving: int = 15,
    tol: float = 1e-7,
    *,
    settings: SettingsMode = "default",
) -> HazardModelResult:
    r"""Fit an IPCW Fine-Gray subdistribution hazard model (`cmprsk::crr`; Fine & Gray, JASA 1999).

    Subjects experiencing a competing event ($E_i \notin \{0, \text{target\_event}\}$) at $T_i < t_k$
    remain in the subdistribution risk set at target failure time $t_k$ with exact time-varying
    weight $w_i(t_k) = \hat{G}(t_k-) / \hat{G}(T_i-)$, using left-continuous Kaplan-Meier censoring
    survival $\hat{G}(t-)$. Standard errors use the Fine-Gray robust Huber-White score sandwich
    estimator incorporating censoring-martingale estimation uncertainty (Eqs. 8-10).
    """
    _validate_settings(settings)
    eps_g_eff = (0.0 if settings == "R_default" else 0.05) if eps_g is None else float(eps_g)

    x, names = _extract_design(X, coef_names)
    t = np.asarray(times, dtype=np.float64).ravel().copy()
    ev = np.asarray(event_types, dtype=int).ravel()
    if x.shape[0] != t.size or t.size != ev.size:
        raise ValueError("X, times, and event_types must share the same sample count.")

    uniq_t, g_left_uniq, _, dc_uniq, yc_uniq = _km_censoring_curves(t, ev, eps_g=eps_g_eff)
    g_minus_subj = g_left_uniq[np.searchsorted(uniq_t, t)]

    is_target = (ev == target_event).astype(int)
    is_competing = (ev != 0) & (ev != target_event)
    is_censored = ev == 0

    uniq_target_times = np.unique(t[is_target == 1])
    g_minus_target_times = g_left_uniq[np.searchsorted(uniq_t, uniq_target_times)]

    p = x.shape[1]
    beta0 = np.zeros(p, dtype=np.float64)

    def _eval(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return _fine_gray_loglik_and_derivatives(
            b,
            x,
            t,
            is_target,
            is_competing,
            g_minus_subj,
            uniq_target_times,
            g_minus_target_times,
            ridge_lambda=ridge_lambda,
        )

    if uniq_target_times.size == 0:
        beta, converged = beta0, True
    else:
        beta, converged = _newton_maximize_with_backtracking(
            beta0, _eval, max_iter=max_iter, max_step_halving=max_step_halving, tol=tol
        )

    ll, _, info = _eval(beta)
    if uniq_target_times.size == 0:
        cov = np.zeros((p, p), dtype=np.float64)
    else:
        cov = _fine_gray_sandwich_covariance(
            beta=beta,
            info=info,
            x=x,
            times=t,
            is_target=is_target,
            is_competing=is_competing,
            is_censored=is_censored,
            g_minus_subj=g_minus_subj,
            uniq_target_times=uniq_target_times,
            g_minus_target_times=g_minus_target_times,
            uniq_t=uniq_t,
            dc_uniq=dc_uniq,
            yc_uniq=yc_uniq,
            censoring_correction=censoring_correction,
            strict_comp_tie=(settings == "R_default"),
        )

    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    z_stat = beta / se
    p_val = 2.0 * stats.norm.sf(np.abs(z_stat))
    shr = np.exp(np.clip(beta, -50.0, 50.0))
    z_crit = float(stats.norm.isf(0.025)) if settings == "R_default" else 1.959964
    ci_low = np.exp(np.clip(beta - z_crit * se, -50.0, 50.0))
    ci_high = np.exp(np.clip(beta + z_crit * se, -50.0, 50.0))

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
    *,
    settings: SettingsMode = "default",
) -> RobustWLSResult:
    """Fit Weighted Least Squares with White-Huber HC3 leverage-corrected covariance (`vcovHC(..., 'HC3')`).

    Ideal for heteroskedastic regression, deconvolved cell-type eQTLs (pass `weights = cell_fraction**2`),
    and small-sample regression with influential observations.
    """
    _validate_settings(settings)
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
        if w.size != x.shape[0]:
            raise ValueError("weights and X must share the same sample count.")
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("Weights must be finite and non-negative.")
        w_sqrt = np.sqrt(w)

    xw = x * w_sqrt[:, None]
    yw = y_vec * w_sqrt
    xtx_inv = np.linalg.pinv(xw.T @ xw)
    beta = xtx_inv @ (xw.T @ yw)
    resid = yw - xw @ beta

    h_diag = np.clip(np.sum((xw @ xtx_inv) * xw, axis=1), 0.0, 1.0 - 1e-12)
    hc3_scale = (resid / (1.0 - h_diag)) ** 2
    meat = (xw * hc3_scale[:, None]).T @ xw
    cov_hc3 = xtx_inv @ meat @ xtx_inv

    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-24))
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
