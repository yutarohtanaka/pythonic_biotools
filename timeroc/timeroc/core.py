"""
Core algorithms for IPCW time-dependent ROC curves, competing-risks AUCs,
IID martingale influence functions, simultaneous confidence bands, and paired
biomarker comparisons.

Reference:
  Blanche, P., Dartigues, J.-F., & Jacqmin-Gadda, H. (2013).
  "Estimating and comparing time-dependent areas under receiver operating
  characteristic curves for censored event times with competing risks."
  Statistics in Medicine, 32(30), 5381-5397.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ConfidenceBandResult:
    """Pointwise CIs and Gaussian-multiplier simultaneous confidence bands for AUC(t)."""

    definition: int
    level: float
    times: np.ndarray
    auc: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    cb_lower: np.ndarray
    cb_upper: np.ndarray
    z_pointwise: float
    c_simultaneous: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.times,
                f"AUC_{self.definition}": self.auc,
                "SE": self.se,
                "CI_lower": self.ci_lower,
                "CI_upper": self.ci_upper,
                "CB_lower": self.cb_lower,
                "CB_upper": self.cb_upper,
            }
        )


@dataclass(frozen=True)
class TimeROCResult:
    """Result container for `time_roc`."""

    n_subjects: int
    cause: int
    has_competing_risks: bool
    weighting: str
    times: np.ndarray
    n_cases: np.ndarray
    n_survivors: np.ndarray
    n_competing: np.ndarray
    cum_inc_1: np.ndarray
    cum_inc_2: np.ndarray
    surv_prob: np.ndarray
    auc_1: np.ndarray
    auc_2: np.ndarray
    se_auc_1: np.ndarray | None
    se_auc_2: np.ndarray | None
    cov_auc_1: np.ndarray | None
    cov_auc_2: np.ndarray | None
    iid_auc_1: np.ndarray | None
    iid_auc_2: np.ndarray | None
    cutpoints: np.ndarray
    se: np.ndarray
    sp_1: np.ndarray
    fp_1: np.ndarray
    sp_2: np.ndarray
    fp_2: np.ndarray
    # Internal cohort fingerprint for validating paired `compare_timeroc` calls
    _T: np.ndarray
    _delta: np.ndarray

    def confidence_intervals(
        self,
        level: float = 0.95,
        *,
        definition: Literal[1, 2] = 1,
        n_sim: int = 2000,
        seed: int | np.random.Generator | None = 42,
    ) -> ConfidenceBandResult:
        """Compute pointwise Wald CIs and Gaussian-multiplier simultaneous confidence bands."""
        if not (0.0 < level < 1.0):
            raise ValueError("level must lie strictly in (0, 1).")
        if definition not in (1, 2):
            raise ValueError("definition must be 1 or 2.")
        if n_sim < 50:
            raise ValueError("n_sim must be >= 50.")

        iid_mat = self.iid_auc_1 if definition == 1 else self.iid_auc_2
        auc_vec = self.auc_1 if definition == 1 else self.auc_2
        se_vec = self.se_auc_1 if definition == 1 else self.se_auc_2
        if iid_mat is None or se_vec is None:
            raise ValueError("IID influence functions were not computed (re-run with iid=True, weighting='marginal').")

        z_crit = float(stats.norm.ppf(0.5 + 0.5 * level))
        ci_low = auc_vec - z_crit * se_vec
        ci_high = auc_vec + z_crit * se_vec

        valid_cols = np.isfinite(auc_vec) & np.isfinite(se_vec) & (se_vec > 0.0)
        if not np.any(valid_cols):
            c_sim = z_crit
        else:
            rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
            g_mult = rng.standard_normal((n_sim, self.n_subjects))
            phi_valid = iid_mat[:, valid_cols]
            se_valid = se_vec[valid_cols]
            z_boot = (g_mult @ phi_valid) / (self.n_subjects * se_valid[None, :])
            sup_abs_z = np.max(np.abs(z_boot), axis=1)
            c_sim = float(np.quantile(sup_abs_z, level))

        cb_low = auc_vec - c_sim * se_vec
        cb_high = auc_vec + c_sim * se_vec

        return ConfidenceBandResult(
            definition=definition,
            level=level,
            times=self.times.copy(),
            auc=auc_vec.copy(),
            se=se_vec.copy(),
            ci_lower=ci_low,
            ci_upper=ci_high,
            cb_lower=cb_low,
            cb_upper=cb_high,
            z_pointwise=z_crit,
            c_simultaneous=c_sim,
        )

    def roc_curve(self, time_index: int = 0) -> pd.DataFrame:
        """Return threshold-by-threshold ROC coordinates `(cutpoint, Se, Sp_1, FP_1, Sp_2, FP_2)` at `times[time_index]`."""
        if not (0 <= time_index < self.times.size):
            raise ValueError(f"time_index {time_index} out of range [0, {self.times.size}).")
        return pd.DataFrame(
            {
                "cutpoint": self.cutpoints,
                "Se": self.se[:, time_index],
                "Sp_1": self.sp_1[:, time_index],
                "FP_1": self.fp_1[:, time_index],
                "Sp_2": self.sp_2[:, time_index],
                "FP_2": self.fp_2[:, time_index],
            }
        )

    def to_dataframe(
        self,
        level: float = 0.95,
        *,
        n_sim: int = 2000,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """Return per-horizon summary DataFrame including cumulative incidences, AUCs, SEs, and CIs."""
        cols: dict[str, object] = {
            "time": self.times,
            "n_cases": self.n_cases,
            "n_survivors": self.n_survivors,
            "n_competing": self.n_competing,
            "F_1": self.cum_inc_1,
            "F_2": self.cum_inc_2,
            "S": self.surv_prob,
            "AUC_1": self.auc_1,
            "AUC_2": self.auc_2,
        }
        if self.iid_auc_1 is not None and self.iid_auc_2 is not None:
            cb1 = self.confidence_intervals(level=level, definition=1, n_sim=n_sim, seed=seed)
            cb2 = self.confidence_intervals(level=level, definition=2, n_sim=n_sim, seed=seed)
            cols["SE_1"] = cb1.se
            cols["CI_1_lower"] = cb1.ci_lower
            cols["CI_1_upper"] = cb1.ci_upper
            cols["CB_1_lower"] = cb1.cb_lower
            cols["CB_1_upper"] = cb1.cb_upper
            cols["SE_2"] = cb2.se
            cols["CI_2_lower"] = cb2.ci_lower
            cols["CI_2_upper"] = cb2.ci_upper
            cols["CB_2_lower"] = cb2.cb_lower
            cols["CB_2_upper"] = cb2.cb_upper
        return pd.DataFrame(cols)

    def summary(self) -> str:
        """Format human-readable TimeROC summary."""
        df = self.to_dataframe()
        header = (
            f"Time-Dependent IPCW ROC Summary (n={self.n_subjects}, cause={self.cause}, "
            f"competing_risks={self.has_competing_risks}, weighting='{self.weighting}')\n"
            + "=" * 78
        )
        return header + "\n" + df.round(4).to_string(index=False)


@dataclass(frozen=True)
class CompareTimeROCResult:
    """Paired two-biomarker comparison result (`compare_timeroc`, Blanche et al. 2013)."""

    definition: int
    times: np.ndarray
    auc_1: np.ndarray
    auc_2: np.ndarray
    delta_auc: np.ndarray
    se_delta: np.ndarray
    z_stat: np.ndarray
    p_values: np.ndarray
    p_values_adjusted: np.ndarray
    cov_delta: np.ndarray
    wald_stat: float
    wald_df: int
    p_wald_joint: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.times,
                "AUC_marker1": self.auc_1,
                "AUC_marker2": self.auc_2,
                "delta_AUC": self.delta_auc,
                "SE_delta": self.se_delta,
                "z_stat": self.z_stat,
                "p_value": self.p_values,
                "p_value_adjusted": self.p_values_adjusted,
            }
        )

    def summary(self) -> str:
        df = self.to_dataframe().round(4)
        return (
            f"Paired TimeROC Comparison (Definition {self.definition})\n"
            + "=" * 68
            + "\n"
            + df.to_string(index=False)
            + f"\nJoint Multi-Horizon Wald Test: chi2 = {self.wald_stat:.4f} (df = {self.wald_df}), p = {self.p_wald_joint:.4e}"
        )


def _reverse_kaplan_meier(T: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reverse Kaplan-Meier censoring survival (events precede censorings at ties).

    At each distinct censoring time u_m (where at least one delta_i == 0):
      d_m^C = #{i : T_i = u_m, delta_i = 0}
      Y_m^C = #{i : T_i > u_m} + d_m^C
      G(t)  = prod_{u_m <= t} (1 - d_m^C / Y_m^C)
      G(t-) = prod_{u_m <  t} (1 - d_m^C / Y_m^C)

    Returns
    -------
    u_cens : (M_c,) sorted unique censoring times
    d_cens : (M_c,) censoring counts at u_cens
    y_cens : (M_c,) censoring risk-set sizes at u_cens
    g_after : (M_c,) G(u_m) right-continuous values after each censoring jump
    """
    cens_mask = delta == 0
    if not np.any(cens_mask):
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    u_cens, d_counts = np.unique(T[cens_mask], return_counts=True)
    d_cens = d_counts.astype(np.float64)
    sorted_T = np.sort(T)
    n_strictly_greater = T.size - np.searchsorted(sorted_T, u_cens, side="right")
    y_cens = n_strictly_greater.astype(np.float64) + d_cens

    factors = np.maximum(1.0 - d_cens / np.maximum(y_cens, 1.0), 0.0)
    g_after = np.cumprod(factors)
    return u_cens, d_cens, y_cens, g_after


def _eval_km_at(u_cens: np.ndarray, g_after: np.ndarray, query: np.ndarray, left_limit: bool) -> np.ndarray:
    """Evaluate G(q-) (`left_limit=True`, strictly u_m < q) or G(q) (`left_limit=False`, u_m <= q)."""
    q_arr = np.asarray(query, dtype=np.float64)
    if u_cens.size == 0:
        return np.ones_like(q_arr, dtype=np.float64)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(u_cens, q_arr, side=side) - 1
    out = np.ones_like(q_arr, dtype=np.float64)
    valid = idx >= 0
    out[valid] = g_after[idx[valid]]
    return out


def _fit_cox_censoring_survival(
    T: np.ndarray,
    delta: np.ndarray,
    X_cov: np.ndarray,
    eval_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Breslow Cox PH model for censoring (delta == 0) and return `G_minus_subject (n,)` and `G_t (n, L)`."""
    n, p = X_cov.shape
    cens_ind = (delta == 0).astype(np.float64)
    n_cens = int(np.sum(cens_ind))
    if n_cens == 0:
        return np.ones(n, dtype=np.float64), np.ones((n, eval_times.size), dtype=np.float64)

    # Standardize covariates for numerical stability during Newton-Raphson
    mu = np.mean(X_cov, axis=0)
    sd = np.std(X_cov, axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X_cov - mu[None, :]) / sd_safe[None, :]

    # Reverse-KM risk sets at distinct censoring times: Y_i^C(u_m) = I(T_i > u_m or (T_i == u_m and delta_i == 0))
    u_cens = np.unique(T[delta == 0])
    beta = np.zeros(p, dtype=np.float64)

    # Precompute risk-set membership matrix (M_c, n) for fast exact Breslow updates
    in_risk = (T[None, :] > u_cens[:, None]) | ((T[None, :] == u_cens[:, None]) & (delta[None, :] == 0))
    is_event = (T[None, :] == u_cens[:, None]) & (delta[None, :] == 0)
    d_m = np.sum(is_event, axis=1).astype(np.float64)  # (M_c,)
    sum_z_events = is_event.astype(np.float64) @ Z  # (M_c, p)

    def _neg_loglik_grad_hess(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(Z @ b, -50.0, 50.0)
        exp_eta = np.exp(eta)
        risk_weights = in_risk.astype(np.float64) * exp_eta[None, :]  # (M_c, n)
        s0 = np.sum(risk_weights, axis=1)  # (M_c,)
        s1 = risk_weights @ Z  # (M_c, p)
        e_bar = s1 / s0[:, None]  # (M_c, p)

        nll = float(-np.sum(np.sum(sum_z_events * b[None, :], axis=1) - d_m * np.log(s0)))
        grad = -np.sum(sum_z_events - d_m[:, None] * e_bar, axis=0)

        # Hessian = sum_m d_m * (S2 / S0 - e_bar e_bar^T)
        s2 = np.einsum("mi,ip,iq->mpq", risk_weights, Z, Z)
        cov_m = s2 / s0[:, None, None] - np.einsum("mp,mq->mpq", e_bar, e_bar)
        hess = np.einsum("m,mpq->pq", d_m, cov_m)
        return nll, grad, hess

    nll_curr, grad, hess = _neg_loglik_grad_hess(beta)
    for _ in range(30):
        if np.linalg.norm(grad, ord=np.inf) < 1e-8:
            break
        reg = 1e-8 * np.eye(p)
        step = -np.linalg.solve(hess + reg, grad)
        alpha = 1.0
        while alpha > 1e-6:
            cand = beta + alpha * step
            nll_cand, grad_cand, hess_cand = _neg_loglik_grad_hess(cand)
            if nll_cand <= nll_curr + 1e-12:
                beta, nll_curr, grad, hess = cand, nll_cand, grad_cand, hess_cand
                break
            alpha *= 0.5
        else:
            break
        if np.linalg.norm(alpha * step) < 1e-9:
            break

    eta_hat = np.clip(Z @ beta, -50.0, 50.0)
    exp_eta_hat = np.exp(eta_hat)
    s0_hat = np.sum(in_risk.astype(np.float64) * exp_eta_hat[None, :], axis=1)
    d_lambda0 = d_m / np.maximum(s0_hat, 1e-300)
    cum_lambda0 = np.cumsum(d_lambda0)  # (M_c,)

    # G(T_i- | Z_i): sum over u_m < T_i
    idx_minus = np.searchsorted(u_cens, T, side="left") - 1
    lam0_minus = np.where(idx_minus >= 0, cum_lambda0[np.maximum(idx_minus, 0)], 0.0)
    g_minus_subj = np.exp(-lam0_minus * exp_eta_hat)

    # G(t_l | Z_i): sum over u_m <= t_l
    idx_t = np.searchsorted(u_cens, eval_times, side="right") - 1
    lam0_t = np.where(idx_t >= 0, cum_lambda0[np.maximum(idx_t, 0)], 0.0)  # (L,)
    g_t_subj = np.exp(-exp_eta_hat[:, None] * lam0_t[None, :])  # (n, L)

    return g_minus_subj, g_t_subj


def _concordance_partials(
    marker_inv: np.ndarray,
    n_uniq_m: int,
    w1: np.ndarray,
    wc: np.ndarray,
    n: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute O(n) Mann-Whitney concordance numerator and per-subject partial averages A_C(M_i), B_1(M_i)."""
    wc_grp = np.bincount(marker_inv, weights=wc, minlength=n_uniq_m)
    w1_grp = np.bincount(marker_inv, weights=w1, minlength=n_uniq_m)

    wc_less = np.concatenate([[0.0], np.cumsum(wc_grp[:-1])])
    a_c_uniq = (wc_less + 0.5 * wc_grp) / float(n)
    a_c = a_c_uniq[marker_inv]

    w1_greater = np.concatenate([np.cumsum(w1_grp[1:][::-1])[::-1], [0.0]])
    b_1_uniq = (w1_greater + 0.5 * w1_grp) / float(n)
    b_1 = b_1_uniq[marker_inv]

    mu = float(np.mean(w1 * a_c))
    return mu, a_c, b_1


def _compute_auc_iid_single_horizon(
    T: np.ndarray,
    delta: np.ndarray,
    horizon: float,
    w1: np.ndarray,
    wc: np.ndarray,
    auc_hat: float,
    a_c: np.ndarray,
    b_1: np.ndarray,
    u_cens: np.ndarray,
    d_cens: np.ndarray,
    y_cens: np.ndarray,
) -> np.ndarray:
    """Compute exact O(n) Blanche et al. (2013) IID influence vector `phi_i(t)` at one horizon `t`."""
    n = T.size
    p1 = float(np.mean(w1))
    pc = float(np.mean(wc))
    if not np.isfinite(auc_hat) or p1 <= 0.0 or pc <= 0.0:
        return np.full(n, np.nan, dtype=np.float64)

    # Term I: U-statistic Hajek projection treating censoring survival G(.) as known
    tilde_a_c = a_c - auc_hat * pc
    tilde_b_1 = b_1 - auc_hat * p1
    u_known = w1 * tilde_a_c + wc * tilde_b_1  # sum_i u_known[i] == 0 identically

    # Term II: Kaplan-Meier censoring-weight martingale perturbation over u_m <= horizon
    m_active = int(np.searchsorted(u_cens, horizon, side="right"))
    if m_active == 0:
        return u_known / (p1 * pc)

    u_sub = u_cens[:m_active]
    d_sub = d_cens[:m_active]
    y_sub = y_cens[:m_active]

    # H_m(t) = (1 / Y_m^C) * sum_{k : T_k > u_m} u_known[k] for m = 0 .. m_active - 1
    # Note T_k > u_sub[m] <=> m < searchsorted(u_sub, T_k, side="left")
    pos_strict = np.searchsorted(u_sub, T, side="left")  # in {0, ..., m_active}
    bucket_u = np.bincount(pos_strict, weights=u_known, minlength=m_active + 1)
    # sum_{k : pos_strict[k] > m} u_known[k] = suffix sum of bucket_u[m+1 :]
    sum_u_gt = np.cumsum(bucket_u[1:][::-1])[::-1]  # length m_active
    h_m = sum_u_gt / np.maximum(y_sub, 1.0)  # (m_active,)

    # M_i^corr(t) = sum_{m < m_active} h_m * [I(T_i == u_m, delta_i == 0) - Y_i^C(u_m) * d_m / Y_m^C]
    # Part A: jump at censoring event T_i == u_m (with delta_i == 0 and T_i <= horizon)
    m_corr = np.zeros(n, dtype=np.float64)
    is_cens_before_t = (delta == 0) & (T <= horizon)
    if np.any(is_cens_before_t):
        cens_idx = np.searchsorted(u_sub, T[is_cens_before_t], side="left")
        m_corr[is_cens_before_t] += h_m[cens_idx]

    # Part B: integrated compensator -sum_{m < m_active} (h_m * d_m / y_m) * Y_i^C(u_m)
    # Recall Y_i^C(u_m) = 1 iff u_m < T_i OR (u_m == T_i and delta_i == 0)!
    g_m = h_m * d_sub / np.maximum(y_sub, 1.0)
    prefix_g = np.concatenate([[0.0], np.cumsum(g_m)])  # length m_active + 1
    # Number of m in {0..m_active-1} included in subject i's censoring risk set:
    upper_m = np.where(
        delta == 0,
        np.searchsorted(u_sub, T, side="right"),
        np.searchsorted(u_sub, T, side="left"),
    )
    m_corr -= prefix_g[upper_m]

    return (u_known + m_corr) / (p1 * pc)


def time_roc(
    T: Sequence[float] | np.ndarray,
    delta: Sequence[int] | np.ndarray,
    marker: Sequence[float] | np.ndarray,
    times: Sequence[float] | np.ndarray,
    *,
    cause: int = 1,
    weighting: Literal["marginal", "cox"] = "marginal",
    other_markers: np.ndarray | pd.DataFrame | None = None,
    cutpoints: Sequence[float] | np.ndarray | None = None,
    iid: bool | None = None,
) -> TimeROCResult:
    """Estimate IPCW time-dependent ROC curves, competing-risks AUCs, and IID influence functions."""
    t_arr = np.asarray(T, dtype=np.float64).ravel()
    d_raw = np.asarray(delta)
    if d_raw.ndim != 1 and d_raw.size != t_arr.size:
        d_raw = d_raw.ravel()
    m_arr = np.asarray(marker, dtype=np.float64).ravel()
    horizons = np.asarray(times, dtype=np.float64).ravel()

    n = t_arr.size
    if n < 3 or d_raw.size != n or m_arr.size != n:
        raise ValueError("T, delta, and marker must be 1D arrays of identical length n >= 3.")
    if horizons.size == 0:
        raise ValueError("times must contain at least one evaluation horizon.")
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(m_arr)) and np.all(np.isfinite(horizons))):
        raise ValueError("T, marker, and times must contain only finite values (no NaN or Inf).")
    if np.any(t_arr < 0.0) or np.any(horizons <= 0.0):
        raise ValueError("Follow-up times T must be >= 0 and evaluation horizons times must be > 0.")
    if not np.all(np.isfinite(d_raw)):
        raise ValueError("delta must be finite non-negative integers.")

    d_float = np.asarray(d_raw, dtype=np.float64).ravel()
    if np.any(d_float < 0.0) or not np.all(d_float == np.floor(d_float)):
        raise ValueError("delta must contain non-negative integer event codes (0=censored, >=1=events).")
    d_arr = d_float.astype(np.int64)

    if int(cause) <= 0 or cause != int(cause):
        raise ValueError("cause must be a positive integer event code.")
    cause_int = int(cause)
    if not np.any(d_arr == cause_int):
        raise ValueError(f"No primary events with delta == {cause_int} observed in cohort.")

    if weighting not in ("marginal", "cox"):
        raise ValueError("weighting must be 'marginal' or 'cox'.")

    compute_iid = (weighting == "marginal") if iid is None else bool(iid)
    if compute_iid and weighting != "marginal":
        raise ValueError("Analytic IID representation requires weighting='marginal' (Blanche et al., 2013).")

    has_cr = bool(np.any((d_arr != 0) & (d_arr != cause_int)))
    n_times = horizons.size

    # Step 1: Estimate censoring survival weights G(T_i-) and G(t_l)
    if weighting == "marginal":
        u_cens, d_cens, y_cens, g_after = _reverse_kaplan_meier(t_arr, d_arr)
        g_minus_subj = _eval_km_at(u_cens, g_after, t_arr, left_limit=True)  # (n,)
        g_at_horizons = _eval_km_at(u_cens, g_after, horizons, left_limit=False)  # (L,)
        g_t_subj = np.broadcast_to(g_at_horizons[None, :], (n, n_times))
    else:
        u_cens = np.array([], dtype=np.float64)
        d_cens = np.array([], dtype=np.float64)
        y_cens = np.array([], dtype=np.float64)
        if other_markers is None:
            X_cov = m_arr[:, None]
        else:
            om = np.asarray(other_markers, dtype=np.float64)
            if om.ndim == 1:
                om = om[:, None]
            if om.shape[0] != n or not np.all(np.isfinite(om)):
                raise ValueError("other_markers must have n rows and finite values.")
            X_cov = np.column_stack([m_arr, om])
        g_minus_subj, g_t_subj = _fit_cox_censoring_survival(t_arr, d_arr, X_cov, horizons)

    # Step 2: Prepare ROC cutpoints & sorted marker buckets
    uniq_m, marker_inv = np.unique(m_arr, return_inverse=True)
    if cutpoints is None:
        cuts = np.concatenate([[-np.inf], uniq_m])
    else:
        cuts = np.asarray(cutpoints, dtype=np.float64).ravel()
        if cuts.size == 0 or np.any(np.isnan(cuts)):
            raise ValueError("cutpoints must be non-empty without NaN.")

    n_cuts = cuts.size
    se_mat = np.full((n_cuts, n_times), np.nan, dtype=np.float64)
    sp1_mat = np.full((n_cuts, n_times), np.nan, dtype=np.float64)
    fp1_mat = np.full((n_cuts, n_times), np.nan, dtype=np.float64)
    sp2_mat = np.full((n_cuts, n_times), np.nan, dtype=np.float64)
    fp2_mat = np.full((n_cuts, n_times), np.nan, dtype=np.float64)

    auc_1 = np.full(n_times, np.nan, dtype=np.float64)
    auc_2 = np.full(n_times, np.nan, dtype=np.float64)
    f_1 = np.full(n_times, np.nan, dtype=np.float64)
    f_2 = np.full(n_times, np.nan, dtype=np.float64)
    s_prob = np.full(n_times, np.nan, dtype=np.float64)
    n_cases_vec = np.zeros(n_times, dtype=np.int64)
    n_surv_vec = np.zeros(n_times, dtype=np.int64)
    n_comp_vec = np.zeros(n_times, dtype=np.int64)

    iid_1 = np.full((n, n_times), np.nan, dtype=np.float64) if compute_iid else None
    iid_2 = np.full((n, n_times), np.nan, dtype=np.float64) if compute_iid else None

    # Precompute index in `uniq_m` for threshold evaluation M_i > c:
    # M_i > c <=> uniq_m[k] > c <=> k >= searchsorted(uniq_m, c, side="right")
    cut_pos = np.searchsorted(uniq_m, cuts, side="right")  # in {0, ..., uniq_m.size}

    for l_idx, th in enumerate(horizons):
        case_mask = (t_arr <= th) & (d_arr == cause_int)
        comp_mask = (t_arr <= th) & (d_arr != 0) & (d_arr != cause_int)
        surv_mask = t_arr > th

        n_cases_vec[l_idx] = int(np.sum(case_mask))
        n_comp_vec[l_idx] = int(np.sum(comp_mask))
        n_surv_vec[l_idx] = int(np.sum(surv_mask))

        g_t_col = g_t_subj[:, l_idx]
        if np.any(case_mask & (g_minus_subj <= 0.0)) or np.any(comp_mask & (g_minus_subj <= 0.0)):
            continue
        if np.any(surv_mask & (g_t_col <= 0.0)):
            continue

        w1 = np.where(case_mask, 1.0 / np.maximum(g_minus_subj, 1e-300), 0.0)
        w2 = np.where(comp_mask, 1.0 / np.maximum(g_minus_subj, 1e-300), 0.0)
        w0 = np.where(surv_mask, 1.0 / np.maximum(g_t_col, 1e-300), 0.0)
        wc1 = w0 + w2  # Definition 1 controls: survivors + competing events
        wc2 = w0  # Definition 2 controls: strictly event-free survivors

        f_1[l_idx] = float(np.mean(w1))
        f_2[l_idx] = float(np.mean(w2))
        s_prob[l_idx] = float(np.mean(w0))

        sum_w1 = float(np.sum(w1))
        sum_wc1 = float(np.sum(wc1))
        sum_wc2 = float(np.sum(wc2))

        if sum_w1 > 0.0:
            w1_grp = np.bincount(marker_inv, weights=w1, minlength=uniq_m.size)
            suffix_w1 = np.concatenate([np.cumsum(w1_grp[::-1])[::-1], [0.0]])
            se_mat[:, l_idx] = suffix_w1[cut_pos] / sum_w1

        if sum_wc1 > 0.0:
            wc1_grp = np.bincount(marker_inv, weights=wc1, minlength=uniq_m.size)
            suffix_wc1 = np.concatenate([np.cumsum(wc1_grp[::-1])[::-1], [0.0]])
            fp1 = suffix_wc1[cut_pos] / sum_wc1
            fp1_mat[:, l_idx] = fp1
            sp1_mat[:, l_idx] = 1.0 - fp1

        if sum_wc2 > 0.0:
            wc2_grp = np.bincount(marker_inv, weights=wc2, minlength=uniq_m.size)
            suffix_wc2 = np.concatenate([np.cumsum(wc2_grp[::-1])[::-1], [0.0]])
            fp2 = suffix_wc2[cut_pos] / sum_wc2
            fp2_mat[:, l_idx] = fp2
            sp2_mat[:, l_idx] = 1.0 - fp2

        if sum_w1 > 0.0 and sum_wc1 > 0.0:
            mu1, ac1, b1_1 = _concordance_partials(marker_inv, uniq_m.size, w1, wc1, n)
            auc_1[l_idx] = mu1 / (f_1[l_idx] * float(np.mean(wc1)))
            if compute_iid and iid_1 is not None:
                iid_1[:, l_idx] = _compute_auc_iid_single_horizon(
                    t_arr, d_arr, float(th), w1, wc1, auc_1[l_idx], ac1, b1_1, u_cens, d_cens, y_cens
                )

        if sum_w1 > 0.0 and sum_wc2 > 0.0:
            mu2, ac2, b1_2 = _concordance_partials(marker_inv, uniq_m.size, w1, wc2, n)
            auc_2[l_idx] = mu2 / (f_1[l_idx] * float(np.mean(wc2)))
            if compute_iid and iid_2 is not None:
                iid_2[:, l_idx] = _compute_auc_iid_single_horizon(
                    t_arr, d_arr, float(th), w1, wc2, auc_2[l_idx], ac2, b1_2, u_cens, d_cens, y_cens
                )

    cov_1: np.ndarray | None = None
    cov_2: np.ndarray | None = None
    se_1: np.ndarray | None = None
    se_2: np.ndarray | None = None
    if compute_iid and iid_1 is not None and iid_2 is not None:
        cov_1 = (iid_1.T @ iid_1) / (float(n) ** 2)
        cov_2 = (iid_2.T @ iid_2) / (float(n) ** 2)
        se_1 = np.sqrt(np.maximum(np.diag(cov_1), 0.0))
        se_2 = np.sqrt(np.maximum(np.diag(cov_2), 0.0))

    return TimeROCResult(
        n_subjects=n,
        cause=cause_int,
        has_competing_risks=has_cr,
        weighting=weighting,
        times=horizons.copy(),
        n_cases=n_cases_vec,
        n_survivors=n_surv_vec,
        n_competing=n_comp_vec,
        cum_inc_1=f_1,
        cum_inc_2=f_2,
        surv_prob=s_prob,
        auc_1=auc_1,
        auc_2=auc_2,
        se_auc_1=se_1,
        se_auc_2=se_2,
        cov_auc_1=cov_1,
        cov_auc_2=cov_2,
        iid_auc_1=iid_1,
        iid_auc_2=iid_2,
        cutpoints=cuts,
        se=se_mat,
        sp_1=sp1_mat,
        fp_1=fp1_mat,
        sp_2=sp2_mat,
        fp_2=fp2_mat,
        _T=t_arr.copy(),
        _delta=d_arr.copy(),
    )


def compare_timeroc(
    fit1: TimeROCResult,
    fit2: TimeROCResult,
    *,
    definition: Literal[1, 2] = 1,
    n_sim: int = 2000,
    seed: int | np.random.Generator | None = 42,
) -> CompareTimeROCResult:
    """Compare two rival biomarkers measured on the same cohort via paired IID influence differences."""
    if definition not in (1, 2):
        raise ValueError("definition must be 1 or 2.")
    if fit1.n_subjects != fit2.n_subjects or fit1.cause != fit2.cause:
        raise ValueError("fit1 and fit2 must be fit on the exact same cohort and primary cause.")
    if not (np.allclose(fit1._T, fit2._T, atol=1e-12) and np.array_equal(fit1._delta, fit2._delta)):
        raise ValueError("fit1 and fit2 follow-up times T and event indicators delta do not match.")
    if fit1.times.shape != fit2.times.shape or not np.allclose(fit1.times, fit2.times, atol=1e-12):
        raise ValueError("fit1 and fit2 must be evaluated at identical horizons `times`.")

    phi1 = fit1.iid_auc_1 if definition == 1 else fit1.iid_auc_2
    phi2 = fit2.iid_auc_1 if definition == 1 else fit2.iid_auc_2
    if phi1 is None or phi2 is None:
        raise ValueError("Both fit1 and fit2 must have IID representations (iid=True, weighting='marginal').")

    auc1 = fit1.auc_1 if definition == 1 else fit1.auc_2
    auc2 = fit2.auc_1 if definition == 1 else fit2.auc_2
    delta_auc = auc1 - auc2
    delta_phi = phi1 - phi2  # (n, L)

    n = fit1.n_subjects
    cov_delta = (delta_phi.T @ delta_phi) / (float(n) ** 2)
    se_delta = np.sqrt(np.maximum(np.diag(cov_delta), 0.0))

    z_stat = np.full_like(delta_auc, np.nan)
    p_vals = np.full_like(delta_auc, np.nan)
    for l_idx in range(delta_auc.size):
        if np.isfinite(delta_auc[l_idx]) and np.isfinite(se_delta[l_idx]):
            if se_delta[l_idx] > 0.0:
                z_stat[l_idx] = delta_auc[l_idx] / se_delta[l_idx]
                p_vals[l_idx] = float(2.0 * stats.norm.sf(abs(z_stat[l_idx])))
            elif abs(delta_auc[l_idx]) == 0.0:
                z_stat[l_idx] = 0.0
                p_vals[l_idx] = 1.0

    # Multiplicity-adjusted simultaneous p-values across horizons via Gaussian multiplier max-|Z|
    p_adj = p_vals.copy()
    valid = np.isfinite(z_stat) & (se_delta > 0.0)
    if int(np.sum(valid)) >= 2:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        g_mult = rng.standard_normal((n_sim, n))
        z_boot = (g_mult @ delta_phi[:, valid]) / (float(n) * se_delta[valid][None, :])
        sup_abs = np.max(np.abs(z_boot), axis=1)
        valid_indices = np.flatnonzero(valid)
        for idx_col in valid_indices:
            p_adj[idx_col] = float((np.sum(sup_abs >= abs(z_stat[idx_col])) + 1.0) / (n_sim + 1.0))

    # Joint multi-timepoint Wald chi-square test using rank-revealing Moore-Penrose pseudoinverse
    finite_mask = np.isfinite(delta_auc) & np.all(np.isfinite(cov_delta), axis=1)
    if not np.any(finite_mask):
        wald_stat, wald_df, p_wald = float("nan"), 0, float("nan")
    else:
        d_sub = delta_auc[finite_mask]
        c_sub = 0.5 * (cov_delta[np.ix_(finite_mask, finite_mask)] + cov_delta[np.ix_(finite_mask, finite_mask)].T)
        eigs, vecs = np.linalg.eigh(c_sub)
        max_eig = float(np.max(eigs)) if eigs.size > 0 else 0.0
        if max_eig <= 1e-20:
            wald_stat = 0.0 if np.allclose(d_sub, 0.0) else float("inf")
            wald_df = int(d_sub.size)
            p_wald = 1.0 if wald_stat == 0.0 else 0.0
        else:
            keep_eigs = eigs > 1e-10 * max_eig
            wald_df = int(np.sum(keep_eigs))
            proj = vecs[:, keep_eigs].T @ d_sub
            wald_stat = float(np.sum((proj**2) / eigs[keep_eigs]))
            p_wald = float(stats.chi2.sf(wald_stat, df=wald_df))

    return CompareTimeROCResult(
        definition=definition,
        times=fit1.times.copy(),
        auc_1=auc1.copy(),
        auc_2=auc2.copy(),
        delta_auc=delta_auc,
        se_delta=se_delta,
        z_stat=z_stat,
        p_values=p_vals,
        p_values_adjusted=p_adj,
        cov_delta=cov_delta,
        wald_stat=wald_stat,
        wald_df=wald_df,
        p_wald_joint=p_wald,
    )
