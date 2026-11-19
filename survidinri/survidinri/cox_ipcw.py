"""Weighted Cox Proportional Hazards, Kaplan-Meier censoring survival, and IPCW utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate mandatory ``settings`` parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings '{settings}'. Must be one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class SortedSurvivalIndex:
    """Pre-sorted survival time structure for fast repeated perturbation evaluation."""

    n: int
    order: np.ndarray
    inv_order: np.ndarray
    time_s: np.ndarray
    event_s: np.ndarray
    uniq_times: np.ndarray
    grp_starts: np.ndarray
    grp_ends: np.ndarray
    le_t0_groups: np.ndarray
    t0: float


def build_survival_index(time: np.ndarray, event: np.ndarray, t0: float) -> SortedSurvivalIndex:
    """Pre-sort observations by ascending time (events before censorings at ties)."""
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    n = int(t.size)
    if n == 0 or e.size != n:
        raise ValueError("time and event must be non-empty 1D arrays of equal length.")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("All survival/censoring times must be finite and positive.")
    if not np.all(np.isin(e, [0, 1])):
        raise ValueError("event indicator must contain only 0 (censored) and 1 (event).")
    if not np.isfinite(t0) or t0 <= 0:
        raise ValueError(f"Horizon t0 must be positive and finite, got {t0}.")
    if not np.any((t <= t0) & (e == 1)):
        raise ValueError(f"No observed events occur at or before horizon t0={t0}.")
    if not np.any(t > t0):
        raise ValueError(f"No subjects remain under follow-up past horizon t0={t0}.")

    # Sort ascending by time; at tied times put events (-e == -1) before censorings (-e == 0)
    order = np.lexsort((-e, t))
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n, dtype=np.int64)

    time_s = t[order]
    event_s = e[order]
    uniq_times, grp_starts = np.unique(time_s, return_index=True)
    grp_ends = np.append(grp_starts[1:], n)
    le_t0_groups = uniq_times <= t0

    return SortedSurvivalIndex(
        n=n,
        order=order,
        inv_order=inv_order,
        time_s=time_s,
        event_s=event_s,
        uniq_times=uniq_times,
        grp_starts=grp_starts,
        grp_ends=grp_ends,
        le_t0_groups=le_t0_groups,
        t0=float(t0),
    )


def compute_ipcw_weights_indexed(
    surv_idx: SortedSurvivalIndex,
    obs_weights: np.ndarray,
    g_min: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute perturbed KM censoring survival G*(t) and exact IPCW case/control weights.

    Uses standard survival tie convention where events precede censorings at tied times:
    - Case weight (subject i with X_i <= t0, Delta_i == 1): ``1 / G*(X_i-)``
    - Control weight (subject i with X_i > t0): ``1 / G*(t0)``

    Returns arrays in the ORIGINAL (unsorted) subject order.
    """
    xi_s = obs_weights[surv_idx.order]
    is_event = surv_idx.event_s == 1
    is_cens = ~is_event

    # Total weight per distinct time group, and censoring/event weights per group
    grp_total = np.add.reduceat(xi_s, surv_idx.grp_starts)
    grp_event = np.add.reduceat(xi_s * is_event, surv_idx.grp_starts)
    grp_cens = np.add.reduceat(xi_s * is_cens, surv_idx.grp_starts)

    # Risk set at start of group u: sum_{X_j >= u} xi_j
    risk_ge_u = np.cumsum(grp_total[::-1])[::-1]
    # Since events at u occur just before censorings at u, risk set for censoring at u is:
    risk_cens_u = np.maximum(risk_ge_u - grp_event, 0.0)

    haz_cens = np.where(
        (grp_cens > 0.0) & (risk_cens_u > 0.0),
        np.clip(grp_cens / np.maximum(risk_cens_u, 1e-15), 0.0, 1.0 - 1e-12),
        0.0,
    )
    one_minus_h = 1.0 - haz_cens
    g_after = np.cumprod(one_minus_h)
    g_before = np.concatenate([[1.0], g_after[:-1]])

    g_before_safe = np.maximum(g_before, g_min)
    if np.any(surv_idx.le_t0_groups):
        last_le = int(np.nonzero(surv_idx.le_t0_groups)[0][-1])
        g_t0 = float(max(g_after[last_le], g_min))
    else:
        g_t0 = 1.0

    # Map group-level G*(X_i-) back to sorted observations
    grp_lengths = surv_idx.grp_ends - surv_idx.grp_starts
    g_minus_obs_s = np.repeat(g_before_safe, grp_lengths)

    le_t0_obs_s = surv_idx.time_s <= surv_idx.t0
    w1_s = np.where(le_t0_obs_s & is_event, 1.0 / g_minus_obs_s, 0.0)
    w0_s = np.where(~le_t0_obs_s, 1.0 / g_t0, 0.0)

    return w1_s[surv_idx.inv_order], w0_s[surv_idx.inv_order], g_t0


def fit_weighted_cox_risk_indexed(
    surv_idx: SortedSurvivalIndex,
    z_std_s: np.ndarray,
    obs_weights: np.ndarray,
    ties: Literal["breslow", "efron"] = "efron",
    init_beta: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit observation-weighted Cox PH on pre-sorted standardized covariates.

    Parameters
    ----------
    surv_idx : SortedSurvivalIndex
        Pre-sorted survival time structure.
    z_std_s : np.ndarray
        Standardized covariate matrix in ``surv_idx.order``, shape ``(n, p)``.
    obs_weights : np.ndarray
        Non-negative observation weights in original subject order (e.g. ``1`` or ``Exp(1)``).
    ties : {"breslow", "efron"}
        Tie-breaking approximation for partial likelihood and baseline hazard.
    init_beta : np.ndarray | None
        Optional warm-start parameter vector of shape ``(p,)``.

    Returns
    -------
    beta_std : np.ndarray
        Fitted coefficient vector on standardized covariates, shape ``(p,)``.
    cumhaz_t0 : float
        Baseline cumulative hazard ``Lambda_0(t0)`` evaluated at ``z_std == 0``.
    risk_t0 : np.ndarray
        Predicted event probabilities ``1 - exp(-Lambda_0(t0) * exp(z_std @ beta))`` in
        ORIGINAL subject order, shape ``(n,)``.
    """
    xi_s = obs_weights[surv_idx.order]
    n, p = z_std_s.shape
    is_event = surv_idx.event_s == 1

    # Null (intercept-only / zero-covariate) Cox model reduces to Nelson-Aalen cumulative hazard
    if p == 0:
        grp_tot = np.add.reduceat(xi_s, surv_idx.grp_starts)
        grp_ev = np.add.reduceat(xi_s * is_event, surv_idx.grp_starts)
        risk_set = np.cumsum(grp_tot[::-1])[::-1]
        dhaz = np.where(
            surv_idx.le_t0_groups & (grp_ev > 0) & (risk_set > 0),
            grp_ev / np.maximum(risk_set, 1e-15),
            0.0,
        )
        ch_t0 = float(np.sum(dhaz))
        prob = float(1.0 - np.exp(-ch_t0))
        return np.zeros(0, dtype=float), ch_t0, np.full(n, prob, dtype=float)

    beta = np.zeros(p, dtype=float) if init_beta is None else np.asarray(init_beta, dtype=float).copy()
    starts = surv_idx.grp_starts
    ends = surv_idx.grp_ends
    n_grps = starts.size

    # Pre-identify event groups and separate untied (or Breslow) vs Efron multi-tie groups
    ev_counts = np.add.reduceat(is_event.astype(np.int64), starts)
    ev_grp_idx = np.flatnonzero(ev_counts > 0)
    tied_ev_grps = ev_grp_idx[ev_counts[ev_grp_idx] > 1]
    untied_ev_grps = ev_grp_idx[ev_counts[ev_grp_idx] == 1]
    eye_p = np.eye(p, dtype=float) * 1e-8

    def _neg_log_pl_grad_hess(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(z_std_s @ b, -30.0, 30.0)
        exp_eta = xi_s * np.exp(eta)
        wz = exp_eta[:, None] * z_std_s  # (n, p)
        wzz = wz[:, :, None] * z_std_s[:, None, :]  # (n, p, p)

        grp_s0 = np.add.reduceat(exp_eta, starts)
        grp_s1 = np.add.reduceat(wz, starts, axis=0)
        grp_s2 = np.add.reduceat(wzz, starts, axis=0)

        rs0 = np.cumsum(grp_s0[::-1])[::-1]
        rs1 = np.cumsum(grp_s1[::-1], axis=0)[::-1]
        rs2 = np.cumsum(grp_s2[::-1], axis=0)[::-1]

        xi_ev_obs = np.where(is_event, xi_s, 0.0)
        ll = float(np.sum(xi_ev_obs * eta))
        grad = np.sum(xi_ev_obs[:, None] * z_std_s, axis=0)

        grp_d_xi = np.add.reduceat(xi_ev_obs, starts)

        if ties == "breslow" or tied_ev_grps.size == 0:
            # Fully vectorized over all event groups!
            d_g = grp_d_xi[ev_grp_idx]
            s0_g = np.maximum(rs0[ev_grp_idx], 1e-15)
            s1_g = rs1[ev_grp_idx]
            s2_g = rs2[ev_grp_idx]
            mean_z = s1_g / s0_g[:, None]
            ll -= float(np.sum(d_g * np.log(s0_g)))
            grad -= np.sum(d_g[:, None] * mean_z, axis=0)
            info_terms = s2_g / s0_g[:, None, None] - mean_z[:, :, None] * mean_z[:, None, :]
            hess = np.sum(d_g[:, None, None] * info_terms, axis=0)
            return -ll, -grad, hess

        hess = np.zeros((p, p), dtype=float)
        if untied_ev_grps.size > 0:
            d_u = grp_d_xi[untied_ev_grps]
            s0_u = np.maximum(rs0[untied_ev_grps], 1e-15)
            s1_u = rs1[untied_ev_grps]
            s2_u = rs2[untied_ev_grps]
            mean_u = s1_u / s0_u[:, None]
            ll -= float(np.sum(d_u * np.log(s0_u)))
            grad -= np.sum(d_u[:, None] * mean_u, axis=0)
            info_u = s2_u / s0_u[:, None, None] - mean_u[:, :, None] * mean_u[:, None, :]
            hess += np.sum(d_u[:, None, None] * info_u, axis=0)

        for g in tied_ev_grps:
            lo, hi = int(starts[g]), int(ends[g])
            ev_mask = is_event[lo:hi]
            xi_ev = xi_s[lo:hi][ev_mask]
            d_xi = float(np.sum(xi_ev))
            if d_xi <= 0.0:
                continue
            z_ev = z_std_s[lo:hi][ev_mask]
            exp_ev = exp_eta[lo:hi][ev_mask]
            m_ties = int(xi_ev.size)

            s0_g_sc = float(rs0[g])
            s1_g_sc = rs1[g]
            s2_g_sc = rs2[g]

            d_s0 = float(np.sum(exp_ev))
            d_s1 = np.sum(exp_ev[:, None] * z_ev, axis=0)
            d_s2 = np.sum((exp_ev[:, None] * z_ev)[:, :, None] * z_ev[:, None, :], axis=0)
            fracs = np.arange(m_ties, dtype=float) / float(m_ties)
            s0_l = np.maximum(s0_g_sc - fracs * d_s0, 1e-15)  # (m_ties,)
            s1_l = s1_g_sc[None, :] - fracs[:, None] * d_s1[None, :]  # (m_ties, p)
            s2_l = s2_g_sc[None, :, :] - fracs[:, None, None] * d_s2[None, :, :]
            mean_l = s1_l / s0_l[:, None]
            w_step = d_xi / float(m_ties)
            ll -= w_step * float(np.sum(np.log(s0_l)))
            grad -= w_step * np.sum(mean_l, axis=0)
            hess += w_step * np.sum(
                s2_l / s0_l[:, None, None] - mean_l[:, :, None] * mean_l[:, None, :], axis=0
            )

        return -ll, -grad, hess

    nll, ngrad, hess = _neg_log_pl_grad_hess(beta)
    for _ in range(max_iter):
        step = -np.linalg.solve(hess + eye_p, ngrad)
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 3.0:
            step *= 3.0 / step_norm

        alpha_ls = 1.0
        accepted = False
        for _ in range(8):
            cand = beta + alpha_ls * step
            cand_nll, cand_ngrad, cand_hess = _neg_log_pl_grad_hess(cand)
            if np.isfinite(cand_nll) and cand_nll <= nll + 1e-9 * (1.0 + abs(nll)):
                beta = cand
                prev_nll = nll
                nll, ngrad, hess = cand_nll, cand_ngrad, cand_hess
                accepted = True
                break
            alpha_ls *= 0.5

        if not accepted:
            break
        if np.max(np.abs(alpha_ls * step)) < tol or abs(prev_nll - nll) < tol * (1.0 + abs(nll)):
            break

    # Compute baseline cumulative hazard at t0 (vectorized over untied/Breslow groups)
    eta_f = np.clip(z_std_s @ beta, -30.0, 30.0)
    exp_f = xi_s * np.exp(eta_f)
    grp_s0_f = np.add.reduceat(exp_f, starts)
    rs0_f = np.cumsum(grp_s0_f[::-1])[::-1]
    grp_d_xi_f = np.add.reduceat(np.where(is_event, xi_s, 0.0), starts)

    le_ev_grps = ev_grp_idx[surv_idx.le_t0_groups[ev_grp_idx]]
    if ties == "breslow" or tied_ev_grps.size == 0:
        cumhaz_t0 = float(np.sum(grp_d_xi_f[le_ev_grps] / np.maximum(rs0_f[le_ev_grps], 1e-15)))
    else:
        le_untied = untied_ev_grps[surv_idx.le_t0_groups[untied_ev_grps]]
        cumhaz_t0 = float(np.sum(grp_d_xi_f[le_untied] / np.maximum(rs0_f[le_untied], 1e-15)))
        le_tied = tied_ev_grps[surv_idx.le_t0_groups[tied_ev_grps]]
        for g in le_tied:
            lo, hi = int(starts[g]), int(ends[g])
            ev_mask = is_event[lo:hi]
            xi_ev = xi_s[lo:hi][ev_mask]
            d_xi = float(np.sum(xi_ev))
            if d_xi <= 0.0:
                continue
            m_ties = int(xi_ev.size)
            d_s0 = float(np.sum(exp_f[lo:hi][ev_mask]))
            fracs = np.arange(m_ties, dtype=float) / float(m_ties)
            cumhaz_t0 += (d_xi / float(m_ties)) * float(
                np.sum(1.0 / np.maximum(float(rs0_f[g]) - fracs * d_s0, 1e-15))
            )

    risk_s = 1.0 - np.exp(-cumhaz_t0 * np.exp(eta_f))
    risk_s = np.clip(risk_s, 0.0, 1.0)
    return beta, float(cumhaz_t0), risk_s[surv_idx.inv_order]


def standardize_covariates(
    covs: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert covariate input into standardized 2D array ``(n, p)`` with (means, scales)."""
    if covs is None:
        return np.zeros((n, 0), dtype=float), np.zeros(0), np.ones(0)
    arr = np.asarray(covs, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"Covariate matrix must have {n} rows, got shape {arr.shape}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Covariate matrix contains NaN or infinite values.")
    p = arr.shape[1]
    if p == 0:
        return np.zeros((n, 0), dtype=float), np.zeros(0), np.ones(0)
    means = np.mean(arr, axis=0)
    scales = np.std(arr, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (arr - means[None, :]) / scales[None, :], means, scales


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute exact step-CDF weighted median for non-negative weights."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    pos = w > 0.0
    if not np.any(pos):
        return 0.0
    vp = v[pos]
    wp = w[pos]
    if vp.size == 1:
        return float(vp[0])
    order = np.argsort(vp, kind="mergesort")
    vs = vp[order]
    ws = wp[order]
    total = float(np.sum(ws))
    if total <= 0.0:
        return float(np.median(vs))
    cum = np.cumsum(ws) / total
    idx = int(np.searchsorted(cum, 0.5 - 1e-12, side="left"))
    idx = min(idx, vs.size - 1)
    if abs(cum[idx] - 0.5) <= 1e-12 and (idx + 1) < vs.size:
        return float(0.5 * (vs[idx] + vs[idx + 1]))
    return float(vs[idx])
