"""Pairwise Gehan and Peron Kaplan-Meier scoring rules for prioritized endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .endpoints import OperatorKind

EPS_TOL = 1e-12


@dataclass(frozen=True)
class KMAtomDistribution:
    """Right-continuous Kaplan-Meier survival curve decomposed into jump atoms + censored tail."""

    jump_times: np.ndarray       # (K,) strictly increasing event times with positive KM drop
    jump_probs: np.ndarray       # (K,) discrete probability mass -dS(t_k) > 0
    surv_after_jumps: np.ndarray # (K,) right-continuous S(t_k)
    t_max: float                 # maximum observed follow-up time in arm/stratum
    tail_prob: float             # S(t_max) >= 0 (positive iff largest observation is right-censored)
    atom_times: np.ndarray       # (A,) jump_times followed by t_max if tail_prob > 0
    atom_status: np.ndarray      # (A,) 1 for event jump atoms, 0 for censored tail atom
    atom_probs: np.ndarray       # (A,) probability masses summing to 1.0

    def surv_at(self, t: np.ndarray) -> np.ndarray:
        """Evaluate right-continuous S(t) = P(T > t) at arbitrary query times ``t``."""
        t_arr = np.asarray(t, dtype=float)
        if self.jump_times.size == 0:
            return np.ones_like(t_arr, dtype=float)
        idx = np.searchsorted(self.jump_times, t_arr, side="right") - 1
        out = np.ones_like(t_arr, dtype=float)
        valid = idx >= 0
        out[valid] = self.surv_after_jumps[idx[valid]]
        return out

    def first_atom_after(self, x_cens: np.ndarray) -> np.ndarray:
        """Index into ``atom_times`` of the first atom strictly after censored time ``x_cens``.

        Note: if ``x_cens == t_max`` and ``tail_prob > 0``, all jump_times are ``<= t_max``,
        so ``searchsorted(jump_times, x_cens, side='right') == K``, pointing directly to
        the tail atom ``(t_max, status=0, prob=tail_prob)`` representing ``T > t_max``!
        """
        return np.searchsorted(self.jump_times, x_cens, side="right")


def fit_km_atoms(time: np.ndarray, status: np.ndarray) -> KMAtomDistribution:
    """Fit right-continuous Kaplan-Meier survival curve for one treatment arm within a stratum."""
    t = np.asarray(time, dtype=float).ravel()
    d = np.asarray(status, dtype=int).ravel()
    valid = np.isfinite(t) & np.isfinite(d)
    t = t[valid]
    d = d[valid]

    if t.size == 0:
        return KMAtomDistribution(
            jump_times=np.empty(0),
            jump_probs=np.empty(0),
            surv_after_jumps=np.empty(0),
            t_max=0.0,
            tail_prob=1.0,
            atom_times=np.array([0.0]),
            atom_status=np.array([0], dtype=int),
            atom_probs=np.array([1.0]),
        )

    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    ds = d[order]
    t_max = float(ts[-1])

    uniq_t, grp_starts = np.unique(ts, return_index=True)
    n_total = ts.size
    n_at_risk = n_total - grp_starts
    d_events = np.add.reduceat((ds == 1).astype(np.int64), grp_starts)

    ev_mask = d_events > 0
    if not np.any(ev_mask):
        return KMAtomDistribution(
            jump_times=np.empty(0),
            jump_probs=np.empty(0),
            surv_after_jumps=np.empty(0),
            t_max=t_max,
            tail_prob=1.0,
            atom_times=np.array([t_max]),
            atom_status=np.array([0], dtype=int),
            atom_probs=np.array([1.0]),
        )

    j_times = uniq_t[ev_mask]
    factors = 1.0 - d_events[ev_mask] / n_at_risk[ev_mask].astype(float)
    s_after = np.cumprod(factors)
    s_before = np.concatenate([[1.0], s_after[:-1]])
    j_probs = s_before - s_after
    tail_prob = float(max(0.0, s_after[-1]))

    if tail_prob > 1e-15:
        a_times = np.concatenate([j_times, [t_max]])
        a_status = np.concatenate([np.ones(j_times.size, dtype=int), [0]])
        a_probs = np.concatenate([j_probs, [tail_prob]])
    else:
        tail_prob = 0.0
        a_times = j_times.copy()
        a_status = np.ones(j_times.size, dtype=int)
        a_probs = j_probs.copy()

    return KMAtomDistribution(
        jump_times=j_times,
        jump_probs=j_probs,
        surv_after_jumps=s_after,
        t_max=t_max,
        tail_prob=tail_prob,
        atom_times=a_times,
        atom_status=a_status,
        atom_probs=a_probs,
    )


def score_pairs_continuous_binary(
    x_t: np.ndarray,
    x_c: np.ndarray,
    threshold: float = 0.0,
    operator: OperatorKind = "higher",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score all m x n pairs for a binary or continuous endpoint."""
    xt = np.asarray(x_t, dtype=float)[:, None]  # (m, 1)
    xc = np.asarray(x_c, dtype=float)[None, :]  # (1, n)

    miss = np.isnan(xt) | np.isnan(xc)
    diff = (xt - xc) if operator == "higher" else (xc - xt)
    tau_eff = max(float(threshold), EPS_TOL)

    win = np.where(miss, 0.0, (diff >= tau_eff - 1e-15).astype(float))
    loss = np.where(miss, 0.0, (diff <= -(tau_eff - 1e-15)).astype(float))
    uninf = miss.astype(float)
    neutral = np.maximum(0.0, 1.0 - win - loss - uninf)
    return win, loss, neutral, uninf


def score_pairs_tte_gehan(
    x_t: np.ndarray,
    d_t: np.ndarray,
    x_c: np.ndarray,
    d_c: np.ndarray,
    threshold: float = 0.0,
    operator: OperatorKind = "higher",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Gehan pairwise scoring for a right-censored TTE endpoint."""
    xt = np.asarray(x_t, dtype=float)[:, None]
    dt = np.asarray(d_t, dtype=float)[:, None]
    xc = np.asarray(x_c, dtype=float)[None, :]
    dc = np.asarray(d_c, dtype=float)[None, :]

    miss = np.isnan(xt) | np.isnan(xc) | np.isnan(dt) | np.isnan(dc)
    xt_s = np.where(miss, 0.0, xt)
    xc_s = np.where(miss, 0.0, xc)
    et = (~miss) & (dt == 1)
    ec = (~miss) & (dc == 1)
    ct = (~miss) & (dt == 0)
    cc = (~miss) & (dc == 0)

    tau = float(threshold)
    tau_pt = max(tau, EPS_TOL)

    # Case (1, 1): both uncensored -> strict > 0 when tau=0, >= tau when tau > 0
    win_11 = et & ec & ((xt_s - xc_s) >= (tau_pt - 1e-15 if tau > 0 else EPS_TOL))
    loss_11 = et & ec & ((xc_s - xt_s) >= (tau_pt - 1e-15 if tau > 0 else EPS_TOL))
    neut_11 = et & ec & (~win_11) & (~loss_11)

    # Case (0, 1): T_i > x_i and T_j == y_j -> if x_i >= y_j + tau, T_i > y_j + tau is certain!
    win_01 = ct & ec & ((xt_s - xc_s) >= tau - 1e-15)

    # Case (1, 0): T_i == x_i and T_j > y_j -> if y_j >= x_i + tau, T_j > x_i + tau is certain!
    loss_10 = et & cc & ((xc_s - xt_s) >= tau - 1e-15)

    win = (win_11 | win_01).astype(float)
    loss = (loss_11 | loss_10).astype(float)
    neutral = neut_11.astype(float)
    uninf = np.maximum(0.0, 1.0 - win - loss - neutral)

    if operator == "lower":
        return loss, win, neutral, uninf
    return win, loss, neutral, uninf


def score_pairs_tte_peron(
    x_t: np.ndarray,
    d_t: np.ndarray,
    x_c: np.ndarray,
    d_c: np.ndarray,
    km_t: KMAtomDistribution,
    km_c: KMAtomDistribution,
    threshold: float = 0.0,
    operator: OperatorKind = "higher",
    preserve_tail_uninf: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Peron et al. (2016) / Ozenne et al. (2021) Kaplan-Meier survival scoring rule.

    Computes conditional win, loss, neutral, and uninformative probabilities for every
    treatment-control pair ``(i, j)`` in ``O(K_T * K_C + m * n)`` time via exact Stieltjes
    suffix-sum integration over the arm-specific KM distributions ``km_t`` and ``km_c``.
    """
    xt = np.asarray(x_t, dtype=float)
    dt = np.asarray(d_t, dtype=float)
    xc = np.asarray(x_c, dtype=float)
    dc = np.asarray(d_c, dtype=float)
    m, n = xt.size, xc.size

    # Start from Gehan deterministic base (covers all (1, 1) pairs and resolved (0,1)/(1,0) pairs)
    win, loss, neutral, uninf = score_pairs_tte_gehan(
        xt, dt, xc, dc, threshold=threshold, operator="higher"
    )

    if not preserve_tail_uninf:
        # Optional Efron tail completion: treat tail atom as a point event at t_max
        if km_t.tail_prob > 0:
            st_atoms = np.ones_like(km_t.atom_status)
        else:
            st_atoms = km_t.atom_status
        if km_c.tail_prob > 0:
            sc_atoms = np.ones_like(km_c.atom_status)
        else:
            sc_atoms = km_c.atom_status
    else:
        st_atoms = km_t.atom_status
        sc_atoms = km_c.atom_status

    tau = float(threshold)
    miss_t = np.isnan(xt) | np.isnan(dt)
    miss_c = np.isnan(xc) | np.isnan(dc)

    # Case 2: Treatment censored (dt == 0), Control event (dc == 1), unresolved by Gehan (xt < xc + tau)
    ct_idx = np.flatnonzero((~miss_t) & (dt == 0))
    ec_idx = np.flatnonzero((~miss_c) & (dc == 1))
    if ct_idx.size > 0 and ec_idx.size > 0 and km_t.atom_times.size > 0:
        st_x = km_t.surv_at(xt[ct_idx])  # (m_c,)
        k0_t = km_t.first_atom_after(xt[ct_idx])  # (m_c,)
        # Compare all A_T atoms of Treatment against observed Control events xc[ec_idx]
        aw, al, an, au = score_pairs_tte_gehan(
            km_t.atom_times, st_atoms, xc[ec_idx], np.ones(ec_idx.size), threshold=tau, operator="higher"
        )
        pw = km_t.atom_probs[:, None] * aw
        pl = km_t.atom_probs[:, None] * al
        pn = km_t.atom_probs[:, None] * an
        pu = km_t.atom_probs[:, None] * au

        # 1D suffix sums over Treatment atoms (padded with trailing zero row)
        n_at = km_t.atom_times.size
        sw = np.vstack([np.cumsum(pw[::-1, :], axis=0)[::-1, :], np.zeros((1, ec_idx.size))])
        sl = np.vstack([np.cumsum(pl[::-1, :], axis=0)[::-1, :], np.zeros((1, ec_idx.size))])
        sn = np.vstack([np.cumsum(pn[::-1, :], axis=0)[::-1, :], np.zeros((1, ec_idx.size))])
        su = np.vstack([np.cumsum(pu[::-1, :], axis=0)[::-1, :], np.zeros((1, ec_idx.size))])

        k0_clamped = np.minimum(k0_t, n_at)
        sub_xt = xt[ct_idx][:, None]
        sub_xc = xc[ec_idx][None, :]
        unresolved = (sub_xt - sub_xc) < (tau - 1e-15)
        pos_surv = (st_x[:, None] > 1e-15) & unresolved
        denom = np.where(pos_surv, st_x[:, None], 1.0)

        ix = np.ix_(ct_idx, ec_idx)
        win[ix] = np.where(pos_surv, sw[k0_clamped, :] / denom, win[ix])
        loss[ix] = np.where(pos_surv, sl[k0_clamped, :] / denom, loss[ix])
        neutral[ix] = np.where(pos_surv, sn[k0_clamped, :] / denom, neutral[ix])
        uninf[ix] = np.where(pos_surv, su[k0_clamped, :] / denom, uninf[ix])

    # Case 3: Treatment event (dt == 1), Control censored (dc == 0), unresolved by Gehan (xc < xt + tau)
    et_idx = np.flatnonzero((~miss_t) & (dt == 1))
    cc_idx = np.flatnonzero((~miss_c) & (dc == 0))
    if et_idx.size > 0 and cc_idx.size > 0 and km_c.atom_times.size > 0:
        sc_y = km_c.surv_at(xc[cc_idx])  # (n_c,)
        l0_c = km_c.first_atom_after(xc[cc_idx])  # (n_c,)
        aw, al, an, au = score_pairs_tte_gehan(
            xt[et_idx], np.ones(et_idx.size), km_c.atom_times, sc_atoms, threshold=tau, operator="higher"
        )
        pw = aw * km_c.atom_probs[None, :]
        pl = al * km_c.atom_probs[None, :]
        pn = an * km_c.atom_probs[None, :]
        pu = au * km_c.atom_probs[None, :]

        n_ac = km_c.atom_times.size
        sw = np.hstack([np.cumsum(pw[:, ::-1], axis=1)[:, ::-1], np.zeros((et_idx.size, 1))])
        sl = np.hstack([np.cumsum(pl[:, ::-1], axis=1)[:, ::-1], np.zeros((et_idx.size, 1))])
        sn = np.hstack([np.cumsum(pn[:, ::-1], axis=1)[:, ::-1], np.zeros((et_idx.size, 1))])
        su = np.hstack([np.cumsum(pu[:, ::-1], axis=1)[:, ::-1], np.zeros((et_idx.size, 1))])

        l0_clamped = np.minimum(l0_c, n_ac)
        sub_xt = xt[et_idx][:, None]
        sub_xc = xc[cc_idx][None, :]
        unresolved = (sub_xc - sub_xt) < (tau - 1e-15)
        pos_surv = (sc_y[None, :] > 1e-15) & unresolved
        denom = np.where(pos_surv, sc_y[None, :], 1.0)

        ix = np.ix_(et_idx, cc_idx)
        win[ix] = np.where(pos_surv, sw[:, l0_clamped] / denom, win[ix])
        loss[ix] = np.where(pos_surv, sl[:, l0_clamped] / denom, loss[ix])
        neutral[ix] = np.where(pos_surv, sn[:, l0_clamped] / denom, neutral[ix])
        uninf[ix] = np.where(pos_surv, su[:, l0_clamped] / denom, uninf[ix])

    # Case 4: Both censored (dt == 0 and dc == 0) -> 2D Stieltjes integral over T_i > x_i, T_j > y_j
    if ct_idx.size > 0 and cc_idx.size > 0 and km_t.atom_times.size > 0 and km_c.atom_times.size > 0:
        st_x = km_t.surv_at(xt[ct_idx])
        sc_y = km_c.surv_at(xc[cc_idx])
        k0_t = np.minimum(km_t.first_atom_after(xt[ct_idx]), km_t.atom_times.size)
        l0_c = np.minimum(km_c.first_atom_after(xc[cc_idx]), km_c.atom_times.size)

        aw, al, an, au = score_pairs_tte_gehan(
            km_t.atom_times, st_atoms, km_c.atom_times, sc_atoms, threshold=tau, operator="higher"
        )
        joint_p = km_t.atom_probs[:, None] * km_c.atom_probs[None, :]

        def _suffix2d(mat: np.ndarray) -> np.ndarray:
            s2 = np.cumsum(np.cumsum(mat[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
            padded = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1), dtype=float)
            padded[:-1, :-1] = s2
            return padded

        q_w = _suffix2d(joint_p * aw)
        q_l = _suffix2d(joint_p * al)
        q_n = _suffix2d(joint_p * an)
        q_u = _suffix2d(joint_p * au)

        joint_surv = st_x[:, None] * sc_y[None, :]
        pos_surv = joint_surv > 1e-15
        denom = np.where(pos_surv, joint_surv, 1.0)

        grid_k, grid_l = np.ix_(k0_t, l0_c)
        ix = np.ix_(ct_idx, cc_idx)
        win[ix] = np.where(pos_surv, q_w[grid_k, grid_l] / denom, 0.0)
        loss[ix] = np.where(pos_surv, q_l[grid_k, grid_l] / denom, 0.0)
        neutral[ix] = np.where(pos_surv, q_n[grid_k, grid_l] / denom, 0.0)
        uninf[ix] = np.where(pos_surv, q_u[grid_k, grid_l] / denom, 1.0)

    # Enforce exact probability simplex bounds against floating-point rounding
    win = np.clip(win, 0.0, 1.0)
    loss = np.clip(loss, 0.0, 1.0)
    neutral = np.clip(neutral, 0.0, 1.0)
    uninf = np.clip(1.0 - (win + loss + neutral), 0.0, 1.0)

    if operator == "lower":
        return loss, win, neutral, uninf
    return win, loss, neutral, uninf
