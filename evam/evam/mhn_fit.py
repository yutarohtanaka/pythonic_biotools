"""Mutual Hazard Networks (MHN; Schill et al. 2020; Schill et al. 2024) ordering & optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import (
    EvamResult,
    SettingsMode,
    check_binary,
    ctmc_order_exact,
    ctmc_order_mc,
    lattice_bits,
    validate_settings,
)


def rates_from_theta(theta: np.ndarray) -> np.ndarray:
    """R[s, i] = exp(theta_ii + sum_{j in s} theta_ij) for i not in s (standard MHN rate)."""
    n = theta.shape[0]
    bits = lattice_bits(n)
    logr = np.diag(theta)[None, :] + bits @ (theta - np.diag(np.diag(theta))).T
    return np.exp(logr) * (1 - bits)


def order_from_theta(
    theta: np.ndarray,
    events: list[str],
    exact_max_n: int = 14,
    n_paths: int = 4000,
) -> tuple[pd.DataFrame, str]:
    """Compute pairwise CTMC acquisition ordering probabilities from an MHN `log_theta` matrix."""
    n = len(events)
    if n <= exact_max_n:
        O = ctmc_order_exact(rates_from_theta(theta))
        route = "ctmc_exact"
    else:
        off = theta - np.diag(np.diag(theta))
        O = ctmc_order_mc(lambda st: np.exp(np.diag(theta) + off @ st.astype(float)), n, n_paths)
        route = "ctmc_mc"
    return pd.DataFrame(O, index=events, columns=events), route


def _fit_mhn_builtin_exact(
    X_arr: np.ndarray,
    lam: float,
    maxit: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Exact $2^n$ lattice L1-penalized L-BFGS-B solver for `cMHN` and `oMHN` (`n <= 12`)."""
    N, n = X_arr.shape
    S = 1 << n
    states = X_arr.astype(np.int64) @ (1 << np.arange(n, dtype=np.int64))
    p_emp = np.bincount(states, minlength=S).astype(np.float64) / float(N)
    obs_states = p_emp > 0.0

    bits = lattice_bits(n)
    popcount = bits.sum(axis=1).astype(np.int64)
    layers = [np.flatnonzero(popcount == w) for w in range(n + 1)]
    off_mask = ~np.eye(n, dtype=bool)
    m_off = n * (n - 1)
    is_omhn = kind == "oMHN"

    def _ll_and_eff_grad(eff_th: np.ndarray) -> tuple[float, np.ndarray]:
        off = eff_th - np.diag(np.diag(eff_th))
        logr = np.clip(np.diag(eff_th)[None, :] + bits @ off.T, -30.0, 30.0)
        R = np.exp(logr) * (1.0 - bits)
        E = 1.0 + R.sum(axis=1)

        u = np.zeros(S, dtype=np.float64)
        u[0] = 1.0
        for w in range(n):
            idx_s = layers[w]
            qs = u[idx_s] / E[idx_s]
            for i in range(n):
                mask_i = bits[idx_s, i] == 0.0
                if np.any(mask_i):
                    src = idx_s[mask_i]
                    np.add.at(u, src | (1 << i), qs[mask_i] * R[src, i])

        p = np.maximum(u / E, 1e-300)
        ll = float(np.sum(p_emp[obs_states] * np.log(p[obs_states])))

        ubar = np.zeros(S, dtype=np.float64)
        G = np.zeros((S, n), dtype=np.float64)
        ubar[layers[n]] = p_emp[layers[n]] / np.maximum(u[layers[n]], 1e-300)
        for w in range(n - 1, -1, -1):
            idx_s = layers[w]
            qbar = np.zeros(idx_s.size, dtype=np.float64)
            for i in range(n):
                mask_i = bits[idx_s, i] == 0.0
                if np.any(mask_i):
                    src_local = np.flatnonzero(mask_i)
                    src = idx_s[src_local]
                    qbar[src_local] += R[src, i] * ubar[src | (1 << i)]
            ubar_s = p_emp[idx_s] / np.maximum(u[idx_s], 1e-300) + qbar / E[idx_s]
            ubar[idx_s] = ubar_s
            for i in range(n):
                mask_i = bits[idx_s, i] == 0.0
                if np.any(mask_i):
                    src_local = np.flatnonzero(mask_i)
                    src = idx_s[src_local]
                    G[src, i] = R[src, i] * p[src] * (ubar[src | (1 << i)] - ubar_s[src_local])

        grad_eff = G.T @ bits
        np.fill_diagonal(grad_eff, G.sum(axis=0))
        return ll, grad_eff

    def _obj_and_grad(z: np.ndarray) -> tuple[float, np.ndarray]:
        diag = z[:n]
        u_pos = z[n : n + m_off]
        u_neg = z[n + m_off : n + 2 * m_off]
        theta = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(theta, diag)
        theta[off_mask] = u_pos - u_neg

        if is_omhn:
            w_pos = z[n + 2 * m_off : n + 2 * m_off + n]
            w_neg = z[n + 2 * m_off + n :]
            obs_v = w_pos - w_neg
            eff_th = theta - obs_v[None, :] * (1.0 - np.eye(n))
        else:
            w_pos = w_neg = np.array([], dtype=np.float64)
            eff_th = theta

        ll, g_eff = _ll_and_eff_grad(eff_th)
        pen = lam * (float(np.sum(u_pos + u_neg)) + (float(np.sum(w_pos + w_neg)) if is_omhn else 0.0))
        obj = -ll + pen

        g_diag = -np.diag(g_eff)
        g_off = -g_eff[off_mask]
        grad_u_pos = g_off + lam
        grad_u_neg = -g_off + lam
        if is_omhn:
            # eff_th[i, j] = theta[i, j] - obs[j] for i != j
            g_obs = np.sum(g_eff * (1.0 - np.eye(n)), axis=0)
            grad_w_pos = g_obs + lam
            grad_w_neg = -g_obs + lam
            grad_z = np.concatenate([g_diag, grad_u_pos, grad_u_neg, grad_w_pos, grad_w_neg])
        else:
            grad_z = np.concatenate([g_diag, grad_u_pos, grad_u_neg])
        return obj, grad_z

    dim = n + 2 * m_off + (2 * n if is_omhn else 0)
    z0 = np.zeros(dim, dtype=np.float64)
    freqs = np.clip(X_arr.mean(axis=0), 1e-4, 1.0 - 1e-4)
    z0[:n] = np.log(freqs / (1.0 - freqs))
    bounds = [(None, None)] * n + [(0.0, None)] * (dim - n)

    opt = minimize(
        _obj_and_grad,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxit), "ftol": 1e-12, "gtol": 1e-8},
    )
    zx = opt.x
    theta_hat = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(theta_hat, zx[:n])
    theta_hat[off_mask] = zx[n : n + m_off] - zx[n + m_off : n + 2 * m_off]
    if is_omhn:
        obs_hat: np.ndarray | None = zx[n + 2 * m_off : n + 2 * m_off + n] - zx[n + 2 * m_off + n :]
        eff_hat = theta_hat - obs_hat[None, :] * (1.0 - np.eye(n))
    else:
        obs_hat = None
        eff_hat = theta_hat
    score = float(-opt.fun)
    return eff_hat, obs_hat, score


def fit_mhn(
    X: pd.DataFrame,
    lam: float | None = None,
    maxit: int = 5000,
    kind: str | None = None,
    *,
    round_result: bool | None = None,
    settings: SettingsMode = "default",
) -> EvamResult:
    """Fit classical (`cMHN`) or observation-bias-corrected (`oMHN`) Mutual Hazard Networks.

    - `settings='default'`: `kind='cMHN'`, `round_result=False`, filters minor edges `|theta_ij| <= 0.05`.
    - `settings='R_default'`: Matches `LearnMHN` defaults (`kind='oMHN'`, `round_result=True` to 2 decimals,
      retaining all non-zero off-diagonal edges).
    """
    validate_settings(settings)
    kind_eff = ("oMHN" if settings == "R_default" else "cMHN") if kind is None else kind
    if kind_eff not in ("cMHN", "oMHN"):
        raise ValueError(f"kind must be 'cMHN' or 'oMHN', got {kind_eff!r}")
    round_eff = (settings == "R_default") if round_result is None else bool(round_result)

    X = check_binary(X)
    events = list(X.columns)
    N = len(X)
    n = len(events)
    lam_eff = 1.0 / float(N) if lam is None else float(lam)

    try:
        from mhn.optimizers import cMHNOptimizer, oMHNOptimizer  # type: ignore[import-not-found]

        opt = (cMHNOptimizer if kind_eff == "cMHN" else oMHNOptimizer)()
        opt.load_data_matrix(X.to_numpy().astype(np.int32))
        res = opt.train(lam=lam_eff, maxit=maxit, round_result=round_eff)
        full = np.asarray(res.log_theta, float)
        obs = full[n].copy() if full.shape[0] > n else None
        if kind_eff == "oMHN" and hasattr(res, "get_equivalent_classical_mhn"):
            theta = np.asarray(res.get_equivalent_classical_mhn().log_theta, float)
        elif obs is not None:
            theta = full[:n, :n] - obs[None, :] * (1.0 - np.eye(n))
        else:
            theta = full[:n, :n]
        pkg_name = "mhn"
        score_val = float(res.meta["score"]) if hasattr(res, "meta") and "score" in res.meta else None
    except ImportError as exc:
        if n > 12:
            raise ImportError(
                "Optional package 'mhn' is required for `fit_mhn` when n > 12. Install via `pip install 'evam[mhn]'`."
            ) from exc
        theta, obs, score_val = _fit_mhn_builtin_exact(
            X.to_numpy(dtype=np.int32), lam=lam_eff, maxit=maxit, kind=kind_eff
        )
        pkg_name = "builtin_exact"

    if round_eff:
        theta = np.round(theta, 2) + 0.0
        if obs is not None:
            obs = np.round(obs, 2) + 0.0

    edge_thresh = 0.0 if settings == "R_default" else 0.05
    rows = [
        (events[j], events[i], float(theta[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j and abs(theta[i, j]) > edge_thresh
    ]
    edges = pd.DataFrame(rows, columns=["From", "To", "weight"])
    order, route = order_from_theta(theta, events)
    return EvamResult(
        kind_eff,
        events,
        edges,
        order,
        model=dict(
            log_theta=pd.DataFrame(theta, index=events, columns=events),
            observation_rates=None if obs is None else dict(zip(events, obs)),
            lam=lam_eff,
        ),
        meta=dict(
            order_route=route,
            package=pkg_name,
            score=score_val,
            round_result=round_eff,
            settings=settings,
        ),
    )
