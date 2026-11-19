"""Mutual Hazard Networks (MHN; Schill et al. 2020) ordering & optional `mhn` (spang-lab/LearnMHN) wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EvamResult, check_binary, ctmc_order_exact, ctmc_order_mc, lattice_bits


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


def fit_mhn(
    X: pd.DataFrame,
    lam: float | None = None,
    maxit: int = 5000,
    kind: str = "cMHN",
) -> EvamResult:
    """Fit classical (`cMHN`) or observation-bias-corrected (`oMHN`) Mutual Hazard Networks.

    Requires optional dependency `mhn` (`pip install mhn`).
    """
    try:
        from mhn.optimizers import cMHNOptimizer, oMHNOptimizer
    except ImportError as exc:
        raise ImportError(
            "Optional package 'mhn' is required for `fit_mhn`. Install via `pip install 'evam[mhn]'`."
        ) from exc

    if kind not in ("cMHN", "oMHN"):
        raise ValueError(f"kind must be 'cMHN' or 'oMHN', got {kind!r}")
    X = check_binary(X)
    events = list(X.columns)
    N = len(X)
    n = len(events)
    lam = 1.0 / N if lam is None else lam
    opt = (cMHNOptimizer if kind == "cMHN" else oMHNOptimizer)()
    opt.load_data_matrix(X.to_numpy().astype(np.int32))
    res = opt.train(lam=lam, maxit=maxit, round_result=False)
    full = np.asarray(res.log_theta, float)
    obs = full[n] if full.shape[0] > n else None
    if kind == "oMHN" and hasattr(res, "get_equivalent_classical_mhn"):
        theta = np.asarray(res.get_equivalent_classical_mhn().log_theta, float)
    elif obs is not None:
        theta = full[:n, :n] - obs[None, :] * (1.0 - np.eye(n))
    else:
        theta = full[:n, :n]
    rows = [
        (events[j], events[i], theta[i, j])
        for i in range(n)
        for j in range(n)
        if i != j and abs(theta[i, j]) > 0.05
    ]
    edges = pd.DataFrame(rows, columns=["From", "To", "weight"])
    order, route = order_from_theta(theta, events)
    return EvamResult(
        kind,
        events,
        edges,
        order,
        model=dict(
            log_theta=pd.DataFrame(theta, index=events, columns=events),
            observation_rates=None if obs is None else dict(zip(events, obs)),
            lam=lam,
        ),
        meta=dict(
            order_route=route,
            package="mhn",
            score=float(res.meta["score"]) if "score" in res.meta else None,
        ),
    )
