"""Oncogenetic trees (Desper et al. 1999), following the algorithm of the R package `Oncotree` (Szabo & Boucher)
as used by EvAM-Tools' `ot_proc` (Diaz-Uriarte & Herrera-Nieto, AGPL-3.0). Derivative Python translation.

Each event i picks the parent j (Root included, p_Root = 1) with p_j > p_i that maximises
    w(j -> i) = log( p_ij / ( p_i * (p_i + p_j) ) ),
ties going to the lowest index. Because parents are strictly more frequent, the result is always a tree.
Observed edge weight is P(i | j) = p_ij / p_j. Optional false-positive/negative rates (epos, eneg) are fitted
by L-BFGS-B on the squared error of the marginals, as in `oncotree.fit`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import EvamResult, SettingsMode, check_binary, structural_order, validate_settings


def _parents(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X includes the Root column at index 0. Returns parent index array (-1 for Root) and obs weights."""
    N, m = X.shape
    p2 = X.T @ X / N
    p1 = np.diag(p2).copy()
    parent = np.full(m, -1)
    obs_w = np.ones(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, m):
            best, bj = -np.inf, -1
            for j in range(m):
                if j == i or not p1[j] > p1[i]:
                    continue
                w = np.log(p2[j, i] / (p1[i] * (p1[j] + p1[i])))
                if w > best:
                    best, bj = w, j
            parent[i] = bj
            obs_w[i] = p2[i, bj] / p1[bj]
    return parent, obs_w


def _marginals(parent: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = len(parent)
    out = np.zeros(m)
    out[0] = 1.0
    done = {0}
    while len(done) < m:
        for i in range(1, m):
            if i not in done and parent[i] in done:
                out[i] = out[parent[i]] * w[i]
                done.add(i)
    return out


def _error_weights(
    parent: np.ndarray,
    obs_w: np.ndarray,
    md: np.ndarray,
    epos: float,
    eneg: float,
) -> np.ndarray:
    """Port of `error.rates<-`: back out the error-free edge probabilities."""
    m = len(parent)
    est = np.ones(m)
    for i in range(1, m):
        pj, pi = md[parent[i]], md[i]
        est[i] = (obs_w[i] * pj - epos * (pi + pj) + epos ** 2) / ((pj - epos) * (1 - eneg - epos))
    if (est[1:] < 0).any() or (est[1:] >= 1).any() or not np.isfinite(est).all():
        raise ValueError("edge probability outside [0, 1)")
    return est


def fit_ot(
    X: pd.DataFrame,
    estimate_errors: bool = True,
    *,
    settings: SettingsMode = "default",
) -> EvamResult:
    validate_settings(settings)
    X = check_binary(X)
    events = list(X.columns)
    keep = [c for c in events if X[c].nunique() > 1]  # formatData drops constant events
    if not keep:
        edges = pd.DataFrame(columns=["From", "To", "weight", "weight_err_corrected"])
        order = structural_order(events, {e: [] for e in events}, "and")
        return EvamResult(
            "OT",
            events,
            edges,
            order,
            model=dict(parent={}, eps=dict(epos=0.0, eneg=0.0), marginal_pred={}, dropped_constant=list(events)),
            meta=dict(order_route="structural", error_fit="none", settings=settings),
        )
    A = np.column_stack([np.ones(len(X), int)] + [X[c].to_numpy() for c in keep]).astype(float)
    names = ["Root"] + keep
    parent, obs_w = _parents(A)
    md = A.mean(0)
    pred_clean = _marginals(parent, obs_w)
    eps = (0.0, 0.0)
    est_w = None
    err_fit = "none"
    minp = pred_clean[1:].min()
    if estimate_errors and minp - 0.001 > 0:
        err_fit = "std"

        def mse(e: np.ndarray) -> float:
            try:
                w = _error_weights(parent, obs_w, md, float(e[0]), float(e[1]))
            except ValueError:
                return float(len(names))
            pm = e[0] + (1 - e[0] - e[1]) * _marginals(parent, w)
            return float(((md[1:] - pm[1:]) ** 2).sum())

        r = minimize(mse, [0.0, 0.0], method="L-BFGS-B", bounds=[(0, minp - 0.001), (0, 1 - minp)])
        eps = (float(r.x[0]), float(r.x[1]))
        try:
            est_w = _error_weights(parent, obs_w, md, *eps)
        except ValueError:
            est_w = None
    rows = [
        (names[parent[i]], names[i], obs_w[i], (est_w[i] if est_w is not None else np.nan))
        for i in range(1, len(names))
    ]
    edges = pd.DataFrame(rows, columns=["From", "To", "weight", "weight_err_corrected"])
    pdict = {names[i]: [names[parent[i]]] for i in range(1, len(names))}
    order = structural_order(events, {e: pdict.get(e, []) for e in events}, "and")
    return EvamResult(
        "OT",
        events,
        edges,
        order,
        model=dict(
            parent=pdict,
            eps=dict(epos=eps[0], eneg=eps[1]),
            marginal_pred=dict(zip(names, pred_clean)),
            dropped_constant=[c for c in events if c not in keep],
        ),
        meta=dict(order_route="structural", error_fit=err_fit, settings=settings),
    )
