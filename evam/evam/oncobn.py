"""OncoBN-style Disjunctive (DBN) / Conjunctive (CBN) Bayesian networks of mutations (Nicol et al. 2021).
Model re-implemented from the paper and checked against the OncoBN R package's likelihood definition; derivative
of the EvAM-Tools `do_OncoBN` interface (Diaz-Uriarte & Herrera-Nieto, AGPL-3.0).

Each event i has a parent set Pa(i) (|Pa| <= k, acyclic; empty means 'hangs from WT'). If the parent condition holds
(OR for DBN, AND for CBN; always true for WT) then x_i ~ Bernoulli(theta_i), otherwise x_i ~ Bernoulli(epsilon).
- `settings='default'`: `model='DBN'`, `penalty='bic'`, enforces progression positivity `theta_i > epsilon`.
- `settings='R_default'`: Matches `phillipnicol/OncoBN::fitCPN` defaults (`model='CBN'`, `penalty='none'`
  unpenalized conditional log-likelihood without `theta > epsilon` rejection, adding `+ N` offset to `loglik`).
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .base import EvamResult, SettingsMode, check_binary, structural_order, validate_settings


def _local(x_i: np.ndarray, cond: np.ndarray, eps: float) -> tuple[float, float]:
    """Log-lik contribution of one node and its theta. cond: boolean per sample (parent condition satisfied)."""
    n_c = int(cond.sum())
    c1 = int((x_i[cond] == 1).sum())
    theta = c1 / n_c if n_c > 0 else 0.0
    ll = 0.0
    if n_c > 0:
        if c1 > 0:
            ll += c1 * np.log(theta)
        if n_c - c1 > 0:
            ll += (n_c - c1) * np.log(1 - theta)
    nc = ~cond
    z1 = int((x_i[nc] == 1).sum())
    z0 = int(nc.sum()) - z1
    ll += z1 * np.log(max(eps, 1e-12)) + z0 * np.log(max(1 - eps, 1e-12))
    return float(ll), float(theta)


def fit_oncobn(
    X: pd.DataFrame,
    model: str | None = None,
    k: int = 3,
    epsilon: float | None = None,
    search: str | None = None,
    penalty: str | None = None,
    *,
    settings: SettingsMode = "default",
) -> EvamResult:
    validate_settings(settings)
    model_eff = ("CBN" if settings == "R_default" else "DBN") if model is None else model
    penalty_eff = ("none" if settings == "R_default" else "bic") if penalty is None else penalty

    if model_eff not in ("DBN", "CBN"):
        raise ValueError("model must be DBN or CBN")
    if penalty_eff not in ("bic", "aic", "none"):
        raise ValueError(f"penalty must be 'bic', 'aic', or 'none', got {penalty_eff!r}")

    X = check_binary(X)
    events = list(X.columns)
    A = X.to_numpy()
    N, n = A.shape
    eps = float(epsilon if epsilon is not None else max(A.mean(0).min() / 2, 1e-6))
    eps = float(np.clip(eps, 1e-6, 1.0 - 1e-6))
    search = search or ("dp" if n <= 14 else "freq_order")
    agg = (lambda cols: A[:, cols].any(1)) if model_eff == "DBN" else (lambda cols: A[:, cols].all(1))

    if penalty_eff == "bic":
        edge_pen = 0.5 * float(np.log(max(N, 1)))
    elif penalty_eff == "aic":
        edge_pen = 1.0
    else:
        edge_pen = 0.0

    enforce_theta_gt_eps = settings != "R_default"
    raw_cache: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}

    def local_raw(i: int, P: tuple[int, ...]) -> tuple[float, float]:
        key = (i, P)
        if key not in raw_cache:
            cond = np.ones(N, bool) if len(P) == 0 else agg(list(P))
            raw_cache[key] = _local(A[:, i], cond, eps)
        return raw_cache[key]

    def local_scored(i: int, P: tuple[int, ...]) -> float:
        ll, theta = local_raw(i, P)
        if len(P) > 0:
            if enforce_theta_gt_eps and not (theta > eps):
                return float("-inf")
            ll -= edge_pen * len(P)
        return ll

    parents: dict[int, tuple[int, ...]] = {}
    if search == "dp":
        full = (1 << n) - 1
        best_local: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

        def blocal(i: int, U: int) -> tuple[float, tuple[int, ...]]:
            key = (i, U)
            if key in best_local:
                return best_local[key]
            members = [j for j in range(n) if (U >> j) & 1]
            bs, bp = local_scored(i, ()), ()
            for r in range(1, min(k, len(members)) + 1):
                for P in itertools.combinations(members, r):
                    s = local_scored(i, P)
                    if s > bs + 1e-12:
                        bs, bp = s, P
            best_local[key] = (bs, bp)
            return best_local[key]

        score: dict[int, tuple[float, int | None, tuple[int, ...] | None]] = {0: (0.0, None, None)}
        for S in range(1, 1 << n):
            top: tuple[float, int, tuple[int, ...]] | None = None
            for v in range(n):
                if not (S >> v) & 1:
                    continue
                rest = S & ~(1 << v)
                s, P = blocal(v, rest)
                tot = score[rest][0] + s
                if top is None or tot > top[0] + 1e-12:
                    top = (tot, v, P)
            assert top is not None
            score[S] = top
        S = full
        while S:
            _, v, P = score[S]
            assert v is not None and P is not None
            parents[v] = P
            S &= ~(1 << v)
        total_score = score[full][0]
    elif search == "freq_order":
        order_idx = np.argsort(-A.mean(0), kind="stable")
        total_score = 0.0
        for pos, i in enumerate(order_idx):
            members = [int(x) for x in order_idx[:pos]]
            bs, bp = local_scored(int(i), ()), ()
            for r in range(1, min(k, len(members)) + 1):
                for P in itertools.combinations(members, r):
                    s = local_scored(int(i), P)
                    if s > bs + 1e-12:
                        bs, bp = s, P
            parents[int(i)] = bp
            total_score += bs
    else:
        raise ValueError(f"Unsupported search={search!r}; expected 'dp' or 'freq_order'")

    thetas, rows, pdict = {}, [], {}
    raw_loglik = 0.0
    for i in range(n):
        P = tuple(parents[i])
        ll_i, th_i = local_raw(i, P)
        raw_loglik += ll_i
        thetas[events[i]] = th_i
        pdict[events[i]] = [events[j] for j in P] or ["Root"]
        for j in P or ():
            rows.append((events[j], events[i], th_i))
        if not P:
            rows.append(("Root", events[i], th_i))

    ll_offset = float(N) if settings == "R_default" else 0.0
    edges = pd.DataFrame(rows, columns=["From", "To", "weight"])
    order = structural_order(
        events,
        {e: ([] if p == ["Root"] else p) for e, p in pdict.items()},
        "or" if model_eff == "DBN" else "and",
    )
    return EvamResult(
        f"OncoBN_{model_eff}",
        events,
        edges,
        order,
        model=dict(
            parents=pdict,
            theta=thetas,
            epsilon=eps,
            loglik=float(raw_loglik + ll_offset),
            score=float(total_score + ll_offset),
            penalty=penalty_eff,
            k=k,
        ),
        meta=dict(order_route="structural", search=search, penalty=penalty_eff, settings=settings),
    )
