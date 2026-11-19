"""OncoBN-style Disjunctive (DBN) / Conjunctive (CBN) Bayesian networks of mutations (Nicol et al. 2021).
Model re-implemented from the paper and checked against the OncoBN R package's likelihood definition; derivative
of the EvAM-Tools `do_OncoBN` interface (Diaz-Uriarte & Herrera-Nieto, AGPL-3.0).

Each event i has a parent set Pa(i) (|Pa| <= k, acyclic; empty means 'hangs from WT'). If the parent condition holds
(OR for DBN, AND for CBN; always true for WT) then x_i ~ Bernoulli(theta_i), otherwise x_i ~ Bernoulli(epsilon).
theta_i is the MLE (fraction of samples with condition true and x_i = 1). Score = log-likelihood; structure search
is exact DP over subsets (search='dp', n <= ~14) or, for larger n, exhaustive parent sets under the order of
decreasing frequency (search='freq_order').
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .base import EvamResult, check_binary, structural_order


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
    model: str = "DBN",
    k: int = 3,
    epsilon: float | None = None,
    search: str | None = None,
) -> EvamResult:
    if model not in ("DBN", "CBN"):
        raise ValueError("model must be DBN or CBN")
    X = check_binary(X)
    events = list(X.columns)
    A = X.to_numpy()
    N, n = A.shape
    eps = float(epsilon if epsilon is not None else max(A.mean(0).min() / 2, 1e-6))
    eps = float(np.clip(eps, 1e-6, 1.0 - 1e-6))
    search = search or ("dp" if n <= 14 else "freq_order")
    agg = (lambda cols: A[:, cols].any(1)) if model == "DBN" else (lambda cols: A[:, cols].all(1))

    cache: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}

    def local(i: int, P: tuple[int, ...]) -> tuple[float, float]:
        key = (i, P)
        if key not in cache:
            cond = np.ones(N, bool) if len(P) == 0 else agg(list(P))
            cache[key] = _local(A[:, i], cond, eps)
        return cache[key]

    parents: dict[int, tuple[int, ...]] = {}
    if search == "dp":
        full = (1 << n) - 1
        best_local: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

        def blocal(i: int, U: int) -> tuple[float, tuple[int, ...]]:
            key = (i, U)
            if key in best_local:
                return best_local[key]
            members = [j for j in range(n) if (U >> j) & 1]
            bs, bp = local(i, ())[0], ()
            for r in range(1, min(k, len(members)) + 1):
                for P in itertools.combinations(members, r):
                    s = local(i, P)[0]
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
        total = score[full][0]
    elif search == "freq_order":
        order_idx = np.argsort(-A.mean(0), kind="stable")
        total = 0.0
        for pos, i in enumerate(order_idx):
            members = [int(x) for x in order_idx[:pos]]
            bs, bp = local(int(i), ())[0], ()
            for r in range(1, min(k, len(members)) + 1):
                for P in itertools.combinations(members, r):
                    s = local(int(i), P)[0]
                    if s > bs + 1e-12:
                        bs, bp = s, P
            parents[int(i)] = bp
            total += bs
    else:
        raise ValueError(f"Unsupported search={search!r}; expected 'dp' or 'freq_order'")

    thetas, rows, pdict = {}, [], {}
    for i in range(n):
        P = tuple(parents[i])
        thetas[events[i]] = local(i, P)[1]
        pdict[events[i]] = [events[j] for j in P] or ["Root"]
        for j in P or ():
            rows.append((events[j], events[i], thetas[events[i]]))
        if not P:
            rows.append(("Root", events[i], thetas[events[i]]))
    edges = pd.DataFrame(rows, columns=["From", "To", "weight"])
    order = structural_order(
        events,
        {e: ([] if p == ["Root"] else p) for e, p in pdict.items()},
        "or" if model == "DBN" else "and",
    )
    return EvamResult(
        f"OncoBN_{model}",
        events,
        edges,
        order,
        model=dict(parents=pdict, theta=thetas, epsilon=eps, loglik=float(total), k=k),
        meta=dict(order_route="structural", search=search),
    )
