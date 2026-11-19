"""Frequency-only baselines: no model of interactions, 'earlier' simply means 'more frequent'."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .base import EvamResult, check_binary, empty_order


def fit_frequency(X: pd.DataFrame) -> EvamResult:
    """P(a before b) = f_a / (f_a + f_b)."""
    X = check_binary(X)
    events = list(X.columns)
    f = X.mean().to_numpy()
    denom = f[:, None] + f[None, :]
    a = np.divide(f[:, None], denom, out=np.full_like(denom, 0.5, dtype=float), where=denom > 0)
    np.fill_diagonal(a, np.nan)
    o = pd.DataFrame(a, index=events, columns=events)
    return EvamResult(
        "frequency",
        events,
        pd.DataFrame(columns=["From", "To", "weight"]),
        o,
        model=dict(freq=dict(zip(events, f))),
        meta=dict(order_route="frequency"),
    )


def _bh(p: list[float] | np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    o = np.argsort(p_arr)
    q = p_arr[o] * len(p_arr) / (np.arange(len(p_arr)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def fit_frequency_fisher(X: pd.DataFrame, alpha: float = 0.05) -> EvamResult:
    """Pairwise temporal-priority + positive-association Fisher test (BH-corrected across pairs).

    Score 1/0 for called pairs, 0.5 otherwise.
    """
    X = check_binary(X)
    events = list(X.columns)
    A = X.to_numpy()
    n = len(events)
    pairs, rows = list(itertools.combinations(range(n), 2)), []
    for i, j in pairs:
        a, b = (i, j) if A[:, i].mean() >= A[:, j].mean() else (j, i)
        xa, xb = A[:, a], A[:, b]
        t = [
            [(xa & xb).sum(), (xa & (1 - xb)).sum()],
            [((1 - xa) & xb).sum(), ((1 - xa) & (1 - xb)).sum()],
        ]
        pos = (xa == 1).any() and (xa == 0).any() and xb[xa == 1].mean() > xb[xa == 0].mean()
        rows.append((a, b, fisher_exact(t)[1], pos))
    q = _bh([r[2] for r in rows]) if rows else []
    o = empty_order(events)
    for (a, b, _, pos), qq in zip(rows, q):
        if qq < alpha and pos:
            o.iloc[a, b] = 1.0
            o.iloc[b, a] = 0.0
    return EvamResult(
        "frequency_fisher",
        events,
        pd.DataFrame(columns=["From", "To", "weight"]),
        o,
        meta=dict(order_route="frequency", alpha=alpha),
    )
