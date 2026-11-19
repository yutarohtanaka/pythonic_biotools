"""Common result type and shared 'A before B' ordering scale for Evolutionary Accumulation Models (EvAMs).

Every method returns an `EvamResult`. `order.loc[a, b]` is the model's probability-like score that event `a` is
acquired before event `b` (0.5 = no preference / unordered, diagonal NaN, and order[a,b] + order[b,a] == 1).
It is derived by one of four routes, recorded in `meta["order_route"]`:
  * "ctmc_exact": exact over the 2^n genotype lattice for models with a rate for every (genotype, event) --
    MHN, CBN. P(a before b) = prob. that, on the random path from the empty genotype to the full one, a is added
    while b is still absent.
  * "ctmc_mc": the same by Monte-Carlo path sampling when n is too large for the lattice.
  * "structural": for models that only encode a restriction graph (OT, OncoBN): 1 if a must precede b
    (ancestor under tree/AND, dominator under OR), 0 if b must precede a, else 0.5.
  * "frequency": f_a / (f_a + f_b), the no-structure baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class EvamResult:
    method: str
    events: list[str]
    edges: pd.DataFrame            # columns From, To, weight; From may be "Root"
    order: pd.DataFrame            # P(row before column), events x events
    model: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def pairwise(self) -> pd.DataFrame:
        """Long table of unordered pairs (a, b) with a < b in event order and P(a before b)."""
        rows = [
            (a, b, self.order.loc[a, b])
            for i, a in enumerate(self.events)
            for b in self.events[i + 1 :]
        ]
        return pd.DataFrame(rows, columns=["a", "b", "p_a_before_b"])


def check_binary(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a samples x events DataFrame")
    A = X.to_numpy()
    if not np.isin(A, (0, 1)).all():
        raise ValueError("X must be binary (0/1)")
    return X.astype(int)


def empty_order(events: list[str]) -> pd.DataFrame:
    a = np.full((len(events), len(events)), 0.5)
    np.fill_diagonal(a, np.nan)
    return pd.DataFrame(a, index=events, columns=events)


# ---------------------------------------------------------------- CTMC path ordering
def lattice_bits(n: int) -> np.ndarray:
    s = np.arange(1 << n)
    return ((s[:, None] >> np.arange(n)) & 1).astype(float)


def ctmc_order_exact(R: np.ndarray) -> np.ndarray:
    """R[s, i]: rate of acquiring event i in genotype s (bit i of s; entries with i in s are ignored).
    Returns O[a, b] = P(a is acquired before b) on the embedded jump chain from state 0 to the full state."""
    S, n = R.shape
    bits = lattice_bits(n)
    R = np.where(bits == 0, R, 0.0)
    tot = R.sum(1)
    P = np.divide(R, tot[:, None], out=np.zeros_like(R), where=tot[:, None] > 0)
    v = np.zeros(S)
    v[0] = 1.0
    for s in range(S):  # s | (1<<i) > s, so numeric order is topological
        if v[s] == 0:
            continue
        for i in np.nonzero(P[s])[0]:
            v[s | (1 << i)] += v[s] * P[s, i]
    O = (v[:, None] * P).T @ (1.0 - bits)  # O[a,b] = sum_s v[s] P[s,a] [b not in s]
    np.fill_diagonal(O, np.nan)
    return O


def ctmc_order_mc(rate_fn, n: int, n_paths: int = 4000, seed: int = 0) -> np.ndarray:
    """Monte-Carlo path ordering. rate_fn(state_bool[n]) -> rates[n] (0 for impossible events)."""
    rng = np.random.default_rng(seed)
    before = np.zeros((n, n))
    for _ in range(n_paths):
        st = np.zeros(n, bool)
        pos = np.full(n, n + 1)
        for t in range(n):
            r = np.where(st, 0.0, rate_fn(st))
            if r.sum() <= 0:
                break
            i = rng.choice(n, p=r / r.sum())
            st[i] = True
            pos[i] = t
        before += pos[:, None] < pos[None, :]
    O = before / n_paths
    np.fill_diagonal(O, np.nan)
    return O


# ---------------------------------------------------------------- structural ordering
def structural_order(events: list[str], parents: dict, semantics: str) -> pd.DataFrame:
    """parents[e] = list of parent events (empty/['Root'] -> hangs from Root). semantics 'and' (tree/CBN: all parents
    needed -> a precedes b iff a is an ancestor) or 'or' (a precedes b iff a dominates b: every Root->b path hits a)."""
    if semantics not in ("and", "or"):
        raise ValueError(f"semantics must be 'and' or 'or', got {semantics!r}")
    pa = {e: [p for p in parents.get(e, []) if p != "Root"] for e in events}
    o = empty_order(events)
    anc: dict[str, set[str]] = {}
    dom: dict[str, set[str]] = {}

    def ancestors(e: str, stack: tuple[str, ...] = ()) -> set[str]:
        if e in anc:
            return anc[e]
        if e in stack:
            raise ValueError("cycle in restriction graph")
        out: set[str] = set()
        for p in pa[e]:
            out |= {p} | ancestors(p, stack + (e,))
        anc[e] = out
        return out

    def dominators(e: str, stack: tuple[str, ...] = ()) -> set[str]:
        if e in dom:
            return dom[e]
        if e in stack:
            raise ValueError("cycle in restriction graph")
        if not pa[e]:
            dom[e] = set()
            return dom[e]
        sets = [({p} | dominators(p, stack + (e,))) for p in pa[e]]
        dom[e] = set.intersection(*sets)
        return dom[e]

    for ev in events:
        if semantics == "and":
            ancestors(ev)
        else:
            dominators(ev)

    for a in events:
        for b in events:
            if a == b:
                continue
            a_first = (a in ancestors(b)) if semantics == "and" else (a in dominators(b))
            if a_first:
                o.loc[a, b] = 1.0
                o.loc[b, a] = 0.0
    return o
