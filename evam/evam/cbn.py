"""H-CBN (hidden conjunctive Bayesian networks, Gerstung & Beerenwinkel) through the external `h-cbn` C program,
driven as in EvAM-Tools' `run.cbn`: initial poset from an oncogenetic tree (or linear order), simulated-annealing
poset search (`h-cbn -s -T temp -N steps -w`), then a final `h-cbn -w` re-run on the selected poset to get lambdas.
Requires the `h-cbn` binary via `$EVAM_HCBN` or `PATH`.
Derivative of EvAM-Tools R code (Diaz-Uriarte & Herrera-Nieto, AGPL-3.0); `ct-cbn` C program is GPL.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd

from .base import EvamResult, check_binary, ctmc_order_exact, ctmc_order_mc, lattice_bits
from .ot import fit_ot


def hcbn_binary() -> str:
    b = os.environ.get("EVAM_HCBN") or shutil.which("h-cbn")
    if not b:
        raise FileNotFoundError("h-cbn binary not found (set EVAM_HCBN or put it on PATH)")
    return b


def _write_poset(path: str, n: int, pairs: list[tuple[int, int]]) -> None:
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for a, b in pairs:
            f.write(f"{a} {b}\n")
        f.write("0\n")


def _read_poset(path: str) -> tuple[int, list[tuple[int, int]]]:
    with open(path) as fh:
        toks = fh.read().split()
    n = int(toks[0])
    v = [int(t) for t in toks[1:-1]]
    return n, [(v[i], v[i + 1]) for i in range(0, len(v), 2)]


def fit_cbn(
    X: pd.DataFrame,
    init_poset: str = "OT",
    temp: float = 1.0,
    steps: int | None = None,
    threads: int = 1,
    keep_dir: str | None = None,
) -> EvamResult:
    X = check_binary(X)
    events = list(X.columns)
    n = len(events)
    A = X.to_numpy()
    if steps is None:
        steps = 200 if n <= 4 else 1000 if n == 5 else 10000
    binp = hcbn_binary()
    root = keep_dir or tempfile.mkdtemp(prefix="hcbn_")
    base = os.path.join(root, "m")
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    try:
        with open(base + ".pat", "w") as f:
            f.write(f"{A.shape[0]} {n + 1}\n")
            for row in A:
                f.write("1 " + " ".join(map(str, row)) + "\n")
        with open(base + ".prf", "w") as f:
            f.write(" ".join(["null"] + events) + "\n")
        if init_poset == "OT":
            ot = fit_ot(X, estimate_errors=False).model["parent"]
            idx = {e: i + 1 for i, e in enumerate(events)}
            pairs = sorted((idx[p[0]], idx[c]) for c, p in ot.items() if p[0] != "Root")
        elif init_poset == "linear":
            order = list(np.argsort(-A.mean(0), kind="stable") + 1)
            pairs = [(int(order[i]), int(order[i + 1])) for i in range(n - 1)]
        else:
            raise ValueError(init_poset)
        _write_poset(base + ".poset", n, pairs)

        def run(*a: str):
            return subprocess.run([binp, "-f", base, *a], env=env, check=True, capture_output=True)

        run("-s", "-T", f"{temp:f}", "-N", str(int(steps)), "-w")
        lam_init = np.loadtxt(base + ".lambda")[1:]
        _, pset = _read_poset(os.path.join(base, "00000.poset"))
        shutil.copy(os.path.join(base, "00000.poset"), base + ".poset")
        run("-w")
        lam = np.loadtxt(base + ".lambda")[1:]
        ll = None
        if os.path.exists(base + ".logliknew"):
            try:
                ll = float(np.loadtxt(base + ".logliknew").ravel()[-1])
            except Exception:
                pass
    finally:
        if keep_dir is None:
            shutil.rmtree(root, ignore_errors=True)

    parents: dict[str, list[str]] = {e: [] for e in events}
    for a, b in pset:
        parents[events[b - 1]].append(events[a - 1])
    rows = [(p, c, lam[events.index(c)]) for c in events for p in (parents[c] or ["Root"])]
    edges = pd.DataFrame(rows, columns=["From", "To", "weight"])
    pidx = [[events.index(p) for p in parents[e]] for e in events]
    lamd = dict(zip(events, lam))
    if n <= 14:
        bits = lattice_bits(n)
        R = np.zeros((1 << n, n))
        for i in range(n):
            ok = np.ones(1 << n, bool) if not pidx[i] else bits[:, pidx[i]].all(1).astype(bool)
            R[:, i] = np.where(ok, lam[i], 0.0)
        O = ctmc_order_exact(R)
        route = "ctmc_exact"
    else:
        lam_arr = np.asarray(lam)

        def rate_fn(st: np.ndarray) -> np.ndarray:
            return np.array([lam_arr[i] if all(st[j] for j in pidx[i]) else 0.0 for i in range(n)])

        O = ctmc_order_mc(rate_fn, n)
        route = "ctmc_mc"
    return EvamResult(
        "H-CBN",
        events,
        edges,
        pd.DataFrame(O, index=events, columns=events),
        model=dict(
            parents={e: parents[e] or ["Root"] for e in events},
            lambda_=lamd,
            lambda_init=dict(zip(events, lam_init)),
            loglik=ll,
        ),
        meta=dict(order_route=route, init_poset=init_poset, steps=int(steps), binary=binp),
    )
