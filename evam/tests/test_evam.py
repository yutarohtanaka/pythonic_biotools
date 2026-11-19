from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evam import (
    ctmc_order_exact,
    ctmc_order_mc,
    fit_frequency,
    fit_frequency_fisher,
    fit_oncobn,
    fit_ot,
    order_from_theta,
    rates_from_theta,
    structural_order,
)
from evam.ot import _error_weights


def test_evam_published_models():
    rng = np.random.default_rng(123)
    n = 180
    a = rng.binomial(1, 0.70, size=n)
    b = np.where(a == 1, rng.binomial(1, 0.75, size=n), rng.binomial(1, 0.04, size=n))
    c = np.where((a == 1) & (b == 1), rng.binomial(1, 0.70, size=n), rng.binomial(1, 0.03, size=n))
    d = rng.binomial(1, 0.25, size=n)
    X = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})

    ot = fit_ot(X)
    assert ot.order.loc["A", "B"] == 1.0 and ot.order.loc["B", "A"] == 0.0

    cbn = fit_oncobn(X, model="CBN", k=2, search="dp")
    assert cbn.order.loc["A", "C"] == 1.0
    # Independent noise event D should hang from Root under BIC regularization
    assert cbn.model["parents"]["D"] == ["Root"]

    freq = fit_frequency(X)
    assert freq.order.loc["A", "B"] > 0.5

    theta = np.array([[0.5, 0.0, 0.0], [1.8, -1.2, 0.0], [0.0, 1.8, -1.8]])
    order_df, route = order_from_theta(theta, ["A", "B", "C"])
    assert route == "ctmc_exact" and order_df.loc["A", "B"] > 0.75


def test_dbn_cbn_noise_non_overfitting_fisher_and_ctmc_mc():
    rng = np.random.default_rng(2026)

    # 1. DBN OR-gate recovery: A and B are independent roots, C occurs iff (A or B), D depends on C
    n = 260
    a = rng.binomial(1, 0.45, size=n)
    b = rng.binomial(1, 0.45, size=n)
    c = np.where((a == 1) | (b == 1), rng.binomial(1, 0.80, size=n), rng.binomial(1, 0.02, size=n))
    d = np.where(c == 1, rng.binomial(1, 0.75, size=n), rng.binomial(1, 0.02, size=n))
    X_dbn = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})

    dbn = fit_oncobn(X_dbn, model="DBN", k=2, epsilon=0.03, search="dp")
    assert set(dbn.model["parents"]["C"]) == {"A", "B"}
    # Under OR semantics, neither A nor B alone dominates C, whereas C dominates D!
    assert dbn.order.loc["A", "C"] == 0.5 and dbn.order.loc["B", "C"] == 0.5
    assert dbn.order.loc["C", "D"] == 1.0 and dbn.order.loc["D", "C"] == 0.0

    # 2. Pure independent Bernoulli noise + mutually exclusive pair: BIC + (theta > epsilon) must NOT overfit edges
    X_noise = pd.DataFrame(
        {
            "N1": rng.binomial(1, 0.50, size=240),
            "N2": rng.binomial(1, 0.35, size=240),
            "N3": rng.binomial(1, 0.25, size=240),
            "N4": rng.binomial(1, 0.15, size=240),
        }
    )
    cbn_noise = fit_oncobn(X_noise, model="CBN", k=2)
    dbn_noise = fit_oncobn(X_noise, model="DBN", k=2)
    assert all(p == ["Root"] for p in cbn_noise.model["parents"].values())
    assert all(p == ["Root"] for p in dbn_noise.model["parents"].values())

    # Strictly mutually exclusive events (theta == 0 <= eps) must never become parent-child
    m1 = np.r_[np.ones(80, dtype=int), np.zeros(120, dtype=int)]
    m2 = np.r_[np.zeros(80, dtype=int), np.ones(60, dtype=int), np.zeros(60, dtype=int)]
    X_me = pd.DataFrame({"M1": m1, "M2": m2})
    me_fit = fit_oncobn(X_me, model="DBN", epsilon=0.05)
    assert me_fit.model["parents"]["M1"] == ["Root"] and me_fit.model["parents"]["M2"] == ["Root"]

    # 3. structural_order dominators diamond DAG & cycle detection
    diamond_parents = {"R1": [], "B1": ["R1"], "B2": ["R1"], "Sink": ["B1", "B2"]}
    ord_or = structural_order(["R1", "B1", "B2", "Sink"], diamond_parents, semantics="or")
    assert ord_or.loc["R1", "Sink"] == 1.0 and ord_or.loc["B1", "Sink"] == 0.5

    with pytest.raises(ValueError, match="cycle"):
        structural_order(["A", "B"], {"A": ["B"], "B": ["A"]}, semantics="or")
    with pytest.raises(ValueError, match="cycle"):
        structural_order(["A"], {"A": ["A"]}, semantics="and")

    # 4. OT _error_weights rejects negative edge probabilities
    with pytest.raises(ValueError, match="outside"):
        _error_weights(
            parent=np.array([-1, 0]),
            obs_w=np.array([1.0, 0.05]),
            md=np.array([1.0, 0.05]),
            epos=0.20,
            eneg=0.05,
        )

    # 5. fit_frequency_fisher calls positively associated ordered pairs
    ff = fit_frequency_fisher(X_dbn, alpha=0.05)
    assert ff.order.loc["C", "D"] == 1.0 and ff.order.loc["D", "C"] == 0.0

    # 6. ctmc_order_mc agrees closely with ctmc_order_exact
    theta = np.array([[0.2, 0.0, 0.0], [1.6, -1.0, 0.0], [0.0, 1.5, -1.5]])
    exact_O = ctmc_order_exact(rates_from_theta(theta))
    off = theta - np.diag(np.diag(theta))
    mc_O = ctmc_order_mc(lambda st: np.exp(np.diag(theta) + off @ st.astype(float)), 3, n_paths=6000, seed=11)
    mask = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(mc_O[mask], exact_O[mask], atol=0.03)
