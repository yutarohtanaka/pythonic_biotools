from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evam import (
    ctmc_order_exact,
    ctmc_order_mc,
    fit_frequency,
    fit_frequency_fisher,
    fit_mhn,
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


def test_r_default_settings(monkeypatch):
    import sys
    import types

    rng = np.random.default_rng(77)
    n_samp = 150
    a = rng.binomial(1, 0.70, size=n_samp)
    b = np.where(a == 1, rng.binomial(1, 0.75, size=n_samp), rng.binomial(1, 0.04, size=n_samp))
    c = np.where((a == 1) & (b == 1), rng.binomial(1, 0.70, size=n_samp), rng.binomial(1, 0.03, size=n_samp))
    X = pd.DataFrame({"A": a, "B": b, "C": c})

    # 1. Invalid settings raises ValueError across fit_ot, fit_oncobn, fit_mhn
    for fn in (
        lambda: fit_ot(X, settings="unknown"),  # type: ignore[arg-type]
        lambda: fit_oncobn(X, settings="unknown"),  # type: ignore[arg-type]
        lambda: fit_mhn(X, settings="unknown"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="settings"):
            fn()

    # 2. fit_ot R_default
    ot_def = fit_ot(X, settings="default")
    ot_r = fit_ot(X, settings="R_default")
    pd.testing.assert_frame_equal(ot_r.order, ot_def.order)

    # 3. fit_oncobn R_default: defaults to model="CBN", penalty="none", +N loglik offset, no theta > eps rejection
    obn_def = fit_oncobn(X)
    obn_r = fit_oncobn(X, settings="R_default")
    assert obn_def.method == "OncoBN_DBN" and obn_def.model["penalty"] == "bic"
    assert obn_r.method == "OncoBN_CBN" and obn_r.model["penalty"] == "none"

    obn_unpen_def = fit_oncobn(X, model="CBN", penalty="none", settings="default")
    assert obn_r.model["loglik"] == pytest.approx(obn_unpen_def.model["loglik"] + len(X), rel=1e-12)

    # Mutually exclusive pair where theta == 0 < epsilon=0.45:
    # default rejects parent edge (theta <= epsilon), whereas R_default (OncoBN::fitCPN) permits it when ll improves
    m1 = np.r_[np.ones(60, dtype=int), np.zeros(40, dtype=int)]
    m2 = np.r_[np.zeros(60, dtype=int), np.ones(18, dtype=int), np.zeros(22, dtype=int)]
    X_me = pd.DataFrame({"M1": m1, "M2": m2})
    me_def = fit_oncobn(X_me, model="CBN", penalty="none", epsilon=0.45, settings="default")
    me_r = fit_oncobn(X_me, epsilon=0.45, settings="R_default")
    assert me_def.model["parents"]["M2"] == ["Root"]
    assert me_r.model["parents"]["M2"] == ["M1"]

    # 4. fit_mhn R_default (both built-in exact solver and mocked LearnMHN optimizer)
    mhn_def = fit_mhn(X, maxit=200, settings="default")
    mhn_r = fit_mhn(X, maxit=200, settings="R_default")
    assert mhn_def.method == "cMHN"
    assert mhn_r.method == "oMHN"
    theta_r = mhn_r.model["log_theta"].to_numpy()
    np.testing.assert_allclose(theta_r, np.round(theta_r, 2), atol=1e-12)

    # Verify small non-zero off-diagonal edge (0.03) is retained in R_default but filtered (>0.05) in default
    class _MockRes:
        def __init__(self, lt):
            self.log_theta = lt
            self.meta = {"score": -1.23}

        def get_equivalent_classical_mhn(self):
            n_ev = self.log_theta.shape[1]
            eff = self.log_theta[:n_ev, :n_ev] - self.log_theta[n_ev][None, :] * (1.0 - np.eye(n_ev))
            return _MockRes(eff)

    class _MockOpt:
        is_o = False

        def load_data_matrix(self, mat):
            self._n = mat.shape[1]

        def train(self, lam=0.01, maxit=5000, round_result=False):
            raw = np.array(
                [
                    [-0.412, 0.034, 0.0],
                    [1.256, -1.104, 0.0],
                    [0.0, 0.889, -1.523],
                ]
            )
            if self.is_o:
                raw = np.vstack([raw, np.array([0.004, 0.0, 0.0])])
            if round_result:
                raw = np.round(raw, 2)
            return _MockRes(raw)

    class _MockCMHN(_MockOpt):
        is_o = False

    class _MockOMHN(_MockOpt):
        is_o = True

    fake_pkg = types.ModuleType("mhn")
    fake_opt_mod = types.ModuleType("mhn.optimizers")
    fake_opt_mod.cMHNOptimizer = _MockCMHN  # type: ignore[attr-defined]
    fake_opt_mod.oMHNOptimizer = _MockOMHN  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mhn", fake_pkg)
    monkeypatch.setitem(sys.modules, "mhn.optimizers", fake_opt_mod)

    fit_mock_def = fit_mhn(X, settings="default")
    fit_mock_r = fit_mhn(X, settings="R_default")
    assert fit_mock_def.method == "cMHN" and len(fit_mock_def.edges) == 2
    assert fit_mock_r.method == "oMHN" and len(fit_mock_r.edges) == 3  # retains 0.03 edge!
    assert 0.03 in set(np.round(fit_mock_r.edges["weight"].to_numpy(), 2))

