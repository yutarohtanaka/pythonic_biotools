from __future__ import annotations

import numpy as np
import pytest
from indexevent import indexevent, simexllhd, simexprofileCI, simexprofilellhd


def test_indexevent_cwls_hedges_olkin_and_simex():
    rng = np.random.default_rng(19)
    m = 200
    b_true = -0.40
    x_true = rng.normal(0.0, 0.06, m)
    xse = np.full(m, 0.015)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    yb = b_true * x_true + rng.normal(0.0, yse)

    for method in ("CWLS", "Hedges-Olkin", "Simex"):
        res = indexevent(xb, xse, yb, yse, method=method, B=15, seed=7)
        assert abs(res["b"] - b_true) < 0.08
        assert res["b_se"] > 0.0
        assert res["b_ci"][0] < res["b"] < res["b_ci"][1]
        assert res["ybeta_adj"].shape == (m,)


def test_indexevent_allele_flip_coding_invariance():
    rng = np.random.default_rng(109)
    m = 120
    b_true = 0.55
    x_true = rng.uniform(0.03, 0.12, m) * rng.choice([-1.0, 1.0], m)
    xse = np.full(m, 0.012)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    xb[0] = 0.0  # test xb == 0 orientation sign (+1)
    yb = 0.02 + b_true * x_true + rng.normal(0.0, yse)

    flip = np.where(np.arange(m) % 3 == 0, -1.0, 1.0)
    flip[0] = 1.0

    for method in ("CWLS", "Hedges-Olkin", "Simex"):
        r1 = indexevent(xb, xse, yb, yse, method=method, B=10, seed=42)
        r2 = indexevent(xb * flip, xse, yb * flip, yse, method=method, B=10, seed=42)
        np.testing.assert_allclose(r1["b"], r2["b"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["b_se"], r2["b_se"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["b_ci"], r2["b_ci"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["ybeta_adj"] * flip, r2["ybeta_adj"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["yse_adj"], r2["yse_adj"], rtol=1e-12, atol=1e-12)


def test_indexevent_weak_instruments_positive_se_and_simex_profile_ci():
    rng = np.random.default_rng(83)
    m = 60
    # Ultra-weak instruments where var(xb) < mean(xse^2), making denom / ho negative
    xse = np.full(m, 0.10)
    yse = np.full(m, 0.05)
    xb = rng.normal(0.0, 0.04, m)
    yb = rng.normal(0.0, 0.05, m)

    for method in ("CWLS", "Hedges-Olkin"):
        res = indexevent(xb, xse, yb, yse, method=method)
        assert res["b_se"] >= 0.0
        assert res["b_ci"][0] <= res["b_ci"][1]

    # Exact zero ybeta -> b_raw == 0 safe guard
    res_zero = indexevent(np.linspace(0.05, 0.20, 30), np.full(30, 0.01), np.zeros(30), np.full(30, 0.02), method="CWLS")
    assert res_zero["b"] == 0.0 and res_zero["b_se"] >= 0.0

    # Verify simexprofilellhd / simexprofileCI in both lb_raw <= 0 and lb_raw > 0 regimes
    lams = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    gamma_true = 0.08  # log(gamma) = -2.525 < -0.84 (exposes legacy missing-log bug!)
    p_true = -0.45
    means = p_true / (1.0 + (lams + 1.0) * gamma_true)
    sim_est = np.column_stack([lams, means, np.full_like(lams, 1e-4)])
    ci = simexprofileCI(sim_est, variance_ratio=2.5)
    np.testing.assert_allclose(ci[0], p_true, atol=1e-3)
    assert ci[1] < ci[0] < ci[2]
    # Also test lb_raw > 0 regime (large |p| * sqrt(variance_ratio) > 1)
    val_pos_lb = simexprofilellhd(2.0, sim_est, variance_ratio=1.5)
    assert np.isfinite(val_pos_lb)

    # Strict method validation
    for bad in ("", "c", "h", "s", "unknown"):
        with pytest.raises(ValueError):
            indexevent(xb, xse, yb, yse, method=bad)


def test_r_default_settings():
    rng = np.random.default_rng(109)
    m = 120
    b_true = 0.55
    x_true = rng.uniform(0.03, 0.12, m) * rng.choice([-1.0, 1.0], m)
    xse = np.full(m, 0.012)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    yb = 0.025 + b_true * x_true + rng.normal(0.0, yse)

    flip = np.where(np.arange(m) % 3 == 0, -1.0, 1.0)

    # In settings="R_default", indexevent.R runs raw signed WLS without orienting xb >= 0
    r_signed_1 = indexevent(xb, xse, yb, yse, method="CWLS", settings="R_default")
    r_signed_2 = indexevent(xb * flip, xse, yb * flip, yse, method="CWLS", settings="R_default")
    assert not np.isclose(r_signed_1["b"], r_signed_2["b"])

    # Verify exact raw signed CWLS formula against R_default
    w = 1.0 / (yse ** 2)
    sw = float(w.sum())
    numer = float(sw * np.sum(w * xb * yb) - np.sum(w * xb) * np.sum(w * yb))
    denom = float(sw * np.sum(w * xb ** 2) - np.sum(w * xb) ** 2 - sw * np.sum(w * xse ** 2))
    np.testing.assert_allclose(r_signed_1["b"], numer / denom, rtol=1e-13)

    with pytest.raises(ValueError, match="Unknown settings"):
        indexevent(xb, xse, yb, yse, settings="bad")  # type: ignore[arg-type]
