from __future__ import annotations

import numpy as np
import pytest
from colliderbias import cwbls, cwls, method_cb


def test_cwls_and_cwbls():
    rng = np.random.default_rng(7)
    m = 250
    x_true = rng.normal(0.0, 0.08, m)
    d_true = 0.5 * x_true + rng.normal(0.0, 0.06, m)
    y_true = 0.30 * x_true - 0.40 * d_true
    se = np.full(m, 0.015)

    xb = x_true + rng.normal(0.0, se)
    db = d_true + rng.normal(0.0, se)
    yb = y_true + rng.normal(0.0, se)

    r_cb = cwls(db, se, -0.40 * d_true + rng.normal(0.0, se), se, model="ivw")
    assert abs(r_cb["b"] - (-0.40)) < 0.06

    r_biv = cwbls(xb, se, db, se, yb, se, n_boot=100, seed=3)
    assert abs(r_biv["b"] - 0.30) < 0.08
    assert abs(r_biv["beta_d"] - (-0.40)) < 0.08
    assert r_biv["b_se"] > 0 and r_biv["beta_d_se"] > 0
    assert r_biv["b_ci"][0] < r_biv["b"] < r_biv["b_ci"][1]
    assert r_biv["beta_d_ci"][0] < r_biv["beta_d"] < r_biv["beta_d_ci"][1]
    assert r_biv["b_pval"] < 1e-4 and r_biv["beta_d_pval"] < 1e-4


def test_method_cb_egger_coding_invariance_weighted2_zero_xb_and_yse_adj():
    rng = np.random.default_rng(91)
    m = 120
    b_true = -0.50
    x_true = rng.uniform(0.03, 0.12, m) * rng.choice([-1.0, 1.0], m)
    xse = np.full(m, 0.012)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    xb[0] = 0.0  # must NOT divide by zero under weighted="2"
    yb = 0.015 + b_true * x_true + rng.normal(0.0, yse)

    flip = np.where(np.arange(m) % 2 == 0, -1.0, 1.0)
    flip[0] = 1.0

    for w_mode in ("0", "1", "2"):
        r1 = method_cb(xb, xse, yb, yse, weighted=w_mode, model="egger")
        r2 = method_cb(xb * flip, xse, yb * flip, yse, weighted=w_mode, model="egger")
        np.testing.assert_allclose(r1["b"], r2["b"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["b_se"], r2["b_se"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["ybeta_adj"] * flip, r2["ybeta_adj"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r1["yse_adj"], r2["yse_adj"], rtol=1e-12, atol=1e-12)

        # Verify yse_adj includes + b_se^2 * xbeta^2 + b_se^2 * xse^2
        b, b_se = r1["b"], r1["b_se"]
        expected_yse_adj = np.sqrt(yse ** 2 + b ** 2 * xse ** 2 + b_se ** 2 * xb ** 2 + b_se ** 2 * xse ** 2)
        np.testing.assert_allclose(r1["yse_adj"], expected_yse_adj, rtol=1e-13)

    # Sample overlap correlation changes yse_adj in expected direction (-2 * b * rho * xse * yse)
    r_ov = method_cb(xb, xse, yb, yse, weighted="2", model="ivw", rho_overlap=-0.4)
    r_no = method_cb(xb, xse, yb, yse, weighted="2", model="ivw", rho_overlap=0.0)
    # Here b < 0 and rho_overlap < 0 so -2 * b * rho < 0 -> smaller yse_adj
    assert np.all(r_ov["yse_adj"] < r_no["yse_adj"])

    with pytest.raises(ValueError):
        method_cb(xb, xse, yb, yse, rho_overlap=1.5)
