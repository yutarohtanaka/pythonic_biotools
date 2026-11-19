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


def test_r_default_settings():
    rng = np.random.default_rng(77)
    m = 100
    b_true = -0.45
    x_true = rng.uniform(0.04, 0.14, m) * rng.choice([-1.0, 1.0], m)
    xse = np.full(m, 0.012)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    xb[0] = 0.0  # test xbeta == 0 guard in R_default weighted="2"
    yb = 0.025 + b_true * x_true + rng.normal(0.0, yse)

    flip = np.where(np.arange(m) % 2 == 0, -1.0, 1.0)
    flip[0] = 1.0

    # 1) R_default Egger does NOT orient xb >= 0, so sign flipping changes Egger intercept/slope
    r_egg_1 = method_cb(xb, xse, yb, yse, weighted="1", model="egger", settings="R_default")
    r_egg_2 = method_cb(xb * flip, xse, yb * flip, yse, weighted="1", model="egger", settings="R_default")
    assert not np.isclose(r_egg_1["b"], r_egg_2["b"])

    # 2) R_default yse_adj is exact 2-term sqrt(yse^2 + b^2 * xse^2)
    r_w2 = method_cb(xb, xse, yb, yse, weighted="2", model="ivw", settings="R_default")
    np.testing.assert_allclose(
        r_w2["yse_adj"],
        np.sqrt(yse ** 2 + (r_w2["b"] ** 2) * (xse ** 2)),
        rtol=1e-13,
    )
    r_w2_def = method_cb(xb, xse, yb, yse, weighted="2", model="ivw", settings="default")
    assert np.all(r_w2["yse_adj"] < r_w2_def["yse_adj"])

    # 3) R_default weighted="2" matches exact ratio weights 1 / (yse^2 + (yb / xb_safe)^2 * xse^2)
    xb_safe = np.where(xb == 0.0, 1e-12, xb)
    w_exp = 1.0 / (yse ** 2 + ((yb / xb_safe) ** 2) * (xse ** 2))
    b_raw_exp = np.sum(w_exp * xb * yb) / np.sum(w_exp * xb ** 2)
    ho_exp = np.sum(w_exp * xb ** 2) / (np.sum(w_exp * xb ** 2) - np.sum(w_exp * xse ** 2))
    np.testing.assert_allclose(r_w2["b"], b_raw_exp * ho_exp, rtol=1e-13)

    with pytest.raises(ValueError, match="Unknown settings"):
        method_cb(xb, xse, yb, yse, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        cwls(xb, xse, yb, yse, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        cwbls(xb, xse, xb, xse, yb, yse, settings="bad")  # type: ignore[arg-type]
