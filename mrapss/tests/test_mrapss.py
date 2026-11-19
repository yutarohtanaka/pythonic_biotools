from __future__ import annotations

import numpy as np
import pytest
from mrapss import estimate_background_ldsc, fit_mrapss


def test_mrapss_null_ldsc_intercept_calibration_without_ldscore():
    rng = np.random.default_rng(1234)
    m = 5000
    zx = rng.normal(0.0, 1.0, m)
    zy = 0.25 * zx + np.sqrt(1.0 - 0.25 ** 2) * rng.normal(0.0, 1.0, m)

    bg = estimate_background_ldsc(zx, zy, ldscore=None)
    cx, cy = bg["C"][0, 0], bg["C"][1, 1]
    assert abs(cx - 1.0) < 0.06
    assert abs(cy - 1.0) < 0.06
    assert abs(bg["rho_e"] - 0.25) < 0.06
    np.testing.assert_allclose(bg["Omega"], np.zeros((2, 2)), atol=1e-12)


def test_mrapss_nonuniform_ldscore_and_polygenic_omega():
    rng = np.random.default_rng(31)
    m = 2000
    l2 = rng.uniform(1.0, 35.0, m)
    # True polygenic background Omega = [[0.03, 0.012], [0.012, 0.04]], C = [[1.05, 0.25], [0.25, 1.02]]
    omega_true = np.array([[0.03, 0.012], [0.012, 0.04]])
    c_true = np.array([[1.05, 0.25], [0.25, 1.02]])

    zx = np.empty(m)
    zy = np.empty(m)
    z_raw = rng.normal(size=(m, 2))
    for j in range(m):
        cov_j = omega_true * l2[j] + c_true
        L = np.linalg.cholesky(cov_j)
        v = L @ z_raw[j]
        zx[j], zy[j] = v[0], v[1]

    bg = estimate_background_ldsc(zx, zy, ldscore=l2)
    assert bg["Omega"][0, 0] > 0.015
    assert bg["Omega"][1, 1] > 0.020
    assert abs(bg["Omega"][0, 1] - 0.012) < 0.015

    # Add foreground causal effect b_true = 0.35 on 80 SNPs
    b_true = 0.35
    xse = np.full(m, 0.02)
    yse = np.full(m, 0.02)
    fg = np.zeros(m)
    fg[:80] = rng.normal(0.0, 0.16, 80) * np.sqrt(l2[:80] / np.mean(l2))
    xb = fg + xse * zx
    yb = b_true * fg + yse * zy

    res = fit_mrapss(xb, xse, yb, yse, ldscore=l2, iv_p_thresh=1e-4)
    assert abs(res["b"] - b_true) < 0.12
    assert res["p_lrt"] < 1e-4
    assert res["pval"] < 1e-4
    assert res["converged"]


def test_mrapss_large_effect_and_iv_threshold_handling():
    rng = np.random.default_rng(79)
    m = 600
    b_true = 10.0  # exceeds legacy [-5, 5] bound
    xse = np.full(m, 0.01)
    yse = np.full(m, 0.10)
    fg = np.zeros(m)
    fg[:50] = rng.normal(0.0, 0.09, 50)
    xb = fg + rng.normal(0.0, xse)
    yb = b_true * fg + rng.normal(0.0, yse)

    res = fit_mrapss(xb, xse, yb, yse, iv_p_thresh=1e-4)
    assert abs(res["b"] - b_true) < 1.2
    assert res["p_lrt"] < 1e-6

    # Pure null data where 0 SNPs pass 5e-8: must warn and raise ValueError unless allow_threshold_relaxation=True
    xb_null = rng.normal(0.0, xse)
    yb_null = rng.normal(0.0, yse)
    with pytest.warns(UserWarning, match="pass iv_p_thresh"):
        with pytest.raises(ValueError, match="0 instruments pass"):
            fit_mrapss(xb_null, xse, yb_null, yse, iv_p_thresh=1e-12, allow_threshold_relaxation=False)

    with pytest.warns(UserWarning, match="pass iv_p_thresh"):
        res_relaxed = fit_mrapss(xb_null, xse, yb_null, yse, iv_p_thresh=1e-12, allow_threshold_relaxation=True)
    assert res_relaxed["n_snps_used"] >= 20
    assert res_relaxed["p_lrt"] > 0.01


def test_mrapss_with_sample_overlap():
    rng = np.random.default_rng(31)
    m = 800
    b_true = 0.30
    xse = np.full(m, 0.02)
    yse = np.full(m, 0.02)
    fg = np.zeros(m)
    fg[:80] = rng.normal(0.0, 0.14, 80)
    e1 = rng.normal(0.0, 1.0, m)
    e2 = 0.3 * e1 + np.sqrt(1.0 - 0.09) * rng.normal(0.0, 1.0, m)

    xb = fg + xse * e1
    yb = b_true * fg + yse * e2

    res = fit_mrapss(xb, xse, yb, yse, iv_p_thresh=1e-3)
    assert abs(res["b"] - b_true) < 0.10
    assert res["p_lrt"] < 1e-4
    assert res["converged"]


def test_r_default_settings():
    rng = np.random.default_rng(909)
    m = 2500
    l2 = rng.uniform(1.0, 30.0, m)
    omega_true = np.array([[0.025, 0.010], [0.010, 0.030]])
    # Intercepts > 1.5 to verify R_default IRWLS does NOT clip to [0.8, 1.5]
    c_true = np.array([[1.85, 0.35], [0.35, 1.75]])

    z_raw = rng.normal(size=(m, 2))
    zx = np.empty(m)
    zy = np.empty(m)
    for j in range(m):
        L = np.linalg.cholesky(omega_true * l2[j] + c_true)
        v = L @ z_raw[j]
        zx[j], zy[j] = v[0], v[1]

    bg_def = estimate_background_ldsc(zx, zy, ldscore=l2, settings="default")
    bg_r = estimate_background_ldsc(zx, zy, ldscore=l2, settings="R_default")
    assert bg_def["C"][0, 0] == 1.5
    assert bg_def["C"][1, 1] == 1.5
    assert bg_r["C"][0, 0] > 1.65
    assert bg_r["C"][1, 1] > 1.58
    assert abs(bg_r["Omega"][0, 1] - 0.010) < 0.015

    b_true = 0.35
    xse = np.full(m, 0.02)
    yse = np.full(m, 0.02)
    fg = np.zeros(m)
    fg[:90] = rng.normal(0.0, 0.16, 90)
    xb = fg + xse * zx
    yb = b_true * fg + yse * zy

    res_r = fit_mrapss(xb, xse, yb, yse, ldscore=l2, iv_p_thresh=1e-4, settings="R_default")
    assert abs(res_r["b"] - b_true) < 0.12
    assert res_r["pval"] == res_r["p_lrt"]
    np.testing.assert_allclose(
        res_r["b_se"],
        abs(res_r["b"]) / np.sqrt(max(res_r["lrt_stat"], 1e-12)),
        rtol=1e-12,
    )

    # Fallback when b == 0 exactly (zero outcome effects with zero overlap)
    yb_zero = np.zeros(m)
    res_zero = fit_mrapss(xb, xse, yb_zero, yse, iv_p_thresh=1e-4, rho_e=0.0, settings="R_default")
    assert abs(res_zero["b"]) < 1e-6
    assert np.isfinite(res_zero["b_se"]) and res_zero["b_se"] > 0.0

    with pytest.raises(ValueError, match="Unknown settings"):
        estimate_background_ldsc(zx, zy, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        fit_mrapss(xb, xse, yb, yse, settings="invalid")  # type: ignore[arg-type]
