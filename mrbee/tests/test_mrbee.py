from __future__ import annotations

import numpy as np
from mrbee import fit_mrbee, fit_mvmr_bee


def test_mrbee_recovery_with_outliers():
    rng = np.random.default_rng(41)
    m = 500
    b_true = -0.35
    x_true = np.zeros(m)
    x_true[:60] = rng.normal(0.0, 0.12, 60)
    xse = np.full(m, 0.015)
    yse = np.full(m, 0.018)
    pleio = np.zeros(m)
    pleio[:2] = np.array([0.35, -0.40])

    xb = x_true + rng.normal(0.0, xse)
    yb = b_true * x_true + pleio + rng.normal(0.0, yse)

    res = fit_mrbee(xb, xse, yb, yse, inst_p=1e-3, rho_e=0.0)
    assert abs(res["b"] - b_true) < 0.08
    assert res["n_outliers"] >= 1
    assert res["converged"] is True


def test_mrbee_nonconvergence_flag_large_b_and_sample_overlap():
    rng = np.random.default_rng(141)
    m = 600
    b_true = -12.0  # exceeds legacy [-5, 5] clamp
    x_true = np.zeros(m)
    x_true[:80] = rng.normal(0.0, 0.10, 80)
    xse = np.full(m, 0.012)
    yse = np.full(m, 0.12)

    # Sample overlap rho_e = 0.45
    rho_true = 0.45
    e1 = rng.normal(0.0, 1.0, m)
    e2 = rho_true * e1 + np.sqrt(1.0 - rho_true ** 2) * rng.normal(0.0, 1.0, m)
    xb = x_true + xse * e1
    yb = b_true * x_true + yse * e2

    res = fit_mrbee(xb, xse, yb, yse, inst_p=1e-4, rho_e=None)
    assert abs(res["b"] - b_true) < 0.8
    assert abs(res["rho_e"] - rho_true) < 0.12
    assert res["converged"] is True

    # With max_iter=1, loop cannot verify stationarity across iterations -> converged must be False
    res_nc = fit_mrbee(xb, xse, yb, yse, inst_p=1e-4, max_iter=1, tol=1e-15)
    assert res_nc["converged"] is False


def test_fit_mvmr_bee_with_overlap_and_imrp_outliers():
    rng = np.random.default_rng(209)
    m = 80
    theta_true = np.array([0.65, -0.45])
    bx_true = rng.normal(0.0, 0.08, size=(m, 2))
    sx = np.full((m, 2), 0.025)  # substantial exposure measurement error to test bias correction
    sy = np.full(m, 0.03)

    # Correlated estimation errors across [X1, X2, y]
    rxy = np.array(
        [
            [1.0, 0.35, 0.30],
            [0.35, 1.0, -0.25],
            [0.30, -0.25, 1.0],
        ]
    )
    L = np.linalg.cholesky(rxy)
    z = rng.normal(size=(m, 3)) @ L.T

    pleio = np.zeros(m)
    pleio[:3] = np.array([0.45, -0.50, 0.40])

    bx_obs = bx_true + sx * z[:, :2]
    by_obs = bx_true @ theta_true + pleio + sy * z[:, 2]

    res = fit_mvmr_bee(bx_obs, sx, by_obs, sy, Rxy=rxy, pleio_p_thresh=0.05)
    assert res["converged"] is True
    assert res["n_outliers"] >= 3
    assert np.all( ~res["valid_mask"][:3] )
    assert np.allclose(res["b"], theta_true, atol=0.10)
    assert res["b_se"].shape == (2,) and np.all(res["b_se"] > 0)
    assert res["b_ci"].shape == (2, 2)
