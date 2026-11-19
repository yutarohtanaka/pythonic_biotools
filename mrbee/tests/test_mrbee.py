from __future__ import annotations

import numpy as np
import pytest
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


def test_r_default_settings():
    rng = np.random.default_rng(512)
    m = 400
    b_true = -0.40
    x_true = np.zeros(m)
    x_true[:70] = rng.normal(0.0, 0.11, 70)
    x_true[:3] = np.array([0.12, -0.14, 0.13])
    xse = np.full(m, 0.015)
    yse = np.full(m, 0.018)
    pleio = np.zeros(m)
    pleio[:3] = np.array([0.38, -0.42, 0.35])

    xb = x_true + rng.normal(0.0, xse)
    yb = b_true * x_true + pleio + rng.normal(0.0, yse)

    res_def = fit_mrbee(xb, xse, yb, yse, inst_p=1e-3, rho_e=0.0, settings="default")
    res_r = fit_mrbee(xb, xse, yb, yse, inst_p=1e-3, rho_e=0.0, settings="R_default")

    assert res_def["gamma_wc"] > 1.0
    assert res_r["gamma_wc"] == 1.0
    assert res_r["tau2"] == 0.0
    assert res_r["n_outliers"] >= 2
    assert res_r["converged"] is True
    assert abs(res_r["b"] - b_true) < 0.08

    # Multivariable R_default with correlated errors and IMRP empirical-variance Sidak detection
    m_mv = 90
    theta_true = np.array([0.55, -0.35])
    bx_true = rng.normal(0.0, 0.09, size=(m_mv, 2))
    sx = np.full((m_mv, 2), 0.02)
    sy = np.full(m_mv, 0.025)
    rxy = np.array([[1.0, 0.3, 0.25], [0.3, 1.0, -0.2], [0.25, -0.2, 1.0]])
    z = rng.normal(size=(m_mv, 3)) @ np.linalg.cholesky(rxy).T
    pleio_mv = np.zeros(m_mv)
    pleio_mv[:3] = np.array([0.45, -0.50, 0.42])

    bx_obs = bx_true + sx * z[:, :2]
    by_obs = bx_true @ theta_true + pleio_mv + sy * z[:, 2]

    mv_r = fit_mvmr_bee(bx_obs, sx, by_obs, sy, Rxy=rxy, settings="R_default")
    assert mv_r["converged"] is True
    assert mv_r["tau2"] == 0.0
    assert mv_r["n_outliers"] >= 3
    assert np.all(~mv_r["valid_mask"][:3])
    assert np.allclose(mv_r["b"], theta_true, atol=0.10)

    # Exact verification of leverage-adjusted sandwich covariance on outlier-free fit
    mv_no_out = fit_mvmr_bee(bx_obs[3:], sx[3:], by_obs[3:], sy[3:], Rxy=rxy, pleio_p_thresh=0.0, settings="R_default")
    by_std = by_obs[3:] / sy[3:]
    bx_std = bx_obs[3:] / sy[3:, None]
    sxx_std = (sx[3:, :, None] * rxy[:2, :2][None, :, :] * sx[3:, None, :]) / (sy[3:, None, None] ** 2)
    sxy_std = (sy[3:, None] * sx[3:] * rxy[:2, 2][None, :]) / (sy[3:, None] ** 2)
    h_mat = bx_std.T @ bx_std - sxx_std.sum(axis=0)
    h_inv = np.linalg.pinv(h_mat)
    th_exp = np.linalg.solve(h_mat, bx_std.T @ by_std - sxy_std.sum(axis=0))
    h_ii = np.einsum("ip,pq,iq->i", bx_std, h_inv, bx_std)
    lev = np.maximum(1.0 - h_ii, 0.5)
    psi = ((bx_std * by_std[:, None] - sxy_std) - (bx_std * (bx_std @ th_exp)[:, None] - np.einsum("ikl,l->ik", sxx_std, th_exp))) / lev[:, None]
    cov_exp = h_inv @ ((len(by_std) / (len(by_std) - 2)) * (psi.T @ psi)) @ h_inv.T
    np.testing.assert_allclose(mv_no_out["b"], th_exp, rtol=1e-12)
    np.testing.assert_allclose(mv_no_out["cov"], cov_exp, rtol=1e-12)

    with pytest.raises(ValueError, match="Unknown settings"):
        fit_mrbee(xb, xse, yb, yse, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        fit_mvmr_bee(bx_obs, sx, by_obs, sy, settings="bad")  # type: ignore[arg-type]
