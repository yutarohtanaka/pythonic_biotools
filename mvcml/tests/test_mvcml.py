from __future__ import annotations

import numpy as np
import pytest
from mvcml import adjusted_variances, cml_ma, make_corr, mvcml_adjust, mvcml_bic, mvcml_dp


def test_mvcml_adjust_multivariable_and_bic_ma_weights():
    rng = np.random.default_rng(51)
    s = 80
    bx_true = np.zeros((s, 2))
    bx_true[:25] = rng.normal(0.0, 0.15, (25, 2))
    b_true = np.array([0.35, -0.25])
    sx = np.full((s, 2), 0.012)
    sy = np.full(s, 0.015)
    pleio = np.zeros(s)
    pleio[0] = 0.25

    bx = bx_true + rng.normal(0.0, sx)
    byc = bx_true @ b_true + pleio + rng.normal(0.0, sy)

    res = mvcml_adjust(bx, sx, byc, sy, N=50_000, iv_p_thresh=1e-4)
    assert np.allclose(res["b"], b_true, atol=0.08)
    assert res["ybeta_adj"].shape == (s,)

    fit = res["fit"]
    bic = fit["bic"]
    expected_w = np.exp(-0.5 * (bic - np.min(bic)))
    expected_w /= np.sum(expected_w)
    np.testing.assert_allclose(fit["ma_weights"], expected_w, rtol=1e-13)
    np.testing.assert_allclose(fit["b_ma"], fit["btotal"] @ expected_w, rtol=1e-13)
    np.testing.assert_allclose(fit["b_bic"], fit["btotal"][:, fit["k_invalid"]], rtol=1e-13)


def test_mvcml_null_convergence_and_ld_adjusted_variances():
    rng = np.random.default_rng(152)
    s = 40
    p = 2
    bx_true = np.zeros((s, p))
    bx_true[:18] = rng.normal(0.0, 0.14, (18, p))
    sx = np.full((s, p), 0.012)
    sy = np.full(s, 0.015)

    rho_yx = np.array([0.25, -0.20])
    rho_xx = np.array([[1.0, 0.35], [0.35, 1.0]])
    R = make_corr(p, rho_yx=rho_yx, rho_xx=rho_xx)
    L_r = np.linalg.cholesky(R)

    # Generate correlated estimation noise under null b_true = [0.0, 0.0]
    errs = np.empty((s, p + 1))
    for i in range(s):
        d = np.concatenate([[sy[i]], sx[i]])
        errs[i] = d * (L_r @ rng.normal(size=p + 1))

    byc = errs[:, 0]
    bx = bx_true + errs[:, 1:]

    # Build a valid banded LD matrix across all S SNPs
    idx = np.arange(s)
    ld_full = 0.6 ** np.abs(idx[:, None] - idx[None, :])

    res = mvcml_adjust(
        bx,
        sx,
        byc,
        sy,
        N=40_000,
        iv_p_thresh=1e-4,
        rho_yx=rho_yx,
        rho_xx=rho_xx,
        ld=ld_full,
    )
    assert np.allclose(res["b"], np.zeros(p), atol=0.06)
    assert res["yse_adj_ld"] is not None
    assert res["yse_adj_ld"].shape == (s,)
    assert np.all(np.isfinite(res["yse_adj_ld"])) and np.all(res["yse_adj_ld"] > 0.0)

    # Verify symmetric default (legacy_sigma_gzh=False) differs from legacy_sigma_gzh=True when rho_yx != 0
    iv_idx = res["iv_idx"]
    _, v_ld_sym = adjusted_variances(bx, sx, byc, sy, res["fit"], R, iv_idx, ld=ld_full[:, iv_idx], legacy_sigma_gzh=False)
    _, v_ld_leg = adjusted_variances(bx, sx, byc, sy, res["fit"], R, iv_idx, ld=ld_full[:, iv_idx], legacy_sigma_gzh=True)
    assert not np.allclose(v_ld_sym, v_ld_leg)


def test_mvcml_dp_data_perturbation():
    rng = np.random.default_rng(305)
    m = 20
    p = 2
    b_true = np.array([0.40, -0.30])
    bx_true = rng.normal(0.0, 0.12, (m, p))
    sx = np.full((m, p), 0.015)
    sy = np.full(m, 0.02)
    # Borderline & strong pleiotropic outliers to create selection uncertainty
    pleio = np.zeros(m)
    pleio[:3] = np.array([0.22, -0.09, 0.08])

    R = make_corr(p, rho_yx=[0.15, -0.10], rho_xx=[[1.0, 0.25], [0.25, 1.0]])
    D = np.concatenate([sy[:, None], sx], axis=1)
    SIG = D[:, :, None] * R[None] * D[:, None, :]

    bx = np.empty_like(bx_true)
    byc = np.empty(m)
    for i in range(m):
        e = rng.multivariate_normal(np.zeros(p + 1), SIG[i])
        byc[i] = bx_true[i] @ b_true + pleio[i] + e[0]
        bx[i] = bx_true[i] + e[1:]

    dp_res = mvcml_dp(N=30_000, betax=bx, betayc=byc, SIG=SIG, num_pert=50, seed=42)
    assert np.allclose(dp_res["b"], b_true, atol=0.10)
    assert dp_res["b_se"].shape == (p,) and np.all(dp_res["b_se"] > 0.0)
    assert dp_res["pert_b_ma"].shape == (50, p)
    # DP captures invalid-IV selection uncertainty so SE is >= fixed-k BIC SE
    assert np.all(dp_res["se_bic_dp"] >= 0.85 * dp_res["se_bic"])

    adj_dp = mvcml_adjust(bx, sx, byc, sy, N=30_000, iv_p_thresh=1.0, dp=True, num_pert=30, seed=42)
    assert np.allclose(adj_dp["b"], b_true, atol=0.10)


def test_r_default_settings():
    rng = np.random.default_rng(909)
    m, p = 24, 2
    b_true = np.array([0.45, -0.30])
    bx_true = rng.normal(0.0, 0.14, (m, p))
    sx = np.column_stack([np.full(m, 0.011), np.full(m, 0.023)])
    sy = np.full(m, 0.041)
    pleio = np.zeros(m)
    pleio[:2] = np.array([0.24, -0.20])

    rho_yx = np.array([0.32, -0.24])
    rho_xx = np.array([[1.0, 0.28], [0.28, 1.0]])

    # Build covariance in default [Y, X1, X2] order AND in R_default [X1, X2, Y] order
    R_def = make_corr(p, rho_yx=rho_yx, rho_xx=rho_xx, settings="default")
    R_up = make_corr(p, rho_yx=rho_yx, rho_xx=rho_xx, settings="R_default")
    np.testing.assert_allclose(R_up[:p, :p], rho_xx)
    np.testing.assert_allclose(R_up[:p, p], rho_yx)
    np.testing.assert_allclose(R_up[p, :p], rho_yx)

    D_def = np.concatenate([sy[:, None], sx], axis=1)
    D_up = np.concatenate([sx, sy[:, None]], axis=1)
    SIG_def = D_def[:, :, None] * R_def[None] * D_def[:, None, :]
    SIG_up = D_up[:, :, None] * R_up[None] * D_up[:, None, :]

    bx = np.empty_like(bx_true)
    byc = np.empty(m)
    for i in range(m):
        e = rng.multivariate_normal(np.zeros(p + 1), SIG_def[i])
        byc[i] = bx_true[i] @ b_true + pleio[i] + e[0]
        bx[i] = bx_true[i] + e[1:]

    # 1. Invalid settings raises ValueError across all entrypoints
    with pytest.raises(ValueError, match="Unknown settings"):
        cml_ma(30_000, bx, byc, SIG_def, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        mvcml_bic(30_000, bx, byc, SIG_def, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        mvcml_dp(30_000, bx, byc, SIG_def, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        mvcml_adjust(bx, sx, byc, sy, N=30_000, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        make_corr(p, settings="invalid")  # type: ignore[arg-type]

    # 2. SIG ordering [X1, ..., Xp, Y] in R_default matches [Y, X1, ..., Xp] in default at tight tolerance
    bic_def = mvcml_bic(30_000, bx, byc, SIG_def, threshold=1e-9, settings="default")
    bic_up = mvcml_bic(30_000, bx, byc, SIG_up, threshold=1e-9, settings="R_default")
    np.testing.assert_allclose(bic_up["b"], bic_def["b"], atol=1e-5)
    assert np.allclose(bic_up["b"], b_true, atol=0.10)

    # 3. Single outer iteration (maxit=1) verifies betaxest = b_exp.copy() initialization vs zeros
    one_def = mvcml_bic(30_000, bx, byc, SIG_def, maxit=1, settings="default")
    one_up = mvcml_bic(30_000, bx, byc, SIG_up, maxit=1, settings="R_default")
    assert not np.allclose(one_def["btotal"], one_up["btotal"], atol=1e-9)

    # Default threshold=1e-4 L2 norm in R_default vs threshold=1e-6 L1 norm in default
    fit_up_default_tol = mvcml_bic(30_000, bx, byc, SIG_up, settings="R_default")
    fit_up_explicit_tol = mvcml_bic(30_000, bx, byc, SIG_up, threshold=1e-4, settings="R_default")
    np.testing.assert_allclose(fit_up_default_tol["btotal"], fit_up_explicit_tol["btotal"], rtol=0.0, atol=0.0)

    # 4. mvcml_dp in R_default reports b_bic_dp as primary b, b_se, pval, Covb (vs b_ma_dp in default)
    dp_def = mvcml_dp(30_000, bx, byc, SIG_def, num_pert=30, seed=17, settings="default")
    dp_up = mvcml_dp(30_000, bx, byc, SIG_up, num_pert=30, seed=17, settings="R_default")
    np.testing.assert_allclose(dp_def["b"], dp_def["b_ma_dp"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dp_up["b"], dp_up["b_bic_dp"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dp_up["b_se"], dp_up["se_bic_dp"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dp_up["pval"], dp_up["pval_bic_dp"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dp_up["Covb"], dp_up["Covb_bic_dp"], rtol=0.0, atol=0.0)
    assert np.allclose(dp_up["b"], b_true, atol=0.12)

    # mvcml_adjust end-to-end with settings="R_default" and LD
    idx = np.arange(m)
    ld_mat = 0.5 ** np.abs(idx[:, None] - idx[None, :])
    adj_up = mvcml_adjust(
        bx, sx, byc, sy, N=30_000, iv_p_thresh=1.0, rho_yx=rho_yx, rho_xx=rho_xx, ld=ld_mat, dp=True, num_pert=25, seed=17, settings="R_default"
    )
    assert np.allclose(adj_up["b"], b_true, atol=0.12)
    assert adj_up["yse_adj_ld"] is not None and np.all(adj_up["yse_adj_ld"] > 0.0)
