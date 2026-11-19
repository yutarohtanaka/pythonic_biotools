from __future__ import annotations

import numpy as np
from mvcml import adjusted_variances, cml_ma, make_corr, mvcml_adjust, mvcml_dp


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
