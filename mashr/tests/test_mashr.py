"""Comprehensive deterministic test suite for `mashr`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal

from mashr import (
    ExtremeDeconvolutionResult,
    MashData,
    MashResult,
    autoselect_grid,
    compute_loglik_matrix,
    compute_posterior_matrices,
    cov_canonical,
    cov_ed,
    cov_pca,
    estimate_null_correlation_simple,
    fit_extreme_deconvolution,
    get_estimated_pi,
    get_lfsr,
    get_n_significant_conditions,
    get_pairwise_sharing,
    get_pm,
    get_psd,
    get_significant_results,
    mash,
    mash_set_data,
    mash_update_data,
    optimize_pi,
)


def _make_corr_matrix(R: int, rho: float) -> np.ndarray:
    mat = np.full((R, R), rho, dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    return mat


def test_canonical_covariances_and_normalization():
    R = 5
    U_can = cov_canonical(R)
    expected_keys = {
        "identity",
        "singleton_cond_0",
        "singleton_cond_1",
        "singleton_cond_2",
        "singleton_cond_3",
        "singleton_cond_4",
        "equal_effects",
        "simple_het_0.25",
        "simple_het_0.5",
        "simple_het_0.75",
    }
    assert set(U_can.keys()) == expected_keys
    for name, mat in U_can.items():
        assert mat.shape == (R, R), name
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
        evals = np.linalg.eigvalsh(mat)
        assert np.min(evals) >= -1e-12, f"{name} min eig {np.min(evals)}"
        assert np.isclose(np.max(np.diag(mat)), 1.0, atol=1e-12), name

    # R = 1 returns single identity matrix
    u_r1 = cov_canonical(1)
    assert list(u_r1.keys()) == ["identity"]
    np.testing.assert_allclose(u_r1["identity"], np.array([[1.0]]))


def test_cov_pca_and_cov_ed_rank1_and_planted_recovery():
    rng = np.random.default_rng(101)
    N, R = 600, 4

    # Plant two distinct full-rank covariances with heteroscedastic noise and correlated V
    A1 = np.array(
        [
            [2.0, 0.8, 0.0, 0.0],
            [0.8, 1.5, 0.2, 0.0],
            [0.0, 0.2, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.4],
        ]
    )
    A2 = np.array(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.1, 0.0],
            [0.0, 0.1, 2.2, -1.0],
            [0.0, 0.0, -1.0, 1.8],
        ]
    )
    U1_true = A1 @ A1.T
    U2_true = A2 @ A2.T

    V_true = np.eye(R)
    V_true[0, 1] = V_true[1, 0] = 0.4
    V_true[2, 3] = V_true[3, 2] = -0.3

    z_comp = rng.choice([0, 1], size=N, p=[0.6, 0.4])
    beta = np.zeros((N, R))
    n1 = int(np.sum(z_comp == 0))
    n2 = N - n1
    beta[z_comp == 0] = rng.multivariate_normal(np.zeros(R), U1_true, size=n1)
    beta[z_comp == 1] = rng.multivariate_normal(np.zeros(R), U2_true, size=n2)

    Shat = rng.uniform(0.3, 0.8, size=(N, R))
    noise = np.empty((N, R))
    for j in range(N):
        Vj = (Shat[j, :, None] * V_true) * Shat[j, None, :]
        noise[j] = rng.multivariate_normal(np.zeros(R), Vj)
    Bhat = beta + noise

    data = mash_set_data(Bhat, Shat, V=V_true)

    U1_init = 0.5 * U1_true + 1.5 * np.eye(R)
    U2_init = 0.5 * U2_true + 1.5 * np.eye(R)
    ed_res = fit_extreme_deconvolution(
        data,
        {"comp_1": U1_init, "comp_2": U2_init},
        subset=np.arange(N),
        max_iter=300,
        tol=1e-5,
        batch_size=128,
    )
    assert isinstance(ed_res, ExtremeDeconvolutionResult)
    assert ed_res.converged is True
    assert ed_res.n_strong == N
    # Monotonic EM log-likelihood
    assert np.all(np.diff(ed_res.log_likelihood_history) >= -1e-9)

    # Match recovered components up to label permutation
    U1_est = ed_res.U_list["comp_1"]
    U2_est = ed_res.U_list["comp_2"]
    err_direct = np.linalg.norm(U1_est - U1_true, "fro") + np.linalg.norm(
        U2_est - U2_true, "fro"
    )
    err_swap = np.linalg.norm(U1_est - U2_true, "fro") + np.linalg.norm(
        U2_est - U1_true, "fro"
    )
    err_init = np.linalg.norm(U1_init - U1_true, "fro") + np.linalg.norm(
        U2_init - U2_true, "fro"
    )
    assert min(err_direct, err_swap) < 0.5 * err_init
    if err_direct <= err_swap:
        assert abs(ed_res.weights[0] - 0.6) < 0.10
        assert abs(ed_res.weights[1] - 0.4) < 0.10
    else:
        assert abs(ed_res.weights[1] - 0.6) < 0.10
        assert abs(ed_res.weights[0] - 0.4) < 0.10

    df_ed = ed_res.to_dataframe()
    assert list(df_ed.columns) == ["component", "weight", "trace", "max_diag", "rank"]
    assert len(df_ed) == 2

    # Rank-1 PCA + rank-2 tPCA initialized and refined with cov_ed
    R2 = 5
    v_a = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    v_b = np.array([0.0, 0.0, 0.5, 1.0, -1.0])
    v_a /= np.linalg.norm(v_a)
    v_b /= np.linalg.norm(v_b)
    strong_b = np.outer(rng.normal(0, 4.0, 120), v_a) + np.outer(
        rng.normal(0, 2.5, 120), v_b
    )
    obs_strong = strong_b + rng.normal(0, 0.5, size=(120, R2))
    data_pca = mash_set_data(obs_strong, np.full((120, R2), 0.5))
    U_pca = cov_pca(data_pca, npc=2, thresh=1.5)
    assert set(U_pca.keys()) == {"PCA_1", "PCA_2", "tPCA"}
    assert np.linalg.matrix_rank(U_pca["PCA_1"], tol=1e-10) == 1

    ed_pca = fit_extreme_deconvolution(data_pca, U_pca, thresh=1.5, max_iter=150)
    assert np.all(np.diff(ed_pca.log_likelihood_history) >= -1e-9)
    for name, mat in ed_pca.U_list.items():
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(mat)) >= -1e-12, name

    # Principal subspace alignment of tPCA-refined matrix with span{v_a, v_b}
    u_tpca_ed = cov_ed(data_pca, U_pca, thresh=1.5, max_iter=150)["tPCA"]
    evals, evecs = np.linalg.eigh(u_tpca_ed)
    top2 = evecs[:, -2:]
    true_basis = np.column_stack([v_a, v_b])
    # Subspace projection cosine sum should be close to 2.0
    subspace_overlap = float(np.sum((top2.T @ true_basis) ** 2))
    assert subspace_overlap > 1.85


def test_mash_planted_shrinkage_sharing_and_heterogeneous():
    rng = np.random.default_rng(202)
    J, R = 1500, 4

    # 60% null, 20% equal_effects, 10% singleton in cond_0, 10% simple_het (rho=0.5)
    beta_true = np.zeros((J, R))
    n_null = 900
    n_eq = 300
    n_sing = 150
    n_het = 150

    eq_vals = rng.normal(0.0, 2.5, size=n_eq)
    beta_true[n_null : n_null + n_eq, :] = eq_vals[:, None]

    beta_true[n_null + n_eq : n_null + n_eq + n_sing, 0] = rng.normal(0.0, 3.0, size=n_sing)

    cov_het = _make_corr_matrix(R, 0.5) * 6.0
    beta_true[n_null + n_eq + n_sing :, :] = rng.multivariate_normal(
        np.zeros(R), cov_het, size=n_het
    )

    Shat = rng.uniform(0.4, 0.7, size=(J, R))
    Bhat = beta_true + rng.normal(0.0, Shat)

    data = mash_set_data(Bhat, Shat)
    U_can = cov_canonical(data)
    res = mash(data, U_can)

    assert res.converged is True
    mse_raw = float(np.mean((Bhat - beta_true) ** 2))
    mse_mash = float(np.mean((res.posterior_mean - beta_true) ** 2))
    assert mse_mash < 0.5 * mse_raw

    est_pi = get_estimated_pi(res)
    assert abs(est_pi["null"] - 0.60) < 0.12
    assert est_pi["equal_effects"] > 0.10
    assert est_pi["singleton_cond_0"] > 0.04

    # Check low false discovery proportion on null rows
    sig_idx = get_significant_results(res, thresh=0.05)
    assert sig_idx.size > 150
    fdp = float(np.mean(sig_idx < n_null))
    assert fdp < 0.05

    # Pairwise sharing on equal-effects vs singleton subsets
    eq_data = mash_set_data(Bhat[n_null : n_null + n_eq], Shat[n_null : n_null + n_eq])
    eq_res = mash(eq_data, U_can, grid=res.grid, pi_init=res.pi, fix_g=True)
    share_eq = get_pairwise_sharing(eq_res, factor=0.5, lfsr_thresh=0.05)
    off_diag_eq = share_eq[~np.eye(R, dtype=bool)]
    assert np.all(off_diag_eq >= 0.90)

    sing_data = mash_set_data(
        Bhat[n_null + n_eq : n_null + n_eq + n_sing],
        Shat[n_null + n_eq : n_null + n_eq + n_sing],
    )
    sing_res = mash(sing_data, U_can, grid=res.grid, pi_init=res.pi, fix_g=True)
    share_sing = get_pairwise_sharing(sing_res, factor=0.5, lfsr_thresh=0.05)
    # Condition 0 vs others should have low sharing
    assert np.all(share_sing[0, 1:] <= 0.25)


def test_estimate_null_correlation_simple_recovery():
    rng = np.random.default_rng(303)
    J, R = 2500, 4
    V_true = np.eye(R)
    V_true[0, 1] = V_true[1, 0] = 0.65
    V_true[2, 3] = V_true[3, 2] = -0.45

    beta = np.zeros((J, R))
    beta[:500] = rng.normal(0.0, 5.0, size=(500, R))
    Shat = rng.uniform(0.5, 1.0, size=(J, R))
    noise = rng.multivariate_normal(np.zeros(R), V_true, size=J) * Shat
    Bhat = beta + noise

    data_id = mash_set_data(Bhat, Shat)
    # With z_thresh=2.0, box truncation |Z_r| < 2 attenuates correlations slightly;
    # z_thresh=3.0 retains 99.7% of null mass while excluding |Z|>4 signals.
    V_est_2 = estimate_null_correlation_simple(data_id, z_thresh=2.0)
    V_est = estimate_null_correlation_simple(data_id, z_thresh=3.0)

    assert V_est.shape == (R, R)
    np.testing.assert_allclose(V_est, V_est.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(V_est), np.ones(R), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(V_est)) > 0.0
    assert np.max(np.abs(V_est_2 - V_true)) < 0.15
    assert np.max(np.abs(V_est - V_true)) < 0.06

    data_v = mash_update_data(data_id, V=V_est)
    U_can = cov_canonical(data_v)
    res_id = mash(data_id, U_can)
    res_v = mash(data_v, U_can)
    assert res_v.loglik > res_id.loglik

    # Small-J0 singular covariance fallback still yields strictly PD correlation matrix
    small_data = mash_set_data(Bhat[-3:], Shat[-3:])
    V_ridge = estimate_null_correlation_simple(small_data, z_thresh=5.0, ridge=1e-4)
    assert np.min(np.linalg.eigvalsh(V_ridge)) >= 1e-4 - 1e-10
    np.testing.assert_allclose(np.diag(V_ridge), np.ones(R), atol=1e-12)


def test_univariate_r1_exact_ashr_equivalence():
    rng = np.random.default_rng(404)
    J = 50
    Bhat = rng.normal(0.0, 1.8, size=(J, 1))
    Shat = rng.uniform(0.3, 1.5, size=(J, 1))
    grid = np.array([0.5, 2.0])
    pi = np.array([0.5, 0.3, 0.2])

    data = mash_set_data(Bhat, Shat)
    U_list = {"identity": np.ones((1, 1))}
    res = mash(data, U_list, grid=grid, pi_init=pi, fix_g=True)

    # Closed-form 1D Gaussian conjugate equations
    b = Bhat[:, 0]
    s2 = Shat[:, 0] ** 2
    vars_m = np.column_stack([s2, grid[0] + s2, grid[1] + s2])  # (J, 3)
    ref_ll = -0.5 * (np.log(2.0 * np.pi * vars_m) + (b[:, None] ** 2) / vars_m)

    log_unnorm = ref_ll + np.log(pi)[None, :]
    ref_w = np.exp(log_unnorm - logsumexp(log_unnorm, axis=1, keepdims=True))

    mu_m = np.zeros((J, 3))
    var_m = np.zeros((J, 3))
    for m, om in enumerate([0.0, grid[0], grid[1]]):
        if om > 0:
            mu_m[:, m] = (om / (om + s2)) * b
            var_m[:, m] = (om * s2) / (om + s2)

    ref_pm = np.sum(ref_w * mu_m, axis=1, keepdims=True)
    ref_second = np.sum(ref_w * (var_m + mu_m ** 2), axis=1, keepdims=True)
    ref_psd = np.sqrt(np.maximum(0.0, ref_second - ref_pm ** 2))

    # Exact point mass at 0 for m=0
    p_pos = np.sum(ref_w[:, 1:] * ndtr(mu_m[:, 1:] / np.sqrt(var_m[:, 1:])), axis=1)
    p_neg = np.sum(ref_w[:, 1:] * ndtr(-mu_m[:, 1:] / np.sqrt(var_m[:, 1:])), axis=1)
    ref_lfsr = (1.0 - np.maximum(p_pos, p_neg))[:, None]

    np.testing.assert_allclose(res.loglik_matrix, ref_ll, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.posterior_weights, ref_w, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.posterior_mean, ref_pm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.posterior_sd, ref_psd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.lfsr, ref_lfsr, rtol=1e-12, atol=1e-12)


def test_analytical_equal_effects_and_singleton_pointmass():
    rng = np.random.default_rng(505)
    J, R = 20, 4
    s0 = 0.5
    omega = 2.0
    Bhat = rng.normal(0.0, 2.0, size=(J, R))
    Shat = np.full((J, R), s0)
    data = mash_set_data(Bhat, Shat)

    # (a) Equal effects closed form: mu = (R*omega / (s0^2 + R*omega)) * mean(b_j)
    res_eq = mash(
        data,
        {"equal_effects": np.ones((R, R))},
        grid=np.array([omega]),
        pi_init=np.array([0.0, 1.0]),
        fix_g=True,
    )
    shrink = (R * omega) / (s0 ** 2 + R * omega)
    expected_pm = np.repeat((shrink * np.mean(Bhat, axis=1))[:, None], R, axis=1)
    expected_sd = np.full((J, R), np.sqrt((omega * (s0 ** 2)) / (s0 ** 2 + R * omega)))
    np.testing.assert_allclose(res_eq.posterior_mean, expected_pm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_eq.posterior_sd, expected_sd, rtol=1e-12, atol=1e-12)

    # (b) Singleton U = e_0 e_0^T even with correlated V: conditions 1..3 are exact point mass 0
    V_corr = _make_corr_matrix(R, 0.5)
    B_sing = np.column_stack([np.full(J, 4.0), rng.normal(0.0, 1.0, size=(J, R - 1))])
    B_sing[0, 0] = 0.0  # exact tie at zero
    data_corr = mash_set_data(B_sing, np.full((J, R), 0.5), V=V_corr)
    u_sing0 = np.zeros((R, R))
    u_sing0[0, 0] = 1.0
    res_sing = mash(
        data_corr,
        {"singleton_0": u_sing0},
        grid=np.array([4.0]),
        pi_init=np.array([0.0, 1.0]),
        fix_g=True,
    )
    np.testing.assert_allclose(res_sing.posterior_mean[:, 1:], 0.0, atol=1e-15)
    np.testing.assert_allclose(res_sing.posterior_sd[:, 1:], 0.0, atol=1e-15)
    np.testing.assert_allclose(res_sing.zero_prob[:, 1:], 1.0, atol=1e-15)
    np.testing.assert_allclose(res_sing.lfsr[:, 1:], 1.0, atol=1e-15)
    assert np.all(res_sing.lfsr[1:, 0] < 0.05)

    # (c) Pure null => exact lfsr == 1.0 (not 0.5) and exact tie at 0 => (1 + w_0) / 2
    res_null = mash(
        data_corr,
        {"identity": np.eye(R)},
        grid=np.array([1.0]),
        pi_init=np.array([1.0, 0.0]),
        fix_g=True,
    )
    np.testing.assert_allclose(res_null.lfsr, 1.0, atol=1e-15)
    np.testing.assert_allclose(res_null.zero_prob, 1.0, atol=1e-15)

    # At b = (0, 0), R = 2, N(0; 0, I_2) / N(0; 0, 2 I_2) = 2 => w_0 = 0.4 / (0.4 + 0.6 * 0.5) = 4/7
    B_zero = np.zeros((3, 2))
    res_tie = mash(
        mash_set_data(B_zero, 1.0),
        {"identity": np.eye(2)},
        grid=np.array([1.0]),
        pi_init=np.array([0.4, 0.6]),
        fix_g=True,
    )
    w_zero_post = 0.4 / (0.4 + 0.6 * 0.5)
    np.testing.assert_allclose(res_tie.lfsr, (1.0 + w_zero_post) / 2.0, atol=1e-12)


def test_ee_vs_ez_alpha_invariance_and_scale():
    rng = np.random.default_rng(606)
    J, R = 120, 3
    s0 = 2.5
    Bhat = rng.normal(0.0, 3.0, size=(J, R))
    Shat = np.full((J, R), s0)
    V = _make_corr_matrix(R, 0.3)

    data_ee = mash_set_data(Bhat, Shat, alpha=0.0, V=V)
    data_ez = mash_update_data(data_ee, alpha=1.0)
    U_can = cov_canonical(data_ee)

    grid_ez = np.array([0.25, 1.0, 4.0])
    grid_ee = (s0 ** 2) * grid_ez

    res_ee = mash(data_ee, U_can, grid=grid_ee, tol=1e-9)
    res_ez = mash(data_ez, U_can, grid=grid_ez, tol=1e-9)

    np.testing.assert_allclose(res_ee.pi, res_ez.pi, atol=1e-7)
    np.testing.assert_allclose(res_ee.posterior_weights, res_ez.posterior_weights, atol=1e-7)
    np.testing.assert_allclose(res_ee.posterior_mean, res_ez.posterior_mean, atol=1e-7)
    np.testing.assert_allclose(res_ee.posterior_sd, res_ez.posterior_sd, atol=1e-7)
    np.testing.assert_allclose(res_ee.lfsr, res_ez.lfsr, atol=1e-7)
    assert np.isclose(
        res_ee.loglik, res_ez.loglik - J * R * np.log(s0), rtol=1e-10, atol=1e-10
    )

    # Heterogeneous Shat with alpha=1.0 returns posterior_mean on Bhat scale (not Z scale)
    S_het = rng.uniform(5.0, 10.0, size=(J, R))
    B_het = rng.normal(0.0, 20.0, size=(J, R))
    res_ez_het = mash(mash_set_data(B_het, S_het, alpha=1.0), U_can)
    assert np.max(np.abs(res_ez_het.posterior_mean)) > np.max(np.abs(B_het / S_het))


def test_global_null_calibration_and_collapsed_scale_grid():
    rng = np.random.default_rng(707)
    J, R = 1500, 4
    V = _make_corr_matrix(R, 0.4)
    Shat = rng.uniform(0.5, 1.2, size=(J, R))
    Bhat = rng.multivariate_normal(np.zeros(R), V, size=J) * Shat

    data = mash_set_data(Bhat, Shat, V=V)
    res = mash(data, cov_canonical(data), prior="nullbiased")

    est_pi = get_estimated_pi(res)
    assert est_pi["null"] > 0.85
    sig_frac = float(np.mean(np.min(res.lfsr, axis=1) < 0.05))
    assert sig_frac <= 0.05

    # Weak/zero-signal collapsed delta_max == 0 never collapses grid
    data_small = mash_set_data(np.full((10, 3), 0.1), np.full((10, 3), 1.0))
    grid_small = autoselect_grid(data_small)
    assert grid_small.ndim == 1 and grid_small.size >= 2
    assert np.all(np.isfinite(grid_small)) and np.all(grid_small > 0.0)
    assert np.all(np.diff(grid_small) > 0.0)

    data_zero = mash_set_data(np.zeros((10, 3)), np.ones((10, 3)))
    grid_zero = autoselect_grid(data_zero)
    assert grid_zero.size >= 2 and np.all(np.diff(grid_zero) > 0.0)


def test_heterogeneous_s_orders_of_magnitude_and_extreme_z_stability():
    rng = np.random.default_rng(808)
    J, R = 40, 4
    V = _make_corr_matrix(R, 0.6)

    # S spanning 8 orders of magnitude: 1e-4 to 1e4
    exps = rng.uniform(-4.0, 4.0, size=(J, R))
    Shat = 10.0 ** exps
    Bhat = rng.normal(0.0, 1.0, size=(J, R)) * Shat
    # Inject extreme |Z| = 100 signals
    Bhat[0, :] = 100.0 * Shat[0, :]
    Bhat[1, 0] = -100.0 * Shat[1, 0]

    data = mash_set_data(Bhat, Shat, V=V)
    U_list = {"identity": np.eye(R), "equal_effects": np.ones((R, R))}
    grid = np.array([1e-6, 1.0, 1e4])

    ll_mat = compute_loglik_matrix(data, U_list, grid, use_pointmass=True)
    assert np.all(np.isfinite(ll_mat))

    # Compare against scipy.stats.multivariate_normal.logpdf where its unscaled eigh
    # succeeds, and against an independent Jacobi-equilibrated slogdet/solve reference
    # everywhere (including kappa ~ 1e16 rows where scipy raises LinAlgError).
    comps = [np.zeros((R, R))]
    for u in U_list.values():
        for w in grid:
            comps.append(w * u)
    scipy_checked = 0
    for j in range(10):
        Vj = data.get_cov(j)
        for m, sig in enumerate(comps):
            C = sig + Vj
            d = np.sqrt(np.diag(C))
            C_eq = C / np.outer(d, d)
            _, logdet_eq = np.linalg.slogdet(C_eq)
            logdet = 2.0 * np.sum(np.log(d)) + logdet_eq
            b_eq = Bhat[j] / d
            maha = float(b_eq @ np.linalg.solve(C_eq, b_eq))
            ref_eq = -0.5 * (R * np.log(2.0 * np.pi) + logdet + maha)
            tol_m = 1e-5 if m == 6 else 1e-9
            np.testing.assert_allclose(ll_mat[j, m], ref_eq, rtol=tol_m, atol=tol_m)
            try:
                ref_sp = multivariate_normal.logpdf(
                    Bhat[j], mean=np.zeros(R), cov=C, allow_singular=False
                )
                np.testing.assert_allclose(ll_mat[j, m], ref_sp, rtol=1e-6, atol=1e-6)
                scipy_checked += 1
            except np.linalg.LinAlgError:
                pass
    assert scipy_checked >= 40

    res = mash(data, U_list, grid=grid, output_posterior_cov=True)
    assert np.all(np.isfinite(res.posterior_mean))
    assert np.all(np.isfinite(res.posterior_sd))
    assert np.all(np.isfinite(res.lfsr))
    assert np.all((res.lfsr >= 0.0) & (res.lfsr <= 1.0))
    assert res.lfsr[0, 0] < 1e-6


def test_optimize_pi_squarem_vs_em_kkt_optimality():
    rng = np.random.default_rng(909)
    J, R = 800, 3
    beta = np.zeros((J, R))
    beta[:200] = rng.normal(0.0, 2.0, size=(200, 1))
    Bhat = beta + rng.normal(0.0, 1.0, size=(J, R))
    data = mash_set_data(Bhat, 1.0)
    U_can = cov_canonical(data)
    grid = autoselect_grid(data, grid_mult=2.0)
    ll_mat = compute_loglik_matrix(data, U_can, grid)

    pi_mx, ll_mx, pen_mx, it_mx, conv_mx, kkt_mx = optimize_pi(
        ll_mat, opt_method="mixsqp", tol=1e-7, max_iter=200
    )
    pi_sq, ll_sq, pen_sq, it_sq, conv_sq, kkt_sq = optimize_pi(
        ll_mat, opt_method="squarem", tol=1e-6, max_iter=2000
    )
    pi_em, ll_em, pen_em, it_em, conv_em, kkt_em = optimize_pi(
        ll_mat, opt_method="em", tol=1e-5, max_iter=5000
    )

    assert conv_mx and kkt_mx <= 1e-7
    assert conv_sq and kkt_sq <= 1e-6
    assert conv_em and kkt_em <= 1e-5
    assert it_mx < it_sq < it_em
    assert np.isclose(np.sum(pi_mx), 1.0, atol=1e-12)
    assert np.all(pi_mx >= 0.0)
    np.testing.assert_allclose(pi_mx, pi_sq, atol=1e-4)
    np.testing.assert_allclose(pi_sq, pi_em, atol=5e-4)
    assert abs(pen_mx - pen_sq) < 1e-6
    assert abs(pen_sq - pen_em) < 1e-5

    M = ll_mat.shape[1]
    pi_unif = np.full(M, 1.0 / M)
    c = np.max(ll_mat, axis=1)
    ll_unif = float(np.sum(c) + np.sum(np.log(np.exp(ll_mat - c[:, None]) @ pi_unif)))
    assert pen_mx > ll_unif


def test_dataframe_io_result_methods_pcov_and_fix_g():
    rng = np.random.default_rng(1010)
    J, R = 60, 3
    features = [f"gene_{i}" for i in range(J)]
    conds = ["liver", "brain", "heart"]
    B_df = pd.DataFrame(rng.normal(0.0, 1.5, size=(J, R)), index=features, columns=conds)
    B_df.iloc[:15, :] += 3.5
    S_df = pd.DataFrame(rng.uniform(0.4, 0.9, size=(J, R)), index=features, columns=conds)
    V_df = pd.DataFrame(_make_corr_matrix(R, 0.2), index=conds, columns=conds)

    data = mash_set_data(B_df, S_df, V=V_df)
    assert data.feature_names == features
    assert data.condition_names == conds

    with pytest.raises(FrozenInstanceError):
        data.alpha = 1.0  # type: ignore[misc]

    res = mash(data, cov_canonical(data), output_posterior_cov=True)
    assert isinstance(res, MashResult)
    with pytest.raises(FrozenInstanceError):
        res.loglik = 0.0  # type: ignore[misc]

    assert res.posterior_cov is not None and res.posterior_cov.shape == (J, R, R)
    diag_cov = np.diagonal(res.posterior_cov, axis1=1, axis2=2)
    np.testing.assert_allclose(diag_cov, res.posterior_sd ** 2, rtol=1e-12, atol=1e-12)

    df_tidy = res.to_dataframe(sort_by_lfsr=True)
    assert list(df_tidy.columns) == [
        "feature",
        "condition",
        "posterior_mean",
        "posterior_sd",
        "lfsr",
        "negative_prob",
        "positive_prob",
        "zero_prob",
    ]
    assert len(df_tidy) == J * R
    assert np.all(np.diff(df_tidy["lfsr"].to_numpy()) >= -1e-15)

    df_sig = res.significant(thresh=0.05)
    assert np.all(df_sig["lfsr"] < 0.05)

    pm_df = res.posterior_mean_dataframe()
    psd_df = res.posterior_sd_dataframe()
    lfsr_df = res.lfsr_dataframe()
    assert list(pm_df.index) == features and list(pm_df.columns) == conds
    np.testing.assert_allclose(pm_df.to_numpy(), get_pm(res))
    np.testing.assert_allclose(psd_df.to_numpy(), get_psd(res))
    np.testing.assert_allclose(lfsr_df.to_numpy(), get_lfsr(res))

    pi_df = res.estimated_pi_dataframe()
    assert list(pi_df.columns) == ["component", "weight"]
    assert np.isclose(pi_df["weight"].sum(), 1.0, atol=1e-12)

    share_df = res.pairwise_sharing_dataframe(conditions=["liver", "heart"])
    assert list(share_df.index) == ["liver", "heart"]
    assert list(share_df.columns) == ["liver", "heart"]

    n_sig_cond = get_n_significant_conditions(res, thresh=0.05)
    assert n_sig_cond.shape == (J,) and np.all((n_sig_cond >= 0) & (n_sig_cond <= R))
    summary_str = res.summary()
    assert "MashResult" in summary_str and "converged=" in summary_str

    # Out-of-sample fix_g=True scoring
    data_oos = mash_set_data(B_df.iloc[:10], S_df.iloc[:10], V=V_df)
    res_oos = mash(data_oos, res.U_list, grid=res.grid, pi_init=res.pi, fix_g=True)
    assert res_oos.n_iter == 0 and res_oos.converged is True
    assert res_oos.posterior_mean.shape == (10, R)


def test_edge_cases_nans_r1_j1_and_validation_errors():
    # NaN / Inf / non-positive Shat / asymmetric or non-PD V rejections
    with pytest.raises(ValueError):
        mash_set_data(np.array([[1.0, np.nan], [0.0, 1.0]]), 1.0)
    with pytest.raises(ValueError):
        mash_set_data(np.ones((2, 2)), np.array([[1.0, np.inf], [1.0, 1.0]]))
    with pytest.raises(ValueError):
        mash_set_data(np.ones((2, 2)), np.array([[1.0, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError):
        mash_set_data(np.ones((2, 2)), -0.5)
    with pytest.raises(ValueError):
        mash_set_data(np.ones((2, 2)), 1.0, V=np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        mash_set_data(np.ones((2, 2)), 1.0, V=np.array([[1.0, 2.0], [2.0, 1.0]]))

    # zero_shat_reset works when Bhat == 0 at Shat == 0, rejects when Bhat != 0
    d_reset = mash_set_data(
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        zero_shat_reset=1e6,
    )
    assert d_reset.Shat[0, 0] == 1e6
    with pytest.raises(ValueError):
        mash_set_data(
            np.array([[1.0, 1.0], [2.0, 3.0]]),
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            zero_shat_reset=1e6,
        )

    # Mismatched DataFrame index/columns
    df1 = pd.DataFrame([[1.0, 2.0]], index=["a"], columns=["c0", "c1"])
    df2 = pd.DataFrame([[1.0, 1.0]], index=["b"], columns=["c0", "c1"])
    with pytest.raises(ValueError):
        mash_set_data(df1, df2)

    # cov_pca error on 0 strong signals or npc > R
    d_weak = mash_set_data(np.full((10, 3), 0.1), 1.0)
    with pytest.raises(ValueError):
        cov_pca(d_weak, npc=2, thresh=2.0)
    with pytest.raises(ValueError):
        cov_pca(d_weak, npc=4, thresh=0.0)

    # estimate_null_correlation_simple with < 2 null units
    d_strong = mash_set_data(np.full((5, 3), 10.0), 1.0)
    with pytest.raises(ValueError):
        estimate_null_correlation_simple(d_strong, z_thresh=2.0)

    # R = 1 end-to-end including estimate_null_correlation_simple & mash
    rng = np.random.default_rng(1111)
    d_r1 = mash_set_data(rng.normal(0.0, 1.0, size=(100, 1)), 1.0)
    v_r1 = estimate_null_correlation_simple(d_r1)
    assert v_r1.shape == (1, 1) and np.isclose(v_r1[0, 0], 1.0)
    res_r1 = mash(d_r1, cov_canonical(d_r1))
    assert res_r1.posterior_mean.shape == (100, 1)

    # J = 1 single-effect inference
    d_j1 = mash_set_data(np.array([[3.0, 3.0, 0.0]]), 1.0)
    res_j1 = mash(d_j1, cov_canonical(d_j1))
    assert res_j1.posterior_mean.shape == (1, 3)

    # get_pairwise_sharing returns NaN when 0 significant units exist
    d_pure_null = mash_set_data(np.zeros((20, 3)), 1.0)
    res_pure_null = mash(d_pure_null, cov_canonical(d_pure_null))
    sharing_nan = get_pairwise_sharing(res_pure_null, lfsr_thresh=0.05)
    assert np.all(np.isnan(sharing_nan))


def test_scale_invariance_sub_picoscale_and_large_scale():
    # 1. Exact sub-1e-12 heterogeneity, scaled V, unnormalized U, and lfsr exactness
    S_ht = np.array([[1e-13, 1e-13], [5e-13, 5e-13], [1e-12, 1e-12]])
    B_ht = S_ht.copy()  # Z = (1, 1)
    d_ht = mash_set_data(B_ht, S_ht)
    assert d_ht.is_common_cov is False

    ll_ht = compute_loglik_matrix(d_ht, {"id": np.eye(2)}, grid=np.array([1e-25]), use_pointmass=True)
    exact_null_r2 = -0.5 * (2.0 * np.log(2.0 * np.pi * 1e-24) + 2.0)
    np.testing.assert_allclose(ll_ht[2, 0], exact_null_r2, rtol=1e-12, atol=1e-12)

    r_ht = mash(d_ht, {"id": np.eye(2)}, grid=np.array([1e-25]), pi_init=np.array([0.0, 1.0]), fix_g=True)
    exact_lfsr = float(ndtr(-np.sqrt(10.0 / 11.0)))
    np.testing.assert_allclose(r_ht.lfsr[0], exact_lfsr, rtol=1e-12, atol=1e-12)

    V_scaled = 1e-13 * np.array([[1.0, 0.8], [0.8, 1.0]])
    d_vs = mash_set_data(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]), V=V_scaled)
    assert d_vs.is_diagonal_v is False
    # Unnormalized correlated U at 1e-26 scale must not trigger diagonal shortcut
    U_tiny_corr = {"corr": 1e-26 * np.array([[1.0, 0.8], [0.8, 1.0]])}
    ll_tiny_u = compute_loglik_matrix(d_ht, U_tiny_corr, grid=np.array([1.0]), use_pointmass=False)
    C_tiny_ref = U_tiny_corr["corr"] + d_ht.get_cov(0)
    _, ld_u = np.linalg.slogdet(C_tiny_ref)
    exact_ll_u0 = -0.5 * (2.0 * np.log(2.0 * np.pi) + ld_u + B_ht[0] @ np.linalg.solve(C_tiny_ref, B_ht[0]))
    np.testing.assert_allclose(ll_tiny_u[0, 0], exact_ll_u0, rtol=1e-10, atol=1e-10)

    # 2. Mixed signals (40 planted signals + 160 pure-null units) scaled by 1e-13 and 1e+6
    rng = np.random.default_rng(12345)
    J, R = 200, 4
    S_base = rng.uniform(1.0, 10.0, size=(J, R))
    V_corr = _make_corr_matrix(R, 0.35)

    for V_use in (None, V_corr):
        if V_use is None:
            B_base = rng.normal(0.0, 1.0, size=(J, R)) * S_base
        else:
            B_base = rng.multivariate_normal(np.zeros(R), V_corr, size=J) * S_base
        B_base[:40, :] += 5.0 * S_base[:40, :]

        d_ref = mash_set_data(B_base, S_base, V=V_use)
        r_ref = mash(d_ref, cov_canonical(d_ref), output_posterior_cov=True)
        null_disc_ref = int(np.sum(np.min(r_ref.lfsr[40:], axis=1) < 0.05))
        assert null_disc_ref == 0

        for scale_c in (1e-13, 1e6):
            d_c = mash_set_data(B_base * scale_c, S_base * scale_c, V=V_use)
            assert d_c.is_common_cov is False
            r_c = mash(d_c, cov_canonical(d_c), output_posterior_cov=True)

            np.testing.assert_allclose(r_c.grid / (scale_c ** 2), r_ref.grid, rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(
                r_c.loglik_matrix + R * np.log(scale_c),
                r_ref.loglik_matrix,
                rtol=1e-9,
                atol=1e-8,
            )
            np.testing.assert_allclose(r_c.pi, r_ref.pi, atol=1e-6)
            np.testing.assert_allclose(r_c.posterior_weights, r_ref.posterior_weights, atol=1e-6)
            np.testing.assert_allclose(r_c.lfsr, r_ref.lfsr, atol=1e-6)
            np.testing.assert_allclose(r_c.posterior_mean / scale_c, r_ref.posterior_mean, atol=1e-6)
            np.testing.assert_allclose(r_c.posterior_sd / scale_c, r_ref.posterior_sd, atol=1e-6)
            assert r_c.posterior_cov is not None and r_ref.posterior_cov is not None
            np.testing.assert_allclose(
                r_c.posterior_cov / (scale_c ** 2), r_ref.posterior_cov, atol=1e-6
            )
            assert int(np.sum(np.min(r_c.lfsr[40:], axis=1) < 0.05)) == 0

    # 3. Pure-null dataset (160 units) scaled by 1e-13 and 1e+6
    J_null = 160
    S_null = rng.uniform(1.0, 10.0, size=(J_null, R))
    B_null = rng.normal(0.0, 1.0, size=(J_null, R)) * S_null
    d_null_ref = mash_set_data(B_null, S_null)
    r_null_ref = mash(d_null_ref, cov_canonical(d_null_ref))
    assert int(np.sum(np.min(r_null_ref.lfsr, axis=1) < 0.05)) == 0

    for scale_c in (1e-13, 1e6):
        d_null_c = mash_set_data(B_null * scale_c, S_null * scale_c)
        r_null_c = mash(d_null_c, cov_canonical(d_null_c))
        np.testing.assert_allclose(r_null_c.lfsr, r_null_ref.lfsr, atol=1e-6)
        np.testing.assert_allclose(r_null_c.posterior_mean / scale_c, r_null_ref.posterior_mean, atol=1e-6)
        assert int(np.sum(np.min(r_null_c.lfsr, axis=1) < 0.05)) == 0

    # 4. cov_pca + cov_ed at extreme large/tiny scales without false PSD validation error
    B_ed = rng.normal(0.0, 3.0, size=(100, R))
    S_ed = np.full((100, R), 0.5)
    for scale_c in (1e-13, 1e6):
        d_ed = mash_set_data(B_ed * scale_c, S_ed * scale_c)
        u_pca = cov_pca(d_ed, npc=2, thresh=2.0)
        ed_res = fit_extreme_deconvolution(d_ed, u_pca, thresh=2.0, max_iter=50)
        df_ed = ed_res.to_dataframe()
        assert np.all(df_ed["rank"] >= 1)

