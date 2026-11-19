from __future__ import annotations

import numpy as np
import pytest

from pythonic_biostats import (
    bh_fdr,
    compute_mdes_interaction_hr,
    fit_cox_ph_efron,
    fit_fine_gray_ipcw,
    fit_wls_hc3,
    kaplan_meier_censoring_survival,
    partial_corr_pval,
    permutation_empirical_fdr,
    spearman_matrix,
    storey_qvalue,
    ttest_ind_safe,
)
from pythonic_biostats.survival import _fine_gray_loglik_and_derivatives, _km_censoring_curves


def test_cox_efron_and_fine_gray_competing_risks():
    rng = np.random.default_rng(123)
    n = 250
    x = rng.normal(0.0, 1.0, size=(n, 2))
    t1 = rng.exponential(1.0 / np.exp(0.9 * x[:, 0]))
    t2 = rng.exponential(1.0 / np.full(n, 0.35))
    tc = rng.uniform(0.5, 4.0, size=n)
    times = np.minimum(np.minimum(t1, t2), tc)
    events = np.zeros(n, dtype=int)
    events[(t1 <= t2) & (t1 <= tc)] = 1
    events[(t2 < t1) & (t2 <= tc)] = 2

    cox_res = fit_cox_ph_efron(x, times, events, target_event=1, coef_names=["driver", "noise"])
    fg_res = fit_fine_gray_ipcw(x, times, events, target_event=1, coef_names=["driver", "noise"])
    assert cox_res.converged and cox_res.beta[0] > 0.4
    assert fg_res.converged and fg_res.beta[0] > 0.3
    assert 1.5 < compute_mdes_interaction_hr(n_events=100, prop_stratum=0.35) < 2.2


def test_fine_gray_time_varying_weights_under_heavy_late_censoring():
    # Verify exact Fine & Gray (1999, Eq. 6) time-varying risk-set weights:
    # Before any censoring occurs (t = 1.5), a competing event at T = 1.0 MUST carry weight 1.0
    # (NOT G(t_max)/G(1.0)), whereas after heavy late censoring (t = 5.0), its weight decays to G(5.0-)/G(1.0-).
    times_toy = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 6.0])
    events_toy = np.array([2, 1, 0, 0, 0, 0, 1, 0])
    x_toy = np.array([[1.0], [0.0], [0.5], [-0.5], [0.2], [-0.2], [1.0], [0.0]])

    uniq_t, g_left, _, _, _ = _km_censoring_curves(times_toy, events_toy, eps_g=0.01)
    g_minus_subj = g_left[np.searchsorted(uniq_t, times_toy)]
    uniq_target = np.array([1.5, 5.0])
    g_minus_target = g_left[np.searchsorted(uniq_t, uniq_target)]

    # At t_1 = 1.5, no censoring has occurred yet -> G(1.5-) == 1.0 == G(1.0-)
    assert g_minus_target[0] == pytest.approx(1.0)
    # At t_2 = 5.0, four censorings occurred at 2.0, 2.5, 3.0, 3.5:
    # G(5.0-) = (5/6)*(4/5)*(3/4)*(2/3) = 2/6 = 1/3
    assert g_minus_target[1] == pytest.approx(1.0 / 3.0)

    ll0, grad0, _ = _fine_gray_loglik_and_derivatives(
        beta=np.array([0.0]),
        x=x_toy,
        times=times_toy,
        is_target=(events_toy == 1).astype(int),
        is_competing=(events_toy == 2),
        g_minus_subj=g_minus_subj,
        uniq_target_times=uniq_target,
        g_minus_target_times=g_minus_target,
    )
    # At beta = 0:
    # t = 1.5: 7 subjects >= 1.5 (weight 1) + 1 competing at 1.0 (weight 1.0/1.0 = 1.0) -> S0 = 8.0
    # t = 5.0: 2 subjects >= 5.0 (weight 1) + 1 competing at 1.0 (weight (1/3)/1.0 = 1/3) -> S0 = 7/3
    expected_ll0 = -np.log(8.0) - np.log(7.0 / 3.0)
    assert ll0 == pytest.approx(expected_ll0, rel=1e-12)
    assert np.isfinite(grad0[0])

    # Competing-risk divergence vs Cause-Specific Cox PH under heavy late censoring:
    # Covariate x strongly accelerates competing event 2 while having zero direct effect on cause-1 hazard.
    # Cause-specific Cox PH estimates beta ~ 0, whereas Fine-Gray subdistribution hazard estimates beta < 0.
    rng = np.random.default_rng(2026)
    n = 500
    x_sim = rng.normal(0.0, 1.0, size=(n, 1))
    t_cause1 = rng.exponential(scale=1.0 / 0.35, size=n)
    t_cause2 = rng.exponential(scale=1.0 / (0.60 * np.exp(1.2 * x_sim[:, 0])), size=n)
    # Heavy late censoring after t = 1.2
    t_cens = rng.uniform(1.2, 2.8, size=n)
    t_obs = np.minimum(np.minimum(t_cause1, t_cause2), t_cens)
    ev_obs = np.zeros(n, dtype=int)
    ev_obs[(t_cause1 <= t_cause2) & (t_cause1 <= t_cens)] = 1
    ev_obs[(t_cause2 < t_cause1) & (t_cause2 <= t_cens)] = 2

    cox = fit_cox_ph_efron(x_sim, t_obs, ev_obs, target_event=1)
    fg = fit_fine_gray_ipcw(x_sim, t_obs, ev_obs, target_event=1)
    assert cox.converged and fg.converged
    assert abs(cox.beta[0]) < 0.25
    assert fg.beta[0] < -0.35
    assert fg.p_value[0] < 0.01
    assert fg.se[0] > 0.0 and np.isfinite(fg.se[0])

    # Verify left-continuous vs right-continuous KM censoring helper
    km_left = kaplan_meier_censoring_survival(times_toy, events_toy, eps_g=0.01, left_continuous=True)
    km_right = kaplan_meier_censoring_survival(times_toy, events_toy, eps_g=0.01, left_continuous=False)
    assert km_left[2] == pytest.approx(1.0)  # just before censoring at t=2.0
    assert km_right[2] == pytest.approx(5.0 / 6.0)  # right after censoring at t=2.0


def test_wls_hc3_exact_analytical_leverage():
    rng = np.random.default_rng(99)
    n = 35
    X_raw = rng.normal(size=(n, 2))
    # Include an extreme high-leverage observation
    X_raw[0] = np.array([6.0, -5.0])
    weights = rng.uniform(0.05, 2.5, size=n)
    y = 1.2 + 0.7 * X_raw[:, 0] - 0.4 * X_raw[:, 1] + rng.normal(0.0, 0.5 / np.sqrt(weights), size=n)

    res = fit_wls_hc3(X_raw, y, weights=weights, add_intercept=True, coef_names=["x1", "x2"])

    # Analytical WLS + MacKinnon-White HC3 formula:
    X_design = np.column_stack([np.ones(n), X_raw])
    W = np.diag(weights)
    xtwx_inv = np.linalg.inv(X_design.T @ W @ X_design)
    beta_exact = xtwx_inv @ (X_design.T @ W @ y)
    e = y - X_design @ beta_exact
    # Hat matrix H = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}, so h_ii = w_i x_i^T (X^T W X)^{-1} x_i
    H = np.sqrt(weights)[:, None] * (X_design @ xtwx_inv @ X_design.T) * np.sqrt(weights)[None, :]
    h_ii = np.diag(H)
    omega = np.diag((weights**2) * (e**2) / ((1.0 - h_ii) ** 2))
    cov_exact = xtwx_inv @ (X_design.T @ omega @ X_design) @ xtwx_inv
    se_exact = np.sqrt(np.diag(cov_exact))

    np.testing.assert_allclose(res.beta, beta_exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.cov_hc3, cov_exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.se_hc3, se_exact, rtol=1e-12, atol=1e-12)

    # Rescaling weights by any positive constant c > 0 must leave beta and HC3 SE invariant
    res_scaled = fit_wls_hc3(X_raw, y, weights=17.5 * weights, add_intercept=True, coef_names=["x1", "x2"])
    np.testing.assert_allclose(res_scaled.beta, res.beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_scaled.se_hc3, res.se_hc3, rtol=1e-12, atol=1e-12)


def test_permutation_empirical_fdr_null_calibration_and_partial_corr_rank():
    rng = np.random.default_rng(7)
    n = 40
    y = np.linspace(-1.0, 1.0, n)
    # Construct features: exact orthogonal (r = 0), constant zero-variance, NaN feature, and perfect correlation
    x_zero_corr = y**2 - np.mean(y**2)  # symmetric quadratic is strictly orthogonal to symmetric linear y!
    x_const = np.full(n, 3.14)
    x_nan = rng.normal(size=n)
    x_nan[5] = np.nan
    x_signal = 2.5 * y + rng.normal(0.0, 0.05, size=n)
    X_null_bg = rng.normal(size=(n, 80))

    X = np.column_stack([x_signal, x_zero_corr, x_const, x_nan, X_null_bg])
    obs_r, emp_p, emp_fdr = permutation_empirical_fdr(
        X, y, method="pearson", n_perm=400, n_bins=2000, random_state=0
    )

    assert abs(obs_r[1]) < 1e-14
    # Exact r = 0, constant feature, and NaN feature must all get empirical_p == 1.0
    assert emp_p[1] == pytest.approx(1.0)
    assert np.isnan(obs_r[2]) and emp_p[2] == pytest.approx(1.0)
    assert np.isnan(obs_r[3]) and emp_p[3] == pytest.approx(1.0)
    assert emp_fdr[0] < 0.01

    assert spearman_matrix(X[:, [0, 1]], y)[0] > 0.95

    # Verify partial_corr_pval uses matrix_rank([1, C]) so duplicate/intercept columns don't corrupt df
    z = rng.normal(size=n)
    r1, p1 = partial_corr_pval(x_signal, y, covariates=z)
    cov_dup = np.column_stack([np.ones(n), z, 2.0 * z])  # rank([1, cov_dup]) == 2, same as [1, z]!
    r2, p2 = partial_corr_pval(x_signal, y, covariates=cov_dup)
    assert r1 == pytest.approx(r2, rel=1e-12)
    assert p1 == pytest.approx(p2, rel=1e-12)

    with pytest.raises(ValueError, match="Non-finite"):
        ttest_ind_safe([1.0, np.nan, 2.0], [1.0, 2.0, 3.0])


def test_storey_qvalue_pi0_recovery_and_monotonicity():
    rng = np.random.default_rng(42)
    m = 4000
    pi0_true = 0.70
    n_null = int(m * pi0_true)
    n_alt = m - n_null
    p_null = rng.uniform(0.0, 1.0, size=n_null)
    p_alt = rng.beta(0.15, 8.0, size=n_alt)
    pvals = np.concatenate([p_alt, p_null])

    qvals, pi0_hat = storey_qvalue(pvals)
    assert abs(pi0_hat - pi0_true) < 0.05
    assert np.all((qvals >= 0.0) & (qvals <= 1.0))

    # Monotonicity with respect to p-values and exact scaling relative to BH FDR
    order = np.argsort(pvals)
    assert np.all(np.diff(qvals[order]) >= -1e-15)
    np.testing.assert_allclose(qvals, pi0_hat * bh_fdr(pvals), rtol=1e-12, atol=1e-12)

    # Pure uniform null: pi0_hat ~ 1.0
    p_pure_null = rng.uniform(0.0, 1.0, size=2000)
    _, pi0_null = storey_qvalue(p_pure_null)
    assert 0.90 <= pi0_null <= 1.0
