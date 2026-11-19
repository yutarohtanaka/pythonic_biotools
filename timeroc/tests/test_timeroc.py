"""Comprehensive unit tests for `timeroc` (Blanche, Dartigues & Jacqmin-Gadda, Stat Med 2013)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from timeroc import compare_timeroc, time_roc


def _simulate_competing_risks(
    rng: np.random.Generator,
    n: int = 250,
    beta1: float = 0.9,
    beta2: float = 0.0,
    cens_rate: float = 0.35,
):
    marker = rng.standard_normal(n)
    t1 = rng.exponential(scale=np.exp(-beta1 * marker))
    t2 = rng.exponential(scale=2.2 * np.exp(-beta2 * marker))
    tc = rng.exponential(scale=1.0 / max(cens_rate, 1e-6), size=n)

    t_event = np.minimum(t1, t2)
    event_type = np.where(t1 <= t2, 1, 2)
    T = np.minimum(t_event, tc)
    delta = np.where(tc < t_event, 0, event_type)
    return T, delta, marker


def test_uncensored_exact_mann_whitney_and_trapezoidal_roc_area():
    rng = np.random.default_rng(10)
    n = 120
    # Include deliberate ties in marker values
    marker = np.round(rng.normal(0.0, 1.0, size=n), 1)
    T = rng.exponential(scale=np.exp(-0.8 * marker))
    delta = np.ones(n, dtype=int)  # Zero censoring, zero competing risks

    horizon = 1.0
    fit = time_roc(T, delta, marker, times=[horizon], cause=1, iid=True)

    cases = marker[T <= horizon]
    controls = marker[T > horizon]
    u_stat, _ = stats.mannwhitneyu(cases, controls, alternative="two-sided")
    exact_auc = u_stat / (cases.size * controls.size)

    assert fit.has_competing_risks is False
    assert np.isclose(fit.auc_1[0], exact_auc, rtol=1e-13)
    assert np.isclose(fit.auc_2[0], exact_auc, rtol=1e-13)

    # Trapezoidal integration of (FP_1, Se) matches Mann-Whitney AUC to machine precision even with ties
    roc_df = fit.roc_curve(0)
    trapz_auc = float(np.trapezoid(roc_df["Se"].to_numpy()[::-1], roc_df["FP_1"].to_numpy()[::-1]))
    assert np.isclose(trapz_auc, exact_auc, rtol=1e-13)


def test_ipcw_partition_of_unity_with_ties_and_competing_risks():
    rng = np.random.default_rng(22)
    # Discretize times to force heavy ties among primary events, competing events, and censorings!
    T_raw, delta, marker = _simulate_competing_risks(rng, n=200, beta1=0.7, beta2=0.3, cens_rate=0.4)
    T_tied = np.ceil(T_raw * 8.0) / 8.0

    horizons = [0.25, 0.50, 0.75, 1.00, 1.25]
    fit = time_roc(T_tied, delta, marker, times=horizons, cause=1)

    total_prob = fit.cum_inc_1 + fit.cum_inc_2 + fit.surv_prob
    np.testing.assert_allclose(total_prob, np.ones_like(total_prob), rtol=1e-13, atol=1e-14)
    assert fit.has_competing_risks is True


def test_competing_risks_definition1_vs_definition2_behavior():
    rng = np.random.default_rng(33)
    # High marker increases BOTH primary event (cause=1) AND competing event (cause=2)!
    # Event-free survivors (T > t) strongly enrich for LOW marker values -> AUC_2 > AUC_1!
    T, delta, marker = _simulate_competing_risks(rng, n=500, beta1=1.0, beta2=1.1, cens_rate=0.25)
    fit_cr = time_roc(T, delta, marker, times=[0.6, 1.0], cause=1)
    assert np.all(fit_cr.auc_2 > fit_cr.auc_1 + 0.04)

    # Without competing risks (re-coding competing events as censored or survival-only cohort),
    # Definition 1 and Definition 2 coincide identically!
    delta_surv_only = np.where(delta == 2, 0, delta)
    fit_surv = time_roc(T, delta_surv_only, marker, times=[0.6, 1.0], cause=1, iid=True)
    np.testing.assert_allclose(fit_surv.auc_1, fit_surv.auc_2, rtol=1e-14)
    np.testing.assert_allclose(fit_surv.iid_auc_1, fit_surv.iid_auc_2, rtol=1e-14)


def test_iid_influence_function_matches_leave_one_out_jackknife():
    """Verify exact Blanche et al. (2013) martingale IID representation against LOO jackknife."""
    rng = np.random.default_rng(44)
    n = 160
    T, delta, marker = _simulate_competing_risks(rng, n=n, beta1=0.85, beta2=0.45, cens_rate=0.40)
    horizons = [0.5, 0.9]

    fit = time_roc(T, delta, marker, times=horizons, cause=1, iid=True)
    assert fit.iid_auc_1 is not None and fit.iid_auc_2 is not None

    # Exact zero-mean property of empirical influence functions
    np.testing.assert_allclose(np.sum(fit.iid_auc_1, axis=0), [0.0, 0.0], atol=1e-11)
    np.testing.assert_allclose(np.sum(fit.iid_auc_2, axis=0), [0.0, 0.0], atol=1e-11)

    # Compare per-subject influence functions against brute-force leave-one-subject-out jackknife!
    jk_phi_1 = np.empty((n, 2), dtype=np.float64)
    jk_phi_2 = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = time_roc(T[mask], delta[mask], marker[mask], times=horizons, cause=1, iid=False)
        jk_phi_1[i] = (n - 1) * (fit.auc_1 - sub.auc_1)
        jk_phi_2[i] = (n - 1) * (fit.auc_2 - sub.auc_2)

    for col in range(2):
        r1 = np.corrcoef(fit.iid_auc_1[:, col], jk_phi_1[:, col])[0, 1]
        r2 = np.corrcoef(fit.iid_auc_2[:, col], jk_phi_2[:, col])[0, 1]
        assert r1 > 0.996
        assert r2 > 0.996
        se_jk_1 = float(np.sqrt(np.mean(jk_phi_1[:, col] ** 2) / n))
        se_jk_2 = float(np.sqrt(np.mean(jk_phi_2[:, col] ** 2) / n))
        assert np.isclose(fit.se_auc_1[col], se_jk_1, rtol=0.04)
        assert np.isclose(fit.se_auc_2[col], se_jk_2, rtol=0.04)


def test_monotone_transformation_and_orientation_invariance():
    rng = np.random.default_rng(55)
    T, delta, marker = _simulate_competing_risks(rng, n=180, beta1=0.75, beta2=-0.2, cens_rate=0.3)

    fit_orig = time_roc(T, delta, marker, times=[0.5, 1.0], cause=1, iid=True)
    # Strictly increasing nonlinear transform preserves rank order identically
    fit_mono = time_roc(T, delta, np.exp(0.7 * marker) + 5.0, times=[0.5, 1.0], cause=1, iid=True)

    np.testing.assert_allclose(fit_orig.auc_1, fit_mono.auc_1, rtol=1e-14)
    np.testing.assert_allclose(fit_orig.auc_2, fit_mono.auc_2, rtol=1e-14)
    np.testing.assert_allclose(fit_orig.iid_auc_1, fit_mono.iid_auc_1, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fit_orig.iid_auc_2, fit_mono.iid_auc_2, rtol=1e-13, atol=1e-14)

    # Negating marker flips AUC -> 1 - AUC and negates IID influence functions
    fit_neg = time_roc(T, delta, -marker, times=[0.5, 1.0], cause=1, iid=True)
    np.testing.assert_allclose(fit_neg.auc_1, 1.0 - fit_orig.auc_1, rtol=1e-14)
    np.testing.assert_allclose(fit_neg.iid_auc_1, -fit_orig.iid_auc_1, rtol=1e-13, atol=1e-14)


def test_confidence_intervals_and_simultaneous_bands():
    rng = np.random.default_rng(66)
    T, delta, marker = _simulate_competing_risks(rng, n=300, beta1=0.8, beta2=0.2, cens_rate=0.3)
    fit = time_roc(T, delta, marker, times=[0.3, 0.6, 0.9, 1.2], cause=1, iid=True)

    cb1 = fit.confidence_intervals(level=0.95, definition=1, n_sim=3000, seed=123)
    assert cb1.c_simultaneous > cb1.z_pointwise
    assert np.all(cb1.cb_lower <= cb1.ci_lower)
    assert np.all(cb1.cb_upper >= cb1.ci_upper)
    assert cb1.to_dataframe().shape == (4, 7)

    # At a single timepoint, simultaneous critical value ~ 1.96 normal quantile
    fit_1t = time_roc(T, delta, marker, times=[0.8], cause=1, iid=True)
    cb_1t = fit_1t.confidence_intervals(level=0.95, definition=1, n_sim=5000, seed=123)
    assert np.isclose(cb_1t.c_simultaneous, cb_1t.z_pointwise, atol=0.06)


def test_cox_conditional_ipcw_informative_censoring():
    rng = np.random.default_rng(77)
    n = 600
    marker = rng.standard_normal(n)
    cov_z = rng.standard_normal(n)
    t_event = rng.exponential(scale=np.exp(-0.9 * marker))
    # Informative censoring strongly driven by marker and cov_z
    t_cens = rng.exponential(scale=np.exp(-0.8 * marker - 0.7 * cov_z) * 1.8)

    T = np.minimum(t_event, t_cens)
    delta = np.where(t_event <= t_cens, 1, 0)

    # Oracle uncensored benchmark on a large draw from the same DGP
    m_ora = rng.standard_normal(8000)
    t_ora = rng.exponential(scale=np.exp(-0.9 * m_ora))
    u_ora, _ = stats.mannwhitneyu(m_ora[t_ora <= 0.8], m_ora[t_ora > 0.8])
    true_auc = u_ora / (np.sum(t_ora <= 0.8) * np.sum(t_ora > 0.8))

    fit_cox = time_roc(T, delta, marker, times=[0.8], weighting="cox", other_markers=cov_z)
    assert fit_cox.iid_auc_1 is None
    assert abs(fit_cox.auc_1[0] - true_auc) < 0.04


def test_compare_timeroc_power_null_and_self_identity():
    rng = np.random.default_rng(88)
    n = 320
    T, delta, strong_marker = _simulate_competing_risks(rng, n=n, beta1=1.0, beta2=0.2, cens_rate=0.3)
    weak_marker = 0.25 * strong_marker + rng.standard_normal(n)

    horizons = [0.4, 0.8, 1.2]
    fit_strong = time_roc(T, delta, strong_marker, times=horizons, cause=1, iid=True)
    fit_weak = time_roc(T, delta, weak_marker, times=horizons, cause=1, iid=True)

    cmp = compare_timeroc(fit_strong, fit_weak, definition=1, n_sim=2500, seed=42)
    assert np.all(cmp.delta_auc > 0.10)
    assert np.all(cmp.p_values < 0.005)
    assert np.all(cmp.p_values_adjusted >= cmp.p_values)
    assert cmp.p_wald_joint < 0.001
    assert cmp.wald_df == 3
    assert "Joint Multi-Horizon Wald Test" in cmp.summary()

    # Comparing two equally noisy transformations under H0: delta_AUC ~ 0, p > 0.05
    noise_a = rng.standard_normal(n)
    noise_b = rng.standard_normal(n)
    fit_a = time_roc(T, delta, noise_a, times=horizons, cause=1, iid=True)
    fit_b = time_roc(T, delta, noise_b, times=horizons, cause=1, iid=True)
    cmp_null = compare_timeroc(fit_a, fit_b, definition=1, seed=42)
    assert cmp_null.p_wald_joint > 0.05

    # Self-comparison handles rank-0 zero-covariance gracefully
    cmp_self = compare_timeroc(fit_strong, fit_strong, definition=1)
    np.testing.assert_allclose(cmp_self.delta_auc, 0.0, atol=1e-15)
    assert cmp_self.wald_stat == 0.0
    assert cmp_self.p_wald_joint == 1.0


def test_summary_and_dataframe_serialization():
    rng = np.random.default_rng(99)
    T, delta, marker = _simulate_competing_risks(rng, n=150, beta1=0.8, beta2=0.3, cens_rate=0.25)
    fit = time_roc(T, delta, marker, times=[0.5, 1.0], cause=1, iid=True)
    df = fit.to_dataframe()
    assert df.shape[0] == 2
    assert {"AUC_1", "AUC_2", "SE_1", "CB_1_lower", "CB_1_upper"}.issubset(df.columns)
    assert "Time-Dependent IPCW ROC Summary" in fit.summary()


def test_invalid_inputs_raise_value_error():
    T = np.array([0.2, 0.5, 0.8, 1.2, 1.5])
    delta = np.array([1, 0, 1, 2, 0])
    marker = np.array([1.2, -0.4, 0.8, 0.1, -1.0])

    # NaN in marker or T
    m_nan = marker.copy()
    m_nan[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        time_roc(T, delta, m_nan, times=[0.6])

    # Negative follow-up time or non-positive horizon
    t_neg = T.copy()
    t_neg[0] = -0.1
    with pytest.raises(ValueError, match=">= 0"):
        time_roc(t_neg, delta, marker, times=[0.6])
    with pytest.raises(ValueError, match="> 0"):
        time_roc(T, delta, marker, times=[0.0])

    # Non-integer or negative delta
    with pytest.raises(ValueError, match="integer"):
        time_roc(T, np.array([1.5, 0, 1, 2, 0]), marker, times=[0.6])

    # Missing primary cause in cohort
    with pytest.raises(ValueError, match="No primary events"):
        time_roc(T, delta, marker, times=[0.6], cause=9)

    # iid=True with weighting='cox' raises informative ValueError
    with pytest.raises(ValueError, match="marginal"):
        time_roc(T, delta, marker, times=[0.6], weighting="cox", iid=True)

    # Mismatched cohorts in compare_timeroc
    fit1 = time_roc(T, delta, marker, times=[0.6], iid=True)
    fit2 = time_roc(T + 0.1, delta, marker, times=[0.6], iid=True)
    with pytest.raises(ValueError, match="do not match"):
        compare_timeroc(fit1, fit2)
