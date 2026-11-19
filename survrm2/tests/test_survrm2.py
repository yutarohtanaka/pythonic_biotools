"""Unit tests for survrm2 (Restricted Mean Survival Time & RMTL inference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import survrm2
from survrm2 import (
    SurvRM2,
    rmst1,
    rmst2,
    rmst2_curve,
    rmst2reg,
)


def test_rmst1_closed_form_uncensored_and_censored() -> None:
    """Verify exact hand-calculated KM step integral and Greenwood variance."""
    # Uncensored sample: times [2, 4, 6, 8], status [1, 1, 1, 1], tau = 7.0
    # S(t): [0,2)->1.0, [2,4)->0.75, [4,6)->0.50, [6,7]->0.25
    # RMST(7) = 2*1.0 + 2*0.75 + 2*0.50 + 1*0.25 = 4.75
    time = np.array([2.0, 4.0, 6.0, 8.0])
    status = np.array([1, 1, 1, 1])
    res = rmst1(time, status, tau=7.0, settings="R_default")

    assert res.rmst == pytest.approx(4.75, abs=1e-12)
    assert res.rmtl == pytest.approx(2.25, abs=1e-12)
    assert res.var_rmst == pytest.approx(res.var_rmtl, abs=1e-15)

    # Tail areas A_k from t_k to 7:
    # t1=2: 1.5 + 1.0 + 0.25 = 2.75, v1 = 1/(4*3) = 1/12
    # t2=4: 1.0 + 0.25 = 1.25, v2 = 1/(3*2) = 1/6
    # t3=6: 0.25, v3 = 1/(2*1) = 1/2
    expected_var = (2.75 ** 2) / 12.0 + (1.25 ** 2) / 6.0 + (0.25 ** 2) / 2.0
    assert res.var_rmst == pytest.approx(expected_var, abs=1e-12)
    assert res.se_rmst == pytest.approx(np.sqrt(expected_var), abs=1e-12)


def test_rmst1_boundary_atom_default_vs_r_default() -> None:
    """Verify Y_k == d_k boundary atom handling in default vs R_default."""
    time = np.array([1.0, 3.0, 5.0, 5.0])
    status = np.array([1, 1, 1, 1])  # At t=5.0, Y_k = 2, d_k = 2
    res_def = rmst1(time, status, tau=6.0, settings="default")
    res_r = rmst1(time, status, tau=6.0, settings="R_default")

    # At t=5.0, S(5) = 0, so RMST(6) = 1*1 + 2*(3/4) + 2*(2/4) + 1*0 = 3.5 in both
    assert res_def.rmst == pytest.approx(3.5, abs=1e-12)
    assert res_r.rmst == pytest.approx(3.5, abs=1e-12)

    # Boundary hazard_var at t=5.0: default gives d_k / Y_k^2 = 2 / 4 = 0.5; R_default gives 0.0
    assert float(res_def.km_df.iloc[-1]["hazard_var"]) == pytest.approx(0.5, abs=1e-12)
    assert float(res_r.km_df.iloc[-1]["hazard_var"]) == pytest.approx(0.0, abs=1e-12)


def test_tau_admissibility_event_vs_censored_max() -> None:
    """Verify survRM2 tau admissibility when max time is event vs censored."""
    # Arm 0 ends with an EVENT at t=8 -> S_0(8) = 0, defined on [8, inf)
    # Arm 1 ends with a CENSOR at t=12 -> S_1(12) > 0, defined only up to 12
    t0 = np.array([2.0, 4.0, 6.0, 8.0])
    d0 = np.array([1, 1, 1, 1])
    t1 = np.array([3.0, 5.0, 9.0, 12.0])
    d1 = np.array([1, 1, 1, 0])

    time = np.concatenate([t0, t1])
    status = np.concatenate([d0, d1])
    arm = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Default tau = min(8, 12) = 8.0
    res_auto = rmst2(time, status, arm, tau=None, settings="R_default")
    assert res_auto.tau == pytest.approx(8.0)

    # tau = 10.0 (> 8.0, <= 12.0) IS admissible because arm 0 reached S=0 at t=8!
    res_10 = rmst2(time, status, arm, tau=10.0, settings="R_default")
    assert res_10.arm0.rmst == pytest.approx(res_auto.arm0.rmst, abs=1e-12)
    assert res_10.arm1.rmst > res_auto.arm1.rmst

    # tau = 13.0 (> 12.0 where arm 1 is censored) MUST raise ValueError
    with pytest.raises(ValueError, match="exceeds"):
        rmst2(time, status, arm, tau=13.0, settings="R_default")


def test_rmst2_unadjusted_contrasts_and_log_delta_ci() -> None:
    """Verify unadjusted RMST diff, RMST ratio, and RMTL ratio formulas."""
    rng = np.random.default_rng(123)
    n_arm = 120
    t0 = rng.exponential(scale=4.0, size=n_arm)
    t1 = rng.exponential(scale=8.0, size=n_arm)
    c0 = rng.exponential(scale=12.0, size=n_arm)
    c1 = rng.exponential(scale=12.0, size=n_arm)

    time = np.concatenate([np.minimum(t0, c0), np.minimum(t1, c1)])
    status = np.concatenate([(t0 <= c0).astype(int), (t1 <= c1).astype(int)])
    arm = np.concatenate([np.zeros(n_arm, dtype=int), np.ones(n_arm, dtype=int)])

    res = rmst2(time, status, arm, tau=6.0, settings="R_default")
    assert res.rmst_diff_10.est == pytest.approx(res.arm1.rmst - res.arm0.rmst, abs=1e-12)
    assert res.rmst_ratio_10.est == pytest.approx(res.arm1.rmst / res.arm0.rmst, abs=1e-12)
    assert res.rmtl_ratio_10.est == pytest.approx(res.arm1.rmtl / res.arm0.rmtl, abs=1e-12)

    # Arm 1 has double the mean survival of Arm 0 -> diff > 0, ratio > 1, RMTL ratio < 1
    assert res.rmst_diff_10.est > 0.5
    assert res.rmst_diff_10.pval < 0.01
    assert res.rmst_ratio_10.est > 1.15
    assert res.rmtl_ratio_10.est < 0.85

    # Dict-style access on ContrastResult
    assert res.rmst_diff_10["est"] == res.rmst_diff_10.est
    assert res.rmst_diff_10.ci == (res.rmst_diff_10.ci_lower, res.rmst_diff_10.ci_upper)


def test_exponential_population_rmst_consistency() -> None:
    """Check large-sample convergence to analytical exponential RMST integral."""
    # For Exp(rate = lambda), RMST(tau) = (1 - exp(-lambda * tau)) / lambda
    rng = np.random.default_rng(2026)
    n = 1500
    lam = 0.25
    tau = 5.0
    true_rmst = (1.0 - np.exp(-lam * tau)) / lam

    t_true = rng.exponential(scale=1.0 / lam, size=n)
    c_time = rng.uniform(3.0, 12.0, size=n)
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    res = rmst1(time, status, tau=tau, settings="default")
    assert abs(res.rmst - true_rmst) < 2.5 * res.se_rmst
    assert res.rmst_ci[0] < true_rmst < res.rmst_ci[1]


def test_ipcw_zero_covariates_matches_kaplan_meier_exactly() -> None:
    """Prove Satten-Datta theorem: IPCW with empty covariates equals KM RMST even with ties!"""
    # Include exact ties between events and censorings!
    t0 = np.array([2.0, 2.0, 4.0, 5.0, 7.0, 9.0])
    d0 = np.array([1, 0, 1, 0, 1, 1])
    t1 = np.array([3.0, 3.0, 6.0, 6.0, 8.0, 10.0])
    d1 = np.array([1, 0, 1, 0, 1, 1])

    time = np.concatenate([t0, t1])
    status = np.concatenate([d0, d1])
    arm = np.array([0] * 6 + [1] * 6)
    empty_cov = np.empty((12, 0), dtype=np.float64)

    km_res = rmst2(time, status, arm, tau=8.5, settings="R_default")
    ipcw_id = rmst2reg(time, status, arm, empty_cov, tau=8.5, link="identity", settings="R_default")
    ipcw_log = rmst2reg(time, status, arm, empty_cov, tau=8.5, link="log_rmst", settings="R_default")
    ipcw_rmtl = rmst2reg(time, status, arm, empty_cov, tau=8.5, link="log_rmtl", settings="R_default")

    assert ipcw_id.contrast_est == pytest.approx(km_res.rmst_diff_10.est, abs=1e-12)
    assert ipcw_log.contrast_est == pytest.approx(km_res.rmst_ratio_10.est, abs=1e-10)
    assert ipcw_rmtl.contrast_est == pytest.approx(km_res.rmtl_ratio_10.est, abs=1e-10)


def test_martingale_influence_sum_zero_and_variance_reduction() -> None:
    """Verify Tian et al. (2014) influence vectors sum to zero and project out censoring noise."""
    rng = np.random.default_rng(77)
    n = 240
    arm = rng.integers(0, 2, size=n)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    cov_df = pd.DataFrame({"biomarker": x1, "stage": x2})

    t_true = rng.exponential(scale=np.exp(1.6 + 0.5 * arm - 0.35 * x1 - 0.2 * x2))
    c_time = rng.exponential(scale=7.5, size=n)
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    for settings_mode in ("default", "R_default"):
        for lnk in ("identity", "log_rmst", "log_rmtl"):
            mod = rmst2reg(
                time, status, arm, cov_df, tau=6.0, link=lnk, settings=settings_mode  # type: ignore[arg-type]
            )
            assert mod.converged
            # Sum of influence functions must vanish to machine precision
            col_sums = np.sum(mod.influence_matrix, axis=0)
            np.testing.assert_allclose(col_sums, 0.0, atol=1e-8)

            # Covariance matrix must be symmetric positive semidefinite
            np.testing.assert_allclose(mod.vcov, mod.vcov.T, atol=1e-12)
            eigvals = np.linalg.eigvalsh(mod.vcov)
            assert np.all(eigvals >= -1e-12)


def test_ancova_recovers_true_additive_and_multiplicative_effects() -> None:
    """Verify covariate-adjusted ANCOVA recovers true arm effect under confounding."""
    rng = np.random.default_rng(314)
    n = 400
    x = rng.normal(0.0, 1.0, size=n)
    # Confounded treatment assignment depending on prognostic covariate x
    prob_arm = 1.0 / (1.0 + np.exp(-1.1 * x))
    arm = rng.binomial(1, prob_arm)

    # True survival strongly improved by arm (+2.0 additive before tau) and confounded by -x
    t_true = np.clip(4.0 + 2.0 * arm - 1.2 * x + rng.normal(0.0, 0.7, size=n), 0.2, 20.0)
    c_time = rng.uniform(5.0, 15.0, size=n)
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    res = rmst2(time, status, arm, tau=8.0, covariates=pd.Series(x, name="confounder"), settings="default")
    assert res.adjusted
    assert res.adj_rmst_diff_10 is not None
    # Unadjusted difference is downward-biased by confounding (~0.9), whereas adjusted recovers ~2.0!
    assert abs(res.adj_rmst_diff_10.est - 2.0) < 0.35
    assert abs(res.rmst_diff_10.est - 2.0) > 0.55


def test_default_vs_r_default_differential_arm_censoring() -> None:
    """Verify pooled KM (default) vs per-arm KM (R_default) differ under arm-dependent censoring."""
    rng = np.random.default_rng(999)
    n_arm = 100
    arm = np.array([0] * n_arm + [1] * n_arm)
    x = rng.normal(0.0, 1.0, size=2 * n_arm)
    t_true = rng.exponential(scale=np.exp(1.5 + 0.4 * arm - 0.3 * x))
    # Heavy censoring in arm 1, light censoring in arm 0
    c_time = np.where(
        arm == 1,
        rng.exponential(scale=3.5, size=2 * n_arm),
        rng.exponential(scale=25.0, size=2 * n_arm),
    )
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    res_def = rmst2(time, status, arm, tau=4.5, covariates=x, settings="default")
    res_r = rmst2(time, status, arm, tau=4.5, covariates=x, settings="R_default")

    assert res_def.adj_rmst_diff_10 is not None and res_r.adj_rmst_diff_10 is not None
    assert res_def.adj_rmst_diff_10.est != pytest.approx(res_r.adj_rmst_diff_10.est, abs=1e-4)


def test_non_proportional_hazards_delayed_effect_detection() -> None:
    """Verify RMST detects delayed immunotherapy-like benefit."""
    rng = np.random.default_rng(55)
    n_arm = 250
    # Identical hazard on [0, 2], then 4x lower hazard in arm 1 after t=2
    u0 = rng.uniform(size=n_arm)
    t0 = -np.log(u0) / 0.35
    u1 = rng.uniform(size=n_arm)
    # Piecewise exponential survival S1(2) = exp(-0.7)
    s_knot = np.exp(-0.35 * 2.0)
    t1 = np.where(
        u1 > s_knot,
        -np.log(u1) / 0.35,
        2.0 - np.log(u1 / s_knot) / 0.08,
    )
    c = rng.uniform(8.0, 14.0, size=2 * n_arm)
    time = np.minimum(np.concatenate([t0, t1]), c)
    status = (np.concatenate([t0, t1]) <= c).astype(int)
    arm = np.array([0] * n_arm + [1] * n_arm)

    curve = rmst2_curve(time, status, arm, tau_grid=[1.8, 4.0, 8.0], settings="default")
    assert curve.shape[0] == 3
    # Analytical difference at tau=8: exp(-0.7)*((1-exp(-0.48))/0.08 - (1-exp(-2.1))/0.35) ~= 1.121
    true_diff_8 = s_knot * ((1.0 - np.exp(-0.08 * 6.0)) / 0.08 - (1.0 - np.exp(-0.35 * 6.0)) / 0.35)
    assert abs(float(curve.loc[0, "rmst_diff_10"])) < 0.3
    assert abs(float(curve.loc[2, "rmst_diff_10"]) - true_diff_8) < 0.35
    assert float(curve.loc[2, "rmst_diff_pval"]) < 1e-3


def test_martingale_se_shorter_than_naive_ipcw_se() -> None:
    """Prove censoring-martingale projection strictly reduces variance vs naive IPCW."""
    rng = np.random.default_rng(2024)
    n = 300
    arm = rng.integers(0, 2, size=n)
    x = rng.normal(0.0, 1.0, size=n)
    t_true = rng.exponential(scale=np.exp(1.5 + 0.35 * arm - 0.3 * x))
    c_time = rng.exponential(scale=5.0, size=n)  # substantial censoring before tau=6
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    mod = rmst2reg(time, status, arm, x, tau=6.0, link="identity", settings="R_default")
    # Compare trace of martingale-corrected meat B^T B vs naive u^T u on intercept
    z_mat = np.column_stack([np.ones(n), arm, x])
    y_star = np.minimum(time, 6.0)
    mu = z_mat @ mod.coef_df["coef"].to_numpy()
    u_mat = z_mat * (mod.ipcw_weights * (y_star - mu))[:, None]
    naive_meat_00 = float(np.sum(u_mat[:, 0] ** 2))
    mart_meat_00 = float(np.sum(mod.influence_matrix[:, 0] ** 2))
    assert mart_meat_00 < naive_meat_00


def test_rmst2_curve_with_covariates_and_auto_grid() -> None:
    """Verify rmst2_curve automatic grid generation with covariate adjustment."""
    rng = np.random.default_rng(101)
    n = 120
    arm = rng.integers(0, 2, size=n)
    cov = rng.normal(size=(n, 2))
    t_true = rng.exponential(scale=np.exp(1.7 + 0.3 * arm - 0.2 * cov[:, 0]))
    c_time = rng.uniform(4.0, 12.0, size=n)
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    df = rmst2_curve(time, status, arm, covariates=cov, n_grid=6, settings="default")
    assert len(df) == 6
    assert "adj_rmst_diff_10" in df.columns
    assert np.all(np.diff(df["tau"].to_numpy()) > 0.0)


def test_survrm2_oop_estimator_workflow() -> None:
    """Test SurvRM2 class fit, fit_single, curve, summary, and to_dataframe."""
    rng = np.random.default_rng(88)
    n = 100
    arm = np.array([0] * 50 + [1] * 50)
    t_true = rng.exponential(scale=np.where(arm == 1, 7.0, 4.0))
    c_time = rng.uniform(5.0, 10.0, size=n)
    time = np.minimum(t_true, c_time)
    status = (t_true <= c_time).astype(int)

    model = SurvRM2(tau=5.0, alpha=0.05, settings="default")
    with pytest.raises(RuntimeError):
        model.summary()

    model.fit(time, status, arm)
    summary_txt = model.summary()
    assert "Two-Sample RMST Analysis" in summary_txt
    df = model.to_dataframe()
    assert df.shape[0] == 3

    single = model.fit_single(time[arm == 1], status[arm == 1])
    assert single.rmst == pytest.approx(model.result_.arm1.rmst)  # type: ignore[union-attr]
    assert "Single-Arm RMST" in single.summary()
    assert single.to_dataframe().shape[0] == 2

    curve_df = model.curve(n_grid=5)
    assert curve_df.shape[0] == 5


def test_zero_events_before_tau_handled_cleanly() -> None:
    """When zero events occur before tau, RMST == tau, SE == 0, and RMTL ratio is NaN without warnings."""
    time = np.array([5.0, 6.0, 7.0, 8.0, 5.5, 6.5, 7.5, 8.5])
    status = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    arm = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    res = rmst2(time, status, arm, tau=4.0, settings="default")
    assert res.arm1.rmst == pytest.approx(4.0)
    assert res.arm0.rmst == pytest.approx(4.0)
    assert res.rmst_diff_10.est == pytest.approx(0.0)
    assert res.rmst_diff_10.se == pytest.approx(0.0)
    assert res.rmst_ratio_10.est == pytest.approx(1.0)
    assert np.isnan(res.rmtl_ratio_10.est)


def test_settings_validation_every_entrypoint() -> None:
    """Every public function and class must reject unknown settings strings with ValueError."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5])
    status = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    arm = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cov = np.ones(8)

    with pytest.raises(ValueError, match="settings"):
        rmst1(time[:4], status[:4], tau=2.0, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        rmst2(time, status, arm, tau=2.0, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        rmst2reg(time, status, arm, cov, tau=2.0, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        rmst2_curve(time, status, arm, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        SurvRM2(settings="bad")  # type: ignore[arg-type]


def test_input_validation_errors() -> None:
    """Check invalid time, status, arm, alpha, tau, and covariates inputs raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        rmst1([-1.0, 2.0], [1, 1], tau=1.5)
    with pytest.raises(ValueError, match="binary"):
        rmst1([1.0, 2.0], [1, 2], tau=1.5)
    with pytest.raises(ValueError, match="alpha"):
        rmst1([1.0, 2.0], [1, 1], tau=1.5, alpha=1.2)
    with pytest.raises(ValueError, match="positive"):
        rmst1([1.0, 2.0], [1, 1], tau=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        rmst2([1.0, 2.0, 3.0], [1, 1, 1], [0, 1, 1], tau=2.0)
    with pytest.raises(ValueError, match="rows"):
        rmst2([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], [0, 0, 1, 1], tau=2.0, covariates=np.ones(3))
    assert hasattr(survrm2, "__version__")
