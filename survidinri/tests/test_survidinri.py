"""Comprehensive unit tests for the standalone survidinri package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survidinri import SurvIDINRI, idi_inf, idi_inf_predictions, weighted_median
from survidinri.cox_ipcw import build_survival_index, compute_ipcw_weights_indexed


def _simulate_weibull_cox_trial(
    n: int = 350,
    beta_base: float = 0.6,
    beta_marker: float = 1.1,
    cens_rate: float = 0.12,
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """Simulate right-censored survival data from a proportional hazards model."""
    rng = np.random.default_rng(seed)
    z_clin = rng.normal(0.0, 1.0, size=n)
    z_mark = rng.normal(0.0, 1.0, size=n)
    linpred = beta_base * z_clin + beta_marker * z_mark
    # Exponential baseline hazard lambda0 = 0.18
    u = rng.uniform(1e-8, 1.0, size=n)
    true_t = -np.log(u) / (0.18 * np.exp(linpred))
    cens_t = rng.exponential(scale=1.0 / max(cens_rate, 1e-6), size=n) + 0.2
    obs_time = np.minimum(true_t, cens_t)
    event = (true_t <= cens_t).astype(int)
    df = pd.DataFrame({"clin": z_clin, "marker": z_mark})
    t0 = 3.0
    return obs_time, event, df, t0


def test_ipcw_kaplan_meier_weight_conservation_with_ties() -> None:
    """Verify exact KM redistribute-to-the-right identity sum(xi*(w1+w0)) == sum(xi)."""
    time = np.array([1.0, 2.0, 2.0, 3.5, 3.5, 5.0, 6.0, 8.0])
    event = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    t0 = 4.0
    s_idx = build_survival_index(time, event, t0)

    # Unit weights
    unit = np.ones_like(time)
    w1, w0, g_t0 = compute_ipcw_weights_indexed(s_idx, unit, g_min=1e-12)
    assert np.isclose(np.sum(w1 + w0), float(time.size), atol=1e-12)
    assert 0.0 < g_t0 < 1.0

    # Arbitrary Exp(1) perturbation weights must also satisfy exact conservation!
    rng = np.random.default_rng(7)
    for _ in range(20):
        xi = rng.exponential(1.0, size=time.size)
        w1_p, w0_p, _ = compute_ipcw_weights_indexed(s_idx, xi, g_min=1e-12)
        assert np.isclose(np.sum(xi * (w1_p + w0_p)), np.sum(xi), rtol=1e-12)


def test_uncensored_limit_matches_exact_binary_pencina_idi_and_nri() -> None:
    """With zero censoring before t0, IPCW measures reduce identically to binary IDI/NRI."""
    rng = np.random.default_rng(42)
    n = 120
    y_bin = rng.binomial(1, 0.35, size=n)
    # Event times in (0.5, 2.5) if y_bin == 1, else > 5.0; all uncensored (event == 1)
    time = np.where(y_bin == 1, rng.uniform(0.5, 2.5, size=n), rng.uniform(5.5, 9.0, size=n))
    event = np.ones(n, dtype=int)
    t0 = 4.0

    p0 = np.clip(0.35 + rng.normal(0.0, 0.08, size=n), 0.05, 0.95)
    p1 = np.clip(p0 + np.where(y_bin == 1, 0.15, -0.10) + rng.normal(0.0, 0.05, size=n), 0.01, 0.99)

    res = idi_inf_predictions(
        time, event, p0, p1, t0=t0, npert=100, risk_thresholds=[0.30, 0.50], settings="default"
    )

    diff = p1 - p0
    ev = y_bin == 1
    nev = ~ev
    expected_idi = np.mean(diff[ev]) - np.mean(diff[nev])
    expected_nri_ev = np.mean((diff[ev] > 0).astype(float) - (diff[ev] < 0).astype(float))
    expected_nri_nev = np.mean((diff[nev] < 0).astype(float) - (diff[nev] > 0).astype(float))
    expected_cnri = expected_nri_ev + expected_nri_nev

    c0 = np.digitize(p0, [0.30, 0.50])
    c1 = np.digitize(p1, [0.30, 0.50])
    expected_cat = np.mean((c1[ev] > c0[ev]).astype(float) - (c1[ev] < c0[ev]).astype(float)) + np.mean(
        (c1[nev] < c0[nev]).astype(float) - (c1[nev] > c0[nev]).astype(float)
    )

    assert np.isclose(res.m1_idi.est, expected_idi, atol=1e-12)
    assert np.isclose(res.m2_nri.est, expected_cnri, atol=1e-12)
    assert np.isclose(res.half_cNRI.est, 0.5 * expected_cnri, atol=1e-12)
    assert res.cat_nri is not None
    assert np.isclose(res.cat_nri.est, expected_cat, atol=1e-12)


def test_idi_inf_strong_biomarker_yields_significant_improvement() -> None:
    """Adding a strong prognostic marker produces positive significant IDI, NRI, and MED."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=350, beta_marker=1.15, seed=2026)
    res = idi_inf(
        time=time,
        event=event,
        covs0=df[["clin"]],
        covs1=df[["clin", "marker"]],
        t0=t0,
        npert=200,
        risk_thresholds=[0.20, 0.50],
        seed=42,
        settings="default",
    )

    assert res.m1_idi.est > 0.10
    assert res.m1_idi.pval < 1e-4
    assert res.m1_idi.lower > 0.0

    assert res.m2_nri.est > 0.40
    assert res.m2_nri.pval < 1e-4
    assert res.m3_med.est > 0.05
    assert res.r2_idi.est > 0.10
    assert res.cat_nri is not None and res.cat_nri.est > 0.15


def test_idi_inf_null_noise_marker_is_nonsignificant() -> None:
    """Adding an uninformative noise covariate yields near-zero IDI/NRI with p > 0.05."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=320, beta_marker=0.0, seed=888)
    res = idi_inf(
        time=time,
        event=event,
        covs0=df[["clin"]],
        covs1=df[["clin", "marker"]],
        t0=t0,
        npert=200,
        seed=19,
        settings="default",
    )
    assert abs(res.m1_idi.est) < 0.03
    assert res.m1_idi.lower <= 0.0 <= res.m1_idi.upper
    assert res.m1_idi.pval > 0.05


def test_settings_default_vs_r_default_half_cnri_and_quantile_ci_parity() -> None:
    """Verify exact 2x relationship between default full cNRI and R_default Uno half-cNRI."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=250, beta_marker=0.9, seed=73)

    res_def = idi_inf(
        time, event, df[["clin"]], df[["clin", "marker"]], t0=t0, npert=200, seed=5, settings="default"
    )
    res_r = idi_inf(
        time, event, df[["clin"]], df[["clin", "marker"]], t0=t0, npert=200, seed=5, settings="R_default"
    )

    # With continuous untied failure times, Efron and Breslow coincide identically!
    assert np.allclose(res_def.risk0, res_r.risk0, atol=1e-10)
    assert np.allclose(res_def.risk1, res_r.risk1, atol=1e-10)
    assert np.isclose(res_def.m1_idi.est, res_r.m1_idi.est, atol=1e-10)
    # R_default M2 is on Uno et al.'s [-1, 1] half-cNRI scale; default is on [-2, 2]
    assert np.isclose(res_def.m2_nri.est, 2.0 * res_r.m2_nri.est, atol=1e-10)
    assert np.isclose(res_def.half_cNRI.est, res_r.m2_nri.est, atol=1e-10)
    assert np.isclose(res_def.m2_nri.se, 2.0 * res_r.m2_nri.se, atol=1e-10)
    assert res_r.m2_nri.lower < res_r.m2_nri.est < res_r.m2_nri.upper


def test_efron_vs_breslow_with_tied_survival_times() -> None:
    """Verify Efron (default) vs Breslow (R_default) tie handling on discretized times."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=260, beta_marker=1.0, seed=31)
    # Round times to 0.5-year intervals to induce heavy ties
    tied_time = np.maximum(0.5, np.round(time * 2.0) / 2.0)

    res_efron = idi_inf(
        tied_time, event, df[["clin"]], df[["clin", "marker"]], t0=t0, npert=100, settings="default"
    )
    res_breslow = idi_inf(
        tied_time, event, df[["clin"]], df[["clin", "marker"]], t0=t0, npert=100, settings="R_default"
    )
    assert res_efron.beta1 is not None and res_breslow.beta1 is not None
    # Under heavy ties, Breslow attenuates log-HRs slightly toward 0 relative to Efron
    assert not np.allclose(res_efron.beta1, res_breslow.beta1, atol=1e-5)
    assert np.all(np.abs(res_breslow.beta1) < np.abs(res_efron.beta1))
    assert np.allclose(res_efron.beta1, res_breslow.beta1, rtol=0.25)


def test_null_baseline_covs0_none_supported() -> None:
    """Verify covs0=None fits an intercept-only Nelson-Aalen baseline survival model."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=200, beta_marker=1.0, seed=12)
    res = idi_inf(time, event, covs0=None, covs1=df[["clin", "marker"]], t0=t0, npert=80)
    # Under null baseline model, all subjects share identical baseline predicted risk
    assert np.ptp(res.risk0) < 1e-12
    assert res.m1_idi.est > 0.10
    assert res.m1_idi.pval < 1e-3


def test_identical_predictions_yield_exact_zero_and_pval_one() -> None:
    """When risk0 == risk1, all reclassification differences must be identically 0."""
    time, event, _, t0 = _simulate_weibull_cox_trial(n=150, seed=9)
    rng = np.random.default_rng(9)
    p = rng.uniform(0.1, 0.8, size=150)
    res = idi_inf_predictions(
        time, event, p, p, t0=t0, npert=50, risk_thresholds=[0.3, 0.6], settings="R_default"
    )
    assert res.m1_idi.est == 0.0
    assert res.m2_nri.est == 0.0
    assert res.m3_med.est == 0.0
    assert res.cat_nri is not None and res.cat_nri.est == 0.0
    assert res.m1_idi.pval == 1.0
    assert res.m2_nri.pval == 1.0


def test_survidinri_oop_estimator_summary_and_dataframe() -> None:
    """Test SurvIDINRI class interface, summary formatting, and DataFrame export."""
    time, event, df, t0 = _simulate_weibull_cox_trial(n=180, beta_marker=0.95, seed=64)
    est = SurvIDINRI(t0=t0, npert=80, risk_thresholds=[0.25], seed=11, settings="default")
    res = est.fit(time, event, df[["clin"]], df[["clin", "marker"]])

    out_df = res.to_dataframe()
    assert "M1_IDI" in out_df["metric"].values
    assert "M2_NRI" in out_df["metric"].values
    assert "M3_MED" in out_df["metric"].values
    assert "cat_NRI" in out_df["metric"].values

    summary_txt = res.summary()
    assert "SurvIDINRI Summary" in summary_txt
    assert "M1_IDI" in summary_txt


def test_weighted_median_exact_interpolation() -> None:
    """Verify weighted_median handles uniform and skewed weights accurately."""
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.isclose(weighted_median(vals, np.ones(4)), 25.0)
    # Heavy weight on 40.0 shifts median to upper element
    assert weighted_median(vals, np.array([0.05, 0.05, 0.05, 0.85])) == 40.0


def test_input_validation_errors() -> None:
    """Verify clear ValueErrors on invalid t0 horizons, probabilities, and thresholds."""
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 1])
    p0 = np.array([0.2, 0.3, 0.4, 0.5])
    p1 = np.array([0.25, 0.35, 0.35, 0.55])

    # Horizon before all events
    with pytest.raises(ValueError, match="t0"):
        idi_inf_predictions(time, event, p0, p1, t0=0.2)
    # Horizon after maximum follow-up
    with pytest.raises(ValueError, match="t0"):
        idi_inf_predictions(time, event, p0, p1, t0=10.0)
    # Probabilities outside [0, 1]
    with pytest.raises(ValueError, match="risk"):
        idi_inf_predictions(time, event, p0, np.array([0.2, 1.2, 0.3, 0.4]), t0=2.5)
    # Invalid categorical threshold >= 1
    with pytest.raises(ValueError, match="risk_thresholds"):
        idi_inf_predictions(time, event, p0, p1, t0=2.5, risk_thresholds=[0.2, 1.0])


def test_invalid_settings_raises_value_error() -> None:
    """Every primary survidinri function/class must reject unknown settings strings."""
    time = np.array([1.0, 2.0, 3.0, 5.0])
    event = np.array([1, 0, 1, 0])
    cov = np.array([[0.1], [0.5], [-0.3], [0.8]])
    p = np.array([0.2, 0.3, 0.4, 0.5])

    with pytest.raises(ValueError, match="settings"):
        idi_inf(time, event, cov, cov, t0=2.5, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        idi_inf_predictions(time, event, p, p, t0=2.5, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        SurvIDINRI(t0=2.5, settings="wrong")  # type: ignore[arg-type]
