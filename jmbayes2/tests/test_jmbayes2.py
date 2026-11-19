"""Unit tests for jmbayes2 (Bayesian joint longitudinal-survival & competing-risks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import jmbayes2
from jmbayes2 import (
    BSplineHazardBasis,
    GK15_NODES,
    GK15_WEIGHTS,
    JMBayes2,
    fit_lme,
    gauss_kronrod_15,
    jm,
    predict_dynamic,
)


def _simulate_joint_data(
    seed: int = 42,
    n_subjects: int = 120,
    alpha_val: float = 0.65,
    alpha_slope: float = 0.0,
    gamma_trt: float = -0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Simulate joint longitudinal + right-censored survival dataset via exact inverse-CDF."""
    rng = np.random.default_rng(seed)
    d_true = np.array([[0.75, 0.08], [0.08, 0.09]], dtype=np.float64)
    b_true = rng.multivariate_normal([0.0, 0.0], d_true, size=n_subjects)
    trt = rng.integers(0, 2, size=n_subjects)

    beta0, beta1 = 1.5, 0.35
    base_h0 = 0.045

    # Hazard h_i(t) = h0 * exp(gamma*trt + alpha_val*(beta0+b0 + (beta1+b1)*t) + alpha_slope*(beta1+b1))
    # = A_i * exp(B_i * t) -> exact Gompertz inverse-CDF!
    icpt_i = beta0 + b_true[:, 0]
    slp_i = beta1 + b_true[:, 1]
    log_a = np.log(base_h0) + gamma_trt * trt + alpha_val * icpt_i + alpha_slope * slp_i
    a_i = np.exp(log_a)
    b_rate = alpha_val * slp_i

    u = rng.uniform(1e-6, 1.0 - 1e-6, size=n_subjects)
    neg_log_u = -np.log(u)
    t_true = np.where(
        np.abs(b_rate) > 1e-5,
        np.log(np.maximum(1.0 + (b_rate * neg_log_u) / a_i, 1e-6)) / np.where(np.abs(b_rate) > 1e-5, b_rate, 1.0),
        neg_log_u / a_i,
    )
    t_true = np.where(t_true > 0.05, t_true, 0.05 + neg_log_u / (a_i + 0.1))
    c_time = rng.uniform(2.5, 7.5, size=n_subjects)
    obs_time = np.minimum(t_true, c_time)
    event = (t_true <= c_time).astype(int)

    surv_df = pd.DataFrame(
        {
            "id": np.arange(n_subjects),
            "time": obs_time,
            "event": event,
            "trt": trt.astype(float),
        }
    )

    long_rows: list[dict[str, float | int]] = []
    visit_grid = np.array([0.0, 0.5, 1.0, 1.8, 2.8, 4.0, 5.5])
    for i in range(n_subjects):
        v_i = visit_grid[visit_grid < obs_time[i]]
        if v_i.size == 0:
            v_i = np.array([0.0])
        y_i = icpt_i[i] + slp_i[i] * v_i + rng.normal(0.0, 0.25, size=v_i.size)
        for tv, yv in zip(v_i, y_i):
            long_rows.append({"id": i, "obstime": float(tv), "y": float(yv)})

    return surv_df, pd.DataFrame(long_rows), b_true


def test_gk15_exact_polynomial_and_exponential_quadrature() -> None:
    """Verify 15-node Gauss-Kronrod integrates polynomials up to degree 22 to ~1e-15."""
    assert GK15_NODES.shape == (15,)
    assert GK15_WEIGHTS.shape == (15,)
    assert float(np.sum(GK15_WEIGHTS)) == pytest.approx(2.0, abs=1e-14)

    for deg in range(0, 23):
        approx = float(np.sum(GK15_WEIGHTS * (GK15_NODES ** deg)))
        exact = 0.0 if (deg % 2 == 1) else (2.0 / (deg + 1))
        assert approx == pytest.approx(exact, abs=1e-13)

    nodes, weights = gauss_kronrod_15(np.array([0.0, 1.0]), np.array([3.0, 4.0]))
    approx_exp = np.sum(weights * np.exp(0.5 * nodes), axis=1)
    exact_exp = (np.exp(0.5 * np.array([3.0, 4.0])) - np.exp(0.5 * np.array([0.0, 1.0]))) / 0.5
    np.testing.assert_allclose(approx_exp, exact_exp, rtol=1e-13)


def test_bspline_hazard_basis_partition_of_unity_and_penalty() -> None:
    """Verify B-spline basis sums to 1 everywhere and D2 penalty annihilates linear trends."""
    times = np.linspace(0.1, 8.0, 80)
    ev_mask = np.ones(80, dtype=bool)
    for deg in (2, 3):
        basis = BSplineHazardBasis.from_times(times, ev_mask, n_knots=5, degree=deg)
        b_mat = basis.evaluate(np.linspace(0.0, 9.0, 30))
        np.testing.assert_allclose(np.sum(b_mat, axis=1), 1.0, atol=1e-12)

        ones = np.ones(basis.n_basis)
        lin = np.arange(basis.n_basis, dtype=np.float64)
        np.testing.assert_allclose(basis.penalty_matrix @ ones, 0.0, atol=1e-12)
        np.testing.assert_allclose(basis.penalty_matrix @ lin, 0.0, atol=1e-11)


def test_fit_lme_reml_parameter_and_ranef_recovery() -> None:
    """Verify REML LME recovers fixed effects, 2x2 D, sigma^2, and Empirical Bayes b_i."""
    _, long_df, b_true = _simulate_joint_data(seed=10, n_subjects=160, alpha_val=0.0)
    lme = fit_lme(long_df, random_slope=True, settings="default")

    assert lme.converged
    assert lme.D.shape == (2, 2)
    realized_icpt = 1.5 + float(np.mean(b_true[:, 0]))
    realized_slope = 0.35 + float(np.mean(b_true[:, 1]))
    assert abs(float(lme.fixed_effects["intercept"]) - realized_icpt) < 0.06
    assert abs(float(lme.fixed_effects["obstime"]) - realized_slope) < 0.04
    assert abs(lme.sigma - 0.25) < 0.04
    assert abs(lme.D[0, 0] - 0.75) < 0.22

    # Empirical Bayes random intercepts should correlate strongly with true b_i0
    corr_b0 = np.corrcoef(lme.ranef["b0"].to_numpy(), b_true[:, 0])[0, 1]
    assert corr_b0 > 0.90
    assert "Linear Mixed-Effects Fit" in lme.summary()
    assert lme.to_dataframe().shape[0] == 2


def test_fit_lme_random_intercept_only_and_fixed_covariates() -> None:
    """Verify 1x1 D when random_slope=False and extra baseline fixed_cols."""
    rng = np.random.default_rng(21)
    rows = []
    for i in range(60):
        bi = rng.normal(0.0, 0.6)
        xi = float(rng.normal(0.0, 1.0))
        for t in (0.0, 1.0, 2.0, 3.0):
            rows.append(
                {
                    "id": i,
                    "obstime": t,
                    "x_base": xi,
                    "y": 1.0 + 0.5 * t - 0.8 * xi + bi + rng.normal(0.0, 0.2),
                }
            )
    df = pd.DataFrame(rows)
    lme = fit_lme(df, fixed_cols=["x_base"], random_slope=False, settings="R_default")
    assert lme.D.shape == (1, 1)
    assert lme.ranef.shape == (60, 1)
    assert abs(float(lme.fixed_effects["x_base"]) - (-0.8)) < 0.12


def test_jm_single_event_recovers_association_and_diagnostics() -> None:
    """Verify Bayesian joint model recovers positive alpha_val, trt HR, DIC, WAIC, and Rhat."""
    surv_df, long_df, _ = _simulate_joint_data(
        seed=42, n_subjects=130, alpha_val=0.70, gamma_trt=-0.55
    )
    fit = jm(
        surv_df,
        long_df,
        surv_covs=["trt"],
        functional_forms=("value",),
        n_iter=420,
        burn_in=140,
        seed=42,
        settings="default",
    )

    alpha_mean = float(fit.assoc_df.loc[0, "mean"])
    alpha_lo = float(fit.assoc_df.loc[0, "ci_lower"])
    assert 0.30 < alpha_mean < 1.15
    assert alpha_lo > 0.15

    trt_mean = float(fit.surv_df_summary.loc[0, "mean"])
    assert trt_mean < -0.10

    assert np.isfinite(fit.dic)
    assert np.isfinite(fit.waic)
    assert fit.posterior_draws["alpha"].shape == (280, 1, 1)
    assert all(rh < 1.35 for rh in fit.rhat.values())
    assert "Bayesian Joint Model" in fit.summary()
    assert fit.to_dataframe().shape[0] >= 5


def test_jm_multiform_value_slope_area() -> None:
    """Verify joint model supports simultaneous ('value', 'slope', 'area') functional forms."""
    surv_df, long_df, _ = _simulate_joint_data(
        seed=77, n_subjects=95, alpha_val=0.45, alpha_slope=0.60
    )
    fit = jm(
        surv_df,
        long_df,
        functional_forms=("value", "slope", "area"),
        n_iter=320,
        burn_in=100,
        seed=77,
        settings="default",
    )
    assert list(fit.assoc_df["form"]) == ["value", "slope", "area"]
    assert fit.posterior_draws["alpha"].shape == (220, 1, 3)


def test_jm_competing_risks_cause_specific_associations() -> None:
    """Verify cause-specific competing risks recovers opposite-direction biomarker links."""
    rng = np.random.default_rng(108)
    n = 140
    b0 = rng.normal(0.0, 0.9, size=n)
    b1 = rng.normal(0.0, 0.15, size=n)
    # Cause 1 driven by high biomarker (+0.75 * b0); Cause 2 driven by low biomarker (-0.70 * b0)
    t1 = rng.exponential(scale=np.exp(1.7 - 0.75 * b0))
    t2 = rng.exponential(scale=np.exp(1.8 + 0.70 * b0))
    c = rng.uniform(2.5, 6.5, size=n)

    obs_t = np.minimum(np.minimum(t1, t2), c)
    event = np.zeros(n, dtype=int)
    ev1 = (t1 <= t2) & (t1 <= c)
    ev2 = (t2 < t1) & (t2 <= c)
    event[ev1] = 1
    event[ev2] = 2

    surv_df = pd.DataFrame({"id": np.arange(n), "time": obs_t, "event": event})
    long_rows = []
    for i in range(n):
        vis = np.linspace(0.0, max(obs_t[i] * 0.85, 0.05), 4)
        yi = 1.0 + 0.2 * vis + b0[i] + b1[i] * vis + rng.normal(0.0, 0.2, size=4)
        for tv, yv in zip(vis, yi):
            long_rows.append({"id": i, "obstime": float(tv), "y": float(yv)})
    long_df = pd.DataFrame(long_rows)

    fit_cr = jm(
        surv_df,
        long_df,
        competing_risks=True,
        functional_forms=("value",),
        n_iter=360,
        burn_in=120,
        seed=108,
        settings="default",
    )
    assert fit_cr.spec.causes == (1, 2)
    assert fit_cr.assoc_df.shape[0] == 2
    alpha_c1 = float(fit_cr.assoc_df.loc[fit_cr.assoc_df["cause"] == 1, "mean"].iloc[0])
    alpha_c2 = float(fit_cr.assoc_df.loc[fit_cr.assoc_df["cause"] == 2, "mean"].iloc[0])
    assert alpha_c1 > 0.25
    assert alpha_c2 < -0.25


def test_predict_dynamic_biomarker_stratification_and_monotonicity() -> None:
    """High-biomarker trajectory yields lower dynamic survival than low-biomarker trajectory."""
    surv_df, long_df, _ = _simulate_joint_data(seed=42, n_subjects=110, alpha_val=0.75)
    fit = jm(surv_df, long_df, surv_covs=["trt"], n_iter=320, burn_in=100, seed=42)

    low_pat = pd.DataFrame(
        {"id": [901, 901, 901], "obstime": [0.0, 0.8, 1.6], "y": [0.2, 0.3, 0.4]}
    )
    high_pat = pd.DataFrame(
        {"id": [902, 902, 902], "obstime": [0.0, 0.8, 1.6], "y": [3.8, 4.3, 4.9]}
    )
    horizons = [2.5, 3.5, 5.0]

    pred_low = predict_dynamic(
        fit, low_pat, t_landmark=1.8, horizons=horizons, surv_covs_row={"trt": 0.0}, n_mcmc=120, seed=11
    )
    pred_high = predict_dynamic(
        fit, high_pat, t_landmark=1.8, horizons=horizons, surv_covs_row={"trt": 0.0}, n_mcmc=120, seed=11
    )

    # Survival monotonically decreases with horizon
    assert np.all(np.diff(pred_low["surv_prob"].to_numpy()) < 0.0)
    assert np.all(np.diff(pred_high["surv_prob"].to_numpy()) < 0.0)

    # High biomarker has markedly worse conditional survival than low biomarker
    assert float(pred_high.iloc[-1]["surv_prob"]) < float(pred_low.iloc[-1]["surv_prob"]) - 0.15

    # Real-time trajectory update: if patient 901 subsequently spikes at t=2.2, survival at u=5.0 drops!
    spiking_pat = pd.concat(
        [low_pat, pd.DataFrame({"id": [901], "obstime": [2.2], "y": [5.5]})],
        ignore_index=True,
    )
    pred_after_spike = predict_dynamic(
        fit, spiking_pat, t_landmark=2.3, horizons=[5.0], surv_covs_row={"trt": 0.0}, n_mcmc=120, seed=11
    )
    pred_stable = predict_dynamic(
        fit,
        pd.concat([low_pat, pd.DataFrame({"id": [901], "obstime": [2.2], "y": [0.45]})], ignore_index=True),
        t_landmark=2.3,
        horizons=[5.0],
        surv_covs_row={"trt": 0.0},
        n_mcmc=120,
        seed=11,
    )
    assert float(pred_after_spike.iloc[0]["surv_prob"]) < float(pred_stable.iloc[0]["surv_prob"]) - 0.10


def test_predict_dynamic_strict_landmark_filtering_no_future_leakage() -> None:
    """Observations strictly after t_landmark must never leak into landmark predictions."""
    surv_df, long_df, _ = _simulate_joint_data(seed=51, n_subjects=80, alpha_val=0.6)
    fit = jm(surv_df, long_df, n_iter=240, burn_in=80, seed=51)

    hist_clean = pd.DataFrame({"id": [1, 1], "obstime": [0.0, 1.0], "y": [1.5, 1.8]})
    hist_with_future = pd.DataFrame(
        {"id": [1, 1, 1], "obstime": [0.0, 1.0, 2.5], "y": [1.5, 1.8, 99.0]}
    )

    p1 = predict_dynamic(fit, hist_clean, t_landmark=1.2, horizons=[2.5, 4.0], n_mcmc=80, seed=7)
    p2 = predict_dynamic(fit, hist_with_future, t_landmark=1.2, horizons=[2.5, 4.0], n_mcmc=80, seed=7)

    assert int(p2.iloc[0]["n_obs_used"]) == 2
    np.testing.assert_allclose(p1["surv_prob"].to_numpy(), p2["surv_prob"].to_numpy(), atol=1e-14)


def test_predict_dynamic_competing_risks_conservation_identity() -> None:
    """Verify S(u | t_L) + sum_k CIF_k(u | t_L) == 1 across horizons under GK15 nested quadrature."""
    rng = np.random.default_rng(64)
    n = 90
    b0 = rng.normal(0.0, 0.7, size=n)
    t1 = rng.exponential(scale=np.exp(1.6 - 0.5 * b0))
    t2 = rng.exponential(scale=np.exp(1.7 + 0.4 * b0))
    c = rng.uniform(2.5, 6.0, size=n)
    obs_t = np.minimum(np.minimum(t1, t2), c)
    event = np.where((t1 <= t2) & (t1 <= c), 1, np.where((t2 < t1) & (t2 <= c), 2, 0))

    surv_df = pd.DataFrame({"id": np.arange(n), "time": obs_t, "event": event})
    long_rows = [
        {"id": i, "obstime": float(tv), "y": float(1.0 + b0[i] + rng.normal(0.0, 0.2))}
        for i in range(n)
        for tv in np.linspace(0.0, max(obs_t[i] * 0.8, 0.05), 3)
    ]
    fit_cr = jm(
        surv_df, pd.DataFrame(long_rows), competing_risks=True, random_slope=False, n_iter=240, burn_in=80, seed=64
    )

    pat = pd.DataFrame({"id": [0, 0], "obstime": [0.0, 0.8], "y": [1.4, 1.6]})
    pred_cr = predict_dynamic(fit_cr, pat, t_landmark=1.0, horizons=[2.0, 3.5], n_mcmc=60, seed=9, settings="R_default")

    for h_val, grp in pred_cr.groupby("horizon"):
        cif_sum = float(grp["cif"].sum())
        s_prob = float(grp["surv_prob"].iloc[0])
        assert (s_prob + cif_sum) == pytest.approx(1.0, abs=1e-4)
        assert float(grp["event_prob"].iloc[0]) == pytest.approx(cif_sum, abs=1e-4)


def test_settings_default_vs_r_default_differences() -> None:
    """Verify spline degree, MCMC tail P, and event_prob differences between default and R_default."""
    surv_df, long_df, _ = _simulate_joint_data(seed=42, n_subjects=90, alpha_val=0.80)
    fit_def = jm(surv_df, long_df, n_iter=240, burn_in=80, seed=42, settings="default")
    fit_r = jm(surv_df, long_df, n_iter=240, burn_in=80, seed=42, settings="R_default")

    # Cubic (degree=3) vs Quadratic (degree=2) B-spline log-baseline hazard
    assert fit_def.spec.basis.degree == 3
    assert fit_r.spec.basis.degree == 2

    # Strictly positive parameter (sigma) has P == 0.0 in R_default vs >= 1/(S+1) in default
    p_sig_r = float(fit_r.long_df_summary.loc[fit_r.long_df_summary["term"] == "sigma", "pval"].iloc[0])
    p_sig_def = float(fit_def.long_df_summary.loc[fit_def.long_df_summary["term"] == "sigma", "pval"].iloc[0])
    assert p_sig_r == 0.0
    assert p_sig_def > 0.0

    # predict_dynamic returns event_prob = 1 - surv_prob in R_default
    pat = long_df[long_df["id"] == 0]
    p_def = predict_dynamic(fit_def, pat, t_landmark=1.0, horizons=[2.5], n_mcmc=50, settings="default")
    p_r = predict_dynamic(fit_r, pat, t_landmark=1.0, horizons=[2.5], n_mcmc=50, settings="R_default")
    assert "event_prob" not in p_def.columns
    assert "event_prob" in p_r.columns
    assert float(p_r.iloc[0]["event_prob"] + p_r.iloc[0]["surv_prob"]) == pytest.approx(1.0, abs=1e-12)


def test_jmbayes2_oop_estimator_and_string_ids() -> None:
    """Test JMBayes2 class API, string subject IDs, and long_covs merged from surv_df."""
    surv_df, long_df, _ = _simulate_joint_data(seed=19, n_subjects=75, alpha_val=0.55)
    surv_df["id"] = [f"PT_{i:03d}" for i in surv_df["id"]]
    long_df["id"] = [f"PT_{i:03d}" for i in long_df["id"]]

    est = JMBayes2(
        random_slope=True,
        functional_forms="value",
        n_iter=220,
        burn_in=70,
        seed=19,
        settings="R_default",
    )
    with pytest.raises(RuntimeError):
        est.summary()

    # Pass trt in long_covs even though trt lives in surv_df!
    est.fit(surv_df, long_df, surv_covs=["trt"], long_covs=["trt"])
    assert "Bayesian Joint Model" in est.summary()
    assert est.to_dataframe().shape[0] >= 6

    pat_df = long_df[long_df["id"].isin(["PT_000", "PT_001"])].merge(
        surv_df[["id", "trt"]], on="id", how="left"
    )
    preds = est.predict_dynamic(pat_df, t_landmark=0.8, horizons=[2.0, 3.0], n_mcmc=50)
    assert preds.shape[0] == 4
    assert set(preds["id"]) == {"PT_000", "PT_001"}


def test_settings_validation_every_entrypoint() -> None:
    """Every public function and class must raise ValueError on unknown settings."""
    surv_df, long_df, _ = _simulate_joint_data(seed=3, n_subjects=30)
    fit = jm(surv_df, long_df, n_iter=100, burn_in=40, seed=3)

    with pytest.raises(ValueError, match="settings"):
        fit_lme(long_df, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        jm(surv_df, long_df, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        predict_dynamic(fit, long_df[long_df["id"] == 0], t_landmark=0.5, horizons=[1.5], settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        JMBayes2(settings="invalid")  # type: ignore[arg-type]


def test_input_validation_errors() -> None:
    """Verify clear ValueErrors on invalid functional_forms, horizons, IDs, and MCMC iterations."""
    surv_df, long_df, _ = _simulate_joint_data(seed=5, n_subjects=30)
    fit = jm(surv_df, long_df, n_iter=90, burn_in=40, seed=5)

    with pytest.raises(ValueError, match="functional form"):
        jm(surv_df, long_df, functional_forms=("invalid_form",))
    with pytest.raises(ValueError, match="n_iter"):
        jm(surv_df, long_df, n_iter=50, burn_in=50)
    with pytest.raises(ValueError, match="Subject IDs"):
        jm(surv_df.iloc[:-1], long_df)
    with pytest.raises(ValueError, match="strictly greater"):
        predict_dynamic(fit, long_df[long_df["id"] == 0], t_landmark=2.0, horizons=[1.5, 3.0])
    assert hasattr(jmbayes2, "__version__")


def test_predict_dynamic_at_baseline_landmark_zero_and_no_id_col() -> None:
    """Verify predict_dynamic works at t_landmark=0.0 and when newdata_long_df omits id_col."""
    surv_df, long_df, _ = _simulate_joint_data(seed=91, n_subjects=65, alpha_val=0.60)
    fit = jm(surv_df, long_df, n_iter=180, burn_in=60, seed=91)

    single_no_id = pd.DataFrame({"obstime": [0.0], "y": [2.2]})
    pred = predict_dynamic(fit, single_no_id, t_landmark=0.0, horizons=[1.0, 2.5], n_mcmc=60, seed=4)
    assert pred.shape[0] == 2
    assert int(pred.iloc[0]["n_obs_used"]) == 1
    assert 0.0 < float(pred.iloc[1]["surv_prob"]) < float(pred.iloc[0]["surv_prob"]) < 1.0


def test_jm_area_only_functional_form_cumulative_exposure() -> None:
    """Verify area-only cumulative biomarker exposure functional form fits and predicts cleanly."""
    surv_df, long_df, _ = _simulate_joint_data(seed=133, n_subjects=75, alpha_val=0.55)
    fit_area = jm(
        surv_df,
        long_df,
        functional_forms=("area",),
        n_iter=200,
        burn_in=70,
        seed=133,
        settings="R_default",
    )
    assert list(fit_area.assoc_df["form"]) == ["area"]
    assert float(fit_area.assoc_df.loc[0, "mean"]) > 0.05

    pat_hi = pd.DataFrame({"id": [1, 1], "obstime": [0.0, 1.5], "y": [4.0, 4.5]})
    pat_lo = pd.DataFrame({"id": [2, 2], "obstime": [0.0, 1.5], "y": [0.1, 0.2]})
    pred = predict_dynamic(
        fit_area,
        pd.concat([pat_hi, pat_lo], ignore_index=True),
        t_landmark=1.6,
        horizons=[3.5],
        n_mcmc=60,
        seed=12,
        settings="R_default",
    )
    assert float(pred.loc[pred["id"] == 1, "event_prob"].iloc[0]) > float(
        pred.loc[pred["id"] == 2, "event_prob"].iloc[0]
    )
