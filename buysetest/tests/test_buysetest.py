"""Comprehensive unit tests for the standalone buysetest package (BuyseTest & WR::pwreg)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from buysetest import (
    BuyseTest,
    Endpoint,
    buyse_test,
    fit_km_atoms,
    pwreg,
    score_pairs_tte_gehan,
    score_pairs_tte_peron,
)


def _simulate_prioritized_trial(
    m: int = 120,
    n: int = 120,
    hr_os: float = 0.55,
    hr_pfs: float = 0.60,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate a 2-arm prioritized oncology trial: OS -> PFS -> Grade>=3 Toxicity."""
    rng = np.random.default_rng(seed)
    arm = np.array([0] * n + [1] * m)
    total = m + n
    region = rng.choice(["NA", "EU", "APAC"], size=total, p=[0.4, 0.4, 0.2])
    age = rng.normal(62.0, 8.0, size=total)

    os_rate = 0.004 * np.where(arm == 1, hr_os, 1.0) * np.exp(0.02 * (age - 62.0))
    pfs_rate = 0.009 * np.where(arm == 1, hr_pfs, 1.0) * np.exp(0.02 * (age - 62.0))

    true_os = rng.exponential(1.0 / os_rate)
    true_pfs = np.minimum(true_os, rng.exponential(1.0 / pfs_rate))
    cens = rng.uniform(120.0, 420.0, size=total)

    os_time = np.minimum(true_os, cens)
    os_ev = (true_os <= cens).astype(int)
    pfs_time = np.minimum(true_pfs, cens)
    pfs_ev = (true_pfs <= cens).astype(int)
    tox = rng.binomial(1, np.where(arm == 1, 0.18, 0.32))

    return pd.DataFrame(
        {
            "id": np.arange(1, total + 1),
            "arm": arm,
            "region": region,
            "age": age,
            "os_time": os_time,
            "os_ev": os_ev,
            "pfs_time": pfs_time,
            "pfs_ev": pfs_ev,
            "tox": tox,
        }
    )


def test_peron_vs_gehan_exact_probabilities_and_km_tail_uninf() -> None:
    """Verify exact hand-calculable Peron survival integral and KM tail uninf conservation."""
    # Treatment arm: events at 4, 8, and right-censored at 10 -> S_T(4)=2/3, S_T(8)=1/3, tail=1/3 at t_max=10
    xt = np.array([4.0, 8.0, 10.0])
    dt = np.array([1, 1, 0])
    # Control arm: events at 2, 6, 12 -> S_C(2)=2/3, S_C(6)=1/3, S_C(12)=0, tail=0
    xc = np.array([2.0, 6.0, 12.0])
    dc = np.array([1, 1, 1])

    km_t = fit_km_atoms(xt, dt)
    km_c = fit_km_atoms(xc, dc)
    assert np.isclose(km_t.tail_prob, 1.0 / 3.0)
    assert np.isclose(km_c.tail_prob, 0.0)

    w_p, l_p, n_p, u_p = score_pairs_tte_peron(xt, dt, xc, dc, km_t, km_c, threshold=0.0)
    assert np.allclose(w_p + l_p + n_p + u_p, 1.0, atol=1e-12)

    # Check censored Treatment subject (xt=10, dt=0) vs Control events [2, 6, 12]:
    # vs xc=2: 10 >= 2 -> win = 1.0
    # vs xc=6: 10 >= 6 -> win = 1.0
    # vs xc=12: 10 < 12, and S_T(10) = 1/3 is purely tail mass > 10 vs 12 -> uninf = 1.0!
    assert np.isclose(w_p[2, 0], 1.0)
    assert np.isclose(w_p[2, 1], 1.0)
    assert np.isclose(u_p[2, 2], 1.0)

    # Now check a Treatment subject censored at x=5 (S_T(5)=2/3: jump 1/3 at 8, tail 1/3 > 10) vs xc=6:
    w_5, l_5, n_5, u_5 = score_pairs_tte_peron(
        np.array([5.0]), np.array([0]), np.array([6.0, 12.0]), np.array([1, 1]), km_t, km_c
    )
    # vs xc=6: both jump at 8 (>6) and tail (>10>6) win -> P(T > 6 | T > 5) = (2/3)/(2/3) = 1.0!
    assert np.isclose(w_5[0, 0], 1.0)
    # vs xc=12: jump at 8 (<12) is loss (prob (1/3)/(2/3) = 0.5), tail (>10 vs 12) is uninf (0.5)!
    assert np.isclose(l_5[0, 1], 0.5)
    assert np.isclose(u_5[0, 1], 0.5)

    # Under Gehan, (xt=5, dt=0) vs (xc=6, dc=1) is 100% uninformative
    wg, lg, ng, ug = score_pairs_tte_gehan(
        np.array([5.0]), np.array([0]), np.array([6.0]), np.array([1])
    )
    assert ug[0, 0] == 1.0 and wg[0, 0] == 0.0


def test_hierarchical_propagation_and_probability_conservation() -> None:
    """Verify prioritized 3-layer composite propagation and simplex sum == 1 at every layer."""
    df = _simulate_prioritized_trial(m=90, n=90, seed=11)
    eps = [
        Endpoint("os_time", kind="tte", status_col="os_ev", operator="higher", threshold=30.0),
        Endpoint("pfs_time", kind="tte", status_col="pfs_ev", operator="higher", threshold=15.0),
        Endpoint("tox", kind="binary", operator="lower"),
    ]
    res = buyse_test(df, treatment="arm", endpoints=eps, scoring_rule="Peron", settings="default")
    tot = res.favorable + res.unfavorable + res.neutral + res.uninf
    assert np.allclose(tot, 1.0, atol=1e-12)
    # Cumulative favorable and unfavorable probabilities must be monotonically non-decreasing
    assert np.all(np.diff(res.favorable) >= -1e-15)
    assert np.all(np.diff(res.unfavorable) >= -1e-15)
    # After the uncensored binary toxicity endpoint, uninf must drop to 0!
    assert np.isclose(res.uninf[-1], 0.0, atol=1e-12)
    assert res.net_benefit.est[-1] > 0.15
    assert res.win_ratio.est[-1] > 1.3
    assert res.net_benefit.pval[-1] < 0.01


def test_continuous_single_endpoint_matches_mann_whitney_u_statistic() -> None:
    """For an uncensored continuous endpoint, P_win + 0.5*P_neut equals exact Mann-Whitney AUC."""
    rng = np.random.default_rng(51)
    x_ctrl = rng.normal(0.0, 1.0, size=60)
    x_trt = rng.normal(0.65, 1.0, size=70)
    df = pd.DataFrame(
        {
            "trt": [0] * 60 + [1] * 70,
            "score": np.concatenate([x_ctrl, x_trt]),
        }
    )
    res = buyse_test(
        df, treatment="trt", endpoints=[Endpoint("score", kind="continuous")], settings="R_default"
    )
    u_stat, _ = mannwhitneyu(x_trt, x_ctrl, alternative="two-sided")
    auc = u_stat / (60.0 * 70.0)

    mw_buyse = res.favorable[0] + 0.5 * res.neutral[0]
    assert np.isclose(mw_buyse, auc, atol=1e-12)
    assert np.isclose(res.win_odds.est[0], auc / (1.0 - auc), atol=1e-12)


def test_settings_default_order1_vs_r_default_order2_h_decomposition() -> None:
    """Verify R_default order=2 H-decomposition variance >= default order=1 Hajek variance."""
    df = _simulate_prioritized_trial(m=50, n=50, seed=99)
    eps = [
        Endpoint("os_time", kind="tte", status_col="os_ev", threshold=20.0),
        Endpoint("tox", kind="binary", operator="lower"),
    ]
    res_def = buyse_test(df, treatment="arm", endpoints=eps, settings="default")
    res_r = buyse_test(df, treatment="arm", endpoints=eps, settings="R_default")

    assert res_def.h_order == 1
    assert res_r.h_order == 2
    assert np.allclose(res_def.net_benefit.est, res_r.net_benefit.est, atol=1e-12)
    assert np.allclose(res_def.win_ratio.est, res_r.win_ratio.est, atol=1e-12)
    # Second-order term S_11 / (m*n) is positive semi-definite -> order=2 SE >= order=1 SE!
    assert np.all(res_r.net_benefit.se >= res_def.net_benefit.se - 1e-15)
    assert np.all(res_r.win_ratio.se >= res_def.win_ratio.se - 1e-15)
    # Transformed CIs respect parameter boundaries
    assert np.all((res_r.net_benefit.lower > -1.0) & (res_r.net_benefit.upper < 1.0))
    assert np.all(res_r.win_ratio.lower > 0.0)


def test_strata_pooling_modes_buyse_cmh_equal() -> None:
    """Test stratified GPC across Buyse, CMH, and equal stratum pooling weights."""
    df = _simulate_prioritized_trial(m=120, n=120, seed=44)
    ep = Endpoint("os_time", kind="tte", status_col="os_ev", threshold=10.0)

    r_buyse = buyse_test(df, "arm", ep, strata="region", pool="Buyse", settings="default")
    r_cmh = buyse_test(df, "arm", ep, strata="region", pool="CMH", settings="default")
    r_eq = buyse_test(df, "arm", ep, strata="region", pool="equal", settings="default")

    assert r_buyse.n_strata == 3
    for r in (r_buyse, r_cmh, r_eq):
        assert r.win_ratio.est[0] > 1.2
        assert np.isfinite(r.win_ratio.se[0])


def test_bootstrap_and_permutation_resampling_inference() -> None:
    """Verify bootstrap and permutation inference modes in buyse_test."""
    df = _simulate_prioritized_trial(m=60, n=60, hr_os=0.45, seed=71)
    ep = Endpoint("os_time", kind="tte", status_col="os_ev")

    r_ustat = buyse_test(df, "arm", ep, method_inference="u-statistic", settings="R_default")
    r_boot = buyse_test(
        df, "arm", ep, method_inference="bootstrap", n_resampling=80, seed=12, settings="R_default"
    )
    r_perm = buyse_test(
        df, "arm", ep, method_inference="permutation", n_resampling=80, seed=12, settings="R_default"
    )

    assert np.isclose(r_ustat.net_benefit.se[0], r_boot.net_benefit.se[0], rtol=0.30)
    assert r_perm.net_benefit.pval[0] < 0.05


def test_smart_string_arm_labels_and_oop_buysetest_wrapper() -> None:
    """Verify 'placebo' vs 'drug' automatic control recognition and BuyseTest class."""
    df = _simulate_prioritized_trial(m=70, n=70, hr_os=0.50, seed=19)
    df["arm_label"] = np.where(df["arm"] == 1, "drug", "placebo")
    ep = Endpoint("os_time", kind="tte", status_col="os_ev")

    bt = BuyseTest(scoring_rule="Peron", settings="R_default")
    res = bt.fit(df, treatment="arm_label", endpoints=[ep])
    assert res.control_level == "placebo"
    assert res.treatment_level == "drug"
    assert res.win_ratio.est[0] > 1.3
    assert "BuyseTest Generalized Pairwise Comparisons" in res.summary()
    assert len(res.to_dataframe()) == 1


def test_correction_uninf_modes() -> None:
    """Verify correction_uninf=1 and 2 redistribute uninformative mass to 0."""
    df = _simulate_prioritized_trial(m=60, n=60, seed=303)
    ep = Endpoint("os_time", kind="tte", status_col="os_ev", threshold=25.0)

    r0 = buyse_test(df, "arm", ep, scoring_rule="Gehan", correction_uninf=0)
    r1 = buyse_test(df, "arm", ep, scoring_rule="Gehan", correction_uninf=1)
    r2 = buyse_test(df, "arm", ep, scoring_rule="Gehan", correction_uninf=2)

    assert r0.uninf[0] > 0.05
    assert np.isclose(r1.uninf[0], 0.0, atol=1e-12)
    assert np.isclose(r2.uninf[0], 0.0, atol=1e-12)
    assert np.isclose(r1.favorable[0] + r1.unfavorable[0] + r1.neutral[0], 1.0, atol=1e-12)


def test_pwreg_uncensored_binary_treatment_equals_exact_win_ratio() -> None:
    """With zero censoring and a single binary treatment covariate, exp(beta) == Win Ratio!"""
    rng = np.random.default_rng(17)
    n_c, n_t = 55, 55
    arm = np.array([0] * n_c + [1] * n_t)
    # Uncensored survival times (all event == 1)
    t_os = rng.exponential(scale=np.where(arm == 1, 22.0, 11.0))
    ev_os = np.ones(n_c + n_t, dtype=int)
    df = pd.DataFrame({"arm": arm, "os": t_os, "ev": ev_os})

    ep = Endpoint("os", kind="tte", status_col="ev")
    bt_res = buyse_test(df, treatment="arm", endpoints=[ep], scoring_rule="Gehan")
    pw_res = pwreg(df, treatment_col="arm", endpoints=[ep], settings="default")

    assert np.isclose(pw_res.win_ratio[0], bt_res.win_ratio.est[0], rtol=1e-6)
    assert np.isclose(pw_res.pval[0], bt_res.win_ratio.pval[0], rtol=0.15)


def test_pwreg_multivariable_composite_tte_recovery_default_and_r_default() -> None:
    """Verify Mao & Wang (2020) pwreg recovers treatment & covariate log-WRs under censoring."""
    df = _simulate_prioritized_trial(m=130, n=130, hr_os=0.50, hr_pfs=0.55, seed=512)
    eps = [
        Endpoint("os_time", kind="tte", status_col="os_ev"),
        Endpoint("pfs_time", kind="tte", status_col="pfs_ev"),
    ]
    fit_def = pwreg(
        df, treatment_col="arm", covariates=["age"], endpoints=eps, settings="default"
    )
    fit_r = pwreg(
        df, treatment_col="arm", covariates=["age"], endpoints=eps, settings="R_default"
    )

    # Under PH with HR_OS ~ 0.50, treatment Win Ratio ~ 1 / HR ~ 1.8 - 2.0 > 1 (log WR > 0),
    # while older age increases hazard -> conditional Win Ratio < 1 (log WR < 0)
    assert fit_def.coef_names == ["arm", "age"]
    assert fit_def.win_ratio[0] > 1.4 and fit_def.pval[0] < 1e-3
    assert fit_def.win_ratio[1] < 1.0
    assert fit_def.layer_pair_counts[0] > 0 and fit_def.layer_pair_counts[1] > 0

    assert np.allclose(fit_def.coef, fit_r.coef, rtol=0.05)
    assert np.all(fit_r.se > 0)
    assert "Proportional Win-Fractions Regression" in fit_r.summary()


def test_pwreg_long_multi_record_format_parity() -> None:
    """Verify long multi-row-per-ID format (status 1=death, 2=nonfatal, 0=cens) matches wide format."""
    df = _simulate_prioritized_trial(m=60, n=60, hr_os=0.55, hr_pfs=0.60, seed=81)
    long_rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        sid = int(r["id"])
        # Non-fatal progression if pfs_ev == 1 and pfs_time < os_time
        if int(r["pfs_ev"]) == 1 and float(r["pfs_time"]) < float(r["os_time"]):
            long_rows.append(
                {"id": sid, "time": float(r["pfs_time"]), "status": 2, "arm": r["arm"], "age": r["age"]}
            )
        # Terminal death (status=1) or censoring (status=0)
        long_rows.append(
            {
                "id": sid,
                "time": float(r["os_time"]),
                "status": 1 if int(r["os_ev"]) == 1 else 0,
                "arm": r["arm"],
                "age": r["age"],
            }
        )
    long_df = pd.DataFrame(long_rows)

    fit_long = pwreg(
        long_df,
        id_col="id",
        time_col="time",
        status_col="status",
        treatment_col="arm",
        covariates=["age"],
        settings="R_default",
    )
    eps_wide = [
        Endpoint("os_time", kind="tte", status_col="os_ev"),
        Endpoint("pfs_time", kind="tte", status_col="pfs_ev"),
    ]
    fit_wide = pwreg(
        df, treatment_col="arm", covariates=["age"], endpoints=eps_wide, settings="R_default"
    )
    assert np.allclose(fit_long.coef, fit_wide.coef, atol=1e-10)
    assert np.allclose(fit_long.se, fit_wide.se, atol=1e-10)


def test_endpoint_validation_errors() -> None:
    """Verify Endpoint dataclass validates kind, status_col, operator, and threshold."""
    with pytest.raises(ValueError, match="status_col"):
        Endpoint("os_time", kind="tte", status_col=None)
    with pytest.raises(ValueError, match="threshold"):
        Endpoint("score", kind="continuous", threshold=-1.0)
    with pytest.raises(ValueError, match="operator"):
        Endpoint("score", kind="continuous", operator="bigger")  # type: ignore[arg-type]


def test_invalid_settings_raises_value_error() -> None:
    """Every primary buysetest function/class must reject unknown settings strings."""
    df = pd.DataFrame({"arm": [0, 0, 1, 1], "t": [2.0, 4.0, 5.0, 8.0], "ev": [1, 1, 1, 0]})
    ep = Endpoint("t", kind="tte", status_col="ev")

    with pytest.raises(ValueError, match="settings"):
        buyse_test(df, "arm", ep, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        BuyseTest(settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        pwreg(df, treatment_col="arm", endpoints=[ep], settings="wrong")  # type: ignore[arg-type]


def test_buysetest_result_plot_stacked_bars_and_annotations() -> None:
    df = _simulate_prioritized_trial(m=80, n=80, seed=313)
    eps = [
        Endpoint("os_time", kind="tte", status_col="os_ev", operator="higher", threshold=25.0),
        Endpoint("pfs_time", kind="tte", status_col="pfs_ev", operator="higher", threshold=10.0),
        Endpoint("tox", kind="binary", operator="lower"),
    ]
    res = buyse_test(df, treatment="arm", endpoints=eps, scoring_rule="Peron")

    fig = res.plot(figsize=(8, 4))
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.patches) == 4 * len(eps)
        assert any("WR=" in txt.get_text() and "NB=" in txt.get_text() for txt in ax.texts)
    finally:
        plt.close(fig)


def test_plot_buyse_test_incremental_and_custom_ax() -> None:
    from buysetest import plot_buyse_test

    df = _simulate_prioritized_trial(m=60, n=60, seed=414)
    eps = [
        Endpoint("os_time", kind="tte", status_col="os_ev", operator="higher"),
        Endpoint("tox", kind="binary", operator="lower"),
    ]
    est = BuyseTest(scoring_rule="Gehan")
    with pytest.raises(RuntimeError, match="fit"):
        est.plot()

    res = est.fit(df, treatment="arm", endpoints=eps)
    fig0 = est.plot()
    import matplotlib.pyplot as plt

    plt.close(fig0)

    fig_ext, ax_ext = plt.subplots(figsize=(7.5, 3.8))
    try:
        out_fig = plot_buyse_test(
            res, ax=ax_ext, cumulative=False, title="Incremental GPC Breakdown"
        )
        assert out_fig is fig_ext
        assert ax_ext.get_title() == "Incremental GPC Breakdown"
        assert len(ax_ext.patches) == 4 * len(eps)
    finally:
        plt.close(fig_ext)

