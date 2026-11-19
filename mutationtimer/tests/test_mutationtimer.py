"""
Comprehensive deterministic test suite for `mutationtimer` covering analytical
expected-VAF identities, male X/Y haploid germline scaling, Gerstung et al.
(Nature 2020) `pi_to_time` rate equations, Beta-Binomial EM planted gain-time
recovery, 4-class PCAWG mutation timing classification, subclonal cluster
resolution, Whole-Genome Doubling (`WGD`) synchronicity, and `settings` parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutationtimer import (
    MutationTimeR,
    MutationTimerConfig,
    MutationTimeResult,
    WGDTimingResult,
    beta_binomial_logpmf,
    estimate_wgd_timing,
    expected_vaf,
    mutation_time,
    pi_to_all_times,
    pi_to_time,
    time_cn_gains,
)


def _simulate_timed_segment_variants(
    chrom: str,
    start: int,
    end: int,
    major_cn: int,
    minor_cn: int,
    purity: float,
    gain_t: float,
    n_clonal: int,
    n_subclonal: int = 0,
    subclonal_phi: float = 0.40,
    depth: int = 110,
    normal_cn: int = 2,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate somatic variant read counts from the exact evolutionary lineage model
    on a single copy-number segment (`1:1`, `2:1`, `2:0`, `2:2`, `3:1`, `3:0`).
    """
    rng = np.random.default_rng(seed)
    total_n = n_clonal + n_subclonal
    positions = np.sort(
        rng.choice(np.arange(start + 100, end - 100, 10, dtype=np.int64), size=total_n, replace=False)
    )

    # Determine expected clonal multiplicity proportions from gain_t
    if major_cn == 1:
        pi_c = np.array([1.0])
    elif major_cn == 2 and minor_cn == 1:
        mu = np.array([3.0 - 2.0 * gain_t, gain_t])
        pi_c = mu / np.sum(mu)
    elif major_cn == 2 and minor_cn == 0:
        mu = np.array([2.0 * (1.0 - gain_t), gain_t])
        pi_c = mu / np.sum(mu)
    elif major_cn == 2 and minor_cn == 2:
        mu = np.array([4.0 * (1.0 - gain_t), 2.0 * gain_t])
        pi_c = mu / np.sum(mu)
    else:
        raise ValueError("Unsupported state in helper")

    mults = rng.choice(np.arange(1, major_cn + 1), size=n_clonal, p=pi_c)
    phis = np.ones(n_clonal, dtype=np.float64)
    if n_subclonal > 0:
        mults = np.concatenate([mults, np.ones(n_subclonal, dtype=np.int64)])
        phis = np.concatenate([phis, np.full(n_subclonal, subclonal_phi, dtype=np.float64)])

    vafs = expected_vaf(
        multiplicity=mults,
        phi=phis,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
    )
    dp = rng.poisson(depth, size=total_n)
    dp = np.maximum(dp, 40)
    alt = rng.binomial(dp, vafs)
    ref = dp - alt

    return pd.DataFrame(
        {
            "chrom": [chrom] * total_n,
            "pos": positions,
            "alt_count": alt,
            "ref_count": ref,
            "true_m": mults,
            "true_phi": phis,
        }
    )


def test_settings_validation_raises_value_error() -> None:
    segs = pd.DataFrame(
        {"chrom": ["chr1"], "start": [1], "end": [10_000_000], "major_cn": [2], "minor_cn": [1]}
    )
    vars_df = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [5000], "alt_count": [25], "total_depth": [80]}
    )

    with pytest.raises(ValueError, match="Unknown settings"):
        mutation_time(vars_df, segs, purity=0.8, settings="invalid_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        MutationTimeR(purity=0.8, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        pi_to_time(2, 1, [0.5, 0.5], settings="wrong")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        time_cn_gains(segs, {0: [0.5, 0.5]}, settings="wrong")  # type: ignore[arg-type]

    segs_timed = segs.assign(T=[0.5])
    with pytest.raises(ValueError, match="Unknown settings"):
        estimate_wgd_timing(segs_timed, settings="wrong")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        MutationTimerConfig(purity=0.8, settings="wrong")  # type: ignore[arg-type]


def test_expected_vaf_autosomal_and_male_sex_chromosomes() -> None:
    # Autosomal 2:1 at purity 0.8: denom = 0.8 * 3 + 2 * 0.2 = 2.8
    f_m2 = float(expected_vaf(2, 1.0, purity=0.8, major_cn=2, minor_cn=1, normal_cn=2))
    f_m1 = float(expected_vaf(1, 1.0, purity=0.8, major_cn=2, minor_cn=1, normal_cn=2))
    assert np.isclose(f_m2, 1.6 / 2.8, atol=1e-12)
    assert np.isclose(f_m1, 0.8 / 2.8, atol=1e-12)

    # Male chrX 1:0 at purity 0.7: C_norm = 1 => denom = 0.7 * 1 + 1 * 0.3 = 1.0 => f = 0.7
    f_male_x = float(expected_vaf(1, 1.0, purity=0.7, major_cn=1, minor_cn=0, normal_cn=1))
    assert np.isclose(f_male_x, 0.70, atol=1e-12)


def test_pi_to_time_exact_closed_form_identities() -> None:
    for mode in ("default", "R_default"):
        for true_t in (0.0, 0.2, 0.5, 0.85, 1.0):
            # 2:1 -> mu1 = 3 - 2T, mu2 = T
            pi_21 = np.array([3.0 - 2.0 * true_t, true_t])
            pi_21 /= pi_21.sum()
            assert np.isclose(pi_to_time(2, 1, pi_21, settings=mode), true_t, atol=1e-12)

            # 2:0 -> mu1 = 2*(1 - T), mu2 = T
            pi_20 = np.array([2.0 * (1.0 - true_t), true_t])
            pi_20 /= pi_20.sum()
            assert np.isclose(pi_to_time(2, 0, pi_20, settings=mode), true_t, atol=1e-12)

            # 2:2 -> mu1 = 4*(1 - T), mu2 = 2*T
            pi_22 = np.array([4.0 * (1.0 - true_t), 2.0 * true_t])
            pi_22 /= pi_22.sum()
            assert np.isclose(pi_to_time(2, 2, pi_22, settings=mode), true_t, atol=1e-12)

    # Uninformative diploid 1:1 returns NaN
    assert np.isnan(pi_to_time(1, 1, [1.0]))


def test_pi_to_all_times_multi_event_high_copy_states() -> None:
    # 3:1 sequential gains at T1 = 0.25, T2 = 0.65:
    # mu3 = T1 = 0.25, mu2 = T2 - T1 = 0.40, mu1 = 4 - T1 - 2*T2 = 2.45
    t1_true, t2_true = 0.25, 0.65
    mu_31 = np.array([4.0 - t1_true - 2.0 * t2_true, t2_true - t1_true, t1_true])
    pi_31 = mu_31 / mu_31.sum()
    times_31 = pi_to_all_times(3, 1, pi_31)
    assert len(times_31) == 2
    assert np.allclose(times_31, [t1_true, t2_true], atol=1e-12)

    # 3:0 sequential gains at T1 = 0.30, T2 = 0.75:
    t1_30, t2_30 = 0.30, 0.75
    mu_30 = np.array([3.0 - t1_30 - 2.0 * t2_30, t2_30 - t1_30, t1_30])
    times_30 = pi_to_all_times(3, 0, mu_30 / mu_30.sum())
    assert np.allclose(times_30, [t1_30, t2_30], atol=1e-12)

    # 4:0 balanced doubling 1:0 -> 2:0 -> 4:0 at T1 = 0.20, T2 = 0.70:
    # mu4 = T1, mu3 = 0, mu2 = 2*(T2 - T1), mu1 = 4*(1 - T2)
    t1_40, t2_40 = 0.20, 0.70
    mu_40 = np.array([4.0 * (1.0 - t2_40), 2.0 * (t2_40 - t1_40), 0.0, t1_40])
    times_40 = pi_to_all_times(4, 0, mu_40 / mu_40.sum())
    assert np.allclose(times_40, [t1_40, t2_40], atol=1e-12)


def test_beta_binomial_logpmf_probabilities_sum_to_one() -> None:
    n = 50
    ks = np.arange(0, n + 1)
    log_p = beta_binomial_logpmf(ks, np.full_like(ks, n), np.array([0.35]), rho_bb=0.01)
    assert np.isclose(np.sum(np.exp(log_p)), 1.0, atol=1e-10)


def test_planted_gain_timing_recovery_across_states() -> None:
    purity = 0.80
    segs = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [1, 1, 1],
            "end": [50_000_000, 50_000_000, 50_000_000],
            "major_cn": [2, 2, 2],
            "minor_cn": [1, 0, 2],
        }
    )
    v1 = _simulate_timed_segment_variants("chr1", 1, 50_000_000, 2, 1, purity, gain_t=0.25, n_clonal=160, seed=11)
    v2 = _simulate_timed_segment_variants("chr2", 1, 50_000_000, 2, 0, purity, gain_t=0.70, n_clonal=160, seed=22)
    v3 = _simulate_timed_segment_variants("chr3", 1, 50_000_000, 2, 2, purity, gain_t=0.50, n_clonal=160, seed=33)
    variants = pd.concat([v1, v2, v3], ignore_index=True)

    res = mutation_time(variants, segs, purity=purity, n_boot=150, seed=42, settings="default")
    assert isinstance(res, MutationTimeResult)
    t_est = res.segments_df["T"].to_numpy()
    assert abs(t_est[0] - 0.25) < 0.08
    assert abs(t_est[1] - 0.70) < 0.08
    assert abs(t_est[2] - 0.50) < 0.08

    # Check bootstrap 95% CIs bracket point estimates and stay inside [0, 1]
    assert np.all(res.segments_df["T_lo"] <= res.segments_df["T"])
    assert np.all(res.segments_df["T"] <= res.segments_df["T_hi"])
    assert np.all((res.segments_df["T_lo"] >= 0.0) & (res.segments_df["T_hi"] <= 1.0))


def test_four_pcawg_timing_classes_accuracy() -> None:
    purity = 0.85
    segs = pd.DataFrame(
        {
            "chrom": ["1", "2"],
            "start": [1, 1],
            "end": [40_000_000, 40_000_000],
            "major_cn": [2, 1],
            "minor_cn": [0, 1],
        }
    )
    # Segment 1 (2:0): clonal early (m=2), clonal late (m=1), and subclonal (phi=0.35)
    v_gained = _simulate_timed_segment_variants(
        "chr1", 1, 40_000_000, 2, 0, purity, gain_t=0.5,
        n_clonal=120, n_subclonal=40, subclonal_phi=0.35, depth=140, seed=77
    )
    # Segment 2 (1:1): clonal [NA] (m=1) and subclonal (phi=0.35)
    v_diploid = _simulate_timed_segment_variants(
        "chr2", 1, 40_000_000, 1, 1, purity, gain_t=0.0,
        n_clonal=80, n_subclonal=30, subclonal_phi=0.35, depth=140, seed=88
    )
    variants = pd.concat([v_gained, v_diploid], ignore_index=True)
    clusters = pd.DataFrame({"ccf": [0.35]})

    for mode in ("default", "R_default"):
        res = mutation_time(
            variants, segs, purity=purity, clusters_df=clusters,
            n_boot=50, seed=42, settings=mode
        )
        vdf = res.variants_df
        observed_classes = set(vdf["timing_class"].unique())
        assert observed_classes == {
            "clonal [early]",
            "clonal [late]",
            "clonal [NA]",
            "subclonal",
        }

        # Check accuracy on Segment 1 early vs late vs subclonal
        s1 = vdf[vdf["segment_id"] == 0]
        early_acc = (s1[s1["true_m"] == 2]["timing_class"] == "clonal [early]").mean()
        late_acc = (
            s1[(s1["true_m"] == 1) & (s1["true_phi"] == 1.0)]["timing_class"] == "clonal [late]"
        ).mean()
        sub_acc = (s1[s1["true_phi"] < 1.0]["timing_class"] == "subclonal").mean()
        assert early_acc > 0.90
        assert late_acc > 0.90
        assert sub_acc > 0.85

        # Diploid clonal variants must be "clonal [NA]"
        s2_clonal = vdf[(vdf["segment_id"] == 1) & (vdf["true_phi"] == 1.0)]
        assert (s2_clonal["timing_class"] == "clonal [NA]").mean() > 0.92


def test_male_chrx_haploid_germline_mut_cn_accuracy() -> None:
    purity = 0.70
    segs = pd.DataFrame(
        {"chrom": ["chrX"], "start": [1], "end": [30_000_000], "major_cn": [2], "minor_cn": [0]}
    )
    # Male X with 2:0 gain: C_norm = 1 => denom = 0.7*2 + 0.3*1 = 1.7
    vars_x = _simulate_timed_segment_variants(
        "X", 1, 30_000_000, 2, 0, purity=purity, gain_t=0.50,
        n_clonal=120, depth=130, normal_cn=1, seed=101
    )
    res_male = mutation_time(vars_x, segs, purity=purity, gender="male", n_boot=40, seed=1)
    m2_mean = res_male.variants_df.loc[res_male.variants_df["true_m"] == 2, "mut_cn"].mean()
    m1_mean = res_male.variants_df.loc[res_male.variants_df["true_m"] == 1, "mut_cn"].mean()
    assert abs(m2_mean - 2.0) < 0.08
    assert abs(m1_mean - 1.0) < 0.08
    assert abs(res_male.segments_df.loc[0, "T"] - 0.50) < 0.10


def test_wgd_synchronous_detection_vs_diploid_negative_control() -> None:
    purity = 0.80
    true_wgd_t = 0.42
    wgd_segs = pd.DataFrame(
        {
            "chrom": ["1", "2", "3", "4", "5"],
            "start": [1] * 5,
            "end": [60_000_000] * 5,
            "major_cn": [2, 2, 2, 2, 1],
            "minor_cn": [2, 2, 0, 2, 1],
        }
    )
    parts = [
        _simulate_timed_segment_variants("1", 1, 60_000_000, 2, 2, purity, true_wgd_t, 140, seed=201),
        _simulate_timed_segment_variants("2", 1, 60_000_000, 2, 2, purity, true_wgd_t, 140, seed=202),
        _simulate_timed_segment_variants("3", 1, 60_000_000, 2, 0, purity, true_wgd_t, 120, seed=203),
        _simulate_timed_segment_variants("4", 1, 60_000_000, 2, 2, purity, true_wgd_t, 140, seed=204),
        _simulate_timed_segment_variants("5", 1, 60_000_000, 1, 1, purity, 0.0, 90, seed=205),
    ]
    res_wgd = mutation_time(pd.concat(parts, ignore_index=True), wgd_segs, purity=purity, n_boot=100, seed=9)
    wgd = res_wgd.wgd_result
    assert isinstance(wgd, WGDTimingResult)
    assert wgd.is_wgd is True
    assert wgd.is_synchronous is True
    assert abs(wgd.wgd_time - true_wgd_t) < 0.07
    assert wgd.wgd_time_lo <= wgd.wgd_time <= wgd.wgd_time_hi

    # Diploid negative control (mostly 1:1 with a single focal 2:1 trisomy)
    dip_segs = pd.DataFrame(
        {
            "chrom": ["1", "2", "3", "4"],
            "start": [1] * 4,
            "end": [60_000_000] * 4,
            "major_cn": [1, 1, 1, 2],
            "minor_cn": [1, 1, 1, 1],
        }
    )
    dip_vars = pd.concat(
        [
            _simulate_timed_segment_variants("1", 1, 60_000_000, 1, 1, purity, 0.0, 80, seed=301),
            _simulate_timed_segment_variants("4", 1, 60_000_000, 2, 1, purity, 0.6, 80, seed=302),
        ],
        ignore_index=True,
    )
    res_dip = mutation_time(dip_vars, dip_segs, purity=purity, n_boot=50, seed=9)
    assert res_dip.wgd_result.is_wgd is False


def test_default_vs_r_default_prior_and_mut_cn_behavior() -> None:
    # Low-mutation 2:0 segment with 4 purely late mutations (m=1, alt ~ 20/80 at purity 0.5 => VAF 0.25)
    segs = pd.DataFrame(
        {"chrom": ["1"], "start": [1], "end": [10_000_000], "major_cn": [2], "minor_cn": [0]}
    )
    vars_df = pd.DataFrame(
        {
            "chrom": ["1"] * 4,
            "pos": [1000, 2000, 3000, 4000],
            "alt_count": [19, 21, 20, 18],
            "total_depth": [80, 80, 80, 80],
        }
    )
    res_def = mutation_time(vars_df, segs, purity=0.5, n_boot=20, seed=7, settings="default")
    res_r = mutation_time(vars_df, segs, purity=0.5, n_boot=20, seed=7, settings="R_default")

    # Laplace alpha=1.05 keeps unused m=2 strictly bounded away from zero collapse,
    # whereas R_default weak 1/K prior collapses empty m=2 state near zero
    pi_m2_def = float(res_def.segment_pis[0][1])
    pi_m2_r = float(res_r.segment_pis[0][1])
    assert pi_m2_def > 5e-3
    assert pi_m2_r < 5e-4
    assert pi_m2_def > 10.0 * pi_m2_r

    # In R_default, mut_cn equals raw inverted mut_cn identically
    assert np.allclose(res_r.variants_df["mut_cn"], res_r.variants_df["mut_cn_raw"])
    # In default, empirical-Bayes shrinkage moves mut_cn closer to posterior multiplicity 1.0
    err_def = np.mean(np.abs(res_def.variants_df["mut_cn"] - 1.0))
    err_r = np.mean(np.abs(res_r.variants_df["mut_cn"] - 1.0))
    assert err_def <= err_r


def test_standalone_time_cn_gains_mapping_and_sequence_inputs() -> None:
    segs = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [1, 1, 1],
            "end": [20_000_000, 20_000_000, 20_000_000],
            "major_cn": [2, 2, 1],
            "minor_cn": [1, 0, 1],
            "n_mutations": [100, 100, 50],
        }
    )
    pis_seq = [np.array([0.8, 0.2]), np.array([0.5, 0.5]), np.array([1.0])]
    timed_seq = time_cn_gains(segs, pis_seq, n_boot=100, seed=42, settings="default")
    timed_map = time_cn_gains(segs, {0: pis_seq[0], 1: pis_seq[1], 2: pis_seq[2]}, n_boot=100, seed=42)

    assert np.allclose(timed_seq["T"].dropna(), timed_map["T"].dropna())
    assert np.isclose(timed_seq.loc[0, "T"], 3.0 * 0.2 / (0.4 + 0.8), atol=1e-12)
    assert np.isclose(timed_seq.loc[1, "T"], 2.0 * 0.5 / (1.0 + 0.5), atol=1e-12)
    assert np.isnan(timed_seq.loc[2, "T"])


def test_mutationtimer_class_interface_and_summary() -> None:
    segs = pd.DataFrame(
        {"chrom": ["1"], "start": [1], "end": [25_000_000], "major_cn": [2], "minor_cn": [1]}
    )
    vars_df = _simulate_timed_segment_variants("1", 1, 25_000_000, 2, 1, 0.75, gain_t=0.45, n_clonal=180, seed=55)

    model = MutationTimeR(purity=0.75, gender="female", n_boot=50, seed=42, settings="R_default")
    res = model.run(vars_df, segs)
    summary = res.summary()
    df_out = res.to_dataframe()

    assert summary["n_variants"] == 180
    assert summary["n_timed_segments"] == 1
    assert summary["settings"] == "R_default"
    assert len(df_out) == 180
    assert abs(res.segments_df.loc[0, "T"] - 0.45) < 0.10


def test_unmapped_and_empty_variants_handled_gracefully() -> None:
    segs = pd.DataFrame(
        {"chrom": ["1"], "start": [1_000], "end": [10_000], "major_cn": [2], "minor_cn": [1]}
    )
    # Variant outside segment coordinates + empty test
    vars_unmapped = pd.DataFrame(
        {"chrom": ["1", "9"], "pos": [99_999, 5_000], "alt_count": [15, 20], "total_depth": [60, 60]}
    )
    res_unmapped = mutation_time(vars_unmapped, segs, purity=0.8, n_boot=10)
    assert list(res_unmapped.variants_df["segment_id"]) == [-1, -1]
    assert list(res_unmapped.variants_df["timing_class"]) == ["NA", "NA"]

    empty_vars = pd.DataFrame(columns=["chrom", "pos", "alt_count", "total_depth"])
    res_empty = mutation_time(empty_vars, segs, purity=0.8, n_boot=10)
    assert res_empty.variants_df.empty
    assert res_empty.segments_df.loc[0, "n_mutations"] == 0
