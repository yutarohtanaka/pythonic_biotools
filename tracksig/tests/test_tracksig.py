"""Comprehensive unit tests for tracksig and TrackSigFreq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multinomial

from tracksig import (
    COSMIC_96_CONTEXTS,
    SegmentScorer,
    TrackSig,
    compute_ccf_pseudotime,
    detect_active_signatures,
    fit_multinomial_signatures_em,
    run_pelt,
    tracksig,
)


def _make_signature_panel() -> pd.DataFrame:
    """Create 5 distinct, well-separated 96-channel mutational signatures."""
    mat = np.full((96, 5), 0.001, dtype=float)
    # SBS1-like: peaked at N[C>T]G (C>T is block 2: indices 32..47)
    mat[34:42, 0] = 0.11
    # SBS2-like: T[C>T]N & T[C>G]N (APOBEC)
    mat[28:32, 1] = 0.12
    mat[44:48, 1] = 0.12
    # SBS4-like: C>A (indices 0..15)
    mat[0:16, 2] = 0.06
    # SBS5-like: broad clock-like flat background
    mat[:, 3] = 1.0 / 96.0
    # SBS17-like: C[T>G]T / N[T>G]T (indices 80..95)
    mat[88:96, 4] = 0.115
    mat /= mat.sum(axis=0, keepdims=True)
    return pd.DataFrame(
        mat,
        index=list(COSMIC_96_CONTEXTS),
        columns=["SBS1", "SBS2", "SBS4", "SBS5", "SBS17"],
    )


def _simulate_cohort(
    sig_df: pd.DataFrame,
    epochs: list[tuple[int, float, dict[str, float]]],
    purity: float = 0.85,
    depth: int = 100,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate SNVs across evolutionary epochs (n_muts, true_ccf, weights_dict)."""
    rng = np.random.default_rng(seed)
    contexts: list[str] = []
    alts: list[int] = []
    depths: list[int] = []

    for n_m, ccf, weights in epochs:
        mix = np.zeros(96, dtype=float)
        for s_name, w in weights.items():
            mix += w * sig_df[s_name].to_numpy()
        mix /= mix.sum()
        ctx_epoch = rng.choice(COSMIC_96_CONTEXTS, size=n_m, p=mix)
        # Diploid heterozygous m=1 -> expected VAF = purity * ccf / 2
        exp_vaf = purity * ccf / 2.0
        d_vec = rng.poisson(depth, size=n_m).clip(30, 300)
        a_vec = rng.binomial(d_vec, exp_vaf)
        contexts.extend(ctx_epoch.tolist())
        alts.extend(a_vec.tolist())
        depths.extend(d_vec.tolist())

    df = pd.DataFrame(
        {
            "context": contexts,
            "alt_count": alts,
            "total_depth": depths,
            "total_cn": 2,
            "major_cn": 1,
        }
    )
    # Shuffle input rows to verify sorting from scratch
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


def test_settings_validation_all_entrypoints() -> None:
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(sig_df, [(120, 1.0, {"SBS1": 1.0})], seed=1)
    counts = np.ones(96, dtype=float)

    with pytest.raises(ValueError, match="Unknown settings"):
        compute_ccf_pseudotime(mut_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        fit_multinomial_signatures_em(counts, sig_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        detect_active_signatures(mut_df, sig_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        tracksig(mut_df, sig_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        TrackSig(settings="bad")  # type: ignore[arg-type]


def test_ccf_pseudotime_multiplicity_and_purity() -> None:
    """Verify Beta-Binomial multiplicity m=2 vs m=1 selection under CN=3, major_cn=2."""
    # Purity = 0.8, total_cn = 3, normal_cn = 2 -> eff_cn = 2.4 + 0.4 = 2.8
    # Clonal m=2 (CCF=1.0) -> expected VAF = 1.6 / 2.8 = 0.5714
    # Subclonal m=1 (CCF=0.5) -> expected VAF = 0.4 / 2.8 = 0.1429
    df = pd.DataFrame(
        {
            "context": [COSMIC_96_CONTEXTS[0]] * 60 + [COSMIC_96_CONTEXTS[1]] * 60,
            "alt_count": [114] * 60 + [29] * 60,
            "total_depth": [200] * 120,
            "total_cn": [3] * 120,
            "major_cn": [2] * 120,
        }
    )
    out = compute_ccf_pseudotime(df, purity=0.8, bin_size=100, settings="default")
    assert len(out) == 120
    # Decreasing CCF order
    assert np.all(np.diff(out["ccf_cont"].to_numpy()) <= 1e-12)
    clonal_part = out.iloc[:60]
    sub_part = out.iloc[60:]
    assert np.all(clonal_part["multiplicity"].to_numpy() == 2)
    assert np.all(sub_part["multiplicity"].to_numpy() == 1)
    assert np.isclose(clonal_part["ccf"].mean(), 1.0, atol=0.05)
    assert np.isclose(sub_part["ccf"].mean(), 0.5, atol=0.05)


def test_wes_adaptive_binning_vs_r_default_strict_bin_size() -> None:
    sig_df = _make_signature_panel()
    wes_df = _simulate_cohort(sig_df, [(80, 1.0, {"SBS1": 0.7, "SBS5": 0.3})], seed=7)

    # Default mode adapts bin_size down to >=25 so 80 SNVs yield 3 bins
    binned_def = compute_ccf_pseudotime(wes_df, purity=0.85, bin_size=100, settings="default")
    assert binned_def["bin_id"].max() == 3

    # R_default enforces strict binSize=100 unless overridden
    with pytest.raises(ValueError, match="settings='R_default' requires at least bin_size=100"):
        compute_ccf_pseudotime(wes_df, purity=0.85, bin_size=100, settings="R_default")

    # Explicit override works cleanly in R_default
    binned_r_override = compute_ccf_pseudotime(
        wes_df, purity=0.85, bin_size=40, settings="R_default"
    )
    assert binned_r_override["bin_id"].max() == 2


def test_multinomial_em_exact_recovery_and_logpmf_parity() -> None:
    sig_df = _make_signature_panel()
    phi = sig_df.to_numpy()
    true_pi = np.array([0.45, 0.30, 0.0, 0.25, 0.0])
    rng = np.random.default_rng(99)
    counts = rng.multinomial(5000, phi @ true_pi)

    res = fit_multinomial_signatures_em(counts, sig_df, tol=1e-9, max_iter=1000)
    exposures, ll = res
    assert np.isclose(np.sum(exposures), 1.0, atol=1e-12)
    assert np.all(exposures >= 0.0)
    assert np.allclose(exposures, true_pi, atol=0.03)

    expected_ll = float(multinomial.logpmf(counts, n=int(counts.sum()), p=phi @ exposures))
    assert np.isclose(ll, expected_ll, atol=1e-8)


def test_multinomial_em_2d_pooling_and_zero_counts() -> None:
    sig_df = _make_signature_panel()
    zero_counts = np.zeros(96, dtype=float)
    pi_zero, ll_zero = fit_multinomial_signatures_em(zero_counts, sig_df)
    assert np.allclose(pi_zero, np.full(5, 0.2))
    assert ll_zero == 0.0

    rng = np.random.default_rng(11)
    bin_mat = rng.multinomial(200, sig_df["SBS2"].to_numpy(), size=4)
    pi_2d, ll_2d = fit_multinomial_signatures_em(bin_mat, sig_df)
    pi_1d, ll_1d = fit_multinomial_signatures_em(bin_mat.sum(axis=0), sig_df)
    assert np.allclose(pi_2d, pi_1d, atol=1e-9)
    assert np.isclose(ll_2d, ll_1d, atol=1e-9)


def test_detect_active_signatures_prefilter() -> None:
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [(400, 1.0, {"SBS1": 0.65, "SBS17": 0.35})],
        purity=0.9,
        seed=21,
    )
    active = detect_active_signatures(mut_df, sig_df, active_threshold=0.05)
    assert set(active) == {"SBS1", "SBS17"}


def test_pelt_pruning_matches_unpruned_exact_dp() -> None:
    """Verify PELT pruning with K_0=0 produces identical solution to O(T^2) DP."""
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [
            (300, 1.0, {"SBS1": 0.8, "SBS5": 0.2}),
            (300, 0.45, {"SBS2": 0.85, "SBS5": 0.15}),
            (200, 0.20, {"SBS17": 0.9, "SBS5": 0.1}),
        ],
        purity=0.9,
        seed=33,
    )
    binned = compute_ccf_pseudotime(mut_df, purity=0.9, bin_size=100)
    n_bins = int(binned["bin_id"].max())
    ctx_map = {c: i for i, c in enumerate(COSMIC_96_CONTEXTS)}
    ch = np.array([ctx_map[c] for c in binned["context"]], dtype=int)
    b_ids = binned["bin_id"].to_numpy(dtype=int)
    bin_counts = np.zeros((n_bins, 96), dtype=float)
    np.add.at(bin_counts, (b_ids - 1, ch), 1.0)

    bin_n = np.array([np.sum(b_ids == b) for b in range(1, n_bins + 1)], dtype=float)
    cc = binned["ccf_cont"].to_numpy(dtype=float)
    bin_s1 = np.array([np.sum(cc[b_ids == b]) for b in range(1, n_bins + 1)], dtype=float)
    bin_s2 = np.array([np.sum(cc[b_ids == b] ** 2) for b in range(1, n_bins + 1)], dtype=float)
    bounds = np.empty(n_bins + 1, dtype=float)
    bounds[0] = np.inf
    bounds[n_bins] = -np.inf
    for b in range(1, n_bins):
        bounds[b] = 0.5 * (np.min(cc[b_ids == b]) + np.max(cc[b_ids == b + 1]))

    for method in ("Signature", "SigFreq"):
        scorer = SegmentScorer(
            bin_counts=bin_counts,
            phi_active=sig_df.to_numpy(),
            score_method=method,
            bin_n=bin_n,
            bin_sum_ccf=bin_s1,
            bin_sum_ccf2=bin_s2,
            boundaries=bounds,
            global_sigma=float(np.sqrt(np.mean(binned["ccf_sd"].to_numpy() ** 2))),
        )
        pen = 5.0 * np.log(len(mut_df))
        cpts_pelt, cost_pelt = run_pelt(scorer.cost, n_bins=n_bins, penalty=pen, use_pruning=True)
        cpts_dp, cost_dp = run_pelt(scorer.cost, n_bins=n_bins, penalty=pen, use_pruning=False)
        assert cpts_pelt == cpts_dp
        assert np.isclose(cost_pelt, cost_dp, atol=1e-10)


def test_null_stationary_single_clone_no_false_changepoints() -> None:
    """A single homogeneous clonal population must produce 0 changepoints in both modes."""
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [(600, 1.0, {"SBS1": 0.6, "SBS5": 0.4})],
        purity=0.85,
        depth=90,
        seed=101,
    )
    res_sig = tracksig(mut_df, sig_df, purity=0.85, bin_size=100, score_method="Signature")
    res_freq = tracksig(mut_df, sig_df, purity=0.85, bin_size=100, score_method="SigFreq")
    assert res_sig.changepoints == []
    assert res_freq.changepoints == []
    assert len(res_freq.segment_exposures_df) == 1


def test_signature_shift_detection_at_same_ccf_or_subclone() -> None:
    """Detect signature shift from SBS1/SBS5 (clonal) to SBS2/SBS4 (subclonal)."""
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [
            (400, 1.0, {"SBS1": 0.75, "SBS5": 0.25}),
            (400, 0.35, {"SBS2": 0.70, "SBS4": 0.30}),
        ],
        purity=0.9,
        depth=110,
        seed=202,
    )
    res = tracksig(mut_df, sig_df, purity=0.9, bin_size=100, score_method="Signature")
    assert len(res.changepoints) == 1
    assert res.changepoints[0] in (3, 4, 5)
    seg1 = res.segment_exposures_df.iloc[0]
    seg2 = res.segment_exposures_df.iloc[1]
    assert seg1["SBS1"] > 0.55
    assert seg2["SBS2"] > 0.50


def test_sigfreq_detects_signature_preserving_subclonal_sweep() -> None:
    """TrackSigFreq detects a subclonal sweep when mutational signatures do NOT change."""
    sig_df = _make_signature_panel()
    # Both clones have identical signature weights (70% SBS1 + 30% SBS5)
    mut_df = _simulate_cohort(
        sig_df,
        [
            (400, 1.0, {"SBS1": 0.70, "SBS5": 0.30}),
            (400, 0.30, {"SBS1": 0.70, "SBS5": 0.30}),
        ],
        purity=0.90,
        depth=120,
        seed=303,
    )
    res_sig_only = tracksig(
        mut_df, sig_df, purity=0.90, bin_size=100, score_method="Signature"
    )
    res_sig_freq = tracksig(
        mut_df, sig_df, purity=0.90, bin_size=100, score_method="SigFreq"
    )
    assert res_sig_only.changepoints == []
    assert len(res_sig_freq.changepoints) >= 1
    assert any(abs(cp - 4) <= 1 for cp in res_sig_freq.changepoints)


def test_multi_epoch_trajectory_and_decreasing_ccfs() -> None:
    """Three distinct evolutionary epochs produce two ordered changepoints."""
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [
            (300, 1.00, {"SBS1": 0.85, "SBS5": 0.15}),
            (300, 0.52, {"SBS2": 0.85, "SBS5": 0.15}),
            (300, 0.18, {"SBS17": 0.85, "SBS5": 0.15}),
        ],
        purity=0.95,
        depth=140,
        seed=404,
    )
    res = tracksig(mut_df, sig_df, purity=0.95, bin_size=100, score_method="SigFreq")
    assert len(res.changepoints) == 2
    assert abs(res.changepoints[0] - 3) <= 1
    assert abs(res.changepoints[1] - 6) <= 1
    assert res.changepoint_ccfs[0] > res.changepoint_ccfs[1]


def test_nonparametric_bootstrap_support_and_credible_bands() -> None:
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [
            (300, 1.0, {"SBS1": 0.85, "SBS5": 0.15}),
            (300, 0.35, {"SBS2": 0.85, "SBS5": 0.15}),
        ],
        purity=0.9,
        depth=120,
        seed=505,
    )
    res = tracksig(
        mut_df,
        sig_df,
        purity=0.9,
        bin_size=100,
        score_method="SigFreq",
        bootstrap_iters=15,
        seed=77,
    )
    assert res.changepoint_support is not None
    cp = res.changepoints[0]
    assert res.changepoint_support[cp] >= 0.7
    assert res.bin_exposure_ci_low_df is not None
    assert res.bin_exposure_ci_high_df is not None
    for s_name in res.active_signatures:
        lo = res.bin_exposure_ci_low_df[s_name].to_numpy()
        hi = res.bin_exposure_ci_high_df[s_name].to_numpy()
        assert np.all(hi >= lo - 1e-12)


def test_r_default_penalty_scaling_and_oop_estimator() -> None:
    sig_df = _make_signature_panel()
    mut_df = _simulate_cohort(
        sig_df,
        [
            (300, 1.0, {"SBS1": 0.8, "SBS5": 0.2}),
            (300, 0.35, {"SBS4": 0.8, "SBS5": 0.2}),
        ],
        purity=0.9,
        depth=100,
        seed=606,
    )
    res_def = tracksig(
        mut_df, sig_df, purity=0.9, bin_size=100, score_method="Signature", settings="default"
    )
    est_r = TrackSig(
        purity=0.9, bin_size=100, score_method="Signature", settings="R_default"
    )
    res_r = est_r.fit(mut_df, sig_df)

    # Penalty in default scales with log(N=600), whereas R_default scales with log(T=6)
    assert res_def.penalty_per_changepoint > res_r.penalty_per_changepoint
    assert np.isclose(
        res_r.penalty_per_changepoint,
        max(1, len(res_r.active_signatures) - 1) * np.log(6),
    )
    text = res_def.summary()
    assert "TrackSigResult" in text
    df_out = res_def.to_dataframe()
    assert len(df_out) == 6
    assert np.allclose(df_out[res_def.active_signatures].sum(axis=1), 1.0)


def test_direct_ccf_and_vaf_columns_input() -> None:
    sig_df = _make_signature_panel()
    rng = np.random.default_rng(707)
    ctx1 = rng.choice(COSMIC_96_CONTEXTS, size=200, p=sig_df["SBS1"].to_numpy())
    ctx2 = rng.choice(COSMIC_96_CONTEXTS, size=200, p=sig_df["SBS17"].to_numpy())
    df_ccf = pd.DataFrame(
        {
            "context": np.concatenate([ctx1, ctx2]),
            "ccf": np.concatenate([np.full(200, 1.0), np.full(200, 0.35)]),
        }
    )
    res = tracksig(df_ccf, sig_df, bin_size=100, score_method="SigFreq")
    assert res.changepoints == [2]

    df_vaf = pd.DataFrame(
        {
            "context": np.concatenate([ctx1, ctx2]),
            "vaf": np.concatenate([rng.uniform(0.42, 0.52, 200), rng.uniform(0.12, 0.20, 200)]),
        }
    )
    res_vaf = tracksig(df_vaf, sig_df, purity=1.0, bin_size=100, score_method="Signature")
    assert res_vaf.changepoints == [2]
