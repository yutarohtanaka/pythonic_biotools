from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from chromvar import (
    build_log_odds_pwm,
    compute_deviations,
    compute_expectations,
    compute_variability,
    compute_variant_pwm_disruption,
    extract_gtf_tss,
    get_background_peaks,
    nearest_tss,
    run_raw_count_pychromvar,
    scan_sequence_pwm,
    signed_tss_distance,
)


def test_chromvar_expectations_bg_deviations_and_variability(tmp_path):
    bad = pd.DataFrame([[0.15, 1.23], [0.0, 0.88]], index=["p1", "p2"], columns=["S1", "S2"])
    with pytest.raises(ValueError, match="integer fragment counts"):
        run_raw_count_pychromvar(bad, genome_fasta="dummy.fa")

    rng = np.random.default_rng(42)
    n_peaks, n_samples = 120, 20
    peaks = [f"peak_{i}" for i in range(n_peaks)]
    samples = [f"cell_{j}" for j in range(n_samples)]
    gc = rng.uniform(0.35, 0.65, size=n_peaks)

    counts = rng.poisson(5.0, size=(n_peaks, n_samples))
    counts[:25, 10:] += rng.poisson(15.0, size=(25, 10))
    counts[25:50, :10] += rng.poisson(15.0, size=(25, 10))
    counts_df = pd.DataFrame(counts, index=peaks, columns=samples)

    matches = pd.DataFrame(
        {
            "Motif_Active": np.r_[np.ones(25), np.zeros(n_peaks - 25)],
            "Motif_Null": rng.binomial(1, 0.2, size=n_peaks),
        },
        index=peaks,
    )

    exp = compute_expectations(counts_df)
    assert np.isclose(exp.sum(), 1.0)
    bg = get_background_peaks(counts_df, gc, n_iterations=30, seed=7)
    res = compute_deviations(counts_df, matches, bg, exp)
    assert res.z_scores.loc["Motif_Active", samples[10:]].mean() > 3.0
    assert abs(res.z_scores.loc["Motif_Null"].mean()) < 1.5

    var_df = compute_variability(res.z_scores, n_boot=50, seed=7)
    assert var_df.loc["Motif_Active", "variability"] > var_df.loc["Motif_Null", "variability"]
    assert var_df.loc["Motif_Active", "p_value_adj"] < 0.01

    gtf_file = tmp_path / "mini.gtf"
    gtf_file.write_text(
        'chr1\tHAVANA\tgene\t1001\t5000\t.\t+\t.\tgene_id "ENSG0001.1"; gene_name "PLUSG"; gene_type "protein_coding";\n'
        'chr1\tHAVANA\tgene\t8001\t12000\t.\t-\t.\tgene_id "ENSG0002.1"; gene_name "MINUSG"; gene_type "protein_coding";\n'
    )
    tss_df = extract_gtf_tss(gtf_file)
    assert signed_tss_distance("chr1", 11500, "MINUSG", tss_df) == 499
    assert list(nearest_tss("chr1", [1200, 11800], tss_df)["gene_name"]) == ["PLUSG", "MINUSG"]

    pfm = np.array(
        [
            [0.15, 0.45, 0.10, 0.30],
            [0.03, 0.03, 0.04, 0.90],
            [0.03, 0.03, 0.90, 0.04],
            [0.02, 0.02, 0.02, 0.94],
            [0.02, 0.02, 0.02, 0.94],
            [0.02, 0.02, 0.02, 0.94],
            [0.35, 0.10, 0.15, 0.40],
            [0.20, 0.35, 0.15, 0.30],
        ]
    )
    pwm = build_log_odds_pwm(pfm)
    seq = "A" * 40 + "CTGTTTAC" + "C" * 40 + "GTAAACAG" + "G" * 20
    hits = scan_sequence_pwm(seq, pwm, min_log2_odds=6.5, tss_position_in_seq=100)
    assert len(hits) == 2
    assert compute_variant_pwm_disruption("CTGTTTAC", "CTGGGGAC", pwm) < -5.0


def test_sparse_csr_parity_nan_variability_and_pwm_windows():
    rng = np.random.default_rng(99)
    n_peaks, n_samples = 80, 16
    samples = [f"S{j}" for j in range(n_samples)]
    motifs = ["M1", "M2", "M3"]
    gc = rng.uniform(0.30, 0.70, size=n_peaks)

    counts = rng.poisson(1.5, size=(n_peaks, n_samples))
    counts[:, 0] += 1  # ensure strictly positive sample depths
    for j in range(1, n_samples):
        counts[0, j] += 1
    matches = rng.binomial(1, 0.25, size=(n_peaks, 3))

    counts_df = pd.DataFrame(counts, columns=samples)
    matches_df = pd.DataFrame(matches, columns=motifs)
    counts_csr = sparse.csr_matrix(counts)
    matches_csc = sparse.csc_matrix(matches)

    with pytest.raises(ValueError, match="integer fragment counts"):
        compute_expectations(sparse.csr_matrix([[0.5, 1.0], [0.0, 2.0]]))

    exp_dense = compute_expectations(counts_df)
    exp_sparse = compute_expectations(counts_csr)
    np.testing.assert_allclose(exp_sparse, exp_dense, rtol=1e-13, atol=1e-13)

    bg_dense = get_background_peaks(counts_df, gc, n_iterations=25, seed=11)
    bg_sparse = get_background_peaks(counts_csr, gc, n_iterations=25, seed=11)
    np.testing.assert_array_equal(bg_sparse, bg_dense)

    res_dense = compute_deviations(counts_df, matches_df, bg_dense, exp_dense)
    res_sparse = compute_deviations(
        counts_csr, matches_csc, bg_sparse, exp_sparse, sample_names=samples, motif_names=motifs
    )
    pd.testing.assert_frame_equal(res_sparse.deviations, res_dense.deviations, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(res_sparse.z_scores, res_dense.z_scores, rtol=1e-12, atol=1e-12)

    # NaN-safe compute_variability: partial NaN in M1 and all-NaN row M_empty must not poison BH FDR
    z_nan = res_dense.z_scores.copy()
    z_nan.iloc[0, 0] = np.nan
    z_nan.loc["M_empty"] = np.nan
    var_nan = compute_variability(z_nan, n_boot=40, seed=3)
    assert np.all(np.isfinite(var_nan.loc[["M1", "M2", "M3"], "bootstrap_lower_bound"]))
    assert np.all(np.isfinite(var_nan.loc[["M1", "M2", "M3"], "p_value_adj"]))
    assert np.isnan(var_nan.loc["M_empty", "p_value"]) and np.isnan(var_nan.loc["M_empty", "p_value_adj"])

    # Scale-invariant build_log_odds_pwm on raw count matrices vs frequency matrices
    pfm_probs = np.array(
        [
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.85],
            [0.05, 0.05, 0.85, 0.05],
            [0.85, 0.05, 0.05, 0.05],
        ]
    )
    pwm_from_prob = build_log_odds_pwm(pfm_probs, pseudocount=0.01)
    pwm_from_counts = build_log_odds_pwm(pfm_probs * 2500.0, pseudocount=0.01)
    np.testing.assert_allclose(pwm_from_counts, pwm_from_prob, rtol=1e-13, atol=1e-13)

    # Overlapping k-mer window scan in compute_variant_pwm_disruption (k=4, consensus CTGA):
    # SNV T->C at position 1 inside a 15-bp flanking window ("AAAAA" + "CTGA" + "AAAAAA" -> "AAAAA" + "CCGA" + "AAAAAA")
    ref_flank = "AAAAACTGAAAAAAA"
    alt_flank = "AAAAACCGAAAAAAA"
    delta = compute_variant_pwm_disruption(ref_flank, alt_flank, pwm_from_prob)
    assert delta < -2.5
    # Creation of a de novo motif ("AAAAACCGAAAAAAA" -> "AAAAACTGAAAAAAA") has positive symmetrical disruption
    assert compute_variant_pwm_disruption(alt_flank, ref_flank, pwm_from_prob) == pytest.approx(-delta, rel=1e-12)


def test_r_default_settings():
    import itertools
    from chromvar import pwm_threshold_from_pvalue

    rng = np.random.default_rng(2026)
    counts = rng.poisson(6.0, size=(60, 12)) + 1
    gc = rng.uniform(0.35, 0.65, size=60)
    matches = rng.binomial(1, 0.25, size=(60, 3))
    bg_init = get_background_peaks(counts, gc, n_iterations=10, seed=1)
    res_init = compute_deviations(counts, matches, bg_init)
    pfm_counts = np.array(
        [
            [2.0, 90.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 90.0],
            [2.0, 2.0, 94.0, 2.0],
            [92.0, 2.0, 3.0, 3.0],
            [2.0, 92.0, 3.0, 3.0],
            [3.0, 2.0, 3.0, 92.0],
            [2.0, 3.0, 93.0, 2.0],
            [93.0, 2.0, 2.0, 3.0],
        ]
    )
    pwm_dummy = build_log_odds_pwm(pfm_counts)

    # 1. Invalid settings raises ValueError across all 5 functions
    for fn in (
        lambda: get_background_peaks(counts, gc, settings="invalid"),  # type: ignore[arg-type]
        lambda: compute_deviations(counts, matches, bg_init, settings="invalid"),  # type: ignore[arg-type]
        lambda: compute_variability(res_init.z_scores, settings="invalid"),  # type: ignore[arg-type]
        lambda: build_log_odds_pwm(pfm_counts, settings="invalid"),  # type: ignore[arg-type]
        lambda: scan_sequence_pwm("ACGTACGTACGT", pwm_dummy, settings="invalid"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="settings"):
            fn()

    # 2. get_background_peaks: zero-fragment peak raises ValueError in R_default, allowed in default
    counts_zero_peak = counts.copy()
    counts_zero_peak[0, :] = 0
    bg_def_z = get_background_peaks(counts_zero_peak, gc, n_iterations=10, seed=5, settings="default")
    assert bg_def_z.shape == (60, 10)
    with pytest.raises(ValueError, match="> 0 total fragments"):
        get_background_peaks(counts_zero_peak, gc, n_iterations=10, seed=5, settings="R_default")

    bg_r = get_background_peaks(counts, gc, n_iterations=25, seed=5, settings="R_default")
    assert bg_r.shape == (60, 25) and bg_r.min() >= 0 and bg_r.max() < 60

    # 3. compute_deviations: masks dev and z to NaN wherever expected = E_motif * sample_depth < 1.0
    # Make Motif_Rare match only peak 0 (fraction ~ 1/60), and give sample 0 depth = 12 -> expected ~ 0.2 < 1.0!
    counts_low = np.full((60, 8), 20, dtype=int)
    counts_low[:, 0] = 0
    counts_low[:12, 0] = 1  # sample 0 depth = 12
    matches_rare = pd.DataFrame(
        {
            "Motif_Common": np.r_[np.ones(30), np.zeros(30)],
            "Motif_Rare": np.r_[np.ones(1), np.zeros(59)],
        }
    )
    res_def_low = compute_deviations(counts_low, matches_rare, bg_r, settings="default")
    res_r_low = compute_deviations(counts_low, matches_rare, bg_r, settings="R_default")
    assert np.isfinite(res_def_low.z_scores.iloc[1, 0])
    assert np.isnan(res_r_low.deviations.iloc[1, 0]) and np.isnan(res_r_low.z_scores.iloc[1, 0])
    assert np.all(np.isfinite(res_r_low.z_scores.loc["Motif_Common"]))

    # 4. compute_variability: default n_boot = 1000 in R_default matches explicit n_boot=1000
    var_r_auto = compute_variability(res_r_low.z_scores, seed=19, settings="R_default")
    var_r_1000 = compute_variability(res_r_low.z_scores, n_boot=1000, seed=19, settings="default")
    pd.testing.assert_frame_equal(var_r_auto, var_r_1000)

    # 5. build_log_odds_pwm: TFBSTools raw-count pseudocount = 0.8 (count-scale shrinkage)
    bg_u = np.array([[0.25, 0.25, 0.25, 0.25]])
    pwm_r = build_log_odds_pwm(pfm_counts, settings="R_default")
    expected_pwm_r = np.log2(((pfm_counts + 0.8 * bg_u) / (pfm_counts.sum(axis=1, keepdims=True) + 0.8)) / bg_u)
    np.testing.assert_allclose(pwm_r, expected_pwm_r, rtol=1e-13, atol=1e-13)
    # Scaling raw counts by 10x increases information content in R_default (unlike scale-invariant default)
    pwm_r_deep = build_log_odds_pwm(pfm_counts * 10.0, settings="R_default")
    assert pwm_r_deep.max() > pwm_r.max()

    # 6. Exact DP background score-distribution threshold vs brute-force 4^8 enumeration & dedup_window_bp=0
    all_kmers = np.array(list(itertools.product(range(4), repeat=8)))  # (65536, 8)
    exact_scores = pwm_r[np.arange(8)[None, :], all_kmers].sum(axis=1)
    thresh_dp = pwm_threshold_from_pvalue(pwm_r, p_cutoff=5e-5)
    emp_tail_p = float(np.mean(exact_scores >= thresh_dp))
    assert 0.0 < emp_tail_p <= 5e-5 + 1e-12

    # Tandem direct repeat CTGACTGACTGACTGA has 3 overlapping 8-mer consensus matches at offsets 0, 4, 8:
    # default (dedup_window_bp=8) suppresses offset 4, whereas R_default (dedup_window_bp=0) retains all 3!
    tandem_seq = "CTGACTGACTGACTGA"
    hits_def = scan_sequence_pwm(tandem_seq, pwm_r, min_log2_odds=thresh_dp, settings="default")
    hits_r = scan_sequence_pwm(tandem_seq, pwm_r, settings="R_default")
    assert [h.offset_5prime for h in hits_r] == [0, 4, 8]
    assert len(hits_def) == 2

