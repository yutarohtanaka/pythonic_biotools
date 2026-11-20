"""Unit tests for the standalone monalisa package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import monalisa
from monalisa import (
    DINUCLEOTIDES,
    BinnedMotifEnrichmentResult,
    MonaLisa,
    MotifScanResult,
    StabilitySelectionResult,
    bin_genomic_changes,
    calc_binned_motif_enrichment,
    calc_sequence_composition,
    pfm_to_pwm,
    rand_lasso_stab_sel,
    scan_sequences_pwm,
)


def _consensus_to_pfm(consensus: str, weight: float = 100.0) -> np.ndarray:
    """Create a sharp (4, L) PFM from an unambiguous ACGT consensus string."""
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    pfm = np.ones((4, len(consensus)), dtype=np.float64)
    for pos, ch in enumerate(consensus.upper()):
        pfm[base_map[ch], pos] = weight
    return pfm


def _random_dna(
    rng: np.random.Generator, n_seqs: int, length: int, gc_prob: float | np.ndarray = 0.5
) -> list[str]:
    """Generate random DNA sequences with specified per-sequence GC probability."""
    alphabet = np.array(["A", "C", "G", "T"])
    gc_arr = np.broadcast_to(np.asarray(gc_prob, dtype=np.float64), (n_seqs,))
    seqs: list[str] = []
    for i in range(n_seqs):
        g = float(np.clip(gc_arr[i], 0.05, 0.95))
        probs = [0.5 * (1.0 - g), 0.5 * g, 0.5 * g, 0.5 * (1.0 - g)]
        chars = rng.choice(alphabet, size=length, p=probs)
        seqs.append("".join(chars.tolist()))
    return seqs


def _reverse_complement(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]


def test_settings_validation_all_public_apis() -> None:
    """Every primary function and class in monalisa rejects unknown settings strings."""
    pfm = _consensus_to_pfm("TGAGTCA")
    pwm = pfm_to_pwm(pfm)
    assert isinstance(pwm, np.ndarray)
    seqs = ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA", "ATATATATATATATAT", "CGCGCGCGCGCGCGCG"]
    vals = [-0.5, -0.1, 0.1, 0.6]
    bins = bin_genomic_changes(vals, n_bins=3)

    with pytest.raises(ValueError, match="Unknown settings"):
        bin_genomic_changes(vals, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        pfm_to_pwm(pfm, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        scan_sequences_pwm(seqs, {"AP1": pwm}, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        calc_sequence_composition(seqs, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        calc_binned_motif_enrichment(seqs, bins=bins, pwms={"AP1": pwm}, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        rand_lasso_stab_sel(np.ones((6, 2)), np.arange(6, dtype=float), settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        MonaLisa(settings="bad")  # type: ignore[arg-type]


def test_bin_genomic_changes_equaln_equalwidth_and_cutpoints() -> None:
    """bin_genomic_changes produces ordered categories with symmetric neutral 'zero' bin."""
    rng = np.random.default_rng(11)
    vals = rng.normal(0.0, 0.4, size=500)

    bins_eq_n = bin_genomic_changes(vals, method="equalN", n_bins=5, min_abs_val=0.0, zero_bin=True)
    assert isinstance(bins_eq_n, pd.Categorical)
    assert bins_eq_n.ordered is True
    assert len(bins_eq_n.categories) == 5
    assert bins_eq_n.categories[2] == "zero"
    counts = pd.Series(bins_eq_n).value_counts()
    assert counts.min() >= 90 and counts.max() <= 110

    # Explicit min_abs_val threshold
    bins_min_abs = bin_genomic_changes(
        vals, method="equalWidth", n_bins=5, min_abs_val=0.15, zero_bin=True
    )
    assert "zero" in bins_min_abs.categories
    assert np.all((bins_min_abs == "zero") == (np.abs(vals) <= 0.15))

    # Cutpoints mode
    bins_cp = bin_genomic_changes(
        vals, method="cutpoints", cutpoints=[-5.0, -0.3, 0.3, 5.0], min_abs_val=0.1, zero_bin=True
    )
    assert "zero" in bins_cp.categories
    assert np.all((bins_cp == "zero") == (np.abs(vals) <= 0.1))

    # Without zero_bin
    bins_no_zero = bin_genomic_changes(vals, method="equalN", n_bins=4, zero_bin=False)
    assert "zero" not in bins_no_zero.categories
    assert len(bins_no_zero.categories) == 4


def test_pfm_to_pwm_properties_and_dict_support() -> None:
    """pfm_to_pwm computes exact log2-odds and handles (4, L), (L, 4) DataFrame, and dict inputs."""
    uniform_pfm = np.full((4, 6), 25.0)
    pwm_u = pfm_to_pwm(uniform_pfm, pseudocount=1.0)
    assert isinstance(pwm_u, np.ndarray)
    np.testing.assert_allclose(pwm_u, 0.0, atol=1e-12)

    # Transposed DataFrame with columns A,C,G,T
    pfm_ap1 = _consensus_to_pfm("TGAGTCA", weight=99.0)
    df_transposed = pd.DataFrame(pfm_ap1.T, columns=["A", "C", "G", "T"])
    pwm_1 = pfm_to_pwm(pfm_ap1, pseudocount=1.0)
    pwm_2 = pfm_to_pwm(df_transposed, pseudocount=1.0)
    np.testing.assert_allclose(pwm_1, pwm_2, atol=1e-12)
    assert pwm_1.shape == (4, 7)

    # Dict batch conversion
    pwm_dict = pfm_to_pwm({"AP1": pfm_ap1, "UNIF": uniform_pfm}, pseudocount=0.8)
    assert isinstance(pwm_dict, dict)
    assert set(pwm_dict.keys()) == {"AP1", "UNIF"}


def test_scan_sequences_pwm_forward_reverse_complement_and_boundary_safety() -> None:
    """scan_sequences_pwm detects forward & reverse-complement sites and never crosses sequence boundaries."""
    motif_seq = "ACCGGTAC"  # Wait: use asymmetric motif so '+' and '-' strands are distinct!
    asym_motif = "GATTACAG"  # RC is CTGTAATC
    rc_motif = _reverse_complement(asym_motif)
    assert rc_motif == "CTGTAATC"

    pwm = pfm_to_pwm(_consensus_to_pfm(asym_motif, weight=200.0), pseudocount=0.5)
    assert isinstance(pwm, np.ndarray)

    # Seq 0 has forward hit at [5, 13); Seq 1 has RC hit at [3, 11);
    # Seq 2 ends with "GATTA" and Seq 3 starts with "CAG..." -> must NOT trigger cross-boundary hit!
    seqs = [
        "AAAAA" + asym_motif + "TTTTT",
        "CCC" + rc_motif + "GGGG",
        "TTTTTTTTGATTA",
        "CAGAAAAAAAAA",
    ]
    res_both = scan_sequences_pwm(seqs, {"M1": pwm}, min_score="90%", both_strands=True)
    assert isinstance(res_both, MotifScanResult)
    counts_df, hits_df = res_both

    assert counts_df["M1"].tolist() == [1, 1, 0, 0]
    assert len(hits_df) == 2
    assert hits_df.iloc[0]["seq_idx"] == 0
    assert (hits_df.iloc[0]["start"], hits_df.iloc[0]["end"], hits_df.iloc[0]["strand"]) == (5, 13, "+")
    assert hits_df.iloc[1]["seq_idx"] == 1
    assert (hits_df.iloc[1]["start"], hits_df.iloc[1]["end"], hits_df.iloc[1]["strand"]) == (3, 11, "-")

    # Forward strand only skips reverse-complement site in Seq 1
    res_fwd = scan_sequences_pwm(seqs, {"M1": pwm}, min_score="90%", both_strands=False)
    assert res_fwd.counts["M1"].tolist() == [1, 0, 0, 0]


def test_calc_sequence_composition_gc_cpg_and_dinucleotides() -> None:
    """calc_sequence_composition accurately computes GC fraction, CpG O/E, and 16 dinucleotide frequencies."""
    # "ACGTACGT" -> 2 A, 2 C, 2 G, 2 T -> gc_frac = 0.5
    # 7 dinucleotides: AC(2), CG(2), GT(2), TA(1) -> sum = 1.0
    seqs = ["ACGTACGT", "AAAAATTTTT", "CCGGCCGGN"]
    comp_def = calc_sequence_composition(seqs, settings="default")
    comp_r = calc_sequence_composition(seqs, settings="R_default")

    assert list(comp_def.columns) == ["gc_frac", "cpg_oe", *DINUCLEOTIDES]
    assert abs(comp_def.loc[0, "gc_frac"] - 0.5) < 1e-12
    assert abs(comp_def.loc[1, "gc_frac"] - 0.0) < 1e-12
    assert abs(comp_def.loc[2, "gc_frac"] - 1.0) < 1e-12
    assert abs(comp_def.loc[1, "cpg_oe"] - 0.0) < 1e-12

    # Dinucleotide frequencies sum to 1.0 on non-trivial sequences
    np.testing.assert_allclose(
        comp_def[list(DINUCLEOTIDES)].sum(axis=1).to_numpy(), [1.0, 1.0, 1.0], atol=1e-12
    )
    # R_default vs default CpG O/E on "ACGTACGT" (N_mono=8, N_di=7, n_C=2, n_G=2, n_CG=2):
    # R_default: (2 * 8) / (2 * 2) = 4.0; default: (2/7) / (0.25 * 0.25) = 32/7
    assert abs(comp_r.loc[0, "cpg_oe"] - 4.0) < 1e-12
    assert abs(comp_def.loc[0, "cpg_oe"] - (32.0 / 7.0)) < 1e-12


def test_binned_motif_enrichment_recovers_planted_motifs_and_corrects_gc_bias() -> None:
    """Iterative composition weighting eliminates pure GC-artifact motif enrichment while preserving true TF signal."""
    rng = np.random.default_rng(2025)
    n_per_bin = 80
    seq_len = 160

    # Construct 3 bins:
    # - neg_bin: low GC (0.35), planted AT-neutral motif "GATTACAG"
    # - zero_bin: mixed GC spanning [0.30, 0.75] (acts as diverse composition pool)
    # - pos_bin: high GC (0.70), planted causal TF motif "TGAGTCAG" (50% GC), NO planted "CCCCGGGG"
    seqs_neg = _random_dna(rng, n_per_bin, seq_len, gc_prob=0.35)
    seqs_zero = _random_dna(rng, n_per_bin * 2, seq_len, gc_prob=rng.uniform(0.30, 0.75, n_per_bin * 2))
    seqs_pos = _random_dna(rng, n_per_bin, seq_len, gc_prob=0.70)

    causal_pos = "TGAGTCAG"
    causal_neg = "GATTACAG"
    gc_artifact = "CCCCGGGG"  # Pure GC-rich motif arising randomly in high-GC sequences!

    for i in range(60):
        seqs_pos[i] = seqs_pos[i][:40] + causal_pos + seqs_pos[i][48:]
        seqs_neg[i] = seqs_neg[i][:60] + causal_neg + seqs_neg[i][68:]

    all_seqs = seqs_neg + seqs_zero + seqs_pos
    bins = pd.Categorical(
        ["hypo"] * n_per_bin + ["zero"] * (2 * n_per_bin) + ["hyper"] * n_per_bin,
        categories=["hypo", "zero", "hyper"],
        ordered=True,
    )
    pwms = pfm_to_pwm(
        {
            "CAUSAL_POS": _consensus_to_pfm(causal_pos, 150.0),
            "CAUSAL_NEG": _consensus_to_pfm(causal_neg, 150.0),
            "GC_ARTIFACT": _consensus_to_pfm(gc_artifact, 150.0),
        }
    )
    assert isinstance(pwms, dict)

    # Compare unweighted (no sequences passed to hit matrix) vs composition-weighted!
    scan = scan_sequences_pwm(all_seqs, pwms, min_score="85%")
    enr_unweighted = calc_binned_motif_enrichment(
        scan.counts, bins=bins, sequences=None, settings="default"
    )
    enr_weighted = calc_binned_motif_enrichment(
        scan.counts, bins=bins, sequences=all_seqs, settings="default"
    )
    enr_r = calc_binned_motif_enrichment(
        all_seqs, bins=bins, pwms=pwms, min_score="85%", settings="R_default"
    )

    # Unweighted falsely inflates GC_ARTIFACT in GC-rich "hyper" bin; composition weighting suppresses it!
    assert enr_unweighted.log2enr.loc["GC_ARTIFACT", "hyper"] > 1.0
    assert abs(enr_weighted.log2enr.loc["GC_ARTIFACT", "hyper"]) < 0.65
    assert abs(enr_r.log2enr.loc["GC_ARTIFACT", "hyper"]) < 0.55

    # Meanwhile planted CAUSAL_POS in "hyper" and CAUSAL_NEG in "hypo" remain strongly enriched!
    assert enr_weighted.log2enr.loc["CAUSAL_POS", "hyper"] > 2.0
    assert enr_weighted.neg_log10_padj.loc["CAUSAL_POS", "hyper"] > 3.0
    assert enr_weighted.neg_log10_padj.loc["GC_ARTIFACT", "hyper"] < 0.5
    assert enr_weighted.log2enr.loc["CAUSAL_NEG", "hypo"] > 2.0
    assert enr_r.top_motifs("hyper", n=1).iloc[0]["motif"] == "CAUSAL_POS"
    assert enr_r.top_motifs("hypo", n=1).iloc[0]["motif"] == "CAUSAL_NEG"

    # Verify weighted background GC closely matches foreground GC for "hyper"
    comp_all = calc_sequence_composition(all_seqs)
    fg_gc_mean = comp_all.loc[bins == "hyper", "gc_frac"].mean()
    w_hyper = enr_weighted.bin_weights["hyper"]
    bg_gc_weighted = np.average(comp_all.loc[bins != "hyper", "gc_frac"], weights=w_hyper)
    assert abs(fg_gc_mean - bg_gc_weighted) < 0.03


def test_r_default_vs_default_pseudocount_log2enr_shrinkage() -> None:
    """R_default uses pseudocount 8.0 (log2((fg+8)/(bg+8))) vs default pseudocount 1.0."""
    hits = pd.DataFrame({"M1": [1]*10 + [0]*40, "M2": [0]*50})
    bins = pd.Categorical(["fg"]*10 + ["bg"]*40, categories=["fg", "bg"], ordered=True)

    res_def = calc_binned_motif_enrichment(hits, bins=bins, settings="default")
    res_r = calc_binned_motif_enrichment(hits, bins=bins, settings="R_default")

    assert res_def.pseudocount_log2enr == 1.0
    assert res_r.pseudocount_log2enr == 8.0
    # fg sum = 10, bg_weighted = 0 -> default: log2(11/1), R_default: log2(18/8)
    assert abs(res_def.log2enr.loc["M1", "fg"] - np.log2(11.0 / 1.0)) < 1e-12
    assert abs(res_r.log2enr.loc["M1", "fg"] - np.log2(18.0 / 8.0)) < 1e-12


def test_background_modes_zerobin_and_allbins() -> None:
    """calc_binned_motif_enrichment supports background='zeroBin' and 'allBins'."""
    hits = pd.DataFrame(
        {
            "TF_A": [2] * 20 + [0] * 20 + [0] * 20,
            "TF_B": [0] * 20 + [0] * 20 + [2] * 20,
        }
    )
    bins = pd.Categorical(
        ["down"] * 20 + ["zero"] * 20 + ["up"] * 20,
        categories=["down", "zero", "up"],
        ordered=True,
    )

    res_zero = calc_binned_motif_enrichment(hits, bins=bins, background="zeroBin")
    assert res_zero.log2enr.loc["TF_A", "zero"] == 0.0
    assert res_zero.log2enr.loc["TF_A", "down"] > 3.0
    assert res_zero.log2enr.loc["TF_B", "up"] > 3.0

    res_all = calc_binned_motif_enrichment(hits, bins=bins, background="allBins")
    assert isinstance(res_all, BinnedMotifEnrichmentResult)
    tidy = res_all.to_dataframe()
    assert tidy.shape == (6, 8)


def test_rand_lasso_stab_sel_identifies_true_motifs_and_absorbs_gc_bias() -> None:
    """Complementary-Pairs Randomized Lasso selects true causal motifs and rejects GC-confounded decoys."""
    rng = np.random.default_rng(77)
    n_regions = 240
    gc_levels = rng.uniform(0.25, 0.75, size=n_regions)
    seqs = _random_dna(rng, n_regions, 120, gc_prob=gc_levels)
    comp = calc_sequence_composition(seqs)

    # 20 candidate motifs; M_TRUE_1 and M_TRUE_2 drive y_change alongside strong GC bias;
    # M_GC_DECOY is strongly correlated with GC% but has ZERO direct causal effect on y_change!
    x_dict: dict[str, np.ndarray] = {}
    for j in range(17):
        x_dict[f"NOISE_{j}"] = rng.poisson(0.4, size=n_regions).astype(float)
    x_dict["M_TRUE_1"] = rng.binomial(2, 0.35, size=n_regions).astype(float)
    x_dict["M_TRUE_2"] = rng.binomial(2, 0.30, size=n_regions).astype(float)
    x_dict["M_GC_DECOY"] = rng.poisson(3.0 * comp["gc_frac"].to_numpy()).astype(float)

    x_df = pd.DataFrame(x_dict)
    y_change = (
        1.4 * x_df["M_TRUE_1"].to_numpy()
        - 1.3 * x_df["M_TRUE_2"].to_numpy()
        + 3.5 * comp["gc_frac"].to_numpy()
        + rng.normal(0.0, 0.35, size=n_regions)
    )

    stab = rand_lasso_stab_sel(
        x_df,
        y_change=y_change,
        sequences=seqs,
        include_seq_bias=True,
        cutoff=0.8,
        pfer=1.0,
        n_subsamples=80,
        n_alphas=25,
        seed=42,
        settings="default",
    )
    assert isinstance(stab, StabilitySelectionResult)
    assert stab.weakness == 0.5
    assert "gc_frac" in stab.sel_prob.index and "cpg_oe" in stab.sel_prob.index
    assert set(stab.selected_motifs) == {"M_TRUE_1", "M_TRUE_2"}
    assert "M_GC_DECOY" not in stab.selected_motifs
    assert "gc_frac" in stab.selected_features
    assert stab.prob_path_df.shape == (22, 25)

    # Also check R_default mode (weakness=0.8, cutoff=0.8)
    stab_r = rand_lasso_stab_sel(
        x_df,
        y_change=y_change,
        sequences=seqs,
        include_seq_bias=True,
        n_subsamples=80,
        seed=42,
        settings="R_default",
    )
    assert stab_r.weakness == 0.8
    assert "M_TRUE_1" in stab_r.selected_motifs
    assert "M_TRUE_2" in stab_r.selected_motifs


def test_rand_lasso_stab_sel_null_pfer_control_and_constant_columns() -> None:
    """Under pure noise (and zero-variance motif columns), empirical PFER stays bounded by target pfer."""
    false_positives = []
    for trial_seed in range(100, 108):
        rng = np.random.default_rng(trial_seed)
        x_null = rng.poisson(0.5, size=(160, 30)).astype(float)
        x_null[:, 0] = 0.0  # constant all-zero motif column must not divide by zero
        y_null = rng.normal(0.0, 1.0, size=160)

        res = rand_lasso_stab_sel(
            x_null,
            y_null,
            include_seq_bias=False,
            cutoff=0.8,
            pfer=0.5,
            n_subsamples=50,
            seed=trial_seed,
        )
        assert res.sel_prob.iloc[0] == 0.0
        false_positives.append(len(res.selected_motifs))
        if trial_seed == 100:
            summary_df = res.to_dataframe()
            assert summary_df.shape == (30, 4)

    assert float(np.mean(false_positives)) <= 0.5


def test_monalisa_high_level_class_workflow() -> None:
    """MonaLisa OOP workflow runs both binned enrichment and stability selection end-to-end."""
    rng = np.random.default_rng(33)
    seqs = _random_dna(rng, 120, 100, gc_prob=0.5)
    motif = "ACCGTTAG"
    y = rng.normal(0.0, 0.3, size=120)
    for i in range(30):
        seqs[i] = seqs[i][:20] + motif + seqs[i][28:]
        y[i] += 1.5

    pwms = pfm_to_pwm({"TARGET_TF": _consensus_to_pfm(motif, 120.0), "DECOY": _consensus_to_pfm("GGGGCCCC", 120.0)})
    runner = MonaLisa(settings="R_default")
    enr = runner.run_binned_enrichment(seqs, y, pwms=pwms, n_bins=3, min_score="85%")
    stab = runner.run_stability_selection(seqs, y, pwms=pwms, n_subsamples=50, min_score="85%", seed=7)

    assert enr.top_motifs(enr.bins.categories[-1], n=1).iloc[0]["motif"] == "TARGET_TF"
    assert "TARGET_TF" in stab.selected_motifs
    assert hasattr(monalisa, "__version__")


def test_invalid_inputs_raise_clear_errors() -> None:
    """Malformed inputs raise descriptive ValueError / KeyError exceptions."""
    with pytest.raises(ValueError, match="method"):
        bin_genomic_changes([1.0, 2.0, 3.0], method="invalid_method")

    with pytest.raises(ValueError, match="non-negative"):
        pfm_to_pwm(np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="pwms must be provided"):
        calc_binned_motif_enrichment(["ACGTACGT", "TTTTAAAA"], bins=["a", "b"], pwms=None)

    hits = pd.DataFrame({"M1": [1, 0, 1, 0]})
    enr = calc_binned_motif_enrichment(hits, bins=["b1", "b1", "b2", "b2"])
    with pytest.raises(KeyError, match="not found"):
        enr.top_motifs("nonexistent_bin")

    with pytest.raises(ValueError, match="cutoff"):
        rand_lasso_stab_sel(np.ones((10, 2)), np.arange(10, dtype=float), cutoff=0.4)


def test_scan_sequences_pwm_numeric_threshold_ambiguous_n_and_short_seqs() -> None:
    """Numeric min_score, ambiguous 'N' bases, and sequences shorter than PWM width work cleanly."""
    pwm = pfm_to_pwm(_consensus_to_pfm("TGAGTCA", weight=120.0))
    assert isinstance(pwm, np.ndarray)

    seqs = [
        "AC",  # shorter than motif width 7
        "NNNNNNN",  # all ambiguous bases -> score 0.0
        "AAATGAGTCATTT",  # exact consensus match -> score ~ 13.7 bits
    ]
    res = scan_sequences_pwm(seqs, pwm, min_score=10.0, both_strands=True)
    assert res.counts["motif_0"].tolist() == [0, 0, 1]
    assert int(res.counts.to_numpy().sum()) == len(res.hits)


def test_bin_genomic_changes_nonnegative_and_all_zero_inputs() -> None:
    """bin_genomic_changes handles one-sided positive signals and all-zero vectors."""
    pos_vals = np.linspace(0.0, 5.0, 100)
    bins_pos = bin_genomic_changes(pos_vals, method="equalN", n_bins=4, zero_bin=True)
    assert bins_pos.categories[0] == "zero"
    assert len(bins_pos.categories) == 4

    all_z = np.zeros(20)
    bins_z = bin_genomic_changes(all_z, n_bins=3, zero_bin=True)
    assert list(bins_z.categories) == ["zero"]
    assert np.all(bins_z == "zero")


def test_enrichment_to_dataframe_schema_and_pearson_sign_consistency() -> None:
    """log2enr and pearson_resid share identical signs across enriched and depleted bins."""
    hits = np.array([[3, 0], [2, 0], [3, 0], [0, 2], [0, 3], [0, 2]], dtype=float)
    bins = ["bin_A", "bin_A", "bin_A", "bin_B", "bin_B", "bin_B"]
    enr = calc_binned_motif_enrichment(hits, bins=bins, settings="default")
    df = enr.to_dataframe()
    assert list(df.columns) == [
        "motif",
        "bin",
        "log2enr",
        "pearson_resid",
        "neg_log10_p",
        "neg_log10_padj",
        "sum_fg",
        "sum_bg_weighted",
    ]
    np.testing.assert_array_equal(
        np.sign(df["log2enr"].to_numpy()), np.sign(df["pearson_resid"].to_numpy())
    )


def test_rand_lasso_reproducibility_with_seed_and_motif_scan_input() -> None:
    """rand_lasso_stab_sel accepts MotifScanResult directly and is deterministic for a fixed seed."""
    rng = np.random.default_rng(515)
    seqs = _random_dna(rng, 80, 90, gc_prob=0.5)
    motif = "CTAGTCAG"
    y = rng.normal(0.0, 0.25, size=80)
    for i in range(25):
        seqs[i] = seqs[i][:15] + motif + seqs[i][23:]
        y[i] += 1.6

    pwms = pfm_to_pwm({"M_HIT": _consensus_to_pfm(motif, 150.0), "M_BG": _consensus_to_pfm("AAACCCGG", 150.0)})
    scan = scan_sequences_pwm(seqs, pwms, min_score="85%")

    r1 = rand_lasso_stab_sel(scan, y, sequences=seqs, n_subsamples=40, seed=99)
    r2 = rand_lasso_stab_sel(scan, y, sequences=seqs, n_subsamples=40, seed=99)
    pd.testing.assert_series_equal(r1.sel_prob, r2.sel_prob)
    pd.testing.assert_frame_equal(r1.prob_path_df, r2.prob_path_df)
    assert r1.selected_motifs == ["M_HIT"]


def test_binned_motif_enrichment_plot_heatmap_metrics() -> None:
    rng = np.random.default_rng(606)
    seqs = _random_dna(rng, 90, 80, gc_prob=0.5)
    vals = np.linspace(-1.0, 1.0, 90)
    for i in range(60, 90):
        seqs[i] = seqs[i][:10] + "TGAGTCA" + seqs[i][17:]
    pwms = pfm_to_pwm(
        {
            "AP1": _consensus_to_pfm("TGAGTCA", 120.0),
            "CTCF": _consensus_to_pfm("CCGCGAGG", 80.0),
            "GATA": _consensus_to_pfm("GATAAG", 100.0),
        }
    )
    bins = bin_genomic_changes(vals, n_bins=3, min_abs_val=0.2)
    enr = calc_binned_motif_enrichment(seqs, bins=bins, pwms=pwms)

    fig = enr.plot(metric="log2enr", top_n=3, figsize=(7.5, 4.5))
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # main heatmap + colorbar
        assert len(fig.axes[0].images) == 1
    finally:
        plt.close(fig)

    fig_padj = enr.plot(metric="-log10(padj)", top_n=2)
    try:
        assert isinstance(fig_padj, Figure)
        assert len(fig_padj.axes) == 2
    finally:
        plt.close(fig_padj)

    with pytest.raises(ValueError, match="Unknown metric"):
        enr.plot(metric="invalid_metric")


def test_stability_selection_plot_trajectories_and_cutoff() -> None:
    from monalisa import plot_stability_selection

    rng = np.random.default_rng(707)
    seqs = _random_dna(rng, 70, 85, gc_prob=0.5)
    y = rng.normal(0.0, 0.25, size=70)
    for i in range(22):
        seqs[i] = seqs[i][:12] + "CTAGTCAG" + seqs[i][20:]
        y[i] += 1.5
    pwms = pfm_to_pwm(
        {
            "DRIVER": _consensus_to_pfm("CTAGTCAG", 140.0),
            "NOISE1": _consensus_to_pfm("AAACCCGG", 120.0),
            "NOISE2": _consensus_to_pfm("TTTGGGAA", 120.0),
        }
    )
    scan = scan_sequences_pwm(seqs, pwms, min_score="85%")
    stab = rand_lasso_stab_sel(scan, y, sequences=seqs, n_subsamples=30, seed=17)

    fig = stab.plot(top_n=4, figsize=(8, 4.5))
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) >= 4
        assert "Stability Selection" in fig.axes[0].get_title()
    finally:
        plt.close(fig)

    fig_ext, ax_ext = plt.subplots(figsize=(7.5, 4.0))
    try:
        out_fig = plot_stability_selection(stab, top_n=2, ax=ax_ext, title="Custom Stability")
        assert out_fig is fig_ext
        assert ax_ext.get_title() == "Custom Stability"
    finally:
        plt.close(fig_ext)

