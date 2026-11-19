from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    # Matched total accessibility across peaks 0..24 (open in cells 10..19) and peaks 25..49 (open in cells 0..9)
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
