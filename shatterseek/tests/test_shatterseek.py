from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shatterseek import (
    classify_chromothripsis,
    detect_chromothripsis,
    infer_sv_type,
    max_oscillating_cn_segments,
    pval_chr_breakpoint_enrichment,
    pval_exp_cluster,
    pval_fragment_joins,
)


def _make_oscillating_cn(chrom: str, start: int, step: int, states: list[int]) -> pd.DataFrame:
    rows = []
    curr = start
    for st in states:
        rows.append(
            {
                "chrom": chrom,
                "start": curr,
                "end": curr + step,
                "total_cn": st,
                "minor_cn": 0 if st == 1 else 1,
            }
        )
        curr += step
    return pd.DataFrame(rows)


def test_high_confidence_intrachromosomal_chromothripsis():
    """Detects canonical High-Confidence chromothripsis (>=6 interleaved SVs, >=7 2-state CN oscillations)."""
    # 8 interleaved SVs balanced across DEL (+,-), DUP (-,+), h2hINV (+,+), t2tINV (-,-)
    orientations = [
        ("+", "-"),
        ("-", "+"),
        ("+", "+"),
        ("-", "-"),
        ("+", "-"),
        ("-", "+"),
        ("+", "+"),
        ("-", "-"),
    ]
    rng = np.random.default_rng(12)
    # Generate Poisson-process breakpoints in [20M, 36M] and pair into interleaved chain
    pts = np.sort(20_000_000 + np.cumsum(rng.exponential(scale=900_000, size=16)).astype(int))
    sv_rows = []
    for k in range(8):
        s1, s2 = orientations[k]
        sv_rows.append(
            {
                "chrom1": "chr1",
                "pos1": int(pts[k]),
                "strand1": s1,
                "chrom2": "chr1",
                "pos2": int(pts[k + 8]),
                "strand2": s2,
            }
        )
    sv_df = pd.DataFrame(sv_rows)

    # 9 oscillating CN segments between CN=2 and CN=1 spanning [20M, 33.5M] inside the SV cluster
    cn_df = _make_oscillating_cn("chr1", 20_000_000, 1_500_000, [2, 1, 2, 1, 2, 1, 2, 1, 2])

    res = detect_chromothripsis(sv_df, cn_df, genome="GRCh38")
    assert res.has_chromothripsis
    assert res.has_high_confidence
    assert len(res.clusters) == 1

    c = res.clusters[0]
    assert c.confidence == "High"
    assert c.n_interleaved_svs == 8
    assert (c.n_del, c.n_dup, c.n_h2h_inv, c.n_t2t_inv) == (2, 2, 2, 2)
    assert np.isclose(c.pval_fragment_joins, 1.0, atol=1e-12)
    assert c.pval_exp_cluster > 0.05
    assert c.pval_chr_breakpoint_enrichment < 1e-6
    assert c.max_number_oscillating_cn_segments_2_states == 9
    assert c.interspersed_loh
    assert res.to_dataframe().shape[0] == 1


def test_multi_chromosomal_translocation_high_confidence_chromothripsis():
    """
    PCAWG Multi-chromosomal High-Confidence rule: >=3 interleaved SVs + >=4 TRAs +
    >=7 oscillating 2-state CN segments links partner clusters across chr1 and chr2!
    """
    sv_df = pd.DataFrame(
        [
            # 3 interleaved SVs on chr1 [10M..22M]
            {"chrom1": "1", "pos1": 10_000_000, "strand1": "+", "chrom2": "1", "pos2": 18_000_000, "strand2": "-"},
            {"chrom1": "1", "pos1": 12_000_000, "strand1": "-", "chrom2": "1", "pos2": 20_000_000, "strand2": "+"},
            {"chrom1": "1", "pos1": 14_000_000, "strand1": "+", "chrom2": "1", "pos2": 22_000_000, "strand2": "+"},
            # 3 interleaved SVs on chr2 [50M..62M]
            {"chrom1": "2", "pos1": 50_000_000, "strand1": "-", "chrom2": "2", "pos2": 58_000_000, "strand2": "-"},
            {"chrom1": "2", "pos1": 52_000_000, "strand1": "+", "chrom2": "2", "pos2": 60_000_000, "strand2": "-"},
            {"chrom1": "2", "pos1": 54_000_000, "strand1": "-", "chrom2": "2", "pos2": 62_000_000, "strand2": "+"},
            # 4 interchromosomal translocations bridging chr1 cluster <-> chr2 cluster
            {"chrom1": "1", "pos1": 11_000_000, "strand1": "+", "chrom2": "2", "pos2": 51_000_000, "strand2": "-"},
            {"chrom1": "1", "pos1": 13_000_000, "strand1": "-", "chrom2": "2", "pos2": 53_000_000, "strand2": "+"},
            {"chrom1": "1", "pos1": 15_000_000, "strand1": "+", "chrom2": "2", "pos2": 55_000_000, "strand2": "+"},
            {"chrom1": "1", "pos1": 19_000_000, "strand1": "-", "chrom2": "2", "pos2": 59_000_000, "strand2": "-"},
        ]
    )
    cn_chr1 = _make_oscillating_cn("chr1", 9_000_000, 2_000_000, [2, 1, 2, 1, 2, 1, 2, 1])
    cn_chr2 = _make_oscillating_cn("chr2", 49_000_000, 2_000_000, [2, 1, 2, 1, 2, 1, 2])
    cn_df = pd.concat([cn_chr1, cn_chr2], ignore_index=True)

    res = detect_chromothripsis(sv_df, cn_df)
    df = res.to_dataframe().set_index("chrom")
    assert df.loc["chr1", "confidence"] == "High"
    assert df.loc["chr2", "confidence"] == "High"
    assert df.loc["chr1", "n_tra"] == 4
    assert df.loc["chr1", "interchromosomal_chroms"] == "chr2"
    assert df.loc["chr1", "linked_cluster_ids"] == "chr2_cluster_1"


def test_low_confidence_tier_and_tandem_duplicator_rejection():
    """
    4-6 oscillating CN segments with >=6 random interleaved SVs yields 'Low', whereas
    pure tandem duplications (skewed orientations, pval_fragment_joins << 0.05) yield 'None'!
    """
    orientations_random = [("+", "-"), ("-", "+"), ("+", "+"), ("-", "-"), ("+", "-"), ("-", "+")]
    sv_random = pd.DataFrame(
        [
            {
                "chrom1": "chr3",
                "pos1": 10_000_000 + k * 1_000_000,
                "strand1": orientations_random[k][0],
                "chrom2": "chr3",
                "pos2": 16_500_000 + k * 1_000_000,
                "strand2": orientations_random[k][1],
            }
            for k in range(6)
        ]
    )
    # 5 oscillating CN states -> Low confidence!
    cn_5 = _make_oscillating_cn("chr3", 9_000_000, 3_000_000, [2, 1, 2, 1, 2])
    res_low = detect_chromothripsis(sv_random, cn_5)
    assert res_low.clusters[0].confidence == "Low"
    assert res_low.has_chromothripsis
    assert not res_low.has_high_confidence

    # Tandem duplicator phenotype: 12 interleaved DUPs (-, +) and 0 DEL/INV -> rejected ('None')!
    sv_tdup = pd.DataFrame(
        [
            {
                "chrom1": "chr3",
                "pos1": 10_000_000 + k * 500_000,
                "strand1": "-",
                "chrom2": "chr3",
                "pos2": 17_000_000 + k * 500_000,
                "strand2": "+",
            }
            for k in range(12)
        ]
    )
    cn_9 = _make_oscillating_cn("chr3", 9_000_000, 1_500_000, [2, 3, 2, 3, 2, 3, 2, 3, 2])
    res_tdup = detect_chromothripsis(sv_tdup, cn_9)
    assert res_tdup.clusters[0].pval_fragment_joins < 1e-6
    assert res_tdup.clusters[0].confidence == "None"


def test_nested_vs_interleaved_sv_graph_separation():
    """
    Strictly nested SVs (a1 < a2 < b2 < b1) are NOT interleaved and must NOT merge
    into a single interleaved cluster!
    """
    # 4 concentric nested deletions: [10M, 50M], [15M, 45M], [20M, 40M], [25M, 35M]
    sv_nested = pd.DataFrame(
        [
            {"chrom1": "chr4", "pos1": 10_000_000, "pos2": 50_000_000, "sv_type": "DEL"},
            {"chrom1": "chr4", "pos1": 15_000_000, "pos2": 45_000_000, "sv_type": "DUP"},
            {"chrom1": "chr4", "pos1": 20_000_000, "pos2": 40_000_000, "sv_type": "h2hINV"},
            {"chrom1": "chr4", "pos1": 25_000_000, "pos2": 35_000_000, "sv_type": "t2tINV"},
        ]
    )
    cn_df = _make_oscillating_cn("chr4", 5_000_000, 5_000_000, [2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
    res = detect_chromothripsis(sv_nested, cn_df)
    assert len(res.clusters) == 4
    assert all(c.n_interleaved_svs == 0 for c in res.clusters)
    assert not res.has_chromothripsis


def test_oscillating_cn_2_and_3_states_and_adjacent_duplicate_merging():
    """Verifies exact 2-state and 3-state CN oscillation lengths and duplicate collapsing."""
    # Adjacent duplicate CNs [2, 2, 1, 1, 2, 1, 2, 2] collapse to [2, 1, 2, 1, 2] -> length 5!
    osc2, osc3 = max_oscillating_cn_segments([2, 2, 1, 1, 2, 1, 2, 2])
    assert osc2 == 5
    assert osc3 == 5

    # 3-state polyploid chain [3, 2, 1, 2, 3, 1, 2, 3] -> max 2-state is 3 ([2,1,2]), max 3-state is 8!
    osc2_b, osc3_b = max_oscillating_cn_segments([3, 2, 1, 2, 3, 1, 2, 3])
    assert osc2_b == 3
    assert osc3_b == 8

    # Constant CN has 0 oscillations
    assert max_oscillating_cn_segments([2, 2, 2]) == (0, 0)


def test_sv_orientation_inference_and_reversed_coordinate_swapping():
    """Infers DEL, DUP, h2hINV, t2tINV, TRA and swaps strands when pos1 > pos2."""
    assert infer_sv_type("1", 100, "+", "1", 500, "-")[-1] == "DEL"
    assert infer_sv_type("1", 100, "-", "1", 500, "+")[-1] == "DUP"
    assert infer_sv_type("1", 100, "+", "1", 500, "+")[-1] == "h2hINV"
    assert infer_sv_type("1", 100, "-", "1", 500, "-")[-1] == "t2tINV"
    assert infer_sv_type("1", 500, "-", "2", 100, "+")[-1] == "TRA"

    # Reversed intrachromosomal coordinates: (500, '-') -> (100, '+') swaps to (100, '+') -> (500, '-') == DEL!
    c1, p1, s1, c2, p2, s2, st = infer_sv_type("chr5", 500, "-", "chr5", 100, "+")
    assert (c1, p1, s1, p2, s2, st) == ("chr5", 100, "+", 500, "-", "DEL")


def test_statistical_criteria_methods_chisq_gtest_exact_and_anderson_darling():
    """Tests exact/G-test fragment joins and KS/Anderson-Darling exponential spacing tests."""
    # Balanced counts -> p == 1.0 across chisq, gtest, exact
    for m in ("chisq", "gtest", "exact"):
        assert np.isclose(pval_fragment_joins([3, 3, 3, 3], method=m), 1.0, atol=1e-12)

    # Skewed counts -> small p across chisq, gtest, exact
    p_chi = pval_fragment_joins([12, 0, 0, 0], method="chisq")
    p_g = pval_fragment_joins([12, 0, 0, 0], method="gtest")
    p_ex = pval_fragment_joins([12, 0, 0, 0], method="exact")
    assert p_chi < 1e-6 and p_g < 1e-6 and p_ex < 1e-6

    # Exponential breakpoints pass both KS and Stephens Anderson-Darling (> 0.05)
    rng = np.random.default_rng(77)
    exp_bp = np.cumsum(rng.exponential(scale=50_000, size=30))
    assert pval_exp_cluster(exp_bp, method="ks") > 0.05
    assert pval_exp_cluster(exp_bp, method="anderson") > 0.05

    # Bimodal/fragile-site spike spacings (25 tiny 10-bp gaps + 1 giant 10-Mb gap) fail both!
    bimodal_bp = np.cumsum(np.array([10.0] * 25 + [10_000_000.0]))
    assert pval_exp_cluster(bimodal_bp, method="ks") < 1e-4
    assert pval_exp_cluster(bimodal_bp, method="anderson") < 1e-3

    # Binomial enrichment test
    p_enr = pval_chr_breakpoint_enrichment(15, 16, 100_000_000, 3_000_000_000)
    assert p_enr < 1e-15


def test_input_validation_and_empty_sv_edge_case():
    """Cleanly handles 0-SV normal genomes and raises on invalid inputs."""
    cn_df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [248_000_000], "total_cn": [2]})
    empty_sv = pd.DataFrame(columns=["chrom1", "pos1", "strand1", "chrom2", "pos2", "strand2"])
    res = detect_chromothripsis(empty_sv, cn_df)
    assert not res.has_chromothripsis
    assert len(res.clusters) == 0

    with pytest.raises(ValueError, match="pos1 == pos2"):
        infer_sv_type("chr1", 1000, "+", "chr1", 1000, "-")

    with pytest.raises(ValueError, match="alpha"):
        classify_chromothripsis(
            n_interleaved_svs=8,
            n_tra=0,
            max_oscillating_cn_2_states=8,
            pval_fragment_joins=0.5,
            pval_chr_breakpoint_enrichment=1e-5,
            pval_exp_cluster=0.5,
            alpha=1.5,
        )
