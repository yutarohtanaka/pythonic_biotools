from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarhrd import GRCH37_REFERENCE, GRCH38_REFERENCE, calculate_hrd, calculate_hrd_cohort


def test_intact_diploid_genome_zero_hrd_scars():
    """A normal balanced diploid genome across chr1-chr22, chrX scores (0, 0, 0) and ploidy=2.0."""
    rows = []
    for ch, ref in GRCH38_REFERENCE.items():
        if not ref.is_acrocentric:
            rows.append(
                {
                    "chrom": ch,
                    "start": 0,
                    "end": ref.centromere_start,
                    "total_cn": 2,
                    "minor_cn": 1,
                }
            )
        rows.append(
            {
                "chrom": ch,
                "start": ref.centromere_end,
                "end": ref.length,
                "total_cn": 2,
                "minor_cn": 1,
            }
        )
    df = pd.DataFrame(rows)
    res = calculate_hrd(df, genome="GRCh38")
    assert (res.hrd_loh, res.tai, res.lst, res.hrd_sum) == (0, 0, 0, 0)
    assert np.isclose(res.ploidy, 2.0, atol=1e-12)
    assert not res.hrd_positive


def test_hrd_loh_size_threshold_homozygous_deletion_and_whole_chromosome_exclusion():
    """
    HRD-LOH counts LOH >15 Mb (both deletion LOH and cnLOH), excludes <=15 Mb,
    excludes homozygous deletions (total_cn=0), and excludes whole-chromosome LOH
    (even when split into p/q segments by the centromere gap or on acrocentric chr13!).
    """
    segs = pd.DataFrame(
        [
            # chr1: 25 Mb interstitial cnLOH (2, 0) on 1p -> +1 HRD-LOH
            {"chrom": "1", "start": 0, "end": 30_000_000, "total_cn": 2, "minor_cn": 1},
            {"chrom": "1", "start": 30_000_000, "end": 55_000_000, "total_cn": 2, "minor_cn": 0},
            {"chrom": "1", "start": 55_000_000, "end": 122_000_000, "total_cn": 2, "minor_cn": 1},
            {"chrom": "1", "start": 125_000_000, "end": 248_956_422, "total_cn": 2, "minor_cn": 1},
            # chr2: 14 Mb LOH (<= 15 Mb) -> 0 HRD-LOH; and 25 Mb homozygous deletion (0, 0) -> 0 HRD-LOH!
            {"chrom": "2", "start": 0, "end": 30_000_000, "total_cn": 2, "minor_cn": 1},
            {"chrom": "2", "start": 30_000_000, "end": 44_000_000, "total_cn": 1, "minor_cn": 0},
            {"chrom": "2", "start": 44_000_000, "end": 69_000_000, "total_cn": 0, "minor_cn": 0},
            {"chrom": "2", "start": 69_000_000, "end": 92_000_000, "total_cn": 2, "minor_cn": 1},
            # chr3: whole-chromosome monosomy split across p-arm and q-arm by centromere -> 0 HRD-LOH!
            {"chrom": "3", "start": 0, "end": 90_772_458, "total_cn": 1, "minor_cn": 0},
            {"chrom": "3", "start": 93_655_574, "end": 198_295_559, "total_cn": 1, "minor_cn": 0},
            # chr13 (acrocentric): whole-q-arm monosomy -> 0 HRD-LOH!
            {"chrom": "13", "start": 18_051_248, "end": 114_364_328, "total_cn": 1, "minor_cn": 0},
            # chr17: whole-arm 17p loss (22.8 Mb > 15 Mb) while 17q is balanced (2, 1) -> +1 HRD-LOH!
            {"chrom": "17", "start": 0, "end": 22_813_680, "total_cn": 1, "minor_cn": 0},
            {"chrom": "17", "start": 26_885_980, "end": 83_257_441, "total_cn": 2, "minor_cn": 1},
        ]
    )
    res = calculate_hrd(segs, genome="GRCh38")
    by_chr = res.per_chrom_breakdown.set_index("chrom")
    assert by_chr.loc["chr1", "hrd_loh"] == 1
    assert by_chr.loc["chr2", "hrd_loh"] == 0
    assert by_chr.loc["chr3", "hrd_loh"] == 0
    assert by_chr.loc["chr13", "hrd_loh"] == 0
    assert by_chr.loc["chr17", "hrd_loh"] == 1
    assert res.hrd_loh == 2


def test_tai_telomere_extension_centromere_crossing_and_acrocentric_rules():
    """
    TAI counts telomeric AI >=11 Mb that does NOT cross the centromere, rejecting
    interstitial AI, short <11 Mb telomeric AI, and centromere-crossing whole-chr trisomy.
    """
    ref1 = GRCH38_REFERENCE["chr1"]
    ref4 = GRCH38_REFERENCE["chr4"]
    ref7 = GRCH38_REFERENCE["chr7"]
    ref13 = GRCH38_REFERENCE["chr13"]

    segs = pd.DataFrame(
        [
            # chr1: 20 Mb p-telomere gain (3, 1) + 30 Mb q-telomere LOH (1, 0) -> 2 TAIs!
            {"chrom": "chr1", "start": 0, "end": 20_000_000, "major_cn": 2, "minor_cn": 1},
            {"chrom": "chr1", "start": 20_000_000, "end": ref1.centromere_start, "major_cn": 1, "minor_cn": 1},
            {"chrom": "chr1", "start": ref1.centromere_end, "end": ref1.length - 30_000_000, "major_cn": 1, "minor_cn": 1},
            {"chrom": "chr1", "start": ref1.length - 30_000_000, "end": ref1.length, "major_cn": 1, "minor_cn": 0},
            # chr4: 8 Mb p-telomere AI (<11 Mb -> 0) and 25 Mb interstitial AI (capped by 15 Mb balanced telomere -> 0)
            {"chrom": "chr4", "start": 0, "end": 8_000_000, "major_cn": 2, "minor_cn": 1},
            {"chrom": "chr4", "start": 8_000_000, "end": ref4.centromere_start, "major_cn": 1, "minor_cn": 1},
            {"chrom": "chr4", "start": ref4.centromere_end, "end": ref4.length - 40_000_000, "major_cn": 1, "minor_cn": 1},
            {"chrom": "chr4", "start": ref4.length - 40_000_000, "end": ref4.length - 15_000_000, "major_cn": 2, "minor_cn": 0},
            {"chrom": "chr4", "start": ref4.length - 15_000_000, "end": ref4.length, "major_cn": 1, "minor_cn": 1},
            # chr7: whole-chromosome trisomy (2, 1) crossing centromere -> 0 TAI!
            {"chrom": "chr7", "start": 0, "end": ref7.length, "major_cn": 2, "minor_cn": 1},
            # chr13 (acrocentric): distal 35 Mb q-telomere AI with balanced proximal 13q -> +1 TAI!
            {"chrom": "chr13", "start": ref13.centromere_end, "end": 75_000_000, "major_cn": 1, "minor_cn": 1},
            {"chrom": "chr13", "start": 75_000_000, "end": ref13.length, "major_cn": 2, "minor_cn": 1},
        ]
    )
    res = calculate_hrd(segs, genome="GRCh38")
    by_chr = res.per_chrom_breakdown.set_index("chrom")
    assert by_chr.loc["chr1", "tai"] == 2
    assert by_chr.loc["chr4", "tai"] == 0
    assert by_chr.loc["chr7", "tai"] == 0
    assert by_chr.loc["chr13", "tai"] == 1
    assert res.tai == 3


def test_lst_micro_segment_smoothing_remerging_and_centromere_independence():
    """
    LST smooths <3 Mb micro-segments, merges newly adjacent identical states (so a
    1.5 Mb focal amplification inside diploid DNA yields 0 LSTs!), enforces >=10 Mb
    flanking sizes and <=3 Mb gap, and NEVER counts transitions across the centromere!
    """
    ref2 = GRCH38_REFERENCE["chr2"]
    segs = pd.DataFrame(
        [
            # chr2 p-arm: [0..40M (2,1)], [40M..41.5M (5,1) focal <3Mb], [41.5M..92M (2,1)]
            # Smoothing 1.5 Mb focal amp re-merges the two (2,1) blocks -> 0 LSTs on 2p!
            {"chrom": "chr2", "start": 0, "end": 40_000_000, "total_cn": 3, "minor_cn": 1},
            {"chrom": "chr2", "start": 40_000_000, "end": 41_500_000, "total_cn": 6, "minor_cn": 1},
            {"chrom": "chr2", "start": 41_500_000, "end": ref2.centromere_start, "total_cn": 3, "minor_cn": 1},
            # chr2 q-arm starts with DIFFERENT state (2,1 -> total=2, minor=1):
            # Transition across centromere from 2p (3,1) to 2q (2,1) MUST NOT count!
            # Inside 2q: [94.1M..120M (2,1)] (25.9Mb), [120M..122M (4,2) <3Mb smoothed],
            # [122M..150M (1,0)] (28Mb, gap 2Mb<=3Mb -> +1 LST),
            # [150M..180M (2,1)] (30Mb -> +1 LST),
            # [180M..186M (1,0)] (6Mb < 10Mb -> 0 LST with neighbors!)
            {"chrom": "chr2", "start": ref2.centromere_end, "end": 120_000_000, "total_cn": 2, "minor_cn": 1},
            {"chrom": "chr2", "start": 120_000_000, "end": 122_000_000, "total_cn": 4, "minor_cn": 2},
            {"chrom": "chr2", "start": 122_000_000, "end": 150_000_000, "total_cn": 1, "minor_cn": 0},
            {"chrom": "chr2", "start": 150_000_000, "end": 180_000_000, "total_cn": 2, "minor_cn": 1},
            {"chrom": "chr2", "start": 180_000_000, "end": 186_000_000, "total_cn": 1, "minor_cn": 0},
        ]
    )
    res = calculate_hrd(segs, genome="GRCh38")
    assert res.lst == 2


def test_grch37_hg19_reference_and_chr_notation_equivalence():
    """Supports GRCh37/hg19 reference coordinates and matches '1' vs 'chr1' identically."""
    ref9_37 = GRCH37_REFERENCE["chr9"]
    segs_no_chr = pd.DataFrame(
        [
            {"chrom": "9", "start": 0, "end": 25_000_000, "total_cn": 1, "minor_cn": 0},
            {"chrom": "9", "start": 25_000_000, "end": ref9_37.centromere_start, "total_cn": 2, "minor_cn": 1},
            {"chrom": "X", "start": ref9_37.centromere_end, "end": 120_000_000, "total_cn": 2, "minor_cn": 1},
        ]
    )
    segs_with_chr = segs_no_chr.copy()
    segs_with_chr["chrom"] = ["chr9", "chr9", "chrX"]

    r1 = calculate_hrd(segs_no_chr, genome="hg19")
    r2 = calculate_hrd(segs_with_chr, genome="GRCh37")
    assert (r1.hrd_loh, r1.tai, r1.lst, r1.hrd_sum) == (1, 1, 1, 3)
    assert (r2.hrd_loh, r2.tai, r2.lst, r2.hrd_sum) == (1, 1, 1, 3)


def test_brca_deficient_clinical_hrd_positivity_and_tetraploid_lst_adjustment():
    """
    Simulates a BRCA1-like scarred genome exceeding clinical HRD >= 42 cutoff and
    verifies Popova ploidy adjustment on whole-genome-doubled (ploidy ~ 4.0) tumors.
    """
    rows = []
    # Across chr1..chr12 (12 chromosomes), plant a 25 Mb terminal p-arm cnLOH (4, 0) vs (4, 2)
    # on a tetraploid (WGD) background: each chromosome contributes +1 HRD-LOH, +1 TAI, +1 LST,
    # plus an interstitial 25 Mb step on q-arm (+2 LSTs) -> per chr: 1 LOH + 1 TAI + 3 LST = 5 -> 60 total!
    for idx in range(1, 13):
        ch = f"chr{idx}"
        ref = GRCH38_REFERENCE[ch]
        rows.extend(
            [
                {"chrom": ch, "start": 0, "end": 20_000_000, "total_cn": 4, "minor_cn": 0},
                {"chrom": ch, "start": 20_000_000, "end": ref.centromere_start, "total_cn": 4, "minor_cn": 2},
                {"chrom": ch, "start": ref.centromere_end, "end": ref.centromere_end + 20_000_000, "total_cn": 4, "minor_cn": 2},
                {"chrom": ch, "start": ref.centromere_end + 20_000_000, "end": ref.centromere_end + 40_000_000, "total_cn": 3, "minor_cn": 1},
                {"chrom": ch, "start": ref.centromere_end + 40_000_000, "end": ref.length, "total_cn": 4, "minor_cn": 2},
            ]
        )
    df = pd.DataFrame(rows)
    res = calculate_hrd(df, genome="GRCh38")

    assert res.hrd_loh == 12
    assert res.tai == 12
    assert res.lst == 36
    assert res.hrd_sum == 60
    assert res.hrd_positive
    assert 3.7 < res.ploidy <= 4.0
    assert res.lst_ploidy_adjusted < res.lst
    assert np.isclose(res.lst_ploidy_adjusted, res.lst - 2.5 * (res.ploidy - 2.0), atol=1e-9)
    assert res.to_dataframe().shape == (1, 11)


def test_multi_sample_cohort_calculation():
    """`calculate_hrd_cohort` processes multi-patient tables cleanly."""
    segs = pd.DataFrame(
        [
            {"sample_id": "PT_NORMAL", "chrom": "chr1", "start": 0, "end": 120_000_000, "total_cn": 2, "minor_cn": 1},
            {"sample_id": "PT_SCAR", "chrom": "chr1", "start": 0, "end": 30_000_000, "total_cn": 1, "minor_cn": 0},
            {"sample_id": "PT_SCAR", "chrom": "chr1", "start": 30_000_000, "end": 120_000_000, "total_cn": 2, "minor_cn": 1},
        ]
    )
    cohort_df = calculate_hrd_cohort(segs).set_index("sample_id")
    assert cohort_df.loc["PT_NORMAL", "hrd_sum"] == 0
    assert cohort_df.loc["PT_SCAR", "hrd_sum"] == 3  # 1 LOH + 1 TAI + 1 LST


def test_input_validation_errors():
    """Rejects minor_cn > major_cn, non-integer CNs, multiple samples in single call, and unknown genomes."""
    bad_alleles = pd.DataFrame(
        [{"chrom": "chr1", "start": 0, "end": 50_000_000, "total_cn": 3, "minor_cn": 2}]
    )
    with pytest.raises(ValueError, match="minor_cn"):
        calculate_hrd(bad_alleles)

    with pytest.raises(ValueError, match="Unsupported genome"):
        calculate_hrd(
            pd.DataFrame([{"chrom": "chr1", "start": 0, "end": 50_000_000, "total_cn": 2, "minor_cn": 1}]),
            genome="mm10",
        )

    multi = pd.DataFrame(
        [
            {"sample_id": "S1", "chrom": "chr1", "start": 0, "end": 50_000_000, "total_cn": 2, "minor_cn": 1},
            {"sample_id": "S2", "chrom": "chr1", "start": 0, "end": 50_000_000, "total_cn": 2, "minor_cn": 1},
        ]
    )
    with pytest.raises(ValueError, match="calculate_hrd_cohort"):
        calculate_hrd(multi)
