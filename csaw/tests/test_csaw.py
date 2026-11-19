"""Comprehensive unit tests for the standalone csaw package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from csaw import (
    combine_tests,
    filter_windows_global,
    filter_windows_local,
    get_best_test,
    glm_ql_fit_test,
    merge_windows,
    norm_factors,
    norm_offsets,
    window_counts,
)


def _write_synthetic_bam(
    path: Path,
    chrom_sizes: dict[str, int],
    reads: list[tuple[str, int, int, bool, int]],
) -> None:
    """Write single-end alignments (chrom, start, read_len, is_reverse, mapq) to a BAM file."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": c, "LN": length} for c, length in chrom_sizes.items()],
    }
    ref_ids = {c: idx for idx, c in enumerate(chrom_sizes.keys())}
    reads_sorted = sorted(reads, key=lambda r: (ref_ids[r[0]], r[1]))
    with pysam.AlignmentFile(str(path), "wb", header=header) as out_bam:
        for i, (chrom, start, rlen, is_rev, mapq) in enumerate(reads_sorted):
            seg = pysam.AlignedSegment(out_bam.header)
            seg.query_name = f"read_{i}"
            seg.query_sequence = "A" * rlen
            seg.flag = 16 if is_rev else 0
            seg.reference_id = ref_ids[chrom]
            seg.reference_start = start
            seg.mapping_quality = mapq
            seg.cigar = [(0, rlen)]
            seg.query_qualities = pysam.qualitystring_to_array("I" * rlen)
            out_bam.write(seg)


def _write_paired_bam(
    path: Path,
    chrom_sizes: dict[str, int],
    fragments: list[tuple[str, int, int, int]],
) -> None:
    """Write paired-end fragments (chrom, frag_start, frag_end, mapq) to a BAM file."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": c, "LN": length} for c, length in chrom_sizes.items()],
    }
    ref_ids = {c: idx for idx, c in enumerate(chrom_sizes.keys())}
    records: list[tuple[int, int, pysam.AlignedSegment]] = []
    with pysam.AlignmentFile(str(path), "wb", header=header) as out_bam:
        for i, (chrom, f_start, f_end, mapq) in enumerate(fragments):
            rlen = min(35, max(10, (f_end - f_start) // 2))
            rid = ref_ids[chrom]
            r2_start = max(f_start, f_end - rlen)

            r1 = pysam.AlignedSegment(out_bam.header)
            r1.query_name = f"frag_{i}"
            r1.query_sequence = "A" * rlen
            r1.flag = 0x1 | 0x2 | 0x20 | 0x40  # paired, proper, mate rev, read1
            r1.reference_id = rid
            r1.reference_start = f_start
            r1.mapping_quality = mapq
            r1.cigar = [(0, rlen)]
            r1.next_reference_id = rid
            r1.next_reference_start = r2_start
            r1.template_length = f_end - f_start
            r1.query_qualities = pysam.qualitystring_to_array("I" * rlen)

            r2 = pysam.AlignedSegment(out_bam.header)
            r2.query_name = f"frag_{i}"
            r2.query_sequence = "T" * rlen
            r2.flag = 0x1 | 0x2 | 0x10 | 0x80  # paired, proper, rev, read2
            r2.reference_id = rid
            r2.reference_start = r2_start
            r2.mapping_quality = mapq
            r2.cigar = [(0, rlen)]
            r2.next_reference_id = rid
            r2.next_reference_start = f_start
            r2.template_length = -(f_end - f_start)
            r2.query_qualities = pysam.qualitystring_to_array("I" * rlen)

            records.append((rid, f_start, r1))
            records.append((rid, r2_start, r2))

        for _, _, seg in sorted(records, key=lambda x: (x[0], x[1])):
            out_bam.write(seg)


def test_window_counts_settings_default_vs_r_default_and_directional_ext() -> None:
    """Verify directional SE extension and default vs R_default window width & MAPQ presets."""
    # Forward read at 200 extends to [200, 300); reverse read with end=320 extends to [220, 320).
    # Low-MAPQ read at 210 has mapq=5 (filtered in default min_mapq=20, kept in R_default min_mapq=0).
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [200, 285, 210],
            "end": [235, 320, 245],
            "strand": ["+", "-", "+"],
            "mapq": [30, 30, 5],
        }
    )
    chrom_sizes = {"chr1": 600}

    wd_def = window_counts(
        fragments=[df],
        chrom_sizes=chrom_sizes,
        filter_min_total=1,
        settings="default",
    )
    assert wd_def.width == 150
    assert wd_def.spacing == 50
    assert wd_def.totals.tolist() == [2]
    # Window [200, 350) overlaps both high-MAPQ extended reads [200, 300) and [220, 320)
    w_200 = wd_def.to_dataframe().query("start == 200")
    assert len(w_200) == 1
    assert int(w_200["sample_1"].iloc[0]) == 2

    wd_r = window_counts(
        fragments=[df],
        chrom_sizes=chrom_sizes,
        filter_min_total=1,
        settings="R_default",
    )
    assert wd_r.width == 50
    assert wd_r.spacing == 50
    assert wd_r.totals.tolist() == [3]  # mapq=5 read retained under R_default (min_mapq=0)
    w_250_r = wd_r.to_dataframe().query("start == 250")
    assert int(w_250_r["sample_1"].iloc[0]) == 3


def test_window_counts_pysam_se_and_pe_and_bin_mode(tmp_path: Path) -> None:
    """Test real pysam BAM reading in SE, PE ('both'), and 10 kb bin_mode."""
    chrom_sizes = {"chr1": 25000}
    bam_se = tmp_path / "se_sample.bam"
    reads = [("chr1", 1000, 36, False, 40)] * 8 + [("chr1", 1080, 36, True, 40)] * 7
    reads += [("chr1", 15000, 36, False, 40)] * 5
    _write_synthetic_bam(bam_se, chrom_sizes, reads)

    wd_se = window_counts(bam_files=[bam_se], filter_min_total=10, settings="default")
    assert wd_se.totals.tolist() == [20]
    assert wd_se.n_windows >= 2
    assert int(np.max(wd_se.counts)) == 15

    # Bin mode tiles 3 non-overlapping 10kb bins [0, 10000), [10000, 20000), [20000, 25000)
    bins_se = window_counts(bam_files=[bam_se], width=10000, bin_mode=True, settings="default")
    assert bins_se.n_windows == 3
    assert bins_se.counts[:, 0].tolist() == [15, 5, 0]

    # Paired-end BAM test
    bam_pe = tmp_path / "pe_sample.bam"
    frags = [("chr1", 5000, 5180, 35)] * 12
    _write_paired_bam(bam_pe, chrom_sizes, frags)
    wd_pe = window_counts(bam_files=[bam_pe], pe="both", filter_min_total=10, settings="default")
    assert wd_pe.totals.tolist() == [12]
    assert int(np.max(wd_pe.counts)) == 12


def test_filter_windows_global_and_local() -> None:
    """Verify global background median filtering and local flanking neighborhood filtering."""
    rng = np.random.default_rng(123)
    chrom_sizes = {"chr1": 100_000}
    dfs = []
    for s in range(4):
        # Uniform background: 200 reads across 100 kb (~0.3 reads per 150 bp window)
        bg_pos = rng.integers(0, 99_800, size=200)
        # Sharp peak at 20,000 bp (40 reads per library)
        peak_pos = rng.integers(20_000, 20_080, size=40)
        pos = np.concatenate([bg_pos, peak_pos])
        dfs.append(
            pd.DataFrame(
                {
                    "chrom": "chr1",
                    "start": pos,
                    "end": pos + 36,
                    "strand": "+",
                    "mapq": 40,
                }
            )
        )

    win = window_counts(fragments=dfs, chrom_sizes=chrom_sizes, filter_min_total=2, settings="default")
    bg_bins = window_counts(fragments=dfs, chrom_sizes=chrom_sizes, width=10000, bin_mode=True)

    glob_res = filter_windows_global(win, background_bins=bg_bins, min_fc=4.0, settings="default")
    assert glob_res.filtered_data is not None
    kept_df = glob_res.filtered_data.windows
    # Peak windows around 20,000 must survive global enrichment filtering
    assert np.any((kept_df["start"] <= 20_050) & (kept_df["end"] >= 20_050))
    assert len(kept_df) < len(win)

    # Local filtering: compare inner 150 bp counts vs surrounding 1500 bp counts
    surround_counts = win.counts.copy()
    # Suppose first half of windows sit in a high-CNV plateau where surround == 10 * inner,
    # whereas true sharp peaks have surround == inner + tiny flank
    flank_matrix = win.counts * 10
    sharp_mask = (win.windows["start"].to_numpy() >= 19_900) & (
        win.windows["start"].to_numpy() <= 20_100
    )
    flank_matrix[sharp_mask, :] = surround_counts[sharp_mask, :] + 1

    loc_res = filter_windows_local(
        win,
        flank_counts=flank_matrix,
        flank_width_ratio=10.0,
        min_fc=3.0,
        inner_included=True,
        settings="default",
    )
    assert np.all(loc_res.keep[sharp_mask])
    assert not np.any(loc_res.keep[~sharp_mask])


def test_norm_factors_tmm_composition_bias_recovery() -> None:
    """Verify TMM on background bins recovers true sequencing depth under heavy ChIP bias."""
    rng = np.random.default_rng(77)
    n_bins = 500
    true_depth = np.array([1.0, 1.0, 2.0, 2.0])
    # 85% background bins proportional to true_depth; 15% ChIP-enriched bins 8x higher in group 2
    counts = np.zeros((n_bins, 4), dtype=float)
    for j in range(4):
        lam_bg = 50.0 * true_depth[j]
        counts[:425, j] = rng.poisson(lam_bg, size=425)
        lam_peak = (400.0 if j >= 2 else 50.0) * true_depth[j]
        counts[425:, j] = rng.poisson(lam_peak, size=75)

    totals = counts.sum(axis=0)
    # Naive library totals overestimate group 2 depth because of extra ChIP peaks;
    # TMM factors * totals should recover the true 1 : 1 : 2 : 2 background ratio!
    nf = norm_factors(counts, totals=totals, settings="R_default")
    assert np.isclose(np.prod(nf) ** 0.25, 1.0, atol=1e-6)
    eff_libs = totals * nf
    ratio_g2_g1 = np.mean(eff_libs[2:]) / np.mean(eff_libs[:2])
    assert np.isclose(ratio_g2_g1, 2.0, rtol=0.06)


def test_norm_offsets_removes_nonlinear_efficiency_trend() -> None:
    """Verify loess norm_offsets tracks abundance-dependent ChIP efficiency curves."""
    rng = np.random.default_rng(99)
    n_win = 300
    base_mu = np.exp(np.linspace(np.log(15.0), np.log(500.0), n_win))
    # Sample 2 & 3 have a smooth abundance-dependent efficiency boost without true DB
    log_eff_boost = 0.5 * (np.log(base_mu) - np.mean(np.log(base_mu)))
    counts = np.column_stack(
        [
            rng.poisson(base_mu),
            rng.poisson(base_mu),
            rng.poisson(base_mu * np.exp(log_eff_boost)),
            rng.poisson(base_mu * np.exp(log_eff_boost)),
        ]
    )
    totals = np.full(4, 200_000.0)
    offsets = norm_offsets(counts, totals=totals, span=0.35, settings="default")
    assert offsets.shape == (n_win, 4)
    # Row means must equal mean(log(totals))
    assert np.allclose(np.mean(offsets, axis=1), np.mean(np.log(totals)), atol=1e-8)
    # Offset difference between group 2 and group 1 should correlate strongly with log_eff_boost
    est_diff = np.mean(offsets[:, 2:], axis=1) - np.mean(offsets[:, :2], axis=1)
    corr = np.corrcoef(est_diff, log_eff_boost)[0, 1]
    assert corr > 0.95


def test_glm_ql_fit_test_detects_differential_binding_and_controls_null() -> None:
    """Test QL NB F-test accuracy on simulated NB window counts (coef & contrast)."""
    rng = np.random.default_rng(2026)
    n_win = 240
    n_samp = 6  # 3 control vs 3 treatment
    design = np.column_stack([np.ones(n_samp), np.array([0, 0, 0, 1, 1, 1], dtype=float)])
    totals = np.full(n_samp, 1_000_000.0)

    # Generate NB counts with dispersion phi = 0.08:
    # Windows 0..19: +2.0 log2FC up; Windows 20..39: -2.0 log2FC down; Windows 40..239: null
    phi_true = 0.08
    r_nb = 1.0 / phi_true
    base_means = rng.uniform(30.0, 180.0, size=n_win)
    true_lfc = np.zeros(n_win)
    true_lfc[:20] = 2.0
    true_lfc[20:40] = -2.0

    counts = np.zeros((n_win, n_samp), dtype=int)
    for j in range(n_samp):
        fc_mult = 2.0 ** (true_lfc * design[j, 1])
        mu = base_means * fc_mult
        p_nb = r_nb / (r_nb + mu)
        counts[:, j] = rng.negative_binomial(r_nb, p_nb)

    tab_def = glm_ql_fit_test(counts, design=design, totals=totals, settings="default")
    tab_con = glm_ql_fit_test(
        counts, design=design, contrast=[0.0, 1.0], totals=totals, settings="R_default"
    )

    assert np.allclose(tab_def["logFC"], tab_con["logFC"], atol=1e-4)
    assert np.median(tab_def["logFC"].iloc[:20]) > 1.6
    assert np.median(tab_def["logFC"].iloc[20:40]) < -1.6
    # High power on true DB windows and controlled false discovery on null windows
    assert np.mean(tab_def["FDR"].iloc[:40] < 0.05) >= 0.90
    assert np.mean(tab_def["FDR"].iloc[40:] < 0.05) <= 0.05


def test_glm_ql_fit_test_structural_zero_group_and_multi_df_contrast() -> None:
    """Verify stability on all-zero control windows and multi-df ANOVA contrasts."""
    rng = np.random.default_rng(55)
    # 3 groups x 3 replicates = 9 samples
    g1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    g2 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    g3 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    design = np.column_stack([g1, g2, g3])

    counts = rng.poisson(40, size=(60, 9))
    # Window 0 has structural zeros in group 1 and strong enrichment in group 2
    counts[0, :3] = 0
    counts[0, 3:6] = [85, 92, 78]
    counts[0, 6:] = [38, 42, 41]

    c_anova = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    tab = glm_ql_fit_test(counts, design=design, contrast=c_anova, settings="default")
    assert np.all(np.isfinite(tab["F_stat"]))
    assert np.all(np.isfinite(tab["pval"]))
    assert tab["pval"].iloc[0] < 1e-4
    assert "logFC_1" in tab.columns and "logFC_2" in tab.columns


def test_merge_windows_tolerance_and_max_width_splitting() -> None:
    """Test single-linkage window merging within tol and super-cluster splitting at max_width."""
    # Cluster 1: 3 windows within tol=100 -> [100, 450)
    # Cluster 2: separated by 500 bp -> [1000, 7200) spanning 6200 bp > max_width=2500
    starts = [100, 200, 350] + list(range(1000, 7100, 100))
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * len(starts),
            "start": starts,
            "end": [s + 100 for s in starts],
        }
    )
    res = merge_windows(df, tol=100, max_width=2500, settings="default")
    assert res.ids.shape[0] == len(df)
    assert len(np.unique(res.ids[:3])) == 1
    assert int(res.regions["width"].iloc[0]) == 350
    # Super-cluster of width 6200 split into ceil(6200/2500) = 3 sub-regions, each <= 2500 bp
    assert np.all(res.regions["width"] <= 2500)
    assert len(res.regions) == 4


def test_combine_tests_exact_weighted_simes_and_direction_parity() -> None:
    """Verify exact weighted Simes math and default vs R_default directional window counts."""
    # Region 0: 4 up-regulated significant windows -> exact unweighted Simes
    # p = [0.004, 0.006, 0.009, 0.20] -> 4*0.004=0.016, 4/2*0.006=0.012, 4/3*0.009=0.012, 4/4*0.20=0.20 -> 0.012
    # Region 1: weighted Simes with weights [2, 1, 1], p = [0.03, 0.02, 0.50]:
    # sorted p: 0.02 (w=1), 0.03 (w=2), 0.50 (w=1), W=4 -> 0.02*(4/1)=0.08, 0.03*(4/3)=0.04, 0.50*(4/4)=0.50 -> 0.04
    # Region 2: mixed significant region (one strongly up, one strongly down)
    # Region 3: sub-threshold noise region (p = [0.40, 0.60], logFC = [0.8, -0.7])
    merged_ids = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    tab = pd.DataFrame(
        {
            "pval": [0.004, 0.006, 0.009, 0.20, 0.03, 0.02, 0.50, 0.001, 0.002, 0.40, 0.60],
            "logFC": [1.5, 1.2, 0.9, -0.8, -1.1, -1.4, 0.1, 1.8, -1.6, 0.8, -0.7],
        }
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    comb_r = combine_tests(merged_ids, tab, weights=weights, fc_threshold=0.5, settings="R_default")
    assert np.isclose(comb_r.loc[0, "pval"], 0.012, atol=1e-12)
    assert comb_r.loc[0, "direction"] == "up"
    assert int(comb_r.loc[0, "num_up_logFC"]) == 3
    assert int(comb_r.loc[0, "num_down_logFC"]) == 0  # p=0.20 window excluded!

    assert np.isclose(comb_r.loc[1, "pval"], 0.04, atol=1e-12)
    assert comb_r.loc[1, "direction"] == "down"
    # rep_test for region 1 minimizes p_j / w_j: 0.03/2=0.015 vs 0.02/1=0.02 -> index 4
    assert int(comb_r.loc[1, "rep_test"]) == 4

    assert comb_r.loc[2, "direction"] == "mixed"
    assert int(comb_r.loc[2, "num_up_logFC"]) == 1 and int(comb_r.loc[2, "num_down_logFC"]) == 1

    # Region 3 is non-significant (pval = 0.60 > 0.05): R_default strictly reports 0 sig up/down windows
    assert int(comb_r.loc[3, "num_up_logFC"]) == 0
    assert int(comb_r.loc[3, "num_down_logFC"]) == 0

    comb_def = combine_tests(merged_ids, tab, weights=weights, fc_threshold=0.5, settings="default")
    assert np.allclose(comb_def["pval"], comb_r["pval"])
    assert int(comb_def.loc[3, "num_up_logFC"]) == 1


def test_get_best_test_bonferroni_and_by_abundance() -> None:
    """Verify summit window selection and weighted Bonferroni p-values in get_best_test."""
    merged_ids = np.array([0, 0, 0, 1, 1])
    tab = pd.DataFrame(
        {
            "pval": [0.05, 0.002, 0.10, 0.01, 0.04],
            "logFC": [0.4, 2.1, 0.8, -1.5, -0.9],
            "logCPM": [3.2, 6.8, 7.5, 5.1, 4.0],
        }
    )
    best_p = get_best_test(merged_ids, tab, settings="default")
    assert int(best_p.loc[0, "rep_test"]) == 1
    assert np.isclose(best_p.loc[0, "pval"], 0.006, atol=1e-12)  # 3 * 0.002
    assert np.isclose(best_p.loc[0, "rep_logFC"], 2.1)

    best_ab = get_best_test(merged_ids, tab, by_abundance=True, settings="R_default")
    assert int(best_ab.loc[0, "rep_test"]) == 2  # highest logCPM = 7.5
    assert np.isclose(best_ab.loc[0, "pval"], 0.10, atol=1e-12)


def test_end_to_end_csaw_bam_differential_binding_pipeline(tmp_path: Path) -> None:
    """Full end-to-end csaw workflow from synthetic ChIP-seq BAMs to FDR-controlled regions."""
    rng = np.random.default_rng(314)
    chrom_sizes = {"chr1": 200_000}
    bams = []
    for s in range(6):
        is_treated = s >= 3
        reads: list[tuple[str, int, int, bool, int]] = []
        # Uniform Poisson genomic background across 200 kb (20 x 10 kb bins)
        n_bg = int(rng.poisson(1000))
        for pos in rng.integers(100, 199_700, size=n_bg):
            reads.append(("chr1", int(pos), 36, bool(rng.integers(0, 2)), 40))
        # Shared non-differential peaks at 30 kb and 90 kb (Poisson ~40 reads in all samples)
        for center in (30_000, 90_000):
            n_sh = int(rng.poisson(40))
            for pos in rng.integers(center - 40, center + 40, size=n_sh):
                is_rev = bool(rng.integers(0, 2))
                r_pos = int(pos + 90) if is_rev else int(pos)
                reads.append(("chr1", r_pos, 36, is_rev, 40))
        # True gained ChIP peak at 150 kb (Poisson ~65 reads in treated vs ~6 in control)
        n_db = int(rng.poisson(65 if is_treated else 6))
        for pos in rng.integers(149_960, 150_040, size=n_db):
            is_rev = bool(rng.integers(0, 2))
            r_pos = int(pos + 90) if is_rev else int(pos)
            reads.append(("chr1", r_pos, 36, is_rev, 40))

        bam_p = tmp_path / f"chip_{s}.bam"
        _write_synthetic_bam(bam_p, chrom_sizes, reads)
        bams.append(bam_p)

    win = window_counts(bam_files=bams, settings="default")
    bins = window_counts(bam_files=bams, width=10000, bin_mode=True, settings="default")
    filt = filter_windows_global(win, background_bins=bins, min_fc=3.0, settings="default")
    assert filt.filtered_data is not None and len(filt.filtered_data) > 0

    tmm = norm_factors(bins, settings="default")
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    tab = glm_ql_fit_test(filt.filtered_data, design=design, norm_factors=tmm, settings="default")
    merged = merge_windows(filt.filtered_data, tol=100, max_width=5000, settings="default")
    regions = combine_tests(merged, tab, settings="default")

    sig = regions[regions["FDR"] < 0.05]
    assert len(sig) == 1
    assert int(sig["start"].iloc[0]) <= 150_050 <= int(sig["end"].iloc[0])
    assert sig["direction"].iloc[0] == "up"
    assert float(sig["rep_logFC"].iloc[0]) > 2.0


def test_invalid_settings_raises_value_error() -> None:
    """Every primary csaw function must reject unknown settings strings with ValueError."""
    df = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [150]})
    mat = np.ones((4, 4), dtype=float)
    design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
    tab = pd.DataFrame({"pval": [0.01, 0.5], "logFC": [1.2, -0.2], "logCPM": [4.0, 3.0]})

    with pytest.raises(ValueError, match="settings"):
        window_counts(fragments=[df], settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        filter_windows_global(mat, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        filter_windows_local(mat, mat * 5, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        norm_factors(mat, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        norm_offsets(mat, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        glm_ql_fit_test(mat, design=design, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        merge_windows(df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        combine_tests([0, 0], tab, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        get_best_test([0, 0], tab, settings="bad")  # type: ignore[arg-type]
