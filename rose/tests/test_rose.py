from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from rose import call_rose_super_enhancers, load_bed_intervals


def test_rose_super_enhancer_stitching_and_threshold():
    tss_df = pd.DataFrame([{"chrom": "chr8", "tss": 127_736_000, "gene_name": "GENE_A", "gene_type": "protein_coding"}])
    peaks = [
        {"chrom": "chr8", "start": 127_735_500, "end": 127_736_500, "score": 500.0},
        {"chrom": "chr8", "start": 127_800_000, "end": 127_804_000, "score": 320.0},
        {"chrom": "chr8", "start": 127_810_000, "end": 127_815_000, "score": 410.0},
        {"chrom": "chr8", "start": 127_822_000, "end": 127_828_000, "score": 380.0},
    ] + [
        {"chrom": "chr1", "start": 1_000_000 + i * 50_000, "end": 1_000_500 + i * 50_000, "score": 10.0 + (i % 5)}
        for i in range(40)
    ]
    res = call_rose_super_enhancers(pd.DataFrame(peaks), tss_df=tss_df, stitch_bp=12_500, tss_exclusion_bp=2_500)
    assert len(res.stitched_regions) == 41
    assert res.super_enhancers.iloc[0]["n_constituent_peaks"] == 3


def test_rose_broad_promoter_exclusion_flat_signal_and_merged_blacklist(tmp_path, monkeypatch):
    # 1. Missing bigWig / control bigWig must raise FileNotFoundError instead of silently falling back to SPM
    peaks_min = pd.DataFrame([{"chrom": "chr1", "start": 10_000, "end": 11_000, "score": 20.0}])
    with pytest.raises(FileNotFoundError, match="bigwig_path"):
        call_rose_super_enhancers(peaks_min, bigwig_path=tmp_path / "nonexistent_chip.bw")

    chip_bw = tmp_path / "chip.bw"
    chip_bw.write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match="control_bigwig_path"):
        call_rose_super_enhancers(peaks_min, bigwig_path=chip_bw, control_bigwig_path=tmp_path / "missing_ctrl.bw")
    with pytest.raises(ValueError, match="control_bigwig_path requires bigwig_path"):
        call_rose_super_enhancers(peaks_min, control_bigwig_path=chip_bw)

    # 2. Control bigWig subtraction via mocked pyBigWig
    ctrl_bw = tmp_path / "ctrl.bw"
    ctrl_bw.write_bytes(b"stub")

    class _FakeBW:
        def __init__(self, path: str):
            self.is_ctrl = "ctrl" in path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stats(self, chrom: str, s: int, e: int, type: str = "mean"):
            if self.is_ctrl:
                return [2.0]
            return [20.0 if chrom == "chr8" else 2.5]

    fake_mod = types.ModuleType("pyBigWig")
    fake_mod.open = lambda p: _FakeBW(p)
    monkeypatch.setitem(sys.modules, "pyBigWig", fake_mod)

    bw_peaks = pd.DataFrame(
        [{"chrom": "chr8", "start": 100_000, "end": 110_000, "score": 1.0}]
        + [{"chrom": "chr1", "start": 1_000_000 + i * 50_000, "end": 1_001_000 + i * 50_000, "score": 1.0} for i in range(25)]
    )
    res_bw = call_rose_super_enhancers(bw_peaks, bigwig_path=chip_bw, control_bigwig_path=ctrl_bw)
    assert res_bw.signal_unit == "bp_x_bigwig_bg_subtracted"
    assert len(res_bw.super_enhancers) == 1
    assert res_bw.super_enhancers.iloc[0]["signal"] == pytest.approx(10_000 * (20.0 - 2.0))

    # 3. Broad promoter peak (width 6 kb > 2.5 kb) overlapping TSS must be excluded by default so flanking
    # distal enhancers (16 kb apart without the promoter bridge) do NOT stitch across the TSS!
    tss_df = pd.DataFrame([{"chrom": "chr2", "tss": 500_000, "gene_name": "MYGENE"}])
    promoter_bridge_peaks = pd.DataFrame(
        [
            {"chrom": "chr2", "start": 491_000, "end": 492_000, "score": 50.0},  # upstream enhancer
            {"chrom": "chr2", "start": 497_000, "end": 503_000, "score": 400.0},  # 6 kb broad promoter spanning TSS
            {"chrom": "chr2", "start": 508_000, "end": 509_000, "score": 50.0},  # downstream enhancer (508k - 492k = 16k > 12.5k!)
        ]
    )
    res_excl = call_rose_super_enhancers(promoter_bridge_peaks, tss_df=tss_df, stitch_bp=12_500, tss_exclusion_bp=2_500)
    assert len(res_excl.stitched_regions) == 2  # separated into 2 distal enhancers!
    res_allow = call_rose_super_enhancers(
        promoter_bridge_peaks,
        tss_df=tss_df,
        stitch_bp=12_500,
        tss_exclusion_bp=2_500,
        allow_broad_promoter_peaks=True,
    )
    assert len(res_allow.stitched_regions) == 1  # stitched into 1 domain when broad promoter peak is kept

    # 4. Flat constant signals must call ZERO super-enhancers (strict sig > cutoff_val)
    flat_peaks = pd.DataFrame(
        [{"chrom": "chr1", "start": 100_000 + i * 50_000, "end": 101_000 + i * 50_000, "score": 25.0} for i in range(30)]
    )
    res_flat = call_rose_super_enhancers(flat_peaks)
    assert len(res_flat.super_enhancers) == 0

    # 5. load_bed_intervals merges overlapping intervals & >5 Mb wide centromeric blacklist blocks work cleanly
    bl_df = pd.DataFrame(
        [
            {"chrom": "chr1", "start": 100, "end": 500},
            {"chrom": "chr1", "start": 300, "end": 12_000_000},  # 12 Mb wide block (> 5 Mb!)
            {"chrom": "chr1", "start": 15_000_000, "end": 15_001_000},
        ]
    )
    merged = load_bed_intervals(bl_df)
    assert merged["chr1"] == [(100, 12_000_000), (15_000_000, 15_001_000)]

    test_peaks_bl = pd.DataFrame(
        [
            {"chrom": "chr1", "start": 11_500_000, "end": 11_500_500, "score": 100.0},  # 11.5 Mb inside wide blacklist!
            {"chrom": "chr1", "start": 13_000_000, "end": 13_000_500, "score": 100.0},  # outside blacklist
        ]
    )
    res_bl = call_rose_super_enhancers(test_peaks_bl, blacklist=bl_df)
    assert len(res_bl.stitched_regions) == 1
    assert int(res_bl.stitched_regions.iloc[0]["start"]) == 13_000_000


def test_rose_settings_validation():
    peaks = pd.DataFrame([{"chrom": "chr1", "start": 1000, "end": 2000, "score": 10.0}])
    with pytest.raises(ValueError, match="settings"):
        call_rose_super_enhancers(peaks, settings="invalid")  # type: ignore[arg-type]


def test_rose_r_default_chry_containment_unstitching_and_cutoff():
    """
    Verify Young Lab ROSE_main.py & ROSE_callSuper.R parity under settings='R_default':
    1. Excludes chrY when standard_chroms_only=True.
    2. TSS exclusion (getContainers) only drops peaks strictly contained in [tss - W, tss + W].
    3. Unstitches stitched domains spanning > 2 distinct gene TSS ± 50 bp windows.
    4. Uses calculate_cutoff minimizing y - slope * x with slope = (max - min) / N, x = 1..N.
    """
    # 1 & 2: chrY exclusion + getContainers TSS containment
    tss_df = pd.DataFrame(
        [
            {"chrom": "chr1", "tss": 100_000, "gene_name": "G1"},
            # Three distinct genes spaced by 8 kb on chr2 for unstitching test (>2 genes)
            {"chrom": "chr2", "tss": 500_000, "gene_name": "CLUST_A"},
            {"chrom": "chr2", "tss": 508_000, "gene_name": "CLUST_B"},
            {"chrom": "chr2", "tss": 516_000, "gene_name": "CLUST_C"},
            # Two distinct genes (with 2 isoforms of ISO_A -> still only 2 distinct genes <= 2!) on chr3
            {"chrom": "chr3", "tss": 800_000, "gene_name": "ISO_A"},
            {"chrom": "chr3", "tss": 808_000, "gene_name": "ISO_A"},
            {"chrom": "chr3", "tss": 816_000, "gene_name": "ISO_B"},
        ]
    )
    peaks = pd.DataFrame(
        [
            # chrY peak: kept in default, excluded in R_default
            {"chrom": "chrY", "start": 50_000, "end": 51_000, "score": 30.0},
            # chr1 fully contained in [97_500, 102_500] -> excluded in BOTH modes
            {"chrom": "chr1", "start": 99_000, "end": 101_000, "score": 50.0},
            # chr1 partially overlapping [97_500, 102_500] (starts at 96_000 < 97_500) ->
            # excluded in default, KEPT in R_default!
            {"chrom": "chr1", "start": 96_000, "end": 99_000, "score": 50.0},
            # chr2: 4 intergenic peaks (strictly outside all ±2.5 kb promoter windows) stitching
            # from 494k..521k across 3 distinct TSS±50bp windows (CLUST_A, CLUST_B, CLUST_C).
            # In default: 1 stitched region of 4 peaks; in R_default: unstitched back to 4 single peaks!
            {"chrom": "chr2", "start": 494_000, "end": 495_000, "score": 40.0},
            {"chrom": "chr2", "start": 503_500, "end": 504_500, "score": 40.0},
            {"chrom": "chr2", "start": 511_500, "end": 512_500, "score": 40.0},
            {"chrom": "chr2", "start": 520_000, "end": 521_000, "score": 40.0},
            # chr3: 4 intergenic peaks stitching across 3 TSS rows belonging to only 2 distinct genes (<=2)
            # Stays stitched (n_constituent_peaks == 4) in R_default!
            {"chrom": "chr3", "start": 794_000, "end": 795_000, "score": 40.0},
            {"chrom": "chr3", "start": 803_500, "end": 804_500, "score": 40.0},
            {"chrom": "chr3", "start": 811_500, "end": 812_500, "score": 40.0},
            {"chrom": "chr3", "start": 820_000, "end": 821_000, "score": 40.0},
        ]
    )

    res_def = call_rose_super_enhancers(
        peaks, tss_df=tss_df, stitch_bp=12_500, tss_exclusion_bp=2_500, settings="default"
    )
    res_r = call_rose_super_enhancers(
        peaks, tss_df=tss_df, stitch_bp=12_500, tss_exclusion_bp=2_500, settings="R_default"
    )

    assert "chrY" in set(res_def.stitched_regions["chrom"])
    assert "chrY" not in set(res_r.stitched_regions["chrom"])

    assert "chr1" not in set(res_def.stitched_regions["chrom"])
    assert "chr1" in set(res_r.stitched_regions["chrom"])

    chr2_def = res_def.stitched_regions.loc[res_def.stitched_regions["chrom"] == "chr2"]
    chr2_r = res_r.stitched_regions.loc[res_r.stitched_regions["chrom"] == "chr2"]
    assert len(chr2_def) == 1 and int(chr2_def.iloc[0]["n_constituent_peaks"]) == 4
    assert len(chr2_r) == 4 and set(chr2_r["n_constituent_peaks"]) == {1}

    chr3_r = res_r.stitched_regions.loc[res_r.stitched_regions["chrom"] == "chr3"]
    assert len(chr3_r) == 1 and int(chr3_r.iloc[0]["n_constituent_peaks"]) == 4

    # 4. Cutoff formula difference: 10 isolated peaks with raw scores [100]*8 + [112, 200]
    # default (x_norm = i/9, y_norm = sig/max_sig) picks idx=8 (112) -> 1 SE (200)
    # R_default (slope = (max - min)/N = 10, x = 1..10) picks idx=7 (100) -> 2 SEs (112, 200)
    cutoff_scores = [100.0] * 8 + [112.0, 200.0]
    cutoff_peaks = pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "start": 1_000_000 + i * 50_000,
                "end": 1_001_000 + i * 50_000,
                "score": sc,
            }
            for i, sc in enumerate(cutoff_scores)
        ]
    )
    c_def = call_rose_super_enhancers(cutoff_peaks, settings="default")
    c_r = call_rose_super_enhancers(cutoff_peaks, settings="R_default")
    assert len(c_def.super_enhancers) == 1
    assert len(c_r.super_enhancers) == 2


def test_rose_result_plot_hockey_stick_returns_figure() -> None:
    tss_df = pd.DataFrame(
        [
            {"chrom": "chr8", "tss": 127_736_000, "gene_name": "MYC"},
            {"chrom": "chr1", "tss": 950_000, "gene_name": "CTRL1"},
        ]
    )
    peaks = [
        {"chrom": "chr8", "start": 127_800_000, "end": 127_804_000, "score": 360.0},
        {"chrom": "chr8", "start": 127_810_000, "end": 127_815_000, "score": 420.0},
    ] + [
        {
            "chrom": "chr1",
            "start": 1_000_000 + i * 50_000,
            "end": 1_000_500 + i * 50_000,
            "score": 12.0 + (i % 4),
        }
        for i in range(30)
    ]
    res = call_rose_super_enhancers(pd.DataFrame(peaks), tss_df=tss_df)
    fig = res.plot(top_n_labels=3, figsize=(7, 5))
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.collections) == 2
        assert len(ax.lines) >= 2
        assert any("MYC" in txt.get_text() for txt in ax.texts)
    finally:
        plt.close(fig)


def test_plot_rose_hockey_stick_accepts_dataframe_and_custom_ax() -> None:
    from rose import plot_rose_hockey_stick

    df_raw = pd.DataFrame(
        {
            "chrom": ["chr1"] * 20,
            "start": [i * 50_000 for i in range(20)],
            "end": [i * 50_000 + 1000 for i in range(20)],
            "signal": [float(10 + i) for i in range(18)] + [180.0, 320.0],
            "nearest_gene": [f"G{i}" for i in range(18)] + ["BCL2", "BRD4"],
        }
    )
    fig0 = plot_rose_hockey_stick(df_raw, top_n_labels=2)
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    plt.close(fig0)

    fig_ext, ax_ext = plt.subplots(figsize=(6.5, 4.5))
    try:
        fig_ret = plot_rose_hockey_stick(
            df_raw, ax=ax_ext, top_n_labels=2, title="Custom Hockey Stick"
        )
        assert isinstance(fig_ret, Figure)
        assert fig_ret is fig_ext
        assert ax_ext.get_title() == "Custom Hockey Stick"
        assert any("BRD4" in txt.get_text() for txt in ax_ext.texts)
    finally:
        plt.close(fig_ext)

