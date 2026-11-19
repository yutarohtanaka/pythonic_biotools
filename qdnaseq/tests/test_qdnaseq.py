"""
Comprehensive unit tests for `qdnaseq` and bundled `ACE` covering `pysam` BAM
read binning (`min_mapq=30` vs `minMapq=37`), genomic QC filtering, iterative
2D `(GC, mappability)` LOESS bias correction, outlier smoothing, Anscombe `log2`
vs `sqrt` CBS segmentation, empirical-Bayes 5-state calling, `ACE::squaremodel`
cellularity/ploidy fitting, end-to-end `QDNAseqPipeline`, and `"default"` vs
`"R_default"` parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from qdnaseq import (
    ACEResult,
    QDNAseqData,
    QDNAseqPipeline,
    QDNAseqPipelineResult,
    apply_filters,
    bin_read_counts,
    call_bins,
    correct_counts,
    estimate_correction,
    normalize_bins,
    run_ace,
    segment_bins,
    smooth_outlier_bins,
    squaremodel,
)


def _simulate_lpwgs_dataset(
    *,
    cellularity: float = 0.50,
    seg_cn_specs: list[tuple[str, int, int]] | None = None,
    seed: int = 42,
    mean_reads: float = 400.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Simulate shallow/low-pass WGS binned counts with realistic nonlinear 2D
    `(GC, mappability)` bias + Poisson/overdispersed noise + blacklisted/sex bins.
    """
    rng = np.random.default_rng(seed)
    if seg_cn_specs is None:
        # Equal 80 bins per segment: C in {2, 2, 3, 1} -> true ploidy = 2.0
        seg_cn_specs = [
            ("chr1", 2, 80),
            ("chr2", 2, 80),
            ("chr3", 3, 80),
            ("chr4", 1, 80),
        ]

    total_auto_bins = sum(nb for _, _, nb in seg_cn_specs)
    true_ploidy = float(
        sum(cn * nb for _, cn, nb in seg_cn_specs) / max(total_auto_bins, 1)
    )
    bulk_denom = cellularity * true_ploidy + 2.0 * (1.0 - cellularity)

    bin_rows = []
    counts_list = []

    for chrom, cn_val, n_bins in seg_cn_specs:
        starts = np.arange(1, 1 + n_bins * 100_000, 100_000, dtype=np.int64)
        ends = starts + 99_999
        gc = rng.uniform(32.0, 62.0, size=n_bins)
        mapp = rng.uniform(65.0, 100.0, size=n_bins)

        # Smooth 2D (GC, mappability) bias factor in [0.55, 1.45]
        gc_z = (gc - 45.0) / 10.0
        map_z = (mapp - 85.0) / 15.0
        bias_2d = np.exp(0.28 * gc_z - 0.14 * (gc_z**2) + 0.18 * map_z)

        rel_cn = (cellularity * cn_val + 2.0 * (1.0 - cellularity)) / bulk_denom
        expected_cnt = mean_reads * bias_2d * rel_cn
        sim_cnt = rng.poisson(np.maximum(expected_cnt, 1.0)).astype(np.float64)

        for i in range(n_bins):
            bin_rows.append(
                {
                    "chrom": chrom,
                    "start": int(starts[i]),
                    "end": int(ends[i]),
                    "gc": float(gc[i]),
                    "mappability": float(mapp[i]),
                    "blacklist": 0.0,
                    "residual": float(rng.normal(0.0, 0.7)),
                }
            )
            counts_list.append(sim_cnt[i])

    # Append problematic QC bins that `apply_filters` must exclude:
    # blacklisted bin, low-mappability bin, high-residual bin, and chrX bin
    qc_extras = [
        ("chr1", 90_000_001, 90_100_000, 45.0, 95.0, 25.0, 0.2, 5000.0),
        ("chr2", 90_000_001, 90_100_000, 45.0, 30.0, 0.0, 0.2, 40.0),
        ("chr3", 90_000_001, 90_100_000, 45.0, 95.0, 0.0, 6.5, 3000.0),
        ("chrX", 1, 100_000, 44.0, 95.0, 0.0, 0.1, 220.0),
    ]
    for ch, st, en, g_v, m_v, bl_v, res_v, cnt_v in qc_extras:
        bin_rows.append(
            {
                "chrom": ch,
                "start": st,
                "end": en,
                "gc": g_v,
                "mappability": m_v,
                "blacklist": bl_v,
                "residual": res_v,
            }
        )
        counts_list.append(cnt_v)

    bins_df = pd.DataFrame(bin_rows)
    counts_df = pd.DataFrame({"cfdna_01": counts_list})
    return bins_df, counts_df, true_ploidy


def test_pysam_bam_bin_read_counts_mapq_default_vs_r_default(tmp_path: Path):
    bam_path = tmp_path / "sample_lpwgs.bam"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"LN": 500_000, "SN": "chr1"}]}

    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as outf:
        # Bin 1 [1..100000]: 5 reads with MAPQ=42, 3 reads with MAPQ=33, 2 duplicates with MAPQ=60
        for i in range(5):
            a = pysam.AlignedSegment(outf.header)
            a.query_name = f"hi_{i}"
            a.query_sequence = "A" * 50
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 1000 + i * 100
            a.mapping_quality = 42
            a.cigar = ((0, 50),)
            outf.write(a)
        for i in range(3):
            a = pysam.AlignedSegment(outf.header)
            a.query_name = f"mid_{i}"
            a.query_sequence = "C" * 50
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 5000 + i * 100
            a.mapping_quality = 33  # >=30 (default) but <37 (R_default)
            a.cigar = ((0, 50),)
            outf.write(a)
        for i in range(2):
            a = pysam.AlignedSegment(outf.header)
            a.query_name = f"dup_{i}"
            a.query_sequence = "G" * 50
            a.flag = 1024  # PCR duplicate -> excluded in both modes
            a.reference_id = 0
            a.reference_start = 9000 + i * 100
            a.mapping_quality = 60
            a.cigar = ((0, 50),)
            outf.write(a)

    pysam.index(str(bam_path))

    bins_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1, 100_001],
            "end": [100_000, 200_000],
            "gc": [45.0, 48.0],
            "mappability": [95.0, 95.0],
        }
    )

    qd_def = bin_read_counts(bins_df, bam_paths=bam_path, settings="default")
    qd_r = bin_read_counts(bins_df, bam_paths=bam_path, settings="R_default")

    assert float(qd_def.counts["sample_lpwgs"].iloc[0]) == 8.0
    assert float(qd_r.counts["sample_lpwgs"].iloc[0]) == 5.0
    assert float(qd_def.counts["sample_lpwgs"].iloc[1]) == 0.0


def test_apply_filters_excludes_blacklist_low_mappability_residual_and_sex_chroms():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(seed=10)
    qd = bin_read_counts(bins_df, counts_df=counts_df)
    assert qd.n_used_bins == len(bins_df)

    qd_filt = apply_filters(
        qd, residual=True, blacklist=True, mappability=50.0, chromosomes=("X", "Y", "MT")
    )
    # Exactly the 4 appended QC artifact/chrX bins must be filtered out
    assert qd_filt.n_used_bins == len(bins_df) - 4
    assert not bool(qd_filt.bins.iloc[-1]["use"])  # chrX
    assert not bool(qd_filt.bins.iloc[-2]["use"])  # residual=6.5
    assert not bool(qd_filt.bins.iloc[-3]["use"])  # mappability=30
    assert not bool(qd_filt.bins.iloc[-4]["use"])  # blacklist=25


def test_2d_loess_correction_removes_gc_and_mappability_bias():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(cellularity=0.50, seed=17, mean_reads=600.0)
    qd = apply_filters(bin_read_counts(bins_df, counts_df=counts_df))

    # Measure raw relative dispersion on diploid chromosomes chr1 + chr2 before vs after 2D LOESS
    dip_mask = qd.bins["chrom"].isin(["chr1", "chr2"]).to_numpy() & qd.bins["use"].to_numpy()
    raw_dip = qd.counts["cfdna_01"].to_numpy()[dip_mask]
    raw_cv = float(np.std(raw_dip) / np.mean(raw_dip))

    qd_est = estimate_correction(qd, max_iter=3, cutoff=4.0, span=0.65)
    qd_cor = correct_counts(qd_est, method="median")
    cor_dip = qd_cor.copynumber["cfdna_01"].to_numpy()[dip_mask]
    cor_cv = float(np.std(cor_dip) / np.mean(cor_dip))

    assert cor_cv < 0.35 * raw_cv
    assert abs(float(np.median(cor_dip)) - 1.0) < 0.04


def test_smooth_outlier_bins_removes_singleton_spike():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(cellularity=0.50, seed=23, mean_reads=600.0)
    qd = normalize_bins(correct_counts(apply_filters(bin_read_counts(bins_df, counts_df=counts_df))))
    assert qd.copynumber is not None

    # Inject isolated 8x spike at bin 20 of chr1
    qd.copynumber.loc[20, "cfdna_01"] = 8.0
    qd_sm = smooth_outlier_bins(qd, max_sd=4.0, window=5)
    assert qd_sm.copynumber is not None
    assert abs(float(qd_sm.copynumber.loc[20, "cfdna_01"]) - 1.0) < 0.15


def test_segment_bins_default_vs_r_default_and_call_bins_probabilities():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(cellularity=0.55, seed=31, mean_reads=650.0)
    qd_base = smooth_outlier_bins(
        normalize_bins(correct_counts(apply_filters(bin_read_counts(bins_df, counts_df=counts_df))))
    )

    for mode in ("default", "R_default"):
        qd_seg = segment_bins(qd_base, settings=mode)
        segs = qd_seg.segments["cfdna_01"]
        assert len(segs) >= 4
        # Check chr3 gain (~1.275) and chr4 loss (~0.725)
        chr3_seg = float(segs.loc[segs["chrom"] == "chr3", "segmented"].mean())
        chr4_seg = float(segs.loc[segs["chrom"] == "chr4", "segmented"].mean())
        assert 1.20 <= chr3_seg <= 1.35
        assert 0.65 <= chr4_seg <= 0.80

        qd_call = call_bins(qd_seg, settings=mode)
        assert qd_call.calls is not None
        probs = qd_call.probabilities["cfdna_01"]
        active = qd_call.bins["use"].to_numpy()
        row_sums = probs.to_numpy()[active].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

        calls = qd_call.calls["cfdna_01"].to_numpy()
        chr3_mask = (qd_call.bins["chrom"] == "chr3").to_numpy() & active
        chr4_mask = (qd_call.bins["chrom"] == "chr4").to_numpy() & active
        assert float(np.median(calls[chr3_mask])) == 1.0  # gain
        assert float(np.median(calls[chr4_mask])) == -1.0  # loss


def test_ace_squaremodel_recovers_diploid_cfdna_cellularity():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(cellularity=0.42, seed=44, mean_reads=800.0)
    qd = segment_bins(
        smooth_outlier_bins(
            normalize_bins(correct_counts(apply_filters(bin_read_counts(bins_df, counts_df=counts_df))))
        )
    )

    # In default mode (penploidy=0.5, pencellularity=0.5), diploid solution is uniquely rank 1
    ace_def = run_ace(qd, settings="default")
    assert isinstance(ace_def, ACEResult)
    assert abs(ace_def.cellularity - 0.42) <= 0.05
    assert abs(ace_def.ploidy - 2.0) <= 0.15
    assert ace_def.error_matrix.shape[0] > 50
    assert ace_def.error_matrix.shape[1] == 96
    assert {"rank", "cellularity", "ploidy", "error"}.issubset(ace_def.optima.columns)

    seg_table = ace_def.to_dataframe()
    c_by_chr = seg_table.groupby("chrom")["absolute_cn"].median().to_dict()
    assert c_by_chr["chr1"] == 2
    assert c_by_chr["chr3"] == 3
    assert c_by_chr["chr4"] == 1

    # In unpenalized R_default mode (penploidy=0, pencellularity=0), diploid solution is present in ranked optima
    ace_r = run_ace(qd, settings="R_default")
    dip_opt = ace_r.optima.loc[(ace_r.optima["ploidy"] - 2.0).abs() <= 0.15]
    assert not dip_opt.empty
    assert abs(float(dip_opt.iloc[0]["cellularity"]) - 0.42) <= 0.05


def test_ace_recovers_triploid_tumor_across_rmse_mae_and_smre():
    # Identifiable triploid tumor: C in {0 (25 bins), 3 (75 bins), 3 (75 bins), 4 (75 bins)}
    # -> exact true ploidy = 750 / 250 = 3.0, cellularity = 0.72 (> 2/3, uniquely identifiable)
    specs = [("chr1", 0, 25), ("chr2", 3, 75), ("chr3", 3, 75), ("chr4", 4, 75)]
    bins_df, counts_df, true_ploidy = _simulate_lpwgs_dataset(
        cellularity=0.72, seg_cn_specs=specs, seed=88, mean_reads=800.0
    )
    assert np.isclose(true_ploidy, 3.0)

    qd = segment_bins(
        normalize_bins(correct_counts(apply_filters(bin_read_counts(bins_df, counts_df=counts_df)))),
        settings="R_default",
    )

    for metric in ("RMSE", "MAE", "SMRE"):
        res = squaremodel(qd, method=metric, penploidy=0.0, pencellularity=0.0, settings="R_default")
        assert abs(res.cellularity - 0.72) <= 0.05
        assert abs(res.ploidy - 3.0) <= 0.20


def test_ace_direct_segments_dataframe_input_zero_noise():
    # Exact noiseless synthetic segments at cellularity=0.60, ploidy=2.0 -> r in {0.7, 1.0, 1.3}
    segs_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3", "chr4"],
            "start": [1, 1, 1, 1],
            "end": [20_000_000] * 4,
            "num_bins": [100, 100, 100, 100],
            "segmented": [1.0, 1.0, 1.3, 0.7],
        }
    )
    for mode in ("default", "R_default"):
        res = run_ace(segs_df, settings=mode)
        assert abs(res.cellularity - 0.60) <= 0.02
        assert abs(res.ploidy - 2.0) <= 0.05
        assert list(res.segments["absolute_cn"]) == [2, 2, 3, 1]


def test_qdnaseq_pipeline_end_to_end_multisample():
    bins_df, counts_1, _ = _simulate_lpwgs_dataset(cellularity=0.48, seed=101, mean_reads=700.0)
    _, counts_2, _ = _simulate_lpwgs_dataset(cellularity=0.72, seed=102, mean_reads=700.0)
    multi_counts = pd.DataFrame(
        {
            "patient_A": counts_1["cfdna_01"],
            "patient_B": counts_2["cfdna_01"],
        }
    )

    pipe = QDNAseqPipeline(settings="default")
    res = pipe.run(bins_df, counts_df=multi_counts)

    assert isinstance(res, QDNAseqPipelineResult)
    assert abs(res.ace_results["patient_A"].cellularity - 0.48) <= 0.06
    assert abs(res.ace_results["patient_B"].cellularity - 0.72) <= 0.06
    assert "QDNAseqData" in res.summary()
    assert len(res.to_dataframe(sample="patient_B")) >= 4


def test_call_bins_custom_cutoffs_and_normalize_modes():
    bins_df, counts_df, _ = _simulate_lpwgs_dataset(cellularity=0.60, seed=12, mean_reads=500.0)
    qd = bin_read_counts(bins_df, counts_df=counts_df)
    qd = apply_filters(qd)
    qd = correct_counts(qd, method="mean")
    for norm_m in ("median", "mean", "mode"):
        qd_n = normalize_bins(qd, method=norm_m)
        assert qd_n.copynumber is not None
    qd_seg = segment_bins(qd)
    qd_cut = call_bins(qd_seg, cutoffs=(-1.2, -0.25, 0.25, 0.9))
    assert qd_cut.calls is not None
    assert "QDNAseqData" in qd_cut.summary()


def test_invalid_settings_and_arguments_raise_value_error():
    bins_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1, 100_001],
            "end": [100_000, 200_000],
            "gc": [45.0, 46.0],
            "mappability": [90.0, 92.0],
        }
    )
    cnt = pd.Series([100.0, 105.0], name="s1")
    qd = bin_read_counts(bins_df, counts_df=cnt)

    for bad in ("wrong", "R", "v1", ""):
        with pytest.raises(ValueError, match="Unknown settings"):
            bin_read_counts(bins_df, counts_df=cnt, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            apply_filters(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            estimate_correction(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            correct_counts(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            normalize_bins(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            smooth_outlier_bins(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            segment_bins(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            call_bins(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            run_ace(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            squaremodel(qd, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            QDNAseqPipeline(settings=bad)  # type: ignore[arg-type]
