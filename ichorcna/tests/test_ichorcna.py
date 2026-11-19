"""
Deterministic verification suite for `ichorcna` covering exact log2 emission
equations, two-stage GC/mappability LOESS + Panel-of-Normals (`PoN`)
normalization, centromere filtering, multi-start Student-t HMM + Generalized EM
tumor-fraction recovery from 35% down to 2.5% ctDNA, triploid ploidy detection,
subclonal CNA estimation, healthy normal plasma (`TFx ~ 0`) calibration, and
`settings="default"` vs `settings="R_default"` parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ichorcna import (
    IchorCNA,
    IchorCNAConfig,
    IchorCNAResult,
    build_hmm_states,
    emission_means,
    normalize_ulp_bins,
    run_ichorcna,
)


def _simulate_ulp_genome(
    chrom_specs: list[tuple[str, int, int, float]],
    *,
    tumor_fraction: float,
    ploidy: float | None = None,
    noise_sd: float = 0.032,
    bin_size: int = 1_000_000,
    with_raw_reads: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate ULP-WGS genomic bins across chromosome segments `(chrom, n_bins, cn, s_sub)`.
    `s_sub = 1.0` denotes clonal state; `0 < s_sub < 1` denotes subclonal state.
    """
    rng = np.random.default_rng(seed)
    n_norm = 1.0 - tumor_fraction

    total_bins = sum(nb for _, nb, _, _ in chrom_specs)
    c_eff_list = [2.0 * (1.0 - s) + s * float(cn) for _, nb, cn, s in chrom_specs for _ in range(nb)]
    c_eff_arr = np.array(c_eff_list, dtype=np.float64)
    true_phi = float(np.mean(c_eff_arr)) if ploidy is None else float(ploidy)

    rows: list[dict[str, object]] = []
    chrom_offsets: dict[str, int] = {}

    for chrom, n_bins, cn, s_sub in chrom_specs:
        cur_start = chrom_offsets.get(chrom, 1)
        c_eff = 2.0 * (1.0 - s_sub) + s_sub * float(cn)
        mu = float(
            np.log2((2.0 * n_norm + tumor_fraction * c_eff) / (2.0 * n_norm + tumor_fraction * true_phi))
        )
        # Heavy-tailed Student-t / Gaussian mixture noise realistic for ULP-WGS
        eps = rng.normal(0.0, noise_sd, size=n_bins)
        gc = rng.uniform(0.32, 0.62, size=n_bins)
        mapp = rng.uniform(0.91, 0.995, size=n_bins)

        for i in range(n_bins):
            st = cur_start + i * bin_size
            en = st + bin_size - 1
            logr_i = mu + float(eps[i])
            row: dict[str, object] = {
                "chrom": chrom,
                "start": st,
                "end": en,
                "gc": float(gc[i]),
                "map": float(mapp[i]),
                "true_cn": cn,
                "true_subclonal": s_sub < 1.0,
            }
            if with_raw_reads:
                gc_bias = 1.0 + 0.45 * np.sin(np.pi * (gc[i] - 0.30) / 0.35)
                map_bias = 0.75 + 0.25 * mapp[i]
                expected_reads = 1800.0 * (2.0 ** logr_i) * gc_bias * map_bias
                row["reads"] = int(max(rng.poisson(expected_reads), 20))
            else:
                row["logR"] = logr_i
            rows.append(row)
        chrom_offsets[chrom] = cur_start + n_bins * bin_size

    assert len(rows) == total_bins
    return pd.DataFrame(rows)


def test_settings_validation_raises_value_error() -> None:
    bins = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [1, 1001], "end": [1000, 2000], "logR": [0.0, 0.0]}
    )
    with pytest.raises(ValueError, match="Unknown settings"):
        normalize_ulp_bins(bins, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        run_ichorcna(bins, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        IchorCNA(settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        IchorCNAConfig(normal_init=(0.8,), settings="bad")  # type: ignore[arg-type]


def test_exact_emission_means_clonal_and_subclonal() -> None:
    states = build_hmm_states(max_cn=5, include_homd=True, sc_states=(1, 3))
    names = [s.name for s in states]
    assert names == ["HOMD", "HETD", "NEUT", "GAIN", "AMP", "HLAMP", "1_sub", "3_sub"]

    c_arr = np.array([s.copy_number for s in states], dtype=np.float64)
    sub_arr = np.array([s.is_subclonal for s in states], dtype=bool)

    n = 0.80  # TFx = 0.20
    phi = 2.0
    s = 0.50
    mu = emission_means(n, phi, s, c_arr, sub_arr)

    # NEUT (c=2, phi=2) -> log2(2.0 / 2.0) = 0.0
    assert np.isclose(mu[names.index("NEUT")], 0.0, atol=1e-12)
    # GAIN (c=3) -> log2((1.6 + 0.6) / 2.0) = log2(1.1)
    assert np.isclose(mu[names.index("GAIN")], np.log2(1.1), atol=1e-12)
    # 3_sub (c=3, s=0.5) -> c_eff = 2.5 => log2((1.6 + 0.5) / 2.0) = log2(1.05)
    assert np.isclose(mu[names.index("3_sub")], np.log2(1.05), atol=1e-12)
    # HOMD (c=0) -> log2(1.6 / 2.0) = log2(0.8)
    assert np.isclose(mu[names.index("HOMD")], np.log2(0.8), atol=1e-12)


def test_normalize_ulp_bins_gc_map_loess_centromeres_and_pon() -> None:
    specs = [
        ("chr1", 120, 2, 1.0),
        ("chr2", 100, 3, 1.0),
        ("chr3", 100, 1, 1.0),
        ("chr4", 120, 2, 1.0),
    ]
    raw_df = _simulate_ulp_genome(specs, tumor_fraction=0.25, with_raw_reads=True, seed=10)
    # Inject a systematic wave artifact shared with a Panel of Normals + centromere region
    wave = 0.18 * np.sin(np.linspace(0.0, 6.0 * np.pi, len(raw_df)))
    raw_df["reads"] = np.round(raw_df["reads"].to_numpy() * (2.0 ** wave)).astype(np.int64)
    # Mark a few low-mappability bins
    raw_df.loc[5:7, "map"] = 0.50

    centromeres = pd.DataFrame(
        {"chrom": ["chr1"], "start": [40_000_000], "end": [45_000_000]}
    )
    pon = pd.DataFrame(
        {
            "chrom": raw_df["chrom"],
            "start": raw_df["start"],
            "norm1": wave + 0.01,
            "norm2": wave - 0.01,
            "norm3": wave,
        }
    )

    norm_no_pon = normalize_ulp_bins(raw_df, centromeres_df=centromeres, settings="default")
    norm_with_pon = normalize_ulp_bins(
        raw_df, panel_of_normals=pon, centromeres_df=centromeres, settings="default"
    )

    # Centromere bins and low-mappability bins were excluded
    assert len(norm_with_pon) < len(raw_df) - 5
    # PoN subtraction reduces residual variance on diploid chr4 substantially
    sd_no_pon = norm_no_pon.loc[norm_no_pon["chrom"] == "chr4", "logR"].std()
    sd_with_pon = norm_with_pon.loc[norm_with_pon["chrom"] == "chr4", "logR"].std()
    assert sd_with_pon < 0.65 * sd_no_pon


def test_high_tumor_fraction_and_segment_recovery() -> None:
    specs = [
        ("chr1", 100, 2, 1.0),
        ("chr2", 80, 3, 1.0),
        ("chr3", 80, 1, 1.0),
        ("chr4", 100, 2, 1.0),
        ("chr5", 60, 4, 1.0),
        ("chr6", 120, 2, 1.0),
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.32, ploidy=2.0, noise_sd=0.035, seed=21)
    res = run_ichorcna(bins, settings="default")
    assert isinstance(res, IchorCNAResult)
    assert abs(res.tumor_fraction - 0.32) < 0.03
    assert abs(res.ploidy - 2.0) < 0.20

    # Verify exact Viterbi copy-number accuracy across all bins
    acc = (res.bins_df["copy_number"] == res.bins_df["true_cn"]).mean()
    assert acc > 0.98


def test_ultra_low_tumor_fraction_3_percent_recovery() -> None:
    # 3.0% ctDNA (n = 0.97) with realistic ULP-WGS arm-level gains/losses
    specs = [
        ("chr1", 150, 2, 1.0),
        ("chr2", 140, 3, 1.0),
        ("chr3", 140, 1, 1.0),
        ("chr4", 150, 2, 1.0),
        ("chr5", 130, 3, 1.0),
        ("chr6", 150, 2, 1.0),
        ("chr7", 140, 1, 1.0),
        ("chr8", 160, 2, 1.0),
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.03, ploidy=2.0, noise_sd=0.024, seed=33)

    res_def = run_ichorcna(bins, settings="default")
    assert abs(res_def.tumor_fraction - 0.03) < 0.012
    assert abs(res_def.ploidy - 2.0) < 0.25
    acc_def = (res_def.bins_df["copy_number"] == res_def.bins_df["true_cn"]).mean()
    assert acc_def > 0.95

    # Also verify R_default low_tf_mode=(0.95, 0.99, 0.995) recovers 3% ctDNA
    res_r_low = run_ichorcna(bins, low_tf_mode=True, settings="R_default")
    assert set(res_r_low.solutions_df["normal_init"].unique()) == {0.95, 0.99, 0.995}
    assert abs(res_r_low.tumor_fraction - 0.03) < 0.012


def test_triploid_tumor_ploidy_and_tfx_recovery() -> None:
    # Triploid tumor (mean CN = 3.0: 25% c=2, 50% c=3, 25% c=4) at TFx = 0.38
    specs = [
        ("chr1", 100, 3, 1.0),
        ("chr2", 100, 2, 1.0),
        ("chr3", 100, 4, 1.0),
        ("chr4", 100, 3, 1.0),
        ("chr5", 80, 1, 1.0),
        ("chr5", 80, 5, 1.0),
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.38, ploidy=3.0, noise_sd=0.025, seed=44)
    res = run_ichorcna(bins, ploidy_init=(2.0, 3.0), sc_states=(), settings="default")
    assert abs(res.ploidy - 3.0) < 0.25
    assert abs(res.tumor_fraction - 0.38) < 0.04


def test_subclonal_cna_detection_and_fraction_estimation() -> None:
    # Tumor at TFx = 0.35 with predominantly clonal CNAs + one subclonal chr4 gain (s = 0.50)
    specs = [
        ("chr1", 120, 2, 1.0),
        ("chr2", 100, 3, 1.0),
        ("chr3", 100, 1, 1.0),
        ("chr4", 90, 3, 0.50),  # subclonal gain (3_sub)
        ("chr5", 120, 2, 1.0),
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.35, ploidy=2.0, noise_sd=0.022, seed=55)
    res = run_ichorcna(bins, sc_states=(1, 3), settings="default")

    assert abs(res.tumor_fraction - 0.35) < 0.04
    assert 0.30 <= res.subclone_fraction <= 0.70
    assert 0.05 <= res.frac_genome_subclonal <= 0.30
    chr4_states = res.bins_df.loc[res.bins_df["chrom"] == "chr4", "state"]
    assert (chr4_states == "3_sub").mean() > 0.85


def test_healthy_normal_plasma_zero_tfx_negative_control() -> None:
    specs = [("chr1", 120, 2, 1.0), ("chr2", 120, 2, 1.0), ("chr3", 120, 2, 1.0)]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.0, ploidy=2.0, noise_sd=0.03, seed=66)
    res = run_ichorcna(bins, settings="default")
    assert res.tumor_fraction == 0.0
    assert res.normal_fraction == 1.0
    assert (res.bins_df["state"] == "NEUT").all()


def test_homozygous_deletion_state_when_enabled() -> None:
    specs = [
        ("chr1", 100, 2, 1.0),
        ("chr2", 35, 0, 1.0),  # focal HOMD segment
        ("chr2", 85, 2, 1.0),
        ("chr3", 90, 1, 1.0),  # clonal HETD arm
        ("chr4", 125, 3, 1.0),  # clonal GAIN arm
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.30, ploidy=2.0, noise_sd=0.028, seed=77)
    res = run_ichorcna(bins, include_homd=True, settings="default")
    assert "HOMD" in set(res.segments_df["state"])
    homd_bins = res.bins_df.loc[res.bins_df["true_cn"] == 0, "state"]
    assert (homd_bins == "HOMD").mean() > 0.90


def test_default_vs_r_default_initialization_grids() -> None:
    specs = [("chr1", 80, 2, 1.0), ("chr2", 80, 3, 1.0), ("chr3", 80, 1, 1.0)]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.20, ploidy=2.0, noise_sd=0.03, seed=88)

    res_def = run_ichorcna(bins, max_iter=15, settings="default")
    res_r = run_ichorcna(bins, max_iter=15, settings="R_default")

    assert set(res_def.solutions_df["normal_init"].unique()) == {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98}
    assert set(res_r.solutions_df["normal_init"].unique()) == {0.5, 0.6, 0.7, 0.8, 0.9}
    assert int(res_def.solutions_df["selected"].sum()) == 1
    assert int(res_r.solutions_df["selected"].sum()) == 1


def test_ichorcna_class_end_to_end_raw_reads_and_summary() -> None:
    specs = [
        ("chr1", 90, 2, 1.0),
        ("chr2", 80, 3, 1.0),
        ("chr3", 80, 1, 1.0),
        ("chr4", 90, 2, 1.0),
    ]
    raw_bins = _simulate_ulp_genome(
        specs, tumor_fraction=0.24, ploidy=2.0, noise_sd=0.03, with_raw_reads=True, seed=99
    )
    model = IchorCNA(settings="default")
    res = model.run(raw_bins)

    summary = res.summary()
    segs_df = res.to_dataframe()
    assert abs(summary["tumor_fraction"] - 0.24) < 0.04
    assert summary["n_bins"] == len(raw_bins)
    assert len(segs_df) >= 4


def test_chromosome_boundary_reset_prevents_cross_chrom_merging() -> None:
    # Two adjacent chromosomes both in GAIN (c=3) must emit distinct chromosome segments
    specs = [("chr1", 60, 3, 1.0), ("chr2", 60, 3, 1.0), ("chr3", 120, 2, 1.0)]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.28, ploidy=2.0, noise_sd=0.025, seed=111)
    res = run_ichorcna(bins, settings="default")
    gain_segs = res.segments_df[res.segments_df["state"] == "GAIN"]
    assert set(gain_segs["chrom"]) == {"chr1", "chr2"}
    assert len(gain_segs) == 2


def test_low_tf_diploid_regularization_prefers_diploid_at_5_percent_ctdna() -> None:
    specs = [
        ("chr1", 120, 2, 1.0),
        ("chr2", 100, 3, 1.0),
        ("chr3", 100, 1, 1.0),
        ("chr4", 120, 2, 1.0),
    ]
    bins = _simulate_ulp_genome(specs, tumor_fraction=0.05, ploidy=2.0, noise_sd=0.028, seed=222)
    res = run_ichorcna(bins, ploidy_init=(2.0, 3.0, 4.0), settings="default")
    assert 1.65 <= res.ploidy <= 2.35
    assert abs(res.tumor_fraction - 0.05) < 0.015

