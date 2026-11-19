"""
Deterministic verification suite for `facets` covering analytical identities,
GC LOWESS normalization, SNP thinning, bivariate Hotelling-T2 CBS, planted-truth
purity/ploidy/ASCN/CCF/WGD recovery, null diploid calibration, ASCAT solver
concordance, and boundary/edge cases.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from facets import (
    AscatResult,
    FacetsConfig,
    FacetsResult,
    PreProcResult,
    SegmentedResult,
    ascat_baf,
    ascat_invert_cn,
    ascat_logr,
    canonical_chrom,
    compare_facets_ascat,
    compute_cnlr,
    compute_valor,
    em_cncf,
    fit_ascat,
    hotelling_t2_cbs,
    pre_proc_sample,
    preprocess_sample,
    proc_sample,
    run_ascat,
    run_facets,
)


def _simulate_pileup(
    segment_specs: list[tuple[str, int, int, int, int, int, float]],
    *,
    seed: int,
    nor_mean_dp: float = 85.0,
    tum_mean_dp: float = 95.0,
    het_prob: float = 0.45,
    gc_amplitude: float = 0.0,
    nb_size: float | None = None,
) -> pd.DataFrame:
    """
    Simulate realistic unphased tumor-normal SNP pileup counts from segment specs:
    each tuple is `(chrom, start, end, n_snps, c_major, c_minor, phi)`.
    """
    rng = np.random.default_rng(seed)

    # Compute bulk DNA average D_bar weighted by segment bp span
    spans = np.array([max(1, end - start + 1) for _, start, end, *_ in segment_specs], dtype=np.float64)
    d_segs = np.array(
        [2.0 * (1.0 - phi) + phi * (cmaj + cmin) for *_, cmaj, cmin, phi in segment_specs],
        dtype=np.float64,
    )
    d_bar = float(np.sum(spans * d_segs) / np.sum(spans))

    rows: list[pd.DataFrame] = []
    for (chrom, start, end, n_snps, cmaj, cmin, phi), d_s in zip(segment_specs, d_segs):
        positions = np.sort(
            rng.choice(np.arange(start, end + 1, 500, dtype=np.int64), size=n_snps, replace=False)
        )
        gc = rng.uniform(0.25, 0.75, size=n_snps)
        gc_wave = gc_amplitude * (np.sin(2.0 * np.pi * gc) + 0.5 * (gc - 0.5) ** 2)

        exp_tum_dp = np.maximum(
            tum_mean_dp * (d_s / max(d_bar, 1e-6)) * np.power(2.0, gc_wave),
            0.1,
        )
        if nb_size is None:
            nor_dp = np.maximum(rng.poisson(nor_mean_dp, size=n_snps), 40)
            tum_dp = rng.poisson(exp_tum_dp)
        else:
            nor_dp = np.maximum(
                rng.negative_binomial(
                    nb_size, nb_size / (nb_size + nor_mean_dp), size=n_snps
                ),
                40,
            )
            tum_dp = rng.negative_binomial(
                nb_size, nb_size / (nb_size + exp_tum_dp)
            )

        is_het = rng.random(n_snps) < het_prob
        nor_alt = np.zeros(n_snps, dtype=np.int64)
        tum_alt = np.zeros(n_snps, dtype=np.int64)

        # Homozygous ref (0) or homozygous alt (1) on non-het loci
        hom_alt = (~is_het) & (rng.random(n_snps) < 0.35)
        nor_alt[hom_alt] = nor_dp[hom_alt]
        tum_alt[hom_alt] = tum_dp[hom_alt]

        # Unphased germline heterozygous SNPs
        n_het = int(np.sum(is_het))
        if n_het > 0:
            nor_alt[is_het] = rng.binomial(nor_dp[is_het], 0.5)
            major_is_alt = rng.random(n_het) < 0.5
            if d_s > 1e-6:
                vaf_major = (1.0 - phi + phi * cmaj) / d_s
                vaf_minor = (1.0 - phi + phi * cmin) / d_s
                vaf_t = np.where(major_is_alt, vaf_major, vaf_minor)
            else:
                vaf_t = np.full(n_het, 0.5)
            vaf_t = np.clip(vaf_t, 0.0, 1.0)
            tum_alt[is_het] = rng.binomial(tum_dp[is_het], vaf_t)

        rows.append(
            pd.DataFrame(
                {
                    "chrom": chrom,
                    "pos": positions,
                    "nor_dp": nor_dp,
                    "tum_dp": tum_dp,
                    "nor_alt": nor_alt,
                    "tum_alt": tum_alt,
                    "gc": gc,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def test_analytical_logr_logor_and_ascat_equations() -> None:
    """Verify exact closed-form Haldane valor/lorvar identities and ASCAT inversion."""
    ta = np.array([30, 0, 45, 12], dtype=np.float64)
    tr = np.array([10, 40, 15, 38], dtype=np.float64)
    na = np.array([25, 20, 0, 22], dtype=np.float64)
    nr = np.array([25, 20, 40, 18], dtype=np.float64)

    valor, lorvar = compute_valor(ta, tr, na, nr)
    expected_v = np.log(((ta + 0.5) * (nr + 0.5)) / ((tr + 0.5) * (na + 0.5)))
    expected_lv = 1.0 / (ta + 0.5) + 1.0 / (tr + 0.5) + 1.0 / (na + 0.5) + 1.0 / (nr + 0.5)
    np.testing.assert_allclose(valor, expected_v, atol=1e-12)
    np.testing.assert_allclose(lorvar, expected_lv, atol=1e-12)

    # Allele swap invariance of valor^2 and lorvar; sign flip of valor
    valor_swap, lorvar_swap = compute_valor(tr, ta, nr, na)
    np.testing.assert_allclose(valor_swap, -valor, atol=1e-12)
    np.testing.assert_allclose(valor_swap**2, valor**2, atol=1e-12)
    np.testing.assert_allclose(lorvar_swap, lorvar, atol=1e-12)

    # Exact ASCAT forward & inverse identities
    states = [(1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (4, 2)]
    for rho in (0.3, 0.7, 1.0):
        psi_t = 2.65
        for gamma_a in (0.55, 1.0):
            for n_a, n_b in states:
                r = ascat_logr(n_a, n_b, rho, psi_t, gamma_ascat=gamma_a)
                b = ascat_baf(n_a, n_b, rho, minor_allele=True)
                na_rec, nb_rec = ascat_invert_cn(r, b, rho, psi_t, gamma_ascat=gamma_a)
                np.testing.assert_allclose(na_rec, float(n_a), atol=1e-12)
                np.testing.assert_allclose(nb_rec, float(n_b), atol=1e-12)

                # Identity linking ASCAT BAF and FACETS mu_O
                if b > 0.0:
                    mu_o_from_baf = np.log((1.0 - b) / b)
                    mu_o_direct = np.log((1.0 - rho + rho * n_a) / (1.0 - rho + rho * n_b))
                    np.testing.assert_allclose(mu_o_from_baf, mu_o_direct, atol=1e-12)


def test_gc_lowess_normalization_removes_nonlinear_wave() -> None:
    """Plant non-linear sinusoidal/quadratic GC bias and verify LOWESS removal."""
    rng = np.random.default_rng(102)
    n = 3000
    pos = np.arange(1_000_000, 1_000_000 + n * 500, 500, dtype=np.int64)
    gc = rng.uniform(0.20, 0.80, size=n)
    bias = 0.8 * np.sin(2.0 * np.pi * gc) + 0.4 * (gc - 0.5) ** 2

    nor_dp = np.maximum(rng.poisson(85.0, size=n), 40)
    tum_dp = rng.poisson(85.0 * np.power(2.0, bias))
    nor_alt = rng.binomial(nor_dp, 0.5)
    tum_alt = rng.binomial(np.maximum(tum_dp, 1), 0.5)
    tum_alt = np.minimum(tum_alt, tum_dp)

    pileup = pd.DataFrame(
        {
            "chrom": "chr1",
            "pos": pos,
            "nor_dp": nor_dp,
            "tum_dp": tum_dp,
            "nor_alt": nor_alt,
            "tum_alt": tum_alt,
            "gc": gc,
        }
    )

    pre = pre_proc_sample(pileup, snp_nbhd=0)
    df = pre.to_dataframe()
    assert pre.gc_corrected is True

    corr_raw = abs(float(np.corrcoef(df["cnlr_raw"], df["gc"])[0, 1]))
    corr_norm = abs(float(np.corrcoef(df["cnlr"], df["gc"])[0, 1]))
    mad_raw = float(np.median(np.abs(df["cnlr_raw"] - np.median(df["cnlr_raw"]))))
    mad_norm = float(np.median(np.abs(df["cnlr"] - np.median(df["cnlr"]))))

    assert corr_raw > 0.35
    assert corr_norm < 0.05
    assert mad_raw > 0.40
    assert mad_norm < 0.25
    assert abs(float(np.median(df["cnlr"]))) < 0.05


def test_snp_nbhd_thinning_and_het_filtering() -> None:
    """Verify neighborhood SNP thinning >= 250 bp, depth filtering, and het calls."""
    rng = np.random.default_rng(103)
    # Dense 20-bp spaced clusters across chr1 and chr2, plus depth outliers
    pos_c1 = np.concatenate([np.arange(1000, 1400, 20), np.arange(5000, 5400, 20)])
    pos_c2 = np.concatenate([np.arange(2000, 2400, 20), np.arange(8000, 8400, 20)])
    n_c1 = pos_c1.size
    n_c2 = pos_c2.size
    n_tot = n_c1 + n_c2

    nor_dp = np.full(n_tot, 80, dtype=np.int64)
    nor_dp[0] = 10  # below ndepth=35
    nor_dp[-1] = 1500  # above ndepthmax=1000
    tum_dp = np.full(n_tot, 80, dtype=np.int64)

    # Alternate hom (vaf=0.05) and het (vaf=0.50) within dense windows
    nor_alt = np.where(np.arange(n_tot) % 2 == 1, 40, 4)
    tum_alt = nor_alt.copy()

    pileup = pd.DataFrame(
        {
            "chrom": ["1"] * n_c1 + ["chr2"] * n_c2,
            "pos": np.concatenate([pos_c1, pos_c2]),
            "nor_dp": nor_dp,
            "tum_dp": tum_dp,
            "nor_alt": nor_alt,
            "tum_alt": tum_alt,
        }
    )

    pre = pre_proc_sample(pileup, ndepth=35, ndepthmax=1000, het_thresh=0.25, snp_nbhd=250)
    df = pre.to_dataframe()
    assert (df["nor_dp"] >= 35).all() and (df["nor_dp"] <= 1000).all()

    for ch, grp in df.groupby("chrom"):
        diffs = np.diff(grp["pos"].to_numpy(dtype=np.int64))
        assert np.all(diffs >= 250), f"Thinning violated on {ch}: min diff={diffs.min()}"

    # Check het preference inside dense clusters and het threshold correctness
    vaf = df["nor_alt"] / df["nor_dp"]
    expected_het = (vaf >= 0.25) & (vaf <= 0.75)
    pd.testing.assert_series_equal(df["het"], expected_het, check_names=False)
    assert df["het"].all()


def test_bivariate_hotelling_cbs_detects_cn_neutral_loh_and_gain() -> None:
    """
    Verify bivariate Hotelling-T2 CBS detects a copy-neutral LOH (2,0) segment
    (zero logR shift, pure allelic logOR^2 shift) alongside a (2,1) gain.
    """
    specs = [
        ("chr1", 1_000_000, 15_000_000, 180, 1, 1, 0.75),
        ("chr1", 15_500_000, 35_000_000, 180, 2, 0, 0.75),  # cnLOH: tcn=2 -> logR=0!
        ("chr1", 35_500_000, 55_000_000, 180, 1, 1, 0.75),
        ("chr1", 55_500_000, 75_000_000, 180, 2, 1, 0.75),  # gain: logR>0, logOR>0
    ]
    pileup = _simulate_pileup(specs, seed=104, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.5)
    pre = pre_proc_sample(pileup, snp_nbhd=0)
    seg = proc_sample(pre)
    segs_df = seg.to_dataframe()

    # Should recover 4 segments on chr1p (before centromere at 122 Mb)
    assert len(segs_df) == 4
    # Flanking (1,1) segments should share the same cluster ID
    assert segs_df.iloc[0]["segclust"] == segs_df.iloc[2]["segclust"]
    # cnLOH segment should have high mafR and near-zero cnlr_mean
    assert segs_df.iloc[1]["mafR"] > 0.8
    assert abs(segs_df.iloc[1]["cnlr_mean"] - segs_df.iloc[0]["cnlr_mean"]) < 0.25

    # Univariate CBS (no logOR) misses the cnLOH breakpoints
    x1 = pre.snps["cnlr"].to_numpy(dtype=np.float64)
    uni_segs = hotelling_t2_cbs(x1, None, None, cval=150.0)
    assert len(uni_segs) <= 2


def test_planted_truth_purity_ploidy_ascn_recovery() -> None:
    """
    End-to-end recovery of planted purity (0.65), ploidy, dipLogR,
    (1,1), (1,0) LOH, (2,0) cnLOH, (2,1) gain, (3,1) amp, and (0,0) homdel.
    """
    alpha = 0.65
    specs = [
        ("chr1", 1_000_000, 50_000_000, 500, 1, 1, alpha),
        ("chr1", 50_500_000, 90_000_000, 400, 1, 0, alpha),   # LOH
        ("chr2", 1_000_000, 45_000_000, 450, 2, 0, alpha),   # cnLOH
        ("chr2", 45_500_000, 85_000_000, 400, 2, 1, alpha),  # gain
        ("chr3", 1_000_000, 45_000_000, 450, 1, 1, alpha),
        ("chr3", 45_500_000, 75_000_000, 350, 3, 1, alpha),  # amp
        ("chr4", 1_000_000, 35_000_000, 400, 1, 1, alpha),
        ("chr5", 1_000_000, 25_000_000, 280, 0, 0, alpha),   # homdel
        ("chr6", 1_000_000, 45_000_000, 450, 1, 1, alpha),
    ]
    spans = np.array([e - s + 1 for _, s, e, *_ in specs], dtype=np.float64)
    true_tcns = np.array([cm + cn for *_, cm, cn, _ in specs], dtype=np.float64)
    true_ploidy = float(np.sum(spans * true_tcns) / np.sum(spans))

    pileup = _simulate_pileup(
        specs, seed=105, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.45, gc_amplitude=0.35
    )
    res = run_facets(pileup)
    assert isinstance(res, FacetsResult)
    assert res.is_diploid_null is False
    assert res.is_wgd is False
    assert res.purity is not None
    assert abs(res.purity - alpha) < 0.05
    assert abs(res.ploidy - true_ploidy) < 0.15

    # Evaluate bp-weighted exact (tcn, lcn) recovery against planted segments
    segs = res.to_dataframe()
    matched_bp = 0.0
    total_bp = 0.0
    for ch, s0, e0, _, cm_t, cn_t, _ in specs:
        sub = segs[segs["chrom"] == ch]
        for _, row in sub.iterrows():
            ov = max(0.0, float(min(int(row["end"]), e0) - max(int(row["start"]), s0) + 1))
            if ov <= 0:
                continue
            total_bp += ov
            if int(row["c_major"]) == cm_t and int(row["c_minor"]) == cn_t:
                matched_bp += ov

    assert matched_bp / total_bp >= 0.95

    # Verify LOH and homozygous deletion flags explicitly
    chr1_loh = segs[(segs["chrom"] == "chr1") & (segs["start"] >= 50_000_000)]
    assert (chr1_loh["tcn"] == 1).all() and (chr1_loh["loh"]).all()

    chr2_cnloh = segs[(segs["chrom"] == "chr2") & (segs["end"] <= 46_000_000)]
    assert (chr2_cnloh["tcn"] == 2).all() and (chr2_cnloh["cn_loh"]).all()

    chr5_homdel = segs[segs["chrom"] == "chr5"]
    assert (chr5_homdel["tcn"] == 0).all() and (chr5_homdel["lcn"] == 0).all()
    assert not chr5_homdel["loh"].any()


def test_subclonal_cellular_fraction_recovery() -> None:
    """
    Verify subclonal (1,0) deletion CCF=0.40 recovery alongside clonal alpha=0.65.
    (Note: phi == alpha / 2 is algebraically degenerate with clonal WGD (2,1).)
    """
    specs = [
        ("chr1", 1_000_000, 60_000_000, 500, 1, 1, 0.65),
        ("chr2", 1_000_000, 50_000_000, 450, 1, 0, 0.65),  # clonal loss
        ("chr3", 1_000_000, 50_000_000, 450, 2, 1, 0.65),  # clonal gain
        ("chr4", 1_000_000, 50_000_000, 450, 1, 0, 0.40),  # subclonal loss
        ("chr5", 1_000_000, 50_000_000, 450, 1, 1, 0.65),
    ]
    pileup = _simulate_pileup(specs, seed=106, nor_mean_dp=100.0, tum_mean_dp=110.0, het_prob=0.50)
    res = run_facets(pileup)

    assert res.is_wgd is False
    assert res.purity is not None
    assert abs(res.purity - 0.65) < 0.06

    segs = res.to_dataframe()
    chr4_segs = segs[segs["chrom"] == "chr4"]
    assert len(chr4_segs) >= 1
    assert (chr4_segs["tcn"] == 1).all()
    assert (chr4_segs["c_minor"] == 0).all()
    assert (chr4_segs["is_subclonal"]).all()
    assert abs(float(chr4_segs["cf_em"].mean()) - 0.40) < 0.08


def test_whole_genome_doubling_disambiguation() -> None:
    """
    Verify WGD disambiguation: tetraploid tumor dominated by (2,2) with (2,1),
    (2,0), (3,2), and minor (1,1) calls is_wgd=True and recovers alpha=0.70.
    """
    alpha = 0.70
    specs = [
        ("chr1", 1_000_000, 80_000_000, 600, 2, 2, alpha),   # dominant WGD (2,2)
        ("chr2", 1_000_000, 70_000_000, 500, 2, 2, alpha),   # dominant WGD (2,2)
        ("chr3", 1_000_000, 55_000_000, 450, 2, 1, alpha),   # post-WGD loss (2,1)
        ("chr4", 1_000_000, 45_000_000, 400, 2, 0, alpha),   # cnLOH (2,0)
        ("chr5", 1_000_000, 45_000_000, 400, 3, 2, alpha),   # gain (3,2)
        ("chr6", 1_000_000, 35_000_000, 350, 1, 1, alpha),   # pre-WGD diploid relic (1,1)
    ]
    spans = np.array([e - s + 1 for _, s, e, *_ in specs], dtype=np.float64)
    true_tcns = np.array([cm + cn for *_, cm, cn, _ in specs], dtype=np.float64)
    true_ploidy = float(np.sum(spans * true_tcns) / np.sum(spans))

    pileup = _simulate_pileup(specs, seed=107, nor_mean_dp=95.0, tum_mean_dp=105.0, het_prob=0.45)
    res = run_facets(pileup)

    assert res.is_wgd is True
    assert res.is_diploid_null is False
    assert res.purity is not None
    assert abs(res.purity - alpha) < 0.06
    assert abs(res.ploidy - true_ploidy) < 0.20

    segs = res.to_dataframe()
    chr1_major = int(segs.loc[segs["chrom"] == "chr1", "c_major"].iloc[0])
    chr1_minor = int(segs.loc[segs["chrom"] == "chr1", "c_minor"].iloc[0])
    assert (chr1_major, chr1_minor) == (2, 2)

    chr4_row = segs[segs["chrom"] == "chr4"].iloc[0]
    assert (int(chr4_row["c_major"]), int(chr4_row["c_minor"])) == (2, 0)
    assert bool(chr4_row["cn_loh"]) is True


def test_null_calibration_pure_diploid_and_zero_purity() -> None:
    """
    Verify pure normal / balanced diploid genomes trigger <= 1 spurious CBS split
    and return is_diploid_null=True, purity=None, ploidy=2.0, (2,1) everywhere.
    """
    for rep_seed in (201, 202, 203):
        specs = [
            ("chr1", 1_000_000, 80_000_000, 500, 1, 1, 0.0),
            ("chr2", 1_000_000, 75_000_000, 500, 1, 1, 0.0),
            ("chr3", 1_000_000, 70_000_000, 500, 1, 1, 0.0),
            ("chr4", 1_000_000, 65_000_000, 500, 1, 1, 0.0),
        ]
        pileup = _simulate_pileup(specs, seed=rep_seed, nor_mean_dp=80.0, tum_mean_dp=80.0, het_prob=0.45)
        pre = pre_proc_sample(pileup)
        seg = proc_sample(pre)
        # 4 arms -> expect <= 5 segments (at most 1 false split genome-wide)
        assert len(seg.segments) <= 5

        res = em_cncf(seg)
        assert res.is_diploid_null is True
        assert res.purity is None
        assert res.ploidy == pytest.approx(2.0, abs=1e-6)
        assert res.is_wgd is False

        segs_df = res.to_dataframe()
        lengths = np.maximum(segs_df["end"] - segs_df["start"] + 1, 1)
        ok_mask = (segs_df["tcn"] == 2) & (segs_df["lcn"] == 1) & (~segs_df["loh"])
        assert float(lengths[ok_mask].sum() / lengths.sum()) >= 0.99


def test_ascat_solver_and_facets_concordance() -> None:
    """
    Verify `fit_ascat` recovers exact planted (purity=0.55, ploidy=2.45) under
    gamma_ascat in {0.55, 1.0} and achieves high concordance with FACETS.
    """
    true_rho = 0.55
    seg_states = [
        ("chr1", 1, 35_000_000, 1, 1),
        ("chr2", 1, 30_000_000, 2, 1),
        ("chr3", 1, 15_000_000, 1, 0),
        ("chr4", 1, 15_000_000, 2, 0),
        ("chr5", 1, 20_000_000, 3, 1),
    ]
    lengths = np.array([e - s + 1 for _, s, e, *_ in seg_states], dtype=np.float64)
    true_ploidy = float(
        np.sum(lengths * np.array([ma + mi for *_, ma, mi in seg_states])) / np.sum(lengths)
    )
    assert abs(true_ploidy - 2.45) < 0.05

    for gamma_a in (0.55, 1.0):
        rows = []
        for ch, s, e, ma, mi in seg_states:
            rows.append(
                {
                    "chrom": ch,
                    "start": s,
                    "end": e,
                    "logr": ascat_logr(ma, mi, true_rho, true_ploidy, gamma_ascat=gamma_a),
                    "baf": ascat_baf(ma, mi, true_rho, minor_allele=True),
                    "nhet": 100,
                }
            )
        seg_df = pd.DataFrame(rows)
        ares = fit_ascat(seg_df, gamma_ascat=gamma_a)
        assert isinstance(ares, AscatResult)
        assert abs(ares.purity - true_rho) < 0.03
        assert abs(ares.ploidy - true_ploidy) < 0.10
        assert ares.goodness_of_fit > 0.98
        assert list(ares.to_dataframe()["n_major"]) == [r[3] for r in seg_states]
        assert list(ares.to_dataframe()["n_minor"]) == [r[4] for r in seg_states]

    # End-to-end concordance between FACETS and ASCAT on simulated pileup
    alpha = 0.70
    pileup_specs = [
        ("chr1", 1_000_000, 50_000_000, 400, 1, 1, alpha),
        ("chr2", 1_000_000, 40_000_000, 350, 1, 0, alpha),
        ("chr3", 1_000_000, 40_000_000, 350, 2, 1, alpha),
        ("chr4", 1_000_000, 35_000_000, 300, 2, 0, alpha),
    ]
    pileup = _simulate_pileup(pileup_specs, seed=109, nor_mean_dp=95.0, tum_mean_dp=105.0, het_prob=0.50)
    fres = run_facets(pileup)
    ares_from_facets = fit_ascat(fres, gamma_ascat=1.0)
    comp = compare_facets_ascat(fres, ares_from_facets)
    assert float(comp["purity_abs_diff"].iloc[0]) < 0.08
    assert float(comp["ploidy_abs_diff"].iloc[0]) < 0.20
    assert float(comp["tcn_bp_concordance"].iloc[0]) >= 0.95
    assert float(comp["lcn_bp_concordance"].iloc[0]) >= 0.95


def test_edge_cases_low_nhet_nans_zero_depth_unmatched_and_validation() -> None:
    """
    Verify:
      - low_nhet (< min_nhet) sets lcn=NaN while retaining tcn and lcn_raw
      - tum_dp=0 and NaN gc values do not cause NaN/inf crashes
      - diploid_logr user override is held fixed
      - '1' vs 'chr1' equivalence and unmatched mode work cleanly
      - invalid inputs raise descriptive ValueErrors
      - .to_dataframe() and .summary() work on all primary result classes
    """
    # 1. Low-nhet focal amplification + NaN gc + tum_dp=0 locus
    specs = [
        ("1", 1_000_000, 40_000_000, 350, 1, 1, 0.70),
        ("1", 40_500_000, 43_000_000, 45, 3, 1, 0.70),
        ("1", 43_500_000, 80_000_000, 350, 1, 1, 0.70),
        ("23", 1_000_000, 35_000_000, 300, 1, 0, 0.70),
    ]
    pileup = _simulate_pileup(specs, seed=110, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.45)
    # Force only 3 het SNPs in the focal chr1 amplification (40.5M - 43.0M)
    focal_mask = (pileup["chrom"] == "1") & (pileup["pos"] >= 40_500_000) & (pileup["pos"] <= 43_000_000)
    focal_idx = np.flatnonzero(focal_mask)
    # Make all but 3 homozygous in normal
    for k, idx in enumerate(focal_idx):
        if k < 3:
            pileup.loc[idx, "nor_alt"] = pileup.loc[idx, "nor_dp"] // 2
        else:
            pileup.loc[idx, "nor_alt"] = 0
            pileup.loc[idx, "tum_alt"] = 0

    # Inject NaNs in gc and zero tum_dp locus
    pileup.loc[0:20, "gc"] = np.nan
    pileup.loc[5, "tum_dp"] = 0
    pileup.loc[5, "tum_alt"] = 0

    pre = pre_proc_sample(pileup, min_nhet=15)
    seg = proc_sample(pre)
    res = em_cncf(seg)

    for obj in (pre, seg, res):
        assert isinstance(obj.to_dataframe(), pd.DataFrame)
        assert isinstance(obj.summary(), str) and len(obj.summary()) > 10
    assert isinstance(res.sample_summary_dataframe(), pd.DataFrame)

    assert canonical_chrom("23") == "chrX"
    assert "chrX" in set(res.segments["chrom"])
    low_het_segs = res.segments[res.segments["low_nhet"]]
    assert len(low_het_segs) >= 1
    assert low_het_segs["lcn"].isna().all()
    assert low_het_segs["lcn_raw"].notna().all()
    # Focal amp tcn should still be >= 3
    focal_out = res.segments[
        (res.segments["chrom"] == "chr1")
        & (res.segments["start"] <= 43_000_000)
        & (res.segments["end"] >= 40_500_000)
        & (res.segments["low_nhet"])
    ]
    assert len(focal_out) >= 1
    assert (focal_out["tcn"] >= 3).all()

    # 2. Locked diploid_logr override
    res_locked = em_cncf(seg, diploid_logr=-0.25)
    assert res_locked.dip_logr == pytest.approx(-0.25, abs=1e-12)

    # 3. Unmatched tumor-only mode
    res_unmatched = run_facets(pileup.drop(columns=["gc"]), unmatched=True)
    assert isinstance(res_unmatched, FacetsResult)
    assert res_unmatched.purity is not None

    # 4. Validation errors
    with pytest.raises(ValueError, match="empty"):
        pre_proc_sample(pileup.iloc[0:0])
    with pytest.raises(ValueError, match="Missing required"):
        pre_proc_sample(pileup.drop(columns=["tum_dp"]))
    bad_alt = pileup.copy()
    bad_alt.loc[0, "nor_alt"] = bad_alt.loc[0, "nor_dp"] + 5
    with pytest.raises(ValueError, match="nor_alt > nor_dp"):
        pre_proc_sample(bad_alt)
    with pytest.raises(ValueError, match="No SNPs remain"):
        pre_proc_sample(pileup, ndepth=10000, ndepthmax=20000)


def test_vectorized_runtime_10k_snps() -> None:
    """Verify full pipeline on 10,000 SNPs executes cleanly and rapidly."""
    specs = [
        ("chr1", 1_000_000, 100_000_000, 2500, 1, 1, 0.70),
        ("chr2", 1_000_000, 100_000_000, 2500, 2, 1, 0.70),
        ("chr3", 1_000_000, 100_000_000, 2500, 1, 0, 0.70),
        ("chr4", 1_000_000, 100_000_000, 2500, 2, 0, 0.70),
    ]
    pileup = _simulate_pileup(specs, seed=111, nor_mean_dp=80.0, tum_mean_dp=90.0, het_prob=0.40)
    t0 = time.perf_counter()
    res = run_facets(pileup, snp_nbhd=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    assert res.purity is not None
    assert abs(res.purity - 0.70) < 0.05


def test_gc_string_na_and_inf_validation() -> None:
    """Verify object-dtype string 'NA' in `gc` coerces to NaN while 'inf' still raises ValueError."""
    specs = [("chr1", 1_000_000, 50_000_000, 120, 1, 1, 0.0)]
    pileup = _simulate_pileup(specs, seed=501)
    pileup["gc"] = pileup["gc"].astype(str)
    pileup.loc[0:10, "gc"] = "NA"

    pre = pre_proc_sample(pileup)
    assert pre.gc_corrected is True
    assert pre.snps["gc"].isna().sum() >= 10

    bad_inf = pileup.copy()
    bad_inf.loc[0, "gc"] = "inf"
    with pytest.raises(ValueError, match="infinite"):
        pre_proc_sample(bad_inf)


def test_unmatched_het_thresh_parity_run_facets_vs_pre_proc_sample() -> None:
    """
    Verify `run_facets(..., unmatched=True)`, `FacetsConfig(unmatched=True)`, and
    `pre_proc_sample(..., unmatched=True)` all share the 0.10 unmatched het_thresh
    default while honoring explicit het_thresh overrides.
    """
    specs = [
        ("chr1", 1_000_000, 50_000_000, 300, 1, 1, 0.70),
        ("chr2", 1_000_000, 50_000_000, 300, 1, 0, 0.70),  # clonal LOH: tumor VAF ~ 0.231 < 0.25
    ]
    pileup = _simulate_pileup(specs, seed=502, het_prob=0.50)

    pre_direct = pre_proc_sample(pileup, unmatched=True)
    cfg_unmatched = FacetsConfig(unmatched=True)
    assert cfg_unmatched.effective_het_thresh == pytest.approx(0.10)
    pre_cfg = pre_proc_sample(pileup, config=cfg_unmatched)
    res_run = run_facets(pileup, unmatched=True)

    assert pre_direct.n_het == pre_cfg.n_het
    assert res_run.snps is not None
    assert int(res_run.snps["het"].sum()) == pre_direct.n_het

    # Explicit override of het_thresh=0.25 must still be respected in both APIs
    pre_strict = pre_proc_sample(pileup, unmatched=True, het_thresh=0.25)
    res_strict = run_facets(pileup, unmatched=True, het_thresh=0.25)
    assert pre_strict.n_het < pre_direct.n_het
    assert res_strict.snps is not None
    assert int(res_strict.snps["het"].sum()) == pre_strict.n_het


def test_focal_sub1pct_high_copy_amplification_recovery() -> None:
    """
    Verify high-BIC focal amplifications spanning < 1% of genome (0.47% 8-copy and
    0.80% 9-copy) are NOT collapsed to diploid null.
    """
    for frac_label, end_bp, cmaj, seed in [
        ("0.47pct_8cn", 51_500_000, 7, 801),
        ("0.80pct_9cn", 52_100_000, 8, 802),
    ]:
        specs = [
            ("chr1", 1_000_000, 100_000_000, 600, 1, 1, 0.70),
            ("chr2", 1_000_000, 100_000_000, 600, 1, 1, 0.70),
            ("chr3", 1_000_000, 50_000_000, 300, 1, 1, 0.70),
            ("chr3", 50_100_000, end_bp, 40, cmaj, 1, 0.70),
            ("chr3", end_bp + 100_000, 100_000_000, 300, 1, 1, 0.70),
        ]
        pileup = _simulate_pileup(specs, seed=seed, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.45)
        res = run_facets(pileup, snp_nbhd=0)
        assert res.is_diploid_null is False, f"{frac_label} falsely collapsed to diploid null"
        assert res.purity is not None
        assert res.ploidy > 2.01
        segs = res.to_dataframe()
        focal = segs[
            (segs["chrom"] == "chr3")
            & (segs["start"] >= 49_500_000)
            & (segs["end"] <= end_bp + 500_000)
            & (segs["tcn"] > 2)
        ]
        assert len(focal) >= 1, f"{frac_label} focal segment missing"
        # A lone 12-het focal amplicon trades off CN vs purity (e.g. tcn=7@0.87 vs 8@0.70)
        assert int(focal["tcn"].max()) >= 6

    # When purity is anchored by a clonal (1,0) arm loss, exact focal TCN=9 and purity=0.70 recover
    specs_anchored = [
        ("chr1", 1_000_000, 100_000_000, 600, 1, 1, 0.70),
        ("chr2", 1_000_000, 100_000_000, 600, 1, 0, 0.70),
        ("chr3", 1_000_000, 50_000_000, 300, 1, 1, 0.70),
        ("chr3", 50_100_000, 52_100_000, 40, 8, 1, 0.70),
        ("chr3", 52_200_000, 100_000_000, 300, 1, 1, 0.70),
    ]
    res_anc = run_facets(
        _simulate_pileup(specs_anchored, seed=803, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.45),
        snp_nbhd=0,
    )
    assert res_anc.is_diploid_null is False
    assert res_anc.purity is not None and abs(res_anc.purity - 0.70) < 0.05
    anc_segs = res_anc.to_dataframe()
    anc_focal = anc_segs[(anc_segs["chrom"] == "chr3") & (anc_segs["tcn"] > 2)]
    assert int(anc_focal["tcn"].max()) == 9


def test_wgd_odd_states_without_diploid_relic_and_diploid_asymmetric() -> None:
    """
    Verify WGD genomes (2,2)+(4,2)+(2,0)+(3,2) and (2,2)+(4,2)+(2,0)+(2,1)
    without pre-WGD (1,1) relic recover is_wgd=True and true integer states,
    while diploid asymmetric genomes without (1,1) still resolve as is_wgd=False.
    """
    for odd_maj, odd_min, seed, exp_ploidy in [
        (3, 2, 804, 4.20),
        (2, 1, 805, 3.80),
    ]:
        specs = [
            ("chr1", 1_000_000, 80_000_000, 500, 2, 2, 0.70),
            ("chr2", 1_000_000, 50_000_000, 400, 4, 2, 0.70),
            ("chr3", 1_000_000, 50_000_000, 400, 2, 0, 0.70),
            ("chr4", 1_000_000, 45_000_000, 400, odd_maj, odd_min, 0.70),
        ]
        pileup = _simulate_pileup(specs, seed=seed, nor_mean_dp=95.0, tum_mean_dp=105.0, het_prob=0.45)
        res = run_facets(pileup)
        assert res.is_wgd is True
        assert res.is_diploid_null is False
        assert res.purity is not None
        assert abs(res.purity - 0.70) < 0.06
        assert abs(res.ploidy - exp_ploidy) < 0.15
        segs = res.to_dataframe()
        calls = [(int(r["c_major"]), int(r["c_minor"])) for _, r in segs.iterrows()]
        assert calls == [(2, 2), (4, 2), (2, 0), (odd_maj, odd_min)]

    # Diploid asymmetric genome with no (1,1) segment must remain is_wgd=False
    specs_asym = [
        ("chr1", 1_000_000, 70_000_000, 500, 2, 1, 0.70),
        ("chr2", 1_000_000, 60_000_000, 450, 1, 0, 0.70),
        ("chr3", 1_000_000, 50_000_000, 400, 2, 0, 0.70),
    ]
    res_asym = run_facets(
        _simulate_pileup(specs_asym, seed=403, nor_mean_dp=95.0, tum_mean_dp=105.0, het_prob=0.45)
    )
    assert res_asym.is_wgd is False
    assert res_asym.is_diploid_null is False
    assert res_asym.purity is not None
    assert abs(res_asym.purity - 0.70) < 0.06


def test_low_depth_12x_pure_diploid_null_calibration() -> None:
    """Verify 12x low-depth pure-diploid genomes maintain variance-debiased null calibration."""
    specs = [
        ("chr1", 1_000_000, 80_000_000, 500, 1, 1, 0.0),
        ("chr2", 1_000_000, 75_000_000, 500, 1, 1, 0.0),
        ("chr3", 1_000_000, 70_000_000, 500, 1, 1, 0.0),
        ("chr4", 1_000_000, 65_000_000, 500, 1, 1, 0.0),
    ]
    rng = np.random.default_rng(912)
    pileup = _simulate_pileup(specs, seed=912, nor_mean_dp=12.0, tum_mean_dp=12.0, het_prob=0.45)
    # Override the 40x helper floor to generate genuine ~12x normal and tumor depths
    pileup["nor_dp"] = np.maximum(rng.poisson(12.0, size=len(pileup)), 5)
    pileup["tum_dp"] = np.maximum(rng.poisson(12.0, size=len(pileup)), 5)
    is_h = pileup["nor_alt"] > 0
    pileup["nor_alt"] = np.where(is_h, rng.binomial(pileup["nor_dp"], 0.5), 0)
    pileup["tum_alt"] = np.where(is_h, rng.binomial(pileup["tum_dp"], 0.5), 0)

    res = run_facets(pileup, ndepth=5)
    assert res.is_diploid_null is True
    assert res.purity is None
    assert res.ploidy == pytest.approx(2.0, abs=1e-6)
    assert float(res.segments["mafR"].max()) == pytest.approx(0.0, abs=1e-6)
    assert len(res.segments) <= 5


def test_all_homozygous_zero_het_sample() -> None:
    """Verify n_het == 0 samples handle both pure-diploid null and multi-state TCN recovery."""
    # 1. Pure diploid all-homozygous -> diploid null
    specs_null = [
        ("chr1", 1_000_000, 60_000_000, 400, 1, 1, 0.0),
        ("chr2", 1_000_000, 60_000_000, 400, 1, 1, 0.0),
    ]
    res_null = run_facets(_simulate_pileup(specs_null, seed=920, het_prob=0.0))
    assert res_null.is_diploid_null is True
    assert res_null.purity is None

    # 2. Multi-state all-homozygous -> recovers TCN from logR while masking LCN to NaN
    specs_cnv = [
        ("chr1", 1_000_000, 60_000_000, 450, 1, 1, 0.70),
        ("chr2", 1_000_000, 60_000_000, 450, 3, 1, 0.70),
        ("chr3", 1_000_000, 60_000_000, 450, 1, 0, 0.70),
    ]
    res_cnv = run_facets(
        _simulate_pileup(specs_cnv, seed=921, nor_mean_dp=95.0, tum_mean_dp=105.0, het_prob=0.0)
    )
    assert res_cnv.is_diploid_null is False
    assert res_cnv.purity is not None
    assert abs(res_cnv.purity - 0.70) < 0.08
    segs = res_cnv.to_dataframe()
    assert segs["low_nhet"].all()
    assert segs["lcn"].isna().all()
    assert list(segs["tcn"]) == [2, 4, 1]


def test_subclonal_gain_alongside_clonal_gain_without_sigma_inflation() -> None:
    """
    Regression test for sigma_logor2 inflation when <35% of pre-segments are
    allelically balanced: chr1 (1,1), chr2 (1,0), chr3 (2,1)@alpha, chr4 (2,1)@phi
    must keep sigma_logor2 near sqrt(2), separate clonal vs subclonal (2,1)
    clusters, and accurately recover cf_em under both (0.76, 0.50) and (0.65, 0.40).
    """
    for alpha, phi_sub in [(0.76, 0.50), (0.65, 0.40)]:
        specs = [
            ("chr1", 1_000_000, 60_000_000, 500, 1, 1, alpha),
            ("chr2", 1_000_000, 50_000_000, 450, 1, 0, alpha),
            ("chr3", 1_000_000, 50_000_000, 450, 2, 1, alpha),
            ("chr4", 1_000_000, 50_000_000, 450, 2, 1, phi_sub),
        ]
        pileup = _simulate_pileup(
            specs,
            seed=77,
            nor_mean_dp=95.0,
            tum_mean_dp=105.0,
            het_prob=0.45,
            nb_size=None,
        )
        pre = pre_proc_sample(pileup)
        seg = proc_sample(pre)
        assert seg.sigma_logor2 == pytest.approx(np.sqrt(2.0), abs=0.25)

        res = em_cncf(seg)
        assert res.is_wgd is False
        assert res.purity is not None
        assert abs(res.purity - alpha) < 0.05

        segs = res.to_dataframe()
        assert len(segs) == 4
        assert segs["segclust"].nunique() == 4
        chr3 = segs[segs["chrom"] == "chr3"].iloc[0]
        chr4 = segs[segs["chrom"] == "chr4"].iloc[0]
        assert int(chr3["segclust"]) != int(chr4["segclust"])
        assert (int(chr3["c_major"]), int(chr3["c_minor"])) == (2, 1)
        assert (int(chr4["c_major"]), int(chr4["c_minor"])) == (2, 1)
        assert bool(chr3["is_subclonal"]) is False
        assert bool(chr4["is_subclonal"]) is True
        assert abs(float(chr3["cf_em"]) - alpha) < 0.05
        assert abs(float(chr4["cf_em"]) - phi_sub) < 0.05


def test_wgd_3_2_at_alpha_055_under_nb_depth_dispersion() -> None:
    """
    Regression test for WGD (2,2)+(4,2)+(2,0)+(3,2) at alpha=0.55 under realistic
    overdispersed depth (nb_size=80.0): sigma_logor2 must not inflate from the
    imbalanced arms, enabling WGD detection, ploidy ~4.2, and exact ASCN calls.
    """
    alpha = 0.55
    specs = [
        ("chr1", 1_000_000, 80_000_000, 500, 2, 2, alpha),
        ("chr2", 1_000_000, 50_000_000, 400, 4, 2, alpha),
        ("chr3", 1_000_000, 50_000_000, 400, 2, 0, alpha),
        ("chr4", 1_000_000, 45_000_000, 400, 3, 2, alpha),
    ]
    for seed in (10, 20, 30):
        pileup = _simulate_pileup(
            specs,
            seed=seed,
            nor_mean_dp=95.0,
            tum_mean_dp=105.0,
            het_prob=0.45,
            nb_size=80.0,
        )
        pre = pre_proc_sample(pileup)
        seg = proc_sample(pre)
        assert seg.sigma_logor2 == pytest.approx(np.sqrt(2.0), abs=0.25)

        res = em_cncf(seg)
        assert res.is_wgd is True
        assert res.is_diploid_null is False
        assert res.purity is not None
        assert abs(res.purity - alpha) < 0.05
        assert abs(res.ploidy - 4.20) < 0.10
        segs = res.to_dataframe()
        calls = [(int(r["c_major"]), int(r["c_minor"])) for _, r in segs.iterrows()]
        assert calls == [(2, 2), (4, 2), (2, 0), (3, 2)]

    # Also verify 0%-balanced genome falls back cleanly to sqrt(2.0)
    asym_specs = [
        ("chr1", 1_000_000, 70_000_000, 500, 2, 1, 0.70),
        ("chr2", 1_000_000, 60_000_000, 450, 1, 0, 0.70),
        ("chr3", 1_000_000, 50_000_000, 400, 2, 0, 0.70),
    ]
    seg_asym = proc_sample(
        pre_proc_sample(
            _simulate_pileup(asym_specs, seed=403, nor_mean_dp=95.0, tum_mean_dp=105.0, nb_size=80.0)
        )
    )
    assert seg_asym.sigma_logor2 == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_settings_validation_and_r_default_procreads_equations() -> None:
    """
    Verify strict `settings` validation and `mskcc/facets` `procreads.R` equations
    under `settings="R_default"`:
      - `lowess_frac` defaults to 0.20 (vs 0.25 in default)
      - depth ratio `log2(1 + rCountT * tscl) - log2(1 + rCountN) - gcbias`
      - unmatched mode Tukey +1/6 offset in `valor`/`lorvar` + `rCountT >= 50` het gate
    """
    specs = [("chr1", 1_000_000, 40_000_000, 120, 1, 1, 0.0)]
    pileup = _simulate_pileup(specs, seed=930)
    pre_def = preprocess_sample(pileup, settings="default")
    seg_def = proc_sample(pre_def, settings="default")

    with pytest.raises(ValueError, match="settings"):
        FacetsConfig(settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        preprocess_sample(pileup, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        proc_sample(pre_def, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        em_cncf(seg_def, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        run_facets(pileup, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        run_ascat(seg_def, settings="bad_mode")  # type: ignore[arg-type]

    assert FacetsConfig(settings="default").effective_lowess_frac == pytest.approx(0.25)
    assert FacetsConfig(settings="R_default").effective_lowess_frac == pytest.approx(0.20)

    # Exact procreads.R depth ratio check without GC bias
    t_dp = np.array([0.0, 50.0, 120.0, 230.0], dtype=np.float64)
    n_dp = np.array([40.0, 60.0, 100.0, 200.0], dtype=np.float64)
    cnlr_r, cnlr_raw_r, _ = compute_cnlr(t_dp, n_dp, gc=None, settings="R_default")
    tscl = float(np.sum(n_dp) / np.sum(t_dp))
    expected_cnlr_r = np.log2(1.0 + t_dp * tscl) - np.log2(1.0 + n_dp)
    np.testing.assert_allclose(cnlr_r, expected_cnlr_r, atol=1e-12)
    np.testing.assert_allclose(cnlr_raw_r, expected_cnlr_r, atol=1e-12)

    # Exact Tukey +1/6 unmatched valor/lorvar vs Haldane +0.5 default
    ta = np.array([30.0, 0.0, 18.0], dtype=np.float64)
    tr = np.array([25.0, 55.0, 42.0], dtype=np.float64)
    val_r, lv_r = compute_valor(ta, tr, unmatched=True, settings="R_default")
    np.testing.assert_allclose(
        val_r, np.log((ta + 1.0 / 6.0) / (tr + 1.0 / 6.0)), atol=1e-12
    )
    np.testing.assert_allclose(
        lv_r, 1.0 / (ta + 1.0 / 6.0) + 1.0 / (tr + 1.0 / 6.0), atol=1e-12
    )

    # Unmatched rCountT >= 50 heterozygous SNP gate
    unm_df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 20,
            "pos": np.arange(1_000_000, 1_000_000 + 20 * 10_000, 10_000, dtype=np.int64),
            "nor_dp": np.full(20, 80, dtype=np.int64),
            "tum_dp": np.array([45] * 10 + [60] * 10, dtype=np.int64),
            "nor_alt": np.full(20, 40, dtype=np.int64),
            "tum_alt": np.array([22] * 10 + [30] * 10, dtype=np.int64),
        }
    )
    pre_unm_def = preprocess_sample(unm_df, unmatched=True, settings="default")
    pre_unm_r = preprocess_sample(unm_df, unmatched=True, settings="R_default")
    assert pre_unm_def.n_het == 20
    assert pre_unm_r.n_het == 10
    assert (pre_unm_r.snps.loc[pre_unm_r.snps["tum_dp"] < 50, "het"] == False).all()  # noqa: E712
    assert (pre_unm_r.snps.loc[pre_unm_r.snps["tum_dp"] >= 50, "het"] == True).all()  # noqa: E712


def test_r_default_cbs2rank_clustering_and_emcncf_end_to_end() -> None:
    """
    Verify `settings="R_default"` rank-based bivariate CBS (`cbs2rank`),
    OCN (<0.04) + Mann-Whitney (p > 0.001) segment clustering, and `emcncf`
    surrogate moment EM end-to-end on planted ASCN + outlier-contaminated data.
    """
    alpha = 0.65
    specs = [
        ("chr1", 1_000_000, 50_000_000, 500, 1, 1, alpha),
        ("chr1", 50_500_000, 95_000_000, 450, 1, 0, alpha),   # LOH
        ("chr2", 1_000_000, 50_000_000, 500, 2, 0, alpha),   # cnLOH
        ("chr2", 50_500_000, 95_000_000, 450, 2, 1, alpha),  # gain
        ("chr3", 1_000_000, 50_000_000, 500, 1, 1, alpha),
        ("chr3", 50_500_000, 85_000_000, 400, 3, 1, alpha),  # amp
        ("chr4", 1_000_000, 50_000_000, 500, 1, 1, alpha),
    ]
    pileup = _simulate_pileup(
        specs, seed=940, nor_mean_dp=90.0, tum_mean_dp=100.0, het_prob=0.48, gc_amplitude=0.30
    )
    # Inject extreme somatic/genotyping allelic outliers in diploid chr1
    het_chr1 = np.flatnonzero(
        (pileup["chrom"] == "chr1")
        & (pileup["pos"] < 40_000_000)
        & (np.abs(pileup["nor_alt"] / pileup["nor_dp"] - 0.5) < 0.15)
    )
    for out_idx in het_chr1[:3]:
        pileup.loc[out_idx, "tum_alt"] = pileup.loc[out_idx, "tum_dp"]

    res_r = run_facets(pileup, settings="R_default")
    assert res_r.is_diploid_null is False
    assert res_r.purity is not None
    assert abs(res_r.purity - alpha) < 0.06

    segs_r = res_r.to_dataframe()
    # Flanking/replicate diploid (1,1) segments on chr1, chr3, chr4 should share one cluster
    dip_clusts = set(
        segs_r.loc[(segs_r["c_major"] == 1) & (segs_r["c_minor"] == 1), "segclust"].tolist()
    )
    assert len(dip_clusts) == 1
    # cnLOH on chr2 must NOT merge with diploid (1,1) despite matching OCN ~ 1.0
    cnloh_rows = segs_r[(segs_r["chrom"] == "chr2") & (segs_r["end"] <= 51_000_000)]
    assert len(cnloh_rows) >= 1
    assert (cnloh_rows["c_major"] == 2).all() and (cnloh_rows["c_minor"] == 0).all()
    assert not set(cnloh_rows["segclust"].tolist()).intersection(dip_clusts)


def test_r_default_run_ascat_gamma055_rho105_and_cascading_tiers() -> None:
    """
    Verify `run_ascat(..., settings="R_default")` uses default `gamma=0.55`,
    `rho` grid `[0.10, 1.05]`, `0.05` balanced-segment weight, and 4-tier cascading.
    """
    true_rho = 0.60
    seg_states = [
        ("chr1", 1, 60_000_000, 1, 1),
        ("chr2", 1, 30_000_000, 2, 1),
        ("chr3", 1, 25_000_000, 1, 0),
        ("chr4", 1, 20_000_000, 2, 0),
        ("chr5", 1, 20_000_000, 3, 1),
    ]
    lengths = np.array([e - s + 1 for _, s, e, *_ in seg_states], dtype=np.float64)
    true_ploidy = float(
        np.sum(lengths * np.array([ma + mi for *_, ma, mi in seg_states])) / np.sum(lengths)
    )
    rows_055 = [
        {
            "chrom": ch,
            "start": s,
            "end": e,
            "logr": ascat_logr(ma, mi, true_rho, true_ploidy, gamma_ascat=0.55),
            "baf": ascat_baf(ma, mi, true_rho, minor_allele=True),
            "nhet": 120,
        }
        for ch, s, e, ma, mi in seg_states
    ]
    seg_df = pd.DataFrame(rows_055)

    # Without specifying gamma_ascat, settings="R_default" defaults to gamma=0.55
    ares_r = run_ascat(seg_df, settings="R_default")
    assert ares_r.gamma_ascat == pytest.approx(0.55)
    assert abs(ares_r.purity - true_rho) < 0.03
    assert abs(ares_r.ploidy - true_ploidy) < 0.10
    assert list(ares_r.to_dataframe()["n_major"]) == [r[3] for r in seg_states]
    assert list(ares_r.to_dataframe()["n_minor"]) == [r[4] for r in seg_states]

    # Pure tumor (rho=1.0) with slight over-compressed BAF/logR where continuous optimum sits near ~1.02
    rows_pure = [
        {
            "chrom": ch,
            "start": s,
            "end": e,
            "logr": ascat_logr(ma, mi, 1.0, true_ploidy, gamma_ascat=0.55) * 1.015,
            "baf": max(1e-4, ascat_baf(ma, mi, 1.0, minor_allele=True) * 0.99),
            "nhet": 120,
        }
        for ch, s, e, ma, mi in seg_states
    ]
    ares_pure = run_ascat(pd.DataFrame(rows_pure), settings="R_default")
    assert ares_pure.purity == pytest.approx(1.0, abs=0.03)
    assert abs(ares_pure.ploidy - true_ploidy) < 0.15



