"""
Comprehensive unit tests for `purecn` covering GC + SVD Tangent coverage
normalization, CBS segmentation, joint 2D (purity, ploidy) optimization,
Beta-Binomial somatic-vs-germline classification, CN-LOH, CCF, TMB, CIN,
and `"default"` vs `"R_default"` parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purecn import (
    CINValue,
    CoverageNormalizationResult,
    PureCN,
    PureCNResult,
    PureCNSolution,
    TMBValue,
    beta_binomial_logpmf,
    call_cin,
    call_tmb,
    compute_somatic_priors,
    estimate_overdispersion,
    expected_log2_ratio,
     get_purecn_config,
    grid_search_purity_ploidy,
    normalize_tumor_only_coverage,
    predict_somatic,
    run_purecn,
)


def _simulate_panel_tumor(
    *,
    purity: float = 0.60,
    seed: int = 42,
    n_normals: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Simulate targeted clinical panel tumor coverage, Panel of Normals with systematic
    bait-level capture effects + GC bias, and germline + somatic variants.

    Segment design (equal 80 bins each across 4 chromosomes):
      - chr1: C=2, M=1 (balanced diploid)
      - chr2: C=3, M=1 (single-copy gain)
      - chr3: C=1, M=0 (hemizygous deletion LOH)
      - chr4: C=2, M=0 (copy-neutral LOH)
    True average ploidy = (2 + 3 + 1 + 2) / 4 = 2.0.
    """
    rng = np.random.default_rng(seed)
    seg_specs = [
        ("chr1", 2, 1, 80),
        ("chr2", 3, 1, 80),
        ("chr3", 1, 0, 80),
        ("chr4", 2, 0, 80),
    ]
    true_ploidy = 2.0
    bulk_denom = purity * true_ploidy + 2.0 * (1.0 - purity)

    n_total = sum(s[3] for s in seg_specs)
    bait_effect = rng.normal(0.0, 0.35, size=n_total)  # Shared systematic capture profile
    pon_latent = rng.normal(0.0, 0.15, size=(n_total, 2))

    cov_rows = []
    var_rows = []
    bin_cursor = 0
    for chrom, c_tot, m_min, n_bins in seg_specs:
        m_maj = c_tot - m_min
        starts = np.arange(100_000, 100_000 + n_bins * 50_000, 50_000, dtype=np.int64)
        ends = starts + 500
        gc = rng.uniform(0.30, 0.70, size=n_bins)
        gc_bias = 0.45 * np.sin(2.0 * np.pi * (gc - 0.30)) - 0.30 * (gc - 0.50) ** 2

        local_bait = bait_effect[bin_cursor : bin_cursor + n_bins]
        local_lat = pon_latent[bin_cursor : bin_cursor + n_bins] @ np.array([0.8, -0.5])

        seg_ratio = (purity * c_tot + 2.0 * (1.0 - purity)) / bulk_denom
        log2_true = np.log2(seg_ratio) + gc_bias + local_bait + local_lat
        noise = rng.normal(0.0, 0.06, size=n_bins)
        tum_cov = 400.0 * np.power(2.0, log2_true + noise)

        for b_i in range(n_bins):
            cov_rows.append(
                {
                    "chrom": chrom,
                    "start": int(starts[b_i]),
                    "end": int(ends[b_i]),
                    "gc": float(gc[b_i]),
                    "coverage": float(tum_cov[b_i]),
                }
            )

        # Simulate 10 germline het SNPs per chromosome
        seg_denom = purity * c_tot + 2.0 * (1.0 - purity)
        snp_pos = starts[:10] + 100
        for p_idx, pos in enumerate(snp_pos):
            dp = 350
            g_copies = m_maj if (p_idx % 2 == 0) else m_min
            exp_af = (purity * g_copies + (1.0 - purity)) / seg_denom
            alt = int(rng.binomial(dp, exp_af))
            var_rows.append(
                {
                    "chrom": chrom,
                    "pos": int(pos),
                    "depth": dp,
                    "alt_count": alt,
                    "pop_af": 0.18,
                    "is_dbsnp": True,
                    "cosmic_cnt": 0,
                    "truth_class": "germline_het",
                }
            )

        # Simulate 4 clonal somatic SNVs (m=1) per chromosome
        som_pos = starts[20:24] + 150
        for pos in som_pos:
            dp = 400
            exp_af = (purity * 1.0) / seg_denom
            alt = int(rng.binomial(dp, exp_af))
            var_rows.append(
                {
                    "chrom": chrom,
                    "pos": int(pos),
                    "depth": dp,
                    "alt_count": alt,
                    "pop_af": 0.0,
                    "is_dbsnp": False,
                    "cosmic_cnt": 0,
                    "truth_class": "somatic_clonal",
                }
            )

        bin_cursor += n_bins

    tumor_cov_df = pd.DataFrame(cov_rows)
    gc_all = tumor_cov_df["gc"].to_numpy()
    gc_wave_all = 0.45 * np.sin(2.0 * np.pi * (gc_all - 0.30)) - 0.30 * (gc_all - 0.50) ** 2

    # Build Panel of Normals sharing bait_effect + latent capture modes + GC bias
    pon_cols = {}
    for j in range(n_normals):
        weights = rng.normal(0.0, 1.0, size=2)
        n_log2 = (
            gc_wave_all
            + bait_effect
            + pon_latent @ weights
            + rng.normal(0.0, 0.04, size=n_total)
        )
        pon_cols[f"normal_{j}"] = 380.0 * np.power(2.0, n_log2)
    pon_df = pd.DataFrame(pon_cols)

    # Add 1 COSMIC hotspot somatic driver on chr1 and 2 subclonal SNVs (CCF=0.40) on chr1
    chr1_denom = purity * 2.0 + 2.0 * (1.0 - purity)
    var_rows.append(
        {
            "chrom": "chr1",
            "pos": 2_500_200,
            "depth": 500,
            "alt_count": int(round(500 * (purity * 1.0) / chr1_denom)),
            "pop_af": 0.0,
            "is_dbsnp": False,
            "cosmic_cnt": 15,
            "truth_class": "somatic_hotspot",
        }
    )
    var_rows.append(
        {
            "chrom": "chr1",
            "pos": 2_600_200,
            "depth": 600,
            "alt_count": int(round(600 * (purity * 0.40) / chr1_denom)),
            "pop_af": 0.0,
            "is_dbsnp": False,
            "cosmic_cnt": 0,
            "truth_class": "somatic_subclonal",
        }
    )

    variants_df = pd.DataFrame(var_rows)
    return tumor_cov_df, pon_df, variants_df, true_ploidy


def test_beta_binomial_pmf_normalizes_and_limits():
    n = 40
    ks = np.arange(n + 1)
    for rho in (0.0, 1e-4, 0.01, 0.10):
        lp = beta_binomial_logpmf(ks, n, 0.35, rho_overdisp=rho)
        total_prob = float(np.sum(np.exp(lp)))
        assert np.isclose(total_prob, 1.0, atol=1e-6)

    # Overdispersed Beta-Binomial assigns higher tail mass than pure Binomial
    tail_bin = float(beta_binomial_logpmf(32, 40, 0.5, rho_overdisp=0.0))
    tail_bb = float(beta_binomial_logpmf(32, 40, 0.5, rho_overdisp=0.02))
    assert tail_bb > tail_bin


def test_expected_log2_ratio_identities():
    # Balanced state C == ploidy always yields 0.0 log2 ratio for any purity
    for rho in (0.2, 0.5, 0.85):
        assert np.isclose(float(expected_log2_ratio(2, rho, 2.0)), 0.0)
        assert np.isclose(float(expected_log2_ratio(4, rho, 4.0)), 0.0)
    # Single-copy gain C=3 in diploid tumor at purity=0.6 -> (1.8 + 0.8) / 2.0 = 1.3
    assert np.isclose(float(expected_log2_ratio(3, 0.6, 2.0)), np.log2(1.3))


def test_coverage_gc_and_pon_tangent_normalization_reduces_noise():
    tumor_cov_df, pon_df, _, _ = _simulate_panel_tumor(purity=0.60, seed=11)

    res_no_pon = normalize_tumor_only_coverage(
        tumor_cov_df, normal_panel=None, gc_correct=True, tangent_norm=False, settings="default"
    )
    res_pon = normalize_tumor_only_coverage(
        tumor_cov_df, normal_panel=pon_df, gc_correct=True, tangent_norm=True, settings="default"
    )

    assert isinstance(res_pon, CoverageNormalizationResult)
    assert res_pon.gc_corrected is True
    assert res_pon.tangent_corrected is True
    # Tangent PoN subtraction must substantially reduce bin-to-bin systematic noise
    assert res_pon.noise_sd < 0.35 * res_no_pon.noise_sd
    assert len(res_pon.segments) >= 4
    assert "PureCN CoverageNormalizationResult" in res_pon.summary()


def test_grid_search_recovers_diploid_purity_and_ploidy():
    tumor_cov_df, pon_df, variants_df, true_ploidy = _simulate_panel_tumor(
        purity=0.60, seed=21
    )
    cov_res = normalize_tumor_only_coverage(tumor_cov_df, normal_panel=pon_df)
    solutions = grid_search_purity_ploidy(
        cov_res.segments, variants_df=variants_df, settings="default"
    )

    assert len(solutions) >= 1
    best = solutions[0]
    assert isinstance(best, PureCNSolution)
    assert abs(best.purity - 0.60) <= 0.05
    assert abs(best.ploidy - true_ploidy) <= 0.20


def test_grid_search_recovers_triploid_aneuploid_tumor():
    rng = np.random.default_rng(99)
    true_purity = 0.65
    # Equal-sized segments with C in {2, 3, 3, 4} -> true ploidy = 3.0
    c_truth = [2, 3, 3, 4]
    true_ploidy = float(np.mean(c_truth))
    denom = true_purity * true_ploidy + 2.0 * (1.0 - true_purity)

    seg_rows = []
    var_rows = []
    for idx, c_val in enumerate(c_truth, start=1):
        chrom = f"chr{idx}"
        seg_l2 = float(expected_log2_ratio(c_val, true_purity, true_ploidy))
        seg_rows.append(
            {
                "chrom": chrom,
                "start": 1,
                "end": 20_000_000,
                "num_mark": 80,
                "seg_mean": seg_l2,
                "seg_sd": 0.07,
                "size_bp": 20_000_000,
            }
        )
        loc_den = true_purity * c_val + 2.0 * (1.0 - true_purity)
        # Add clonal somatic m=1 and m=2 variants + germline het SNPs
        for k in range(6):
            dp = 450
            af_s = (true_purity * 1.0) / loc_den
            var_rows.append(
                {
                    "chrom": chrom,
                    "pos": 100_000 + k * 10_000,
                    "depth": dp,
                    "alt_count": int(rng.binomial(dp, af_s)),
                    "pop_af": 0.0,
                    "is_dbsnp": False,
                }
            )
        m_min = c_val // 2
        for k in range(8):
            dp = 400
            g_c = (c_val - m_min) if k % 2 == 0 else m_min
            af_g = (true_purity * g_c + (1.0 - true_purity)) / loc_den
            var_rows.append(
                {
                    "chrom": chrom,
                    "pos": 500_000 + k * 10_000,
                    "depth": dp,
                    "alt_count": int(rng.binomial(dp, af_g)),
                    "pop_af": 0.25,
                    "is_dbsnp": True,
                }
            )
    assert denom > 0
    sols = grid_search_purity_ploidy(
        pd.DataFrame(seg_rows), variants_df=pd.DataFrame(var_rows), settings="default"
    )
    best = sols[0]
    assert abs(best.purity - true_purity) <= 0.05
    assert abs(best.ploidy - true_ploidy) <= 0.25


def test_predict_somatic_classifies_somatic_germline_cnloh_and_ccf():
    tumor_cov_df, pon_df, variants_df, _ = _simulate_panel_tumor(purity=0.60, seed=7)
    cov_res = normalize_tumor_only_coverage(tumor_cov_df, normal_panel=pon_df)
    sols = grid_search_purity_ploidy(cov_res, variants_df=variants_df)
    best = sols[0]

    pred = predict_somatic(
        variants_df, best.segments, purity=best.purity, ploidy=best.ploidy, settings="default"
    )
    required_cols = {
        "posterior_somatic",
        "ml_somatic",
        "ml_C",
        "ml_M",
        "loh",
        "allelic_imbalance",
        "cell_fraction",
    }
    assert required_cols.issubset(pred.columns)

    truth = variants_df["truth_class"].to_numpy()
    germ_calls = pred.loc[truth == "germline_het", "ml_somatic"].to_numpy()
    som_calls = pred.loc[truth == "somatic_clonal", "ml_somatic"].to_numpy()
    assert np.mean(~germ_calls) >= 0.95
    assert np.mean(som_calls) >= 0.90

    # Verify CN-LOH on chr4 (C=2, M=0 -> loh=True, allelic_imbalance == 1.0)
    chr4_rows = pred[pred["chrom"] == "chr4"]
    assert np.all(chr4_rows["ml_C"] == 2)
    assert np.all(chr4_rows["ml_M"] == 0)
    assert np.all(chr4_rows["loh"])
    assert np.allclose(chr4_rows["allelic_imbalance"], 1.0)

    # Verify balanced diploid on chr1 (C=2, M=1 -> loh=False, allelic_imbalance == 0.0)
    chr1_rows = pred[pred["chrom"] == "chr1"]
    assert np.all(chr1_rows["ml_C"] == 2)
    assert np.all(~chr1_rows["loh"])
    assert np.allclose(chr1_rows["allelic_imbalance"], 0.0)

    # Verify subclonal variant CCF ~ 0.40 on chr1
    sub_row = pred.loc[truth == "somatic_subclonal"].iloc[0]
    assert bool(sub_row["ml_somatic"]) is True
    assert 0.25 <= float(sub_row["cell_fraction"]) <= 0.55

    # Clonal somatic SNVs should have CCF near 1.0
    clonal_chr1 = pred.loc[(truth == "somatic_clonal") & (pred["chrom"] == "chr1")]
    assert float(clonal_chr1["cell_fraction"].median()) > 0.85


def test_somatic_priors_and_overdispersion_default_vs_r_default():
    vdf = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [100, 200, 300, 400],
            "depth": [200, 200, 200, 200],
            "alt_count": [60, 95, 100, 70],
            "pop_af": [0.0, 0.0, 0.05, 0.0],
            "is_dbsnp": [False, True, True, False],
            "cosmic_cnt": [0, 0, 0, 5],
        }
    )
    from purecn.likelihood import standardize_variants_df

    vstd = standardize_variants_df(vdf)
    p_def = compute_somatic_priors(vstd, get_purecn_config("default"))
    p_r = compute_somatic_priors(vstd, get_purecn_config("R_default"))

    # Novel: 0.5 in both; dbSNP: 0.05 vs 0.10; Common SNP: 0.001 vs 0.0005; COSMIC>=2: 0.98 vs 0.999
    assert np.isclose(p_def[0], 0.5) and np.isclose(p_r[0], 0.5)
    assert np.isclose(p_def[1], 0.05) and np.isclose(p_r[1], 0.10)
    assert np.isclose(p_def[2], 0.001) and np.isclose(p_r[2], 0.0005)
    assert np.isclose(p_def[3], 0.98) and np.isclose(p_r[3], 0.999)

    # Empirical overdispersion from overdispersed het SNPs
    rng = np.random.default_rng(123)
    p_draws = rng.beta(20.0, 20.0, size=30)  # rho = 1 / (40 + 1) ~ 0.024
    dps = np.full(30, 300)
    alts = rng.binomial(dps, p_draws)
    het_df = standardize_variants_df(
        pd.DataFrame(
            {
                "chrom": "chr1",
                "pos": np.arange(1, 31) * 1000,
                "depth": dps,
                "alt_count": alts,
                "pop_af": 0.25,
                "is_dbsnp": True,
            }
        )
    )
    est_rho = estimate_overdispersion(het_df)
    assert 0.005 <= est_rho <= 0.05


def test_call_tmb_default_vs_r_default_and_hotspot_filtering():
    var_res = pd.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "pos": [100, 200, 300, 400, 500],
            "af": [0.25, 0.20, 0.30, 0.48, 0.01],
            "alt_count": [50, 40, 60, 96, 2],
            "depth": [200, 200, 200, 200, 200],
            "pop_af": [0.0, 0.0, 0.0, 0.15, 0.0],
            "prior_somatic": [0.5, 0.5, 0.999, 0.001, 0.5],
            "posterior_somatic": [0.95, 0.65, 0.999, 0.001, 0.90],
            "ml_somatic": [True, True, True, False, True],
            "cosmic_cnt": [0, 0, 10, 0, 0],
        }
    )

    # default: keeps pos 100 (0.95) and pos 200 (0.65); drops pos 300 (hotspot), pos 400 (germline), pos 500 (low AF)
    tmb_def = call_tmb(var_res, callable_mb=2.0, exclude_hotspot=True, settings="default")
    assert isinstance(tmb_def, TMBValue)
    assert isinstance(tmb_def, float)
    assert tmb_def.n_mutations == 2
    assert np.isclose(float(tmb_def), 1.0)

    # R_default: requires posterior_somatic >= 0.8 and prior_somatic <= 0.9 -> keeps only pos 100
    tmb_r = call_tmb(var_res, callable_mb=2.0, exclude_hotspot=True, settings="R_default")
    assert tmb_r.n_mutations == 1
    assert np.isclose(float(tmb_r), 0.5)

    # Including hotspots restores pos 300
    tmb_with_hot = call_tmb(var_res, callable_mb=2.0, exclude_hotspot=False, settings="R_default")
    assert tmb_with_hot.n_mutations == 2


def test_call_cin_ploidy_adjusted_and_loh():
    segs = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3", "chr4"],
            "start": [1, 1, 1, 1],
            "end": [10_000_000] * 4,
            "seg_mean": [0.0, 0.3785, -0.415, 0.0],
            "C": [2, 3, 1, 2],
            "M": [1, 1, 0, 0],
            "loh": [False, False, True, True],
            "size_bp": [10_000_000] * 4,
        }
    )
    cin = call_cin(segs, purity=0.60, ploidy=2.0, settings="default")
    assert isinstance(cin, CINValue)
    # chr2 (C=3), chr3 (C=1, LOH), chr4 (C=2, CN-LOH) are altered; chr1 (C=2, M=1) is normal -> 3/4 = 0.75
    assert np.isclose(float(cin), 0.75)


def test_end_to_end_run_purecn_and_class_interface():
    tumor_cov_df, pon_df, variants_df, _ = _simulate_panel_tumor(purity=0.60, seed=55)

    model = PureCN(settings="R_default")
    res = model.run(
        tumor_cov_df,
        variants_df=variants_df,
        normal_panel=pon_df,
        callable_mb=2.0,
    )
    assert isinstance(res, PureCNResult)
    assert abs(res.purity - 0.60) <= 0.05
    assert abs(res.ploidy - 2.0) <= 0.25
    assert float(res.tmb) > 0.0
    assert 0.50 <= float(res.cin) <= 0.90
    assert 0.35 <= res.loh_fraction <= 0.65
    assert "PureCNResult" in res.summary()
    assert len(res.to_dataframe()) >= 4

    # Also verify functional `run_purecn` directly on pre-segmented input
    res2 = run_purecn(segments_df=res.segments, variants_df=variants_df, callable_mb=2.0)
    assert abs(res2.purity - res.purity) <= 0.05


def test_r_default_max_non_clonal_penalty():
    # Construct segments with noisy intermediate subclonal log2 shifts that violate clonal states at high purity
    segs = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3", "chr4"],
            "start": [1, 1, 1, 1],
            "end": [25_000_000] * 4,
            "num_mark": [60, 60, 60, 60],
            "seg_mean": [0.0, np.log2(1.25), np.log2(0.75), 0.0],
            "seg_sd": [0.05, 0.05, 0.05, 0.05],
            "size_bp": [25_000_000] * 4,
        }
    )
    sols_r = grid_search_purity_ploidy(segs, settings="R_default")
    assert sols_r[0].subclonal_fraction <= 0.20
    assert 0.15 <= sols_r[0].purity <= 0.95


def test_invalid_settings_and_edge_cases_raise_value_error():
    segs = pd.DataFrame(
        {"chrom": ["chr1"], "start": [1], "end": [100_000], "seg_mean": [0.0]}
    )
    cov = pd.DataFrame(
        {"chrom": ["chr1"], "start": [1], "end": [1000], "coverage": [100.0]}
    )
    vdf = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [500], "depth": [100], "alt_count": [30]}
    )

    for bad_setting in ("invalid", "R", "python", ""):
        with pytest.raises(ValueError, match="Unknown settings"):
            normalize_tumor_only_coverage(cov, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            grid_search_purity_ploidy(segs, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            predict_somatic(vdf, segs, 0.5, 2.0, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            call_tmb(pd.DataFrame(), 1.0, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            call_cin(segs, 0.5, 2.0, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            run_purecn(segments_df=segs, settings=bad_setting)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            PureCN(settings=bad_setting)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="callable_mb"):
        call_tmb(pd.DataFrame(), callable_mb=0.0)

    # Empty variants DataFrame returns typed empty columns cleanly
    empty_pred = predict_somatic(pd.DataFrame(), segs, purity=0.5, ploidy=2.0)
    assert empty_pred.empty
    assert "posterior_somatic" in empty_pred.columns
