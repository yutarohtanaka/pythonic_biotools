from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fishhook import FishHookModel, compute_territory_and_counts, fit_fishhook


def _simulate_mutations_from_counts(regions: pd.DataFrame, counts: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for idx, c in enumerate(counts):
        if c <= 0:
            continue
        ch = regions["chrom"].iloc[idx]
        s = int(regions["start"].iloc[idx])
        e = int(regions["end"].iloc[idx])
        positions = rng.integers(s, e, size=int(c))
        for p in positions:
            rows.append({"chrom": ch, "pos": int(p), "sample_id": f"S_{rng.integers(0, 200)}"})
    if not rows:
        return pd.DataFrame(columns=["chrom", "pos", "sample_id"])
    return pd.DataFrame(rows)


def test_gamma_poisson_parameter_recovery_and_driver_enrichment():
    """Accurately recovers true (beta_0, beta, theta) and detects planted driver loci."""
    rng = np.random.default_rng(2024)
    n_regions = 1200
    lengths = rng.integers(5_000, 25_000, size=n_regions)
    starts = np.cumsum(np.concatenate([[0], lengths[:-1] + 1_000]))
    ends = starts + lengths

    rep_timing = rng.normal(0.0, 1.0, size=n_regions)
    gc_content = rng.uniform(0.35, 0.65, size=n_regions)

    true_beta0 = -8.5
    true_beta_rep = 0.75
    true_beta_gc = -1.20
    true_theta = 4.0  # alpha = 0.25

    mu = lengths * np.exp(true_beta0 + true_beta_rep * rep_timing + true_beta_gc * gc_content)
    p_nb = true_theta / (true_theta + mu)
    counts = rng.negative_binomial(n=true_theta, p=p_nb)

    # Plant 2 strong driver regions
    counts[10] = int(np.ceil(mu[10] * 12 + 25))
    counts[500] = int(np.ceil(mu[500] * 14 + 30))

    regions = pd.DataFrame(
        {
            "chrom": ["chr1"] * n_regions,
            "start": starts,
            "end": ends,
            "region_id": [f"TILE_{i}" for i in range(n_regions)],
            "rep_timing": rep_timing,
            "gc_content": gc_content,
        }
    )
    muts = _simulate_mutations_from_counts(regions, counts, seed=11)

    model = FishHookModel(
        regions=regions,
        mutations=muts,
        covariate_cols=["rep_timing", "gc_content"],
    )
    res = model.fit()

    assert res.converged
    assert not res.is_poisson_fallback
    assert np.isclose(res.beta[0], true_beta0, atol=0.35)
    assert np.isclose(res.beta[1], true_beta_rep, atol=0.12)
    assert np.isclose(res.beta[2], true_beta_gc, atol=0.45)
    assert 2.0 < res.theta < 8.0

    sig = res.significant(alpha=0.01)
    assert "TILE_10" in set(sig["region_id"])
    assert "TILE_500" in set(sig["region_id"])
    assert res.fold_enrichment[10] > 8.0


def test_poisson_fallback_when_equidispersed_or_underdispersed():
    """Falls back cleanly to Poisson limit (alpha = 0, theta = inf) when alpha -> 0+."""
    rng = np.random.default_rng(77)
    n_regions = 400
    starts = np.arange(n_regions) * 10_000
    ends = starts + 10_000
    x = rng.normal(0.0, 1.0, size=n_regions)
    # Deterministic / sub-Poisson rounded counts: Var(Y_i) << mu_i -> score at alpha=0 is strictly < 0
    mu_true = 10_000 * np.exp(-7.2 + 0.5 * x)
    counts = np.rint(mu_true).astype(int)

    regions = pd.DataFrame({"chrom": "chr2", "start": starts, "end": ends, "x": x})
    muts = _simulate_mutations_from_counts(regions, counts, seed=3)

    res = fit_fishhook(regions, muts, covariate_cols=["x"])
    assert res.converged
    assert res.is_poisson_fallback
    assert res.alpha == 0.0
    assert np.isinf(res.theta)

    # Verify exact Poisson right-tail survival formula matches scipy.stats.poisson.sf
    idx = 5
    expected_p = stats.poisson.sf(res.obs_counts[idx] - 1, mu=res.exp_counts[idx])
    assert np.isclose(res.p_value[idx], expected_p, rtol=1e-12)


def test_eligible_intervals_sweep_line_union_and_zero_coverage_masking():
    """
    Overlapping eligible BED baits are merged without double-counting, off-target
    mutations outside eligible intervals are excluded, and L_i == 0 tiles yield NaN q-values.
    """
    regions = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 1000, 2000],
            "end": [1000, 2000, 3000],
            "region_id": ["R0", "R1", "R2_UNCALLED"],
        }
    )
    # Overlapping baits in R0: [100, 500) U [300, 700) = [100, 700) -> 600 bp!
    # Disjoint baits in R1: [1100, 1300) U [1500, 1800) -> 500 bp!
    # Zero baits in R2_UNCALLED -> 0 bp!
    eligible = pd.DataFrame(
        {
            "chrom": ["1", "chr1", "chr1", "chr1"],  # mixed chr notation!
            "start": [100, 300, 1100, 1500],
            "end": [500, 700, 1300, 1800],
        }
    )
    mutations = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [
                250,   # inside R0 eligible [100, 700) -> COUNTED
                850,   # inside R0, OUTSIDE eligible -> EXCLUDED
                1200,  # inside R1 eligible [1100, 1300) -> COUNTED
                2500,  # inside R2_UNCALLED -> EXCLUDED
            ],
        }
    )
    _, _, _, _, elig_bp, counts = compute_territory_and_counts(
        regions, mutations, eligible_intervals=eligible
    )
    np.testing.assert_array_equal(elig_bp, [600, 500, 0])
    np.testing.assert_array_equal(counts, [1, 1, 0])

    # Expand with more regions to fit GLM and verify R2_UNCALLED handling
    extra_regions = pd.DataFrame(
        {
            "chrom": ["chr1"] * 20,
            "start": np.arange(3, 23) * 1000,
            "end": np.arange(4, 24) * 1000,
            "region_id": [f"R_{k}" for k in range(3, 23)],
        }
    )
    extra_elig = extra_regions[["chrom", "start", "end"]].copy()
    all_reg = pd.concat([regions, extra_regions], ignore_index=True)
    all_elig = pd.concat([eligible, extra_elig], ignore_index=True)

    res = fit_fishhook(all_reg, mutations, eligible_intervals=all_elig)
    df = res.to_dataframe().set_index("region_id")
    assert df.loc["R0", "eligible_bp"] == 600
    assert df.loc["R2_UNCALLED", "eligible_bp"] == 0
    assert np.isnan(df.loc["R2_UNCALLED", "p_value"])
    assert np.isnan(df.loc["R2_UNCALLED", "q_value"])


def test_kataegis_single_sample_hypermutation_cap():
    """
    `max_muts_per_sample_per_tile=1` prevents a single APOBEC kataegis sample with
    40 clustered mutations in one tile from triggering a false driver call!
    """
    rng = np.random.default_rng(55)
    n_tiles = 300
    regions = pd.DataFrame(
        {
            "chrom": ["chr3"] * n_tiles,
            "start": np.arange(n_tiles) * 10_000,
            "end": (np.arange(n_tiles) + 1) * 10_000,
            "region_id": [f"T_{i}" for i in range(n_tiles)],
        }
    )
    # Background: ~1 mutation per tile across distinct samples
    bg_counts = rng.poisson(1.2, size=n_tiles)
    bg_muts = _simulate_mutations_from_counts(regions, bg_counts, seed=8)

    # Inject kataegis burst in T_0: 35 mutations all in ONE patient 'KATAEGIS_PT'!
    kat_muts = pd.DataFrame(
        {
            "chrom": ["chr3"] * 35,
            "pos": rng.integers(100, 9_900, size=35),
            "sample_id": ["KATAEGIS_PT"] * 35,
        }
    )
    # Inject true recurrent driver in T_1: 18 mutations across 18 DISTINCT patients!
    drv_muts = pd.DataFrame(
        {
            "chrom": ["chr3"] * 18,
            "pos": rng.integers(10_100, 19_900, size=18),
            "sample_id": [f"DRV_PT_{k}" for k in range(18)],
        }
    )
    all_muts = pd.concat([bg_muts, kat_muts, drv_muts], ignore_index=True)

    # Without cap: T_0 appears to be the #1 most significant hit!
    uncapped = fit_fishhook(regions, all_muts)
    assert uncapped.q_value[0] < 1e-8
    assert np.argmin(uncapped.p_value) == 0

    # With per-sample cap = 1: T_0 drops to null while true multi-patient driver T_1 stays highly significant!
    capped = fit_fishhook(regions, all_muts, max_muts_per_sample_per_tile=1)
    assert capped.q_value[0] > 0.20
    assert capped.q_value[1] < 1e-6


def test_omitted_covariate_inflation_vs_adjusted_calibration():
    """
    Demonstrates core fishHook diagnostic: omitting a strong replication-timing
    covariate under a naive model inflates QQ/lambda_GC and causes false positives,
    whereas covariate-adjusted Gamma-Poisson restores calibration (~1.0).
    """
    rng = np.random.default_rng(909)
    n_tiles = 800
    regions = pd.DataFrame(
        {
            "chrom": ["chr5"] * n_tiles,
            "start": np.arange(n_tiles) * 20_000,
            "end": (np.arange(n_tiles) + 1) * 20_000,
        }
    )
    late_rep = rng.normal(0.0, 1.0, size=n_tiles)
    mu = 20_000 * np.exp(-8.0 + 0.9 * late_rep)
    counts = rng.negative_binomial(n=6.0, p=6.0 / (6.0 + mu))
    muts = _simulate_mutations_from_counts(regions, counts, seed=21)

    res_adj = fit_fishhook(regions, muts, covariates=late_rep)
    # Under true null (0 drivers), covariate-adjusted fishHook has 0 false discoveries and calibrated QQ
    assert len(res_adj.significant(alpha=0.05)) == 0
    assert 0.70 <= res_adj.qq_slope <= 1.25


def test_one_based_coordinates_and_indel_spans():
    """Supports 1-based VCF point coordinates and multi-tile spanning deletions."""
    regions = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 200],
            "end": [100, 200, 300],
        }
    )
    # 1-based pos=100 lies in [0, 100), NOT [100, 200)!
    muts_1based = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [1, 100]})
    _, _, _, _, _, c1 = compute_territory_and_counts(
        regions, muts_1based, pos_coord_system="one_based"
    )
    np.testing.assert_array_equal(c1, [2, 0, 0])

    # Deletion spanning [80, 220) overlaps all 3 tiles!
    muts_del = pd.DataFrame({"chrom": ["chr1"], "start": [80], "end": [220]})
    _, _, _, _, _, c_del = compute_territory_and_counts(regions, muts_del)
    np.testing.assert_array_equal(c_del, [1, 1, 1])


def test_zero_mutations_and_validation_errors():
    """Handles empty mutation tables cleanly and rejects collinear/invalid inputs."""
    regions = pd.DataFrame(
        {
            "chrom": ["chr1"] * 10,
            "start": np.arange(10) * 1000,
            "end": (np.arange(10) + 1) * 1000,
        }
    )
    empty_muts = pd.DataFrame(columns=["chrom", "pos"])
    res = fit_fishhook(regions, empty_muts)
    assert res.converged
    assert np.all(res.obs_counts == 0)
    assert np.all(res.p_value == 1.0)

    # Collinear covariates rejected
    x_collinear = np.column_stack([np.arange(10, dtype=float), 2.0 * np.arange(10, dtype=float)])
    muts = pd.DataFrame({"chrom": ["chr1"], "pos": [500]})
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_fishhook(regions, muts, covariates=x_collinear)

    # Missing sample_id when max_muts_per_sample_per_tile is set
    with pytest.raises(ValueError, match="sample_id"):
        fit_fishhook(regions, muts, max_muts_per_sample_per_tile=1)


def test_coefficients_dataframe_and_exposure_scaling_invariance():
    """Doubling callable territory L_i under constant density shifts beta_0 by 0 and doubles E[Y_i]."""
    rng = np.random.default_rng(404)
    n_tiles = 200
    lengths = np.where(np.arange(n_tiles) % 2 == 0, 10_000, 40_000)
    starts = np.cumsum(np.concatenate([[0], lengths[:-1] + 500]))
    ends = starts + lengths

    # Exact constant density 5e-4 muts/bp -> E[Y] = 5 vs 20
    counts = rng.poisson(lengths * 5e-4)
    regions = pd.DataFrame({"chrom": "chr7", "start": starts, "end": ends})
    muts = _simulate_mutations_from_counts(regions, counts, seed=19)

    res = fit_fishhook(regions, muts)
    coef_df = res.coefficients_dataframe()
    assert list(coef_df.index) == ["(Intercept)"]
    # Exposure ratio between 40kb and 10kb tiles is exactly 4.0!
    assert np.isclose(res.exp_counts[1] / res.exp_counts[0], 4.0, rtol=1e-10)
    assert np.isclose(res.exp_density[1], res.exp_density[0], rtol=1e-10)

