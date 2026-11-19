"""Comprehensive unit tests for the mrlap package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mrlap import (
    MRlap,
    MRlapResult,
    estimate_exposure_polygenicity,
    mrlap,
    truncated_moments_closed_form,
    truncated_moments_gauss_hermite,
)


def _simulate_mrlap_gwas(
    m_snps: int = 8000,
    nx: float = 40_000.0,
    ny: float = 40_000.0,
    h2_x: float = 0.25,
    pi_x: float = 0.04,
    h2_y_direct: float = 0.15,
    pi_y: float = 0.05,
    alpha_true: float = 0.30,
    c_xy: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    snps = [f"rs{i}" for i in range(m_snps)]
    ld = np.ones(m_snps, dtype=float)

    c_x = rng.binomial(1, pi_x, size=m_snps).astype(float)
    beta_x = rng.normal(0.0, 1.0, size=m_snps) * np.sqrt(
        c_x * (h2_x / (m_snps * pi_x))
    )

    c_y = rng.binomial(1, pi_y, size=m_snps).astype(float)
    gamma_y = rng.normal(0.0, 1.0, size=m_snps) * np.sqrt(
        c_y * (h2_y_direct / (m_snps * pi_y))
    )
    beta_y = alpha_true * beta_x + gamma_y

    cov_eps = np.array(
        [
            [1.0 / nx, c_xy / np.sqrt(nx * ny)],
            [c_xy / np.sqrt(nx * ny), 1.0 / ny],
        ],
        dtype=float,
    )
    eps = rng.multivariate_normal([0.0, 0.0], cov_eps, size=m_snps)

    bx_hat = beta_x + eps[:, 0]
    by_hat = beta_y + eps[:, 1]
    sx = np.full(m_snps, 1.0 / np.sqrt(nx), dtype=float)
    sy = np.full(m_snps, 1.0 / np.sqrt(ny), dtype=float)

    exp_df = pd.DataFrame({"snp": snps, "beta": bx_hat, "se": sx, "n": nx})
    out_df = pd.DataFrame({"snp": snps, "beta": by_hat, "se": sy, "n": ny})
    return exp_df, out_df, ld


def test_gauss_hermite_matches_closed_form_and_monte_carlo() -> None:
    nx = np.full(50, 35_000.0)
    ny = np.full(50, 45_000.0)
    z_thresh = float(stats.norm.isf(5e-8 / 2.0))
    eb2_cf, ebb_cf, ebey_cf, lam_cf = truncated_moments_closed_form(
        nx, ny, h2_x=0.25, pi_x=0.03, c_xy=0.35, m_eff=8000.0, z_thresh=z_thresh
    )
    eb2_gh, ebb_gh, ebey_gh, lam_gh = truncated_moments_gauss_hermite(
        nx, ny, h2_x=0.25, pi_x=0.03, c_xy=0.35, m_eff=8000.0, z_thresh=z_thresh
    )
    assert np.isclose(eb2_cf, eb2_gh, rtol=1e-10)
    assert np.isclose(ebb_cf, ebb_gh, rtol=1e-10)
    assert np.isclose(ebey_cf, ebey_gh, rtol=1e-10)
    assert np.isclose(lam_cf, lam_gh, rtol=1e-10)

    # Monte Carlo check
    rng = np.random.default_rng(0)
    n_mc = 400_000
    s2 = 0.25 / (8000.0 * 0.03)
    b_true = rng.normal(0.0, np.sqrt(s2), size=n_mc) * rng.binomial(1, 0.03, size=n_mc)
    eps = rng.multivariate_normal(
        [0.0, 0.0],
        [[1.0 / 35_000.0, 0.35 / np.sqrt(35_000.0 * 45_000.0)],
         [0.35 / np.sqrt(35_000.0 * 45_000.0), 1.0 / 45_000.0]],
        size=n_mc,
    )
    b_hat = b_true + eps[:, 0]
    sel = np.abs(b_hat * np.sqrt(35_000.0)) > z_thresh
    lam_mc = float(np.mean(b_true[sel] * b_hat[sel]) / np.mean(b_hat[sel] ** 2))
    assert abs(lam_cf - lam_mc) < 0.015


def test_winners_curse_and_weak_iv_deattenuation_under_zero_overlap() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=10_000,
        nx=12_000.0,
        ny=40_000.0,
        h2_x=0.25,
        pi_x=0.15,
        alpha_true=0.40,
        c_xy=0.0,
        seed=10,
    )
    res = mrlap(
        exp_df, out_df, ld_scores=ld, mr_threshold=1e-4, settings="default", seed=10
    )
    assert isinstance(res, MRlapResult)
    # Observed IVW is attenuated toward zero by Winner's curse + weak IVs
    assert res.lambda_attenuation < 0.80
    assert res.alpha_obs < 0.33
    # MRlap corrects the attenuation back near 0.40
    assert abs(res.alpha_corrected - 0.40) < abs(res.alpha_obs - 0.40)
    assert abs(res.alpha_corrected - 0.40) < 0.08
    assert res.p_diff < 0.05


def test_full_sample_overlap_under_null_eliminates_false_positive() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=10_000,
        nx=12_000.0,
        ny=12_000.0,
        h2_x=0.22,
        pi_x=0.18,
        alpha_true=0.0,
        c_xy=0.55,
        seed=22,
    )
    res = mrlap(
        exp_df, out_df, ld_scores=ld, mr_threshold=1e-3, settings="default", seed=22
    )
    # Standard IVW suffers strong false-positive bias toward c_xy * (1 - lambda)
    assert res.alpha_obs > 0.12
    assert res.p_obs < 1e-4
    # MRlap removes sample-overlap bias under the null
    assert abs(res.alpha_corrected) < 0.08
    assert res.p_diff < 1e-3


def test_combined_causal_effect_and_opposing_sample_overlap() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=10_000,
        nx=12_000.0,
        ny=12_000.0,
        h2_x=0.25,
        pi_x=0.15,
        alpha_true=0.35,
        c_xy=-0.45,
        seed=34,
    )
    res = mrlap(
        exp_df, out_df, ld_scores=ld, mr_threshold=1e-4, settings="R_default", seed=34
    )
    # Negative overlap + Winner's curse drags alpha_obs strongly below 0.35
    assert res.alpha_obs < 0.22
    assert abs(res.alpha_corrected - 0.35) < 0.08
    assert res.p_diff < 1e-3


def test_strong_instrument_regime_has_lambda_near_one() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=6000,
        nx=250_000.0,
        ny=250_000.0,
        h2_x=0.35,
        pi_x=0.008,
        alpha_true=0.25,
        c_xy=0.30,
        seed=45,
    )
    res = mrlap(exp_df, out_df, ld_scores=ld, mr_threshold=5e-8, settings="default")
    assert res.lambda_attenuation > 0.96
    assert abs(res.alpha_corrected - 0.25) < 0.04
    assert abs(res.alpha_obs - res.alpha_corrected) < 0.03


def test_polygenicity_truncated_mle_distinguishes_sparse_vs_polygenic() -> None:
    z_thresh = float(stats.norm.isf(1e-5 / 2.0))
    exp_sparse, _, _ = _simulate_mrlap_gwas(
        m_snps=8000, nx=35_000.0, h2_x=0.30, pi_x=0.006, seed=51
    )
    exp_poly, _, _ = _simulate_mrlap_gwas(
        m_snps=8000, nx=35_000.0, h2_x=0.30, pi_x=0.08, seed=52
    )

    sel_s = np.abs(exp_sparse["beta"] / exp_sparse["se"]) > z_thresh
    sel_p = np.abs(exp_poly["beta"] / exp_poly["se"]) > z_thresh

    pi_s, _, _ = estimate_exposure_polygenicity(
        exp_sparse.loc[sel_s, "beta"].to_numpy(),
        exp_sparse.loc[sel_s, "n"].to_numpy(),
        h2_x=0.30,
        m_eff=8000.0,
        z_thresh=z_thresh,
    )
    pi_p, _, _ = estimate_exposure_polygenicity(
        exp_poly.loc[sel_p, "beta"].to_numpy(),
        exp_poly.loc[sel_p, "n"].to_numpy(),
        h2_x=0.30,
        m_eff=8000.0,
        z_thresh=z_thresh,
    )
    assert pi_s < 0.02
    assert pi_p > 0.03
    assert pi_p > 2.5 * pi_s


def test_ld_pruning_with_correlation_matrix() -> None:
    exp_df, out_df, _ = _simulate_mrlap_gwas(
        m_snps=400, nx=80_000.0, ny=80_000.0, h2_x=0.35, pi_x=0.08, seed=60
    )
    # Create duplicate/tag blocks of size 2 with r = 0.95
    ld_R = np.eye(400, dtype=float)
    for idx in range(0, 40, 2):
        ld_R[idx, idx + 1] = 0.95
        ld_R[idx + 1, idx] = 0.95
        exp_df.loc[idx + 1, "beta"] = 0.96 * exp_df.loc[idx, "beta"]
        out_df.loc[idx + 1, "beta"] = 0.96 * out_df.loc[idx, "beta"]

    ld_scores = np.sum(ld_R * ld_R, axis=1)
    res_unpruned = mrlap(exp_df, out_df, ld_scores=ld_scores, ld_R=None, mr_threshold=1e-4)
    res_pruned = mrlap(
        exp_df,
        out_df,
        ld_scores=ld_scores,
        ld_R=ld_R,
        mr_threshold=1e-4,
        mr_pruning_ld=0.001,
    )
    assert res_pruned.n_instruments < res_unpruned.n_instruments


def test_allele_harmonization_flips_swapped_outcome_alleles() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=4000, nx=60_000.0, ny=60_000.0, alpha_true=0.30, seed=71
    )
    exp_a = exp_df.copy()
    out_a = out_df.copy()
    exp_a["ea"] = "A"
    exp_a["nea"] = "G"
    out_a["ea"] = "A"
    out_a["nea"] = "G"

    # Flip half of outcome alleles and negate their betas
    flip = np.arange(len(out_a)) % 2 == 1
    out_a.loc[flip, "ea"] = "G"
    out_a.loc[flip, "nea"] = "A"
    out_a.loc[flip, "beta"] = -out_a.loc[flip, "beta"]

    res_orig = mrlap(exp_df, out_df, ld_scores=ld, seed=71)
    res_harm = mrlap(exp_a, out_a, ld_scores=ld, seed=71)
    assert np.isclose(res_orig.alpha_obs, res_harm.alpha_obs, atol=1e-10)
    assert np.isclose(res_orig.alpha_corrected, res_harm.alpha_corrected, atol=1e-10)


def test_unstandardized_per_allele_scaling_invariance() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=4000, nx=60_000.0, ny=60_000.0, alpha_true=0.28, seed=82
    )
    rng = np.random.default_rng(82)
    maf = rng.uniform(0.05, 0.50, size=4000)
    scale = 1.0 / np.sqrt(2.0 * maf * (1.0 - maf))

    exp_unstd = exp_df.copy()
    out_unstd = out_df.copy()
    exp_unstd["beta"] *= scale
    exp_unstd["se"] *= scale
    out_unstd["beta"] *= scale * 2.5  # also different outcome trait SD!
    out_unstd["se"] *= scale * 2.5

    res1 = mrlap(exp_df, out_df, ld_scores=ld, seed=82)
    res2 = mrlap(exp_unstd, out_unstd, ld_scores=ld, seed=82)
    assert np.isclose(res1.alpha_corrected, res2.alpha_corrected, atol=1e-10)


def test_default_vs_r_default_and_exact_diff_z_formula() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=5000,
        nx=30_000.0,
        ny=30_000.0,
        alpha_true=0.30,
        c_xy=0.25,
        seed=91,
    )
    res_d = mrlap(exp_df, out_df, ld_scores=ld, mr_threshold=1e-5, settings="default")
    res_r = mrlap(exp_df, out_df, ld_scores=ld, mr_threshold=1e-5, settings="R_default")

    assert res_d.n_sim == 500
    assert res_r.n_sim == 1000
    expected_z = (res_r.alpha_obs - res_r.alpha_corrected) / np.sqrt(
        res_r.alpha_obs_se**2 + res_r.alpha_corrected_se**2 - 2.0 * res_r.cov_obs_corr
    )
    assert np.isclose(res_r.diff_stat, expected_z, atol=1e-12)


def test_mrlap_class_interface_and_exports() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(
        m_snps=4000, nx=50_000.0, ny=50_000.0, alpha_true=0.25, seed=105
    )
    est = MRlap(mr_threshold=1e-5, settings="default", seed=105)
    res = est.fit(exp_df, out_df, ld_scores=ld)
    assert est.result_ is res
    df = res.to_dataframe()
    arch_df = res.architecture_dataframe()
    assert len(df) == 1
    assert len(arch_df) == 1
    assert "alpha_corrected" in df.columns
    assert "MRlap Summary" in res.summary()


def test_invalid_settings_raises_value_error() -> None:
    exp_df, out_df, ld = _simulate_mrlap_gwas(m_snps=200, seed=1)
    with pytest.raises(ValueError, match="Unknown settings"):
        mrlap(exp_df, out_df, ld_scores=ld, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        MRlap(settings="wrong")  # type: ignore[arg-type]


def test_validation_errors_on_bad_thresholds_or_inputs() -> None:
    exp_df, out_df, _ = _simulate_mrlap_gwas(
        m_snps=100, nx=200.0, ny=200.0, h2_x=0.05, pi_x=0.5, seed=3
    )
    with pytest.raises(ValueError, match="mr_threshold"):
        mrlap(exp_df, out_df, mr_threshold=1.5)
    with pytest.raises(ValueError, match="at least 2 instruments"):
        mrlap(exp_df, out_df, mr_threshold=1e-15)
    with pytest.raises(ValueError, match="At least 30"):
        mrlap(exp_df.iloc[:15], out_df.iloc[:15])
