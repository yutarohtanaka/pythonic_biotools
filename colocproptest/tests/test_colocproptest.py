"""Comprehensive unit tests for colocPropTest (frequentist proportional colocalization)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from colocproptest import (
    ColocPropTest,
    acat_combine,
    bh_fdr,
    proptest_pc,
    run_coloc_prop_test,
    select_tester_pairs,
    test_pair_proportionality,
)


def _ar1_ld(m: int, rho: float = 0.65) -> np.ndarray:
    idx = np.arange(m)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _simulate_summary_stats(
    beta1_true: np.ndarray,
    beta2_true: np.ndarray,
    se1: np.ndarray,
    se2: np.ndarray,
    ld: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw correlated marginal GWAS summary statistics b_t ~ N(beta_t, D_t R D_t)."""
    chol = np.linalg.cholesky(ld + 1e-12 * np.eye(len(ld)))
    b1 = beta1_true + se1 * (chol @ rng.normal(size=len(ld)))
    b2 = beta2_true + se2 * (chol @ rng.normal(size=len(ld)))
    return b1, b2


def test_settings_validation_all_entrypoints() -> None:
    ld = _ar1_ld(6)
    b1 = np.ones(6) * 0.2
    se = np.full(6, 0.05)

    with pytest.raises(ValueError, match="Unknown settings"):
        select_tester_pairs(b1, se, b1, se, ld, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        test_pair_proportionality(b1[:2], se[:2], b1[:2], se[:2], ld[:2, :2], settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        proptest_pc(b1, se, b1, se, ld, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_coloc_prop_test((b1, se), (b1, se), ld, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        ColocPropTest(settings="bad")  # type: ignore[arg-type]


def test_pair_proportionality_matches_direct_profile_gls() -> None:
    """Verify angular 2x2 solver matches brute-force profile-GLS Q(eta) to machine precision."""
    b1 = np.array([0.42, -0.19])
    se1 = np.array([0.05, 0.08])
    b2 = np.array([-0.31, 0.55])
    se2 = np.array([0.06, 0.04])
    ld = np.array([[1.0, 0.58], [0.58, 1.0]])

    sigma1 = np.outer(se1, se1) * ld
    sigma2 = np.outer(se2, se2) * ld

    def _direct_q(eta: float) -> float:
        diff = b2 - eta * b1
        cov = sigma2 + (eta * eta) * sigma1
        return float(diff @ np.linalg.solve(cov, diff))

    direct_opt = minimize_scalar(_direct_q, bounds=(-20.0, 20.0), method="bounded")
    res = test_pair_proportionality(b1, se1, b2, se2, ld)

    assert np.isclose(res.stat, float(direct_opt.fun), atol=1e-11)
    assert np.isclose(res.eta_hat, float(direct_opt.x), atol=1e-6)
    assert res.chisq == res.stat
    assert res.pvalue == res.p_value


def test_pair_proportionality_exact_trait_swap_symmetry() -> None:
    """Swapping Trait 1 and Trait 2 preserves Q_min and p-value identically, inverting eta_hat."""
    b1 = np.array([0.50, 0.15])
    se1 = np.array([0.04, 0.07])
    b2 = np.array([0.35, -0.22])
    se2 = np.array([0.05, 0.06])
    ld = np.array([[1.0, -0.45], [-0.45, 1.0]])

    res_12 = test_pair_proportionality(b1, se1, b2, se2, ld)
    res_21 = test_pair_proportionality(b2, se2, b1, se1, ld)

    assert np.isclose(res_12.stat, res_21.stat, atol=1e-12)
    assert np.isclose(res_12.p_value, res_21.p_value, atol=1e-12)
    assert np.isclose(res_12.eta_hat * res_21.eta_hat, 1.0, atol=1e-7)


def test_zero_exposure_infinite_slope_no_division_by_zero() -> None:
    """When Trait 1 is identically zero and Trait 2 is non-zero, theta=pi/2 gives Q=0 without NaN."""
    b1 = np.array([0.0, 0.0])
    se1 = np.array([0.05, 0.05])
    b2 = np.array([4.0, 2.0])
    se2 = np.array([0.05, 0.05])
    ld = np.array([[1.0, 0.5], [0.5, 1.0]])

    res = test_pair_proportionality(b1, se1, b2, se2, ld)
    assert np.isfinite(res.stat)
    assert np.isclose(res.stat, 0.0, atol=1e-12)
    assert np.isclose(res.p_value, 1.0, atol=1e-12)
    assert np.isclose(res.theta_hat, 0.5 * np.pi, atol=1e-7)
    assert np.isinf(res.eta_hat)


def test_pair_null_chi2_type1_error_calibration() -> None:
    """Monte Carlo verification of exact chi^2_1 null calibration under correlated errors."""
    rng = np.random.default_rng(2024)
    ld = np.array([[1.0, 0.65], [0.65, 1.0]])
    se1 = np.array([0.04, 0.06])
    se2 = np.array([0.05, 0.03])
    beta1_true = np.array([0.45, 0.45 * 0.65])
    beta2_true = 1.35 * beta1_true

    stats = []
    pvals = []
    for _ in range(300):
        b1, b2 = _simulate_summary_stats(beta1_true, beta2_true, se1, se2, ld, rng)
        r = test_pair_proportionality(b1, se1, b2, se2, ld)
        stats.append(r.stat)
        pvals.append(r.p_value)

    p_arr = np.asarray(pvals)
    s_arr = np.asarray(stats)
    type1_rate = float(np.mean(p_arr < 0.05))
    assert 0.015 <= type1_rate <= 0.095
    # Theoretical median of chi^2_1 is 0.4549
    assert 0.30 <= float(np.median(s_arr)) <= 0.65


def test_distinct_causal_variants_strongly_rejected() -> None:
    """Two distinct causal variants (H3) violate proportionality and are strongly rejected."""
    rng = np.random.default_rng(17)
    m = 30
    ld = _ar1_ld(m, 0.60)
    se1 = np.full(m, 0.04)
    se2 = np.full(m, 0.04)
    # Trait 1 causal at SNP 6; Trait 2 causal at SNP 20
    beta1 = 0.45 * ld[:, 6]
    beta2 = 0.45 * ld[:, 20]
    b1, b2 = _simulate_summary_stats(beta1, beta2, se1, se2, ld, rng)

    res = run_coloc_prop_test(
        {"beta": b1, "se": se1},
        {"beta": b2, "se": se2},
        ld,
        alpha=0.05,
        settings="default",
    )
    assert res.reject_proportionality is True
    assert res.evidence_for_coloc is False
    assert res.min_fdr < 1e-4
    assert res.p_acat < 1e-4
    assert res.p_pc_omnibus < 1e-4


def test_robustness_to_differential_per_snp_sample_sizes() -> None:
    """Variable per-SNP sample sizes (Wallace, Robins & Johnson 2025) do NOT cause false rejection."""
    rng = np.random.default_rng(31)
    m = 30
    ld = _ar1_ld(m, 0.65)
    # Odd SNPs missing from 80% of meta-analysis cohorts -> sqrt(5)x larger SE in Trait 1!
    se1 = np.where(np.arange(m) % 2 == 0, 0.03, 0.03 * np.sqrt(5.0))
    se2 = np.full(m, 0.035)

    # Single shared causal variant at SNP 12 with true proportionality slope eta = 0.75
    beta1 = 0.40 * ld[:, 12]
    beta2 = 0.75 * beta1
    b1, b2 = _simulate_summary_stats(beta1, beta2, se1, se2, ld, rng)

    res = run_coloc_prop_test((b1, se1), (b2, se2), ld, alpha=0.05, settings="default")
    assert res.reject_proportionality is False
    assert res.evidence_for_coloc is True
    assert res.eta_ci[0] <= 0.75 <= res.eta_ci[1]
    assert np.isclose(res.eta_hat, 0.75, atol=0.15)


def test_multiple_shared_causal_variants_negative_slope() -> None:
    """Two shared causal variants with proportional negative slope eta = -1.4 retain H0."""
    rng = np.random.default_rng(44)
    m = 32
    ld = _ar1_ld(m, 0.55)
    se1 = np.full(m, 0.03)
    se2 = np.full(m, 0.04)

    beta1 = 0.35 * ld[:, 7] + 0.28 * ld[:, 22]
    beta2 = -1.40 * beta1
    b1, b2 = _simulate_summary_stats(beta1, beta2, se1, se2, ld, rng)

    res = run_coloc_prop_test((b1, se1), (b2, se2), ld, settings="default")
    assert res.reject_proportionality is False
    assert res.evidence_for_coloc is True
    assert res.eta_ci[0] <= -1.40 <= res.eta_ci[1]


def test_out_of_sample_ld_reference_robustness() -> None:
    """Moderate out-of-sample LD mismatch under shared causal variant retains colocalization."""
    rng = np.random.default_rng(58)
    m = 25
    ld_true = _ar1_ld(m, 0.65)
    ld_ref = _ar1_ld(m, 0.58)  # Out-of-sample reference LD panel

    se1 = np.full(m, 0.04)
    se2 = np.full(m, 0.04)
    beta1 = 0.36 * ld_true[:, 10]
    beta2 = 0.90 * beta1
    b1, b2 = _simulate_summary_stats(beta1, beta2, se1, se2, ld_true, rng)

    res = run_coloc_prop_test((b1, se1), (b2, se2), ld_ref, settings="default")
    assert res.reject_proportionality is False
    assert res.evidence_for_coloc is True


def test_single_trait_associated_null_second_trait_not_called_colocalized() -> None:
    """When Trait 2 is null (eta=0), proportionality is NOT rejected, but evidence_for_coloc is False."""
    rng = np.random.default_rng(72)
    m = 24
    ld = _ar1_ld(m, 0.6)
    se1 = np.full(m, 0.04)
    se2 = np.full(m, 0.04)
    beta1 = 0.40 * ld[:, 10]
    beta2 = np.zeros(m)
    b1, b2 = _simulate_summary_stats(beta1, beta2, se1, se2, ld, rng)

    res = run_coloc_prop_test((b1, se1), (b2, se2), ld, settings="default")
    assert res.reject_proportionality is False
    assert res.evidence_for_coloc is False
    assert res.eta_ci[0] <= 0.0 <= res.eta_ci[1]


def test_both_traits_null_fieller_unbounded_ci() -> None:
    """When Trait 1 has no signal, Fieller / profile-GLS 95% CI is (-inf, +inf)."""
    ld = _ar1_ld(12, 0.5)
    se = np.full(12, 0.05)
    b1 = np.zeros(12)
    b2 = np.zeros(12)
    pc_res = proptest_pc(b1, se, b2, se, ld, var_explained=0.90)
    assert pc_res.eta_ci == (-np.inf, np.inf)
    assert np.isclose(pc_res.p_value, 1.0)


def test_select_tester_pairs_ld_pruning_and_caps() -> None:
    m = 20
    ld = _ar1_ld(m, 0.95)  # High collinearity between adjacent variants
    se = np.full(m, 0.04)
    b1 = 0.35 * ld[:, 10]
    b2 = 0.35 * ld[:, 10]

    pairs = select_tester_pairs(b1, se, b2, se, ld, max_testers=6, max_r2=0.70, min_abs_z=2.0)
    assert len(pairs) >= 1
    for i, j in pairs:
        assert i < j
        assert ld[i, j] ** 2 <= 0.70 + 1e-12


def test_r_default_parity_varbeta_dicts_and_pc_threshold() -> None:
    m = 25
    ld = _ar1_ld(m, 0.6)
    se1 = np.full(m, 0.04)
    se2 = np.full(m, 0.05)
    b1 = 0.38 * ld[:, 8]
    b2 = 0.50 * b1
    snps = [f"rs{i}" for i in range(m)]

    df1 = pd.DataFrame({"snp": snps, "beta": b1, "varbeta": se1**2})
    d2 = {"snp": snps, "beta": b2, "varbeta": se2**2}

    res_def = run_coloc_prop_test(df1, d2, pd.DataFrame(ld), settings="default")
    est_r = ColocPropTest(alpha=0.05, settings="R_default")
    res_r = est_r.fit(df1, d2, pd.DataFrame(ld))

    # R_default uses thr=0.80 cumulative variance vs 0.95 in default -> fewer PCs
    assert res_r.n_pc < res_def.n_pc
    assert res_r.evidence_for_coloc is True
    assert "ColocPropTestResult" in res_r.summary()
    out_df = res_r.to_dataframe()
    assert {"snp_1", "snp_2", "stat", "p_value", "fdr", "eta_hat"}.issubset(set(out_df.columns))


def test_bh_fdr_and_acat_edge_cases() -> None:
    p = np.array([0.001, 0.02, 0.04, 0.50])
    q = bh_fdr(p)
    # Manual R p.adjust([0.001, 0.02, 0.04, 0.50], method="BH") = [0.004, 0.04, 0.05333333, 0.50]
    assert np.allclose(q, [0.004, 0.04, 0.04 * 4.0 / 3.0, 0.50], atol=1e-12)
    assert acat_combine(np.array([1e-25, 0.5, 0.8])) < 1e-20
    assert acat_combine(np.zeros(0)) == 1.0
