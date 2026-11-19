"""Comprehensive unit tests for `hyprcoloc` (Foley et al., Nature Communications 2021)."""

from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from scipy import special

from hyprcoloc import hyprcoloc, log_abf, sensitivity_analysis, subset_coloc


def _reference_coloc_abf_2trait(
    b1: np.ndarray,
    s1: np.ndarray,
    b2: np.ndarray,
    s2: np.ndarray,
    w1: float = 0.15,
    w2: float = 0.15,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
) -> dict[str, float | np.ndarray]:
    """Exact Wakefield/Giambartolomei 2-trait `coloc.abf` reference."""
    v1, v2 = s1**2, s2**2
    r1 = w1**2 / (v1 + w1**2)
    r2 = w2**2 / (v2 + w2**2)
    lbf1 = 0.5 * np.log(1.0 - r1) + 0.5 * r1 * (b1**2 / v1)
    lbf2 = 0.5 * np.log(1.0 - r2) + 0.5 * r2 * (b2**2 / v2)

    lh0 = 0.0
    lh1 = np.log(p1) + float(special.logsumexp(lbf1))
    lh2 = np.log(p2) + float(special.logsumexp(lbf2))
    lbf_sh = lbf1 + lbf2
    sum_sh = float(special.logsumexp(lbf_sh))
    lh4 = np.log(p12) + sum_sh
    log_prod = float(special.logsumexp(lbf1) + special.logsumexp(lbf2))
    lh3 = np.log(p1) + np.log(p2) + sum_sh + np.log(max(np.exp(log_prod - sum_sh) - 1.0, 1e-300))

    hyps = np.array([lh0, lh1, lh2, lh3, lh4])
    post = np.exp(hyps - special.logsumexp(hyps))
    snp_pp = np.exp(lbf_sh - sum_sh)
    return {
        "PP.H0": float(post[0]),
        "PP.H1": float(post[1]),
        "PP.H2": float(post[2]),
        "PP.H3": float(post[3]),
        "PP.H4": float(post[4]),
        "snp_pp": snp_pp,
    }


def test_exact_2trait_coloc_abf_equivalence():
    rng = np.random.default_rng(42)
    m = 120
    ses = rng.uniform(0.03, 0.08, size=(m, 2))

    # Test across shared-causal (H4), distinct-causal (H3), single-trait (H1), and null (H0) regimes
    for scenario in ("H4", "H3", "H1", "H0"):
        betas = rng.normal(0.0, ses * 0.6)
        if scenario == "H4":
            betas[35, 0] += 0.45
            betas[35, 1] -= 0.50
        elif scenario == "H3":
            betas[20, 0] += 0.50
            betas[85, 1] += 0.55
        elif scenario == "H1":
            betas[50, 0] += 0.50

        p1 = 1e-4
        p12 = 1e-5
        pc = p12 / p1  # 0.10 maps pi_2 = p1 * pc = p12 identically
        ref = _reference_coloc_abf_2trait(
            betas[:, 0], ses[:, 0], betas[:, 1], ses[:, 1], w1=0.15, w2=0.20, p1=p1, p2=p1, p12=p12
        )
        sub = subset_coloc(
            betas,
            ses,
            binary_outcomes=[False, True],
            prior_1=p1,
            prior_c=pc,
        )
        expected_pr = ref["PP.H3"] + ref["PP.H4"]
        expected_pa = ref["PP.H4"] / expected_pr

        assert np.isclose(sub.regional_prob, expected_pr, rtol=1e-12, atol=1e-14)
        assert np.isclose(sub.align_prob, expected_pa, rtol=1e-12, atol=1e-14)
        assert np.isclose(sub.posterior_prob, ref["PP.H4"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(sub.snp_posteriors, ref["snp_pp"], rtol=1e-12, atol=1e-14)


def test_exact_newton_symmetric_sum_vs_bruteforce_5traits():
    """Verify O(M m) Newton identity matches explicit O(M m^2) enumeration of all m-choose-2 subsets."""
    rng = np.random.default_rng(123)
    m_snps, q_traits = 60, 5
    ses = rng.uniform(0.04, 0.09, size=(m_snps, q_traits))
    betas = rng.normal(0.0, ses)
    betas[15, :4] += 0.35

    lbf = log_abf(betas, ses)
    p1, pc = 1e-4, 0.02
    gamma = 1.0 - pc
    log_pi = [0.0, np.log(p1)]
    for k in range(2, q_traits + 1):
        log_pi.append(log_pi[-1] + np.log(1.0 - gamma ** (k - 1)))

    s_lbf = np.sum(lbf, axis=1)
    log_w_all = float(special.logsumexp(s_lbf))
    log_l_all = log_pi[q_traits] + log_w_all

    # Brute-force m-1 and m-2 terms
    bf_m1 = [ float(special.logsumexp(np.sum(lbf[:, list(comb)], axis=1))) for comb in combinations(range(q_traits), q_traits - 1) ]
    bf_m2 = [ float(special.logsumexp(np.sum(lbf[:, list(comb)], axis=1))) for comb in combinations(range(q_traits), q_traits - 2) ]
    bf_d = []
    for q in range(q_traits):
        rem = [r for r in range(q_traits) if r != q]
        w_rem = float(special.logsumexp(np.sum(lbf[:, rem], axis=1)))
        w_q = float(special.logsumexp(lbf[:, q]))
        bf_d.append(log_w_all + np.log(np.exp(w_rem + w_q - log_w_all) - 1.0))

    log_l_align_alt = log_pi[q_traits - 1] + log_pi[1] + float(special.logsumexp(bf_d))
    log_reg_num = float(np.logaddexp(log_l_all, log_l_align_alt))
    log_non_reg = float(
        special.logsumexp(
            [0.0, log_pi[q_traits - 1] + float(special.logsumexp(bf_m1)), log_pi[q_traits - 2] + float(special.logsumexp(bf_m2))]
        )
    )
    pr_bf = float(np.exp(log_reg_num - np.logaddexp(log_reg_num, log_non_reg)))
    pa_bf = float(np.exp(log_l_all - log_reg_num))

    sub = subset_coloc(betas, ses, prior_1=p1, prior_c=pc)
    assert np.isclose(sub.regional_prob, pr_bf, rtol=1e-13, atol=1e-15)
    assert np.isclose(sub.align_prob, pa_bf, rtol=1e-13, atol=1e-15)
    assert np.isclose(sub.posterior_prob, pr_bf * pa_bf, rtol=1e-13, atol=1e-15)


def test_extreme_zscore_numerical_stability():
    """Verify zero overflow/NaN when z = 65 (z^2 = 4225, lABF > 1500 >> 709 float64 exp limit)."""
    m, q = 100, 4
    ses = np.full((m, q), 0.02)
    betas = np.zeros((m, q))
    betas[42, :] = 65.0 * 0.02  # z = 65 across all 4 traits

    res = hyprcoloc(betas, ses, snp_ids=[f"rs{i}" for i in range(m)])
    assert len(res.clusters) == 1
    cl = res.clusters[0]
    assert cl.candidate_snp == "rs42"
    assert np.isclose(cl.posterior_prob, 1.0, atol=1e-6)
    assert np.isclose(cl.posterior_explained_by_snp, 1.0, atol=1e-10)
    assert cl.credible_set_95 == ["rs42"]


def test_multi_cluster_branch_and_bound_partitioning():
    rng = np.random.default_rng(2025)
    m, q = 200, 6
    ses = np.full((m, q), 0.05)
    betas = rng.normal(0.0, ses * 0.5)
    snp_ids = [f"rs{i + 1}" for i in range(m)]
    traits = ["CAD", "LDL", "APOB", "SBP", "DBP", "Height_null"]

    # Cluster 1: CAD, LDL, APOB share causal variant rs31 (index 30)
    betas[30, [0, 1, 2]] += np.array([0.42, 0.48, -0.45])
    # Cluster 2: SBP, DBP share distinct causal variant rs151 (index 150)
    betas[150, [3, 4]] += np.array([0.50, 0.46])

    for mode in ("regional", "align"):
        res = hyprcoloc(betas, ses, trait_names=traits, snp_ids=snp_ids, bb_selection=mode)
        found_sets = {frozenset(c.traits): c for c in res.clusters}
        assert frozenset(["CAD", "LDL", "APOB"]) in found_sets
        assert frozenset(["SBP", "DBP"]) in found_sets
        assert res.uncolocalized_traits == ["Height_null"]

        c_lipid = found_sets[frozenset(["CAD", "LDL", "APOB"])]
        c_bp = found_sets[frozenset(["SBP", "DBP"])]
        assert c_lipid.candidate_snp == "rs31"
        assert c_lipid.posterior_prob > 0.85
        assert "rs31" in c_lipid.credible_set_95
        assert c_bp.candidate_snp == "rs151"
        assert c_bp.posterior_prob > 0.85


def test_distinct_variants_rejected_by_alignment_filter():
    rng = np.random.default_rng(77)
    m, q = 150, 3
    ses = np.full((m, q), 0.04)
    betas = rng.normal(0.0, ses * 0.4)
    # Each trait has a strong association at a DIFFERENT variant
    betas[10, 0] += 0.45
    betas[60, 1] += 0.45
    betas[110, 2] += 0.45

    res = hyprcoloc(betas, ses)
    assert res.clusters == []
    assert len(res.uncolocalized_traits) == 3

    # Evaluating without branch-and-bound (`bb_alg=False`) shows high P_R but near-zero P_A
    res_no_bb = hyprcoloc(betas, ses, bb_alg=False)
    assert res_no_bb.clusters[0].regional_prob > 0.95
    assert res_no_bb.clusters[0].align_prob < 1e-4

    # `reg_only` groups regionally associated traits without requiring single-variant alignment
    res_reg_only = hyprcoloc(betas, ses, bb_selection="reg_only")
    assert len(res_reg_only.clusters) == 1
    assert len(res_reg_only.clusters[0].traits) == 3


def test_null_calibration_zero_false_clusters():
    rng = np.random.default_rng(314)
    false_clusters = 0
    max_pp = 0.0
    for _ in range(40):
        ses = rng.uniform(0.04, 0.08, size=(120, 4))
        betas = rng.normal(0.0, ses)
        res = hyprcoloc(betas, ses)
        false_clusters += len(res.clusters)
        sub = subset_coloc(betas, ses)
        max_pp = max(max_pp, sub.posterior_prob)

    assert false_clusters <= 1
    assert max_pp < 0.10


def test_allele_coding_invariance_with_and_without_sample_overlap():
    rng = np.random.default_rng(909)
    m, q = 80, 3
    ses = rng.uniform(0.03, 0.06, size=(m, q))
    betas = rng.normal(0.0, ses * 0.5)
    betas[25, :] += np.array([0.38, -0.35, 0.40])

    flips = rng.choice([-1.0, 1.0], size=(m, 1))
    betas_flipped = betas * flips

    cor = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    for tc in (None, cor):
        r1 = hyprcoloc(betas, ses, trait_cor=tc)
        r2 = hyprcoloc(betas_flipped, ses, trait_cor=tc)
        assert len(r1.clusters) == len(r2.clusters) == 1
        assert np.isclose(r1.clusters[0].regional_prob, r2.clusters[0].regional_prob, rtol=1e-13)
        assert np.isclose(r1.clusters[0].align_prob, r2.clusters[0].align_prob, rtol=1e-13)
        assert np.isclose(r1.clusters[0].posterior_prob, r2.clusters[0].posterior_prob, rtol=1e-13)
        np.testing.assert_allclose(r1.clusters[0].snp_posteriors, r2.clusters[0].snp_posteriors, rtol=1e-13)


def test_sample_overlap_discounts_shared_noise_and_sensitivity_helper():
    rng = np.random.default_rng(515)
    m = 100
    ses = np.full((m, 2), 0.05)
    betas = rng.normal(0.0, 0.01, size=(m, 2))
    # Moderate concordant bump that mimics shared-sample correlation rho = 0.85
    betas[12, :] = 0.20

    # Identity trait_cor matches None identically
    sub_none = subset_coloc(betas, ses)
    sub_eye = subset_coloc(betas, ses, trait_cor=np.eye(2))
    assert np.isclose(sub_none.posterior_prob, sub_eye.posterior_prob, rtol=1e-13)

    # Strong positive error correlation rho = 0.85 discounts the shared concordant fluctuation
    cor_high = np.array([[1.0, 0.85], [0.85, 1.0]])
    sub_overlap = subset_coloc(betas, ses, trait_cor=cor_high)
    assert sub_overlap.posterior_prob < sub_none.posterior_prob

    # Sensitivity grid helper and DataFrame / summary serialization
    betas_df = pd.DataFrame(betas, index=[f"rs{i}" for i in range(m)], columns=["TraitA", "TraitB"])
    ses_df = pd.DataFrame(ses, index=betas_df.index, columns=betas_df.columns)
    betas_df.iloc[12, :] = 0.40

    sens = sensitivity_analysis(
        betas_df,
        ses_df,
        prior_c_grid=(0.05, 0.02),
        reg_thresh_grid=(0.5, 0.7),
        align_thresh_grid=(0.5, 0.7),
        ld_matrix=np.eye(m),
    )
    assert sens.n_configurations == 8
    sim_df = sens.to_dataframe()
    assert sim_df.shape == (2, 2)
    assert sim_df.loc["TraitA", "TraitB"] > 0.8
    assert "Sensitivity Analysis" in sens.summary()

    fit = hyprcoloc(betas_df, ses_df, prior_2=0.98)
    out_df = fit.to_dataframe()
    assert out_df.loc[0, "candidate_snp"] == "rs12"
    assert "rs12" in fit.credible_set(1, 0.95)
    assert "HyPrColoc" in fit.summary()


def test_invalid_inputs_raise_value_error():
    betas = np.zeros((10, 3))
    ses = np.full((10, 3), 0.05)

    # NaN / Inf
    b_nan = betas.copy()
    b_nan[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        hyprcoloc(b_nan, ses)

    # Non-positive SE
    s_bad = ses.copy()
    s_bad[2, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        hyprcoloc(betas, s_bad)

    # Fewer than 2 traits
    with pytest.raises(ValueError, match="Q >= 2"):
        hyprcoloc(betas[:, :1], ses[:, :1])

    # Non-PD trait_cor
    bad_cor = np.ones((3, 3))
    with pytest.raises(ValueError, match="positive definite"):
        hyprcoloc(betas, ses, trait_cor=bad_cor)

    # Invalid priors / thresholds
    with pytest.raises(ValueError, match="prior_1"):
        hyprcoloc(betas, ses, prior_1=0.0)
    with pytest.raises(ValueError, match="reg_thresh"):
        hyprcoloc(betas, ses, reg_thresh=1.0)
