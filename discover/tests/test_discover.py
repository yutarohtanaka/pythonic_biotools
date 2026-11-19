from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from discover import (
    bh_fdr,
    discover_fdr,
    fit_discover_background,
    groupwise_discover_test,
    pairwise_discover_test,
    poisson_binomial_pmf,
    poisson_binomial_pval,
)


def test_background_marginal_constraints_and_boundary_rows():
    """Expected row and column sums match observed counts to <1e-9, including 0/full rows/cols."""
    rng = np.random.default_rng(42)
    gene_logit = rng.normal(-1.5, 0.8, size=25)
    tumor_logit = rng.normal(0.0, 1.4, size=80)
    probs_true = 1.0 / (1.0 + np.exp(-(gene_logit[:, None] + tumor_logit[None, :])))
    events = rng.binomial(1, probs_true).astype(float)

    # Inject boundary cases: unmutated gene, ubiquitous gene, unmutated tumor, fully mutated tumor
    events[0, :] = 0.0
    events[1, :] = 1.0
    events[:, 0] = 0.0
    events[1, 0] = 1.0  # keep consistent

    df = pd.DataFrame(
        events,
        index=[f"GENE_{i}" for i in range(25)],
        columns=[f"TUMOR_{j}" for j in range(80)],
    )
    bg = fit_discover_background(df)

    assert bg.converged
    assert bg.max_marginal_error < 1e-8
    np.testing.assert_allclose(bg.bg_probs.sum(axis=1), events.sum(axis=1), atol=1e-8)
    np.testing.assert_allclose(bg.bg_probs.sum(axis=0), events.sum(axis=0), atol=1e-8)
    np.testing.assert_allclose(bg.bg_probs[0, :], 0.0, atol=1e-12)
    assert bg.to_dataframe().shape == (25, 80)


def test_background_nan_masked_panel_entries():
    """Supports NaN unprofiled panel entries (e.g., AACR GENIE multi-panel cohorts)."""
    rng = np.random.default_rng(101)
    events = rng.binomial(1, 0.25, size=(18, 60)).astype(float)
    # Genes 10..17 are unprofiled in Panel A (tumors 0..24)
    events[10:, :25] = np.nan
    # Random sporadic missing entries
    events[2, 40:45] = np.nan

    bg = fit_discover_background(events)
    assert bg.converged
    assert not bg.fully_observed
    assert np.array_equal(np.isnan(bg.bg_probs), np.isnan(events))

    obs_row = np.nansum(events, axis=1)
    exp_row = np.nansum(bg.bg_probs, axis=1)
    obs_col = np.nansum(events, axis=0)
    exp_col = np.nansum(bg.bg_probs, axis=0)
    np.testing.assert_allclose(exp_row, obs_row, atol=1e-8)
    np.testing.assert_allclose(exp_col, obs_col, atol=1e-8)


def test_stratified_background_fitting():
    """Stratified fitting enforces row and column marginals independently per subtype/MSI stratum."""
    rng = np.random.default_rng(7)
    mss = rng.binomial(1, 0.08, size=(20, 50)).astype(float)
    msi = rng.binomial(1, 0.45, size=(20, 30)).astype(float)
    events = np.hstack([mss, msi])
    strata = np.array(["MSS"] * 50 + ["MSI"] * 30)

    bg = fit_discover_background(events, strata=strata)
    assert bg.converged
    assert bg.mu.shape == (20, 2)
    np.testing.assert_allclose(bg.bg_probs[:, :50].sum(axis=1), mss.sum(axis=1), atol=1e-8)
    np.testing.assert_allclose(bg.bg_probs[:, 50:].sum(axis=1), msi.sum(axis=1), atol=1e-8)
    np.testing.assert_allclose(bg.bg_probs.sum(axis=0), events.sum(axis=0), atol=1e-8)


def test_poisson_binomial_dft_vs_dp_and_binomial_ground_truth():
    """Hong (2013) log-polar DFT matches exact DP to <1e-13 and reduces to Binomial for equal p."""
    # Homogeneous p -> exact Binomial(n=100, p=0.17)
    p_hom = np.full(100, 0.17)
    pmf_dft = poisson_binomial_pmf(p_hom, method="dft")
    pmf_dp = poisson_binomial_pmf(p_hom, method="dp")
    pmf_binom = stats.binom.pmf(np.arange(101), 100, 0.17)

    np.testing.assert_allclose(pmf_dft, pmf_binom, atol=1e-13)
    np.testing.assert_allclose(pmf_dp, pmf_binom, atol=1e-14)

    # Heterogeneous p across 120 trials including deterministic 0s, 0.5, and 1s
    rng = np.random.default_rng(123)
    p_het = np.concatenate([[0.0, 1.0, 0.5, 0.5], rng.uniform(0.001, 0.85, size=116)])
    pmf_het_dft = poisson_binomial_pmf(p_het, method="dft")
    pmf_het_dp = poisson_binomial_pmf(p_het, method="dp")
    np.testing.assert_allclose(pmf_het_dft, pmf_het_dp, atol=1e-13)
    assert pmf_het_dft[0] == 0.0  # one deterministic 1.0 shifts minimum support to >= 1

    # Verify Lancaster mid-p identity: p_less(mid_p=True) + p_greater(mid_p=True) == 1.0
    for k in [1, 20, 45, 80]:
        p_lo = poisson_binomial_pval(k, p_het, alternative="less", mid_p=True)
        p_hi = poisson_binomial_pval(k, p_het, alternative="greater", mid_p=True)
        assert np.isclose(p_lo + p_hi, 1.0, atol=1e-12)
        p_lo_exact = poisson_binomial_pval(k, p_het, alternative="less", mid_p=False)
        assert p_lo <= p_lo_exact + 1e-15


def test_hypermutator_confounding_fisher_false_positive_vs_discover_null():
    """
    Core Canisius et al. (2016) motivation: independent genes mutated heavily in
    hypermutators trigger extreme false-positive co-occurrence under Fisher's exact
    test, whereas DISCOVER properly stays null!
    """
    rng = np.random.default_rng(2026)
    n_genes = 40
    # 160 low-TMB tumors (~3% mutation rate) + 40 hypermutator tumors (~65% mutation rate)
    p_low = np.full((n_genes, 160), 0.03)
    p_high = np.full((n_genes, 40), 0.65)
    probs = np.hstack([p_low, p_high])
    # All genes are generated COMPLETELY INDEPENDENTLY given tumor TMB!
    events = rng.binomial(1, probs).astype(float)

    bg = fit_discover_background(events)
    # Test pair (0, 1) with Fisher's exact test vs DISCOVER co-occurrence test
    g0, g1 = events[0].astype(int), events[1].astype(int)
    n11 = int(np.sum((g0 == 1) & (g1 == 1)))
    n10 = int(np.sum((g0 == 1) & (g1 == 0)))
    n01 = int(np.sum((g0 == 0) & (g1 == 1)))
    n00 = int(np.sum((g0 == 0) & (g1 == 0)))
    _, fisher_p = stats.fisher_exact([[n11, n10], [n01, n00]], alternative="greater")

    res = pairwise_discover_test(bg, g1=[0], g2=[1], alternative="co-occurrence")
    discover_p = float(res.p_value[0])

    # Fisher's exact test falsely screams co-occurrence (p < 1e-4), while DISCOVER stays null!
    assert fisher_p < 1e-4
    assert discover_p > 0.05

    # Across all 780 independent gene pairs, DISCOVER yields near-zero false discoveries at q <= 0.05
    all_co = pairwise_discover_test(bg, alternative="co-occurrence")
    fdr_rate = float(np.mean(all_co.q_value <= 0.05))
    assert fdr_rate < 0.02


def test_true_mutual_exclusivity_and_co_occurrence_detection():
    """Detects planted pairwise mutual exclusivity (BRAF/NRAS) and co-occurrence."""
    rng = np.random.default_rng(99)
    n_tumors = 180
    # 30 background genes so tumor TMB is well-estimated
    bg_genes = rng.binomial(1, 0.25, size=(30, n_tumors)).astype(float)

    # Plant strictly mutually exclusive pair G_MEX1 (60 muts) and G_MEX2 (60 muts) -> 0 overlap
    mex1 = np.zeros(n_tumors)
    mex2 = np.zeros(n_tumors)
    mex1[:60] = 1.0
    mex2[60:120] = 1.0

    # Plant strongly co-occurring pair G_CO1 and G_CO2 (both mutated in same 45 tumors)
    co1 = np.zeros(n_tumors)
    co2 = np.zeros(n_tumors)
    co1[120:165] = 1.0
    co2[120:165] = 1.0

    mat = np.vstack([mex1, mex2, co1, co2, bg_genes])
    names = ["BRAF", "NRAS", "CO_A", "CO_B"] + [f"BG_{i}" for i in range(30)]
    df = pd.DataFrame(mat, index=names)

    bg = fit_discover_background(df)
    sub = bg.subset(genes=["BRAF", "NRAS", "CO_A", "CO_B"])

    res_ex = pairwise_discover_test(sub, alternative="exclusivity")
    df_ex = res_ex.to_dataframe().set_index(["gene1", "gene2"])
    assert df_ex.loc[("BRAF", "NRAS"), "obs_co_mutations"] == 0
    assert df_ex.loc[("BRAF", "NRAS"), "exp_co_mutations"] > 10.0
    assert df_ex.loc[("BRAF", "NRAS"), "q_value"] < 1e-6

    res_co = pairwise_discover_test(sub, alternative="co-occurrence")
    df_co = res_co.to_dataframe().set_index(["gene1", "gene2"])
    assert df_co.loc[("CO_A", "CO_B"), "obs_co_mutations"] == 45
    assert df_co.loc[("CO_A", "CO_B"), "q_value"] < 1e-10


def test_discover_discrete_fdr_vs_classical_bh_under_sparse_genes():
    """
    Adding ultra-sparse uninformative gene pairs (p_min ~ 0.49) inflates classical BH,
    whereas Canisius/Carlson discrete `discover_fdr` leaves truly significant q-values intact!
    """
    p_signal = np.array([1e-4, 5e-4])
    pmin_signal = np.array([1e-6, 1e-5])

    # 500 ultra-rare pairs that each have minimum attainable p-value 0.48
    p_sparse = np.full(500, 0.48)
    pmin_sparse = np.full(500, 0.48)

    p_all = np.concatenate([p_signal, p_sparse])
    pmin_all = np.concatenate([pmin_signal, pmin_sparse])

    q_signal_only = discover_fdr(p_signal, pmin_signal)
    q_disc_all = discover_fdr(p_all, pmin_all)
    q_bh_all = bh_fdr(p_all)

    # Discrete FDR ignores the 500 unattainable tests at tau <= 5e-4!
    np.testing.assert_allclose(q_disc_all[:2], q_signal_only, rtol=1e-12)
    assert np.all(q_disc_all[:2] < 1e-3)
    # Classical BH is >250x more conservative here!
    assert q_bh_all[0] > 0.02


def test_groupwise_impurity_coverage_exclusivity_and_2gene_equivalence():
    """
    Tests all three groupwise statistics ('impurity', 'coverage', 'exclusivity')
    and verifies that 2-gene 'impurity' identically matches pairwise DISCOVER!
    """
    rng = np.random.default_rng(314)
    n_tumors = 150
    bg_genes = rng.binomial(1, 0.25, size=(25, n_tumors)).astype(float)

    # 3-gene mutually exclusive pathway (mutated in disjoint blocks of 40 tumors each)
    p1 = np.zeros(n_tumors)
    p2 = np.zeros(n_tumors)
    p3 = np.zeros(n_tumors)
    p1[0:40] = 1.0
    p2[40:80] = 1.0
    p3[80:120] = 1.0

    mat = np.vstack([p1, p2, p3, bg_genes])
    names = ["TP53", "MDM2", "CDKN2A"] + [f"BG_{i}" for i in range(25)]
    bg = fit_discover_background(pd.DataFrame(mat, index=names))

    for method in ("impurity", "coverage", "exclusivity"):
        gw = groupwise_discover_test(
            bg,
            gene_sets={"p53_pathway": ["TP53", "MDM2", "CDKN2A"]},
            method=method,
            direction="exclusivity",
        )
        df_gw = gw.to_dataframe()
        assert df_gw.shape[0] == 1
        assert float(df_gw["p_value"].iloc[0]) < 1e-6

    # Exact mathematical identity: 2-gene groupwise "impurity" == pairwise DISCOVER test!
    pw_2 = pairwise_discover_test(bg, g1=["TP53"], g2=["MDM2"], alternative="exclusivity")
    gw_2 = groupwise_discover_test(bg, genes=["TP53", "MDM2"], method="impurity", direction="exclusivity")
    np.testing.assert_allclose(gw_2.exp_stat, pw_2.exp_co_mutations, rtol=1e-12)
    np.testing.assert_allclose(gw_2.p_value, pw_2.p_value, rtol=1e-12)


def test_input_validation_errors():
    """Rigorous validation rejects non-binary entries, invalid probabilities, and mismatched strata."""
    with pytest.raises(ValueError, match="binary"):
        fit_discover_background(np.array([[0.0, 2.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="strata"):
        fit_discover_background(np.array([[0.0, 1.0], [1.0, 0.0]]), strata=["A"])

    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        poisson_binomial_pmf([0.2, -0.1])

    with pytest.raises(ValueError, match="at least 2"):
        bg = fit_discover_background(np.array([[1.0, 0.0], [0.0, 1.0]]))
        groupwise_discover_test(bg, genes=["G0"])

    bg_ok = fit_discover_background(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="settings"):
        pairwise_discover_test(bg_ok, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        groupwise_discover_test(bg_ok, genes=["G0", "G1"], settings="bad")  # type: ignore[arg-type]


def test_r_default_inclusive_p_and_exact_dbh_fdr():
    """
    Verify NKI-CCB/DISCOVER (fdr.f90, grouptest.py) parity under settings='R_default':
    1. Forces mid_p=False (inclusive tails P(X <= x) / P(X >= x)).
    2. Computes exact DBH discrete Benjamini-Hochberg FDR:
       pi0 * sum_r F_r(p_obs) / rank with pi0 = min(1.0, sum(p) / sum(E_0[P])).
    """
    rng = np.random.default_rng(2025)
    n_tumors = 80
    bg_genes = rng.binomial(1, 0.2, size=(12, n_tumors)).astype(float)
    mex1 = np.zeros(n_tumors)
    mex2 = np.zeros(n_tumors)
    mex1[:25] = 1.0
    mex2[25:50] = 1.0
    mat = np.vstack([mex1, mex2, bg_genes])
    names = ["MEX_A", "MEX_B"] + [f"G_{i}" for i in range(12)]
    bg = fit_discover_background(pd.DataFrame(mat, index=names))

    # Pairwise exclusivity: mid_p=True in default -> p(0) = 0.5 * Pr(X=0);
    # settings='R_default' forces mid_p=False even if mid_p=True is left at default!
    sub = bg.subset(genes=["MEX_A", "MEX_B", "G_0", "G_1", "G_2", "G_3"])
    res_def = pairwise_discover_test(sub, alternative="exclusivity", settings="default")
    res_r = pairwise_discover_test(sub, alternative="exclusivity", settings="R_default")

    assert res_def.mid_p is True
    assert res_r.mid_p is False
    # For k_obs == 0 on (MEX_A, MEX_B), inclusive p_value is exactly 2x mid-p p_value!
    np.testing.assert_allclose(res_r.p_value[0], 2.0 * res_def.p_value[0], rtol=1e-12)

    # Verify exact DBH q_value against brute-force reference formula from fdr.f90
    df_r = res_r.to_dataframe()
    m = len(df_r)
    name_to_idx = {g: i for i, g in enumerate(sub.gene_names)}
    supports_ref = []
    e0_ref = []
    for _, row in df_r.iterrows():
        i = name_to_idx[row["gene1"]]
        j = name_to_idx[row["gene2"]]
        pk = sub.bg_probs[i] * sub.bg_probs[j]
        pmf = poisson_binomial_pmf(pk, method="auto")
        cdf = np.clip(np.cumsum(pmf), 0.0, 1.0)
        supports_ref.append(cdf)
        e0_ref.append(float(np.dot(pmf, cdf)))
        np.testing.assert_allclose(row["p_value"], cdf[int(row["obs_co_mutations"])], rtol=1e-12)

    p_obs = df_r["p_value"].to_numpy(dtype=np.float64)
    pi0_ref = min(1.0, float(np.sum(p_obs)) / float(np.sum(e0_ref)))
    order = np.argsort(p_obs, kind="mergesort")
    p_sorted = p_obs[order]
    raw_q_ref = np.zeros(m, dtype=np.float64)
    for k in range(m):
        tau = p_sorted[k]
        sum_f = sum(
            float(np.max(s[s <= tau * (1.0 + 1e-14)])) if np.any(s <= tau * (1.0 + 1e-14)) else 0.0
            for s in supports_ref
        )
        raw_q_ref[k] = pi0_ref * sum_f / float(k + 1)
    q_sorted_ref = np.clip(np.minimum.accumulate(raw_q_ref[::-1])[::-1], 0.0, 1.0)
    q_ref = np.empty(m, dtype=np.float64)
    q_ref[order] = q_sorted_ref
    np.testing.assert_allclose(res_r.q_value, q_ref, rtol=1e-12, atol=1e-14)

    # Also verify groupwise_discover_test under settings='R_default' (both exclusivity & coverage)
    gsets = {
        "S1": ["MEX_A", "MEX_B", "G_0"],
        "S2": ["G_1", "G_2", "G_3"],
        "S3": ["G_4", "G_5", "G_6"],
    }
    gw_def = groupwise_discover_test(bg, gene_sets=gsets, method="coverage", settings="default")
    gw_r = groupwise_discover_test(bg, gene_sets=gsets, method="coverage", settings="R_default")
    assert gw_def.mid_p is True
    assert gw_r.mid_p is False
    assert np.all(gw_r.p_value >= gw_def.p_value - 1e-15)
    assert gw_r.q_value[0] < 0.05

    # 2-gene groupwise impurity == pairwise exclusivity holds in R_default too!
    pw_2_r = pairwise_discover_test(bg, g1=["MEX_A"], g2=["MEX_B"], settings="R_default")
    gw_2_r = groupwise_discover_test(bg, genes=["MEX_A", "MEX_B"], method="impurity", settings="R_default")
    np.testing.assert_allclose(gw_2_r.p_value, pw_2_r.p_value, rtol=1e-12)
    np.testing.assert_allclose(gw_2_r.q_value, pw_2_r.q_value, rtol=1e-12)
