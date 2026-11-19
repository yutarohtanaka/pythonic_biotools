"""Comprehensive unit tests for `smr_heidi` (Zhu et al. 2016; Wu et al. 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from smr_heidi import heidi_test, pchisqsum, run_locus_smr_heidi, smr_multi, smr_single


def _simulate_cis_locus(
    rng: np.random.Generator,
    n_gwas: int = 8000,
    n_eqtl: int = 1500,
    m_snps: int = 12,
    rho_ar1: float = 0.78,
) -> np.ndarray:
    """Construct an AR(1) cis-LD correlation matrix with realistic moderate correlations."""
    idx = np.arange(m_snps)
    return rho_ar1 ** np.abs(idx[:, None] - idx[None, :])


def test_smr_single_exact_identities():
    b_zy, se_zy = 0.084, 0.014
    b_zx, se_zx = 0.420, 0.030
    res = smr_single(b_zy, se_zy, b_zx, se_zx, snp_id="rs100")

    assert np.isclose(res.b_xy, 0.20, rtol=1e-14)
    # Exact algebraic equivalence: (b_xy / se_xy)^2 == z_zy^2 * z_zx^2 / (z_zy^2 + z_zx^2)
    wald_chi2 = (res.b_xy / res.se_xy) ** 2
    assert np.isclose(res.t_smr, wald_chi2, rtol=1e-14)
    assert np.isclose(res.p_smr, stats.chi2.sf(wald_chi2, df=1), rtol=1e-14)
    assert res.to_dataframe().shape == (1, 9)
    assert "rs100" in res.summary()


def test_pchisqsum_imhof_saddlepoint_satterthwaite_accuracy():
    # Equal weights [1, 1, 1, 1, 1] => exact central chi^2_5
    w_eq = np.ones(5)
    for q in (2.0, 5.0, 12.5, 22.0):
        exact = float(stats.chi2.sf(q, df=5))
        assert np.isclose(pchisqsum(q, w_eq, method="imhof"), exact, rtol=1e-9, atol=1e-10)
        assert np.isclose(pchisqsum(q, w_eq, method="auto"), exact, rtol=1e-9, atol=1e-10)
        assert np.isclose(pchisqsum(q, w_eq, method="satterthwaite"), exact, rtol=1e-12)

    # Extreme far tail q = 80 (exact ~ 8.58e-16): auto / saddlepoint stay accurate without cancellation
    exact_far = float(stats.chi2.sf(80.0, df=5))
    p_auto_far = pchisqsum(80.0, w_eq, method="auto")
    p_sadd_far = pchisqsum(80.0, w_eq, method="saddlepoint")
    assert np.isclose(p_auto_far, exact_far, rtol=0.05)
    assert np.isclose(p_sadd_far, exact_far, rtol=0.05)

    # Rank-1 weights [4, 0, 0, 0] => exact 4 * chi^2_1
    w_r1 = np.array([4.0, 0.0, 0.0, 0.0])
    assert np.isclose(pchisqsum(10.0, w_r1, method="auto"), stats.chi2.sf(2.5, df=1), rtol=1e-12)

    # General unequal weights verified against Imhof vs Saddlepoint & MC
    lam = np.array([2.4, 1.3, 0.75, 0.25])
    p_im = pchisqsum(14.0, lam, method="imhof")
    p_sp = pchisqsum(14.0, lam, method="saddlepoint")
    assert np.isclose(p_im, p_sp, rtol=0.03)


def test_heidi_pleiotropy_null_calibration():
    """Under a single shared causal variant (pleiotropy), p_HEIDI is calibrated ~ Uniform(0, 1)."""
    rng = np.random.default_rng(2026)
    m = 10
    R = _simulate_cis_locus(rng, m_snps=m, rho_ar1=0.75)
    L = np.linalg.cholesky(R)

    causal_idx = 4
    true_beta_zx = 0.45 * R[:, causal_idx]  # marginal eQTL effects across LD block
    true_b_xy = 0.25
    true_beta_zy = true_b_xy * true_beta_zx  # exact mediation / single shared causal variant

    se_zx = np.full(m, 0.025)
    se_zy = np.full(m, 0.015)

    n_reps = 60
    p_heidis = []
    for _ in range(n_reps):
        b_zx = true_beta_zx + se_zx * (L @ rng.standard_normal(m))
        b_zy = true_beta_zy + se_zy * (L @ rng.standard_normal(m))
        res_h = heidi_test(b_zy, se_zy, b_zx, se_zx, R)
        assert res_h.n_snps >= 4
        p_heidis.append(res_h.p_heidi)

    p_arr = np.asarray(p_heidis)
    # Type-I error rate at alpha = 0.05 under true pleiotropy should be near 0.05
    assert 0.0 <= float(np.mean(p_arr < 0.05)) <= 0.14
    ks = stats.kstest(p_arr, "uniform")
    assert ks.pvalue > 0.02


def test_heidi_rejects_linkage_two_distinct_causal_snps():
    """When expression and trait have distinct causal SNPs in LD, HEIDI strongly rejects pleiotropy."""
    rng = np.random.default_rng(77)
    m = 12
    R = _simulate_cis_locus(rng, m_snps=m, rho_ar1=0.78)
    L = np.linalg.cholesky(R)

    # SNP 2 causes gene expression X; distinct SNP 6 causes GWAS trait Y directly!
    true_bzx = 0.50 * R[:, 2]
    true_bzy = 0.18 * R[:, 6]
    se_zx = np.full(m, 0.020)
    se_zy = np.full(m, 0.012)

    b_zx = true_bzx + se_zx * (L @ rng.standard_normal(m))
    b_zy = true_bzy + se_zy * (L @ rng.standard_normal(m))

    top_i = int(np.argmax((b_zx / se_zx) ** 2))
    smr_res = smr_single(b_zy[top_i], se_zy[top_i], b_zx[top_i], se_zx[top_i])
    heidi_res = heidi_test(b_zy, se_zy, b_zx, se_zx, R, top_idx=top_i)

    # Spurious single-SNP SMR signal due to LD, cleanly caught by HEIDI!
    assert smr_res.p_smr < 1e-4
    assert heidi_res.p_heidi < 1e-5
    assert heidi_res.snp_table().shape[0] == heidi_res.n_snps


def test_heidi_ld_pruning_window_and_min_snps():
    m = 6
    # Construct exact PSD factor-model LD matrix R = F F^T + diag(1 - f^2) wrt top SNP 0:
    # SNP 1: r = 0.98 (r^2 = 0.9604 > 0.9 -> pruned as too collinear)
    # SNP 2: r = 0.10 (r^2 = 0.0100 < 0.05 -> pruned as too weak LD)
    # SNP 3, 4: r = 0.50, 0.60 (r^2 = 0.25, 0.36 -> retained!)
    # SNP 5: r = 0.55, but weak eQTL p > p_het (pruned!)
    f = np.array([1.0, 0.98, 0.10, 0.50, 0.60, 0.55])
    R = np.outer(f, f) + np.diag(1.0 - f**2)
    np.fill_diagonal(R, 1.0)

    b_zx = np.array([0.50, 0.48, 0.30, 0.35, 0.32, 0.01])
    se_zx = np.full(m, 0.03)
    b_zy = 0.2 * b_zx
    se_zy = np.full(m, 0.02)

    res = heidi_test(b_zy, se_zy, b_zx, se_zx, R, top_idx=0, min_snps=3)
    assert res.selected_indices == [0, 3, 4]
    assert res.n_snps == 3

    # If min_snps=5, returns NaN gracefully without crashing
    res_few = heidi_test(b_zy, se_zy, b_zx, se_zx, R, top_idx=0, min_snps=5)
    assert np.isnan(res_few.t_heidi)
    assert np.isnan(res_few.p_heidi)


def test_smr_multi_power_gain_and_null_control():
    rng = np.random.default_rng(404)
    m = 8
    # Two near-independent LD blocks (4 SNPs each) harboring two independent cis-eQTLs!
    block = _simulate_cis_locus(rng, m_snps=4, rho_ar1=0.6)
    R = np.block([[block, np.zeros((4, 4))], [np.zeros((4, 4)), block]])

    se_zx = np.full(m, 0.025)
    se_zy = np.full(m, 0.020)
    b_zx = np.zeros(m)
    b_zx[:4] = 0.28 * block[:, 0]
    b_zx[4:] = 0.27 * block[:, 0]
    b_zy = 0.22 * b_zx

    top_i = int(np.argmax((b_zx / se_zx) ** 2))
    single = smr_single(b_zy[top_i], se_zy[top_i], b_zx[top_i], se_zx[top_i])
    multi = smr_multi(b_zy, se_zy, b_zx, se_zx, R, ld_prune_r2=0.1)

    assert multi.n_snps == 2
    assert multi.p_smr_multi < single.p_smr
    assert np.isclose(multi.b_xy_gls, 0.22, atol=1e-6)


def test_locus_runner_allele_and_ld_sign_flip_invariance():
    rng = np.random.default_rng(1234)
    m = 10
    R_base = _simulate_cis_locus(rng, m_snps=m, rho_ar1=0.72)
    L = np.linalg.cholesky(R_base)
    snps = [f"rs{i + 1}" for i in range(m)]
    a1 = ["A", "C", "G", "T", "A", "G", "C", "T", "A", "C"]
    a2 = ["G", "T", "A", "C", "C", "T", "G", "A", "T", "G"]

    se_zx = np.full(m, 0.022)
    se_zy = np.full(m, 0.015)
    b_zx = 0.42 * R_base[:, 3] + se_zx * (L @ rng.standard_normal(m))
    b_zy = 0.30 * b_zx + se_zy * (L @ rng.standard_normal(m))

    gwas_df = pd.DataFrame({"SNP": snps, "A1": a1, "A2": a2, "b": b_zy, "se": se_zy})
    eqtl_df = pd.DataFrame({"SNP": snps, "A1": a1, "A2": a2, "b": b_zx, "se": se_zx})

    base_res = run_locus_smr_heidi(
        gwas_df, eqtl_df, R_base, ld_snps=snps, ld_a1=a1, ld_a2=a2, probe_id="ENSG0001"
    )

    # Randomly flip alleles in GWAS, independently flip alleles in eQTL, and independently flip LD!
    g_flip = rng.choice([False, True], size=m)
    e_flip = rng.choice([False, True], size=m)
    l_flip = rng.choice([False, True], size=m)

    gwas_swapped = gwas_df.copy()
    for i in range(m):
        if g_flip[i]:
            gwas_swapped.loc[i, ["A1", "A2"]] = [a2[i], a1[i]]
            gwas_swapped.loc[i, "b"] = -gwas_swapped.loc[i, "b"]

    eqtl_swapped = eqtl_df.copy()
    for i in range(m):
        if e_flip[i]:
            eqtl_swapped.loc[i, ["A1", "A2"]] = [a2[i], a1[i]]
            eqtl_swapped.loc[i, "b"] = -eqtl_swapped.loc[i, "b"]

    ld_a1_sw = [a2[i] if l_flip[i] else a1[i] for i in range(m)]
    ld_a2_sw = [a1[i] if l_flip[i] else a2[i] for i in range(m)]
    l_signs = np.where(l_flip, -1.0, 1.0)
    R_swapped = R_base * np.outer(l_signs, l_signs)
    np.fill_diagonal(R_swapped, 1.0)

    swapped_res = run_locus_smr_heidi(
        gwas_swapped,
        eqtl_swapped,
        R_swapped,
        ld_snps=snps,
        ld_a1=ld_a1_sw,
        ld_a2=ld_a2_sw,
        probe_id="ENSG0001",
    )

    assert np.isclose(base_res.smr.b_xy, swapped_res.smr.b_xy, rtol=1e-12)
    assert np.isclose(base_res.smr.se_xy, swapped_res.smr.se_xy, rtol=1e-12)
    assert np.isclose(base_res.smr.p_smr, swapped_res.smr.p_smr, rtol=1e-12)
    assert np.isclose(base_res.heidi.t_heidi, swapped_res.heidi.t_heidi, rtol=1e-12)
    assert np.isclose(base_res.heidi.p_heidi, swapped_res.heidi.p_heidi, rtol=1e-11)
    assert base_res.smr_multi is not None and swapped_res.smr_multi is not None
    assert np.isclose(base_res.smr_multi.p_smr_multi, swapped_res.smr_multi.p_smr_multi, rtol=1e-11)
    assert "ENSG0001" in swapped_res.summary()
    assert swapped_res.to_dataframe().shape == (1, 15)


def test_incompatible_alleles_filtering_and_error():
    snps = ["rs1", "rs2", "rs3", "rs4", "rs5"]
    R = _simulate_cis_locus(np.random.default_rng(1), m_snps=5, rho_ar1=0.7)
    eqtl_df = pd.DataFrame(
        {
            "snp": snps,
            "a1": ["A", "C", "G", "T", "A"],
            "a2": ["G", "T", "A", "C", "G"],
            "beta": [0.45, 0.38, 0.35, 0.32, 0.30],
            "se": [0.02] * 5,
        }
    )
    gwas_df = eqtl_df.copy()
    gwas_df["beta"] = 0.2 * eqtl_df["beta"]
    # Corrupt rs5 alleles to incompatible A/C vs A/G
    gwas_df.loc[4, "a2"] = "C"

    res = run_locus_smr_heidi(gwas_df, eqtl_df, R, drop_incompatible=True)
    assert res.n_harmonized_snps == 4

    with pytest.raises(ValueError, match="Incompatible"):
        run_locus_smr_heidi(gwas_df, eqtl_df, R, drop_incompatible=False)


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError, match="zero"):
        smr_single(0.1, 0.02, 0.0, 0.02)
    with pytest.raises(ValueError, match="positive"):
        smr_single(0.1, -0.02, 0.3, 0.02)
    with pytest.raises(ValueError, match="finite"):
        smr_single(np.nan, 0.02, 0.3, 0.02)

    R = np.eye(4)
    b = np.array([0.4, 0.3, 0.2, 0.2])
    s = np.full(4, 0.02)

    R_asym = R.copy()
    R_asym[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        heidi_test(b, s, b, s, R_asym)

    R_bad_diag = R.copy()
    R_bad_diag[2, 2] = 0.8
    with pytest.raises(ValueError, match="diagonal"):
        heidi_test(b, s, b, s, R_bad_diag)
