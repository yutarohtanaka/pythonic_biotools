"""
Comprehensive unit tests for `genomicsem` (Grotzinger et al., Nature Human Behaviour 2019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomicsem import (
    GenomicCovStruct,
    GenomicSEM,
    commonfactor,
    commonfactor_gwas,
    ldsc_covstruct,
    parse_lavaan_syntax,
    ram_implied_covariance_and_jacobian,
    unvech,
    usermodel,
    vech,
)


def _make_four_trait_cf_covstruct(settings: str = "default") -> tuple[GenomicCovStruct, np.ndarray, np.ndarray]:
    """Create exact 4-trait common-factor covariance structure with realistic V_S."""
    lam_true = np.array([0.55, 0.48, 0.42, 0.36], dtype=float)
    u_true = np.array([0.12, 0.16, 0.20, 0.24], dtype=float)
    S = np.outer(lam_true, lam_true) + np.diag(u_true)
    V_S = np.diag([1.2, 0.8, 0.7, 0.6, 1.4, 0.8, 0.7, 1.5, 0.8, 1.6]) * 1e-4
    I_mat = np.array(
        [
            [1.02, 0.25, 0.10, 0.05],
            [0.25, 1.01, 0.15, 0.08],
            [0.10, 0.15, 1.00, 0.20],
            [0.05, 0.08, 0.20, 1.03],
        ]
    )
    cov = GenomicCovStruct(
        S=S,
        V_S=V_S,
        I_mat=I_mat,
        trait_names=["T1", "T2", "T3", "T4"],
        settings=settings,  # type: ignore[arg-type]
    )
    return cov, lam_true, u_true


def test_invalid_settings_raises() -> None:
    """Every primary function and class raises ValueError on unknown settings."""
    cov, _, _ = _make_four_trait_cf_covstruct()
    with pytest.raises(ValueError, match="Unknown settings"):
        GenomicCovStruct(S=cov.S, V_S=cov.V_S, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        ldsc_covstruct((cov.S, cov.V_S), settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        parse_lavaan_syntax("F1 =~ T1 + T2 + T3", ["T1", "T2", "T3"], settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        usermodel(cov, "F1 =~ T1 + T2 + T3 + T4", settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        commonfactor(cov, settings="bad")  # type: ignore[arg-type]
    snps = [pd.DataFrame({"SNP": ["rs1"], "BETA": [0.01], "SE": [0.005]}) for _ in range(4)]
    with pytest.raises(ValueError, match="Unknown settings"):
        commonfactor_gwas(snps, cov, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        GenomicSEM(settings="bad")  # type: ignore[arg-type]


def test_vech_and_unvech_column_major_roundtrip() -> None:
    """vech follows exact GenomicSEM/lavaan column-major lower-triangular ordering."""
    M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    v = vech(M)
    np.testing.assert_array_equal(v, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(unvech(v, 3), M)


def test_ldsc_covstruct_recovers_genetic_covariance_and_intercepts() -> None:
    """ldsc_covstruct estimates S, bivariate sample-overlap I_mat, and jackknife V_S."""
    rng = np.random.default_rng(2019)
    M_snps = 20_000
    m_ref = 400_000.0
    N1, N2, N3 = 150_000.0, 160_000.0, 140_000.0
    ell = rng.uniform(1.0, 35.0, size=M_snps)

    S_true = np.array(
        [
            [0.30, 0.18, 0.12],
            [0.18, 0.25, 0.10],
            [0.12, 0.10, 0.22],
        ]
    )
    I_true = np.array(
        [
            [1.00, 0.35, 0.00],
            [0.35, 1.00, 0.20],
            [0.00, 0.20, 1.00],
        ]
    )

    # Generate Z-scores matching exact LDSC marginal moments:
    # Cov(Z_j) = (sqrt(N_p N_q) / m_ref) S_true * ell_j + I_true
    L_S = np.linalg.cholesky(S_true)
    L_I = np.linalg.cholesky(I_true)
    sqrt_n = np.sqrt([N1, N2, N3])
    g_part = (
        rng.standard_normal((M_snps, 3))
        @ L_S.T
        * np.sqrt(ell / m_ref)[:, None]
        * sqrt_n[None, :]
    )
    e_part = rng.standard_normal((M_snps, 3)) @ L_I.T
    Z = g_part + e_part

    sumstats = [
        pd.DataFrame({"Z": Z[:, p], "N": n_p})
        for p, n_p in enumerate([N1, N2, N3])
    ]
    cov = ldsc_covstruct(
        sumstats,
        ld_scores=ell,
        m_snps=m_ref,
        trait_names=["Cog", "EA", "SES"],
        n_blocks=100,
        settings="default",
    )

    assert cov.S.shape == (3, 3)
    assert cov.V_S.shape == (6, 6)
    assert cov.I_mat is not None and cov.I_mat.shape == (3, 3)
    np.testing.assert_allclose(cov.S, S_true, atol=0.04)
    np.testing.assert_allclose(cov.I_mat, I_true, atol=0.06)
    assert np.all(np.diag(cov.V_S) > 0.0)


def test_analytical_ram_jacobian_matches_finite_differences() -> None:
    """Analytical RAM Jacobian matches central finite differences to 1e-8."""
    spec = parse_lavaan_syntax(
        "F1 =~ T1 + T2\nF2 =~ T3 + T4\nF2 ~ F1\nT1 ~~ T3",
        observed_traits=["T1", "T2", "T3", "T4"],
        std_lv=True,
    )
    theta = np.array([0.5, 0.4, 0.6, 0.35, 0.45, 0.12, 0.18, 0.22, 0.15, 0.19])
    assert len(theta) == spec.n_free
    _, _, J_ana = ram_implied_covariance_and_jacobian(theta, spec)

    eps = 1e-6
    J_num = np.zeros_like(J_ana)
    for r in range(spec.n_free):
        tp = theta.copy()
        tm = theta.copy()
        tp[r] += eps
        tm[r] -= eps
        sp, _, _ = ram_implied_covariance_and_jacobian(tp, spec)
        sm, _, _ = ram_implied_covariance_and_jacobian(tm, spec)
        J_num[:, r] = (vech(sp) - vech(sm)) / (2.0 * eps)

    np.testing.assert_allclose(J_ana, J_num, atol=1e-8, rtol=1e-6)


def test_commonfactor_dwls_and_ml_recover_true_parameters() -> None:
    """commonfactor recovers exact factor loadings, residual variances, and CFI=1, SRMR=0."""
    cov, lam_true, u_true = _make_four_trait_cf_covstruct()

    res_dwls = commonfactor(cov, estimation="DWLS", settings="default")
    res_ml = commonfactor(cov, estimation="ML", settings="default")

    for res in (res_dwls, res_ml):
        assert res.converged is True
        assert res.modelfit["df"] == 2  # 10 moments - 8 params
        assert res.modelfit["chisq"] < 1e-6
        assert res.modelfit["CFI"] == pytest.approx(1.0, abs=1e-5)
        assert res.modelfit["SRMR"] < 1e-5

        loads = res.results_df[res.results_df["op"] == "=~"]["est"].to_numpy()
        np.testing.assert_allclose(loads, lam_true, atol=1e-4)

        resids = res.results_df[
            (res.results_df["op"] == "~~") & (res.results_df["lhs"].isin(["T1", "T2", "T3", "T4"]))
        ]["est"].to_numpy()
        np.testing.assert_allclose(resids, u_true, atol=1e-4)
        assert np.all(res.results_df.loc[res.results_df["free"], "se"] > 0.0)


def test_correlated_two_factor_model_recovers_interfactor_rg() -> None:
    """usermodel estimates correlated two-factor CFA and detects misfit of one-factor CFA."""
    # 6 traits: F1 -> T1..T3, F2 -> T4..T6, true Corr(F1, F2) = 0.45
    L = np.array(
        [
            [0.60, 0.0],
            [0.55, 0.0],
            [0.50, 0.0],
            [0.0, 0.58],
            [0.0, 0.52],
            [0.0, 0.48],
        ]
    )
    Phi = np.array([[1.0, 0.45], [0.45, 1.0]])
    U = np.diag([0.15, 0.18, 0.20, 0.16, 0.19, 0.22])
    S = L @ Phi @ L.T + U
    p_star = 6 * 7 // 2
    V_S = np.eye(p_star) * 5e-5
    traits = [f"T{i + 1}" for i in range(6)]
    cov = GenomicCovStruct(S=S, V_S=V_S, trait_names=traits)

    two_fac_str = """
    F1 =~ T1 + T2 + T3
    F2 =~ T4 + T5 + T6
    F1 ~~ F2
    """
    res_2f = usermodel(cov, two_fac_str, settings="default")
    res_1f = commonfactor(cov, settings="default")

    cov_row = res_2f.results_df[
        (res_2f.results_df["lhs"] == "F1") & (res_2f.results_df["rhs"] == "F2")
    ].iloc[0]
    assert cov_row["est"] == pytest.approx(0.45, abs=1e-3)
    assert res_2f.modelfit["df"] == 8  # 21 - 13
    assert res_2f.modelfit["CFI"] > 0.999
    assert res_2f.modelfit["SRMR"] < 1e-4

    # Misspecified 1-factor model has substantial chisq & worse AIC/SRMR
    assert res_1f.modelfit["chisq"] > 50.0
    assert res_2f.modelfit["AIC"] < res_1f.modelfit["AIC"]


def test_gwas_by_subtraction_cholesky_model() -> None:
    """usermodel fits exact Cholesky GWAS-by-subtraction model with fixed 0* residual variances."""
    # Cog (T1): h2 = 0.25; EA (T2): h2 = 0.20, genetic cov = 0.15
    # Exact Cholesky: l11 = sqrt(0.25) = 0.50; l21 = 0.15 / 0.50 = 0.30;
    # l22 (NonCog) = sqrt(0.20 - 0.09) = sqrt(0.11) = 0.33166248
    S = np.array([[0.25, 0.15], [0.15, 0.20]])
    V_S = np.diag([1e-4, 0.8e-4, 1e-4])
    cov = GenomicCovStruct(S=S, V_S=V_S, trait_names=["Cog", "EA"])

    cholesky_syntax = """
    CogLat =~ NA*Cog + NA*EA
    NonCog =~ 0*Cog + NA*EA
    CogLat ~~ 1*CogLat
    NonCog ~~ 1*NonCog
    CogLat ~~ 0*NonCog
    Cog ~~ 0*Cog
    EA ~~ 0*EA
    """
    res = usermodel(cov, cholesky_syntax, settings="default")
    assert res.modelfit["df"] == 0
    df_p = res.results_df

    l11 = float(df_p[(df_p["lhs"] == "CogLat") & (df_p["rhs"] == "Cog")]["est"].iloc[0])
    l21 = float(df_p[(df_p["lhs"] == "CogLat") & (df_p["rhs"] == "EA")]["est"].iloc[0])
    l22 = float(df_p[(df_p["lhs"] == "NonCog") & (df_p["rhs"] == "EA")]["est"].iloc[0])
    u_cog = float(df_p[(df_p["lhs"] == "Cog") & (df_p["op"] == "~~")]["est"].iloc[0])

    assert l11 == pytest.approx(0.50, abs=1e-4)
    assert l21 == pytest.approx(0.30, abs=1e-4)
    assert l22 == pytest.approx(np.sqrt(0.11), abs=1e-4)
    assert u_cog == 0.0  # Fixed 0* residual variance is NOT pushed to 1e-6!


def test_heywood_case_default_bounded_vs_r_default_unconstrained() -> None:
    """On a Heywood covariance matrix, default bounds residual variance >= 1e-6 while R_default allows < 0."""
    # Construct 3-trait covariance where implied u_1 = S_11 - (S_12 * S_13 / S_23) = 0.30 - (0.24*0.24/0.16) = -0.06 < 0!
    S_heywood = np.array(
        [
            [0.30, 0.24, 0.24],
            [0.24, 0.35, 0.16],
            [0.24, 0.16, 0.35],
        ]
    )
    V_S = np.eye(6) * 1e-4

    res_def = commonfactor((S_heywood, V_S), settings="default")
    res_r = commonfactor((S_heywood, V_S), settings="R_default")

    u1_def = float(
        res_def.results_df[(res_def.results_df["lhs"] == "T1") & (res_def.results_df["op"] == "~~")][
            "est"
        ].iloc[0]
    )
    u1_r = float(
        res_r.results_df[(res_r.results_df["lhs"] == "T1") & (res_r.results_df["op"] == "~~")][
            "est"
        ].iloc[0]
    )

    assert u1_def >= 1e-6 - 1e-12
    assert u1_r == pytest.approx(-0.06, abs=1e-3)


def test_structural_regression_and_unit_loading_std_lv_false() -> None:
    """usermodel supports directed regressions (T3 ~ F1) and unit-loading identification (std_lv=False)."""
    cov, lam_true, _ = _make_four_trait_cf_covstruct()
    res_ul = usermodel(
        cov,
        "F1 =~ T1 + T2 + T3\nT4 ~ F1",
        std_lv=False,
        settings="default",
    )
    df_p = res_ul.results_df
    first_load = df_p[(df_p["lhs"] == "F1") & (df_p["rhs"] == "T1")].iloc[0]
    f_var = df_p[(df_p["lhs"] == "F1") & (df_p["op"] == "~~")].iloc[0]
    reg_t4 = df_p[(df_p["lhs"] == "T4") & (df_p["op"] == "~")].iloc[0]

    assert bool(first_load["free"]) is False
    assert float(first_load["est"]) == 1.0
    assert float(f_var["est"]) == pytest.approx(lam_true[0] ** 2, abs=1e-3)
    assert float(reg_t4["est"]) == pytest.approx(lam_true[3] / lam_true[0], abs=1e-3)


def test_commonfactor_gwas_recovers_beta_f_and_flags_q_snp_heterogeneity() -> None:
    """commonfactor_gwas recovers common-pathway SNP effects and flags heterogeneous SNPs via Q_SNP."""
    cov, lam_true, _ = _make_four_trait_cf_covstruct()
    # SNP 1: pure common-factor SNP with true beta_F = 0.030 -> gamma = 0.030 * lam_true (Q_SNP ~ 0)
    # SNP 2: null SNP -> gamma = 0 (beta_F ~ 0, Q_SNP ~ 0)
    # SNP 3: heterogeneous trait-1-only SNP -> gamma = [0.045, 0, 0, 0] (large Q_SNP, tiny Q_pval!)
    # SNP 4: opposite-sign discordant SNP -> gamma = [0.03, -0.03, 0.02, -0.02] (beta_F ~ 0, huge Q_SNP!)
    gamma_true = np.vstack(
        [
            0.030 * lam_true,
            np.zeros(4),
            np.array([0.045, 0.0, 0.0, 0.0]),
            np.array([0.030, -0.030, 0.020, -0.020]),
        ]
    )
    se_val = 0.0035
    snp_dfs = [
        pd.DataFrame(
            {
                "SNP": ["rs_common", "rs_null", "rs_t1_specific", "rs_discordant"],
                "BETA": gamma_true[:, p],
                "SE": np.full(4, se_val),
            }
        )
        for p in range(4)
    ]

    gwas = commonfactor_gwas(snp_dfs, cov, smooth_check=True, settings="default").set_index("SNP")

    # rs_common: accurate beta_F, strong GWAS p_value, non-significant Q_SNP
    assert gwas.loc["rs_common", "beta_F"] == pytest.approx(0.030, abs=2e-3)
    assert gwas.loc["rs_common", "p_value"] < 1e-8
    assert gwas.loc["rs_common", "Q_df"] == 3
    assert gwas.loc["rs_common", "Q_pval"] > 0.50

    # rs_null: non-significant beta_F and non-significant Q_SNP
    assert abs(gwas.loc["rs_null", "beta_F"]) < 1e-6
    assert gwas.loc["rs_null", "p_value"] > 0.90
    assert gwas.loc["rs_null", "Q_pval"] > 0.90

    # rs_t1_specific & rs_discordant: strongly flagged by Q_SNP heterogeneity test!
    assert gwas.loc["rs_t1_specific", "Q_SNP"] > 50.0
    assert gwas.loc["rs_t1_specific", "Q_pval"] < 1e-10
    assert gwas.loc["rs_discordant", "Q_SNP"] > 80.0
    assert gwas.loc["rs_discordant", "Q_pval"] < 1e-15


def test_commonfactor_gwas_smooth_check_handles_non_psd_snp() -> None:
    """smooth_check detects and projects impossible non-PSD SNP-trait covariances."""
    cov, lam_true, _ = _make_four_trait_cf_covstruct()
    # Impossible SNP effect > 1 on standardized scale makes augmented (k+1)x(k+1) matrix non-PSD
    snp_dfs = [
        pd.DataFrame(
            {
                "SNP": ["rs_ok", "rs_nonpsd"],
                "BETA": [0.02 * lam_true[p], 0.85 if p == 0 else -0.85],
                "SE": [0.004, 0.05],
            }
        )
        for p in range(4)
    ]
    gwas = commonfactor_gwas(snp_dfs, cov, smooth_check=True, settings="R_default").set_index("SNP")
    assert bool(gwas.loc["rs_ok", "smoothed"]) is False
    assert gwas.loc["rs_ok", "Z_smooth"] == 0.0
    assert bool(gwas.loc["rs_nonpsd", "smoothed"]) is True
    assert gwas.loc["rs_nonpsd", "Z_smooth"] > 0.0


def test_genomicsem_class_and_maf_standardization() -> None:
    """GenomicSEM class runs end-to-end pipeline and standardizes raw per-allele BETAs via MAF."""
    cov, lam_true, _ = _make_four_trait_cf_covstruct()
    maf = np.array([0.25, 0.10])
    sd_g = np.sqrt(2.0 * maf * (1.0 - maf))
    # Supply raw per-allele BETA = beta_std / sd_g so MAF standardization recovers exact beta_F = 0.025
    snp_dfs = [
        pd.DataFrame(
            {
                "SNP": ["rs_maf1", "rs_maf2"],
                "MAF": maf,
                "BETA": (0.025 * lam_true[p]) / sd_g,
                "SE": 0.003 / sd_g,
            }
        )
        for p in range(4)
    ]

    sem = GenomicSEM(estimation="DWLS", settings="R_default")
    sem.ldsc((cov.S, cov.V_S, cov.I_mat), trait_names=cov.trait_names)
    cf = sem.commonfactor()
    assert cf.converged is True
    assert "UserModelResult" in cf.summary()

    gwas = sem.commonfactor_gwas(snp_dfs)
    np.testing.assert_allclose(gwas["beta_F"].to_numpy(), [0.025, 0.025], atol=1e-3)
