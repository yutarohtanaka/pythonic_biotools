"""
Comprehensive unit tests for `lava` (Werme et al., Nature Genetics 2022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lava import (
    LAVA,
    LAVALocus,
    process_locus,
    project_psd_matrix,
    run_bivar,
    run_multireg,
    run_pcor,
    run_univ,
)


def _simulate_locus_gwas(
    rng: np.random.Generator,
    M: int = 80,
    N: int = 40_000,
    ld_rho: float = 0.80,
    true_effects: dict[str, np.ndarray] | None = None,
    overlap_cor: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], np.ndarray]:
    """Simulate realistic correlated GWAS Z-scores at an LD block."""
    idx = np.arange(M)
    R = ld_rho ** np.abs(idx[:, None] - idx[None, :])
    L_R = np.linalg.cholesky(R)

    if true_effects is None:
        b1 = rng.normal(0.0, 0.025, size=M)
        true_effects = {"T1": b1, "T2": 0.7 * b1 + rng.normal(0.0, 0.018, size=M)}

    traits = list(true_effects.keys())
    P = len(traits)
    if overlap_cor is None:
        overlap_cor = np.eye(P)

    L_C = np.linalg.cholesky(overlap_cor)
    # Matrix-normal noise E of shape (M, P) with row cov R and col cov overlap_cor: L_R @ Z0 @ L_C.T
    Z0 = rng.standard_normal((M, P))
    noise = L_R @ Z0 @ L_C.T

    cols: dict[str, np.ndarray] = {}
    for j, t in enumerate(traits):
        cols[t] = np.sqrt(N) * (R @ true_effects[t]) + noise[:, j]

    z_df = pd.DataFrame(cols, index=[f"rs{i + 1}" for i in range(M)])
    sample_sizes = {t: float(N) for t in traits}
    return z_df, R, sample_sizes, overlap_cor


def test_invalid_settings_raises() -> None:
    """All primary functions and classes raise ValueError on unknown settings."""
    rng = np.random.default_rng(1)
    z_df, R, ns, _ = _simulate_locus_gwas(rng, M=20)
    locus = process_locus(z_df, R, ns)

    with pytest.raises(ValueError, match="Unknown settings"):
        process_locus(z_df, R, ns, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_univ(locus, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_bivar(locus, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_pcor(locus, ("T1", "T2"), ["T1"], settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_multireg(locus, ["T1"], "T2", settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        LAVA(settings="bad")  # type: ignore[arg-type]


def test_process_locus_pca_variance_pruning_and_dimensions() -> None:
    """process_locus prunes collinear LD matrix to K < M PCs explaining >= lim variance."""
    rng = np.random.default_rng(10)
    z_df, R, ns, _ = _simulate_locus_gwas(rng, M=80, ld_rho=0.88)
    loc_99 = process_locus(z_df, R, ns, lim=0.99, locus_id="chr6_mhc")
    loc_90 = process_locus(z_df, R, ns, lim=0.90, locus_id="chr6_mhc")

    assert isinstance(loc_99, LAVALocus)
    assert loc_99.locus_id == "chr6_mhc"
    assert 1 <= loc_90.K < loc_99.K < 80
    assert loc_99.delta_hat.shape == (loc_99.K, 2)
    assert loc_99.omega.shape == (2, 2)
    assert loc_99.sigma.shape == (2, 2)


def test_psd_projection_default_vs_r_default() -> None:
    """Default projects non-PSD matrices with 1e-6 PD floor; R_default clips at exact 0."""
    non_psd = np.array([[1e-3, 1.2e-3], [1.2e-3, 1e-3]], dtype=float)
    assert np.min(np.linalg.eigvalsh(non_psd)) < 0.0

    proj_def = project_psd_matrix(non_psd, settings="default", pd_floor=1e-6)
    proj_r = project_psd_matrix(non_psd, settings="R_default")

    assert np.min(np.linalg.eigvalsh(proj_def)) == pytest.approx(1e-6, abs=1e-12)
    assert np.min(np.linalg.eigvalsh(proj_r)) == pytest.approx(0.0, abs=1e-12)


def test_univariate_heritability_recovery_and_null_calibration() -> None:
    """run_univ recovers positive local heritability on causal trait and non-significant p on null."""
    rng = np.random.default_rng(42)
    M, N = 70, 50_000
    b_causal = np.zeros(M)
    b_causal[[10, 30, 50]] = [0.06, -0.05, 0.05]  # causal SNPs
    b_null = np.zeros(M)

    z_df, R, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.7, true_effects={"Causal": b_causal, "Null": b_null}
    )
    true_h2 = float(b_causal @ R @ b_causal)

    locus = process_locus(z_df, R, ns)
    univ = run_univ(locus).set_index("trait")

    assert univ.loc["Causal", "p"] < 1e-10
    assert abs(univ.loc["Causal", "h2_loc"] - true_h2) < 0.004
    assert univ.loc["Null", "p"] > 0.05
    assert univ.loc["Null", "h2_loc"] < 0.002


def test_bivariate_positive_and_negative_local_rg() -> None:
    """run_bivar accurately estimates positive and negative local genetic correlations."""
    rng = np.random.default_rng(123)
    M, N = 75, 60_000
    g1 = rng.normal(0.0, 0.022, size=M)
    g_orth = rng.normal(0.0, 0.022, size=M)
    g_pos = 0.80 * g1 + 0.60 * g_orth
    g_neg = -0.75 * g1 + 0.66 * g_orth

    z_df, R, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.75, true_effects={"A": g1, "B_pos": g_pos, "C_neg": g_neg}
    )
    locus = process_locus(z_df, R, ns)
    biv = run_bivar(locus, n_sim=3000, seed=99)

    row_pos = biv[(biv["trait1"] == "A") & (biv["trait2"] == "B_pos")].iloc[0]
    row_neg = biv[(biv["trait1"] == "A") & (biv["trait2"] == "C_neg")].iloc[0]

    assert 0.60 < row_pos["rg"] < 0.95
    assert row_pos["ci_low"] > 0.40
    assert row_pos["p"] < 1e-4

    assert -0.95 < row_neg["rg"] < -0.50
    assert row_neg["ci_high"] < -0.30
    assert row_neg["p"] < 1e-4


def test_r_default_univ_thresh_gate_in_run_bivar() -> None:
    """settings='R_default' filters out univariate-null traits (p >= 0.05) in run_bivar."""
    rng = np.random.default_rng(314)
    M, N = 60, 40_000
    g = rng.normal(0.0, 0.025, size=M)
    effects = {"Active1": g, "Active2": 0.85 * g + rng.normal(0.0, 0.01, size=M), "NullTrait": np.zeros(M)}
    z_df, R, ns, _ = _simulate_locus_gwas(rng, M=M, N=N, true_effects=effects)

    locus = process_locus(z_df, R, ns)
    univ = run_univ(locus).set_index("trait")
    assert univ.loc["NullTrait", "p"] >= 0.05

    biv_def = run_bivar(locus, n_sim=1000, seed=1, settings="default")
    biv_r = run_bivar(locus, n_sim=1000, seed=1, settings="R_default")

    assert len(biv_def) == 3  # all 3 pairs evaluated in default
    assert len(biv_r) == 1  # NullTrait filtered out in R_default
    assert set([biv_r.loc[0, "trait1"], biv_r.loc[0, "trait2"]]) == {"Active1", "Active2"}


def test_sample_overlap_debiases_spurious_phenotypic_correlation() -> None:
    """Sample overlap matrix prevents false-positive rg between genetically independent traits."""
    rng = np.random.default_rng(777)
    M, N = 70, 8_000
    # Genetically orthogonal traits with realistic local h2 = 0.004 and heavy sample overlap C_12 = 0.80
    idx = np.arange(M)
    R = 0.7 ** np.abs(idx[:, None] - idx[None, :])
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    b1[:15] = 1.0
    b2[45:60] = 1.0
    b2 = b2 - (float(b1 @ R @ b2) / float(b1 @ R @ b1)) * b1
    b1 = b1 / np.sqrt(float(b1 @ R @ b1)) * np.sqrt(0.004)
    b2 = b2 / np.sqrt(float(b2 @ R @ b2)) * np.sqrt(0.004)

    overlap = np.array([[1.0, 0.80], [0.80, 1.0]])
    z_df, _, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.7, true_effects={"T1": b1, "T2": b2}, overlap_cor=overlap
    )

    loc_naive = process_locus(z_df, R, ns, sample_overlap=None)
    loc_corr = process_locus(z_df, R, ns, sample_overlap=overlap)

    biv_naive = run_bivar(loc_naive, n_sim=2000, seed=5)
    biv_corr = run_bivar(loc_corr, n_sim=2000, seed=5)

    # Naive ignoring sample overlap severely inflates rg; overlap-corrected removes the bias
    assert biv_naive.loc[0, "rg"] > 0.35
    assert biv_naive.loc[0, "p"] < 0.01
    assert abs(biv_corr.loc[0, "rg"]) < biv_naive.loc[0, "rg"] - 0.25
    assert biv_corr.loc[0, "p"] > 0.05


def test_multi_conditioner_pcor_and_validation_errors() -> None:
    """run_pcor supports conditioning on multiple traits Z1, Z2 and validates disjoint inputs."""
    rng = np.random.default_rng(1234)
    M, N = 60, 70_000
    idx = np.arange(M)
    R = 0.75 ** np.abs(idx[:, None] - idx[None, :])
    z1 = rng.normal(0.0, 0.02, size=M)
    z2 = rng.normal(0.0, 0.02, size=M)
    z2 -= (float(z2 @ R @ z1) / float(z1 @ R @ z1)) * z1
    ex = rng.normal(0.0, 0.018, size=M)
    ey = rng.normal(0.0, 0.018, size=M)
    for basis in (z1, z2):
        ex -= (float(ex @ R @ basis) / float(basis @ R @ basis)) * basis
        ey -= (float(ey @ R @ basis) / float(basis @ R @ basis)) * basis
    ey -= (float(ey @ R @ ex) / float(ex @ R @ ex)) * ex
    x = 0.7 * z1 + 0.6 * z2 + ex
    y = 0.7 * z1 + 0.6 * z2 + ey

    z_df, _, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.75, true_effects={"X": x, "Y": y, "Z1": z1, "Z2": z2}
    )
    locus = process_locus(z_df, R, ns)
    pcor2 = run_pcor(locus, pair=("X", "Y"), cond_traits=["Z1", "Z2"], n_sim=1500, seed=3)
    assert pcor2.loc[0, "cond_traits"] == "Z1,Z2"
    assert abs(pcor2.loc[0, "pcor"]) < 0.35

    with pytest.raises(ValueError, match="disjoint"):
        run_pcor(locus, pair=("X", "Y"), cond_traits=["X", "Z1"])
    with pytest.raises(ValueError, match="must not appear in predictors"):
        run_multireg(locus, predictors=["X", "Y"], outcome="Y")


def test_partial_correlation_removes_confounder_driven_association() -> None:
    """run_pcor collapses spurious marginal rg(X, Y) when mediated/confounded by Z."""
    rng = np.random.default_rng(2022)
    M, N = 80, 80_000
    b_z = rng.normal(0.0, 0.025, size=M)
    e_x = rng.normal(0.0, 0.015, size=M)
    e_y = rng.normal(0.0, 0.015, size=M)
    # Orthogonalize independent residuals so X and Y share genetics ONLY via Z
    idx = np.arange(M)
    R = 0.75 ** np.abs(idx[:, None] - idx[None, :])
    e_x -= (float(e_x @ R @ b_z) / float(b_z @ R @ b_z)) * b_z
    e_y -= (float(e_y @ R @ b_z) / float(b_z @ R @ b_z)) * b_z
    e_y -= (float(e_y @ R @ e_x) / float(e_x @ R @ e_x)) * e_x

    b_x = 0.85 * b_z + e_x
    b_y = 0.85 * b_z + e_y

    z_df, _, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.75, true_effects={"X": b_x, "Y": b_y, "Z": b_z}
    )
    locus = process_locus(z_df, R, ns)

    biv_xy = run_bivar(locus, traits=["X", "Y"], n_sim=2000, seed=12)
    pcor_xy_z = run_pcor(locus, pair=("X", "Y"), cond_traits=["Z"], n_sim=2000, seed=12)

    assert biv_xy.loc[0, "rg"] > 0.55
    assert biv_xy.loc[0, "p"] < 1e-4
    assert abs(pcor_xy_z.loc[0, "pcor"]) < 0.18
    assert pcor_xy_z.loc[0, "p"] > 0.05


def test_multireg_recovers_standardized_slopes_and_joint_r2() -> None:
    """run_multireg recovers standardized joint genetic slopes gamma and joint R^2."""
    rng = np.random.default_rng(888)
    M, N = 80, 90_000
    idx = np.arange(M)
    R = 0.72 ** np.abs(idx[:, None] - idx[None, :])

    # Construct two orthogonal unit-genetic-variance predictors X1, X2 and outcome Y
    u1 = rng.normal(0.0, 1.0, size=M)
    u1 = u1 / np.sqrt(float(u1 @ R @ u1)) * np.sqrt(0.02)
    u2 = rng.normal(0.0, 1.0, size=M)
    u2 -= (float(u2 @ R @ u1) / float(u1 @ R @ u1)) * u1
    u2 = u2 / np.sqrt(float(u2 @ R @ u2)) * np.sqrt(0.02)
    u_res = rng.normal(0.0, 1.0, size=M)
    u_res -= (float(u_res @ R @ u1) / float(u1 @ R @ u1)) * u1
    u_res -= (float(u_res @ R @ u2) / float(u2 @ R @ u2)) * u2
    u_res = u_res / np.sqrt(float(u_res @ R @ u_res)) * np.sqrt(0.02)

    # True standardized slopes: gamma_1 = 0.6, gamma_2 = 0.5 -> R^2 = 0.36 + 0.25 = 0.61
    b_y = 0.60 * u1 + 0.50 * u2 + np.sqrt(1.0 - 0.61) * u_res

    z_df, _, ns, _ = _simulate_locus_gwas(
        rng, M=M, N=N, ld_rho=0.72, true_effects={"X1": u1, "X2": u2, "Y": b_y}
    )
    locus = process_locus(z_df, R, ns)
    mreg = run_multireg(locus, predictors=["X1", "X2"], outcome="Y", n_sim=3000, seed=42)

    m_idx = mreg.set_index("predictor")
    assert abs(m_idx.loc["X1", "gamma"] - 0.60) < 0.12
    assert abs(m_idx.loc["X2", "gamma"] - 0.50) < 0.12
    assert m_idx.loc["X1", "p"] < 1e-3
    assert m_idx.loc["X2", "p"] < 1e-3
    assert abs(mreg.loc[0, "r2"] - 0.61) < 0.15
    assert mreg.loc[0, "p_joint"] < 1e-4


def test_single_predictor_multireg_matches_bivar() -> None:
    """With a single predictor, run_multireg gamma and r2 match run_bivar rg and r2."""
    rng = np.random.default_rng(55)
    z_df, R, ns, _ = _simulate_locus_gwas(rng, M=50, N=45_000)
    locus = process_locus(z_df, R, ns)

    biv = run_bivar(locus, n_sim=1500, seed=21)
    mr = run_multireg(locus, predictors=["T1"], outcome="T2", n_sim=1500, seed=21)

    assert mr.loc[0, "gamma"] == pytest.approx(biv.loc[0, "rg"], abs=1e-10)
    assert mr.loc[0, "r2"] == pytest.approx(biv.loc[0, "r2"], abs=1e-10)


def test_r_default_adaptive_refinement_and_lava_class() -> None:
    """LAVA class runs full workflow in R_default with 1e4->1e5 refinement on strong signal."""
    rng = np.random.default_rng(909)
    M, N = 65, 80_000
    g = rng.normal(0.0, 0.03, size=M)
    effects = {
        "P1": g,
        "P2": 0.9 * g + rng.normal(0.0, 0.01, size=M),
        "P3": 0.6 * g + rng.normal(0.0, 0.02, size=M),
    }
    z_df, R, ns, _ = _simulate_locus_gwas(rng, M=M, N=N, true_effects=effects)
    z_df_with_snp = z_df.reset_index().rename(columns={"index": "SNP"})

    model = LAVA(n_sim=1000, seed=17, settings="R_default")
    loc = model.process_locus(z_df_with_snp, R, ns, locus_id="loc_test")
    assert loc.traits == ["P1", "P2", "P3"]

    u_res = model.run_univ()
    b_res = model.run_bivar()
    p_res = model.run_pcor(pair=("P1", "P3"), cond_traits="P2")
    m_res = model.run_multireg(predictors=["P1", "P2"], outcome="P3")

    assert len(u_res) == 3
    assert b_res.loc[0, "p"] < 1e-6  # smooth tiny p-value via conditional Gaussian + refinement
    assert len(p_res) == 1
    assert len(m_res) == 2
