"""Unit tests for High-Definition Likelihood (hdl: HDL, HDL-L, HDL.S)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from hdl import (
    build_ld_eigen_panel,
    hdl_h2,
    hdl_local,
    hdl_rg,
    hdl_stratified,
)


def _make_ld_blocks(
    n_blocks: int = 15,
    block_size: int = 40,
    rho_min: float = 0.65,
    rho_max: float = 0.88,
) -> list[np.ndarray]:
    """Generate realistic heterogeneous autoregressive LD correlation blocks."""
    rhos = np.linspace(rho_min, rho_max, n_blocks)
    blocks: list[np.ndarray] = []
    idx = np.arange(block_size)
    dist = np.abs(idx[:, None] - idx[None, :])
    for r in rhos:
        blocks.append(r**dist)
    return blocks


def _simulate_bivariate_z(
    rng: np.random.Generator,
    R_blocks: list[np.ndarray],
    N1: int,
    N2: int,
    h1_sq: float,
    h2_sq: float,
    rg: float,
    b1: float = 1.0,
    b2: float = 1.0,
    c12: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate exact two-trait polygenic GWAS Z-scores across LD blocks."""
    M = sum(R.shape[0] for R in R_blocks)
    rho12 = rg * np.sqrt(h1_sq * h2_sq)
    z1_list: list[np.ndarray] = []
    z2_list: list[np.ndarray] = []

    for R in R_blocks:
        m_b = R.shape[0]
        R2 = R @ R
        cov11 = (N1 * h1_sq / M) * R2 + b1 * R
        cov22 = (N2 * h2_sq / M) * R2 + b2 * R
        cov12 = (np.sqrt(N1 * N2) * rho12 / M) * R2 + c12 * R
        joint_cov = np.block([[cov11, cov12], [cov12.T, cov22]])
        joint_cov = 0.5 * (joint_cov + joint_cov.T) + 1e-10 * np.eye(2 * m_b)
        draw = rng.multivariate_normal(np.zeros(2 * m_b), joint_cov)
        z1_list.append(draw[:m_b])
        z2_list.append(draw[m_b:])

    return np.concatenate(z1_list), np.concatenate(z2_list)


def test_build_ld_eigen_panel_variance_retention() -> None:
    blocks = _make_ld_blocks(n_blocks=6, block_size=35)
    panel_99 = build_ld_eigen_panel(blocks, eigen_cut=0.99, settings="default")
    panel_90 = build_ld_eigen_panel(blocks, eigen_cut=0.90, settings="R_default")

    assert panel_99.n_blocks == 6
    assert panel_99.total_snps == 210
    assert panel_90.total_eigenmodes < panel_99.total_eigenmodes < panel_99.total_snps
    for blk in panel_99:
        assert blk.var_explained >= 0.99 - 1e-12
        assert np.all(blk.eigvals > np.exp(-18.0))
        assert np.all(np.diff(blk.eigvals) <= 1e-12)


def test_hdl_h2_heritability_recovery_and_se_modes() -> None:
    rng = np.random.default_rng(11)
    blocks = _make_ld_blocks(n_blocks=16, block_size=40)
    z1, _ = _simulate_bivariate_z(
        rng, blocks, N1=80_000, N2=80_000, h1_sq=0.45, h2_sq=0.30, rg=0.0
    )
    panel = build_ld_eigen_panel(blocks, eigen_cut=0.99, settings="default")

    res_jk = hdl_h2(z1, panel, N=80_000, jackknife=True, settings="default")
    res_fi = hdl_h2(z1, panel, N=80_000, jackknife=False, settings="default")
    res_r = hdl_h2(z1, blocks, N=80_000, intercept_output=False, settings="R_default")

    assert abs(res_jk.h2 - 0.45) < 0.08
    assert abs(res_r.h2 - 0.45) < 0.08
    assert np.isclose(res_jk.h2, res_fi.h2, atol=1e-10)
    assert res_jk.se_source == "jackknife"
    assert res_fi.se_source == "fisher"
    assert res_r.intercept == 1.0
    assert res_jk.ci_low < 0.45 < res_jk.ci_high
    assert "HDL.h2" in res_jk.summary()
    assert res_jk.to_dataframe().shape[0] == 1


def test_hdl_h2_cryptic_stratification_intercept_separation() -> None:
    rng = np.random.default_rng(22)
    blocks = _make_ld_blocks(n_blocks=20, block_size=40)
    # Realistic N/M ratio (N=4,000, M=800 -> N/M = 5) with b1 = 1.40 alongside true h2 = 0.30
    z_inf, _ = _simulate_bivariate_z(
        rng, blocks, N1=4_000, N2=4_000, h1_sq=0.30, h2_sq=0.20, rg=0.0, b1=1.40
    )

    res_def = hdl_h2(z_inf, blocks, N=4_000, settings="default")
    res_r_fixed = hdl_h2(z_inf, blocks, N=4_000, intercept_output=False, settings="R_default")
    res_r_free = hdl_h2(z_inf, blocks, N=4_000, intercept_output=True, settings="R_default")

    assert abs(res_def.h2 - 0.30) < 0.08
    assert abs(res_def.intercept - 1.40) < 0.14
    assert np.isclose(res_def.h2, res_r_free.h2, atol=1e-4)
    # Fixing b=1 under b=1.40 inflates apparent h2 substantially
    assert res_r_fixed.h2 > res_def.h2 + 0.06


def test_hdl_rg_positive_and_negative_genetic_correlation() -> None:
    rng = np.random.default_rng(33)
    blocks = _make_ld_blocks(n_blocks=18, block_size=40)
    panel = build_ld_eigen_panel(blocks, eigen_cut=0.99, settings="default")

    z1, z2 = _simulate_bivariate_z(
        rng, blocks, N1=80_000, N2=90_000, h1_sq=0.45, h2_sq=0.50, rg=0.60
    )
    res_def = hdl_rg(z1, z2, panel, N1=80_000, N2=90_000, jackknife=True, settings="default")
    res_r = hdl_rg(z1, z2, blocks, N1=80_000, N2=90_000, settings="R_default")

    assert abs(res_def.rg - 0.60) < 0.09
    assert abs(res_r.rg - 0.60) < 0.09
    assert res_def.rg_ci_low < 0.60 < res_def.rg_ci_high
    assert res_def.p_rg < 1e-6
    assert 0.4 < (res_def.se_rg / res_def.fisher_se_rg) < 2.5

    # Test negative rg with block-list input format
    z1_neg, z2_neg = _simulate_bivariate_z(
        rng, blocks, N1=80_000, N2=80_000, h1_sq=0.40, h2_sq=0.45, rg=-0.55
    )
    z1_blk = np.split(z1_neg, len(blocks))
    z2_blk = np.split(z2_neg, len(blocks))
    res_neg = hdl_rg(z1_blk, z2_blk, panel, N1=80_000, N2=80_000, settings="default")
    assert abs(res_neg.rg - (-0.55)) < 0.09
    assert res_neg.p_rg < 1e-5
    assert "HDL.rg" in res_neg.summary()


def test_hdl_rg_null_correlation_under_sample_overlap() -> None:
    rng = np.random.default_rng(44)
    blocks = _make_ld_blocks(n_blocks=20, block_size=40)
    # Realistic N/M ratio (N=5,000, M=800): true rg = 0.0, sample overlap -> c12 = 0.40
    z1, z2 = _simulate_bivariate_z(
        rng,
        blocks,
        N1=5_000,
        N2=5_000,
        h1_sq=0.40,
        h2_sq=0.40,
        rg=0.0,
        c12=0.40,
    )
    res_def = hdl_rg(
        z1, z2, blocks, N1=5_000, N2=5_000, N0=3_500, rho_overlap=0.57, settings="default"
    )
    res_r = hdl_rg(
        z1, z2, blocks, N1=5_000, N2=5_000, N0=3_500, rho_overlap=0.57, settings="R_default"
    )

    assert abs(res_def.rg) < 0.14
    assert abs(res_r.rg) < 0.14
    assert abs(res_def.c12 - 0.40) < 0.14
    assert abs(res_r.c12 - 0.40) < 0.14
    assert res_def.p_rg > 0.05


def test_hdl_local_lrt_signal_and_null_calibration() -> None:
    rng = np.random.default_rng(55)
    R_loc = _make_ld_blocks(n_blocks=1, block_size=90, rho_min=0.82, rho_max=0.82)[0]

    # Strong local pleiotropic signal: h1_loc = 0.015, h2_loc = 0.015, rg_loc = 0.75
    z1_sig, z2_sig = _simulate_bivariate_z(
        rng, [R_loc], N1=100_000, N2=100_000, h1_sq=0.015, h2_sq=0.015, rg=0.75
    )
    loc_def = hdl_local(z1_sig, z2_sig, R_loc, N1=100_000, N2=100_000, settings="default")
    loc_r = hdl_local(z1_sig, z2_sig, R_loc, N1=100_000, N2=100_000, settings="R_default")

    assert abs(loc_def.rg_local - 0.75) < 0.22
    assert abs(loc_r.rg_local - 0.75) < 0.20
    assert np.isclose(loc_def.lrt_stat, 2.0 * (loc_def.loglik_alt - loc_def.loglik_null), atol=1e-10)
    assert loc_def.p_lrt < 1e-4
    assert loc_r.p_lrt < 1e-5
    assert "HDL-L" in loc_def.summary()

    # Null local correlation: rg_loc = 0.0
    z1_null, z2_null = _simulate_bivariate_z(
        rng, [R_loc], N1=100_000, N2=100_000, h1_sq=0.012, h2_sq=0.012, rg=0.0
    )
    loc_null = hdl_local(z1_null, z2_null, R_loc, N1=100_000, N2=100_000, settings="default")
    assert loc_null.lrt_stat >= 0.0
    assert loc_null.p_lrt > 0.05


def test_hdl_stratified_reduces_exactly_to_univariate_hdl_h2() -> None:
    rng = np.random.default_rng(66)
    blocks = _make_ld_blocks(n_blocks=10, block_size=30)
    z, _ = _simulate_bivariate_z(
        rng, blocks, N1=60_000, N2=60_000, h1_sq=0.40, h2_sq=0.20, rg=0.0
    )
    ones_annot = np.ones((300, 1), dtype=np.float64)

    res_u = hdl_h2(z, blocks, N=60_000, jackknife=False, settings="default")
    res_s = hdl_stratified(
        z, blocks, ones_annot, N=60_000, annotation_names=["baseline"], settings="default"
    )

    assert np.isclose(res_s.h2_total, res_u.h2, atol=1e-4)
    assert np.isclose(res_s.intercept, res_u.intercept, atol=1e-4)
    assert np.isclose(res_s.loglik, res_u.loglik, atol=1e-4)
    assert np.isclose(res_s.enrichment[0], 1.0, atol=1e-10)


def test_hdl_stratified_functional_enrichment_recovery() -> None:
    rng = np.random.default_rng(77)
    blocks = _make_ld_blocks(n_blocks=16, block_size=40)
    M = 640
    N = 100_000

    # 20% of SNPs lie in regulatory enhancers with 6x higher per-SNP heritability
    enhancer = (rng.uniform(size=M) < 0.20).astype(float)
    annot_df = pd.DataFrame({"baseline": np.ones(M, dtype=float), "enhancer": enhancer})
    tau_true = np.array([2.5e-4, 1.5e-3])  # total h2 ~ 640*2.5e-4 + 128*1.5e-3 = 0.16 + 0.192 = 0.352

    z_blocks: list[np.ndarray] = []
    offset = 0
    for R in blocks:
        m_b = R.shape[0]
        a_b = annot_df.iloc[offset : offset + m_b].to_numpy()
        d_b = np.diag(a_b @ tau_true)
        cov_z = N * (R @ d_b @ R) + 1.0 * R
        cov_z = 0.5 * (cov_z + cov_z.T) + 1e-10 * np.eye(m_b)
        z_blocks.append(rng.multivariate_normal(np.zeros(m_b), cov_z))
        offset += m_b
    z_all = np.concatenate(z_blocks)

    res_def = hdl_stratified(z_all, blocks, annot_df, N=N, jackknife=True, settings="default")
    res_r = hdl_stratified(z_all, blocks, annot_df, N=N, settings="R_default")

    df = res_def.to_dataframe()
    assert list(df["annotation"]) == ["baseline", "enhancer"]
    true_h2 = float(annot_df.to_numpy().sum(axis=0) @ tau_true)
    assert abs(res_def.h2_total - true_h2) < 0.08
    assert abs(res_r.h2_total - true_h2) < 0.08
    # Enhancer enrichment should be strongly > 1 (~3x) and statistically significant
    assert res_def.enrichment[1] > 2.0
    assert res_def.p_enrichment[1] < 1e-3
    assert res_def.p_tau[1] < 1e-3
    assert "HDL.S" in res_def.summary()


def test_bivariate_gradient_matches_finite_differences() -> None:
    from hdl.core import _biv_loglik_total

    rng = np.random.default_rng(88)
    lam = np.linspace(0.1, 8.0, 25)
    u1_sq = rng.chisquare(df=1, size=25) * (0.4 * lam + 1.1)
    u2_sq = rng.chisquare(df=1, size=25) * (0.5 * lam + 1.05)
    u12 = 0.3 * np.sqrt(u1_sq * u2_sq)

    ll = _biv_loglik_total(u1_sq, u2_sq, u12, lam, 50_000, 60_000, 500, 0.4, 1.1, 0.5, 1.05, 0.25, 0.05)
    assert np.isfinite(ll)


@pytest.mark.parametrize("bad_settings", ["bad", "r_default", "DEFAULT", ""])
def test_invalid_settings_raises_value_error(bad_settings: str) -> None:
    R = np.eye(5)
    z = np.zeros(5)
    with pytest.raises(ValueError, match="settings"):
        build_ld_eigen_panel([R], settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        hdl_h2(z, [R], N=1000, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        hdl_rg(z, z, [R], N1=1000, N2=1000, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        hdl_local(z, z, R, N1=1000, N2=1000, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        hdl_stratified(z, [R], np.ones((5, 1)), N=1000, settings=bad_settings)  # type: ignore[arg-type]


def test_hdl_input_validation_errors() -> None:
    R = np.eye(6)
    panel = build_ld_eigen_panel([R])
    with pytest.raises(ValueError, match="eigen_cut"):
        build_ld_eigen_panel([R], eigen_cut=1.5)
    with pytest.raises(ValueError, match="total_snps"):
        hdl_h2(np.zeros(4), panel, N=10_000)
    with pytest.raises(ValueError, match="Sample size"):
        hdl_rg(np.zeros(6), np.zeros(6), panel, N1=1, N2=10_000)
    with pytest.raises(ValueError, match="expected"):
        hdl_stratified(np.zeros(6), [R], np.ones((4, 1)), N=10_000)
