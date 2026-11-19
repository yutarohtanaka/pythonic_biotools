"""Unit tests for the cismr correlated drug-target cis-MR package."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cismr import (
    CisMR,
    mr_genius,
    mr_genius_summary,
    mr_mvpcgmm,
    mr_pcgmm,
    mr_pcivw,
    mr_spi,
    prepare_ld_matrix,
)


def _simulate_cis_summary(
    rng: np.random.Generator,
    p: int = 18,
    theta: float = 0.45,
    rho_ld: float = 0.75,
    nx: int = 40_000,
    ny: int = 60_000,
    pleiotropy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate realistic two-sample correlated cis-locus summary statistics."""
    idx = np.arange(p)
    R = rho_ld ** np.abs(idx[:, None] - idx[None, :])
    causal_gamma = np.zeros(p)
    causal_gamma[[1, p // 2, p - 2]] = [0.09, -0.07, 0.08]
    marg_bx = R @ causal_gamma
    marg_by = theta * marg_bx
    if pleiotropy is not None:
        marg_by = marg_by + R @ pleiotropy

    sx = np.full(p, 1.0 / np.sqrt(nx))
    sy = np.full(p, 1.0 / np.sqrt(ny))
    Sigma_x = (R * sx[:, None]) * sx[None, :]
    Sigma_y = (R * sy[:, None]) * sy[None, :]
    bx = rng.multivariate_normal(marg_bx, Sigma_x)
    by = rng.multivariate_normal(marg_by, Sigma_y)
    return bx, sx, by, sy, R


def test_mr_pcgmm_causal_recovery_both_settings() -> None:
    rng = np.random.default_rng(101)
    bx, sx, by, sy, R = _simulate_cis_summary(rng, p=20, theta=0.5, rho_ld=0.8)

    res_def = mr_pcgmm(bx, sx, by, sy, R, nx=40_000, thres=0.99, settings="default")
    res_r = mr_pcgmm(bx, sx, by, sy, R, nx=40_000, thres=0.99, settings="R_default")

    assert abs(res_def.estimate - 0.5) < 0.08
    assert abs(res_r.estimate - 0.5) < 0.08
    assert np.isclose(res_def.estimate, res_r.estimate, atol=0.02)
    assert res_def.ci_low < 0.5 < res_def.ci_high
    assert res_r.ci_low < 0.5 < res_r.ci_high
    assert res_def.f_stat > 20.0
    assert 1 <= res_def.n_pcs < 20
    assert res_def.j_pval > 0.01
    assert "PC-GMM" in res_def.summary()
    assert res_def.to_dataframe().shape == (1, 15)


def test_mr_pcgmm_null_effect_and_exact_r_default_2step() -> None:
    rng = np.random.default_rng(202)
    bx, sx, by, sy, R = _simulate_cis_summary(rng, p=15, theta=0.0, rho_ld=0.7)

    res_r = mr_pcgmm(bx, sx, by, sy, R, nx=30_000, thres=0.99, robust=True, settings="R_default")
    res_r_nonrob = mr_pcgmm(
        bx, sx, by, sy, R, nx=30_000, thres=0.99, robust=False, settings="R_default"
    )
    assert abs(res_r.estimate) < 0.10
    assert res_r.pval > 0.05
    # Verify t_{K-1} vs normal critical ratio in R_default
    crit_t = stats.t.ppf(0.975, df=res_r.j_df)
    assert np.isclose(res_r.ci_high - res_r.estimate, crit_t * res_r.se, rtol=1e-10)
    crit_z = stats.norm.ppf(0.975)
    assert np.isclose(
        res_r_nonrob.ci_high - res_r_nonrob.estimate, crit_z * res_r_nonrob.se, rtol=1e-10
    )


def test_mr_pcgmm_overidentification_j_detects_pleiotropy() -> None:
    rng = np.random.default_rng(303)
    pleio = np.zeros(18)
    pleio[[1, 10, 15]] = [0.06, -0.07, 0.05]
    bx, sx, by, sy, R = _simulate_cis_summary(
        rng, p=18, theta=0.4, rho_ld=0.65, pleiotropy=pleio
    )

    res_rob = mr_pcgmm(bx, sx, by, sy, R, nx=50_000, robust=True, settings="default")
    res_nonrob = mr_pcgmm(bx, sx, by, sy, R, nx=50_000, robust=False, settings="default")

    assert res_rob.j_df >= 2
    assert res_rob.j_stat > res_rob.j_df * 3.0
    assert res_rob.j_pval < 1e-4
    assert res_rob.se > res_nonrob.se * 1.5


def test_prepare_ld_matrix_psd_projection_handles_indefinite_ld() -> None:
    rng = np.random.default_rng(404)
    bx, sx, by, sy, R = _simulate_cis_summary(rng, p=12, theta=0.35, rho_ld=0.8)
    # Perturb off-diagonals so R has a small negative eigenvalue (common with noisy reference LD)
    noise = rng.normal(0.0, 0.08, size=R.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    R_noisy = R + noise
    R_noisy[0, 1] = R_noisy[1, 0] = 0.99
    R_noisy[1, 2] = R_noisy[2, 1] = 0.99
    R_noisy[0, 2] = R_noisy[2, 0] = -0.20  # Violates triangle / PSD condition

    assert np.min(np.linalg.eigvalsh(R_noisy)) < 0.0
    R_clean = prepare_ld_matrix(R_noisy, 12, settings="default")
    assert np.allclose(np.diag(R_clean), 1.0)
    assert np.min(np.linalg.eigvalsh(R_clean)) > 0.0

    res = mr_pcgmm(bx, sx, by, sy, R_noisy, nx=40_000, settings="default")
    assert np.isfinite(res.estimate) and res.se > 0.0


def test_mr_mvpcgmm_recovers_multiple_cis_exposures() -> None:
    rng = np.random.default_rng(505)
    p, m = 24, 2
    idx = np.arange(p)
    R = 0.7 ** np.abs(idx[:, None] - idx[None, :])

    gamma1 = np.zeros(p)
    gamma2 = np.zeros(p)
    gamma1[[2, 8, 18]] = [0.10, -0.08, 0.07]
    gamma2[[4, 13, 21]] = [-0.09, 0.11, 0.08]

    theta_true = np.array([0.40, -0.30])
    marg_Bx = np.column_stack([R @ gamma1, R @ gamma2])
    marg_by = marg_Bx @ theta_true

    nx, ny = 50_000, 60_000
    sx_col = np.full(p, 1.0 / np.sqrt(nx))
    sy = np.full(p, 1.0 / np.sqrt(ny))
    Sigma_x = (R * sx_col[:, None]) * sx_col[None, :]
    Sigma_y = (R * sy[:, None]) * sy[None, :]

    Bx = np.column_stack(
        [
            rng.multivariate_normal(marg_Bx[:, 0], Sigma_x),
            rng.multivariate_normal(marg_Bx[:, 1], Sigma_x),
        ]
    )
    Sx = np.column_stack([sx_col, sx_col])
    by = rng.multivariate_normal(marg_by, Sigma_y)

    res_def = mr_mvpcgmm(
        Bx, Sx, by, sy, R, nx=nx, exposure_names=["IsoformA", "IsoformB"], settings="default"
    )
    res_r = mr_mvpcgmm(Bx, Sx, by, sy, R, nx=nx, settings="R_default")

    assert np.allclose(res_def.estimates, theta_true, atol=0.08)
    assert np.allclose(res_r.estimates, theta_true, atol=0.08)
    assert np.all(res_def.cond_f_stats > 15.0)
    assert res_def.j_pval > 0.01
    df = res_def.to_dataframe()
    assert list(df["exposure"]) == ["IsoformA", "IsoformB"]


def test_mr_mvpcgmm_conditional_f_detects_collinear_exposures() -> None:
    rng = np.random.default_rng(606)
    p = 16
    R = 0.6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    g1 = np.zeros(p)
    g1[[2, 9, 13]] = [0.12, -0.10, 0.09]
    marg1 = R @ g1
    # Exposure 2 is genetically proportional to Exposure 1 (rank-1 genetic effect)
    marg2 = 0.8 * marg1
    sx_col = np.full(p, 0.005)
    Sx = np.column_stack([sx_col, sx_col])
    Bx = np.column_stack(
        [
            marg1 + rng.normal(0.0, 0.005, size=p),
            marg2 + rng.normal(0.0, 0.005, size=p),
        ]
    )
    by = 0.5 * marg1 + rng.normal(0.0, 0.005, size=p)
    sy = np.full(p, 0.005)

    res = mr_mvpcgmm(Bx, Sx, by, sy, R, nx=40_000, settings="default")
    # Conditional Sanderson-Windmeijer F must flag weak conditional identification (< 5)
    assert np.all(res.cond_f_stats < 5.0)


def test_mr_spi_plurality_valid_selection_with_pleiotropic_outliers() -> None:
    rng = np.random.default_rng(707)
    p = 12
    R = 0.4 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    theta_true = 0.60
    bx = np.array([0.08, -0.09, 0.11, 0.07, -0.10, 0.09, 0.08, -0.07, 0.10, 0.09, -0.08, 0.10])
    sx = np.full(p, 0.006)
    sy = np.full(p, 0.006)

    # First 8 SNPs are valid; last 4 SNPs are invalid pleiotropic outliers
    pleio = np.zeros(p)
    pleio[8:] = np.array([0.08, -0.09, 0.07, 0.10])
    Sigma_y = (R * sy[:, None]) * sy[None, :]
    by = rng.multivariate_normal(theta_true * bx + pleio, Sigma_y)

    res_def = mr_spi(bx, sx, by, sy, ld_R=R, n=50_000, alpha=0.05, settings="default")
    res_r = mr_spi(bx, sx, by, sy, ld_R=R, n=50_000, alpha=0.05, settings="R_default")

    assert abs(res_def.estimate - theta_true) < 0.06
    assert abs(res_r.estimate - theta_true) < 0.06
    assert set(res_def.selected_indices.tolist()) == set(range(8))
    assert set(res_def.invalid_indices.tolist()) == {8, 9, 10, 11}
    assert res_def.ci_low < theta_true < res_def.ci_high


def test_mr_spi_max_clique_disambiguates_bridge_snp() -> None:
    # A noisy bridge SNP (larger sy) is compatible with both a 4-SNP valid cluster (0.50)
    # and a 3-SNP pleiotropic cluster (1.00), whereas the two clusters are mutually incompatible.
    bx = np.full(8, 0.10)
    sx = np.full(8, 0.002)
    sy = np.array([0.003, 0.003, 0.003, 0.003, 0.025, 0.003, 0.003, 0.003])
    ratios = np.array([0.49, 0.50, 0.50, 0.51, 0.75, 0.99, 1.00, 1.01])
    by = ratios * bx

    res_def = mr_spi(bx, sx, by, sy, ld_R=None, alpha=0.05, settings="default")
    res_r = mr_spi(bx, sx, by, sy, ld_R=None, alpha=0.05, settings="R_default")

    # In default mode, every pair in selected_indices must be mutually compatible (a clique)
    sub_vote = res_def.voting_matrix[np.ix_(res_def.selected_indices, res_def.selected_indices)]
    assert np.all(sub_vote)
    assert abs(res_def.estimate - 0.50) < 0.02
    assert abs(res_r.estimate - 0.50) > 0.15
    assert np.isfinite(res_r.estimate)


def test_mr_genius_individual_identifies_effect_when_all_snps_pleiotropic() -> None:
    rng = np.random.default_rng(808)
    n = 6_000
    G = rng.binomial(2, [0.30, 0.40, 0.25], size=(n, 3)).astype(float)
    C = rng.normal(0.0, 1.0, size=(n, 2))
    U = rng.normal(0.0, 1.0, size=n)  # Unmeasured confounder!

    # Strong heteroskedasticity in X driven by G, PLUS confounding U
    sd_x = 0.6 + 0.45 * G[:, 0] + 0.35 * G[:, 1] + 0.30 * G[:, 2]
    eps_x = sd_x * rng.normal(0.0, 1.0, size=n)
    X = 0.3 * G[:, 0] - 0.2 * G[:, 1] + 0.25 * G[:, 2] + 0.5 * C[:, 0] + 0.6 * U + eps_x

    # ALL 3 SNPs have direct additive pleiotropic effects on Y!
    theta_true = 0.50
    direct_pleio = 0.8 * G[:, 0] - 0.7 * G[:, 1] + 0.6 * G[:, 2]
    Y = theta_true * X + direct_pleio - 0.4 * C[:, 1] + 0.6 * U + rng.normal(0.0, 1.0, size=n)

    res_def = mr_genius(G, X, Y, covariates=C, settings="default")
    res_r = mr_genius(G, X, Y, covariates=C, settings="R_default")

    assert abs(res_def.estimate - theta_true) < 0.07
    assert abs(res_r.estimate - theta_true) < 0.08
    assert res_def.ci_low < theta_true < res_def.ci_high
    assert res_def.het_f_stat > 25.0
    assert res_def.het_f_pval < 1e-10


def test_mr_genius_default_quadratic_saturation_handles_dominance() -> None:
    rng = np.random.default_rng(909)
    n = 5_000
    g = rng.binomial(2, 0.35, size=n).astype(float)
    # Quadratic dominance in E[X|G] PLUS true heteroskedasticity in Var(X|G)
    sd_x = 0.5 + 0.5 * g
    X = 0.4 * g + 0.35 * (g**2) + sd_x * rng.normal(0.0, 1.0, size=n)
    theta_true = -0.35
    Y = theta_true * X + 0.9 * g + rng.normal(0.0, 1.0, size=n)

    res_def = mr_genius(g, X, Y, settings="default")
    assert abs(res_def.estimate - theta_true) < 0.07
    assert res_def.ci_low < theta_true < res_def.ci_high


def test_mr_genius_summary_vqtl_and_allele_flip_invariance() -> None:
    rng = np.random.default_rng(1010)
    p = 10
    R = 0.5 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    theta_true = 0.45

    bx = np.array([0.12, -0.10, 0.09, 0.11, -0.08, 0.07, 0.05, -0.06, 0.04, -0.05])
    # First 6 SNPs are genuine exposure vQTLs; last 4 have zero vQTL effect but strong pleiotropy!
    v_gx = np.array([0.08, -0.07, 0.06, 0.09, -0.06, 0.05, 0.0, 0.0, 0.0, 0.0])
    se_vgx = np.full(p, 0.008)
    sx = np.full(p, 0.005)
    sy = np.full(p, 0.005)

    # Pleiotropy orthogonal to v_gx in LD metric: R @ v_gx = 0 on pleiotropic direction
    pleio = np.zeros(p)
    pleio[6:] = [0.09, -0.08, 0.07, -0.09]
    # Orthogonalize pleio wrt R^-1 v_gx so v_gx^T R^-1 pleio = 0
    w_v = np.linalg.solve(R, v_gx)
    pleio = pleio - (float(w_v @ pleio) / float(w_v @ w_v)) * w_v
    by = theta_true * bx + pleio + rng.normal(0.0, 0.002, size=p)

    res_def = mr_genius_summary(bx, sx, by, sy, v_gx, se_vgx=se_vgx, ld_R=R, settings="default")
    res_r = mr_genius_summary(bx, sx, by, sy, v_gx, se_vgx=se_vgx, ld_R=R, settings="R_default")

    assert abs(res_def.estimate - theta_true) < 0.05
    assert abs(res_r.estimate - theta_true) < 0.05

    # Flip alleles on odd-indexed SNPs: bx, by, v_gx, and R signs flip; estimate must be invariant
    signs = np.where(np.arange(p) % 2 == 0, 1.0, -1.0)
    R_flip = (R * signs[:, None]) * signs[None, :]
    res_flip = mr_genius_summary(
        bx * signs, sx, by * signs, sy, v_gx * signs, se_vgx=se_vgx, ld_R=R_flip, settings="default"
    )
    assert np.isclose(res_def.estimate, res_flip.estimate, atol=1e-10)
    assert np.isclose(res_def.se, res_flip.se, atol=1e-10)


def test_cismr_unified_sensitivity_suite() -> None:
    rng = np.random.default_rng(1111)
    bx, sx, by, sy, R = _simulate_cis_summary(rng, p=14, theta=0.35, rho_ld=0.6)
    v_gx = 0.6 * bx
    se_vgx = np.full(14, 0.008)

    n = 2_000
    g_ind = rng.binomial(2, 0.3, size=(n, 2)).astype(float)
    x_ind = 0.3 * g_ind[:, 0] + (0.5 + 0.4 * g_ind[:, 0] + 0.3 * g_ind[:, 1]) * rng.normal(size=n)
    y_ind = 0.35 * x_ind + rng.normal(size=n)

    runner = CisMR(thres=0.99, robust=True, settings="default")
    report = runner(
        bx, sx, by, sy, ld_R=R, nx=40_000, v_gx=v_gx, se_vgx=se_vgx, G=g_ind, X=x_ind, Y=y_ind
    )
    df = report.to_dataframe()
    assert df.shape[0] == 5
    assert set(df["method"]) == {
        "PC-GMM",
        "PC-IVW",
        "MR-SPI",
        "MR-GENIUS (summary_vqtl)",
        "MR-GENIUS (individual)",
    }
    assert np.all(np.abs(df["estimate"] - 0.35) < 0.12)
    assert "CisMR Drug-Target Sensitivity Suite" in report.summary()


@pytest.mark.parametrize("bad_settings", ["invalid", "r_default", "DEFAULT", ""])
def test_invalid_settings_raises_value_error(bad_settings: str) -> None:
    bx = np.array([0.1, 0.12, -0.09])
    sx = np.array([0.01, 0.01, 0.01])
    by = np.array([0.05, 0.06, -0.045])
    sy = np.array([0.01, 0.01, 0.01])
    R = np.eye(3)

    with pytest.raises(ValueError, match="settings"):
        mr_pcgmm(bx, sx, by, sy, R, nx=10_000, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        mr_pcivw(bx, sx, by, sy, R, nx=10_000, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        mr_mvpcgmm(
            np.column_stack([bx, bx[::-1]]),
            np.column_stack([sx, sx]),
            by,
            sy,
            R,
            nx=10_000,
            settings=bad_settings,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="settings"):
        mr_spi(bx, sx, by, sy, ld_R=R, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        mr_genius(np.ones((20, 1)), np.ones(20), np.ones(20), settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        mr_genius_summary(bx, sx, by, sy, bx, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        CisMR(settings=bad_settings)  # type: ignore[arg-type]


def test_input_validation_edge_cases() -> None:
    R = np.eye(3)
    with pytest.raises(ValueError, match="strictly positive"):
        mr_pcgmm([0.1, 0.2, 0.1], [0.0, 0.01, 0.01], [0.1, 0.2, 0.1], [0.01, 0.01, 0.01], R, nx=1000)
    with pytest.raises(ValueError, match="exceed number of exposures"):
        mr_mvpcgmm(np.ones((2, 2)), np.ones((2, 2)) * 0.1, [0.1, 0.2], [0.1, 0.1], np.eye(2), nx=1000)
    with pytest.raises(ValueError, match="strength screening"):
        mr_spi([0.001, -0.001], [0.1, 0.1], [0.01, -0.01], [0.1, 0.1])
    with pytest.raises(ValueError, match="numerically zero"):
        g = np.tile([0.0, 1.0], 25)
        x = 0.5 * g + np.tile([1.0, -1.0, 1.0, -1.0], 13)[:50]  # Homoskedastic residual
        y = 0.3 * x
        mr_genius(g, x, y)
