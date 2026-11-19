"""Comprehensive unit tests for the lhcmr package."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

from lhcmr import (
    LHCMR,
    LHCMRResult,
    PARAM_NAMES,
    bivariate_ldsc,
    lhc_mr,
)
from lhcmr.likelihood import (
    canonicalize_reflection,
    loglik_only,
    neg_loglik_and_grad,
    prepare_lhc_data,
    swap_factor_x_with_u,
)


def _simulate_lhcmr_sumstats(
    m_snps: int = 4000,
    nx: float = 100_000.0,
    ny: float = 100_000.0,
    alpha_xy: float = 0.30,
    alpha_yx: float = 0.0,
    t_x: float = 0.18,
    t_y: float = 0.15,
    h2_x: float = 0.30,
    h2_y: float = 0.25,
    pi_x: float = 0.06,
    pi_y: float = 0.06,
    pi_u: float = 0.015,
    i_x: float = 1.02,
    i_y: float = 1.03,
    i_xy: float = 0.25,
    constant_ld: bool = False,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if constant_ld:
        ld = np.ones(m_snps, dtype=float)
    else:
        ld = 1.0 + rng.gamma(shape=2.0, scale=1.5, size=m_snps)

    c_x = rng.binomial(1, pi_x, size=m_snps).astype(float)
    c_y = rng.binomial(1, pi_y, size=m_snps).astype(float)
    c_u = rng.binomial(1, pi_u, size=m_snps).astype(float)

    gx = rng.normal(0.0, 1.0, size=m_snps) * np.sqrt(
        c_x * ld * (h2_x / (m_snps * pi_x))
    )
    gy = rng.normal(0.0, 1.0, size=m_snps) * np.sqrt(
        c_y * ld * (h2_y / (m_snps * pi_y))
    )
    gu = rng.normal(0.0, 1.0, size=m_snps) * np.sqrt(
        c_u * ld * (1.0 / (m_snps * pi_u))
    )

    eta_x = gx + t_x * gu
    eta_y = gy + t_y * gu
    delta = 1.0 - alpha_xy * alpha_yx
    true_bx = (eta_x + alpha_yx * eta_y) / delta
    true_by = (alpha_xy * eta_x + eta_y) / delta

    cov_e = np.array(
        [
            [i_x / nx, i_xy / np.sqrt(nx * ny)],
            [i_xy / np.sqrt(nx * ny), i_y / ny],
        ],
        dtype=float,
    )
    eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_e, size=m_snps)
    bx = true_bx + eps[:, 0]
    by = true_by + eps[:, 1]
    sx = np.full(m_snps, 1.0 / np.sqrt(nx), dtype=float)
    sy = np.full(m_snps, 1.0 / np.sqrt(ny), dtype=float)
    return bx, sx, by, sy, ld


def test_stage1_bivariate_ldsc_recovers_intercepts_and_heritabilities() -> None:
    rng = np.random.default_rng(7)
    n_sim = 30_000
    m_ref = 200_000.0
    nx = 120_000.0
    ny = 120_000.0
    ld = 1.0 + rng.gamma(shape=2.0, scale=1.5, size=n_sim)
    gx = rng.normal(0.0, np.sqrt(ld * (0.30 / m_ref)), size=n_sim)
    gy = 0.25 * gx + rng.normal(0.0, np.sqrt(ld * (0.25 / m_ref)), size=n_sim)
    cov_e = np.array([[1.05 / nx, 0.32 / np.sqrt(nx * ny)], [0.32 / np.sqrt(nx * ny), 1.08 / ny]])
    eps = rng.multivariate_normal([0.0, 0.0], cov_e, size=n_sim)
    zx = (gx + eps[:, 0]) * np.sqrt(nx)
    zy = (gy + eps[:, 1]) * np.sqrt(ny)

    res = bivariate_ldsc(zx, zy, ld, nx, ny, m_snps=m_ref, settings="default")
    assert abs(res.i_x - 1.05) < 0.08
    assert abs(res.i_y - 1.08) < 0.08
    assert abs(res.i_xy - 0.32) < 0.06
    assert abs(res.h2_x - 0.30) < 0.08
    assert res.g_xy > 0.03


def test_analytical_gradient_matches_finite_differences() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(m_snps=300, seed=11)
    nx_arr = np.full(300, 100_000.0)
    ny_arr = np.full(300, 100_000.0)
    data = prepare_lhc_data(bx, by, nx_arr, ny_arr, ld, m_eff=300.0)
    p0 = np.array(
        [0.22, -0.08, 0.16, 0.12, 0.28, 0.24, 0.06, 0.05, 0.02, 1.02, 1.03, 0.20],
        dtype=float,
    )
    _, g_analytic = neg_loglik_and_grad(p0, lambda x: x, data)
    g_central = np.empty_like(p0)
    eps = 1e-6
    for k in range(p0.size):
        pp = p0.copy()
        pm = p0.copy()
        pp[k] += eps
        pm[k] -= eps
        fp, _ = neg_loglik_and_grad(pp, lambda x: x, data)
        fm, _ = neg_loglik_and_grad(pm, lambda x: x, data)
        g_central[k] = (fp - fm) / (2.0 * eps)
    rel_err = float(np.linalg.norm(g_analytic - g_central) / np.linalg.norm(g_central))
    assert rel_err < 5e-8
    assert optimize.check_grad(
        lambda v: neg_loglik_and_grad(v, lambda x: x, data)[0] / 300.0,
        lambda v: neg_loglik_and_grad(v, lambda x: x, data)[1] / 300.0,
        p0,
        epsilon=1e-6,
    ) < 1e-4


def test_unidirectional_forward_causal_recovery() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=4000,
        nx=120_000,
        ny=120_000,
        alpha_xy=0.35,
        alpha_yx=0.0,
        t_x=0.15,
        t_y=0.12,
        seed=21,
    )
    res = lhc_mr(
        bx, sx, by, sy, ld, nx=120_000, ny=120_000, n_starts=4, n_blocks=15, seed=21
    )
    assert isinstance(res, LHCMRResult)
    assert abs(res.alpha_xy - 0.35) < 0.08
    assert abs(res.alpha_yx) < 0.08
    assert res.alpha_xy_pval < 1e-3
    assert res.lrt["unidirectional_xy_vs_confounder_only"].p_value < 1e-3


def test_reverse_causal_effect_recovery() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=4000,
        nx=120_000,
        ny=120_000,
        alpha_xy=0.0,
        alpha_yx=-0.32,
        t_x=0.14,
        t_y=0.10,
        seed=33,
    )
    res = lhc_mr(
        bx, sx, by, sy, ld, nx=120_000, ny=120_000, n_starts=4, n_blocks=15, seed=33
    )
    assert abs(res.alpha_yx - (-0.32)) < 0.08
    assert abs(res.alpha_xy) < 0.08
    assert res.alpha_yx_pval < 1e-3


def test_bidirectional_causal_effects_recovery() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=4500,
        nx=140_000,
        ny=140_000,
        alpha_xy=0.30,
        alpha_yx=-0.22,
        t_x=0.14,
        t_y=0.12,
        seed=44,
    )
    res = lhc_mr(
        bx, sx, by, sy, ld, nx=140_000, ny=140_000, n_starts=5, n_blocks=15, seed=44
    )
    assert abs(res.alpha_xy - 0.30) < 0.09
    assert abs(res.alpha_yx - (-0.22)) < 0.09
    assert abs(res.alpha_xy * res.alpha_yx) < 0.95
    assert res.lrt["bidirectional_vs_confounder_only"].p_value < 1e-4


def test_pure_heritable_confounder_avoids_spurious_causal_effect() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=4500,
        nx=130_000,
        ny=130_000,
        alpha_xy=0.0,
        alpha_yx=0.0,
        t_x=0.25,
        t_y=0.25,
        pi_x=0.07,
        pi_y=0.07,
        pi_u=0.012,
        seed=55,
    )
    res = lhc_mr(
        bx, sx, by, sy, ld, nx=130_000, ny=130_000, n_starts=5, n_blocks=15, seed=55
    )
    assert abs(res.alpha_xy) < 0.08
    assert abs(res.alpha_yx) < 0.08
    assert res.t_x >= 0.0
    assert res.t_x * res.t_y > 0.02
    assert res.lrt["bidirectional_vs_no_confounder"].p_value < 1e-3


def test_reflection_swap_preserves_exact_loglik_and_canonicalizes() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(m_snps=400, seed=66)
    data = prepare_lhc_data(
        bx, by, np.full(400, 100_000.0), np.full(400, 100_000.0), ld, m_eff=400.0
    )
    p_orig = np.array(
        [0.30, -0.05, 0.18, 0.12, 0.30, 0.25, 0.06, 0.06, 0.015, 1.0, 1.0, 0.1],
        dtype=float,
    )
    p_swapped = swap_factor_x_with_u(p_orig)
    assert p_swapped is not None
    ll_orig = loglik_only(data, p_orig)
    ll_swap = loglik_only(data, p_swapped)
    assert np.isclose(ll_orig, ll_swap, atol=1e-8)

    p_canon, was_swapped = canonicalize_reflection(p_swapped)
    assert was_swapped is True
    assert np.allclose(p_canon[:9], p_orig[:9], atol=1e-6)


def test_r_default_outlier_filtering_and_fixed_intercepts() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=3000,
        nx=8_000,
        ny=8_000,
        alpha_xy=0.28,
        alpha_yx=0.0,
        t_x=0.0,
        t_y=0.0,
        h2_x=0.25,
        h2_y=0.20,
        pi_x=0.40,
        pi_y=0.40,
        i_x=1.04,
        i_y=1.06,
        seed=77,
    )
    # Ensure background SNPs are below 80, then inject 5 extreme outlier SNPs with z^2 = 225 > 80
    bx = np.clip(bx, -8.0 * sx, 8.0 * sx)
    by = np.clip(by, -8.0 * sy, 8.0 * sy)
    bx_out = bx.copy()
    bx_out[:5] = 15.0 * sx[:5]  # z^2 = 225 > 80

    res_default = lhc_mr(
        bx_out,
        sx,
        by,
        sy,
        ld,
        nx=8_000,
        ny=8_000,
        n_starts=3,
        n_blocks=10,
        settings="default",
    )
    res_r = lhc_mr(
        bx_out,
        sx,
        by,
        sy,
        ld,
        nx=8_000,
        ny=8_000,
        n_starts=3,
        n_blocks=10,
        settings="R_default",
    )
    assert res_default.n_snps == 3000
    assert res_r.n_snps == 2995
    assert res_r.stage1_ldsc.n_outliers_removed == 5
    # In R_default, Stage 2 keeps i_x and i_y fixed at Stage-1 LDSC estimates
    assert np.isclose(res_r.i_x, res_r.stage1_ldsc.i_x, atol=1e-12)
    assert np.isclose(res_r.i_y, res_r.stage1_ldsc.i_y, atol=1e-12)
    assert abs(res_r.alpha_xy - 0.28) < 0.14


def test_constant_ld_scores_independent_snps_supported() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=3000,
        nx=100_000,
        ny=100_000,
        alpha_xy=0.26,
        alpha_yx=0.0,
        t_x=0.12,
        t_y=0.10,
        constant_ld=True,
        seed=88,
    )
    res = lhc_mr(
        bx, sx, by, sy, ld, nx=100_000, ny=100_000, n_starts=3, n_blocks=10, seed=88
    )
    assert abs(res.alpha_xy - 0.26) < 0.09


def test_per_allele_unstandardized_sumstats_invariance() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=3000, nx=100_000, ny=100_000, alpha_xy=0.30, seed=99
    )
    rng = np.random.default_rng(99)
    maf = rng.uniform(0.05, 0.50, size=3000)
    scale = 1.0 / np.sqrt(2.0 * maf * (1.0 - maf))

    res_std = lhc_mr(
        bx, sx, by, sy, ld, nx=100_000, ny=100_000, n_starts=3, n_blocks=10, seed=99
    )
    res_allele = lhc_mr(
        bx * scale,
        sx * scale,
        by * scale,
        sy * scale,
        ld,
        nx=100_000,
        ny=100_000,
        n_starts=3,
        n_blocks=10,
        seed=99,
    )
    assert np.isclose(res_std.alpha_xy, res_allele.alpha_xy, atol=1e-4)
    assert np.isclose(res_std.log_likelihood, res_allele.log_likelihood, atol=1e-2)


def test_lhcmr_class_interface_and_dataframe_exports() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(
        m_snps=2000, nx=100_000, ny=100_000, alpha_xy=0.25, seed=101
    )
    model = LHCMR(n_starts=3, n_blocks=10, seed=101, settings="default")
    res = model.fit(bx, sx, by, sy, ld, nx=100_000, ny=100_000)
    assert model.result_ is res

    df = res.to_dataframe()
    assert list(df["parameter"]) == list(PARAM_NAMES)
    assert {"estimate", "se", "jackknife_se", "hessian_se", "z_score", "p_value"}.issubset(
        df.columns
    )
    assert np.all(df["se"] > 0.0)
    assert np.all((df["p_value"] >= 0.0) & (df["p_value"] <= 1.0))

    lrt_df = res.lrt_dataframe()
    assert len(lrt_df) == 6
    assert np.all(lrt_df["stat"] >= 0.0)
    assert "LHC-MR Summary" in res.summary()


def test_invalid_settings_raises_value_error() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(m_snps=100, seed=1)
    with pytest.raises(ValueError, match="Unknown settings"):
        lhc_mr(bx, sx, by, sy, ld, nx=50_000, ny=50_000, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        LHCMR(settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        bivariate_ldsc(bx / sx, by / sy, ld, 50_000, 50_000, settings="nope")  # type: ignore[arg-type]


def test_input_validation_errors() -> None:
    bx, sx, by, sy, ld = _simulate_lhcmr_sumstats(m_snps=50, seed=2)
    with pytest.raises(ValueError, match="At least 30 SNPs"):
        lhc_mr(bx[:20], sx[:20], by[:20], sy[:20], ld[:20], nx=50_000, ny=50_000)
    with pytest.raises(ValueError, match="strictly positive"):
        sx_bad = sx.copy()
        sx_bad[0] = 0.0
        lhc_mr(bx, sx_bad, by, sy, ld, nx=50_000, ny=50_000)
    with pytest.raises(ValueError, match="identical 1D shapes"):
        lhc_mr(bx, sx, by[:-1], sy, ld, nx=50_000, ny=50_000)
