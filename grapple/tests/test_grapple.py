"""Comprehensive unit tests for the grapple package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grapple import (
    GRAPPLE,
    GrappleMVMRResult,
    GrappleResult,
    estimate_cor_mat,
    grapple_mvmr,
    grapple_robust_est,
    psi_deriv,
    psi_loss,
    resolve_loss_params,
    rho_loss,
    select_grapple_instruments,
)


def test_loss_derivatives_match_finite_differences() -> None:
    t = np.array([-4.2, -2.1, -0.8, 0.0, 0.6, 1.9, 3.8], dtype=float)
    eps = 1e-6
    for loss_name, k_val in (("l2", 0.0), ("huber", 1.345), ("tukey", 3.0), ("tukey", 4.685)):
        num_psi = (rho_loss(t + eps, loss_name, k_val) - rho_loss(t - eps, loss_name, k_val)) / (
            2.0 * eps
        )
        assert np.allclose(psi_loss(t, loss_name, k_val), num_psi, atol=1e-7)

        t_smooth = np.array([-2.1, -0.8, 0.0, 0.6, 1.9, 3.8], dtype=float)
        num_psip = (
            psi_loss(t_smooth + eps, loss_name, k_val)
            - psi_loss(t_smooth - eps, loss_name, k_val)
        ) / (2.0 * eps)
        assert np.allclose(psi_deriv(t_smooth, loss_name, k_val), num_psip, atol=1e-6)


def test_estimate_cor_mat_recovers_bivariate_and_trivariate_overlap() -> None:
    rng = np.random.default_rng(12)
    true_cor = np.array(
        [
            [1.0, 0.45, -0.30],
            [0.45, 1.0, 0.25],
            [-0.30, 0.25, 1.0],
        ],
        dtype=float,
    )
    z = rng.multivariate_normal([0.0, 0.0, 0.0], true_cor, size=80_000)
    # Contaminate 500 SNPs with strong genetic effects (filtered out by p > 0.5)
    z[:500, 0] += rng.normal(0.0, 8.0, size=500)
    z[:500, 1] += 0.8 * z[:500, 0]

    df = pd.DataFrame(z, columns=["z_x1", "z_x2", "z_y"])
    est_d = estimate_cor_mat(df, p_thres=0.5, settings="default")
    est_r = estimate_cor_mat(df, p_thres=0.5, settings="R_default")

    assert np.allclose(est_d, true_cor, atol=0.05)
    assert np.allclose(est_r, true_cor, atol=0.06)
    assert np.all(np.linalg.eigvalsh(est_d) > 0.0)


def test_select_grapple_instruments_weak_and_ld_pruning() -> None:
    rng = np.random.default_rng(23)
    m = 500
    sx = np.full(m, 0.02)
    bx = rng.normal(0.0, 0.02, size=m)
    # Inject 10 strong/moderate signals in 5 correlated pairs
    ld_R = np.eye(m, dtype=float)
    for idx in range(0, 10, 2):
        bx[idx] = 0.11  # z = 5.5
        bx[idx + 1] = 0.095  # z = 4.75, tagged with r = 0.9
        ld_R[idx, idx + 1] = 0.90
        ld_R[idx + 1, idx] = 0.90
    # Inject 6 weak IVs (z = 4.1 -> p ~ 4.1e-5 <= 1e-4)
    bx[20:26] = 0.082

    by = 0.3 * bx + rng.normal(0.0, 0.02, size=m)
    df = pd.DataFrame(
        {"snp": [f"rs{i}" for i in range(m)], "bx": bx, "sx": sx, "by": by, "sy": sx}
    )

    pruned = select_grapple_instruments(df, ld_R=ld_R, p_thres=1e-4, r2_prune=0.001)
    assert "selection_pvals" in pruned.columns
    assert np.all(pruned["selection_pvals"] <= 1e-4)
    # Only the stronger lead SNP (even index) of each correlated pair should survive
    for idx in range(0, 10, 2):
        assert f"rs{idx}" in set(pruned["snp"])
        assert f"rs{idx + 1}" not in set(pruned["snp"])
    assert len(pruned) >= 11


def test_weak_iv_with_strong_sample_overlap_unbiased_under_null() -> None:
    rng = np.random.default_rng(31)
    m = 250
    sx = np.full(m, 0.02)
    sy = np.full(m, 0.02)
    true_bx = rng.normal(0.0, 0.045, size=m)  # weak instruments (mean F ~ 6)
    cor_xy = 0.65
    cov_e = np.array([[0.02**2, cor_xy * 0.02**2], [cor_xy * 0.02**2, 0.02**2]])
    eps = rng.multivariate_normal([0.0, 0.0], cov_e, size=m)
    bx_hat = true_bx + eps[:, 0]
    by_hat = 0.0 * true_bx + eps[:, 1]  # true beta = 0!

    # Select weak instruments with |z_x| > 2.5 (creating Winner's curse + overlap bias in naive IVW)
    sel = np.abs(bx_hat / sx) > 2.5
    bx_s, sx_s, by_s, sy_s = bx_hat[sel], sx[sel], by_hat[sel], sy[sel]

    ivw_naive = float(np.sum(bx_s * by_s) / np.sum(bx_s * bx_s))
    assert ivw_naive > 0.08  # Naive IVW biased upward by overlap

    res = grapple_robust_est(
        bx_s, sx_s, by_s, sy_s, cor_xy=cor_xy, settings="default"
    )
    assert isinstance(res, GrappleResult)
    assert abs(res.beta_hat) < 0.06
    assert res.p_value > 0.05


def test_weak_iv_causal_effect_and_sandwich_se_calibration() -> None:
    rng = np.random.default_rng(42)
    m = 180
    sx = np.full(m, 0.02)
    sy = np.full(m, 0.025)
    true_bx = rng.normal(0.0, 0.06, size=m)
    cor_xy = 0.30
    cov_e = np.array([[0.02**2, cor_xy * 0.02 * 0.025], [cor_xy * 0.02 * 0.025, 0.025**2]])
    eps = rng.multivariate_normal([0.0, 0.0], cov_e, size=m)

    bx_hat = true_bx + eps[:, 0]
    by_hat = 0.35 * true_bx + eps[:, 1]

    res = grapple_robust_est(bx_hat, sx, by_hat, sy, cor_xy=cor_xy, settings="default")
    assert abs(res.beta_hat - 0.35) < 0.05
    assert 0.01 < res.beta_se < 0.06
    assert res.p_value < 1e-6


def test_multimodal_pleiotropic_pathway_discovery_and_marker_snps() -> None:
    rng = np.random.default_rng(55)
    # Pathway 1: 40 SNPs with causal effect +0.45
    # Pathway 2: 35 SNPs with distinct pleiotropic pathway -0.40
    n1, n2 = 40, 35
    sx = np.full(n1 + n2, 0.015)
    sy = np.full(n1 + n2, 0.015)
    true_bx1 = rng.uniform(0.08, 0.14, size=n1) * rng.choice([-1.0, 1.0], size=n1)
    true_bx2 = rng.uniform(0.08, 0.14, size=n2) * rng.choice([-1.0, 1.0], size=n2)
    true_bx = np.concatenate([true_bx1, true_bx2])
    true_by = np.concatenate([0.45 * true_bx1, -0.40 * true_bx2])

    bx_hat = true_bx + rng.normal(0.0, sx)
    by_hat = true_by + rng.normal(0.0, sy)
    snp_ids = [f"path1_snp{i}" for i in range(n1)] + [f"path2_snp{i}" for i in range(n2)]

    res_r = grapple_robust_est(
        bx_hat, sx, by_hat, sy, snp_ids=snp_ids, diagnosis=True, settings="R_default"
    )
    assert res_r.is_multimodal is True
    assert res_r.n_modes == 2

    mode_betas = sorted([m.beta_hat for m in res_r.modes])
    assert abs(mode_betas[0] - (-0.40)) < 0.06
    assert abs(mode_betas[1] - 0.45) < 0.06

    # Verify marker SNPs accurately map to their respective pathways
    pos_mode = max(res_r.modes, key=lambda m: m.beta_hat)
    neg_mode = min(res_r.modes, key=lambda m: m.beta_hat)
    assert all(s.startswith("path1_") for s in pos_mode.marker_snps[:5])
    assert all(s.startswith("path2_") for s in neg_mode.marker_snps[:5])

    # With diagnosis=False, only the single global maximum mode is returned
    res_single = grapple_robust_est(
        bx_hat, sx, by_hat, sy, snp_ids=snp_ids, diagnosis=False, settings="R_default"
    )
    assert res_single.n_modes == 1
    assert res_single.is_multimodal is False


def test_default_vs_r_default_tukey_tuning_and_prominence() -> None:
    loss_d, k_d, mode_d = resolve_loss_params("tukey", None, settings="default")
    loss_r, k_r, mode_r = resolve_loss_params("tukey", None, settings="R_default")
    assert (loss_d, k_d, mode_d) == ("tukey", 4.685, "default")
    assert (loss_r, k_r, mode_r) == ("tukey", 3.0, "R_default")

    rng = np.random.default_rng(64)
    # Pathways (+0.35 and -0.28) separated by ~3.5 residual SDs where k=3.0 resolves both modes
    bx = np.full(70, 0.11) * rng.choice([-1.0, 1.0], size=70)
    sx = np.full(70, 0.015)
    sy = np.full(70, 0.015)
    by = np.concatenate([0.35 * bx[:35], -0.28 * bx[35:]]) + rng.normal(0.0, 0.012, size=70)

    res_d = grapple_robust_est(bx, sx, by, sy, settings="default")
    res_r = grapple_robust_est(bx, sx, by, sy, settings="R_default")
    assert res_d.k == 4.685
    assert res_d.prominence_threshold == 0.5
    assert res_r.k == 3.0
    assert res_r.prominence_threshold == 1.0
    assert res_r.n_modes == 2


def test_tukey_and_huber_resist_directional_pleiotropic_outliers() -> None:
    rng = np.random.default_rng(77)
    m = 100
    sx = np.full(m, 0.015)
    sy = np.full(m, 0.015)
    bx = rng.normal(0.0, 0.08, size=m)
    by = 0.25 * bx + rng.normal(0.0, 0.015, size=m)
    # Corrupt 8 SNPs with massive directional pleiotropy
    by[:8] += 0.35

    res_l2 = grapple_robust_est(bx, sx, by, sy, loss_function="l2", diagnosis=False)
    res_huber = grapple_robust_est(bx, sx, by, sy, loss_function="huber", diagnosis=False)
    res_tukey = grapple_robust_est(bx, sx, by, sy, loss_function="tukey", diagnosis=False)

    assert abs(res_tukey.beta_hat - 0.25) < 0.03
    assert abs(res_huber.beta_hat - 0.25) < abs(res_l2.beta_hat - 0.25)


def test_grapple_mvmr_recovers_direct_effects_with_overlap_and_outliers() -> None:
    rng = np.random.default_rng(88)
    m, p = 160, 3
    beta_true = np.array([0.35, -0.25, 0.0], dtype=float)
    sx = np.full((m, p), 0.02)
    sy = np.full(m, 0.02)

    true_bx = rng.normal(0.0, 0.08, size=(m, p))
    # Correlated exposures too
    true_bx[:, 1] += 0.35 * true_bx[:, 0]

    cor_mat = np.array(
        [
            [1.0, 0.30, 0.15, 0.40],
            [0.30, 1.0, 0.20, -0.25],
            [0.15, 0.20, 1.0, 0.35],
            [0.40, -0.25, 0.35, 1.0],
        ],
        dtype=float,
    )
    sd_all = np.full(p + 1, 0.02)
    cov_all = np.outer(sd_all, sd_all) * cor_mat
    eps = rng.multivariate_normal(np.zeros(p + 1), cov_all, size=m)

    bx_hat = true_bx + eps[:, :p]
    by_hat = true_bx @ beta_true + eps[:, p]
    # Add 6 pleiotropic outlier SNPs
    by_hat[:6] += 0.30

    res = grapple_mvmr(
        bx_hat,
        sx,
        by_hat,
        sy,
        cor_mat=cor_mat,
        exposure_names=["LDL", "HDL", "TG"],
        settings="R_default",
    )
    assert isinstance(res, GrappleMVMRResult)
    assert np.allclose(res.beta_hat, beta_true, atol=0.06)
    assert res.p_values[0] < 1e-4
    assert res.p_values[1] < 1e-3
    assert res.p_values[2] > 0.01
    assert np.all(res.inlier_weights[:6] < 0.05)  # Outliers rejected by Tukey biweight!


def test_grapple_mvmr_multivariable_instrument_selection() -> None:
    rng = np.random.default_rng(92)
    m = 300
    df = pd.DataFrame(
        {
            "snp": [f"rs{i}" for i in range(m)],
            "bx1": rng.normal(0.0, 0.02, size=m),
            "bx2": rng.normal(0.0, 0.02, size=m),
            "sx1": np.full(m, 0.02),
            "sx2": np.full(m, 0.02),
            "by": rng.normal(0.0, 0.02, size=m),
            "sy": np.full(m, 0.02),
        }
    )
    df.loc[:9, "bx1"] = 0.10
    df.loc[10:19, "bx2"] = -0.10
    sel = select_grapple_instruments(df, p_thres=1e-4)
    assert len(sel) >= 20


def test_grapple_class_end_to_end_and_dataframe_exports() -> None:
    rng = np.random.default_rng(104)
    m_all = 40_000
    cor_true = np.array([[1.0, 0.35], [0.35, 1.0]])
    eps = rng.multivariate_normal([0.0, 0.0], 0.02**2 * cor_true, size=m_all)
    true_bx = np.zeros(m_all)
    true_bx[:80] = rng.normal(0.0, 0.09, size=80)
    bx = true_bx + eps[:, 0]
    by = 0.30 * true_bx + eps[:, 1]
    sx = np.full(m_all, 0.02)
    sy = np.full(m_all, 0.02)

    df = pd.DataFrame(
        {"snp": [f"rs{i}" for i in range(m_all)], "bx": bx, "sx": sx, "by": by, "sy": sy}
    )

    model = GRAPPLE(p_thres=1e-4, settings="default")
    cmat = model.estimate_cor_mat(df, p_thres=0.5)
    assert abs(cmat[0, 1] - 0.35) < 0.06

    ivs = model.select_instruments(df)
    res = model.fit(ivs["bx"], ivs["sx"], ivs["by"], ivs["sy"], snp_ids=list(ivs["snp"]))
    assert abs(res.beta_hat - 0.30) < 0.06

    mode_df = res.to_dataframe()
    marker_df = res.marker_dataframe()
    assert len(mode_df) >= 1
    assert len(marker_df) >= 1
    assert "GRAPPLE Summary" in res.summary()


def test_grapple_class_mvmr_workflow() -> None:
    rng = np.random.default_rng(112)
    m = 80
    bx = rng.normal(0.0, 0.08, size=(m, 2))
    sx = np.full((m, 2), 0.02)
    sy = np.full(m, 0.02)
    by = 0.30 * bx[:, 0] - 0.20 * bx[:, 1] + rng.normal(0.0, 0.02, size=m)

    model = GRAPPLE(settings="R_default")
    mv_res = model.fit_mvmr(bx, sx, by, sy, exposure_names=["BMI", "SBP"])
    df = mv_res.to_dataframe()
    assert list(df["exposure"]) == ["BMI", "SBP"]
    assert abs(mv_res.beta_hat[0] - 0.30) < 0.06
    assert abs(mv_res.beta_hat[1] - (-0.20)) < 0.06
    assert "GRAPPLE MVMR Summary" in mv_res.summary()


def test_invalid_settings_raises_value_error_everywhere() -> None:
    bx = np.array([0.1, -0.1, 0.15, -0.12])
    sx = np.full(4, 0.02)
    df = pd.DataFrame({"z1": np.linspace(-1, 1, 50), "z2": np.linspace(-1, 1, 50)})
    ss = pd.DataFrame({"bx": bx, "sx": sx, "by": bx, "sy": sx})

    with pytest.raises(ValueError, match="Unknown settings"):
        grapple_robust_est(bx, sx, bx, sx, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        grapple_mvmr(bx[:, None], sx[:, None], bx, sx, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        estimate_cor_mat(df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        select_grapple_instruments(ss, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        GRAPPLE(settings="bad")  # type: ignore[arg-type]


def test_input_and_hyperparameter_validation_errors() -> None:
    bx = np.array([0.1, -0.1, 0.15, -0.12])
    sx = np.full(4, 0.02)
    with pytest.raises(ValueError, match="Unknown loss_function"):
        grapple_robust_est(bx, sx, bx, sx, loss_function="cauchy")
    with pytest.raises(ValueError, match="cor_xy"):
        grapple_robust_est(bx, sx, bx, sx, cor_xy=1.0)
    with pytest.raises(ValueError, match="positive definite"):
        bad_cor = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]])
        grapple_mvmr(
            np.column_stack([bx, -bx]),
            np.column_stack([sx, sx]),
            bx,
            sx,
            cor_mat=bad_cor,
        )
