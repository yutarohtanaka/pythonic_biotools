from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from twosamplemr import (
    harmonise_mr_data,
    ivw_mr,
    mr_egger,
    multivariable_mr,
    run_mr_suite,
    select_mr_instruments,
    weighted_median_mr,
)


def test_mr_suite_coding_invariance_and_mvmr():
    rng = np.random.default_rng(2026)
    k = 30
    bx = rng.uniform(0.08, 0.25, size=k) * rng.choice([-1.0, 1.0], size=k)
    sx = np.full(k, 0.015)
    conf = rng.normal(0.0, 0.12, size=k)
    sc = np.full(k, 0.015)
    true_theta = 0.60
    sy = np.full(k, 0.03)
    by = true_theta * bx + 0.35 * conf + rng.normal(0.0, sy)

    res1 = run_mr_suite(bx, sx, by, sy, covariate_exp_betas=conf, covariate_exp_ses=sc, seed=10)
    flip = np.where(np.arange(k) % 2 == 0, -1.0, 1.0)
    res2 = run_mr_suite(bx * flip, sx, by * flip, sy, covariate_exp_betas=conf * flip, covariate_exp_ses=sc, seed=10)

    np.testing.assert_allclose(res1.beta_ivw, res2.beta_ivw, atol=1e-10)
    np.testing.assert_allclose(res1.beta_egger, res2.beta_egger, atol=1e-10)
    np.testing.assert_allclose(res1.egger_intercept, res2.egger_intercept, atol=1e-10)
    np.testing.assert_allclose(res1.beta_weighted_median, res2.beta_weighted_median, atol=1e-10)
    np.testing.assert_allclose(res1.se_weighted_median, res2.se_weighted_median, atol=1e-10)
    np.testing.assert_allclose(res1.conditional_f_stat, res2.conditional_f_stat, atol=1e-10)
    assert abs(res1.beta_mvmr - true_theta) < 0.12
    assert res1.conditional_f_stat > 20.0
    df_summary = res1.to_dataframe()
    assert len(df_summary) == 5


def test_weighted_median_resamples_exposure_uncertainty():
    rng = np.random.default_rng(77)
    k = 25
    bx = rng.uniform(0.10, 0.22, size=k)
    by = 0.50 * bx + rng.normal(0.0, 0.01, size=k)
    sy = np.full(k, 0.01)
    sx_small = np.full(k, 1e-6)
    sx_large = np.full(k, 0.04)

    wm_no_sx = weighted_median_mr(bx, by, sy, se_exp=None, n_boot=1000, seed=123)
    wm_tiny_sx = weighted_median_mr(bx, by, sy, se_exp=sx_small, n_boot=1000, seed=123)
    wm_large_sx = weighted_median_mr(bx, by, sy, se_exp=sx_large, n_boot=1000, seed=123)

    # Larger exposure SE must increase parametric bootstrap SE per Bowden et al. (2016)
    assert wm_large_sx["se"] > wm_tiny_sx["se"] * 1.35
    np.testing.assert_allclose(wm_no_sx["beta"], wm_large_sx["beta"], atol=1e-12)


def test_mvmr_sanderson_windmeijer_exposure_weighting_and_t_pvals():
    rng = np.random.default_rng(88)
    k = 20
    c_true = rng.normal(0.0, 0.15, size=(k, 2))
    delta_true = np.array([1.2, -0.8])
    # Primary exposure is collinear + small independent signal
    bx = c_true @ delta_true + rng.normal(0.0, 0.06, size=k)
    sx = rng.uniform(0.01, 0.03, size=k)
    sc = rng.uniform(0.015, 0.035, size=(k, 2))
    by = 0.4 * bx - 0.2 * c_true[:, 0] + 0.3 * c_true[:, 1] + rng.normal(0.0, 0.05, size=k)
    # Wildly heteroskedastic outcome SE should NOT change Sanderson-Windmeijer conditional F!
    sy1 = np.full(k, 0.05)
    sy2 = np.concatenate([np.full(k // 2, 0.005), np.full(k - k // 2, 5.0)])

    mv1 = multivariable_mr(bx, sx, c_true, by, sy1, se_exp_covariates=sc)
    mv2 = multivariable_mr(bx, sx, c_true, by, sy2, se_exp_covariates=sc)
    np.testing.assert_allclose(mv1["conditional_f_stat"], mv2["conditional_f_stat"], rtol=1e-12)

    # Accounting for covariate exposure SEs reduces conditional F vs assuming sc = 0
    mv_no_sc = multivariable_mr(bx, sx, c_true, by, sy1, se_exp_covariates=None)
    assert mv1["conditional_f_stat"] < mv_no_sc["conditional_f_stat"]

    # Verify Student-t_{K - P} p-value (df = 20 - 3 = 17)
    t_stat = abs(mv1["beta_primary"] / mv1["se_primary"])
    expected_p = float(2.0 * stats.t.sf(t_stat, df=k - 3))
    np.testing.assert_allclose(mv1["pval_primary"], expected_p, rtol=1e-13)


def test_palindromic_nan_filter_clumping_and_harmonise_mr_data():
    df = pd.DataFrame(
        {
            "snp": ["rs1", "rs2_pal_mid", "rs3_ld", "rs4_indep", "rs5_pal_nan", "rs6_pal_ok"],
            "ref": ["A", "A", "G", "C", "C", "A"],
            "alt": ["G", "T", "A", "T", "G", "T"],
            "eaf": [0.25, 0.50, 0.30, 0.15, np.nan, 0.12],
            "pval_exp": [1e-12, 1e-15, 1e-9, 1e-10, 1e-20, 1e-11],
        }
    )
    r2 = np.eye(6)
    r2[0, 2] = r2[2, 0] = 0.85
    clumped = select_mr_instruments(df, r2_matrix=r2, r2_threshold=0.01)
    # rs2_pal_mid (eaf=0.50) and rs5_pal_nan (eaf=NaN) dropped; rs3_ld clumped against rs1
    assert list(clumped["snp"]) == ["rs1", "rs6_pal_ok", "rs4_indep"]

    # Test harmonise_mr_data across direct, swapped, strand-flipped, palindromic EAF-resolved, and ambiguous
    exp_df = pd.DataFrame(
        {
            "snp": ["s_same", "s_swap", "s_comp", "s_comp_swap", "s_pal_same", "s_pal_opp", "s_pal_amb", "s_pal_nan", "s_bad"],
            "effect_allele": ["A", "A", "A", "A", "A", "A", "C", "A", "A"],
            "other_allele": ["G", "G", "G", "G", "T", "T", "G", "T", "G"],
            "beta": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "se": [0.02] * 9,
            "eaf": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.48, np.nan, 0.20],
        }
    )
    out_df = pd.DataFrame(
        {
            "snp": ["s_same", "s_swap", "s_comp", "s_comp_swap", "s_pal_same", "s_pal_opp", "s_pal_amb", "s_pal_nan", "s_bad"],
            "effect_allele": ["A", "G", "T", "C", "T", "A", "C", "A", "A"],
            "other_allele": ["G", "A", "C", "T", "A", "T", "G", "T", "C"],
            "beta": [0.10, -0.10, 0.10, -0.10, 0.10, -0.10, 0.10, 0.10, 0.10],
            "se": [0.03] * 9,
            "eaf": [0.21, 0.79, 0.22, 0.78, 0.19, 0.81, 0.51, 0.20, 0.20],
        }
    )
    harm2 = harmonise_mr_data(exp_df, out_df, action=2)
    assert list(harm2["snp"]) == ["s_same", "s_swap", "s_comp", "s_comp_swap", "s_pal_same", "s_pal_opp"]
    np.testing.assert_allclose(harm2["beta_out"].to_numpy(), np.full(6, 0.10), atol=1e-12)
    assert np.all(harm2["eaf_out"].to_numpy() < 0.30)

    harm3 = harmonise_mr_data(exp_df, out_df, action=3)
    assert list(harm3["snp"]) == ["s_same", "s_swap", "s_comp", "s_comp_swap"]

    with pytest.raises(ValueError):
        harmonise_mr_data(exp_df, out_df, action=99)


def test_r_default_settings():
    # 1. Strict settings validation across all entrypoints
    bx = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    sx = np.array([0.01, 0.08, 0.01, 0.09, 0.01])
    by = np.array([0.02, 0.18, 0.06, 0.35, 0.10])
    sy = np.array([0.02, 0.01, 0.02, 0.01, 0.02])
    cov_b = np.array([0.05, -0.10, 0.12, -0.08, 0.15])

    for bad in ("invalid", "r_default", "", None):
        with pytest.raises(ValueError, match="Unknown settings"):
            ivw_mr(bx, by, sy, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            mr_egger(bx, by, sy, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            weighted_median_mr(bx, by, sy, se_exp=sx, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            multivariable_mr(bx, sx, cov_b, by, sy, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            run_mr_suite(bx, sx, by, sy, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            select_mr_instruments(pd.DataFrame({"pval_exp": [1e-10]}), settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            harmonise_mr_data(pd.DataFrame(), pd.DataFrame(), settings=bad)  # type: ignore[arg-type]

    # 2. weighted_median_mr: 2nd-order delta-method weights + default n_boot=1000
    wm_def = weighted_median_mr(bx, by, sy, se_exp=sx, seed=99, settings="default")
    wm_r = weighted_median_mr(bx, by, sy, se_exp=sx, seed=99, settings="R_default")
    wm_r_1000 = weighted_median_mr(bx, by, sy, se_exp=sx, n_boot=1000, seed=99, settings="R_default")
    wm_r_200 = weighted_median_mr(bx, by, sy, se_exp=sx, n_boot=200, seed=99, settings="R_default")

    ratios = by / bx
    w1 = (bx**2) / (sy**2)
    w1 /= np.sum(w1)
    w2 = 1.0 / ((sy**2) / (bx**2) + (by**2) * (sx**2) / (bx**4))
    w2 /= np.sum(w2)
    ord_r = np.argsort(ratios, kind="mergesort")
    expected_beta_def = float(np.interp(0.5, np.cumsum(w1[ord_r]) - 0.5 * w1[ord_r], ratios[ord_r]))
    expected_beta_r = float(np.interp(0.5, np.cumsum(w2[ord_r]) - 0.5 * w2[ord_r], ratios[ord_r]))

    np.testing.assert_allclose(wm_def["beta"], expected_beta_def, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wm_r["beta"], expected_beta_r, rtol=1e-12, atol=1e-12)
    assert abs(wm_r["beta"] - wm_def["beta"]) > 0.05
    np.testing.assert_allclose(wm_r["se"], wm_r_1000["se"], rtol=1e-13, atol=1e-13)
    assert wm_r["se"] != wm_r_200["se"]

    # 3. multivariable_mr: unweighted OLS delta_hat + un-clamped residual variance + Student-t_{K-P} p-values
    rng = np.random.default_rng(2027)
    k = 18
    c_mat = rng.normal(0.0, 0.2, size=(k, 2))
    # Heteroskedastic sx creates divergence between unweighted OLS delta_hat and WLS/IRLS delta_hat
    sx_mv = np.concatenate([np.full(k // 2, 0.005), np.full(k - k // 2, 0.08)])
    sc_mv = rng.uniform(0.01, 0.03, size=(k, 2))
    bx_mv = c_mat @ np.array([1.1, -0.7]) + np.concatenate(
        [rng.normal(0.0, 0.02, k // 2), rng.normal(0.25, 0.05, k - k // 2)]
    )
    sy_mv = np.full(k, 0.10)
    # Small residual noise so raw_s2 = rss / (K - P) < 1.0 (underdispersed)
    by_mv = 0.50 * bx_mv - 0.25 * c_mat[:, 0] + 0.15 * c_mat[:, 1] + rng.normal(0.0, 0.03, size=k)

    mv_def = multivariable_mr(bx_mv, sx_mv, c_mat, by_mv, sy_mv, se_exp_covariates=sc_mv, settings="default")
    mv_r = multivariable_mr(bx_mv, sx_mv, c_mat, by_mv, sy_mv, se_exp_covariates=sc_mv, settings="R_default")

    # Verify exact WSpiller/MVMR equations
    x_mat = np.column_stack([bx_mv, c_mat])
    w_out = 1.0 / (sy_mv**2)
    xtwx_inv = np.linalg.pinv(x_mat.T @ (x_mat * w_out[:, None]))
    theta_hat = xtwx_inv @ (x_mat.T @ (w_out * by_mv))
    df_mvmr = k - 3
    raw_s2 = float(np.sum(w_out * ((by_mv - x_mat @ theta_hat) ** 2)) / df_mvmr)
    assert raw_s2 < 0.5  # strictly < 1.0 so un-clamped SE differs from clamped SE

    expected_se_r = np.sqrt(raw_s2 * np.diag(xtwx_inv))
    expected_p_r = 2.0 * stats.t.sf(np.abs(theta_hat / expected_se_r), df=df_mvmr)
    np.testing.assert_allclose(mv_r["all_betas"], theta_hat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mv_r["all_ses"], expected_se_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mv_r["all_pvals"], expected_p_r, rtol=1e-12, atol=1e-12)
    assert mv_r["se_primary"] < mv_def["se_primary"] * 0.75

    delta_ols = np.linalg.pinv(c_mat.T @ c_mat) @ (c_mat.T @ bx_mv)
    var_cond_ols = sx_mv**2 + (sc_mv**2) @ (delta_ols**2)
    expected_f_r = float(np.sum(((bx_mv - c_mat @ delta_ols) ** 2) / var_cond_ols) / (k - 2))
    np.testing.assert_allclose(mv_r["conditional_f_stat"], expected_f_r, rtol=1e-12, atol=1e-12)
    assert not np.isclose(mv_r["conditional_f_stat"], mv_def["conditional_f_stat"], rtol=1e-3)

    # Verify run_mr_suite forwards settings="R_default"
    suite_r = run_mr_suite(
        bx_mv, sx_mv, by_mv, sy_mv, covariate_exp_betas=c_mat, covariate_exp_ses=sc_mv, seed=42, settings="R_default"
    )
    np.testing.assert_allclose(suite_r.se_mvmr, mv_r["se_primary"], rtol=1e-12)
    np.testing.assert_allclose(suite_r.conditional_f_stat, mv_r["conditional_f_stat"], rtol=1e-12)

    # 4. select_mr_instruments: default r2_threshold = 0.001 in R_default vs 0.01 in default
    df_clump = pd.DataFrame(
        {
            "snp": ["rs_lead", "rs_mid_ld", "rs_indep"],
            "ref": ["A", "C", "G"],
            "alt": ["G", "T", "A"],
            "eaf": [0.30, 0.25, 0.40],
            "pval_exp": [1e-15, 1e-10, 1e-9],
        }
    )
    r2_mat = np.array(
        [
            [1.0, 0.005, 0.0002],
            [0.005, 1.0, 0.0001],
            [0.0002, 0.0001, 1.0],
        ]
    )
    sel_def = select_mr_instruments(df_clump, r2_matrix=r2_mat, settings="default")
    sel_r = select_mr_instruments(df_clump, r2_matrix=r2_mat, settings="R_default")
    sel_r_override = select_mr_instruments(df_clump, r2_matrix=r2_mat, r2_threshold=0.02, settings="R_default")
    assert list(sel_def["snp"]) == ["rs_lead", "rs_mid_ld", "rs_indep"]
    assert list(sel_r["snp"]) == ["rs_lead", "rs_indep"]
    assert list(sel_r_override["snp"]) == ["rs_lead", "rs_mid_ld", "rs_indep"]

