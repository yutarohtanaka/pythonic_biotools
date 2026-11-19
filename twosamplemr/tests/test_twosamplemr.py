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
