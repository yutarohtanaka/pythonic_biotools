from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import polygamma
from scipy.stats import norm

from limma import (
    limma_top_table,
    limma_trend,
    moderated_lm_contrast,
    moderated_ttest,
    squeeze_var,
    trigamma_inverse,
)


def test_trigamma_inverse_accuracy_shape_and_edge_cases():
    y_grid = np.array([[1e-4, 0.05, 0.25], [1.5, 10.0, 250.0], [1e3, 1e4, 5e4]])
    recovered = trigamma_inverse(polygamma(1, y_grid))
    assert recovered.shape == y_grid.shape
    np.testing.assert_allclose(recovered, y_grid, rtol=1e-10, atol=1e-12)

    assert float(trigamma_inverse(np.inf)) == 0.0
    assert np.isnan(trigamma_inverse(np.nan))
    assert np.isnan(trigamma_inverse(0.0))
    assert np.isnan(trigamma_inverse(-3.5))
    assert np.isnan(trigamma_inverse(-np.inf))


def test_moderated_ttest_and_lm_contrast_parity_and_df0_inf_shrinkage():
    rng = np.random.default_rng(42)
    n1, n2, n_genes = 6, 6, 200
    X = rng.normal(0.0, 1.0, size=(n1 + n2, n_genes))
    group = np.array([False] * n1 + [True] * n2)
    X[group, :5] += 3.5

    res_t = moderated_ttest(X, group, feature_names=[f"G{i}" for i in range(n_genes)])
    design = np.column_stack([np.ones(n1 + n2), group.astype(float)])
    res_lm = moderated_lm_contrast(X, design, contrast=[0.0, 1.0], feature_names=[f"G{i}" for i in range(n_genes)])
    np.testing.assert_allclose(res_t.t_stat, res_lm.t_stat, atol=1e-10)
    np.testing.assert_allclose(res_t.p_value, res_lm.p_value, atol=1e-12)
    top = limma_top_table(pd.DataFrame(X, columns=[f"G{i}" for i in range(n_genes)]), group)
    assert set(top.index[:5]) == {"G0", "G1", "G2", "G3", "G4"}

    # Construct a dataset whose empirical log-variance dispersion < trigamma(df1 / 2), forcing df0 == inf:
    base_template = np.array([-1.2, -0.6, -0.2, 0.2, 0.6, 1.2])
    X_homog = np.tile(np.r_[base_template, base_template][:, None], (1, 50))
    # Tiny jitter across genes so sample variances are nearly identical (~0.72), guaranteeing evar < 0 -> df0 == inf
    X_homog += rng.normal(0.0, 0.01, size=X_homog.shape)
    X_homog[group, 0] += 2.0

    res_inf_t = moderated_ttest(X_homog, group)
    res_inf_lm = moderated_lm_contrast(X_homog, design, contrast=[0.0, 1.0])
    assert np.isinf(res_inf_t.df_prior) and np.isinf(res_inf_lm.df_prior)
    # Must shrink 100% to var_prior (s20) and evaluate exact normal survival p-values
    np.testing.assert_allclose(res_inf_t.var_post, res_inf_t.var_prior, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_inf_lm.var_post, res_inf_lm.var_prior, rtol=1e-12, atol=1e-12)
    expected_norm_p = 2.0 * norm.sf(np.abs(res_inf_t.t_stat))
    np.testing.assert_allclose(res_inf_t.p_value, expected_norm_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_inf_lm.p_value, expected_norm_p, rtol=1e-12, atol=1e-12)


def test_nan_propagation_and_bh_fdr_uninflated_m():
    rng = np.random.default_rng(101)
    n1, n2, n_genes = 5, 5, 40
    X = rng.normal(0.0, 1.0, size=(n1 + n2, n_genes))
    group = np.array([False] * n1 + [True] * n2)
    X[group, 0] += 4.0

    # Compare clean 40-gene run vs appending 20 all-NaN columns (must NOT inflate BH m!)
    res_clean = moderated_ttest(X, group)
    X_with_nans = np.column_stack([X, np.full((n1 + n2, 20), np.nan)])
    res_nan_cols = moderated_ttest(X_with_nans, group)

    assert np.all(np.isnan(res_nan_cols.p_value[40:]))
    assert np.all(np.isnan(res_nan_cols.adj_p_val[40:]))
    np.testing.assert_allclose(res_nan_cols.adj_p_val[:40], res_clean.adj_p_val, rtol=1e-12, atol=1e-12)

    # Partial NaNs in gene 1: moderated_ttest and moderated_lm_contrast must still agree
    X_part = X.copy()
    X_part[0, 1] = np.nan
    X_part[6, 1] = np.nan
    X_part[:, 2] = np.nan
    design = np.column_stack([np.ones(n1 + n2), group.astype(float)])
    rt = moderated_ttest(X_part, group)
    rlm = moderated_lm_contrast(X_part, design, contrast=[0.0, 1.0])
    assert np.isfinite(rt.t_stat[1]) and rt.t_stat[1] == pytest.approx(rlm.t_stat[1], rel=1e-10)
    assert np.isnan(rt.p_value[2]) and np.isnan(rt.adj_p_val[2])
    assert np.isnan(rlm.p_value[2]) and np.isnan(rlm.adj_p_val[2])


def test_robust_ebayes_and_limma_trend():
    rng = np.random.default_rng(2024)
    n1, n2, n_genes = 6, 6, 300
    group = np.array([False] * n1 + [True] * n2)
    X = rng.normal(0.0, 1.0, size=(n1 + n2, n_genes))
    # Inject 6 extreme hypervariable outlier genes (sigma = 20 -> var ~ 400)
    X[:, -6:] = rng.normal(0.0, 20.0, size=(n1 + n2, 6))

    res_std = moderated_ttest(X, group, robust=False)
    res_rob = moderated_ttest(X, group, robust=True)

    # Non-robust df_prior collapses drastically due to the 6 variance outliers,
    # whereas robust=True preserves strong prior df on bulk inlier genes while downweighting outliers!
    assert res_std.df_prior < 5.0
    assert np.median(res_rob.df_prior[:-6]) > 15.0
    assert np.max(res_rob.df_prior[-6:]) < 0.05
    # Outlier posterior variance stays near its true large sample variance rather than overshrinking to ~1
    assert np.all(res_rob.var_post[-6:] > 50.0)

    # Test limma_trend on strong mean-variance relationship: log(sigma^2) decreases with AveExpr
    ave_levels = np.linspace(2.0, 12.0, n_genes)
    true_sd = np.exp(-0.25 * (ave_levels - 6.0))
    X_trend = ave_levels[None, :] + rng.normal(0.0, 1.0, size=(n1 + n2, n_genes)) * true_sd[None, :]
    res_tr = limma_trend(X_trend, group)
    assert isinstance(res_tr.var_prior, np.ndarray)
    # Prior variance at low expression must be substantially larger than at high expression
    assert res_tr.var_prior[0] > 5.0 * res_tr.var_prior[-1]

    # Direct squeeze_var check
    vp, s20, df0, _ = squeeze_var(np.ones(20), df1=8)
    assert np.isinf(df0) and np.allclose(vp, s20)


def test_r_default_settings():
    rng = np.random.default_rng(31415)

    # 1. Invalid settings must raise ValueError across public entrypoints
    X_tiny = rng.normal(size=(6, 10))
    grp_tiny = [False, False, False, True, True, True]
    design_tiny = np.column_stack([np.ones(6), np.asarray(grp_tiny, dtype=float)])
    for fn in (
        lambda: squeeze_var(np.ones(10), df1=4, settings="bad"),  # type: ignore[arg-type]
        lambda: moderated_ttest(X_tiny, grp_tiny, settings="bad"),  # type: ignore[arg-type]
        lambda: moderated_lm_contrast(X_tiny, design_tiny, [0.0, 1.0], settings="bad"),  # type: ignore[arg-type]
        lambda: limma_trend(X_tiny, grp_tiny, settings="bad"),  # type: ignore[arg-type]
        lambda: limma_top_table(X_tiny, grp_tiny, settings="bad"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="settings"):
            fn()

    # 2. Smyth (2004) B log-odds (`b_stat` / `lods` / `topTable['B']`)
    n1, n2, n_genes = 6, 6, 200
    group = np.array([False] * n1 + [True] * n2)
    X = rng.normal(0.0, 1.0, size=(n1 + n2, n_genes))
    X[group, :4] += 4.5
    res_def = moderated_ttest(X, group)
    res_r = moderated_ttest(X, group, settings="R_default")
    assert res_def.b_stat is not None and res_r.b_stat is not None
    np.testing.assert_array_equal(res_r.lods, res_r.b_stat)
    assert np.all(res_r.b_stat[:4] > 2.0)
    assert np.median(res_r.b_stat[4:]) < -3.0
    # Under constant df_total & unscaled_var, B is strictly increasing in |t_stat|
    order_abs_t = np.argsort(np.abs(res_r.t_stat))
    assert np.all(np.diff(res_r.b_stat[order_abs_t]) >= -1e-12)
    top_df = res_r.to_dataframe()
    assert "B" in top_df.columns

    # 3. evar <= 0: s20 = mean(s^2) and df_total capped at sum(df_resid) in R_default
    vars_homog = np.linspace(0.95, 1.05, 30)
    vp_r, s20_r, df0_r, dft_r = squeeze_var(vars_homog, df1=8, settings="R_default")
    assert np.isinf(df0_r)
    assert s20_r == pytest.approx(float(np.mean(vars_homog)), rel=1e-12)
    assert dft_r == pytest.approx(30.0 * 8.0, rel=1e-12)
    np.testing.assert_allclose(vp_r, s20_r, rtol=1e-12)

    # 4. Flooring tiny positive variances at 1e-5 * median(s^2) in R_default prevents df0 collapse
    vars_tiny_outlier = rng.chisquare(df=8, size=500) / 8.0
    vars_tiny_outlier[0] = 1e-40
    _, _, df0_def_tiny, _ = squeeze_var(vars_tiny_outlier, df1=8, settings="default")
    _, _, df0_r_tiny, _ = squeeze_var(vars_tiny_outlier, df1=8, settings="R_default")
    assert df0_def_tiny < 0.6
    assert df0_r_tiny > 6.0

    # 5. R_default robust=True (128-node Gauss-Legendre Winsorized log-F + df2_outlier mixture)
    X_rob = rng.normal(0.0, 1.0, size=(n1 + n2, 300))
    X_rob[:, -6:] = rng.normal(0.0, 20.0, size=(n1 + n2, 6))
    res_rob_r = moderated_ttest(X_rob, group, robust=True, settings="R_default")
    assert isinstance(res_rob_r.df_prior, np.ndarray)
    assert np.median(res_rob_r.df_prior[:-6]) > 15.0
    # Outliers get positive finite df2_outlier mixture (< 1.0) rather than 0, preserving sample variance
    assert np.all((res_rob_r.df_prior[-6:] > 0.05) & (res_rob_r.df_prior[-6:] < 1.0))
    assert np.all(res_rob_r.var_post[-6:] > 50.0)

    # 6. R_default trend=True (4-df natural cubic regression spline ns(covariate, df=4))
    ave_levels = np.linspace(2.0, 12.0, 200)
    true_sd = np.exp(-0.25 * (ave_levels - 6.0))
    X_tr = ave_levels[None, :] + rng.normal(0.0, 1.0, size=(n1 + n2, 200)) * true_sd[None, :]
    res_tr_def = limma_trend(X_tr, group, settings="default")
    res_tr_r = limma_trend(X_tr, group, settings="R_default")
    res_tr_rob_r = limma_trend(X_tr, group, robust=True, settings="R_default")
    assert res_tr_r.var_prior[0] > 5.0 * res_tr_r.var_prior[-1]
    assert not np.allclose(res_tr_r.var_prior, res_tr_def.var_prior)
    assert res_tr_rob_r.var_prior[0] > 5.0 * res_tr_rob_r.var_prior[-1]

