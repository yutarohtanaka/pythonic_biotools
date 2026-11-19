from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from decoupler_ulm import project_gep_programs


def test_project_gep_programs():
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(60)]
    w_df = pd.DataFrame({"Prog1": np.r_[np.ones(15), np.zeros(45)], "Prog2": np.r_[np.zeros(45), np.ones(15)]}, index=genes)
    expr_df = pd.DataFrame(
        {
            "S1": np.r_[rng.uniform(50, 120, 15), rng.uniform(1, 5, 45)],
            "S2": np.r_[rng.uniform(1, 5, 45), rng.uniform(50, 120, 15)],
            "S3": rng.uniform(5, 10, 60),
        },
        index=genes,
    )
    scores = project_gep_programs(expr_df, w_df, is_log_transformed=False, method="ulm_tstat")
    assert scores.loc["S1", "Prog1"] > 3.0 and scores.loc["S2", "Prog2"] > 3.0


def test_single_sample_ulm_exact_ols_pvalues_and_nan_handling():
    rng = np.random.default_rng(42)
    genes = [f"G{i}" for i in range(50)]
    w1 = rng.uniform(0.0, 2.0, size=50)
    w2 = rng.uniform(0.0, 2.0, size=50)
    w_df = pd.DataFrame({"P1": w1, "P2": w2}, index=genes)

    y1 = 1.5 + 2.2 * w1 + rng.normal(0.0, 0.4, size=50)
    y2 = 2.0 + 1.8 * w2 + rng.normal(0.0, 0.4, size=50)
    expr_log = pd.DataFrame({"SampleA": y1, "SampleB": y2}, index=genes)

    # 1. Single-sample invariance (decoupler.run_ulm exact mode, center_across_samples=False)
    scores_multi, pvals_multi = project_gep_programs(
        expr_log, w_df, is_log_transformed=True, method="ulm_tstat", return_pvalues=True
    )
    scores_single, pvals_single = project_gep_programs(
        expr_log[["SampleA"]], w_df, is_log_transformed=True, method="ulm_tstat", return_pvalues=True
    )
    pd.testing.assert_frame_equal(scores_single, scores_multi.loc[["SampleA"]], rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(pvals_single, pvals_multi.loc[["SampleA"]], rtol=1e-12, atol=1e-12)

    # Exact match against scipy.stats.linregress OLS t-statistic and two-sided p-value
    lr = stats.linregress(w1, y1)
    expected_t = lr.slope / lr.stderr
    assert scores_single.loc["SampleA", "P1"] == pytest.approx(expected_t, rel=1e-10)
    assert pvals_single.loc["SampleA", "P1"] == pytest.approx(lr.pvalue, rel=1e-10)

    # 2. Cohort-centered mode (center_across_samples=True / zscore_genes=True)
    sc_cent = project_gep_programs(expr_log, w_df, is_log_transformed=True, center_across_samples=True)
    sc_alias = project_gep_programs(expr_log, w_df, is_log_transformed=True, zscore_genes=True)
    pd.testing.assert_frame_equal(sc_cent, sc_alias)
    assert sc_cent.loc["SampleA", "P1"] > 3.0

    # 3. NaN resilience in log2_dot, top_k_zscore, and ulm_tstat
    expr_nan = expr_log.copy()
    expr_nan.iloc[0, 0] = np.nan
    expr_nan.iloc[3, 1] = np.nan
    w_nan = w_df.copy()
    w_nan.iloc[1, 0] = np.nan

    dot_scores = project_gep_programs(expr_nan, w_nan, is_log_transformed=True, method="log2_dot")
    assert np.all(np.isfinite(dot_scores.to_numpy()))

    topk_scores, topk_pvals = project_gep_programs(
        expr_nan, w_nan, is_log_transformed=True, method="top_k_zscore", top_k_markers=15, return_pvalues=True
    )
    assert np.all(np.isfinite(topk_scores.to_numpy()))
    assert np.all((topk_pvals.to_numpy() >= 0.0) & (topk_pvals.to_numpy() <= 1.0))

    ulm_nan_scores, ulm_nan_pvals = project_gep_programs(
        expr_nan, w_nan, is_log_transformed=True, method="ulm_tstat", return_pvalues=True
    )
    assert np.all(np.isfinite(ulm_nan_scores.to_numpy()))
    assert ulm_nan_pvals.loc["SampleA", "P1"] < 1e-6
