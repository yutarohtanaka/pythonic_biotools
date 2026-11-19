from __future__ import annotations
import numpy as np
import pandas as pd
from twosamplemr import run_mr_suite, select_mr_instruments

def test_mr_suite_coding_invariance_and_mvmr():
    rng = np.random.default_rng(2026)
    k = 30
    bx = rng.uniform(0.08, 0.25, size=k) * rng.choice([-1.0, 1.0], size=k)
    sx = np.full(k, 0.015)
    conf = rng.normal(0.0, 0.12, size=k)
    true_theta = 0.60
    sy = np.full(k, 0.03)
    by = true_theta * bx + 0.35 * conf + rng.normal(0.0, sy)

    res1 = run_mr_suite(bx, sx, by, sy, covariate_exp_betas=conf, seed=10)
    flip = np.where(np.arange(k) % 2 == 0, -1.0, 1.0)
    res2 = run_mr_suite(bx * flip, sx, by * flip, sy, covariate_exp_betas=conf * flip, seed=10)

    np.testing.assert_allclose(res1.beta_ivw, res2.beta_ivw, atol=1e-10)
    np.testing.assert_allclose(res1.beta_egger, res2.beta_egger, atol=1e-10)
    np.testing.assert_allclose(res1.egger_intercept, res2.egger_intercept, atol=1e-10)
    assert abs(res1.beta_mvmr - true_theta) < 0.12
    assert res1.conditional_f_stat > 20.0

def test_palindromic_filter_and_ld_clumping():
    df = pd.DataFrame({
        "snp": ["rs1", "rs2_pal", "rs3_ld", "rs4_indep"],
        "ref": ["A", "A", "G", "C"],
        "alt": ["G", "T", "A", "T"],
        "eaf": [0.25, 0.50, 0.30, 0.15],
        "pval_exp": [1e-12, 1e-15, 1e-9, 1e-10],
    })
    r2 = np.eye(4)
    r2[0, 2] = r2[2, 0] = 0.85
    clumped = select_mr_instruments(df, r2_matrix=r2, r2_threshold=0.01)
    assert list(clumped["snp"]) == ["rs1", "rs4_indep"]
