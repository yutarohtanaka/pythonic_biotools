from __future__ import annotations
import numpy as np
import pytest
from pythonic_biostats import (
    compute_mdes_interaction_hr, fit_cox_ph_efron, fit_fine_gray_ipcw, fit_wls_hc3,
    partial_corr_pval, permutation_empirical_fdr, spearman_matrix, ttest_ind_safe,
)

def test_cox_efron_and_fine_gray_competing_risks():
    rng = np.random.default_rng(123)
    n = 250
    x = rng.normal(0.0, 1.0, size=(n, 2))
    t1 = rng.exponential(1.0 / np.exp(0.9 * x[:, 0]))
    t2 = rng.exponential(1.0 / np.full(n, 0.35))
    tc = rng.uniform(0.5, 4.0, size=n)
    times = np.minimum(np.minimum(t1, t2), tc)
    events = np.zeros(n, dtype=int)
    events[(t1 <= t2) & (t1 <= tc)] = 1
    events[(t2 < t1) & (t2 <= tc)] = 2

    cox_res = fit_cox_ph_efron(x, times, events, target_event=1, coef_names=["driver", "noise"])
    fg_res = fit_fine_gray_ipcw(x, times, events, target_event=1, coef_names=["driver", "noise"])
    assert cox_res.converged and cox_res.beta[0] > 0.4
    assert fg_res.converged and fg_res.beta[0] > 0.3
    assert 1.5 < compute_mdes_interaction_hr(n_events=100, prop_stratum=0.35) < 2.2

def test_wls_hc3_and_highdim_corr():
    rng = np.random.default_rng(7)
    n = 60
    g = rng.normal(size=n)
    frac = rng.uniform(0.2, 0.9, size=n)
    y = 1.5 + 0.8 * g + rng.normal(0.0, 0.3 / frac, size=n)
    wls = fit_wls_hc3(g, y, weights=frac**2, add_intercept=True, coef_names=["genotype"])
    assert abs(wls.beta[1] - 0.8) < 0.25

    X = rng.normal(size=(40, 150))
    X[:, 0] = 2.0 * g[:40] + rng.normal(0.0, 0.2, size=40)
    assert spearman_matrix(X, g[:40])[0] > 0.85
    _, _, emp_fdr = permutation_empirical_fdr(X, g[:40], n_perm=500, random_state=0)
    assert emp_fdr[0] < 0.05
    r_part, p_part = partial_corr_pval(X[:, 0], g[:40], covariates=X[:, 1])
    assert r_part > 0.85 and p_part < 1e-6
    with pytest.raises(ValueError, match="Non-finite"):
        ttest_ind_safe([1.0, np.nan, 2.0], [1.0, 2.0, 3.0])
