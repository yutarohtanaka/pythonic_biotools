from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from logistf import fit_firth_logistic


def test_firth_logistic_handles_complete_separation():
    x = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
    y = (x > 0).astype(int)
    res = fit_firth_logistic(x, y, add_intercept=True, coef_names=["biomarker"])
    assert res.converged is True
    assert np.all(np.isfinite(res.beta))
    assert 0.5 < res.beta[1] < 10.0
    assert res.p_value[1] < 0.05
    assert res.ci_low_or[1] > 1.0 and np.isfinite(res.ci_high_or[1])


def test_quasi_complete_and_complete_separation_profile_ci_and_plrt():
    # 1. Quasi-complete separation from Heinze & Schemper (2002) 2x2 table:
    # x=0 -> 3 zeros, 0 ones; x=1 -> 1 zero, 4 ones
    y_quasi = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x_quasi = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=float)

    res_prof = fit_firth_logistic(x_quasi, y_quasi, ci_method="profile", pl=True)
    res_wald = fit_firth_logistic(x_quasi, y_quasi, ci_method="wald", pl=False)

    assert res_prof.converged is True
    # Exact analytical Firth solution: p0 = 0.5/4 = 1/8, p1 = 4.5/6 = 3/4 -> OR = (3/1)/(1/7) = 21
    assert res_prof.beta[0] == pytest.approx(-np.log(7.0), rel=1e-6)
    assert res_prof.beta[1] == pytest.approx(np.log(21.0), rel=1e-6)
    assert res_prof.odds_ratio[1] == pytest.approx(21.0, rel=1e-5)

    # Exact analytical PLRT deviance:
    ll_hat = 0.5 * np.log(1 / 8) + 3.5 * np.log(7 / 8) + 4.5 * np.log(3 / 4) + 1.5 * np.log(1 / 4)
    ll_null = 10.0 * np.log(0.5)
    expected_p_plrt = float(stats.chi2.sf(2.0 * (ll_hat - ll_null), df=1))
    assert res_prof.p_value[1] == pytest.approx(expected_p_plrt, rel=1e-5)
    assert res_prof.p_value[1] < 0.05

    # Profile CI is asymmetric on beta scale and strictly excludes OR = 1 (dual to PLRT p < 0.05),
    # whereas symmetric Wald fails to reject H0 (ci_low_or < 1, p > 0.10) and inflates upper OR!
    assert res_prof.ci_low_or[1] > 1.0
    assert np.isfinite(res_prof.ci_high_or[1])
    upper_dist = res_prof.ci_high_beta[1] - res_prof.beta[1]
    lower_dist = res_prof.beta[1] - res_prof.ci_low_beta[1]
    assert upper_dist > 1.25 * lower_dist  # right-skewed profile likelihood under separation at x=0
    assert res_wald.ci_low_or[1] < 1.0 and res_wald.p_value[1] > 0.10

    # 2. Full complete separation: y = [0,0,0,0,1,1,1,1], x = [0,0,0,0,1,1,1,1]
    # Exact analytical Firth solution: p0 = 0.5/5 = 0.1, p1 = 4.5/5 = 0.9 -> OR = 9 / (1/9) = 81
    x_sep = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    res_sep = fit_firth_logistic(x_sep, y_quasi, ci_method="profile", pl=True)
    res_sep_wald = fit_firth_logistic(x_sep, y_quasi, ci_method="wald", pl=False)
    assert res_sep.converged is True
    assert res_sep.beta[1] == pytest.approx(np.log(81.0), rel=1e-6)
    expected_sep_p = float(stats.chi2.sf(2.0 * (2.0 * (0.5 * np.log(0.1) + 4.5 * np.log(0.9)) - 10.0 * np.log(0.5)), df=1))
    assert res_sep.p_value[1] == pytest.approx(expected_sep_p, rel=1e-5)
    assert res_sep.p_value[1] < 0.01
    assert res_sep.ci_low_or[1] > 2.5 and np.isfinite(res_sep.ci_high_or[1])
    # Symmetric Wald upper CI explodes vastly beyond lower distance, while profile CI is properly asymmetric
    assert res_sep_wald.ci_low_or[1] < res_sep.ci_low_or[1]

    # Auto-detection when caller already supplies an intercept column [1, x]
    X_with_const = np.column_stack([np.ones_like(x_sep), x_sep])
    res_auto = fit_firth_logistic(X_with_const, y_quasi)
    assert res_auto.beta.shape == (2,)
    np.testing.assert_allclose(res_auto.beta, res_sep.beta, rtol=1e-6)
