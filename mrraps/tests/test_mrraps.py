from __future__ import annotations

import warnings
import numpy as np
import pytest
from mrraps import mr_raps, mr_raps_shrinkage, raps_mle, raps_shrinkage


def test_mr_raps_huber_and_tukey_with_outliers():
    rng = np.random.default_rng(21)
    m = 120
    beta_true = 0.45
    gamma = rng.normal(0.0, 0.05, m)
    se_exp = np.full(m, 0.015)
    se_out = np.full(m, 0.02)
    pleio = rng.normal(0.0, 0.025, m)
    pleio[:3] += np.array([0.25, -0.30, 0.28])  # idiosyncratic outliers

    b_exp = gamma + rng.normal(0.0, se_exp)
    b_out = beta_true * gamma + pleio + rng.normal(0.0, se_out)

    for loss in ("huber", "tukey"):
        fit = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function=loss, seed=5)
        assert abs(fit["beta_hat"] - beta_true) < 0.12
        assert fit["beta_se"] > 0
        assert fit["tau2_hat"] >= 0.0


def test_mr_raps_scale_invariance_100x():
    rng = np.random.default_rng(104)
    m = 100
    beta_true = 0.60
    gamma = rng.normal(0.0, 0.06, m)
    se_exp = rng.uniform(0.01, 0.02, m)
    se_out = rng.uniform(0.015, 0.03, m)
    pleio = rng.normal(0.0, 0.035, m)  # genuine overdispersion tau2 > 0

    b_exp = gamma + rng.normal(0.0, se_exp)
    b_out = beta_true * gamma + pleio + rng.normal(0.0, se_out)

    for estimator in ("shrinkage", "mle"):
        for loss in ("l2", "huber", "tukey"):
            f1 = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function=loss, estimator=estimator, seed=9)
            # Rescaling both exposure and outcome by 100x leaves beta_hat identical
            f_both = mr_raps(b_exp * 100.0, b_out * 100.0, se_exp * 100.0, se_out * 100.0, over_dispersion=True, loss_function=loss, estimator=estimator, seed=9)
            np.testing.assert_allclose(f_both["beta_hat"], f1["beta_hat"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(f_both["beta_se"], f1["beta_se"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(f_both["tau2_hat"], f1["tau2_hat"] * 1e4, rtol=1e-7, atol=1e-9)

            # Rescaling only outcome (b_out, se_out) by 100x scales beta_hat by 100x and tau2_hat by 10,000x
            f_out = mr_raps(b_exp, b_out * 100.0, se_exp, se_out * 100.0, over_dispersion=True, loss_function=loss, estimator=estimator, seed=9)
            np.testing.assert_allclose(f_out["beta_hat"] / 100.0, f1["beta_hat"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(f_out["beta_se"] / 100.0, f1["beta_se"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(f_out["tau2_hat"] / 1e4, f1["tau2_hat"], rtol=1e-7, atol=1e-9)


def test_mr_raps_mle_no_overdispersion_and_pruning_warning():
    rng = np.random.default_rng(55)
    m = 80
    beta_true = -0.35
    gamma = rng.normal(0.0, 0.07, m)
    se_exp = np.full(m, 0.012)
    se_out = np.full(m, 0.015)
    b_exp = gamma + rng.normal(0.0, se_exp)
    b_out = beta_true * gamma + rng.normal(0.0, se_out)

    for estimator in ("shrinkage", "mle"):
        for loss in ("l2", "huber", "tukey"):
            fit = mr_raps(
                b_exp,
                b_out,
                se_exp,
                se_out,
                over_dispersion=False,
                loss_function=loss,
                estimator=estimator,
                seed=3,
            )
            assert abs(fit["beta_hat"] - beta_true) < 0.08
            assert fit["tau2_hat"] == 0.0
            assert fit["beta_se"] > 0.0

    # Extraordinarily large se_exp triggers warnings.warn (not stderr print)
    se_exp_bad = se_exp.copy()
    se_exp_bad[0] = 0.50
    with pytest.warns(UserWarning, match="Pruning 1 SNP"):
        fit_pruned = mr_raps(b_exp, b_out, se_exp_bad, se_out, over_dispersion=True, estimator="mle", loss_function="huber")
    assert abs(fit_pruned["beta_hat"] - beta_true) < 0.08

    with pytest.raises(ValueError):
        mr_raps(b_exp, b_out, se_exp, se_out, estimator="invalid")


def test_r_default_settings():
    rng = np.random.default_rng(314)
    m = 100
    beta_true = 0.50
    gamma = rng.normal(0.0, 0.06, m)
    se_exp = np.full(m, 0.015)
    se_out = np.full(m, 0.020)
    pleio = rng.normal(0.0, 0.035, m)  # tau2_true ~ 0.001225
    b_exp = gamma + rng.normal(0.0, se_exp)
    b_out = beta_true * gamma + pleio + rng.normal(0.0, se_out)

    # 1. Invalid settings raises ValueError across entrypoints
    with pytest.raises(ValueError, match="Unknown settings"):
        mr_raps(b_exp, b_out, se_exp, se_out, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        mr_raps_shrinkage(b_exp, b_out, se_exp, se_out, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        raps_shrinkage(b_exp, b_out, se_exp, se_out, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        raps_mle(b_exp, b_out, se_exp, se_out, settings="invalid")  # type: ignore[arg-type]

    # 2. On standard GWAS SE scale (~0.02), tau2_hat ~ 1.2e-3 << 10 * median(se_out) = 0.2 -> identical estimates
    fit_def = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function="huber", seed=7, settings="default")
    fit_r = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function="huber", seed=7, settings="R_default")
    assert fit_def["tau2_hat"] > 5e-4
    np.testing.assert_allclose(fit_r["beta_hat"], fit_def["beta_hat"], rtol=1e-12)
    np.testing.assert_allclose(fit_r["tau2_hat"], fit_def["tau2_hat"], rtol=1e-12)

    # 3. Rescaling outcome by 10,000x gives median(se_out) = 200.0 and tau2_hat ~ 1.2e5:
    #    - default checks tau2 <= 100 * median(se_out^2) = 4e6 -> retains positive tau2_hat!
    #    - R_default checks tau2 <= 10 * median(se_out) = 2000.0 -> rejects root and falls back to tau2_hat = 0!
    scale = 10_000.0
    fit_def_big = mr_raps(
        b_exp, b_out * scale, se_exp, se_out * scale, over_dispersion=True, loss_function="huber", seed=7, settings="default"
    )
    assert fit_def_big["tau2_hat"] > 2_000.0
    with pytest.warns(UserWarning, match="Cannot find solution with finite overdispersion"):
        fit_r_big = mr_raps(
            b_exp, b_out * scale, se_exp, se_out * scale, over_dispersion=True, loss_function="huber", seed=7, settings="R_default"
        )
    assert fit_r_big["tau2_hat"] == 0.0
    assert fit_r_big["tau2_se"] == 0.0
