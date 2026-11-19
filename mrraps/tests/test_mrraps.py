from __future__ import annotations

import numpy as np
from mrraps import mr_raps


def test_mr_raps_huber_and_tukey_with_outliers():
    rng = np.random.default_rng(21)
    m = 120
    beta_true = 0.45
    gamma = rng.normal(0.0, 0.05, m)
    se_exp = np.full(m, 0.015)
    se_out = np.full(m, 0.02)
    pleio = rng.normal(0.0, 0.01, m)
    pleio[:3] += np.array([0.25, -0.30, 0.28])  # idiosyncratic outliers

    b_exp = gamma + rng.normal(0.0, se_exp)
    b_out = beta_true * gamma + pleio + rng.normal(0.0, se_out)

    for loss in ("huber", "tukey"):
        fit = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function=loss, seed=5)
        assert abs(fit["beta_hat"] - beta_true) < 0.12
        assert fit["beta_se"] > 0
