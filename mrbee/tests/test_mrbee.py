from __future__ import annotations

import numpy as np
from mrbee import fit_mrbee


def test_mrbee_recovery_with_outliers():
    rng = np.random.default_rng(41)
    m = 500
    b_true = -0.35
    x_true = np.zeros(m)
    x_true[:60] = rng.normal(0.0, 0.12, 60)
    xse = np.full(m, 0.015)
    yse = np.full(m, 0.018)
    pleio = np.zeros(m)
    pleio[:2] = np.array([0.35, -0.40])

    xb = x_true + rng.normal(0.0, xse)
    yb = b_true * x_true + pleio + rng.normal(0.0, yse)

    res = fit_mrbee(xb, xse, yb, yse, inst_p=1e-3, rho_e=0.0)
    assert abs(res["b"] - b_true) < 0.08
    assert res["n_outliers"] >= 1
