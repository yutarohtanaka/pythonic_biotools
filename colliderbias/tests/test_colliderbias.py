from __future__ import annotations

import numpy as np
from colliderbias import cwbls, cwls


def test_cwls_and_cwbls():
    rng = np.random.default_rng(7)
    m = 250
    x_true = rng.normal(0.0, 0.08, m)
    d_true = 0.5 * x_true + rng.normal(0.0, 0.06, m)
    y_true = 0.30 * x_true - 0.40 * d_true
    se = np.full(m, 0.015)

    xb = x_true + rng.normal(0.0, se)
    db = d_true + rng.normal(0.0, se)
    yb = y_true + rng.normal(0.0, se)

    r_cb = cwls(db, se, -0.40 * d_true + rng.normal(0.0, se), se, model="ivw")
    assert abs(r_cb["b"] - (-0.40)) < 0.06

    r_biv = cwbls(xb, se, db, se, yb, se, n_boot=50, seed=3)
    assert abs(r_biv["b"] - 0.30) < 0.08
