from __future__ import annotations

import numpy as np
from indexevent import indexevent


def test_indexevent_cwls_hedges_olkin_and_simex():
    rng = np.random.default_rng(19)
    m = 200
    b_true = -0.40
    x_true = rng.normal(0.0, 0.06, m)
    xse = np.full(m, 0.015)
    yse = np.full(m, 0.015)
    xb = x_true + rng.normal(0.0, xse)
    yb = b_true * x_true + rng.normal(0.0, yse)

    for method in ("CWLS", "Hedges-Olkin", "Simex"):
        res = indexevent(xb, xse, yb, yse, method=method, B=5, seed=7)
        assert abs(res["b"] - b_true) < 0.08
        assert res["ybeta_adj"].shape == (m,)
