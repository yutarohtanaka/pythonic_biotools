from __future__ import annotations

import numpy as np
import pandas as pd
from evam import fit_frequency, fit_oncobn, fit_ot, order_from_theta


def test_evam_published_models():
    rng = np.random.default_rng(123)
    n = 180
    a = rng.binomial(1, 0.70, size=n)
    b = np.where(a == 1, rng.binomial(1, 0.75, size=n), rng.binomial(1, 0.04, size=n))
    c = np.where((a == 1) & (b == 1), rng.binomial(1, 0.70, size=n), rng.binomial(1, 0.03, size=n))
    d = rng.binomial(1, 0.25, size=n)
    X = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})

    ot = fit_ot(X)
    assert ot.order.loc["A", "B"] == 1.0 and ot.order.loc["B", "A"] == 0.0

    cbn = fit_oncobn(X, model="CBN", k=2, search="dp")
    assert cbn.order.loc["A", "C"] == 1.0

    freq = fit_frequency(X)
    assert freq.order.loc["A", "B"] > 0.5

    theta = np.array([[0.5, 0.0, 0.0], [1.8, -1.2, 0.0], [0.0, 1.8, -1.8]])
    order_df, route = order_from_theta(theta, ["A", "B", "C"])
    assert route == "ctmc_exact" and order_df.loc["A", "B"] > 0.75
