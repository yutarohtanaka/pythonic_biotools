from __future__ import annotations

import numpy as np
from mrapss import fit_mrapss


def test_mrapss_with_sample_overlap():
    rng = np.random.default_rng(31)
    m = 800
    b_true = 0.30
    xse = np.full(m, 0.02)
    yse = np.full(m, 0.02)
    fg = np.zeros(m)
    fg[:80] = rng.normal(0.0, 0.14, 80)
    # Correlated estimation error (sample overlap rho_e = 0.3)
    e1 = rng.normal(0.0, 1.0, m)
    e2 = 0.3 * e1 + np.sqrt(1.0 - 0.09) * rng.normal(0.0, 1.0, m)

    xb = fg + xse * e1
    yb = b_true * fg + yse * e2

    res = fit_mrapss(xb, xse, yb, yse, iv_p_thresh=1e-3)
    assert abs(res["b"] - b_true) < 0.10
    assert res["converged"]
