from __future__ import annotations

import numpy as np
from mvcml import mvcml_adjust


def test_mvcml_adjust_multivariable():
    rng = np.random.default_rng(51)
    s = 80
    bx_true = np.zeros((s, 2))
    bx_true[:25] = rng.normal(0.0, 0.15, (25, 2))
    b_true = np.array([0.35, -0.25])
    sx = np.full((s, 2), 0.012)
    sy = np.full(s, 0.015)
    pleio = np.zeros(s)
    pleio[0] = 0.25

    bx = bx_true + rng.normal(0.0, sx)
    byc = bx_true @ b_true + pleio + rng.normal(0.0, sy)

    res = mvcml_adjust(bx, sx, byc, sy, N=50_000, iv_p_thresh=1e-4)
    assert np.allclose(res["b"], b_true, atol=0.08)
    assert res["ybeta_adj"].shape == (s,)
