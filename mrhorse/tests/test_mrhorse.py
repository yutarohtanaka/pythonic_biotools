from __future__ import annotations

import numpy as np
from mrhorse import mr_horse, mvmr_horse


def test_mr_horse_and_mvmr_horse():
    rng = np.random.default_rng(13)
    n = 40
    bx_true = rng.normal(0.0, 0.12, (n, 2))
    theta_true = np.array([0.40, -0.25])
    pleio = np.zeros(n)
    pleio[:3] = np.array([0.15, -0.18, 0.12])
    sx = np.full((n, 2), 0.015)
    sy = np.full(n, 0.02)

    bx = bx_true + rng.normal(0.0, sx)
    by_uni = 0.40 * bx_true[:, 0] + pleio + rng.normal(0.0, sy)
    by_mv = bx_true @ theta_true + pleio + rng.normal(0.0, sy)

    r_uni = mr_horse(by_uni, sy, bx[:, 0], sx[:, 0], n_chains=2, n_iter=300, n_burnin=300, seed=1)
    assert abs(r_uni["estimate"][0]["Estimate"] - 0.40) < 0.12

    r_mv = mvmr_horse(by_mv, sy, bx, sx, n_chains=2, n_iter=300, n_burnin=300, seed=2)
    est = np.array([e["Estimate"] for e in r_mv["estimate"]])
    assert np.allclose(est, theta_true, atol=0.15)
