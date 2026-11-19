from __future__ import annotations

import numpy as np
from mrhorse import gelman_rubin, mr_horse, mvmr_horse
from mrhorse.core import _qform, _trunc_norm


def test_trunc_norm_extreme_tails_never_violates_bounds():
    rng = np.random.default_rng(101)
    # Extreme left and right >50 sigma tails
    draws_hi = np.array([_trunc_norm(mean=25.0, sd=0.1, lo=-10.0, hi=10.0, rng=rng) for _ in range(500)])
    draws_lo = np.array([_trunc_norm(mean=-25.0, sd=0.1, lo=-10.0, hi=10.0, rng=rng) for _ in range(500)])
    draws_6sig = np.array([_trunc_norm(mean=0.0, sd=1.0, lo=6.0, hi=8.0, rng=rng) for _ in range(500)])

    assert np.all((draws_hi >= -10.0) & (draws_hi <= 10.0))
    assert np.all((draws_lo >= -10.0) & (draws_lo <= 10.0))
    assert np.all((draws_6sig >= 6.0) & (draws_6sig <= 8.0))
    assert np.mean(draws_hi) > 9.95
    assert np.mean(draws_lo) < -9.95


def test_qform_tau_phi_underflow_guard_and_gelman_rubin():
    bx0 = np.zeros((4, 2))
    mx = np.zeros(2)
    vx0 = np.ones(2)
    s = np.ones(2)
    rho = np.array([[0.3, -0.2]] * 4)
    alpha = np.array([0.0, 1e-80, -1e-80, 0.0])
    phi = np.full(4, 1e-200)
    tau = 1e-200
    qf = _qform(bx0, mx, s, vx0, rho, alpha, phi, tau)
    assert np.all(np.isfinite(qf))

    # gelman_rubin must use all post-burnin draws (with n=3 per chain, halving would leave n=1 and fail ddof=1)
    rng = np.random.default_rng(7)
    short_chains = rng.normal(0.5, 0.02, size=(3, 3))
    rhat_short = gelman_rubin(short_chains)
    assert np.isfinite(rhat_short) and rhat_short > 0.5

    long_chains = rng.normal(1.0, 0.1, size=(4, 500))
    rhat = gelman_rubin(long_chains)
    assert 0.98 < rhat < 1.05


def test_mr_horse_and_mvmr_horse_correlated_pleiotropy_and_rhat():
    rng = np.random.default_rng(13)
    n = 60
    bx_true = rng.normal(0.0, 0.14, (n, 2))
    theta_true = np.array([0.40, -0.25])

    # Correlated pleiotropy affecting first 8 variants differently across exposures 1 & 2
    pleio = np.zeros(n)
    u_conf = rng.normal(0.0, 0.15, 8)
    bx_true[:8, 0] += 0.8 * u_conf
    bx_true[:8, 1] -= 0.7 * u_conf
    pleio[:8] = 1.1 * u_conf

    sx = np.full((n, 2), 0.015)
    sy = np.full(n, 0.02)

    bx = bx_true + rng.normal(0.0, sx)
    by_uni = 0.40 * bx_true[:, 0] + pleio + rng.normal(0.0, sy)
    by_mv = bx_true @ theta_true + pleio + rng.normal(0.0, sy)

    r_uni = mr_horse(
        by_uni,
        sy,
        bx[:, 0],
        sx[:, 0],
        n_chains=2,
        n_iter=600,
        n_burnin=600,
        seed=1,
        save=("theta", "bx0"),
    )
    assert abs(r_uni["estimate"][0]["Estimate"] - 0.40) < 0.10
    assert r_uni["estimate"][0]["Rhat"] < 1.1
    assert "bx0" in r_uni["extra"]
    assert r_uni["extra"]["bx0"][0].shape == (600, n, 1)

    r_mv = mvmr_horse(
        by_mv,
        sy,
        bx,
        sx,
        n_chains=2,
        n_iter=600,
        n_burnin=600,
        seed=2,
        save=("theta", "rho", "bx0"),
    )
    est = np.array([e["Estimate"] for e in r_mv["estimate"]])
    rhats = np.array([e["Rhat"] for e in r_mv["estimate"]])
    assert np.allclose(est, theta_true, atol=0.12)
    assert np.all(rhats < 1.1)

    # Verify per-exposure rho shape (draws, N, K) and valid positive-definite correlation matrix
    rho_draws = r_mv["extra"]["rho"][0]
    assert rho_draws.shape == (600, n, 2)
    assert not np.allclose(rho_draws[:, :, 0], rho_draws[:, :, 1])
    norm_sq = np.sum(rho_draws ** 2, axis=-1)
    assert np.all(norm_sq < 1.0)
    # Check full (K+1)x(K+1) correlation matrix Cholesky on final draw
    last_rho = rho_draws[-1]
    for i in range(n):
        corr_i = np.eye(3)
        corr_i[0, 1:] = last_rho[i]
        corr_i[1:, 0] = last_rho[i]
        L = np.linalg.cholesky(corr_i)
        assert np.all(np.diag(L) > 0.0)
