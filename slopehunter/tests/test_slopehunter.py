from __future__ import annotations

import numpy as np
import pandas as pd
from slopehunter import harmonise_effects, hunt, sh_adj, shclust


def test_slopehunter_recovery_and_harmonisation():
    rng = np.random.default_rng(42)
    m = 300
    b_true = -0.35
    xbeta_true = np.concatenate([rng.normal(0.0, 0.08, 240), rng.normal(0.0, 0.08, 60)])
    direct_y = np.concatenate([np.zeros(240), rng.normal(0.0, 0.12, 60)])
    xse = np.full(m, 0.01)
    yse = np.full(m, 0.012)
    xbeta = xbeta_true + rng.normal(0.0, xse)
    ybeta = direct_y + b_true * xbeta_true + rng.normal(0.0, yse)

    res = hunt(xbeta, xse, ybeta, yse, xp_thresh=0.05, bootstrapping=True, M=30, seed=11)
    assert abs(res["b"] - b_true) < 0.08
    yb_adj, ys_adj = sh_adj(res, xbeta, xse, ybeta, yse)
    assert yb_adj.shape == (m,)
    assert np.all(ys_adj > yse)

    x_df = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4_strand", "rs5_strand_swap"],
            "EA": ["A", "A", "A", "A", "A"],
            "OA": ["G", "G", "T", "G", "G"],
            "BETA": [0.1, 0.2, 0.3, 0.15, 0.25],
            "SE": [0.02] * 5,
        }
    )
    y_df = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4_strand", "rs5_strand_swap"],
            "EA": ["A", "G", "A", "T", "C"],
            "OA": ["G", "A", "T", "C", "T"],
            "BETA": [0.05, -0.1, 0.1, 0.07, -0.12],
            "SE": [0.03] * 5,
        }
    )
    harm = harmonise_effects(x_df, y_df)
    assert list(harm["SNP"]) == ["rs1", "rs2", "rs4_strand", "rs5_strand_swap"]
    np.testing.assert_allclose(harm["ybeta"].to_numpy(), [0.05, 0.10, 0.07, 0.12])


def test_slopehunter_negative_loglkl_large_scale_traits():
    """Verify EM does NOT abort prematurely at iter=20 when trait SD > 0.5 makes loglkl < 0."""
    rng = np.random.default_rng(88)
    m = 350
    b_true = -0.45
    # Scale SD ~ 4.0 so bivariate Gaussian density < 1 and loglkl << 0
    xbeta_true = np.concatenate([rng.normal(0.0, 4.0, 270), rng.normal(0.0, 4.0, 80)])
    direct_y = np.concatenate([np.zeros(270), rng.normal(0.0, 6.0, 80)])
    xse = np.full(m, 0.4)
    yse = np.full(m, 0.5)
    xbeta = xbeta_true + rng.normal(0.0, xse)
    ybeta = direct_y + b_true * xbeta_true + rng.normal(0.0, yse)

    fit = shclust(xbeta, ybeta, rng=np.random.default_rng(19))
    assert fit["loglkl"] < 0.0
    assert fit["iter"] > 20  # Legacy bug exited immediately at iter == 20 whenever loglkl < 0!
    assert abs(fit["b"] - b_true) < 0.08
    pt = fit["Fit"]["pt"].to_numpy()
    assert np.all((pt >= 1e-7) & (pt <= 1.0 - 1e-7))


def test_slopehunter_signed_bootstrap_se_at_null():
    """Verify bootstrap SE uses signed bts (no 40% downward folded-normal bias near b = 0)."""
    rng = np.random.default_rng(314)
    m = 250
    xbeta_true = rng.normal(0.0, 0.12, m)
    direct_y = np.concatenate([np.zeros(190), rng.normal(0.0, 0.10, 60)])
    xse = np.full(m, 0.01)
    yse = np.full(m, 0.015)
    xbeta = xbeta_true + rng.normal(0.0, xse)
    ybeta = direct_y + 0.0 * xbeta_true + rng.normal(0.0, yse)

    res = hunt(xbeta, xse, ybeta, yse, xp_thresh=0.05, bootstrapping=True, M=50, seed=23)
    bts = res["Bts_est"]
    # Bootstrap draws straddle 0 with both signs; SE must match signed sample SD
    assert np.any(bts > 0) and np.any(bts < 0)
    np.testing.assert_allclose(res["bse"], np.std(bts, ddof=1), rtol=1e-13)
    assert res["bse"] > np.std(np.abs(bts), ddof=1) * 1.15
