from __future__ import annotations

import numpy as np
import pandas as pd
from slopehunter import harmonise_effects, hunt, sh_adj


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

    x_df = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3"], "EA": ["A", "A", "A"], "OA": ["G", "G", "T"], "BETA": [0.1, 0.2, 0.3], "SE": [0.02, 0.02, 0.02]})
    y_df = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3"], "EA": ["A", "G", "A"], "OA": ["G", "A", "T"], "BETA": [0.05, -0.1, 0.1], "SE": [0.03, 0.03, 0.03]})
    harm = harmonise_effects(x_df, y_df)
    assert list(harm["SNP"]) == ["rs1", "rs2"]  # rs3 A/T palindromic dropped
    assert np.isclose(harm.loc[1, "ybeta"], 0.1)  # swapped allele sign flipped
