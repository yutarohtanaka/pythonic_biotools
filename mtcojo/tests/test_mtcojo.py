from __future__ import annotations

import numpy as np
import pandas as pd
from mtcojo import mtcojo, mtcojo_ldsc


def test_mtcojo_ldsc_and_conditioning():
    rng = np.random.default_rng(61)
    m = 100_000
    n_ind = 100_000.0
    l2 = rng.uniform(1.0, 40.0, m)
    h2_y, h2_x, rg = 0.25, 0.30, 0.50
    cov_g = rg * np.sqrt(h2_y * h2_x)

    zy = np.empty(m)
    zx = np.empty(m)
    sd_y = np.sqrt(1.0 + n_ind * h2_y * l2 / m)
    sd_x = np.sqrt(1.0 + n_ind * h2_x * l2 / m)
    corr = (n_ind * cov_g * l2 / m) / (sd_y * sd_x)
    e1 = rng.normal(size=m)
    e2 = rng.normal(size=m)
    zy = sd_y * e1
    zx = sd_x * (corr * e1 + np.sqrt(np.maximum(0.0, 1.0 - corr ** 2)) * e2)

    z_mat = np.column_stack([zy, zx])
    n_mat = np.full((m, 2), n_ind)
    inter, slope, ok = mtcojo_ldsc(z_mat, n_mat, l2, l2, float(m))
    assert ok
    assert abs(slope[0, 1] - rg) < 0.12

    sub = 200
    df = pd.DataFrame(
        {
            "SNP": [f"rs{i}" for i in range(sub)],
            "CHR": 1,
            "BP": np.arange(sub) * 1000,
            "XBETA": zx[:sub] * 0.01,
            "XSE": np.full(sub, 0.01),
            "YBETA": zy[:sub] * 0.01,
            "YSE": np.full(sub, 0.01),
            "XN": n_ind,
            "YN": n_ind,
        }
    )
    res = mtcojo(df, bxy_external=[0.40])
    assert np.allclose(res["ybeta_adj"], df["YBETA"] - 0.40 * df["XBETA"])
