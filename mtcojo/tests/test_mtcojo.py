from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from mtcojo import (
    align_sumstats_to_bim,
    cond_multiple,
    genotype_corr,
    gsmr_meta,
    mtcojo,
    mtcojo_ldsc,
    pchisqsum,
)


def test_mtcojo_ldsc_and_conditioning():
    rng = np.random.default_rng(61)
    m = 100_000
    n_ind = 100_000.0
    l2 = rng.uniform(1.0, 40.0, m)
    h2_y, h2_x, rg = 0.25, 0.30, 0.50
    cov_g = rg * np.sqrt(h2_y * h2_x)

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


def test_pchisqsum_moderate_and_extreme_tails():
    lam_eq = np.ones(5)
    for x in (8.0, 20.0, 60.0, 500.0):
        p_est = pchisqsum(x, lam_eq)
        p_exact = float(stats.chi2.sf(x, df=5))
        np.testing.assert_allclose(p_est, p_exact, rtol=5e-2)

    # Extreme tail x = 500 and x = 1_000_000 (exceeding _Kp(lmax)) must give p ~ 0, NOT p ~ 1!
    lam_w = np.array([2.4, 1.3, 0.8, 0.35, 0.15])
    p_500 = pchisqsum(500.0, lam_w)
    assert 0.0 <= p_500 < 1e-40
    p_huge = pchisqsum(1_000_000.0, lam_eq)
    assert 0.0 <= p_huge < 1e-250
    assert pchisqsum(0.0, lam_eq) == 1.0


def test_gsmr_meta_v2_heidi_outliers_and_all_nan_genotype_handling():
    rng = np.random.default_rng(707)
    n_ind = 400
    m = 40
    bxy_true = 0.45

    # Simulate independent SNPs spaced >10 Mb apart
    maf = rng.uniform(0.2, 0.45, m)
    G = rng.binomial(2, maf[None, :], size=(n_ind, m)).astype(float)
    G[:, -1] = np.nan  # all-NaN genotype column to test safe handling

    corr_mat = genotype_corr(G)
    assert np.all(np.isfinite(corr_mat))
    assert corr_mat[-1, -1] == 1.0 and np.all(corr_mat[-1, :-1] == 0.0)

    bzx_true = np.zeros(m)
    bzx_true[:24] = rng.uniform(0.10, 0.18, 24) * rng.choice([-1.0, 1.0], 24)
    bzx_se = np.full(m, 0.015)
    bzy_se = np.full(m, 0.020)

    pleio = np.zeros(m)
    pleio[0] = 0.35
    pleio[1] = -0.38

    bzx = bzx_true + rng.normal(0.0, bzx_se)
    bzy = bxy_true * bzx_true + pleio + rng.normal(0.0, bzy_se)
    bzx_p = stats.chi2.sf((bzx / bzx_se) ** 2, 1)
    bzy_p = stats.chi2.sf((bzy / bzy_se) ** 2, 1)

    valid = np.ones(m, dtype=bool)
    valid[-1] = False
    chrom = np.ones(m, dtype=int)
    bp = np.arange(m, dtype=float) * 20_000_000.0  # > 10 Mb apart

    res = gsmr_meta(
        bzx,
        bzx_se,
        bzx_p,
        bzy,
        bzy_se,
        bzy_p,
        valid,
        chrom,
        bp,
        G,
        nsnp_min=10,
    )
    assert res["err"] is None
    assert abs(res["bxy"] - bxy_true) < 0.08
    assert 0 in res["pleio"] and 1 in res["pleio"]
    assert res["heidi_p"] >= 0.01


def test_cond_multiple_ncov2_and_interleaved_chrom_check():
    rng = np.random.default_rng(808)
    s = 60
    bzy = rng.normal(0.0, 0.05, s)
    bzy_se = np.full(s, 0.02)
    bzx = rng.normal(0.0, 0.06, (s, 2))
    bzx_se = np.full((s, 2), 0.015)
    bxy = np.array([0.35, -0.25])

    ldsc_intercept = np.array(
        [
            [1.02, 0.15, -0.10],
            [0.15, 1.01, 0.20],
            [-0.10, 0.20, 1.03],
        ]
    )
    ldsc_slope = np.array(
        [
            [0.25, 0.30, -0.20],
            [0.30, 0.35, 0.40],
            [-0.20, 0.40, 0.28],
        ]
    )
    vp = np.array([1.0, 1.2, 0.9])

    b_cond, se_cond = cond_multiple(bzy, bzy_se, bzx, bzx_se, bxy, ldsc_intercept, ldsc_slope, vp)
    assert b_cond.shape == (s,) and se_cond.shape == (s,)
    assert np.all(np.isfinite(b_cond)) and np.all(se_cond > 0.0)

    # Verify interleaved chromosomes raise ValueError in mtcojo when GSMR clumping runs
    G = rng.binomial(2, 0.3, size=(100, 12)).astype(float)
    df_interleaved = pd.DataFrame(
        {
            "SNP": [f"rs{i}" for i in range(12)],
            "CHR": [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3],
            "BP": [100, 200, 100, 200, 300, 400, 300, 400, 100, 200, 300, 400],
            "XBETA": np.full(12, 0.2),
            "XSE": np.full(12, 0.02),
            "YBETA": np.full(12, 0.1),
            "YSE": np.full(12, 0.02),
        }
    )
    with pytest.raises(ValueError, match="sorted by CHR then BP"):
        mtcojo(df_interleaved, G=G)


def test_align_sumstats_to_bim():
    bim_df = pd.DataFrame(
        {
            "CHR": [1, 1, 1, 1, 1, 1],
            "SNP": ["rs_same", "rs_swap", "rs_comp", "rs_comp_swap", "rs_pal", "rs_bad"],
            "CM": 0.0,
            "BP": [1000, 2000, 3000, 4000, 5000, 6000],
            "A1": ["A", "A", "A", "A", "A", "A"],
            "A2": ["G", "G", "G", "G", "T", "G"],
        }
    )
    G = np.array(
        [
            [0.0, 2.0, 1.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 2.0],
        ]
    )
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs_comp_swap", "rs_pal", "rs_swap", "rs_same", "rs_bad"],
            "A1": ["C", "T", "G", "A", "A"],
            "A2": ["T", "A", "A", "G", "C"],
            "FREQ": [0.75, 0.50, 0.80, 0.25, 0.30],
            "XBETA": [-0.20, 0.10, -0.15, 0.12, 0.05],
            "XSE": [0.02] * 5,
            "XBETA2": [-0.30, 0.10, -0.25, 0.18, 0.05],
            "XSE2": [0.02] * 5,
            "YBETA": [-0.10, 0.05, -0.08, 0.06, 0.02],
            "YSE": [0.02] * 5,
        }
    )

    df_aligned, G_aligned = align_sumstats_to_bim(sumstats, bim_df, G)
    # Ordered by bim_df; rs_pal (A/T palindromic) and rs_bad (A/C vs A/G) dropped
    assert list(df_aligned["SNP"]) == ["rs_same", "rs_swap", "rs_comp_swap"]
    assert list(df_aligned["A1"]) == ["A", "A", "A"]
    assert list(df_aligned["A2"]) == ["G", "G", "G"]
    np.testing.assert_allclose(df_aligned["FREQ"].to_numpy(), [0.25, 0.20, 0.25])
    np.testing.assert_allclose(df_aligned["XBETA"].to_numpy(), [0.12, 0.15, 0.20])
    np.testing.assert_allclose(df_aligned["XBETA2"].to_numpy(), [0.18, 0.25, 0.30])
    np.testing.assert_allclose(df_aligned["YBETA"].to_numpy(), [0.06, 0.08, 0.10])
    np.testing.assert_allclose(G_aligned, G[:, [0, 1, 3]])

    # Also test align_target='sumstats' (flips genotypes 2 - G on swapped SNPs instead of betas)
    df_s, G_s = align_sumstats_to_bim(sumstats, bim_df, G, align_target="sumstats")
    np.testing.assert_allclose(df_s["XBETA"].to_numpy(), [0.12, -0.15, -0.20])
    np.testing.assert_allclose(G_s[:, 0], G[:, 0])
    np.testing.assert_allclose(G_s[:, 1], 2.0 - G[:, 1])
    np.testing.assert_allclose(G_s[:, 2], 2.0 - G[:, 3])
