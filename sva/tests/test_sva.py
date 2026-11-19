"""Deterministic unit and statistical validation tests for `sva`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import f_oneway, kstest

from sva import (
    FTestResult,
    NumSVResult,
    SVAResult,
    f_pvalue,
    irwsva,
    num_sv,
    sva,
    svaseq,
    twostepsva,
)
from sva.stats import edge_lfdr


def _subspace_canonical_corrs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return canonical correlations between column spaces of `A` and `B`."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def test_num_sv_be_and_leek_planted_rank_and_null_calibration() -> None:
    rng = np.random.default_rng(42)
    m, n = 600, 30
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])

    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    Y = rng.normal(size=(m, n))
    Y[:80] += 1.5 * treat[None, :]
    Y[80:280] += 1.8 * z1[None, :]
    Y[280:460] += 1.8 * z2[None, :]

    res_be = num_sv(Y, mod, method="be", B=25, seed=42)
    res_leek = num_sv(Y, mod, method="leek")

    assert isinstance(res_be, NumSVResult)
    assert res_be.n_sv == 2
    assert int(res_be) == 2
    assert res_leek.n_sv == 2

    df_be = res_be.to_dataframe()
    assert list(df_be.columns) == [
        "component",
        "singular_value",
        "variance_fraction",
        "p_value",
        "threshold",
        "selected",
    ]
    assert int(df_be["selected"].sum()) == 2
    assert "n_sv=2" in res_be.summary()
    assert "n_sv=2" in res_leek.summary()

    # Pure null (homoscedastic and heteroscedastic)
    Y_null_homo = rng.normal(size=(m, n))
    sigmas = rng.uniform(0.5, 2.0, size=(m, 1))
    Y_null_hetero = rng.normal(size=(m, n)) * sigmas

    assert num_sv(Y_null_homo, mod, method="be", seed=42).n_sv == 0
    assert num_sv(Y_null_homo, mod, method="leek").n_sv == 0
    assert num_sv(Y_null_hetero, mod, method="be", seed=42).n_sv == 0
    assert num_sv(Y_null_hetero, mod, method="leek").n_sv == 0


def test_num_sv_dimensions_n_gt_g_large_p_vfilter_and_rank_deficient_mod() -> None:
    rng = np.random.default_rng(1000)

    # N > G (60 samples, 20 genes, p=3)
    n_ng, m_ng = 60, 20
    mod_ng = np.column_stack([np.ones(n_ng), rng.normal(size=(n_ng, 2))])
    Y_ng_null = rng.normal(size=(m_ng, n_ng))
    assert num_sv(Y_ng_null, mod_ng, method="be", B=50, sig_level=0.05, seed=1000).n_sv == 0
    assert num_sv(Y_ng_null, mod_ng, method="leek").n_sv == 0

    z_ng = rng.normal(size=n_ng)
    Y_ng_sig = Y_ng_null.copy()
    Y_ng_sig[:12] += 2.0 * z_ng[None, :]
    assert num_sv(Y_ng_sig, mod_ng, method="be", seed=1000).n_sv == 1
    assert num_sv(Y_ng_sig, mod_ng, method="leek").n_sv == 1

    # G >> N with large p (n=20, p=8) under pure null: checks post-permutation (I - H_mod)
    n_lp, m_lp = 20, 200
    mod_lp = np.column_stack([np.ones(n_lp), rng.normal(size=(n_lp, 7))])
    Y_lp_null = rng.normal(size=(m_lp, n_lp))
    res_lp = num_sv(Y_lp_null, mod_lp, method="be", B=50, sig_level=0.05, seed=7)
    assert res_lp.n_sv == 0
    assert res_lp.residual_df == n_lp - 8

    # Rank-deficient mod (duplicated column) + vfilter
    mod_dup = np.column_stack([mod_ng, mod_ng[:, 1]])
    res_dup = num_sv(Y_ng_sig, mod_dup, method="leek", vfilter=15)
    assert res_dup.residual_df == n_ng - 3
    assert res_dup.n_features_used == 15
    assert res_dup.n_sv == 1


def test_irwsva_planted_batch_recovery_and_posterior_weights() -> None:
    rng = np.random.default_rng(2025)
    m, n = 800, 24
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))

    batch = rng.normal(size=n)
    batch -= treat * (np.dot(batch, treat - treat.mean()) / np.dot(treat - treat.mean(), treat - treat.mean()))
    batch = (batch - batch.mean()) / batch.std()

    Y = rng.normal(size=(m, n))
    Y[:100] += 2.2 * treat[None, :]         # bio-only (0..99)
    Y[100:300] += 2.2 * batch[None, :]      # batch-only (100..299)
    Y[300:350] += 1.8 * treat[None, :] + 1.8 * batch[None, :]  # both (300..349)
    # null: 350..799

    res = irwsva(Y, mod, mod0, n_sv=1, B=5, seed=2025)
    assert isinstance(res, SVAResult)
    assert res.method == "irw"
    assert res.n_sv == 1
    assert res.sv.shape == (n, 1)
    assert res.pprob_b is not None

    r = abs(float(np.corrcoef(res.sv[:, 0], batch)[0, 1]))
    assert r > 0.95

    w_batch = float(np.mean(res.weights[100:300]))
    w_bio = float(np.mean(res.weights[:100]))
    pb_bio = float(np.mean(res.pprob_b[:100]))
    pb_null = float(np.mean(res.pprob_b[350:]))
    assert w_batch > w_bio + 0.5
    assert pb_bio > pb_null + 0.5

    post_df = res.feature_posteriors_dataframe()
    assert list(post_df.columns) == ["feature", "pprob_gam", "pprob_b", "weight"]
    assert len(post_df) == m


def test_irwsva_correlated_batch_restores_f_pvalue_calibration() -> None:
    rng = np.random.default_rng(99)
    m, n = 800, 30
    treat = np.repeat([0.0, 1.0], n // 2)
    x_c = (treat - treat.mean()) / treat.std()

    z_orth = rng.normal(size=n)
    z_orth = z_orth - x_c * (np.dot(z_orth, x_c) / np.dot(x_c, x_c))
    z_orth = z_orth / z_orth.std()
    # Confounded latent batch factor correlated with treatment (r = 0.50)
    z = 0.50 * x_c + np.sqrt(1.0 - 0.50 ** 2) * z_orth

    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))

    Y = rng.normal(size=(m, n))
    Y[:80] += 2.0 * x_c[None, :]        # true DE genes
    Y[80:380] += 2.0 * z[None, :]       # batch-only confounded genes (b_g = 0)
    # 380:800 null genes

    res = irwsva(Y, mod, mod0, n_sv=1, B=5, seed=99)
    r_z = abs(float(np.corrcoef(res.sv[:, 0], z)[0, 1]))
    assert r_z > 0.95

    # Compare against naive residual PCA (strictly orthogonal to mod)
    Q1, _ = np.linalg.qr(mod)
    R = Y - (Y @ Q1) @ Q1.T
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    r_pca = abs(float(np.corrcoef(Vt[0], z)[0, 1]))
    assert r_z > r_pca

    mod_aug = res.augment_design(mod)
    mod0_aug = res.augment_design(mod0)
    assert np.linalg.matrix_rank(mod_aug) == np.linalg.matrix_rank(mod) + 1

    unadj = f_pvalue(Y, mod, mod0)
    adj = f_pvalue(Y, mod_aug, mod0_aug)

    # Batch-only genes (80:380) should be heavily inflated unadjusted, calibrated after SVA
    fp_batch_unadj = float(np.mean(unadj.q_value[80:380] <= 0.05))
    fp_batch_adj = float(np.mean(adj.q_value[80:380] <= 0.05))
    tpr_adj = float(np.mean(adj.q_value[:80] <= 0.05))
    assert fp_batch_unadj > 0.50
    assert fp_batch_adj < 0.05
    assert tpr_adj > 0.90


def test_twostepsva_and_sva_dispatch_modes() -> None:
    rng = np.random.default_rng(314)
    m, n = 500, 24
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))
    batch = rng.normal(size=n)

    Y = rng.normal(size=(m, n))
    Y[:60] += 2.0 * treat[None, :]
    Y[60:260] += 2.0 * batch[None, :]

    res_direct = twostepsva(Y, mod, mod0, n_sv=1, seed=314)
    res_dispatch = sva(Y, mod, mod0, n_sv=1, method="two-step", seed=314)
    assert res_direct.method == "two-step"
    assert res_direct.pprob_b is None
    np.testing.assert_allclose(res_direct.sv, res_dispatch.sv, atol=1e-12)
    np.testing.assert_allclose(res_direct.weights, res_dispatch.weights, atol=1e-12)
    assert abs(float(np.corrcoef(res_direct.sv[:, 0], batch)[0, 1])) > 0.90

    # Supervised mode using known negative-control batch-affected probes
    controls = np.zeros(m, dtype=np.float64)
    controls[60:260] = 1.0
    res_sup = sva(Y, mod, mod0, n_sv=1, controls=controls, method="supervised")
    assert res_sup.method == "supervised"
    assert res_sup.pprob_b is None
    np.testing.assert_allclose(res_sup.pprob_gam, controls)
    np.testing.assert_allclose(res_sup.weights, controls)
    assert abs(float(np.corrcoef(res_sup.sv[:, 0], batch)[0, 1])) > 0.95

    with pytest.raises(ValueError, match="vfilter"):
        sva(Y, mod, mod0, controls=controls, vfilter=100)
    with pytest.raises(ValueError, match="Unsupported sva method"):
        sva(Y, mod, mod0, method="invalid")


def test_svaseq_count_recovery_library_norm_and_validation() -> None:
    rng = np.random.default_rng(777)
    m, n = 500, 20
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))

    batch = rng.normal(size=n)
    batch = (batch - batch.mean()) / batch.std()
    log_mu = np.full((m, n), 3.0)
    log_mu[:50] += 1.2 * treat[None, :]
    log_mu[50:250] += 1.0 * batch[None, :]

    mu = np.exp(log_mu)
    shape = 10.0
    counts = rng.poisson(rng.gamma(shape, mu / shape)).astype(float)
    counts[-10:] = 0.0  # all-zero genes

    res = svaseq(counts, mod, mod0, n_sv=1, constant=1.0, seed=777)
    assert res.sv.shape == (n, 1)
    assert len(res.feature_names) == m - 10
    assert abs(float(np.corrcoef(res.sv[:, 0], batch)[0, 1])) > 0.90

    # 5x library-size variation handled cleanly with library_size_norm=True
    lib_factors = rng.uniform(0.4, 2.0, size=(1, n))
    counts_lib = rng.poisson(mu * lib_factors).astype(float)
    counts_lib[-5:] = 0.0
    res_lib = svaseq(
        counts_lib, mod, mod0, n_sv=1, constant=1.0, library_size_norm=True, seed=777
    )
    assert abs(float(np.corrcoef(res_lib.sv[:, 0], batch)[0, 1])) > 0.90

    # Validation checks
    bad_counts = counts.copy()
    bad_counts[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        svaseq(bad_counts, mod, mod0)
    with pytest.raises(ValueError, match="constant must be > 0"):
        svaseq(counts, mod, mod0, constant=0.0)


def test_f_pvalue_analytical_parity_invariance_and_nan_bh_fdr() -> None:
    rng = np.random.default_rng(512)
    m, n = 80, 24
    group = np.repeat([0, 1], n // 2)
    mod = np.column_stack([np.ones(n), group.astype(float)])
    mod0 = np.ones((n, 1))

    Y = rng.normal(size=(m, n))
    Y[:15, group == 1] += 1.6

    res = f_pvalue(Y, mod, mod0)
    assert isinstance(res, FTestResult)

    # Exact parity with scipy.stats.f_oneway for two-group OLS
    f_ref = np.empty(m)
    p_ref = np.empty(m)
    for g in range(m):
        fg, pg = f_oneway(Y[g, group == 0], Y[g, group == 1])
        f_ref[g] = fg
        p_ref[g] = pg

    np.testing.assert_allclose(res.f_stat, f_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.p_value, p_ref, rtol=1e-10, atol=1e-12)

    # Invariance to sign flips and KxK orthogonal rotation of surrogate variables
    S = rng.normal(size=(n, 2))
    Q_rot, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    res_s1 = f_pvalue(Y, np.hstack([mod, S]), np.hstack([mod0, S]))
    res_s2 = f_pvalue(Y, np.hstack([mod, S @ Q_rot]), np.hstack([mod0, S @ Q_rot]))
    np.testing.assert_allclose(res_s1.f_stat, res_s2.f_stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_s1.p_value, res_s2.p_value, rtol=1e-12, atol=1e-12)

    # NaN-aware BH-FDR: appending all-NaN rows must NOT inflate q-values of valid rows
    nan_rows = np.full((25, n), np.nan)
    Y_with_nans = np.vstack([Y, nan_rows])
    res_nan = f_pvalue(Y_with_nans, mod, mod0)
    np.testing.assert_allclose(res_nan.q_value[:m], res.q_value, rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(res_nan.p_value[m:]))
    assert np.all(np.isnan(res_nan.q_value[m:]))

    # Partial NaNs in selected rows must match manual OLS on observed samples
    Y_partial = Y.copy()
    Y_partial[0, [0, 3]] = np.nan
    Y_partial[1, [12, 15, 19]] = np.nan
    res_part = f_pvalue(Y_partial, mod, mod0)
    for row_idx in [0, 1]:
        obs = np.isfinite(Y_partial[row_idx])
        fg, pg = f_oneway(Y_partial[row_idx, obs & (group == 0)], Y_partial[row_idx, obs & (group == 1)])
        assert res_part.df2[row_idx] == int(np.sum(obs)) - 2
        np.testing.assert_allclose(res_part.f_stat[row_idx], fg, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(res_part.p_value[row_idx], pg, rtol=1e-10, atol=1e-12)


def test_null_calibration_h0_f_pvalue_and_forced_sv() -> None:
    rng = np.random.default_rng(1024)
    m, n = 400, 30
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))

    Y_null = rng.normal(size=(m, n))
    assert num_sv(Y_null, mod, method="be", seed=1024).n_sv == 0
    assert num_sv(Y_null, mod, method="leek").n_sv == 0

    f_null = f_pvalue(Y_null, mod, mod0)
    type1 = float(np.mean(f_null.p_value <= 0.05))
    assert 0.02 <= type1 <= 0.08
    ks_stat, ks_p = kstest(f_null.p_value, "uniform")
    assert ks_p > 0.01

    # Forced n_sv=2 under pure null still yields finite surrogate variables
    sv_forced = irwsva(Y_null, mod, mod0, n_sv=2, B=3, seed=1024)
    assert sv_forced.sv.shape == (n, 2)
    assert np.all(np.isfinite(sv_forced.sv))
    f_forced = f_pvalue(Y_null, sv_forced.augment_design(mod), sv_forced.augment_design(mod0))
    assert np.all(np.isfinite(f_forced.p_value))


def test_extreme_de_weight_collapse_n_sv_zero_and_boundary_errors() -> None:
    rng = np.random.default_rng(88)
    m, n = 300, 24
    treat = np.repeat([0.0, 1.0], n // 2)
    x_c = (treat - treat.mean()) / treat.std()
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))
    batch = rng.normal(size=n)
    batch = batch - x_c * (np.dot(batch, x_c) / np.dot(x_c, x_c))
    batch = (batch - batch.mean()) / batch.std()

    # (a) 0% DE and 0% batch (weight collapse guard)
    Y_0_0 = rng.normal(size=(m, n))
    res_0_0 = irwsva(Y_0_0, mod, mod0, n_sv=1, B=5, seed=88)
    assert np.all(np.isfinite(res_0_0.sv))
    assert np.linalg.matrix_rank(res_0_0.augment_design(mod)) == 3

    # (b) 0% DE + 1 batch factor
    Y_0_1 = rng.normal(size=(m, n))
    Y_0_1[:150] += 2.0 * batch[None, :]
    res_0_1 = irwsva(Y_0_1, mod, mod0, n_sv=1, B=5, seed=88)
    assert abs(float(np.corrcoef(res_0_1.sv[:, 0], batch)[0, 1])) > 0.95

    # (c) 100% DE + 1 batch factor (pi0_mod -> 0)
    Y_100 = rng.normal(size=(m, n))
    Y_100 += 2.2 * x_c[None, :]
    Y_100[:150] += 2.2 * batch[None, :]
    res_100 = irwsva(Y_100, mod, mod0, n_sv=1, B=5, seed=88)
    assert np.all(np.isfinite(res_100.sv))
    assert np.linalg.matrix_rank(res_100.augment_design(mod)) == 3
    cc = _subspace_canonical_corrs(
        res_100.augment_design(mod), np.column_stack([mod, batch])
    )
    assert float(np.min(cc)) > 0.95

    # Constant p-values in edge_lfdr should not crash KDE
    lfdr_const = edge_lfdr(np.ones(50))
    np.testing.assert_allclose(lfdr_const, 1.0)

    # n_sv == 0 across sva, irwsva, twostepsva
    for fn in (sva, irwsva, twostepsva):
        res0 = fn(Y_0_0, mod, mod0, n_sv=0)
        assert res0.n_sv == 0
        assert res0.sv.shape == (n, 0)
        df0 = res0.to_dataframe()
        assert df0.shape == (n, 0)

    # Non-nested mod0 and excessive n_sv raise ValueError
    bad_mod0 = np.column_stack([np.ones(n), rng.normal(size=n)])
    with pytest.raises(ValueError, match="nested"):
        f_pvalue(Y_0_0, mod, bad_mod0)
    with pytest.raises(ValueError, match="residual df"):
        irwsva(Y_0_0, mod, mod0, n_sv=n - 2)


def test_anndata_dataframe_ndarray_sparse_orientation_parity() -> None:
    rng = np.random.default_rng(404)
    m, n = 200, 18
    treat = np.array(["ctrl"] * 9 + ["stim"] * 9)
    treat_num = (treat == "stim").astype(float)
    mod_arr = np.column_stack([np.ones(n), treat_num])
    mod0_arr = np.ones((n, 1))

    batch = rng.normal(size=n)
    Y = rng.gamma(5.0, 2.0, size=(m, n))
    Y[:30, treat_num == 1.0] += 4.0
    Y[30:120] += 3.0 * (batch - batch.min())[None, :]

    gene_names = [f"gene_{i}" for i in range(m)]
    sample_names = [f"sample_{j}" for j in range(n)]

    df_feat_rows = pd.DataFrame(Y, index=gene_names, columns=sample_names)
    df_samp_rows = pd.DataFrame(Y.T, index=sample_names, columns=gene_names)
    sp_feat_rows = sparse.csr_matrix(Y)
    adata = ad.AnnData(
        X=sparse.csr_matrix(Y.T),
        obs=pd.DataFrame({"condition": pd.Categorical(treat)}, index=sample_names),
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["raw_counts"] = np.round(Y.T).astype(float)

    # num_sv parity across ndarray, DataFrame, transposed DataFrame, sparse, and AnnData
    nsv_arr = num_sv(Y, mod_arr, method="be", seed=404)
    nsv_df = num_sv(df_feat_rows, mod_arr, method="be", seed=404)
    nsv_df_t = num_sv(df_samp_rows, mod_arr, method="be", seed=404, features_as_rows=False)
    nsv_sp = num_sv(sp_feat_rows, mod_arr, method="be", seed=404)
    nsv_ad = num_sv(adata, "condition", method="be", seed=404)
    assert nsv_arr.n_sv == nsv_df.n_sv == nsv_df_t.n_sv == nsv_sp.n_sv == nsv_ad.n_sv
    np.testing.assert_allclose(nsv_arr.singular_values, nsv_ad.singular_values, atol=1e-12)

    # sva parity and preserved names
    sva_arr = sva(Y, mod_arr, mod0_arr, n_sv=1, seed=404)
    sva_df = sva(df_feat_rows, mod_arr, mod0_arr, n_sv=1, seed=404)
    sva_ad = sva(adata, ["condition"], [], n_sv=1, seed=404)
    np.testing.assert_allclose(sva_arr.sv, sva_df.sv, atol=1e-12)
    np.testing.assert_allclose(sva_arr.sv, sva_ad.sv, atol=1e-12)
    sv_df = sva_ad.to_dataframe()
    assert list(sv_df.index) == sample_names
    assert list(sv_df.columns) == ["SV1"]

    # f_pvalue parity
    fp_arr = f_pvalue(Y, mod_arr, mod0_arr)
    fp_ad = f_pvalue(adata, "condition")
    np.testing.assert_allclose(fp_arr.p_value, fp_ad.p_value, atol=1e-12)
    assert fp_ad.to_dataframe()["feature"].tolist() == gene_names
    assert len(fp_ad.significant(alpha=0.05)) > 0

    # svaseq on AnnData layer
    seq_ad = svaseq(adata, "condition", n_sv=1, layer="raw_counts", seed=404)
    seq_arr = svaseq(np.round(Y), mod_arr, mod0_arr, n_sv=1, seed=404)
    np.testing.assert_allclose(seq_ad.sv, seq_arr.sv, atol=1e-12)

    assert "SVAResult(" in sva_ad.summary()
    assert "FTestResult(" in fp_ad.summary()

    # Frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        sva_ad.n_sv = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        nsv_ad.n_sv = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fp_ad.df1 = 99  # type: ignore[misc]


def test_num_sv_leek_heteroscedastic_null_and_planted_factor_recovery() -> None:
    rng = np.random.default_rng(123)

    # Strong heteroscedastic pure nulls where m_eff < min(m_u, d_e)
    for m, n, n_hi in [(600, 60, 12), (500, 60, 15), (20, 60, 5)]:
        mod = np.column_stack([np.ones(n), np.repeat([0.0, 1.0], n // 2)])
        sigmas = np.full((m, 1), 1e-3, dtype=np.float64)
        sigmas[:n_hi] = 1.0
        Y_null = rng.normal(size=(m, n)) * sigmas
        res_null = num_sv(Y_null, mod, method="leek")
        assert res_null.n_sv == 0

    # Planted latent factors K=1, 2, 3 under heteroscedastic noise are accurately recovered
    m_p, n_p = 600, 60
    mod_p = np.column_stack([np.ones(n_p), np.repeat([0.0, 1.0], n_p // 2)])
    for K in (1, 2, 3):
        sig_p = rng.uniform(0.5, 2.0, size=(m_p, 1))
        Y_sig = rng.normal(size=(m_p, n_p)) * sig_p
        for k in range(K):
            z_k = rng.normal(size=n_p)
            start = (k * m_p) // (K + 1)
            end = ((k + 1) * m_p) // (K + 1)
            Y_sig[start:end] += 2.0 * z_k[None, :]
        assert num_sv(Y_sig, mod_p, method="leek").n_sv == K


def test_f_pvalue_scale_invariance_extreme_magnitudes() -> None:
    rng = np.random.default_rng(2026)
    m, n = 30, 20
    treat = np.repeat([0.0, 1.0], n // 2)
    mod = np.column_stack([np.ones(n), treat])
    mod0 = np.ones((n, 1))

    Y = rng.normal(size=(m, n))
    Y[:6] += 1.8 * treat[None, :]
    # Include a partial-NaN row, a constant row, and an all-zero row
    Y[6, [1, 7, 14]] = np.nan
    Y[7] = 5.0
    Y[8] = 0.0

    ref = f_pvalue(Y, mod, mod0)
    assert np.all(np.isfinite(ref.p_value[:7]))
    assert np.all(np.isfinite(ref.p_value[9:]))
    assert np.isnan(ref.p_value[7]) and np.isnan(ref.p_value[8])

    for scale in (1e-10, 1e10):
        scaled = f_pvalue(Y * scale, mod, mod0)
        np.testing.assert_allclose(scaled.f_stat, ref.f_stat, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(scaled.p_value, ref.p_value, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(scaled.q_value, ref.q_value, rtol=1e-12, atol=1e-12, equal_nan=True)

