from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import mrclust
from mrclust import (
    ClusterFit,
    ClusterInfo,
    MRClustResult,
    compute_ratio_estimates,
    mr_clust_em,
    pr_clust,
)


def test_planted_multi_cluster_null_and_junk_recovery() -> None:
    rng = np.random.default_rng(123)
    theta_true = np.concatenate(
        [
            np.full(20, -0.60),
            np.full(20, 0.85),
            np.zeros(12),
            np.array([-4.1, -3.3, -2.5, -2.1, 2.2, 2.8, 3.6, 4.3]),
        ]
    )
    true_labels = np.array(
        ["1"] * 20 + ["2"] * 20 + ["Null"] * 12 + ["Junk"] * 8
    )
    n = theta_true.size
    snps = [f"rs{1000 + i}" for i in range(n)]
    bx = rng.uniform(0.05, 0.12, size=n)
    bxse = rng.uniform(0.003, 0.008, size=n)
    byse = rng.uniform(0.004, 0.009, size=n)
    by = theta_true * bx + rng.normal(0.0, byse)

    res = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, snp_names=snps, seed=42)

    assert res.best_k == 2
    assert res.best_model.cluster_means.shape == (2,)
    np.testing.assert_allclose(res.best_model.cluster_means, [-0.60, 0.85], atol=0.08)

    acc_all = float(np.mean(np.array(res.best_cluster) == true_labels))
    assert acc_all >= 0.90

    df = res.to_dataframe()
    confident = df[df["passes_threshold"]]
    assert len(confident) >= 45
    acc_conf = float(
        np.mean(confident["cluster"].to_numpy() == true_labels[df["passes_threshold"].to_numpy()])
    )
    assert acc_conf >= 0.95


def test_single_vs_two_cluster_bic_separation() -> None:
    rng = np.random.default_rng(2025)

    # Regime 1: 1 substantive cluster (theta=0.55) + Null
    theta1_true = np.concatenate([np.full(28, 0.55), np.zeros(12)])
    se1 = rng.uniform(0.04, 0.08, size=theta1_true.size)
    theta1 = theta1_true + rng.normal(0.0, se1)
    res1 = mr_clust_em(theta1, se1, k_max=5, seed=11)
    bic1 = {m.k: m.bic for m in res1.all_models}
    assert res1.best_k == 1
    assert bic1[1] < bic1[0]
    assert bic1[1] < bic1[2]
    assert abs(res1.best_model.cluster_means[0] - 0.55) < 0.06

    # Regime 2: 2 moderately close substantive clusters (0.25 and 0.70) + outliers
    theta2_true = np.concatenate([np.full(28, 0.25), np.full(28, 0.70), np.array([-2.2, 2.5, -1.9, 2.1])])
    se2 = np.concatenate([rng.uniform(0.04, 0.06, size=56), rng.uniform(0.05, 0.08, size=4)])
    theta2 = theta2_true + rng.normal(0.0, se2)
    res2 = mr_clust_em(theta2, se2, k_max=5, seed=11)
    bic2 = {m.k: m.bic for m in res2.all_models}
    assert res2.best_k == 2
    assert bic2[2] < bic2[1]
    np.testing.assert_allclose(res2.best_model.cluster_means, [0.25, 0.70], atol=0.08)


def test_null_calibration_pure_zero_effect() -> None:
    rng = np.random.default_rng(777)
    n_reps = 15
    k_zero_count = 0
    null_fracs: list[float] = []

    for rep in range(n_reps):
        j = 40
        bx = rng.uniform(0.06, 0.14, size=j)
        byse = rng.uniform(0.005, 0.010, size=j)
        by = rng.normal(0.0, byse)
        res = mr_clust_em(bx=bx, by=by, byse=byse, k_max=4, seed=42 + rep)
        if res.best_k == 0:
            k_zero_count += 1
            assert res.best_model.cluster_means.shape == (0,)
            cdf = res.clusters_dataframe()
            assert list(cdf["cluster"]) == ["Null", "Junk"]
            df = res.to_dataframe()
            assert len(df) == j
            null_fracs.append(float(np.mean(np.array(res.best_cluster) == "Null")))

    assert k_zero_count >= 13
    assert min(null_fracs) >= 0.85


def test_outliers_absorbed_by_junk_without_biasing_cluster() -> None:
    rng = np.random.default_rng(314)
    n_main = 32
    theta_main = 0.50 + rng.normal(0.0, 0.06, size=n_main)
    se_main = np.full(n_main, 0.06)
    theta_out = np.array([3.0, -2.5, 4.0])
    se_out = np.array([0.07, 0.08, 0.06])

    theta = np.concatenate([theta_main, theta_out])
    se = np.concatenate([se_main, se_out])
    snps = [f"snp_{i}" for i in range(theta.size)]

    res = mr_clust_em(theta, se, snp_names=snps, k_max=5, seed=42)
    assert res.best_k == 1
    assert abs(res.best_model.cluster_means[0] - 0.50) < 0.05
    assert list(res.best_cluster[-3:]) == ["Junk", "Junk", "Junk"]


def test_delta_method_first_and_second_order_variances() -> None:
    rng = np.random.default_rng(99)
    bx = rng.uniform(0.05, 0.20, size=25)
    bxse = rng.uniform(0.005, 0.02, size=25)
    byse = rng.uniform(0.004, 0.015, size=25)
    by = 0.6 * bx + rng.normal(0.0, byse)

    th1, se1 = compute_ratio_estimates(bx, by, byse, variance_order=1)
    th2, se2 = compute_ratio_estimates(bx, by, byse, bxse=bxse, variance_order="second")
    th2_zero_bxse, se2_zero_bxse = compute_ratio_estimates(
        bx, by, byse, bxse=np.zeros_like(bx), variance_order=2
    )

    expected_th = by / bx
    expected_se1 = byse / np.abs(bx)
    expected_se2 = np.sqrt((byse ** 2) / (bx ** 2) + (by ** 2) * (bxse ** 2) / (bx ** 4))

    np.testing.assert_allclose(th1, expected_th, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(se1, expected_se1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(th2, expected_th, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(se2, expected_se2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(se2_zero_bxse, se1, rtol=1e-12, atol=1e-12)
    assert np.all(se2 > se1)

    res2 = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, variance_order=2, k_max=3, seed=5)
    assert res2.variance_order == 2
    np.testing.assert_allclose(res2.theta_se, se2, rtol=1e-12, atol=1e-12)


def test_inverse_variance_weighting_m_step_identity() -> None:
    # 10 high-precision SNPs around 0.40 (SE=0.015), 10 low-precision SNPs around 1.00 (SE=0.35)
    theta = np.concatenate([np.full(10, 0.40), np.full(10, 1.00)])
    se = np.concatenate([np.full(10, 0.015), np.full(10, 0.35)])

    # Single-cluster fit: unweighted mean is 0.70, IVW mean is ~0.401
    res_single = mr_clust_em(theta, se, k_min=1, k_max=1, seed=42)
    assert res_single.best_k == 1
    assert abs(res_single.best_model.cluster_means[0] - 0.40) < 0.02

    # Exact M-step fixed-point identity holds for every substantive cluster across K=1 and K=2
    res_multi = mr_clust_em(theta, se, k_min=1, k_max=2, seed=42)
    for res in (res_single, res_multi):
        fit = res.best_model
        inv_var = 1.0 / (res.theta_se ** 2)
        for m in range(fit.k):
            r_m = fit.responsibilities[:, m]
            ivw_mean = float(np.sum(r_m * inv_var * res.theta) / np.sum(r_m * inv_var))
            np.testing.assert_allclose(fit.cluster_means[m], ivw_mean, atol=1e-10, rtol=1e-10)


def test_em_log_likelihood_and_bic_formula_invariants() -> None:
    rng = np.random.default_rng(55)
    theta = np.concatenate([np.full(18, -0.4), np.full(18, 0.6), np.zeros(9)]) + rng.normal(0.0, 0.05, size=45)
    se = rng.uniform(0.04, 0.07, size=45)

    res = mr_clust_em(theta, se, k_min=0, k_max=5, seed=19)
    for fit in res.all_models:
        assert fit.n_components == fit.k + 2
        assert fit.n_parameters == 2 * fit.k + 1
        expected_bic = -2.0 * fit.log_likelihood + fit.n_parameters * np.log(res.n_snps)
        expected_aic = -2.0 * fit.log_likelihood + 2.0 * fit.n_parameters
        assert np.isclose(fit.bic, expected_bic, rtol=1e-12, atol=1e-12)
        assert np.isclose(fit.aic, expected_aic, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fit.responsibilities.sum(axis=1), 1.0, atol=1e-12)
        assert np.isclose(fit.weights.sum(), 1.0, atol=1e-12)
        if fit.k > 1:
            assert np.all(np.diff(fit.cluster_means) >= 0.0)


def test_split_merge_rescues_small_tight_cluster() -> None:
    rng = np.random.default_rng(808)
    theta_main = 0.40 + rng.normal(0.0, 0.05, size=25)
    se_main = np.full(25, 0.05)

    theta_null = rng.normal(0.0, 0.05, size=10)
    se_null = np.full(10, 0.05)

    theta_tight = 1.60 + rng.normal(0.0, 0.015, size=7)
    se_tight = np.full(7, 0.02)

    theta_junk = np.array([-3.0, -2.1, -1.5, 2.5, 3.2, 4.1])
    se_junk = np.full(6, 0.07)

    theta = np.concatenate([theta_main, theta_null, theta_tight, theta_junk])
    se = np.concatenate([se_main, se_null, se_tight, se_junk])

    res = mr_clust_em(theta, se, k_max=5, seed=42)
    bic = {m.k: m.bic for m in res.all_models}
    assert res.best_k == 2
    assert bic[2] < bic[1]
    np.testing.assert_allclose(res.best_model.cluster_means, [0.40, 1.60], atol=0.08)


def test_probability_filtering_and_pr_clust() -> None:
    # Two clusters at 0.25 and 0.75 + explicit ambiguous boundary SNP at 0.50
    rng = np.random.default_rng(101)
    th1 = 0.25 + rng.normal(0.0, 0.03, size=20)
    th2 = 0.75 + rng.normal(0.0, 0.03, size=20)
    se_c = np.full(40, 0.035)
    theta = np.concatenate([th1, th2, [0.50]])
    se = np.concatenate([se_c, [0.10]])
    snps = [f"rs{i}" for i in range(41)]

    res = mr_clust_em(theta, se, snp_names=snps, k_min=2, k_max=2, obs_prob_thresh=0.8, seed=42)
    df = res.to_dataframe()
    ambig = df.loc[df["observation"] == "rs40"].iloc[0]
    assert ambig["best_cluster"] in ("1", "2")
    assert 0.45 < ambig["probability"] < 0.70
    assert not bool(ambig["passes_threshold"])
    assert ambig["cluster"] == "Unassigned"

    filt_strict = pr_clust(res, obs_prob_thresh=0.8)
    assert "rs40" not in set(filt_strict["observation"])

    filt_loose = pr_clust(res, obs_prob_thresh=0.45)
    assert "rs40" in set(filt_loose["observation"])

    # min_cluster_size excludes clusters smaller than threshold
    filt_size = pr_clust(res, obs_prob_thresh=0.8, min_cluster_size=25)
    assert filt_size.empty


def test_allele_coding_sign_flip_invariance_and_determinism() -> None:
    rng = np.random.default_rng(5150)
    n = 45
    theta_true = np.concatenate([np.full(18, -0.5), np.full(18, 0.7), np.zeros(9)])
    bx = rng.uniform(0.05, 0.15, size=n)
    bxse = rng.uniform(0.004, 0.01, size=n)
    byse = rng.uniform(0.004, 0.01, size=n)
    by = theta_true * bx + rng.normal(0.0, byse)

    flip_mask = rng.random(n) < 0.5
    bx_flip = np.where(flip_mask, -bx, bx)
    by_flip = np.where(flip_mask, -by, by)

    res_a = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, variance_order=2, k_max=4, seed=7)
    res_b = mr_clust_em(bx=bx_flip, bxse=bxse, by=by_flip, byse=byse, variance_order=2, k_max=4, seed=7)
    res_a_repeat = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, variance_order=2, k_max=4, seed=7)

    np.testing.assert_allclose(res_a.theta, res_b.theta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_a.theta_se, res_b.theta_se, rtol=1e-12, atol=1e-12)
    assert res_a.best_k == res_b.best_k
    np.testing.assert_allclose(res_a.best_model.cluster_means, res_b.best_model.cluster_means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_a.responsibilities, res_b.responsibilities, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        res_a.bic_dataframe()["bic"].to_numpy(),
        res_b.bic_dataframe()["bic"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    pd.testing.assert_frame_equal(res_a.to_dataframe(include_probs=True), res_a_repeat.to_dataframe(include_probs=True))


def test_unit_scale_equivariance() -> None:
    rng = np.random.default_rng(64)
    n = 44
    theta_true = np.concatenate([np.full(17, -0.35), np.full(17, 0.80), np.zeros(10)])
    bx = rng.uniform(0.06, 0.14, size=n)
    byse = rng.uniform(0.004, 0.008, size=n)
    by = theta_true * bx + rng.normal(0.0, byse)

    scale = 50.0
    res1 = mr_clust_em(bx=bx, by=by, byse=byse, k_max=4, seed=42)
    res50 = mr_clust_em(bx=bx, by=scale * by, byse=scale * byse, k_max=4, seed=42)

    assert res1.best_k == res50.best_k
    np.testing.assert_allclose(res50.best_model.cluster_means, scale * res1.best_model.cluster_means, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res50.best_model.cluster_ses, scale * res1.best_model.cluster_ses, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res50.best_model.weights, res1.best_model.weights, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res50.responsibilities, res1.responsibilities, rtol=1e-10, atol=1e-10)


def test_near_zero_bx_overflow_protection_and_exact_zero_rejection() -> None:
    rng = np.random.default_rng(909)
    n = 25
    bx = rng.uniform(0.06, 0.14, size=n)
    bxse = rng.uniform(0.005, 0.01, size=n)
    byse = rng.uniform(0.004, 0.008, size=n)
    by = 0.55 * bx + rng.normal(0.0, byse)

    # Inject a tiny non-zero |bx| = 1e-80 where naive bx**4 underflows to 0.0
    bx[0] = 1e-80
    by[0] = 1e-80

    res = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, variance_order=2, k_max=3, seed=42)
    assert np.all(np.isfinite(res.theta))
    assert np.all(np.isfinite(res.theta_se))
    assert res.theta_se[0] > 1e70
    assert np.all(np.isfinite(res.responsibilities))
    assert res.best_k == 1
    assert abs(res.best_model.cluster_means[0] - 0.55) < 0.05
    # With an enormous SE (~1e78), SNP 0 carries virtually zero IVW weight and does not bias cluster 1
    assert res.best_model.cluster_ses[0] < 0.05

    # Exact bx == 0.0 must raise clear ValueError
    bx_zero = bx.copy()
    bx_zero[3] = 0.0
    with pytest.raises(ValueError, match="exact zero"):
        compute_ratio_estimates(bx_zero, by, byse)


def test_weak_iv_near_zero_bx_with_outliers_does_not_inflate_junk() -> None:
    rng = np.random.default_rng(5150)
    bx_base = rng.uniform(0.06, 0.14, size=36)
    bxse = rng.uniform(0.005, 0.01, size=36)
    byse = rng.uniform(0.004, 0.008, size=36)
    by_base = 0.50 * bx_base + rng.normal(0.0, byse)
    by_base[1:4] = np.array([3.2, -2.5, 4.0]) * bx_base[1:4]

    res_clean = mr_clust_em(
        bx=bx_base[1:],
        bxse=bxse[1:],
        by=by_base[1:],
        byse=byse[1:],
        variance_order=2,
        k_max=5,
        seed=42,
    )
    assert res_clean.best_k == 1
    assert res_clean.best_cluster[:3] == ("Junk", "Junk", "Junk")

    for bx0, by0 in ((1e-4, 0.01), (1e-80, 0.01), (1e-80, 1e-80)):
        bx = bx_base.copy()
        by = by_base.copy()
        bx[0] = bx0
        by[0] = by0
        res = mr_clust_em(
            bx=bx,
            bxse=bxse,
            by=by,
            byse=byse,
            variance_order=2,
            k_max=5,
            seed=42,
        )
        assert res.best_k == 1
        assert abs(res.best_model.cluster_means[0] - 0.50) < 0.05
        assert np.isclose(res.junk_scale, res_clean.junk_scale, rtol=1e-12, atol=1e-12)
        assert np.isclose(res.junk_mean, res_clean.junk_mean, rtol=1e-12, atol=1e-12)
        assert res.best_cluster[1:4] == ("Junk", "Junk", "Junk")
        assert res.best_cluster[0] == "Junk"
        assert res.cluster_membership[0] in ("Junk", "Unassigned")
        assert res.cluster_membership[1:4] == ("Junk", "Junk", "Junk")

    # Unit scale equivariance holds even with weak-IV SNP present
    bx_s = bx_base.copy()
    by_s = by_base.copy()
    bx_s[0], by_s[0] = 1e-4, 0.01
    res_1 = mr_clust_em(bx=bx_s, bxse=bxse, by=by_s, byse=byse, variance_order=2, k_max=4, seed=42)
    res_50 = mr_clust_em(
        bx=bx_s, bxse=bxse, by=50.0 * by_s, byse=50.0 * byse, variance_order=2, k_max=4, seed=42
    )
    assert res_50.best_k == res_1.best_k == 1
    np.testing.assert_allclose(res_50.best_model.cluster_means, 50.0 * res_1.best_model.cluster_means, rtol=1e-10)
    np.testing.assert_allclose(res_50.responsibilities, res_1.responsibilities, rtol=1e-10, atol=1e-12)


def test_input_modes_schemas_frozen_dataclasses_and_edge_cases() -> None:
    # Importing mrclust must not eagerly import matplotlib
    assert "matplotlib" not in sys.modules or mrclust.__name__ == "mrclust"

    rng = np.random.default_rng(12)
    n = 30
    bx = rng.uniform(0.06, 0.12, size=n)
    bxse = rng.uniform(0.004, 0.008, size=n)
    byse = rng.uniform(0.004, 0.008, size=n)
    by = np.concatenate([np.full(18, 0.5), np.zeros(12)]) * bx + rng.normal(0.0, byse)
    snps = [f"rs_{i}" for i in range(n)]

    theta, theta_se = compute_ratio_estimates(bx, by, byse, bxse=bxse, variance_order=2)
    df_harmonised = pd.DataFrame(
        {
            "snp": snps,
            "beta_exp": bx,
            "se_exp": bxse,
            "beta_out": by,
            "se_out": byse,
        }
    )

    res_theta = mr_clust_em(theta, theta_se, obs_names=snps, k_max=3, seed=42)
    res_beta = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, snp_names=snps, variance_order=2, k_max=3, seed=42)
    res_df = mr_clust_em(data=df_harmonised, variance_order=2, k_max=3, seed=42)

    pd.testing.assert_frame_equal(
        res_theta.to_dataframe(include_probs=True)[["observation", "theta", "theta_se", "cluster", "probability"]],
        res_beta.to_dataframe(include_probs=True)[["observation", "theta", "theta_se", "cluster", "probability"]],
    )
    pd.testing.assert_frame_equal(res_beta.to_dataframe(include_probs=True), res_df.to_dataframe(include_probs=True))

    # Schema checks
    df_main = res_df.to_dataframe(include_probs=True, sort_by_cluster=True)
    for col in [
        "observation",
        "theta",
        "theta_se",
        "cluster",
        "best_cluster",
        "cluster_class",
        "cluster_mean",
        "probability",
        "passes_threshold",
        "bx",
        "bxse",
        "by",
        "byse",
    ]:
        assert col in df_main.columns

    df_clust = res_df.clusters_dataframe()
    assert list(df_clust.columns) == [
        "cluster",
        "cluster_class",
        "cluster_mean",
        "cluster_se",
        "weight",
        "effective_n",
        "n_assigned",
        "n_assigned_thresh",
    ]
    pd.testing.assert_frame_equal(df_clust, res_df.cluster_summary())

    df_bic = res_df.bic_dataframe()
    assert list(df_bic.columns) == [
        "k",
        "n_components",
        "n_parameters",
        "log_likelihood",
        "bic",
        "aic",
        "is_best",
        "converged",
        "n_iter",
    ]
    assert isinstance(res_df.summary(), str) and "MR-Clust Summary" in res_df.summary()
    assert isinstance(res_df.best_model, ClusterFit)
    assert isinstance(res_df.clusters[0], ClusterInfo)
    assert isinstance(res_df.clusters[0].to_dataframe(), pd.DataFrame)
    assert isinstance(res_df.best_model.to_dataframe(), pd.DataFrame)

    # Frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        res_df.best_k = 99  # type: ignore[misc]

    # Degenerate all-identical theta + alternative junk distributions complete cleanly
    res_ident = mr_clust_em(np.full(10, 0.5), np.full(10, 0.05), k_max=2, junk_dist="normal", seed=1)
    assert np.all(np.isfinite(res_ident.responsibilities))
    res_unif = mr_clust_em(theta, theta_se, k_max=2, junk_dist="uniform", seed=1)
    assert np.all(np.isfinite(res_unif.responsibilities))

    # Validation error checks
    with pytest.raises(ValueError):
        mr_clust_em([0.5], [0.1])
    with pytest.raises(ValueError):
        mr_clust_em(bx=[0.1, 0.2], by=[0.1, 0.2], byse=[0.0, 0.1])
    with pytest.raises(ValueError):
        mr_clust_em([0.1, 0.2], [-0.1, 0.1])
    with pytest.raises(ValueError):
        mr_clust_em([0.1, np.nan], [0.1, 0.1])
    with pytest.raises(ValueError):
        mr_clust_em([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        mr_clust_em(theta, theta_se, obs_prob_thresh=1.5)
    with pytest.raises(ValueError):
        mr_clust_em(theta, theta_se, k_min=3, k_max=2)


def test_strong_plus_weak_ivs_do_not_collapse_cluster() -> None:
    k0_count = 0
    misclass_count = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        bx_s = rng.uniform(0.08, 0.12, size=10)
        byse_s = rng.uniform(0.004, 0.006, size=10)
        by_s = 0.50 * bx_s + rng.normal(0.0, byse_s)
        bx_w = rng.uniform(0.0003, 0.0007, size=10)
        byse_w = rng.uniform(0.004, 0.006, size=10)
        by_w = 0.50 * bx_w + rng.normal(0.0, byse_w)
        res = mr_clust_em(
            bx=np.concatenate([bx_s, bx_w]),
            by=np.concatenate([by_s, by_w]),
            byse=np.concatenate([byse_s, byse_w]),
            k_max=4,
            seed=42,
        )
        k0_count += int(res.best_k == 0)
        misclass_count += int(any(c == "Junk" for c in res.best_cluster[:10]))

    assert k0_count == 0
    assert misclass_count == 0


def test_bimodal_precision_recovery_across_junk_distributions() -> None:
    rng = np.random.default_rng(42)
    th_b = np.concatenate(
        [0.3 + rng.normal(0.0, 1e-4, 11), 0.3 + rng.normal(0.0, 0.05, 25), [3.0, -2.5, 4.0]]
    )
    se_b = np.concatenate([np.full(11, 1e-4), np.full(25, 0.05), np.full(3, 0.05)])

    th_1 = np.concatenate([0.3 + rng.normal(0.0, 1e-4, 12), 1.2 + rng.normal(0.0, 0.05, 28)])
    se_1 = np.concatenate([np.full(12, 1e-4), np.full(28, 0.05)])

    th_a = np.concatenate([0.3 + rng.normal(0.0, 1e-9, 11), 1.2 + rng.normal(0.0, 0.05, 26)])
    se_a = np.concatenate([np.full(11, 1e-9), np.full(26, 0.05)])

    for jd in ("t", "normal", "uniform"):
        res_b = mr_clust_em(th_b, se_b, k_max=5, junk_dist=jd, seed=42)
        assert res_b.best_k == 1
        assert abs(res_b.best_model.cluster_means[0] - 0.3) < 0.05
        assert res_b.best_cluster[-3:] == ("Junk", "Junk", "Junk")

        res_1 = mr_clust_em(th_1, se_1, k_max=5, junk_dist=jd, seed=42)
        assert res_1.best_k == 2
        np.testing.assert_allclose(res_1.best_model.cluster_means, [0.3, 1.2], atol=0.05)
        assert sum(c == "Junk" for c in res_1.best_cluster) == 0

        res_a = mr_clust_em(th_a, se_a, k_max=5, junk_dist=jd, seed=42)
        assert res_a.best_k == 2
        np.testing.assert_allclose(res_a.best_model.cluster_means, [0.3, 1.2], atol=0.05)
        assert sum(c == "Junk" for c in res_a.best_cluster) == 0


def test_majority_weak_snps_do_not_inflate_junk_or_hallucinate_clusters() -> None:
    rng = np.random.default_rng(7)
    th_w = np.concatenate(
        [0.5 + rng.normal(0.0, 0.05, 7), [3.0, -2.5, 4.0], rng.normal(0.0, 1e4, 35)]
    )
    se_w = np.concatenate([np.full(7, 0.05), np.full(3, 0.05), np.full(35, 1e4)])

    for jd in ("t", "normal", "uniform"):
        res = mr_clust_em(th_w, se_w, k_max=5, junk_dist=jd, seed=42)
        assert res.best_k == 1
        assert abs(res.best_model.cluster_means[0] - 0.5) < 0.08
        assert res.junk_scale < 20.0
        assert res.best_cluster[7:10] == ("Junk", "Junk", "Junk")
        assert res.cluster_membership[7:10] == ("Junk", "Junk", "Junk")
        assert all(m == "Unassigned" for m in res.cluster_membership[10:])


def test_multicluster_extreme_scale_equivariance() -> None:
    rng = np.random.default_rng(2025)
    th_s = np.concatenate([np.full(20, 0.25), np.full(20, 0.70), np.zeros(10), [-2.5, 3.0]])
    se_s = np.concatenate([rng.uniform(0.03, 0.05, 50), [0.05, 0.05]])
    th_s[:50] += rng.normal(0.0, se_s[:50])

    res_ref = mr_clust_em(th_s, se_s, k_max=4, seed=42)
    assert res_ref.best_k == 2

    for c in (1e-100, 1e-40, -2.5, 1e40, 1e100):
        res_c = mr_clust_em(c * th_s, abs(c) * se_s, k_max=4, seed=42)
        assert res_c.best_k == 2
        if c > 0:
            np.testing.assert_allclose(
                res_c.best_model.cluster_means / c, res_ref.best_model.cluster_means, rtol=1e-7
            )
            np.testing.assert_allclose(
                res_c.best_model.weights, res_ref.best_model.weights, atol=1e-6
            )
            np.testing.assert_allclose(
                res_c.responsibilities, res_ref.responsibilities, atol=1e-6
            )
        else:
            np.testing.assert_allclose(
                np.sort(res_c.best_model.cluster_means / c),
                res_ref.best_model.cluster_means,
                rtol=1e-7,
            )


def test_extreme_finite_ratio_ward_linkage_overflow_guard_and_uninformative_gating() -> None:
    rng = np.random.default_rng(5150)
    bx = rng.uniform(0.06, 0.14, size=36)
    byse = rng.uniform(0.004, 0.008, size=36)
    by = 0.50 * bx + rng.normal(0.0, byse)
    by[1:4] = np.array([3.2, -2.5, 4.0]) * bx[1:4]

    bx[0] = 1e-160
    by[0] = 0.01

    res = mr_clust_em(bx=bx, by=by, byse=byse, variance_order=1, k_max=5, seed=42)
    assert res.best_k == 1
    assert abs(res.best_model.cluster_means[0] - 0.50) < 0.05
    assert res.best_cluster[:4] == ("Junk", "Junk", "Junk", "Junk")
    assert res.cluster_membership[0] == "Unassigned"
    assert res.cluster_membership[1:4] == ("Junk", "Junk", "Junk")


def test_cis_eqtl_plus_polygenic_and_near_zero_bimodal_not_misclassified_as_junk() -> None:
    # 1 strong cis-eQTL + 25 polygenic SNPs near theta=0.10 (zero outliers)
    rng = np.random.default_rng(99)
    bx = np.concatenate([[0.45], rng.uniform(0.03, 0.06, 25)])
    byse = np.concatenate([[0.001], rng.uniform(0.008, 0.012, 25)])
    by = 0.10 * bx + rng.normal(0.0, byse)
    res1 = mr_clust_em(bx=bx, by=by, byse=byse, k_max=4, seed=42)
    assert res1.best_k == 1
    assert res1.best_cluster.count("Junk") == 0

    # Probe X4: near-zero bimodal precision (11@0.02 se=1e-4 + 25@0.02 se=0.05 + 3 outliers)
    rng2 = np.random.default_rng(42)
    th4 = np.concatenate(
        [0.02 + rng2.normal(0.0, 1e-4, 11), 0.02 + rng2.normal(0.0, 0.05, 25), [3.0, -2.5, 4.0]]
    )
    se4 = np.concatenate([np.full(11, 1e-4), np.full(25, 0.05), np.full(3, 0.05)])
    res2 = mr_clust_em(th4, se4, k_max=5, seed=42)
    assert res2.best_k == 1
    assert res2.best_cluster[11:36].count("Junk") == 0
    assert res2.best_cluster[-3:] == ("Junk", "Junk", "Junk")

    # Midpoint SNP between Null (0.0) and cluster '1' (0.20) must not be forced into Junk
    rng3 = np.random.default_rng(42)
    th_mid = np.concatenate([rng3.normal(0.0, 1e-4, 11), 0.20 + rng3.normal(0.0, 0.05, 24), [0.10]])
    se_mid = np.concatenate([np.full(11, 1e-4), np.full(25, 0.05)])
    res_mid = mr_clust_em(th_mid, se_mid, k_max=5, seed=42)
    assert res_mid.best_cluster[-1] in ("1", "Null")
    assert res_mid.cluster_membership[-1] == "Unassigned"


def test_include_null_false_with_hyperprecise_anchor_and_outliers() -> None:
    rng = np.random.default_rng(42)
    th = np.concatenate([[0.5], 0.5 + rng.normal(0.0, 0.05, 30), [3.0, -2.5, 4.0]])
    se = np.concatenate([[1e-5], np.full(30, 0.05), np.full(3, 0.05)])
    res = mr_clust_em(th, se, k_min=1, k_max=4, include_null=False, seed=42)
    assert res.best_k == 1
    assert res.best_cluster[:31] == ("1",) * 31
    assert res.best_cluster[-3:] == ("Junk", "Junk", "Junk")

    # Without outliers: single precise SNP at 0.0 + 20 cluster SNPs at 0.15
    rng2 = np.random.default_rng(11)
    th2 = np.concatenate([[0.0], 0.15 + rng2.normal(0.0, 0.05, 20)])
    se2 = np.concatenate([[1e-5], np.full(20, 0.05)])
    res2 = mr_clust_em(th2, se2, k_min=1, k_max=4, include_null=False, seed=42)
    assert res2.best_k == 1
    assert sum(c == "1" for c in res2.best_cluster) >= 18


def test_bimodal_pure_null_and_major_locus_mr_no_collapse() -> None:
    # Bimodal Pure Null: 10@1e-4 + 25@0.05 across 20 seeds -> 0/500 normal SNPs in Junk
    junk_normal = 0
    for s in range(20):
        rng = np.random.default_rng(s)
        se = np.concatenate([np.full(10, 1e-4), np.full(25, 0.05)])
        th = rng.normal(0.0, se)
        res = mr_clust_em(th, se, k_max=4, seed=42)
        junk_normal += res.best_cluster[10:].count("Junk")
    assert junk_normal == 0

    # Major-locus MR: 1@0.004 + 25@0.50 across theta in {0.0, 0.25, 0.50}
    for theta_true, expected_k in ((0.0, 0), (0.25, 1), (0.50, 1)):
        rng_ml = np.random.default_rng(7)
        se_ml = np.concatenate([[0.004], np.full(25, 0.50)])
        th_ml = theta_true + rng_ml.normal(0.0, se_ml)
        res_ml = mr_clust_em(th_ml, se_ml, k_max=4, seed=42)
        assert res_ml.best_k == expected_k
        assert res_ml.best_cluster.count("Junk") == 0


def test_diag2b_with_second_cluster_and_high_z0_weak_outlier() -> None:
    for jd in ("t", "normal", "uniform"):
        rng1 = np.random.default_rng(7)
        th_2c = np.concatenate(
            [
                0.5 + rng1.normal(0.0, 0.05, 7),
                [3.0, -2.5, 4.0],
                2000.0 + rng1.normal(0.0, 50.0, 5),
                rng1.normal(0.0, 1e4, 35),
            ]
        )
        se_2c = np.concatenate([np.full(7, 0.05), np.full(3, 0.05), np.full(5, 50.0), np.full(35, 1e4)])
        r_2c = mr_clust_em(th_2c, se_2c, k_max=6, junk_dist=jd, seed=42)
        assert r_2c.best_k == 2
        assert r_2c.best_cluster[7:10] == ("Junk", "Junk", "Junk")
        assert set(r_2c.cluster_membership[15:]) == {"Unassigned"}

        rng2 = np.random.default_rng(7)
        th_hw = np.concatenate(
            [0.5 + rng2.normal(0.0, 0.05, 7), [3.0, -2.5, 4.0], rng2.normal(0.0, 1e4, 35), [1e4]]
        )
        se_hw = np.concatenate([np.full(7, 0.05), np.full(3, 0.05), np.full(35, 1e4), [500.0]])
        r_hw = mr_clust_em(th_hw, se_hw, k_max=5, junk_dist=jd, seed=42)
        assert r_hw.best_k == 1
        assert r_hw.best_cluster[7:10] == ("Junk", "Junk", "Junk")
        assert set(r_hw.cluster_membership[10:45]) == {"Unassigned"}
        assert r_hw.best_cluster[45] == "Junk"


def test_strong_plus_weak_uniform_and_numerical_overflow_guards() -> None:
    # 10 strong + 0/10 weak with junk_dist="uniform" (including seed=17 tail)
    for m_weak in (0, 10):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            bx_s = rng.uniform(0.08, 0.12, size=10)
            byse_s = rng.uniform(0.004, 0.006, size=10)
            by_s = 0.50 * bx_s + rng.normal(0.0, byse_s)
            bx_w = rng.uniform(0.0003, 0.0007, size=m_weak)
            byse_w = rng.uniform(0.004, 0.006, size=m_weak)
            by_w = 0.50 * bx_w + rng.normal(0.0, byse_w)
            res = mr_clust_em(
                bx=np.concatenate([bx_s, bx_w]),
                by=np.concatenate([by_s, by_w]),
                byse=np.concatenate([byse_s, byse_w]),
                k_max=4,
                junk_dist="uniform",
                seed=42,
            )
            assert res.best_k == 1
            assert "Junk" not in res.best_cluster[:10]

    # compute_ratio_estimates overflow emits no RuntimeWarning before ValueError
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="overflowed"):
            compute_ratio_estimates(
                [1e-160, 0.1], [0.01, 0.05], [0.005, 0.005], bxse=[0.005, 0.005], variance_order=2
            )
        assert len(wrec) == 0

    # include_junk=False with 1e200 extreme outlier produces finite responsibilities without warnings
    with warnings.catch_warnings(record=True) as wrec2:
        warnings.simplefilter("always")
        th_inf = np.concatenate([0.5 + np.random.default_rng(1).normal(0.0, 0.05, 25), [1e200]])
        se_inf = np.full(26, 0.05)
        res_inf = mr_clust_em(th_inf, se_inf, k_max=3, include_junk=False, seed=42)
        assert np.all(np.isfinite(res_inf.responsibilities))
        np.testing.assert_allclose(res_inf.responsibilities.sum(axis=1), 1.0, atol=1e-12)
        assert len(wrec2) == 0


