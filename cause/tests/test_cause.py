"""
Deterministic verification suite for `cause` (CAUSE Mendelian randomization & PSIS-LOO).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cause import (
    CauseModelFit,
    CauseParams,
    CauseResult,
    PSISLOOResult,
    cause,
    compare_elpd,
    est_cause_params,
    format_cause_data,
    gpdfit,
    gwas_merge,
    ld_prune,
    psis_loo,
)


def _simulate_cause_gwas(
    rng: np.random.Generator,
    n_snps: int = 4000,
    n_causal_x: int = 120,
    gamma: float = 0.0,
    eta: float = 0.0,
    q: float = 0.0,
    rho: float = 0.0,
    sigma_x: float = 0.06,
    sigma_u_pleio: float = 0.015,
    n_pleio_snps: int = 100,
    se_x: float = 0.015,
    se_y: float = 0.020,
) -> pd.DataFrame:
    """Simulate two-sample GWAS summary statistics under the CAUSE structural model."""
    beta_x_true = np.zeros(n_snps, dtype=np.float64)
    if n_causal_x > 0:
        idx_x = rng.choice(n_snps, size=n_causal_x, replace=False)
        beta_x_true[idx_x] = rng.normal(0.0, sigma_x, size=n_causal_x)

    theta_true = np.zeros(n_snps, dtype=np.float64)
    if n_pleio_snps > 0:
        idx_u = rng.choice(n_snps, size=n_pleio_snps, replace=False)
        theta_true[idx_u] = rng.normal(0.0, sigma_u_pleio, size=n_pleio_snps)

    z_shared = rng.binomial(1, q, size=n_snps).astype(np.float64)
    beta_y_true = (gamma + z_shared * eta) * beta_x_true + theta_true

    # Correlated sampling errors with correlation rho
    e1 = rng.normal(0.0, 1.0, size=n_snps)
    e2 = rho * e1 + np.sqrt(max(1.0 - rho * rho, 1e-12)) * rng.normal(0.0, 1.0, size=n_snps)
    s1 = np.full(n_snps, se_x, dtype=np.float64)
    s2 = np.full(n_snps, se_y, dtype=np.float64)
    b1_hat = beta_x_true + s1 * e1
    b2_hat = beta_y_true + s2 * e2

    return pd.DataFrame(
        {
            "snp": [f"rs{i + 1}" for i in range(n_snps)],
            "beta_hat_1": b1_hat,
            "seb1": s1,
            "beta_hat_2": b2_hat,
            "seb2": s2,
            "p1": 2.0 * stats.norm.sf(np.abs(b1_hat / s1)),
            "p2": 2.0 * stats.norm.sf(np.abs(b2_hat / s2)),
        }
    )


def _ivw_pvalue(b1: np.ndarray, s1: np.ndarray, b2: np.ndarray, s2: np.ndarray) -> tuple[float, float, float]:
    """Return `(beta_ivw, p_ivw_fe, p_ivw_re)` for fixed-effects and multiplicative RE IVW."""
    w = 1.0 / (s2 * s2)
    info = float(np.sum(w * b1 * b1))
    beta_ivw = float(np.sum(w * b1 * b2) / info)
    se_fe = float(np.sqrt(1.0 / info))
    p_fe = float(2.0 * stats.norm.sf(abs(beta_ivw / se_fe)))
    resid = b2 - beta_ivw * b1
    chi2_het = float(np.sum(w * resid * resid))
    phi = max(chi2_het / float(max(b1.size - 1, 1)), 1.0)
    se_re = float(np.sqrt(phi / info))
    p_re = float(2.0 * stats.norm.sf(abs(beta_ivw / se_re)))
    return beta_ivw, p_fe, p_re


def test_gwas_merge_allele_strand_palindromic_and_missing():
    exp_df = pd.DataFrame(
        {
            "snp": ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs1"],
            "A1": ["A", "A", "A", "A", "A", "A", "A", "A", "G"],
            "A2": ["G", "G", "G", "G", "T", "T", "G", "G", "A"],
            "beta_hat": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, np.nan, 9.99],
            "se": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        }
    )
    out_df = pd.DataFrame(
        {
            "snp": ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs9"],
            "A1": ["A", "G", "T", "C", "A", "T", "A", "A", "A"],
            "A2": ["G", "A", "C", "T", "T", "A", "C", "G", "G"],
            "beta_hat": [0.05, 0.12, 0.18, 0.24, 0.31, 0.41, 0.55, 0.10, 0.20],
            "se": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.01],
        }
    )

    merged_drop = gwas_merge(exp_df, out_df, drop_ambiguous=True, allow_strand_flip=True)
    assert merged_drop["snp"].tolist() == ["rs1", "rs2", "rs3", "rs4"]
    np.testing.assert_allclose(merged_drop["beta_hat_1"].to_numpy(), [0.10, 0.20, 0.30, 0.40])
    # rs1 same (+1), rs2 swap (-1), rs3 strand comp (+1), rs4 strand comp + swap (-1)
    np.testing.assert_allclose(merged_drop["beta_hat_2"].to_numpy(), [0.05, -0.12, 0.18, -0.24])
    assert merged_drop["A1"].tolist() == ["A", "A", "A", "A"]
    assert merged_drop["A2"].tolist() == ["G", "G", "G", "G"]

    # When drop_ambiguous=False, exact-match palindromic rs5 (A/T vs A/T) is retained,
    # whereas swapped/opposite-strand palindromic rs6 (A/T vs T/A) remains excluded.
    merged_keep = gwas_merge(exp_df, out_df, drop_ambiguous=False, allow_strand_flip=True)
    assert merged_keep["snp"].tolist() == ["rs1", "rs2", "rs3", "rs4", "rs5"]
    assert float(merged_keep.loc[merged_keep["snp"] == "rs5", "beta_hat_2"].iloc[0]) == pytest.approx(0.31)


def test_ld_prune_r2_matrix_and_window_kb():
    df = pd.DataFrame(
        {
            "snp": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "chr": ["1", "1", "1", "2", "2"],
            "pos": [100_000, 150_000, 900_000, 100_000, 200_000],
            "beta_hat_1": [0.10, 0.15, 0.08, 0.12, 0.01],
            "seb1": [0.02, 0.02, 0.02, 0.02, 0.02],
            "beta_hat_2": [0.02, 0.03, 0.01, 0.04, 0.00],
            "seb2": [0.02, 0.02, 0.02, 0.02, 0.02],
        }
    )
    # p1 order: rs2 (~6e-14), rs4 (~2e-9), rs1 (~6e-7), rs3 (~6e-5), rs5 (~0.62 > 1e-3)
    pruned_win = ld_prune(df, pval_thresh=1e-3, window_kb=200.0)
    assert pruned_win["snp"].tolist() == ["rs2", "rs4", "rs3"]

    r2 = np.eye(5)
    r2[0, 1] = r2[1, 0] = 0.85  # rs1 in LD with rs2
    r2[2, 3] = r2[3, 2] = 0.25  # rs3 in LD with rs4
    kept_snps = ld_prune(df, r2_matrix=r2, pval_thresh=1e-3, r2_thresh=0.01, return_snps=True)
    assert kept_snps == ["rs2", "rs4"]


@pytest.mark.parametrize("rho_true", [-0.40, 0.45])
def test_est_cause_params_recovers_sample_overlap_rho_and_null_mix(rho_true: float):
    rng = np.random.default_rng(1001 if rho_true < 0 else 1002)
    df = _simulate_cause_gwas(
        rng,
        n_snps=8000,
        n_causal_x=150,
        gamma=0.0,
        eta=0.0,
        q=0.0,
        rho=rho_true,
        n_pleio_snps=150,
    )
    params = est_cause_params(df)
    assert isinstance(params, CauseParams)
    assert abs(params.rho - rho_true) < 0.06
    assert -1.0 < params.rho < 1.0
    assert np.isclose(np.sum(params.pi), 1.0)
    assert params.converged
    null_mask = (params.S1 == 0.0) & (params.S2 == 0.0)
    assert float(np.sum(params.pi[null_mask])) > 0.70


def test_cause_detects_planted_causal_effect():
    rng = np.random.default_rng(2025)
    df = _simulate_cause_gwas(
        rng,
        n_snps=4000,
        n_causal_x=120,
        gamma=0.40,
        eta=0.25,
        q=0.05,
        rho=0.30,
        sigma_x=0.06,
        sigma_u_pleio=0.015,
        n_pleio_snps=80,
    )
    params = est_cause_params(df)
    pruned = ld_prune(df, pval_thresh=1e-3)
    res = cause(df, param_ests=params, variants=pruned["snp"], seed=42)

    assert isinstance(res, CauseResult)
    assert res.p_value < 0.01
    assert res.delta_elpd < 0.0
    assert res.z_score < -2.5
    assert 0.25 <= res.causal.gamma_median <= 0.55
    assert res.causal.gamma_ci_low <= 0.40 <= res.causal.gamma_ci_high


def test_cause_rejects_correlated_pleiotropy_sharing_confounding():
    rng = np.random.default_rng(3030)
    df = _simulate_cause_gwas(
        rng,
        n_snps=4000,
        n_causal_x=180,
        gamma=0.0,
        eta=0.65,
        q=0.35,
        rho=0.25,
        sigma_x=0.08,
        sigma_u_pleio=0.015,
        n_pleio_snps=80,
        se_x=0.012,
        se_y=0.015,
    )
    params = est_cause_params(df)
    pruned = ld_prune(df, pval_thresh=1e-3)
    _, p_ivw_fe, p_ivw_re = _ivw_pvalue(
        pruned["beta_hat_1"].to_numpy(),
        pruned["seb1"].to_numpy(),
        pruned["beta_hat_2"].to_numpy(),
        pruned["seb2"].to_numpy(),
    )
    assert p_ivw_fe < 1e-10  # Naive FE-IVW severely false-positives on correlated pleiotropy
    assert p_ivw_re < 0.05   # Even RE-IVW falsely rejects null under correlated pleiotropy

    res = cause(df, param_ests=params, variants=pruned["snp"], seed=7)
    assert res.p_value > 0.05
    assert res.delta_elpd >= -1.0
    assert abs(res.causal.gamma_median) < 0.15
    assert res.sharing.eta_median > 0.20


def test_cause_null_calibration_under_pure_null_and_sharing():
    rng = np.random.default_rng(4040)
    n_reps = 20
    p_vals_pure: list[float] = []
    p_vals_share: list[float] = []

    for rep in range(n_reps):
        df_pure = _simulate_cause_gwas(
            rng,
            n_snps=1500,
            n_causal_x=80,
            gamma=0.0,
            eta=0.0,
            q=0.0,
            rho=0.30,
            n_pleio_snps=50,
        )
        res_pure = cause(df_pure, pval_thresh=1e-3, n_grid=25, n_q=15, n_samples=1000, seed=rep)
        p_vals_pure.append(res_pure.p_value)

        df_share = _simulate_cause_gwas(
            rng,
            n_snps=1500,
            n_causal_x=90,
            gamma=0.0,
            eta=0.40,
            q=0.25,
            rho=0.30,
            n_pleio_snps=50,
        )
        res_share = cause(df_share, pval_thresh=1e-3, n_grid=25, n_q=15, n_samples=1000, seed=rep)
        p_vals_share.append(res_share.p_value)

    p_pure = np.asarray(p_vals_pure)
    p_share = np.asarray(p_vals_share)
    assert float(np.mean(p_pure < 0.05)) <= 0.10
    assert float(np.mean(p_share < 0.05)) <= 0.10
    assert float(np.mean(p_pure)) >= 0.35
    assert float(np.mean(p_share)) >= 0.35


def test_psis_loo_consistency_with_exact_grid_loo_and_overflow_safety():
    rng = np.random.default_rng(5050)
    n_obs = 15
    # Normal-normal conjugate toy model on discrete grid for exact LOO reference
    theta_grid = np.linspace(-1.5, 1.5, 401)
    y = rng.normal(0.3, 0.6, size=n_obs)
    se = np.full(n_obs, 0.6)
    log_lik_grid = stats.norm.logpdf(y[None, :], loc=theta_grid[:, None], scale=se[None, :])
    log_prior = stats.norm.logpdf(theta_grid, loc=0.0, scale=1.0)
    log_post = np.sum(log_lik_grid, axis=1) + log_prior
    w_grid = np.exp(log_post - np.max(log_post))
    w_grid /= np.sum(w_grid)

    # Exact discrete-grid LOO predictive density: p(y_j | y_{-j}) = 1 / sum_g (w_g / L_{g,j})
    # Compute stably via logsumexp(log(w_g) - log L_{g,j})
    log_w = np.log(np.maximum(w_grid, 1e-300))
    from scipy.special import logsumexp

    exact_elpd = -logsumexp(log_w[:, None] - log_lik_grid, axis=0)

    # Resample S = 4000 draws systematically from w_grid
    n_draws = 4000
    u = (rng.uniform(0.0, 1.0 / n_draws)) + np.arange(n_draws) / n_draws
    cdf = np.cumsum(w_grid)
    cdf[-1] = 1.0
    idx = np.minimum(np.searchsorted(cdf, u, side="left"), w_grid.size - 1)
    loo_res = psis_loo(log_lik_grid[idx])

    assert isinstance(loo_res, PSISLOOResult)
    assert abs(loo_res.elpd - float(np.sum(exact_elpd))) < 0.15
    assert float(np.corrcoef(loo_res.pointwise_elpd, exact_elpd)[0, 1]) > 0.99

    # Overflow safety & constant column test: spread > 900 in column 0, constant in column 1
    ll_extreme = np.zeros((500, 2), dtype=np.float64)
    ll_extreme[:, 0] = -np.concatenate(([950.0], rng.exponential(2.0, size=499)))
    ll_extreme[:, 1] = -3.25
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        ext_res = psis_loo(ll_extreme)
    assert len(caught) == 0
    assert np.all(np.isfinite(ext_res.pointwise_elpd))
    assert np.all(np.isfinite(ext_res.pareto_k))
    assert ext_res.pareto_k[0] > 0.7
    assert ext_res.n_high_pareto_k == 1
    assert ext_res.pareto_k[1] == pytest.approx(0.0)
    assert ext_res.pointwise_p_loo[1] == pytest.approx(0.0)
    assert ext_res.pointwise_elpd[1] == pytest.approx(-3.25)

    # Direct gpdfit check
    k_gpd, s_gpd = gpdfit(rng.pareto(2.0, size=200), wip=True)
    assert np.isfinite(k_gpd) and s_gpd > 0.0


def test_cause_allele_coding_invariance():
    rng = np.random.default_rng(6060)
    df = _simulate_cause_gwas(
        rng, n_snps=1200, n_causal_x=80, gamma=0.35, eta=0.25, q=0.10, rho=0.25
    )
    signs = rng.choice([-1.0, 1.0], size=len(df))
    df_flipped = df.copy()
    df_flipped["beta_hat_1"] = df["beta_hat_1"] * signs
    df_flipped["beta_hat_2"] = df["beta_hat_2"] * signs

    p1 = est_cause_params(df)
    p2 = est_cause_params(df_flipped)
    assert p1.rho == pytest.approx(p2.rho, rel=1e-12, abs=1e-12)

    r1 = cause(df, param_ests=p1, pval_thresh=1e-3, seed=123)
    r2 = cause(df_flipped, param_ests=p2, pval_thresh=1e-3, seed=123)

    assert r1.delta_elpd == pytest.approx(r2.delta_elpd, rel=1e-12, abs=1e-12)
    assert r1.se_delta_elpd == pytest.approx(r2.se_delta_elpd, rel=1e-12, abs=1e-12)
    assert r1.z_score == pytest.approx(r2.z_score, rel=1e-12, abs=1e-12)
    assert r1.p_value == pytest.approx(r2.p_value, rel=1e-12, abs=1e-12)
    assert r1.causal.gamma_median == pytest.approx(r2.causal.gamma_median, rel=1e-12, abs=1e-12)
    assert r1.causal.eta_median == pytest.approx(r2.causal.eta_median, rel=1e-12, abs=1e-12)
    assert r1.causal.q_median == pytest.approx(r2.causal.q_median, rel=1e-12, abs=1e-12)


def test_cause_unit_scale_equivariance():
    rng = np.random.default_rng(7070)
    df = _simulate_cause_gwas(
        rng, n_snps=1500, n_causal_x=100, gamma=0.35, eta=0.25, q=0.10, rho=0.20
    )
    scale = 100.0
    df_scaled = df.copy()
    df_scaled["beta_hat_2"] = df["beta_hat_2"] * scale
    df_scaled["seb2"] = df["seb2"] * scale

    r1 = cause(df, pval_thresh=1e-3, seed=99)
    r2 = cause(df_scaled, pval_thresh=1e-3, seed=99)

    assert r2.params.rho == pytest.approx(r1.params.rho, rel=1e-6, abs=1e-6)
    assert r2.causal.gamma_median == pytest.approx(r1.causal.gamma_median * scale, rel=1e-2)
    assert r2.causal.eta_median == pytest.approx(r1.causal.eta_median * scale, rel=1e-2)
    assert r2.causal.q_median == pytest.approx(r1.causal.q_median, rel=1e-2)
    assert r2.delta_elpd == pytest.approx(r1.delta_elpd, rel=1e-2)
    assert r2.p_value == pytest.approx(r1.p_value, rel=1e-2, abs=1e-3)


def test_cause_api_dataclasses_and_error_handling():
    rng = np.random.default_rng(8080)
    df = _simulate_cause_gwas(
        rng, n_snps=600, n_causal_x=40, gamma=0.30, eta=0.0, q=0.0, rho=0.15
    )

    # Parity across column-naming conventions and raw 1D array input
    df_ts = df.rename(
        columns={
            "beta_hat_1": "beta_exp",
            "seb1": "se_exp",
            "beta_hat_2": "beta_out",
            "seb2": "se_out",
        }
    )
    fmt_df = format_cause_data(df_ts)
    assert list(fmt_df.columns) == ["snp", "beta_hat_1", "seb1", "beta_hat_2", "seb2", "p1", "p2"]

    res_df = cause(df, pval_thresh=1e-3, n_grid=21, n_q=11, n_samples=500, seed=1)
    res_ts = cause(df_ts, pval_thresh=1e-3, n_grid=21, n_q=11, n_samples=500, seed=1)
    res_arr = cause(
        beta_hat_1=df["beta_hat_1"].to_numpy(),
        seb1=df["seb1"].to_numpy(),
        beta_hat_2=df["beta_hat_2"].to_numpy(),
        seb2=df["seb2"].to_numpy(),
        snp=df["snp"].tolist(),
        pval_thresh=1e-3,
        n_grid=21,
        n_q=11,
        n_samples=500,
        seed=1,
    )
    assert res_df.delta_elpd == pytest.approx(res_ts.delta_elpd, rel=1e-12)
    assert res_df.delta_elpd == pytest.approx(res_arr.delta_elpd, rel=1e-12)

    # Frozen dataclass immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        res_df.p_value = 0.99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res_df.params.rho = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res_df.causal.gamma_median = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res_df.causal.loo.elpd = 0.0  # type: ignore[misc]

    # DataFrame & summary schemas
    elpd_df = res_df.to_dataframe()
    assert list(elpd_df.columns) == [
        "model1",
        "model2",
        "delta_elpd",
        "se_delta_elpd",
        "z",
        "p_value",
    ]
    assert len(elpd_df) == 3

    post_df = res_df.posterior_dataframe()
    assert list(post_df.columns) == [
        "model",
        "parameter",
        "median",
        "ci_low",
        "ci_high",
        "mean",
        "sd",
    ]
    assert len(post_df) == 6

    snp_df = res_df.snp_table()
    assert list(snp_df.columns) == [
        "snp",
        "beta_hat_1",
        "seb1",
        "beta_hat_2",
        "seb2",
        "pr_z_sharing",
        "pr_z_causal",
        "elpd_null",
        "elpd_sharing",
        "elpd_causal",
        "pareto_k_sharing",
        "pareto_k_causal",
    ]
    assert len(snp_df) == res_df.n_instruments
    assert len(res_df.params.to_dataframe()) == res_df.params.pi.size
    assert "CAUSE Summary" in res_df.summary()
    assert "CAUSE Nuisance Parameters" in res_df.params.summary()
    assert "PSIS-LOO Summary" in res_df.causal.loo.summary()
    assert isinstance(res_df.sharing, CauseModelFit)

    # Validation & error handling
    df_nan = df.copy()
    df_nan.loc[0, "beta_hat_1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        est_cause_params(df_nan)

    df_bad_se = df.copy()
    df_bad_se.loc[0, "seb2"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        est_cause_params(df_bad_se)

    with pytest.raises(ValueError, match="strictly in"):
        est_cause_params(df, rho=1.0)

    with pytest.raises(ValueError, match="At least 2 instruments"):
        cause(df.iloc[:1])

    with pytest.raises(ValueError, match="qalpha and qbeta"):
        cause(df, qalpha=0.0)

    with pytest.raises(KeyError):
        format_cause_data(pd.DataFrame({"snp": ["rs1"], "foo": [1.0]}))

    with pytest.raises(ValueError):
        compare_elpd({"null": res_df.loo_null})


def test_r_default_settings():
    rng = np.random.default_rng(9090)
    df = _simulate_cause_gwas(
        rng,
        n_snps=2500,
        n_causal_x=120,
        gamma=0.45,
        eta=0.0,
        q=0.0,
        rho=0.45,
        sigma_x=0.08,
        n_pleio_snps=40,
    )

    # 1. Invalid settings raises ValueError on both est_cause_params and cause
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        est_cause_params(df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        cause(df, pval_thresh=1e-3, settings="bad")  # type: ignore[arg-type]

    # 2. Variance grid includes 2 * max and 4 * max tail SD points (4x and 16x tail variances)
    p_def = est_cause_params(df, settings="default")
    p_r = est_cause_params(df, settings="R_default")
    assert p_def.grid_S1 is not None and p_r.grid_S1 is not None
    assert p_r.grid_S1.size == p_def.grid_S1.size + 2
    assert p_r.grid_S2.size == p_def.grid_S2.size + 2
    np.testing.assert_allclose(p_r.grid_S1[:-2], p_def.grid_S1, rtol=1e-13)
    assert p_r.grid_S1[-2] == pytest.approx(4.0 * p_def.grid_S1[-1], rel=1e-13)
    assert p_r.grid_S1[-1] == pytest.approx(16.0 * p_def.grid_S1[-1], rel=1e-13)

    # 3. Joint optimization of rho with arctanh(rho) ~ N(0, 0.5^2) prior in est_cause_params
    assert p_r.converged
    assert abs(p_r.rho - 0.45) < 0.10
    assert p_r.n_null_snps_rho == len(df)
    # Verify N(0, 0.5^2) prior on arctanh(rho) regularizes extreme rho toward 0 on small N
    df_small = _simulate_cause_gwas(
        np.random.default_rng(4242),
        n_snps=80,
        n_causal_x=0,
        gamma=0.0,
        eta=0.0,
        q=0.0,
        rho=0.80,
        n_pleio_snps=0,
    )
    p_small_def = est_cause_params(df_small, settings="default")
    p_small_r = est_cause_params(df_small, settings="R_default")
    assert abs(p_small_r.rho) < abs(p_small_def.rho)

    # 4. Exact sigma_g via P(|N(0, sigma_g)| > max(|b2 / b1|)) = 0.05 on SNPs with p1 < 1e-3
    #    and default n_samples = 1000 (vs 2000 in "default")
    res_def = cause(df, param_ests=p_def, pval_thresh=1e-3, n_grid=21, n_q=11, seed=5, settings="default")
    res_r = cause(df, param_ests=p_r, pval_thresh=1e-3, n_grid=21, n_q=11, seed=5, settings="R_default")
    assert res_def.n_samples == 2000 and res_def.causal.log_lik_draws.shape[0] == 2000
    assert res_r.n_samples == 1000 and res_r.causal.log_lik_draws.shape[0] == 1000

    p1_inst = 2.0 * stats.norm.sf(np.abs(res_r.beta_hat_1 / res_r.seb1))
    strong = p1_inst < 1e-3
    assert np.any(strong)
    max_ratio = float(np.max(np.abs(res_r.beta_hat_2[strong] / res_r.beta_hat_1[strong])))
    tail_prob = float(2.0 * stats.norm.sf(max_ratio, loc=0.0, scale=res_r.sigma_g))
    assert tail_prob == pytest.approx(0.05, rel=1e-12)
    assert res_r.p_value < 0.05

    # Explicit n_samples override is respected in R_default
    res_r_600 = cause(
        df, param_ests=p_r, pval_thresh=1e-3, n_grid=21, n_q=11, n_samples=600, seed=5, settings="R_default"
    )
    assert res_r_600.n_samples == 600 and res_r_600.causal.log_lik_draws.shape[0] == 600

    # Fallback when no SNPs satisfy p1 < 1e-3
    weak_df = pd.DataFrame(
        {
            "snp": [f"rs_w{i}" for i in range(10)],
            "beta_hat_1": np.linspace(0.01, 0.03, 10),
            "seb1": np.full(10, 0.02),
            "beta_hat_2": np.linspace(0.005, 0.015, 10),
            "seb2": np.full(10, 0.02),
        }
    )
    res_weak = cause(weak_df, n_grid=15, n_q=9, n_samples=200, seed=1, settings="R_default")
    assert np.isfinite(res_weak.sigma_g) and res_weak.sigma_g > 0.0
