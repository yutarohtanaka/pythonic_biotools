"""Comprehensive unit tests for `mrpresso` (Verbanck et al., Nature Genetics 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mrpresso import MRPressoResult, MVMRPressoResult, leave_one_out_ivw, mr_presso, mvmr_presso


def _brute_force_loo(X: np.ndarray, y: np.ndarray, sy: np.ndarray, weighted_rss: bool = False):
    """Reference brute-force leave-one-out weighted least squares for exact verification."""
    n, p = X.shape
    w = 1.0 / (sy**2)
    loo_betas = np.zeros((n, p), dtype=np.float64)
    rss_j = np.zeros(n, dtype=np.float64)
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        Xm, ym, wm = X[mask], y[mask], w[mask]
        beta_j = np.linalg.solve((Xm.T * wm) @ Xm, (Xm.T * wm) @ ym)
        loo_betas[j] = beta_j
        resid = y[j] - float(X[j] @ beta_j)
        rss_j[j] = (w[j] * resid**2) if weighted_rss else (resid**2)
    return loo_betas, rss_j, float(np.sum(rss_j))


def test_exact_leave_one_out_univariable_and_sherman_morrison():
    rng = np.random.default_rng(101)
    # Univariable
    x = rng.normal(0.15, 0.08, size=25)
    sy = rng.uniform(0.01, 0.04, size=25)
    y = 0.6 * x + rng.normal(0.0, sy)

    loo_1d, rss_j_1d, rss_tot_1d = leave_one_out_ivw(y, x, sy, weighted_rss=False)
    bf_b, bf_rss_j, bf_tot = _brute_force_loo(x[:, None], y, sy, weighted_rss=False)
    np.testing.assert_allclose(loo_1d, bf_b[:, 0], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(rss_j_1d, bf_rss_j, rtol=1e-13, atol=1e-15)
    assert np.isclose(rss_tot_1d, bf_tot, rtol=1e-13)

    # Multivariable P = 3 Sherman-Morrison exact match (both unweighted and weighted RSS)
    X3 = rng.normal(0.1, 0.12, size=(30, 3))
    sy3 = rng.uniform(0.015, 0.05, size=30)
    y3 = X3 @ np.array([0.5, -0.35, 0.2]) + rng.normal(0.0, sy3)
    for weighted in (False, True):
        loo_3d, rss_j_3d, rss_tot_3d = leave_one_out_ivw(y3, X3, sy3, weighted_rss=weighted)
        bf_b3, bf_rss3, bf_tot3 = _brute_force_loo(X3, y3, sy3, weighted_rss=weighted)
        np.testing.assert_allclose(loo_3d, bf_b3, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(rss_j_3d, bf_rss3, rtol=1e-12, atol=1e-14)
        assert np.isclose(rss_tot_3d, bf_tot3, rtol=1e-12)


def test_noiseless_collinear_null_exact():
    bx = np.linspace(0.05, 0.30, 20)
    sx = np.full(20, 0.01)
    sy = np.full(20, 0.02)
    by = 0.75 * bx

    res = mr_presso(by, bx, sy, sx, nb_distribution=1000, seed=7)
    assert isinstance(res, MRPressoResult)
    assert np.isclose(res.beta_raw, 0.75, atol=1e-14)
    assert np.isclose(res.se_raw, 0.0, atol=1e-14)
    assert np.isclose(res.rss_obs, 0.0, atol=1e-15)
    assert np.isclose(res.p_global, 1.0, atol=1e-12)
    assert res.outlier_indices == []
    assert res.beta_corrected is None
    assert res.p_distortion is None


def test_directional_outliers_detected_and_corrected():
    rng = np.random.default_rng(2024)
    n = 35
    bx = rng.uniform(0.08, 0.25, size=n)
    sx = np.full(n, 0.008)
    sy = np.full(n, 0.015)
    true_beta = 0.40
    by = true_beta * bx + rng.normal(0.0, sy)

    # Plant two strong directional horizontal pleiotropy outliers (~9-10 SDs)
    by[2] += 0.14
    by[17] += 0.16

    res = mr_presso(by, bx, sy, sx, nb_distribution=4000, seed=42)
    assert res.p_global < 0.001
    assert res.p_global_formatted.startswith("<")
    assert set(res.outlier_indices) == {2, 17}
    assert res.beta_corrected is not None
    assert abs(res.beta_raw - true_beta) > 0.05
    assert abs(res.beta_corrected - true_beta) < 0.03
    assert res.distortion_coeff is not None and res.distortion_coeff > 10.0
    assert res.p_distortion is not None and res.p_distortion < 0.02

    # Analytical chi-square outlier mode identifies the exact same outliers
    res_chi = mr_presso(by, bx, sy, sx, nb_distribution=4000, outlier_method="chisq", seed=42)
    assert set(res_chi.outlier_indices) == {2, 17}
    assert np.isclose(res_chi.beta_corrected, res.beta_corrected, rtol=1e-12)


def test_balanced_pleiotropy_global_significant_without_distortion():
    rng = np.random.default_rng(99)
    n = 40
    bx = np.full(n, 0.15)
    sx = np.full(n, 0.005)
    sy = np.full(n, 0.012)
    by = 0.50 * bx + rng.normal(0.0, 0.004, size=n)
    # Two equal and opposite outliers cancel out in the raw IVW slope
    by[0] += 0.30
    by[1] -= 0.30

    res = mr_presso(by, bx, sy, sx, nb_distribution=3000, seed=99)
    assert res.p_global < 0.002
    assert set(res.outlier_indices) == {0, 1}
    assert res.beta_corrected is not None
    assert abs(res.beta_raw - res.beta_corrected) < 0.01
    assert res.p_distortion is not None and res.p_distortion > 0.25


def test_null_calibration_nonzero_causal_effect():
    """Verify Uniform(0, 1) calibration under H0 with non-zero causal slope and exposure error."""
    rng = np.random.default_rng(31415)
    n_reps = 80
    n_snps = 25
    true_beta = 0.65
    p_globals = np.empty(n_reps, dtype=np.float64)

    for r in range(n_reps):
        mu_x = rng.uniform(0.08, 0.25, size=n_snps)
        sx = rng.uniform(0.006, 0.012, size=n_snps)
        sy = rng.uniform(0.015, 0.030, size=n_snps)
        bx = mu_x + rng.normal(0.0, sx)
        by = true_beta * mu_x + rng.normal(0.0, sy)
        fit = mr_presso(by, bx, sy, sx, outlier_test=False, nb_distribution=600, seed=1000 + r)
        p_globals[r] = fit.p_global

    type1_rate = float(np.mean(p_globals < 0.05))
    assert 0.0 <= type1_rate <= 0.12
    ks = stats.kstest(p_globals, "uniform")
    assert ks.pvalue > 0.02


def test_allele_coding_invariance_univariable_and_multivariable():
    rng = np.random.default_rng(777)
    n = 30
    bx = rng.uniform(0.07, 0.25, size=n)
    sx = rng.uniform(0.005, 0.012, size=n)
    sy = rng.uniform(0.01, 0.025, size=n)
    by = -0.35 * bx + rng.normal(0.0, sy)
    by[5] += 0.28

    flip_signs = rng.choice([-1.0, 1.0], size=n)
    res_orig = mr_presso(by, bx, sy, sx, nb_distribution=2500, seed=555)
    res_flip = mr_presso(by * flip_signs, bx * flip_signs, sy, sx, nb_distribution=2500, seed=555)

    assert np.isclose(res_orig.beta_raw, res_flip.beta_raw, rtol=1e-13)
    assert np.isclose(res_orig.rss_obs, res_flip.rss_obs, rtol=1e-13)
    assert np.isclose(res_orig.p_global, res_flip.p_global, rtol=1e-13)
    assert res_orig.outlier_indices == res_flip.outlier_indices
    assert np.isclose(res_orig.beta_corrected, res_flip.beta_corrected, rtol=1e-13)
    assert np.isclose(res_orig.distortion_coeff, res_flip.distortion_coeff, rtol=1e-13)
    assert np.isclose(res_orig.p_distortion, res_flip.p_distortion, rtol=1e-13)


def test_multivariable_mr_presso_and_dataframe_interface():
    rng = np.random.default_rng(888)
    n = 45
    X = rng.normal(0.12, 0.09, size=(n, 2))
    SX = np.full((n, 2), 0.008)
    sy = np.full(n, 0.018)
    true_b = np.array([0.45, -0.30])
    y = X @ true_b + rng.normal(0.0, sy)
    y[3] += 0.38
    y[21] -= 0.42

    df = pd.DataFrame(
        {
            "Y_b": y,
            "Y_se": sy,
            "E1_b": X[:, 0],
            "E2_b": X[:, 1],
            "E1_se": SX[:, 0],
            "E2_se": SX[:, 1],
        },
        index=[f"rs{i + 1}" for i in range(n)],
    )

    mv_res = mr_presso(
        "Y_b",
        ["E1_b", "E2_b"],
        "Y_se",
        ["E1_se", "E2_se"],
        data=df,
        nb_distribution=3000,
        seed=123,
    )
    assert isinstance(mv_res, MVMRPressoResult)
    assert mv_res.p_global < 0.001
    assert set(mv_res.outlier_snps) == {"rs4", "rs22"}
    assert mv_res.beta_corrected is not None
    np.testing.assert_allclose(mv_res.beta_corrected, true_b, atol=0.05)
    assert mv_res.p_distortion_joint is not None

    main_df = mv_res.to_dataframe()
    snp_df = mv_res.outlier_table()
    assert main_df.shape[0] == 4
    assert snp_df.shape[0] == n
    assert "Multivariable MR-PRESSO" in mv_res.summary()

    # Direct mvmr_presso array call matches DataFrame call identically
    mv_direct = mvmr_presso(
        y, X, sy, SX, snp_ids=list(df.index), exposure_names=["E1_b", "E2_b"], nb_distribution=3000, seed=123
    )
    np.testing.assert_allclose(mv_direct.beta_corrected, mv_res.beta_corrected, rtol=1e-13)


def test_serialization_and_weighted_rss_option():
    rng = np.random.default_rng(12)
    bx = rng.uniform(0.05, 0.20, size=20)
    sx = np.full(20, 0.01)
    sy = rng.uniform(0.01, 0.03, size=20)
    by = 0.3 * bx + rng.normal(0.0, sy)
    by[0] += 0.25

    res = mr_presso(by, bx, sy, sx, weighted_rss=True, nb_distribution=2000, seed=12)
    assert res.weighted_rss is True
    assert res.p_global < 0.01
    assert 0 in res.outlier_indices
    df_main = res.to_dataframe()
    df_out = res.outlier_table()
    assert list(df_main["mr_analysis"]) == ["Raw", "Outlier-corrected"]
    assert df_out.loc[0, "is_outlier"] == True  # noqa: E712
    assert "MR-PRESSO Summary" in res.summary()


def test_invalid_inputs_raise_value_error():
    bx = np.array([0.1, 0.2, 0.3, 0.4])
    by = np.array([0.05, 0.1, 0.15, 0.2])
    sx = np.array([0.01, 0.01, 0.01, 0.01])
    sy = np.array([0.02, 0.02, 0.02, 0.02])

    # NaN / Inf checks
    by_nan = by.copy()
    by_nan[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mr_presso(by_nan, bx, sy, sx)

    bx_inf = bx.copy()
    bx_inf[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        mr_presso(by, bx_inf, sy, sx)

    # Non-positive SEs
    sy_bad = sy.copy()
    sy_bad[2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        mr_presso(by, bx, sy_bad, sx)

    sx_bad = sx.copy()
    sx_bad[0] = -0.01
    with pytest.raises(ValueError, match="positive"):
        mr_presso(by, bx, sy, sx_bad)

    # Too few instruments (< 3 for univariable)
    with pytest.raises(ValueError, match="instruments"):
        mr_presso(by[:2], bx[:2], sy[:2], sx[:2])

    # Invalid threshold or nb_distribution
    with pytest.raises(ValueError, match="signif_threshold"):
        mr_presso(by, bx, sy, sx, signif_threshold=1.5)
    with pytest.raises(ValueError, match="nb_distribution"):
        mr_presso(by, bx, sy, sx, nb_distribution=5)


def test_r_default_settings():
    rng = np.random.default_rng(2026)
    n = 30
    bx = rng.uniform(0.08, 0.25, size=n)
    sx = rng.uniform(0.015, 0.030, size=n)
    sy = rng.uniform(0.012, 0.028, size=n)
    true_beta = 0.45
    by = true_beta * bx + rng.normal(0.0, sy * 0.6)
    # Plant two strong directional pleiotropy outliers
    by[4] += 0.25
    by[19] += 0.28

    # 1. Invalid settings raises ValueError across entrypoints
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        mr_presso(by, bx, sy, sx, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        mvmr_presso(
            by,
            np.column_stack([bx, bx * 0.5 + 0.02]),
            sy,
            np.column_stack([sx, sx]),
            settings="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        leave_one_out_ivw(by, bx, sy, settings="bad")  # type: ignore[arg-type]

    # 2. Defaults in R_default: weighted_rss=True, nb_distribution=1000; explicit overrides respected
    res_def = mr_presso(by, bx, sy, sx, seed=11, settings="default")
    res_r = mr_presso(by, bx, sy, sx, seed=11, settings="R_default")
    assert res_def.weighted_rss is False and res_def.nb_distribution == 5000
    assert res_r.weighted_rss is True and res_r.nb_distribution == 1000

    res_r_custom = mr_presso(
        by, bx, sy, sx, nb_distribution=400, weighted_rss=False, seed=11, settings="R_default"
    )
    assert res_r_custom.weighted_rss is False and res_r_custom.nb_distribution == 400

    # 3. Weighted RSS matches 1 / sy^2 weighting in getRSS_LOO
    loo_b_r, rss_j_r, rss_tot_r = leave_one_out_ivw(by, bx, sy, settings="R_default")
    expected_rss_j = ((by - loo_b_r * bx) ** 2) / (sy**2)
    np.testing.assert_allclose(rss_j_r, expected_rss_j, rtol=1e-13)
    assert np.isclose(res_r.rss_obs, rss_tot_r, rtol=1e-13)

    # 4. Raw count / K empirical p-values (0 / K == 0.0 vs (0 + 1) / (K + 1))
    res_def_k = mr_presso(by, bx, sy, sx, nb_distribution=500, seed=99, settings="default")
    res_r_k = mr_presso(by, bx, sy, sx, nb_distribution=500, seed=99, settings="R_default")
    assert res_def_k.p_global == pytest.approx(1.0 / 501.0)
    assert res_r_k.p_global == 0.0
    assert set(res_r_k.outlier_indices) == {4, 19}
    assert res_r_k.p_distortion == 0.0
    assert res_def_k.p_distortion >= 1.0 / 501.0

    # 5. Simulated Outlier Test uses fixed observed LOO slope (mr_presso.R L103-105)
    from mrpresso.core import _canonicalize_signs, _simulate_rss_univariable

    X_canon, y_canon = _canonicalize_signs(bx[:, None], by)
    x_c = X_canon[:, 0]
    loo_c, r_obs_sq_c, _ = leave_one_out_ivw(y_canon, x_c, sy, weighted_rss=True)
    pred_c = loo_c * x_c
    w = 1.0 / (sy**2)

    K = 200
    _, p_emp_r, exp_rss_r = _simulate_rss_univariable(
        x_c, sx, pred_c, sy, w, r_obs_sq_c, K, True, np.random.default_rng(321),
        loo_betas_obs=loo_c, settings="R_default",
    )
    _, p_emp_def, exp_rss_def = _simulate_rss_univariable(
        x_c, sx, pred_c, sy, w, r_obs_sq_c, K, True, np.random.default_rng(321),
        loo_betas_obs=loo_c, settings="default",
    )

    # Direct single-batch reference against fixed observed LOO slope
    ref_rng = np.random.default_rng(321)
    x_sim_ref = x_c[None, :] + sx[None, :] * ref_rng.standard_normal((K, n))
    y_sim_ref = pred_c[None, :] + sy[None, :] * ref_rng.standard_normal((K, n))
    r_sq_fixed_ref = ((y_sim_ref - x_sim_ref * loo_c[None, :]) ** 2) * w[None, :]
    np.testing.assert_allclose(exp_rss_r, np.mean(r_sq_fixed_ref, axis=0), rtol=1e-13)
    np.testing.assert_allclose(p_emp_r, np.mean(r_sq_fixed_ref >= r_obs_sq_c[None, :], axis=0), rtol=1e-13)
    assert not np.allclose(exp_rss_r, exp_rss_def)

    # 6. Distortion Test BiasObs = (Beta_all - Beta_no_outliers) / |Beta_no_outliers| with raw count / K
    from mrpresso.core import _bootstrap_distortion

    keep = np.ones(n, dtype=bool)
    keep[[4, 19]] = False
    b_raw_arr = np.array([res_r.beta_raw])
    b_corr_arr = np.array([res_r.beta_corrected])
    dc_r, pd_r, _ = _bootstrap_distortion(
        X_canon[keep], y_canon[keep], w[keep], b_raw_arr, b_corr_arr, 300,
        np.random.default_rng(777), settings="R_default",
    )
    boot_rng = np.random.default_rng(777)
    n_val = int(np.sum(keep))
    idx_b = boot_rng.integers(0, n_val, size=(300, n_val))
    xb = X_canon[keep][idx_b, 0]
    yb = y_canon[keep][idx_b]
    wb = w[keep][idx_b]
    b_boot = np.sum(wb * xb * yb, axis=1) / np.sum(wb * (xb**2), axis=1)
    bias_obs_ref = (b_raw_arr[0] - b_corr_arr[0]) / abs(b_corr_arr[0])
    bias_exp_ref = (b_boot - b_corr_arr[0]) / abs(b_corr_arr[0])
    p_dist_ref = float(np.mean(np.abs(bias_exp_ref) >= abs(bias_obs_ref)))
    assert np.isclose(dc_r[0], 100.0 * bias_obs_ref, rtol=1e-13)
    assert np.isclose(pd_r[0], p_dist_ref, rtol=1e-13)
