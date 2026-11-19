"""
Comprehensive verification suite for `variancepartition`:
- Formula parser vs DesignSpec.from_matrices parity
- Spectral vs MME REML vs brute-force N x N NLL & finite-difference gradient oracle
- Planted-truth ICC variance recovery (single & crossed random effects)
- Sum-to-1, reference-level encoding invariance, and weight scale invariance
- Exact Satterthwaite df on balanced between/within-subject split-plot designs
- R=0 parity with WLS + Empirical Bayes moderation (`limma`)
- `dream` + `squeeze_var` DE power, `robust=True`, `trend=True`, and `z.std`
- Null calibration under repeated measures (dream vs inflated naive OLS)
- Boundary `sigma_r^2 = 0`, constant gene, collinear random/fixed, rank-deficient X
- AnnData (CSR sparse + layer weights), `genes_as_rows=True`, NaN-preserving BH-FDR
- CI isolation (no patsy/statsmodels/joblib) & frozen dataclass contract
"""

from __future__ import annotations

import dataclasses
import sys
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma
from scipy.stats import false_discovery_control, ks_1samp, uniform

import variancepartition as vp
from variancepartition import (
    DesignSpec,
    DreamFit,
    DreamResult,
    LMMFitResult,
    VarPartResult,
    build_design,
    dream,
    e_bayes,
    extract_var_part,
    fit_extract_var_part_model,
    fit_f_dist,
    fit_lmm,
    make_contrasts,
    squeeze_var,
    top_table,
    trigamma_inverse,
)


def _naive_nn_reml_nll(
    y: np.ndarray,
    X: np.ndarray,
    Z_list: list[np.ndarray],
    w: np.ndarray,
    sigma_r_sq: np.ndarray,
    sigma_e_sq: float,
) -> float:
    """Brute-force O(N^3) REML negative log-likelihood oracle."""
    N, p = X.shape
    V = np.diag(sigma_e_sq / w)
    for sr2, Z_r in zip(sigma_r_sq, Z_list):
        V = V + sr2 * (Z_r @ Z_r.T)
    L_V, _ = cho_factor(V, lower=True)
    logdet_V = 2.0 * float(np.sum(np.log(np.diag(L_V))))
    Vinv_X = cho_solve((L_V, True), X)
    XtVinvX = X.T @ Vinv_X
    L_X, _ = cho_factor(XtVinvX, lower=True)
    logdet_XtVinvX = 2.0 * float(np.sum(np.log(np.diag(L_X))))
    Vinv_y = cho_solve((L_V, True), y)
    beta = cho_solve((L_X, True), X.T @ Vinv_y)
    r = y - X @ beta
    Vinv_r = cho_solve((L_V, True), r)
    quad = float(r @ Vinv_r)
    return 0.5 * ((N - p) * np.log(2.0 * np.pi) + logdet_V + logdet_XtVinvX + quad)


def test_formula_parser_and_explicit_matrix_parity() -> None:
    rng = np.random.default_rng(101)
    N, G = 48, 25
    subjects = np.repeat([f"S{i}" for i in range(12)], 4)
    batches = np.tile(["B1", "B2", "B3", "B4"], 12)
    cond = np.tile(["ctrl", "ctrl", "treated", "treated"], 12)
    age = rng.normal(40.0, 8.0, size=N)

    meta = pd.DataFrame(
        {
            "condition": pd.Categorical(cond, categories=["ctrl", "treated"]),
            "age": age,
            "subject": subjects,
            "batch": batches,
        },
        index=[f"samp_{i}" for i in range(N)],
    )
    Y = rng.normal(0.0, 1.0, size=(N, G))
    # Add real effects so variance components are non-boundary
    subj_eff = rng.normal(0.0, 1.2, size=(12, G))
    subj_codes = pd.Categorical(subjects).codes
    Y += subj_eff[subj_codes] + 0.8 * (cond == "treated")[:, None]

    res_formula = fit_extract_var_part_model(
        Y, "~ condition + age + (1 | subject) + (1 | batch)", data=meta
    )

    spec_formula = build_design("~ condition + age + (1 | subject) + (1 | batch)", meta)
    spec_manual = DesignSpec.from_matrices(
        spec_formula.X,
        random_effects={"subject": meta["subject"], "batch": meta["batch"]},
        fixed_names=spec_formula.fixed_names,
        fixed_terms=spec_formula.fixed_terms,
        sample_names=meta.index.tolist(),
    )
    res_spec = fit_extract_var_part_model(Y, spec_manual)
    res_mat = fit_extract_var_part_model(
        Y,
        design=spec_formula.X,
        random_effects=[spec_formula.random_matrices[0], spec_formula.random_matrices[1]],
    )

    assert_allclose(res_formula.fractions, res_spec.fractions, atol=1e-10, rtol=1e-10)
    assert_allclose(res_formula.variances, res_spec.variances, atol=1e-10, rtol=1e-10)
    assert_allclose(res_formula.fractions[:, -1], res_mat.fractions[:, -1], atol=1e-10)

    # Nested and uncorrelated random slope formulas
    spec_nested = build_design("~ condition + (1 | batch/subject)", meta)
    assert spec_nested.random_names == ["batch", "batch:subject"]
    assert spec_nested.n_random_terms == 2

    spec_uncorr = build_design("~ condition + (1 + age || subject)", meta)
    assert spec_uncorr.random_names == ["subject", "subject:age"]
    assert spec_uncorr.n_random_terms == 2

    spec_noint = build_design("~ 0 + condition + (1 | subject)", meta)
    assert not spec_noint.has_intercept
    assert spec_noint.fixed_names == ["condition_ctrl", "condition_treated"]

    # Invalid formulas must raise ValueError
    with pytest.raises(ValueError, match="Correlated"):
        build_design("~ condition + (1 + age | subject)", meta)
    with pytest.raises(ValueError):
        build_design("~ missing_col + (1 | subject)", meta)
    with pytest.raises(ValueError):
        build_design("condition + (1 | subject)", meta)


def test_spectral_vs_general_mme_and_naive_oracle() -> None:
    rng = np.random.default_rng(202)
    N, G = 60, 30
    subjects = np.repeat([f"subj_{i}" for i in range(15)], 4)
    batches = np.tile([f"b_{k}" for k in range(3)], 20)
    trt = np.tile(["ctrl", "trt"], 30)
    meta = pd.DataFrame({"trt": trt, "subject": subjects, "batch": batches})
    w_shared = rng.uniform(0.5, 2.5, size=N)

    # Generate signal with random subject + random batch
    spec1 = build_design("~ trt + (1 | subject)", meta)
    Z1 = spec1.random_matrices[0]
    u1 = rng.normal(0.0, 1.1, size=(Z1.shape[1], G))
    eps = rng.normal(0.0, 0.7, size=(N, G)) / np.sqrt(w_shared)[:, None]
    Y = 5.0 + 0.9 * spec1.X[:, 1:2] + Z1 @ u1 + eps

    fit_spec = fit_lmm(Y, spec1, weights=w_shared, _force_mme=False)
    fit_mme = fit_lmm(Y, spec1, weights=w_shared, _force_mme=True)

    assert_allclose(fit_spec.betas, fit_mme.betas, atol=1e-7, rtol=1e-7)
    assert_allclose(fit_spec.sigma_r_sq, fit_mme.sigma_r_sq, atol=1e-7, rtol=1e-7)
    assert_allclose(fit_spec.sigma_e_sq, fit_mme.sigma_e_sq, atol=1e-7, rtol=1e-7)
    assert_allclose(fit_spec.reml_nll, fit_mme.reml_nll, atol=1e-7, rtol=1e-7)

    # Check exact REML NLL against brute-force O(N^3) oracle for R=1 and R=2
    for g in range(5):
        nll_oracle = _naive_nn_reml_nll(
            Y[:, g],
            spec1.X,
            spec1.random_matrices,
            w_shared,
            fit_spec.sigma_r_sq[g],
            float(fit_spec.sigma_e_sq[g]),
        )
        assert_allclose(fit_spec.reml_nll[g], nll_oracle, atol=1e-9, rtol=1e-9)

    # Check R=2 REML NLL & analytical gradient via central finite differences
    spec2 = build_design("~ trt + (1 | subject) + (1 | batch)", meta)
    Z2 = spec2.random_matrices[1]
    Y2 = Y + Z2 @ rng.normal(0.0, 0.8, size=(Z2.shape[1], G))
    w_2d = rng.uniform(0.4, 3.0, size=(N, G))
    fit_r2 = fit_lmm(Y2[:, :5], spec2, weights=w_2d[:, :5])
    for g in range(5):
        nll_oracle2 = _naive_nn_reml_nll(
            Y2[:, g],
            spec2.X,
            spec2.random_matrices,
            w_2d[:, g],
            fit_r2.sigma_r_sq[g],
            float(fit_r2.sigma_e_sq[g]),
        )
        assert_allclose(fit_r2.reml_nll[g], nll_oracle2, atol=1e-8, rtol=1e-8)

        # Verify analytical gradient in variancepartition.lmm._mme_profile_eval
        # matches central finite differences and brute-force N x N REML NLL
        from variancepartition.lmm import _mme_profile_eval, _solve_mme_single_gene

        d_g = np.sqrt(w_2d[:, g])
        X_w = d_g[:, None] * spec2.X
        Z_w = d_g[:, None] * np.hstack(spec2.random_matrices)
        y_w = d_g * Y2[:, g]
        beta_ols = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
        e_ols = y_w - X_w @ beta_ols
        S_XX_g = X_w.T @ X_w
        S_ZZ_g = Z_w.T @ Z_w
        S_XZ_g = X_w.T @ Z_w
        s_Zy_g = Z_w.T @ e_ols
        s_yy_g = float(e_ols @ e_ols)
        b_sizes = np.array([Z1.shape[1], Z2.shape[1]])
        b_slices = [slice(0, Z1.shape[1]), slice(Z1.shape[1], Z1.shape[1] + Z2.shape[1])]
        df_res_g = float(N - spec2.n_fixed)

        gamma_test = np.array([0.8, 0.35], dtype=np.float64)
        f_prof, grad_analytic, se2_prof, *_ = _mme_profile_eval(
            gamma_test,
            S_XX=S_XX_g,
            S_ZZ=S_ZZ_g,
            S_XZ=S_XZ_g,
            s_Zy_resid=s_Zy_g,
            s_yy_resid=s_yy_g,
            df_resid=df_res_g,
            block_sizes=b_sizes,
            block_slices=b_slices,
            gamma_max=1e6,
            need_full=True,
        )
        grad_fd = np.empty(2, dtype=np.float64)
        eps_fd = 1e-6
        for k in range(2):
            gp = gamma_test.copy()
            gm = gamma_test.copy()
            gp[k] += eps_fd
            gm[k] -= eps_fd
            fp, _ = _mme_profile_eval(
                gp,
                S_XX=S_XX_g,
                S_ZZ=S_ZZ_g,
                S_XZ=S_XZ_g,
                s_Zy_resid=s_Zy_g,
                s_yy_resid=s_yy_g,
                df_resid=df_res_g,
                block_sizes=b_sizes,
                block_slices=b_slices,
                gamma_max=1e6,
            )
            fm, _ = _mme_profile_eval(
                gm,
                S_XX=S_XX_g,
                S_ZZ=S_ZZ_g,
                S_XZ=S_XZ_g,
                s_Zy_resid=s_Zy_g,
                s_yy_resid=s_yy_g,
                df_resid=df_res_g,
                block_sizes=b_sizes,
                block_slices=b_slices,
                gamma_max=1e6,
            )
            grad_fd[k] = (fp - fm) / (2.0 * eps_fd)
        assert_allclose(grad_analytic, grad_fd, atol=1e-7, rtol=1e-7)

        res_test = _solve_mme_single_gene(
            S_XX=S_XX_g,
            S_ZZ=S_ZZ_g,
            S_XZ=S_XZ_g,
            s_Zy_resid=s_Zy_g,
            s_yy_resid=s_yy_g,
            alpha_ols=beta_ols,
            y_w_norm_sq=float(y_w @ y_w),
            logdet_w=float(np.sum(np.log(w_2d[:, g]))),
            df_resid=df_res_g,
            block_sizes=b_sizes,
            block_slices=b_slices,
            gamma_max=1e6,
            max_iter=50,
            tol=1e-8,
            compute_fisher=False,
            gamma_override=gamma_test,
        )
        nll_oracle_prof = _naive_nn_reml_nll(
            Y2[:, g], spec2.X, spec2.random_matrices, w_2d[:, g], gamma_test * se2_prof, se2_prof
        )
        assert_allclose(res_test[5], nll_oracle_prof, atol=1e-9)


def test_planted_truth_icc_recovery_single_and_crossed_re() -> None:
    rng = np.random.default_rng(303)
    m_subj, m_batch = 30, 6
    N = 120
    subjects = np.repeat(np.arange(m_subj), 4)
    batches = rng.integers(0, m_batch, size=N)
    cond = np.tile(["A", "B"], N // 2)
    meta = pd.DataFrame(
        {
            "condition": cond,
            "subject": [f"s{s}" for s in subjects],
            "batch": [f"b{b}" for b in batches],
        }
    )
    spec = build_design("~ condition + (1 | subject) + (1 | batch)", meta)
    Z_s, Z_b = spec.random_matrices
    x_c = spec.X[:, 1]
    x_c_var = float(np.var(x_c, ddof=1))

    G = 120
    Y = np.empty((N, G), dtype=np.float64)
    true_fracs = np.empty((G, 4), dtype=np.float64)  # condition, subject, batch, Residuals

    for g in range(G):
        if g < 40:
            # Subject-dominated (~0.70)
            b_c, s_s, s_b, s_e = 0.2, np.sqrt(0.70), np.sqrt(0.08), np.sqrt(0.20)
        elif g < 80:
            # Batch-dominated (~0.55)
            b_c, s_s, s_b, s_e = 0.2, np.sqrt(0.10), np.sqrt(0.55), np.sqrt(0.33)
        else:
            # Condition-dominated (~0.60)
            b_c, s_s, s_b, s_e = 1.6, np.sqrt(0.10), np.sqrt(0.05), np.sqrt(0.25)

        u_s = rng.normal(0.0, s_s, size=m_subj)
        u_b = rng.normal(0.0, s_b, size=m_batch)
        eps = rng.normal(0.0, s_e, size=N)
        Y[:, g] = 3.0 + b_c * x_c + Z_s @ u_s + Z_b @ u_b + eps

        v_c = (b_c**2) * x_c_var
        v_s = s_s**2
        v_b = s_b**2
        v_e = s_e**2
        tot = v_c + v_s + v_b + v_e
        true_fracs[g] = [v_c / tot, v_s / tot, v_b / tot, v_e / tot]

    vp_res = fit_extract_var_part_model(
        Y, "~ condition + (1 | subject) + (1 | batch)", data=meta
    )
    assert isinstance(vp_res, VarPartResult)
    assert vp_res.term_names == ["condition", "subject", "batch", "Residuals"]

    # Check median recovery in each planted regime
    assert abs(np.median(vp_res.fractions[:40, 1]) - np.median(true_fracs[:40, 1])) < 0.08
    assert abs(np.median(vp_res.fractions[40:80, 2]) - np.median(true_fracs[40:80, 2])) < 0.08
    assert abs(np.median(vp_res.fractions[80:, 0]) - np.median(true_fracs[80:, 0])) < 0.08

    # Check overall correlation between estimated and true variance fractions
    for col in range(3):
        corr = float(np.corrcoef(vp_res.fractions[:, col], true_fracs[:, col])[0, 1])
        assert corr > 0.85, f"Low correlation {corr:.3f} for term {vp_res.term_names[col]}"


def test_varpart_sum_to_one_weight_scale_and_reference_invariance() -> None:
    rng = np.random.default_rng(404)
    N, G = 60, 40
    z_latent = rng.normal(0.0, 1.0, size=N)
    cov1 = z_latent + rng.normal(0.0, 0.5, size=N)
    cov2 = -0.8 * z_latent + rng.normal(0.0, 0.5, size=N)
    tissue = np.tile(["liver", "brain", "heart"], 20)
    subj = np.repeat([f"S{i}" for i in range(15)], 4)

    meta1 = pd.DataFrame(
        {
            "cov1": cov1,
            "cov2": cov2,
            "tissue": pd.Categorical(tissue, categories=["liver", "brain", "heart"]),
            "subject": subj,
        }
    )
    meta2 = meta1.copy()
    meta2["tissue"] = pd.Categorical(tissue, categories=["heart", "liver", "brain"])

    Y = rng.normal(2.0, 1.0, size=(N, G))
    Y += (
        0.9 * cov1[:, None]
        + 0.7 * cov2[:, None]
        + (tissue == "brain")[:, None] * 1.2
        - (tissue == "heart")[:, None] * 0.6
    )
    W = rng.uniform(0.3, 4.0, size=(N, G))

    vp1 = fit_extract_var_part_model(
        Y, "~ cov1 + cov2 + tissue + (1 | subject)", data=meta1, weights=W
    )
    vp2 = fit_extract_var_part_model(
        Y, "~ cov1 + cov2 + tissue + (1 | subject)", data=meta2, weights=W
    )
    vp_scaled = fit_extract_var_part_model(
        Y, "~ cov1 + cov2 + tissue + (1 | subject)", data=meta1, weights=37.5 * W
    )

    assert_allclose(vp1.fractions.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(vp1.fractions >= 0.0)
    assert np.all(vp1.fractions <= 1.0)
    assert_allclose(vp1.fractions, vp2.fractions, atol=1e-7, rtol=1e-7)
    assert_allclose(vp1.fractions, vp_scaled.fractions, atol=1e-7, rtol=1e-7)

    # Also verify dream invariance to weight scaling
    d1 = dream(Y, "~ cov1 + cov2 + tissue + (1 | subject)", data=meta1, contrasts="cov1", weights=W)
    d2 = dream(
        Y, "~ cov1 + cov2 + tissue + (1 | subject)", data=meta1, contrasts="cov1", weights=37.5 * W
    )
    assert isinstance(d1, DreamResult) and isinstance(d2, DreamResult)
    assert_allclose(d1.t_stat, d2.t_stat, atol=1e-7, rtol=1e-7)
    assert_allclose(d1.p_value, d2.p_value, atol=1e-7, rtol=1e-7)
    assert_allclose(d1.df_satterthwaite, d2.df_satterthwaite, atol=1e-7, rtol=1e-7)


def test_satterthwaite_df_balanced_between_and_within_subject() -> None:
    rng = np.random.default_rng(505)
    m_subj = 16
    n_rep = 4
    N = m_subj * n_rep
    subjects = np.repeat([f"S{i}" for i in range(m_subj)], n_rep)
    # Between-subject treatment: 8 ctrl, 8 trt
    trt = np.repeat(["ctrl", "trt"], (m_subj // 2) * n_rep)
    # Within-subject time: 2 pre, 2 post per subject
    time = np.tile(["pre", "pre", "post", "post"], m_subj)

    meta = pd.DataFrame(
        {
            "treatment": pd.Categorical(trt, categories=["ctrl", "trt"]),
            "time": pd.Categorical(time, categories=["pre", "post"]),
            "subject": subjects,
        }
    )

    G = 20
    subj_codes = np.repeat(np.arange(m_subj), n_rep)
    u = rng.normal(0.0, np.sqrt(1.5), size=(m_subj, G))
    eps = rng.normal(0.0, np.sqrt(0.4), size=(N, G))
    Y = 4.0 + u[subj_codes] + eps

    res = dream(
        Y,
        "~ treatment + time + (1 | subject)",
        data=meta,
        contrasts=["treatment_trt", "time_post"],
        ebayes=False,
    )
    assert isinstance(res, DreamFit)

    # Exact split-plot ANOVA degrees of freedom:
    # between-subject factor -> m - 2 = 14
    # within-subject factor -> N - m - 1 = 64 - 16 - 1 = 47
    assert_allclose(res.df_satterthwaite[:, 0], 14.0, atol=0.05)
    assert_allclose(res.df_satterthwaite[:, 1], 47.0, atol=0.05)


def test_dream_reduces_to_limma_when_r0() -> None:
    rng = np.random.default_rng(606)
    N, G = 30, 80
    group = np.array(["ctrl"] * 15 + ["trt"] * 15)
    age = rng.normal(50.0, 10.0, size=N)
    meta = pd.DataFrame(
        {
            "group": pd.Categorical(group, categories=["ctrl", "trt"]),
            "age": age,
        }
    )
    W = rng.uniform(0.4, 2.5, size=(N, G))
    Y = rng.normal(5.0, 1.0, size=(N, G)) / np.sqrt(W)
    Y[:, :15] += 1.4 * (group == "trt")[:, None]

    res = dream(
        Y,
        "~ group + age",
        data=meta,
        contrasts="group_trt",
        weights=W,
        ebayes=True,
    )
    assert isinstance(res, DreamResult)

    # Independent WLS + squeeze_var reference
    X = build_design("~ group + age", meta).X
    c = np.array([0.0, 1.0, 0.0])
    df1 = float(N - X.shape[1])
    assert_allclose(res.df_satterthwaite[:, 0], df1, atol=1e-12)

    betas_ref = np.empty((G, 3))
    se2_ref = np.empty(G)
    uvar_ref = np.empty(G)
    for g in range(G):
        d = np.sqrt(W[:, g])
        Xw = d[:, None] * X
        yw = d * Y[:, g]
        xtx_inv = np.linalg.inv(Xw.T @ Xw)
        b = xtx_inv @ (Xw.T @ yw)
        r = yw - Xw @ b
        betas_ref[g] = b
        se2_ref[g] = float(r @ r) / df1
        uvar_ref[g] = float(c @ xtx_inv @ c)

    vp_ref, s20_ref, df0_ref, dftot_ref = squeeze_var(se2_ref, df1)
    t_ref = (betas_ref @ c) / np.sqrt(vp_ref * uvar_ref)

    assert_allclose(res.log_fc[:, 0], betas_ref @ c, atol=1e-11)
    assert_allclose(res.var_post[:, 0], vp_ref, atol=1e-11)
    assert_allclose(res.t_stat[:, 0], t_ref, atol=1e-11)
    assert_allclose(float(res.var_prior[0]), float(s20_ref), atol=1e-11)
    assert_allclose(float(res.df_prior[0]), float(df0_ref), atol=1e-11)


def test_dream_ebayes_moderation_de_recovery_and_robust() -> None:
    rng = np.random.default_rng(707)
    m_subj, n_rep, G = 12, 3, 300
    N = m_subj * n_rep
    subjects = np.repeat([f"S{i}" for i in range(m_subj)], n_rep)
    # Within-subject condition
    cond = np.tile(["ctrl", "trt", "trt"], m_subj)
    meta = pd.DataFrame(
        {
            "condition": pd.Categorical(cond, categories=["ctrl", "trt"]),
            "subject": subjects,
        }
    )
    # Sample gene variances from scaled inverse chi-squared (d0=6, s0^2=0.4)
    d0_true, s02_true = 6.0, 0.4
    chi2_draws = rng.chisquare(df=d0_true, size=G)
    sigma_e2_true = (d0_true * s02_true) / chi2_draws

    subj_codes = np.repeat(np.arange(m_subj), n_rep)
    u = rng.normal(0.0, 1.0, size=(m_subj, G))
    eps = rng.normal(0.0, 1.0, size=(N, G)) * np.sqrt(sigma_e2_true)[None, :]
    Y = 6.0 + u[subj_codes] + eps
    de_mask = np.zeros(G, dtype=bool)
    de_mask[:40] = True
    Y[:, de_mask] += 1.5 * (cond == "trt")[:, None]

    res = dream(
        Y,
        "~ condition + (1 | subject)",
        data=meta,
        contrasts="condition_trt",
        ebayes=True,
    )
    assert isinstance(res, DreamResult)

    # Empirical Bayes shrinkage & prior recovery
    df_prior_val = float(res.df_prior[0])
    assert 2.0 <= df_prior_val <= 20.0
    s20_val = float(res.var_prior[0])
    # Shrinkage moves extreme sample variances toward s0^2
    dist_raw = np.mean(np.abs(res.sigma_e_sq - s20_val))
    dist_post = np.mean(np.abs(res.var_post[:, 0] - s20_val))
    assert dist_post < dist_raw

    # High DE recovery at FDR < 0.05
    called = res.adj_p_val[:, 0] < 0.05
    recall = float(np.mean(called[de_mask]))
    fdp = float(np.sum(called & ~de_mask) / max(1, np.sum(called)))
    assert recall > 0.85
    assert fdp < 0.15

    # Inject 5 extreme variance outliers (100x residual SD) and test robust=True
    Y_out = Y.copy()
    Y_out[:, -5:] += rng.normal(0.0, 10.0, size=(N, 5))
    fit_out = dream(
        Y_out,
        "~ condition + (1 | subject)",
        data=meta,
        contrasts="condition_trt",
        ebayes=False,
    )
    res_nonrobust = e_bayes(fit_out, robust=False)
    res_robust = e_bayes(fit_out, robust=True)

    # Non-robust df_prior collapses sharply; robust keeps inlier df_prior close to original
    assert float(res_nonrobust.df_prior[0]) < 0.6 * df_prior_val
    inlier_df_prior = np.asarray(res_robust.df_prior[0])[:-5]
    outlier_df_prior = np.asarray(res_robust.df_prior[0])[-5:]
    assert float(np.median(inlier_df_prior)) > 0.7 * df_prior_val
    assert float(np.max(outlier_df_prior)) < float(np.median(inlier_df_prior))

    # Also test trend=True and top_table API
    res_trend = e_bayes(fit_out, trend=True, robust=True)
    tt = top_table(res_trend, coef="condition_trt", number=10, sort_by="P.Value")
    assert tt.shape == (10, 9)
    assert list(tt.columns) == [
        "logFC",
        "AveExpr",
        "t",
        "P.Value",
        "adj.P.Val",
        "z.std",
        "df.Satt",
        "df.total",
        "s2.post",
    ]
    assert np.all(np.isfinite(res_trend.z_std[:, 0]))


def test_null_calibration_with_repeated_measures() -> None:
    rng = np.random.default_rng(808)
    m_subj, n_rep, G = 12, 6, 500
    N = m_subj * n_rep
    subjects = np.repeat([f"S{i}" for i in range(m_subj)], n_rep)
    # Between-subject treatment under H0
    trt = np.repeat(["ctrl", "trt"], (m_subj // 2) * n_rep)
    meta = pd.DataFrame(
        {
            "treatment": pd.Categorical(trt, categories=["ctrl", "trt"]),
            "subject": subjects,
        }
    )

    subj_codes = np.repeat(np.arange(m_subj), n_rep)
    u = rng.normal(0.0, np.sqrt(2.0), size=(m_subj, G))
    eps = rng.normal(0.0, np.sqrt(0.5), size=(N, G))
    Y = 5.0 + u[subj_codes] + eps

    res_dream = dream(
        Y, "~ treatment + (1 | subject)", data=meta, contrasts="treatment_trt", ebayes=True
    )
    assert isinstance(res_dream, DreamResult)
    p_dream = res_dream.p_value[:, 0]
    type1_dream = float(np.mean(p_dream < 0.05))
    ks_res = ks_1samp(p_dream, uniform.cdf)

    assert 0.02 <= type1_dream <= 0.075, f"dream Type I error out of range: {type1_dream:.4f}"
    assert ks_res.pvalue > 0.01, f"dream null p-values non-uniform: KS p={ks_res.pvalue:.4g}"

    # Naive OLS ignoring subject random effect severely inflates Type I error
    res_ols = dream(Y, "~ treatment", data=meta, contrasts="treatment_trt", ebayes=True)
    type1_ols = float(np.mean(res_ols.p_value[:, 0] < 0.05))
    assert type1_ols > 0.30, f"Expected inflated OLS Type I error, got {type1_ols:.4f}"


def test_boundary_zero_re_collinear_and_constant_gene() -> None:
    rng = np.random.default_rng(909)
    N, G = 40, 25
    subjects = np.repeat([f"S{i}" for i in range(10)], 4)
    trt = np.tile(["ctrl", "trt"], 20)
    meta = pd.DataFrame(
        {
            "trt": pd.Categorical(trt, categories=["ctrl", "trt"]),
            "subject": subjects,
            "subject_copy": subjects,
        }
    )

    # Pure white noise (no subject effect -> many genes hit sigma_r^2 = 0 boundary)
    Y = rng.normal(0.0, 1.0, size=(N, G))
    # Gene 0: constant flat gene
    Y[:, 0] = 5.0

    # Duplicate/collinear random effects: (1 | subject) + (1 | subject_copy)
    vp_dup = fit_extract_var_part_model(
        Y, "~ trt + (1 | subject) + (1 | subject_copy)", data=meta
    )
    assert np.any(vp_dup.singular)
    # Constant gene row sums to 1.0 with Residuals = 1.0
    assert_allclose(vp_dup.fractions[0], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    assert_allclose(vp_dup.fractions.sum(axis=1), 1.0, atol=1e-12)

    d_dup = dream(
        Y, "~ trt + (1 | subject) + (1 | subject_copy)", data=meta, contrasts="trt_trt"
    )
    assert isinstance(d_dup, DreamResult)
    assert np.all(np.isfinite(d_dup.df_satterthwaite[1:, 0]))
    assert np.all(d_dup.df_satterthwaite[1:, 0] >= 1.0)
    assert np.all(d_dup.df_satterthwaite[1:, 0] <= N - 2)

    # Collinear fixed + random factor: (1 | subject) + subject
    d_col_fr = dream(Y, "~ trt + subject + (1 | subject)", data=meta, contrasts="trt_trt")
    assert np.all(np.isfinite(d_col_fr.p_value[1:, 0]))

    # Rank-deficient fixed design X: estimable vs non-estimable contrast
    X_full = build_design("~ trt", meta).X
    X_rank_def = np.column_stack([X_full, X_full[:, 0] + 2.0 * X_full[:, 1]])
    spec_rd = DesignSpec.from_matrices(
        X_rank_def,
        random_effects={"subject": meta["subject"]},
        fixed_names=["Intercept", "trt_trt", "lin_combo"],
    )
    d_rd = dream(
        Y,
        spec_rd,
        contrasts={"est_c": [0.0, 1.0, 2.0], "nonest_c": [0.0, 1.0, 0.0]},
    )
    assert isinstance(d_rd, DreamResult)
    assert np.all(np.isfinite(d_rd.log_fc[:, 0]))
    assert np.all(np.isfinite(d_rd.p_value[1:, 0]))
    assert np.all(np.isnan(d_rd.log_fc[:, 1]))
    assert np.all(np.isnan(d_rd.p_value[:, 1]))


def test_anndata_sparse_genes_as_rows_and_nan_bh_hygiene() -> None:
    rng = np.random.default_rng(1010)
    N, G = 40, 60
    subjects = np.repeat([f"S{i}" for i in range(10)], 4)
    cond = np.tile(["ctrl", "trt"], 20)
    obs = pd.DataFrame(
        {
            "cond": pd.Categorical(cond, categories=["ctrl", "trt"]),
            "subject": subjects,
        },
        index=[f"cell_{i}" for i in range(N)],
    )
    var = pd.DataFrame(index=[f"gene_{g}" for g in range(G)])

    subj_codes = np.repeat(np.arange(10), 4)
    Y = 3.0 + rng.normal(0.0, 0.8, size=(10, G))[subj_codes] + rng.normal(0.0, 0.5, size=(N, G))
    Y[:, :10] += 1.5 * (cond == "trt")[:, None]
    W = rng.uniform(0.5, 2.5, size=(N, G))

    # Inject 10 all-NaN genes and 5 partial-NaN genes (including 1 non-estimable on trt)
    Y_nan = Y.copy()
    Y_nan[:, -10:] = np.nan
    Y_nan[:4, -15:-11] = np.nan
    # Make gene -11 missing all 'trt' samples -> contrast 'cond_trt' non-estimable
    Y_nan[cond == "trt", -11] = np.nan

    adata = ad.AnnData(
        X=sparse.csr_matrix(Y_nan),
        obs=obs,
        var=var,
        layers={"voom_weights": W},
    )

    res_adata = dream(
        adata,
        "~ cond + (1 | subject)",
        contrasts="cond_trt",
        weights="voom_weights",
    )
    df_ng = pd.DataFrame(Y_nan, index=obs.index, columns=var.index)
    res_df = dream(
        df_ng,
        "~ cond + (1 | subject)",
        data=obs,
        contrasts="cond_trt",
        weights=W,
    )
    res_gar = dream(
        Y_nan.T,
        "~ cond + (1 | subject)",
        data=obs,
        contrasts="cond_trt",
        weights=W.T,
        genes_as_rows=True,
        gene_names=var.index.tolist(),
    )

    assert_allclose(res_adata.log_fc, res_df.log_fc, atol=1e-11, equal_nan=True)
    assert_allclose(res_adata.p_value, res_df.p_value, atol=1e-11, equal_nan=True)
    assert_allclose(res_adata.adj_p_val, res_df.adj_p_val, atol=1e-11, equal_nan=True)
    assert_allclose(res_adata.log_fc, res_gar.log_fc, atol=1e-11, equal_nan=True)

    # Check NaN preservation and zero BH m-inflation
    p_col = res_adata.p_value[:, 0]
    adj_col = res_adata.adj_p_val[:, 0]
    assert np.all(np.isnan(p_col[-11:]))
    assert np.all(np.isnan(adj_col[-11:]))
    valid_mask = np.isfinite(p_col)
    expected_bh = false_discovery_control(p_col[valid_mask], method="bh")
    assert_allclose(adj_col[valid_mask], expected_bh, atol=1e-12)

    # Check shape mismatch hint when genes_as_rows=True is forgotten
    with pytest.raises(ValueError, match="genes_as_rows=True"):
        fit_lmm(Y_nan.T, "~ cond + (1 | subject)", data=obs)

    # Check non-positive weights raise ValueError
    W_bad = W.copy()
    W_bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lmm(Y, "~ cond + (1 | subject)", data=obs, weights=W_bad)


def test_ci_isolation_and_dataclass_contracts() -> None:
    rng = np.random.default_rng(1111)
    N, G = 24, 12
    meta = pd.DataFrame(
        {
            "cond": np.tile(["ctrl", "trt"], 12),
            "subj": np.repeat([f"s{i}" for i in range(6)], 4),
        }
    )
    Y = rng.normal(0.0, 1.0, size=(N, G))

    # Block patsy, statsmodels, joblib, numba to prove pure NumPy/SciPy/Pandas operation
    blocked = {
        "patsy": None,
        "statsmodels": None,
        "statsmodels.api": None,
        "joblib": None,
        "numba": None,
    }
    with mock.patch.dict(sys.modules, blocked):
        spec = build_design("~ cond + (1 | subj)", meta)
        lmm_res = fit_lmm(Y, spec)
        vp_res = extract_var_part(lmm_res)
        dream_fit = dream(Y, spec, contrasts="cond_trt", ebayes=False)
        dream_res = e_bayes(dream_fit)

    # Frozen dataclass checks & .to_dataframe() / .summary() contracts
    for obj in (spec, lmm_res, vp_res, dream_fit, dream_res):
        assert dataclasses.is_dataclass(obj)
        df_out = obj.to_dataframe()
        assert isinstance(df_out, pd.DataFrame)
        summ = obj.summary()
        assert isinstance(summ, (str, pd.DataFrame))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, "gene_names", [])  # type: ignore[misc]

    # Verify trigamma_inverse and df0=inf limit
    vals = np.array([0.01, 0.2, 2.5, 50.0, np.inf, -1.0, np.nan])
    inv = trigamma_inverse(vals)
    from scipy.special import polygamma

    assert_allclose(polygamma(1, inv[:4]), vals[:4], rtol=1e-10, atol=1e-12)
    assert inv[4] == 0.0
    assert np.isnan(inv[5]) and np.isnan(inv[6])

    # Homogeneous variances -> df_prior == inf and 100% shrinkage to s20
    s20_inf, df0_inf = fit_f_dist(np.full(50, 0.5), df1=10.0)
    expected_s20_inf = 0.5 * float(np.exp(-digamma(5.0) + np.log(5.0)))
    assert np.isinf(df0_inf)
    assert_allclose(s20_inf, expected_s20_inf, atol=1e-12)
    vp_inf, s20_sq, df0_sq, dftot_inf = squeeze_var(np.full(50, 0.5), df1=10.0)
    assert np.isinf(df0_sq) and np.isinf(dftot_inf)
    assert_allclose(vp_inf, s20_sq, atol=1e-12)

    # Nonlinear contrast must raise ValueError
    with pytest.raises(ValueError):
        make_contrasts({"bad": "Intercept * cond_trt"}, spec.fixed_names)


def test_constant_genes_invariance_and_missing_layer_and_weighted_fixed_var() -> None:
    rng = np.random.default_rng(1212)
    N, m_subj, m_batch = 60, 15, 4
    subj = np.repeat([f"S{i}" for i in range(m_subj)], N // m_subj)
    batch = np.array([f"B{(i + j) % m_batch}" for i in range(m_subj) for j in range(N // m_subj)])
    cond = np.array([("trt" if (i % 2 == 1) else "ctrl") for i in range(m_subj) for _ in range(N // m_subj)])
    meta = pd.DataFrame(
        {
            "cond": pd.Categorical(cond, categories=["ctrl", "trt"]),
            "subject": subj,
            "batch": batch,
        }
    )

    G_real = 40
    Y_base = rng.normal(5.0, 1.0, size=(N, G_real))
    W_base = rng.uniform(0.5, 2.5, size=(N, G_real))
    real_idx = [i for i in range(G_real + 1) if i != 2]

    for formula in ("~ cond", "~ cond + (1 | subject)", "~ cond + (1 | subject) + (1 | batch)"):
        for robust in (False, True):
            dr_ref = dream(Y_base, formula, data=meta, contrasts="cond_trt", robust=robust)
            assert isinstance(dr_ref, DreamResult)
            for const_val in (4.2, 0.0, 1e-5):
                Y_aug = np.column_stack([Y_base[:, :2], np.full(N, const_val), Y_base[:, 2:]])
                dr_aug = dream(Y_aug, formula, data=meta, contrasts="cond_trt", robust=robust)
                assert isinstance(dr_aug, DreamResult)
                assert dr_aug.sigma_e_sq[2] == 0.0
                assert np.isnan(dr_aug.t_stat[2, 0])
                assert np.isnan(dr_aug.p_value[2, 0])
                assert np.isnan(dr_aug.adj_p_val[2, 0])
                assert_allclose(dr_aug.var_prior[0], dr_ref.var_prior[0], atol=1e-12, rtol=1e-12)
                dp_aug = dr_aug.df_prior[0][real_idx] if robust else dr_aug.df_prior[0]
                assert_allclose(dp_aug, dr_ref.df_prior[0], atol=1e-12, rtol=1e-12, equal_nan=True)
                assert_allclose(dr_aug.t_stat[real_idx, 0], dr_ref.t_stat[:, 0], atol=1e-12)
                assert_allclose(dr_aug.p_value[real_idx, 0], dr_ref.p_value[:, 0], atol=1e-12)
                assert_allclose(dr_aug.adj_p_val[real_idx, 0], dr_ref.adj_p_val[:, 0], atol=1e-12)

            # Also verify unmoderated DreamFit.to_dataframe() and 2D-weighted zero-residual gene
            Y_const = np.column_stack([Y_base[:, :2], np.full(N, 4.2), Y_base[:, 2:]])
            W_const = np.column_stack([W_base[:, :2], np.ones(N), W_base[:, 2:]])
            df_unmod = dream(
                Y_const, formula, data=meta, contrasts="cond_trt", weights=W_const, ebayes=False
            ).to_dataframe(sort_by_pvalue=False)
            assert np.isnan(df_unmod.iloc[2]["t"])
            assert np.isnan(df_unmod.iloc[2]["P.Value"])
            assert np.isnan(df_unmod.iloc[2]["adj.P.Val"])

    # Missing AnnData layer must raise ValueError (not KeyError)
    adata = ad.AnnData(X=Y_base, obs=meta)
    with pytest.raises(ValueError, match="missing_layer"):
        fit_lmm(adata, "~ cond + (1 | subject)", layer="missing_layer")

    # weighted_fixed_var=True with partial NaNs in Y and matching NaNs in 2D weights
    Y_pnan = Y_base[:, :12].copy()
    W_pnan = W_base[:, :12].copy()
    Y_pnan[:3, 0] = np.nan
    W_pnan[:3, 0] = np.nan
    vp_wfv = fit_extract_var_part_model(
        Y_pnan,
        "~ cond + (1 | subject)",
        data=meta,
        weights=W_pnan,
        weighted_fixed_var=True,
    )
    assert np.all(np.isfinite(vp_wfv.fractions))
    assert_allclose(vp_wfv.fractions.sum(axis=1), 1.0, atol=1e-12)


def test_dream_ebayes_between_subject_null_calibration() -> None:
    rng = np.random.default_rng(1313)
    m_subj, n_rep, G = 8, 6, 800
    N = m_subj * n_rep
    subjects = np.repeat([f"D{i}" for i in range(m_subj)], n_rep)
    trt = np.repeat(["ctrl", "trt"], (m_subj // 2) * n_rep)
    meta = pd.DataFrame(
        {
            "trt": pd.Categorical(trt, categories=["ctrl", "trt"]),
            "donor": subjects,
        }
    )
    subj_codes = np.repeat(np.arange(m_subj), n_rep)

    for d0 in (None, 4.0, 8.0):
        if d0 is None:
            se2 = np.full(G, 0.5)
            su2 = np.full(G, 2.0)
        else:
            se2 = (d0 * 0.5) / rng.chisquare(d0, size=G)
            su2 = (d0 * 2.0) / rng.chisquare(d0, size=G)
        Y = (
            5.0
            + rng.normal(0.0, 1.0, size=(m_subj, G))[subj_codes] * np.sqrt(su2)[None, :]
            + rng.normal(0.0, 1.0, size=(N, G)) * np.sqrt(se2)[None, :]
        )
        res = dream(Y, "~ trt + (1 | donor)", data=meta, contrasts="trt_trt", ebayes=True)
        assert isinstance(res, DreamResult)
        p = res.p_value[:, 0]
        fpr05 = float(np.mean(p <= 0.05))
        fpr01 = float(np.mean(p <= 0.01))
        ks_p = float(ks_1samp(p, uniform.cdf).pvalue)
        assert fpr05 <= 0.065, f"Between-subject ebayes FPR@0.05 inflated ({fpr05:.4f}) for d0={d0}"
        assert fpr01 <= 0.02, f"Between-subject ebayes FPR@0.01 inflated ({fpr01:.4f}) for d0={d0}"
        assert ks_p > 0.05, f"Between-subject ebayes null p-values non-uniform (KS p={ks_p:.4g}) for d0={d0}"

        # Idempotency of e_bayes(DreamResult)
        res2 = e_bayes(res)
        assert_allclose(res2.p_value, res.p_value, atol=1e-13, equal_nan=True)


def test_extreme_scale_invariance_and_constant_genes() -> None:
    rng = np.random.default_rng(1414)
    m_subj, n_rep, G = 12, 4, 24
    N = m_subj * n_rep
    meta = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], (m_subj // 2) * n_rep), categories=["ctrl", "trt"]),
            "donor": np.repeat([f"S{i}" for i in range(m_subj)], n_rep),
            "batch": np.tile(["B1", "B2", "B3", "B4"], m_subj),
        }
    )
    subj_codes = np.repeat(np.arange(m_subj), n_rep)
    batch_codes = np.tile(np.arange(4), m_subj)
    Y_base = (
        5.0
        + rng.uniform(-1.5, 1.5, size=(1, G))
        + rng.normal(0.0, 0.8, size=(m_subj, G))[subj_codes]
        + rng.normal(0.0, 0.4, size=(4, G))[batch_codes]
        + rng.normal(0.0, 0.5, size=(N, G))
    )
    Y_zm = Y_base - np.mean(Y_base, axis=0, keepdims=True)
    # Genes 0 and 1: constant genes (4.2 and 0.0)
    Y_base[:, 0] = 4.2
    Y_base[:, 1] = 0.0
    # Partial NaNs on gene 4
    Y_pnan = Y_base.copy()
    Y_pnan[:3, 4] = np.nan
    W_2d = rng.uniform(0.4, 3.0, size=(N, G))

    for label, Y_in, use_trend in (("nonzero", Y_base, True), ("partial_nan", Y_pnan, False), ("zero_mean", Y_zm, False)):
        for formula in ("~ trt", "~ trt + (1 | donor)", "~ trt + (1 | donor) + (1 | batch)"):
            for use_w in (False, True):
                if label == "partial_nan" and formula != "~ trt + (1 | donor)":
                    continue
                kw = {"weights": W_2d} if use_w else {}
                dr_ref = dream(Y_in, formula, data=meta, contrasts="trt_trt", trend=use_trend, robust=True, **kw)
                vp_ref = fit_extract_var_part_model(Y_in, formula, data=meta, weighted_fixed_var=use_w, **kw)
                assert isinstance(dr_ref, DreamResult)
                for scale in (1e-15, 1e-12, 1e+12):
                    dr_s = dream(Y_in * scale, formula, data=meta, contrasts="trt_trt", trend=use_trend, robust=True, **kw)
                    vp_s = fit_extract_var_part_model(Y_in * scale, formula, data=meta, weighted_fixed_var=use_w, **kw)
                    assert isinstance(dr_s, DreamResult)
                    if label != "zero_mean":
                        assert dr_s.sigma_e_sq[0] == 0.0 and dr_s.sigma_e_sq[1] == 0.0
                        assert np.isnan(dr_s.p_value[0, 0]) and np.isnan(dr_s.p_value[1, 0])
                        assert np.all(np.isfinite(dr_s.p_value[2:, 0]))
                        assert vp_s.total_variance[0] == 0.0 and vp_s.total_variance[1] == 0.0
                        assert np.all(vp_s.total_variance[2:] > 0.0)
                    else:
                        assert np.all(np.isfinite(dr_s.p_value[:, 0]))
                        assert np.all(vp_s.total_variance > 0.0)
                    assert_allclose(dr_s.p_value, dr_ref.p_value, atol=1e-7, rtol=1e-7, equal_nan=True)
                    assert_allclose(vp_s.fractions, vp_ref.fractions, atol=1e-7, rtol=1e-7, equal_nan=True)


def test_voom_2d_weights_cov_beta_fixed_and_r0_r1_continuity() -> None:
    rng = np.random.default_rng(1515)
    m_subj, n_rep, G = 12, 2, 300
    N = m_subj * n_rep
    meta = pd.DataFrame(
        {
            "time": pd.Categorical(np.tile(["t0", "t1"], m_subj), categories=["t0", "t1"]),
            "donor": np.repeat([f"S{i}" for i in range(m_subj)], n_rep),
        }
    )

    # Heterogeneous 1000x gene-scale voom weights under pure white noise (sigma_r^2 == 0)
    v_true = np.exp(rng.uniform(np.log(0.001), np.log(1.0), size=G))
    W_voom = np.broadcast_to((1.0 / v_true)[None, :], (N, G)).copy() * rng.uniform(0.8, 1.2, size=(N, G))
    Y_noise = 5.0 + rng.normal(0.0, 1.0, size=(N, G)) / np.sqrt(W_voom)

    for robust in (False, True):
        r0 = dream(Y_noise, "~ time", data=meta, weights=W_voom, contrasts="time_t1", ebayes=True, robust=robust)
        r1 = dream(Y_noise, "~ time + (1 | donor)", data=meta, weights=W_voom, contrasts="time_t1", ebayes=True, robust=robust)
        assert isinstance(r0, DreamResult) and isinstance(r1, DreamResult)
        df0_r1 = float(np.median(r1.df_prior[0]))
        assert df0_r1 > 20.0, f"df_prior collapsed under 2D voom weights (robust={robust}): {df0_r1}"

    # Verify exact cov_beta_fixed == cov_beta_unscaled on boundary singular fits (sigma_r^2 == 0)
    # and exact R=0 vs R=1 e_bayes parity when df_satterthwaite is aligned
    fit1 = dream(Y_noise, "~ time + (1 | donor)", data=meta, weights=W_voom, contrasts="time_t1", ebayes=False)
    assert isinstance(fit1, DreamFit)
    sing = fit1.lmm_fit.singular
    assert np.any(sing)
    assert fit1.lmm_fit.cov_beta_fixed is not None
    assert_allclose(
        fit1.lmm_fit.cov_beta_fixed[sing],
        fit1.lmm_fit.cov_beta_unscaled[sing],
        atol=1e-13,
        rtol=1e-13,
    )

    # When all genes hit sigma_r^2 == 0 (negative within-donor correlation),
    # v_raw == sigma_e_sq and cov_beta_fixed == cov_beta_unscaled == R=0 unscaled_var
    z_pair = rng.normal(0.0, 1.0, size=(m_subj, G))
    Y_neg = np.empty((N, G), dtype=np.float64)
    Y_neg[0::2] = 5.0 + (z_pair + rng.normal(0.0, 0.2, size=(m_subj, G))) / np.sqrt(W_voom[0::2])
    Y_neg[1::2] = 5.0 + (-z_pair + rng.normal(0.0, 0.2, size=(m_subj, G))) / np.sqrt(W_voom[1::2])
    fit0_neg = dream(Y_neg, "~ time", data=meta, weights=W_voom, contrasts="time_t1", ebayes=False)
    fit1_neg = dream(Y_neg, "~ time + (1 | donor)", data=meta, weights=W_voom, contrasts="time_t1", ebayes=False)
    assert isinstance(fit0_neg, DreamFit) and isinstance(fit1_neg, DreamFit)
    assert np.all(fit1_neg.lmm_fit.singular)
    assert_allclose(fit1_neg.sigma_e_sq, fit0_neg.sigma_e_sq, atol=1e-12, rtol=1e-12)
    assert_allclose(fit1_neg.unscaled_var, fit0_neg.unscaled_var, atol=1e-12, rtol=1e-12)
    assert fit1_neg.lmm_fit.cov_beta_fixed is not None
    assert_allclose(fit1_neg.lmm_fit.cov_beta_fixed, fit0_neg.lmm_fit.cov_beta_unscaled, atol=1e-12, rtol=1e-12)
    fit1_neg_df0 = dataclasses.replace(fit1_neg, df_satterthwaite=fit0_neg.df_satterthwaite)
    eb0_neg = e_bayes(fit0_neg, robust=True)
    eb1_neg = e_bayes(fit1_neg_df0, robust=True)
    assert_allclose(eb1_neg.p_value, eb0_neg.p_value, atol=1e-12, rtol=1e-12)
    assert_allclose(eb1_neg.var_post, eb0_neg.var_post, atol=1e-12, rtol=1e-12)

    # Repeated measures (sigma_r^2 > 0) with 1000x heterogeneous W_voom equalizing true variance
    m8, nr8, G8 = 8, 4, 400
    N8 = m8 * nr8
    meta8 = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], (m8 // 2) * nr8), categories=["ctrl", "trt"]),
            "donor": np.repeat([f"D{i}" for i in range(m8)], nr8),
        }
    )
    sc8 = np.repeat(np.arange(m8), nr8)
    v8 = np.exp(rng.uniform(np.log(0.001), np.log(1.0), size=G8))
    W8 = np.broadcast_to((1.0 / v8)[None, :], (N8, G8)).copy() * rng.uniform(0.8, 1.2, size=(N8, G8))
    Y8 = (
        5.0
        + rng.normal(0.0, 1.0, size=(m8, G8))[sc8] * np.sqrt(2.0 * v8)[None, :]
        + rng.normal(0.0, 1.0, size=(N8, G8)) / np.sqrt(W8)
    )
    r8 = dream(Y8, "~ trt + (1 | donor)", data=meta8, weights=W8, contrasts="trt_trt", ebayes=True)
    assert isinstance(r8, DreamResult)
    assert float(r8.df_prior[0]) > 20.0
    p8 = r8.p_value[:, 0]
    assert 0.02 <= float(np.mean(p8 < 0.05)) <= 0.08
    assert float(ks_1samp(p8, uniform.cdf).pvalue) > 0.01


def test_weight_scale_satterthwaite_invariance_and_high_icc() -> None:
    rng = np.random.default_rng(1616)
    m_subj, n_rep, G = 12, 6, 30
    N = m_subj * n_rep
    meta = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], N // 2), categories=["ctrl", "trt"]),
            "time": pd.Categorical(np.tile(["t0", "t1"], N // 2), categories=["t0", "t1"]),
            "donor": pd.Categorical(np.repeat([f"S{i}" for i in range(m_subj)], n_rep)),
            "batch": np.tile([f"B{b}" for b in range(3)], N // 3),
        }
    )
    d_idx = np.repeat(np.arange(m_subj), n_rep)
    b_idx = np.tile(np.arange(3), N // 3)
    w_base = np.exp(rng.uniform(0.0, np.log(1000.0), size=G))
    W = w_base[None, :] * rng.uniform(0.7, 1.3, size=(N, G))
    s2 = 8.0 / rng.chisquare(8.0, size=G)
    Y1 = (
        5.0
        + rng.normal(0.0, 1.0, size=(m_subj, G))[d_idx] * np.sqrt(0.67 * s2 / w_base)[None, :]
        + rng.normal(0.0, 1.0, size=(N, G)) * np.sqrt(s2)[None, :] / np.sqrt(W)
    )
    Y2 = Y1 + rng.normal(0.0, 1.0, size=(3, G))[b_idx] * np.sqrt(0.30 * s2 / w_base)[None, :]

    # 1. Global W -> c * W invariance across R=1 and R=2 (between & within contrasts)
    ref_r1 = dream(
        Y1, "~ trt + time + (1 | donor)", data=meta, weights=W, contrasts=["trt_trt", "time_t1"], ebayes=True
    )
    ref_r2 = dream(
        Y2,
        "~ trt + time + (1 | donor) + (1 | batch)",
        data=meta,
        weights=W,
        contrasts=["trt_trt", "time_t1"],
        ebayes=True,
    )
    assert isinstance(ref_r1, DreamResult) and isinstance(ref_r2, DreamResult)
    for c in (1e-8, 1e-6, 1e+6, 1e+8):
        res_r1 = dream(
            Y1, "~ trt + time + (1 | donor)", data=meta, weights=W * c, contrasts=["trt_trt", "time_t1"], ebayes=True
        )
        res_r2 = dream(
            Y2,
            "~ trt + time + (1 | donor) + (1 | batch)",
            data=meta,
            weights=W * c,
            contrasts=["trt_trt", "time_t1"],
            ebayes=True,
        )
        assert isinstance(res_r1, DreamResult) and isinstance(res_r2, DreamResult)
        assert_allclose(res_r1.df_satterthwaite, ref_r1.df_satterthwaite, atol=1e-7, rtol=1e-7)
        assert_allclose(res_r1.p_value, ref_r1.p_value, atol=1e-7, rtol=1e-7)
        assert_allclose(np.asarray(res_r1.df_prior), np.asarray(ref_r1.df_prior), atol=1e-6, rtol=1e-6)
        assert_allclose(res_r2.df_satterthwaite, ref_r2.df_satterthwaite, atol=1e-7, rtol=1e-7)
        assert_allclose(res_r2.p_value, ref_r2.p_value, atol=1e-7, rtol=1e-7)
        assert_allclose(np.asarray(res_r2.df_prior), np.asarray(ref_r2.df_prior), atol=1e-6, rtol=1e-6)

    # 2. Unnormalized raw inverse-variance voom weights (v = 1e6 -> W = 1e-6)
    m10, nr10 = 10, 4
    N10 = m10 * nr10
    meta10 = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], (m10 // 2) * nr10), categories=["ctrl", "trt"]),
            "donor": np.repeat([f"D{i}" for i in range(m10)], nr10),
        }
    )
    sc10 = np.repeat(np.arange(m10), nr10)
    v_mix = np.array([1.0] * 10 + [1e6] * 10, dtype=np.float64)
    W_mix = np.broadcast_to((1.0 / v_mix)[None, :], (N10, 20)).copy()
    Y_mix = (
        5.0
        + rng.normal(0.0, 1.0, size=(m10, 20))[sc10] * np.sqrt(2.0 * v_mix)[None, :]
        + rng.normal(0.0, 1.0, size=(N10, 20)) / np.sqrt(W_mix)
    )
    res_mix = dream(Y_mix, "~ trt + (1 | donor)", data=meta10, weights=W_mix, contrasts="trt_trt", ebayes=False)
    assert isinstance(res_mix, DreamFit)
    assert_allclose(res_mix.df_satterthwaite[:, 0], 8.0, atol=0.05)

    # 3. Unweighted high-ICC genes (sigma_r^2 = 1e5 * sigma_e^2)
    Y_hi = 5.0 + rng.normal(0.0, np.sqrt(1e5), size=(m10, 12))[sc10] + rng.normal(0.0, 1.0, size=(N10, 12))
    res_hi = dream(Y_hi, "~ trt + (1 | donor)", data=meta10, contrasts="trt_trt", ebayes=False)
    assert isinstance(res_hi, DreamFit)
    assert_allclose(res_hi.df_satterthwaite[:, 0], 8.0, atol=0.05)

    # 4. Null FPR@0.01 calibration & exact normalized parity on raw small weights (w ~ [1e-6, 3e-5])
    m8, nr8, G8 = 8, 6, 400
    N8 = m8 * nr8
    meta8 = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], N8 // 2), categories=["ctrl", "trt"]),
            "donor": np.repeat([f"S{i}" for i in range(m8)], nr8),
        }
    )
    d8 = np.repeat(np.arange(m8), nr8)
    wb8 = np.exp(rng.uniform(np.log(1e-6), np.log(3e-5), size=G8))
    W8 = wb8[None, :] * rng.uniform(0.7, 1.3, size=(N8, G8))
    s2_8 = 8.0 / rng.chisquare(8.0, size=G8)
    Y8 = (
        5.0
        + rng.normal(0.0, 1.0, size=(m8, G8))[d8] * np.sqrt(0.67 * s2_8 / wb8)[None, :]
        + rng.normal(0.0, 1.0, size=(N8, G8)) * np.sqrt(s2_8)[None, :] / np.sqrt(W8)
    )
    raw8 = dream(Y8, "~ trt + (1 | donor)", data=meta8, weights=W8, contrasts="trt_trt", ebayes=True)
    norm8 = dream(Y8, "~ trt + (1 | donor)", data=meta8, weights=W8 / np.mean(W8), contrasts="trt_trt", ebayes=True)
    assert isinstance(raw8, DreamResult) and isinstance(norm8, DreamResult)
    assert_allclose(raw8.df_satterthwaite, norm8.df_satterthwaite, atol=1e-7, rtol=1e-7)
    assert_allclose(raw8.p_value, norm8.p_value, atol=1e-7, rtol=1e-7)
    assert float(np.mean(raw8.p_value[:, 0] <= 0.01)) <= 0.02
    assert float(np.mean(raw8.p_value[:, 0] <= 0.05)) <= 0.075


def test_random_effect_matrix_rescaling_invariance() -> None:
    rng = np.random.default_rng(1717)
    m1, m2, nr, G = 10, 4, 4, 16
    N = m1 * nr
    d_idx = np.repeat(np.arange(m1), nr)
    b_idx = np.tile(np.arange(m2), m1)
    trt = np.repeat(["ctrl", "trt"], N // 2)
    X = np.column_stack([np.ones(N), (trt == "trt").astype(float)])
    Z1 = np.eye(m1)[d_idx]
    Z2 = np.eye(m2)[b_idx]
    Y = (
        4.0
        + 0.5 * X[:, 1:2]
        + Z1 @ rng.normal(0.0, 1.0, size=(m1, G))
        + Z2 @ rng.normal(0.0, 0.7, size=(m2, G))
        + rng.normal(0.0, 0.4, size=(N, G))
    )
    Y[:3, 0] = np.nan
    Y[:, 1] = 3.5
    W = rng.uniform(0.3, 3.0, size=(N, G))

    # R=1 spectral + 2D-weight/partial-NaN paths across Z -> c_z * Z
    vp_r1_ref = fit_extract_var_part_model(Y, design=X, random_effects=[Z1])
    dr_r1_ref = dream(Y, design=X, random_effects=[Z1], contrasts={"trt": [0.0, 1.0]}, ebayes=True)
    dr_r1_2d_ref = dream(Y, design=X, random_effects=[Z1], weights=W, contrasts={"trt": [0.0, 1.0]}, ebayes=True)
    assert isinstance(dr_r1_ref, DreamResult) and isinstance(dr_r1_2d_ref, DreamResult)

    for cz in (1e-12, 1e-6, 1e+6, 1e+12):
        vp_cz = fit_extract_var_part_model(Y, design=X, random_effects=[Z1 * cz])
        dr_cz = dream(Y, design=X, random_effects=[Z1 * cz], contrasts={"trt": [0.0, 1.0]}, ebayes=True)
        dr_2d_cz = dream(
            Y, design=X, random_effects=[Z1 * cz], weights=W, contrasts={"trt": [0.0, 1.0]}, ebayes=True
        )
        assert isinstance(dr_cz, DreamResult) and isinstance(dr_2d_cz, DreamResult)
        assert_allclose(vp_cz.fractions, vp_r1_ref.fractions, atol=1e-7, rtol=1e-7, equal_nan=True)
        assert_allclose(dr_cz.df_satterthwaite, dr_r1_ref.df_satterthwaite, atol=1e-7, rtol=1e-7, equal_nan=True)
        assert_allclose(dr_cz.p_value, dr_r1_ref.p_value, atol=1e-7, rtol=1e-7, equal_nan=True)
        assert_allclose(dr_2d_cz.df_satterthwaite, dr_r1_2d_ref.df_satterthwaite, atol=1e-7, rtol=1e-7, equal_nan=True)
        assert_allclose(dr_2d_cz.p_value, dr_r1_2d_ref.p_value, atol=1e-7, rtol=1e-7, equal_nan=True)

    # R=2 asymmetric Z scaling (1e-10 vs 1e+10) combined with W * 1e-8
    vp_r2_ref = fit_extract_var_part_model(Y, design=X, random_effects=[Z1, Z2], weights=W)
    dr_r2_ref = dream(Y, design=X, random_effects=[Z1, Z2], weights=W, contrasts={"trt": [0.0, 1.0]}, ebayes=True)
    vp_r2_sc = fit_extract_var_part_model(
        Y, design=X, random_effects=[Z1 * 1e-10, Z2 * 1e+10], weights=W * 1e-8
    )
    dr_r2_sc = dream(
        Y,
        design=X,
        random_effects=[Z1 * 1e-10, Z2 * 1e+10],
        weights=W * 1e-8,
        contrasts={"trt": [0.0, 1.0]},
        ebayes=True,
    )
    assert isinstance(dr_r2_ref, DreamResult) and isinstance(dr_r2_sc, DreamResult)
    assert_allclose(vp_r2_sc.fractions, vp_r2_ref.fractions, atol=1e-7, rtol=1e-7, equal_nan=True)
    assert_allclose(dr_r2_sc.df_satterthwaite, dr_r2_ref.df_satterthwaite, atol=1e-7, rtol=1e-7, equal_nan=True)
    assert_allclose(dr_r2_sc.p_value, dr_r2_ref.p_value, atol=1e-7, rtol=1e-7, equal_nan=True)


def test_collinear_random_effect_blocks_and_partial_nan_confounding() -> None:
    rng = np.random.default_rng(1818)
    m_donor, n_rep, G = 10, 4, 12
    N = m_donor * n_rep
    meta = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], N // 2), categories=["ctrl", "trt"]),
            "batch_collinear": np.repeat(["B0", "B1"], N // 2),
            "donor": np.repeat([f"D{i}" for i in range(m_donor)], n_rep),
        }
    )
    d_idx = np.repeat(np.arange(m_donor), n_rep)
    Y = (
        5.0
        + rng.normal(0.0, 1.0, size=(m_donor, G))[d_idx]
        + rng.normal(0.0, 0.5, size=(N, G))
    )
    W = rng.uniform(0.5, 2.5, size=(N, G))

    # Case A: ~ trt + (1 | batch_collinear) -> df_satt == N - 2 = 38.0 (spectral & MME & 2D-W)
    for kw in ({}, {"weights": W}, {"weights": W * 1e-8}):
        dr_a = dream(
            Y, "~ trt + (1 | batch_collinear)", data=meta, contrasts="trt_trt", ebayes=False, **kw
        )
        assert isinstance(dr_a, DreamFit)
        assert_allclose(dr_a.df_satterthwaite[:, 0], float(N - 2), atol=1e-12)
        assert_allclose(dr_a.lmm_fit.sigma_r_sq[:, 0], 0.0, atol=1e-12)
        assert dr_a.lmm_fit.fisher_info is not None
        assert_allclose(dr_a.lmm_fit.fisher_info[:, 0, :], 0.0, atol=1e-12)
        assert_allclose(dr_a.lmm_fit.fisher_info[:, :, 0], 0.0, atol=1e-12)
        assert_allclose(dr_a.lmm_fit.blups[0], 0.0, atol=1e-12)

    fit_a_mme = fit_lmm(
        Y, "~ trt + (1 | batch_collinear)", data=meta, compute_fisher=True, _force_mme=True
    )
    assert_allclose(fit_a_mme.sigma_r_sq[:, 0], 0.0, atol=1e-12)
    assert fit_a_mme.fisher_info is not None
    assert_allclose(fit_a_mme.fisher_info[:, 0, :], 0.0, atol=1e-12)
    assert_allclose(fit_a_mme.fisher_info[:, :, 0], 0.0, atol=1e-12)

    # Case B: ~ trt + (1 | donor) + (1 | batch_collinear) -> matches ~ trt + (1 | donor), 0 fake variance
    for kw in ({}, {"weights": W}, {"weights": W * 1e-8}):
        dr_ref_b = dream(Y, "~ trt + (1 | donor)", data=meta, contrasts="trt_trt", ebayes=False, **kw)
        vp_ref_b = fit_extract_var_part_model(Y, "~ trt + (1 | donor)", data=meta, **kw)
        dr_b = dream(
            Y,
            "~ trt + (1 | donor) + (1 | batch_collinear)",
            data=meta,
            contrasts="trt_trt",
            ebayes=False,
            **kw,
        )
        vp_b = fit_extract_var_part_model(
            Y, "~ trt + (1 | donor) + (1 | batch_collinear)", data=meta, **kw
        )
        assert isinstance(dr_ref_b, DreamFit) and isinstance(dr_b, DreamFit)
        if not kw:
            assert_allclose(dr_b.df_satterthwaite[:, 0], float(m_donor - 2), atol=0.1)
        assert_allclose(dr_b.df_satterthwaite, dr_ref_b.df_satterthwaite, atol=1e-7)
        assert_allclose(dr_b.lmm_fit.sigma_r_sq[:, 1], 0.0, atol=1e-12)
        assert dr_b.lmm_fit.fisher_info is not None
        assert_allclose(dr_b.lmm_fit.fisher_info[:, 1, :], 0.0, atol=1e-12)
        assert_allclose(dr_b.lmm_fit.fisher_info[:, :, 1], 0.0, atol=1e-12)
        assert_allclose(vp_b.fractions[:, 2], 0.0, atol=1e-12)
        assert_allclose(vp_b.fractions[:, [0, 1, 3]], vp_ref_b.fractions, atol=1e-7)

    # Case C: Partial-NaN induced collinearity on Gene 0 (4 batches B0,B1 in ctrl; B2,B3 in trt; missing B1 & B3)
    meta_c = pd.DataFrame(
        {
            "trt": pd.Categorical(np.repeat(["ctrl", "trt"], 20), categories=["ctrl", "trt"]),
            "batch": np.repeat(["B0", "B1", "B2", "B3"], 10),
        }
    )
    Y_c = (
        5.0
        + rng.normal(0.0, 1.0, size=(4, G))[np.repeat(np.arange(4), 10)]
        + rng.normal(0.0, 0.5, size=(40, G))
    )
    Y_c[(meta_c["batch"] == "B1") | (meta_c["batch"] == "B3"), 0] = np.nan
    dr_c = dream(Y_c, "~ trt + (1 | batch)", data=meta_c, contrasts="trt_trt", ebayes=False)
    assert isinstance(dr_c, DreamFit)
    assert_allclose(dr_c.df_satterthwaite[0, 0], 18.0, atol=1e-12)
    assert dr_c.lmm_fit.sigma_r_sq[0, 0] == 0.0
    assert dr_c.lmm_fit.fisher_info is not None
    assert_allclose(dr_c.lmm_fit.fisher_info[0, 0, :], 0.0, atol=1e-12)
    assert_allclose(dr_c.lmm_fit.fisher_info[0, :, 0], 0.0, atol=1e-12)
    assert np.all(dr_c.df_satterthwaite[1:, 0] < 10.0)

    # Case D: Within-subject ~ trt + subject + (1 | subject) -> df_satt == N - rank(X) = 29.0 for all contrasts
    meta_d = pd.DataFrame(
        {
            "trt": pd.Categorical(np.tile(["ctrl", "trt"], 20), categories=["ctrl", "trt"]),
            "subject": pd.Categorical(np.repeat([f"S{i}" for i in range(m_donor)], n_rep)),
        }
    )
    Y_d = (
        5.0
        + rng.normal(0.0, 1.0, size=(m_donor, G))[d_idx]
        + rng.normal(0.0, 0.5, size=(N, G))
    )
    for kw in ({}, {"weights": W}, {"weights": W * 1e+8}):
        dr_d = dream(
            Y_d,
            "~ trt + subject + (1 | subject)",
            data=meta_d,
            contrasts=["trt_trt", "subject_S1", "subject_S5"],
            ebayes=False,
            **kw,
        )
        assert isinstance(dr_d, DreamFit)
        assert_allclose(dr_d.df_satterthwaite, 29.0, atol=1e-12)
        assert_allclose(dr_d.lmm_fit.sigma_r_sq[:, 0], 0.0, atol=1e-12)
        assert dr_d.lmm_fit.fisher_info is not None
        assert_allclose(dr_d.lmm_fit.fisher_info[:, 0, :], 0.0, atol=1e-12)
        assert_allclose(dr_d.lmm_fit.fisher_info[:, :, 0], 0.0, atol=1e-12)

