"""Deterministic unit and statistical validation tests for `cibersortx`."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import optimize, sparse

from cibersortx import (
    BatchAlignmentResult,
    DeconvolutionResult,
    GroupGEPResult,
    HighResGEPResult,
    SignatureMatrixResult,
    b_mode_adjust,
    build_signature_matrix,
    cibersort,
    combat_align,
    deconvolve,
    impute_gep_group,
    impute_gep_highres,
    s_mode_adjust,
)


def _make_signature_and_mixtures(
    *,
    n_genes: int = 120,
    n_types: int = 4,
    n_samples: int = 10,
    noise_sd: float = 0.02,
    scale_col0: float = 4.0,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    cell_types = ["Z_Macrophage", "B_Cell", "T_CD8", "Fibroblast"][:n_types]
    sample_names = [f"S_{i:02d}" for i in range(n_samples - 1, -1, -1)]
    gene_names = [f"GENE_{g:03d}" for g in range(n_genes)]

    B = rng.uniform(0.5, 3.0, size=(n_genes, n_types))
    block = n_genes // n_types
    for k in range(n_types):
        start = k * block
        stop = (k + 1) * block if k < n_types - 1 else n_genes
        B[start:stop, k] += rng.uniform(12.0, 25.0, size=stop - start)
    B[:, 0] *= scale_col0

    F_true = rng.dirichlet(np.full(n_types, 1.5), size=n_samples)
    M_clean = B @ F_true.T
    M = np.maximum(M_clean + rng.normal(0.0, noise_sd * np.mean(M_clean), size=M_clean.shape), 0.0)

    B_df = pd.DataFrame(B, index=gene_names, columns=cell_types)
    M_df = pd.DataFrame(M, index=gene_names, columns=sample_names)
    return B_df, M_df, F_true, sample_names, cell_types


def test_deconvolve_planted_truth_and_nu_selection() -> None:
    B_df, M_df, F_true, sample_names, cell_types = _make_signature_and_mixtures(
        n_genes=120, n_types=4, n_samples=10, noise_sd=0.005, scale_col0=4.0, seed=7
    )

    res = deconvolve(M_df, B_df, n_permutations=0, seed=7)
    res_alias = cibersort(M_df, B_df, n_permutations=0, seed=7)

    assert isinstance(res, DeconvolutionResult)
    np.testing.assert_allclose(res.fractions, res_alias.fractions, atol=1e-12)
    assert res.sample_names == sample_names
    assert res.cell_types == cell_types
    assert res.n_genes_used == 120
    assert res.batch_mode == "none"

    np.testing.assert_allclose(res.fractions.sum(axis=1), 1.0, atol=1e-12)
    max_err = float(np.max(np.abs(res.fractions - F_true)))
    assert max_err < 0.02, f"Max fraction error {max_err} exceeded 0.02"
    assert np.all(res.pearson_r > 0.99)
    assert set(np.unique(res.best_nu)).issubset({0.25, 0.5, 0.75})

    df_full = res.to_dataframe()
    df_frac = res.fractions_dataframe()
    assert df_full.shape == (10, 8)
    assert df_frac.shape == (10, 4)
    assert list(df_full.index) == sample_names
    assert list(df_full.columns) == cell_types + ["p_value", "pearson_r", "rmse", "best_nu"]
    assert "DeconvolutionResult(" in res.summary()

    # AnnData (obs x var) parity with DataFrame (genes x samples)
    adata_mix = ad.AnnData(X=M_df.T.to_numpy(dtype=np.float64))
    adata_mix.obs_names = sample_names
    adata_mix.var_names = list(M_df.index)
    res_ad = deconvolve(adata_mix, B_df, n_permutations=0, seed=7)
    np.testing.assert_allclose(res_ad.fractions, res.fractions, atol=1e-12)


def test_deconvolve_null_calibration_and_signal_significance() -> None:
    B_df, M_sig_df, _, _, _ = _make_signature_and_mixtures(
        n_genes=100, n_types=4, n_samples=5, noise_sd=0.02, scale_col0=1.0, seed=19
    )
    rng = np.random.default_rng(42)
    M_null = rng.lognormal(mean=1.5, sigma=1.0, size=(100, 12))
    null_cols = [f"Null_{i}" for i in range(12)]
    M_null_df = pd.DataFrame(M_null, index=B_df.index, columns=null_cols)
    M_all = pd.concat([M_sig_df, M_null_df], axis=1)

    res = deconvolve(M_all, B_df, n_permutations=99, seed=42)
    p_sig = res.p_value[:5]
    p_null = res.p_value[5:]

    assert np.all(p_sig <= 0.05)
    assert np.all(res.pearson_r[:5] > 0.90)
    assert np.all(res.p_value >= 1.0 / 100.0)  # Phipson-Smyth +1 smoothing
    assert float(np.median(p_null)) > 0.20
    assert int(np.sum(p_null <= 0.05)) <= 2

    res_pooled = deconvolve(M_all, B_df, n_permutations=99, permutation_mode="pooled", seed=42)
    assert np.all(res_pooled.p_value[:5] <= 0.05)


def test_build_signature_matrix_anndata_and_cond_optimization() -> None:
    rng = np.random.default_rng(101)
    n_cells_per_type = 80
    cell_types = ["Myeloid", "B_cell", "CD8_T", "Epithelial"]
    n_genes = 250
    genes = [f"G_{i:03d}" for i in range(n_genes)]

    counts_list = []
    labels = []
    for k, ct in enumerate(cell_types):
        base = rng.poisson(2.0, size=(n_cells_per_type, n_genes)).astype(np.float64)
        marker_start = k * 30
        marker_end = marker_start + 30
        base[:, marker_start:marker_end] += rng.poisson(25.0, size=(n_cells_per_type, 30))
        counts_list.append(base)
        labels.extend([ct] * n_cells_per_type)

    X_dense = np.vstack(counts_list)
    obs = pd.DataFrame({"cell_type": labels}, index=[f"cell_{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=genes)

    adata_dense = ad.AnnData(X=X_dense, obs=obs, var=var)
    adata_sparse = ad.AnnData(X=np.zeros_like(X_dense), obs=obs.copy(), var=var.copy())
    adata_sparse.layers["counts"] = sparse.csr_matrix(X_dense)

    sig_dense = build_signature_matrix(
        adata_dense,
        "cell_type",
        min_genes=10,
        max_genes=30,
        step=5,
        seed=0,
    )
    sig_sparse = build_signature_matrix(
        adata_sparse,
        "cell_type",
        layer="counts",
        min_genes=10,
        max_genes=30,
        step=5,
        seed=0,
    )

    assert isinstance(sig_dense, SignatureMatrixResult)
    assert sig_dense.cell_types == cell_types  # non-alphabetical order preserved
    assert sig_dense.condition_number == pytest.approx(min(sig_dense.condition_curve.values()))
    np.testing.assert_allclose(sig_dense.signature, sig_sparse.signature, rtol=1e-12, atol=1e-12)
    assert sig_dense.gene_names == sig_sparse.gene_names

    # Planted markers (G_000..G_119) should dominate selected genes
    planted_markers = {f"G_{i:03d}" for i in range(120)}
    overlap = len(set(sig_dense.gene_names) & planted_markers)
    assert overlap / len(sig_dense.gene_names) >= 0.85

    df_sig = sig_dense.to_dataframe()
    assert df_sig.shape == (len(sig_dense.gene_names), 4)
    assert "log2_fc" in sig_dense.deg_dataframe().columns

    # Welch t-test + replicate partitioning path
    sig_tt = build_signature_matrix(
        adata_dense,
        "cell_type",
        replicates_per_type=4,
        test_method="ttest",
        min_genes=10,
        max_genes=25,
        step=5,
        seed=0,
    )
    assert len(sig_tt.gene_names) >= 10


def test_b_mode_batch_alignment_improves_biased_bulk() -> None:
    rng = np.random.default_rng(202)
    B_df, M_clean_df, F_true, _, _ = _make_signature_and_mixtures(
        n_genes=150, n_types=4, n_samples=30, noise_sd=0.01, scale_col0=1.0, seed=202
    )
    eff_g = np.exp(rng.normal(0.0, 0.9, size=(150, 1)))
    M_biased = pd.DataFrame(
        M_clean_df.to_numpy() * eff_g,
        index=M_clean_df.index,
        columns=M_clean_df.columns,
    )

    res_none = deconvolve(M_biased, B_df, batch_mode="none", n_permutations=0, seed=0)
    res_b = deconvolve(M_biased, B_df, batch_mode="B", n_permutations=0, seed=0)
    res_b_direct = b_mode_adjust(M_biased, B_df, n_permutations=0, seed=0)

    mae_none = float(np.mean(np.abs(res_none.fractions - F_true)))
    mae_b = float(np.mean(np.abs(res_b.fractions - F_true)))
    r_none = float(np.corrcoef(res_none.fractions.ravel(), F_true.ravel())[0, 1])
    r_b = float(np.corrcoef(res_b.fractions.ravel(), F_true.ravel())[0, 1])

    assert res_b.batch_mode == "B"
    assert res_b.adjusted_mixture is not None
    np.testing.assert_allclose(res_b.fractions, res_b_direct.fractions, atol=1e-12)
    assert mae_b < mae_none, f"Expected MAE reduction: {mae_none:.4f} -> {mae_b:.4f}"
    assert r_b > r_none
    corr_biased = float(np.corrcoef(M_biased.to_numpy().ravel(), M_clean_df.to_numpy().ravel())[0, 1])
    corr_adj = float(np.corrcoef(res_b.adjusted_mixture.ravel(), M_clean_df.to_numpy().ravel())[0, 1])
    assert corr_adj > 0.90 and corr_adj > corr_biased + 0.25


def test_s_mode_batch_alignment_improves_sc_reference() -> None:
    rng = np.random.default_rng(303)
    n_genes = 150
    n_types = 4
    cell_types = ["CT_A", "CT_B", "CT_C", "CT_D"]
    genes = [f"G_{g:03d}" for g in range(n_genes)]

    B_bulk = rng.uniform(2.0, 6.0, size=(n_genes, n_types))
    block = n_genes // n_types
    for k in range(n_types):
        s = k * block
        e = (k + 1) * block if k < n_types - 1 else n_genes
        B_bulk[s:e, k] += rng.uniform(20.0, 40.0, size=e - s)

    # Strong gene-specific scRNA capture efficiency bias relative to bulk
    cap_g = np.exp(rng.normal(0.0, 0.9, size=n_genes))
    sc_cells = []
    sc_labels = []
    for k, ct in enumerate(cell_types):
        mu_sc = B_bulk[:, k] * cap_g
        for _ in range(50):
            cell_expr = np.maximum(mu_sc * rng.lognormal(0.0, 0.15, size=n_genes), 0.0)
            sc_cells.append(cell_expr)
            sc_labels.append(ct)
    sc_mat = np.vstack(sc_cells)  # (200 cells, 150 genes)
    sc_adata = ad.AnnData(
        X=sc_mat,
        obs=pd.DataFrame({"cell_type": sc_labels}, index=[f"c_{i}" for i in range(len(sc_labels))]),
        var=pd.DataFrame(index=genes),
    )

    # Build biased signature B_S directly from sc_adata cell-type means
    B_S = np.column_stack(
        [np.mean(sc_mat[np.asarray(sc_labels) == ct, :], axis=0) for ct in cell_types]
    )
    B_S_df = pd.DataFrame(B_S, index=genes, columns=cell_types)

    F_true = rng.dirichlet(np.ones(n_types), size=30)
    M_bulk = pd.DataFrame(
        B_bulk @ F_true.T,
        index=genes,
        columns=[f"Bulk_{i}" for i in range(30)],
    )

    res_none = deconvolve(M_bulk, B_S_df, batch_mode="none", n_permutations=0, seed=0)
    res_s = deconvolve(
        M_bulk,
        B_S_df,
        batch_mode="S",
        sc_ref=sc_adata,
        sc_cell_type_labels="cell_type",
        n_synthetic=100,
        cells_per_mixture=100,
        n_permutations=0,
        seed=0,
    )
    res_s_direct = s_mode_adjust(
        M_bulk,
        B_S_df,
        sc_adata,
        "cell_type",
        n_synthetic=100,
        cells_per_mixture=100,
        n_permutations=0,
        seed=0,
    )

    mae_none = float(np.mean(np.abs(res_none.fractions - F_true)))
    mae_s = float(np.mean(np.abs(res_s.fractions - F_true)))
    r_s = float(np.corrcoef(res_s.fractions.ravel(), F_true.ravel())[0, 1])

    assert res_s.batch_mode == "S"
    assert res_s.adjusted_signature is not None
    np.testing.assert_allclose(res_s.fractions, res_s_direct.fractions, atol=1e-12)
    assert mae_s <= 0.50 * mae_none, f"Expected >=50% MAE reduction: {mae_none:.4f} -> {mae_s:.4f}"
    assert r_s > 0.95


def test_combat_identity_invariance_and_degenerate_cases() -> None:
    rng = np.random.default_rng(404)
    Y = rng.uniform(1.0, 50.0, size=(40, 12))
    # Inject constant zero-variance genes and near-constant genes
    Y[0, :] = 0.0
    Y[1, :] = 7.5
    df = pd.DataFrame(Y, index=[f"G_{i}" for i in range(40)], columns=[f"S_{j}" for j in range(12)])

    res_id = combat_align(df, df, space="reference")
    assert isinstance(res_id, BatchAlignmentResult)
    np.testing.assert_allclose(res_id.adjusted_target, Y, atol=1e-12)
    assert res_id.to_dataframe("target").shape == (40, 12)

    res_common = combat_align(df, df, space="common")
    assert res_common.adjusted_reference is not None
    np.testing.assert_allclose(
        res_common.adjusted_target, res_common.adjusted_reference, atol=1e-12
    )

    # Identical gene variances across batches (S_i^2 == 0)
    Y_homog_1 = np.tile(np.arange(10.0), (20, 1)) + 1.0
    Y_homog_2 = Y_homog_1 * 2.0 + 3.0
    res_homog = combat_align(Y_homog_1, Y_homog_2)
    assert np.all(np.isfinite(res_homog.adjusted_target))
    assert np.min(res_homog.adjusted_target) >= 0.0

    # Single-sample batch (n1 == 1) -> warning + finite mean-only alignment
    with pytest.warns(UserWarning, match="Batch size < 2"):
        res_single = combat_align(Y[:, :1], Y[:, 1:])
    assert np.all(np.isfinite(res_single.adjusted_target))
    assert np.min(res_single.adjusted_target) >= 0.0


def test_impute_gep_group_recovery_and_noise_filtering() -> None:
    rng = np.random.default_rng(505)
    n_genes, n_samples, n_types = 60, 40, 4
    genes = [f"G_{g:02d}" for g in range(n_genes)]
    samples = [f"Sample_{s:02d}" for s in range(n_samples)]
    cts = ["Abundant_A", "Moderate_B", "Moderate_C", "Rare_D"]

    # Rare_D has low mean fraction (~0.03), Abundant_A has high fraction (~0.50)
    raw_f = rng.dirichlet([10.0, 5.0, 4.0, 0.7], size=n_samples)
    F_df = pd.DataFrame(raw_f, index=samples, columns=cts)

    G_true = rng.uniform(10.0, 30.0, size=(n_genes, n_types))
    G_true[20:40, 3] = 0.0  # pure zero in Rare_D
    G_true[40:60, :] = 0.0  # all-zero genes

    M = G_true @ raw_f.T + rng.normal(0.0, 0.3, size=(n_genes, n_samples))
    M = np.maximum(M, 0.0)
    M[40:60, :] = 0.0  # genuinely unexpressed genes
    M_df = pd.DataFrame(M, index=genes, columns=samples)

    grp = impute_gep_group(M_df, F_df, n_bootstrap=50, seed=505)
    assert isinstance(grp, GroupGEPResult)

    # Strong recovery on abundant/moderate cell types
    r_abundant = float(np.corrcoef(grp.gep_raw[:40, 0], G_true[:40, 0])[0, 1])
    assert r_abundant > 0.98
    # Bootstrap SE should be smaller for abundant CT0 than rarer CT3
    assert float(np.median(grp.se[:20, 0])) < float(np.median(grp.se[:20, 3]))

    # Strongly expressed genes pass filter; pure-zero genes in CT3 and unexpressed genes are filtered
    assert np.all(grp.passed_filter[:20, :3])
    assert not np.any(grp.passed_filter[40:60, :])
    assert int(np.sum(grp.passed_filter[20:40, 3])) <= 2
    assert np.all(np.isnan(grp.gep_filtered[~grp.passed_filter]))

    assert grp.gep_dataframe(filtered=True).shape == (60, 4)
    tidy = grp.to_dataframe(filtered=False)
    assert tidy.shape == (240, 9)

    # Absent cell type (all zero column) -> NaN in gep_filtered without altering valid q_values
    F_absent = F_df.copy()
    F_absent["Rare_D"] = 0.0
    grp_absent = impute_gep_group(M_df, F_absent, n_bootstrap=30, seed=505)
    assert np.all(np.isnan(grp_absent.gep_filtered[:, 3]))
    assert np.all(np.isnan(grp_absent.q_value[:, 3]))
    assert np.all(np.isfinite(grp_absent.q_value[:20, 0]))


def test_impute_gep_highres_subtype_purification_and_null_specificity() -> None:
    rng = np.random.default_rng(606)
    n_genes, n_samples, n_types = 24, 45, 4
    genes = [f"G_{g:02d}" for g in range(n_genes)]
    samples = [f"Pat_{s:02d}" for s in range(n_samples)]
    cts = ["Epithelial", "T_cell", "Fibroblast", "Macrophage"]

    F = rng.dirichlet([3.0, 2.0, 2.0, 2.0], size=n_samples)
    F_df = pd.DataFrame(F, index=samples, columns=cts)

    # Plant subtype gradient in Epithelial (CT0) for genes 0..7; constant for 8..23
    X_true = np.zeros((n_genes, n_samples, n_types), dtype=np.float64)
    for k in range(n_types):
        base_ct = rng.uniform(4.0, 10.0, size=n_genes)
        X_true[:, :, k] = base_ct[:, None]
    X_true[:8, :22, 0] = 20.0
    X_true[:8, 22:, 0] = 2.0

    M = np.einsum("gnk,nk->gn", X_true, F) + rng.normal(0.0, 0.4, size=(n_genes, n_samples))
    M = np.maximum(M, 0.0)
    M_df = pd.DataFrame(M, index=genes, columns=samples)

    hires = impute_gep_highres(M_df, F_df, seed=0)
    assert isinstance(hires, HighResGEPResult)

    df_ct0 = hires.gep_dataframe("Epithelial", filtered=False)
    assert df_ct0.shape == (24, 45)
    # Correlation across varying genes in CT0
    corrs = [
        float(np.corrcoef(df_ct0.iloc[g].to_numpy(), X_true[g, :, 0])[0, 1])
        for g in range(8)
    ]
    assert float(np.mean(corrs)) > 0.85
    mean_a = float(np.mean(df_ct0.iloc[:8, :22].to_numpy()))
    mean_b = float(np.mean(df_ct0.iloc[:8, 22:].to_numpy()))
    assert mean_a > 12.0 and mean_b < 5.0

    # Default settings must recover DE genes via BH-FDR (is_variable & gep_filtered)
    tp_recall = float(np.mean(hires.is_variable[:8, 0]))
    assert tp_recall >= 0.8, f"Expected default High-Res is_variable recall >= 0.8, got {tp_recall:.3f}"
    assert float(np.mean(np.isfinite(hires.gep_filtered[:8, :, 0]))) >= 0.8
    var_mask = np.broadcast_to(hires.is_variable[:, None, :], hires.gep_filtered.shape)
    assert np.all(np.isfinite(hires.gep_filtered[var_mask]))
    assert np.all(np.isnan(hires.gep_filtered[~var_mask]))
    assert int(np.sum(hires.is_variable[:8, 1:])) <= 1

    # Constant null genes should mostly remain non-variable
    null_fp = int(np.sum(hires.is_variable[8:, :]))
    assert null_fp <= int(0.10 * (16 * 4)), f"Too many null false positives: {null_fp}"

    # Sparse DE set (15% DE: 3/20 genes) with default parameters must still achieve q <= 0.05
    X_sparse = X_true[:20, :, :].copy()
    X_sparse[3:8, :, 0] = rng.uniform(4.0, 10.0, size=(5, 1))
    M_sparse = np.maximum(
        np.einsum("gnk,nk->gn", X_sparse, F) + rng.normal(0.0, 0.4, size=(20, n_samples)),
        0.0,
    )
    hires_sparse = impute_gep_highres(
        pd.DataFrame(M_sparse, index=genes[:20], columns=samples),
        F_df,
        seed=0,
    )
    assert float(np.mean(hires_sparse.is_variable[:3, 0])) >= 0.8
    assert np.all(hires_sparse.q_value[:3, 0] <= 0.05)
    assert int(np.sum(hires_sparse.is_variable[3:, :])) <= int(0.10 * (17 * 4))

    tidy = hires.to_dataframe(only_variable=False)
    assert tidy.shape == (24 * 4, 8)


def test_qr_nnls_analytical_parity_and_determinism() -> None:
    rng = np.random.default_rng(707)
    n_genes, n_samples, n_types = 30, 25, 4
    F = rng.dirichlet(np.ones(n_types), size=n_samples)
    G_true = rng.uniform(0.0, 15.0, size=(n_genes, n_types))
    # Force some zero true entries to exercise active NNLS boundary constraints
    G_true[:10, 1] = 0.0
    M = np.maximum(G_true @ F.T + rng.normal(0.0, 0.5, size=(n_genes, n_samples)), 0.0)

    grp_1 = impute_gep_group(M, F, n_bootstrap=10, ridge_alpha=0.0, seed=7)
    grp_2 = impute_gep_group(M, F, n_bootstrap=10, ridge_alpha=0.0, seed=7)

    # Exact parity with direct scipy.optimize.nnls
    for g in range(n_genes):
        direct_x, _ = optimize.nnls(F, M[g, :])
        np.testing.assert_allclose(grp_1.gep_raw[g, :], direct_x, atol=1e-11, rtol=1e-11)

    # Bit-exact determinism
    np.testing.assert_array_equal(grp_1.se, grp_2.se)
    np.testing.assert_array_equal(grp_1.gep_raw, grp_2.gep_raw)

    hires_1 = impute_gep_highres(M[:8], F, n_permutations=10, seed=7)
    hires_2 = impute_gep_highres(M[:8], F, n_permutations=10, seed=7)
    np.testing.assert_array_equal(hires_1.gep_raw, hires_2.gep_raw)
    np.testing.assert_array_equal(hires_1.p_value, hires_2.p_value)


def test_edge_cases_and_validation_errors() -> None:
    B_df, M_df, _, _, _ = _make_signature_and_mixtures(
        n_genes=30, n_types=3, n_samples=8, noise_sd=0.01, seed=808
    )

    # 1. Zero-sum SVR weights on strongly anti-correlated bulk sample -> no NaN/ZeroDivisionError
    B_anti = np.column_stack([B_df.to_numpy()[:, 0] * -1.0 + 100.0])
    # Construct sample that is negative linear combination of standardized signature
    b_std = (B_df.to_numpy() - B_df.to_numpy().mean()) / B_df.to_numpy().std(ddof=1)
    m_anti = -np.sum(b_std, axis=1)
    m_anti = m_anti - np.min(m_anti) + 1.0
    M_test = M_df.copy()
    M_test["AntiSample"] = m_anti
    res_anti = deconvolve(M_test, B_df, n_permutations=19, seed=1)
    assert np.all(np.isfinite(res_anti.fractions[-1, :]))
    assert np.all(res_anti.fractions[-1, :] >= 0.0)
    assert res_anti.p_value[-1] >= 0.05

    # 2. Zero-variance bulk sample alongside valid samples -> UserWarning + NaN row only for flat sample
    M_flat = M_df.copy()
    M_flat.iloc[:, 2] = 5.0
    with pytest.warns(UserWarning, match="zero variance"):
        res_flat = deconvolve(M_flat, B_df, n_permutations=0)
    assert np.all(np.isnan(res_flat.fractions[2, :]))
    assert np.all(np.isfinite(res_flat.fractions[0, :]))

    # 3. N = 1 bulk sample in B-mode -> UserWarning + valid unadjusted fallback
    with pytest.warns(UserWarning, match="at least 2 valid bulk samples"):
        res_b1 = deconvolve(M_df.iloc[:, :1], B_df, batch_mode="B", n_permutations=0)
    assert np.all(np.isfinite(res_b1.fractions))

    # 4. High-Res window clamping (w > N, w <= K, w == N)
    F_small = np.full((8, 3), 1.0 / 3.0)
    F_small[:, 0] = np.linspace(0.1, 0.7, 8)
    F_small[:, 1] = (1.0 - F_small[:, 0]) * 0.6
    F_small[:, 2] = 1.0 - F_small[:, 0] - F_small[:, 1]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hr_big_w = impute_gep_highres(M_df.iloc[:5, :], F_small, window_size=50, n_permutations=0)
        hr_small_w = impute_gep_highres(M_df.iloc[:5, :], F_small, window_size=2, n_permutations=0)
    assert hr_big_w.window_size == 8
    assert hr_small_w.window_size == 2
    assert len(rec) >= 2
    assert np.all(np.isfinite(hr_big_w.gep_raw)) and np.min(hr_big_w.gep_raw) >= 0.0
    assert np.all(np.isfinite(hr_small_w.gep_raw)) and np.min(hr_small_w.gep_raw) >= 0.0

    # 5. Validation ValueError / KeyError checks
    with pytest.raises(ValueError, match="Shared gene count"):
        deconvolve(M_df.iloc[:2, :], B_df, n_permutations=0)

    M_nan = M_df.copy()
    M_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        deconvolve(M_nan, B_df, n_permutations=0)

    M_neg = M_df.copy()
    M_neg.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        deconvolve(M_neg, B_df, n_permutations=0)

    M_dup = M_df.copy()
    M_dup.index = ["DUP"] * 2 + list(M_df.index[2:])
    with pytest.raises(ValueError, match="Duplicate gene"):
        deconvolve(M_dup, B_df, n_permutations=0)

    with pytest.raises(ValueError, match="batch_mode='S'"):
        deconvolve(M_df, B_df, batch_mode="S")

    with pytest.raises(ValueError, match="n_samples > n_cell_types"):
        impute_gep_group(M_df.iloc[:, :3], np.full((3, 3), 1.0 / 3.0))

    with pytest.raises(ValueError, match="n_samples > n_cell_types \\+ 2"):
        impute_gep_highres(M_df.iloc[:, :5], np.full((5, 3), 1.0 / 3.0))

    with pytest.raises(KeyError, match="Unknown cell_type"):
        hr_big_w.gep_dataframe("NonExistentCT")

    hr_mean_smooth = impute_gep_highres(
        M_df.iloc[:5, :],
        F_small,
        aggregation="mean",
        sort_by="bulk",
        smooth_window=3,
        n_permutations=5,
        seed=0,
    )
    assert np.all(np.isfinite(hr_mean_smooth.gep_raw))
    assert np.all(np.isfinite(hr_mean_smooth.p_value))

    adata_none = ad.AnnData(X=None, obs=pd.DataFrame(index=M_df.columns), var=pd.DataFrame(index=M_df.index))
    with pytest.raises(ValueError, match="AnnData expression matrix is None"):
        deconvolve(adata_none, B_df, n_permutations=0)

