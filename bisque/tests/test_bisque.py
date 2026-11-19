from __future__ import annotations

from dataclasses import FrozenInstanceError
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import nnls

from bisque import (
    MarkerDeconvolutionResult,
    ReferenceDeconvolutionResult,
    SCReference,
    generate_sc_reference,
    marker_based_decomposition,
    reference_based_decomposition,
)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_c * x_c) * np.sum(y_c * y_c)))
    if denom <= 1e-15:
        return 0.0
    return float(np.sum(x_c * y_c) / denom)


def _simulate_reference_cohort(
    *,
    seed: int = 42,
    n_donors: int = 12,
    n_bulk: int = 24,
    n_genes: int = 90,
    cell_types: list[str] | None = None,
    cells_per_donor: int = 90,
    donor_names: list[str] | None = None,
    bulk_names: list[str] | None = None,
):
    rng = np.random.default_rng(seed)
    if cell_types is None:
        cell_types = ["B_cell", "Macrophage", "T_cell"]
    n_ct = len(cell_types)
    genes = [f"Gene_{g:03d}" for g in range(n_genes)]

    if donor_names is None:
        donor_names = [f"Donor_{d:02d}" for d in range(n_donors)]
    if bulk_names is None:
        bulk_names = [f"Bulk_{i:02d}" for i in range(n_bulk)]

    # Base cell-type signatures Z_true (n_genes x n_ct) with clear cell-type markers
    Z_true = rng.uniform(5.0, 25.0, size=(n_genes, n_ct))
    block = n_genes // n_ct
    for k in range(n_ct):
        start = k * block
        end = (k + 1) * block if k < n_ct - 1 else n_genes
        Z_true[start:end, k] += rng.uniform(60.0, 140.0, size=end - start)

    # Single-cell donor proportions and single-cell counts
    sc_props_true = rng.dirichlet(np.full(n_ct, 2.0), size=n_donors)
    sc_rows = []
    sc_ct = []
    sc_subj = []
    for d in range(n_donors):
        counts_ct = rng.multinomial(cells_per_donor, sc_props_true[d])
        for k, ct in enumerate(cell_types):
            n_k = int(counts_ct[k])
            if n_k == 0:
                continue
            lam = Z_true[:, k][None, :]
            cells = rng.poisson(lam, size=(n_k, n_genes)).astype(np.float64)
            sc_rows.append(cells)
            sc_ct.extend([ct] * n_k)
            sc_subj.extend([donor_names[d]] * n_k)

    sc_X = np.vstack(sc_rows)
    sc_adata = ad.AnnData(X=sc_X)
    sc_adata.var_names = genes
    sc_adata.obs_names = [f"Cell_{i:05d}" for i in range(sc_X.shape[0])]
    sc_adata.obs["cell_type"] = sc_ct
    sc_adata.obs["subject_id"] = sc_subj

    # Bulk ground-truth proportions and unshifted clean expression
    bulk_props_true = rng.dirichlet(np.full(n_ct, 1.8), size=n_bulk)
    Y_bulk_clean = bulk_props_true @ Z_true.T  # (n_bulk, n_genes)

    return sc_adata, Y_bulk_clean, bulk_props_true, Z_true, genes, donor_names, bulk_names


def test_reference_planted_truth_platform_shift():
    """Bisque recovers planted proportions under strong affine platform shift where naive NNLS fails."""
    rng = np.random.default_rng(42)
    sc_adata, Y_bulk_clean, bulk_props_true, _, genes, _, bulk_names = (
        _simulate_reference_cohort(seed=42, n_donors=30, n_bulk=30, n_genes=90)
    )

    # Strong gene-specific multiplicative + additive cross-platform distortion
    b_g = np.exp(rng.normal(0.0, 2.0, size=len(genes)))
    a_g = rng.uniform(500.0, 5000.0, size=len(genes))
    B_shifted = a_g[None, :] + Y_bulk_clean * b_g[None, :]

    bulk_df = pd.DataFrame(B_shifted, index=bulk_names, columns=genes)

    # Naive unadjusted NNLS directly against SC signature fails
    sc_ref = generate_sc_reference(sc_adata, normalize_cpm=False)
    naive_props = np.zeros_like(bulk_props_true)
    for i in range(B_shifted.shape[0]):
        q_hat, _ = nnls(sc_ref.signature_matrix, B_shifted[i, :])
        naive_props[i, :] = q_hat / np.sum(q_hat)
    naive_mae = float(np.mean(np.abs(naive_props - bulk_props_true)))
    naive_r = float(
        np.mean([_pearsonr(naive_props[:, k], bulk_props_true[:, k]) for k in range(3)])
    )
    assert naive_mae > 0.15 or naive_r < 0.6

    # Bisque affine alignment recovers true proportions accurately
    res = reference_based_decomposition(
        bulk_df,
        sc_adata,
        use_overlap=False,
        normalize_cpm=False,
    )
    assert isinstance(res, ReferenceDeconvolutionResult)
    assert res.mode == "non_overlapping"
    assert not res.used_fallback

    est = res.to_dataframe().to_numpy()
    corrs = [_pearsonr(est[:, k], bulk_props_true[:, k]) for k in range(3)]
    mae = float(np.mean(np.abs(est - bulk_props_true)))
    assert min(corrs) > 0.95
    assert mae < 0.05


def test_reference_overlap_mode_recovery():
    """Overlapping paired regression mode shrinks uncorrelated noisy genes and recovers held-out bulk."""
    rng = np.random.default_rng(202)
    n_overlap = 14
    n_heldout = 16
    n_donors = 18
    n_genes = 120
    cell_types = ["B_cell", "Macrophage", "T_cell"]
    donor_names = [f"Subj_{d:02d}" for d in range(n_donors)]

    sc_adata, _, _, _, genes, _, _ = _simulate_reference_cohort(
        seed=202,
        n_donors=n_donors,
        n_bulk=n_heldout,
        n_genes=n_genes,
        cell_types=cell_types,
        cells_per_donor=200,
        donor_names=donor_names,
    )

    sc_ref = generate_sc_reference(sc_adata, normalize_cpm=False)
    # Build bulk for overlapping donors (using SC donor proportions) + held-out samples
    overlap_ids = donor_names[:n_overlap]
    overlap_props = sc_ref.sc_props[:n_overlap, :]
    heldout_ids = [f"HeldOut_{i:02d}" for i in range(n_heldout)]
    heldout_props = rng.dirichlet([1.8, 1.8, 1.8], size=n_heldout)

    all_props = np.vstack([overlap_props, heldout_props])
    all_ids = overlap_ids + heldout_ids
    Y_clean = all_props @ sc_ref.signature_matrix.T

    # 75% genes obey affine shift; 25% evenly interleaved genes are uncorrelated technical noise
    b_g = np.exp(rng.normal(0.0, 1.2, size=n_genes))
    a_g = rng.uniform(50.0, 1500.0, size=n_genes)
    B = a_g[None, :] + Y_clean * b_g[None, :]
    noise_cols = np.arange(0, n_genes, 4)
    B[:, noise_cols] = rng.normal(
        500.0, 150.0, size=(len(all_ids), len(noise_cols))
    )

    bulk_adata = ad.AnnData(X=B)
    bulk_adata.obs_names = all_ids
    bulk_adata.var_names = genes

    res = reference_based_decomposition(
        bulk_adata,
        sc_adata,
        use_overlap=True,
        normalize_cpm=False,
    )
    assert res.mode == "overlapping"
    assert res.overlapping_subjects == overlap_ids
    assert len(res.overlapping_subjects) == n_overlap

    est_df = res.to_dataframe()
    est_heldout = est_df.loc[heldout_ids].to_numpy()
    corrs = [_pearsonr(est_heldout[:, k], heldout_props[:, k]) for k in range(3)]
    mae = float(np.mean(np.abs(est_heldout - heldout_props)))
    assert min(corrs) > 0.95
    assert mae < 0.05


def test_analytical_sum_to_one_and_affine_invariance():
    """Estimated proportions lie on the simplex and are invariant to positive per-gene affine scaling."""
    rng = np.random.default_rng(303)
    sc_adata, Y_bulk_clean, _, _, genes, _, bulk_names = _simulate_reference_cohort(
        seed=303, n_donors=10, n_bulk=18, n_genes=60
    )
    bulk_df = pd.DataFrame(Y_bulk_clean, index=bulk_names, columns=genes)

    res_orig = reference_based_decomposition(
        bulk_df, sc_adata, use_overlap=False, normalize_cpm=False
    )
    np.testing.assert_allclose(res_orig.bulk_props.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(res_orig.bulk_props >= 0.0)

    # Positive per-gene affine transform B' = c_g + d_g * B
    c_g = rng.uniform(0.0, 5000.0, size=len(genes))
    d_g = rng.uniform(0.2, 50.0, size=len(genes))
    bulk_aff = pd.DataFrame(
        c_g[None, :] + Y_bulk_clean * d_g[None, :], index=bulk_names, columns=genes
    )
    res_aff = reference_based_decomposition(
        bulk_aff, sc_adata, use_overlap=False, normalize_cpm=False
    )

    np.testing.assert_allclose(res_orig.bulk_props, res_aff.bulk_props, atol=1e-10)
    np.testing.assert_allclose(
        res_orig.transformed_bulk, res_aff.transformed_bulk, atol=1e-10
    )

    # Result dataclass structure, accessors, summary, and immutability
    df = res_orig.to_dataframe()
    assert df.shape == (18, 3)
    assert list(df.index) == bulk_names
    assert res_orig.sc_dataframe().shape == (10, 3)
    assert res_orig.signature_dataframe().shape == (len(res_orig.genes_used), 3)
    assert res_orig.transformed_bulk_dataframe().shape == (18, len(res_orig.genes_used))
    summary = res_orig.summary()
    assert "ReferenceDeconvolutionResult" in summary
    assert "non_overlapping" in summary
    with pytest.raises(FrozenInstanceError):
        res_orig.mode = "tampered"  # type: ignore[misc]


def test_non_alphabetical_cell_types_and_sample_order():
    """Non-alphabetical cell types and bulk sample names are preserved without label scrambling."""
    non_alpha_ct = ["Z_Stromal", "A_Epithelial", "M_Immune", "B_Endothelial"]
    non_alpha_bulk = [f"Sample_{tag}" for tag in ["Z9", "A1", "M5", "B2", "Q7", "K3", "X0", "C8", "D4", "E6", "W1", "F2", "G9", "H3", "J5", "L4", "P8", "R2", "T6", "U1"]]
    sc_adata, Y_bulk_clean, bulk_props_true, _, genes, _, _ = _simulate_reference_cohort(
        seed=404,
        n_donors=32,
        n_bulk=len(non_alpha_bulk),
        n_genes=120,
        cell_types=non_alpha_ct,
        cells_per_donor=180,
        bulk_names=non_alpha_bulk,
    )
    # Apply platform shift and CPM counts
    rng = np.random.default_rng(404)
    b_g = rng.lognormal(0.0, 0.2, size=len(genes))
    B_counts = rng.poisson(np.maximum(1.0, Y_bulk_clean * b_g[None, :] * 10.0)).astype(
        np.float64
    )
    bulk_adata = ad.AnnData(X=sparse.csr_matrix(B_counts))
    bulk_adata.obs_names = non_alpha_bulk
    bulk_adata.var_names = genes

    sc_sparse = sc_adata.copy()
    sc_sparse.X = sparse.csr_matrix(sc_sparse.X)

    # Preserve explicit non-alphabetical order via select_ct
    res = reference_based_decomposition(
        bulk_adata,
        sc_sparse,
        select_ct=non_alpha_ct,
        use_overlap=False,
        normalize_cpm=True,
    )
    df = res.to_dataframe()
    assert list(df.index) == non_alpha_bulk
    assert list(df.columns) == non_alpha_ct
    np.testing.assert_allclose(df.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    mae = float(np.mean(np.abs(df.to_numpy() - bulk_props_true)))
    corrs = [_pearsonr(df[ct].to_numpy(), bulk_props_true[:, k]) for k, ct in enumerate(non_alpha_ct)]
    assert mae < 0.05
    assert min(corrs) > 0.95

    # Without select_ct (sorted alphabetically), proportions must still align to exact named columns
    res_sorted = reference_based_decomposition(
        bulk_adata,
        sc_sparse,
        use_overlap=False,
        normalize_cpm=True,
    )
    df_sorted = res_sorted.to_dataframe()
    assert list(df_sorted.columns) == sorted(non_alpha_ct)
    for k, ct in enumerate(non_alpha_ct):
        assert _pearsonr(df_sorted[ct].to_numpy(), bulk_props_true[:, k]) > 0.95


def test_old_cpm_and_marker_subsetting():
    """Verify marker intersection, select_ct reordering, SCReference, and old_cpm ordering effects."""
    rng = np.random.default_rng(505)
    sc_adata, Y_bulk_clean, _, _, genes, _, bulk_names = _simulate_reference_cohort(
        seed=505, n_donors=6, n_bulk=14, n_genes=60
    )
    # Append 10 omitted genes whose expression varies hugely across bulk samples
    extra_genes = [f"Extra_{j:02d}" for j in range(10)]
    all_genes = genes + extra_genes
    sc_X_ext = np.hstack([sc_adata.X, rng.uniform(1.0, 5.0, size=(sc_adata.n_obs, 10))])
    sc_ext = ad.AnnData(X=sc_X_ext, obs=sc_adata.obs.copy())
    sc_ext.var_names = all_genes
    sc_ext.obs_names = sc_adata.obs_names

    extra_bulk = np.zeros((len(bulk_names), 10), dtype=np.float64)
    extra_bulk[:7, :] = 10.0
    extra_bulk[7:, :] = 20000.0
    bulk_ext = pd.DataFrame(
        np.hstack([Y_bulk_clean, extra_bulk]), index=bulk_names, columns=all_genes
    )

    markers = genes[:45]
    select_ct = ["T_cell", "B_cell"]
    res_new = reference_based_decomposition(
        bulk_ext,
        sc_ext,
        markers=markers,
        select_ct=select_ct,
        old_cpm=False,
        normalize_cpm=True,
    )
    res_old = reference_based_decomposition(
        bulk_ext,
        sc_ext,
        markers=markers,
        select_ct=select_ct,
        old_cpm=True,
        normalize_cpm=True,
    )
    assert set(res_new.genes_used).issubset(set(markers))
    assert res_new.cell_types == select_ct
    assert res_old.cell_types == select_ct
    # old_cpm=False includes Extra_* in library size while old_cpm=True drops them before CPM
    assert not np.allclose(res_new.bulk_props, res_old.bulk_props, atol=1e-3)

    sc_ref = generate_sc_reference(sc_ext, markers=markers, select_ct=select_ct)
    assert isinstance(sc_ref, SCReference)
    assert sc_ref.cell_types == select_ct
    assert sc_ref.to_dataframe().shape == (len(markers), 2)


def test_marker_pca_recovery_and_weighted_pca():
    """Marker-based PCA recovers planted abundances and logFC weighting improves noisy markers."""
    rng = np.random.default_rng(606)
    n_bulk = 30
    cell_types = ["Epithelial", "Fibroblast", "Macrophage"]
    n_ct = len(cell_types)
    props_true = rng.dirichlet(np.full(n_ct, 2.0), size=n_bulk)

    # 8 strong markers + 4 noisy weak markers per cell type
    genes = []
    markers_dict: dict[str, list[str]] = {}
    marker_rows = []
    X_cols = []
    for k, ct in enumerate(cell_types):
        ct_genes = []
        for j in range(8):
            g = f"{ct}_strong_{j}"
            genes.append(g)
            ct_genes.append(g)
            expr = 50.0 + 400.0 * props_true[:, k] + rng.normal(0.0, 12.0, size=n_bulk)
            X_cols.append(np.maximum(1.0, expr))
            marker_rows.append({"cluster": ct, "gene": g, "avg_logFC": 3.5})
        for j in range(4):
            g = f"{ct}_weak_{j}"
            genes.append(g)
            ct_genes.append(g)
            expr = 200.0 + 10.0 * props_true[:, k] + rng.normal(0.0, 80.0, size=n_bulk)
            X_cols.append(np.maximum(1.0, expr))
            marker_rows.append({"cluster": ct, "gene": g, "avg_logFC": 0.1})
        markers_dict[ct] = ct_genes

    # Add background housekeeping genes so CPM scaling is non-trivial
    for j in range(20):
        genes.append(f"Housekeeping_{j}")
        X_cols.append(rng.uniform(400.0, 600.0, size=n_bulk))

    B = np.column_stack(X_cols)
    bulk_df = pd.DataFrame(
        B, index=[f"Sample_{i:02d}" for i in range(n_bulk)], columns=genes
    )
    markers_df = pd.DataFrame(marker_rows)

    res_unweighted = marker_based_decomposition(
        bulk_df,
        markers_dict,
        weighted_pca=False,
        normalize_cpm=False,
    )
    assert isinstance(res_unweighted, MarkerDeconvolutionResult)
    df_unw = res_unweighted.to_dataframe()
    corrs_unw = [_pearsonr(df_unw[ct].to_numpy(), props_true[:, k]) for k, ct in enumerate(cell_types)]
    assert min(corrs_unw) > 0.90

    res_weighted = marker_based_decomposition(
        bulk_df,
        markers_df,
        weighted_pca=True,
        normalize_cpm=False,
    )
    df_w = res_weighted.to_dataframe()
    corrs_w = [_pearsonr(df_w[ct].to_numpy(), props_true[:, k]) for k, ct in enumerate(cell_types)]
    assert np.mean(corrs_w) > np.mean(corrs_unw)
    assert res_weighted.weights_used is not None
    assert "MarkerDeconvolutionResult" in res_weighted.summary()


def test_marker_pca_sign_alignment_and_normalizations():
    """Sign alignment always orients PC1 positively with marker mean; minmax/proportional scale cleanly."""
    rng = np.random.default_rng(707)
    n_bulk = 25
    cell_types = ["Type_C", "Type_A", "Type_B"]
    props = rng.dirichlet([1.5, 2.0, 1.2], size=n_bulk)

    markers = {}
    cols = []
    genes = []
    for k, ct in enumerate(cell_types):
        ct_m = []
        for j in range(8):
            g = f"{ct}_m{j}"
            genes.append(g)
            ct_m.append(g)
            cols.append(20.0 + 300.0 * props[:, k] + rng.normal(0.0, 15.0, size=n_bulk))
        markers[ct] = ct_m

    B = np.maximum(1.0, np.column_stack(cols))
    bulk_df = pd.DataFrame(B, index=[f"S_{i}" for i in range(n_bulk)], columns=genes)

    res_none = marker_based_decomposition(
        bulk_df, markers, normalize_cpm=False, normalization="none"
    )
    for ct in cell_types:
        sub = bulk_df[markers[ct]].to_numpy()
        z_mean = np.mean((sub - np.mean(sub, axis=0)) / np.std(sub, axis=0, ddof=1), axis=1)
        assert _pearsonr(res_none.to_dataframe()[ct].to_numpy(), z_mean) > 0.9

    # Permuting rows does not invert sign
    perm = rng.permutation(n_bulk)
    res_perm = marker_based_decomposition(
        bulk_df.iloc[perm], markers, normalize_cpm=False, normalization="none"
    )
    np.testing.assert_allclose(
        res_perm.to_dataframe().loc[bulk_df.index].to_numpy(),
        res_none.to_dataframe().to_numpy(),
        atol=1e-10,
    )

    res_mm = marker_based_decomposition(
        bulk_df, markers, normalize_cpm=False, normalization="minmax"
    )
    df_mm = res_mm.to_dataframe()
    np.testing.assert_allclose(df_mm.min(axis=0).to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(df_mm.max(axis=0).to_numpy(), 1.0, atol=1e-12)

    res_prop = marker_based_decomposition(
        bulk_df, markers, normalize_cpm=False, normalization="proportional"
    )
    df_prop = res_prop.to_dataframe()
    assert (df_prop.to_numpy() >= 0.0).all()
    np.testing.assert_allclose(df_prop.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_marker_null_calibration():
    """Uncorrelated null marker expression yields near-zero correlation against planted proportions."""
    rng = np.random.default_rng(808)
    n_bulk = 60
    cell_types = ["CT1", "CT2", "CT3"]
    props = rng.dirichlet([2.0, 2.0, 2.0], size=n_bulk)

    genes = [f"G_{i}" for i in range(30)]
    markers = {
        "CT1": genes[0:10],
        "CT2": genes[10:20],
        "CT3": genes[20:30],
    }

    # Pure i.i.d. random noise bulk uncorrelated with props
    B_null = rng.uniform(10.0, 100.0, size=(n_bulk, len(genes)))
    res_null = marker_based_decomposition(
        B_null,
        markers,
        bulk_gene_names=genes,
        normalize_cpm=False,
    )
    df_null = res_null.to_dataframe()
    null_corrs = [abs(_pearsonr(df_null[ct].to_numpy(), props[:, k])) for k, ct in enumerate(cell_types)]
    assert max(null_corrs) < 0.35

    # Compare against planted signal with same sample size
    B_sig = rng.normal(0.0, 5.0, size=(n_bulk, len(genes)))
    for k, ct in enumerate(cell_types):
        B_sig[:, k * 10 : (k + 1) * 10] += 100.0 * props[:, k : k + 1] + 50.0
    res_sig = marker_based_decomposition(
        B_sig,
        markers,
        bulk_gene_names=genes,
        normalize_cpm=False,
    )
    sig_corrs = [
        _pearsonr(res_sig.to_dataframe()[ct].to_numpy(), props[:, k])
        for k, ct in enumerate(cell_types)
    ]
    assert min(sig_corrs) > 0.95


def test_anndata_sparse_and_dataframe_parity():
    """AnnData (dense, layer, CSR, CSC), DataFrame, and ndarray produce identical outputs."""
    sc_adata, Y_bulk_clean, _, _, genes, _, bulk_names = _simulate_reference_cohort(
        seed=909, n_donors=5, n_bulk=12, n_genes=60
    )
    rng = np.random.default_rng(909)
    bulk_counts = rng.poisson(Y_bulk_clean * 5.0).astype(np.float64)

    bulk_df = pd.DataFrame(bulk_counts, index=bulk_names, columns=genes)
    bulk_dense_ad = ad.AnnData(X=bulk_counts.copy())
    bulk_dense_ad.obs_names = bulk_names
    bulk_dense_ad.var_names = genes

    bulk_layer_ad = ad.AnnData(X=np.zeros_like(bulk_counts))
    bulk_layer_ad.obs_names = bulk_names
    bulk_layer_ad.var_names = genes
    bulk_layer_ad.layers["counts"] = sparse.csr_matrix(bulk_counts)

    bulk_csr_ad = ad.AnnData(X=sparse.csr_matrix(bulk_counts))
    bulk_csr_ad.obs_names = bulk_names
    bulk_csr_ad.var_names = genes

    bulk_csc_ad = ad.AnnData(X=sparse.csc_matrix(bulk_counts))
    bulk_csc_ad.obs_names = bulk_names
    bulk_csc_ad.var_names = genes

    sc_csr = sc_adata.copy()
    sc_csr.X = sparse.csr_matrix(sc_csr.X)
    sc_csc = sc_adata.copy()
    sc_csc.X = sparse.csc_matrix(sc_csc.X)
    sc_df = pd.DataFrame(sc_adata.X, index=sc_adata.obs_names, columns=genes)

    ref_base = reference_based_decomposition(bulk_dense_ad, sc_adata)
    ref_df = reference_based_decomposition(
        bulk_df,
        sc_df,
        cell_types=sc_adata.obs["cell_type"].to_numpy(),
        subject_names=sc_adata.obs["subject_id"].to_numpy(),
    )
    ref_arr = reference_based_decomposition(
        bulk_counts,
        sc_adata.X,
        cell_types=sc_adata.obs["cell_type"].to_numpy(),
        subject_names=sc_adata.obs["subject_id"].to_numpy(),
        bulk_sample_names=bulk_names,
        bulk_gene_names=genes,
        sc_gene_names=genes,
    )
    ref_csr = reference_based_decomposition(bulk_csr_ad, sc_csr)
    ref_csc = reference_based_decomposition(bulk_csc_ad, sc_csc)
    ref_layer = reference_based_decomposition(
        bulk_layer_ad, sc_csr, bulk_layer="counts"
    )

    for other in (ref_df, ref_arr, ref_csr, ref_csc, ref_layer):
        pd.testing.assert_frame_equal(
            ref_base.to_dataframe(), other.to_dataframe(), atol=1e-12, rtol=1e-12
        )

    # Marker mode parity across formats
    markers = {
        "B_cell": genes[:15],
        "Macrophage": genes[20:35],
        "T_cell": genes[40:55],
    }
    m_base = marker_based_decomposition(bulk_dense_ad, markers, normalization="proportional")
    m_df = marker_based_decomposition(bulk_df, markers, normalization="proportional")
    m_csr = marker_based_decomposition(bulk_csr_ad, markers, normalization="proportional")
    m_layer = marker_based_decomposition(
        bulk_layer_ad, markers, bulk_layer="counts", normalization="proportional"
    )
    for other in (m_df, m_csr, m_layer):
        pd.testing.assert_frame_equal(
            m_base.to_dataframe(), other.to_dataframe(), atol=1e-12, rtol=1e-12
        )


def test_edge_cases_zero_var_single_marker_and_missing_markers():
    """Zero-variance genes, unique_markers, min_gene=1, and missing-marker cell types behave cleanly."""
    sc_adata, Y_bulk_clean, bulk_props_true, _, genes, _, bulk_names = (
        _simulate_reference_cohort(seed=1010, n_donors=8, n_bulk=14, n_genes=60)
    )

    # Inject all-zero and constant-across-bulk genes
    zero_genes = ["Zero_0", "Zero_1", "Const_0"]
    all_genes = genes + zero_genes
    sc_X_ext = np.hstack(
        [
            sc_adata.X,
            np.zeros((sc_adata.n_obs, 2)),
            np.full((sc_adata.n_obs, 1), 15.0),
        ]
    )
    sc_ext = ad.AnnData(X=sc_X_ext, obs=sc_adata.obs.copy())
    sc_ext.var_names = all_genes
    sc_ext.obs_names = sc_adata.obs_names

    bulk_ext = pd.DataFrame(
        np.hstack(
            [
                Y_bulk_clean,
                np.zeros((len(bulk_names), 2)),
                np.full((len(bulk_names), 1), 42.0),
            ]
        ),
        index=bulk_names,
        columns=all_genes,
    )

    res_ref = reference_based_decomposition(bulk_ext, sc_ext, normalize_cpm=False)
    for zg in zero_genes:
        assert zg not in res_ref.genes_used
    assert not np.isnan(res_ref.bulk_props).any()
    assert float(np.mean(np.abs(res_ref.bulk_props - bulk_props_true))) < 0.05

    # Marker mode: insufficient markers -> UserWarning + NaN column, plus unique_markers filtering
    markers_mixed = {
        "B_cell": genes[:10] + ["Shared_G"],
        "Macrophage": genes[20:30] + ["Shared_G"],
        "MissingCT": ["NotInBulk_1", "NotInBulk_2", "Const_0"],
    }
    # Add Shared_G to bulk_ext
    bulk_ext["Shared_G"] = Y_bulk_clean[:, 0]
    with pytest.warns(UserWarning, match="MissingCT"):
        res_m = marker_based_decomposition(
            bulk_ext,
            markers_mixed,
            min_gene=5,
            unique_markers=True,
            normalize_cpm=False,
            normalization="proportional",
        )
    df_m = res_m.to_dataframe()
    assert "Shared_G" not in res_m.markers_used["B_cell"]
    assert "Shared_G" not in res_m.markers_used["Macrophage"]
    assert df_m["MissingCT"].isna().all()
    assert np.isfinite(df_m[["B_cell", "Macrophage"]].to_numpy()).all()
    np.testing.assert_allclose(
        df_m[["B_cell", "Macrophage"]].sum(axis=1).to_numpy(), 1.0, atol=1e-12
    )

    # Single-marker (min_gene=1) works without SVD crash
    res_single_m = marker_based_decomposition(
        bulk_ext,
        {"B_cell": [genes[0]], "Macrophage": [genes[20]]},
        min_gene=1,
        normalize_cpm=False,
    )
    assert np.isfinite(res_single_m.bulk_props).all()
    assert res_single_m.variance_explained["B_cell"] == pytest.approx(1.0)


def test_boundary_single_donor_overlap_fallback_and_errors():
    """Cover D=1 and N_b=1 fallbacks, overlap=0/1 behavior, zero-sum NNLS, and input errors."""
    sc_adata, Y_bulk_clean, bulk_props_true, _, genes, donor_names, bulk_names = (
        _simulate_reference_cohort(seed=1111, n_donors=6, n_bulk=10, n_genes=60)
    )

    # 1. Single-donor sc (D == 1) falls back to untransformed NNLS
    sc_one_donor = sc_adata[sc_adata.obs["subject_id"] == donor_names[0]].copy()
    with pytest.warns(UserWarning, match="fallback"):
        res_d1 = reference_based_decomposition(
            pd.DataFrame(Y_bulk_clean, index=bulk_names, columns=genes),
            sc_one_donor,
            normalize_cpm=False,
        )
    assert res_d1.mode == "untransformed_fallback"
    assert res_d1.used_fallback is True
    np.testing.assert_allclose(res_d1.bulk_props.sum(axis=1), 1.0, atol=1e-12)
    assert float(np.mean(np.abs(res_d1.bulk_props - bulk_props_true))) < 0.05

    # 2. Single-sample bulk (1D ndarray -> N_b == 1) falls back cleanly
    with pytest.warns(UserWarning, match="fallback"):
        res_n1 = reference_based_decomposition(
            Y_bulk_clean[0, :],
            sc_adata,
            bulk_gene_names=genes,
            normalize_cpm=False,
        )
    assert res_n1.mode == "untransformed_fallback"
    assert res_n1.bulk_props.shape == (1, 3)
    np.testing.assert_allclose(res_n1.bulk_props.sum(axis=1), 1.0, atol=1e-12)

    # 3. use_overlap=True with 0 overlapping IDs raises ValueError
    bulk_df = pd.DataFrame(Y_bulk_clean, index=bulk_names, columns=genes)
    with pytest.raises(ValueError, match="0 overlapping"):
        reference_based_decomposition(bulk_df, sc_adata, use_overlap=True)

    # 4. use_overlap=True with 1 overlapping ID warns and falls back to non_overlapping
    overlap_one_names = [donor_names[0]] + bulk_names[1:]
    bulk_one_ov = pd.DataFrame(Y_bulk_clean, index=overlap_one_names, columns=genes)
    with pytest.warns(UserWarning, match="Only 1 overlapping"):
        res_ov1 = reference_based_decomposition(
            bulk_one_ov, sc_adata, use_overlap=True, normalize_cpm=False
        )
    assert res_ov1.mode == "non_overlapping"
    assert res_ov1.overlapping_subjects == [donor_names[0]]

    # 5. Degenerate zero-sum NNLS sample produces warning + NaN row rather than ZeroDivisionError
    bulk_with_zero = bulk_df.copy()
    bulk_with_zero.iloc[0, :] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_z_fb = reference_based_decomposition(
            bulk_with_zero, sc_one_donor, normalize_cpm=True
        )
    msgs = [str(w.message) for w in caught]
    assert any("fallback" in m for m in msgs)
    assert any("zero-sum weights" in m for m in msgs)
    assert np.isnan(res_z_fb.bulk_props[0, :]).all()
    assert np.isfinite(res_z_fb.bulk_props[1:, :]).all()

    # 6. subject_names=None on DataFrame treats all cells as single donor (D=1 fallback)
    sc_df = pd.DataFrame(sc_adata.X, index=sc_adata.obs_names, columns=genes)
    with pytest.warns(UserWarning, match="fallback"):
        res_subj_none = reference_based_decomposition(
            bulk_df,
            sc_df,
            cell_types=sc_adata.obs["cell_type"].to_numpy(),
            subject_names=None,
            normalize_cpm=False,
        )
    assert res_subj_none.mode == "untransformed_fallback"
    assert res_subj_none.sc_subject_names == ["subject_0"]

    # 7. Missing (NaN) single-cell annotations are dropped with warning
    ct_with_nan = sc_adata.obs["cell_type"].astype(object).to_numpy().copy()
    ct_with_nan[:3] = np.nan
    with pytest.warns(UserWarning, match="Dropping 3 single cells"):
        res_nan_ct = reference_based_decomposition(
            bulk_df,
            sc_df,
            cell_types=ct_with_nan,
            subject_names=sc_adata.obs["subject_id"].to_numpy(),
            normalize_cpm=False,
        )
    assert np.isfinite(res_nan_ct.bulk_props).all()

    # 8. max_gene caps markers per cell type
    res_max = marker_based_decomposition(
        bulk_df,
        {"B_cell": genes[:15], "Macrophage": genes[20:35]},
        min_gene=3,
        max_gene=6,
        normalize_cpm=False,
    )
    assert len(res_max.markers_used["B_cell"]) == 6
    assert len(res_max.markers_used["Macrophage"]) == 6

    # 9. Validation errors on malformed inputs
    bad_nan = bulk_df.copy()
    bad_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        reference_based_decomposition(bad_nan, sc_adata)

    bad_neg = bulk_df.copy()
    bad_neg.iloc[0, 0] = -5.0
    with pytest.raises(ValueError, match="negative"):
        reference_based_decomposition(bad_neg, sc_adata, normalize_cpm=True)

    with pytest.raises(ValueError, match="not found"):
        reference_based_decomposition(bulk_df, sc_adata, cell_types="missing_col")

    disjoint_bulk = pd.DataFrame(
        Y_bulk_clean, index=bulk_names, columns=[f"Other_{i}" for i in range(len(genes))]
    )
    with pytest.raises(ValueError, match="No common genes"):
        reference_based_decomposition(disjoint_bulk, sc_adata)

    with pytest.raises(ValueError, match="min_gene"):
        marker_based_decomposition(bulk_df, {"B_cell": genes[:5]}, min_gene=0)

    with pytest.raises(ValueError, match="> 0"):
        marker_based_decomposition(
            bulk_df,
            {"B_cell": genes[:6]},
            weights={"B_cell": {g: -1.0 for g in genes[:6]}},
            weighted_pca=True,
        )


def test_settings_r_default_simplex_qp_overlap_remaining_and_old_cpm():
    """
    Verify `settings="R_default"` vs `settings="default"` across `bisque`:
    1. Strict `ValueError` on unknown `settings`.
    2. Exact equality-and-inequality-constrained simplex QP (`limSolve::lsei`) matches
       closed-form KKT optimum `(70/164, 94/164)` where `nnls` + division gives `(3/23, 20/23)`.
    3. Default `old_cpm=True` and `use_overlap=True` (when overlapping subject IDs exist),
       returning deconvolved proportions strictly on `remaining` non-overlapping bulk samples
       and dropping `anyNA` zero-variance training genes.
    4. `marker_based_decomposition(..., settings="R_default")` defaults to `minmax` + weighted PCA.
    """
    from bisque.reference import _solve_simplex_qp

    sc_adata, Y_bulk_clean, bulk_props_true, _, genes, donor_names, bulk_names = (
        _simulate_reference_cohort(seed=2026, n_donors=12, n_bulk=14, n_genes=60)
    )
    bulk_df = pd.DataFrame(Y_bulk_clean, index=bulk_names, columns=genes)

    # 1. Invalid settings raises ValueError
    with pytest.raises(ValueError, match="settings"):
        reference_based_decomposition(bulk_df, sc_adata, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        marker_based_decomposition(bulk_df, {"B_cell": genes[:8]}, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        generate_sc_reference(sc_adata, settings="bad_mode")  # type: ignore[arg-type]

    # 2. Analytical simplex QP vs unconstrained NNLS + post-hoc normalization
    Z_test = np.array([[10.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    b_test = np.array([5.0, 2.0], dtype=np.float64)
    p_qp, rnorm_qp = _solve_simplex_qp(Z_test.T @ Z_test, Z_test, b_test)
    q_nnls, _ = nnls(Z_test, b_test)
    p_nnls = q_nnls / np.sum(q_nnls)
    np.testing.assert_allclose(p_qp, np.array([70.0 / 164.0, 94.0 / 164.0]), atol=1e-10)
    np.testing.assert_allclose(p_nnls, np.array([0.3 / 2.3, 2.0 / 2.3]), atol=1e-10)
    assert np.isclose(rnorm_qp, float(np.linalg.norm(Z_test @ p_qp - b_test)), atol=1e-12)

    # Single-donor untransformed check where Z columns have asymmetric norms so QP != NNLS/sum
    sc_1 = ad.AnnData(
        X=np.array([[10.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64),
    )
    sc_1.obs_names = ["C0", "C1"]
    sc_1.var_names = ["G0", "G1", "G2"]
    sc_1.obs["cell_type"] = ["CT_A", "CT_B"]
    sc_1.obs["subject_id"] = ["D0", "D0"]
    bulk_1 = pd.DataFrame([[5.0, 2.0, 1.5]], index=["S0"], columns=["G0", "G1", "G2"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r_qp = reference_based_decomposition(
            bulk_1, sc_1, normalize_cpm=False, settings="R_default"
        )
        r_nnls = reference_based_decomposition(
            bulk_1, sc_1, normalize_cpm=False, settings="default"
        )
    np.testing.assert_allclose(r_qp.bulk_props.sum(axis=1), 1.0, atol=1e-12)
    assert not np.allclose(r_qp.bulk_props, r_nnls.bulk_props, atol=1e-2)

    # 3. Default old_cpm=True and automatic use_overlap=True returning `remaining` bulk samples
    #    and dropping anyNA zero-variance training genes
    rng = np.random.default_rng(2026)
    sc_ref = generate_sc_reference(sc_adata, normalize_cpm=False)
    n_ov = 8
    ov_ids = donor_names[:n_ov]
    rem_ids = [f"Rem_{i:02d}" for i in range(10)]
    rem_props = rng.dirichlet([1.8, 1.8, 1.8], size=10)
    all_props = np.vstack([sc_ref.sc_props[:n_ov], rem_props])
    B_ov = all_props @ sc_ref.signature_matrix.T
    # Force Gene_000 to be 0.0 on all overlapping bulk samples X_train (remains zero-variance after CPM)
    # while varying across remaining samples so global variance > 0
    B_ov[:n_ov, 0] = 0.0
    B_ov[n_ov:, 0] = np.linspace(10.0, 200.0, 10)
    # Add an extra unshared bulk-only gene with moderate variation to test default old_cpm=True
    extra_col = np.concatenate([np.full(n_ov, 50.0), np.linspace(100.0, 600.0, 10)])[:, None]
    bulk_ov_df = pd.DataFrame(
        np.hstack([B_ov, extra_col]),
        index=ov_ids + rem_ids,
        columns=genes + ["BulkOnly_Extra"],
    )

    res_r_ov = reference_based_decomposition(bulk_ov_df, sc_adata, settings="R_default")
    res_r_explicit = reference_based_decomposition(
        bulk_ov_df, sc_adata, use_overlap=True, old_cpm=True, settings="R_default"
    )
    res_def_ov = reference_based_decomposition(
        bulk_ov_df, sc_adata, use_overlap=True, old_cpm=False, settings="default"
    )

    assert res_r_ov.mode == "overlapping"
    assert res_r_ov.sample_names == rem_ids
    assert len(res_def_ov.sample_names) == n_ov + 10
    assert "Gene_000" not in res_r_ov.genes_used
    assert "Gene_000" in res_def_ov.genes_used
    pd.testing.assert_frame_equal(res_r_ov.to_dataframe(), res_r_explicit.to_dataframe())
    np.testing.assert_allclose(res_r_ov.bulk_props.sum(axis=1), 1.0, atol=1e-12)
    assert float(np.mean(np.abs(res_r_ov.bulk_props - rem_props))) < 0.05

    # 4. marker_based_decomposition under settings="R_default": defaults to minmax + weighted PCA
    m_rows = [
        {"cluster": "B_cell", "gene": g, "avg_logFC": 2.5 if j < 4 else 1.2}
        for j, g in enumerate(genes[:8])
    ] + [
        {"cluster": "Macrophage", "gene": g, "avg_logFC": 2.0}
        for g in genes[20:28]
    ]
    m_df = pd.DataFrame(m_rows)
    m_r = marker_based_decomposition(bulk_df, m_df, settings="R_default")
    m_def = marker_based_decomposition(bulk_df, m_df, settings="default")
    assert m_r.normalization == "minmax"
    assert m_r.weighted_pca is True
    assert m_def.normalization == "none"
    assert m_def.weighted_pca is False
    np.testing.assert_allclose(m_r.bulk_props.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(m_r.bulk_props.max(axis=0), 1.0, atol=1e-12)

