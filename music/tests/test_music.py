import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from music import (
    music2_prop,
    music_basis,
    music_iter,
    music_prop,
    music_prop_cluster,
)


def _make_synthetic_sc_and_bulk(seed=42, sparse=False):
    rng = np.random.default_rng(seed)
    n_genes = 120
    gene_names = [f"Gene_{g}" for g in range(n_genes)]

    # Distinct cell-type expression profiles for 3 cell types across 4 donors
    ct_names = ["B_cell", "Macrophage", "T_cell"]
    base_profiles = np.ones((3, n_genes), dtype=float)
    base_profiles[0, 0:35] = 25.0
    base_profiles[1, 35:70] = 25.0
    base_profiles[2, 70:105] = 25.0
    # Gene_119 is all-zero everywhere to test non_zero=True filtering
    base_profiles[:, 119] = 0.0
    base_theta = base_profiles / base_profiles.sum(axis=1, keepdims=True)
    base_theta[:, 119] = 0.0

    cell_sizes = {"B_cell": 1500.0, "Macrophage": 3000.0, "T_cell": 2000.0}

    rows = []
    obs_ct = []
    obs_sample = []
    for s_idx in range(4):
        sid = f"Donor_{s_idx}"
        for c_idx, ct in enumerate(ct_names):
            # Mild subject-level variation
            theta_sc = base_theta[c_idx].copy()
            noise = rng.lognormal(mean=0.0, sigma=0.08, size=n_genes)
            noise[119] = 0.0
            theta_sc = theta_sc * noise
            theta_sc /= theta_sc.sum()
            for _ in range(25):
                lam = cell_sizes[ct] * theta_sc
                counts = rng.poisson(lam).astype(float)
                counts[119] = 0.0
                rows.append(counts)
                obs_ct.append(ct)
                obs_sample.append(sid)

    sc_X = np.vstack(rows)
    if sparse:
        sc_X = csr_matrix(sc_X)

    sc_adata = ad.AnnData(
        X=sc_X,
        obs=pd.DataFrame(
            {"cell_type": obs_ct, "subject_id": obs_sample},
            index=[f"Cell_{i}" for i in range(len(obs_ct))],
        ),
    )
    sc_adata.var_names = gene_names

    # Construct 3 bulk samples with known cell-type proportions
    true_props = np.array(
        [
            [0.60, 0.25, 0.15],
            [0.15, 0.55, 0.30],
            [0.25, 0.20, 0.55],
        ]
    )
    S_vec = np.array([cell_sizes[ct] for ct in ct_names])
    D_true = base_theta.T * S_vec  # (n_genes, 3)

    bulk_rows = []
    for i in range(true_props.shape[0]):
        expected = 50.0 * (D_true @ true_props[i])
        counts = rng.poisson(expected).astype(float)
        counts[119] = 0.0
        bulk_rows.append(counts)

    bulk_X = np.vstack(bulk_rows)
    if sparse:
        bulk_X = csr_matrix(bulk_X)

    bulk_adata = ad.AnnData(X=bulk_X)
    bulk_adata.var_names = gene_names
    bulk_adata.obs_names = [f"Bulk_{i}" for i in range(true_props.shape[0])]

    return sc_adata, bulk_adata, pd.DataFrame(
        true_props, index=bulk_adata.obs_names, columns=ct_names
    )


def test_music_basis_zero_gene_and_zero_donor_cluster_nan():
    sc_adata, _, _ = _make_synthetic_sc_and_bulk(seed=10)

    # Inject a donor-cluster cell group with 0 total counts across all genes:
    # must be treated as NaN (NOT 0.0) so M_S and M_theta are not biased toward 0!
    X_dense = sc_adata.X.copy()
    zero_cell = np.zeros((3, X_dense.shape[1]), dtype=float)
    X_aug = np.vstack([X_dense, zero_cell])
    obs_aug = pd.concat(
        [
            sc_adata.obs,
            pd.DataFrame(
                {"cell_type": ["B_cell"] * 3, "subject_id": ["Donor_Zero"] * 3},
                index=[f"ZeroCell_{i}" for i in range(3)],
            ),
        ],
    )
    sc_aug = ad.AnnData(X=X_aug, obs=obs_aug)
    sc_aug.var_names = sc_adata.var_names

    basis_orig = music_basis(
        sc_adata, "cell_type", "subject_id", ct_cov=True, non_zero=True
    )
    basis_aug = music_basis(
        sc_aug, "cell_type", "subject_id", ct_cov=True, non_zero=True
    )

    # All-zero Gene_119 filtered out
    assert "Gene_119" not in basis_aug["Design_mtx"].index
    # Donor_Zero B_cell library size must be NaN, leaving M_S and M_theta unchanged
    assert np.isnan(basis_aug["S"].loc["Donor_Zero", "B_cell"])
    np.testing.assert_allclose(
        basis_aug["M_S"].values, basis_orig["M_S"].values, rtol=1e-12
    )
    np.testing.assert_allclose(
        basis_aug["M_theta"].values, basis_orig["M_theta"].values, rtol=1e-12
    )
    assert basis_aug["Sigma_ct"].shape == (
        len(basis_aug["Design_mtx"]),
        3,
        3,
    )


def test_music_iter_irls_multistep_convergence_and_downweighting():
    """
    Verify music_iter normalizes p_weight to sum to 1 inside every iteration even
    when library sizes S are large (e.g. 20,000, so raw q ~ 0.005 < eps=0.01),
    and downweights high-Sigma corrupted genes over unweighted NNLS.
    """
    rng = np.random.default_rng(7)
    n_genes = 80
    theta = np.zeros((n_genes, 2), dtype=float)
    theta[:40, 0] = rng.uniform(0.5, 1.5, size=40)
    theta[40:, 1] = rng.uniform(0.5, 1.5, size=40)
    # First 10 genes are shared/corrupted with huge cross-subject variance
    theta[:10, :] = 2.0
    theta /= theta.sum(axis=0, keepdims=True)

    S = np.array([20000.0, 20000.0])  # Large S so raw NNLS sum(q) ~ 100 / 20000 = 0.005
    D = theta * S

    true_p = np.array([0.70, 0.30])
    Y_clean = D @ true_p
    # Corrupt the high-Sigma genes (0..9) in bulk sample toward cell type 2
    Y_corrupted = Y_clean.copy()
    Y_corrupted[:10] *= 4.0
    Y_ra = Y_corrupted / Y_corrupted.sum()

    Sigma = np.full((n_genes, 2), 1e-7, dtype=float)
    Sigma[:10, :] = 0.05  # High cross-subject variance on corrupted genes

    p_weight_0, p_nnls, _, w_gene = music_iter(
        Y_ra, D, S, Sigma, iter_max=0, eps=1e-5
    )
    p_weight_full, _, r2, _ = music_iter(
        Y_ra, D, S, Sigma, iter_max=200, eps=1e-5
    )

    assert np.sum(p_weight_full) == pytest.approx(1.0)
    assert np.sum(p_nnls) == pytest.approx(1.0)
    # Corrupted genes receive much smaller weights than clean marker genes
    assert np.mean(w_gene[:10]) < 0.05 * np.mean(w_gene[10:])
    # Weighted IRLS beats unweighted NNLS on recovering true_p = [0.70, 0.30]
    err_weighted = np.sum(np.abs(p_weight_full - true_p))
    err_nnls = np.sum(np.abs(p_nnls - true_p))
    assert err_weighted < err_nnls
    assert err_weighted < 0.08
    # Confirm IRLS took multiple steps beyond iter_max=0 when eps=1e-5
    assert not np.allclose(p_weight_full, p_weight_0, atol=1e-6)
    assert np.isfinite(r2)


def test_music_prop_sparse_and_single_donor_cluster():
    sc_adata, bulk_adata, true_df = _make_synthetic_sc_and_bulk(seed=21, sparse=True)
    assert isinstance(sc_adata.X, csr_matrix)
    assert isinstance(bulk_adata.X, csr_matrix)

    res = music_prop(
        bulk_adata=bulk_adata,
        sc_adata=sc_adata,
        cluster_col="cell_type",
        sample_col="subject_id",
        ct_cov=True,
    )
    est = res["Est_prop_weighted"]
    assert list(est.index) == list(bulk_adata.obs_names)
    assert list(est.columns) == list(true_df.columns)
    np.testing.assert_allclose(est.sum(axis=1).values, 1.0, atol=1e-6)
    np.testing.assert_allclose(est.values, true_df.values, atol=0.06)

    # Edge case: restrict Macrophage to a single donor so Sigma['Macrophage'] is NaN
    keep_mask = ~(
        (sc_adata.obs["cell_type"] == "Macrophage")
        & (sc_adata.obs["subject_id"] != "Donor_0")
    )
    sc_single_donor_ct = sc_adata[keep_mask].copy()
    res_single = music_prop(
        bulk_adata=bulk_adata,
        sc_adata=sc_single_donor_ct,
        cluster_col="cell_type",
        sample_col="subject_id",
    )
    est_single = res_single["Est_prop_weighted"]
    assert "Macrophage" in est_single.columns
    assert not est_single.isna().any().any()
    np.testing.assert_allclose(est_single.values, true_df.values, atol=0.08)


def test_music_prop_cluster_hierarchical_and_single_donor_subcluster():
    rng = np.random.default_rng(99)
    n_genes = 120
    genes = [f"G_{i}" for i in range(n_genes)]

    # 2 major groups: Lymphoid -> [CD4_T, CD8_T], Myeloid -> [Macrophage]
    # Genes 0..39 mark Lymphoid, 40..79 mark Myeloid;
    # Genes 80..99 distinguish CD4_T vs CD8_T within Lymphoid!
    sub_cts = ["CD4_T", "CD8_T", "Macrophage"]
    ct_to_group = {"CD4_T": "Lymphoid", "CD8_T": "Lymphoid", "Macrophage": "Myeloid"}

    profiles = np.ones((3, n_genes), dtype=float)
    profiles[0, 0:40] = 20.0
    profiles[1, 0:40] = 20.0
    profiles[2, 40:80] = 25.0
    profiles[0, 80:95] = 35.0   # CD4_T specific markers
    profiles[1, 95:110] = 35.0  # CD8_T specific markers
    theta = profiles / profiles.sum(axis=1, keepdims=True)

    rows, obs_g, obs_c, obs_s = [], [], [], []
    for d_idx in range(3):
        sid = f"Subj_{d_idx}"
        for c_idx, ct in enumerate(sub_cts):
            # Make CD8_T present in ONLY Subj_0 (single-donor subcluster -> NaN Sigma!)
            if ct == "CD8_T" and d_idx > 0:
                continue
            for _ in range(25):
                rows.append(rng.poisson(2000.0 * theta[c_idx]).astype(float))
                obs_c.append(ct)
                obs_g.append(ct_to_group[ct])
                obs_s.append(sid)

    sc_adata = ad.AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame(
            {"group": obs_g, "cell_type": obs_c, "subject_id": obs_s},
            index=[f"Cell_{i}" for i in range(len(obs_c))],
        ),
    )
    sc_adata.var_names = genes

    true_p = np.array([[0.45, 0.25, 0.30], [0.15, 0.35, 0.50]])
    bulk_X = np.vstack(
        [rng.poisson(80.0 * 2000.0 * (theta.T @ p)).astype(float) for p in true_p]
    )
    bulk_adata = ad.AnnData(X=bulk_X)
    bulk_adata.var_names = genes
    bulk_adata.obs_names = ["Sample_A", "Sample_B"]

    clusters_type = {"Lymphoid": ["CD4_T", "CD8_T"], "Myeloid": ["Macrophage"]}
    group_markers = {
        "Lymphoid": [f"G_{i}" for i in range(80, 110)],
        "Myeloid": [],
    }

    res = music_prop_cluster(
        bulk_adata=bulk_adata,
        sc_adata=sc_adata,
        group_markers=group_markers,
        groups_col="group",
        clusters_col="cell_type",
        samples_col="subject_id",
        clusters_type=clusters_type,
    )
    est = res["Est_prop_weighted_cluster"]
    assert list(est.columns) == ["CD4_T", "CD8_T", "Macrophage"]
    assert not est.isna().any().any()
    np.testing.assert_allclose(est.sum(axis=1).values, 1.0, atol=1e-6)
    np.testing.assert_allclose(est.values, true_p, atol=0.10)


def test_music2_prop_sample_order_de_genes_and_single_case():
    sc_adata, _, _ = _make_synthetic_sc_and_bulk(seed=55)
    rng = np.random.default_rng(55)
    genes = list(sc_adata.var_names)
    n_genes = len(genes)

    basis = music_basis(sc_adata, "cell_type", "subject_id")
    D_full = np.zeros((n_genes, 3), dtype=float)
    idx_nz = [genes.index(g) for g in basis["Design_mtx"].index]
    D_full[idx_nz, :] = basis["Design_mtx"].values

    # Non-alphabetical sample names to verify exact sample order preservation
    ctrl_names = [f"Ctrl_{tag}" for tag in ["Z", "A", "M", "B", "K", "C", "X", "D", "Q", "E", "W", "F"]]
    case_names = [f"Case_{tag}" for tag in ["9", "2", "7", "1", "5", "8", "3", "6", "0", "4", "11", "10"]]

    ctrl_props = rng.dirichlet([2.5, 2.5, 2.5], size=len(ctrl_names))
    case_props = rng.dirichlet([3.0, 2.0, 2.5], size=len(case_names))

    ctrl_X = np.vstack(
        [rng.poisson(60.0 * (D_full @ p)).astype(float) for p in ctrl_props]
    )

    # Spike strong case-specific DE into Gene_0, Gene_1, Gene_2
    spiked_de = ["Gene_0", "Gene_1", "Gene_2"]
    D_case = D_full.copy()
    for g_name in spiked_de:
        D_case[genes.index(g_name), 0] *= 25.0

    case_X = np.vstack(
        [rng.poisson(60.0 * (D_case @ p)).astype(float) for p in case_props]
    )

    bulk_ctrl = ad.AnnData(X=ctrl_X)
    bulk_ctrl.var_names = genes
    bulk_ctrl.obs_names = ctrl_names

    bulk_case = ad.AnnData(X=case_X)
    bulk_case.var_names = genes
    bulk_case.obs_names = case_names

    res = music2_prop(
        bulk_control_adata=bulk_ctrl,
        bulk_case_adata=bulk_case,
        sc_adata=sc_adata,
        cluster_col="cell_type",
        sample_col="subject_id",
        n_resample=15,
        maxiter=25,
        seed=123,
    )

    est_prop = res["Est_prop"]
    # Exact sample order preservation: ctrl_names followed by case_names
    assert list(est_prop.index) == ctrl_names + case_names
    assert "DE_genes" in res
    # Spiked strong DE genes should be detected
    assert len(set(spiked_de).intersection(res["DE_genes"])) >= 2
    np.testing.assert_allclose(
        est_prop.loc[case_names].values, case_props, atol=0.10
    )

    # Small sample edge case: n_case == 1 must not crash!
    single_case = bulk_case[["Case_9"], :].copy()
    res_single_case = music2_prop(
        bulk_control_adata=bulk_ctrl,
        bulk_case_adata=single_case,
        sc_adata=sc_adata,
        cluster_col="cell_type",
        sample_col="subject_id",
        n_resample=5,
        maxiter=5,
        seed=123,
    )
    assert list(res_single_case["Est_prop"].index) == ctrl_names + ["Case_9"]
    assert not res_single_case["Est_prop"].isna().any().any()
