import importlib
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pythonic_bayesprism import (
    Prism,
    optimize_psi_one_gamma,
    plot_bulk_vs_sc,
    plot_cor_phi,
)


def _make_synthetic_cohort(seed=42, n_genes=120, n_samples=8, tumor_props=None):
    """
    Create single-cell reference with cell types ['B', 'Macrophage', 'Tumor']
    (where 'Tumor' is LAST alphabetically!) and two distinct malignant sub-states
    ('Tumor_1', 'Tumor_2') so we can test both tumor-mode column alignment and NMF
    embedding learning.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]

    # Cell states mapped cleanly to ['B', 'Macrophage', 'Tumor']
    states = ["B_naive", "B_mem", "Macro_M1", "Macro_M2", "Tumor_1", "Tumor_2"]
    state_to_ct = {
        "B_naive": "B",
        "B_mem": "B",
        "Macro_M1": "Macrophage",
        "Macro_M2": "Macrophage",
        "Tumor_1": "Tumor",
        "Tumor_2": "Tumor",
    }

    profiles = np.ones((len(states), n_genes), dtype=float)
    # B markers: 0..29
    profiles[0, 0:18] = 30.0
    profiles[1, 12:30] = 30.0
    # Macrophage markers: 30..59
    profiles[2, 30:48] = 30.0
    profiles[3, 42:60] = 30.0
    # Tumor_1 markers: 60..89, Tumor_2 markers: 90..119
    profiles[4, 60:90] = 35.0
    profiles[5, 90:120] = 35.0
    phi_true = profiles / profiles.sum(axis=1, keepdims=True)

    ref_rows = []
    obs_ct = []
    obs_cs = []
    for s_idx, cs in enumerate(states):
        for _ in range(25):
            ref_rows.append(rng.poisson(1500.0 * phi_true[s_idx]).astype(float))
            obs_cs.append(cs)
            obs_ct.append(state_to_ct[cs])

    ref_adata = ad.AnnData(
        X=np.vstack(ref_rows),
        obs=pd.DataFrame(
            {"cell_type": obs_ct, "cell_state": obs_cs},
            index=[f"Cell_{i}" for i in range(len(obs_cs))],
        ),
    )
    ref_adata.var_names = gene_names

    # Default ground truth: 20% B, 10% Macrophage, 70% Tumor
    if tumor_props is None:
        true_ct_props = np.tile(np.array([0.20, 0.10, 0.70]), (n_samples, 1))
    else:
        true_ct_props = np.asarray(tumor_props, dtype=float)

    mix_rows = []
    for i in range(n_samples):
        p_b, p_mac, p_tum = true_ct_props[i]
        # Alternate samples between Tumor_1-dominant and Tumor_2-dominant
        if i % 2 == 0:
            w_state = np.array(
                [0.5 * p_b, 0.5 * p_b, 0.5 * p_mac, 0.5 * p_mac, 0.85 * p_tum, 0.15 * p_tum]
            )
        else:
            w_state = np.array(
                [0.5 * p_b, 0.5 * p_b, 0.5 * p_mac, 0.5 * p_mac, 0.15 * p_tum, 0.85 * p_tum]
            )
        expected = 20000.0 * (w_state @ phi_true)
        mix_rows.append(rng.poisson(expected).astype(float))

    mix_adata = ad.AnnData(X=np.vstack(mix_rows))
    mix_adata.var_names = gene_names
    mix_adata.obs_names = [f"Sample_{i}" for i in range(n_samples)]

    return ref_adata, mix_adata, phi_true, true_ct_props


def test_lazy_matplotlib_import_in_headless_env(monkeypatch):
    """Ensure importing pythonic_bayesprism does not import matplotlib or seaborn."""
    for mod in list(sys.modules.keys()):
        if mod.startswith(("matplotlib", "seaborn", "pythonic_bayesprism")):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Simulate an environment where matplotlib and seaborn are not installed
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "seaborn", None)

    mod = importlib.import_module("pythonic_bayesprism")
    assert hasattr(mod, "Prism")


def test_input_validation_nan_inf_and_negative():
    ref_adata, mix_adata, _, _ = _make_synthetic_cohort(seed=1, n_samples=2)

    bad_mix_nan = mix_adata.copy()
    bad_mix_nan.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        Prism(ref_adata, bad_mix_nan, cell_type_labels="cell_type")

    bad_ref_neg = ref_adata.copy()
    bad_ref_neg.X[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        Prism(bad_ref_neg, mix_adata, cell_type_labels="cell_type")

    with pytest.raises(ValueError, match="not found"):
        Prism(ref_adata, mix_adata, cell_type_labels="cell_type", key="NonExistent", outlier_fraction=1.0)


def test_tumor_mode_last_alphabetical_key_and_nmf_embedding():
    """
    (a) Tumor mode when key='Tumor' is LAST alphabetically (['B', 'Macrophage', 'Tumor'])
        with ground-truth 70% Tumor / 20% B / 10% Macrophage fractions — asserting Tumor
        recovers ~0.70 and is NOT scrambled into 'B'!
    (d) Also verifies learn_embedding_nmf recovers K=2 malignant programs whose sample
        weights omega sum to the Tumor fraction (~0.70).
    """
    ref_adata, mix_adata, _, _ = _make_synthetic_cohort(seed=42, n_samples=6)

    bp = Prism(
        reference=ref_adata,
        mixture=mix_adata,
        cell_type_labels="cell_type",
        cell_state_labels="cell_state",
        key="Tumor",
        outlier_fraction=1.0,
    )
    assert list(bp.ct_names) == ["B", "Macrophage", "Tumor"]

    bp.run(
        n_cores=1,
        update_gibbs=True,
        gibbs_control={
            "chain_length": 160,
            "burn_in": 80,
            "thinning": 2,
            "alpha": 1.0,
            "seed": 123,
        },
        opt_control={"maxiter": 200, "optimizer": "MAP", "sigma": 2.0},
    )

    df = bp.get_results(output_type="dataframe")
    assert list(df.columns) == ["B", "Macrophage", "Tumor"]
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-6)

    # Assert Tumor (~0.70), B (~0.20), Macrophage (~0.10) are NOT scrambled!
    np.testing.assert_allclose(df["Tumor"].values, 0.70, atol=0.06)
    np.testing.assert_allclose(df["B"].values, 0.20, atol=0.05)
    np.testing.assert_allclose(df["Macrophage"].values, 0.10, atol=0.05)

    res_adata = bp.get_results(output_type="anndata")
    np.testing.assert_allclose(res_adata.obs["Tumor"].values, df["Tumor"].values)

    # (d) Test learn_embedding_nmf with K=2 malignant programs
    emb = bp.learn_embedding_nmf(
        K=2,
        cycle=3,
        n_cores=1,
        gibbs_control={
            "chain_length": 100,
            "burn_in": 50,
            "thinning": 2,
            "alpha": 1.0,
            "seed": 123,
        },
        opt_control={"maxiter": 100, "optimizer": "MAP", "sigma": 2.0},
        nmf_control={"nrun": 5, "seed": 42},
    )
    assert set(emb.keys()) == {"eta_prior", "eta_post", "omega"}
    assert emb["eta_post"].shape == (2, len(bp.gene_names))
    np.testing.assert_allclose(emb["eta_post"].sum(axis=1), 1.0, atol=1e-6)
    assert emb["omega"].shape == (6, 2)
    # Total malignant program weight per sample matches Tumor fraction (1 - B - Macrophage ~ 0.70)
    np.testing.assert_allclose(
        emb["omega"].sum(axis=1), df["Tumor"].values, atol=0.02
    )


def test_non_tumor_mode_key_none():
    """(b) Non-tumor mode (key=None) recovers ground-truth cell-type proportions."""
    true_props = np.array(
        [
            [0.50, 0.30, 0.20],
            [0.15, 0.60, 0.25],
            [0.25, 0.20, 0.55],
        ]
    )
    ref_adata, mix_adata, _, _ = _make_synthetic_cohort(
        seed=7, n_samples=3, tumor_props=true_props
    )

    bp = Prism(
        reference=ref_adata,
        mixture=mix_adata,
        cell_type_labels="cell_type",
        cell_state_labels="cell_state",
        key=None,
        outlier_fraction=1.0,
    )
    bp.run(
        n_cores=1,
        update_gibbs=True,
        gibbs_control={
            "chain_length": 160,
            "burn_in": 80,
            "thinning": 2,
            "alpha": 1.0,
            "seed": 42,
        },
        opt_control={"maxiter": 200, "optimizer": "MAP", "sigma": 2.0},
    )

    df = bp.get_results(output_type="dataframe")
    np.testing.assert_allclose(df.values, true_props, atol=0.06)


def test_spatial_run_st_shared_gamma_mle():
    """
    (c) Spatial run_st (optimizer='MLE' by default) using optimize_psi_one_gamma
        under a shared platform gene-scaling shift across all cell states.
    """
    rng = np.random.default_rng(19)
    true_props = np.array(
        [
            [0.60, 0.25, 0.15],
            [0.20, 0.55, 0.25],
            [0.15, 0.20, 0.65],
            [0.40, 0.40, 0.20],
        ]
    )
    ref_adata, mix_adata, phi_true, _ = _make_synthetic_cohort(
        seed=19, n_samples=4, tumor_props=true_props
    )

    # Apply a shared spatial platform capture efficiency log-shift gamma across genes
    gamma_shift = rng.normal(0.0, 0.6, size=phi_true.shape[1])
    phi_shifted = phi_true * np.exp(gamma_shift)[np.newaxis, :]
    phi_shifted /= phi_shifted.sum(axis=1, keepdims=True)

    st_rows = []
    for p_b, p_mac, p_tum in true_props:
        w_state = np.array(
            [0.5 * p_b, 0.5 * p_b, 0.5 * p_mac, 0.5 * p_mac, 0.5 * p_tum, 0.5 * p_tum]
        )
        st_rows.append(rng.poisson(5000.0 * (w_state @ phi_shifted)).astype(float))
    # Set one gene completely to zero across all ST spots to test zero-count stability
    st_X = np.vstack(st_rows)
    st_X[:, 0] = 0.0
    st_adata = ad.AnnData(X=st_X)
    st_adata.var_names = mix_adata.var_names
    st_adata.obs_names = [f"Spot_{i}" for i in range(len(true_props))]

    bp_st = Prism(
        reference=ref_adata,
        mixture=st_adata,
        cell_type_labels="cell_type",
        cell_state_labels="cell_state",
        key=None,
        outlier_fraction=1.0,
    )
    bp_st.run_st(
        n_cores=1,
        gibbs_control={
            "chain_length": 160,
            "burn_in": 80,
            "thinning": 2,
            "alpha": 1.0,
            "seed": 99,
        },
        opt_control={"maxiter": 300},  # Relies on run_st default optimizer='MLE'
    )

    df_st = bp_st.get_results(output_type="dataframe")
    assert not df_st.isna().any().any()
    np.testing.assert_allclose(df_st.sum(axis=1).values, 1.0, atol=1e-6)
    np.testing.assert_allclose(df_st.values, true_props, atol=0.08)

    # Verify reference_update rows sum to 1 and get_exp returns valid N x G expression
    np.testing.assert_allclose(bp_st.reference_update.sum(axis=1), 1.0, atol=1e-6)
    z_b = bp_st.get_exp("type", "B")
    assert z_b.shape == (4, len(bp_st.gene_names))

    # Direct unit test on optimize_psi_one_gamma recovering exact known gamma scaling
    Z_gt_exact = (10000.0 * phi_shifted).T  # (G, T)
    psi_est = optimize_psi_one_gamma(
        phi_true, Z_gt_exact, {"maxiter": 500}, pseudo_min=1e-8
    )
    np.testing.assert_allclose(psi_est, phi_shifted, atol=1e-4)


def test_plots_min_exp_and_identity_line(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_adata, mix_adata, _, _ = _make_synthetic_cohort(seed=3, n_samples=3)
    bp = Prism(
        reference=ref_adata,
        mixture=mix_adata,
        cell_type_labels="cell_type",
        cell_state_labels="cell_state",
        key=None,
        outlier_fraction=1.0,
    )

    grid = plot_cor_phi(bp, min_exp=3, show=False)
    assert grid is not None
    plt.close("all")

    fig, ax = plot_bulk_vs_sc(bp, show=False)
    lines = ax.get_lines()
    assert len(lines) >= 1
    xdata, ydata = lines[0].get_xdata(), lines[0].get_ydata()
    # True y = x identity line must have identical x and y endpoints!
    np.testing.assert_allclose(xdata, ydata, rtol=1e-12)
    plt.close(fig)
