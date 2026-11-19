"""Unit tests for the standalone `saturn` DTU package."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import polygamma

from saturn import (
    DTUFit,
    FilteredDTUData,
    estimate_empirical_null,
    filter_dtu,
    fit_dtu,
    squeeze_var,
    stagewise_dtu,
    test_dtu,
    trigamma_inverse,
)


def _simulate_dtu_dataset(
    n_per_group: int = 6,
    n_null_genes: int = 35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate overdispersed Dirichlet-Multinomial bulk DTU counts."""
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per_group
    sample_names = [f"S{i}" for i in range(n_samples)]
    group = np.array(["ctrl"] * n_per_group + ["treat"] * n_per_group)
    design_df = pd.DataFrame({"condition": group}, index=sample_names)

    tx_records: list[tuple[str, str]] = []
    cols: dict[str, np.ndarray] = {}

    # 3 true DTU genes (2 x 2-isoform, 1 x 3-isoform)
    dtu_specs = [
        ("GENE_DTU1", [0.75, 0.25], [0.20, 0.80]),
        ("GENE_DTU2", [0.20, 0.80], [0.75, 0.25]),
        ("GENE_DTU3", [0.65, 0.20, 0.15], [0.15, 0.65, 0.20]),
    ]
    for g_name, p_ctrl, p_treat in dtu_specs:
        k_iso = len(p_ctrl)
        gene_counts = np.zeros((n_samples, k_iso), dtype=np.float64)
        for i in range(n_samples):
            total_n = int(rng.integers(120, 240))
            base_p = np.array(p_ctrl if i < n_per_group else p_treat)
            alpha_dir = 25.0 * base_p
            probs = rng.dirichlet(alpha_dir)
            gene_counts[i] = rng.multinomial(total_n, probs)
        for k in range(k_iso):
            tx_id = f"{g_name}_tx{k}"
            tx_records.append((tx_id, g_name))
            cols[tx_id] = gene_counts[:, k]

    # Null multi-isoform genes (3 isoforms each, >= 100 total transcripts so locfdr activates)
    for g in range(n_null_genes):
        g_name = f"GENE_NULL{g}"
        base_p = rng.dirichlet([2.5, 2.0, 1.5])
        gene_counts = np.zeros((n_samples, 3), dtype=np.float64)
        for i in range(n_samples):
            total_n = int(rng.integers(90, 200))
            probs = rng.dirichlet(20.0 * base_p)
            gene_counts[i] = rng.multinomial(total_n, probs)
        for k in range(3):
            tx_id = f"{g_name}_tx{k}"
            tx_records.append((tx_id, g_name))
            cols[tx_id] = gene_counts[:, k]

    counts_df = pd.DataFrame(cols, index=sample_names)
    tx2gene_df = pd.DataFrame(tx_records, columns=["transcript_id", "gene_id"])
    return counts_df, tx2gene_df, design_df


def test_invalid_settings_raises() -> None:
    counts_df, tx2gene_df, design_df = _simulate_dtu_dataset(n_null_genes=4)
    with pytest.raises(ValueError, match="Unknown settings"):
        filter_dtu(counts_df, tx2gene_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        fit_dtu(counts_df, tx2gene_df, design_df, settings="bad")  # type: ignore[arg-type]
    fit = fit_dtu(counts_df, tx2gene_df, design_df, settings="default")
    with pytest.raises(ValueError, match="Unknown settings"):
        test_dtu(fit, contrast=[0, 1], settings="bad")  # type: ignore[arg-type]
    res = test_dtu(fit, contrast=[0, 1], settings="default")
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError, match="Unknown settings"):
        stagewise_dtu(res, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        squeeze_var(np.ones(5), np.full(5, 4.0), settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        estimate_empirical_null(np.zeros(10), settings="bad")  # type: ignore[arg-type]


def test_trigamma_inverse_accuracy() -> None:
    y_true = np.array([0.15, 0.8, 2.5, 12.0, 80.0])
    tri_vals = polygamma(1, y_true)
    y_rec = trigamma_inverse(tri_vals)
    np.testing.assert_allclose(y_rec, y_true, rtol=1e-9)
    assert np.isinf(trigamma_inverse(0.0))


def test_squeeze_var_default_vs_r_default() -> None:
    rng = np.random.default_rng(10)
    df = np.full(200, 8.0)
    # True scaled F(8, 12) around s0^2 = 1.6 plus 3 extreme hypervariable outliers
    s2 = 1.6 * rng.f(dfnum=8.0, dfden=12.0, size=200)
    s2[:3] = np.array([80.0, 110.0, 95.0])

    eb_r = squeeze_var(s2, df, settings="R_default")
    eb_def = squeeze_var(s2, df, settings="default")

    assert isinstance(eb_r.df_prior, float)
    assert isinstance(eb_def.df_prior, np.ndarray)
    # Robust mode preserves higher bulk df_prior against variance inflation by outliers,
    # while selectively reducing df_prior on extreme outlier transcripts
    assert eb_def.df_prior[-1] > eb_r.df_prior
    assert eb_def.df_prior[0] < eb_def.df_prior[-1]
    assert eb_def.s2_post[0] > eb_r.s2_post[0]


def test_empirical_null_fallback_and_recalibration() -> None:
    rng = np.random.default_rng(2024)
    # Small panel (<50) falls back to (0.0, 1.0) in default mode
    small_z = rng.normal(0.4, 1.5, size=30)
    mu_small, sig_small = estimate_empirical_null(small_z, settings="default")
    assert (mu_small, sig_small) == (0.0, 1.0)

    # Large panel with inflated null N(0.25, 1.40^2) + 5% extreme DTU tails
    z_null = rng.normal(0.25, 1.40, size=1900)
    z_alt = rng.choice([-6.5, 6.5], size=100)
    z_mix = np.concatenate([z_null, z_alt])

    mu_def, sig_def = estimate_empirical_null(z_mix, settings="default")
    mu_r, sig_r = estimate_empirical_null(z_mix, settings="R_default")

    assert 0.12 < mu_def < 0.38
    assert 1.25 < sig_def < 1.55
    assert 0.10 < mu_r < 0.40
    assert 1.20 < sig_r < 1.60


def test_filter_dtu_removes_low_counts_and_single_isoform_genes() -> None:
    counts = pd.DataFrame(
        {
            "g1_tx1": [30, 25, 28, 32],
            "g1_tx2": [15, 18, 20, 12],
            "g1_tx3": [1, 0, 2, 1],      # low count -> dropped
            "g2_tx1": [40, 50, 45, 38],
            "g2_tx2": [0, 2, 1, 0],      # dropped -> leaves g2 with single isoform -> g2 dropped!
            "g3_tx1": [22, 19, 25, 30],  # originally single isoform -> dropped
        },
        index=["s1", "s2", "s3", "s4"],
    )
    tx2gene = pd.DataFrame(
        {
            "transcript_id": ["g1_tx1", "g1_tx2", "g1_tx3", "g2_tx1", "g2_tx2", "g3_tx1"],
            "gene_id": ["g1", "g1", "g1", "g2", "g2", "g3"],
        }
    )
    filt = filter_dtu(counts, tx2gene, min_count=10, min_samples=3, remove_single_isoform=True)
    assert isinstance(filt, FilteredDTUData)
    assert list(filt.transcript_ids) == ["g1_tx1", "g1_tx2"]
    np.testing.assert_allclose(filt.gene_totals.ravel(), [45, 43, 48, 44])
    np.testing.assert_allclose(filt.other_counts[:, 0], [15, 18, 20, 12])

    adata_conv = filt.to_anndata()
    assert adata_conv.shape == (4, 2)


def test_all_zero_vs_all_one_switch_default_vs_r_default() -> None:
    """All-0 vs all-1 complete separation remains testable in default and flags boundary in R_default."""
    counts_df, tx2gene_df, design_df = _simulate_dtu_dataset(n_per_group=5, n_null_genes=25, seed=7)
    # Inject a complete all-0 vs all-1 isoform switch gene!
    n_samples = len(counts_df)
    counts_df["SWITCH_tx0"] = np.array([150] * 5 + [0] * 5, dtype=np.float64)
    counts_df["SWITCH_tx1"] = np.array([0] * 5 + [150] * 5, dtype=np.float64)
    tx2gene_df = pd.concat(
        [
            tx2gene_df,
            pd.DataFrame(
                {"transcript_id": ["SWITCH_tx0", "SWITCH_tx1"], "gene_id": ["GENE_SWITCH", "GENE_SWITCH"]}
            ),
        ],
        ignore_index=True,
    )
    assert len(counts_df) == n_samples

    fit_def = fit_dtu(counts_df, tx2gene_df, design_df, settings="default")
    res_def = test_dtu(fit_def, contrast="condition_treat", settings="default")
    assert isinstance(res_def, pd.DataFrame)

    # In Pythonic default, complete separation stays finite (|beta| <= 15) and highly significant!
    assert np.all(np.abs(fit_def.coefficients.loc[["SWITCH_tx0", "SWITCH_tx1"]].to_numpy()) <= 15.0)
    assert res_def.loc["SWITCH_tx0", "fdr_empirical"] < 1e-3
    assert res_def.loc["SWITCH_tx1", "fdr_empirical"] < 1e-3
    assert res_def.loc["SWITCH_tx0", "estimates"] < -4.0
    assert res_def.loc["SWITCH_tx1", "estimates"] > 4.0

    # In R_default, unpenalized quasibinomial IRLS drives |beta| > 15 and flags boundary=True
    fit_r = fit_dtu(counts_df, tx2gene_df, design_df, settings="R_default")
    assert fit_r.boundary.loc["SWITCH_tx0"]
    assert fit_r.boundary.loc["SWITCH_tx1"]


def test_bulk_dtu_end_to_end_and_r_transposed_counts() -> None:
    counts_df, tx2gene_df, design_df = _simulate_dtu_dataset(n_per_group=6, n_null_genes=35, seed=99)
    # Pass R-style (transcripts x samples) DataFrame to verify automatic orientation
    counts_r_layout = counts_df.T

    filt = filter_dtu(counts_r_layout, tx2gene_df, min_count=5, min_samples=3, settings="default")
    fit = fit_dtu(filt, design=design_df, settings="default")
    res = test_dtu(fit, contrast="condition_treat", diag_plot=True, settings="default")
    assert isinstance(res, pd.DataFrame)
    assert "diag_data" in res.attrs

    sig_genes = set(res.loc[res["fdr_empirical"] < 0.05, "gene_id"])
    assert {"GENE_DTU1", "GENE_DTU2", "GENE_DTU3"}.issubset(sig_genes)
    false_pos = {g for g in sig_genes if g.startswith("GENE_NULL")}
    assert len(false_pos) <= 1


def test_sparse_scrna_anndata_active_gene_subsetting() -> None:
    """Verify sparse AnnData scRNA-seq where many cells have n_gi == 0."""
    rng = np.random.default_rng(314)
    n_cells = 160
    cell_type = np.array(["TypeA"] * 80 + ["TypeB"] * 80)

    # 20 genes x 2 isoforms = 40 transcripts; ~55% of cells have 0 expression for each gene
    n_genes = 20
    counts = np.zeros((n_cells, 2 * n_genes), dtype=np.float64)
    gene_ids: list[str] = []
    tx_ids: list[str] = []

    for g in range(n_genes):
        g_id = f"SCG_{g}"
        tx_ids.extend([f"{g_id}_iso0", f"{g_id}_iso1"])
        gene_ids.extend([g_id, g_id])
        expressed = rng.random(n_cells) < 0.45
        for i in np.flatnonzero(expressed):
            tot = int(rng.integers(8, 25))
            if g == 0:
                p0 = 0.85 if i < 80 else 0.15
            else:
                p0 = 0.50
            c0 = rng.binomial(tot, p0)
            counts[i, 2 * g] = c0
            counts[i, 2 * g + 1] = tot - c0

    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"cell_type": cell_type}, index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame({"gene_id": gene_ids}, index=tx_ids),
    )

    filt = filter_dtu(adata, min_count=3, min_samples=10, settings="default")
    fit = fit_dtu(filt, design="cell_type", settings="default")
    # Residual df varies across genes according to expressing cell count (< n_cells - 2)
    assert (fit.df_residual < (n_cells - 2)).all()
    assert fit.df_residual.nunique() > 1

    res = test_dtu(fit, contrast="cell_type_TypeB", settings="default")
    assert isinstance(res, pd.DataFrame)
    assert res.loc["SCG_0_iso0", "fdr_empirical"] < 0.01
    assert res.loc["SCG_0_iso1", "fdr_empirical"] < 0.01


def test_multi_contrast_dict_and_matrix() -> None:
    counts_df, tx2gene_df, design_df = _simulate_dtu_dataset(n_per_group=5, n_null_genes=10, seed=12)
    fit = fit_dtu(counts_df, tx2gene_df, design_df, settings="default")

    res_dict = test_dtu(
        fit,
        contrast={"treat_vs_ctrl": [0.0, 1.0], "ctrl_vs_treat": [0.0, -1.0]},
        settings="default",
    )
    assert isinstance(res_dict, dict)
    np.testing.assert_allclose(
        res_dict["treat_vs_ctrl"]["tstat"].to_numpy(),
        -res_dict["ctrl_vs_treat"]["tstat"].to_numpy(),
    )

    c_mat = np.array([[0.0, 0.0], [1.0, -1.0]])
    res_mat = test_dtu(fit, contrast=c_mat, settings="default")
    assert isinstance(res_mat, dict) and len(res_mat) == 2


def test_stagewise_dtu_simes_and_shaffer_confirmation() -> None:
    # Construct synthetic test_res_df with a 2-isoform DTU gene, a 4-isoform DTU gene, and 18 null genes
    rows = [
        ("G2_tx1", "G2", 1e-6),
        ("G2_tx2", "G2", 2e-5),
        ("G4_tx1", "G4", 1e-7),
        ("G4_tx2", "G4", 3e-6),
        ("G4_tx3", "G4", 0.65),
        ("G4_tx4", "G4", 0.80),
    ]
    rng = np.random.default_rng(88)
    for g in range(18):
        for k in range(3):
            rows.append((f"N{g}_tx{k}", f"N{g}", float(rng.uniform(0.15, 0.99))))

    df_in = pd.DataFrame(rows, columns=["transcript_id", "gene_id", "pval_empirical"]).set_index(
        "transcript_id", drop=False
    )
    stage_df = stagewise_dtu(df_in, alpha=0.05, settings="default")

    # Both G2 transcripts confirmed
    assert stage_df.loc["G2_tx1", "gene_significant"]
    assert stage_df.loc["G2_tx1", "tx_significant"]
    assert stage_df.loc["G2_tx2", "tx_significant"]

    # For 4-isoform G4, top 2 switching transcripts confirmed while 2 unchanged remain False
    assert stage_df.loc["G4_tx1", "gene_significant"]
    assert stage_df.loc["G4_tx1", "tx_significant"]
    assert stage_df.loc["G4_tx2", "tx_significant"]
    assert not stage_df.loc["G4_tx3", "tx_significant"]
    assert not stage_df.loc["G4_tx4", "tx_significant"]

    # Null genes unscreened & unconfirmed
    assert not stage_df.loc["N0_tx0", "gene_significant"]
    assert not stage_df.loc["N0_tx0", "tx_significant"]


def test_fit_summary_and_to_dataframe() -> None:
    counts_df, tx2gene_df, design_df = _simulate_dtu_dataset(n_per_group=4, n_null_genes=6, seed=5)
    fit = fit_dtu(counts_df, tx2gene_df, design_df, settings="R_default")
    assert isinstance(fit, DTUFit)
    s = fit.summary()
    assert "DTUFit" in s and "R_default" in s
    df_fit = fit.to_dataframe()
    assert {"transcript_id", "gene_id", "dispersion", "df_residual", "converged", "boundary"}.issubset(
        df_fit.columns
    )
