"""Unit tests for the standalone `cside` package."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import gammaln

from cside import (
    CSIDEConfig,
    CSIDEResult,
    build_spline_design,
    estimate_sigma_gene,
    poisson_lognormal_moments,
    run_cside,
    run_cside_nonparam,
    run_cside_regions,
    run_cside_single,
)


def _simulate_spatial_mixture(
    n_spots: int = 180,
    seed: int = 42,
    sigma_low: float = 0.15,
    sigma_high: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate spatial spots with 2 deconvolved cell types and known DE."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n_spots, 2))
    x_cov = coords[:, 0] / 100.0  # continuous gradient in [0, 1]

    # Dirichlet cell-type proportions across 2 cell types (Tumor, Stroma)
    weights = rng.dirichlet([1.5, 1.5], size=n_spots)
    n_umi = rng.integers(1500, 3500, size=n_spots).astype(np.float64)

    # 6 genes:
    # gene_0: strong +1.2 logFC ONLY in Tumor, 0.0 in Stroma (low sigma)
    # gene_1: strong -1.1 logFC ONLY in Stroma, 0.0 in Tumor (low sigma)
    # gene_2: null in both cell types (low sigma)
    # gene_3: null in both cell types (high overdispersion sigma_high)
    # gene_4: +0.9 logFC in Tumor, +0.0 in Stroma (moderate sigma)
    # gene_5: ultra-low expression gene (filtered out by gene_threshold in R_default)
    n_genes = 6
    sigmas = [sigma_low, sigma_low, sigma_low, sigma_high, 0.25, 0.10]
    # alpha[gene] shape (2, 2): rows=[intercept, slope], cols=[Tumor, Stroma]
    alphas = [
        np.array([[-4.2, -4.5], [1.25, 0.0]]),
        np.array([[-4.3, -3.8], [0.0, -1.15]]),
        np.array([[-4.0, -4.0], [0.0, 0.0]]),
        np.array([[-3.8, -3.9], [0.0, 0.0]]),
        np.array([[-4.1, -4.4], [0.95, 0.0]]),
        np.array([[-11.5, -11.5], [0.0, 0.0]]),
    ]

    design = np.column_stack([np.ones(n_spots), x_cov])
    counts = np.zeros((n_spots, n_genes), dtype=np.float64)
    for g in range(n_genes):
        eta = design @ alphas[g]
        rate = np.sum(weights * np.exp(eta), axis=1)
        eps = rng.normal(0.0, sigmas[g], size=n_spots)
        counts[:, g] = rng.poisson(n_umi * rate * np.exp(eps))

    spot_names = [f"spot_{i}" for i in range(n_spots)]
    gene_names = [f"gene_{g}" for g in range(n_genes)]
    counts_df = pd.DataFrame(counts, index=spot_names, columns=gene_names)
    weights_df = pd.DataFrame(weights, index=spot_names, columns=["Tumor", "Stroma"])
    return counts_df, weights_df, x_cov, n_umi, coords


def test_invalid_settings_raises() -> None:
    counts_df, weights_df, x_cov, _, coords = _simulate_spatial_mixture(n_spots=60)
    with pytest.raises(ValueError, match="Unknown settings"):
        CSIDEConfig(settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        build_spline_design(coords, df=6, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_cside(counts_df, weights_df, x_cov, settings="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_cside_single(counts_df, weights_df, x_cov, settings="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_cside_regions(counts_df, weights_df, ["A"] * 30 + ["B"] * 30, settings="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        run_cside_nonparam(counts_df, weights_df, coords=coords, settings="wrong")  # type: ignore[arg-type]


def test_config_default_vs_r_default_parameters() -> None:
    cfg_def = CSIDEConfig(settings="default")
    assert cfg_def.cell_type_threshold == 15.0
    assert cfg_def.weight_threshold == 0.75
    assert cfg_def.gene_threshold == 1e-5
    assert cfg_def.fdr == 0.05
    assert cfg_def.log_fc_thresh == 0.25

    cfg_r_full = CSIDEConfig(settings="R_default", doublet_mode=False)
    assert cfg_r_full.cell_type_threshold == 125.0
    assert cfg_r_full.weight_threshold == 0.8
    assert cfg_r_full.gene_threshold == 5e-5
    assert cfg_r_full.fdr == 0.01
    assert cfg_r_full.log_fc_thresh == 0.4

    cfg_r_doublet = CSIDEConfig(settings="R_default", doublet_mode=True)
    assert cfg_r_doublet.weight_threshold == 0.99

    cfg_override = CSIDEConfig(settings="R_default", cell_type_threshold=20.0, fdr=0.05)
    assert cfg_override.cell_type_threshold == 20.0
    assert cfg_override.fdr == 0.05


def test_poisson_lognormal_moments_exact_derivatives() -> None:
    y = np.array([0.0, 3.0, 18.0, 120.0])
    lam = np.array([0.8, 2.5, 10.0, 45.0])
    lf = gammaln(y + 1.0)

    for sigma in [0.05, 0.4, 1.1]:
        ll, d1, q = poisson_lognormal_moments(y, lam, lf, sigma)
        eps = 1e-5
        ll_plus, _, _ = poisson_lognormal_moments(y, lam * np.exp(eps), lf, sigma)
        ll_minus, _, _ = poisson_lognormal_moments(y, lam * np.exp(-eps), lf, sigma)

        fd_d1 = (ll_plus - ll_minus) / (2.0 * eps)
        fd_q = -(ll_plus - 2.0 * ll + ll_minus) / (eps * eps)

        np.testing.assert_allclose(d1, fd_d1, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(q, fd_q, rtol=1e-3, atol=1e-4)
        assert np.all(q > 0.0)


def test_estimate_sigma_gene_distinguishes_dispersion() -> None:
    rng = np.random.default_rng(123)
    n = 400
    lam = np.full(n, 25.0)
    y_low = rng.poisson(lam * np.exp(rng.normal(0.0, 0.10, size=n))).astype(np.float64)
    y_high = rng.poisson(lam * np.exp(rng.normal(0.0, 0.85, size=n))).astype(np.float64)

    sig_low = estimate_sigma_gene(y_low, lam, gammaln(y_low + 1.0))
    sig_high = estimate_sigma_gene(y_high, lam, gammaln(y_high + 1.0))

    assert sig_low < 0.25
    assert 0.65 < sig_high < 1.05


def test_build_spline_design_thin_plate_and_bspline() -> None:
    rng = np.random.default_rng(7)
    coords = pd.DataFrame(
        rng.uniform(-50.0, 50.0, size=(120, 2)),
        index=[f"s{i}" for i in range(120)],
        columns=["x", "y"],
    )
    tps = build_spline_design(coords, df=10, basis="thin_plate", settings="default")
    assert tps.shape == (120, 10)
    assert list(tps.index) == list(coords.index)
    np.testing.assert_allclose(tps["Intercept"].to_numpy(), 1.0)
    # Non-intercept columns are zero-mean and span 1.0
    np.testing.assert_allclose(tps.iloc[:, 1:].mean(axis=0).to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.ptp(tps.iloc[:, 1:].to_numpy(), axis=0), 1.0, atol=1e-10)

    tps_r = build_spline_design(coords, df=10, basis="thin_plate", settings="R_default")
    assert tps_r.shape == (120, 10)

    bsp = build_spline_design(coords, df=9, basis="bspline", settings="default")
    assert bsp.shape == (120, 9)
    np.testing.assert_allclose(bsp["Intercept"].to_numpy(), 1.0)


def test_build_spline_design_large_spot_knot_subsampling() -> None:
    rng = np.random.default_rng(99)
    coords = rng.uniform(0.0, 1.0, size=(650, 2))
    tps = build_spline_design(coords, df=12, basis="thin_plate", settings="default")
    assert tps.shape == (650, 12)
    assert np.all(np.isfinite(tps.to_numpy()))


def test_run_cside_single_cell_type_specific_recovery() -> None:
    counts_df, weights_df, x_cov, n_umi, _ = _simulate_spatial_mixture(n_spots=220, seed=11)
    res = run_cside_single(
        counts_df,
        cell_type_weights=weights_df,
        explanatory_variable=x_cov,
        n_umi=n_umi,
        settings="default",
    )
    assert isinstance(res, CSIDEResult)
    tumor_df = res.all_gene_list["Tumor"]
    stroma_df = res.all_gene_list["Stroma"]

    # gene_0 is strongly positive in Tumor ONLY
    assert tumor_df.loc["gene_0", "significant"]
    assert 0.85 < tumor_df.loc["gene_0", "log_fc"] < 1.65
    assert not stroma_df.loc["gene_0", "significant"]

    # gene_1 is strongly negative in Stroma ONLY
    assert stroma_df.loc["gene_1", "significant"]
    assert -1.55 < stroma_df.loc["gene_1", "log_fc"] < -0.75
    assert not tumor_df.loc["gene_1", "significant"]

    # Null genes remain non-significant despite overdispersion
    assert not tumor_df.loc["gene_2", "significant"]
    assert not stroma_df.loc["gene_2", "significant"]
    assert not tumor_df.loc["gene_3", "significant"]
    assert not stroma_df.loc["gene_3", "significant"]

    # Overdispersed gene_3 has substantially higher estimated sigma_g than gene_2
    assert res.sigma_gene["gene_3"] > res.sigma_gene["gene_2"] + 0.25


def test_run_cside_r_default_parity_and_overrides() -> None:
    counts_df, weights_df, x_cov, n_umi, _ = _simulate_spatial_mixture(n_spots=300, seed=21)
    # With 300 spots and balanced Dirichlet([1.5, 1.5]), each cell type has ~150 total weight >= 125
    res_r = run_cside_single(
        counts_df,
        cell_type_weights=weights_df,
        explanatory_variable=x_cov,
        n_umi=n_umi,
        settings="R_default",
    )
    assert res_r.settings == "R_default"
    assert res_r.params["cell_type_threshold"] == 125.0
    assert res_r.params["fdr"] == 0.01
    assert res_r.params["log_fc_thresh"] == 0.4
    # Ultra-low gene_5 is filtered out by R_default gene_threshold=5e-5
    assert "gene_5" not in res_r.all_gene_list["Tumor"].index
    assert res_r.all_gene_list["Tumor"].loc["gene_0", "significant"]

    # On 80 spots (each cell type weight ~40 < 125), R_default raises unless overridden
    c80, w80, x80, u80, _ = _simulate_spatial_mixture(n_spots=80, seed=21)
    with pytest.raises(ValueError, match="cell_type_threshold"):
        run_cside_single(c80, w80, x80, n_umi=u80, settings="R_default")

    res_override = run_cside_single(
        c80, w80, x80, n_umi=u80, cell_type_threshold=20.0, settings="R_default"
    )
    assert set(res_override.cell_types) == {"Tumor", "Stroma"}


def test_singular_covariate_ridge_vs_r_default_test_error() -> None:
    """Verify default stabilizes singular cell-type covariates while R_default flags test_error."""
    rng = np.random.default_rng(55)
    n_spots = 100
    # CellType B appears with varying weights where x_cov == 0, but is completely
    # absent where x_cov == 1 (so TypeB slope is singular while TypeA is identifiable!)
    x_cov = np.array([0.0] * 60 + [1.0] * 40)
    w_a = np.concatenate([rng.uniform(0.25, 0.85, size=60), np.ones(40)])
    w_b = 1.0 - w_a
    weights_df = pd.DataFrame({"TypeA": w_a, "TypeB": w_b})
    n_umi = np.full(n_spots, 2000.0)

    rate_a = np.exp(-4.0 + 1.1 * x_cov)
    rate_b = np.exp(-4.2)
    y0 = rng.poisson(n_umi * (w_a * rate_a + w_b * rate_b))
    y1 = rng.poisson(n_umi * 0.015)
    counts_df = pd.DataFrame({"g_de": y0, "g_null": y1}, dtype=np.float64)

    res_def = run_cside(
        counts_df,
        weights_df,
        x_cov,
        n_umi=n_umi,
        cell_type_threshold=10.0,
        settings="default",
    )
    # Default ridge keeps TypeB finite and non-NaN
    assert np.all(np.isfinite(res_def.all_gene_list["TypeB"]["se"].to_numpy()))
    assert res_def.all_gene_list["TypeA"].loc["g_de", "significant"]

    res_r = run_cside(
        counts_df,
        weights_df,
        x_cov,
        n_umi=n_umi,
        cell_type_threshold=10.0,
        settings="R_default",
    )
    # R_default strict Hessian check flags TypeB slope as unconverged (NaN p_val)
    assert not res_r.all_gene_list["TypeB"].loc["g_de", "converged"]
    assert np.isnan(res_r.all_gene_list["TypeB"].loc["g_de", "p_val"])
    # Meanwhile TypeA remains converged and significant!
    assert res_r.all_gene_list["TypeA"].loc["g_de", "converged"]
    assert res_r.all_gene_list["TypeA"].loc["g_de", "significant"]


def test_run_cside_regions_multi_region() -> None:
    rng = np.random.default_rng(77)
    n_spots = 180
    regions = np.array(["Cortex"] * 60 + ["Hippocampus"] * 60 + ["Thalamus"] * 60)
    weights = rng.dirichlet([2.0, 2.0], size=n_spots)
    weights_df = pd.DataFrame(weights, columns=["Neuron", "Glia"])
    n_umi = np.full(n_spots, 2500.0)

    # Neuron specifically upregulated in Hippocampus (+1.3 logFC)
    log_rate_neuron = np.where(regions == "Hippocampus", -3.4, -4.7)
    log_rate_glia = np.full(n_spots, -4.4)
    y_reg = rng.poisson(
        n_umi * (weights[:, 0] * np.exp(log_rate_neuron) + weights[:, 1] * np.exp(log_rate_glia))
    )
    y_null = rng.poisson(n_umi * np.exp(-4.2))
    counts_df = pd.DataFrame({"Calb1": y_reg, "Actb": y_null}, dtype=np.float64)

    res = run_cside_regions(
        counts_df,
        weights_df,
        region_labels=regions,
        n_umi=n_umi,
        settings="default",
    )
    assert res.all_gene_list["Neuron"].loc["Calb1", "significant"]
    assert not res.all_gene_list["Glia"].loc["Calb1", "significant"]
    assert not res.all_gene_list["Neuron"].loc["Actb", "significant"]


def test_run_cside_nonparam_recovers_nonlinear_hotspot() -> None:
    rng = np.random.default_rng(101)
    n_spots = 200
    coords = rng.uniform(-1.0, 1.0, size=(n_spots, 2))
    r2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    hotspot = np.exp(-3.0 * r2)  # radial Gaussian dome at origin

    weights = rng.dirichlet([2.0, 2.0], size=n_spots)
    weights_df = pd.DataFrame(weights, columns=["Epithelial", "Fibroblast"])
    n_umi = np.full(n_spots, 3000.0)

    # Hotspot gene active only in Epithelial cells
    y_hot = rng.poisson(
        n_umi
        * (
            weights[:, 0] * np.exp(-4.5 + 1.5 * hotspot)
            + weights[:, 1] * np.exp(-4.4)
        )
    )
    y_flat = rng.poisson(n_umi * np.exp(-4.3))
    counts_df = pd.DataFrame({"HotspotGene": y_hot, "FlatGene": y_flat}, dtype=np.float64)

    res = run_cside_nonparam(
        counts_df,
        weights_df,
        coords=coords,
        df=8,
        basis="thin_plate",
        n_umi=n_umi,
        settings="default",
    )
    assert res.all_gene_list["Epithelial"].loc["HotspotGene", "significant"]
    assert not res.all_gene_list["Fibroblast"].loc["HotspotGene", "significant"]
    assert not res.all_gene_list["Epithelial"].loc["FlatGene", "significant"]


def test_anndata_and_sparse_csr_support() -> None:
    counts_df, weights_df, x_cov, n_umi, coords = _simulate_spatial_mixture(
        n_spots=160, seed=33
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts_df.to_numpy(dtype=np.float64)),
        obs=pd.DataFrame(
            {"n_umi": n_umi, "gradient": x_cov},
            index=counts_df.index,
        ),
        var=pd.DataFrame(index=counts_df.columns),
    )
    adata.obsm["spatial"] = coords
    adata.obsm["cell_type_weights"] = weights_df

    res_single = run_cside_single(
        adata, explanatory_variable=adata.obs["gradient"], settings="default"
    )
    assert res_single.all_gene_list["Tumor"].loc["gene_0", "significant"]

    res_np = run_cside_nonparam(adata, df=7, basis="bspline", settings="default")
    assert res_np.all_gene_list["Tumor"].loc["gene_0", "significant"]


def test_summary_and_to_dataframe_exports() -> None:
    counts_df, weights_df, x_cov, n_umi, _ = _simulate_spatial_mixture(n_spots=150, seed=19)
    res = run_cside_single(counts_df, weights_df, x_cov, n_umi=n_umi, settings="default")
    text = res.summary()
    assert "CSIDEResult" in text
    assert "Tumor" in text and "Stroma" in text

    df_all = res.to_dataframe(significant_only=False)
    df_sig = res.to_dataframe(significant_only=True)
    expected_cols = {
        "cell_type",
        "gene",
        "log_fc",
        "se",
        "z_stat",
        "p_val",
        "q_val",
        "significant",
        "sigma_gene",
    }
    assert expected_cols.issubset(df_all.columns)
    assert len(df_sig) < len(df_all)
    assert df_sig["significant"].all()
