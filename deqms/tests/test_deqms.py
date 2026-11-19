"""Unit tests for the standalone deqms package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import deqms
from deqms import (
    DEqMS,
    DEqMSFit,
    LimmaFit,
    ProteinSummarizationResult,
    equal_median_normalization,
    lm_fit,
    median_sweeping,
    output_result,
    peptide_profile_summarization,
    spectra_count_ebayes,
)


def _simulate_count_dependent_proteomics(
    seed: int = 123,
    n_proteins: int = 400,
    n_per_group: int = 4,
    n_de: int = 30,
) -> tuple[pd.DataFrame, pd.Series, list[str], np.ndarray]:
    """Simulate proteomics log2 abundances where true variance scales as a + b / count."""
    rng = np.random.default_rng(seed)
    counts = rng.choice(
        [1, 2, 3, 4, 6, 10, 16, 25, 40],
        size=n_proteins,
        p=[0.24, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03],
    )
    true_s02 = 0.015 + 0.35 / counts
    d0_true = 8.0
    chi2_draws = rng.chisquare(df=d0_true, size=n_proteins)
    sigma2 = (d0_true * true_s02) / chi2_draws

    n_samples = 2 * n_per_group
    sample_names = [f"ctrl_{i+1}" for i in range(n_per_group)] + [
        f"treat_{i+1}" for i in range(n_per_group)
    ]
    design = ["ctrl"] * n_per_group + ["treat"] * n_per_group

    y = rng.normal(loc=0.0, scale=np.sqrt(sigma2)[:, np.newaxis], size=(n_proteins, n_samples))
    de_indices = np.arange(n_de)
    # Give true DE proteins realistic fold changes across high-count and low-count proteins
    signs = rng.choice([-1.0, 1.0], size=n_de)
    y[de_indices, n_per_group:] += (signs * 1.1)[:, np.newaxis]

    prot_ids = [f"P{i:04d}" for i in range(n_proteins)]
    expr_df = pd.DataFrame(y, index=prot_ids, columns=sample_names)
    count_series = pd.Series(counts, index=prot_ids, name="count", dtype=np.int64)
    return expr_df, count_series, design, de_indices


def test_settings_validation_all_public_apis() -> None:
    """Every primary function and class must reject unknown settings strings."""
    df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [2.0, 3.0]}, index=["P1", "P2"])
    psm_df = pd.DataFrame(
        {"protein": ["P1", "P1", "P2"], "s1": [10.0, 10.5, 12.0], "s2": [11.0, 11.5, 9.0]}
    )

    with pytest.raises(ValueError, match="Unknown settings"):
        equal_median_normalization(df, settings="bad_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        median_sweeping(psm_df, settings="bad_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        peptide_profile_summarization(psm_df, settings="bad_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        lm_fit(df, design=[0, 1], settings="bad_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        spectra_count_ebayes(df, design=[0, 1], count=[2, 5], settings="bad_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        DEqMS(settings="bad_mode")  # type: ignore[arg-type]

    expr_sim, cnt_sim, design_sim, _ = _simulate_count_dependent_proteomics(n_proteins=40)
    valid_fit = spectra_count_ebayes(expr_sim, design=design_sim, count=cnt_sim)
    with pytest.raises(ValueError, match="Unknown settings"):
        output_result(valid_fit, settings="bad_mode")  # type: ignore[arg-type]


def test_equal_median_normalization_dataframe_and_ndarray() -> None:
    """Column medians should be zero when center_zero=True and equal when False."""
    raw = np.array(
        [[10.0, 22.0, 5.0], [12.0, 25.0, 8.0], [14.0, 28.0, 11.0], [20.0, np.nan, 15.0]]
    )
    df = pd.DataFrame(raw, index=["A", "B", "C", "D"], columns=["S1", "S2", "S3"])

    norm_df = equal_median_normalization(df, center_zero=True, settings="R_default")
    assert isinstance(norm_df, pd.DataFrame)
    assert list(norm_df.index) == ["A", "B", "C", "D"]
    np.testing.assert_allclose(np.nanmedian(norm_df.to_numpy(), axis=0), [0.0, 0.0, 0.0], atol=1e-12)

    norm_arr = equal_median_normalization(raw, center_zero=False, settings="default")
    assert isinstance(norm_arr, np.ndarray)
    meds = np.nanmedian(norm_arr, axis=0)
    np.testing.assert_allclose(meds, np.full(3, meds[0]), atol=1e-12)
    assert meds[0] != 0.0


def test_median_sweeping_tmt_raw_and_reference_channel() -> None:
    """median_sweeping recovers true fold-changes and exact PSM counts."""
    rng = np.random.default_rng(7)
    rows = []
    # P_DE has 8 PSMs with +1.5 log2 fold change in treat vs ctrl
    for _ in range(8):
        base = rng.uniform(14.0, 18.0)
        rows.append(
            {
                "protein": "P_DE",
                "ref": 2.0**base,
                "c1": 2.0 ** (base + rng.normal(0, 0.05)),
                "c2": 2.0 ** (base + rng.normal(0, 0.05)),
                "t1": 2.0 ** (base + 1.5 + rng.normal(0, 0.05)),
                "t2": 2.0 ** (base + 1.5 + rng.normal(0, 0.05)),
            }
        )
    # 25 background proteins with 2..5 PSMs and 0 fold change so column medians reflect background
    for p_idx in range(25):
        n_psm = int(rng.integers(2, 6))
        for _ in range(n_psm):
            base = rng.uniform(13.0, 19.0)
            rows.append(
                {
                    "protein": f"BG_{p_idx}",
                    "ref": 2.0**base,
                    "c1": 2.0 ** (base + rng.normal(0, 0.05)),
                    "c2": 2.0 ** (base + rng.normal(0, 0.05)),
                    "t1": 2.0 ** (base + rng.normal(0, 0.05)),
                    "t2": 2.0 ** (base + rng.normal(0, 0.05)),
                }
            )

    psm_df = pd.DataFrame(rows)
    res = median_sweeping(
        psm_df,
        group_col="protein",
        sample_cols=["c1", "c2", "t1", "t2"],
        ref_cols="ref",
        log_scaled=False,
        settings="default",
    )
    assert isinstance(res, ProteinSummarizationResult)
    abund_df, count_ser = res
    assert count_ser.loc["P_DE"] == 8
    assert abund_df.shape == (26, 4)
    np.testing.assert_allclose(np.nanmedian(abund_df.to_numpy(), axis=0), 0.0, atol=1e-12)

    est_lfc = abund_df.loc["P_DE", ["t1", "t2"]].mean() - abund_df.loc["P_DE", ["c1", "c2"]].mean()
    assert abs(est_lfc - 1.5) < 0.12


def test_peptide_profile_summarization_tukey_median_polish_with_nans() -> None:
    """Tukey median polish removes peptide ionization offsets and preserves all-NaN sample cells."""
    # Additive model: peptide_effect + sample_effect, with an outlier cell and an all-NaN sample
    pep_effects = np.array([-3.0, 0.0, 2.5, 5.0])
    sample_profile = np.array([20.0, 20.2, 22.0, 22.3, np.nan])
    mat = pep_effects[:, np.newaxis] + sample_profile[np.newaxis, :]
    mat[0, 2] += 12.0  # single outlier spike in one peptide-sample cell
    mat[2, 0] = np.nan  # partial missingness

    df_pep = pd.DataFrame(
        mat, columns=["s1", "s2", "s3", "s4", "s5"]
    )
    df_pep.insert(0, "protein", "PROT_A")
    df_pep2 = pd.DataFrame(
        {
            "protein": ["PROT_B", "PROT_B"],
            "s1": [18.0, 19.0],
            "s2": [18.2, 19.2],
            "s3": [18.1, 19.1],
            "s4": [18.3, 19.3],
            "s5": [18.0, 19.0],
        }
    )
    combined = pd.concat([df_pep, df_pep2], ignore_index=True)

    res = peptide_profile_summarization(
        combined, group_col="protein", method="tukey_median_polish", settings="default"
    )
    assert res.count.to_dict() == {"PROT_A": 4, "PROT_B": 2}
    assert np.isnan(res.abundance.loc["PROT_A", "s5"])
    # Relative sample differences (s3 - s1) should be unaffected by the single peptide outlier!
    diff_31 = res.abundance.loc["PROT_A", "s3"] - res.abundance.loc["PROT_A", "s1"]
    assert abs(diff_31 - 2.0) < 1e-6


def test_variance_trend_monotonic_shrinkage_with_psm_count() -> None:
    """Prior variance sca_priorvar decreases strongly with increasing peptide/PSM count."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(seed=42, n_proteins=500)
    fit = spectra_count_ebayes(
        expr_df, design=design, count=counts, fit_method="loess", settings="R_default"
    )
    assert isinstance(fit, DEqMSFit)
    assert fit.sca_dfprior > 2.0

    c1_var = float(np.mean(fit.sca_priorvar[counts.to_numpy() == 1]))
    c10_var = float(np.mean(fit.sca_priorvar[counts.to_numpy() == 10]))
    c40_var = float(np.mean(fit.sca_priorvar[counts.to_numpy() == 40]))
    assert c1_var > c10_var > c40_var
    assert c1_var > 3.0 * c40_var


@pytest.mark.parametrize("fit_method", ["loess", "spline", "nls"])
def test_all_three_fit_methods_agree_on_trend(fit_method: str) -> None:
    """loess, natural cubic spline (df=4), and nls (a + b/count) all capture count-variance trend."""
    expr_df, counts, design, de_idx = _simulate_count_dependent_proteomics(seed=99, n_proteins=350)
    fit = spectra_count_ebayes(
        expr_df, design=design, count=counts, fit_method=fit_method, settings="default"
    )
    res_df = output_result(fit, sort_by="sca.P.Value")
    assert res_df.shape == (350, 11)
    assert np.all(np.isfinite(fit.sca_priorvar))
    # Correlation of log prior variance across methods with -log2(count) should be strongly negative
    corr = np.corrcoef(np.log2(counts.to_numpy()), np.log(fit.sca_priorvar))[0, 1]
    assert corr < -0.85

    top30 = set(res_df.index[:30])
    true_de_names = {f"P{i:04d}" for i in de_idx}
    assert len(top30 & true_de_names) >= 18


def test_deqms_improves_sensitivity_for_high_count_proteins_and_controls_one_hit_wonders() -> None:
    """DEqMS gives sharper significance than untrended limma on multi-peptide proteins and penalizes 1-PSM noise."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(seed=2024, n_proteins=500, n_de=0)

    # Inject identical moderate fold-change (0.45) and identical small sample residual variance
    # into one 1-PSM protein and one 30-PSM protein
    expr_mod = expr_df.copy()
    counts_mod = counts.copy()
    counts_mod.iloc[0] = 1
    counts_mod.iloc[1] = 30
    template_ctrl = np.array([-0.06, 0.04, -0.02, 0.04])
    template_treat = template_ctrl + 0.45
    expr_mod.iloc[0] = np.concatenate([template_ctrl, template_treat])
    expr_mod.iloc[1] = np.concatenate([template_ctrl, template_treat])

    fit = spectra_count_ebayes(expr_mod, design=design, count=counts_mod, settings="R_default")
    tbl = output_result(fit, sort_by="none", settings="R_default")

    # Standard limma treats row 0 (1 PSM) and row 1 (30 PSMs) identically!
    assert abs(tbl.loc["P0000", "t"] - tbl.loc["P0001", "t"]) < 1e-12
    assert abs(tbl.loc["P0000", "P.Value"] - tbl.loc["P0001", "P.Value"]) < 1e-12

    # DEqMS shrinks 1-PSM protein toward larger 1-PSM prior variance and 30-PSM toward smaller prior variance!
    assert tbl.loc["P0001", "sca.t"] > tbl.loc["P0000", "sca.t"] * 1.5
    assert tbl.loc["P0001", "sca.P.Value"] < tbl.loc["P0000", "sca.P.Value"] * 0.05


def test_r_default_schema_and_nonrobust_vs_default_robust_outliers() -> None:
    """default (robust=True) protects sca_dfprior and outlier posterior variance against hypervariable proteins."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(seed=55, n_proteins=300, n_de=0)
    clean_r = spectra_count_ebayes(expr_df, design=design, count=counts, settings="R_default")

    # Contaminate 8 proteins with massive outlier noise (sd = 15)
    contaminated = expr_df.copy()
    rng = np.random.default_rng(55)
    contaminated.iloc[:8] = rng.normal(0.0, 15.0, size=(8, expr_df.shape[1]))

    fit_r = spectra_count_ebayes(contaminated, design=design, count=counts, settings="R_default")
    fit_def = spectra_count_ebayes(contaminated, design=design, count=counts, settings="default")

    # R_default scalar prior df collapses due to extreme variance inflation; default robust stays stable!
    assert fit_r.robust is False
    assert fit_def.robust is True
    assert fit_r.sca_dfprior < 0.5 * clean_r.sca_dfprior
    assert fit_def.sca_dfprior > 1.8 * fit_r.sca_dfprior

    # Outlier rows get downweighted per-protein prior df in default mode
    assert np.median(fit_def.sca_dfprior_vec[:8]) < 0.25 * fit_def.sca_dfprior
    np.testing.assert_allclose(fit_r.sca_dfprior_vec, fit_r.sca_dfprior)

    # Exact 11-column Bioconductor DEqMS::outputResult schema
    out_r = output_result(fit_r, settings="R_default")
    assert list(out_r.columns) == [
        "logFC",
        "AveExpr",
        "t",
        "P.Value",
        "adj.P.Val",
        "B",
        "gene",
        "count",
        "sca.t",
        "sca.P.Value",
        "sca.adj.pval",
    ]


def test_missing_values_per_protein_ols_and_residual_df() -> None:
    """Partially missing NaN protein rows get exact per-protein QR/OLS and residual degrees of freedom."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(
        seed=77, n_proteins=120, n_per_group=4
    )
    expr_nan = expr_df.copy()
    # Protein 0: 1 missing in ctrl, 1 missing in treat -> N_obs = 6, df_resid = 4 (vs 6 for complete)
    expr_nan.iloc[0, 0] = np.nan
    expr_nan.iloc[0, 5] = np.nan
    # Protein 1: completely missing in all treat samples -> contrast non-estimable -> NaN output
    expr_nan.iloc[1, 4:] = np.nan
    # Protein 2: only 1 observation in ctrl and 1 in treat -> df_resid = 0 -> NaN variance/p-value
    expr_nan.iloc[2, 1:4] = np.nan
    expr_nan.iloc[2, 5:] = np.nan

    fit = spectra_count_ebayes(expr_nan, design=design, count=counts, settings="default")
    assert fit.df_residual[0] == 4.0
    assert fit.df_residual[3] == 6.0
    assert np.isfinite(fit.sca_p[0, fit.default_coef])
    assert np.isnan(fit.sca_p[1, fit.default_coef])
    assert np.isnan(fit.sca_p[2, fit.default_coef])

    expected_lfc_0 = np.nanmean(expr_nan.iloc[0, 4:]) - np.nanmean(expr_nan.iloc[0, :4])
    assert abs(fit.coefficients[0, fit.default_coef] - expected_lfc_0) < 1e-12


def test_custom_design_and_contrast_matrix_multi_comparison() -> None:
    """2D design matrix + multi-column contrast matrix supports multi-group TMT comparisons."""
    rng = np.random.default_rng(314)
    n_prot = 150
    counts = rng.integers(1, 20, size=n_prot)
    # 3 groups x 3 replicates = 9 samples
    groups = ["G0"] * 3 + ["G1"] * 3 + ["G2"] * 3
    y = rng.normal(0.0, 0.25, size=(n_prot, 9))
    y[0, 3:6] += 2.0  # G1 specific upregulation
    y[1, 6:9] -= 2.2  # G2 specific downregulation

    contrast = pd.DataFrame(
        {"G1_vs_G0": [-1.0, 1.0, 0.0], "G2_vs_G0": [-1.0, 0.0, 1.0]},
        index=["G0", "G1", "G2"],
    )
    fit = spectra_count_ebayes(y, design=groups, contrast=contrast, count=counts)

    tbl_g1 = output_result(fit, coef="G1_vs_G0", sort_by="sca.P.Value")
    tbl_g2 = output_result(fit, coef="G2_vs_G0", sort_by="sca.P.Value")
    assert tbl_g1.index[0] == "protein_0"
    assert tbl_g1.loc["protein_0", "logFC"] > 1.7
    assert tbl_g2.index[0] == "protein_1"
    assert tbl_g2.loc["protein_1", "logFC"] < -1.8


def test_null_calibration_under_hierarchical_inverse_chisq_model() -> None:
    """Under exact hierarchical null, sca.P.Value is well-calibrated (neither inflated nor collapsed)."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(
        seed=808, n_proteins=800, n_per_group=3, n_de=0
    )
    fit = spectra_count_ebayes(expr_df, design=design, count=counts, settings="R_default")
    pvals = fit.sca_p[:, fit.default_coef]
    frac_005 = float(np.mean(pvals < 0.05))
    mean_p = float(np.mean(pvals))
    assert 0.025 <= frac_005 <= 0.08
    assert 0.45 <= mean_p <= 0.55
    assert 4.5 <= fit.sca_dfprior <= 14.0


def test_deqms_class_end_to_end_and_precomputed_fit_inputs() -> None:
    """DEqMS OOP wrapper and pre-fitted LimmaFit / dict inputs produce identical results."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(seed=19, n_proteins=100)

    # 1. Direct ProteinSummarizationResult input
    summary_input = ProteinSummarizationResult(abundance=expr_df, count=counts)
    model = DEqMS(fit_method="loess", span=0.75, settings="default")
    tbl_class = model.run(summary_input, design=design)
    assert model.sca_dfprior > 0
    assert model.sca_priorvar.shape == (100,)
    assert model.sca_postvar.shape == (100,)

    # 2. Pre-fitted LimmaFit input
    pre_fit = lm_fit(expr_df, design=design)
    fit_from_lm = spectra_count_ebayes(pre_fit, count=counts, settings="default")
    tbl_from_lm = fit_from_lm.to_dataframe()
    pd.testing.assert_frame_equal(tbl_class, tbl_from_lm)

    # 3. Dict fit input
    fit_dict = {
        "coefficients": pre_fit.coefficients[:, [pre_fit.default_coef]],
        "stdev_unscaled": pre_fit.stdev_unscaled[:, [pre_fit.default_coef]],
        "s2": pre_fit.s2,
        "df_residual": pre_fit.df_residual,
        "AveExpr": pre_fit.ave_expr,
        "protein_names": pre_fit.protein_names,
        "count": counts.to_numpy(),
    }
    fit_from_dict = spectra_count_ebayes(fit_dict, settings="default")
    np.testing.assert_allclose(
        fit_from_dict.sca_p[:, 0], fit_from_lm.sca_p[:, pre_fit.default_coef]
    )


def test_invalid_inputs_and_zero_count_rejected() -> None:
    """Invalid counts (<1), mismatched lengths, unknown fit_method, or unfitted DEqMS raise clear errors."""
    expr_df, counts, design, _ = _simulate_count_dependent_proteomics(seed=5, n_proteins=30)

    bad_counts = counts.copy()
    bad_counts.iloc[3] = 0
    with pytest.raises(ValueError, match=">= 1"):
        spectra_count_ebayes(expr_df, design=design, count=bad_counts, settings="R_default")

    with pytest.raises(ValueError, match="count length"):
        spectra_count_ebayes(expr_df, design=design, count=[1, 2, 3])

    with pytest.raises(ValueError, match="Unknown fit_method"):
        spectra_count_ebayes(expr_df, design=design, count=counts, fit_method="poly")

    unfitted = DEqMS()
    with pytest.raises(RuntimeError, match="must be called"):
        unfitted.output_result()

    assert hasattr(deqms, "__version__")


def test_median_sweeping_without_ref_cols_log_scaled() -> None:
    """median_sweeping with ref_cols=None uses spectrum-row median across samples."""
    psm_df = pd.DataFrame(
        {
            "protein": ["P1", "P1", "P1", "P2", "P2", "P3", "P3"],
            "s1": [10.0, 12.0, 14.0, 11.0, 13.0, 9.0, 10.0],
            "s2": [10.1, 12.1, 13.9, 11.1, 12.9, 9.1, 9.9],
            "s3": [12.0, 14.0, 16.0, 11.0, 13.0, 9.0, 10.0],
            "s4": [12.1, 14.1, 15.9, 11.1, 12.9, 9.1, 9.9],
        }
    )
    res = median_sweeping(psm_df, group_col="protein", log_scaled=True, settings="R_default")
    assert res.count.to_dict() == {"P1": 3, "P2": 2, "P3": 2}
    lfc_p1 = res.abundance.loc["P1", ["s3", "s4"]].mean() - res.abundance.loc["P1", ["s1", "s2"]].mean()
    assert abs(lfc_p1 - 2.0) < 1e-10


def test_homogeneous_variance_infinite_prior_df_and_log_odds_monotonicity() -> None:
    """When all proteins share exact constant true variance, d0 -> inf and log-odds B is finite and monotonic."""
    rng = np.random.default_rng(101)
    n_prot = 200
    counts = rng.integers(1, 30, size=n_prot)
    # Constant true variance across proteins -> sample variance of log(s2) <= trigamma(df/2) often triggers d0 -> inf
    y = rng.normal(0.0, 0.2, size=(n_prot, 8))
    y[:10, 4:] += np.linspace(0.6, 2.0, 10)[:, np.newaxis]
    fit = spectra_count_ebayes(y, design=[0, 0, 0, 0, 1, 1, 1, 1], count=counts, settings="R_default")
    assert np.all(np.isfinite(fit.b))
    assert np.all(np.isfinite(fit.sca_b))

    tbl_b = output_result(fit, sort_by="B")
    assert np.all(np.diff(tbl_b["B"].to_numpy()) <= 1e-12)
    # Stronger |sca.t| yields higher sca_b when stdev_unscaled is constant
    order_abs_t = np.argsort(np.abs(fit.sca_t[:, fit.default_coef]))
    sca_b_sorted = fit.sca_b[order_abs_t, fit.default_coef]
    assert np.all(np.diff(sca_b_sorted) >= -1e-10)
