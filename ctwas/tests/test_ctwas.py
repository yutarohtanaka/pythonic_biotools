"""Comprehensive unit tests for cTWAS (impute_expr_z, Empirical Bayes priors, group-prior SuSiE-RSS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctwas import (
    CTWAS,
    estimate_ctwas_priors,
    impute_expr_z,
    regularize_ld_matrix,
    run_ctwas,
    susie_rss_group_prior,
)


def _ar1_ld(p: int, rho: float = 0.6) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _simulate_ctwas_regions(
    n_regions: int = 24,
    n_snps_per_reg: int = 40,
    n_samples: int = 40000,
    seed: int = 42,
):
    """Simulate multi-region GWAS + cis-eQTL weights with enriched causal genes and pleiotropic SNPs."""
    rng = np.random.default_rng(seed)
    r_snp = _ar1_ld(n_snps_per_reg, rho=0.55)
    chol = np.linalg.cholesky(r_snp)
    snp_ids = [f"rs{s}" for s in range(n_snps_per_reg)]

    regions = []
    causal_genes: list[str] = []
    causal_snps: list[tuple[str, str]] = []

    for r_idx in range(n_regions):
        reg_id = f"reg_{r_idx + 1}"
        # 2 cis-genes per region anchored around SNPs 6..8 and SNPs 26..28
        w1 = np.zeros(n_snps_per_reg)
        w1[[5, 6, 7]] = [0.35, 0.75, 0.30]
        w2 = np.zeros(n_snps_per_reg)
        w2[[25, 26, 27]] = [0.30, 0.80, -0.25]

        g1_name = f"GENE_{r_idx + 1}_A"
        g2_name = f"GENE_{r_idx + 1}_B"
        weights = [
            {"gene_id": g1_name, "snps": snp_ids, "weights": w1},
            {"gene_id": g2_name, "snps": snp_ids, "weights": w2},
        ]

        # Normalize expression weights so Var(X w_g) = 1
        w1_norm = w1 / np.sqrt(w1 @ r_snp @ w1)
        w2_norm = w2 / np.sqrt(w2 @ r_snp @ w2)

        # Noise z ~ N(0, R_snp)
        z_noise = chol @ rng.normal(size=n_snps_per_reg)
        mean_z = np.zeros(n_snps_per_reg)

        if r_idx < 10:
            # Causal gene G1 with PVE = 0.002 -> sqrt(n * PVE) = sqrt(80) ~ 8.94
            sign = 1.0 if r_idx % 2 == 0 else -1.0
            gamma_g = sign * np.sqrt(n_samples * 0.002)
            mean_z += gamma_g * (r_snp @ w1_norm)
            causal_genes.append(g1_name)
        elif r_idx in (10, 11):
            # Direct non-coding pleiotropic SNP at index 18 (not an eQTL weight SNP)
            gamma_s = np.sqrt(n_samples * 0.0015)
            mean_z += gamma_s * r_snp[:, 18]
            causal_snps.append((reg_id, "rs18"))

        z_obs = mean_z + z_noise
        gwas_df = pd.DataFrame({"snp": snp_ids, "z": z_obs})
        imp = impute_expr_z(gwas_df, r_snp, weights, region_id=reg_id)
        regions.append(imp)

    return regions, causal_genes, causal_snps


def test_settings_validation_all_entrypoints() -> None:
    r_snp = _ar1_ld(10)
    gwas_df = pd.DataFrame({"snp": [f"rs{i}" for i in range(10)], "z": np.zeros(10)})
    weights = [{"gene_id": "G1", "snps": ["rs1", "rs2"], "weights": [0.5, 0.5]}]
    imp = impute_expr_z(gwas_df, r_snp, weights)

    with pytest.raises(ValueError, match="Unknown settings"):
        impute_expr_z(gwas_df, r_snp, weights, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        estimate_ctwas_priors([imp], settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        susie_rss_group_prior(
            imp.z_joint, imp.R_joint, imp.is_gene, 0.05, 0.001, 1e-3, 1e-4, 10000, settings="invalid"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unknown settings"):
        run_ctwas([imp], n_samples=10000, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        CTWAS(settings="invalid")  # type: ignore[arg-type]


def test_impute_expr_z_matches_individual_level_genotypes() -> None:
    """Verify imputed Z-scores and joint R_tilde against individual-level X and Y."""
    rng = np.random.default_rng(123)
    n_ind, p_snp = 2500, 15
    r_true = _ar1_ld(p_snp, 0.65)
    x_raw = rng.multivariate_normal(np.zeros(p_snp), r_true, size=n_ind)
    x_std = (x_raw - x_raw.mean(axis=0)) / x_raw.std(axis=0, ddof=0)
    r_emp = (x_std.T @ x_std) / n_ind

    y = 0.25 * x_std[:, 3] - 0.18 * x_std[:, 9] + rng.normal(scale=0.9, size=n_ind)
    y_std = (y - y.mean()) / y.std(ddof=0)
    z_snp = (x_std.T @ y_std) / np.sqrt(n_ind)

    snp_ids = [f"rs{i}" for i in range(p_snp)]
    gwas_df = pd.DataFrame({"snp": snp_ids, "z": z_snp})
    w_g1 = np.zeros(p_snp)
    w_g1[[2, 3, 4]] = [0.4, 0.9, -0.3]
    w_g2 = np.zeros(p_snp)
    w_g2[[3, 8, 9]] = [0.2, 0.5, 0.7]

    e1 = x_std @ w_g1
    e1_std = (e1 - e1.mean()) / e1.std(ddof=0)
    e2 = x_std @ w_g2
    e2_std = (e2 - e2.mean()) / e2.std(ddof=0)

    z_g1_emp = float((e1_std @ y_std) / np.sqrt(n_ind))
    z_g2_emp = float((e2_std @ y_std) / np.sqrt(n_ind))
    joint_features = np.column_stack([e1_std, e2_std, x_std])
    r_joint_emp = (joint_features.T @ joint_features) / n_ind

    imp = impute_expr_z(
        gwas_df,
        r_emp,
        {"G1": w_g1, "G2": w_g2},
        region_id="test_blk",
    )
    assert np.allclose(imp.gene_z_df["z"].to_numpy(), [z_g1_emp, z_g2_emp], atol=1e-10)
    assert np.allclose(imp.R_joint, r_joint_emp, atol=1e-10)


def test_impute_expr_z_flexible_beta_se_and_dataframe_ld() -> None:
    p = 8
    snps = [f"v{i}" for i in range(p)]
    r_df = pd.DataFrame(_ar1_ld(p, 0.4), index=snps, columns=snps)
    gwas_df = pd.DataFrame(
        {"rsid": snps, "beta": np.linspace(-0.2, 0.4, p), "se": np.full(p, 0.05)}
    )
    w_df = pd.DataFrame({"snp": ["v1", "v2"], "weight": [0.6, 0.8]})
    imp = impute_expr_z(gwas_df, r_df, {"ENSG1": w_df})
    assert imp.z_joint.shape == (p + 1,)
    assert imp.is_gene.sum() == 1
    assert imp.is_cis_snp.sum() == 2


def test_estimate_ctwas_priors_learns_gene_enrichment_and_mediated_h2() -> None:
    regions, _, _ = _simulate_ctwas_regions(n_regions=24, n_snps_per_reg=40, n_samples=40000)
    priors = estimate_ctwas_priors(regions, max_iter=50, tol=1e-5, n_samples=40000)

    # True causal rate: 10/48 ~ 0.208 for genes vs 2/960 ~ 0.0021 for SNPs (~100x enrichment)
    assert priors.pi_gene > 0.10
    assert priors.pi_snp < 0.02
    assert priors.enrichment > 15.0
    assert priors.prop_h2_gene > 0.65
    assert priors["enrichment"] == priors.enrichment
    assert "pi_gene" in priors.keys()


def test_priors_default_stabilization_vs_r_default_thinning() -> None:
    """Verify non-zero prior floor on pure-null tiny dataset in default and thin=0.1 in R_default."""
    r_snp = _ar1_ld(30, 0.5)
    snps = [f"rs{i}" for i in range(30)]
    null_regs = []
    for k in range(3):
        gwas_df = pd.DataFrame({"snp": snps, "z": np.zeros(30)})
        w = [{"gene_id": f"G{k}", "snps": ["rs3", "rs4"], "weights": [0.7, 0.3]}]
        null_regs.append(impute_expr_z(gwas_df, r_snp, w, region_id=f"null_{k}"))

    pri_def = estimate_ctwas_priors(null_regs, settings="default")
    assert pri_def.thin == 1.0
    assert pri_def.pi_gene > 0.0
    assert pri_def.pi_snp > 0.0

    regions, _, _ = _simulate_ctwas_regions(n_regions=20, n_snps_per_reg=40, seed=55)
    pri_r = estimate_ctwas_priors(regions, settings="R_default", n_samples=40000)
    assert pri_r.thin == 0.1
    assert pri_r.enrichment > 10.0


def test_regularize_ld_matrix_handles_singular_gene_snp_matrix() -> None:
    """Imputed gene + cis-SNP joint R has exact 0 eigenvalues; regularization makes it SPD."""
    r_snp = _ar1_ld(12, 0.7)
    gwas_df = pd.DataFrame({"snp": [f"rs{i}" for i in range(12)], "z": np.ones(12)})
    imp = impute_expr_z(
        gwas_df,
        r_snp,
        [{"gene_id": "G1", "snps": ["rs2", "rs3", "rs4"], "weights": [0.4, 0.8, 0.2]}],
    )
    raw_eigs = np.linalg.eigvalsh(imp.R_joint)
    assert np.min(raw_eigs) < 1e-12

    r_reg = regularize_ld_matrix(imp.R_joint, eig_floor=1e-4)
    reg_eigs = np.linalg.eigvalsh(r_reg)
    assert np.min(reg_eigs) >= 1e-4 - 1e-12
    assert np.allclose(np.diag(r_reg), 1.0)


def test_pleiotropic_noncoding_snp_eliminates_false_positive_twas_gene() -> None:
    """Direct causal SNP in LD with eQTL causes high marginal |z_gene|, but cTWAS rejects gene."""
    p = 25
    r_snp = _ar1_ld(p, 0.75)
    snps = [f"rs{i}" for i in range(p)]
    # Direct causal SNP is rs10 with z = 9.0; gene uses eQTL rs8 (LD r = 0.75^2 = 0.5625 -> z_gene = 5.06!)
    z_snp = 9.0 * r_snp[:, 10]
    gwas_df = pd.DataFrame({"snp": snps, "z": z_snp})
    imp = impute_expr_z(
        gwas_df,
        r_snp,
        [{"gene_id": "FALSE_POS_GENE", "snps": ["rs8"], "weights": [1.0]}],
    )
    # Marginal TWAS Z-score exceeds genome-wide significance (5.06 > 4.5)!
    assert abs(float(imp.gene_z_df["z"].iloc[0])) > 5.0

    fit = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.05,
        pi_snp=0.005,
        var_gene=0.002,
        var_snp=0.002,
        n_samples=40000,
        L=5,
    )
    gene_pip = float(fit.susie_pip[0])
    causal_snp_pip = float(fit.susie_pip[1 + 10])
    assert causal_snp_pip > 0.95
    assert gene_pip < 0.05


def test_bystander_coregulated_gene_hitchhiking_eliminated() -> None:
    """Two correlated genes (G1 causal, G2 bystander via LD): cTWAS isolates G1."""
    p = 30
    r_snp = _ar1_ld(p, 0.70)
    snps = [f"rs{i}" for i in range(p)]
    w1 = np.zeros(p)
    w1[[5, 6]] = [0.6, 0.8]
    w1 /= np.sqrt(w1 @ r_snp @ w1)
    w2 = np.zeros(p)
    w2[[8, 9]] = [0.7, 0.7]
    w2 /= np.sqrt(w2 @ r_snp @ w2)

    # Signal generated purely by causal gene G1
    z_snp = 8.5 * (r_snp @ w1)
    gwas_df = pd.DataFrame({"snp": snps, "z": z_snp})
    imp = impute_expr_z(gwas_df, r_snp, {"CAUSAL_G1": w1, "BYSTANDER_G2": w2})
    # Bystander G2 has large marginal TWAS |z| > 3.5 due to LD
    assert abs(float(imp.gene_z_df.loc[imp.gene_z_df["gene_id"] == "BYSTANDER_G2", "z"].iloc[0])) > 3.5

    fit = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.10,
        pi_snp=0.001,
        var_gene=0.002,
        var_snp=0.001,
        n_samples=40000,
        L=5,
    )
    assert fit.susie_pip[0] > 0.90
    assert fit.susie_pip[1] < 0.05


def test_multi_causal_region_recovers_both_causal_gene_and_independent_snp() -> None:
    """Region with 1 causal gene + 1 independent direct SNP yields 2 distinct credible sets."""
    p = 40
    r_snp = _ar1_ld(p, 0.45)
    snps = [f"rs{i}" for i in range(p)]
    w_g = np.zeros(p)
    w_g[[3, 4, 5]] = [0.4, 0.8, 0.4]
    w_g /= np.sqrt(w_g @ r_snp @ w_g)

    # Causal gene at index 3..5 + independent causal SNP at rs33
    z_snp = 8.0 * (r_snp @ w_g) + 7.5 * r_snp[:, 33]
    gwas_df = pd.DataFrame({"snp": snps, "z": z_snp})
    imp = impute_expr_z(gwas_df, r_snp, {"GENE_CAUSAL": w_g})

    fit = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.10,
        pi_snp=0.005,
        var_gene=0.002,
        var_snp=0.0015,
        n_samples=40000,
        L=5,
    )
    assert fit.susie_pip[0] > 0.90  # GENE_CAUSAL
    assert fit.susie_pip[1 + 33] > 0.90  # rs33
    assert len(fit.credible_sets) == 2


def test_null_region_has_low_pips_and_zero_credible_sets() -> None:
    rng = np.random.default_rng(88)
    p = 35
    r_snp = _ar1_ld(p, 0.5)
    z_null = np.linalg.cholesky(r_snp) @ rng.normal(scale=0.6, size=p)
    gwas_df = pd.DataFrame({"snp": [f"rs{i}" for i in range(p)], "z": z_null})
    imp = impute_expr_z(
        gwas_df,
        r_snp,
        [{"gene_id": "G_NULL", "snps": ["rs5", "rs6"], "weights": [0.6, 0.6]}],
    )
    fit = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.02,
        pi_snp=0.001,
        var_gene=0.0015,
        var_snp=0.001,
        n_samples=40000,
        L=5,
    )
    assert np.max(fit.susie_pip) < 0.15
    assert fit.credible_sets == []


def test_end_to_end_run_ctwas_default_vs_r_default_stage2_gate() -> None:
    regions, causal_genes, causal_snps = _simulate_ctwas_regions(
        n_regions=24, n_snps_per_reg=40, n_samples=40000, seed=42
    )
    res_def = run_ctwas(regions, n_samples=40000, L=5, settings="default")
    est_r = CTWAS(L=5, settings="R_default")
    res_r = est_r.fit(regions, n_samples=40000)

    # All regions run Stage 2 in default; only high-gene-PIP regions rerun in R_default
    assert res_def.gene_df["rerun_stage2"].all()
    assert not res_r.gene_df["rerun_stage2"].all()
    assert res_r.gene_df["rerun_stage2"].any()

    # High sensitivity & specificity on true causal genes
    high_pip_genes = set(
        res_def.gene_df.loc[res_def.gene_df["susie_pip"] > 0.75, "gene_id"]
    )
    assert len(high_pip_genes.intersection(set(causal_genes))) >= 8
    # Non-causal genes in pleiotropic SNP regions 11 & 12 stay suppressed
    pleio_genes = res_def.gene_df[res_def.gene_df["region_id"].isin(["reg_11", "reg_12"])]
    assert np.all(pleio_genes["susie_pip"].to_numpy() < 0.20)
    # While direct pleiotropic SNPs rs18 in reg_11 & reg_12 have high PIP
    for reg_id, s_id in causal_snps:
        sub_s = res_def.snp_df[
            (res_def.snp_df["region_id"] == reg_id) & (res_def.snp_df["snp_id"] == s_id)
        ]
        assert float(sub_s["susie_pip"].iloc[0]) > 0.75

    summary_str = res_def.summary()
    assert "CTWASResult" in summary_str
    combined = res_def.to_dataframe()
    assert len(combined) == len(res_def.gene_df) + len(res_def.snp_df)
    assert combined["susie_pip"].is_monotonic_decreasing


def test_plain_dict_region_inputs_and_overlapping_eqtls() -> None:
    p = 20
    r_snp = _ar1_ld(p, 0.5)
    z_snp = 7.5 * r_snp[:, 4]
    # Plain dict input without ImputedRegionData
    reg_dict = {
        "region_id": "dict_blk",
        "z_joint": np.concatenate([[7.2, 1.1], z_snp]),
        "R_joint": np.eye(p + 2),
        "is_gene": np.array([True, True] + [False] * p),
    }
    res = run_ctwas([reg_dict], n_samples=30000, L=3, settings="default")
    assert len(res.gene_df) == 2
    assert len(res.snp_df) == p


def test_susie_rss_group_prior_prioritizes_enriched_gene_over_collinear_eqtl() -> None:
    """When gene and single top eQTL are strongly correlated, higher pi_gene favors the gene."""
    p = 15
    r_snp = _ar1_ld(p, 0.5)
    snps = [f"rs{i}" for i in range(p)]
    w_g = np.zeros(p)
    w_g[[4, 5]] = [0.95, 0.15]
    w_g /= np.sqrt(w_g @ r_snp @ w_g)
    z_snp = 8.0 * (r_snp @ w_g)
    imp = impute_expr_z(pd.DataFrame({"snp": snps, "z": z_snp}), r_snp, {"G_TOP": w_g})

    fit = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.20,
        pi_snp=0.001,
        var_gene=0.002,
        var_snp=0.002,
        n_samples=40000,
        L=5,
    )
    assert fit.susie_pip[0] > 0.85
    assert fit.susie_pip[0] > np.max(fit.susie_pip[1:])
    assert fit.converged


def test_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="regions must contain at least 1"):
        estimate_ctwas_priors([])
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        run_ctwas([{"z_joint": np.ones(3), "is_gene": np.array([True, False, False])}], n_samples=0)
    with pytest.raises(ValueError, match="thin must be in"):
        estimate_ctwas_priors(
            [{"z_joint": np.ones(3), "is_gene": np.array([True, False, False])}], thin=1.5
        )
