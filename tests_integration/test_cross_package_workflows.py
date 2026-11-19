"""End-to-end cross-package integration tests spanning pythonic_biotools."""

from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("all", repo_root=REPO_ROOT)


def test_workspace_cli_bundles_and_all_70_imports():
    status = biotools.check_bundle_imports("all", repo_root=REPO_ROOT)
    assert len(status) == 70
    assert all(status.values())
    for b in ("oncology", "statgen", "singlecell", "epigenomics", "proteomics", "survival"):
        pkgs = biotools.resolve_bundle_packages(b)
        assert len(pkgs) >= 5
        assert set(pkgs).issubset(set(status.keys()))


def test_oncology_copy_number_hrd_timing_and_signatures_pipeline():
    from chord import chord_predict, extract_sigs_chord
    from mmsig import make_synthetic_cosmic_reference, mm_fit_signatures
    from mutationtimer import estimate_wgd_timing, mutation_time
    from scarhrd import calculate_hrd
    from tracksig import COSMIC_96_CONTEXTS, tracksig

    rng = np.random.default_rng(42)
    cn_segments = pd.DataFrame(
        [
            {"chrom": "chr1", "start": 1_000_000, "end": 45_000_000, "major_cn": 2, "minor_cn": 0, "total_cn": 2},
            {"chrom": "chr1", "start": 45_000_000, "end": 120_000_000, "major_cn": 2, "minor_cn": 2, "total_cn": 4},
            {"chrom": "chr2", "start": 1_000_000, "end": 35_000_000, "major_cn": 2, "minor_cn": 0, "total_cn": 2},
            {"chrom": "chr2", "start": 35_000_000, "end": 95_000_000, "major_cn": 2, "minor_cn": 2, "total_cn": 4},
            {"chrom": "chr3", "start": 1_000_000, "end": 80_000_000, "major_cn": 2, "minor_cn": 1, "total_cn": 3},
        ]
    )

    # 1. Copy-number HRD genomic scar scoring (scarHRD)
    hrd_res = calculate_hrd(cn_segments, genome="GRCh38", settings="default")
    hrd_r = calculate_hrd(cn_segments, genome="GRCh38", settings="R_default")
    assert hrd_res.hrd_loh >= 2 and hrd_r.hrd_loh >= 2

    # 2. Evolutionary timing of somatic SNVs + WGD estimation (MutationTimeR)
    purity = 0.80
    positions = rng.integers(2_000_000, 115_000_000, size=240)
    chroms = np.where(np.arange(240) < 120, "chr1", "chr2")
    # Early pre-gain/WGD mutations (m=2 -> VAF ~ 0.8*2 / (0.8*4 + 0.4) = 0.44) vs late (m=1 -> ~0.22)
    is_early = np.arange(240) < 140
    vaf_true = np.where(is_early, 0.46, 0.21)
    depths = np.full(240, 100)
    alts = rng.binomial(depths, vaf_true)
    sig_ref = make_synthetic_cosmic_reference(["SBS1", "SBS3", "SBS5", "SBS13"])
    ctxs = np.where(
        is_early,
        rng.choice(COSMIC_96_CONTEXTS, size=240, p=sig_ref["SBS1"].to_numpy()),
        rng.choice(COSMIC_96_CONTEXTS, size=240, p=sig_ref["SBS3"].to_numpy()),
    )
    variants_df = pd.DataFrame(
        {
            "chrom": chroms,
            "pos": positions,
            "alt_count": alts,
            "total_depth": depths,
            "context": ctxs,
        }
    )

    mt_res = mutation_time(variants_df, cn_segments, purity=purity, n_boot=30, settings="default")
    wgd_res = estimate_wgd_timing(mt_res.segments_df, settings="default")
    assert "clonal [early]" in set(mt_res.variants_df["timing_class"])
    assert 0.0 <= wgd_res.wgd_time <= 1.0

    # 3. Signature trajectory segmentation (TrackSig) + bootstrap signature refitting (mmsig)
    ts_res = tracksig(variants_df, sig_ref, purity=purity, bin_size=80, settings="default")
    assert ts_res.bin_exposures_df.shape[0] == 3

    catalog_df = variants_df["context"].value_counts().reindex(sig_ref.index, fill_value=0).to_frame("TUMOR_1")
    mm_res = mm_fit_signatures(catalog_df, sig_ref, n_boot=40, settings="default")
    assert mm_res.exposures.loc["TUMOR_1", "SBS1"] > 0.1

    # 4. Mutational-context BRCA1/BRCA2 HRD classification (CHORD)
    indel_df = pd.DataFrame({"indel_type": ["del.mh"] * 45 + ["del.rep"] * 15, "mh_len": [3] * 45 + [0] * 15})
    sv_df = pd.DataFrame({"sv_type": ["DUP"] * 25 + ["DEL"] * 5, "sv_len": [15_000] * 30})
    chord_feats = extract_sigs_chord(snv_df=variants_df, indel_df=indel_df, sv_df=sv_df, sample_id="TUMOR_1")
    chord_res = chord_predict(chord_feats, settings="default")
    assert 0.0 <= float(chord_res.to_dataframe()["p_hrd"].iloc[0]) <= 1.0


def test_liquid_biopsy_and_multiregion_phylogeny_pipeline():
    from conipher import run_conipher
    from ichorcna import emission_means, run_ichorcna
    from qdnaseq import run_ace

    rng = np.random.default_rng(101)
    # 1. ULP-WGS cfDNA bins -> ichorCNA + QDNAseq ACE
    n_bins = 120
    tf_true = 0.25
    states_cn = np.concatenate([np.full(50, 2), np.full(40, 3), np.full(30, 1)])
    expected_logr = np.log2((2.0 * (1.0 - tf_true) + tf_true * states_cn) / 2.0)
    bins_df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 60 + ["chr2"] * 60,
            "start": np.tile(np.arange(0, 60_000_000, 1_000_000), 2),
            "end": np.tile(np.arange(1_000_000, 61_000_000, 1_000_000), 2),
            "logR": expected_logr + rng.normal(0.0, 0.04, size=n_bins),
        }
    )
    ichor_res = run_ichorcna(bins_df, max_iter=15, settings="default")
    assert 0.10 <= ichor_res.tumor_fraction <= 0.45

    seg_for_ace = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [1, 50_000_001, 10_000_001],
            "end": [50_000_000, 60_000_000, 40_000_000],
            "num_bins": [50, 10, 30],
            "seg_mean": [1.0, 1.0 + 0.5 * tf_true, 1.0 - 0.5 * tf_true],
        }
    )
    ace_res = run_ace(seg_for_ace, settings="default")
    assert 0.05 <= ace_res.cellularity <= 1.0

    # 2. Multi-region somatic mutations -> CONIPHER tree reconstruction
    rows = []
    # Truncal cluster (CCF ~ 1.0 in R1, R2) + Branch A (R1=0.6, R2=0.0) + Branch B (R1=0.0, R2=0.55)
    specs = [(1.0, 1.0, 15), (0.65, 0.0, 10), (0.0, 0.60, 10)]
    mut_idx = 0
    for ccf1, ccf2, n_m in specs:
        for _ in range(n_m):
            mut_idx += 1
            for reg, ccf in (("R1", ccf1), ("R2", ccf2)):
                p_vaf = min(0.49, 0.8 * ccf / 2.0)
                rows.append(
                    {
                        "mutation_id": f"M{mut_idx}",
                        "region": reg,
                        "alt_count": int(rng.binomial(120, p_vaf)),
                        "depth": 120,
                        "major_cn": 1,
                        "minor_cn": 1,
                    }
                )
    con_res = run_conipher(
        pd.DataFrame(rows),
        purity_by_region={"R1": 0.8, "R2": 0.8},
        max_clusters=5,
        min_cluster_size=5,
        settings="default",
    )
    assert con_res.n_alternative_trees >= 1
    assert len(con_res.parent_child_df) >= 1


def test_clinical_trial_survival_timeroc_idinri_rmst_winratio_and_jmbayes2():
    from buysetest import Endpoint, buyse_test
    from jmbayes2 import jm, predict_dynamic
    from survidinri import idi_inf
    from survrm2 import rmst2
    from timeroc import time_roc

    rng = np.random.default_rng(2026)
    n = 90
    arm = np.repeat([0, 1], n // 2)
    biomarker = rng.normal(0.0, 1.0, size=n)
    stage = rng.integers(0, 2, size=n).astype(float)
    linpred = -0.65 * arm + 0.85 * biomarker + 0.4 * stage
    t_true = rng.exponential(scale=np.exp(-linpred) * 6.0)
    t_cens = rng.uniform(4.0, 14.0, size=n)
    time = np.minimum(t_true, t_cens)
    event = (t_true <= t_cens).astype(int)

    # 1. Horizon AUC (timeROC) + Incremental biomarker IDI/NRI (survIDINRI)
    roc_res = time_roc(T=time, delta=event, marker=biomarker, cause=1, times=[3.0, 5.0], settings="default")
    assert np.all(roc_res.auc_1 > 0.55)

    cov0 = np.column_stack([stage])
    cov1 = np.column_stack([stage, biomarker])
    idi_res = idi_inf(time, event, cov0, cov1, t0=5.0, npert=60, seed=7, settings="default")
    assert idi_res.m1_idi.est > 0.0

    # 2. Restricted Mean Survival Time (survRM2) + Composite Win Ratio (BuyseTest)
    rmst_res = rmst2(time, event, arm, tau=7.0, covariates=cov1, settings="default")
    assert rmst_res.arm1.rmst > rmst_res.arm0.rmst

    trial_df = pd.DataFrame(
        {
            "arm": arm,
            "os_time": time,
            "os_event": event,
            "tox_grade3": rng.binomial(1, np.where(arm == 1, 0.15, 0.35)),
        }
    )
    bt_res = buyse_test(
        trial_df,
        treatment="arm",
        endpoints=[
            Endpoint("os_time", kind="tte", status_col="os_event", operator="higher", threshold=0.25),
            Endpoint("tox_grade3", kind="binary", operator="lower"),
        ],
        settings="default",
    )
    assert float(bt_res.to_dataframe()["win_ratio"].iloc[-1]) > 1.0

    # 3. Joint longitudinal biomarker + survival dynamic prediction (JMbayes2)
    surv_sub = pd.DataFrame({"id": np.arange(40), "time": time[:40], "event": event[:40], "arm": arm[:40]})
    long_rows = []
    for i in range(40):
        t_grid = [t for t in (0.0, 1.0, 2.0, 3.0) if t < time[i]] or [0.0]
        for tj in t_grid:
            long_rows.append(
                {
                    "id": i,
                    "obstime": tj,
                    "y": biomarker[i] + 0.15 * tj * (1 - arm[i]) + rng.normal(0.0, 0.2),
                    "arm": arm[i],
                }
            )
    jm_fit = jm(
        surv_sub,
        pd.DataFrame(long_rows),
        surv_covs=["arm"],
        long_covs=["obstime", "arm"],
        n_iter=160,
        burn_in=60,
        seed=12,
        settings="default",
    )
    new_pat = pd.DataFrame({"id": [999, 999], "obstime": [0.0, 1.5], "y": [0.8, 1.1], "arm": [0, 0]})
    dyn = predict_dynamic(
        jm_fit,
        new_pat,
        t_landmark=1.5,
        horizons=[3.0, 5.0],
        surv_covs_row={"arm": 0.0},
        n_mcmc=60,
        settings="default",
    )
    assert len(dyn) == 2 and np.all((dyn["surv_prob"] >= 0.0) & (dyn["surv_prob"] <= 1.0))


def test_spatial_cside_saturn_and_mashr_multiomics_pipeline():
    from cside import run_cside_single
    from mashr import cov_canonical, mash, mash_set_data
    from saturn import filter_dtu, fit_dtu, test_dtu

    rng = np.random.default_rng(303)
    # 1. Spatial cell-type-specific DE (CSIDE on AnnData)
    n_spots, n_genes = 80, 12
    x_cov = np.linspace(0.0, 1.0, n_spots)
    w_ct1 = rng.uniform(0.3, 0.8, size=n_spots)
    weights = np.column_stack([w_ct1, 1.0 - w_ct1])
    n_umi = np.full(n_spots, 2000.0)
    rates = np.zeros((n_spots, n_genes))
    for g in range(n_genes):
        slope = 1.4 if g == 0 else 0.0
        rates[:, g] = n_umi * (weights[:, 0] * 0.02 * np.exp(slope * x_cov) + weights[:, 1] * 0.02)
    counts = rng.poisson(rates)
    adata_sp = ad.AnnData(X=counts.astype(float))
    adata_sp.var_names = [f"Gene{g}" for g in range(n_genes)]
    adata_sp.obsm["cell_type_weights"] = pd.DataFrame(weights, index=adata_sp.obs_names, columns=["Tumor", "Stroma"])
    adata_sp.obs["hypoxia"] = x_cov

    cside_res = run_cside_single(adata_sp, explanatory_variable=adata_sp.obs["hypoxia"], settings="default")
    df_cside = cside_res.to_dataframe()
    assert "Gene0" in set(df_cside["gene"]) if "gene" in df_cside.columns else len(df_cside) > 0

    # 2. Isoform Differential Transcript Usage (satuRn) + Cross-condition shrinkage (mashr)
    tx_rows, tx2gene = [], []
    n_samp = 12
    grp = np.array([0] * 6 + [1] * 6)
    for g in range(25):
        p1 = np.where(grp == 1, 0.75, 0.25) if g == 0 else np.full(n_samp, 0.5)
        tot = rng.integers(80, 140, size=n_samp)
        c1 = rng.binomial(tot, p1)
        tx_rows.extend([c1, tot - c1])
        tx2gene.extend(
            [
                {"transcript_id": f"TX_{g}_a", "gene_id": f"G_{g}"},
                {"transcript_id": f"TX_{g}_b", "gene_id": f"G_{g}"},
            ]
        )
    tx_df = pd.DataFrame(
        np.vstack(tx_rows),
        index=[r["transcript_id"] for r in tx2gene],
        columns=[f"S{i}" for i in range(n_samp)],
    )
    design_df = pd.DataFrame({"intercept": 1.0, "cond": grp}, index=tx_df.columns)
    filt = filter_dtu(tx_df, pd.DataFrame(tx2gene), min_count=5, min_samples=2)
    dtu_fit = fit_dtu(filt.counts, filt.tx2gene, design_df, settings="default")
    dtu_res = test_dtu(dtu_fit, contrast=[0.0, 1.0], settings="default")
    assert dtu_res.loc["TX_0_a", "pval_regular"] < 1e-3

    # Pass multi-condition effects into mashr
    b_hat = np.column_stack([dtu_res["estimates"].to_numpy(), dtu_res["estimates"].to_numpy() * 0.9])
    se_hat = np.column_stack([dtu_res["se"].to_numpy(), dtu_res["se"].to_numpy() * 1.05])
    mdata = mash_set_data(b_hat, se_hat, settings="default")
    mres = mash(mdata, cov_canonical(mdata), settings="default")
    assert mres.posterior_mean.shape == b_hat.shape


def test_epigenomics_dss_csaw_monalisa_and_deqms_proteomics_pipeline():
    from csaw import combine_tests, glm_ql_fit_test, merge_windows, window_counts
    from deqms import median_sweeping, output_result, spectra_count_ebayes
    from dss import call_dmr, dml_test
    from monalisa import calc_binned_motif_enrichment, pfm_to_pwm, rand_lasso_stab_sel

    rng = np.random.default_rng(404)
    # 1. DSS bisulfite DML/DMR -> monaLisa TF motif enrichment & stability selection
    pos = np.arange(1000, 1000 + 60 * 35, 35)
    ctrl_bs, trt_bs = [], []
    for _ in range(3):
        n_cov = rng.integers(25, 45, size=60)
        p_c = np.full(60, 0.20)
        p_t = np.where((pos >= 1350) & (pos <= 1750), 0.80, 0.20)
        ctrl_bs.append(pd.DataFrame({"chr": "chr1", "pos": pos, "N": n_cov, "X": rng.binomial(n_cov, p_c)}))
        trt_bs.append(pd.DataFrame({"chr": "chr1", "pos": pos, "N": n_cov, "X": rng.binomial(n_cov, p_t)}))

    dml_df = dml_test(ctrl_bs, trt_bs, settings="default")
    dmr_df = call_dmr(dml_df, p_threshold=1e-3, minCG=3, minlen=50, settings="default")
    assert len(dmr_df) >= 1

    # Feed region methylation shifts + sequences into monaLisa ("CCACAA" motif)
    pfm_ctcf = np.array([[5, 5, 90, 5, 90, 90], [90, 90, 2, 90, 3, 3], [3, 3, 5, 3, 5, 5], [2, 2, 3, 2, 2, 2]], float)
    pwms = {"CTCF_like": pfm_to_pwm(pfm_ctcf)}
    reg_seqs = ["ATCGCCACAAATCGCGA" * 5] * 30 + ["ATATGCATGCTAGCTAG" * 5] * 30
    y_delta = np.concatenate([rng.normal(0.55, 0.08, 30), rng.normal(-0.05, 0.08, 30)])
    bins = np.where(y_delta > 0.25, "hyper", "neutral")
    enr = calc_binned_motif_enrichment(reg_seqs, bins, pwms=pwms, settings="default")
    assert enr.to_dataframe().shape[0] == 2

    from monalisa import scan_sequences_pwm

    scan = scan_sequences_pwm(reg_seqs, pwms)
    X_hits = np.column_stack([scan.counts, rng.poisson(1.0, size=(60, 4))])
    stab = rand_lasso_stab_sel(
        pd.DataFrame(X_hits, columns=["CTCF_like", "N1", "N2", "N3", "N4"]),
        y_delta,
        n_subsamples=40,
        cutoff=0.6,
        settings="default",
    )
    assert float(stab.sel_prob["CTCF_like"]) >= float(stab.sel_prob[["N1", "N2", "N3", "N4"]].max())

    # 2. csaw sliding-window differential ChIP/ATAC binding
    frags = []
    for s_idx in range(6):
        is_trt = s_idx >= 3
        n_peak = 90 if is_trt else 15
        starts = np.concatenate([rng.integers(500, 620, size=n_peak), rng.integers(2000, 2150, size=35)])
        frags.append(pd.DataFrame({"chrom": "chr1", "pos": starts, "strand": "+", "mapq": 40}))
    wdata = window_counts(fragments=frags, chrom_sizes={"chr1": 5000}, settings="default")
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    ql_tab = glm_ql_fit_test(wdata, design=design, settings="default")
    merged = merge_windows(wdata, tol=100, settings="default")
    comb = combine_tests(merged, ql_tab, settings="default")
    assert float(comb["FDR"].min()) < 0.05

    # 3. Quantitative PSM -> protein differential abundance (DEqMS)
    psm_rows = []
    for p_idx in range(40):
        n_psms = 12 if p_idx < 10 else 2
        shift = 1.4 if p_idx == 0 else 0.0
        for _ in range(n_psms):
            vals = np.concatenate([rng.normal(20.0, 0.25, 3), rng.normal(20.0 + shift, 0.25, 3)])
            row = {"protein": f"PROT_{p_idx}"}
            row.update({f"TMT_{k}": vals[k] for k in range(6)})
            psm_rows.append(row)
    swept = median_sweeping(pd.DataFrame(psm_rows), group_col="protein", settings="default")
    deq_fit = spectra_count_ebayes(
        swept.abundance,
        design=np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]),
        count=swept.count,
        settings="default",
    )
    top_prot = output_result(deq_fit, coef=1, settings="default")
    assert top_prot.index[0] == "PROT_0" or top_prot["gene"].iloc[0] == "PROT_0"


def test_statgen_hdl_lava_genomicsem_ctwas_colocproptest_and_cismr_pipeline():
    from cismr import mr_pcgmm, mr_spi
    from colocproptest import run_coloc_prop_test
    from ctwas import impute_expr_z, susie_rss_group_prior
    from fgwas import fit_fgwas
    from genomicsem import GenomicCovStruct, commonfactor
    from grapple import grapple_robust_est
    from hdl import build_ld_eigen_panel, hdl_rg
    from lava import process_locus, run_pcor, run_univ

    rng = np.random.default_rng(505)
    # 1. Locus LD + correlated molecular cis-QTL & GWAS summary statistics
    m = 30
    idx = np.arange(m)
    ld_R = 0.7 ** np.abs(idx[:, None] - idx[None, :])
    n_gwas = 40_000

    b_eqtl_true = np.zeros(m)
    b_eqtl_true[[5, 18]] = [0.18, 0.14]
    theta_causal = 0.35
    b_gwas_true = theta_causal * b_eqtl_true

    se = np.full(m, 1.0 / np.sqrt(n_gwas))
    L = np.linalg.cholesky(ld_R)
    bx = ld_R @ b_eqtl_true + se * (L @ rng.standard_normal(m))
    by = ld_R @ b_gwas_true + se * (L @ rng.standard_normal(m))

    # cis-MR (PC-GMM & MR-SPI) + Proportional Colocalization (colocPropTest)
    pcgmm_res = mr_pcgmm(bx, se, by, se, ld_R=ld_R, nx=n_gwas, ny=n_gwas, settings="default")
    spi_res = mr_spi(bx, se, by, se, ld_R=ld_R, n=n_gwas, settings="default")
    assert pcgmm_res.theta == pytest.approx(theta_causal, abs=0.12)
    assert spi_res.estimate == pytest.approx(theta_causal, abs=0.15)

    prop_res = run_coloc_prop_test(
        {"beta": bx, "se": se},
        {"beta": by, "se": se},
        LD=ld_R,
        settings="default",
    )
    assert not prop_res.reject_proportionality

    # 2. Causal TWAS (cTWAS) joint gene + direct-SNP SuSiE-RSS
    gwas_z_df = pd.DataFrame({"id": [f"rs{i}" for i in range(m)], "z": by / se})
    weights = [{"gene_id": "GENE_A", "snps": ["rs5", "rs18"], "weights": [0.18, 0.14]}]
    imp = impute_expr_z(gwas_z_df, ld_R, weights, settings="default")
    susie_res = susie_rss_group_prior(
        imp.z_joint,
        imp.R_joint,
        imp.is_gene,
        pi_gene=0.15,
        pi_snp=0.005,
        var_gene=0.05,
        var_snp=0.005,
        n_samples=n_gwas,
        settings="default",
    )
    assert float(susie_res.pip[imp.is_gene][0]) > 0.5

    # 3. LAVA partial local genetic correlation + HDL + GenomicSEM + fgwas + GRAPPLE
    b_z_conf = np.zeros(m)
    b_z_conf[10] = 0.20
    z_lava = pd.DataFrame(
        {
            "T1": bx / se + np.sqrt(n_gwas) * (ld_R @ b_z_conf),
            "T2": by / se + np.sqrt(n_gwas) * (ld_R @ b_z_conf),
            "T3": np.sqrt(n_gwas) * (ld_R @ b_z_conf) + (L @ rng.standard_normal(m)),
        }
    )
    locus = process_locus(z_lava, ld_R, sample_sizes={"T1": n_gwas, "T2": n_gwas, "T3": n_gwas})
    assert (run_univ(locus)["p"] < 1e-3).all()
    pcor = run_pcor(locus, pair=("T1", "T2"), cond_traits=["T3"], n_sim=2000)
    assert float(pcor["pcor"].iloc[0]) > 0.2

    panel = build_ld_eigen_panel([ld_R], settings="default")
    hdl_res = hdl_rg(z_lava["T1"].to_numpy(), z_lava["T2"].to_numpy(), panel, N1=n_gwas, N2=n_gwas, jackknife=False)
    assert np.isfinite(hdl_res.rg)

    lam = np.array([0.6, 0.5, 0.45])
    S_sem = np.outer(lam, lam) + np.diag([0.15, 0.18, 0.20])
    sem_fit = commonfactor(GenomicCovStruct(S=S_sem, V_S=np.eye(6) * 1e-4, trait_names=["A", "B", "C"]))
    assert sem_fit.modelfit["CFI"] > 0.95

    fg_df = pd.DataFrame(
        {
            "SNPID": [f"rs{i}" for i in range(m)],
            "SEGNUMBER": [0] * 15 + [1] * 15,
            "BETA": by,
            "SE": se,
            "enhancer": [1 if i in (5, 18) else 0 for i in range(m)],
        }
    )
    fg_res = fit_fgwas(fg_df, annotations=["enhancer"], block_col="SEGNUMBER", fine_map=True, settings="default")
    assert float(fg_res.PPA_snp.sum()) == pytest.approx(2.0, abs=1e-5)

    # Multimodal / robust MR check via GRAPPLE
    bx_iv = np.full(25, 0.08)
    sx_iv = np.full(25, 0.01)
    by_iv = 0.40 * bx_iv + rng.normal(0.0, 0.01, size=25)
    grap_res = grapple_robust_est(bx_iv, sx_iv, by_iv, np.full(25, 0.01), settings="default")
    assert len(grap_res.modes) >= 1
