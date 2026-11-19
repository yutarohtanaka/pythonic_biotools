# Problem-Driven Guide: Using Published Tools in `pythonic_biotools`

Every directory in `pythonic_biotools` is a standalone Python implementation of a published bioinformatics, oncology, clinical trial, or statistical genetics tool. Use the lookup below to match your analysis question to the appropriate library.

---

## 1. Mendelian Randomization, cis-MR, Causal TWAS, Colocalization & Genetic Architecture

### *Does an exposure causally influence an outcome in two-sample or multivariable summary statistics?*
- **Standard `IVW`, coding-invariant `MR-Egger`, `Weighted Median`, `MVMR` (Sanderson–Windmeijer conditional $F$), allele harmonization, and LD clumping**: [`twosamplemr`](../twosamplemr/README.md) (`run_mr_suite`, `harmonise_mr_data`, `select_mr_instruments`).
- **Correlated cis-molecular instruments at a single drug-target gene locus (`cis-pQTL` / `cis-eQTL` / `cis-sQTL`) with LD principal-component GMM (`PC-GMM`, `MV-PC-GMM`), pair-voting valid-IV clique discovery (`MR-SPI`), or heteroskedastic variance-IV identification (`MR-GENIUS`)**: [`cismr`](../cismr/README.md) (`mr_pcgmm`, `mr_mvpcgmm`, `mr_spi`, `mr_genius`, `mr_genius_summary`, `CisMR`; Burgess et al., 2020; Yao et al., 2024; Tchetgen Tchetgen et al., 2021).
- **Testing for global horizontal pleiotropy, identifying specific outlier SNPs, and testing pre- vs post-outlier causal distortion (univariable or multivariable)**: [`mrpresso`](../mrpresso/README.md) (`mr_presso`, `mvmr_presso`; Verbanck et al., 2018).
- **Many weak instruments with systematic pleiotropic overdispersion ($\tau^2$) or heavy-tailed outliers**: [`mrraps`](../mrraps/README.md) (`mr_raps` with `huber` or `tukey` loss; Zhao et al., 2020).
- **Correlated + uncorrelated horizontal pleiotropy (univariable or multivariable)**: [`mrhorse`](../mrhorse/README.md) (`mr_horse`, `mvmr_horse`; Grant & Burgess, 2024).
- **Polygenic background pleiotropy + instrument-selection truncation (`|z_X| > t_0`) + sample overlap**: [`mrapss`](../mrapss/README.md) (`fit_mrapss`; Hu et al., 2022) or [`mrbee`](../mrbee/README.md) (`fit_mrbee`, `fit_mvmr_bee`; Lorincz-Comi et al., 2024).
- **Multivariable constrained maximum likelihood with BIC Model Averaging and Data Perturbation (`cML-DP`)**: [`mvcml`](../mvcml/README.md) (`cml_ma`, `mvcml_dp`, `mvcml_adjust`; Lin et al., 2023).
- **Separating causal effects from correlated and uncorrelated horizontal pleiotropy via genome-wide nuisance estimation and `PSIS-LOO` $\Delta\text{ELPD}$ model comparison**: [`cause`](../cause/README.md) (`gwas_merge`, `est_cause_params`, `ld_prune`, `cause`, `compare_elpd`; Morrison et al., 2020).
- **Clustering genetic instruments into $K$ distinct causal pathway groups, a null component, and pleiotropic outliers**: [`mrclust`](../mrclust/README.md) (`mr_clust_em`, `pr_clust`, `compute_ratio_estimates`, `two_stage_plot`; Foley et al., 2021).

### *Do two or more traits (GWAS, eQTL, pQTL, sQTL) share causal variants, mediate risk, or share genetic correlation / functional enrichment?*
- **Two-trait Bayesian colocalization (`PP.H0`–`PP.H4` and 95% credible sets)**: [`coloc`](../coloc/README.md) (`coloc_abf`; Wakefield 2009; Giambartolomei et al., 2014).
- **Multi-trait colocalization across 3–100+ traits simultaneously with branch-and-bound cluster discovery**: [`hyprcoloc`](../hyprcoloc/README.md) (`hyprcoloc`, `sensitivity_analysis`; Foley et al., 2021).
- **LD-robust frequentist proportional colocalization when multiple causal variants exist or in-sample LD is unavailable**: [`colocproptest`](../colocproptest/README.md) (`run_coloc_prop_test`, `test_pair_proportionality`, `proptest_pc`; Wallace, 2023).
- **Testing cis-xQTL mediation (`SMR` / `SMR-multi`) vs. linkage of distinct causal variants (`HEIDI`) using local LD**: [`smr-heidi`](../smr-heidi/README.md) (`run_locus_smr_heidi`, `smr_single`, `smr_multi`, `heidi_test`; Zhu et al., 2016; Wu et al., 2018).
- **Jointly fine-mapping imputed gene expression / protein traits alongside direct non-coding cis-SNPs to eliminate TWAS LD hitchhiking**: [`ctwas`](../ctwas/README.md) (`impute_expr_z`, `estimate_ctwas_priors`, `susie_rss_group_prior`, `run_ctwas`; Zhao et al., 2024).
- **High-definition full-likelihood global (`HDL`), local single-locus (`HDL-L`), and annotation-stratified (`HDL.S`) heritability and genetic correlation**: [`hdl`](../hdl/README.md) (`build_ld_eigen_panel`, `hdl_rg`, `hdl_h2`, `hdl_local`, `hdl_stratified`; Ning et al., 2020).
- **Quantifying epigenomic/functional annotation enrichment in GWAS and re-weighting causal fine-mapping posteriors (`PPA`)**: [`fgwas`](../fgwas/README.md) (`fit_fgwas`, `cross_validate_fgwas`, `stepwise_fgwas`; Pickrell, 2014).
- **Jointly shrinking and quantifying effect sharing across multi-tissue xQTL or multi-trait summary statistics**: [`mashr`](../mashr/README.md) (`mash_set_data`, `cov_canonical`, `cov_pca`, `cov_ed`, `mash`, `get_pairwise_sharing`, `get_lfsr`; Urbut et al., 2019).

---

## 2. Conditional GWAS & Collider / Index-Event Bias Adjustment

### *How do I adjust a conditional or subsequent-event GWAS (e.g., prognosis, progression, or covariate-adjusted trait) for collider / index-event bias?*
- **Discriminative mixture clustering separating pure-index SNPs from pleiotropic SNPs**: [`slopehunter`](../slopehunter/README.md) (`hunt`, `sh_adj`; Mahmoud et al., 2022).
- **Regression-dilution-corrected genome-wide slope (`CWLS`, `Hedges-Olkin`, `SIMEX`)**: [`indexevent`](../indexevent/README.md) (`indexevent`; Dudbridge et al., 2019).
- **Instrumental-effect `CWLS` (`ivw`/`egger`) or 3-sample exposure–collider–outcome `CWBLS`**: [`colliderbias`](../colliderbias/README.md) (`cwls`, `cwbls`; Cai et al., 2022; Cai & Dudbridge, 2024).
- **Multi-trait summary conditioning with bivariate LDSC sample-overlap correction, `.bim` allele alignment, and `GSMR` + `HEIDI-outlier` filtering**: [`mtcojo`](../mtcojo/README.md) (`mtcojo`, `align_sumstats_to_bim`; Zhu et al., 2018).

---

## 3. Differential Expression, Isoform DTU, Mixed Models, Shrinkage & Clinical Survival Biostatistics

- **Small-sample continuous differential expression / proteomics / CRISPR screen moderation (`eBayes`, `limma-trend`, `robust=True`)**: [`limma`](../limma/README.md) (`moderated_ttest`, `moderated_lm_contrast`, `limma_trend`, `limma_top_table`; Smyth, 2004; Phipson et al., 2016).
- **Differential transcript, isoform, or exon usage (`DTU`) in bulk or sparse scRNA-seq with quasi-binomial Empirical Bayes moderation, `locfdr` empirical null calibration, and two-stage `stageR` OFDR**: [`saturn`](../saturn/README.md) (`filter_dtu`, `fit_dtu`, `test_dtu`, `stagewise_dtu`; Gilis et al., 2022).
- **Estimating latent technical/batch confounding factors (`irwsva`, `twostepsva`, `svaseq`, `num_sv`) and adjusting nested linear F-tests**: [`sva`](../sva/README.md) (`num_sv`, `sva`, `irwsva`, `twostepsva`, `svaseq`, `f_pvalue`; Leek & Storey, 2007; Leek, 2014).
- **Decomposing gene-expression variance across fixed and random covariates (`ICC`) and testing repeated-measures mixed-model contrasts with Satterthwaite df and Empirical Bayes moderation (`dream`)**: [`variancepartition`](../variancepartition/README.md) (`fit_extract_var_part_model`, `fit_var_part_model`, `dream`, `e_bayes`, `top_table`; Hoffman & Schadt, 2016; Hoffman & Roussos, 2021).
- **Fast multi-subject Negative Binomial mixed-model differential expression and overdispersion estimation on sparse single-cell `AnnData` (`NEBULA-LN` / `NEBULA-HL`)**: [`nebula`](../nebula/README.md) (`nebula`, `group_cell`; He et al., 2021).
- **Data-driven multivariate adaptive shrinkage across $R$ conditions (treatments, tissues, cell types) with Extreme Deconvolution priors, `lfsr`, and pairwise sharing**: [`mashr`](../mashr/README.md) (`mash_set_data`, `estimate_null_correlation_simple`, `cov_canonical`, `cov_pca`, `cov_ed`, `mash`, `get_significant_results`, `get_pairwise_sharing`; Urbut et al., 2019).
- **Binary outcomes with rare events or complete/quasi-complete separation (Profile Penalized Likelihood CIs & PLRT)**: [`logistf`](../logistf/README.md) (`fit_firth_logistic`; Firth, 1993; Heinze & Schemper, 2002).
- **Time-to-event Cox PH (`Efron` ties), time-varying IPCW Fine–Gray competing risks (`cmprsk`), heteroskedastic WLS (`HC3`), pooled-null permutation FDR, or Storey cubic-spline `qvalue`**: [`biostats`](../biostats/README.md) (`fit_cox_ph_efron`, `fit_fine_gray_ipcw`, `fit_wls_hc3`, `storey_qvalue`).
- **Evaluating and comparing continuous prognostic biomarkers across time horizons under right-censoring and competing risks**: [`timeroc`](../timeroc/README.md) (`time_roc`, `compare_timeroc`; Blanche et al., 2013).
- **Quantifying incremental prognostic reclassification (`IDI`, continuous/categorical `NRI`, median risk improvement) when adding genomic biomarkers to clinical Cox models under right-censoring**: [`survidinri`](../survidinri/README.md) (`idi_inf`, `idi_inf_predictions`; Uno et al., 2013).
- **Analyzing hierarchical prioritized composite clinical-trial endpoints (e.g., OS $\succ$ PFS $\succ$ Toxicity / QoL) via Win Ratio, Net Benefit, Win Odds (with Péron KM censoring correction), or covariate-adjusted Proportional Win-Fractions Regression (`pwreg`)**: [`buysetest`](../buysetest/README.md) (`buyse_test`, `Endpoint`, `pwreg`; Péron et al., 2016; Mao & Wang, 2020).

---

## 4. Bulk & Spatial Deconvolution, Xenograft Filtering, Methylation & Chromatin Regulation

- **Removing mouse host reads from xenograft WGS/WES/ATAC/ChIP/RNA BAMs**: [`xenofilterR`](../xenofilterR/README.md) (`run_xenofilter`, `Xenofilter`) locally or [`xenofilter_bam.wdl`](../wdl-workflows/xenofilter_bam.wdl) on Terra/Cromwell (Kluin et al., 2018).
- **Estimating cell-type fractions from bulk RNA-seq with a multi-donor scRNA-seq reference**: [`music`](../music/README.md) (`music_prop`, `music2_prop`, `music_prop_cluster`; Wang et al., 2019; Fan et al., 2022).
- **Cross-platform bulk RNA-seq deconvolution learning gene-specific affine domain alignments from single-cell pseudobulks (with or without overlapping donors), or marker-gene PCA scoring**: [`bisque`](../bisque/README.md) (`reference_based_decomposition`, `marker_based_decomposition`, `generate_sc_reference`; Jew et al., 2020).
- **Jointly estimating cell-type fractions, sample-specific deconvolved expression matrices, and spatial spot profiles**: [`bayesprism`](../bayesprism/README.md) (`Prism.run`, `Prism.run_st`; Chu et al., 2022).
- **Signature matrix selection, linear $\nu$-SVR bulk deconvolution, `B-mode`/`S-mode` batch alignment, Group-Mode GEP estimation, and High-Resolution sample-specific purification** *(clean-room implementation)*: [`cibersortx`](../cibersortx/README.md) (`build_signature_matrix`, `deconvolve`, `cibersort`, `b_mode_adjust`, `s_mode_adjust`, `impute_gep_group`, `impute_gep_highres`; Newman et al., 2015, 2019).
- **Cell-type-specific differential expression across spatial transcriptomics gradients, anatomic niches, or tumor boundaries given spot cell-type weights**: [`cside`](../cside/README.md) (`run_cside`, `run_cside_single`, `run_cside_regions`, `run_cside_nonparam`; Cable et al., 2022).
- **Bisulfite / EM-seq differential DNA methylation (`DML` & `DMR`) with BSmooth smoothing, Beta-Binomial Empirical Bayes dispersion shrinkage, multifactor `DSS-general` GLS, and `DMRcate` kernel smoothing**: [`dss`](../dss/README.md) (`dml_test`, `call_dml`, `call_dmr`, `dml_fit_multifactor`, `dml_test_multifactor`, `dmrcate`; Wu et al., 2013; Park & Wu, 2016; Peters et al., 2015).
- **Peak-free sliding-window differential ChIP-seq / ATAC-seq / CUT&RUN binding with TMM/loess normalization, NB Quasi-Likelihood F-tests, and Simes region-level FDR**: [`csaw`](../csaw/README.md) (`window_counts`, `filter_windows_global`, `norm_factors`, `norm_offsets`, `glm_ql_fit_test`, `merge_windows`, `combine_tests`; Lun & Smyth, 2016).
- **Calling super-enhancers from H3K27ac ChIP-seq or ATAC-seq peaks (with optional Input/IgG bigWig subtraction)**: [`rose`](../rose/README.md) (`call_rose_super_enhancers`; Whyte et al., 2013; Lovén et al., 2013).
- **Inferring sparse-capable GC- and accessibility-corrected TF motif deviations & variability from raw fragment counts**: [`chromvar`](../chromvar/README.md) (`compute_expectations`, `get_background_peaks`, `compute_deviations`, `compute_variability`; Schep et al., 2017).
- **Inferring cis peak-to-gene links, Domains of Regulatory Chromatin (DORCs), kNN-smoothed DORC activity scores, and TF-DORC gene regulatory networks from paired scATAC+scRNA multiome data**: [`figr`](../figr/README.md) (`run_gene_peak_corr`, `call_dorcs`, `get_dorc_scores`, `smooth_scores_knn`, `run_figr_grn`; Kartha et al., 2022).
- **Scoring continuous NMF / gene-set factor weights via Univariate Linear Models**: [`decoupler-ulm`](../decoupler-ulm/README.md) (`project_gep_programs`; Badia-i-Mompel et al., 2022).

---

## 5. Somatic Cancer Genomics, Liquid Biopsy, Mutational Signatures, Scars & Cohort Curation

- **Calling allele-specific somatic copy number, tumor purity, ploidy, LOH, and clonal cellular fractions from matched tumor-normal SNP pileups**: [`facets`](../facets/README.md) (`run_facets`, `pre_proc_sample`, `proc_sample`, `em_cncf`, `fit_ascat`, `compare_facets_ascat`; Shen & Seshan, 2016; Van Loo et al., 2010).
- **Inferring purity, ploidy, somatic-vs-germline status, LOH, CCF, and artifact-cleaned TMB/CIN from tumor-only clinical targeted panels**: [`purecn`](../purecn/README.md) (`normalize_tumor_only_coverage`, `grid_search_purity_ploidy`, `predict_somatic`, `call_tmb`, `run_purecn`; Riester et al., 2016).
- **Calling copy-number profiles and reference-free cellularity/ploidy (`ACE`) from ultra-low-pass WGS (`lpWGS`) and cfDNA liquid biopsies**: [`qdnaseq`](../qdnaseq/README.md) (`bin_read_counts`, `estimate_correction`, `segment_bins`, `call_bins`, `run_ace`; Scheinin et al., 2014; Poell et al., 2019).
- **Detecting evolutionary changepoints in mutational signature activities across CCF pseudotime (`PELT`)**: [`tracksig`](../tracksig/README.md) (`compute_ccf_pseudotime`, `tracksig`, `TrackSig`; Rubanova et al., 2020).
- **Quantifying gene-level and exome-wide somatic selection ($dN/dS$ for missense, nonsense, splice, and indels) with 192-trinucleotide rates and epigenomic covariates**: [`dndscv`](../dndscv/README.md) (`build_refcds`, `build_refcds_from_gtf`, `simulate_refcds`, `dndscv`, `fit_dndscv`; Martincorena et al., 2017).
- **Calling haplotype-aware allele-specific single-cell CNVs, CN-neutral LOH, malignant vs normal cells, and single-cell ML tumor phylogenies from scRNA-seq**: [`numbat`](../numbat/README.md) (`run_numbat`, `run_joint_hmm`, `reconstruct_phylogeny`, `score_single_cells`, `annotate_anndata`; Gao et al., 2023).
- **Testing somatic mutual exclusivity and co-occurrence while controlling for tumor-specific mutation burden (`NaN`-panel-aware)**: [`discover`](../discover/README.md) (`fit_discover_background`, `pairwise_discover_test`, `groupwise_discover_test`; Canisius et al., 2016).
- **Calling noncoding or interval-level somatic driver enrichment adjusting for callable territory, replication timing, and chromatin covariates**: [`fishhook`](../fishhook/README.md) (`fit_fishhook`; Imielinski et al., 2017).
- **Detecting and classifying chromothripsis from interleaved structural variants and oscillating copy-number states**: [`shatterseek`](../shatterseek/README.md) (`detect_chromothripsis`, `classify_chromothripsis`; Cortés-Ciriano et al., 2020).
- **Computing copy-number HRD genomic scars (`HRD-LOH`, `TAI`, `LST`, `HRD-sum`) OR mutational-context `BRCA1`- vs. `BRCA2`-type HRD probabilities**: [`scarhrd`](../scarhrd/README.md) (`calculate_hrd`, `scar_score`; Sztupinszki et al., 2018) and [`chord`](../chord/README.md) (`extract_sigs_chord`, `chord_predict`; Nguyen et al., 2020).
- **Inferring cross-sectional mutation trees, BIC-penalized OR/AND Bayesian networks, or Mutual Hazard Networks**: [`evam`](../evam/README.md) (`fit_ot`, `fit_oncobn`, `fit_cbn`, `fit_mhn`; Diaz-Uriarte & Herrera-Nieto, 2022).
- **Ingesting `cBioPortal` studies, parsing `GISTIC2.0` peaks, masking unsequenced `AACR GENIE` panel genes as `NaN`, repairing spreadsheet-corrupted `HGNC` symbols, and parsing `Cellosaurus`**: [`cohort-curation`](../cohort-curation/README.md).

---

## 6. Recommended Multi-Tool Workflows

- **Matched-normal WGS/WES oncology pipeline**: Filter xenograft host reads with [`xenofilterR`](../xenofilterR/README.md) $\to$ call purity/ploidy/allele-specific CNVs with [`facets`](../facets/README.md) $\to$ combine copy-number HRD scars ([`scarhrd`](../scarhrd/README.md)) with `BRCA1`- vs. `BRCA2`-type SNV/indel/SV subtyping ([`chord`](../chord/README.md)) and chromothripsis ([`shatterseek`](../shatterseek/README.md)) $\to$ segment evolutionary signature shifts with [`tracksig`](../tracksig/README.md) $\to$ identify drivers and mutual exclusivity with [`dndscv`](../dndscv/README.md), [`discover`](../discover/README.md), and [`evam`](../evam/README.md).
- **Clinical tumor-only panel & cfDNA liquid-biopsy pipeline**: Curate `AACR GENIE` / `cBioPortal` panel matrices with [`cohort-curation`](../cohort-curation/README.md), call tumor-only purity/ploidy, somatic status, LOH, and TMB with [`purecn`](../purecn/README.md), and track serial cfDNA `lpWGS` copy-number burden & cellularity with [`qdnaseq`](../qdnaseq/README.md) (`ACE`) $\to$ evaluate clinical trial composite survival benefit with [`timeroc`](../timeroc/README.md), [`survidinri`](../survidinri/README.md), and [`buysetest`](../buysetest/README.md).
- **Spatial & multi-subject single-cell transcriptomics**: Deconvolve spatial spot mixtures with [`bayesprism`](../bayesprism/README.md) (`run_st`) $\to$ detect niche-dependent cell-type-specific spatial DE with [`cside`](../cside/README.md); for cohort scRNA-seq, test gene-level counts with [`nebula`](../nebula/README.md), isoform switches with [`saturn`](../saturn/README.md), and cross-condition effect sharing with [`mashr`](../mashr/README.md).
- **Drug-target genetics & causal fine-mapping**: Estimate global/local genetic correlation with [`hdl`](../hdl/README.md), integrate enhancer priors from [`rose`](../rose/README.md)/[`csaw`](../csaw/README.md) via [`fgwas`](../fgwas/README.md), separate causal molecular traits from LD hitchhiking with [`ctwas`](../ctwas/README.md) and [`smr-heidi`](../smr-heidi/README.md), validate multi-causal proportional colocalization with [`colocproptest`](../colocproptest/README.md) and [`hyprcoloc`](../hyprcoloc/README.md), and estimate cis-drug-target causal effects across correlated variants with [`cismr`](../cismr/README.md).
