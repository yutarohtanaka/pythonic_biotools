# Problem-Driven Guide: Using Published Tools in `pythonic_biotools`

Every directory in `pythonic_biotools` is a standalone Python implementation of a published bioinformatics, oncology, or statistical genetics tool. Use the lookup below to match your analysis question to the appropriate library.

---

## 1. Mendelian Randomization, Colocalization & Molecular QTL Mediation

### *Does an exposure causally influence an outcome in two-sample or multivariable summary statistics?*
- **Standard `IVW`, coding-invariant `MR-Egger`, `Weighted Median`, `MVMR` (Sanderson–Windmeijer conditional $F$), allele harmonization, and LD clumping**: [`twosamplemr`](../twosamplemr/README.md) (`run_mr_suite`, `harmonise_mr_data`, `select_mr_instruments`).
- **Testing for global horizontal pleiotropy, identifying specific outlier SNPs, and testing pre- vs post-outlier causal distortion (univariable or multivariable)**: [`mrpresso`](../mrpresso/README.md) (`mr_presso`, `mvmr_presso`; Verbanck et al., 2018).
- **Many weak instruments with systematic pleiotropic overdispersion ($\tau^2$) or heavy-tailed outliers**: [`mrraps`](../mrraps/README.md) (`mr_raps` with `huber` or `tukey` loss; Zhao et al., 2020).
- **Correlated + uncorrelated horizontal pleiotropy (univariable or multivariable)**: [`mrhorse`](../mrhorse/README.md) (`mr_horse`, `mvmr_horse`; Grant & Burgess, 2024).
- **Polygenic background pleiotropy + instrument-selection truncation (`|z_X| > t_0`) + sample overlap**: [`mrapss`](../mrapss/README.md) (`fit_mrapss`; Hu et al., 2022) or [`mrbee`](../mrbee/README.md) (`fit_mrbee`, `fit_mvmr_bee`; Lorincz-Comi et al., 2024).
- **Multivariable constrained maximum likelihood with BIC Model Averaging and Data Perturbation (`cML-DP`)**: [`mvcml`](../mvcml/README.md) (`cml_ma`, `mvcml_dp`, `mvcml_adjust`; Lin et al., 2023).

### *Do two or more traits (GWAS, eQTL, pQTL, sQTL) share a causal variant vs. distinct LD-linked variants at a locus?*
- **Two-trait colocalization (`PP.H0`–`PP.H4` and 95% credible sets)**: [`coloc`](../coloc/README.md) (`coloc_abf`; Wakefield 2009; Giambartolomei et al., 2014).
- **Multi-trait colocalization across 3–100+ traits simultaneously with branch-and-bound cluster discovery**: [`hyprcoloc`](../hyprcoloc/README.md) (`hyprcoloc`, `sensitivity_analysis`; Foley et al., 2021).
- **Testing cis-xQTL mediation (`SMR` / `SMR-multi`) vs. linkage of distinct causal variants (`HEIDI`) using local LD**: [`smr-heidi`](../smr-heidi/README.md) (`run_locus_smr_heidi`, `smr_single`, `smr_multi`, `heidi_test`; Zhu et al., 2016; Wu et al., 2018).

---

## 2. Conditional GWAS & Collider / Index-Event Bias Adjustment

### *How do I adjust a conditional or subsequent-event GWAS (e.g., prognosis, progression, or covariate-adjusted trait) for collider / index-event bias?*
- **Discriminative mixture clustering separating pure-index SNPs from pleiotropic SNPs**: [`slopehunter`](../slopehunter/README.md) (`hunt`, `sh_adj`; Mahmoud et al., 2022).
- **Regression-dilution-corrected genome-wide slope (`CWLS`, `Hedges-Olkin`, `SIMEX`)**: [`indexevent`](../indexevent/README.md) (`indexevent`; Dudbridge et al., 2019).
- **Instrumental-effect `CWLS` (`ivw`/`egger`) or 3-sample exposure–collider–outcome `CWBLS`**: [`colliderbias`](../colliderbias/README.md) (`cwls`, `cwbls`; Cai et al., 2022; Cai & Dudbridge, 2024).
- **Multi-trait summary conditioning with bivariate LDSC sample-overlap correction, `.bim` allele alignment, and `GSMR` + `HEIDI-outlier` filtering**: [`mtcojo`](../mtcojo/README.md) (`mtcojo`, `align_sumstats_to_bim`; Zhu et al., 2018).

---

## 3. Differential Expression, Penalized Regression & Competing-Risks Survival

- **Small-sample continuous differential expression / proteomics / CRISPR screen moderation (`eBayes`, `limma-trend`, `robust=True`)**: [`limma`](../limma/README.md) (`moderated_ttest`, `moderated_lm_contrast`, `limma_trend`, `limma_top_table`; Smyth, 2004; Phipson et al., 2016).
- **Binary outcomes with rare events or complete/quasi-complete separation (Profile Penalized Likelihood CIs & PLRT)**: [`logistf`](../logistf/README.md) (`fit_firth_logistic`; Firth, 1993; Heinze & Schemper, 2002).
- **Time-to-event Cox PH (`Efron` ties), time-varying IPCW Fine–Gray competing risks (`cmprsk`), heteroskedastic WLS (`HC3`), pooled-null permutation FDR, or Storey cubic-spline `qvalue`**: [`biostats`](../biostats/README.md) (`fit_cox_ph_efron`, `fit_fine_gray_ipcw`, `fit_wls_hc3`, `storey_qvalue`).
- **Evaluating and comparing continuous prognostic biomarkers across time horizons under right-censoring and competing risks**: [`timeroc`](../timeroc/README.md) (`time_roc`, `compare_timeroc`; Blanche et al., 2013).

---

## 4. Bulk Deconvolution, Xenograft Filtering & Epigenomics

- **Removing mouse host reads from xenograft WGS/WES/ATAC/ChIP/RNA BAMs**: [`xenofilterR`](../xenofilterR/README.md) (`run_xenofilter`) locally or [`xenofilter_bam.wdl`](../wdl-workflows/xenofilter_bam.wdl) on Terra/Cromwell (Kluin et al., 2018).
- **Estimating cell-type fractions from bulk RNA-seq with a multi-donor scRNA-seq reference**: [`music`](../music/README.md) (`music_prop`, `music2_prop`, `music_prop_cluster`; Wang et al., 2019; Fan et al., 2022).
- **Jointly estimating cell-type fractions, sample-specific deconvolved expression matrices, and spatial spot profiles**: [`bayesprism`](../bayesprism/README.md) (`Prism.run`, `Prism.run_st`; Chu et al., 2022).
- **Calling super-enhancers from H3K27ac ChIP-seq or ATAC-seq peaks (with optional Input/IgG bigWig subtraction)**: [`rose`](../rose/README.md) (`call_rose_super_enhancers`; Whyte et al., 2013; Lovén et al., 2013).
- **Inferring sparse-capable GC- and accessibility-corrected TF motif deviations & variability from raw fragment counts**: [`chromvar`](../chromvar/README.md) (`compute_expectations`, `get_background_peaks`, `compute_deviations`, `compute_variability`; Schep et al., 2017).
- **Scoring continuous NMF / gene-set factor weights via Univariate Linear Models**: [`decoupler-ulm`](../decoupler-ulm/README.md) (`project_gep_programs`; Badia-i-Mompel et al., 2022).

---

## 5. Somatic Cancer Genomics, Structural Scars, Progression & Cohort Curation

- **Testing somatic mutual exclusivity and co-occurrence while controlling for tumor-specific mutation burden (`NaN`-panel-aware)**: [`discover`](../discover/README.md) (`fit_discover_background`, `pairwise_discover_test`, `groupwise_discover_test`; Canisius et al., 2016).
- **Calling noncoding or interval-level somatic driver enrichment adjusting for callable territory, replication timing, and chromatin covariates**: [`fishhook`](../fishhook/README.md) (`fit_fishhook`; Imielinski et al., 2017).
- **Detecting and classifying chromothripsis from interleaved structural variants and oscillating copy-number states**: [`shatterseek`](../shatterseek/README.md) (`detect_chromothripsis`, `classify_chromothripsis`; Cortés-Ciriano et al., 2020).
- **Computing Homologous Recombination Deficiency genomic scars (`HRD-LOH`, `TAI`, `LST`, `HRD-sum`) from allele-specific copy number**: [`scarhrd`](../scarhrd/README.md) (`calculate_hrd`, `calculate_hrd_cohort`; Sztupinszki et al., 2018).
- **Inferring cross-sectional mutation trees, BIC-penalized OR/AND Bayesian networks, or Mutual Hazard Networks**: [`evam`](../evam/README.md) (`fit_ot`, `fit_oncobn`, `fit_cbn`, `fit_mhn`; Diaz-Uriarte & Herrera-Nieto, 2022).
- **Ingesting `cBioPortal` studies, parsing `GISTIC2.0` peaks, masking unsequenced `AACR GENIE` panel genes as `NaN`, repairing spreadsheet-corrupted `HGNC` symbols, and parsing `Cellosaurus`**: [`cohort-curation`](../cohort-curation/README.md).
