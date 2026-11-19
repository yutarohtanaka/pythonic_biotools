# Problem-Driven Guide: Using Published Tools in `pythonic_biotools`

Every directory in `pythonic_biotools` is a standalone Python implementation of a published bioinformatics or statistical genetics tool. Use the lookup below to match your analysis question to the appropriate library.

---

## 1. Mendelian Randomization & Genetic Colocalization

### *Does an exposure causally influence an outcome in two-sample or multivariable summary statistics?*
- **Standard `IVW`, coding-invariant `MR-Egger`, `Weighted Median`, `MVMR`, and LD clumping**: [`twosamplemr`](../twosamplemr/README.md) (`run_mr_suite`, `select_mr_instruments`).
- **Many weak instruments with systematic pleiotropic overdispersion ($\tau^2$) or outliers**: [`mrraps`](../mrraps/README.md) (`mr_raps` with `huber` or `tukey` loss; Zhao et al., 2020).
- **Correlated + uncorrelated horizontal pleiotropy (univariable or multivariable)**: [`mrhorse`](../mrhorse/README.md) (`mr_horse`, `mvmr_horse`; Grant & Burgess, 2024).
- **Polygenic background pleiotropy + instrument-selection truncation (`|z_X| > t_0`) + sample overlap**: [`mrapss`](../mrapss/README.md) (`fit_mrapss`; Hu et al., 2022) or [`mrbee`](../mrbee/README.md) (`fit_mrbee`; Lorincz-Comi et al., 2024).
- **Multivariable constrained maximum likelihood with BIC invalid-IV selection**: [`mvcml`](../mvcml/README.md) (`cml_ma`, `mvcml_adjust`; Lin et al., 2023).

### *Do two traits (e.g., GWAS + eQTL/pQTL) share a single causal variant at a locus?*
- **Use**: [`coloc`](../coloc/README.md) (`coloc_abf`; Wakefield 2009; Giambartolomei et al., 2014), computing posterior probabilities `PP.H0`–`PP.H4` and 95% credible sets in numerically stable log-space.

---

## 2. Conditional GWAS & Collider / Index-Event Bias Adjustment

### *How do I adjust a conditional or subsequent-event GWAS (e.g., prognosis, progression, or covariate-adjusted trait) for collider / index-event bias?*
- **Discriminative mixture clustering separating pure-index SNPs from pleiotropic SNPs**: [`slopehunter`](../slopehunter/README.md) (`hunt`, `sh_adj`; Mahmoud et al., 2022).
- **Regression-dilution-corrected genome-wide slope (`CWLS`, `Hedges-Olkin`, `SIMEX`)**: [`indexevent`](../indexevent/README.md) (`indexevent`; Dudbridge et al., 2019).
- **Instrumental-effect `CWLS` (`ivw`/`egger`) or 3-sample exposure–collider–outcome `CWBLS`**: [`colliderbias`](../colliderbias/README.md) (`cwls`, `cwbls`; Cai et al., 2022; Cai & Dudbridge, 2024).
- **Multi-trait summary conditioning with bivariate LDSC sample-overlap correction and `GSMR` + `HEIDI-outlier` filtering**: [`mtcojo`](../mtcojo/README.md) (`mtcojo`; Zhu et al., 2018).

---

## 3. Differential Expression, Penalized Regression & Survival Analysis

- **Small-sample continuous differential expression / proteomics / CRISPR screen moderation**: [`limma`](../limma/README.md) (`moderated_ttest`, `moderated_lm_contrast`, `limma_top_table`; Smyth, 2004).
- **Binary outcomes with rare events or complete/quasi-complete separation**: [`logistf`](../logistf/README.md) (`fit_firth_logistic`; Firth, 1993; Heinze & Schemper, 2002).
- **Time-to-event Cox PH (`Efron` ties), Fine–Gray IPCW competing risks (`cmprsk`), heteroskedastic WLS (`HC3`), or vectorized genome-wide correlation scans with pooled-null permutation FDR**: [`biostats`](../biostats/README.md).

---

## 4. Bulk Deconvolution, Xenograft Filtering & Epigenomics

- **Removing mouse host reads from xenograft WGS/WES/ATAC/ChIP/RNA BAMs**: [`xenofilterR`](../xenofilterR/README.md) locally or [`xenofilter_bam.wdl`](../wdl-workflows/xenofilter_bam.wdl) on Terra/Cromwell (Kluin et al., 2018).
- **Estimating cell-type fractions from bulk RNA-seq with a multi-donor scRNA-seq reference**: [`music`](../music/README.md) (`music_prop`, `music2_prop`, `music_prop_cluster`; Wang et al., 2019; Fan et al., 2022).
- **Jointly estimating cell-type fractions and sample-specific deconvolved expression matrices**: [`bayesprism`](../bayesprism/pythonic_bayesprism/pythonic_bayesprism.md) (Chu et al., 2022).
- **Calling super-enhancers from H3K27ac ChIP-seq or ATAC-seq peaks**: [`rose`](../rose/README.md) (`call_rose_super_enhancers`; Whyte et al., 2013; Lovén et al., 2013).
- **Inferring GC- and accessibility-corrected TF motif deviations & variability from raw fragment counts**: [`chromvar`](../chromvar/README.md) (`compute_expectations`, `get_background_peaks`, `compute_deviations`, `compute_variability`; Schep et al., 2017).
- **Scoring continuous NMF / gene-set factor weights in bulk or pseudobulk matrices via Univariate Linear Models**: [`decoupler-ulm`](../decoupler-ulm/README.md) (`project_gep_programs`; Badia-i-Mompel et al., 2022).

---

## 5. Cross-Sectional Cancer Progression & Public Genomics Curation

- **Inferring cross-sectional mutation trees, OR/AND Bayesian networks, or Mutual Hazard Networks**: [`evam`](../evam/README.md) (`fit_ot`, `fit_oncobn`, `fit_cbn`, `fit_mhn`; Diaz-Uriarte & Herrera-Nieto, 2022).
- **Ingesting public `cBioPortal` studies, parsing Broad `GISTIC2.0` peaks, masking unsequenced `AACR GENIE` panel genes as `NaN`, repairing spreadsheet-corrupted `HGNC` symbols, and parsing `Cellosaurus` cell-line metadata**: [`cohort-curation`](../cohort-curation/README.md).
