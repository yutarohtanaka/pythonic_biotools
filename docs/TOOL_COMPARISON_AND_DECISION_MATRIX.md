# Tool Comparison & Decision Matrix

Use the comparison tables below to choose among published methods implemented across all 57 standalone Python packages in `pythonic_biotools`.

---

## 1. Two-Sample, Multivariable & Correlated `cis`-Locus Mendelian Randomization

| Package | Published Method | Instrument Regime | Pleiotropy Assumption | Weak-IV / Overlap Robust? | Multivariable? |
|---|---|---|---|---|---|
| **[`twosamplemr`](../twosamplemr/README.md)** | `IVW`, `MR-Egger`, `Weighted Median`, `MVMR` | Independent genome-wide | Balanced (`IVW`), InSIDE (`Egger`), or ≥50% weight (`WM`) | Requires $F > 10$ | **Yes** (`multivariable_mr`) |
| **[`cismr`](../cismr/README.md)** | `PC-GMM`, `MV-PC-GMM`, `MR-SPI`, `MR-GENIUS` | **Correlated cis-locus ($r^2 \le 0.99$)** | Sargan $J$ overid (`PC-GMM`), plurality clique (`MR-SPI`), or additive pleiotropy via vQTL (`MR-GENIUS`) | **Yes** (LD-PC projection) | **Yes** (`mr_mvpcgmm`) |
| **[`mrpresso`](../mrpresso/README.md)** | `MR-PRESSO` / `MVMR-PRESSO` (Verbanck et al., 2018) | Independent genome-wide | Sparse horizontal pleiotropic outliers (Global, Outlier & Distortion tests) | Requires $F > 10$ | **Yes** (`mvmr_presso`) |
| **[`mrraps`](../mrraps/README.md)** | `mr.raps` (Zhao et al., 2020) | Independent genome-wide | Overdispersed $\tau^2$ + Huber/Tukey outliers | **Yes** (profile score) | No |
| **[`mrhorse`](../mrhorse/README.md)** | `MR-Horse` / `MVMR-Horse` (Grant & Burgess, 2024) | Independent genome-wide | **Correlated + uncorrelated** Horseshoe prior | Yes (latent exposure) | **Yes** (`mvmr_horse`) |
| **[`mrapss`](../mrapss/README.md)** | `MR-APSS` (Hu et al., 2022) | Clumped (`|z_X| > t_0`) | Foreground–background mixture + LDSC $\Omega$ | **Yes** (LDSC intercept $C$) | No |
| **[`mrbee`](../mrbee/README.md)** | `MRBEE` (Lorincz-Comi et al., 2024) | Independent genome-wide | IMRP Sidak outlier removal + $\tau^2$ | **Yes** ($R_{xy}$ subtracted) | **Yes** (`fit_mvmr_bee`) |
| **[`mvcml`](../mvcml/README.md)** | `MVMR-cML` (Lin et al., 2023) | Independent genome-wide | BIC Model Averaging + Data Perturbation (`cML-DP`) | **Yes** (`rho_yx` matrix) | **Yes** (`cml_ma`, `mvcml_dp`) |
| **[`cause`](../cause/README.md)** | `CAUSE` (Morrison et al., 2020) | Genome-wide + LD-pruned | Correlated ($\eta, q$) + uncorrelated bivariate `ash` (`PSIS-LOO`) | **Yes** (null $\hat{\rho}$ MLE) | No |
| **[`mrclust`](../mrclust/README.md)** | `MR-Clust` (Foley et al., 2021) | Independent genome-wide | $K$ causal pathway clusters + Null + Student-$t$ Junk | Requires $F > 10$ | No |

---

## 2. Colocalization, Causal TWAS, Genetic Correlation & Functional Fine-Mapping

| Package | Published Method | Scope & Traits | LD Matrix Needed? | Primary Output |
|---|---|---|---|---|
| **[`coloc`](../coloc/README.md)** | `coloc.abf` (Giambartolomei et al., 2014) | 2 traits (single causal variant) | No | Exact `PP.H0`–`PP.H4` & 95% credible set |
| **[`hyprcoloc`](../hyprcoloc/README.md)** | `HyPrColoc` (Foley et al., 2021) | **$K \ge 2$ traits** ($O(MK)$ up to 100+ traits) | Optional (`ld_matrix`, `trait_cor`) | Colocalized trait clusters, `PP = P_R * P_A`, candidate SNP |
| **[`colocproptest`](../colocproptest/README.md)** | `colocPropTest` (Wallace, 2023) | 2 traits (**multiple causal variants safe**) | Out-of-sample LD safe | Frequentist proportionality $p$/FDR across tester pairs + LD-PC omnibus & $\hat{\eta}$ CI |
| **[`smr-heidi`](../smr-heidi/README.md)** | `SMR`, `SMR-multi` & `HEIDI` (Zhu et al., 2016; Wu et al., 2018) | cis-xQTL + GWAS | **Yes** (`HEIDI` & `SMR-multi`) | Causal mediation $\hat{b}_{xy}$, $p_{\text{SMR}}$, and $p_{\text{HEIDI}}$ (pleiotropy vs. linkage) |
| **[`ctwas`](../ctwas/README.md)** | `cTWAS` (Zhao et al., 2024) | Multi-gene cis-weights + direct cis-SNPs + GWAS | **Yes** | Empirical Bayes $(\pi_G, \pi_S, \sigma_G^2, \sigma_S^2)$, mediated $h_G^2$, & joint SuSiE-RSS PIPs |
| **[`hdl`](../hdl/README.md)** | `HDL`, `HDL-L` & `HDL.S` (Ning et al., 2020) | Genome-wide, single-locus, or stratified GWAS | **Yes** (regularized LD eigenpanels) | High-precision global $r_g$, local $r_{g,\text{loc}}$, and annotation-partitioned $h^2$ |
| **[`fgwas`](../fgwas/README.md)** | `fgwas` (Pickrell, 2014) | Genome-wide GWAS + epigenomic annotations | Block windows or `-fine` regions | Log-enrichment $\hat{\gamma}_a$ (profile CIs, `-xv` ridge) & re-weighted SNP `PPA` |

---

## 3. Conditional GWAS & Index-Event / Collider-Bias Adjustment

| Package | Published Method | Core Slope Assumption | Handles Direct Pleiotropic Loci? | LD Reference Needed? |
|---|---|---|---|---|
| **[`slopehunter`](../slopehunter/README.md)** | `SlopeHunter` (Mahmoud et al., 2022) | Two-component EM mixture ("Hunted" vs. "Pleiotropic") | **Yes** | LD-pruned SNPs |
| **[`indexevent`](../indexevent/README.md)** | `indexevent` (Dudbridge et al., 2019) | Orthogonal direct effects (`CWLS`, `Hedges-Olkin`, `SIMEX`) | Orthogonal assumption | LD-pruned SNPs |
| **[`colliderbias`](../colliderbias/README.md)** | `ColliderBias` (Cai et al., 2022; Cai & Dudbridge, 2024) | Instrumental `CWLS` (`ivw`/`egger`) or 3-trait `CWBLS` | Via Egger or 3-trait `CWBLS` | LD-pruned SNPs |
| **[`mtcojo`](../mtcojo/README.md)** | `GCTA-mtCOJO` & `GSMR` (Zhu et al., 2018) | GSMR v2 causal effect $b_{xy}$ + bivariate LDSC | **Yes** (global `HEIDI-outlier`) | **Yes** (unless `bxy_external` given) |

---

## 4. Bulk Deconvolution & Spatial Cell-Type Differential Expression

| Package | Published Method | Statistical Engine | Batch / Spatial Modeling | Primary Output & When to Prefer |
|---|---|---|---|---|
| **[`music`](../music/README.md)** | `MuSiC` / `MuSiC2` (Wang et al., 2019; Fan et al., 2022) | Cross-subject variance-weighted IRLS NNLS | Cross-subject gene weighting | Fast proportion estimation across multi-donor references or case/control shifts |
| **[`bayesprism`](../bayesprism/README.md)** | `BayesPrism` (Chu et al., 2022) | Multinomial–Dirichlet Gibbs + `L-BFGS-B` MAP | Reference cell-state prior adaptation + `run_st` | Proportions **and** patient-specific cell-type expression $Z_{n,g,k}$ + spatial spot weights |
| **[`bisque`](../bisque/README.md)** | `Bisque` (Jew et al., 2020) | Affine pseudobulk-to-bulk alignment + NNLS / PCA | Learns gene-specific linear platform transform | Strong scRNA-seq vs. bulk platform bias, or marker-only scoring |
| **[`cibersortx`](../cibersortx/README.md)** | `CIBERSORT` / `CIBERSORTx` *(clean-room)* (Newman et al., 2015, 2019) | Linear $\nu$-SVR + sliding-window NNLS | `B-mode` & `S-mode` Empirical Bayes | Proportions + empirical $p$-values, Group-Mode GEPs, & High-Res purification |
| **[`cside`](../cside/README.md)** | `CSIDE` (`spacexr`, Cable et al., 2022) | Gauss–Hermite Poisson–lognormal mixture GLM | 2D thin-plate / B-spline spatial bases & niche covariates | **Post-deconvolution spatial DE**: identifies genes changing *within* a specific cell type across space |

---

## 5. Somatic Cancer Genomics, Liquid Biopsy, Mutational Signatures & HRD

| Package | Published Method | Input Modality | Matched Normal Needed? | Primary Biological Question |
|---|---|---|---|---|
| **[`facets`](../facets/README.md)** | `FACETS` & `ASCAT` (Shen & Seshan, 2016; Van Loo et al., 2010) | Tumor-normal SNP pileups ($\log R$, $\log\text{OR}$) | **Yes** (`unmatched=True` optional) | Purity, ploidy, integer $(C_{\text{major}}, C_{\text{minor}})$, LOH, and clonal CCF in WES/WGS |
| **[`purecn`](../purecn/README.md)** | `PureCN` (Riester et al., 2016) | **Tumor-only targeted panel** coverage + variants + dbSNP/PoN | **No** (tumor-only) | Purity/ploidy, Beta-Binomial **somatic vs. germline** status, LOH, CCF, and cleaned TMB/CIN |
| **[`qdnaseq`](../qdnaseq/README.md)** | `QDNAseq` & `ACE` (Scheinin et al., 2014; Poell et al., 2019) | **Shallow/low-pass WGS (`0.1x–1x` cfDNA / FFPE)** | **No** (2D GC+mappability loess) | Binned copy-number segments/calls + reference-free `ACE` cellularity & ploidy |
| **[`scarhrd`](../scarhrd/README.md)** | `scarHRD` (Sztupinszki et al., 2018) | Allele-specific CN segments (`total_cn`, `minor_cn`) | Via `facets` / `purecn` | Copy-number genomic scars: `HRD-LOH` + `TAI` + `LST` (`HRD-sum >= 42`) |
| **[`chord`](../chord/README.md)** | `CHORD` (Nguyen et al., 2020) | Somatic SNVs (96 SBS) + indels (`bimh` MH) + SVs | Somatic callset | Mutational-signature HRD probability **and `BRCA1`-type vs. `BRCA2`-type** subtyping |
| **[`tracksig`](../tracksig/README.md)** | `TrackSig` / `TrackSigFreq` (Rubanova et al., 2020) | Somatic SNVs (96 SBS + VAF/CCF + CN) + signatures | Somatic callset | `PELT` evolutionary pseudotime changepoints where mutational signatures shift |
| **[`shatterseek`](../shatterseek/README.md)** | `ShatterSeek` (Cortés-Ciriano et al., 2020) | Somatic SVs + segmented CN | Somatic callset | High/low-confidence chromothripsis vs. focal amplification / BFB |
| **[`dndscv`](../dndscv/README.md)** | `dNdScv` (Martincorena et al., 2017) | Somatic SNVs/indels + `RefCDS` coding reference | Somatic callset | Gene-level & global $dN/dS$ driver selection ($w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}}, w_{\text{ind}}$) |
| **[`numbat`](../numbat/README.md)** | `Numbat` (Gao et al., 2023) | scRNA-seq counts + phased allele counts (`AnnData`) | Diploid reference | Single-cell allele-specific CNVs, CN-neutral LOH, malignant calls & ML phylogeny |
| **[`discover`](../discover/README.md)** | `DISCOVER` (Canisius et al., 2016) | Binary alteration matrix (`{0, 1, NaN}`) | Cohort matrix | Pairwise/groupwise mutual exclusivity & co-occurrence controlling tumor burden |
| **[`fishhook`](../fishhook/README.md)** | `fishHook` (Imielinski et al., 2017) | Somatic SNVs/indels + intervals + chromatin tracks | Cohort callset | Noncoding enhancer/promoter driver recurrence enrichment |
| **[`evam`](../evam/README.md)** | `Oncotree`, `OncoBN`, `H-CBN`, `MHN` | Cross-sectional binary somatic genotypes | Cohort matrix | Evolutionary accumulation order & epistatic progression trajectories |

---

## 6. Differential Expression, Isoform DTU, Mixed Models & Multivariate Shrinkage

| Package | Published Method | Data Modality & Design | Latent / Random Effects | Primary Output |
|---|---|---|---|---|
| **[`limma`](../limma/README.md)** | `limma` (`eBayes`, `limma-trend`) (Smyth, 2004; Phipson et al., 2016) | Continuous log-expression, proteomics, CRISPR | Fixed covariates | Empirical Bayes `squeezeVar` moderated $t$/$F$ statistics & BH-FDR |
| **[`saturn`](../saturn/README.md)** | `satuRn` & `stageR` (Gilis et al., 2022) | **Bulk or sparse scRNA-seq isoform/exon counts** | Fixed design (`AnnData`) | Quasi-binomial EB moderated **differential transcript usage (`DTU`)**, `locfdr` empirical null, & `stageR` OFDR |
| **[`sva`](../sva/README.md)** | `SVA` / `irwsva` / `svaseq` (Leek & Storey, 2007; Leek, 2014) | Arrays or moderated counts | **Unmeasured latent batch** | Surrogate variables $S$ (`augment_design`) + nested `f_pvalue` |
| **[`variancepartition`](../variancepartition/README.md)** | `variancePartition` & `dream` (Hoffman & Roussos, 2021) | Continuous/weighted bulk or pseudobulk | **Known random effects** REML | ICC variance shares + Satterthwaite df + `dream` EB moderated DE |
| **[`nebula`](../nebula/README.md)** | `NEBULA-LN` / `NEBULA-HL` (He et al., 2021) | **Multi-subject scRNA-seq raw counts** | **Subject random effect** + cell NB | Fast $O(N)$ Negative Binomial GLMM Wald $\log\text{FC}$ & overdispersions |
| **[`mashr`](../mashr/README.md)** | `mash` + Extreme Deconvolution (Urbut et al., 2019) | Multi-condition $(\hat{B}, \hat{S})$ matrices | Null correlation $V$ | Cross-condition Empirical Bayes posteriors, `lfsr`, & pairwise sharing |

---

## 7. Clinical Survival, Competing Risks, Biomarker Reclassification & Composite Trial Endpoints

| Package | Published Method | Censored Outcome Setting | Statistical Estimand | When to Prefer |
|---|---|---|---|---|
| **[`biostats`](../biostats/README.md)** | `Cox PH` (`Efron`), `Fine–Gray` (`cmprsk`), `logistf` | Right-censored survival, competing risks, or rare binary | Hazard ratio (`HR`), subdistribution `sHR`, or Firth `OR` | Regression modeling of hazard / cumulative incidence or separated binary endpoints |
| **[`timeroc`](../timeroc/README.md)** | `timeROC` (Blanche et al., 2013) | Right-censored survival & competing risks | Horizon-specific IPCW $\text{AUC}_1(t), \text{AUC}_2(t)$ & simultaneous bands | Evaluating discrimination of a single continuous biomarker or comparing two markers' AUCs |
| **[`survidinri`](../survidinri/README.md)** | `survIDINRI` (Uno et al., 2013) | Right-censored nested Cox prognostic models at $t_0$ | IPCW **`IDI` (`M1`)**, continuous/categorical **`NRI` (`M2`)**, & median improvement (`M3`) | Quantifying incremental patient risk reclassification from adding a genomic score to clinical staging |
| **[`buysetest`](../buysetest/README.md)** | `BuyseTest` (Péron et al., 2016) & `WR::pwreg` (Mao & Wang, 2020) | **Prioritized composite endpoints** (OS $\succ$ PFS $\succ$ Toxicity/QoL) | **Win Ratio**, **Net Benefit**, **Win Odds** (Péron KM rule) & covariate-adjusted log-WR (`pwreg`) | Two-arm randomized clinical trials or observational comparative effectiveness with hierarchical endpoints |

---

## 8. DNA Methylation, Peak-Free ChIP/ATAC Differential Binding & Chromatin Regulation

| Package | Published Method | Input Modality | Statistical / Null Model | Primary Biological Output |
|---|---|---|---|---|
| **[`dss`](../dss/README.md)** | `DSS`, `DSS-general` & `DMRcate` (Wu et al., 2013; Park & Wu, 2016; Peters et al., 2015) | Bisulfite / EM-seq `(N, X)` CpG read counts | BSmooth tricube + log-normal EB Beta-Binomial dispersion + arcsine GLS + Satterthwaite kernel | Differentially Methylated Loci (`DML`) & Regions (`DMR`) for two-group or multifactor designs |
| **[`csaw`](../csaw/README.md)** | `csaw` (Lun & Smyth, 2016) | ChIP-seq / ATAC / CUT&RUN BAMs or fragments | Sliding windows + 10 kb TMM / loess offsets + NB Quasi-Likelihood F-test + **Simes' procedure** | **Peak-free** differentially bound regions (`up`/`down`/`mixed`) with rigorous region-level FDR |
| **[`rose`](../rose/README.md)** | `ROSE` (Whyte et al., 2013; Lovén et al., 2013) | H3K27ac ChIP / ATAC peaks + signal | Ranked stitched-enhancer tangent inflection point | Super-enhancer vs. typical enhancer calls |
| **[`chromvar`](../chromvar/README.md)** | `chromVAR` (Schep et al., 2017) | scATAC / ATAC peak counts + PWMs | GC- and accessibility-matched background peaks | Per-cell bias-corrected TF motif deviation $Z$-scores & variability |
| **[`figr`](../figr/README.md)** | `FigR` (Kartha et al., 2022) | Paired scATAC + scRNA multiome + motifs | Background-matched cis peak-to-gene Spearman $Z$-tests | DORC regulatory genes, kNN-smoothed DORC scores, & signed TF-DORC GRNs |
| **[`decoupler-ulm`](../decoupler-ulm/README.md)** | `decoupleR` (`ULM`) (Badia-i-Mompel et al., 2022) | Expression matrix + weighted gene sets | Univariate linear model $t$-statistics | Continuous pathway / regulon / NMF activity scores |
