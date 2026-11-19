# Pythonic Biotools

Pythonic implementations of published R, C++, Fortran, and command-line bioinformatics, cancer genomics, spatial/single-cell omics, quantitative proteomics, clinical trial biostatistics, and statistical genetics libraries.

Each directory is an independent, self-contained package (`pyproject.toml` + PEP 561 `py.typed` + `pytest` suite), paired with a root workspace bundle manager ([`pyproject.toml`](pyproject.toml) + [`pythonic_biotools_cli.py`](pythonic_biotools_cli.py)), end-to-end cross-package integration tests ([`tests_integration/`](tests_integration/README.md)), runnable multi-tool demos ([`examples/`](examples/README.md)), and frozen upstream R parity benchmarks ([`benchmarks/r_parity/`](benchmarks/r_parity/README.md)). Any individual tool or domain bundle can be installed and used without `rpy2`, compiled Fortran toolchains, or external R sessions.

---

## Documentation Guides

- **[Use Cases by Biological & Statistical Problem (`docs/USE_CASES_BY_PROBLEM.md`)](docs/USE_CASES_BY_PROBLEM.md)** — Problem-driven guide mapping analysis tasks to all 70 standalone tools and multi-tool workflows.
- **[Tool Comparison & Decision Matrix (`docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md`)](docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md)** — Side-by-side comparison tables across Mendelian Randomization, cis-MR/causal TWAS/local & structural genetics, conditional/collider GWAS, bulk & spatial deconvolution/DE, differential expression/proteomics/isoform DTU/mixed models, clinical survival & trial endpoints, somatic cancer genomics/evolution/ctDNA, and epigenomics/methylation/motif regulation.
- **[Upstream Parity Guide & Dual Scorecard (`docs/UPSTREAM_PARITY_AND_SCORECARD.md`)](docs/UPSTREAM_PARITY_AND_SCORECARD.md)** — Complete 70-package source-to-source comparison against upstream R/C++/Fortran repositories and Bioconda, documenting `settings="default"` (`89.5%` mean parity) vs. `settings="R_default"` (`98.1%` mean parity, `95%–100%` across all 70 packages).
- **[Cross-Package Integration Suite (`tests_integration/README.md`)](tests_integration/README.md)**, **[Multi-Tool Demos (`examples/README.md`)](examples/README.md)** & **[Golden `R_default` Benchmarks (`benchmarks/r_parity/README.md`)](benchmarks/r_parity/README.md)**.

---

## Tools Implemented (70 Python Packages + WDL Workflows)

### 1. Bulk & Spatial Deconvolution, Spatial Cell-Type DE & Xenograft Filtering
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`xenofilterR/`](xenofilterR/README.md)** | `xenofilterR` | MIT | **XenofilteR**: Read-level `NM` + CIGAR clip edit-distance filtering of host reads from xenograft BAMs (Kluin et al., *BMC Bioinformatics* 2018). |
| **[`music/`](music/README.md)** | `music` | MIT | **MuSiC / MuSiC2**: Cross-subject variance-weighted NNLS bulk RNA-seq deconvolution on `AnnData` (Wang et al., *Nat Commun* 2019; Fan et al., *Brief Bioinform* 2022). |
| **[`bayesprism/`](bayesprism/README.md)** | `pythonic_bayesprism` | MIT | **BayesPrism**: Bayesian Gibbs deconvolution of cell-type fractions, sample-specific cell-type expression, and spatial MLE updates (Chu et al., *Nat Cancer* 2022). |
| **[`bisque/`](bisque/README.md)** | `bisque` | MIT | **Bisque**: Reference-based bulk RNA-seq deconvolution learning gene-specific affine domain alignments from single-cell pseudobulks prior to NNLS, plus marker-gene PCA scoring on `AnnData` (Jew et al., *Nat Commun* 2020). |
| **[`cibersortx/`](cibersortx/README.md)** | `cibersortx` | MIT | **CIBERSORT / CIBERSORTx** *(clean-room)*: Signature matrix construction, linear $\nu$-SVR deconvolution, `B-mode`/`S-mode` Empirical Bayes batch alignment, Group-Mode GEP estimation, and High-Resolution sample-specific purification (Newman et al., *Nat Methods* 2015; *Nat Biotechnol* 2019). |
| **[`cside/`](cside/README.md)** | `cside` | MIT | **CSIDE (`spacexr`)**: Cell-type-specific differential expression in spatial transcriptomics via Gauss–Hermite Poisson–lognormal mixture GLMs and 2D thin-plate / B-spline spatial bases on `AnnData` (Cable et al., *Nat Methods* 2022). |

### 2. Mendelian Randomization, Causal TWAS, Colocalization & Multivariate Genetic Architecture
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`coloc/`](coloc/README.md)** | `coloc` | MIT | **coloc (`coloc.abf`)**: Log-space Approximate Bayes Factor two-trait colocalization (Wakefield, *Genet Epidemiol* 2009; Giambartolomei et al., *PLOS Genet* 2014). |
| **[`hyprcoloc/`](hyprcoloc/README.md)** | `hyprcoloc` | MIT | **HyPrColoc**: Deterministic $O(MQ)$ multi-trait Bayesian colocalization and branch-and-bound trait clustering (Foley et al., *Nat Commun* 2021). |
| **[`colocproptest/`](colocproptest/README.md)** | `colocproptest` | MIT | **colocPropTest**: LD-robust frequentist proportional colocalization testing across SNP tester pairs and LD principal components under multiple causal variants (Wallace, *AJHG* 2023). |
| **[`smr-heidi/`](smr-heidi/README.md)** | `smr_heidi` | GPL-3.0+ | **SMR, SMR-multi & HEIDI**: cis-xQTL + GWAS Summary-data-based MR and LD-aware locus `HEIDI` heterogeneity testing (Zhu et al., *Nat Genet* 2016; Wu et al., *Nat Commun* 2018). |
| **[`ctwas/`](ctwas/README.md)** | `ctwas` | MIT | **cTWAS**: Causal Transcriptome/Proteome-Wide Association Study separating causal molecular traits from LD hitchhiking and direct non-coding SNPs via Empirical Bayes EM + group-prior SuSiE-RSS (Zhao et al., *Nat Genet* 2024). |
| **[`cismr/`](cismr/README.md)** | `cismr` | MIT | **PC-GMM, MV-PC-GMM, MR-SPI & MR-GENIUS**: Correlated drug-target cis-MR via LD principal-component GMM with Sargan $J$-tests (Burgess et al., 2020), pair-voting valid-IV clique selection (Yao et al., *Biometrics* 2024), and heteroskedastic G-estimation (Tchetgen Tchetgen et al., 2021). |
| **[`hdl/`](hdl/README.md)** | `hdl` | MIT | **HDL, HDL-L & HDL.S**: High-Definition Likelihood global, local single-locus, and annotation-stratified heritability and genetic correlation over regularized LD eigenpanels (Ning et al., *Nat Genet* 2020). |
| **[`lava/`](lava/README.md)** | `lava` | MIT | **LAVA**: Univariate local $h^2$, bivariate local $r_g$, partial local genetic correlation $r_g(X,Y\mid\mathbf{Z})$, and multivariate local genetic multiple regression (Werme et al., *Nat Genet* 2022). |
| **[`genomicsem/`](genomicsem/README.md)** | `genomicsem` | MIT | **GenomicSEM**: `lavaan`-syntax Reticular Action Model (`DWLS`/`ML`) structural equation modeling over genetic covariance matrices and vectorized Common-Factor SNP GWAS with $Q_{\text{SNP}}$ heterogeneity tests (Grotzinger et al., *Nat Hum Behav* 2019). |
| **[`fgwas/`](fgwas/README.md)** | `fgwas` | MIT | **fgwas**: Hierarchical functional annotation enrichment, cross-validated ridge regularization, stepwise model selection, and re-weighted Bayesian GWAS fine-mapping (Pickrell, *AJHG* 2014). |
| **[`twosamplemr/`](twosamplemr/README.md)** | `twosamplemr` | MIT | **TwoSampleMR / MVMR**: `IVW` (Burgess et al., 2013), coding-invariant `MR-Egger` (Bowden et al., *IJE* 2015), `Weighted Median` (Bowden et al., 2016), `MVMR` with Sanderson–Windmeijer $F$ (Sanderson et al., 2019), allele harmonization & LD clumping. |
| **[`mrpresso/`](mrpresso/README.md)** | `mrpresso` | GPL-3.0+ | **MR-PRESSO / MVMR-PRESSO**: Vectorized horizontal pleiotropy Global, Outlier, and Distortion tests (Verbanck et al., *Nat Genet* 2018). |
| **[`mrraps/`](mrraps/README.md)** | `mrraps` | GPL-3.0+ | **mr.raps**: Robust Adjusted Profile Score two-sample MR with `l2`, `huber`, and `tukey` losses (Zhao et al., *Ann Stat* 2020). |
| **[`mrhorse/`](mrhorse/README.md)** | `mrhorse` | MIT | **MR-Horse / MVMR-Horse**: Univariable & multivariable Bayesian Horseshoe MR for correlated pleiotropy (Grant & Burgess, *AJHG* 2024). |
| **[`mrapss/`](mrapss/README.md)** | `mrapss` | MIT | **MR-APSS**: Foreground–background mixture MR accounting for pleiotropy, selection truncation, and sample overlap (Hu et al., *Nat Commun* 2022). |
| **[`mrbee/`](mrbee/README.md)** | `mrbee` | MIT | **MRBEE**: Univariable & multivariable bias-corrected estimating equations MR with IMRP Sidak outlier removal (Lorincz-Comi et al., *AJHG* 2024). |
| **[`mvcml/`](mvcml/README.md)** | `mvcml` | MIT | **MVMR-cML**: Multivariable Constrained Maximum Likelihood MR with BIC Model Averaging, Data Perturbation (`cML-DP`), & conditional GWAS adjustment (Lin et al., *AJHG* 2023). |
| **[`cause/`](cause/README.md)** | `cause` | GPL-3.0+ | **CAUSE**: Genome-wide nuisance estimation ($\rho$, bivariate `ash` mixture), 2D grid posterior integration separating causal effects from correlated and uncorrelated pleiotropy, and `PSIS-LOO` $\Delta\text{ELPD}$ comparison (Morrison et al., *Nat Genet* 2020). |
| **[`mrclust/`](mrclust/README.md)** | `mrclust` | MIT | **MR-Clust**: Log-space Gaussian-mixture EM with delta-method variances clustering genetic instruments into $K$ causal pathways plus null and Student-$t$ junk/outlier components via BIC selection (Foley et al., *Bioinformatics* 2021). |
| **[`lhcmr/`](lhcmr/README.md)** | `lhcmr` | MIT | **LHC-MR**: Bidirectional causal effect estimation ($\alpha_{X\to Y}, \alpha_{Y\to X}$) simultaneous with a latent heritable confounder $U$ and sample overlap via 8-component genome-wide structural Gaussian mixtures (Darrous et al., *Nat Commun* 2021). |
| **[`mrlap/`](mrlap/README.md)** | `mrlap` | MIT | **MRlap**: Analytical joint correction for sample overlap, Winner's curse, and weak-instrument bias in two-sample/one-sample MR with formal bias difference testing (Mounier & Kutalik, *Genet Epidemiol* 2023). |
| **[`grapple/`](grapple/README.md)** | `grapple` | MIT | **GRAPPLE**: Weak-instrument robust profile-likelihood MR with multimodal pleiotropic pathway discovery, marker SNP attribution, and overlap-aware multivariable MR (Wang et al., *PNAS* 2021). |

### 3. Conditional GWAS & Index-Event / Collider-Bias Adjustment
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`slopehunter/`](slopehunter/README.md)** | `slopehunter` | MIT | **SlopeHunter**: Two-component Gaussian-mixture EM collider-bias adjustment (Mahmoud et al., *Nat Commun* 2022). |
| **[`indexevent/`](indexevent/README.md)** | `indexevent` | MIT | **indexevent**: Coding-invariant `CWLS`, `Hedges-Olkin`, and `SIMEX` profile-likelihood index-event bias adjustment (Dudbridge et al., *Nat Commun* 2019). |
| **[`colliderbias/`](colliderbias/README.md)** | `colliderbias` | MIT | **ColliderBias**: Instrumental-effect `CWLS` (`methodCB`) and bivariate `CWBLS` (Cai et al., *Genet Epidemiol* 2022; Cai & Dudbridge, 2024). |
| **[`mtcojo/`](mtcojo/README.md)** | `mtcojo` | GPL-3.0+ | **GCTA-mtCOJO & GSMR**: Multi-trait conditional & joint GWAS summary analysis, two-step LDSC, `.bim` allele alignment, and `HEIDI-outlier` filtering (Zhu et al., *Nat Commun* 2018). |

### 4. Differential Expression, Proteomics, Isoform DTU, Mixed Models & Clinical Survival Biostatistics
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`limma/`](limma/README.md)** | `limma` | MIT | **limma (`eBayes`)**: Empirical Bayes moderated linear models, contrasts, `limma-trend`, and `robust=True` outlier variance winsorization (Smyth, *SAGMB* 2004; Phipson et al., *NAR* 2016). |
| **[`deqms/`](deqms/README.md)** | `deqms` | MIT | **DEqMS**: PSM/peptide-count-dependent Empirical Bayes differential protein expression (`median_sweeping`, `spectra_count_ebayes`) for TMT, DIA, and label-free proteomics (Zhu et al., *Mol Cell Proteomics* 2020). |
| **[`saturn/`](saturn/README.md)** | `saturn` | MIT | **satuRn & stageR**: Fast quasi-binomial Empirical Bayes differential transcript/exon usage (`DTU`), `locfdr` empirical-null recalibration, and two-stage OFDR screening on `AnnData` or bulk counts (Gilis et al., *F1000Res* 2022). |
| **[`sva/`](sva/README.md)** | `sva` | MIT | **SVA / `svaseq`**: Buja–Eyuboglu and asymptotic `num_sv` rank estimation, Iteratively Reweighted SVA (`irwsva`), two-step/supervised SVA, moderated log-count `svaseq`, and vectorized nested `f_pvalue` F-tests (Leek & Storey, *PLOS Genet* 2007; Leek, *NAR* 2014). |
| **[`variancepartition/`](variancepartition/README.md)** | `variancepartition` | MIT | **variancePartition & `dream`**: Batched REML linear mixed models, ICC variance decomposition across fixed and random covariates, Satterthwaite degrees of freedom, and Empirical Bayes (`squeeze_var`) repeated-measures differential expression (Hoffman & Schadt, *BMC Bioinformatics* 2016; Hoffman & Roussos, *Bioinformatics* 2021). |
| **[`nebula/`](nebula/README.md)** | `nebula` | MIT | **NEBULA (`NEBULA-LN` / `NEBULA-HL`)**: Fast multi-subject Negative Binomial mixed models (`NBGMM`, `PMM`, `NBLMM`) for sparse `AnnData` scRNA-seq using analytical marginal moments and h-likelihood saddlepoint approximation (He et al., *Commun Biol* 2021). |
| **[`mashr/`](mashr/README.md)** | `mashr` | MIT | **mashr (`mash`)**: Multivariate adaptive shrinkage across $R$ conditions combining canonical priors, `cov_pca`, and Extreme Deconvolution EM (`cov_ed`) with SQUAREM mixture weights, posterior inference, and `lfsr` (Urbut et al., *Nat Genet* 2019). |
| **[`logistf/`](logistf/README.md)** | `logistf` | MIT | **logistf**: Firth Jeffreys-prior penalized logistic regression with Profile Penalized Likelihood CIs and Penalized LRTs (Firth, *Biometrika* 1993; Heinze & Schemper, *Stat Med* 2002). |
| **[`biostats/`](biostats/README.md)** | `pythonic_biostats` | MIT | **survival / cmprsk / sandwich / qvalue**: Efron-tie Cox PH (Efron, 1977), time-varying IPCW Fine–Gray competing risks with martingale sandwich SEs (Fine & Gray, *JASA* 1999), HC3 WLS (MacKinnon & White, 1985), pooled-null permutation FDR, and Storey cubic-spline `qvalue` (Storey & Tibshirani, *PNAS* 2003). |
| **[`timeroc/`](timeroc/README.md)** | `timeroc` | GPL-2.0+ | **timeROC**: IPCW time-dependent ROC curves, competing-risks `AUC_1`/`AUC_2`, asymptotic IID influence-function SEs, simultaneous confidence bands, and paired biomarker comparisons (Blanche et al., *Stat Med* 2013). |
| **[`survidinri/`](survidinri/README.md)** | `survidinri` | MIT | **survIDINRI**: Right-censored survival Integrated Discrimination Improvement (`IDI`), continuous/categorical Net Reclassification Improvement (`NRI`), and median risk improvement with multiplier perturbation resampling (Uno et al., *Stat Med* 2013). |
| **[`survrm2/`](survrm2/README.md)** | `survrm2` | MIT | **survRM2**: Unadjusted and ANCOVA covariate-adjusted Restricted Mean Survival Time (`RMST` / `RMTL`) differences and ratios under non-proportional hazards (Uno et al., *JCO* 2014; Tian et al., *Biostatistics* 2014). |
| **[`buysetest/`](buysetest/README.md)** | `buysetest` | MIT | **BuyseTest & `WR::pwreg`**: Stratified Generalized Pairwise Comparisons (Win Ratio, Net Benefit, Win Odds) with Péron Kaplan–Meier survival-integral censoring correction and proportional win-fractions regression (Péron et al., *Stat Med* 2016; Mao & Wang, *JASA* 2020). |
| **[`jmbayes2/`](jmbayes2/README.md)** | `jmbayes2` | MIT | **JMbayes2**: Bayesian joint longitudinal biomarker + right-censored / competing-risks survival modeling (15-node Gauss–Kronrod quadrature + Adaptive MCMC) and dynamic landmark risk prediction (Rizopoulos et al.; Papageorgiou et al., *Stat Med* 2019). |

### 5. Cancer Somatic Genomics, Evolution, ctDNA, Epigenomics & Motif Regulation
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`facets/`](facets/README.md)** | `facets` | MIT | **FACETS & ASCAT**: GC/depth LOWESS normalization, bivariate Hotelling-$T^2$ CBS on $\log R$ and $\log\text{OR}^2$, and Gaussian/folded-normal EM for allele-specific copy number, purity, ploidy, LOH, and CCF, paired with `ASCAT` equations (Shen & Seshan, *NAR* 2016; Van Loo et al., *PNAS* 2010). |
| **[`purecn/`](purecn/README.md)** | `purecn` | MIT | **PureCN**: Tumor-only clinical panel GC/Tangent coverage normalization, 2D purity/ploidy optimization, Beta-Binomial somatic-vs-germline classification, LOH, CCF, and artifact-cleaned TMB/CIN (Riester et al., *Source Code Biol Med* 2016). |
| **[`qdnaseq/`](qdnaseq/README.md)** | `qdnaseq` | MIT | **QDNAseq & ACE**: Shallow/low-pass WGS (`lpWGS`) & cfDNA 2D `(GC, mappability)` loess read-depth correction, CBS copy-number calling, and `ACE` reference-free cellularity/ploidy estimation (Scheinin et al., *Genome Res* 2014; Poell et al., *Bioinformatics* 2019). |
| **[`ichorcna/`](ichorcna/README.md)** | `ichorcna` | MIT | **ichorCNA**: Ultra-low-pass WGS (`ULP-WGS`) cell-free ctDNA tumor fraction (`TFx` down to 2–3%), ploidy, and subclonal CNA Student-$t$ Hidden Markov Model (Adalsteinsson et al., *Nat Commun* 2017). |
| **[`mutationtimer/`](mutationtimer/README.md)** | `mutationtimer` | MIT | **MutationTimeR**: Evolutionary timing of somatic point mutations (`early`/`late`/`subclonal`), copy-number gains, and Whole-Genome Doubling (`WGD`) (Gerstung et al., *Nature* 2020). |
| **[`conipher/`](conipher/README.md)** | `conipher` | MIT | **CONIPHER**: Multi-region tumor somatic PhyloCCF clustering, Pigeonhole/Crossing-Rule clonal ancestry graph construction, and phylogenetic tree enumeration (Grigoriadis et al., *Nat Protoc* 2023). |
| **[`chord/`](chord/README.md)** | `chord` | MIT | **CHORD & mutSigExtractor**: Pan-cancer HRD prediction and `BRCA1`-type vs. `BRCA2`-type subtyping from 96-trinucleotide SNV, microhomology indel (`bimh`), and SV length/type contexts (Nguyen et al., *Nat Commun* 2020). |
| **[`tracksig/`](tracksig/README.md)** | `tracksig` | MIT | **TrackSig & TrackSigFreq**: CCF evolutionary pseudotime ordering, SQUAREM multinomial signature EM, and Pruned Exact Linear Time (`PELT`) changepoint segmentation of mutational signature trajectories (Rubanova et al., *Nat Commun* 2020). |
| **[`mmsig/`](mmsig/README.md)** | `mmsig` | MIT | **mmsig**: Mutational signature refitting with backward cosine-similarity pruning, multinomial bootstrap CIs, and transcriptional strand-bias confirmation for low-burden/panel tumors (Rustad et al., *Commun Biol* 2021). |
| **[`dndscv/`](dndscv/README.md)** | `dndscv` | MIT | **dNdScv**: Coding-strand 192-context `RefCDS` builder, 192-parameter trinucleotide Poisson MLE, Negative Binomial covariate regression, and gene-level $w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}}, w_{\text{ind}}$ LRTs (Martincorena et al., *Cell* 2017). |
| **[`numbat/`](numbat/README.md)** | `numbat` | MIT | **Numbat**: Haplotype-aware Beta-Binomial + Poisson-LogNormal HMM (with optional `numba` acceleration and pure NumPy/SciPy fallback) for single-cell allele-specific CNVs, CN-neutral LOH, malignant cell scoring, and NJ + NNI ML tumor phylogenies on `AnnData` (Gao et al., *Nat Biotechnol* 2023). |
| **[`discover/`](discover/README.md)** | `discover` | MIT | **DISCOVER**: Maximum-entropy background mutation burden modeling and exact DFT Poisson-Binomial pairwise/groupwise somatic mutual exclusivity & co-occurrence (Canisius et al., *Genome Biol* 2016). |
| **[`fishhook/`](fishhook/README.md)** | `fishhook` | MIT | **fishHook**: Gamma–Poisson (Negative Binomial) somatic mutation enrichment across genomic intervals with callable-territory offsets and chromatin covariates (Imielinski et al., *Cell* 2017). |
| **[`shatterseek/`](shatterseek/README.md)** | `shatterseek` | MIT | **ShatterSeek**: Chromothripsis detection and PCAWG confidence classification from interleaved structural variants and oscillating copy-number chains (Cortés-Ciriano et al., *Nat Genet* 2020). |
| **[`scarhrd/`](scarhrd/README.md)** | `scarhrd` | MIT | **scarHRD**: Homologous Recombination Deficiency genomic scar scoring (`HRD-LOH`, `TAI`, `LST`, and clinical `HRD-sum`) across `GRCh38`/`GRCh37` (Sztupinszki et al., *NPJ Breast Cancer* 2018). |
| **[`evam/`](evam/README.md)** | `evam` | AGPL-3.0+ | **EvAM-Tools (`Oncotree`, `OncoBN`, `H-CBN`, `MHN`)**: Cross-sectional evolutionary accumulation models with BIC complexity control (Desper et al., 1999; Gerstung et al., 2009; Schill et al., 2020; Nicol et al., 2021; Diaz-Uriarte & Herrera-Nieto, *Bioinformatics* 2022). |
| **[`dss/`](dss/README.md)** | `dss` | MIT | **DSS & DMRcate**: Bisulfite / EM-seq differential DNA methylation (`DML` & `DMR`) via BSmooth tricube smoothing, log-normal Empirical Bayes Beta-Binomial dispersion shrinkage, multifactor arcsine GLS (`DSS-general`), and Gaussian-kernel Satterthwaite `DMRcate` (Wu et al., *NAR* 2013; Park & Wu, *Bioinformatics* 2016; Peters et al., 2015). |
| **[`csaw/`](csaw/README.md)** | `csaw` | MIT | **csaw**: Peak-free sliding-window differential ChIP-seq / ATAC-seq / CUT&RUN binding analysis with TMM/loess normalization, NB Quasi-Likelihood F-tests, and Simes region-level FDR control (Lun & Smyth, *NAR* 2016). |
| **[`monalisa/`](monalisa/README.md)** | `monalisa` | MIT | **monaLisa**: GC/CpG composition-matched binned TF motif enrichment and Complementary-Pairs Randomized Lasso stability selection across regulatory regions (Machlab et al., *Bioinformatics* 2022). |
| **[`chromvar/`](chromvar/README.md)** | `chromvar` | MIT | **chromVAR**: Sparse-ready GC- and accessibility-corrected TF motif deviations & variability (Schep et al., *Nat Methods* 2017), plus GTF TSS & log-odds PWM scanning (Wasserman & Sandelin, 2004). |
| **[`figr/`](figr/README.md)** | `figr` | MIT | **FigR**: Paired scATAC+scRNA cis peak-to-gene Spearman Z-tests against GC/accessibility-matched background peaks, DORC calling, kNN-smoothed DORC activity scoring, and TF-DORC gene regulatory networks (Kartha et al., *Cell Genomics* 2022). |
| **[`rose/`](rose/README.md)** | `rose` | MIT | **ROSE**: Rank Ordering of Super-Enhancers with optional Input/IgG bigWig control subtraction (Whyte et al., *Cell* 2013; Lovén et al., *Cell* 2013). |
| **[`decoupler-ulm/`](decoupler-ulm/README.md)** | `decoupler_ulm` | MIT | **decoupleR (`ULM`)**: Univariate Linear Model continuous gene-program activity projection (Badia-i-Mompel et al., *Bioinform Adv* 2022). |
| **[`cohort-curation/`](cohort-curation/README.md)** | `cohort_curation` | MIT | **Public Genomics Parsers**: `cBioPortal` REST API (Cerami et al., 2012), `GISTIC2.0` wide peaks (Mermel et al., 2011), `AACR GENIE` panel coverage masking (2017), `HGNC` / Excel symbol repair (Zeeberg et al., 2004), and `Cellosaurus` (Bairoch, 2018). |
| **[`wdl-workflows/`](wdl-workflows/README.md)** | WDL 1.0 | MIT | **Terra / Cromwell WDLs**: `fastp` QC (Chen et al., 2018) and `XenofilteR` CIGAR-aware BAM filtering (Kluin et al., 2018). |

---

## Installation, Domain Bundles & Testing

### Option 1: Install Domain Bundles via Root Workspace (`pyproject.toml` / `uv`)
```bash
# Install root workspace CLI + scientific dependencies for a domain bundle
pip install -e ".[oncology]"     # or .[statgen], .[singlecell], .[epigenomics], .[proteomics], .[survival], .[all]

# Editable-install all standalone packages in a bundle (or inspect bundles)
pythonic-biotools list --bundle oncology
pythonic-biotools install oncology --no-deps
pythonic-biotools check-imports --bundle all
```

### Option 2: Install Any Individual Package Standalone
```bash
pip install -e ./mutationtimer
pip install -e ./genomicsem
pip install -e ./deqms
pip install -e ./survrm2

# Run unit, cross-package integration, and R_default golden benchmark suites
pytest mutationtimer/tests -q
pytest tests_integration benchmarks/r_parity -q
```

---

## Upstream Parity Mode (`settings="R_default"`)

Every package supports `settings="default"` (statistically hardened, numerically stabilized Pythonic defaults) and `settings="R_default"` (strict formula-, threshold-, and convention-level replication of the original R / C++ / Fortran tool). See [`docs/UPSTREAM_PARITY_AND_SCORECARD.md`](docs/UPSTREAM_PARITY_AND_SCORECARD.md) for the full 70-package dual scorecard (`89.5%` mean parity under `default` vs. `98.1%` mean parity under `R_default`).

---

## Licensing Note

Root repository scaffolding and MIT-designated clean-room packages are licensed under the [MIT License](LICENSE). Individual ports derived from copyleft upstream statistical packages carry their respective compatible copyleft SPDX licenses in their local `pyproject.toml` (`GPL-2.0-or-later` in `timeroc`; `GPL-3.0-or-later` in `cause`, `mrpresso`, `mrraps`, `mtcojo`, and `smr-heidi`; `AGPL-3.0-or-later` in `evam`).
