# Pythonic Biotools

Pythonic implementations of published R, C++, Fortran, and command-line bioinformatics, cancer genomics, and statistical genetics libraries.

Each directory is an independent, self-contained package (`pyproject.toml` + `pytest` suite) so any individual tool—such as [`slopehunter`](slopehunter/README.md), [`mrpresso`](mrpresso/README.md), [`hyprcoloc`](hyprcoloc/README.md), [`facets`](facets/README.md), [`dndscv`](dndscv/README.md), [`sva`](sva/README.md), [`variancepartition`](variancepartition/README.md), [`mashr`](mashr/README.md), [`numbat`](numbat/README.md), [`discover`](discover/README.md), [`shatterseek`](shatterseek/README.md), [`limma`](limma/README.md), or [`chromvar`](chromvar/README.md)—can be installed and used standalone without `rpy2`, compiled Fortran toolchains, or external R sessions.

---

## Documentation Guides

- **[Use Cases by Biological & Statistical Problem (`docs/USE_CASES_BY_PROBLEM.md`)](docs/USE_CASES_BY_PROBLEM.md)** — Problem-driven guide mapping analysis tasks to standalone tools and multi-tool workflows.
- **[Tool Comparison & Decision Matrix (`docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md`)](docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md)** — Side-by-side comparison tables for selecting among Mendelian Randomization, colocalization/mediation, conditional/collider-adjusted GWAS, bulk deconvolution, differential expression/mixed-model/shrinkage, somatic cancer genomics/progression-modeling, and epigenomics/single-cell multi-omics tools.
- **[Upstream Parity Guide & Dual Scorecard (`docs/UPSTREAM_PARITY_AND_SCORECARD.md`)](docs/UPSTREAM_PARITY_AND_SCORECARD.md)** — Complete 42-package source-to-source comparison against upstream R/C++/Fortran repositories and Bioconda, documenting `settings="default"` (`88.3%` mean parity) vs. `settings="R_default"` (`97.8%` mean parity, `95%–100%` across all 42 packages).

---

## Tools Implemented (42 Python Packages + WDL Workflows)

### 1. Bulk Deconvolution & Xenograft Read Filtering
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`xenofilterR/`](xenofilterR/README.md)** | `xenofilterR` | MIT | **XenofilteR**: Read-level `NM` + CIGAR clip edit-distance filtering of host reads from xenograft BAMs (Kluin et al., *BMC Bioinformatics* 2018). |
| **[`music/`](music/README.md)** | `music` | MIT | **MuSiC / MuSiC2**: Cross-subject variance-weighted NNLS bulk RNA-seq deconvolution on `AnnData` (Wang et al., *Nat Commun* 2019; Fan et al., *Brief Bioinform* 2022). |
| **[`bayesprism/`](bayesprism/README.md)** | `pythonic_bayesprism` | MIT | **BayesPrism**: Bayesian Gibbs deconvolution of cell-type fractions, sample-specific cell-type expression, and spatial MLE updates (Chu et al., *Nat Cancer* 2022). |
| **[`bisque/`](bisque/README.md)** | `bisque` | MIT | **Bisque**: Reference-based bulk RNA-seq deconvolution learning gene-specific affine domain alignments from single-cell pseudobulks prior to NNLS, plus marker-gene PCA scoring on `AnnData` (Jew et al., *Nat Commun* 2020). |
| **[`cibersortx/`](cibersortx/README.md)** | `cibersortx` | MIT | **CIBERSORT / CIBERSORTx** *(independent clean-room implementation)*: Signature matrix construction, linear $\nu$-SVR deconvolution, `B-mode`/`S-mode` Empirical Bayes batch alignment, Group-Mode NNLS cell-type GEP estimation, and High-Resolution sample-specific purification implemented from published mathematical formulations (Newman et al., *Nat Methods* 2015; *Nat Biotechnol* 2019). |

### 2. Mendelian Randomization, Colocalization & Locus Mediation
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`coloc/`](coloc/README.md)** | `coloc` | MIT | **coloc (`coloc.abf`)**: Log-space Approximate Bayes Factor two-trait colocalization (Wakefield, *Genet Epidemiol* 2009; Giambartolomei et al., *PLOS Genet* 2014). |
| **[`hyprcoloc/`](hyprcoloc/README.md)** | `hyprcoloc` | MIT | **HyPrColoc**: Deterministic $O(MQ)$ multi-trait Bayesian colocalization and branch-and-bound trait clustering (Foley et al., *Nat Commun* 2021). |
| **[`smr-heidi/`](smr-heidi/README.md)** | `smr_heidi` | GPL-3.0+ | **SMR, SMR-multi & HEIDI**: cis-xQTL + GWAS Summary-data-based MR and LD-aware locus `HEIDI` heterogeneity testing (Zhu et al., *Nat Genet* 2016; Wu et al., *Nat Commun* 2018). |
| **[`twosamplemr/`](twosamplemr/README.md)** | `twosamplemr` | MIT | **TwoSampleMR / MVMR**: `IVW` (Burgess et al., 2013), coding-invariant `MR-Egger` (Bowden et al., *IJE* 2015), `Weighted Median` (Bowden et al., 2016), `MVMR` with Sanderson–Windmeijer $F$ (Sanderson et al., 2019), allele harmonization & LD clumping. |
| **[`mrpresso/`](mrpresso/README.md)** | `mrpresso` | GPL-3.0+ | **MR-PRESSO / MVMR-PRESSO**: Vectorized horizontal pleiotropy Global, Outlier, and Distortion tests (Verbanck et al., *Nat Genet* 2018). |
| **[`mrraps/`](mrraps/README.md)** | `mrraps` | GPL-3.0+ | **mr.raps**: Robust Adjusted Profile Score two-sample MR with `l2`, `huber`, and `tukey` losses (Zhao et al., *Ann Stat* 2020). |
| **[`mrhorse/`](mrhorse/README.md)** | `mrhorse` | MIT | **MR-Horse / MVMR-Horse**: Univariable & multivariable Bayesian Horseshoe MR for correlated pleiotropy (Grant & Burgess, *AJHG* 2024). |
| **[`mrapss/`](mrapss/README.md)** | `mrapss` | MIT | **MR-APSS**: Foreground–background mixture MR accounting for pleiotropy, selection truncation, and sample overlap (Hu et al., *Nat Commun* 2022). |
| **[`mrbee/`](mrbee/README.md)** | `mrbee` | MIT | **MRBEE**: Univariable & multivariable bias-corrected estimating equations MR with IMRP Sidak outlier removal (Lorincz-Comi et al., *AJHG* 2024). |
| **[`mvcml/`](mvcml/README.md)** | `mvcml` | MIT | **MVMR-cML**: Multivariable Constrained Maximum Likelihood MR with BIC Model Averaging, Data Perturbation (`cML-DP`), & conditional GWAS adjustment (Lin et al., *AJHG* 2023). |
| **[`cause/`](cause/README.md)** | `cause` | GPL-3.0+ | **CAUSE**: Genome-wide nuisance estimation ($\rho$, bivariate `ash` mixture), 2D grid posterior integration separating causal effects from correlated and uncorrelated pleiotropy, and `PSIS-LOO` $\Delta\text{ELPD}$ comparison (Morrison et al., *Nat Genet* 2020). |
| **[`mrclust/`](mrclust/README.md)** | `mrclust` | MIT | **MR-Clust**: Log-space Gaussian-mixture EM with delta-method variances clustering genetic instruments into $K$ causal pathways plus null and Student-$t$ junk/outlier components via BIC selection (Foley et al., *Bioinformatics* 2021). |

### 3. Conditional GWAS & Index-Event / Collider-Bias Adjustment
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`slopehunter/`](slopehunter/README.md)** | `slopehunter` | MIT | **SlopeHunter**: Two-component Gaussian-mixture EM collider-bias adjustment (Mahmoud et al., *Nat Commun* 2022). |
| **[`indexevent/`](indexevent/README.md)** | `indexevent` | MIT | **indexevent**: Coding-invariant `CWLS`, `Hedges-Olkin`, and `SIMEX` profile-likelihood index-event bias adjustment (Dudbridge et al., *Nat Commun* 2019). |
| **[`colliderbias/`](colliderbias/README.md)** | `colliderbias` | MIT | **ColliderBias**: Instrumental-effect `CWLS` (`methodCB`) and bivariate `CWBLS` (Cai et al., *Genet Epidemiol* 2022; Cai & Dudbridge, 2024). |
| **[`mtcojo/`](mtcojo/README.md)** | `mtcojo` | GPL-3.0+ | **GCTA-mtCOJO & GSMR**: Multi-trait conditional & joint GWAS summary analysis, two-step LDSC, `.bim` allele alignment, and `HEIDI-outlier` filtering (Zhu et al., *Nat Commun* 2018). |

### 4. Differential Expression, Confounding, Mixed Models, Shrinkage & Survival Biostatistics
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`limma/`](limma/README.md)** | `limma` | MIT | **limma (`eBayes`)**: Empirical Bayes moderated linear models, contrasts, `limma-trend`, and `robust=True` outlier variance winsorization (Smyth, *SAGMB* 2004; Phipson et al., *NAR* 2016). |
| **[`sva/`](sva/README.md)** | `sva` | MIT | **SVA / `svaseq`**: Buja–Eyuboglu and asymptotic `num_sv` rank estimation, Iteratively Reweighted SVA (`irwsva`), two-step/supervised SVA, moderated log-count `svaseq`, and vectorized nested `f_pvalue` F-tests (Leek & Storey, *PLOS Genet* 2007; Leek, *NAR* 2014). |
| **[`variancepartition/`](variancepartition/README.md)** | `variancepartition` | MIT | **variancePartition & `dream`**: Batched REML linear mixed models, ICC variance decomposition across fixed and random covariates, Satterthwaite degrees of freedom, and Empirical Bayes (`squeeze_var`) repeated-measures differential expression (Hoffman & Schadt, *BMC Bioinformatics* 2016; Hoffman & Roussos, *Bioinformatics* 2021). |
| **[`nebula/`](nebula/README.md)** | `nebula` | MIT | **NEBULA (`NEBULA-LN` / `NEBULA-HL`)**: Fast multi-subject Negative Binomial mixed models (`NBGMM`, `PMM`, `NBLMM`) for sparse `AnnData` scRNA-seq using analytical marginal moments and h-likelihood saddlepoint approximation (He et al., *Commun Biol* 2021). |
| **[`mashr/`](mashr/README.md)** | `mashr` | MIT | **mashr (`mash`)**: Multivariate adaptive shrinkage across $R$ conditions combining canonical priors, `cov_pca`, and Extreme Deconvolution EM (`cov_ed`) with SQUAREM mixture weights, posterior inference, and `lfsr` (Urbut et al., *Nat Genet* 2019). |
| **[`logistf/`](logistf/README.md)** | `logistf` | MIT | **logistf**: Firth Jeffreys-prior penalized logistic regression with Profile Penalized Likelihood CIs and Penalized LRTs (Firth, *Biometrika* 1993; Heinze & Schemper, *Stat Med* 2002). |
| **[`biostats/`](biostats/README.md)** | `pythonic_biostats` | MIT | **survival / cmprsk / sandwich / qvalue**: Efron-tie Cox PH (Efron, 1977), time-varying IPCW Fine–Gray competing risks with martingale sandwich SEs (Fine & Gray, *JASA* 1999), HC3 WLS (MacKinnon & White, 1985), pooled-null permutation FDR, and Storey cubic-spline `qvalue` (Storey & Tibshirani, *PNAS* 2003). |
| **[`timeroc/`](timeroc/README.md)** | `timeroc` | GPL-2.0+ | **timeROC**: IPCW time-dependent ROC curves, competing-risks `AUC_1`/`AUC_2`, asymptotic IID influence-function SEs, simultaneous confidence bands, and paired biomarker comparisons (Blanche et al., *Stat Med* 2013). |

### 5. Cancer Somatic Genomics, Structural Variants, Epigenomics & Public Cohorts
| Directory | Import Name | License | Published Tool & Citation |
|---|---|---|---|
| **[`facets/`](facets/README.md)** | `facets` | MIT | **FACETS & ASCAT**: GC/depth LOWESS normalization, bivariate Hotelling-$T^2$ CBS on $\log R$ and $\log\text{OR}^2$, and Gaussian/folded-normal EM for allele-specific copy number, purity, ploidy, LOH, and CCF, paired with `ASCAT` equations (Shen & Seshan, *NAR* 2016; Van Loo et al., *PNAS* 2010). |
| **[`dndscv/`](dndscv/README.md)** | `dndscv` | MIT | **dNdScv**: Coding-strand 192-context `RefCDS` builder, 192-parameter trinucleotide Poisson MLE, Negative Binomial covariate regression, and gene-level $w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}}, w_{\text{ind}}$ LRTs (Martincorena et al., *Cell* 2017). |
| **[`numbat/`](numbat/README.md)** | `numbat` | MIT | **Numbat**: Haplotype-aware Beta-Binomial + Poisson-LogNormal HMM (with optional `numba` acceleration and pure NumPy/SciPy fallback) for single-cell allele-specific CNVs, CN-neutral LOH, malignant cell scoring, and NJ + NNI ML tumor phylogenies on `AnnData` (Gao et al., *Nat Biotechnol* 2023). |
| **[`discover/`](discover/README.md)** | `discover` | MIT | **DISCOVER**: Maximum-entropy background mutation burden modeling and exact DFT Poisson-Binomial pairwise/groupwise somatic mutual exclusivity & co-occurrence (Canisius et al., *Genome Biol* 2016). |
| **[`fishhook/`](fishhook/README.md)** | `fishhook` | MIT | **fishHook**: Gamma–Poisson (Negative Binomial) somatic mutation enrichment across genomic intervals with callable-territory offsets and chromatin covariates (Imielinski et al., *Cell* 2017). |
| **[`shatterseek/`](shatterseek/README.md)** | `shatterseek` | MIT | **ShatterSeek**: Chromothripsis detection and PCAWG confidence classification from interleaved structural variants and oscillating copy-number chains (Cortés-Ciriano et al., *Nat Genet* 2020). |
| **[`scarhrd/`](scarhrd/README.md)** | `scarhrd` | MIT | **scarHRD**: Homologous Recombination Deficiency genomic scar scoring (`HRD-LOH`, `TAI`, `LST`, and clinical `HRD-sum`) across `GRCh38`/`GRCh37` (Sztupinszki et al., *NPJ Breast Cancer* 2018). |
| **[`evam/`](evam/README.md)** | `evam` | AGPL-3.0+ | **EvAM-Tools (`Oncotree`, `OncoBN`, `H-CBN`, `MHN`)**: Cross-sectional evolutionary accumulation models with BIC complexity control (Desper et al., 1999; Gerstung et al., 2009; Schill et al., 2020; Nicol et al., 2021; Diaz-Uriarte & Herrera-Nieto, *Bioinformatics* 2022). |
| **[`chromvar/`](chromvar/README.md)** | `chromvar` | MIT | **chromVAR**: Sparse-ready GC- and accessibility-corrected TF motif deviations & variability (Schep et al., *Nat Methods* 2017), plus GTF TSS & log-odds PWM scanning (Wasserman & Sandelin, 2004). |
| **[`figr/`](figr/README.md)** | `figr` | MIT | **FigR**: Paired scATAC+scRNA cis peak-to-gene Spearman Z-tests against GC/accessibility-matched background peaks, DORC calling, kNN-smoothed DORC activity scoring, and TF-DORC gene regulatory networks (Kartha et al., *Cell Genomics* 2022). |
| **[`rose/`](rose/README.md)** | `rose` | MIT | **ROSE**: Rank Ordering of Super-Enhancers with optional Input/IgG bigWig control subtraction (Whyte et al., *Cell* 2013; Lovén et al., *Cell* 2013). |
| **[`decoupler-ulm/`](decoupler-ulm/README.md)** | `decoupler_ulm` | MIT | **decoupleR (`ULM`)**: Univariate Linear Model continuous gene-program activity projection (Badia-i-Mompel et al., *Bioinform Adv* 2022). |
| **[`cohort-curation/`](cohort-curation/README.md)** | `cohort_curation` | MIT | **Public Genomics Parsers**: `cBioPortal` REST API (Cerami et al., 2012), `GISTIC2.0` wide peaks (Mermel et al., 2011), `AACR GENIE` panel coverage masking (2017), `HGNC` / Excel symbol repair (Zeeberg et al., 2004), and `Cellosaurus` (Bairoch, 2018). |
| **[`wdl-workflows/`](wdl-workflows/README.md)** | WDL 1.0 | MIT | **Terra / Cromwell WDLs**: `fastp` QC (Chen et al., 2018) and `XenofilteR` CIGAR-aware BAM filtering (Kluin et al., 2018). |

---

## Repository Layout & Key Dependencies

```text
pythonic_biotools/
├── bayesprism/  biostats/  bisque/  cause/  chromvar/  cibersortx/  cohort-curation/
├── colliderbias/  coloc/  decoupler-ulm/  discover/  dndscv/  docs/  evam/  facets/
├── figr/  fishhook/  hyprcoloc/  indexevent/  limma/  logistf/  mashr/  mrapss/
├── mrbee/  mrclust/  mrhorse/  mrpresso/  mrraps/  mtcojo/  music/  mvcml/  nebula/
├── numbat/  rose/  scarhrd/  shatterseek/  slopehunter/  smr-heidi/  sva/  timeroc/
├── twosamplemr/  variancepartition/  wdl-workflows/  xenofilterR/
```

- **Core scientific Python stack**: `numpy`, `scipy`, `pandas` across all packages.
- **Single-cell & ML extensions**: `anndata` (`music`, `bayesprism`, `bisque`, `cibersortx`, `numbat`, `nebula`, `figr`; optional in `sva`, `variancepartition`), `scikit-learn` (`cibersortx`, `numbat`, `figr`).
- **Genomics & optional acceleration**: `pysam` (`xenofilterR`; optional FASTA support in `dndscv`), `requests` (`cohort-curation`), `joblib` (optional parallelization in `variancepartition`), and optional `numba` JIT acceleration in `numbat` (`pip install -e "./numbat[accel]"`) with a pure vectorized NumPy/SciPy fallback (`backend="auto"` / `"numpy"`).

---

## Licensing Note

Root repository scaffolding and MIT-designated packages are licensed under the [MIT License](LICENSE). Individual ports derived from copyleft upstream statistical packages carry their respective compatible copyleft SPDX licenses in their local `pyproject.toml` (`GPL-2.0-or-later` in `timeroc`; `GPL-3.0-or-later` in `cause`, `mrpresso`, `mrraps`, `mtcojo`, and `smr-heidi`; `AGPL-3.0-or-later` in `evam`).

---

## Upstream Parity Mode (`settings="R_default"`)

Every package supports `settings="default"` (statistically hardened, numerically stabilized Pythonic defaults) and `settings="R_default"` (strict formula-, threshold-, and convention-level replication of the original R / C++ / Fortran tool):

```python
from xenofilterR import Xenofilter
from mrbee import fit_mrbee
from facets import run_facets

xf = Xenofilter(mismatch_cutoff=4, settings="R_default")
res_mrbee = fit_mrbee(bx, sx, by, sy, settings="R_default")
res_facets = run_facets(pileup_df, settings="R_default")
```

See [`docs/UPSTREAM_PARITY_AND_SCORECARD.md`](docs/UPSTREAM_PARITY_AND_SCORECARD.md) for the full 42-package dual scorecard (`88.3%` mean parity under `default` vs. `97.8%` mean parity under `R_default`).

---

## Installation & Testing

Install any standalone package directly:

```bash
pip install -e ./mrpresso
pip install -e ./hyprcoloc
pip install -e ./discover
pip install -e ./limma
pip install -e ./facets

# Run unit tests for an individual package
pytest mrpresso/tests -q
```
