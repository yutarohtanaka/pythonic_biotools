# Pythonic Biotools

Pythonic implementations of published R, C++, and command-line bioinformatics and statistical genetics libraries.

Each directory is an independent, self-contained package (`pyproject.toml` + unit tests) so any individual tool—such as [`slopehunter`](slopehunter/README.md), [`limma`](limma/README.md), [`chromvar`](chromvar/README.md), [`rose`](rose/README.md), or [`coloc`](coloc/README.md)—can be installed and used standalone.

---

## Documentation Guides

- **[Use Cases by Biological & Statistical Problem (`docs/USE_CASES_BY_PROBLEM.md`)](docs/USE_CASES_BY_PROBLEM.md)** — Problem-driven guide mapping analysis tasks to standalone tools and multi-tool workflows.
- **[Tool Comparison & Decision Matrix (`docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md`)](docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md)** — Side-by-side comparison tables for selecting among related Mendelian Randomization, conditional/collider-adjusted GWAS, deconvolution, epigenomics, and progression-modeling tools.

---

## Tools Implemented

### 1. Bulk Deconvolution & Xenograft Read Filtering
| Directory | Package | Published Tool & Citation |
|---|---|---|
| **[`xenofilterR/`](xenofilterR/README.md)** | `xenofilterR` | **XenofilteR**: Read-level edit-distance filtering of host reads from xenograft BAMs (Kluin et al., *BMC Bioinformatics* 2018). |
| **[`music/`](music/README.md)** | `music` | **MuSiC / MuSiC2**: Cross-subject variance-weighted NNLS bulk RNA-seq deconvolution on `AnnData` (Wang et al., *Nat Commun* 2019; Fan et al., *Brief Bioinform* 2022). |
| **[`bayesprism/`](bayesprism/pythonic_bayesprism/pythonic_bayesprism.md)** | `pythonic_bayesprism` | **BayesPrism**: Bayesian Gibbs deconvolution of cell-type fractions and sample-specific cell-type expression (Chu et al., *Nat Cancer* 2022). |

### 2. Mendelian Randomization & Genetic Colocalization
| Directory | Package | Published Tool & Citation |
|---|---|---|
| **[`coloc/`](coloc/README.md)** | `coloc` | **coloc (`coloc.abf`)**: Approximate Bayes Factor colocalization (Wakefield, *Genet Epidemiol* 2009; Giambartolomei et al., *PLOS Genet* 2014). |
| **[`twosamplemr/`](twosamplemr/README.md)** | `twosamplemr` | **TwoSampleMR / MVMR**: `IVW` (Burgess et al., 2013), `MR-Egger` (Bowden et al., *IJE* 2015), `Weighted Median` (Bowden et al., 2016), `MVMR` (Sanderson et al., *Stat Med* 2019), & LD clumping. |
| **[`mrraps/`](mrraps/README.md)** | `mrraps` | **mr.raps**: Robust Adjusted Profile Score two-sample MR with `l2`, `huber`, and `tukey` losses (Zhao et al., *Ann Stat* 2020). |
| **[`mrhorse/`](mrhorse/README.md)** | `mrhorse` | **MR-Horse / MVMR-Horse**: Univariable & multivariable Bayesian Horseshoe MR for correlated pleiotropy (Grant & Burgess, *AJHG* 2024). |
| **[`mrapss/`](mrapss/README.md)** | `mrapss` | **MR-APSS**: Foreground–background mixture MR accounting for pleiotropy, selection truncation, and sample overlap (Hu et al., *Nat Commun* 2022). |
| **[`mrbee/`](mrbee/README.md)** | `mrbee` | **MRBEE**: Bias-corrected estimating equations MR with Winner's Curse & IMRP outlier removal (Lorincz-Comi et al., *AJHG* 2024). |
| **[`mvcml/`](mvcml/README.md)** | `mvcml` | **MVMR-cML**: Multivariable Constrained Maximum Likelihood MR & conditional GWAS adjustment (Lin et al., *AJHG* 2023). |

### 3. Conditional GWAS & Index-Event / Collider-Bias Adjustment
| Directory | Package | Published Tool & Citation |
|---|---|---|
| **[`slopehunter/`](slopehunter/README.md)** | `slopehunter` | **SlopeHunter**: Two-component Gaussian-mixture EM collider-bias adjustment (Mahmoud et al., *Nat Commun* 2022). |
| **[`indexevent/`](indexevent/README.md)** | `indexevent` | **indexevent**: `CWLS`, `Hedges-Olkin`, and `SIMEX` profile-likelihood index-event bias adjustment (Dudbridge et al., *Nat Commun* 2019). |
| **[`colliderbias/`](colliderbias/README.md)** | `colliderbias` | **ColliderBias**: Instrumental-effect `CWLS` (`methodCB`) and bivariate `CWBLS` (Cai et al., *Genet Epidemiol* 2022; Cai & Dudbridge, 2024). |
| **[`mtcojo/`](mtcojo/README.md)** | `mtcojo` | **GCTA-mtCOJO & GSMR**: Multi-trait conditional & joint GWAS summary analysis, two-step LDSC, and `HEIDI-outlier` filtering (Zhu et al., *Nat Commun* 2018). |

### 4. Differential Expression, Regression & Survival Biostatistics
| Directory | Package | Published Tool & Citation |
|---|---|---|
| **[`limma/`](limma/README.md)** | `limma` | **limma (`eBayes`)**: Empirical Bayes moderated linear models, contrasts, and $t$-statistics (Smyth, *SAGMB* 2004; Phipson et al., *NAR* 2016). |
| **[`logistf/`](logistf/README.md)** | `logistf` | **logistf**: Firth Jeffreys-prior penalized logistic regression for rare events and separation (Firth, *Biometrika* 1993; Heinze & Schemper, *Stat Med* 2002). |
| **[`biostats/`](biostats/README.md)** | `pythonic_biostats` | **survival / cmprsk / sandwich / qvalue**: Efron-tie Cox PH (Efron, 1977), IPCW Fine–Gray competing risks (Fine & Gray, *JASA* 1999), HC3 WLS (MacKinnon & White, 1985), and pooled-null permutation FDR (Storey & Tibshirani, *PNAS* 2003). |

### 5. Epigenomics, Gene-Program Activity, Cancer Progression & Public Databases
| Directory | Package | Published Tool & Citation |
|---|---|---|
| **[`chromvar/`](chromvar/README.md)** | `chromvar` | **chromVAR**: GC- and accessibility-corrected TF motif deviations & variability (Schep et al., *Nat Methods* 2017), plus GTF TSS & log-odds PWM scanning (Wasserman & Sandelin, 2004). |
| **[`rose/`](rose/README.md)** | `rose` | **ROSE**: Rank Ordering of Super-Enhancers (Whyte et al., *Cell* 2013; Lovén et al., *Cell* 2013). |
| **[`decoupler-ulm/`](decoupler-ulm/README.md)** | `decoupler_ulm` | **decoupleR (`ULM`)**: Univariate Linear Model continuous gene-program activity projection (Badia-i-Mompel et al., *Bioinform Adv* 2022). |
| **[`evam/`](evam/README.md)** | `evam` | **EvAM-Tools (`Oncotree`, `OncoBN`, `H-CBN`, `MHN`)**: Cross-sectional evolutionary accumulation models (Desper et al., 1999; Gerstung et al., 2009; Schill et al., 2020; Nicol et al., 2021; Diaz-Uriarte & Herrera-Nieto, *Bioinformatics* 2022). |
| **[`cohort-curation/`](cohort-curation/README.md)** | `cohort_curation` | **Public Genomics Parsers**: `cBioPortal` REST API (Cerami et al., 2012), `GISTIC2.0` wide peaks (Mermel et al., 2011), `AACR GENIE` panel coverage masking (2017), `HGNC` / Excel symbol repair (Zeeberg et al., 2004), and `Cellosaurus` (Bairoch, 2018). |
| **[`wdl-workflows/`](wdl-workflows/README.md)** | WDL 1.0 | **Terra / Cromwell WDLs**: `fastp` QC (Chen et al., 2018) and `XenofilteR` BAM filtering (Kluin et al., 2018). |

---

## Installation & Testing

Install any standalone package directly:

```bash
pip install -e ./slopehunter
pip install -e ./limma
pip install -e ./chromvar

# Run unit tests for an individual package
pytest slopehunter/tests -q
```
