# Tool Comparison & Decision Matrix

Use the comparison tables below to choose among published methods implemented in `pythonic_biotools`.

---

## 1. Two-Sample & Multivariable Mendelian Randomization

| Package | Published Method | Pleiotropy Assumption | Weak-IV Robust? | Sample Overlap Safe? | Multivariable? |
|---|---|---|---|---|---|
| **[`twosamplemr`](../twosamplemr/README.md)** | `IVW`, `MR-Egger`, `Weighted Median`, `MVMR` | Balanced (`IVW`), InSIDE (`Egger`), or ≥50% valid weight (`WM`) | Requires $F > 10$ | Low overlap | **Yes** (`multivariable_mr`) |
| **[`mrpresso`](../mrpresso/README.md)** | `MR-PRESSO` / `MVMR-PRESSO` (Verbanck et al., 2018) | Sparse horizontal pleiotropic outliers (Global, Outlier & Distortion tests) | Requires $F > 10$ | Low overlap | **Yes** (`mvmr_presso`) |
| **[`mrraps`](../mrraps/README.md)** | `mr.raps` (Zhao et al., 2020) | Overdispersed $\tau^2$ + Huber/Tukey outliers | **Yes** (profile score) | Low–moderate | No |
| **[`mrhorse`](../mrhorse/README.md)** | `MR-Horse` / `MVMR-Horse` (Grant & Burgess, 2024) | **Correlated + uncorrelated** Horseshoe prior | Yes (latent exposure) | Moderate | **Yes** (`mvmr_horse`) |
| **[`mrapss`](../mrapss/README.md)** | `MR-APSS` (Hu et al., 2022) | Foreground–background mixture + LDSC $\Omega$ | **Yes** (`|z_X| > t_0` truncation) | **Yes** (LDSC intercept $C$) | No |
| **[`mrbee`](../mrbee/README.md)** | `MRBEE` (Lorincz-Comi et al., 2024) | IMRP Sidak outlier removal + $\tau^2$ | **Yes** (bias-corrected EE + Mills ratio) | **Yes** ($\rho_e$ / $R_{xy}$ subtracted) | **Yes** (`fit_mvmr_bee`) |
| **[`mvcml`](../mvcml/README.md)** | `MVMR-cML` (Lin et al., 2023) | BIC Model Averaging + Data Perturbation (`cML-DP`) over $k$ invalid IVs | Yes (errors-in-variables cML) | **Yes** (`rho_yx` matrix) | **Yes** (`cml_ma`, `mvcml_dp`) |

---

## 2. Locus Colocalization & Molecular QTL Mediation

| Package | Published Method | Number of Traits | LD Matrix Needed? | Primary Output |
|---|---|---|---|---|
| **[`coloc`](../coloc/README.md)** | `coloc.abf` (Giambartolomei et al., 2014) | 2 traits | No (single causal variant per trait) | Exact `PP.H0`–`PP.H4` & 95% credible set |
| **[`hyprcoloc`](../hyprcoloc/README.md)** | `HyPrColoc` (Foley et al., 2021) | **$K \ge 2$ traits** (scales to 100+ traits in $O(MK)$) | Optional (`ld_matrix`, `trait_cor`) | Colocalized trait clusters, `PP = P_R * P_A`, candidate SNP & % explained |
| **[`smr-heidi`](../smr-heidi/README.md)** | `SMR`, `SMR-multi` & `HEIDI` (Zhu et al., 2016; Wu et al., 2018) | cis-xQTL + GWAS | **Yes** (for `HEIDI` & `SMR-multi`) | Causal mediation effect $\hat{b}_{xy}$, $p_{\text{SMR}}$, and $p_{\text{HEIDI}}$ (pleiotropy vs. linkage) |

---

## 3. Conditional GWAS & Index-Event / Collider-Bias Adjustment

| Package | Published Method | Core Slope Assumption | Handles Direct Pleiotropic Loci? | LD Reference Needed? |
|---|---|---|---|---|
| **[`slopehunter`](../slopehunter/README.md)** | `SlopeHunter` (Mahmoud et al., 2022) | Two-component EM mixture ("Hunted" vs. "Pleiotropic") | **Yes** | LD-pruned SNPs |
| **[`indexevent`](../indexevent/README.md)** | `indexevent` (Dudbridge et al., 2019) | Orthogonal direct effects (`CWLS`, `Hedges-Olkin`, `SIMEX`) | Orthogonal assumption | LD-pruned SNPs |
| **[`colliderbias`](../colliderbias/README.md)** | `ColliderBias` (Cai et al., 2022; Cai & Dudbridge, 2024) | Instrumental `CWLS` (`ivw`/`egger`) or 3-trait `CWBLS` | Via Egger or 3-trait `CWBLS` | LD-pruned SNPs |
| **[`mtcojo`](../mtcojo/README.md)** | `GCTA-mtCOJO` & `GSMR` (Zhu et al., 2018) | GSMR v2 causal effect $b_{xy}$ + bivariate LDSC | **Yes** (global `HEIDI-outlier`) | **Yes** (unless `bxy_external` given) |

---

## 4. Bulk Transcriptome Deconvolution

| Criterion | [`music`](../music/README.md) (`MuSiC` / `MuSiC2`) | [`bayesprism`](../bayesprism/README.md) (`BayesPrism`) |
|---|---|---|
| **Primary Output** | Cell-type proportions | Cell-type proportions **and** sample-specific cell-type expression $Z_{n,g,k}$ |
| **Statistical Engine** | Cross-subject variance-weighted IRLS NNLS | Data-augmentation multinomial–Dirichlet Gibbs sampling + `L-BFGS-B` MAP / spatial shared-$\gamma$ MLE reference update |
| **When to Prefer** | Fast proportion estimation across multi-donor references, hierarchical clusters (`music_prop_cluster`), or case/control shifts (`music2_prop`) | Bulk oncology cohorts with patient-specific malignant expression, spatial spots (`run_st`), or downstream cell-type DE |

---

## 5. Somatic Cancer Genomics, Structural Scars & Progression Modeling

| Package | Published Method | Input Modality | Confounder / Null Model | Primary Biological Question |
|---|---|---|---|---|
| **[`discover`](../discover/README.md)** | `DISCOVER` (Canisius et al., 2016) | Binary alteration matrix (`{0, 1, NaN}`) | MaxEnt gene $\times$ tumor mutation-burden matrix + DFT Poisson-Binomial | Pairwise or pathway mutual exclusivity & co-occurrence free of hypermutator artifacts |
| **[`fishhook`](../fishhook/README.md)** | `fishHook` (Imielinski et al., 2017) | Somatic SNVs/indels + genomic intervals + covariates | Gamma–Poisson GLM with $\log L_i$ callable offset + chromatin/RT covariates | Noncoding enhancer, promoter, or tile driver recurrence enrichment |
| **[`shatterseek`](../shatterseek/README.md)** | `ShatterSeek` (Cortés-Ciriano et al., 2020) | Somatic SVs + segmented CN | Multinomial join orientation + exponential spacing + binomial chr enrichment | High/low-confidence chromothripsis vs. focal amplification / BFB |
| **[`scarhrd`](../scarhrd/README.md)** | `scarHRD` (Sztupinszki et al., 2018) | Allele-specific CN segments (`total_cn`, `minor_cn`) | Centromere/telomere arm topology + $<3\text{ Mb}$ smoothing | `HRD-LOH` + `TAI` + `LST` (`HRD-sum >= 42` BRCAness / PARP sensitivity) |
| **[`evam`](../evam/README.md)** | `Oncotree`, `OncoBN` (BIC), `H-CBN`, `MHN` | Cross-sectional binary somatic genotypes | Branching tree, OR/AND DAG, conjunctive poset, or multiplicative hazard CTMC | Evolutionary accumulation order & epistatic progression trajectories |
