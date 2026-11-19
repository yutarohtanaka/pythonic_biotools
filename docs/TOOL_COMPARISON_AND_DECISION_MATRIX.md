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
| **[`cause`](../cause/README.md)** | `CAUSE` (Morrison et al., 2020) | Correlated ($\eta, q$) + uncorrelated bivariate `ash` mixture (`PSIS-LOO`) | **Yes** (genome-wide LD-pruned + all SNPs) | **Yes** (truncated null $\hat{\rho}$ MLE) | No |
| **[`mrclust`](../mrclust/README.md)** | `MR-Clust` (Foley et al., 2021) | $K$ causal pathway clusters + Null + Student-$t$ Junk/outlier | Requires $F > 10$ (2nd-order SEs) | Low overlap | No |

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

| Package | Published Method | Statistical Engine | Cross-Platform Batch Alignment | Primary Output & Cell-Type GEP | When to Prefer |
|---|---|---|---|---|---|
| **[`music`](../music/README.md)** | `MuSiC` / `MuSiC2` (Wang et al., 2019; Fan et al., 2022) | Cross-subject variance-weighted IRLS NNLS | Cross-subject gene weighting | Cell-type proportions | Fast proportion estimation across multi-donor references, hierarchical clusters (`music_prop_cluster`), or case/control shifts (`music2_prop`) |
| **[`bayesprism`](../bayesprism/README.md)** | `BayesPrism` (Chu et al., 2022) | Multinomial–Dirichlet Gibbs sampling + `L-BFGS-B` MAP / shared-$\gamma$ MLE | Reference cell-state prior adaptation | Cell-type proportions **and** sample-specific expression $Z_{n,g,k}$ + spatial profiles (`run_st`) | Bulk oncology cohorts with patient-specific malignant expression, spatial spots, or downstream cell-type DE |
| **[`bisque`](../bisque/README.md)** | `Bisque` (Jew et al., 2020) | Gene-specific affine transformation + NNLS (or marker PCA SVD) | **Yes** (learns linear/shrinkage mapping from scRNA pseudobulk to bulk; supports overlapping donors) | Cell-type proportions (or relative marker abundance scores) | Strong scRNA-seq vs. bulk RNA-seq technical/platform bias, or marker-only scoring when no full scRNA reference exists |
| **[`cibersortx`](../cibersortx/README.md)** | `CIBERSORT` / `CIBERSORTx` *(clean-room)* (Newman et al., 2015, 2019) | Linear $\nu$-SVR ($\nu \in \{0.25, 0.5, 0.75\}$) + QR-NNLS / sliding-window ridge | **Yes** (`B-mode` bulk-to-signature & `S-mode` scRNA-to-bulk Empirical Bayes) | Proportions + empirical $p$-values, Group-Mode GEPs (`impute_gep_group`), & High-Res sample-specific GEPs (`impute_gep_highres`) | Signature-matrix deconvolution with empirical permutation $p$-values, microarray/scRNA batch alignment, or sample-level GEP purification |

---

## 5. Somatic Cancer Genomics, Structural Scars & Progression Modeling

| Package | Published Method | Input Modality | Confounder / Null Model | Primary Biological Question |
|---|---|---|---|---|
| **[`facets`](../facets/README.md)** | `FACETS` & `ASCAT` (Shen & Seshan, 2016; Van Loo et al., 2010) | Tumor-normal SNP pileups ($\log R$, $\log\text{OR}$) | GC/depth LOWESS + Hotelling-$T^2$ CBS + diploid baseline EM | Purity, ploidy, integer $(C_{\text{major}}, C_{\text{minor}})$, LOH, and clonal CCF in bulk WES/WGS |
| **[`dndscv`](../dndscv/README.md)** | `dNdScv` (Martincorena et al., 2017) | Somatic SNVs/indels + `RefCDS` coding reference | 192-trinucleotide Poisson MLE + NB2 epigenomic covariate offset | Gene-level & global $dN/dS$ driver selection ($w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}}, w_{\text{ind}}$) |
| **[`numbat`](../numbat/README.md)** | `Numbat` (Gao et al., 2023) | scRNA-seq counts + phased allele counts (`AnnData`) | Beta-Binomial + Poisson-LogNormal haplotype HMM + Haldane switch | Single-cell allele-specific CNVs, CN-neutral LOH, malignant cell calls & ML tumor phylogeny |
| **[`discover`](../discover/README.md)** | `DISCOVER` (Canisius et al., 2016) | Binary alteration matrix (`{0, 1, NaN}`) | MaxEnt gene $\times$ tumor mutation-burden matrix + DFT Poisson-Binomial | Pairwise or pathway mutual exclusivity & co-occurrence free of hypermutator artifacts |
| **[`fishhook`](../fishhook/README.md)** | `fishHook` (Imielinski et al., 2017) | Somatic SNVs/indels + genomic intervals + covariates | Gamma–Poisson GLM with $\log L_i$ callable offset + chromatin/RT covariates | Noncoding enhancer, promoter, or tile driver recurrence enrichment |
| **[`shatterseek`](../shatterseek/README.md)** | `ShatterSeek` (Cortés-Ciriano et al., 2020) | Somatic SVs + segmented CN | Multinomial join orientation + exponential spacing + binomial chr enrichment | High/low-confidence chromothripsis vs. focal amplification / BFB |
| **[`scarhrd`](../scarhrd/README.md)** | `scarHRD` (Sztupinszki et al., 2018) | Allele-specific CN segments (`total_cn`, `minor_cn`) | Centromere/telomere arm topology + $<3\text{ Mb}$ smoothing | `HRD-LOH` + `TAI` + `LST` (`HRD-sum >= 42` BRCAness / PARP sensitivity) |
| **[`evam`](../evam/README.md)** | `Oncotree`, `OncoBN` (BIC), `H-CBN`, `MHN` | Cross-sectional binary somatic genotypes | Branching tree, OR/AND DAG, conjunctive poset, or multiplicative hazard CTMC | Evolutionary accumulation order & epistatic progression trajectories |

---

## 6. Differential Expression, Latent Confounding, Mixed Models & Multivariate Shrinkage

| Package | Published Method | Data Modality & Design | Latent Confounding / Random Effects | Variance Moderation / Shrinkage & Primary Output |
|---|---|---|---|---|
| **[`limma`](../limma/README.md)** | `limma` (`eBayes`, `limma-trend`) (Smyth, 2004; Phipson et al., 2016) | Continuous log-expression, proteomics, CRISPR screens (fixed-effects design) | Known fixed covariates only | Empirical Bayes variance moderation (`squeezeVar`, optional `trend=True`, `robust=True`), moderated $t$/$F$ statistics & BH-FDR |
| **[`sva`](../sva/README.md)** | `SVA` / `irwsva` / `svaseq` (Leek & Storey, 2007; Leek, 2014) | Continuous arrays or moderated log-counts (`svaseq`), fixed primary design | **Unmeasured latent batch/confounding** (`num_sv` + posterior weighted SVD orthogonal to primary signal) | Surrogate variables $S$ (`augment_design`) + nested F-test (`f_pvalue`) with BH-FDR |
| **[`variancepartition`](../variancepartition/README.md)** | `variancePartition` & `dream` (Hoffman & Schadt, 2016; Hoffman & Roussos, 2021) | Continuous/precision-weighted bulk or pseudobulk expression (`N x G`), repeated measures | **Known random effects** $(1 \mid \text{subject}), (1 \mid \text{batch})$ via batched REML LMM | ICC variance fractions summing to $1.0$, contrast-specific Satterthwaite df, Empirical Bayes `squeeze_var`, & `dream` $t$/$z$ tests |
| **[`nebula`](../nebula/README.md)** | `NEBULA-LN` / `NEBULA-HL` (He et al., 2021) | **Multi-subject scRNA-seq raw counts** (`scipy.sparse` / `AnnData`) | **Known subject random effect** ($\omega_i$) + cell-level Gamma overdispersion ($\phi$) | Fast $O(N)$ Negative Binomial GLMM Wald $\log\text{FC}$, $p$-values, subject ($\sigma^2$) & cell ($\phi$) overdispersions |
| **[`mashr`](../mashr/README.md)** | `mash` + Extreme Deconvolution (Urbut et al., 2019; Bovy et al., 2011) | Multi-condition summary statistics $(\hat{B}, \hat{S})$ across $R$ conditions (post-DE or post-xQTL) | Residual null correlation $V$ across conditions (`estimate_null_correlation_simple`) | **Joint cross-condition Empirical Bayes shrinkage** (`cov_canonical`, `cov_pca`, `cov_ed`), posterior means/SDs, `lfsr`, & pairwise sharing |

- **When to use `sva` vs. `variancepartition`**: Use `sva` when technical/biological confounders are **unknown or unrecorded**; use `variancepartition` when batch, donor, or repeated-measure covariates are **observed** and you want to quantify each covariate's ICC variance share or run `dream` mixed-model DE.
- **When to use `variancepartition` (`dream`) vs. `nebula`**: Use `dream` on continuous/log-normalized bulk or pseudobulk matrices with precision weights and complex multi-random-effect formulas; use `nebula` directly on large cell-level sparse UMI count matrices across many subjects without collapsing to pseudobulk.
- **When to use `mashr` alongside `limma` / `dream`**: Fit per-condition effect estimates and SEs first (e.g. via `limma` or `dream`), then pass the $(J \times R)$ matrix to `mashr` to borrow strength across conditions and quantify tissue/condition-shared vs. specific effects.

---

## 7. Epigenomics, Chromatin Regulation & Gene-Program Inference

| Package | Published Method | Input Modality | Background / Null Model | Primary Biological Output |
|---|---|---|---|---|
| **[`chromvar`](../chromvar/README.md)** | `chromVAR` (Schep et al., 2017) | scATAC / ATAC peak-by-cell counts + motif PWM matches | GC-content and mean-accessibility-matched background peaks | Per-cell bias-corrected TF motif deviation $Z$-scores and global motif variability |
| **[`figr`](../figr/README.md)** | `FigR` (Kartha et al., 2022) | Paired scATAC + scRNA multiome (or metacells) + GTF TSS + motifs | GC- and accessibility-matched background peak Spearman $Z$-tests | Significant cis peak-to-gene links, DORCs ($n_{\text{peaks}} \ge k$), kNN-smoothed DORC scores, & signed TF-DORC GRNs |
| **[`rose`](../rose/README.md)** | `ROSE` (Whyte et al., 2013; Lovén et al., 2013) | ChIP-seq / ATAC peaks + BAM/bigWig signal (optional Input/IgG) | Ranked stitched-enhancer tangent inflection point | Super-enhancer vs. typical enhancer calls and constituent peak stitching |
| **[`decoupler-ulm`](../decoupler-ulm/README.md)** | `decoupleR` (`ULM`) (Badia-i-Mompel et al., 2022) | Expression matrix + weighted gene-program / regulon network | Univariate linear model $t$-value against gene-level weights | Continuous gene-program / TF / NMF factor activity scores per sample or cell |
