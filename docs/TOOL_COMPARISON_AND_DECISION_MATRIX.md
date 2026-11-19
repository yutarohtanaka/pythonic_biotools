# Tool Comparison & Decision Matrix

Use the comparison tables below to choose among published methods implemented in `pythonic_biotools`.

---

## 1. Two-Sample & Multivariable Mendelian Randomization

| Package | Published Method | Pleiotropy Assumption | Weak-IV Robust? | Sample Overlap Safe? | Multivariable? |
|---|---|---|---|---|---|
| **[`twosamplemr`](../twosamplemr/README.md)** | `IVW`, `MR-Egger`, `Weighted Median`, `MVMR` | Balanced (`IVW`), InSIDE (`Egger`), or ≥50% valid weight (`WM`) | Requires $F > 10$ | Low overlap | Yes (`multivariable_mr`) |
| **[`mrraps`](../mrraps/README.md)** | `mr.raps` (Zhao et al., 2020) | Overdispersed $\tau^2$ + Huber/Tukey outliers | **Yes** (profile score) | Low–moderate | No |
| **[`mrhorse`](../mrhorse/README.md)** | `MR-Horse` / `MVMR-Horse` (Grant & Burgess, 2024) | **Correlated + uncorrelated** Horseshoe prior | Yes (latent exposure) | Moderate | **Yes** (`mvmr_horse`) |
| **[`mrapss`](../mrapss/README.md)** | `MR-APSS` (Hu et al., 2022) | Foreground–background mixture + LDSC $\Omega$ | **Yes** (`|z_X| > t_0` truncation) | **Yes** (LDSC intercept $C$) | No |
| **[`mrbee`](../mrbee/README.md)** | `MRBEE` (Lorincz-Comi et al., 2024) | IMRP Sidak outlier removal + $\tau^2$ | **Yes** (bias-corrected EE + Mills ratio) | **Yes** ($\rho_e$ subtracted) | Univariable |
| **[`mvcml`](../mvcml/README.md)** | `MVMR-cML` (Lin et al., 2023) | BIC subset selection of $k$ invalid IVs | Yes (errors-in-variables cML) | **Yes** (`rho_yx` matrix) | **Yes** |

---

## 2. Conditional GWAS & Index-Event / Collider-Bias Adjustment

| Package | Published Method | Core Slope Assumption | Handles Direct Pleiotropic Loci? | LD Reference Needed? |
|---|---|---|---|---|
| **[`slopehunter`](../slopehunter/README.md)** | `SlopeHunter` (Mahmoud et al., 2022) | Two-component EM mixture ("Hunted" vs. "Pleiotropic") | **Yes** | LD-pruned SNPs |
| **[`indexevent`](../indexevent/README.md)** | `indexevent` (Dudbridge et al., 2019) | Orthogonal direct effects (`CWLS`, `Hedges-Olkin`, `SIMEX`) | Orthogonal assumption | LD-pruned SNPs |
| **[`colliderbias`](../colliderbias/README.md)** | `ColliderBias` (Cai et al., 2022; Cai & Dudbridge, 2024) | Instrumental `CWLS` (`ivw`/`egger`) or 3-trait `CWBLS` | Via Egger or 3-trait `CWBLS` | LD-pruned SNPs |
| **[`mtcojo`](../mtcojo/README.md)** | `GCTA-mtCOJO` & `GSMR` (Zhu et al., 2018) | GSMR causal effect $b_{xy}$ + bivariate LDSC | **Yes** (`HEIDI-outlier` removal) | **Yes** (unless `bxy_external` given) |

---

## 3. Bulk Transcriptome Deconvolution

| Criterion | [`music`](../music/README.md) (`MuSiC` / `MuSiC2`) | [`bayesprism`](../bayesprism/pythonic_bayesprism/pythonic_bayesprism.md) (`BayesPrism`) |
|---|---|---|
| **Primary Output** | Cell-type proportions | Cell-type proportions **and** sample-specific cell-type expression $Z_{n,g,k}$ |
| **Statistical Engine** | Cross-subject variance-weighted NNLS | Collapsed Gibbs sampling + conjugate-gradient MAP reference update |
| **When to Prefer** | Fast proportion estimation across multi-donor references or case/control shifts (`music2_prop`) | Bulk oncology cohorts with patient-specific malignant expression or downstream cell-type-specific DE |

---

## 4. Cross-Sectional Evolutionary Accumulation Models ([`evam`](../evam/README.md))

| Function | Published Model | Graph Topology | Ordering Metric |
|---|---|---|---|
| **`fit_ot`** | `Oncotree` (Desper et al., 1999) | Directed branching tree (≤1 parent per event) | Structural ancestor (`1` / `0` / `0.5`) |
| **`fit_oncobn`** | `OncoBN` (Nicol et al., 2021) | Acyclic DAG (`DBN` OR-gate or `CBN` AND-gate, $\le k$ parents) | Structural dominator (`DBN`) or ancestor (`CBN`) |
| **`fit_cbn`** | `H-CBN` (Gerstung et al., 2009) | Conjunctive poset (`ct-cbn` binary wrapper) | Exact ($n \le 14$) or MC CTMC path probability |
| **`fit_mhn`** | `MHN` (Schill et al., 2020) | Pairwise multiplicative log-hazard network (`mhn` wrapper) | Exact ($n \le 14$) or MC CTMC path probability |
