# `twosamplemr`: Summary-Data Mendelian Randomization & Instrument Clumping

Standalone Python implementation of classical Two-Sample Mendelian Randomization (`MRCIEU/TwoSampleMR`) and Multivariable Mendelian Randomization (`WSpiller/MVMR`), including inverse-variance weighted MR (`ivw_mr`), coding-invariant MR-Egger (`mr_egger`), Weighted Median MR (`weighted_median_mr`), Multivariable MR with Sanderson–Windmeijer conditional $F$-statistics (`multivariable_mr`), allele harmonisation (`harmonise_mr_data`), and palindromic-aware LD clumping (`select_mr_instruments`).

## Citation

> Hemani, G., Zheng, J., Elsworth, B., et al. (2018). The MR-Base platform supports systematic causal inference across the human phenome. *eLife*, 7, e34408. https://doi.org/10.7554/eLife.34408

> Bowden, J., Davey Smith, G., Haycock, P. C., & Burgess, S. (2016). Consistent estimation in Mendelian randomization with some invalid instruments using a weighted median estimator. *Genetic Epidemiology*, 40(4), 304–314. https://doi.org/10.1002/gepi.21965

> Sanderson, E., Spiller, W., & Bowden, J. (2021). Testing and correcting for weak and pleiotropic instruments in two-sample multivariable Mendelian randomization. *Statistics in Medicine*, 40(25), 5434–5452. https://doi.org/10.1002/sim.9133

## Statistical Model

Given $K$ harmonised genetic instruments with exposure effects $(\hat{\beta}_{X,j}, s_{X,j})$ and outcome effects $(\hat{\beta}_{Y,j}, s_{Y,j})$:

1. **Multiplicative Random-Effects IVW & Cochran's $Q$** (`ivw_mr`):
   $$\hat{\beta}_{\text{IVW}} = \frac{\sum_{j=1}^K w_j \hat{\beta}_{X,j}\hat{\beta}_{Y,j}}{\sum_{j=1}^K w_j \hat{\beta}_{X,j}^2}, \qquad Q = \sum_{j=1}^K w_j (\hat{\beta}_{Y,j} - \hat{\beta}_{\text{IVW}}\hat{\beta}_{X,j})^2, \qquad w_j = s_{Y,j}^{-2},$$
   with standard error $\text{se}(\hat{\beta}_{\text{IVW}}) = \sqrt{\max(1, Q / (K - 1)) / \sum_j w_j \hat{\beta}_{X,j}^2}$ and normal two-sided $p$-value.

2. **Coding-Invariant MR-Egger** (`mr_egger`):
   Orients instrument alleles so $\tilde{\beta}_{X,j} = |\hat{\beta}_{X,j}| \ge 0$ and $\tilde{\beta}_{Y,j} = \operatorname{sign}(\hat{\beta}_{X,j})\hat{\beta}_{Y,j}$, then fits weighted least squares $\tilde{\beta}_{Y,j} = \alpha_0 + \beta_{\text{Egger}} \tilde{\beta}_{X,j} + \varepsilon_j$ with weights $s_{Y,j}^{-2}$ and Student-$t_{K-2}$ inference.

3. **Weighted Median MR** (`weighted_median_mr`):
   Computes the interpolated median of ordered Wald ratios $\hat{r}_j = \hat{\beta}_{Y,j} / \hat{\beta}_{X,j}$ using normalized instrument weights $w_j / \sum_i w_i$ centered at cumulative midpoint $\sum_{i \le (j)} \tilde{w}_{(i)} - \frac{1}{2}\tilde{w}_{(j)}$, with parametric bootstrap standard errors (`n_boot = 1000`).

4. **Multivariable MR & Conditional $F$-Statistics** (`multivariable_mr`):
   Jointly regresses $\hat{\beta}_Y$ on $P$ exposures $X = [\hat{\beta}_{X,1}, C]$ with inverse-variance weights $W = \operatorname{diag}(s_{Y,j}^{-2})$ and evaluates weak-instrument strength for the primary exposure via Sanderson–Windmeijer residual projection $\hat{\beta}_{X,1} - C\hat{\delta}$ normalized by exposure measurement-error variance $s_{X,1,j}^2 + \sum_{m=2}^P \hat{\delta}_m^2 s_{C,jm}^2$.

## Python Quickstart

```python
import numpy as np
from twosamplemr import (
    harmonise_mr_data,
    select_mr_instruments,
    ivw_mr,
    mr_egger,
    weighted_median_mr,
    multivariable_mr,
    run_mr_suite,
)

rng = np.random.default_rng(42)
k = 30
bx = rng.uniform(0.08, 0.25, size=k)
sx = np.full(k, 0.015)
cov = rng.normal(0.0, 0.12, size=k)
sc = np.full(k, 0.015)
sy = np.full(k, 0.03)
by = 0.55 * bx + 0.30 * cov + rng.normal(0.0, sy)

# Default Pythonic settings
res_default = run_mr_suite(
    bx, sx, by, sy, covariate_exp_betas=cov, covariate_exp_ses=sc, settings="default"
)
print(res_default.to_dataframe())

# Upstream R parity settings (MRCIEU/TwoSampleMR + WSpiller/MVMR)
res_r = run_mr_suite(
    bx, sx, by, sy, covariate_exp_betas=cov, covariate_exp_ses=sc, settings="R_default"
)
wm_r = weighted_median_mr(bx, by, sy, se_exp=sx, settings="R_default")
mv_r = multivariable_mr(bx, sx, cov, by, sy, se_exp_covariates=sc, settings="R_default")
```

## `settings="default"` vs `settings="R_default"`

All public entrypoints (`weighted_median_mr`, `multivariable_mr`, `select_mr_instruments`, `harmonise_mr_data`, `ivw_mr`, `mr_egger`, `run_mr_suite`) accept `settings: Literal["default", "R_default"] = "default"`.

| Function / Parameter | `settings="default"` | `settings="R_default"` (`TwoSampleMR` / `MVMR`) |
| :--- | :--- | :--- |
| **`weighted_median_mr` weights** | 1st-order delta weights $w_j = \hat{\beta}_{X,j}^2 / s_{Y,j}^2$ | 2nd-order delta weights $w_j = \left(s_{Y,j}^2 / \hat{\beta}_{X,j}^2 + \hat{\beta}_{Y,j}^2 s_{X,j}^2 / \hat{\beta}_{X,j}^4\right)^{-1}$ |
| **`weighted_median_mr` `n_boot`** | `1000` | `1000` |
| **`multivariable_mr` $\hat{\delta}$ projection** | Exposure-variance-weighted WLS / IRLS | Unweighted OLS $\hat{\delta} = (C^\top C)^{-1} C^\top \hat{\beta}_{X,1}$ (`MVMR::strength_mvmr`) |
| **`multivariable_mr` residual variance** | Clamped $\max(1, \text{RSS} / (K - P))$ | Un-clamped $\text{RSS} / (K - P)$ (`MVMR::ivw_mvmr`) |
| **`multivariable_mr` $p$-values** | Two-sided Student-$t_{K-P}$ | Two-sided Student-$t_{K-P}$ |
| **`select_mr_instruments` `r2_threshold`** | `0.01` | `0.001` (`TwoSampleMR::clump_data`) |

Explicit caller-supplied arguments (`n_boot=...`, `r2_threshold=...`) always override mode defaults.

## Output Schema

- **`run_mr_suite(...)` -> `MRSummaryResult`** (frozen dataclass):
  - Scalar attributes: `exposure_name`, `outcome_name`, `n_instruments`, `mean_f_stat`, `conditional_f_stat`, `beta_ivw`, `se_ivw`, `pval_ivw`, `cochran_q`, `cochran_q_pval`, `beta_egger`, `se_egger`, `pval_egger`, `egger_intercept`, `se_egger_intercept`, `pval_egger_intercept`, `beta_weighted_median`, `se_weighted_median`, `pval_weighted_median`, `beta_mvmr`, `se_mvmr`, `pval_mvmr`.
  - `res.to_dataframe()` returns columns `["method", "beta", "se", "p_value"]`.
- **`ivw_mr(...)` -> `dict[str, float]`**: `{"beta", "se", "pval", "cochran_q", "cochran_q_pval"}`.
- **`mr_egger(...)` -> `dict[str, float]`**: `{"beta", "se", "pval", "intercept", "se_intercept", "pval_intercept"}`.
- **`weighted_median_mr(...)` -> `dict[str, float]`**: `{"beta", "se", "pval"}`.
- **`multivariable_mr(...)` -> `dict[str, float | np.ndarray]`**: `{"beta_primary", "se_primary", "pval_primary", "conditional_f_stat", "all_betas", "all_ses", "all_pvals"}`.
- **`harmonise_mr_data(...)` -> `pd.DataFrame`**: `["snp", "effect_allele", "other_allele", "beta_exp", "se_exp", "beta_out", "se_out", "eaf_exp", "eaf_out", "pval_exp", "pval_out", "palindromic"]`.

## Testing

```bash
PYTHONPATH=twosamplemr /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest twosamplemr/tests
```
