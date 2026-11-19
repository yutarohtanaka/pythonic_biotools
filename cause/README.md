# `cause`

Standalone Python implementation of **CAUSE** (*Causal Analysis Using Summary Effect Estimates*) for genome-wide two-sample Mendelian randomization accounting for correlated and uncorrelated horizontal pleiotropy via Bayesian grid integration and PSIS-LOO, with exact upstream R parity mode.

## Citation

> Morrison, J., Knoblauch, N., Marcus, J. H., Stephens, M., & He, X. (2020). Mendelian randomization accounting for correlated and uncorrelated pleiotropic effects using genome-wide summary statistics. *Nature Genetics*, 52(7), 700–706. https://doi.org/10.1038/s41588-020-0631-4

> Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432. https://doi.org/10.1007/s11222-016-9696-4

## Statistical Model

Given harmonized summary statistics $(\hat{\beta}_{1,j}, s_{1,j})$ for exposure $X$ and $(\hat{\beta}_{2,j}, s_{2,j})$ for outcome $Y$:

1. **Structural Mixture Model**:
   Each variant $j$ acts through a latent pleiotropic sharing indicator $Z_j \sim \text{Bernoulli}(q)$:
   $$\begin{pmatrix}\hat{\beta}_{1,j} \\ \hat{\beta}_{2,j}\end{pmatrix} \;\Big|\; Z_j, \gamma, \eta \;\sim\; \mathcal{N}\!\left(\begin{pmatrix}0 \\ 0\end{pmatrix},\; \begin{pmatrix}1 & 0 \\ \gamma + Z_j \eta & 1\end{pmatrix} \begin{pmatrix}\sigma_{1,k}^2 & 0 \\ 0 & \sigma_{2,k}^2\end{pmatrix} \begin{pmatrix}1 & \gamma + Z_j \eta \\ 0 & 1\end{pmatrix} + \begin{pmatrix}s_{1,j}^2 & \rho s_{1,j} s_{2,j} \\ \rho s_{1,j} s_{2,j} & s_{2,j}^2\end{pmatrix}\right),$$
   summed over empirical-Bayes bivariate variance components $\{(\sigma_{1,k}^2, \sigma_{2,k}^2, \pi_k)\}_{k=1}^K$ with sample-overlap correlation $\rho \in (-1, 1)$.

2. **Nuisance Parameter Estimation (`est_cause_params`)**:
   Estimates $\hat{\rho}$ and sparse `ash`-style bivariate variance weights $\hat{\boldsymbol{\pi}}$ across genome-wide variants with Dirichlet concentration weight `null_wt` on the $(0, 0)$ null component.

3. **Bayesian Posterior Integration & PSIS-LOO Comparison (`cause`)**:
   Compares the **Sharing** model ($\gamma = 0, \eta \sim \mathcal{N}(0, \sigma_g^2), q \sim \text{Beta}(q_\alpha, q_\beta)$) against the **Causal** model ($\gamma, \eta \sim \mathcal{N}(0, \sigma_g^2), q \sim \text{Beta}(q_\alpha, q_\beta)$) via two-stage adaptive quadrature, systematic posterior resampling, and Pareto-smoothed importance sampling leave-one-out cross-validation (`psis_loo`, `compare_elpd`):
   $$\Delta\text{ELPD} = \text{ELPD}_{\text{sharing}} - \text{ELPD}_{\text{causal}}, \qquad z = \frac{\Delta\text{ELPD}}{\text{SE}(\Delta\text{ELPD})}, \qquad p = \Phi(z).$$

## Python Quickstart

```python
import numpy as np
import pandas as pd
from cause import cause, est_cause_params, gwas_merge, ld_prune

rng = np.random.default_rng(0)
n = 2000
b1 = rng.normal(0.0, 0.015, size=n)
b1[:80] += rng.normal(0.0, 0.08, size=80)
s1 = np.full(n, 0.015)
s2 = np.full(n, 0.020)
b2 = 0.40 * b1 + rng.normal(0.0, 0.020, size=n)
df = pd.DataFrame({"snp": [f"rs{i}" for i in range(n)], "beta_hat_1": b1, "seb1": s1, "beta_hat_2": b2, "seb2": s2})

# 1. Default Pythonic mode
params = est_cause_params(df, settings="default")
top_snps = ld_prune(df, pval_thresh=1e-3)
res = cause(df, param_ests=params, variants=top_snps["snp"], settings="default")
print(res.summary())

# 2. Upstream R parity mode (matching jean997/cause)
params_r = est_cause_params(df, settings="R_default")
res_r = cause(df, param_ests=params_r, variants=top_snps["snp"], settings="R_default")
elpd_df = res_r.to_dataframe()
post_df = res_r.posterior_dataframe()
```

## `settings="default"` vs `settings="R_default"`

| Aspect | `settings="default"` | `settings="R_default"` (`jean997/cause`) |
| :--- | :--- | :--- |
| **1D variance/SD grid (`est_cause_params`)** | Geometric SD grid $\{0, \sigma_{\min}, \dots, \sigma_{\max}\}$ squared | Appends `2 * max` and `4 * max` tail SD points ($4\times$ and $16\times$ tail variances) |
| **Sample-overlap $\rho$ estimation (`est_cause_params`)** | Two-stage truncated bivariate normal MLE on null SNPs followed by SQUAREM/EM | Joint MAP optimization of $(\boldsymbol{\pi}, \rho)$ with upstream prior $\operatorname{arctanh}(\rho) \sim \mathcal{N}(0, 0.5^2)$ |
| **Effect prior SD `sigma_g` (`cause`)** | Robust $80\text{th}$-percentile instrument ratio scale | Upstream `eta_prior_sd`: $P(\|\mathcal{N}(0, \sigma_g)\| > \max\|\hat{\beta}_2 / \hat{\beta}_1\|) = 0.05$ across SNPs with $p_1 < 10^{-3}$ (with safe fallback if none pass $p_1 < 10^{-3}$) |
| **Default posterior draws `n_samples`** | `2000` | `1000` (when not explicitly overridden) |

### Parity Notes

- When `param_ests=None` is passed to `cause(..., settings="R_default")`, `cause` automatically forwards `settings="R_default"` to `est_cause_params`.
- Under `settings="R_default"`, $\sigma_g = \max_{p_1 < 10^{-3}} |\hat{\beta}_2 / \hat{\beta}_1| / \Phi^{-1}(0.975)$ satisfies $2(1 - \Phi(M_{\text{ratio}} / \sigma_g)) = 0.05$ identically, and joint $(\boldsymbol{\pi}, \rho)$ MAP estimation regularizes $\operatorname{arctanh}(\rho)$ toward zero under small sample sizes via $\mathcal{N}(0, 0.5^2)$.

## Output Schema

- **`CauseParams`** (returned by `est_cause_params`):
  - Attributes: `rho`, `S1`, `S2`, `pi`, `n_snps`, `n_null_snps_rho`, `log_likelihood`, `converged`, `n_iter`, `grid_S1`, `grid_S2`, `settings`.
  - `params.to_dataframe()`: Active mixture grid `["S1", "S2", "pi"]`.
- **`CauseResult`** (returned by `cause`):
  - Attributes: `n_instruments`, `snp_ids`, `beta_hat_1`, `seb1`, `beta_hat_2`, `seb2`, `params`, `sigma_g`, `qalpha`, `qbeta`, `max_q`, `elpd_null`, `elpd_sharing`, `elpd_causal`, `delta_elpd`, `se_delta_elpd`, `z_score`, `p_value`, `sharing` (`CauseModelFit`), `causal` (`CauseModelFit`), `loo_null` (`PSISLOOResult`), `elpd_table`, `n_samples`, `settings`.
  - `res.to_dataframe()`: Model comparison table `["model1", "model2", "delta_elpd", "se_delta_elpd", "z", "p_value"]`.
  - `res.posterior_dataframe()`: Posterior parameter summaries `["model", "parameter", "median", "ci_low", "ci_high", "mean", "sd"]`.
  - `res.snp_table()`: Per-instrument sharing probabilities and pointwise PSIS-LOO diagnostics `["snp", "beta_hat_1", "seb1", "beta_hat_2", "seb2", "pr_z_sharing", "pr_z_causal", "elpd_null", "elpd_sharing", "elpd_causal", "pareto_k_sharing", "pareto_k_causal"]`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/cause /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest cause/tests
```

## License

GPL-3.0-or-later (matching upstream `jean997/cause`).
