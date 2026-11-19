# `indexevent`: Adjustment for Index-Event Bias in Genetic Association Studies

Clean-room Python implementation of the R **`indexevent`** package (Dudbridge et al., *Nature Communications* 2019), providing Corrected Weighted Least Squares (`CWLS`), `Hedges-Olkin`, and Simulation-Extrapolation (`Simex`, with profile-likelihood confidence intervals) estimators for removing index-event bias in subsequent-trait GWAS summary statistics.

## Citation

> Dudbridge, F., Allen, R. J., Sheehan, N. A., Schmidt, A. F., Lee, J. C., Jenkins, R. G., Wain, L. V., Hingorani, A. D., & Patel, R. S. (2019). Adjustment for index event bias in genome-wide association studies of subsequent events. *Nature Communications*, 10, 1561. https://doi.org/10.1038/s41467-019-09381-w

- Upstream R reference: `FrankDudbridge/indexevent` (`R/indexevent.R`).

## Statistical Model

Given index-trait GWAS summary statistics $(\hat{\beta}_{X,j}, s_{X,j})$ and subsequent-event GWAS summary statistics $(\hat{\beta}_{Y,j}, s_{Y,j})$ conditioned on the index event:

1. **Regression-Dilution-Corrected Slope Estimation**:
   Across independent SNPs ($j \in \mathcal{P}$) with weights $w_j = 1 / s_{Y,j}^2$ (or $w_j = 1$ if `weighted=False`), WLS fits $\hat{\beta}_{Y,j} = \alpha + \beta_{\text{raw}} \hat{\beta}_{X,j} + \varepsilon_j$, and corrects regression dilution from exposure measurement error $s_{X,j}^2$ via one of three estimators:
   - **`CWLS`**: $\hat{b} = \frac{(\sum w_j)(\sum w_j \hat{\beta}_X \hat{\beta}_Y) - (\sum w_j \hat{\beta}_X)(\sum w_j \hat{\beta}_Y)}{(\sum w_j)(\sum w_j \hat{\beta}_X^2) - (\sum w_j \hat{\beta}_X)^2 - (\sum w_j)(\sum w_j s_X^2)}$.
   - **`Hedges-Olkin`**: $\hat{b} = \hat{\beta}_{\text{raw}} \cdot \frac{\operatorname{Var}(\hat{\beta}_X)}{\operatorname{Var}(\hat{\beta}_X) - \overline{s_X^2}}$.
   - **`Simex`**: Simulates pseudo-exposures $\tilde{\beta}_{X,j}(\lambda) \sim \mathcal{N}(\hat{\beta}_{X,j}, \lambda s_{X,j}^2)$ across a grid of $\lambda > 0$ ($B$ replicates each), fits the rational extrapolation curve $\mathbb{E}[\hat{\beta}(\lambda)] = \frac{b}{1 + (\lambda + 1)\gamma}$ by maximum likelihood, and inverts the profile likelihood (`simexprofilellhd`, `simexprofileCI`) for 95% confidence limits.

2. **Summary-Statistic Adjustment**:
   $$\hat{\beta}_{Y,j}^{\text{adj}} = \hat{\beta}_{Y,j} - \hat{b} \hat{\beta}_{X,j}, \qquad s_{Y,j}^{\text{adj}} = \sqrt{s_{Y,j}^2 + \hat{b}^2 s_{X,j}^2 + s_{\hat{b}}^2 \hat{\beta}_{X,j}^2 + s_{\hat{b}}^2 s_{X,j}^2}.$$

## Python Quickstart

```python
import numpy as np
from indexevent import indexevent

rng = np.random.default_rng(19)
m = 200
x_true = rng.normal(0.0, 0.06, m)
xse = np.full(m, 0.015)
yse = np.full(m, 0.015)
xbeta = x_true + rng.normal(0.0, xse)
ybeta = -0.40 * x_true + rng.normal(0.0, yse)

# Pythonic default mode (allele-coding invariant orientation xb >= 0)
res_cwls = indexevent(xbeta, xse, ybeta, yse, method="CWLS", settings="default")
print(res_cwls["b"], res_cwls["b_se"], res_cwls["b_ci"])

# Exact upstream R parity mode (FrankDudbridge/indexevent raw signed WLS)
res_r = indexevent(xbeta, xse, ybeta, yse, method="CWLS", settings="R_default")
print(res_r["b"], res_r["b_se"])

# SIMEX with profile-likelihood CI
res_simex = indexevent(xbeta, xse, ybeta, yse, method="Simex", B=20, seed=2018)
print(res_simex["b"], res_simex["b_ci"])
```

## `settings="default"` vs `settings="R_default"`

`indexevent` accepts `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values):

| Feature | `settings="default"` | `settings="R_default"` (`FrankDudbridge/indexevent`) |
| :--- | :--- | :--- |
| **Allele Sign Orientation** | Pre-orients $\hat{\beta}_{X,j} \ge 0$ (multiplying $(\hat{\beta}_{X,j}, \hat{\beta}_{Y,j})$ by $\operatorname{sign}(\hat{\beta}_{X,j})$) so intercept-bearing WLS/SIMEX is invariant to arbitrary reference-allele coding | Raw signed WLS on $(\hat{\beta}_{X,j}, \hat{\beta}_{Y,j})$ without sign orientation (`indexevent.R`) |
| **Adjusted Variance & SIMEX Profile CI** | Exact 4-term `yse_adj` and exact $\chi^2_1$ profile-likelihood root bracketing | Identical |

## Output Schema

### `indexevent(...) -> dict`
- `ybeta_adj` (`np.ndarray`): Adjusted subsequent-event effect estimates.
- `yse_adj` (`np.ndarray`): Adjusted standard errors incorporating slope estimation uncertainty.
- `ychisq_adj` (`np.ndarray`), `yp_adj` (`np.ndarray`): Adjusted $\chi^2_1$ test statistics and $p$-values.
- `b` (`float`), `b_se` (`float`), `b_ci` (`np.ndarray` of shape `(2,)`): Dilution-corrected slope, SE, and 95% CI.
- `b_raw` (`float`): Uncorrected WLS slope.
- `simex_estimates` (`np.ndarray | None`): `(L + 1, 3)` matrix `[lambda, mean_slope, sandwich_var]` when `method="Simex"`, else `None`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/indexevent \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest indexevent/tests -v
```
