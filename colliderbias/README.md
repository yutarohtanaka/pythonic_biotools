# `colliderbias`: CWLS and CWBLS Estimators for Collider & Index-Event Bias

Clean-room Python implementation of the R **`ColliderBias`** package (Cai et al., *Genetic Epidemiology* 2022; Cai & Dudbridge, 2024), providing Corrected Weighted Least Squares (`method_cb` / `cwls`) and Corrected Weighted Bivariate Least Squares (`cwbls`) to estimate and remove collider / selection bias in conditioned or disease-progression GWAS summary statistics.

## Citation

> Cai, S., Hartley, A., Mahmoud, O., Tilling, K., & Dudbridge, F. (2022). Adjusting for collider bias in genetic association studies using instrumental variable methods. *Genetic Epidemiology*, 46(5–6), 303–316. https://doi.org/10.1002/gepi.22455

- Upstream R reference: `SiyangCai/ColliderBias` (`methodCB.R`, `CWBLS.R`).

## Statistical Model

1. **Corrected Weighted Least Squares (`method_cb` / `cwls`)**:
   When a subsequent-trait GWAS $\hat{\beta}_{Y,j}$ is conditioned on an index/collider trait $X$, the induced collider slope $b$ is estimated across independent index-trait instruments ($j \in \mathcal{P}$) under either zero-intercept (`model="ivw"`) or free-intercept (`model="egger"`) weighted regression with regression-dilution (Hedges-Olkin) correction:
   $$\hat{b} = \hat{b}_{\text{WLS}} \cdot \hat{\kappa}_{\text{HO}}, \qquad \hat{\kappa}_{\text{HO}}^{\text{IVW}} = \frac{\sum w_j \hat{\beta}_{X,j}^2}{\sum w_j \hat{\beta}_{X,j}^2 - \sum w_j s_{X,j}^2}.$$
   Genome-wide outcome effects are then adjusted via $\hat{\beta}_{Y,j}^{\text{adj}} = \hat{\beta}_{Y,j} - \hat{b}\,\hat{\beta}_{X,j}$.

2. **Corrected Weighted Bivariate Least Squares (`cwbls`)**:
   Jointly estimates the direct effect of an exposure ($\hat{\beta}_X$) and collider/disease trait ($\hat{\beta}_D$) on an outcome ($\hat{\beta}_Y$) with inverse-outcome-variance weighting $w_j = 1 / s_{Y,j}^2$, subtracting both $\sum w_j s_{X,j}^2$ and $\sum w_j s_{D,j}^2$ from the normal equations, with nonparametric bootstrap standard errors (`n_boot`).

## Python Quickstart

```python
import numpy as np
from colliderbias import method_cb, cwls, cwbls

rng = np.random.default_rng(7)
m = 200
x_true = rng.normal(0.0, 0.08, m)
se = np.full(m, 0.015)
xbeta = x_true + rng.normal(0.0, se)
ybeta = -0.40 * x_true + rng.normal(0.0, se)

# Pythonic default mode (coding-invariant Egger, 5-term adjusted SE)
res_def = method_cb(xbeta, se, ybeta, se, weighted="2", model="ivw", settings="default")
print(res_def["b"], res_def["b_se"])

# Exact upstream R parity mode (ColliderBias::methodCB)
res_r = method_cb(xbeta, se, ybeta, se, weighted="2", model="ivw", settings="R_default")
print(res_r["b"], res_r["b_se"])

# Bivariate CWBLS (exposure X + collider D -> outcome Y)
dbeta = 0.5 * x_true + rng.normal(0.0, se)
biv = cwbls(xbeta, se, dbeta, se, ybeta, se, n_boot=100, seed=42)
print(biv["b"], biv["beta_d"])
```

## `settings="default"` vs `settings="R_default"`

`method_cb`, `cwls`, and `cwbls` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values):

| Feature | `settings="default"` | `settings="R_default"` (`ColliderBias::methodCB`) |
| :--- | :--- | :--- |
| **`model="egger"` Allele Orientation** | Orients $\hat{\beta}_{X,j} \ge 0$ for allele-coding invariance | Raw signed regression (no sign re-orientation) |
| **Second-Order Weights (`weighted="2"`)** | Iterated 1-step slope $1 / (s_{Y,j}^2 + \hat{b}_1^2 s_{X,j}^2 - 2\hat{b}_1 \rho s_{X,j}s_{Y,j})$ | Direct ratio weights $1 / (s_{Y,j}^2 + (\hat{\beta}_{Y,j} / \tilde{\beta}_{X,j})^2 s_{X,j}^2)$ with $10^{-12}$ zero guard |
| **Adjusted Outcome SE (`yse_adj`)** | Full 5-term propagation $\sqrt{s_Y^2 + b^2 s_X^2 + s_b^2 \hat{\beta}_X^2 + s_b^2 s_X^2 - 2 b \rho s_X s_Y}$ | 2-term plug-in $\sqrt{s_{Y,j}^2 + \hat{b}^2 s_{X,j}^2}$ (`methodCB.R`) |

## Output Schema

### `method_cb` / `cwls` -> `dict`
- `ybeta_adj`, `yse_adj`, `ychisq_adj`, `yp_adj` (`np.ndarray`): Bias-adjusted outcome $\hat{\beta}$, SE, $\chi^2_1$ statistics, and $p$-values.
- `b`, `b_se`, `b_pval` (`float`), `b_ci` (`list[float]`): Regression-dilution-corrected collider slope, SE, $p$-value, and 95% CI.
- `b_raw`, `b_raw_se`, `ho_factor` (`float`): Uncorrected WLS slope, SE, and Hedges-Olkin factor.

### `cwbls` -> `dict`
- `b`, `b_se`, `b_ci`, `pval`, `b_pval`: Direct exposure effect estimate, bootstrap SE, 95% CI, and $p$-value.
- `beta_d`, `beta_d_se`, `beta_d_ci`, `beta_d_pval`: Direct collider effect estimate, bootstrap SE, 95% CI, and $p$-value.
- `boot`, `boot_d` (`np.ndarray`): Bootstrap replicate arrays.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/colliderbias \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest colliderbias/tests -v
```
