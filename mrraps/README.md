# `mrraps`: Robust Adjusted Profile Score Mendelian Randomization (`mr.raps`)

Standalone Python implementation of **`mr.raps`** (*Robust Adjusted Profile Score*) for two-sample summary-data Mendelian Randomization with many weak instruments, systematic pleiotropic overdispersion ($\tau^2$), and idiosyncratic pleiotropic outliers (`l2`, `huber`, and `tukey` biweight losses).

## Citation

- **MR-RAPS**: Zhao, Q., Wang, J., Hemani, G., Bowden, J., & Small, D. S. (2020). Statistical inference in two-sample summary-data Mendelian randomization using robust adjusted profile score. *The Annals of Statistics*, 48(3), 1742–1769. https://doi.org/10.1214/19-AOS1866
- **Upstream R package**: `qingyuanzhao/mr.raps` (`mr.raps`, `mr.raps.shrinkage`, `mr.raps.mle`).

## Statistical Model

Given $m$ independent genetic instruments with two-sample summary associations $\hat{\gamma}_j \sim \mathcal{N}(\gamma_j, \sigma_{Xj}^2)$ and $\hat{\Gamma}_j \sim \mathcal{N}(\beta \gamma_j + \alpha_j, \sigma_{Yj}^2)$ where systematic pleiotropy follows $\alpha_j \sim \mathcal{N}(0, \tau^2)$, define standardized profile residuals and empirical-Bayes shrunken instrument strengths:
$$t_j(\beta, \tau^2) = \frac{\hat{\Gamma}_j - \beta \hat{\gamma}_j}{\sqrt{\beta^2 \sigma_{Xj}^2 + \sigma_{Yj}^2 + \tau^2}}, \qquad \hat{\gamma}_j(\beta, \tau^2) = \frac{\hat{\gamma}_j / \sigma_{Xj}^2 + \beta \hat{\Gamma}_j / (\sigma_{Yj}^2 + \tau^2)}{1 / \sigma_{Xj}^2 + \beta^2 / (\sigma_{Yj}^2 + \tau^2)}.$$
For bounded/robust loss $\rho(\cdot)$ (`l2`, `huber` with $k = 1.345$, or `tukey` with $k = 4.685$) with score derivative $\psi = \rho'$ and Gaussian consistency constant $\delta = \mathbb{E}[Z \psi(Z)]$, `raps_shrinkage` solves the unbiased profile-score estimating equations $\Psi(\beta, \tau^2) = \mathbf{0}$:
$$\psi_1(\beta, \tau^2) = \sum_{j=1}^m \frac{\hat{\gamma}_j(\beta, \tau^2) \, \psi(t_j(\beta, \tau^2))}{\sqrt{\beta^2 \sigma_{Xj}^2 + \sigma_{Yj}^2 + \tau^2}} = 0, \qquad \psi_2(\beta, \tau^2) = \sum_{j=1}^m \frac{t_j(\beta, \tau^2)\psi(t_j(\beta, \tau^2)) - \delta}{\beta^2 \sigma_{Xj}^2 + \sigma_{Yj}^2 + \tau^2} = 0,$$
with sandwich asymptotic covariance $V = V_2^{-1} V_1 V_2^{-\top}$. Classical profile-likelihood estimators (`estimator="mle"`) are also provided via `raps_mle`.

## Python Quickstart

```python
import numpy as np
from mrraps import mr_raps, mr_raps_shrinkage

rng = np.random.default_rng(21)
m = 100
gamma = rng.normal(0.0, 0.06, m)
se_exp = np.full(m, 0.015)
se_out = np.full(m, 0.020)
b_exp = gamma + rng.normal(0.0, se_exp)
b_out = 0.45 * gamma + rng.normal(0.0, 0.025, m) + rng.normal(0.0, se_out)

# Scale-equivariant Pythonic default
fit_default = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function="huber", seed=5)
print(fit_default["beta_hat"], fit_default["beta_se"], fit_default["beta_p_value"])

# Exact upstream R mr.raps.shrinkage parity mode
fit_r = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function="huber", seed=5, settings="R_default")
print(fit_r["beta_hat"], fit_r["tau2_hat"])
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`mr_raps`, `mr_raps_shrinkage`, `raps_shrinkage`, `raps_mle`, `raps_simple`, `raps_simple_robust`, `raps_overdispersed`, `raps_overdispersed_robust`) accept `settings: Literal["default", "R_default"] = "default"`.

| Feature | `settings="default"` | `settings="R_default"` (`qingyuanzhao/mr.raps` Parity) |
| :--- | :--- | :--- |
| **`raps_shrinkage` upper $\tau^2$ root bound** | Variance-homogeneous bound $\hat{\tau}^2 \le 100 \cdot \operatorname{median}(\mathrm{se\_out}^2)$ | Exact upstream `mr.raps.shrinkage` bound $\hat{\tau}^2 \le 10 \cdot \operatorname{median}(\mathrm{se\_out})$ |
| **Scale equivariance under outcome rescaling** | Exact equivariance under arbitrary rescaling $(b_Y, s_Y) \mapsto c(b_Y, s_Y)$ | Matches upstream R behavior (rejects $\hat{\tau}^2 > 10 \operatorname{median}(s_Y)$ and falls back to $\tau^2 = 0$ when $s_Y$ units are large) |

## Output Schema

`mr_raps(..., estimator="shrinkage")` / `raps_shrinkage` returns a `dict` with keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `beta_hat` | `float` | Point estimate of causal effect $\hat{\beta}$ |
| `beta_se` | `float` | Sandwich asymptotic standard error of $\hat{\beta}$ |
| `beta_p_value` | `float` | Two-sided Wald normal p-value for $H_0: \beta = 0$ |
| `tau2_hat` | `float` | Estimated pleiotropic overdispersion variance $\hat{\tau}^2$ (`0.0` when `over_dispersion=False`) |
| `tau2_se` | `float` | Asymptotic standard error of $\hat{\tau}^2$ |
| `beta_init` | `float` | Initial `raps_mle` causal estimate used to select the closest valid score root |
| `n_roots_other` | `int` | Count of distinct alternative score roots found across random restarts |
| `other_roots` | `list[float]` | Alternative $\hat{\beta}$ roots (if multiple roots exist) |

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mrraps \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mrraps/tests
```
