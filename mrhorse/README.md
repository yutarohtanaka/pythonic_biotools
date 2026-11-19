# `mrhorse`: Bayesian Horseshoe Mendelian Randomization (`MR-Horse` & `MVMR-Horse`)

Standalone Python implementation of **MR-Horse** and **MVMR-Horse** for univariable and multivariable summary-data Bayesian Mendelian Randomization accounting for both correlated and uncorrelated horizontal pleiotropy via a Horseshoe shrinkage prior, with zero external JAGS dependency.

## Citation

- **MR-Horse & MVMR-Horse**: Grant, A. J., & Burgess, S. (2024). A Bayesian approach to Mendelian randomization using summary statistics in the univariable and multivariable settings with correlated pleiotropy. *The American Journal of Human Genetics*, 111(1), 165–180. https://doi.org/10.1016/j.ajhg.2023.12.002
- **Upstream R/JAGS reference**: `aj-grant/mrhorse` (`mr_horse.R`).

## Statistical Model

For $i = 1, \dots, N$ genetic variants and $K \ge 1$ exposures with summary estimates $(\hat{\beta}_{Yi}, s_{Yi})$ and $(\hat{\boldsymbol{\beta}}_{Xi}, \mathbf{s}_{Xi})$:
$$\hat{\beta}_{Yi} \mid \boldsymbol{\beta}_{X0,i}, \alpha_i, \boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\beta}_{X0,i}^\top \boldsymbol{\theta} + \alpha_i, \; s_{Yi}^2), \qquad \hat{\beta}_{Xik} \mid \beta_{X0,ik} \sim \mathcal{N}(\beta_{X0,ik}, \; s_{Xik}^2).$$
Latent true exposure associations $\beta_{X0,ik}$ and variant-specific pleiotropic effects $\alpha_i$ follow a joint Gaussian-Horseshoe hierarchy capturing correlated pleiotropy (e.g. heritable confounders):
$$\alpha_i \mid \tau, \phi_i \sim \mathcal{N}(0, \tau^2 \phi_i^2), \qquad \phi_i \sim \mathcal{C}^+(0, 1), \qquad \tau \sim \mathcal{C}^+(0, 1),$$
$$\operatorname{Cov}\!\begin{pmatrix} \alpha_i \\ \boldsymbol{\beta}_{X0,i} \end{pmatrix} = \begin{pmatrix} \tau^2 \phi_i^2 & \tau \phi_i \boldsymbol{\rho}_i^\top S \\ \tau \phi_i S \boldsymbol{\rho}_i & S^2 \end{pmatrix}, \qquad S = \operatorname{diag}(\sqrt{v_{x0,1}}, \dots, \sqrt{v_{x0,K}}),$$
with $\theta_k \sim \operatorname{Uniform}(-10, 10)$ and shifted $\operatorname{Beta}(10, 10)$ priors on pleiotropy–exposure correlations $\rho$. Posterior inference is performed via conjugate block-Gibbs updates for $(\boldsymbol{\beta}_{X0}, \boldsymbol{\alpha}, \boldsymbol{\theta}, \mathbf{m}_x)$, vectorized griddy-Gibbs updates for $\boldsymbol{\rho}$, and adaptive random-walk Metropolis updates for $(\boldsymbol{\phi}, \tau, \mathbf{v}_{x0})$, summarized with multi-chain Gelman–Rubin $\hat{R}$ diagnostics.

## Python Quickstart

```python
import numpy as np
from mrhorse import mr_horse, mvmr_horse

rng = np.random.default_rng(42)
n, k = 50, 2
bx_true = rng.normal(0.0, 0.12, (n, k))
sx = np.full((n, k), 0.015)
sy = np.full(n, 0.020)
bx = bx_true + rng.normal(0.0, sx)
by = bx_true @ np.array([0.40, -0.25]) + rng.normal(0.0, sy)

# Univariable MR-Horse
res_uni = mr_horse(by, sy, bx[:, 0], sx[:, 0], n_chains=3, n_iter=2000, n_burnin=2000, seed=1)
print(res_uni["estimate"][0])

# Multivariable MVMR-Horse: Pythonic default vs upstream R/JAGS parity mode
res_mv_default = mvmr_horse(by, sy, bx, sx, n_chains=3, n_iter=2000, n_burnin=2000, seed=2, settings="default")
res_mv_r = mvmr_horse(by, sy, bx, sx, n_chains=3, n_iter=2000, n_burnin=2000, seed=2, settings="R_default")
print(res_mv_r["estimate"])
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`mr_horse`, `mvmr_horse`, `horse`) accept `settings: Literal["default", "R_default"] = "default"` and optional `n_thin: int | None = None`.

| Feature | `settings="default"` | `settings="R_default"` (`aj-grant/mrhorse` Parity) |
| :--- | :--- | :--- |
| **Post-burnin thinning (`n_thin=None`)** | Auto-thins to `max(1, n_iter // 1000)` (~1,000 draws/chain for large `n_iter`) | Keeps all post-burnin draws unthinned (`n_thin = 1` by default) |
| **MVMR pleiotropy correlation $\boldsymbol{\rho}_i$** | Per-exposure vector $(\rho_{i1}, \dots, \rho_{iK})$ subject to $\|\boldsymbol{\rho}_i\|_2 < \sqrt{1 - 10^{-6}}$ | Shared scalar $\rho_i$ per SNP across all $K$ exposures (`mr_horse.R:28-55`), truncated to $|\rho_i| < \frac{1}{\sqrt{K}} - 10^{-6}$ so $1 - K\rho_i^2 > 0$ |
| **Explicit `n_thin` override** | Respected when `n_thin` is passed explicitly | Respected when `n_thin` is passed explicitly |

## Output Schema

`mr_horse`, `mvmr_horse`, and `horse` return a `dict` with keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `estimate` | `list[dict]` | Per-exposure summary dicts containing `Parameter` (`"theta[k]"`), `Estimate` (posterior mean), `SD`, `q2_5`, `q97_5`, and `Rhat` (Gelman–Rubin PSRF) |
| `samples` | `dict[str, list[np.ndarray]]` | Per-chain 1D posterior draw arrays for each `"theta[k]"` |
| `extra` | `dict[str, list[np.ndarray]]` | Optional saved latent parameter traces requested via `save=(...)` (`"alpha"`, `"rho"`, `"phi"`, `"tau"`, `"vx0"`, `"mx"`, `"bx0"`) |
| `thin` | `int` | Effective thinning interval applied to post-burnin iterations |

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mrhorse \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mrhorse/tests
```
