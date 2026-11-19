# `mashr`: Data-Driven Multivariate Adaptive Shrinkage

Clean-room Python implementation of **mash** (*multivariate adaptive shrinkage*, Urbut et al., *Nature Genetics* 2019) and **Extreme Deconvolution** (Bovy, Hogg & Roweis, *Annals of Applied Statistics* 2011) for joint effect-size estimation, empirical Bayes shrinkage, local false sign rate (`lfsr`) inference, and cross-condition effect sharing across $R$ conditions.

## Citation

> Urbut, S. M., Wang, G., Carbonetto, P., & Stephens, M. (2019). Flexible statistical methods for estimating and testing effects in genomic studies with multiple conditions. *Nature Genetics*, 51(1), 187–195. https://doi.org/10.1038/s41588-018-0268-8

> Bovy, J., Hogg, D. W., & Roweis, S. T. (2011). Extreme deconvolution: Inferring complete distribution functions from noisy, heterogeneous and incomplete observations. *The Annals of Applied Statistics*, 5(2B), 1657–1677. https://doi.org/10.1214/10-AOAS439

> Kim, Y., Carbonetto, P., Stephens, M., & Anitescu, M. (2020). A fast algorithm for maximum likelihood estimation of mixture proportions using sequential quadratic programming. *Journal of Computational and Graphical Statistics*, 29(2), 261–273. https://doi.org/10.1080/10618600.2019.1689985

## Statistical Model

1. **Unified $\alpha$-Scale Observation Model** (`mash_set_data`, `mash_update_data`):
   Given $J \times R$ observed effect estimates $\hat{B}$, standard errors $\hat{S} > 0$, and residual correlation/covariance matrix $V \succ 0$, `mashr` models the transformed observations
   $$\tilde{b}_j = \hat{b}_j \oslash \hat{s}_j^\alpha, \qquad \tilde{s}_j = \hat{s}_j^{1-\alpha}, \qquad \tilde{V}_j = \operatorname{diag}(\tilde{s}_j)\,V\,\operatorname{diag}(\tilde{s}_j),$$
   supporting Exchangeable Effects (**EE**, $\alpha = 0$) and Exchangeable $Z$-scores (**EZ**, $\alpha = 1$), transforming posteriors back to the original $\beta$ scale via $\hat{s}_j^\alpha$. Finite Student-$t$ degrees of freedom (`df`) are converted to equivalent Gaussian standard errors via exact two-sided tail probability matching.

2. **Null Correlation & Prior Covariance Library**:
   - `estimate_null_correlation_simple`: estimates $V$ from near-null units ($\max_r |Z_{j,r}| < z_{\text{thresh}}$) with SPD eigenvalue flooring.
   - `cov_canonical`: canonical prior matrices (`identity`, `singletons`, `equal_effects`, `simple_het`).
   - `cov_pca`: uncentered SVD rank-1 and rank-$K$ `tPCA` covariances from strong signals.
   - `cov_ed` / `fit_extreme_deconvolution`: Bovy's Extreme Deconvolution EM refinement under heteroscedastic $\tilde{V}_j$ via Woodbury/Joseph-form PSD updates.

3. **Empirical Bayes Mixture Optimization & Exact Posteriors** (`optimize_pi`, `mash`):
   Models $\tilde{\beta}_j \sim \pi_0 \delta_0 + \sum_{k=1}^K \sum_{l=1}^L \pi_{k,l}\,\mathcal{N}_R(0, \omega_l U_k)$ over an automated geometric scale grid (`autoselect_grid`), maximizes the Dirichlet-penalized log-likelihood
   $$\sum_{j=1}^J \log \left(\sum_{m=0}^{M-1} \pi_m L_{j,m}\right) + \sum_{m=0}^{M-1} (\alpha_m - 1)\log \pi_m$$
   via active-set `mixSQP` (or `squarem` / `em`), and computes exact mixture posterior moments and point-mass-aware $\operatorname{lfsr}_{j,r} = 1 - \max(P(\beta_{j,r} > 0 \mid \hat{B}), P(\beta_{j,r} < 0 \mid \hat{B}))$.

## Python Quickstart

```python
import numpy as np
from mashr import (
    mash_set_data,
    mash_update_data,
    estimate_null_correlation_simple,
    cov_canonical,
    cov_pca,
    cov_ed,
    mash,
)

rng = np.random.default_rng(42)
Bhat = rng.normal(0.0, 1.0, size=(500, 4))
Bhat[:50, :] += 2.5
Shat = np.ones_like(Bhat)

# Default Pythonic mode
data = mash_set_data(Bhat, Shat, settings="default")
V_hat = estimate_null_correlation_simple(data)
data = mash_update_data(data, V=V_hat)

U_can = cov_canonical(data)
U_ed = cov_ed(data, cov_pca(data, npc=2))
res = mash(data, {**U_can, **U_ed}, settings="default")
print(res.summary())

# Upstream R parity mode (stephenslab/mashr + mixsqp)
data_r = mash_set_data(Bhat, Shat, V=V_hat, settings="R_default")
res_r = mash(data_r, {**U_can, **U_ed}, settings="R_default")
print(res_r.to_dataframe(sort_by_lfsr=True).head())
```

## `settings="default"` vs `settings="R_default"`

Primary entrypoints (`mash_set_data`, `mash_update_data`, `optimize_pi`, `mash`) accept `settings: Literal["default", "R_default"] = "default"`. When `data` is created with `mash_set_data(..., settings="R_default")`, `mash(data, ...)` automatically inherits `data.settings` unless overridden by an explicit `settings=...` argument.

| Feature | `settings="default"` | `settings="R_default"` (`stephenslab/mashr`) |
| :--- | :--- | :--- |
| **`mixSQP` KKT convergence `tol`** | `1e-6` | Strict `1e-8` (`mixsqp` default `eps = 1e-8`) |
| **Default Dirichlet `prior`** | `"nullbiased"` (`[10, 1, ..., 1]`) | `"nullbiased"` (`[10, 1, ..., 1]` with `nullweight=10.0`) |
| **`mash_set_data` `Shat <= 0` & `df`** | Strict `Shat > 0` check (`zero_shat_reset` optional) + finite-`df` $t$-to-$Z$ conversion | Exact `mashr::mash_set_data` validation + finite-`df` two-sided $t$-tail conversion |

Explicit user overrides (`tol=...`, `prior=...`, `nullweight=...`) are respected in both modes.

## Output Schema

`mash(...)` returns a frozen `MashResult` dataclass with structured array attributes and DataFrame helpers:

- **Core `(J, R)` arrays**: `posterior_mean`, `posterior_sd`, `lfsr`, `negative_prob`, `positive_prob`, `zero_prob` (plus optional `(J, R, R)` `posterior_cov`).
- **Model fit attributes**: `pi` (`(M,)`), `posterior_weights` (`(J, M)`), `grid` (`(L,)`), `U_list`, `loglik_matrix` (`(J, M)`), `loglik`, `penalized_loglik`, `null_loglik`, `converged`, `n_iter`, `kkt_gap`, `settings`.
- **`res.to_dataframe(sort_by_lfsr=False)` / `res.significant(thresh=0.05)`**:
  - Columns: `["feature", "condition", "posterior_mean", "posterior_sd", "lfsr", "negative_prob", "positive_prob", "zero_prob"]`.
- **Matrix DataFrames**: `res.posterior_mean_dataframe()`, `res.posterior_sd_dataframe()`, `res.lfsr_dataframe()`, `res.estimated_pi_dataframe()` (`["component", "weight"]`), and `res.pairwise_sharing_dataframe()`.

## Testing

```bash
PYTHONPATH=mashr /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mashr/tests
```
