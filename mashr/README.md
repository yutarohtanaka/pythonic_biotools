# `mashr`: Data-Driven Multivariate Adaptive Shrinkage

Clean-room Python implementation of **mash** (*multivariate adaptive shrinkage*, Urbut et al., *Nature Genetics* 2019) and **Extreme Deconvolution** (Bovy, Hogg & Roweis, *Annals of Applied Statistics* 2011) for joint effect-size estimation, empirical Bayes shrinkage, local false sign rate (`lfsr`) inference, and cross-condition effect sharing across $R$ conditions.

## Citation

> Urbut, S. M., Wang, G., Carbonetto, P., & Stephens, M. (2019). Flexible statistical methods for estimating and testing effects in genomic studies with multiple conditions. *Nature Genetics*, 51(1), 187–195. https://doi.org/10.1038/s41588-018-0268-8

> Bovy, J., Hogg, D. W., & Roweis, S. T. (2011). Extreme deconvolution: Inferring complete distribution functions from noisy, heterogeneous and incomplete observations. *The Annals of Applied Statistics*, 5(2B), 1657–1677. https://doi.org/10.1214/10-AOAS439

## Mathematical & Algorithmic Overview

1. **Unified $\alpha$-Scale Model (`mash_set_data`, `mash_update_data`)**:
   Given $J \times R$ observed effect estimates $\hat{B}$, standard errors $\hat{S} > 0$, and residual correlation/covariance matrix $V \succ 0$, `mashr` operates on the transformed observations
   $$\tilde{b}_j = \hat{b}_j \oslash \hat{s}_j^\alpha, \qquad \tilde{s}_j = \hat{s}_j^{1-\alpha}, \qquad \tilde{V}_j = \operatorname{diag}(\tilde{s}_j)\,V\,\operatorname{diag}(\tilde{s}_j),$$
   supporting both Exchangeable Effects (**EE**, $\alpha = 0$) and Exchangeable $Z$-scores (**EZ**, $\alpha = 1$), while transforming posteriors back to the original $\beta$ scale via $a_j = \hat{s}_j^\alpha$.

2. **Null Correlation & Prior Covariance Library**:
   - `estimate_null_correlation_simple`: estimates $V$ from near-null units ($\max_r |Z_{j,r}| < z_{\text{thresh}}$) with eigenvalue flooring to guarantee strict positive definiteness.
   - `cov_canonical`: builds canonical prior matrices (`identity`, `singletons`, `equal_effects`, `simple_het`).
   - `cov_pca`: extracts truncated uncentered SVD/PCA rank-1 and rank-$K$ covariances from strong signals.
   - `cov_ed` / `fit_extreme_deconvolution`: refines data-driven covariances $\{U_k\}$ via Bovy's Extreme Deconvolution EM under heteroscedastic $\tilde{V}_j$ using Woodbury/Joseph-form PSD updates.

3. **Empirical Bayes Mixture Optimization & Exact Posteriors (`mash`)**:
   - Models $\tilde{\beta}_j \sim \pi_0 \delta_0 + \sum_{k=1}^K \sum_{l=1}^L \pi_{k,l}\,\mathcal{N}_R(0, \omega_l U_k)$ over an automated geometric scale grid (`autoselect_grid`).
   - Maximizes the Dirichlet-penalized mixture log-likelihood via projected SQUAREM (`optimize_pi`) with KKT optimality gap stopping.
   - Computes exact mixture posterior means, SDs, optional $(J, R, R)$ posterior covariances, exact point-mass-aware $\operatorname{lfsr}_{j,r} = 1 - \max(P_{\text{pos}}, P_{\text{neg}})$, significant results (`get_significant_results`), and pairwise effect sharing (`get_pairwise_sharing`).

## Installation

```bash
pip install -e ./mashr
```

## Quickstart

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
    get_significant_results,
    get_pairwise_sharing,
)

rng = np.random.default_rng(42)
Bhat = rng.normal(0.0, 1.0, size=(500, 4))
Shat = np.ones_like(Bhat)

data = mash_set_data(Bhat, Shat)
V_hat = estimate_null_correlation_simple(data)
data = mash_update_data(data, V=V_hat)

U_can = cov_canonical(data)
U_pca = cov_pca(data, npc=2)
U_ed = cov_ed(data, U_pca)

res = mash(data, {**U_can, **U_ed})
print(res.summary())
print(res.to_dataframe(sort_by_lfsr=True).head())
```

## License

MIT (clean-room implementation from published equations in Urbut et al., 2019 and Bovy et al., 2011).
