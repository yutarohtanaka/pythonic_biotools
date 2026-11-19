# `cause`: Causal Analysis Using Summary Effect Estimates

Standalone Python implementation of **CAUSE** (Morrison et al., *Nature Genetics* 2020) for genome-wide two-sample Mendelian randomization distinguishing causal effects from correlated and uncorrelated horizontal pleiotropy via Bayesian model comparison and PSIS-LOO.

## Citation

> Morrison, J., Knoblauch, N., Marcus, J. H., Stephens, M., & He, X. (2020). Mendelian randomization accounting for correlated and uncorrelated pleiotropic effects using genome-wide summary statistics. *Nature Genetics*, 52(7), 700–706. https://doi.org/10.1038/s41588-020-0631-4

> Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.

## Mathematical & Algorithmic Overview

1. **GWAS Harmonization (`gwas_merge`, `format_cause_data`, `ld_prune`)**:
   Aligns exposure and outcome summary statistics across effect/other alleles (supporting allele swaps, strand complementation, and palindromic SNP filtering) and performs deterministic $p$-value-ordered LD pruning.

2. **Genome-Wide Nuisance Parameter Estimation (`est_cause_params`)**:
   - **Sample-overlap correlation ($\hat{\rho}$)**: Maximizes the exact truncated bivariate normal log-likelihood over null SNPs ($|z_1|, |z_2| < c$) using 32-point Gauss-Legendre quadrature to eliminate selection attenuation bias.
   - **Bivariate `ash`-style variance mixture ($\hat{\boldsymbol{\pi}}$)**: Fits an empirical-Bayes bivariate zero-mean Gaussian mixture over outer-product variance grids $(S_{1,k}, S_{2,k})$ via SQUAREM/EM with Dirichlet null weighting (`null_wt`).

3. **Posterior Grid Integration (`cause`)**:
   Models SNP $j$'s structural slope as $\gamma$ with probability $1 - q$ and $\gamma + \eta$ with probability $q$ ($Z_j \sim \text{Bernoulli}(q)$), comparing the **Sharing** model ($\gamma = 0$) against the **Causal** model ($\gamma \neq 0$) under $\gamma, \eta \sim \mathcal{N}(0, \sigma_g^2)$ and $q \sim \text{Beta}(q_\alpha, q_\beta)$ using exact 1D slope factorization, cancellation-free determinant evaluation, and two-stage adaptive grid subdivision.

4. **PSIS-LOO Model Comparison (`psis_loo`, `compare_elpd`)**:
   Evaluates pointwise expected log predictive density (ELPD) via Pareto-smoothed importance sampling with empirical-Bayes generalized Pareto tail fitting (`gpdfit`) and tests $\Delta\text{ELPD} = \text{ELPD}_{\text{sharing}} - \text{ELPD}_{\text{causal}}$ with a paired one-sided normal $z$-test ($p = \Phi(z)$).

## Installation

```bash
pip install -e ./cause
```

## Quickstart

```python
from cause import cause, est_cause_params, gwas_merge, ld_prune

# merged = gwas_merge(exp_df, out_df)
# params = est_cause_params(merged)
# top_snps = ld_prune(merged, pval_thresh=1e-3)
# res = cause(merged, param_ests=params, variants=top_snps["snp"])
# print(res.summary())
# print(res.to_dataframe())
```

## License

`GPL-3.0-or-later`
