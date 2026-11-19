# `mrhorse`: Python Implementation of `MR-Horse` & `MVMR-Horse`

Standalone Python implementation of **MR-Horse** and **MVMR-Horse** (Grant & Burgess, *AJHG* 2024) for univariable and multivariable Bayesian Mendelian Randomization with a Horseshoe prior on correlated and uncorrelated pleiotropic effects. Uses conjugate/griddy-Gibbs MCMC with Gelman–Rubin $\hat{R}$ diagnostics and zero external JAGS requirement.

## Installation

```bash
pip install -e ./mrhorse
```

## Quickstart

```python
from mrhorse import mr_horse, mvmr_horse

res = mr_horse(beta_out, se_out, beta_exp, se_exp, n_chains=3, n_iter=2000, n_burnin=2000, seed=42)
print(res["estimate"][0])
```

## Reference

- Grant, A. J. & Burgess, S. A Bayesian approach to Mendelian randomization using summary statistics in the univariable and multivariable settings with correlated pleiotropy. *American Journal of Human Genetics* 111, 165–180 (2024).
