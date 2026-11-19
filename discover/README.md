# `discover`: Discrete Independence Statistic Controlling for Observations with Varying Event Rates

Python implementation of **DISCOVER** (Canisius, Martens & Wessels, *Genome Biology* 2016) for detecting somatic **mutual exclusivity** and **co-occurrence** in heterogeneous cancer cohorts while accounting for tumor-specific background mutation burdens and targeted-panel coverage masks.

## Citations

- Canisius, S., Martens, J. W. M., & Wessels, L. F. A. A novel independence test for somatic alterations in cancer shows that biology drives mutual exclusivity but chance explains most co-occurrence. *Genome Biology* **17**, 261 (2016). https://doi.org/10.1186/s13059-016-1114-x
- Hong, Y. On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis* **59**, 41–51 (2013).
- Upstream NKI-CCB DISCOVER repository: https://github.com/NKI-CCB/DISCOVER

## Statistical & Algorithmic Overview

1. **Maximum-Entropy Background Model (`fit_discover_background`)**:
   Fits $P_{ij} = \Pr(A_{ij} = 1) = \sigma(\mu_i + \lambda_j)$ across observed entries of $A \in \{0, 1, \text{NaN}\}^{M \times N}$ via L-BFGS-B + coordinate-block Newton polishing so expected row and column sums match observed gene mutation counts and tumor-specific mutation burdens (optionally stratified via `strata`).
2. **Exact Poisson–Binomial Tail Evaluation (`poisson_binomial_pmf`, `poisson_binomial_pval`)**:
   Computes exact PMFs via Hong (2013) log-polar DFT (`irfft`) and exact non-negative DP refinement for extreme tails down to $\sim 10^{-300}$.
3. **Pairwise & Groupwise Pathway Tests (`pairwise_discover_test`, `groupwise_discover_test`)**:
   Tests pairwise co-mutation counts and groupwise pathway `"impurity"`, `"coverage"`, and `"exclusivity"` statistics with discrete-support-aware FDR control.

## Quickstart

```python
from discover import (
    fit_discover_background,
    groupwise_discover_test,
    pairwise_discover_test,
)

bg = fit_discover_background(alteration_df, strata=tumor_subtypes)

# Pairwise mutual exclusivity (settings="default" or settings="R_default")
mutex_res = pairwise_discover_test(
    bg.subset(genes=["BRAF", "NRAS", "KRAS", "TP53"]),
    alternative="exclusivity",
    settings="R_default",
)
print(mutex_res.to_dataframe())

# Groupwise pathway mutual exclusivity
pathway_res = groupwise_discover_test(
    bg,
    gene_sets={"MAPK": ["BRAF", "NRAS", "KRAS"]},
    method="impurity",
    direction="exclusivity",
    settings="R_default",
)
print(pathway_res.to_dataframe())
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Uses Lancaster (1961) mid-$p$ correction (`mid_p=True` by default: $\Pr(X < x) + 0.5 \Pr(X = x)$ for lower tail, $\Pr(X > x) + 0.5 \Pr(X = x)$ for upper tail).
  - Applies Carlson/Canisius minimum-attainable-$p$ adaptive discrete FDR (`discover_fdr`: $m(\tau) = \#\{r : p_{r,\min} \le \tau\}$).
- **`settings="R_default"` (Exact `NKI-CCB/DISCOVER` `fdr.f90` & `grouptest.py` parity)**:
  - Forces `mid_p=False` (inclusive discrete tails $\Pr(X \le x)$ and $\Pr(X \ge x)$).
  - Computes exact **`DBH`** discrete Benjamini–Hochberg FDR (`dbh_fdr`, matching `fdr.f90`):
    $$q_{(k)} = \min_{j \ge k} \min\!\left(1, \; \frac{\hat{\pi}_0 \sum_{r=1}^M F_r(p_{(j)})}{j}\right), \qquad \hat{\pi}_0 = \min\!\left(1, \; \frac{\sum_r p_r}{\sum_r \mathbb{E}_0[P_r]}\right)$$
    where $F_r(\tau)$ is the largest attainable p-value $\le \tau$ under test $r$'s discrete Poisson–Binomial support.

## Key Outputs

- `DiscoverMatrix`: fitted background probability matrix (`events`, `bg_probs`, `mask`, `.subset()`, `.to_dataframe()`).
- `PairwiseDiscoverResult` & `GroupwiseDiscoverResult`: frozen dataclasses with `p_value`, `p_min`, `q_value`, `.to_dataframe()`, and `.significant()`.

## Running Tests

```bash
PYTHONPATH=discover /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest discover/tests -q
```
