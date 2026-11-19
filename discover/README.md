# `discover`: Discrete Independence Statistic Controlling for Observations with Varying Event Rates

Clean-room Python implementation of **DISCOVER** (Canisius, Martens & Wessels, *Genome Biology* 2016) for detecting somatic **mutual exclusivity** and **co-occurrence** in heterogeneous cancer cohorts while accounting for tumor-specific background mutation burdens and panel coverage masks.

## Citation

> Canisius, S., Martens, J. W. M., & Wessels, L. F. A. (2016). A novel independence test for somatic alterations in cancer shows that biology drives mutual exclusivity but chance explains most co-occurrence. *Genome Biology*, 17, 261. https://doi.org/10.1186/s13059-016-1114-x
>
> Hong, Y. (2013). On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis*, 59, 41–51.
>
> Carlson, J. M., Heckerman, D., Shani, G. (2009). Estimating false discovery rates for contingency tables. *BMC Bioinformatics*, 10, S12.

## Mathematical Overview

1. **Maximum-Entropy Background Model (`fit_discover_background`)**:
   Given an $M \times N$ somatic alteration matrix $A \in \{0, 1, \text{NaN}\}^{M \times N}$ ($M$ genes $\times$ $N$ tumors, where `NaN` marks unprofiled targeted-panel entries), DISCOVER fits:
   $$P_{ij} = \Pr(A_{ij} = 1) = \frac{1}{1 + \exp(-(\mu_i + \lambda_j))}$$
   via L-BFGS-B + coordinate-block Newton polishing so expected row sums match observed gene alteration counts ($\sum_{j \in \mathcal{O}_i} P_{ij} = \sum_{j \in \mathcal{O}_i} A_{ij}$) and expected column sums match tumor-specific background mutation burdens ($\sum_{i \in \mathcal{O}_j} P_{ij} = \sum_{i \in \mathcal{O}_j} A_{ij}$), optionally stratified by subtype (`strata`). This prevents hypermutated (e.g., MSI / POLE) tumors from driving spurious co-occurrence calls as in Fisher's exact test.

2. **Exact Poisson–Binomial Distribution (`poisson_binomial_pmf`, `poisson_binomial_pval`)**:
   Evaluates the exact distribution of $S = \sum_{k=1}^N \text{Bernoulli}(p_k)$ via Hong (2013) log-polar Discrete Fourier Transform (`np.fft.irfft`) and exact non-negative dynamic programming, with Lancaster (1961) mid-$p$ correction (`mid_p=True` default).

3. **Pairwise & Groupwise Pathway Tests (`pairwise_discover_test`, `groupwise_discover_test`)**:
   - Pairwise lower-tail (`exclusivity`) and upper-tail (`co-occurrence`) tests with discrete-support-aware Benjamini–Hochberg FDR (`discover_fdr`).
   - Groupwise $G$-gene pathway statistics: `"impurity"` ($\Pr(\sum_{i \in S} X_{i,k} \ge 2)$), `"coverage"` ($1 - \prod_{i \in S}(1 - P_{i,k})$), and `"exclusivity"` ($\Pr(\sum_{i \in S} X_{i,k} = 1)$).

## Quickstart

```python
import numpy as np
import pandas as pd
from discover import (
    fit_discover_background,
    pairwise_discover_test,
    groupwise_discover_test,
)

# Fit genome/panel-wide tumor-specific background probabilities
bg = fit_discover_background(alteration_df, strata=tumor_subtypes)

# Pairwise mutual exclusivity with adaptive discrete FDR
mutex_res = pairwise_discover_test(bg.subset(genes=["BRAF", "NRAS", "KRAS", "TP53"]))
print(mutex_res.to_dataframe())

# Groupwise pathway mutual exclusivity ("impurity", "coverage", or "exclusivity")
pathway_res = groupwise_discover_test(
    bg,
    gene_sets={"MAPK": ["BRAF", "NRAS", "KRAS"]},
    method="impurity",
    direction="exclusivity",
)
print(pathway_res.to_dataframe())
```

## License

MIT
