# `fishhook`: Gamma–Poisson Somatic Mutation Enrichment Modeling

Clean-room Python implementation of **fishHook** (Imielinski, Guo & Meyerson, *Cell* 2017) for identifying driver genes, enhancers, and non-coding regulatory elements enriched for somatic mutations after accounting for callable genomic territory and regional epigenomic/replication-timing covariates.

## Citation

> Imielinski, M., Guo, W., & Meyerson, M. (2017). Insertions and Deletions Target Lineage-Defining Genes in Human Cancers. *Cell*, 168(3), 460–472.e14. https://doi.org/10.1016/j.cell.2016.12.025

## Mathematical Overview

1. **Callable Genomic Territory & Per-Sample Kataegis Capping**:
   For each target genomic interval $R_i = [\text{start}_i, \text{end}_i)$ and optional callable territory mask $\mathcal{E}$ (`eligible_intervals`), exact covered exposure $L_i = |R_i \cap \mathcal{E}|$ is computed in $O((R + E)\log E)$ via sweep-line interval union and vectorized binary search. Optional `max_muts_per_sample_per_tile` caps mutations per `(region, sample_id)` to prevent localized hypermutation (e.g., APOBEC kataegis or AID) in single tumors from creating false recurrence peaks.

2. **Gamma–Poisson (Negative Binomial Type-2) Regression with Exposure Offset**:
   Models mutation counts $Y_i$ on active regions ($L_i > 0$) via:
   $$\mathbb{E}[Y_i] = \mu_i = L_i \exp(\beta_0 + X_i \beta), \qquad \text{Var}(Y_i) = \mu_i + \alpha \mu_i^2 \quad \left(\theta = \frac{1}{\alpha}\right)$$
   fitted by alternating damped Fisher-scoring IRLS for $(\beta_0, \beta)$ and exact profile maximum likelihood for $\theta$, with automatic Poisson fallback as $\alpha \to 0^+$.

3. **Exact Right-Tail Survival Enrichment & Diagnostics**:
   Evaluates exact right-tail probabilities $\Pr(Y \ge y_i \mid \hat{\mu}_i, \hat{\theta})$ via `scipy.stats.nbinom.sf(y_i - 1, n=theta, p=theta / (theta + mu_i))`, fold-enrichment $(y_i / L_i) / (\hat{\mu}_i / L_i)$, genomic inflation diagnostics ($\lambda_{\text{GC}}$ and QQ $-\log_{10} p$ slope), and Benjamini–Hochberg FDR $q$-values.

## Quickstart

```python
import pandas as pd
from fishhook import FishHookModel, fit_fishhook

res = fit_fishhook(
    regions=regions_df,
    mutations=mutations_df,
    eligible_intervals=callable_exome_bed,
    covariate_cols=["late_replication", "gc_content", "chromatin_open"],
    max_muts_per_sample_per_tile=1,
)
print(res.coefficients_dataframe())
print(res.significant(alpha=0.05))
```

## License

MIT
