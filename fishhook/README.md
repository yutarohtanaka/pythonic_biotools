# `fishhook`: Gamma–Poisson Somatic Mutation Enrichment Modeling

Python implementation of **fishHook** (Imielinski, Guo & Meyerson, *Cell* 2017) for identifying driver genes, enhancers, and non-coding regulatory elements enriched (or depleted) for somatic mutations after accounting for callable genomic territory and regional epigenomic / replication-timing covariates.

## Citation

- Imielinski, M., Guo, W., & Meyerson, M. Insertions and Deletions Target Lineage-Defining Genes in Human Cancers. *Cell* **168**, 460–472.e14 (2017). https://doi.org/10.1016/j.cell.2016.12.025
- Upstream Marcin Imielinski Lab R package: https://github.com/mskilab/fishHook

## Statistical & Algorithmic Overview

1. **Callable Genomic Territory & Per-Sample Kataegis Capping (`compute_territory_and_counts`)**:
   Computes exact covered base pairs $L_i = |R_i \cap \mathcal{E}|$ in $O((R + E)\log E)$ via sweep-line interval union and vectorized binary search, with optional per-`(region, sample_id)` mutation capping (`max_muts_per_sample_per_tile`).
2. **Gamma–Poisson (NB2) Regression with Exposure Offset (`fit_fishhook`, `FishHookModel`)**:
   Fits $\mathbb{E}[Y_i] = \mu_i = L_i \exp(\beta_0 + X_i \beta)$, $\text{Var}(Y_i) = \mu_i + \alpha \mu_i^2$ ($\theta = 1/\alpha$) via damped Fisher-scoring IRLS and profile MLE for $\theta$, with automatic Poisson fallback as $\alpha \to 0^+$.
3. **Survival Enrichment, Depletion & QQ Diagnostics**:
   Computes right-tail enrichment p-values, BH-FDR `q_value`, fold-enrichment, and QQ inflation metrics (`lambda_gc`, `qq_slope`).

## Quickstart

```python
from fishhook import FishHookModel, fit_fishhook

res = fit_fishhook(
    regions=regions_df,
    mutations=mutations_df,
    eligible_intervals=callable_exome_bed,
    covariate_cols=["late_replication", "gc_content"],
    settings="R_default",
)
print(res.coefficients_dataframe())
print(res.to_dataframe().head())
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Leaves `max_muts_per_sample_per_tile=None` unless explicitly supplied.
  - Computes exact deterministic discrete right-tail survival p-values $\Pr(Y \ge y_i)$ (so $y_i = 0$ has $p_i = 1.0$).
  - Fits interior-band QQ slope (excluding $p = 1.0$ and top 1% driver tail) alongside Devlin & Roeder median $\chi^2_1$ `lambda_gc`.
  - `.to_dataframe()` returns the 12 core enrichment columns.
- **`settings="R_default"` (Exact `mskilab/fishHook` `fishhook.R` parity)**:
  - Defaults `max_muts_per_sample_per_tile = 1` (`idcap = 1`) automatically whenever `mutations` contains a `sample_id` column.
  - Computes uniform randomized p-values on $[\Pr(Y > y_i), \Pr(Y \ge y_i)]$ seeded deterministically via `seed: int = 42` so null discrete p-values are strictly $\text{Uniform}(0, 1)$.
  - Fits all-point zero-intercept OLS `lm(-log10(p_obs) ~ -log10(p_exp) - 1)` for QQ `lambda`.
  - Computes left-tail randomized depletion p-values and BH-FDR (`p_neg`, `fdr_neg`) and includes them in `.to_dataframe()`.

## Key Outputs

- `FishHookResult`: frozen dataclass with `.to_dataframe()`, `.coefficients_dataframe()`, `.significant()`, `beta`, `theta`, `alpha`, `lambda_gc`, `qq_slope`, and (in `settings="R_default"`) `p_neg` / `fdr_neg`.

## Running Tests

```bash
PYTHONPATH=fishhook /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest fishhook/tests -q
```
