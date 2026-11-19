# `variancepartition`: Mixed-Effects Variance Partitioning & `dream` Differential Expression

Standalone, pure NumPy/SciPy/Pandas Python implementation of **variancePartition** (Hoffman & Schadt, *BMC Bioinformatics* 2016) and **dream** (Hoffman & Roussos, *Bioinformatics* 2021) for linear mixed-effects variance decomposition, precision-weighted repeated-measures differential expression, Satterthwaite degrees of freedom, and Empirical Bayes variance moderation.

## Citations

> Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of variation in complex gene expression studies. *BMC Bioinformatics* 17:483. https://doi.org/10.1186/s12859-016-1323-z
>
> Hoffman GE, Roussos P (2021). dream: powerful differential expression analysis for repeated measures designs. *Bioinformatics* 37(2):192–201. https://doi.org/10.1093/bioinformatics/btaa687

## `settings` Modes (`"default"` vs `"R_default"`)

`fit_extract_var_part_model`, `extract_var_part`, `fit_lmm`, and `dream` accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`GabrielHoffman/variancePartition` Parity) |
| :--- | :--- | :--- |
| **Variance Partitioning Fit (`fit_extract_var_part_model`)** | Defaults to REML (`reml=True`). | Defaults to Maximum Likelihood (`reml=False`), matching `fitExtractVarPartModel.R`. |
| **Fixed-Effect Variance (`extract_var_part`)** | Sample variance (`ddof=1`, optional precision weighting via `weighted_fixed_var=True`). | `calcVarPart.R`: $(N-1)/N$ population variance (`ddof=0`) per term rescaled by $\operatorname{Var}_{\text{pop}}(X \hat{\beta}) / \sum_k v_k$ so correlated fixed predictors sum to total fixed-effect variance. |
| **Residual Variance (`extract_var_part`)** | Weight-scale-invariant marginal variance $\hat{\sigma}_e^2 \cdot \frac{1}{N}\sum_{i=1}^N w_i^{-1}$. | Unweighted $\hat{\sigma}_e^2$ (`attr(VarCorr(fit), "sc")^2`). |
| **`dream` Differential Expression** | REML + Expected Fisher Information Satterthwaite df + `squeeze_var` Empirical Bayes moderation. | REML + Satterthwaite df + `squeeze_var` Empirical Bayes moderation. |

## Quickstart

```python
from variancepartition import fit_extract_var_part_model, dream, top_table

# 1. Variance partitioning (Pythonic REML default vs R_default ML + calcVarPart.R)
vp = fit_extract_var_part_model(
    expr,
    "~ condition + age + (1 | subject) + (1 | batch)",
    data=meta,
    settings="default",
)
vp_r = fit_extract_var_part_model(
    expr,
    "~ condition + age + (1 | subject) + (1 | batch)",
    data=meta,
    settings="R_default",
)
print(vp_r.to_dataframe(sort_by="subject").head())

# 2. Repeated-measures differential expression with dream + eBayes
res = dream(
    expr,
    "~ condition + time + (1 | subject)",
    data=meta,
    contrasts="condition_treated",
    settings="R_default",
)
print(top_table(res).head())
```

## License

MIT
