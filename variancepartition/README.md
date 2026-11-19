# `variancepartition`: Mixed-Effects Variance Partitioning & `dream` Differential Expression

Standalone, pure NumPy/SciPy/Pandas Python implementation of **variancePartition** (Hoffman & Schadt, *BMC Bioinformatics* 2016) and **dream** (Hoffman & Roussos, *Bioinformatics* 2021) for linear mixed-effects variance decomposition, precision-weighted repeated-measures differential expression, Satterthwaite degrees of freedom, and Empirical Bayes variance moderation.

## Citations

> Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of variation in complex gene expression studies. *BMC Bioinformatics* 17:483. https://doi.org/10.1186/s12859-016-1323-z

> Hoffman GE, Roussos P (2021). dream: powerful differential expression analysis for repeated measures designs. *Bioinformatics* 37(2):192–201. https://doi.org/10.1093/bioinformatics/btaa687

## Mathematical & Algorithmic Overview

1. **Precision-Weighted Linear Mixed Models (`fit_lmm`, `fit_var_part_model`)**:
   For each gene $g \in \{1, \dots, G\}$ across $N$ samples (default `samples x genes` orientation matching `AnnData` `obs x var`, with `genes_as_rows=True` supported for `genes x samples` arrays):
   $$y_g = X \beta_g + \sum_{r=1}^R Z_r u_{g,r} + \epsilon_g, \qquad u_{g,r} \sim \mathcal{N}(0, \sigma_{g,r}^2 I_{m_r}), \qquad \epsilon_g \sim \mathcal{N}(0, \sigma_{g,e}^2 W_g^{-1})$$
   - **$R = 0$**: Closed-form batched WLS/OLS.
   - **$R = 1$ (shared/uniform weights)**: Precomputed orthogonal-projection SVD spectral trick (FaST-LMM style) reducing each gene's REML optimization to 1D.
   - **General $R \ge 1$ / gene-specific weights**: Symmetric-scaled $m \times m$ Henderson/Woodbury profiled REML in variance ratios $\gamma_r = \sigma_{g,r}^2 / \sigma_{g,e}^2 \ge 0$ with exact analytical gradients and OLS-residualized stability.

2. **ICC Variance Partitioning (`extract_var_part`, `fit_extract_var_part_model`)**:
   Decomposes total variance per gene into non-intercept fixed-effect term variances $\operatorname{Var}(X_k \hat{\beta}_{g,k})$, random-effect variances $\hat{\sigma}_{g,r}^2 \frac{1}{N}\operatorname{tr}(Z_r^T Z_r)$, and average marginal residual variance $\hat{\sigma}_{g,e}^2 \frac{1}{N}\sum_{i=1}^N w_{g,i}^{-1}$, summing strictly to $1.0$ and invariant to global scaling of precision weights.

3. **Satterthwaite Degrees of Freedom & `dream` Empirical Bayes (`dream`, `e_bayes`, `top_table`)**:
   Computes the exact $O(m^3)$ REML Expected Fisher Information $I_E(\hat{\theta}_g)$ directly from $S = \tilde{Z}_g^T \tilde{P}_g \tilde{Z}_g$ in variance-component parameterization $(\sigma_1^2, \dots, \sigma_R^2, \sigma_e^2)$ (smooth up to boundary $\hat{\sigma}_r^2 = 0$; $N - \operatorname{rank}(X)$ when $R = 0$), derives contrast-specific Satterthwaite degrees of freedom $\nu_{g,\ell} \in [\nu_{\min}, N_g - r_X]$, applies `squeeze_var` Empirical Bayes shrinkage over varying $\nu_{g,\ell}$ (with optional `trend=True` and `robust=True` Phipson et al. 2016 outlier downweighting), and computes `NaN`-preserving Benjamini–Hochberg FDR and standardized normal $z$-scores (`z.std`).

## Installation

```bash
pip install -e ./variancepartition
```

## Quickstart

```python
import numpy as np
import pandas as pd
from variancepartition import fit_extract_var_part_model, dream, top_table

# Expression matrix: (N_samples, G_genes)
expr = pd.DataFrame(...)
meta = pd.DataFrame(...)  # columns: condition, time, subject, batch

# 1. Variance partitioning across fixed and random covariates
vp = fit_extract_var_part_model(
    expr,
    "~ condition + (1 | subject) + (1 | batch)",
    data=meta,
)
print(vp.to_dataframe(sort_by="subject").head())
print(vp.summary())

# 2. Repeated-measures differential expression with dream + eBayes
res = dream(
    expr,
    "~ condition + time + (1 | subject)",
    data=meta,
    contrasts="condition_treated",
)
print(top_table(res).head())
```

## License

MIT
