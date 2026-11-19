# `nebula`: Multi-Subject Single-Cell Negative Binomial Mixed Models

Standalone Python implementation of **NEBULA** (He et al., *Communications Biology* 2021) for fast differential expression and overdispersion analysis across large-scale multi-subject single-cell RNA-seq data.

## Citation

> He, L., Davila-Velderrain, J., Sumida, T. S., Hafler, D. A., Kellis, M., & Kulminski, A. M. (2021). NEBULA is a fast negative binomial mixed model for differential or co-expression analysis of large-scale multi-subject single-cell data. *Communications Biology*, 4, 625. https://doi.org/10.1038/s42003-021-02146-6

## Mathematical & Algorithmic Overview

1. **Hierarchical Model (`NBGMM`, `PMM`, `NBLMM`)**: For gene $g$, subject $i \in \{1,\dots,m\}$, and cell $j \in \{1,\dots,n_i\}$:
   $$y_{ij} \mid \omega_i, \nu_{ij} \sim \text{Poisson}\!\left(t_{ij} \exp(\mathbf{x}_{ij}^\top\boldsymbol{\beta} + \omega_i)\nu_{ij}\right)$$
   with cell-level Gamma overdispersion $\nu_{ij} \sim \text{Gamma}(\phi^{-1}, \phi^{-1})$ ($\text{Var}(\nu_{ij}) = \phi$) and subject random effect $e^{\omega_i} \sim \text{Gamma}(\sigma^{-2}, \sigma^{-2})$ (`NBGMM` / `PMM`) or $\omega_i \sim \mathcal{N}(0, \sigma^2)$ (`NBLMM`).
2. **Subject Grouping (`group_cell`)**: Stably orders cells into contiguous subject blocks using `kind="mergesort"` without densifying sparse count matrices.
3. **`NEBULA-LN` (Analytical Marginal Moments)**: Swaps the subject-cell integration order and expands around $\nu_{ij} = 1$ to obtain a closed-form $O(N)$ marginal log-likelihood and exact analytical gradients optimized via `L-BFGS-B`.
4. **`NEBULA-HL` (H-Likelihood Gamma-Saddlepoint / Laplace)**: Solves the strictly concave per-subject mode equation $h_i'(\omega_i) = 0$ via vectorized damped Newton-Raphson and optimizes the Gamma-saddlepoint / Laplace marginal log-likelihood with exact Implicit Function Theorem gradients. Automatically selected by `method="auto"` whenever $\kappa_{\text{cps}} = N/m < 20$ or $\kappa_{\text{cps}} / \hat{\phi}_{\text{LN}} < \text{cutoff\_cell}$.
5. **Inference**: GLMM Schur-complement Fisher information $I_{\beta\beta} = H_{\beta\beta} - H_{\beta\omega}H_{\omega\omega}^{-1}H_{\omega\beta}$, Wald statistics, and `NaN`-aware Benjamini–Hochberg FDR.

## Installation

```bash
pip install -e ./nebula
```

## Quickstart

```python
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from nebula import nebula

# AnnData (`obs x var` = cells x genes) with sparse counts
adata = ad.AnnData(
    X=sparse.csr_matrix(counts_cells_by_genes),
    obs=pd.DataFrame({"subject": subject_ids, "condition": cond, "cell_cov": cov}),
)
res = nebula(adata, id="subject", pred=["condition", "cell_cov"], method="auto")
print(res.summary())
print(res.to_dataframe().head())
```

## License

MIT (clean-room Python implementation matching the MIT-licensed `lhe17/nebula` algorithm).
