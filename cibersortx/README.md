# `cibersortx`: Cross-Platform & High-Resolution Bulk Deconvolution

Clean-room Python implementation of **CIBERSORT** and **CIBERSORTx** (Newman et al., *Nature Methods* 2015; *Nature Biotechnology* 2019) built purely from the published mathematical formulations for signature matrix construction, linear $\nu$-SVR bulk deconvolution, ComBat cross-platform batch alignment (`B-mode` and `S-mode`), Group-Mode cell-type GEP estimation, and High-Resolution sample-specific expression purification.

## Citations

> Newman, A. M., Liu, C. L., Green, M. R., Gentles, A. J., Feng, W., Xu, Y., Hoang, C. D., Diehn, M., & Alizadeh, A. A. (2015). Robust enumeration of cell subsets from tissue expression profiles. *Nature Methods*, 12(5), 453–457. https://doi.org/10.1038/nmeth.3337
>
> Newman, A. M., Steen, C. B., Liu, C. L., Gentles, A. J., Chaudhuri, A. A., Scherer, F., Khodadoust, M. S., Esfahani, M. S., Luca, B. A., Steiner, D., Diehn, M., & Alizadeh, A. A. (2019). Determining cell type abundance and expression from bulk tissues with digital cytometry. *Nature Biotechnology*, 37(7), 773–782. https://doi.org/10.1038/s41587-019-0114-2

## Mathematical & Algorithmic Overview

1. **Signature Matrix Construction (`build_signature_matrix`)**:
   Normalizes scRNA-seq/replicate profiles to CP10K, performs one-vs-rest differential expression screening (Mann–Whitney $U$ or Welch $t$-test with Benjamini–Hochberg FDR and $\log_2\text{FC}$ filtering), and sweeps top $g_{\text{top}}$ markers per cell type to minimize the $2$-norm condition number $\kappa_2(B) = \sigma_{\max}(B)/\sigma_{\min}(B)$.

2. **Linear $\nu$-SVR Deconvolution (`deconvolve` / `cibersort`)**:
   Applies global Z-score standardization across the intersected signature matrix $\tilde{B} = (B - \mu_B)/\sigma_B$ and per-sample Z-score standardization of bulk profiles $m_j$. Fits linear $\nu$-SVR across $\nu \in \{0.25, 0.5, 0.75\}$ ($C = 1.0$), clips negative primal weights $w_k^+ = \max(\hat{w}_k, 0)$, normalizes to the simplex $\hat{f}_j = w^+ / \sum_k w_k^+$, selects $\nu_j^*$ minimizing standardized RMSE, and computes Phipson–Smyth permutation $p$-values $p_j = (1 + \#\{r^{(b)} \ge r_j\}) / (N_{\text{perm}} + 1)$.

3. **Cross-Platform Batch Alignment (`B-mode` & `S-mode`, `combat_align`)**:
   Uses two-batch Empirical Bayes location-scale adjustment in $\log_2(x + 1)$ space:
   - **`B-mode`**: Maps observed bulk $M$ onto signature-space synthetic mixtures $\tilde{M} = B(\hat{F}^{(0)})^T$ before re-deconvolution.
   - **`S-mode`**: Generates Dirichlet-weighted pseudobulks $\tilde{M}_{\text{sc}}$ from single-cell reference $S$, estimates platform parameters between $\tilde{M}_{\text{sc}}$ and bulk $M$, and transforms the signature $B_S$ into bulk space.

4. **Group-Mode GEP Estimation (`impute_gep_group`)**:
   Solves QR-accelerated non-negative least squares $\min_{x \ge 0} \|F x - m_g\|_2^2$ per gene, estimates bootstrap standard errors $\widehat{\text{SE}}_{g,k}$, coefficient of variation $\text{CV}_{g,k}$, one-sided normal $p$-values, and `NaN`-aware BH-FDR $q$-values with adaptive noise filtering.

5. **High-Resolution Sample-Specific Purification (`impute_gep_highres`)**:
   Orders bulk samples per $(g, k)$ by residualized expression $(M_{g,n} - \sum_{j \neq k} F_{n,j} \hat{x}_{g,j}^{\text{grp}})/\max(F_{n,k}, 0.05)$, solves sliding-window non-negative ridge/NNLS across window width $w$, aggregates overlapping windows by median, and tests sample-specific variability against a Group-mode residual permutation null.

## Installation

```bash
pip install -e ./cibersortx
```

## Quickstart

```python
from cibersortx import (
    build_signature_matrix,
    deconvolve,
    impute_gep_group,
    impute_gep_highres,
)

# 1. Build signature matrix from AnnData (obs x var)
sig = build_signature_matrix(sc_adata, cell_type_labels="cell_type", layer="counts")

# 2. Deconvolve bulk expression (with optional B-mode or S-mode batch alignment)
res = deconvolve(bulk_df, sig, batch_mode="B", n_permutations=100, seed=0)
print(res.to_dataframe().head())

# 3. Group-mode cell-type expression & High-Resolution sample-specific purification
grp = impute_gep_group(bulk_df, res, n_bootstrap=50, seed=0)
hires = impute_gep_highres(bulk_df, res, n_permutations=199, seed=0)
print(hires.gep_dataframe(sig.cell_types[0]).head())
```

## Matrix Orientation Conventions

- **`AnnData` inputs** (`mixture`, `sc_ref`, `reference`): standard `obs x var` (`samples/cells x genes`).
- **`pandas.DataFrame` / `numpy.ndarray` inputs**: `genes x samples` (or `genes x cells` / `genes x cell_types`), with `samples_as_rows=True` / `cells_as_rows=True` supported when passing `samples x genes` DataFrames.

## License

MIT (clean-room implementation from published mathematical formulations of Newman et al., 2015 and 2019).
