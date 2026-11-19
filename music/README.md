# `music`: Multi-Subject Single-Cell Deconvolution (MuSiC & MuSiC2)

Python implementation of **MuSiC** (Wang et al., *Nature Communications* 2019) and **MuSiC2** (Fan et al., *Briefings in Bioinformatics* 2022) for bulk RNA-seq cell-type deconvolution from multi-subject scRNA-seq references, natively supporting `AnnData` and `scipy.sparse` count matrices.

## Citations

- Wang, X., Park, J., Susztak, K., Zhang, N. R., & Li, M. Bulk tissue cell type deconvolution with multi-subject single-cell expression reference. *Nature Communications* **10**, 380 (2019). https://doi.org/10.1038/s41467-018-08023-x
- Fan, J., Lyu, Y., Zhang, Q., Wang, X., Li, M., & Xiao, R. MuSiC2: cell-type deconvolution for multi-condition bulk RNA-seq data. *Briefings in Bioinformatics* **23**, bbac430 (2022). https://doi.org/10.1093/bib/bbac430
- Upstream R package: https://github.com/xuranw/MuSiC

## Statistical & Algorithmic Overview

1. **Multi-Subject Reference Basis (`music_basis`)**:
   Computes per-donor relative expression $\theta_{i,k,g}$ and cell size $S_{i,k}$, aggregating cross-donor mean abundance `M_theta`, mean library size `M_S`, design matrix $D = M_\theta \odot M_S$, and cross-donor expression variance `Sigma` (plus optional cross-cell-type covariance `Sigma_ct`).
2. **Iteratively Reweighted NNLS (`music_iter`, `music_prop`)**:
   Weights each gene inversely by residual plus cross-subject variance,
   $$w_g = \frac{1}{\nu + r_g^2 + \sum_k (q_k S_k)^2 \Sigma_{g,k}},$$
   solving weighted NNLS (`scipy.optimize.nnls`) until simplex proportions $p = q / \sum_k q_k$ converge.
3. **Tree-Guided & Multi-Condition Deconvolution (`music_prop_cluster`, `music2_prop`)**:
   Supports two-level hierarchical group-to-subcluster deconvolution (`music_prop_cluster`) and iterative removal of condition-associated cell-type-specific DE genes between control and case cohorts (`music2_prop`).

## Quickstart

```python
from music import music2_prop, music_prop, music_prop_cluster

# Standard MuSiC (settings="default" or settings="R_default")
res = music_prop(
    bulk_adata=bulk_adata,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    settings="R_default",
)
print(res["Est_prop_weighted"].head())

# Multi-condition MuSiC2
res_m2 = music2_prop(
    bulk_control_adata=bulk_ctrl,
    bulk_case_adata=bulk_case,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    settings="R_default",
)
print(res_m2["Est_prop"].head())
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Retains single-donor cell types whose cross-donor variance `Sigma` is `NaN` (substituting `0.0` in gene-weight denominators).
  - Precomputes `music_basis` once outside the `music2_prop` loop and subsets it to `l_genes` per iteration.
  - Returns `{'Est_prop_weighted', 'Est_prop_allgene', 'R_squared'}` from `music_prop`.
- **`settings="R_default"` (Exact `xuranw/MuSiC` `analysis.R` & `music2.R` parity)**:
  - Drops cell types with `Sigma = NaN` (single-donor cell types) in `music_prop`, `music_prop_cluster`, and `music2_prop` (`analysis.R:122`).
  - Recomputes `music_basis` directly on `sc_adata[:, l_genes]` inside every `music2_prop` iteration (`music2.R:121`), renormalizing donor-level $\theta$ and $\Sigma$ on the retained non-DE gene set.
  - Populates `'Var_prop'` (delta-method NNLS sandwich variance) and `'Weight_gene'` (`genes x samples` final IRLS weights) in the `music_prop` return dictionary.

## Key Outputs

- `music_prop`: dict with `'Est_prop_weighted'`, `'Est_prop_allgene'`, `'R_squared'`, and (in `settings="R_default"`) `'Var_prop'` and `'Weight_gene'`.
- `music_prop_cluster`: dict with `'Est_prop_weighted_cluster'`.
- `music2_prop`: dict with `'Est_prop'`, `'convergence'`, `'DE_genes'`, and `'n_iter'` / `'id_not_converge'`.

## Running Tests

```bash
PYTHONPATH=music /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest music/tests -q
```
