# `bisque`: Cross-Platform Reference & Marker Bulk RNA-seq Deconvolution

Clean-room Python implementation of **Bisque** (Jew et al., *Nature Communications* 2020) for estimating cell-type proportions from bulk RNA-seq using either single-cell RNA-seq reference profiles with gene-specific affine domain alignment or marker-gene PCA scoring.

## Citation

> Jew, B., Alvarez, M., Rahmani, E., Miao, Z., Ko, A., Garske, K. M., Sul, J. H., Pietiläinen, K. H., Pajukanta, P., & Halperin, E. (2020). Accurate estimation of cell composition in bulk expression through robust integration of single-cell information. *Nature Communications*, 11(1), 1971. https://doi.org/10.1038/s41467-020-15816-6

## Mathematical & Algorithmic Overview

1. **CPM Normalization & Single-Cell Pseudobulk Construction (`generate_sc_reference`)**:
   Bulk and single-cell expression matrices ($N \times G$, supporting dense/sparse `AnnData`, `pandas.DataFrame`, and `numpy.ndarray`) are converted to counts-per-million ($\text{CPM}_{i,g} = 10^6 X_{i,g} / \sum_{g'} X_{i,g'}$). From single-cell annotations across $K$ cell types and $D$ donors, Bisque computes the cell-type signature matrix $Z \in \mathbb{R}^{G \times K}$ (mean CPM per cell type), observed donor proportions $P_{\text{sc}} \in \mathbb{R}^{D \times K}$, and synthetic pseudobulks $Y_{\text{sc}} = P_{\text{sc}} Z^T$.

2. **Gene-Specific Affine Domain Alignment (`reference_based_decomposition`)**:
   - **Non-overlapping cohorts (`use_overlap=False`)**: Aligns z-scored bulk expression per gene $g$ to the single-cell pseudobulk mean $\mu_{\text{sc}, g}$ and MSE-minimizing shrinkage standard deviation $\tilde{\sigma}_{\text{sc}, g} = \sqrt{\frac{1}{D+1}\sum_{j=1}^D (Y_{\text{sc}, j, g} - \mu_{\text{sc}, g})^2}$:
     $$\tilde{B}_{i, g} = \mu_{\text{sc}, g} + \tilde{\sigma}_{\text{sc}, g} \left(\frac{B_{i, g} - \mu_{\text{bulk}, g}}{s_{\text{bulk}, g}}\right)$$
   - **Overlapping subjects (`use_overlap=True`)**: Standardizes paired bulk ($X_{\text{train}}$) and pseudobulk ($Y_{\text{train}}$) across overlapping donors $\mathcal{O}$, estimates correlation slope $\hat{\beta}_g = \langle x^z_g, y^z_g \rangle / \|x^z_g\|_2^2$, and transforms bulk samples via:
     $$\tilde{B}_{i, g} = \mu_{Y, g} + s_{Y, g} \hat{\beta}_g \left(\frac{B_{i, g} - \mu_{X, g}}{s_{X, g}}\right)$$
   - Transformed bulk samples $\tilde{B}_{i, :}$ are decomposed against $Z$ via `scipy.optimize.nnls` and normalized to sum to 1.

3. **Marker-Based PCA Scoring (`marker_based_decomposition`)**:
   For each cell type $k$ with marker genes $M_k$, standardized (and optionally logFC-weighted) bulk expression $Z_k^{(w)} \in \mathbb{R}^{N_b \times |M_k|}$ is decomposed by thin SVD to extract PC1 scores $s_k = U[:, 0]\Sigma[0]$, deterministically sign-aligned against the weighted mean standardized marker profile $\bar{z}_k$, with optional `"minmax"` or `"proportional"` normalization.

## Installation

```bash
pip install -e ./bisque
```

## Quickstart

```python
from bisque import reference_based_decomposition, marker_based_decomposition

# Reference-based deconvolution (AnnData, DataFrame, or ndarray)
ref_res = reference_based_decomposition(
    bulk_adata,
    sc_adata,
    cell_types="cell_type",
    subject_names="subject_id",
    use_overlap=False,
)
print(ref_res.to_dataframe().head())
print(ref_res.summary())

# Marker-based deconvolution
marker_res = marker_based_decomposition(
    bulk_adata,
    markers={"B_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "PAX5"]},
    normalization="proportional",
)
print(marker_res.to_dataframe().head())
```

## Exact `BisqueRNA` v1.0.5 Parity (`settings="R_default"`)

`reference_based_decomposition`, `marker_based_decomposition`, and `generate_sc_reference` accept `settings: Literal["default", "R_default"] = "default"` (strictly validated; raises `ValueError` on unknown values):

| Component | `settings="default"` (Pythonic default) | `settings="R_default"` (`cozygene/bisque` v1.0.5 parity) |
| :--- | :--- | :--- |
| **Deconvolution solver** | Unconstrained `scipy.optimize.nnls` + post-hoc simplex normalization | Exact equality-and-inequality-constrained simplex QP ($\min_p \|Z p - b\|_2^2$ s.t. $\sum_k p_k = 1, p_k \ge 0$ via `SLSQP`, matching `limSolve::lsei`) |
| **Default `old_cpm`** | `False` (CPM across full transcriptome before intersecting genes) | `True` (intersects common/marker genes before CPM, matching `old.cpm = TRUE`) |
| **Default `use_overlap` & output samples** | `False`; returns all bulk samples and imputes zero-variance training genes via cohort shrinkage | `True` when overlapping subject IDs exist between `bulk` and `sc`; drops `anyNA` zero-variance training genes and returns deconvolved proportions on `remaining` non-overlapping bulk samples (when `remaining` is non-empty) |
| **Marker-based defaults** | `normalization="none"`, `weighted_pca=False` | `normalization="minmax"`, `weighted_pca=True` (when weights exist) |

```python
ref_r = reference_based_decomposition(bulk_adata, sc_adata, settings="R_default")
marker_r = marker_based_decomposition(bulk_adata, markers_df, settings="R_default")
```

## License

MIT (clean-room implementation from published equations in Jew et al., 2020).
