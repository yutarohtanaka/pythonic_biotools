# Pythonic MuSiC: Bulk RNA-seq Deconvolution

This document outlines the Python implementation of the **MuSiC** (Multi-subject Single Cell deconvolution) algorithm, originally developed in R. The Python version is designed for modern single-cell ecosystems, making it natively compatible with `scanpy` and `AnnData` objects.

## Overview

MuSiC estimates cell-type proportions in bulk RNA-seq data by leveraging single-cell RNA-seq references. Unlike traditional methods that rely solely on cell-type specific mean expression, MuSiC incorporates **cross-subject variance** and **cell-type-specific library sizes** into a weighted non-negative least squares (NNLS) model. This improves accuracy, especially for closely related cell types or datasets with high biological variance.

The Python implementation provides the core pipeline and its extended version:
- `music.py`:
  - `music_basis`: Computes cross-subject mean relative abundance ($\theta$), variance ($\Sigma$), cross-cell-type covariance (`Sigma_ct`), and average library size ($S$) from single-cell AnnData.
  - `music_prop`: The main pipeline wrapper that processes bulk and single-cell AnnData objects, subsets common genes, scales data, and iteratively solves the weighted NNLS.
- `music2.py`:
  - `music2_prop`: A multi-iteration extension designed for cases where bulk data and scRNA-seq references come from different clinical conditions. It iteratively identifies differentially expressed (DE) genes using bulk case/control status and removes them from the basis to improve accuracy.

## How to Use

The pipelines expect `AnnData` objects: one for bulk RNA-seq (or two for `music2_prop`) and one for single-cell reference.

### 1. Standard MuSiC (`music_prop`)
```python
import scanpy as sc
from music import music_prop

# Load single-cell reference and bulk data
sc_adata = sc.read_h5ad("sc_reference.h5ad")  # Obs: cells, Vars: genes
bulk_adata = sc.read_h5ad("bulk_data.h5ad")   # Obs: samples, Vars: genes

# Run MuSiC deconvolution
results = music_prop(
    bulk_adata=bulk_adata,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    ct_cov=True,      # Enable cross-cell-type covariance modeling if desired
    iter_max=1000,
    eps=0.01
)

# Results is a dictionary with Pandas DataFrames
est_proportions = results['Est_prop_weighted']
print(est_proportions.head())
```

### 2. MuSiC2 (`music2_prop`)
```python
import scanpy as sc
from music2 import music2_prop

bulk_control = sc.read_h5ad("bulk_control.h5ad")
bulk_case = sc.read_h5ad("bulk_case.h5ad")
sc_adata = sc.read_h5ad("sc_reference.h5ad")

results = music2_prop(
    bulk_control_adata=bulk_control,
    bulk_case_adata=bulk_case,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    ct_cov=False,
    maxiter=200
)

# Includes final proportions, convergence status, and the list of DE genes removed.
final_proportions = results['Est_prop']
print(final_proportions.head())
```

## Methodological Adaptations & Changes

To seamlessly integrate MuSiC into the Python/AnnData ecosystem, several key adaptations were made:

### 1. AnnData Compatibility & Matrix Orientations
- **R implementation:** Uses `SingleCellExperiment` and `ExpressionSet` objects where matrices are typically oriented as **Genes x Cells/Samples**.
- **Python implementation:** Native `AnnData` standard is **Cells/Samples x Genes**. The Python pipeline handles this implicitly, computing variances and means along the correct axes without requiring transpositions by the user.

### 2. Sparse Matrix Handling
Single-cell data often comes in highly sparse formats (e.g., `scipy.sparse.csr_matrix`). The Python pipeline safely converts chunks of the matrix to dense format (`.toarray()`) on-the-fly per cell-type/subject grouping. This avoids memory blowouts while allowing efficient computation of subject-level statistics.

### 3. Variance Calculation (NaN handling)
- **R implementation:** R's `var()` function handles single-observation arrays by returning `NA`.
- **Python implementation:** We use `np.nanvar(..., ddof=1)`. When calculating variance for a cell type present in only one subject, NumPy correctly produces `NaN` (suppressing division-by-zero warnings). These are masked later just like in the R pipeline, ensuring only robustly represented cell types are utilized.

### 4. NNLS Optimization
- **R implementation:** Uses the `nnls` package.
- **Python implementation:** Relies on `scipy.optimize.nnls`. The underlying Fortran routines are identical in behavior, ensuring mathematical equivalency.

### 5. Cross-Cell-Type Covariance (`ct_cov`)
- We have fully implemented the cross-cell-type covariance modeling in `music.py`, computing a gene-wise covariance matrix across cell types. We successfully utilized `np.einsum` to quickly and memory-efficiently resolve the tensor multiplication previously done iteratively in R.

### 6. Gene Subsetting and Relative Abundance Order
In the R version, calculating the relative abundance of the bulk matrix occurs *after* subsetting for common genes between the bulk and the single-cell reference. The Python implementation rigorously replicates this order of operations:
1. Find common genes.
2. Subset the bulk raw counts to common genes.
3. Normalize the subsetted bulk counts to sum to 1 across the common genes (relative abundance).

### 3. Hierarchical / Cluster-based Deconvolution (`music_prop_cluster`)
```python
import scanpy as sc
from music_cluster import music_prop_cluster

# Load single-cell reference and bulk data
sc_adata = sc.read_h5ad("sc_reference.h5ad")  # Obs: cells, Vars: genes
bulk_adata = sc.read_h5ad("bulk_data.h5ad")   # Obs: samples, Vars: genes

# Define groups and their sub-clusters
clusters_type = {
    'T_cells': ['CD4_T', 'CD8_T', 'Treg'],
    'B_cells': ['Memory_B', 'Naive_B'],
    'Myeloid': ['Macrophage', 'Dendritic']
}

# Define marker genes distinguishing sub-clusters within each group
group_markers = {
    'T_cells': ['CD4', 'CD8A', 'FOXP3'],
    'B_cells': ['CD27', 'IGHD'],
    'Myeloid': ['CD68', 'CD1C']
}

results = music_prop_cluster(
    bulk_adata=bulk_adata,
    sc_adata=sc_adata,
    group_markers=group_markers,
    groups_col="cell_type_group",
    clusters_col="cell_type",
    samples_col="subject_id",
    clusters_type=clusters_type,
    iter_max=1000
)

# Results contain hierarchical proportions for all sub-clusters
est_proportions = results['Est_prop_weighted_cluster']
print(est_proportions.head())
```
