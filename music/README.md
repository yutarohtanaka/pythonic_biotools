# Pythonic MuSiC: Bulk RNA-seq Deconvolution

Python implementation of the **MuSiC** (Multi-subject Single Cell deconvolution; Wang et al., *Nature Communications* 2019) and **MuSiC2** (Fan et al., *Briefings in Bioinformatics* 2022) algorithms, natively compatible with `AnnData` and `scipy.sparse` count matrices.

## Installation

```bash
pip install ./music
```

## Overview

MuSiC estimates cell-type proportions in bulk RNA-seq data by leveraging multi-subject single-cell RNA-seq references, incorporating **cross-subject expression variance** ($\Sigma$) and **cell-type-specific library sizes** ($S$) into an iteratively reweighted non-negative least squares (NNLS) model.

Exported functions from `music`:
- `music_basis`: Computes cross-subject mean relative abundance (`M_theta`), cross-subject variance (`Sigma`), optional cross-cell-type covariance (`Sigma_ct`), library sizes (`S`, `M_S`), and design matrix (`Design_mtx`).
- `music_iter`: Core iteratively reweighted NNLS solver (`scipy.optimize.nnls`) with simplex convergence testing.
- `music_prop`: End-to-end bulk RNA-seq deconvolution pipeline.
- `music2_prop`: Iterative cross-condition deconvolution that detects and removes condition-associated cell-type-specific DE genes between bulk control and case cohorts.
- `music_prop_cluster`: Hierarchical / tree-guided deconvolution across major cell groups and closely related sub-clusters.

## Usage

```python
from music import music_prop, music2_prop, music_prop_cluster
```

### 1. Standard MuSiC (`music_prop`)

```python
import anndata as ad
from music import music_prop

sc_adata = ad.read_h5ad("sc_reference.h5ad")   # Obs: cells, Vars: genes
bulk_adata = ad.read_h5ad("bulk_data.h5ad")    # Obs: samples, Vars: genes

results = music_prop(
    bulk_adata=bulk_adata,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    ct_cov=False,
    iter_max=1000,
    eps=0.01,
)

est_proportions = results["Est_prop_weighted"]
print(est_proportions.head())
```

### 2. MuSiC2 (`music2_prop`)

```python
import anndata as ad
from music import music2_prop

bulk_control = ad.read_h5ad("bulk_control.h5ad")
bulk_case = ad.read_h5ad("bulk_case.h5ad")
sc_adata = ad.read_h5ad("sc_reference.h5ad")

results = music2_prop(
    bulk_control_adata=bulk_control,
    bulk_case_adata=bulk_case,
    sc_adata=sc_adata,
    cluster_col="cell_type",
    sample_col="subject_id",
    maxiter=200,
)

final_proportions = results["Est_prop"]
de_genes = results["DE_genes"]
```

### 3. Hierarchical / Cluster-based Deconvolution (`music_prop_cluster`)

```python
import anndata as ad
from music import music_prop_cluster

sc_adata = ad.read_h5ad("sc_reference.h5ad")
bulk_adata = ad.read_h5ad("bulk_data.h5ad")

clusters_type = {
    "T_cells": ["CD4_T", "CD8_T", "Treg"],
    "B_cells": ["Memory_B", "Naive_B"],
    "Myeloid": ["Macrophage", "Dendritic"],
}

group_markers = {
    "T_cells": ["CD4", "CD8A", "FOXP3"],
    "B_cells": ["CD27", "IGHD"],
    "Myeloid": ["CD68", "CD1C"],
}

results = music_prop_cluster(
    bulk_adata=bulk_adata,
    sc_adata=sc_adata,
    group_markers=group_markers,
    groups_col="cell_type_group",
    clusters_col="cell_type",
    samples_col="subject_id",
    clusters_type=clusters_type,
    iter_max=1000,
)

est_proportions = results["Est_prop_weighted_cluster"]
```

## Methodological Details

- **AnnData Orientation**: Expects standard `AnnData` matrices (`obs x var`, i.e., cells/samples by genes) in dense or sparse (`scipy.sparse`) format.
- **Zero-count & Single-donor Handling**: All-zero genes are filtered (`non_zero=True`); donor-cluster combinations with zero total counts are treated as `NaN` when computing cross-donor means and variances, while `NaN` variance estimates for single-donor clusters fall back cleanly to `0.0` in NNLS weighting (matching R `MuSiC`'s `na.rm = TRUE`).
- **Deterministic Gene Indexing**: Uses `pandas.Index.intersection` throughout to guarantee deterministic results across Python runs.
