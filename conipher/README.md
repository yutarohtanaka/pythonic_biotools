# `conipher`: Multi-Region Somatic Mutation Clustering & Phylogenetic Reconstruction

Standalone Python implementation of **CONIPHER** (Grigoriadis et al., *Nature Protocols* 2023) for multi-region somatic PhyloCCF estimation, Variational Dirichlet Process subclone clustering, copy-number-error cluster pruning, and phylogenetic tree enumeration under the Sum Condition (Pigeonhole Principle) and Crossing Rule.

## Citations

> Grigoriadis, K., Huebner, A., Bunkum, A., Colliver, E., Frankell, A. M., Hill, M. S., Thol, K., Birkbak, N. J., Swanton, C., Zaccaria, S., & McGranahan, N. (2023). CONIPHER: a computational framework for scalable phylogenetic reconstruction with error correction. *Nature Protocols*, 18(1), 159–183. https://doi.org/10.1038/s41596-022-00793-z

## `settings` Modes (`"default"` vs `"R_default"`)

All primary pipeline functions (`compute_phylo_ccf`, `cluster_ccf_multiregion`, `reconstruct_trees`, `run_conipher`) and `CONIPHER` / `ConipherConfig` accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`McGranahanLab/CONIPHER` v2.2 Parity) |
| :--- | :--- | :--- |
| **Multiplicity & PhyloCCF** | Beta-Binomial (`rho=0.01`) multiplicity selection + Jeffreys Beta posterior standard errors (`ccf_se`). | Binomial maximum-likelihood integer multiplicity selection across $\{1, \dots, C_{\text{maj}}\}$. |
| **Multi-Region Clustering** | Heteroskedastic SE-weighted Variational Dirichlet Process Gaussian Mixture (`alpha=0.5`) + SE-aware centroid merging. | Dirichlet Process mixture (`alpha=1.0`), `min_cluster_size = 5`, and fixed `0.085` regional merge threshold. |
| **Truncal Clone Collapse** | Merges ubiquitous clusters with `mean_ccf >= 0.85` (or 95% upper CI $\ge 0.92$ and `mean_ccf >= 0.80`) across all regions. | Fixed truncal collapse threshold `mean_ccf >= 0.85` across all sampled regions. |
| **Sum Condition & Crossing Rule** | SE-calibrated statistical tolerance $\max(0.05, 1.96 \, \text{SE}_{\text{pair}})$ per region when cluster SEs are available. | Fixed `ccf_buffer = 0.10` (`10` CCF percentage points on `[0, 1]` scale) for both Sum Condition and Crossing Rule. |
| **Tree Conflict Pruning** | Iteratively prunes conflicting copy-number-error/noise clusters ranked by mutation count and conflict severity. | Greedy smallest-cluster (`min n_mutations`) drop on irreconcilable tree feasibility conflicts. |

## Quickstart

```python
from conipher import (
    CONIPHER,
    compute_phylo_ccf,
    cluster_ccf_multiregion,
    reconstruct_trees,
    run_conipher,
)

# 1. End-to-end multi-region phylogenetic reconstruction
purity_map = {"R1": 0.80, "R2": 0.70, "R3": 0.75}
res = run_conipher(
    mutations_df,
    purity_by_region=purity_map,
    max_clusters=10,
    min_cluster_size=5,
    settings="default",
)
print(res.summary())
print(res.parent_child_df)
print(res.clone_proportions_df.head())

# 2. Exact McGranahanLab/CONIPHER v2.2 parity mode
res_r = CONIPHER(ccf_buffer=0.10, min_cluster_size=5, settings="R_default").fit(
    mutations_df, purity_by_region=purity_map
)
print(f"Alternative valid trees: {res_r.n_alternative_trees}")
```

## License

MIT (clean-room implementation from published mathematical equations in Grigoriadis et al., *Nature Protocols* 2023).
