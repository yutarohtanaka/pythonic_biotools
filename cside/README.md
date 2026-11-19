# `cside`: Cell-Type-Specific Spatial Differential Expression

Standalone Python implementation of **CSIDE** (Cable et al., *Nature Methods* 2022; `dmcable/spacexr`) for cell-type-specific differential expression in spatial transcriptomics (Visium, Slide-seq, Xenium/CosMx binned spots) given deconvolved cell-type proportions (e.g. from `BayesPrism`, `rctd-py`, or `cell2location`).

## Citations

> Cable, D. M., Murray, E., Shanmugam, V., Zhang, S., Zou, L. S., Diao, M., Chen, H., Macosko, E. Z., Irizarry, R. A., & Chen, F. (2022). Cell type-specific inference of differential expression in spatial transcriptomics. *Nature Methods*, 19(9), 1076–1087. https://doi.org/10.1038/s41592-022-01575-3
>
> Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal Statistical Society: Series B*, 65(1), 95–114. https://doi.org/10.1111/1467-9868.00374

## Statistical Model

For spatial spot $i$, total UMI count $N_i$, deconvolved cell-type weights $w_{ik}$, and spatial covariates $X_{id}$, CSIDE models gene counts via an overdispersed Poisson–lognormal mixture GLM:

$$Y_{ig} \mid \epsilon_{ig} \sim \text{Poisson}\!\left(N_i e^{\epsilon_{ig}} \sum_{k=1}^K w_{ik} \exp\!\left(\sum_{d=1}^D X_{id} \alpha_{dk}\right)\right), \quad \epsilon_{ig} \sim \mathcal{N}(0, \sigma_g^2)$$

estimating gene-specific overdispersion $\hat{\sigma}_g$ via mode-centered Gauss–Hermite quadrature (`roots_hermite`) and fitting cell-type $\times$ covariate parameters $\alpha_{dk}$ with analytical Fisher information covariance.

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`run_cside`, `run_cside_single`, `run_cside_regions`, `run_cside_nonparam`, `build_spline_design`, and `CSIDEConfig`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`spacexr::run.CSIDE` Parity) |
| :--- | :--- | :--- |
| **Cell-Type Filtering (`cell_type_threshold`)** | `15.0` aggregate spot weight (friendly to medium-sized panels; explicit overrides respected). | `125.0` aggregate spot weight matching `spacexr` (explicit parameter overrides respected). |
| **Spot Filtering (`weight_threshold`)** | `0.75` modeled cell-type weight sum per spot. | `0.99` when `doublet_mode=True`; `0.80` when `doublet_mode=False`. |
| **Gene Pre-Filter (`gene_threshold`)** | `1e-5` mean normalized UMI fraction $\frac{1}{I}\sum_i Y_{ig}/N_i$. | `5e-5` mean normalized UMI fraction. |
| **Significance Thresholds (`fdr`, `log_fc_thresh`)** | `fdr=0.05`, `log_fc_thresh=0.25` (natural log scale). | `fdr=0.01`, `log_fc_thresh=0.40` (natural log scale matching `spacexr`). |
| **Hessian Conditioning & Non-Parametric Splines** | Mild ridge (`1e-4`) stabilization on ill-conditioned spline Hessians + rotation-invariant joint spline Wald $\chi^2$ test. | Unpenalized Fisher information with strict Hessian eigenvalue/SE `test_error` filter + Bonferroni-min basis test. |

## Quickstart

```python
from cside import (
    build_spline_design,
    run_cside,
    run_cside_nonparam,
    run_cside_regions,
    run_cside_single,
)

# 1. Single continuous spatial covariate (AnnData or DataFrame/ndarray)
res = run_cside_single(
    adata,  # adata.X (spots x genes), adata.obsm["cell_type_weights"]
    explanatory_variable=adata.obs["hypoxia_score"],
    settings="default",
)
print(res.summary())
print(res.to_dataframe(significant_only=True).head())

# 2. Non-parametric 2D thin-plate regression spline spatial gradients
res_np = run_cside_nonparam(adata, df=12, basis="thin_plate", settings="default")

# 3. Upstream spacexr::run.CSIDE R parity mode across discrete anatomical regions
res_r = run_cside_regions(
    counts_df,
    cell_type_weights=weights_df,
    region_labels=regions,
    doublet_mode=False,
    settings="R_default",
)
```

## License

MIT (clean-room implementation from published mathematical equations in Cable et al. 2022 and Wood 2003).
