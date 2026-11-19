# `cibersortx`: Cross-Platform & High-Resolution Bulk Deconvolution

Clean-room Python implementation of **CIBERSORT** and **CIBERSORTx** (Newman et al., *Nature Methods* 2015; *Nature Biotechnology* 2019) built purely from the published mathematical formulations for signature matrix construction, linear $\nu$-SVR bulk deconvolution, ComBat cross-platform batch alignment (`B-mode` and `S-mode`), Group-Mode cell-type GEP estimation, and High-Resolution sample-specific expression purification.

## Citations

> Newman, A. M., Liu, C. L., Green, M. R., Gentles, A. J., Feng, W., Xu, Y., Hoang, C. D., Diehn, M., & Alizadeh, A. A. (2015). Robust enumeration of cell subsets from tissue expression profiles. *Nature Methods*, 12(5), 453–457. https://doi.org/10.1038/nmeth.3337
>
> Newman, A. M., Steen, C. B., Liu, C. L., Gentles, A. J., Chaudhuri, A. A., Scherer, F., Khodadoust, M. S., Esfahani, M. S., Luca, B. A., Steiner, D., Diehn, M., & Alizadeh, A. A. (2019). Determining cell type abundance and expression from bulk tissues with digital cytometry. *Nature Biotechnology*, 37(7), 773–782. https://doi.org/10.1038/s41587-019-0114-2

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions (`build_signature_matrix`, `combat_align`, `deconvolve`, `impute_gep_group`, `impute_gep_highres`) accept `settings: Literal["default", "R_default", "cibersortx_default"] = "default"` (`"cibersortx_default"` is an alias for `"R_default"`):

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (Newman et al., *Nat Biotechnol* 2019 Protocol) |
| :--- | :--- | :--- |
| **Signature Matrix (`build_signature_matrix`)** | Direct single-cell Mann–Whitney $U$ one-vs-rest screening unless `replicates_per_type` / `test_method` are overridden. | Applies the $>50\%$ within-phenotype non-zero expression fraction gate, pools single cells into random mini-bulk replicates, and screens via Welch two-sample `ttest` ($q \le 0.01$, $\text{FC} \ge 1.5$). |
| **S-Mode Batch Alignment (`deconvolve(..., batch_mode="S")`)** | ComBat aligns Dirichlet pseudobulks to bulk mixtures and transforms $B_S$ directly via learned ComBat parameters. | Samples single-cell barcodes into synthetic mini-bulks $M_{\text{synth}}$, aligns $M_{\text{synth}} \to M_{\text{bulk}}$ via ComBat, and recovers $\hat{B}_{\text{adj}}$ via NNLS regression on the known mixing fractions $F_{\text{synth}}$ (Newman et al. 2019 Fig. 1c). |
| **Deconvolution & GEP Imputation** | Linear $\nu$-SVR ($\nu \in \{0.25, 0.5, 0.75\}$), QR-NNLS Group-Mode, and sliding-window High-Resolution Mode. | Identical $\nu$-SVR, QR-NNLS Group-Mode, and sliding-window High-Resolution Mode. |

## Quickstart

```python
from cibersortx import (
    build_signature_matrix,
    deconvolve,
    impute_gep_group,
    impute_gep_highres,
)

# 1. Build signature matrix using exact Newman et al. (2019) scRNA-seq protocol
sig = build_signature_matrix(
    sc_adata,
    cell_type_labels="cell_type",
    layer="counts",
    settings="R_default",
)

# 2. Deconvolve bulk expression with S-mode cross-platform alignment
res = deconvolve(
    bulk_df,
    sig,
    batch_mode="S",
    sc_ref=sc_adata,
    sc_cell_type_labels="cell_type",
    n_permutations=100,
    seed=0,
    settings="R_default",
)
print(res.to_dataframe().head())

# 3. Group-mode cell-type expression & High-Resolution sample-specific purification
grp = impute_gep_group(bulk_df, res, n_bootstrap=50, seed=0, settings="R_default")
hires = impute_gep_highres(bulk_df, res, n_permutations=199, seed=0, settings="R_default")
```

## Matrix Orientation Conventions

- **`AnnData` inputs** (`mixture`, `sc_ref`, `reference`): standard `obs x var` (`samples/cells x genes`).
- **`pandas.DataFrame` / `numpy.ndarray` inputs**: `genes x samples` (or `genes x cells` / `genes x cell_types`), with `samples_as_rows=True` / `cells_as_rows=True` supported when passing `samples x genes` DataFrames.

## License

MIT (clean-room implementation from published mathematical formulations of Newman et al., 2015 and 2019).
