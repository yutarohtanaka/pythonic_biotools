# `figr`: Domains of Regulatory Chromatin (DORCs) & cis Peak-to-Gene Networks

Standalone Python implementation of **FigR** (Kartha et al., *Cell Genomics* 2022) for inferring cis peak-to-gene associations, Domains of Regulatory Chromatin (DORCs), kNN-smoothed per-cell DORC activity scores, and transcription factor (TF)–DORC gene regulatory networks from paired single-cell multiome (scATAC-seq + scRNA-seq) or metacell profiles.

## Citation

- Kartha, V. K. et al. Functional inference of gene regulation using single-cell multi-omics. *Cell Genomics* **2**, 100166 (2022). https://doi.org/10.1016/j.xgen.2022.100166
- Upstream Buenrostro Lab R package: https://github.com/buenrostrolab/FigR

## Statistical & Algorithmic Overview

1. **Cis Peak-to-Gene Correlation (`run_gene_peak_corr`)**:
   Maps ATAC peaks within `±window_pad` (`50 kb` default) of strand-aware 0-based GTF TSSs, computes exact tie-averaged Spearman correlations $r_{\text{obs}}$ on sparse or dense matrices, and evaluates empirical Z-scores against $B$ GC-content and mean-accessibility-matched background peaks (`get_background_peaks`) in regularized Mahalanobis space:
   $$Z_{g,p} = \frac{r_{\text{obs}}(g, p) - \mu_{\text{bg}}(g, p)}{\sigma_{\text{bg}}(g, p)}$$
2. **DORC Calling (`call_dorcs`)**:
   Filters significant cis peak-gene links, deduplicates `(gene, peak)` pairs, ranks genes by linked peak multiplicity, and calls DORCs exceeding a minimum linked-peak cutoff.
3. **kNN-Smoothed DORC Activity Scoring (`get_dorc_scores`, `smooth_scores_knn`)**:
   Library-size normalizes scATAC counts per cell (`scale_factor = 1e4`), sums/averages linked peak accessibility per DORC, and smooths cell profiles over a row-stochastic cell kNN graph $S^{\text{smooth}} = W S^{\text{raw}}$.
4. **TF–DORC Gene Regulatory Network Inference (`run_figr_grn`)**:
   Combines TF motif enrichment across DORC peaks with TF-RNA vs. DORC-accessibility correlation across cells into a signed regulation score distinguishing candidate activators ($> 0$) and repressors ($< 0$).

## Quickstart

```python
from figr import call_dorcs, get_dorc_scores, run_figr_grn, run_gene_peak_corr

corr_res = run_gene_peak_corr(
    rna_adata, atac_adata, tss=tss_df, peak_gc="gc_content", settings="R_default"
)
dorcs = call_dorcs(corr_res, p_cut=0.05, settings="R_default")
scores = get_dorc_scores(atac_adata, dorcs, embedding=lsi_embedding, k=30)
grn = run_figr_grn(
    scores,
    rna_adata,
    dorc_links=dorcs,
    peak_motifs=motif_df,
    atac=atac_adata,
    peak_gc="gc_content",
    settings="R_default",
)
print(grn.to_dataframe().head())
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - `run_gene_peak_corr`: Retains multi-mapping cis peaks (`keep_multi_mapping_peaks=True`) and all tested pairs (`p_cut=None`).
  - `call_dorcs`: Calls DORCs at `dorc_k = 5` (`>= 5` linked peaks).
  - `run_figr_grn`: Uses `score_mode="product"` (`sign(r) * (-log10 p_motif) * (-log10 p_corr)`), `corr_p_method="ttest"`, one-sided motif enrichment p-values (`norm.sf(z)`), and unpooled per-DORC peaks (`dorc_k_pool=0`).
- **`settings="R_default"` (Exact `buenrostrolab/FigR` `cisCor.R` / `DORC.R` / `FigR.R` parity)**:
  - `run_gene_peak_corr`: Sets `keep_multi_mapping_peaks=False` (retaining only the maximum-`r_obs` gene per peak) and `p_cut=None`.
  - `call_dorcs`: Uses `dorc_k = 7` (`>= 7` significant linked peaks, matching `dorcJPlot` `cutoff = 7`).
  - `run_figr_grn`: Uses `score_mode="kartha_union"` ($\text{sign}(r) \cdot -\log_{10}(1 - (1 - p_{\text{enrich}})(1 - p_{\text{corr}}))$), `corr_p_method="z_across_tfs"`, two-sided motif enrichment p-value `2 * norm.sf(abs(z))` (setting enrichment $-\log_{10} p$ to `0` when $z < 0$), and `dorc_k_pool=30` (pooling peaks across top-$k$ k-NN correlated DORCs when $\ge 2$ DORCs exist).

## Key Outputs

- `GenePeakCorrResult`: frozen dataclass with `.to_dataframe()`, `.significant()`, and `.summary()`.
- `DORCCallResult` & `DORCScoresResult`: ranked DORC calls (`dorc_genes`, `.dorc_table()`) and raw/kNN-smoothed `cells x dorcs` matrices.
- `FigRGRNResult`: TF–DORC regulatory edges with `dorc_reg_score`, `role` (`'activator'` / `'repressor'` / `'neutral'`), `.significant()`, `.drivers()`, and `.to_adjacency()`.

## Running Tests

```bash
PYTHONPATH=figr /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest figr/tests -q
```
