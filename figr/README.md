# `figr`: Domains of Regulatory Chromatin (DORCs) & cis Peak-to-Gene Networks

Standalone Python implementation of **FigR** (Kartha et al., *Cell Genomics* 2022) for inferring cis peak-to-gene associations, Domains of Regulatory Chromatin (DORCs), kNN-smoothed per-cell DORC activity scores, and transcription factor (TF)-DORC gene regulatory networks from paired single-cell multiome (scATAC-seq + scRNA-seq) or metacell profiles.

## Citation

> Kartha VK, Duarte FM, Hu Y, Ma S, Chew JG, Lareau CA, Earl A, Burkett ZD, Kohlway AS, Lebofsky R, Buenrostro JD. Functional inference of gene regulation using single-cell multi-omics. *Cell Genomics*. 2022;2(9):100166. doi:[10.1016/j.xgen.2022.100166](https://doi.org/10.1016/j.xgen.2022.100166)

## Mathematical & Algorithmic Overview

1. **Cis Peak-to-Gene Correlation (`run_gene_peak_corr`)**:
   - Maps ATAC peaks overlapping a symmetric window (`±window_pad`, default `50,000` bp) around strand-aware 0-based GTF TSS coordinates (`+`: `start - 1`; `-`: `end - 1`).
   - Computes exact tie-averaged Spearman correlations $r_{\text{obs}}$ across aligned cells or metacells without unnecessary densification of sparse scATAC matrices.
   - Samples $B$ GC-content and mean-accessibility-matched background peaks in regularized Mahalanobis-whitened $(\log_{10} \text{mean\_acc}, \text{GC})$ space (`get_background_peaks`) and evaluates an empirical null Z-test:
     $$Z_{g,p} = \frac{r_{\text{obs}}(g, p) - \mu_{\text{bg}}(g, p)}{\sigma_{\text{bg}}(g, p)}$$
     with `NaN`-aware Benjamini–Hochberg FDR control.
2. **DORC Calling (`call_dorcs`)**:
   - Filters significant cis peak-gene links, deduplicates `(gene, peak)` pairs, ranks genes by linked peak count, and calls DORCs with $n_{\text{peaks}} \ge k_{\text{DORC}}$ (default `5`).
3. **kNN-Smoothed DORC Activity Scoring (`get_dorc_scores`, `smooth_scores_knn`)**:
   - Library-size normalizes scATAC counts per cell (`scale_factor = 1e4`), aggregates linked peak accessibility per DORC (`"sum"` or `"mean"`), and smooths sparse single-cell scores via a row-stochastic cell kNN matrix $S^{\text{smooth}} = W S^{\text{raw}}$.
4. **TF-DORC Gene Regulatory Network Inference (`run_figr_grn`)**:
   - Tests TF motif enrichment in each DORC's linked peaks against matched background peaks (`"background_z"` or `"hypergeometric"`), evaluates TF-RNA vs. DORC-score correlation across cells, and combines significance into a signed regulation score separating activators ($> 0$) from repressors ($< 0$).

## Installation

```bash
pip install -e ./figr
```

## Quickstart

```python
from figr import (
    run_gene_peak_corr,
    call_dorcs,
    get_dorc_scores,
    run_figr_grn,
)

corr_res = run_gene_peak_corr(rna_adata, atac_adata, tss=tss_df, peak_gc=gc_vec)
dorcs = call_dorcs(corr_res, dorc_k=5, p_cut=0.05)
scores = get_dorc_scores(atac_adata, dorcs, embedding=lsi_embedding, k=30)
grn = run_figr_grn(
    scores,
    rna_adata,
    dorc_links=dorcs,
    peak_motifs=motif_df,
    atac=atac_adata,
    peak_gc=gc_vec,
)
print(grn.to_dataframe().head())
```

## License

MIT (clean-room implementation from Kartha et al., *Cell Genomics* 2022).
