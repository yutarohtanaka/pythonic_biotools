# `numbat`: Haplotype-Aware Single-Cell CNV & Tumor Phylogeny

Standalone Python implementation of **Numbat** (Gao et al., *Nature Biotechnology* 2023) for haplotype-aware allele-specific somatic copy number calling, copy-neutral LOH detection, malignant vs normal cell classification, and single-cell maximum-likelihood tumor phylogeny reconstruction on `AnnData` or count matrices.

## Citation

> Gao, T., Soldatov, R., Sarkar, H., Kurkiewicz, A., Biederstedt, E., Loh, P.-R., & Kharchenko, P. V. "Haplotype-aware analysis of somatic copy number variations from single-cell transcriptomes." *Nature Biotechnology* 41, 417–426 (2023). https://doi.org/10.1038/s41587-022-01468-y

## `settings` Modes (`"default"` vs `"R_default"`)

`run_numbat`, `run_joint_hmm`, `fit_allele_hmm`, `fit_expression_hmm`, and `NumbatConfig` accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`kharchenkolab/numbat` v1.5.2 Parity) |
| :--- | :--- | :--- |
| **Joint HMM State Catalog** | 11-state phased CNV space (`DEFAULT_STATES`: `neu`, `del_p/m`, `amp_p/m`, `amp2_p/m`, `loh_p/m`, `bamp`, `bdel`). | 15-state `s15` catalog (`S15_STATES`, matching `hahmmr::run_joint_hmm_s15`): `neu`, `bamp`, `bdel` (`00`, $\phi < 1$), clonal & subclonal mono-allelic deletions, clonal & subclonal CN-neutral LOH, and single & multi-level unbalanced amplifications. |
| **Segment Refinement** | Discrete state parameters $(\phi_k, \theta_k)$ used for segment LLR and single-cell scoring alongside reported $(\hat{\phi}_{\text{mle}}, \hat{\theta}_{\text{mle}})$. | Continuous $(\hat{\phi}_{\text{mle}}, \hat{\theta}_{\text{mle}})$ optimization refines segment log-likelihood ratios and continuous single-cell scoring parameters. |
| **Multi-Pseudobulk Consensus (`retest_bulks`)** | Greedy highest-`log_bf` non-overlapping segment deduplication across pseudobulk HMM runs. | Full `retest_bulks` pass re-evaluating consensus CNV intervals across pooled tumor and per-clone pseudobulks to compute exact per-segment LLRs, Bayesian 6-state posteriors, and clonal CN-neutral LOH. |

## Quickstart

```python
from numbat import NumbatConfig, annotate_anndata, run_numbat

# Default Pythonic pipeline
res = run_numbat(adata, allele_df=allele_df, ref_cells="is_ref", settings="default")

# Exact kharchenkolab/numbat v1.5.2 (s15 + continuous MLE + retest_bulks) parity mode
res_r = run_numbat(adata, allele_df=allele_df, ref_cells="is_ref", settings="R_default")
print(res_r.summary())
print(res_r.segments_dataframe())
adata = annotate_anndata(adata, res_r)
```

## Matrix Orientation

- `AnnData` inputs follow standard `obs x var` (`cells x genes`), supporting both dense arrays and `scipy.sparse` matrices.
- Standalone `pandas.DataFrame` or `numpy.ndarray` count matrices also expect `(cells x genes)` orientation paired with a `gene_info` DataFrame containing `chrom` and `pos` (or `start`/`end`).

## License

MIT (clean-room implementation from published equations in Gao et al., *Nature Biotechnology* 2023).
