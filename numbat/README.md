# `numbat`: Haplotype-Aware Single-Cell CNV & Tumor Phylogeny

Standalone Python implementation of **Numbat** (Gao et al., *Nature Biotechnology* 2023) for haplotype-aware allele-specific somatic copy number calling, copy-neutral LOH detection, malignant vs normal cell classification, and single-cell maximum-likelihood tumor phylogeny reconstruction on `AnnData` or count matrices.

## Citation

> Gao, T., Soldatov, R., Sarkar, H., Kurkiewicz, A., Biederstedt, E., Loh, P.-R., & Kharchenko, P. V. "Haplotype-aware analysis of somatic copy number variations from single-cell transcriptomes." *Nature Biotechnology* 41, 417–426 (2023). https://doi.org/10.1038/s41587-022-01468-y

## Mathematical & Algorithmic Overview

1. **Beta-Binomial Phased Allele Model**:
   Heterozygous phased SNP counts $(Y_t, N_t)$ follow a Beta-Binomial distribution $\text{BB}(Y_t; N_t, \gamma \mu_k, \gamma(1 - \mu_k))$ across an 11-state phased CNV space (`neu`, `del_p/m`, `amp_p/m`, `amp2_p/m`, `loh_p/m`, `bamp`, `bdel`) collapsing onto 6 biological states (`neu`, `amp`, `del`, `loh`, `bamp`, `bdel`).
2. **Haldane Haplotype Phase-Switch Transitions**:
   Population phasing switch errors along genetic distance $d_t$ (in cM) follow Haldane's mapping function:
   $$p_s(t) = \frac{1 - \exp(-2 \nu d_t)}{2}$$
   allowing symmetric mirrored haplotype states (`+1`/`-1`) to flip seamlessly inside continuous CNV segments while resetting state priors at chromosome boundaries.
3. **Poisson-LogNormal (PLN) Expression Model**:
   Gene UMI counts $X_{c,g}$ are modeled via $Q$-point Gauss-Hermite quadrature in log-space:
   $$X_{c,g} \mid z \sim \text{Poisson}\!\left(\exp(\log(L_c \lambda_g^{\text{ref}}) + \delta_c + \log \phi_k + \sigma_g z)\right), \quad z \sim \mathcal{N}(0, 1).$$
4. **Haplotype-Aware HMM (`numba` + NumPy Parity)**:
   Log-space Forward-Backward and Viterbi algorithms execute identically via optional `@numba.njit` acceleration (`backend="numba"`) or pure vectorized NumPy/SciPy (`backend="numpy"`).
5. **Single-Cell Scoring & NJ + NNI ML Phylogeny**:
   Per-cell segment Bayes factors (`p_cnv_x`, `p_cnv_y`, `p_cnv`) feed into Neighbor-Joining initialization (rooted at a diploid outgroup), $O(C \cdot M)$ post-order infinite-sites branch likelihood scoring, steepest-ascent Nearest-Neighbor Interchange (NNI) refinement, and EM subclonal posterior assignment.

## Installation

```bash
pip install -e ./numbat
# Optional Numba acceleration:
pip install -e "./numbat[accel]"
```

## Quickstart

```python
from numbat import NumbatConfig, annotate_anndata, run_numbat

res = run_numbat(
    adata,
    allele_df=allele_df,
    ref_cells="is_ref",
    config=NumbatConfig(backend="auto"),
)
print(res.summary())
print(res.segments_dataframe())
print(res.to_dataframe().head())
adata = annotate_anndata(adata, res)
```

## Matrix Orientation

- `AnnData` inputs follow standard `obs x var` (`cells x genes`), supporting both dense arrays and `scipy.sparse` matrices.
- Standalone `pandas.DataFrame` or `numpy.ndarray` count matrices also expect `(cells x genes)` orientation paired with a `gene_info` DataFrame containing `chrom` and `pos` (or `start`/`end`).

## License

MIT (clean-room implementation from published equations in Gao et al., *Nature Biotechnology* 2023).
