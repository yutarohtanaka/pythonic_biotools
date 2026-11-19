# `facets`: Allele-Specific Somatic Copy Number, Tumor Purity & Ploidy

Standalone Python implementation of **FACETS** (Shen & Seshan, *Nucleic Acids Research* 2016) for allele-specific somatic copy-number analysis, tumor purity, ploidy, and clonal cellular fraction (CCF) estimation from tumor-normal SNP pileups, paired with continuous-BAF **ASCAT** equations (Van Loo et al., *PNAS* 2010).

## Citations

> Shen, R., & Seshan, V. E. (2016). FACETS: allele-specific copy number and clonal heterogeneity analysis tool for high-throughput DNA sequencing. *Nucleic Acids Research*, 44(16), e131. https://doi.org/10.1093/nar/gkw520
>
> Van Loo, P., Nordgard, S. H., Lingjærde, O. C., Russnes, H. G., Rye, I. H., Sun, W., ... & Børresen-Dale, A. L. (2010). Allele-specific copy number analysis of tumors. *Proceedings of the National Academy of Sciences*, 107(39), 16910–16915. https://doi.org/10.1073/pnas.1009843107

## Mathematical & Algorithmic Overview

1. **Preprocessing & GC/Depth Normalization (`pre_proc_sample`)**:
   - Filters normal read depth (`ndepth <= nor_dp <= ndepthmax`), classifies heterozygous SNPs via normal VAF `[het_thresh, 1 - het_thresh]` (or folded tumor VAF when `unmatched=True`), and thins adjacent SNPs by `snp_nbhd` bp while preferring heterozygous loci.
   - Computes library-centered total copy $\log_2$ ratio (`cnlr`) and removes non-linear GC bias via pure-NumPy Cleveland tricube LOWESS.
   - Computes Haldane-Anscombe ($+0.5$) allelic log-odds ratio (`valor`) and delta-method variance (`lorvar`):
     $$\text{valor}_i = \ln\!\left(\frac{(a_{T,i} + 0.5)(r_{N,i} + 0.5)}{(r_{T,i} + 0.5)(a_{N,i} + 0.5)}\right), \qquad \widehat{\text{mafR}}_s = \max\!\left(0,\; \frac{\sum_{i \in \mathcal{H}_s} w_{O,i}(\text{valor}_i^2 - \text{lorvar}_i)}{\sum_{i \in \mathcal{H}_s} w_{O,i}}\right)$$

2. **Two-Stage Bivariate Hotelling-$T^2$ CBS (`proc_sample`)**:
   - Jointly segments $\log R$ (all SNPs) and Winsorized $\log\text{OR}^2 / \text{lorvar}$ (heterozygous SNPs) per chromosome arm via vectorized prefix-sum Hotelling-$T^2$ Circular Binary Segmentation (`pre_seg` at `pre_cval=25` for residual noise estimation, followed by `seg` at `cval=150` and post-CBS pruning).
   - Hierarchically clusters indistinguishable segments (`cluster_segments`) and identifies candidate allelically balanced diploid baselines (`find_diploid_logr`).

3. **Gaussian + Folded-Normal EM (`em_cncf`) & ASCAT (`fit_ascat`)**:
   - Multi-start EM over $(\alpha, \gamma)$ (including WGD shifts $\gamma_{\text{WGD}} = \text{dipLogR} - \log_2(1 + \alpha)$) jointly models Gaussian $\log R$ and folded-normal $\log\text{OR}$ likelihoods over integer copy-number states $(C_{\text{major}}, C_{\text{minor}})$, estimating purity $\alpha$, ploidy $\psi$, integer `tcn`/`lcn`, `loh`/`cn_loh`, and clonal/subclonal cellular fractions (`cf_em`), guarded by a BIC pure-diploid null check.
   - Exposes exact Van Loo et al. (2010) continuous-BAF equations (`ascat_logr`, `ascat_baf`, `ascat_invert_cn`) and 2D grid goodness-of-fit solver (`fit_ascat`).

## Installation

```bash
pip install -e ./facets
```

## Quickstart

```python
from facets import run_facets, fit_ascat, compare_facets_ascat

res = run_facets(pileup_df, genome="GRCh38")
print(res.summary())
print(res.to_dataframe().head())

ascat_res = fit_ascat(res)
print(compare_facets_ascat(res, ascat_res))
```

## License

MIT (clean-room implementation from published mathematical equations in Shen & Seshan 2016 and Van Loo et al. 2010).
