# `scarhrd`: Genomic Scar Homologous Recombination Deficiency (HRD) Scoring

Python implementation of **scarHRD** (Sztupinszki et al., *NPJ Breast Cancer* 2018) for computing Homologous Recombination Deficiency genomic scar scores (**HRD-LOH**, **TAI**, **LST**, and clinical **HRD-sum**) from allele-specific copy-number segmentation profiles (`GRCh38`/`hg38` and `GRCh37`/`hg19`).

## Citations

> Sztupinszki, Z., Diossy, M., Krzystanek, M., Reiniger, L., Csabai, I., Favero, F., Birkbak, N. J., Eklund, A. C., Syed, A., & Szallasi, Z. (2018). Migrating the SNP array-based homologous recombination deficiency measures to next generation sequencing data of breast cancer. *NPJ Breast Cancer*, 4, 16. https://doi.org/10.1038/s41523-018-0066-6
>
> Abkevich, V., et al. (2012). Patterns of genomic loss of heterozygosity predict homologous recombination repair defects in epithelial ovarian cancer. *British Journal of Cancer*, 107(10), 1776–1782.
>
> Birkbak, N. J., et al. (2012). Telomeric allelic imbalance indicates defective DNA repair and sensitivity to DNA-damaging agents. *Cancer Discovery*, 2(4), 366–375.
>
> Popova, T., et al. (2012). Ploidy and large-scale genomic instability consistently identify basal-like breast carcinomas with BRCA1/2 inactivation. *Cancer Research*, 72(21), 5454–5462.

## Mathematical & Algorithmic Overview

Built-in chromosome length and centromere boundary tables cover `chr1`–`chr22` and `chrX` for both **`GRCh38` (`hg38`)** and **`GRCh37` (`hg19`)**, accepting `"chr1"` and `"1"` notation interchangeably.

1. **`hrd_loh` (Abkevich et al., 2012)**:
   Number of Loss-of-Heterozygosity regions (`minor_cn == 0` and `total_cn >= 1`, excluding homozygous deletions `total_cn == 0`) longer than `15 Mb` (`15_000_000 bp`) that do **not** span an entire chromosome.
2. **`tai` (Telomeric Allelic Imbalance; Birkbak et al., 2012)**:
   Number of allelic-imbalance regions (`major_cn != minor_cn`, `total_cn >= 1`) of length $\ge 11\text{ Mb}$ (`11_000_000 bp`) extending to the subtelomeric end of a chromosome arm without crossing the centromere.
3. **`lst` (Large-Scale State Transitions; Popova et al., 2012)**:
   Evaluated independently on each chromosome arm (`p` and `q`, never across the centromere): iteratively filters/smooths segments $< 3\text{ Mb}$ (`3_000_000 bp`), merges resulting adjacent identical `(major_cn, minor_cn)` states, and counts chromosomal breaks between adjacent segments both $\ge 10\text{ Mb}$ (`10_000_000 bp`) separated by $\le 3\text{ Mb}$. Also computes Popova ploidy-adjusted `lst_ploidy_adjusted` and clinical `hrd_sum = hrd_loh + tai + lst` (with standard `>= 42` HRD-positivity call).

## Quickstart

```python
from scarhrd import calculate_hrd

res = calculate_hrd(allele_segments_df, genome="GRCh38")
print(res.hrd_loh, res.tai, res.lst, res.hrd_sum, res.hrd_positive)
print(res.to_dataframe())
```

## License

MIT
