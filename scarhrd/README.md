# `scarhrd`: Genomic Scar Homologous Recombination Deficiency (HRD) Scoring

Python implementation of **scarHRD** (Sztupinszki et al., *NPJ Breast Cancer* 2018) for computing Homologous Recombination Deficiency genomic scar scores (**HRD-LOH**, **TAI**, **LST**, and clinical **HRD-sum**) from allele-specific copy-number segmentation profiles (`GRCh38`/`hg38` and `GRCh37`/`hg19`).

## Citations

- Sztupinszki, Z. et al. Migrating the SNP array-based homologous recombination deficiency measures to next generation sequencing data of breast cancer. *NPJ Breast Cancer* **4**, 16 (2018). https://doi.org/10.1038/s41523-018-0066-6
- Abkevich, V. et al. Patterns of genomic loss of heterozygosity predict homologous recombination repair defects in epithelial ovarian cancer. *British Journal of Cancer* **107**, 1776–1782 (2012).
- Birkbak, N. J. et al. Telomeric allelic imbalance indicates defective DNA repair and sensitivity to DNA-damaging agents. *Cancer Discovery* **2**, 366–375 (2012).
- Popova, T. et al. Ploidy and large-scale genomic instability consistently identify basal-like breast carcinomas with BRCA1/2 inactivation. *Cancer Research* **72**, 5454–5462 (2012).
- Upstream R package: https://github.com/sztup/scarHRD

## Statistical & Algorithmic Overview

Built-in chromosome length and centromere boundary tables cover `chr1`–`chr22` and `chrX` for both **`GRCh38` (`hg38`)** and **`GRCh37` (`hg19`)**, accepting `"chr1"` and `"1"` notation interchangeably.

1. **`hrd_loh` (Loss of Heterozygosity)**: Counts LOH regions (`minor_cn == 0` and `total_cn >= 1`, excluding homozygous deletions `total_cn == 0`) longer than `15 Mb` (`15_000_000 bp`) that do not span an entire chromosome.
2. **`tai` (Telomeric Allelic Imbalance)**: Counts allelic-imbalance regions (`major_cn != minor_cn`, `total_cn >= 1`) extending to the telomeric end of a chromosome arm without crossing the centromere.
3. **`lst` (Large-Scale State Transitions)**: Evaluated per chromosome arm independently (`p` and `q`): iteratively removes micro-segments `< 3 Mb`, merges newly adjacent identical `(major_cn, minor_cn)` states, and counts breaks between adjacent segments both `>= 10 Mb` within `3 Mb`. Combines components into clinical `hrd_sum = hrd_loh + tai + lst` (default positivity threshold `>= 42`).

## Quickstart

```python
from scarhrd import calculate_hrd, calculate_hrd_cohort

# Single-sample scoring (settings="default" or settings="R_default")
res = calculate_hrd(allele_segments_df, genome="GRCh38", settings="R_default")
print(res.hrd_loh, res.tai, res.lst, res.hrd_sum, res.hrd_positive)
print(res.to_dataframe())

# Multi-sample cohort scoring
cohort_df = calculate_hrd_cohort(cohort_segments_df, genome="GRCh38", settings="R_default")
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Evaluates `chr1`–`chr22` and `chrX` (`include_chr_x=True` by default).
  - Applies `< 3 Mb` shortest-first (`argmin`) micro-segment smoothing before evaluating `LST`, `HRD-LOH`, and `TAI`.
  - Requires `TAI` segments to be `>= 11 Mb` (`min_tai_bp`) and reach within `10 Mb` (`telomere_margin_bp`) of the chromosome end; treats any `major_cn != minor_cn` (`total_cn >= 1`) as allelic imbalance.
  - Uses `<= 3_000_000 bp` gap tolerance for `LST` and `lst_ploidy_adjusted = max(0.0, lst - 2.5 * (ploidy - 2.0))`.
- **`settings="R_default"` (Exact `sztup/scarHRD` v0.1.1 parity)**:
  - Evaluates autosomes `chr1`–`chr22` only (`include_chr_x=False` unless explicitly set).
  - **`HRD-LOH` (`calc.hrd.R`)**: Counts LOH segments `> 15_000_000 bp` (excluding whole-chromosome LOH) **without** `< 3 Mb` LST pre-smoothing.
  - **`TAI` (`calc.ai_new.R`)**: Filters segments `>= 1_000_000 bp` without `< 3 Mb` smoothing; treats the outermost arm segment as telomeric whenever `end <= cen_start` (`p`-arm) or `start >= cen_end` (`q`-arm) without `11 Mb` length or `10 Mb` telomere margin gates; applies modal odd-ploidy neutrality (`round(ploidy) % 2 == 1 and total_cn == round(ploidy) and minor_cn != 0` is **not** allelic imbalance).
  - **`LST` (`calc.lst.R`)**: Iteratively pops `< 3_000_000 bp` segments in **left-to-right coordinate order**, merges adjacent identical `(total_cn, minor_cn)` states, counts breaks between `>= 10_000_000 bp` segments with strict gap `< 3_000_000 bp`, and sets `lst_ploidy_adjusted = max(0.0, lst - 15.5 * ploidy)`.

## Key Outputs

- `ScarHRDResult`: frozen dataclass containing `sample_id`, `genome`, `hrd_loh`, `tai`, `lst`, `hrd_sum`, `ploidy`, `lst_ploidy_adjusted`, `hrd_sum_ploidy_adjusted`, `hrd_positive`, `hrd_threshold`, `per_chrom_breakdown`, and `.to_dataframe()`.
- `calculate_hrd_cohort`: `pandas.DataFrame` with one summary row per sample.

## Running Tests

```bash
PYTHONPATH=scarhrd /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest scarhrd/tests -q
```
