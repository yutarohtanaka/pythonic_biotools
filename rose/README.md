# `rose`: Rank Ordering of Super-Enhancers (Python)

Standalone Python implementation of the **ROSE** super-enhancer calling algorithm (Whyte et al., *Cell* 2013; Lovén et al., *Cell* 2013) with blacklist exclusion, promoter-window (`±2.5 kb` TSS) filtering, distal enhancer stitching (`12.5 kb`), SPM or `bigWig` (with optional Input/IgG control subtraction) signal quantification, geometric tangent-slope inflection thresholding, and nearest-gene annotation.

## Citations

- Whyte, W. A. et al. Master transcription factors and mediator establish super-enhancers at key cell identity genes. *Cell* **153**, 307–319 (2013). https://doi.org/10.1016/j.cell.2013.03.035
- Lovén, J. et al. Selective inhibition of tumor oncogenes by disruption of super-enhancers. *Cell* **153**, 320–334 (2013). https://doi.org/10.1016/j.cell.2013.03.036
- Upstream Young Lab ROSE codebase: https://bitbucket.org/young_computation/rose

## Statistical & Algorithmic Overview

1. **Constituent Filtering**: Filters raw ChIP-seq / ATAC-seq constituent peaks against optional ENCODE blacklist regions, standard chromosomes, and proximal `TSS ± tss_exclusion_bp` (`2.5 kb` default) promoter windows.
2. **Enhancer Stitching**: Merges adjacent constituent enhancers separated by `<= stitch_bp` (`12.5 kb` default) into stitched regulatory domains.
3. **Signal Integration**: Sums Signal-Per-Million (`SPM`) constituent scores or integrates `width_bp * max(0, chip_mean - control_mean)` directly from `.bigWig` tracks via `pyBigWig`.
4. **Inflection-Point Thresholding**: Ranks stitched domains in ascending signal order and identifies the geometric tangent inflection point separating typical enhancers (`signal <= cutoff`) from super-enhancers (`signal > cutoff`).

## Quickstart

```python
from rose import call_rose_super_enhancers

# Call ROSE super-enhancers (supports settings="default" or settings="R_default")
res = call_rose_super_enhancers(
    peaks="H3K27ac_peaks.narrowPeak",
    tss_df=tss_df,
    stitch_bp=12_500,
    tss_exclusion_bp=2_500,
    settings="R_default",
)
print(f"Cutoff: {res.cutoff_signal:.2f} ({len(res.super_enhancers)} super-enhancers)")
print(res.super_enhancers.head())

# Plot classic ROSE hockey-stick curve with tangent cutoff & top super-enhancer labels
fig = res.plot(top_n_labels=5, figsize=(7, 5))
fig.savefig("rose_hockey_stick.png", dpi=300)
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Retains `chr1`–`chr22`, `chrX`, and `chrY` when `standard_chroms_only=True`.
  - Excludes any constituent peak **partially or fully overlapping** `TSS ± tss_exclusion_bp` (preventing broad promoter peaks from stitching distal enhancers across a TSS).
  - Keeps stitched domains intact regardless of how many silent/intervening TSSs they span.
  - Finds the inflection threshold maximizing `x_norm - y_norm` (`x_norm = 0..1` over `N - 1`, `y_norm = sig / max_sig`).
- **`settings="R_default"` (Exact Young Lab `ROSE_main.py` & `ROSE_callSuper.R` parity)**:
  - Excludes `chrY` when `standard_chroms_only=True`.
  - **TSS containment (`getContainers`)**: Excludes ONLY constituent peaks **strictly/fully contained** inside `[tss - tss_exclusion_bp, tss + tss_exclusion_bp]`.
  - **Multi-gene unstitching (`ROSE_main.py` L98–115)**: Reverts any stitched region overlapping `> 2` distinct gene `TSS ± 50 bp` windows back into its original un-stitched constituent peaks.
  - **Cutoff (`ROSE_callSuper.R` `calculate_cutoff`)**: Finds the point minimizing `y - slope * x` where `slope = (max_sig - min_sig) / N` and `x = 1..N`.

## Key Outputs

`ROSEResult` frozen dataclass with attributes:
- `super_enhancers`: `pandas.DataFrame` of super-enhancer calls sorted by descending signal (`rank_descending = 1, 2, ...`).
- `stitched_regions`: `pandas.DataFrame` of all stitched/unstitched enhancers with `chrom`, `start`, `end`, `width_bp`, `signal`, `n_constituent_peaks`, `rank_ascending`, `rank_descending`, `x_norm`, `y_norm`, `is_super_enhancer`, and `nearest_gene`.
- `cutoff_signal`, `cutoff_rank_ascending`, and `signal_unit`.

## Running Tests

```bash
PYTHONPATH=rose /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest rose/tests -q
```
