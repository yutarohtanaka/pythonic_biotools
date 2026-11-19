# `csaw`: Sliding-Window Differential ChIP-seq, ATAC-seq & CUT&RUN Analysis

Standalone Python implementation of **csaw** (Lun & Smyth, *Nucleic Acids Research* 2016) for peak-free sliding-window differential binding and chromatin accessibility analysis, paired with Quasi-Likelihood Negative Binomial GLMs (Lun, Chen & Smyth 2016) and rigorous region-level false discovery rate (FDR) control via weighted Simes' procedure.

## Citations

> Lun, A. T. L., & Smyth, G. K. (2016). csaw: a Bioconductor package for differential binding analysis of ChIP-seq data using sliding windows. *Nucleic Acids Research*, 44(5), e45. https://doi.org/10.1093/nar/gkv1191
>
> Lun, A. T. L., Chen, Y., & Smyth, G. K. (2016). It's DE-licious: A recipe for differential expression analyses of RNA-seq experiments using quasi-likelihood methods in edgeR. *Methods in Molecular Biology*, 1418, 391–416. https://doi.org/10.1007/978-1-4939-3578-9_19
>
> Robinson, M. D., & Oshlack, A. (2010). A scaling normalization method for differential expression analysis of RNA-seq data. *Genome Biology*, 11(3), R25. https://doi.org/10.1186/gb-2010-11-3-r25

## `settings` Modes (`"default"` vs `"R_default"`)

Every primary function (`window_counts`, `filter_windows_global`, `filter_windows_local`, `norm_factors`, `norm_offsets`, `glm_ql_fit_test`, `merge_windows`, `combine_tests`, and `get_best_test`) accepts `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`LTLA/csaw` v1.44 Parity) |
| :--- | :--- | :--- |
| **Window Tiling (`window_counts`)** | `width=150` bp (nucleosome/TF window), `spacing=50` bp, `ext=100` bp, `min_mapq=20`, `filter_min_total=10`. | `csaw::windowCounts`: `width=50` bp, `spacing=50` bp, `ext=100` bp, `min_mapq=0` (`minq=NA`), `filter_min_total=10`. |
| **Background Binning (`bin_mode=True`)** | `width=10000` bp non-overlapping bins, 5'-end / PE-midpoint counting (`ext=1`), `filter_min_total=0`. | Identical `csaw::windowCounts(..., bin=TRUE, width=10000)` 5'-end / midpoint binning. |
| **Normalization (`norm_factors` / `norm_offsets`)** | TMM scaling (`logratio_trim=0.3`, `sum_trim=0.05`) on 10 kb bins or Cleveland tricube loess offsets (`span=0.3`). | Exact `edgeR::calcNormFactors` 75th-percentile reference selection & rank trimming; row-centered loess `normOffsets`. |
| **QL Dispersion (`glm_ql_fit_test`)** | Cox–Reid trended NB dispersion + **robust** Empirical Bayes `squeezeVar` (`robust=True`) attenuating outlier prior df. | Standard non-robust `edgeR::glmQLFit(..., robust=FALSE)` unless `robust=True` is explicitly passed. |
| **Region Combination (`combine_tests`)** | Weighted Simes' $p_{\text{region}}$, active Simes-set directional counts (`fc_threshold=0.5`), BH region `FDR`. | `csaw` v1.44 directional rule: counts `num_up_logFC` / `num_down_logFC` strictly among windows with cluster Simes $q \le 0.05$ and $|\text{logFC}| \ge 0.5$. |

## Quickstart

```python
import numpy as np
from csaw import (
    window_counts,
    filter_windows_global,
    norm_factors,
    glm_ql_fit_test,
    merge_windows,
    combine_tests,
    get_best_test,
)

# 1. Count extended reads across sliding windows & 10 kb background bins
win_data = window_counts(bam_files=bams, settings="default")
bg_bins = window_counts(bam_files=bams, width=10000, bin_mode=True, settings="default")

# 2. Filter low-enrichment background windows & compute composition-bias TMM factors
filt = filter_windows_global(win_data, background_bins=bg_bins, min_fc=3.0, settings="default")
tmm = norm_factors(bg_bins, settings="default")

# 3. Quasi-Likelihood NB F-test + region merging + weighted Simes FDR control
design = np.column_stack([np.ones(len(bams)), [0, 0, 0, 1, 1, 1]])
tab = glm_ql_fit_test(filt.filtered_data, design=design, norm_factors=tmm, settings="default")
merged = merge_windows(filt.filtered_data, tol=100, max_width=5000, settings="default")
regions = combine_tests(merged, tab, fc_threshold=0.5, settings="default")
summits = get_best_test(merged, tab, settings="default")
```

## License

MIT (clean-room implementation from published mathematical equations in Lun & Smyth 2016, Lun et al. 2016, and Robinson & Oshlack 2010).
