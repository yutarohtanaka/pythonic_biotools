# `shatterseek`: Chromothripsis Detection from SVs & Copy-Number Profiles

Python implementation of the **ShatterSeek** statistical framework and **PCAWG** confidence classification criteria (Cortés-Ciriano et al., *Nature Genetics* 2020) for detecting localized single- and multi-chromosomal chromothripsis from whole-genome structural variants (SVs) and copy-number (CN) segments.

## Citation

- Cortés-Ciriano, I. et al. Comprehensive analysis of chromothripsis in 2,658 human cancers using whole-genome sequencing. *Nature Genetics* **52**, 331–341 (2020). https://doi.org/10.1038/s41588-019-0576-7
- Upstream Park Lab R package: https://github.com/parklab/ShatterSeek

## Statistical & Algorithmic Overview

1. **Interleaved SV Graph Clustering**:
   Connects intrachromosomal SVs (`DEL`, `DUP`, `h2hINV`, `t2tINV`) whose spans partially overlap without nesting ($a_1 < a_2 < b_1 < b_2$) into connected components and links interchromosomal translocations (`TRA`) to cluster intervals.
2. **Contiguous Copy-Number Oscillation Chains**:
   Quantifies `max_number_oscillating_cn_segments_2_states` and `max_number_oscillating_cn_segments_3_states` alongside interspersed LOH segments.
3. **Statistical Criteria & PCAWG Classification (`classify_chromothripsis`)**:
   - `pval_fragment_joins`: Multinomial goodness-of-fit ($\chi^2_3$, $G$-test, or exact) against uniform $(1/4, 1/4, 1/4, 1/4)$ join orientations.
   - `pval_exp_cluster`: KS or Anderson–Darling exponential test on consecutive intra-cluster breakpoint spacings.
   - `pval_chr_breakpoint_enrichment`: Binomial test comparing per-chromosome SV counts against callable chromosome length (`GRCh38` / `GRCh37`).
   - Classifies clusters into `"High"`, `"Low"`, or `"None"` PCAWG confidence tiers.

## Quickstart

```python
from shatterseek import detect_chromothripsis

res = detect_chromothripsis(sv_df, cn_segments_df, genome="GRCh38", settings="R_default")
print(res.has_high_confidence)
print(res.to_dataframe())
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Evaluates all chromosomes/contigs (including `chrY`) and reports **all** interleaved SV connected components per chromosome.
  - Pre-merges adjacent equal-CN segments before counting oscillations and defines 3-state oscillation as the longest contiguous collapsed subarray with $\le 3$ distinct CN states.
  - Uses `tra_margin_bp = 0`, tests `pval_fragment_joins` on intrachromosomal SV types only, runs `pval_exp_cluster` for $\ge 3$ breakpoints, and uses one-sided (`alternative="greater"`) Binomial enrichment.
- **`settings="R_default"` (Exact `parklab/ShatterSeek` v1.1 `function_criteria.R` parity)**:
  - Restricts chromosomes to `chr1`–`chr22` and `chrX` (excluding `chrY`) and reports only the **single largest interleaved SV cluster per chromosome**.
  - Does **not** pre-merge adjacent equal-CN segments, and computes 3-state CN oscillation as the longest contiguous run satisfying `abs(cn[i] - cn[i - 2]) <= 1` (`%in% c(0, 1)`).
  - Links interchromosomal `TRA`s within `tra_margin_bp = 10_000`, includes linked `TRA` `(strand1, strand2)` orientations in the 4-way `pval_fragment_joins` count, requires $\ge 6$ breakpoints for `pval_exp_cluster`, and uses a two-sided Binomial enrichment test (`alternative="two-sided"`).

## Key Outputs

- `ShatterSeekResult`: frozen dataclass with `clusters` (`list[ChromothripsisCluster]`), `has_chromothripsis`, `has_high_confidence`, and `.to_dataframe()`.
- `ChromothripsisCluster`: per-cluster coordinates, SV/TRA counts, 2- and 3-state CN oscillation lengths, statistical p-values, linked chromosomes/clusters, and PCAWG `confidence` (`"High"`, `"Low"`, `"None"`).

## Running Tests

```bash
PYTHONPATH=shatterseek /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest shatterseek/tests -q
```
