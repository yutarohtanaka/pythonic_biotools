# `shatterseek`: Chromothripsis Detection from SVs & Copy-Number Profiles

Clean-room Python implementation of the **ShatterSeek** statistical framework and **PCAWG** confidence classification rules (Cortés-Ciriano et al., *Nature Genetics* 2020) for detecting localized single- and multi-chromosomal chromothripsis from whole-genome structural variants (SVs) and copy-number (CN) segments.

## Citation

> Cortés-Ciriano, I., Lee, J. J.-K., Xi, R., Jain, D., Jung, Y. L., Yang, L., Gordenin, D., Klimczak, L. J., Zhang, C.-Z., Pellman, D. S., & Park, P. J. (2020). Comprehensive analysis of chromothripsis in 2,658 human cancers using whole-genome sequencing. *Nature Genetics*, 52(3), 331–341. https://doi.org/10.1038/s41588-019-0576-7

## Mathematical & Algorithmic Overview

1. **Interleaved SV Graph Clustering**:
   Intrachromosomal SVs (`DEL`, `DUP`, `h2hINV`, `t2tINV`, inferred automatically from `(strand1, strand2)` when omitted) on chromosome $c$ are connected in an undirected graph when their spans partially overlap without nesting ($a_1 < a_2 < b_1 < b_2$ or $a_2 < a_1 < b_2 < b_1$). Connected components define candidate SV clusters, linked across chromosomes via interchromosomal translocations (`TRA`).

2. **Contiguous Oscillating Copy-Number Chains**:
   Evaluates `max_number_oscillating_cn_segments_2_states` (strict $x, y, x, y, \dots$ alternation) and `max_number_oscillating_cn_segments_3_states` (aneuploid/subclonal up-to-3-state oscillations) alongside interspersed LOH segments.

3. **Statistical Criteria & PCAWG Tiers (`classify_chromothripsis`)**:
   - `pval_fragment_joins`: Multinomial goodness-of-fit ($\chi^2_3$, $G$-test, or exact multinomial) against uniform $(1/4, 1/4, 1/4, 1/4)$ across `("DEL", "DUP", "h2hINV", "t2tINV")`. Random re-ligation yields $p > 0.05$, rejecting tandem-duplicator or BFB fold-back artifacts.
   - `pval_exp_cluster`: Kolmogorov–Smirnov or D'Agostino–Stephens Anderson–Darling exponential goodness-of-fit on consecutive intra-cluster breakpoint spacings.
   - `pval_chr_breakpoint_enrichment`: One-sided Binomial test for chromosome-level SV enrichment weighted by callable chromosome length (`GRCh38` / `GRCh37`).
   - **High-Confidence**: ($\ge 6$ interleaved intrachromosomal SVs + $\ge 7$ oscillating 2-state CN segments) OR ($\ge 3$ interleaved SVs + $\ge 4$ `TRA`s + $\ge 7$ oscillating 2-state CN segments), with `pval_fragment_joins > 0.05`.
   - **Low-Confidence**: $\ge 6$ interleaved SVs + $4\text{–}6$ oscillating 2-state CN segments + `pval_fragment_joins > 0.05`.

## Quickstart

```python
from shatterseek import detect_chromothripsis

res = detect_chromothripsis(sv_df, cn_segments_df, genome="GRCh38")
print(res.has_high_confidence)
print(res.to_dataframe())
```

## License

MIT (clean-room implementation from published equations and criteria in Cortés-Ciriano et al., 2020).
