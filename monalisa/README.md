# `monalisa`: Binned Motif Enrichment & Randomized Lasso Stability Selection

Standalone Python implementation of **monaLisa** (`fmicompbio/monaLisa` v1.12; Machlab et al., *Bioinformatics* 2022) for sequence-composition-corrected binned transcription factor motif enrichment and Complementary-Pairs Randomized Lasso Stability Selection (Meinshausen & Bühlmann, *JRSS-B* 2010; Shah & Samworth, *JRSS-B* 2013) across regulatory genomic regions such as differentially methylated regions (DMRs), differential ChIP/ATAC-seq windows, and enhancers.

## Citations

> Machlab, D., Burger, L., Soneson, C., Rijli, F. M., Schübeler, D., & Stadler, M. B. (2022). monaLisa: an R/Bioconductor package for identifying regulatory motifs. *Bioinformatics*, 38(9), 2624–2625. https://doi.org/10.1093/bioinformatics/btac102
>
> Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 72(4), 417–473. https://doi.org/10.1111/j.1467-9868.2010.00740.x
>
> Shah, R. D., & Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 75(1), 55–80. https://doi.org/10.1111/j.1467-9868.2011.01034.x

## `settings` Modes (`"default"` vs `"R_default"`)

All primary pipeline functions and classes (`bin_genomic_changes`, `pfm_to_pwm`, `scan_sequences_pwm`, `calc_sequence_composition`, `calc_binned_motif_enrichment`, `rand_lasso_stab_sel`, and `MonaLisa`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`fmicompbio/monaLisa` v1.12 Parity) |
| :--- | :--- | :--- |
| **Sequence Composition (`calc_sequence_composition`)** | Exact dinucleotide-to-mononucleotide probability ratio $\frac{n_{\text{CG}}/N_2}{(n_{\text{C}}/N_1)(n_{\text{G}}/N_1)}$ for `cpg_oe`. | Sequence-width Gardiner-Garden & Frommer ratio $\frac{n_{\text{CG}} \cdot N_1}{n_{\text{C}} \cdot n_{\text{G}}}$ matching `monaLisa` CpG O/E. |
| **Background Weighting (`calc_binned_motif_enrichment`)** | 10-bin joint `(GC, CpG_OE)` Iterative Proportional Fitting (raking, `max_iter=20`) + Kish effective background sample size. | 8-bin GC (`[0, .2, .3, ..., .8, 1]`) + 16-dinucleotide iterative k-mer rescaling (`maxIter=20`), zeroing unpopulated GC bins. |
| **Shrunk $\log_2$ Enrichment (`pseudocount_log2enr`)** | `pseudocount_log2enr=1.0`: $\log_2\!\left(\frac{\text{fg} + 1}{\text{bg\_weighted} + 1}\right)$. | `pseudocount_log2enr=8.0`: $\log_2\!\left(\frac{\text{fg} + 8}{\text{bg\_weighted} + 8}\right)$ matching `calcBinnedMotifEnrR`. |
| **Randomized Lasso (`rand_lasso_stab_sel`)** | `weakness=0.5`, `cutoff=0.8`, complementary-pairs ($2B$ half-samples) with Shah–Samworth `PFER` path calibration. | `weakness=0.8`, `cutoff=0.8`, matching upstream `monaLisa::randLassoStabSel` feature penalty rescaling. |

## Quickstart

```python
from monalisa import (
    MonaLisa,
    bin_genomic_changes,
    calc_binned_motif_enrichment,
    pfm_to_pwm,
    rand_lasso_stab_sel,
    scan_sequences_pwm,
)

# 1. Convert JASPAR-style PFMs to log2-odds PWMs and bin genomic effect sizes
pwms = pfm_to_pwm(pfm_dict, pseudocount=1.0, settings="default")
bins = bin_genomic_changes(delta_meth, method="equalN", n_bins=5, min_abs_val=0.05)

# 2. Composition-corrected binned motif enrichment (default & R_default parity)
enr = calc_binned_motif_enrichment(sequences, bins=bins, pwms=pwms, settings="default")
print(enr.top_motifs(bin_name=bins.categories[-1], n=5))

enr_r = calc_binned_motif_enrichment(sequences, bins=bins, pwms=pwms, settings="R_default")

# 3. Complementary-Pairs Randomized Lasso Stability Selection with GC/CpG bias control
scan = scan_sequences_pwm(sequences, pwms, min_score="80%", both_strands=True)
stab = rand_lasso_stab_sel(
    scan.counts,
    y_change=delta_meth,
    sequences=sequences,
    include_seq_bias=True,
    cutoff=0.8,
    pfer=1.0,
    settings="R_default",
)
print("Selected causal motifs:", stab.selected_motifs)
```

## License

MIT (clean-room implementation from published mathematical equations in Machlab et al. 2022, Meinshausen & Bühlmann 2010, and Shah & Samworth 2013).
