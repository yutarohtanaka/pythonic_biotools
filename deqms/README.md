# `deqms`: Peptide/PSM-Count-Dependent Empirical Bayes for Quantitative Proteomics

Standalone Python implementation of **DEqMS** (`yafeng/DEqMS` v1.22; Zhu et al., *Molecular & Cellular Proteomics* 2020) for peptide- and PSM-count-dependent Empirical Bayes differential protein expression across isobaric TMT/iTRAQ, DIA, and label-free (LFQ) mass-spectrometry proteomics, paired with standard **limma** moderated $t$-statistics (Smyth, *SAGMB* 2004; Phipson et al., *Annals of Applied Statistics* 2016) for direct side-by-side comparison.

## Citations

> Zhu, Y., Orre, L. M., Zhou Tran, Y., Mermelekas, G., Johansson, H. J., Malyutina, A., Anders, S., & Lehtiö, J. (2020). DEqMS: A Method for Accurate Variance Estimation in Differential Protein Expression Analysis. *Molecular & Cellular Proteomics*, 19(6), 1047–1057. https://doi.org/10.1074/mcp.TIR119.001646
>
> Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1), Article 3. https://doi.org/10.2202/1544-6115.1027
>
> Phipson, B., Lee, S., Majewski, I. J., Alexander, W. S., & Smyth, G. K. (2016). Robust hyperparameter estimation protects against hypervariable genes and improves power to detect differential expression. *The Annals of Applied Statistics*, 10(2), 946–963. https://doi.org/10.1214/16-AOAS920

## `settings` Modes (`"default"` vs `"R_default"`)

All primary pipeline functions and classes (`equal_median_normalization`, `median_sweeping`, `peptide_profile_summarization`, `lm_fit`, `spectra_count_ebayes`, `output_result`, and `DEqMS`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`yafeng/DEqMS` v1.22 Bioconductor Parity) |
| :--- | :--- | :--- |
| **Column Normalization (`equal_median_normalization`)** | Sample-column median centering to `0.0` (optional global median preservation via `center_zero=False`). | Exact `DEqMS::equalMedianNormalization` sample-column median subtraction to `0.0`. |
| **PSM Sweeping (`median_sweeping`)** | Spectrum-row reference/median subtraction, per-protein median summarization + quantified PSM count, and column equalization. | `DEqMS::medianSweeping`: row median/reference channel subtraction, protein median aggregation, and column median centering. |
| **LFQ/DIA Summarization (`peptide_profile_summarization`)** | Tight-tolerance Tukey two-way median polish (`eps=1e-4`, `max_iter=50`) preserving `NaN` for completely unobserved sample-proteins. | `stats::medpolish` defaults (`eps=0.01`, `max_iter=10`, `na.rm=TRUE`) matching `DEqMS::peptideProfileSummarization`. |
| **Variance-vs-Count Trend (`spectra_count_ebayes`)** | Degree-1 tricube `loess` (`span=0.75`), natural cubic `spline` (`df=4`), or `nls` ($a + b / c_g$) fitted on proteins with residual $d_g \ge 2$. | Degree-1 tricube `loess` (`span=0.75`), `spline`, or `nls` across all proteins with $d_g \ge 1$ and strict $\min(c_g) \ge 1$. |
| **Hyperparameter Shrinkage (`sca_dfprior`)** | `robust=True`: Winsorized log-$F$ moment estimation + per-protein outlier prior degrees of freedom (`sca_dfprior_vec`). | `robust=False`: non-robust Smyth trigamma moment equation yielding scalar `sca.dfprior` and exact 11-column `outputResult` schema. |

## Quickstart

```python
import numpy as np
import pandas as pd
from deqms import (
    DEqMS,
    equal_median_normalization,
    median_sweeping,
    output_result,
    peptide_profile_summarization,
    spectra_count_ebayes,
)

# 1. Summarize TMT PSM table to protein log2 ratios + PSM counts
prot_res = median_sweeping(
    psm_df,
    group_col="protein",
    sample_cols=["ctrl_1", "ctrl_2", "ctrl_3", "treat_1", "treat_2", "treat_3"],
    log_scaled=False,
    settings="default",
)

# 2. Fit count-dependent Empirical Bayes (default Pythonic robust mode)
design = ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat"]
fit = spectra_count_ebayes(prot_res, design=design, settings="default")
table = output_result(fit, sort_by="sca.P.Value")
print(table[["logFC", "count", "P.Value", "sca.P.Value", "sca.adj.pval"]].head())

# 3. Exact Bioconductor DEqMS v1.22 parity mode via object-oriented API
model_r = DEqMS(fit_method="loess", span=0.75, settings="R_default")
table_r = model_r.run(prot_res.abundance, design=design, count=prot_res.count)
print(f"Estimated scalar prior df: {model_r.sca_dfprior:.2f}")
```

## License

MIT (clean-room implementation from published mathematical equations in Zhu et al. 2020, Smyth 2004, and Phipson et al. 2016).
