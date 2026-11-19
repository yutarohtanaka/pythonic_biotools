# `dss`: Beta-Binomial Empirical Bayes Differential DNA Methylation & DMRcate

Standalone Python implementation of **DSS** (Wu et al., *NAR* 2013; Feng et al., *NAR* 2014; Park & Wu, *Bioinformatics* 2016; `haowulab/DSS` v2.58) for Beta-Binomial Empirical Bayes bisulfite and EM-seq differential DNA methylation (`DML` and `DMR` calling, plus multifactor `DSS-general` GLS), bundled with **DMRcate** Gaussian-kernel Satterthwaite $\chi^2$ DMR detection (Peters et al., *Epigenetics & Chromatin* 2015).

## Citations

> Wu, H., Wang, C., & Wu, Z. (2013). A new shrinkage estimator for dispersion improves differential expression detection in RNA-seq data. *Nucleic Acids Research*, 41(3), e15. https://doi.org/10.1093/nar/gks1148
>
> Feng, H., Conneely, K. N., & Wu, H. (2014). A Bayesian hierarchical model to detect differentially methylated loci from single nucleotide resolution sequencing data. *Nucleic Acids Research*, 42(8), e69. https://doi.org/10.1093/nar/gku154
>
> Park, Y., & Wu, H. (2016). Differential methylation analysis for BS-seq data under general experimental design. *Bioinformatics*, 32(10), 1446–1453. https://doi.org/10.1093/bioinformatics/btw026
>
> Peters, T. J., Buckley, M. J., Statham, A. L., Pidsley, R., Samaras, K., Lord, R. V., Clark, S. J., & Molloy, P. L. (2015). De novo identification of differentially methylated regions in the human genome. *Epigenetics & Chromatin*, 8, 6. https://doi.org/10.1186/1756-8935-8-6

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`dml_test`, `call_dml`, `call_dmr`, `dml_fit_multifactor`, `dml_test_multifactor`, `dmrcate`, and `DSSConfig`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`DSS` v2.58 & `DMRcate` Parity) |
| :--- | :--- | :--- |
| **BSmooth Tricube Smoothing (`dml_test`)** | `smoothing=True` (`smoothing_span=500` bp) by default when `smoothing=None`. | `smoothing=False` by default in `dml_test` unless explicitly passed as `True`. |
| **Dispersion Shrinkage (`shrink_dispersion_bb`)** | Log-normal Empirical Bayes Beta-Binomial MAP shrinkage clamped to `[1e-5, 0.99]`. | Identical log-normal Empirical Bayes Beta-Binomial MAP shrinkage in `[1e-5, 0.99]`. |
| **DML Calling (`call_dml`)** | `delta=0.1`, `p_threshold=1e-5`, composite tail probability $\Phi(-\frac{\|d\|-\delta}{s}) + \Phi(-\frac{\|d\|+\delta}{s})$. | Exact `DSS::callDML` `delta=0.1`, `p_threshold=1e-5`, and `postprob.overThreshold`. |
| **DMR Calling (`call_dmr`)** | `p_threshold=1e-4`, `minlen=50`, `minCG=3`, `dis_merge=100` (preserved without clamping), `pct_sig=0.5`. | `p_threshold=1e-5`, `minlen=50`, `minCG=3`, `dis_merge` clamped to `minlen` (`50`) whenever `dis_merge > minlen` matching `DSS::callDMR`, `pct_sig=0.5`. |
| **Kernel DMRs (`dmrcate`)** | Gaussian smoothing ($\sigma = \lambda_{\text{bp}} / C$) of $t_i^2$ with Satterthwaite `(K1, K2, K3)` $\chi^2$ df & BH-FDR `< 0.05`. | Identical Peters et al. (2015) `DMRcate` Satterthwaite $\chi^2$ kernel aggregation. |

## Quickstart

```python
from dss import (
    call_dml,
    call_dmr,
    dml_fit_multifactor,
    dml_test,
    dml_test_multifactor,
    dmrcate,
)

# 1. Two-group Beta-Binomial DML & DMR calling (with BSmooth tricube smoothing)
dml_res = dml_test(ctrl_frames, tumor_frames, settings="default")
dmls = call_dml(dml_res, delta=0.1, p_threshold=1e-5, settings="default")
dmrs = call_dmr(dml_res, settings="default")

# 2. Bundled DMRcate Gaussian-kernel Satterthwaite chi^2 DMR detection
dmrs_kernel = dmrcate(dml_res, lambda_bp=1000, C=2.0, fdr=0.05, settings="default")

# 3. Park & Wu (2016) DSS-general multifactor / paired / continuous covariate GLS
mf_fit = dml_fit_multifactor(N_mat, X_mat, coords_df, design=design_df, settings="R_default")
mf_res = dml_test_multifactor(mf_fit, coef="treatment_drug", settings="R_default")
```

## License

MIT (clean-room implementation from published mathematical equations in Wu et al. 2013, Feng et al. 2014, Park & Wu 2016, and Peters et al. 2015).
