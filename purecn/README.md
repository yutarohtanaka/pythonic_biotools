# `purecn`: Tumor-Only Purity/Ploidy, Somatic Classification, LOH, CCF & TMB

Clean-room Python implementation of **PureCN** (Riester et al., *Source Code for Biology and Medicine* 2016) for tumor-only clinical capture panel copy-number normalization, joint 2D purity/ploidy optimization, Beta-Binomial somatic-vs-germline SNV classification, allele-specific loss of heterozygosity (LOH), cancer cell fraction (CCF) clonality, chromosomal instability (CIN), and artifact-cleaned tumor mutational burden (TMB).

## Citation

> Riester, M., Singh, A. P., Brannon, A. R., Yu, K., Campbell, C. D., Chiang, D. Y., & Morrissey, M. P. (2016). PureCN: copy number calling and SNV classification using targeted short read sequencing. *Source Code for Biology and Medicine*, 11(1), 13. https://doi.org/10.1186/s13029-016-0060-z

## `settings` Modes (`"default"` vs `"R_default"`)

Every primary function and class (`normalize_tumor_only_coverage`, `grid_search_purity_ploidy`, `predict_somatic`, `call_tmb`, `call_cin`, `run_purecn`, and `PureCN`) accepts `settings: Literal["default", "R_default"] = "default"` and raises `ValueError` on unknown values:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`lima1/PureCN` v2.16 Parity) |
| :--- | :--- | :--- |
| **Coverage & PoN Tangent (`normalize_tumor_only_coverage`)** | Haldane $+0.5$ median-scaled $\log_2$ depth, robust tricube LOWESS (`frac=0.30`), SVD Tangent PoN subtraction, `cbs_undo_sd=0.75`. | Mean-floored $\log_2$ coverage, LOWESS (`frac=0.25`), SVD Tangent PoN subtraction, `cbs_undo_sd=1.00`. |
| **2D Purity/Ploidy Grid (`grid_search_purity_ploidy`)** | `purity` $\in [0.10, 0.99]$ (`0.01`), `ploidy` $\in [1.50, 6.00]$ (`0.05`), smooth subclonal mixture likelihood. | `test.purity = seq(0.15, 0.95, 0.01)`, `ploidy` $\in [1.40, 6.00]$ (`0.05`), `max.non.clonal = 0.20` subclonal genome penalty/gate. |
| **Allelic Overdispersion (`predict_somatic`)** | Fixed Beta-Binomial `rho_overdisp = 0.005` unless overridden. | Empirical Beta-Binomial MLE overdispersion estimated from germline heterozygous SNPs around $p = 0.5$ (`min = 1e-4`). |
| **dbSNP / POPAF / COSMIC Priors (`predict_somatic`)** | Novel (`pop_af < 1e-4`): `0.5`; rare/dbSNP: `0.05`; common (`pop_af >= 1e-3`): `0.001`; COSMIC ($\ge 2$): `0.98`; somatic call `posterior_somatic >= 0.5`. | Novel: `0.5`; dbSNP: `0.1`; COSMIC ($\ge 2$): `0.999`; common (`pop_af > 0.001`): `0.0005`; ML call `posterior_somatic >= 0.5`. |
| **Tumor Mutational Burden (`call_tmb`)** | `posterior_somatic >= 0.5`, `af >= min_af`, `pop_af <= max_pop_af`, hotspot screen (`prior_somatic <= 0.95`, `cosmic_cnt < 2`). | `PureCN::callTMB` parity: requires `posterior_somatic >= 0.8` and `prior_somatic <= 0.9` when `exclude_hotspot=True`. |

## Quickstart

```python
from purecn import (
    PureCN,
    call_cin,
    call_tmb,
    grid_search_purity_ploidy,
    normalize_tumor_only_coverage,
    predict_somatic,
    run_purecn,
)

# Step-by-step tumor-only panel normalization, purity/ploidy fit, and SNV classification
cov_res = normalize_tumor_only_coverage(
    tumor_cov_df, normal_panel=pon_df, gc_correct=True, tangent_norm=True, settings="default"
)
solutions = grid_search_purity_ploidy(cov_res.segments, variants_df=variants_df, settings="default")
best = solutions[0]

var_calls = predict_somatic(
    variants_df, best.segments, purity=best.purity, ploidy=best.ploidy, settings="default"
)
tmb = call_tmb(var_calls, callable_mb=1.5, exclude_hotspot=True, settings="default")
cin = call_cin(best.segments, purity=best.purity, ploidy=best.ploidy, settings="default")

# End-to-end PureCN pipeline (default or R_default parity mode)
res = PureCN(settings="R_default").run(
    tumor_cov_df, variants_df=variants_df, normal_panel=pon_df, callable_mb=1.5
)
print(res.summary())
print(res.to_dataframe().head())
```

## License

MIT (clean-room implementation from published mathematical equations in Riester et al. 2016).
