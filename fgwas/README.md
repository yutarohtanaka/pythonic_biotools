# `fgwas`: Functional Annotation Enrichment & Joint Bayesian GWAS Fine-Mapping

Standalone Python implementation of **fgwas** (Pickrell, *The American Journal of Human Genetics* 2014) for jointly learning genome-wide or fine-mapping functional annotation enrichments and re-weighting per-SNP Bayesian posterior probabilities of causality (`PPA`).

## Citations

> Pickrell, J. K. (2014). Joint analysis of functional genomic data and genome-wide association studies of 18 human traits. *The American Journal of Human Genetics*, 94(4), 559–573. https://doi.org/10.1016/j.ajhg.2014.03.004
>
> Wakefield, J. (2009). Bayes factors for genome-wide association studies: comparison with P-values. *Genetic Epidemiology*, 33(1), 79–86. https://doi.org/10.1002/gepi.20359

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions (`wakefield_lnabf`, `fit_fgwas`, `cross_validate_fgwas`, and `stepwise_fgwas`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (Upstream C++ `joepickrell/fgwas` v0.3.6 Parity) |
| :--- | :--- | :--- |
| **Wakefield Prior Grid (`V`)** | Mixture prior `V=(0.01, 0.1, 0.5)` across both genome-wide and fine-mapping modes unless overridden. | `V=(0.01, 0.1, 0.5)` genome-wide; switches to single `V=(0.1,)` when `fine_map=True` (matching `fgwas -fine` `-w 0.1`). |
| **Default Ridge Penalty (`penalty`)** | Mild stabilizing L2 penalty `penalty=0.1` when `len(annotations) > 0` to prevent separation blowup. | Unpenalized MLE `penalty=0.0` (`window_size=5000` SNPs/block). |
| **Profile-Likelihood 95% CIs** | Exact asymptotic $\chi^2_1(0.95) / 2 = 1.9207$ log-likelihood drop on each $\gamma_a$. | Exact `2.0` log-likelihood-unit drop (`max_llk - 2.0`, matching C++ `Fgwas_params::get_ci`). |
| **Cross-Validation (`-xv`)** | Seeded randomized block fold assignment across `penalties=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0)`. | Deterministic modulo block split (`p=0.2` default grid start when `penalties=None`). |
| **Stepwise Selection** | Forward addition at $\Delta\ell > 1.9207$ (`penalty=0.05`) followed by CV tuning and backward CV pruning. | Unpenalized forward addition at $\Delta\ell > 2.0$ followed by CV tuning and backward CV pruning. |

## Quickstart

```python
from fgwas import wakefield_lnabf, fit_fgwas, cross_validate_fgwas, stepwise_fgwas

# 1. Compute log Approximate Bayes Factors (lnABF) across mixture prior V
lnabf = wakefield_lnabf(df["beta"], df["se"], V=(0.01, 0.1, 0.5), settings="default")

# 2. Genome-wide hierarchical enrichment & regional/SNP posterior probabilities
fit = fit_fgwas(df, annotations=["dnase_peak", "coding"], window_size=5000, settings="default")
print(fit.summary())
print(fit.to_dataframe())

# 3. Upstream C++ joepickrell/fgwas v0.3.6 fine-mapping parity mode (fgwas -fine)
fit_fine = fit_fgwas(df_loci, annotations=["dnase_peak"], block_col="SEGNUMBER", fine_map=True, settings="R_default")
snp_posteriors = fit_fine.snp_dataframe(df_loci)
region_posteriors = fit_fine.region_dataframe()

# 4. Cross-validated ridge penalty tuning (-xv) & multi-annotation stepwise selection
cv_res = cross_validate_fgwas(df, annotations=["dnase_peak", "coding"], n_folds=10, settings="default")
step_res = stepwise_fgwas(df, candidate_annotations=["dnase_peak", "coding", "intron_noise"], settings="default")
print(step_res.selected_annotations, step_res.best_penalty)
```

## Key Exports

- `wakefield_lnabf`: Overflow-safe log Approximate Bayes Factor over scalar or grid prior variances `V`.
- `fit_fgwas`, `FgwasResult`: Exact log-sum-exp L-BFGS-B solver for genome-wide (`pi_region`, `gamma`) and fine-mapping (`-fine`) enrichment, profile-likelihood 95% CIs, AIC, `PPA_region`, and `PPA_snp`.
- `cross_validate_fgwas`, `stepwise_fgwas`: Block cross-validation (`-xv`) for L2 ridge penalty tuning and forward-addition + backward-CV annotation selection.

## License

MIT (clean-room implementation from published mathematical equations in Pickrell, *AJHG* 2014 and Wakefield, *Genet. Epidemiol.* 2009).
