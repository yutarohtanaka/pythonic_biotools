# `saturn`: Quasi-Binomial Empirical Bayes Differential Transcript Usage

Standalone Python implementation of **satuRn** (Gilis et al., *F1000Research* 2022; `statOmics/satuRn` v1.18) for fast, scalable quasi-binomial Empirical Bayes differential transcript, isoform, and exon usage (`DTU`) in bulk and single-cell RNA-seq, paired with two-stage gene/transcript OFDR control (**stageR**, Van den Berge et al., *Genome Biology* 2017).

## Citations

> Gilis, J., Vitting-Seerup, K., Van den Berge, K., & Clement, L. (2022). satuRn: Scalable analysis of differential transcript usage for bulk and single-cell RNA-sequencing applications. *F1000Research*, 10, 374. https://doi.org/10.12688/f1000research.51749.2
>
> Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1), Article 3. https://doi.org/10.2202/1544-6115.1027
>
> Efron, B. (2004). Large-scale simultaneous hypothesis testing: the choice of a null hypothesis. *Journal of the American Statistical Association*, 99(465), 96–104. https://doi.org/10.1198/016214504000000089
>
> Van den Berge, K., Soneson, C., Robinson, M. D., & Clement, L. (2017). stageR: a general stage-wise method for controlling the gene-level false discovery rate in differential expression and differential transcript usage. *Genome Biology*, 18, 151. https://doi.org/10.1186/s13059-017-1277-0

## Statistical Model

For transcript $t$ of parent gene $g(t)$ across active samples/cells $i$ with $n_{g(t), i} = \sum_{t' \in g(t)} y_{t'i} > 0$, `saturn` fits a quasi-binomial logit GLM:

$$\mathbb{E}[y_{ti} \mid n_{gi}] = n_{gi}\pi_{ti}, \quad \text{logit}(\pi_{ti}) = x_i^\top \beta_t, \quad \text{Var}(y_{ti} \mid n_{gi}) = \phi_t n_{gi}\pi_{ti}(1 - \pi_{ti})$$

shrinking quasi-dispersions $\hat{\phi}_t$ via Empirical Bayes moderation (`squeeze_var`) and recalibrating moderated quasi-$t$ statistics against Efron's empirical null $(\hat{\mu}_0, \hat{\sigma}_0)$.

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions (`filter_dtu`, `fit_dtu`, `test_dtu`, `stagewise_dtu`, `squeeze_var`, and `estimate_empirical_null`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`statOmics/satuRn` v1.18 Parity) |
| :--- | :--- | :--- |
| **Quasi-Binomial IRLS (`fit_dtu`)** | Kosmidis & Firth (2021) invariant Jeffreys-prior + `1e-6` ridge IRLS (`max_abs_beta=15.0`) so complete all-0 vs all-1 isoform switches remain finite and testable without Hauck–Donner SE blow-up. | Unpenalized `stats::glm.fit` `quasibinomial` IRLS (`maxit=25, eps=1e-8`), flagging non-converged or boundary `\|beta\| > 15` models. |
| **Variance Moderation (`squeeze_var`)** | Robust Empirical Bayes `fitFDistRobustly` (`robust=True`) with Winsorized log-variance moments and per-outlier prior df reduction. | Non-robust `limma::squeezeVar` (`robust=False`) moment matching over $\ln F(d_t, d_0)$. |
| **Empirical Null (`test_dtu`)** | Central-matching truncated-Gaussian MLE $(\hat{\mu}_0, \hat{\sigma}_0)$ when `n_transcripts >= 50` (graceful `(0.0, 1.0)` fallback on small panels `< 50`). | Efron's Poisson-GLM polynomial histogram `locfdr` central-peak MLE $(\hat{\mu}_0, \hat{\sigma}_0)$ empirical recalibration. |
| **Two-Stage Screening (`stagewise_dtu`)** | Gene-level Simes screening + Shaffer DTU modified Holm confirmation controlling OFDR at `alpha`. | Identical `stageR` (`method="dtu"`) Simes + Shaffer OFDR procedure. |

## Quickstart

```python
from saturn import filter_dtu, fit_dtu, stagewise_dtu, test_dtu

# 1. Filter low-count isoforms and single-isoform genes (AnnData or DataFrames)
filtered = filter_dtu(adata, min_count=10, min_samples=3, settings="default")

# 2. Fit vectorized quasi-binomial logit GLMs across active cells/samples (n_gi > 0)
fit = fit_dtu(filtered, design="condition", settings="default")
print(fit.summary())

# 3. Moderated quasi-t test + Efron empirical null recalibration + stageR OFDR
res = test_dtu(fit, contrast="condition_treated", settings="default")
stage_res = stagewise_dtu(res, alpha=0.05, settings="default")

# 4. Upstream statOmics/satuRn v1.18 parity mode
fit_r = fit_dtu(counts_df, tx2gene=tx2gene_df, design=design_df, settings="R_default")
res_r = test_dtu(fit_r, contrast=[0, 1], settings="R_default")
```

## License

MIT (clean-room implementation from published mathematical equations in Gilis et al. 2022, Smyth 2004, Efron 2004, and Van den Berge et al. 2017).
