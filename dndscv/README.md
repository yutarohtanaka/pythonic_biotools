# `dndscv`: Maximum-Likelihood Somatic Selection ($dN/dS$) in Cancer

Clean-room Python implementation of **dNdScv** (Martincorena et al., *Cell* 2017) for quantifying gene-level and exome-wide somatic selection ($w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}}, w_{\text{ind}}$) while accounting for 192 coding-strand trinucleotide substitution rates, mutational sequence composition, and gene-level epigenomic/expression covariates.

## Citation

> Martincorena, I., Raine, K. M., Gerstung, M., Dawson, K. J., Haase, K., Van Loo, P., Davies, H., Stratton, M. R., & Campbell, P. J. (2017). Universal Patterns of Selection in Cancer and Somatic Tissues. *Cell*, 171(5), 1029–1041.e21. https://doi.org/10.1016/j.cell.2017.09.042

## Mathematical & Algorithmic Overview

1. **Coding-Strand 192-Context `RefCDS` Opportunity Matrices ($L_{g, r, c}$)**:
   Constructs per-gene opportunity arrays across all $64 \times 3 = 192$ coding-strand (`5'->3'`) trinucleotide substitutions and four impact classes (`syn`, `mis`, `non`, `spl`) from FASTA + CDS/GTF tables or synthetic sequences. At exon-intron junctions, codon translation uses the spliced CDS while trinucleotide flanking contexts preserve the unspliced genomic DNA bases. Essential splice sites cover the 4 intronic positions (`-2, -1` acceptor and `+1, +2` donor) of internal introns.

2. **192-Parameter Trinucleotide Substitution Rate Poisson MLE (`192r_3w`)**:
   Jointly fits the 192 background substitution rates $\hat{r}_j$ and global selection ratios $(w_{\text{mis}}, w_{\text{non}}, w_{\text{spl}})$ via alternating closed-form coordinate ascent on the globally concave Poisson log-likelihood:
   $$n_{\cdot, j, \text{syn}} \sim \text{Poisson}(r_j L_{\cdot, j, \text{syn}}), \qquad n_{\cdot, j, c} \sim \text{Poisson}(w_c r_j L_{\cdot, j, c})$$
   Global `wtru` (`192r_2w`) and `wall` (`192r_1w`) selection estimates include exact Schur-complement Wald and profile-likelihood confidence intervals.

3. **Negative Binomial Covariate Regression (`dndscv_nb`)**:
   Models unexplained gene-to-gene synonymous variation via Gamma-Poisson (NB2) regression with exposure offset $\log E_{g,\text{syn}}$ and epigenomic/expression covariates $X_g$:
   $$t_g \sim \text{Gamma}\!\left(\theta, \frac{\theta}{\rho_g}\right), \quad \log \rho_g = X_g^\top \beta, \quad n_{g,\text{syn}} \sim \text{NB2}(\mu_g = E_{g,\text{syn}}\rho_g, \theta)$$
   using damped Fisher-scoring IRLS alternating with bounded profile-$\theta$ Newton-Raphson and automatic Poisson fallback as $\theta \to \infty$. A separate NB2/Poisson model with offset $\log(\text{cds\_length}_g)$ is fitted for coding indels.

4. **Gene-Level Conditional LRTs & NaN-Aware BH-FDR**:
   Integrating out the Gamma prior $t_g$ yields closed-form empirical-Bayes selection MLEs $\hat{w}_{c,g}^{\text{CV}}$ and exact likelihood-ratio tests (`p_mis_cv`, `p_non_cv`, `p_spl_cv`, `p_tru_cv`, `p_all_cv`, `p_ind_cv`), combined with Fisher's method (`p_global_cv`) and Benjamini-Hochberg FDR computed strictly over finite $p$-values. Pure Poisson contingency LRTs (`w_p`, `p_*_p`) are also reported.

## Installation

```bash
pip install -e ./dndscv
```

## Quickstart

```python
from dndscv import build_refcds, dndscv, simulate_refcds

refcds = simulate_refcds(n_genes=200, cds_length=900, n_covariates=2, seed=42)
# mutations_df: columns (sampleID, chr, pos, ref, mut) or TCGA MAF columns
res = dndscv(mutations_df, refcds)
print(res.global_dnds_dataframe())
print(res.significant(alpha=0.05))
```

## Exact `im3sanger/dndscv` Parity (`settings="R_default"`)

`dndscv(...)` (and `DnDScvModel`) accepts `settings: Literal["default", "R_default"] = "default"` (strictly validated; raises `ValueError` on unknown values):

| Component | `settings="default"` (Pythonic default) | `settings="R_default"` (`im3sanger/dndscv` `dndscv.R` parity) |
| :--- | :--- | :--- |
| **NB covariate optimizer** | Damped IRLS + bounded Brent/Newton `log(theta)` profile MLE (`max_iter=100`) | `MASS::glm.nb` (`maxit=25`) alternating IRLS + `MASS::theta.ml` moment-initialized Newton `theta` updates |
| **`theta` fallback policy** | Falls back to Poisson when `theta` hits boundary or `ll_nb <= ll_pois` | Strict `dndscv.R` `tryCatch` convergence fallback: any non-convergent outer `glm.nb` or diverging `theta.ml` (`theta >= 1e4` or `theta < 1e-3`) reverts cleanly to Poisson GLM |
| **`to_dataframe()` / `sel_cv` schema** | Rich 42-column Pythonic table (`cds_length`, `exp_*`, `mu_*`, `w_*_cv`, `w_*_p`, etc.) | Exact upstream `sel_cv` column ordering (`gene_name, n_syn, n_mis, n_non, n_spl, wmis_cv, wnon_cv, wspl_cv, wind_cv, pmis_cv, ptrunc_cv, pallsubs_cv, pind_cv, qmis_cv, qtrunc_cv, qallsubs_cv, qind_cv, pglobal_cv, qglobal_cv` when indels active; 14-column SNV table otherwise) |

```python
res_r = dndscv(mutations_df, refcds, settings="R_default")
print(res_r.to_dataframe(sort_by_pvalue=True).head())
```

## License

MIT (clean-room implementation derived from the published mathematical formulations in Martincorena et al., *Cell* 2017).
