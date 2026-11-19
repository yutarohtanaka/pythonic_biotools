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

## License

MIT (clean-room implementation derived from the published mathematical formulations in Martincorena et al., *Cell* 2017).
