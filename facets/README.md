# `facets`: Allele-Specific Somatic Copy Number, Tumor Purity & Ploidy

Standalone Python implementation of **FACETS** (Shen & Seshan, *Nucleic Acids Research* 2016) for allele-specific somatic copy-number analysis, tumor purity, ploidy, and clonal cellular fraction (CCF) estimation from tumor-normal SNP pileups, paired with continuous-BAF **ASCAT** equations and solver (Van Loo et al., *PNAS* 2010).

## Citations

> Shen, R., & Seshan, V. E. (2016). FACETS: allele-specific copy number and clonal heterogeneity analysis tool for high-throughput DNA sequencing. *Nucleic Acids Research*, 44(16), e131. https://doi.org/10.1093/nar/gkw520
>
> Van Loo, P., Nordgard, S. H., Lingjærde, O. C., Russnes, H. G., Rye, I. H., Sun, W., ... & Børresen-Dale, A. L. (2010). Allele-specific copy number analysis of tumors. *Proceedings of the National Academy of Sciences*, 107(39), 16910–16915. https://doi.org/10.1073/pnas.1009843107

## `settings` Modes (`"default"` vs `"R_default"`)

All primary pipeline functions (`preprocess_sample` / `pre_proc_sample`, `proc_sample`, `em_cncf`, `run_facets`, and `run_ascat` / `fit_ascat`, plus `FacetsConfig`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`mskcc/facets` v0.6.2 & `ASCAT` v3.2.0 Parity) |
| :--- | :--- | :--- |
| **Preprocessing (`preprocess_sample`)** | Library-centered Haldane $+0.5$ $\log_2$ depth ratio, `lowess_frac=0.25`, Haldane $+0.5$ `valor`/`lorvar`. | `procreads.R`: `lowess_frac=0.20`, $\log_2(1 + r_T \cdot \text{tscl}) - \log_2(1 + r_N) - \text{gcbias}$; when `unmatched=True`, Tukey $+1/6$ offset + $r_T \ge 50$ het gate. |
| **Bivariate CBS (`proc_sample`)** | Two-stage parametric Hotelling-$T^2$ CBS over `cnlr` and Winsorized $\text{valor}^2 / \text{lorvar}$ with post-CBS pruning. | `cbs2rank.f`: rank-transforms $(\text{cnlr}, |\text{valor}|)$ into centered uniform ranks $(r - \frac{n+1}{2}) / \sqrt{n(n+1)/12}$ before circular bivariate segmentation. |
| **Segment Clustering (`cluster_segments`)** | Agglomerative bivariate Hotelling-$T^2$ segment clustering. | Merges segments with OCN median difference $< 0.04$ and Mann–Whitney U $p > 0.001$ on heterozygous $|\text{valor}|$. |
| **Mixture EM (`em_cncf`)** | Gaussian $\log R$ + exact folded-normal $\log\text{OR}$ likelihood (`max_iter=100`). | `emcncf`: allele log-likelihood surrogate weights $1 / (|x_{2,ij} - \lambda_k| + 10^{-6})$ with `max_iter=10` IRLS moment updates. |
| **ASCAT Solver (`run_ascat`)** | `gamma_ascat=1.0` (NGS default), $\rho \in [0.10, 1.00]$, uniform segment span weights. | `gamma_ascat=0.55`, $\rho \in [0.10, 1.05]$, $0.05$ weight on balanced segments, and 4-tier cascading local-minimum acceptance. |

## Quickstart

```python
from facets import run_facets, preprocess_sample, proc_sample, em_cncf, run_ascat, compare_facets_ascat

# Default Pythonic pipeline
res = run_facets(pileup_df, genome="GRCh38", settings="default")
print(res.summary())
print(res.to_dataframe().head())

# Exact upstream mskcc/facets v0.6.2 & VanLoo-lab/ascat v3.2.0 parity mode
res_r = run_facets(pileup_df, genome="GRCh38", settings="R_default")
ascat_r = run_ascat(res_r, gamma_ascat=1.0, settings="R_default")
print(compare_facets_ascat(res_r, ascat_r))

# Classic 3-panel stacked FACETS genome-wide visualization (optional `pip install "facets[plot]"`)
fig = res.plot(figsize=(11, 7))
fig.savefig("facets_profile.png", dpi=300)
```

## License

MIT (clean-room implementation from published mathematical equations in Shen & Seshan 2016 and Van Loo et al. 2010).
