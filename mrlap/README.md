# `mrlap`: Analytical MR Correction for Sample Overlap, Weak IVs & Winner's Curse

Standalone Python implementation of **MRlap** (Mounier & Kutalik, *Genetic Epidemiology* 2023) for analytical Mendelian Randomization bias correction simultaneously accounting for **arbitrary sample overlap**, **weak-instrument attenuation**, and **Winner's curse** from genome-wide summary statistics.

## Citations

> Mounier, N., & Kutalik, Z. (2023). Bias correction for inverse variance weighting Mendelian randomization. *Genetic Epidemiology*, 47(4), 314–331. https://doi.org/10.1002/gepi.22522
>
> Bulik-Sullivan, B., Finucane, H. K., Anttila, V., Gusev, A., Day, F. R., Loh, P. R., ... & Neale, B. M. (2015). An atlas of genetic correlations across human diseases and traits. *Nature Genetics*, 47(11), 1236–1241. https://doi.org/10.1038/ng.3406

## Analytical Bias Model

Under a spike-and-slab exposure genetic architecture $\beta_X \sim \pi_X \mathcal{N}(0, \sigma_X^2) + (1 - \pi_X)\delta_0$ ($\sigma_X^2 = h_X^2 / (M \pi_X)$) and discovery selection $|\hat{z}_X| > T$ ($T = \Phi^{-1}(1 - p_{\text{thresh}}/2)$), the expected observed IVW estimator satisfies:

$$\mathbb{E}[\hat{\alpha}_{\text{obs}}] = \lambda_{\text{attenuation}} \, \alpha_{\text{true}} + c_{XY} \sqrt{\frac{N_X}{N_Y}} (1 - \lambda_{\text{attenuation}}), \qquad \lambda_{\text{attenuation}} = \frac{\mathbb{E}[\beta_X \hat{\beta}_X \mid |\hat{z}_X| > T]}{\mathbb{E}[\hat{\beta}_X^2 \mid |\hat{z}_X| > T]}$$

where $(h_X^2, h_Y^2, r_g, c_{XY})$ are estimated via bivariate LDSC, $\pi_X$ via truncated spike-and-slab MLE on pruned discovery IVs, and $\text{SE}(\hat{\alpha}_{\text{corrected}})$, $\widehat{\operatorname{Cov}}(\hat{\alpha}_{\text{obs}}, \hat{\alpha}_{\text{corrected}})$, and $Z_{\text{diff}}$ via parametric genetic-architecture resampling.

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`mrlap`, `MRlap`, and `estimate_exposure_polygenicity`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`n-mounier/MRlap` v0.0.3 Parity) |
| :--- | :--- | :--- |
| **Conditional Moments** | 32-node Gauss–Hermite quadrature + normal Mills-ratio conditional expectations. | Closed-form truncated bivariate Gaussian conditional moments. |
| **Polygenicity Bounds** | Stabilized interior spike-and-slab optimization $\pi_X \in [10^{-5}, 0.50]$. | Wide `MRlap` v0.0.3 search domain $\pi_X \in [10^{-7}, 0.999]$. |
| **IV Selection & Clumping** | `mr_threshold = 5e-8`, greedy LD clumping at `mr_pruning_ld = 0.001`. | `MR_threshold = 5e-8`, `MR_pruning_LD = 0.001` matching `MRlap::MRlap`. |
| **Architecture Resampling** | `n_sim = 500` parametric draws for `alpha_corrected_se`, `cov_obs_corr`, and $Z_{\text{diff}}$. | `n_sim = 1000` parametric architecture draws and exact difference Z-test $Z_{\text{diff}}$. |

## Quickstart

```python
from mrlap import MRlap, mrlap

# Analytical correction in Pythonic default mode
res = mrlap(exposure_df, outcome_df, ld_scores=ld, mr_threshold=5e-8, settings="default")
print(res.summary())
print(res.to_dataframe())

# Exact upstream n-mounier/MRlap v0.0.3 parity mode
est_r = MRlap(mr_threshold=5e-8, mr_pruning_ld=0.001, settings="R_default")
res_r = est_r.fit(exposure_df, outcome_df, ld_scores=ld)
print(f"Corrected alpha: {res_r.alpha_corrected:.4f}, p_diff: {res_r.p_diff:.3e}")
```

## Output Schema

- **`MRlapResult`**: `alpha_obs`, `alpha_obs_se`, `p_obs`, `alpha_corrected`, `alpha_corrected_se`, `p_corrected`, `cov_obs_corr`, `diff_stat`, `p_diff`, `lambda_attenuation`, `bias_overlap`, `n_instruments`, and `architecture` (`GeneticArchitectureResult`).

## License

MIT (clean-room implementation from published mathematical equations in Mounier & Kutalik 2023).
