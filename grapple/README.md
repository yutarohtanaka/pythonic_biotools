# `grapple`: Robust Profile-Likelihood MR, Multimodal Pathways & MVMR

Standalone Python implementation of **GRAPPLE** (*Genome-wide mR Analysis under Pervasive PLEiotropy*, Wang et al., *PNAS* 2021) for weak-instrument Mendelian Randomization, **multimodal pleiotropic pathway discovery**, marker SNP identification, and sample-overlap-aware multivariable MR.

## Citations

> Wang, J., Zhao, Q., Bowden, J., Hemani, G., Davey Smith, G., Small, D. S., & Zhang, N. R. (2021). Causal inference for heritable phenotypic risk factors using heterogeneous genetic instruments. *Proceedings of the National Academy of Sciences*, 118(15), e2015061118. https://doi.org/10.1073/pnas.2015061118
>
> Zhao, Q., Wang, J., Hemani, G., Bowden, J., & Small, D. S. (2020). Statistical inference in two-sample summary-data Mendelian randomization using robust adjusted profile score. *The Annals of Statistics*, 48(3), 1742–1769. https://doi.org/10.1214/19-AOS1866

## Robust Adjusted Profile Log-Likelihood

Given exposure summary statistics $(\hat{\gamma}_{X,j}, \sigma_{X,j})$, outcome summary statistics $(\hat{\gamma}_{Y,j}, \sigma_{Y,j})$, and null-SNP error correlation $\rho_{XY}$, GRAPPLE evaluates the orthogonalized robust profile log-likelihood across weak and strong instruments:

$$t_j(\beta) = \frac{\hat{\gamma}_{Y,j} - \beta \hat{\gamma}_{X,j}}{\sqrt{\sigma_{Y,j}^2 + \beta^2 \sigma_{X,j}^2 - 2\beta \rho_{XY}\sigma_{X,j}\sigma_{Y,j}}}, \qquad \ell_{\text{prof}}(\beta) = -\sum_{j=1}^M \rho_k(t_j(\beta))$$

Under bounded Tukey biweight loss $\rho_k$, distinct causal or pleiotropic pathways appear as separate local modes of $\ell_{\text{prof}}(\beta)$, refined via Brent score root-solving $\Psi(\beta) = 0$ with Stein-orthogonalized sandwich standard errors.

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`estimate_cor_mat`, `select_grapple_instruments`, `grapple_robust_est`, `grapple_mvmr`, and `GRAPPLE`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`jingshuw/GRAPPLE` v0.2 Parity) |
| :--- | :--- | :--- |
| **Tukey Tuning Constant $k$** | `loss_function="tukey"`, `k = 4.685` ($95\%$ asymptotic Gaussian efficiency). | `loss_function="tukey"`, `k = 3.0` (`GRAPPLE` tighter biweight for sharper multimodal separation). |
| **Mode Valley Prominence** | Valley persistence filter $\Delta \ell_{\text{prof}} \ge 0.5$ log-likelihood units. | `supp_drop >= 1.0` ($2 \Delta \ell_{\text{prof}} \ge 2.0$ chi-square prominence rule) when `diagnosis=True`. |
| **Error Correlation (`estimate_cor_mat`)** | Pairwise Plackett truncated-box bivariate normal MLE + PSD correlation projection. | Joint null SNP filter ($p > p_{\text{thres}}$ across all traits) + truncated variance adjustment. |
| **Weak-IV Selection** | `p_thres = 1e-4`, greedy LD pruning at `r2_prune = 0.001`. | `p_thres = 1e-4`, Bonferroni multi-exposure selection matching `GRAPPLE::getInput`. |

## Quickstart

```python
from grapple import GRAPPLE, estimate_cor_mat, grapple_mvmr, grapple_robust_est, select_grapple_instruments

# 1. Estimate sample-overlap correlation from null SNPs and select weak+strong IVs
cor_mat = estimate_cor_mat(z_null_df, p_thres=0.5, settings="default")
ivs = select_grapple_instruments(sumstats_df, ld_R=ld_R, p_thres=1e-4, settings="default")

# 2. Univariable multimodal pathway diagnosis (default vs upstream R_default k=3.0)
res = grapple_robust_est(ivs["bx"], ivs["sx"], ivs["by"], ivs["sy"], cor_xy=cor_mat[0, 1], settings="default")
res_r = grapple_robust_est(ivs["bx"], ivs["sx"], ivs["by"], ivs["sy"], cor_xy=cor_mat[0, 1], settings="R_default")
print(res_r.summary())
print(res_r.marker_dataframe().head())

# 3. Sample-overlap-aware Multivariable MR across P exposures
mv_res = grapple_mvmr(Bx, Sx, by, sy, cor_mat=cor_mat_mv, settings="R_default")
print(mv_res.to_dataframe())

# 4. Multimodal profile log-likelihood curve with mode 95% CIs (`pip install "grapple[plot]"`)
fig = res_r.plot(figsize=(8, 4.5))
fig.savefig("grapple_profile.png", dpi=300)
```

## License

MIT (clean-room implementation from published mathematical equations in Wang et al. 2021).
