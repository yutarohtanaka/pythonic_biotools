# `limma`: Empirical Bayes Moderated Linear Models & $t$-Tests

Standalone Python implementation (`pythonic-limma`) of Bioconductor **`limma`** empirical Bayes variance moderation, supporting two-group differential testing (`moderated_ttest`), general linear model contrasts (`moderated_lm_contrast`), mean–variance trend priors (`limma_trend`), Phipson et al. hypervariable-outlier robustness (`robust=True`), and Smyth's empirical Bayes $B$ log-posterior-odds statistic (`b_stat` / `lods`).

---

## Published Citations

- **Smyth, G. K.** (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1), Article 3. https://doi.org/10.2202/1544-6115.1027
- **Law, C. W., Chen, Y., Shi, W., & Smyth, G. K.** (2014). voom: precision weights unlock linear model analysis tools for RNA-seq read counts. *Genome Biology*, 15(2), R29. https://doi.org/10.1186/gb-2014-15-2-r29
- **Ritchie, M. E., Phipson, B., Wu, D., Hu, Y., Law, C. W., Shi, W., & Smyth, G. K.** (2015). limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Research*, 43(7), e47. https://doi.org/10.1093/nar/gkv007
- **Phipson, B., Lee, S., Majewski, I. J., Alexander, W. S., & Smyth, G. K.** (2016). Robust hyperparameter estimation protects against hypervariable genes and improves power to detect differential expression. *Annals of Applied Statistics*, 10(2), 946–963. https://doi.org/10.1214/16-AOAS920

---

## Statistical Formulation

For each feature $g \in \{1, \dots, G\}$ with residual variance $s_g^2$ on $d_g$ residual degrees of freedom ($d_g s_g^2 \mid \sigma_g^2 \sim \sigma_g^2 \chi^2_{d_g}$), `limma` places a conjugate scaled-inverse-$\chi^2$ prior across features:

$$\frac{1}{\sigma_g^2} \sim \frac{1}{d_0 s_{0,g}^2} \chi^2_{d_0}$$

1. **Prior Hyperparameter Estimation (`fit_f_dist`, `trigamma_inverse`)**:
   Under the marginal $F(d_g, d_0)$ distribution of $s_g^2 / s_{0,g}^2$, the bias-corrected log-variance $e_g = \log(s_g^2) - \psi(d_g / 2) + \log(d_g / 2)$ satisfies $\mathbb{E}[e_g] = \log(s_{0,g}^2) - \psi(d_0 / 2) + \log(d_0 / 2)$ and $\text{Var}(e_g) = \psi_1(d_g / 2) + \psi_1(d_0 / 2)$. Subtracting $\overline{\psi_1(d_g / 2)}$ from the empirical variance of $e_g$ (or spline/LOWESS trend residuals) and inverting $\psi_1(d_0 / 2)$ via 4 Newton–Raphson iterations on convex $1/\psi_1(y)$ yields $(s_{0,g}^2, d_0)$.

2. **Posterior Variance Moderation (`squeeze_var`)**:
   $$\tilde{s}_g^2 = \frac{d_0 s_{0,g}^2 + d_g s_g^2}{d_0 + d_g}, \qquad \tilde{t}_g = \frac{\hat{\beta}_g}{\tilde{s}_g \sqrt{u_g}} \sim t_{d_0 + d_g}$$
   where $u_g$ is the unscaled contrast variance ($1/n_1 + 1/n_2$ for two-group comparisons or $c^\top (X^\top X)^+ c$ for general linear models).

3. **Empirical Bayes Log-Odds (`b_stat` / `lods`)**:
   Estimates the prior contrast variance $v_0 \in [\text{stdev\_coef\_lim}_0^2, \text{stdev\_coef\_lim}_1^2]$ from the top $\lceil \frac{p}{2} G \rceil$ order statistics of $|\tilde{t}_g|$ (`limma::tmixture.vector`, default $p = 0.01$) and evaluates Smyth (2004) Eq. (6):
   $$B_g = \log\!\left(\frac{p}{1 - p}\right) + \frac{1}{2}\log(r_g) + \frac{d_{\text{tot}, g} + 1}{2}\left[\log\!\left(1 + \frac{\tilde{t}_g^2}{d_{\text{tot}, g}}\right) - \log\!\left(1 + \frac{r_g \tilde{t}_g^2}{d_{\text{tot}, g}}\right)\right], \quad r_g = \frac{u_g}{u_g + v_0}$$

---

## Quickstart

```python
import numpy as np
import pandas as pd
from limma import limma_top_table, limma_trend, moderated_lm_contrast, moderated_ttest

rng = np.random.default_rng(42)
X = pd.DataFrame(rng.normal(size=(12, 500)), columns=[f"Gene_{i}" for i in range(500)])
group = np.array([False] * 6 + [True] * 6)
X.loc[group, ["Gene_0", "Gene_1", "Gene_2"]] += 3.2

# 1. Standard or Bioconductor R-exact moderated two-group t-test
res = moderated_ttest(X, group, robust=True, settings="R_default")
top = res.to_dataframe().head(10)

# 2. General linear model design + arbitrary contrast vector
design = np.column_stack([np.ones(12), group.astype(float), rng.normal(size=12)])
res_lm = moderated_lm_contrast(X, design, contrast=[0.0, 1.0, 0.0], trend=True, settings="R_default")

# 3. Convenience topTable wrapper
df_trend = limma_top_table(X, group, robust=True, trend=True, settings="R_default")
```

---

## `settings="default"` vs `settings="R_default"` Reference

All primary functions (`fit_f_dist`, `squeeze_var`, `moderated_ttest`, `moderated_lm_contrast`, `limma_trend`, `limma_top_table`) accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Aspect | `settings="default"` | `settings="R_default"` (Bioconductor `limma` v3.68) |
|---|---|---|
| **Near-zero $s_g^2$ handling** | Exact raw positive $\log(s_g^2)$ | Floors positive variances at $10^{-5} \cdot \text{median}(s^2)$ (`fitFDist.R`) |
| **`evar <= 0` ($d_0 = \infty$) prior scale** | $\exp(\bar{e})$ on bias-corrected log scale | Arithmetic $\text{mean}(s^2)$ when untrended (`fitFDist.R`) |
| **`df_total` upper cap** | Uncapped ($\infty$ when $d_0 = \infty$, normal tail) | Capped at $\sum_g d_g$ (`df.pooled` in `ebayes.R`) |
| **`trend=True` mean–variance curve** | Fast quantile-anchored tricube LOWESS (`span=0.5`) | Natural cubic regression spline `ns(AveExpr, df=min(4, max(1, G//10)))` (`fitFDist.R`) |
| **`robust=True` bulk prior** | Asymmetric Winsorized Gaussian moments | Exact 128-node Gauss–Legendre Winsorized $\log F(d_1, d_2)$ quadrature + `brentq` (`fitFDistRobustly.R`) |
| **`robust=True` outlier $d_{0,g}$** | Multiplicative tail-ratio downweighting $w_g d_0$ | Solves $d_{2,\text{outlier}}$ from max upper residual; $d_{0,g} = p_{\text{inlier},g} d_0 + (1 - p_{\text{inlier},g}) d_{2,\text{outlier}}$ (`prob_vals = (n - r + 0.5)/n`) |

---

## Public API & Output Schema

### Functions
- `moderated_ttest(X, group, feature_names=None, robust=False, trend=False, proportion=0.01, stdev_coef_lim=(0.1, 4.0), *, settings="default") -> ModeratedTestResult`
- `moderated_lm_contrast(Y, design, contrast, feature_names=None, robust=False, trend=False, proportion=0.01, stdev_coef_lim=(0.1, 4.0), *, settings="default") -> ModeratedTestResult`
- `limma_trend(X, group, feature_names=None, robust=False, proportion=0.01, stdev_coef_lim=(0.1, 4.0), *, settings="default") -> ModeratedTestResult`
- `limma_top_table(X, group, ..., *, settings="default") -> pd.DataFrame`
- `squeeze_var(var_g, df1, covariate=None, robust=False, winsor_tail_p=(0.05, 0.10), *, settings="default") -> tuple[var_post, s20, df0, df_total]`
- `fit_f_dist(var_g, df1, covariate=None, robust=False, winsor_tail_p=(0.05, 0.10), *, settings="default") -> tuple[s20, df0]`
- `trigamma_inverse(x) -> np.ndarray`

### `ModeratedTestResult` & `.to_dataframe()` Columns
| Attribute | `to_dataframe()` Column | Description |
|---|---|---|
| `log_fc` | `logFC` | Group difference ($\bar{X}_2 - \bar{X}_1$) or linear contrast estimate $c^\top \hat{\beta}_g$ |
| `ave_expr` | `AveExpr` | Mean expression across observed samples for feature $g$ |
| `t_stat` | `t` | Empirical Bayes moderated $t$-statistic $\tilde{t}_g$ |
| `p_value` | `P.Value` | Two-sided moderated $t$ (or normal if $d_{\text{tot}} = \infty$) $p$-value |
| `adj_p_val` | `adj.P.Val` | NaN-preserving Benjamini–Hochberg FDR $q$-value |
| `b_stat` / `lods` | `B` | Smyth (2004) empirical Bayes log-posterior-odds of differential expression |
| `var_post` | `s2_post` | Shrunk posterior variance $\tilde{s}_g^2$ (`var_prior`, `df_prior` stored on result) |

---

## Testing

```bash
PYTHONPATH=limma /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest limma/tests -v
```

## License

MIT.
