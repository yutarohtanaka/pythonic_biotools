# `mrapss`: Mendelian Randomization Accounting for Pleiotropy and Sample Structure

Clean-room Python implementation of **MR-APSS** (Hu et al., *Nature Communications* 2022) for two-sample Mendelian randomization using a foreground-background bivariate Gaussian mixture model with exact instrument-selection truncation adjustment ($|Z_{X,j}| \ge t_0$) and bivariate LD-score background covariance estimation.

## Citation

> Hu, X., Zhao, J., Lin, Z., Wang, Y., Peng, H., Zhao, H., Wan, X., & Yang, C. (2022). Mendelian randomization for causal inference accounting for pleiotropy and sample structure using genome-wide summary statistics. *Nature Communications*, 13, 3098. https://doi.org/10.1038/s41467-022-30858-w

- Upstream R reference: `YangLabHKUST/MR-APSS` (`MRAPSS.R`, `MRAPSS_EM_funcs.R`, `est_paras.R`).

## Statistical Model

1. **Genome-Wide Background Estimation via Bivariate LDSC (`estimate_background_ldsc`)**:
   Across genome-wide $Z$-scores $(Z_{X,j}, Z_{Y,j})$ and LD scores $\ell_j$, MR-APSS estimates the polygenic background covariance matrix $\Omega = \begin{pmatrix} \omega_X & \omega_{XY} \\ \omega_{XY} & \omega_Y \end{pmatrix}$ (capturing polygenic correlated pleiotropy) and intercept matrix $C = \begin{pmatrix} c_X & c_{XY} \\ c_{XY} & c_Y \end{pmatrix}$ (capturing population stratification, cryptic relatedness, and cohort overlap $\rho_e = c_{XY}/\sqrt{c_X c_Y}$):
   $$\operatorname{Cov}(Z_{X,j}, Z_{Y,j} \mid Z_j = 0) = V_{0,j} = \Omega \ell_j + C.$$

2. **Selection-Truncated Foreground-Background Mixture (`fit_mrapss`)**:
   For independent candidate instruments passing $|Z_{X,j}| \ge t_0$ (where $t_0 = \Phi^{-1}(1 - p_{\text{iv}}/2)$) with standard-error ratio $c_j = s_{X,j} / s_{Y,j}$, the foreground covariance under causal effect $\beta$, foreground instrument variance $\sigma_X^2$, and foreground uncorrelated pleiotropy variance $\tau_Y^2$ is:
   $$V_{1,j} = V_{0,j} + \ell_j \begin{pmatrix} \sigma_X^2 & \beta c_j \sigma_X^2 \\ \beta c_j \sigma_X^2 & \beta^2 c_j^2 \sigma_X^2 + \tau_Y^2 \end{pmatrix}.$$
   Conditioning each component density on the selection event $|Z_{X,j}| \ge t_0$ eliminates Winner's Curse bias:
   $$\ell(\beta, \pi_0, \sigma_X^2, \tau_Y^2) = \sum_{j=1}^M \log \left[ \pi_0 \frac{\mathcal{N}(z_j; 0, V_{1,j})}{2\Phi(-t_0 / \sqrt{V_{1,j,11}})} + (1 - \pi_0) \frac{\mathcal{N}(z_j; 0, V_{0,j})}{2\Phi(-t_0 / \sqrt{V_{0,j,11}})} \right].$$

3. **Likelihood Ratio Test (LRT)**:
   Hypothesis testing for $H_0: \beta = 0$ compares the full model against the profiled null model ($\beta = 0$ fixed) via $\Lambda = 2(\ell_{\text{full}} - \ell_{\text{null}}) \sim \chi^2_1$.

## Python Quickstart

```python
import numpy as np
from mrapss import estimate_background_ldsc, fit_mrapss

rng = np.random.default_rng(31)
m = 2000
ld_scores = rng.uniform(1.0, 30.0, m)
xse = np.full(m, 0.02)
yse = np.full(m, 0.02)
fg = np.zeros(m)
fg[:80] = rng.normal(0.0, 0.15, 80)
xbeta = fg + rng.normal(0.0, xse)
ybeta = 0.35 * fg + rng.normal(0.0, yse)

# Pythonic default mode (MAP-regularized multi-start + Huber-White sandwich SE)
res_def = fit_mrapss(xbeta, xse, ybeta, yse, ldscore=ld_scores, iv_p_thresh=1e-4)
print(res_def["b"], res_def["b_se"], res_def["pval"], res_def["p_lrt"])

# Exact upstream R parity mode (2-step 3-iter IRWLS LDSC + unpenalized EM/MLE + LRT SE)
res_r = fit_mrapss(xbeta, xse, ybeta, yse, ldscore=ld_scores, iv_p_thresh=1e-4, settings="R_default")
print(res_r["b"], res_r["b_se"], res_r["pval"])
```

## `settings="default"` vs `settings="R_default"`

Both `estimate_background_ldsc` and `fit_mrapss` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values):

| Feature | `settings="default"` | `settings="R_default"` (`YangLabHKUST/MR-APSS`) |
| :--- | :--- | :--- |
| **Bivariate LDSC Weights** | Single-pass $1/\ell_j$ WLS | 2-step **IRWLS** across 3 iterations with Isserlis heteroskedastic weights $1 / [\max(\ell_j, 1)(a_j b_j + c_j^2)]$ (`est_paras.R`) |
| **LDSC Intercept Clipping** | Clips $c_X, c_Y$ to `[0.8, 1.5]` | Unclipped positive intercepts (`c_X, c_Y > 0`) |
| **Mixture Objective** | Weakly informative MAP prior on $\pi_0, \log\sigma_X^2, \log\tau_Y^2$ | **Unregularized** marginal mixture log-likelihood (zero MAP penalty) via EM + multi-start optimization |
| **Convergence Tolerance** | `tol=1e-6` | Tight `tol=1e-8` |
| **Primary `b_se` & `pval`** | Sandwich SE `se_sand` and Wald `pval` (plus `p_lrt`) | `b_se = abs(b) / sqrt(max(lrt_stat, 1e-12))` (Hessian fallback if `b == 0`) and `pval = p_lrt` (`MRAPSS.R` L77–78) |

## Output Schema

### `estimate_background_ldsc(...) -> dict`
- `C` (`np.ndarray`, shape `(2, 2)`): Bivariate LDSC intercept matrix.
- `Omega` (`np.ndarray`, shape `(2, 2)`): Per-unit-LD-score genetic covariance matrix.
- `rho_e` (`float`): Estimated cohort-overlap / error correlation $c_{XY} / \sqrt{c_X c_Y}$.

### `fit_mrapss(...) -> dict`
- `b` (`float`): Estimated causal effect $\hat{\beta}$.
- `b_se` (`float`): Standard error of $\hat{\beta}$.
- `pval`, `p` (`float`): Primary $p$-value (`p_lrt` in `R_default`, sandwich Wald $p$-value in `default`).
- `p_lrt` (`float`), `lrt_stat` (`float`): Profile likelihood-ratio test $p$-value and $\chi^2_1$ statistic.
- `b_ci` (`list[float]`): 95% confidence interval `[lower, upper]`.
- `n_snps_used` (`int`): Number of candidate instruments passing `iv_p_thresh`.
- `pi0` (`float`): Foreground mixture weight among selected instruments.
- `sigma_x2` (`float`), `tau_y2` (`float`): Estimated foreground exposure and pleiotropy variances.
- `rho_e` (`float`), `C` (`list[list[float]]`), `Omega` (`list[list[float]]`): Background LDSC parameters.
- `converged` (`bool`): Optimizer convergence flag.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mrapss \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mrapss/tests -v
```
