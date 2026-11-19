# `mrbee`: Mendelian Randomization with Bias-Corrected Estimating Equations

Clean-room Python implementation of **MRBEE** and **MVMR-BEE** (Lorincz-Comi et al., *American Journal of Human Genetics* 2024) for univariable and multivariable Mendelian randomization using bias-corrected estimating equations, sample-overlap covariance adjustment, Winner's Curse truncation correction, and iterative IMRP horizontal-pleiotropy outlier filtering.

## Citation

> Lorincz-Comi, N. J., Yang, Y., Li, G., & Zhu, X. (2024). MRBEE: A bias-corrected multivariable Mendelian randomization method. *The American Journal of Human Genetics*, 111(2), 366–378. https://doi.org/10.1016/j.ajhg.2024.01.005

- Upstream R reference: `noahlorinczcomi/MRBEE` (`MRBEE.IMRP.R`, `MRBEE.IMRP.UV.R`, `basicfunction.R`, v1.0.0).

## Statistical Model

Given $M$ genetic instruments with $P$ exposure effect estimates $\hat{b}_{X,j} \in \mathbb{R}^P$, outcome effect estimates $\hat{b}_{y,j} \in \mathbb{R}$, and estimation-error covariances $\Sigma_{XX,j} = \operatorname{diag}(s_{X,j}) R_{XX} \operatorname{diag}(s_{X,j})$ and $\sigma_{Xy,j} = s_{y,j} (s_{X,j} \odot r_{Xy})$ derived from the $(P+1)\times(P+1)$ cohort-overlap error correlation matrix $R_{Xy}$:

1. **Bias-Corrected Estimating Equations (BEE)**:
   Ordinary IVW / MVMR-IVW is attenuated toward the null (or toward the observational correlation under cohort overlap) because $\mathbb{E}[\hat{b}_{X,j} \hat{b}_{X,j}^\top] = \beta_{X,j} \beta_{X,j}^\top + \Sigma_{XX,j}$. MRBEE subtracts the measurement-error moments directly on the valid instrument set $V$:
   $$\hat{\theta} = \left( \sum_{j \in V} w_j (\hat{b}_{X,j} \hat{b}_{X,j}^\top - \Sigma_{XX,j}) \right)^{-1} \left( \sum_{j \in V} w_j (\hat{b}_{X,j} \hat{b}_{y,j} - \sigma_{Xy,j}) \right).$$

2. **Iterative Mendelian Randomization Pleiotropy (IMRP) Outlier Detection**:
   Horizontal-pleiotropy outliers ($|\gamma_j| > 0$) are detected iteratively via Sidak family-wise error control across all $M$ candidate instruments ($1 - (1 - p_j)^M < \alpha_{\text{pleio}}$) and excluded from $V$ until the valid mask and $\hat{\theta}$ stabilize.

3. **Sandwich Covariance Estimation**:
   Asymptotic standard errors and $(P \times P)$ covariance matrices are computed from the per-SNP bias-corrected score vectors $\psi_j(\hat{\theta}) = w_j \left[ (\hat{b}_{X,j} \hat{b}_{y,j} - \sigma_{Xy,j}) - (\hat{b}_{X,j} \hat{b}_{X,j}^\top - \Sigma_{XX,j}) \hat{\theta} \right]$ via the Huber-White sandwich estimator $H^{-1} M H^{-\top}$.

## Python Quickstart

```python
import numpy as np
from mrbee import fit_mrbee, fit_mvmr_bee

rng = np.random.default_rng(42)
m = 300
x_true = np.zeros(m)
x_true[:60] = rng.normal(0.0, 0.12, 60)
xse = np.full(m, 0.015)
yse = np.full(m, 0.018)
xbeta = x_true + rng.normal(0.0, xse)
ybeta = -0.35 * x_true + rng.normal(0.0, yse)

# Pythonic default mode (iterative weights + Winner's Curse Mills-ratio + DL tau^2)
res_def = fit_mrbee(xbeta, xse, ybeta, yse, inst_p=1e-3, settings="default")
print(res_def["b"], res_def["b_se"], res_def["n_outliers"])

# Exact upstream R parity mode (MRBEE v1.0.0 MRBEE.IMRP.UV / MRBEE.IMRP)
res_r = fit_mrbee(xbeta, xse, ybeta, yse, inst_p=1e-3, settings="R_default")
print(res_r["b"], res_r["b_se"], res_r["n_outliers"])

# Multivariable MRBEE (P = 2 exposures)
bX = rng.normal(0.0, 0.09, size=(80, 2))
seX = np.full((80, 2), 0.02)
sey = np.full(80, 0.025)
by = bX @ np.array([0.5, -0.3]) + rng.normal(0.0, sey)
mv_res = fit_mvmr_bee(bX, seX, by, sey, settings="R_default")
print(mv_res["b"], mv_res["b_se"])
```

## `settings="default"` vs `settings="R_default"`

Both `fit_mrbee` and `fit_mvmr_bee` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values):

| Feature | `settings="default"` | `settings="R_default"` (`noahlorinczcomi/MRBEE` v1.0.0) |
| :--- | :--- | :--- |
| **Instrument Weighting** | Iterative $\theta$-dependent inverse variance $w_j(\theta) = (\sigma_{y,j}^2 + \theta^\top \Sigma_{XX,j}\theta - 2\sigma_{Xy,j}^\top\theta + \tau^2)^{-1}$ | Up-front pre-standardization by $s_{y,j}$ (`w_j = 1 / sey^2` fixed; `tau2 = 0`) |
| **Winner's Curse Correction (`fit_mrbee`)** | Truncated-normal Mills-ratio `wc_correct=True` by default | `wc_correct=False` by default (`gamma_wc = 1.0`) |
| **Initialization & Norm Bound** | Direct bias-corrected IVW start | Huber M-regression (`RLM` IRLS) + `3 * MAD` residual trimming + $\|\theta\|_2 \le 3\|\theta_{\text{init}}\|_2$ |
| **IMRP Variance Estimator** | Analytic null variance $v_{\text{null},j}(\theta)$ | Empirical residual variance `vare = var(resid_std[valid], ddof=1)` (`var.est="variance"`) |
| **Sandwich Covariance** | Unadjusted valid scores scaled by $n_{\text{valid}} / (n_{\text{valid}} - P)$ | Score residuals divided by leverage $\max(1 - H_{ii}, 0.5)$ and scaled by $n_{\text{total}} / \max(n_{\text{valid}} - P, 1)$ |

## Output Schema

### `fit_mrbee(...) -> dict`
- `b` (`float`): Bias-corrected causal effect estimate $\hat{\beta}$.
- `b_se` (`float`): Sandwich standard error of $\hat{\beta}$.
- `pval` (`float`): Two-sided Wald $p$-value.
- `b_ci` (`list[float]`): 95% normal confidence interval `[lower, upper]`.
- `n_snps_used` (`int`): Number of valid non-outlier instruments retained.
- `n_outliers` (`int`): Number of horizontal-pleiotropy outliers removed by IMRP.
- `rho_e` (`float`): Estimated or supplied estimation-error correlation $\rho_e$.
- `gamma_wc` (`float`): Winner's Curse variance inflation factor (`1.0` when `wc_correct=False`).
- `tau2` (`float`): Residual overdispersion variance (`0.0` in `R_default`).
- `converged` (`bool`): Whether IMRP iterations stabilized within `max_iter`.

### `fit_mvmr_bee(...) -> dict`
- `b` (`np.ndarray`, shape `(P,)`): Direct causal effect vector $\hat{\theta}$.
- `b_se` (`np.ndarray`, shape `(P,)`): Standard errors per exposure.
- `cov` (`np.ndarray`, shape `(P, P)`): Sandwich covariance matrix of $\hat{\theta}$.
- `pval` (`np.ndarray`, shape `(P,)`): Two-sided Wald $p$-values per exposure.
- `b_ci` (`np.ndarray`, shape `(P, 2)`): 95% confidence intervals per exposure.
- `n_snps_used`, `n_outliers` (`int`): Valid and outlier instrument counts.
- `valid_mask` (`np.ndarray`, shape `(M,)`, `bool`): Boolean mask of retained valid instruments.
- `tau2` (`float`): Residual overdispersion variance (`0.0` in `R_default`).
- `converged` (`bool`): Convergence indicator.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mrbee \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mrbee/tests -v
```
