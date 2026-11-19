# `mvcml`: Multivariable Constrained Maximum Likelihood MR & Conditional GWAS Bias Adjustment

Standalone Python implementation of **MVMR-cML** (`mvcml_bic`, `cml_ma`, `mvcml_dp`) for Multivariable Mendelian Randomization with Constrained Maximum Likelihood under correlated and uncorrelated pleiotropy, together with **conditional GWAS collider-bias adjustment** (`mvcml_adjust`, `adjusted_variances`).

## Citation

- **MVMR-cML**: Lin, Z., Xue, H., & Pan, W. (2023). Robust multivariable Mendelian randomization based on constrained maximum likelihood. *The American Journal of Human Genetics*, 110(4), 592–605. https://doi.org/10.1016/j.ajhg.2023.02.014
- **Conditional GWAS Adjustment**: Lin, Z., Deng, Y., & Pan, W. (2023). Accounting for collider bias in conditional genome-wide association studies via multivariable constrained maximum likelihood.
- **Upstream R package**: `ZhaotongL/MVMR-cML` (`MVmr_cML`, `MVmr_cML_DP`).

## Statistical Model

For $m$ candidate genetic instruments and $p$ exposures with summary estimates $(\hat{\beta}_{Yi}, \hat{\boldsymbol{\beta}}_{Xi}) \sim \mathcal{N}((\boldsymbol{\beta}_{Xi}^\top \mathbf{b} + r_i, \; \boldsymbol{\beta}_{Xi}^\top), \Sigma_i)$ where $\Sigma_i \in \mathbb{R}^{(p+1)\times(p+1)}$ captures estimation error variances and cross-trait sample-overlap correlations:
$$\min_{\mathbf{b}, \{\boldsymbol{\beta}_{Xi}\}, \{r_i\}} \sum_{i=1}^m \begin{pmatrix} \hat{\beta}_{Yi} - \boldsymbol{\beta}_{Xi}^\top \mathbf{b} - r_i \\ \hat{\boldsymbol{\beta}}_{Xi} - \boldsymbol{\beta}_{Xi} \end{pmatrix}^\top \Sigma_i^{-1} \begin{pmatrix} \hat{\beta}_{Yi} - \boldsymbol{\beta}_{Xi}^\top \mathbf{b} - r_i \\ \hat{\boldsymbol{\beta}}_{Xi} - \boldsymbol{\beta}_{Xi} \end{pmatrix} \quad \text{subject to } \| \mathbf{r} \|_0 \le k.$$
Across candidate invalid-instrument cardinalities $k \in \{0, \dots, m - (p + 1)\}$:
1. **Profile Coordinate Descent**: Alternates closed-form hard-thresholding of the $k$ largest per-SNP profile likelihood gains $r_i$, linear system updates for latent $\boldsymbol{\beta}_{Xi}$, and GLS updates for $\mathbf{b}$.
2. **BIC Selection & Model Averaging (`mvcml_bic`, `cml_ma`)**: Evaluates $\operatorname{BIC}(k) = \ell_k + k \ln N$, selecting $k^* = \arg\min_k \operatorname{BIC}(k)$ (`b_bic`) or forming BIC softmax weights $w_k \propto \exp(-\frac{1}{2}\operatorname{BIC}(k))$ (`b_ma`) with Schur-complement asymptotic covariance $\operatorname{Cov}(\hat{\mathbf{b}})$.
3. **Data Perturbation (`mvcml_dp`)**: Resamples $B = \text{num\_pert}$ perturbed summary datasets $(\hat{\beta}_Y^{(b)}, \hat{\boldsymbol{\beta}}_X^{(b)}) \sim \mathcal{N}((\hat{\beta}_Y, \hat{\boldsymbol{\beta}}_X), \Sigma)$ to account for invalid-IV selection uncertainty.
4. **Conditional GWAS Adjustment (`mvcml_adjust`)**: De-biases collider-adjusted summary statistics $\hat{\beta}_{GY,\text{adj}} = \hat{\beta}_{GY\mid X} - \hat{\boldsymbol{\beta}}_{GX}\hat{\mathbf{b}}$ with analytic delta-method / LD-aware variance propagation (`adjusted_variances`).

## Python Quickstart

```python
import numpy as np
from mvcml import make_corr, mvcml_adjust, mvcml_bic, mvcml_dp

rng = np.random.default_rng(42)
m, p = 25, 2
bx = rng.normal(0.0, 0.14, (m, p))
sx = np.full((m, p), 0.015)
sy = np.full(m, 0.020)
by = bx @ np.array([0.40, -0.25]) + rng.normal(0.0, sy)

# 1. Pythonic default: SIG ordered [Y, X_1, ..., X_p]; mvcml_dp reports model-averaged b_ma_dp
R_def = make_corr(p, rho_yx=[0.15, -0.10], settings="default")
D_def = np.concatenate([sy[:, None], sx], axis=1)
SIG_def = D_def[:, :, None] * R_def[None] * D_def[:, None, :]
res_dp_def = mvcml_dp(N=50_000, betax=bx, betayc=by, SIG=SIG_def, num_pert=100, seed=1, settings="default")

# 2. Upstream ZhaotongL/MVMR-cML parity: SIG ordered [X_1, ..., X_p, Y]; mvcml_dp reports b_bic_dp
R_up = make_corr(p, rho_yx=[0.15, -0.10], settings="R_default")
D_up = np.concatenate([sx, sy[:, None]], axis=1)
SIG_up = D_up[:, :, None] * R_up[None] * D_up[:, None, :]
res_bic_up = mvcml_bic(N=50_000, betax=bx, betayc=by, SIG=SIG_up, settings="R_default")
res_dp_up = mvcml_dp(N=50_000, betax=bx, betayc=by, SIG=SIG_up, num_pert=100, seed=1, settings="R_default")
print("MVMR-cML-DP (R_default):", res_dp_up["b"], res_dp_up["b_se"], res_dp_up["pval"])
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`mvcml_bic`, `mvcml_dp`, `cml_ma`, `mvcml_adjust`, `make_corr`, `adjusted_variances`) accept `settings: Literal["default", "R_default"] = "default"`.

| Feature | `settings="default"` | `settings="R_default"` (`ZhaotongL/MVMR-cML` Parity) |
| :--- | :--- | :--- |
| **`SIG` / `R` variable ordering** | Outcome-first `[Y, X_1, ..., X_p]` | Exposures-first `[X_1, ..., X_p, Y]` (`ZhaotongL/MVMR-cML` convention) |
| **Latent `betaxest` initialization** | Initialized to zeros `np.zeros((m, p))` | Initialized directly to observed exposures `b_exp.copy()` |
| **Convergence criterion (`threshold=None`)** | L1 norm $\|\mathbf{b}^{\text{new}} - \mathbf{b}^{\text{old}}\|_1 < 10^{-6}$ | L2 norm $\|\mathbf{b}^{\text{new}} - \mathbf{b}^{\text{old}}\|_2 < 10^{-4}$ |
| **`mvcml_dp` primary estimate (`b`, `b_se`, `pval`)** | BIC model-averaged `b_ma_dp`, `se_ma_dp`, `pval_ma_dp` | BIC-selected `b_bic_dp`, `se_bic_dp`, `pval_bic_dp` (`MVmr_cML_DP`) |

## Output Schema

- **`mvcml_bic` / `cml_ma`**: Returns a `dict` with `b`, `b_ma`, `b_bic`, `se_ma`, `se_bic`, `pval_ma`, `pval_bic`, `Covb`, `Covb_ma`, `Covb_bic`, `ma_weights`, `btotal`, `bic`, `k_invalid`, `validuse`, `Omega` (`mvcml_bic` sets `b = b_bic`, `b_se = se_bic`, `pval = pval_bic`, `Covb = Covb_bic`).
- **`mvcml_dp`**: Returns a `dict` with `b`, `b_se`, `Covb`, `pval`, `b_bic_dp`, `se_bic_dp`, `pval_bic_dp`, `Covb_bic_dp`, `b_ma_dp`, `se_ma_dp`, `pval_ma_dp`, `Covb_ma_dp`, `pert_b_bic`, `pert_b_ma`, `bic`, `k_invalid`, `validuse`, `Omega`.
- **`mvcml_adjust`**: Returns a `dict` with `b`, `covb`, `b_se`, `ybeta_adj`, `yse_adj_simple`, `yse_adj_ld`, `iv_idx`, `valid_iv_idx`, `fit`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mvcml \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mvcml/tests
```
