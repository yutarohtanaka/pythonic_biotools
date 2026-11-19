# `logistf`: Firth Jeffreys-Prior Penalized Logistic Regression

Standalone Python implementation of CRAN **`logistf`** for small-sample bias-reduced logistic regression, Penalized Likelihood Ratio Tests (**PLRT**), and asymmetric Profile Penalized Likelihood confidence intervals under rare events, complete separation, or quasi-complete separation.

---

## Published Citations

- **Firth, D.** (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38. https://doi.org/10.1093/biomet/80.1.27
- **Heinze, G., & Schemper, M.** (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine*, 21(16), 2409–2419. https://doi.org/10.1002/sim.1047
- **Heinze, G., Ploner, M., Dunkler, D., Southworth, H., & Jiricka, L.** (2023). *logistf: Firth's Bias-Reduced Logistic Regression*. R package on CRAN (`georgheinze/logistf`).

---

## Statistical Formulation

Given design matrix $X \in \mathbb{R}^{n \times p}$ and binary outcome $y \in \{0, 1\}^n$ with $\pi_i(\beta) = \text{expit}(x_i^\top \beta)$ and GLM working weights $w_i = \pi_i (1 - \pi_i)$:

1. **Jeffreys-Invariant Penalized Log-Likelihood**:
   $$\ell^*(\beta) = \sum_{i=1}^n \big[y_i \log \pi_i + (1 - y_i)\log(1 - \pi_i)\big] + \frac{1}{2} \log \det I(\beta), \qquad I(\beta) = X^\top W X$$
   Adding $\frac{1}{2}\log \det I(\beta)$ removes the leading $O(n^{-1})$ small-sample bias of the MLE and guarantees finite, unique estimates $\hat{\beta}$ even under complete separation (e.g., $2 \times 2$ tables with zero cells).

2. **Modified Score Equations**:
   Let $h_i = w_i \, x_i^\top I(\beta)^{-1} x_i$ be the leverage (hat-matrix diagonal) of observation $i$. Differentiating $\ell^*(\beta)$ yields the modified score vector:
   $$U^*(\beta) = X^\top \big(y - \pi + h \odot (0.5 - \pi)\big) = 0$$

3. **Penalized Likelihood Ratio Test & Profile Penalized Likelihood CIs**:
   - **PLRT (`pl=True`)**: For each coordinate $j$, maximizes $\ell^*(\beta)$ subject to $\beta_j = 0$ to obtain $\tilde{\ell}^*_j(0)$ and evaluates $p_j = P\big(\chi^2_1 \ge 2[\ell^*(\hat{\beta}) - \tilde{\ell}^*_j(0)]\big)$.
   - **Profile CIs (`ci_method="profile"`)**: Solves $\tilde{\ell}^*_j(\beta_j) = \ell^*(\hat{\beta}) - \frac{1}{2}\chi^2_{1, 1-\alpha}$ separately on $(-\infty, \hat{\beta}_j]$ and $[\hat{\beta}_j, +\infty)$ via tangent warm-started constrained scoring and `brentq` root-finding, capturing exact asymmetry under separation where symmetric Wald intervals fail.

---

## Quickstart

```python
import numpy as np
from logistf import fit_firth_logistic

# Quasi-complete separation (2x2 table with a zero cell: exact Firth OR = 21.0)
x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

res = fit_firth_logistic(
    x,
    y,
    add_intercept=True,
    coef_names=["treatment"],
    ci_method="profile",
    pl=True,
    settings="R_default",
)
print(res.to_dataframe())
```

---

## `settings="default"` vs `settings="R_default"` Reference

`fit_firth_logistic` accepts `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Aspect | `settings="default"` | `settings="R_default"` (CRAN `logistf` `src/logistf.c`) |
|---|---|---|
| **Scoring update matrix** | Standard Fisher information $I(\beta) = X^\top W X$ | Augmented Newton–Fisher matrix $I_{\text{aug}}(\beta) = X^\top \text{diag}(w_i(1 + h_i)) X$ (`src/logistf.c`) |
| **Default `max_iter`** | `80` (with BFGS fallback if needed) | `25` (`logistf.control(maxit = 25)`, no post-hoc BFGS override) |
| **Default `tol`** | `1e-7` | `1e-5` (`lconv = gconv = xconv = 1e-5`) |
| **Asymptotic covariance $\widehat{\text{Cov}}(\hat{\beta})$** | $I(\hat{\beta})^{-1} = (X^\top W X)^{-1}$ | $I(\hat{\beta})^{-1} = (X^\top W X)^{-1}$ |

---

## Public API & Output Schema

### Function Signature
```python
fit_firth_logistic(
    X: np.ndarray | pd.DataFrame,
    y: Sequence[int | float | bool],
    add_intercept: bool = True,
    coef_names: Sequence[str] | None = None,
    ridge_lambda: float = 0.0,
    ci_method: str = "profile",  # {'profile', 'wald'}
    pl: bool = True,
    alpha: float = 0.05,
    max_iter: int | None = None,
    max_step_halving: int = 15,
    tol: float | None = None,
    *,
    settings: Literal["default", "R_default"] = "default",
) -> FirthLogisticResult
```

### `FirthLogisticResult` & `.to_dataframe()` Columns
| Field | `to_dataframe()` Column | Description |
|---|---|---|
| `coef_names` | *(Index)* | Parameter names (auto-prepends `(Intercept)` when `add_intercept=True`) |
| `beta` | `beta` | Bias-reduced log-odds coefficients $\hat{\beta}$ |
| `se` | `se` | Asymptotic standard errors $\sqrt{\text{diag}(I(\hat{\beta})^{-1})}$ |
| `z_stat` | `z` | Wald $z$-statistics $\hat{\beta}_j / \text{se}_j$ |
| `p_value` | `p_value` | 1-df PLRT $p$-values (`pl=True`) or two-sided Wald $p$-values (`pl=False`) |
| `odds_ratio` | `OR` | Exponentiated odds ratios $\exp(\hat{\beta})$ |
| `ci_low_or`, `ci_high_or` | `OR_95CI_low`, `OR_95CI_high` | `(1 - alpha)` profile (or Wald) confidence limits on the OR scale |
| `ci_low_beta`, `ci_high_beta` | *(Attributes)* | `(1 - alpha)` confidence limits on the log-odds $\beta$ scale |
| `penalized_loglik`, `converged`, `n_iter` | *(Attributes)* | Maximized $\ell^*(\hat{\beta})$, convergence flag, and iteration count |

---

## Testing

```bash
PYTHONPATH=logistf /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest logistf/tests -v
```

## License

MIT.
