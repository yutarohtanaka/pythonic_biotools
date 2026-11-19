# `biostats` (`pythonic_biostats`): Survival, Competing Risks, Robust WLS & High-Dimensional FDR

Standalone Python implementations (`pythonic_biostats`) of Efron-tie Cause-Specific Cox Proportional Hazards (`survival::coxph`), Fine–Gray IPCW subdistribution hazard regression (`cmprsk::crr`), MacKinnon–White HC3 heteroskedasticity-consistent Weighted Least Squares (`sandwich::vcovHC`), Storey–Tibshirani spline $q$-values (`qvalue::qvalue`), and vectorized high-dimensional correlation scans with memory-bounded pooled-null permutation empirical FDR.

---

## Published Citations

- **Efron, B.** (1977). The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), 557–565. https://doi.org/10.1080/01621459.1977.10480613
- **Fine, J. P., & Gray, R. J.** (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496–509. https://doi.org/10.1080/01621459.1999.10474144
- **MacKinnon, J. G., & White, H.** (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325. https://doi.org/10.1016/0304-4076(85)90158-7
- **Storey, J. D., & Tibshirani, R.** (2003). Statistical significance for genomewide studies. *Proceedings of the National Academy of Sciences*, 100(16), 9440–9445. https://doi.org/10.1073/pnas.1530509100

---

## Statistical Formulation

1. **Efron-Tie Cause-Specific Cox PH (`fit_cox_ph_efron`)**:
   At each distinct target failure time $t_k$ with $d_k$ tied events $\mathcal{D}_k$ and risk set $\mathcal{R}_k = \{i : T_i \ge t_k\}$:
   $$\ell_{\text{Efron}}(\beta) = \sum_{k} \left[ \sum_{i \in \mathcal{D}_k} \eta_i - \sum_{\ell=0}^{d_k - 1} \log\!\left(\sum_{j \in \mathcal{R}_k} e^{\eta_j} - \frac{\ell}{d_k}\sum_{i \in \mathcal{D}_k} e^{\eta_i}\right) \right]$$

2. **Fine–Gray IPCW Competing-Risks Regression (`fit_fine_gray_ipcw`)**:
   Subjects experiencing a competing event ($E_i \notin \{0, \text{target\_event}\}$) at $T_i < t_k$ remain in the subdistribution risk set at $t_k$ with exact time-varying weight $w_i(t_k) = \hat{G}(t_k-) / \hat{G}(T_i-)$ from left-continuous Kaplan–Meier censoring survival $\hat{G}(t-)$. Asymptotic standard errors use the Fine & Gray (1999, Eqs. 8–10) Huber–White sandwich estimator $I(\hat{\beta})^{-1} \big[\sum_i (\hat{\eta}_i + \hat{\psi}_i)(\hat{\eta}_i + \hat{\psi}_i)^\top\big] I(\hat{\beta})^{-1}$, accounting for censoring-weight estimation uncertainty $\hat{\psi}_i$.

3. **MacKinnon–White HC3 Weighted Least Squares (`fit_wls_hc3`)**:
   On transformed $(\tilde{X}, \tilde{y}) = (W^{1/2} X, W^{1/2} y)$ with leverage $h_{ii} = \tilde{x}_i^\top (\tilde{X}^\top \tilde{X})^+ \tilde{x}_i$:
   $$\widehat{\text{Cov}}_{\text{HC3}}(\hat{\beta}) = (\tilde{X}^\top \tilde{X})^+ \left(\sum_{i=1}^n \frac{\tilde{e}_i^2}{(1 - h_{ii})^2} \tilde{x}_i \tilde{x}_i^\top\right) (\tilde{X}^\top \tilde{X})^+$$

4. **Storey–Tibshirani Spline $q$-Values (`storey_qvalue`)**:
   Evaluates $\hat{\pi}_0(\lambda)$ across $\lambda \in \{0.05, 0.10, \dots, 0.95\}$, smooths with a 3-df cubic spline evaluated at $\max(\lambda)$, and computes monotone step-down $q_{(i)} = \min_{j \ge i} \frac{\hat{\pi}_0 m p_{(j)}}{j}$.

---

## Quickstart

```python
import numpy as np
from pythonic_biostats import (
    fit_cox_ph_efron,
    fit_fine_gray_ipcw,
    fit_wls_hc3,
    permutation_empirical_fdr,
    storey_qvalue,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
times = rng.exponential(scale=np.exp(-0.7 * X[:, 0]))
events = rng.choice([0, 1, 2], size=200, p=[0.35, 0.45, 0.20])

# Cause-specific Cox PH & Fine-Gray subdistribution hazard in CRAN cmprsk parity mode
cox = fit_cox_ph_efron(X, times, events, target_event=1, coef_names=["biomarker", "cov"], settings="R_default")
fg = fit_fine_gray_ipcw(X, times, events, target_event=1, coef_names=["biomarker", "cov"], settings="R_default")
print(fg.to_dataframe())

# Bioconductor qvalue::pi0est parity mode
pvals = np.concatenate([rng.beta(0.2, 8.0, 300), rng.uniform(0.0, 1.0, 700)])
qvals, pi0_hat = storey_qvalue(pvals, settings="R_default")
```

---

## `settings="default"` vs `settings="R_default"` Reference

`fit_cox_ph_efron`, `fit_fine_gray_ipcw`, `fit_wls_hc3`, and `storey_qvalue` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Function | `settings="default"` | `settings="R_default"` (CRAN / Bioconductor Upstream) |
|---|---|---|
| **`fit_fine_gray_ipcw`** | `eps_g = 0.05` lower clamp on $\hat{G}(t-)$; `times <= u_c` at competing/censoring ties in $\hat{\psi}_i$ | `eps_g = 0.0` (unclamped KM) and strict `times < u_c` at tied competing/censoring times (`cmprsk` `crr.f:347`) |
| **`storey_qvalue`** | Counts strict $p_j > \lambda$; fits 3-df regression natural cubic spline | Counts $p_j \ge \lambda$ (`qvalue` `pi0est.R:118`); fits exact Reinsch penalized smoothing spline `smooth.spline(df=3)` via `scipy.interpolate.make_smoothing_spline` |
| **`fit_cox_ph_efron`** | Efron partial likelihood (`z_crit = 1.959964`) | Efron partial likelihood (`survival::coxph`, exact `norm.isf(0.025)`) |
| **`fit_wls_hc3`** | Exact MacKinnon–White HC3 leverage covariance | Exact `sandwich::vcovHC(..., type="HC3")` parity |

---

## Public API & Output Schema

### Survival & Regression Models
- `fit_cox_ph_efron(X, times, event_types, target_event=1, ..., *, settings="default") -> HazardModelResult`
- `fit_fine_gray_ipcw(X, times, event_types, target_event=1, eps_g=None, ..., *, settings="default") -> HazardModelResult`
  - `HazardModelResult.to_dataframe()` columns: `beta`, `se`, `z`, `p_value`, `HR`, `HR_95CI_low`, `HR_95CI_high` (plus `partial_loglik`, `n_samples`, `n_target_events`, `converged`).
- `fit_wls_hc3(X, y, weights=None, add_intercept=False, coef_names=None, *, settings="default") -> RobustWLSResult`
  - `RobustWLSResult.to_dataframe()` columns: `beta`, `se_HC3`, `t`, `p_value` (plus `df_resid`, `cov_hc3`).
- `kaplan_meier_censoring_survival`, `compute_mdes_interaction_hr`

### High-Dimensional Correlation & Multiple Testing
- `storey_qvalue(pvals, lambdas=None, *, settings="default") -> tuple[np.ndarray, float]`
- `bh_fdr(pvals) -> np.ndarray`
- `permutation_empirical_fdr(X, y, observed_rho=None, method="spearman", n_perm=10_000, n_bins=4000, random_state=0) -> tuple[obs_rho, empirical_p, empirical_fdr]`
- `spearman_matrix`, `pearson_matrix`, `bootstrap_spearman_ci`, `partial_corr_pval`, `ttest_ind_safe`, `mannwhitneyu_safe`, `fisher_exact_two_sided`

---

## Testing

```bash
PYTHONPATH=biostats /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest biostats/tests -v
```

## License

MIT.
