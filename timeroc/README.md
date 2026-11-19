# `timeroc`: IPCW Time-Dependent ROC Curves & Competing-Risks $\text{AUC}(t)$ Inference

Standalone Python implementation of CRAN **`timeROC`** for Inverse Probability of Censoring Weighted (**IPCW**) time-dependent ROC curves, competing-risks cumulative/dynamic $\text{AUC}(t)$ estimators, $O(n \log n)$ martingale IID influence functions, Gaussian-multiplier simultaneous confidence bands, and paired two-biomarker hypothesis tests.

---

## Published Citations

- **Blanche, P., Dartigues, J.-F., & Jacqmin-Gadda, H.** (2013). Estimating and comparing time-dependent areas under receiver operating characteristic curves for censored event times with competing risks. *Statistics in Medicine*, 32(30), 5381–5397. https://doi.org/10.1002/sim.5958
- **Hung, H., & Chiang, C.-T.** (2010). Estimation methods for time-dependent AUC models with survival data. *Canadian Journal of Statistics*, 38(1), 8–26. https://doi.org/10.1002/cjs.10046

---

## Statistical Formulation

Given follow-up times $T_i \ge 0$, event codes $\delta_i \in \{0, 1, 2, \dots\}$ (`0` = right-censored, `cause` = primary event of interest, other positive integers = competing events), and continuous baseline marker $M_i$:

1. **IPCW Censoring Survival $\hat{G}(t)$**:
   - `weighting="marginal"`: Reverse Kaplan–Meier estimator where failure events precede censorings at ties, preserving exact partition of unity $\hat{F}_1(t) + \hat{F}_2(t) + \hat{S}(t) \equiv 1$.
   - `weighting="cox"`: Conditional Breslow Cox PH censoring model given `marker` and optional `other_markers`.

2. **Competing-Risks Sensitivity, Specificities & Concordance $\text{AUC}(t)$**:
   Primary cases carry weight $w_{1,i}(t) = \mathbb{I}(T_i \le t, \delta_i = \text{cause}) / \hat{G}(T_i-)$ (or strict $T_i < t$ in `settings="R_default"`), event-free survivors carry $w_{0,i}(t) = \mathbb{I}(T_i > t) / \hat{G}(t)$, and prior competing events carry $w_{2,i}(t) = \mathbb{I}(T_i \le t, \delta_i \notin \{0, \text{cause}\}) / \hat{G}(T_i-)$. Two control definitions distinguish clinical vs. etiological discrimination:
   - **Survivors + Prior Competing Events** ($w_0 + w_2$): discriminates primary event by $t$ against all subjects who did not experience the primary event by $t$.
   - **Strictly Event-Free Survivors** ($w_0$): discriminates primary event by $t$ against subjects surviving free of any event past $t$.

3. **Martingale IID Influence Functions (`iid=True`)**:
   Decomposes $\sqrt{n}(\widehat{\text{AUC}}_d(t) - \text{AUC}_d(t)) = \frac{1}{\sqrt{n}}\sum_{i=1}^n \hat{\phi}_{d,i}(t) + o_p(1)$ in $O(n \log n)$ time into the U-statistic Hájek projection plus the reverse-KM censoring martingale integral over $u_m \le t$, enabling analytic standard errors, Gaussian-multiplier simultaneous confidence bands across multiple horizons, and paired biomarker $\Delta\text{AUC}(t)$ z-tests + joint multi-horizon Wald $\chi^2$ inference (`compare_timeroc`).

---

## Quickstart

```python
import numpy as np
from timeroc import compare_timeroc, time_roc

rng = np.random.default_rng(42)
n = 300
m1 = rng.normal(size=n)
m2 = 0.4 * m1 + rng.normal(size=n)
t_event = rng.exponential(scale=np.exp(-0.8 * m1))
t_comp = rng.exponential(scale=2.5, size=n)
t_cens = rng.exponential(scale=3.0, size=n)

T = np.minimum(np.minimum(t_event, t_comp), t_cens)
delta = np.where(T == t_cens, 0, np.where(t_event <= t_comp, 1, 2))

fit1 = time_roc(T, delta, m1, times=[0.5, 1.0, 1.5], cause=1, iid=True, settings="R_default")
fit2 = time_roc(T, delta, m2, times=[0.5, 1.0, 1.5], cause=1, iid=True, settings="R_default")
print(fit1.summary())

cmp = compare_timeroc(fit1, fit2, definition=1, settings="R_default")
print(cmp.summary())
```

---

## `settings="default"` vs `settings="R_default"` Reference

`time_roc` and `compare_timeroc` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Aspect | `settings="default"` | `settings="R_default"` (CRAN `timeROC` v0.4.1 `timeROC_3.R`) |
|---|---|---|
| **`auc_1` / `sp_1` / `fp_1` controls** | Survivors + prior competing events ($w_0 + w_2$, Blanche et al. 2013 Paper Def. 1) | Strictly event-free survivors ($T > t$, $w_0$, CRAN `timeROC`'s `AUC_1` / `Spec_1`) |
| **`auc_2` / `sp_2` / `fp_2` controls** | Strictly event-free survivors ($T > t$, $w_0$, Blanche et al. 2013 Paper Def. 2) | Survivors + prior competing events ($w_0 + w_2$, CRAN `timeROC`'s `AUC_2` / `Spec_2`) |
| **Case tie rule at horizon $t$** | Inclusive $T_i \le t$ (`delta == cause`) | Strict $T_i < t$ in point estimates (`Cases <- (T < t & delta == cause)` in `timeROC_3.R`) |
| **IID SE & covariance scaling** | Plug-in empirical second moment (`ddof=0`, $/n^2$) | Sample SD / covariance (`ddof=1`, `apply(iid, 2, sd) / sqrt(n)` in `timeROC_3.R` & `compare.R`) |

---

## Public API & Output Dataclasses

### Functions
- `time_roc(T, delta, marker, times, *, cause=1, weighting="marginal", other_markers=None, cutpoints=None, iid=None, settings="default") -> TimeROCResult`
- `compare_timeroc(fit1, fit2, *, definition=1, n_sim=2000, seed=42, settings="default") -> CompareTimeROCResult`

### Result Types & `.to_dataframe()` Columns
- **`TimeROCResult`**:
  - `.to_dataframe()` columns: `time`, `n_cases`, `n_survivors`, `n_competing`, `F_1`, `F_2`, `S`, `AUC_1`, `AUC_2`, plus (when `iid=True`) `SE_1`, `CI_1_lower`, `CI_1_upper`, `CB_1_lower`, `CB_1_upper`, `SE_2`, `CI_2_lower`, `CI_2_upper`, `CB_2_lower`, `CB_2_upper`.
  - `.roc_curve(time_index=0)` columns: `cutpoint`, `Se`, `Sp_1`, `FP_1`, `Sp_2`, `FP_2`.
  - `.confidence_intervals(level=0.95, *, definition=1, n_sim=2000, seed=42) -> ConfidenceBandResult`.
- **`CompareTimeROCResult`**:
  - `.to_dataframe()` columns: `time`, `AUC_marker1`, `AUC_marker2`, `delta_AUC`, `SE_delta`, `z_stat`, `p_value`, `p_value_adjusted`, plus joint test attributes `wald_stat`, `wald_df`, `p_wald_joint`.

---

## Testing

```bash
PYTHONPATH=timeroc /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest timeroc/tests -v
```

## License

GPL-2.0-or-later (matching upstream CRAN `timeROC`).
