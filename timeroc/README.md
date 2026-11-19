# timeroc

Inverse Probability of Censoring Weighted (**IPCW**) time-dependent ROC curves, cumulative/dynamic $\text{AUC}(t)$ estimators, simultaneous confidence bands, and paired two-biomarker inference for right-censored survival and competing-risks data.

## Reference

> Blanche, P., Dartigues, J.-F., & Jacqmin-Gadda, H. (2013). Estimating and comparing time-dependent areas under receiver operating characteristic curves for censored event times with competing risks. *Statistics in Medicine*, 32(30), 5381–5397. https://doi.org/10.1002/sim.5958

## Mathematical Overview

Given follow-up times $T_i$, event indicators $\delta_i \in \{0, 1, 2, \dots\}$ (`0` = right-censored, `cause=1` = primary event, other positive integers = competing events), and baseline marker $M_i$:

1. **IPCW Censoring Survival $\hat{G}(t)$**:
   Estimated either nonparametrically via reverse Kaplan–Meier (`weighting="marginal"`, where events precede censorings at ties so $\hat{F}_1(t) + \hat{F}_2(t) + \hat{S}(t) \equiv 1$) or conditionally via Breslow Cox proportional hazards (`weighting="cox"`).

2. **Competing-Risks Sensitivity, Specificities, and $\text{AUC}(t)$**:
   - Cases ($T_i \le t, \delta_i = \text{cause}$) carry weight $w_{1,i}(t) = 1 / \hat{G}(T_i-)$, yielding $\widehat{\text{Se}}(c, t)$ and cumulative incidence $\hat{F}_1(t)$.
   - **Definition 1 (`ROC_1`, `AUC_1`)**: Controls include both event-free survivors ($T_j > t$, weight $1/\hat{G}(t)$) and subjects with a competing event by $t$ ($T_j \le t, \delta_j \notin \{0, \text{cause}\}$, weight $1/\hat{G}(T_j-)$).
   - **Definition 2 (`ROC_2`, `AUC_2`)**: Controls are strictly event-free survivors ($T_j > t$, weight $1/\hat{G}(t)$).

3. **Asymptotic IID Influence-Function Inference (`iid=True`)**:
   Computes exact $O(n \log n)$ per-subject IID representations $\hat{\phi}_{d,i}(t)$ accounting for Kaplan–Meier censoring martingale perturbations, yielding analytic standard errors, pointwise CIs, Gaussian-multiplier simultaneous confidence bands across multiple horizons, and paired two-biomarker comparison tests (`compare_timeroc`) with both per-horizon $z$-tests and a joint multi-horizon Wald $\chi^2$ test.

## Quickstart

```python
import numpy as np
from timeroc import time_roc, compare_timeroc

rng = np.random.default_rng(42)
n = 300
m1 = rng.normal(size=n)
m2 = 0.4 * m1 + rng.normal(size=n)
t_event = rng.exponential(scale=np.exp(-0.8 * m1))
t_comp = rng.exponential(scale=2.5, size=n)
t_cens = rng.exponential(scale=3.0, size=n)

T = np.minimum(np.minimum(t_event, t_comp), t_cens)
delta = np.where(T == t_cens, 0, np.where(t_event <= t_comp, 1, 2))

fit1 = time_roc(T, delta, m1, times=[0.5, 1.0, 1.5], cause=1, iid=True)
fit2 = time_roc(T, delta, m2, times=[0.5, 1.0, 1.5], cause=1, iid=True)
print(fit1.summary())

cmp = compare_timeroc(fit1, fit2, definition=1)
print(cmp.summary())
```

## License

GPL-2.0-or-later (matching upstream CRAN `timeROC`).
