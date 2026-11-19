# `jmbayes2`: Bayesian Joint Longitudinal & Survival / Competing-Risks Modeling

Standalone Python implementation of **JMbayes2** (`drizopoulos/JMbayes2` v0.5; Rizopoulos, *Joint Models for Longitudinal and Time-to-Event Data*; Papageorgiou et al., *Statistics in Medicine* 2019) for Bayesian joint modeling of longitudinal biomarker trajectories and right-censored or competing-risks survival outcomes with individualized dynamic risk prediction.

## Citations

> Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event Data: With Applications in R*. Chapman & Hall/CRC. https://doi.org/10.1201/b12208
>
> Papageorgiou, G., Mauff, K., Tomer, A., & Rizopoulos, D. (2019). An overview of joint modeling of time-to-event and longitudinal outcomes. *Annual Review of Statistics and Its Application* / *Statistics in Medicine*, 38(1), 78–98. https://doi.org/10.1002/sim.8103

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`fit_lme`, `jm`, `predict_dynamic`, and `JMBayes2`) accept `settings: Literal["default", "R_default"] = "default"` and raise `ValueError` on unknown values:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`drizopoulos/JMbayes2` v0.5 Parity) |
| :--- | :--- | :--- |
| **Hazard Quadrature (`jm`)** | 15-node Gauss–Kronrod rule (`GK15`) over $[0, T_i]$ and $[t_{\text{landmark}}, u]$. | 15-node Gauss–Kronrod rule (`GK_k = 15`) matching `JMbayes2::jm`. |
| **Baseline Hazard Spline (`jm`)** | Cubic B-spline (`degree=3`, `n_knots=6`) with `Gamma(1.0, 0.01)` smoothing penalty precision $\tau_h$. | Quadratic B-spline (`degree=2`) with hierarchical `Gamma(1.0, 0.005)` precision update on second-difference penalty $\tau_h$. |
| **MCMC Tail Probability (`jm`)** | Finite-sample smoothed two-sided tail probability floored at $1 / (S + 1)$. | Exact two-sided MCMC tail probability $P = 2 \min(\text{mean}(\theta > 0), \text{mean}(\theta < 0))$. |
| **Dynamic Risk (`predict_dynamic`)** | Returns posterior mean and 95% credible intervals for conditional survival `surv_prob` (and cause-specific `cif`). | Also returns predicted cumulative event probability `event_prob = 1 - S(u | t_landmark)` matching `JMbayes2::predict(..., process="event")`. |

## Quickstart

```python
import numpy as np
import pandas as pd
from jmbayes2 import fit_lme, jm, predict_dynamic, JMBayes2

rng = np.random.default_rng(42)
n = 120
b = rng.multivariate_normal([0.0, 0.0], [[0.8, 0.1], [0.1, 0.08]], size=n)
trt = rng.integers(0, 2, size=n)
t_true = rng.exponential(scale=np.exp(2.0 - 0.5 * trt - 0.6 * b[:, 0]))
c_time = rng.uniform(3.0, 9.0, size=n)
surv_df = pd.DataFrame({
    "id": np.arange(n),
    "time": np.minimum(t_true, c_time),
    "event": (t_true <= c_time).astype(int),
    "trt": trt,
})

rows = []
for i in range(n):
    visits = np.linspace(0.0, surv_df.loc[i, "time"] * 0.9, 5)
    y_i = 2.0 + 0.4 * visits + b[i, 0] + b[i, 1] * visits + rng.normal(0.0, 0.25, size=5)
    for t_v, y_v in zip(visits, y_i):
        rows.append({"id": i, "obstime": t_v, "y": y_v})
long_df = pd.DataFrame(rows)

# Fit joint model linking biomarker value and slope to survival hazard
fit = jm(surv_df, long_df, surv_covs=["trt"], functional_forms=("value", "slope"), settings="default")
print(fit.summary())

# Real-time dynamic prediction for patient 0 surviving past landmark t = 2.0
patient_hist = long_df[long_df["id"] == 0]
pred_df = predict_dynamic(fit, patient_hist, t_landmark=2.0, horizons=[3.0, 4.5, 6.0], settings="R_default")
print(pred_df)
```

## License

MIT (clean-room implementation from published mathematical equations in Rizopoulos 2012 and Papageorgiou et al. 2019).
