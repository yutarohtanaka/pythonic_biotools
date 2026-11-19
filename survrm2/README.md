# `survrm2`: Restricted Mean Survival Time (RMST) & RMTL Inference

Standalone Python implementation of **survRM2** (`uno1lab/survRM2` v1.0-4; Uno et al., *Journal of Clinical Oncology* 2014; Tian et al., *Biostatistics* 2014) for single-arm, two-sample, and ANCOVA covariate-adjusted **Restricted Mean Survival Time (`RMST`)** and **Restricted Mean Time Lost (`RMTL`)** estimation under non-proportional hazards.

## Citations

> Uno, H., Claggett, B., Tian, L., Inoue, E., Gallo, P., Miyata, T., ... & Wei, L. J. (2014). Moving beyond the hazard ratio in quantifying the between-group difference in survival analysis. *Journal of Clinical Oncology*, 32(22), 2380–2385. https://doi.org/10.1200/JCO.2014.55.2208
>
> Tian, L., Zhao, H., & Wei, L. J. (2014). Predicting the restricted mean event time with the subject's baseline covariates in survival analysis. *Biostatistics*, 15(2), 231–244. https://doi.org/10.1093/biostatistics/kxt050

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`rmst1`, `rmst2`, `rmst2reg`, `rmst2_curve`, and `SurvRM2`) accept `settings: Literal["default", "R_default"] = "default"` and raise `ValueError` on unknown values:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`uno1lab/survRM2` v1.0-4 Parity) |
| :--- | :--- | :--- |
| **Truncation Horizon (`tau`)** | Auto-selects `min(max(time[arm==0]), max(time[arm==1]))` and checks valid KM support. | Enforces `survRM2::rmst2` admissibility rule (checks last observation event vs censor; raises `ValueError` beyond positive KM support). |
| **Greenwood Boundary Atom (`rmst1`)** | Evaluates boundary event atom $Y_k = d_k$ smoothly via $d_k / Y_k^2$. | Exact `survRM2::rmst1` Greenwood hazard increment $d_k / (Y_k(Y_k - d_k))$ with `0` when $Y_k = d_k$. |
| **IPCW Censoring KM (`rmst2reg`)** | Pooled Kaplan–Meier $\hat{G}(t-)$ across arms with `G_min=1e-3` tail floor. | Separate per-arm Kaplan–Meier $\hat{G}_a(t-)$ ($a \in \{0, 1\}$) with unfloored IPCW weights. |
| **Sandwich Covariance (`rmst2reg`)** | Pooled censoring-martingale influence projection $\mathbf{B}_i$ across `"identity"`, `"log_rmst"`, and `"log_rmtl"`. | Separate per-arm censoring-martingale influence projection matching `survRM2::rmst2reg`. |

## Quickstart

```python
import numpy as np
import pandas as pd
from survrm2 import rmst1, rmst2, rmst2_curve, SurvRM2

rng = np.random.default_rng(42)
n = 200
arm = rng.integers(0, 2, size=n)
age = rng.normal(0.0, 1.0, size=n)
true_t = rng.exponential(scale=np.exp(1.8 + 0.45 * arm - 0.25 * age))
cens_t = rng.exponential(scale=14.0, size=n)
time = np.minimum(true_t, cens_t)
status = (true_t <= cens_t).astype(int)

# Unadjusted two-sample RMST difference, RMST ratio, and RMTL ratio
res = rmst2(time, status, arm, tau=8.0, settings="default")
print(res.summary())

# Tian et al. (2014) IPCW covariate-adjusted ANCOVA in R_default mode
cov_df = pd.DataFrame({"age": age})
res_adj = rmst2(time, status, arm, tau=8.0, covariates=cov_df, settings="R_default")
print(res_adj.to_dataframe())

# Truncation-horizon sensitivity curve and object-oriented estimator
curve_df = rmst2_curve(time, status, arm, n_grid=10, settings="default")
est = SurvRM2(tau=8.0, settings="R_default").fit(time, status, arm, covariates=cov_df)
```

## License

MIT (clean-room implementation from published mathematical equations in Uno et al. 2014 and Tian et al. 2014).
