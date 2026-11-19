# `survidinri`: Censored Survival IDI, Continuous/Categorical NRI & Median Improvement

Standalone Python implementation of **survIDINRI** (Uno et al., *Statistics in Medicine* 2013) for evaluating incremental prognostic value of biomarkers and risk models under right-censored survival data via Inverse Probability of Censoring Weighting (IPCW) and unified `Exp(1)` multiplier perturbation resampling.

## Citations

> Uno, H., Tian, L., Cai, T., Kohane, I. S., & Wei, L. J. (2013). A unified inference procedure for a class of measures to assess improvement in risk prediction systems with survival data. *Statistics in Medicine*, 32(14), 2430–2442. https://doi.org/10.1002/sim.5647
>
> Pencina, M. J., D'Agostino, R. B., & Vasan, R. S. (2008). Evaluating the added predictive ability of a new marker: from area under the ROC curve to reclassification and beyond. *Statistics in Medicine*, 27(2), 157–172. https://doi.org/10.1002/sim.2929

## `settings` Modes (`"default"` vs `"R_default"`)

All primary interfaces (`idi_inf`, `idi_inf_predictions`, and `SurvIDINRI`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`cran/survIDINRI` v1.1-2 Parity) |
| :--- | :--- | :--- |
| **Cox PH Tie Handling** | Exact weighted **Efron** partial likelihood & baseline hazard $\hat{\Lambda}_0(t_0)$. | Weighted **Breslow** partial likelihood & baseline hazard matching upstream Fortran routines. |
| **Continuous NRI (`m2_nri`)** | Conventional full-scale continuous $\text{cNRI} = \text{NRI}_+ + \text{NRI}_- \in [-2, 2]$ (also exposing `half_cNRI`). | Exact Uno et al. (2013) **half-cNRI** scale $M_2 = \frac{1}{2}\text{NRI}_+ + \frac{1}{2}\text{NRI}_- \in [-1, 1]$ matching `IDI.INF`. |
| **IPCW Tail Stabilization** | Kaplan–Meier $\hat{G}(t)$ floored at `G_min=1e-3` for extreme-horizon tail stability. | Unfloored Kaplan–Meier censoring weights (`G_min=1e-12`). |
| **Perturbation CIs** | Normal Wald $(1-\alpha)$ CIs $\hat{\theta} \pm z_{1-\alpha/2}\widehat{\text{SE}}^*$ from `npert` $\text{Exp}(1)$ perturbations. | Empirical $(\alpha/2, 1-\alpha/2)$ quantiles of perturbed estimates $\{\hat{\theta}^{(b)}\}_{b=1}^{\text{npert}}$. |
| **Summary Measures** | `m1_idi` (IPCW discrimination slope IDI), `m2_nri`, `m3_med` (median improvement), `r2_idi`, optional `cat_nri`. | Identical $M_1, M_2, M_3$ point definitions with half-scale $M_2$ and empirical quantile intervals. |

## Quickstart

```python
import numpy as np
from survidinri import idi_inf, idi_inf_predictions

# Fit baseline (clinical) vs augmented (clinical + genomic biomarker) Cox models at t0 = 5.0 years
res = idi_inf(
    time=surv_time,
    event=status,
    covs0=clinical_df[["age", "stage"]],
    covs1=clinical_df[["age", "stage", "biomarker_score"]],
    t0=5.0,
    npert=300,
    risk_thresholds=[0.10, 0.25],
    settings="default",
)
print(res.summary())

# Exact R survIDINRI::IDI.INF parity mode (Breslow ties, [-1, 1] half-cNRI, quantile CIs)
res_r = idi_inf(surv_time, status, covs0, covs1, t0=5.0, npert=300, settings="R_default")

# Evaluate pre-computed external / cross-validated t0 event risk predictions
res_ext = idi_inf_predictions(surv_time, status, risk0=pred_base, risk1=pred_aug, t0=5.0)
```

## License

MIT (clean-room implementation from published mathematical equations in Uno et al. 2013 and Pencina et al. 2008).
