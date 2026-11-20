# `buysetest`: Generalized Pairwise Comparisons & Proportional Win-Fractions Regression

Standalone Python implementation of **BuyseTest** (Buyse 2010; Péron et al., *Statistics in Medicine* 2016; Ozenne et al., *Journal of Statistical Software* 2021) for prioritized composite clinical-trial endpoints (Net Benefit, Win Ratio, Win Odds), bundled with covariate-adjusted **Proportional Win-Fractions Regression (`WR::pwreg`)** (Mao & Wang, *JASA* 2020).

## Citations

> Buyse, M. (2010). Generalized pairwise comparisons of prioritized outcomes in the two-sample problem. *Statistics in Medicine*, 29(30), 3245–3257. https://doi.org/10.1002/sim.3923
>
> Péron, J., Roy, P., Ozenne, B., Roche, L., & Buyse, M. (2016). The net chance of a longer survival as a patient-oriented measure of treatment benefit in randomized clinical trials. *Statistics in Medicine*, 35(7), 978–990. https://doi.org/10.1002/sim.6950
>
> Ozenne, B., Budtz-Jørgensen, E., & Péron, J. (2021). Exact simulation-free estimation of the net benefit in clinical trials with prioritized outcomes: the BuyseTest package. *Journal of Statistical Software*, 98(11), 1–44. https://doi.org/10.18637/jss.v098.i11
>
> Mao, L., & Wang, T. (2020). A class of proportional win-fractions regression models for composite outcomes. *Journal of the American Statistical Association*, 115(529), 106–117. https://doi.org/10.1080/01621459.2019.1660175

## `settings` Modes (`"default"` vs `"R_default"`)

All primary interfaces (`buyse_test`, `BuyseTest`, and `pwreg`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`bozenne/BuyseTest` v3.3.9 & `WR::pwreg` Parity) |
| :--- | :--- | :--- |
| **TTE Pairwise Rule (`scoring_rule`)** | Péron's KM Stieltjes integral (`scoring_rule="Peron"`), `correction_uninf=0`, propagating `neutral + uninf` downward. | Identical Péron KM integral preserving explicit `uninf` tail mass when $\hat{S}(\tau_{\max}) > 0$. |
| **U-Statistic Variance (`buyse_test`)** | First-order Hájek U-statistic H-projection ($V_U = \frac{1}{m}S_{10} + \frac{1}{n}S_{01}$, `order=1`). | Second-order finite-sample H-decomposition ($V_U = \frac{1}{m}S_{10} + \frac{1}{n}S_{01} + \frac{1}{mn}S_{11}$, `order=2`). |
| **Delta-Method Transformations** | Fisher `atanh` CIs/p-values for `net_benefit` ($\Delta$) and `log` CIs/p-values for `win_ratio` & `win_odds`. | Exact `BuyseTest` `atanh` (`netBenefit`) and `log` (`winRatio` / `mannWhitney` odds) parity. |
| **Stratum Pooling (`pool`)** | Buyse pair-count weights $w_s \propto m_s n_s$ (`"CMH"` and `"equal"` also supported). | Identical Buyse pair-count weights $w_s \propto m_s n_s$. |
| **Win Regression (`pwreg`)** | IPCW pairwise score equations (`G_min=1e-3`) + degree-2 U-statistic Hájek sandwich covariance. | Unfloored pooled KM censoring weights + Mao & Wang (2020) censoring-martingale influence adjustment. |

## Quickstart

```python
from buysetest import Endpoint, buyse_test, pwreg

endpoints = [
    Endpoint("os_time", kind="tte", status_col="os_event", operator="higher", threshold=30.0),
    Endpoint("pfs_time", kind="tte", status_col="pfs_event", operator="higher", threshold=15.0),
    Endpoint("grade3_tox", kind="binary", operator="lower"),
]

# Hierarchical Generalized Pairwise Comparisons (Péron KM scoring + H-decomposition inference)
res = buyse_test(trial_df, treatment="arm", endpoints=endpoints, strata="region", settings="default")
print(res.summary())

# Exact BuyseTest v3.3.9 second-order H-decomposition parity mode
res_r = buyse_test(trial_df, treatment="arm", endpoints=endpoints, settings="R_default")

# Covariate-adjusted Proportional Win-Fractions Regression (Mao & Wang, JASA 2020)
fit_wr = pwreg(trial_df, treatment_col="arm", covariates=["age", "ecog"], endpoints=endpoints[:2])
print(fit_wr.summary())

# Hierarchical stacked Win / Loss / Neutral / Uninformative bar plot (`pip install "buysetest[plot]"`)
fig = res.plot(figsize=(8, 4))
fig.savefig("buysetest_hierarchy.png", dpi=300)
```

## License

MIT (clean-room implementation from published mathematical equations in Buyse 2010, Péron et al. 2016, Ozenne et al. 2021, and Mao & Wang 2020).
