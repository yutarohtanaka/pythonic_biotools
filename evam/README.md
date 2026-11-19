# `evam`: Evolutionary Accumulation & Cancer Progression Models

Standalone Python implementations of cross-sectional cancer progression and Evolutionary Accumulation Models (**EvAMs**), unifying Oncogenetic Trees (`Oncotree`), Disjunctive/Conjunctive Bayesian Networks (`OncoBN`), Hidden Conjunctive Bayesian Networks (`ct-cbn`), Mutual Hazard Networks (`LearnMHN`), and Continuous-Time Markov Chain (**CTMC**) pairwise acquisition ordering (`EvAM-Tools`) under a common `EvamResult` interface.

---

## Published Citations

- **Desper, R., Jiang, F., Kallioniemi, O. P., Moch, H., Papadimitriou, C. H., & Schäffer, A. A.** (1999). Inferring tree models for oncogenesis from comparative genome hybridization data. *Journal of Computational Biology*, 6(1), 37–51. https://doi.org/10.1089/cmb.1999.6.37
- **Gerstung, M., Baudis, M., Moch, H., & Beerenwinkel, N.** (2009). Quantifying cancer progression with conjunctive Bayesian networks. *Bioinformatics*, 25(21), 2809–2815. https://doi.org/10.1093/bioinformatics/btp505
- **Schill, R., Solbrig, S., Wettig, T., & Spang, R.** (2020). Modelling cancer progression using Mutual Hazard Networks. *Bioinformatics*, 36(1), 241–249. https://doi.org/10.1093/bioinformatics/btz513
- **Nicol, P. B., Coombes, K. R., Deaver, C., Chkrebtii, O., Paul, S., Toland, A. E., & Asiaee, A.** (2021). Oncogenetic network estimation with disjunctive Bayesian networks. *PLOS Computational Biology*, 17(5), e1008945. https://doi.org/10.1371/journal.pcbi.1008945
- **Diaz-Uriarte, R., & Herrera-Nieto, P.** (2022). EvAM-Tools: tools for evolutionary accumulation and cancer progression models. *Bioinformatics*, 38(24), 5457–5459. https://doi.org/10.1093/bioinformatics/btac710
- **Schill, R., Klever, M., Rupp, K., Hu, Y. L.,rusch, P., Wettig, T., & Spang, R.** (2024). Overcoming observation bias for cancer progression modeling. *Journal of Machine Learning Research* (`oMHN` / `LearnMHN`).

---

## Statistical Formulation

Given a cross-sectional binary genotype matrix $X \in \{0, 1\}^{N \times n}$ across $N$ independent tumors and $n$ somatic events:

1. **Oncogenetic Trees (`fit_ot`)**:
   Each event $i$ selects the strictly more frequent parent $j \in \{\text{Root}, 1, \dots, n\}$ ($p_j > p_i$) maximizing $w(j \to i) = \log\!\big(p_{ij} / [p_i(p_i + p_j)]\big)$ with edge transition probability $p_{ij}/p_j$ and optional L-BFGS-B false-positive/false-negative `(epos, eneg)` correction (`Oncotree::oncotree.fit`).

2. **Disjunctive & Conjunctive Bayesian Networks (`fit_oncobn`)**:
   Searches over acyclic parent sets $|\text{Pa}(i)| \le k$ via exact subset dynamic programming (`search="dp"` for $n \le 14$) or frequency-ordered search (`search="freq_order"`), modeling $x_i \mid \text{Pa}(i) \sim \text{Bernoulli}(\theta_i)$ when the OR (`DBN`) or AND (`CBN`) parent condition holds and $\text{Bernoulli}(\epsilon)$ otherwise.

3. **Mutual Hazard Networks (`fit_mhn`, `rates_from_theta`, `order_from_theta`)**:
   Parameterizes genotype transition rates on $\{0, 1\}^n$ as $R_{s, i} = \exp\!\big(\theta_{ii} + \sum_{j \in s} \theta_{ij}\big)$ ($i \notin s$), maximizing the $\ell_1$-penalized time-marginalized log-likelihood under constant (`cMHN`) or event-modulated (`oMHN`) tumor observation hazards. Uses `spang-lab/LearnMHN` when installed, with a built-in exact $2^n$-lattice reverse-mode adjoint `L-BFGS-B` solver for $n \le 12$.

4. **Shared Acquisition Ordering Scale (`order`)**:
   Every model produces an $n \times n$ matrix `order.loc[a, b]` $= P(a \text{ before } b) \in [0, 1]$ (`order[a,b] + order[b,a] == 1`, diagonal `NaN`) via exact embedded-chain DP over $\{0, 1\}^n$ (`ctmc_order_exact`), Monte Carlo jump paths (`ctmc_order_mc`), DAG ancestor/dominator reachability (`structural_order`), or marginal frequency ratios (`fit_frequency`, `fit_frequency_fisher`).

---

## Quickstart

```python
import numpy as np
import pandas as pd
from evam import fit_mhn, fit_oncobn, fit_ot

rng = np.random.default_rng(42)
N = 200
a = rng.binomial(1, 0.70, N)
b = np.where(a == 1, rng.binomial(1, 0.75, N), rng.binomial(1, 0.04, N))
c = np.where((a == 1) & (b == 1), rng.binomial(1, 0.70, N), rng.binomial(1, 0.03, N))
X = pd.DataFrame({"APC": a, "KRAS": b, "TP53": c})

ot_res = fit_ot(X, settings="R_default")
bn_res = fit_oncobn(X, settings="R_default")
mhn_res = fit_mhn(X, settings="R_default")

print(mhn_res.edges)
print(mhn_res.pairwise())
```

---

## `settings="default"` vs `settings="R_default"` Reference

`fit_ot`, `fit_oncobn`, and `fit_mhn` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Function | `settings="default"` | `settings="R_default"` (Upstream R / CRAN / GitHub) |
|---|---|---|
| **`fit_oncobn`** | `model="DBN"`, `penalty="bic"`, enforces progression positivity $\theta_i > \epsilon$, raw conditional `loglik` | Matches `phillipnicol/OncoBN::fitCPN` defaults: `model="CBN"`, `penalty="none"`, no $\theta_i > \epsilon$ rejection, and adds `+ N` offset to `loglik` |
| **`fit_mhn`** | `kind="cMHN"`, `round_result=False`, prunes minor off-diagonal edges $|\theta_{ij}| \le 0.05$ | Matches `spang-lab/LearnMHN` defaults: `kind="oMHN"`, `round_result=True` (2 decimals), retains all non-zero off-diagonal edges ($\theta_{ij} \ne 0$) |
| **`fit_ot`** | Exact `Oncotree::oncotree.fit` parent weights + `(epos, eneg)` L-BFGS-B error model | Exact `Oncotree::oncotree.fit` parity |

---

## Public API & `EvamResult` Schema

### Functions
- `fit_ot(X, estimate_errors=True, *, settings="default") -> EvamResult`
- `fit_oncobn(X, model=None, k=3, epsilon=None, search=None, penalty=None, *, settings="default") -> EvamResult`
- `fit_mhn(X, lam=None, maxit=5000, kind=None, *, round_result=None, settings="default") -> EvamResult`
- `fit_cbn(X, init_poset="OT", temp=1.0, steps=None, threads=1, keep_dir=None) -> EvamResult` (requires external `h-cbn` binary)
- `fit_frequency(X)`, `fit_frequency_fisher(X, alpha=0.05)`
- `rates_from_theta`, `order_from_theta`, `ctmc_order_exact`, `ctmc_order_mc`, `structural_order`

### `EvamResult` Fields
| Field / Method | Type | Description |
|---|---|---|
| `method` | `str` | Fitted model identifier (`"OT"`, `"OncoBN_DBN"`, `"OncoBN_CBN"`, `"cMHN"`, `"oMHN"`, `"H-CBN"`, etc.) |
| `events` | `list[str]` | Ordered somatic event names from `X.columns` |
| `edges` | `pd.DataFrame` | Directed edge table with columns `From`, `To`, `weight` (plus `weight_err_corrected` for `OT`) |
| `order` | `pd.DataFrame` | $n \times n$ matrix where `order.loc[a, b]` $= P(a \text{ acquired before } b)$ |
| `.pairwise()` | `pd.DataFrame` | Tidy unordered pairs table with columns `a`, `b`, `p_a_before_b` |
| `model`, `meta` | `dict` | Model-specific parameter estimates (`log_theta`, `theta`, `parents`, `loglik`) and routing metadata |

---

## Testing

```bash
PYTHONPATH=evam /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest evam/tests -v
```

## License

AGPL-3.0-or-later (matching upstream `EvAM-Tools`).
