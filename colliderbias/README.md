# `colliderbias`: Python Implementation of `ColliderBias` (CWLS & CWBLS)

Standalone Python port of the R package **`ColliderBias`** (Cai et al., *Genetic Epidemiology* 2022; Cai & Dudbridge, 2024) providing Corrected Weighted Least Squares (`method_cb` / `cwls`) and Corrected Weighted Bivariate Least Squares (`cwbls`).

## Installation

```bash
pip install -e ./colliderbias
```

## Quickstart

```python
from colliderbias import cwls, cwbls

res_cwls = cwls(xbeta, xse, ybeta, yse, weighted="1", model="ivw")
res_cwbls = cwbls(exp_beta, exp_se, dis_beta, dis_se, out_beta, out_se, n_boot=200, seed=0)
```

## References

- Cai, S., Hartley, A., Mahmoud, O., Tilling, K., & Dudbridge, F. Adjusting for collider bias in genetic association studies using instrumental variable methods. *Genetic Epidemiology* 46, 303–316 (2022).
- Cai, S. & Dudbridge, F. Estimating causal effects on disease progression correcting for collider bias. *Genetic Epidemiology* (2024).
