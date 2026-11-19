# `mrraps`: Python Implementation of `mr.raps`

Standalone Python port of the R package **`mr.raps`** (Zhao et al., *Annals of Statistics* 2020; GPL-3.0-or-later) for two-sample summary-data Mendelian Randomization with many weak instruments, systematic pleiotropic overdispersion ($\tau^2$), and outlier-robust loss functions (`l2`, `huber`, `tukey`).

## Installation

```bash
pip install -e ./mrraps
```

## Quickstart

```python
from mrraps import mr_raps

res = mr_raps(b_exp, b_out, se_exp, se_out, over_dispersion=True, loss_function="huber")
print(res["beta_hat"], res["beta_se"], res["beta_p_value"])
```

## Reference

- Zhao, Q., Wang, J., Hemani, G., Bowden, J., & Small, D. S. Statistical inference in two-sample summary-data Mendelian randomization using robust adjusted profile score. *Annals of Statistics* 48, 1742–1769 (2020).
