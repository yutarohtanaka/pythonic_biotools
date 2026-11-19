# `slopehunter`: Python Implementation of SlopeHunter

Standalone Python port of the R package **SlopeHunter** (Mahmoud et al., *Nature Communications* 2022) for adjusting conditional / subsequent-trait GWAS summary statistics for index-event (collider) bias via two-component bivariate Gaussian mixture EM clustering.

## Installation

```bash
pip install -e ./slopehunter
```

## Quickstart

```python
from slopehunter import hunt, sh_adj

fit = hunt(xbeta, xse, ybeta, yse, xp_thresh=0.001, bootstrapping=True, M=100, seed=777)
ybeta_adj, yse_adj = sh_adj(fit, xbeta, xse, ybeta, yse)
print("Adjustment factor b:", fit["b"], "SE:", fit["bse"])
```

## Reference

- Mahmoud, O., Dudbridge, F., Davey Smith, G., Munafo, M., & Tilling, K. A robust tool for discriminative adjustment for collider bias in genetic association studies. *Nature Communications* 13, 610 (2022).
