# `indexevent`: Python Implementation of `indexevent`

Standalone Python port of the R package **`indexevent`** (Dudbridge et al., *Nature Communications* 2019; Cai et al., 2022) for adjusting subsequent-trait GWAS summary statistics for index-event bias using Corrected Weighted Least Squares (`CWLS`), `Hedges-Olkin`, or Simulation-Extrapolation (`Simex`) with profile-likelihood confidence intervals.

## Installation

```bash
pip install -e ./indexevent
```

## Quickstart

```python
from indexevent import indexevent

res = indexevent(xbeta, xse, ybeta, yse, method="CWLS")
print("Adjusted slope:", res["b"], "SE:", res["b_se"])
```

## Reference

- Dudbridge, F. et al. Adjustment for index event bias in genome-wide association studies of subsequent events. *Nature Communications* 10, 1561 (2019).
