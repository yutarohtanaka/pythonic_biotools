# `mrapss`: Python Implementation of `MR-APSS`

Standalone Python implementation of **MR-APSS** (Hu et al., *Nature Communications* 2022) for two-sample Mendelian Randomization accounting for polygenic background pleiotropy, instrument-selection Winner's Curse (`|z_X| > t_0`), and sample overlap via bivariate LD-score regression.

## Installation

```bash
pip install -e ./mrapss
```

## Quickstart

```python
from mrapss import fit_mrapss

res = fit_mrapss(xbeta, xse, ybeta, yse, ldscore=ld_scores, iv_p_thresh=5e-5)
print(res["b"], res["b_se"], res["pval"])
```

## Reference

- Hu, X. et al. Mendelian randomization for causal inference accounting for pleiotropy and sample structure using genome-wide summary statistics. *Nature Communications* 13, 3098 (2022).
