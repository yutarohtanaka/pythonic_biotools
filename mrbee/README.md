# `mrbee`: Python Implementation of `MRBEE`

Standalone Python implementation of **MRBEE** (Lorincz-Comi et al., *AJHG* 2024) for Mendelian Randomization using bias-corrected estimating equations with sample-overlap adjustment, Winner's Curse Mills-ratio truncation correction, and IMRP horizontal-pleiotropy outlier removal.

## Installation

```bash
pip install -e ./mrbee
```

## Quickstart

```python
from mrbee import fit_mrbee

res = fit_mrbee(xbeta, xse, ybeta, yse, inst_p=5e-5, wc_correct=True)
print(res["b"], res["b_se"], res["n_outliers"])
```

## Reference

- Lorincz-Comi, N. J., Yang, Y., Li, G., & Zhu, X. MRBEE: A bias-corrected multivariable Mendelian randomization method. *American Journal of Human Genetics* 111, 366–378 (2024).
