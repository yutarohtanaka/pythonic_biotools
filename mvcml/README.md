# `mvcml`: Python Implementation of `MVMR-cML` & Conditional GWAS Bias Adjustment

Standalone Python implementation of Multivariable Mendelian Randomization with Constrained Maximum Likelihood (`cml_ma`) and conditional GWAS collider-bias adjustment (`mvcml_adjust`) following Lin et al. (*AJHG* 2023).

## Installation

```bash
pip install -e ./mvcml
```

## Quickstart

```python
from mvcml import mvcml_adjust

res = mvcml_adjust(betaGX, sdGX, betaGYC, sdGYC, N=100_000, iv_p_thresh=5e-7)
print("Estimated slope(s):", res["b"], "SE:", res["b_se"])
```

## Reference

- Lin, Z., Deng, Y., & Pan, W. Multivariable Mendelian randomization with constrained maximum likelihood. *American Journal of Human Genetics* 110 (2023).
