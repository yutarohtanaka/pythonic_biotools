# `mtcojo`: Python Implementation of `GCTA-mtCOJO` & `GSMR`

Standalone Python implementation of **GCTA-mtCOJO** and **GSMR** (Zhu et al., *Nature Communications* 2018) for multi-trait-based conditional and joint analysis of GWAS summary statistics, including two-step univariate and bivariate LD-score regression (`mtcojo_ldsc`), HEIDI-outlier pleiotropy filtering (`gsmr_meta`), and single/multi-covariate summary conditioning (`mtcojo`).

## Installation

```bash
pip install -e ./mtcojo
```

## Quickstart

```python
from mtcojo import mtcojo

res = mtcojo(df, G=ref_genotypes, ldsc={"ref_ld": l2, "w_ld": wl2, "M": 1_000_000})
print("Conditional causal effect bxy:", res["bxy"])
```

## Reference

- Zhu, Z. et al. Causal associations between risk factors and common diseases inferred from GWAS summary data. *Nature Communications* 9, 224 (2018).
