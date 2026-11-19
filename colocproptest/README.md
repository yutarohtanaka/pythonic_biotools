# `colocproptest`: Frequentist Proportional Genetic Colocalization Testing

Standalone Python implementation of **colocPropTest** (Wallace, *Genetic Epidemiology* 2013; Wallace, Robins & Johnson, *bioRxiv* 2025; CRAN `colocPropTest` v0.9.3) for LD-robust frequentist proportional colocalization testing across pairs of GWAS/QTL traits under multiple causal variants, out-of-sample LD, and variable per-SNP sample sizes.

## Citations

> Wallace, C. (2013). Statistical testing of shared genetic control for potentially related traits. *Genetic Epidemiology*, 37(8), 802–813. https://doi.org/10.1002/gepi.21765
>
> Wallace, C., Robins, C., & Johnson, T. (2025). Variable information across SNPs in GWAS data can cause false rejections of colocalisation which can be resolved by proportional colocalisation tests. *bioRxiv*. https://doi.org/10.1101/2025.09.08.674910
>
> Wallace, C. (2024). `colocPropTest`: Proportional Testing for Colocalisation Analysis (CRAN v0.9.3). https://doi.org/10.32614/CRAN.package.colocPropTest

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`select_tester_pairs`, `test_pair_proportionality`, `proptest_pc`, `run_coloc_prop_test`, and `ColocPropTest`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`chr1swallace/colocPropTest` v0.9.3 Parity) |
| :--- | :--- | :--- |
| **Tester Pair Selection (`select_tester_pairs`)** | Top `max_testers=15` informative variants (`min_abs_z=2.0`) pruned at `max_r2=0.85` to avoid near-singular 2x2 LD blocks. | Top `n_top=15` marginal Z-score variants pruned at `r^2 < 0.99`, scoring all $\binom{m}{2}$ tester pairs. |
| **2x2 Profile-GLS Solver (`test_pair_proportionality`)** | Exact simultaneous diagonalization + $\theta \in [0, \pi)$ angular profile-GLS ($\eta = \tan\theta$), invariant as $\eta \to \pm\infty$. | Exact correlated-error profile-GLS $\chi^2_1$ test across each tester pair. |
| **Multiplicity & Omnibus Decision (`run_coloc_prop_test`)** | Rejects $H_0$ when `min_fdr < alpha` or ACAT Cauchy omnibus `p_acat < alpha`; requires $\max|z_1| \ge 3.0$ and $\max|z_2| \ge 3.0$ for `evidence_for_coloc`. | Rejects proportionality ($H_0$ of colocalization) via Benjamini–Hochberg `min(FDR) < alpha` (`0.05`). |
| **LD-PC Omnibus & CI (`proptest_pc`)** | Projects $(\hat{\mathbf{b}}_1, \hat{\mathbf{b}}_2)$ onto LD PCs explaining `95%` variance with exact Fieller / profile-GLS 95% CI for $\eta$. | PC-projected omnibus $\chi^2_{K-1}$ test at `thr = 0.80` cumulative LD variance. |

## Quickstart

```python
import numpy as np
from colocproptest import (
    ColocPropTest,
    proptest_pc,
    run_coloc_prop_test,
    select_tester_pairs,
    test_pair_proportionality,
)

# Simulate a 30-SNP locus with a shared causal variant (true proportionality eta = 0.6)
m = 30
ld = 0.6 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
se1 = np.full(m, 0.04)
se2 = np.full(m, 0.05)
beta1 = 0.40 * ld[:, 12]
beta2 = 0.60 * beta1

# 1. Default Pythonic proportional colocalization pipeline
res = run_coloc_prop_test({"beta": beta1, "se": se1}, {"beta": beta2, "se": se2}, ld, settings="default")
print(res.summary())
print(res.to_dataframe().head())

# 2. Upstream CRAN colocPropTest v0.9.3 parity mode (supports coloc varbeta dicts)
res_r = ColocPropTest(alpha=0.05, settings="R_default").fit(
    {"beta": beta1, "varbeta": se1**2},
    {"beta": beta2, "varbeta": se2**2},
    ld,
)
print("Reject H0:", res_r.reject_proportionality, "PC omnibus p:", res_r.p_pc_omnibus)
```

## License

MIT (clean-room implementation from the published mathematical equations in Wallace 2013 and Wallace, Robins & Johnson 2025).
