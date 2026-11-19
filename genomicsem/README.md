# `genomicsem`: Genomic Structural Equation Modeling & Multivariate Factor GWAS

Standalone Python implementation of **GenomicSEM** (Grotzinger et al., *Nature Human Behaviour* 2019) for multivariate LD Score Regression (LDSC) block-jackknife covariance estimation, `lavaan`-syntax Reticular Action Model (`RAM`) structural equation modeling (`DWLS` & `ML`), GWAS-by-subtraction Cholesky models, and vectorized Common-Factor SNP GWAS with $Q_{\text{SNP}}$ heterogeneity testing.

## Citations

> Grotzinger, A. D., Rhemtulla, M., de Vlaming, R., Ritchie, S. J., Mallard, T. T., Hill, W. D., Ip, H. F., Marioni, R. E., McIntosh, A. M., Deary, I. J., Koellinger, P. D., Harden, K. P., Nivard, M. G., & Tucker-Drob, E. M. (2019). Genomic structural equation modelling provides insights into the multivariate genetic architecture of complex traits. *Nature Human Behaviour*, 3(5), 513–525. https://doi.org/10.1038/s41562-019-0566-x
>
> Demange, P. A., Malanchini, M., Mallard, T. T., Biroli, P., Cox, S. R., Grotzinger, A. D., ... & Nivard, M. G. (2021). Investigating the genetic architecture of noncognitive skills using GWAS-by-subtraction. *Nature Genetics*, 53(1), 35–44. https://doi.org/10.1038/s41588-020-00754-2

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`ldsc_covstruct`, `parse_lavaan_syntax`, `usermodel`, `commonfactor`, `commonfactor_gwas`, `GenomicCovStruct`, and `GenomicSEM`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`GenomicSEM` v0.0.5 Parity) |
| :--- | :--- | :--- |
| **LDSC Jackknife (`ldsc_covstruct`)** | Leave-one-block-out WLS jackknife ($B=200$) with PSD-stabilized $V_S$ (`floor=1e-10`). | Raw unregularized block-jackknife $V_S$ with spectral pseudo-inverse floor `1e-7`. |
| **Residual Variances (`usermodel`)** | `estimation="DWLS"`, Heywood-safe lower bound $\ge 10^{-6}$ on free variances (`0*` fixed variances preserved). | `estimation="DWLS"`, unconstrained RAM variances allowing negative Heywood estimates (`GenomicSEM`/`OpenMx`). |
| **Model Fit (`chisq`, `CFI`, `SRMR`)** | Browne residual $\chi^2$, `AIC`, `SRMR`, and `CFI` with stabilized $V_S$ inversion. | `CFI` against exact diagonal genetic independence model `diag(diag(S))` with `1e-7` eigenvalue floor. |
| **Factor GWAS (`commonfactor_gwas`)** | Vectorized joint Gauss–Newton DWLS $\hat{\beta}_{F,j}$, full sandwich SE, PSD check (`1e-8`), and $Q_{\text{SNP}} \sim \chi^2_{k-1}$. | `GenomicSEM::commonfactorGWAS` parity with `1e-7` smoothing/pseudo-inverse floor and $Q_{\text{SNP}}$ heterogeneity test. |

## Quickstart

```python
import numpy as np
import pandas as pd
from genomicsem import GenomicCovStruct, commonfactor, commonfactor_gwas, usermodel

# 1. Specify or estimate LDSC genetic covariance S (k x k) and sampling covariance V_S
lam = np.array([0.55, 0.50, 0.45, 0.40])
S = np.outer(lam, lam) + np.diag([0.12, 0.15, 0.18, 0.20])
V_S = np.eye(10) * 1e-4
cov = GenomicCovStruct(S=S, V_S=V_S, trait_names=["MDD", "ANX", "PTSD", "NEUR"])

# 2. Fit a common factor model or custom lavaan model (e.g., correlated two-factor)
cf = commonfactor(cov, settings="default")
print(cf.summary())
print(cf.results_df[["lhs", "op", "rhs", "est", "se", "p_value", "std_est"]])

two_fac = usermodel(cov, "F1 =~ MDD + ANX\nF2 =~ PTSD + NEUR\nF1 ~~ F2", settings="R_default")
print(two_fac.modelfit)

# 3. Run vectorized multivariate Common-Factor SNP GWAS with Q_SNP heterogeneity test
snps = [
    pd.DataFrame({"SNP": ["rs1", "rs2"], "BETA": [0.022 * l, (0.04 if i == 0 else 0.0)], "SE": [0.004, 0.004]})
    for i, l in enumerate(lam)
]
gwas_df = commonfactor_gwas(snps, cov, settings="default")
print(gwas_df[["SNP", "beta_F", "se", "p_value", "Q_SNP", "Q_pval"]])
```

## License

MIT (clean-room implementation from published mathematical equations in Grotzinger et al., *Nature Human Behaviour* 2019).
