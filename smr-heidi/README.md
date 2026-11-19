# `smr-heidi`: Summary-data-based Mendelian Randomization & HEIDI

Standalone Python implementation of **SMR** (*Summary-data-based Mendelian Randomization*), **HEIDI** (*HEterogeneity In Dependent Instruments*), and **SMR-multi** for integrating GWAS and cis-xQTL (eQTL, mQTL, pQTL) summary statistics with linkage disequilibrium (LD) reference matrices.

## Citation

- **SMR & HEIDI**: Zhu, Z., Zhang, F., Hu, H., Bakshi, A., Robinson, M. R., Powell, J. E., Montgomery, G. W., Goddard, M. E., Wray, N. R., Visscher, P. M., & Yang, J. (2016). Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets. *Nature Genetics*, 48(5), 481–487. https://doi.org/10.1038/ng.3538
- **SMR-multi**: Wu, Y., Zeng, J., Zhang, F., Zhu, Z., Qi, T., Zheng, Z., Lloyd-Jones, L. R., Marioni, R. E., Martin, N. G., Montgomery, G. W., Deary, I. J., Wray, N. R., Visscher, P. M., McRae, A. F., & Yang, J. (2018). Integrative analysis of omics summary data reveals putative mechanisms underlying complex traits. *Nature Communications*, 9(1), 918. https://doi.org/10.1038/s41467-018-03371-0
- **Upstream C++ reference**: Yang Lab `smr` (`SMR_data.cpp`, `StatFunc.cpp`).

## Statistical Model

1. **Single-Instrument SMR (`smr_single`)**:
   At the top cis-xQTL instrument ($\arg\max_i z_{zx,i}^2$, where $z_{zx,i} = \hat{b}_{zx,i} / s_{zx,i}$ and $z_{zy,i} = \hat{b}_{zy,i} / s_{zy,i}$), the two-sample Wald ratio, delta-method standard error, and SMR test statistic are:
   $$\hat{b}_{xy} = \frac{\hat{b}_{zy}}{\hat{b}_{zx}}, \qquad \operatorname{SE}(\hat{b}_{xy}) = \frac{\sqrt{\hat{b}_{zy}^2 s_{zx}^2 + \hat{b}_{zx}^2 s_{zy}^2}}{\hat{b}_{zx}^2}, \qquad T_{\text{SMR}} = \frac{z_{zy}^2 z_{zx}^2}{z_{zy}^2 + z_{zx}^2} \sim \chi^2_1.$$

2. **HEIDI Dependent-Instrument Heterogeneity Test (`heidi_test`, `est_cov_bxy`)**:
   To distinguish pleiotropy/causality (a single shared causal variant) from linkage (distinct causal variants in LD), HEIDI selects up to $m = 20$ LD-pruned cis-SNPs with $p_{\text{eQTL}} < 1.57 \times 10^{-3}$ and $0.05 \le r_{i,\text{top}}^2 \le 0.90$. Defining deviations $d_i = \hat{b}_{xy(i)} - \hat{b}_{xy(\text{top})}$ for $i = 1, \dots, m-1$ with covariance $V_d = D \operatorname{Cov}(\hat{\mathbf{b}}_{xy}) D^\top$ and correlation matrix $\Psi = \operatorname{diag}(V_d)^{-1/2} V_d \operatorname{diag}(V_d)^{-1/2}$, HEIDI evaluates:
   $$T_{\text{HEIDI}} = \sum_{i=1}^{m-1} z_{d,i}^2, \qquad p_{\text{HEIDI}} = \Pr\!\left(\sum_{l=1}^{m-1} \lambda_l \chi^2_1 \ge T_{\text{HEIDI}}\right), \qquad \lambda = \operatorname{eigvalsh}(\Psi).$$

3. **Multi-SNP SMR (`smr_multi`)**:
   Aggregates $T_{\text{SMR},i}$ across LD-pruned cis-xQTLs ($p_{\text{eQTL}} < 5 \times 10^{-8}$, $r^2 \le 0.1$) via quadratic-form tail probability against the eigenvalues of the pruned LD submatrix $R_{\text{sub}}$, alongside a generalized least-squares (GLS) pooled causal effect estimate $\hat{b}_{xy,\text{GLS}}$.

## Python Quickstart

```python
import numpy as np
import pandas as pd
from smr_heidi import est_cov_bxy, heidi_test, pchisqsum, run_locus_smr_heidi, smr_multi, smr_single

# 1. Single-instrument SMR
single = smr_single(b_zy=0.084, se_zy=0.014, b_zx=0.420, se_zx=0.030, snp_id="rs100")
print(single.summary())

# 2. End-to-end locus SMR + HEIDI + SMR-multi (Pythonic default vs YangLab C++ R_default)
m = 8
idx = np.arange(m)
R = 0.72 ** np.abs(idx[:, None] - idx[None, :])
snps = [f"rs{i + 1}" for i in range(m)]
b_zx = 0.45 * R[:, 2]
b_zy = 0.25 * b_zx
gwas_df = pd.DataFrame({"SNP": snps, "A1": ["A"] * m, "A2": ["G"] * m, "b": b_zy, "se": [0.015] * m})
eqtl_df = pd.DataFrame({"SNP": snps, "A1": ["A"] * m, "A2": ["G"] * m, "b": b_zx, "se": [0.025] * m})

res_default = run_locus_smr_heidi(gwas_df, eqtl_df, R, probe_id="ENSG000001", settings="default")
res_upstream = run_locus_smr_heidi(gwas_df, eqtl_df, R, probe_id="ENSG000001", settings="R_default")
print(res_default.to_dataframe())
print(res_upstream.to_dataframe())
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`run_locus_smr_heidi` / `smr_heidi_locus`, `heidi_test`, `smr_multi`, `smr_single`, `est_cov_bxy`, `pchisqsum`) accept `settings: Literal["default", "R_default"] = "default"`.

| Component | `settings="default"` | `settings="R_default"` (YangLab `smr` C++ Parity) |
| :--- | :--- | :--- |
| **`est_cov_bxy` covariance** | Second-order Zhu et al. (2016) Eq. 9: $r_{ij}\frac{s_{zy,i}s_{zy,j}}{\hat{b}_{zx,i}\hat{b}_{zx,j}} + r_{ij}\frac{\hat{b}_{xy,i}\hat{b}_{xy,j}}{z_{zx,i}z_{zx,j}}$ | Includes upstream `SMR_data.cpp` 4th-order bias-product correction $-\frac{\hat{b}_{xy,i}\hat{b}_{xy,j}}{z_{zx,i}^2 z_{zx,j}^2}$ plus $+10^{-8} I_m$ diagonal ridge |
| **Pairwise LD pruning (`heidi_test`, `smr_multi`)** | Greedy forward selection ordered by descending $z_{zx}^2$ | `rm_cor_sbat` iterative degree-based pruning: removes highest-degree SNP with $|r| > r_{\max}$ first, breaking ties by removing smaller $|z_{zx}|$ (protecting top SNP in HEIDI) |
| **`pchisqsum` tail evaluation** | Exact Imhof/Davies quadrature (`p > 1e-7`) + Kuonen saddlepoint in extreme tails (`p <= 1e-7`) + Satterthwaite fallback | Kuonen (1999) Lugannani–Rice saddlepoint on $(-\infty, \frac{1}{2\lambda_{\max}})$ with Satterthwaite two-moment fallback (`StatFunc.cpp`) |
| **`smr_multi` GLS regularization** | Relative diagonal ridge $10^{-10} \overline{\operatorname{diag}(\Sigma)} I$ | Uses the $+10^{-8} I$ ridge already incorporated by `est_cov_bxy(..., settings="R_default")` |

## Output Schema

- **`SMRSingleResult`** (`smr_single`): `snp_id`, `b_xy`, `se_xy`, `t_smr`, `p_smr`, `b_zy`, `se_zy`, `z_zy`, `b_zx`, `se_zx`, `z_zx`, plus `.to_dataframe()` and `.summary()`.
- **`HeidiResult`** (`heidi_test`): `top_snp`, `top_idx`, `n_snps`, `selected_indices`, `selected_snps`, `t_heidi`, `p_heidi`, `b_xy_selected`, `d_values`, `z_d`, `corr_d`, `eigenvalues`, plus `.to_dataframe()`, `.snp_table()`, and `.summary()`.
- **`SMRMultiResult`** (`smr_multi`): `n_snps`, `selected_indices`, `selected_snps`, `t_smr_per_snp`, `t_smr_sum`, `t_smr_mean`, `p_smr_multi`, `b_xy_gls`, `se_xy_gls`, `eigenvalues`, plus `.to_dataframe()` and `.summary()`.
- **`LocusSMRResult`** (`run_locus_smr_heidi` / `smr_heidi_locus`): `probe_id`, `n_harmonized_snps`, `n_flipped_gwas`, `n_flipped_ld`, `smr`, `heidi`, `smr_multi`, `harmonized_snps`, plus `.to_dataframe()` and `.summary()`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/smr-heidi \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest smr-heidi/tests
```
