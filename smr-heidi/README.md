# smr-heidi

Python implementation of **SMR** (*Summary-data-based Mendelian Randomization*), **HEIDI** (*HEterogeneity In Dependent Instruments*), and **SMR-multi** for integrating GWAS and xQTL (eQTL/mQTL/pQTL) summary statistics.

## References

- **SMR & HEIDI**: Zhu, Z., Zhang, F., Hu, H., et al. (2016). Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets. *Nature Genetics*, 48(5), 481–487. https://doi.org/10.1038/ng.3538
- **SMR-multi**: Wu, Y., Zeng, J., Zhang, F., et al. (2018). Integrative analysis of omics summary data reveals putative mechanisms underlying complex traits. *Nature Communications*, 9(1), 918. https://doi.org/10.1038/s41467-018-03371-0

## Mathematical Overview

1. **Single-Instrument SMR (`smr_single`)**:
   At the top cis-xQTL instrument ($\arg\max z_{zx}^2$), estimates the causal effect of molecular exposure $x$ on trait $y$:
   $$\hat{b}_{xy} = \frac{\hat{b}_{zy}}{\hat{b}_{zx}}, \qquad \text{SE}(\hat{b}_{xy}) = \frac{\sqrt{\hat{b}_{zy}^2 s_{zx}^2 + \hat{b}_{zx}^2 s_{zy}^2}}{\hat{b}_{zx}^2}, \qquad T_{\text{SMR}} = \frac{z_{zy}^2 z_{zx}^2}{z_{zy}^2 + z_{zx}^2} \sim \chi^2_1.$$

2. **HEIDI Test for Pleiotropy vs. Linkage (`heidi_test`)**:
   Across $m$ LD-pruned cis-xQTLs ($p_{\text{eQTL}} < 1.57 \times 10^{-3}$, $0.05 \le r_{i,\text{top}}^2 \le 0.90$, up to $m=20$), tests $H_0: b_{xy(1)} = \dots = b_{xy(m)}$ via $d_i = \hat{b}_{xy(i)} - \hat{b}_{xy(\text{top})}$ ($i = 1, \dots, k=m-1$) using the exact dependent-instrument covariance matrix:
   $$\operatorname{Cov}(\hat{b}_{xy(i)}, \hat{b}_{xy(j)}) = r_{ij} \frac{s_{zy(i)} s_{zy(j)}}{\hat{b}_{zx(i)} \hat{b}_{zx(j)}} + r_{ij} \frac{\hat{b}_{xy(i)} \hat{b}_{xy(j)}}{z_{zx(i)} z_{zx(j)}},$$
   evaluating $\Pr\!\left(\sum_{l=1}^k \lambda_l \chi^2_1 \ge \sum_{i=1}^k z_{d,i}^2\right)$ where $\lambda = \operatorname{eigvalsh}(\Psi)$ via Imhof/Davies characteristic-function integration, Kuonen saddlepoint approximation, and Satterthwaite fallback (`pchisqsum`).

3. **Multi-SNP SMR (`smr_multi`)**:
   Combines $T_{\text{SMR}, i} = \frac{z_{zy,i}^2 z_{zx,i}^2}{z_{zy,i}^2 + z_{zx,i}^2}$ across LD-pruned cis-xQTLs ($r^2 \le 0.1$) accounting for their LD correlation matrix $R$.

4. **End-to-End Locus Harmonization (`run_locus_smr_heidi`)**:
   Harmonizes GWAS and xQTL summary DataFrames with a signed LD matrix $R$, automatically orienting swapped `A1`/`A2` alleles and LD row/column signs.

## Quickstart

```python
import numpy as np
import pandas as pd
from smr_heidi import run_locus_smr_heidi, smr_single, heidi_test, smr_multi

res_single = smr_single(b_zy=0.08, se_zy=0.015, b_zx=0.40, se_zx=0.03, snp_id="rs1")
print(res_single.summary())
```

## License

GPL-3.0-or-later (matching upstream Yang Lab `smr`).
