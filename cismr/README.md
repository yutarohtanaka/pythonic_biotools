# `cismr`: Correlated Drug-Target cis-Mendelian Randomization

Standalone Python implementation of correlated cis-locus Mendelian Randomization estimators for molecular quantitative trait loci (cis-pQTL / cis-eQTL / cis-vQTL):
- **PC-GMM & MV-PC-GMM** (Burgess et al., *Genetic Epidemiology* 2017/2020; Grant & Burgess, *Statistics in Medicine* 2022)
- **MR-SPI** — Selecting Promoting Instruments via LD-aware pairwise ratio compatibility voting (Yao et al., *Biometrics* 2024)
- **MR-GENIUS** — Heteroskedastic G-estimation under pervasive additive horizontal pleiotropy (Tchetgen Tchetgen et al., *Statistical Science* 2021)

## Citations

> Burgess, S., Zuber, V., Valdes-Marquez, E., Sun, B. B., & Hopewell, J. C. (2017). Mendelian randomization with fine-mapped genetic variants: Choosing from all variants in a gene region. *Genetic Epidemiology*, 41(8), 714–725. https://doi.org/10.1002/gepi.22295
>
> Grant, A. J., & Burgess, S. (2022). An efficient and robust approach to Mendelian randomization with measured pleiotropic variants in a high-dimensional setting. *Statistics in Medicine*, 41(12), 2132–2147. https://doi.org/10.1002/sim.9278
>
> Yao, M., Miller, G. W., Vardarajan, B. N., Belsky, D. W., Guo, Z., & Lin, Z. (2024). Selecting valid instrumental variables in Mendelian randomization with cis-variants. *Biometrics*, 80(1), ujad041. https://doi.org/10.1093/biomtc/ujad041
>
> Tchetgen Tchetgen, E., Sun, B., & Walter, S. (2021). The GENIUS approach to robust Mendelian randomization inference. *Statistical Science*, 36(3), 443–464. https://doi.org/10.1214/20-STS792

## `settings` Modes (`"default"` vs `"R_default"`)

All primary estimators (`mr_pcgmm`, `mr_pcivw`, `mr_mvpcgmm`, `mr_spi`, `mr_genius`, `mr_genius_summary`, and `CisMR`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`MendelianRandomization` v0.10.0 & `MR.SPI` Parity) |
| :--- | :--- | :--- |
| **LD Matrix Conditioning** | Projects `ld_R` onto PSD correlation matrices (`min_eig=1e-6`) with unit diagonal. | Uses symmetrized `rho` directly (`Phi0 = outer(bx/sy, bx/sy) * rho`) without pre-projection. |
| **PC-GMM & MV-PC-GMM** | Multi-step iterative GMM (`tol=1e-8`) + bounded CUE refinement; overdispersion-robust SE with normal critical values. | Exact 2-step GMM update; scales SE by $\sqrt{\max(1, J/(K - P))}$ with $t_{K-P}$ critical values when `robust=True`. |
| **MR-SPI Valid-Set Voting** | Finite-sample pairwise Bonferroni threshold $\Phi^{-1}(1 - \alpha / (s(s-1)))$ + Bron–Kerbosch maximum-clique selection. | Exact $\sqrt{2 \log p}$ pair voting threshold + degree-maximizing anchor voter rule. |
| **MR-GENIUS (Individual)** | Doubly Neyman-orthogonalized $(X, Y)$ residuals with quadratic genotype saturation + iterated efficient GMM. | Linear working mean model $E[X \mid 1, G, C]$ with 2-step efficient GMM matching `bluosun/MR-GENIUS`. |
| **MR-GENIUS (Summary vQTL)** | Empirical-Bayes attenuation shrinkage on noisy `v_gx` weights + iterative LD-whitened variance-instrument solver. | Raw `v_gx` weights with 2-step LD-whitened variance-instrument update. |

## Quickstart

```python
import numpy as np
from cismr import CisMR, mr_pcgmm, mr_mvpcgmm, mr_spi, mr_genius, mr_genius_summary

# 1. Correlated cis-locus PC-GMM & Hansen-Sargan J-test
res_pc = mr_pcgmm(bx, sx, by, sy, ld_R=R, nx=50_000, thres=0.99, settings="default")
print(res_pc.summary())

# 2. Upstream MendelianRandomization::mr_pcgmm parity mode
res_pc_r = mr_pcgmm(bx, sx, by, sy, ld_R=R, nx=50_000, thres=0.99, settings="R_default")

# 3. MR-SPI plurality valid cis-instrument selection under invalid pleiotropic SNPs
res_spi = mr_spi(bx, sx, by, sy, ld_R=R, n=50_000, alpha=0.05, settings="default")
print(res_spi.selected_indices, res_spi.estimate, res_spi.ci_low, res_spi.ci_high)

# 4. Multivariable cis-MR across co-localized cis-isoforms / proteins
res_mv = mr_mvpcgmm(Bx, Sx, by, sy, ld_R=R, nx=50_000, settings="default")
print(res_mv.to_dataframe())

# 5. Complete drug-target sensitivity report (PC-GMM, PC-IVW, MR-SPI, MR-GENIUS)
suite = CisMR(thres=0.99, settings="default")
report = suite.fit(bx, sx, by, sy, ld_R=R, nx=50_000, v_gx=v_gx, se_vgx=se_vgx)
print(report.to_dataframe())
```

## License

MIT (clean-room implementation from published mathematical equations in Burgess et al. 2017/2020, Grant & Burgess 2022, Yao et al. 2024, and Tchetgen Tchetgen et al. 2021).
