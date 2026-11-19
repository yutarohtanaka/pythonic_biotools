# `lhcmr`: Latent Heritable Confounder Mendelian Randomization

Standalone Python implementation of **LHC-MR** (Darrous et al., *Nature Communications* 2021) for bidirectional causal effect estimation between exposure $X$ and outcome $Y$ in the presence of an unmeasured heritable confounder $U$ and arbitrary sample overlap using genome-wide GWAS summary statistics.

## Citations

> Darrous, L., Mounier, N., & Kutalik, Z. (2021). Simultaneous estimation of bi-directional causal effects and heritable confounding from GWAS summary statistics. *Nature Communications*, 12(1), 7274. https://doi.org/10.1038/s41467-021-26970-w
>
> Bulik-Sullivan, B. K., Loh, P. R., Finucane, H. K., Ripke, S., Yang, J., ... & Neale, B. M. (2015). LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. *Nature Genetics*, 47(3), 291–295. https://doi.org/10.1038/ng.3211

## Statistical Model

Across genome-wide SNPs $j = 1, \dots, M$ with LD scores $\ell_j$, standardized GWAS summary statistics follow an exact 8-component ($c = (c_X, c_Y, c_U) \in \{0, 1\}^3$) structural bivariate Gaussian mixture:

$$\begin{pmatrix}\beta_{X,j} \\ \beta_{Y,j}\end{pmatrix} = (I - A)^{-1} \begin{pmatrix} \gamma_{X,j} + t_X u_j \\ \gamma_{Y,j} + t_Y u_j \end{pmatrix}, \qquad A = \begin{pmatrix} 0 & \alpha_{Y\to X} \\ \alpha_{X\to Y} & 0 \end{pmatrix}$$

with sampling covariance $\Sigma_{e,j} = \begin{pmatrix} i_X / N_X & i_{XY} / \sqrt{N_X N_Y} \\ i_{XY} / \sqrt{N_X N_Y} & i_Y / N_Y \end{pmatrix}$ and LD-score-scaled genetic covariance $\ell_j (I - A)^{-1} D_c (I - A)^{-\top}$ under Darrous et al. identifiability ($t_X \ge 0$, $\operatorname{Var}(G_U) = 1$, $|\alpha_{X\to Y}\alpha_{Y\to X}| < 0.95$).

## `settings` Modes (`"default"` vs `"R_default"`)

All primary entry points (`lhc_mr`, `LHCMR`, and `bivariate_ldsc`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`LizaDarrous/lhcMR` Parity) |
| :--- | :--- | :--- |
| **SNP Outlier Filtering** | Retains all finite SNPs unless `filter_outliers=True`. | Filters extreme chi-square SNPs via $z^2 > \max(80, 0.001 N)$ matching LDSC/`lhcMR`. |
| **Stage-2 Parameterization** | Smooth softplus/logit transforms on heritabilities and polygenicities with Stage-2 intercept refinement. | Fixes $(i_X, i_Y)$ at Stage-1 LDSC estimates; direct bounded `L-BFGS-B` on $[-2, 2]$ for causal/confounder slopes. |
| **Multi-Start Initialization** | Deterministic LDSC/kurtosis anchors + scrambled Sobol quasi-random starts (`n_starts=5`). | Deterministic anchors + uniform random starts on `lhcMR` slope/polygenicity intervals. |
| **Reflection Check & SEs** | Factor-swap canonicalization ($X \leftrightarrow U, Y \leftrightarrow U$) + genomic block-jackknife SEs (`n_blocks=20`). | Full post-fit $(t_X, t_Y)$ swap/sign L-BFGS-B re-optimization likelihood comparison + block-jackknife SEs. |

## Quickstart

```python
import numpy as np
from lhcmr import LHCMR, bivariate_ldsc, lhc_mr

# Fit bidirectional LHC-MR in Pythonic default mode
res = lhc_mr(bx, sx, by, sy, ld_scores=ld, nx=80_000, ny=80_000, settings="default")
print(res.summary())
print(res.to_dataframe())
print(res.lrt_dataframe())

# Upstream LizaDarrous/lhcMR R parity mode via class interface
est_r = LHCMR(n_starts=5, n_blocks=20, settings="R_default")
res_r = est_r.fit(bx, sx, by, sy, ld_scores=ld, nx=80_000, ny=80_000)
```

## Nested Likelihood Ratio Tests & Output Schema

- **`LHCMRResult`**: Exposes `alpha_xy`, `alpha_yx`, `t_x`, `t_y`, `h2_x`, `h2_y`, `pi_x`, `pi_y`, `pi_u`, `i_x`, `i_y`, `i_xy`, dictionaries `se`, `jackknife_se`, `hessian_se`, `z_scores`, `p_values`, `stage1_ldsc`, and `lrt`.
- **Nested LRTs (`res.lrt_dataframe()`)**: Evaluates `bidirectional` vs `unidirectional_xy` ($H_0: \alpha_{Y\to X} = 0$), `unidirectional_yx` ($H_0: \alpha_{X\to Y} = 0$), `confounder_only` ($H_0: \alpha_{X\to Y} = \alpha_{Y\to X} = 0$), and `no_confounder` ($H_0: t_X = t_Y = 0$).

## License

MIT (clean-room implementation from published mathematical equations in Darrous et al. 2021).
