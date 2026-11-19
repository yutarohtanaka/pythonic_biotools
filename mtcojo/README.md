# `mtcojo`: Multi-Trait Conditional & Joint GWAS Analysis (`GCTA-mtCOJO` & `GSMR`)

Standalone Python implementation of **GCTA-mtCOJO** and **GSMR** for multi-trait conditional analysis of GWAS summary statistics free of collider bias, incorporating univariate/bivariate LD-score regression (`mtcojo_ldsc`), LD clumping and single/global HEIDI-outlier pleiotropy filtering (`gsmr_meta`), and single/multi-covariate GWAS conditioning (`mtcojo`).

## Citation

- **GCTA-mtCOJO & GSMR**: Zhu, Z., Zheng, Z., Zhang, F., Wu, Y., Trzaskowski, M., Maier, R., Robinson, M. R., McGrath, J. J., Visscher, P. M., Wray, N. R., & Yang, J. (2018). Causal associations between risk factors and common diseases inferred from GWAS summary data. *Nature Communications*, 9(1), 224. https://doi.org/10.1038/s41467-017-02317-2
- **Upstream C++ reference**: Yang Lab `gcta` (`--mtcojo`, `--gsmr`).

## Statistical Model

To estimate the marginal SNP associations of target trait $y$ conditional on $C \ge 1$ covariate risk factors $\mathbf{x} = (x_1, \dots, x_C)$ without inducing individual-level collider bias:
1. **Cross-Trait LDSC (`mtcojo_ldsc`)**: Estimates SNP heritabilities $h_t^2$, genetic correlations $r_{g,ij}$ (forming covariate genetic correlation matrix $R_m$), and bivariate LDSC intercepts $I_{ij}$ capturing estimation-error correlation from sample overlap.
2. **Causal Effect Estimation (`gsmr_meta`)**: For each covariate $x_c$, selects clumped genome-wide significant instruments, removes pleiotropic variants via iterative single-SNP and global HEIDI-outlier tests (`pchisqsum` saddlepoint + Satterthwaite), and estimates marginal $\hat{b}_{x_c y}$ via dependent-instrument GLS.
3. **Multi-Trait Summary Conditioning (`cond_single`, `cond_multiple`)**: Converts marginal causal slopes $\hat{\mathbf{b}}_{xy}$ to joint conditional effects $\hat{\mathbf{b}}_j = D_m^{-1} R_m^{-1} D_m \hat{\mathbf{b}}_{xy}$ (where $D_m = \operatorname{diag}(\sqrt{h_{x_c}^2 V_{p,x_c}})$) and computes covariate-adjusted summary statistics:
   $$\hat{b}_{zy\mid \mathbf{x}} = \hat{b}_{zy} - \hat{\mathbf{b}}_{z\mathbf{x}}^\top \hat{\mathbf{b}}_j, \qquad \operatorname{Var}(\hat{b}_{zy\mid \mathbf{x}}) = s_{zy}^2 + (\mathbf{s}_{z\mathbf{x}} \odot \hat{\mathbf{b}}_j)^\top I_{xx} (\mathbf{s}_{z\mathbf{x}} \odot \hat{\mathbf{b}}_j) - 2 s_{zy} \sum_{c=1}^C s_{zx_c} \hat{b}_{j,c} I_{y,x_c}.$$

## Python Quickstart

```python
import numpy as np
import pandas as pd
from mtcojo import align_sumstats_to_bim, mtcojo

sub = 100
df = pd.DataFrame(
    {
        "SNP": [f"rs{i}" for i in range(sub)],
        "CHR": 1,
        "BP": np.arange(sub) * 1000,
        "A1": ["A"] * sub,
        "A2": ["G"] * sub,
        "YFREQ": np.full(sub, 0.30),
        "XFREQ": np.full(sub, 0.31),
        "XBETA": np.linspace(-0.05, 0.05, sub),
        "XSE": np.full(sub, 0.01),
        "YBETA": np.linspace(-0.03, 0.03, sub),
        "YSE": np.full(sub, 0.01),
        "YN": 100_000.0,
        "XN": 100_000.0,
    }
)

# Pythonic default mode
res_def = mtcojo(df, bxy_external=[0.40], settings="default")

# Upstream YangLab gcta --mtcojo parity mode
res_r = mtcojo(df, bxy_external=[0.40], freq_thresh=0.20, settings="R_default")
print(res_r["ybeta_adj"][:5], res_r["yse_adj"][:5])
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`mtcojo`, `align_sumstats_to_bim`, `gsmr_meta`, `mtcojo_ldsc`, `cond_single`, `cond_multiple`, `pchisqsum`) accept `settings: Literal["default", "R_default"] = "default"`.

| Feature | `settings="default"` | `settings="R_default"` (YangLab `gcta --mtcojo` Parity) |
| :--- | :--- | :--- |
| **Allele alignment (`align_sumstats_to_bim`)** | Drops palindromic SNPs (`A/T`, `C/G`) and resolves strand-complement pairs (`A/G <-> T/C`) | Exact-string matching `(A1, A2) == (ref_A1, ref_A2)` or swapped `(ref_A2, ref_A1)` **without** strand-complement flipping |
| **Allele-frequency QC (`mtcojo`)** | Compares `FREQ` against LD reference `ref_af` (raising `ValueError` if $>5\%$ fail) | Checks **all pairwise differences** $|f_a - f_b| > \text{freq\_thresh}$ across target (`FREQ`/`YFREQ`), each active covariate (`XFREQ*`), and LD reference (`ref_af`), filtering SNPs failing any pair |
| **Per-trait $V_p$ heterozygosity** | Uses shared `2 * FREQ * (1 - FREQ)` across traits | Uses each covariate's own `XFREQ*` when provided |

## Output Schema

- **`align_sumstats_to_bim`**: Returns `(df_aligned, G_aligned)` where `df_aligned` is ordered to `bim_df` with swapped alleles oriented (`YBETA`, `XBETA*` negated and `FREQ`/`YFREQ`/`XFREQ*` mapped to $1 - f$ when `align_target="bim"`).
- **`mtcojo`**: Returns a `dict` with keys:
  - `ybeta_adj` (`np.ndarray`): Conditional target effect sizes $\hat{b}_{zy\mid\mathbf{x}}$ (`NaN` for filtered SNPs)
  - `yse_adj` (`np.ndarray`): Conditional standard errors $\operatorname{SE}(\hat{b}_{zy\mid\mathbf{x}})$ (`NaN` for filtered SNPs)
  - `bxy` (`np.ndarray`): Marginal causal effect estimates $\hat{b}_{x_c y}$ for $c = 1, \dots, C$
  - `gsmr` (`list[dict]`): Per-covariate GSMR output dictionaries (`bxy`, `se`, `p`, `n_snps`, `heidi_p`, `instruments`, `pleio`, `err`)
  - `ldsc` (`dict | None`): LDSC `intercept` and `slope` matrices when `ldsc` inputs are supplied
  - `vp` (`np.ndarray`): Estimated phenotypic variances $(V_{p,y}, V_{p,x_1}, \dots, V_{p,x_C})$
  - `keep` (`np.ndarray`): Boolean mask of SNPs passing QC and pairwise allele-frequency filters

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/mtcojo \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mtcojo/tests
```
