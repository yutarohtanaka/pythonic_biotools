# `hdl`: High-Definition Likelihood Heritability, Genetic Correlation & Enrichment

Standalone Python implementation of **HDL** (Ning, Pawitan, & Shen, *Nature Genetics* 2020) for full-likelihood estimation of SNP heritability, global genetic correlation (`HDL`), single-locus local genetic correlation (`HDL-L`), and functional-annotation stratified heritability & enrichment (`HDL.S`) from GWAS summary Z-scores and block-diagonal LD reference panels.

## Citations

> Ning, Z., Pawitan, Y., & Shen, X. (2020). High-definition likelihood inference of genetic correlation across human complex traits. *Nature Genetics*, 52(8), 859–864. https://doi.org/10.1038/s41588-020-0653-y
>
> Li, Y., Pawitan, Y., & Shen, X. (2023). An enhanced framework for local genetic correlation analysis (`HDL-L`). *Nature Genetics* / *Bioinformatics*.

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions (`build_ld_eigen_panel`, `hdl_h2`, `hdl_rg`, `hdl_local`, and `hdl_stratified`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`zhenin/HDL` v1.4.3 Parity) |
| :--- | :--- | :--- |
| **LD Eigen-Panel (`build_ld_eigen_panel`)** | Unit-diagonal correlation normalization, `eigen_cut=0.99`, `lim=exp(-18)` eigenvalue floor. | Direct symmetric `eigh` with `eigen.cut=0.99` (`cumsum / sum > 0.99`) and `lim=exp(-18)`. |
| **Univariate Intercepts (`hdl_h2`, `hdl_rg`)** | Estimates intercepts $(b_1, b_2)$ freely by default to absorb cryptic population stratification. | Constrains $b_1 = b_2 = 1.0$ when `intercept_output=False` (free when `True`), while estimating cross-trait $c_{12}$ freely. |
| **Bivariate Likelihood (`hdl_rg`)** | Joint multivariate Gaussian L-BFGS-B polish initialized from LDSC moments, PSD-clamped $\|\hat{r}_g\| \le 1.5$, block jackknife SEs. | Two-stage marginal + conditional bivariate L-BFGS-B with `h2 >= 1e-6` bounds and piecewise delete-1 block jackknife SEs. |
| **Local `HDL-L` (`hdl_local`)** | Profiles intercepts within stabilized bounds when $K \ge 15$; exact nested $\chi^2_1$ LRT against $H_0: \rho_{12,\text{loc}} = 0$. | Fixes $b_1 = b_2 = 1.0$ and $c_{12} = \frac{N_0}{\sqrt{N_1 N_2}} \rho_{\text{overlap}}$ with $\chi^2_1$ LRT against $H_0: \rho_{12,\text{loc}} = 0$. |
| **Stratified `HDL.S` (`hdl_stratified`)** | Estimates baseline noise intercept $b_0$ freely alongside per-annotation $\tau_a$ in reduced LD eigenspace. | Fixes $b_0 = 1.0$ alongside per-annotation $\tau_a$ initialized from stratified LDSC moments. |

## Quickstart

```python
import numpy as np
from hdl import build_ld_eigen_panel, hdl_h2, hdl_rg, hdl_local, hdl_stratified

# 1. Precompute regularized LD block eigendecompositions (eigen_cut=0.99)
panel = build_ld_eigen_panel(R_blocks, eigen_cut=0.99, settings="default")

# 2. Single-trait heritability & global genetic correlation with block-jackknife SEs
res_h2 = hdl_h2(z1, panel, N=80_000, jackknife=True, settings="default")
res_rg = hdl_rg(z1, z2, panel, N1=80_000, N2=100_000, jackknife=True, settings="default")
print(res_h2.summary())
print(res_rg.summary())

# 3. Exact upstream zhenin/HDL v1.4.3 parity mode (b1 = b2 = 1.0 constrained)
res_rg_r = hdl_rg(z1, z2, panel, N1=80_000, N2=100_000, intercept_output=False, settings="R_default")

# 4. Single-locus local genetic correlation (HDL-L) with chi2_1 LRT p-value
res_loc = hdl_local(z1_loc, z2_loc, R_locus, N1=80_000, N2=100_000, settings="default")
print(res_loc.rg_local, res_loc.lrt_stat, res_loc.p_lrt)

# 5. Functional-annotation stratified heritability & enrichment (HDL.S)
res_strat = hdl_stratified(z1, R_blocks, annotations_blocks, N=80_000, settings="default")
print(res_strat.to_dataframe())
```

## Key Exports

- `build_ld_eigen_panel`, `LDEigenPanel`, `LDEigenBlock`: Regularized block-LD eigendecomposition retaining top `eigen_cut` variance modes above `lim`.
- `hdl_rg`, `hdl_h2`: Global bivariate and univariate full-likelihood estimation with analytical Fisher-information SEs and leave-one-block-out jackknife SEs.
- `hdl_local`, `hdl_stratified`: Single-locus local genetic correlation (`HDL-L`) with LRT testing and functional-annotation partitioned heritability (`HDL.S`).

## License

MIT (clean-room implementation from published mathematical equations in Ning, Pawitan, & Shen, *Nature Genetics* 2020).
