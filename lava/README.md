# `lava`: Local Analysis of [co]Variant Association

Standalone Python implementation of **LAVA** (Werme et al., *Nature Genetics* 2022) for univariate local SNP heritability ($h^2_{\text{loc}}$), bivariate local genetic correlation ($r_{g,\text{loc}}$), partial local genetic correlation ($r_g(X, Y \mid \mathbf{Z})$), and multivariate local genetic multiple regression across $P \ge 2$ complex traits.

## Citations

> Werme, J., van der Sluis, S., Posthuma, D., & de Leeuw, C. A. (2022). An integrated framework for local genetic correlation analysis. *Nature Genetics*, 54(3), 274–282. https://doi.org/10.1038/s41588-022-01017-y

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`process_locus`, `run_univ`, `run_bivar`, `run_pcor`, `run_multireg`, and `LAVA`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`josefin-werme/LAVA` v0.1.0 Parity) |
| :--- | :--- | :--- |
| **LD PCA Pruning (`process_locus`)** | Eigendecomposes $R = Q \Lambda Q^\top$ retaining top $K$ PCs explaining `lim=0.99` variance; computes $\hat{\Omega} = \frac{1}{K}\hat{\Delta}^\top\hat{\Delta} - \Sigma$. | Identical `lim=0.99` PC projection and method-of-moments unbiased covariance subtraction. |
| **Non-PSD $\hat{\Omega}$ Projection** | Positive-definite eigenvalue floor `1e-6` when projecting non-PSD $\hat{\Omega}$ so Schur & regression inverses stay strictly PD. | Spectral zero-clipping `max(eig, 0)` SVD reconstruction matching `LAVA` `posdef` with Moore–Penrose pseudo-inverse. |
| **Univariate Screening (`run_bivar`)** | `univ_thresh = 1.0` (evaluates all requested trait pairs unless `univ_thresh` is explicitly specified). | `univ_thresh = 0.05` automatic univariate $h^2_{\text{loc}}$ gate filtering out traits with $p_{\text{univ}} \ge 0.05$. |
| **Monte Carlo Inference** | `n_sim = 10000` Wishart simulation CIs and Rao–Blackwellized conditional normal-product / $\chi^2$ p-values. | `n_sim = 10000` with adaptive `1e4 -> 1e5` Monte Carlo refinement whenever initial $p < 10^{-3}$. |

## Quickstart

```python
import numpy as np
import pandas as pd
from lava import process_locus, run_univ, run_bivar, run_pcor, run_multireg

# 1. Construct locus LD matrix R (M x M) and marginal GWAS Z-scores (M x P)
rng = np.random.default_rng(42)
M, N = 60, 50_000
idx = np.arange(M)
ld_R = 0.75 ** np.abs(idx[:, None] - idx[None, :])
L = np.linalg.cholesky(ld_R)

b_z = rng.normal(0, 0.03, size=M)
b_x = 0.8 * b_z + rng.normal(0, 0.015, size=M)
b_y = 0.8 * b_z + rng.normal(0, 0.015, size=M)
z_df = pd.DataFrame(
    {
        "X": np.sqrt(N) * (ld_R @ b_x) + L @ rng.standard_normal(M),
        "Y": np.sqrt(N) * (ld_R @ b_y) + L @ rng.standard_normal(M),
        "Z": np.sqrt(N) * (ld_R @ b_z) + L @ rng.standard_normal(M),
    }
)

# 2. Project locus and run univariate, bivariate, partial correlation, and multiple regression
locus = process_locus(z_df, ld_R, sample_sizes={"X": N, "Y": N, "Z": N}, settings="default")
print(run_univ(locus))
print(run_bivar(locus, settings="R_default"))
print(run_pcor(locus, pair=("X", "Y"), cond_traits=["Z"]))
print(run_multireg(locus, predictors=["X", "Z"], outcome="Y"))
```

## API Overview

- `process_locus(...) -> LAVALocus`: LD eigendecomposition, whitened PC projection, and sample-overlap-aware $\hat{\Omega}$ estimation.
- `run_univ(locus, ...)`: Exact central $F(K, N_p - K - 1)$ / $\chi^2_K$ test for local genetic variance and $h^2_{\text{loc}, p}$.
- `run_bivar(locus, ...)`: Pairwise local $r_g$, $r^2$, non-central Wishart 95% CIs, and bivariate p-values.
- `run_pcor(locus, pair, cond_traits, ...)`: Conditional Schur complement partial local $r_g(X, Y \mid \mathbf{Z})$.
- `run_multireg(locus, predictors, outcome, ...)`: Standardized joint genetic slopes $\hat{\boldsymbol{\gamma}}$, joint $R^2$, and conditional/omnibus p-values.

## License

MIT (clean-room implementation from published mathematical equations in Werme et al., *Nature Genetics* 2022).
