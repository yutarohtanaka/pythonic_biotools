# `sva`: Surrogate Variable Analysis for Genomics & Transcriptomics

Clean-room Python implementation of **Surrogate Variable Analysis (SVA)** (Leek & Storey, *PLOS Genetics* 2007; Leek, *Biometrics* 2011; Leek, *Nucleic Acids Research* 2014) for estimating latent technical/biological confounding factors, computing moderated log-count surrogate variables (`svaseq`), estimating latent dimensionality (`num_sv`), and performing vectorized nested-model F-tests (`f_pvalue`) with `NaN`-aware Benjamini–Hochberg FDR control.

## Citations

> Leek, J. T., & Storey, J. D. (2007). Capturing heterogeneity in gene expression studies by surrogate variable analysis. *PLOS Genetics*, 3(9), e161. https://doi.org/10.1371/journal.pgen.0030161
>
> Leek, J. T. (2011). Asymptotic conditional singular value decomposition for high-dimensional genomic data. *Biometrics*, 67(2), 344–352. https://doi.org/10.1111/j.1541-0420.2010.01455.x
>
> Leek, J. T. (2014). svaseq: removing batch effects and other unwanted noise from sequencing data. *Nucleic Acids Research*, 42(21), e161. https://doi.org/10.1093/nar/gku864

## Mathematical & Algorithmic Overview

1. **Rank Estimation (`num_sv`)**:
   - **Buja–Eyuboglu (`method="be"`)**: Residualizes $Y$ against the primary model $X_1$ via $R = Y(I - H_1)$, computes singular-value variance fractions $T_k = d_k^2 / \sum_j d_j^2$, independently permutes columns within each row, re-projects orthogonal to $X_1$, and sequentially compares empirical permutation p-values against `sig_level`.
   - **Asymptotic (`method="leek"`)**: Computes deflated row-variance profiles $v_g^{(k)}$ and solves the Silverstein (1995) generalized Marchenko–Pastur / El Karoui (2007) Tracy–Widom upper-edge bound on $\lambda_k = d_k^2 / m_u$.

2. **Iteratively Reweighted SVA (`irwsva`)**:
   Alternates between:
   - Empirical-Bayes local FDR estimation (`edge_lfdr` via Storey $\hat{\pi}_0$ and probit-space Gaussian KDE) for surrogate-driven variation $p^{(\gamma)}$ comparing $[X_1, S]$ vs $X_1$ and primary biological signal $p^{(b)}$ comparing $[X_1, S]$ vs $[X_0, S]$.
   - Posterior weights $w_g = \Pr(\gamma_g = 1 \mid \text{data}) \cdot \Pr(b_g = 0 \mid \text{data})$ applied to $Y_0 = Y(I - H_0)$ prior to weighted SVD and subspace matching, preserving latent batch structure correlated with $X_1$ while avoiding collinearity.

3. **Two-Step, Supervised, and Count SVA (`twostepsva`, `sva`, `svaseq`)**:
   - `twostepsva` isolates surrogate-affected features ($\text{lfdr} \le 0.10$) from Step 1 residual SVD before Step 2 SVD.
   - `sva` dispatches `"irw"`, `"two-step"`, and `"supervised"` (using known negative-control probe weights `controls`).
   - `svaseq` applies moderated log transformation $Y = \log(C + c)$ (with optional median library-size normalization) and filters zero-variance count features before SVA.

4. **Vectorized Nested F-Tests (`f_pvalue`)**:
   Compares full model $X_1$ against nested null model $X_0$ ($\operatorname{col}(X_0) \subseteq \operatorname{col}(X_1)$) via orthogonal projections, handling partial `NaN`s per feature and applying Benjamini–Hochberg FDR strictly over finite p-values.

## Matrix Orientation

- **`AnnData` inputs**: Always interpreted in standard `obs x var` (`n_samples x n_features`) orientation (supporting `.layers[layer]`, `scipy.sparse` matrices, and `dat.obs` column design builders).
- **`numpy.ndarray` / `pandas.DataFrame` inputs**: Interpreted as `(n_features, n_samples)` by default (`features_as_rows=True`, matching Bioconductor `sva`), or `(n_samples, n_features)` when `features_as_rows=False`.

## Installation

```bash
pip install -e ./sva
```

## Quickstart

```python
import numpy as np
from sva import f_pvalue, num_sv, sva

rng = np.random.default_rng(42)
n_genes, n_samples = 500, 24
treat = np.repeat([0.0, 1.0], n_samples // 2)
mod = np.column_stack([np.ones(n_samples), treat])
mod0 = np.ones((n_samples, 1))

batch = rng.normal(size=n_samples)
Y = rng.normal(size=(n_genes, n_samples))
Y[:50] += 1.5 * treat[None, :]
Y[50:250] += 2.0 * batch[None, :]

n_est = num_sv(Y, mod, method="be", seed=42)
sv_res = sva(Y, mod, mod0, n_sv=n_est.n_sv, seed=42)
mod_adj = sv_res.augment_design(mod)
mod0_adj = sv_res.augment_design(mod0)

res = f_pvalue(Y, mod_adj, mod0_adj)
print(sv_res.summary())
print(res.to_dataframe(sort_by_pvalue=True).head())
```

## Exact Bioconductor `sva` Parity (`settings="R_default"`)

All primary estimators (`num_sv`, `sva`, `irwsva`, `twostepsva`, `svaseq`, and `edge_lfdr`) accept `settings: Literal["default", "R_default"] = "default"` (strictly validated; raises `ValueError` on unknown values):

| Component | `settings="default"` (Pythonic default) | `settings="R_default"` (Bioconductor `sva` exact parity) |
| :--- | :--- | :--- |
| **`edge_lfdr` $\hat{\pi}_0$ estimator** | Storey & Tibshirani (2003) cubic polynomial over $\lambda \in [0.05, 0.90]$ | Fixed $\lambda = 0.8$: $\hat{\pi}_0 = \min(1.0, \operatorname{mean}(p \ge 0.8) / 0.2)$ (`helper.R`) |
| **`edge_lfdr` density ratio** | Direct probit-space Gaussian KDE ratio $\hat{\pi}_0 \phi(z) / \hat{g}(z)$ | GCV cubic smoothing spline interpolation of probit log-density ratio over 512-point `bw.nrd0` KDE grid |
| **`vfilter` feature selection** | Retains `vfilter` top-variance features | Retains `vfilter - 1` top-variance features (`rank(-var) < vfilter` per `num.sv.R` & `svaseq.R`, filtering pre-log counts in `svaseq`) |

```python
from sva import num_sv, sva, svaseq

n_r = num_sv(Y, mod, method="be", vfilter=200, seed=42, settings="R_default")
sv_r = sva(Y, mod, mod0, n_sv=n_r.n_sv, vfilter=200, seed=42, settings="R_default")
```

## License

MIT
