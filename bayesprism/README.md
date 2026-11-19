# `bayesprism`: Bayesian Cell-Type & Expression Deconvolution

Python implementation of **BayesPrism** (Chu et al., *Nature Cancer* 2022) for Bayesian cell-type fraction deconvolution, sample-specific gene expression reconstruction, spatial transcriptomics scaling, and *de novo* malignant program discovery, natively integrated with `AnnData` and `pandas`.

## Citation

- Chu, T., Wang, Z., Pe'er, D., & Danko, C. G. Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. *Nature Cancer* **3**, 505–517 (2022). https://doi.org/10.1038/s43018-022-00356-3
- Upstream Danko Lab R package: https://github.com/Danko-Lab/BayesPrism

## Statistical & Algorithmic Overview

1. **Initial Collapsed/Blocked Gibbs Sampling**:
   Samples latent read assignments $Z_{n,g,s} \sim \text{Multinomial}(X_{n,g}, \propto \theta_{n,s} \phi_{s,g})$ and cell-state fractions $\theta_n \sim \text{Dirichlet}(\sum_g Z_{n,g,\cdot} + \alpha)$, aggregating posterior cell-state profiles into canonical cell types.
2. **MAP / MLE Reference Updating (`update_reference`)**:
   Updates reference expression profiles ($\psi$) via sample-specific MLE for malignant cells (`key`) and L-BFGS-B MAP log-fold-change optimization ($\sigma = 2.0$) for non-malignant microenvironment cell types (or shared-$\gamma$ MLE across states in spatial transcriptomics `run_st`).
3. **Final Gibbs Fraction Estimation & NMF Tumor Embedding (`learn_embedding_nmf`)**:
   Re-estimates refined cell-type proportions $\theta_f$ (and posterior uncertainty `theta_cv`) and optionally decomposes sample-specific malignant expression `psi_mal` into $K$ latent tumor programs $(\eta, \omega)$ via EM initialized by NMF.

## Quickstart

```python
from pythonic_bayesprism import Prism, plot_bulk_vs_sc, plot_cor_phi

bp = Prism(
    reference=ref_adata,
    mixture=mix_adata,
    cell_type_labels="cell_type",
    cell_state_labels="cell_state",
    key="Tumor",
    settings="R_default",
)
bp.run(n_cores=1, update_gibbs=True)

theta_df = bp.get_results(output_type="dataframe")
print(theta_df.head())
print("Posterior CV shape:", bp.theta_cv.shape)

# Spatial transcriptomics mode & de novo tumor embedding
emb = bp.learn_embedding_nmf(K=3, cycle=20, settings="R_default")
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Accumulates running posterior means `(Z, theta)` without storing per-draw arrays (`theta_cv` omitted).
  - Initializes `learn_embedding_nmf` with single-start Frobenius-norm NMF.
  - Clusters `plot_cor_phi` using Ward linkage on Euclidean distance between correlation rows.
- **`settings="R_default"` (Exact `Danko-Lab/BayesPrism` parity)**:
  - Computes and stores **`theta_cv`** (posterior $\text{SD} / \text{mean}$ across thinned post-burn-in Gibbs draws, matching `ThetaPost@theta.cv`, including state-summed cell-type draws) on `bp.theta_cv` and inside `posterior_*` dictionaries across `run()` and `run_st()`.
  - Uses Brunet **KL-divergence** (`beta_loss="kullback-leibler"`, `solver="mu"`, multi-start `n_runs=10` selecting minimal KL reconstruction error) NMF initialization in `learn_embedding_nmf` (`learn_embedding.R`).
  - Uses **`1 - cor_mat`** pairwise distance (`complete` hierarchical linkage) in `plot_cor_phi`.

## Key Outputs

- `bp.get_results(output_type="dataframe" | "anndata")`: deconvolved cell-type proportions (`samples x cell_types`).
- `bp.get_exp(state_or_type="type" | "state", cell_name=...)`: sample-specific posterior read matrix (`samples x genes`).
- `bp.theta_cv`: `(samples x cell_types)` posterior coefficient of variation (when `settings="R_default"`).
- `bp.learn_embedding_nmf(...)`: dict with `'eta_prior'`, `'eta_post'`, and `'omega'`.

## Running Tests

```bash
PYTHONPATH=bayesprism /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest bayesprism/tests -q
```
