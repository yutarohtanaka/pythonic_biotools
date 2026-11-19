# Pythonic BayesPrism

Python implementation of **BayesPrism** (Chu et al., *Nature Cancer* 2022) for Bayesian cell-type fraction deconvolution and sample-specific gene expression reconstruction from bulk RNA-seq and spatial transcriptomics data, natively integrated with `AnnData` and `pandas`.

## Installation

```bash
pip install ./bayesprism
# With optional plotting utilities:
pip install "./bayesprism[plots]"
```

## Quick Start

```python
import anndata as ad
from pythonic_bayesprism import Prism

ref_adata = ad.read_h5ad("reference_scRNA.h5ad")
mix_adata = ad.read_h5ad("mixture_bulk.h5ad")

bp = Prism(
    reference=ref_adata,
    mixture=mix_adata,
    cell_type_labels="cell_type",
    cell_state_labels="cell_state",
    key="Tumor",  # Malignant cell type key (or None for non-tumor tissues)
)

bp.run(
    n_cores=4,
    update_gibbs=True,
    gibbs_control={"chain_length": 1000, "burn_in": 500, "thinning": 2, "alpha": 1.0, "seed": 123},
    opt_control={"maxiter": 100000, "optimizer": "MAP", "sigma": 2.0},
)

res_df = bp.get_results(output_type="dataframe")
res_adata = bp.get_results(output_type="anndata")
```

## Advanced Capabilities

### 1. Spatial Transcriptomics (`bp.run_st()`)
Uses shared-$\gamma$ Maximum Likelihood Estimation (`optimizer="MLE"` by default) across cell states (`optimize_psi_one_gamma`) to model platform/capture scaling factors between single-cell references and spatial spots:

```python
bp_st = Prism(
    reference=ref_adata,
    mixture=st_adata,
    cell_type_labels="cell_type",
    cell_state_labels="cell_state",
    key=None,
)
bp_st.run_st(n_cores=4)
```

### 2. De Novo Tumor Embedding Learning (`bp.learn_embedding_nmf()`)
Learns $K$ malignant expression programs ($\eta$) and sample weights ($\omega$) via Expectation-Maximization initialized by NMF on reconstructed tumor expression (`psi_mal`):

```python
embedding_res = bp.learn_embedding_nmf(K=3, cycle=20, n_cores=4)
# Keys: 'eta_prior', 'eta_post', 'omega'
```

### 3. Cell-Type-Specific Expression Extraction (`bp.get_exp()`)
```python
Z_b_cells = bp.get_exp(state_or_type="type", cell_name="B")
```

### 4. Reference & Mixture QC Plots (`plots.py`)
`matplotlib` and `seaborn` are imported lazily when plotting functions are invoked, so `pythonic_bayesprism` works in headless environments without plotting libraries installed:

```python
from pythonic_bayesprism import plot_cor_phi, plot_bulk_vs_sc

plot_cor_phi(bp, min_exp=3, cmap="RdBu_r")
plot_bulk_vs_sc(bp)
```

## Citation

- Chu, T., Wang, Z., Pe'er, D. & Danko, C. G. Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. *Nature Cancer* **3**, 505–517 (2022).
