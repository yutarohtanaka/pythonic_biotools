# BayesPrism Python Pipeline

We have successfully translated the core BayesPrism deconvolution tool from its original R codebase into a fully functioning, Scanpy-compatible Python pipeline. The pipeline handles initialization, Gibbs sampling, and Maximum A Posteriori (MAP) optimizations for reference updates. 

## Python Integration
The `Prism` pipeline seamlessly connects with Python's single-cell ecosystem, namely `anndata.AnnData` and `pandas.DataFrame`. 
- `reference`: Accommodates either an `AnnData` object of raw counts or a `pandas.DataFrame`. If `AnnData` is supplied, `cell_type_labels` and `cell_state_labels` refer to column names in `.obs`.
- `mixture`: Accommodates an `AnnData` or `pandas.DataFrame` representing the bulk gene expression profile. 
- Results are accessible through the `get_results(output_type='anndata')` method which returns an `AnnData` where `.X` contains the bulk mixture and `.obs` contains the final deconvolved fraction matrix ($\theta$).

## Methodological Adaptations
While maintaining mathematical equivalence to the original R `run.prism` formulation, several adaptations were made:
1. **Parallelization (`multiprocessing`)**: 
   - Gibbs sampling in R originally used `snowfall` for multi-core parallelism. The Python port implements this natively with `multiprocessing.Pool` utilizing parallel threads for sampling independent $Z$ and $\theta$ estimates across bulk samples $N$. 
   - Note: Parallelizing `scipy.optimize.minimize` across cell types (using `multiprocessing`) requires passing independent inputs down to workers, which was structured inside the internal worker methods `_worker_initial_gibbs` and `_worker_tumor_gibbs`. 
2. **MAP Optimization with L-BFGS-B**: 
   - The R script implements optimization utilizing a specific bounded routine (`Rcgminu.R`). The Python script substitutes this with `scipy.optimize.minimize(..., method='L-BFGS-B')`, solving for maximum-a-posteriori using exact gradients calculated as mathematically formulated by the BayesPrism authors. 
   - Out-of-bounds error handling ensures extreme estimations ($>20$) revert to zero vectors as natively handled in R (`opt.gamma[apply(abs(opt.gamma),1,max) > 20,] <- 0`).
3. **Gibbs Sampler Vectorization**: 
   - Instead of iteratively looping to compute probabilities (`rmultinom`), matrix projections compute relative likelihood distributions (`prob_mat = phi * theta_n_i[:, np.newaxis]`), drastically boosting scalar mathematical operations in NumPy.
4. **Log-Sum-Exp Trick**:
   - Both MAP probabilities and gradient step solvers were secured with numerically stable implementations using `scipy.special.logsumexp`, avoiding underflow overflow errors on highly sparse expression variables. 

## Example Usage
```python
import anndata as ad
from pythonic_bayesprism import Prism

# Load AnnData Objects
ref_adata = ad.read_h5ad("reference_scRNA.h5ad")
mix_adata = ad.read_h5ad("mixture_bulk.h5ad")

# Initialize Pipeline
bp = Prism(
    reference=ref_adata,
    mixture=mix_adata,
    cell_type_labels='cell_type',      # from ref_adata.obs['cell_type']
    cell_state_labels='cell_state',    # from ref_adata.obs['cell_state']
    key='Tumor'                        # Designate tumor reference keyword
)

# Run Deconvolution Pipeline
bp.run(
    n_cores=4, 
    update_gibbs=True,
    gibbs_control={'chain_length': 1000, 'burn_in': 500, 'thinning': 2, 'alpha': 1.0, 'seed': 123},
    opt_control={'maxiter': 100000, 'optimizer': 'MAP', 'sigma': 2.0}
)

# Fetch fraction matrix directly appended to an AnnData
res_adata = bp.get_results(output_type='anndata')
print(res_adata.obs) # Displays Cell Type Fractions per Sample!
```

## Full Replication of Advanced Features
To achieve 100% replication of the original BayesPrism tool, the following advanced features are also fully implemented in Python:

### 1. Spatial Transcriptomics Support (`bp.run_st()`)
BayesPrism supports a specialized mode for Spatial Transcriptomics (ST) data, which suffers from severe sparsity and missing cell types at the spot level. Instead of tumor-specific MLE optimization, this mode relies solely on MAP estimation across all cell states to preserve mathematical stability.
```python
# Run specific mode for ST spots
bp.run_st(
    n_cores=4,
    gibbs_control={'chain_length': 1000, 'burn_in': 500, 'thinning': 2, 'alpha': 1.0, 'seed': 123},
    opt_control={'maxiter': 100000, 'optimizer': 'MAP', 'sigma': 2.0}
)
```

### 2. De Novo Tumor Embedding Learning (`bp.learn_embedding_nmf()`)
Learn novel tumor sub-clones/expression programs not captured in the scRNA reference by using an EM algorithm initialized with Non-negative Matrix Factorization (NMF).
```python
# Learn K=3 tumor programs via EM-NMF
embedding_res = bp.learn_embedding_nmf(
    K=3, 
    cycle=50, 
    n_cores=4
)
# Returns a dictionary containing:
# 'eta_prior' : initialized NMF programs
# 'eta_post' : EM-optimized tumor programs (K x G)
# 'omega' : fractional representation of these programs per bulk sample
```

### 3. Posterior Expression Matrix ($Z$) Extraction (`bp.get_exp()`)
If you want to perform bulk Cell-Type-Specific Differential Expression (DE), you can extract the sample-specific posterior expression tensor ($Z$) for any given cell type or state.
```python
# Get deconvolved expression matrix of B cells across all bulk samples (N x G)
Z_Bcells = bp.get_exp(state_or_type='type', cell_name='B')
```

### 4. Built-in Visualization Utilities (`plots.py`)
Visualizing expression correlations to QC references and bulk mixtures:
```python
from pythonic_bayesprism import plot_cor_phi, plot_bulk_vs_sc

# Plot hierarchical clustering heatmap of cell type correlation profiles
plot_cor_phi(bp, min_exp=3, cmap="RdBu_r")

# Scatter plot of bulk vs scRNA expression with correlation scores
plot_bulk_vs_sc(bp)
```
