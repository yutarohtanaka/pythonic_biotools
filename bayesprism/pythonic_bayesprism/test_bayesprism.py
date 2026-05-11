import numpy as np
import pandas as pd
import anndata as ad
from pythonic_bayesprism import Prism

# Generate dummy scRNA-seq (reference)
np.random.seed(42)
ref_X = np.random.poisson(lam=1.5, size=(100, 50)).astype(float)
ref_obs = pd.DataFrame({
    'cell_type': np.random.choice(['T', 'B', 'Tumor'], size=100),
    'cell_state': np.random.choice(['T_CD4', 'T_CD8', 'B_memory', 'B_naive', 'Tumor_1', 'Tumor_2'], size=100)
})
# Fix consistency between cell state and type
ref_obs.loc[ref_obs['cell_state'].str.startswith('T'), 'cell_type'] = 'T'
ref_obs.loc[ref_obs['cell_state'].str.startswith('B'), 'cell_type'] = 'B'
ref_obs.loc[ref_obs['cell_state'].str.startswith('Tumor'), 'cell_type'] = 'Tumor'

ref_adata = ad.AnnData(X=ref_X, obs=ref_obs)
ref_adata.var_names = [f'Gene_{i}' for i in range(50)]

# Generate dummy bulk RNA-seq (mixture)
mix_X = np.random.poisson(lam=50, size=(10, 50)).astype(float)
mix_adata = ad.AnnData(X=mix_X)
mix_adata.var_names = [f'Gene_{i}' for i in range(50)]
mix_adata.obs_names = [f'Sample_{i}' for i in range(10)]

if __name__ == '__main__':
    print("Initializing Prism...")
    bp = Prism(
        reference=ref_adata,
        mixture=mix_adata,
        cell_type_labels='cell_type',
        cell_state_labels='cell_state',
        key='Tumor',
        outlier_fraction=1.0
    )

    print("Running Prism...")
    bp.run(
        n_cores=2,
        update_gibbs=True,
        gibbs_control={'chain_length': 100, 'burn_in': 50, 'thinning': 2, 'alpha': 1.0, 'seed': 123},
        opt_control={'maxiter': 50, 'optimizer': 'MAP', 'sigma': 2.0}
    )

    res = bp.get_results(output_type='dataframe')
    print("Results:")
    print(res)
