import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from typing import Union, List, Dict, Optional, Tuple
import multiprocessing
import warnings

def _norm_to_one(ref: np.ndarray, pseudo_min: float = 1e-8) -> np.ndarray:
    """
    Normalize expression matrix to sum to 1 for each row, 
    with zero entries replaced by pseudo_min.
    """
    G = ref.shape[1]
    row_sums = ref.sum(axis=1, keepdims=True)
    # Handle rows that might have sum 0 (though should be filtered)
    row_sums[row_sums == 0] = 1.0 
    
    phi = (ref / row_sums) * (1 - pseudo_min * G) + pseudo_min
    
    # If the minimum value is greater than zero, simply normalize by total depth
    min_val = ref.min(axis=1)
    which_row = min_val > 0
    if np.any(which_row):
        phi[which_row] = ref[which_row] / row_sums[which_row]
        
    return phi

def _collapse(ref: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sum up reads for each level in labels.
    Returns collapsed matrix and unique labels array.
    """
    valid_idx = ~pd.isna(labels)
    if not np.all(valid_idx):
        warnings.warn("NA found in labels. Excluding these cells.")
        labels = labels[valid_idx]
        ref = ref[valid_idx]
        
    uniq_labels = np.unique(labels)
    ref_collapsed = np.zeros((len(uniq_labels), ref.shape[1]))
    
    for i, label in enumerate(uniq_labels):
        mask = (labels == label)
        if sparse.issparse(ref):
            ref_collapsed[i] = np.array(ref[mask].sum(axis=0)).flatten()
        else:
            ref_collapsed[i] = np.sum(ref[mask], axis=0)
            
    return ref_collapsed, uniq_labels

def _filter_bulk_outlier(mixture: np.ndarray, outlier_cut: float, outlier_fraction: float) -> np.ndarray:
    """Filter genes in mixture whose expression fraction is greater than outlier.cut 
    in more than outlier.fraction of bulk data."""
    mixture_norm = mixture / np.maximum(mixture.sum(axis=1, keepdims=True), 1)
    outlier_idx = np.mean(mixture_norm > outlier_cut, axis=0) > outlier_fraction
    
    if np.sum(outlier_idx) > 0:
        print(f"Number of outlier genes filtered from mixture = {np.sum(outlier_idx)}")
        
    return ~outlier_idx

class Prism:
    def __init__(self,
                 reference: Union[ad.AnnData, pd.DataFrame, np.ndarray],
                 mixture: Union[ad.AnnData, pd.DataFrame, np.ndarray],
                 cell_type_labels: Union[str, np.ndarray, List[str]],
                 cell_state_labels: Optional[Union[str, np.ndarray, List[str]]] = None,
                 key: Optional[str] = None,
                 outlier_cut: float = 0.01,
                 outlier_fraction: float = 0.1,
                 pseudo_min: float = 1e-8,
                 gene_names_ref: Optional[np.ndarray] = None,
                 gene_names_mix: Optional[np.ndarray] = None,
                 sample_names_mix: Optional[np.ndarray] = None):
        """
        Construct a Prism object from user provided data.
        
        Parameters:
        - reference: AnnData object of single-cell data, or raw counts matrix (cells x genes).
        - mixture: AnnData object or DataFrame of bulk data (samples x genes).
        - cell_type_labels: Array of cell type labels or column name in reference.obs.
        - cell_state_labels: Array of cell state labels or column name in reference.obs. If None, uses cell_type_labels.
        - key: String corresponding to malignant cells in cell_type_labels.
        - outlier_cut, outlier_fraction: Filter genes heavily expressed in few bulks.
        - pseudo_min: Minimum value to replace zeros after normalization.
        """
        self.key = key
        self.pseudo_min = pseudo_min
        
        # 1. Parse reference
        if isinstance(reference, ad.AnnData):
            ref_mat = reference.X
            if isinstance(cell_type_labels, str):
                ct_labels = reference.obs[cell_type_labels].values
            else:
                ct_labels = np.array(cell_type_labels)
                
            if cell_state_labels is None:
                cs_labels = ct_labels
            elif isinstance(cell_state_labels, str):
                cs_labels = reference.obs[cell_state_labels].values
            else:
                cs_labels = np.array(cell_state_labels)
            
            ref_genes = reference.var_names.values
        else:
            ref_mat = reference.values if isinstance(reference, pd.DataFrame) else reference
            ct_labels = np.array(cell_type_labels)
            cs_labels = np.array(cell_state_labels) if cell_state_labels is not None else ct_labels
            ref_genes = gene_names_ref if gene_names_ref is not None else np.arange(ref_mat.shape[1])
            if isinstance(reference, pd.DataFrame):
                ref_genes = reference.columns.values
                
        # 2. Parse mixture
        if isinstance(mixture, ad.AnnData):
            mix_mat = mixture.X
            mix_genes = mixture.var_names.values
            self.sample_names = mixture.obs_names.values
        elif isinstance(mixture, pd.DataFrame):
            mix_mat = mixture.values
            mix_genes = mixture.columns.values
            self.sample_names = mixture.index.values
        else:
            mix_mat = mixture
            mix_genes = gene_names_mix if gene_names_mix is not None else np.arange(mix_mat.shape[1])
            self.sample_names = sample_names_mix if sample_names_mix is not None else np.array([f"mixture_{i}" for i in range(mix_mat.shape[0])])
            
        if mix_mat.ndim == 1:
            mix_mat = mix_mat.reshape(1, -1)
            
        if sparse.issparse(ref_mat): ref_mat = ref_mat.toarray()
        if sparse.issparse(mix_mat): mix_mat = mix_mat.toarray()
        
        # Validation
        if np.min(ref_mat) < 0 or np.min(mix_mat) < 0:
            raise ValueError("Input contains negative values. Please make sure input is untransformed raw count.")
            
        # Check mapping
        df_map = pd.DataFrame({'ct': ct_labels, 'cs': cs_labels})
        if df_map.groupby('cs')['ct'].nunique().max() > 1:
            raise ValueError("One or more cell states belong to multiple cell types!")
            
        # Remove outlier genes in bulk
        keep_bulk_genes = _filter_bulk_outlier(mix_mat, outlier_cut, outlier_fraction)
        mix_mat = mix_mat[:, keep_bulk_genes]
        mix_genes = mix_genes[keep_bulk_genes]
        
        # Remove all-zero genes in reference
        keep_ref_genes = np.array(ref_mat.sum(axis=0) > 0).flatten()
        ref_mat = ref_mat[:, keep_ref_genes]
        ref_genes = ref_genes[keep_ref_genes]
        
        # Intersect genes
        shared_genes, ref_idx, mix_idx = np.intersect1d(ref_genes, mix_genes, return_indices=True)
        if len(shared_genes) == 0:
            raise ValueError("Gene names of reference and mixture do not match!")
        if len(shared_genes) < 100:
            warnings.warn("Very few genes from reference and mixture match!")
            
        ref_mat = ref_mat[:, ref_idx]
        self.mixture = mix_mat[:, mix_idx]
        self.gene_names = shared_genes
        
        print("Normalizing reference...")
        # Collapse and normalize
        ref_cs, self.cs_names = _collapse(ref_mat, cs_labels)
        ref_ct, self.ct_names = _collapse(ref_mat, ct_labels)
        
        self.phi_cellState = _norm_to_one(ref_cs, pseudo_min)
        self.phi_cellType = _norm_to_one(ref_ct, pseudo_min)
        
        # Mapping dict
        self.map_dict = df_map.drop_duplicates().groupby('ct')['cs'].apply(list).to_dict()
        
    def run(self, n_cores=1, update_gibbs=True, 
            gibbs_control=None, opt_control=None):
        """
        Run the BayesPrism deconvolution.
        """
        from .gibbs import run_gibbs, update_reference
        
        if gibbs_control is None:
            gibbs_control = {'chain_length': 1000, 'burn_in': 500, 'thinning': 2, 'alpha': 1.0, 'seed': 123}
        if opt_control is None:
            opt_control = {'maxiter': 100000, 'optimizer': 'MAP', 'sigma': 2.0}
            
        print("Starting initial Gibbs sampling...")
        # Run initial Gibbs over cell states
        Z_cs, theta_cs = run_gibbs(
            X=self.mixture, 
            phi=self.phi_cellState, 
            alpha=gibbs_control['alpha'], 
            chain_length=gibbs_control['chain_length'], 
            burn_in=gibbs_control['burn_in'], 
            thinning=gibbs_control['thinning'],
            seed=gibbs_control['seed'],
            n_cores=n_cores,
            sample_Z=True
        )
        self.posterior_initial_cellState = {'Z': Z_cs, 'theta': theta_cs}
        
        # Merge cell states to cell types
        print("Merging cell states to cell types...")
        N, G, K_cs = Z_cs.shape
        K_ct = len(self.ct_names)
        Z_ct = np.zeros((N, G, K_ct))
        theta_ct = np.zeros((N, K_ct))
        
        cs_to_idx = {name: i for i, name in enumerate(self.cs_names)}
        
        for k, ct in enumerate(self.ct_names):
            cs_list = self.map_dict[ct]
            cs_indices = [cs_to_idx[cs] for cs in cs_list]
            Z_ct[:, :, k] = Z_cs[:, :, cs_indices].sum(axis=2)
            theta_ct[:, k] = theta_cs[:, cs_indices].sum(axis=1)
            
        self.posterior_initial_cellType = {'Z': Z_ct, 'theta': theta_ct}
        
        if not update_gibbs:
            return self
            
        print("Updating reference matrix...")
        # Update Reference
        psi = update_reference(
            Z_ct=Z_ct, 
            phi_prime=self.phi_cellType, 
            ct_names=self.ct_names,
            key=self.key, 
            opt_control=opt_control,
            pseudo_min=self.pseudo_min,
            n_cores=n_cores
        )
        self.reference_update = psi
        
        print("Starting final Gibbs sampling...")
        if self.key is None:
            # psi is just a phi matrix
            _, theta_f = run_gibbs(
                X=self.mixture, 
                phi=psi, 
                alpha=gibbs_control['alpha'], 
                chain_length=gibbs_control['chain_length'], 
                burn_in=gibbs_control['burn_in'], 
                thinning=gibbs_control['thinning'],
                seed=gibbs_control['seed'],
                n_cores=n_cores,
                sample_Z=False
            )
        else:
            # psi is a dict with psi_mal (N x G) and psi_env (K-1 x G)
            theta_f = np.zeros((N, K_ct))
            # we need a specialized final gibbs for tumor case
            from .gibbs import run_gibbs_tumor
            theta_f = run_gibbs_tumor(
                X=self.mixture,
                psi_mal=psi['psi_mal'],
                psi_env=psi['psi_env'],
                alpha=gibbs_control['alpha'], 
                chain_length=gibbs_control['chain_length'], 
                burn_in=gibbs_control['burn_in'], 
                thinning=gibbs_control['thinning'],
                seed=gibbs_control['seed'],
                n_cores=n_cores
            )
            
        self.posterior_theta_f = theta_f
        print("Done.")
        return self
        
    def get_results(self, output_type="anndata"):
        """
        Returns the deconvolution results. 
        output_type: "anndata" or "dataframe"
        """
        if hasattr(self, 'posterior_theta_f'):
            res_theta = self.posterior_theta_f
            res_Z = None # Z is not re-sampled in final
        else:
            res_theta = self.posterior_initial_cellType['theta']
            res_Z = self.posterior_initial_cellType['Z']
            
        df = pd.DataFrame(res_theta, index=self.sample_names, columns=self.ct_names)
        
        if output_type == "dataframe":
            return df
        elif output_type == "anndata":
            # Return an AnnData where X is the mixture, and obs contains theta
            adata = ad.AnnData(X=self.mixture, obs=df)
            adata.var_names = self.gene_names
            return adata

    def get_exp(self, state_or_type, cell_name):
        """
        Extract posterior mean of sample specific gene expression profile for the given cell type or state.
        Returns a N x G matrix.
        """
        if state_or_type == "state":
            idx = list(self.cs_names).index(cell_name)
            if hasattr(self, 'posterior_cellState'):
                return self.posterior_cellState['Z'][:, :, idx]
            return self.posterior_initial_cellState['Z'][:, :, idx]
        elif state_or_type == "type":
            idx = list(self.ct_names).index(cell_name)
            if hasattr(self, 'posterior_cellType'):
                return self.posterior_cellType['Z'][:, :, idx]
            return self.posterior_initial_cellType['Z'][:, :, idx]
        else:
            raise ValueError("state_or_type must be 'state' or 'type'")

    def run_st(self, n_cores=1, gibbs_control=None, opt_control=None):
        """
        Run the BayesPrism deconvolution for spatial transcriptomics.
        """
        from .gibbs import run_gibbs, update_reference
        
        if gibbs_control is None:
            gibbs_control = {'chain_length': 1000, 'burn_in': 500, 'thinning': 2, 'alpha': 1.0, 'seed': 123}
        if opt_control is None:
            opt_control = {'maxiter': 100000, 'optimizer': 'MAP', 'sigma': 2.0}
            
        print("Starting initial Gibbs sampling...")
        Z_cs, theta_cs = run_gibbs(
            X=self.mixture, 
            phi=self.phi_cellState, 
            alpha=gibbs_control['alpha'], 
            chain_length=gibbs_control['chain_length'], 
            burn_in=gibbs_control['burn_in'], 
            thinning=gibbs_control['thinning'],
            seed=gibbs_control['seed'],
            n_cores=n_cores,
            sample_Z=True
        )
        self.posterior_initial_cellState = {'Z': Z_cs, 'theta': theta_cs}
        
        print("Updating reference matrix...")
        # ST does MAP estimation for cell STATES (no tumor MLE)
        psi = update_reference(
            Z_ct=Z_cs, 
            phi_prime=self.phi_cellState, 
            ct_names=self.cs_names,
            key=None, 
            opt_control=opt_control,
            pseudo_min=self.pseudo_min,
            n_cores=n_cores
        )
        self.reference_update = psi
        
        print("Starting final Gibbs sampling...")
        Z_cs_f, theta_cs_f = run_gibbs(
            X=self.mixture, 
            phi=psi, 
            alpha=gibbs_control['alpha'], 
            chain_length=gibbs_control['chain_length'], 
            burn_in=gibbs_control['burn_in'], 
            thinning=gibbs_control['thinning'],
            seed=gibbs_control['seed'],
            n_cores=n_cores,
            sample_Z=True
        )
        
        self.posterior_cellState = {'Z': Z_cs_f, 'theta': theta_cs_f}
        
        print("Merging cell states to cell types...")
        N, G, K_cs = Z_cs_f.shape
        K_ct = len(self.ct_names)
        Z_ct = np.zeros((N, G, K_ct))
        theta_ct = np.zeros((N, K_ct))
        
        cs_to_idx = {name: i for i, name in enumerate(self.cs_names)}
        
        for k, ct in enumerate(self.ct_names):
            cs_list = self.map_dict[ct]
            cs_indices = [cs_to_idx[cs] for cs in cs_list]
            Z_ct[:, :, k] = Z_cs_f[:, :, cs_indices].sum(axis=2)
            theta_ct[:, k] = theta_cs_f[:, cs_indices].sum(axis=1)
            
        self.posterior_cellType = {'Z': Z_ct, 'theta': theta_ct}
        
        self.posterior_theta_f = theta_ct
        print("Done.")
        return self

    def learn_embedding_nmf(self, K, cycle=50, n_cores=1, gibbs_control=None, opt_control=None, nmf_control=None):
        from .embedding import learn_embedding_nmf
        return learn_embedding_nmf(self, K, cycle, n_cores, gibbs_control, opt_control, nmf_control)
