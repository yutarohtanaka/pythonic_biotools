import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.sparse import issparse

def _get_counts(adata, layer=None):
    """Utility function to extract count matrix from AnnData."""
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if issparse(X):
        return X.toarray()
    return np.array(X)

def music_basis(adata, cluster_col, sample_col, layer=None, markers=None, select_ct=None, cell_sizes=None, ct_cov=False):
    """
    Computes the MuSiC basis including Design matrix, Library sizes, and cross-subject variances.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data object. Rows are cells, columns are genes.
    cluster_col : str
        Column name in adata.obs containing cell type annotations.
    sample_col : str
        Column name in adata.obs containing sample/subject annotations.
    layer : str, optional
        Layer in adata to use for raw counts. If None, uses adata.X.
    markers : list, optional
        List of genes to use.
    select_ct : list, optional
        List of cell types to include.
    cell_sizes : dict, optional
        Dictionary mapping cell type to cell size. If provided, these are used instead of estimating from data.
    ct_cov : bool, optional
        If True, calculates the cross-cell-type covariance matrix. Default False.
        
    Returns
    -------
    dict
        Contains Design matrix, Sigma, S, M_S, and M_theta as pandas DataFrames/Series.
    """
    X_counts = _get_counts(adata, layer)
    obs = adata.obs
    
    clusters = obs[cluster_col].astype(str).values
    samples = obs[sample_col].astype(str).values
    
    unique_clusters = np.unique(clusters) if select_ct is None else np.array(select_ct)
    unique_samples = np.unique(samples)
    
    genes = adata.var_names.values
    
    # Pre-allocate theta matrix: (n_samples, n_clusters, n_genes)
    theta_all = np.full((len(unique_samples), len(unique_clusters), len(genes)), np.nan)
    S_mtx = np.full((len(unique_samples), len(unique_clusters)), np.nan)
    
    for j, ct in enumerate(unique_clusters):
        for i, sid in enumerate(unique_samples):
            mask = (clusters == ct) & (samples == sid)
            n_cells = np.sum(mask)
            if n_cells > 0:
                y = X_counts[mask, :]
                y_sum = np.sum(y)
                if y_sum > 0:
                    theta_all[i, j, :] = np.sum(y, axis=0) / y_sum
                else:
                    theta_all[i, j, :] = 0.0
                S_mtx[i, j] = y_sum / n_cells

    M_theta = np.nanmean(theta_all, axis=0) # (n_clusters, n_genes)
    M_theta = M_theta.T # (n_genes, n_clusters)
    
    # Sigma calculation
    with np.errstate(invalid='ignore', divide='ignore'):
        # (n_genes, n_clusters)
        Sigma = np.nanvar(theta_all, axis=0, ddof=1).T
    
    Sigma_ct_res = None
    if ct_cov:
        Sigma_ct_list = []
        for g in range(len(genes)):
            theta_g = theta_all[:, :, g] # (n_samples, n_clusters)
            cov_mat = pd.DataFrame(theta_g).cov().values
            Sigma_ct_list.append(cov_mat)
        Sigma_ct_res = np.array(Sigma_ct_list) # (n_genes, n_clusters, n_clusters)
        
    M_S = np.nanmean(S_mtx, axis=0)
    
    if cell_sizes is not None:
        for i, ct in enumerate(unique_clusters):
            if ct in cell_sizes:
                M_S[i] = cell_sizes[ct]
    
    D = M_theta * M_S
    
    D_df = pd.DataFrame(D, index=genes, columns=unique_clusters)
    M_theta_df = pd.DataFrame(M_theta, index=genes, columns=unique_clusters)
    Sigma_df = pd.DataFrame(Sigma, index=genes, columns=unique_clusters)
    S_df = pd.DataFrame(S_mtx, index=unique_samples, columns=unique_clusters)
    M_S_series = pd.Series(M_S, index=unique_clusters)
    
    if markers is not None:
        markers = [m for m in markers if m in genes]
        D_df = D_df.loc[markers]
        M_theta_df = M_theta_df.loc[markers]
        Sigma_df = Sigma_df.loc[markers]
        if ct_cov:
            marker_indices = [np.where(genes == m)[0][0] for m in markers]
            Sigma_ct_res = Sigma_ct_res[marker_indices, :, :]
            
    res = {
        'Design_mtx': D_df,
        'M_theta': M_theta_df,
        'Sigma': Sigma_df,
        'S': S_df,
        'M_S': M_S_series
    }
    if ct_cov:
        res['Sigma_ct'] = Sigma_ct_res
        
    return res

def music_iter(Y, D, S, Sigma, Sigma_ct=None, iter_max=1000, nu=0.0001, eps=0.01, centered=False, normalize=False):
    """
    Core iteration of MuSiC algorithm: weighted NNLS.
    """
    if centered:
        D = D - np.mean(D)
        Y = Y - np.mean(Y)
    
    if normalize:
        D_std = np.std(D, ddof=1)
        D = D / D_std
        S = S * D_std
        Y = Y / np.std(Y, ddof=1)
    else:
        Y = Y * 100
        
    p_nnls, _ = nnls(D, Y)
    r = Y - D @ p_nnls
    
    if Sigma_ct is not None:
        Sp = p_nnls * S
        cov_term = np.einsum('i,gij,j->g', Sp, np.nan_to_num(Sigma_ct), Sp)
    else:
        cov_term = np.sum(((p_nnls * S)**2) * Sigma, axis=1)
        
    weight_gene = 1 / (nu + r**2 + cov_term)
    
    Y_weight = Y * np.sqrt(weight_gene)
    D_weight = D * np.sqrt(weight_gene)[:, None]
    
    p_weight, _ = nnls(D_weight, Y_weight)
    r = Y - D @ p_weight
    
    for i in range(iter_max):
        if Sigma_ct is not None:
            Sp = p_weight * S
            cov_term = np.einsum('i,gij,j->g', Sp, np.nan_to_num(Sigma_ct), Sp)
        else:
            cov_term = np.sum(((p_weight * S)**2) * Sigma, axis=1)
            
        weight_gene = 1 / (nu + r**2 + cov_term)
        Y_weight = Y * np.sqrt(weight_gene)
        D_weight = D * np.sqrt(weight_gene)[:, None]
        
        p_weight_new, _ = nnls(D_weight, Y_weight)
        r_new = Y - D @ p_weight_new
        
        if np.sum(np.abs(p_weight_new - p_weight)) < eps:
            p_weight = p_weight_new
            r = r_new
            break
            
        p_weight = p_weight_new
        r = r_new
        
    fitted = D @ p_weight
    with np.errstate(invalid='ignore', divide='ignore'):
        var_Y = np.var(Y, ddof=1)
        if var_Y > 0:
            R_squared = 1 - np.var(Y - fitted, ddof=1) / var_Y
        else:
            R_squared = np.nan
            
    p_weight_sum = np.sum(p_weight)
    p_nnls_sum = np.sum(p_nnls)
    
    p_weight_norm = p_weight / p_weight_sum if p_weight_sum > 0 else p_weight
    p_nnls_norm = p_nnls / p_nnls_sum if p_nnls_sum > 0 else p_nnls
    
    return p_weight_norm, p_nnls_norm, R_squared, weight_gene

def music_prop(bulk_adata, sc_adata, cluster_col, sample_col, sc_layer=None, bulk_layer=None, 
               markers=None, select_ct=None, cell_sizes=None, ct_cov=False, iter_max=1000, nu=0.0001, eps=0.01, 
               centered=False, normalize=False):
    """
    Runs MuSiC deconvolution.
    """
    basis = music_basis(sc_adata, cluster_col, sample_col, layer=sc_layer, 
                        markers=markers, select_ct=select_ct, cell_sizes=cell_sizes, ct_cov=ct_cov)
    
    D_df = basis['Design_mtx']
    Sigma_df = basis['Sigma']
    M_S_series = basis['M_S']
    Sigma_ct = basis.get('Sigma_ct', None)
    
    valid_ct = (~Sigma_df.isna().any(axis=0)) & (~D_df.isna().any(axis=0)) & (~M_S_series.isna())
    valid_ct = valid_ct[valid_ct].index
    
    if len(valid_ct) <= 1:
        raise ValueError("Not enough valid cell types without NaN values! Ensure each cell type exists across multiple samples.")
        
    valid_idx = [D_df.columns.get_loc(c) for c in valid_ct]
    
    D_df = D_df[valid_ct]
    Sigma_df = Sigma_df[valid_ct]
    M_S_series = M_S_series[valid_ct]
    if Sigma_ct is not None:
        Sigma_ct = Sigma_ct[:, valid_idx, :][:, :, valid_idx]
    
    bulk_genes = bulk_adata.var_names
    sc_genes = D_df.index
    
    common_genes = list(set(bulk_genes).intersection(sc_genes))
    if markers is not None:
        common_genes = list(set(common_genes).intersection(markers))
        
    if len(common_genes) < 0.2 * min(len(bulk_genes), len(sc_genes)):
        raise ValueError("Too few common genes between bulk and single-cell reference!")
        
    common_genes_idx = [D_df.index.get_loc(g) for g in common_genes]
    D_df = D_df.iloc[common_genes_idx]
    Sigma_df = Sigma_df.iloc[common_genes_idx]
    if Sigma_ct is not None:
        Sigma_ct = Sigma_ct[common_genes_idx, :, :]
    
    bulk_X = _get_counts(bulk_adata, bulk_layer)
    bulk_gene_idx = [bulk_adata.var_names.get_loc(g) for g in common_genes]
    bulk_X_common = bulk_X[:, bulk_gene_idx]
    
    bulk_sums = np.sum(bulk_X_common, axis=1, keepdims=True)
    bulk_sums[bulk_sums == 0] = 1
    Y_ra = bulk_X_common / bulk_sums
    
    res_p_weight = []
    res_p_nnls = []
    res_r2 = []
    
    D_mat = D_df.values
    Sigma_mat = Sigma_df.values
    S_vec = M_S_series.values
    
    for i in range(Y_ra.shape[0]):
        Y_i = Y_ra[i, :]
        mask = Y_i != 0
        if np.sum(mask) == 0:
            res_p_weight.append(np.full(len(valid_ct), np.nan))
            res_p_nnls.append(np.full(len(valid_ct), np.nan))
            res_r2.append(np.nan)
            continue
            
        Y_i_sub = Y_i[mask]
        D_mat_sub = D_mat[mask, :]
        Sigma_mat_sub = Sigma_mat[mask, :]
        Sigma_ct_sub = Sigma_ct[mask, :, :] if Sigma_ct is not None else None
        
        pw, pnnls, r2, _ = music_iter(
            Y_i_sub, D_mat_sub, S_vec, Sigma_mat_sub, Sigma_ct=Sigma_ct_sub,
            iter_max=iter_max, nu=nu, eps=eps, centered=centered, normalize=normalize
        )
        
        res_p_weight.append(pw)
        res_p_nnls.append(pnnls)
        res_r2.append(r2)
        
    df_weight = pd.DataFrame(res_p_weight, index=bulk_adata.obs_names, columns=valid_ct)
    df_nnls = pd.DataFrame(res_p_nnls, index=bulk_adata.obs_names, columns=valid_ct)
    
    return {
        'Est_prop_weighted': df_weight,
        'Est_prop_allgene': df_nnls,
        'R_squared': pd.Series(res_r2, index=bulk_adata.obs_names)
    }
