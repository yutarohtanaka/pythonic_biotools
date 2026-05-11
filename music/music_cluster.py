import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .music import music_basis, music_iter, _get_counts
import warnings

def music_prop_cluster(bulk_adata, sc_adata, group_markers, groups_col, clusters_col, samples_col, clusters_type,
                       sc_layer=None, bulk_layer=None, iter_max=1000, nu=0.0001, eps=0.01, centered=False, normalize=False):
    """
    MuSiC Deconvolution with Clusters
    
    This function calculates the MuSiC deconvolution proportions using a hierarchical approach with clusters.
    
    Parameters
    ----------
    bulk_adata : AnnData
        Bulk RNA-seq data. Rows are samples, columns are genes.
    sc_adata : AnnData
        Single-cell reference. Rows are cells, columns are genes.
    group_markers : dict
        Dictionary mapping group names to lists of differential expressed genes within groups.
        Keys must match keys of `clusters_type`.
    groups_col : str
        Column name in sc_adata.obs for cluster groups.
    clusters_col : str
        Column name in sc_adata.obs for specific cell types (sub-clusters).
    samples_col : str
        Column name in sc_adata.obs for samples/subjects.
    clusters_type : dict
        Dictionary mapping group names to lists of specific cell types (sub-clusters).
    sc_layer : str, optional
        Layer in sc_adata for raw counts. None uses sc_adata.X.
    bulk_layer : str, optional
        Layer in bulk_adata for raw counts. None uses bulk_adata.X.
        
    Returns
    -------
    dict
        Dictionary containing 'Est_prop_weighted_cluster' as Pandas DataFrame.
    """
    if set(group_markers.keys()) != set(clusters_type.keys()):
        raise ValueError("Keys for group_markers and clusters_type must match!")
        
    group_names = list(clusters_type.keys())
    sub_cluster_names = []
    for g in group_names:
        sub_cluster_names.extend(clusters_type[g])
        
    # Get basis for groups
    cluster_sc_basis = music_basis(sc_adata, groups_col, samples_col, layer=sc_layer, select_ct=group_names)
    
    # Get basis for sub-clusters
    sc_basis = music_basis(sc_adata, clusters_col, samples_col, layer=sc_layer, select_ct=sub_cluster_names)
    
    # Genes handling
    bulk_genes = bulk_adata.var_names
    sc_genes = sc_basis['Design_mtx'].index
    cm_gene = list(set(bulk_genes).intersection(sc_genes))
    
    if len(cm_gene) < 0.2 * min(len(bulk_genes), len(sc_genes)):
        raise ValueError("Too few common genes!")
        
    # Intersect group markers with common genes
    group_markers = {k: list(set(v).intersection(cm_gene)) for k, v in group_markers.items()}
    
    D1 = sc_basis['Design_mtx'].loc[cm_gene]
    M_S = sc_basis['M_S']
    Sigma = sc_basis['Sigma'].loc[cm_gene]
    
    all_markers = []
    for m in group_markers.values():
        all_markers.extend(m)
    cluster_diff = list(set(all_markers))
    cluster_select = list(set(cm_gene) - set(cluster_diff))
    
    D1_cluster = cluster_sc_basis['Design_mtx'].loc[cluster_select, group_names]
    M_S_cluster = cluster_sc_basis['M_S'][group_names]
    Sigma_cluster = cluster_sc_basis['Sigma'].loc[cluster_select, group_names]
    
    D1_sub = cluster_sc_basis['Design_mtx'].loc[cluster_diff, group_names]
    Sigma_sub = cluster_sc_basis['Sigma'].loc[cluster_diff, group_names]
    
    bulk_X = _get_counts(bulk_adata, bulk_layer)
    bulk_gene_idx = [bulk_adata.var_names.get_loc(g) for g in cm_gene]
    bulk_X_common = bulk_X[:, bulk_gene_idx]
    
    bulk_sums = np.sum(bulk_X_common, axis=1, keepdims=True)
    bulk_sums[bulk_sums == 0] = 1
    Y_ra = bulk_X_common / bulk_sums
    
    gene_to_idx = {g: i for i, g in enumerate(cm_gene)}
    
    res_p_weight_cluster = []
    
    for i in range(Y_ra.shape[0]):
        Y_i = Y_ra[i, :]
        mask = Y_i != 0
        if np.sum(mask) == 0:
            res_p_weight_cluster.append(np.full(len(sub_cluster_names), np.nan))
            continue
            
        valid_genes = [cm_gene[idx] for idx in range(len(cm_gene)) if mask[idx]]
        
        valid_cluster_select = [g for g in cluster_select if g in valid_genes]
        D1_cluster_temp = D1_cluster.loc[valid_cluster_select].values
        Y_cluster_temp = Y_i[[gene_to_idx[g] for g in valid_cluster_select]]
        Sigma_cluster_temp = Sigma_cluster.loc[valid_cluster_select].values
        
        pw_cluster, _, _, _ = music_iter(
            Y_cluster_temp, D1_cluster_temp, M_S_cluster.values, Sigma_cluster_temp,
            iter_max=iter_max, nu=nu, eps=eps, centered=centered, normalize=normalize
        )
        
        p_weight = []
        for j, g_name in enumerate(group_names):
            sub_ct = clusters_type[g_name]
            if len(sub_ct) == 1:
                p_weight.append(pw_cluster[j])
            else:
                if pw_cluster[j] == 0:
                    p_weight.extend([0.0] * len(sub_ct))
                else:
                    c_marker = [g for g in group_markers[g_name] if g in valid_genes]
                    
                    if len(c_marker) == 0:
                        p_weight.extend([pw_cluster[j] / len(sub_ct)] * len(sub_ct))
                        continue
                        
                    Y_cm = Y_i[[gene_to_idx[g] for g in c_marker]]
                    D1_sub_temp = D1_sub.loc[c_marker].values
                    
                    term1 = D1_sub_temp[:, j] * pw_cluster[j]
                    term2 = (Y_cm - D1_sub_temp @ pw_cluster) * pw_cluster[j]
                    
                    Y_sub = term1 + term2
                    
                    mask_sub = Y_sub > 0
                    if np.sum(mask_sub) == 0:
                        p_weight.extend([pw_cluster[j] / len(sub_ct)] * len(sub_ct))
                        continue
                        
                    Y_sub_filt = Y_sub[mask_sub]
                    c_marker_filt = [c_marker[idx] for idx in range(len(c_marker)) if mask_sub[idx]]
                    
                    D1_temp = D1.loc[c_marker_filt, sub_ct].values
                    M_S_temp = M_S[sub_ct].values
                    Sigma_temp = Sigma.loc[c_marker_filt, sub_ct].values
                    
                    pw_sub, _, _, _ = music_iter(
                        Y_sub_filt, D1_temp, M_S_temp, Sigma_temp,
                        iter_max=iter_max, nu=nu, eps=eps, centered=centered, normalize=normalize
                    )
                    
                    sub_weights = pw_cluster[j] * pw_sub
                    p_weight.extend(sub_weights.tolist())
                    
        res_p_weight_cluster.append(p_weight)
        
    df_weight_cluster = pd.DataFrame(res_p_weight_cluster, index=bulk_adata.obs_names, columns=sub_cluster_names)
    
    return {
        'Est_prop_weighted_cluster': df_weight_cluster
    }
