import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .music import music_prop, _get_counts
import warnings

def music2_prop(bulk_control_adata, bulk_case_adata, sc_adata, cluster_col, sample_col, 
                sc_layer=None, bulk_layer=None, select_ct=None, markers=None, cell_sizes=None, ct_cov=False,
                expr_low=20, prop_r=0.1, eps_c=0.05, eps_r=0.01, n_resample=20, sample_prop=0.5, 
                cutoff_expr=0.05, cutoff_c=0.05, cutoff_r=0.01, maxiter=200, 
                iter_max_music=1000, nu=0.0001, eps_music=0.01, centered=False, normalize=False):
    """
    MuSiC2 Deconvolution
    
    This function is used to deconvolve bulk RNA-seq data using single-cell reference generated 
    under a different condition. It identifies and removes differentially expressed genes between 
    clinical conditions to refine the deconvolution iteratively.
    
    Parameters
    ----------
    bulk_control_adata : AnnData
        Bulk RNA-seq data for control group. Rows are samples, columns are genes.
    bulk_case_adata : AnnData
        Bulk RNA-seq data for case group. Rows are samples, columns are genes.
    sc_adata : AnnData
        Single-cell reference. Rows are cells, columns are genes.
    cluster_col : str
        Column name in sc_adata.obs for cell types.
    sample_col : str
        Column name in sc_adata.obs for samples/subjects.
    sc_layer : str, optional
        Layer in sc_adata for raw counts. None uses sc_adata.X.
    bulk_layer : str, optional
        Layer in bulk_*_adata for raw counts. None uses bulk_*_adata.X.
    select_ct : list, optional
        List of cell types to include.
    markers : list, optional
        List of marker genes to restrict analysis to.
    cell_sizes : dict, optional
        Pre-computed cell sizes.
    ct_cov : bool, optional
        Calculate cross-cell-type covariance. Default False.
    expr_low : numeric, optional
        Cutoff for defining lowly expressed genes in bulk data. Default 20.
    prop_r : numeric, optional
        Cutoff for defining rare cell types. Default 0.1.
    eps_c : numeric, optional
        Convergence cutoff for common cell types. Default 0.05.
    eps_r : numeric, optional
        Convergence cutoff for rare cell types. Default 0.01.
    n_resample : int, optional
        Number of resamples used for detecting cell-type-specific DE genes. Default 20.
    sample_prop : float, optional
        Proportion of samples to be randomly sampled without replacement. Default 0.5.
    cutoff_expr : float, optional
        Cutoff for defining lowly expressed genes over resamples. Default 0.05.
    cutoff_c : float, optional
        Cutoff for defining cell-type-specific DE genes for common cell types. Default 0.05.
    cutoff_r : float, optional
        Cutoff for defining cell-type-specific DE genes for rare cell types. Default 0.01.
    maxiter : int, optional
        Maximum number of iterations. Default 200.
        
    Returns
    -------
    dict
        Contains 'Est_prop', 'convergence', 'n_iter', 'DE_genes' (or 'id_not_converge' if failed).
    """
    
    bulk_control_genes = bulk_control_adata.var_names
    bulk_case_genes = bulk_case_adata.var_names
    sc_genes = sc_adata.var_names
    
    gene_bulk = list(set(bulk_control_genes).intersection(bulk_case_genes))
    if len(gene_bulk) < 0.1 * min(len(bulk_control_genes), len(bulk_case_genes)):
        raise ValueError("Not enough common genes between bulk control and case data!")
        
    gene_all = list(set(gene_bulk).intersection(sc_genes))
    if len(gene_all) < 0.2 * min(len(gene_bulk), len(sc_genes)):
        raise ValueError("Not enough genes between bulk and single-cell data!")
        
    if markers is not None:
        gene_all = list(set(gene_all).intersection(markers))
        
    bulk_control_X = _get_counts(bulk_control_adata[:, gene_all], layer=bulk_layer)
    bulk_case_X = _get_counts(bulk_case_adata[:, gene_all], layer=bulk_layer)
    
    bulk_mtx_all = np.vstack([bulk_control_X, bulk_case_X])
    expr = np.mean(bulk_mtx_all, axis=0)
    exp_genel = np.array(gene_all)[expr >= expr_low]
    
    # Initial cell type deconvolution
    res_control = music_prop(bulk_control_adata, sc_adata, cluster_col, sample_col, 
                             sc_layer=sc_layer, bulk_layer=bulk_layer, markers=gene_all, 
                             select_ct=select_ct, cell_sizes=cell_sizes, ct_cov=ct_cov,
                             iter_max=iter_max_music, nu=nu, eps=eps_music, centered=centered, normalize=normalize)
    prop_control = res_control['Est_prop_weighted']
    
    res_case_ini = music_prop(bulk_case_adata, sc_adata, cluster_col, sample_col, 
                              sc_layer=sc_layer, bulk_layer=bulk_layer, markers=gene_all, 
                              select_ct=select_ct, cell_sizes=cell_sizes, ct_cov=ct_cov,
                              iter_max=iter_max_music, nu=nu, eps=eps_music, centered=centered, normalize=normalize)
    prop_case_ini = res_case_ini['Est_prop_weighted']
    
    prop_case_fix = pd.DataFrame(columns=prop_control.columns)
    
    ncell = prop_control.shape[1]
    cell_types = prop_control.columns
    
    iter_num = 1
    id_conv = []
    
    all_converge = False
    lr_all = []
    
    while iter_num <= maxiter:
        prop_all = pd.concat([prop_control, prop_case_ini, prop_case_fix])
        
        n_ctrl_samples = bulk_control_X.shape[0]
        n_case_samples = bulk_case_X.shape[0]
        
        MOD0 = np.zeros((ncell, len(exp_genel)))
        MOD1 = np.zeros((ncell, len(exp_genel)))
        LOGFC = []
        
        exp_genel_idx = [gene_all.index(g) for g in exp_genel]
        
        for i in range(n_resample):
            id_h_idx = np.random.choice(n_ctrl_samples, int(np.round(n_ctrl_samples * sample_prop)), replace=False)
            control_s = bulk_control_X[id_h_idx][:, exp_genel_idx]
            prop_h = prop_control.iloc[id_h_idx].values
            
            mod0 = np.zeros((ncell, len(exp_genel)))
            for g in range(len(exp_genel)):
                x_g = control_s[:, g]
                p, _ = nnls(prop_h, x_g)
                mod0[:, g] = p
            MOD0 += mod0
            
            id_d_idx = np.random.choice(n_case_samples, int(np.round(n_case_samples * sample_prop)), replace=False)
            case_s = bulk_case_X[id_d_idx][:, exp_genel_idx]
            sampled_case_names = bulk_case_adata.obs_names[id_d_idx]
            prop_case_combined = pd.concat([prop_case_ini, prop_case_fix])
            prop_d = prop_case_combined.loc[sampled_case_names].values
            
            mod1 = np.zeros((ncell, len(exp_genel)))
            for g in range(len(exp_genel)):
                x_g = case_s[:, g]
                p, _ = nnls(prop_d, x_g)
                mod1[:, g] = p
            MOD1 += mod1
            
            LOGFC.append(np.log1p(mod1) - np.log1p(mod0))
            
        MOD0 /= n_resample
        MOD1 /= n_resample
        
        LOGFC = np.array(LOGFC)
        mean_logfc = np.mean(LOGFC, axis=0)
        sd_logfc = np.std(LOGFC, axis=0, ddof=1)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            rcv = np.where(mean_logfc == 0, 0, mean_logfc / sd_logfc)
        abs_rcv_logfc = np.abs(rcv)
        
        mex = prop_all.mean(axis=0)
        
        lr = []
        for c_idx, celltype in enumerate(cell_types):
            m = mex[celltype]
            rh = MOD0[c_idx, :]
            rd = MOD1[c_idx, :]
            
            thresh_rh = np.percentile(rh, cutoff_expr * 100)
            thresh_rd = np.percentile(rd, cutoff_expr * 100)
            
            llr = np.intersect1d(exp_genel[rh <= thresh_rh], exp_genel[rd <= thresh_rd])
            
            x = abs_rcv_logfc[c_idx, :]
            valid_mask = ~np.isin(exp_genel, llr)
            x_valid = x[valid_mask]
            genes_valid = exp_genel[valid_mask]
            
            if len(x_valid) > 0:
                if m >= prop_r:
                    thresh = np.percentile(x_valid[~np.isnan(x_valid)], (1 - cutoff_c) * 100)
                else:
                    thresh = np.percentile(x_valid[~np.isnan(x_valid)], (1 - cutoff_r) * 100)
                    
                lr.extend(genes_valid[x_valid >= thresh])
                
        lr = list(np.unique(lr))
        lr_all = list(np.unique(lr_all + lr))
        
        l_genes = list(set(gene_all) - set(lr))
        
        unconverged_cases = prop_case_ini.index
        case_sample_adata = bulk_case_adata[unconverged_cases, :]
        
        res_case_new = music_prop(case_sample_adata, sc_adata, cluster_col, sample_col,
                                  sc_layer=sc_layer, bulk_layer=bulk_layer, markers=l_genes,
                                  select_ct=select_ct, cell_sizes=cell_sizes, ct_cov=ct_cov,
                                  iter_max=iter_max_music, nu=nu, eps=eps_music, centered=centered, normalize=normalize)
                                  
        prop_case_new = res_case_new['Est_prop_weighted']
        
        pc = np.abs(prop_case_new - prop_case_ini)
        conv = np.ones_like(pc)
        
        for c_idx, celltype in enumerate(cell_types):
            ini_vals = prop_case_ini[celltype].values
            rare_mask = ini_vals <= prop_r
            common_mask = ini_vals > prop_r
            
            conv[rare_mask, c_idx] = np.where(pc.values[rare_mask, c_idx] < eps_r, 0, 1)
            
            with np.errstate(invalid='ignore', divide='ignore'):
                pct_change = pc.values[common_mask, c_idx] / ini_vals[common_mask]
            conv[common_mask, c_idx] = np.where(pct_change < eps_c, 0, 1)
            
        convf = np.all(conv == 0, axis=1)
        
        new_conv_ids = prop_case_new.index[convf].tolist()
        id_conv.extend(new_conv_ids)
        
        prop_CASE = pd.concat([prop_case_new, prop_case_fix])
        
        prop_case_ini = prop_CASE.loc[~prop_CASE.index.isin(id_conv)]
        prop_case_fix = prop_CASE.loc[prop_CASE.index.isin(id_conv)]
        
        if len(prop_case_ini) == 0:
            all_converge = True
            break
            
        iter_num += 1
        
    prop_all_final = pd.concat([prop_control, prop_case_ini, prop_case_fix])
    
    if all_converge:
        return {
            'Est_prop': prop_all_final,
            'convergence': True,
            'n_iter': iter_num,
            'DE_genes': lr_all
        }
    else:
        return {
            'Est_prop': prop_all_final,
            'convergence': False,
            'id_not_converge': prop_case_ini.index.tolist()
        }
