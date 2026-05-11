import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def plot_cor_phi(bp, min_exp=3, cmap="RdBu_r", title="Cell Type Correlation"):
    """
    Plots the heatmap for the correlation between each cell type from the reference.
    """
    ref_ct = bp.phi_cellType.copy()
    
    # Filter lowly expressed genes
    keep_genes = ref_ct.sum(axis=0) >= min_exp
    ref_ct = ref_ct[:, keep_genes]
    
    # Center and scale
    ref_ct_log = np.log2(ref_ct + bp.pseudo_min)
    ref_ct_centered = ref_ct_log - ref_ct_log.mean(axis=0)
    
    cor_mat = np.corrcoef(ref_ct_centered)
    
    sns.clustermap(
        cor_mat,
        xticklabels=bp.ct_names,
        yticklabels=bp.ct_names,
        cmap=cmap,
        method='ward',
        metric='euclidean',
        figsize=(8, 8)
    )
    plt.title(title)
    plt.show()

def plot_bulk_vs_sc(bp):
    """
    Makes a scatter plot of bulk total expression vs scRNA-seq total expression 
    and shows pearson, spearman correlation coef and MSE.
    """
    # Total expression in scRNA (sum of reference)
    sc_total = bp.phi_cellType.sum(axis=0)
    sc_norm = sc_total / sc_total.sum()
    
    # Total expression in bulk
    bulk_total = bp.mixture.sum(axis=0)
    bulk_norm = bulk_total / bulk_total.sum()
    
    log_sc = np.log2(np.maximum(sc_norm, 1e-15))
    log_bulk = np.log2(np.maximum(bulk_norm, 1e-15))
    
    cor_ps, _ = pearsonr(log_sc, log_bulk)
    cor_sp, _ = spearmanr(log_sc, log_bulk)
    mse = np.mean((log_sc - log_bulk)**2)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(log_sc, log_bulk, s=5, alpha=0.5, c='gray')
    
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'r--')
    
    plt.text(xlim[0] + 0.75*(xlim[1]-xlim[0]), ylim[0] + 0.3*(ylim[1]-ylim[0]), f"R={cor_ps:.3f}")
    plt.text(xlim[0] + 0.75*(xlim[1]-xlim[0]), ylim[0] + 0.2*(ylim[1]-ylim[0]), f"rho={cor_sp:.3f}")
    plt.text(xlim[0] + 0.75*(xlim[1]-xlim[0]), ylim[0] + 0.1*(ylim[1]-ylim[0]), f"MSE={mse:.3g}")
    
    plt.xlabel("log2 total expression in scRNA-seq")
    plt.ylabel("log2 total expression in bulk")
    plt.title("Bulk vs scRNA-seq")
    plt.show()
