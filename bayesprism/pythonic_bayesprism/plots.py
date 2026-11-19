import numpy as np
from scipy.stats import pearsonr, spearmanr


def plot_cor_phi(
    bp, min_exp=3, cmap="RdBu_r", title="Cell Type Correlation", show=True
):
    """
    Plot hierarchical clustermap of pairwise correlations across reference cell types.

    Imports matplotlib and seaborn lazily so importing pythonic_bayesprism works
    in headless environments without plotting libraries installed.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    ref_ct = bp.phi_cellType.copy()

    # Filter lowly expressed genes using raw collapsed reference counts if available;
    # otherwise skip raw count thresholding on normalized phi (whose rows sum to 1).
    if min_exp > 0 and hasattr(bp, "ref_ct_raw") and bp.ref_ct_raw is not None:
        keep_genes = bp.ref_ct_raw.sum(axis=0) >= min_exp
        if np.sum(keep_genes) >= 2:
            ref_ct = ref_ct[:, keep_genes]

    ref_ct_log = np.log2(ref_ct + bp.pseudo_min)
    ref_ct_centered = ref_ct_log - ref_ct_log.mean(axis=0, keepdims=True)

    cor_mat = np.corrcoef(ref_ct_centered)

    grid = sns.clustermap(
        cor_mat,
        xticklabels=bp.ct_names,
        yticklabels=bp.ct_names,
        cmap=cmap,
        method="ward",
        metric="euclidean",
        figsize=(8, 8),
    )
    grid.fig.suptitle(title)
    if show:
        plt.show()
    return grid


def plot_bulk_vs_sc(bp, show=True):
    """
    Scatter plot of log2 relative expression in bulk vs single-cell reference,
    with a true y = x identity line and Pearson/Spearman/MSE summary statistics.
    """
    import matplotlib.pyplot as plt

    if hasattr(bp, "ref_ct_raw") and bp.ref_ct_raw is not None:
        sc_total = bp.ref_ct_raw.sum(axis=0)
    else:
        sc_total = bp.phi_cellType.sum(axis=0)
    sc_norm = sc_total / np.maximum(sc_total.sum(), 1e-15)

    bulk_total = bp.mixture.sum(axis=0)
    bulk_norm = bulk_total / np.maximum(bulk_total.sum(), 1e-15)

    log_sc = np.log2(np.maximum(sc_norm, 1e-15))
    log_bulk = np.log2(np.maximum(bulk_norm, 1e-15))

    cor_ps, _ = pearsonr(log_sc, log_bulk)
    cor_sp, _ = spearmanr(log_sc, log_bulk)
    mse = float(np.mean((log_sc - log_bulk) ** 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(log_sc, log_bulk, s=5, alpha=0.5, c="gray")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    span = hi - lo
    ax.text(lo + 0.65 * span, lo + 0.30 * span, f"R={cor_ps:.3f}")
    ax.text(lo + 0.65 * span, lo + 0.20 * span, f"rho={cor_sp:.3f}")
    ax.text(lo + 0.65 * span, lo + 0.10 * span, f"MSE={mse:.3g}")

    ax.set_xlabel("log2 total expression in scRNA-seq")
    ax.set_ylabel("log2 total expression in bulk")
    ax.set_title("Bulk vs scRNA-seq")
    if show:
        plt.show()
    return fig, ax
