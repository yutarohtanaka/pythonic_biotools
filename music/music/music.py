from typing import Literal
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.sparse import issparse

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def _get_counts(adata, layer=None):
    """Extract dense float64 count matrix (observations x genes) from AnnData."""
    X = adata.layers[layer] if layer is not None else adata.X
    if issparse(X):
        return X.toarray().astype(float, copy=False)
    return np.asarray(X, dtype=float)


def music_basis(
    adata,
    cluster_col,
    sample_col,
    layer=None,
    markers=None,
    select_ct=None,
    cell_sizes=None,
    ct_cov=False,
    non_zero=True,
):
    """
    Compute the MuSiC basis: Design matrix, library sizes, and cross-subject variances.

    Parameters
    ----------
    adata : AnnData
        Single-cell reference (cells x genes).
    cluster_col : str
        Column name in adata.obs containing cell type annotations.
    sample_col : str
        Column name in adata.obs containing donor/sample annotations.
    layer : str, optional
        Layer in adata to use for raw counts. If None, uses adata.X.
    markers : sequence of str, optional
        Subset of genes to retain in the returned basis matrices.
    select_ct : sequence of str, optional
        Cell types to include (preserves specified order).
    cell_sizes : dict, optional
        Mapping from cell type to pre-specified cell size S_k.
    ct_cov : bool, default=False
        If True, computes gene-wise cross-cell-type covariance matrices.
    non_zero : bool, default=True
        If True, eliminates genes with zero total counts across all cells.

    Returns
    -------
    dict
        Keys: 'Design_mtx', 'M_theta', 'Sigma', 'S', 'M_S', and optionally 'Sigma_ct'.
    """
    X_counts = _get_counts(adata, layer)
    genes = pd.Index(adata.var_names)

    if non_zero:
        keep_nz = np.sum(X_counts, axis=0) > 0
        X_counts = X_counts[:, keep_nz]
        genes = genes[keep_nz]

    obs = adata.obs
    clusters = obs[cluster_col].astype(str).values
    samples = obs[sample_col].astype(str).values

    if select_ct is None:
        unique_clusters = np.unique(clusters)
    else:
        unique_clusters = pd.Index(select_ct).unique().astype(str).values
    unique_samples = np.unique(samples)

    n_samples = len(unique_samples)
    n_clusters = len(unique_clusters)
    n_genes = len(genes)

    theta_all = np.full((n_samples, n_clusters, n_genes), np.nan, dtype=float)
    S_mtx = np.full((n_samples, n_clusters), np.nan, dtype=float)

    for j, ct in enumerate(unique_clusters):
        ct_mask = clusters == ct
        for i, sid in enumerate(unique_samples):
            mask = ct_mask & (samples == sid)
            n_cells = int(np.sum(mask))
            if n_cells > 0:
                y = X_counts[mask, :]
                y_sum = float(np.sum(y))
                if y_sum > 0:
                    theta_all[i, j, :] = np.sum(y, axis=0) / y_sum
                    S_mtx[i, j] = y_sum / n_cells
                # Donor-cluster groups with 0 total counts remain NaN (not 0.0)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M_theta = np.nanmean(theta_all, axis=0).T  # (n_genes, n_clusters)
        Sigma = np.nanvar(theta_all, axis=0, ddof=1).T  # (n_genes, n_clusters)
        M_S = np.nanmean(S_mtx, axis=0)  # (n_clusters,)

    Sigma_ct_res = None
    if ct_cov:
        Sigma_ct_res = np.full((n_genes, n_clusters, n_clusters), np.nan, dtype=float)
        valid_ij = ~np.isnan(S_mtx)
        for j1 in range(n_clusters):
            for j2 in range(j1, n_clusters):
                both = valid_ij[:, j1] & valid_ij[:, j2]
                n_both = int(np.sum(both))
                if n_both > 1:
                    a = theta_all[both, j1, :] - np.mean(
                        theta_all[both, j1, :], axis=0, keepdims=True
                    )
                    b = theta_all[both, j2, :] - np.mean(
                        theta_all[both, j2, :], axis=0, keepdims=True
                    )
                    cov_g = np.sum(a * b, axis=0) / (n_both - 1)
                    Sigma_ct_res[:, j1, j2] = cov_g
                    Sigma_ct_res[:, j2, j1] = cov_g

    if cell_sizes is not None:
        for i, ct in enumerate(unique_clusters):
            if ct in cell_sizes:
                M_S[i] = float(cell_sizes[ct])

    D = M_theta * M_S

    D_df = pd.DataFrame(D, index=genes, columns=unique_clusters)
    M_theta_df = pd.DataFrame(M_theta, index=genes, columns=unique_clusters)
    Sigma_df = pd.DataFrame(Sigma, index=genes, columns=unique_clusters)
    S_df = pd.DataFrame(S_mtx, index=unique_samples, columns=unique_clusters)
    M_S_series = pd.Series(M_S, index=unique_clusters)

    if markers is not None:
        marker_genes = genes.intersection(pd.Index(markers), sort=False)
        marker_idx = genes.get_indexer(marker_genes)
        D_df = D_df.iloc[marker_idx]
        M_theta_df = M_theta_df.iloc[marker_idx]
        Sigma_df = Sigma_df.iloc[marker_idx]
        if ct_cov:
            Sigma_ct_res = Sigma_ct_res[marker_idx, :, :]

    res = {
        "Design_mtx": D_df,
        "M_theta": M_theta_df,
        "Sigma": Sigma_df,
        "S": S_df,
        "M_S": M_S_series,
    }
    if ct_cov:
        res["Sigma_ct"] = Sigma_ct_res

    return res


def music_iter(
    Y,
    D,
    S,
    Sigma,
    Sigma_ct=None,
    iter_max=1000,
    nu=0.0001,
    eps=0.01,
    centered=False,
    normalize=False,
    return_var=False,
):
    """
    Iteratively reweighted NNLS solver for MuSiC (Wang et al., 2019).

    Maintains raw NNLS coefficients q_raw on the scale Y ~ D @ q_raw so that
    residual variance r^2 and cross-subject variance ((q_raw * S)^2 * Sigma)
    share identical units, while normalizing simplex proportions
    p = q_raw / sum(q_raw) inside every iteration for convergence testing.
    """
    D = np.array(D, dtype=float, copy=True)
    Y = np.array(Y, dtype=float, copy=True)
    S = np.array(S, dtype=float, copy=True)
    Sigma_clean = np.nan_to_num(Sigma, nan=0.0)
    Sigma_ct_clean = np.nan_to_num(Sigma_ct, nan=0.0) if Sigma_ct is not None else None

    if centered:
        D = D - np.mean(D)
        Y = Y - np.mean(Y)

    if normalize:
        D_std = np.std(D, ddof=1)
        if D_std > 0:
            D = D / D_std
            S = S * D_std
        Y_std = np.std(Y, ddof=1)
        if Y_std > 0:
            Y = Y / Y_std
    else:
        Y = Y * 100.0

    q_nnls, _ = nnls(D, Y)
    q_nnls_sum = np.sum(q_nnls)
    p_nnls = q_nnls / q_nnls_sum if q_nnls_sum > 0 else q_nnls.copy()
    r = Y - D @ q_nnls

    if Sigma_ct_clean is not None:
        Sp = q_nnls * S
        cov_term = np.einsum("i,gij,j->g", Sp, Sigma_ct_clean, Sp)
    else:
        cov_term = np.sum(((q_nnls * S) ** 2) * Sigma_clean, axis=1)

    weight_gene = 1.0 / np.maximum(nu + r**2 + cov_term, 1e-15)
    sqrt_w = np.sqrt(weight_gene)
    Y_weight = Y * sqrt_w
    D_weight = D * sqrt_w[:, None]

    q_weight, _ = nnls(D_weight, Y_weight)
    q_weight_sum = np.sum(q_weight)
    p_weight = q_weight / q_weight_sum if q_weight_sum > 0 else q_weight.copy()
    r = Y_weight - D_weight @ q_weight

    for _ in range(iter_max):
        if Sigma_ct_clean is not None:
            Sp = q_weight * S
            cov_term = np.einsum("i,gij,j->g", Sp, Sigma_ct_clean, Sp)
        else:
            cov_term = np.sum(((q_weight * S) ** 2) * Sigma_clean, axis=1)

        weight_gene = 1.0 / np.maximum(nu + r**2 + cov_term, 1e-15)
        sqrt_w = np.sqrt(weight_gene)
        Y_weight = Y * sqrt_w
        D_weight = D * sqrt_w[:, None]

        q_weight_new, _ = nnls(D_weight, Y_weight)
        q_new_sum = np.sum(q_weight_new)
        p_weight_new = (
            q_weight_new / q_new_sum if q_new_sum > 0 else q_weight_new.copy()
        )
        r_new = Y_weight - D_weight @ q_weight_new

        if np.sum(np.abs(p_weight_new - p_weight)) < eps:
            q_weight = q_weight_new
            p_weight = p_weight_new
            r = r_new
            break

        q_weight = q_weight_new
        p_weight = p_weight_new
        r = r_new

    fitted = D @ q_weight
    with np.errstate(invalid="ignore", divide="ignore"):
        var_Y = np.var(Y, ddof=1)
        if var_Y > 0 and len(Y) > 1:
            R_squared = 1.0 - np.var(Y - fitted, ddof=1) / var_Y
        else:
            R_squared = np.nan

    if not return_var:
        return p_weight, p_nnls, R_squared, weight_gene

    # NNLS sandwich variance Var(p_weight) via delta method on p = q / sum(q)
    q_sum = float(np.sum(q_weight))
    if q_sum > 0 and len(Y) > 0:
        xpx = D_weight.T @ D_weight
        inv_xpx = np.linalg.pinv(xpx)
        meat = (D_weight * (r ** 2)[:, None]).T @ D_weight
        cov_q = inv_xpx @ meat @ inv_xpx
        jac = (np.eye(len(p_weight), dtype=float) - np.outer(p_weight, np.ones_like(p_weight))) / q_sum
        cov_p = jac @ cov_q @ jac.T
        var_p = np.maximum(np.diag(cov_p), 0.0)
    else:
        var_p = np.full(len(p_weight), np.nan, dtype=float)

    return p_weight, p_nnls, R_squared, weight_gene, var_p


def _music_prop_from_basis(
    bulk_X,
    bulk_obs_names,
    bulk_genes,
    basis,
    markers=None,
    non_zero=True,
    check_overlap=True,
    iter_max=1000,
    nu=0.0001,
    eps=0.01,
    centered=False,
    normalize=False,
    settings: SettingsMode = "default",
):
    """Run MuSiC deconvolution given a precomputed scRNA-seq basis dictionary."""
    validate_settings(settings)
    D_df = basis["Design_mtx"]
    Sigma_df = basis["Sigma"]
    M_S_series = basis["M_S"]
    Sigma_ct = basis.get("Sigma_ct", None)

    valid_ct_mask = (~D_df.isna().any(axis=0)) & (~M_S_series.isna())
    if settings == "R_default":
        # analysis.R:122 drops cell types with Sigma = NaN (single-donor cell types)
        valid_ct_mask = valid_ct_mask & (~Sigma_df.isna().any(axis=0))
    valid_ct = valid_ct_mask[valid_ct_mask].index

    if len(valid_ct) == 0:
        raise ValueError(
            "No valid cell types with non-zero counts found in single-cell reference!"
        )

    valid_idx = D_df.columns.get_indexer(valid_ct)
    D_df = D_df.iloc[:, valid_idx]
    Sigma_df = Sigma_df.iloc[:, valid_idx]
    M_S_series = M_S_series.iloc[valid_idx]
    if Sigma_ct is not None:
        Sigma_ct = Sigma_ct[:, valid_idx, :][:, :, valid_idx]

    bulk_genes_idx = pd.Index(bulk_genes)
    sc_genes_idx = D_df.index

    common_genes = bulk_genes_idx.intersection(sc_genes_idx)
    if check_overlap and len(common_genes) < 0.2 * min(
        len(bulk_genes_idx), len(sc_genes_idx)
    ):
        raise ValueError("Too few common genes between bulk and single-cell reference!")

    if markers is not None:
        common_genes = common_genes.intersection(pd.Index(markers))

    if len(common_genes) == 0:
        raise ValueError("No common genes remain after marker subsetting!")

    sc_loc = D_df.index.get_indexer(common_genes)
    bulk_loc = bulk_genes_idx.get_indexer(common_genes)

    D_mat = D_df.values[sc_loc, :]
    Sigma_mat = np.nan_to_num(Sigma_df.values[sc_loc, :], nan=0.0)
    Sigma_ct_mat = (
        np.nan_to_num(Sigma_ct[sc_loc, :, :], nan=0.0)
        if Sigma_ct is not None
        else None
    )
    bulk_X_common = bulk_X[:, bulk_loc]
    used_genes = common_genes

    if non_zero:
        keep_nz = (np.sum(bulk_X_common, axis=0) > 0) & (np.sum(D_mat, axis=1) > 0)
        bulk_X_common = bulk_X_common[:, keep_nz]
        D_mat = D_mat[keep_nz, :]
        Sigma_mat = Sigma_mat[keep_nz, :]
        used_genes = used_genes[keep_nz]
        if Sigma_ct_mat is not None:
            Sigma_ct_mat = Sigma_ct_mat[keep_nz, :, :]

    bulk_sums = np.sum(bulk_X_common, axis=1, keepdims=True)
    bulk_sums = np.where(bulk_sums == 0, 1.0, bulk_sums)
    Y_ra = bulk_X_common / bulk_sums

    S_vec = M_S_series.values
    res_p_weight = []
    res_p_nnls = []
    res_r2 = []
    res_var_p = []
    weight_gene_mat = (
        np.full((len(used_genes), Y_ra.shape[0]), np.nan, dtype=float)
        if settings == "R_default"
        else None
    )

    for i in range(Y_ra.shape[0]):
        Y_i = Y_ra[i, :]
        mask = Y_i > 0
        if np.sum(mask) == 0:
            res_p_weight.append(np.full(len(valid_ct), np.nan))
            res_p_nnls.append(np.full(len(valid_ct), np.nan))
            res_r2.append(np.nan)
            if settings == "R_default":
                res_var_p.append(np.full(len(valid_ct), np.nan))
            continue

        Y_i_sub = Y_i[mask]
        D_mat_sub = D_mat[mask, :]
        Sigma_mat_sub = Sigma_mat[mask, :]
        Sigma_ct_sub = Sigma_ct_mat[mask, :, :] if Sigma_ct_mat is not None else None

        if settings == "R_default":
            pw, pnnls, r2, w_sub, vp = music_iter(
                Y_i_sub,
                D_mat_sub,
                S_vec,
                Sigma_mat_sub,
                Sigma_ct=Sigma_ct_sub,
                iter_max=iter_max,
                nu=nu,
                eps=eps,
                centered=centered,
                normalize=normalize,
                return_var=True,
            )
            res_var_p.append(vp)
            assert weight_gene_mat is not None
            weight_gene_mat[mask, i] = w_sub
        else:
            pw, pnnls, r2, _ = music_iter(
                Y_i_sub,
                D_mat_sub,
                S_vec,
                Sigma_mat_sub,
                Sigma_ct=Sigma_ct_sub,
                iter_max=iter_max,
                nu=nu,
                eps=eps,
                centered=centered,
                normalize=normalize,
            )

        res_p_weight.append(pw)
        res_p_nnls.append(pnnls)
        res_r2.append(r2)

    df_weight = pd.DataFrame(res_p_weight, index=bulk_obs_names, columns=valid_ct)
    df_nnls = pd.DataFrame(res_p_nnls, index=bulk_obs_names, columns=valid_ct)

    out = {
        "Est_prop_weighted": df_weight,
        "Est_prop_allgene": df_nnls,
        "R_squared": pd.Series(res_r2, index=bulk_obs_names),
    }
    if settings == "R_default":
        out["Var_prop"] = pd.DataFrame(res_var_p, index=bulk_obs_names, columns=valid_ct)
        out["Weight_gene"] = pd.DataFrame(
            weight_gene_mat, index=used_genes, columns=bulk_obs_names
        )
    return out


def music_prop(
    bulk_adata,
    sc_adata,
    cluster_col,
    sample_col,
    sc_layer=None,
    bulk_layer=None,
    markers=None,
    select_ct=None,
    cell_sizes=None,
    ct_cov=False,
    non_zero=True,
    iter_max=1000,
    nu=0.0001,
    eps=0.01,
    centered=False,
    normalize=False,
    settings: SettingsMode = "default",
):
    """
    Estimate cell-type proportions in bulk RNA-seq data using MuSiC.
    """
    validate_settings(settings)
    basis = music_basis(
        sc_adata,
        cluster_col,
        sample_col,
        layer=sc_layer,
        markers=None,
        select_ct=select_ct,
        cell_sizes=cell_sizes,
        ct_cov=ct_cov,
        non_zero=non_zero,
    )
    bulk_X = _get_counts(bulk_adata, bulk_layer)
    return _music_prop_from_basis(
        bulk_X=bulk_X,
        bulk_obs_names=bulk_adata.obs_names,
        bulk_genes=bulk_adata.var_names,
        basis=basis,
        markers=markers,
        non_zero=non_zero,
        check_overlap=(markers is None),
        iter_max=iter_max,
        nu=nu,
        eps=eps,
        centered=centered,
        normalize=normalize,
        settings=settings,
    )
