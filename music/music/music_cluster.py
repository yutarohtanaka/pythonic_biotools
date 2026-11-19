import numpy as np
import pandas as pd
from .music import SettingsMode, _get_counts, music_basis, music_iter, validate_settings


def music_prop_cluster(
    bulk_adata,
    sc_adata,
    group_markers,
    groups_col,
    clusters_col,
    samples_col,
    clusters_type,
    sc_layer=None,
    bulk_layer=None,
    non_zero=True,
    iter_max=1000,
    nu=0.0001,
    eps=0.01,
    centered=False,
    normalize=False,
    settings: SettingsMode = "default",
):
    """
    Hierarchical / tree-guided MuSiC deconvolution across major cell groups and sub-clusters.
    """
    validate_settings(settings)
    if set(group_markers.keys()) != set(clusters_type.keys()):
        raise ValueError("Keys for group_markers and clusters_type must match!")

    group_names = list(clusters_type.keys())
    sub_cluster_names = []
    for g in group_names:
        sub_cluster_names.extend(clusters_type[g])

    cluster_sc_basis = music_basis(
        sc_adata,
        groups_col,
        samples_col,
        layer=sc_layer,
        select_ct=group_names,
        non_zero=non_zero,
    )
    sc_basis = music_basis(
        sc_adata,
        clusters_col,
        samples_col,
        layer=sc_layer,
        select_ct=sub_cluster_names,
        non_zero=non_zero,
    )

    eff_clusters_type = {g: list(clusters_type[g]) for g in group_names}
    if settings == "R_default":
        valid_grp_mask = (
            (~cluster_sc_basis["Design_mtx"].isna().any(axis=0))
            & (~cluster_sc_basis["M_S"].isna())
            & (~cluster_sc_basis["Sigma"].isna().any(axis=0))
        )
        valid_sub_mask = (
            (~sc_basis["Design_mtx"].isna().any(axis=0))
            & (~sc_basis["M_S"].isna())
            & (~sc_basis["Sigma"].isna().any(axis=0))
        )
        valid_sub_set = set(valid_sub_mask[valid_sub_mask].index)
        new_group_names = []
        new_sub_names = []
        new_clusters_type = {}
        for g in group_names:
            if not bool(valid_grp_mask.get(g, False)):
                continue
            kept_subs = [s for s in eff_clusters_type[g] if s in valid_sub_set]
            if kept_subs:
                new_group_names.append(g)
                new_clusters_type[g] = kept_subs
                new_sub_names.extend(kept_subs)
        if not new_sub_names:
            raise ValueError(
                "No valid cell types with finite cross-subject Sigma remain in single-cell reference!"
            )
        group_names = new_group_names
        sub_cluster_names = new_sub_names
        eff_clusters_type = new_clusters_type

    bulk_genes = pd.Index(bulk_adata.var_names)
    sc_genes = sc_basis["Design_mtx"].index.intersection(
        cluster_sc_basis["Design_mtx"].index
    )
    cm_gene = bulk_genes.intersection(sc_genes)

    if len(cm_gene) < 0.2 * min(len(bulk_genes), len(sc_genes)):
        raise ValueError("Too few common genes!")

    bulk_X = _get_counts(bulk_adata, bulk_layer)
    bulk_gene_idx = bulk_genes.get_indexer(cm_gene)
    bulk_X_common = bulk_X[:, bulk_gene_idx]

    if non_zero:
        keep_nz = np.sum(bulk_X_common, axis=0) > 0
        bulk_X_common = bulk_X_common[:, keep_nz]
        cm_gene = cm_gene[keep_nz]

    # Deterministic intersections for group markers and non-marker genes
    group_markers_clean = {
        k: cm_gene.intersection(pd.Index(v)).tolist() for k, v in group_markers.items()
    }
    all_markers = []
    for m in group_markers_clean.values():
        all_markers.extend(m)
    cluster_diff = cm_gene.intersection(pd.Index(all_markers))
    cluster_select = cm_gene.difference(cluster_diff, sort=False)

    if len(cluster_select) == 0:
        raise ValueError(
            "No non-marker common genes remain to estimate top-level cluster proportions!"
        )

    D1 = sc_basis["Design_mtx"].loc[cm_gene]
    M_S = sc_basis["M_S"]
    Sigma = sc_basis["Sigma"].loc[cm_gene].fillna(0.0)

    D1_cluster = cluster_sc_basis["Design_mtx"].loc[cluster_select, group_names]
    M_S_cluster = cluster_sc_basis["M_S"][group_names]
    Sigma_cluster = (
        cluster_sc_basis["Sigma"].loc[cluster_select, group_names].fillna(0.0)
    )

    D1_sub = cluster_sc_basis["Design_mtx"].loc[cluster_diff, group_names]

    bulk_sums = np.sum(bulk_X_common, axis=1, keepdims=True)
    bulk_sums = np.where(bulk_sums == 0, 1.0, bulk_sums)
    Y_ra = bulk_X_common / bulk_sums

    gene_to_idx = {g: i for i, g in enumerate(cm_gene)}
    res_p_weight_cluster = []

    for i in range(Y_ra.shape[0]):
        Y_i = Y_ra[i, :]
        mask = Y_i > 0
        if np.sum(mask) == 0:
            res_p_weight_cluster.append(np.full(len(sub_cluster_names), np.nan))
            continue

        valid_gene_set = {cm_gene[idx] for idx in np.flatnonzero(mask)}
        valid_cluster_select = [g for g in cluster_select if g in valid_gene_set]

        if len(valid_cluster_select) == 0:
            res_p_weight_cluster.append(np.full(len(sub_cluster_names), np.nan))
            continue

        D1_cluster_temp = np.nan_to_num(
            D1_cluster.loc[valid_cluster_select].values, nan=0.0
        )
        Y_cluster_temp = Y_i[[gene_to_idx[g] for g in valid_cluster_select]]
        Sigma_cluster_temp = np.nan_to_num(
            Sigma_cluster.loc[valid_cluster_select].values, nan=0.0
        )
        M_S_cluster_vals = np.nan_to_num(M_S_cluster.values, nan=0.0)

        pw_cluster, _, _, _ = music_iter(
            Y_cluster_temp,
            D1_cluster_temp,
            M_S_cluster_vals,
            Sigma_cluster_temp,
            iter_max=iter_max,
            nu=nu,
            eps=eps,
            centered=centered,
            normalize=normalize,
        )

        p_weight = []
        for j, g_name in enumerate(group_names):
            sub_ct = eff_clusters_type[g_name]
            if len(sub_ct) == 1:
                p_weight.append(float(pw_cluster[j]))
            elif pw_cluster[j] == 0:
                p_weight.extend([0.0] * len(sub_ct))
            else:
                c_marker = [
                    g for g in group_markers_clean[g_name] if g in valid_gene_set
                ]
                if len(c_marker) == 0:
                    p_weight.extend([float(pw_cluster[j]) / len(sub_ct)] * len(sub_ct))
                    continue

                Y_cm = Y_i[[gene_to_idx[g] for g in c_marker]]
                D1_sub_temp = np.nan_to_num(D1_sub.loc[c_marker].values, nan=0.0)

                # Scale top-level proportions to bulk relative-abundance units:
                # Y_g ~ sum_k (pw_k * D_{g,k}) / sum_k (pw_k * S_k)
                denom_scale = float(np.sum(pw_cluster * M_S_cluster_vals))
                if denom_scale > 0:
                    q_cluster = pw_cluster / denom_scale
                else:
                    q_cluster = np.zeros_like(pw_cluster)

                other_contrib = D1_sub_temp @ q_cluster - D1_sub_temp[:, j] * q_cluster[j]
                Y_sub = Y_cm - other_contrib

                mask_sub = Y_sub > 0
                if np.sum(mask_sub) == 0:
                    p_weight.extend([float(pw_cluster[j]) / len(sub_ct)] * len(sub_ct))
                    continue

                Y_sub_filt = Y_sub[mask_sub]
                c_marker_filt = [
                    c_marker[idx] for idx in np.flatnonzero(mask_sub)
                ]

                D1_temp = np.nan_to_num(
                    D1.loc[c_marker_filt, sub_ct].values, nan=0.0
                )
                M_S_temp = np.nan_to_num(M_S[sub_ct].values, nan=0.0)
                Sigma_temp = np.nan_to_num(
                    Sigma.loc[c_marker_filt, sub_ct].values, nan=0.0
                )

                pw_sub, _, _, _ = music_iter(
                    Y_sub_filt,
                    D1_temp,
                    M_S_temp,
                    Sigma_temp,
                    iter_max=iter_max,
                    nu=nu,
                    eps=eps,
                    centered=centered,
                    normalize=normalize,
                )

                sub_weights = pw_cluster[j] * pw_sub
                p_weight.extend(sub_weights.tolist())

        res_p_weight_cluster.append(p_weight)

    df_weight_cluster = pd.DataFrame(
        res_p_weight_cluster, index=bulk_adata.obs_names, columns=sub_cluster_names
    )

    return {"Est_prop_weighted_cluster": df_weight_cluster}
