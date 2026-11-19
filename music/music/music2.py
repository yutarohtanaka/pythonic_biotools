import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .music import _get_counts, _music_prop_from_basis, music_basis


def music2_prop(
    bulk_control_adata,
    bulk_case_adata,
    sc_adata,
    cluster_col,
    sample_col,
    sc_layer=None,
    bulk_layer=None,
    select_ct=None,
    markers=None,
    cell_sizes=None,
    ct_cov=False,
    non_zero=True,
    expr_low=20,
    prop_r=0.1,
    eps_c=0.05,
    eps_r=0.01,
    n_resample=20,
    sample_prop=0.5,
    cutoff_expr=0.05,
    cutoff_c=0.05,
    cutoff_r=0.01,
    maxiter=200,
    iter_max_music=1000,
    nu=0.0001,
    eps_music=0.01,
    centered=False,
    normalize=False,
    seed=None,
):
    """
    MuSiC2 Deconvolution across clinical conditions (Fan et al., Briefings in Bioinformatics 2022).

    Iteratively identifies condition-associated cell-type-specific differentially
    expressed (DE) genes between bulk control and case cohorts and removes them from
    the precomputed scRNA-seq basis until case proportions converge.

    Returns
    -------
    dict
        Keys: 'Est_prop' (ordered as bulk_control_adata.obs_names + bulk_case_adata.obs_names),
        'convergence', 'n_iter' (when converged), 'DE_genes' (final iteration DE gene list),
        and 'id_not_converge' (when not converged).
    """
    rng = np.random.default_rng(seed)

    bulk_control_genes = pd.Index(bulk_control_adata.var_names)
    bulk_case_genes = pd.Index(bulk_case_adata.var_names)

    gene_bulk = bulk_control_genes.intersection(bulk_case_genes)
    if len(gene_bulk) < 0.1 * min(len(bulk_control_genes), len(bulk_case_genes)):
        raise ValueError("Not enough common genes between bulk control and case data!")

    # Compute scRNA-seq basis ONCE outside the outer loop
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
    sc_genes = basis["Design_mtx"].index

    gene_all = gene_bulk.intersection(sc_genes)
    if len(gene_all) < 0.2 * min(len(gene_bulk), len(sc_genes)):
        raise ValueError("Not enough genes between bulk and single-cell data!")

    if markers is not None:
        gene_all = gene_all.intersection(pd.Index(markers))
        if len(gene_all) == 0:
            raise ValueError("No common genes remain after marker filtering!")

    ctrl_full_X = _get_counts(bulk_control_adata, layer=bulk_layer)
    case_full_X = _get_counts(bulk_case_adata, layer=bulk_layer)

    ctrl_gene_loc = bulk_control_genes.get_indexer(gene_all)
    case_gene_loc = bulk_case_genes.get_indexer(gene_all)
    bulk_control_X = ctrl_full_X[:, ctrl_gene_loc]
    bulk_case_X = case_full_X[:, case_gene_loc]

    bulk_mtx_all = np.vstack([bulk_control_X, bulk_case_X])
    expr = np.mean(bulk_mtx_all, axis=0)
    exp_mask = expr >= expr_low
    exp_genel = gene_all[exp_mask].values
    exp_genel_idx = np.flatnonzero(exp_mask)

    # Initial deconvolution using precomputed basis
    res_control = _music_prop_from_basis(
        bulk_X=bulk_control_X,
        bulk_obs_names=bulk_control_adata.obs_names,
        bulk_genes=gene_all,
        basis=basis,
        markers=None,
        non_zero=non_zero,
        check_overlap=False,
        iter_max=iter_max_music,
        nu=nu,
        eps=eps_music,
        centered=centered,
        normalize=normalize,
    )
    prop_control = res_control["Est_prop_weighted"]

    res_case_ini = _music_prop_from_basis(
        bulk_X=bulk_case_X,
        bulk_obs_names=bulk_case_adata.obs_names,
        bulk_genes=gene_all,
        basis=basis,
        markers=None,
        non_zero=non_zero,
        check_overlap=False,
        iter_max=iter_max_music,
        nu=nu,
        eps=eps_music,
        centered=centered,
        normalize=normalize,
    )
    prop_case_ini = res_case_ini["Est_prop_weighted"]
    prop_case_fix = pd.DataFrame(columns=prop_control.columns, dtype=float)

    ncell = prop_control.shape[1]
    cell_types = prop_control.columns

    n_ctrl_samples = bulk_control_X.shape[0]
    n_case_samples = bulk_case_X.shape[0]
    n_ctrl_sub = min(n_ctrl_samples, max(1, int(np.round(n_ctrl_samples * sample_prop))))
    n_case_sub = min(n_case_samples, max(1, int(np.round(n_case_samples * sample_prop))))
    eff_resample = max(1, int(n_resample))

    case_name_to_row = {name: idx for idx, name in enumerate(bulk_case_adata.obs_names)}

    iter_num = 1
    id_conv = set()
    all_converge = False
    lr = []

    while iter_num <= maxiter:
        prop_case_combined = pd.concat([prop_case_ini, prop_case_fix]).loc[
            bulk_case_adata.obs_names
        ]
        prop_all = pd.concat([prop_control, prop_case_combined])

        lr = []
        if len(exp_genel) > 0:
            MOD0 = np.zeros((ncell, len(exp_genel)), dtype=float)
            MOD1 = np.zeros((ncell, len(exp_genel)), dtype=float)
            LOGFC = np.zeros((eff_resample, ncell, len(exp_genel)), dtype=float)

            ctrl_exp_X = bulk_control_X[:, exp_genel_idx]
            case_exp_X = bulk_case_X[:, exp_genel_idx]
            prop_ctrl_vals = prop_control.values
            prop_case_vals = prop_case_combined.values

            for r_idx in range(eff_resample):
                id_h_idx = rng.choice(n_ctrl_samples, size=n_ctrl_sub, replace=False)
                control_s = ctrl_exp_X[id_h_idx, :]
                prop_h = prop_ctrl_vals[id_h_idx, :]

                mod0 = np.zeros((ncell, len(exp_genel)), dtype=float)
                for g in range(len(exp_genel)):
                    p, _ = nnls(prop_h, control_s[:, g])
                    mod0[:, g] = p
                MOD0 += mod0

                id_d_idx = rng.choice(n_case_samples, size=n_case_sub, replace=False)
                case_s = case_exp_X[id_d_idx, :]
                prop_d = prop_case_vals[id_d_idx, :]

                mod1 = np.zeros((ncell, len(exp_genel)), dtype=float)
                for g in range(len(exp_genel)):
                    p, _ = nnls(prop_d, case_s[:, g])
                    mod1[:, g] = p
                MOD1 += mod1

                LOGFC[r_idx] = np.log1p(mod1) - np.log1p(mod0)

            MOD0 /= eff_resample
            MOD1 /= eff_resample

            mean_logfc = np.mean(LOGFC, axis=0)
            if eff_resample > 1:
                sd_logfc = np.std(LOGFC, axis=0, ddof=1)
            else:
                sd_logfc = np.zeros_like(mean_logfc)

            # Guard zero/NaN resampling SD (e.g. n_case == 1 or constant zero genes)
            denom = np.where((sd_logfc > 0) & np.isfinite(sd_logfc), sd_logfc, 1e-6)
            rcv = np.where(mean_logfc == 0, 0.0, mean_logfc / denom)
            abs_rcv_logfc = np.abs(rcv)

            mex = prop_all.mean(axis=0)

            for c_idx, celltype in enumerate(cell_types):
                m = mex[celltype]
                rh = MOD0[c_idx, :]
                rd = MOD1[c_idx, :]

                thresh_rh = np.percentile(rh, cutoff_expr * 100.0)
                thresh_rd = np.percentile(rd, cutoff_expr * 100.0)
                low_both = (rh <= thresh_rh) & (rd <= thresh_rd)
                valid_mask = ~low_both

                x_valid = abs_rcv_logfc[c_idx, valid_mask]
                genes_valid = exp_genel[valid_mask]
                finite_vals = x_valid[np.isfinite(x_valid)]

                if len(finite_vals) > 0:
                    q_pct = (1.0 - cutoff_c) * 100.0 if m >= prop_r else (1.0 - cutoff_r) * 100.0
                    thresh = np.percentile(finite_vals, q_pct)
                    lr.extend(genes_valid[x_valid >= thresh].tolist())

            lr = pd.Index(lr).unique().tolist()

        lr_set = set(lr)
        l_genes = [g for g in gene_all if g not in lr_set]
        if len(l_genes) == 0:
            l_genes = list(gene_all)
            lr = []

        unconverged_names = prop_case_ini.index
        unconverged_rows = [case_name_to_row[name] for name in unconverged_names]

        res_case_new = _music_prop_from_basis(
            bulk_X=bulk_case_X[unconverged_rows, :],
            bulk_obs_names=unconverged_names,
            bulk_genes=gene_all,
            basis=basis,
            markers=l_genes,
            non_zero=non_zero,
            check_overlap=False,
            iter_max=iter_max_music,
            nu=nu,
            eps=eps_music,
            centered=centered,
            normalize=normalize,
        )
        prop_case_new = res_case_new["Est_prop_weighted"]

        pc = np.abs(prop_case_new.values - prop_case_ini.values)
        conv = np.ones_like(pc, dtype=int)

        for c_idx, celltype in enumerate(cell_types):
            ini_vals = prop_case_ini[celltype].values
            rare_mask = ini_vals <= prop_r
            common_mask = ini_vals > prop_r

            conv[rare_mask, c_idx] = np.where(pc[rare_mask, c_idx] < eps_r, 0, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                pct_change = pc[common_mask, c_idx] / ini_vals[common_mask]
            conv[common_mask, c_idx] = np.where(pct_change < eps_c, 0, 1)

        convf = np.all(conv == 0, axis=1)
        new_conv_ids = prop_case_new.index[convf].tolist()
        id_conv.update(new_conv_ids)

        prop_CASE = pd.concat([prop_case_new, prop_case_fix])
        is_conv = prop_CASE.index.isin(id_conv)
        prop_case_ini = prop_CASE.loc[~is_conv]
        prop_case_fix = prop_CASE.loc[is_conv]

        if len(prop_case_ini) == 0:
            all_converge = True
            break

        iter_num += 1

    # Preserve exact original sample ordering: control obs_names followed by case obs_names
    prop_case_final = pd.concat([prop_case_ini, prop_case_fix]).loc[
        bulk_case_adata.obs_names
    ]
    prop_all_final = pd.concat(
        [prop_control.loc[bulk_control_adata.obs_names], prop_case_final]
    )

    if all_converge:
        return {
            "Est_prop": prop_all_final,
            "convergence": True,
            "n_iter": iter_num,
            "DE_genes": lr,
        }
    else:
        return {
            "Est_prop": prop_all_final,
            "convergence": False,
            "DE_genes": lr,
            "id_not_converge": [
                name
                for name in bulk_case_adata.obs_names
                if name in set(prop_case_ini.index)
            ],
        }
