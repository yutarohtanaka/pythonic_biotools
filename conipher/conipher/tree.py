"""
CONIPHER phylogenetic tree reconstruction (`reconstruct_trees`) enforcing the
Sum Condition (Pigeonhole Principle), Crossing Rule, iterative CN-error cluster
pruning, spanning arborescence enumeration, and regional subclone composition
(Grigoriadis et al., Nature Protocols 2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .clustering import ClusterResult
from .io import SettingsMode, validate_settings


def _parse_cluster_ccf_input(
    cluster_ccf_df: pd.DataFrame | ClusterResult,
    mutation_assignments: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract `(cluster_ids, regions, phi_mat, se_mat, n_muts)` from `ClusterResult`,
    wide `cluster_ccf_df`, or long `cluster_ccf_df`.
    """
    if isinstance(cluster_ccf_df, ClusterResult):
        res = cluster_ccf_df
        c_ids = res.cluster_ccf_df["cluster_id"].astype(np.int64).to_numpy()
        n_muts = res.cluster_ccf_df["n_mutations"].astype(np.int64).to_numpy()
        regions = list(res.regions)
        phi_mat = np.clip(
            res.cluster_ccf_df[regions].to_numpy(dtype=np.float64), 0.0, 1.0
        )
        se_mat = res.cluster_se_df[regions].to_numpy(dtype=np.float64)
        return c_ids, regions, phi_mat, se_mat, n_muts

    if not isinstance(cluster_ccf_df, pd.DataFrame) or cluster_ccf_df.empty:
        raise ValueError("cluster_ccf_df must be a non-empty DataFrame or ClusterResult.")

    df = cluster_ccf_df.copy()
    if "region" in df.columns and ("ccf" in df.columns or "phylo_ccf" in df.columns):
        val_col = "ccf" if "ccf" in df.columns else "phylo_ccf"
        if "cluster_id" not in df.columns:
            raise ValueError("Long-format cluster_ccf_df must contain 'cluster_id'.")
        regions = list(dict.fromkeys(df["region"].astype(str)))
        c_ids_list = sorted(int(c) for c in df["cluster_id"].unique() if int(c) > 0)
        piv = (
            df.pivot_table(index="cluster_id", columns="region", values=val_col, aggfunc="mean")
            .reindex(index=c_ids_list, columns=regions)
            .fillna(0.0)
        )
        phi_mat = np.clip(piv.to_numpy(dtype=np.float64), 0.0, 1.0)
        if "ccf_se" in df.columns:
            piv_se = (
                df.pivot_table(
                    index="cluster_id", columns="region", values="ccf_se", aggfunc="mean"
                )
                .reindex(index=c_ids_list, columns=regions)
                .fillna(0.03)
            )
            se_mat = piv_se.to_numpy(dtype=np.float64)
        else:
            se_mat = np.full_like(phi_mat, 0.03)

        if "n_mutations" in df.columns:
            nm_map = df.groupby("cluster_id")["n_mutations"].first().to_dict()
            n_muts = np.array([int(nm_map.get(cid, 20)) for cid in c_ids_list], dtype=np.int64)
        else:
            n_muts = np.full(len(c_ids_list), 20, dtype=np.int64)
        return np.array(c_ids_list, dtype=np.int64), regions, phi_mat, se_mat, n_muts

    # Wide format
    if "cluster_id" in df.columns:
        c_ids = df["cluster_id"].astype(np.int64).to_numpy()
    else:
        c_ids = np.arange(1, len(df) + 1, dtype=np.int64)

    if "n_mutations" in df.columns:
        n_muts = df["n_mutations"].astype(np.int64).to_numpy()
    elif mutation_assignments is not None and "cluster_id" in mutation_assignments.columns:
        vc = mutation_assignments["cluster_id"].value_counts().to_dict()
        n_muts = np.array([int(vc.get(int(cid), 10)) for cid in c_ids], dtype=np.int64)
    else:
        n_muts = np.full(len(df), 20, dtype=np.int64)

    meta = {"cluster_id", "n_mutations", "is_truncal"}
    reg_cols = [
        c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(reg_cols) < 2:
        raise ValueError("cluster_ccf_df must contain at least 2 region CCF columns.")

    regions = [str(c) for c in reg_cols]
    phi_mat = np.clip(df[reg_cols].to_numpy(dtype=np.float64), 0.0, 1.0)
    se_mat = np.full_like(phi_mat, 0.03)
    return c_ids, regions, phi_mat, se_mat, n_muts


def _select_truncal_index(phi_mat: np.ndarray, n_muts: np.ndarray, settings: SettingsMode) -> int:
    """Identify index of the ubiquitous truncal root cluster."""
    # Prefer clusters with >= 0.85 across all regions (CONIPHER truncal definition)
    ubiq_85 = np.flatnonzero(np.all(phi_mat >= 0.85, axis=1))
    if ubiq_85.size > 0:
        scores = np.mean(phi_mat[ubiq_85], axis=1) + 1e-6 * n_muts[ubiq_85]
        return int(ubiq_85[int(np.argmax(scores))])

    # Fallback: cluster present in maximum number of regions, breaking ties by min_r then mean_r
    n_pres = np.sum(phi_mat >= 0.05, axis=1)
    min_ccf = np.min(phi_mat, axis=1)
    mean_ccf = np.mean(phi_mat, axis=1)
    best = max(
        range(phi_mat.shape[0]),
        key=lambda i: (int(n_pres[i]), float(min_ccf[i]), float(mean_ccf[i]), int(n_muts[i])),
    )
    return int(best)


def _build_directed_ancestry_adj(
    phi: np.ndarray,
    se: np.ndarray,
    n_muts: np.ndarray,
    root_idx: int,
    ccf_buffer: float,
    use_se_tolerance: bool,
) -> list[list[int]]:
    """
    Construct candidate parent lists `cand_parents[v]` for each node `v` enforcing:
    1. Ancestry Dominance: `phi[u, r] + tol >= phi[v, r]` for all regions `r`, and
       an absent clone (`phi[u, r] <= 0.015`) cannot parent a present clone (`phi[v, r] >= 0.05`).
    2. Crossing Rule: if `phi[u, r1] > phi[v, r1] + tol` and `phi[v, r2] > phi[u, r2] + tol`,
       neither `u -> v` nor `v -> u` is permitted.
    3. Deterministic tie-breaking when two clones have nearly identical CCF across all regions.
    """
    k = phi.shape[0]
    mean_ccf = np.mean(phi, axis=1)

    # Strict rank order with root_idx guaranteed first
    rank_key = [
        (
            1 if i == root_idx else 0,
            round(float(mean_ccf[i]), 6),
            int(n_muts[i]),
            -i,
        )
        for i in range(k)
    ]

    cand_parents: list[list[int]] = [[] for _ in range(k)]
    for v in range(k):
        if v == root_idx:
            continue
        parents_v: list[int] = []
        for u in range(k):
            if u == v:
                continue

            if use_se_tolerance:
                tol = np.maximum(0.05, 1.96 * np.sqrt(se[u] ** 2 + se[v] ** 2))
            else:
                tol = np.full(phi.shape[1], ccf_buffer, dtype=np.float64)

            # Crossing Rule check
            u_exceeds_v = bool(np.any(phi[u] > phi[v] + tol))
            v_exceeds_u = bool(np.any(phi[v] > phi[u] + tol))
            if u_exceeds_v and v_exceeds_u:
                continue

            # Dominance check: u must >= v - tol in EVERY region
            if bool(np.any(phi[u] + tol < phi[v])):
                continue

            # Presence inheritance: if v is clearly present in region r, u cannot be absent in r
            present_v_absent_u = (phi[v] >= 0.05) & (phi[u] <= 0.015)
            if bool(np.any(present_v_absent_u)):
                continue

            # Break exact mutual symmetry so identical clones cannot form 2-cycles
            u_can_parent_v = True
            if not u_exceeds_v and not v_exceeds_u:
                # Mutual tolerance overlap across all regions: enforce topological rank
                present_u_absent_v = bool(np.any((phi[u] >= 0.05) & (phi[v] <= 0.015)))
                if not present_u_absent_v and rank_key[u] < rank_key[v]:
                    u_can_parent_v = False

            if u_can_parent_v:
                parents_v.append(u)

        # Prefer parents with closest higher CCF (most direct ancestor) first in search
        parents_v.sort(key=lambda u: float(np.sum(np.maximum(0.0, phi[u] - phi[v]))))
        cand_parents[v] = parents_v

    return cand_parents


def _enumerate_valid_trees(
    phi: np.ndarray,
    se: np.ndarray,
    n_muts: np.ndarray,
    root_idx: int,
    ccf_buffer: float,
    use_se_tolerance: bool,
    max_trees: int = 500,
) -> list[dict[int, int]]:
    """
    Enumerate all valid rooted spanning arborescences `{child_idx: parent_idx}`
    satisfying the Sum Condition (Pigeonhole Principle) at every internal node:

        phi[u, r] + tau[u, r] >= sum_{v in children(u)} phi[v, r]   for all r.
    """
    k, n_reg = phi.shape
    if k == 1:
        return [{}]

    cand_parents = _build_directed_ancestry_adj(
        phi=phi,
        se=se,
        n_muts=n_muts,
        root_idx=root_idx,
        ccf_buffer=ccf_buffer,
        use_se_tolerance=use_se_tolerance,
    )

    non_root = [v for v in range(k) if v != root_idx]
    if any(len(cand_parents[v]) == 0 for v in non_root):
        return []

    # Assign most constrained nodes first (MRV heuristic) then higher mean CCF
    non_root.sort(key=lambda v: (len(cand_parents[v]), -float(np.mean(phi[v]))))

    child_sum = np.zeros((k, n_reg), dtype=np.float64)
    child_se2_sum = np.zeros((k, n_reg), dtype=np.float64)
    parent_of: dict[int, int] = {}
    valid_trees: list[dict[int, int]] = []

    def _creates_cycle(child: int, cand_parent: int) -> bool:
        cur = cand_parent
        while cur in parent_of:
            cur = parent_of[cur]
            if cur == child:
                return True
        return False

    def _backtrack(pos: int) -> None:
        if len(valid_trees) >= max_trees:
            return
        if pos == len(non_root):
            # Verify full tree reachability from root_idx (guaranteed if acyclic with k-1 edges)
            valid_trees.append(dict(parent_of))
            return

        v = non_root[pos]
        for u in cand_parents[v]:
            if _creates_cycle(v, u):
                continue

            new_sum = child_sum[u] + phi[v]
            new_se2 = child_se2_sum[u] + (se[v] ** 2)
            if use_se_tolerance:
                tau = np.maximum(0.05, 1.96 * np.sqrt((se[u] ** 2) + new_se2))
            else:
                tau = ccf_buffer

            # Monotonic Pigeonhole pruning: since phi >= 0, exceeding bound now can never recover
            if bool(np.any(new_sum > phi[u] + tau + 1e-12)):
                continue

            parent_of[v] = u
            child_sum[u] = new_sum
            child_se2_sum[u] = new_se2

            _backtrack(pos + 1)

            del parent_of[v]
            child_sum[u] -= phi[v]
            child_se2_sum[u] -= se[v] ** 2
            if len(valid_trees) >= max_trees:
                return

    _backtrack(0)
    return valid_trees


def _score_tree(parent_map: dict[int, int], phi: np.ndarray) -> tuple[float, float]:
    """
    Score a valid tree for ranking (lower excess violation first, higher CONIPHER
    edge-sharing weight second). Returns `(pigeonhole_excess, -edge_weight)`.
    """
    k, n_reg = phi.shape
    ch_sum = np.zeros((k, n_reg), dtype=np.float64)
    edge_weight = 0.0
    for v, u in parent_map.items():
        ch_sum[u] += phi[v]
        edge_weight += float(np.sum(np.minimum(phi[u], phi[v])))

    used_parents = list(set(parent_map.values()))
    if used_parents:
        excess = float(
            np.sum(np.maximum(0.0, ch_sum[used_parents] - phi[used_parents]))
        )
    else:
        excess = 0.0
    return (round(excess, 6), -round(edge_weight, 6))


@dataclass
class TreeResult:
    """Phylogenetic tree reconstruction output from `reconstruct_trees`."""

    parent_child_df: pd.DataFrame
    trees: list[pd.DataFrame]
    n_alternative_trees: int
    truncal_cluster: int
    retained_clusters: list[int]
    dropped_clusters: list[int]
    clone_proportions_df: pd.DataFrame
    cluster_ccf_df: pd.DataFrame
    settings: SettingsMode

    def summary(self) -> dict[str, Any]:
        """Return dictionary summary of reconstructed phylogenetic tree(s)."""
        return {
            "truncal_cluster": self.truncal_cluster,
            "n_retained_clusters": len(self.retained_clusters),
            "dropped_clusters": self.dropped_clusters,
            "n_alternative_trees": self.n_alternative_trees,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return primary phylogenetic tree edge list (`parent_child_df`)."""
        return self.parent_child_df.copy()


def reconstruct_trees(
    cluster_ccf_df: pd.DataFrame | ClusterResult,
    mutation_assignments: pd.DataFrame | None = None,
    ccf_buffer: float = 0.10,
    settings: SettingsMode = "default",
) -> TreeResult:
    """
    Reconstruct multi-region tumor phylogenetic trees enforcing CONIPHER's
    Sum Condition (Pigeonhole Principle) and Crossing Rule, iteratively pruning
    copy-number-error/noise clusters when conflicts prevent tree feasibility,
    and computing regional subclone proportions.

    Parameters
    ----------
    cluster_ccf_df : pd.DataFrame | ClusterResult
        Cluster CCF profiles across `R >= 2` regions (from `cluster_ccf_multiregion`,
        wide table, or long table).
    mutation_assignments : pd.DataFrame | None, default=None
        Optional per-mutation cluster assignments used to infer cluster sizes if
        `n_mutations` is absent from `cluster_ccf_df`.
    ccf_buffer : float, default=0.10
        CCF tolerance (`[0, 1]` scale) for Sum Condition and Crossing Rule in
        `"R_default"` mode (or when custom non-default `ccf_buffer` is passed).
    settings : {"default", "R_default"}, default="default"
        `"default"` uses SE-calibrated tolerances (`max(0.05, 1.96 * se_pair)`)
        when standard errors are available; `"R_default"` uses fixed
        `ccf_buffer = 0.10` and greedy smallest-cluster conflict pruning.

    Returns
    -------
    TreeResult
        Primary tree (`parent_child_df`), all valid spanning arborescences (`trees`),
        `n_alternative_trees`, pruned clusters (`dropped_clusters`), and regional
        subclone composition (`clone_proportions_df`).
    """
    validate_settings(settings)
    if ccf_buffer < 0.0 or not np.isfinite(ccf_buffer):
        raise ValueError(f"ccf_buffer must be >= 0, got {ccf_buffer!r}.")

    c_ids, regions, phi_all, se_all, n_muts_all = _parse_cluster_ccf_input(
        cluster_ccf_df, mutation_assignments=mutation_assignments
    )
    has_explicit_se = isinstance(cluster_ccf_df, ClusterResult) or (
        isinstance(cluster_ccf_df, pd.DataFrame) and "ccf_se" in cluster_ccf_df.columns
    )
    use_se = (settings == "default") and has_explicit_se and np.isclose(ccf_buffer, 0.10)

    root_global = _select_truncal_index(phi_all, n_muts_all, settings=settings)
    active = list(range(len(c_ids)))
    dropped_global: list[int] = []

    valid_maps: list[dict[int, int]] = []
    while True:
        sub_phi = phi_all[active]
        sub_se = se_all[active]
        sub_nm = n_muts_all[active]
        root_local = active.index(root_global)

        valid_maps = _enumerate_valid_trees(
            phi=sub_phi,
            se=sub_se,
            n_muts=sub_nm,
            root_idx=root_local,
            ccf_buffer=ccf_buffer,
            use_se_tolerance=use_se,
        )
        if valid_maps or len(active) <= 1:
            break

        # Iteratively prune conflicting / smallest non-truncal cluster (CONIPHER error correction)
        non_root_locals = [idx_l for idx_l in range(len(active)) if idx_l != root_local]

        # First test whether dropping a SINGLE non-truncal cluster restores tree feasibility
        single_drop_Resolvable: list[int] = []
        for cand_l in non_root_locals:
            trial_active = [active[j] for j in range(len(active)) if j != cand_l]
            trial_root = trial_active.index(root_global)
            trial_trees = _enumerate_valid_trees(
                phi=phi_all[trial_active],
                se=se_all[trial_active],
                n_muts=n_muts_all[trial_active],
                root_idx=trial_root,
                ccf_buffer=ccf_buffer,
                use_se_tolerance=use_se,
                max_trees=1,
            )
            if trial_trees:
                single_drop_Resolvable.append(cand_l)

        cand_adj = _build_directed_ancestry_adj(
            phi=sub_phi,
            se=sub_se,
            n_muts=sub_nm,
            root_idx=root_local,
            ccf_buffer=ccf_buffer,
            use_se_tolerance=use_se,
        )

        if single_drop_Resolvable:
            # Drop smallest mutation-count cluster among single-drop conflict resolvers
            drop_local = min(
                single_drop_Resolvable,
                key=lambda l_idx: (int(sub_nm[l_idx]), len(cand_adj[l_idx])),
            )
        else:
            # Prioritize unreachable orphans (0 candidate parents), then smallest cluster size
            orphans = [l_idx for l_idx in non_root_locals if len(cand_adj[l_idx]) == 0]
            pool = orphans if orphans else non_root_locals
            drop_local = min(
                pool,
                key=lambda l_idx: (int(sub_nm[l_idx]), float(np.mean(sub_phi[l_idx]))),
            )

        dropped_g = active.pop(drop_local)
        dropped_global.append(int(c_ids[dropped_g]))

    sub_phi = phi_all[active]
    retained_ids = [int(c_ids[g]) for g in active]
    truncal_id = int(c_ids[root_global])

    # Rank valid trees so primary tree has minimal Pigeonhole excess & maximal shared CCF support
    valid_maps.sort(key=lambda pm: _score_tree(pm, sub_phi))

    tree_dfs: list[pd.DataFrame] = []
    for pm in valid_maps:
        edges = sorted(
            [(int(c_ids[active[u]]), int(c_ids[active[v]])) for v, u in pm.items()],
            key=lambda pair: (pair[0], pair[1]),
        )
        tree_dfs.append(pd.DataFrame(edges, columns=["parent", "child"]))

    if not tree_dfs:
        tree_dfs = [pd.DataFrame(columns=["parent", "child"])]

    primary_df = tree_dfs[0]
    best_pm = valid_maps[0] if valid_maps else {}

    # Compute regional subclone cellular composition q_{u, r} = max(0, phi_{u, r} - sum_{v in ch(u)} phi_{v, r})
    k_ret, n_reg = sub_phi.shape
    child_ccf_sum = np.zeros((k_ret, n_reg), dtype=np.float64)
    for v_l, u_l in best_pm.items():
        child_ccf_sum[u_l] += sub_phi[v_l]

    raw_subclone_ccf = np.maximum(0.0, sub_phi - child_ccf_sum)
    col_sums = np.sum(raw_subclone_ccf, axis=0, keepdims=True)
    clone_prop = np.where(
        col_sums > 1e-12,
        raw_subclone_ccf / np.maximum(col_sums, 1e-12),
        0.0,
    )

    prop_rows: list[dict[str, Any]] = []
    for l_i, cid in enumerate(retained_ids):
        for r_i, reg in enumerate(regions):
            prop_rows.append(
                {
                    "cluster_id": cid,
                    "region": reg,
                    "cluster_ccf": float(sub_phi[l_i, r_i]),
                    "subclone_ccf": float(raw_subclone_ccf[l_i, r_i]),
                    "clone_proportion": float(clone_prop[l_i, r_i]),
                }
            )
    clone_proportions_df = pd.DataFrame(prop_rows)

    ret_ccf_rows: list[dict[str, Any]] = []
    for l_i, g_i in enumerate(active):
        row_d: dict[str, Any] = {
            "cluster_id": int(c_ids[g_i]),
            "n_mutations": int(n_muts_all[g_i]),
            "is_truncal": int(c_ids[g_i]) == truncal_id,
        }
        for r_i, reg in enumerate(regions):
            row_d[reg] = float(sub_phi[l_i, r_i])
        ret_ccf_rows.append(row_d)
    retained_ccf_df = pd.DataFrame(ret_ccf_rows)

    return TreeResult(
        parent_child_df=primary_df,
        trees=tree_dfs,
        n_alternative_trees=len(valid_maps) if valid_maps else 1,
        truncal_cluster=truncal_id,
        retained_clusters=retained_ids,
        dropped_clusters=dropped_global,
        clone_proportions_df=clone_proportions_df,
        cluster_ccf_df=retained_ccf_df,
        settings=settings,
    )
