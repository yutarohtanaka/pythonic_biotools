"""Single-cell tumor phylogeny reconstruction (NJ + O(CM) ML scoring + NNI) and clone EM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class CloneAssignment:
    """Per-cell clonal posterior probabilities, clone labels, and tumor/normal calls."""

    cell_ids: tuple[str, ...]
    clone_opt: np.ndarray
    compartment_opt: tuple[str, ...]
    p_cnv: np.ndarray
    aneuploidy_score: np.ndarray
    clone_probs: np.ndarray
    clone_ids: tuple[int, ...]
    clone_genotypes: np.ndarray
    event_names: tuple[str, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-cell clone assignments and posterior probabilities as a DataFrame."""
        data: dict[str, object] = {
            "cell": list(self.cell_ids),
            "clone_opt": self.clone_opt.astype(np.int64),
            "compartment_opt": list(self.compartment_opt),
            "p_cnv": self.p_cnv.astype(np.float64),
            "aneuploidy_score": self.aneuploidy_score.astype(np.float64),
        }
        for idx, cid in enumerate(self.clone_ids):
            data[f"p_clone_{cid}"] = self.clone_probs[:, idx]
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return a concise text summary of clonal composition."""
        n_cells = len(self.cell_ids)
        n_tumor = sum(1 for c in self.compartment_opt if c == "tumor")
        n_normal = n_cells - n_tumor
        return (
            f"CloneAssignment(n_cells={n_cells}, n_clones={len(self.clone_ids)}, "
            f"tumor={n_tumor}, normal={n_normal})"
        )


@dataclass(frozen=True)
class PhylogenyTree:
    """Maximum-likelihood single-cell tumor phylogeny reconstructed via NJ + NNI."""

    newick: str
    n_cells: int
    n_events: int
    n_clones: int
    log_likelihood: float
    initial_log_likelihood: float
    edges: tuple[tuple[int, int], ...]
    node_labels: dict[int, str]
    branch_lengths: np.ndarray
    event_branches: dict[str, int]
    clone_genotypes: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return tree edges, branch lengths, and mapped evolutionary events as a DataFrame."""
        cols = ["parent", "child", "parent_label", "child_label", "branch_length", "events"]
        if not self.edges:
            return pd.DataFrame(columns=cols)
        node_events: dict[int, list[str]] = {}
        for ev, node_id in self.event_branches.items():
            node_events.setdefault(int(node_id), []).append(ev)

        rows = []
        for idx, (u, v) in enumerate(self.edges):
            ev_str = ",".join(sorted(node_events.get(int(v), [])))
            rows.append(
                {
                    "parent": int(u),
                    "child": int(v),
                    "parent_label": self.node_labels.get(int(u), f"node_{u}"),
                    "child_label": self.node_labels.get(int(v), f"node_{v}"),
                    "branch_length": float(self.branch_lengths[idx]),
                    "events": ev_str,
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def to_newick(self) -> str:
        """Return the Newick representation of the rooted phylogeny."""
        return self.newick

    def summary(self) -> str:
        """Return a concise text summary of the reconstructed tumor phylogeny."""
        return (
            f"PhylogenyTree(n_cells={self.n_cells}, n_events={self.n_events}, "
            f"n_clones={self.n_clones}, ll_init={self.initial_log_likelihood:.2f}, "
            f"ll_opt={self.log_likelihood:.2f})"
        )


def _pairwise_event_distance(prob_mat: np.ndarray) -> np.ndarray:
    """Compute expected pairwise disagreement distance including diploid outgroup at row 0."""
    n_leaves, n_events = prob_mat.shape
    p_with_outgroup = np.vstack([np.full((1, n_events), 1e-4, dtype=np.float64), prob_mat])
    # D_ab = mean_m [P_a (1 - P_b) + (1 - P_a) P_b] = mean_m(P_a + P_b - 2 P_a P_b)
    q = 1.0 - p_with_outgroup
    dist = (p_with_outgroup @ q.T + q @ p_with_outgroup.T) / float(max(1, n_events))
    np.fill_diagonal(dist, 0.0)
    return np.maximum(dist, 0.0)


def _neighbor_joining_unrooted(dist_mat: np.ndarray) -> tuple[dict[int, list[tuple[int, float]]], int]:
    """Deterministic Saitou-Nei Neighbor-Joining on an (N x N) distance matrix."""
    n_taxa = dist_mat.shape[0]
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_taxa)}
    if n_taxa <= 1:
        return adj, n_taxa
    if n_taxa == 2:
        d01 = max(0.0, float(dist_mat[0, 1]))
        adj[0].append((1, d01))
        adj[1].append((0, d01))
        return adj, n_taxa

    active = list(range(n_taxa))
    d = np.array(dist_mat, dtype=np.float64, copy=True)
    next_id = n_taxa

    while len(active) > 2:
        r = len(active)
        row_sums = d.sum(axis=1)
        q_mat = (r - 2) * d - row_sums[:, None] - row_sums[None, :]
        # Mask lower triangle and diagonal
        iu = np.triu_indices(r, k=1)
        q_upper = q_mat[iu]
        min_q = float(np.min(q_upper))
        # Deterministic lexicographical tie-breaking within 1e-12
        tie_hits = np.flatnonzero(q_upper <= min_q + 1e-12)
        best_flat = int(tie_hits[0])
        a = int(iu[0][best_flat])
        b = int(iu[1][best_flat])

        u = active[a]
        v = active[b]
        w = next_id
        next_id += 1
        adj[w] = []

        dab = float(d[a, b])
        len_u = max(0.0, 0.5 * dab + (float(row_sums[a]) - float(row_sums[b])) / (2.0 * (r - 2)))
        len_v = max(0.0, dab - len_u)

        adj[w].append((u, len_u))
        adj[u].append((w, len_u))
        adj[w].append((v, len_v))
        adj[v].append((w, len_v))

        keep_idx = [k for k in range(r) if k != a and k != b]
        new_d_row = np.array(
            [max(0.0, 0.5 * (float(d[a, k]) + float(d[b, k]) - dab)) for k in keep_idx],
            dtype=np.float64,
        )

        m_rem = len(keep_idx)
        new_d = np.zeros((m_rem + 1, m_rem + 1), dtype=np.float64)
        if m_rem > 0:
            sub = d[np.ix_(keep_idx, keep_idx)]
            new_d[:m_rem, :m_rem] = sub
            new_d[:m_rem, m_rem] = new_d_row
            new_d[m_rem, :m_rem] = new_d_row

        active = [active[k] for k in keep_idx] + [w]
        d = new_d

    u, v = active[0], active[1]
    len_uv = max(0.0, float(d[0, 1]))
    adj[u].append((v, len_uv))
    adj[v].append((u, len_uv))
    return adj, next_id


def _root_at_outgroup(
    adj: dict[int, list[tuple[int, float]]],
    n_leaves: int,
) -> tuple[int, dict[int, tuple[int, ...]], dict[int, int], dict[int, float]]:
    """Root an unrooted NJ tree at diploid outgroup taxon 0 (leaves are 1..n_leaves)."""
    if n_leaves == 1:
        return 1, {1: ()}, {1: -1}, {1: 0.0}

    # Taxon 0 has exactly one neighbor in the NJ tree
    root_node, out_len = adj[0][0]
    children: dict[int, tuple[int, ...]] = {}
    parent: dict[int, int] = {root_node: -1}
    branch_len: dict[int, float] = {root_node: max(0.0, out_len)}

    stack = [root_node]
    visited = {0, root_node}
    while stack:
        u = stack.pop()
        ch = []
        for v, w in sorted(adj[u], key=lambda item: item[0]):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                branch_len[v] = max(0.0, float(w))
                ch.append(v)
                stack.append(v)
        children[u] = tuple(ch)

    return root_node, children, parent, branch_len


def _compute_post_order(root: int, children: dict[int, tuple[int, ...]]) -> list[int]:
    """Compute iterative post-order traversal of a rooted tree."""
    order: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        else:
            stack.append((node, True))
            for ch in reversed(children.get(node, ())):
                stack.append((ch, False))
    return order


def _score_tree_exact(
    root: int,
    children: dict[int, tuple[int, ...]],
    leaf_delta: np.ndarray,
    base_ll: np.ndarray,
) -> tuple[float, np.ndarray, dict[int, np.ndarray], list[int]]:
    """Score every event across all branches in one O(C * M) post-order traversal."""
    post_order = _compute_post_order(root, children)
    n_events = leaf_delta.shape[1]
    subtree_delta: dict[int, np.ndarray] = {}

    for u in post_order:
        ch = children[u]
        if not ch:
            subtree_delta[u] = leaf_delta[u - 1]
        else:
            acc = np.zeros(n_events, dtype=np.float64)
            for c in ch:
                acc += subtree_delta[c]
            subtree_delta[u] = acc

    n_nodes = len(post_order)
    # Candidate attachment points: every node u in post_order plus empty (-1)
    scores = np.empty((n_nodes + 1, n_events), dtype=np.float64)
    for idx, u in enumerate(post_order):
        scores[idx] = base_ll + subtree_delta[u]
    scores[n_nodes] = base_ll

    best_row = np.argmax(scores, axis=0)
    best_scores = scores[best_row, np.arange(n_events)]
    best_nodes = np.array(
        [-1 if r == n_nodes else post_order[int(r)] for r in best_row],
        dtype=np.int64,
    )
    return float(np.sum(best_scores)), best_nodes, subtree_delta, post_order


def _nni_optimize(
    root: int,
    children: dict[int, tuple[int, ...]],
    parent: dict[int, int],
    leaf_delta: np.ndarray,
    base_ll: np.ndarray,
    max_iter: int = 50,
) -> tuple[float, np.ndarray]:
    """Greedy steepest-ascent NNI topology refinement maximizing infinite-sites ML."""
    best_ll, best_nodes, _, post_order = _score_tree_exact(root, children, leaf_delta, base_ll)
    if max_iter <= 0 or len(post_order) <= 3:
        return best_ll, best_nodes

    for _ in range(max_iter):
        improved = False
        best_move: tuple[int, int, int, int] | None = None
        best_cand_ll = best_ll + 1e-9

        for v in post_order:
            u = parent[v]
            ch_v = children[v]
            if u == -1 or len(ch_v) != 2:
                continue
            ch_u = children[u]
            if len(ch_u) != 2:
                continue
            s = ch_u[1] if ch_u[0] == v else ch_u[0]
            l_node, r_node = ch_v

            for swap_node in (l_node, r_node):
                other_v = r_node if swap_node == l_node else l_node
                # Apply tentative NNI swap: s <-> swap_node
                children[u] = (v, swap_node) if ch_u[0] == v else (swap_node, v)
                children[v] = (s, other_v) if ch_v[0] == swap_node else (other_v, s)
                parent[swap_node] = u
                parent[s] = v

                cand_ll, _, _, _ = _score_tree_exact(root, children, leaf_delta, base_ll)
                if cand_ll > best_cand_ll:
                    best_cand_ll = cand_ll
                    best_move = (u, v, s, swap_node)
                    improved = True

                # Revert swap
                children[u] = ch_u
                children[v] = ch_v
                parent[swap_node] = v
                parent[s] = u

        if not improved or best_move is None:
            break

        u, v, s, swap_node = best_move
        ch_u = children[u]
        ch_v = children[v]
        l_v, r_v = ch_v
        other_v = r_v if swap_node == l_v else l_v
        children[u] = (v, swap_node) if ch_u[0] == v else (swap_node, v)
        children[v] = (s, other_v) if ch_v[0] == swap_node else (other_v, s)
        parent[swap_node] = u
        parent[s] = v
        best_ll, best_nodes, _, post_order = _score_tree_exact(root, children, leaf_delta, base_ll)

    return best_ll, best_nodes


def _collect_leaves(
    node: int,
    children: dict[int, tuple[int, ...]],
    n_leaves: int,
) -> list[int]:
    """Collect 1-based leaf indices beneath `node`."""
    if node == -1:
        return []
    leaves: list[int] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        ch = children.get(cur, ())
        if not ch:
            if 1 <= cur <= n_leaves:
                leaves.append(cur)
        else:
            for c in ch:
                stack.append(c)
    return leaves


def _build_newick(
    node: int,
    children: dict[int, tuple[int, ...]],
    branch_len: dict[int, float],
    node_labels: dict[int, str],
    is_root: bool = True,
) -> str:
    """Serialize rooted tree into Newick format."""
    ch = children.get(node, ())
    if not ch:
        label = node_labels.get(node, f"leaf_{node}")
        return label if is_root else f"{label}:{branch_len.get(node, 0.0):.6f}"
    sub = ",".join(
        _build_newick(c, children, branch_len, node_labels, is_root=False) for c in ch
    )
    if is_root:
        return f"({sub});"
    return f"({sub}):{branch_len.get(node, 0.0):.6f}"


def compute_clone_posteriors(
    llr_mat: np.ndarray,
    clone_genotypes: np.ndarray,
    tumor_prob_threshold: float = 0.5,
    n_em_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Compute per-cell clone posterior probabilities via EM over clone priors."""
    n_cells, n_events = llr_mat.shape
    n_clones = clone_genotypes.shape[0]
    if n_clones == 1 or n_events == 0:
        probs = np.ones((n_cells, 1), dtype=np.float64)
        clone_opt = np.ones(n_cells, dtype=np.int64)
        p_cnv = np.zeros(n_cells, dtype=np.float64)
        comp = tuple("normal" for _ in range(n_cells))
        return probs, clone_opt, comp, p_cnv

    # log P(D_c | G_k) = sum_m G_{k,m} * LLR_{c,m}
    log_lik = llr_mat @ clone_genotypes.T  # (C, K)
    pi = np.full(n_clones, 1.0 / n_clones, dtype=np.float64)

    probs = np.empty_like(log_lik)
    for _ in range(max(1, n_em_iter)):
        log_joint = log_lik + np.log(np.clip(pi, 1e-6, 1.0))[None, :]
        m = np.max(log_joint, axis=1, keepdims=True)
        exp_j = np.exp(log_joint - m)
        probs = exp_j / np.sum(exp_j, axis=1, keepdims=True)
        pi = np.maximum(np.mean(probs, axis=0), 1e-4)
        pi = pi / np.sum(pi)

    clone_opt = 1 + np.argmax(probs, axis=1).astype(np.int64)
    p_cnv = np.clip(1.0 - probs[:, 0], 0.0, 1.0)
    compartment = tuple(
        "tumor" if float(p) >= tumor_prob_threshold else "normal" for p in p_cnv
    )
    return probs, clone_opt, compartment, p_cnv


def _extract_surviving_clones(
    g_hat: np.ndarray,
    prob_mat: np.ndarray,
    llr_mat: np.ndarray,
    min_cells: int,
    tau: float,
) -> np.ndarray:
    """Prune small/weak clades from tree-imputed genotypes, returning unique clone genotypes."""
    n_cells, n_events = g_hat.shape
    if n_events == 0:
        return np.zeros((1, 0), dtype=np.int64)

    g_work = g_hat.copy()
    # Prune individual events covering fewer than min_cells cells
    for m in range(n_events):
        if int(np.sum(g_work[:, m])) < min_cells:
            g_work[:, m] = 0

    normal_geno = tuple([0] * n_events)
    counts: dict[tuple[int, ...], int] = {}
    cell_genos = [tuple(int(x) for x in g_work[c]) for c in range(n_cells)]
    for g in cell_genos:
        counts[g] = counts.get(g, 0) + 1

    surviving: set[tuple[int, ...]] = {normal_geno}
    small_genos: list[tuple[int, ...]] = []
    for g, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if g == normal_geno:
            continue
        if cnt >= min_cells:
            surviving.add(g)
        else:
            small_genos.append(g)

    # Merge any subclone with < min_cells into closest surviving genotype (by LLR)
    if small_genos:
        surv_list = sorted(surviving)
        surv_arr = np.asarray(surv_list, dtype=np.float64)
        for g in small_genos:
            idx = np.array([i for i, cg in enumerate(cell_genos) if cg == g], dtype=np.int64)
            ll_scores = np.sum(llr_mat[idx] @ surv_arr.T, axis=0)
            best_target = surv_list[int(np.argmax(ll_scores))]
            for i in idx:
                cell_genos[i] = best_target

    # Iteratively merge pairs of surviving tumor clones whose probability separation < tau
    while len(surviving) > 2:
        tumor_clones = sorted(g for g in surviving if g != normal_geno)
        merged_any = False
        for i in range(len(tumor_clones)):
            if merged_any:
                break
            for j in range(i + 1, len(tumor_clones)):
                ga = tumor_clones[i]
                gb = tumor_clones[j]
                diff_cols = [m for m in range(n_events) if ga[m] != gb[m]]
                if not diff_cols:
                    continue
                idx_a = [c for c, cg in enumerate(cell_genos) if cg == ga]
                idx_b = [c for c, cg in enumerate(cell_genos) if cg == gb]
                if not idx_a or not idx_b:
                    continue
                # Mean posterior probability separation on differing events
                p_a = prob_mat[idx_a][:, diff_cols]
                p_b = prob_mat[idx_b][:, diff_cols]
                sep = float(np.mean(np.abs(np.mean(p_a, axis=0) - np.mean(p_b, axis=0))))
                if sep < tau:
                    # Merge smaller into larger
                    keep, drop = (ga, gb) if len(idx_a) >= len(idx_b) else (gb, ga)
                    surviving.discard(drop)
                    for c in range(n_cells):
                        if cell_genos[c] == drop:
                            cell_genos[c] = keep
                    merged_any = True
                    break
        if not merged_any:
            break

    # Deterministic ordering: normal first, then tumor clones by descending assigned count
    final_counts = {g: 0 for g in surviving}
    for cg in cell_genos:
        if cg in final_counts:
            final_counts[cg] += 1
    tumor_sorted = sorted(
        (g for g in surviving if g != normal_geno),
        key=lambda g: (-final_counts[g], g),
    )
    ordered = [normal_geno] + tumor_sorted
    return np.asarray(ordered, dtype=np.int64)


def reconstruct_phylogeny(
    prob_mat: np.ndarray,
    llr_mat: np.ndarray | None = None,
    cell_ids: Sequence[str] | None = None,
    event_names: Sequence[str] | None = None,
    aneuploidy_score: np.ndarray | None = None,
    tau: float = 0.3,
    min_cells: int | None = None,
    max_tree_cells: int = 400,
    max_nni_iter: int = 50,
    tumor_prob_threshold: float = 0.5,
) -> tuple[PhylogenyTree, CloneAssignment]:
    """Reconstruct ML single-cell tumor phylogeny (NJ + NNI) and clonal posteriors."""
    p_arr = np.clip(np.asarray(prob_mat, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    if p_arr.ndim != 2 or p_arr.shape[0] == 0:
        raise ValueError("prob_mat must be a 2D array with at least 1 cell.")
    n_cells, n_events = p_arr.shape

    if llr_mat is None:
        llr_arr = np.log(p_arr) - np.log(1.0 - p_arr)
    else:
        llr_arr = np.asarray(llr_mat, dtype=np.float64)
        if llr_arr.shape != (n_cells, n_events):
            raise ValueError(
                f"llr_mat shape {llr_arr.shape} != prob_mat shape {(n_cells, n_events)}."
            )

    c_ids = (
        tuple(str(c) for c in cell_ids)
        if cell_ids is not None
        else tuple(f"cell_{i}" for i in range(n_cells))
    )
    if len(c_ids) != n_cells:
        raise ValueError("cell_ids length does not match prob_mat rows.")

    e_names = (
        tuple(str(e) for e in event_names)
        if event_names is not None
        else tuple(f"seg_{m + 1}" for m in range(n_events))
    )
    if len(e_names) != n_events:
        raise ValueError("event_names length does not match prob_mat cols.")

    aneup = (
        np.asarray(aneuploidy_score, dtype=np.float64)
        if aneuploidy_score is not None
        else (np.mean(p_arr, axis=1) if n_events > 0 else np.zeros(n_cells, dtype=np.float64))
    )

    eff_min_cells = (
        int(min_cells)
        if min_cells is not None
        else max(3, min(50, n_cells // 10))
    )
    if n_cells < 6:
        eff_min_cells = min(eff_min_cells, 1)

    if n_events == 0:
        clone_genotypes = np.zeros((1, 0), dtype=np.int64)
        probs, clone_opt, comp, p_cnv = compute_clone_posteriors(
            llr_arr, clone_genotypes, tumor_prob_threshold=tumor_prob_threshold
        )
        labels = {i + 1: c_ids[i] for i in range(n_cells)}
        if n_cells == 1:
            newick = f"{c_ids[0]};"
            edges: tuple[tuple[int, int], ...] = ()
            bl = np.zeros(0, dtype=np.float64)
        else:
            root_id = n_cells + 1
            labels[root_id] = "root"
            edges = tuple((root_id, i + 1) for i in range(n_cells))
            bl = np.zeros(n_cells, dtype=np.float64)
            newick = "(" + ",".join(f"{c_ids[i]}:0.000000" for i in range(n_cells)) + ");"

        geno_df = pd.DataFrame(index=["clone_1"], columns=[])
        tree = PhylogenyTree(
            newick=newick,
            n_cells=n_cells,
            n_events=0,
            n_clones=1,
            log_likelihood=0.0,
            initial_log_likelihood=0.0,
            edges=edges,
            node_labels=labels,
            branch_lengths=bl,
            event_branches={},
            clone_genotypes=geno_df,
        )
        clones = CloneAssignment(
            cell_ids=c_ids,
            clone_opt=clone_opt,
            compartment_opt=comp,
            p_cnv=p_cnv,
            aneuploidy_score=aneup,
            clone_probs=probs,
            clone_ids=(1,),
            clone_genotypes=clone_genotypes,
            event_names=(),
        )
        return tree, clones

    # Optional micro-cluster compression for very large cohorts
    if n_cells > max_tree_cells:
        km = KMeans(n_clusters=max_tree_cells, random_state=0, n_init=3)
        cluster_assign = km.fit_predict(p_arr)
        unique_cl = np.unique(cluster_assign)
        n_leaves = len(unique_cl)
        leaf_prob = np.zeros((n_leaves, n_events), dtype=np.float64)
        leaf_delta = np.zeros((n_leaves, n_events), dtype=np.float64)
        leaf_names = []
        cell_to_leaf = np.empty(n_cells, dtype=np.int64)
        raw_delta = np.log(p_arr) - np.log(1.0 - p_arr)
        for idx, cid in enumerate(unique_cl):
            mask = cluster_assign == cid
            cell_to_leaf[mask] = idx + 1
            leaf_prob[idx] = np.mean(p_arr[mask], axis=0)
            leaf_delta[idx] = np.sum(raw_delta[mask], axis=0)
            leaf_names.append(f"cluster_{idx + 1}")
    else:
        n_leaves = n_cells
        leaf_prob = p_arr
        leaf_delta = np.log(p_arr) - np.log(1.0 - p_arr)
        leaf_names = list(c_ids)
        cell_to_leaf = np.arange(1, n_cells + 1, dtype=np.int64)

    base_ll = np.sum(np.log(1.0 - p_arr), axis=0)
    dist_mat = _pairwise_event_distance(leaf_prob)
    adj, _ = _neighbor_joining_unrooted(dist_mat)
    root, children, parent, branch_len = _root_at_outgroup(adj, n_leaves)

    init_ll, _init_nodes, _, _ = _score_tree_exact(root, children, leaf_delta, base_ll)
    opt_ll, best_nodes = _nni_optimize(
        root, children, parent, leaf_delta, base_ll, max_iter=max_nni_iter
    )

    # Map best branches back to cell genotypes
    g_hat = np.zeros((n_cells, n_events), dtype=np.int64)
    event_branches: dict[str, int] = {}
    for m in range(n_events):
        node_m = int(best_nodes[m])
        event_branches[e_names[m]] = node_m
        if node_m != -1:
            leaves_m = set(_collect_leaves(node_m, children, n_leaves))
            for c in range(n_cells):
                if int(cell_to_leaf[c]) in leaves_m:
                    g_hat[c, m] = 1

    clone_geno_arr = _extract_surviving_clones(
        g_hat=g_hat,
        prob_mat=p_arr,
        llr_mat=llr_arr,
        min_cells=eff_min_cells,
        tau=tau,
    )
    n_clones = clone_geno_arr.shape[0]
    probs, clone_opt, comp, p_cnv = compute_clone_posteriors(
        llr_arr, clone_geno_arr, tumor_prob_threshold=tumor_prob_threshold
    )

    # Build PhylogenyTree metadata
    post_order = _compute_post_order(root, children)
    node_labels: dict[int, str] = {}
    for u in post_order:
        if 1 <= u <= n_leaves:
            node_labels[u] = leaf_names[u - 1]
        elif u == root:
            node_labels[u] = "root"
        else:
            node_labels[u] = f"internal_{u}"

    edges_list: list[tuple[int, int]] = []
    bl_list: list[float] = []
    for u in reversed(post_order):
        for v in children.get(u, ()):
            edges_list.append((int(u), int(v)))
            bl_list.append(float(branch_len.get(v, 0.0)))

    newick = _build_newick(root, children, branch_len, node_labels, is_root=True)
    clone_ids = tuple(range(1, n_clones + 1))
    geno_df = pd.DataFrame(
        clone_geno_arr,
        index=[f"clone_{cid}" for cid in clone_ids],
        columns=list(e_names),
    )

    tree = PhylogenyTree(
        newick=newick,
        n_cells=n_cells,
        n_events=n_events,
        n_clones=n_clones,
        log_likelihood=float(opt_ll),
        initial_log_likelihood=float(init_ll),
        edges=tuple(edges_list),
        node_labels=node_labels,
        branch_lengths=np.asarray(bl_list, dtype=np.float64),
        event_branches=event_branches,
        clone_genotypes=geno_df,
    )
    clones = CloneAssignment(
        cell_ids=c_ids,
        clone_opt=clone_opt,
        compartment_opt=comp,
        p_cnv=p_cnv,
        aneuploidy_score=aneup,
        clone_probs=probs,
        clone_ids=clone_ids,
        clone_genotypes=clone_geno_arr,
        event_names=e_names,
    )
    return tree, clones
