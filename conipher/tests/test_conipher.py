"""
Deterministic verification suite for `conipher` covering multi-region PhyloCCF
and multiplicity recovery, Variational Dirichlet Process subclone clustering,
truncal collapse, small-artifact filtering, Sum Condition (Pigeonhole Principle)
and Crossing Rule phylogenetic constraints, alternative spanning arborescence
enumeration, copy-number-error cluster pruning, regional subclone composition,
and `settings="default"` vs `settings="R_default"` parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conipher import (
    CONIPHER,
    ClusterResult,
    ConipherConfig,
    ConipherResult,
    TreeResult,
    cluster_ccf_multiregion,
    compute_phylo_ccf,
    reconstruct_trees,
    run_conipher,
)


def _simulate_multiregion_mutations(
    clone_ccfs: list[tuple[int, dict[str, float]]],
    purity_by_region: dict[str, float],
    *,
    depth: int = 150,
    gain_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate multi-region somatic read counts from planted clone sizes & regional
    CCFs `[(n_mutations, {"R1": ccf1, "R2": ccf2, ...}), ...]`.
    """
    rng = np.random.default_rng(seed)
    regions = list(purity_by_region.keys())
    rows: list[dict[str, object]] = []
    mut_counter = 0

    for clone_idx, (n_muts, reg_ccf_map) in enumerate(clone_ccfs, start=1):
        for _ in range(n_muts):
            mut_counter += 1
            mid = f"mut_{mut_counter:04d}"
            is_gained_locus = bool(rng.random() < gain_fraction)
            for reg in regions:
                phi = float(reg_ccf_map[reg])
                rho = float(purity_by_region[reg])
                if is_gained_locus and phi >= 0.99:
                    maj_cn, min_cn, true_m = 2, 1, 2
                else:
                    maj_cn, min_cn, true_m = 1, 1, 1

                denom = rho * float(maj_cn + min_cn) + 2.0 * (1.0 - rho)
                exp_vaf = (rho * float(true_m) * phi) / denom
                dp = int(max(rng.poisson(depth), 50))
                alt = int(rng.binomial(dp, np.clip(exp_vaf, 0.0, 1.0)))
                rows.append(
                    {
                        "mutation_id": mid,
                        "region": reg,
                        "alt_count": alt,
                        "depth": dp,
                        "major_cn": maj_cn,
                        "minor_cn": min_cn,
                        "true_clone": clone_idx,
                        "true_ccf": phi,
                    }
                )
    return pd.DataFrame(rows)


def test_settings_validation_raises_value_error() -> None:
    purity = {"R1": 0.8, "R2": 0.7}
    muts = pd.DataFrame(
        {
            "mutation_id": ["m1", "m1"],
            "region": ["R1", "R2"],
            "alt_count": [30, 25],
            "depth": [80, 80],
            "major_cn": [1, 1],
            "minor_cn": [1, 1],
        }
    )
    ccf_wide = pd.DataFrame({"cluster_id": [1], "n_mutations": [20], "R1": [1.0], "R2": [1.0]})

    with pytest.raises(ValueError, match="Unknown settings"):
        compute_phylo_ccf(muts, purity, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        cluster_ccf_multiregion(ccf_wide, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        reconstruct_trees(ccf_wide, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        run_conipher(muts, purity, settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        CONIPHER(settings="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown settings"):
        ConipherConfig(settings="bad")  # type: ignore[arg-type]


def test_single_region_raises_value_error() -> None:
    muts = pd.DataFrame(
        {
            "mutation_id": ["m1"],
            "region": ["R1"],
            "alt_count": [30],
            "depth": [80],
            "major_cn": [1],
            "minor_cn": [1],
        }
    )
    with pytest.raises(ValueError, match="at least 2 regions"):
        compute_phylo_ccf(muts, {"R1": 0.8})


def test_compute_phylo_ccf_exact_formula_and_multiplicity_selection() -> None:
    # Region R1: purity = 0.8, 2:1 gain, m=2 clonal => denom = 0.8*3 + 0.2*2 = 2.8, f = 1.6/2.8
    # Region R2: purity = 0.6, 1:1 diploid, subclonal phi = 0.5 => denom = 2.0, f = 0.3/2.0 = 0.15
    muts = pd.DataFrame(
        {
            "mutation_id": ["m1", "m1"],
            "region": ["R1", "R2"],
            "alt_count": [160, 30],
            "depth": [280, 200],
            "major_cn": [2, 1],
            "minor_cn": [1, 1],
        }
    )
    res = compute_phylo_ccf(muts, {"R1": 0.8, "R2": 0.6}, settings="default")
    r1 = res.loc[res["region"] == "R1"].iloc[0]
    r2 = res.loc[res["region"] == "R2"].iloc[0]

    assert int(r1["multiplicity"]) == 2
    assert np.isclose(r1["phylo_ccf"], 1.0, atol=1e-12)
    assert int(r2["multiplicity"]) == 1
    assert np.isclose(r2["phylo_ccf"], 0.50, atol=1e-12)
    assert r1["ccf_lower"] < 1.0 < r1["ccf_upper"]
    assert r2["ccf_se"] > 0.0


def test_crossing_rule_forbids_ancestor_descendant_edge() -> None:
    # Clone 1: truncal [1.0, 1.0]
    # Clone 2: [0.80, 0.15] (dominant in R1, minor in R2)
    # Clone 3: [0.15, 0.80] (minor in R1, dominant in R2)
    # Crossing Rule forbids 2 -> 3 and 3 -> 2; unique valid tree is 1 -> 2 and 1 -> 3!
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "n_mutations": [60, 35, 35],
            "R1": [1.00, 0.80, 0.15],
            "R2": [1.00, 0.15, 0.80],
        }
    )
    tree_res = reconstruct_trees(cluster_df, ccf_buffer=0.10, settings="R_default")
    assert isinstance(tree_res, TreeResult)
    assert tree_res.n_alternative_trees == 1
    edges = set(map(tuple, tree_res.parent_child_df[["parent", "child"]].to_numpy()))
    assert edges == {(1, 2), (1, 3)}


def test_pigeonhole_sum_condition_enforces_nested_linear_chain() -> None:
    # Clone 1: [1.0, 1.0]
    # Clone 2: [0.75, 0.70]
    # Clone 3: [0.55, 0.50]
    # If both 2 and 3 were children of 1, child sum = [1.30, 1.20] > 1.0 + 0.10!
    # Therefore Pigeonhole Principle forces the unique linear chain 1 -> 2 -> 3!
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "n_mutations": [50, 30, 25],
            "R1": [1.00, 0.75, 0.55],
            "R2": [1.00, 0.70, 0.50],
        }
    )
    for mode in ("default", "R_default"):
        tree_res = reconstruct_trees(cluster_df, ccf_buffer=0.10, settings=mode)
        assert tree_res.n_alternative_trees == 1
        edges = set(map(tuple, tree_res.parent_child_df[["parent", "child"]].to_numpy()))
        assert edges == {(1, 2), (2, 3)}


def test_alternative_valid_spanning_trees_enumeration() -> None:
    # Clone 1: [1.0, 1.0], Clone 2: [0.55, 0.55], Clone 3: [0.30, 0.30]
    # Both branching (1->2, 1->3 since 0.55+0.30 = 0.85 <= 1.0) AND linear (1->2->3 since 0.30 <= 0.55)
    # are biologically valid!
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "n_mutations": [60, 30, 25],
            "R1": [1.00, 0.55, 0.30],
            "R2": [1.00, 0.55, 0.30],
        }
    )
    tree_res = reconstruct_trees(cluster_df, ccf_buffer=0.10, settings="R_default")
    assert tree_res.n_alternative_trees == 2
    tree_sets = [
        set(map(tuple, tdf[["parent", "child"]].to_numpy())) for tdf in tree_res.trees
    ]
    assert {(1, 2), (2, 3)} in tree_sets
    assert {(1, 2), (1, 3)} in tree_sets


def test_copy_number_error_cluster_pruning_restores_tree_feasibility() -> None:
    # Clones 1, 2, 3 form a clean branching tree (1->2 [0.75, 0.0], 1->3 [0.0, 0.75]).
    # Clone 4 is a small 6-mutation CN-error artifact at [0.70, 0.70]:
    # - 4 cannot parent 2 or 3 (since 0.70 < 0.75 in R1/R2 if tight buffer, or check sum:
    #   at [0.60, 0.60], 4 crosses 2 [0.85, 0.0] and 3 [0.0, 0.85]! Wait:
    #   2 has [0.85, 0.0], 3 has [0.0, 0.85], 4 has [0.60, 0.60]:
    #   - 4 crosses 2 (0.85 > 0.60 in R1, 0.60 > 0.0 in R2) => 2 and 4 cannot be ancestors!
    #   - 4 crosses 3 (0.60 > 0.0 in R1, 0.85 > 0.60 in R2) => 3 and 4 cannot be ancestors!
    #   - So 2, 3, AND 4 would ALL have to be direct children of 1, giving sum = 1.45 in BOTH R1 and R2 > 1.10!)
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4],
            "n_mutations": [70, 40, 40, 6],
            "R1": [1.00, 0.85, 0.00, 0.60],
            "R2": [1.00, 0.00, 0.85, 0.60],
        }
    )
    for mode in ("default", "R_default"):
        tree_res = reconstruct_trees(cluster_df, ccf_buffer=0.10, settings=mode)
        assert tree_res.dropped_clusters == [4]
        assert set(tree_res.retained_clusters) == {1, 2, 3}
        edges = set(map(tuple, tree_res.parent_child_df[["parent", "child"]].to_numpy()))
        assert edges == {(1, 2), (1, 3)}


def test_regional_clone_proportions_sum_to_one() -> None:
    # Exact tree: 1 -> 2, 1 -> 3, 2 -> 4
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4],
            "n_mutations": [60, 35, 30, 20],
            "R1": [1.00, 0.60, 0.30, 0.25],
            "R2": [1.00, 0.20, 0.70, 0.00],
        }
    )
    tree_res = reconstruct_trees(cluster_df, ccf_buffer=0.05, settings="R_default")
    props = tree_res.clone_proportions_df
    for reg in ("R1", "R2"):
        sub = props[props["region"] == reg].set_index("cluster_id")
        assert np.isclose(sub["clone_proportion"].sum(), 1.0, atol=1e-12)

    r1_sub = props[props["region"] == "R1"].set_index("cluster_id")["clone_proportion"]
    assert np.isclose(r1_sub.loc[1], 0.10, atol=1e-6)
    assert np.isclose(r1_sub.loc[2], 0.35, atol=1e-6)
    assert np.isclose(r1_sub.loc[3], 0.30, atol=1e-6)
    assert np.isclose(r1_sub.loc[4], 0.25, atol=1e-6)


def test_truncal_collapse_and_small_artifact_removal_in_clustering() -> None:
    rng = np.random.default_rng(12)
    # Two slightly offset truncal clouds ([0.98, 0.96] and [0.90, 0.91]) + one valid private clone + 3-mut artifact
    x_tr1 = rng.normal([0.98, 0.96], 0.02, size=(30, 2))
    x_tr2 = rng.normal([0.90, 0.91], 0.02, size=(25, 2))
    x_sub = rng.normal([0.45, 0.00], 0.02, size=(25, 2))
    x_sub[:, 1] = np.maximum(0.0, x_sub[:, 1])
    x_art = np.array([[0.0, 0.55], [0.0, 0.54], [0.0, 0.56]])

    all_x = np.vstack([x_tr1, x_tr2, x_sub, x_art])
    wide_df = pd.DataFrame(all_x, columns=["R1", "R2"])
    wide_df["mutation_id"] = [f"m_{i}" for i in range(len(wide_df))]

    res = cluster_ccf_multiregion(
        wide_df, max_clusters=8, min_cluster_size=5, seed=42, settings="R_default"
    )
    assert isinstance(res, ClusterResult)
    # Two truncal clouds collapse into 1 truncal cluster, private subclone kept, 3-mut artifact removed!
    assert res.n_clusters == 2
    assert res.n_removed_mutations == 3


def test_end_to_end_run_conipher_planted_three_region_phylogeny() -> None:
    purity = {"R1": 0.80, "R2": 0.75, "R3": 0.85}
    # Planted tree across 3 regions:
    # Clone 1 (truncal): [1.0, 1.0, 1.0] (50 muts)
    # Clone 2 (shared R1+R2 branch): [0.75, 0.70, 0.0] (35 muts)
    # Clone 3 (nested R1 child of Clone 2): [0.40, 0.0, 0.0] (30 muts)
    # Clone 4 (private R3 branch): [0.0, 0.0, 0.65] (35 muts)
    clone_specs = [
        (50, {"R1": 1.00, "R2": 1.00, "R3": 1.00}),
        (35, {"R1": 0.75, "R2": 0.70, "R3": 0.00}),
        (30, {"R1": 0.40, "R2": 0.00, "R3": 0.00}),
        (35, {"R1": 0.00, "R2": 0.00, "R3": 0.65}),
    ]
    muts_df = _simulate_multiregion_mutations(clone_specs, purity, depth=180, seed=101)

    for mode in ("default", "R_default"):
        res = run_conipher(
            muts_df, purity_by_region=purity, max_clusters=8, min_cluster_size=5, seed=42, settings=mode
        )
        assert isinstance(res, ConipherResult)
        assert len(res.retained_clusters) == 4
        assert len(res.dropped_clusters) == 0
        # Verify tree has 3 directed edges rooted at truncal_cluster and 1 grandchild at depth 2
        edges = res.parent_child_df
        assert len(edges) == 3
        parent_map = dict(zip(edges["child"], edges["parent"]))
        depths = {}
        for cid in res.retained_clusters:
            d = 0
            cur = cid
            while cur in parent_map:
                cur = parent_map[cur]
                d += 1
            depths[cid] = d
        assert sorted(depths.values()) == [0, 1, 1, 2]


def test_default_se_calibrated_tolerance_vs_r_default_fixed_buffer() -> None:
    # High-precision clusters (tiny SE = 0.008) where child sum exceeds parent by 0.08:
    # R_default (fixed ccf_buffer = 0.10) allows 1 -> 2 and 1 -> 3 simultaneously (1.08 <= 1.10),
    # whereas default SE-calibrated tolerance (max(0.05, 1.96 * se) = 0.05) rejects 1.08 > 1.05!
    long_df = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2, 3, 3],
            "region": ["R1", "R2", "R1", "R2", "R1", "R2"],
            "ccf": [1.00, 1.00, 0.58, 0.58, 0.50, 0.50],
            "ccf_se": [0.008, 0.008, 0.008, 0.008, 0.008, 0.008],
            "n_mutations": [80, 80, 40, 40, 40, 40],
        }
    )
    res_r = reconstruct_trees(long_df, ccf_buffer=0.10, settings="R_default")
    res_def = reconstruct_trees(long_df, ccf_buffer=0.10, settings="default")

    # R_default permits both branching AND linear (2 trees), while SE-calibrated default enforces linear (1 tree)!
    assert res_r.n_alternative_trees == 2
    assert res_def.n_alternative_trees == 1
    edges_def = set(map(tuple, res_def.parent_child_df[["parent", "child"]].to_numpy()))
    assert edges_def == {(1, 2), (2, 3)}


def test_conipher_class_methods_and_summary() -> None:
    purity = pd.Series({"R1": 0.75, "R2": 0.80})
    clone_specs = [
        (40, {"R1": 1.00, "R2": 1.00}),
        (25, {"R1": 0.60, "R2": 0.00}),
        (25, {"R1": 0.00, "R2": 0.55}),
    ]
    muts_df = _simulate_multiregion_mutations(clone_specs, {"R1": 0.75, "R2": 0.80}, depth=160, seed=7)

    model = CONIPHER(max_clusters=6, min_cluster_size=5, seed=42, settings="default")
    res = model.run(muts_df, purity_by_region=purity)
    summary = res.summary()
    df_tree = res.to_dataframe()

    assert summary["n_regions"] == 2
    assert summary["n_retained_clusters"] == 3
    assert len(df_tree) == 2


def test_greedy_smallest_cluster_drop_preference_between_conflicting_clones() -> None:
    # Clones 2 (50 muts) and 3 (8 muts) conflict via Sum Condition ([0.75, 0.75] + [0.70, 0.70] = 1.45 > 1.10)
    # while crossing in R1 vs R2 ([0.80, 0.55] vs [0.55, 0.80] => neither can parent the other, and sum = 1.35 > 1.10!).
    # Dropping EITHER 2 or 3 resolves the tree, and greedy smallest-cluster drop must drop Clone 3 (8 muts) and keep Clone 2 (50 muts)!
    cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "n_mutations": [80, 50, 8],
            "R1": [1.00, 0.80, 0.55],
            "R2": [1.00, 0.55, 0.80],
        }
    )
    res = reconstruct_trees(cluster_df, ccf_buffer=0.10, settings="R_default")
    assert res.dropped_clusters == [3]
    assert res.retained_clusters == [1, 2]

