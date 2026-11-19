"""
End-to-end `run_conipher` pipeline and `CONIPHER` class combining multi-region
PhyloCCF calculation, Variational Dirichlet Process clustering, and phylogenetic
tree reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .ccf import compute_phylo_ccf
from .clustering import ClusterResult, cluster_ccf_multiregion
from .io import ConipherConfig, SettingsMode, validate_settings
from .tree import TreeResult, reconstruct_trees


@dataclass
class ConipherResult:
    """Complete multi-region phylogenetic output from `run_conipher` / `CONIPHER`."""

    phylo_ccf_df: pd.DataFrame
    cluster_ccf_df: pd.DataFrame
    mutation_assignments_df: pd.DataFrame
    parent_child_df: pd.DataFrame
    trees: list[pd.DataFrame]
    n_alternative_trees: int
    truncal_cluster: int
    retained_clusters: list[int]
    dropped_clusters: list[int]
    clone_proportions_df: pd.DataFrame
    cluster_result: ClusterResult
    tree_result: TreeResult
    settings: SettingsMode

    def summary(self) -> dict[str, Any]:
        """Return dictionary summary of multi-region CONIPHER inference."""
        return {
            "n_mutations": int(len(self.mutation_assignments_df)),
            "n_regions": len(self.cluster_result.regions),
            "regions": self.cluster_result.regions,
            "n_initial_clusters": self.cluster_result.n_clusters,
            "n_retained_clusters": len(self.retained_clusters),
            "dropped_clusters": self.dropped_clusters,
            "truncal_cluster": self.truncal_cluster,
            "n_alternative_trees": self.n_alternative_trees,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return primary phylogenetic tree edge list (`parent_child_df`)."""
        return self.parent_child_df.copy()


def run_conipher(
    mutations_df: pd.DataFrame,
    purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
    max_clusters: int = 10,
    min_cluster_size: int = 5,
    ccf_buffer: float = 0.10,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> ConipherResult:
    """
    Execute the complete CONIPHER multi-region somatic phylogenetic pipeline
    (Grigoriadis et al., Nature Protocols 2023):
    1. `compute_phylo_ccf` across `R >= 2` tumor regions,
    2. `cluster_ccf_multiregion` via Variational Dirichlet Process clustering,
       truncal collapse, and artifact filtering,
    3. `reconstruct_trees` enforcing the Sum Condition (Pigeonhole Principle)
       and Crossing Rule with iterative CN-error cluster pruning.

    Parameters
    ----------
    mutations_df : pd.DataFrame
        Multi-region mutation counts and allele-specific copy numbers.
    purity_by_region : Mapping[str, float] | pd.Series | pd.DataFrame
        Regional tumor purities `rho_r in (0, 1]` across `R >= 2` regions.
    max_clusters : int, default=10
        Maximum candidate clusters in DP mixture clustering.
    min_cluster_size : int, default=5
        Minimum mutation count per retained cluster.
    ccf_buffer : float, default=0.10
        CCF tolerance (`[0, 1]` scale) for tree constraints.
    seed : int, default=42
        Random seed for deterministic clustering.
    settings : {"default", "R_default"}, default="default"
        `"default"` uses SE-calibrated Variational DP clustering and SE-calibrated
        pigeonhole tolerances; `"R_default"` reproduces CONIPHER v2.2 fixed
        `ccf_buffer = 0.10`, `mean_ccf >= 0.85` truncal collapse, and greedy
        smallest-cluster pruning.

    Returns
    -------
    ConipherResult
        PhyloCCF table, cluster profiles, primary + alternative spanning trees,
        and regional subclone proportions.
    """
    validate_settings(settings)
    cfg = ConipherConfig(
        max_clusters=int(max_clusters),
        min_cluster_size=int(min_cluster_size),
        ccf_buffer=float(ccf_buffer),
        seed=int(seed),
        settings=settings,
    )

    phylo_df = compute_phylo_ccf(
        mutations_df=mutations_df,
        purity_by_region=purity_by_region,
        settings=cfg.settings,
    )
    clust_res = cluster_ccf_multiregion(
        ccf_df=phylo_df,
        max_clusters=cfg.max_clusters,
        min_cluster_size=cfg.min_cluster_size,
        seed=cfg.seed,
        settings=cfg.settings,
    )
    tree_res = reconstruct_trees(
        cluster_ccf_df=clust_res,
        mutation_assignments=clust_res.mutation_assignments_df,
        ccf_buffer=cfg.ccf_buffer,
        settings=cfg.settings,
    )

    return ConipherResult(
        phylo_ccf_df=phylo_df,
        cluster_ccf_df=tree_res.cluster_ccf_df,
        mutation_assignments_df=clust_res.mutation_assignments_df,
        parent_child_df=tree_res.parent_child_df,
        trees=tree_res.trees,
        n_alternative_trees=tree_res.n_alternative_trees,
        truncal_cluster=tree_res.truncal_cluster,
        retained_clusters=tree_res.retained_clusters,
        dropped_clusters=tree_res.dropped_clusters,
        clone_proportions_df=tree_res.clone_proportions_df,
        cluster_result=clust_res,
        tree_result=tree_res,
        settings=cfg.settings,
    )


class CONIPHER:
    """
    Object-oriented interface for CONIPHER multi-region somatic mutation
    clustering and phylogenetic tree reconstruction.
    """

    def __init__(
        self,
        max_clusters: int = 10,
        min_cluster_size: int = 5,
        ccf_buffer: float = 0.10,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings = validate_settings(settings)
        self.config = ConipherConfig(
            max_clusters=int(max_clusters),
            min_cluster_size=int(min_cluster_size),
            ccf_buffer=float(ccf_buffer),
            seed=int(seed),
            settings=self.settings,
        )
        self.result_: ConipherResult | None = None

    def compute_phylo_ccf(
        self,
        mutations_df: pd.DataFrame,
        purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute multi-region PhyloCCF and standard errors."""
        return compute_phylo_ccf(
            mutations_df=mutations_df,
            purity_by_region=purity_by_region,
            settings=self.settings,
        )

    def cluster(self, ccf_df: pd.DataFrame) -> ClusterResult:
        """Cluster multi-region PhyloCCF profiles."""
        return cluster_ccf_multiregion(
            ccf_df=ccf_df,
            max_clusters=self.config.max_clusters,
            min_cluster_size=self.config.min_cluster_size,
            seed=self.config.seed,
            settings=self.settings,
        )

    def reconstruct_trees(
        self,
        cluster_ccf_df: pd.DataFrame | ClusterResult,
        mutation_assignments: pd.DataFrame | None = None,
    ) -> TreeResult:
        """Reconstruct valid rooted phylogenetic trees from cluster CCFs."""
        return reconstruct_trees(
            cluster_ccf_df=cluster_ccf_df,
            mutation_assignments=mutation_assignments,
            ccf_buffer=self.config.ccf_buffer,
            settings=self.settings,
        )

    def fit(
        self,
        mutations_df: pd.DataFrame,
        purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> ConipherResult:
        """Fit end-to-end CONIPHER pipeline on multi-region somatic mutations."""
        self.result_ = run_conipher(
            mutations_df=mutations_df,
            purity_by_region=purity_by_region,
            max_clusters=self.config.max_clusters,
            min_cluster_size=self.config.min_cluster_size,
            ccf_buffer=self.config.ccf_buffer,
            seed=self.config.seed,
            settings=self.settings,
        )
        return self.result_

    def run(
        self,
        mutations_df: pd.DataFrame,
        purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> ConipherResult:
        """Alias for `.fit(...)`."""
        return self.fit(mutations_df=mutations_df, purity_by_region=purity_by_region)
