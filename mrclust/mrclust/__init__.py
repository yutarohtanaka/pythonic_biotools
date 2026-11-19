"""
`mrclust`: Python implementation of MR-Clust (Foley et al., Bioinformatics 2021)
for clustering genetic variants in Mendelian randomization with similar causal estimates.
"""

from .core import (
    ClusterFit,
    ClusterInfo,
    MRClustResult,
    compute_ratio_estimates,
    mr_clust_em,
    pr_clust,
)
from .plots import two_stage_plot

__all__ = [
    "ClusterFit",
    "ClusterInfo",
    "MRClustResult",
    "compute_ratio_estimates",
    "mr_clust_em",
    "pr_clust",
    "two_stage_plot",
]

__version__ = "0.1.0"
