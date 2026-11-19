"""
`conipher`: Python implementation of CONIPHER (Grigoriadis et al., Nature
Protocols 2023) for multi-region somatic mutation PhyloCCF clustering and
phylogenetic tree reconstruction.
"""

from __future__ import annotations

from .ccf import compute_phylo_ccf
from .clustering import ClusterResult, cluster_ccf_multiregion
from .core import CONIPHER, ConipherResult, run_conipher
from .io import ConipherConfig, SettingsMode, normalize_purity_map, validate_settings
from .tree import TreeResult, reconstruct_trees

__all__ = [
    "SettingsMode",
    "ConipherConfig",
    "ClusterResult",
    "TreeResult",
    "ConipherResult",
    "CONIPHER",
    "compute_phylo_ccf",
    "cluster_ccf_multiregion",
    "reconstruct_trees",
    "run_conipher",
    "normalize_purity_map",
    "validate_settings",
]

__version__ = "0.1.0"
