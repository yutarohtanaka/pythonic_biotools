"""
`numbat`: Haplotype-aware analysis of somatic copy number variations and tumor
phylogenies from single-cell transcriptomes (Gao et al., Nature Biotechnology 2023).
"""

from .clones import SegmentCall, score_single_cells
from .config import COLLAPSED_STATES, DEFAULT_STATES, CNVStateSpec, NumbatConfig
from .core import NumbatResult, annotate_anndata, run_numbat
from .expression import (
    compute_diploid_offset,
    compute_ref_expression,
    gaussian_logratio_logpdf,
    pln_logpmf,
)
from .hmm import (
    HMMResult,
    beta_binomial_logpmf,
    fit_allele_hmm,
    fit_expression_hmm,
    has_numba,
    phase_hmm_loglik,
    phase_switch_prob,
    run_joint_hmm,
)
from .phylogeny import (
    CloneAssignment,
    PhylogenyTree,
    compute_clone_posteriors,
    reconstruct_phylogeny,
)

__all__ = [
    "COLLAPSED_STATES",
    "DEFAULT_STATES",
    "CNVStateSpec",
    "NumbatConfig",
    "HMMResult",
    "SegmentCall",
    "CloneAssignment",
    "PhylogenyTree",
    "NumbatResult",
    "run_numbat",
    "run_joint_hmm",
    "fit_allele_hmm",
    "fit_expression_hmm",
    "reconstruct_phylogeny",
    "score_single_cells",
    "compute_clone_posteriors",
    "compute_ref_expression",
    "compute_diploid_offset",
    "pln_logpmf",
    "gaussian_logratio_logpdf",
    "beta_binomial_logpmf",
    "phase_switch_prob",
    "phase_hmm_loglik",
    "annotate_anndata",
    "has_numba",
]

__version__ = "0.1.0"
