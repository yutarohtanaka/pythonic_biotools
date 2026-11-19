"""CSIDE: Cell-type-specific differential expression in spatial transcriptomics."""

from .core import (
    CSIDEConfig,
    CSIDEResult,
    run_cside,
    run_cside_nonparam,
    run_cside_regions,
    run_cside_single,
)
from .likelihood import (
    estimate_sigma_gene,
    fit_gene_cside,
    poisson_lognormal_moments,
)
from .splines import build_spline_design, validate_settings

__version__ = "0.1.0"

__all__ = [
    "CSIDEConfig",
    "CSIDEResult",
    "build_spline_design",
    "estimate_sigma_gene",
    "fit_gene_cside",
    "poisson_lognormal_moments",
    "run_cside",
    "run_cside_nonparam",
    "run_cside_regions",
    "run_cside_single",
    "validate_settings",
]
