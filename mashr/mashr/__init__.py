"""
`mashr`: Clean-room Python implementation of Multivariate Adaptive Shrinkage
(Urbut et al., *Nature Genetics* 2019) and Bovy's Extreme Deconvolution EM
(Bovy, Hogg & Roweis, *Annals of Applied Statistics* 2011).
"""

from .covariances import (
    ExtremeDeconvolutionResult,
    cov_canonical,
    cov_ed,
    cov_pca,
    fit_extreme_deconvolution,
)
from .data import (
    MashData,
    estimate_null_correlation_simple,
    mash_set_data,
    mash_update_data,
)
from .posterior import (
    autoselect_grid,
    compute_loglik_matrix,
    compute_posterior_matrices,
    mash,
    optimize_pi,
)
from .results import (
    MashResult,
    get_estimated_pi,
    get_lfsr,
    get_n_significant_conditions,
    get_pairwise_sharing,
    get_pm,
    get_psd,
    get_significant_results,
)

__all__ = [
    "MashData",
    "mash_set_data",
    "mash_update_data",
    "estimate_null_correlation_simple",
    "cov_canonical",
    "cov_pca",
    "cov_ed",
    "fit_extreme_deconvolution",
    "ExtremeDeconvolutionResult",
    "autoselect_grid",
    "compute_loglik_matrix",
    "optimize_pi",
    "compute_posterior_matrices",
    "mash",
    "MashResult",
    "get_significant_results",
    "get_n_significant_conditions",
    "get_pairwise_sharing",
    "get_pm",
    "get_psd",
    "get_lfsr",
    "get_estimated_pi",
]

__version__ = "0.1.0"
