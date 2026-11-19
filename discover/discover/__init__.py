"""
`discover`: Discrete Independence Statistic Controlling for Observations with
Varying Event Rates (Canisius, Martens & Wessels, Genome Biology 2016).
"""

from .core import (
    DiscoverMatrix,
    GroupwiseDiscoverResult,
    PairwiseDiscoverResult,
    fit_discover_background,
    groupwise_discover_test,
    pairwise_discover_test,
)
from .poisson_binomial import (
    bh_fdr,
    discover_fdr,
    poisson_binomial_pmf,
    poisson_binomial_pval,
)

__all__ = [
    "DiscoverMatrix",
    "PairwiseDiscoverResult",
    "GroupwiseDiscoverResult",
    "fit_discover_background",
    "pairwise_discover_test",
    "groupwise_discover_test",
    "poisson_binomial_pmf",
    "poisson_binomial_pval",
    "discover_fdr",
    "bh_fdr",
]

__version__ = "0.1.0"
