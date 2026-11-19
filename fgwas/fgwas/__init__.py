"""`fgwas`: Hierarchical functional annotation enrichment and Bayesian GWAS fine-mapping.

Implements Pickrell (2014, AJHG) Wakefield log-ABF calculation, hierarchical
regional + within-locus softmax enrichment MLE, fine-mapping posterior
re-weighting, cross-validated ridge penalty tuning (-xv), and stepwise selection.
"""

from __future__ import annotations

from .abf import normalize_prior_variances, validate_settings, wakefield_lnabf
from .core import FgwasResult, fit_fgwas
from .selection import (
    CrossValFgwasResult,
    StepwiseFgwasResult,
    cross_validate_fgwas,
    stepwise_fgwas,
)

__all__ = [
    "FgwasResult",
    "CrossValFgwasResult",
    "StepwiseFgwasResult",
    "wakefield_lnabf",
    "fit_fgwas",
    "cross_validate_fgwas",
    "stepwise_fgwas",
    "normalize_prior_variances",
    "validate_settings",
]

__version__ = "0.1.0"
