"""`mrlap`: Analytical MR correction for sample overlap, weak-instrument bias, and Winner's curse (Mounier & Kutalik, 2023)."""

from .architecture import (
    GeneticArchitectureResult,
    estimate_bivariate_ldsc_with_jackknife,
    estimate_exposure_polygenicity,
    truncated_moments_closed_form,
    truncated_moments_gauss_hermite,
)
from .core import MRlap, MRlapResult, mrlap

__all__ = [
    "MRlap",
    "MRlapResult",
    "GeneticArchitectureResult",
    "mrlap",
    "estimate_bivariate_ldsc_with_jackknife",
    "estimate_exposure_polygenicity",
    "truncated_moments_closed_form",
    "truncated_moments_gauss_hermite",
]

__version__ = "0.1.0"
