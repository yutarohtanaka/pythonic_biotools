"""
`fishhook`: Gamma-Poisson somatic mutation enrichment modeling across genomic
intervals and covariates (Imielinski, Guo & Meyerson, Cell 2017).
"""

from .core import FishHookModel, FishHookResult, fit_fishhook
from .intervals import compute_territory_and_counts, merge_eligible_by_chrom

__all__ = [
    "FishHookModel",
    "FishHookResult",
    "fit_fishhook",
    "compute_territory_and_counts",
    "merge_eligible_by_chrom",
]

__version__ = "0.1.0"
