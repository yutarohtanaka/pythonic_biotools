"""
`scarhrd`: Genomic scar Homologous Recombination Deficiency (HRD) scoring
(`hrd_loh`, `tai`, `lst`, and clinical `hrd_sum`; Sztupinszki et al., 2018).
"""

from .core import ScarHRDResult, calculate_hrd, calculate_hrd_cohort
from .reference import (
    ACROCENTRIC_CHROMS,
    GRCH37_REFERENCE,
    GRCH38_REFERENCE,
    ChromArmReference,
    canonical_hrd_chrom,
    get_genome_reference,
)

__all__ = [
    "ScarHRDResult",
    "calculate_hrd",
    "calculate_hrd_cohort",
    "ChromArmReference",
    "GRCH38_REFERENCE",
    "GRCH37_REFERENCE",
    "ACROCENTRIC_CHROMS",
    "canonical_hrd_chrom",
    "get_genome_reference",
]

__version__ = "0.1.0"
