"""
`dndscv`: Clean-room Python implementation of dNdScv (Martincorena et al., Cell 2017)
for maximum-likelihood somatic selection (dN/dS) analysis in cancer.
"""

from .annotate import AnnotationResult, annotate_mutations
from .core import DnDScvModel, DnDScvResult, dndscv, fit_dndscv
from .refcds import (
    IMPACT_CLASSES,
    TRINUC_CONTEXTS,
    GeneRef,
    RefCDS,
    build_refcds,
    build_refcds_from_gtf,
    build_synthetic_refcds,
    simulate_refcds,
)

__all__ = [
    "TRINUC_CONTEXTS",
    "IMPACT_CLASSES",
    "GeneRef",
    "RefCDS",
    "AnnotationResult",
    "DnDScvModel",
    "DnDScvResult",
    "build_refcds",
    "build_refcds_from_gtf",
    "build_synthetic_refcds",
    "simulate_refcds",
    "annotate_mutations",
    "dndscv",
    "fit_dndscv",
]

__version__ = "0.1.0"
