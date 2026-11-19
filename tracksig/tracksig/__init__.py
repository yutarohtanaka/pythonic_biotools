"""
`tracksig`: Clean-room Python implementation of TrackSig and TrackSigFreq
(Rubanova et al., Nature Communications 2020; Harrigan et al., PSB 2020)
for evolutionary pseudotime changepoint segmentation of mutational signatures.
"""

from __future__ import annotations

from .ccf import COSMIC_96_CONTEXTS, SettingsMode, compute_ccf_pseudotime, validate_settings
from .core import TrackSig, TrackSigResult, tracksig
from .em import (
    MultinomialEMResult,
    detect_active_signatures,
    fit_multinomial_signatures_em,
    normalize_signature_matrix,
)
from .pelt import SegmentScorer, run_pelt

__all__ = [
    "COSMIC_96_CONTEXTS",
    "SettingsMode",
    "validate_settings",
    "compute_ccf_pseudotime",
    "MultinomialEMResult",
    "normalize_signature_matrix",
    "fit_multinomial_signatures_em",
    "detect_active_signatures",
    "SegmentScorer",
    "run_pelt",
    "TrackSigResult",
    "TrackSig",
    "tracksig",
]

__version__ = "0.1.0"
