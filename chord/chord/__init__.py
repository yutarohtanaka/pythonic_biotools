"""
`chord`: Clean-room Python implementation of CHORD and mutSigExtractor
(Nguyen et al., Nature Communications 2020) for pan-cancer HRD detection and
BRCA1-type vs. BRCA2-type classification from SNV, indel, and SV contexts.
"""

from __future__ import annotations

from .channels import (
    ALL_CHORD_CHANNELS,
    INDEL_CHANNELS,
    INDEL_MH_DEL_CHANNELS,
    INDEL_NONE_CHANNELS,
    INDEL_REP_DEL_CHANNELS,
    INDEL_REP_INS_CHANNELS,
    SBS96_CHANNELS,
    SV_CHANNELS,
    classify_indel_sequence,
    classify_sv_channel,
    normalize_sbs96_context,
    revcomp,
)
from .classifier import (
    CHORDClassifier,
    CHORDResult,
    align_contexts_df,
    chord_predict,
    compute_chord_features,
    simulate_chord_archetypes,
)
from .config import CHORDConfig, SettingsMode, get_chord_config, validate_settings
from .extract import extract_sigs_chord

__all__ = [
    "SettingsMode",
    "CHORDConfig",
    "get_chord_config",
    "validate_settings",
    "SBS96_CHANNELS",
    "INDEL_CHANNELS",
    "INDEL_REP_DEL_CHANNELS",
    "INDEL_REP_INS_CHANNELS",
    "INDEL_MH_DEL_CHANNELS",
    "INDEL_NONE_CHANNELS",
    "SV_CHANNELS",
    "ALL_CHORD_CHANNELS",
    "revcomp",
    "normalize_sbs96_context",
    "classify_indel_sequence",
    "classify_sv_channel",
    "extract_sigs_chord",
    "CHORDResult",
    "CHORDClassifier",
    "chord_predict",
    "align_contexts_df",
    "compute_chord_features",
    "simulate_chord_archetypes",
]

__version__ = "0.1.0"
