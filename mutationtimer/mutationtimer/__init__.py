"""
`mutationtimer`: Python implementation of MutationTimeR (Gerstung et al., Nature 2020)
for evolutionary timing of somatic mutations, copy-number gains, and Whole-Genome Doubling.
"""

from __future__ import annotations

from .core import (
    MutationTimeR,
    MutationTimeResult,
    beta_binomial_logpmf,
    expected_vaf,
    mutation_time,
)
from .io import (
    MutationTimerConfig,
    SettingsMode,
    canonical_chrom,
    map_variants_to_segments,
    normal_copy_number,
    validate_settings,
)
from .timing import (
    WGDTimingResult,
    estimate_wgd_timing,
    pi_to_all_times,
    pi_to_time,
    time_cn_gains,
)

__all__ = [
    "SettingsMode",
    "MutationTimerConfig",
    "MutationTimeResult",
    "WGDTimingResult",
    "MutationTimeR",
    "mutation_time",
    "pi_to_time",
    "pi_to_all_times",
    "time_cn_gains",
    "estimate_wgd_timing",
    "expected_vaf",
    "beta_binomial_logpmf",
    "canonical_chrom",
    "normal_copy_number",
    "map_variants_to_segments",
    "validate_settings",
]

__version__ = "0.1.0"
