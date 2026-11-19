"""
`purecn`: Clean-room Python implementation of PureCN (Riester et al., Source Code
for Biology and Medicine 2016) for tumor-only clinical panel purity/ploidy
optimization, somatic-vs-germline classification, LOH, clonality (CCF), and TMB.
"""

from __future__ import annotations

from .config import PureCNConfig, SettingsMode, get_purecn_config, validate_settings
from .core import PureCN, PureCNResult, run_purecn
from .coverage import (
    CoverageNormalizationResult,
    gc_correct_log2,
    normalize_tumor_only_coverage,
    segment_log2_ratios,
    tangent_normalize,
)
from .likelihood import (
    PureCNSolution,
    beta_binomial_logpmf,
    compute_somatic_priors,
    estimate_overdispersion,
    expected_log2_ratio,
    grid_search_purity_ploidy,
)
from .metrics import CINValue, TMBValue, call_cin, call_tmb, compute_loh_fraction
from .somatic import predict_somatic

__all__ = [
    "SettingsMode",
    "PureCNConfig",
    "get_purecn_config",
    "validate_settings",
    "CoverageNormalizationResult",
    "PureCNSolution",
    "PureCNResult",
    "PureCN",
    "TMBValue",
    "CINValue",
    "normalize_tumor_only_coverage",
    "gc_correct_log2",
    "tangent_normalize",
    "segment_log2_ratios",
    "expected_log2_ratio",
    "beta_binomial_logpmf",
    "compute_somatic_priors",
    "estimate_overdispersion",
    "grid_search_purity_ploidy",
    "predict_somatic",
    "call_tmb",
    "call_cin",
    "compute_loh_fraction",
    "run_purecn",
]

__version__ = "0.1.0"
