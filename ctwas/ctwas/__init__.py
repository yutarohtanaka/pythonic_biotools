"""
`ctwas`: Clean-room Python implementation of cTWAS (Zhao et al., Nature Genetics 2024)
for causal Transcriptome/Proteome-Wide Association Studies adjusting for LD
hitchhiking and direct non-coding SNP pleiotropy via Empirical Bayes and joint
group-prior SuSiE-RSS fine-mapping.
"""

from __future__ import annotations

from .core import CTWAS, CTWASResult, run_ctwas
from .impute import ImputedRegionData, SettingsMode, impute_expr_z, validate_settings
from .priors import CTWASPriors, estimate_ctwas_priors
from .susie_rss import SuSiERSSGroupResult, regularize_ld_matrix, susie_rss_group_prior

__all__ = [
    "SettingsMode",
    "validate_settings",
    "ImputedRegionData",
    "impute_expr_z",
    "CTWASPriors",
    "estimate_ctwas_priors",
    "SuSiERSSGroupResult",
    "regularize_ld_matrix",
    "susie_rss_group_prior",
    "CTWASResult",
    "CTWAS",
    "run_ctwas",
]

__version__ = "0.1.0"
