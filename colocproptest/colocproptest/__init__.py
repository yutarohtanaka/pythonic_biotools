"""
`colocproptest`: Clean-room Python implementation of colocPropTest
(Wallace 2013; Wallace, Robins & Johnson 2025) for LD-robust frequentist
proportional colocalization testing under multiple causal variants and
differential SNP sample sizes.
"""

from __future__ import annotations

from .core import (
    ColocPropTest,
    ColocPropTestResult,
    acat_combine,
    bh_fdr,
    run_coloc_prop_test,
)
from .pairs import (
    PairPropTestResult,
    SettingsMode,
    select_tester_pairs,
    test_pair_proportionality,
    validate_settings,
)
from .pc import PCPropTestResult, proptest_pc

__all__ = [
    "SettingsMode",
    "validate_settings",
    "PairPropTestResult",
    "select_tester_pairs",
    "test_pair_proportionality",
    "PCPropTestResult",
    "proptest_pc",
    "bh_fdr",
    "acat_combine",
    "ColocPropTestResult",
    "ColocPropTest",
    "run_coloc_prop_test",
]

__version__ = "0.1.0"
