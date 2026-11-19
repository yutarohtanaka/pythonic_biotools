"""`hdl`: High-Definition Likelihood heritability, genetic correlation, and enrichment.

Implements global HDL (Ning et al., Nature Genetics 2020), local HDL-L, and
functional-annotation stratified HDL.S.
"""

from __future__ import annotations

from .core import (
    HDLH2Result,
    HDLLocalResult,
    HDLRgResult,
    hdl_h2,
    hdl_local,
    hdl_rg,
)
from .panel import (
    LDEigenBlock,
    LDEigenPanel,
    build_ld_eigen_panel,
    ensure_panel,
    validate_settings,
)
from .stratified import HDLStratifiedResult, hdl_stratified

__all__ = [
    "LDEigenBlock",
    "LDEigenPanel",
    "HDLH2Result",
    "HDLRgResult",
    "HDLLocalResult",
    "HDLStratifiedResult",
    "build_ld_eigen_panel",
    "ensure_panel",
    "hdl_h2",
    "hdl_rg",
    "hdl_local",
    "hdl_stratified",
    "validate_settings",
]

__version__ = "0.1.0"
