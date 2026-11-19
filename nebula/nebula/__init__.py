"""
`nebula`: Fast negative binomial mixed models for multi-subject single-cell
differential and co-expression analysis (He et al., Communications Biology 2021).
"""

from .core import (
    GroupedCellData,
    GroupedData,
    NebulaResult,
    group_cell,
    nebula,
)

__all__ = [
    "GroupedCellData",
    "GroupedData",
    "NebulaResult",
    "group_cell",
    "nebula",
]

__version__ = "0.1.0"
