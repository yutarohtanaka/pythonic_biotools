"""
`cibersortx`: Clean-room Python implementation of CIBERSORT and CIBERSORTx
(Newman et al., Nature Methods 2015; Nature Biotechnology 2019).
"""

from .batch import (
    BatchAlignmentResult,
    b_mode_adjust,
    combat_align,
    s_mode_adjust,
)
from .deconvolve import (
    DeconvolutionResult,
    cibersort,
    deconvolve,
)
from .gep import (
    GroupGEPResult,
    HighResGEPResult,
    impute_gep_group,
    impute_gep_highres,
)
from .signature import (
    SignatureMatrixResult,
    build_signature_matrix,
)

__all__ = [
    "SignatureMatrixResult",
    "DeconvolutionResult",
    "BatchAlignmentResult",
    "GroupGEPResult",
    "HighResGEPResult",
    "build_signature_matrix",
    "deconvolve",
    "cibersort",
    "combat_align",
    "b_mode_adjust",
    "s_mode_adjust",
    "impute_gep_group",
    "impute_gep_highres",
]

__version__ = "0.1.0"
