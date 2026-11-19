"""
`bisque`: Clean-room Python implementation of Bisque (Jew et al., Nature Communications 2020)
for cross-platform reference-based and marker-based bulk RNA-seq deconvolution.
"""

from .marker import MarkerDeconvolutionResult, marker_based_decomposition
from .reference import (
    ReferenceDeconvolutionResult,
    SCReference,
    generate_sc_reference,
    reference_based_decomposition,
)

__all__ = [
    "ReferenceDeconvolutionResult",
    "MarkerDeconvolutionResult",
    "SCReference",
    "reference_based_decomposition",
    "marker_based_decomposition",
    "generate_sc_reference",
]

__version__ = "0.1.0"
