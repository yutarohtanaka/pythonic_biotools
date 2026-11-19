"""
`genomicsem`: Python implementation of GenomicSEM (Grotzinger et al., Nature Human
Behaviour 2019) for multivariate LDSC covariance modeling, Reticular Action Model
(RAM) structural equation modeling, and vectorized common-factor SNP GWAS.
"""

from __future__ import annotations

from .gwas import GenomicSEM, commonfactor_gwas
from .ldsc import (
    GenomicCovStruct,
    duplication_matrix,
    ldsc_covstruct,
    pinv_eig_floor,
    stabilize_psd,
    unvech,
    vech,
    vech_indices,
)
from .model import (
    UserModelResult,
    commonfactor,
    ram_implied_covariance_and_jacobian,
    usermodel,
)
from .syntax import ModelTermSpec, RAMSpec, parse_lavaan_syntax

__all__ = [
    "GenomicCovStruct",
    "GenomicSEM",
    "ModelTermSpec",
    "RAMSpec",
    "UserModelResult",
    "commonfactor",
    "commonfactor_gwas",
    "duplication_matrix",
    "ldsc_covstruct",
    "parse_lavaan_syntax",
    "pinv_eig_floor",
    "ram_implied_covariance_and_jacobian",
    "stabilize_psd",
    "unvech",
    "usermodel",
    "vech",
    "vech_indices",
]

__version__ = "0.1.0"
