"""`cismr`: Correlated drug-target cis-Mendelian Randomization.

Implements PC-GMM & MV-PC-GMM (Burgess et al., Genetic Epidemiology 2020;
Grant & Burgess, Statistics in Medicine 2022), MR-SPI (Yao et al., Biometrics 2024),
and MR-GENIUS (Tchetgen Tchetgen et al., Statistical Science 2021).
"""

from __future__ import annotations

from .core import CisMR, CisMRReport
from .genius import MRGeniusResult, mr_genius, mr_genius_summary
from .pcgmm import MVPCGMMResult, PCGMMResult, mr_mvpcgmm, mr_pcgmm, mr_pcivw
from .spi import MRSPIResult, mr_spi
from .utils import prepare_ld_matrix, validate_settings

__all__ = [
    "CisMR",
    "CisMRReport",
    "PCGMMResult",
    "MVPCGMMResult",
    "MRSPIResult",
    "MRGeniusResult",
    "mr_pcgmm",
    "mr_pcivw",
    "mr_mvpcgmm",
    "mr_spi",
    "mr_genius",
    "mr_genius_summary",
    "prepare_ld_matrix",
    "validate_settings",
]

__version__ = "0.1.0"
