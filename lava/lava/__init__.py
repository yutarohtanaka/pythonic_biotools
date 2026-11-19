"""
`lava`: Python implementation of LAVA (Werme et al., Nature Genetics 2022)
for univariate local heritability, bivariate and partial local genetic correlation,
and multivariate local genetic multiple regression.
"""

from __future__ import annotations

from .core import LAVA, run_bivar, run_multireg, run_pcor, run_univ
from .locus import LAVALocus, process_locus, project_psd_matrix
from .simulation import (
    bivariate_null_pvalue,
    omnibus_multireg_null_pvalue,
    simulate_wishart_omega,
)

__all__ = [
    "LAVA",
    "LAVALocus",
    "bivariate_null_pvalue",
    "omnibus_multireg_null_pvalue",
    "process_locus",
    "project_psd_matrix",
    "run_bivar",
    "run_multireg",
    "run_pcor",
    "run_univ",
    "simulate_wishart_omega",
]

__version__ = "0.1.0"
