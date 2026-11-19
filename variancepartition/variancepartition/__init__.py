"""
`variancepartition`: Mixed-effects linear modeling, ICC variance decomposition,
and `dream` Empirical Bayes differential expression (Hoffman & Schadt,
BMC Bioinformatics 2016; Hoffman & Roussos, Bioinformatics 2021).
"""

from .dream import DreamFit, DreamResult, dream, e_bayes, eBayes, top_table, topTable
from .ebayes import fit_f_dist, squeeze_var, trigamma_inverse
from .formula import DesignSpec, build_design, make_contrasts
from .lmm import LMMFitResult, fit_lmm, fit_var_part_model
from .varpart import VarPartResult, extract_var_part, fit_extract_var_part_model

__all__ = [
    "DesignSpec",
    "LMMFitResult",
    "VarPartResult",
    "DreamFit",
    "DreamResult",
    "build_design",
    "make_contrasts",
    "fit_lmm",
    "fit_var_part_model",
    "extract_var_part",
    "fit_extract_var_part_model",
    "dream",
    "e_bayes",
    "eBayes",
    "top_table",
    "topTable",
    "fit_f_dist",
    "squeeze_var",
    "trigamma_inverse",
]

__version__ = "0.1.0"
