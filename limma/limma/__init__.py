"""limma: Empirical Bayes moderated linear models & t-tests (Smyth 2004; Law et al. 2014; Phipson et al. 2016)."""
from .core import (
    ModeratedTestResult,
    fit_f_dist,
    limma_top_table,
    limma_trend,
    moderated_lm_contrast,
    moderated_ttest,
    squeeze_var,
    trigamma_inverse,
)

__all__ = [
    "ModeratedTestResult",
    "fit_f_dist",
    "limma_top_table",
    "limma_trend",
    "moderated_lm_contrast",
    "moderated_ttest",
    "squeeze_var",
    "trigamma_inverse",
]
