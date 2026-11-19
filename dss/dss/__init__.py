"""DSS & DMRcate: Beta-Binomial Empirical Bayes differential DNA methylation."""

from .core import (
    BSData,
    DSSConfig,
    call_dml,
    call_dmr,
    dml_test,
    make_bsseq_data,
)
from .dispersion import estimate_prior_hyperparams, shrink_dispersion_bb
from .dmrcate import dmrcate
from .multifactor import MultifactorFit, dml_fit_multifactor, dml_test_multifactor
from .smoothing import (
    compute_group_mean_and_var,
    smooth_counts_tricube,
    validate_settings,
)

__version__ = "0.1.0"

__all__ = [
    "BSData",
    "DSSConfig",
    "MultifactorFit",
    "call_dml",
    "call_dmr",
    "compute_group_mean_and_var",
    "dml_fit_multifactor",
    "dml_test",
    "dml_test_multifactor",
    "dmrcate",
    "estimate_prior_hyperparams",
    "make_bsseq_data",
    "shrink_dispersion_bb",
    "smooth_counts_tricube",
    "validate_settings",
]
