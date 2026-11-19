"""satuRn: Quasi-binomial Empirical Bayes differential transcript usage."""

from .core import (
    DTUFit,
    FilteredDTUData,
    filter_dtu,
    fit_dtu,
    stagewise_dtu,
    test_dtu,
)
from .eb import (
    SqueezeVarResult,
    estimate_empirical_null,
    squeeze_var,
    trigamma_inverse,
    validate_settings,
)

__version__ = "0.1.0"

__all__ = [
    "DTUFit",
    "FilteredDTUData",
    "SqueezeVarResult",
    "estimate_empirical_null",
    "filter_dtu",
    "fit_dtu",
    "squeeze_var",
    "stagewise_dtu",
    "test_dtu",
    "trigamma_inverse",
    "validate_settings",
]
