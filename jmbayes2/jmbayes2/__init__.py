"""jmbayes2: Bayesian Joint Longitudinal and Survival / Competing-Risks Modeling."""

from .lme import LMEFit, fit_lme
from .mcmc import (
    JMBayes2,
    JMModelSpec,
    JMResult,
    VALID_FUNCTIONAL_FORMS,
    jm,
)
from .predict import predict_dynamic
from .quadrature import (
    BSplineHazardBasis,
    GK15_NODES,
    GK15_WEIGHTS,
    SettingsMode,
    VALID_SETTINGS,
    gauss_kronrod_15,
    validate_settings,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SettingsMode",
    "VALID_SETTINGS",
    "VALID_FUNCTIONAL_FORMS",
    "validate_settings",
    "GK15_NODES",
    "GK15_WEIGHTS",
    "gauss_kronrod_15",
    "BSplineHazardBasis",
    "LMEFit",
    "fit_lme",
    "JMModelSpec",
    "JMResult",
    "JMBayes2",
    "jm",
    "predict_dynamic",
]
