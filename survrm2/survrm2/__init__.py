"""survrm2: Restricted Mean Survival Time (RMST) and RMTL Clinical-Trial Inference."""

from .ancova import (
    LinkType,
    RMST2AdjustedModelResult,
    SettingsMode,
    VALID_LINKS,
    VALID_SETTINGS,
    rmst2reg,
    validate_settings,
)
from .core import (
    ContrastResult,
    RMST1Result,
    RMST2Result,
    SurvRM2,
    rmst1,
    rmst2,
    rmst2_curve,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SettingsMode",
    "LinkType",
    "VALID_SETTINGS",
    "VALID_LINKS",
    "validate_settings",
    "ContrastResult",
    "RMST1Result",
    "RMST2AdjustedModelResult",
    "RMST2Result",
    "SurvRM2",
    "rmst1",
    "rmst2",
    "rmst2reg",
    "rmst2_curve",
]
