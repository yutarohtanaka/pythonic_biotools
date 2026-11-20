"""buysetest: Generalized Pairwise Comparisons (BuyseTest) & Proportional Win-Fractions Regression (pwreg)."""

from .core import BuyseTest, BuyseTestResult, StatisticLayerInference, buyse_test
from .endpoints import Endpoint, SettingsMode, validate_settings
from .plotting import plot_buyse_test
from .pwreg import PWRegResult, pwreg
from .scoring import (
    KMAtomDistribution,
    fit_km_atoms,
    score_pairs_continuous_binary,
    score_pairs_tte_gehan,
    score_pairs_tte_peron,
)

__version__ = "0.1.0"

__all__ = [
    "BuyseTest",
    "BuyseTestResult",
    "Endpoint",
    "KMAtomDistribution",
    "PWRegResult",
    "SettingsMode",
    "StatisticLayerInference",
    "buyse_test",
    "fit_km_atoms",
    "plot_buyse_test",
    "pwreg",
    "score_pairs_continuous_binary",
    "score_pairs_tte_gehan",
    "score_pairs_tte_peron",
    "validate_settings",
]
