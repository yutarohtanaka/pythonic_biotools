"""
Configuration presets and settings validation for `purecn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` mode and raise `ValueError` on unknown values."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class PureCNConfig:
    """Algorithmic hyperparameters for `"default"` vs `"R_default"` PureCN modes."""

    settings: SettingsMode
    min_purity: float
    max_purity: float
    purity_step: float
    min_ploidy: float
    max_ploidy: float
    ploidy_step: float
    rho_overdisp: float | None
    min_rho_overdisp: float
    novel_prior_somatic: float
    dbsnp_prior_somatic: float
    common_pop_prior_somatic: float
    cosmic_hotspot_prior_somatic: float
    common_pop_af_cutoff: float
    somatic_posterior_cutoff: float
    max_non_clonal: float | None
    tmb_posterior_cutoff: float
    tmb_max_prior_somatic: float
    cbs_undo_sd: float
    lowess_frac: float

    def default_purity_grid(self) -> np.ndarray:
        n_steps = int(round((self.max_purity - self.min_purity) / self.purity_step)) + 1
        return np.round(np.linspace(self.min_purity, self.max_purity, n_steps), 6)

    def default_ploidy_grid(self) -> np.ndarray:
        n_steps = int(round((self.max_ploidy - self.min_ploidy) / self.ploidy_step)) + 1
        return np.round(np.linspace(self.min_ploidy, self.max_ploidy, n_steps), 6)


def get_purecn_config(settings: SettingsMode = "default") -> PureCNConfig:
    """Return preset `PureCNConfig` for `"default"` or `"R_default"`."""
    mode = validate_settings(settings)
    if mode == "default":
        return PureCNConfig(
            settings="default",
            min_purity=0.10,
            max_purity=0.99,
            purity_step=0.01,
            min_ploidy=1.50,
            max_ploidy=6.00,
            ploidy_step=0.05,
            rho_overdisp=0.005,
            min_rho_overdisp=1e-4,
            novel_prior_somatic=0.5,
            dbsnp_prior_somatic=0.05,
            common_pop_prior_somatic=0.001,
            cosmic_hotspot_prior_somatic=0.98,
            common_pop_af_cutoff=1e-3,
            somatic_posterior_cutoff=0.5,
            max_non_clonal=None,
            tmb_posterior_cutoff=0.5,
            tmb_max_prior_somatic=0.95,
            cbs_undo_sd=0.75,
            lowess_frac=0.30,
        )
    return PureCNConfig(
        settings="R_default",
        min_purity=0.15,
        max_purity=0.95,
        purity_step=0.01,
        min_ploidy=1.40,
        max_ploidy=6.00,
        ploidy_step=0.05,
        rho_overdisp=None,  # Estimated empirically from germline het SNPs (min 1e-4)
        min_rho_overdisp=1e-4,
        novel_prior_somatic=0.5,
        dbsnp_prior_somatic=0.1,
        common_pop_prior_somatic=0.0005,
        cosmic_hotspot_prior_somatic=0.999,
        common_pop_af_cutoff=1e-3,
        somatic_posterior_cutoff=0.5,
        max_non_clonal=0.20,
        tmb_posterior_cutoff=0.80,
        tmb_max_prior_somatic=0.90,
        cbs_undo_sd=1.00,
        lowess_frac=0.25,
    )
