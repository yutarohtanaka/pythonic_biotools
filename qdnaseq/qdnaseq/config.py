"""
Configuration presets and settings validation for `qdnaseq` and bundled `ACE`.
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
class QDNAseqConfig:
    """Algorithmic defaults distinguishing `"default"` vs `"R_default"` modes."""

    settings: SettingsMode
    min_mapq: int
    mappability_cutoff: float
    residual_sd_cutoff: float
    loess_max_iter: int
    loess_cutoff: float
    loess_span: float
    seg_transform: Literal["anscombe_log2", "sqrt"]
    seg_alpha: float
    seg_undo_sd: float
    ace_method: Literal["RMSE", "MAE", "SMRE"]
    ace_penploidy: float
    ace_pencellularity: float
    ace_max_ploidy: float

    def default_cellularity_grid(self) -> np.ndarray:
        return np.round(np.linspace(0.05, 1.00, 96), 4)

    def default_ploidy_grid(self) -> np.ndarray:
        n_steps = int(round((self.ace_max_ploidy - 1.00) / 0.05)) + 1
        return np.round(np.linspace(1.00, self.ace_max_ploidy, n_steps), 4)


def get_qdnaseq_config(settings: SettingsMode = "default") -> QDNAseqConfig:
    """Return `QDNAseqConfig` for `"default"` or `"R_default"`."""
    mode = validate_settings(settings)
    if mode == "default":
        return QDNAseqConfig(
            settings="default",
            min_mapq=30,
            mappability_cutoff=50.0,
            residual_sd_cutoff=4.0,
            loess_max_iter=3,
            loess_cutoff=4.0,
            loess_span=0.65,
            seg_transform="anscombe_log2",
            seg_alpha=1e-4,
            seg_undo_sd=0.75,
            ace_method="RMSE",
            ace_penploidy=0.5,
            ace_pencellularity=0.5,
            ace_max_ploidy=6.0,
        )
    return QDNAseqConfig(
        settings="R_default",
        min_mapq=37,
        mappability_cutoff=50.0,
        residual_sd_cutoff=4.0,
        loess_max_iter=3,
        loess_cutoff=4.0,
        loess_span=0.65,
        seg_transform="sqrt",
        seg_alpha=1e-4,
        seg_undo_sd=1.0,
        ace_method="RMSE",
        ace_penploidy=0.0,
        ace_pencellularity=0.0,
        ace_max_ploidy=5.0,
    )
