"""
Configuration presets and settings validation for `chord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
class CHORDConfig:
    """Hyperparameters distinguishing `"default"` vs `"R_default"` CHORD modes."""

    settings: SettingsMode
    laplace_alpha: float
    include_log_burden: bool
    strict_low_indel_mask: bool
    hrd_cutoff: float
    min_indel_load: int
    msi_cutoff: int


def get_chord_config(settings: SettingsMode = "default") -> CHORDConfig:
    """Return `CHORDConfig` for `"default"` or `"R_default"`."""
    mode = validate_settings(settings)
    if mode == "default":
        return CHORDConfig(
            settings="default",
            laplace_alpha=0.25,
            include_log_burden=True,
            strict_low_indel_mask=False,
            hrd_cutoff=0.5,
            min_indel_load=50,
            msi_cutoff=14_000,
        )
    return CHORDConfig(
        settings="R_default",
        laplace_alpha=0.0,
        include_log_burden=False,
        strict_low_indel_mask=True,
        hrd_cutoff=0.5,
        min_indel_load=50,
        msi_cutoff=14_000,
    )
