"""
Configuration, settings validation, and multi-region input normalization
for `conipher`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, cast

import numpy as np
import pandas as pd

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[SettingsMode, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` mode and raise `ValueError` on unknown values."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings {settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return cast(SettingsMode, settings)


@dataclass(frozen=True)
class ConipherConfig:
    """Configuration parameters for `conipher` clustering and tree building."""

    max_clusters: int = 10
    min_cluster_size: int = 5
    ccf_buffer: float = 0.10
    seed: int = 42
    settings: SettingsMode = "default"

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if self.max_clusters < 1:
            raise ValueError(f"max_clusters must be >= 1, got {self.max_clusters!r}.")
        if self.min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1, got {self.min_cluster_size!r}.")
        if self.ccf_buffer < 0.0 or not np.isfinite(self.ccf_buffer):
            raise ValueError(f"ccf_buffer must be >= 0, got {self.ccf_buffer!r}.")


def normalize_purity_map(
    purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
) -> dict[str, float]:
    """Validate and standardize `purity_by_region` across `R >= 2` tumor regions."""
    if isinstance(purity_by_region, pd.DataFrame):
        if "region" not in purity_by_region.columns or "purity" not in purity_by_region.columns:
            raise ValueError("purity_by_region DataFrame must contain 'region' and 'purity'.")
        raw_map = dict(
            zip(
                purity_by_region["region"].astype(str),
                purity_by_region["purity"].astype(np.float64),
            )
        )
    elif isinstance(purity_by_region, pd.Series):
        raw_map = {str(k): float(v) for k, v in purity_by_region.items()}
    elif isinstance(purity_by_region, Mapping):
        raw_map = {str(k): float(v) for k, v in purity_by_region.items()}
    else:
        raise ValueError("purity_by_region must be a Mapping, Series, or DataFrame.")

    if len(raw_map) < 2:
        raise ValueError(
            f"CONIPHER multi-region analysis requires at least 2 regions, got {len(raw_map)}."
        )

    for reg, pur in raw_map.items():
        if not (0.0 < pur <= 1.0) or not np.isfinite(pur):
            raise ValueError(f"Region {reg!r} purity must lie in (0, 1], got {pur!r}.")

    return raw_map
