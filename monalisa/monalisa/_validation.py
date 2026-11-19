"""Validation helpers for monaLisa."""

from __future__ import annotations

from typing import Literal, TypeAlias

SettingsMode: TypeAlias = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, str] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate the ``settings`` parameter and return it typed."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]
