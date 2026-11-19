"""Prioritized composite clinical-trial endpoint specification and settings validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SettingsMode = Literal["default", "R_default"]
EndpointKind = Literal["tte", "binary", "continuous"]
OperatorKind = Literal["higher", "lower"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate mandatory ``settings`` parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings '{settings}'. Must be one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class Endpoint:
    """Single layer of a prioritized hierarchical composite clinical-trial outcome.

    Parameters
    ----------
    col : str
        Column name in ``data`` holding follow-up time (for ``kind="tte"``) or outcome values.
    kind : {"tte", "binary", "continuous"}
        Endpoint data type.
    status_col : str | None
        Event indicator column (``1`` = event, ``0`` = right-censored); required when ``kind="tte"``.
    operator : {"higher", "lower"}
        Direction of clinical benefit: ``"higher"`` when larger values/longer survival are
        favorable (e.g., OS, PFS, QoL), ``"lower"`` when smaller values are favorable (e.g., toxicity).
    threshold : float
        Minimum clinically relevant difference ``tau >= 0`` required to declare a win or loss.
    """

    col: str
    kind: EndpointKind = "tte"
    status_col: str | None = None
    operator: OperatorKind = "higher"
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.col:
            raise ValueError("Endpoint 'col' must be a non-empty column name.")
        if self.kind not in ("tte", "binary", "continuous"):
            raise ValueError(
                f"Invalid Endpoint kind '{self.kind}'. Must be 'tte', 'binary', or 'continuous'."
            )
        if self.operator not in ("higher", "lower"):
            raise ValueError(
                f"Invalid Endpoint operator '{self.operator}'. Must be 'higher' or 'lower'."
            )
        if self.threshold < 0.0:
            raise ValueError(
                f"Endpoint threshold must be non-negative, got {self.threshold}."
            )
        if self.kind == "tte" and not self.status_col:
            raise ValueError(
                f"Time-to-event Endpoint '{self.col}' requires 'status_col'."
            )
