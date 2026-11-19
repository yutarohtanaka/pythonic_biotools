"""
Configuration, validation, chromosome ordering, and HMM state definitions
for `ichorcna`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import numpy as np

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[SettingsMode, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` mode and raise `ValueError` on unknown values."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings {settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return cast(SettingsMode, settings)


def canonical_chrom(chrom: object) -> str:
    """Canonicalize chromosome name (`'chr1' -> '1'`, `'chrX' -> 'X'`)."""
    raw = str(chrom).strip()
    if raw.lower().startswith("chr"):
        raw = raw[3:]
    up = raw.upper()
    if up == "23":
        return "X"
    if up == "24":
        return "Y"
    if up in ("M", "MT"):
        return "MT"
    return up


def chrom_sort_key(chrom_canon: str) -> tuple[int, str]:
    """Natural genomic sort key (`1..22`, `X`, `Y`, `MT`, others)."""
    if chrom_canon.isdigit():
        return (0, f"{int(chrom_canon):04d}")
    if chrom_canon == "X":
        return (1, "X")
    if chrom_canon == "Y":
        return (2, "Y")
    if chrom_canon == "MT":
        return (3, "MT")
    return (4, chrom_canon)


@dataclass(frozen=True)
class HMMState:
    """Single discrete copy-number HMM state in `ichorcna`."""

    name: str
    copy_number: int
    is_subclonal: bool


@dataclass(frozen=True)
class IchorCNAConfig:
    """Configuration for `ichorcna` HMM + Generalized EM inference."""

    normal_init: tuple[float, ...]
    ploidy_init: tuple[float, ...] = (2.0, 3.0)
    max_cn: int = 5
    include_homd: bool = False
    sc_states: tuple[int, ...] = (1, 3)
    txn_e: float = 0.9999999
    txn_strength: float = 10000.0
    max_iter: int = 50
    em_tol: float = 1e-3
    student_df: float = 2.1
    low_tf_mode: bool = False
    settings: SettingsMode = "default"

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if self.max_cn < 3:
            raise ValueError(f"max_cn must be >= 3, got {self.max_cn!r}.")
        if not (0.5 <= self.txn_e < 1.0):
            raise ValueError(f"txn_e must lie in [0.5, 1.0), got {self.txn_e!r}.")
        if self.txn_strength <= 0.0:
            raise ValueError(f"txn_strength must be > 0, got {self.txn_strength!r}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter!r}.")
        for n0 in self.normal_init:
            if not (0.0 < float(n0) < 1.0) or not np.isfinite(n0):
                raise ValueError(f"All normal_init values must lie in (0, 1), got {n0!r}.")
        for p0 in self.ploidy_init:
            if not (0.5 <= float(p0) <= 8.0) or not np.isfinite(p0):
                raise ValueError(f"All ploidy_init values must lie in [0.5, 8.0], got {p0!r}.")


def build_hmm_states(
    max_cn: int = 5,
    include_homd: bool = False,
    sc_states: Sequence[int] = (1, 3),
) -> list[HMMState]:
    """
    Construct ordered clonal and subclonal HMM copy-number states matching
    `ichorCNA` (`HOMD` optional, `HETD`, `NEUT`, `GAIN`, `AMP`, `HLAMP...`,
    plus subclonal `1_sub`, `3_sub`, ...).
    """
    if max_cn < 3:
        raise ValueError(f"max_cn must be >= 3, got {max_cn!r}.")

    states: list[HMMState] = []
    if include_homd:
        states.append(HMMState(name="HOMD", copy_number=0, is_subclonal=False))

    clonal_names = {1: "HETD", 2: "NEUT", 3: "GAIN", 4: "AMP"}
    for c in range(1, max_cn + 1):
        if c in clonal_names:
            label = clonal_names[c]
        elif c == 5:
            label = "HLAMP"
        else:
            label = f"HLAMP{c - 4}"
        states.append(HMMState(name=label, copy_number=c, is_subclonal=False))

    seen_sc: set[int] = set()
    for sc in sc_states:
        c_sc = int(sc)
        if c_sc == 2 or c_sc < 0 or c_sc > max_cn or c_sc in seen_sc:
            continue
        seen_sc.add(c_sc)
        states.append(HMMState(name=f"{c_sc}_sub", copy_number=c_sc, is_subclonal=True))

    return states
