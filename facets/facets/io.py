"""
Input validation, column alias normalization, and configuration container for `facets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .reference import canonical_chrom, chrom_sort_key, get_genome_reference

SettingsType = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsType:
    """Validate `settings` mode strictly ('default' or 'R_default')."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"`settings` must be 'default' or 'R_default', got {settings!r}."
        )
    return settings  # type: ignore[return-value]

_COLUMN_ALIASES: dict[str, str] = {
    "chrom": "chrom",
    "chromosome": "chrom",
    "chr": "chrom",
    "pos": "pos",
    "position": "pos",
    "start": "pos",
    "nor_dp": "nor_dp",
    "nor.dp": "nor_dp",
    "normal_dp": "nor_dp",
    "file1dp": "nor_dp",
    "tum_dp": "tum_dp",
    "tum.dp": "tum_dp",
    "tumor_dp": "tum_dp",
    "file2dp": "tum_dp",
    "nor_alt": "nor_alt",
    "nor.rd": "nor_alt",
    "normal_alt": "nor_alt",
    "file1a": "nor_alt",
    "tum_alt": "tum_alt",
    "tum.rd": "tum_alt",
    "tumor_alt": "tum_alt",
    "file2a": "tum_alt",
    "gc": "gc",
    "gc_content": "gc",
    "het": "het",
    "is_het": "het",
}

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "nor_dp",
    "tum_dp",
    "nor_alt",
    "tum_alt",
)


@dataclass(frozen=True)
class FacetsConfig:
    """Configuration parameters for the FACETS preprocessing, CBS, and EM pipeline."""

    genome: str = "GRCh38"
    ndepth: int = 35
    ndepthmax: int | None = 1000
    het_thresh: float | None = None
    snp_nbhd: int = 250
    pre_cval: float = 25.0
    cval: float = 150.0
    min_nhet: int = 15
    min_seg_snps: int = 5
    diploid_logr: float | None = None
    delta_cn: float = 0.5
    max_cn: int = 10
    unmatched: bool = False
    lowess_frac: float | None = None
    lowess_iter: int = 3
    keep_low_nhet_lcn: bool = False
    seed: int | None = None
    settings: SettingsType = "default"

    @property
    def effective_het_thresh(self) -> float:
        """Return configured `het_thresh`, or mode default (`0.10` unmatched, `0.25` matched)."""
        if self.het_thresh is not None:
            return float(self.het_thresh)
        return 0.10 if self.unmatched else 0.25

    @property
    def effective_lowess_frac(self) -> float:
        """Return configured `lowess_frac`, or mode default (`0.20` R_default, `0.25` default)."""
        if self.lowess_frac is not None:
            return float(self.lowess_frac)
        return 0.20 if self.settings == "R_default" else 0.25

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        get_genome_reference(self.genome, include_chr_x=True, include_chr_y=True)
        if self.ndepth < 1:
            raise ValueError(f"`ndepth` must be >= 1, got {self.ndepth}.")
        if self.ndepthmax is not None and self.ndepthmax < self.ndepth:
            raise ValueError(
                f"`ndepthmax` ({self.ndepthmax}) must be >= `ndepth` ({self.ndepth})."
            )
        if self.het_thresh is not None and not (0.0 < self.het_thresh <= 0.5):
            raise ValueError(
                f"`het_thresh` must be in (0.0, 0.5], got {self.het_thresh}."
            )
        if self.snp_nbhd < 0:
            raise ValueError(f"`snp_nbhd` must be >= 0, got {self.snp_nbhd}.")
        if self.pre_cval <= 0.0:
            raise ValueError(f"`pre_cval` must be > 0, got {self.pre_cval}.")
        if self.cval <= 0.0:
            raise ValueError(f"`cval` must be > 0, got {self.cval}.")
        if self.min_nhet < 1:
            raise ValueError(f"`min_nhet` must be >= 1, got {self.min_nhet}.")
        if self.min_seg_snps < 2:
            raise ValueError(
                f"`min_seg_snps` must be >= 2, got {self.min_seg_snps}."
            )
        if self.diploid_logr is not None and not np.isfinite(self.diploid_logr):
            raise ValueError(
                f"`diploid_logr` must be finite when provided, got {self.diploid_logr}."
            )
        if self.delta_cn <= 0.0:
            raise ValueError(f"`delta_cn` must be > 0, got {self.delta_cn}.")
        if self.max_cn < 2:
            raise ValueError(f"`max_cn` must be >= 2, got {self.max_cn}.")
        if self.lowess_frac is not None and not (0.0 < self.lowess_frac <= 1.0):
            raise ValueError(
                f"`lowess_frac` must be in (0.0, 1.0], got {self.lowess_frac}."
            )
        if self.lowess_iter < 1:
            raise ValueError(
                f"`lowess_iter` must be >= 1, got {self.lowess_iter}."
            )


def validate_and_normalize_pileup(
    pileup: pd.DataFrame,
    genome: str = "GRCh38",
) -> pd.DataFrame:
    """
    Validate and normalize a tumor-normal SNP pileup DataFrame.

    Maps canonical column aliases, enforces count bounds (0 <= alt <= dp),
    normalizes chromosome names ('1' -> 'chr1', '23' -> 'chrX'), maps SNPs to
    p/q chromosome arms, deduplicates (chrom, pos) retaining highest total depth,
    and sorts deterministically using mergesort.
    """
    if not isinstance(pileup, pd.DataFrame):
        raise ValueError("`pileup` must be a pandas DataFrame.")
    if len(pileup) == 0:
        raise ValueError("`pileup` DataFrame cannot be empty.")

    ref_map = get_genome_reference(genome, include_chr_x=True, include_chr_y=True)

    rename_map: dict[str, str] = {}
    seen_targets: set[str] = set()
    for col in pileup.columns:
        key = str(col).strip().lower()
        target = _COLUMN_ALIASES.get(key)
        if target is not None:
            if target in seen_targets:
                raise ValueError(
                    f"Duplicate alias columns mapping to '{target}' in `pileup`."
                )
            seen_targets.add(target)
            rename_map[col] = target

    missing = [c for c in _REQUIRED_COLUMNS if c not in seen_targets]
    if missing:
        raise ValueError(
            f"Missing required pileup column(s): {missing}. "
            f"Expected at least {_REQUIRED_COLUMNS}."
        )

    df = pileup[list(rename_map.keys())].rename(columns=rename_map).copy()

    if df["chrom"].isna().any():
        raise ValueError("`chrom` column contains null/NaN values.")
    chrom_norm = [canonical_chrom(c) for c in df["chrom"].tolist()]
    df["chrom"] = chrom_norm

    for num_col in ("pos", "nor_dp", "tum_dp", "nor_alt", "tum_alt"):
        vals = pd.to_numeric(df[num_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column '{num_col}' contains non-numeric or non-finite values.")
        if np.any(np.floor(vals) != vals):
            raise ValueError(f"Column '{num_col}' must contain integer values.")
        df[num_col] = vals.astype(np.int64)

    if (df["pos"] < 1).any():
        raise ValueError("`pos` must contain 1-based positive integers (>= 1).")
    if (df["nor_dp"] < 0).any() or (df["tum_dp"] < 0).any():
        raise ValueError("`nor_dp` and `tum_dp` must be >= 0.")
    if (df["nor_alt"] < 0).any() or (df["tum_alt"] < 0).any():
        raise ValueError("`nor_alt` and `tum_alt` must be >= 0.")
    if (df["nor_alt"] > df["nor_dp"]).any():
        raise ValueError("Found loci with `nor_alt > nor_dp`.")
    if (df["tum_alt"] > df["tum_dp"]).any():
        raise ValueError("Found loci with `tum_alt > tum_dp`.")

    if "gc" in df.columns:
        gc_vals = pd.to_numeric(df["gc"], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(np.isinf(gc_vals)):
            raise ValueError("`gc` column contains infinite values.")
        finite_gc = np.isfinite(gc_vals)
        if np.any(finite_gc):
            finite_arr = gc_vals[finite_gc]
            if np.any(finite_arr < 0.0) or np.any(finite_arr > 100.0):
                raise ValueError("`gc` values must lie within [0, 1] or [0, 100].")
            if np.any(finite_arr > 1.0):
                gc_vals = np.where(finite_gc, gc_vals / 100.0, np.nan)
        df["gc"] = gc_vals
    else:
        df["gc"] = np.nan

    if "het" in df.columns:
        het_series = df["het"]
        if het_series.dtype == bool:
            df["het"] = het_series.to_numpy(dtype=bool)
        else:
            df["het"] = het_series.fillna(False).astype(bool)

    df["nor_ref"] = df["nor_dp"] - df["nor_alt"]
    df["tum_ref"] = df["tum_dp"] - df["tum_alt"]

    arms: list[str] = []
    for ch, p in zip(df["chrom"].tolist(), df["pos"].tolist()):
        ref = ref_map.get(ch)
        arms.append(ref.map_pos_to_arm(int(p)) if ref is not None else ch)
    df["arm"] = arms

    # Sort deterministically, then deduplicate exact (chrom, pos) retaining highest total depth
    sort_keys = [chrom_sort_key(c) for c in df["chrom"].tolist()]
    df["_chrom_group"] = [k[0] for k in sort_keys]
    df["_chrom_rank"] = [k[1] for k in sort_keys]
    df["_chrom_str"] = [k[2] for k in sort_keys]
    df["_total_dp"] = df["nor_dp"] + df["tum_dp"]
    df["_orig_idx"] = np.arange(len(df), dtype=np.int64)

    if df.duplicated(subset=["chrom", "pos"]).any():
        df = (
            df.sort_values(
                by=["_chrom_group", "_chrom_rank", "_chrom_str", "pos", "_total_dp", "_orig_idx"],
                ascending=[True, True, True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["chrom", "pos"], keep="first")
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values(
            by=["_chrom_group", "_chrom_rank", "_chrom_str", "pos", "_orig_idx"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    df = df.drop(columns=["_chrom_group", "_chrom_rank", "_chrom_str", "_total_dp", "_orig_idx"])

    ordered_cols = [
        "chrom",
        "arm",
        "pos",
        "nor_dp",
        "tum_dp",
        "nor_alt",
        "tum_alt",
        "nor_ref",
        "tum_ref",
        "gc",
    ]
    if "het" in df.columns:
        ordered_cols.append("het")
    return df[ordered_cols]
