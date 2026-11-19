"""
Input validation, chromosome canonicalization, and variant-to-segment mapping
for `mutationtimer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

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
class MutationTimerConfig:
    """Configuration parameters for `mutationtimer` inference."""

    purity: float
    gender: Literal["female", "male"] = "female"
    n_boot: int = 200
    seed: int = 42
    rho_bb: float = 0.01
    max_em_iter: int = 100
    em_tol: float = 1e-7
    settings: SettingsMode = "default"

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if not (0.0 < float(self.purity) <= 1.0) or not np.isfinite(self.purity):
            raise ValueError(f"purity must be in (0, 1], got {self.purity!r}.")
        if self.gender not in ("female", "male"):
            raise ValueError(
                f"gender must be 'female' or 'male', got {self.gender!r}."
            )
        if self.n_boot < 0:
            raise ValueError(f"n_boot must be >= 0, got {self.n_boot!r}.")
        if not (0.0 < self.rho_bb < 1.0):
            raise ValueError(f"rho_bb must be in (0, 1), got {self.rho_bb!r}.")


def canonical_chrom(chrom: object) -> str:
    """Canonicalize chromosome labels (`'chr1' -> '1'`, `'chrX' -> 'X'`, `'23' -> 'X'`)."""
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


def normalize_gender(gender: str) -> Literal["female", "male"]:
    """Normalize sex/gender string into `'female'` or `'male'`."""
    if not isinstance(gender, str):
        raise ValueError(f"gender must be a string, got {gender!r}.")
    g = gender.strip().lower()
    if g in ("female", "f", "xx"):
        return "female"
    if g in ("male", "m", "xy"):
        return "male"
    raise ValueError(
        f"Unrecognized gender {gender!r}. Expected 'female' or 'male'."
    )


def normal_copy_number(chrom_canon: str, gender: Literal["female", "male"]) -> int:
    """Return germline diploid/haploid copy number C_norm (1 on male X/Y, else 2)."""
    if gender == "male" and chrom_canon in ("X", "Y"):
        return 1
    return 2


def validate_and_prepare_segments(cn_segments_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize allele-specific copy-number segments."""
    if not isinstance(cn_segments_df, pd.DataFrame) or cn_segments_df.empty:
        raise ValueError("cn_segments_df must be a non-empty pandas DataFrame.")

    df = cn_segments_df.copy().reset_index(drop=True)
    col_map: dict[str, str] = {}
    if "chrom" not in df.columns:
        if "chr" in df.columns:
            col_map["chr"] = "chrom"
        elif "chromosome" in df.columns:
            col_map["chromosome"] = "chrom"
    if "major_cn" not in df.columns and "nMajor" in df.columns:
        col_map["nMajor"] = "major_cn"
    if "minor_cn" not in df.columns and "nMinor" in df.columns:
        col_map["nMinor"] = "minor_cn"
    if col_map:
        df = df.rename(columns=col_map)

    required = ("chrom", "start", "end", "major_cn", "minor_cn")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"cn_segments_df is missing required columns: {missing}."
        )

    df["chrom_canon"] = [canonical_chrom(c) for c in df["chrom"]]
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    if bool(np.any(df["end"] < df["start"])):
        raise ValueError("cn_segments_df has segments with end < start.")

    df["major_cn"] = np.round(df["major_cn"].astype(np.float64)).astype(np.int64)
    df["minor_cn"] = np.round(df["minor_cn"].astype(np.float64)).astype(np.int64)
    if bool(np.any(df["major_cn"] < 0)) or bool(np.any(df["minor_cn"] < 0)):
        raise ValueError("major_cn and minor_cn must be non-negative.")

    # Ensure major_cn >= minor_cn
    swap_mask = df["major_cn"] < df["minor_cn"]
    if bool(np.any(swap_mask)):
        maj = np.maximum(df["major_cn"].to_numpy(), df["minor_cn"].to_numpy())
        mn = np.minimum(df["major_cn"].to_numpy(), df["minor_cn"].to_numpy())
        df["major_cn"] = maj
        df["minor_cn"] = mn

    if "clonal_frequency" in df.columns:
        cf = df["clonal_frequency"].astype(np.float64).to_numpy()
        if np.any(~np.isfinite(cf)) or np.any(cf <= 0.0) or np.any(cf > 1.0):
            raise ValueError("clonal_frequency in cn_segments_df must lie in (0, 1].")
        df["clonal_frequency"] = cf
    else:
        df["clonal_frequency"] = 1.0

    if "segment_id" not in df.columns:
        df["segment_id"] = np.arange(len(df), dtype=np.int64)

    return df


def validate_and_prepare_variants(variants_df: pd.DataFrame) -> pd.DataFrame:
    """Validate somatic variant read-count DataFrame."""
    if not isinstance(variants_df, pd.DataFrame):
        raise ValueError("variants_df must be a pandas DataFrame.")
    if variants_df.empty:
        out = variants_df.copy()
        for col in ("chrom", "pos", "alt_count", "total_depth", "chrom_canon"):
            if col not in out.columns:
                out[col] = pd.Series(dtype="object" if "chrom" in col else "int64")
        return out

    df = variants_df.copy().reset_index(drop=True)
    if "chrom" not in df.columns:
        if "chr" in df.columns:
            df = df.rename(columns={"chr": "chrom"})
        elif "chromosome" in df.columns:
            df = df.rename(columns={"chromosome": "chrom"})
    if "pos" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "pos"})

    if "alt_count" not in df.columns:
        for cand in ("t_alt_count", "alt_reads", "var_counts", "alt"):
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                df["alt_count"] = df[cand]
                break

    if "total_depth" not in df.columns:
        for cand in ("depth", "dp", "t_depth"):
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                df["total_depth"] = df[cand]
                break
        if "total_depth" not in df.columns:
            for ref_cand in ("ref_count", "t_ref_count", "ref_reads", "ref"):
                if (
                    ref_cand in df.columns
                    and pd.api.types.is_numeric_dtype(df[ref_cand])
                    and "alt_count" in df.columns
                ):
                    df["total_depth"] = df["alt_count"] + df[ref_cand]
                    break

    required = ("chrom", "pos", "alt_count", "total_depth")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"variants_df is missing required columns: {missing}. "
            "Provide ('chrom', 'pos', 'alt_count') and either 'total_depth' or 'ref_count'."
        )

    df["chrom_canon"] = [canonical_chrom(c) for c in df["chrom"]]
    df["pos"] = df["pos"].astype(np.int64)
    df["alt_count"] = np.round(df["alt_count"].astype(np.float64)).astype(np.int64)
    df["total_depth"] = np.round(df["total_depth"].astype(np.float64)).astype(np.int64)

    if bool(np.any(df["alt_count"] < 0)) or bool(np.any(df["total_depth"] < 0)):
        raise ValueError("alt_count and total_depth must be non-negative.")
    if bool(np.any(df["alt_count"] > df["total_depth"])):
        raise ValueError("alt_count cannot exceed total_depth.")

    return df


def map_variants_to_segments(
    variants_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> np.ndarray:
    """
    Map each variant in `variants_df` to the row index `[0, len(segments_df)-1]`
    of its overlapping segment in `segments_df` (1-based closed `[start, end]`),
    or `-1` if unmapped.
    """
    n_var = len(variants_df)
    seg_indices = np.full(n_var, -1, dtype=np.int64)
    if n_var == 0 or segments_df.empty:
        return seg_indices

    var_chroms = variants_df["chrom_canon"].to_numpy()
    var_pos = variants_df["pos"].to_numpy(dtype=np.int64)

    seg_chroms = segments_df["chrom_canon"].to_numpy()
    seg_starts = segments_df["start"].to_numpy(dtype=np.int64)
    seg_ends = segments_df["end"].to_numpy(dtype=np.int64)

    for chrom in np.unique(var_chroms):
        v_idx = np.flatnonzero(var_chroms == chrom)
        s_idx = np.flatnonzero(seg_chroms == chrom)
        if s_idx.size == 0:
            continue
        order = np.argsort(seg_starts[s_idx], kind="mergesort")
        s_sorted = s_idx[order]
        starts = seg_starts[s_sorted]
        ends = seg_ends[s_sorted]

        vp = var_pos[v_idx]
        # Candidate segment is last segment with start <= pos
        cand = np.searchsorted(starts, vp, side="right") - 1
        valid = (cand >= 0) & (vp <= ends[np.maximum(cand, 0)])
        seg_indices[v_idx[valid]] = s_sorted[cand[valid]]

        # Fallback linear check for any overlapping/nested segments on remaining variants
        rem = v_idx[~valid]
        for vi in rem:
            p = var_pos[vi]
            hits = s_sorted[(starts <= p) & (p <= ends)]
            if hits.size > 0:
                seg_indices[vi] = hits[0]

    return seg_indices


def extract_subclonal_phis(
    clusters_df: pd.DataFrame | None,
    purity: float,
    settings: SettingsMode = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract subclonal cluster CCFs `phi_k in (0, 1)` and relative cluster weights
    from `clusters_df`. When `clusters_df is None`, provides a single default
    subclonal state at `phi = 0.5` with small prior weight.
    """
    validate_settings(settings)
    if clusters_df is None:
        return np.array([0.5], dtype=np.float64), np.array([0.10], dtype=np.float64)

    if not isinstance(clusters_df, pd.DataFrame) or clusters_df.empty:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    df = clusters_df.copy()
    if "ccf" in df.columns:
        phis = df["ccf"].astype(np.float64).to_numpy()
    elif "phi" in df.columns:
        phis = df["phi"].astype(np.float64).to_numpy()
    elif "clonal_frequency" in df.columns:
        phis = df["clonal_frequency"].astype(np.float64).to_numpy()
    elif "cellular_prevalence" in df.columns:
        phis = df["cellular_prevalence"].astype(np.float64).to_numpy() / purity
    elif "proportion" in df.columns:
        prop = df["proportion"].astype(np.float64).to_numpy()
        # In R MutationTimeR, `clusters$proportion` is cellularity (with clonal row ~ purity)
        has_clonal_purity_row = bool(np.any(np.abs(prop - purity) <= 0.025))
        if np.max(prop) > purity + 1e-6:
            phis = prop
        elif has_clonal_purity_row or settings == "R_default":
            phis = prop / purity
        else:
            phis = prop
    else:
        raise ValueError(
            "clusters_df must contain 'ccf', 'phi', 'cellular_prevalence', or 'proportion'."
        )

    if "n_ssms" in df.columns:
        weights = np.maximum(df["n_ssms"].astype(np.float64).to_numpy(), 1.0)
    elif "n_mutations" in df.columns:
        weights = np.maximum(df["n_mutations"].astype(np.float64).to_numpy(), 1.0)
    elif "weight" in df.columns:
        weights = np.maximum(df["weight"].astype(np.float64).to_numpy(), 1e-3)
    else:
        weights = np.ones_like(phis, dtype=np.float64)

    sub_mask = (phis > 1e-4) & (phis < 0.98)
    sub_phis = np.clip(phis[sub_mask], 0.01, 0.97)
    sub_weights = weights[sub_mask]
    if sub_weights.size > 0:
        sub_weights = sub_weights / np.sum(sub_weights) * 0.25
    return sub_phis, sub_weights
