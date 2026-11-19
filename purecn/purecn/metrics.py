"""
Artifact-cleaned tumor mutational burden (`call_tmb`), chromosomal instability
(`call_cin`), and genome-wide LOH fraction (`compute_loh_fraction`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SettingsMode, get_purecn_config, validate_settings
from .coverage import CoverageNormalizationResult
from .likelihood import expected_log2_ratio, standardize_segments_df


class TMBValue(float):
    """Float-compatible TMB result (mutations/Mb) carrying filtered mutation metadata."""

    tmb: float
    n_mutations: int
    callable_mb: float
    variants: pd.DataFrame

    def __new__(
        cls,
        value: float,
        *,
        n_mutations: int = 0,
        callable_mb: float = 1.0,
        variants: pd.DataFrame | None = None,
    ) -> TMBValue:
        obj = super().__new__(cls, float(value))
        obj.tmb = float(value)
        obj.n_mutations = int(n_mutations)
        obj.callable_mb = float(callable_mb)
        obj.variants = variants if variants is not None else pd.DataFrame()
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "tmb": self.tmb,
            "n_mutations": self.n_mutations,
            "callable_mb": self.callable_mb,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class CINValue(float):
    """Float-compatible Chromosomal Instability (CIN) fraction carrying summary metrics."""

    cin: float
    cin_ploidy_adjusted: float
    cin_unadjusted: float
    altered_bp: int
    total_bp: int

    def __new__(
        cls,
        value: float,
        *,
        cin_ploidy_adjusted: float = 0.0,
        cin_unadjusted: float = 0.0,
        altered_bp: int = 0,
        total_bp: int = 1,
    ) -> CINValue:
        obj = super().__new__(cls, float(value))
        obj.cin = float(value)
        obj.cin_ploidy_adjusted = float(cin_ploidy_adjusted)
        obj.cin_unadjusted = float(cin_unadjusted)
        obj.altered_bp = int(altered_bp)
        obj.total_bp = int(total_bp)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "cin": self.cin,
            "cin_ploidy_adjusted": self.cin_ploidy_adjusted,
            "cin_unadjusted": self.cin_unadjusted,
            "altered_bp": self.altered_bp,
            "total_bp": self.total_bp,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def call_tmb(
    variants_res_df: pd.DataFrame,
    callable_mb: float,
    min_af: float = 0.03,
    max_pop_af: float = 0.001,
    exclude_hotspot: bool = True,
    settings: SettingsMode = "default",
) -> TMBValue:
    """
    Compute artifact- and germline-cleaned somatic Tumor Mutational Burden (mutations/Mb)
    following `PureCN::callTMB`.
    """
    cfg = get_purecn_config(validate_settings(settings))
    mb = float(callable_mb)
    if mb <= 0.0 or not np.isfinite(mb):
        raise ValueError(f"`callable_mb` must be > 0, got {callable_mb}.")
    if not isinstance(variants_res_df, pd.DataFrame):
        raise ValueError("`variants_res_df` must be a pandas DataFrame.")
    if variants_res_df.empty:
        return TMBValue(0.0, n_mutations=0, callable_mb=mb, variants=variants_res_df.copy())

    df = variants_res_df.copy()
    if "posterior_somatic" not in df.columns:
        raise ValueError(
            "`variants_res_df` must contain `posterior_somatic` (output of `predict_somatic`)."
        )

    mask = df["posterior_somatic"].astype(np.float64) >= cfg.tmb_posterior_cutoff

    if "af" in df.columns:
        mask &= df["af"].astype(np.float64) >= float(min_af)
    elif "alt_count" in df.columns and "depth" in df.columns:
        af = df["alt_count"].astype(np.float64) / np.maximum(
            df["depth"].astype(np.float64), 1.0
        )
        mask &= af >= float(min_af)

    if "pop_af" in df.columns:
        mask &= df["pop_af"].fillna(0.0).astype(np.float64) <= float(max_pop_af)

    if "alt_count" in df.columns:
        mask &= df["alt_count"].astype(np.float64) >= 3.0

    if "is_coding" in df.columns:
        mask &= df["is_coding"].fillna(True).astype(bool)

    if exclude_hotspot:
        if "prior_somatic" in df.columns:
            mask &= df["prior_somatic"].astype(np.float64) <= cfg.tmb_max_prior_somatic
        if "cosmic_cnt" in df.columns:
            mask &= df["cosmic_cnt"].fillna(0).astype(np.int64) < 2
        if "is_hotspot" in df.columns:
            mask &= ~df["is_hotspot"].fillna(False).astype(bool)

    kept = df.loc[mask].reset_index(drop=True)
    n_mut = len(kept)
    tmb_val = float(n_mut) / mb
    return TMBValue(tmb_val, n_mutations=n_mut, callable_mb=mb, variants=kept)


def _ensure_segment_cn(
    segs: pd.DataFrame,
    purity: float,
    ploidy: float,
    max_cn: int = 8,
) -> pd.DataFrame:
    """Populate integer `C` on `segs` if not already present."""
    out = segs.copy()
    if "C" not in out.columns:
        c_grid = np.arange(max_cn + 1, dtype=np.float64)
        exp_l2 = expected_log2_ratio(c_grid[np.newaxis, :], purity, ploidy)  # (1, C+1)
        y = out["seg_mean"].to_numpy(dtype=np.float64)[:, np.newaxis]
        out["C"] = np.argmin(np.abs(y - exp_l2), axis=1).astype(np.int64)
    return out


def call_cin(
    segments_df: pd.DataFrame | CoverageNormalizationResult,
    purity: float,
    ploidy: float,
    settings: SettingsMode = "default",
) -> CINValue:
    """
    Compute genome fraction affected by Chromosomal Instability (`PureCN::callCIN`),
    accounting for ploidy-adjusted copy-number alterations and allele-specific LOH.
    """
    validate_settings(settings)
    rho = float(purity)
    psi = float(ploidy)
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"`purity` must be in (0, 1], got {purity}.")
    if psi <= 0.0:
        raise ValueError(f"`ploidy` must be > 0, got {ploidy}.")

    segs = _ensure_segment_cn(standardize_segments_df(segments_df), rho, psi)
    spans = segs["size_bp"].to_numpy(dtype=np.int64)
    total_bp = max(int(np.sum(spans)), 1)

    c_vals = segs["C"].to_numpy(dtype=np.int64)
    ref_cn = max(1, int(round(psi)))

    if "loh" in segs.columns:
        loh_mask = segs["loh"].fillna(False).astype(bool).to_numpy()
    elif "M" in segs.columns:
        loh_mask = segs["M"].to_numpy(dtype=np.int64) == 0
    else:
        loh_mask = c_vals <= 1

    alt_ploidy = (c_vals != ref_cn) | loh_mask
    alt_diploid = (c_vals != 2) | loh_mask

    altered_bp = int(np.sum(spans[alt_ploidy]))
    cin_adj = float(altered_bp) / float(total_bp)
    cin_unadj = float(np.sum(spans[alt_diploid])) / float(total_bp)

    return CINValue(
        cin_adj,
        cin_ploidy_adjusted=cin_adj,
        cin_unadjusted=cin_unadj,
        altered_bp=altered_bp,
        total_bp=total_bp,
    )


def compute_loh_fraction(segments_df: pd.DataFrame) -> float:
    """Compute base-pair weighted fraction of segmented genome in LOH (`M == 0`)."""
    if segments_df.empty:
        return 0.0
    spans = np.maximum(
        1,
        segments_df["end"].to_numpy(dtype=np.int64)
        - segments_df["start"].to_numpy(dtype=np.int64)
        + 1,
    )
    total_bp = max(int(np.sum(spans)), 1)
    if "loh" in segments_df.columns:
        loh_mask = segments_df["loh"].fillna(False).astype(bool).to_numpy()
    elif "M" in segments_df.columns:
        loh_mask = segments_df["M"].to_numpy(dtype=np.int64) == 0
    elif "C" in segments_df.columns:
        loh_mask = segments_df["C"].to_numpy(dtype=np.int64) <= 1
    else:
        return 0.0
    return float(np.sum(spans[loh_mask])) / float(total_bp)
