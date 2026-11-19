"""
End-to-end PureCN pipeline (`run_purecn`, `PureCN`, `PureCNResult`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import SettingsMode, validate_settings
from .coverage import CoverageNormalizationResult, normalize_tumor_only_coverage
from .likelihood import PureCNSolution, grid_search_purity_ploidy
from .metrics import CINValue, TMBValue, call_cin, call_tmb, compute_loh_fraction
from .somatic import predict_somatic


@dataclass(frozen=True)
class PureCNResult:
    """Complete result container returned by `run_purecn` and `PureCN.run`."""

    purity: float
    ploidy: float
    solutions: list[PureCNSolution]
    segments: pd.DataFrame
    variants: pd.DataFrame
    tmb: TMBValue
    cin: CINValue
    loh_fraction: float
    settings: SettingsMode = "default"
    bins: pd.DataFrame | None = None

    def summary(self) -> str:
        n_som = (
            int(np.sum(self.variants["ml_somatic"].to_numpy(dtype=bool)))
            if not self.variants.empty and "ml_somatic" in self.variants.columns
            else 0
        )
        return (
            f"PureCNResult (settings={self.settings})\n"
            f"  Optimal purity:   {self.purity:.3f}\n"
            f"  Optimal ploidy:   {self.ploidy:.3f}\n"
            f"  Local optima:     {len(self.solutions)}\n"
            f"  Segments:         {len(self.segments)}\n"
            f"  Variants (som):   {len(self.variants)} ({n_som} somatic)\n"
            f"  TMB (mut/Mb):     {float(self.tmb):.2f}\n"
            f"  CIN fraction:     {float(self.cin):.3f}\n"
            f"  LOH fraction:     {self.loh_fraction:.3f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return allele-specific copy-number segment DataFrame."""
        return self.segments.copy()


def run_purecn(
    tumor_cov_df: pd.DataFrame | None = None,
    variants_df: pd.DataFrame | None = None,
    *,
    segments_df: pd.DataFrame | CoverageNormalizationResult | None = None,
    normal_panel: pd.DataFrame | np.ndarray | None = None,
    callable_mb: float | None = None,
    gc_correct: bool = True,
    tangent_norm: bool = True,
    purity_grid: np.ndarray | list[float] | None = None,
    ploidy_grid: np.ndarray | list[float] | None = None,
    max_cn: int = 8,
    base_error: float = 0.001,
    rho_overdisp: float | None = None,
    exclude_hotspot: bool = True,
    settings: SettingsMode = "default",
) -> PureCNResult:
    """
    Run the end-to-end PureCN tumor-only clinical panel workflow:
    1. Coverage normalization + Panel-of-Normals Tangent normalization + CBS segmentation
       (if `tumor_cov_df` is supplied; otherwise uses `segments_df`).
    2. Joint 2D `(purity, ploidy)` optimization across segments and SNVs.
    3. Beta-Binomial somatic-vs-germline classification, allele-specific `(C, M)`, LOH, and CCF.
    4. Artifact-cleaned somatic TMB, CIN, and LOH fraction calculation.
    """
    mode = validate_settings(settings)

    bins_df: pd.DataFrame | None = None
    if segments_df is not None:
        if isinstance(segments_df, CoverageNormalizationResult):
            bins_df = segments_df.bins
            segs_in = segments_df.segments
        else:
            segs_in = segments_df
    elif tumor_cov_df is not None:
        # Check if user passed pre-segmented DataFrame positionally as first arg
        if "seg_mean" in tumor_cov_df.columns and "coverage" not in tumor_cov_df.columns:
            segs_in = tumor_cov_df
        else:
            cov_res = normalize_tumor_only_coverage(
                tumor_cov_df,
                normal_panel=normal_panel,
                gc_correct=gc_correct,
                tangent_norm=tangent_norm,
                settings=mode,
            )
            bins_df = cov_res.bins
            segs_in = cov_res.segments
    else:
        raise ValueError("Provide either `tumor_cov_df` or `segments_df` to `run_purecn`.")

    solutions = grid_search_purity_ploidy(
        segs_in,
        variants_df=variants_df,
        purity_grid=purity_grid,
        ploidy_grid=ploidy_grid,
        max_cn=max_cn,
        settings=mode,
    )
    best = solutions[0]
    best_segs = best.segments.copy()

    if variants_df is not None and not variants_df.empty:
        var_res = predict_somatic(
            variants_df,
            best_segs,
            purity=best.purity,
            ploidy=best.ploidy,
            base_error=base_error,
            rho_overdisp=rho_overdisp,
            settings=mode,
        )
    else:
        var_res = pd.DataFrame()

    if callable_mb is None:
        if bins_df is not None:
            span_bp = float(
                np.sum(
                    np.maximum(
                        1,
                        bins_df["end"].to_numpy(dtype=np.int64)
                        - bins_df["start"].to_numpy(dtype=np.int64)
                        + 1,
                    )
                )
            )
            eff_mb = max(span_bp / 1e6, 0.1)
        else:
            span_bp = float(np.sum(best_segs["size_bp"].to_numpy(dtype=np.float64)))
            eff_mb = max(span_bp / 1e6, 1.0)
    else:
        eff_mb = float(callable_mb)

    tmb_val = (
        call_tmb(
            var_res,
            callable_mb=eff_mb,
            exclude_hotspot=exclude_hotspot,
            settings=mode,
        )
        if not var_res.empty
        else TMBValue(0.0, n_mutations=0, callable_mb=eff_mb)
    )
    cin_val = call_cin(best_segs, purity=best.purity, ploidy=best.ploidy, settings=mode)
    loh_frac = compute_loh_fraction(best_segs)

    return PureCNResult(
        purity=best.purity,
        ploidy=best.ploidy,
        solutions=solutions,
        segments=best_segs,
        variants=var_res,
        tmb=tmb_val,
        cin=cin_val,
        loh_fraction=loh_frac,
        settings=mode,
        bins=bins_df,
    )


class PureCN:
    """Object-oriented interface for PureCN tumor-only copy-number and SNV analysis."""

    def __init__(
        self,
        settings: SettingsMode = "default",
        *,
        gc_correct: bool = True,
        tangent_norm: bool = True,
        max_cn: int = 8,
        base_error: float = 0.001,
        rho_overdisp: float | None = None,
        exclude_hotspot: bool = True,
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        self.gc_correct = gc_correct
        self.tangent_norm = tangent_norm
        self.max_cn = max_cn
        self.base_error = base_error
        self.rho_overdisp = rho_overdisp
        self.exclude_hotspot = exclude_hotspot

    def run(
        self,
        tumor_cov_df: pd.DataFrame | None = None,
        variants_df: pd.DataFrame | None = None,
        *,
        segments_df: pd.DataFrame | CoverageNormalizationResult | None = None,
        normal_panel: pd.DataFrame | np.ndarray | None = None,
        callable_mb: float | None = None,
        purity_grid: np.ndarray | list[float] | None = None,
        ploidy_grid: np.ndarray | list[float] | None = None,
    ) -> PureCNResult:
        """Execute PureCN analysis and return `PureCNResult`."""
        return run_purecn(
            tumor_cov_df=tumor_cov_df,
            variants_df=variants_df,
            segments_df=segments_df,
            normal_panel=normal_panel,
            callable_mb=callable_mb,
            gc_correct=self.gc_correct,
            tangent_norm=self.tangent_norm,
            purity_grid=purity_grid,
            ploidy_grid=ploidy_grid,
            max_cn=self.max_cn,
            base_error=self.base_error,
            rho_overdisp=self.rho_overdisp,
            exclude_hotspot=self.exclude_hotspot,
            settings=self.settings,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> PureCNResult:
        return self.run(*args, **kwargs)
