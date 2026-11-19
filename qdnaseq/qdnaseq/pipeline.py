"""
End-to-end `QDNAseqPipeline` combining QDNAseq shallow-WGS processing with
reference-free ACE absolute copy-number fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .ace import ACEResult, run_ace
from .config import SettingsMode, validate_settings
from .correction import (
    correct_counts,
    estimate_correction,
    normalize_bins,
    smooth_outlier_bins,
)
from .data import QDNAseqData, apply_filters, bin_read_counts
from .segment import call_bins, segment_bins


@dataclass(frozen=True)
class QDNAseqPipelineResult:
    """Result container from `QDNAseqPipeline.run`."""

    qdata: QDNAseqData
    ace_results: dict[str, ACEResult]
    settings: SettingsMode

    @property
    def primary_ace(self) -> ACEResult:
        first_key = next(iter(self.ace_results))
        return self.ace_results[first_key]

    @property
    def cellularity(self) -> float:
        return self.primary_ace.cellularity

    @property
    def ploidy(self) -> float:
        return self.primary_ace.ploidy

    @property
    def segments(self) -> pd.DataFrame:
        return self.primary_ace.segments

    def to_dataframe(self, sample: str | None = None) -> pd.DataFrame:
        if sample is not None and sample in self.ace_results:
            return self.ace_results[sample].segments.copy()
        return self.primary_ace.segments.copy()

    def summary(self) -> str:
        lines = [self.qdata.summary()]
        for sname, ares in self.ace_results.items():
            lines.append(
                f"  ACE [{sname}]: cellularity={ares.cellularity:.3f}, "
                f"ploidy={ares.ploidy:.3f}, error={ares.error:.4f}"
            )
        return "\n".join(lines)


class QDNAseqPipeline:
    """End-to-end shallow/low-pass WGS and cfDNA copy-number + ACE pipeline."""

    def __init__(
        self,
        settings: SettingsMode = "default",
        *,
        min_mapq: int | None = None,
        mappability: float = 50.0,
        blacklist: bool = True,
        residual: bool | float = True,
        chromosomes: Sequence[str] = ("X", "Y", "MT"),
        smooth_outliers: bool = True,
        ace_method: str | None = None,
        penploidy: float | None = None,
        pencellularity: float | None = None,
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        self.min_mapq = min_mapq
        self.mappability = mappability
        self.blacklist = blacklist
        self.residual = residual
        self.chromosomes = tuple(chromosomes)
        self.smooth_outliers = smooth_outliers
        self.ace_method = ace_method
        self.penploidy = penploidy
        self.pencellularity = pencellularity

    def run(
        self,
        bins_df: pd.DataFrame,
        bam_paths: str | Path | Sequence[str | Path] | dict[str, str | Path] | None = None,
        counts_df: pd.DataFrame | pd.Series | None = None,
    ) -> QDNAseqPipelineResult:
        """Execute full QDNAseq + ACE pipeline across one or more samples."""
        qd = bin_read_counts(
            bins_df,
            bam_paths=bam_paths,
            counts_df=counts_df,
            min_mapq=self.min_mapq,
            settings=self.settings,
        )
        qd = apply_filters(
            qd,
            residual=self.residual,
            blacklist=self.blacklist,
            mappability=self.mappability,
            chromosomes=self.chromosomes,
            settings=self.settings,
        )
        qd = estimate_correction(qd, settings=self.settings)
        qd = correct_counts(qd, settings=self.settings)
        qd = normalize_bins(qd, settings=self.settings)
        if self.smooth_outliers:
            qd = smooth_outlier_bins(qd, settings=self.settings)
        qd = segment_bins(qd, settings=self.settings)
        qd = call_bins(qd, settings=self.settings)

        ace_map: dict[str, ACEResult] = {}
        for sname in qd.sample_names:
            ace_map[sname] = run_ace(
                qd,
                method=self.ace_method,  # type: ignore[arg-type]
                penploidy=self.penploidy,
                pencellularity=self.pencellularity,
                sample=sname,
                settings=self.settings,
            )

        return QDNAseqPipelineResult(
            qdata=qd,
            ace_results=ace_map,
            settings=self.settings,
        )
