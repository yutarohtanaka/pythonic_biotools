"""
`qdnaseq`: Clean-room Python implementation of QDNAseq (Scheinin et al., Genome
Research 2014) and ACE (Poell et al., Bioinformatics 2019) for shallow/low-pass
WGS and cfDNA liquid-biopsy copy-number calling and cellularity/ploidy estimation.
"""

from __future__ import annotations

from .ace import ACEResult, run_ace, squaremodel
from .config import QDNAseqConfig, SettingsMode, get_qdnaseq_config, validate_settings
from .correction import (
    correct_counts,
    estimate_correction,
    fit_2d_loess_surface,
    normalize_bins,
    smooth_outlier_bins,
)
from .data import QDNAseqData, apply_filters, bin_read_counts, canonical_chrom_name
from .pipeline import QDNAseqPipeline, QDNAseqPipelineResult
from .segment import call_bins, segment_bins

__all__ = [
    "SettingsMode",
    "QDNAseqConfig",
    "get_qdnaseq_config",
    "validate_settings",
    "canonical_chrom_name",
    "QDNAseqData",
    "ACEResult",
    "QDNAseqPipelineResult",
    "QDNAseqPipeline",
    "bin_read_counts",
    "apply_filters",
    "fit_2d_loess_surface",
    "estimate_correction",
    "correct_counts",
    "normalize_bins",
    "smooth_outlier_bins",
    "segment_bins",
    "call_bins",
    "squaremodel",
    "run_ace",
]

__version__ = "0.1.0"
