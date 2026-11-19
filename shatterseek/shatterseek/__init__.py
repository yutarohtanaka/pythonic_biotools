"""
`shatterseek`: Clean-room Python implementation of ShatterSeek chromothripsis
detection and statistical classification (Cortes-Ciriano et al., Nature Genetics 2020).
"""

from .core import (
    ChromothripsisCluster,
    ShatterSeekResult,
    classify_chromothripsis,
    detect_chromothripsis,
    infer_sv_type,
)
from .stats import (
    GRCH37_CHROM_LENGTHS,
    GRCH38_CHROM_LENGTHS,
    get_reference_chrom_lengths,
    max_oscillating_cn_segments,
    pval_chr_breakpoint_enrichment,
    pval_exp_cluster,
    pval_fragment_joins,
)

__all__ = [
    "ChromothripsisCluster",
    "ShatterSeekResult",
    "detect_chromothripsis",
    "classify_chromothripsis",
    "infer_sv_type",
    "pval_fragment_joins",
    "pval_exp_cluster",
    "pval_chr_breakpoint_enrichment",
    "max_oscillating_cn_segments",
    "get_reference_chrom_lengths",
    "GRCH38_CHROM_LENGTHS",
    "GRCH37_CHROM_LENGTHS",
]

__version__ = "0.1.0"
