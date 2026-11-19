"""
`figr`: Python implementation of FigR (Kartha et al., Cell Genomics 2022) for cis
peak-to-gene correlation, Domains of Regulatory Chromatin (DORCs), kNN-smoothed DORC
activity scoring, and TF-DORC gene regulatory network inference.
"""

from .background import get_background_peaks
from .corr import GenePeakCorrResult, run_gene_peak_corr
from .dorc import (
    DORCCallResult,
    DORCScoresResult,
    build_knn_graph,
    call_dorcs,
    get_dorc_scores,
    smooth_scores_knn,
)
from .grn import FigRGRNResult, run_figr_grn
from .io import extract_gtf_tss

__all__ = [
    "GenePeakCorrResult",
    "DORCCallResult",
    "DORCScoresResult",
    "FigRGRNResult",
    "run_gene_peak_corr",
    "call_dorcs",
    "get_dorc_scores",
    "smooth_scores_knn",
    "run_figr_grn",
    "get_background_peaks",
    "extract_gtf_tss",
    "build_knn_graph",
]

__version__ = "0.1.0"
