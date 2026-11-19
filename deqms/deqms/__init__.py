"""DEqMS: Peptide/PSM-count-dependent Empirical Bayes for quantitative proteomics."""

from .core import DEqMS, DEqMSFit, LimmaFit, lm_fit, output_result, spectra_count_ebayes
from .preprocessing import (
    ProteinSummarizationResult,
    equal_median_normalization,
    median_sweeping,
    peptide_profile_summarization,
)

__version__ = "0.1.0"

__all__ = [
    "DEqMS",
    "DEqMSFit",
    "LimmaFit",
    "ProteinSummarizationResult",
    "equal_median_normalization",
    "lm_fit",
    "median_sweeping",
    "output_result",
    "peptide_profile_summarization",
    "spectra_count_ebayes",
]
