"""monaLisa: Binned sequence-motif enrichment and Randomized Lasso Stability Selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._validation import SettingsMode, validate_settings
from .binning import bin_genomic_changes
from .composition import DINUCLEOTIDES, calc_sequence_composition
from .enrichment import BinnedMotifEnrichmentResult, calc_binned_motif_enrichment
from .motifs import MotifScanResult, pfm_to_pwm, scan_sequences_pwm
from .stability_selection import StabilitySelectionResult, rand_lasso_stab_sel


class MonaLisa:
    """High-level object-oriented monaLisa workflow runner."""

    def __init__(self, settings: SettingsMode = "default") -> None:
        validate_settings(settings)
        self.settings: SettingsMode = settings
        self.enrichment_: BinnedMotifEnrichmentResult | None = None
        self.stability_: StabilitySelectionResult | None = None

    def run_binned_enrichment(
        self,
        sequences: Sequence[str] | pd.Series | np.ndarray,
        values_or_bins: np.ndarray | pd.Series | pd.Categorical | Sequence[Any],
        pwms: Mapping[str, np.ndarray | pd.DataFrame]
        | Sequence[np.ndarray | pd.DataFrame]
        | np.ndarray
        | pd.DataFrame,
        n_bins: int = 5,
        min_abs_val: float = 0.0,
        zero_bin: bool = True,
        background: str = "otherBins",
        pseudocount_log2enr: float | None = None,
        min_score: str | float = "80%",
        both_strands: bool = True,
    ) -> BinnedMotifEnrichmentResult:
        """Bin continuous changes (if needed) and compute binned motif enrichment."""
        if isinstance(values_or_bins, pd.Categorical) or (
            isinstance(values_or_bins, pd.Series)
            and isinstance(values_or_bins.dtype, pd.CategoricalDtype)
        ):
            bins = values_or_bins
        else:
            arr = np.asarray(values_or_bins)
            if np.issubdtype(arr.dtype, np.number):
                bins = bin_genomic_changes(
                    arr,
                    n_bins=n_bins,
                    min_abs_val=min_abs_val,
                    zero_bin=zero_bin,
                    settings=self.settings,
                )
            else:
                bins = values_or_bins

        self.enrichment_ = calc_binned_motif_enrichment(
            sequences_or_hits=sequences,
            bins=bins,
            pwms=pwms,
            background=background,
            pseudocount_log2enr=pseudocount_log2enr,
            min_score=min_score,
            both_strands=both_strands,
            settings=self.settings,
        )
        return self.enrichment_

    def run_stability_selection(
        self,
        sequences_or_hits: Sequence[str] | pd.Series | pd.DataFrame | np.ndarray | MotifScanResult,
        y_change: np.ndarray | pd.Series | Sequence[float],
        pwms: Mapping[str, np.ndarray | pd.DataFrame]
        | Sequence[np.ndarray | pd.DataFrame]
        | np.ndarray
        | pd.DataFrame
        | None = None,
        sequences: Sequence[str] | pd.Series | np.ndarray | None = None,
        include_seq_bias: bool = True,
        cutoff: float = 0.8,
        pfer: float = 1.0,
        n_subsamples: int = 100,
        weakness: float | None = None,
        n_alphas: int = 25,
        min_score: str | float = "80%",
        both_strands: bool = True,
        seed: int = 42,
    ) -> StabilitySelectionResult:
        """Scan sequences (if needed) and run Complementary-Pairs Randomized Lasso."""
        if pwms is not None and not isinstance(
            sequences_or_hits, (pd.DataFrame, MotifScanResult)
        ):
            seq_list = [str(s) for s in sequences_or_hits]  # type: ignore[union-attr]
            scan_res = scan_sequences_pwm(
                seq_list,
                pwms=pwms,
                min_score=min_score,
                both_strands=both_strands,
                settings=self.settings,
            )
            x_motifs: pd.DataFrame | np.ndarray | MotifScanResult = scan_res.counts
            seq_cov: Sequence[str] | pd.Series | np.ndarray | None = seq_list
        else:
            x_motifs = sequences_or_hits  # type: ignore[assignment]
            seq_cov = sequences

        self.stability_ = rand_lasso_stab_sel(
            X_motifs=x_motifs,
            y_change=y_change,
            sequences=seq_cov,
            include_seq_bias=include_seq_bias,
            cutoff=cutoff,
            pfer=pfer,
            n_subsamples=n_subsamples,
            weakness=weakness,
            n_alphas=n_alphas,
            seed=seed,
            settings=self.settings,
        )
        return self.stability_


__version__ = "0.1.0"

__all__ = [
    "BinnedMotifEnrichmentResult",
    "DINUCLEOTIDES",
    "MonaLisa",
    "MotifScanResult",
    "StabilitySelectionResult",
    "bin_genomic_changes",
    "calc_binned_motif_enrichment",
    "calc_sequence_composition",
    "pfm_to_pwm",
    "rand_lasso_stab_sel",
    "scan_sequences_pwm",
]
