"""Global and local enrichment window filtering (csaw::filterWindowsGlobal / Local)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import SettingsMode, ave_log_cpm, scaled_average, validate_settings
from .windows import WindowData


@dataclass
class FilterResult:
    """Result of global or local ChIP/ATAC enrichment filtering."""

    abundances: np.ndarray
    back_abundances: np.ndarray | float
    filter_stat: np.ndarray
    keep: np.ndarray
    min_fc: float
    filtered_data: WindowData | None = None


def filter_windows_global(
    window_data: WindowData | np.ndarray,
    background_bins: WindowData | np.ndarray | None = None,
    min_fc: float = 3.0,
    bin_width_ratio: float | None = None,
    totals: np.ndarray | None = None,
    prior_count: float = 2.0,
    settings: SettingsMode = "default",
) -> FilterResult:
    """Filter sliding windows by enrichment above median global genomic background.

    Implements ``csaw::filterWindowsGlobal`` (Lun & Smyth, *NAR* 2016).

    Parameters
    ----------
    window_data : WindowData | np.ndarray
        Sliding-window counts or ``WindowData``.
    background_bins : WindowData | np.ndarray | None
        Large genomic bins (e.g., 10 kb from ``window_counts(..., bin_mode=True)``) used
        to estimate the median uninterested background read density across the genome.
    min_fc : float
        Minimum fold-enrichment over global background required to retain a window.
    bin_width_ratio : float | None
        Ratio ``bin_width / window_width`` when raw numpy arrays are passed instead of
        ``WindowData``. Defaults to ``10000 / 150`` (``"default"``) or ``10000 / 50``
        (``"R_default"``) if raw arrays are provided without ``bin_width_ratio``.
    totals : np.ndarray | None
        Library sizes if raw numpy arrays are passed.
    prior_count : float
        Prior count for ``ave_log_cpm``.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    FilterResult
        Window abundances, background estimate, log2 fold-enrichment ``filter_stat``,
        boolean ``keep`` mask, and ``filtered_data`` (when ``window_data`` is ``WindowData``).
    """
    validate_settings(settings)
    if min_fc <= 0:
        raise ValueError(f"min_fc must be positive, got {min_fc}.")

    if isinstance(window_data, WindowData):
        win_counts = window_data.counts
        lib_sizes = window_data.totals if totals is None else np.asarray(totals, dtype=float)
        win_width = float(window_data.width)
    else:
        win_counts = np.asarray(window_data, dtype=float)
        lib_sizes = np.sum(win_counts, axis=0) if totals is None else np.asarray(totals, dtype=float)
        win_width = 150.0 if settings == "default" else 50.0

    win_ab = ave_log_cpm(win_counts, lib_sizes=lib_sizes, prior_count=prior_count)

    if background_bins is not None:
        if isinstance(background_bins, WindowData):
            bg_counts = background_bins.counts
            bg_libs = background_bins.totals
            scale = (
                float(bin_width_ratio)
                if bin_width_ratio is not None
                else float(background_bins.width) / win_width
            )
        else:
            bg_counts = np.asarray(background_bins, dtype=float)
            bg_libs = lib_sizes
            scale = (
                float(bin_width_ratio)
                if bin_width_ratio is not None
                else (10000.0 / win_width)
            )
        bg_ab_bins = scaled_average(
            bg_counts, lib_sizes=bg_libs, scale=scale, prior_count=prior_count
        )
        global_bg = float(np.median(bg_ab_bins)) if bg_ab_bins.size > 0 else -np.inf
    else:
        global_bg = float(np.median(win_ab)) if win_ab.size > 0 else -np.inf

    filter_stat = win_ab - global_bg
    keep = filter_stat >= np.log2(min_fc)
    filtered = window_data.subset(keep) if isinstance(window_data, WindowData) else None

    return FilterResult(
        abundances=win_ab,
        back_abundances=global_bg,
        filter_stat=filter_stat,
        keep=keep,
        min_fc=min_fc,
        filtered_data=filtered,
    )


def filter_windows_local(
    window_data: WindowData | np.ndarray,
    flank_counts: WindowData | np.ndarray,
    flank_width_ratio: float | None = None,
    min_fc: float = 3.0,
    inner_included: bool = True,
    totals: np.ndarray | None = None,
    prior_count: float = 2.0,
    settings: SettingsMode = "default",
) -> FilterResult:
    """Filter sliding windows by enrichment above their immediate local neighborhood.

    Implements ``csaw::filterWindowsLocal`` (Lun & Smyth, *NAR* 2016).

    Parameters
    ----------
    window_data : WindowData | np.ndarray
        Inner sliding-window counts or ``WindowData``.
    flank_counts : WindowData | np.ndarray
        Surrounding neighborhood counts aligned 1-to-1 with the rows of ``window_data``.
    flank_width_ratio : float | None
        Total ratio of surrounding locus width to inner window width (defaults to
        ``flank_counts.width / window_data.width`` if both are ``WindowData``, else ``5.0``).
    min_fc : float
        Minimum fold-enrichment over local flank required to retain a window.
    inner_included : bool
        If ``True`` (matching ``csaw::filterWindowsLocal`` where ``surround`` encompasses the
        inner window), subtracts ``window_counts`` from ``flank_counts`` and uses effective
        width ratio ``flank_width_ratio - 1``. If ``False``, treats ``flank_counts`` as pure
        flanking counts of width ratio ``flank_width_ratio``.
    totals : np.ndarray | None
        Library sizes if raw numpy arrays are passed.
    prior_count : float
        Prior count for ``ave_log_cpm``.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    FilterResult
        Window abundances, local background abundances, ``filter_stat``, ``keep`` mask,
        and ``filtered_data``.
    """
    validate_settings(settings)
    if min_fc <= 0:
        raise ValueError(f"min_fc must be positive, got {min_fc}.")

    if isinstance(window_data, WindowData):
        win_counts = window_data.counts
        lib_sizes = window_data.totals if totals is None else np.asarray(totals, dtype=float)
        win_width = float(window_data.width)
    else:
        win_counts = np.asarray(window_data, dtype=float)
        lib_sizes = np.sum(win_counts, axis=0) if totals is None else np.asarray(totals, dtype=float)
        win_width = 150.0 if settings == "default" else 50.0

    if isinstance(flank_counts, WindowData):
        f_counts = flank_counts.counts
        ratio = (
            float(flank_width_ratio)
            if flank_width_ratio is not None
            else float(flank_counts.width) / win_width
        )
    else:
        f_counts = np.asarray(flank_counts, dtype=float)
        ratio = 5.0 if flank_width_ratio is None else float(flank_width_ratio)

    if f_counts.shape != win_counts.shape:
        raise ValueError(
            f"flank_counts shape {f_counts.shape} must match window_data counts shape {win_counts.shape}."
        )

    if inner_included:
        if ratio <= 1.0:
            raise ValueError(
                f"When inner_included=True, flank_width_ratio ({ratio}) must exceed 1.0."
            )
        pure_flank = np.maximum(0.0, f_counts - win_counts)
        eff_scale = ratio - 1.0
    else:
        if ratio <= 0.0:
            raise ValueError(f"flank_width_ratio ({ratio}) must be positive.")
        pure_flank = np.maximum(0.0, f_counts)
        eff_scale = ratio

    win_ab = ave_log_cpm(win_counts, lib_sizes=lib_sizes, prior_count=prior_count)
    flank_ab = scaled_average(
        pure_flank, lib_sizes=lib_sizes, scale=eff_scale, prior_count=prior_count
    )

    filter_stat = win_ab - flank_ab
    keep = filter_stat >= np.log2(min_fc)
    filtered = window_data.subset(keep) if isinstance(window_data, WindowData) else None

    return FilterResult(
        abundances=win_ab,
        back_abundances=flank_ab,
        filter_stat=filter_stat,
        keep=keep,
        min_fc=min_fc,
        filtered_data=filtered,
    )
