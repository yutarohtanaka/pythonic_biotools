"""
Publication-grade genome-wide copy-number visualization for `ichorcna`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .io import canonical_chrom, chrom_sort_key

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import IchorCNAResult


def _get_pyplot() -> tuple[Any, Any]:
    """Lazily import `matplotlib` with a writable sandbox-safe `MPLCONFIGDIR`."""
    import os
    import tempfile

    mpl_default = os.path.expanduser("~/.matplotlib")
    if "MPLCONFIGDIR" not in os.environ and not os.access(mpl_default, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install 'matplotlib>=3.7'`."
        ) from exc
    return matplotlib, plt


_STATE_COLORS: dict[str, str] = {
    "HOMD": "#08519c",
    "HETD": "#2b8cbe",
    "1_sub": "#67a9cf",
    "NEUT": "#737373",
    "3_sub": "#fd8d3c",
    "GAIN": "#e6550d",
    "AMP": "#cb181d",
    "HLAMP": "#7f0000",
}

_STATE_ORDER: tuple[str, ...] = (
    "HOMD",
    "HETD",
    "1_sub",
    "NEUT",
    "3_sub",
    "GAIN",
    "AMP",
    "HLAMP",
)


def _color_for_state(state_name: str) -> str:
    if state_name in _STATE_COLORS:
        return _STATE_COLORS[state_name]
    if state_name.startswith("HLAMP"):
        return _STATE_COLORS["HLAMP"]
    if state_name.endswith("_sub"):
        try:
            cn_val = int(state_name.split("_", 1)[0])
        except ValueError:
            cn_val = 3
        return _STATE_COLORS["1_sub"] if cn_val < 2 else _STATE_COLORS["3_sub"]
    return "#475569"


def plot_ichorcna(
    result: IchorCNAResult,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (11.0, 4.0),
    *,
    title: str | None = None,
) -> Figure:
    """
    Plot genome-wide `ichorCNA` `logR` profile colored by somatic copy-number state,
    horizontal segment median bars, chromosome boundaries, and `TFx`/`Ploidy` banner.

    Parameters
    ----------
    result : IchorCNAResult
        Fitted `ichorCNA` result returned by `run_ichorcna` or `IchorCNA.fit`.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(11.0, 4.0)
        Figure size in inches when `ax is None`.
    title : str | None, default=None
        Custom title override. Defaults to a summary banner with `TFx` and `Ploidy`.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure containing the genome-wide copy-number plot.
    """
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    bins_df = result.bins_df
    segs_df = result.segments_df

    banner = (
        title
        if title is not None
        else (
            f"ichorCNA Genome-Wide Profile  |  "
            f"Tumor Fraction (TFx) = {result.tumor_fraction:.3f} ({result.tumor_fraction:.1%})  |  "
            f"Ploidy = {result.ploidy:.2f}"
            + (
                f"  |  Subclone = {result.subclone_fraction:.1%}"
                if result.subclone_fraction > 0.0
                else ""
            )
        )
    )

    if bins_df.empty:
        ax.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel(r"Copy Number ($\log_2$ Ratio)")
        ax.set_title(banner, fontsize=10.5, fontweight="semibold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig

    # Build ordered chromosome offsets (in Mb)
    raw_chroms = list(dict.fromkeys(bins_df["chrom"].astype(str).tolist()))
    ordered_chroms = sorted(raw_chroms, key=lambda c: chrom_sort_key(canonical_chrom(c)))

    chrom_offsets: dict[str, float] = {}
    chrom_centers: list[float] = []
    chrom_labels: list[str] = []
    chrom_boundaries: list[float] = [0.0]

    cum_mb = 0.0
    for idx_c, ch in enumerate(ordered_chroms):
        b_sub = bins_df[bins_df["chrom"].astype(str) == ch]
        max_end = float(b_sub["end"].max()) if not b_sub.empty else 0.0
        if not segs_df.empty:
            s_sub = segs_df[segs_df["chrom"].astype(str) == ch]
            if not s_sub.empty:
                max_end = max(max_end, float(s_sub["end"].max()))
        span_mb = max(max_end * 1e-6, 0.5)
        chrom_offsets[ch] = cum_mb
        chrom_centers.append(cum_mb + 0.5 * span_mb)
        chrom_labels.append(canonical_chrom(ch))
        if idx_c % 2 == 1:
            ax.axvspan(cum_mb, cum_mb + span_mb, color="#64748b", alpha=0.05, zorder=0)
        cum_mb += span_mb
        chrom_boundaries.append(cum_mb)

    for b_pos in chrom_boundaries[1:-1]:
        ax.axvline(b_pos, color="#cbd5e1", linestyle="--", linewidth=0.75, zorder=1)

    ax.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)

    # Compute cumulative Mb coordinate for each bin
    bin_chrom_str = bins_df["chrom"].astype(str)
    offsets_arr = bin_chrom_str.map(chrom_offsets).to_numpy(dtype=np.float64)
    starts_mb = bins_df["start"].to_numpy(dtype=np.float64) * 1e-6
    ends_mb = bins_df["end"].to_numpy(dtype=np.float64) * 1e-6
    x_mid = offsets_arr + 0.5 * (starts_mb + ends_mb)
    y_logr = bins_df["logR"].to_numpy(dtype=np.float64)
    states_col = (
        bins_df["state"].astype(str).to_numpy()
        if "state" in bins_df.columns
        else np.full(len(bins_df), "NEUT", dtype=object)
    )

    present_states = list(dict.fromkeys(states_col.tolist()))
    ordered_present = [s for s in _STATE_ORDER if s in present_states] + [
        s for s in present_states if s not in _STATE_ORDER
    ]

    pt_size = 10.0 if len(bins_df) <= 1500 else 5.5
    rasterize_pts = len(bins_df) > 5000

    for st_name in ordered_present:
        mask = states_col == st_name
        if not np.any(mask):
            continue
        ax.scatter(
            x_mid[mask],
            y_logr[mask],
            s=pt_size,
            c=_color_for_state(st_name),
            alpha=0.82,
            edgecolors="none",
            label=st_name,
            zorder=3,
            rasterized=rasterize_pts,
        )

    # Horizontal segment median bars
    if not segs_df.empty and {"chrom", "start", "end", "median_logR"}.issubset(
        segs_df.columns
    ):
        seg_added_label = False
        for _, seg in segs_df.iterrows():
            ch_s = str(seg["chrom"])
            if ch_s not in chrom_offsets:
                continue
            off = chrom_offsets[ch_s]
            x0 = off + float(seg["start"]) * 1e-6
            x1 = off + float(seg["end"]) * 1e-6
            y_seg = float(seg["median_logR"])
            ax.hlines(
                y_seg,
                x0,
                x1,
                colors="#0f172a",
                linewidth=2.1,
                zorder=4,
                label="Segment median" if not seg_added_label else None,
            )
            seg_added_label = True

    ax.set_xlim(0.0, max(cum_mb, 1.0))
    y_finite = y_logr[np.isfinite(y_logr)]
    if y_finite.size > 0:
        y_lo = min(-1.2, float(np.percentile(y_finite, 1)) - 0.25)
        y_hi = max(1.4, float(np.percentile(y_finite, 99)) + 0.35)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_labels, fontsize=8.5)
    ax.tick_params(axis="x", length=3)
    ax.set_xlabel("Chromosome", fontsize=9.5)
    ax.set_ylabel(r"Copy Number ($\log_2$ Ratio)", fontsize=9.5)
    ax.set_title(banner, fontsize=10.5, fontweight="semibold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        ncol=min(len(ordered_present) + 1, 7),
        frameon=True,
        framealpha=0.9,
        edgecolor="#e2e8f0",
        fontsize=8.0,
        markerscale=1.2,
    )
    return fig
