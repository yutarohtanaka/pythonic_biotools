"""
Evolutionary pseudotime mutational-signature trajectory visualization for `tracksig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import TrackSigResult


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


_SIG_PALETTE: tuple[str, ...] = (
    "#2563eb",
    "#dc2626",
    "#d97706",
    "#059669",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#ea580c",
    "#65a30d",
)


def plot_tracksig(
    result: TrackSigResult,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (9.0, 4.5),
    *,
    show_raw: bool = True,
    title: str | None = None,
) -> Figure:
    """
    Plot evolutionary mutational signature trajectories across decreasing CCF
    pseudotime bins with optional bootstrap 95% credible bands and vertical
    dashed `PELT` changepoint boundaries.

    Parameters
    ----------
    result : TrackSigResult
        Fitted trajectory returned by `tracksig` or `TrackSig.fit`.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(9.0, 4.5)
        Figure size in inches when `ax is None`.
    show_raw : bool, default=True
        Overlay faint per-bin unpooled EM exposure estimates behind segmented trajectories.
    title : str | None, default=None
        Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered evolutionary timeline figure.
    """
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    bin_df = result.bin_exposures_df
    raw_df = result.raw_bin_exposures_df
    x_bins = bin_df["bin_id"].to_numpy(dtype=np.float64)
    mean_ccfs = bin_df["mean_ccf"].to_numpy(dtype=np.float64)
    n_bins = len(x_bins)

    has_ci = (
        result.bin_exposure_ci_low_df is not None
        and result.bin_exposure_ci_high_df is not None
    )

    for idx_s, sig_name in enumerate(result.active_signatures):
        color = _SIG_PALETTE[idx_s % len(_SIG_PALETTE)]
        y_exp = bin_df[sig_name].to_numpy(dtype=np.float64)

        if has_ci and result.bin_exposure_ci_low_df is not None and result.bin_exposure_ci_high_df is not None:
            if (
                sig_name in result.bin_exposure_ci_low_df.columns
                and sig_name in result.bin_exposure_ci_high_df.columns
            ):
                y_lo = result.bin_exposure_ci_low_df[sig_name].to_numpy(dtype=np.float64)
                y_hi = result.bin_exposure_ci_high_df[sig_name].to_numpy(dtype=np.float64)
                ax.fill_between(
                    x_bins,
                    y_lo,
                    y_hi,
                    color=color,
                    alpha=0.18,
                    zorder=2,
                )

        if show_raw and sig_name in raw_df.columns:
            y_raw = raw_df[sig_name].to_numpy(dtype=np.float64)
            ax.scatter(
                x_bins,
                y_raw,
                s=20.0,
                color=color,
                alpha=0.28,
                edgecolors="none",
                zorder=3,
            )

        ax.plot(
            x_bins,
            y_exp,
            marker="o",
            markersize=4.8,
            linewidth=2.2,
            color=color,
            label=sig_name,
            zorder=4,
        )

    # Vertical dashed PELT changepoint lines at bin boundaries (cp + 0.5)
    cp_labeled = False
    for cp_bin, cp_ccf in zip(result.changepoints, result.changepoint_ccfs):
        x_cp = float(cp_bin) + 0.5
        ax.axvline(
            x_cp,
            color="#0f172a",
            linestyle="--",
            linewidth=1.6,
            zorder=5,
            label="PELT changepoint" if not cp_labeled else None,
        )
        cp_labeled = True

        sup_txt = ""
        if result.changepoint_support is not None and cp_bin in result.changepoint_support:
            sup_txt = f"\n(boot={result.changepoint_support[cp_bin]:.0%})"
        ax.text(
            x_cp,
            1.02,
            f"CCF={cp_ccf:.2f}{sup_txt}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="medium",
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
                "alpha": 0.92,
            },
            zorder=6,
        )

    # Format x-ticks with bin ID and decreasing mean CCF
    if n_bins > 0:
        step = max(1, int(np.ceil(n_bins / 10)))
        tick_idx = np.arange(0, n_bins, step)
        if tick_idx[-1] != n_bins - 1 and (n_bins - 1 - tick_idx[-1]) >= max(1, step // 2):
            tick_idx = np.append(tick_idx, n_bins - 1)
        ax.set_xticks(x_bins[tick_idx])
        ax.set_xticklabels(
            [f"{int(x_bins[i])}\n({mean_ccfs[i]:.2f})" for i in tick_idx],
            fontsize=8.2,
        )
        ax.set_xlim(float(x_bins[0]) - 0.45, float(x_bins[-1]) + 0.45)

    ax.set_ylim(-0.03, 1.14 if result.changepoints else 1.05)
    ax.set_xlabel("Evolutionary Pseudotime Bin (Mean CCF: Clonal \u2192 Subclonal)", fontsize=9.5)
    ax.set_ylabel("Mutational Signature Exposure", fontsize=9.5)

    banner = (
        title
        if title is not None
        else (
            f"TrackSig ({result.score_method}) Evolutionary Trajectory  |  "
            f"SNVs = {result.n_mutations} ({result.n_bins} bins)  |  "
            f"Changepoints = {len(result.changepoints)}  |  BIC = {result.bic:.1f}"
        )
    )
    ax.set_title(banner, fontsize=10.5, fontweight="semibold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linestyle=":", linewidth=0.8, zorder=0)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=8.5,
    )
    return fig
