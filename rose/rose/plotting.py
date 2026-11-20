"""
Classic ROSE ranked enhancer "hockey-stick" curve and tangent inflection visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import ROSEResult


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


def _prepare_rose_table(
    result_or_df: ROSEResult | pd.DataFrame,
) -> tuple[pd.DataFrame, float, int, str]:
    """Normalize either a `ROSEResult` or stitched-enhancer `DataFrame` for plotting."""
    from .core import ROSEResult

    if isinstance(result_or_df, ROSEResult):
        df = result_or_df.stitched_regions.copy()
        if not df.empty and "rank_ascending" in df.columns:
            df = df.sort_values("rank_ascending", kind="mergesort").reset_index(drop=True)
        return (
            df,
            float(result_or_df.cutoff_signal),
            int(result_or_df.cutoff_rank_ascending),
            str(result_or_df.signal_unit),
        )

    if not isinstance(result_or_df, pd.DataFrame):
        raise TypeError(
            "result_or_df must be a ROSEResult or pandas DataFrame of stitched enhancers."
        )

    df = result_or_df.copy()
    if df.empty:
        return df, 0.0, 0, "Signal"

    if "signal" not in df.columns:
        if "score" in df.columns:
            df["signal"] = df["score"].astype(float)
        else:
            raise ValueError("DataFrame must contain a 'signal' (or 'score') column.")

    df = df.sort_values("signal", ascending=True, kind="mergesort").reset_index(drop=True)
    n_dom = len(df)
    sigs = df["signal"].to_numpy(dtype=np.float64)
    if "rank_ascending" not in df.columns:
        df["rank_ascending"] = np.arange(1, n_dom + 1, dtype=np.int64)
    if "rank_descending" not in df.columns:
        df["rank_descending"] = np.arange(n_dom, 0, -1, dtype=np.int64)

    max_sig = float(sigs[-1]) if sigs[-1] > 0.0 else 1.0
    x_norm = np.arange(n_dom, dtype=np.float64) / max(1, n_dom - 1)
    y_norm = sigs / max_sig
    cutoff_idx = int(np.argmax(x_norm - y_norm))
    cutoff_val = float(sigs[cutoff_idx])
    cutoff_rank = int(df["rank_ascending"].iloc[cutoff_idx])

    if "is_super_enhancer" not in df.columns:
        df["is_super_enhancer"] = sigs > cutoff_val
    else:
        se_sub = df.loc[~df["is_super_enhancer"].astype(bool)]
        if not se_sub.empty:
            cutoff_val = float(se_sub["signal"].max())
            cutoff_rank = int(se_sub["rank_ascending"].max())

    return df, cutoff_val, cutoff_rank, "Signal"


def plot_rose_hockey_stick(
    result_or_df: ROSEResult | pd.DataFrame,
    ax: Axes | None = None,
    top_n_labels: int = 5,
    figsize: tuple[float, float] = (7.0, 5.0),
    *,
    title: str | None = None,
) -> Figure:
    """
    Plot the classic ROSE ranked stitched-enhancer "hockey-stick" curve with
    geometric tangent cutoff threshold separating Typical Enhancers from
    Super-Enhancers and annotating top-ranked Super-Enhancers.

    Parameters
    ----------
    result_or_df : ROSEResult | pd.DataFrame
        `ROSEResult` from `call_rose_super_enhancers` or stitched-enhancer `DataFrame`.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    top_n_labels : int, default=5
        Number of top-ranked Super-Enhancers to label by `nearest_gene` (or locus).
    figsize : tuple[float, float], default=(7.0, 5.0)
        Figure size in inches when `ax is None`.
    title : str | None, default=None
        Custom plot title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered ROSE hockey-stick figure.
    """
    _, plt = _get_pyplot()

    df, cutoff_sig, cutoff_rank, sig_unit = _prepare_rose_table(result_or_df)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if df.empty:
        ax.set_xlabel("Enhancers Ranked by Increasing Signal")
        ax.set_ylabel(f"Enhancer Signal ({sig_unit})")
        ax.set_title(
            title if title is not None else "ROSE Ranked Enhancer Profile (0 regions)",
            fontsize=10.5,
            fontweight="semibold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig

    ranks = df["rank_ascending"].to_numpy(dtype=np.float64)
    sigs = df["signal"].to_numpy(dtype=np.float64)
    is_se = df["is_super_enhancer"].to_numpy(dtype=bool)
    n_se = int(np.sum(is_se))
    n_te = int(len(df) - n_se)

    # Connecting ranked signal backbone
    ax.plot(ranks, sigs, color="#94a3b8", linewidth=1.4, zorder=2)

    if np.any(~is_se):
        ax.scatter(
            ranks[~is_se],
            sigs[~is_se],
            s=18.0,
            c="#64748b",
            alpha=0.75,
            edgecolors="none",
            label=f"Typical Enhancers (n={n_te})",
            zorder=3,
        )

    if np.any(is_se):
        ax.scatter(
            ranks[is_se],
            sigs[is_se],
            s=36.0,
            c="#dc2626",
            alpha=0.92,
            edgecolors="#7f1d1d",
            linewidths=0.45,
            label=f"Super-Enhancers (n={n_se})",
            zorder=4,
        )

    # Dashed cutoff lines + geometric tangent line at inflection point
    ax.axhline(
        cutoff_sig,
        color="#334155",
        linestyle="--",
        linewidth=1.25,
        label=f"Inflection Cutoff ({cutoff_sig:.1f})",
        zorder=3,
    )
    if cutoff_rank > 0:
        ax.axvline(
            float(cutoff_rank),
            color="#94a3b8",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
        )

    if len(ranks) >= 2 and sigs[-1] > sigs[0] and cutoff_rank > 0:
        slope = (float(sigs[-1]) - float(sigs[0])) / max(float(ranks[-1] - ranks[0]), 1.0)
        x_tan = np.linspace(float(ranks[0]), float(ranks[-1]), 100)
        y_tan = cutoff_sig + slope * (x_tan - float(cutoff_rank))
        valid_tan = (y_tan >= min(0.0, float(sigs[0]))) & (y_tan <= float(sigs[-1]) * 1.05)
        if np.any(valid_tan):
            ax.plot(
                x_tan[valid_tan],
                y_tan[valid_tan],
                color="#d97706",
                linestyle=":",
                linewidth=1.25,
                label="Tangent slope",
                zorder=2,
            )

    # Annotate top Super-Enhancers
    if top_n_labels > 0 and n_se > 0:
        se_df = df.loc[is_se].sort_values("signal", ascending=False, kind="mergesort")
        top_df = se_df.head(int(top_n_labels))
        x_span = max(float(ranks[-1] - ranks[0]), 1.0)
        y_span = max(float(sigs[-1] - min(0.0, float(sigs[0]))), 1.0)

        for rank_i, (_, row) in enumerate(top_df.iterrows()):
            gene_lbl = str(row.get("nearest_gene", "")).strip()
            if not gene_lbl and {"chrom", "start", "end"}.issubset(df.columns):
                gene_lbl = f"{row['chrom']}:{int(row['start']):,}-{int(row['end']):,}"
            elif not gene_lbl:
                gene_lbl = f"SE #{rank_i + 1}"

            px = float(row["rank_ascending"])
            py = float(row["signal"])
            tx = max(float(ranks[0]) + 0.05 * x_span, px - (0.18 + 0.04 * (rank_i % 2)) * x_span)
            ty = py - 0.055 * rank_i * y_span
            ax.annotate(
                gene_lbl,
                xy=(px, py),
                xytext=(tx, ty),
                ha="right",
                va="center",
                fontsize=8.5,
                fontweight="semibold",
                color="#991b1b",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#dc2626",
                    "lw": 0.85,
                    "alpha": 0.7,
                },
                zorder=6,
            )

    ax.set_xlabel("Stitched Enhancers Ranked by Signal", fontsize=9.5)
    ax.set_ylabel(f"Enhancer Signal ({sig_unit})", fontsize=9.5)
    banner = (
        title
        if title is not None
        else f"ROSE Super-Enhancer Hockey-Stick  |  {n_se} Super / {n_te} Typical"
    )
    ax.set_title(banner, fontsize=10.5, fontweight="semibold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=8.3,
    )
    return fig
