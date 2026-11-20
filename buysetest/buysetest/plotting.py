"""
Hierarchical prioritized-endpoint pairwise proportion bar chart for `buysetest`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import BuyseTestResult


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


def plot_buyse_test(
    result: BuyseTestResult,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    *,
    cumulative: bool = True,
    title: str | None = None,
) -> Figure:
    """
    Plot horizontal stacked pairwise proportion bars (`Favorable (Win)`,
    `Unfavorable (Loss)`, `Neutral`, and `Uninformative` %) across hierarchical
    prioritized composite endpoints alongside cumulative `Win Ratio` and
    `Net Benefit` annotations.

    Parameters
    ----------
    result : BuyseTestResult
        Fitted Generalized Pairwise Comparisons result returned by `buyse_test` or `BuyseTest.fit`.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(8.0, 4.0)
        Figure size in inches when `ax is None`.
    cumulative : bool, default=True
        When `True` (standard BuyseTest cumulative hierarchy), plots cumulative
        `favorable` / `unfavorable` + remaining `neutral` / `uninf` at each priority layer
        (each bar summing to 100%). When `False`, plots layer-specific incremental
        `delta_favorable` and `delta_unfavorable`.
    title : str | None, default=None
        Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered BuyseTest hierarchical stacked bar figure.
    """
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    n_ep = len(result.endpoint_names)
    y_pos = np.arange(n_ep, dtype=np.float64)

    if cumulative:
        win_pct = np.asarray(result.favorable, dtype=np.float64) * 100.0
        loss_pct = np.asarray(result.unfavorable, dtype=np.float64) * 100.0
    else:
        win_pct = np.asarray(result.delta_favorable, dtype=np.float64) * 100.0
        loss_pct = np.asarray(result.delta_unfavorable, dtype=np.float64) * 100.0

    neut_pct = np.asarray(result.neutral, dtype=np.float64) * 100.0
    uninf_pct = np.asarray(result.uninf, dtype=np.float64) * 100.0

    segments = [
        ("Favorable (Win)", win_pct, "#0d9488", "#ffffff"),
        ("Unfavorable (Loss)", loss_pct, "#dc2626", "#ffffff"),
        ("Neutral", neut_pct, "#94a3b8", "#0f172a"),
        ("Uninformative", uninf_pct, "#e2e8f0", "#334155"),
    ]

    bar_h = min(0.62, max(0.32, 1.8 / max(n_ep, 2)))
    left = np.zeros(n_ep, dtype=np.float64)

    for seg_label, widths, facecolor, textcolor in segments:
        ax.barh(
            y_pos,
            widths,
            left=left,
            height=bar_h,
            color=facecolor,
            edgecolor="#ffffff",
            linewidth=0.8,
            label=seg_label,
            zorder=3,
        )
        for i in range(n_ep):
            w_val = float(widths[i])
            if w_val >= 8.5:
                ax.text(
                    float(left[i] + 0.5 * w_val),
                    float(y_pos[i]),
                    f"{w_val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    fontweight="semibold",
                    color=textcolor,
                    zorder=4,
                )
        left = left + widths

    # Cumulative Win Ratio & Net Benefit annotations to the right of each bar
    for i in range(n_ep):
        wr_val = float(result.win_ratio.est[i])
        nb_val = float(result.net_benefit.est[i])
        nb_pv = float(result.net_benefit.pval[i])

        if np.isinf(wr_val):
            wr_str = "WR=\u221e"
        elif np.isnan(wr_val):
            wr_str = "WR=NA"
        else:
            wr_str = f"WR={wr_val:.2f}"

        nb_str = f"NB={nb_val * 100.0:+.1f}%"
        pv_str = f" (p={nb_pv:.2g})" if np.isfinite(nb_pv) else ""

        ax.text(
            float(left[i]) + 1.8,
            float(y_pos[i]),
            f"{wr_str}  |  {nb_str}{pv_str}",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="medium",
            color="#0f172a",
            zorder=5,
        )

    y_labels = []
    for idx_e, ep_name in enumerate(result.endpoint_names):
        thr = float(result.thresholds[idx_e])
        thr_str = f" (\u03c4={thr:g})" if thr > 0.0 else ""
        y_labels.append(f"{idx_e + 1}. {ep_name}{thr_str}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.8)
    ax.invert_yaxis()

    max_right = float(np.max(left)) if left.size > 0 else 100.0
    ax.set_xlim(0.0, max(142.0, max_right + 42.0))
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8.5)
    ax.axvline(100.0, color="#cbd5e1", linestyle=":", linewidth=0.9, zorder=1)
    ax.set_xlabel("Pairwise Comparison Proportion (%)", fontsize=9.5)
    ax.set_ylabel("Prioritized Endpoint", fontsize=9.5)

    banner = (
        title
        if title is not None
        else (
            f"BuyseTest Hierarchical Decomposition ({result.scoring_rule})  |  "
            f"{result.treatment_level} (m={result.n_treatment}) vs "
            f"{result.control_level} (n={result.n_control})"
        )
    )
    ax.set_title(banner, fontsize=10.0, fontweight="semibold", pad=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )
    return fig
