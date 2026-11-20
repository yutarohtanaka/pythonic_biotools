"""
Publication-grade visualizations for `monalisa`:
  1. Diverging / sequential binned motif enrichment heatmaps (`plot_binned_motif_enrichment`)
  2. Complementary-Pairs Randomized Lasso stability selection paths (`plot_stability_selection`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .enrichment import BinnedMotifEnrichmentResult
    from .stability_selection import StabilitySelectionResult


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


_METRIC_ALIASES: dict[str, str] = {
    "log2enr": "log2enr",
    "neg_log10_padj": "neg_log10_padj",
    "-log10(padj)": "neg_log10_padj",
    "padj": "neg_log10_padj",
    "neg_log10_p": "neg_log10_p",
    "-log10(p)": "neg_log10_p",
    "pearson_resid": "pearson_resid",
}

_PATH_PALETTE: tuple[str, ...] = (
    "#dc2626",
    "#2563eb",
    "#d97706",
    "#059669",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#ea580c",
    "#65a30d",
)


def plot_binned_motif_enrichment(
    result: BinnedMotifEnrichmentResult,
    metric: str = "log2enr",
    top_n: int = 15,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    *,
    title: str | None = None,
) -> Figure:
    """
    Plot a binned transcription factor motif enrichment heatmap (`log2enr`,
    `-log10(padj)`, `-log10(p)`, or `pearson_resid`) across quantitative
    genomic change bins (`monaLisa::plotMotifHeatmaps`).

    Parameters
    ----------
    result : BinnedMotifEnrichmentResult
        Output of `calc_binned_motif_enrichment` or `MonaLisa.run_binned_enrichment`.
    metric : str, default="log2enr"
        Heatmap quantity: `"log2enr"`, `"neg_log10_padj"` (`"-log10(padj)"`),
        `"neg_log10_p"` (`"-log10(p)"`), or `"pearson_resid"`.
    top_n : int, default=15
        Maximum number of top-ranked motifs to display.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(8.0, 5.0)
        Figure size in inches when `ax is None`.
    title : str | None, default=None
        Custom plot title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered motif enrichment heatmap figure.
    """
    if metric not in _METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric={metric!r}. Expected one of {sorted(_METRIC_ALIASES)}."
        )
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    _, plt = _get_pyplot()
    canon_metric = _METRIC_ALIASES[metric]
    mat_df: pd.DataFrame = getattr(result, canon_metric)
    padj_df = result.neg_log10_padj
    enr_df = result.log2enr

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Rank motifs by peak -log10(padj), breaking ties by peak |log2enr|
    max_padj = np.nanmax(padj_df.to_numpy(dtype=np.float64), axis=1)
    max_abs_enr = np.nanmax(np.abs(enr_df.to_numpy(dtype=np.float64)), axis=1)
    rank_df = pd.DataFrame(
        {
            "motif": list(mat_df.index),
            "max_padj": np.nan_to_num(max_padj, nan=0.0),
            "max_abs_enr": np.nan_to_num(max_abs_enr, nan=0.0),
        }
    ).sort_values(
        by=["max_padj", "max_abs_enr"],
        ascending=[False, False],
        kind="mergesort",
    )
    chosen_motifs = rank_df["motif"].head(int(top_n)).tolist()

    sub_mat = mat_df.loc[chosen_motifs]
    sub_padj = padj_df.loc[chosen_motifs]
    vals = np.nan_to_num(sub_mat.to_numpy(dtype=np.float64), nan=0.0)
    padj_vals = np.nan_to_num(sub_padj.to_numpy(dtype=np.float64), nan=0.0)

    is_diverging = canon_metric in ("log2enr", "pearson_resid")
    if is_diverging:
        vmax = max(float(np.max(np.abs(vals))) if vals.size > 0 else 1.0, 0.25)
        vmin = -vmax
        cmap_name = "RdBu_r"
        cbar_label = (
            r"Motif Enrichment ($\log_2$ fold-change)"
            if canon_metric == "log2enr"
            else "Pearson Residual (Z)"
        )
    else:
        vmin = 0.0
        vmax = max(float(np.max(vals)) if vals.size > 0 else 2.0, 1.0)
        cmap_name = "YlOrRd"
        cbar_label = (
            r"$-\log_{10}(p_{\mathrm{adj}})$"
            if canon_metric == "neg_log10_padj"
            else r"$-\log_{10}(p)$"
        )

    im = ax.imshow(vals, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
    cbar.set_label(cbar_label, fontsize=8.8)
    cbar.ax.tick_params(labelsize=8.0)

    n_rows, n_cols = vals.shape
    if n_rows <= 25 and n_cols <= 12:
        span = max(vmax - vmin, 1e-6)
        for r_i in range(n_rows):
            for c_i in range(n_cols):
                nlp = float(padj_vals[r_i, c_i])
                if nlp >= 3.0:
                    star = "***"
                elif nlp >= 2.0:
                    star = "**"
                elif nlp >= -float(np.log10(0.05)):
                    star = "*"
                else:
                    star = ""
                if star:
                    norm_v = (float(vals[r_i, c_i]) - vmin) / span
                    txt_col = (
                        "#ffffff"
                        if (
                            (is_diverging and (norm_v < 0.22 or norm_v > 0.78))
                            or (not is_diverging and norm_v > 0.60)
                        )
                        else "#0f172a"
                    )
                    ax.text(
                        c_i,
                        r_i,
                        star,
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        fontweight="bold",
                        color=txt_col,
                    )

    bin_labels = [str(c) for c in sub_mat.columns]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(bin_labels, rotation=25 if n_cols > 5 else 0, ha="right" if n_cols > 5 else "center", fontsize=8.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(chosen_motifs, fontsize=8.5)
    ax.set_xlabel("Genomic Change Bin", fontsize=9.5)
    ax.set_ylabel("Transcription Factor Motif", fontsize=9.5)

    banner = (
        title
        if title is not None
        else (
            f"monaLisa Binned Motif Enrichment ({canon_metric})  |  "
            f"background={result.background}  |  * FDR < 0.05"
        )
    )
    ax.set_title(banner, fontsize=10.2, fontweight="semibold", pad=8)
    return fig


def plot_stability_selection(
    result: StabilitySelectionResult,
    top_n: int = 10,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8.0, 4.5),
    *,
    title: str | None = None,
) -> Figure:
    """
    Plot Complementary-Pairs Randomized Lasso selection probability trajectories
    across regularization path steps (`monaLisa::plotStabilityPaths`), highlighting
    features above `cutoff`.

    Parameters
    ----------
    result : StabilitySelectionResult
        Output of `rand_lasso_stab_sel` or `MonaLisa.run_stability_selection`.
    top_n : int, default=10
        Maximum number of top-probability features to color and include in legend.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(8.0, 4.5)
        Figure size in inches when `ax is None`.
    title : str | None, default=None
        Custom plot title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered stability selection path figure.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    path_df = result.prob_path_df
    n_steps = path_df.shape[1]
    x_steps = np.arange(1, n_steps + 1, dtype=np.float64)

    ordered_feats = result.sel_prob.sort_values(ascending=False, kind="mergesort")
    highlight_feats = list(ordered_feats.head(int(top_n)).index)
    highlight_set = set(highlight_feats)
    motif_set = set(result.motif_names)

    # Background unselected trajectories
    bg_labeled = False
    for feat in path_df.index:
        if feat in highlight_set:
            continue
        ax.plot(
            x_steps,
            path_df.loc[feat].to_numpy(dtype=np.float64),
            color="#cbd5e1",
            linewidth=1.0,
            alpha=0.65,
            zorder=1,
            label="Other features" if not bg_labeled else None,
        )
        bg_labeled = True

    # Highlighted top trajectories
    for idx_f, feat in enumerate(highlight_feats):
        prob_max = float(result.sel_prob.loc[feat])
        is_selected = prob_max >= result.cutoff
        is_motif = feat in motif_set
        color = (
            _PATH_PALETTE[idx_f % len(_PATH_PALETTE)]
            if is_selected
            else ("#475569" if idx_f >= 3 else _PATH_PALETTE[idx_f % len(_PATH_PALETTE)])
        )
        ls = "-" if is_motif else "--"
        lw = 2.3 if is_selected else 1.5
        alpha = 0.96 if is_selected else 0.72
        tag = "" if is_motif else " [bias]"
        ax.plot(
            x_steps,
            path_df.loc[feat].to_numpy(dtype=np.float64),
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            zorder=4 if is_selected else 3,
            label=f"{feat}{tag} ({prob_max:.2f})",
        )

    ax.axhline(
        result.cutoff,
        color="#dc2626",
        linestyle=":",
        linewidth=1.6,
        zorder=5,
        label=f"Cutoff = {result.cutoff:.2f} (PFER \u2264 {result.pfer:.2g})",
    )

    ax.set_xlim(1.0, max(float(n_steps), 2.0))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Regularization Path Step (Decreasing Lasso Penalty \u03b1)", fontsize=9.5)
    ax.set_ylabel("Selection Probability", fontsize=9.5)

    banner = (
        title
        if title is not None
        else (
            f"monaLisa Randomized Lasso Stability Selection  |  "
            f"Selected Motifs = {len(result.selected_motifs)}  |  "
            f"2B = {2 * result.n_subsamples} subsamples"
        )
    )
    ax.set_title(banner, fontsize=10.2, fontweight="semibold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linestyle=":", linewidth=0.8, zorder=0)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=7.8,
        ncol=1 if len(highlight_feats) <= 6 else 2,
    )
    return fig
