"""
Robust adjusted profile log-likelihood and multimodal pathway visualization for `grapple`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import GrappleResult


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


_MODE_COLORS: tuple[str, ...] = (
    "#dc2626",
    "#2563eb",
    "#d97706",
    "#059669",
    "#7c3aed",
    "#0891b2",
)


def plot_grapple_profile(
    result: GrappleResult,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8.0, 4.5),
    *,
    title: str | None = None,
) -> Figure:
    """
    Plot the GRAPPLE robust adjusted profile log-likelihood curve across the
    causal effect grid $\\beta$, marking every detected causal / pleiotropic
    local mode and its Stein-orthogonalized 95% confidence interval.

    Parameters
    ----------
    result : GrappleResult
        Fitted univariable GRAPPLE output returned by `grapple_robust_est` or `GRAPPLE.fit`.
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(8.0, 4.5)
        Figure size in inches when `ax is None`.
    title : str | None, default=None
        Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered GRAPPLE profile log-likelihood figure.
    """
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    grid_b = np.asarray(result.grid_beta, dtype=np.float64)
    grid_ll = np.asarray(result.grid_llk, dtype=np.float64)

    y_min = float(np.min(grid_ll)) if grid_ll.size > 0 else -1.0
    y_max = float(np.max(grid_ll)) if grid_ll.size > 0 else 0.0
    for m in result.modes:
        y_max = max(y_max, float(m.profile_llk))
    y_span = max(y_max - y_min, 1.0)
    y_floor = y_min - 0.06 * y_span

    # Null reference at beta = 0
    if grid_b.size > 0 and float(grid_b[0]) <= 0.0 <= float(grid_b[-1]):
        ax.axvline(0.0, color="#94a3b8", linestyle=":", linewidth=1.1, zorder=1, label=r"Null ($\beta = 0$)")

    # Main profile log-likelihood curve
    ax.plot(
        grid_b,
        grid_ll,
        color="#0f172a",
        linewidth=2.2,
        label="Profile log-likelihood",
        zorder=3,
    )
    ax.fill_between(
        grid_b,
        y_floor,
        grid_ll,
        color="#64748b",
        alpha=0.07,
        zorder=1,
    )

    # Detected modes with vertical markers & 95% CIs
    for idx_m, m in enumerate(result.modes):
        c_mode = _MODE_COLORS[idx_m % len(_MODE_COLORS)]
        ax.axvspan(
            m.ci_low,
            m.ci_high,
            color=c_mode,
            alpha=0.12,
            zorder=2,
        )
        ax.vlines(
            m.beta_hat,
            y_floor,
            m.profile_llk,
            colors=c_mode,
            linestyles="--",
            linewidth=1.6,
            zorder=4,
        )
        ax.plot(
            [m.beta_hat],
            [m.profile_llk],
            marker="o",
            markersize=6.5,
            color=c_mode,
            markeredgecolor="#ffffff",
            markeredgewidth=0.9,
            zorder=5,
        )
        # Horizontal 95% CI whisker just above each mode peak
        y_ci = m.profile_llk + (0.035 + 0.04 * (idx_m % 2)) * y_span
        ax.errorbar(
            m.beta_hat,
            y_ci,
            xerr=[[max(0.0, m.beta_hat - m.ci_low)], [max(0.0, m.ci_high - m.beta_hat)]],
            fmt="none",
            ecolor=c_mode,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.3,
            zorder=5,
        )
        # Keep empty plot handle with clean legend entry
        ax.plot(
            [],
            [],
            color=c_mode,
            linestyle="--",
            marker="o",
            markersize=5.0,
            label=(
                f"Mode {m.mode_id}: \u03b2={m.beta_hat:+.3f} "
                f"[{m.ci_low:+.3f}, {m.ci_high:+.3f}] (p={m.p_value:.2g})"
            ),
        )

    ax.set_ylim(y_floor, y_max + 0.16 * y_span)
    if grid_b.size >= 2:
        x_lo = min(float(grid_b[0]), *(m.ci_low for m in result.modes))
        x_hi = max(float(grid_b[-1]), *(m.ci_high for m in result.modes))
        pad_x = 0.03 * max(x_hi - x_lo, 0.1)
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)

    modality_str = (
        f"Multimodal ({result.n_modes} modes)"
        if result.is_multimodal
        else "Unimodal (1 mode)"
    )
    banner = (
        title
        if title is not None
        else (
            f"GRAPPLE Robust Profile Log-Likelihood  |  {modality_str}  |  "
            f"loss={result.loss_function} (k={result.k:.2f}, M={result.n_instruments})"
        )
    )
    ax.set_title(banner, fontsize=10.3, fontweight="semibold", pad=8)
    ax.set_xlabel(r"Causal / Pathway Effect ($\beta$)", fontsize=9.5)
    ax.set_ylabel("Robust Adjusted Profile Log-Likelihood", fontsize=9.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=8.2,
    )
    return fig
