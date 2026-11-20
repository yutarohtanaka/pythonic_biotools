"""
Publication-grade Kaplan-Meier RMST / RMTL visualization for `survrm2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import RMST1Result, RMST2Result


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


def _km_curve_from_raw(t_arr: np.ndarray, d_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute full right-continuous KM knots `(uniq_times, surv)` across all observed times."""
    uniq = np.unique(t_arr)
    surv = np.empty(uniq.size, dtype=np.float64)
    cur_s = 1.0
    for idx, tk in enumerate(uniq):
        yk = float(np.sum(t_arr >= tk))
        dk = float(np.sum((t_arr == tk) & (d_arr == 1)))
        if yk > 0.0:
            cur_s *= 1.0 - dk / yk
        surv[idx] = cur_s
    return uniq, surv


def _step_arrays_up_to_tau(
    uniq_times: np.ndarray,
    surv: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(t_step, s_step)` on `[0, tau]` suitable for `where='post'`."""
    mask = uniq_times <= tau
    ut = uniq_times[mask]
    sv = surv[mask]
    t_list: list[float] = [0.0]
    s_list: list[float] = [1.0]
    for tk, sk in zip(ut, sv):
        t_list.append(float(tk))
        s_list.append(float(sk))
    if not ut.size or float(ut[-1]) < tau:
        t_list.append(float(tau))
        s_list.append(float(sv[-1]) if sv.size > 0 else 1.0)
    return np.asarray(t_list, dtype=np.float64), np.asarray(s_list, dtype=np.float64)


def _eval_km_right_continuous(
    uniq_times: np.ndarray,
    surv: np.ndarray,
    grid_t: np.ndarray,
) -> np.ndarray:
    """Evaluate right-continuous KM survival $\\hat{S}(t)$ at arbitrary points `grid_t`."""
    if uniq_times.size == 0:
        return np.ones_like(grid_t, dtype=np.float64)
    idx = np.searchsorted(uniq_times, grid_t, side="right") - 1
    out = np.ones_like(grid_t, dtype=np.float64)
    valid = idx >= 0
    out[valid] = surv[idx[valid]]
    return out


def _extract_arm_knots(arm_res: RMST1Result) -> tuple[np.ndarray, np.ndarray]:
    """Extract `(uniq_times, surv)` from `RMST1Result` (including post-tau if stored)."""
    if arm_res.km_full_df is not None and not arm_res.km_full_df.empty:
        return (
            arm_res.km_full_df["time"].to_numpy(dtype=np.float64),
            arm_res.km_full_df["surv"].to_numpy(dtype=np.float64),
        )
    if not arm_res.km_df.empty:
        return (
            arm_res.km_df["time"].to_numpy(dtype=np.float64),
            arm_res.km_df["surv"].to_numpy(dtype=np.float64),
        )
    return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)


def plot_rmst1(
    result: RMST1Result,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    *,
    title: str | None = None,
) -> Figure:
    """Plot single-arm Kaplan-Meier curve shading RMST vs RMTL on `[0, tau]`."""
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    ut, sv = _extract_arm_knots(result)
    t_step, s_step = _step_arrays_up_to_tau(ut, sv, result.tau)

    ax.fill_between(
        t_step,
        0.0,
        s_step,
        step="post",
        color="#2563eb",
        alpha=0.20,
        label=f"RMST(\u03c4) = {result.rmst:.2f}",
    )
    ax.fill_between(
        t_step,
        s_step,
        1.0,
        step="post",
        color="#f59e0b",
        alpha=0.14,
        label=f"RMTL(\u03c4) = {result.rmtl:.2f}",
    )
    ax.step(t_step, s_step, where="post", color="#1d4ed8", linewidth=2.3, label="Kaplan\u2013Meier S(t)")

    if ut.size > 0 and float(ut[-1]) > result.tau:
        post_mask = ut >= result.tau
        s_at_tau = float(_eval_km_right_continuous(ut, sv, np.array([result.tau]))[0])
        t_post = np.concatenate(([result.tau], ut[post_mask]))
        s_post = np.concatenate(([s_at_tau], sv[post_mask]))
        ax.step(t_post, s_post, where="post", color="#1d4ed8", linestyle="--", linewidth=1.5, alpha=0.5)

    ax.axvline(
        result.tau,
        color="#334155",
        linestyle="--",
        linewidth=1.4,
        label=f"Horizon \u03c4 = {result.tau:.2g}",
    )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.0, max(result.tau * 1.08, float(ut[-1]) * 1.04 if ut.size else result.tau * 1.08))
    ax.set_xlabel("Follow-up Time", fontsize=9.5)
    ax.set_ylabel("Survival Probability", fontsize=9.5)
    banner = (
        title
        if title is not None
        else (
            f"Single-Arm RMST & RMTL (\u03c4 = {result.tau:.2g}, n = {result.n})  |  "
            f"RMST = {result.rmst:.2f} [{result.rmst_ci[0]:.2f}, {result.rmst_ci[1]:.2f}]"
        )
    )
    ax.set_title(banner, fontsize=10.2, fontweight="semibold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#e2e8f0", fontsize=8.5)
    return fig


def plot_rmst2(
    result: RMST2Result,
    time: Sequence[float] | np.ndarray | None = None,
    status: Sequence[int] | np.ndarray | None = None,
    arm: Sequence[int] | np.ndarray | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.5, 4.8),
    *,
    arm_labels: tuple[str, str] = ("Control (Arm 0)", "Active (Arm 1)"),
    title: str | None = None,
) -> Figure:
    """
    Plot two-arm Kaplan-Meier step survival curves shaded over $[0, \\tau]$
    illustrating $\\Delta\\text{RMST}(\\tau) = \\text{RMST}_1(\\tau) - \\text{RMST}_0(\\tau)$.

    Uses KM step curves stored directly inside `result.arm1` and `result.arm0`
    so `result.plot()` works immediately with zero extra arguments; if raw
    `(time, status, arm)` arrays are optionally supplied, full post-$\\tau$
    follow-up curves are also recomputed from them.

    Parameters
    ----------
    result : RMST2Result
        Fitted two-sample output returned by `rmst2` or `SurvRM2.fit`.
    time, status, arm : optional 1D array-like
        Optional raw survival data (not required; stored KM curves are used by default).
    ax : matplotlib.axes.Axes | None, default=None
        Optional target axes. If `None`, creates a new `(fig, ax)` with `figsize`.
    figsize : tuple[float, float], default=(7.5, 4.8)
        Figure size in inches when `ax is None`.
    arm_labels : tuple[str, str], default=("Control (Arm 0)", "Active (Arm 1)")
        Display labels for `(arm == 0, arm == 1)`.
    title : str | None, default=None
        Custom plot title override.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered two-sample RMST Kaplan-Meier figure.
    """
    _, plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if time is not None and status is not None and arm is not None:
        t_arr = np.asarray(time, dtype=np.float64)
        d_arr = np.asarray(status, dtype=np.int64)
        a_arr = np.asarray(arm, dtype=np.int64)
        ut0, sv0 = _km_curve_from_raw(t_arr[a_arr == 0], d_arr[a_arr == 0])
        ut1, sv1 = _km_curve_from_raw(t_arr[a_arr == 1], d_arr[a_arr == 1])
    else:
        ut0, sv0 = _extract_arm_knots(result.arm0)
        ut1, sv1 = _extract_arm_knots(result.arm1)

    tau = float(result.tau)
    t0_step, s0_step = _step_arrays_up_to_tau(ut0, sv0, tau)
    t1_step, s1_step = _step_arrays_up_to_tau(ut1, sv1, tau)

    # Unified step grid over [0, tau] for exact shaded area between S_1(t) and S_0(t)
    knots_union = np.unique(
        np.concatenate(
            [
                np.array([0.0, tau], dtype=np.float64),
                ut0[ut0 <= tau],
                ut1[ut1 <= tau],
            ]
        )
    )
    s0_u = _eval_km_right_continuous(ut0, sv0, knots_union)
    s1_u = _eval_km_right_continuous(ut1, sv1, knots_union)

    diff = result.rmst_diff_10
    shade_color = "#2563eb" if diff.est >= 0.0 else "#dc2626"
    ax.fill_between(
        knots_union,
        s0_u,
        s1_u,
        step="post",
        color=shade_color,
        alpha=0.18,
        zorder=2,
        label=f"\u0394RMST(\u03c4) = {diff.est:+.2f} (p = {diff.pval:.3g})",
    )

    col0 = "#ea580c"
    col1 = "#2563eb"

    ax.step(
        t1_step,
        s1_step,
        where="post",
        color=col1,
        linewidth=2.3,
        zorder=4,
        label=f"{arm_labels[1]} (RMST = {result.arm1.rmst:.2f})",
    )
    ax.step(
        t0_step,
        s0_step,
        where="post",
        color=col0,
        linewidth=2.3,
        zorder=4,
        label=f"{arm_labels[0]} (RMST = {result.arm0.rmst:.2f})",
    )

    # Post-tau dashed continuations if observations exist past tau
    max_obs_t = tau
    for ut_arm, sv_arm, col_arm in ((ut1, sv1, col1), (ut0, sv0, col0)):
        if ut_arm.size > 0 and float(ut_arm[-1]) > tau:
            max_obs_t = max(max_obs_t, float(ut_arm[-1]))
            post_m = ut_arm > tau
            s_tau = float(_eval_km_right_continuous(ut_arm, sv_arm, np.array([tau]))[0])
            t_p = np.concatenate(([tau], ut_arm[post_m]))
            s_p = np.concatenate(([s_tau], sv_arm[post_m]))
            ax.step(
                t_p,
                s_p,
                where="post",
                color=col_arm,
                linestyle="--",
                linewidth=1.5,
                alpha=0.45,
                zorder=3,
            )

    ax.axvline(
        tau,
        color="#334155",
        linestyle="--",
        linewidth=1.5,
        zorder=3,
        label=f"Horizon \u03c4 = {tau:.2g}",
    )

    # Compact statistical callout box
    pct_ci = int(round((1.0 - result.alpha) * 100))
    stat_lines = [
        f"\u0394RMST: {diff.est:+.2f} ({pct_ci}% CI [{diff.ci_lower:+.2f}, {diff.ci_upper:+.2f}])",
        f"RMST Ratio: {result.rmst_ratio_10.est:.2f}  |  p = {diff.pval:.3g}",
    ]
    if result.adjusted and result.adj_rmst_diff_10 is not None:
        adj_d = result.adj_rmst_diff_10
        stat_lines.append(
            f"Adj. \u0394RMST: {adj_d.est:+.2f} [{adj_d.ci_lower:+.2f}, {adj_d.ci_upper:+.2f}] (p={adj_d.pval:.3g})"
        )

    ax.text(
        0.03,
        0.06,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.93,
        },
        zorder=6,
    )

    ax.set_xlim(0.0, max(tau * 1.08, max_obs_t * 1.03))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Follow-up Time", fontsize=9.5)
    ax.set_ylabel("Kaplan\u2013Meier Survival Probability", fontsize=9.5)

    banner = (
        title
        if title is not None
        else (
            f"Two-Sample RMST Comparison (\u03c4 = {tau:.2g})  |  "
            f"n\u2081 = {result.arm1.n}, n\u2080 = {result.arm0.n}"
        )
    )
    ax.set_title(banner, fontsize=10.3, fontweight="semibold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linestyle=":", linewidth=0.8, zorder=0)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=8.2,
    )
    return fig
