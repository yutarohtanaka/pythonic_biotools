"""Complementary-Pairs Randomized Lasso Stability Selection (monaLisa)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from ._validation import SettingsMode, validate_settings
from .composition import calc_sequence_composition
from .motifs import MotifScanResult


def _calibrate_q_target(
    n_features: int,
    n_samples: int,
    cutoff: float,
    pfer: float,
    n_subsamples: int,
) -> float:
    """Calibrate expected active-set size ``q`` via Meinshausen-Buhlmann / Shah-Samworth PFER bound."""
    _ = n_subsamples
    p = float(n_features)
    q_mb = float(np.sqrt(pfer * (2.0 * cutoff - 1.0) * p))
    half_n = max(2, n_samples // 2)
    q_cap = min(p, max(1.0, 0.75 * half_n))
    return float(np.clip(q_mb, 1.0, q_cap))


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Zero-center and unit-scale columns safely (constant columns become 0.0)."""
    xc = x - np.mean(x, axis=0, keepdims=True)
    stds = np.std(xc, axis=0)
    out = np.zeros_like(xc, dtype=np.float64)
    active = stds > 1e-12
    if np.any(active):
        out[:, active] = xc[:, active] / stds[np.newaxis, active]
    return out


@dataclass
class StabilitySelectionResult:
    """Result container for ``rand_lasso_stab_sel``."""

    sel_prob: pd.Series
    prob_path_df: pd.DataFrame
    selected_motifs: list[str]
    selected_features: list[str]
    motif_names: list[str]
    alphas: np.ndarray
    cutoff: float
    pfer: float
    q_target: float
    weakness: float
    n_subsamples: int
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary DataFrame sorted by descending ``sel_prob``."""
        motif_set = set(self.motif_names)
        df = pd.DataFrame(
            {
                "feature": list(self.sel_prob.index),
                "is_motif": [f in motif_set for f in self.sel_prob.index],
                "sel_prob": self.sel_prob.to_numpy(),
                "selected": (self.sel_prob.to_numpy() >= self.cutoff),
            },
            index=self.sel_prob.index,
        )
        return df.sort_values(by="sel_prob", ascending=False, kind="mergesort")

    def plot(
        self,
        top_n: int = 10,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8.0, 4.5),
        *,
        title: str | None = None,
    ) -> Figure:
        """Plot Randomized Lasso selection probability trajectories (`monalisa.plot_stability_selection`)."""
        from .plotting import plot_stability_selection

        return plot_stability_selection(
            self,
            top_n=top_n,
            ax=ax,
            figsize=figsize,
            title=title,
        )


def rand_lasso_stab_sel(
    X_motifs: pd.DataFrame | np.ndarray | MotifScanResult,
    y_change: np.ndarray | pd.Series | Sequence[float],
    sequences: Sequence[str] | pd.Series | np.ndarray | None = None,
    include_seq_bias: bool = True,
    cutoff: float = 0.8,
    pfer: float = 1.0,
    n_subsamples: int = 100,
    weakness: float | None = None,
    n_alphas: int = 25,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> StabilitySelectionResult:
    """Complementary-Pairs Randomized Lasso Stability Selection for motif discovery.

    Implements ``monaLisa::randLassoStabSel`` (Machlab et al. 2022; Meinshausen
    & Bühlmann 2010; Shah & Samworth 2013):
    - ``settings="default"``: ``weakness=0.5``.
    - ``settings="R_default"``: ``weakness=0.8``, ``cutoff=0.8`` upstream parity.
    """
    validate_settings(settings)

    w_min = (
        (0.8 if settings == "R_default" else 0.5)
        if weakness is None
        else float(weakness)
    )
    if not (0.0 < w_min <= 1.0):
        raise ValueError(f"weakness must lie in (0, 1], got {w_min}.")
    if not (0.5 < cutoff <= 1.0):
        raise ValueError(f"cutoff must lie in (0.5, 1.0], got {cutoff}.")
    if pfer <= 0.0:
        raise ValueError(f"pfer must be > 0, got {pfer}.")
    if n_subsamples < 1:
        raise ValueError(f"n_subsamples must be >= 1, got {n_subsamples}.")
    if n_alphas < 2:
        raise ValueError(f"n_alphas must be >= 2, got {n_alphas}.")

    if isinstance(X_motifs, MotifScanResult):
        x_df_or_arr: pd.DataFrame | np.ndarray = X_motifs.counts
    else:
        x_df_or_arr = X_motifs

    if isinstance(x_df_or_arr, pd.DataFrame):
        motif_names = [str(c) for c in x_df_or_arr.columns]
        x_mat = x_df_or_arr.to_numpy(dtype=np.float64, copy=True)
    else:
        x_mat = np.asarray(x_df_or_arr, dtype=np.float64).copy()
        if x_mat.ndim != 2 or x_mat.size == 0:
            raise ValueError("X_motifs must be a non-empty 2D matrix or DataFrame.")
        motif_names = [f"motif_{j}" for j in range(x_mat.shape[1])]

    y = np.asarray(y_change, dtype=np.float64)
    n_samples, n_motifs = x_mat.shape
    if y.ndim != 1 or y.shape[0] != n_samples:
        raise ValueError(
            f"y_change length ({y.shape}) must match X_motifs rows ({n_samples})."
        )
    if n_samples < 4:
        raise ValueError("At least 4 regions are required for half-sample splitting.")
    if not np.all(np.isfinite(x_mat)) or not np.all(np.isfinite(y)):
        raise ValueError("X_motifs and y_change must contain only finite entries.")

    feature_names = list(motif_names)
    if sequences is not None and include_seq_bias:
        seq_list = [str(s) for s in sequences]
        if len(seq_list) != n_samples:
            raise ValueError(
                f"sequences length ({len(seq_list)}) must match X_motifs rows ({n_samples})."
            )
        comp = calc_sequence_composition(seq_list, settings=settings)
        gc_name = "gc_frac" if "gc_frac" not in feature_names else "_seq_bias_gc_frac"
        cpg_name = "cpg_oe" if "cpg_oe" not in feature_names else "_seq_bias_cpg_oe"
        bias_mat = comp[["gc_frac", "cpg_oe"]].to_numpy(dtype=np.float64)
        x_full = np.column_stack([x_mat, bias_mat])
        feature_names.extend([gc_name, cpg_name])
    else:
        x_full = x_mat

    n_features = x_full.shape[1]
    q_target = _calibrate_q_target(
        n_features=n_features,
        n_samples=n_samples,
        cutoff=cutoff,
        pfer=pfer,
        n_subsamples=n_subsamples,
    )

    x_std = _standardize_columns(x_full)
    y_centered = y - float(np.mean(y))
    alpha_max = float(np.max(np.abs(x_std.T @ y_centered)) / n_samples)

    step_cols = [f"step_{k}" for k in range(n_alphas)]
    if alpha_max <= 1e-15 or float(np.std(y_centered)) <= 1e-15:
        zero_series = pd.Series(
            np.zeros(n_features, dtype=np.float64),
            index=pd.Index(feature_names, name="feature"),
            name="sel_prob",
        )
        zero_df = pd.DataFrame(
            np.zeros((n_features, n_alphas), dtype=np.float64),
            index=pd.Index(feature_names, name="feature"),
            columns=step_cols,
        )
        return StabilitySelectionResult(
            sel_prob=zero_series,
            prob_path_df=zero_df,
            selected_motifs=[],
            selected_features=[],
            motif_names=motif_names,
            alphas=np.zeros(n_alphas, dtype=np.float64),
            cutoff=cutoff,
            pfer=pfer,
            q_target=q_target,
            weakness=w_min,
            n_subsamples=n_subsamples,
            settings=settings,
        )

    rng = np.random.default_rng(seed)
    half_n = n_samples // 2

    # Quick 4-half-sample pilot to calibrate alpha_min so path resolves [0, q_target] cleanly
    pilot_grid = np.geomspace(alpha_max * 1.02, alpha_max * 0.01, 40)
    pilot_cum_q = np.zeros(40, dtype=np.float64)
    for _ in range(4):
        p_idx = rng.choice(n_samples, size=half_n, replace=False)
        xp = _standardize_columns(x_full[p_idx])
        yp = y[p_idx] - float(np.mean(y[p_idx]))
        wp = rng.uniform(w_min, 1.0, size=n_features)
        if float(np.std(yp)) > 1e-15 and float(np.max(np.abs(xp))) > 0.0:
            _, p_coefs, _ = lasso_path(
                xp * wp[np.newaxis, :], yp, alphas=pilot_grid, max_iter=3000, tol=1e-3
            )
            p_act = np.maximum.accumulate(np.abs(p_coefs) > 1e-12, axis=1)
            pilot_cum_q += np.sum(p_act, axis=0) / 4.0

    reached = np.flatnonzero(pilot_cum_q >= q_target)
    if reached.size > 0:
        stop_idx = min(pilot_grid.size - 1, int(reached[0]) + 1)
        alpha_min = float(pilot_grid[stop_idx])
    else:
        alpha_min = float(alpha_max * 0.02)
    alpha_min = min(alpha_min, alpha_max * 0.5)

    alphas = np.geomspace(alpha_max * 1.02, alpha_min, n_alphas)

    # Complementary-pairs Randomized Lasso across 2 * n_subsamples half-samples
    sel_counts = np.zeros((n_features, n_alphas), dtype=np.float64)
    cum_sel_counts = np.zeros(n_alphas, dtype=np.float64)
    total_fits = 2 * n_subsamples

    for _ in range(n_subsamples):
        perm = rng.permutation(n_samples)
        halves = (perm[:half_n], perm[half_n : 2 * half_n])
        for sub_idx in halves:
            xs = _standardize_columns(x_full[sub_idx])
            ys = y[sub_idx] - float(np.mean(y[sub_idx]))
            w_rand = rng.uniform(w_min, 1.0, size=n_features)
            if float(np.std(ys)) <= 1e-15 or float(np.max(np.abs(xs))) <= 0.0:
                continue
            _, coefs, _ = lasso_path(
                xs * w_rand[np.newaxis, :],
                ys,
                alphas=alphas,
                max_iter=5000,
                tol=1e-3,
            )
            nonzero = np.abs(coefs) > 1e-12
            sel_counts += nonzero.astype(np.float64)
            cum_nonzero = np.maximum.accumulate(nonzero, axis=1)
            cum_sel_counts += np.sum(cum_nonzero, axis=0)

    prob_path = sel_counts / total_fits
    q_along_path = cum_sel_counts / total_fits

    # Enforce Meinshausen-Buhlmann expected active-set bound q <= q_target
    allowed_steps = np.flatnonzero(q_along_path <= q_target)
    if allowed_steps.size == 0:
        valid_end = 1
    else:
        valid_end = int(allowed_steps[-1]) + 1

    max_sel_prob = np.max(prob_path[:, :valid_end], axis=1)

    feat_index = pd.Index(feature_names, name="feature")
    sel_prob_series = pd.Series(max_sel_prob, index=feat_index, name="sel_prob")
    prob_path_df = pd.DataFrame(prob_path, index=feat_index, columns=step_cols)

    # Sort selected features/motifs by descending selection probability
    selected_order = np.argsort(-max_sel_prob, kind="mergesort")
    motif_set = set(motif_names)
    selected_features = [
        feature_names[int(i)]
        for i in selected_order
        if max_sel_prob[int(i)] >= cutoff
    ]
    selected_motifs = [f for f in selected_features if f in motif_set]

    return StabilitySelectionResult(
        sel_prob=sel_prob_series,
        prob_path_df=prob_path_df,
        selected_motifs=selected_motifs,
        selected_features=selected_features,
        motif_names=motif_names,
        alphas=alphas,
        cutoff=cutoff,
        pfer=pfer,
        q_target=q_target,
        weakness=w_min,
        n_subsamples=n_subsamples,
        settings=settings,
    )
