"""Partitioning continuous genomic effect sizes into ordered monaLisa bins."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._validation import SettingsMode, validate_settings


def _compute_edges_1d(
    sub_vals: np.ndarray, n_sub_bins: int, method: str
) -> np.ndarray:
    """Compute strictly increasing bin edges for a 1D subset of values."""
    v_min = float(np.min(sub_vals))
    v_max = float(np.max(sub_vals))
    if n_sub_bins <= 1 or v_max <= v_min:
        return np.array([v_min, v_max], dtype=np.float64)

    if method == "equalN":
        probs = np.linspace(0.0, 1.0, n_sub_bins + 1)
        raw_edges = np.quantile(sub_vals, probs)
    else:  # equalWidth
        raw_edges = np.linspace(v_min, v_max, n_sub_bins + 1)

    edges = np.unique(raw_edges)
    if edges.size < 2:
        return np.array([v_min, v_max], dtype=np.float64)
    return edges


def _assign_tail_bins(
    sub_vals: np.ndarray, edges: np.ndarray, is_negative_tail: bool
) -> tuple[list[str], list[str]]:
    """Assign each value in ``sub_vals`` to formatted interval labels."""
    n_intervals = edges.size - 1
    if n_intervals <= 1:
        lo, hi = float(edges[0]), float(edges[-1])
        lbl = f"[{lo:.3g},{hi:.3g})" if is_negative_tail else f"({lo:.3g},{hi:.3g}]"
        return [lbl] * sub_vals.size, [lbl]

    cat_names: list[str] = []
    for k in range(n_intervals):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if is_negative_tail:
            left_br = "["
            right_br = ")" if k < n_intervals - 1 else "]"
        else:
            left_br = "(" if k == 0 else "["
            right_br = "]" if k == n_intervals - 1 else ")"
        cat_names.append(f"{left_br}{lo:.3g},{hi:.3g}{right_br}")

    # Map values to 0 .. n_intervals - 1
    idx = np.searchsorted(edges[1:-1], sub_vals, side="right")
    idx = np.clip(idx, 0, n_intervals - 1)
    assigned = [cat_names[int(i)] for i in idx]
    return assigned, cat_names


def bin_genomic_changes(
    values: np.ndarray | pd.Series | Sequence[float],
    method: str = "equalN",
    n_bins: int = 5,
    min_abs_val: float = 0.0,
    zero_bin: bool = True,
    cutpoints: Sequence[float] | np.ndarray | None = None,
    settings: SettingsMode = "default",
) -> pd.Categorical:
    """Partition continuous genomic region effect sizes into ordered bins.

    Implements ``monaLisa::bin`` (Machlab et al. 2022) supporting ``"equalN"``
    (quantile), ``"equalWidth"``, and ``"cutpoints"`` binning with an optional
    neutral ``"zero"`` bin centered around ``0``.
    """
    validate_settings(settings)

    valid_methods = ("equalN", "equalWidth", "cutpoints")
    if cutpoints is not None and method != "cutpoints":
        method = "cutpoints"
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method={method!r}. Expected one of {valid_methods}."
        )

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D numeric sequence.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite numeric entries.")
    if min_abs_val < 0.0:
        raise ValueError(f"min_abs_val must be >= 0, got {min_abs_val}.")

    if method == "cutpoints":
        if cutpoints is None:
            raise ValueError("cutpoints must be provided when method='cutpoints'.")
        cp = np.asarray(cutpoints, dtype=np.float64)
        if cp.ndim != 1 or cp.size < 2 or not np.all(np.diff(cp) > 0):
            raise ValueError(
                "cutpoints must be a strictly increasing 1D sequence of >= 2 breaks."
            )
        cut_res = pd.cut(arr, bins=cp, include_lowest=True)
        assert isinstance(cut_res, pd.Categorical)
        str_cats = [str(c) for c in cut_res.categories]
        str_vals: list[str | float] = [
            str(v) if pd.notna(v) else np.nan for v in cut_res
        ]
        if zero_bin and min_abs_val > 0.0:
            z_mask = np.abs(arr) <= min_abs_val
            for idx_z in np.flatnonzero(z_mask):
                str_vals[int(idx_z)] = "zero"
            if "zero" not in str_cats:
                # Insert "zero" between negative and positive intervals
                mid_pos = int(np.searchsorted(cp[:-1], 0.0))
                mid_pos = min(max(mid_pos, 0), len(str_cats))
                str_cats.insert(mid_pos, "zero")
        return pd.Categorical(str_vals, categories=str_cats, ordered=True)

    if n_bins < 1 or (zero_bin and n_bins < 2):
        raise ValueError(
            f"Invalid n_bins={n_bins} (must be >= 2 when zero_bin=True, else >= 1)."
        )

    if not zero_bin:
        active_mask = np.abs(arr) >= min_abs_val if min_abs_val > 0.0 else np.ones(arr.size, dtype=bool)
        if not np.any(active_mask):
            raise ValueError("No values satisfy |value| >= min_abs_val.")
        edges = _compute_edges_1d(arr[active_mask], n_bins, method)
        assigned_sub, cats = _assign_tail_bins(arr[active_mask], edges, is_negative_tail=False)
        out_labels: list[str | float] = [np.nan] * arr.size
        for pos_i, lbl in zip(np.flatnonzero(active_mask), assigned_sub):
            out_labels[int(pos_i)] = lbl
        return pd.Categorical(out_labels, categories=cats, ordered=True)

    # zero_bin == True
    if min_abs_val > 0.0:
        c0 = float(min_abs_val)
    else:
        if np.all(arr == 0.0):
            return pd.Categorical(["zero"] * arr.size, categories=["zero"], ordered=True)
        c0 = float(np.quantile(np.abs(arr), 1.0 / n_bins))

    z_mask = np.abs(arr) <= c0
    neg_mask = arr < -c0
    pos_mask = arr > c0

    n_neg_pts = int(np.sum(neg_mask))
    n_pos_pts = int(np.sum(pos_mask))
    k_nz = n_bins - 1

    if n_neg_pts > 0 and n_pos_pts > 0:
        if k_nz == 1:
            # With n_bins == 2 and symmetric tails, collapse non-zero into single tail or 1 neg + 1 pos
            k_neg, k_pos = 1, 1
        else:
            if method == "equalN":
                frac_neg = n_neg_pts / (n_neg_pts + n_pos_pts)
            else:
                span_neg = max(-c0 - float(np.min(arr[neg_mask])), 1e-12)
                span_pos = max(float(np.max(arr[pos_mask])) - c0, 1e-12)
                frac_neg = span_neg / (span_neg + span_pos)
            k_neg = max(1, min(k_nz - 1, int(np.rint(k_nz * frac_neg))))
            k_pos = k_nz - k_neg
    elif n_neg_pts > 0:
        k_neg, k_pos = k_nz, 0
    elif n_pos_pts > 0:
        k_neg, k_pos = 0, k_nz
    else:
        return pd.Categorical(["zero"] * arr.size, categories=["zero"], ordered=True)

    full_labels = [""] * arr.size
    for idx_z in np.flatnonzero(z_mask):
        full_labels[int(idx_z)] = "zero"

    ordered_cats: list[str] = []
    if n_neg_pts > 0 and k_neg > 0:
        neg_edges = _compute_edges_1d(arr[neg_mask], k_neg, method)
        neg_edges[-1] = max(neg_edges[-1], -c0)
        neg_assigned, neg_cats = _assign_tail_bins(
            arr[neg_mask], neg_edges, is_negative_tail=True
        )
        for idx_n, lbl in zip(np.flatnonzero(neg_mask), neg_assigned):
            full_labels[int(idx_n)] = lbl
        ordered_cats.extend(neg_cats)

    if np.any(z_mask):
        ordered_cats.append("zero")

    if n_pos_pts > 0 and k_pos > 0:
        pos_edges = _compute_edges_1d(arr[pos_mask], k_pos, method)
        pos_edges[0] = min(pos_edges[0], c0)
        pos_assigned, pos_cats = _assign_tail_bins(
            arr[pos_mask], pos_edges, is_negative_tail=False
        )
        for idx_p, lbl in zip(np.flatnonzero(pos_mask), pos_assigned):
            full_labels[int(idx_p)] = lbl
        ordered_cats.extend(pos_cats)

    return pd.Categorical(full_labels, categories=ordered_cats, ordered=True)
