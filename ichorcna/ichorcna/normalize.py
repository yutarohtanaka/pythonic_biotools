"""
Two-stage GC + mappability LOESS read-depth normalization, centromere filtering,
and Panel-of-Normals (`PoN`) log2 ratio correction for ultra-low-pass WGS bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .io import SettingsMode, canonical_chrom, chrom_sort_key, validate_settings


def lowess_1d(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.25,
    n_anchors: int = 128,
) -> np.ndarray:
    """
    Fast deterministic 1D tricube-weighted local linear regression (LOWESS)
    evaluated on anchor quantiles and linearly interpolated across `x`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = x_arr.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n < 4 or np.ptp(x_arr) < 1e-12:
        return np.full(n, float(np.median(y_arr)), dtype=np.float64)

    span = max(3, min(n, int(np.ceil(frac * n))))
    order = np.argsort(x_arr, kind="mergesort")
    xs = x_arr[order]
    ys = y_arr[order]

    m = min(n, n_anchors)
    anchor_idx = np.unique(np.linspace(0, n - 1, m, dtype=np.int64))
    anchors = xs[anchor_idx]
    anchor_preds = np.empty(anchors.size, dtype=np.float64)

    for j, x0 in enumerate(anchors):
        dists = np.abs(xs - x0)
        # Distance to the span-th nearest neighbor
        kth = np.partition(dists, span - 1)[span - 1]
        h = max(float(kth), 1e-9)
        u = dists / h
        mask = u < 1.0
        if np.sum(mask) < 2:
            anchor_preds[j] = float(np.median(ys))
            continue
        w = (1.0 - (u[mask] ** 3)) ** 3
        xm = xs[mask] - x0
        ym = ys[mask]
        sw = float(np.sum(w))
        swx = float(np.sum(w * xm))
        swxx = float(np.sum(w * xm * xm))
        swy = float(np.sum(w * ym))
        swxy = float(np.sum(w * xm * ym))
        det = sw * swxx - swx * swx
        if det > 1e-12 * sw * max(swxx, 1.0):
            b0 = (swxx * swy - swx * swxy) / det
        else:
            b0 = swy / max(sw, 1e-12)
        anchor_preds[j] = b0

    # Deduplicate any identical anchor coordinates before interpolation
    uniq_anchors, uniq_idx = np.unique(anchors, return_index=True)
    uniq_preds = anchor_preds[uniq_idx]
    if uniq_anchors.size == 1:
        return np.full(n, float(uniq_preds[0]), dtype=np.float64)
    return np.interp(x_arr, uniq_anchors, uniq_preds)


def _subtract_panel_of_normals(
    df: pd.DataFrame,
    panel_of_normals: pd.DataFrame | pd.Series | np.ndarray,
) -> np.ndarray:
    """Subtract Panel-of-Normals median logR per genomic bin."""
    logr = df["logR"].to_numpy(dtype=np.float64).copy()
    if isinstance(panel_of_normals, (np.ndarray, pd.Series)):
        pon_vec = np.asarray(panel_of_normals, dtype=np.float64).ravel()
        if pon_vec.size != len(df):
            raise ValueError(
                f"1D panel_of_normals length ({pon_vec.size}) must match valid bins ({len(df)})."
            )
        valid_p = np.isfinite(pon_vec)
        logr[valid_p] = logr[valid_p] - pon_vec[valid_p]
        return logr

    if not isinstance(panel_of_normals, pd.DataFrame) or panel_of_normals.empty:
        raise ValueError("panel_of_normals DataFrame must be non-empty.")

    pon = panel_of_normals.copy()
    if "chrom" not in pon.columns and "chr" in pon.columns:
        pon = pon.rename(columns={"chr": "chrom"})
    if "chrom" not in pon.columns or "start" not in pon.columns:
        raise ValueError("panel_of_normals DataFrame must contain 'chrom' and 'start'.")

    pon["chrom_canon"] = [canonical_chrom(c) for c in pon["chrom"]]
    pon["start"] = pon["start"].astype(np.int64)

    meta_cols = {"chrom", "chr", "chrom_canon", "start", "end", "gc", "map", "mappability"}
    if "pon_median" in pon.columns:
        pon_med = pon["pon_median"].astype(np.float64).to_numpy()
    elif "median_logR" in pon.columns:
        pon_med = pon["median_logR"].astype(np.float64).to_numpy()
    elif "logR" in pon.columns:
        pon_med = pon["logR"].astype(np.float64).to_numpy()
    else:
        num_cols = [
            c
            for c in pon.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(pon[c])
        ]
        if not num_cols:
            raise ValueError(
                "panel_of_normals DataFrame must contain 'pon_median', 'median_logR', 'logR', "
                "or numeric normal-sample logR columns."
            )
        mat = pon[num_cols].to_numpy(dtype=np.float64)
        pon_med = np.nanmedian(mat, axis=1)

    pon_lookup = pd.DataFrame(
        {"chrom_canon": pon["chrom_canon"], "start": pon["start"], "_pon_med": pon_med}
    ).drop_duplicates(subset=["chrom_canon", "start"])

    merged = df[["chrom_canon", "start"]].merge(
        pon_lookup, on=["chrom_canon", "start"], how="left"
    )
    med_vals = merged["_pon_med"].to_numpy(dtype=np.float64)
    has_med = np.isfinite(med_vals)
    logr[has_med] = logr[has_med] - med_vals[has_med]
    return logr


def normalize_ulp_bins(
    bins_df: pd.DataFrame,
    panel_of_normals: pd.DataFrame | pd.Series | np.ndarray | None = None,
    centromeres_df: pd.DataFrame | None = None,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Perform two-stage GC + mappability LOESS copy-ratio normalization (`HMMcopy` /
    `ichorCNA` workflow), centromere exclusion, and optional Panel-of-Normals
    (`PoN`) median subtraction.

    Parameters
    ----------
    bins_df : pd.DataFrame
        Genomic windows table with `chrom`, `start`, `end`, and either `reads` + `gc`
        (plus optional `map` / `mappability`) or pre-computed `logR`.
    panel_of_normals : pd.DataFrame | pd.Series | np.ndarray | None, default=None
        Optional Panel of Normals (`chrom`, `start`, and `pon_median` / multiple
        normal sample `logR` columns) subtracted from normalized `logR`.
    centromeres_df : pd.DataFrame | None, default=None
        Optional centromere coordinates (`chrom`, `start`, `end`); overlapping bins
        are excluded prior to LOESS fitting and HMM segmentation.
    settings : {"default", "R_default"}, default="default"
        `"default"` uses `min_map = 0.85`, 2-stage LOESS (`frac=0.25` GC, `0.35` map),
        and 5%-trimmed mean baseline centering; `"R_default"` uses `ichorCNA` /
        `HMMcopy` defaults (`min_map = 0.90`, median baseline centering).

    Returns
    -------
    pd.DataFrame
        Sorted, filtered genomic bins DataFrame containing normalized `logR`.
    """
    validate_settings(settings)
    if not isinstance(bins_df, pd.DataFrame) or bins_df.empty:
        raise ValueError("bins_df must be a non-empty pandas DataFrame.")

    df = bins_df.copy()
    if "chrom" not in df.columns:
        if "chr" in df.columns:
            df = df.rename(columns={"chr": "chrom"})
        elif "chromosome" in df.columns:
            df = df.rename(columns={"chromosome": "chrom"})

    for req in ("chrom", "start", "end"):
        if req not in df.columns:
            raise ValueError(f"bins_df is missing required column {req!r}.")

    df["chrom_canon"] = [canonical_chrom(c) for c in df["chrom"]]
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)

    # Sort naturally by chromosome then genomic coordinate
    sort_keys = [chrom_sort_key(c) for c in df["chrom_canon"]]
    df["_c_group"] = [k[0] for k in sort_keys]
    df["_c_label"] = [k[1] for k in sort_keys]
    df = df.sort_values(["_c_group", "_c_label", "start", "end"], kind="mergesort")
    df = df.drop(columns=["_c_group", "_c_label"]).reset_index(drop=True)

    valid_mask = ~df["chrom_canon"].isin(["Y", "MT"])
    if not bool(np.any(valid_mask)):
        valid_mask = np.ones(len(df), dtype=bool)

    # Filter centromere-overlapping bins
    if centromeres_df is not None and not centromeres_df.empty:
        cen = centromeres_df.copy()
        if "chrom" not in cen.columns and "chr" in cen.columns:
            cen = cen.rename(columns={"chr": "chrom"})
        cen["chrom_canon"] = [canonical_chrom(c) for c in cen["chrom"]]
        c_starts = cen["start"].astype(np.int64).to_numpy()
        c_ends = cen["end"].astype(np.int64).to_numpy()
        c_chroms = cen["chrom_canon"].to_numpy()

        b_starts = df["start"].to_numpy(dtype=np.int64)
        b_ends = df["end"].to_numpy(dtype=np.int64)
        b_chroms = df["chrom_canon"].to_numpy()

        in_cen = np.zeros(len(df), dtype=bool)
        for j in range(len(cen)):
            in_cen |= (
                (b_chroms == c_chroms[j])
                & (b_starts <= c_ends[j])
                & (b_ends >= c_starts[j])
            )
        valid_mask &= ~in_cen

    # Mappability column standardization
    if "map" not in df.columns and "mappability" in df.columns:
        df["map"] = df["mappability"]
    has_map = "map" in df.columns
    if has_map:
        map_vals = df["map"].astype(np.float64).to_numpy()
        min_map = 0.85 if settings == "default" else 0.90
        valid_mask &= np.isfinite(map_vals) & (map_vals >= min_map)

    # Read count column standardization
    read_col = None
    for cand in ("reads", "read_count", "count", "coverage"):
        if cand in df.columns:
            read_col = cand
            break

    if read_col is not None:
        if "gc" not in df.columns:
            raise ValueError("bins_df with read counts must include a 'gc' column.")
        reads = df[read_col].astype(np.float64).to_numpy()
        gc = df["gc"].astype(np.float64).to_numpy()
        if np.nanmax(gc) > 1.05:
            gc = gc / 100.0
            df["gc"] = gc
        valid_mask &= np.isfinite(reads) & (reads > 0.0) & np.isfinite(gc) & (gc > 0.0)

        df = df.loc[valid_mask].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid genomic bins remained after centromere/mappability/read filtering.")

        r_val = df[read_col].astype(np.float64).to_numpy()
        gc_val = df["gc"].astype(np.float64).to_numpy()

        # Trim extreme 1% tails when fitting GC LOESS so focal high-level amplifications do not warp curve
        q_lo, q_hi = np.quantile(r_val, [0.01, 0.99])
        fit_sub = (r_val >= q_lo) & (r_val <= q_hi)
        if np.sum(fit_sub) < 10:
            fit_sub = np.ones_like(r_val, dtype=bool)

        gc_frac = 0.25 if settings == "default" else 0.20
        gc_curve_sub = lowess_1d(gc_val[fit_sub], r_val[fit_sub], frac=gc_frac)
        order_sub = np.argsort(gc_val[fit_sub], kind="mergesort")
        ux, uidx = np.unique(gc_val[fit_sub][order_sub], return_index=True)
        uy = gc_curve_sub[order_sub][uidx]
        gc_pred = (
            np.interp(gc_val, ux, uy)
            if ux.size >= 2
            else np.full_like(r_val, float(np.median(r_val)))
        )
        gc_pred = np.maximum(gc_pred, 1e-3)
        cor_gc = r_val / gc_pred

        # Stage 2: Mappability LOESS correction
        if has_map:
            m_val = df["map"].astype(np.float64).to_numpy()
            if np.ptp(m_val) > 1e-6:
                mq_lo, mq_hi = np.quantile(cor_gc, [0.01, 0.99])
                m_sub = (cor_gc >= mq_lo) & (cor_gc <= mq_hi)
                if np.sum(m_sub) < 10:
                    m_sub = np.ones_like(cor_gc, dtype=bool)
                map_curve_sub = lowess_1d(m_val[m_sub], cor_gc[m_sub], frac=0.35)
                m_ord = np.argsort(m_val[m_sub], kind="mergesort")
                mux, muidx = np.unique(m_val[m_sub][m_ord], return_index=True)
                muy = map_curve_sub[m_ord][muidx]
                map_pred = (
                    np.interp(m_val, mux, muy)
                    if mux.size >= 2
                    else np.full_like(cor_gc, float(np.median(cor_gc)))
                )
                cor_map = cor_gc / np.maximum(map_pred, 1e-3)
            else:
                cor_map = cor_gc
        else:
            cor_map = cor_gc

        if settings == "default":
            baseline = float(stats.trim_mean(cor_map, proportiontocut=0.05))
        else:
            baseline = float(np.median(cor_map))
        baseline = max(baseline, 1e-6)
        df["cor_gc"] = cor_gc
        df["cor_map"] = cor_map
        df["logR"] = np.log2(np.maximum(cor_map / baseline, 1e-6))
    else:
        if "logR" not in df.columns:
            raise ValueError("bins_df must contain either read counts ('reads' + 'gc') or 'logR'.")
        valid_mask &= np.isfinite(df["logR"].astype(np.float64).to_numpy())
        df = df.loc[valid_mask].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid genomic bins remained after filtering.")
        df["logR"] = df["logR"].astype(np.float64)

    if panel_of_normals is not None:
        df["logR"] = _subtract_panel_of_normals(df, panel_of_normals)
        if settings == "R_default":
            df["logR"] = df["logR"] - float(np.median(df["logR"].to_numpy(dtype=np.float64)))

    return df.drop(columns=["chrom_canon"])
