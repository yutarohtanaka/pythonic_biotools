"""
Simultaneous 2D (GC, mappability) locally weighted regression (`estimate_correction`),
bias correction (`correct_counts`), normalization (`normalize_bins`), and singleton
spike smoothing (`smooth_outlier_bins`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .config import SettingsMode, get_qdnaseq_config, validate_settings
from .data import QDNAseqData


def _eval_local_poly_2d(
    u_train: np.ndarray,
    v_train: np.ndarray,
    y_train: np.ndarray,
    u_eval: np.ndarray,
    v_eval: np.ndarray,
    *,
    span: float = 0.65,
    is_2d: bool = True,
) -> np.ndarray:
    """
    Evaluate tricube locally weighted quadratic regression at `(u_eval, v_eval)`
    with small ridge stabilization on slope/curvature terms.
    """
    n_tr = y_train.size
    if n_tr == 0:
        return np.ones_like(u_eval, dtype=np.float64)
    if n_tr < 8:
        return np.full_like(u_eval, float(np.median(y_train)), dtype=np.float64)

    k = max(8, min(n_tr, int(np.ceil(span * n_tr))))
    preds = np.empty(u_eval.size, dtype=np.float64)

    for idx in range(u_eval.size):
        u0 = float(u_eval[idx])
        v0 = float(v_eval[idx])
        du = u_train - u0
        dv = (v_train - v0) if is_2d else np.zeros_like(du)
        dists = np.hypot(du, dv) if is_2d else np.abs(du)

        if k < n_tr:
            part = np.argpartition(dists, k - 1)[:k]
        else:
            part = np.arange(n_tr)

        d_sub = dists[part]
        h = float(np.max(d_sub))
        if h <= 1e-10:
            preds[idx] = float(np.median(y_train[part]))
            continue

        q = np.clip(d_sub / (h * 1.001), 0.0, 1.0)
        w = (1.0 - q**3) ** 3
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            preds[idx] = float(np.median(y_train[part]))
            continue

        du_s = du[part]
        y_s = y_train[part]
        if is_2d:
            dv_s = dv[part]
            x_des = np.column_stack(
                [np.ones_like(du_s), du_s, dv_s, du_s * du_s, du_s * dv_s, dv_s * dv_s]
            )
        else:
            x_des = np.column_stack([np.ones_like(du_s), du_s, du_s * du_s])

        xw = x_des * w[:, np.newaxis]
        xtx = xw.T @ x_des
        # Tiny ridge penalty on non-intercept coefficients prevents corner extrapolation spikes
        ridge = np.diag([0.0] + [1e-3 * w_sum] * (x_des.shape[1] - 1))
        rhs = xw.T @ y_s
        try:
            beta = np.linalg.solve(xtx + ridge, rhs)
            val = float(beta[0])
        except np.linalg.LinAlgError:
            val = float(np.sum(w * y_s) / w_sum)
        preds[idx] = val

    return preds


def fit_2d_loess_surface(
    gc: np.ndarray,
    mappability: np.ndarray,
    counts: np.ndarray,
    use_mask: np.ndarray,
    *,
    max_iter: int = 3,
    cutoff: float = 4.0,
    span: float = 0.65,
) -> np.ndarray:
    """
    Simultaneous 2D `(GC, mappability)` LOESS surface estimator with iterative
    `|z| > cutoff` outlier masking matching Scheinin et al. (2014) `estimateCorrection`.
    """
    n_all = counts.size
    active = (
        use_mask
        & np.isfinite(gc)
        & np.isfinite(mappability)
        & np.isfinite(counts)
    )
    if int(np.sum(active)) < 6:
        med = float(np.nanmedian(counts[counts > 0])) if np.any(counts > 0) else 1.0
        return np.full(n_all, max(med, 1e-3), dtype=np.float64)

    g_act = gc[active]
    m_act = mappability[active]
    g_mean, g_sd = float(np.mean(g_act)), float(np.std(g_act))
    m_mean, m_sd = float(np.mean(m_act)), float(np.std(m_act))

    has_gc_var = g_sd > 1e-6
    has_map_var = m_sd > 1e-6

    if not has_gc_var and not has_map_var:
        med = float(np.median(counts[active]))
        return np.full(n_all, max(med, 1e-3), dtype=np.float64)

    # Standardize covariates (put varying axis in `u` if only 1 axis varies)
    if has_gc_var and has_map_var:
        u_all = (np.nan_to_num(gc, nan=g_mean) - g_mean) / g_sd
        v_all = (np.nan_to_num(mappability, nan=m_mean) - m_mean) / m_sd
        is_2d = True
    elif has_gc_var:
        u_all = (np.nan_to_num(gc, nan=g_mean) - g_mean) / g_sd
        v_all = np.zeros(n_all, dtype=np.float64)
        is_2d = False
    else:
        u_all = (np.nan_to_num(mappability, nan=m_mean) - m_mean) / m_sd
        v_all = np.zeros(n_all, dtype=np.float64)
        is_2d = False

    inlier = active.copy()
    surface_all = np.zeros(n_all, dtype=np.float64)

    u_min, u_max = float(np.min(u_all[active])), float(np.max(u_all[active]))
    v_min, v_max = float(np.min(v_all[active])), float(np.max(v_all[active]))

    use_grid = int(np.sum(active)) > 220
    if use_grid:
        u_knots = np.linspace(u_min - 1e-6, u_max + 1e-6, 26)
        if is_2d:
            v_knots = np.linspace(v_min - 1e-6, v_max + 1e-6, 14)
            ug, vg = np.meshgrid(u_knots, v_knots, indexing="ij")
            u_eval = ug.ravel()
            v_eval = vg.ravel()
        else:
            v_knots = np.array([0.0])
            u_eval = u_knots
            v_eval = np.zeros_like(u_knots)
    else:
        u_knots = np.empty(0)
        v_knots = np.empty(0)
        u_eval = u_all
        v_eval = v_all

    min_inliers = max(8, int(0.25 * np.sum(active)))
    for _ in range(max(1, int(max_iter))):
        u_tr = u_all[inlier]
        v_tr = v_all[inlier]
        y_tr = counts[inlier]

        raw_fit = _eval_local_poly_2d(
            u_tr, v_tr, y_tr, u_eval, v_eval, span=span, is_2d=is_2d
        )
        if use_grid:
            if is_2d:
                grid_vals = raw_fit.reshape(u_knots.size, v_knots.size)
                interp = RegularGridInterpolator(
                    (u_knots, v_knots),
                    grid_vals,
                    bounds_error=False,
                    fill_value=None,
                )
                pts = np.column_stack(
                    [
                        np.clip(u_all, u_knots[0], u_knots[-1]),
                        np.clip(v_all, v_knots[0], v_knots[-1]),
                    ]
                )
                surface_all = np.asarray(interp(pts), dtype=np.float64)
            else:
                surface_all = np.interp(u_all, u_knots, raw_fit)
        else:
            surface_all = raw_fit

        # Iterative outlier masking (masks CNAs with |z| > cutoff)
        resid = counts[active] - surface_all[active]
        med_r = float(np.median(resid))
        mad_r = float(np.median(np.abs(resid - med_r)))
        sd_r = max(1.4826 * mad_r, 1e-4)
        cand_inlier = active.copy()
        cand_inlier[active] = np.abs(resid - med_r) <= float(cutoff) * sd_r
        if int(np.sum(cand_inlier)) >= min_inliers:
            inlier = cand_inlier

    floor_val = max(0.05 * float(np.median(counts[inlier])), 1e-3)
    return np.maximum(surface_all, floor_val)


def estimate_correction(
    qdata: QDNAseqData,
    max_iter: int = 3,
    cutoff: float = 4.0,
    span: float = 0.65,
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Estimate simultaneous 2D `(GC, mappability)` bias surface across samples
    with iterative CNA outlier exclusion (`QDNAseq::estimateCorrection`).
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    if not isinstance(qdata, QDNAseqData):
        raise ValueError("`qdata` must be a `QDNAseqData` instance.")
    if span <= 0.05 or span > 1.0:
        raise ValueError(f"`span` must be in (0.05, 1.0], got {span}.")

    gc = qdata.bins["gc"].to_numpy(dtype=np.float64)
    mapp = qdata.bins["mappability"].to_numpy(dtype=np.float64)
    use = qdata.bins["use"].to_numpy(dtype=bool)

    fit_cols: dict[str, np.ndarray] = {}
    for sname in qdata.sample_names:
        cnt = qdata.counts[sname].to_numpy(dtype=np.float64)
        surf = fit_2d_loess_surface(
            gc,
            mapp,
            cnt,
            use,
            max_iter=max_iter,
            cutoff=cutoff,
            span=span,
        )
        fit_cols[sname] = surf

    return qdata.clone(fit=pd.DataFrame(fit_cols), settings=cfg.settings)


def correct_counts(
    qdata: QDNAseqData,
    method: Literal["median", "mean", "none"] = "median",
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Divide raw bin counts by the 2D `(GC, mappability)` expected fit surface
    (`QDNAseq::correctCounts`). Runs `estimate_correction` automatically if needed.
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    if method not in ("median", "mean", "none"):
        raise ValueError(f"Unknown `method`={method!r}. Expected 'median', 'mean', or 'none'.")

    cur = qdata if qdata.fit is not None else estimate_correction(qdata, settings=cfg.settings)
    assert cur.fit is not None
    use = cur.bins["use"].to_numpy(dtype=bool)

    cn_cols: dict[str, np.ndarray] = {}
    for sname in cur.sample_names:
        raw = cur.counts[sname].to_numpy(dtype=np.float64)
        fit = np.maximum(cur.fit[sname].to_numpy(dtype=np.float64), 1e-6)
        ratio = raw / fit
        active_r = ratio[use & np.isfinite(ratio)]
        if active_r.size > 0 and method != "none":
            scale = (
                float(np.median(active_r))
                if method == "median"
                else float(np.mean(active_r))
            )
            if scale > 1e-8:
                ratio = ratio / scale
        cn_cols[sname] = ratio

    return cur.clone(copynumber=pd.DataFrame(cn_cols), settings=cfg.settings)


def normalize_bins(
    qdata: QDNAseqData,
    method: Literal["median", "mean", "mode"] = "median",
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """Normalize corrected bin-level copynumber ratios across active bins (`QDNAseq::normalizeBins`)."""
    cfg = get_qdnaseq_config(validate_settings(settings))
    if method not in ("median", "mean", "mode"):
        raise ValueError(f"Unknown `method`={method!r}. Expected 'median', 'mean', or 'mode'.")

    cur = qdata if qdata.copynumber is not None else correct_counts(qdata, settings=cfg.settings)
    assert cur.copynumber is not None
    use = cur.bins["use"].to_numpy(dtype=bool)

    norm_cols: dict[str, np.ndarray] = {}
    for sname in cur.sample_names:
        vals = cur.copynumber[sname].to_numpy(dtype=np.float64).copy()
        act = vals[use & np.isfinite(vals) & (vals > 0)]
        if act.size > 0:
            if method == "median":
                center = float(np.median(act))
            elif method == "mean":
                center = float(np.mean(act))
            else:
                hist, edges = np.histogram(act, bins=min(50, max(10, act.size // 10)))
                best_b = int(np.argmax(hist))
                center = 0.5 * float(edges[best_b] + edges[best_b + 1])
            if center > 1e-8:
                vals = vals / center
        norm_cols[sname] = vals

    return cur.clone(copynumber=pd.DataFrame(norm_cols), settings=cfg.settings)


def smooth_outlier_bins(
    qdata: QDNAseqData,
    max_sd: float = 4.0,
    window: int = 5,
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Smooth isolated singleton bin spikes exceeding `max_sd` local MADs from flanking
    chromosomal neighbors (`QDNAseq::smoothOutlierBins`).
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    cur = qdata if qdata.copynumber is not None else normalize_bins(qdata, settings=cfg.settings)
    assert cur.copynumber is not None

    bins = cur.bins
    use = bins["use"].to_numpy(dtype=bool)
    win = max(2, int(window))

    smoothed_cols: dict[str, np.ndarray] = {}
    for sname in cur.sample_names:
        arr = cur.copynumber[sname].to_numpy(dtype=np.float64).copy()
        for chrom in pd.unique(bins["chrom"]):
            c_idx = np.flatnonzero((bins["chrom"].to_numpy() == chrom) & use)
            if c_idx.size < 2 * win + 1:
                continue
            sub = arr[c_idx]
            diffs = np.diff(sub) / np.sqrt(2.0)
            noise_sd = max(1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))), 1e-4)

            for loc_i in range(c_idx.size):
                lo = max(0, loc_i - win)
                hi = min(c_idx.size, loc_i + win + 1)
                neigh = np.concatenate([sub[lo:loc_i], sub[loc_i + 1 : hi]])
                if neigh.size < 2:
                    continue
                n_med = float(np.median(neigh))
                # Only smooth isolated spikes where neighbors themselves agree closely
                if abs(sub[loc_i] - n_med) > float(max_sd) * noise_sd:
                    arr[c_idx[loc_i]] = n_med
        smoothed_cols[sname] = arr

    return cur.clone(copynumber=pd.DataFrame(smoothed_cols), settings=cfg.settings)
