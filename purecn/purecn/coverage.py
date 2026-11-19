"""
Tumor-only coverage GC-bias normalization, SVD/pseudo-inverse Panel-of-Normals
Tangent normalization, and CBS/PELT log2 ratio segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .config import SettingsMode, get_purecn_config, validate_settings


@dataclass(frozen=True)
class CoverageNormalizationResult:
    """Normalized bin-level and segmented log2 coverage result."""

    bins: pd.DataFrame
    segments: pd.DataFrame
    gc_corrected: bool
    tangent_corrected: bool
    noise_sd: float
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return segmented copy-number DataFrame."""
        return self.segments.copy()

    def summary(self) -> str:
        return (
            f"PureCN CoverageNormalizationResult (settings={self.settings})\n"
            f"  Bins:               {len(self.bins)}\n"
            f"  Segments:           {len(self.segments)}\n"
            f"  GC corrected:       {self.gc_corrected}\n"
            f"  Tangent normalized: {self.tangent_corrected}\n"
            f"  Bin noise SD:       {self.noise_sd:.4f}"
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, key: Any) -> Any:
        return self.segments[key]


def _resolve_coverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize input coverage DataFrame column names."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("`tumor_cov_df` must be a non-empty pandas DataFrame.")
    out = df.copy()
    col_map: dict[str, str] = {}
    lower_to_orig = {c.lower(): c for c in out.columns}

    if "chrom" not in out.columns:
        for alias in ("chr", "chromosome", "seqnames", "contig"):
            if alias in lower_to_orig:
                col_map[lower_to_orig[alias]] = "chrom"
                break
    if "start" not in out.columns:
        for alias in ("pos", "position", "bin_start"):
            if alias in lower_to_orig:
                col_map[lower_to_orig[alias]] = "start"
                break
    if "coverage" not in out.columns:
        for alias in ("counts", "count", "depth", "dp", "reads", "tumor_cov"):
            if alias in lower_to_orig:
                col_map[lower_to_orig[alias]] = "coverage"
                break
    if "gc" not in out.columns:
        for alias in ("gc_content", "pct_gc", "gc_frac"):
            if alias in lower_to_orig:
                col_map[lower_to_orig[alias]] = "gc"
                break

    if col_map:
        out = out.rename(columns=col_map)

    for req in ("chrom", "start", "coverage"):
        if req not in out.columns:
            raise ValueError(f"`tumor_cov_df` is missing required column {req!r}.")

    if "end" not in out.columns:
        out["end"] = out["start"].astype(np.int64) + 200

    out["chrom"] = out["chrom"].astype(str)
    out["start"] = out["start"].astype(np.int64)
    out["end"] = np.maximum(out["end"].astype(np.int64), out["start"])
    out["coverage"] = out["coverage"].astype(np.float64)

    if "gc" in out.columns:
        gc_vals = out["gc"].astype(np.float64).to_numpy()
        finite_gc = gc_vals[np.isfinite(gc_vals)]
        if finite_gc.size > 0 and float(np.nanmax(finite_gc)) > 1.5:
            gc_vals = gc_vals / 100.0
        out["gc"] = np.clip(gc_vals, 0.0, 1.0)

    return out.reset_index(drop=True)


def gc_correct_log2(
    log2_ratio: np.ndarray,
    gc: np.ndarray,
    *,
    frac: float = 0.30,
    n_iter: int = 3,
) -> np.ndarray:
    """
    Robust Cleveland tricube LOWESS + quadratic fallback GC-bias correction.
    """
    y = np.asarray(log2_ratio, dtype=np.float64).copy()
    x = np.asarray(gc, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    n_val = int(np.sum(valid))
    if n_val < 6 or float(np.std(x[valid])) < 1e-8:
        return y - float(np.nanmedian(y))

    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv, kind="mergesort")
    xs = xv[order]
    ys = yv[order]

    n_knots = min(120, n_val)
    if n_val > 150:
        knots = np.unique(
            np.concatenate(
                [
                    np.quantile(xs, np.linspace(0.0, 1.0, n_knots)),
                    np.linspace(xs[0], xs[-1], 40),
                ]
            )
        )
    else:
        knots = xs

    k_win = max(6, min(n_val, int(np.ceil(frac * n_val))))
    rob_w = np.ones(n_val, dtype=np.float64)
    trend_sorted = np.zeros(n_val, dtype=np.float64)

    for _ in range(max(1, n_iter)):
        knot_preds = np.empty(knots.size, dtype=np.float64)
        for idx, x0 in enumerate(knots):
            dists = np.abs(xs - x0)
            if k_win < n_val:
                part = np.argpartition(dists, k_win - 1)[:k_win]
            else:
                part = np.arange(n_val)
            h = float(np.max(dists[part]))
            if h <= 1e-12:
                w = rob_w[part]
                w_sum = float(np.sum(w))
                knot_preds[idx] = (
                    float(np.sum(w * ys[part]) / w_sum)
                    if w_sum > 0
                    else float(np.median(ys[part]))
                )
                continue
            u = np.clip(dists[part] / (h * 1.001), 0.0, 1.0)
            w = ((1.0 - u**3) ** 3) * rob_w[part]
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                knot_preds[idx] = float(np.median(ys[part]))
                continue
            dx = xs[part] - x0
            swx = float(np.sum(w * dx))
            swxx = float(np.sum(w * dx * dx))
            swy = float(np.sum(w * ys[part]))
            swxy = float(np.sum(w * dx * ys[part]))
            denom = w_sum * swxx - swx * swx
            if denom > 1e-12 * w_sum * w_sum:
                b0 = (swxx * swy - swx * swxy) / denom
            else:
                b0 = swy / w_sum
            knot_preds[idx] = b0

        if knots.size == n_val and np.array_equal(knots, xs):
            trend_sorted = knot_preds
        else:
            trend_sorted = np.interp(xs, knots, knot_preds)

        resid = ys - trend_sorted
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = max(6.0 * mad, 1e-4)
        v = np.clip(np.abs(resid) / scale, 0.0, 1.0)
        rob_w = (1.0 - v * v) ** 2

    trend = np.zeros_like(y)
    inv_order = np.empty(n_val, dtype=np.int64)
    inv_order[order] = np.arange(n_val)
    trend[valid] = trend_sorted[inv_order]
    corrected = y - trend
    return corrected


def tangent_normalize(
    tumor_log2: np.ndarray,
    normal_panel: pd.DataFrame | np.ndarray,
    gc: np.ndarray | None = None,
    *,
    lowess_frac: float = 0.30,
) -> np.ndarray:
    """
    SVD / Moore-Penrose pseudo-inverse Tangent normalization against a Panel of Normals.

    Projects `tumor_log2` onto the orthogonal complement of the normal-panel
    systematic noise subspace: `y_tan = y - U_r (U_r^T y)`.
    """
    y = np.asarray(tumor_log2, dtype=np.float64).copy()
    if isinstance(normal_panel, pd.DataFrame):
        num_cols = normal_panel.select_dtypes(include=[np.number]).columns.tolist()
        drop_meta = {"start", "end", "pos", "position", "gc", "mappability", "blacklist"}
        pon_cols = [c for c in num_cols if c.lower() not in drop_meta]
        if not pon_cols:
            raise ValueError("`normal_panel` DataFrame contains no numeric normal coverage columns.")
        pon = normal_panel[pon_cols].to_numpy(dtype=np.float64)
    else:
        pon = np.asarray(normal_panel, dtype=np.float64)
        if pon.ndim == 1:
            pon = pon[:, np.newaxis]

    if pon.shape[0] != y.shape[0]:
        raise ValueError(
            f"`normal_panel` row count ({pon.shape[0]}) must match tumor bin count ({y.shape[0]})."
        )
    if pon.shape[1] == 0:
        return y

    # Convert linear depths to centered log2 relative profiles if non-negative linear coverage
    finite_pon = pon[np.isfinite(pon)]
    is_linear_counts = (
        finite_pon.size > 0
        and float(np.min(finite_pon)) >= 0.0
        and float(np.median(finite_pon)) > 2.0
    )

    pon_log2 = np.zeros_like(pon, dtype=np.float64)
    for j in range(pon.shape[1]):
        col = np.nan_to_num(pon[:, j], nan=0.0)
        if is_linear_counts:
            med = max(float(np.median(col + 0.5)), 0.5)
            l2 = np.log2((np.maximum(col, 0.0) + 0.5) / med)
            if gc is not None:
                l2 = gc_correct_log2(l2, gc, frac=lowess_frac, n_iter=2)
        else:
            l2 = col.copy()
        l2 = l2 - float(np.median(l2))
        pon_log2[:, j] = l2

    # Subtract median across normals first (reference profile), then SVD residual subspace
    ref_profile = np.median(pon_log2, axis=1)
    y_rel = y - ref_profile
    pon_centered = pon_log2 - ref_profile[:, np.newaxis]

    if pon_centered.shape[1] >= 1 and float(np.std(pon_centered)) > 1e-10:
        u_mat, s_vals, _ = linalg.svd(pon_centered, full_matrices=False)
        s_max = float(s_vals[0]) if s_vals.size > 0 else 0.0
        if s_max > 1e-10:
            keep = s_vals > (1e-5 * s_max)
            u_basis = u_mat[:, keep]
            y_rel = y_rel - u_basis @ (u_basis.T @ y_rel)

    return y_rel


def _estimate_diff_sd(y: np.ndarray) -> float:
    """Robust per-bin noise SD from first differences."""
    if y.size < 3:
        return max(float(np.std(y)), 0.05) if y.size > 1 else 0.10
    diffs = np.diff(y) / np.sqrt(2.0)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sd = 1.4826 * mad
    if sd <= 1e-6:
        sd = float(np.std(diffs))
    return max(sd, 0.02)


def _binary_segment_1d(
    y: np.ndarray,
    *,
    alpha: float = 1e-4,
    min_mark: int = 3,
    undo_sd: float = 0.75,
) -> list[tuple[int, int]]:
    """
    Exact prefix-sum recursive binary segmentation (CBS/PELT-style) with
    analytic Welch/pooled t-statistic gating and SD-undo pruning.
    """
    n = y.size
    if n == 0:
        return []
    if n < 2 * min_mark:
        return [(0, n)]

    noise_sd = _estimate_diff_sd(y)
    prefix = np.concatenate([[0.0], np.cumsum(y)])

    # Critical z threshold adjusted for multiple testing across segment splits
    z_crit_base = float(stats.norm.isf(alpha / 2.0))

    breakpoints: set[int] = {0, n}

    def _recurse(start: int, end: int) -> None:
        m = end - start
        if m < 2 * min_mark:
            return
        ks = np.arange(start + min_mark, end - min_mark + 1)
        n_left = (ks - start).astype(np.float64)
        n_right = (end - ks).astype(np.float64)
        sum_left = prefix[ks] - prefix[start]
        sum_right = prefix[end] - prefix[ks]
        mean_left = sum_left / n_left
        mean_right = sum_right / n_right

        se = noise_sd * np.sqrt(1.0 / n_left + 1.0 / n_right)
        z_scores = np.abs(mean_left - mean_right) / np.maximum(se, 1e-9)
        best_idx = int(np.argmax(z_scores))
        max_z = float(z_scores[best_idx])
        k_best = int(ks[best_idx])

        # Also check ternary internal arc (CBS internal segment change)
        best_arc: tuple[int, int] | None = None
        if m >= 3 * min_mark and m <= 600:
            # Coarse-to-fine scan for internal focal CNV arc [i, j)
            step = max(1, m // 80)
            grid = np.arange(start + min_mark, end - min_mark + 1, step)
            total_sum = prefix[end] - prefix[start]
            for idx_i in range(grid.size):
                i_pt = int(grid[idx_i])
                for idx_j in range(idx_i + 1, grid.size):
                    j_pt = int(grid[idx_j])
                    arc_len = j_pt - i_pt
                    if arc_len < min_mark or (m - arc_len) < min_mark:
                        continue
                    arc_sum = prefix[j_pt] - prefix[i_pt]
                    out_sum = total_sum - arc_sum
                    mean_in = arc_sum / arc_len
                    mean_out = out_sum / (m - arc_len)
                    se_arc = noise_sd * np.sqrt(1.0 / arc_len + 1.0 / (m - arc_len))
                    z_arc = abs(mean_in - mean_out) / max(se_arc, 1e-9)
                    if z_arc > max_z + 0.5:
                        max_z = z_arc
                        best_arc = (i_pt, j_pt)

        eff_crit = max(z_crit_base, np.sqrt(2.0 * np.log(max(m, 2))))
        if max_z >= eff_crit:
            if best_arc is not None:
                i_pt, j_pt = best_arc
                breakpoints.add(i_pt)
                breakpoints.add(j_pt)
                _recurse(start, i_pt)
                _recurse(i_pt, j_pt)
                _recurse(j_pt, end)
            else:
                breakpoints.add(k_best)
                _recurse(start, k_best)
                _recurse(k_best, end)

    _recurse(0, n)
    pts = sorted(breakpoints)
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    # SD-undo pruning: merge adjacent segments closer than undo_sd * noise_sd
    if undo_sd > 0.0 and len(segs) > 1:
        merged = True
        while merged and len(segs) > 1:
            merged = False
            diffs = []
            for idx in range(len(segs) - 1):
                s1, e1 = segs[idx]
                s2, e2 = segs[idx + 1]
                m1 = float(np.mean(y[s1:e1]))
                m2 = float(np.mean(y[s2:e2]))
                diffs.append(abs(m1 - m2))
            min_idx = int(np.argmin(diffs))
            if diffs[min_idx] < undo_sd * noise_sd:
                s1, _ = segs[min_idx]
                _, e2 = segs[min_idx + 1]
                segs = segs[:min_idx] + [(s1, e2)] + segs[min_idx + 2 :]
                merged = True

    return segs


def segment_log2_ratios(
    bins_df: pd.DataFrame,
    *,
    undo_sd: float = 0.75,
    alpha: float = 1e-4,
    min_mark: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment bin-level `log2_ratio` per chromosome and return `(bins_df, segments_df)`."""
    bins_out = bins_df.copy()
    seg_rows: list[dict[str, Any]] = []
    seg_ids = np.zeros(len(bins_out), dtype=np.int64)
    seg_means = np.zeros(len(bins_out), dtype=np.float64)

    seg_counter = 0
    # Preserve chromosome order of appearance
    for chrom in pd.unique(bins_out["chrom"]):
        mask = (bins_out["chrom"] == chrom).to_numpy()
        idx = np.flatnonzero(mask)
        sub = bins_out.iloc[idx]
        order = np.argsort(sub["start"].to_numpy(), kind="mergesort")
        idx_sorted = idx[order]

        y = bins_out.loc[idx_sorted, "log2_ratio"].to_numpy(dtype=np.float64)
        spans = _binary_segment_1d(y, alpha=alpha, min_mark=min_mark, undo_sd=undo_sd)
        for s_lo, s_hi in spans:
            seg_idx = idx_sorted[s_lo:s_hi]
            seg_y = y[s_lo:s_hi]
            s_mean = float(np.mean(seg_y))
            s_sd = float(np.std(seg_y, ddof=1)) if seg_y.size > 1 else 0.05
            s_start = int(bins_out.loc[seg_idx[0], "start"])
            s_end = int(bins_out.loc[seg_idx[-1], "end"])
            seg_ids[seg_idx] = seg_counter
            seg_means[seg_idx] = s_mean
            seg_rows.append(
                {
                    "seg_id": seg_counter,
                    "chrom": str(chrom),
                    "start": s_start,
                    "end": s_end,
                    "num_mark": int(seg_y.size),
                    "seg_mean": s_mean,
                    "seg_sd": max(s_sd, 0.02),
                    "size_bp": max(1, s_end - s_start + 1),
                }
            )
            seg_counter += 1

    bins_out["seg_id"] = seg_ids
    bins_out["seg_mean"] = seg_means
    segments_df = pd.DataFrame(seg_rows)
    return bins_out, segments_df


def normalize_tumor_only_coverage(
    tumor_cov_df: pd.DataFrame,
    normal_panel: pd.DataFrame | np.ndarray | None = None,
    gc_correct: bool = True,
    tangent_norm: bool = True,
    settings: SettingsMode = "default",
) -> CoverageNormalizationResult:
    """
    Normalize tumor-only interval coverage via GC-bias regression, SVD/pseudo-inverse
    Panel-of-Normals Tangent normalization, and CBS/PELT segmentation.
    """
    cfg = get_purecn_config(validate_settings(settings))
    bins = _resolve_coverage_columns(tumor_cov_df)

    cov = np.maximum(bins["coverage"].to_numpy(dtype=np.float64), 0.0)
    if cfg.settings == "default":
        denom = max(float(np.median(cov + 0.5)), 0.5)
        log2_r = np.log2((cov + 0.5) / denom)
    else:
        cov_floor = np.maximum(cov, 1.0)
        denom = max(float(np.mean(cov_floor)), 1.0)
        log2_r = np.log2(cov_floor / denom)

    did_gc = False
    gc_arr: np.ndarray | None = None
    if "gc" in bins.columns:
        gc_arr = bins["gc"].to_numpy(dtype=np.float64)
        if gc_correct:
            log2_r = gc_correct_log2(log2_r, gc_arr, frac=cfg.lowess_frac, n_iter=3)
            did_gc = True

    did_tangent = False
    if tangent_norm and normal_panel is not None:
        log2_r = tangent_normalize(
            log2_r,
            normal_panel,
            gc=gc_arr if gc_correct else None,
            lowess_frac=cfg.lowess_frac,
        )
        did_tangent = True

    # Mass-balance normalization: E[2^log2_ratio] = 1.0 so log2_ratio matches
    # log2((rho * C_s + 2*(1-rho)) / (rho * ploidy + 2*(1-rho))) exactly
    mean_lin = float(np.mean(np.power(2.0, np.clip(log2_r, -8.0, 8.0))))
    if mean_lin > 1e-8:
        log2_r = log2_r - np.log2(mean_lin)

    bins["log2_ratio"] = log2_r
    noise_sd = _estimate_diff_sd(log2_r)
    bins_seg, segments_df = segment_log2_ratios(
        bins,
        undo_sd=cfg.cbs_undo_sd,
    )

    return CoverageNormalizationResult(
        bins=bins_seg,
        segments=segments_df,
        gc_corrected=did_gc,
        tangent_corrected=did_tangent,
        noise_sd=noise_sd,
        settings=cfg.settings,
    )
