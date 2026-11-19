"""Window clustering and region-level FDR control via Simes' procedure (csaw::mergeWindows / combineTests)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import SettingsMode, benjamini_hochberg, validate_settings
from .windows import WindowData


@dataclass
class MergeResult:
    """Clustered genomic regions and window-to-region assignments from ``merge_windows``."""

    ids: np.ndarray
    regions: pd.DataFrame


def merge_windows(
    windows_df: WindowData | pd.DataFrame,
    tol: int = 100,
    max_width: int | None = 5000,
    settings: SettingsMode = "default",
) -> MergeResult:
    """Merge overlapping or nearby genomic windows into candidate binding regions.

    Implements ``csaw::mergeWindows`` (Lun & Smyth, *NAR* 2016): windows separated by at
    most ``tol`` bp on the same chromosome are chained into clusters, and super-clusters
    exceeding ``max_width`` bp are partitioned into sub-regions of width ``<= max_width``
    to prevent runaway chaining across broad domains.

    Parameters
    ----------
    windows_df : WindowData | pd.DataFrame
        Window coordinates containing ``chrom``, ``start``, ``end`` (or ``WindowData``).
    tol : int
        Maximum gap (bp) between adjacent windows to merge into one region (``tol >= 0``).
    max_width : int | None
        Maximum allowed span (bp) for a merged region; oversized clusters are split evenly.
        Pass ``None`` to disable splitting.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    MergeResult
        0-based ``ids`` array aligned with ``windows_df`` rows, and ``regions`` DataFrame.
    """
    validate_settings(settings)
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}.")
    if max_width is not None and max_width <= 0:
        raise ValueError(f"max_width must be positive when provided, got {max_width}.")

    df = windows_df.windows if isinstance(windows_df, WindowData) else windows_df
    for req in ("chrom", "start", "end"):
        if req not in df.columns:
            raise ValueError(f"windows_df must contain column '{req}'.")

    n = len(df)
    if n == 0:
        empty_reg = pd.DataFrame(
            columns=["region_id", "chrom", "start", "end", "width", "n_windows"]
        )
        return MergeResult(ids=np.empty(0, dtype=np.int64), regions=empty_reg)

    chroms = df["chrom"].astype(str).to_numpy()
    starts = df["start"].to_numpy(dtype=np.int64)
    ends = df["end"].to_numpy(dtype=np.int64)

    # Preserve chromosome order of first appearance, sorting by (start, end) within chrom
    chrom_order = {c: idx for idx, c in enumerate(pd.unique(chroms))}
    chrom_codes = np.array([chrom_order[c] for c in chroms], dtype=np.int64)
    order = np.lexsort((ends, starts, chrom_codes))

    s_chrom = chroms[order]
    s_start = starts[order]
    s_end = ends[order]

    # Primary single-linkage chaining within tol
    raw_groups: list[tuple[int, int]] = []
    g_start_idx = 0
    cur_end = int(s_end[0])
    for i in range(1, n):
        if s_chrom[i] == s_chrom[i - 1] and int(s_start[i]) - cur_end <= tol:
            if int(s_end[i]) > cur_end:
                cur_end = int(s_end[i])
        else:
            raw_groups.append((g_start_idx, i))
            g_start_idx = i
            cur_end = int(s_end[i])
    raw_groups.append((g_start_idx, n))

    sorted_ids = np.empty(n, dtype=np.int64)
    reg_records: list[dict[str, object]] = []
    next_id = 0

    for lo, hi in raw_groups:
        c_name = str(s_chrom[lo])
        c_start = int(s_start[lo])
        c_end = int(np.max(s_end[lo:hi]))
        span = c_end - c_start

        if max_width is None or span <= max_width or (hi - lo) == 1:
            sorted_ids[lo:hi] = next_id
            reg_records.append(
                {
                    "region_id": next_id,
                    "chrom": c_name,
                    "start": c_start,
                    "end": c_end,
                    "width": c_end - c_start,
                    "n_windows": hi - lo,
                }
            )
            next_id += 1
        else:
            # Split super-cluster into n_sub equal genomic partitions as in csaw::mergeWindows
            n_sub = int(np.ceil(span / float(max_width)))
            mids = 0.5 * (s_start[lo:hi] + s_end[lo:hi])
            bins = np.floor((mids - c_start) / (span / float(n_sub))).astype(np.int64)
            bins = np.clip(bins, 0, n_sub - 1)
            for b in np.unique(bins):
                sub_mask = bins == b
                sub_indices = np.arange(lo, hi)[sub_mask]
                sub_s = int(np.min(s_start[sub_indices]))
                sub_e = int(np.max(s_end[sub_indices]))
                sorted_ids[sub_indices] = next_id
                reg_records.append(
                    {
                        "region_id": next_id,
                        "chrom": c_name,
                        "start": sub_s,
                        "end": sub_e,
                        "width": sub_e - sub_s,
                        "n_windows": int(sub_indices.size),
                    }
                )
                next_id += 1

    ids = np.empty(n, dtype=np.int64)
    ids[order] = sorted_ids
    regions_df = pd.DataFrame(reg_records)
    return MergeResult(ids=ids, regions=regions_df)


def combine_tests(
    merged_ids: MergeResult | Sequence[int] | np.ndarray,
    tab_df: pd.DataFrame,
    weights: Sequence[float] | np.ndarray | None = None,
    fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    pval_col: str = "pval",
    fc_col: str = "logFC",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Combine window-level tests into region-level inference via weighted Simes' procedure.

    Implements ``csaw::combineTests`` (Benjamini & Hochberg 1997; Lun & Smyth, *NAR* 2016):

    .. math::

        p_{\\text{region}} = \\min_{1 \\le k \\le K} \\frac{p_{(k)} \\sum_{j=1}^K w_j}{\\sum_{j=1}^k w_{(j)}}

    Parameters
    ----------
    merged_ids : MergeResult | Sequence[int] | np.ndarray
        Region cluster IDs per window (or :class:`MergeResult` from :func:`merge_windows`,
        which automatically attaches genomic coordinates to the output table).
    tab_df : pd.DataFrame
        Window-level results DataFrame containing ``pval_col`` and ``fc_col``.
    weights : Sequence[float] | np.ndarray | None
        Optional positive window weights for weighted Simes' procedure.
    fc_threshold : float
        Minimum ``|logFC|`` threshold for counting ``num_up_logFC`` and ``num_down_logFC``
        (default ``0.5``).
    pval_threshold : float
        Within-cluster Simes/BH adjusted significance threshold used under
        ``settings="R_default"`` (``csaw`` v1.44 parity) for ``num_up_logFC``/``num_down_logFC``.
    pval_col : str
        Column name for window p-values in ``tab_df``.
    fc_col : str
        Column name for window log2 fold-changes in ``tab_df``.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    pd.DataFrame
        Region-level summary table with ``region_id`` (plus ``chrom``, ``start``, ``end``
        if ``MergeResult`` passed), ``num_tests``, ``num_up_logFC``, ``num_down_logFC``,
        ``rep_test``, ``rep_logFC``, ``direction``, ``pval``, and ``FDR``.
    """
    validate_settings(settings)
    if fc_threshold < 0:
        raise ValueError(f"fc_threshold must be non-negative, got {fc_threshold}.")
    if pval_col not in tab_df.columns:
        raise ValueError(f"tab_df is missing p-value column '{pval_col}'.")
    if fc_col not in tab_df.columns:
        raise ValueError(f"tab_df is missing logFC column '{fc_col}'.")

    regions_meta: pd.DataFrame | None = None
    if isinstance(merged_ids, MergeResult):
        ids = np.asarray(merged_ids.ids, dtype=np.int64)
        regions_meta = merged_ids.regions
    else:
        ids = np.asarray(merged_ids, dtype=np.int64)

    n_win = len(tab_df)
    if ids.shape[0] != n_win:
        raise ValueError(
            f"merged_ids length ({ids.shape[0]}) must match tab_df rows ({n_win})."
        )

    if weights is None:
        w_all = np.ones(n_win, dtype=float)
    else:
        w_all = np.asarray(weights, dtype=float).ravel()
        if w_all.shape[0] != n_win:
            raise ValueError(
                f"weights length ({w_all.shape[0]}) must match tab_df rows ({n_win})."
            )
        if np.any(w_all <= 0):
            raise ValueError("All window weights must be strictly positive.")

    p_all = np.clip(tab_df[pval_col].to_numpy(dtype=float), 0.0, 1.0)
    fc_all = tab_df[fc_col].to_numpy(dtype=float)

    if n_win == 0:
        cols = [
            "region_id",
            "num_tests",
            "num_up_logFC",
            "num_down_logFC",
            "rep_test",
            "rep_logFC",
            "direction",
            "pval",
            "FDR",
        ]
        return pd.DataFrame(columns=cols)

    # Group indices by region_id efficiently via stable sort
    sort_by_id = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[sort_by_id]
    uniq_ids, group_starts = np.unique(sorted_ids, return_index=True)
    group_ends = np.append(group_starts[1:], n_win)

    rows: list[dict[str, object]] = []
    for g_idx, reg_id in enumerate(uniq_ids):
        win_indices = sort_by_id[group_starts[g_idx] : group_ends[g_idx]]
        p_sub = p_all[win_indices]
        w_sub = w_all[win_indices]
        fc_sub = fc_all[win_indices]
        k_tests = int(win_indices.size)

        # Weighted Simes' procedure: sort ascending by p_j (ties broken by larger weight)
        p_ord = np.lexsort((-w_sub, p_sub))
        p_s = p_sub[p_ord]
        w_s = w_sub[p_ord]
        fc_s = fc_sub[p_ord]
        orig_s = win_indices[p_ord]

        total_w = float(np.sum(w_s))
        cum_w = np.cumsum(w_s)
        simes_vals = p_s * (total_w / cum_w)

        # Within-cluster weighted BH/Simes step-up adjusted p-values:
        # Note min(q_s) == q_s[0] == min(simes_vals) == p_region identically!
        q_s = np.minimum.accumulate(simes_vals[::-1])[::-1]
        q_s = np.clip(q_s, 0.0, 1.0)
        p_region = float(q_s[0])

        # Representative window: lowest weighted p-value (p_j / w_j), matching csaw
        rep_pos = int(np.argmin(p_s / w_s))
        rep_test = int(orig_s[rep_pos])
        rep_lfc = float(fc_s[rep_pos])

        if settings == "R_default":
            # csaw v1.44 exact parity: count up/down strictly among individually
            # Simes/Holm-significant windows (q_s <= pval_threshold) with |logFC| >= fc_threshold
            sig_mask = q_s <= pval_threshold
            n_up = int(np.sum(sig_mask & (fc_s >= fc_threshold)))
            n_down = int(np.sum(sig_mask & (fc_s <= -fc_threshold)))
            if n_up > 0 and n_down == 0:
                direction = "up"
            elif n_down > 0 and n_up == 0:
                direction = "down"
            elif n_up > 0 and n_down > 0:
                direction = "mixed"
            else:
                # Sub-threshold cluster: inspect Simes-contributing prefix (q_s <= p_region + 1e-15),
                # falling back to rep_logFC sign
                contrib = q_s <= (p_region + 1e-15)
                has_pos = bool(np.any(contrib & (fc_s > 0)))
                has_neg = bool(np.any(contrib & (fc_s < 0)))
                if has_pos and not has_neg:
                    direction = "up"
                elif has_neg and not has_pos:
                    direction = "down"
                elif has_pos and has_neg:
                    direction = "mixed"
                else:
                    direction = "up" if rep_lfc > 0 else ("down" if rep_lfc < 0 else "mixed")
        else:
            # Pythonic default: evaluate active Simes-contributing / significant windows
            active_mask = q_s <= max(pval_threshold, p_region + 1e-15)
            n_up = int(np.sum(active_mask & (fc_s >= fc_threshold)))
            n_down = int(np.sum(active_mask & (fc_s <= -fc_threshold)))
            if n_up > 0 and n_down == 0:
                direction = "up"
            elif n_down > 0 and n_up == 0:
                direction = "down"
            elif n_up > 0 and n_down > 0:
                direction = "mixed"
            else:
                contrib = q_s <= (p_region + 1e-15)
                has_pos = bool(np.any(contrib & (fc_s > 0)))
                has_neg = bool(np.any(contrib & (fc_s < 0)))
                if has_pos and not has_neg:
                    direction = "up"
                elif has_neg and not has_pos:
                    direction = "down"
                else:
                    direction = "up" if rep_lfc > 0 else ("down" if rep_lfc < 0 else "mixed")

        rows.append(
            {
                "region_id": int(reg_id),
                "num_tests": k_tests,
                "num_up_logFC": n_up,
                "num_down_logFC": n_down,
                "rep_test": rep_test,
                "rep_logFC": rep_lfc,
                "direction": direction,
                "pval": p_region,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df["FDR"] = benjamini_hochberg(out_df["pval"].to_numpy(dtype=float))

    if regions_meta is not None and len(regions_meta) > 0:
        coord_cols = [
            c for c in ("region_id", "chrom", "start", "end", "width") if c in regions_meta.columns
        ]
        out_df = regions_meta[coord_cols].merge(out_df, on="region_id", how="right")
    return out_df


def get_best_test(
    merged_ids: MergeResult | Sequence[int] | np.ndarray,
    tab_df: pd.DataFrame,
    weights: Sequence[float] | np.ndarray | None = None,
    by_abundance: bool = False,
    pval_col: str = "pval",
    fc_col: str = "logFC",
    cpm_col: str = "logCPM",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Identify the summit/best window in each merged region with Bonferroni correction.

    Implements ``csaw::getBestTest`` (Lun & Smyth, *NAR* 2016). Either selects the window
    minimizing ``pval / weight`` with exact weighted Bonferroni region p-value
    ``min(1, p_best * sum(w) / w_best)``, or when ``by_abundance=True`` selects the
    highest-abundance summit window and reports its p-value directly.
    """
    validate_settings(settings)
    if pval_col not in tab_df.columns:
        raise ValueError(f"tab_df is missing p-value column '{pval_col}'.")
    if fc_col not in tab_df.columns:
        raise ValueError(f"tab_df is missing logFC column '{fc_col}'.")
    if by_abundance and cpm_col not in tab_df.columns:
        raise ValueError(f"by_abundance=True requires '{cpm_col}' column in tab_df.")

    regions_meta: pd.DataFrame | None = None
    if isinstance(merged_ids, MergeResult):
        ids = np.asarray(merged_ids.ids, dtype=np.int64)
        regions_meta = merged_ids.regions
    else:
        ids = np.asarray(merged_ids, dtype=np.int64)

    n_win = len(tab_df)
    if ids.shape[0] != n_win:
        raise ValueError(
            f"merged_ids length ({ids.shape[0]}) must match tab_df rows ({n_win})."
        )

    if weights is None:
        w_all = np.ones(n_win, dtype=float)
    else:
        w_all = np.asarray(weights, dtype=float).ravel()
        if w_all.shape[0] != n_win or np.any(w_all <= 0):
            raise ValueError("weights must match tab_df length and be strictly positive.")

    p_all = np.clip(tab_df[pval_col].to_numpy(dtype=float), 0.0, 1.0)
    fc_all = tab_df[fc_col].to_numpy(dtype=float)
    cpm_all = tab_df[cpm_col].to_numpy(dtype=float) if cpm_col in tab_df.columns else None

    if n_win == 0:
        return pd.DataFrame(
            columns=["region_id", "num_tests", "rep_test", "rep_logFC", "pval", "FDR"]
        )

    sort_by_id = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[sort_by_id]
    uniq_ids, group_starts = np.unique(sorted_ids, return_index=True)
    group_ends = np.append(group_starts[1:], n_win)

    rows: list[dict[str, object]] = []
    for g_idx, reg_id in enumerate(uniq_ids):
        win_indices = sort_by_id[group_starts[g_idx] : group_ends[g_idx]]
        p_sub = p_all[win_indices]
        w_sub = w_all[win_indices]
        fc_sub = fc_all[win_indices]

        if by_abundance and cpm_all is not None:
            cpm_sub = cpm_all[win_indices]
            best_local = int(np.argmax(cpm_sub))
            p_reg = float(p_sub[best_local])
        else:
            scaled = p_sub / w_sub
            best_local = int(np.argmin(scaled))
            p_reg = float(min(1.0, scaled[best_local] * float(np.sum(w_sub))))

        rec: dict[str, object] = {
            "region_id": int(reg_id),
            "num_tests": int(win_indices.size),
            "rep_test": int(win_indices[best_local]),
            "rep_logFC": float(fc_sub[best_local]),
            "pval": p_reg,
        }
        if cpm_all is not None:
            rec["rep_logCPM"] = float(cpm_all[win_indices[best_local]])
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df["FDR"] = benjamini_hochberg(out_df["pval"].to_numpy(dtype=float))
    if regions_meta is not None and len(regions_meta) > 0:
        coord_cols = [
            c for c in ("region_id", "chrom", "start", "end", "width") if c in regions_meta.columns
        ]
        out_df = regions_meta[coord_cols].merge(out_df, on="region_id", how="right")
    return out_df
