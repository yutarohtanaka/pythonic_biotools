"""
Pythonic implementation of the ROSE (Rank Ordering of Super-Enhancers) algorithm
(Whyte et al., Cell 2013; Loven et al., Cell 2013).

Performs blacklist filtering, promoter-window (±2.5 kb TSS) exclusion prior to stitching,
distal enhancer stitching (default 12.5 kb), SPM or bigWig (with optional Input/IgG control
subtraction) signal integration, exact geometric tangent-slope ("hockey-stick") inflection
thresholding, and nearest-gene annotation.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class ROSEResult:
    """Output of `call_rose_super_enhancers`."""

    super_enhancers: pd.DataFrame
    stitched_regions: pd.DataFrame
    cutoff_signal: float
    cutoff_rank_ascending: int
    signal_unit: str

    def plot(
        self,
        ax: Axes | None = None,
        top_n_labels: int = 5,
        figsize: tuple[float, float] = (7.0, 5.0),
        *,
        title: str | None = None,
    ) -> Figure:
        """Plot ROSE ranked enhancer hockey-stick curve (`rose.plot_rose_hockey_stick`)."""
        from .plotting import plot_rose_hockey_stick

        return plot_rose_hockey_stick(
            self,
            ax=ax,
            top_n_labels=top_n_labels,
            figsize=figsize,
            title=title,
        )


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping or adjacent half-open intervals `[(start, end), ...]`."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: (iv[0], iv[1]))
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def load_bed_intervals(bed_source: str | Path | pd.DataFrame | None) -> dict[str, list[tuple[int, int]]]:
    """Load genomic intervals `{chrom: [(start, end), ...]}` sorted and merged into disjoint intervals."""
    if bed_source is None:
        return {}
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    if isinstance(bed_source, pd.DataFrame):
        for _, r in bed_source.iterrows():
            by_chrom.setdefault(str(r["chrom"]), []).append((int(r["start"]), int(r["end"])))
    else:
        with _open_maybe_gz(Path(bed_source)) as fp:
            for line in fp:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    by_chrom.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    return {c: _merge_intervals(ivs) for c, ivs in by_chrom.items()}


def _build_interval_index(
    interval_dict: dict[str, list[tuple[int, int]]],
) -> dict[str, tuple[list[int], list[int]]]:
    """Precompute `(starts, ends)` arrays per chromosome on disjoint merged intervals for O(log M) lookup."""
    index: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, ivs in interval_dict.items():
        merged = _merge_intervals(ivs)
        if merged:
            index[chrom] = ([iv[0] for iv in merged], [iv[1] for iv in merged])
    return index


def _overlaps_indexed(
    start: int,
    end: int,
    chrom_index: tuple[list[int], list[int]] | None,
) -> bool:
    """Exact O(log M) half-open overlap test against pre-indexed disjoint sorted intervals."""
    if chrom_index is None:
        return False
    starts, ends = chrom_index
    if not starts:
        return False
    idx = bisect.bisect_left(starts, end)
    return idx > 0 and ends[idx - 1] > start


def _overlaps_sorted(
    start: int,
    end: int,
    intervals: list[tuple[int, int]],
    starts: list[int] | None = None,
) -> bool:
    """Overlap helper supporting direct interval lists (merges if `starts` is not precomputed)."""
    if not intervals:
        return False
    if starts is not None:
        idx = bisect.bisect_left(starts, end)
        return idx > 0 and intervals[idx - 1][1] > start
    merged = _merge_intervals(intervals)
    m_starts = [iv[0] for iv in merged]
    idx = bisect.bisect_left(m_starts, end)
    return idx > 0 and merged[idx - 1][1] > start


def _contained_in_any_tss_window(
    start: int,
    end: int,
    sorted_tss: list[int] | None,
    window_bp: int,
) -> bool:
    """
    Return True iff `[start, end]` is fully contained inside at least one
    `[tss - window_bp, tss + window_bp]` interval (matching ROSE getContainers).
    Algebraically: `tss - W <= start` and `end <= tss + W` <=> `end - W <= tss <= start + W`.
    """
    if not sorted_tss or window_bp <= 0:
        return False
    lo = end - window_bp
    hi = start + window_bp
    if lo > hi:
        return False
    idx = bisect.bisect_left(sorted_tss, lo)
    return idx < len(sorted_tss) and sorted_tss[idx] <= hi


def _count_overlapping_tss50_genes(
    start: int,
    end: int,
    chrom_tss_genes: tuple[list[int], list[str]] | None,
    half_window_bp: int = 50,
) -> int:
    """
    Count distinct genes whose `[tss - 50, tss + 50]` window overlaps `[start, end]`
    (matching Young Lab ROSE_main.py L98-115 unstitching rule).
    """
    if chrom_tss_genes is None:
        return 0
    tss_positions, gene_ids = chrom_tss_genes
    if not tss_positions:
        return 0
    # Overlap of [start, end] with [tss - 50, tss + 50] <=> start - 50 <= tss <= end + 50
    lo_idx = bisect.bisect_left(tss_positions, start - half_window_bp)
    hi_idx = bisect.bisect_right(tss_positions, end + half_window_bp)
    if lo_idx >= hi_idx:
        return 0
    return len(set(gene_ids[lo_idx:hi_idx]))


def call_rose_super_enhancers(
    peaks: str | Path | pd.DataFrame,
    blacklist: str | Path | pd.DataFrame | None = None,
    tss_df: pd.DataFrame | None = None,
    bigwig_path: str | Path | None = None,
    control_bigwig_path: str | Path | None = None,
    stitch_bp: int = 12_500,
    tss_exclusion_bp: int = 2_500,
    allow_broad_promoter_peaks: bool = False,
    max_promoter_peak_width_bp: int = 2_500,
    score_col_index: int = 8,
    standard_chroms_only: bool = True,
    settings: SettingsMode = "default",
) -> ROSEResult:
    """Call stitched regulatory domains and ROSE super-enhancers from MACS2 narrowPeaks or DataFrame.

    Parameters
    ----------
    peaks : path to `.narrowPeak[.gz]` / `.bed[.gz]` or DataFrame
        If DataFrame, expects columns `['chrom', 'start', 'end']` and optional `'score'`.
        If file path, reads `score_col_index` (default `8` = MACS2 `-log10(q)` column 9; falls back
        to column 5 if fewer than 9 columns).
    blacklist : optional BED path or DataFrame
        ENCODE blacklist intervals to exclude before stitching.
    tss_df : optional DataFrame with columns `['chrom', 'tss', 'gene_name']`
        Used both to exclude constituent promoter peaks overlapping `±tss_exclusion_bp` before stitching
        and to annotate stitched enhancers with their nearest gene TSS.
    bigwig_path : optional path to ChIP/ATAC `.bigWig`
        When provided (requires `pyBigWig`), integrates `width_bp * mean_signal` across each stitched
        enhancer; raises `FileNotFoundError` if the path does not exist.
    control_bigwig_path : optional path to Input/IgG control `.bigWig`
        When provided alongside `bigwig_path`, subtracts mean control signal `width_bp * max(0, chip - ctrl)`.
    allow_broad_promoter_peaks : bool, default False
        When `False` (standard ROSE), excludes ALL peaks overlapping `TSS ± tss_exclusion_bp` so broad
        promoter peaks cannot stitch enhancers across a TSS. When `True`, only excludes promoter-overlapping
        peaks narrower than or equal to `max_promoter_peak_width_bp`.
    settings : {"default", "R_default"}, default "default"
        `"default"` preserves Pythonic ROSE behavior; `"R_default"` matches Young Lab
        `ROSE_main.py` and `ROSE_callSuper.R` (excludes `chrY`, `getContainers` strict TSS
        containment, unstitching across `>2` distinct `TSS ± 50 bp` windows, and
        `calculate_cutoff` `y - slope * x` minimization).
    """
    _validate_settings(settings)
    if control_bigwig_path is not None and bigwig_path is None:
        raise ValueError("control_bigwig_path requires bigwig_path to be provided.")
    if bigwig_path is not None and not Path(bigwig_path).exists():
        raise FileNotFoundError(f"bigwig_path does not exist: {bigwig_path}")
    if control_bigwig_path is not None and not Path(control_bigwig_path).exists():
        raise FileNotFoundError(f"control_bigwig_path does not exist: {control_bigwig_path}")

    if settings == "R_default":
        std_chroms = (
            {f"chr{i}" for i in range(1, 23)}
            | {"chrX"}
            | {str(i) for i in range(1, 23)}
            | {"X"}
        )
    else:
        std_chroms = (
            {f"chr{i}" for i in range(1, 23)}
            | {"chrX", "chrY"}
            | {str(i) for i in range(1, 23)}
            | {"X", "Y"}
        )
    bl_dict = load_bed_intervals(blacklist)
    bl_index = _build_interval_index(bl_dict)

    tss_windows: dict[str, list[tuple[int, int]]] = {}
    sorted_tss_by_chrom: dict[str, list[int]] = {}
    tss_gene_index: dict[str, tuple[list[int], list[str]]] = {}
    if tss_df is not None and {"chrom", "tss"}.issubset(tss_df.columns):
        has_gene_col = "gene_name" in tss_df.columns
        per_chrom_pairs: dict[str, list[tuple[int, str]]] = {}
        for row_idx, r in tss_df.iterrows():
            chrom_s = str(r["chrom"])
            pos = int(r["tss"])
            if tss_exclusion_bp > 0:
                tss_windows.setdefault(chrom_s, []).append(
                    (pos - tss_exclusion_bp, pos + tss_exclusion_bp)
                )
            gid = str(r["gene_name"]) if has_gene_col else f"{chrom_s}:{pos}:{row_idx}"
            per_chrom_pairs.setdefault(chrom_s, []).append((pos, gid))
        for chrom_s, pairs in per_chrom_pairs.items():
            pairs.sort(key=lambda x: x[0])
            sorted_tss_by_chrom[chrom_s] = [p[0] for p in pairs]
            tss_gene_index[chrom_s] = (
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            )
    tss_index = _build_interval_index(tss_windows)

    def _exclude_peak(chrom: str, s: int, e: int) -> bool:
        if _overlaps_indexed(s, e, bl_index.get(chrom)):
            return True
        if settings == "R_default":
            if tss_exclusion_bp > 0 and _contained_in_any_tss_window(
                s, e, sorted_tss_by_chrom.get(chrom), tss_exclusion_bp
            ):
                return True
            return False
        if _overlaps_indexed(s, e, tss_index.get(chrom)):
            if not allow_broad_promoter_peaks or (e - s) <= max_promoter_peak_width_bp:
                return True
        return False

    raw_peaks: list[tuple[str, int, int, float]] = []
    if isinstance(peaks, pd.DataFrame):
        has_score = "score" in peaks.columns
        for _, r in peaks.iterrows():
            chrom = str(r["chrom"])
            if standard_chroms_only and chrom not in std_chroms:
                continue
            s, e = int(r["start"]), int(r["end"])
            if _exclude_peak(chrom, s, e):
                continue
            sc = float(r["score"]) if has_score else float(max(e - s, 1))
            raw_peaks.append((chrom, s, e, sc))
    else:
        with _open_maybe_gz(Path(peaks)) as fp:
            for line in fp:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                chrom = p[0]
                if standard_chroms_only and chrom not in std_chroms:
                    continue
                s, e = int(p[1]), int(p[2])
                if _exclude_peak(chrom, s, e):
                    continue
                if len(p) > score_col_index:
                    sc = float(p[score_col_index])
                elif len(p) >= 5 and p[4] not in (".", ""):
                    sc = float(p[4])
                else:
                    sc = float(max(e - s, 1))
                raw_peaks.append((chrom, s, e, sc))

    if bigwig_path is not None:
        signal_unit = "bp_x_bigwig_bg_subtracted" if control_bigwig_path is not None else "bp_x_bigwig_mean"
    else:
        signal_unit = "SPM"

    empty_cols = [
        "chrom", "start", "end", "width_bp", "signal", "n_constituent_peaks",
        "rank_ascending", "rank_descending", "x_norm", "y_norm", "is_super_enhancer", "nearest_gene",
    ]
    if not raw_peaks:
        empty_df = pd.DataFrame(columns=empty_cols)
        return ROSEResult(empty_df, empty_df.copy(), 0.0, 0, signal_unit)

    total_score = sum(x[3] for x in raw_peaks)
    spm_scale = (total_score / 1e6) if total_score > 0.0 else 1.0
    by_chrom: dict[str, list[tuple[int, int, float]]] = {}
    for chrom, s, e, sc in raw_peaks:
        by_chrom.setdefault(chrom, []).append((s, e, sc / spm_scale))

    stitched: list[list] = []
    for chrom, ivs in by_chrom.items():
        ivs.sort(key=lambda x: x[0])
        cur_s, cur_e, cur_sig, n_sub = ivs[0][0], ivs[0][1], ivs[0][2], 1
        cur_members: list[tuple[int, int, float]] = [ivs[0]]
        for s, e, sig in ivs[1:]:
            if s - cur_e <= stitch_bp:
                cur_e = max(cur_e, e)
                cur_sig += sig
                n_sub += 1
                cur_members.append((s, e, sig))
            else:
                if (
                    settings == "R_default"
                    and n_sub > 1
                    and _count_overlapping_tss50_genes(
                        cur_s, cur_e, tss_gene_index.get(chrom)
                    )
                    > 2
                ):
                    for ms, me, msig in cur_members:
                        stitched.append([chrom, ms, me, msig, 1])
                else:
                    stitched.append([chrom, cur_s, cur_e, cur_sig, n_sub])
                cur_s, cur_e, cur_sig, n_sub = s, e, sig, 1
                cur_members = [(s, e, sig)]
        if (
            settings == "R_default"
            and n_sub > 1
            and _count_overlapping_tss50_genes(cur_s, cur_e, tss_gene_index.get(chrom))
            > 2
        ):
            for ms, me, msig in cur_members:
                stitched.append([chrom, ms, me, msig, 1])
        else:
            stitched.append([chrom, cur_s, cur_e, cur_sig, n_sub])

    if bigwig_path is not None:
        import pyBigWig

        if control_bigwig_path is not None:
            with pyBigWig.open(str(bigwig_path)) as bw, pyBigWig.open(str(control_bigwig_path)) as ctrl_bw:
                for item in stitched:
                    chrom, s, e = item[0], item[1], item[2]
                    val = bw.stats(chrom, s, e, type="mean")[0]
                    cval = ctrl_bw.stats(chrom, s, e, type="mean")[0]
                    mean_chip = float(val) if val is not None and np.isfinite(val) else 0.0
                    mean_ctrl = float(cval) if cval is not None and np.isfinite(cval) else 0.0
                    item[3] = (e - s) * max(0.0, mean_chip - mean_ctrl)
        else:
            with pyBigWig.open(str(bigwig_path)) as bw:
                for item in stitched:
                    chrom, s, e = item[0], item[1], item[2]
                    val = bw.stats(chrom, s, e, type="mean")[0]
                    mean_fc = float(val) if val is not None and np.isfinite(val) else 0.0
                    item[3] = (e - s) * mean_fc

    stitched.sort(key=lambda x: x[3])
    n_dom = len(stitched)
    sigs = np.array([item[3] for item in stitched], dtype=np.float64)
    max_sig = float(sigs[-1]) if sigs[-1] > 0.0 else 1.0
    x_norm = np.arange(n_dom, dtype=np.float64) / max(1, n_dom - 1)
    y_norm = sigs / max_sig
    if settings == "R_default":
        # ROSE_callSuper.R calculate_cutoff: minimize y - slope * x (slope = (max - min)/N, x = 1..N)
        y_vec = np.maximum(sigs, 0.0)
        x_vec = np.arange(1, n_dom + 1, dtype=np.float64)
        slope = (float(y_vec[-1]) - float(y_vec[0])) / float(max(1, n_dom))
        cutoff_idx = int(np.argmin(y_vec - slope * x_vec))
    else:
        # Geometric 45-degree tangent cutoff: maximize (x_norm - y_norm)
        cutoff_idx = int(np.argmax(x_norm - y_norm))
    cutoff_val = float(sigs[cutoff_idx])

    # Optional nearest-TSS gene annotation
    tss_by_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if tss_df is not None and {"chrom", "tss", "gene_name"}.issubset(tss_df.columns):
        for chrom, grp in tss_df.groupby("chrom"):
            g_sorted = grp.sort_values("tss")
            tss_by_chrom[str(chrom)] = (
                g_sorted["tss"].to_numpy(dtype=np.int64),
                g_sorted["gene_name"].astype(str).to_numpy(),
            )

    rows = []
    for idx, item in enumerate(stitched):
        chrom, s, e, sig, n_sub = item
        mid = (s + e) // 2
        nearest_g = ""
        if chrom in tss_by_chrom:
            pos_arr, name_arr = tss_by_chrom[chrom]
            ins = int(np.clip(np.searchsorted(pos_arr, mid), 0, len(pos_arr) - 1))
            best = ins
            if ins > 0 and abs(mid - pos_arr[ins - 1]) <= abs(pos_arr[ins] - mid):
                best = ins - 1
            nearest_g = str(name_arr[best])

        rows.append(
            {
                "chrom": chrom,
                "start": s,
                "end": e,
                "width_bp": e - s,
                "signal": float(sig),
                "n_constituent_peaks": int(n_sub),
                "rank_ascending": idx + 1,
                "rank_descending": n_dom - idx,
                "x_norm": float(x_norm[idx]),
                "y_norm": float(y_norm[idx]),
                "is_super_enhancer": bool(sig > cutoff_val),
                "nearest_gene": nearest_g,
            }
        )

    all_df = pd.DataFrame(rows)
    se_df = all_df.loc[all_df["is_super_enhancer"]].sort_values("rank_descending").reset_index(drop=True)
    return ROSEResult(
        super_enhancers=se_df,
        stitched_regions=all_df,
        cutoff_signal=cutoff_val,
        cutoff_rank_ascending=cutoff_idx + 1,
        signal_unit=signal_unit,
    )
