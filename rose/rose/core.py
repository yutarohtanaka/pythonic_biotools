"""
Pythonic implementation of the ROSE (Rank Ordering of Super-Enhancers) algorithm
(Whyte et al., Cell 2013; Loven et al., Cell 2013).

Performs blacklist filtering, promoter-window (±2.5 kb TSS) exclusion prior to stitching,
distal enhancer stitching (default 12.5 kb), SPM or bigWig signal integration, exact
geometric tangent-slope ("hockey-stick") inflection thresholding, and nearest-gene annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ROSEResult:
    """Output of `call_rose_super_enhancers`."""

    super_enhancers: pd.DataFrame
    stitched_regions: pd.DataFrame
    cutoff_signal: float
    cutoff_rank_ascending: int
    signal_unit: str


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_bed_intervals(bed_source: str | Path | pd.DataFrame | None) -> dict[str, list[tuple[int, int]]]:
    """Load genomic intervals `{chrom: [(start, end), ...]}` sorted by start coordinate."""
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
    for c in by_chrom:
        by_chrom[c].sort(key=lambda iv: iv[0])
    return by_chrom


def _overlaps_sorted(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    if not intervals:
        return False
    starts = [iv[0] for iv in intervals]
    import bisect

    idx = bisect.bisect_left(starts, end)
    for j in range(max(0, idx - 2), idx):
        if start < intervals[j][1] and end > intervals[j][0]:
            return True
    # Also scan backwards while potentially overlapping wide intervals
    for j in range(idx - 3, -1, -1):
        if intervals[j][1] <= start:
            # intervals are sorted by start only, so check conservative window
            if start - intervals[j][0] > 5_000_000:
                break
        elif start < intervals[j][1] and end > intervals[j][0]:
            return True
    return False


def call_rose_super_enhancers(
    peaks: str | Path | pd.DataFrame,
    blacklist: str | Path | pd.DataFrame | None = None,
    tss_df: pd.DataFrame | None = None,
    bigwig_path: str | Path | None = None,
    stitch_bp: int = 12_500,
    tss_exclusion_bp: int = 2_500,
    max_promoter_peak_width_bp: int = 2_500,
    score_col_index: int = 8,
    standard_chroms_only: bool = True,
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
        Used both to exclude constituent promoter peaks within `±tss_exclusion_bp` before stitching
        and to annotate stitched enhancers with their nearest gene TSS.
    bigwig_path : optional path to `.bigWig`
        When provided (requires `pyBigWig`), integrates `width_bp * mean_signal` across each stitched
        enhancer; otherwise uses Signal-Per-Million (SPM) normalized peak scores.
    """
    std_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"} | {str(i) for i in range(1, 23)} | {"X", "Y"}
    bl_dict = load_bed_intervals(blacklist)

    tss_windows: dict[str, list[tuple[int, int]]] = {}
    if tss_df is not None and {"chrom", "tss"}.issubset(tss_df.columns) and tss_exclusion_bp > 0:
        for _, r in tss_df.iterrows():
            pos = int(r["tss"])
            tss_windows.setdefault(str(r["chrom"]), []).append((pos - tss_exclusion_bp, pos + tss_exclusion_bp))
        for c in tss_windows:
            tss_windows[c].sort(key=lambda iv: iv[0])

    raw_peaks: list[tuple[str, int, int, float]] = []
    if isinstance(peaks, pd.DataFrame):
        has_score = "score" in peaks.columns
        for _, r in peaks.iterrows():
            chrom = str(r["chrom"])
            if standard_chroms_only and chrom not in std_chroms:
                continue
            s, e = int(r["start"]), int(r["end"])
            if _overlaps_sorted(s, e, bl_dict.get(chrom, [])):
                continue
            if (e - s) <= max_promoter_peak_width_bp and _overlaps_sorted(s, e, tss_windows.get(chrom, [])):
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
                if _overlaps_sorted(s, e, bl_dict.get(chrom, [])):
                    continue
                if (e - s) <= max_promoter_peak_width_bp and _overlaps_sorted(s, e, tss_windows.get(chrom, [])):
                    continue
                if len(p) > score_col_index:
                    sc = float(p[score_col_index])
                elif len(p) >= 5 and p[4] not in (".", ""):
                    sc = float(p[4])
                else:
                    sc = float(max(e - s, 1))
                raw_peaks.append((chrom, s, e, sc))

    empty_cols = [
        "chrom", "start", "end", "width_bp", "signal", "n_constituent_peaks",
        "rank_ascending", "rank_descending", "x_norm", "y_norm", "is_super_enhancer", "nearest_gene",
    ]
    if not raw_peaks:
        empty_df = pd.DataFrame(columns=empty_cols)
        return ROSEResult(empty_df, empty_df.copy(), 0.0, 0, "SPM")

    total_score = sum(x[3] for x in raw_peaks)
    spm_scale = (total_score / 1e6) if total_score > 0.0 else 1.0
    by_chrom: dict[str, list[tuple[int, int, float]]] = {}
    for chrom, s, e, sc in raw_peaks:
        by_chrom.setdefault(chrom, []).append((s, e, sc / spm_scale))

    stitched: list[list] = []
    for chrom, ivs in by_chrom.items():
        ivs.sort(key=lambda x: x[0])
        cur_s, cur_e, cur_sig, n_sub = ivs[0][0], ivs[0][1], ivs[0][2], 1
        for s, e, sig in ivs[1:]:
            if s - cur_e <= stitch_bp:
                cur_e = max(cur_e, e)
                cur_sig += sig
                n_sub += 1
            else:
                stitched.append([chrom, cur_s, cur_e, cur_sig, n_sub])
                cur_s, cur_e, cur_sig, n_sub = s, e, sig, 1
        stitched.append([chrom, cur_s, cur_e, cur_sig, n_sub])

    signal_unit = "SPM"
    if bigwig_path is not None and Path(bigwig_path).exists():
        import pyBigWig

        with pyBigWig.open(str(bigwig_path)) as bw:
            for item in stitched:
                chrom, s, e = item[0], item[1], item[2]
                val = bw.stats(chrom, s, e, type="mean")[0]
                mean_fc = float(val) if val is not None else 0.0
                item[3] = (e - s) * mean_fc
        signal_unit = "bp_x_bigwig_mean"

    stitched.sort(key=lambda x: x[3])
    n_dom = len(stitched)
    sigs = np.array([item[3] for item in stitched], dtype=np.float64)
    max_sig = float(sigs[-1]) if sigs[-1] > 0.0 else 1.0
    x_norm = np.arange(n_dom, dtype=np.float64) / max(1, n_dom - 1)
    y_norm = sigs / max_sig
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
                "is_super_enhancer": bool(idx >= cutoff_idx),
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
