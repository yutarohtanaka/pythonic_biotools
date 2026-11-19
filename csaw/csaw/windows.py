"""Sliding-window and background-bin read/fragment counting (csaw::windowCounts)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pysam

from .utils import SettingsMode, validate_settings


@dataclass
class WindowData:
    """Container for sliding-window genomic coordinates, counts, and library totals.

    Mirrors Bioconductor's ``RangedSummarizedExperiment`` returned by ``csaw::windowCounts``.
    """

    windows: pd.DataFrame
    counts: np.ndarray
    totals: np.ndarray
    sample_names: list[str]
    width: int
    spacing: int
    ext: int
    bin_mode: bool = False
    settings: SettingsMode = "default"

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[1])

    def subset(self, mask: np.ndarray | Sequence[bool] | Sequence[int]) -> WindowData:
        """Subset windows while preserving full pre-filter library ``totals``."""
        arr_mask = np.asarray(mask)
        sub_win = self.windows.iloc[arr_mask].reset_index(drop=True)
        sub_counts = self.counts[arr_mask, :]
        return WindowData(
            windows=sub_win,
            counts=sub_counts,
            totals=self.totals.copy(),
            sample_names=list(self.sample_names),
            width=self.width,
            spacing=self.spacing,
            ext=self.ext,
            bin_mode=self.bin_mode,
            settings=self.settings,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Combine window coordinates and per-sample counts into a single DataFrame."""
        df = self.windows.copy().reset_index(drop=True)
        for j, name in enumerate(self.sample_names):
            df[name] = self.counts[:, j]
        return df

    def summary(self) -> str:
        mode = "bin" if self.bin_mode else "sliding-window"
        return (
            f"WindowData({mode}: {self.n_windows} windows x {self.n_samples} samples, "
            f"width={self.width}, spacing={self.spacing}, ext={self.ext}, "
            f"totals={self.totals.tolist()}, settings='{self.settings}')"
        )


def _extract_intervals_from_df(
    df: pd.DataFrame,
    ext: int,
    shift: int,
    min_mapq: int,
    bin_mode: bool,
    pe: str,
    max_frag: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert a single-sample read/fragment DataFrame into 0-based [start, end) intervals."""
    if len(df) == 0:
        return {}
    if "chrom" not in df.columns:
        raise ValueError("Fragment/read DataFrame must contain a 'chrom' column.")

    work = df
    if "mapq" in work.columns and min_mapq > 0:
        work = work[work["mapq"].to_numpy() >= min_mapq]
    if len(work) == 0:
        return {}

    if "start" in work.columns:
        starts_raw = work["start"].to_numpy(dtype=np.int64)
    elif "pos" in work.columns:
        starts_raw = work["pos"].to_numpy(dtype=np.int64)
    else:
        raise ValueError("Fragment/read DataFrame must contain a 'start' or 'pos' column.")

    has_end = "end" in work.columns
    ends_raw = work["end"].to_numpy(dtype=np.int64) if has_end else (starts_raw + 1)

    if pe == "both":
        if not has_end:
            raise ValueError("Paired-end mode (pe='both') requires an 'end' column in DataFrame.")
        frag_len = ends_raw - starts_raw
        valid_pe = (frag_len > 0) & (frag_len <= max_frag)
        work = work.iloc[valid_pe]
        starts_raw = starts_raw[valid_pe]
        ends_raw = ends_raw[valid_pe]
        if len(work) == 0:
            return {}
        if bin_mode:
            mid = (starts_raw + ends_raw) // 2
            iv_start = mid
            iv_end = mid + 1
        else:
            iv_start = starts_raw
            iv_end = ends_raw
    else:
        if "strand" in work.columns:
            strands = work["strand"].astype(str).to_numpy()
            is_rev = strands == "-"
        else:
            is_rev = np.zeros(len(work), dtype=bool)

        eff_ext = max(1, int(ext))
        p5_fwd = starts_raw + shift
        p5_rev = ends_raw - shift
        iv_start = np.where(is_rev, p5_rev - eff_ext, p5_fwd)
        iv_end = np.where(is_rev, p5_rev, p5_fwd + eff_ext)

    iv_start = np.maximum(0, iv_start)
    iv_end = np.maximum(iv_start + 1, iv_end)
    chroms = work["chrom"].astype(str).to_numpy()

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom in pd.unique(chroms):
        mask = chroms == chrom
        out[str(chrom)] = (iv_start[mask], iv_end[mask])
    return out


def _extract_intervals_from_bam(
    bam_path: str | Path,
    ext: int,
    shift: int,
    min_mapq: int,
    bin_mode: bool,
    pe: str,
    max_frag: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Read BAM via pysam and extract extended SE reads or PE fragments per chromosome."""
    header_sizes: dict[str, int] = {}
    per_chrom_starts: dict[str, list[int]] = {}
    per_chrom_ends: dict[str, list[int]] = {}
    eff_ext = max(1, int(ext))

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for ref, length in zip(bam.references, bam.lengths):
            header_sizes[str(ref)] = int(length)

        if pe == "both":
            pending: dict[tuple[str, int], pysam.AlignedSegment] = {}
            for read in bam.fetch(until_eof=True):
                if (
                    read.is_unmapped
                    or read.mate_is_unmapped
                    or not read.is_paired
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_duplicate
                    or read.is_qcfail
                    or read.reference_id < 0
                    or read.reference_id != read.next_reference_id
                ):
                    continue
                key = (read.query_name or "", read.reference_id)
                mate = pending.pop(key, None)
                if mate is None:
                    pending[key] = read
                    continue
                # Both mates retrieved: check min_mapq on both mates as in csaw::readParam(minq=...)
                if read.mapping_quality < min_mapq or mate.mapping_quality < min_mapq:
                    continue
                r_start = min(int(read.reference_start), int(mate.reference_start))
                r_end = max(int(read.reference_end), int(mate.reference_end))
                frag_len = r_end - r_start
                if frag_len <= 0 or frag_len > max_frag:
                    continue
                chrom = str(bam.get_reference_name(read.reference_id))
                if bin_mode:
                    mid = (r_start + r_end) // 2
                    s_val, e_val = mid, mid + 1
                else:
                    s_val, e_val = r_start, r_end
                per_chrom_starts.setdefault(chrom, []).append(max(0, s_val))
                per_chrom_ends.setdefault(chrom, []).append(max(1, e_val))
        else:
            for read in bam.fetch(until_eof=True):
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_duplicate
                    or read.is_qcfail
                    or read.reference_id < 0
                    or read.mapping_quality < min_mapq
                ):
                    continue
                if pe == "first" and (not read.is_paired or not read.is_read1):
                    continue
                if pe == "second" and (not read.is_paired or not read.is_read2):
                    continue
                chrom = str(bam.get_reference_name(read.reference_id))
                if read.is_reverse:
                    r_end = int(read.reference_end) - shift
                    s_val = max(0, r_end - eff_ext)
                    e_val = max(s_val + 1, r_end)
                else:
                    r_start = int(read.reference_start) + shift
                    s_val = max(0, r_start)
                    e_val = max(s_val + 1, r_start + eff_ext)
                per_chrom_starts.setdefault(chrom, []).append(s_val)
                per_chrom_ends.setdefault(chrom, []).append(e_val)

    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, s_list in per_chrom_starts.items():
        intervals[chrom] = (
            np.asarray(s_list, dtype=np.int64),
            np.asarray(per_chrom_ends[chrom], dtype=np.int64),
        )
    return intervals, header_sizes


def _count_overlaps_sliding(
    starts: np.ndarray,
    ends: np.ndarray,
    chrom_len: int,
    width: int,
    spacing: int,
    n_win: int,
) -> tuple[np.ndarray, int]:
    """Count intervals [starts, ends) overlapping windows [m*spacing, min(L, m*spacing+width))."""
    if starts.size == 0 or n_win == 0:
        return np.zeros(n_win, dtype=np.int64), 0
    # Clamp intervals to chromosome boundaries [0, chrom_len)
    a = np.clip(starts, 0, chrom_len)
    b = np.clip(ends, 0, chrom_len)
    valid = b > a
    a = a[valid]
    b = b[valid]
    n_valid = int(a.size)
    if n_valid == 0:
        return np.zeros(n_win, dtype=np.int64), 0

    # Window m is [m * spacing, min(chrom_len, m * spacing + width)).
    # Since b <= chrom_len and a >= 0, overlap with [m*spacing, m*spacing + width) is exact:
    # m * spacing < b  AND  m * spacing + width > a
    m_min = np.maximum(0, np.floor_divide(a - width, spacing) + 1)
    m_max = np.minimum(n_win - 1, np.floor_divide(b - 1, spacing))

    hit = m_min <= m_max
    if not np.any(hit):
        return np.zeros(n_win, dtype=np.int64), n_valid

    diff = np.zeros(n_win + 1, dtype=np.int64)
    np.add.at(diff, m_min[hit], 1)
    np.add.at(diff, m_max[hit] + 1, -1)
    return np.cumsum(diff[:-1]), n_valid


def window_counts(
    bam_files: Sequence[str | Path] | None = None,
    fragments: Sequence[pd.DataFrame] | pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    width: int | None = None,
    spacing: int | None = None,
    ext: int | None = None,
    shift: int = 0,
    min_mapq: int | None = None,
    filter_min_total: int | None = None,
    bin_mode: bool = False,
    pe: Literal["none", "both", "first", "second"] = "none",
    max_frag: int = 1000,
    sample_names: Sequence[str] | None = None,
    settings: SettingsMode = "default",
) -> WindowData:
    """Count directional extended SE reads or PE fragments across sliding genomic windows.

    Clean-room implementation of ``csaw::windowCounts`` (Lun & Smyth, *NAR* 2016).

    Parameters
    ----------
    bam_files : Sequence[str | Path] | None
        Paths to BAM files read via ``pysam.AlignmentFile``. Mutually exclusive with ``fragments``.
    fragments : Sequence[pd.DataFrame] | pd.DataFrame | Mapping[str, pd.DataFrame] | None
        Per-sample read/fragment DataFrames with ``chrom``, ``pos``/``start``, and optional
        ``end``, ``strand``, ``mapq`` columns. If a single DataFrame has a ``sample`` column,
        it is split by unique sample names in order of appearance.
    chrom_sizes : Mapping[str, int] | None
        Chromosome sizes ``{chrom: length}``. Extracted from BAM headers or inferred from
        fragment coordinates when omitted.
    width : int | None
        Window width in bp. Defaults to ``10000`` when ``bin_mode=True``, else ``150`` under
        ``settings="default"`` or ``50`` under ``settings="R_default"``.
    spacing : int | None
        Step between adjacent window starts in bp. Defaults to ``width`` when ``bin_mode=True``,
        else ``50``.
    ext : int | None
        Directional 3' read extension length for single-end reads. Defaults to ``1`` when
        ``bin_mode=True`` (5'-end binning, avoiding double-counting across bins), else ``100``.
    shift : int
        Directional 5' shift applied before extension (e.g., Tn5 shift).
    min_mapq : int | None
        Minimum MAPQ filter. Defaults to ``20`` under ``settings="default"`` and ``0`` under
        ``settings="R_default"``.
    filter_min_total : int | None
        Minimum count sum across all libraries required to retain a window. Defaults to ``0``
        when ``bin_mode=True``, else ``10``.
    bin_mode : bool
        If ``True``, tile non-overlapping genomic bins (``spacing = width``, ``ext = 1``,
        ``filter_min_total = 0`` unless explicitly overridden) for background/TMM estimation.
    pe : {"none", "both", "first", "second"}
        Paired-end read handling mode matching ``csaw::readParam(pe=...)``.
    max_frag : int
        Maximum allowed paired-end template length when ``pe="both"``.
    sample_names : Sequence[str] | None
        Optional sample identifiers.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    WindowData
        Sliding-window coordinates, filtered count matrix, and pre-filter library ``totals``.
    """
    validate_settings(settings)
    if (bam_files is None) == (fragments is None):
        raise ValueError("Provide exactly one of 'bam_files' or 'fragments'.")
    if pe not in ("none", "both", "first", "second"):
        raise ValueError(f"Unknown pe mode '{pe}'. Must be 'none', 'both', 'first', or 'second'.")

    if bin_mode:
        eff_width = 10000 if width is None else int(width)
        eff_spacing = eff_width if spacing is None else int(spacing)
        eff_ext = 1 if ext is None else int(ext)
        eff_filter = 0 if filter_min_total is None else int(filter_min_total)
    else:
        if settings == "default":
            eff_width = 150 if width is None else int(width)
        else:
            eff_width = 50 if width is None else int(width)
        eff_spacing = 50 if spacing is None else int(spacing)
        eff_ext = 100 if ext is None else int(ext)
        eff_filter = 10 if filter_min_total is None else int(filter_min_total)

    if min_mapq is None:
        eff_mapq = 20 if settings == "default" else 0
    else:
        eff_mapq = int(min_mapq)

    if eff_width <= 0 or eff_spacing <= 0:
        raise ValueError(f"width ({eff_width}) and spacing ({eff_spacing}) must be positive.")

    sample_intervals: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    inferred_sizes: dict[str, int] = {}

    if bam_files is not None:
        bam_list = list(bam_files)
        if len(bam_list) == 0:
            raise ValueError("bam_files cannot be empty.")
        names = (
            [str(s) for s in sample_names]
            if sample_names is not None
            else [Path(p).stem for p in bam_list]
        )
        for p in bam_list:
            ivs, hdr = _extract_intervals_from_bam(
                p,
                ext=eff_ext,
                shift=shift,
                min_mapq=eff_mapq,
                bin_mode=bin_mode,
                pe=pe,
                max_frag=max_frag,
            )
            sample_intervals.append(ivs)
            for c, sz in hdr.items():
                inferred_sizes[c] = max(inferred_sizes.get(c, 0), sz)
    else:
        if isinstance(fragments, Mapping):
            names_default = [str(k) for k in fragments.keys()]
            df_list = list(fragments.values())
        elif isinstance(fragments, pd.DataFrame):
            if "sample" in fragments.columns:
                uniq_samples = list(pd.unique(fragments["sample"]))
                names_default = [str(s) for s in uniq_samples]
                df_list = [fragments[fragments["sample"] == s] for s in uniq_samples]
            else:
                names_default = ["sample_1"]
                df_list = [fragments]
        else:
            df_list = list(fragments)
            names_default = [f"sample_{i + 1}" for i in range(len(df_list))]

        if len(df_list) == 0:
            raise ValueError("fragments cannot be empty.")
        names = [str(s) for s in sample_names] if sample_names is not None else names_default

        for df in df_list:
            ivs = _extract_intervals_from_df(
                df,
                ext=eff_ext,
                shift=shift,
                min_mapq=eff_mapq,
                bin_mode=bin_mode,
                pe=pe,
                max_frag=max_frag,
            )
            sample_intervals.append(ivs)
            for c, (_, e_arr) in ivs.items():
                if e_arr.size > 0:
                    inferred_sizes[c] = max(inferred_sizes.get(c, 0), int(np.max(e_arr)) + eff_spacing)

    if len(names) != len(sample_intervals):
        raise ValueError(
            f"sample_names length ({len(names)}) does not match sample count ({len(sample_intervals)})."
        )

    eff_chrom_sizes: dict[str, int] = (
        {str(k): int(v) for k, v in chrom_sizes.items()}
        if chrom_sizes is not None
        else inferred_sizes
    )

    n_samples = len(sample_intervals)
    totals = np.zeros(n_samples, dtype=np.int64)
    win_chroms: list[np.ndarray] = []
    win_starts: list[np.ndarray] = []
    win_ends: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []

    for chrom, chrom_len in eff_chrom_sizes.items():
        if chrom_len <= 0:
            continue
        # Mirrors csaw::windowCounts tiling starts 0, spacing, 2*spacing, ... < chrom_len
        starts_grid = np.arange(0, chrom_len, eff_spacing, dtype=np.int64)
        n_win = int(starts_grid.size)
        if n_win == 0:
            continue
        ends_grid = np.minimum(chrom_len, starts_grid + eff_width)

        block = np.zeros((n_win, n_samples), dtype=np.int64)
        for j, ivs in enumerate(sample_intervals):
            if chrom in ivs:
                s_arr, e_arr = ivs[chrom]
                col_counts, n_valid = _count_overlaps_sliding(
                    s_arr, e_arr, chrom_len, eff_width, eff_spacing, n_win
                )
                block[:, j] = col_counts
                totals[j] += n_valid

        win_chroms.append(np.full(n_win, chrom, dtype=object))
        win_starts.append(starts_grid)
        win_ends.append(ends_grid)
        count_blocks.append(block)

    if len(count_blocks) == 0:
        empty_df = pd.DataFrame(columns=["chrom", "start", "end", "width"])
        return WindowData(
            windows=empty_df,
            counts=np.zeros((0, n_samples), dtype=np.int64),
            totals=totals,
            sample_names=names,
            width=eff_width,
            spacing=eff_spacing,
            ext=eff_ext,
            bin_mode=bin_mode,
            settings=settings,
        )

    all_chroms = np.concatenate(win_chroms)
    all_starts = np.concatenate(win_starts)
    all_ends = np.concatenate(win_ends)
    all_counts = np.vstack(count_blocks)

    if eff_filter > 0:
        keep = np.sum(all_counts, axis=1) >= eff_filter
        all_chroms = all_chroms[keep]
        all_starts = all_starts[keep]
        all_ends = all_ends[keep]
        all_counts = all_counts[keep, :]

    windows_df = pd.DataFrame(
        {
            "chrom": all_chroms.astype(str),
            "start": all_starts,
            "end": all_ends,
            "width": all_ends - all_starts,
        }
    )

    return WindowData(
        windows=windows_df,
        counts=all_counts,
        totals=totals,
        sample_names=names,
        width=eff_width,
        spacing=eff_spacing,
        ext=eff_ext,
        bin_mode=bin_mode,
        settings=settings,
    )
