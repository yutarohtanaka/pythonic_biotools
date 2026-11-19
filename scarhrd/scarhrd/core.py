"""
Core genomic scar Homologous Recombination Deficiency (HRD) scoring algorithms:
- `hrd_loh`: Abkevich et al. (Br J Cancer 2012)
- `tai`: Birkbak et al. (Cancer Discov 2012)
- `lst`: Popova et al. (Cancer Res 2012)
- Integrated NGS `scarHRD` workflow: Sztupinszki et al. (NPJ Breast Cancer 2018)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .reference import ChromArmReference, canonical_hrd_chrom, get_genome_reference

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass
class _Seg:
    start: int
    end: int
    major_cn: int
    minor_cn: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def total_cn(self) -> int:
        return self.major_cn + self.minor_cn

    @property
    def is_loh(self) -> bool:
        return self.minor_cn == 0 and self.total_cn >= 1

    @property
    def is_ai(self) -> bool:
        return self.total_cn >= 1 and self.major_cn != self.minor_cn


@dataclass(frozen=True)
class ScarHRDResult:
    """Genomic scar HRD score breakdown and clinical HRD status."""

    sample_id: str
    genome: str
    hrd_loh: int
    tai: int
    lst: int
    hrd_sum: int
    ploidy: float
    lst_ploidy_adjusted: float
    hrd_sum_ploidy_adjusted: float
    hrd_positive: bool
    hrd_threshold: float
    per_chrom_breakdown: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return a 1-row summary DataFrame of HRD component and sum scores."""
        return pd.DataFrame(
            [
                {
                    "sample_id": self.sample_id,
                    "hrd_loh": self.hrd_loh,
                    "tai": self.tai,
                    "lst": self.lst,
                    "hrd_sum": self.hrd_sum,
                    "ploidy": self.ploidy,
                    "lst_ploidy_adjusted": self.lst_ploidy_adjusted,
                    "hrd_sum_ploidy_adjusted": self.hrd_sum_ploidy_adjusted,
                    "hrd_positive": self.hrd_positive,
                    "hrd_threshold": self.hrd_threshold,
                    "genome": self.genome,
                }
            ]
        )


def _merge_identical_states(segs: list[_Seg], max_gap_bp: int) -> list[_Seg]:
    """Merge adjacent segments sharing `(major_cn, minor_cn)` within `max_gap_bp`."""
    if not segs:
        return []
    merged: list[_Seg] = [
        _Seg(segs[0].start, segs[0].end, segs[0].major_cn, segs[0].minor_cn)
    ]
    for s in segs[1:]:
        prev = merged[-1]
        if (
            prev.major_cn == s.major_cn
            and prev.minor_cn == s.minor_cn
            and (s.start - prev.end) <= max_gap_bp
        ):
            prev.end = max(prev.end, s.end)
        else:
            merged.append(_Seg(s.start, s.end, s.major_cn, s.minor_cn))
    return merged


def _smooth_arm_segments(
    segs: list[_Seg],
    *,
    min_segment_bp: int,
    max_merge_gap_bp: int,
) -> list[_Seg]:
    """
    Iteratively smooth/filter out micro-segments shorter than `min_segment_bp`
    (3 Mb in Popova et al., 2012 / scarHRD) and merge newly adjacent identical
    allele-specific copy-number states.
    """
    cur = _merge_identical_states(segs, max_merge_gap_bp)
    if min_segment_bp <= 0:
        return cur

    while cur:
        lengths = [s.length for s in cur]
        min_idx = int(np.argmin(lengths))
        if lengths[min_idx] >= min_segment_bp:
            break
        cur.pop(min_idx)
        cur = _merge_identical_states(cur, max_merge_gap_bp)

    return cur


def _is_ai_r_default(seg: _Seg, modal_ploidy: int) -> bool:
    """Allelic imbalance predicate in sztup/scarHRD v0.1.1 calc.ai_new.R."""
    if seg.total_cn < 1:
        return False
    if modal_ploidy % 2 == 1 and seg.total_cn == modal_ploidy and seg.minor_cn != 0:
        return False
    return seg.major_cn != seg.minor_cn


def _smooth_arm_segments_r_default(
    segs: list[_Seg],
    *,
    min_segment_bp: int,
    max_lst_gap_bp: int,
) -> list[list[_Seg]]:
    """
    Match sztup/scarHRD v0.1.1 calc.lst.R per-arm smoothing:
    1. Split arm into contiguous sub-blocks where initial gap >= max_lst_gap_bp.
    2. Within each block, merge adjacent identical (major_cn, minor_cn) states and
       iteratively pop the FIRST segment < min_segment_bp in left-to-right order.
    """
    if not segs:
        return []

    blocks: list[list[_Seg]] = [[_Seg(segs[0].start, segs[0].end, segs[0].major_cn, segs[0].minor_cn)]]
    for s in segs[1:]:
        prev = blocks[-1][-1]
        if (s.start - prev.end) >= max_lst_gap_bp:
            blocks.append([_Seg(s.start, s.end, s.major_cn, s.minor_cn)])
        else:
            blocks[-1].append(_Seg(s.start, s.end, s.major_cn, s.minor_cn))

    def _shrink_identical(block: list[_Seg]) -> list[_Seg]:
        if not block:
            return []
        out: list[_Seg] = [_Seg(block[0].start, block[0].end, block[0].major_cn, block[0].minor_cn)]
        for item in block[1:]:
            last = out[-1]
            if last.major_cn == item.major_cn and last.minor_cn == item.minor_cn:
                last.end = max(last.end, item.end)
            else:
                out.append(_Seg(item.start, item.end, item.major_cn, item.minor_cn))
        return out

    smoothed_blocks: list[list[_Seg]] = []
    for blk in blocks:
        cur = _shrink_identical(blk)
        if min_segment_bp > 0:
            while cur:
                first_short = next(
                    (idx for idx, seg in enumerate(cur) if seg.length < min_segment_bp),
                    None,
                )
                if first_short is None:
                    break
                cur.pop(first_short)
                cur = _shrink_identical(cur)
        if cur:
            smoothed_blocks.append(cur)
    return smoothed_blocks


def _score_single_chromosome_r_default(
    raw_segs: list[_Seg],
    ref: ChromArmReference,
    *,
    est_ploidy: float,
    min_loh_bp: int,
    min_lst_bp: int,
    min_segment_bp: int,
    max_lst_gap_bp: int,
    centromere_margin_bp: int,
    min_tai_filter_bp: int = 1_000_000,
) -> tuple[int, int, int]:
    """
    Exact sztup/scarHRD v0.1.1 (calc.hrd.R, calc.ai_new.R, calc.lst.R) per-chromosome scoring.
    """
    p_raw, q_raw, _ = _split_chrom_into_arms(raw_segs, ref)

    # ------------------------------------------------------------------
    # 1. HRD-LOH (calc.hrd.R): > 15 Mb LOH without <3 Mb LST pre-smoothing
    # ------------------------------------------------------------------
    p_loh_runs = _merge_boolean_runs(p_raw, "is_loh", max_lst_gap_bp)
    q_loh_runs = _merge_boolean_runs(q_raw, "is_loh", max_lst_gap_bp)

    combined_loh_lengths: list[int] = []
    bridged_cen_loh = (
        len(p_loh_runs) > 0
        and len(q_loh_runs) > 0
        and p_loh_runs[-1][2]
        and q_loh_runs[0][2]
        and p_loh_runs[-1][1] >= (ref.centromere_start - centromere_margin_bp)
        and q_loh_runs[0][0] <= (ref.centromere_end + centromere_margin_bp)
    )
    if bridged_cen_loh:
        for s_r, e_r, flag in p_loh_runs[:-1]:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
        bridged_len = (p_loh_runs[-1][1] - p_loh_runs[-1][0]) + (
            q_loh_runs[0][1] - q_loh_runs[0][0]
        )
        combined_loh_lengths.append(bridged_len)
        for s_r, e_r, flag in q_loh_runs[1:]:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
    else:
        for s_r, e_r, flag in p_loh_runs:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
        for s_r, e_r, flag in q_loh_runs:
            if flag:
                combined_loh_lengths.append(e_r - s_r)

    all_chr_segs = q_raw if ref.is_acrocentric and not p_raw else (p_raw + q_raw)
    total_loh_bp = sum(combined_loh_lengths)
    all_called_are_loh = len(all_chr_segs) > 0 and all(s.is_loh for s in all_chr_segs)
    is_whole_chr_loh = (
        all_called_are_loh and total_loh_bp >= 0.75 * ref.callable_length
    ) or (total_loh_bp >= 0.90 * ref.callable_length and len(combined_loh_lengths) == 1)

    if is_whole_chr_loh:
        loh_count = 0
    else:
        loh_count = sum(1 for ln in combined_loh_lengths if ln > min_loh_bp)

    # ------------------------------------------------------------------
    # 2. TAI (calc.ai_new.R): filter >= 1 Mb without <3 Mb smoothing,
    #    telomeric iff end <= cen_start (p) or start >= cen_end (q),
    #    modal odd-ploidy neutrality.
    # ------------------------------------------------------------------
    modal_ploidy = int(round(est_ploidy))
    clipped_sorted: list[_Seg] = []
    for s in raw_segs:
        st = max(0, s.start)
        en = min(ref.length, s.end) if s.end <= ref.length + 5_000_000 else s.end
        if en - st >= min_tai_filter_bp:
            clipped_sorted.append(_Seg(st, en, s.major_cn, s.minor_cn))
    clipped_sorted.sort(key=lambda x: (x.start, x.end))

    chr_ai_runs: list[tuple[int, int, bool]] = []
    for s in clipped_sorted:
        flag = _is_ai_r_default(s, modal_ploidy)
        if not chr_ai_runs:
            chr_ai_runs.append((s.start, s.end, flag))
            continue
        p_st, p_en, p_flag = chr_ai_runs[-1]
        bridges_cen = (
            p_en >= (ref.centromere_start - centromere_margin_bp)
            and s.start <= (ref.centromere_end + centromere_margin_bp)
        )
        if flag == p_flag and ((s.start - p_en) <= max_lst_gap_bp or bridges_cen):
            chr_ai_runs[-1] = (p_st, max(p_en, s.end), p_flag)
        else:
            chr_ai_runs.append((s.start, s.end, flag))

    tai_count = 0
    if not ref.is_acrocentric and len(chr_ai_runs) > 0:
        ps, pe, p_flag = chr_ai_runs[0]
        if p_flag and pe <= ref.centromere_start:
            tai_count += 1

    if len(chr_ai_runs) > 0:
        qs, qe, q_flag = chr_ai_runs[-1]
        acrocentric_whole_chr = (
            ref.is_acrocentric
            and len(chr_ai_runs) == 1
            and qs <= (ref.centromere_end + centromere_margin_bp)
            and (qe - qs) >= 0.75 * ref.q_arm_length
        )
        if q_flag and qs >= ref.centromere_end and not acrocentric_whole_chr:
            tai_count += 1

    # ------------------------------------------------------------------
    # 3. LST (calc.lst.R): left-to-right <3 Mb pop, strict gap < 3 Mb
    # ------------------------------------------------------------------
    lst_count = 0
    for arm_raw in (p_raw, q_raw):
        blocks = _smooth_arm_segments_r_default(
            arm_raw,
            min_segment_bp=min_segment_bp,
            max_lst_gap_bp=max_lst_gap_bp,
        )
        for blk in blocks:
            for k in range(len(blk) - 1):
                left_s = blk[k]
                right_s = blk[k + 1]
                if (
                    left_s.length >= min_lst_bp
                    and right_s.length >= min_lst_bp
                    and (right_s.start - left_s.end) < max_lst_gap_bp
                    and (
                        left_s.major_cn != right_s.major_cn
                        or left_s.minor_cn != right_s.minor_cn
                    )
                ):
                    lst_count += 1

    return loh_count, tai_count, lst_count


def _merge_boolean_runs(
    segs: list[_Seg],
    predicate_name: str,
    max_gap_bp: int,
) -> list[tuple[int, int, bool]]:
    """Collapse contiguous segments sharing boolean `is_loh` or `is_ai` status."""
    if not segs:
        return []
    first_flag = bool(getattr(segs[0], predicate_name))
    runs: list[tuple[int, int, bool]] = [(segs[0].start, segs[0].end, first_flag)]
    for s in segs[1:]:
        flag = bool(getattr(s, predicate_name))
        p_start, p_end, p_flag = runs[-1]
        if flag == p_flag and (s.start - p_end) <= max_gap_bp:
            runs[-1] = (p_start, max(p_end, s.end), p_flag)
        else:
            runs.append((s.start, s.end, flag))
    return runs


def _split_chrom_into_arms(
    raw_segs: list[_Seg],
    ref: ChromArmReference,
) -> tuple[list[_Seg], list[_Seg], bool]:
    """
    Split chromosome segments at centromere `[centromere_start, centromere_end]`.
    Also detects if any raw AI segment directly spanned across the centromere.
    """
    p_segs: list[_Seg] = []
    q_segs: list[_Seg] = []
    raw_cross_cen_ai = False

    cs = ref.centromere_start
    ce = ref.centromere_end
    chr_len = ref.length

    for s in raw_segs:
        st = max(0, s.start)
        en = min(chr_len, s.end) if s.end <= chr_len + 5_000_000 else s.end
        if en <= st:
            continue

        if st < cs and en > ce and s.is_ai:
            raw_cross_cen_ai = True

        if st < cs:
            p_end = min(en, cs)
            if p_end > st:
                p_segs.append(_Seg(st, p_end, s.major_cn, s.minor_cn))
        if en > ce:
            q_start = max(st, ce)
            if en > q_start:
                q_segs.append(_Seg(q_start, en, s.major_cn, s.minor_cn))

    p_segs.sort(key=lambda x: (x.start, x.end))
    q_segs.sort(key=lambda x: (x.start, x.end))
    return p_segs, q_segs, raw_cross_cen_ai


def _score_single_chromosome(
    raw_segs: list[_Seg],
    ref: ChromArmReference,
    *,
    min_loh_bp: int,
    min_tai_bp: int,
    min_lst_bp: int,
    min_segment_bp: int,
    max_lst_gap_bp: int,
    telomere_margin_bp: int,
    centromere_margin_bp: int,
) -> tuple[int, int, int]:
    """Compute `(hrd_loh, tai, lst)` on a single chromosome."""
    p_raw, q_raw, raw_cross_cen_ai = _split_chrom_into_arms(raw_segs, ref)

    p_smooth = _smooth_arm_segments(
        p_raw, min_segment_bp=min_segment_bp, max_merge_gap_bp=max_lst_gap_bp
    )
    q_smooth = _smooth_arm_segments(
        q_raw, min_segment_bp=min_segment_bp, max_merge_gap_bp=max_lst_gap_bp
    )

    # ------------------------------------------------------------------
    # 1. LST (Popova et al., 2012): per arm independently, never across centromere
    # ------------------------------------------------------------------
    lst_count = 0
    for arm_segs in (p_smooth, q_smooth):
        for k in range(len(arm_segs) - 1):
            left_s = arm_segs[k]
            right_s = arm_segs[k + 1]
            if (
                left_s.length >= min_lst_bp
                and right_s.length >= min_lst_bp
                and (right_s.start - left_s.end) <= max_lst_gap_bp
                and (left_s.major_cn != right_s.major_cn or left_s.minor_cn != right_s.minor_cn)
            ):
                lst_count += 1

    # ------------------------------------------------------------------
    # 2. HRD-LOH (Abkevich et al., 2012): LOH > 15 Mb, NOT whole chromosome
    # ------------------------------------------------------------------
    p_loh_runs = _merge_boolean_runs(p_smooth, "is_loh", max_lst_gap_bp)
    q_loh_runs = _merge_boolean_runs(q_smooth, "is_loh", max_lst_gap_bp)

    # Link pericentromeric / whole-chromosome LOH bridging p-arm centromere to q-arm centromere
    combined_loh_lengths: list[int] = []
    bridged_cen_loh = (
        len(p_loh_runs) > 0
        and len(q_loh_runs) > 0
        and p_loh_runs[-1][2]
        and q_loh_runs[0][2]
        and p_loh_runs[-1][1] >= (ref.centromere_start - centromere_margin_bp)
        and q_loh_runs[0][0] <= (ref.centromere_end + centromere_margin_bp)
    )

    if bridged_cen_loh:
        for s_r, e_r, flag in p_loh_runs[:-1]:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
        bridged_len = (p_loh_runs[-1][1] - p_loh_runs[-1][0]) + (
            q_loh_runs[0][1] - q_loh_runs[0][0]
        )
        combined_loh_lengths.append(bridged_len)
        for s_r, e_r, flag in q_loh_runs[1:]:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
    else:
        for s_r, e_r, flag in p_loh_runs:
            if flag:
                combined_loh_lengths.append(e_r - s_r)
        for s_r, e_r, flag in q_loh_runs:
            if flag:
                combined_loh_lengths.append(e_r - s_r)

    all_chr_segs = (q_smooth if ref.is_acrocentric and not p_smooth else (p_smooth + q_smooth))
    total_loh_bp = sum(combined_loh_lengths)
    all_called_are_loh = len(all_chr_segs) > 0 and all(s.is_loh for s in all_chr_segs)

    # Whole-chromosome LOH check (monosomy or uniparental disomy across callable chromosome)
    is_whole_chr_loh = (
        all_called_are_loh and total_loh_bp >= 0.75 * ref.callable_length
    ) or (total_loh_bp >= 0.90 * ref.callable_length and len(combined_loh_lengths) == 1)

    if is_whole_chr_loh:
        loh_count = 0
    else:
        loh_count = sum(1 for ln in combined_loh_lengths if ln > min_loh_bp)

    # ------------------------------------------------------------------
    # 3. TAI (Birkbak et al., 2012): Telomeric AI >= 11 Mb, NOT crossing centromere
    # ------------------------------------------------------------------
    tai_count = 0
    p_ai_runs = _merge_boolean_runs(p_smooth, "is_ai", max_lst_gap_bp)
    q_ai_runs = _merge_boolean_runs(q_smooth, "is_ai", max_lst_gap_bp)

    p_touches_cen_ai = (
        len(p_ai_runs) > 0
        and p_ai_runs[-1][2]
        and p_ai_runs[-1][1] >= (ref.centromere_start - centromere_margin_bp)
    )
    q_touches_cen_ai = (
        len(q_ai_runs) > 0
        and q_ai_runs[0][2]
        and q_ai_runs[0][0] <= (ref.centromere_end + centromere_margin_bp)
    )
    crosses_cen_ai = raw_cross_cen_ai or (p_touches_cen_ai and q_touches_cen_ai)

    # Evaluate p-arm telomere (non-acrocentric, or acrocentric only if p-arm explicitly called)
    if not ref.is_acrocentric and len(p_ai_runs) > 0:
        ps, pe, p_flag = p_ai_runs[0]
        p_is_single_crossing_run = len(p_ai_runs) == 1 and crosses_cen_ai
        if (
            p_flag
            and ps <= telomere_margin_bp
            and (pe - ps) >= min_tai_bp
            and not p_is_single_crossing_run
        ):
            tai_count += 1

    # Evaluate q-arm telomere
    if len(q_ai_runs) > 0:
        qs, qe, q_flag = q_ai_runs[-1]
        if ref.is_acrocentric:
            # On acrocentric chromosomes (13, 14, 15, 21, 22), whole-q-arm AI extending
            # into the centromere represents whole-chromosome gain/loss, not telomeric AI.
            q_crosses_cen = raw_cross_cen_ai or (len(q_ai_runs) == 1 and q_touches_cen_ai)
        else:
            q_crosses_cen = len(q_ai_runs) == 1 and crosses_cen_ai

        if (
            q_flag
            and qe >= (ref.length - telomere_margin_bp)
            and (qe - qs) >= min_tai_bp
            and not q_crosses_cen
        ):
            tai_count += 1

    return loh_count, tai_count, lst_count


def _parse_allele_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(segments, pd.DataFrame):
        segments = pd.DataFrame(segments)
    if len(segments) == 0:
        raise ValueError("segments must contain at least 1 allele-specific CN segment.")

    req = {"chrom", "start", "end", "minor_cn"}
    missing = req - set(segments.columns)
    if missing:
        raise ValueError(f"segments is missing required columns: {sorted(missing)}.")

    has_total = "total_cn" in segments.columns
    has_major = "major_cn" in segments.columns
    if not has_total and not has_major:
        raise ValueError("segments must provide either 'total_cn' or 'major_cn' alongside 'minor_cn'.")

    check_cols = ["chrom", "start", "end", "minor_cn"] + (
        ["total_cn"] if has_total else ["major_cn"]
    )
    if segments[check_cols].isna().any().any():
        raise ValueError("segments contains NaN values in required coordinate or copy-number columns.")

    starts = np.asarray(segments["start"], dtype=np.int64)
    ends = np.asarray(segments["end"], dtype=np.int64)
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("Every segment must satisfy 0 <= start < end.")

    minor_f = np.asarray(segments["minor_cn"], dtype=np.float64)
    if np.any(minor_f < 0) or not np.all(np.isclose(minor_f, np.rint(minor_f))):
        raise ValueError("minor_cn must consist of non-negative integers.")
    minor = np.rint(minor_f).astype(np.int64)

    if has_total:
        total_f = np.asarray(segments["total_cn"], dtype=np.float64)
        if np.any(total_f < 0) or not np.all(np.isclose(total_f, np.rint(total_f))):
            raise ValueError("total_cn must consist of non-negative integers.")
        total = np.rint(total_f).astype(np.int64)
        major = total - minor
    else:
        major_f = np.asarray(segments["major_cn"], dtype=np.float64)
        if np.any(major_f < 0) or not np.all(np.isclose(major_f, np.rint(major_f))):
            raise ValueError("major_cn must consist of non-negative integers.")
        major = np.rint(major_f).astype(np.int64)
        total = major + minor

    if np.any(minor > major):
        raise ValueError(
            "minor_cn must not exceed major_cn (i.e. require 2 * minor_cn <= total_cn)."
        )

    canon_chroms = [canonical_hrd_chrom(c) for c in segments["chrom"]]
    out = pd.DataFrame(
        {
            "chrom": canon_chroms,
            "start": starts,
            "end": ends,
            "total_cn": total,
            "major_cn": major,
            "minor_cn": minor,
        }
    )
    if "sample_id" in segments.columns:
        out["sample_id"] = [str(s) for s in segments["sample_id"]]
    return out


def calculate_hrd(
    segments: pd.DataFrame,
    *,
    genome: str = "GRCh38",
    ploidy: float | None = None,
    sample_id: str = "Sample_1",
    include_chr_x: bool | None = None,
    min_loh_bp: int = 15_000_000,
    min_tai_bp: int = 11_000_000,
    min_lst_bp: int = 10_000_000,
    min_segment_bp: int = 3_000_000,
    max_lst_gap_bp: int = 3_000_000,
    telomere_margin_bp: int = 10_000_000,
    centromere_margin_bp: int = 5_000_000,
    ploidy_lst_slope: float = 2.5,
    use_ploidy_adjusted_sum: bool = False,
    hrd_threshold: float = 42.0,
    settings: SettingsMode = "default",
) -> ScarHRDResult:
    """
    Compute `scarHRD` genomic scar scores (`hrd_loh`, `tai`, `lst`, and `hrd_sum`)
    from allele-specific copy-number segments (Sztupinszki et al., 2018).
    """
    _validate_settings(settings)
    if ploidy is not None and ploidy <= 0.0:
        raise ValueError(f"ploidy must be > 0, got {ploidy}.")
    if min_loh_bp < 0 or min_tai_bp < 0 or min_lst_bp < 0 or min_segment_bp < 0:
        raise ValueError("Size thresholds must be non-negative.")

    eff_include_chr_x = (
        (settings != "R_default") if include_chr_x is None else bool(include_chr_x)
    )
    ref_map = get_genome_reference(genome, include_chr_x=eff_include_chr_x)
    parsed = _parse_allele_segments(segments)

    if "sample_id" in parsed.columns:
        uniq_samples = parsed["sample_id"].unique()
        if len(uniq_samples) > 1:
            raise ValueError(
                f"segments contains {len(uniq_samples)} sample_ids; pass a single sample to "
                "`calculate_hrd` or use `calculate_hrd_cohort` for multi-sample tables."
            )
        sample_id = str(uniq_samples[0])

    valid = parsed.loc[parsed["chrom"].isin(ref_map.keys())].copy()
    if len(valid) == 0:
        chrom_desc = "chr1-chr22, chrX" if eff_include_chr_x else "chr1-chr22"
        raise ValueError(f"No segments mapped to evaluated chromosomes ({chrom_desc}).")

    seg_lengths = (valid["end"] - valid["start"]).to_numpy(dtype=np.float64)
    est_ploidy = (
        float(ploidy)
        if ploidy is not None
        else float(np.sum(seg_lengths * valid["total_cn"].to_numpy(dtype=np.float64)) / np.sum(seg_lengths))
    )

    segs_by_chrom: dict[str, list[_Seg]] = {}
    for row in valid.itertuples(index=False):
        segs_by_chrom.setdefault(str(row.chrom), []).append(
            _Seg(int(row.start), int(row.end), int(row.major_cn), int(row.minor_cn))
        )

    chrom_rows: list[dict[str, object]] = []
    total_loh = 0
    total_tai = 0
    total_lst = 0

    for ch, ref in ref_map.items():
        ch_segs = segs_by_chrom.get(ch, [])
        if not ch_segs:
            chrom_rows.append({"chrom": ch, "hrd_loh": 0, "tai": 0, "lst": 0})
            continue
        if settings == "R_default":
            c_loh, c_tai, c_lst = _score_single_chromosome_r_default(
                ch_segs,
                ref,
                est_ploidy=est_ploidy,
                min_loh_bp=min_loh_bp,
                min_lst_bp=min_lst_bp,
                min_segment_bp=min_segment_bp,
                max_lst_gap_bp=max_lst_gap_bp,
                centromere_margin_bp=centromere_margin_bp,
            )
        else:
            c_loh, c_tai, c_lst = _score_single_chromosome(
                ch_segs,
                ref,
                min_loh_bp=min_loh_bp,
                min_tai_bp=min_tai_bp,
                min_lst_bp=min_lst_bp,
                min_segment_bp=min_segment_bp,
                max_lst_gap_bp=max_lst_gap_bp,
                telomere_margin_bp=telomere_margin_bp,
                centromere_margin_bp=centromere_margin_bp,
            )
        total_loh += c_loh
        total_tai += c_tai
        total_lst += c_lst
        chrom_rows.append({"chrom": ch, "hrd_loh": c_loh, "tai": c_tai, "lst": c_lst})

    hrd_sum = int(total_loh + total_tai + total_lst)
    if settings == "R_default":
        lst_adj = float(max(0.0, float(total_lst) - 15.5 * est_ploidy))
    else:
        lst_adj = float(
            max(0.0, float(total_lst) - ploidy_lst_slope * (est_ploidy - 2.0))
        )
    hrd_sum_adj = float(total_loh + total_tai) + lst_adj

    score_for_call = hrd_sum_adj if use_ploidy_adjusted_sum else float(hrd_sum)
    is_pos = bool(score_for_call >= hrd_threshold)

    return ScarHRDResult(
        sample_id=sample_id,
        genome=genome,
        hrd_loh=total_loh,
        tai=total_tai,
        lst=total_lst,
        hrd_sum=hrd_sum,
        ploidy=est_ploidy,
        lst_ploidy_adjusted=lst_adj,
        hrd_sum_ploidy_adjusted=hrd_sum_adj,
        hrd_positive=is_pos,
        hrd_threshold=float(hrd_threshold),
        per_chrom_breakdown=pd.DataFrame(chrom_rows),
    )


def calculate_hrd_cohort(
    segments: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    genome: str = "GRCh38",
    include_chr_x: bool | None = None,
    min_loh_bp: int = 15_000_000,
    min_tai_bp: int = 11_000_000,
    min_lst_bp: int = 10_000_000,
    min_segment_bp: int = 3_000_000,
    max_lst_gap_bp: int = 3_000_000,
    telomere_margin_bp: int = 10_000_000,
    centromere_margin_bp: int = 5_000_000,
    ploidy_lst_slope: float = 2.5,
    use_ploidy_adjusted_sum: bool = False,
    hrd_threshold: float = 42.0,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Compute scarHRD scores across a multi-sample allele-specific segment cohort."""
    _validate_settings(settings)
    if not isinstance(segments, pd.DataFrame):
        segments = pd.DataFrame(segments)
    if sample_col not in segments.columns:
        raise ValueError(f"segments must contain sample column '{sample_col}'.")

    dfs: list[pd.DataFrame] = []
    for sid, grp in segments.groupby(sample_col, sort=False):
        sub = grp.copy()
        if sample_col != "sample_id":
            sub["sample_id"] = sub[sample_col]
        res = calculate_hrd(
            sub,
            genome=genome,
            sample_id=str(sid),
            include_chr_x=include_chr_x,
            min_loh_bp=min_loh_bp,
            min_tai_bp=min_tai_bp,
            min_lst_bp=min_lst_bp,
            min_segment_bp=min_segment_bp,
            max_lst_gap_bp=max_lst_gap_bp,
            telomere_margin_bp=telomere_margin_bp,
            centromere_margin_bp=centromere_margin_bp,
            ploidy_lst_slope=ploidy_lst_slope,
            use_ploidy_adjusted_sum=use_ploidy_adjusted_sum,
            hrd_threshold=hrd_threshold,
            settings=settings,
        )
        dfs.append(res.to_dataframe())

    if not dfs:
        raise ValueError("segments contains no samples.")
    return pd.concat(dfs, ignore_index=True)
