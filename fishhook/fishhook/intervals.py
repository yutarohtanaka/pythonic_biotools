"""
Sweep-line genomic interval intersection, callable-territory coverage, and
per-sample capped mutation counting for fishHook.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

_STD_CHROMS = {str(i) for i in range(1, 26)} | {"X", "Y", "M", "MT"}


def canonical_chrom(chrom: str) -> str:
    """Normalize '1' <-> 'chr1' while leaving custom contig names unchanged."""
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        suffix = c[3:].upper()
        if suffix == "MT":
            suffix = "M"
        if suffix in _STD_CHROMS:
            return f"chr{suffix}"
        return c
    upper = c.upper()
    if upper == "MT":
        return "chrM"
    if upper in _STD_CHROMS:
        return f"chr{upper}"
    return c


def parse_regions(regions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Validate and extract (chrom, start, end, region_id) from target regions."""
    if not isinstance(regions, pd.DataFrame):
        regions = pd.DataFrame(regions)
    if len(regions) == 0:
        raise ValueError("regions must contain at least 1 genomic interval.")

    req = {"chrom", "start", "end"}
    missing = req - set(regions.columns)
    if missing:
        raise ValueError(f"regions is missing required columns: {sorted(missing)}.")

    if regions[["chrom", "start", "end"]].isna().any().any():
        raise ValueError("regions contains NaN values in chrom, start, or end.")

    chrom_raw = np.asarray([str(c).strip() for c in regions["chrom"]], dtype=object)
    start = np.asarray(regions["start"], dtype=np.int64)
    end = np.asarray(regions["end"], dtype=np.int64)

    if np.any(start < 0):
        raise ValueError("Region start coordinates must be >= 0.")
    if np.any(end <= start):
        raise ValueError("Every target region must have end > start.")

    if "region_id" in regions.columns:
        if regions["region_id"].isna().any():
            raise ValueError("region_id column must not contain NaN.")
        region_ids = [str(r) for r in regions["region_id"]]
    else:
        region_ids = [f"{chrom_raw[i]}:{start[i]}-{end[i]}" for i in range(len(regions))]

    return chrom_raw, start, end, region_ids


def parse_mutations(
    mutations: pd.DataFrame,
    *,
    pos_coord_system: Literal["zero_based", "one_based"] = "zero_based",
    require_sample_id: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Parse somatic mutation positions `(chrom, pos)` or intervals `(chrom, start, end)`
    into 0-based half-open `[start, end)` intervals.
    """
    if pos_coord_system not in ("zero_based", "one_based"):
        raise ValueError(
            f"Invalid pos_coord_system '{pos_coord_system}'. Expected 'zero_based' or 'one_based'."
        )
    if not isinstance(mutations, pd.DataFrame):
        mutations = pd.DataFrame(mutations)

    has_sample = "sample_id" in mutations.columns
    if require_sample_id and not has_sample:
        raise ValueError(
            "max_muts_per_sample_per_tile requires a 'sample_id' column in mutations."
        )

    if len(mutations) == 0:
        empty_s = np.array([], dtype=object)
        empty_i = np.array([], dtype=np.int64)
        return empty_s, empty_i, empty_i, (empty_s if has_sample else None)

    if "chrom" not in mutations.columns:
        raise ValueError("mutations must contain a 'chrom' column.")

    has_interval = "start" in mutations.columns and "end" in mutations.columns
    has_pos = "pos" in mutations.columns
    if not has_interval and not has_pos:
        raise ValueError(
            "mutations must contain either ('chrom', 'pos') or ('chrom', 'start', 'end')."
        )

    chrom_col = mutations["chrom"]
    if chrom_col.isna().any():
        raise ValueError("mutations 'chrom' column contains NaN.")
    chrom = np.asarray([canonical_chrom(c) for c in chrom_col], dtype=object)

    if has_interval:
        if mutations[["start", "end"]].isna().any().any():
            raise ValueError("mutations start/end columns contain NaN.")
        start = np.asarray(mutations["start"], dtype=np.int64).copy()
        end = np.asarray(mutations["end"], dtype=np.int64).copy()
        # Handle 1-bp closed point representation where start == end
        eq_mask = start == end
        if np.any(eq_mask):
            if pos_coord_system == "one_based":
                start[eq_mask] -= 1
            else:
                end[eq_mask] += 1
    else:
        if mutations["pos"].isna().any():
            raise ValueError("mutations 'pos' column contains NaN.")
        pos = np.asarray(mutations["pos"], dtype=np.int64)
        if pos_coord_system == "one_based":
            if np.any(pos < 1):
                raise ValueError("1-based mutation 'pos' coordinates must be >= 1.")
            start = pos - 1
            end = pos.copy()
        else:
            if np.any(pos < 0):
                raise ValueError("0-based mutation 'pos' coordinates must be >= 0.")
            start = pos.copy()
            end = pos + 1

    if np.any(start < 0) or np.any(end <= start):
        raise ValueError("Invalid mutation intervals: require 0 <= start < end.")

    samples: np.ndarray | None = None
    if has_sample:
        if mutations["sample_id"].isna().any():
            raise ValueError("mutations 'sample_id' column contains NaN.")
        samples = np.asarray([str(s) for s in mutations["sample_id"]], dtype=object)

    return chrom, start, end, samples


def merge_eligible_by_chrom(
    eligible_intervals: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sweep-line union of overlapping/adjacent eligible territory intervals per chromosome.

    Returns dict mapping canonical chromosome -> (starts, ends, prefix_lengths).
    """
    if not isinstance(eligible_intervals, pd.DataFrame):
        eligible_intervals = pd.DataFrame(eligible_intervals)
    if len(eligible_intervals) == 0:
        raise ValueError("eligible_intervals must not be empty when provided.")

    req = {"chrom", "start", "end"}
    missing = req - set(eligible_intervals.columns)
    if missing:
        raise ValueError(f"eligible_intervals is missing required columns: {sorted(missing)}.")
    if eligible_intervals[["chrom", "start", "end"]].isna().any().any():
        raise ValueError("eligible_intervals contains NaN values.")

    chroms = np.asarray([canonical_chrom(c) for c in eligible_intervals["chrom"]], dtype=object)
    starts = np.asarray(eligible_intervals["start"], dtype=np.int64)
    ends = np.asarray(eligible_intervals["end"], dtype=np.int64)
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("eligible_intervals must satisfy 0 <= start < end.")

    merged: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ch in dict.fromkeys(chroms.tolist()):
        mask = chroms == ch
        s_ch = starts[mask]
        e_ch = ends[mask]
        order = np.argsort(s_ch, kind="mergesort")
        s_ch = s_ch[order]
        e_ch = e_ch[order]

        out_s: list[int] = [int(s_ch[0])]
        out_e: list[int] = [int(e_ch[0])]
        for idx in range(1, s_ch.size):
            si = int(s_ch[idx])
            ei = int(e_ch[idx])
            if si <= out_e[-1]:
                if ei > out_e[-1]:
                    out_e[-1] = ei
            else:
                out_s.append(si)
                out_e.append(ei)

        arr_s = np.asarray(out_s, dtype=np.int64)
        arr_e = np.asarray(out_e, dtype=np.int64)
        pref = np.zeros(arr_s.size + 1, dtype=np.int64)
        pref[1:] = np.cumsum(arr_e - arr_s)
        merged[ch] = (arr_s, arr_e, pref)

    return merged


def _vectorized_covered_bp(
    q_starts: np.ndarray,
    q_ends: np.ndarray,
    e_starts: np.ndarray,
    e_ends: np.ndarray,
    e_pref: np.ndarray,
) -> np.ndarray:
    """
    Exact intersection lengths |R_i cap E| for an array of query intervals [q_starts, q_ends)
    against disjoint sorted eligible intervals [e_starts, e_ends) in O(Q log E) time.
    """
    idx_left = np.searchsorted(e_ends, q_starts, side="right")
    idx_right = np.searchsorted(e_starts, q_ends, side="left")

    n_q = q_starts.size
    covered = np.zeros(n_q, dtype=np.int64)

    has_ov = idx_left < idx_right
    if not np.any(has_ov):
        return covered

    single = has_ov & ((idx_left + 1) == idx_right)
    if np.any(single):
        l_s = idx_left[single]
        covered[single] = np.minimum(q_ends[single], e_ends[l_s]) - np.maximum(
            q_starts[single], e_starts[l_s]
        )

    multi = has_ov & ((idx_left + 1) < idx_right)
    if np.any(multi):
        l_m = idx_left[multi]
        r_m = idx_right[multi] - 1
        left_part = np.minimum(q_ends[multi], e_ends[l_m]) - np.maximum(
            q_starts[multi], e_starts[l_m]
        )
        right_part = np.minimum(q_ends[multi], e_ends[r_m]) - np.maximum(
            q_starts[multi], e_starts[r_m]
        )
        mid_part = e_pref[r_m] - e_pref[l_m + 1]
        covered[multi] = left_part + mid_part + right_part

    return covered


def _scalar_overlaps_eligible(
    os: int,
    oe: int,
    e_starts: np.ndarray,
    e_ends: np.ndarray,
) -> bool:
    """Return True iff [os, oe) intersects sorted disjoint [e_starts, e_ends)."""
    idx = int(np.searchsorted(e_ends, os, side="right"))
    return idx < e_starts.size and int(e_starts[idx]) < oe


def compute_territory_and_counts(
    regions: pd.DataFrame,
    mutations: pd.DataFrame,
    *,
    eligible_intervals: pd.DataFrame | None = None,
    max_muts_per_sample_per_tile: int | None = None,
    pos_coord_system: Literal["zero_based", "one_based"] = "zero_based",
    restrict_mutations_to_eligible: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Compute exact covered base pairs L_i and observed (optionally per-sample capped)
    somatic mutation counts Y_i per genomic region.
    """
    if max_muts_per_sample_per_tile is not None and max_muts_per_sample_per_tile < 1:
        raise ValueError("max_muts_per_sample_per_tile must be >= 1 when specified.")

    r_chrom_raw, r_start, r_end, r_ids = parse_regions(regions)
    r_chrom = np.asarray([canonical_chrom(c) for c in r_chrom_raw], dtype=object)

    m_chrom, m_start, m_end, m_samples = parse_mutations(
        mutations,
        pos_coord_system=pos_coord_system,
        require_sample_id=(max_muts_per_sample_per_tile is not None),
    )

    n_regions = r_start.size
    if eligible_intervals is None:
        eligible_bp = (r_end - r_start).astype(np.int64)
        elig_map = None
    else:
        elig_map = merge_eligible_by_chrom(eligible_intervals)
        eligible_bp = np.zeros(n_regions, dtype=np.int64)
        for ch in dict.fromkeys(r_chrom.tolist()):
            if ch not in elig_map:
                continue
            rmask = r_chrom == ch
            es, ee, ep = elig_map[ch]
            eligible_bp[rmask] = _vectorized_covered_bp(
                r_start[rmask], r_end[rmask], es, ee, ep
            )

    obs_counts = np.zeros(n_regions, dtype=np.int64)
    if m_start.size == 0:
        return r_chrom_raw, r_start, r_end, r_ids, eligible_bp, obs_counts

    # Index regions by chromosome using sorted starts + prefix-maximum ends
    muts_by_chrom: dict[str, np.ndarray] = {}
    for ch in dict.fromkeys(m_chrom.tolist()):
        muts_by_chrom[ch] = np.where(m_chrom == ch)[0]

    for ch in dict.fromkeys(r_chrom.tolist()):
        if ch not in muts_by_chrom:
            continue
        elig_entry = None if elig_map is None else elig_map.get(ch)
        if elig_map is not None and restrict_mutations_to_eligible and elig_entry is None:
            continue

        r_idx_ch = np.where(r_chrom == ch)[0]
        rs = r_start[r_idx_ch]
        re = r_end[r_idx_ch]
        order = np.argsort(rs, kind="mergesort")
        rs_s = rs[order]
        re_s = re[order]
        orig_r = r_idx_ch[order]
        pref_max_end = np.maximum.accumulate(re_s)

        m_idx_ch = muts_by_chrom[ch]
        ms_ch = m_start[m_idx_ch]
        me_ch = m_end[m_idx_ch]

        right_bounds = np.searchsorted(rs_s, me_ch, side="left")
        left_bounds = np.searchsorted(pref_max_end, ms_ch, side="right")

        if max_muts_per_sample_per_tile is None:
            for k in range(m_idx_ch.size):
                lb = int(left_bounds[k])
                rb = int(right_bounds[k])
                if lb >= rb:
                    continue
                ms_k = int(ms_ch[k])
                me_k = int(me_ch[k])
                for pos_in_s in range(lb, rb):
                    if int(re_s[pos_in_s]) <= ms_k:
                        continue
                    if elig_entry is not None and restrict_mutations_to_eligible:
                        os = max(int(rs_s[pos_in_s]), ms_k)
                        oe = min(int(re_s[pos_in_s]), me_k)
                        if not _scalar_overlaps_eligible(os, oe, elig_entry[0], elig_entry[1]):
                            continue
                    obs_counts[int(orig_r[pos_in_s])] += 1
        else:
            assert m_samples is not None
            samp_ch = m_samples[m_idx_ch]
            per_tile_sample: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            cap = int(max_muts_per_sample_per_tile)
            for k in range(m_idx_ch.size):
                lb = int(left_bounds[k])
                rb = int(right_bounds[k])
                if lb >= rb:
                    continue
                ms_k = int(ms_ch[k])
                me_k = int(me_ch[k])
                sid = str(samp_ch[k])
                for pos_in_s in range(lb, rb):
                    if int(re_s[pos_in_s]) <= ms_k:
                        continue
                    r_global = int(orig_r[pos_in_s])
                    if per_tile_sample[r_global][sid] >= cap:
                        continue
                    if elig_entry is not None and restrict_mutations_to_eligible:
                        os = max(int(rs_s[pos_in_s]), ms_k)
                        oe = min(int(re_s[pos_in_s]), me_k)
                        if not _scalar_overlaps_eligible(os, oe, elig_entry[0], elig_entry[1]):
                            continue
                    per_tile_sample[r_global][sid] += 1

            for r_global, sample_dict in per_tile_sample.items():
                obs_counts[r_global] = sum(min(c, cap) for c in sample_dict.values())

    return r_chrom_raw, r_start, r_end, r_ids, eligible_bp, obs_counts
