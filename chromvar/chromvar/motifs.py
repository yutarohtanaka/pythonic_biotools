"""
Strand-aware GENCODE/Ensembl GTF TSS extraction, nearest-TSS mapping, and generalized
Position Weight Matrix (PWM) promoter sequence scanning & variant disruption scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import re
from typing import Literal, Sequence

import numpy as np
import pandas as pd

SettingsMode = Literal["default", "R_default"]

_GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(f"settings must be 'default' or 'R_default', got {settings!r}.")
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class PWMMotifHit:
    """Detected strand-maximized PWM binding motif match along a sequence."""

    sequence_id: str
    offset_5prime: int
    tss_relative_coord: int | None
    matched_kmer: str
    strand: str
    log2_odds_score: float


def reverse_complement(seq: str) -> str:
    """Return the DNA reverse complement of `seq`."""
    return seq.translate(_COMPLEMENT_MAP)[::-1]


def extract_gtf_tss(gtf_path: str | Path) -> pd.DataFrame:
    """Extract strand-aware 0-based TSS coordinates for all `gene` entries in a GENCODE/Ensembl GTF."""
    p = Path(gtf_path)
    opener = gzip.open if str(p).endswith(".gz") else open
    rows = []
    with opener(p, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            attrs = dict(_GTF_ATTR.findall(c[8]))
            start0, end = int(c[3]) - 1, int(c[4])
            strand = c[6]
            tss = start0 if strand == "+" else end - 1
            gid = attrs.get("gene_id", "").split(".")[0]
            gname = attrs.get("gene_name", gid)
            gtype = attrs.get("gene_type", attrs.get("gene_biotype", "unknown"))
            rows.append((gid, gname, gtype, c[0], strand, tss, start0, end))
    return pd.DataFrame(
        rows,
        columns=["gene_id", "gene_name", "gene_type", "chrom", "strand", "tss", "start", "end"],
    )


def signed_tss_distance(chrom: str, pos: int, gene_name: str, tss_df: pd.DataFrame) -> int:
    """Signed distance from `pos` to `gene_name` TSS (positive = downstream of TSS in transcription direction)."""
    sub = tss_df.loc[tss_df["gene_name"] == gene_name]
    if sub.empty:
        raise KeyError(f"Gene {gene_name!r} not found in tss_df.")
    g = sub.iloc[0]
    if str(g["chrom"]) != str(chrom):
        raise ValueError(f"{gene_name} is on {g['chrom']}, not {chrom}.")
    d = int(pos) - int(g["tss"])
    return d if g["strand"] == "+" else -d


def nearest_tss(
    chrom: str,
    positions: Sequence[int] | np.ndarray,
    tss_df: pd.DataFrame,
    gene_types: Sequence[str] | None = ("protein_coding",),
) -> pd.DataFrame:
    """Find the nearest gene TSS on `chrom` for an array of genomic positions."""
    pos_arr = np.asarray(positions, dtype=np.int64).ravel()
    sub = tss_df.loc[tss_df["chrom"].astype(str) == str(chrom)]
    if gene_types is not None and "gene_type" in sub.columns:
        sub = sub.loc[sub["gene_type"].isin(gene_types)]
    sub = sub.sort_values("tss")
    if sub.empty:
        return pd.DataFrame({"gene_name": [None] * pos_arr.size, "distance_bp": [np.nan] * pos_arr.size})
    x = sub["tss"].to_numpy(dtype=np.int64)
    if x.size == 1:
        return pd.DataFrame({"gene_name": [sub["gene_name"].iloc[0]] * pos_arr.size, "distance_bp": pos_arr - x[0]})
    idx = np.clip(np.searchsorted(x, pos_arr), 1, x.size - 1)
    left, right = x[idx - 1], x[idx]
    best = np.where(np.abs(pos_arr - left) <= np.abs(right - pos_arr), idx - 1, idx)
    return pd.DataFrame({"gene_name": sub["gene_name"].to_numpy()[best], "distance_bp": pos_arr - x[best]})


def build_log_odds_pwm(
    pfm: np.ndarray,
    pseudocount: float | None = None,
    bg_probs: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    *,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Convert a `(K, 4)` Position Frequency Matrix (`A, C, G, T`) into a log2-odds PWM.

    - `settings='default'`: Normalizes each position's base counts/frequencies to sum to 1.0 before adding
      `pseudocount` (default `0.01`), yielding scale-invariant PWMs.
    - `settings='R_default'`: Matches Bioconductor `TFBSTools::toPWM(..., type='log2probratio', pseudocounts=0.8)`
      by adding `bg * pseudocount` (`default=0.8`) directly on the raw count scale `(pfm + bg * pc) / (N + pc)`.
    """
    _validate_settings(settings)
    arr = np.asarray(pfm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected PFM of shape (K, 4) for [A, C, G, T], got {arr.shape}.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("PFM entries must be finite and non-negative.")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each position in PFM must have positive total frequency/count.")

    bg = np.asarray(bg_probs, dtype=np.float64).reshape(1, 4)
    if np.any(~np.isfinite(bg)) or np.any(bg <= 0.0):
        raise ValueError("Background base probabilities must be finite and strictly positive.")
    bg = bg / bg.sum()

    if settings == "R_default":
        pc = 0.8 if pseudocount is None else float(pseudocount)
        if pc < 0.0 or not np.isfinite(pc):
            raise ValueError("pseudocount must be finite and >= 0.")
        probs = (arr + bg * pc) / (row_sums + pc)
        return np.log2(probs / bg)

    pc = 0.01 if pseudocount is None else float(pseudocount)
    if pc < 0.0 or not np.isfinite(pc):
        raise ValueError("pseudocount must be finite and >= 0.")
    p_norm = arr / row_sums
    probs = (p_norm + pc) / (1.0 + 4.0 * pc)
    return np.log2(probs / bg)


def pwm_threshold_from_pvalue(
    pwm: np.ndarray,
    p_cutoff: float = 5e-5,
    bg_probs: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    n_grid_bins: int = 80_000,
) -> float:
    """Compute exact background score-distribution threshold `T` with `P(S >= T) <= p_cutoff` via 1D DP (`MOODS`)."""
    if not (0.0 < p_cutoff <= 1.0) or not np.isfinite(p_cutoff):
        raise ValueError(f"p_cutoff must lie in (0, 1], got {p_cutoff!r}.")

    mat = np.asarray(pwm, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != 4 or not np.all(np.isfinite(mat)):
        raise ValueError("pwm must be a finite (K, 4) array.")

    bg = np.asarray(bg_probs, dtype=np.float64).ravel()
    if bg.size != 4 or np.any(~np.isfinite(bg)) or np.any(bg <= 0.0):
        raise ValueError("bg_probs must contain 4 strictly positive finite probabilities.")
    bg = bg / bg.sum()

    row_mins = mat.min(axis=1)
    row_maxs = mat.max(axis=1)
    min_score = float(row_mins.sum())
    max_score = float(row_maxs.sum())
    if p_cutoff >= 1.0:
        return min_score - 1e-9

    span = max_score - min_score
    if span <= 1e-12:
        return max_score + 1e-6

    k_len = mat.shape[0]
    scale = float(max(n_grid_bins, 4000)) / span
    shifted = mat - row_mins[:, None]
    offsets = np.rint(shifted * scale).astype(np.int64)

    dp = np.array([1.0], dtype=np.float64)
    cur_max = 0
    for k_idx in range(k_len):
        row_off = offsets[k_idx]
        next_max = cur_max + int(row_off.max())
        nxt = np.zeros(next_max + 1, dtype=np.float64)
        for b_idx in range(4):
            shift = int(row_off[b_idx])
            nxt[shift : shift + cur_max + 1] += bg[b_idx] * dp
        dp = nxt
        cur_max = next_max

    tail_prob = np.cumsum(dp[::-1])[::-1]
    occ = np.flatnonzero(dp > 0.0)
    passing_occ = occ[tail_prob[occ] <= p_cutoff + 1e-15]
    if passing_occ.size == 0:
        return max_score + 1e-6

    idx_cut = int(passing_occ[0])
    if idx_cut <= int(occ[0]):
        return min_score - 1e-9

    cut_pos = int(np.searchsorted(occ, idx_cut))
    prev_occ = int(occ[cut_pos - 1])
    thresh = min_score + 0.5 * float(idx_cut + prev_occ) / scale
    if idx_cut == cur_max:
        thresh = min(thresh, max_score - 1e-9)
    return float(thresh)


def score_kmer_strand_max(kmer: str, pwm: np.ndarray) -> tuple[float, str]:
    """Compute strand-maximized log2-odds PWM score for a DNA k-mer of length `pwm.shape[0]`."""
    k = pwm.shape[0]
    if len(kmer) != k:
        raise ValueError(f"k-mer length ({len(kmer)}) must match PWM length ({k}).")
    up = kmer.upper()
    rc = reverse_complement(up)
    if any(b not in _BASE_TO_IDX for b in up):
        return float("-inf"), "+"
    s_plus = sum(float(pwm[pos, _BASE_TO_IDX[b]]) for pos, b in enumerate(up))
    s_minus = sum(float(pwm[pos, _BASE_TO_IDX[b]]) for pos, b in enumerate(rc))
    return (s_plus, "+") if s_plus >= s_minus else (s_minus, "-")


def compute_variant_pwm_disruption(ref_kmer: str, alt_kmer: str, pwm: np.ndarray) -> float:
    r"""Compute $\Delta S_{\text{PWM}} = \max_w S_{\text{PWM}}(\text{Alt}_w) - \max_w S_{\text{PWM}}(\text{Ref}_w)$
    across all $k$-mer windows overlapping the variant.

    Accepts exact length-$k$ sequences or longer flanking sequences (`len >= k`, e.g. `2k - 1` bp centered
    on a SNV or indel) and evaluates all $k$-mer windows spanning the altered allele.
    """
    k = pwm.shape[0]
    ref_up = ref_kmer.upper()
    alt_up = alt_kmer.upper()
    if len(ref_up) < k or len(alt_up) < k:
        raise ValueError(
            f"ref_kmer ({len(ref_up)} bp) and alt_kmer ({len(alt_up)} bp) must have length >= PWM length ({k})."
        )

    if len(ref_up) == k and len(alt_up) == k:
        s_ref, _ = score_kmer_strand_max(ref_up, pwm)
        s_alt, _ = score_kmer_strand_max(alt_up, pwm)
        return float(s_alt - s_ref)

    if ref_up == alt_up:
        return 0.0

    # Identify common flanking prefix & suffix so only k-mer windows overlapping the variant are scored
    min_len = min(len(ref_up), len(alt_up))
    p_len = 0
    while p_len < min_len and ref_up[p_len] == alt_up[p_len]:
        p_len += 1
    s_len = 0
    while s_len < (min_len - p_len) and ref_up[len(ref_up) - 1 - s_len] == alt_up[len(alt_up) - 1 - s_len]:
        s_len += 1

    def _max_overlapping_score(seq: str) -> float:
        start_i = max(0, p_len - k + 1)
        last_i = min(len(seq) - k, max(0, len(seq) - s_len - 1))
        if last_i < start_i:
            start_i = min(start_i, len(seq) - k)
            last_i = start_i
        return max(score_kmer_strand_max(seq[i : i + k], pwm)[0] for i in range(start_i, last_i + 1))

    return float(_max_overlapping_score(alt_up) - _max_overlapping_score(ref_up))


def scan_sequence_pwm(
    sequence: str,
    pwm: np.ndarray,
    min_log2_odds: float | None = None,
    sequence_id: str = "seq",
    tss_position_in_seq: int | None = None,
    dedup_window_bp: int | None = None,
    *,
    p_cutoff: float | None = None,
    bg_probs: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    settings: SettingsMode = "default",
) -> list[PWMMotifHit]:
    """Scan a DNA sequence with any `(K, 4)` log2-odds PWM on both strands.

    When `settings='R_default'` (or when `p_cutoff` is explicitly provided), uses exact dynamic-programming
    background score-distribution p-value thresholding (`p_cutoff = 5e-5`, MOODS/`motifmatchr` convention)
    and disables NMS window deduplication (`dedup_window_bp = 0`) unless explicitly overridden.
    """
    _validate_settings(settings)
    if p_cutoff is not None:
        thresh = pwm_threshold_from_pvalue(pwm, p_cutoff=p_cutoff, bg_probs=bg_probs)
    elif min_log2_odds is not None:
        thresh = float(min_log2_odds)
    elif settings == "R_default":
        thresh = pwm_threshold_from_pvalue(pwm, p_cutoff=5e-5, bg_probs=bg_probs)
    else:
        thresh = 6.0

    win_bp = (0 if settings == "R_default" else 8) if dedup_window_bp is None else int(dedup_window_bp)

    seq = sequence.upper()
    k = pwm.shape[0]
    raw_hits: list[PWMMotifHit] = []
    for i in range(max(0, len(seq) - k + 1)):
        win = seq[i : i + k]
        sc, strand = score_kmer_strand_max(win, pwm)
        if sc >= thresh:
            rel = (i - tss_position_in_seq) if tss_position_in_seq is not None else None
            raw_hits.append(PWMMotifHit(sequence_id, i, rel, win, strand, sc))

    if win_bp <= 1:
        return raw_hits

    # Non-maximum suppression within `win_bp`
    raw_hits.sort(key=lambda h: -h.log2_odds_score)
    kept: list[PWMMotifHit] = []
    for h in raw_hits:
        if all(abs(h.offset_5prime - kh.offset_5prime) >= win_bp for kh in kept):
            kept.append(h)
    kept.sort(key=lambda h: h.offset_5prime)
    return kept
