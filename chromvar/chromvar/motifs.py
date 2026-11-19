"""
Strand-aware GENCODE/Ensembl GTF TSS extraction, nearest-TSS mapping, and generalized
Position Weight Matrix (PWM) promoter sequence scanning & variant disruption scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

_GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


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
    pseudocount: float = 0.01,
    bg_probs: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> np.ndarray:
    """Convert a `(K, 4)` Position Frequency Matrix (`A, C, G, T`) into a log2-odds PWM."""
    arr = np.asarray(pfm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected PFM of shape (K, 4) for [A, C, G, T], got {arr.shape}.")
    bg = np.asarray(bg_probs, dtype=np.float64).reshape(1, 4)
    probs = (arr + pseudocount) / (arr.sum(axis=1, keepdims=True) + 4.0 * pseudocount)
    return np.log2(probs / bg)


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
    r"""Compute $\Delta S_{\text{PWM}} = S_{\text{PWM}}(\text{Alt}) - S_{\text{PWM}}(\text{Ref})$."""
    s_ref, _ = score_kmer_strand_max(ref_kmer, pwm)
    s_alt, _ = score_kmer_strand_max(alt_kmer, pwm)
    return float(s_alt - s_ref)


def scan_sequence_pwm(
    sequence: str,
    pwm: np.ndarray,
    min_log2_odds: float = 6.0,
    sequence_id: str = "seq",
    tss_position_in_seq: int | None = None,
    dedup_window_bp: int = 8,
) -> list[PWMMotifHit]:
    """Scan a DNA sequence with any `(K, 4)` log2-odds PWM on both strands."""
    seq = sequence.upper()
    k = pwm.shape[0]
    raw_hits: list[PWMMotifHit] = []
    for i in range(max(0, len(seq) - k + 1)):
        win = seq[i : i + k]
        sc, strand = score_kmer_strand_max(win, pwm)
        if sc >= min_log2_odds:
            rel = (i - tss_position_in_seq) if tss_position_in_seq is not None else None
            raw_hits.append(PWMMotifHit(sequence_id, i, rel, win, strand, sc))

    # Non-maximum suppression within `dedup_window_bp`
    raw_hits.sort(key=lambda h: -h.log2_odds_score)
    kept: list[PWMMotifHit] = []
    for h in raw_hits:
        if all(abs(h.offset_5prime - kh.offset_5prime) >= dedup_window_bp for kh in kept):
            kept.append(h)
    kept.sort(key=lambda h: h.offset_5prime)
    return kept
