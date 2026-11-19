"""Position Frequency/Weight Matrix conversion and vectorized sequence scanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from ._validation import SettingsMode, validate_settings
from .composition import DNA_BASES, encode_dna_sequence


def _coerce_4xl_matrix(mat: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Coerce a PFM/PWM array or DataFrame into shape ``(4, L)`` ordered ``A, C, G, T``."""
    if isinstance(mat, pd.DataFrame):
        idx_upper = [str(x).upper() for x in mat.index]
        col_upper = [str(x).upper() for x in mat.columns]
        if set(idx_upper) == set(DNA_BASES) and mat.shape[0] == 4:
            reordered = mat.copy()
            reordered.index = idx_upper
            arr = reordered.loc[list(DNA_BASES)].to_numpy(dtype=np.float64, copy=True)
        elif set(col_upper) == set(DNA_BASES) and mat.shape[1] == 4:
            reordered = mat.copy()
            reordered.columns = col_upper
            arr = reordered[list(DNA_BASES)].to_numpy(dtype=np.float64, copy=True).T
        else:
            arr = mat.to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(mat, dtype=np.float64).copy()

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Motif matrix must be a non-empty 2D array or DataFrame.")
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    if arr.shape[0] != 4:
        raise ValueError(
            f"Motif matrix must have 4 nucleotide rows (A, C, G, T); got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Motif matrix must contain only finite entries.")
    return arr


def pfm_to_pwm(
    pfm: np.ndarray | pd.DataFrame | Mapping[str, np.ndarray | pd.DataFrame],
    pseudocount: float = 1.0,
    bg_prob: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    settings: SettingsMode = "default",
) -> np.ndarray | dict[str, np.ndarray]:
    """Convert a Position Frequency Matrix (or dict of PFMs) to log2-odds PWM(s).

    Matches ``TFBSTools::toPWM(..., type='log2probratio')``:
    ``W[b, l] = log2(((C[b, l] + pseudocount * bg[b]) / (sum_b' C[b', l] + pseudocount)) / bg[b])``.
    """
    validate_settings(settings)

    if pseudocount < 0.0:
        raise ValueError(f"pseudocount must be >= 0, got {pseudocount}.")
    bg = np.asarray(bg_prob, dtype=np.float64)
    if bg.shape != (4,) or np.any(bg <= 0.0) or not np.all(np.isfinite(bg)):
        raise ValueError("bg_prob must be 4 strictly positive finite background probabilities.")
    bg = bg / np.sum(bg)

    if isinstance(pfm, Mapping):
        return {
            str(name): pfm_to_pwm(
                mat, pseudocount=pseudocount, bg_prob=bg, settings=settings
            )  # type: ignore[misc]
            for name, mat in pfm.items()
        }

    counts = _coerce_4xl_matrix(pfm)
    if np.any(counts < 0.0):
        raise ValueError("PFM entries must be non-negative.")

    col_sums = np.sum(counts, axis=0, keepdims=True)
    denom = col_sums + pseudocount
    if np.any(denom <= 0.0):
        raise ValueError("PFM column sums + pseudocount must be strictly positive.")

    probs = (counts + pseudocount * bg[:, np.newaxis]) / denom
    if np.any(probs <= 0.0):
        raise ValueError(
            "Zero nucleotide probability encountered; use a positive pseudocount."
        )
    return np.log2(probs / bg[:, np.newaxis])


def _normalize_pwms_input(
    pwms: Mapping[str, np.ndarray | pd.DataFrame]
    | Sequence[np.ndarray | pd.DataFrame]
    | np.ndarray
    | pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Standardize PWM inputs into ``dict[motif_name, (4, L) ndarray]``."""
    if isinstance(pwms, Mapping):
        if len(pwms) == 0:
            raise ValueError("pwms mapping must not be empty.")
        return {str(k): _coerce_4xl_matrix(v) for k, v in pwms.items()}

    if isinstance(pwms, (np.ndarray, pd.DataFrame)):
        arr = np.asarray(pwms)
        if arr.ndim == 2:
            return {"motif_0": _coerce_4xl_matrix(pwms)}

    pwm_list = list(pwms)  # type: ignore[arg-type]
    if len(pwm_list) == 0:
        raise ValueError("pwms sequence must not be empty.")
    return {f"motif_{j}": _coerce_4xl_matrix(m) for j, m in enumerate(pwm_list)}


def _resolve_pwm_threshold(pwm: np.ndarray, min_score: str | float | int) -> float:
    """Resolve ``Biostrings::matchPWM`` relative string (``'80%'``) or numeric threshold."""
    if isinstance(min_score, (int, float, np.floating, np.integer)):
        return float(min_score)
    if isinstance(min_score, str):
        s = min_score.strip()
        if s.endswith("%"):
            pct = float(s[:-1]) / 100.0
            if not (0.0 <= pct <= 1.0):
                raise ValueError(f"Percentage min_score out of [0%, 100%]: {min_score!r}")
            max_s = float(np.sum(np.max(pwm, axis=0)))
            min_s = float(np.sum(np.min(pwm, axis=0)))
            if max_s > 0.0:
                return pct * max_s
            return min_s + pct * (max_s - min_s)
        return float(s)
    raise TypeError(f"Unsupported min_score type: {type(min_score)!r}")


@dataclass(frozen=True)
class MotifScanResult:
    """Region × motif hit count matrix paired with genomic/sequence hit coordinates.

    Supports both attribute access (``res.counts``, ``res.hits``) and 2-tuple
    unpacking (``counts_df, hits_df = res``).
    """

    counts: pd.DataFrame
    hits: pd.DataFrame

    @property
    def hit_matrix(self) -> pd.DataFrame:
        return self.counts

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.counts
        yield self.hits


def scan_sequences_pwm(
    sequences: Sequence[str] | pd.Series | np.ndarray,
    pwms: Mapping[str, np.ndarray | pd.DataFrame]
    | Sequence[np.ndarray | pd.DataFrame]
    | np.ndarray
    | pd.DataFrame,
    min_score: str | float = "80%",
    both_strands: bool = True,
    settings: SettingsMode = "default",
) -> MotifScanResult:
    """Vectorized forward + reverse-complement PWM sequence scanner.

    Concatenates encoded sequences separated by a ``-inf`` boundary sentinel so
    each motif is scanned across all sequences simultaneously in pure NumPy
    without crossing sequence boundaries.
    """
    validate_settings(settings)

    seq_list = [str(s) for s in sequences]
    n_seqs = len(seq_list)
    if n_seqs == 0:
        raise ValueError("sequences must be non-empty.")

    pwm_dict = _normalize_pwms_input(pwms)
    motif_names = list(pwm_dict.keys())
    n_motifs = len(motif_names)

    # Encode all sequences once into a single flat uint8 array with boundary token 5
    encoded_chunks: list[np.ndarray] = []
    seq_starts = np.empty(n_seqs, dtype=np.int64)
    cursor = 0
    sentinel = np.array([5], dtype=np.uint8)

    for idx_s, s in enumerate(seq_list):
        seq_starts[idx_s] = cursor
        enc = encode_dna_sequence(s)
        encoded_chunks.append(enc)
        encoded_chunks.append(sentinel)
        cursor += enc.size + 1

    flat_seq = np.concatenate(encoded_chunks) if encoded_chunks else np.empty(0, dtype=np.uint8)
    total_len = flat_seq.size

    count_mat = np.zeros((n_seqs, n_motifs), dtype=np.int64)
    hit_seq_chunks: list[np.ndarray] = []
    hit_motif_chunks: list[np.ndarray] = []
    hit_start_chunks: list[np.ndarray] = []
    hit_end_chunks: list[np.ndarray] = []
    hit_strand_chunks: list[np.ndarray] = []
    hit_score_chunks: list[np.ndarray] = []

    for m_idx, m_name in enumerate(motif_names):
        w_fwd = pwm_dict[m_name]
        width = w_fwd.shape[1]
        if total_len < width:
            continue

        thresh = _resolve_pwm_threshold(w_fwd, min_score)
        n_win = total_len - width + 1

        # Extended lookup table: rows 0..3 = A,C,G,T; row 4 = N (0.0); row 5 = boundary (-1e9)
        strand_specs: list[tuple[str, np.ndarray]] = [("+", w_fwd)]
        if both_strands:
            strand_specs.append(("-", w_fwd[::-1, ::-1]))

        for strand_sym, w_mat in strand_specs:
            w_ext = np.zeros((6, width), dtype=np.float64)
            w_ext[:4, :] = w_mat
            w_ext[4, :] = 0.0
            w_ext[5, :] = -1e9

            scores = np.zeros(n_win, dtype=np.float64)
            for pos_l in range(width):
                scores += w_ext[flat_seq[pos_l : pos_l + n_win], pos_l]

            hit_flat_pos = np.flatnonzero(scores >= thresh - 1e-12)
            if hit_flat_pos.size == 0:
                continue

            seq_idx_arr = np.searchsorted(seq_starts, hit_flat_pos, side="right") - 1
            local_start = hit_flat_pos - seq_starts[seq_idx_arr]
            local_end = local_start + width

            count_mat[:, m_idx] += np.bincount(seq_idx_arr, minlength=n_seqs)

            hit_seq_chunks.append(seq_idx_arr)
            hit_motif_chunks.append(np.full(hit_flat_pos.size, m_name, dtype=object))
            hit_start_chunks.append(local_start)
            hit_end_chunks.append(local_end)
            hit_strand_chunks.append(np.full(hit_flat_pos.size, strand_sym, dtype=object))
            hit_score_chunks.append(scores[hit_flat_pos])

    seq_index = (
        sequences.index if isinstance(sequences, pd.Series) else pd.RangeIndex(n_seqs)
    )
    counts_df = pd.DataFrame(count_mat, index=seq_index, columns=motif_names)

    if hit_seq_chunks:
        hits_df = pd.DataFrame(
            {
                "seq_idx": np.concatenate(hit_seq_chunks),
                "motif": np.concatenate(hit_motif_chunks),
                "start": np.concatenate(hit_start_chunks),
                "end": np.concatenate(hit_end_chunks),
                "strand": np.concatenate(hit_strand_chunks),
                "score": np.concatenate(hit_score_chunks),
            }
        ).sort_values(
            by=["seq_idx", "start", "motif", "strand"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        hits_df = pd.DataFrame(
            {
                "seq_idx": pd.Series(dtype=np.int64),
                "motif": pd.Series(dtype=object),
                "start": pd.Series(dtype=np.int64),
                "end": pd.Series(dtype=np.int64),
                "strand": pd.Series(dtype=object),
                "score": pd.Series(dtype=np.float64),
            }
        )

    return MotifScanResult(counts=counts_df, hits=hits_df)
