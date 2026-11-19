"""Sequence GC%, CpG observed/expected, and 16-dinucleotide composition."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._validation import SettingsMode, validate_settings

DNA_BASES: tuple[str, str, str, str] = ("A", "C", "G", "T")
DINUCLEOTIDES: tuple[str, ...] = tuple(
    b1 + b2 for b1 in DNA_BASES for b2 in DNA_BASES
)

_ASCII_LUT = np.full(256, 4, dtype=np.uint8)
_ASCII_LUT[ord("A")] = 0
_ASCII_LUT[ord("a")] = 0
_ASCII_LUT[ord("C")] = 1
_ASCII_LUT[ord("c")] = 1
_ASCII_LUT[ord("G")] = 2
_ASCII_LUT[ord("g")] = 2
_ASCII_LUT[ord("T")] = 3
_ASCII_LUT[ord("t")] = 3
_ASCII_LUT[ord("U")] = 3
_ASCII_LUT[ord("u")] = 3


def encode_dna_sequence(seq: str) -> np.ndarray:
    """Encode a DNA string to uint8 array (A=0, C=1, G=2, T=3, other=4)."""
    raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    return _ASCII_LUT[raw]


def calc_sequence_composition(
    sequences: Sequence[str] | pd.Series | np.ndarray,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Compute per-sequence ``gc_frac``, ``cpg_oe``, and 16-dinucleotide frequencies.

    - ``settings="default"``: ``cpg_oe = (n_CG / N_dinuc) / ((n_C / N_mono) * (n_G / N_mono))``
      (probability ratio definition).
    - ``settings="R_default"``: ``cpg_oe = (n_CG * N_mono) / (n_C * n_G)``
      (exact ``monaLisa`` / Gardiner-Garden & Frommer sequence-width definition).
    """
    validate_settings(settings)

    seq_list = [str(s) for s in sequences]
    n_seqs = len(seq_list)
    if n_seqs == 0:
        raise ValueError("sequences must be non-empty.")

    gc_frac = np.zeros(n_seqs, dtype=np.float64)
    cpg_oe = np.zeros(n_seqs, dtype=np.float64)
    dinuc_freqs = np.zeros((n_seqs, 16), dtype=np.float64)

    cg_idx = DINUCLEOTIDES.index("CG")

    for i, seq in enumerate(seq_list):
        codes = encode_dna_sequence(seq)
        valid_mono = codes < 4
        n_mono = int(np.sum(valid_mono))
        if n_mono == 0:
            continue

        mono_counts = np.bincount(codes[valid_mono], minlength=4)
        n_c = int(mono_counts[1])
        n_g = int(mono_counts[2])
        gc_frac[i] = (n_c + n_g) / n_mono

        if codes.size >= 2:
            b1 = codes[:-1]
            b2 = codes[1:]
            valid_di = (b1 < 4) & (b2 < 4)
            n_di = int(np.sum(valid_di))
            if n_di > 0:
                di_codes = (b1[valid_di] << 2) | b2[valid_di]
                di_counts = np.bincount(di_codes, minlength=16).astype(np.float64)
                dinuc_freqs[i, :] = di_counts / n_di
                n_cg = float(di_counts[cg_idx])
                if n_c > 0 and n_g > 0:
                    if settings == "R_default":
                        cpg_oe[i] = (n_cg * n_mono) / (n_c * n_g)
                    else:
                        p_c = n_c / n_mono
                        p_g = n_g / n_mono
                        cpg_oe[i] = (n_cg / n_di) / (p_c * p_g)

    cols = ["gc_frac", "cpg_oe", *DINUCLEOTIDES]
    data = np.column_stack([gc_frac, cpg_oe, dinuc_freqs])
    index = sequences.index if isinstance(sequences, pd.Series) else pd.RangeIndex(n_seqs)
    return pd.DataFrame(data, index=index, columns=cols)
