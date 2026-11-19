"""
Extraction of 96 SBS + 17 Indel + 19 SV CHORD mutational contexts (`extract_sigs_chord`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pysam

from .channels import (
    ALL_CHORD_CHANNELS,
    INDEL_CHANNELS,
    SBS96_CHANNELS,
    SV_CHANNELS,
    classify_indel_sequence,
    classify_sv_channel,
    normalize_sbs96_context,
)
from .config import SettingsMode, validate_settings


class _FastaAccessor:
    """Unified genomic sequence fetcher supporting `pysam.FastaFile`, paths, or dicts."""

    def __init__(self, ref_fasta: str | Path | pysam.FastaFile | dict[str, str] | None) -> None:
        self._owns_handle = False
        self._fasta: pysam.FastaFile | None = None
        self._dict: dict[str, str] | None = None
        if ref_fasta is None:
            return
        if isinstance(ref_fasta, dict):
            self._dict = {str(k): str(v).upper() for k, v in ref_fasta.items()}
        elif isinstance(ref_fasta, pysam.FastaFile):
            self._fasta = ref_fasta
        elif isinstance(ref_fasta, (str, Path)):
            self._fasta = pysam.FastaFile(str(ref_fasta))
            self._owns_handle = True
        else:
            raise ValueError(
                "`ref_fasta` must be a path, `pysam.FastaFile`, `dict[chrom, seq]`, or `None`."
            )

    def close(self) -> None:
        if self._owns_handle and self._fasta is not None:
            self._fasta.close()

    @property
    def available(self) -> bool:
        return self._fasta is not None or self._dict is not None

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """0-based half-open `[start0, end0)` fetch with chr-prefix fallback."""
        s0 = max(0, int(start0))
        e0 = max(s0, int(end0))
        if e0 == s0:
            return ""
        ch = str(chrom)
        alt_ch = ch[3:] if ch.startswith("chr") else f"chr{ch}"

        if self._dict is not None:
            seq = self._dict.get(ch)
            if seq is None:
                seq = self._dict.get(alt_ch, "")
            return seq[s0 : min(e0, len(seq))]

        if self._fasta is not None:
            refs = set(self._fasta.references)
            target = ch if ch in refs else (alt_ch if alt_ch in refs else None)
            if target is None:
                return ""
            ref_len = self._fasta.get_reference_length(target)
            if s0 >= ref_len:
                return ""
            return self._fasta.fetch(target, s0, min(e0, ref_len)).upper()
        return ""


def _resolve_sample_col(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    for cand in ("sample", "sample_id", "Sample", "SAMPLE"):
        if cand in df.columns:
            return cand
    return None


def _extract_snv_counts(
    snv_df: pd.DataFrame,
    fasta: _FastaAccessor,
    counts: dict[str, np.ndarray],
    sample_col: str | None,
    default_sample: str,
) -> None:
    sbs_idx = {ch: idx for idx, ch in enumerate(SBS96_CHANNELS)}
    cols_lower = {c.lower(): c for c in snv_df.columns}

    ch_col = None
    for cand in ("channel", "sbs_channel"):
        if cand in cols_lower:
            ch_col = cols_lower[cand]
            break

    ctx_col = None
    for cand in ("context", "trinuc", "trinucleotide", "ref_context"):
        if cand in cols_lower:
            ctx_col = cols_lower[cand]
            break

    ref_col = cols_lower.get("ref") or cols_lower.get("ref_allele")
    alt_col = cols_lower.get("alt") or cols_lower.get("alt_allele")
    chrom_col = cols_lower.get("chrom") or cols_lower.get("chr")
    pos_col = cols_lower.get("pos") or cols_lower.get("start") or cols_lower.get("position")

    for row in snv_df.itertuples(index=False):
        row_dict = row._asdict()
        sid = str(row_dict[sample_col]) if sample_col is not None else default_sample
        vec = counts[sid]

        if ch_col is not None:
            val = str(row_dict[ch_col]).strip()
            if val in sbs_idx:
                vec[sbs_idx[val]] += 1
                continue

        if ctx_col is not None:
            raw_ctx = str(row_dict[ctx_col]).strip()
            if raw_ctx in sbs_idx:
                vec[sbs_idx[raw_ctx]] += 1
                continue
        else:
            raw_ctx = ""

        if ref_col is None or alt_col is None:
            continue
        ref = str(row_dict[ref_col]).strip().upper()
        alt = str(row_dict[alt_col]).strip().upper()
        if len(ref) != 1 or len(alt) != 1:
            continue

        if not raw_ctx and fasta.available and chrom_col is not None and pos_col is not None:
            pos1 = int(row_dict[pos_col])
            raw_ctx = fasta.fetch(str(row_dict[chrom_col]), pos1 - 2, pos1 + 1)

        if raw_ctx:
            norm = normalize_sbs96_context(ref, alt, raw_ctx)
            if norm is not None and norm in sbs_idx:
                vec[sbs_idx[norm]] += 1


def _extract_indel_counts(
    indel_df: pd.DataFrame,
    fasta: _FastaAccessor,
    counts: dict[str, np.ndarray],
    sample_col: str | None,
    default_sample: str,
) -> None:
    offset = len(SBS96_CHANNELS)
    indel_idx = {ch: offset + idx for idx, ch in enumerate(INDEL_CHANNELS)}
    cols_lower = {c.lower(): c for c in indel_df.columns}

    ch_col = cols_lower.get("channel") or cols_lower.get("indel_channel")
    itype_col = cols_lower.get("indel_type") or cols_lower.get("type")
    mh_col = cols_lower.get("mh_len") or cols_lower.get("bimh") or cols_lower.get("mh")
    rep_col = cols_lower.get("rep_len") or cols_lower.get("repeat_len")
    ilen_col = cols_lower.get("indel_len") or cols_lower.get("length")
    ref_col = cols_lower.get("ref") or cols_lower.get("ref_allele")
    alt_col = cols_lower.get("alt") or cols_lower.get("alt_allele")
    lf_col = cols_lower.get("left_flank") or cols_lower.get("upstream")
    rf_col = cols_lower.get("right_flank") or cols_lower.get("downstream")
    chrom_col = cols_lower.get("chrom") or cols_lower.get("chr")
    pos_col = cols_lower.get("pos") or cols_lower.get("start") or cols_lower.get("position")

    for row in indel_df.itertuples(index=False):
        row_dict = row._asdict()
        sid = str(row_dict[sample_col]) if sample_col is not None else default_sample
        vec = counts[sid]

        if ch_col is not None:
            c_str = str(row_dict[ch_col]).strip()
            if c_str in indel_idx:
                vec[indel_idx[c_str]] += 1
                continue

        if itype_col is not None:
            t_str = str(row_dict[itype_col]).strip()
            if t_str in indel_idx:
                vec[indel_idx[t_str]] += 1
                continue

        # Precomputed mh_len / rep_len / indel_type path when flanking sequences are not needed
        has_flanks = (lf_col is not None or rf_col is not None) or (
            fasta.available and chrom_col is not None and pos_col is not None
        )
        has_precomputed = mh_col is not None or rep_col is not None or itype_col is not None

        if has_precomputed and not (ref_col is not None and alt_col is not None and has_flanks):
            t_low = str(row_dict[itype_col]).strip().lower() if itype_col is not None else ""
            if ref_col is not None and alt_col is not None and not t_low:
                r_s = str(row_dict[ref_col])
                a_s = str(row_dict[alt_col])
                is_del = len(r_s) >= len(a_s)
                raw_len = abs(len(r_s) - len(a_s))
            else:
                is_del = not t_low.startswith("ins")
                raw_len = 1

            if ilen_col is not None and pd.notna(row_dict[ilen_col]):
                raw_len = max(1, int(abs(float(row_dict[ilen_col]))))
            cap_l = min(max(1, raw_len), 5)

            mh_val = (
                int(float(row_dict[mh_col]))
                if mh_col is not None and pd.notna(row_dict[mh_col])
                else 0
            )
            rep_val = (
                int(float(row_dict[rep_col]))
                if rep_col is not None and pd.notna(row_dict[rep_col])
                else 0
            )

            if is_del:
                if "rep" in t_low or (rep_val >= 1 and mh_val == 0 and "mh" not in t_low):
                    ch_name = f"del.rep.len.{min(max(1, raw_len if ilen_col else rep_val), 5)}"
                elif "mh" in t_low or mh_val >= 1:
                    ch_name = f"del.mh.bimh.{min(max(1, mh_val), 5)}"
                else:
                    ch_name = "del.none"
            else:
                if "rep" in t_low or rep_val >= 1:
                    ch_name = f"ins.rep.len.{min(max(1, raw_len if ilen_col else rep_val), 5)}"
                else:
                    ch_name = "ins.none"

            vec[indel_idx[ch_name]] += 1
            continue

        # Sequence-based exact classification
        if ref_col is None or alt_col is None:
            continue
        ref = str(row_dict[ref_col]).strip().upper()
        alt = str(row_dict[alt_col]).strip().upper()
        if ref == alt:
            continue

        lf = (
            str(row_dict[lf_col]).strip().upper()
            if lf_col is not None and pd.notna(row_dict[lf_col])
            else ""
        )
        rf = (
            str(row_dict[rf_col]).strip().upper()
            if rf_col is not None and pd.notna(row_dict[rf_col])
            else ""
        )

        if not lf and not rf and fasta.available and chrom_col is not None and pos_col is not None:
            pos1 = int(row_dict[pos_col])
            ch = str(row_dict[chrom_col])
            # 1-based VCF coordinate `pos1` corresponds to 0-based `pos1 - 1`
            # `ref` spans `[pos1 - 1, pos1 - 1 + len(ref))`
            flank_pad = max(50, 3 * max(len(ref), len(alt)))
            lf = fasta.fetch(ch, max(0, pos1 - 1 - flank_pad), pos1 - 1)
            rf = fasta.fetch(ch, pos1 - 1 + len(ref), pos1 - 1 + len(ref) + flank_pad)

        ch_name_opt = classify_indel_sequence(ref, alt, left_flank=lf, right_flank=rf)
        if ch_name_opt is not None and ch_name_opt in indel_idx:
            vec[indel_idx[ch_name_opt]] += 1


def _extract_sv_counts(
    sv_df: pd.DataFrame,
    counts: dict[str, np.ndarray],
    sample_col: str | None,
    default_sample: str,
) -> None:
    offset = len(SBS96_CHANNELS) + len(INDEL_CHANNELS)
    sv_idx = {ch: offset + idx for idx, ch in enumerate(SV_CHANNELS)}
    cols_lower = {c.lower(): c for c in sv_df.columns}

    ch_col = cols_lower.get("channel") or cols_lower.get("sv_channel")
    type_col = cols_lower.get("sv_type") or cols_lower.get("svtype") or cols_lower.get("type")
    len_col = cols_lower.get("sv_len") or cols_lower.get("svlen") or cols_lower.get("length")
    c1_col = cols_lower.get("chrom1") or cols_lower.get("chr1") or cols_lower.get("chrom")
    c2_col = cols_lower.get("chrom2") or cols_lower.get("chr2")
    p1_col = cols_lower.get("pos1") or cols_lower.get("start") or cols_lower.get("pos")
    p2_col = cols_lower.get("pos2") or cols_lower.get("end")

    for row in sv_df.itertuples(index=False):
        row_dict = row._asdict()
        sid = str(row_dict[sample_col]) if sample_col is not None else default_sample
        vec = counts[sid]

        if ch_col is not None:
            raw_c = str(row_dict[ch_col]).strip()
            if raw_c in sv_idx:
                vec[sv_idx[raw_c]] += 1
                continue

        # Inter-chromosomal translocation check
        if c1_col is not None and c2_col is not None:
            ch1 = str(row_dict[c1_col]).replace("chr", "")
            ch2 = str(row_dict[c2_col]).replace("chr", "")
            if ch1 and ch2 and ch1 != ch2:
                vec[sv_idx["TRA"]] += 1
                continue

        if type_col is None:
            continue
        sv_t = str(row_dict[type_col])

        sv_l: float | None = None
        if len_col is not None and pd.notna(row_dict[len_col]):
            sv_l = abs(float(row_dict[len_col]))
        elif p1_col is not None and p2_col is not None:
            if pd.notna(row_dict[p1_col]) and pd.notna(row_dict[p2_col]):
                sv_l = abs(float(row_dict[p2_col]) - float(row_dict[p1_col]))

        mapped = classify_sv_channel(sv_t, sv_l)
        if mapped is not None and mapped in sv_idx:
            vec[sv_idx[mapped]] += 1


def extract_sigs_chord(
    snv_df: pd.DataFrame | None = None,
    indel_df: pd.DataFrame | None = None,
    sv_df: pd.DataFrame | None = None,
    ref_fasta: str | Path | pysam.FastaFile | dict[str, str] | None = None,
    sample_id: str = "sample",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Extract the 3-channel CHORD mutational signature context matrix (96 SBS + 17 Indel
    + 19 SV = 132 channels) for a single sample or multi-sample cohort.
    """
    validate_settings(settings)

    snv_s_col = _resolve_sample_col(snv_df)
    indel_s_col = _resolve_sample_col(indel_df)
    sv_s_col = _resolve_sample_col(sv_df)

    sample_order: list[str] = []
    seen: set[str] = set()
    for df_part, s_col in ((snv_df, snv_s_col), (indel_df, indel_s_col), (sv_df, sv_s_col)):
        if df_part is not None and not df_part.empty and s_col is not None:
            for val in df_part[s_col].astype(str):
                if val not in seen:
                    seen.add(val)
                    sample_order.append(val)

    if not sample_order:
        sample_order = [str(sample_id)]

    n_channels = len(ALL_CHORD_CHANNELS)
    counts: dict[str, np.ndarray] = {
        sid: np.zeros(n_channels, dtype=np.int64) for sid in sample_order
    }

    fasta = _FastaAccessor(ref_fasta)
    try:
        if snv_df is not None and not snv_df.empty:
            _extract_snv_counts(snv_df, fasta, counts, snv_s_col, str(sample_id))
        if indel_df is not None and not indel_df.empty:
            _extract_indel_counts(indel_df, fasta, counts, indel_s_col, str(sample_id))
        if sv_df is not None and not sv_df.empty:
            _extract_sv_counts(sv_df, counts, sv_s_col, str(sample_id))
    finally:
        fasta.close()

    mat = np.vstack([counts[sid] for sid in sample_order])
    out_df = pd.DataFrame(mat, index=sample_order, columns=list(ALL_CHORD_CHANNELS))
    out_df.index.name = "sample"
    return out_df
