"""Turnkey bioinformatics file readers for pythonic_biotools pipelines."""

from __future__ import annotations

import gzip
import io as _pyio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "BismarkCovData",
    "maf_to_binary_matrix",
    "read_bismark_cov",
    "read_cn_segments",
    "read_gatk_copy_ratios",
    "read_gwas_sumstats",
    "read_maf",
    "read_maxquant_diann",
    "read_plink_ld",
    "read_vcf_variants",
]

_SILENT_MAF_CLASSES = frozenset(
    {
        "silent",
        "synonymous_snv",
        "synonymous_variant",
        "intron",
        "3'utr",
        "5'utr",
        "3'flank",
        "5'flank",
        "igr",
        "rna",
        "lincrna",
        "targeted_region",
    }
)


def _open_text(path: str | Path) -> _pyio.TextIOBase:
    """Open plain-text or gzip-compressed file in UTF-8 text mode."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b" or p.name.endswith(".gz"):
        return gzip.open(p, mode="rt", encoding="utf-8", errors="replace")  # type: ignore[return-value]
    return p.open(mode="r", encoding="utf-8", errors="replace")


def _read_table_skipping_prefixes(
    path_or_df: str | Path | pd.DataFrame,
    skip_prefixes: tuple[str, ...] = ("#",),
    sep: str | None = "\t",
) -> pd.DataFrame:
    """Read a tabular file skipping leading metadata lines starting with `skip_prefixes`."""
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()

    lines: list[str] = []
    with _open_text(path_or_df) as fh:
        for line in fh:
            if not line.strip():
                continue
            if skip_prefixes and line.startswith(skip_prefixes):
                continue
            lines.append(line)

    if not lines:
        raise ValueError(f"No tabular records found in {path_or_df!r}.")

    buf = _pyio.StringIO("".join(lines))
    if sep is None:
        first = lines[0]
        if "\t" in first:
            return pd.read_csv(buf, sep="\t", low_memory=False)
        if "," in first and " " not in first.strip():
            return pd.read_csv(buf, sep=",", low_memory=False)
        return pd.read_csv(buf, sep=r"\s+", engine="python")
    return pd.read_csv(buf, sep=sep, low_memory=False)


def _find_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    case_insensitive: bool = True,
) -> str | None:
    """Find first matching column name in `df` from `candidates`."""
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    if case_insensitive:
        lower_map = {str(c).lower(): str(c) for c in df.columns}
        for cand in candidates:
            hit = lower_map.get(cand.lower())
            if hit is not None:
                return hit
    return None


def _infer_variant_type(ref_arr: Sequence[str], alt_arr: Sequence[str]) -> list[str]:
    """Infer MAF Variant_Type ('SNP', 'DNP', 'TNP', 'ONP', 'INS', 'DEL') from alleles."""
    out: list[str] = []
    for r, a in zip(ref_arr, alt_arr):
        if r in ("-", "") or (len(a) > len(r) and a != "-"):
            out.append("INS")
        elif a in ("-", "") or len(r) > len(a):
            out.append("DEL")
        elif len(r) == 1:
            out.append("SNP")
        elif len(r) == 2:
            out.append("DNP")
        elif len(r) == 3:
            out.append("TNP")
        else:
            out.append("ONP")
    return out


# ---------------------------------------------------------------------------
# 1. MAF reader & gene x sample binary / NaN alteration matrix
# ---------------------------------------------------------------------------


def read_maf(
    path_or_df: str | Path | pd.DataFrame,
    keep_silent: bool = True,
    sample_col: str = "Tumor_Sample_Barcode",
) -> pd.DataFrame:
    """Read GDC / cBioPortal / AACR GENIE `.maf` or `.maf.gz` into standardized columns.

    Normalized columns:
    `sample_id`, `gene`, `chrom`, `pos`, `ref`, `alt`, `alt_count`, `ref_count`,
    `total_depth`, `vaf`, `variant_classification`, `variant_type`.
    """
    raw = _read_table_skipping_prefixes(path_or_df, skip_prefixes=("#",), sep="\t")
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "gene",
                "chrom",
                "pos",
                "ref",
                "alt",
                "alt_count",
                "ref_count",
                "total_depth",
                "vaf",
                "variant_classification",
                "variant_type",
            ]
        )

    s_col = _find_col(
        raw,
        (sample_col, "Tumor_Sample_Barcode", "sample_id", "sampleID", "Sample_ID", "sample"),
    )
    g_col = _find_col(raw, ("Hugo_Symbol", "gene", "gene_name", "Gene", "Symbol"))
    c_col = _find_col(raw, ("Chromosome", "chrom", "chr", "CHROM"))
    p_col = _find_col(raw, ("Start_Position", "Start_position", "pos", "POS", "start"))
    r_col = _find_col(raw, ("Reference_Allele", "ref", "REF"))
    a2_col = _find_col(raw, ("Tumor_Seq_Allele2", "alt", "ALT", "mut"))
    a1_col = _find_col(raw, ("Tumor_Seq_Allele1",))

    missing_roles = [
        role
        for role, col in (
            ("sample_id", s_col),
            ("chrom", c_col),
            ("pos", p_col),
            ("ref", r_col),
            ("alt", a2_col),
        )
        if col is None
    ]
    if missing_roles:
        raise ValueError(
            f"MAF input is missing required column(s) for {missing_roles}. "
            f"Found columns: {list(raw.columns)}"
        )

    assert s_col is not None and c_col is not None and p_col is not None
    assert r_col is not None and a2_col is not None

    n = len(raw)
    pos_num = pd.to_numeric(raw[p_col], errors="coerce")
    if pos_num.isna().any():
        raise ValueError("MAF Start_Position / pos contains non-numeric or missing values.")
    pos_arr = pos_num.to_numpy(dtype=np.int64)
    if np.any(pos_arr < 1):
        raise ValueError("MAF genomic positions must be 1-based positive integers (>= 1).")

    ref_ser = raw[r_col].astype(str).str.strip().str.upper()
    alt_ser = raw[a2_col].astype(str).str.strip().str.upper()
    if a1_col is not None and a1_col != a2_col:
        a1_ser = raw[a1_col].astype(str).str.strip().str.upper()
        use_a1 = (alt_ser == ref_ser) & (a1_ser != ref_ser) & (a1_ser != "") & (a1_ser != "NAN")
        alt_ser = alt_ser.where(~use_a1, a1_ser)

    gene_ser = (
        raw[g_col].fillna("UNKNOWN").astype(str).str.strip()
        if g_col is not None
        else pd.Series(["UNKNOWN"] * n, index=raw.index)
    )

    alt_cnt_col = _find_col(
        raw, ("t_alt_count", "alt_count", "tum_alt", "t_var_count", "ad_alt")
    )
    ref_cnt_col = _find_col(raw, ("t_ref_count", "ref_count", "tum_ref", "ad_ref"))
    dp_col = _find_col(raw, ("t_depth", "total_depth", "depth", "dp", "tum_dp"))
    vaf_col = _find_col(raw, ("t_vaf", "vaf", "tumor_f", "i_tumorvaf_wu", "af"))

    alt_cnt = (
        pd.to_numeric(raw[alt_cnt_col], errors="coerce").to_numpy(dtype=np.float64)
        if alt_cnt_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    ref_cnt = (
        pd.to_numeric(raw[ref_cnt_col], errors="coerce").to_numpy(dtype=np.float64)
        if ref_cnt_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    tot_dp = (
        pd.to_numeric(raw[dp_col], errors="coerce").to_numpy(dtype=np.float64)
        if dp_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    vaf = (
        pd.to_numeric(raw[vaf_col], errors="coerce").to_numpy(dtype=np.float64)
        if vaf_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )

    # Reconcile total_depth, ref_count, alt_count, and vaf
    need_dp = ~np.isfinite(tot_dp) & np.isfinite(alt_cnt) & np.isfinite(ref_cnt)
    tot_dp[need_dp] = alt_cnt[need_dp] + ref_cnt[need_dp]

    need_alt = ~np.isfinite(alt_cnt) & np.isfinite(tot_dp) & np.isfinite(vaf)
    alt_cnt[need_alt] = np.round(np.clip(vaf[need_alt], 0.0, 1.0) * tot_dp[need_alt])

    need_ref = ~np.isfinite(ref_cnt) & np.isfinite(tot_dp) & np.isfinite(alt_cnt)
    ref_cnt[need_ref] = np.maximum(tot_dp[need_ref] - alt_cnt[need_ref], 0.0)

    need_vaf = ~np.isfinite(vaf) & np.isfinite(alt_cnt) & np.isfinite(tot_dp) & (tot_dp > 0.0)
    vaf[need_vaf] = alt_cnt[need_vaf] / tot_dp[need_vaf]

    vc_col = _find_col(raw, ("Variant_Classification", "variant_classification", "Consequence"))
    vc_ser = (
        raw[vc_col].fillna("").astype(str).str.strip()
        if vc_col is not None
        else pd.Series([""] * n, index=raw.index)
    )

    vt_col = _find_col(raw, ("Variant_Type", "variant_type"))
    if vt_col is not None:
        vt_ser = raw[vt_col].fillna("").astype(str).str.strip()
    else:
        vt_ser = pd.Series(
            _infer_variant_type(ref_ser.tolist(), alt_ser.tolist()), index=raw.index
        )

    out = pd.DataFrame(
        {
            "sample_id": raw[s_col].astype(str).str.strip(),
            "gene": gene_ser,
            "chrom": raw[c_col].astype(str).str.strip(),
            "pos": pos_arr,
            "ref": ref_ser,
            "alt": alt_ser,
            "alt_count": alt_cnt,
            "ref_count": ref_cnt,
            "total_depth": tot_dp,
            "vaf": vaf,
            "variant_classification": vc_ser,
            "variant_type": vt_ser,
        },
        index=raw.index,
    )

    for extra_col in ("context", "pop_af"):
        found = _find_col(raw, (extra_col,))
        if found is not None and extra_col not in out.columns:
            out[extra_col] = raw[found]

    if not keep_silent:
        is_silent = out["variant_classification"].str.lower().isin(_SILENT_MAF_CLASSES)
        out = out.loc[~is_silent].copy()

    return out.reset_index(drop=True)


def maf_to_binary_matrix(
    maf_df: pd.DataFrame,
    panel_covered_df: pd.DataFrame | None = None,
    nonsyn_only: bool = True,
) -> pd.DataFrame:
    """Convert MAF mutations into a `{0.0, 1.0, NaN}` `gene x sample` matrix for `discover` & `evam`."""
    if not isinstance(maf_df, pd.DataFrame):
        raise TypeError("maf_df must be a pandas DataFrame.")

    if {"sample_id", "gene"}.issubset(maf_df.columns):
        norm_df = maf_df.copy()
        if "variant_classification" not in norm_df.columns:
            norm_df["variant_classification"] = ""
    else:
        norm_df = read_maf(maf_df, keep_silent=True)

    all_samples = list(pd.unique(norm_df["sample_id"].astype(str)))
    if panel_covered_df is not None:
        if not isinstance(panel_covered_df, pd.DataFrame):
            raise TypeError("panel_covered_df must be a gene x sample pandas DataFrame or None.")
        for s in panel_covered_df.columns.astype(str):
            if s not in all_samples:
                all_samples.append(s)

    work = norm_df
    if nonsyn_only and (work["variant_classification"].astype(str).str.strip() != "").any():
        is_silent = work["variant_classification"].astype(str).str.lower().isin(_SILENT_MAF_CLASSES)
        work = work.loc[~is_silent]

    work = work.loc[
        work["gene"].notna()
        & (work["gene"].astype(str).str.strip() != "")
        & (work["gene"].astype(str).str.strip() != "UNKNOWN")
    ]

    all_genes = list(pd.unique(work["gene"].astype(str)))
    if panel_covered_df is not None:
        for g in panel_covered_df.index.astype(str):
            if g not in all_genes:
                all_genes.append(g)

    mat = pd.DataFrame(
        0.0,
        index=pd.Index(all_genes, name="gene"),
        columns=pd.Index(all_samples, name="sample_id"),
        dtype=np.float64,
    )

    if not work.empty and all_genes and all_samples:
        pairs = work[["gene", "sample_id"]].astype(str).drop_duplicates()
        g_idx = mat.index.get_indexer(pairs["gene"])
        s_idx = mat.columns.get_indexer(pairs["sample_id"])
        valid = (g_idx >= 0) & (s_idx >= 0)
        arr = mat.to_numpy()
        arr[g_idx[valid], s_idx[valid]] = 1.0

    if panel_covered_df is not None:
        cov = panel_covered_df.copy()
        cov.index = cov.index.astype(str)
        cov.columns = cov.columns.astype(str)
        cov_aligned = cov.reindex(index=mat.index, columns=mat.columns)
        uncovered = cov_aligned.isna() | (cov_aligned.astype(float) == 0.0)
        mat = mat.mask(uncovered, np.nan)

    return mat


# ---------------------------------------------------------------------------
# 2. Pure-Python VCF / VCF.gz variant parser
# ---------------------------------------------------------------------------


def _parse_float_token(tok: str | None) -> float:
    if tok is None or tok in ("", "."):
        return float("nan")
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def _parse_info_field(info_str: str) -> dict[str, str]:
    if not info_str or info_str == ".":
        return {}
    out: dict[str, str] = {}
    for item in info_str.split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
        else:
            out[item] = "True"
    return out


def read_vcf_variants(
    path_or_lines: str | Path | Iterable[str],
    sample: str | int | None = None,
    pass_only: bool = True,
) -> pd.DataFrame:
    """Parse `.vcf` / `.vcf.gz` (or iterable of VCF lines) into variant count DataFrame.

    Extracts `chrom`, `pos`, `id`, `ref`, `alt`, `qual`, `filter`, `AF`, `pop_af`
    from `INFO` (`AF`, `gnomAD_AF`, `POPAF`), and `GT`, `ref_count`, `alt_count`,
    `total_depth`, `vaf` from `FORMAT`/sample columns.
    """
    columns = [
        "chrom",
        "pos",
        "id",
        "ref",
        "alt",
        "qual",
        "filter",
        "AF",
        "pop_af",
        "GT",
        "ref_count",
        "alt_count",
        "total_depth",
        "vaf",
        "sample_id",
    ]

    if isinstance(path_or_lines, (str, Path)):
        fh = _open_text(path_or_lines)
        close_fh = True
        line_iter: Iterable[str] = fh
    else:
        close_fh = False
        line_iter = path_or_lines

    sample_names: list[str] = []
    target_sample_idx: int | None = None
    target_sample_name: str = ""
    rows: list[dict[str, object]] = []

    try:
        for raw_line in line_iter:
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                hdr = line[1:].split("\t")
                if len(hdr) > 9:
                    sample_names = hdr[9:]
                    if sample is None:
                        target_sample_idx = 0
                    elif isinstance(sample, int):
                        if sample < 0 or sample >= len(sample_names):
                            raise ValueError(
                                f"Sample index {sample} out of range for VCF samples {sample_names}."
                            )
                        target_sample_idx = sample
                    else:
                        if sample not in sample_names:
                            raise ValueError(
                                f"Sample {sample!r} not found in VCF header samples {sample_names}."
                            )
                        target_sample_idx = sample_names.index(sample)
                    target_sample_name = sample_names[target_sample_idx]
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError(
                    f"Malformed VCF data line (expected >= 8 tab-separated fields, got {len(parts)}): {line!r}"
                )

            chrom, pos_s, vid, ref, alt_field, qual_s, filt, info_s = parts[:8]
            if pass_only and filt not in ("PASS", "."):
                continue

            try:
                pos = int(pos_s)
            except ValueError as exc:
                raise ValueError(f"Invalid integer POS {pos_s!r} in VCF record.") from exc

            qual = _parse_float_token(qual_s)
            info_map = _parse_info_field(info_s)

            alts = alt_field.split(",")
            info_af_tokens = info_map.get("AF", "").split(",") if "AF" in info_map else []

            # Population AF from gnomAD_AF / AF_popmax / MAX_AF / POPAF
            pop_af_tokens: list[str] = []
            pop_af_is_phred = False
            for pk in ("gnomAD_AF", "gnomad_AF", "AF_popmax", "MAX_AF", "ExAC_AF"):
                if pk in info_map:
                    pop_af_tokens = info_map[pk].split(",")
                    break
            else:
                if "POPAF" in info_map:
                    pop_af_tokens = info_map["POPAF"].split(",")
                    pop_af_is_phred = True

            fmt_map: dict[str, str] = {}
            s_name = target_sample_name
            if len(parts) >= 10:
                s_idx = target_sample_idx if target_sample_idx is not None else 0
                if isinstance(sample, int) and target_sample_idx is None:
                    s_idx = sample
                if 9 + s_idx >= len(parts):
                    raise ValueError(f"Sample column index {s_idx} missing on VCF line at {chrom}:{pos}.")
                if not s_name:
                    s_name = f"SAMPLE_{s_idx + 1}"
                fmt_keys = parts[8].split(":")
                fmt_vals = parts[9 + s_idx].split(":")
                fmt_map = dict(zip(fmt_keys, fmt_vals))

            gt_val = fmt_map.get("GT", "")
            dp_val = _parse_float_token(fmt_map.get("DP"))
            if not np.isfinite(dp_val) and "DP" in info_map:
                dp_val = _parse_float_token(info_map["DP"])

            ad_tokens = fmt_map.get("AD", "").split(",") if "AD" in fmt_map else []
            fmt_af_tokens: list[str] = []
            for fk in ("AF", "VAF", "FA"):
                if fk in fmt_map:
                    fmt_af_tokens = fmt_map[fk].split(",")
                    break

            for alt_i, alt_allele in enumerate(alts):
                if alt_allele == ".":
                    continue

                af_info = (
                    _parse_float_token(info_af_tokens[alt_i])
                    if alt_i < len(info_af_tokens)
                    else (
                        _parse_float_token(info_af_tokens[0])
                        if len(info_af_tokens) == 1
                        else float("nan")
                    )
                )

                raw_pop = (
                    _parse_float_token(pop_af_tokens[alt_i])
                    if alt_i < len(pop_af_tokens)
                    else (
                        _parse_float_token(pop_af_tokens[0])
                        if len(pop_af_tokens) == 1
                        else float("nan")
                    )
                )
                if np.isfinite(raw_pop) and pop_af_is_phred and raw_pop > 1.0:
                    pop_af = float(10.0 ** (-raw_pop))
                else:
                    pop_af = raw_pop

                ref_cnt = _parse_float_token(ad_tokens[0]) if len(ad_tokens) >= 2 else float("nan")
                alt_cnt = (
                    _parse_float_token(ad_tokens[alt_i + 1])
                    if len(ad_tokens) > alt_i + 1
                    else float("nan")
                )

                tot_dp = dp_val
                if not np.isfinite(tot_dp) and np.isfinite(ref_cnt) and np.isfinite(alt_cnt):
                    tot_dp = ref_cnt + alt_cnt

                fmt_vaf = (
                    _parse_float_token(fmt_af_tokens[alt_i])
                    if alt_i < len(fmt_af_tokens)
                    else (
                        _parse_float_token(fmt_af_tokens[0])
                        if len(fmt_af_tokens) == 1
                        else float("nan")
                    )
                )

                if np.isfinite(alt_cnt) and np.isfinite(tot_dp) and tot_dp > 0.0:
                    vaf = alt_cnt / tot_dp
                elif np.isfinite(fmt_vaf):
                    vaf = fmt_vaf
                    if not np.isfinite(alt_cnt) and np.isfinite(tot_dp):
                        alt_cnt = float(np.round(np.clip(vaf, 0.0, 1.0) * tot_dp))
                        if not np.isfinite(ref_cnt):
                            ref_cnt = max(tot_dp - alt_cnt, 0.0)
                else:
                    vaf = af_info

                rows.append(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "id": vid,
                        "ref": ref.upper(),
                        "alt": alt_allele.upper(),
                        "qual": qual,
                        "filter": filt,
                        "AF": af_info,
                        "pop_af": pop_af,
                        "GT": gt_val,
                        "ref_count": ref_cnt,
                        "alt_count": alt_cnt,
                        "total_depth": tot_dp,
                        "vaf": vaf,
                        "sample_id": s_name,
                    }
                )
    finally:
        if close_fh:
            fh.close()  # type: ignore[union-attr]

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# 3. GATK4 / CNVkit copy-ratio bins & allele-specific CN segments
# ---------------------------------------------------------------------------


def read_gatk_copy_ratios(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Parse GATK4 `DenoisedCopyRatios.tsv` / `ReadCounts.tsv` or CNVkit `.cnr`.

    Automatically skips `@HD` / `@SQ` SAM sequence dictionary headers and `#` comments,
    returning normalized `chrom`, `start`, `end`, `logR`, `log2_ratio`, and `coverage`
    columns for `purecn`, `qdnaseq`, and `ichorcna`.
    """
    raw = _read_table_skipping_prefixes(path_or_df, skip_prefixes=("@", "#"), sep="\t")
    if raw.empty:
        raise ValueError("Copy-ratio input contains 0 genomic bins.")

    c_col = _find_col(raw, ("CONTIG", "chromosome", "chrom", "chr", "seqnames"))
    s_col = _find_col(raw, ("START", "start", "pos", "position"))
    e_col = _find_col(raw, ("END", "end"))
    lr_col = _find_col(
        raw,
        ("LOG2_COPY_RATIO", "log2", "logR", "log2_ratio", "cnlr", "log_ratio"),
    )
    cov_col = _find_col(raw, ("coverage", "depth", "COUNT", "counts", "reads", "tumor_cov"))

    if c_col is None or s_col is None:
        raise ValueError(
            f"Copy-ratio table must contain chromosome and start columns; got {list(raw.columns)}."
        )
    if lr_col is None and cov_col is None:
        raise ValueError(
            f"Copy-ratio table must provide log2 ratio ('LOG2_COPY_RATIO'/'log2'/'logR') "
            f"or linear depth ('COUNT'/'depth'/'coverage'); got {list(raw.columns)}."
        )

    starts = pd.to_numeric(raw[s_col], errors="coerce").to_numpy(dtype=np.int64)
    ends = (
        pd.to_numeric(raw[e_col], errors="coerce").to_numpy(dtype=np.int64)
        if e_col is not None
        else starts + 200
    )
    ends = np.maximum(ends, starts)

    if lr_col is not None:
        log2_r = pd.to_numeric(raw[lr_col], errors="coerce").to_numpy(dtype=np.float64)
        if cov_col is not None:
            cov = pd.to_numeric(raw[cov_col], errors="coerce").to_numpy(dtype=np.float64)
        else:
            cov = 100.0 * np.power(2.0, np.clip(log2_r, -10.0, 10.0))
    else:
        assert cov_col is not None
        cov = np.maximum(
            pd.to_numeric(raw[cov_col], errors="coerce").to_numpy(dtype=np.float64), 0.0
        )
        med = max(float(np.nanmedian(cov + 0.5)), 0.5)
        log2_r = np.log2((cov + 0.5) / med)

    out = pd.DataFrame(
        {
            "chrom": raw[c_col].astype(str).str.strip(),
            "start": starts,
            "end": ends,
            "logR": log2_r,
            "log2_ratio": log2_r,
            "coverage": cov,
        }
    )

    for opt in ("gc", "weight", "gene"):
        opt_col = _find_col(raw, (opt,))
        if opt_col is not None:
            out[opt] = raw[opt_col].to_numpy()

    return out.reset_index(drop=True)


def read_cn_segments(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Parse GATK4 `ModelSegments` `.cr.seg`, CNVkit `.cns`, FACETS, or PureCN segments.

    Skips `@HD` / `@SQ` SAM headers and normalizes `chrom`, `start`, `end`, `logR`,
    `log2_ratio`, `seg_mean`, `num_bins`, `major_cn`, `minor_cn`, and `total_cn`
    for `scarhrd`, `mutationtimer`, `purecn`, `qdnaseq`, and `facets`.
    """
    raw = _read_table_skipping_prefixes(path_or_df, skip_prefixes=("@", "#"), sep="\t")
    if raw.empty:
        raise ValueError("Copy-number segment input contains 0 segments.")

    c_col = _find_col(raw, ("CONTIG", "chromosome", "chrom", "chr", "seqnames", "Chromosome"))
    s_col = _find_col(raw, ("START", "start", "loc.start", "Start"))
    e_col = _find_col(raw, ("END", "end", "loc.end", "End"))
    if c_col is None or s_col is None or e_col is None:
        raise ValueError(
            f"Segment table must contain chrom/start/end columns; got {list(raw.columns)}."
        )

    n = len(raw)
    starts = pd.to_numeric(raw[s_col], errors="coerce").to_numpy(dtype=np.int64)
    ends = pd.to_numeric(raw[e_col], errors="coerce").to_numpy(dtype=np.int64)
    if np.any(ends < starts):
        raise ValueError("Segment table contains rows with end < start.")

    lr_col = _find_col(
        raw,
        (
            "MEAN_LOG2_COPY_RATIO",
            "log2_ratio",
            "logR",
            "log2",
            "seg_mean",
            "seg.mean",
            "cnlr_median",
        ),
    )
    nb_col = _find_col(
        raw,
        ("NUM_POINTS_COPY_RATIO", "num_bins", "n_bins", "probes", "num_mark", "num.mark"),
    )
    tot_col = _find_col(
        raw, ("total_cn", "tcn", "tcn.em", "cn", "C", "copy_number", "Modal_Total_CN")
    )
    maj_col = _find_col(raw, ("major_cn", "cn1", "Major_CN", "A1"))
    min_col = _find_col(raw, ("minor_cn", "lcn", "lcn.em", "cn2", "M", "Minor_CN", "A2"))

    tot = (
        pd.to_numeric(raw[tot_col], errors="coerce").to_numpy(dtype=np.float64)
        if tot_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    maj = (
        pd.to_numeric(raw[maj_col], errors="coerce").to_numpy(dtype=np.float64)
        if maj_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    mino = (
        pd.to_numeric(raw[min_col], errors="coerce").to_numpy(dtype=np.float64)
        if min_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )

    # Derive missing CN component when 2 of (major_cn, minor_cn, total_cn) are provided
    m_tot = ~np.isfinite(tot) & np.isfinite(maj) & np.isfinite(mino)
    tot[m_tot] = maj[m_tot] + mino[m_tot]

    m_maj = ~np.isfinite(maj) & np.isfinite(tot) & np.isfinite(mino)
    maj[m_maj] = tot[m_maj] - mino[m_maj]

    m_min = ~np.isfinite(mino) & np.isfinite(tot) & np.isfinite(maj)
    mino[m_min] = tot[m_min] - maj[m_min]

    both_allele = np.isfinite(maj) & np.isfinite(mino)
    hi = np.where(both_allele, np.maximum(maj, mino), maj)
    lo = np.where(both_allele, np.minimum(maj, mino), mino)
    maj, mino = hi, lo

    if lr_col is not None:
        log2_r = pd.to_numeric(raw[lr_col], errors="coerce").to_numpy(dtype=np.float64)
    elif np.any(np.isfinite(tot)):
        log2_r = np.where(
            np.isfinite(tot), np.log2(np.maximum(tot, 0.05) / 2.0), np.nan
        )
    else:
        log2_r = np.full(n, np.nan, dtype=np.float64)

    num_bins = (
        pd.to_numeric(raw[nb_col], errors="coerce").to_numpy(dtype=np.float64)
        if nb_col is not None
        else np.full(n, np.nan, dtype=np.float64)
    )

    out = pd.DataFrame(
        {
            "chrom": raw[c_col].astype(str).str.strip(),
            "start": starts,
            "end": ends,
            "logR": log2_r,
            "log2_ratio": log2_r,
            "seg_mean": log2_r,
            "num_bins": num_bins,
            "major_cn": maj,
            "minor_cn": mino,
            "total_cn": tot,
        }
    )

    for extra in ("sample_id", "Sample", "ID", "clonal_frequency"):
        ex_col = _find_col(raw, (extra,), case_insensitive=False)
        if ex_col is not None:
            key = "sample_id" if extra in ("sample_id", "Sample", "ID") else extra
            if key not in out.columns:
                out[key] = raw[ex_col].to_numpy()

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Bismark .bismark.cov / .cov.gz & MethylDackel bedGraph reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BismarkCovData:
    """Container holding both per-replicate `DSS` DataFrames and aligned count matrices."""

    frames: list[pd.DataFrame]
    N_mat: np.ndarray
    X_mat: np.ndarray
    coords_df: pd.DataFrame
    sample_names: list[str]

    def __iter__(self) -> Iterator[object]:
        """Allow `frames, N_mat, X_mat, coords_df = read_bismark_cov(...)` unpacking."""
        yield self.frames
        yield self.N_mat
        yield self.X_mat
        yield self.coords_df

    def to_matrices(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Return `(N_mat, X_mat, coords_df)` tuple for `dss.dml_fit_multifactor`."""
        return self.N_mat, self.X_mat, self.coords_df


def _parse_single_bismark_or_bedgraph(
    path: str | Path | pd.DataFrame,
    min_cov: int,
) -> pd.DataFrame:
    """Parse one Bismark `.cov`, MethylDackel `.bedGraph`, or `DSS` `chr/pos/N/X` table."""
    if isinstance(path, pd.DataFrame):
        df_in = path.copy()
        if {"chr", "pos", "N", "X"}.issubset(df_in.columns):
            sub = df_in[["chr", "pos", "N", "X"]].copy()
            sub["chr"] = sub["chr"].astype(str)
            sub["pos"] = pd.to_numeric(sub["pos"], errors="raise").astype(np.int64)
            sub["N"] = pd.to_numeric(sub["N"], errors="raise").astype(np.int64)
            sub["X"] = pd.to_numeric(sub["X"], errors="raise").astype(np.int64)
            sub = sub.loc[sub["N"] >= min_cov]
            if sub.duplicated(subset=["chr", "pos"]).any():
                sub = sub.groupby(["chr", "pos"], as_index=False, sort=False)[["N", "X"]].sum()
            return sub.reset_index(drop=True)
        raise ValueError(
            f"DataFrame passed to read_bismark_cov must contain ('chr', 'pos', 'N', 'X'); "
            f"got {list(df_in.columns)}."
        )

    p = Path(path)
    has_track_header = False
    is_bedgraph_name = ".bedgraph" in p.name.lower()
    data_lines: list[str] = []

    with _open_text(p) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("track", "browser", "#")):
                if s.startswith("track"):
                    has_track_header = True
                continue
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame(columns=["chr", "pos", "N", "X"])

    first_tokens = data_lines[0].rstrip("\r\n").split("\t")
    if len(first_tokens) == 1:
        first_tokens = data_lines[0].split()

    # Check if file has a named header (e.g. DSS TSV `chr\tpos\tN\tX`)
    try:
        float(first_tokens[1])
        has_named_header = False
    except (IndexError, ValueError):
        has_named_header = True

    buf = _pyio.StringIO("".join(data_lines))
    sep = "\t" if "\t" in data_lines[0] else r"\s+"

    if has_named_header:
        tbl = pd.read_csv(buf, sep=sep, engine="c" if sep == "\t" else "python")
        if {"chr", "pos", "N", "X"}.issubset(tbl.columns):
            out = pd.DataFrame(
                {
                    "chr": tbl["chr"].astype(str),
                    "pos": pd.to_numeric(tbl["pos"], errors="raise").to_numpy(dtype=np.int64),
                    "N": pd.to_numeric(tbl["N"], errors="raise").to_numpy(dtype=np.int64),
                    "X": pd.to_numeric(tbl["X"], errors="raise").to_numpy(dtype=np.int64),
                }
            )
        else:
            c_c = _find_col(tbl, ("chr", "chrom", "chromosome"))
            p_c = _find_col(tbl, ("start", "pos"))
            m_c = _find_col(tbl, ("meth", "methylated", "X", "count_methylated"))
            u_c = _find_col(tbl, ("unmeth", "unmethylated", "count_unmethylated"))
            if None in (c_c, p_c, m_c, u_c):
                raise ValueError(f"Unrecognized header in methylation file {p}: {list(tbl.columns)}")
            meth = pd.to_numeric(tbl[m_c], errors="raise").to_numpy(dtype=np.int64)  # type: ignore[index]
            unmeth = pd.to_numeric(tbl[u_c], errors="raise").to_numpy(dtype=np.int64)  # type: ignore[index]
            out = pd.DataFrame(
                {
                    "chr": tbl[c_c].astype(str),  # type: ignore[index]
                    "pos": pd.to_numeric(tbl[p_c], errors="raise").to_numpy(dtype=np.int64),  # type: ignore[index]
                    "N": meth + unmeth,
                    "X": meth,
                }
            )
    else:
        tbl = pd.read_csv(
            buf,
            sep=sep,
            header=None,
            engine="c" if sep == "\t" else "python",
        )
        if tbl.shape[1] < 6:
            raise ValueError(
                f"Expected 6-column Bismark .cov or MethylDackel bedGraph in {p}; got {tbl.shape[1]} columns."
            )
        starts = pd.to_numeric(tbl.iloc[:, 1], errors="raise").to_numpy(dtype=np.int64)
        ends = pd.to_numeric(tbl.iloc[:, 2], errors="raise").to_numpy(dtype=np.int64)
        meth = pd.to_numeric(tbl.iloc[:, 4], errors="raise").to_numpy(dtype=np.int64)
        unmeth = pd.to_numeric(tbl.iloc[:, 5], errors="raise").to_numpy(dtype=np.int64)

        # MethylDackel bedGraph uses 0-based half-open [start, end); Bismark .cov uses 1-based start == end
        is_zero_based = (has_track_header or is_bedgraph_name) and bool(np.all(ends > starts))
        pos = starts + 1 if is_zero_based else starts

        out = pd.DataFrame(
            {
                "chr": tbl.iloc[:, 0].astype(str),
                "pos": pos,
                "N": meth + unmeth,
                "X": meth,
            }
        )

    out = out.loc[out["N"] >= min_cov].copy()
    if out.duplicated(subset=["chr", "pos"]).any():
        out = out.groupby(["chr", "pos"], as_index=False, sort=False)[["N", "X"]].sum()
    return out.reset_index(drop=True)


def read_bismark_cov(
    paths: str | Path | pd.DataFrame | Sequence[str | Path | pd.DataFrame],
    sample_names: Sequence[str] | None = None,
    min_cov: int = 1,
) -> BismarkCovData:
    """Parse Bismark `.bismark.cov` / `.cov.gz` or MethylDackel `.bedGraph` replicates.

    Returns a `BismarkCovData` object containing:
    - `frames`: list of per-replicate `DSS` DataFrames (`chr`, `pos`, `N`, `X`) for `dss.dml_test`
    - `N_mat`, `X_mat`, `coords_df`: aligned `(n_cpgs, n_samples)` matrices and coordinates
      for `dss.dml_fit_multifactor`.
    """
    if min_cov < 0:
        raise ValueError(f"min_cov must be >= 0, got {min_cov}.")

    if isinstance(paths, (str, Path, pd.DataFrame)):
        path_list: list[str | Path | pd.DataFrame] = [paths]
    else:
        path_list = list(paths)

    if len(path_list) == 0:
        raise ValueError("paths must contain at least one Bismark/bedGraph file.")

    if sample_names is not None:
        names = [str(s) for s in sample_names]
        if len(names) != len(path_list):
            raise ValueError(
                f"Length of sample_names ({len(names)}) must match paths ({len(path_list)})."
            )
    else:
        names = []
        for idx, item in enumerate(path_list):
            if isinstance(item, (str, Path)):
                stem = Path(item).name
                for suf in (".gz", ".cov", ".bismark", ".bedGraph", ".bedgraph", ".tsv", ".txt"):
                    if stem.endswith(suf):
                        stem = stem[: -len(suf)]
                names.append(stem or f"S{idx + 1}")
            else:
                names.append(f"S{idx + 1}")

    frames = [_parse_single_bismark_or_bedgraph(item, min_cov=min_cov) for item in path_list]

    non_empty_coords = [f[["chr", "pos"]] for f in frames if not f.empty]
    if not non_empty_coords:
        empty_coords = pd.DataFrame(columns=["chr", "pos"])
        empty_mat = np.zeros((0, len(frames)), dtype=np.float64)
        return BismarkCovData(
            frames=frames,
            N_mat=empty_mat,
            X_mat=empty_mat.copy(),
            coords_df=empty_coords,
            sample_names=names,
        )

    coords_df = (
        pd.concat(non_empty_coords, ignore_index=True)
        .drop_duplicates(subset=["chr", "pos"])
        .sort_values(["chr", "pos"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_cpgs = len(coords_df)
    n_samples = len(frames)
    n_mat = np.zeros((n_cpgs, n_samples), dtype=np.float64)
    x_mat = np.zeros((n_cpgs, n_samples), dtype=np.float64)

    key_index = pd.MultiIndex.from_frame(coords_df[["chr", "pos"]])
    for r, sub in enumerate(frames):
        if sub.empty:
            continue
        sub_idx = pd.MultiIndex.from_frame(sub[["chr", "pos"]])
        locs = key_index.get_indexer(sub_idx)
        n_mat[locs, r] = sub["N"].to_numpy(dtype=np.float64)
        x_mat[locs, r] = sub["X"].to_numpy(dtype=np.float64)

    return BismarkCovData(
        frames=frames,
        N_mat=n_mat,
        X_mat=x_mat,
        coords_df=coords_df,
        sample_names=names,
    )


# ---------------------------------------------------------------------------
# 5. MaxQuant proteinGroups.txt & DIA-NN pg_matrix.tsv reader
# ---------------------------------------------------------------------------

_DIANN_META_COLS = frozenset(
    {
        "Protein.Group",
        "Protein.Ids",
        "Protein.Names",
        "Genes",
        "First.Protein.Description",
        "N.Sequences",
        "N.Proteotypic.Sequences",
        "Peptides",
        "count",
    }
)


def read_maxquant_diann(
    path_or_df: str | Path | pd.DataFrame,
    platform: Literal["auto", "maxquant", "diann"] = "auto",
    intensity_prefix: str | None = None,
    filter_contaminants: bool = True,
    log2_transform: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Parse MaxQuant `proteinGroups.txt` or DIA-NN `pg_matrix.tsv` into `(abundance_df, count_series)`.

    Ready directly for `deqms.spectra_count_ebayes(abundance_df, design=..., count=count_series)`
    and `limma`.
    """
    if platform not in ("auto", "maxquant", "diann"):
        raise ValueError(
            f"Unknown platform={platform!r}. Expected 'auto', 'maxquant', or 'diann'."
        )

    raw = _read_table_skipping_prefixes(path_or_df, skip_prefixes=(), sep="\t")
    if raw.empty:
        raise ValueError("Proteomics input table is empty.")

    cols = list(raw.columns)
    eff_platform = platform
    if eff_platform == "auto":
        if any(c in cols for c in ("Protein.Group", "Protein.Ids", "First.Protein.Description")):
            eff_platform = "diann"
        elif any(
            c in cols
            or c.startswith(("LFQ intensity ", "Reporter intensity ", "Intensity "))
            for c in cols
            if c in ("Majority protein IDs", "Protein IDs", "Razor + unique peptides")
            or c.startswith(("LFQ intensity ", "Reporter intensity "))
        ):
            eff_platform = "maxquant"
        else:
            eff_platform = "diann"

    df = raw.copy()

    if eff_platform == "maxquant":
        if filter_contaminants:
            for flag_col in (
                "Potential contaminant",
                "Reverse",
                "Only identified by site",
                "Contaminant",
            ):
                if flag_col in df.columns:
                    is_flagged = df[flag_col].fillna("").astype(str).str.strip() == "+"
                    df = df.loc[~is_flagged]

        id_col = _find_col(
            df,
            ("Majority protein IDs", "Protein IDs", "Gene names", "protein", "gene"),
            case_insensitive=False,
        )
        if id_col is None:
            id_col = str(df.columns[0])

        if intensity_prefix is not None:
            sample_cols = [c for c in df.columns if str(c).startswith(intensity_prefix)]
            clean_names = [str(c)[len(intensity_prefix) :].strip() or str(c) for c in sample_cols]
        else:
            sample_cols = []
            used_prefix = ""
            for pref in (
                "LFQ intensity ",
                "Reporter intensity corrected ",
                "Reporter intensity ",
                "Intensity ",
            ):
                cand = [
                    c
                    for c in df.columns
                    if str(c).startswith(pref)
                    and not str(c).startswith("Reporter intensity count ")
                ]
                if cand:
                    sample_cols = cand
                    used_prefix = pref
                    break
            clean_names = [str(c)[len(used_prefix) :].strip() for c in sample_cols]

        if not sample_cols:
            raise ValueError(
                f"Could not find MaxQuant intensity columns in {list(df.columns)}."
            )

        cnt_col = _find_col(
            df,
            ("Razor + unique peptides", "Unique peptides", "Peptides", "MS/MS count", "count"),
            case_insensitive=False,
        )
        if cnt_col is not None:
            counts = pd.to_numeric(df[cnt_col], errors="coerce").to_numpy(dtype=np.float64)
        else:
            counts = np.ones(len(df), dtype=np.float64)

        prot_ids = (
            df[id_col]
            .fillna("")
            .astype(str)
            .str.split(";")
            .str[0]
            .str.strip()
            .tolist()
        )
    else:
        # DIA-NN pg_matrix.tsv
        id_col = _find_col(
            df,
            ("Protein.Group", "Protein.Ids", "Genes", "protein", "gene"),
            case_insensitive=False,
        )
        if id_col is None:
            id_col = str(df.columns[0])

        if filter_contaminants:
            for check_c in ("Protein.Group", "Protein.Ids", id_col):
                if check_c in df.columns:
                    s_upper = df[check_c].fillna("").astype(str).str.upper()
                    bad = s_upper.str.startswith(("CONT_", "REV_", "CRAP"))
                    df = df.loc[~bad]

        if intensity_prefix is not None:
            sample_cols = [c for c in df.columns if str(c).startswith(intensity_prefix)]
            clean_names = [str(c)[len(intensity_prefix) :].strip() or str(c) for c in sample_cols]
        else:
            meta_exclude = set(_DIANN_META_COLS) | {id_col}
            sample_cols = [c for c in df.columns if c not in meta_exclude]
            clean_names = []
            for c in sample_cols:
                base = Path(str(c).replace("\\", "/")).name
                for ext in (".mzML.gz", ".mzml.gz", ".mzML", ".mzml", ".raw", ".RAW", ".dia", ".wiff", ".d"):
                    if base.endswith(ext) and len(base) > len(ext):
                        base = base[: -len(ext)]
                        break
                clean_names.append(base)

        if not sample_cols:
            raise ValueError(
                f"Could not identify DIA-NN sample intensity columns in {list(df.columns)}."
            )

        cnt_col = _find_col(
            df,
            ("N.Sequences", "N.Proteotypic.Sequences", "Peptides", "count"),
            case_insensitive=False,
        )
        mat_preview = df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if cnt_col is not None:
            counts = pd.to_numeric(df[cnt_col], errors="coerce").to_numpy(dtype=np.float64)
        else:
            # Default count in standard 5-meta-col DIA-NN pg_matrix: quantified runs (>= 1)
            counts = np.maximum(np.sum(np.isfinite(mat_preview) & (mat_preview > 0.0), axis=1), 1).astype(
                np.float64
            )

        prot_ids = (
            df[id_col]
            .fillna("")
            .astype(str)
            .str.split(";")
            .str[0]
            .str.strip()
            .tolist()
        )

    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    mat = np.where(mat <= 0.0, np.nan, mat)
    if log2_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.log2(mat)

    has_signal = np.any(np.isfinite(mat), axis=1) & np.isfinite(counts) & (counts >= 1.0)
    mat = mat[has_signal]
    counts = counts[has_signal]
    kept_ids = [pid if pid else f"PROT_{i + 1}" for i, (pid, ok) in enumerate(zip(prot_ids, has_signal)) if ok]

    # Deduplicate any repeated gene/protein IDs cleanly
    seen: dict[str, int] = {}
    unique_ids: list[str] = []
    for pid in kept_ids:
        if pid not in seen:
            seen[pid] = 1
            unique_ids.append(pid)
        else:
            seen[pid] += 1
            unique_ids.append(f"{pid}_{seen[pid]}")

    idx = pd.Index(unique_ids, name="protein")
    abundance_df = pd.DataFrame(mat, index=idx, columns=clean_names, dtype=np.float64)
    count_series = pd.Series(counts, index=idx, name="count", dtype=np.float64)
    return abundance_df, count_series


# ---------------------------------------------------------------------------
# 6. GWAS summary statistics harmonizer & PLINK square LD reader
# ---------------------------------------------------------------------------


def read_gwas_sumstats(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read and harmonize GWAS summary statistics into standardized columns.

    Output columns: `snp`, `ea`, `nea`, `beta`, `se`, `z`, `pval`, `n`, `eaf`.
    Automatically converts `OR` to `log(OR)` and mutually derives missing `(beta, se, z, pval)`.
    """
    raw = _read_table_skipping_prefixes(path_or_df, skip_prefixes=("##",), sep=None)
    if raw.empty:
        raise ValueError("GWAS summary-statistics input is empty.")

    snp_col = _find_col(
        raw, ("snp", "rsid", "id", "markername", "snpid", "variant_id", "hm_rsid", "rs_id", "#snp")
    )
    if snp_col is None:
        raise ValueError(
            f"GWAS summary statistics missing SNP identifier column; got {list(raw.columns)}."
        )

    ea_col = _find_col(
        raw, ("ea", "a1", "alt", "effect_allele", "allele1", "tested_allele", "hm_effect_allele")
    )
    nea_col = _find_col(
        raw,
        (
            "nea",
            "a2",
            "ref",
            "other_allele",
            "oa",
            "non_effect_allele",
            "allele2",
            "allele0",
            "hm_other_allele",
        ),
    )
    beta_col = _find_col(raw, ("beta", "b", "effect", "hm_beta", "log_odds"))
    or_col = _find_col(raw, ("or", "odds_ratio", "hm_odds_ratio"))
    se_col = _find_col(raw, ("se", "stderr", "standard_error", "sebeta", "hm_standard_error"))
    z_col = _find_col(raw, ("z", "zscore", "z_score", "stat", "tstat"))
    p_col = _find_col(
        raw, ("pval", "p", "p_value", "pvalue", "p-value", "p_bolt_lmm", "firth_p")
    )
    mlog10p_col = _find_col(raw, ("mlog10p", "log10p", "neg_log10_p"))
    n_col = _find_col(raw, ("n", "samplesize", "sample_size", "n_total", "neff", "n_eff", "obs_ct"))
    eaf_col = _find_col(
        raw,
        ("eaf", "af", "frq", "freq", "a1freq", "af_allele1", "maf", "hm_effect_allele_frequency"),
    )

    if all(c is None for c in (beta_col, or_col, z_col, p_col, mlog10p_col)):
        raise ValueError(
            f"GWAS summary statistics must provide at least one effect/test column "
            f"(BETA, OR, Z, or P); got {list(raw.columns)}."
        )

    m = len(raw)
    ea = (
        raw[ea_col].fillna("").astype(str).str.strip().str.upper()
        if ea_col is not None
        else pd.Series([""] * m, index=raw.index)
    )
    nea = (
        raw[nea_col].fillna("").astype(str).str.strip().str.upper()
        if nea_col is not None
        else pd.Series([""] * m, index=raw.index)
    )

    if beta_col is not None:
        beta = pd.to_numeric(raw[beta_col], errors="coerce").to_numpy(dtype=np.float64)
    elif or_col is not None:
        or_vals = pd.to_numeric(raw[or_col], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(or_vals > 0.0, np.log(or_vals), np.nan)
    else:
        beta = np.full(m, np.nan, dtype=np.float64)

    se = (
        pd.to_numeric(raw[se_col], errors="coerce").to_numpy(dtype=np.float64)
        if se_col is not None
        else np.full(m, np.nan, dtype=np.float64)
    )
    z = (
        pd.to_numeric(raw[z_col], errors="coerce").to_numpy(dtype=np.float64)
        if z_col is not None
        else np.full(m, np.nan, dtype=np.float64)
    )

    if p_col is not None:
        pval = pd.to_numeric(raw[p_col], errors="coerce").to_numpy(dtype=np.float64)
    elif mlog10p_col is not None:
        ml10 = pd.to_numeric(raw[mlog10p_col], errors="coerce").to_numpy(dtype=np.float64)
        pval = np.power(10.0, -ml10)
    else:
        pval = np.full(m, np.nan, dtype=np.float64)

    n_arr = (
        pd.to_numeric(raw[n_col], errors="coerce").to_numpy(dtype=np.float64)
        if n_col is not None
        else np.full(m, np.nan, dtype=np.float64)
    )
    eaf = (
        pd.to_numeric(raw[eaf_col], errors="coerce").to_numpy(dtype=np.float64)
        if eaf_col is not None
        else np.full(m, np.nan, dtype=np.float64)
    )

    # Mutual derivations across (beta, se, z, pval)
    need_z = ~np.isfinite(z) & np.isfinite(beta) & np.isfinite(se) & (se > 0.0)
    z[need_z] = beta[need_z] / se[need_z]

    need_z_from_p = (
        ~np.isfinite(z)
        & np.isfinite(pval)
        & (pval > 0.0)
        & (pval <= 1.0)
        & np.isfinite(beta)
    )
    if np.any(need_z_from_p):
        sign_b = np.where(beta[need_z_from_p] >= 0.0, 1.0, -1.0)
        z[need_z_from_p] = sign_b * norm.isf(0.5 * pval[need_z_from_p])

    need_se = ~np.isfinite(se) & np.isfinite(beta) & np.isfinite(z) & (np.abs(z) > 1e-15)
    se[need_se] = np.abs(beta[need_se] / z[need_se])

    need_beta = ~np.isfinite(beta) & np.isfinite(z) & np.isfinite(se) & (se > 0.0)
    beta[need_beta] = z[need_beta] * se[need_beta]

    need_p = ~np.isfinite(pval) & np.isfinite(z)
    pval[need_p] = 2.0 * norm.sf(np.abs(z[need_p]))

    out = pd.DataFrame(
        {
            "snp": raw[snp_col].astype(str).str.strip(),
            "ea": ea,
            "nea": nea,
            "beta": beta,
            "se": se,
            "z": z,
            "pval": pval,
            "n": n_arr,
            "eaf": eaf,
        }
    )
    return out.reset_index(drop=True)


def read_plink_ld(
    ld_path: str | Path,
    bim_path: str | Path | None = None,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Load a square PLINK `--r square` `.ld` / `.ld.gz` correlation matrix.

    Returns `R` (`np.ndarray` of shape `(M, M)`) when `bim_path is None`, or
    `(R, snp_ids)` when `bim_path` (`.bim` / `.bim.gz`) is provided.
    """
    lines: list[str] = []
    with _open_text(ld_path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(line)

    if not lines:
        raise ValueError(f"LD matrix file {ld_path!r} is empty.")

    first_tokens = lines[0].split()
    has_header = False
    try:
        for tok in first_tokens:
            float(tok)
    except ValueError:
        has_header = True

    buf = _pyio.StringIO("".join(lines))
    if has_header:
        df = pd.read_csv(buf, sep=r"\s+", header=0, engine="python")
        if df.shape[1] == df.shape[0] + 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            df = df.iloc[:, 1:]
        r_mat = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    else:
        r_mat = np.loadtxt(buf, dtype=np.float64)
        if r_mat.ndim == 1:
            r_mat = r_mat.reshape(1, 1)

    if r_mat.ndim != 2 or r_mat.shape[0] != r_mat.shape[1]:
        raise ValueError(
            f"LD matrix in {ld_path!r} must be square 2D; got shape {r_mat.shape}."
        )
    if not np.all(np.isfinite(r_mat)):
        raise ValueError(f"LD matrix in {ld_path!r} contains NaN or infinite entries.")

    r_mat = 0.5 * (r_mat + r_mat.T)

    if bim_path is None:
        return r_mat

    snp_ids: list[str] = []
    with _open_text(bim_path) as bfh:
        for bline in bfh:
            bs = bline.strip()
            if not bs or bs.startswith("#"):
                continue
            parts = bs.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed PLINK .bim line in {bim_path!r}: {bline!r}")
            snp_ids.append(parts[1])

    if len(snp_ids) != r_mat.shape[0]:
        raise ValueError(
            f"SNP count in {bim_path!r} ({len(snp_ids)}) does not match LD matrix dimension ({r_mat.shape[0]})."
        )
    return r_mat, snp_ids
