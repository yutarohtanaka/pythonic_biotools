"""
Core `QDNAseqData` container, `pysam` BAM read binning (`bin_read_counts`),
and genomic bin QC filtering (`apply_filters`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pysam

from .config import SettingsMode, get_qdnaseq_config, validate_settings


def canonical_chrom_name(chrom: str) -> str:
    """Normalize chromosome string by stripping leading `chr` and mapping `M` -> `MT`."""
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    if c.upper() == "M":
        return "MT"
    return c.upper()


@dataclass
class QDNAseqData:
    """
    Multi-sample binned copy-number data structure analogous to Bioconductor's
    `QDNAseqReadCounts` / `QDNAseqCopyNumbers`.
    """

    bins: pd.DataFrame
    counts: pd.DataFrame
    fit: pd.DataFrame | None = None
    copynumber: pd.DataFrame | None = None
    segmented: pd.DataFrame | None = None
    calls: pd.DataFrame | None = None
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    probabilities: dict[str, pd.DataFrame] = field(default_factory=dict)
    settings: SettingsMode = "default"

    @property
    def sample_names(self) -> list[str]:
        return list(self.counts.columns)

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def n_used_bins(self) -> int:
        return int(self.bins["use"].sum())

    def clone(self, **updates: Any) -> QDNAseqData:
        base = QDNAseqData(
            bins=self.bins.copy(),
            counts=self.counts.copy(),
            fit=self.fit.copy() if self.fit is not None else None,
            copynumber=self.copynumber.copy() if self.copynumber is not None else None,
            segmented=self.segmented.copy() if self.segmented is not None else None,
            calls=self.calls.copy() if self.calls is not None else None,
            segments={k: v.copy() for k, v in self.segments.items()},
            probabilities={k: v.copy() for k, v in self.probabilities.items()},
            settings=self.settings,
        )
        return replace(base, **updates)

    def to_dataframe(self, sample: str | None = None) -> pd.DataFrame:
        """Export bin annotations and processed tracks for a sample (or all samples)."""
        out = self.bins.copy()
        target = sample or (self.sample_names[0] if len(self.sample_names) == 1 else None)
        if target is not None:
            if target not in self.counts.columns:
                raise ValueError(f"Unknown sample {target!r}; available: {self.sample_names}.")
            out["count"] = self.counts[target].to_numpy()
            if self.fit is not None:
                out["fit"] = self.fit[target].to_numpy()
            if self.copynumber is not None:
                out["copynumber"] = self.copynumber[target].to_numpy()
            if self.segmented is not None:
                out["segmented"] = self.segmented[target].to_numpy()
            if self.calls is not None:
                out["call"] = self.calls[target].to_numpy()
        else:
            for s in self.sample_names:
                out[f"{s}_count"] = self.counts[s].to_numpy()
                if self.copynumber is not None:
                    out[f"{s}_copynumber"] = self.copynumber[s].to_numpy()
                if self.segmented is not None:
                    out[f"{s}_segmented"] = self.segmented[s].to_numpy()
                if self.calls is not None:
                    out[f"{s}_call"] = self.calls[s].to_numpy()
        return out

    def summary(self) -> str:
        stage = "raw_counts"
        if self.calls is not None:
            stage = "called"
        elif self.segmented is not None:
            stage = "segmented"
        elif self.copynumber is not None:
            stage = "corrected"
        elif self.fit is not None:
            stage = "estimated"
        return (
            f"QDNAseqData (settings={self.settings}, stage={stage})\n"
            f"  Samples ({len(self.sample_names)}): {', '.join(self.sample_names)}\n"
            f"  Bins: {self.n_used_bins} active / {self.n_bins} total"
        )


def _standardize_bins_df(bins_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardize bin annotation columns and return `(bins_clean, extra_count_cols)`."""
    if not isinstance(bins_df, pd.DataFrame) or bins_df.empty:
        raise ValueError("`bins_df` must be a non-empty pandas DataFrame.")

    df = bins_df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    renames: dict[str, str] = {}
    if "chrom" not in df.columns:
        for alias in ("chr", "chromosome", "seqnames"):
            if alias in lower_map:
                renames[lower_map[alias]] = "chrom"
                break
    if "gc" not in df.columns:
        for alias in ("gc_content", "pct_gc"):
            if alias in lower_map:
                renames[lower_map[alias]] = "gc"
                break
    if "mappability" not in df.columns:
        for alias in ("map", "mapp"):
            if alias in lower_map:
                renames[lower_map[alias]] = "mappability"
                break
    if renames:
        df = df.rename(columns=renames)

    for req in ("chrom", "start", "end"):
        if req not in df.columns:
            raise ValueError(f"`bins_df` is missing required column {req!r}.")

    n = len(df)
    out = pd.DataFrame(
        {
            "chrom": df["chrom"].astype(str),
            "start": df["start"].astype(np.int64),
            "end": np.maximum(df["end"].astype(np.int64), df["start"].astype(np.int64)),
        }
    )

    if "gc" in df.columns:
        gc = df["gc"].astype(np.float64).to_numpy()
        fin = gc[np.isfinite(gc)]
        # QDNAseq uses 0..100 scale; convert 0..1 fractions seamlessly
        if fin.size > 0 and float(np.nanmax(fin)) <= 1.05:
            gc = gc * 100.0
        out["gc"] = gc
    else:
        out["gc"] = np.full(n, 45.0, dtype=np.float64)

    if "mappability" in df.columns:
        mapp = df["mappability"].astype(np.float64).to_numpy()
        fin_m = mapp[np.isfinite(mapp)]
        if fin_m.size > 0 and float(np.nanmax(fin_m)) <= 1.05 and float(np.nanmax(fin_m)) > 0.0:
            mapp = mapp * 100.0
        out["mappability"] = mapp
    else:
        out["mappability"] = np.full(n, 100.0, dtype=np.float64)

    if "blacklist" in df.columns:
        out["blacklist"] = df["blacklist"].fillna(0.0).astype(np.float64)
    else:
        out["blacklist"] = np.zeros(n, dtype=np.float64)

    if "residual" in df.columns:
        out["residual"] = df["residual"].astype(np.float64)
    else:
        out["residual"] = np.zeros(n, dtype=np.float64)

    if "use" in df.columns:
        out["use"] = df["use"].fillna(True).astype(bool) & np.isfinite(out["gc"])
    else:
        out["use"] = np.isfinite(out["gc"].to_numpy())

    known_meta = {
        "chrom",
        "start",
        "end",
        "gc",
        "mappability",
        "blacklist",
        "residual",
        "use",
        "feature",
        "bin_id",
    }
    extra_cols = [
        c
        for c in df.columns
        if c not in known_meta and np.issubdtype(df[c].dtype, np.number)
    ]
    return out.reset_index(drop=True), extra_cols


def _count_bam_in_bins(
    bam_path: str | Path,
    bins: pd.DataFrame,
    min_mapq: int,
) -> np.ndarray:
    """Count primary, non-duplicate alignments with `mapq >= min_mapq` per genomic bin."""
    n_bins = len(bins)
    counts = np.zeros(n_bins, dtype=np.float64)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        bam_refs = set(bam.references)
        for chrom in pd.unique(bins["chrom"]):
            mask = (bins["chrom"] == chrom).to_numpy()
            idxs = np.flatnonzero(mask)
            sub_starts = bins.loc[idxs, "start"].to_numpy(dtype=np.int64)
            sub_ends = bins.loc[idxs, "end"].to_numpy(dtype=np.int64)
            order = np.argsort(sub_starts, kind="mergesort")
            s_sorted = sub_starts[order]
            e_sorted = sub_ends[order]
            idxs_sorted = idxs[order]

            ch_str = str(chrom)
            alt_ch = ch_str[3:] if ch_str.startswith("chr") else f"chr{ch_str}"
            target_ref = ch_str if ch_str in bam_refs else (alt_ch if alt_ch in bam_refs else None)
            if target_ref is None:
                continue

            # Fetch reads across chromosome span (1-based start -> 0-based start - 1)
            region_start0 = max(0, int(s_sorted[0]) - 1)
            region_end0 = int(e_sorted[-1])
            read_pos1_list: list[int] = []
            for read in bam.fetch(target_ref, region_start0, region_end0):
                if (
                    read.is_unmapped
                    or read.is_duplicate
                    or read.is_secondary
                    or read.is_supplementary
                    or read.is_qcfail
                ):
                    continue
                if read.mapping_quality < min_mapq:
                    continue
                # 1-based 5' alignment start coordinate
                pos1 = int(read.reference_start) + 1
                read_pos1_list.append(pos1)

            if not read_pos1_list:
                continue
            r_pos = np.asarray(read_pos1_list, dtype=np.int64)
            cand = np.searchsorted(s_sorted, r_pos, side="right") - 1
            valid = (cand >= 0) & (r_pos <= e_sorted[np.maximum(cand, 0)])
            hit_bins = cand[valid]
            if hit_bins.size > 0:
                bin_tally = np.bincount(hit_bins, minlength=s_sorted.size)
                counts[idxs_sorted] += bin_tally.astype(np.float64)

    return counts


def bin_read_counts(
    bins_df: pd.DataFrame,
    bam_paths: str | Path | Sequence[str | Path] | dict[str, str | Path] | None = None,
    counts_df: pd.DataFrame | pd.Series | np.ndarray | None = None,
    min_mapq: int | None = None,
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Count `pysam` BAM alignments into fixed genomic bins or wrap pre-binned count
    matrices alongside QDNAseq bin annotations (`gc`, `mappability`, `blacklist`, `residual`).
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    eff_mapq = int(min_mapq) if min_mapq is not None else cfg.min_mapq
    if eff_mapq < 0:
        raise ValueError(f"`min_mapq` must be >= 0, got {min_mapq}.")

    bins_clean, extra_cols = _standardize_bins_df(bins_df)
    n = len(bins_clean)

    if bam_paths is not None:
        if isinstance(bam_paths, (str, Path)):
            p = Path(bam_paths)
            bam_map = {p.stem: p}
        elif isinstance(bam_paths, dict):
            bam_map = {str(k): Path(v) for k, v in bam_paths.items()}
        else:
            bam_map = {}
            for idx, item in enumerate(bam_paths):
                p = Path(item)
                name = p.stem if p.stem not in bam_map else f"{p.stem}_{idx}"
                bam_map[name] = p

        if not bam_map:
            raise ValueError("`bam_paths` cannot be empty.")

        count_cols = {
            sname: _count_bam_in_bins(bpath, bins_clean, eff_mapq)
            for sname, bpath in bam_map.items()
        }
        counts_out = pd.DataFrame(count_cols)
    elif counts_df is not None:
        if isinstance(counts_df, pd.DataFrame):
            if len(counts_df) != n:
                raise ValueError(
                    f"`counts_df` length ({len(counts_df)}) must match `bins_df` ({n})."
                )
            counts_out = counts_df.reset_index(drop=True).astype(np.float64)
        elif isinstance(counts_df, pd.Series):
            if len(counts_df) != n:
                raise ValueError(
                    f"`counts_df` length ({len(counts_df)}) must match `bins_df` ({n})."
                )
            sname = str(counts_df.name) if counts_df.name is not None else "sample"
            counts_out = pd.DataFrame({sname: counts_df.to_numpy(dtype=np.float64)})
        else:
            arr = np.asarray(counts_df, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if arr.shape[0] != n:
                raise ValueError(
                    f"`counts_df` row count ({arr.shape[0]}) must match `bins_df` ({n})."
                )
            cols = (
                ["sample"]
                if arr.shape[1] == 1
                else [f"sample_{i + 1}" for i in range(arr.shape[1])]
            )
            counts_out = pd.DataFrame(arr, columns=cols)
    elif extra_cols:
        counts_out = bins_df[extra_cols].reset_index(drop=True).astype(np.float64)
    else:
        raise ValueError(
            "Provide `bam_paths`, `counts_df`, or count columns inside `bins_df`."
        )

    counts_out = counts_out.clip(lower=0.0)
    return QDNAseqData(
        bins=bins_clean,
        counts=counts_out,
        settings=cfg.settings,
    )


def apply_filters(
    qdata: QDNAseqData,
    residual: bool | float = True,
    blacklist: bool | float = True,
    mappability: float | bool = 50.0,
    chromosomes: Sequence[str] = ("X", "Y", "MT"),
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Filter problematic genomic bins by empirical normal residuals, ENCODE blacklist,
    minimum mappability score, and sex/mitochondrial chromosomes (`QDNAseq::applyFilters`).
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    if not isinstance(qdata, QDNAseqData):
        raise ValueError("`qdata` must be a `QDNAseqData` instance.")

    bins = qdata.bins.copy()
    keep = np.isfinite(bins["gc"].to_numpy(dtype=np.float64))

    if chromosomes:
        excl = {canonical_chrom_name(c) for c in chromosomes}
        bin_chroms = [canonical_chrom_name(c) for c in bins["chrom"]]
        keep &= np.array([c not in excl for c in bin_chroms], dtype=bool)

    if blacklist is not False and blacklist is not None:
        bl_vals = bins["blacklist"].to_numpy(dtype=np.float64)
        if isinstance(blacklist, bool):
            keep &= bl_vals <= 0.0
        else:
            keep &= bl_vals <= float(blacklist)

    if mappability is not False and mappability is not None:
        min_mapp = (
            cfg.mappability_cutoff if isinstance(mappability, bool) else float(mappability)
        )
        keep &= bins["mappability"].to_numpy(dtype=np.float64) >= min_mapp

    if residual is not False and residual is not None:
        res_vals = bins["residual"].to_numpy(dtype=np.float64)
        res_cut = (
            cfg.residual_sd_cutoff if isinstance(residual, bool) else float(residual)
        )
        keep &= np.isfinite(res_vals) & (np.abs(res_vals) <= res_cut)

    bins["use"] = keep
    return qdata.clone(bins=bins, settings=cfg.settings)
