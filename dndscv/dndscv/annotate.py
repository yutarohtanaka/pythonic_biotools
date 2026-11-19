"""
Mutation annotation (`annotate_mutations`), MAF/5-column aliasing,
coding-strand 192-context mapping, and sample/gene hypermutation capping.
"""

from __future__ import annotations

import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .refcds import (
    BASES,
    CODON_TABLE,
    COMPLEMENT,
    RefCDS,
    _TRINUC_ALT_TO_IDX,
    canonical_chrom,
    classify_codon_impact,
)

_VALID_ALLELE_CHARS = frozenset(("A", "C", "G", "T", "N", "-"))

_ANNOT_COLUMNS = [
    "sampleID",
    "chr",
    "pos",
    "ref",
    "mut",
    "gene_name",
    "strand",
    "ref_cod",
    "mut_cod",
    "ref3_cod",
    "mut3_cod",
    "trinuc_ind",
    "impact",
    "aachange",
]

_IMPACT_TO_CLS = {
    "Synonymous": 0,
    "Missense": 1,
    "Nonsense": 2,
    "Essential_Splice": 3,
}


@dataclass(frozen=True)
class AnnotationResult:
    """Annotated coding mutations and sufficient statistics for `dndscv`."""

    annot_muts: pd.DataFrame
    n_obs: np.ndarray  # (G, 192, 4) int64 for (syn, mis, non, spl)
    n_ind: np.ndarray  # (G,) int64
    gene_names: list[str]
    excluded_samples: list[str]
    n_raw_muts: int
    n_dedup_muts: int
    n_coding_muts: int
    n_capped_muts: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return annotated coding mutations as a pandas DataFrame."""
        return self.annot_muts.copy()


class _ChromIndex:
    """Per-chromosome sorted exon and splice lookup index."""

    def __init__(self, refcds: RefCDS) -> None:
        self.exon_starts: dict[str, np.ndarray] = {}
        self.exon_ends: dict[str, np.ndarray] = {}
        self.exon_max_end: dict[str, np.ndarray] = {}
        self.exon_meta: dict[str, list[tuple[int, int, int, int]]] = {}  # (g_idx, cds_off, s, e)

        self.splice_pos: dict[str, np.ndarray] = {}
        self.splice_meta: dict[str, list[tuple[int, int]]] = {}  # (g_idx, splice_idx)

        exon_buckets: dict[str, list[tuple[int, int, int, int]]] = {}
        splice_buckets: dict[str, list[tuple[int, int, int]]] = {}

        for g_idx, gene in enumerate(refcds.genes):
            chrom = gene.chrom
            cds_off = 0
            eb = exon_buckets.setdefault(chrom, [])
            for s, e in gene.intervals_cds:
                s_i, e_i = int(s), int(e)
                eb.append((s_i, e_i, g_idx, cds_off))
                cds_off += e_i - s_i + 1

            sb = splice_buckets.setdefault(chrom, [])
            for sp_idx, p in enumerate(gene.intervals_splice):
                sb.append((int(p), g_idx, sp_idx))

        for chrom, items in exon_buckets.items():
            items.sort(key=lambda x: (x[0], x[1], x[2]))
            starts = np.asarray([x[0] for x in items], dtype=np.int64)
            ends = np.asarray([x[1] for x in items], dtype=np.int64)
            self.exon_starts[chrom] = starts
            self.exon_ends[chrom] = ends
            self.exon_max_end[chrom] = np.maximum.accumulate(ends)
            self.exon_meta[chrom] = [(x[2], x[3], x[0], x[1]) for x in items]

        for chrom, items in splice_buckets.items():
            items.sort(key=lambda x: (x[0], x[1]))
            self.splice_pos[chrom] = np.asarray([x[0] for x in items], dtype=np.int64)
            self.splice_meta[chrom] = [(x[1], x[2]) for x in items]

    def query_exons(self, chrom: str, q_start: int, q_end: int) -> list[tuple[int, int, int, int]]:
        starts = self.exon_starts.get(chrom)
        if starts is None or starts.size == 0:
            return []
        ends = self.exon_ends[chrom]
        max_end = self.exon_max_end[chrom]
        meta = self.exon_meta[chrom]

        hi = int(np.searchsorted(starts, q_end, side="right"))
        hits: list[tuple[int, int, int, int]] = []
        for idx in range(hi - 1, -1, -1):
            if max_end[idx] < q_start:
                break
            if ends[idx] >= q_start:
                hits.append(meta[idx])
        hits.reverse()
        return hits

    def query_splice(self, chrom: str, q_start: int, q_end: int) -> list[tuple[int, int]]:
        positions = self.splice_pos.get(chrom)
        if positions is None or positions.size == 0:
            return []
        lo = int(np.searchsorted(positions, q_start, side="left"))
        hi = int(np.searchsorted(positions, q_end, side="right"))
        meta = self.splice_meta[chrom]
        return [meta[idx] for idx in range(lo, hi)]


def _resolve_mut_col(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    col_map = {str(c): str(c) for c in df.columns}
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(
        f"Missing required mutation column for {role} (accepted: {candidates}). "
        f"Found columns: {list(df.columns)}"
    )


def annotate_mutations(
    mutations: pd.DataFrame,
    refcds: RefCDS,
    *,
    max_muts_per_gene_per_sample: int | None = 3,
    max_coding_muts_per_sample: int | None = 3000,
    coord_system: Literal["one_based", "zero_based"] = "one_based",
) -> AnnotationResult:
    """
    Map SNVs and indels onto `RefCDS`, classify impacts across 192 coding-strand
    trinucleotide contexts, and apply hypermutator/per-gene sample caps.
    """
    if not isinstance(mutations, pd.DataFrame):
        raise ValueError("mutations must be a pandas DataFrame.")
    if coord_system not in {"one_based", "zero_based"}:
        raise ValueError(f"Invalid coord_system {coord_system!r}; expected 'one_based' or 'zero_based'.")
    if max_muts_per_gene_per_sample is not None and max_muts_per_gene_per_sample < 1:
        raise ValueError("max_muts_per_gene_per_sample must be >= 1 or None.")
    if max_coding_muts_per_sample is not None and max_coding_muts_per_sample < 1:
        raise ValueError("max_coding_muts_per_sample must be >= 1 or None.")

    n_genes = len(refcds)
    sample_col = _resolve_mut_col(
        mutations, ["sampleID", "Tumor_Sample_Barcode", "sample_id", "sample", "Sample_ID"], "sampleID"
    )
    chr_col = _resolve_mut_col(mutations, ["chr", "chrom", "Chromosome", "CHROM"], "chromosome")
    pos_col = _resolve_mut_col(
        mutations, ["pos", "Start_Position", "Start_position", "POS", "start"], "position"
    )
    ref_col = _resolve_mut_col(mutations, ["ref", "Reference_Allele", "REF"], "reference allele")
    mut_col = _resolve_mut_col(
        mutations, ["mut", "Tumor_Seq_Allele2", "alt", "ALT"], "mutant allele"
    )

    if mutations.empty:
        return AnnotationResult(
            annot_muts=pd.DataFrame(columns=_ANNOT_COLUMNS),
            n_obs=np.zeros((n_genes, 192, 4), dtype=np.int64),
            n_ind=np.zeros(n_genes, dtype=np.int64),
            gene_names=list(refcds.gene_names),
            excluded_samples=[],
            n_raw_muts=0,
            n_dedup_muts=0,
            n_coding_muts=0,
            n_capped_muts=0,
        )

    n_raw = int(len(mutations))
    work = mutations[[sample_col, chr_col, pos_col, ref_col, mut_col]].copy()
    work.columns = ["sampleID", "chr", "pos", "ref", "mut"]

    # Validate non-missing sample/chr/pos and numeric coordinates
    if work["sampleID"].isna().any() or work["chr"].isna().any():
        raise ValueError("mutations contains missing values in sampleID or chromosome columns.")
    pos_num = pd.to_numeric(work["pos"], errors="coerce")
    if pos_num.isna().any() or not np.all(np.isfinite(pos_num.to_numpy(dtype=np.float64))):
        raise ValueError("mutations contains non-numeric or missing genomic positions.")
    pos_int = pos_num.to_numpy(dtype=np.int64)
    if not np.all(pos_num.to_numpy(dtype=np.float64) == pos_int):
        raise ValueError("mutations genomic positions must be integers.")

    if coord_system == "zero_based":
        if np.any(pos_int < 0):
            raise ValueError("zero_based mutation positions must be >= 0.")
        pos_int = pos_int + 1
    else:
        if np.any(pos_int < 1):
            raise ValueError("one_based mutation positions must be >= 1.")

    work["pos"] = pos_int
    work["sampleID"] = work["sampleID"].astype(str).str.strip()
    work["chr"] = [canonical_chrom(c) for c in work["chr"]]

    # Clean alleles; drop missing/identical ref==mut rows
    work = work.loc[work["ref"].notna() & work["mut"].notna()].copy()
    work["ref"] = work["ref"].astype(str).str.strip().str.upper()
    work["mut"] = work["mut"].astype(str).str.strip().str.upper()
    work = work.loc[(work["ref"] != "") & (work["mut"] != "") & (work["ref"] != work["mut"])].copy()

    # Validate allele characters
    for allele_col in ("ref", "mut"):
        for val in work[allele_col]:
            if not set(val).issubset(_VALID_ALLELE_CHARS):
                raise ValueError(f"Invalid characters in {allele_col} allele {val!r}.")

    # Deduplicate exact (sampleID, chr, pos, ref, mut) rows
    work = work.drop_duplicates(subset=["sampleID", "chr", "pos", "ref", "mut"], keep="first").reset_index(
        drop=True
    )
    n_dedup = int(len(work))

    chrom_index = _ChromIndex(refcds)
    genes = refcds.genes

    mapped_rows: list[dict[str, object]] = []
    # Track unique coding/splice/indel variants per sample for hypermutator filter
    sample_unique_muts: dict[str, set[tuple[str, int, str, str]]] = {}

    for row in work.itertuples(index=False):
        sample_id: str = row.sampleID
        chrom: str = row.chr
        pos: int = int(row.pos)
        ref: str = row.ref
        mut: str = row.mut

        is_snv = len(ref) == 1 and len(mut) == 1 and ref in BASES and mut in BASES
        is_indel = (len(ref) != len(mut)) or ("-" in ref) or ("-" in mut)

        if is_snv:
            exon_hits = chrom_index.query_exons(chrom, pos, pos)
            hit_genes: set[int] = set()
            for g_idx, cds_off, exon_s, exon_e in exon_hits:
                if g_idx in hit_genes:
                    continue
                gene = genes[g_idx]
                cds_idx = cds_off + (pos - exon_s) if gene.strand == 1 else cds_off + (exon_e - pos)
                ref_cod = ref if gene.strand == 1 else COMPLEMENT[ref]
                mut_cod = mut if gene.strand == 1 else COMPLEMENT[mut]
                expected_base = gene.seq_cds[cds_idx]
                if expected_base in BASES and ref_cod != expected_base:
                    warnings.warn(
                        f"Reference base mismatch at {chrom}:{pos} in {gene.gene_name}: "
                        f"observed {ref_cod}, expected {expected_base} on coding strand.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                b0 = gene.seq_cds1up[cds_idx]
                b1 = ref_cod
                b2 = gene.seq_cds1down[cds_idx]
                ref3_cod = b0 + b1 + b2
                mut3_cod = b0 + mut_cod + b2
                trinuc_ind = _TRINUC_ALT_TO_IDX.get((ref3_cod, mut_cod), -1)

                codon_idx = cds_idx // 3
                frame = cds_idx % 3
                ref_codon = gene.seq_cds[3 * codon_idx : 3 * codon_idx + 3]
                mut_codon = ref_codon[:frame] + mut_cod + ref_codon[frame + 1 :]
                cls_idx = classify_codon_impact(ref_codon, mut_codon)
                if cls_idx is None:
                    continue
                impact = ("Synonymous", "Missense", "Nonsense")[cls_idx]
                aa_ref = CODON_TABLE.get(ref_codon, "X")
                aa_mut = CODON_TABLE.get(mut_codon, "X")
                aachange = f"{aa_ref}{codon_idx + 1}{aa_mut}"

                hit_genes.add(g_idx)
                mapped_rows.append(
                    {
                        "sampleID": sample_id,
                        "chr": chrom,
                        "pos": pos,
                        "ref": ref,
                        "mut": mut,
                        "gene_name": gene.gene_name,
                        "strand": int(gene.strand),
                        "ref_cod": ref_cod,
                        "mut_cod": mut_cod,
                        "ref3_cod": ref3_cod,
                        "mut3_cod": mut3_cod,
                        "trinuc_ind": trinuc_ind,
                        "impact": impact,
                        "aachange": aachange,
                        "_gene_idx": g_idx,
                    }
                )
                sample_unique_muts.setdefault(sample_id, set()).add((chrom, pos, ref, mut))

            splice_hits = chrom_index.query_splice(chrom, pos, pos)
            for g_idx, sp_idx in splice_hits:
                if g_idx in hit_genes:
                    continue
                gene = genes[g_idx]
                ref_cod = ref if gene.strand == 1 else COMPLEMENT[ref]
                mut_cod = mut if gene.strand == 1 else COMPLEMENT[mut]
                ref3_cod = gene.splice_trinucs[sp_idx] if sp_idx < len(gene.splice_trinucs) else f"N{ref_cod}N"
                if len(ref3_cod) == 3 and ref3_cod[1] in BASES and ref_cod != ref3_cod[1]:
                    warnings.warn(
                        f"Reference base mismatch at splice site {chrom}:{pos} in {gene.gene_name}: "
                        f"observed {ref_cod}, expected {ref3_cod[1]} on coding strand.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                mut3_cod = ref3_cod[0] + mut_cod + ref3_cod[2]
                trinuc_ind = _TRINUC_ALT_TO_IDX.get((ref3_cod, mut_cod), -1)
                hit_genes.add(g_idx)
                mapped_rows.append(
                    {
                        "sampleID": sample_id,
                        "chr": chrom,
                        "pos": pos,
                        "ref": ref,
                        "mut": mut,
                        "gene_name": gene.gene_name,
                        "strand": int(gene.strand),
                        "ref_cod": ref_cod,
                        "mut_cod": mut_cod,
                        "ref3_cod": ref3_cod,
                        "mut3_cod": mut3_cod,
                        "trinuc_ind": trinuc_ind,
                        "impact": "Essential_Splice",
                        "aachange": "splice",
                        "_gene_idx": g_idx,
                    }
                )
                sample_unique_muts.setdefault(sample_id, set()).add((chrom, pos, ref, mut))

        elif is_indel:
            q_start = pos
            q_end = pos if ref == "-" else pos + len(ref) - 1
            exon_hits = chrom_index.query_exons(chrom, q_start, q_end)
            splice_hits = chrom_index.query_splice(chrom, q_start, q_end)
            hit_genes = {h[0] for h in exon_hits} | {h[0] for h in splice_hits}
            for g_idx in sorted(hit_genes):
                gene = genes[g_idx]
                mapped_rows.append(
                    {
                        "sampleID": sample_id,
                        "chr": chrom,
                        "pos": pos,
                        "ref": ref,
                        "mut": mut,
                        "gene_name": gene.gene_name,
                        "strand": int(gene.strand),
                        "ref_cod": ref,
                        "mut_cod": mut,
                        "ref3_cod": "-",
                        "mut3_cod": "-",
                        "trinuc_ind": -1,
                        "impact": "Indel",
                        "aachange": "indel",
                        "_gene_idx": g_idx,
                    }
                )
                sample_unique_muts.setdefault(sample_id, set()).add((chrom, pos, ref, mut))
        else:
            # Complex / MNV (equal length > 1 without '-'): map for reporting, exclude from SNV/indel counts
            q_start = pos
            q_end = pos + len(ref) - 1
            exon_hits = chrom_index.query_exons(chrom, q_start, q_end)
            splice_hits = chrom_index.query_splice(chrom, q_start, q_end)
            hit_genes = {h[0] for h in exon_hits} | {h[0] for h in splice_hits}
            for g_idx in sorted(hit_genes):
                gene = genes[g_idx]
                mapped_rows.append(
                    {
                        "sampleID": sample_id,
                        "chr": chrom,
                        "pos": pos,
                        "ref": ref,
                        "mut": mut,
                        "gene_name": gene.gene_name,
                        "strand": int(gene.strand),
                        "ref_cod": ref,
                        "mut_cod": mut,
                        "ref3_cod": "-",
                        "mut3_cod": "-",
                        "trinuc_ind": -1,
                        "impact": "Complex",
                        "aachange": "complex",
                        "_gene_idx": g_idx,
                    }
                )

    if not mapped_rows:
        return AnnotationResult(
            annot_muts=pd.DataFrame(columns=_ANNOT_COLUMNS),
            n_obs=np.zeros((n_genes, 192, 4), dtype=np.int64),
            n_ind=np.zeros(n_genes, dtype=np.int64),
            gene_names=list(refcds.gene_names),
            excluded_samples=[],
            n_raw_muts=n_raw,
            n_dedup_muts=n_dedup,
            n_coding_muts=0,
            n_capped_muts=0,
        )

    annot_df = pd.DataFrame(mapped_rows)

    # Hypermutator exclusion based on coding+splice+indel counts per sample
    excluded_samples: list[str] = []
    if max_coding_muts_per_sample is not None:
        excluded_samples = sorted(
            sid for sid, muts_set in sample_unique_muts.items() if len(muts_set) > max_coding_muts_per_sample
        )
        if excluded_samples:
            excl_set = set(excluded_samples)
            annot_df = annot_df.loc[~annot_df["sampleID"].isin(excl_set)].copy()

    # Deterministic sort before per-gene-sample capping
    annot_df = annot_df.sort_values(
        ["sampleID", "gene_name", "chr", "pos", "ref", "mut"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Apply per-gene-per-sample cap strictly on coding+splice+indel mutations
    n_capped = 0
    if max_muts_per_gene_per_sample is not None and not annot_df.empty:
        is_counted = annot_df["impact"] != "Complex"
        counted_df = annot_df.loc[is_counted].copy()
        if not counted_df.empty:
            ranks = counted_df.groupby(["sampleID", "gene_name"], sort=False).cumcount()
            keep_counted = ranks < max_muts_per_gene_per_sample
            n_capped = int((~keep_counted).sum())
            keep_mask = np.ones(len(annot_df), dtype=bool)
            keep_mask[counted_df.index[~keep_counted]] = False
            annot_df = annot_df.loc[keep_mask].reset_index(drop=True)

    n_obs = np.zeros((n_genes, 192, 4), dtype=np.int64)
    n_ind = np.zeros(n_genes, dtype=np.int64)

    for g_idx_val, impact, t_idx_val in zip(
        annot_df["_gene_idx"], annot_df["impact"], annot_df["trinuc_ind"]
    ):
        g_idx = int(g_idx_val)
        if impact == "Indel":
            n_ind[g_idx] += 1
        elif impact in _IMPACT_TO_CLS:
            t_idx = int(t_idx_val)
            if 0 <= t_idx < 192:
                n_obs[g_idx, t_idx, _IMPACT_TO_CLS[impact]] += 1

    n_coding = int((annot_df["impact"] != "Complex").sum())
    annot_out = annot_df[_ANNOT_COLUMNS].reset_index(drop=True)
    return AnnotationResult(
        annot_muts=annot_out,
        n_obs=n_obs,
        n_ind=n_ind,
        gene_names=list(refcds.gene_names),
        excluded_samples=excluded_samples,
        n_raw_muts=n_raw,
        n_dedup_muts=n_dedup,
        n_coding_muts=n_coding,
        n_capped_muts=n_capped,
    )
