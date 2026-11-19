"""
Coding reference (`RefCDS`) builders, 192 coding-strand trinucleotide contexts,
standard genetic code, and coordinate/chromosome utilities for `dndscv`.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

BASES: tuple[str, ...] = ("A", "C", "G", "T")
COMPLEMENT: dict[str, str] = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
IMPACT_CLASSES: tuple[str, ...] = ("syn", "mis", "non", "spl")

TRINUC_CONTEXTS: tuple[str, ...] = tuple(
    f"{b0}{b1}{b2}>{b0}{m}{b2}"
    for b0 in BASES
    for b1 in BASES
    for b2 in BASES
    for m in BASES
    if m != b1
)
TRINUC_TO_IDX: dict[str, int] = {ctx: idx for idx, ctx in enumerate(TRINUC_CONTEXTS)}
_TRINUC_ALT_TO_IDX: dict[tuple[str, str], int] = {
    (ctx[:3], ctx[5]): idx for idx, ctx in enumerate(TRINUC_CONTEXTS)
}

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))
_SENSE_CODONS = tuple(c for c, aa in CODON_TABLE.items() if aa != "*")


def canonical_chrom(chrom: Any) -> str:
    """Normalize chromosome names (`1`/`chr1` -> `chr1`, `MT`/`M` -> `chrM`)."""
    if chrom is None or (isinstance(chrom, float) and np.isnan(chrom)):
        raise ValueError("Chromosome name cannot be missing/NaN.")
    raw = str(chrom).strip()
    if not raw or raw.lower() == "nan":
        raise ValueError("Chromosome name cannot be empty.")
    body = raw[3:] if raw.lower().startswith("chr") else raw
    upper = body.upper()
    if upper in {"M", "MT"}:
        return "chrM"
    if upper in {"X", "Y"}:
        return f"chr{upper}"
    if body.isdigit():
        num = int(body)
        if 1 <= num <= 99:
            return f"chr{num}"
    return raw


def reverse_complement(seq: str) -> str:
    """Return the 5'->3' reverse complement of a DNA sequence."""
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


def _normalize_strand(val: Any) -> Literal[1, -1]:
    if isinstance(val, (int, np.integer)):
        if int(val) == 1:
            return 1
        if int(val) == -1:
            return -1
    if isinstance(val, (float, np.floating)) and np.isfinite(val):
        if int(val) == 1:
            return 1
        if int(val) == -1:
            return -1
    s = str(val).strip()
    if s in {"+", "+1", "1"}:
        return 1
    if s in {"-", "-1"}:
        return -1
    raise ValueError(f"Invalid strand value {val!r}; expected '+', '-', 1, or -1.")


def classify_codon_impact(ref_codon: str, mut_codon: str) -> int | None:
    """
    Classify coding SNV impact class index:
    0 = syn, 1 = mis, 2 = non (stop-gain or stop-loss), None = ambiguous.
    """
    aa_ref = CODON_TABLE.get(ref_codon)
    aa_mut = CODON_TABLE.get(mut_codon)
    if aa_ref is None or aa_mut is None:
        return None
    if aa_ref == aa_mut:
        return 0
    if aa_mut == "*" or aa_ref == "*":
        return 2
    return 1


class _FastaAccessor:
    """FASTA reader supporting in-memory dicts, plain/gzipped FASTA, and optional pysam."""

    def __init__(self, source: str | Path | Mapping[str, str]) -> None:
        self._seqs: dict[str, str] = {}
        self._pysam_fa: Any = None
        self._key_map: dict[str, str] = {}
        self._lengths: dict[str, int] = {}

        if isinstance(source, Mapping):
            for k, v in source.items():
                key = str(k)
                seq = str(v).upper()
                self._seqs[key] = seq
                self._lengths[key] = len(seq)
                self._register_key(key)
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {path}")

        # Prefer pysam only for indexed uncompressed FASTA when available
        fai_path = Path(f"{path}.fai")
        if path.suffix != ".gz" and fai_path.exists():
            try:
                import pysam  # type: ignore

                fa = pysam.FastaFile(str(path))
                self._pysam_fa = fa
                for ref_name, length in zip(fa.references, fa.lengths):
                    self._lengths[ref_name] = int(length)
                    self._register_key(ref_name)
                return
            except Exception:
                self._pysam_fa = None

        opener = gzip.open if path.suffix == ".gz" else open
        curr_name: str | None = None
        chunks: list[str] = []
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if curr_name is not None:
                        seq = "".join(chunks).upper()
                        self._seqs[curr_name] = seq
                        self._lengths[curr_name] = len(seq)
                        self._register_key(curr_name)
                    curr_name = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
            if curr_name is not None:
                seq = "".join(chunks).upper()
                self._seqs[curr_name] = seq
                self._lengths[curr_name] = len(seq)
                self._register_key(curr_name)

        if not self._lengths:
            raise ValueError(f"No sequences found in FASTA: {path}")

    def _register_key(self, raw_key: str) -> None:
        self._key_map[raw_key] = raw_key
        norm = canonical_chrom(raw_key)
        self._key_map.setdefault(norm, raw_key)

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self._key_map:
            return self._key_map[chrom]
        norm = canonical_chrom(chrom)
        if norm in self._key_map:
            return self._key_map[norm]
        raise ValueError(f"Chromosome {chrom!r} not found in FASTA.")

    def chrom_length(self, chrom: str) -> int:
        key = self.resolve_chrom(chrom)
        return self._lengths[key]

    def fetch_one_based(self, chrom: str, start_1: int, end_1: int) -> str:
        """Fetch 1-based closed [start_1, end_1] padded with 'N' outside contig bounds."""
        key = self.resolve_chrom(chrom)
        length = self._lengths[key]
        if end_1 < 1 or start_1 > length:
            return "N" * max(0, end_1 - start_1 + 1)

        left_pad = max(0, 1 - start_1)
        right_pad = max(0, end_1 - length)
        s0 = max(0, start_1 - 1)
        e0 = min(length, end_1)

        if self._pysam_fa is not None:
            core = self._pysam_fa.fetch(key, s0, e0).upper()
        else:
            core = self._seqs[key][s0:e0]
        return ("N" * left_pad) + core + ("N" * right_pad)


@dataclass(frozen=True)
class GeneRef:
    """Per-gene coding sequence, genomic intervals, and 192x4 opportunity matrix."""

    gene_name: str
    gene_id: str
    chrom: str
    strand: Literal[1, -1]
    intervals_cds: np.ndarray  # (n_exons, 2), 1-based closed [start, end], ordered 5'->3'
    intervals_splice: np.ndarray  # (n_splice,), 1-based sorted genomic positions
    cds_length: int
    seq_cds: str
    seq_cds1up: str
    seq_cds1down: str
    L: np.ndarray  # (192, 4) int64 for (syn, mis, non, spl)
    splice_trinucs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefCDS:
    """Collection of coding gene references and pooled opportunity matrices."""

    genes: tuple[GeneRef, ...]
    gene_names: list[str]
    L_all: np.ndarray  # (G, 192, 4) int64
    cds_lengths: np.ndarray  # (G,) int64
    covariates: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.genes)

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-gene reference metadata and total substitution opportunities."""
        if len(self.genes) == 0:
            return pd.DataFrame(
                columns=[
                    "gene_name",
                    "gene_id",
                    "chrom",
                    "strand",
                    "n_exons",
                    "n_splice",
                    "cds_length",
                    "L_syn",
                    "L_mis",
                    "L_non",
                    "L_spl",
                ]
            )
        rows = []
        for g in self.genes:
            totals = np.sum(g.L, axis=0)
            rows.append(
                {
                    "gene_name": g.gene_name,
                    "gene_id": g.gene_id,
                    "chrom": g.chrom,
                    "strand": int(g.strand),
                    "n_exons": int(g.intervals_cds.shape[0]),
                    "n_splice": int(g.intervals_splice.size),
                    "cds_length": int(g.cds_length),
                    "L_syn": int(totals[0]),
                    "L_mis": int(totals[1]),
                    "L_non": int(totals[2]),
                    "L_spl": int(totals[3]),
                }
            )
        return pd.DataFrame(rows)

    def subset(self, genes: Sequence[str]) -> RefCDS:
        """Return a new `RefCDS` restricted to the specified gene names."""
        name_to_idx = {name: i for i, name in enumerate(self.gene_names)}
        selected_indices: list[int] = []
        seen: set[str] = set()
        missing: list[str] = []
        for g in genes:
            gs = str(g)
            if gs not in name_to_idx:
                missing.append(gs)
                continue
            if gs not in seen:
                seen.add(gs)
                selected_indices.append(name_to_idx[gs])
        if missing:
            raise ValueError(f"Unknown genes requested in subset: {missing[:10]}")
        if not selected_indices:
            raise ValueError("Subset gene list produced an empty RefCDS.")

        sub_genes = tuple(self.genes[i] for i in selected_indices)
        sub_names = [self.gene_names[i] for i in selected_indices]
        sub_L = self.L_all[selected_indices].copy()
        sub_len = self.cds_lengths[selected_indices].copy()
        sub_cov = None
        if self.covariates is not None:
            sub_cov = self.covariates.iloc[selected_indices].copy()
        return RefCDS(
            genes=sub_genes,
            gene_names=sub_names,
            L_all=sub_L,
            cds_lengths=sub_len,
            covariates=sub_cov,
        )


def _prepare_covariates_df(
    covariates: pd.DataFrame | np.ndarray | None,
    gene_names: list[str],
) -> pd.DataFrame | None:
    if covariates is None:
        return None
    n_genes = len(gene_names)
    if isinstance(covariates, pd.DataFrame):
        if covariates.shape[0] < n_genes:
            raise ValueError(
                f"covariates has {covariates.shape[0]} rows, expected at least {n_genes}."
            )
        # Align by index if gene_names are all present in DataFrame index
        idx_str = covariates.index.astype(str)
        if set(gene_names).issubset(set(idx_str)):
            aligned = covariates.copy()
            aligned.index = idx_str
            aligned = aligned.loc[gene_names].copy()
        elif covariates.shape[0] == n_genes:
            aligned = covariates.copy()
            aligned.index = pd.Index(gene_names, name="gene_name")
        else:
            raise ValueError(
                "covariates DataFrame row count/index does not match RefCDS gene_names."
            )
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="raise").astype(np.float64)
        if not np.all(np.isfinite(aligned.to_numpy(dtype=np.float64))):
            raise ValueError("covariates contain NaN or infinite values.")
        return aligned

    arr = np.asarray(covariates, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"covariates must be 1D or 2D, got {arr.ndim}D.")
    if arr.shape[0] != n_genes:
        raise ValueError(
            f"covariates has {arr.shape[0]} rows, expected {n_genes} matching genes."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("covariates contain NaN or infinite values.")
    cols = [f"X{j}" for j in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.Index(gene_names, name="gene_name"), columns=cols)


def _compute_opportunity_matrix(
    seq_cds: str,
    seq_cds1up: str,
    seq_cds1down: str,
    splice_trinucs: Sequence[str],
) -> np.ndarray:
    L = np.zeros((192, 4), dtype=np.int64)
    cds_len = len(seq_cds)
    for pos in range(cds_len):
        b0 = seq_cds1up[pos]
        b1 = seq_cds[pos]
        b2 = seq_cds1down[pos]
        if b0 not in BASES or b1 not in BASES or b2 not in BASES:
            continue
        codon_start = (pos // 3) * 3
        ref_codon = seq_cds[codon_start : codon_start + 3]
        if len(ref_codon) < 3 or any(b not in BASES for b in ref_codon):
            continue
        frame = pos - codon_start
        tri = b0 + b1 + b2
        for m in BASES:
            if m == b1:
                continue
            mut_codon = ref_codon[:frame] + m + ref_codon[frame + 1 :]
            cls_idx = classify_codon_impact(ref_codon, mut_codon)
            if cls_idx is None:
                continue
            t_idx = _TRINUC_ALT_TO_IDX[(tri, m)]
            L[t_idx, cls_idx] += 1

    for tri in splice_trinucs:
        if len(tri) != 3 or any(b not in BASES for b in tri):
            continue
        b1 = tri[1]
        for m in BASES:
            if m == b1:
                continue
            t_idx = _TRINUC_ALT_TO_IDX[(tri, m)]
            L[t_idx, 3] += 1
    return L


def _validate_cds_sequence(seq_cds: str, gene_name: str, require_valid_cds: bool) -> None:
    cds_len = len(seq_cds)
    if cds_len == 0 or cds_len % 3 != 0:
        raise ValueError(
            f"Gene {gene_name!r} has CDS length {cds_len}, which is not a positive multiple of 3."
        )
    if not require_valid_cds:
        return
    if cds_len < 6:
        raise ValueError(f"Gene {gene_name!r} CDS length {cds_len} < 6 is too short.")
    if seq_cds[:3] != "ATG":
        raise ValueError(
            f"Gene {gene_name!r} does not start with canonical ATG start codon (got {seq_cds[:3]!r})."
        )
    if seq_cds[-3:] not in _STOP_CODONS:
        raise ValueError(
            f"Gene {gene_name!r} does not end with a stop codon TAA/TAG/TGA (got {seq_cds[-3:]!r})."
        )
    for k in range(0, cds_len - 3, 3):
        codon = seq_cds[k : k + 3]
        if codon in _STOP_CODONS:
            raise ValueError(
                f"Gene {gene_name!r} contains an internal stop codon {codon!r} at codon {k // 3 + 1}."
            )


def _resolve_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    col_map = {str(c): str(c) for c in df.columns}
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Missing required column (accepted aliases: {list(candidates)}). Found columns: {list(df.columns)}"
        )
    return None


def build_refcds(
    cds_table: pd.DataFrame,
    fasta: str | Path | Mapping[str, str],
    *,
    covariates: pd.DataFrame | np.ndarray | None = None,
    require_valid_cds: bool = True,
    coord_system: Literal["one_based", "zero_based"] = "one_based",
) -> RefCDS:
    """
    Build a `RefCDS` coding reference from a CDS exon table and reference FASTA.

    Parameters
    ----------
    cds_table : pd.DataFrame
        Exon-level CDS coordinates with columns for gene name (`gene_name`/`gene`),
        chromosome (`chrom`/`chr`), `start`, `end`, `strand` (`+`/`-`/`1`/`-1`),
        and optional `gene_id`/`transcript_id`.
    fasta : str | Path | Mapping[str, str]
        Reference genome FASTA path or mapping `{chrom: sequence}`.
    covariates : pd.DataFrame | np.ndarray | None
        Optional gene-level epigenomic/expression covariates.
    require_valid_cds : bool
        If True, require ATG start, terminal stop codon, and no internal stop codons.
    coord_system : {"one_based", "zero_based"}
        Input exon coordinate convention (`one_based` closed `[start, end]` or
        `zero_based` half-open `[start, end)`).
    """
    if not isinstance(cds_table, pd.DataFrame) or cds_table.empty:
        raise ValueError("cds_table must be a non-empty pandas DataFrame.")
    if coord_system not in {"one_based", "zero_based"}:
        raise ValueError(f"Invalid coord_system {coord_system!r}; expected 'one_based' or 'zero_based'.")

    gene_col = _resolve_col(cds_table, ["gene_name", "gene", "Hugo_Symbol", "symbol"])
    id_col = _resolve_col(cds_table, ["gene_id", "transcript_id"], required=False)
    chrom_col = _resolve_col(cds_table, ["chrom", "chr", "Chromosome"])
    start_col = _resolve_col(cds_table, ["start", "cds_start"])
    end_col = _resolve_col(cds_table, ["end", "cds_end"])
    strand_col = _resolve_col(cds_table, ["strand"])

    assert gene_col and chrom_col and start_col and end_col and strand_col

    fa = _FastaAccessor(fasta)

    df = cds_table.copy()
    if df[[gene_col, chrom_col, start_col, end_col, strand_col]].isna().any().any():
        raise ValueError("cds_table contains NaN values in required columns.")

    starts = df[start_col].to_numpy(dtype=np.int64)
    ends = df[end_col].to_numpy(dtype=np.int64)
    if coord_system == "zero_based":
        if np.any(starts < 0) or np.any(starts >= ends):
            raise ValueError("zero_based exons must satisfy 0 <= start < end.")
        starts = starts + 1
    else:
        if np.any(starts < 1) or np.any(starts > ends):
            raise ValueError("one_based exons must satisfy 1 <= start <= end.")

    df["_start_1"] = starts
    df["_end_1"] = ends
    df["_chrom_norm"] = [canonical_chrom(c) for c in df[chrom_col]]
    df["_strand_int"] = [_normalize_strand(s) for s in df[strand_col]]
    df["_gene_str"] = df[gene_col].astype(str).str.strip()
    df["_gene_id_str"] = df[id_col].astype(str).str.strip() if id_col else df["_gene_str"]

    gene_refs: list[GeneRef] = []
    # Preserve deterministic sorted or first-seen order? First-seen in table is standard;
    # wait: preserving grouped order or sorted? Grouping with sort=False preserves table order.
    for gene_name, grp in df.groupby("_gene_str", sort=False):
        chroms = grp["_chrom_norm"].unique()
        strands = grp["_strand_int"].unique()
        if len(chroms) != 1 or len(strands) != 1:
            raise ValueError(
                f"Gene {gene_name!r} has inconsistent chromosomes or strands across exons."
            )
        chrom = str(chroms[0])
        strand: Literal[1, -1] = 1 if int(strands[0]) == 1 else -1
        gene_id = str(grp["_gene_id_str"].iloc[0])

        exon_arr = grp[["_start_1", "_end_1"]].to_numpy(dtype=np.int64)
        # Check exon non-overlap in genomic order
        asc_order = np.lexsort((exon_arr[:, 1], exon_arr[:, 0]))
        exon_asc = exon_arr[asc_order]
        if np.any(exon_asc[1:, 0] <= exon_asc[:-1, 1]):
            raise ValueError(f"Gene {gene_name!r} has overlapping CDS exons.")

        # Order exons 5' -> 3' in transcript orientation
        if strand == 1:
            exons_5to3 = exon_asc
        else:
            exons_5to3 = exon_asc[::-1].copy()

        chrom_len = fa.chrom_length(chrom)
        cds_chunks: list[str] = []
        up_chunks: list[str] = []
        down_chunks: list[str] = []
        coding_pos_set: set[int] = set()

        for s1, e1 in exons_5to3:
            s_i = int(s1)
            e_i = int(e1)
            if e_i > chrom_len:
                raise ValueError(
                    f"Exon [{s_i}, {e_i}] for {gene_name!r} exceeds {chrom} length ({chrom_len})."
                )
            # Fetch [s_i - 1, e_i + 1] in genomic + strand (length = exon_len + 2)
            ext_plus = fa.fetch_one_based(chrom, s_i - 1, e_i + 1)
            exon_len = e_i - s_i + 1
            for p in range(s_i, e_i + 1):
                coding_pos_set.add(p)

            if strand == 1:
                cds_chunks.append(ext_plus[1 : 1 + exon_len])
                up_chunks.append(ext_plus[0:exon_len])
                down_chunks.append(ext_plus[2 : 2 + exon_len])
            else:
                # Reverse complement ext_plus:
                # ext_plus[0] is genomic s_i - 1 (3' coding neighbor of pos s_i)
                # ext_plus[-1] is genomic e_i + 1 (5' coding neighbor of pos e_i)
                ext_rc = reverse_complement(ext_plus)
                # In ext_rc: index 0 is comp(e_i + 1), index 1..exon_len is rc(exon), index exon_len+1 is comp(s_i - 1)
                cds_chunks.append(ext_rc[1 : 1 + exon_len])
                up_chunks.append(ext_rc[0:exon_len])
                down_chunks.append(ext_rc[2 : 2 + exon_len])

        seq_cds = "".join(cds_chunks)
        seq_cds1up = "".join(up_chunks)
        seq_cds1down = "".join(down_chunks)
        _validate_cds_sequence(seq_cds, gene_name, require_valid_cds)

        # Essential splice sites: internal introns only (-2, -1 acceptor; +1, +2 donor)
        splice_positions: list[int] = []
        n_exons = exons_5to3.shape[0]
        if n_exons >= 2:
            for k in range(n_exons - 1):
                s_curr, e_curr = int(exons_5to3[k, 0]), int(exons_5to3[k, 1])
                s_next, e_next = int(exons_5to3[k + 1, 0]), int(exons_5to3[k + 1, 1])
                if strand == 1:
                    # Donor +1, +2 after e_curr; acceptor -2, -1 before s_next
                    cand = (e_curr + 1, e_curr + 2, s_next - 2, s_next - 1)
                else:
                    # Donor +1, +2 downstream of s_curr; acceptor -2, -1 upstream of e_next
                    cand = (s_curr - 1, s_curr - 2, e_next + 2, e_next + 1)
                for p in cand:
                    if 1 <= p <= chrom_len and p not in coding_pos_set:
                        splice_positions.append(p)

        unique_splice = sorted(set(splice_positions))
        splice_arr = np.asarray(unique_splice, dtype=np.int64)
        splice_trinucs_list: list[str] = []
        for p in unique_splice:
            tri_plus = fa.fetch_one_based(chrom, p - 1, p + 1)
            tri_cod = tri_plus if strand == 1 else reverse_complement(tri_plus)
            splice_trinucs_list.append(tri_cod)
        splice_trinucs = tuple(splice_trinucs_list)

        L_mat = _compute_opportunity_matrix(seq_cds, seq_cds1up, seq_cds1down, splice_trinucs)
        gene_refs.append(
            GeneRef(
                gene_name=str(gene_name),
                gene_id=gene_id,
                chrom=chrom,
                strand=strand,
                intervals_cds=exons_5to3,
                intervals_splice=splice_arr,
                cds_length=len(seq_cds),
                seq_cds=seq_cds,
                seq_cds1up=seq_cds1up,
                seq_cds1down=seq_cds1down,
                L=L_mat,
                splice_trinucs=splice_trinucs,
            )
        )

    if not gene_refs:
        raise ValueError("No valid genes constructed from cds_table.")

    gene_names = [g.gene_name for g in gene_refs]
    L_all = np.stack([g.L for g in gene_refs], axis=0)
    cds_lengths = np.asarray([g.cds_length for g in gene_refs], dtype=np.int64)
    cov_df = _prepare_covariates_df(covariates, gene_names)
    return RefCDS(
        genes=tuple(gene_refs),
        gene_names=gene_names,
        L_all=L_all,
        cds_lengths=cds_lengths,
        covariates=cov_df,
    )


_GTF_ATTR_RE = re.compile(r'([A-Za-z0-9_.-]+)\s+"([^"]+)"')
_GFF3_ATTR_RE = re.compile(r'([A-Za-z0-9_.-]+)=([^;]+)')


def _parse_gtf_attributes(attr_str: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, val in _GTF_ATTR_RE.findall(attr_str):
        attrs[key] = val
    if not attrs:
        for key, val in _GFF3_ATTR_RE.findall(attr_str):
            attrs[key] = val.strip('"')
    return attrs


def build_refcds_from_gtf(
    gtf_path: str | Path,
    fasta: str | Path | Mapping[str, str],
    *,
    covariates: pd.DataFrame | np.ndarray | None = None,
    require_valid_cds: bool = True,
) -> RefCDS:
    """
    Build a `RefCDS` coding reference from a GTF/GTF.GZ annotation file and FASTA.

    Selects the longest CDS transcript per `gene_name` deterministically and
    respects GTF 5' reading-frame phase (`0`, `1`, `2`).
    """
    path = Path(gtf_path)
    if not path.exists():
        raise FileNotFoundError(f"GTF file not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2].upper() != "CDS":
                continue
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = _normalize_strand(parts[6])
            phase_raw = parts[7].strip()
            phase = int(phase_raw) if phase_raw in {"0", "1", "2"} else 0
            attrs = _parse_gtf_attributes(parts[8])
            gene_name = (
                attrs.get("gene_name")
                or attrs.get("gene")
                or attrs.get("gene_id")
                or attrs.get("Name")
            )
            if not gene_name:
                continue
            gene_id = attrs.get("gene_id") or gene_name
            tx_id = attrs.get("transcript_id") or attrs.get("Parent") or gene_id
            records.append(
                {
                    "gene_name": gene_name,
                    "gene_id": gene_id,
                    "transcript_id": tx_id,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "phase": phase,
                }
            )

    if not records:
        raise ValueError(f"No CDS features found in GTF: {path}")

    df = pd.DataFrame(records)
    # Pick canonical longest CDS transcript per gene_name (tie-break by transcript_id)
    df["_len"] = df["end"] - df["start"] + 1
    tx_lens = (
        df.groupby(["gene_name", "transcript_id"], as_index=False)["_len"]
        .sum()
        .sort_values(["gene_name", "_len", "transcript_id"], ascending=[True, False, True], kind="mergesort")
    )
    best_tx = tx_lens.drop_duplicates(subset=["gene_name"], keep="first")[
        ["gene_name", "transcript_id"]
    ]
    df_best = df.merge(best_tx, on=["gene_name", "transcript_id"], how="inner")

    # Apply GTF phase adjustment on 5'-most exon of each transcript
    adjusted_rows: list[dict[str, Any]] = []
    for _, grp in df_best.groupby("gene_name", sort=False):
        strand = int(grp["strand"].iloc[0])
        grp_sorted = grp.sort_values(
            "start" if strand == 1 else "end",
            ascending=(strand == 1),
            kind="mergesort",
        ).reset_index(drop=True)
        first_phase = int(grp_sorted.loc[0, "phase"])
        if first_phase in {1, 2}:
            if strand == 1:
                grp_sorted.loc[0, "start"] = int(grp_sorted.loc[0, "start"]) + first_phase
            else:
                grp_sorted.loc[0, "end"] = int(grp_sorted.loc[0, "end"]) - first_phase
        # Drop any degenerate exon after phase trim
        grp_sorted = grp_sorted.loc[grp_sorted["start"] <= grp_sorted["end"]].copy()
        if not require_valid_cds and not grp_sorted.empty:
            total_bp = int((grp_sorted["end"] - grp_sorted["start"] + 1).sum())
            rem = total_bp % 3
            if rem != 0:
                last_idx = grp_sorted.index[-1]
                if strand == 1:
                    grp_sorted.loc[last_idx, "end"] = int(grp_sorted.loc[last_idx, "end"]) - rem
                else:
                    grp_sorted.loc[last_idx, "start"] = int(grp_sorted.loc[last_idx, "start"]) + rem
                grp_sorted = grp_sorted.loc[grp_sorted["start"] <= grp_sorted["end"]]
        adjusted_rows.extend(grp_sorted.to_dict("records"))

    cds_df = pd.DataFrame(adjusted_rows)
    return build_refcds(
        cds_df,
        fasta=fasta,
        covariates=covariates,
        require_valid_cds=require_valid_cds,
        coord_system="one_based",
    )


def build_synthetic_refcds(
    sequences: Mapping[str, str | Mapping[str, Any]] | pd.DataFrame,
    *,
    covariates: pd.DataFrame | np.ndarray | None = None,
    require_valid_cds: bool = False,
) -> RefCDS:
    """
    Build a `RefCDS` directly from in-memory gene coding sequences without external files.

    `sequences` may map `gene_name -> seq_cds` (or `gene_name -> dict` with `seq`/`exons`,
    `strand`, `chrom`, `flank_5p`, `flank_3p`), or be a DataFrame with `gene_name` and
    `seq_cds` columns.
    """
    if isinstance(sequences, pd.DataFrame):
        if "gene_name" not in sequences.columns or "seq_cds" not in sequences.columns:
            raise ValueError("DataFrame input to build_synthetic_refcds requires 'gene_name' and 'seq_cds'.")
        seq_map: dict[str, dict[str, Any]] = {}
        for _, row in sequences.iterrows():
            gname = str(row["gene_name"])
            seq_map[gname] = {
                "seq": str(row["seq_cds"]),
                "strand": row["strand"] if "strand" in sequences.columns else 1,
                "chrom": row["chrom"] if "chrom" in sequences.columns else "chr1",
            }
    elif isinstance(sequences, Mapping):
        seq_map = {}
        for gname, val in sequences.items():
            if isinstance(val, str):
                seq_map[str(gname)] = {"seq": val, "strand": 1, "chrom": "chr1"}
            elif isinstance(val, Mapping):
                seq_map[str(gname)] = dict(val)
            else:
                raise ValueError(f"Unsupported entry type for gene {gname!r}: {type(val)}")
    else:
        raise ValueError("sequences must be a Mapping or pandas DataFrame.")

    if not seq_map:
        raise ValueError("sequences cannot be empty.")

    # Place each gene on its own synthetic locus with neutral flanking/intron DNA
    fasta_chunks: dict[str, list[str]] = {}
    chrom_cursor: dict[str, int] = {}
    cds_rows: list[dict[str, Any]] = []

    for idx, (gname, spec) in enumerate(seq_map.items()):
        chrom = canonical_chrom(spec.get("chrom", "chr1"))
        strand = _normalize_strand(spec.get("strand", 1))
        seq = str(spec["seq"]).upper()
        if len(seq) == 0 or len(seq) % 3 != 0:
            raise ValueError(f"Gene {gname!r} CDS length {len(seq)} must be a positive multiple of 3.")
        exons_spec = spec.get("exons")
        if exons_spec is not None:
            exon_lengths = [int(x) for x in exons_spec]
            if sum(exon_lengths) != len(seq) or any(x <= 0 for x in exon_lengths):
                raise ValueError(f"Gene {gname!r} exon lengths {exon_lengths} do not sum to CDS length {len(seq)}.")
        else:
            exon_lengths = [len(seq)]

        flank5 = str(spec.get("flank_5p", "A")).upper()
        flank3 = str(spec.get("flank_3p", "T")).upper()
        if len(flank5) != 1 or flank5 not in BASES:
            flank5 = "A"
        if len(flank3) != 1 or flank3 not in BASES:
            flank3 = "T"

        chunks = fasta_chunks.setdefault(chrom, [])
        cursor = chrom_cursor.get(chrom, 0)

        # Split CDS into exon slices in 5'->3' order
        exon_seqs: list[str] = []
        off = 0
        for elen in exon_lengths:
            exon_seqs.append(seq[off : off + elen])
            off += elen

        # Build genomic DNA segment on + strand with 20bp padding & 20bp GT..AG introns
        intron_seq = "GT" + ("ACGT" * 4) + "AG"  # 20 bp
        if strand == 1:
            pad_left = "CGTACGTACG" * 2
            pad_left = pad_left[:-1] + flank5
            chunks.append(pad_left)
            cursor += len(pad_left)
            for k, eseq in enumerate(exon_seqs):
                s1 = cursor + 1
                e1 = cursor + len(eseq)
                chunks.append(eseq)
                cursor += len(eseq)
                cds_rows.append(
                    {
                        "gene_name": gname,
                        "gene_id": str(spec.get("gene_id", gname)),
                        "chrom": chrom,
                        "start": s1,
                        "end": e1,
                        "strand": 1,
                    }
                )
                if k < len(exon_seqs) - 1:
                    chunks.append(intron_seq)
                    cursor += len(intron_seq)
            pad_right = flank3 + "GTACGTACGT" * 2
            chunks.append(pad_right)
            cursor += len(pad_right)
        else:
            # On - strand, genomic + order is reverse of 5'->3' exon order
            pad_left = "CGTACGTACG" * 2
            pad_left = pad_left[:-1] + COMPLEMENT[flank3]
            chunks.append(pad_left)
            cursor += len(pad_left)
            intron_rc = reverse_complement(intron_seq)
            rev_exons = list(reversed(exon_seqs))
            for k, eseq in enumerate(rev_exons):
                eseq_plus = reverse_complement(eseq)
                s1 = cursor + 1
                e1 = cursor + len(eseq_plus)
                chunks.append(eseq_plus)
                cursor += len(eseq_plus)
                cds_rows.append(
                    {
                        "gene_name": gname,
                        "gene_id": str(spec.get("gene_id", gname)),
                        "chrom": chrom,
                        "start": s1,
                        "end": e1,
                        "strand": -1,
                    }
                )
                if k < len(rev_exons) - 1:
                    chunks.append(intron_rc)
                    cursor += len(intron_rc)
            pad_right = COMPLEMENT[flank5] + "GTACGTACGT" * 2
            chunks.append(pad_right)
            cursor += len(pad_right)

        chrom_cursor[chrom] = cursor

    fasta_dict = {chrom: "".join(parts) for chrom, parts in fasta_chunks.items()}
    return build_refcds(
        pd.DataFrame(cds_rows),
        fasta=fasta_dict,
        covariates=covariates,
        require_valid_cds=require_valid_cds,
        coord_system="one_based",
    )


def simulate_refcds(
    n_genes: int = 200,
    *,
    cds_length: int = 900,
    n_exons: int = 3,
    n_covariates: int = 2,
    seed: int = 0,
) -> RefCDS:
    """
    Procedurally simulate a multi-exon, mixed-strand `RefCDS` with synthetic covariates.
    """
    if n_genes < 1:
        raise ValueError("n_genes must be >= 1.")
    if cds_length < 6 or cds_length % 3 != 0:
        raise ValueError("cds_length must be >= 6 and divisible by 3.")
    if n_exons < 1:
        raise ValueError("n_exons must be >= 1.")
    if cds_length < n_exons * 3:
        raise ValueError("cds_length must be at least 3 * n_exons.")
    if n_covariates < 0:
        raise ValueError("n_covariates must be >= 0.")

    rng = np.random.default_rng(seed)
    n_codons = cds_length // 3
    sense_arr = np.asarray(_SENSE_CODONS)
    stop_arr = np.asarray(tuple(sorted(_STOP_CODONS)))

    # Partition cds_length across n_exons cleanly
    base_len = cds_length // n_exons
    rem_len = cds_length % n_exons
    exon_lens = [base_len + (1 if k < rem_len else 0) for k in range(n_exons)]

    seq_spec: dict[str, dict[str, Any]] = {}
    n_chroms = min(10, max(1, n_genes // 20))
    for i in range(n_genes):
        internal = "".join(rng.choice(sense_arr, size=n_codons - 2))
        stop = str(rng.choice(stop_arr))
        seq = "ATG" + internal + stop
        strand = 1 if (i % 2 == 0) else -1
        chrom = f"chr{(i % n_chroms) + 1}"
        seq_spec[f"GENE_{i:04d}"] = {
            "seq": seq,
            "exons": exon_lens,
            "strand": strand,
            "chrom": chrom,
            "flank_5p": str(rng.choice(BASES)),
            "flank_3p": str(rng.choice(BASES)),
        }

    cov_df: pd.DataFrame | None = None
    if n_covariates > 0:
        cov_mat = rng.normal(0.0, 1.0, size=(n_genes, n_covariates))
        cov_df = pd.DataFrame(
            cov_mat,
            index=pd.Index(list(seq_spec.keys()), name="gene_name"),
            columns=[f"cov_{j}" for j in range(n_covariates)],
        )

    return build_synthetic_refcds(
        seq_spec,
        covariates=cov_df,
        require_valid_cds=True,
    )
