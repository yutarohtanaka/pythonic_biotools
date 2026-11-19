"""Input coercion, cell alignment, GTF TSS extraction, peak parsing, and FDR utilities for `figr`."""

from __future__ import annotations

import gzip
from pathlib import Path
import re
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

_GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')
_PEAK_STR_RE = re.compile(r"^(.+)[:_-](\d+)[:_-](\d+)$")


def extract_gtf_tss(
    gtf_path: str | Path,
    *,
    feature_type: str = "gene",
    gene_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract strand-aware 0-based TSS coordinates from a `.gtf` or `.gtf.gz` file.

    Follows GENCODE/Ensembl GTF 1-based inclusive `[start_1, end_1]` to 0-based
    half-open `[start0, end1)` conversion (`start0 = start_1 - 1`, `end = end_1`),
    with `tss = start0` on `+` strand and `tss = end - 1` on `-` strand.
    If `feature_type == "gene"` yields zero records, falls back to `"transcript"`.
    """
    p = Path(gtf_path)
    if not p.exists():
        raise FileNotFoundError(f"GTF file not found: {p}")

    opener = gzip.open if str(p).endswith(".gz") else open
    gene_rows: list[tuple[str, str, str, str, str, int, int, int]] = []
    transcript_rows: list[tuple[str, str, str, str, str, int, int, int]] = []

    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < 9:
                continue
            rec_type = cols[2]
            if rec_type != feature_type and not (feature_type == "gene" and rec_type == "transcript"):
                continue

            attrs = dict(_GTF_ATTR.findall(cols[8]))
            start0 = int(cols[3]) - 1
            end1 = int(cols[4])
            strand = cols[6]
            tss = start0 if strand != "-" else end1 - 1
            gid = attrs.get("gene_id", "").split(".")[0]
            gname = attrs.get("gene_name", "") or gid
            gtype = attrs.get("gene_type", attrs.get("gene_biotype", "unknown"))
            row = (gid, gname, gtype, cols[0], strand, tss, start0, end1)
            if rec_type == feature_type:
                gene_rows.append(row)
            else:
                transcript_rows.append(row)

    rows = gene_rows if gene_rows else (transcript_rows if feature_type == "gene" else [])
    df = pd.DataFrame(
        rows,
        columns=["gene_id", "gene_name", "gene_type", "chrom", "strand", "tss", "start", "end"],
    )
    if gene_types is not None and not df.empty:
        allowed = {str(gt) for gt in gene_types}
        df = df.loc[df["gene_type"].astype(str).isin(allowed)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid `{feature_type}` records extracted from GTF: {p}")
    return df


def coerce_tss_table(tss: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Coerce a GTF path or TSS DataFrame into canonical `['gene_name', 'chrom', 'strand', 'tss']` form."""
    if isinstance(tss, (str, Path)):
        return extract_gtf_tss(tss)
    if not isinstance(tss, pd.DataFrame):
        raise TypeError(f"`tss` must be a DataFrame or GTF path, got {type(tss).__name__}.")
    if tss.empty:
        raise ValueError("`tss` DataFrame cannot be empty.")

    df = tss.copy()
    gene_col = next((c for c in ("gene_name", "gene", "gene_id", "symbol") if c in df.columns), None)
    if gene_col is None:
        raise ValueError("`tss` DataFrame must contain one of ['gene_name', 'gene', 'gene_id', 'symbol'].")

    chrom_col = next((c for c in ("chrom", "chr", "seqnames", "seqname") if c in df.columns), None)
    if chrom_col is None:
        raise ValueError("`tss` DataFrame must contain one of ['chrom', 'chr', 'seqnames', 'seqname'].")

    strand_vals = df["strand"].astype(str).to_numpy() if "strand" in df.columns else np.full(len(df), "+", dtype=object)

    if "tss" in df.columns:
        tss_vals = pd.to_numeric(df["tss"], errors="raise").to_numpy(dtype=np.int64)
    elif "start" in df.columns and "end" in df.columns:
        starts = pd.to_numeric(df["start"], errors="raise").to_numpy(dtype=np.int64)
        ends = pd.to_numeric(df["end"], errors="raise").to_numpy(dtype=np.int64)
        tss_vals = np.where(strand_vals == "-", ends - 1, starts)
    else:
        raise ValueError("`tss` DataFrame must contain 'tss' or both 'start' and 'end'.")

    return pd.DataFrame(
        {
            "gene_name": df[gene_col].astype(str).to_numpy(),
            "chrom": df[chrom_col].astype(str).to_numpy(),
            "strand": strand_vals,
            "tss": tss_vals,
        }
    )


def parse_peak_string(peak_id: str) -> tuple[str, int, int]:
    """Parse a genomic peak coordinate string such as `'chr1:1000-2000'` or `'chr1-1000-2000'`."""
    m = _PEAK_STR_RE.match(str(peak_id).strip())
    if m is None:
        raise ValueError(
            f"Malformed peak coordinate string {peak_id!r}; expected format like 'chr1:1000-2000' or 'chr1-1000-2000'."
        )
    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3))
    if start < 0 or end < start:
        raise ValueError(f"Invalid peak coordinates in {peak_id!r}: start={start}, end={end}.")
    return chrom, start, end


def coerce_peaks(
    peaks: pd.DataFrame | Sequence[str] | None,
    peak_names: Sequence[str],
    atac_data: object | None = None,
) -> pd.DataFrame:
    """Resolve peak genomic coordinates (`peak`, `chrom`, `start`, `end`, `mid`) for `peak_names`."""
    n_peaks = len(peak_names)
    p_names = [str(p) for p in peak_names]

    source_df: pd.DataFrame | None = None
    if isinstance(peaks, pd.DataFrame):
        source_df = peaks.copy()
    elif peaks is not None:
        peak_strs = [str(p) for p in peaks]
        if len(peak_strs) != n_peaks:
            raise ValueError(f"`peaks` length ({len(peak_strs)}) must match ATAC peak count ({n_peaks}).")
        parsed = [parse_peak_string(s) for s in peak_strs]
        chroms, starts, ends = zip(*parsed)
        starts_arr = np.asarray(starts, dtype=np.int64)
        ends_arr = np.asarray(ends, dtype=np.int64)
        return pd.DataFrame(
            {
                "peak": p_names,
                "chrom": list(chroms),
                "start": starts_arr,
                "end": ends_arr,
                "mid": (starts_arr + ends_arr) // 2,
            }
        )
    elif isinstance(atac_data, ad.AnnData):
        var_df = atac_data.var
        chrom_col = next((c for c in ("chrom", "chr", "seqnames", "seqname") if c in var_df.columns), None)
        if chrom_col is not None and "start" in var_df.columns and "end" in var_df.columns:
            source_df = var_df

    if source_df is not None:
        if len(source_df) != n_peaks:
            raise ValueError(f"`peaks` DataFrame rows ({len(source_df)}) must match ATAC peak count ({n_peaks}).")
        chrom_col = next((c for c in ("chrom", "chr", "seqnames", "seqname") if c in source_df.columns), None)
        if chrom_col is None or "start" not in source_df.columns or "end" not in source_df.columns:
            raise ValueError("`peaks` DataFrame must contain ['chrom' (or 'chr'), 'start', 'end'].")
        starts_arr = pd.to_numeric(source_df["start"], errors="raise").to_numpy(dtype=np.int64)
        ends_arr = pd.to_numeric(source_df["end"], errors="raise").to_numpy(dtype=np.int64)
        if np.any(starts_arr < 0) or np.any(ends_arr < starts_arr):
            raise ValueError("`peaks` contains invalid coordinates (`start < 0` or `end < start`).")
        peak_col = source_df["peak"].astype(str).tolist() if "peak" in source_df.columns else p_names
        return pd.DataFrame(
            {
                "peak": peak_col,
                "chrom": source_df[chrom_col].astype(str).tolist(),
                "start": starts_arr,
                "end": ends_arr,
                "mid": (starts_arr + ends_arr) // 2,
            }
        )

    parsed = [parse_peak_string(s) for s in p_names]
    chroms, starts, ends = zip(*parsed)
    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.asarray(ends, dtype=np.int64)
    return pd.DataFrame(
        {
            "peak": p_names,
            "chrom": list(chroms),
            "start": starts_arr,
            "end": ends_arr,
            "mid": (starts_arr + ends_arr) // 2,
        }
    )


def extract_matrix(
    data: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix | sparse.sparray,
    *,
    layer: str | None = None,
    orientation: Literal["cells_by_features", "features_by_cells"] = "cells_by_features",
    feature_names: Sequence[str] | None = None,
    cell_names: Sequence[str] | None = None,
    feature_prefix: str = "feature",
    cell_prefix: str = "cell",
) -> tuple[np.ndarray | sparse.csr_matrix, list[str], list[str], bool, bool]:
    """Coerce `AnnData`, `DataFrame`, `ndarray`, or `scipy.sparse` matrix into `(cells x features)`.

    Returns `(matrix, cell_names, feature_names, has_explicit_cells, has_explicit_features)`.
    Sparse inputs remain `scipy.sparse.csr_matrix` (float64) without dense conversion.
    """
    if orientation not in ("cells_by_features", "features_by_cells"):
        raise ValueError(f"Invalid orientation {orientation!r}; expected 'cells_by_features' or 'features_by_cells'.")

    has_explicit_cells = cell_names is not None
    has_explicit_features = feature_names is not None

    if isinstance(data, ad.AnnData):
        raw = data.layers[layer] if layer is not None else data.X
        if sparse.issparse(raw):
            mat: np.ndarray | sparse.csr_matrix = raw.tocsr().astype(np.float64)
        else:
            mat = np.asarray(raw, dtype=np.float64)
        inferred_cells = data.obs_names.astype(str).tolist()
        inferred_features = data.var_names.astype(str).tolist()
        has_explicit_cells = True
        has_explicit_features = True
    elif isinstance(data, pd.DataFrame):
        if len(data.dtypes) > 0 and all(isinstance(dt, pd.SparseDtype) for dt in data.dtypes):
            sp = data.sparse.to_coo().tocsr().astype(np.float64)
            mat = sp.T.tocsr() if orientation == "features_by_cells" else sp
        else:
            arr = data.to_numpy(dtype=np.float64)
            mat = arr.T if orientation == "features_by_cells" else arr
        if orientation == "cells_by_features":
            inferred_cells = [str(c) for c in data.index]
            inferred_features = [str(f) for f in data.columns]
        else:
            inferred_cells = [str(c) for c in data.columns]
            inferred_features = [str(f) for f in data.index]
        has_explicit_cells = True
        has_explicit_features = True
    elif sparse.issparse(data):
        sp = data.tocsr().astype(np.float64)
        if sp.ndim != 2:
            raise ValueError(f"Input matrix must be 2D, got shape {sp.shape}.")
        mat = sp.T.tocsr() if orientation == "features_by_cells" else sp
        inferred_cells = [f"{cell_prefix}_{i}" for i in range(mat.shape[0])]
        inferred_features = [f"{feature_prefix}_{j}" for j in range(mat.shape[1])]
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Input matrix must be 2D, got shape {arr.shape}.")
        mat = arr.T if orientation == "features_by_cells" else arr
        inferred_cells = [f"{cell_prefix}_{i}" for i in range(mat.shape[0])]
        inferred_features = [f"{feature_prefix}_{j}" for j in range(mat.shape[1])]

    n_cells, n_features = mat.shape
    if n_cells == 0 or n_features == 0:
        raise ValueError(f"Input matrix cannot be empty; got shape ({n_cells}, {n_features}).")

    if cell_names is not None:
        c_names = [str(c) for c in cell_names]
        if len(c_names) != n_cells:
            raise ValueError(f"`cell_names` length ({len(c_names)}) must match cell count ({n_cells}).")
    else:
        c_names = inferred_cells

    if feature_names is not None:
        f_names = [str(f) for f in feature_names]
        if len(f_names) != n_features:
            raise ValueError(f"`feature_names` length ({len(f_names)}) must match feature count ({n_features}).")
    else:
        f_names = inferred_features

    return mat, c_names, f_names, has_explicit_cells, has_explicit_features


def _subset_rows(mat: np.ndarray | sparse.csr_matrix, row_idx: np.ndarray) -> np.ndarray | sparse.csr_matrix:
    return mat[row_idx, :]


def align_cells(
    mat_a: np.ndarray | sparse.csr_matrix,
    cells_a: list[str],
    has_names_a: bool,
    mat_b: np.ndarray | sparse.csr_matrix,
    cells_b: list[str],
    has_names_b: bool,
    *,
    min_cells: int = 3,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix, list[str]]:
    """Align two cell-indexed matrices on shared cell identifiers (preserving `cells_b` order)."""
    if has_names_a and has_names_b:
        if len(set(cells_a)) != len(cells_a):
            raise ValueError("Duplicate cell identifiers detected in first modality.")
        if len(set(cells_b)) != len(cells_b):
            raise ValueError("Duplicate cell identifiers detected in second modality.")
        idx_a = {c: i for i, c in enumerate(cells_a)}
        shared: list[str] = []
        take_a: list[int] = []
        take_b: list[int] = []
        for j, c in enumerate(cells_b):
            i = idx_a.get(c)
            if i is not None:
                shared.append(c)
                take_a.append(i)
                take_b.append(j)
        if len(shared) < min_cells:
            raise ValueError(f"At least {min_cells} shared cells are required, found {len(shared)}.")
        row_a = np.asarray(take_a, dtype=np.int64)
        row_b = np.asarray(take_b, dtype=np.int64)
        if np.array_equal(row_a, np.arange(mat_a.shape[0])) and np.array_equal(row_b, np.arange(mat_b.shape[0])):
            return mat_a, mat_b, shared
        return _subset_rows(mat_a, row_a), _subset_rows(mat_b, row_b), shared

    if mat_a.shape[0] != mat_b.shape[0]:
        raise ValueError(
            f"Unnamed matrices must have equal cell counts, got {mat_a.shape[0]} vs {mat_b.shape[0]}."
        )
    if mat_a.shape[0] < min_cells:
        raise ValueError(f"At least {min_cells} cells are required, got {mat_a.shape[0]}.")
    shared_names = cells_b if has_names_b else cells_a
    return mat_a, mat_b, shared_names


SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` parity mode ('default' or 'R_default')."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid `settings` {settings!r}; expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Compute `NaN`-aware Benjamini–Hochberg adjusted p-values (excluding non-finite entries from `m`)."""
    pvals = np.asarray(p_values, dtype=np.float64)
    q = np.full(pvals.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return q
    pv = np.clip(pvals[valid], 0.0, 1.0)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * n_valid / np.arange(1, n_valid + 1, dtype=np.float64)
    q_sorted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    q_sub = np.empty_like(pv)
    q_sub[order] = q_sorted
    q[valid] = q_sub
    return q
