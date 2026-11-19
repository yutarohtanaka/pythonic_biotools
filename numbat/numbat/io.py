"""Input validation, AnnData/sparse ingestion, and genomic locus ordering for `numbat`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def canonical_chrom_key(chrom: str) -> tuple[int, int, str]:
    """Return a deterministic sort key for standard and non-standard chromosomes."""
    raw = str(chrom).strip()
    s = raw[3:] if raw.lower().startswith("chr") else raw
    s_upper = s.upper()
    if s_upper.isdigit():
        return (0, int(s_upper), "")
    if s_upper == "X":
        return (1, 23, "")
    if s_upper == "Y":
        return (1, 24, "")
    if s_upper in ("M", "MT"):
        return (1, 25, "")
    return (2, 0, raw)


@dataclass(frozen=True)
class PreparedExpression:
    """Validated cell-by-gene expression matrix and genomic metadata."""

    matrix: np.ndarray | sparse.csr_matrix
    is_sparse: bool
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    gene_chrom: np.ndarray
    gene_pos: np.ndarray
    gene_start: np.ndarray
    gene_end: np.ndarray
    ref_indices: np.ndarray | None


@dataclass(frozen=True)
class PreparedAlleles:
    """Validated cell-by-SNP allele count matrices and genomic coordinates."""

    ad_mat: sparse.csr_matrix
    dp_mat: sparse.csr_matrix
    snp_ids: tuple[str, ...]
    snp_chrom: np.ndarray
    snp_pos: np.ndarray
    snp_cm: np.ndarray


@dataclass(frozen=True)
class OrderedLoci:
    """Merged, canonically sorted gene and SNP loci across chromosomes."""

    n_loci: int
    locus_id: tuple[str, ...]
    chrom: np.ndarray
    pos: np.ndarray
    start: np.ndarray
    end: np.ndarray
    has_expr: np.ndarray
    has_allele: np.ndarray
    gene_index: np.ndarray
    snp_index: np.ndarray
    dist_cm: np.ndarray
    chrom_names: tuple[str, ...]
    chrom_offsets: np.ndarray


def _coerce_matrix(
    X: Any,
    require_non_negative: bool = True,
) -> tuple[np.ndarray | sparse.csr_matrix, bool]:
    """Validate and coerce an expression matrix without unnecessary densification."""
    if sparse.issparse(X):
        mat = X.tocsr().astype(np.float64, copy=False)
        if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
            raise ValueError("Expression matrix must be a non-empty 2D matrix.")
        if mat.data.size > 0:
            if not np.all(np.isfinite(mat.data)):
                raise ValueError("Expression matrix contains non-finite values (NaN or Inf).")
            if require_non_negative and np.any(mat.data < 0.0):
                raise ValueError("Expression counts must be non-negative.")
        return mat, True

    if isinstance(X, pd.DataFrame):
        if all(isinstance(dt, pd.SparseDtype) for dt in X.dtypes):
            mat = X.sparse.to_coo().tocsr().astype(np.float64)
            return _coerce_matrix(mat, require_non_negative=require_non_negative)
        arr = X.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Expression matrix must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Expression matrix contains non-finite values (NaN or Inf).")
    if require_non_negative and np.any(arr < 0.0):
        raise ValueError("Expression counts must be non-negative.")
    return np.ascontiguousarray(arr, dtype=np.float64), False


def _resolve_gene_coords(
    var_df: pd.DataFrame,
    n_genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate gene chromosome and genomic coordinate arrays."""
    if len(var_df) != n_genes:
        raise ValueError(
            f"Gene annotation length ({len(var_df)}) does not match number of genes ({n_genes})."
        )
    col_map = {c.lower(): c for c in var_df.columns}
    chrom_col = col_map.get("chrom") or col_map.get("chromosome") or col_map.get("chr")
    if chrom_col is None:
        raise ValueError(
            "Gene coordinate table must contain a 'chrom' (or 'chromosome') column."
        )
    chrom_raw = var_df[chrom_col]
    if chrom_raw.isna().any():
        raise ValueError("Gene 'chrom' column contains missing/NaN values.")
    chrom = chrom_raw.astype(str).to_numpy()

    pos_col = col_map.get("pos") or col_map.get("position")
    start_col = col_map.get("start")
    end_col = col_map.get("end")

    if start_col is not None and end_col is not None:
        start = pd.to_numeric(var_df[start_col], errors="coerce").to_numpy(dtype=np.float64)
        end = pd.to_numeric(var_df[end_col], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            raise ValueError("Gene 'start' and 'end' coordinates must be finite numbers.")
        if np.any(start < 0) or np.any(end < start):
            raise ValueError("Gene coordinates must satisfy 0 <= start <= end.")
        if pos_col is not None:
            pos = pd.to_numeric(var_df[pos_col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(pos)):
                raise ValueError("Gene 'pos' coordinates must be finite numbers.")
        else:
            pos = 0.5 * (start + end)
    elif pos_col is not None:
        pos = pd.to_numeric(var_df[pos_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(pos)) or np.any(pos < 0):
            raise ValueError("Gene 'pos' coordinates must be finite and non-negative.")
        start = pos.copy()
        end = pos.copy()
    else:
        raise ValueError(
            "Gene coordinate table must contain 'pos' or both 'start' and 'end' columns."
        )

    return chrom, pos.astype(np.int64), start.astype(np.int64), end.astype(np.int64)


def _resolve_ref_indices(
    cell_ids: tuple[str, ...],
    ref_cells: str | Sequence[str] | Sequence[int] | np.ndarray | pd.Series | None,
    obs_df: pd.DataFrame | None = None,
    ref_key: str | None = None,
    ref_label: Any = None,
) -> np.ndarray | None:
    """Resolve reference cell specification to sorted unique integer indices."""
    n_cells = len(cell_ids)
    if ref_key is not None:
        if obs_df is None or ref_key not in obs_df.columns:
            raise ValueError(f"Reference key {ref_key!r} not found in adata.obs.")
        col = obs_df[ref_key]
        if ref_label is not None:
            mask = (col == ref_label).to_numpy(dtype=bool)
        else:
            mask = col.to_numpy(dtype=bool)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(
                f"Reference selection via ref_key={ref_key!r}, ref_label={ref_label!r} matched 0 cells."
            )
        return idx

    if ref_cells is None:
        return None

    if isinstance(ref_cells, str):
        if obs_df is not None and ref_cells in obs_df.columns and ref_cells not in cell_ids:
            col = obs_df[ref_cells]
            if pd.api.types.is_bool_dtype(col):
                idx = np.flatnonzero(col.to_numpy(dtype=bool))
                if idx.size == 0:
                    raise ValueError(f"Boolean reference column {ref_cells!r} has 0 True cells.")
                return idx
        ref_seq: Sequence[Any] = [ref_cells]
    elif isinstance(ref_cells, (pd.Series, pd.Index)):
        if pd.api.types.is_bool_dtype(ref_cells) and len(ref_cells) == n_cells:
            idx = np.flatnonzero(ref_cells.to_numpy(dtype=bool))
            if idx.size == 0:
                raise ValueError("Boolean ref_cells mask matched 0 cells.")
            return idx
        ref_seq = ref_cells.tolist()
    elif isinstance(ref_cells, np.ndarray):
        if ref_cells.dtype == bool:
            if ref_cells.shape != (n_cells,):
                raise ValueError(
                    f"Boolean ref_cells mask shape {ref_cells.shape} != ({n_cells},)."
                )
            idx = np.flatnonzero(ref_cells)
            if idx.size == 0:
                raise ValueError("Boolean ref_cells mask matched 0 cells.")
            return idx
        ref_seq = ref_cells.tolist()
    else:
        ref_seq = list(ref_cells)

    if len(ref_seq) == 0:
        raise ValueError("ref_cells was provided as an empty selection.")

    if all(isinstance(x, (bool, np.bool_)) for x in ref_seq) and len(ref_seq) == n_cells:
        idx = np.flatnonzero(np.asarray(ref_seq, dtype=bool))
        if idx.size == 0:
            raise ValueError("Boolean ref_cells mask matched 0 cells.")
        return idx

    if all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in ref_seq):
        idx = np.unique(np.asarray(ref_seq, dtype=np.int64))
        if np.any(idx < 0) or np.any(idx >= n_cells):
            raise ValueError("Integer ref_cells indices are out of bounds.")
        return idx

    cell_lookup = {cid: i for i, cid in enumerate(cell_ids)}
    resolved = []
    missing = []
    for item in ref_seq:
        key = str(item)
        if key in cell_lookup:
            resolved.append(cell_lookup[key])
        else:
            missing.append(key)
    if missing and len(resolved) == 0:
        raise ValueError(f"None of the specified ref_cells matched cell IDs (e.g. {missing[:3]}).")
    if missing:
        raise ValueError(f"Unknown cell IDs in ref_cells: {missing[:5]}.")
    return np.unique(np.asarray(resolved, dtype=np.int64))


def extract_expression_input(
    data: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    gene_info: pd.DataFrame | None = None,
    layer: str | None = None,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    ref_cells: str | Sequence[str] | Sequence[int] | np.ndarray | pd.Series | None = None,
    ref_key: str | None = None,
    ref_label: Any = None,
    require_non_negative: bool = True,
) -> PreparedExpression:
    """Validate and normalize AnnData, DataFrame, or array expression input (`cells x genes`)."""
    obs_df: pd.DataFrame | None = None

    if isinstance(data, ad.AnnData):
        raw_X = data.layers[layer] if layer is not None else data.X
        matrix, is_sparse = _coerce_matrix(raw_X, require_non_negative=require_non_negative)
        c_ids = tuple(str(c) for c in (cell_ids if cell_ids is not None else data.obs_names))
        g_ids = tuple(str(g) for g in (gene_ids if gene_ids is not None else data.var_names))
        obs_df = data.obs
        var_df = gene_info if gene_info is not None else data.var
    else:
        if layer is not None:
            raise ValueError("'layer' argument is only supported for AnnData inputs.")
        matrix, is_sparse = _coerce_matrix(data, require_non_negative=require_non_negative)
        n_cells, n_genes = matrix.shape
        if isinstance(data, pd.DataFrame):
            c_ids = tuple(str(c) for c in (cell_ids if cell_ids is not None else data.index))
            g_ids = tuple(str(g) for g in (gene_ids if gene_ids is not None else data.columns))
        else:
            c_ids = (
                tuple(str(c) for c in cell_ids)
                if cell_ids is not None
                else tuple(f"cell_{i}" for i in range(n_cells))
            )
            g_ids = (
                tuple(str(g) for g in gene_ids)
                if gene_ids is not None
                else tuple(f"gene_{j}" for j in range(n_genes))
            )
        if gene_info is None:
            raise ValueError("gene_info DataFrame is required when input is not an AnnData object.")
        var_df = gene_info

    n_cells, n_genes = matrix.shape
    if len(c_ids) != n_cells:
        raise ValueError(f"cell_ids length ({len(c_ids)}) != matrix rows ({n_cells}).")
    if len(g_ids) != n_genes:
        raise ValueError(f"gene_ids length ({len(g_ids)}) != matrix cols ({n_genes}).")

    if isinstance(var_df, pd.DataFrame) and "gene" in var_df.columns and len(var_df) == n_genes:
        if list(var_df["gene"].astype(str)) != list(g_ids):
            lookup = var_df.assign(_g=var_df["gene"].astype(str)).drop_duplicates("_g").set_index("_g")
            missing_genes = [g for g in g_ids if g not in lookup.index]
            if missing_genes:
                raise ValueError(
                    f"Genes missing from gene_info: {missing_genes[:5]}."
                )
            var_df = lookup.loc[list(g_ids)].reset_index(drop=True)

    gene_chrom, gene_pos, gene_start, gene_end = _resolve_gene_coords(var_df, n_genes)
    ref_indices = _resolve_ref_indices(
        c_ids, ref_cells=ref_cells, obs_df=obs_df, ref_key=ref_key, ref_label=ref_label
    )

    return PreparedExpression(
        matrix=matrix,
        is_sparse=is_sparse,
        cell_ids=c_ids,
        gene_ids=g_ids,
        gene_chrom=gene_chrom,
        gene_pos=gene_pos,
        gene_start=gene_start,
        gene_end=gene_end,
        ref_indices=ref_indices,
    )


def parse_allele_dataframe(
    allele_df: pd.DataFrame | None,
    cell_ids: Sequence[str],
) -> PreparedAlleles | None:
    """Validate and aggregate phased allele counts `(cell, chrom, pos, AD, DP)` into CSR matrices."""
    if allele_df is None or len(allele_df) == 0:
        return None
    if not isinstance(allele_df, pd.DataFrame):
        raise ValueError("allele_df must be a pandas DataFrame or None.")

    col_map = {c.lower(): c for c in allele_df.columns}
    cell_col = col_map.get("cell") or col_map.get("cell_id") or col_map.get("barcode")
    chrom_col = col_map.get("chrom") or col_map.get("chromosome") or col_map.get("chr")
    pos_col = col_map.get("pos") or col_map.get("position") or col_map.get("start")
    ad_col = (
        col_map.get("ad")
        or col_map.get("pad")
        or col_map.get("alt")
        or col_map.get("y")
    )
    dp_col = col_map.get("dp") or col_map.get("depth") or col_map.get("n")
    cm_col = col_map.get("cm")
    snp_col = col_map.get("snp_id") or col_map.get("snp") or col_map.get("rsid")

    missing = []
    if cell_col is None:
        missing.append("cell")
    if chrom_col is None:
        missing.append("chrom")
    if pos_col is None:
        missing.append("pos")
    if ad_col is None:
        missing.append("AD")
    if dp_col is None:
        missing.append("DP")
    if missing:
        raise ValueError(f"allele_df is missing required columns: {missing}.")

    if allele_df[chrom_col].isna().any() or allele_df[cell_col].isna().any():
        raise ValueError("allele_df contains missing values in 'cell' or 'chrom'.")

    pos_vals = pd.to_numeric(allele_df[pos_col], errors="coerce").to_numpy(dtype=np.float64)
    ad_vals = pd.to_numeric(allele_df[ad_col], errors="coerce").to_numpy(dtype=np.float64)
    dp_vals = pd.to_numeric(allele_df[dp_col], errors="coerce").to_numpy(dtype=np.float64)

    if not (
        np.all(np.isfinite(pos_vals))
        and np.all(np.isfinite(ad_vals))
        and np.all(np.isfinite(dp_vals))
    ):
        raise ValueError("allele_df coordinates and allele counts (AD, DP) must be finite.")
    if np.any(pos_vals < 0):
        raise ValueError("allele_df genomic positions must be non-negative.")
    if np.any(ad_vals < 0) or np.any(dp_vals < 0):
        raise ValueError("allele_df AD and DP counts must be non-negative.")
    if np.any(ad_vals > dp_vals):
        raise ValueError("allele_df contains rows with AD > DP.")

    if cm_col is not None:
        cm_vals = pd.to_numeric(allele_df[cm_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(cm_vals)) or np.any(cm_vals < 0):
            raise ValueError("allele_df 'cM' column must contain finite non-negative numbers.")
    else:
        cm_vals = pos_vals * 1e-6

    cell_lookup = {str(cid): idx for idx, cid in enumerate(cell_ids)}
    raw_cells = allele_df[cell_col].astype(str).to_numpy()
    cell_idx = np.array([cell_lookup.get(c, -1) for c in raw_cells], dtype=np.int64)
    keep = cell_idx >= 0
    if not np.any(keep):
        raise ValueError("allele_df has zero cells matching the expression cell IDs.")

    cell_idx = cell_idx[keep]
    chroms = allele_df[chrom_col].astype(str).to_numpy()[keep]
    pos_int = pos_vals[keep].astype(np.int64)
    ad_keep = ad_vals[keep]
    dp_keep = dp_vals[keep]
    cm_keep = cm_vals[keep]
    raw_snp_ids = (
        allele_df[snp_col].astype(str).to_numpy()[keep]
        if snp_col is not None
        else np.array([f"{c}:{p}" for c, p in zip(chroms, pos_int)], dtype=object)
    )

    snp_df = pd.DataFrame(
        {
            "chrom": chroms,
            "pos": pos_int,
            "cm": cm_keep,
            "snp_id": raw_snp_ids,
        }
    )
    unique_snps = (
        snp_df.groupby(["chrom", "pos"], sort=False, as_index=False)
        .agg({"cm": "median", "snp_id": "first"})
    )
    sort_keys = [
        (canonical_chrom_key(c), int(p), i)
        for i, (c, p) in enumerate(zip(unique_snps["chrom"], unique_snps["pos"]))
    ]
    order = np.array([item[2] for item in sorted(sort_keys)], dtype=np.int64)
    unique_snps = unique_snps.iloc[order].reset_index(drop=True)

    snp_chrom = unique_snps["chrom"].astype(str).to_numpy()
    snp_pos = unique_snps["pos"].to_numpy(dtype=np.int64)
    snp_cm = unique_snps["cm"].to_numpy(dtype=np.float64)
    snp_ids = tuple(unique_snps["snp_id"].astype(str).tolist())

    snp_lookup = {(c, int(p)): idx for idx, (c, p) in enumerate(zip(snp_chrom, snp_pos))}
    row_snp_idx = np.array(
        [snp_lookup[(c, int(p))] for c, p in zip(chroms, pos_int)],
        dtype=np.int64,
    )

    n_cells = len(cell_ids)
    n_snps = len(snp_ids)
    ad_mat = sparse.coo_matrix((ad_keep, (cell_idx, row_snp_idx)), shape=(n_cells, n_snps)).tocsr()
    dp_mat = sparse.coo_matrix((dp_keep, (cell_idx, row_snp_idx)), shape=(n_cells, n_snps)).tocsr()
    ad_mat.sum_duplicates()
    dp_mat.sum_duplicates()

    return PreparedAlleles(
        ad_mat=ad_mat,
        dp_mat=dp_mat,
        snp_ids=snp_ids,
        snp_chrom=snp_chrom,
        snp_pos=snp_pos,
        snp_cm=snp_cm,
    )


def build_ordered_loci(
    expr: PreparedExpression,
    alleles: PreparedAlleles | None,
) -> OrderedLoci:
    """Combine gene expression and phased SNP loci into canonically sorted chromosome slices."""
    records: list[tuple[tuple[int, int, str], int, int, int, str, str, int, int, int, int, float]] = []

    for g_idx, (gid, c, p, st, en) in enumerate(
        zip(expr.gene_ids, expr.gene_chrom, expr.gene_pos, expr.gene_start, expr.gene_end)
    ):
        records.append(
            (
                canonical_chrom_key(str(c)),
                int(p),
                0,
                g_idx,
                gid,
                str(c),
                int(st),
                int(en),
                g_idx,
                -1,
                float(p) * 1e-6,
            )
        )

    if alleles is not None:
        for s_idx, (sid, c, p, cm) in enumerate(
            zip(alleles.snp_ids, alleles.snp_chrom, alleles.snp_pos, alleles.snp_cm)
        ):
            records.append(
                (
                    canonical_chrom_key(str(c)),
                    int(p),
                    1,
                    s_idx,
                    sid,
                    str(c),
                    int(p),
                    int(p),
                    -1,
                    s_idx,
                    float(cm),
                )
            )

    if not records:
        raise ValueError("No genomic loci found across genes or alleles.")

    records.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    n_loci = len(records)
    locus_id = tuple(r[4] for r in records)
    chrom = np.array([r[5] for r in records], dtype=object)
    pos = np.array([r[1] for r in records], dtype=np.int64)
    start = np.array([r[6] for r in records], dtype=np.int64)
    end = np.array([r[7] for r in records], dtype=np.int64)
    gene_index = np.array([r[8] for r in records], dtype=np.int64)
    snp_index = np.array([r[9] for r in records], dtype=np.int64)
    has_expr = gene_index >= 0
    has_allele = snp_index >= 0
    raw_cm = np.array([r[10] for r in records], dtype=np.float64)

    key_to_label: dict[tuple[int, int, str], str] = {}
    for g_chrom in expr.gene_chrom:
        g_key = canonical_chrom_key(str(g_chrom))
        if g_key not in key_to_label:
            key_to_label[g_key] = str(g_chrom)

    chrom_names_list: list[str] = []
    offsets = [0]
    dist_cm = np.zeros(n_loci, dtype=np.float64)

    prev_chrom_key: tuple[int, int, str] | None = None
    cur_chrom_label = ""
    prev_snp_cm: float | None = None
    for t in range(n_loci):
        c_key = records[t][0]
        if c_key != prev_chrom_key:
            if prev_chrom_key is not None:
                offsets.append(t)
            cur_chrom_label = key_to_label.get(c_key, str(chrom[t]))
            chrom_names_list.append(cur_chrom_label)
            prev_chrom_key = c_key
            prev_snp_cm = None
        chrom[t] = cur_chrom_label

        if has_allele[t]:
            if prev_snp_cm is not None:
                dist_cm[t] = max(0.0, raw_cm[t] - prev_snp_cm)
            else:
                dist_cm[t] = 0.0
            prev_snp_cm = raw_cm[t]
        else:
            dist_cm[t] = 0.0

    offsets.append(n_loci)

    return OrderedLoci(
        n_loci=n_loci,
        locus_id=locus_id,
        chrom=chrom,
        pos=pos,
        start=start,
        end=end,
        has_expr=has_expr,
        has_allele=has_allele,
        gene_index=gene_index,
        snp_index=snp_index,
        dist_cm=dist_cm,
        chrom_names=tuple(chrom_names_list),
        chrom_offsets=np.asarray(offsets, dtype=np.int64),
    )
