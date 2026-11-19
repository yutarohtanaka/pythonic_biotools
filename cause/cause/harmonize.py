"""
GWAS summary statistics harmonization, formatting, and LD pruning for CAUSE.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

_COMP: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}
_PALINDROMIC: frozenset[str] = frozenset({"AT", "TA", "CG", "GC"})


def _pick_col(
    df: pd.DataFrame,
    preferred: str | None,
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    if preferred is not None:
        if preferred in df.columns:
            return preferred
        if required:
            raise KeyError(
                f"Required column '{preferred}' not found in DataFrame columns {list(df.columns)}."
            )
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        label = preferred if preferred is not None else "/".join(candidates)
        raise KeyError(
            f"Required column '{label}' not found in DataFrame columns {list(df.columns)}."
        )
    return None


def _clean_snp_series(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    is_na = series.isna().to_numpy(dtype=bool)
    cleaned = series.astype(str).str.strip()
    invalid = is_na | (cleaned == "") | (cleaned.str.lower() == "nan") | (cleaned.str.lower() == "none")
    return cleaned, ~invalid.to_numpy(dtype=bool)


def gwas_merge(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    snp_name_cols: tuple[str, str] = ("snp", "snp"),
    beta_hat_cols: tuple[str, str] = ("beta_hat", "beta_hat"),
    se_cols: tuple[str, str] = ("se", "se"),
    A1_cols: tuple[str, str] | None = ("A1", "A1"),
    A2_cols: tuple[str, str] | None = ("A2", "A2"),
    pval_cols: tuple[str, str] | None = None,
    drop_ambiguous: bool = True,
    allow_strand_flip: bool = True,
) -> pd.DataFrame:
    """Merge and allele-align two GWAS summary statistics DataFrames for CAUSE.

    Parameters
    ----------
    X1, X2 : pd.DataFrame
        Exposure (`X1`) and outcome (`X2`) GWAS summary statistics tables.
    snp_name_cols, beta_hat_cols, se_cols : tuple of (str, str)
        Column names in `(X1, X2)` for SNP identifier, effect estimate, and SE.
    A1_cols, A2_cols : tuple of (str, str) or None
        Effect allele (`A1`) and other allele (`A2`) columns in `(X1, X2)`.
        If either is `None`, SNPs are matched by SNP identifier without allele alignment.
    pval_cols : tuple of (str, str) or None
        Optional p-value columns in `(X1, X2)`. If `None`, p-values are computed
        from normal z-scores `2 * norm.sf(|beta_hat / se|)`.
    drop_ambiguous : bool, default True
        If `True`, drop palindromic SNPs (`A/T`, `T/A`, `C/G`, `G/C`). If `False`,
        palindromic SNPs are kept only on exact allele match (`A1_1 == A1_2` and
        `A2_1 == A2_2`).
    allow_strand_flip : bool, default True
        If `True`, resolve opposite-strand matches via Watson-Crick complementation
        for non-palindromic variants.

    Returns
    -------
    pd.DataFrame
        Merged, aligned table with columns `snp`, `beta_hat_1`, `seb1`,
        `beta_hat_2`, `seb2`, optional `A1`, `A2`, and `p1`, `p2`.
    """
    if not isinstance(X1, pd.DataFrame) or not isinstance(X2, pd.DataFrame):
        raise TypeError("X1 and X2 must be pandas DataFrames.")
    if X1.empty or X2.empty:
        raise ValueError("Both X1 and X2 must be non-empty DataFrames.")

    for col, df, label in (
        (snp_name_cols[0], X1, "X1"),
        (beta_hat_cols[0], X1, "X1"),
        (se_cols[0], X1, "X1"),
        (snp_name_cols[1], X2, "X2"),
        (beta_hat_cols[1], X2, "X2"),
        (se_cols[1], X2, "X2"),
    ):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {label}.")

    use_alleles = A1_cols is not None and A2_cols is not None
    if use_alleles:
        for col, df, label in (
            (A1_cols[0], X1, "X1"),
            (A2_cols[0], X1, "X1"),
            (A1_cols[1], X2, "X2"),
            (A2_cols[1], X2, "X2"),
        ):
            if col not in df.columns:
                raise KeyError(f"Allele column '{col}' missing from {label}.")

    if pval_cols is not None:
        if pval_cols[0] not in X1.columns:
            raise KeyError(f"P-value column '{pval_cols[0]}' missing from X1.")
        if pval_cols[1] not in X2.columns:
            raise KeyError(f"P-value column '{pval_cols[1]}' missing from X2.")

    def _prepare_study(
        df: pd.DataFrame,
        snp_col: str,
        beta_col: str,
        se_col: str,
        a1_col: str | None,
        a2_col: str | None,
        p_col: str | None,
        suffix: str,
    ) -> pd.DataFrame:
        snp_clean, snp_ok = _clean_snp_series(df[snp_col])
        beta_vals = pd.to_numeric(df[beta_col], errors="coerce").to_numpy(dtype=np.float64)
        se_vals = pd.to_numeric(df[se_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = snp_ok & np.isfinite(beta_vals) & np.isfinite(se_vals) & (se_vals > 0.0)

        out = pd.DataFrame(
            {
                "snp": snp_clean.to_numpy()[valid],
                f"beta_{suffix}": beta_vals[valid],
                f"se_{suffix}": se_vals[valid],
            }
        )
        if p_col is not None:
            p_vals = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=np.float64)[valid]
            bad_p = (~np.isfinite(p_vals)) | (p_vals < 0.0) | (p_vals > 1.0)
            if np.any(bad_p):
                z = np.abs(out[f"beta_{suffix}"].to_numpy() / out[f"se_{suffix}"].to_numpy())
                p_vals = np.where(bad_p, 2.0 * stats.norm.sf(z), p_vals)
            out[f"p_{suffix}"] = p_vals
        else:
            z = np.abs(out[f"beta_{suffix}"].to_numpy() / out[f"se_{suffix}"].to_numpy())
            out[f"p_{suffix}"] = 2.0 * stats.norm.sf(z)

        if a1_col is not None and a2_col is not None:
            a1_s = df[a1_col].astype(str).str.strip().str.upper().to_numpy()[valid]
            a2_s = df[a2_col].astype(str).str.strip().str.upper().to_numpy()[valid]
            out[f"A1_{suffix}"] = a1_s
            out[f"A2_{suffix}"] = a2_s

        out = out.drop_duplicates(subset=["snp"], keep="first")
        return out

    left = _prepare_study(
        X1,
        snp_name_cols[0],
        beta_hat_cols[0],
        se_cols[0],
        A1_cols[0] if use_alleles else None,
        A2_cols[0] if use_alleles else None,
        pval_cols[0] if pval_cols is not None else None,
        "1",
    )
    right = _prepare_study(
        X2,
        snp_name_cols[1],
        beta_hat_cols[1],
        se_cols[1],
        A1_cols[1] if use_alleles else None,
        A2_cols[1] if use_alleles else None,
        pval_cols[1] if pval_cols is not None else None,
        "2",
    )

    merged = pd.merge(left, right, on="snp", how="inner", sort=False)
    if merged.empty:
        cols = ["snp", "beta_hat_1", "seb1", "beta_hat_2", "seb2"]
        if use_alleles:
            cols.extend(["A1", "A2"])
        cols.extend(["p1", "p2"])
        return pd.DataFrame(columns=cols)

    if use_alleles:
        a1_1 = merged["A1_1"]
        a2_1 = merged["A2_1"]
        a1_2 = merged["A1_2"]
        a2_2 = merged["A2_2"]

        palindromic = (a1_1 + a2_1).isin(_PALINDROMIC) | (a1_2 + a2_2).isin(_PALINDROMIC)
        same = (a1_1 == a1_2) & (a2_1 == a2_2)
        swap = (a1_1 == a2_2) & (a2_1 == a1_2)

        if allow_strand_flip:
            c_a1_2 = a1_2.map(_COMP)
            c_a2_2 = a2_2.map(_COMP)
            comp = (a1_1 == c_a1_2) & (a2_1 == c_a2_2) & (~palindromic)
            comp_swap = (a1_1 == c_a2_2) & (a2_1 == c_a1_2) & (~palindromic)
        else:
            comp = pd.Series(False, index=merged.index)
            comp_swap = pd.Series(False, index=merged.index)

        if drop_ambiguous:
            keep = (~palindromic) & (same | swap | comp | comp_swap)
        else:
            keep = ((~palindromic) & (same | swap | comp | comp_swap)) | (palindromic & same)

        sign = np.where((swap | comp_swap).to_numpy(dtype=bool), -1.0, 1.0)
        keep_mask = keep.to_numpy(dtype=bool)
        merged = merged.loc[keep_mask].copy()
        sign = sign[keep_mask]

        out = pd.DataFrame(
            {
                "snp": merged["snp"].to_numpy(),
                "beta_hat_1": merged["beta_1"].to_numpy(dtype=np.float64),
                "seb1": merged["se_1"].to_numpy(dtype=np.float64),
                "beta_hat_2": merged["beta_2"].to_numpy(dtype=np.float64) * sign,
                "seb2": merged["se_2"].to_numpy(dtype=np.float64),
                "A1": merged["A1_1"].to_numpy(),
                "A2": merged["A2_1"].to_numpy(),
                "p1": merged["p_1"].to_numpy(dtype=np.float64),
                "p2": merged["p_2"].to_numpy(dtype=np.float64),
            }
        )
    else:
        out = pd.DataFrame(
            {
                "snp": merged["snp"].to_numpy(),
                "beta_hat_1": merged["beta_1"].to_numpy(dtype=np.float64),
                "seb1": merged["se_1"].to_numpy(dtype=np.float64),
                "beta_hat_2": merged["beta_2"].to_numpy(dtype=np.float64),
                "seb2": merged["se_2"].to_numpy(dtype=np.float64),
                "p1": merged["p_1"].to_numpy(dtype=np.float64),
                "p2": merged["p_2"].to_numpy(dtype=np.float64),
            }
        )

    return out.reset_index(drop=True)


def _extract_cause_arrays(
    data: pd.DataFrame | None = None,
    *,
    beta_hat_1: Sequence[float] | np.ndarray | str | None = None,
    seb1: Sequence[float] | np.ndarray | str | None = None,
    beta_hat_2: Sequence[float] | np.ndarray | str | None = None,
    seb2: Sequence[float] | np.ndarray | str | None = None,
    snp: Sequence[str] | str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate 1D arrays `(snp_ids, b1, s1, b2, s2)` from DataFrame or arrays."""
    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame or None (got {type(data).__name__}).")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        snp_pref = snp if isinstance(snp, str) else None
        b1_pref = beta_hat_1 if isinstance(beta_hat_1, str) else None
        s1_pref = seb1 if isinstance(seb1, str) else None
        b2_pref = beta_hat_2 if isinstance(beta_hat_2, str) else None
        s2_pref = seb2 if isinstance(seb2, str) else None

        snp_col = _pick_col(data, snp_pref, ("snp", "SNP", "rsid"), required=isinstance(snp, str))
        b1_col = _pick_col(
            data, b1_pref, ("beta_hat_1", "beta_exp", "xbeta", "BETA_exp", "b_exp")
        )
        s1_col = _pick_col(data, s1_pref, ("seb1", "se_exp", "xse", "SE_exp"))
        b2_col = _pick_col(
            data, b2_pref, ("beta_hat_2", "beta_out", "ybeta", "BETA_out", "b_out")
        )
        s2_col = _pick_col(data, s2_pref, ("seb2", "se_out", "yse", "SE_out"))

        if snp is not None and not isinstance(snp, str):
            snp_ids = [str(x) for x in snp]
        elif snp_col is not None:
            snp_ids = data[snp_col].astype(str).tolist()
        else:
            snp_ids = [str(idx) for idx in data.index]

        b1 = np.asarray(data[b1_col], dtype=np.float64)
        s1 = np.asarray(data[s1_col], dtype=np.float64)
        b2 = np.asarray(data[b2_col], dtype=np.float64)
        s2 = np.asarray(data[s2_col], dtype=np.float64)
    else:
        if beta_hat_1 is None or seb1 is None or beta_hat_2 is None or seb2 is None:
            raise ValueError(
                "When data is None, beta_hat_1, seb1, beta_hat_2, and seb2 must all be provided."
            )
        b1 = np.asarray(beta_hat_1, dtype=np.float64)
        s1 = np.asarray(seb1, dtype=np.float64)
        b2 = np.asarray(beta_hat_2, dtype=np.float64)
        s2 = np.asarray(seb2, dtype=np.float64)
        if b1.ndim != 1 or s1.ndim != 1 or b2.ndim != 1 or s2.ndim != 1:
            raise ValueError("beta_hat_1, seb1, beta_hat_2, and seb2 must be 1D arrays.")
        if not (b1.size == s1.size == b2.size == s2.size):
            raise ValueError(
                f"Mismatched array lengths: {(b1.size, s1.size, b2.size, s2.size)}."
            )
        if snp is None:
            snp_ids = [f"snp_{j + 1}" for j in range(b1.size)]
        else:
            snp_ids = [str(x) for x in snp]
            if len(snp_ids) != b1.size:
                raise ValueError(
                    f"Length of snp ({len(snp_ids)}) does not match array length ({b1.size})."
                )

    if b1.size == 0:
        raise ValueError("At least 1 SNP is required.")
    if len(snp_ids) != b1.size:
        raise ValueError(
            f"Length of snp_ids ({len(snp_ids)}) does not match data length ({b1.size})."
        )
    if not (
        np.all(np.isfinite(b1))
        and np.all(np.isfinite(s1))
        and np.all(np.isfinite(b2))
        and np.all(np.isfinite(s2))
    ):
        raise ValueError("All beta_hat and se values must be finite (no NaN or Inf).")
    if np.any(s1 <= 0.0) or np.any(s2 <= 0.0):
        raise ValueError("All standard errors (seb1, seb2) must be strictly positive.")

    return snp_ids, b1, s1, b2, s2


def format_cause_data(
    data: pd.DataFrame | None = None,
    *,
    beta_hat_1: Sequence[float] | np.ndarray | str | None = None,
    seb1: Sequence[float] | np.ndarray | str | None = None,
    beta_hat_2: Sequence[float] | np.ndarray | str | None = None,
    seb2: Sequence[float] | np.ndarray | str | None = None,
    snp: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Validate and standardize summary statistics into canonical CAUSE columns.

    Accepts either a `pd.DataFrame` (supporting `jean997/cause` columns
    `beta_hat_1`, `seb1`, `beta_hat_2`, `seb2` as well as `twosamplemr` aliases
    `beta_exp`, `se_exp`, `beta_out`, `se_out`, `xbeta`/`ybeta`) or 1D arrays.
    """
    snp_ids, b1, s1, b2, s2 = _extract_cause_arrays(
        data,
        beta_hat_1=beta_hat_1,
        seb1=seb1,
        beta_hat_2=beta_hat_2,
        seb2=seb2,
        snp=snp,
    )
    p1 = 2.0 * stats.norm.sf(np.abs(b1 / s1))
    p2 = 2.0 * stats.norm.sf(np.abs(b2 / s2))

    out = pd.DataFrame(
        {
            "snp": snp_ids,
            "beta_hat_1": b1,
            "seb1": s1,
            "beta_hat_2": b2,
            "seb2": s2,
            "p1": p1,
            "p2": p2,
        }
    )
    if data is not None:
        for extra in ("A1", "A2", "chr", "pos"):
            if extra in data.columns:
                out[extra] = data[extra].to_numpy()
    return out


def ld_prune(
    data: pd.DataFrame,
    r2_matrix: pd.DataFrame | np.ndarray | None = None,
    pval_thresh: float = 1e-3,
    r2_thresh: float = 0.01,
    window_kb: float | None = None,
    chrom_col: str = "chr",
    pos_col: str = "pos",
    return_snps: bool = False,
) -> pd.DataFrame | list[str]:
    """Perform p-value-ordered greedy LD pruning on exposure instruments.

    Parameters
    ----------
    data : pd.DataFrame
        Summary statistics table (e.g., output of `gwas_merge` or `format_cause_data`).
    r2_matrix : pd.DataFrame or np.ndarray, optional
        Pairwise LD $r^2$ matrix. If a `DataFrame`, indexed and columned by SNP IDs.
        If an `ndarray`, must match `len(data)` along both axes.
    pval_thresh : float, default 1e-3
        Exposure p-value inclusion threshold (`p1 <= pval_thresh`).
    r2_thresh : float, default 0.01
        Maximum permitted pairwise $r^2$ among retained instruments.
    window_kb : float, optional
        Physical distance clumping window in kilobases (used when `r2_matrix` is `None`).
    chrom_col, pos_col : str
        Chromosome and base-pair position columns used with `window_kb`.
    return_snps : bool, default False
        If `True`, return a `list[str]` of retained SNP IDs instead of a `DataFrame`.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not (0.0 < pval_thresh <= 1.0):
        raise ValueError(f"pval_thresh must be in (0, 1] (got {pval_thresh!r}).")
    if not (0.0 <= r2_thresh <= 1.0):
        raise ValueError(f"r2_thresh must be in [0, 1] (got {r2_thresh!r}).")

    if data.empty:
        return [] if return_snps else data.copy()

    snp_col = _pick_col(data, None, ("snp", "SNP", "rsid"), required=False)
    snp_arr = (
        data[snp_col].astype(str).to_numpy()
        if snp_col is not None
        else data.index.astype(str).to_numpy()
    )

    p_col = _pick_col(data, None, ("p1", "pval_exp", "P1"), required=False)
    if p_col is not None:
        p1 = pd.to_numeric(data[p_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        b1_col = _pick_col(data, None, ("beta_hat_1", "beta_exp", "xbeta", "BETA_exp", "b_exp"))
        s1_col = _pick_col(data, None, ("seb1", "se_exp", "xse", "SE_exp"))
        b1 = pd.to_numeric(data[b1_col], errors="coerce").to_numpy(dtype=np.float64)
        s1 = pd.to_numeric(data[s1_col], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = 2.0 * stats.norm.sf(np.abs(b1 / s1))

    eligible = np.where(np.isfinite(p1) & (p1 <= pval_thresh))[0]
    if eligible.size == 0:
        return [] if return_snps else data.iloc[0:0].copy().reset_index(drop=True)

    # Deterministic sort by (p1, snp)
    order_df = pd.DataFrame(
        {"orig_idx": eligible, "p1": p1[eligible], "snp": snp_arr[eligible]}
    ).sort_values(["p1", "snp"], kind="mergesort")
    sorted_orig_idx = order_df["orig_idx"].to_numpy(dtype=int)
    sorted_snps = order_df["snp"].to_numpy()

    if r2_matrix is not None:
        if isinstance(r2_matrix, pd.DataFrame):
            missing = [s for s in sorted_snps if s not in r2_matrix.index or s not in r2_matrix.columns]
            if missing:
                raise KeyError(
                    f"SNPs missing from r2_matrix index/columns: {missing[:5]}"
                )
            r2_sub = r2_matrix.loc[sorted_snps, sorted_snps].to_numpy(dtype=np.float64)
            kept_pos: list[int] = []
            for pos in range(sorted_orig_idx.size):
                if not kept_pos:
                    kept_pos.append(pos)
                    continue
                vals = r2_sub[pos, kept_pos]
                if not np.any(np.isnan(vals) | (vals > r2_thresh)):
                    kept_pos.append(pos)
            kept_orig_idx = sorted_orig_idx[kept_pos]
        else:
            r2_arr = np.asarray(r2_matrix, dtype=np.float64)
            if r2_arr.shape != (len(data), len(data)):
                raise ValueError(
                    f"r2_matrix ndarray shape {r2_arr.shape} must match ({len(data)}, {len(data)})."
                )
            kept_orig: list[int] = []
            for orig in sorted_orig_idx:
                if not kept_orig:
                    kept_orig.append(int(orig))
                    continue
                vals = r2_arr[orig, kept_orig]
                if not np.any(np.isnan(vals) | (vals > r2_thresh)):
                    kept_orig.append(int(orig))
            kept_orig_idx = np.asarray(kept_orig, dtype=int)
    elif window_kb is not None:
        if window_kb < 0.0:
            raise ValueError(f"window_kb must be non-negative (got {window_kb!r}).")
        if chrom_col not in data.columns or pos_col not in data.columns:
            raise KeyError(
                f"Columns '{chrom_col}' and '{pos_col}' are required when window_kb is specified."
            )
        chroms = data[chrom_col].astype(str).to_numpy()
        positions = pd.to_numeric(data[pos_col], errors="coerce").to_numpy(dtype=np.float64)
        win_bp = float(window_kb) * 1000.0
        kept_by_chrom: dict[str, list[float]] = {}
        kept_orig_list: list[int] = []
        for orig in sorted_orig_idx:
            ch = chroms[orig]
            ps = positions[orig]
            prev_pos = kept_by_chrom.get(ch)
            if prev_pos is None or not np.any(np.abs(np.asarray(prev_pos) - ps) <= win_bp):
                kept_by_chrom.setdefault(ch, []).append(float(ps))
                kept_orig_list.append(int(orig))
        kept_orig_idx = np.asarray(kept_orig_list, dtype=int)
    else:
        kept_orig_idx = sorted_orig_idx

    if return_snps:
        return [str(snp_arr[i]) for i in kept_orig_idx]
    return data.iloc[kept_orig_idx].copy().reset_index(drop=True)
