"""
Instrument QC helpers for Mendelian Randomization:
  - Intermediate-EAF palindromic SNP filtering (`A/T`, `C/G`)
  - Greedy p-value-ordered LD clumping (`r^2 < threshold`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_mr_instruments(
    df: pd.DataFrame,
    r2_matrix: np.ndarray | None = None,
    pval_col: str = "pval_exp",
    ref_col: str = "ref",
    alt_col: str = "alt",
    eaf_col: str = "eaf",
    pval_threshold: float = 5e-8,
    r2_threshold: float = 0.01,
    palindromic_eaf_window: tuple[float, float] = (0.42, 0.58),
) -> pd.DataFrame:
    """Filter ambiguous palindromic SNPs, threshold by exposure p-value, and clump by LD `r^2`.

    Parameters
    ----------
    df : DataFrame
        Summary statistics table indexed 0..M-1 matching rows/columns of `r2_matrix` (if supplied).
    r2_matrix : optional (M, M) ndarray
        Pairwise squared LD correlation matrix aligned to `df` row positions.
    """
    work = df.reset_index(drop=True).copy()
    keep_mask = np.ones(len(work), dtype=bool)

    if {ref_col, alt_col, eaf_col}.issubset(work.columns):
        alleles = work[ref_col].astype(str).str.upper() + work[alt_col].astype(str).str.upper()
        is_palindromic = alleles.isin({"AT", "TA", "CG", "GC"}).to_numpy()
        eaf = work[eaf_col].to_numpy(dtype=np.float64)
        is_ambiguous = (eaf > palindromic_eaf_window[0]) & (eaf < palindromic_eaf_window[1])
        keep_mask &= ~(is_palindromic & is_ambiguous)

    if pval_col in work.columns:
        keep_mask &= work[pval_col].to_numpy(dtype=np.float64) < pval_threshold

    cand_positions = np.flatnonzero(keep_mask)
    if r2_matrix is None or cand_positions.size <= 1:
        return work.iloc[cand_positions].reset_index(drop=True)

    r2 = np.asarray(r2_matrix, dtype=np.float64)
    if r2.shape != (len(work), len(work)):
        raise ValueError(f"r2_matrix shape {r2.shape} must match (len(df), len(df)) = ({len(work)}, {len(work)}).")

    if pval_col in work.columns:
        order = cand_positions[np.argsort(work[pval_col].to_numpy(dtype=np.float64)[cand_positions], kind="mergesort")]
    else:
        order = cand_positions

    kept: list[int] = []
    for pos in order:
        if all(r2[pos, k] < r2_threshold for k in kept):
            kept.append(int(pos))

    return work.iloc[kept].reset_index(drop=True)
