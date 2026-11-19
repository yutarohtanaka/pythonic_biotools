"""
Instrument QC and allele harmonisation helpers for Mendelian Randomization:
  - Two-sample allele harmonisation (`harmonise_mr_data`, supporting `action=1, 2, 3`)
  - Intermediate- or missing-EAF palindromic SNP filtering (`A/T`, `C/G`)
  - Greedy p-value-ordered LD clumping (`r^2 < threshold`)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
_PALINDROMIC = {"AT", "TA", "CG", "GC"}


def _validate_settings(settings: str) -> Literal["default", "R_default"]:
    if not isinstance(settings, str) or settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings {settings!r}; expected 'default' or 'R_default'."
        )
    return "R_default" if settings == "R_default" else "default"


def _pick_col(df: pd.DataFrame, preferred: str | None, candidates: tuple[str, ...], required: bool = True) -> str | None:
    if preferred is not None and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        label = preferred if preferred is not None else "/".join(candidates)
        raise KeyError(f"Required column '{label}' not found in DataFrame columns {list(df.columns)}.")
    return None


def harmonise_mr_data(
    exp_df: pd.DataFrame,
    out_df: pd.DataFrame,
    action: int = 2,
    snp_col: str = "snp",
    ea_col: str | None = None,
    oa_col: str | None = None,
    beta_col: str = "beta",
    se_col: str = "se",
    eaf_col: str = "eaf",
    pval_col: str = "pval",
    palindromic_eaf_window: tuple[float, float] = (0.42, 0.58),
    settings: Literal["default", "R_default"] = "default",
) -> pd.DataFrame:
    """Harmonise exposure and outcome GWAS summary tables (`TwoSampleMR::harmonise_data`).

    Parameters
    ----------
    exp_df, out_df : DataFrame
        Exposure and outcome summary tables containing SNP IDs, effect/other alleles,
        effect sizes, standard errors, and (optionally) effect allele frequencies (`eaf`).
    action : {1, 2, 3}, default 2
        1: Assume all alleles are coded on the forward strand (do not complement; keep palindromics).
        2: Infer positive strand via complementation for non-palindromic SNPs and via EAF for
           palindromic SNPs; drop palindromic SNPs with missing or intermediate EAF in `palindromic_eaf_window`.
        3: Correct strand for non-palindromic SNPs and drop all palindromic SNPs regardless of EAF.
    settings : {"default", "R_default"}, default "default"
        Execution mode (`"default"` or upstream R `"R_default"`).
    """
    _validate_settings(settings)
    if action not in (1, 2, 3):
        raise ValueError(f"action must be 1, 2, or 3 (got {action!r}).")

    snp_x = _pick_col(exp_df, snp_col, ("snp", "SNP", "rsid"))
    snp_y = _pick_col(out_df, snp_col, ("snp", "SNP", "rsid"))
    ea_x_col = _pick_col(exp_df, ea_col, ("effect_allele", "ea", "EA", "A1", "alt"))
    oa_x_col = _pick_col(exp_df, oa_col, ("other_allele", "oa", "OA", "A2", "ref"))
    ea_y_col = _pick_col(out_df, ea_col, ("effect_allele", "ea", "EA", "A1", "alt"))
    oa_y_col = _pick_col(out_df, oa_col, ("other_allele", "oa", "OA", "A2", "ref"))

    bx_col = _pick_col(exp_df, beta_col, ("beta", "beta_exp", "BETA"))
    sx_col = _pick_col(exp_df, se_col, ("se", "se_exp", "SE"))
    by_col = _pick_col(out_df, beta_col, ("beta", "beta_out", "BETA"))
    sy_col = _pick_col(out_df, se_col, ("se", "se_out", "SE"))

    eaf_x_col = _pick_col(exp_df, eaf_col, ("eaf", "eaf_exp", "EAF", "freq"), required=False)
    eaf_y_col = _pick_col(out_df, eaf_col, ("eaf", "eaf_out", "EAF", "freq"), required=False)
    px_col = _pick_col(exp_df, pval_col, ("pval", "pval_exp", "P"), required=False)
    py_col = _pick_col(out_df, pval_col, ("pval", "pval_out", "P"), required=False)

    left = pd.DataFrame(
        {
            "snp": exp_df[snp_x].astype(str),
            "ea_x": exp_df[ea_x_col].astype(str).str.upper(),
            "oa_x": exp_df[oa_x_col].astype(str).str.upper(),
            "beta_exp": pd.to_numeric(exp_df[bx_col], errors="coerce").to_numpy(dtype=np.float64),
            "se_exp": pd.to_numeric(exp_df[sx_col], errors="coerce").to_numpy(dtype=np.float64),
            "eaf_exp": (
                pd.to_numeric(exp_df[eaf_x_col], errors="coerce").to_numpy(dtype=np.float64)
                if eaf_x_col is not None
                else np.full(len(exp_df), np.nan, dtype=np.float64)
            ),
            "pval_exp": (
                pd.to_numeric(exp_df[px_col], errors="coerce").to_numpy(dtype=np.float64)
                if px_col is not None
                else np.full(len(exp_df), np.nan, dtype=np.float64)
            ),
        }
    ).drop_duplicates(subset=["snp"], keep="first")

    right = pd.DataFrame(
        {
            "snp": out_df[snp_y].astype(str),
            "ea_y": out_df[ea_y_col].astype(str).str.upper(),
            "oa_y": out_df[oa_y_col].astype(str).str.upper(),
            "beta_out": pd.to_numeric(out_df[by_col], errors="coerce").to_numpy(dtype=np.float64),
            "se_out": pd.to_numeric(out_df[sy_col], errors="coerce").to_numpy(dtype=np.float64),
            "eaf_out": (
                pd.to_numeric(out_df[eaf_y_col], errors="coerce").to_numpy(dtype=np.float64)
                if eaf_y_col is not None
                else np.full(len(out_df), np.nan, dtype=np.float64)
            ),
            "pval_out": (
                pd.to_numeric(out_df[py_col], errors="coerce").to_numpy(dtype=np.float64)
                if py_col is not None
                else np.full(len(out_df), np.nan, dtype=np.float64)
            ),
        }
    ).drop_duplicates(subset=["snp"], keep="first")

    m = left.merge(right, on="snp", how="inner")
    if m.empty:
        return pd.DataFrame(
            columns=[
                "snp",
                "effect_allele",
                "other_allele",
                "beta_exp",
                "se_exp",
                "beta_out",
                "se_out",
                "eaf_exp",
                "eaf_out",
                "pval_exp",
                "pval_out",
                "palindromic",
            ]
        )

    ea_x = m["ea_x"]
    oa_x = m["oa_x"]
    ea_y = m["ea_y"]
    oa_y = m["oa_y"]
    ea_y_c = ea_y.map(_COMP).fillna("")
    oa_y_c = oa_y.map(_COMP).fillna("")

    pal_x = (ea_x + oa_x).isin(_PALINDROMIC).to_numpy()
    pal_y = (ea_y + oa_y).isin(_PALINDROMIC).to_numpy()
    same_direct = ((ea_x == ea_y) & (oa_x == oa_y)).to_numpy()
    swap_direct = ((ea_x == oa_y) & (oa_x == ea_y)).to_numpy()
    same_comp = ((ea_x == ea_y_c) & (oa_x == oa_y_c)).to_numpy()
    swap_comp = ((ea_x == oa_y_c) & (oa_x == ea_y_c)).to_numpy()

    eaf_x = m["eaf_exp"].to_numpy(dtype=np.float64)
    eaf_y = m["eaf_out"].to_numpy(dtype=np.float64)

    n = len(m)
    keep = np.zeros(n, dtype=bool)
    sign = np.ones(n, dtype=np.float64)

    non_pal = ~pal_x & ~pal_y
    if action == 1:
        # Forward strand assumed: direct match or direct allele swap (both non-palindromic and palindromic)
        keep = same_direct | swap_direct
        sign = np.where(swap_direct, -1.0, 1.0)
    else:
        # Non-palindromic SNPs: handle direct, swap, strand-flip, and strand-flip + swap
        keep[non_pal] = (same_direct | swap_direct | same_comp | swap_comp)[non_pal]
        sign[non_pal] = np.where((swap_direct | swap_comp)[non_pal], -1.0, 1.0)

        if action == 2:
            # Palindromic SNPs with matching allele pair {A,T} or {C,G}: resolve strand/swap via EAF
            pal_compat = pal_x & pal_y & (same_direct | swap_direct)
            lo_w, hi_w = palindromic_eaf_window
            eaf_unambig = (
                np.isfinite(eaf_x)
                & np.isfinite(eaf_y)
                & (eaf_x > 0.0)
                & (eaf_x < 1.0)
                & (eaf_y > 0.0)
                & (eaf_y < 1.0)
                & ~((eaf_x >= lo_w) & (eaf_x <= hi_w))
                & ~((eaf_y >= lo_w) & (eaf_y <= hi_w))
            )
            valid_pal = pal_compat & eaf_unambig
            keep[valid_pal] = True
            same_side = (eaf_x < 0.5) == (eaf_y < 0.5)
            sign[valid_pal] = np.where(same_side[valid_pal], 1.0, -1.0)
        # action == 3 drops all palindromic SNPs (keep[~non_pal] remains False)

    finite_stats = (
        np.isfinite(m["beta_exp"].to_numpy())
        & np.isfinite(m["se_exp"].to_numpy())
        & (m["se_exp"].to_numpy() > 0.0)
        & np.isfinite(m["beta_out"].to_numpy())
        & np.isfinite(m["se_out"].to_numpy())
        & (m["se_out"].to_numpy() > 0.0)
    )
    keep &= finite_stats

    m = m.loc[keep].copy()
    s_keep = sign[keep]
    bx = m["beta_exp"].to_numpy(dtype=np.float64)
    sx = m["se_exp"].to_numpy(dtype=np.float64)
    by = m["beta_out"].to_numpy(dtype=np.float64) * s_keep
    sy = m["se_out"].to_numpy(dtype=np.float64)
    eaf_out_aligned = np.where(s_keep < 0.0, 1.0 - m["eaf_out"].to_numpy(dtype=np.float64), m["eaf_out"].to_numpy(dtype=np.float64))

    pval_x = m["pval_exp"].to_numpy(dtype=np.float64)
    pval_x = np.where(np.isnan(pval_x), 2.0 * stats.norm.sf(np.abs(bx / sx)), pval_x)
    pval_y = m["pval_out"].to_numpy(dtype=np.float64)
    pval_y = np.where(np.isnan(pval_y), 2.0 * stats.norm.sf(np.abs(by / sy)), pval_y)

    return pd.DataFrame(
        {
            "snp": m["snp"].to_numpy(),
            "effect_allele": m["ea_x"].to_numpy(),
            "other_allele": m["oa_x"].to_numpy(),
            "beta_exp": bx,
            "se_exp": sx,
            "beta_out": by,
            "se_out": sy,
            "eaf_exp": m["eaf_exp"].to_numpy(dtype=np.float64),
            "eaf_out": eaf_out_aligned,
            "pval_exp": pval_x,
            "pval_out": pval_y,
            "palindromic": pal_x[keep],
        }
    ).reset_index(drop=True)


def select_mr_instruments(
    df: pd.DataFrame,
    r2_matrix: np.ndarray | None = None,
    pval_col: str = "pval_exp",
    ref_col: str = "ref",
    alt_col: str = "alt",
    eaf_col: str = "eaf",
    pval_threshold: float = 5e-8,
    r2_threshold: float | None = None,
    palindromic_eaf_window: tuple[float, float] = (0.42, 0.58),
    settings: Literal["default", "R_default"] = "default",
) -> pd.DataFrame:
    """Filter ambiguous palindromic SNPs, threshold by exposure p-value, and clump by LD `r^2`.

    Parameters
    ----------
    df : DataFrame
        Summary statistics table indexed 0..M-1 matching rows/columns of `r2_matrix` (if supplied).
    r2_matrix : optional (M, M) ndarray
        Pairwise squared LD correlation matrix aligned to `df` row positions.
    r2_threshold : float, optional
        LD `r^2` clumping threshold (`0.01` under `"default"`, `0.001` under `"R_default"`
        matching `TwoSampleMR::clump_data`).
    settings : {"default", "R_default"}, default "default"
        Execution mode (`"default"` or upstream R `"R_default"`).
    """
    eff_settings = _validate_settings(settings)
    if r2_threshold is None:
        r2_thresh = 0.001 if eff_settings == "R_default" else 0.01
    else:
        r2_thresh = float(r2_threshold)
    if not np.isfinite(r2_thresh) or r2_thresh <= 0.0:
        raise ValueError(f"r2_threshold must be finite and > 0; got {r2_threshold!r}.")

    work = df.reset_index(drop=True).copy()
    keep_mask = np.ones(len(work), dtype=bool)

    rc = ref_col if ref_col in work.columns else ("other_allele" if "other_allele" in work.columns else None)
    ac = alt_col if alt_col in work.columns else ("effect_allele" if "effect_allele" in work.columns else None)
    ec = eaf_col if eaf_col in work.columns else ("eaf_exp" if "eaf_exp" in work.columns else None)

    if rc is not None and ac is not None:
        alleles = work[rc].astype(str).str.upper() + work[ac].astype(str).str.upper()
        is_palindromic = alleles.isin(_PALINDROMIC).to_numpy()
        if ec is not None:
            eaf = pd.to_numeric(work[ec], errors="coerce").to_numpy(dtype=np.float64)
            is_ambiguous = np.isnan(eaf) | ((eaf > palindromic_eaf_window[0]) & (eaf < palindromic_eaf_window[1]))
        else:
            is_ambiguous = np.ones(len(work), dtype=bool)
        keep_mask &= ~(is_palindromic & is_ambiguous)

    if pval_col in work.columns:
        pvals = pd.to_numeric(work[pval_col], errors="coerce").to_numpy(dtype=np.float64)
        keep_mask &= np.isfinite(pvals) & (pvals < pval_threshold)

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
        if all(r2[pos, k] < r2_thresh for k in kept):
            kept.append(int(pos))

    return work.iloc[kept].reset_index(drop=True)
