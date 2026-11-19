"""Bundled DMRcate Gaussian kernel Satterthwaite chi^2 DMR caller (Peters et al. 2015)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .multifactor import _bh_fdr
from .smoothing import SettingsMode, validate_settings


def _gaussian_window_neighbors(
    pos_sorted: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded neighbor indices and Gaussian kernel weights within 5*sigma."""
    n = len(pos_sorted)
    if n == 0:
        return np.zeros((0, 1), dtype=np.int64), np.zeros((0, 1), dtype=np.float64)

    cutoff = max(5.0 * sigma, 1.0)
    left = np.searchsorted(pos_sorted, pos_sorted - cutoff, side="left")
    right = np.searchsorted(pos_sorted, pos_sorted + cutoff, side="right")
    max_w = int(np.max(np.maximum(right - left, 1)))

    offsets = np.arange(max_w, dtype=np.int64)[None, :]
    raw_idx = left[:, None] + offsets
    valid = raw_idx < right[:, None]
    j_idx = np.minimum(raw_idx, n - 1)

    dist = pos_sorted[j_idx] - pos_sorted[:, None]
    k_mat = np.where(valid, np.exp(-0.5 * (dist / sigma) ** 2), 0.0)
    return j_idx, k_mat


def dmrcate(
    dml_df: pd.DataFrame,
    lambda_bp: float = 1000.0,
    C: float = 2.0,
    fdr: float = 0.05,
    min_cpgs: int = 2,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Call DMRs via DMRcate Gaussian kernel smoothing and Satterthwaite chi^2 test.

    Implements Peters et al. (*Epigenetics & Chromatin* 2015): smooths squared
    CpG statistics ``Y_i = stat_i^2`` across genomic coordinates with Gaussian
    bandwidth ``sigma = lambda_bp / C``, evaluates Satterthwaite moments
    ``(K1, K2, K3)`` where ``K1_i = sum_j k_ij Y_j``, ``K2_i = sum_j k_ij``, and
    ``K3_i = sum_j k_ij^2`` so ``K1_i * K2_i / K3_i ~ chi^2(df = K2_i^2 / K3_i)``,
    applies Benjamini-Hochberg FDR thresholding, and collapses nearby significant
    CpGs within ``lambda_bp`` bp into differentially methylated regions.
    """
    validate_settings(settings)
    if lambda_bp <= 0.0 or C <= 0.0:
        raise ValueError("lambda_bp and C must both be strictly positive.")
    if not (0.0 < fdr < 1.0):
        raise ValueError(f"fdr must lie in (0, 1); got {fdr}.")
    if not {"chr", "pos", "stat"}.issubset(dml_df.columns):
        raise ValueError("dml_df must contain 'chr', 'pos', and 'stat' columns.")

    df = dml_df.copy().reset_index(drop=True)
    # Filter to finite test statistics
    valid_mask = np.isfinite(df["stat"].to_numpy(dtype=np.float64))
    df = df.loc[valid_mask].sort_values(["chr", "pos"], kind="mergesort").reset_index(drop=True)

    empty_cols = [
        "chr",
        "start",
        "end",
        "length",
        "nCG",
        "min_smoothed_fdr",
        "stouffer",
        "maxdiff",
        "meandiff",
    ]
    if df.empty:
        out_empty = pd.DataFrame(columns=empty_cols)
        out_empty.attrs["cpg_stats"] = pd.DataFrame()
        return out_empty

    sigma = float(lambda_bp) / float(C)
    n_cpgs = len(df)
    chrom_arr = df["chr"].astype(str).to_numpy()
    pos_arr = df["pos"].to_numpy(dtype=np.float64)
    stat_arr = df["stat"].to_numpy(dtype=np.float64)
    y_sq = stat_arr * stat_arr

    has_diff = "diff" in df.columns
    diff_arr = (
        df["diff"].to_numpy(dtype=np.float64)
        if has_diff
        else np.full(n_cpgs, np.nan, dtype=np.float64)
    )
    raw_pval = (
        df["pval"].to_numpy(dtype=np.float64)
        if "pval" in df.columns
        else 2.0 * norm.sf(np.abs(stat_arr))
    )

    k1_all = np.zeros(n_cpgs, dtype=np.float64)
    k2_all = np.zeros(n_cpgs, dtype=np.float64)
    k3_all = np.zeros(n_cpgs, dtype=np.float64)

    for ch in pd.unique(chrom_arr):
        idx_ch = np.flatnonzero(chrom_arr == ch)
        pos_ch = pos_arr[idx_ch]
        y_ch = y_sq[idx_ch]

        j_idx, k_mat = _gaussian_window_neighbors(pos_ch, sigma=sigma)
        k1_all[idx_ch] = np.sum(k_mat * y_ch[j_idx], axis=1)
        k2_all[idx_ch] = np.sum(k_mat, axis=1)
        k3_all[idx_ch] = np.sum(k_mat * k_mat, axis=1)

    k3_safe = np.maximum(k3_all, 1e-15)
    df_satt = (k2_all * k2_all) / k3_safe
    chi2_scaled = (k1_all * k2_all) / k3_safe
    p_smooth = chi2.sf(chi2_scaled, df=df_satt)
    fdr_smooth = _bh_fdr(p_smooth)

    cpg_stats = pd.DataFrame(
        {
            "chr": chrom_arr,
            "pos": df["pos"].to_numpy(),
            "stat": stat_arr,
            "K1": k1_all,
            "K2": k2_all,
            "K3": k3_all,
            "df_satterthwaite": df_satt,
            "smoothed_pval": p_smooth,
            "smoothed_fdr": fdr_smooth,
        }
    )

    sig_mask = np.isfinite(fdr_smooth) & (fdr_smooth < fdr)
    dmr_rows: list[dict[str, object]] = []

    for ch in pd.unique(chrom_arr):
        ch_indices = np.flatnonzero(chrom_arr == ch)
        sig_in_ch = ch_indices[sig_mask[ch_indices]]
        if len(sig_in_ch) == 0:
            continue

        # Cluster consecutive significant CpGs separated by <= lambda_bp
        clusters: list[tuple[int, int]] = []
        c_start = int(sig_in_ch[0])
        c_prev = int(sig_in_ch[0])
        for idx_curr in sig_in_ch[1:]:
            idx_i = int(idx_curr)
            if (pos_arr[idx_i] - pos_arr[c_prev]) <= lambda_bp:
                c_prev = idx_i
            else:
                clusters.append((c_start, c_prev))
                c_start = idx_i
                c_prev = idx_i
        clusters.append((c_start, c_prev))

        for i_lo, i_hi in clusters:
            n_cg = i_hi - i_lo + 1
            if n_cg < int(min_cpgs):
                continue
            start_pos = int(df["pos"].iloc[i_lo])
            end_pos = int(df["pos"].iloc[i_hi])
            min_fdr_reg = float(np.min(fdr_smooth[i_lo : i_hi + 1]))

            # Stouffer combined Z-test across individual CpGs in region
            p_sub = np.clip(raw_pval[i_lo : i_hi + 1], 1e-300, 1.0 - 1e-15)
            z_unsigned = norm.isf(p_sub)
            stouffer_z = float(np.sum(z_unsigned) / np.sqrt(float(n_cg)))
            stouffer_p = float(norm.sf(stouffer_z))

            if has_diff and np.any(np.isfinite(diff_arr[i_lo : i_hi + 1])):
                d_sub = diff_arr[i_lo : i_hi + 1]
                d_fin = d_sub[np.isfinite(d_sub)]
                max_idx = int(np.argmax(np.abs(d_fin)))
                maxdiff = float(d_fin[max_idx])
                meandiff = float(np.mean(d_fin))
            else:
                maxdiff = float("nan")
                meandiff = float("nan")

            dmr_rows.append(
                {
                    "chr": str(ch),
                    "start": start_pos,
                    "end": end_pos,
                    "length": int(end_pos - start_pos),
                    "nCG": int(n_cg),
                    "min_smoothed_fdr": min_fdr_reg,
                    "stouffer": stouffer_p,
                    "maxdiff": maxdiff,
                    "meandiff": meandiff,
                }
            )

    if not dmr_rows:
        out_df = pd.DataFrame(columns=empty_cols)
    else:
        out_df = (
            pd.DataFrame(dmr_rows)
            .sort_values(["min_smoothed_fdr", "stouffer"], ascending=[True, True])
            .reset_index(drop=True)
        )
    out_df.attrs["cpg_stats"] = cpg_stats
    return out_df
