"""
Cancer cell fraction (CCF) estimation via Beta-Binomial multiplicity scoring
and evolutionary pseudotime binning for TrackSig / TrackSigFreq.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

SettingsMode = Literal["default", "R_default"]

COSMIC_96_CONTEXTS: tuple[str, ...] = tuple(
    f"{five}[{sub}]{three}"
    for sub in ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
    for five in ("A", "C", "G", "T")
    for three in ("A", "C", "G", "T")
)


def validate_settings(settings: str) -> SettingsMode:
    """Validate the mandatory ``settings`` mode parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def _betabinom_logpmf(
    k: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    concentration: float = 200.0,
) -> np.ndarray:
    """Vectorized Beta-Binomial log-PMF parameterized by mean ``p`` and ``concentration``."""
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    alpha = concentration * p_clipped
    beta = concentration * (1.0 - p_clipped)
    log_comb = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    return log_comb + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)


def compute_ccf_pseudotime(
    mutations_df: pd.DataFrame,
    purity: float = 1.0,
    bin_size: int = 100,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Compute per-SNV cancer cell fraction (CCF) and group SNVs into pseudotime bins.

    SNVs are scored across somatic multiplicities ``m in {1, ..., max(1, M_i)}``
    via a Beta-Binomial likelihood given tumor purity ``rho``, variant/reference
    read counts, total copy number, and major copy number (or direct ``ccf`` /
    ``vaf`` columns), sorted in decreasing CCF order (clonal -> subclonal
    evolutionary timeline), and partitioned into ``T`` pseudotime bins.

    Parameters
    ----------
    mutations_df : pd.DataFrame
        Mutation table containing either ``alt_count`` with ``total_depth`` / ``depth``
        or ``ref_count`` (plus optional ``total_cn``, ``major_cn``, ``normal_cn``),
        or a direct ``ccf`` or ``vaf`` column.
    purity : float, default 1.0
        Tumor cellularity ``rho in (0, 1]``.
    bin_size : int, default 100
        Target number of SNVs per evolutionary pseudotime bin.
    settings : {"default", "R_default"}, default "default"
        Pipeline configuration mode. ``"default"`` applies an adaptive ``min_bin_size=25``
        fallback when ``< 200`` mutations are provided so WES cohorts work out of
        the box; ``"R_default"`` enforces strict ``bin_size`` parity (raising
        ``ValueError`` if ``len(mutations_df) < bin_size``).

    Returns
    -------
    pd.DataFrame
        Copy of ``mutations_df`` sorted in decreasing CCF order with added/standardized
        columns ``vaf``, ``multiplicity``, ``ccf``, ``ccf_sd``, ``ccf_cont``,
        ``pseudotime_rank``, and 1-based ``bin_id``.
    """
    validate_settings(settings)
    if not isinstance(mutations_df, pd.DataFrame):
        raise TypeError("mutations_df must be a pandas DataFrame.")
    if not (0.0 < float(purity) <= 1.0):
        raise ValueError(f"purity must be in (0, 1], got {purity}.")
    if int(bin_size) < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}.")

    n_mut = len(mutations_df)
    if n_mut == 0:
        raise ValueError("mutations_df must contain at least 1 mutation.")

    bin_size = int(bin_size)
    if settings == "R_default" and n_mut < bin_size:
        raise ValueError(
            f"settings='R_default' requires at least bin_size={bin_size} SNVs "
            f"(got {n_mut}); override bin_size explicitly or use settings='default'."
        )

    if settings == "default" and n_mut < 200 and bin_size > 25:
        eff_bin_size = min(bin_size, max(25, n_mut // 4)) if n_mut >= 25 else n_mut
    else:
        eff_bin_size = min(bin_size, n_mut)

    df = mutations_df.copy()
    rho = float(purity)

    if "total_cn" in df.columns:
        total_cn = np.maximum(df["total_cn"].to_numpy(dtype=float), 1e-3)
    else:
        total_cn = np.full(n_mut, 2.0, dtype=float)

    if "major_cn" in df.columns:
        major_cn = np.maximum(1, np.rint(df["major_cn"].to_numpy(dtype=float)).astype(int))
    else:
        major_cn = np.maximum(1, np.ceil(total_cn / 2.0).astype(int))

    if "normal_cn" in df.columns:
        normal_cn = np.maximum(df["normal_cn"].to_numpy(dtype=float), 0.0)
    else:
        normal_cn = np.full(n_mut, 2.0, dtype=float)

    eff_cn = rho * total_cn + (1.0 - rho) * normal_cn
    concentration = 200.0

    has_counts = "alt_count" in df.columns and (
        "total_depth" in df.columns or "depth" in df.columns or "ref_count" in df.columns
    )

    if has_counts:
        alt = np.maximum(df["alt_count"].to_numpy(dtype=float), 0.0)
        if "total_depth" in df.columns:
            depth = np.maximum(df["total_depth"].to_numpy(dtype=float), alt)
        elif "depth" in df.columns:
            depth = np.maximum(df["depth"].to_numpy(dtype=float), alt)
        else:
            ref = np.maximum(df["ref_count"].to_numpy(dtype=float), 0.0)
            depth = alt + ref
        depth = np.maximum(depth, 1.0)
        vaf = alt / depth

        # Multiplicity selection via Beta-Binomial likelihood over m in {1, ..., M_i}
        max_m = int(np.max(major_cn))
        multiplicity = np.ones(n_mut, dtype=int)
        raw_phi_m1 = vaf * eff_cn / rho

        if max_m > 1:
            best_ll = np.full(n_mut, -np.inf, dtype=float)
            for m in range(1, max_m + 1):
                eligible = (major_cn >= m) & (raw_phi_m1 > 1.0)
                if not np.any(eligible):
                    continue
                p_clonal = (rho * m) / eff_cn[eligible]
                ll_m = _betabinom_logpmf(
                    alt[eligible], depth[eligible], p_clonal, concentration=concentration
                )
                better = ll_m > best_ll[eligible]
                idx_elig = np.flatnonzero(eligible)
                multiplicity[idx_elig[better]] = m
                best_ll[idx_elig[better]] = ll_m[better]

        # Unbiased somatic CCF estimate given Beta-Binomial MLE multiplicity m_i
        scale_factor = (rho * multiplicity) / eff_cn  # expected VAF per unit CCF
        ccf = np.clip(vaf / scale_factor, 1e-4, 2.5)
        p_hat = np.clip(vaf, 0.02, 0.98)
        ccf_sd = np.maximum((1.0 / scale_factor) * np.sqrt(p_hat * (1.0 - p_hat) / depth), 0.025)
        ccf_kappa = (1.0 / scale_factor) / np.sqrt(depth)

        # Continuity-corrected CCF to break exact discrete read-count ties cleanly
        order_raw = np.argsort(-ccf, kind="mergesort")
        ccf_cont = ccf.copy()
        quant_key = np.round(ccf, 8)
        _, inv_idx, counts = np.unique(quant_key, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            for u_id in np.flatnonzero(counts > 1):
                members = order_raw[inv_idx[order_raw] == u_id]
                k_tie = len(members)
                half_step = 0.5 / np.maximum(depth[members] * scale_factor[members], 2.0)
                offsets = (0.5 - (np.arange(k_tie, dtype=float) + 0.5) / k_tie) * (2.0 * half_step)
                ccf_cont[members] = np.clip(ccf[members] + offsets, 1e-4, 2.5)
    elif "ccf" in df.columns:
        ccf = np.maximum(df["ccf"].to_numpy(dtype=float), 1e-4)
        multiplicity = np.ones(n_mut, dtype=int)
        scale_factor = (rho * multiplicity) / eff_cn
        vaf = (
            df["vaf"].to_numpy(dtype=float)
            if "vaf" in df.columns
            else np.clip(ccf * scale_factor, 0.0, 1.0)
        )
        eff_depth = (
            np.maximum(df["total_depth"].to_numpy(dtype=float), 10.0)
            if "total_depth" in df.columns
            else np.full(n_mut, 100.0, dtype=float)
        )
        p_hat = np.clip(vaf, 0.02, 0.98)
        ccf_sd = np.maximum((1.0 / scale_factor) * np.sqrt(p_hat * (1.0 - p_hat) / eff_depth), 0.025)
        ccf_kappa = (1.0 / scale_factor) / np.sqrt(eff_depth)
        ccf_cont = ccf.copy()
        # Break exact ties smoothly across empirical sampling cell if synthetic constant CCFs are passed
        quant_key = np.round(ccf, 8)
        _, inv_idx, counts = np.unique(quant_key, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            from scipy.special import ndtri

            for u_id in np.flatnonzero(counts > 1):
                members = np.flatnonzero(inv_idx == u_id)
                k_tie = len(members)
                if k_tie >= 5:
                    probs = np.linspace(1.0 - 0.5 / k_tie, 0.5 / k_tie, k_tie)
                    ccf_cont[members] = np.clip(ccf[members] + ccf_sd[members] * ndtri(probs), 1e-4, 2.5)
                else:
                    offsets = np.linspace(1e-3, -1e-3, k_tie)
                    ccf_cont[members] = np.clip(ccf[members] + offsets, 1e-4, 2.5)
    elif "vaf" in df.columns:
        vaf = np.clip(df["vaf"].to_numpy(dtype=float), 1e-5, 1.0 - 1e-5)
        eff_depth = np.full(n_mut, 100.0, dtype=float)
        alt_syn = np.rint(vaf * eff_depth)
        synth_df = df.copy()
        synth_df["alt_count"] = alt_syn
        synth_df["total_depth"] = eff_depth
        return compute_ccf_pseudotime(
            synth_df, purity=purity, bin_size=bin_size, settings=settings
        )
    else:
        raise ValueError(
            "mutations_df must contain ('alt_count' and 'total_depth'/'depth'/'ref_count'), "
            "'ccf', or 'vaf'."
        )

    df["vaf"] = vaf
    df["multiplicity"] = multiplicity
    df["ccf"] = ccf
    df["ccf_sd"] = ccf_sd
    df["ccf_cont"] = ccf_cont
    df.attrs["mean_scale_factor"] = float(np.mean(scale_factor))
    df.attrs["rms_kappa"] = float(np.sqrt(np.mean(ccf_kappa * ccf_kappa)))

    # Decreasing CCF order: clonal (high CCF) -> subclonal (low CCF)
    sort_idx = np.argsort(-ccf_cont, kind="mergesort")
    df = df.iloc[sort_idx].reset_index(drop=True)
    df["pseudotime_rank"] = np.arange(1, n_mut + 1, dtype=int)

    n_bins = max(1, n_mut // eff_bin_size)
    bin_splits = np.array_split(np.arange(n_mut), n_bins)
    bin_ids = np.empty(n_mut, dtype=int)
    for b_idx, rows in enumerate(bin_splits, start=1):
        bin_ids[rows] = b_idx
    df["bin_id"] = bin_ids
    df.attrs["effective_bin_size"] = eff_bin_size
    df.attrs["n_bins"] = n_bins
    return df
