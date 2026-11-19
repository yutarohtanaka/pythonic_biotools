"""
Evolutionary copy-number gain timing (`pi_to_time`, `time_cn_gains`) and
Whole-Genome Doubling (`estimate_wgd_timing`) rate equations from
Gerstung et al., Nature 2020 (10.1038/s41586-019-1907-7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .io import (
    SettingsMode,
    canonical_chrom,
    validate_and_prepare_segments,
    validate_settings,
)


def pi_to_all_times(
    major_cn: int,
    minor_cn: int,
    pi_vec: Sequence[float] | np.ndarray,
    settings: SettingsMode = "default",
) -> list[float]:
    """
    Convert clonal multiplicity mixture proportions `(pi_1, ..., pi_{major_cn})`
    into ordered molecular gain times `[T_1, T_2, ...]` in `[0, 1]` using the
    lineage-conservation rate equations of Gerstung et al. (Nature 2020).

    Parameters
    ----------
    major_cn : int
        Major allele copy number (`>= 0`). Informative timing requires `major_cn >= 2`.
    minor_cn : int
        Minor allele copy number (`0 <= minor_cn <= major_cn`).
    pi_vec : Sequence[float] | np.ndarray
        Mixture proportions where `pi_vec[m - 1]` corresponds to clonal
        multiplicity `m` for `m = 1, ..., major_cn`. Any trailing subclonal
        entries (`index >= major_cn`) are ignored because timing depends on the
        relative proportions of clonal multiplicities.
    settings : {"default", "R_default"}, default="default"
        Parameterization mode (`"default"` or `"R_default"`).

    Returns
    -------
    list[float]
        Ordered molecular gain times in `[0.0, 1.0]`. Empty if `major_cn < 2`
        or clonal probability mass is zero.
    """
    validate_settings(settings)
    maj = int(major_cn)
    mn = int(minor_cn)
    if maj < mn:
        maj, mn = mn, maj
    if maj < 2 or mn < 0:
        return []

    arr = np.asarray(pi_vec, dtype=np.float64).ravel()
    if arr.size < maj:
        padded = np.zeros(maj, dtype=np.float64)
        padded[: arr.size] = arr
        arr = padded

    pi_c = np.maximum(arr[:maj], 0.0)
    if not np.all(np.isfinite(pi_c)) or float(np.sum(pi_c)) <= 0.0:
        return []

    # Exact closed-form two-multiplicity states
    if maj == 2 and mn == 1:
        # Trisomy 2:1 -> T = 3 * pi2 / (2 * pi2 + pi1)
        denom = 2.0 * pi_c[1] + pi_c[0]
        if denom <= 0.0:
            return []
        return [float(np.clip(3.0 * pi_c[1] / denom, 0.0, 1.0))]

    if maj == 2 and mn == 0:
        # CN-LOH 2:0 -> T = 2 * pi2 / (2 * pi2 + pi1)
        denom = 2.0 * pi_c[1] + pi_c[0]
        if denom <= 0.0:
            return []
        return [float(np.clip(2.0 * pi_c[1] / denom, 0.0, 1.0))]

    if maj == 2 and mn == 2:
        # Balanced tetrasomy / WGD 2:2 -> mu2 = 2*T, mu1 = 4*(1 - T)
        # => T = pi2 / (pi2 + 0.5 * pi1) = 2 * pi2 / (2 * pi2 + pi1)
        denom = 2.0 * pi_c[1] + pi_c[0]
        if denom <= 0.0:
            return []
        return [float(np.clip(2.0 * pi_c[1] / denom, 0.0, 1.0))]

    # Universal lineage-descendant conservation: sum_{m=1}^{maj} m * mu_m = C_tot
    c_tot = float(maj + mn)
    mult = np.arange(1, maj + 1, dtype=np.float64)
    denom = float(np.sum(mult * pi_c))
    if denom <= 0.0:
        return []

    if maj == 3 and mn == 0:
        t1 = float(np.clip(3.0 * pi_c[2] / denom, 0.0, 1.0))
        t2 = float(np.clip(3.0 * (pi_c[1] + pi_c[2]) / denom, t1, 1.0))
        return [t1, t2]

    if maj == 3 and mn == 1:
        t1 = float(np.clip(4.0 * pi_c[2] / denom, 0.0, 1.0))
        t2 = float(np.clip(4.0 * (pi_c[1] + pi_c[2]) / denom, t1, 1.0))
        return [t1, t2]

    if maj == 4 and mn == 0:
        t1 = float(np.clip(4.0 * pi_c[3] / denom, 0.0, 1.0))
        if pi_c[2] <= 0.05 * (pi_c[1] + pi_c[3] + 1e-12):
            # Balanced doubling history 1:0 -> 2:0 -> 4:0 (mu4 = T1, mu2 = 2*(T2 - T1))
            t2 = float(np.clip((4.0 * pi_c[3] + 2.0 * pi_c[1]) / denom, t1, 1.0))
            return [t1, t2]
        t2 = float(np.clip(4.0 * (pi_c[2] + pi_c[3]) / denom, t1, 1.0))
        t3 = float(np.clip(4.0 * (pi_c[1] + pi_c[2] + pi_c[3]) / denom, t2, 1.0))
        return [t1, t2, t3]

    if maj == mn:
        # Symmetric polyploid history (both parental alleles gain in parallel)
        times: list[float] = []
        cum = 0.0
        for m in range(maj, 1, -1):
            cum += 0.5 * pi_c[m - 1]
            t_ev = float(np.clip(c_tot * cum / denom, 0.0, 1.0))
            if times and t_ev < times[-1]:
                t_ev = times[-1]
            times.append(t_ev)
        return times

    # General asymmetric gain (maj > mn)
    times = []
    cum = 0.0
    for m in range(maj, 1, -1):
        allele_factor = 0.5 if m <= mn else 1.0
        cum += allele_factor * pi_c[m - 1]
        t_ev = float(np.clip(c_tot * cum / denom, 0.0, 1.0))
        if times and t_ev < times[-1]:
            t_ev = times[-1]
        times.append(t_ev)
    return times


def pi_to_time(
    major_cn: int,
    minor_cn: int,
    pi_vec: Sequence[float] | np.ndarray,
    settings: SettingsMode = "default",
) -> float:
    """
    Convert clonal multiplicity proportions `(pi_1, pi_2, ...)` on a gained
    copy-number segment into primary molecular gain time `T in [0, 1]`
    (`np.nan` if `major_cn < 2`).
    """
    validate_settings(settings)
    times = pi_to_all_times(major_cn, minor_cn, pi_vec, settings=settings)
    if not times:
        return float("nan")
    return times[0]


def time_cn_gains(
    cn_segments_df: pd.DataFrame,
    segment_pis: Mapping[Any, Sequence[float] | np.ndarray] | Sequence[Sequence[float] | np.ndarray],
    n_boot: int = 200,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Compute molecular gain times `T` and bootstrap 95% confidence intervals
    (`T_lo`, `T_hi`) across allele-specific copy-number segments.

    Parameters
    ----------
    cn_segments_df : pd.DataFrame
        Segments table with `chrom`, `start`, `end`, `major_cn`, `minor_cn`
        (and optional `n_mutations` for parametric bootstrap concentration).
    segment_pis : Mapping or Sequence
        Fitted state mixture vectors per segment, indexed either by segment
        position `0..N-1`, `segment_id`, or aligned as a sequence.
    n_boot : int, default=200
        Number of bootstrap replicates for 95% confidence intervals.
    seed : int, default=42
        Random seed for reproducible bootstrap sampling.
    settings : {"default", "R_default"}, default="default"
        Inference settings mode.

    Returns
    -------
    pd.DataFrame
        Timed segments DataFrame with `T`, `T_lo`, `T_hi`, `T_second`, and
        state proportion summaries (`pi_clonal_early`, `pi_clonal_late`,
        `pi_subclonal`).
    """
    validate_settings(settings)
    if n_boot < 0:
        raise ValueError(f"n_boot must be >= 0, got {n_boot!r}.")

    segs = validate_and_prepare_segments(cn_segments_df)
    rng = np.random.default_rng(seed)

    is_mapping = isinstance(segment_pis, Mapping)
    seq_pis = None if is_mapping else list(segment_pis)
    if seq_pis is not None and len(seq_pis) != len(segs):
        raise ValueError(
            f"Length of segment_pis ({len(seq_pis)}) must match cn_segments_df ({len(segs)})."
        )

    has_n_mut = "n_mutations" in segs.columns
    t_vals = np.full(len(segs), np.nan, dtype=np.float64)
    t_lo_vals = np.full(len(segs), np.nan, dtype=np.float64)
    t_hi_vals = np.full(len(segs), np.nan, dtype=np.float64)
    t_sec_vals = np.full(len(segs), np.nan, dtype=np.float64)
    pi_early = np.full(len(segs), np.nan, dtype=np.float64)
    pi_late = np.full(len(segs), np.nan, dtype=np.float64)
    pi_sub = np.full(len(segs), np.nan, dtype=np.float64)

    for idx, row in segs.iterrows():
        i = int(idx)  # type: ignore[arg-type]
        seg_id = row["segment_id"]
        maj = int(row["major_cn"])
        mn = int(row["minor_cn"])

        raw_pi: Sequence[float] | np.ndarray | None = None
        if is_mapping:
            assert isinstance(segment_pis, Mapping)
            if i in segment_pis:
                raw_pi = segment_pis[i]
            elif seg_id in segment_pis:
                raw_pi = segment_pis[seg_id]
        else:
            assert seq_pis is not None
            raw_pi = seq_pis[i]

        if raw_pi is None or maj < 1:
            continue

        pi_arr = np.asarray(raw_pi, dtype=np.float64).ravel()
        if pi_arr.size == 0 or np.sum(pi_arr) <= 0.0:
            continue
        pi_arr = np.maximum(pi_arr, 0.0)
        pi_arr = pi_arr / np.sum(pi_arr)

        c_slice = pi_arr[: min(maj, pi_arr.size)]
        pi_late[i] = float(c_slice[0]) if c_slice.size >= 1 else 0.0
        pi_early[i] = float(np.sum(c_slice[1:])) if c_slice.size >= 2 else 0.0
        pi_sub[i] = float(np.sum(pi_arr[maj:])) if pi_arr.size > maj else 0.0

        if maj < 2:
            continue

        all_t = pi_to_all_times(maj, mn, pi_arr, settings=settings)
        if not all_t:
            continue
        t_vals[i] = all_t[0]
        if len(all_t) >= 2:
            t_sec_vals[i] = all_t[1]

        n_eff = int(row["n_mutations"]) if has_n_mut and pd.notna(row["n_mutations"]) else 60
        if n_eff <= 0:
            t_vals[i] = np.nan
            t_sec_vals[i] = np.nan
            continue

        if n_boot == 0:
            t_lo_vals[i] = t_vals[i]
            t_hi_vals[i] = t_vals[i]
            continue

        # Multinomial-Dirichlet bootstrap over observed state mixture counts
        counts_boot = rng.multinomial(max(n_eff, 1), pi_arr, size=n_boot).astype(np.float64)
        prior_add = 0.05 if settings == "default" else (1.0 / max(pi_arr.size, 1)) * 0.05
        pis_boot = counts_boot + prior_add
        pis_boot /= np.sum(pis_boot, axis=1, keepdims=True)

        boot_times = np.array(
            [pi_to_time(maj, mn, pis_boot[b], settings=settings) for b in range(n_boot)],
            dtype=np.float64,
        )
        valid_bt = boot_times[np.isfinite(boot_times)]
        if valid_bt.size > 0:
            lo, hi = np.percentile(valid_bt, [2.5, 97.5])
            t_lo_vals[i] = float(min(lo, t_vals[i]))
            t_hi_vals[i] = float(max(hi, t_vals[i]))
        else:
            t_lo_vals[i] = t_vals[i]
            t_hi_vals[i] = t_vals[i]

    out = segs.drop(columns=["chrom_canon"]).copy()
    out["pi_clonal_early"] = pi_early
    out["pi_clonal_late"] = pi_late
    out["pi_subclonal"] = pi_sub
    out["T"] = t_vals
    out["T_lo"] = t_lo_vals
    out["T_hi"] = t_hi_vals
    out["T_second"] = t_sec_vals
    return out


@dataclass
class WGDTimingResult:
    """Whole-Genome Doubling status and synchronous timing summary."""

    is_wgd: bool
    wgd_time: float
    wgd_time_lo: float
    wgd_time_hi: float
    is_synchronous: bool
    homogeneity_pval: float
    inter_segment_sd: float
    fraction_major_ge2: float
    mean_ploidy: float
    n_wgd_segments: int
    timed_wgd_segments_df: pd.DataFrame
    settings: SettingsMode = "default"

    def summary(self) -> dict[str, Any]:
        """Return dictionary summary of WGD status and timing."""
        return {
            "is_wgd": self.is_wgd,
            "wgd_time": self.wgd_time,
            "wgd_time_lo": self.wgd_time_lo,
            "wgd_time_hi": self.wgd_time_hi,
            "is_synchronous": self.is_synchronous,
            "homogeneity_pval": self.homogeneity_pval,
            "inter_segment_sd": self.inter_segment_sd,
            "fraction_major_ge2": self.fraction_major_ge2,
            "mean_ploidy": self.mean_ploidy,
            "n_wgd_segments": self.n_wgd_segments,
            "settings": self.settings,
        }


def estimate_wgd_timing(
    timed_segments_df: pd.DataFrame,
    settings: SettingsMode = "default",
) -> WGDTimingResult:
    """
    Evaluate Whole-Genome Doubling (`WGD`) status and estimate synchronous
    molecular WGD timing across `2:2` and `2:0` gained segments following
    PCAWG / Gerstung et al. (Nature 2020).

    Parameters
    ----------
    timed_segments_df : pd.DataFrame
        Output of `time_cn_gains` or `mutation_time(...).segments_df` containing
        `chrom`, `start`, `end`, `major_cn`, `minor_cn`, `T` (and optional
        `T_lo`, `T_hi`, `n_mutations`).
    settings : {"default", "R_default"}, default="default"
        Inference settings mode.

    Returns
    -------
    WGDTimingResult
        WGD classification, inverse-variance weighted synchronous WGD time,
        95% confidence interval, and Cochran's Q synchronicity test.
    """
    validate_settings(settings)
    if not isinstance(timed_segments_df, pd.DataFrame) or timed_segments_df.empty:
        raise ValueError("timed_segments_df must be a non-empty pandas DataFrame.")

    for col in ("start", "end", "major_cn", "minor_cn", "T"):
        if col not in timed_segments_df.columns:
            raise ValueError(f"timed_segments_df is missing required column {col!r}.")

    df = timed_segments_df.copy()
    if "chrom" in df.columns:
        chroms = np.array([canonical_chrom(c) for c in df["chrom"]])
        autosomal = ~np.isin(chroms, ["X", "Y", "MT"])
        if not np.any(autosomal):
            autosomal = np.ones(len(df), dtype=bool)
    else:
        autosomal = np.ones(len(df), dtype=bool)

    spans = np.maximum(
        df["end"].astype(np.float64).to_numpy() - df["start"].astype(np.float64).to_numpy() + 1.0,
        1.0,
    )
    maj = df["major_cn"].astype(np.int64).to_numpy()
    mn = df["minor_cn"].astype(np.int64).to_numpy()
    tot_cn = (maj + mn).astype(np.float64)

    auto_spans = spans[autosomal]
    total_span = float(np.sum(auto_spans))
    frac_maj_ge2 = float(np.sum(auto_spans[maj[autosomal] >= 2]) / max(total_span, 1.0))
    frac_hom_del = float(np.sum(auto_spans[tot_cn[autosomal] == 0]) / max(total_span, 1.0))
    mean_ploidy = float(np.sum(auto_spans * tot_cn[autosomal]) / max(total_span, 1.0))

    # Candidate WGD-informative segments: 2:2 and 2:0 (single-step tetraploid / CN-LOH states)
    t_arr = df["T"].astype(np.float64).to_numpy()
    wgd_state_mask = autosomal & (maj == 2) & np.isin(mn, [0, 2]) & np.isfinite(t_arr)
    if "n_mutations" in df.columns:
        min_muts = 3 if settings == "default" else 5
        n_mut_arr = df["n_mutations"].fillna(0).astype(np.int64).to_numpy()
        if np.any(wgd_state_mask & (n_mut_arr >= min_muts)):
            wgd_state_mask = wgd_state_mask & (n_mut_arr >= min_muts)

    wgd_segs = df.loc[wgd_state_mask].copy().reset_index(drop=True)
    n_wgd_segs = len(wgd_segs)

    # PCAWG / Bielski et al. ploidy-based WGD criterion:
    # 2.9 - 2 * frac_hom_del <= mean_ploidy OR >= 50% of autosomal genome has major_cn >= 2
    ploidy_wgd = (frac_maj_ge2 >= 0.50) or (mean_ploidy >= 2.9 - 2.0 * frac_hom_del)

    if n_wgd_segs == 0:
        return WGDTimingResult(
            is_wgd=bool(ploidy_wgd),
            wgd_time=float("nan"),
            wgd_time_lo=float("nan"),
            wgd_time_hi=float("nan"),
            is_synchronous=False,
            homogeneity_pval=float("nan"),
            inter_segment_sd=float("nan"),
            fraction_major_ge2=frac_maj_ge2,
            mean_ploidy=mean_ploidy,
            n_wgd_segments=0,
            timed_wgd_segments_df=wgd_segs,
            settings=settings,
        )

    seg_t = wgd_segs["T"].astype(np.float64).to_numpy()
    if "T_lo" in wgd_segs.columns and "T_hi" in wgd_segs.columns:
        ci_w = (
            wgd_segs["T_hi"].astype(np.float64).to_numpy()
            - wgd_segs["T_lo"].astype(np.float64).to_numpy()
        )
        seg_se = np.where(np.isfinite(ci_w) & (ci_w > 0), np.maximum(ci_w / 3.92, 0.04), 0.10)
    else:
        seg_se = np.full(n_wgd_segs, 0.10, dtype=np.float64)

    inv_var = 1.0 / (seg_se ** 2)
    w_sum = float(np.sum(inv_var))
    wgd_t = float(np.clip(np.sum(inv_var * seg_t) / w_sum, 0.0, 1.0))
    pooled_se = float(np.sqrt(1.0 / w_sum))

    if n_wgd_segs >= 2:
        q_stat = float(np.sum(inv_var * ((seg_t - wgd_t) ** 2)))
        df_q = n_wgd_segs - 1
        homog_pval = float(stats.chi2.sf(q_stat, df=df_q))
        # DerSimonian-Laird random-effects variance inflation if heterogeneous
        c_denom = w_sum - float(np.sum(inv_var ** 2)) / w_sum
        tau2 = max(0.0, (q_stat - df_q) / max(c_denom, 1e-12))
        re_weights = 1.0 / (seg_se ** 2 + tau2)
        re_sum = float(np.sum(re_weights))
        wgd_t = float(np.clip(np.sum(re_weights * seg_t) / re_sum, 0.0, 1.0))
        pooled_se = float(np.sqrt(1.0 / re_sum))
        inter_sd = float(np.sqrt(np.average((seg_t - wgd_t) ** 2, weights=inv_var)))
        sync_tol = 0.16 if settings == "default" else 0.15
        is_sync = bool((homog_pval >= 0.01) or (inter_sd <= sync_tol))
    else:
        homog_pval = 1.0
        inter_sd = 0.0
        is_sync = True

    wgd_lo = float(np.clip(wgd_t - 1.96 * pooled_se, 0.0, wgd_t))
    wgd_hi = float(np.clip(wgd_t + 1.96 * pooled_se, wgd_t, 1.0))
    is_wgd = bool(ploidy_wgd and (is_sync or frac_maj_ge2 >= 0.60))

    return WGDTimingResult(
        is_wgd=is_wgd,
        wgd_time=wgd_t if is_wgd else wgd_t,
        wgd_time_lo=wgd_lo,
        wgd_time_hi=wgd_hi,
        is_synchronous=is_sync,
        homogeneity_pval=homog_pval,
        inter_segment_sd=inter_sd,
        fraction_major_ge2=frac_maj_ge2,
        mean_ploidy=mean_ploidy,
        n_wgd_segments=n_wgd_segs,
        timed_wgd_segments_df=wgd_segs,
        settings=settings,
    )
