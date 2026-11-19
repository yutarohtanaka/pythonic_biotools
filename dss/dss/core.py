"""Core Beta-Binomial DML and DMR calling functions for DSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .dispersion import shrink_dispersion_bb
from .multifactor import _bh_fdr
from .smoothing import (
    SettingsMode,
    compute_group_mean_and_var,
    smooth_counts_tricube,
    validate_settings,
)


@dataclass
class DSSConfig:
    """Mode-aware default configuration for DSS DML and DMR calling."""

    settings: SettingsMode = "default"
    smoothing: bool | None = None
    smoothing_span: float = 500.0
    p_threshold_dml: float = 1e-5
    p_threshold_dmr: float | None = None
    delta_dml: float = 0.1
    delta_dmr: float = 0.0
    minlen: int = 50
    minCG: int = 3
    dis_merge: int = 100
    pct_sig: float = 0.5

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if self.smoothing is None:
            self.smoothing = self.settings == "default"
        if self.p_threshold_dmr is None:
            self.p_threshold_dmr = 1e-4 if self.settings == "default" else 1e-5
        if self.settings == "R_default" and self.dis_merge > self.minlen:
            self.dis_merge = self.minlen


@dataclass
class BSData:
    """Aligned bisulfite / EM-seq count matrices across CpG loci and samples."""

    coords: pd.DataFrame
    N: np.ndarray
    X: np.ndarray
    sample_names: list[str]


def make_bsseq_data(
    frames: Sequence[pd.DataFrame],
    sample_names: Sequence[str] | None = None,
) -> BSData:
    """Align per-replicate DataFrames ``('chr', 'pos', 'N', 'X')`` into ``BSData``."""
    if len(frames) == 0:
        raise ValueError("frames must contain at least one replicate DataFrame.")
    names = (
        list(map(str, sample_names))
        if sample_names is not None
        else [f"S{i + 1}" for i in range(len(frames))]
    )
    if len(names) != len(frames):
        raise ValueError("Length of sample_names must match length of frames.")

    coord_parts: list[pd.DataFrame] = []
    cleaned_frames: list[pd.DataFrame] = []
    for idx, df in enumerate(frames):
        if not {"chr", "pos", "N", "X"}.issubset(df.columns):
            raise ValueError(
                f"Replicate {idx} is missing required columns ('chr', 'pos', 'N', 'X')."
            )
        sub = df[["chr", "pos", "N", "X"]].copy()
        sub["chr"] = sub["chr"].astype(str)
        sub["pos"] = sub["pos"].astype(np.int64)
        # Aggregate any duplicate (chr, pos) rows within a single sample
        if sub.duplicated(subset=["chr", "pos"]).any():
            sub = sub.groupby(["chr", "pos"], as_index=False, sort=False)[["N", "X"]].sum()
        coord_parts.append(sub[["chr", "pos"]])
        cleaned_frames.append(sub)

    all_coords = (
        pd.concat(coord_parts, ignore_index=True)
        .drop_duplicates(subset=["chr", "pos"])
        .sort_values(["chr", "pos"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_cpgs = len(all_coords)
    n_reps = len(frames)
    n_mat = np.zeros((n_cpgs, n_reps), dtype=np.float64)
    x_mat = np.zeros((n_cpgs, n_reps), dtype=np.float64)

    key_index = pd.MultiIndex.from_frame(all_coords[["chr", "pos"]])
    for r, sub in enumerate(cleaned_frames):
        sub_idx = pd.MultiIndex.from_frame(sub[["chr", "pos"]])
        locs = key_index.get_indexer(sub_idx)
        n_mat[locs, r] = sub["N"].to_numpy(dtype=np.float64)
        x_mat[locs, r] = sub["X"].to_numpy(dtype=np.float64)

    return BSData(coords=all_coords, N=n_mat, X=x_mat, sample_names=names)


def _coerce_two_groups(
    group1: pd.DataFrame | Sequence[pd.DataFrame] | tuple[object, ...] | BSData,
    group2: pd.DataFrame | Sequence[pd.DataFrame] | tuple[object, ...] | BSData,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize ``(group1, group2)`` into aligned ``(coords, N1, X1, N2, X2)``."""
    # Case 1: DataFrame or sequence of DataFrames
    if isinstance(group1, pd.DataFrame):
        group1 = [group1]
    if isinstance(group2, pd.DataFrame):
        group2 = [group2]

    if (
        isinstance(group1, Sequence)
        and not isinstance(group1, tuple)
        and len(group1) > 0
        and isinstance(group1[0], pd.DataFrame)
    ) or (
        isinstance(group1, tuple)
        and len(group1) > 0
        and isinstance(group1[0], pd.DataFrame)
    ):
        g1_list = list(group1)  # type: ignore[arg-type]
        g2_list = list(group2)  # type: ignore[arg-type]
        bs = make_bsseq_data(g1_list + g2_list)
        r1 = len(g1_list)
        return (
            bs.coords,
            bs.N[:, :r1],
            bs.X[:, :r1],
            bs.N[:, r1:],
            bs.X[:, r1:],
        )

    # Case 2: BSData objects
    if isinstance(group1, BSData) and isinstance(group2, BSData):
        frames1 = [
            pd.DataFrame(
                {
                    "chr": group1.coords["chr"],
                    "pos": group1.coords["pos"],
                    "N": group1.N[:, r],
                    "X": group1.X[:, r],
                }
            )
            for r in range(group1.N.shape[1])
        ]
        frames2 = [
            pd.DataFrame(
                {
                    "chr": group2.coords["chr"],
                    "pos": group2.coords["pos"],
                    "N": group2.N[:, r],
                    "X": group2.X[:, r],
                }
            )
            for r in range(group2.N.shape[1])
        ]
        bs = make_bsseq_data(frames1 + frames2)
        r1 = len(frames1)
        return bs.coords, bs.N[:, :r1], bs.X[:, :r1], bs.N[:, r1:], bs.X[:, r1:]

    # Case 3: Matrix tuples (N_matrix, X_matrix, coords_df)
    if isinstance(group1, tuple) and isinstance(group2, tuple):
        if len(group1) == 3:
            n1_raw, x1_raw, coords1 = group1
        else:
            raise ValueError(
                "Tuple group1 must be (N_matrix, X_matrix, coords_df)."
            )
        if len(group2) == 3:
            n2_raw, x2_raw, coords2 = group2
        elif len(group2) == 2:
            n2_raw, x2_raw = group2
            coords2 = coords1
        else:
            raise ValueError(
                "Tuple group2 must be (N_matrix, X_matrix[, coords_df])."
            )

        if not isinstance(coords1, pd.DataFrame) or not {"chr", "pos"}.issubset(coords1.columns):
            raise ValueError("coords_df must be a DataFrame with 'chr' and 'pos' columns.")

        n1 = np.atleast_2d(np.asarray(n1_raw, dtype=np.float64))
        x1 = np.atleast_2d(np.asarray(x1_raw, dtype=np.float64))
        if n1.shape[0] != len(coords1) and n1.shape[1] == len(coords1):
            n1 = n1.T
            x1 = x1.T

        n2 = np.atleast_2d(np.asarray(n2_raw, dtype=np.float64))
        x2 = np.atleast_2d(np.asarray(x2_raw, dtype=np.float64))
        if n2.shape[0] != len(coords2) and n2.shape[1] == len(coords2):
            n2 = n2.T
            x2 = x2.T

        if isinstance(coords2, pd.DataFrame) and not coords1[["chr", "pos"]].reset_index(
            drop=True
        ).equals(coords2[["chr", "pos"]].reset_index(drop=True)):
            frames1 = [
                pd.DataFrame(
                    {"chr": coords1["chr"], "pos": coords1["pos"], "N": n1[:, r], "X": x1[:, r]}
                )
                for r in range(n1.shape[1])
            ]
            frames2 = [
                pd.DataFrame(
                    {"chr": coords2["chr"], "pos": coords2["pos"], "N": n2[:, r], "X": x2[:, r]}
                )
                for r in range(n2.shape[1])
            ]
            bs = make_bsseq_data(frames1 + frames2)
            r1 = len(frames1)
            return bs.coords, bs.N[:, :r1], bs.X[:, :r1], bs.N[:, r1:], bs.X[:, r1:]

        coords_clean = coords1[["chr", "pos"]].copy().reset_index(drop=True)
        coords_clean["chr"] = coords_clean["chr"].astype(str)
        coords_clean["pos"] = coords_clean["pos"].astype(np.int64)
        order = np.argsort(
            coords_clean["pos"].to_numpy(), kind="mergesort"
        )
        # Stable sort by (chr, pos)
        coords_sorted = coords_clean.sort_values(["chr", "pos"], kind="mergesort")
        s_idx = coords_sorted.index.to_numpy()
        _ = order
        return (
            coords_sorted.reset_index(drop=True),
            n1[s_idx],
            x1[s_idx],
            n2[s_idx],
            x2[s_idx],
        )

    raise ValueError(
        "Unsupported group1 / group2 inputs. Pass lists of ('chr','pos','N','X') "
        "DataFrames, BSData objects, or (N_mat, X_mat, coords_df) tuples."
    )


def _initial_group_mean(
    chrom: np.ndarray,
    pos: np.ndarray,
    x_g: np.ndarray,
    n_g: np.ndarray,
    smoothing: bool,
    smoothing_span: float,
) -> np.ndarray:
    """Compute initial group methylation mean prior to dispersion shrinkage."""
    x_tot = np.sum(x_g, axis=1, keepdims=True)
    n_tot = np.sum(n_g, axis=1, keepdims=True)
    if smoothing:
        return smooth_counts_tricube(
            chrom, pos, x_tot, n_tot, smoothing_span=smoothing_span
        )[:, 0]
    has_n = n_tot[:, 0] > 0.0
    n_safe = np.where(has_n, n_tot[:, 0], 1.0)
    return np.clip(np.where(has_n, x_tot[:, 0] / n_safe, 0.5), 0.0, 1.0)


def dml_test(
    group1: pd.DataFrame | Sequence[pd.DataFrame] | tuple[object, ...] | BSData,
    group2: pd.DataFrame | Sequence[pd.DataFrame] | tuple[object, ...] | BSData,
    equal_disp: bool = False,
    smoothing: bool | None = None,
    smoothing_span: float = 500.0,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Perform two-group Beta-Binomial Empirical Bayes DML testing (`DSS::DMLtest`).

    Parameters
    ----------
    group1, group2
        Per-replicate DataFrames ``('chr', 'pos', 'N', 'X')``, ``BSData`` objects,
        or ``(N_matrix, X_matrix, coords_df)`` tuples.
    equal_disp
        Whether both groups share a common dispersion parameter ``phi1 == phi2``.
    smoothing
        Whether to apply local BSmooth tricube smoothing. When ``None``, defaults
        to ``True`` for ``settings="default"`` and ``False`` for
        ``settings="R_default"``.
    smoothing_span
        BSmooth tricube smoothing window width in bp (default ``500.0``).
    settings
        ``"default"`` or ``"R_default"`` (`DSS` v2.58 parity).

    Returns
    -------
    pd.DataFrame
        Columns: ``['chr', 'pos', 'mu1', 'mu2', 'diff', 'diff_se', 'stat',
        'phi1', 'phi2', 'pval', 'fdr']``.
    """
    cfg = DSSConfig(
        settings=settings,
        smoothing=smoothing,
        smoothing_span=smoothing_span,
    )
    assert cfg.smoothing is not None

    coords, n1, x1, n2, x2 = _coerce_two_groups(group1, group2)
    r1 = n1.shape[1]
    r2 = n2.shape[1]
    if r1 == 0 or r2 == 0:
        raise ValueError("Both group1 and group2 must have at least 1 replicate.")
    if r1 == 1 and r2 == 1 and not cfg.smoothing:
        raise ValueError(
            "DML testing without biological replicates requires smoothing=True."
        )

    chrom = coords["chr"].astype(str).to_numpy()
    pos = coords["pos"].to_numpy(dtype=np.float64)

    mu1_init = _initial_group_mean(
        chrom, pos, x1, n1, smoothing=cfg.smoothing, smoothing_span=cfg.smoothing_span
    )
    mu2_init = _initial_group_mean(
        chrom, pos, x2, n2, smoothing=cfg.smoothing, smoothing_span=cfg.smoothing_span
    )

    if equal_disp or ((r1 == 1 or r2 == 1) and not cfg.smoothing):
        groups_in: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if r1 >= 2 or cfg.smoothing:
            groups_in.append((x1, n1, mu1_init))
        if r2 >= 2 or cfg.smoothing:
            groups_in.append((x2, n2, mu2_init))
        phi_common, _, _ = shrink_dispersion_bb(groups_in, settings=cfg.settings)
        phi1 = phi_common.copy()
        phi2 = phi_common.copy()
    else:
        phi1, _, _ = shrink_dispersion_bb([(x1, n1, mu1_init)], settings=cfg.settings)
        phi2, _, _ = shrink_dispersion_bb([(x2, n2, mu2_init)], settings=cfg.settings)

    mu1, var1 = compute_group_mean_and_var(
        chrom,
        pos,
        x1,
        n1,
        phi1,
        smoothing=cfg.smoothing,
        smoothing_span=cfg.smoothing_span,
    )
    mu2, var2 = compute_group_mean_and_var(
        chrom,
        pos,
        x2,
        n2,
        phi2,
        smoothing=cfg.smoothing,
        smoothing_span=cfg.smoothing_span,
    )

    diff = mu1 - mu2
    tot_var = var1 + var2
    ok = np.isfinite(tot_var) & (tot_var > 0.0)
    diff_se = np.where(ok, np.sqrt(np.maximum(tot_var, 0.0)), np.nan)
    stat = np.where(ok, diff / diff_se, np.nan)
    pval = np.where(ok, 2.0 * norm.sf(np.abs(stat)), np.nan)
    fdr = _bh_fdr(pval)

    return pd.DataFrame(
        {
            "chr": coords["chr"].to_numpy(),
            "pos": coords["pos"].to_numpy(),
            "mu1": mu1,
            "mu2": mu2,
            "diff": diff,
            "diff_se": diff_se,
            "stat": stat,
            "phi1": phi1,
            "phi2": phi2,
            "pval": pval,
            "fdr": fdr,
        }
    )


def _composite_tail_pval(
    diff: np.ndarray, se: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact DSS composite-null tail p-value and P(|mu1 - mu2| > delta)."""
    ok = np.isfinite(diff) & np.isfinite(se) & (se > 0.0)
    se_safe = np.where(ok, se, 1.0)
    abs_d = np.where(ok, np.abs(diff), 0.0)

    if delta <= 0.0:
        pval = np.where(ok, 2.0 * norm.sf(abs_d / se_safe), np.nan)
        postprob = np.where(ok, 1.0 - pval, np.nan)
        return pval, postprob

    # Exact two-sided tail probability matching upstream DSS::callDML:
    # pnorm(-(abs(diff) - delta) / se) + pnorm(-(abs(diff) + delta) / se)
    z1 = -(abs_d - delta) / se_safe
    z2 = -(abs_d + delta) / se_safe
    pval_raw = norm.cdf(z1) + norm.cdf(z2)
    pval = np.where(ok, np.clip(pval_raw, 0.0, 1.0), np.nan)

    # Posterior probability P(|D| > delta) for D ~ N(diff, se^2):
    # P(D > delta) + P(D < -delta) = Phi((|diff| - delta)/se) + Phi((-|diff| - delta)/se)
    post_raw = norm.cdf((abs_d - delta) / se_safe) + norm.cdf((-abs_d - delta) / se_safe)
    postprob = np.where(ok, np.clip(post_raw, 0.0, 1.0), np.nan)
    return pval, postprob


def call_dml(
    dml_df: pd.DataFrame,
    delta: float = 0.1,
    p_threshold: float = 1e-5,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Call differentially methylated loci (DMLs) from `dml_test` output (`DSS::callDML`)."""
    validate_settings(settings)
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0; got {delta}.")
    if not (0.0 < p_threshold <= 1.0):
        raise ValueError(f"p_threshold must lie in (0, 1]; got {p_threshold}.")

    df = dml_df.copy()
    if delta > 0.0:
        if not {"diff", "diff_se"}.issubset(df.columns):
            raise ValueError(
                "dml_df must contain 'diff' and 'diff_se' columns when delta > 0."
            )
        p_eff, postprob = _composite_tail_pval(
            df["diff"].to_numpy(dtype=np.float64),
            df["diff_se"].to_numpy(dtype=np.float64),
            delta=float(delta),
        )
        df["pval"] = p_eff
        df["fdr"] = _bh_fdr(p_eff)
        df["postprob.overThreshold"] = postprob
    else:
        if "pval" not in df.columns:
            raise ValueError("dml_df must contain a 'pval' column.")
        p_eff = df["pval"].to_numpy(dtype=np.float64)

    keep = np.isfinite(p_eff) & (p_eff < p_threshold)
    called = df.loc[keep].copy()
    if called.empty:
        return called.reset_index(drop=True)

    called["_abs_stat"] = np.abs(called["stat"].to_numpy(dtype=np.float64))
    called = (
        called.sort_values(["pval", "_abs_stat"], ascending=[True, False])
        .drop(columns=["_abs_stat"])
        .reset_index(drop=True)
    )
    return called


def call_dmr(
    dml_df: pd.DataFrame,
    delta: float = 0.0,
    p_threshold: float | None = None,
    minlen: int = 50,
    minCG: int = 3,
    dis_merge: int = 100,
    pct_sig: float = 0.5,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Call differentially methylated regions (DMRs) (`DSS::callDMR`)."""
    cfg = DSSConfig(
        settings=settings,
        p_threshold_dmr=p_threshold,
        delta_dmr=delta,
        minlen=minlen,
        minCG=minCG,
        dis_merge=dis_merge,
        pct_sig=pct_sig,
    )
    assert cfg.p_threshold_dmr is not None
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0; got {delta}.")

    if not {"chr", "pos", "stat"}.issubset(dml_df.columns):
        raise ValueError("dml_df must contain 'chr', 'pos', and 'stat' columns.")

    df = dml_df.sort_values(["chr", "pos"], kind="mergesort").reset_index(drop=True)
    has_group_means = {"mu1", "mu2"}.issubset(df.columns)

    empty_cols = (
        [
            "chr",
            "start",
            "end",
            "length",
            "nCG",
            "meanMethy1",
            "meanMethy2",
            "diff.Methy",
            "areaStat",
        ]
        if has_group_means
        else ["chr", "start", "end", "length", "nCG", "areaStat"]
    )
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    if delta > 0.0:
        if not {"diff", "diff_se"}.issubset(df.columns):
            raise ValueError(
                "dml_df must contain 'diff' and 'diff_se' columns when delta > 0."
            )
        p_eff, _ = _composite_tail_pval(
            df["diff"].to_numpy(dtype=np.float64),
            df["diff_se"].to_numpy(dtype=np.float64),
            delta=float(delta),
        )
    else:
        p_eff = df["pval"].to_numpy(dtype=np.float64)

    stat_arr = df["stat"].to_numpy(dtype=np.float64)
    pos_arr = df["pos"].to_numpy(dtype=np.int64)
    chrom_arr = df["chr"].astype(str).to_numpy()
    is_sig = np.isfinite(p_eff) & (p_eff < cfg.p_threshold_dmr) & np.isfinite(stat_arr)
    signs = np.sign(stat_arr)

    dmr_records: list[dict[str, object]] = []

    for ch in pd.unique(chrom_arr):
        ch_idx = np.flatnonzero(chrom_arr == ch)
        sig_local = ch_idx[is_sig[ch_idx]]
        if len(sig_local) == 0:
            continue

        # Step 1: initial bumps of consecutive significant CpGs of the same sign
        # separated on the chromosome by <= cfg.dis_merge (or contiguous index runs)
        bumps: list[tuple[int, int, float]] = []
        b_start = int(sig_local[0])
        b_end = int(sig_local[0])
        b_sign = float(signs[b_start])

        for nxt in sig_local[1:]:
            nxt_i = int(nxt)
            nxt_sign = float(signs[nxt_i])
            # Consecutive CpGs within dis_merge of the same sign form initial bump
            if (
                nxt_i == (b_end + 1)
                and nxt_sign == b_sign
                and int(pos_arr[nxt_i] - pos_arr[b_end]) <= cfg.dis_merge
            ):
                b_end = nxt_i
            else:
                bumps.append((b_start, b_end, b_sign))
                b_start = nxt_i
                b_end = nxt_i
                b_sign = nxt_sign
        bumps.append((b_start, b_end, b_sign))

        # Step 2: merge adjacent bumps with identical sign if distance <= cfg.dis_merge
        merged: list[tuple[int, int]] = []
        m_start, m_end, m_sign = bumps[0]
        for nx_start, nx_end, nx_sign in bumps[1:]:
            gap_bp = int(pos_arr[nx_start] - pos_arr[m_end])
            if nx_sign == m_sign and gap_bp <= cfg.dis_merge:
                m_end = nx_end
            else:
                merged.append((m_start, m_end))
                m_start, m_end, m_sign = nx_start, nx_end, nx_sign
        merged.append((m_start, m_end))

        # Step 3: filter by minlen, minCG, and pct_sig
        for i_lo, i_hi in merged:
            start_bp = int(pos_arr[i_lo])
            end_bp = int(pos_arr[i_hi])
            length_bp = end_bp - start_bp
            n_cg = i_hi - i_lo + 1
            n_sig_in_reg = int(np.sum(is_sig[i_lo : i_hi + 1]))

            if length_bp < cfg.minlen:
                continue
            if n_cg < cfg.minCG:
                continue
            if (float(n_sig_in_reg) / float(n_cg)) < cfg.pct_sig:
                continue

            area_stat = float(np.nansum(stat_arr[i_lo : i_hi + 1]))
            rec: dict[str, object] = {
                "chr": str(ch),
                "start": start_bp,
                "end": end_bp,
                "length": length_bp,
                "nCG": n_cg,
            }
            if has_group_means:
                m1 = float(np.nanmean(df["mu1"].to_numpy(dtype=np.float64)[i_lo : i_hi + 1]))
                m2 = float(np.nanmean(df["mu2"].to_numpy(dtype=np.float64)[i_lo : i_hi + 1]))
                rec["meanMethy1"] = m1
                rec["meanMethy2"] = m2
                rec["diff.Methy"] = m1 - m2
            rec["areaStat"] = area_stat
            dmr_records.append(rec)

    if not dmr_records:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame(dmr_records)
    out["_abs_area"] = np.abs(out["areaStat"].to_numpy(dtype=np.float64))
    out = (
        out.sort_values("_abs_area", ascending=False)
        .drop(columns=["_abs_area"])
        .reset_index(drop=True)
    )
    return out
