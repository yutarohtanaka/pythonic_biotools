"""
Multi-start grid search, solution filtering (`"default"` vs `"R_default"`),
segment collapsing, and public `run_ichorcna` / `IchorCNA` API for `ichorcna`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .hmm import (
    EMRunResult,
    build_chrom_batch,
    emission_means,
    estimate_empirical_noise,
    fit_single_ichorcna_em,
)
from .io import (
    HMMState,
    IchorCNAConfig,
    SettingsMode,
    build_hmm_states,
    canonical_chrom,
    chrom_sort_key,
    validate_settings,
)
from .normalize import normalize_ulp_bins


def _collapse_viterbi_segments(
    bins_df: pd.DataFrame,
    viterbi_idx: np.ndarray,
    states: list[HMMState],
    mu_vec: np.ndarray,
    normal_fraction: float,
    subclone_fraction: float,
) -> pd.DataFrame:
    """Collapse consecutive bins with identical `(chrom, viterbi_state)` into segments."""
    n_bins = len(bins_df)
    if n_bins == 0:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "num_bins",
                "state",
                "copy_number",
                "is_subclonal",
                "cellular_prevalence",
                "median_logR",
                "expected_logR",
            ]
        )

    chroms = bins_df["chrom"].to_numpy()
    starts = bins_df["start"].to_numpy(dtype=np.int64)
    ends = bins_df["end"].to_numpy(dtype=np.int64)
    logr = bins_df["logR"].to_numpy(dtype=np.float64)

    tfx = 1.0 - normal_fraction
    seg_rows: list[dict[str, Any]] = []
    seg_start_i = 0

    for i in range(1, n_bins + 1):
        if i == n_bins or chroms[i] != chroms[seg_start_i] or viterbi_idx[i] != viterbi_idx[seg_start_i]:
            st_idx = int(viterbi_idx[seg_start_i])
            st = states[st_idx]
            if st.copy_number == 2 and not st.is_subclonal:
                cell_prev = tfx
            elif st.is_subclonal:
                cell_prev = tfx * subclone_fraction
            else:
                cell_prev = tfx

            seg_rows.append(
                {
                    "chrom": chroms[seg_start_i],
                    "start": int(starts[seg_start_i]),
                    "end": int(ends[i - 1]),
                    "num_bins": int(i - seg_start_i),
                    "state": st.name,
                    "copy_number": int(st.copy_number),
                    "is_subclonal": bool(st.is_subclonal),
                    "cellular_prevalence": float(cell_prev),
                    "median_logR": float(np.median(logr[seg_start_i:i])),
                    "expected_logR": float(mu_vec[st_idx]),
                }
            )
            seg_start_i = i

    return pd.DataFrame(seg_rows)


@dataclass
class IchorCNAResult:
    """Result container returned by `run_ichorcna` and `IchorCNA`."""

    tumor_fraction: float
    normal_fraction: float
    ploidy: float
    subclone_fraction: float
    frac_genome_subclonal: float
    frac_cna_subclonal: float
    segments_df: pd.DataFrame
    bins_df: pd.DataFrame
    solutions_df: pd.DataFrame
    loglik: float
    settings: SettingsMode

    def summary(self) -> dict[str, Any]:
        """Return dictionary summary of ctDNA tumor fraction, ploidy, and CNA metrics."""
        return {
            "tumor_fraction": self.tumor_fraction,
            "normal_fraction": self.normal_fraction,
            "ploidy": self.ploidy,
            "subclone_fraction": self.subclone_fraction,
            "frac_genome_subclonal": self.frac_genome_subclonal,
            "frac_cna_subclonal": self.frac_cna_subclonal,
            "n_segments": int(len(self.segments_df)),
            "n_bins": int(len(self.bins_df)),
            "loglik": self.loglik,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return copy of segmented copy-number calls (`segments_df`)."""
        return self.segments_df.copy()


def run_ichorcna(
    bins_df: pd.DataFrame,
    normal_init: Sequence[float] | None = None,
    ploidy_init: Sequence[float] = (2.0, 3.0),
    max_cn: int = 5,
    include_homd: bool = False,
    sc_states: Sequence[int] = (1, 3),
    txn_e: float = 0.9999999,
    txn_strength: float = 10000.0,
    max_iter: int = 50,
    low_tf_mode: bool = False,
    settings: SettingsMode = "default",
) -> IchorCNAResult:
    """
    Estimate cell-free DNA tumor fraction (`TFx = 1 - n`), tumor ploidy, subclonal
    fraction, and genome-wide somatic copy-number alterations from ultra-low-pass
    WGS (`ULP-WGS`) bins via a multi-start Student-t HMM + Generalized EM
    (Adalsteinsson et al., Nature Communications 2017).

    Parameters
    ----------
    bins_df : pd.DataFrame
        Genomic bins containing `chrom`, `start`, `end`, and `logR` (or raw
        `reads` + `gc`, in which case `normalize_ulp_bins` is applied automatically).
    normal_init : Sequence[float] | None, default=None
        Grid of initial normal fractions `n_0 in (0, 1)`. If `None`, uses
        `(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)` in `"default"` mode, and
        `(0.5, 0.6, 0.7, 0.8, 0.9)` (or `(0.95, 0.99, 0.995)` when `low_tf_mode=True`)
        in `"R_default"` mode.
    ploidy_init : Sequence[float], default=(2.0, 3.0)
        Grid of initial tumor ploidies `phi_0`.
    max_cn : int, default=5
        Maximum integer clonal copy-number state (`>= 3`).
    include_homd : bool, default=False
        Include homozygous deletion state `HOMD (0)` when `True`.
    sc_states : Sequence[int], default=(1, 3)
        Copy numbers modeled with additional subclonal states (`1_sub`, `3_sub`).
    txn_e : float, default=0.9999999
        Self-transition probability parameter on the HMM transition diagonal.
    txn_strength : float, default=10000.0
        Dirichlet prior strength on transition probabilities.
    max_iter : int, default=50
        Maximum Generalized EM iterations per starting grid point.
    low_tf_mode : bool, default=False
        In `"R_default"` mode, switches default `normal_init` to `(0.95, 0.99, 0.995)`
        matching `ichorCNA`'s low-tumor-fraction configuration.
    settings : {"default", "R_default"}, default="default"
        `"default"` combines standard + low-ctDNA starts and applies diploid
        regularization when `tumor_fraction < 0.08` without `HLAMP`; `"R_default"`
        reproduces `ichorCNA` v0.3.2 grid initialization and subclonal solution
        filtering (`frac_cna_subclonal <= 0.70`, `frac_genome_subclonal <= 0.50`).

    Returns
    -------
    IchorCNAResult
        Selected optimal solution, Viterbi segments, annotated bins, and full
        multi-start solutions table.
    """
    validate_settings(settings)
    if not isinstance(bins_df, pd.DataFrame) or bins_df.empty:
        raise ValueError("bins_df must be a non-empty pandas DataFrame.")

    if "logR" not in bins_df.columns:
        work_df = normalize_ulp_bins(bins_df, settings=settings)
    else:
        work_df = bins_df.copy()
        if "chrom" not in work_df.columns and "chr" in work_df.columns:
            work_df = work_df.rename(columns={"chr": "chrom"})
        for req in ("chrom", "start", "end", "logR"):
            if req not in work_df.columns:
                raise ValueError(f"bins_df is missing required column {req!r}.")
        work_df["chrom_canon"] = [canonical_chrom(c) for c in work_df["chrom"]]
        sort_keys = [chrom_sort_key(c) for c in work_df["chrom_canon"]]
        work_df["_cg"] = [k[0] for k in sort_keys]
        work_df["_cl"] = [k[1] for k in sort_keys]
        work_df = work_df.sort_values(["_cg", "_cl", "start", "end"], kind="mergesort")
        work_df = work_df.drop(columns=["chrom_canon", "_cg", "_cl"])
        work_df = work_df[np.isfinite(work_df["logR"].astype(np.float64))].reset_index(drop=True)
        if work_df.empty:
            raise ValueError("bins_df contains no finite logR values.")

    if normal_init is None:
        if settings == "default":
            n_grid = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
        else:
            n_grid = (0.95, 0.99, 0.995) if low_tf_mode else (0.5, 0.6, 0.7, 0.8, 0.9)
    else:
        n_grid = tuple(float(v) for v in normal_init)
        if not n_grid:
            raise ValueError("normal_init cannot be empty.")

    p_grid = tuple(float(v) for v in ploidy_init)
    if not p_grid:
        raise ValueError("ploidy_init cannot be empty.")

    cfg = IchorCNAConfig(
        normal_init=n_grid,
        ploidy_init=p_grid,
        max_cn=int(max_cn),
        include_homd=bool(include_homd),
        sc_states=tuple(int(c) for c in sc_states),
        txn_e=float(txn_e),
        txn_strength=float(txn_strength),
        max_iter=int(max_iter),
        low_tf_mode=bool(low_tf_mode),
        settings=settings,
    )

    states = build_hmm_states(
        max_cn=cfg.max_cn,
        include_homd=cfg.include_homd,
        sc_states=cfg.sc_states,
    )
    copy_numbers = np.array([st.copy_number for st in states], dtype=np.int64)
    is_subclonal = np.array([st.is_subclonal for st in states], dtype=bool)

    chrom_canon_arr = [canonical_chrom(c) for c in work_df["chrom"]]
    batch = build_chrom_batch(chrom_canon_arr)
    y = work_df["logR"].to_numpy(dtype=np.float64)
    sigma0 = estimate_empirical_noise(y, batch)

    runs: list[EMRunResult] = []
    sol_records: list[dict[str, Any]] = []

    for p0 in cfg.ploidy_init:
        for n0 in cfg.normal_init:
            run_res = fit_single_ichorcna_em(
                y=y,
                states=states,
                batch=batch,
                normal_init=n0,
                ploidy_init=p0,
                cfg=cfg,
                sigma0=sigma0,
            )

            vit_cn = copy_numbers[run_res.viterbi_states]
            vit_sub = is_subclonal[run_res.viterbi_states]
            altered_mask = vit_cn != 2
            n_altered = int(np.sum(altered_mask))

            # If zero copy-number alterations are detected genome-wide, ctDNA TFx = 0.0
            if n_altered == 0:
                eff_n = 1.0
                eff_tf = 0.0
                eff_ploidy = 2.0
                eff_sub = 0.0
                frac_gen_sub = 0.0
                frac_cna_sub = 0.0
                frac_hlamp = 0.0
            else:
                eff_n = run_res.normal_fraction
                eff_tf = run_res.tumor_fraction
                eff_ploidy = run_res.ploidy
                frac_gen_sub = float(np.mean(vit_sub))
                frac_cna_sub = float(np.sum(vit_sub & altered_mask) / n_altered)
                eff_sub = run_res.subclone_fraction if frac_gen_sub > 0.0 else 0.0
                frac_hlamp = float(np.mean(vit_cn >= 5))

            runs.append(run_res)
            sol_records.append(
                {
                    "normal_init": n0,
                    "ploidy_init": p0,
                    "normal_fraction": eff_n,
                    "tumor_fraction": eff_tf,
                    "ploidy": eff_ploidy,
                    "subclone_fraction": eff_sub,
                    "frac_genome_subclonal": frac_gen_sub,
                    "frac_cna_subclonal": frac_cna_sub,
                    "frac_hlamp": frac_hlamp,
                    "n_altered_bins": n_altered,
                    "loglik": run_res.loglik,
                    "log_posterior": run_res.log_posterior,
                    "n_iter": run_res.n_iter,
                }
            )

    sol_df = pd.DataFrame(sol_records)

    # Solution selection filter (ichorCNA v0.3.2 subclonal pathology rejection)
    subclonal_ok = (sol_df["frac_cna_subclonal"] <= 0.70) & (
        sol_df["frac_genome_subclonal"] <= 0.50
    )
    cand_indices = (
        np.flatnonzero(subclonal_ok.to_numpy())
        if bool(subclonal_ok.any())
        else np.arange(len(sol_df))
    )

    score_col = "log_posterior"
    if settings == "default":
        # Automatic diploid regularization at low ctDNA (< 8% TFx and < 1% diploid HLAMP)
        diploid_mask = (sol_df["ploidy"] >= 1.65) & (sol_df["ploidy"] <= 2.45)
        dip_cands = cand_indices[diploid_mask.to_numpy()[cand_indices]]
        if dip_cands.size > 0:
            best_dip_idx = int(dip_cands[np.argmax(sol_df.loc[dip_cands, score_col].to_numpy())])
            best_all_idx = int(cand_indices[np.argmax(sol_df.loc[cand_indices, score_col].to_numpy())])
            dip_tf = float(sol_df.loc[best_dip_idx, "tumor_fraction"])
            dip_hlamp = float(sol_df.loc[best_dip_idx, "frac_hlamp"])
            ll_gap = float(sol_df.loc[best_all_idx, score_col] - sol_df.loc[best_dip_idx, score_col])
            if (dip_tf < 0.08 and dip_hlamp < 0.01 and ll_gap < 15.0) or best_dip_idx == best_all_idx:
                chosen_idx = best_dip_idx
            else:
                chosen_idx = best_all_idx
        else:
            chosen_idx = int(cand_indices[np.argmax(sol_df.loc[cand_indices, score_col].to_numpy())])
    else:
        chosen_idx = int(cand_indices[np.argmax(sol_df.loc[cand_indices, score_col].to_numpy())])

    sol_df["selected"] = False
    sol_df.loc[chosen_idx, "selected"] = True

    best_run = runs[chosen_idx]
    best_row = sol_df.iloc[chosen_idx]

    best_n = float(best_row["normal_fraction"])
    best_tf = float(best_row["tumor_fraction"])
    best_phi = float(best_row["ploidy"])
    best_s = float(best_row["subclone_fraction"])
    best_ll = float(best_row["loglik"])

    mu_best = emission_means(
        best_n,
        best_phi,
        best_run.subclone_fraction,
        copy_numbers.astype(np.float64),
        is_subclonal,
    )

    vit_idx = best_run.viterbi_states
    segments_df = _collapse_viterbi_segments(
        bins_df=work_df,
        viterbi_idx=vit_idx,
        states=states,
        mu_vec=mu_best,
        normal_fraction=best_n,
        subclone_fraction=best_s,
    )

    bins_out = work_df.copy()
    bins_out["state"] = [states[i].name for i in vit_idx]
    bins_out["copy_number"] = copy_numbers[vit_idx]
    bins_out["is_subclonal"] = is_subclonal[vit_idx]
    bins_out["expected_logR"] = mu_best[vit_idx]
    bins_out["posterior_max"] = np.max(best_run.gamma, axis=1)

    return IchorCNAResult(
        tumor_fraction=best_tf,
        normal_fraction=best_n,
        ploidy=best_phi,
        subclone_fraction=best_s,
        frac_genome_subclonal=float(best_row["frac_genome_subclonal"]),
        frac_cna_subclonal=float(best_row["frac_cna_subclonal"]),
        segments_df=segments_df,
        bins_df=bins_out,
        solutions_df=sol_df,
        loglik=best_ll,
        settings=settings,
    )


class IchorCNA:
    """
    Object-oriented interface for ultra-low-pass WGS normalization, multi-start
    Student-t HMM copy-number segmentation, and ctDNA tumor-fraction estimation.
    """

    def __init__(
        self,
        normal_init: Sequence[float] | None = None,
        ploidy_init: Sequence[float] = (2.0, 3.0),
        max_cn: int = 5,
        include_homd: bool = False,
        sc_states: Sequence[int] = (1, 3),
        txn_e: float = 0.9999999,
        txn_strength: float = 10000.0,
        max_iter: int = 50,
        low_tf_mode: bool = False,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings = validate_settings(settings)
        self.normal_init = tuple(float(x) for x in normal_init) if normal_init is not None else None
        self.ploidy_init = tuple(float(x) for x in ploidy_init)
        self.max_cn = int(max_cn)
        self.include_homd = bool(include_homd)
        self.sc_states = tuple(int(c) for c in sc_states)
        self.txn_e = float(txn_e)
        self.txn_strength = float(txn_strength)
        self.max_iter = int(max_iter)
        self.low_tf_mode = bool(low_tf_mode)
        self.result_: IchorCNAResult | None = None

    def normalize(
        self,
        bins_df: pd.DataFrame,
        panel_of_normals: pd.DataFrame | pd.Series | np.ndarray | None = None,
        centromeres_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Normalize raw binned ULP-WGS read counts via GC/mappability LOESS + PoN."""
        return normalize_ulp_bins(
            bins_df=bins_df,
            panel_of_normals=panel_of_normals,
            centromeres_df=centromeres_df,
            settings=self.settings,
        )

    def fit(
        self,
        bins_df: pd.DataFrame,
        panel_of_normals: pd.DataFrame | pd.Series | np.ndarray | None = None,
        centromeres_df: pd.DataFrame | None = None,
    ) -> IchorCNAResult:
        """Fit `IchorCNA` on normalized `logR` bins or raw ULP-WGS `reads` bins."""
        if "logR" not in bins_df.columns or panel_of_normals is not None or centromeres_df is not None:
            norm_bins = self.normalize(
                bins_df, panel_of_normals=panel_of_normals, centromeres_df=centromeres_df
            )
        else:
            norm_bins = bins_df

        self.result_ = run_ichorcna(
            bins_df=norm_bins,
            normal_init=self.normal_init,
            ploidy_init=self.ploidy_init,
            max_cn=self.max_cn,
            include_homd=self.include_homd,
            sc_states=self.sc_states,
            txn_e=self.txn_e,
            txn_strength=self.txn_strength,
            max_iter=self.max_iter,
            low_tf_mode=self.low_tf_mode,
            settings=self.settings,
        )
        return self.result_

    def run(
        self,
        bins_df: pd.DataFrame,
        panel_of_normals: pd.DataFrame | pd.Series | np.ndarray | None = None,
        centromeres_df: pd.DataFrame | None = None,
    ) -> IchorCNAResult:
        """Alias for `.fit(...)`."""
        return self.fit(
            bins_df=bins_df,
            panel_of_normals=panel_of_normals,
            centromeres_df=centromeres_df,
        )
