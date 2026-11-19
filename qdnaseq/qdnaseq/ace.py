"""
Bundled ACE (`tgac-vumc/ACE`, Poell et al., Bioinformatics 2019) reference-free
2D cellularity/ploidy solver (`squaremodel`, `run_ace`, `ACEResult`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import SettingsMode, get_qdnaseq_config, validate_settings
from .data import QDNAseqData
from .segment import segment_bins


@dataclass(frozen=True)
class ACEResult:
    """Result container from `run_ace` / `squaremodel`."""

    cellularity: float
    ploidy: float
    error: float
    optima: pd.DataFrame
    error_matrix: pd.DataFrame
    segments: pd.DataFrame
    bins: pd.DataFrame | None = None
    sample: str = "sample"
    method: str = "RMSE"
    settings: SettingsMode = "default"

    def to_dataframe(self) -> pd.DataFrame:
        """Return absolute copy-number segment DataFrame."""
        return self.segments.copy()

    def summary(self) -> str:
        return (
            f"ACEResult (sample={self.sample}, method={self.method}, settings={self.settings})\n"
            f"  Best cellularity: {self.cellularity:.3f}\n"
            f"  Best ploidy:      {self.ploidy:.3f}\n"
            f"  Fit error:        {self.error:.5f}\n"
            f"  Ranked optima:    {len(self.optima)}"
        )


def _extract_ace_segments(
    qdata_or_segments: QDNAseqData | pd.DataFrame,
    sample: str | None,
    settings: SettingsMode,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if isinstance(qdata_or_segments, QDNAseqData):
        qcur = (
            qdata_or_segments
            if qdata_or_segments.segmented is not None
            else segment_bins(qdata_or_segments, settings=settings)
        )
        sname = sample if sample is not None else qcur.sample_names[0]
        if sname not in qcur.segments:
            raise ValueError(f"Unknown sample {sname!r}; available: {qcur.sample_names}.")
        seg_df = qcur.segments[sname].copy()
        bin_df = qcur.to_dataframe(sample=sname)
        return seg_df, bin_df, sname

    if isinstance(qdata_or_segments, pd.DataFrame):
        if qdata_or_segments.empty:
            raise ValueError("`segments` DataFrame cannot be empty.")
        seg_df = qdata_or_segments.copy()
        if "segmented" not in seg_df.columns:
            if "ratio" in seg_df.columns:
                seg_df["segmented"] = seg_df["ratio"].astype(np.float64)
            elif "copynumber" in seg_df.columns:
                seg_df["segmented"] = seg_df["copynumber"].astype(np.float64)
            elif "log2_ratio" in seg_df.columns:
                seg_df["segmented"] = np.power(2.0, seg_df["log2_ratio"].astype(np.float64))
            elif "seg_mean" in seg_df.columns:
                seg_df["segmented"] = np.power(2.0, seg_df["seg_mean"].astype(np.float64))
            else:
                raise ValueError(
                    "Segment DataFrame must include `segmented`, `ratio`, `log2_ratio`, or `seg_mean`."
                )
        if "num_bins" not in seg_df.columns:
            if "num_mark" in seg_df.columns:
                seg_df["num_bins"] = seg_df["num_mark"].astype(np.int64)
            elif "start" in seg_df.columns and "end" in seg_df.columns:
                seg_df["num_bins"] = np.maximum(
                    1, (seg_df["end"].astype(np.int64) - seg_df["start"].astype(np.int64)) // 50_000
                )
            else:
                seg_df["num_bins"] = 10
        return seg_df.reset_index(drop=True), None, sample or "sample"

    raise ValueError("Expected `QDNAseqData` or segment `pd.DataFrame` for ACE.")


def squaremodel(
    qdata_or_segments: QDNAseqData | pd.DataFrame,
    cellularity: np.ndarray | list[float] | None = None,
    ploidy: np.ndarray | list[float] | None = None,
    method: Literal["RMSE", "MAE", "SMRE"] | None = None,
    penploidy: float | None = None,
    pencellularity: float | None = None,
    max_cn: int = 12,
    sample: str | None = None,
    settings: SettingsMode = "default",
) -> ACEResult:
    """
    2D `(cellularity, ploidy)` absolute copy-number grid search implementing
    `ACE::squaremodel` (Poell et al., Bioinformatics 2019).
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    eff_method = (method or cfg.ace_method).upper()
    if eff_method not in ("RMSE", "MAE", "SMRE"):
        raise ValueError(f"Unknown `method`={method!r}. Expected 'RMSE', 'MAE', or 'SMRE'.")

    eff_penp = float(penploidy) if penploidy is not None else cfg.ace_penploidy
    eff_penc = float(pencellularity) if pencellularity is not None else cfg.ace_pencellularity

    c_grid = (
        np.asarray(cellularity, dtype=np.float64)
        if cellularity is not None
        else cfg.default_cellularity_grid()
    )
    p_grid = (
        np.asarray(ploidy, dtype=np.float64)
        if ploidy is not None
        else cfg.default_ploidy_grid()
    )
    if c_grid.size == 0 or p_grid.size == 0:
        raise ValueError("`cellularity` and `ploidy` grids must be non-empty.")
    c_grid = np.clip(c_grid, 0.01, 1.0)
    if np.any(p_grid <= 0.2):
        raise ValueError("`ploidy` grid values must be > 0.2.")

    seg_df, bin_df, sname = _extract_ace_segments(qdata_or_segments, sample, cfg.settings)

    r_raw = seg_df["segmented"].to_numpy(dtype=np.float64)
    w_raw = np.maximum(seg_df["num_bins"].to_numpy(dtype=np.float64), 1.0)
    w_norm = w_raw / np.sum(w_raw)

    mean_r = float(np.sum(w_norm * r_raw))
    if mean_r <= 1e-8:
        raise ValueError("Segment ratios have non-positive weighted mean.")
    r_scaled = r_raw / mean_r  # Exact mass-balance relative ratio around 1.0

    # Grid dimensions: (C_len, P_len, S_len)
    rho = c_grid[:, np.newaxis, np.newaxis]
    psi = p_grid[np.newaxis, :, np.newaxis]
    bulk_d = rho * psi + 2.0 * (1.0 - rho)

    cont_cn = (r_scaled[np.newaxis, np.newaxis, :] * bulk_d - 2.0 * (1.0 - rho)) / rho
    int_cn = np.clip(np.round(cont_cn), 0.0, float(max_cn))
    abs_err = np.abs(cont_cn - int_cn)

    w_bcast = w_norm[np.newaxis, np.newaxis, :]
    if eff_method == "RMSE":
        raw_err = np.sqrt(np.sum(w_bcast * (abs_err * abs_err), axis=-1))
    elif eff_method == "MAE":
        raw_err = np.sum(w_bcast * abs_err, axis=-1)
    else:  # SMRE
        raw_err = np.sum(w_bcast * np.sqrt(abs_err), axis=-1) ** 2

    # Penalize unphysical negative copy numbers (< -0.25)
    neg_overshoot = np.sum(
        w_bcast * np.maximum(0.0, -0.25 - cont_cn),
        axis=-1,
    )

    # Detect step-skipping sub-harmonics where all copy-number steps have GCD >= 2
    int_cn_i64 = int_cn.astype(np.int64)
    diffs_from_first = np.abs(int_cn_i64 - int_cn_i64[:, :, :1])
    gcd_grid = np.gcd.reduce(diffs_from_first, axis=-1)  # 0 if constant CN, else GCD of steps
    has_variation = np.max(int_cn_i64, axis=-1) > np.min(int_cn_i64, axis=-1)
    subharmonic_pen = np.where(has_variation & (gcd_grid >= 2), 0.06, 0.0)

    p_factor = np.power(1.0 + np.abs(p_grid[np.newaxis, :] - 2.0), eff_penp)
    c_factor = np.power(1.0 / c_grid[:, np.newaxis], eff_penc)

    base_err = raw_err + 2.0 * neg_overshoot + subharmonic_pen
    tot_err = (base_err + 1e-4) * p_factor * c_factor - 1e-4

    # Find local minima in 3x3 neighborhood + global minimum
    n_c, n_p = tot_err.shape
    g_flat = int(np.argmin(tot_err))
    gc_i, gp_i = divmod(g_flat, n_p)
    minima: list[tuple[float, int, int]] = [(float(tot_err[gc_i, gp_i]), gc_i, gp_i)]

    for ci in range(n_c):
        clo, chi = max(0, ci - 2), min(n_c, ci + 3)
        for pi in range(n_p):
            if ci == gc_i and pi == gp_i:
                continue
            plo, phi = max(0, pi - 2), min(n_p, pi + 3)
            v = float(tot_err[ci, pi])
            if v <= float(np.min(tot_err[clo:chi, plo:phi])) + 1e-12:
                minima.append((v, ci, pi))

    minima.sort(key=lambda t: t[0])

    dedup: list[tuple[float, int, int]] = []
    for val, ci, pi in minima:
        cell_v = float(c_grid[ci])
        ploi_v = float(p_grid[pi])
        if any(
            abs(cell_v - float(c_grid[kc])) < 0.05 and abs(ploi_v - float(p_grid[kp])) < 0.20
            for _, kc, kp in dedup
        ):
            continue
        dedup.append((val, ci, pi))
        if len(dedup) >= 12:
            break

    opt_rows = []
    for rank_i, (err_val, ci, pi) in enumerate(dedup, start=1):
        mean_c = float(np.sum(w_norm * int_cn[ci, pi]))
        opt_rows.append(
            {
                "rank": rank_i,
                "cellularity": round(float(c_grid[ci]), 4),
                "ploidy": round(float(p_grid[pi]), 4),
                "error": float(err_val),
                "raw_error": float(raw_err[ci, pi]),
                "mean_cn": round(mean_c, 4),
            }
        )
    optima_df = pd.DataFrame(opt_rows)

    best_err, b_ci, b_pi = dedup[0]
    best_cell = round(float(c_grid[b_ci]), 4)
    best_ploi = round(float(p_grid[b_pi]), 4)

    seg_out = seg_df.copy()
    seg_out["continuous_cn"] = cont_cn[b_ci, b_pi]
    seg_out["absolute_cn"] = int_cn[b_ci, b_pi].astype(np.int64)

    bin_out: pd.DataFrame | None = None
    if bin_df is not None and "copynumber" in bin_df.columns:
        bin_out = bin_df.copy()
        b_d = best_cell * best_ploi + 2.0 * (1.0 - best_cell)
        b_r = bin_out["copynumber"].to_numpy(dtype=np.float64) / mean_r
        b_cont = (b_r * b_d - 2.0 * (1.0 - best_cell)) / best_cell
        bin_out["continuous_cn"] = b_cont
        if "segmented" in bin_out.columns:
            s_r = bin_out["segmented"].to_numpy(dtype=np.float64) / mean_r
            s_cont = (s_r * b_d - 2.0 * (1.0 - best_cell)) / best_cell
            s_abs = np.where(
                np.isfinite(s_cont), np.clip(np.round(s_cont), 0.0, float(max_cn)), np.nan
            )
            bin_out["absolute_cn"] = s_abs

    err_mat_df = pd.DataFrame(
        tot_err.T,
        index=pd.Index(np.round(p_grid, 4), name="ploidy"),
        columns=pd.Index(np.round(c_grid, 4), name="cellularity"),
    )

    return ACEResult(
        cellularity=best_cell,
        ploidy=best_ploi,
        error=float(best_err),
        optima=optima_df,
        error_matrix=err_mat_df,
        segments=seg_out,
        bins=bin_out,
        sample=sname,
        method=eff_method,
        settings=cfg.settings,
    )


def run_ace(
    qdata_or_segments: QDNAseqData | pd.DataFrame,
    cellularity: np.ndarray | list[float] | None = None,
    ploidy: np.ndarray | list[float] | None = None,
    method: Literal["RMSE", "MAE", "SMRE"] | None = None,
    penploidy: float | None = None,
    pencellularity: float | None = None,
    max_cn: int = 12,
    sample: str | None = None,
    settings: SettingsMode = "default",
) -> ACEResult:
    """Primary Pythonic alias for `squaremodel` (`ACE` cellularity/ploidy estimation)."""
    return squaremodel(
        qdata_or_segments,
        cellularity=cellularity,
        ploidy=ploidy,
        method=method,
        penploidy=penploidy,
        pencellularity=pencellularity,
        max_cn=max_cn,
        sample=sample,
        settings=settings,
    )
