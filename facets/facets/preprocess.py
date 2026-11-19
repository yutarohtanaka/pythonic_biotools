"""
Preprocessing, depth/heterozygosity filtering, neighborhood SNP thinning,
pure-NumPy GC LOWESS logR normalization, and Haldane-Anscombe allelic logOR.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from .io import FacetsConfig, SettingsType, validate_and_normalize_pileup, validate_settings

_MAD_NORMAL_CONST: float = 0.6744897501960817


@dataclass(frozen=True)
class PreProcResult:
    """Result container from `pre_proc_sample`."""

    snps: pd.DataFrame
    genome: str
    n_raw: int
    n_retained: int
    n_het: int
    sigma_logr: float
    gc_corrected: bool
    config: FacetsConfig

    def to_dataframe(self) -> pd.DataFrame:
        """Return preprocessed SNP-level DataFrame."""
        return self.snps.copy()

    def summary(self) -> str:
        """Return human-readable summary of preprocessing metrics."""
        return (
            f"FACETS PreProcResult (genome={self.genome})\n"
            f"  Raw SNPs:      {self.n_raw}\n"
            f"  Retained SNPs: {self.n_retained}\n"
            f"  Het SNPs:      {self.n_het}\n"
            f"  Sigma logR:    {self.sigma_logr:.4f}\n"
            f"  GC corrected:  {self.gc_corrected}"
        )


def _find_knn_window(x_sorted: np.ndarray, x0: float, k: int) -> tuple[int, int]:
    """
    Find contiguous [left, left + k) slice of `x_sorted` containing the k
    nearest neighbors to `x0` in O(log n) time.
    """
    n = x_sorted.size
    if k >= n:
        return 0, n
    lo = 0
    hi = n - k
    while lo < hi:
        mid = (lo + hi) // 2
        if x0 - x_sorted[mid] <= x_sorted[mid + k - 1] - x0:
            hi = mid
        else:
            lo = mid + 1
    if lo > 0:
        d_cur = max(x0 - x_sorted[lo], x_sorted[lo + k - 1] - x0)
        d_prev = max(x0 - x_sorted[lo - 1], x_sorted[lo + k - 2] - x0)
        if d_prev < d_cur:
            lo -= 1
    return lo, lo + k


def lowess_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.5,
    n_iter: int = 3,
    max_eval_knots: int = 250,
) -> np.ndarray:
    """
    Pure-NumPy Cleveland (1979) robust locally weighted linear regression (LOWESS).

    For large arrays (n > 400), evaluates local linear fits on `max_eval_knots`
    quantile+uniform knots and interpolates cleanly across all points.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("`x` and `y` must be 1D arrays of equal length.")

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    out = np.zeros_like(y_arr, dtype=np.float64)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        if n_valid > 0:
            out[valid] = float(np.median(y_arr[valid]))
        return out

    xv = x_arr[valid]
    yv = y_arr[valid]
    if float(np.std(xv)) <= 1e-12:
        out[valid] = float(np.median(yv))
        return out

    order = np.argsort(xv, kind="mergesort")
    xs = xv[order]
    ys = yv[order]

    k = max(3, min(n_valid, int(np.floor(frac * n_valid))))
    use_knots = n_valid > 400
    if use_knots:
        q_grid = np.quantile(xs, np.linspace(0.0, 1.0, max_eval_knots))
        u_grid = np.linspace(xs[0], xs[-1], max_eval_knots // 2)
        knots = np.unique(np.concatenate([q_grid, u_grid]))
    else:
        knots = xs

    rob_w = np.ones(n_valid, dtype=np.float64)
    y_fit = np.empty(n_valid, dtype=np.float64)

    for it in range(max(1, n_iter)):
        knot_vals = np.empty(knots.size, dtype=np.float64)
        for idx_k, x0 in enumerate(knots):
            left, right = _find_knn_window(xs, float(x0), k)
            xw = xs[left:right]
            yw = ys[left:right]
            rw = rob_w[left:right]

            dist = np.abs(xw - x0)
            d_max = float(np.max(dist))
            if d_max <= 1e-12:
                sw = float(np.sum(rw))
                knot_vals[idx_k] = (
                    float(np.sum(rw * yw) / sw) if sw > 0.0 else float(np.median(yw))
                )
                continue

            u = dist / d_max
            w = rw * (1.0 - u * u * u) ** 3
            if int(np.sum(w > 0.0)) < 2:
                u2 = dist / (d_max * 1.0001)
                w = rw * (1.0 - u2 * u2 * u2) ** 3

            sw = float(np.sum(w))
            if sw <= 1e-15:
                knot_vals[idx_k] = float(np.median(yw))
                continue

            dx = xw - x0
            x_bar = float(np.sum(w * dx) / sw)
            y_bar = float(np.sum(w * yw) / sw)
            dx_c = dx - x_bar
            denom = float(np.sum(w * dx_c * dx_c))
            if denom <= 1e-15:
                knot_vals[idx_k] = y_bar
            else:
                slope = float(np.sum(w * dx_c * (yw - y_bar)) / denom)
                knot_vals[idx_k] = y_bar - slope * x_bar

        if use_knots:
            y_fit = np.interp(xs, knots, knot_vals)
        else:
            y_fit = knot_vals

        if it + 1 < n_iter:
            resid = ys - y_fit
            scale = float(np.median(np.abs(resid)))
            if scale <= 1e-12:
                break
            u_r = np.clip(np.abs(resid) / (6.0 * scale), 0.0, 1.0)
            rob_w = (1.0 - u_r * u_r) ** 2

    fitted_valid = np.empty(n_valid, dtype=np.float64)
    fitted_valid[order] = y_fit
    out[valid] = fitted_valid
    return out


def estimate_logr_sigma(
    cnlr: np.ndarray,
    arm: np.ndarray | pd.Series | None = None,
) -> float:
    """Estimate local logR noise SD from within-arm consecutive differences."""
    vals = np.asarray(cnlr, dtype=np.float64)
    if vals.size < 2:
        return 1e-4

    if arm is not None:
        arm_arr = np.asarray(arm)
        same_arm = arm_arr[1:] == arm_arr[:-1]
        diffs = np.abs(vals[1:] - vals[:-1])[same_arm]
        if diffs.size == 0:
            diffs = np.abs(vals[1:] - vals[:-1])
    else:
        diffs = np.abs(vals[1:] - vals[:-1])

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1e-4
    sigma = float(np.median(diffs)) / (_MAD_NORMAL_CONST * np.sqrt(2.0))
    return max(sigma, 1e-4)


def compute_cnlr(
    tum_dp: np.ndarray,
    nor_dp: np.ndarray,
    gc: np.ndarray | None = None,
    *,
    lowess_frac: float | None = None,
    lowess_iter: int = 3,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute library-size- and GC-LOWESS-normalized total copy log2 ratio (`cnlr`),
    raw library-centered `cnlr_raw`, and theoretical per-locus variance `cnlr_var`.

    When `settings="R_default"`, matches `mskcc/facets` `procreads.R`:
      `cnlr_raw = log2(1 + tum_dp * tscl) - log2(1 + nor_dp)` with `tscl = sum(nor_dp) / sum(tum_dp)`
      and default `lowess_frac = 0.2`.
    """
    validate_settings(settings)
    t_dp = np.asarray(tum_dp, dtype=np.float64)
    n_dp = np.asarray(nor_dp, dtype=np.float64)
    if t_dp.shape != n_dp.shape or t_dp.ndim != 1 or t_dp.size == 0:
        raise ValueError("`tum_dp` and `nor_dp` must be non-empty 1D arrays of equal shape.")
    if np.any(t_dp < 0.0) or np.any(n_dp < 0.0):
        raise ValueError("Depth arrays must be non-negative.")

    eff_frac = (
        float(lowess_frac)
        if lowess_frac is not None
        else (0.20 if settings == "R_default" else 0.5)
    )

    ln2_sq = np.log(2.0) ** 2
    if settings == "R_default":
        sum_t = float(np.sum(t_dp))
        sum_n = float(np.sum(n_dp))
        tscl = (sum_n / sum_t) if sum_t > 0.0 else 1.0
        cnlr_raw = np.log2(1.0 + t_dp * tscl) - np.log2(1.0 + n_dp)
        cnlr_var = np.maximum(
            (1.0 / ln2_sq) * (1.0 / (1.0 + t_dp * tscl) + 1.0 / (1.0 + n_dp)),
            1e-6,
        )
    else:
        l_ratio = np.log2(np.sum(t_dp + 0.5) / np.sum(n_dp + 0.5))
        cnlr_raw = np.log2((t_dp + 0.5) / (n_dp + 0.5)) - l_ratio
        cnlr_var = np.maximum(
            (1.0 / ln2_sq) * (1.0 / (t_dp + 0.5) + 1.0 / (n_dp + 0.5)),
            1e-6,
        )

    cnlr = cnlr_raw.copy()
    if gc is not None:
        gc_arr = np.asarray(gc, dtype=np.float64)
        if gc_arr.shape != t_dp.shape:
            raise ValueError("`gc` must match the shape of `tum_dp`.")
        finite_gc = np.isfinite(gc_arr) & np.isfinite(cnlr_raw)
        if int(np.sum(finite_gc)) >= 10 and float(np.std(gc_arr[finite_gc])) > 1e-8:
            trend = lowess_1d(
                gc_arr[finite_gc],
                cnlr_raw[finite_gc],
                frac=eff_frac,
                n_iter=lowess_iter,
            )
            cnlr[finite_gc] = cnlr_raw[finite_gc] - trend

    return cnlr, cnlr_raw, cnlr_var


def compute_valor(
    tum_alt: np.ndarray,
    tum_ref: np.ndarray,
    nor_alt: np.ndarray | None = None,
    nor_ref: np.ndarray | None = None,
    *,
    het: np.ndarray | None = None,
    unmatched: bool = False,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute allelic log-odds ratio (`valor`) and delta-method variance (`lorvar`).

    Uses Haldane-Anscombe (+0.5) by default; when `settings="R_default"` and
    `unmatched=True`, uses Tukey's (+1/6) count offset matching `procreads.R`.
    Non-heterozygous loci (when `het` is supplied) receive `np.nan`.
    """
    validate_settings(settings)
    ta = np.asarray(tum_alt, dtype=np.float64)
    tr = np.asarray(tum_ref, dtype=np.float64)
    if ta.shape != tr.shape or ta.ndim != 1 or ta.size == 0:
        raise ValueError("`tum_alt` and `tum_ref` must be non-empty 1D arrays of equal shape.")

    if not unmatched:
        if nor_alt is None or nor_ref is None:
            raise ValueError("`nor_alt` and `nor_ref` are required when `unmatched=False`.")
        na = np.asarray(nor_alt, dtype=np.float64)
        nr = np.asarray(nor_ref, dtype=np.float64)
        if na.shape != ta.shape or nr.shape != ta.shape:
            raise ValueError("Normal allele count arrays must match `tum_alt` shape.")
        valor = np.log(((ta + 0.5) * (nr + 0.5)) / ((tr + 0.5) * (na + 0.5)))
        lorvar = (
            1.0 / (ta + 0.5)
            + 1.0 / (tr + 0.5)
            + 1.0 / (na + 0.5)
            + 1.0 / (nr + 0.5)
        )
    else:
        offset = (1.0 / 6.0) if settings == "R_default" else 0.5
        valor = np.log((ta + offset) / (tr + offset))
        lorvar = 1.0 / (ta + offset) + 1.0 / (tr + offset)

    if het is not None:
        het_mask = np.asarray(het, dtype=bool)
        if het_mask.shape != ta.shape:
            raise ValueError("`het` mask must match `tum_alt` shape.")
        valor = np.where(het_mask, valor, np.nan)
        lorvar = np.where(het_mask, lorvar, np.nan)

    return valor, lorvar


def _thin_snps_by_nbhd(
    chrom: np.ndarray,
    pos: np.ndarray,
    het: np.ndarray,
    snp_nbhd: int,
) -> np.ndarray:
    """
    Greedy forward neighborhood thinning per chromosome ensuring consecutive
    retained SNPs are >= `snp_nbhd` bp apart, preferring heterozygous SNPs
    when both homozygous and heterozygous SNPs fall inside the same window.
    """
    n = pos.size
    if snp_nbhd <= 0 or n <= 1:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    last_idx = -1
    last_chrom: object = None
    for i in range(n):
        c = chrom[i]
        p = int(pos[i])
        if c != last_chrom:
            last_chrom = c
            last_idx = i
            keep[i] = True
            continue
        if p - int(pos[last_idx]) >= snp_nbhd:
            last_idx = i
            keep[i] = True
        elif (not bool(het[last_idx])) and bool(het[i]):
            keep[last_idx] = False
            last_idx = i
            keep[i] = True
    return keep


def _resolve_config(
    config: FacetsConfig | None,
    settings: SettingsType,
    kwargs: dict,
) -> FacetsConfig:
    """Merge optional FacetsConfig, `settings`, and keyword overrides cleanly."""
    validate_settings(settings)
    merged = dict(kwargs)
    if config is None:
        merged["settings"] = settings
        return FacetsConfig(**merged)
    eff_settings: SettingsType = (
        config.settings if settings == "default" and config.settings != "default" else settings
    )
    merged["settings"] = eff_settings
    return replace(config, **merged)


def pre_proc_sample(
    pileup: pd.DataFrame,
    *,
    config: FacetsConfig | None = None,
    settings: Literal["default", "R_default"] = "default",
    **kwargs,
) -> PreProcResult:
    """
    Validate, filter, thin, and normalize a tumor-normal SNP pileup DataFrame.
    """
    cfg = _resolve_config(config, settings, kwargs)
    eff_settings = cfg.settings
    df = validate_and_normalize_pileup(pileup, genome=cfg.genome)
    n_raw = len(df)

    depth_mask = (df["nor_dp"] >= cfg.ndepth) & ((df["nor_dp"] + df["tum_dp"]) > 0)
    if cfg.ndepthmax is not None:
        depth_mask = depth_mask & (df["nor_dp"] <= cfg.ndepthmax)

    df = df.loc[depth_mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No SNPs remain after depth filtering "
            f"(ndepth={cfg.ndepth}, ndepthmax={cfg.ndepthmax})."
        )

    eff_thresh = cfg.effective_het_thresh
    if "het" in df.columns:
        het_mask = df["het"].to_numpy(dtype=bool)
    elif not cfg.unmatched:
        nor_vaf = df["nor_alt"].to_numpy(dtype=np.float64) / np.maximum(
            df["nor_dp"].to_numpy(dtype=np.float64), 1.0
        )
        het_mask = (nor_vaf >= eff_thresh) & (nor_vaf <= 1.0 - eff_thresh)
    else:
        tum_dp_arr = df["tum_dp"].to_numpy(dtype=np.float64)
        tum_vaf = df["tum_alt"].to_numpy(dtype=np.float64) / np.maximum(
            tum_dp_arr, 1.0
        )
        het_mask = (tum_vaf >= eff_thresh) & (tum_vaf <= 1.0 - eff_thresh)
        if eff_settings == "R_default":
            het_mask = het_mask & (tum_dp_arr >= 50.0)

    df["het"] = het_mask

    keep_mask = _thin_snps_by_nbhd(
        df["chrom"].to_numpy(),
        df["pos"].to_numpy(dtype=np.int64),
        het_mask,
        cfg.snp_nbhd,
    )
    df = df.loc[keep_mask].reset_index(drop=True)
    het_arr = df["het"].to_numpy(dtype=bool)

    gc_arr = df["gc"].to_numpy(dtype=np.float64)
    finite_gc = np.isfinite(gc_arr)
    gc_corrected = bool(
        int(np.sum(finite_gc)) >= 10 and float(np.std(gc_arr[finite_gc])) > 1e-8
    )

    cnlr, cnlr_raw, cnlr_var = compute_cnlr(
        df["tum_dp"].to_numpy(dtype=np.float64),
        df["nor_dp"].to_numpy(dtype=np.float64),
        gc=gc_arr,
        lowess_frac=cfg.effective_lowess_frac,
        lowess_iter=cfg.lowess_iter,
        settings=eff_settings,
    )
    sigma_logr = estimate_logr_sigma(cnlr, arm=df["arm"].to_numpy())

    valor, lorvar = compute_valor(
        df["tum_alt"].to_numpy(dtype=np.float64),
        df["tum_ref"].to_numpy(dtype=np.float64),
        df["nor_alt"].to_numpy(dtype=np.float64),
        df["nor_ref"].to_numpy(dtype=np.float64),
        het=het_arr,
        unmatched=cfg.unmatched,
        settings=eff_settings,
    )

    logor2_stat = np.where(het_arr, (valor * valor) / lorvar, np.nan)

    df["cnlr"] = cnlr
    df["cnlr_raw"] = cnlr_raw
    df["cnlr_var"] = cnlr_var
    df["valor"] = valor
    df["lorvar"] = lorvar
    df["logor2_stat"] = logor2_stat

    return PreProcResult(
        snps=df,
        genome=cfg.genome,
        n_raw=n_raw,
        n_retained=len(df),
        n_het=int(np.sum(het_arr)),
        sigma_logr=sigma_logr,
        gc_corrected=gc_corrected,
        config=cfg,
    )


preprocess_sample = pre_proc_sample
