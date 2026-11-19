"""
Continuous-BAF ASCAT equations and 2D grid goodness-of-fit solver
(Van Loo et al., PNAS 2010), plus FACETS vs ASCAT concordance comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .em import FacetsResult
from .io import validate_settings
from .segment import SegmentedResult


@dataclass(frozen=True)
class AscatResult:
    """Result container from `fit_ascat` (Van Loo et al., PNAS 2010)."""

    purity: float
    ploidy: float
    goodness_of_fit: float
    psi_bulk: float
    gamma_ascat: float
    segments: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return ASCAT segment-level copy-number DataFrame."""
        return self.segments.copy()

    def summary(self) -> str:
        """Return concise summary of ASCAT fit."""
        return (
            f"ASCAT AscatResult (gamma={self.gamma_ascat:.2f})\n"
            f"  Purity:          {self.purity:.4f}\n"
            f"  Ploidy:          {self.ploidy:.4f}\n"
            f"  Bulk Ploidy:     {self.psi_bulk:.4f}\n"
            f"  Goodness of Fit: {self.goodness_of_fit:.4f}\n"
            f"  Segments:        {len(self.segments)}"
        )


def ascat_logr(
    n_major: np.ndarray | float | int,
    n_minor: np.ndarray | float | int,
    purity: float,
    ploidy: float,
    gamma_ascat: float = 1.0,
) -> np.ndarray | float:
    """
    Van Loo et al. (2010) Eq. 1 forward logR equation:
        r = gamma * log2((2*(1 - rho) + rho*(nA + nB)) / (2*(1 - rho) + rho*psi_t))
    """
    rho = float(purity)
    psi_t = float(ploidy)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"`purity` must be in [0, 1], got {rho}.")
    if psi_t <= 0.0:
        raise ValueError(f"`ploidy` must be > 0, got {psi_t}.")

    scalar_input = np.ndim(n_major) == 0 and np.ndim(n_minor) == 0
    n_a = np.asarray(n_major, dtype=np.float64)
    n_b = np.asarray(n_minor, dtype=np.float64)
    psi_bulk = 2.0 * (1.0 - rho) + rho * psi_t
    num = np.maximum(2.0 * (1.0 - rho) + rho * (n_a + n_b), 1e-6)
    res = float(gamma_ascat) * np.log2(num / max(psi_bulk, 1e-6))
    return float(res) if scalar_input else res


def ascat_baf(
    n_major: np.ndarray | float | int,
    n_minor: np.ndarray | float | int,
    purity: float,
    *,
    minor_allele: bool = True,
) -> np.ndarray | float:
    """
    Van Loo et al. (2010) Eq. 2 forward BAF equation:
        b_minor = (1 - rho + rho * nB) / (2*(1 - rho) + rho*(nA + nB))
    """
    rho = float(purity)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"`purity` must be in [0, 1], got {rho}.")

    scalar_input = np.ndim(n_major) == 0 and np.ndim(n_minor) == 0
    n_a = np.asarray(n_major, dtype=np.float64)
    n_b = np.asarray(n_minor, dtype=np.float64)
    denom = np.maximum(2.0 * (1.0 - rho) + rho * (n_a + n_b), 1e-8)
    b_minor = (1.0 - rho + rho * n_b) / denom
    out = b_minor if minor_allele else (1.0 - b_minor)
    return float(out) if scalar_input else out


def ascat_invert_cn(
    logr: np.ndarray | float,
    baf: np.ndarray | float,
    purity: float,
    ploidy: float,
    gamma_ascat: float = 1.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Exact inverse continuous copy-number equations (Van Loo et al., 2010):
    returns `(n_major_cont, n_minor_cont)` from `(logR, baf)` given `(purity, ploidy)`.
    """
    rho = float(purity)
    psi_t = float(ploidy)
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"`purity` must be in (0, 1], got {rho}.")
    if psi_t <= 0.0:
        raise ValueError(f"`ploidy` must be > 0, got {psi_t}.")
    if gamma_ascat <= 0.0:
        raise ValueError(f"`gamma_ascat` must be > 0, got {gamma_ascat}.")

    scalar_input = np.ndim(logr) == 0 and np.ndim(baf) == 0
    r_arr = np.asarray(logr, dtype=np.float64)
    b_arr = np.asarray(baf, dtype=np.float64)
    b_fold = np.minimum(b_arr, 1.0 - b_arr)

    psi_bulk = 2.0 * (1.0 - rho) + rho * psi_t
    total_term = psi_bulk * np.power(2.0, r_arr / float(gamma_ascat))
    n_minor_cont = (rho - 1.0 + b_fold * total_term) / rho
    n_major_cont = (rho - 1.0 + (1.0 - b_fold) * total_term) / rho
    if scalar_input:
        return float(n_major_cont), float(n_minor_cont)
    return n_major_cont, n_minor_cont


def _extract_ascat_segments(
    segments_or_result: pd.DataFrame | SegmentedResult | FacetsResult,
) -> pd.DataFrame:
    """Normalize input segments DataFrame / FACETS result into standard columns."""
    if isinstance(segments_or_result, (SegmentedResult, FacetsResult)):
        df = segments_or_result.segments.copy()
    elif isinstance(segments_or_result, pd.DataFrame):
        df = segments_or_result.copy()
    else:
        raise ValueError(
            "`segments_or_result` must be a DataFrame, SegmentedResult, or FacetsResult."
        )

    if len(df) == 0:
        raise ValueError("Segment table cannot be empty.")

    col_map = {str(c).lower(): c for c in df.columns}
    if "logr" in col_map:
        logr = df[col_map["logr"]].to_numpy(dtype=np.float64)
    elif "cnlr_mean" in col_map:
        logr = df[col_map["cnlr_mean"]].to_numpy(dtype=np.float64)
    elif "cnlr_median" in col_map:
        logr = df[col_map["cnlr_median"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            "Input must contain 'logr', 'cnlr_mean', or 'cnlr_median' column."
        )

    if "baf" in col_map:
        baf = df[col_map["baf"]].to_numpy(dtype=np.float64)
        baf = np.clip(np.minimum(baf, 1.0 - baf), 1e-4, 0.5)
    elif "mafr" in col_map:
        mafr = np.maximum(df[col_map["mafr"]].to_numpy(dtype=np.float64), 0.0)
        mu_o = np.sqrt(mafr)
        baf = 1.0 / (1.0 + np.exp(mu_o))
    else:
        raise ValueError("Input must contain 'baf' or 'mafR' column.")

    chrom = (
        df[col_map["chrom"]].astype(str).tolist()
        if "chrom" in col_map
        else ["chr1"] * len(df)
    )
    if "start" in col_map and "end" in col_map:
        start = df[col_map["start"]].to_numpy(dtype=np.int64)
        end = df[col_map["end"]].to_numpy(dtype=np.int64)
        lengths = np.maximum((end - start + 1).astype(np.float64), 1.0)
    else:
        start = np.arange(1, len(df) + 1, dtype=np.int64)
        end = start.copy()
        lengths = np.ones(len(df), dtype=np.float64)

    if "weight" in col_map:
        weights = np.maximum(df[col_map["weight"]].to_numpy(dtype=np.float64), 1e-6)
    elif "num_mark" in col_map:
        weights = np.maximum(df[col_map["num_mark"]].to_numpy(dtype=np.float64), 1.0)
    else:
        weights = lengths

    if "nhet" in col_map:
        nhet = df[col_map["nhet"]].to_numpy(dtype=np.int64)
    else:
        nhet = np.full(len(df), 100, dtype=np.int64)

    return pd.DataFrame(
        {
            "chrom": chrom,
            "start": start,
            "end": end,
            "logr": logr,
            "baf": baf,
            "weight": weights,
            "nhet": nhet,
        }
    )


def _find_grid_local_minima(dist_g: np.ndarray) -> np.ndarray:
    """Return boolean mask of 2D 3x3 Moore-neighborhood local minima on `dist_g`."""
    nr, np_ = dist_g.shape
    if nr < 3 or np_ < 3:
        return np.ones_like(dist_g, dtype=bool)
    padded = np.pad(dist_g, pad_width=1, mode="edge")
    is_min = np.ones_like(dist_g, dtype=bool)
    for dr in (-1, 0, 1):
        for dp in (-1, 0, 1):
            if dr == 0 and dp == 0:
                continue
            nbr = padded[1 + dr : 1 + dr + nr, 1 + dp : 1 + dp + np_]
            is_min &= dist_g <= (nbr + 1e-12)
    return is_min


def fit_ascat(
    segments_or_result: pd.DataFrame | SegmentedResult | FacetsResult,
    *,
    gamma_ascat: float | None = None,
    min_nhet: int = 15,
    purity_grid: np.ndarray | None = None,
    ploidy_grid: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> AscatResult:
    """
    Solve for tumor purity (rho), ploidy (psi_t), and integer allele-specific
    copy numbers (n_major, n_minor) via Van Loo et al. (2010) 2D distance grid
    search + Nelder-Mead local refinement.

    - `settings="default"`: `gamma_ascat=1.0` default, `rho` in `[0.10, 1.00]`, uniform segment weights.
    - `settings="R_default"`: `VanLoo-lab/ascat` v3.2.0 defaults (`gamma=0.55`, `rho` grid `[0.10, 1.05]`,
      `0.05` weight on allelically balanced segments, and 4-tier cascading distance acceptance).
    """
    validate_settings(settings)
    eff_gamma = (
        float(gamma_ascat)
        if gamma_ascat is not None
        else (0.55 if settings == "R_default" else 1.0)
    )
    if eff_gamma <= 0.0:
        raise ValueError(f"`gamma_ascat` must be > 0, got {eff_gamma}.")

    seg_df = _extract_ascat_segments(segments_or_result)
    logr = seg_df["logr"].to_numpy(dtype=np.float64)
    baf = seg_df["baf"].to_numpy(dtype=np.float64)
    weights = seg_df["weight"].to_numpy(dtype=np.float64)
    nhet = seg_df["nhet"].to_numpy(dtype=np.int64)

    fit_mask = (
        np.isfinite(logr)
        & np.isfinite(baf)
        & (nhet >= min_nhet)
        & (weights > 0.0)
    )
    if not np.any(fit_mask):
        fit_mask = np.isfinite(logr) & np.isfinite(baf) & (weights > 0.0)
    if not np.any(fit_mask):
        raise ValueError("No valid finite segments available for ASCAT fitting.")

    r_fit = logr[fit_mask]
    b_fit = baf[fit_mask]
    w_fit = weights[fit_mask]
    w_span_norm = w_fit / float(np.sum(w_fit))
    pow2_r = np.power(2.0, r_fit / eff_gamma)

    if settings == "R_default":
        is_bal = np.abs(b_fit - 0.5) < 0.02
        w_ascat = w_fit * np.where(is_bal, 0.05, 1.0)
        if float(np.sum(w_ascat)) <= 0.0:
            w_ascat = w_fit.copy()
        w_norm = w_ascat / float(np.sum(w_ascat))

        rhos = (
            np.asarray(purity_grid, dtype=np.float64)
            if purity_grid is not None
            else np.round(np.arange(0.10, 1.0501, 0.01), 4)
        )
        psis = (
            np.asarray(ploidy_grid, dtype=np.float64)
            if ploidy_grid is not None
            else np.round(np.arange(1.00, 5.5001, 0.05), 4)
        )

        rr = rhos[:, None, None]
        pp = psis[None, :, None]
        psi_bulk_grid = 2.0 * (1.0 - rr) + rr * pp
        tot_grid = psi_bulk_grid * pow2_r[None, None, :]
        nb_grid = (rr - 1.0 + b_fit[None, None, :] * tot_grid) / rr
        na_grid = (rr - 1.0 + (1.0 - b_fit[None, None, :]) * tot_grid) / rr

        nb_int_g = np.maximum(np.rint(nb_grid), 0.0)
        na_int_g = np.maximum(np.rint(na_grid), 0.0)
        dist_g = np.sum(
            w_norm[None, None, :]
            * ((na_grid - na_int_g) ** 2 + (nb_grid - nb_int_g) ** 2),
            axis=2,
        )
        neg_frac_g = np.sum(
            w_span_norm[None, None, :] * ((nb_grid < -0.30) | (na_grid < -0.30)),
            axis=2,
        )
        homdel_frac_g = np.sum(
            w_span_norm[None, None, :] * ((nb_int_g + na_int_g) == 0),
            axis=2,
        )
        psi_impl_g = np.sum(
            w_span_norm[None, None, :] * (na_int_g + nb_int_g),
            axis=2,
        )
        score_g = (
            dist_g
            + 0.02 * ((psi_impl_g - psis[None, :]) ** 2)
            + 2e-4 * np.maximum(0.0, psis[None, :] - 2.2)
        )

        local_min = _find_grid_local_minima(score_g)
        rho_2d = np.broadcast_to(rhos[:, None], score_g.shape)

        # 4-tier cascading ASCAT acceptance rules across rho in [0.10, 1.05]
        tier_masks = [
            # Tier 1: local minima in [0.10, 1.05], physiological ploidy [1.5, 4.8], strict non-neg/homdel
            local_min
            & (rho_2d <= 1.05 + 1e-9)
            & (psi_impl_g >= 1.5)
            & (psi_impl_g <= 4.8)
            & (neg_frac_g <= 0.005)
            & (homdel_frac_g <= 0.05),
            # Tier 2: local minima in [0.10, 1.05], physiological ploidy [1.5, 4.8], relaxed homdel/neg
            local_min
            & (rho_2d <= 1.05 + 1e-9)
            & (psi_impl_g >= 1.5)
            & (psi_impl_g <= 4.8)
            & (neg_frac_g <= 0.03)
            & (homdel_frac_g <= 0.15),
            # Tier 3: expanded extreme ploidy [1.0, 6.0]
            local_min
            & (rho_2d <= 1.05 + 1e-9)
            & (psi_impl_g >= 1.0)
            & (psi_impl_g <= 6.0)
            & (neg_frac_g <= 0.05),
            # Tier 4: unconstrained fallback across entire grid
            np.ones_like(score_g, dtype=bool),
        ]

        chosen_cands: list[tuple[float, float]] = []
        for t_mask in tier_masks:
            if np.any(t_mask):
                cand_rc = np.argwhere(t_mask)
                cand_scores = score_g[t_mask]
                order_t = np.argsort(cand_scores, kind="mergesort")
                for idx_o in order_t:
                    ri, pi = int(cand_rc[idx_o, 0]), int(cand_rc[idx_o, 1])
                    r0, p0 = float(rhos[ri]), float(psis[pi])
                    if not any(
                        abs(r0 - tr) < 0.08 and abs(p0 - tp) < 0.25
                        for tr, tp in chosen_cands
                    ):
                        chosen_cands.append((r0, p0))
                    if len(chosen_cands) >= 3:
                        break
                break

        def _eval_obj_r(rho_val: float, psi_val: float) -> float:
            if rho_val < 0.05 or rho_val > 1.05 or psi_val < 0.8 or psi_val > 6.5:
                return 1e9
            psi_b = 2.0 * (1.0 - rho_val) + rho_val * psi_val
            tot = psi_b * pow2_r
            nb = (rho_val - 1.0 + b_fit * tot) / rho_val
            na = (rho_val - 1.0 + (1.0 - b_fit) * tot) / rho_val
            neg_f = float(np.sum(w_span_norm[(nb < -0.35) | (na < -0.35)]))
            if neg_f > 0.05:
                return 1e6 + 1e3 * neg_f
            nb_i = np.maximum(np.rint(nb), 0.0)
            na_i = np.maximum(np.rint(na), 0.0)
            d_val = float(np.sum(w_norm * ((na - na_i) ** 2 + (nb - nb_i) ** 2)))
            psi_i = float(np.sum(w_span_norm * (na_i + nb_i)))
            return (
                d_val
                + 0.02 * ((psi_i - psi_val) ** 2)
                + 2e-4 * max(0.0, psi_val - 2.2)
            )

        best_params = (
            float(np.clip(chosen_cands[0][0], 0.05, 1.0)),
            float(np.clip(chosen_cands[0][1], 0.8, 6.5)),
        )
        best_val = np.inf
        for r0, p0 in chosen_cands:
            res = minimize(
                lambda x: _eval_obj_r(float(x[0]), float(x[1])),
                x0=np.array([r0, p0], dtype=np.float64),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
            )
            if float(res.fun) < best_val:
                best_val = float(res.fun)
                best_params = (
                    float(np.clip(res.x[0], 0.05, 1.0)),
                    float(np.clip(res.x[1], 0.8, 6.5)),
                )
    else:
        w_norm = w_span_norm
        rhos = (
            np.asarray(purity_grid, dtype=np.float64)
            if purity_grid is not None
            else np.linspace(0.10, 1.00, 46)
        )
        psis = (
            np.asarray(ploidy_grid, dtype=np.float64)
            if ploidy_grid is not None
            else np.linspace(1.50, 5.50, 81)
        )

        def _eval_objective(
            rho_val: float, psi_val: float, penalize_neg: bool = True
        ) -> float:
            if rho_val < 0.05 or rho_val > 1.0 or psi_val < 1.0 or psi_val > 6.5:
                return 1e9
            psi_bulk = 2.0 * (1.0 - rho_val) + rho_val * psi_val
            tot = psi_bulk * pow2_r
            nb = (rho_val - 1.0 + b_fit * tot) / rho_val
            na = (rho_val - 1.0 + (1.0 - b_fit) * tot) / rho_val

            neg_frac = float(np.sum(w_norm[(nb < -0.35) | (na < -0.35)]))
            if penalize_neg and neg_frac > 0.05:
                return 1e6 + 1e3 * neg_frac

            nb_int = np.maximum(np.rint(nb), 0.0)
            na_int = np.maximum(np.rint(na), 0.0)
            dist = float(np.sum(w_norm * ((na - na_int) ** 2 + (nb - nb_int) ** 2)))
            psi_impl = float(np.sum(w_norm * (na_int + nb_int)))
            # Mild self-consistency + parsimony tie-breaker against 2x doubling degeneracy
            return (
                dist
                + 0.02 * ((psi_impl - psi_val) ** 2)
                + 2e-4 * max(0.0, psi_val - 2.2)
            )

        # Vectorized 2D grid evaluation across (R, P, S)
        rr = rhos[:, None, None]
        pp = psis[None, :, None]
        psi_bulk_grid = 2.0 * (1.0 - rr) + rr * pp
        tot_grid = psi_bulk_grid * pow2_r[None, None, :]
        nb_grid = (rr - 1.0 + b_fit[None, None, :] * tot_grid) / rr
        na_grid = (rr - 1.0 + (1.0 - b_fit[None, None, :]) * tot_grid) / rr

        nb_int_g = np.maximum(np.rint(nb_grid), 0.0)
        na_int_g = np.maximum(np.rint(na_grid), 0.0)
        neg_frac_g = np.sum(
            w_norm[None, None, :] * ((nb_grid < -0.35) | (na_grid < -0.35)), axis=2
        )
        dist_g = np.sum(
            w_norm[None, None, :]
            * ((na_grid - na_int_g) ** 2 + (nb_grid - nb_int_g) ** 2),
            axis=2,
        )
        psi_impl_g = np.sum(w_norm[None, None, :] * (na_int_g + nb_int_g), axis=2)
        obj_grid = (
            dist_g
            + 0.02 * ((psi_impl_g - psis[None, :]) ** 2)
            + 2e-4 * np.maximum(0.0, psis[None, :] - 2.2)
        )
        obj_grid = np.where(neg_frac_g > 0.05, 1e6 + 1e3 * neg_frac_g, obj_grid)

        # Pick top 3 well-separated grid minima and refine via Nelder-Mead
        flat_order = np.argsort(obj_grid.ravel(), kind="mergesort")
        top_starts: list[tuple[float, float]] = []
        for flat_idx in flat_order:
            ri, pi = divmod(int(flat_idx), psis.size)
            r0, p0 = float(rhos[ri]), float(psis[pi])
            if not any(
                abs(r0 - tr) < 0.08 and abs(p0 - tp) < 0.25 for tr, tp in top_starts
            ):
                top_starts.append((r0, p0))
            if len(top_starts) >= 3:
                break

        best_params = top_starts[0]
        best_val = np.inf
        for r0, p0 in top_starts:
            res = minimize(
                lambda x: _eval_objective(float(x[0]), float(x[1]), penalize_neg=True),
                x0=np.array([r0, p0], dtype=np.float64),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
            )
            if float(res.fun) < best_val:
                best_val = float(res.fun)
                best_params = (
                    float(np.clip(res.x[0], 0.05, 1.0)),
                    float(np.clip(res.x[1], 1.0, 6.5)),
                )

    rho_hat, psi_hat_init = best_params
    na_cont, nb_cont = ascat_invert_cn(
        logr, baf, rho_hat, psi_hat_init, gamma_ascat=eff_gamma
    )
    na_int = np.maximum(np.rint(na_cont), 0.0).astype(np.int64)
    nb_int = np.maximum(np.rint(nb_cont), 0.0).astype(np.int64)
    # Enforce n_major >= n_minor
    n_major = np.maximum(na_int, nb_int)
    n_minor = np.minimum(na_int, nb_int)
    tcn = n_major + n_minor

    # Compute raw distance metric on fit segments for goodness_of_fit
    na_fit_c, nb_fit_c = ascat_invert_cn(
        r_fit, b_fit, rho_hat, psi_hat_init, gamma_ascat=eff_gamma
    )
    d_raw = float(
        np.sum(
            w_norm
            * (
                (na_fit_c - np.maximum(np.rint(na_fit_c), 0.0)) ** 2
                + (nb_fit_c - np.maximum(np.rint(nb_fit_c), 0.0)) ** 2
            )
        )
    )
    gof = float(np.clip(1.0 - d_raw, 0.0, 1.0))

    bp_weights = np.maximum(
        (seg_df["end"].to_numpy(dtype=np.float64) - seg_df["start"].to_numpy(dtype=np.float64) + 1.0),
        1.0,
    )
    ploidy_final = float(np.sum(bp_weights * tcn) / np.sum(bp_weights))
    psi_bulk = float(2.0 * (1.0 - rho_hat) + rho_hat * ploidy_final)

    out_df = pd.DataFrame(
        {
            "chrom": seg_df["chrom"],
            "start": seg_df["start"],
            "end": seg_df["end"],
            "logr": logr,
            "baf": baf,
            "n_major": n_major,
            "n_minor": n_minor,
            "tcn": tcn,
            "n_major_cont": np.maximum(na_cont, nb_cont),
            "n_minor_cont": np.minimum(na_cont, nb_cont),
            "loh": (n_minor == 0) & (tcn >= 1),
        }
    )

    return AscatResult(
        purity=rho_hat,
        ploidy=ploidy_final,
        goodness_of_fit=gof,
        psi_bulk=psi_bulk,
        gamma_ascat=eff_gamma,
        segments=out_df,
    )


run_ascat = fit_ascat


def compare_facets_ascat(
    facets_res: FacetsResult,
    ascat_res: AscatResult,
) -> pd.DataFrame:
    """
    Compare FACETS and ASCAT sample-level purity/ploidy and segment-level
    integer copy-number concordance.
    """
    f_segs = facets_res.segments
    a_segs = ascat_res.segments
    if len(f_segs) != len(a_segs):
        raise ValueError(
            f"Segment count mismatch: FACETS={len(f_segs)} vs ASCAT={len(a_segs)}."
        )

    weights = np.maximum(
        (f_segs["end"].to_numpy(dtype=np.float64) - f_segs["start"].to_numpy(dtype=np.float64) + 1.0),
        1.0,
    )
    tot_w = float(np.sum(weights))
    tcn_agree = float(
        np.sum(
            weights[
                f_segs["tcn"].to_numpy(dtype=np.int64)
                == a_segs["tcn"].to_numpy(dtype=np.int64)
            ]
        )
        / tot_w
    )
    lcn_agree = float(
        np.sum(
            weights[
                f_segs["c_minor"].to_numpy(dtype=np.int64)
                == a_segs["n_minor"].to_numpy(dtype=np.int64)
            ]
        )
        / tot_w
    )
    loh_agree = float(
        np.sum(
            weights[
                f_segs["loh"].to_numpy(dtype=bool)
                == a_segs["loh"].to_numpy(dtype=bool)
            ]
        )
        / tot_w
    )

    f_purity = np.nan if facets_res.purity is None else float(facets_res.purity)
    rows: list[dict[str, Any]] = [
        {
            "facets_purity": f_purity,
            "ascat_purity": float(ascat_res.purity),
            "purity_abs_diff": (
                abs(f_purity - float(ascat_res.purity))
                if np.isfinite(f_purity)
                else np.nan
            ),
            "facets_ploidy": float(facets_res.ploidy),
            "ascat_ploidy": float(ascat_res.ploidy),
            "ploidy_abs_diff": abs(float(facets_res.ploidy) - float(ascat_res.ploidy)),
            "tcn_bp_concordance": tcn_agree,
            "lcn_bp_concordance": lcn_agree,
            "loh_bp_concordance": loh_agree,
            "ascat_goodness_of_fit": float(ascat_res.goodness_of_fit),
        }
    ]
    return pd.DataFrame(rows)
