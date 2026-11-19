"""Core GRAPPLE robust adjusted profile-likelihood MR, multimodal pathway discovery, and MVMR."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import optimize, stats

from .loss import (
    LossName,
    SettingsMode,
    psi_deriv,
    psi_loss,
    resolve_loss_params,
    rho_loss,
    robust_inlier_weight,
    validate_settings,
)
from .preprocessing import estimate_cor_mat, select_grapple_instruments


@dataclass(frozen=True)
class GrappleMode:
    """Single local mode (causal or pleiotropic pathway) of the GRAPPLE profile likelihood."""

    mode_id: int
    beta_hat: float
    beta_se: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    profile_llk: float
    prominence: float
    n_inliers: int
    marker_snps: list[str]
    marker_scores: dict[str, float]


@dataclass(frozen=True)
class GrappleResult:
    """Output of univariable GRAPPLE robust profile-likelihood MR and mode diagnosis."""

    beta_hat: float
    beta_se: float
    z_score: float
    p_value: float
    n_modes: int
    is_multimodal: bool
    modes: list[GrappleMode]
    grid_beta: np.ndarray
    grid_llk: np.ndarray
    cor_xy: float
    loss_function: LossName
    k: float
    prominence_threshold: float
    n_instruments: int
    snp_ids: list[str]
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary DataFrame across all detected modes."""
        rows = []
        for m in self.modes:
            rows.append(
                {
                    "mode_id": m.mode_id,
                    "beta_hat": m.beta_hat,
                    "beta_se": m.beta_se,
                    "z_score": m.z_score,
                    "p_value": m.p_value,
                    "ci_low": m.ci_low,
                    "ci_high": m.ci_high,
                    "profile_llk": m.profile_llk,
                    "prominence": m.prominence,
                    "n_inliers": m.n_inliers,
                    "top_markers": ",".join(m.marker_snps[:5]),
                }
            )
        return pd.DataFrame(rows)

    def marker_dataframe(self) -> pd.DataFrame:
        """Return tidy DataFrame of top marker SNPs driving each detected mode."""
        rows = []
        for m in self.modes:
            for rank, snp in enumerate(m.marker_snps, start=1):
                rows.append(
                    {
                        "mode_id": m.mode_id,
                        "beta_hat": m.beta_hat,
                        "rank": rank,
                        "snp": snp,
                        "marker_score": m.marker_scores[snp],
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary of GRAPPLE estimation and multimodality diagnosis."""
        lines = [
            f"GRAPPLE Summary (settings={self.settings!r}, loss={self.loss_function!r}, k={self.k:.3f}, "
            f"M={self.n_instruments}, modes={self.n_modes})",
        ]
        for m in self.modes:
            lines.append(
                f"  Mode {m.mode_id}: beta={m.beta_hat:+.4f} (SE={m.beta_se:.4f}, p={m.p_value:.3e}, "
                f"llk={m.profile_llk:.2f}, inliers={m.n_inliers}, markers={m.marker_snps[:3]})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class GrappleMVMRResult:
    """Output of overlap-aware multivariable GRAPPLE (`grapple_mvmr`)."""

    beta_hat: np.ndarray
    beta_se: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    cov_beta: np.ndarray
    exposure_names: list[str]
    profile_llk: float
    converged: bool
    inlier_weights: np.ndarray
    q_stat: float
    q_pval: float
    cor_mat: np.ndarray
    loss_function: LossName
    k: float
    n_instruments: int
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return multivariable causal effect table across all exposures."""
        return pd.DataFrame(
            {
                "exposure": self.exposure_names,
                "beta_hat": self.beta_hat,
                "beta_se": self.beta_se,
                "z_score": self.z_scores,
                "p_value": self.p_values,
                "ci_low": self.beta_hat - 1.959963984540054 * self.beta_se,
                "ci_high": self.beta_hat + 1.959963984540054 * self.beta_se,
            }
        )

    def summary(self) -> str:
        """Human-readable summary of multivariable GRAPPLE results."""
        lines = [
            f"GRAPPLE MVMR Summary (settings={self.settings!r}, loss={self.loss_function!r}, "
            f"k={self.k:.3f}, M={self.n_instruments}, P={len(self.exposure_names)})"
        ]
        for i, name in enumerate(self.exposure_names):
            lines.append(
                f"  {name}: beta={self.beta_hat[i]:+.4f} (SE={self.beta_se[i]:.4f}, "
                f"Z={self.z_scores[i]:+.2f}, p={self.p_values[i]:.3e})"
            )
        return "\n".join(lines)


def _profile_llk_grid(
    beta_grid: np.ndarray,
    bx: np.ndarray,
    sx: np.ndarray,
    by: np.ndarray,
    sy: np.ndarray,
    cor_xy: float,
    loss: LossName,
    k: float,
) -> np.ndarray:
    """Vectorized evaluation of profile log-likelihood across a 1D beta grid."""
    b = beta_grid[None, :]  # (1, G)
    v = (
        sy[:, None] ** 2
        + (b * b) * (sx[:, None] ** 2)
        - 2.0 * b * cor_xy * (sx[:, None] * sy[:, None])
    )
    t = (by[:, None] - b * bx[:, None]) / np.sqrt(np.maximum(v, 1e-30))
    return -np.sum(rho_loss(t, loss, k), axis=0)


def _profile_llk_and_derivs_scalar(
    beta: float,
    bx: np.ndarray,
    sx: np.ndarray,
    by: np.ndarray,
    sy: np.ndarray,
    cor_xy: float,
    loss: LossName,
    k: float,
) -> tuple[float, float, float, float]:
    """Return ``(llk, score, d_score_dbeta, sandwich_se)`` at scalar ``beta``."""
    sx2 = sx * sx
    sy2 = sy * sy
    sxsy = sx * sy

    v = np.maximum(sy2 + (beta * beta) * sx2 - 2.0 * beta * cor_xy * sxsy, 1e-30)
    sqrt_v = np.sqrt(v)
    r = by - beta * bx
    t = r / sqrt_v

    # Exact orthogonalized first and second derivatives of t_j(beta)
    u = bx * (sy2 - beta * cor_xy * sxsy) + by * (beta * sx2 - cor_xy * sxsy)
    dt = -u / (v * sqrt_v)

    vp = 2.0 * beta * sx2 - 2.0 * cor_xy * sxsy
    up = -cor_xy * sxsy * bx + sx2 * by
    d2t = -up / (v * sqrt_v) + 1.5 * u * vp / (v * v * sqrt_v)

    rho_v = rho_loss(t, loss, k)
    psi_v = psi_loss(t, loss, k)
    psip_v = psi_deriv(t, loss, k)

    llk = -float(np.sum(rho_v))
    score_j = -psi_v * dt
    score = float(np.sum(score_j))

    # d(score)/d(beta) = d^2(llk)/d(beta)^2 = -sum_j [psi'(t)*(dt)^2 + psi(t)*d2t]
    neg_d2_llk = float(np.sum(psip_v * (dt * dt) + psi_v * d2t))
    d_score = -neg_d2_llk

    w_in = robust_inlier_weight(t, loss, k)
    m_in = float(np.sum(w_in > 1e-3))
    df_scale = m_in / max(m_in - 1.0, 1.0) if m_in >= 2.0 else 1.0
    meat = df_scale * float(np.sum(score_j * score_j))

    # Fallback to Gauss-Newton bread only if curvature at boundary is non-positive
    bread = neg_d2_llk if neg_d2_llk > 1e-10 else max(float(np.sum(psip_v * dt * dt)), 1e-10)
    var_hat = max(meat / (bread * bread), 1e-16)
    return llk, score, d_score, float(np.sqrt(var_hat))


def _filter_modes_by_prominence(
    peaks: list[tuple[float, float]],
    eval_llk,
    min_prominence: float,
) -> list[tuple[float, float, float]]:
    """1D topological persistence pruning of local maxima by valley drop prominence.

    Returns list of ``(beta_hat, llk_hat, prominence)`` sorted by ``beta_hat``.
    """
    if not peaks:
        return []
    pts = sorted(peaks, key=lambda x: x[0])
    if len(pts) == 1:
        return [(pts[0][0], pts[0][1], float("inf"))]

    # Valley minimum between each adjacent pair of peaks
    valleys: list[float] = []
    for i in range(len(pts) - 1):
        b_l, _ = pts[i]
        b_r, _ = pts[i + 1]
        opt_v = optimize.minimize_scalar(
            lambda b: eval_llk(float(b)), bounds=(b_l, b_r), method="bounded"
        )
        valleys.append(float(opt_v.fun))

    betas = [p[0] for p in pts]
    llks = [p[1] for p in pts]

    while len(betas) > 1:
        # Compute drop from lower peak of each adjacent pair to the valley between them
        drops = [min(llks[i], llks[i + 1]) - valleys[i] for i in range(len(valleys))]
        min_idx = int(np.argmin(drops))
        if drops[min_idx] >= min_prominence:
            break
        # Merge the lower peak into the higher peak
        if llks[min_idx] < llks[min_idx + 1]:
            # Remove peak min_idx
            if min_idx > 0:
                valleys[min_idx - 1] = min(valleys[min_idx - 1], valleys[min_idx])
            betas.pop(min_idx)
            llks.pop(min_idx)
            valleys.pop(min_idx)
        else:
            # Remove peak min_idx + 1
            if min_idx + 1 < len(valleys):
                valleys[min_idx] = min(valleys[min_idx], valleys[min_idx + 1])
                valleys.pop(min_idx + 1)
            else:
                valleys.pop(min_idx)
            betas.pop(min_idx + 1)
            llks.pop(min_idx + 1)

    if len(betas) == 1:
        return [(betas[0], llks[0], float("inf"))]

    # Compute topological prominence of each surviving peak
    global_max_idx = int(np.argmax(llks))
    out: list[tuple[float, float, float]] = []
    for i in range(len(betas)):
        if i == global_max_idx:
            # Prominence relative to second-highest mode's separating valley
            prom = llks[i] - min(valleys)
            out.append((betas[i], llks[i], float(prom)))
            continue
        # Highest valley on path to any strictly higher peak on left / right
        left_higher = [j for j in range(0, i) if llks[j] > llks[i]]
        right_higher = [j for j in range(i + 1, len(betas)) if llks[j] > llks[i]]
        saddle_candidates = []
        if left_higher:
            j_l = max(left_higher)
            saddle_candidates.append(min(valleys[j_l:i]))
        if right_higher:
            j_r = min(right_higher)
            saddle_candidates.append(min(valleys[i:j_r]))
        saddle = max(saddle_candidates) if saddle_candidates else min(valleys)
        out.append((betas[i], llks[i], float(llks[i] - saddle)))

    return out


def grapple_robust_est(
    bx: ArrayLike,
    sx: ArrayLike,
    by: ArrayLike,
    sy: ArrayLike,
    cor_xy: float = 0.0,
    loss_function: str = "tukey",
    k: float | None = None,
    diagnosis: bool = True,
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    n_grid: int = 500,
    snp_ids: list[str] | None = None,
    n_markers: int = 10,
    settings: SettingsMode = "default",
) -> GrappleResult:
    """Estimate causal effects and multimodal pleiotropic pathways via GRAPPLE.

    Implements Wang et al., *PNAS* 2021 (`GRAPPLE::grappleRobustEst`).
    """
    loss, k_val, mode = resolve_loss_params(loss_function, k, settings=settings)
    if not (-1.0 < cor_xy < 1.0):
        raise ValueError("cor_xy must lie strictly in (-1, 1).")
    if grid_min >= grid_max:
        raise ValueError("grid_min must be strictly less than grid_max.")
    if n_grid < 25:
        raise ValueError("n_grid must be >= 25.")

    bx_arr = np.asarray(bx, dtype=float).ravel()
    sx_arr = np.asarray(sx, dtype=float).ravel()
    by_arr = np.asarray(by, dtype=float).ravel()
    sy_arr = np.asarray(sy, dtype=float).ravel()
    if not (bx_arr.shape == sx_arr.shape == by_arr.shape == sy_arr.shape):
        raise ValueError("bx, sx, by, and sy must have identical 1D shapes.")
    if bx_arr.size < 3:
        raise ValueError("At least 3 genetic instruments are required for GRAPPLE.")
    if np.any(sx_arr <= 0) or np.any(sy_arr <= 0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")
    if not (
        np.all(np.isfinite(bx_arr))
        and np.all(np.isfinite(sx_arr))
        and np.all(np.isfinite(by_arr))
        and np.all(np.isfinite(sy_arr))
    ):
        raise ValueError("Summary statistics contain non-finite values.")

    if snp_ids is None:
        ids = [f"rs{i}" for i in range(bx_arr.size)]
    else:
        if len(snp_ids) != bx_arr.size:
            raise ValueError("snp_ids length must match bx.")
        ids = [str(s) for s in snp_ids]

    beta_grid = np.linspace(float(grid_min), float(grid_max), int(n_grid))
    grid_llk = _profile_llk_grid(
        beta_grid, bx_arr, sx_arr, by_arr, sy_arr, cor_xy, loss, k_val
    )

    def eval_llk_scalar(b: float) -> float:
        ll, _, _, _ = _profile_llk_and_derivs_scalar(
            b, bx_arr, sx_arr, by_arr, sy_arr, cor_xy, loss, k_val
        )
        return ll

    def eval_score_scalar(b: float) -> float:
        _, sc, _, _ = _profile_llk_and_derivs_scalar(
            b, bx_arr, sx_arr, by_arr, sy_arr, cor_xy, loss, k_val
        )
        return sc

    # Detect all interior local maxima on the grid and refine via Brent score root-solving
    cand_brackets: list[tuple[float, float]] = []
    for idx in range(1, beta_grid.size - 1):
        if grid_llk[idx] >= grid_llk[idx - 1] and grid_llk[idx] >= grid_llk[idx + 1]:
            if grid_llk[idx] > grid_llk[idx - 1] or grid_llk[idx] > grid_llk[idx + 1]:
                cand_brackets.append((float(beta_grid[idx - 1]), float(beta_grid[idx + 1])))

    refined_peaks: list[tuple[float, float]] = []
    for b_l, b_r in cand_brackets:
        s_l = eval_score_scalar(b_l)
        s_r = eval_score_scalar(b_r)
        if s_l >= 0.0 >= s_r and (s_l > 0.0 or s_r < 0.0):
            try:
                b_hat = float(optimize.brentq(eval_score_scalar, b_l, b_r, xtol=1e-10))
            except ValueError:
                opt = optimize.minimize_scalar(
                    lambda b: -eval_llk_scalar(float(b)),
                    bounds=(b_l, b_r),
                    method="bounded",
                )
                b_hat = float(opt.x)
        else:
            opt = optimize.minimize_scalar(
                lambda b: -eval_llk_scalar(float(b)),
                bounds=(b_l, b_r),
                method="bounded",
            )
            b_hat = float(opt.x)

        ll_hat, _, d_sc, _ = _profile_llk_and_derivs_scalar(
            b_hat, bx_arr, sx_arr, by_arr, sy_arr, cor_xy, loss, k_val
        )
        if d_sc < 0.0:  # strict local maximum
            if not any(abs(b_hat - prev_b) < 1e-4 for prev_b, _ in refined_peaks):
                refined_peaks.append((b_hat, ll_hat))

    # Fallback if monotonic on [grid_min, grid_max]: optimize around IVW slope
    if not refined_peaks:
        w_ivw = 1.0 / (sy_arr * sy_arr)
        b_ivw = float(np.sum(w_ivw * bx_arr * by_arr) / max(float(np.sum(w_ivw * bx_arr * bx_arr)), 1e-12))
        lo = min(float(grid_min), b_ivw - 2.0)
        hi = max(float(grid_max), b_ivw + 2.0)
        opt = optimize.minimize_scalar(
            lambda b: -eval_llk_scalar(float(b)), bounds=(lo, hi), method="bounded"
        )
        b_hat = float(opt.x)
        refined_peaks.append((b_hat, eval_llk_scalar(b_hat)))

    prom_thresh = 0.5 if mode == "default" else 1.0
    if diagnosis:
        kept_peaks = _filter_modes_by_prominence(
            refined_peaks, eval_llk_scalar, min_prominence=prom_thresh
        )
    else:
        best_p = max(refined_peaks, key=lambda x: x[1])
        kept_peaks = [(best_p[0], best_p[1], float("inf"))]

    # Sort modes descending by profile log-likelihood (dominant mode first)
    kept_peaks.sort(key=lambda x: x[1], reverse=True)

    # Precompute per-SNP loss at each mode for marker-SNP specificity scoring
    per_mode_llk_j: list[np.ndarray] = []
    per_mode_t_j: list[np.ndarray] = []
    for b_m, _, _ in kept_peaks:
        v_m = np.maximum(
            sy_arr * sy_arr + (b_m * b_m) * (sx_arr * sx_arr) - 2.0 * b_m * cor_xy * sx_arr * sy_arr,
            1e-30,
        )
        t_m = (by_arr - b_m * bx_arr) / np.sqrt(v_m)
        per_mode_t_j.append(t_m)
        per_mode_llk_j.append(-rho_loss(t_m, loss, k_val))

    mode_objs: list[GrappleMode] = []
    for idx_m, (b_m, ll_m, prom_m) in enumerate(kept_peaks):
        _, _, _, se_m = _profile_llk_and_derivs_scalar(
            b_m, bx_arr, sx_arr, by_arr, sy_arr, cor_xy, loss, k_val
        )
        z_m = float(b_m / max(se_m, 1e-15))
        p_m = float(2.0 * stats.norm.sf(abs(z_m)))
        ci_lo = float(b_m - 1.959963984540054 * se_m)
        ci_hi = float(b_m + 1.959963984540054 * se_m)

        t_m = per_mode_t_j[idx_m]
        w_m = robust_inlier_weight(t_m, loss, k_val)
        n_inliers = int(np.sum(w_m > 0.5))

        v_m = np.maximum(
            sy_arr * sy_arr + (b_m * b_m) * (sx_arr * sx_arr) - 2.0 * b_m * cor_xy * sx_arr * sy_arr,
            1e-30,
        )
        leverage = w_m * (bx_arr * bx_arr) / v_m
        if len(kept_peaks) > 1:
            other_llks = np.column_stack(
                [per_mode_llk_j[j] for j in range(len(kept_peaks)) if j != idx_m]
            )
            llk_gain = np.maximum(per_mode_llk_j[idx_m] - np.max(other_llks, axis=1), 0.0)
            marker_metric = leverage * (1.0 + llk_gain)
        else:
            marker_metric = leverage

        inlier_idx = np.where(w_m > 0.5)[0]
        if inlier_idx.size == 0:
            inlier_idx = np.argsort(np.abs(t_m))[: min(5, bx_arr.size)]
        ranked_idx = inlier_idx[np.argsort(-marker_metric[inlier_idx])][: max(1, int(n_markers))]
        top_snps = [ids[int(j)] for j in ranked_idx]
        score_map = {ids[int(j)]: float(marker_metric[int(j)]) for j in ranked_idx}

        mode_objs.append(
            GrappleMode(
                mode_id=idx_m + 1,
                beta_hat=float(b_m),
                beta_se=float(se_m),
                z_score=z_m,
                p_value=p_m,
                ci_low=ci_lo,
                ci_high=ci_hi,
                profile_llk=float(ll_m),
                prominence=float(prom_m),
                n_inliers=n_inliers,
                marker_snps=top_snps,
                marker_scores=score_map,
            )
        )

    primary = mode_objs[0]
    return GrappleResult(
        beta_hat=primary.beta_hat,
        beta_se=primary.beta_se,
        z_score=primary.z_score,
        p_value=primary.p_value,
        n_modes=len(mode_objs),
        is_multimodal=len(mode_objs) > 1,
        modes=mode_objs,
        grid_beta=beta_grid,
        grid_llk=grid_llk,
        cor_xy=float(cor_xy),
        loss_function=loss,
        k=float(k_val),
        prominence_threshold=float(prom_thresh),
        n_instruments=int(bx_arr.size),
        snp_ids=ids,
        settings=mode,
    )


def _mvmr_neg_llk_and_grad(
    beta: np.ndarray,
    bx_mat: np.ndarray,
    by: np.ndarray,
    sig_xx: np.ndarray,
    sig_xy: np.ndarray,
    sy2: np.ndarray,
    loss: LossName,
    k: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute ``(neg_llk, grad_neg_llk, per_snp_score_matrix)`` for multivariable GRAPPLE."""
    # sig_xx: (M, P, P), sig_xy: (M, P), sy2: (M,)
    sig_xx_b = np.einsum("mpq,q->mp", sig_xx, beta)  # (M, P)
    v = np.maximum(
        sy2 - 2.0 * (sig_xy @ beta) + np.einsum("mp,p->m", sig_xx_b, beta),
        1e-30,
    )
    sqrt_v = np.sqrt(v)
    r = by - (bx_mat @ beta)  # (M,)
    t = r / sqrt_v

    gv = sig_xx_b - sig_xy  # (M, P) = 0.5 * grad_beta(v)
    grad_t = -(bx_mat * v[:, None] + r[:, None] * gv) / (v * sqrt_v)[:, None]  # (M, P)

    rho_v = rho_loss(t, loss, k)
    psi_v = psi_loss(t, loss, k)

    neg_llk = float(np.sum(rho_v))
    # Score of +llk is -psi(t) * grad_t; gradient of neg_llk is +psi(t) * grad_t
    score_j = -psi_v[:, None] * grad_t  # (M, P)
    grad_neg_llk = -np.sum(score_j, axis=0)
    return neg_llk, grad_neg_llk, score_j


def grapple_mvmr(
    Bx: ArrayLike,
    Sx: ArrayLike,
    by: ArrayLike,
    sy: ArrayLike,
    cor_mat: ArrayLike | None = None,
    loss_function: str = "tukey",
    k: float | None = None,
    exposure_names: list[str] | None = None,
    settings: SettingsMode = "default",
) -> GrappleMVMRResult:
    """Fit overlap-aware robust adjusted profile-likelihood Multivariable MR (`GRAPPLE`).

    Parameters
    ----------
    Bx, Sx : array-like of shape (M, P)
        Estimated SNP-exposure effects and standard errors across ``M`` instruments and ``P`` exposures.
    by, sy : array-like of shape (M,)
        Estimated SNP-outcome effects and standard errors.
    cor_mat : array-like of shape (P + 1, P + 1), optional
        Cross-trait error correlation matrix ordered ``(X_1, ..., X_P, Y)``. Defaults to identity.
    """
    loss, k_val, mode = resolve_loss_params(loss_function, k, settings=settings)

    bx_mat = np.asarray(Bx, dtype=float)
    sx_mat = np.asarray(Sx, dtype=float)
    if bx_mat.ndim == 1:
        bx_mat = bx_mat[:, None]
    if sx_mat.ndim == 1:
        sx_mat = sx_mat[:, None]
    by_arr = np.asarray(by, dtype=float).ravel()
    sy_arr = np.asarray(sy, dtype=float).ravel()

    if bx_mat.ndim != 2 or sx_mat.shape != bx_mat.shape:
        raise ValueError("Bx and Sx must be 2D arrays of identical shape (M, P).")
    m_snps, n_exp = bx_mat.shape
    if by_arr.size != m_snps or sy_arr.size != m_snps:
        raise ValueError("by and sy must have length M matching Bx.shape[0].")
    if m_snps <= n_exp + 1:
        raise ValueError("Number of instruments M must exceed P + 1 for MVMR.")
    if np.any(sx_mat <= 0) or np.any(sy_arr <= 0):
        raise ValueError("Standard errors Sx and sy must be strictly positive.")

    if cor_mat is None:
        r_mat = np.eye(n_exp + 1, dtype=float)
    else:
        r_mat = np.asarray(cor_mat, dtype=float)
        if r_mat.shape != (n_exp + 1, n_exp + 1):
            raise ValueError(
                f"cor_mat must have shape ({n_exp + 1}, {n_exp + 1}) for P={n_exp} exposures + 1 outcome."
            )
        if not np.allclose(r_mat, r_mat.T, atol=1e-6) or not np.allclose(
            np.diag(r_mat), 1.0, atol=1e-5
        ):
            raise ValueError("cor_mat must be a symmetric correlation matrix with unit diagonal.")
        eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (r_mat + r_mat.T))))
        if eig_min <= 1e-8:
            raise ValueError("cor_mat must be strictly positive definite.")

    if exposure_names is None:
        exp_names = [f"X{p + 1}" for p in range(n_exp)]
    else:
        if len(exposure_names) != n_exp:
            raise ValueError("exposure_names length must match Bx.shape[1].")
        exp_names = [str(x) for x in exposure_names]

    # Precompute per-SNP error covariance blocks Sigma_XX (M, P, P) and sigma_XY (M, P)
    r_xx = r_mat[:n_exp, :n_exp]
    r_xy = r_mat[:n_exp, n_exp]
    sig_xx = sx_mat[:, :, None] * r_xx[None, :, :] * sx_mat[:, None, :]
    sig_xy = sx_mat * r_xy[None, :] * sy_arr[:, None]
    sy2 = sy_arr * sy_arr

    # Stage 1: Weighted MV-IVW initial estimate -> L2 profile -> Huber warm start -> target loss
    w0 = 1.0 / sy2
    xtwx = (bx_mat.T * w0[None, :]) @ bx_mat + np.eye(n_exp) * 1e-8
    xtwy = bx_mat.T @ (w0 * by_arr)
    beta_ivw = np.linalg.solve(xtwx, xtwy)

    res_l2 = optimize.minimize(
        lambda b: _mvmr_neg_llk_and_grad(b, bx_mat, by_arr, sig_xx, sig_xy, sy2, "l2", 0.0)[:2],
        beta_ivw,
        method="BFGS",
        jac=True,
        options={"maxiter": 150, "gtol": 1e-7},
    )
    beta_l2 = res_l2.x

    res_huber = optimize.minimize(
        lambda b: _mvmr_neg_llk_and_grad(
            b, bx_mat, by_arr, sig_xx, sig_xy, sy2, "huber", 1.345
        )[:2],
        beta_l2,
        method="BFGS",
        jac=True,
        options={"maxiter": 150, "gtol": 1e-7},
    )
    beta_huber = res_huber.x

    if loss == "l2":
        best_beta = beta_l2
        best_nll = float(res_l2.fun)
        converged = bool(res_l2.success)
    elif loss == "huber" and np.isclose(k_val, 1.345):
        best_beta = beta_huber
        best_nll = float(res_huber.fun)
        converged = bool(res_huber.success)
    else:
        rng = np.random.default_rng(0)
        candidates = [
            beta_huber,
            beta_l2,
            beta_ivw,
            np.zeros(n_exp, dtype=float),
            beta_huber + rng.normal(0.0, 0.15, size=n_exp),
            beta_huber + rng.normal(0.0, 0.30, size=n_exp),
        ]
        best_beta = beta_huber
        best_nll = np.inf
        converged = False
        for init_b in candidates:
            opt = optimize.minimize(
                lambda b: _mvmr_neg_llk_and_grad(
                    b, bx_mat, by_arr, sig_xx, sig_xy, sy2, loss, k_val
                )[:2],
                init_b,
                method="BFGS",
                jac=True,
                options={"maxiter": 250, "gtol": 1e-7},
            )
            if float(opt.fun) < best_nll:
                best_nll = float(opt.fun)
                best_beta = opt.x.copy()
                converged = converged or bool(opt.success)

    # Compute exact P x P sandwich covariance at best_beta
    nll_hat, _, score_j = _mvmr_neg_llk_and_grad(
        best_beta, bx_mat, by_arr, sig_xx, sig_xy, sy2, loss, k_val
    )

    hess = np.zeros((n_exp, n_exp), dtype=float)
    eps = 1e-6
    for p in range(n_exp):
        step = eps * max(1.0, abs(float(best_beta[p])))
        bp = best_beta.copy()
        bm = best_beta.copy()
        bp[p] += step
        bm[p] -= step
        _, gp, _ = _mvmr_neg_llk_and_grad(bp, bx_mat, by_arr, sig_xx, sig_xy, sy2, loss, k_val)
        _, gm, _ = _mvmr_neg_llk_and_grad(bm, bx_mat, by_arr, sig_xx, sig_xy, sy2, loss, k_val)
        hess[:, p] = (gp - gm) / (2.0 * step)

    bread = 0.5 * (hess + hess.T)
    eig_vals, eig_vecs = np.linalg.eigh(bread)
    eig_pos = np.maximum(eig_vals, 1e-8)
    bread_inv = (eig_vecs * (1.0 / eig_pos)[None, :]) @ eig_vecs.T

    sig_xx_b = np.einsum("mpq,q->mp", sig_xx, best_beta)
    v_hat = np.maximum(
        sy2 - 2.0 * (sig_xy @ best_beta) + np.einsum("mp,p->m", sig_xx_b, best_beta),
        1e-30,
    )
    t_hat = (by_arr - bx_mat @ best_beta) / np.sqrt(v_hat)
    w_in = robust_inlier_weight(t_hat, loss, k_val)
    m_in = float(np.sum(w_in > 1e-3))
    df_scale = m_in / max(m_in - float(n_exp), 1.0) if m_in > n_exp else 1.0
    meat = df_scale * (score_j.T @ score_j)

    cov_beta = bread_inv @ meat @ bread_inv.T
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-16))
    z_scores = best_beta / se_beta
    p_values = 2.0 * stats.norm.sf(np.abs(z_scores))

    q_stat = float(np.sum(t_hat * t_hat))
    q_pval = float(stats.chi2.sf(q_stat, df=max(m_snps - n_exp, 1)))

    return GrappleMVMRResult(
        beta_hat=best_beta,
        beta_se=se_beta,
        z_scores=z_scores,
        p_values=p_values,
        cov_beta=cov_beta,
        exposure_names=exp_names,
        profile_llk=float(-nll_hat),
        converged=converged,
        inlier_weights=w_in,
        q_stat=q_stat,
        q_pval=q_pval,
        cor_mat=r_mat,
        loss_function=loss,
        k=float(k_val),
        n_instruments=m_snps,
        settings=mode,
    )


class GRAPPLE:
    """Object-oriented interface for univariable and multivariable GRAPPLE."""

    def __init__(
        self,
        loss_function: str = "tukey",
        k: float | None = None,
        diagnosis: bool = True,
        p_thres: float = 1e-4,
        r2_prune: float = 0.001,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        n_grid: int = 500,
        settings: SettingsMode = "default",
    ) -> None:
        loss, k_val, mode = resolve_loss_params(loss_function, k, settings=settings)
        self.settings: SettingsMode = mode
        self.loss_function: LossName = loss
        self.k: float = k_val
        self.diagnosis = bool(diagnosis)
        self.p_thres = float(p_thres)
        self.r2_prune = float(r2_prune)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.n_grid = int(n_grid)
        self.cor_mat_: np.ndarray | None = None
        self.result_: GrappleResult | None = None
        self.mvmr_result_: GrappleMVMRResult | None = None

    def estimate_cor_mat(
        self, z_null_df: pd.DataFrame | ArrayLike, p_thres: float = 0.5
    ) -> np.ndarray:
        """Estimate sample-overlap error correlation matrix and store on ``cor_mat_``."""
        self.cor_mat_ = estimate_cor_mat(
            z_null_df, p_thres=p_thres, settings=self.settings
        )
        return self.cor_mat_

    def select_instruments(
        self, sumstats_df: pd.DataFrame, ld_R: ArrayLike | None = None
    ) -> pd.DataFrame:
        """Select independent weak + strong instruments."""
        return select_grapple_instruments(
            sumstats_df,
            ld_R=ld_R,
            p_thres=self.p_thres,
            r2_prune=self.r2_prune,
            settings=self.settings,
        )

    def fit(
        self,
        bx: ArrayLike,
        sx: ArrayLike,
        by: ArrayLike,
        sy: ArrayLike,
        cor_xy: float | None = None,
        snp_ids: list[str] | None = None,
    ) -> GrappleResult:
        """Fit univariable GRAPPLE robust profile-likelihood MR."""
        c_xy = (
            float(cor_xy)
            if cor_xy is not None
            else (float(self.cor_mat_[0, 1]) if self.cor_mat_ is not None else 0.0)
        )
        self.result_ = grapple_robust_est(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            cor_xy=c_xy,
            loss_function=self.loss_function,
            k=self.k,
            diagnosis=self.diagnosis,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            n_grid=self.n_grid,
            snp_ids=snp_ids,
            settings=self.settings,
        )
        return self.result_

    def fit_mvmr(
        self,
        Bx: ArrayLike,
        Sx: ArrayLike,
        by: ArrayLike,
        sy: ArrayLike,
        cor_mat: ArrayLike | None = None,
        exposure_names: list[str] | None = None,
    ) -> GrappleMVMRResult:
        """Fit overlap-aware multivariable GRAPPLE."""
        c_mat = cor_mat if cor_mat is not None else self.cor_mat_
        self.mvmr_result_ = grapple_mvmr(
            Bx=Bx,
            Sx=Sx,
            by=by,
            sy=sy,
            cor_mat=c_mat,
            loss_function=self.loss_function,
            k=self.k,
            exposure_names=exposure_names,
            settings=self.settings,
        )
        return self.mvmr_result_
