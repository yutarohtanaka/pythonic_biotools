"""
Fast batched REML Linear Mixed Model (LMM) solver across G genes.

Implements three tailored execution paths:
  1. ``R = 0``: closed-form batched weighted/ordinary least squares.
  2. ``R = 1`` (shared/uniform weights): precomputed orthogonal-projection SVD
     spectral REML (FaST-LMM style) reducing optimization to 1D per gene.
  3. General ``R >= 1`` or gene-specific precision weights: symmetric-scaled
     ``m x m`` Henderson/Woodbury profiled REML in variance ratios ``gamma_r >= 0``
     with exact analytical gradients and OLS-residualized numerical stability.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .formula import DesignSpec, coerce_expression_and_weights


@dataclass(frozen=True)
class LMMFitResult:
    """Typed result container for batched REML linear mixed-model fitting."""

    gene_names: list[str]
    design: DesignSpec
    betas: np.ndarray
    sigma_r_sq: np.ndarray
    sigma_e_sq: np.ndarray
    cov_beta_unscaled: np.ndarray
    blups: list[np.ndarray]
    reml_nll: np.ndarray
    ave_expr: np.ndarray
    n_obs: np.ndarray
    df_residual_ols: np.ndarray
    mean_inv_weight: np.ndarray
    converged: np.ndarray
    singular: np.ndarray
    fisher_info: np.ndarray | None
    a_zx: np.ndarray | None
    cov_beta_fixed: np.ndarray | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-gene fixed-effect estimates and variance components as a DataFrame."""
        data: dict[str, Any] = {}
        for j, fname in enumerate(self.design.fixed_names):
            data[f"beta_{fname}"] = self.betas[:, j]
        for r, rname in enumerate(self.design.random_names):
            data[f"sigma_sq_{rname}"] = self.sigma_r_sq[:, r]
        data["sigma_sq_Residuals"] = self.sigma_e_sq
        data["reml_nll"] = self.reml_nll
        data["ave_expr"] = self.ave_expr
        data["n_obs"] = self.n_obs
        data["converged"] = self.converged
        data["singular"] = self.singular
        return pd.DataFrame(data, index=self.gene_names)

    def summary(self) -> str:
        """Return a concise multi-line summary of the LMM fit."""
        n_genes = len(self.gene_names)
        n_conv = int(np.sum(self.converged))
        n_sing = int(np.sum(self.singular))
        med_se2 = float(np.nanmedian(self.sigma_e_sq)) if n_genes > 0 else float("nan")
        lines = [
            f"LMMFitResult(genes={n_genes}, samples={self.design.n_samples}, "
            f"fixed={self.design.n_fixed}, random={self.design.n_random_terms})",
            f"  converged: {n_conv}/{n_genes}, singular: {n_sing}/{n_genes}, median_sigma_e_sq: {med_se2:.4g}",
        ]
        for r, rname in enumerate(self.design.random_names):
            med_r = float(np.nanmedian(self.sigma_r_sq[:, r]))
            lines.append(f"  random '{rname}' (m={self.design.random_matrices[r].shape[1]}): median_sigma_sq={med_r:.4g}")
        return "\n".join(lines)


def _safe_cholesky_lower(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower Cholesky factor and SPD matrix with eigenvalue floor fallback."""
    A_sym = 0.5 * (A + A.T)
    try:
        return np.linalg.cholesky(A_sym), A_sym
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        floor = max(1e-15 * max(float(np.max(eigvals)), 1.0), 1e-15)
        eigvals_clipped = np.maximum(eigvals, floor)
        A_spd = (eigvecs * eigvals_clipped[None, :]) @ eigvecs.T
        A_spd = 0.5 * (A_spd + A_spd.T)
        return np.linalg.cholesky(A_spd), A_spd


def _reduce_fixed_design(
    X: np.ndarray,
) -> tuple[np.ndarray, int, bool, np.ndarray | None, np.ndarray | None]:
    """Return full-rank basis for X along with lifting matrices if rank-deficient.

    Returns ``(X_fit, r_x, is_full_rank, lift_mat, T_proj)`` such that when
    ``not is_full_rank``, ``X = X_fit @ T_proj`` and ``lift_mat = V_r @ Sigma_r^{-1}``
    satisfies ``T_proj @ lift_mat = I_{r_x}``.
    """
    n, p = X.shape
    U_x, s_x, Vt_x = np.linalg.svd(X, full_matrices=False)
    s_max = float(s_x[0]) if s_x.size > 0 else 0.0
    tol = max(n, p) * np.finfo(np.float64).eps * max(s_max, 1.0)
    r_x = int(np.sum(s_x > max(tol, 1e-12 * s_max)))
    if r_x == p:
        return X, p, True, None, None
    U_r = np.ascontiguousarray(U_x[:, :r_x], dtype=np.float64)
    s_r = s_x[:r_x]
    Vt_r = Vt_x[:r_x, :]
    lift_mat = np.ascontiguousarray(Vt_r.T / s_r[None, :], dtype=np.float64)  # (p, r_x)
    T_proj = np.ascontiguousarray(s_r[:, None] * Vt_r, dtype=np.float64)  # (r_x, p)
    return U_r, r_x, False, lift_mat, T_proj


def _lift_estimates(
    alpha_hat: np.ndarray,
    V_alpha_H: np.ndarray,
    A_ZX_fit: np.ndarray | None,
    is_full_rank: bool,
    lift_mat: np.ndarray | None,
    T_proj: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if is_full_rank:
        return alpha_hat, V_alpha_H, A_ZX_fit
    assert lift_mat is not None and T_proj is not None
    beta = lift_mat @ alpha_hat
    V_beta_H = (lift_mat @ V_alpha_H) @ lift_mat.T
    V_beta_H = 0.5 * (V_beta_H + V_beta_H.T)
    A_ZX_orig = (A_ZX_fit @ T_proj) if A_ZX_fit is not None else None
    return beta, V_beta_H, A_ZX_orig


def _mme_profile_eval(
    gamma: np.ndarray,
    *,
    S_XX: np.ndarray,
    S_ZZ: np.ndarray,
    S_XZ: np.ndarray,
    s_Zy_resid: np.ndarray,
    s_yy_resid: float,
    df_resid: float,
    block_sizes: np.ndarray,
    block_slices: list[slice],
    gamma_max: float,
    need_full: bool = False,
):
    """Evaluate profiled REML objective and exact analytical gradient in m x m MME space."""
    R = len(block_slices)
    m = S_ZZ.shape[0]
    r_x = S_XX.shape[0]
    I_rx = np.eye(r_x, dtype=np.float64)

    g = np.clip(np.asarray(gamma, dtype=np.float64), 0.0, gamma_max)
    d_gamma = np.sqrt(g)
    d_vec = np.repeat(d_gamma, block_sizes)

    M = (S_ZZ * d_vec[:, None]) * d_vec[None, :]
    M[np.diag_indices(m)] += 1.0
    L_M, M_spd = _safe_cholesky_lower(M)
    logdet_M = 2.0 * float(np.sum(np.log(np.diag(L_M))))

    R_X = d_vec[:, None] * S_XZ.T
    SZZ_D = S_ZZ * d_vec[None, :]
    r_y = d_vec * s_Zy_resid

    # Single batched solve with M_spd for R_X, SZZ_D.T, and r_y
    rhs_M = np.empty((m, r_x + m + 1), dtype=np.float64)
    rhs_M[:, :r_x] = R_X
    rhs_M[:, r_x : r_x + m] = SZZ_D.T
    rhs_M[:, r_x + m] = r_y
    sol_M = np.linalg.solve(M_spd, rhs_M)
    Minv_RX = sol_M[:, :r_x]
    Minv_SZZ_Dt = sol_M[:, r_x : r_x + m]
    Minv_ry = sol_M[:, r_x + m]

    XtHinvX = S_XX - R_X.T @ Minv_RX
    L_X, XtHinvX_spd = _safe_cholesky_lower(XtHinvX)
    logdet_X = 2.0 * float(np.sum(np.log(np.diag(L_X))))

    XtHinv_y = -R_X.T @ Minv_ry
    rhs_X = np.empty((r_x, r_x + 1), dtype=np.float64)
    rhs_X[:, :r_x] = I_rx
    rhs_X[:, r_x] = XtHinv_y
    sol_X = np.linalg.solve(XtHinvX_spd, rhs_X)
    V_alpha_H = sol_X[:, :r_x]
    V_alpha_H = 0.5 * (V_alpha_H + V_alpha_H.T)
    delta_alpha = sol_X[:, r_x]

    q_val = max((s_yy_resid - float(r_y @ Minv_ry)) - float(delta_alpha @ XtHinv_y), 1e-300)
    sigma_e_sq = q_val / df_resid
    f_prof = df_resid * np.log(sigma_e_sq) + logdet_M + logdet_X

    A_Z = S_ZZ - SZZ_D @ Minv_SZZ_Dt
    A_ZX = S_XZ.T - SZZ_D @ Minv_RX
    S = A_Z - (A_ZX @ V_alpha_H) @ A_ZX.T
    S = 0.5 * (S + S.T)

    Minv_diff = Minv_ry - Minv_RX @ delta_alpha
    psi = (s_Zy_resid - S_XZ.T @ delta_alpha) - SZZ_D @ Minv_diff

    grad = np.empty(R, dtype=np.float64)
    tr_S = np.empty(R, dtype=np.float64)
    for r_idx, sl in enumerate(block_slices):
        tr_r = float(np.trace(S[sl, sl]))
        tr_S[r_idx] = tr_r
        psi_r = psi[sl]
        grad[r_idx] = tr_r - float(psi_r @ psi_r) / sigma_e_sq

    if not need_full:
        return f_prof, grad

    return f_prof, grad, sigma_e_sq, delta_alpha, V_alpha_H, psi, S, tr_S, A_ZX


def _solve_mme_single_gene(
    S_XX: np.ndarray,
    S_ZZ: np.ndarray,
    S_XZ: np.ndarray,
    s_Zy_resid: np.ndarray,
    s_yy_resid: float,
    alpha_ols: np.ndarray,
    y_w_norm_sq: float,
    logdet_w: float,
    df_resid: float,
    block_sizes: np.ndarray,
    block_slices: list[slice],
    gamma_max: float,
    max_iter: int,
    tol: float,
    compute_fisher: bool,
    gamma_override: np.ndarray | None = None,
    is_zero_resid: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    float,
    bool,
    bool,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Solve profiled REML via symmetric-scaled m x m Henderson/Woodbury MME for one gene."""
    R = len(block_slices)
    m = S_ZZ.shape[0]
    r_x = S_XX.shape[0]
    I_rx = np.eye(r_x, dtype=np.float64)

    # Zero-residual flat gene check
    if is_zero_resid or (s_yy_resid <= 1e-27 * max(1e-300, y_w_norm_sq)):
        _, S_XX_spd = _safe_cholesky_lower(S_XX)
        V_alpha0 = np.linalg.solve(S_XX_spd, I_rx)
        V_alpha0 = 0.5 * (V_alpha0 + V_alpha0.T)
        gamma0 = np.zeros(R, dtype=np.float64)
        u_zero = np.zeros(m, dtype=np.float64)
        fisher0 = np.zeros((R + 1, R + 1), dtype=np.float64) if compute_fisher else None
        a_zx0 = S_XZ.T.copy() if compute_fisher else None
        return (
            alpha_ols.copy(),
            gamma0,
            0.0,
            V_alpha0,
            u_zero,
            float("nan"),
            True,
            True,
            fisher0,
            a_zx0,
        )

    def _eval(gamma: np.ndarray, need_full: bool = False):
        return _mme_profile_eval(
            gamma,
            S_XX=S_XX,
            S_ZZ=S_ZZ,
            S_XZ=S_XZ,
            s_Zy_resid=s_Zy_resid,
            s_yy_resid=s_yy_resid,
            df_resid=df_resid,
            block_sizes=block_sizes,
            block_slices=block_slices,
            gamma_max=gamma_max,
            need_full=need_full,
        )

    if gamma_override is not None:
        gamma_hat = np.clip(np.asarray(gamma_override, dtype=np.float64), 0.0, gamma_max)
        converged = True
        collinear_r = np.zeros(R, dtype=bool)
    else:
        gamma_zero = np.zeros(R, dtype=np.float64)
        f0, g0, se2_0, _, _, psi0, S0, tr_S0, _ = _eval(gamma_zero, need_full=True)

        # Method-of-moments initializer and collinear-block detection (Z_r in col(X_obs))
        collinear_r = np.zeros(R, dtype=bool)
        gamma_mom = np.empty(R, dtype=np.float64)
        for r_idx, sl in enumerate(block_slices):
            S_ZZ_rr = S_ZZ[sl, sl]
            ss_ZZ_r = float(np.sum(S_ZZ_rr * S_ZZ_rr))
            S_rr = S0[sl, sl]
            ss_S_r = float(np.sum(S_rr * S_rr))
            if ss_ZZ_r <= 0.0 or ss_S_r <= 1e-24 * ss_ZZ_r:
                collinear_r[r_idx] = True
                gamma_mom[r_idx] = 0.0
            else:
                est_r = (float(psi0[sl] @ psi0[sl]) / max(se2_0, 1e-300) - tr_S0[r_idx]) / ss_S_r
                gamma_mom[r_idx] = float(np.clip(max(est_r, 0.05), 0.02, min(gamma_max, 100.0)))

        free_idx = np.flatnonzero(~collinear_r)
        if free_idx.size == 0:
            gamma_hat = gamma_zero.copy()
            f_best = f0
            converged = True
        else:
            def _obj(g_free: np.ndarray) -> tuple[float, np.ndarray]:
                if free_idx.size == R:
                    val, gr = _eval(g_free, need_full=False)
                    return float(val - f0), gr
                g_full = np.zeros(R, dtype=np.float64)
                g_full[free_idx] = g_free
                val, gr = _eval(g_full, need_full=False)
                return float(val - f0), gr[free_idx]

            bounds_free = [(0.0, gamma_max)] * free_idx.size
            opts = {"ftol": min(1e-12, tol * 1e-3), "gtol": min(1e-8, tol), "maxiter": max_iter}
            res = minimize(
                _obj, gamma_mom[free_idx], jac=True, method="L-BFGS-B", bounds=bounds_free, options=opts
            )
            gamma_hat = np.zeros(R, dtype=np.float64)
            gamma_hat[free_idx] = np.clip(res.x, 0.0, gamma_max)
            f_best = float(res.fun) + f0
            converged = bool(res.success)

            # Fallback restart from 0 if optimization didn't beat gamma=0 or failed
            if f_best > f0 + 1e-10 or not converged:
                res2 = minimize(
                    _obj,
                    np.zeros(free_idx.size, dtype=np.float64),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds_free,
                    options=opts,
                )
                f2 = float(res2.fun) + f0
                if f2 < f_best:
                    gamma_hat = np.zeros(R, dtype=np.float64)
                    gamma_hat[free_idx] = np.clip(res2.x, 0.0, gamma_max)
                    f_best = f2
                    converged = bool(res2.success)

            # Exact boundary check against gamma = 0
            if f0 <= f_best + 1e-10:
                gamma_hat = gamma_zero.copy()
                f_best = f0
                converged = True
            else:
                # Prune near-zero individual variance components cleanly to exact 0.0
                for r_idx in free_idx:
                    if gamma_hat[r_idx] < 1e-8:
                        cand = gamma_hat.copy()
                        cand[r_idx] = 0.0
                        f_cand, _ = _eval(cand, need_full=False)
                        if f_cand <= f_best + 1e-10:
                            gamma_hat = cand
                            f_best = f_cand
                gamma_hat[collinear_r] = 0.0

    f_prof, _, sigma_e_sq, delta_alpha, V_alpha_H, psi, S, tr_S, A_ZX = _eval(
        gamma_hat, need_full=True
    )
    alpha_hat = alpha_ols + delta_alpha
    sigma_r_sq = gamma_hat * sigma_e_sq
    u_hat = np.repeat(gamma_hat, block_sizes) * psi
    reml_nll = 0.5 * (df_resid * np.log(2.0 * np.pi) + f_prof + df_resid - logdet_w)
    singular = bool(np.any(gamma_hat <= 1e-8))

    fisher_info: np.ndarray | None = None
    a_zx_out: np.ndarray | None = None
    if compute_fisher:
        collinear_fisher = collinear_r.copy()
        for j, sl_j in enumerate(block_slices):
            if not collinear_fisher[j] and gamma_hat[j] == 0.0:
                ss_zz_j = float(np.sum(S_ZZ[sl_j, sl_j] ** 2))
                if ss_zz_j <= 0.0 or float(np.sum(S[sl_j, sl_j] ** 2)) <= 1e-24 * ss_zz_j:
                    collinear_fisher[j] = True

        inv_2se4 = 0.5 / max(sigma_e_sq * sigma_e_sq, 1e-300)
        tr_S_eff = tr_S.copy()
        tr_S_eff[collinear_fisher] = 0.0
        F_blocks = np.zeros((R, R), dtype=np.float64)
        for j, sl_j in enumerate(block_slices):
            if collinear_fisher[j]:
                continue
            for k in range(j, R):
                if collinear_fisher[k]:
                    continue
                sl_k = block_slices[k]
                sub = S[sl_j, sl_k]
                val = float(np.sum(sub * sub))
                F_blocks[j, k] = val
                F_blocks[k, j] = val
        I_E = np.empty((R + 1, R + 1), dtype=np.float64)
        I_E[:R, :R] = F_blocks * inv_2se4
        cross = (tr_S_eff - F_blocks @ gamma_hat) * inv_2se4
        I_E[:R, R] = cross
        I_E[R, :R] = cross
        I_E[R, R] = (
            df_resid - 2.0 * float(gamma_hat @ tr_S_eff) + float(gamma_hat @ F_blocks @ gamma_hat)
        ) * inv_2se4
        fisher_info = 0.5 * (I_E + I_E.T)
        a_zx_out = A_ZX

    return (
        alpha_hat,
        sigma_r_sq,
        float(sigma_e_sq),
        V_alpha_H,
        u_hat,
        float(reml_nll),
        converged,
        singular,
        fisher_info,
        a_zx_out,
    )


def _solve_spectral_r1_batch(
    Y_w: np.ndarray,
    X_w: np.ndarray,
    Z_w: np.ndarray,
    logdet_w: float,
    df_resid: float,
    gamma_max: float,
    tol: float,
    compute_fisher: bool,
    is_const: np.ndarray | None = None,
    tss_y: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Fast spectral REML solver for complete genes with R=1 and shared/uniform weights."""
    n, g_cnt = Y_w.shape
    r_x = X_w.shape[1]
    m1 = Z_w.shape[1]

    Q_full, R_full = np.linalg.qr(X_w, mode="complete")
    R_qr = R_full[:r_x, :r_x]
    logdet_xtx = 2.0 * float(np.sum(np.log(np.abs(np.diag(R_qr)))))
    Q_X = Q_full[:, :r_x]
    L = Q_full[:, r_x:]  # (N, N - r_x)

    alpha_ols_all = np.linalg.solve(R_qr, Q_X.T @ Y_w)

    # Precompute sufficient statistics for Woodbury recovery & Fisher info
    S_XX = X_w.T @ X_w
    S_ZZ = Z_w.T @ Z_w
    S_XZ = X_w.T @ Z_w
    E_ols = Y_w - X_w @ alpha_ols_all
    S_Zy_all = Z_w.T @ E_ols
    s_yy_all = np.sum(E_ols * E_ols, axis=0)
    y_norm_all = np.sum(Y_w * Y_w, axis=0)

    if is_const is None:
        mean_w = np.mean(Y_w, axis=0)
        tss_fallback = np.sum((Y_w - mean_w[None, :]) ** 2, axis=0)
        is_zero_all = (tss_fallback / float(n)) <= 1e-24 * (mean_w * mean_w)
        tss_ref = tss_fallback
    else:
        is_zero_all = is_const.copy()
        tss_ref = tss_y if tss_y is not None else y_norm_all
    is_zero_all = (
        is_zero_all
        | ((s_yy_all / df_resid) <= 1e-24 * tss_ref)
        | (s_yy_all <= 1e-27 * y_norm_all)
    )

    # Spectral decomposition of L^T Z_1
    z_norm = float(np.linalg.norm(Z_w))
    U_L, s_L, _ = np.linalg.svd(L.T @ Z_w, full_matrices=False)
    s_max = float(s_L[0]) if s_L.size > 0 else 0.0
    s_ref = max(s_max, z_norm)
    ss_ZZ_0 = float(np.sum(S_ZZ * S_ZZ))
    if ss_ZZ_0 <= 0.0 or float(np.sum(s_L ** 4)) <= 1e-24 * ss_ZZ_0 or s_ref <= 0.0:
        keep = np.zeros(s_L.shape, dtype=bool)
    else:
        keep = s_L > 1e-12 * s_ref
    d_eig = (s_L[keep] ** 2) if np.any(keep) else np.empty(0, dtype=np.float64)
    k = d_eig.size

    Y_perp = L.T @ Y_w  # (N - r_x, G)
    if k > 0:
        R_proj = U_L[:, keep].T @ Y_perp  # (k, G)
        R_sq = R_proj * R_proj
        ss_perp = np.maximum(np.sum(Y_perp * Y_perp, axis=0) - np.sum(R_sq, axis=0), 0.0)
    else:
        R_sq = np.empty((0, g_cnt), dtype=np.float64)
        ss_perp = np.sum(Y_perp * Y_perp, axis=0)

    # Coarse log-grid across all genes in one vectorized pass
    if k > 0:
        grid = np.geomspace(1e-10, gamma_max, 28)
        # denom shape: (n_grid, k)
        denom = 1.0 + grid[:, None] * d_eig[None, :]
        # q_grid shape: (n_grid, G)
        q_grid = ss_perp[None, :] + (1.0 / denom) @ R_sq
        se2_grid = np.maximum(q_grid / df_resid, 1e-300)
        logdet_grid = np.sum(np.log(denom), axis=1)[:, None]
        f_grid = df_resid * np.log(se2_grid) + logdet_grid

        q0 = np.maximum(ss_perp + np.sum(R_sq, axis=0), 1e-300)
        f0_all = df_resid * np.log(q0 / df_resid)
        best_idx = np.argmin(f_grid, axis=0)
    else:
        f0_all = df_resid * np.log(np.maximum(ss_perp / df_resid, 1e-300))
        best_idx = np.zeros(g_cnt, dtype=np.int64)
        grid = np.array([0.0])

    # Allocate outputs
    alpha_out = np.empty((g_cnt, r_x), dtype=np.float64)
    sigma_r_sq_out = np.empty((g_cnt, 1), dtype=np.float64)
    sigma_e_sq_out = np.empty(g_cnt, dtype=np.float64)
    V_alpha_out = np.empty((g_cnt, r_x, r_x), dtype=np.float64)
    u_out = np.empty((g_cnt, m1), dtype=np.float64)
    reml_nll_out = np.empty(g_cnt, dtype=np.float64)
    conv_out = np.ones(g_cnt, dtype=bool)
    sing_out = np.zeros(g_cnt, dtype=bool)
    fisher_out = np.empty((g_cnt, 2, 2), dtype=np.float64) if compute_fisher else None
    a_zx_out = np.empty((g_cnt, m1, r_x), dtype=np.float64) if compute_fisher else None

    log_gmin = np.log(1e-12)
    log_gmax = np.log(gamma_max)
    block_sizes = np.array([m1], dtype=np.int64)
    block_slices = [slice(0, m1)]

    for g in range(g_cnt):
        is_zero_g = bool(is_zero_all[g])
        if is_zero_g:
            gamma_opt = 0.0
        elif k == 0:
            gamma_opt = 0.0
        else:
            r_sq_g = R_sq[:, g]
            ss_p_g = float(ss_perp[g])
            f0_g = float(f0_all[g])

            def _f_eta(eta: float) -> float:
                gam = np.exp(eta)
                inv_term = 1.0 / (1.0 + gam * d_eig)
                se2 = max((ss_p_g + float(np.dot(r_sq_g, inv_term))) / df_resid, 1e-300)
                return (df_resid * np.log(se2) + float(np.sum(np.log1p(gam * d_eig)))) - f0_g

            b_i = int(best_idx[g])
            lo_eta = log_gmin if b_i == 0 else float(np.log(grid[b_i - 1]))
            hi_eta = log_gmax if b_i == len(grid) - 1 else float(np.log(grid[b_i + 1]))
            res_sc = minimize_scalar(
                _f_eta,
                bounds=(lo_eta, hi_eta),
                method="bounded",
                options={"xatol": min(1e-11, tol), "maxiter": 100},
            )
            gamma_sc = float(np.exp(res_sc.x))
            f_sc = float(res_sc.fun) + f0_g

            if f0_g <= f_sc + 1e-10 or gamma_sc < 1e-8:
                gamma_opt = 0.0
            else:
                gamma_opt = gamma_sc

        (
            alpha_g,
            sr2_g,
            se2_g,
            Va_g,
            u_g,
            nll_g,
            conv_g,
            sing_g,
            fi_g,
            azx_g,
        ) = _solve_mme_single_gene(
            S_XX=S_XX,
            S_ZZ=S_ZZ,
            S_XZ=S_XZ,
            s_Zy_resid=S_Zy_all[:, g],
            s_yy_resid=float(s_yy_all[g]),
            alpha_ols=alpha_ols_all[:, g],
            y_w_norm_sq=float(y_norm_all[g]),
            logdet_w=logdet_w,
            df_resid=df_resid,
            block_sizes=block_sizes,
            block_slices=block_slices,
            gamma_max=gamma_max,
            max_iter=50,
            tol=tol,
            compute_fisher=compute_fisher,
            gamma_override=np.array([gamma_opt], dtype=np.float64),
            is_zero_resid=is_zero_g,
        )
        alpha_out[g] = alpha_g
        sigma_r_sq_out[g] = sr2_g
        sigma_e_sq_out[g] = se2_g
        V_alpha_out[g] = Va_g
        u_out[g] = u_g
        reml_nll_out[g] = nll_g
        conv_out[g] = conv_g
        sing_out[g] = sing_g
        if compute_fisher:
            assert fisher_out is not None and a_zx_out is not None
            assert fi_g is not None and azx_g is not None
            fisher_out[g] = fi_g
            a_zx_out[g] = azx_g

    return (
        alpha_out,
        sigma_r_sq_out,
        sigma_e_sq_out,
        V_alpha_out,
        u_out,
        reml_nll_out,
        conv_out,
        sing_out,
        fisher_out,
        a_zx_out,
    )


def fit_lmm(
    expr: Any,
    formula: str | DesignSpec | None = None,
    data: pd.DataFrame | None = None,
    *,
    design: np.ndarray | pd.DataFrame | None = None,
    random_effects: (
        Sequence[np.ndarray | pd.Series | pd.DataFrame | str]
        | Mapping[str, np.ndarray | pd.Series | pd.DataFrame]
        | None
    ) = None,
    weights: np.ndarray | pd.DataFrame | str | None = None,
    layer: str | None = None,
    genes_as_rows: bool = False,
    gene_names: Sequence[str] | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    gamma_max: float = 1e6,
    compute_fisher: bool = False,
    n_jobs: int = 1,
    _force_mme: bool = False,
) -> LMMFitResult:
    """Fit batched REML linear mixed models across G genes.

    Parameters
    ----------
    expr : AnnData | pd.DataFrame | np.ndarray
        Expression matrix. Default orientation is ``(N_samples, G_genes)`` (matching
        ``AnnData`` ``obs x var``). Set ``genes_as_rows=True`` for ``(G_genes, N_samples)``.
    formula : str | DesignSpec | None
        Mixed-effects formula string (e.g. ``"~ condition + (1 | subject)"``) or a
        pre-built ``DesignSpec``.
    data : pd.DataFrame | None
        Sample metadata DataFrame when using a formula string (defaults to ``expr.obs``
        for ``AnnData``).
    design : np.ndarray | pd.DataFrame | None
        Explicit fixed-effects matrix ``X`` when ``formula`` is omitted.
    random_effects : Sequence | Mapping | None
        Random-effect grouping vectors/matrices when ``formula`` is omitted.
    weights : np.ndarray | pd.DataFrame | str | None
        Optional strictly positive precision weights (``1D (N,)``, ``2D (N, G)``,
        or ``AnnData`` layer name).
    layer : str | None
        ``AnnData`` layer to use instead of ``.X``.
    genes_as_rows : bool
        Transpose DataFrame/ndarray input from ``(G, N)`` to ``(N, G)``.
    gene_names : Sequence[str] | None
        Optional override for gene names.
    max_iter : int
        Maximum L-BFGS-B iterations per gene.
    tol : float
        Optimizer tolerance.
    gamma_max : float
        Upper bound on variance ratios ``sigma_r^2 / sigma_e^2``.
    compute_fisher : bool
        If True, compute and cache the ``O(m^3)`` REML Expected Fisher Information
        and ``A_ZX`` matrices required for Satterthwaite degrees of freedom.
    n_jobs : int
        Number of parallel workers (uses ``joblib`` if installed, else ``ThreadPoolExecutor``).
    """
    Y, W, shared_weights, design_spec, resolved_gene_names = coerce_expression_and_weights(
        expr,
        formula=formula,
        data=data,
        design=design,
        random_effects=random_effects,
        weights=weights,
        layer=layer,
        genes_as_rows=genes_as_rows,
        gene_names=gene_names,
    )

    N, G = Y.shape
    p = design_spec.n_fixed
    R = design_spec.n_random_terms
    m = design_spec.total_random_levels

    betas = np.full((G, p), np.nan, dtype=np.float64)
    sigma_r_sq = np.full((G, R), np.nan, dtype=np.float64)
    sigma_e_sq = np.full(G, np.nan, dtype=np.float64)
    cov_beta_unscaled = np.full((G, p, p), np.nan, dtype=np.float64)
    cov_beta_fixed = np.full((G, p, p), np.nan, dtype=np.float64)
    blups = [np.full((G, z.shape[1]), np.nan, dtype=np.float64) for z in design_spec.random_matrices]
    reml_nll = np.full(G, np.nan, dtype=np.float64)
    ave_expr = np.full(G, np.nan, dtype=np.float64)
    n_obs = np.zeros(G, dtype=np.int64)
    df_residual_ols = np.full(G, np.nan, dtype=np.float64)
    mean_inv_weight = np.full(G, np.nan, dtype=np.float64)
    converged = np.zeros(G, dtype=bool)
    singular = np.zeros(G, dtype=bool)
    fisher_info = np.full((G, R + 1, R + 1), np.nan, dtype=np.float64) if compute_fisher else None
    a_zx_arr = np.full((G, m, p), np.nan, dtype=np.float64) if (compute_fisher and R > 0) else None

    X_full = design_spec.X
    Z_blocks = design_spec.random_matrices
    block_sizes = np.array([z.shape[1] for z in Z_blocks], dtype=np.int64)
    block_slices: list[slice] = []
    pos = 0
    for sz in block_sizes:
        block_slices.append(slice(pos, pos + int(sz)))
        pos += int(sz)

    z_scales = np.ones(R, dtype=np.float64)
    if R > 0:
        z_norm_blocks: list[np.ndarray] = []
        sqrt_n = float(np.sqrt(N))
        for r_idx, z_mat in enumerate(Z_blocks):
            z_rms = float(np.linalg.norm(z_mat)) / sqrt_n
            if z_rms > 0.0 and abs(z_rms - 1.0) > 1e-14:
                z_scales[r_idx] = z_rms
                z_norm_blocks.append(z_mat / z_rms)
            else:
                z_norm_blocks.append(z_mat)
        Z_full = np.hstack(z_norm_blocks)
    else:
        Z_full = np.empty((N, 0), dtype=np.float64)
    z_var_scales = z_scales * z_scales

    finite_mask = np.isfinite(Y)
    complete_mask = np.all(finite_mask, axis=0)
    complete_idx = np.flatnonzero(complete_mask)
    incomplete_idx = np.flatnonzero(~complete_mask)

    # Fast batched path for complete genes when weights are shared/uniform
    if complete_idx.size > 0 and shared_weights:
        n_comp = complete_idx.size
        Y_c = Y[:, complete_idx]
        mean_c = np.mean(Y_c, axis=0)
        ave_expr[complete_idx] = mean_c
        n_obs[complete_idx] = N
        tss_c = np.sum((Y_c - mean_c[None, :]) ** 2, axis=0)
        is_const_c = (tss_c / float(N)) <= 1e-24 * (mean_c * mean_c)

        if W is None:
            w_scale = 1.0
            d_w = np.ones(N, dtype=np.float64)
            logdet_w = 0.0
            mean_inv_w = 1.0
        else:
            w_scale = float(np.mean(W))
            w_norm = W / w_scale
            d_w = np.sqrt(w_norm)
            logdet_w = float(np.sum(np.log(w_norm)))
            mean_inv_w = float(np.mean(1.0 / W))
        mean_inv_weight[complete_idx] = mean_inv_w

        X_fit, r_x, is_full_rank, lift_mat, T_proj = _reduce_fixed_design(X_full)
        df_res = float(N - r_x)
        df_residual_ols[complete_idx] = df_res

        X_w = d_w[:, None] * X_fit
        Y_w = d_w[:, None] * Y_c

        Q_X, R_qr = np.linalg.qr(X_w, mode="reduced")
        I_rx = np.eye(r_x, dtype=np.float64)
        R_inv = np.linalg.solve(R_qr, I_rx)
        V_alpha_0 = R_inv @ R_inv.T
        V_alpha_0 = 0.5 * (V_alpha_0 + V_alpha_0.T)
        if is_full_rank:
            V_beta_0 = V_alpha_0
        else:
            assert lift_mat is not None
            V_beta_0 = (lift_mat @ V_alpha_0) @ lift_mat.T
            V_beta_0 = 0.5 * (V_beta_0 + V_beta_0.T)
        cov_beta_fixed[complete_idx] = (V_beta_0 / w_scale)[None, :, :]

        if R == 0:
            alpha_hat = np.linalg.solve(R_qr, Q_X.T @ Y_w)
            resid_w = Y_w - X_w @ alpha_hat
            rss_c = np.sum(resid_w * resid_w, axis=0)
            y_norm_c = np.sum(Y_w * Y_w, axis=0)
            zero_c = (
                is_const_c
                | ((rss_c / df_res) <= 1e-24 * tss_c)
                | (rss_c <= 1e-27 * y_norm_c)
            )
            se2_norm = np.where(zero_c, 0.0, rss_c / df_res)
            se2_vec = se2_norm * w_scale
            logdet_xtx = 2.0 * float(np.sum(np.log(np.abs(np.diag(R_qr)))))
            with np.errstate(divide="ignore"):
                nll_raw = 0.5 * (
                    df_res * np.log(2.0 * np.pi)
                    + df_res * np.log(np.maximum(se2_norm, 1e-300))
                    + logdet_xtx
                    + df_res
                    - logdet_w
                )
            nll_vec = np.where(zero_c, np.nan, nll_raw)

            if is_full_rank:
                betas[complete_idx] = alpha_hat.T
            else:
                assert lift_mat is not None
                betas[complete_idx] = (lift_mat @ alpha_hat).T
            cov_beta_unscaled[complete_idx] = (V_beta_0 / w_scale)[None, :, :]

            sigma_e_sq[complete_idx] = se2_vec
            reml_nll[complete_idx] = nll_vec
            converged[complete_idx] = True
            singular[complete_idx] = False
            if compute_fisher:
                assert fisher_info is not None
                fisher_info[complete_idx, 0, 0] = np.where(
                    zero_c,
                    0.0,
                    df_res / np.maximum(2.0 * se2_vec * se2_vec, 1e-300),
                )

        elif R == 1 and not _force_mme:
            Z_w = d_w[:, None] * Z_full
            (
                alpha_out,
                sr2_out,
                se2_out,
                Va_out,
                u_out,
                nll_out,
                conv_out,
                sing_out,
                fi_out,
                azx_out,
            ) = _solve_spectral_r1_batch(
                Y_w=Y_w,
                X_w=X_w,
                Z_w=Z_w,
                logdet_w=logdet_w,
                df_resid=df_res,
                gamma_max=gamma_max,
                tol=tol,
                compute_fisher=compute_fisher,
                is_const=is_const_c,
                tss_y=tss_c,
            )
            for idx_local, g_idx in enumerate(complete_idx):
                b_g, Vb_g, azx_g = _lift_estimates(
                    alpha_out[idx_local],
                    Va_out[idx_local],
                    azx_out[idx_local] if azx_out is not None else None,
                    is_full_rank,
                    lift_mat,
                    T_proj,
                )
                betas[g_idx] = b_g
                cov_beta_unscaled[g_idx] = Vb_g / w_scale
                if a_zx_arr is not None and azx_g is not None:
                    a_zx_arr[g_idx] = azx_g * (w_scale * z_scales[0])
            sigma_r_sq[complete_idx] = sr2_out / z_var_scales[None, :]
            sigma_e_sq[complete_idx] = se2_out * w_scale
            blups[0][complete_idx] = u_out / z_scales[0]
            reml_nll[complete_idx] = nll_out
            converged[complete_idx] = conv_out
            singular[complete_idx] = sing_out
            if fisher_info is not None and fi_out is not None:
                if w_scale != 1.0 or z_var_scales[0] != 1.0:
                    fi_mult = np.concatenate([z_var_scales, [1.0 / w_scale]])
                    fisher_info[complete_idx] = fi_out * (
                        fi_mult[None, :, None] * fi_mult[None, None, :]
                    )
                else:
                    fisher_info[complete_idx] = fi_out

        else:
            # General R >= 1 batched MME sufficient statistics
            Z_w = d_w[:, None] * Z_full
            S_XX = X_w.T @ X_w
            S_ZZ = Z_w.T @ Z_w
            S_XZ = X_w.T @ Z_w
            alpha_ols_all = np.linalg.solve(R_qr, Q_X.T @ Y_w)
            E_ols = Y_w - X_w @ alpha_ols_all
            S_Zy_all = Z_w.T @ E_ols
            s_yy_all = np.sum(E_ols * E_ols, axis=0)
            y_norm_all = np.sum(Y_w * Y_w, axis=0)
            zero_c = (
                is_const_c
                | ((s_yy_all / df_res) <= 1e-24 * tss_c)
                | (s_yy_all <= 1e-27 * y_norm_all)
            )

            def _fit_one_complete(local_i: int):
                return _solve_mme_single_gene(
                    S_XX=S_XX,
                    S_ZZ=S_ZZ,
                    S_XZ=S_XZ,
                    s_Zy_resid=S_Zy_all[:, local_i],
                    s_yy_resid=float(s_yy_all[local_i]),
                    alpha_ols=alpha_ols_all[:, local_i],
                    y_w_norm_sq=float(y_norm_all[local_i]),
                    logdet_w=logdet_w,
                    df_resid=df_res,
                    block_sizes=block_sizes,
                    block_slices=block_slices,
                    gamma_max=gamma_max,
                    max_iter=max_iter,
                    tol=tol,
                    compute_fisher=compute_fisher,
                    is_zero_resid=bool(zero_c[local_i]),
                )

            results_list = _parallel_map(_fit_one_complete, range(n_comp), n_jobs=n_jobs)
            fi_mult_mme = np.concatenate([z_var_scales, [1.0 / w_scale]])
            scale_mme_z = bool(np.any(z_scales != 1.0))
            for local_i, res_g in enumerate(results_list):
                g_idx = int(complete_idx[local_i])
                alpha_g, sr2_g, se2_g, Va_g, u_g, nll_g, conv_g, sing_g, fi_g, azx_g = res_g
                b_g, Vb_g, azx_lifted = _lift_estimates(
                    alpha_g, Va_g, azx_g, is_full_rank, lift_mat, T_proj
                )
                betas[g_idx] = b_g
                sigma_r_sq[g_idx] = sr2_g / z_var_scales
                sigma_e_sq[g_idx] = se2_g * w_scale
                cov_beta_unscaled[g_idx] = Vb_g / w_scale
                for r_idx, sl in enumerate(block_slices):
                    blups[r_idx][g_idx] = u_g[sl] / z_scales[r_idx]
                reml_nll[g_idx] = nll_g
                converged[g_idx] = conv_g
                singular[g_idx] = sing_g
                if fisher_info is not None and fi_g is not None:
                    if w_scale != 1.0 or scale_mme_z:
                        fisher_info[g_idx] = fi_g * (fi_mult_mme[:, None] * fi_mult_mme[None, :])
                    else:
                        fisher_info[g_idx] = fi_g
                if a_zx_arr is not None and azx_lifted is not None:
                    azx_out_g = azx_lifted * w_scale
                    if scale_mme_z:
                        for r_idx, sl in enumerate(block_slices):
                            azx_out_g[sl] *= z_scales[r_idx]
                    a_zx_arr[g_idx] = azx_out_g

    # Per-gene path for gene-specific weights and/or partial-NaN genes
    per_gene_indices = (
        np.arange(G, dtype=np.int64) if not shared_weights else incomplete_idx
    )
    if per_gene_indices.size > 0:
        # Pre-check rank for complete design when only weights vary
        full_design_reduced = _reduce_fixed_design(X_full) if not shared_weights else None

        def _fit_per_gene(g_idx_int: int):
            obs = finite_mask[:, g_idx_int]
            n_g = int(np.sum(obs))
            if n_g == 0:
                return None

            y_g = Y[obs, g_idx_int]
            ave_g = float(np.mean(y_g))
            tss_g = float(np.sum((y_g - ave_g) ** 2))
            is_const_g = (tss_g / float(n_g)) <= 1e-24 * (ave_g * ave_g)

            if W is None:
                w_g = np.ones(n_g, dtype=np.float64)
            elif W.ndim == 1:
                w_g = W[obs]
            else:
                w_g = W[obs, g_idx_int]

            if n_g == N and full_design_reduced is not None:
                X_fit, r_x, is_full_rank, lift_mat, T_proj = full_design_reduced
                Z_g = Z_full
            else:
                X_obs = X_full[obs]
                X_fit, r_x, is_full_rank, lift_mat, T_proj = _reduce_fixed_design(X_obs)
                Z_g = Z_full[obs] if R > 0 else np.empty((n_g, 0), dtype=np.float64)

            df_res_g = float(n_g - r_x)
            mean_inv_w_g = float(np.mean(1.0 / w_g))
            if df_res_g < 1.0:
                return (n_g, ave_g, df_res_g, mean_inv_w_g, None)

            w_scale_g = float(np.mean(w_g))
            w_norm_g = w_g / w_scale_g
            d_g = np.sqrt(w_norm_g)
            logdet_w_g = float(np.sum(np.log(w_norm_g)))
            X_w = d_g[:, None] * X_fit
            y_w = d_g * y_g

            Q_X, R_qr = np.linalg.qr(X_w, mode="reduced")
            alpha_ols = np.linalg.solve(R_qr, Q_X.T @ y_w)
            I_rx = np.eye(r_x, dtype=np.float64)
            R_inv = np.linalg.solve(R_qr, I_rx)
            V_alpha_0 = R_inv @ R_inv.T
            V_alpha_0 = 0.5 * (V_alpha_0 + V_alpha_0.T)
            if is_full_rank:
                Vb_0_g = V_alpha_0
            else:
                assert lift_mat is not None
                Vb_0_g = (lift_mat @ V_alpha_0) @ lift_mat.T
                Vb_0_g = 0.5 * (Vb_0_g + Vb_0_g.T)

            if R == 0:
                resid_w = y_w - X_w @ alpha_ols
                rss_g = float(resid_w @ resid_w)
                y_norm_g = float(y_w @ y_w)
                is_zero_g = (
                    is_const_g
                    or ((rss_g / df_res_g) <= 1e-24 * tss_g)
                    or (rss_g <= 1e-27 * y_norm_g)
                )
                if is_zero_g:
                    se2_g = 0.0
                    nll_g = float("nan")
                    fi_g = np.zeros((1, 1), dtype=np.float64) if compute_fisher else None
                else:
                    se2_norm = rss_g / df_res_g
                    se2_g = se2_norm * w_scale_g
                    logdet_xtx = 2.0 * float(np.sum(np.log(np.abs(np.diag(R_qr)))))
                    nll_g = 0.5 * (
                        df_res_g * np.log(2.0 * np.pi)
                        + df_res_g * np.log(max(se2_norm, 1e-300))
                        + logdet_xtx
                        + df_res_g
                        - logdet_w_g
                    )
                    fi_g = (
                        np.array([[df_res_g / max(2.0 * se2_g * se2_g, 1e-300)]], dtype=np.float64)
                        if compute_fisher
                        else None
                    )
                b_g, _, _ = _lift_estimates(
                    alpha_ols, V_alpha_0, None, is_full_rank, lift_mat, T_proj
                )
                res_tuple = (
                    b_g,
                    np.empty(0, dtype=np.float64),
                    se2_g,
                    Vb_0_g / w_scale_g,
                    Vb_0_g / w_scale_g,
                    np.empty(0, dtype=np.float64),
                    nll_g,
                    True,
                    False,
                    fi_g,
                    None,
                )
                return (n_g, ave_g, df_res_g, mean_inv_w_g, res_tuple)

            Z_w = d_g[:, None] * Z_g
            e_ols = y_w - X_w @ alpha_ols
            S_XX = X_w.T @ X_w
            S_ZZ = Z_w.T @ Z_w
            S_XZ = X_w.T @ Z_w
            s_Zy_resid = Z_w.T @ e_ols
            s_yy_resid = float(e_ols @ e_ols)
            y_norm_g = float(y_w @ y_w)
            is_zero_g = (
                is_const_g
                or ((s_yy_resid / df_res_g) <= 1e-24 * tss_g)
                or (s_yy_resid <= 1e-27 * y_norm_g)
            )

            (
                alpha_g,
                sr2_g,
                se2_norm,
                Va_g,
                u_g,
                nll_g,
                conv_g,
                sing_g,
                fi_g,
                azx_g,
            ) = _solve_mme_single_gene(
                S_XX=S_XX,
                S_ZZ=S_ZZ,
                S_XZ=S_XZ,
                s_Zy_resid=s_Zy_resid,
                s_yy_resid=s_yy_resid,
                alpha_ols=alpha_ols,
                y_w_norm_sq=y_norm_g,
                logdet_w=logdet_w_g,
                df_resid=df_res_g,
                block_sizes=block_sizes,
                block_slices=block_slices,
                gamma_max=gamma_max,
                max_iter=max_iter,
                tol=tol,
                compute_fisher=compute_fisher,
                is_zero_resid=is_zero_g,
            )
            b_g, Vb_g, azx_lifted = _lift_estimates(
                alpha_g, Va_g, azx_g, is_full_rank, lift_mat, T_proj
            )
            scale_z_pg = bool(np.any(z_scales != 1.0))
            if fi_g is not None and (w_scale_g != 1.0 or scale_z_pg):
                fi_mult_pg = np.concatenate([z_var_scales, [1.0 / w_scale_g]])
                fi_g = fi_g * (fi_mult_pg[:, None] * fi_mult_pg[None, :])
            if scale_z_pg:
                u_g = u_g.copy()
                for r_idx, sl in enumerate(block_slices):
                    u_g[sl] /= z_scales[r_idx]
            azx_out_pg: np.ndarray | None = None
            if azx_lifted is not None:
                azx_out_pg = azx_lifted * w_scale_g
                if scale_z_pg:
                    for r_idx, sl in enumerate(block_slices):
                        azx_out_pg[sl] *= z_scales[r_idx]
            res_tuple = (
                b_g,
                sr2_g / z_var_scales,
                se2_norm * w_scale_g,
                Vb_g / w_scale_g,
                Vb_0_g / w_scale_g,
                u_g,
                nll_g,
                conv_g,
                sing_g,
                fi_g,
                azx_out_pg,
            )
            return (n_g, ave_g, df_res_g, mean_inv_w_g, res_tuple)

        per_gene_results = _parallel_map(
            _fit_per_gene, [int(g) for g in per_gene_indices], n_jobs=n_jobs
        )
        for g_idx_int, out_g in zip(per_gene_indices, per_gene_results):
            g_i = int(g_idx_int)
            if out_g is None:
                continue
            n_g, ave_g, df_res_g, mean_inv_w_g, res_tuple = out_g
            n_obs[g_i] = n_g
            ave_expr[g_i] = ave_g
            df_residual_ols[g_i] = df_res_g
            mean_inv_weight[g_i] = mean_inv_w_g
            if res_tuple is None:
                continue
            b_g, sr2_g, se2_g, Vb_g, Vb_0_g_scaled, u_g, nll_g, conv_g, sing_g, fi_g, azx_lifted = res_tuple
            betas[g_i] = b_g
            sigma_r_sq[g_i] = sr2_g
            sigma_e_sq[g_i] = se2_g
            cov_beta_unscaled[g_i] = Vb_g
            cov_beta_fixed[g_i] = Vb_0_g_scaled
            for r_idx, sl in enumerate(block_slices):
                blups[r_idx][g_i] = u_g[sl]
            reml_nll[g_i] = nll_g
            converged[g_i] = conv_g
            singular[g_i] = sing_g
            if fisher_info is not None and fi_g is not None:
                fisher_info[g_i] = fi_g
            if a_zx_arr is not None and azx_lifted is not None:
                a_zx_arr[g_i] = azx_lifted

    return LMMFitResult(
        gene_names=resolved_gene_names,
        design=design_spec,
        betas=betas,
        sigma_r_sq=sigma_r_sq,
        sigma_e_sq=sigma_e_sq,
        cov_beta_unscaled=cov_beta_unscaled,
        blups=blups,
        reml_nll=reml_nll,
        ave_expr=ave_expr,
        n_obs=n_obs,
        df_residual_ols=df_residual_ols,
        mean_inv_weight=mean_inv_weight,
        converged=converged,
        singular=singular,
        fisher_info=fisher_info,
        a_zx=a_zx_arr,
        cov_beta_fixed=cov_beta_fixed,
    )


def _parallel_map(fn, items, n_jobs: int = 1) -> list[Any]:
    items_list = list(items)
    if n_jobs == 1 or len(items_list) <= 1:
        return [fn(x) for x in items_list]
    try:
        from joblib import Parallel, delayed  # type: ignore

        return list(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fn)(x) for x in items_list))
    except ImportError:
        max_workers = min(len(items_list), max(1, n_jobs if n_jobs > 0 else 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, items_list))


fit_var_part_model = fit_lmm
