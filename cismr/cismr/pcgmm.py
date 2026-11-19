"""Principal Component Generalized Method of Moments (PC-GMM & MV-PC-GMM) for cis-MR.

References
----------
Burgess, S., Zuber, V., Valdes-Marquez, E., Sun, B. B., & Hopewell, J. C. (2017/2020).
Mendelian randomization with fine-mapped genetic variants: Choosing from all variants
in a gene region. Genetic Epidemiology, 41(8), 714-725 / 10.1002/gepi.22295.

Grant, A. J., & Burgess, S. (2022). An efficient and robust approach to Mendelian
randomization with measured pleiotropic variants in a high-dimensional setting.
Statistics in Medicine, 41(12), 2132-2147. https://doi.org/10.1002/sim.9278
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .utils import (
    SettingsType,
    as_1d_float,
    as_2d_float,
    prepare_ld_matrix,
    sym_solve,
    validate_settings,
    wald_inference,
)


@dataclass
class PCGMMResult:
    """Result of univariable correlated cis-MR via PC-GMM or PC-IVW."""

    estimate: float
    se: float
    pval: float
    ci_low: float
    ci_high: float
    n_snps: int
    n_pcs: int
    f_stat: float
    j_stat: float
    j_df: int
    j_pval: float
    var_explained: float
    converged: bool
    n_iter: int
    robust: bool
    thres: float
    method: str
    settings: SettingsType

    @property
    def theta(self) -> float:
        return self.estimate

    @property
    def beta(self) -> float:
        return self.estimate

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pval": self.pval,
            "n_snps": self.n_snps,
            "n_pcs": self.n_pcs,
            "f_stat": self.f_stat,
            "j_stat": self.j_stat,
            "j_df": self.j_df,
            "j_pval": self.j_pval,
            "var_explained": self.var_explained,
            "converged": self.converged,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"{self.method} (settings={self.settings!r}): "
            f"theta={self.estimate:.4f} (SE={self.se:.4f}, p={self.pval:.3e}), "
            f"95% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"PCs={self.n_pcs}/{self.n_snps} ({100.0 * self.var_explained:.2f}% var), "
            f"F={self.f_stat:.2f}, J={self.j_stat:.2f} (df={self.j_df}, p={self.j_pval:.3g})"
        )


@dataclass
class MVPCGMMResult:
    """Result of multivariable correlated cis-MR via MV-PC-GMM."""

    estimates: np.ndarray
    ses: np.ndarray
    pvals: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cov: np.ndarray
    cond_f_stats: np.ndarray
    n_snps: int
    n_exposures: int
    n_pcs: int
    j_stat: float
    j_df: int
    j_pval: float
    var_explained: float
    converged: bool
    n_iter: int
    robust: bool
    thres: float
    settings: SettingsType
    exposure_names: tuple[str, ...] = ()

    def to_dataframe(self) -> pd.DataFrame:
        names = (
            list(self.exposure_names)
            if len(self.exposure_names) == self.n_exposures
            else [f"exposure_{m + 1}" for m in range(self.n_exposures)]
        )
        return pd.DataFrame(
            {
                "exposure": names,
                "estimate": self.estimates,
                "se": self.ses,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "pval": self.pvals,
                "cond_f_stat": self.cond_f_stats,
                "n_pcs": self.n_pcs,
                "j_stat": self.j_stat,
                "j_df": self.j_df,
                "j_pval": self.j_pval,
            }
        )

    def summary(self) -> str:
        df = self.to_dataframe()
        header = (
            f"MV-PC-GMM (settings={self.settings!r}, PCs={self.n_pcs}/{self.n_snps}, "
            f"J={self.j_stat:.2f}, df={self.j_df}, p_J={self.j_pval:.3g}):"
        )
        rows = [
            f"  {row.exposure}: theta={row.estimate:.4f} (SE={row.se:.4f}, "
            f"p={row.pval:.3e}), cond_F={row.cond_f_stat:.2f}"
            for row in df.itertuples()
        ]
        return "\n".join([header, *rows])


def _extract_pcs(
    phi0: np.ndarray,
    thres: float,
    min_pcs: int,
    settings: SettingsType,
) -> tuple[np.ndarray, int, float]:
    """Extract top principal components explaining ``thres`` variance of ``phi0``."""
    phi_sym = 0.5 * (phi0 + phi0.T)
    eigvals, eigvecs = linalg.eigh(phi_sym)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos_vals = np.maximum(eigvals, 0.0)
    total_var = float(np.sum(pos_vals))
    if total_var <= 0.0:
        raise ValueError("Exposure-weighted LD matrix Phi0 has non-positive total variance.")

    cum_ratio = np.cumsum(pos_vals) / total_var
    if settings == "R_default":
        above = np.where(cum_ratio > thres)[0]
        k = int(above[0] + 1) if above.size > 0 else int(np.sum(eigvals > 0.0))
    else:
        above = np.where(cum_ratio >= thres - 1e-15)[0]
        k = int(above[0] + 1) if above.size > 0 else int(np.sum(pos_vals > 0.0))

    max_pos = max(int(np.sum(pos_vals > 1e-12 * total_var)), min_pcs)
    k = max(min_pcs, min(k, max_pos, phi0.shape[0]))
    A = eigvecs[:, :k].copy()
    # Deterministic column sign convention
    for col in range(k):
        max_idx = int(np.argmax(np.abs(A[:, col])))
        if A[max_idx, col] < 0.0:
            A[:, col] = -A[:, col]
    var_explained = float(np.sum(pos_vals[:k]) / total_var)
    return A, k, var_explained


def _compute_pc_fstat(
    bx: np.ndarray,
    sx: np.ndarray,
    R: np.ndarray,
    A: np.ndarray,
    pc_bx: np.ndarray,
    pc_Sx: np.ndarray,
    nx: int | float | None,
) -> float:
    """Compute first-stage PC instrument F-statistic (Burgess et al., 2020)."""
    k = A.shape[1]
    if nx is not None and nx > k + 2:
        df_denom = float(nx) - float(k) - 1.0
        inv_sqrt_v = np.sqrt(np.maximum((float(nx) - 2.0) * (sx**2) + bx**2, 1e-15))
        M_x = A.T @ ((R * inv_sqrt_v[:, None]) * inv_sqrt_v[None, :]) @ A
        r2_pc = float(pc_bx @ sym_solve(M_x, pc_bx))
        if 0.0 < r2_pc < 0.995:
            return float((df_denom / k) * (r2_pc / (1.0 - r2_pc)))
    wald_x = float(pc_bx @ sym_solve(pc_Sx, pc_bx))
    return max(wald_x / max(k, 1), 0.0)


def mr_pcgmm(
    bx: np.ndarray | Sequence[float],
    sx: np.ndarray | Sequence[float],
    by: np.ndarray | Sequence[float],
    sy: np.ndarray | Sequence[float],
    ld_R: np.ndarray,
    nx: int | float,
    ny: int | float | None = None,
    thres: float = 0.99,
    robust: bool = True,
    max_iter: int = 50,
    alpha: float = 0.05,
    settings: SettingsType = "default",
) -> PCGMMResult:
    """Univariable correlated cis-MR using Principal Component GMM (PC-GMM).

    Parameters
    ----------
    bx, sx : array-like of shape (p,)
        Variant-exposure association estimates and standard errors.
    by, sy : array-like of shape (p,)
        Variant-outcome association estimates and standard errors.
    ld_R : ndarray of shape (p, p)
        Signed LD correlation matrix across the ``p`` cis-variants.
    nx : int or float
        Exposure GWAS sample size (used for first-stage PC F-statistic).
    ny : int or float, optional
        Outcome GWAS sample size.
    thres : float, default=0.99
        Proportion of variance in ``Phi0 = outer(bx/sy, bx/sy) * R`` explained
        by the retained principal components ``(0 < thres <= 1)``.
    robust : bool, default=True
        Scale standard error by overdispersion factor ``sqrt(max(1, J / (K - 1)))``.
    max_iter : int, default=50
        Maximum iterations for multi-step / continuously updating GMM in
        ``settings="default"``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    settings : {"default", "R_default"}, default="default"
        Execution mode.
    """
    validate_settings(settings)
    bx_arr = as_1d_float(bx, "bx", min_len=2)
    p = bx_arr.shape[0]
    sx_arr = as_1d_float(sx, "sx", min_len=p)
    by_arr = as_1d_float(by, "by", min_len=p)
    sy_arr = as_1d_float(sy, "sy", min_len=p)
    if not (sx_arr.shape[0] == by_arr.shape[0] == sy_arr.shape[0] == p):
        raise ValueError("bx, sx, by, sy must all have the same length.")
    if np.any(sx_arr <= 0.0) or np.any(sy_arr <= 0.0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")
    if not (0.0 < thres <= 1.0):
        raise ValueError(f"thres must be in (0, 1], got {thres}.")
    if nx is not None and nx <= 2:
        raise ValueError(f"nx must be > 2, got {nx}.")
    if ny is not None and ny <= 2:
        raise ValueError(f"ny must be > 2, got {ny}.")
    if np.all(bx_arr == 0.0):
        raise ValueError("Exposure associations bx cannot be all zero.")

    R = prepare_ld_matrix(ld_R, p, settings=settings)
    w = bx_arr / sy_arr
    phi0 = np.outer(w, w) * R
    A, k, var_explained = _extract_pcs(phi0, thres=thres, min_pcs=1, settings=settings)

    Sigma_X = (R * sx_arr[:, None]) * sx_arr[None, :]
    Sigma_Y = (R * sy_arr[:, None]) * sy_arr[None, :]
    pc_bx = A.T @ bx_arr
    pc_by = A.T @ by_arr
    pc_Sx = 0.5 * (A.T @ Sigma_X @ A + (A.T @ Sigma_X @ A).T)
    pc_Sy = 0.5 * (A.T @ Sigma_Y @ A + (A.T @ Sigma_Y @ A).T)

    # Step 1: initial PC-IVW estimate (Omega = pc_Sy)
    w0_bx = sym_solve(pc_Sy, pc_bx)
    denom0 = float(pc_bx @ w0_bx)
    if denom0 <= 0.0:
        raise ValueError("Degenerate PC exposure projection (pc_bx' Sy^-1 pc_bx <= 0).")
    theta_init = float((pc_bx @ sym_solve(pc_Sy, pc_by)) / denom0)

    def gmm_obj(theta_val: float) -> float:
        g = pc_by - theta_val * pc_bx
        omega = pc_Sy + (theta_val * theta_val) * pc_Sx
        return float(g @ sym_solve(omega, g))

    if settings == "R_default":
        # Exact 2-step GMM update matching MendelianRandomization::mr_pcgmm
        omega_1 = pc_Sy + (theta_init * theta_init) * pc_Sx
        w1_bx = sym_solve(omega_1, pc_bx)
        theta_hat = float((pc_bx @ sym_solve(omega_1, pc_by)) / float(pc_bx @ w1_bx))
        converged = True
        n_iter = 2
    else:
        # Multi-step iterative GMM (tol=1e-8) + local CUE refinement
        theta_curr = theta_init
        converged = False
        n_iter = 1
        for it in range(1, max(max_iter, 1) + 1):
            omega_t = pc_Sy + (theta_curr * theta_curr) * pc_Sx
            wt_bx = sym_solve(omega_t, pc_bx)
            theta_next = float((pc_bx @ sym_solve(omega_t, pc_by)) / float(pc_bx @ wt_bx))
            n_iter = it
            if abs(theta_next - theta_curr) <= 1e-8 * (1.0 + abs(theta_curr)):
                theta_curr = theta_next
                converged = True
                break
            theta_curr = theta_next

        theta_hat = theta_curr
        if k > 1:
            omega_iter = pc_Sy + (theta_curr * theta_curr) * pc_Sx
            se_iter = 1.0 / np.sqrt(max(float(pc_bx @ sym_solve(omega_iter, pc_bx)), 1e-15))
            rad = 3.0 * se_iter + abs(theta_curr) + 0.5
            opt = optimize.minimize_scalar(
                gmm_obj,
                bounds=(theta_curr - rad, theta_curr + rad),
                method="bounded",
                options={"xatol": 1e-8, "maxiter": max(max_iter, 50)},
            )
            if opt.success and np.isfinite(opt.x) and float(opt.fun) <= gmm_obj(theta_curr) + 1e-12:
                theta_hat = float(opt.x)
                converged = True

    omega_final = pc_Sy + (theta_hat * theta_hat) * pc_Sx
    g_final = pc_by - theta_hat * pc_bx
    j_stat = max(float(g_final @ sym_solve(omega_final, g_final)), 0.0) if k > 1 else 0.0
    j_df = max(k - 1, 0)
    j_pval = float(stats.chi2.sf(j_stat, df=j_df)) if j_df > 0 else np.nan

    info = float(pc_bx @ sym_solve(omega_final, pc_bx))
    se_base = 1.0 / np.sqrt(max(info, 1e-18))
    if robust and j_df > 0:
        se_hat = float(se_base * np.sqrt(max(1.0, j_stat / j_df)))
    else:
        se_hat = float(se_base)

    df_crit = j_df if (settings == "R_default" and robust and j_df > 0) else None
    pval, ci_low, ci_high = wald_inference(theta_hat, se_hat, alpha=alpha, df=df_crit)
    f_stat = _compute_pc_fstat(bx_arr, sx_arr, R, A, pc_bx, pc_Sx, nx)

    return PCGMMResult(
        estimate=theta_hat,
        se=se_hat,
        pval=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_snps=p,
        n_pcs=k,
        f_stat=f_stat,
        j_stat=j_stat,
        j_df=j_df,
        j_pval=j_pval,
        var_explained=var_explained,
        converged=converged,
        n_iter=n_iter,
        robust=robust,
        thres=thres,
        method="PC-GMM",
        settings=settings,
    )


def mr_pcivw(
    bx: np.ndarray | Sequence[float],
    sx: np.ndarray | Sequence[float],
    by: np.ndarray | Sequence[float],
    sy: np.ndarray | Sequence[float],
    ld_R: np.ndarray,
    nx: int | float | None = None,
    thres: float = 0.99,
    robust: bool = True,
    alpha: float = 0.05,
    settings: SettingsType = "default",
) -> PCGMMResult:
    """First-order correlated Principal Component IVW (Burgess et al., 2017)."""
    validate_settings(settings)
    bx_arr = as_1d_float(bx, "bx", min_len=2)
    p = bx_arr.shape[0]
    sx_arr = as_1d_float(sx, "sx", min_len=p)
    by_arr = as_1d_float(by, "by", min_len=p)
    sy_arr = as_1d_float(sy, "sy", min_len=p)
    if not (sx_arr.shape[0] == by_arr.shape[0] == sy_arr.shape[0] == p):
        raise ValueError("bx, sx, by, sy must all have the same length.")
    if np.any(sx_arr <= 0.0) or np.any(sy_arr <= 0.0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")
    if not (0.0 < thres <= 1.0):
        raise ValueError(f"thres must be in (0, 1], got {thres}.")

    R = prepare_ld_matrix(ld_R, p, settings=settings)
    w = bx_arr / sy_arr
    phi0 = np.outer(w, w) * R
    A, k, var_explained = _extract_pcs(phi0, thres=thres, min_pcs=1, settings=settings)

    Sigma_X = (R * sx_arr[:, None]) * sx_arr[None, :]
    Sigma_Y = (R * sy_arr[:, None]) * sy_arr[None, :]
    pc_bx = A.T @ bx_arr
    pc_by = A.T @ by_arr
    pc_Sx = 0.5 * (A.T @ Sigma_X @ A + (A.T @ Sigma_X @ A).T)
    pc_Sy = 0.5 * (A.T @ Sigma_Y @ A + (A.T @ Sigma_Y @ A).T)

    w_bx = sym_solve(pc_Sy, pc_bx)
    denom = float(pc_bx @ w_bx)
    theta_hat = float((pc_bx @ sym_solve(pc_Sy, pc_by)) / denom)
    resid = pc_by - theta_hat * pc_bx
    q_stat = max(float(resid @ sym_solve(pc_Sy, resid)), 0.0) if k > 1 else 0.0
    q_df = max(k - 1, 0)
    q_pval = float(stats.chi2.sf(q_stat, df=q_df)) if q_df > 0 else np.nan

    se_base = 1.0 / np.sqrt(max(denom, 1e-18))
    se_hat = float(se_base * np.sqrt(max(1.0, q_stat / q_df))) if (robust and q_df > 0) else se_base
    df_crit = q_df if (settings == "R_default" and robust and q_df > 0) else None
    pval, ci_low, ci_high = wald_inference(theta_hat, se_hat, alpha=alpha, df=df_crit)
    f_stat = _compute_pc_fstat(bx_arr, sx_arr, R, A, pc_bx, pc_Sx, nx)

    return PCGMMResult(
        estimate=theta_hat,
        se=se_hat,
        pval=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_snps=p,
        n_pcs=k,
        f_stat=f_stat,
        j_stat=q_stat,
        j_df=q_df,
        j_pval=q_pval,
        var_explained=var_explained,
        converged=True,
        n_iter=1,
        robust=robust,
        thres=thres,
        method="PC-IVW",
        settings=settings,
    )


def _compute_conditional_pc_fstats(
    pc_Bx: np.ndarray,
    Sx: np.ndarray,
    R: np.ndarray,
    A: np.ndarray,
    cor_x: np.ndarray,
) -> np.ndarray:
    """Compute Sanderson-Windmeijer conditional PC F-statistics (Grant & Burgess, 2022)."""
    k, m_exp = pc_Bx.shape
    cond_f = np.zeros(m_exp, dtype=np.float64)
    if m_exp == 1:
        Sigma_X1 = (R * Sx[:, 0][:, None]) * Sx[:, 0][None, :]
        pc_Sx1 = A.T @ Sigma_X1 @ A
        cond_f[0] = max(float(pc_Bx[:, 0] @ sym_solve(pc_Sx1, pc_Bx[:, 0])) / max(k, 1), 0.0)
        return cond_f

    df_cond = max(k - m_exp + 1, 1)
    for m in range(m_exp):
        other = [j for j in range(m_exp) if j != m]
        b_m = pc_Bx[:, m]
        B_other = pc_Bx[:, other]

        def omega_xm(delta_vec: np.ndarray) -> np.ndarray:
            d = np.zeros(m_exp, dtype=np.float64)
            d[m] = 1.0
            d[other] = -delta_vec
            U = Sx * d[None, :]
            Sigma_d = R * (U @ cor_x @ U.T)
            return 0.5 * (A.T @ Sigma_d @ A + (A.T @ Sigma_d @ A).T)

        omega0 = omega_xm(np.zeros(m_exp - 1, dtype=np.float64))
        lhs0 = B_other.T @ sym_solve(omega0, B_other)
        rhs0 = B_other.T @ sym_solve(omega0, b_m)
        delta = sym_solve(lhs0, rhs0)
        for _ in range(5):
            om = omega_xm(delta)
            lhs = B_other.T @ sym_solve(om, B_other)
            rhs = B_other.T @ sym_solve(om, b_m)
            delta_new = sym_solve(lhs, rhs)
            if np.max(np.abs(delta_new - delta)) < 1e-7:
                delta = delta_new
                break
            delta = delta_new

        om_final = omega_xm(delta)
        resid_m = b_m - B_other @ delta
        q_m = max(float(resid_m @ sym_solve(om_final, resid_m)), 0.0)
        cond_f[m] = q_m / df_cond
    return cond_f


def mr_mvpcgmm(
    Bx: np.ndarray | Sequence[Sequence[float]],
    Sx: np.ndarray | Sequence[Sequence[float]],
    by: np.ndarray | Sequence[float],
    sy: np.ndarray | Sequence[float],
    ld_R: np.ndarray,
    nx: int | float | Sequence[int | float],
    ny: int | float | None = None,
    thres: float = 0.99,
    robust: bool = True,
    cor_x: np.ndarray | None = None,
    max_iter: int = 50,
    alpha: float = 0.05,
    exposure_names: Sequence[str] | None = None,
    settings: SettingsType = "default",
) -> MVPCGMMResult:
    """Multivariable correlated cis-MR via MV-PC-GMM (Grant & Burgess, 2022).

    Parameters
    ----------
    Bx, Sx : array-like of shape (p, M)
        Variant-exposure association estimates and standard errors for ``p``
        correlated cis-SNPs across ``M`` molecular exposures.
    by, sy : array-like of shape (p,)
        Variant-outcome association estimates and standard errors.
    ld_R : ndarray of shape (p, p)
        Signed LD correlation matrix across the ``p`` cis-variants.
    nx : int, float, or sequence of length M
        Exposure GWAS sample size(s).
    ny : int or float, optional
        Outcome GWAS sample size.
    thres : float, default=0.99
        Cumulative variance threshold for principal component extraction.
    robust : bool, default=True
        Apply overdispersion-robust covariance scaling ``max(1, J / (K - M))``.
    cor_x : ndarray of shape (M, M), optional
        Phenotypic correlation matrix across the ``M`` exposures (defaults to ``I_M``).
    max_iter : int, default=50
        Maximum iterations for multi-step / CUE GMM in ``settings="default"``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    exposure_names : sequence of str, optional
        Optional labels for the ``M`` exposures.
    settings : {"default", "R_default"}, default="default"
        Execution mode.
    """
    validate_settings(settings)
    Bx_arr = as_2d_float(Bx, "Bx", min_rows=2)
    Sx_arr = as_2d_float(Sx, "Sx", min_rows=2)
    p, m_exp = Bx_arr.shape
    if Sx_arr.shape != (p, m_exp):
        raise ValueError(f"Sx must match Bx shape {(p, m_exp)}, got {Sx_arr.shape}.")
    if p <= m_exp:
        raise ValueError(
            f"Number of cis-variants p={p} must exceed number of exposures M={m_exp}."
        )
    by_arr = as_1d_float(by, "by", min_len=p)
    sy_arr = as_1d_float(sy, "sy", min_len=p)
    if by_arr.shape[0] != p or sy_arr.shape[0] != p:
        raise ValueError("by and sy must have length p matching Bx.shape[0].")
    if np.any(Sx_arr <= 0.0) or np.any(sy_arr <= 0.0):
        raise ValueError("Standard errors Sx and sy must be strictly positive.")
    if not (0.0 < thres <= 1.0):
        raise ValueError(f"thres must be in (0, 1], got {thres}.")
    nx_arr = np.atleast_1d(np.asarray(nx, dtype=np.float64))
    if np.any(nx_arr <= 2.0):
        raise ValueError("Exposure sample size nx must be > 2.")
    if ny is not None and ny <= 2:
        raise ValueError("Outcome sample size ny must be > 2.")

    if cor_x is None:
        Cx = np.eye(m_exp, dtype=np.float64)
    else:
        Cx = np.asarray(cor_x, dtype=np.float64)
        if Cx.shape != (m_exp, m_exp):
            raise ValueError(f"cor_x must have shape {(m_exp, m_exp)}, got {Cx.shape}.")
        Cx = 0.5 * (Cx + Cx.T)

    R = prepare_ld_matrix(ld_R, p, settings=settings)
    if settings == "R_default":
        w = 1.0 / sy_arr
    else:
        w = np.linalg.norm(Bx_arr, axis=1) / sy_arr
        if np.all(w == 0.0):
            w = 1.0 / sy_arr

    phi0 = np.outer(w, w) * R
    A, k, var_explained = _extract_pcs(
        phi0, thres=thres, min_pcs=min(p, m_exp + 1), settings=settings
    )
    if k < m_exp:
        raise ValueError(f"Extracted only K={k} PCs, need at least M={m_exp}.")

    Sigma_Y = (R * sy_arr[:, None]) * sy_arr[None, :]
    pc_Bx = A.T @ Bx_arr
    pc_by = A.T @ by_arr
    pc_Sy = 0.5 * (A.T @ Sigma_Y @ A + (A.T @ Sigma_Y @ A).T)

    def omega_theta(theta_vec: np.ndarray) -> np.ndarray:
        U = Sx_arr * theta_vec[None, :]
        Sigma_Xt = R * (U @ Cx @ U.T)
        om = pc_Sy + A.T @ Sigma_Xt @ A
        return 0.5 * (om + om.T)

    # Step 1: initial MV-PC-IVW estimate
    lhs0 = pc_Bx.T @ sym_solve(pc_Sy, pc_Bx)
    rhs0 = pc_Bx.T @ sym_solve(pc_Sy, pc_by)
    theta0 = sym_solve(lhs0, rhs0)

    def gmm_mv_obj(theta_vec: np.ndarray) -> float:
        g = pc_by - pc_Bx @ theta_vec
        om = omega_theta(theta_vec)
        return float(g @ sym_solve(om, g))

    if settings == "R_default":
        om1 = omega_theta(theta0)
        theta_hat = sym_solve(
            pc_Bx.T @ sym_solve(om1, pc_Bx),
            pc_Bx.T @ sym_solve(om1, pc_by),
        )
        converged = True
        n_iter = 2
    else:
        theta_curr = theta0.copy()
        converged = False
        n_iter = 1
        for it in range(1, max(max_iter, 1) + 1):
            omt = omega_theta(theta_curr)
            theta_next = sym_solve(
                pc_Bx.T @ sym_solve(omt, pc_Bx),
                pc_Bx.T @ sym_solve(omt, pc_by),
            )
            n_iter = it
            if np.max(np.abs(theta_next - theta_curr)) <= 1e-8 * (
                1.0 + np.max(np.abs(theta_curr))
            ):
                theta_curr = theta_next
                converged = True
                break
            theta_curr = theta_next

        theta_hat = theta_curr
        if k > m_exp:
            om_iter = omega_theta(theta_curr)
            cov_iter = sym_solve(
                pc_Bx.T @ sym_solve(om_iter, pc_Bx), np.eye(m_exp, dtype=np.float64)
            )
            se_iter = np.sqrt(np.maximum(np.diag(cov_iter), 1e-12))
            bounds = [
                (
                    float(theta_curr[m] - (3.0 * se_iter[m] + abs(theta_curr[m]) + 0.5)),
                    float(theta_curr[m] + (3.0 * se_iter[m] + abs(theta_curr[m]) + 0.5)),
                )
                for m in range(m_exp)
            ]
            opt = optimize.minimize(
                gmm_mv_obj,
                theta_curr,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": max(max_iter, 50)},
            )
            if opt.success and np.all(np.isfinite(opt.x)) and float(opt.fun) <= gmm_mv_obj(theta_curr) + 1e-12:
                theta_hat = np.asarray(opt.x, dtype=np.float64)
                converged = True

    om_final = omega_theta(theta_hat)
    g_final = pc_by - pc_Bx @ theta_hat
    j_df = max(k - m_exp, 0)
    j_stat = max(float(g_final @ sym_solve(om_final, g_final)), 0.0) if j_df > 0 else 0.0
    j_pval = float(stats.chi2.sf(j_stat, df=j_df)) if j_df > 0 else np.nan

    info_mat = pc_Bx.T @ sym_solve(om_final, pc_Bx)
    cov_base = sym_solve(info_mat, np.eye(m_exp, dtype=np.float64))
    scale = max(1.0, j_stat / j_df) if (robust and j_df > 0) else 1.0
    cov_hat = scale * cov_base
    ses = np.sqrt(np.maximum(np.diag(cov_hat), 1e-18))

    df_crit = j_df if (settings == "R_default" and robust and j_df > 0) else None
    pvals = np.empty(m_exp, dtype=np.float64)
    ci_low = np.empty(m_exp, dtype=np.float64)
    ci_high = np.empty(m_exp, dtype=np.float64)
    for m in range(m_exp):
        pvals[m], ci_low[m], ci_high[m] = wald_inference(
            float(theta_hat[m]), float(ses[m]), alpha=alpha, df=df_crit
        )

    cond_f = _compute_conditional_pc_fstats(pc_Bx, Sx_arr, R, A, Cx)
    names = tuple(exposure_names) if exposure_names is not None else tuple(
        f"exposure_{m + 1}" for m in range(m_exp)
    )

    return MVPCGMMResult(
        estimates=theta_hat,
        ses=ses,
        pvals=pvals,
        ci_low=ci_low,
        ci_high=ci_high,
        cov=cov_hat,
        cond_f_stats=cond_f,
        n_snps=p,
        n_exposures=m_exp,
        n_pcs=k,
        j_stat=j_stat,
        j_df=j_df,
        j_pval=j_pval,
        var_explained=var_explained,
        converged=converged,
        n_iter=n_iter,
        robust=robust,
        thres=thres,
        settings=settings,
        exposure_names=names,
    )
