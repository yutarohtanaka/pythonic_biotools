"""
Multi-SNP LD Principal-Component (PC) generalized proportionality omnibus test
and exact Fieller / profile-GLS confidence interval for slope ``eta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from .pairs import (
    SettingsMode,
    eval_angular_q_and_grad,
    minimize_angular_gls,
    simultaneous_diagonalize,
    validate_settings,
)


@dataclass
class PCPropTestResult:
    """Result of multi-SNP LD-PC generalized proportionality omnibus test."""

    stat: float
    p_value: float
    df: int
    n_pc: int
    var_explained_achieved: float
    eta_hat: float
    theta_hat: float
    eta_ci: tuple[float, float]


def _compute_profile_eta_ci(
    y: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
    q_min: float,
    theta_hat: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute exact (1 - alpha) Fieller / profile-GLS confidence interval for ``eta = tan(theta)``."""
    crit_delta = float(chi2.ppf(1.0 - float(alpha), df=1))
    q_target = q_min + crit_delta

    q_inf, _ = eval_angular_q_and_grad(0.5 * np.pi, y, x, lam)
    if float(q_inf) <= q_target + 1e-10:
        # Trait 1 denominator includes zero at level alpha -> Fieller unbounded CI
        return (-float(np.inf), float(np.inf))

    # Shift theta_hat into (-pi/2, pi/2) where tan(theta) is continuous and strictly increasing
    theta0 = theta_hat - np.pi if theta_hat > 0.5 * np.pi else theta_hat
    eps = 1e-7
    lo_bound = -0.5 * np.pi + eps
    hi_bound = 0.5 * np.pi - eps
    if not (lo_bound < theta0 < hi_bound):
        return (-float(np.inf), float(np.inf))

    def _diff_fn(th: float) -> float:
        qv, _ = eval_angular_q_and_grad(th, y, x, lam)
        return float(qv) - q_target

    # Left search on [lo_bound, theta0] (outermost crossing if non-monotone)
    left_grid = np.linspace(lo_bound, theta0, 128)
    left_vals, _ = eval_angular_q_and_grad(left_grid, y, x, lam)
    left_diffs = np.asarray(left_vals, dtype=float) - q_target
    neg_left = np.flatnonzero(left_diffs <= 0.0)
    if neg_left.size == 0 or neg_left[0] == 0:
        return (-float(np.inf), float(np.inf))
    k_lo = int(neg_left[0])
    th_lo = float(brentq(_diff_fn, float(left_grid[k_lo - 1]), float(left_grid[k_lo]), xtol=1e-11))

    # Right search on [theta0, hi_bound] (outermost crossing)
    right_grid = np.linspace(theta0, hi_bound, 128)
    right_vals, _ = eval_angular_q_and_grad(right_grid, y, x, lam)
    right_diffs = np.asarray(right_vals, dtype=float) - q_target
    neg_right = np.flatnonzero(right_diffs <= 0.0)
    if neg_right.size == 0 or neg_right[-1] == len(right_grid) - 1:
        return (-float(np.inf), float(np.inf))
    k_hi = int(neg_right[-1])
    th_hi = float(
        brentq(_diff_fn, float(right_grid[k_hi]), float(right_grid[k_hi + 1]), xtol=1e-11)
    )

    return (float(np.tan(th_lo)), float(np.tan(th_hi)))


def proptest_pc(
    b1: np.ndarray,
    se1: np.ndarray,
    b2: np.ndarray,
    se2: np.ndarray,
    LD: np.ndarray,
    var_explained: float = 0.99,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> PCPropTestResult:
    """Multi-SNP LD-PC generalized proportionality ``chi^2_{K-1}`` omnibus test.

    Projects marginal effect estimates ``(b1, b2)`` onto the top ``K`` eigenvectors
    of ``LD`` explaining ``>= var_explained`` cumulative LD variance, computes exact
    ``K x K`` projected sampling covariances ``Omega_t = V_K^T D_t R_psd D_t V_K``
    (preserving proportionality under arbitrary SNP-specific sample sizes / standard
    errors), and minimizes the angular profile-GLS statistic ``Q_K(theta)`` over
    ``theta in [0, pi)`` with ``K - 1`` degrees of freedom.
    """
    validate_settings(settings)
    b1_a = np.asarray(b1, dtype=float)
    s1_a = np.asarray(se1, dtype=float)
    b2_a = np.asarray(b2, dtype=float)
    s2_a = np.asarray(se2, dtype=float)
    r_raw = np.asarray(LD, dtype=float)

    m = len(b1_a)
    if m < 2:
        raise ValueError(f"proptest_pc requires at least 2 SNPs, got {m}.")
    if s1_a.shape != (m,) or b2_a.shape != (m,) or s2_a.shape != (m,):
        raise ValueError("b1, se1, b2, se2 must all have shape (M,).")
    if r_raw.shape != (m, m):
        raise ValueError(f"LD must have shape ({m}, {m}), got {r_raw.shape}.")
    if np.any(s1_a <= 0.0) or np.any(s2_a <= 0.0):
        raise ValueError("Standard errors se1 and se2 must be strictly positive.")
    if not (0.0 < float(var_explained) <= 1.0):
        raise ValueError(f"var_explained must be in (0, 1], got {var_explained}.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    r_sym = 0.5 * (r_raw + r_raw.T)
    eigvals, eigvecs = np.linalg.eigh(r_sym)
    # Sort eigenvalues descending
    desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[desc]
    eigvecs = eigvecs[:, desc]

    eigvals_pos = np.maximum(eigvals, 0.0)
    total_var = float(np.sum(eigvals_pos))
    if total_var <= 1e-12:
        raise ValueError("LD matrix has non-positive total variance.")

    pos_mask = eigvals_pos > 1e-6 * eigvals_pos[0]
    max_rank = int(np.sum(pos_mask))
    if max_rank < 2:
        # Regularize slightly if completely rank-1 collinear
        r_sym = 0.99 * r_sym + 0.01 * np.eye(m, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(r_sym)
        desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[desc]
        eigvecs = eigvecs[:, desc]
        eigvals_pos = np.maximum(eigvals, 0.0)
        total_var = float(np.sum(eigvals_pos))
        max_rank = int(np.sum(eigvals_pos > 1e-6 * eigvals_pos[0]))

    cum_frac = np.cumsum(eigvals_pos) / total_var
    k_req = int(np.searchsorted(cum_frac, float(var_explained) - 1e-12, side="left")) + 1
    n_pc = max(2, min(k_req, max_rank))
    achieved = float(cum_frac[n_pc - 1])

    v_k = eigvecs[:, :n_pc]  # (M, K)
    r_psd = (eigvecs * eigvals_pos[None, :]) @ eigvecs.T

    # Project b1, b2 onto top K LD eigenvectors
    proj_b1 = v_k.T @ b1_a
    proj_b2 = v_k.T @ b2_a

    # Exact projected covariance matrices Omega_t = V_K^T D_t R_psd D_t V_K
    dv1 = s1_a[:, None] * v_k
    dv2 = s2_a[:, None] * v_k
    omega1 = dv1.T @ r_psd @ dv1
    omega2 = dv2.T @ r_psd @ dv2

    y, x, lam = simultaneous_diagonalize(sigma2=omega2, sigma1=omega1, b2=proj_b2, b1=proj_b1)
    q_min, theta_hat, eta_hat = minimize_angular_gls(
        y, x, lam, include_2x2_roots=(n_pc == 2)
    )
    df = n_pc - 1
    p_val = float(chi2.sf(q_min, df=df))
    eta_ci = _compute_profile_eta_ci(y, x, lam, q_min=q_min, theta_hat=theta_hat, alpha=alpha)

    return PCPropTestResult(
        stat=q_min,
        p_value=p_val,
        df=df,
        n_pc=n_pc,
        var_explained_achieved=achieved,
        eta_hat=eta_hat,
        theta_hat=theta_hat,
        eta_ci=eta_ci,
    )
