"""
NEBULA-HL (h-likelihood Newton-Raphson mode solver, Gamma-saddlepoint &
Laplace marginal log-likelihoods, exact IFT gradients, and GLMM Schur-complement
standard errors).
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

from .ln import (
    _compute_fisher_preconditioner,
    _safe_eta_lin,
    _z_minus_log1p,
    poch_dg,
    poch_lg,
    run_lbfgsb,
)


def solve_omega_mode(
    mu: np.ndarray,
    M: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    subj_nz: np.ndarray,
    Y_subj: np.ndarray,
    alpha_s: float,
    alpha_c: float,
    *,
    gaussian_prior: bool = False,
    max_steps: int = 25,
    tol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the per-subject strictly concave 1D mode equation h_i'(omega_i) = 0
    simultaneously across all ``m`` subjects via damped Newton-Raphson.

    When ``gaussian_prior=False`` (NBGMM), ``alpha_s`` is the Gamma shape/rate.
    When ``gaussian_prior=True`` (NBLMM), ``alpha_s`` is ``tau_s = 1 / sigma2``.

    Returns
    -------
    omega, u, p_ij, q_ij, w_ij, W_subj, H_subj
    """
    if gaussian_prior:
        omega = np.log(np.maximum((Y_subj + 0.5) / (M + 0.5), 1e-4))
    else:
        omega = np.log(np.maximum((Y_subj + alpha_s) / (M + alpha_s), 1e-8))
    omega = np.clip(omega, -20.0, 20.0)

    has_nz = y_nz.size > 0
    for _ in range(max_steps):
        u = np.exp(omega)
        umu = u[subj] * mu
        denom = alpha_c + umu
        p_ij = umu / denom
        q_ij = alpha_c / denom
        pq = p_ij * q_ij

        sum_yp = alpha_c * np.bincount(subj, weights=p_ij, minlength=m)
        W_subj = alpha_c * np.bincount(subj, weights=pq, minlength=m)
        if has_nz:
            sum_yp += np.bincount(subj_nz, weights=y_nz * p_ij[cell_idx], minlength=m)
            W_subj += np.bincount(subj_nz, weights=y_nz * pq[cell_idx], minlength=m)

        if gaussian_prior:
            hp = Y_subj - alpha_s * omega - sum_yp
            H_subj = alpha_s + W_subj
        else:
            hp = (Y_subj + alpha_s) - alpha_s * u - sum_yp
            H_subj = alpha_s * u + W_subj

        step = np.clip(hp / np.maximum(H_subj, 1e-12), -2.0, 2.0)
        omega = np.clip(omega + step, -45.0, 45.0)
        if float(np.max(np.abs(step))) < tol:
            break

    u = np.exp(omega)
    umu = u[subj] * mu
    denom = alpha_c + umu
    p_ij = umu / denom
    q_ij = alpha_c / denom
    pq = p_ij * q_ij

    w_ij = alpha_c * pq
    if has_nz:
        w_ij[cell_idx] += y_nz * pq[cell_idx]
    W_subj = np.bincount(subj, weights=w_ij, minlength=m)
    H_subj = (alpha_s + W_subj) if gaussian_prior else (alpha_s * u + W_subj)

    return omega, u, p_ij, q_ij, w_ij, W_subj, H_subj


def _hl_nll_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    subj_nz: np.ndarray,
    Y_subj: np.ndarray,
    Y_nz: np.ndarray,
    const_y_fact: float,
    *,
    gaussian_prior: bool = False,
    laplace_order: int = 2,
) -> tuple[float, np.ndarray]:
    """
    Negative NEBULA-HL marginal log-likelihood and exact analytical gradient
    (Envelope Theorem + Implicit Function Theorem) with respect to
    ``theta = [beta (p), eta_s, eta_c]``.

    - ``gaussian_prior=False, laplace_order=2``: NBGMM Gamma-saddlepoint (exact-Gamma).
    - ``gaussian_prior=False, laplace_order=1``: NBGMM 1st-order Gaussian Laplace.
    - ``gaussian_prior=True``: NBLMM Gaussian random effect Laplace.
    """
    p = X.shape[1]
    beta = theta[:p]
    eta_s = float(theta[p])
    eta_c = float(theta[p + 1])
    alpha_s = float(np.exp(eta_s))  # tau_s = 1/sigma2 if gaussian_prior
    alpha_c = float(np.exp(eta_c))

    eta_lin = X @ beta + log_offset
    pen_weight = max(100.0, float(np.sum(y_nz)) + 1.0) if y_nz.size > 0 else 100.0
    eta_eff, d_eta, pen, d_pen = _safe_eta_lin(eta_lin, pen_weight, limit=30.0)
    mu = np.exp(eta_eff)
    M = np.bincount(subj, weights=mu, minlength=m)

    omega, u, p_ij, q_ij, w_ij, W_subj, H_subj = solve_omega_mode(
        mu,
        M,
        subj,
        m,
        y_nz,
        cell_idx,
        subj_nz,
        Y_subj,
        alpha_s,
        alpha_c,
        gaussian_prior=gaussian_prior,
    )

    has_nz = y_nz.size > 0
    u_cell = u[subj]
    umu = u_cell * mu
    denom = alpha_c + umu
    z = umu / alpha_c
    log1p_z = np.log1p(z)

    ll_const = (
        float(np.sum(poch_lg(y_nz, alpha_c)))
        - const_y_fact
        + (float(np.dot(y_nz, eta_eff[cell_idx])) if has_nz else 0.0)
    )
    ll_cell = -alpha_c * float(np.sum(log1p_z)) - (
        float(np.dot(y_nz, log1p_z[cell_idx])) if has_nz else 0.0
    )

    if gaussian_prior:
        # NBLMM: sum_i [ h_i(omega_hat) + 0.5 * log(2*pi / H_i) ]
        # h_i prior: 0.5*log(alpha_s/(2*pi)) - 0.5*alpha_s*omega^2 + Y_i*omega
        # Laplace +0.5*log(2*pi) cancels -0.5*log(2*pi) in Gaussian prior!
        ll_subj = float(
            np.sum(
                Y_subj * omega
                + 0.5 * eta_s
                - 0.5 * alpha_s * (omega * omega)
                - 0.5 * np.log(H_subj)
            )
        )
    elif laplace_order == 2:
        mu_inv_D = mu / denom
        B_subj = alpha_c * np.bincount(subj, weights=mu_inv_D, minlength=m)
        if has_nz:
            B_subj += np.bincount(subj_nz, weights=y_nz * mu_inv_D[cell_idx], minlength=m)
        ratio_H = np.clip(H_subj / np.maximum(Y_subj + alpha_s, 1e-300), 1e-300, 1.0)
        ll_subj = float(np.sum(poch_lg(Y_nz, alpha_s))) + float(
            np.sum(
                u * B_subj
                - (Y_subj + alpha_s) * np.log1p(B_subj / alpha_s)
                - 0.5 * np.log(ratio_H)
            )
        )
    else:
        # 1st-order Gaussian Laplace on omega_i ~ LogGamma(alpha_s, alpha_s)
        ll_subj = float(
            np.sum(
                (Y_subj + alpha_s) * omega
                - alpha_s * u
                + alpha_s * eta_s
                - gammaln(alpha_s)
                + 0.5 * np.log(2.0 * np.pi / H_subj)
            )
        )

    ll = ll_const + ll_cell + ll_subj

    # Exact IFT Gradients
    w_qp = w_ij * (q_ij - p_ij)
    tau_subj = np.bincount(subj, weights=w_qp, minlength=m)
    if not gaussian_prior:
        tau_subj += alpha_s * u
    kappa = tau_subj / H_subj
    kappa_cell = kappa[subj]
    H_cell = H_subj[subj]

    # Beta gradient (chain rule through _safe_eta_lin)
    g_eta_F = -alpha_c * p_ij
    if has_nz:
        g_eta_F[cell_idx] += y_nz * q_ij[cell_idx]
    g_eta_C = -0.5 / H_cell * (w_qp - kappa_cell * w_ij)
    grad_beta_nll = X.T @ (-(g_eta_F + g_eta_C) * d_eta + d_pen)

    # eta_s gradient
    if gaussian_prior:
        grad_etas = float(
            np.sum(0.5 * (1.0 - alpha_s * (omega * omega)) - 0.5 * alpha_s / H_subj * (1.0 - kappa * omega))
        )
    elif laplace_order == 2:
        g_etas_F = float(np.sum(poch_dg(Y_nz, alpha_s))) + float(
            np.sum(u * B_subj - alpha_s * np.log1p(B_subj / alpha_s))
        )
        g_etas_C = float(
            np.sum(
                0.5
                / H_subj
                * (
                    alpha_s * (W_subj - u * Y_subj) / np.maximum(Y_subj + alpha_s, 1e-300)
                    - kappa * (u * B_subj - Y_subj)
                )
            )
        )
        grad_etas = g_etas_F + g_etas_C
    else:
        g_etas_F = float(alpha_s * np.sum(eta_s + 1.0 - digamma(alpha_s) + omega - u))
        g_etas_C = float(alpha_s * np.sum(-0.5 / H_subj * (u + kappa * (1.0 - u))))
        grad_etas = g_etas_F + g_etas_C

    # eta_c gradient (cancellation-free)
    tz = _z_minus_log1p(z)
    g_etac_F = (
        float(np.sum(poch_dg(y_nz, alpha_c)))
        + (float(np.dot(y_nz, z[cell_idx] / (1.0 + z[cell_idx]))) if has_nz else 0.0)
        + alpha_c * float(np.sum(tz))
    )

    p2q = (p_ij * p_ij) * q_ij
    dH_detac = 2.0 * alpha_c * p2q
    dhp_detac = -alpha_c * (p_ij * p_ij)
    if has_nz:
        pq_nz = p_ij[cell_idx] * q_ij[cell_idx]
        dH_detac[cell_idx] += y_nz * pq_nz * (p_ij[cell_idx] - q_ij[cell_idx])
        dhp_detac[cell_idx] += y_nz * pq_nz
    g_etac_C = -0.5 * float(np.sum((dH_detac + kappa_cell * dhp_detac) / H_cell))
    grad_etac = g_etac_F + g_etac_C

    grad = np.empty(p + 2, dtype=np.float64)
    grad[:p] = grad_beta_nll
    grad[p] = -grad_etas
    grad[p + 1] = -grad_etac
    return -ll + pen, grad


def fit_hl_gene(
    X: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    subj_nz: np.ndarray,
    Y_subj: np.ndarray,
    Y_nz: np.ndarray,
    const_y_fact: float,
    beta0: np.ndarray,
    min_overdisp: float,
    max_overdisp: float,
    max_iter: int,
    tol: float,
    *,
    gaussian_prior: bool = False,
    laplace_order: int = 2,
    sigma2_init: float = 0.5,
    phi_init: float = 0.5,
) -> tuple[np.ndarray, float, float, float, int, int]:
    """
    Fit a single gene with NEBULA-HL (Gamma-saddlepoint / Laplace).

    Returns ``(beta_hat, sigma2_hat, phi_hat, loglik, convergence, n_iter)``.
    """
    p = X.shape[1]
    eta_lo = -float(np.log(max_overdisp))
    eta_hi = -float(np.log(min_overdisp))

    X_pre, T, gamma0 = _compute_fisher_preconditioner(X, beta0, log_offset)
    gamma_bounds: list[tuple[float | None, float | None]] = [(None, None)] * p

    eta_s0 = float(
        np.clip(-np.log(max(sigma2_init, min_overdisp)), eta_lo + 0.01, eta_hi - 0.01)
    )
    eta_c0 = float(
        np.clip(-np.log(max(phi_init, min_overdisp)), eta_lo + 0.01, eta_hi - 0.01)
    )
    s_s = 1.0 / max(1.0, float(np.sqrt(0.25 * max(1, m))))
    s_c = 1.0 / max(1.0, float(np.sqrt(0.25 * max(1, y_nz.size))))

    z0 = np.concatenate([gamma0, [eta_s0 / s_s, eta_c0 / s_c]])
    bounds = gamma_bounds + [(eta_lo / s_s, eta_hi / s_s), (eta_lo / s_c, eta_hi / s_c)]

    def _hl_scaled(z: np.ndarray) -> tuple[float, np.ndarray]:
        th = z.copy()
        th[p] *= s_s
        th[p + 1] *= s_c
        f, g = _hl_nll_and_grad(
            th,
            X_pre,
            log_offset,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            Y_nz,
            const_y_fact,
            gaussian_prior=gaussian_prior,
            laplace_order=laplace_order,
        )
        g[p] *= s_s
        g[p + 1] *= s_c
        return f, g

    z_hat, ll, conv, n_iter = run_lbfgsb(_hl_scaled, z0, bounds, max_iter, tol)
    beta_hat = T @ z_hat[:p]
    sigma2_hat = float(np.exp(-(z_hat[p] * s_s)))
    phi_hat = float(np.exp(-(z_hat[p + 1] * s_c)))
    return beta_hat, sigma2_hat, phi_hat, ll, conv, n_iter


def fit_hl_sigma2_1d(
    X: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    subj_nz: np.ndarray,
    Y_subj: np.ndarray,
    Y_nz: np.ndarray,
    const_y_fact: float,
    beta_fixed: np.ndarray,
    phi_fixed: float,
    min_overdisp: float,
    max_overdisp: float,
    max_iter: int,
    tol: float,
    *,
    gaussian_prior: bool = False,
    laplace_order: int = 1,
    sigma2_init: float = 0.5,
    kappa_obs: float = 100.0,
) -> tuple[float, float, int, int]:
    """
    Optimize only subject-level overdispersion ``sigma^2`` (``eta_s = -log(sigma^2)``)
    under the NEBULA-HL marginal log-likelihood while holding ``beta_fixed`` and
    ``phi_fixed`` fixed (`NBGMM (LN+HL)` intermediate ``kappa`` refit tier in ``nebula.R``).

    Returns ``(sigma2_hat, loglik, convergence, n_iter)``.
    """
    p = X.shape[1]
    eta_lo = -float(np.log(max_overdisp))
    eta_hi = -float(np.log(min_overdisp))
    eta_c_fixed = float(
        np.clip(-np.log(max(phi_fixed, min_overdisp)), eta_lo, eta_hi)
    )
    s_s = 1.0 / max(1.0, float(np.sqrt(0.25 * max(1, m))))

    def _hl_1d_scaled(z_1d: np.ndarray) -> tuple[float, np.ndarray]:
        eta_s = float(z_1d[0]) * s_s
        th = np.empty(p + 2, dtype=np.float64)
        th[:p] = beta_fixed
        th[p] = eta_s
        th[p + 1] = eta_c_fixed
        f, g = _hl_nll_and_grad(
            th,
            X,
            log_offset,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            Y_nz,
            const_y_fact,
            gaussian_prior=gaussian_prior,
            laplace_order=laplace_order,
        )
        return f, np.array([g[p] * s_s], dtype=np.float64)

    cand_s2 = [
        float(np.clip(sigma2_init, min_overdisp * 2.0, max_overdisp * 0.5)),
        float(np.clip(2.0 / max(kappa_obs, 1.0), min_overdisp * 2.0, max_overdisp * 0.5)),
    ]
    best_eta_s0 = float(np.clip(-np.log(cand_s2[0]), eta_lo + 0.01, eta_hi - 0.01))
    best_f0 = _hl_1d_scaled(np.array([best_eta_s0 / s_s], dtype=np.float64))[0]
    for c_s2 in cand_s2[1:]:
        eta_try = float(np.clip(-np.log(c_s2), eta_lo + 0.01, eta_hi - 0.01))
        f_try = _hl_1d_scaled(np.array([eta_try / s_s], dtype=np.float64))[0]
        if np.isfinite(f_try) and f_try < best_f0:
            best_f0 = f_try
            best_eta_s0 = eta_try

    z0 = np.array([best_eta_s0 / s_s], dtype=np.float64)
    bounds: list[tuple[float | None, float | None]] = [(eta_lo / s_s, eta_hi / s_s)]
    z_hat, ll, conv, n_iter = run_lbfgsb(_hl_1d_scaled, z0, bounds, max_iter, tol)
    sigma2_hat = float(np.exp(-(float(z_hat[0]) * s_s)))
    return sigma2_hat, ll, conv, n_iter


def compute_schur_covariance(
    beta: np.ndarray,
    sigma2: float,
    phi: float,
    X: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    subj_nz: np.ndarray,
    Y_subj: np.ndarray,
    *,
    model: str = "NBGMM",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GLMM Schur-complement Fisher information, covariance matrix,
    standard errors, and conditional subject random-effect modes ``omega_hat``.

    I_bb = H_bb - sum_{i=0}^{m-1} (1 / H_ww,i) * H_bw,i * H_bw,i^T
    """
    p = X.shape[1]
    eta_lin = X @ beta + log_offset
    mu = np.exp(np.clip(eta_lin, -30.0, 30.0))
    alpha_s = 1.0 / max(sigma2, 1e-12)

    if model == "PMM":
        M = np.bincount(subj, weights=mu, minlength=m)
        u_hat = (Y_subj + alpha_s) / (alpha_s + M)
        omega = np.log(np.maximum(u_hat, 1e-12))
        w_ij = u_hat[subj] * mu
        W_subj = np.bincount(subj, weights=w_ij, minlength=m)
        H_ww = alpha_s * u_hat + W_subj
    else:
        alpha_c = 1.0 / max(phi, 1e-12)
        M = np.bincount(subj, weights=mu, minlength=m)
        omega, _, _, _, w_ij, _, H_ww = solve_omega_mode(
            mu,
            M,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            alpha_s,
            alpha_c,
            gaussian_prior=(model == "NBLMM"),
            max_steps=20,
            tol=1e-11,
        )

    wX = w_ij[:, None] * X
    H_bb = X.T @ wX
    H_bw = np.column_stack(
        [np.bincount(subj, weights=wX[:, k], minlength=m) for k in range(p)]
    )
    inv_Hww = 1.0 / np.maximum(H_ww, 1e-12)
    I_bb = H_bb - (H_bw.T * inv_Hww[None, :]) @ H_bw
    I_bb = 0.5 * (I_bb + I_bb.T)

    # Symmetric Jacobi equilibration so covariance / SE are invariant to predictor scaling
    diag_I = np.diag(I_bb)
    d = np.sqrt(np.maximum(np.abs(diag_I), 1e-280))
    d_inv = 1.0 / d
    I_eq = (d_inv[:, None] * I_bb) * d_inv[None, :]
    I_eq = 0.5 * (I_eq + I_eq.T)

    try:
        evals, V = np.linalg.eigh(I_eq)
        max_eig_eq = float(np.max(evals)) if evals.size > 0 else 0.0
        evals_safe = np.maximum(evals, 1e-12 * max(max_eig_eq, 1.0))
        cov_eq = (V / evals_safe[None, :]) @ V.T
    except np.linalg.LinAlgError:
        I_reg = I_eq + 1e-12 * np.eye(p, dtype=np.float64)
        cov_eq = np.linalg.pinv(I_reg, rcond=1e-12)

    cov = (d_inv[:, None] * cov_eq) * d_inv[None, :]
    cov = 0.5 * (cov + cov.T)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return cov, se, omega
