"""
NEBULA-LN (analytical marginal moments) and PMM (Poisson-Gamma marginal)
log-likelihoods, analytical gradients, and L-BFGS-B estimators.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, logsumexp


def poch_lg(y_nz: np.ndarray, alpha: float) -> np.ndarray:
    """
    Cancellation-free evaluation of:
        gammaln(y + alpha) - gammaln(alpha) - y * log(alpha)
    for positive counts ``y_nz > 0`` and ``alpha > 0``.
    """
    if y_nz.size == 0:
        return np.empty(0, dtype=np.float64)
    if alpha > 1e4:
        phi = 1.0 / alpha
        small = (y_nz * phi) < 1e-3
        if np.all(small):
            ym1 = y_nz - 1.0
            t1 = 0.5 * y_nz * ym1 * phi
            t2 = -(y_nz * ym1 * (2.0 * y_nz - 1.0) / 12.0) * (phi * phi)
            t3 = (y_nz * y_nz * ym1 * ym1 / 12.0) * (phi * phi * phi)
            return t1 + t2 + t3
        out = np.empty_like(y_nz, dtype=np.float64)
        if np.any(small):
            ys = y_nz[small]
            ym1 = ys - 1.0
            out[small] = (
                0.5 * ys * ym1 * phi
                - (ys * ym1 * (2.0 * ys - 1.0) / 12.0) * (phi * phi)
                + (ys * ys * ym1 * ym1 / 12.0) * (phi * phi * phi)
            )
        big = ~small
        yb = y_nz[big]
        out[big] = gammaln(yb + alpha) - gammaln(alpha) - yb * np.log(alpha)
        return out
    return gammaln(y_nz + alpha) - gammaln(alpha) - y_nz * np.log(alpha)


def poch_dg(y_nz: np.ndarray, alpha: float) -> np.ndarray:
    """
    Cancellation-free evaluation of:
        alpha * (digamma(y + alpha) - digamma(alpha)) - y
    for positive counts ``y_nz > 0`` and ``alpha > 0``.
    """
    if y_nz.size == 0:
        return np.empty(0, dtype=np.float64)
    if alpha > 1e4:
        phi = 1.0 / alpha
        small = (y_nz * phi) < 1e-3
        if np.all(small):
            ym1 = y_nz - 1.0
            t1 = -0.5 * y_nz * ym1 * phi
            t2 = (y_nz * ym1 * (2.0 * y_nz - 1.0) / 6.0) * (phi * phi)
            t3 = -(y_nz * y_nz * ym1 * ym1 / 4.0) * (phi * phi * phi)
            return t1 + t2 + t3
        out = np.empty_like(y_nz, dtype=np.float64)
        if np.any(small):
            ys = y_nz[small]
            ym1 = ys - 1.0
            out[small] = (
                -0.5 * ys * ym1 * phi
                + (ys * ym1 * (2.0 * ys - 1.0) / 6.0) * (phi * phi)
                - (ys * ys * ym1 * ym1 / 4.0) * (phi * phi * phi)
            )
        big = ~small
        yb = y_nz[big]
        out[big] = alpha * (digamma(yb + alpha) - digamma(alpha)) - yb
        return out
    return alpha * (digamma(y_nz + alpha) - digamma(alpha)) - y_nz


def _z_minus_log1p(z: np.ndarray) -> np.ndarray:
    """Cancellation-free evaluation of z / (1 + z) - log1p(z) for z >= 0."""
    small = z < 1e-4
    if np.all(small):
        z2 = z * z
        return -0.5 * z2 + (2.0 / 3.0) * (z2 * z) - 0.75 * (z2 * z2)
    out = np.empty_like(z, dtype=np.float64)
    if np.any(small):
        zs = z[small]
        zs2 = zs * zs
        out[small] = -0.5 * zs2 + (2.0 / 3.0) * (zs2 * zs) - 0.75 * (zs2 * zs2)
    big = ~small
    zb = z[big]
    out[big] = zb / (1.0 + zb) - np.log1p(zb)
    return out


def _safe_eta_lin(
    eta_lin: np.ndarray,
    pen_weight: float,
    limit: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    C^1-smooth, overflow-safe transformation of the linear predictor ``eta_lin``
    paired with a smooth quadratic barrier outside ``[-limit, limit]``.

    Inside ``[-limit, limit]``, ``eta_eff == eta_lin``, ``d_eta == 1``, and
    ``pen == 0`` (exact unmodified likelihood and gradient). Outside
    ``[-limit, limit]``, ``eta_eff`` saturates smoothly into ``(-limit-1, limit+1)``
    via ``tanh`` while ``pen = 0.5 * pen_weight * (|eta_lin| - limit)^2``
    grows quadratically with continuous derivative at ``+-limit``.
    """
    upper = eta_lin > limit
    lower = eta_lin < -limit
    if not (np.any(upper) or np.any(lower)):
        return eta_lin, np.ones_like(eta_lin), 0.0, np.zeros_like(eta_lin)

    eta_eff = eta_lin.copy()
    d_eta = np.ones_like(eta_lin)
    d_pen = np.zeros_like(eta_lin)
    pen = 0.0

    if np.any(upper):
        u = eta_lin[upper] - limit
        e2 = np.exp(-2.0 * u)
        th = (1.0 - e2) / (1.0 + e2)
        eta_eff[upper] = limit + th
        d_eta[upper] = 4.0 * e2 / ((1.0 + e2) ** 2)
        pen += 0.5 * pen_weight * float(np.dot(u, u))
        d_pen[upper] = pen_weight * u

    if np.any(lower):
        u = eta_lin[lower] + limit
        e2 = np.exp(2.0 * u)
        th = (e2 - 1.0) / (e2 + 1.0)
        eta_eff[lower] = -limit + th
        d_eta[lower] = 4.0 * e2 / ((1.0 + e2) ** 2)
        pen += 0.5 * pen_weight * float(np.dot(u, u))
        d_pen[lower] = pen_weight * u

    return eta_eff, d_eta, pen, d_pen


def _ln_nll_and_grad(
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
) -> tuple[float, np.ndarray]:
    """
    Negative NEBULA-LN marginal log-likelihood and exact analytical gradient
    with respect to ``theta = [beta (p), eta_s, eta_c]``, where
    ``alpha_s = exp(eta_s) = 1 / sigma2`` and ``alpha_c = exp(eta_c) = 1 / phi``.
    """
    p = X.shape[1]
    beta = theta[:p]
    eta_s = float(theta[p])
    eta_c = float(theta[p + 1])
    alpha_s = float(np.exp(eta_s))
    alpha_c = float(np.exp(eta_c))

    eta_lin = X @ beta + log_offset
    pen_weight = max(100.0, float(np.sum(y_nz)) + 1.0) if y_nz.size > 0 else 100.0
    eta_eff, d_eta, pen, d_pen = _safe_eta_lin(eta_lin, pen_weight, limit=30.0)
    mu = np.exp(eta_eff)

    M = np.bincount(subj, weights=mu, minlength=m)
    D_s = alpha_s + M
    u_hat = (Y_subj + alpha_s) / D_s
    u_cell = u_hat[subj]
    umu = u_cell * mu
    D_c = alpha_c + umu
    z = umu / alpha_c
    log1p_z = np.log1p(z)

    # Cell log-likelihood (uses eta_eff consistently with mu = exp(eta_eff))
    ll_cell = (
        float(np.sum(poch_lg(y_nz, alpha_c)))
        - const_y_fact
        + (float(np.dot(y_nz, eta_eff[cell_idx])) if y_nz.size > 0 else 0.0)
        - alpha_c * float(np.sum(log1p_z))
        - (float(np.dot(y_nz, log1p_z[cell_idx])) if y_nz.size > 0 else 0.0)
    )

    # Subject log-likelihood
    ll_subj = float(np.sum(poch_lg(Y_nz, alpha_s))) + float(
        np.sum(u_hat * M - (Y_subj + alpha_s) * np.log1p(M / alpha_s))
    )
    ll = ll_cell + ll_subj

    # Exact gradients (chain rule through _safe_eta_lin)
    inv_Dc = 1.0 / D_c
    mu_inv_Dc = mu * inv_Dc
    B_subj = alpha_c * np.bincount(subj, weights=mu_inv_Dc, minlength=m)
    if y_nz.size > 0:
        B_subj += np.bincount(subj_nz, weights=y_nz * mu_inv_Dc[cell_idx], minlength=m)

    delta_subj = M - B_subj
    adj_subj = delta_subj * u_hat / D_s

    g_eta = -(alpha_c * umu * inv_Dc) - adj_subj[subj] * mu
    if y_nz.size > 0:
        g_eta[cell_idx] += y_nz * (alpha_c * inv_Dc[cell_idx])
    grad_beta_nll = X.T @ (-g_eta * d_eta + d_pen)

    grad_etas = float(np.sum(poch_dg(Y_nz, alpha_s))) + float(
        np.sum(
            u_hat * M
            - alpha_s * np.log1p(M / alpha_s)
            + delta_subj * (alpha_s * (M - Y_subj) / (D_s * D_s))
        )
    )

    tz = _z_minus_log1p(z)
    grad_etac = (
        float(np.sum(poch_dg(y_nz, alpha_c)))
        + (float(np.dot(y_nz, z[cell_idx] / (1.0 + z[cell_idx]))) if y_nz.size > 0 else 0.0)
        + alpha_c * float(np.sum(tz))
    )

    grad = np.empty(p + 2, dtype=np.float64)
    grad[:p] = grad_beta_nll
    grad[p] = -grad_etas
    grad[p + 1] = -grad_etac
    return -ll + pen, grad


def _pmm_nll_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    Y_subj: np.ndarray,
    Y_nz: np.ndarray,
    const_y_fact: float,
) -> tuple[float, np.ndarray]:
    """
    Negative PMM (Poisson-Gamma marginal, phi = 0) exact log-likelihood and
    analytical gradient with respect to ``theta = [beta (p), eta_s]``.
    """
    p = X.shape[1]
    beta = theta[:p]
    eta_s = float(theta[p])
    alpha_s = float(np.exp(eta_s))

    eta_lin = X @ beta + log_offset
    pen_weight = max(100.0, float(np.sum(y_nz)) + 1.0) if y_nz.size > 0 else 100.0
    eta_eff, d_eta, pen, d_pen = _safe_eta_lin(eta_lin, pen_weight, limit=30.0)
    mu = np.exp(eta_eff)

    M = np.bincount(subj, weights=mu, minlength=m)
    D_s = alpha_s + M
    u_hat = (Y_subj + alpha_s) / D_s

    ll = (
        ((float(np.dot(y_nz, eta_eff[cell_idx])) if y_nz.size > 0 else 0.0) - const_y_fact)
        + float(np.sum(poch_lg(Y_nz, alpha_s)))
        - float(np.sum((Y_subj + alpha_s) * np.log1p(M / alpha_s)))
    )

    g_eta = -(u_hat[subj] * mu)
    if y_nz.size > 0:
        g_eta[cell_idx] += y_nz
    grad_beta_nll = X.T @ (-g_eta * d_eta + d_pen)
    grad_etas = float(np.sum(poch_dg(Y_nz, alpha_s))) + float(
        np.sum(u_hat * M - alpha_s * np.log1p(M / alpha_s))
    )

    grad = np.empty(p + 1, dtype=np.float64)
    grad[:p] = grad_beta_nll
    grad[p] = -grad_etas
    return -ll + pen, grad


def _equilibrated_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Scale-invariant damped solve for SPD matrix ``A``."""
    p = A.shape[0]
    d = np.sqrt(np.maximum(np.diag(A), 1e-30))
    A_eq = A / np.outer(d, d)
    A_eq[np.diag_indices(p)] += ridge
    step_eq = np.linalg.solve(A_eq, b / d)
    return step_eq / d


def initial_beta_guess(
    X: np.ndarray,
    log_offset: np.ndarray,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    N: int,
    intercept_col: int | None,
) -> np.ndarray:
    """
    Compute a robust initial beta vector via intercept/ridge baseline followed by
    damped NB2 Fisher-scoring (IRLS) steps so L-BFGS-B starts near the MLE even
    with unstandardized predictors or high counts.
    """
    p = X.shape[1]
    total_y = float(np.sum(y_nz))
    log_total_t = float(logsumexp(log_offset))
    log_rate = float(np.log(max(total_y, 0.5)) - log_total_t)

    if intercept_col is not None:
        beta = np.zeros(p, dtype=np.float64)
        c_val = float(X[0, intercept_col])
        beta[intercept_col] = log_rate / c_val
    else:
        y_work = np.full(N, 0.25, dtype=np.float64)
        if y_nz.size > 0:
            y_work[cell_idx] += y_nz
        target = np.log(y_work) - log_offset
        beta = _equilibrated_solve(X.T @ X, X.T @ target, ridge=1e-4)

    phi0 = 0.5
    alpha0 = 1.0 / phi0
    has_nz = y_nz.size > 0
    pen_w = max(100.0, total_y + 1.0)

    def _nb2_ll(b: np.ndarray) -> float:
        eta = X @ b + log_offset
        ee, _, pen, _ = _safe_eta_lin(eta, pen_w, limit=25.0)
        mu_b = np.exp(ee)
        val = -alpha0 * float(np.sum(np.log1p(phi0 * mu_b)))
        if has_nz:
            val += float(np.dot(y_nz, ee[cell_idx] - np.log1p(phi0 * mu_b[cell_idx])))
        return val - pen

    cur_ll = _nb2_ll(beta)
    for _ in range(6):
        eta = np.clip(X @ beta + log_offset, -25.0, 25.0)
        mu = np.exp(eta)
        denom = 1.0 + phi0 * mu
        w = mu / denom
        g = -(X.T @ w)
        if has_nz:
            g += X[cell_idx].T @ (y_nz / denom[cell_idx])
        Xw = np.sqrt(w)[:, None] * X
        XtWX = Xw.T @ Xw
        try:
            step = _equilibrated_solve(XtWX, g, ridge=1e-6)
        except np.linalg.LinAlgError:
            break
        max_deta = float(np.max(np.abs(X @ step)))
        if max_deta > 3.0:
            step = step * (3.0 / max_deta)
        if max_deta < 1e-4:
            break
        accepted = False
        for _ls in range(5):
            b_try = beta + step
            ll_try = _nb2_ll(b_try)
            if ll_try >= cur_ll - 1e-9:
                beta = b_try
                cur_ll = ll_try
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return beta


def _compute_fisher_preconditioner(
    X: np.ndarray,
    beta0: np.ndarray,
    log_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Fisher-whitening QR/SVD preconditioning ``beta = T @ gamma`` so that
    ``(X @ T)^T W0 (X @ T) ≈ I_p`` at ``beta0``.

    Returns ``(X_pre, T, gamma0)`` where ``X_pre = X @ T`` and ``gamma0 = T^{-1} @ beta0``.
    """
    eta0 = np.clip(X @ beta0 + log_offset, -25.0, 25.0)
    mu0 = np.exp(eta0)
    w0 = np.maximum(mu0 / (1.0 + 0.5 * mu0), 1e-10)
    Xw = np.sqrt(w0)[:, None] * X

    col_norms = np.linalg.norm(Xw, axis=0)
    col_scale = np.where(col_norms > 1e-12, 1.0 / col_norms, 1.0)
    inv_col_scale = np.where(col_norms > 1e-12, col_norms, 1.0)
    Xw_eq = Xw * col_scale[None, :]

    _, R = np.linalg.qr(Xw_eq, mode="reduced")
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    s_floor = max(1e-6 * float(S[0]) if S.size > 0 else 1e-6, 1e-12)
    S_safe = np.maximum(S, s_floor)

    T = (col_scale[:, None] * Vt.T) / S_safe[None, :]
    T_inv = (S_safe[:, None] * Vt) * inv_col_scale[None, :]
    X_pre = X @ T
    gamma0 = T_inv @ beta0
    return X_pre, T, gamma0


def _mom_overdisp_guess(
    X: np.ndarray,
    beta0: np.ndarray,
    log_offset: np.ndarray,
    subj: np.ndarray,
    m: int,
    y_nz: np.ndarray,
    cell_idx: np.ndarray,
    Y_subj: np.ndarray,
) -> tuple[float, float]:
    """Fast O(N) method-of-moments initial guesses for ``(sigma2, phi)``."""
    eta0 = np.clip(X @ beta0 + log_offset, -25.0, 25.0)
    mu0 = np.exp(eta0)
    M0 = np.bincount(subj, weights=mu0, minlength=m)
    denom_s = float(np.sum(M0 * M0))
    if denom_s > 1e-8:
        s2_mom = float(np.sum((Y_subj - M0) ** 2 - M0)) / denom_s
    else:
        s2_mom = 0.5
    s2_0 = float(np.clip(s2_mom, 0.05, 5.0))

    u0 = (Y_subj + 1.0 / s2_0) / (M0 + 1.0 / s2_0)
    lam = u0[subj] * mu0
    denom_c = float(np.sum(lam * lam))
    if denom_c > 1e-8:
        num_c = float(np.sum(lam * lam - lam))
        if y_nz.size > 0:
            lam_nz = lam[cell_idx]
            num_c += float(np.sum(y_nz * y_nz - 2.0 * y_nz * lam_nz))
        phi_mom = num_c / denom_c
    else:
        phi_mom = 0.5
    phi_0 = float(np.clip(phi_mom, 0.05, 5.0))
    return s2_0, phi_0


def map_lbfgsb_status(res, max_iter: int) -> tuple[int, int]:
    """Map scipy L-BFGS-B OptimizeResult to NEBULA convergence code and n_iter."""
    n_iter = int(getattr(res, "nit", 0))
    if not np.isfinite(res.fun) or not np.all(np.isfinite(res.x)):
        return -40, n_iter
    if res.status == 0:
        return 1, n_iter
    if res.status == 1 or n_iter >= max_iter:
        return -20, n_iter
    # Status 2 (abnormal line search): if gradient is small & finite, treat as gtol
    jac = getattr(res, "jac", None)
    if jac is not None and np.all(np.isfinite(jac)) and np.max(np.abs(jac)) < 1e-2:
        return -10, n_iter
    return -40, n_iter


def run_lbfgsb(
    obj_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    theta0: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, int, int]:
    """Run L-BFGS-B with analytical gradients and safe retry on line-search stall."""
    gtol_eff = min(tol, 1e-7) if tol <= 1e-5 else tol
    res = minimize(
        obj_fn,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": gtol_eff, "maxcor": 15},
    )
    conv, n_iter = map_lbfgsb_status(res, max_iter)
    if conv == -40 and np.all(np.isfinite(res.x)):
        # Retry once with relaxed ftol from last valid iterate
        res2 = minimize(
            obj_fn,
            res.x,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": max(gtol_eff, 1e-6)},
        )
        conv2, n_iter2 = map_lbfgsb_status(res2, max_iter)
        if np.isfinite(res2.fun) and (not np.isfinite(res.fun) or res2.fun <= res.fun):
            return np.asarray(res2.x, dtype=np.float64), float(-res2.fun), conv2, n_iter + n_iter2
    return np.asarray(res.x, dtype=np.float64), float(-res.fun), conv, n_iter


def fit_ln_gene(
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
    pmm_only: bool = False,
) -> tuple[np.ndarray, float, float, float, int, int]:
    """
    Fit a single gene with NEBULA-LN (or exact PMM if ``pmm_only=True``).

    Returns ``(beta_hat, sigma2_hat, phi_hat, loglik, convergence, n_iter)``.
    """
    p = X.shape[1]
    eta_lo = -float(np.log(max_overdisp))
    eta_hi = -float(np.log(min_overdisp))

    X_pre, T, gamma0 = _compute_fisher_preconditioner(X, beta0, log_offset)
    gamma_bounds: list[tuple[float | None, float | None]] = [(None, None)] * p

    s2_0, phi_0 = _mom_overdisp_guess(
        X, beta0, log_offset, subj, m, y_nz, cell_idx, Y_subj
    )
    eta_s0 = float(np.clip(-np.log(s2_0), eta_lo + 0.1, eta_hi - 0.1))
    eta_c0 = float(np.clip(-np.log(phi_0), eta_lo + 0.1, eta_hi - 0.1))
    s_s = 1.0 / max(1.0, float(np.sqrt(0.25 * max(1, m))))
    s_c = 1.0 / max(1.0, float(np.sqrt(0.25 * max(1, y_nz.size))))

    if pmm_only:
        z0 = np.concatenate([gamma0, [eta_s0 / s_s]])
        bounds = gamma_bounds + [(eta_lo / s_s, eta_hi / s_s)]

        def _pmm_scaled(z: np.ndarray) -> tuple[float, np.ndarray]:
            th = z.copy()
            th[p] *= s_s
            f, g = _pmm_nll_and_grad(
                th, X_pre, log_offset, subj, m, y_nz, cell_idx, Y_subj, Y_nz, const_y_fact
            )
            g[p] *= s_s
            return f, g

        z_hat, ll, conv, n_iter = run_lbfgsb(_pmm_scaled, z0, bounds, max_iter, tol)
        beta_hat = T @ z_hat[:p]
        sigma2_hat = float(np.exp(-(z_hat[p] * s_s)))
        return beta_hat, sigma2_hat, 0.0, ll, conv, n_iter

    z0 = np.concatenate([gamma0, [eta_s0 / s_s, eta_c0 / s_c]])
    bounds = gamma_bounds + [(eta_lo / s_s, eta_hi / s_s), (eta_lo / s_c, eta_hi / s_c)]

    def _ln_scaled(z: np.ndarray) -> tuple[float, np.ndarray]:
        th = z.copy()
        th[p] *= s_s
        th[p + 1] *= s_c
        f, g = _ln_nll_and_grad(
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
        )
        g[p] *= s_s
        g[p + 1] *= s_c
        return f, g

    z_hat, ll, conv, n_iter = run_lbfgsb(_ln_scaled, z0, bounds, max_iter, tol)
    beta_hat = T @ z_hat[:p]
    sigma2_hat = float(np.exp(-(z_hat[p] * s_s)))
    phi_hat = float(np.exp(-(z_hat[p + 1] * s_c)))
    return beta_hat, sigma2_hat, phi_hat, ll, conv, n_iter
