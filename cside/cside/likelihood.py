"""Overdispersed Poisson-lognormal mixture likelihood and optimization for CSIDE."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp, roots_hermite

# Precompute 21-point Gauss-Hermite nodes & log normalized weights once
_GH_ORDER = 21
_GH_Z, _GH_V = roots_hermite(_GH_ORDER)
_GH_LOG_W = np.log(_GH_V) - 0.5 * np.log(np.pi)


def poisson_lognormal_moments(
    y: np.ndarray,
    lam: np.ndarray,
    log_fact_y: np.ndarray,
    sigma: float,
    only_ll: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate marginal log-likelihood and derivatives w.r.t. theta = log(lam).

    Uses mode-centered (Laplace-adaptive) Gauss-Hermite quadrature so that both
    low-count/small-sigma and high-count/large-sigma regimes achieve near
    machine-precision accuracy without lookup tables.

    Parameters
    ----------
    y
        Observed integer counts of shape ``(n_spots,)``.
    lam
        Expected deterministic Poisson rate ``N_i * sum_k w_ik exp(eta_ik)`` of
        shape ``(n_spots,)``.
    log_fact_y
        Precomputed ``gammaln(y + 1)`` of shape ``(n_spots,)``.
    sigma
        Gene-specific log-normal standard deviation ``sigma_g >= 0``.
    only_ll
        If ``True``, skips posterior moment derivatives for fast ``sigma_g`` search.

    Returns
    -------
    ll
        Per-spot marginal log-likelihood ``log P(Y_i = y_i | lam_i, sigma)``.
    d1
        First derivative ``d ll_i / d(log lam_i)``.
    q
        Negative second derivative ``-d^2 ll_i / d(log lam_i)^2 > 0``.
    """
    lam_safe = np.maximum(lam, 1e-300)
    log_lam = np.log(lam_safe)

    if sigma <= 1e-4:
        ll = y * log_lam - lam_safe - log_fact_y
        if sigma > 0.0:
            ll = ll + 0.5 * (sigma * sigma) * ((y - lam_safe) ** 2 - lam_safe)
        if only_ll:
            empty = np.empty(0, dtype=np.float64)
            return ll, empty, empty
        d1 = y - lam_safe
        q = np.maximum(lam_safe, 1e-12)
        return ll, d1, q

    sig2 = sigma * sigma
    inv_sig2 = 1.0 / sig2

    # 4 vectorized Newton steps locate exact concave mode eps_hat to ~1e-13
    shrink = (sig2 * lam_safe) / (1.0 + sig2 * lam_safe)
    eps_hat = np.clip(shrink * np.log((y + 0.5) / (lam_safe + 0.5)), -10.0, 10.0)
    for _ in range(4):
        le = np.minimum(lam_safe * np.exp(eps_hat), 1e16)
        grad_e = y - le - eps_hat * inv_sig2
        hess_e = le + inv_sig2
        eps_hat = np.clip(eps_hat + grad_e / hess_e, -22.0, 22.0)

    le_hat = np.minimum(lam_safe * np.exp(eps_hat), 1e16)
    tau = 1.0 / np.sqrt(le_hat + inv_sig2)

    # Quadrature grid: shape (n_spots, _GH_ORDER)
    eps_grid = eps_hat[:, None] + (np.sqrt(2.0) * tau[:, None]) * _GH_Z[None, :]
    exp_eps_grid = np.exp(np.clip(eps_grid, -35.0, 35.0))

    log_u = (
        _GH_LOG_W[None, :]
        + np.log(tau / sigma)[:, None]
        + y[:, None] * (log_lam[:, None] + eps_grid)
        - lam_safe[:, None] * exp_eps_grid
        - (0.5 * inv_sig2) * (eps_grid * eps_grid)
        - log_fact_y[:, None]
        + (_GH_Z * _GH_Z)[None, :]
    )

    ll = logsumexp(log_u, axis=1)
    if only_ll:
        empty = np.empty(0, dtype=np.float64)
        return ll, empty, empty

    w_post = np.exp(log_u - ll[:, None])

    mean_exp_eps = np.sum(w_post * exp_eps_grid, axis=1)
    mean_eps = np.sum(w_post * eps_grid, axis=1)

    regime_large = (sig2 * le_hat) > 1.0
    d1 = np.where(regime_large, mean_eps * inv_sig2, y - lam_safe * mean_exp_eps)

    # Numerically stable curvature in both small- and large-(sigma^2 * lam) regimes
    diff_exp = exp_eps_grid - mean_exp_eps[:, None]
    var_exp_eps = np.sum(w_post * (diff_exp * diff_exp), axis=1)
    q_small = lam_safe * mean_exp_eps - (lam_safe * lam_safe) * var_exp_eps

    diff_eps = eps_grid - mean_eps[:, None]
    var_eps = np.sum(w_post * (diff_eps * diff_eps), axis=1)
    q_large = inv_sig2 - var_eps * (inv_sig2 * inv_sig2)

    q = np.where(regime_large, q_large, q_small)
    q = np.maximum(q, 1e-12)
    return ll, d1, q


def estimate_sigma_gene(
    y: np.ndarray,
    lam: np.ndarray,
    log_fact_y: np.ndarray,
    sigma_bounds: tuple[float, float] = (0.01, 3.5),
) -> float:
    """Maximize marginal Poisson-lognormal log-likelihood over ``sigma_g``."""
    lo, hi = sigma_bounds
    grid = np.geomspace(lo, hi, 9)
    vals = np.array(
        [
            float(np.sum(poisson_lognormal_moments(y, lam, log_fact_y, float(s), only_ll=True)[0]))
            for s in grid
        ]
    )
    best_idx = int(np.argmax(vals))
    b_lo = float(grid[max(0, best_idx - 1)])
    b_hi = float(grid[min(len(grid) - 1, best_idx + 1)])

    def neg_ll(s: float) -> float:
        ll, _, _ = poisson_lognormal_moments(y, lam, log_fact_y, float(s), only_ll=True)
        return -float(np.sum(ll))

    res = minimize_scalar(
        neg_ll,
        bounds=(b_lo, b_hi),
        method="bounded",
        options={"xatol": 2e-3, "maxiter": 18},
    )
    return float(np.clip(res.x, lo, hi))


def fit_gene_cside(
    y: np.ndarray,
    n_umi: np.ndarray,
    weights: np.ndarray,
    design: np.ndarray,
    intercept_mask: np.ndarray,
    settings: str = "default",
    sigma_init: float = 0.4,
    estimate_sigma: bool = True,
    max_iter: int = 40,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Fit cell-type x covariate parameters ``alpha`` and ``sigma_g`` for one gene.

    Parameters
    ----------
    y
        Counts for gene ``g`` across spots, shape ``(I,)``.
    n_umi
        Total UMI library sizes per spot, shape ``(I,)``.
    weights
        Spot x cell-type proportions, shape ``(I, K)``.
    design
        Spot x covariate design matrix, shape ``(I, D)``.
    intercept_mask
        Boolean mask of length ``D`` indicating baseline/intercept columns.
    settings
        ``"default"`` or ``"R_default"``.
    sigma_init
        Initial overdispersion SD.
    estimate_sigma
        Whether to optimize ``sigma_g`` per gene.
    max_iter
        Maximum damped Fisher-scoring iterations per outer stage.
    tol
        Gradient convergence tolerance.

    Returns
    -------
    alpha
        Fitted coefficient array of shape ``(D, K)``.
    cov
        Asymptotic covariance matrix of ``alpha.ravel()`` of shape ``(D*K, D*K)``.
    sigma_g
        Estimated gene-specific log-normal overdispersion SD.
    converged_mask
        Boolean array of shape ``(D, K)`` indicating non-singular/converged
        parameter estimates.
    log_lik
        Final marginal log-likelihood at ``(alpha, sigma_g)``.
    """
    n_spots, d_dim = design.shape
    k_types = weights.shape[1]
    n_params = d_dim * k_types
    log_fact_y = gammaln(y + 1.0)

    # Ridge penalty vector in C-order (d, k) -> d * K + k
    if settings == "default":
        ridge_d = np.where(intercept_mask, 1e-6, 1e-4)
    else:
        ridge_d = np.zeros(d_dim, dtype=np.float64)
    ridge_vec = np.repeat(ridge_d, k_types)

    # Initialize X @ alpha_k ~ log(mean_rate) across all cell types
    mean_rate = max(float(np.sum(y)) / max(float(np.sum(n_umi)), 1.0), 1e-7)
    log_r0 = np.clip(np.log(mean_rate), -18.0, 2.0)
    c0, _, _, _ = np.linalg.lstsq(design, np.ones(n_spots, dtype=np.float64), rcond=None)
    alpha = np.repeat((log_r0 * c0)[:, None], k_types, axis=1)

    nw = n_umi[:, None] * weights  # shape (I, K)

    def _eval_model(
        alpha_flat: np.ndarray, sig: float, with_hessian: bool = True
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        a_mat = alpha_flat.reshape(d_dim, k_types)
        eta = np.clip(design @ a_mat, -25.0, 8.0)
        lam_ik = nw * np.exp(eta)
        lam = np.maximum(np.sum(lam_ik, axis=1), 1e-300)
        pi_ik = lam_ik / lam[:, None]

        ll_vec, d1, q = poisson_lognormal_moments(y, lam, log_fact_y, sig)
        pen = 0.5 * float(np.dot(ridge_vec, alpha_flat * alpha_flat))
        obj = float(np.sum(ll_vec)) - pen

        # Jacobian J[i, d*K + k] = X[i, d] * pi[i, k]
        jac = (design[:, :, None] * pi_ik[:, None, :]).reshape(n_spots, n_params)
        grad = jac.T @ d1 - ridge_vec * alpha_flat
        if with_hessian:
            h_fs = (jac * q[:, None]).T @ jac + np.diag(ridge_vec)
        else:
            h_fs = np.empty((0, 0), dtype=np.float64)
        return obj, grad, h_fs, lam

    def _solve_alpha(alpha_start: np.ndarray, sig: float, n_it: int) -> np.ndarray:
        cur = alpha_start.ravel().copy()
        obj, grad, h_fs, _ = _eval_model(cur, sig, with_hessian=True)
        lm = 1e-6
        eye_p = np.eye(n_params, dtype=np.float64)

        for _ in range(n_it):
            if float(np.max(np.abs(grad))) < tol:
                break
            try:
                step = np.linalg.solve(h_fs + lm * eye_p, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(h_fs + max(lm, 1e-3) * eye_p, grad, rcond=None)[0]

            max_step = float(np.max(np.abs(step)))
            if max_step > 2.5:
                step = step * (2.5 / max_step)

            accepted = False
            scale = 1.0
            slope = float(np.dot(grad, step))
            for _ls in range(10):
                cand = np.clip(cur + scale * step, -25.0, 10.0)
                cand_obj, _, _, _ = _eval_model(cand, sig, with_hessian=False)
                if np.isfinite(cand_obj) and cand_obj >= obj + 1e-4 * scale * max(slope, 0.0) - 1e-9:
                    cur = cand
                    diff_obj = abs(cand_obj - obj)
                    obj, grad, h_fs, _ = _eval_model(cur, sig, with_hessian=True)
                    lm = max(1e-8, lm * 0.5)
                    accepted = True
                    if diff_obj < 1e-7 and float(np.max(np.abs(grad))) < 1e-3:
                        return cur.reshape(d_dim, k_types)
                    break
                scale *= 0.5

            if not accepted:
                lm = min(1e4, lm * 10.0)

        if float(np.max(np.abs(grad))) > 1e-2:
            # Polish via L-BFGS-B with exact analytical gradient
            def neg_obj_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
                val, g, _, _ = _eval_model(x, sig, with_hessian=False)
                return -val, -g

            bounds = [(-25.0, 10.0)] * n_params
            opt = minimize(
                neg_obj_grad,
                cur,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-9},
            )
            if np.isfinite(opt.fun) and (-opt.fun >= obj):
                cur = opt.x

        return cur.reshape(d_dim, k_types)

    sigma_g = float(sigma_init)
    if estimate_sigma:
        # Stage 1: initial fit at sigma_init, estimate sigma_g, then refit alpha
        alpha = _solve_alpha(alpha, sigma_g, max(12, max_iter // 3))
        _, _, _, lam_hat = _eval_model(alpha.ravel(), sigma_g, with_hessian=False)
        sigma_g = estimate_sigma_gene(y, lam_hat, log_fact_y)
        alpha = _solve_alpha(alpha, sigma_g, max_iter)
        # One light refinement pass if sigma_g shifted noticeably
        _, _, _, lam_hat = _eval_model(alpha.ravel(), sigma_g, with_hessian=False)
        sigma_new = estimate_sigma_gene(y, lam_hat, log_fact_y)
        if abs(sigma_new - sigma_g) > 0.06:
            sigma_g = sigma_new
            alpha = _solve_alpha(alpha, sigma_g, max(12, max_iter // 3))
    else:
        alpha = _solve_alpha(alpha, sigma_g, max_iter)

    alpha_flat = alpha.ravel()
    final_obj, final_grad, h_fs, lam_final = _eval_model(alpha_flat, sigma_g)
    ll_final = float(
        np.sum(poisson_lognormal_moments(y, lam_final, log_fact_y, sigma_g)[0])
    )

    if settings == "default":
        try:
            cov = np.linalg.inv(h_fs)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(h_fs, rcond=1e-10)
        diag_var = np.diag(cov)
        converged_mask = (
            np.isfinite(diag_var) & (diag_var > 0.0)
        ).reshape(d_dim, k_types)
    else:
        # R_default: unpenalized Fisher information & strict test_error check
        # matching spacexr::check_converged_vec
        eta = np.clip(design @ alpha, -25.0, 8.0)
        lam_ik = nw * np.exp(eta)
        lam = np.maximum(np.sum(lam_ik, axis=1), 1e-300)
        pi_ik = lam_ik / lam[:, None]
        _, _, q = poisson_lognormal_moments(y, lam, log_fact_y, sigma_g)
        jac = (design[:, :, None] * pi_ik[:, None, :]).reshape(n_spots, n_params)
        h_unpen = (jac * q[:, None]).T @ jac

        converged_mask = np.ones((d_dim, k_types), dtype=bool)
        # Check per-cell-type Fisher block condition
        for k in range(k_types):
            idx_k = np.arange(d_dim) * k_types + k
            block_k = h_unpen[np.ix_(idx_k, idx_k)]
            eig_k = np.linalg.eigvalsh(0.5 * (block_k + block_k.T))
            if eig_k[0] <= 1e-5 or (eig_k[-1] / max(eig_k[0], 1e-15)) > 1e7:
                converged_mask[:, k] = False

        try:
            cov = np.linalg.inv(h_unpen + 1e-10 * np.eye(n_params))
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(h_unpen, rcond=1e-10)

        diag_var = np.diag(cov)
        se_flat = np.where(diag_var > 0.0, np.sqrt(np.maximum(diag_var, 0.0)), np.inf)
        ok_se = np.isfinite(se_flat) & (se_flat <= 5.0) & (np.abs(final_grad) < 0.5)
        converged_mask &= ok_se.reshape(d_dim, k_types)

    return alpha, cov, sigma_g, converged_mask, ll_final
