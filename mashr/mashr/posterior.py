"""Grid selection, likelihood evaluation, SQUAREM mixture optimization, and exact posteriors."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import nnls
from scipy.special import logsumexp, ndtr

from ._linalg import (
    equilibrated_cholesky,
    gaussian_loglik_batch,
    posterior_moments_joseph,
    symmetrize,
)
from .covariances import _validate_u_dict
from .data import MashData
from .results import MashResult


def autoselect_grid(
    data: MashData,
    grid_mult: float = float(np.sqrt(2.0)),
) -> np.ndarray:
    """Automatically construct geometric variance scale grid ``omega_l = sigma_l^2``.

    Follows Urbut et al. (2019) Eq. 2.2 with safe lower/upper bounds so pure-null or
    weak-signal data never collapse to a degenerate or single-point grid.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    gm = float(grid_mult)
    if not np.isfinite(gm) or gm <= 1.0:
        raise ValueError(f"grid_mult must be finite and > 1.0; got {grid_mult}.")

    st = data.Shat_tilde
    bt = data.Bhat_tilde
    sigma_min = max(float(np.min(st)) / 10.0, 1e-150)
    delta_max = max(0.0, float(np.max(bt * bt - st * st)))
    sigma_max = max(8.0 * sigma_min, 2.0 * np.sqrt(delta_max))

    n_steps = max(2, int(np.ceil(np.log(sigma_max / sigma_min) / np.log(gm))) + 1)
    sigmas = sigma_min * (gm ** np.arange(n_steps, dtype=np.float64))
    return (sigmas * sigmas).astype(np.float64)


def _validate_grid(grid: np.ndarray | Sequence[float]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 1 or g.size == 0:
        raise ValueError("grid must be a non-empty 1D array of positive variances.")
    if not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("grid values must be finite and strictly positive (> 0).")
    return g.copy()


def _normalize_u_mats(
    names: list[str],
    mats: list[np.ndarray],
    normalize_u: bool,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    out_mats: list[np.ndarray] = []
    for name, mat in zip(names, mats):
        if normalize_u:
            max_diag = float(np.max(np.diag(mat)))
            if max_diag <= 0.0:
                raise ValueError(
                    f"Cannot normalize covariance {name!r} with max diagonal {max_diag} <= 0."
                )
            out_mats.append(mat / max_diag)
        else:
            out_mats.append(mat.copy())
    return {name: mat for name, mat in zip(names, out_mats)}, out_mats


def _expand_components(
    u_mats: list[np.ndarray],
    grid: np.ndarray,
    use_pointmass: bool,
) -> list[np.ndarray | None]:
    comps: list[np.ndarray | None] = []
    if use_pointmass:
        comps.append(None)
    for u_mat in u_mats:
        for omega in grid:
            comps.append(float(omega) * u_mat)
    return comps


def compute_loglik_matrix(
    data: MashData,
    U_list: dict[str, np.ndarray] | Sequence[np.ndarray],
    grid: np.ndarray | Sequence[float],
    use_pointmass: bool = True,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate `(J, M)` matrix of multivariate normal log-likelihoods.

    Entry ``(j, m)`` is ``log N(b_tilde_j; 0, Sigma_m + V_tilde_j)`` with
    fast paths for homoscedastic ``is_common_cov`` data and diagonal covariances.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")

    J, R = data.shape
    grid_arr = _validate_grid(grid)
    if isinstance(U_list, dict) and len(U_list) == 0:
        if not use_pointmass:
            raise ValueError("U_list cannot be empty when use_pointmass=False.")
        u_mats: list[np.ndarray] = []
    else:
        _, u_mats = _validate_u_dict(U_list, R)

    comps = _expand_components(u_mats, grid_arr, use_pointmass)
    M = len(comps)
    loglik = np.empty((J, M), dtype=np.float64)

    B_tilde = data.Bhat_tilde

    if data.is_common_cov:
        V0 = data.get_cov(0)
        b_eq_cache: np.ndarray | None = None
        for m, sigma in enumerate(comps):
            C_m = V0 if sigma is None else (sigma + V0)
            L_eq, d = equilibrated_cholesky(C_m)
            b_eq_cache = B_tilde / d[None, :]
            y = np.linalg.solve(L_eq, b_eq_cache.T).T
            diag_L = np.diagonal(L_eq)
            logdet = 2.0 * (
                np.sum(np.log(d)) + np.sum(np.log(np.maximum(diag_L, 1e-300)))
            )
            loglik[:, m] = -0.5 * (
                R * np.log(2.0 * np.pi) + logdet + np.sum(y * y, axis=1)
            )
        return loglik

    def _is_diag_u(u: np.ndarray) -> bool:
        if u.shape[0] <= 1:
            return True
        diag_u = np.maximum(np.diag(u), 0.0)
        off_u = u - np.diag(np.diag(u))
        scale_u = np.sqrt(np.outer(diag_u, diag_u))
        return bool(np.all(np.abs(off_u) <= 1e-12 * scale_u))

    all_u_diag = data.is_diagonal_v and all(_is_diag_u(u) for u in u_mats)
    if all_u_diag:
        v_diag = np.diag(data.V)
        var_noise = (data.Shat_tilde ** 2) * v_diag[None, :]
        b_sq = B_tilde ** 2
        sigma_diags = np.zeros((M, R), dtype=np.float64)
        for m, sigma in enumerate(comps):
            if sigma is not None:
                sigma_diags[m] = np.maximum(np.diag(sigma), 0.0)
        log2pi = np.log(2.0 * np.pi)
        for start in range(0, J, batch_size):
            end = min(start + batch_size, J)
            var_m = sigma_diags[None, :, :] + var_noise[start:end, None, :]
            loglik[start:end, :] = -0.5 * np.sum(
                log2pi + np.log(var_m) + b_sq[start:end, None, :] / var_m,
                axis=-1,
            )
        return loglik

    st = data.Shat_tilde
    V = data.V
    for start in range(0, J, batch_size):
        end = min(start + batch_size, J)
        st_b = st[start:end]
        B_b = B_tilde[start:end]
        V_b = st_b[:, :, None] * V[None, :, :] * st_b[:, None, :]
        for m, sigma in enumerate(comps):
            C_b = V_b if sigma is None else (sigma[None, :, :] + V_b)
            L_eq, d = equilibrated_cholesky(C_b)
            ll_m, _ = gaussian_loglik_batch(L_eq, d, B_b)
            loglik[start:end, m] = ll_m

    return loglik


def optimize_pi(
    loglik_matrix: np.ndarray,
    prior: str | np.ndarray | Sequence[float] = "nullbiased",
    nullweight: float = 10.0,
    use_pointmass: bool = True,
    opt_method: str = "mixsqp",
    max_iter: int = 5000,
    tol: float = 1e-6,
    pi_init: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, float, float, int, bool, float]:
    """Optimize mixture weights ``pi`` on the simplex with Dirichlet prior penalty.

    Supports Mix-SQP active-set sequential quadratic programming (``"mixsqp"``,
    Kim et al. 2020), projected SQUAREM (``"squarem"``), and standard EM (``"em"``),
    certifying global optimality via the exact KKT optimality gap.

    Returns ``(pi, loglik, penalized_loglik, n_iter, converged, kkt_gap)``.
    """
    L_mat = np.asarray(loglik_matrix, dtype=np.float64)
    if L_mat.ndim != 2 or L_mat.shape[0] < 1 or L_mat.shape[1] < 1:
        raise ValueError("loglik_matrix must be a non-empty 2D array.")
    if not np.all(np.isfinite(L_mat)):
        raise ValueError("loglik_matrix contains NaN or Inf values.")

    J, M = L_mat.shape

    nw = float(nullweight)
    if not np.isfinite(nw) or nw < 1.0:
        raise ValueError(f"nullweight must be finite and >= 1.0; got {nullweight}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and >= 0; got {tol}.")
    if opt_method not in {"mixsqp", "squarem", "em"}:
        raise ValueError(
            f"opt_method must be 'mixsqp', 'squarem', or 'em'; got {opt_method!r}."
        )

    if isinstance(prior, str):
        if prior == "nullbiased":
            alpha_prior = np.ones(M, dtype=np.float64)
            if use_pointmass:
                alpha_prior[0] = nw
        elif prior == "uniform":
            alpha_prior = np.ones(M, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown prior {prior!r}; expected 'nullbiased', 'uniform', or array."
            )
    else:
        alpha_prior = np.asarray(prior, dtype=np.float64)
        if alpha_prior.shape != (M,):
            raise ValueError(f"prior array must have shape ({M},); got {alpha_prior.shape}.")
        if not np.all(np.isfinite(alpha_prior)) or np.any(alpha_prior < 1.0):
            raise ValueError("prior pseudo-counts must be finite and >= 1.0.")

    if pi_init is None:
        pi = np.full(M, 1.0 / M, dtype=np.float64)
    else:
        pi = np.asarray(pi_init, dtype=np.float64).copy()
        if pi.shape != (M,):
            raise ValueError(f"pi_init must have shape ({M},); got {pi.shape}.")
        if not np.all(np.isfinite(pi)) or np.any(pi < 0.0) or np.sum(pi) <= 0.0:
            raise ValueError("pi_init must be finite, non-negative, and sum to > 0.")
        pi /= np.sum(pi)

    if M == 1:
        ll = float(np.sum(L_mat[:, 0]))
        return np.array([1.0], dtype=np.float64), ll, ll, 0, True, 0.0

    lam = alpha_prior - 1.0
    pos_lam = lam > 0.0
    pos_lam_idx = np.flatnonzero(pos_lam)
    j_eff = float(J) + float(np.sum(lam))

    c = np.max(L_mat, axis=1)
    sum_c = float(np.sum(c))
    A = np.exp(L_mat - c[:, None])

    def _pen_only(p: np.ndarray) -> float:
        ap = np.maximum(A @ p, 1e-300)
        ll_val = sum_c + float(np.sum(np.log(ap)))
        if pos_lam_idx.size > 0:
            return ll_val + float(
                np.sum(lam[pos_lam_idx] * np.log(np.maximum(p[pos_lam_idx], 1e-300)))
            )
        return ll_val

    def _eval_obj(p: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        ap = np.maximum(A @ p, 1e-300)
        ll_val = sum_c + float(np.sum(np.log(ap)))
        if pos_lam_idx.size > 0:
            pen_val = ll_val + float(
                np.sum(lam[pos_lam_idx] * np.log(np.maximum(p[pos_lam_idx], 1e-300)))
            )
        else:
            pen_val = ll_val
        grad = A.T @ (1.0 / ap)
        return ll_val, pen_val, grad, ap

    def _em_step(p: np.ndarray, grad: np.ndarray | None = None) -> np.ndarray:
        if grad is None:
            ap = np.maximum(A @ p, 1e-300)
            grad = A.T @ (1.0 / ap)
        p_next = (p * grad + lam) / j_eff
        p_next = np.maximum(p_next, 0.0)
        s = float(np.sum(p_next))
        return p_next / s if s > 0.0 else np.full(M, 1.0 / M, dtype=np.float64)

    def _kkt_vec(p: np.ndarray, grad: np.ndarray) -> tuple[float, np.ndarray]:
        g_m = grad.copy()
        if pos_lam_idx.size > 0:
            g_m[pos_lam_idx] += lam[pos_lam_idx] / np.maximum(p[pos_lam_idx], 1e-300)
        return float(np.max(g_m) - j_eff), g_m

    ll, pen_ll, grad, ap = _eval_obj(pi)
    kkt_gap, g_m = _kkt_vec(pi, grad)
    if kkt_gap <= tol:
        return pi, ll, pen_ll, 0, True, kkt_gap

    converged = False
    n_iter = 0

    if opt_method == "mixsqp":
        # Brief 10-step SQUAREM warm-start prunes inactive grid components rapidly.
        warm_steps = min(10, max_iter)
        for _ in range(warm_steps):
            n_iter += 1
            pi1 = _em_step(pi, grad)
            pi2 = _em_step(pi1)
            r = pi1 - pi
            v = (pi2 - pi1) - r
            rn = float(np.linalg.norm(r))
            vn = float(np.linalg.norm(v))
            if vn >= 1e-15 and rn >= 1e-15:
                gam = max(-20.0, min(-1.0, -rn / vn))
                pc = np.maximum(pi - 2.0 * gam * r + (gam * gam) * v, 1e-15)
                pc /= float(np.sum(pc))
                pc = _em_step(pc)
                if _pen_only(pc) >= _pen_only(pi2):
                    pi = pc
                else:
                    pi = pi2
            else:
                pi = pi2
            ll, pen_ll, grad, ap = _eval_obj(pi)
            kkt_gap, g_m = _kkt_vec(pi, grad)
            if kkt_gap <= tol:
                converged = True
                break

        if not converged:
            scale_eq = np.sqrt(j_eff) * 100.0
            for _ in range(n_iter + 1, max_iter + 1):
                n_iter += 1
                idx = np.flatnonzero((pi > 1e-12) | (g_m >= j_eff - 1e-3) | pos_lam)
                k_act = int(idx.size)
                w = 1.0 / ap
                Ba = w[:, None] * A[:, idx]
                rows: list[np.ndarray] = [Ba]
                rhs_parts: list[np.ndarray] = [np.full(J, 2.0, dtype=np.float64)]

                lam_a = lam[idx]
                pos_a = lam_a > 0.0
                if np.any(pos_a):
                    pos_a_idx = np.flatnonzero(pos_a)
                    p_a_pos = np.maximum(pi[idx[pos_a_idx]], 1e-300)
                    d_rows = np.zeros((pos_a_idx.size, k_act), dtype=np.float64)
                    d_rows[np.arange(pos_a_idx.size), pos_a_idx] = (
                        np.sqrt(lam_a[pos_a_idx]) / p_a_pos
                    )
                    rows.append(d_rows)
                    rhs_parts.append(2.0 * np.sqrt(lam_a[pos_a_idx]))

                rows.append(np.full((1, k_act), scale_eq, dtype=np.float64))
                rhs_parts.append(np.array([scale_eq], dtype=np.float64))
                rows.append(1e-6 * np.eye(k_act, dtype=np.float64))
                rhs_parts.append(1e-6 * pi[idx])

                mat_nnls = np.vstack(rows)
                vec_nnls = np.concatenate(rhs_parts)
                # Compress tall NNLS system via thin QR when J >> k_act
                if mat_nnls.shape[0] > 2 * k_act + 16:
                    aug = np.column_stack([mat_nnls, vec_nnls])
                    r_qr = np.linalg.qr(aug, mode="r")
                    mat_solve = r_qr[:, :k_act]
                    vec_solve = r_qr[:, k_act]
                else:
                    mat_solve = mat_nnls
                    vec_solve = vec_nnls

                q_sub, _ = nnls(mat_solve, vec_solve)
                s_q = float(np.sum(q_sub))
                if s_q <= 0.0:
                    pi = _em_step(pi, grad)
                    ll, pen_ll, grad, ap = _eval_obj(pi)
                else:
                    q = np.zeros(M, dtype=np.float64)
                    q[idx] = q_sub / s_q
                    if pos_lam_idx.size > 0:
                        q[pos_lam_idx] = np.maximum(q[pos_lam_idx], 1e-15)
                        q /= float(np.sum(q))
                    dir_p = q - pi
                    slope = float(np.dot(g_m, dir_p))
                    if slope <= 0.0:
                        pi = _em_step(pi, grad)
                        ll, pen_ll, grad, ap = _eval_obj(pi)
                    else:
                        alpha = 1.0
                        accepted = False
                        for _ in range(20):
                            p_try = np.maximum(pi + alpha * dir_p, 0.0)
                            if pos_lam_idx.size > 0:
                                p_try[pos_lam_idx] = np.maximum(p_try[pos_lam_idx], 1e-15)
                            p_try /= float(np.sum(p_try))
                            ll_t, pen_t, grad_t, ap_t = _eval_obj(p_try)
                            if pen_t >= pen_ll + 1e-4 * alpha * slope:
                                pi, ll, pen_ll, grad, ap = p_try, ll_t, pen_t, grad_t, ap_t
                                accepted = True
                                break
                            alpha *= 0.5
                        if not accepted:
                            pi = _em_step(pi, grad)
                            ll, pen_ll, grad, ap = _eval_obj(pi)

                kkt_gap, g_m = _kkt_vec(pi, grad)
                if kkt_gap <= tol:
                    converged = True
                    break

        return pi, ll, pen_ll, n_iter, converged, kkt_gap

    for it in range(1, max_iter + 1):
        n_iter = it
        if opt_method == "em":
            pi = _em_step(pi, grad)
            ll, pen_ll, grad, ap = _eval_obj(pi)
        else:
            pi1 = _em_step(pi, grad)
            pi2 = _em_step(pi1)
            r = pi1 - pi
            v = (pi2 - pi1) - r
            r_norm = float(np.linalg.norm(r))
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-15 or r_norm < 1e-15:
                pi = pi2
                ll, pen_ll, grad, ap = _eval_obj(pi)
            else:
                pen_pi2 = _pen_only(pi2)
                gamma = max(-20.0, min(-1.0, -r_norm / v_norm))
                accepted = False
                for _ in range(6):
                    # Positive floor 1e-15 prevents SQUAREM overshoot from freezing
                    # active unpenalized coordinates at exact 0 under EM.
                    pi_cand = np.maximum(pi - 2.0 * gamma * r + (gamma * gamma) * v, 1e-15)
                    pi_cand /= float(np.sum(pi_cand))
                    if _pen_only(pi_cand) >= pen_pi2 - 1e-14:
                        pi = _em_step(pi_cand)
                        ll, pen_ll, grad, ap = _eval_obj(pi)
                        accepted = True
                        break
                    gamma = 0.5 * (gamma - 1.0)
                if not accepted:
                    pi = pi2
                    ll, pen_ll, grad, ap = _eval_obj(pi)

        kkt_gap, g_m = _kkt_vec(pi, grad)
        if kkt_gap <= tol:
            converged = True
            break

    return pi, ll, pen_ll, n_iter, converged, kkt_gap


def compute_posterior_matrices(
    data: MashData,
    U_list: dict[str, np.ndarray] | Sequence[np.ndarray],
    grid: np.ndarray | Sequence[float],
    pi: np.ndarray | Sequence[float],
    loglik_matrix: np.ndarray,
    use_pointmass: bool = True,
    output_posterior_cov: bool = False,
    batch_size: int = 512,
) -> dict[str, np.ndarray | None]:
    """Compute exact mixture posterior moments and point-mass-aware LFSR."""
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")

    J, R = data.shape
    grid_arr = _validate_grid(grid)
    if isinstance(U_list, dict) and len(U_list) == 0:
        if not use_pointmass:
            raise ValueError("U_list cannot be empty when use_pointmass=False.")
        u_mats: list[np.ndarray] = []
    else:
        _, u_mats = _validate_u_dict(U_list, R)

    comps = _expand_components(u_mats, grid_arr, use_pointmass)
    M = len(comps)

    pi_arr = np.asarray(pi, dtype=np.float64)
    if pi_arr.shape != (M,):
        raise ValueError(f"pi length {pi_arr.shape} does not match M={M}.")
    L_mat = np.asarray(loglik_matrix, dtype=np.float64)
    if L_mat.shape != (J, M):
        raise ValueError(f"loglik_matrix shape {L_mat.shape} != ({J}, {M}).")

    log_pi = np.where(pi_arr > 0.0, np.log(np.maximum(pi_arr, 1e-300)), -np.inf)
    log_unnorm = L_mat + log_pi[None, :]
    log_norm = logsumexp(log_unnorm, axis=1)
    weights = np.exp(log_unnorm - log_norm[:, None])
    weights = np.where(pi_arr[None, :] > 0.0, weights, 0.0)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = np.divide(weights, np.maximum(weights_sum, 1e-300))

    mu_tilde = np.zeros((J, R), dtype=np.float64)
    second_diag_tilde = np.zeros((J, R), dtype=np.float64)
    second_cov_tilde = (
        np.zeros((J, R, R), dtype=np.float64) if output_posterior_cov else None
    )
    pos_prob = np.zeros((J, R), dtype=np.float64)
    neg_prob = np.zeros((J, R), dtype=np.float64)
    zero_prob = np.zeros((J, R), dtype=np.float64)

    B_tilde = data.Bhat_tilde
    st = data.Shat_tilde
    V = data.V
    common_cov = data.is_common_cov
    V0 = data.get_cov(0) if common_cov else None
    v_diag = np.maximum(np.diag(V), 0.0)
    v0_diag = np.maximum(np.diag(V0), 0.0)[None, :] if V0 is not None else None
    eye_r = np.eye(R, dtype=np.float64)

    def _accumulate_sign_probs(
        row_slice: slice,
        w_m: np.ndarray,
        mu_m: np.ndarray,
        psi_diag_m: np.ndarray,
        sigma_diag: np.ndarray,
        v_tilde_diag: np.ndarray,
    ) -> None:
        sd_m = np.sqrt(np.maximum(psi_diag_m, 0.0))
        sd_ref = np.minimum(np.sqrt(sigma_diag), np.sqrt(np.maximum(v_tilde_diag, 0.0)))
        cont = (sigma_diag > 0.0) & (sd_m > 1e-12 * sd_ref)
        if np.all(cont):
            z_m = mu_m / sd_m
            p_pos_m = ndtr(z_m)
            p_neg_m = ndtr(-z_m)
            p_zero_m = np.zeros_like(p_pos_m)
        elif not np.any(cont):
            p_pos_m = (mu_m > 0.0).astype(np.float64)
            p_neg_m = (mu_m < 0.0).astype(np.float64)
            p_zero_m = (mu_m == 0.0).astype(np.float64)
        else:
            safe_sd = np.where(cont, sd_m, 1.0)
            z_m = np.where(cont, mu_m / safe_sd, 0.0)
            p_pos_cont = ndtr(z_m)
            p_neg_cont = ndtr(-z_m)
            p_pos_m = np.where(cont, p_pos_cont, (mu_m > 0.0).astype(np.float64))
            p_neg_m = np.where(cont, p_neg_cont, (mu_m < 0.0).astype(np.float64))
            p_zero_m = np.where(cont, 0.0, (mu_m == 0.0).astype(np.float64))

        pos_prob[row_slice] += w_m * p_pos_m
        neg_prob[row_slice] += w_m * p_neg_m
        zero_prob[row_slice] += w_m * p_zero_m

    for m, sigma in enumerate(comps):
        w_m = weights[:, m : m + 1]
        if float(np.max(w_m)) == 0.0:
            continue

        if sigma is None:
            zero_prob += w_m
            continue

        sigma_diag = np.maximum(np.diag(sigma), 0.0)[None, :]

        if common_cov:
            assert V0 is not None and v0_diag is not None
            C_m = sigma + V0
            L_eq, d = equilibrated_cholesky(C_m)
            b_eq = B_tilde / d[None, :]
            y = np.linalg.solve(L_eq, b_eq.T).T
            W_m = np.linalg.solve(L_eq, sigma / d[:, None])
            mu_m = y @ W_m
            Z_m = np.linalg.solve(L_eq.T, W_m)
            K_m = Z_m.T / d[None, :]
            I_K = eye_r - K_m
            Psi_m = symmetrize(I_K @ sigma @ I_K.T + K_m @ V0 @ K_m.T)
            psi_diag_m = np.maximum(np.diag(Psi_m), 0.0)[None, :]

            mu_tilde += w_m * mu_m
            second_diag_tilde += w_m * (psi_diag_m + mu_m * mu_m)
            if output_posterior_cov:
                assert second_cov_tilde is not None
                second_cov_tilde += w_m[:, :, None] * (
                    Psi_m[None, :, :] + mu_m[:, :, None] * mu_m[:, None, :]
                )
            _accumulate_sign_probs(
                slice(None), w_m, mu_m, psi_diag_m, sigma_diag, v0_diag
            )
        else:
            for start in range(0, J, batch_size):
                end = min(start + batch_size, J)
                w_b = w_m[start:end]
                if float(np.max(w_b)) == 0.0:
                    continue
                st_b = st[start:end]
                B_b = B_tilde[start:end]
                V_b = st_b[:, :, None] * V[None, :, :] * st_b[:, None, :]
                C_b = sigma[None, :, :] + V_b
                L_eq, d = equilibrated_cholesky(C_b)
                _, y = gaussian_loglik_batch(L_eq, d, B_b)
                mu_b, psi_b = posterior_moments_joseph(
                    L_eq, d, y, sigma, V_b, full_cov=output_posterior_cov
                )
                if output_posterior_cov:
                    assert second_cov_tilde is not None
                    psi_diag_b = np.maximum(np.diagonal(psi_b, axis1=-2, axis2=-1), 0.0)
                    second_cov_tilde[start:end] += w_b[:, :, None] * (
                        psi_b + mu_b[:, :, None] * mu_b[:, None, :]
                    )
                else:
                    psi_diag_b = psi_b

                mu_tilde[start:end] += w_b * mu_b
                second_diag_tilde[start:end] += w_b * (psi_diag_b + mu_b * mu_b)

                v_tilde_diag_b = (st_b ** 2) * v_diag[None, :]
                _accumulate_sign_probs(
                    slice(start, end), w_b, mu_b, psi_diag_b, sigma_diag, v_tilde_diag_b
                )

    var_tilde = np.maximum(0.0, second_diag_tilde - mu_tilde * mu_tilde)
    scale = data.scale_factor
    posterior_mean = scale * mu_tilde
    posterior_sd = scale * np.sqrt(var_tilde)

    posterior_cov: np.ndarray | None = None
    if output_posterior_cov:
        assert second_cov_tilde is not None
        cov_tilde = second_cov_tilde - mu_tilde[:, :, None] * mu_tilde[:, None, :]
        posterior_cov = symmetrize((scale[:, :, None] * cov_tilde) * scale[:, None, :])

    pos_prob = np.clip(pos_prob, 0.0, 1.0)
    neg_prob = np.clip(neg_prob, 0.0, 1.0)
    zero_prob = np.clip(1.0 - pos_prob - neg_prob, 0.0, 1.0)
    lfsr = np.clip(1.0 - np.maximum(pos_prob, neg_prob), 0.0, 1.0)

    return {
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "posterior_cov": posterior_cov,
        "lfsr": lfsr,
        "negative_prob": neg_prob,
        "positive_prob": pos_prob,
        "zero_prob": zero_prob,
        "posterior_weights": weights,
    }


def mash(
    data: MashData,
    Ulist: dict[str, np.ndarray] | Sequence[np.ndarray],
    grid: np.ndarray | Sequence[float] | None = None,
    grid_mult: float = float(np.sqrt(2.0)),
    normalize_u: bool = True,
    use_pointmass: bool = True,
    prior: str | np.ndarray | Sequence[float] = "nullbiased",
    nullweight: float = 10.0,
    opt_method: str = "mixsqp",
    max_iter: int = 5000,
    tol: float = 1e-6,
    compute_posterior: bool = True,
    output_posterior_cov: bool = False,
    pi_init: np.ndarray | Sequence[float] | None = None,
    fix_g: bool = False,
    batch_size: int = 512,
) -> MashResult:
    """Fit the Multivariate Adaptive Shrinkage (MASH) empirical Bayes model.

    Parameters
    ----------
    data : MashData
        Input data container created by :func:`mash_set_data`.
    Ulist : dict[str, ndarray] or sequence of ndarray
        Prior covariance matrices of shape `(R, R)`.
    grid : array-like, optional
        Positive variance scale grid. Automatically selected when ``None``.
    grid_mult : float, default=sqrt(2)
        Geometric multiplier for :func:`autoselect_grid`.
    normalize_u : bool, default=True
        Divide each ``U_k`` by its maximum diagonal entry before grid expansion.
    use_pointmass : bool, default=True
        Include an exact point-mass-at-zero null component ``U_0 = 0``.
    prior : {"nullbiased", "uniform"} or array-like, default="nullbiased"
        Dirichlet prior pseudo-counts on mixture weights ``pi``.
    nullweight : float, default=10.0
        Dirichlet pseudo-count on the null component when ``prior="nullbiased"``.
    opt_method : {"mixsqp", "squarem", "em"}, default="mixsqp"
        Convex mixture optimization solver.
    max_iter : int, default=5000
        Maximum mixture optimization iterations.
    tol : float, default=1e-6
        KKT optimality gap stopping tolerance.
    compute_posterior : bool, default=True
        Whether to compute posterior moments and ``lfsr``.
    output_posterior_cov : bool, default=False
        Whether to compute full `(J, R, R)` posterior covariance matrices.
    pi_init : array-like, optional
        Initial or fixed mixture weights of length ``M``.
    fix_g : bool, default=False
        If ``True``, evaluate likelihoods and posteriors using fixed ``grid`` and ``pi_init``.
    batch_size : int, default=512
        Batch size for heteroscedastic likelihood/posterior evaluation.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    J, R = data.shape

    names, mats = _validate_u_dict(Ulist, R)
    u_dict, u_mats = _normalize_u_mats(names, mats, normalize_u=normalize_u)

    if grid is None:
        if fix_g:
            raise ValueError("grid must be explicitly provided when fix_g=True.")
        grid_arr = autoselect_grid(data, grid_mult=grid_mult)
    else:
        grid_arr = _validate_grid(grid)

    K = len(u_mats)
    L = int(grid_arr.size)
    M = (1 if use_pointmass else 0) + K * L

    loglik_mat = compute_loglik_matrix(
        data=data,
        U_list=u_dict,
        grid=grid_arr,
        use_pointmass=use_pointmass,
        batch_size=batch_size,
    )

    if use_pointmass:
        null_loglik = float(np.sum(loglik_mat[:, 0]))
    else:
        null_col = compute_loglik_matrix(
            data=data,
            U_list={},
            grid=np.array([1.0], dtype=np.float64),
            use_pointmass=True,
            batch_size=batch_size,
        )
        null_loglik = float(np.sum(null_col[:, 0]))

    if fix_g:
        if pi_init is None:
            raise ValueError("pi_init must be provided when fix_g=True.")
        pi = np.asarray(pi_init, dtype=np.float64).copy()
        if pi.shape != (M,):
            raise ValueError(f"pi_init must have length M={M}; got {pi.shape}.")
        if not np.all(np.isfinite(pi)) or np.any(pi < 0.0) or np.sum(pi) <= 0.0:
            raise ValueError("pi_init must be finite, non-negative, and sum to > 0.")
        pi /= np.sum(pi)
        c = np.max(loglik_mat, axis=1)
        ap = np.maximum(np.exp(loglik_mat - c[:, None]) @ pi, 1e-300)
        loglik = float(np.sum(c) + np.sum(np.log(ap)))
        pen_loglik = loglik
        n_iter = 0
        converged = True
        kkt_gap = 0.0
    else:
        pi, loglik, pen_loglik, n_iter, converged, kkt_gap = optimize_pi(
            loglik_matrix=loglik_mat,
            prior=prior,
            nullweight=nullweight,
            use_pointmass=use_pointmass,
            opt_method=opt_method,
            max_iter=max_iter,
            tol=tol,
            pi_init=pi_init,
        )

    if compute_posterior:
        post = compute_posterior_matrices(
            data=data,
            U_list=u_dict,
            grid=grid_arr,
            pi=pi,
            loglik_matrix=loglik_mat,
            use_pointmass=use_pointmass,
            output_posterior_cov=output_posterior_cov,
            batch_size=batch_size,
        )
        posterior_mean = post["posterior_mean"]
        posterior_sd = post["posterior_sd"]
        posterior_cov = post["posterior_cov"]
        lfsr = post["lfsr"]
        negative_prob = post["negative_prob"]
        positive_prob = post["positive_prob"]
        zero_prob = post["zero_prob"]
        posterior_weights = post["posterior_weights"]
    else:
        posterior_mean = np.full((J, R), np.nan, dtype=np.float64)
        posterior_sd = np.full((J, R), np.nan, dtype=np.float64)
        posterior_cov = None
        lfsr = np.full((J, R), np.nan, dtype=np.float64)
        negative_prob = np.full((J, R), np.nan, dtype=np.float64)
        positive_prob = np.full((J, R), np.nan, dtype=np.float64)
        zero_prob = np.full((J, R), np.nan, dtype=np.float64)
        posterior_weights = np.full((J, M), np.nan, dtype=np.float64)

    assert posterior_mean is not None
    assert posterior_sd is not None
    assert lfsr is not None
    assert negative_prob is not None
    assert positive_prob is not None
    assert zero_prob is not None
    assert posterior_weights is not None

    return MashResult(
        feature_names=list(data.feature_names),
        condition_names=list(data.condition_names),
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        posterior_cov=posterior_cov,
        lfsr=lfsr,
        negative_prob=negative_prob,
        positive_prob=positive_prob,
        zero_prob=zero_prob,
        posterior_weights=posterior_weights,
        pi=pi,
        grid=grid_arr,
        U_list=u_dict,
        loglik_matrix=loglik_mat,
        loglik=loglik,
        penalized_loglik=pen_loglik,
        null_loglik=null_loglik,
        use_pointmass=use_pointmass,
        alpha=data.alpha,
        converged=converged,
        n_iter=n_iter,
        kkt_gap=kkt_gap,
    )
