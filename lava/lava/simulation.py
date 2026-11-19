"""
Vectorized non-central Wishart sampling and Rao-Blackwellized conditional
normal-product / non-central chi-square p-value integration for `lava`
(Werme et al., Nature Genetics 2022).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ncx2, norm

from .locus import SettingsMode


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root V diag(sqrt(max(eig, 0))) V^T."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    root = np.sqrt(np.maximum(eigvals, 0.0))
    return (eigvecs * root[np.newaxis, :]) @ eigvecs.T


def simulate_wishart_omega(
    K: int,
    sigma: np.ndarray,
    omega_psd: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    settings: SettingsMode = "default",
    pd_floor: float = 1e-6,
) -> np.ndarray:
    """
    Draw `n_sim` unbiased local genetic covariance matrices $\\hat{\\Omega}^{(s)}$ from
    the non-central Wishart sampling model $S^{(s)} \\sim W_P(K, \\Sigma, K \\Omega)$.

    Uses exact Bartlett decomposition for the central Wishart bulk when $K > 2P + 2$,
    achieving $O(n_{\\text{sim}} P^3)$ complexity independent of $K$.
    """
    P = sigma.shape[0]
    L_sig = _matrix_sqrt_psd(sigma)

    # Decompose non-centrality K * omega_psd = M_top^T M_top with rank r <= min(K, P)
    eig_w, vec_w = np.linalg.eigh(0.5 * (omega_psd + omega_psd.T))
    order = np.argsort(eig_w)[::-1]
    r = min(K, P)
    lam_r = np.maximum(eig_w[order[:r]], 0.0) * float(K)
    vec_r = vec_w[:, order[:r]]
    # M_r has shape (r, P): rows are orthogonal vectors whose outer sum is K * omega_psd
    M_r = np.sqrt(lam_r)[:, np.newaxis] * vec_r.T

    if K <= 2 * P + 2:
        Z = rng.standard_normal(size=(n_sim, K, P))
        X = Z @ L_sig.T
        X[:, :r, :] += M_r[np.newaxis, :, :]
        S_sim = np.einsum("skp,skq->spq", X, X)
    else:
        # Top r rows carry all non-centrality
        Z1 = rng.standard_normal(size=(n_sim, r, P))
        X1 = M_r[np.newaxis, :, :] + Z1 @ L_sig.T
        S1 = np.einsum("srp,srq->spq", X1, X1)

        # Remaining nu = K - r >= P + 3 rows follow central Wishart W_P(nu, sigma) via Bartlett
        nu = K - r
        A = np.zeros((n_sim, P, P), dtype=float)
        for i in range(P):
            A[:, i, i] = np.sqrt(rng.chisquare(df=nu - i, size=n_sim))
            if i > 0:
                A[:, i, :i] = rng.standard_normal(size=(n_sim, i))
        W_std = np.einsum("sik,sjk->sij", A, A)
        S2 = np.einsum("pi,sij,qj->spq", L_sig, W_std, L_sig)
        S_sim = S1 + S2

    omega_sim = S_sim / float(K) - sigma[np.newaxis, :, :]
    omega_sim = 0.5 * (omega_sim + np.swapaxes(omega_sim, 1, 2))

    # Vectorized batch spectral PSD projection
    eigvals, eigvecs = np.linalg.eigh(omega_sim)
    if settings == "R_default":
        eig_proj = np.maximum(eigvals, 0.0)
    else:
        needs_proj = eigvals[:, 0] <= 1e-12
        eig_proj = eigvals.copy()
        if np.any(needs_proj):
            eig_proj[needs_proj] = np.maximum(eig_proj[needs_proj], pd_floor)
    out = np.einsum("spj,sj,sqj->spq", eigvecs, eig_proj, eigvecs)
    return 0.5 * (out + np.swapaxes(out, 1, 2))


def _bivar_conditional_p_core(
    K: int,
    omega_aa: float,
    omega_bb: float,
    s_aa: float,
    s_bb: float,
    s_ab: float,
    d_obs: float,
    n_sim: int,
    rng: np.random.Generator,
) -> float:
    """
    Rao-Blackwellized conditional Gaussian tail integral for $H_0: \\omega_{AB} = 0$.

    Evaluates exact $\\mathbb{E}_{\\hat{\\delta}_A}[\\mathbb{P}(|T - K s_{AB}| \\ge d_{\\text{obs}} \\mid \\hat{\\delta}_A)]$
    where $d_{\\text{obs}} = K |\\hat{\\omega}_{AB}|$.
    """
    if d_obs <= 0.0:
        return 1.0
    if s_aa <= 0.0 or s_bb <= 0.0:
        return 1.0

    if K == 1:
        # Variance-matched normal approximation for single-PC edge case
        var_t = max(omega_aa * s_bb + omega_bb * s_aa + s_aa * s_bb + s_ab * s_ab, 1e-30)
        return float(2.0 * norm.sf(d_obs / np.sqrt(var_t)))

    sd_a = np.sqrt(s_aa)
    root_kaa = np.sqrt(max(float(K) * omega_aa, 0.0))
    root_kbb = np.sqrt(max(float(K) * omega_bb, 0.0))

    z1 = rng.standard_normal(size=n_sim)
    z2 = rng.standard_normal(size=n_sim)
    d1 = root_kaa + sd_a * z1
    d2 = sd_a * z2
    if K > 2:
        q_rest = rng.chisquare(df=K - 2, size=n_sim)
        norm_sq = d1 * d1 + d2 * d2 + s_aa * q_rest
    else:
        norm_sq = d1 * d1 + d2 * d2

    slope_s = s_ab / s_aa
    s_b_given_a = max(s_bb - slope_s * s_ab, 0.0)

    # Conditional mean of centered statistic (T - K * s_ab) given delta_hat_A
    mu_tilde = root_kbb * d2 + slope_s * (norm_sq - root_kaa * d1) - float(K) * s_ab
    var_cond = s_b_given_a * norm_sq

    if s_b_given_a <= 1e-15 * s_bb:
        # Degenerate |corr(e_A, e_B)| == 1 fallback
        return float((np.sum(np.abs(mu_tilde) >= d_obs) + 1.0) / (n_sim + 1.0))

    sd_cond = np.sqrt(np.maximum(var_cond, 1e-30))
    p_draws = norm.sf((d_obs - mu_tilde) / sd_cond) + norm.sf((d_obs + mu_tilde) / sd_cond)
    return float(np.clip(np.mean(p_draws), 0.0, 1.0))


def bivariate_null_pvalue(
    K: int,
    omega_xx: float,
    omega_yy: float,
    sigma_2x2: np.ndarray,
    omega_xy_obs: float,
    n_sim: int,
    rng: np.random.Generator,
    settings: SettingsMode = "default",
) -> float:
    """
    Compute two-sided bivariate local genetic covariance p-value under $H_0: \\omega_{XY} = 0$.

    Conditions on whichever trait has larger local genetic signal to minimize Monte Carlo
    variance, and performs `LAVA` adaptive `n_sim -> 10 * n_sim` (`1e4 -> 1e5`) refinement
    in `settings='R_default'` when initial $p < 10^{-3}$.
    """
    w_x = max(float(omega_xx), 0.0)
    w_y = max(float(omega_yy), 0.0)
    s_xx = float(sigma_2x2[0, 0])
    s_yy = float(sigma_2x2[1, 1])
    s_xy = float(sigma_2x2[0, 1])
    d_obs = float(K) * abs(float(omega_xy_obs))

    # Condition on trait with larger genetic variance for minimal MC variance
    if w_x >= w_y:
        w_a, w_b, s_a, s_b = w_x, w_y, s_xx, s_yy
    else:
        w_a, w_b, s_a, s_b = w_y, w_x, s_yy, s_xx

    p_init = _bivar_conditional_p_core(K, w_a, w_b, s_a, s_b, s_xy, d_obs, n_sim, rng)
    if settings == "R_default" and p_init < 1e-3:
        p_refined = _bivar_conditional_p_core(
            K, w_a, w_b, s_a, s_b, s_xy, d_obs, max(n_sim * 10, 100_000), rng
        )
        return p_refined
    return p_init


def omnibus_multireg_null_pvalue(
    K: int,
    omega_xx_mat: np.ndarray,
    omega_yy: float,
    sigma_sub: np.ndarray,
    omega_xy_vec: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    settings: SettingsMode = "default",
) -> float:
    """
    Omnibus p-value for $H_0: \\boldsymbol{\\omega}_{Xy} = \\mathbf{0}$ ($R^2_{\\text{joint}} = 0$)
    in multivariate local genetic multiple regression with $m$ predictors.
    """
    m = omega_xx_mat.shape[0]
    if m == 1:
        return bivariate_null_pvalue(
            K=K,
            omega_xx=float(omega_xx_mat[0, 0]),
            omega_yy=omega_yy,
            sigma_2x2=sigma_sub,
            omega_xy_obs=float(omega_xy_vec[0]),
            n_sim=n_sim,
            rng=rng,
            settings=settings,
        )

    def _eval_mc(num_draws: int) -> float:
        # Null covariance matrix has block-diagonal structure blkdiag(omega_xx_mat, omega_yy)
        omega_0 = np.zeros((m + 1, m + 1), dtype=float)
        omega_0[:m, :m] = omega_xx_mat
        omega_0[m, m] = max(float(omega_yy), 0.0)

        # Observed Mahalanobis-like quadratic form on K * omega_xy_vec
        # Compute theoretical covariance of hat{omega}_{Xy} under H_0:
        # Cov(hat{omega}_{X_i, y}, hat{omega}_{X_j, y}) =
        #   (omega_{X_i, X_j} * sigma_{yy} + sigma_{X_i, X_j} * omega_{yy}
        #    + sigma_{X_i, X_j} * sigma_{yy} + sigma_{X_i, y} * sigma_{X_j, y}) / K
        s_xx = sigma_sub[:m, :m]
        s_yy = float(sigma_sub[m, m])
        s_xy = sigma_sub[:m, m]
        w_y = max(float(omega_yy), 0.0)
        cov_wxy = (
            omega_xx_mat * s_yy
            + s_xx * w_y
            + s_xx * s_yy
            + np.outer(s_xy, s_xy)
        ) / float(max(K, 1))
        cov_inv = np.linalg.pinv(0.5 * (cov_wxy + cov_wxy.T), rcond=1e-10)
        q_obs = float(omega_xy_vec @ cov_inv @ omega_xy_vec)
        if q_obs <= 0.0:
            return 1.0

        # Rao-Blackwellized conditioning on simulated X block when K > m + 1
        if K > m + 1 and float(np.min(np.linalg.eigvalsh(s_xx))) > 1e-14:
            L_sxx = _matrix_sqrt_psd(s_xx)
            s_xx_inv = np.linalg.pinv(s_xx, rcond=1e-12)
            slope_vec = s_xx_inv @ s_xy
            s_y_given_x = max(s_yy - float(s_xy @ slope_vec), 0.0)

            eig_wx, vec_wx = np.linalg.eigh(0.5 * (omega_xx_mat + omega_xx_mat.T))
            M_x = np.sqrt(np.maximum(eig_wx, 0.0) * float(K))[:, np.newaxis] * vec_wx.T  # (m, m)

            # Top m+1 rows: m rows carry M_x, (m+1)-th row carries delta_y = sqrt(K * w_y)
            Z_top = rng.standard_normal(size=(num_draws, m + 1, m))
            X_top = Z_top @ L_sxx.T
            X_top[:, :m, :] += M_x[np.newaxis, :, :]
            S_xx_sim = np.einsum("sri,srj->sij", X_top, X_top)
            if K > m + 1:
                nu = K - (m + 1)
                if nu >= m:
                    A = np.zeros((num_draws, m, m), dtype=float)
                    for idx_b in range(m):
                        A[:, idx_b, idx_b] = np.sqrt(rng.chisquare(df=nu - idx_b, size=num_draws))
                        if idx_b > 0:
                            A[:, idx_b, :idx_b] = rng.standard_normal(size=(num_draws, idx_b))
                    W_std = np.einsum("sik,sjk->sij", A, A)
                    S_xx_sim = S_xx_sim + np.einsum("pi,sij,qj->spq", L_sxx, W_std, L_sxx)
                else:
                    Z_rem = rng.standard_normal(size=(num_draws, nu, m)) @ L_sxx.T
                    S_xx_sim = S_xx_sim + np.einsum("sri,srj->sij", Z_rem, Z_rem)

            # Conditional mean of hat{omega}_{Xy} = (Delta_X^T delta_y)/K - s_xy
            # row m of X_top aligns with delta_y = sqrt(K * w_y)
            term_y = np.sqrt(w_y / float(K)) * X_top[:, m, :]
            # trace-like term from (X_top[:, :m] - M_x)
            diff_top = S_xx_sim - np.einsum("ri,srj->sij", M_x, X_top[:, :m, :])
            mu_wxy = term_y + np.einsum("sij,j->si", diff_top, slope_vec) / float(K) - s_xy[np.newaxis, :]

            if s_y_given_x > 1e-15 * s_yy:
                # Conditional covariance of hat{omega}_{Xy} given Delta_X is (s_y_given_x / K^2) * S_xx_sim
                V_cond = (s_y_given_x / float(K * K)) * S_xx_sim + 1e-18 * np.eye(m)[np.newaxis, :, :]
                # Standardize each draw via cov_inv or V_cond^{-1}: test q_obs = w_obs^T cov_inv w_obs
                # Exact conditional quadratic form w^T V_cond^{-1} w ~ ncx2(df=m, nc=mu^T V_cond^{-1} mu)
                # Use observed standardized norm against mean V_cond to match scale:
                V_bar_inv = np.linalg.pinv(np.mean(V_cond, axis=0) + np.cov(mu_wxy, rowvar=False), rcond=1e-10)
                q_target = float(omega_xy_vec @ V_bar_inv @ omega_xy_vec)
                # Sample normal noise for exact quadratic metric under cov_inv
                eig_v, vec_v = np.linalg.eigh(V_cond)
                z_norm = rng.standard_normal(size=(num_draws, m))
                w_draws = mu_wxy + np.einsum(
                    "sij,sj->si", vec_v, np.sqrt(np.maximum(eig_v, 0.0)) * z_norm
                )
                q_draws = np.einsum("si,ij,sj->s", w_draws, V_bar_inv, w_draws)
                # Smooth tail if q_target exceeds all empirical draws using ncx2 asymptotic match
                emp_hits = int(np.sum(q_draws >= q_target))
                if emp_hits >= 5:
                    return float(emp_hits / num_draws)
                return float(ncx2.sf(q_obs, df=m, nc=0.0))

        # Direct Wishart fallback for small K
        sim_om = simulate_wishart_omega(K, sigma_sub, omega_0, num_draws, rng, settings="R_default")
        w_draws = sim_om[:, :m, m]
        q_draws = np.einsum("si,ij,sj->s", w_draws, cov_inv, w_draws)
        hits = int(np.sum(q_draws >= q_obs))
        return float(hits / num_draws) if hits >= 5 else float(ncx2.sf(q_obs, df=m, nc=0.0))

    p0 = _eval_mc(n_sim)
    if settings == "R_default" and p0 < 1e-3:
        return _eval_mc(max(n_sim * 10, 100_000))
    return p0
