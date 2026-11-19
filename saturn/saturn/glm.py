"""Fast vectorized quasi-binomial logit IRLS solver for satuRn DTU."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, xlogy

from .eb import SettingsMode, validate_settings


def _binomial_deviance(y: np.ndarray, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Compute row-wise binomial deviance ``2 * sum(y log(y / (n*mu)) + ...)``."""
    n_mu = np.maximum(n * mu, 1e-300)
    n_one_minus_mu = np.maximum(n * (1.0 - mu), 1e-300)
    ny = n - y
    term1 = np.where(y > 0.0, xlogy(y, y / n_mu), 0.0)
    term2 = np.where(ny > 0.0, xlogy(ny, ny / n_one_minus_mu), 0.0)
    return 2.0 * np.sum(term1 + term2, axis=1)


def fit_quasibinomial_batch(
    y_batch: np.ndarray,
    n_batch: np.ndarray,
    design_sub: np.ndarray,
    settings: SettingsMode = "default",
    max_abs_beta: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit quasi-binomial logit GLMs simultaneously for transcripts sharing active samples.

    Parameters
    ----------
    y_batch
        Transcript counts of shape ``(n_tx, m_active)`` where ``n_batch > 0``.
    n_batch
        Parent-gene counts of shape ``(n_tx, m_active)`` (all strictly positive).
    design_sub
        Active-sample design matrix of shape ``(m_active, p_cov)``.
    settings
        ``"default"`` (Jeffreys/ridge-stabilized IRLS) or ``"R_default"``
        (unpenalized ``glm.fit`` ``quasibinomial`` IRLS with ``maxit=25, eps=1e-8``).
    max_abs_beta
        Coefficient magnitude bound in ``settings="default"`` (and boundary
        detection threshold in ``settings="R_default"``).

    Returns
    -------
    beta : (n_tx, p_cov)
    cov_unscaled : (n_tx, p_cov, p_cov)
    dispersion : (n_tx,)
    deviance : (n_tx,)
    converged : (n_tx,) bool
    boundary : (n_tx,) bool
    df_residual : int
    """
    validate_settings(settings)
    b_tx, m_obs = y_batch.shape
    p_cov = design_sub.shape[1]
    df_res = m_obs - p_cov

    nan_beta = np.full((b_tx, p_cov), np.nan, dtype=np.float64)
    nan_cov = np.full((b_tx, p_cov, p_cov), np.nan, dtype=np.float64)
    nan_vec = np.full(b_tx, np.nan, dtype=np.float64)
    false_vec = np.zeros(b_tx, dtype=bool)

    if df_res <= 0:
        return nan_beta, nan_cov, nan_vec, nan_vec, false_vec, false_vec, max(0, df_res)

    rank_x = int(np.linalg.matrix_rank(design_sub))
    if rank_x < p_cov:
        return nan_beta, nan_cov, nan_vec, nan_vec, false_vec, false_vec, df_res

    eye_p = np.eye(p_cov, dtype=np.float64)

    if settings == "default":
        # Per-observation Jeffreys/Haldane (+0.25 / +0.5) continuity prior + mild
        # 1e-6 ridge so complete all-0 vs all-1 switches remain finite & scale with N
        delta_j = 0.25
        y_fit = y_batch + delta_j
        n_fit = n_batch + 2.0 * delta_j

        mustart = np.clip((y_fit + 0.5) / (n_fit + 1.0), 1e-6, 1.0 - 1e-6)
        eta = np.log(mustart / (1.0 - mustart))
        w0 = n_fit * mustart * (1.0 - mustart)
        xtwx0 = np.einsum("bm,mp,mq->bpq", w0, design_sub, design_sub) + 1e-6 * eye_p[None, :, :]
        rhs0 = np.einsum("bm,mp->bp", w0 * eta, design_sub)
        beta = np.linalg.solve(xtwx0, rhs0[..., None])[..., 0]
        beta = np.clip(beta, -max_abs_beta, max_abs_beta)

        converged = np.zeros(b_tx, dtype=bool)
        for _ in range(35):
            eta = np.clip(beta @ design_sub.T, -20.0, 20.0)
            mu = expit(eta)
            v = np.maximum(mu * (1.0 - mu), 1e-10)
            w = n_fit * v
            xtwx = (
                np.einsum("bm,mp,mq->bpq", w, design_sub, design_sub)
                + 1e-6 * eye_p[None, :, :]
            )
            score = (y_fit - n_fit * mu) @ design_sub - 1e-6 * beta
            step = np.linalg.solve(xtwx, score[..., None])[..., 0]
            # Damping on large Newton jumps
            max_step = np.max(np.abs(step), axis=1, keepdims=True)
            scale = np.where(max_step > 3.0, 3.0 / np.maximum(max_step, 1e-12), 1.0)
            beta_new = np.clip(beta + scale * step, -max_abs_beta, max_abs_beta)
            diff = np.max(np.abs(beta_new - beta), axis=1)
            beta = beta_new
            converged |= diff < 1e-7
            if np.all(converged):
                break

        eta_final = np.clip(beta @ design_sub.T, -20.0, 20.0)
        mu_final = expit(eta_final)
        v_final = np.maximum(mu_final * (1.0 - mu_final), 1e-9)
        w_cov = n_fit * v_final
        xtwx_final = (
            np.einsum("bm,mp,mq->bpq", w_cov, design_sub, design_sub)
            + 1e-6 * eye_p[None, :, :]
        )
        cov_unscaled = np.linalg.inv(xtwx_final)

        # Pearson quasi-dispersion on observed counts against stabilized mu_final
        pearson_num = (y_batch - n_batch * mu_final) ** 2
        pearson_den = np.maximum(n_batch * v_final, 1e-12)
        dispersion = np.sum(pearson_num / pearson_den, axis=1) / float(df_res)
        deviance = _binomial_deviance(y_batch, n_batch, np.clip(mu_final, 1e-12, 1.0 - 1e-12))
        boundary = np.max(np.abs(beta), axis=1) >= (max_abs_beta - 1e-3)
        converged = np.ones(b_tx, dtype=bool)
        return beta, cov_unscaled, dispersion, deviance, converged, boundary, df_res

    # settings == "R_default": unpenalized R stats::glm.fit(family = quasibinomial())
    mustart = np.clip((y_batch + 0.5) / (n_batch + 1.0), 1e-10, 1.0 - 1e-10)
    eta = np.log(mustart / (1.0 - mustart))
    dev_old = _binomial_deviance(y_batch, n_batch, mustart)
    beta = np.zeros((b_tx, p_cov), dtype=np.float64)
    converged = np.zeros(b_tx, dtype=bool)
    active = np.ones(b_tx, dtype=bool)

    for _ in range(25):
        if not np.any(active):
            break
        eta_a = np.clip(eta[active], -30.0, 30.0)
        mu_a = expit(eta_a)
        v_a = np.maximum(mu_a * (1.0 - mu_a), 1e-15)
        w_a = n_batch[active] * v_a
        z_a = eta_a + (y_batch[active] / n_batch[active] - mu_a) / v_a

        xtwx_a = np.einsum("bm,mp,mq->bpq", w_a, design_sub, design_sub)
        rhs_a = np.einsum("bm,mp->bp", w_a * z_a, design_sub)

        # Solve per active transcript cleanly even if one separates to singular W
        beta_a = np.empty((int(np.sum(active)), p_cov), dtype=np.float64)
        for k_loc in range(beta_a.shape[0]):
            try:
                beta_a[k_loc] = np.linalg.solve(xtwx_a[k_loc], rhs_a[k_loc])
            except np.linalg.LinAlgError:
                beta_a[k_loc] = np.linalg.lstsq(xtwx_a[k_loc], rhs_a[k_loc], rcond=None)[0]

        eta_new = np.clip(beta_a @ design_sub.T, -30.0, 30.0)
        mu_new = np.clip(expit(eta_new), 1e-15, 1.0 - 1e-15)
        dev_new = _binomial_deviance(y_batch[active], n_batch[active], mu_new)

        rel_change = np.abs(dev_new - dev_old[active]) / (0.1 + np.abs(dev_new))
        just_conv = rel_change < 1e-8

        act_idx = np.flatnonzero(active)
        beta[act_idx] = beta_a
        eta[act_idx] = eta_new
        dev_old[act_idx] = dev_new
        converged[act_idx[just_conv]] = True
        active[act_idx[just_conv]] = False

    mu_final = np.clip(expit(np.clip(eta, -30.0, 30.0)), 1e-15, 1.0 - 1e-15)
    v_final = np.maximum(mu_final * (1.0 - mu_final), 1e-15)
    w_final = n_batch * v_final
    xtwx_final = np.einsum("bm,mp,mq->bpq", w_final, design_sub, design_sub)

    cov_unscaled = np.full((b_tx, p_cov, p_cov), np.nan, dtype=np.float64)
    for k_idx in range(b_tx):
        mat_k = xtwx_final[k_idx]
        eig_k = np.linalg.eigvalsh(0.5 * (mat_k + mat_k.T))
        if eig_k[0] > 1e-13 and np.isfinite(eig_k[-1]):
            cov_unscaled[k_idx] = np.linalg.inv(mat_k)
        else:
            converged[k_idx] = False

    pearson_num = (y_batch - n_batch * mu_final) ** 2
    pearson_den = np.maximum(n_batch * v_final, 1e-15)
    dispersion = np.sum(pearson_num / pearson_den, axis=1) / float(df_res)
    deviance = dev_old
    boundary = np.max(np.abs(beta), axis=1) > max_abs_beta

    # Flag non-estimable models with NaN dispersion if covariance failed
    bad_cov = ~np.all(np.isfinite(cov_unscaled), axis=(1, 2))
    dispersion = np.where(bad_cov, np.nan, dispersion)
    return beta, cov_unscaled, dispersion, deviance, converged, boundary, df_res
