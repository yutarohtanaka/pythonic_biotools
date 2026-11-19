"""
Stage-2 group-prior SuSiE-RSS fine-mapping over joint (genes + cis-SNPs) via
Iterative Bayesian Stepwise Selection (IBSS) with PSD LD regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .impute import SettingsMode, validate_settings


@dataclass
class SuSiERSSGroupResult:
    """Result of Stage-2 group-prior SuSiE-RSS fine-mapping on a single LD region."""

    susie_pip: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    null_prob: np.ndarray
    credible_sets: list[dict[str, Any]]
    elbo: float
    n_iter: int
    converged: bool
    R_regularized: np.ndarray

    @property
    def pip(self) -> np.ndarray:
        """Alias for ``susie_pip``."""
        return self.susie_pip


def regularize_ld_matrix(R_joint: np.ndarray, eig_floor: float = 1e-4) -> np.ndarray:
    """Project ``R_joint`` onto the positive-definite correlation cone with floor ``eig_floor``."""
    r_sym = 0.5 * (R_joint + R_joint.T)
    eigvals, eigvecs = np.linalg.eigh(r_sym)
    eigvals_clipped = np.maximum(eigvals, eig_floor)
    r_psd = (eigvecs * eigvals_clipped[None, :]) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(r_psd), 1e-12))
    r_corr = (r_psd / d[:, None]) / d[None, :]
    r_corr = (1.0 - eig_floor) * r_corr + eig_floor * np.eye(len(r_sym), dtype=float)
    r_corr = 0.5 * (r_corr + r_corr.T)
    np.fill_diagonal(r_corr, 1.0)
    return r_corr


def susie_rss_group_prior(
    z_joint: np.ndarray,
    R_joint: np.ndarray,
    is_gene: np.ndarray,
    pi_gene: float,
    pi_snp: float,
    var_gene: float,
    var_snp: float,
    n_samples: int,
    L: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    settings: SettingsMode = "default",
) -> SuSiERSSGroupResult:
    """Group-prior SuSiE-RSS IBSS over joint molecular traits and cis-SNPs.

    Models ``z_joint ~ N(sqrt(n) R b, R)`` with ``b = sum_{l=1}^L b_l`` where each
    single effect ``l`` has group-specific causal prior probabilities ``pi_gene / L``
    vs ``pi_snp / L`` (plus explicit null weight ``pi_0 = 1 - sum_j (pi_j / L)`` so
    inactive effects ``l > L_true`` place posterior mass on the null rather than
    inflating background PIPs) and group-specific prior effect variances
    ``var_gene`` vs ``var_snp``.
    """
    validate_settings(settings)
    z = np.asarray(z_joint, dtype=float)
    r_raw = np.asarray(R_joint, dtype=float)
    ig = np.asarray(is_gene, dtype=bool)

    p = len(z)
    if p == 0:
        raise ValueError("z_joint must be non-empty.")
    if r_raw.shape != (p, p):
        raise ValueError(f"R_joint shape {r_raw.shape} must match ({p}, {p}).")
    if ig.shape != (p,):
        raise ValueError(f"is_gene shape {ig.shape} must match ({p},).")
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if int(L) < 1:
        raise ValueError(f"L must be >= 1, got {L}.")
    if not (0.0 < float(coverage) < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {coverage}.")

    n_eff = float(n_samples)
    num_l = int(L)

    eig_floor = 1e-4 if settings == "default" else 1e-6
    r_reg = regularize_ld_matrix(r_raw, eig_floor=eig_floor)

    # Per-effect prior causal inclusion probabilities and null configuration weight
    pi_vec = np.where(ig, max(float(pi_gene), 0.0), max(float(pi_snp), 0.0)) / float(num_l)
    sum_pi_l = float(np.sum(pi_vec))
    if sum_pi_l <= 0.0:
        # Degenerate zero-prior fallback: pure null
        zeros_lp = np.zeros((num_l, p), dtype=float)
        return SuSiERSSGroupResult(
            susie_pip=np.zeros(p, dtype=float),
            alpha=zeros_lp,
            mu=zeros_lp.copy(),
            mu2=zeros_lp.copy(),
            null_prob=np.ones(num_l, dtype=float),
            credible_sets=[],
            elbo=0.0,
            n_iter=1,
            converged=True,
            R_regularized=r_reg,
        )

    if sum_pi_l >= 0.99:
        pi_vec = pi_vec * (0.99 / sum_pi_l)
        pi_null = 0.01
    else:
        pi_null = max(1.0 - sum_pi_l, 1e-6)

    log_pi = np.where(pi_vec > 0.0, np.log(np.maximum(pi_vec, 1e-300)), -np.inf)
    log_pi_null = float(np.log(pi_null))

    # Standardized noncentrality prior variances V_j = n * var_j (with R_jj = 1)
    v_vec = np.maximum(
        n_eff * np.where(ig, max(float(var_gene), 0.0), max(float(var_snp), 0.0)),
        1e-8,
    )
    post_var_gamma = v_vec / (1.0 + v_vec)  # posterior variance of gamma_lj | j causal
    shrink = post_var_gamma  # since R_jj == 1, mu_gamma = shrink * r_{-l, j}
    lbf_const = -0.5 * np.log1p(v_vec)

    # Variational parameters on gamma = sqrt(n) * b scale
    alpha = np.zeros((num_l, p), dtype=float)
    mu_gamma = np.zeros((num_l, p), dtype=float)
    mu2_gamma = np.zeros((num_l, p), dtype=float)
    null_prob = np.ones(num_l, dtype=float)
    ser_kl = np.zeros(num_l, dtype=float)

    # Maintain R_reg @ sum_l (alpha_l * mu_gamma_l) incrementally in O(P) per effect update
    gamma_bar_total = np.zeros(p, dtype=float)
    r_gamma_total = np.zeros(p, dtype=float)

    prev_elbo = -np.inf
    prev_pip = np.zeros(p, dtype=float)
    converged = False
    n_iter_done = 0

    for it in range(1, int(max_iter) + 1):
        n_iter_done = it
        for l_idx in range(num_l):
            old_bar_l = alpha[l_idx] * mu_gamma[l_idx]
            if np.any(old_bar_l != 0.0):
                gamma_bar_total = gamma_bar_total - old_bar_l
                r_gamma_total = r_gamma_total - (r_reg @ old_bar_l)

            # Partial residual Z-score vector r_{-l} = z - R_reg @ sum_{l' != l} gamma_bar_{l'}
            r_res = z - r_gamma_total

            # Exact single-effect Bayes factor & posterior update
            lbf_l = lbf_const + 0.5 * shrink * (r_res * r_res)
            log_u_feat = log_pi + lbf_l
            log_u_all = np.concatenate([[log_pi_null], log_u_feat])
            log_z_l = float(logsumexp(log_u_all))

            alpha_l = np.exp(log_u_feat - log_z_l)
            null_l = float(np.exp(log_pi_null - log_z_l))
            mu_l = shrink * r_res
            mu2_l = (mu_l * mu_l) + post_var_gamma

            alpha[l_idx] = alpha_l
            null_prob[l_idx] = null_l
            mu_gamma[l_idx] = mu_l
            mu2_gamma[l_idx] = mu2_l

            # Exact KL(q_l || p_l) via SER identity:
            # E_{q_l}[r_{-l}^T gamma_l - 0.5 gamma_l^T diag(R) gamma_l] - KL(q_l || p_l) = log_z_l
            expected_ser_ll = float(np.sum(alpha_l * (r_res * mu_l - 0.5 * mu2_l)))
            ser_kl[l_idx] = expected_ser_ll - log_z_l

            new_bar_l = alpha_l * mu_l
            if np.any(new_bar_l != 0.0):
                gamma_bar_total = gamma_bar_total + new_bar_l
                r_gamma_total = r_gamma_total + (r_reg @ new_bar_l)

        # Compute exact RSS variational ELBO (up to additive data constant -0.5 z^T R^{-1} z)
        r_gamma_total = r_reg @ gamma_bar_total
        elbo = float(np.dot(z, gamma_bar_total) - 0.5 * np.dot(gamma_bar_total, r_gamma_total))
        var_correction = 0.0
        for l_idx in range(num_l):
            bar_l = alpha[l_idx] * mu_gamma[l_idx]
            var_correction += float(np.sum(alpha[l_idx] * mu2_gamma[l_idx]) - np.dot(bar_l, r_reg @ bar_l))
        elbo -= 0.5 * var_correction + float(np.sum(ser_kl))

        pip = 1.0 - np.prod(1.0 - alpha, axis=0)
        pip_diff = float(np.max(np.abs(pip - prev_pip)))
        elbo_diff = abs(elbo - prev_elbo)

        prev_elbo = elbo
        prev_pip = pip
        if it > 1 and elbo_diff < tol and pip_diff < tol:
            converged = True
            break

    susie_pip = np.clip(1.0 - np.prod(1.0 - alpha, axis=0), 0.0, 1.0)

    # Extract purity-filtered non-null credible sets at target coverage
    credible_sets: list[dict[str, Any]] = []
    seen_sets: set[tuple[int, ...]] = set()

    for l_idx in range(num_l):
        non_null_mass = float(np.sum(alpha[l_idx]))
        if non_null_mass < coverage:
            continue
        order = np.argsort(-alpha[l_idx], kind="mergesort")
        csum = np.cumsum(alpha[l_idx, order])
        k_end = int(np.searchsorted(csum, coverage, side="left")) + 1
        cs_indices = [int(i) for i in order[:k_end]]
        achieved_cov = float(csum[min(k_end - 1, p - 1)])

        if len(cs_indices) == 1:
            purity = 1.0
        else:
            sub_r = np.abs(r_raw[np.ix_(cs_indices, cs_indices)])
            purity = float(np.min(sub_r))

        key = tuple(sorted(cs_indices))
        if purity >= float(min_abs_corr) and key not in seen_sets:
            seen_sets.add(key)
            credible_sets.append(
                {
                    "effect_index": l_idx + 1,
                    "indices": cs_indices,
                    "coverage": achieved_cov,
                    "purity": purity,
                    "contains_gene": bool(np.any(ig[cs_indices])),
                }
            )

    # Convert posterior moments from gamma = sqrt(n) * b back to standardized effect b
    inv_sqrt_n = 1.0 / np.sqrt(n_eff)
    mu_b = mu_gamma * inv_sqrt_n
    mu2_b = mu2_gamma * (inv_sqrt_n * inv_sqrt_n)

    return SuSiERSSGroupResult(
        susie_pip=susie_pip,
        alpha=alpha,
        mu=mu_b,
        mu2=mu2_b,
        null_prob=null_prob,
        credible_sets=credible_sets,
        elbo=float(prev_elbo),
        n_iter=n_iter_done,
        converged=converged,
        R_regularized=r_reg,
    )
