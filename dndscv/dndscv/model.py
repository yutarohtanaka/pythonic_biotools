"""
Statistical models for `dndscv`:
1. 192-parameter coding-strand trinucleotide Poisson MLE (`192r_3w` / `192r_syn`)
2. Global selection ratios (`wmis`, `wnon`, `wspl`, `wtru`, `wall`) with Schur-complement
   Wald and profile-likelihood confidence intervals
3. Negative Binomial Type-2 (Gamma-Poisson) covariate regression (`dndscv_nb`) with
   automatic Poisson fallback as theta -> infinity
4. Closed-form gene-level conditional LRTs (`cv` and `p`), indel LRT, Fisher combined
   p-values, and NaN-aware Benjamini-Hochberg FDR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

_MAX_LOG_THETA = 12.5
_MIN_LOG_THETA = -14.0
_R_THETA_POISSON_CEIL = 1e4
_R_THETA_MIN_FLOOR = 1e-3

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Strictly validate `settings` parameter (`'default'` or `'R_default'`)."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unsupported settings={settings!r}; expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class RateModelFit:
    """Result of fitting a 192-trinucleotide rate + global omega Poisson model."""

    r_mle: np.ndarray  # (192,)
    w_mle: np.ndarray  # (K,)
    loglik: float
    converged: bool


@dataclass(frozen=True)
class GlobalDnDsFit:
    """Five global selection ratios (`wmis`, `wnon`, `wspl`, `wtru`, `wall`) and CIs."""

    names: list[str]
    mle: np.ndarray
    cilow: np.ndarray
    cihigh: np.ndarray
    n_obs: np.ndarray
    n_exp: np.ndarray
    p_value: np.ndarray
    r_mle: np.ndarray


@dataclass(frozen=True)
class NBCovariateFit:
    """Negative Binomial Type-2 (or Poisson fallback) covariate regression fit."""

    coef_names: list[str]
    beta: np.ndarray
    se: np.ndarray
    p_beta: np.ndarray
    theta: float
    alpha: float
    is_poisson: bool
    rho_all: np.ndarray  # exp(X @ beta) per gene
    mu_all: np.ndarray  # offset_exp * rho_all per gene
    loglik: float
    converged: bool


def bh_fdr_nan_aware(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR computed strictly over finite p-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    q = np.full(p_arr.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(p_arr)
    p_sub = p_arr[valid_mask]
    m = p_sub.size
    if m == 0:
        return q
    order = np.argsort(p_sub, kind="mergesort")
    p_sorted = p_sub[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(((m * p_sorted) / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_sub = np.empty(m, dtype=np.float64)
    q_sub[order] = q_sorted
    q[valid_mask] = q_sub
    return q


def _poisson_192_loglik(n_mat: np.ndarray, mu_mat: np.ndarray) -> float:
    """Poisson log-likelihood up to data-only constants (0*log(0) = 0)."""
    pos = n_mat > 0
    mu_safe = np.maximum(mu_mat, 1e-300)
    term = np.where(pos, n_mat * np.log(mu_safe), 0.0)
    return float(np.sum(term - mu_mat))


def fit_192_rate_model(
    n_bg: np.ndarray,
    L_bg: np.ndarray,
    *,
    rate_model: Literal["192r_3w", "192r_syn"] = "192r_3w",
    fixed_idx: int | None = None,
    fixed_val: float | None = None,
    w_init: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> RateModelFit:
    """
    Fit the 192-parameter coding-strand trinucleotide substitution rate model.

    `n_bg` and `L_bg` have shape `(192, 1 + K)` where column `0` is `syn`
    and columns `1..K` are the selection impact classes.
    """
    n_mat = np.asarray(n_bg, dtype=np.float64)
    L_mat = np.asarray(L_bg, dtype=np.float64)
    if n_mat.ndim != 2 or L_mat.shape != n_mat.shape or n_mat.shape[0] != 192:
        raise ValueError("n_bg and L_bg must both have shape (192, 1 + K).")
    K = n_mat.shape[1] - 1
    if K < 1:
        raise ValueError("At least one non-synonymous class column is required.")

    total_muts = float(np.sum(n_mat))
    if total_muts == 0.0:
        w0 = np.ones(K, dtype=np.float64)
        if fixed_idx is not None and fixed_val is not None:
            w0[fixed_idx] = float(fixed_val)
        return RateModelFit(
            r_mle=np.zeros(192, dtype=np.float64),
            w_mle=w0,
            loglik=0.0,
            converged=True,
        )

    n_syn_tot = float(np.sum(n_mat[:, 0]))
    if n_syn_tot <= 0.0:
        raise ValueError(
            "Cannot estimate dN/dS: zero synonymous mutations observed in background genes."
        )

    L_syn = L_mat[:, 0]
    L_nonsyn = L_mat[:, 1:]
    n_tot_c = np.sum(n_mat[:, 1:], axis=0)

    if rate_model == "192r_syn":
        r = np.where(L_syn > 0.0, n_mat[:, 0] / np.maximum(L_syn, 1e-300), 0.0)
        exp_c = r @ L_nonsyn
        w = np.where(exp_c > 0.0, n_tot_c / np.maximum(exp_c, 1e-300), 0.0)
        if fixed_idx is not None and fixed_val is not None:
            w[fixed_idx] = float(fixed_val)
        mu_mat = np.column_stack([r * L_syn, r[:, None] * L_nonsyn * w[None, :]])
        return RateModelFit(
            r_mle=r,
            w_mle=w,
            loglik=_poisson_192_loglik(n_mat, mu_mat),
            converged=True,
        )

    if rate_model != "192r_3w":
        raise ValueError(f"Unsupported rate_model {rate_model!r}; expected '192r_3w' or '192r_syn'.")

    n_tot_j = np.sum(n_mat, axis=1)
    w = np.ones(K, dtype=np.float64) if w_init is None else np.asarray(w_init, dtype=np.float64).copy()
    if fixed_idx is not None and fixed_val is not None:
        w[fixed_idx] = float(fixed_val)

    prev_ll = -np.inf
    converged = False
    r = np.zeros(192, dtype=np.float64)

    for _ in range(max_iter):
        denom_j = L_syn + L_nonsyn @ w
        r = np.where(denom_j > 0.0, n_tot_j / np.maximum(denom_j, 1e-300), 0.0)
        exp_c = r @ L_nonsyn
        w_new = np.where(exp_c > 0.0, n_tot_c / np.maximum(exp_c, 1e-300), 0.0)
        if fixed_idx is not None and fixed_val is not None:
            w_new[fixed_idx] = float(fixed_val)

        mu_mat = np.column_stack([r * L_syn, r[:, None] * L_nonsyn * w_new[None, :]])
        ll = _poisson_192_loglik(n_mat, mu_mat)

        rel_w = float(np.max(np.abs(w_new - w) / (1.0 + np.abs(w))))
        rel_ll = abs(ll - prev_ll) / (1.0 + abs(ll)) if np.isfinite(prev_ll) else np.inf
        w = w_new
        prev_ll = ll
        if rel_w <= tol and rel_ll <= tol:
            converged = True
            break

    # Final exact r update at converged w so normalization identity holds tightly
    denom_j = L_syn + L_nonsyn @ w
    r = np.where(denom_j > 0.0, n_tot_j / np.maximum(denom_j, 1e-300), 0.0)
    mu_mat = np.column_stack([r * L_syn, r[:, None] * L_nonsyn * w[None, :]])
    ll = _poisson_192_loglik(n_mat, mu_mat)

    return RateModelFit(r_mle=r, w_mle=w, loglik=ll, converged=converged)


def _schur_wald_se_logw(
    r_mle: np.ndarray,
    w_mle: np.ndarray,
    L_bg: np.ndarray,
) -> np.ndarray:
    """
    Compute SE(log w_k) via the Schur complement of the 192x192 rate Fisher block.
    Returns np.nan for classes with zero expected/observed mutations.
    """
    K = w_mle.size
    se = np.full(K, np.nan, dtype=np.float64)
    active_k = np.where(w_mle > 0.0)[0]
    if active_k.size == 0:
        return se

    mu_syn = r_mle * L_bg[:, 0]
    B_full = r_mle[:, None] * L_bg[:, 1:] * w_mle[None, :]
    B = B_full[:, active_k]
    D = mu_syn + np.sum(B, axis=1)
    pos_j = D > 0.0
    if not np.any(pos_j):
        return se

    D_pos = D[pos_j]
    B_pos = B[pos_j, :]
    M = np.diag(np.sum(B_pos, axis=0))
    info_prof = M - B_pos.T @ (B_pos / D_pos[:, None])

    try:
        cov = np.linalg.inv(info_prof)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info_prof)

    var_diag = np.maximum(np.diag(cov), 0.0)
    se[active_k] = np.sqrt(var_diag)
    return se


def _profile_ci_and_p_single(
    n_bg: np.ndarray,
    L_bg: np.ndarray,
    class_idx: int,
    full_fit: RateModelFit,
    *,
    rate_model: Literal["192r_3w", "192r_syn"],
    ci_level: float,
) -> tuple[float, float, float]:
    """Profile likelihood CI [cilow, cihigh] and LRT p-value (H0: w=1) for class_idx."""
    w_hat = float(full_fit.w_mle[class_idx])
    ll_max = float(full_fit.loglik)
    target_drop = 0.5 * float(stats.chi2.ppf(ci_level, df=1))

    def _prof_ll(w_val: float) -> float:
        fit = fit_192_rate_model(
            n_bg,
            L_bg,
            rate_model=rate_model,
            fixed_idx=class_idx,
            fixed_val=w_val,
            w_init=full_fit.w_mle,
        )
        return fit.loglik

    # p-value for H0: w == 1.0
    ll_null = _prof_ll(1.0)
    lrt_null = max(0.0, 2.0 * (ll_max - ll_null))
    p_val = float(stats.chi2.sf(lrt_null, df=1))

    # Lower bound
    if w_hat <= 1e-12 or float(np.sum(n_bg[:, 1 + class_idx])) == 0.0:
        cilow = 0.0
    else:
        lo_edge = 1e-12
        if (ll_max - _prof_ll(lo_edge)) <= target_drop:
            cilow = 0.0
        else:
            cilow = float(
                optimize.brentq(
                    lambda x: (ll_max - _prof_ll(x)) - target_drop,
                    lo_edge,
                    w_hat,
                    xtol=1e-9,
                    maxiter=80,
                )
            )

    # Upper bound
    hi = max(w_hat * 1.5, 0.25)
    for _ in range(40):
        if (ll_max - _prof_ll(hi)) >= target_drop:
            break
        hi *= 2.0
    if (ll_max - _prof_ll(hi)) < target_drop:
        cihigh = float(np.inf)
    else:
        lo_start = max(w_hat, 1e-12)
        cihigh = float(
            optimize.brentq(
                lambda x: (ll_max - _prof_ll(x)) - target_drop,
                lo_start,
                hi,
                xtol=1e-9,
                maxiter=80,
            )
        )

    return cilow, cihigh, p_val


def _compute_class_ci(
    n_bg: np.ndarray,
    L_bg: np.ndarray,
    fit: RateModelFit,
    class_idx: int,
    se_vec: np.ndarray,
    *,
    rate_model: Literal["192r_3w", "192r_syn"],
    ci_level: float,
    ci_method: Literal["wald", "profile"],
) -> tuple[float, float, float]:
    w_hat = float(fit.w_mle[class_idx])
    n_obs_c = float(np.sum(n_bg[:, 1 + class_idx]))

    if ci_method == "profile" or n_obs_c == 0.0 or not np.isfinite(se_vec[class_idx]):
        return _profile_ci_and_p_single(
            n_bg,
            L_bg,
            class_idx,
            fit,
            rate_model=rate_model,
            ci_level=ci_level,
        )

    z = float(stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    eta = float(np.log(max(w_hat, 1e-300)))
    se_eta = float(se_vec[class_idx])
    cilow = float(np.exp(eta - z * se_eta))
    cihigh = float(np.exp(eta + z * se_eta))
    z_stat = abs(eta) / max(se_eta, 1e-300)
    p_val = float(2.0 * stats.norm.sf(z_stat))
    return cilow, cihigh, p_val


def compute_global_dnds(
    n_obs_bg: np.ndarray,
    L_bg_all: np.ndarray,
    *,
    rate_model: Literal["192r_3w", "192r_syn"] = "192r_3w",
    ci_level: float = 0.95,
    ci_method: Literal["wald", "profile"] = "wald",
) -> GlobalDnDsFit:
    """
    Fit `192r_3w` (`wmis`, `wnon`, `wspl`), `192r_2w` (`wtru`), and `192r_1w` (`wall`)
    with Schur-complement Wald or profile-likelihood CIs.
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")
    if ci_method not in {"wald", "profile"}:
        raise ValueError(f"Invalid ci_method {ci_method!r}; expected 'wald' or 'profile'.")

    n_pool = np.sum(n_obs_bg, axis=0).astype(np.float64)  # (192, 4)
    L_pool = np.sum(L_bg_all, axis=0).astype(np.float64)  # (192, 4)

    names = ["wmis", "wnon", "wspl", "wtru", "wall"]
    if float(np.sum(n_pool)) == 0.0:
        return GlobalDnDsFit(
            names=names,
            mle=np.ones(5, dtype=np.float64),
            cilow=np.zeros(5, dtype=np.float64),
            cihigh=np.full(5, np.inf, dtype=np.float64),
            n_obs=np.zeros(5, dtype=np.float64),
            n_exp=np.zeros(5, dtype=np.float64),
            p_value=np.ones(5, dtype=np.float64),
            r_mle=np.zeros(192, dtype=np.float64),
        )

    # 1. 192r_3w: wmis, wnon, wspl
    fit_3w = fit_192_rate_model(n_pool, L_pool, rate_model=rate_model)
    se_3w = _schur_wald_se_logw(fit_3w.r_mle, fit_3w.w_mle, L_pool)

    # 2. 192r_2w: (mis, tru = non + spl)
    n_2w = np.column_stack([n_pool[:, 0], n_pool[:, 1], n_pool[:, 2] + n_pool[:, 3]])
    L_2w = np.column_stack([L_pool[:, 0], L_pool[:, 1], L_pool[:, 2] + L_pool[:, 3]])
    fit_2w = fit_192_rate_model(n_2w, L_2w, rate_model=rate_model)
    se_2w = _schur_wald_se_logw(fit_2w.r_mle, fit_2w.w_mle, L_2w)

    # 3. 192r_1w: (all = mis + non + spl)
    n_1w = np.column_stack([n_pool[:, 0], n_pool[:, 1] + n_pool[:, 2] + n_pool[:, 3]])
    L_1w = np.column_stack([L_pool[:, 0], L_pool[:, 1] + L_pool[:, 2] + L_pool[:, 3]])
    fit_1w = fit_192_rate_model(n_1w, L_1w, rate_model=rate_model)
    se_1w = _schur_wald_se_logw(fit_1w.r_mle, fit_1w.w_mle, L_1w)

    mles = np.empty(5, dtype=np.float64)
    cilows = np.empty(5, dtype=np.float64)
    cihighs = np.empty(5, dtype=np.float64)
    n_obs_out = np.empty(5, dtype=np.float64)
    n_exp_out = np.empty(5, dtype=np.float64)
    p_vals = np.empty(5, dtype=np.float64)

    for k in range(3):
        mles[k] = fit_3w.w_mle[k]
        n_obs_out[k] = float(np.sum(n_pool[:, 1 + k]))
        n_exp_out[k] = float(fit_3w.r_mle @ L_pool[:, 1 + k])
        cilows[k], cihighs[k], p_vals[k] = _compute_class_ci(
            n_pool, L_pool, fit_3w, k, se_3w, rate_model=rate_model, ci_level=ci_level, ci_method=ci_method
        )

    # wtru (index 1 in 2w)
    mles[3] = fit_2w.w_mle[1]
    n_obs_out[3] = float(np.sum(n_2w[:, 2]))
    n_exp_out[3] = float(fit_2w.r_mle @ L_2w[:, 2])
    cilows[3], cihighs[3], p_vals[3] = _compute_class_ci(
        n_2w, L_2w, fit_2w, 1, se_2w, rate_model=rate_model, ci_level=ci_level, ci_method=ci_method
    )

    # wall (index 0 in 1w non-syn)
    mles[4] = fit_1w.w_mle[0]
    n_obs_out[4] = float(np.sum(n_1w[:, 1]))
    n_exp_out[4] = float(fit_1w.r_mle @ L_1w[:, 1])
    cilows[4], cihighs[4], p_vals[4] = _compute_class_ci(
        n_1w, L_1w, fit_1w, 0, se_1w, rate_model=rate_model, ci_level=ci_level, ci_method=ci_method
    )

    return GlobalDnDsFit(
        names=names,
        mle=mles,
        cilow=cilows,
        cihigh=cihighs,
        n_obs=n_obs_out,
        n_exp=n_exp_out,
        p_value=p_vals,
        r_mle=fit_3w.r_mle,
    )


def _poisson_loglik_vec(y: np.ndarray, mu: np.ndarray, gammaln_y1: np.ndarray) -> float:
    mu_safe = np.maximum(mu, 1e-300)
    term_y = np.where(y > 0, y * np.log(mu_safe), 0.0)
    return float(np.sum(term_y - mu_safe - gammaln_y1))


def _nb2_loglik_vec(
    y: np.ndarray,
    mu: np.ndarray,
    theta: float,
    gammaln_y1: np.ndarray,
) -> float:
    mu_safe = np.maximum(mu, 1e-300)
    t = float(theta)
    y_pos = y > 0
    gamma_diff = np.where(y_pos, special.gammaln(y + t) - special.gammaln(t), 0.0)
    log_p_part = -t * np.log1p(mu_safe / t)
    log_q_part = np.where(y_pos, y * (np.log(mu_safe) - np.log(t + mu_safe)), 0.0)
    return float(np.sum(gamma_diff - gammaln_y1 + log_p_part + log_q_part))


def _fit_irls_given_alpha(
    z_design: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    alpha: float,
    gamma_init: np.ndarray,
    gammaln_y1: np.ndarray,
    *,
    max_inner: int = 40,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Damped Fisher-scoring IRLS for log-link Poisson (alpha=0) or NB2 (alpha>0)."""
    gamma = gamma_init.copy()
    p_dim = z_design.shape[1]
    eye_reg = 1e-11 * np.eye(p_dim, dtype=np.float64)

    def _eval_ll(g_vec: np.ndarray) -> tuple[np.ndarray, float]:
        eta_v = np.clip(offset + z_design @ g_vec, -45.0, 45.0)
        mu_v = np.exp(eta_v)
        if alpha <= 0.0:
            ll_v = _poisson_loglik_vec(y, mu_v, gammaln_y1)
        else:
            ll_v = _nb2_loglik_vec(y, mu_v, 1.0 / alpha, gammaln_y1)
        return mu_v, ll_v

    mu, curr_ll = _eval_ll(gamma)
    converged = False

    for _ in range(max_inner):
        denom = 1.0 + alpha * mu if alpha > 0.0 else np.ones_like(mu)
        grad = z_design.T @ ((y - mu) / denom)
        if float(np.max(np.abs(grad))) <= tol:
            converged = True
            break

        weights = mu / denom
        fisher = (z_design.T * weights) @ z_design + eye_reg
        try:
            step = np.linalg.solve(fisher, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(fisher, grad, rcond=None)[0]

        step_scale = 1.0
        accepted = False
        for _ls in range(25):
            cand = gamma + step_scale * step
            mu_cand, ll_cand = _eval_ll(cand)
            if np.isfinite(ll_cand) and ll_cand >= curr_ll - 1e-11:
                diff_norm = float(np.max(np.abs(cand - gamma)))
                gamma = cand
                mu = mu_cand
                curr_ll = ll_cand
                accepted = True
                if diff_norm <= tol:
                    converged = True
                break
            step_scale *= 0.5

        if not accepted or converged:
            break

    return gamma, mu, curr_ll, converged


def _profile_mle_theta(
    y: np.ndarray,
    mu: np.ndarray,
    gammaln_y1: np.ndarray,
) -> tuple[float, float]:
    """Bounded Brent + Newton-Raphson profile MLE for NB2 dispersion theta."""
    mu_safe = np.maximum(mu, 1e-300)
    opt = optimize.minimize_scalar(
        lambda log_t: -_nb2_loglik_vec(y, mu_safe, float(np.exp(log_t)), gammaln_y1),
        bounds=(_MIN_LOG_THETA, _MAX_LOG_THETA),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    log_t = float(opt.x)
    best_ll = -float(opt.fun)

    y_pos = y > 0
    for _ in range(15):
        if log_t <= _MIN_LOG_THETA + 1e-5 or log_t >= _MAX_LOG_THETA - 1e-5:
            break
        t = float(np.exp(log_t))
        psi_diff = np.where(y_pos, special.digamma(y + t) - special.digamma(t), 0.0)
        d_t = float(np.sum(psi_diff - np.log1p(mu_safe / t) + (mu_safe - y) / (t + mu_safe)))
        g_z = t * d_t
        if abs(g_z) < 1e-9:
            break

        tri_diff = np.where(
            y_pos,
            special.polygamma(1, y + t) - special.polygamma(1, t),
            0.0,
        )
        d2_t = float(
            np.sum(
                tri_diff + 1.0 / t - 2.0 / (t + mu_safe) + (y + t) / ((t + mu_safe) ** 2)
            )
        )
        h_z = (t * t) * d2_t + g_z
        if h_z >= -1e-12:
            break
        step = np.clip(g_z / h_z, -1.0, 1.0)
        cand_z = float(np.clip(log_t - step, _MIN_LOG_THETA, _MAX_LOG_THETA))
        cand_ll = _nb2_loglik_vec(y, mu_safe, float(np.exp(cand_z)), gammaln_y1)
        if cand_ll >= best_ll:
            log_t = cand_z
            best_ll = cand_ll
        else:
            break

    return float(np.exp(log_t)), best_ll


def _theta_ml_newton(
    y: np.ndarray,
    mu: np.ndarray,
    gammaln_y1: np.ndarray,
    *,
    theta_init: float | None = None,
    limit: int = 25,
    eps: float = 1e-6,
) -> tuple[float, float, bool]:
    """
    `MASS::theta.ml` Newton / Fisher-scoring estimator for NB2 size parameter `theta`
    at fixed `mu`, initialized from moment estimate `n / sum((y / mu - 1)^2)`.

    Returns `(theta_hat, loglik, converged)`.
    """
    mu_safe = np.maximum(mu, 1e-300)
    if theta_init is None:
        denom_mom = float(np.sum((y / mu_safe - 1.0) ** 2))
        if not np.isfinite(denom_mom) or denom_mom <= 1e-15:
            return float(np.inf), _poisson_loglik_vec(y, mu_safe, gammaln_y1), False
        t = float(np.clip(float(y.size) / denom_mom, _R_THETA_MIN_FLOOR, _R_THETA_POISSON_CEIL))
    else:
        t = float(theta_init)

    if not np.isfinite(t) or t <= 0.0:
        return float(np.inf), _poisson_loglik_vec(y, mu_safe, gammaln_y1), False

    y_pos = y > 0
    converged = False
    for _ in range(max(1, limit)):
        if t >= _R_THETA_POISSON_CEIL:
            return float(np.inf), _poisson_loglik_vec(y, mu_safe, gammaln_y1), False
        if t < _R_THETA_MIN_FLOOR:
            return t, _nb2_loglik_vec(y, mu_safe, max(t, 1e-8), gammaln_y1), False

        psi_diff = np.where(y_pos, special.digamma(y + t) - special.digamma(t), 0.0)
        score = float(np.sum(psi_diff - np.log1p(mu_safe / t) + (mu_safe - y) / (t + mu_safe)))
        tri_diff = np.where(
            y_pos,
            special.polygamma(1, y + t) - special.polygamma(1, t),
            0.0,
        )
        info = float(
            np.sum(-tri_diff - 1.0 / t + 2.0 / (t + mu_safe) - (y + t) / ((t + mu_safe) ** 2))
        )

        if not np.isfinite(score) or not np.isfinite(info):
            return float(np.inf), _poisson_loglik_vec(y, mu_safe, gammaln_y1), False

        if info <= 1e-14:
            if score >= 0.0:
                del_t = 0.5 * t
            else:
                del_t = -0.5 * t
        else:
            del_t = score / info
            while t + del_t <= 0.1 * t:
                del_t *= 0.5
            del_t = min(del_t, 5.0 * t)

        t_next = t + del_t
        if abs(del_t) <= eps * (1.0 + t) and abs(score) <= 1e-4 * max(1.0, np.sqrt(float(y.size))):
            t = t_next
            converged = True
            break
        t = t_next

    if t >= _R_THETA_POISSON_CEIL or t < _R_THETA_MIN_FLOOR:
        return float(np.inf), _poisson_loglik_vec(y, mu_safe, gammaln_y1), False
    return float(t), _nb2_loglik_vec(y, mu_safe, t, gammaln_y1), converged


def fit_nb_covariate_model(
    y_all: np.ndarray,
    exposure_all: np.ndarray,
    covariates: pd.DataFrame | np.ndarray | None,
    bg_mask: np.ndarray,
    *,
    use_nb: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    settings: SettingsMode = "default",
) -> NBCovariateFit:
    """
    Fit Gamma-Poisson (NB2) regression on active background genes with log(exposure)
    offset and optional covariates, falling back cleanly to Poisson when alpha -> 0+.
    """
    eff_mode = validate_settings(settings)
    y_vec = np.asarray(y_all, dtype=np.float64)
    exp_vec = np.asarray(exposure_all, dtype=np.float64)
    n_total = y_vec.size

    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cov_names = [str(c) for c in covariates.columns]
            x_raw = covariates.to_numpy(dtype=np.float64, copy=True)
        else:
            x_arr = np.asarray(covariates, dtype=np.float64)
            if x_arr.ndim == 1:
                x_raw = x_arr[:, None].copy()
            elif x_arr.ndim == 2:
                x_raw = x_arr.copy()
            else:
                raise ValueError(f"covariates must be 1D or 2D, got {x_arr.ndim}D.")
            cov_names = [f"X{j}" for j in range(x_raw.shape[1])]
        if x_raw.shape[0] != n_total:
            raise ValueError(
                f"covariates has {x_raw.shape[0]} rows, expected {n_total} genes."
            )
        if not np.all(np.isfinite(x_raw)):
            raise ValueError("covariates contain NaN or infinite values.")
    else:
        x_raw = np.empty((n_total, 0), dtype=np.float64)
        cov_names = []

    n_cov = x_raw.shape[1]
    coef_names = ["(Intercept)", *cov_names]

    active_fit_mask = np.asarray(bg_mask, dtype=bool) & (exp_vec > 0.0)
    n_act = int(np.sum(active_fit_mask))
    if n_act == 0:
        raise ValueError("No background genes have positive exposure for regression.")
    if n_act <= n_cov:
        raise ValueError(
            f"Number of active background genes ({n_act}) must exceed number of covariates ({n_cov})."
        )

    y_act = y_vec[active_fit_mask]
    e_act = exp_vec[active_fit_mask]
    offset_act = np.log(e_act)
    x_act = x_raw[active_fit_mask]

    if n_cov > 0:
        x_means = np.mean(x_act, axis=0)
        x_stds = np.std(x_act, axis=0)
        if np.any(x_stds <= 1e-15):
            raise ValueError("Covariate matrix contains a constant (zero-variance) column on active genes.")
        x_std = (x_act - x_means) / x_stds
        z_design = np.column_stack([np.ones(n_act, dtype=np.float64), x_std])
        if np.linalg.matrix_rank(z_design) < n_cov + 1:
            raise ValueError("Covariate design matrix is rank-deficient (collinear columns).")
    else:
        x_means = np.empty(0, dtype=np.float64)
        x_stds = np.empty(0, dtype=np.float64)
        z_design = np.ones((n_act, 1), dtype=np.float64)

    if float(np.sum(y_act)) == 0.0:
        beta = np.zeros(n_cov + 1, dtype=np.float64)
        beta[0] = -np.inf
        return NBCovariateFit(
            coef_names=coef_names,
            beta=beta,
            se=np.full(n_cov + 1, np.nan, dtype=np.float64),
            p_beta=np.full(n_cov + 1, np.nan, dtype=np.float64),
            theta=float(np.inf),
            alpha=0.0,
            is_poisson=True,
            rho_all=np.zeros(n_total, dtype=np.float64),
            mu_all=np.zeros(n_total, dtype=np.float64),
            loglik=0.0,
            converged=True,
        )

    gammaln_y1 = special.gammaln(y_act + 1.0)
    gamma0 = np.zeros(n_cov + 1, dtype=np.float64)
    gamma0[0] = float(np.log(np.sum(y_act) / np.sum(e_act)))

    # 1. Fit exact Poisson GLM (alpha = 0)
    irls_pois_max = 25 if eff_mode == "R_default" else 60
    gamma_pois, mu_pois, ll_pois, conv_pois = _fit_irls_given_alpha(
        z_design, offset_act, y_act, 0.0, gamma0, gammaln_y1, max_inner=irls_pois_max, tol=tol
    )

    if not use_nb:
        gamma_hat = gamma_pois
        mu_hat = mu_pois
        alpha_hat = 0.0
        theta_hat = float(np.inf)
        is_poisson = True
        final_ll = ll_pois
        converged = conv_pois
    elif eff_mode == "R_default":
        # MASS::glm.nb (maxit=25) alternating IRLS + MASS::theta.ml Newton updates
        # with dndscv.R strict theta convergence fallback to Poisson GLM
        alpha_score_zero = 0.5 * float(np.sum((y_act - mu_pois) ** 2 - y_act))
        theta_init, ll_nb_init, th_init_conv = _theta_ml_newton(
            y_act, mu_pois, gammaln_y1, limit=max_iter
        )

        if (
            alpha_score_zero <= 0.0
            or not th_init_conv
            or not np.isfinite(theta_init)
            or theta_init >= _R_THETA_POISSON_CEIL
            or theta_init < _R_THETA_MIN_FLOOR
            or ll_nb_init <= ll_pois + 1e-7
        ):
            gamma_hat = gamma_pois
            mu_hat = mu_pois
            alpha_hat = 0.0
            theta_hat = float(np.inf)
            is_poisson = True
            final_ll = ll_pois
            converged = conv_pois
        else:
            gamma_curr = gamma_pois.copy()
            theta_curr = theta_init
            dev_prev = -2.0 * ll_nb_init
            nb_converged = False

            for _ in range(max_iter):
                alpha_curr = 1.0 / theta_curr
                gamma_next, mu_next, _, conv_b = _fit_irls_given_alpha(
                    z_design,
                    offset_act,
                    y_act,
                    alpha_curr,
                    gamma_curr,
                    gammaln_y1,
                    max_inner=25,
                    tol=min(tol, 1e-8),
                )
                theta_next, ll_next, th_ok = _theta_ml_newton(
                    y_act,
                    mu_next,
                    gammaln_y1,
                    theta_init=theta_curr,
                    limit=max_iter,
                )
                if (
                    not th_ok
                    or not np.isfinite(theta_next)
                    or theta_next >= _R_THETA_POISSON_CEIL
                    or theta_next < _R_THETA_MIN_FLOOR
                ):
                    nb_converged = False
                    break

                dev_curr = -2.0 * ll_next
                rel_dev = abs(dev_prev - dev_curr) / (0.1 + abs(dev_curr))
                rel_th = abs(theta_next - theta_curr) / (0.1 + abs(theta_curr))
                gamma_curr = gamma_next
                theta_curr = theta_next
                dev_prev = dev_curr

                if conv_b and rel_dev <= max(tol, 1e-8) and rel_th <= max(tol, 1e-6):
                    nb_converged = True
                    break

            eta_cand = np.clip(offset_act + z_design @ gamma_curr, -45.0, 45.0)
            mu_cand = np.exp(eta_cand)
            ll_cand = (
                _nb2_loglik_vec(y_act, mu_cand, theta_curr, gammaln_y1)
                if np.isfinite(theta_curr) and theta_curr > 0.0
                else -np.inf
            )

            # Strict theta convergence fallback (dndscv.R tryCatch fallback to Poisson GLM)
            if (
                not nb_converged
                or not np.isfinite(theta_curr)
                or theta_curr >= _R_THETA_POISSON_CEIL
                or theta_curr < _R_THETA_MIN_FLOOR
                or ll_pois >= ll_cand - 1e-7
            ):
                gamma_hat = gamma_pois
                mu_hat = mu_pois
                alpha_hat = 0.0
                theta_hat = float(np.inf)
                is_poisson = True
                final_ll = ll_pois
                converged = conv_pois
            else:
                gamma_hat = gamma_curr
                mu_hat = mu_cand
                theta_hat = theta_curr
                alpha_hat = 1.0 / theta_hat
                is_poisson = False
                final_ll = ll_cand
                converged = True
    else:
        alpha_score_zero = 0.5 * float(np.sum((y_act - mu_pois) ** 2 - y_act))
        theta_init, ll_nb_init = _profile_mle_theta(y_act, mu_pois, gammaln_y1)

        if (
            alpha_score_zero <= 0.0 and ll_nb_init <= ll_pois + 1e-7
        ) or theta_init >= np.exp(_MAX_LOG_THETA - 1e-3):
            gamma_hat = gamma_pois
            mu_hat = mu_pois
            alpha_hat = 0.0
            theta_hat = float(np.inf)
            is_poisson = True
            final_ll = ll_pois
            converged = conv_pois
        else:
            gamma_curr = gamma_pois.copy()
            theta_curr = theta_init
            prev_ll = ll_nb_init
            converged = False

            for _ in range(max_iter):
                alpha_curr = 1.0 / theta_curr
                gamma_next, mu_next, _, conv_b = _fit_irls_given_alpha(
                    z_design,
                    offset_act,
                    y_act,
                    alpha_curr,
                    gamma_curr,
                    gammaln_y1,
                    max_inner=35,
                    tol=min(tol, 1e-9),
                )
                theta_next, ll_next = _profile_mle_theta(y_act, mu_next, gammaln_y1)

                if theta_next >= np.exp(_MAX_LOG_THETA - 1e-3) and ll_pois >= ll_next - 1e-7:
                    gamma_curr = gamma_pois
                    theta_curr = float(np.inf)
                    prev_ll = ll_pois
                    converged = conv_pois
                    break

                beta_diff = float(np.max(np.abs(gamma_next - gamma_curr)))
                ll_diff = abs(ll_next - prev_ll)
                gamma_curr = gamma_next
                theta_curr = theta_next
                prev_ll = ll_next

                if conv_b and beta_diff <= tol and ll_diff <= tol * (1.0 + abs(prev_ll)):
                    converged = True
                    break

            if not np.isfinite(theta_curr) or ll_pois >= prev_ll - 1e-8:
                gamma_hat = gamma_pois
                mu_hat = mu_pois
                alpha_hat = 0.0
                theta_hat = float(np.inf)
                is_poisson = True
                final_ll = ll_pois
            else:
                gamma_hat = gamma_curr
                eta_hat = np.clip(offset_act + z_design @ gamma_hat, -45.0, 45.0)
                mu_hat = np.exp(eta_hat)
                theta_hat = theta_curr
                alpha_hat = 1.0 / theta_hat
                is_poisson = False
                final_ll = _nb2_loglik_vec(y_act, mu_hat, theta_hat, gammaln_y1)

    # Back-transform coefficients and Fisher covariance to original X scale
    weights_final = mu_hat if is_poisson else (mu_hat / (1.0 + alpha_hat * mu_hat))
    fisher_z = (z_design.T * weights_final) @ z_design
    try:
        cov_z = np.linalg.inv(fisher_z)
    except np.linalg.LinAlgError:
        cov_z = np.linalg.pinv(fisher_z)

    t_jac = np.eye(n_cov + 1, dtype=np.float64)
    if n_cov > 0:
        t_jac[0, 1:] = -x_means / x_stds
        for j in range(n_cov):
            t_jac[j + 1, j + 1] = 1.0 / x_stds[j]

    beta_orig = t_jac @ gamma_hat
    cov_orig = t_jac @ cov_z @ t_jac.T
    se_orig = np.sqrt(np.maximum(np.diag(cov_orig), 0.0))
    z_orig = np.where(se_orig > 0.0, beta_orig / np.maximum(se_orig, 1e-300), np.nan)
    p_orig = 2.0 * stats.norm.sf(np.abs(z_orig))

    # Predict for ALL genes
    if n_cov > 0:
        eta_all = np.clip(beta_orig[0] + x_raw @ beta_orig[1:], -45.0, 45.0)
    else:
        eta_all = np.full(n_total, float(np.clip(beta_orig[0], -45.0, 45.0)), dtype=np.float64)
    rho_all = np.exp(eta_all)
    mu_all = np.where(exp_vec > 0.0, exp_vec * rho_all, 0.0)

    return NBCovariateFit(
        coef_names=coef_names,
        beta=beta_orig,
        se=se_orig,
        p_beta=p_orig,
        theta=float(theta_hat),
        alpha=float(alpha_hat),
        is_poisson=is_poisson,
        rho_all=rho_all,
        mu_all=mu_all,
        loglik=float(final_ll),
        converged=converged,
    )


def _n_log_ratio(n: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute n * log(n / denom) with 0 * log(0) = 0."""
    pos = n > 0.0
    out = np.zeros_like(n, dtype=np.float64)
    out[pos] = n[pos] * (np.log(n[pos]) - np.log(np.maximum(denom[pos], 1e-300)))
    return out


def compute_gene_tests_cv(
    n_obs_classes: np.ndarray,  # (G, 4) float64: syn, mis, non, spl
    exp_classes: np.ndarray,  # (G, 4) float64: E_syn, E_mis, E_non, E_spl
    rho_syn: np.ndarray,  # (G,)
    theta_syn: float,
    is_poisson_syn: bool,
) -> dict[str, np.ndarray]:
    """
    Compute closed-form Gamma-Poisson (`w_cv`, `p_*_cv`, `q_*_cv`) and pure Poisson
    (`w_*_p`, `p_*_p`, `q_all_p`) gene-level MLEs and likelihood-ratio tests.
    """
    G = n_obs_classes.shape[0]
    n_syn = n_obs_classes[:, 0]
    E_syn = exp_classes[:, 0]
    has_syn = E_syn > 0.0

    # Conditional posterior background rate multiplier and w_cv MLEs
    w_cv = np.full((G, 3), np.nan, dtype=np.float64)
    t_post = np.full(G, np.nan, dtype=np.float64)
    if is_poisson_syn or not np.isfinite(theta_syn):
        t_post[has_syn] = rho_syn[has_syn]
    else:
        t_post[has_syn] = (theta_syn + n_syn[has_syn]) / (
            theta_syn / np.maximum(rho_syn[has_syn], 1e-300) + E_syn[has_syn]
        )

    for c_idx in range(3):
        E_c = exp_classes[:, 1 + c_idx]
        n_c = n_obs_classes[:, 1 + c_idx]
        valid_c = has_syn & (E_c > 0.0) & (t_post > 0.0)
        w_cv[valid_c, c_idx] = n_c[valid_c] / (t_post[valid_c] * E_c[valid_c])

    # Pure Poisson dNdSloc point estimates
    w_p = np.full((G, 3), np.nan, dtype=np.float64)
    for c_idx in range(3):
        E_c = exp_classes[:, 1 + c_idx]
        n_c = n_obs_classes[:, 1 + c_idx]
        valid_e = has_syn & (E_c > 0.0)
        pos_syn = valid_e & (n_syn > 0.0)
        w_p[pos_syn, c_idx] = (n_c[pos_syn] / E_c[pos_syn]) / (n_syn[pos_syn] / E_syn[pos_syn])
        inf_case = valid_e & (n_syn == 0.0) & (n_c > 0.0)
        w_p[inf_case, c_idx] = np.inf

    # Hypothesis subsets (indices into columns 1,2,3 -> mis=0, non=1, spl=2)
    hypotheses = {
        "mis": (0,),
        "non": (1,),
        "spl": (2,),
        "tru": (1, 2),
        "all": (0, 1, 2),
    }

    out: dict[str, np.ndarray] = {
        "w_mis_cv": w_cv[:, 0],
        "w_non_cv": w_cv[:, 1],
        "w_spl_cv": w_cv[:, 2],
        "w_mis_p": w_p[:, 0],
        "w_non_p": w_p[:, 1],
        "w_spl_p": w_p[:, 2],
    }

    for hname, cols in hypotheses.items():
        p_cv = np.full(G, np.nan, dtype=np.float64)
        p_p = np.full(G, np.nan, dtype=np.float64)

        # Effective df per gene depends on which constrained classes have E_c > 0
        E_sub = exp_classes[:, [1 + c for c in cols]]  # (G, len(cols))
        n_sub = n_obs_classes[:, [1 + c for c in cols]]
        avail = E_sub > 0.0
        df_vec = np.sum(avail, axis=1)
        valid_h = has_syn & (df_vec > 0)

        if np.any(valid_h):
            # --- CV LRT ---
            n_sum_c = np.sum(np.where(avail, n_sub, 0.0), axis=1)
            E_sum_c = np.sum(np.where(avail, E_sub, 0.0), axis=1)
            rho_v = np.maximum(rho_syn[valid_h], 1e-300)
            n_syn_v = n_syn[valid_h]
            E_syn_v = E_syn[valid_h]
            n_sub_v = n_sub[valid_h]
            E_sub_v = E_sub[valid_h]
            avail_v = avail[valid_h]
            n_sum_v = n_sum_c[valid_h]
            E_sum_v = E_sum_c[valid_h]

            # Expected neutral counts under covariate model
            mu_sub_v = rho_v[:, None] * E_sub_v
            term_classes = np.zeros(n_syn_v.size, dtype=np.float64)
            for j in range(len(cols)):
                m_j = avail_v[:, j]
                if np.any(m_j):
                    term_classes[m_j] += _n_log_ratio(n_sub_v[m_j, j], mu_sub_v[m_j, j])

            if is_poisson_syn or not np.isfinite(theta_syn):
                # Exact theta -> inf limit
                diff_lin = np.sum(np.where(avail_v, n_sub_v - mu_sub_v, 0.0), axis=1)
                lam_cv = 2.0 * (term_classes - diff_lin)
            else:
                t = float(theta_syn)
                N_C = n_syn_v + n_sum_v
                E_C = E_syn_v + E_sum_v
                term_syn = (t + n_syn_v) * (np.log1p(n_syn_v / t) - np.log1p(rho_v * E_syn_v / t))
                term_comb = (t + N_C) * (np.log1p(N_C / t) - np.log1p(rho_v * E_C / t))
                lam_cv = 2.0 * (term_classes + term_syn - term_comb)

            lam_cv = np.maximum(lam_cv, 0.0)
            p_cv[valid_h] = stats.chi2.sf(lam_cv, df=df_vec[valid_h])

            # --- Pure Poisson Contingency LRT ---
            term_p_classes = np.zeros(n_syn_v.size, dtype=np.float64)
            for j in range(len(cols)):
                m_j = avail_v[:, j]
                if np.any(m_j):
                    term_p_classes[m_j] += _n_log_ratio(n_sub_v[m_j, j], E_sub_v[m_j, j])
            N_C = n_syn_v + n_sum_v
            E_C = E_syn_v + E_sum_v
            lam_p = 2.0 * (
                _n_log_ratio(n_syn_v, E_syn_v) + term_p_classes - _n_log_ratio(N_C, E_C)
            )
            lam_p = np.maximum(lam_p, 0.0)
            p_p[valid_h] = stats.chi2.sf(lam_p, df=df_vec[valid_h])

        out[f"p_{hname}_cv"] = p_cv
        out[f"p_{hname}_p"] = p_p
        out[f"q_{hname}_cv"] = bh_fdr_nan_aware(p_cv)

    out["q_all_p"] = bh_fdr_nan_aware(out["p_all_p"])
    return out


def compute_indel_tests(
    n_ind: np.ndarray,
    cds_lengths: np.ndarray,
    covariates: pd.DataFrame | np.ndarray | None,
    bg_mask: np.ndarray,
    *,
    use_nb: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    settings: SettingsMode = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NBCovariateFit | None]:
    """
    Fit NB2/Poisson covariate model for indels and compute gene-level indel LRTs.
    Returns `(w_ind_cv, p_ind_cv, q_ind_cv, mu_ind, nb_fit)`.
    """
    eff_mode = validate_settings(settings)
    n_ind_f = np.asarray(n_ind, dtype=np.float64)
    lens_f = np.asarray(cds_lengths, dtype=np.float64)
    G = n_ind_f.size
    has_len = lens_f > 0.0

    if float(np.sum(n_ind_f)) == 0.0:
        w_ind = np.where(has_len, 0.0, np.nan)
        p_ind = np.where(has_len, 1.0, np.nan)
        q_ind = bh_fdr_nan_aware(p_ind)
        return w_ind, p_ind, q_ind, np.zeros(G, dtype=np.float64), None

    # If background genes have 0 indels (e.g. only known cancer genes have indels), fit on all genes with cds_length > 0
    fit_mask = np.asarray(bg_mask, dtype=bool)
    if float(np.sum(n_ind_f[fit_mask & has_len])) == 0.0:
        fit_mask = has_len

    nb_fit = fit_nb_covariate_model(
        n_ind_f,
        lens_f,
        covariates,
        fit_mask,
        use_nb=use_nb,
        max_iter=max_iter,
        tol=tol,
        settings=eff_mode,
    )
    mu_ind = nb_fit.mu_all
    w_ind = np.full(G, np.nan, dtype=np.float64)
    p_ind = np.full(G, np.nan, dtype=np.float64)

    valid = has_len & (mu_ind > 0.0)
    if np.any(valid):
        y_v = n_ind_f[valid]
        mu_v = mu_ind[valid]
        w_ind[valid] = y_v / mu_v
        if nb_fit.is_poisson or not np.isfinite(nb_fit.theta):
            lam = 2.0 * (_n_log_ratio(y_v, mu_v) - (y_v - mu_v))
        else:
            t = float(nb_fit.theta)
            lam = 2.0 * (
                _n_log_ratio(y_v, mu_v)
                + (t + y_v) * (np.log1p(mu_v / t) - np.log1p(y_v / t))
            )
        lam = np.maximum(lam, 0.0)
        p_ind[valid] = stats.chi2.sf(lam, df=1)

    q_ind = bh_fdr_nan_aware(p_ind)
    return w_ind, p_ind, q_ind, mu_ind, nb_fit
