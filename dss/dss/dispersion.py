"""Log-normal Empirical Bayes Beta-Binomial dispersion estimation for DSS."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .smoothing import SettingsMode, validate_settings

_PHI_MIN = 1e-5
_PHI_MAX = 0.99
_N_GRID = 45
_THETA_GRID = np.linspace(np.log(_PHI_MIN), np.log(_PHI_MAX), _N_GRID)
_PHI_GRID = np.exp(_THETA_GRID)
_M_GRID = (1.0 / _PHI_GRID) - 1.0
_D_THETA = float(_THETA_GRID[1] - _THETA_GRID[0])


def _parabolic_refine(vals: np.ndarray, theta_grid: np.ndarray, step: float) -> np.ndarray:
    """Refine discrete grid maxima via 3-point quadratic interpolation."""
    idx = np.argmax(vals, axis=1)
    theta_hat = theta_grid[idx].copy()
    interior = (idx > 0) & (idx < (vals.shape[1] - 1))
    if np.any(interior):
        rows = np.flatnonzero(interior)
        q = idx[rows]
        f_m = vals[rows, q - 1]
        f_0 = vals[rows, q]
        f_p = vals[rows, q + 1]
        denom = f_m - 2.0 * f_0 + f_p
        ok = denom < -1e-12
        delta = np.zeros_like(denom)
        delta[ok] = np.clip(0.5 * (f_m[ok] - f_p[ok]) / denom[ok], -0.5, 0.5)
        theta_hat[rows] = theta_grid[q] + delta * step
    return theta_hat


def _accum_bb_loglik(
    x_mat: np.ndarray, n_mat: np.ndarray, mu_vec: np.ndarray
) -> np.ndarray:
    """Evaluate Beta-Binomial log-likelihood across ``_PHI_GRID`` for all CpGs."""
    n_cpgs, n_reps = x_mat.shape
    ll = np.zeros((n_cpgs, _N_GRID), dtype=np.float64)
    if n_cpgs == 0 or n_reps == 0:
        return ll

    mu_safe = np.clip(np.asarray(mu_vec, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    alpha_iq = mu_safe[:, None] * _M_GRID[None, :]
    beta_iq = (1.0 - mu_safe[:, None]) * _M_GRID[None, :]
    lg_m = gammaln(_M_GRID)[None, :]
    lg_a = gammaln(alpha_iq)
    lg_b = gammaln(beta_iq)

    for r in range(n_reps):
        xr = x_mat[:, r : r + 1]
        nr = n_mat[:, r : r + 1]
        active = (nr[:, 0] > 0.0)[:, None]
        nr_safe = np.where(active, nr, 1.0)
        xr_safe = np.clip(np.where(active, xr, 0.0), 0.0, nr_safe)
        term = (
            lg_m
            - gammaln(nr_safe + _M_GRID[None, :])
            + gammaln(xr_safe + alpha_iq)
            - lg_a
            + gammaln((nr_safe - xr_safe) + beta_iq)
            - lg_b
        )
        ll += np.where(active, term, 0.0)
    return ll


def estimate_prior_hyperparams(
    theta_mle: np.ndarray, boundary_idx: np.ndarray
) -> tuple[float, float]:
    """Estimate log-normal prior ``ln(phi) ~ N(m0, tau0^2)`` across interior CpGs."""
    interior = np.isfinite(theta_mle) & (boundary_idx > 0) & (boundary_idx < (_N_GRID - 1))
    vals = theta_mle[interior]
    if len(vals) < 5:
        # Fallback when nearly all CpGs show purely binomial variance
        finite_vals = theta_mle[np.isfinite(theta_mle)]
        if len(finite_vals) == 0:
            return float(np.log(0.01)), 1.0
        m0 = float(np.clip(np.median(finite_vals), np.log(1e-4), np.log(0.5)))
        return m0, 1.0

    m0 = float(np.median(vals))
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    iqr_sd = float((q75 - q25) / 1.3489795)
    tau0 = float(np.clip(iqr_sd, 0.25, 2.5))
    return m0, tau0


def shrink_dispersion_bb(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    settings: SettingsMode = "default",
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Estimate profile MLE and Empirical Bayes MAP Beta-Binomial dispersions.

    Parameters
    ----------
    groups
        List of ``(x_mat, n_mat, mu_vec)`` tuples summed into the joint
        Beta-Binomial log-likelihood (length 1 for group-specific dispersion;
        length 2 when ``equal_disp=True``).
    settings
        ``"default"`` or ``"R_default"``.

    Returns
    -------
    phi_post : (n_cpgs,)
        MAP posterior shrunken dispersion clamped to ``[1e-5, 0.99]``.
    phi_mle : (n_cpgs,)
        Unshrunken profile MLE dispersion clamped to ``[1e-5, 0.99]``.
    prior_params : (m0, tau0)
        Estimated log-normal prior mean and standard deviation on ``ln(phi)``.
    """
    validate_settings(settings)
    if not groups or len(groups[0][2]) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, (float(np.log(0.01)), 1.0)

    n_cpgs = len(groups[0][2])
    ll_total = np.zeros((n_cpgs, _N_GRID), dtype=np.float64)
    eff_reps = np.zeros(n_cpgs, dtype=np.int64)
    for x_g, n_g, mu_g in groups:
        ll_total += _accum_bb_loglik(x_g, n_g, mu_g)
        eff_reps += np.sum(n_g > 0.0, axis=1)

    grid_idx = np.argmax(ll_total, axis=1)
    theta_mle = _parabolic_refine(ll_total, _THETA_GRID, _D_THETA)
    # Mask CpGs with zero informative observations
    theta_mle = np.where(eff_reps > 0, theta_mle, np.nan)
    m0, tau0 = estimate_prior_hyperparams(theta_mle, grid_idx)

    prior_pen = -0.5 * ((_THETA_GRID - m0) / tau0) ** 2
    post_grid = ll_total + prior_pen[None, :]
    theta_map = _parabolic_refine(post_grid, _THETA_GRID, _D_THETA)
    theta_map = np.where(eff_reps > 0, theta_map, m0)

    phi_mle = np.clip(np.exp(np.where(np.isfinite(theta_mle), theta_mle, m0)), _PHI_MIN, _PHI_MAX)
    phi_post = np.clip(np.exp(theta_map), _PHI_MIN, _PHI_MAX)
    return phi_post, phi_mle, (m0, tau0)
