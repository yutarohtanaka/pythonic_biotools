"""
Pareto-Smoothed Importance Sampling Leave-One-Out Cross-Validation (PSIS-LOO).

References:
  - Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC. Statistics and
    Computing, 27(5), 1413-1432.
  - Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
    Pareto smoothed importance sampling. JMLR, 25(72), 1-58.
  - Zhang, J., & Stephens, M. A. (2009). A new and efficient estimation method
    for the generalized Pareto distribution. Technometrics, 51(3), 316-325.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats


@dataclass(frozen=True)
class PSISLOOResult:
    """Result container for PSIS-LOO predictive accuracy diagnostics."""

    elpd: float
    se_elpd: float
    p_loo: float
    looic: float
    pointwise_elpd: np.ndarray
    pointwise_p_loo: np.ndarray
    pareto_k: np.ndarray

    @property
    def n_high_pareto_k(self) -> int:
        """Number of observations with diagnostic Pareto shape k > 0.7."""
        return int(np.sum(self.pareto_k > 0.7))

    @property
    def max_pareto_k(self) -> float:
        """Maximum diagnostic Pareto shape k across observations."""
        return float(np.max(self.pareto_k)) if self.pareto_k.size > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-observation ELPD, effective parameters, and Pareto-k table."""
        return pd.DataFrame(
            {
                "elpd": self.pointwise_elpd,
                "p_loo": self.pointwise_p_loo,
                "pareto_k": self.pareto_k,
            }
        )

    def summary(self) -> str:
        """Return formatted PSIS-LOO diagnostic summary."""
        k = self.pareto_k
        n_good = int(np.sum(k <= 0.5))
        n_ok = int(np.sum((k > 0.5) & (k <= 0.7)))
        n_bad = int(np.sum(k > 0.7))
        n_total = int(k.size)
        lines = [
            "PSIS-LOO Summary",
            "=" * 50,
            f"ELPD_loo: {self.elpd:.4f} (SE = {self.se_elpd:.4f})",
            f"p_loo:    {self.p_loo:.4f}",
            f"LOOIC:    {self.looic:.4f}",
            "Pareto k diagnostic counts:",
            f"  good (<= 0.5):       {n_good} / {n_total}",
            f"  ok   (0.5, 0.7]:     {n_ok} / {n_total}",
            f"  bad  (> 0.7):        {n_bad} / {n_total}",
        ]
        return "\n".join(lines)


def gpdfit(
    x: np.ndarray,
    prior: float = 3.0,
    min_grid_pts: int = 30,
    wip: bool = True,
) -> tuple[float, float]:
    """Empirical-Bayes Generalized Pareto Distribution fit (Zhang & Stephens 2009).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-negative tail excesses.
    prior : float, default 3.0
        Prior scale hyperparameter in Zhang & Stephens grid construction.
    min_grid_pts : int, default 30
        Minimum grid size (`G = min_grid_pts + floor(sqrt(N))`).
    wip : bool, default True
        Whether to apply Vehtari et al. weakly informative Beta/normal shrinkage
        pulling shape `k` toward `0.5` for small tail sizes.

    Returns
    -------
    tuple of (float, float)
        Estimated `(k, sigma)` shape and scale parameters.
    """
    x_arr = np.sort(np.asarray(x, dtype=np.float64))
    n = x_arr.size
    if n == 0:
        return 0.0, 0.0
    if np.ptp(x_arr) <= 1e-12 or int(np.sum(x_arr > 0.0)) < 5:
        return 0.0, float(np.mean(x_arr))

    g_gp = int(min_grid_pts + np.floor(np.sqrt(n)))
    x_max = float(x_arr[-1])
    if x_max <= 0.0:
        return 0.0, 0.0

    # Scale excesses by x_max (GPD is a scale family; k is scale-invariant)
    x_scaled = x_arr / x_max
    idx_q1 = max(0, int(np.floor(n / 4.0 + 0.5)) - 1)
    x_star = max(float(x_scaled[idx_q1]), 1e-50)
    x_mean_scaled = max(float(np.mean(x_scaled)), 1e-300)

    ii = np.arange(1, g_gp + 1, dtype=np.float64)
    theta_grid = 1.0 + (1.0 - np.sqrt(g_gp / (ii - 0.5))) / (prior * x_star)

    # Keep strictly feasible theta where 1 - theta * x_m > 0 for all m
    valid = np.min(1.0 - theta_grid[:, None] * x_scaled[None, :], axis=1) > 0.0
    theta_grid = theta_grid[valid]
    if theta_grid.size == 0:
        return 0.0, float(x_mean_scaled * x_max)

    k_grid = np.mean(np.log1p(-theta_grid[:, None] * x_scaled[None, :]), axis=1)
    ell = np.empty_like(k_grid)
    near_zero = (np.abs(k_grid) < 1e-12) | (np.abs(theta_grid) < 1e-15)
    ell[near_zero] = -n * (np.log(x_mean_scaled) + 1.0)
    nz = ~near_zero
    if np.any(nz):
        ratio = np.maximum(-theta_grid[nz] / k_grid[nz], 1e-300)
        ell[nz] = n * (np.log(ratio) - k_grid[nz] - 1.0)

    w = np.exp(ell - np.max(ell))
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return 0.0, float(x_mean_scaled * x_max)
    w /= w_sum

    theta_hat = float(np.sum(w * theta_grid))
    if not np.isfinite(theta_hat) or 1.0 - theta_hat <= 0.0:
        return 0.0, float(x_mean_scaled * x_max)

    k_hat = float(np.mean(np.log1p(-theta_hat * x_scaled)))
    if abs(theta_hat) > 1e-15:
        sigma_scaled = float(-k_hat / theta_hat)
    else:
        sigma_scaled = float(x_mean_scaled)

    if not np.isfinite(sigma_scaled) or sigma_scaled <= 0.0:
        sigma_scaled = float(x_mean_scaled)

    if wip:
        k_hat = float((n * k_hat + 10.0 * 0.5) / (n + 10.0))

    return float(k_hat), float(sigma_scaled * x_max)


def psis_loo(log_lik: np.ndarray) -> PSISLOOResult:
    """Compute PSIS-LOO ELPD, effective parameters `p_loo`, and Pareto-k values.

    Parameters
    ----------
    log_lik : np.ndarray
        2D array of shape `(S, J)` containing pointwise log-likelihood values
        for `S` posterior draws across `J` observations (SNPs).

    Returns
    -------
    PSISLOOResult
    """
    ll_mat = np.asarray(log_lik, dtype=np.float64)
    if ll_mat.ndim != 2:
        raise ValueError(f"log_lik must be a 2D array of shape (S, J) (got shape {ll_mat.shape}).")
    n_draws, n_obs = ll_mat.shape
    if n_draws < 1 or n_obs < 1:
        raise ValueError(f"log_lik must have S >= 1 and J >= 1 (got shape {ll_mat.shape}).")
    if not np.all(np.isfinite(ll_mat)):
        raise ValueError("log_lik must contain only finite values.")

    tail_len = min(n_draws // 5, int(np.floor(3.0 * np.sqrt(n_draws))))

    pointwise_elpd = np.empty(n_obs, dtype=np.float64)
    pointwise_p_loo = np.empty(n_obs, dtype=np.float64)
    pareto_k = np.empty(n_obs, dtype=np.float64)
    log_n_draws = np.log(float(n_draws))

    if tail_len >= 5:
        p_m = (np.arange(1, tail_len + 1, dtype=np.float64) - 0.5) / float(tail_len)
        neg_log_1m_p = -np.log1p(-p_m)
    else:
        neg_log_1m_p = np.array([], dtype=np.float64)

    for j in range(n_obs):
        ll = ll_mat[:, j]
        lppd_j = float(special.logsumexp(ll)) - log_n_draws

        if np.ptp(ll) < 1e-12:
            pointwise_elpd[j] = float(ll[0])
            pointwise_p_loo[j] = 0.0
            pareto_k[j] = 0.0
            continue

        lw = -ll
        lw -= np.max(lw)  # max(lw) == 0.0

        k_j = 0.0
        if tail_len >= 5:
            order = np.argsort(lw, kind="mergesort")
            lw_s = lw[order]
            u_lw = float(lw_s[n_draws - tail_len - 1])
            top_lw = lw_s[n_draws - tail_len :]

            # Overflow-safe excess scaling when max(lw) - u_lw is very large
            max_top = float(lw_s[-1])
            scale_lw = max(u_lw, max_top - 250.0)
            if u_lw >= max_top - 250.0:
                excess = np.expm1(top_lw - u_lw)
            else:
                excess = np.exp(top_lw - scale_lw) - np.exp(u_lw - scale_lw)
            excess = np.maximum(excess, 0.0)

            k_j, sig_j = gpdfit(excess, wip=True)
            if np.ptp(excess) > 1e-12 and int(np.sum(excess > 0.0)) >= 5 and sig_j > 0.0:
                if abs(k_j) > 1e-8:
                    exp_arg = k_j * neg_log_1m_p
                    # Overflow-safe log(q_gpd) for arbitrary k_j
                    log_abs_expm1 = np.where(
                        exp_arg > 50.0,
                        exp_arg,
                        np.log(np.maximum(np.abs(np.expm1(np.minimum(exp_arg, 50.0))), 1e-300)),
                    )
                    log_q_gpd = np.log(sig_j) - np.log(abs(k_j)) + log_abs_expm1
                else:
                    log_q_gpd = np.log(sig_j) + np.log(neg_log_1m_p)
                lw_tail = scale_lw + np.logaddexp(u_lw - scale_lw, log_q_gpd)
                lw_tail = np.minimum(lw_tail, max_top)
                lw[order[n_draws - tail_len :]] = lw_tail

        log_w = lw - float(special.logsumexp(lw))
        elpd_j = float(special.logsumexp(log_w + ll))
        pointwise_elpd[j] = elpd_j
        pointwise_p_loo[j] = lppd_j - elpd_j
        pareto_k[j] = float(k_j)

    elpd_total = float(np.sum(pointwise_elpd))
    se_elpd = (
        float(np.sqrt(n_obs) * np.std(pointwise_elpd, ddof=1))
        if n_obs > 1
        else 0.0
    )
    p_loo_total = float(np.sum(pointwise_p_loo))

    return PSISLOOResult(
        elpd=elpd_total,
        se_elpd=se_elpd,
        p_loo=p_loo_total,
        looic=-2.0 * elpd_total,
        pointwise_elpd=pointwise_elpd,
        pointwise_p_loo=pointwise_p_loo,
        pareto_k=pareto_k,
    )


def compare_elpd(elpd_models: dict[str, PSISLOOResult]) -> pd.DataFrame:
    """Compare nested CAUSE models using paired pointwise ELPD differences.

    Follows the `jean997/cause` sign convention:
    `delta_elpd = ELPD(model1) - ELPD(model2)`, `z = delta_elpd / se_delta_elpd`,
    and one-sided `p_value = norm.cdf(z)` testing whether `model2` improves
    predictive fit over `model1` (negative `delta_elpd` and `z` favor `model2`).
    """
    if not isinstance(elpd_models, dict) or len(elpd_models) < 2:
        raise ValueError("elpd_models must be a dict containing at least 2 PSISLOOResult entries.")

    lengths = {name: res.pointwise_elpd.size for name, res in elpd_models.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All models must have identical observation counts (got {lengths}).")

    standard_pairs = [("null", "sharing"), ("null", "causal"), ("sharing", "causal")]
    if all(k in elpd_models for k in ("null", "sharing", "causal")):
        pairs = standard_pairs
    else:
        keys = list(elpd_models.keys())
        pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

    rows: list[dict[str, object]] = []
    for m1, m2 in pairs:
        r1 = elpd_models[m1]
        r2 = elpd_models[m2]
        diff = r1.pointwise_elpd - r2.pointwise_elpd
        n_obs = diff.size
        delta = float(np.sum(diff))
        se = float(np.sqrt(n_obs) * np.std(diff, ddof=1)) if n_obs > 1 else 0.0
        if se > 0.0:
            z = delta / se
        elif delta == 0.0:
            z = 0.0
        else:
            z = float(np.sign(delta) * np.inf)
        p_val = float(stats.norm.cdf(z))
        rows.append(
            {
                "model1": m1,
                "model2": m2,
                "delta_elpd": delta,
                "se_delta_elpd": se,
                "z": z,
                "p_value": p_val,
            }
        )

    return pd.DataFrame(rows)
