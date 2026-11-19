"""
Pruned Exact Linear Time (PELT) dynamic programming solver and segment cost
functions for TrackSig ("Signature") and TrackSigFreq ("SigFreq").
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import log_ndtr


def _log_normal_interval_prob(z_lo: np.ndarray, z_hi: np.ndarray) -> np.ndarray:
    """Compute ``log(Phi(z_hi) - Phi(z_lo))`` stably across extreme tails."""
    # Ensure z_hi >= z_lo
    z_hi_eff = np.maximum(z_hi, z_lo + 1e-9)
    # Use symmetry Phi(b) - Phi(a) = Phi(-a) - Phi(-b) when both > 0 so log_ndtr stays accurate
    pos_both = (z_lo > 0.0) & (z_hi_eff > 0.0)
    log_hi = np.where(pos_both, log_ndtr(-z_lo), log_ndtr(z_hi_eff))
    log_lo = np.where(pos_both, log_ndtr(-z_hi_eff), log_ndtr(z_lo))
    diff = np.clip(log_lo - log_hi, -700.0, -1e-15)
    return log_hi + np.log(-np.expm1(diff))


class SegmentScorer:
    """Cached segment cost evaluator ``C(a, b) = -2 * log L(a, b)`` for 1-based bins ``a..b``."""

    def __init__(
        self,
        bin_counts: np.ndarray,
        phi_active: np.ndarray,
        score_method: str,
        bin_n: np.ndarray,
        bin_sum_ccf: np.ndarray,
        bin_sum_ccf2: np.ndarray,
        boundaries: np.ndarray,
        global_sigma: float,
        mean_scale_factor: float | None = None,
        rms_kappa: float | None = None,
    ) -> None:
        self.bin_counts = bin_counts
        self.phi_active = phi_active
        self.score_method = score_method
        self.n_bins, self.n_ctx = bin_counts.shape
        self.n_sig = phi_active.shape[1]

        # Prefix sums of 96-channel counts for O(C) pooling over [a, b]
        self.prefix_counts = np.zeros((self.n_bins + 1, self.n_ctx), dtype=float)
        np.cumsum(bin_counts, axis=0, out=self.prefix_counts[1:])

        # Prefix sums of CCF moments for O(G) truncated Gaussian mixture evaluation
        self.total_muts = float(np.sum(bin_n))
        self.prefix_n = np.zeros(self.n_bins + 1, dtype=float)
        self.prefix_ccf = np.zeros(self.n_bins + 1, dtype=float)
        self.prefix_ccf2 = np.zeros(self.n_bins + 1, dtype=float)
        np.cumsum(bin_n, out=self.prefix_n[1:])
        np.cumsum(bin_sum_ccf, out=self.prefix_ccf[1:])
        np.cumsum(bin_sum_ccf2, out=self.prefix_ccf2[1:])

        self.boundaries = boundaries  # length n_bins + 1: c_0 > c_1 > ... > c_T
        self.global_sigma = max(float(global_sigma), 0.025)
        max_ccf = float(max(1.5, boundaries[1] + 0.25)) if len(boundaries) > 2 else 1.5
        self.ccf_grid = np.linspace(0.02, min(2.2, max_ccf), 120)
        if mean_scale_factor is not None and rms_kappa is not None:
            p_grid = np.clip(float(mean_scale_factor) * self.ccf_grid, 0.02, 0.98)
            self.sigma_grid = np.maximum(
                float(rms_kappa) * np.sqrt(p_grid * (1.0 - p_grid)), 0.025
            )
        else:
            self.sigma_grid = np.full_like(self.ccf_grid, self.global_sigma)
        self._cache: dict[tuple[int, int], tuple[float, np.ndarray, float]] = {}

    def _fit_sig_cat_ll(self, pooled_counts: np.ndarray) -> tuple[np.ndarray, float]:
        """Fast categorical log-likelihood maximization on pooled segment counts."""
        total_n = float(np.sum(pooled_counts))
        if total_n <= 0.0:
            return np.full(self.n_sig, 1.0 / self.n_sig), 0.0
        pos = pooled_counts > 0.0
        n_pos = pooled_counts[pos]
        phi_pos = self.phi_active[pos, :]

        if self.n_sig == 1:
            mix = np.maximum(phi_pos[:, 0], 1e-300)
            return np.array([1.0]), float(np.sum(n_pos * np.log(mix)))

        pi = np.full(self.n_sig, 1.0 / self.n_sig, dtype=float)
        for _ in range(300):
            mix1 = np.maximum(phi_pos @ pi, 1e-300)
            p1 = pi * ((n_pos / mix1) @ phi_pos) / total_n
            p1 /= float(np.sum(p1))
            if np.max(np.abs(p1 - pi)) < 1e-6:
                pi = p1
                break
            mix2 = np.maximum(phi_pos @ p1, 1e-300)
            p2 = p1 * ((n_pos / mix2) @ phi_pos) / total_n
            p2 /= float(np.sum(p2))
            if np.max(np.abs(p2 - p1)) < 1e-6:
                pi = p2
                break
            r = p1 - pi
            v = (p2 - p1) - r
            vv = float(np.dot(v, v))
            if vv > 1e-28:
                alpha = min(-1.0, -np.sqrt(float(np.dot(r, r)) / vv))
                cand = np.maximum(pi - 2.0 * alpha * r + (alpha * alpha) * v, 0.0)
                cs = float(np.sum(cand))
                if cs > 0.0:
                    cand /= cs
                    ll_cand = float(np.sum(n_pos * np.log(np.maximum(phi_pos @ cand, 1e-300))))
                    ll_p2 = float(np.sum(n_pos * np.log(np.maximum(phi_pos @ p2, 1e-300))))
                    if ll_cand >= ll_p2 - 1e-12:
                        mix_c = np.maximum(phi_pos @ cand, 1e-300)
                        pi = cand * ((n_pos / mix_c) @ phi_pos) / total_n
                        pi /= float(np.sum(pi))
                        continue
            pi = p2

        mix = np.maximum(phi_pos @ pi, 1e-300)
        return pi, float(np.sum(n_pos * np.log(mix)))

    def _fit_ccf_ll(self, a: int, b: int) -> tuple[float, float]:
        """Exact piecewise-truncated Gaussian CCF mixture log-likelihood on bins ``a..b``."""
        n_ab = self.prefix_n[b] - self.prefix_n[a - 1]
        if n_ab <= 0.0:
            return 1.0, 0.0

        sum1 = self.prefix_ccf[b] - self.prefix_ccf[a - 1]
        sum2 = self.prefix_ccf2[b] - self.prefix_ccf2[a - 1]
        grid = self.ccf_grid
        sig = self.sigma_grid
        sig2 = sig * sig

        # Unnormalized Gaussian log-density sum across i in [a, b] over grid phi
        quad = sum2 - 2.0 * sum1 * grid + n_ab * (grid * grid)
        ll_gauss = -0.5 * n_ab * np.log(2.0 * np.pi * sig2) - 0.5 * quad / sig2

        # Truncation interval [c_b, c_{a-1}] where c_0 = +inf, c_T = -inf
        hi = self.boundaries[a - 1]
        lo = self.boundaries[b]
        z_hi = np.full_like(grid, 40.0) if np.isposinf(hi) else (hi - grid) / sig
        z_lo = np.full_like(grid, -40.0) if np.isneginf(lo) else (lo - grid) / sig
        log_delta_f = _log_normal_interval_prob(z_lo, z_hi)

        ll_vec = n_ab * np.log(n_ab / self.total_muts) + ll_gauss - n_ab * log_delta_f
        best_idx = int(np.argmax(ll_vec))
        return float(grid[best_idx]), float(ll_vec[best_idx])

    def evaluate(self, a: int, b: int) -> tuple[float, np.ndarray, float]:
        """Return ``(cost, exposures, fitted_ccf)`` for 1-based bin interval ``[a, b]``."""
        key = (a, b)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        pooled = self.prefix_counts[b] - self.prefix_counts[a - 1]
        pi, ll_sig = self._fit_sig_cat_ll(pooled)
        n_ab = self.prefix_n[b] - self.prefix_n[a - 1]
        mean_ccf = float((self.prefix_ccf[b] - self.prefix_ccf[a - 1]) / max(n_ab, 1.0))

        if self.score_method == "SigFreq":
            fitted_ccf, ll_ccf = self._fit_ccf_ll(a, b)
            total_ll = ll_sig + ll_ccf
        else:
            fitted_ccf = mean_ccf
            total_ll = ll_sig

        cost = -2.0 * total_ll
        res = (cost, pi, fitted_ccf)
        self._cache[key] = res
        return res

    def cost(self, a: int, b: int) -> float:
        """Return ``-2 * log L(a, b)`` for 1-based bin interval ``[a, b]``."""
        return self.evaluate(a, b)[0]


def run_pelt(
    cost_fn: Callable[[int, int], float],
    n_bins: int,
    penalty: float,
    use_pruning: bool = True,
) -> tuple[list[int], float]:
    """Pruned Exact Linear Time (PELT) optimal partitioning over ``1 .. n_bins``.

    Minimizes ``sum_{j=0}^m C(tau_j + 1, tau_{j+1}) + m * penalty`` over all
    changepoint sets ``0 = tau_0 < tau_1 < ... < tau_m < tau_{m+1} = n_bins``
    using the exact Killick et al. (2012) pruning inequality
    ``F(tau) + C(tau + 1, t) > F(t)`` (with ``K_0 = 0`` for maximum-likelihood
    deviance costs).

    Returns
    -------
    changepoints : list[int]
        Sorted 1-based bin indices ``tau_1, ..., tau_m`` in ``1 .. n_bins - 1``
        marking the end of each non-terminal segment.
    penalized_cost : float
        Optimal objective ``F(n_bins) = sum_j C(a_j, b_j) + m * penalty``.
    """
    if n_bins <= 1:
        c1 = cost_fn(1, 1) if n_bins == 1 else 0.0
        return [], float(c1)

    f_opt = np.full(n_bins + 1, np.inf, dtype=float)
    f_opt[0] = -float(penalty)
    prev_cp = np.zeros(n_bins + 1, dtype=int)
    candidates: list[int] = [0]

    for t in range(1, n_bins + 1):
        best_val = np.inf
        best_tau = 0
        unpenalized_vals: list[tuple[int, float]] = []

        for tau in candidates:
            seg_cost = cost_fn(tau + 1, t)
            val_no_pen = f_opt[tau] + seg_cost
            unpenalized_vals.append((tau, val_no_pen))
            val_with_pen = val_no_pen + penalty
            if val_with_pen < best_val:
                best_val = val_with_pen
                best_tau = tau

        f_opt[t] = best_val
        prev_cp[t] = best_tau

        if use_pruning:
            # Exact PELT pruning (K_0 = 0): keep tau if F(tau) + C(tau+1, t) <= F(t) + eps
            tol_prune = 1e-11
            next_candidates = [
                tau for tau, val_np in unpenalized_vals if val_np <= best_val + tol_prune
            ]
            next_candidates.append(t)
            candidates = next_candidates
        else:
            candidates.append(t)

    # Backtrack optimal changepoints
    cpts: list[int] = []
    cur = int(prev_cp[n_bins])
    while cur > 0:
        cpts.append(cur)
        cur = int(prev_cp[cur])
    cpts.reverse()
    return cpts, float(f_opt[n_bins])
