"""
Exact Poisson-Binomial distribution and discrete false discovery rate control.

Implements:
- Hong (2013) Discrete Fourier Transform (DFT) exact PMF in log-polar form
  (`np.fft.rfft` / `np.fft.irfft`) and exact non-negative dynamic programming.
- Lancaster (1961) mid-p correction matching Canisius et al. (2016).
- Discrete-support-aware Benjamini-Hochberg FDR (`discover_fdr`; Carlson et al.,
  2009; Canisius, Martens & Wessels, Genome Biology 2016).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _validate_probs(probs: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).ravel()
    if np.any(np.isnan(p)):
        raise ValueError("Probabilities must not contain NaN values.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("All probabilities must lie in [0, 1].")
    return p


def _dp_interior_pmf(p: np.ndarray) -> np.ndarray:
    """Exact non-negative O(m^2) dynamic programming on interior p in (0, 1)."""
    m = p.size
    dp = np.zeros(m + 1, dtype=np.float64)
    dp[0] = 1.0
    for idx in range(m):
        pk = float(p[idx])
        qk = 1.0 - pk
        dp[1 : idx + 2] = dp[1 : idx + 2] * qk + dp[0 : idx + 1] * pk
        dp[0] *= qk
    total = dp.sum()
    if total > 0.0:
        dp /= total
    return dp


def _dft_interior_pmf(p: np.ndarray) -> np.ndarray:
    """
    Hong (2013) exact DFT algorithm in log-polar form via `np.fft.irfft`.

    Evaluates the characteristic function at conjugate DFT roots:
        cf[l] = E[exp(-i * 2*pi * l * S / (m + 1))]
              = prod_{k=1}^m (1 - p_k + p_k * exp(-i * omega * l))
    for l = 0, ..., floor((m + 1) / 2), matching `np.fft.rfft(pmf)`.
    """
    m = p.size
    n_grid = m + 1
    n_rfft = n_grid // 2 + 1
    l_idx = np.arange(n_rfft, dtype=np.float64)
    omega_l = (2.0 * np.pi / n_grid) * l_idx  # shape (n_rfft,)

    cos_wl = np.cos(omega_l)  # shape (n_rfft,)
    sin_wl = np.sin(omega_l)  # shape (n_rfft,)
    sin_half_sq = np.sin(0.5 * omega_l) ** 2  # shape (n_rfft,)

    # Broadcast over (m, n_rfft):
    # |z_{k,l}|^2 = 1 - 4 * p_k * (1 - p_k) * sin^2(omega * l / 2)
    four_pq = 4.0 * p * (1.0 - p)  # shape (m,)
    arg = -four_pq[:, None] * sin_half_sq[None, :]  # in [-1, 0]

    # Exact zero occurs iff arg == -1 (i.e. p_k = 0.5 and l = n_grid / 2 when n_grid is even)
    zero_col = np.any(arg <= -1.0 + 1e-15, axis=0)
    safe_arg = np.maximum(arg, -1.0 + 1e-300)

    log_mod = 0.5 * np.sum(np.log1p(safe_arg), axis=0)
    real_part = (1.0 - p)[:, None] + p[:, None] * cos_wl[None, :]
    imag_part = -p[:, None] * sin_wl[None, :]
    phase = np.sum(np.arctan2(imag_part, real_part), axis=0)

    cf = np.exp(log_mod) * (np.cos(phase) + 1j * np.sin(phase))
    cf[0] = 1.0 + 0.0j
    if np.any(zero_col):
        cf[zero_col] = 0.0 + 0.0j

    pmf = np.fft.irfft(cf, n=n_grid)
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum()
    if total > 0.0:
        pmf /= total
    return pmf


def poisson_binomial_pmf(
    probs: np.ndarray | list[float],
    method: Literal["auto", "dft", "dp"] = "auto",
) -> np.ndarray:
    """
    Compute the exact PMF on {0, 1, ..., N} of S = sum_{k=1}^N Bernoulli(p_k).

    Parameters
    ----------
    probs : array-like of shape (N,)
        Success probabilities p_k in [0, 1].
    method : {"auto", "dft", "dp"}, default="auto"
        - "dft": Hong (2013) log-polar Discrete Fourier Transform (`irfft`).
        - "dp": Exact non-negative dynamic programming recurrence.
        - "auto": Uses exact DP for small interior m <= 64 and Hong DFT for m > 64.

    Returns
    -------
    pmf : np.ndarray of shape (N + 1,)
        Exact probabilities Pr(S = k) for k = 0, ..., N.
    """
    if method not in ("auto", "dft", "dp"):
        raise ValueError(f"Unknown method '{method}'. Expected 'auto', 'dft', or 'dp'.")

    p = _validate_probs(probs)
    n_total = p.size
    if n_total == 0:
        return np.array([1.0], dtype=np.float64)

    n_ones = int(np.sum(p == 1.0))
    interior = p[(p > 0.0) & (p < 1.0)]
    m = interior.size

    if m == 0:
        out = np.zeros(n_total + 1, dtype=np.float64)
        out[n_ones] = 1.0
        return out

    use_dp = method == "dp" or (method == "auto" and m <= 64)
    core_pmf = _dp_interior_pmf(interior) if use_dp else _dft_interior_pmf(interior)

    if m == n_total:
        return core_pmf

    out = np.zeros(n_total + 1, dtype=np.float64)
    out[n_ones : n_ones + m + 1] = core_pmf
    return out


def poisson_binomial_pval(
    k_obs: int,
    probs: np.ndarray | list[float],
    alternative: Literal["less", "greater", "two-sided", "exclusivity", "co-occurrence"] = "less",
    mid_p: bool = True,
    method: Literal["auto", "dft", "dp"] = "auto",
) -> float:
    """
    Exact tail probability under a Poisson-Binomial null distribution.

    When `method="auto"`, if the DFT tail estimate approaches the IEEE float64
    FFT floor (< 1e-12), automatically refines via exact non-negative DP so
    extreme tail p-values (down to ~1e-300) retain full relative accuracy.
    """
    if int(k_obs) != k_obs:
        raise ValueError(f"k_obs must be an integer, got {k_obs}.")
    k = int(k_obs)

    alt_map = {
        "less": "less",
        "exclusivity": "less",
        "greater": "greater",
        "co-occurrence": "greater",
        "co_occurrence": "greater",
        "two-sided": "two-sided",
    }
    if alternative not in alt_map:
        raise ValueError(
            f"Invalid alternative '{alternative}'. Expected 'less', 'greater', "
            "'two-sided', 'exclusivity', or 'co-occurrence'."
        )
    alt = alt_map[alternative]

    p = _validate_probs(probs)
    n_total = p.size
    if k < 0 or k > n_total:
        raise ValueError(f"k_obs={k} is outside valid support [0, {n_total}].")

    pmf = poisson_binomial_pmf(p, method=method)
    pval = _tail_from_pmf(k, pmf, alt, mid_p=mid_p)

    # Refine extreme tails below FFT precision floor using exact non-negative DP
    if method == "auto" and p.size > 64 and pval < 1e-12:
        pmf_dp = poisson_binomial_pmf(p, method="dp")
        pval = _tail_from_pmf(k, pmf_dp, alt, mid_p=mid_p)

    return float(np.clip(pval, 0.0, 1.0))


def _tail_from_pmf(
    k: int,
    pmf: np.ndarray,
    alt: str,
    *,
    mid_p: bool,
) -> float:
    pk = float(pmf[k])
    weight_k = 0.5 if mid_p else 1.0
    if alt == "less":
        lower = float(np.sum(pmf[:k])) if k > 0 else 0.0
        return lower + weight_k * pk
    if alt == "greater":
        upper = float(np.sum(pmf[k + 1 :])) if k + 1 < pmf.size else 0.0
        return upper + weight_k * pk
    # two-sided minimum-likelihood / double-tail
    lower = (float(np.sum(pmf[:k])) if k > 0 else 0.0) + weight_k * pk
    upper = (float(np.sum(pmf[k + 1 :])) if k + 1 < pmf.size else 0.0) + weight_k * pk
    return min(1.0, 2.0 * min(lower, upper))


def bh_fdr(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Classical Benjamini-Hochberg (1995) step-up FDR q-values."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return np.array([], dtype=np.float64)
    if np.any(np.isnan(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1] without NaN.")

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = (m * p_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=np.float64)
    q[order] = q_sorted
    return q


def discover_fdr(
    p_values: np.ndarray | list[float],
    p_mins: np.ndarray | list[float],
) -> np.ndarray:
    """
    Adaptive discrete-support Benjamini-Hochberg FDR (Carlson et al., 2009;
    Canisius, Martens & Wessels, Genome Biology 2016).

    For sorted observed p-value threshold tau = p_(k), only hypotheses whose
    minimum attainable discrete p-value satisfies p_{r,min} <= tau can possibly
    reject at level tau under H_0. Thus effective multiplicity at rank k is:
        m(p_(k)) = #{r : p_{r,min} <= p_(k)}
    followed by step-up monotonicity enforcement:
        q_(k) = min(1, min_{j >= k} m(p_(j)) * p_(j) / j).
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    pmin = np.asarray(p_mins, dtype=np.float64).ravel()
    if p.shape != pmin.shape:
        raise ValueError(f"p_values shape {p.shape} != p_mins shape {pmin.shape}.")
    m = p.size
    if m == 0:
        return np.array([], dtype=np.float64)
    if np.any(np.isnan(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1] without NaN.")
    if np.any(np.isnan(pmin)) or np.any((pmin < 0.0) | (pmin > 1.0)):
        raise ValueError("p_mins must lie in [0, 1] without NaN.")

    # Guard against tiny floating-point rounding where p_r < p_{r,min} by ~1e-16
    pmin_safe = np.minimum(pmin, p)

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    pmin_sorted = np.sort(pmin_safe)

    # m(tau) via vectorized right-side binary search
    m_eff = np.searchsorted(pmin_sorted, p_sorted + 1e-15, side="right").astype(np.float64)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    m_eff = np.maximum(m_eff, ranks)

    q_sorted = (m_eff * p_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=np.float64)
    q[order] = q_sorted
    return q
