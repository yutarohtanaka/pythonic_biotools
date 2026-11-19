"""
Tester-pair selection and exact 2x2 angular profile-GLS proportionality testing
for colocPropTest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate mandatory ``settings`` mode parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass
class PairPropTestResult:
    """Exact 2-SNP frequentist proportionality test result."""

    stat: float
    p_value: float
    eta_hat: float
    theta_hat: float
    df: int = 1

    @property
    def chisq(self) -> float:
        """Alias for chi-square statistic ``stat``."""
        return self.stat

    @property
    def pvalue(self) -> float:
        """Alias for ``p_value``."""
        return self.p_value

    def to_dict(self) -> dict[str, float | int]:
        return {
            "stat": self.stat,
            "p_value": self.p_value,
            "eta_hat": self.eta_hat,
            "theta_hat": self.theta_hat,
            "df": self.df,
        }


def simultaneous_diagonalize(
    sigma2: np.ndarray,
    sigma1: np.ndarray,
    b2: np.ndarray,
    b1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneously diagonalize SPD matrices ``(sigma2, sigma1)``.

    Finds invertible ``P`` such that ``P^T sigma2 P = I`` and
    ``P^T sigma1 P = diag(lam)``, returning ``(y, x, lam)`` where
    ``y = P^T b2`` and ``x = P^T b1``.
    """
    k = sigma2.shape[0]
    jitter2 = 1e-12 * max(float(np.mean(np.diag(sigma2))), 1e-12)
    jitter1 = 1e-12 * max(float(np.mean(np.diag(sigma1))), 1e-12)
    s2 = 0.5 * (sigma2 + sigma2.T) + jitter2 * np.eye(k)
    s1 = 0.5 * (sigma1 + sigma1.T) + jitter1 * np.eye(k)

    chol2 = np.linalg.cholesky(s2)
    inv_chol2 = np.linalg.solve(chol2, np.eye(k))
    m_mat = inv_chol2 @ s1 @ inv_chol2.T
    m_mat = 0.5 * (m_mat + m_mat.T)
    lam, u_mat = np.linalg.eigh(m_mat)
    lam = np.maximum(lam, 1e-15)

    # P^T = U^T L_2^{-1}
    pt = u_mat.T @ inv_chol2
    y = pt @ b2
    x = pt @ b1
    return y, x, lam


def eval_angular_q_and_grad(
    theta: float | np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Evaluate exact angular profile-GLS quadratic form ``Q(theta)`` and ``dQ/dtheta``."""
    scalar_in = np.ndim(theta) == 0
    th = np.atleast_1d(np.asarray(theta, dtype=float))  # (G,)
    ct = np.cos(th)[:, None]  # (G, 1)
    st = np.sin(th)[:, None]  # (G, 1)

    u = ct * y[None, :] - st * x[None, :]  # (G, K)
    du = -st * y[None, :] - ct * x[None, :]  # (G, K)
    w = (ct * ct) + (st * st) * lam[None, :]  # (G, K)
    dw = 2.0 * ct * st * (lam[None, :] - 1.0)  # (G, K)

    q_terms = (u * u) / w
    dq_terms = (2.0 * u * du * w - (u * u) * dw) / (w * w)

    q_val = np.sum(q_terms, axis=1)
    dq_val = np.sum(dq_terms, axis=1)
    if scalar_in:
        return float(q_val[0]), float(dq_val[0])
    return q_val, dq_val


def _secular_2x2_coeffs_desc(y1: float, y2: float, x1: float, x2: float, l1: float, l2: float) -> np.ndarray:
    """Closed-form degree-6 polynomial coefficients (descending powers) of dQ/d(eta) = 0 for K=2."""

    def _branch(yk: float, xk: float, lk: float, lm: float) -> np.ndarray:
        # (yk - xk*e)*(xk + lk*yk*e) = c0 + c1*e + c2*e^2
        c0 = yk * xk
        c1 = lk * yk * yk - xk * xk
        c2 = -lk * xk * yk
        # (1 + lm*e^2)^2 = 1 + 2*lm*e^2 + lm^2*e^4
        lm2 = lm * lm
        # Product in ascending powers e^0 .. e^6:
        return np.array(
            [
                c0,
                c1,
                c2 + 2.0 * lm * c0,
                2.0 * lm * c1,
                2.0 * lm * c2 + lm2 * c0,
                lm2 * c1,
                lm2 * c2,
            ],
            dtype=float,
        )

    asc = _branch(y1, x1, l1, l2) + _branch(y2, x2, l2, l1)
    return asc[::-1]


def minimize_angular_gls(
    y: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
    include_2x2_roots: bool = False,
) -> tuple[float, float, float]:
    """Find the exact global minimum of ``Q(theta)`` over ``theta in [0, pi)``.

    Returns ``(q_min, theta_hat, eta_hat)`` with ``theta_hat in [0, pi)`` and
    ``eta_hat = tan(theta_hat)``.
    """
    grid = np.linspace(0.0, np.pi, 128, endpoint=False)
    extra: list[float] = [0.0, 0.5 * np.pi]

    if include_2x2_roots and len(y) == 2:
        coeffs_desc = _secular_2x2_coeffs_desc(
            float(y[0]), float(y[1]), float(x[0]), float(x[1]), float(lam[0]), float(lam[1])
        )
        nz = np.flatnonzero(np.abs(coeffs_desc) > 1e-24)
        if nz.size >= 2:
            roots = np.roots(coeffs_desc[nz[0] :])
            real_roots = roots[np.abs(roots.imag) < 1e-7].real
            for r_val in real_roots:
                extra.append(float(np.mod(np.arctan(r_val), np.pi)))

    cand_arr = np.concatenate([grid, np.asarray(extra, dtype=float)])
    ct = np.cos(cand_arr)[:, None]
    st = np.sin(cand_arr)[:, None]
    u = ct * y[None, :] - st * x[None, :]
    w = (ct * ct) + (st * st) * lam[None, :]
    q_arr = np.sum((u * u) / w, axis=1)

    best_idx = int(np.argmin(q_arr))
    best_q = float(q_arr[best_idx])
    best_theta = float(np.mod(cand_arr[best_idx], np.pi))

    # Local Brent polish around top candidate (periodic on R with period pi)
    half_win = np.pi / 64.0

    def _q_scalar(th: float) -> float:
        c = float(np.cos(th))
        s = float(np.sin(th))
        num = c * y - s * x
        den = (c * c) + (s * s) * lam
        return float(np.sum((num * num) / den))

    opt = minimize_scalar(
        _q_scalar,
        bounds=(best_theta - half_win, best_theta + half_win),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if float(opt.fun) < best_q:
        best_q = float(opt.fun)
        best_theta = float(np.mod(float(opt.x), np.pi))

    best_q = max(0.0, best_q)
    best_theta = float(np.mod(best_theta, np.pi))
    if abs(np.cos(best_theta)) < 1e-12:
        eta_hat = float(np.inf)
    else:
        eta_hat = float(np.tan(best_theta))
    return best_q, best_theta, eta_hat


def select_tester_pairs(
    b1: np.ndarray,
    se1: np.ndarray,
    b2: np.ndarray,
    se2: np.ndarray,
    LD: np.ndarray,
    max_testers: int = 15,
    max_r2: float = 0.95,
    min_abs_z: float = 2.0,
    settings: SettingsMode = "default",
) -> list[tuple[int, int]]:
    """Select informative, non-collinear SNP tester pairs ``(i, j)`` in a genomic locus.

    Ranks variants by combined marginal signal ``max(|z1|, |z2|)`` (tie-broken by
    ``z1^2 + z2^2``), greedily LD-prunes near-duplicate variants exceeding ``max_r2``
    up to ``max_testers`` variants, and returns all pairs ``(i, j)`` (``i < j``)
    with ``LD[i, j]^2 <= max_r2``. If fewer than 2 variants satisfy ``>= min_abs_z``,
    gracefully falls back to the top non-collinear variants so null/weak loci remain
    testable without degeneracies.
    """
    validate_settings(settings)
    b1_a = np.asarray(b1, dtype=float)
    se1_a = np.asarray(se1, dtype=float)
    b2_a = np.asarray(b2, dtype=float)
    se2_a = np.asarray(se2, dtype=float)
    r_mat = np.asarray(LD, dtype=float)

    m = len(b1_a)
    if m < 2:
        raise ValueError(f"At least 2 SNPs are required to form tester pairs, got {m}.")
    if se1_a.shape != (m,) or b2_a.shape != (m,) or se2_a.shape != (m,):
        raise ValueError("b1, se1, b2, se2 must all be 1D arrays of identical length.")
    if r_mat.shape != (m, r_mat.shape[0]) or r_mat.shape != (m, m):
        raise ValueError(f"LD must have shape ({m}, {m}), got {r_mat.shape}.")
    if np.any(se1_a <= 0.0) or np.any(se2_a <= 0.0):
        raise ValueError("Standard errors se1 and se2 must be strictly positive.")
    if int(max_testers) < 2:
        raise ValueError(f"max_testers must be >= 2, got {max_testers}.")
    if not (0.0 < float(max_r2) < 1.0):
        raise ValueError(f"max_r2 must be in (0, 1), got {max_r2}.")

    z1 = np.abs(b1_a / se1_a)
    z2 = np.abs(b2_a / se2_a)
    max_z = np.maximum(z1, z2)
    sum_z2 = (z1 * z1) + (z2 * z2)

    # Sort descending by (max_z, sum_z2)
    order = np.lexsort((-sum_z2, -max_z))
    r2_mat = r_mat * r_mat

    def _greedy_select(cand_indices: np.ndarray) -> list[int]:
        chosen: list[int] = []
        for idx in cand_indices:
            i_int = int(idx)
            if all(r2_mat[i_int, c] <= max_r2 for c in chosen):
                chosen.append(i_int)
                if len(chosen) >= int(max_testers):
                    break
        return chosen

    strong_order = order[max_z[order] >= float(min_abs_z)]
    selected = _greedy_select(strong_order)
    if len(selected) < 2:
        # Fallback to top variants across full locus while keeping already chosen
        for idx in order:
            i_int = int(idx)
            if i_int not in selected and all(r2_mat[i_int, c] <= max_r2 for c in selected):
                selected.append(i_int)
                if len(selected) >= min(int(max_testers), max(2, len(selected))):
                    if len(selected) >= 2 and max_z[i_int] < float(min_abs_z):
                        # Under weak/null loci, keep up to min(max_testers, 6) non-collinear testers
                        if len(selected) >= min(int(max_testers), 6):
                            break

    selected_sorted = sorted(selected)
    pairs: list[tuple[int, int]] = []
    for a_i in range(len(selected_sorted)):
        for b_i in range(a_i + 1, len(selected_sorted)):
            u, v = selected_sorted[a_i], selected_sorted[b_i]
            if r2_mat[u, v] <= max_r2:
                pairs.append((u, v))
    return pairs


def test_pair_proportionality(
    b1_pair: np.ndarray | list[float],
    se1_pair: np.ndarray | list[float],
    b2_pair: np.ndarray | list[float],
    se2_pair: np.ndarray | list[float],
    LD_pair: np.ndarray,
    settings: SettingsMode = "default",
) -> PairPropTestResult:
    """Exact 2-SNP frequentist test of ``H_0: beta_2 = eta * beta_1``.

    Minimizes the correlated summary-statistic profile-GLS quadratic form
    ``Q(eta) = (b2 - eta b1)^T (Sigma_2 + eta^2 Sigma_1)^{-1} (b2 - eta b1)``
    via simultaneous diagonalization and exact angular parameterization
    ``theta in [0, pi)`` (``eta = tan(theta)``), avoiding division-by-zero even
    when Trait 1 is completely null (``eta -> inf``) and yielding an exact
    ``chi^2_1`` test statistic and p-value under ``H_0``.
    """
    validate_settings(settings)
    b1_v = np.asarray(b1_pair, dtype=float)
    s1_v = np.asarray(se1_pair, dtype=float)
    b2_v = np.asarray(b2_pair, dtype=float)
    s2_v = np.asarray(se2_pair, dtype=float)
    r_2x2 = np.asarray(LD_pair, dtype=float)

    if b1_v.shape != (2,) or s1_v.shape != (2,) or b2_v.shape != (2,) or s2_v.shape != (2,):
        raise ValueError("b1_pair, se1_pair, b2_pair, se2_pair must all have shape (2,).")
    if r_2x2.shape != (2, 2):
        raise ValueError(f"LD_pair must have shape (2, 2), got {r_2x2.shape}.")
    if np.any(s1_v <= 0.0) or np.any(s2_v <= 0.0):
        raise ValueError("Pair standard errors must be strictly positive.")

    r_off = float(np.clip(0.5 * (r_2x2[0, 1] + r_2x2[1, 0]), -0.99999, 0.99999))
    r_clean = np.array([[1.0, r_off], [r_off, 1.0]], dtype=float)

    sigma1 = np.outer(s1_v, s1_v) * r_clean
    sigma2 = np.outer(s2_v, s2_v) * r_clean

    y, x, lam = simultaneous_diagonalize(sigma2=sigma2, sigma1=sigma1, b2=b2_v, b1=b1_v)
    q_min, theta_hat, eta_hat = minimize_angular_gls(y, x, lam, include_2x2_roots=True)
    p_val = float(chi2.sf(q_min, df=1))

    return PairPropTestResult(
        stat=q_min,
        p_value=p_val,
        eta_hat=eta_hat,
        theta_hat=theta_hat,
        df=1,
    )


# Prevent pytest from collecting this library API function as a test case when imported
test_pair_proportionality.__test__ = False  # type: ignore[attr-defined]
