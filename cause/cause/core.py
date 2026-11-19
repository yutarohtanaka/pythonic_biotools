"""
Core statistical engine for CAUSE (Causal Analysis Using Summary Effect Estimates).

Reference:
  Morrison, J., Knoblauch, N., Marcus, J. H., Stephens, M., & He, X. (2020).
  "Mendelian randomization accounting for correlated and uncorrelated pleiotropic
  effects using genome-wide summary statistics." Nature Genetics, 52(7), 700-706.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .harmonize import _extract_cause_arrays
from .psis import PSISLOOResult, compare_elpd, psis_loo


@dataclass(frozen=True)
class CauseParams:
    """Genome-wide nuisance parameter estimates from `est_cause_params`."""

    rho: float
    S1: np.ndarray
    S2: np.ndarray
    pi: np.ndarray
    n_snps: int
    n_null_snps_rho: int
    log_likelihood: float
    converged: bool
    n_iter: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return active bivariate mixture grid `(S1, S2, pi)` sorted by descending `pi`."""
        df = pd.DataFrame(
            {
                "S1": self.S1,
                "S2": self.S2,
                "pi": self.pi,
            }
        )
        return df.sort_values("pi", ascending=False, kind="mergesort").reset_index(drop=True)

    def summary(self) -> str:
        """Return formatted summary of estimated nuisance parameters."""
        null_mask = (self.S1 == 0.0) & (self.S2 == 0.0)
        pi_null = float(np.sum(self.pi[null_mask])) if np.any(null_mask) else 0.0
        lines = [
            "CAUSE Nuisance Parameters",
            "=" * 50,
            f"SNPs used:          {self.n_snps} (null SNPs for rho: {self.n_null_snps_rho})",
            f"Sample overlap rho: {self.rho:.6f}",
            f"Active components:  {self.pi.size} (null weight pi_0 = {pi_null:.4f})",
            f"Log-likelihood:     {self.log_likelihood:.4f}",
            f"Converged:          {self.converged} (iterations = {self.n_iter})",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CauseModelFit:
    """Posterior parameter summary and PSIS-LOO fit for a CAUSE model (`sharing` or `causal`)."""

    model_name: Literal["sharing", "causal"]
    gamma_median: float
    gamma_ci_low: float
    gamma_ci_high: float
    gamma_mean: float
    gamma_sd: float
    eta_median: float
    eta_ci_low: float
    eta_ci_high: float
    eta_mean: float
    eta_sd: float
    q_median: float
    q_ci_low: float
    q_ci_high: float
    q_mean: float
    q_sd: float
    posterior_sharing_prob: np.ndarray
    log_lik_draws: np.ndarray
    loo: PSISLOOResult

    def to_dataframe(self) -> pd.DataFrame:
        """Return 3-row parameter posterior summary (`gamma`, `eta`, `q`)."""
        return pd.DataFrame(
            [
                {
                    "parameter": "gamma",
                    "median": self.gamma_median,
                    "ci_low": self.gamma_ci_low,
                    "ci_high": self.gamma_ci_high,
                    "mean": self.gamma_mean,
                    "sd": self.gamma_sd,
                },
                {
                    "parameter": "eta",
                    "median": self.eta_median,
                    "ci_low": self.eta_ci_low,
                    "ci_high": self.eta_ci_high,
                    "mean": self.eta_mean,
                    "sd": self.eta_sd,
                },
                {
                    "parameter": "q",
                    "median": self.q_median,
                    "ci_low": self.q_ci_low,
                    "ci_high": self.q_ci_high,
                    "mean": self.q_mean,
                    "sd": self.q_sd,
                },
            ]
        )


@dataclass(frozen=True)
class CauseResult:
    """Full CAUSE Bayesian Mendelian randomization model comparison result."""

    n_instruments: int
    snp_ids: list[str]
    beta_hat_1: np.ndarray
    seb1: np.ndarray
    beta_hat_2: np.ndarray
    seb2: np.ndarray
    params: CauseParams
    sigma_g: float
    qalpha: float
    qbeta: float
    max_q: float
    elpd_null: float
    elpd_sharing: float
    elpd_causal: float
    delta_elpd: float
    se_delta_elpd: float
    z_score: float
    p_value: float
    sharing: CauseModelFit
    causal: CauseModelFit
    loo_null: PSISLOOResult
    elpd_table: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return pairwise ELPD model comparison table (`model1`, `model2`, `delta_elpd`, `se_delta_elpd`, `z`, `p_value`)."""
        return self.elpd_table.copy()

    def posterior_dataframe(self) -> pd.DataFrame:
        """Return combined posterior summaries for `sharing` and `causal` models."""
        sh_df = self.sharing.to_dataframe()
        sh_df.insert(0, "model", "sharing")
        ca_df = self.causal.to_dataframe()
        ca_df.insert(0, "model", "causal")
        return pd.concat([sh_df, ca_df], ignore_index=True)

    def snp_table(self) -> pd.DataFrame:
        """Return per-instrument posterior sharing probabilities and pointwise PSIS-LOO diagnostics."""
        return pd.DataFrame(
            {
                "snp": self.snp_ids,
                "beta_hat_1": self.beta_hat_1,
                "seb1": self.seb1,
                "beta_hat_2": self.beta_hat_2,
                "seb2": self.seb2,
                "pr_z_sharing": self.sharing.posterior_sharing_prob,
                "pr_z_causal": self.causal.posterior_sharing_prob,
                "elpd_null": self.loo_null.pointwise_elpd,
                "elpd_sharing": self.sharing.loo.pointwise_elpd,
                "elpd_causal": self.causal.loo.pointwise_elpd,
                "pareto_k_sharing": self.sharing.loo.pareto_k,
                "pareto_k_causal": self.causal.loo.pareto_k,
            }
        )

    def summary(self) -> str:
        """Format human-readable CAUSE model comparison and posterior summary."""
        lines = [
            "CAUSE Summary",
            "=" * 65,
            f"Instruments: {self.n_instruments} | rho = {self.params.rho:.4f} | sigma_g = {self.sigma_g:.4g}",
            f"Prior on q:  Beta({self.qalpha:g}, {self.qbeta:g}) on (0, {self.max_q:g}]",
            "",
            "ELPD Model Comparison (delta_elpd = ELPD_model1 - ELPD_model2):",
        ]
        for row in self.elpd_table.itertuples(index=False):
            lines.append(
                f"  {row.model1:>7s} vs {row.model2:<7s}: "
                f"delta_elpd = {row.delta_elpd:9.3f} (SE = {row.se_delta_elpd:7.3f}), "
                f"z = {row.z:7.3f}, p = {row.p_value:.4g}"
            )
        lines.extend(
            [
                "",
                "Posterior Medians & 95% Credible Intervals:",
                f"  Sharing (gamma = 0):",
                f"    eta = {self.sharing.eta_median:.4g} [{self.sharing.eta_ci_low:.4g}, {self.sharing.eta_ci_high:.4g}]",
                f"    q   = {self.sharing.q_median:.4g} [{self.sharing.q_ci_low:.4g}, {self.sharing.q_ci_high:.4g}]",
                f"  Causal:",
                f"    gamma = {self.causal.gamma_median:.4g} [{self.causal.gamma_ci_low:.4g}, {self.causal.gamma_ci_high:.4g}]",
                f"    eta   = {self.causal.eta_median:.4g} [{self.causal.eta_ci_low:.4g}, {self.causal.eta_ci_high:.4g}]",
                f"    q     = {self.causal.q_median:.4g} [{self.causal.q_ci_low:.4g}, {self.causal.q_ci_high:.4g}]",
            ]
        )
        n_bad_sh = self.sharing.loo.n_high_pareto_k
        n_bad_ca = self.causal.loo.n_high_pareto_k
        if n_bad_sh > 0 or n_bad_ca > 0:
            lines.append(
                f"\nWarning: High Pareto k (> 0.7) detected: sharing={n_bad_sh}, causal={n_bad_ca}."
            )
        return "\n".join(lines)


def _canonicalize_signs(
    b1: np.ndarray, b2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Orient each SNP so exposure effect `b1 >= 0`, preserving coding invariance."""
    signs = np.where(b1 >= 0.0, 1.0, -1.0)
    return b1 * signs, b2 * signs


def _estimate_rho_trunc_mle(
    z1: np.ndarray,
    z2: np.ndarray,
    null_p_thresh: float,
) -> tuple[float, int]:
    """Estimate sample-overlap correlation via truncated bivariate normal MLE."""
    n_total = z1.size
    if n_total < 3:
        return 0.0, n_total

    c = float(stats.norm.ppf(1.0 - null_p_thresh / 2.0))
    mask = (np.abs(z1) < c) & (np.abs(z2) < c)
    if int(np.sum(mask)) < 20:
        c = 3.0
        mask = (np.abs(z1) < c) & (np.abs(z2) < c)
    if int(np.sum(mask)) < 10:
        c = np.inf
        mask = np.ones(n_total, dtype=bool)

    n0 = int(np.sum(mask))
    if n0 < 3:
        return 0.0, n0

    z1_0 = z1[mask]
    z2_0 = z2[mask]
    s11 = float(np.dot(z1_0, z1_0))
    s22 = float(np.dot(z2_0, z2_0))
    s12 = float(np.dot(z1_0, z2_0))

    gl_x, gl_w = np.polynomial.legendre.leggauss(32)

    def neg_ll(rho: float) -> float:
        one_minus_r2 = 1.0 - rho * rho
        ll = -0.5 * n0 * np.log(one_minus_r2) - (s11 - 2.0 * rho * s12 + s22) / (
            2.0 * one_minus_r2
        )
        if np.isfinite(c):
            sd_cond = np.sqrt(one_minus_r2)
            t = c * gl_x
            phi_t = stats.norm.pdf(t)
            cond_prob = stats.norm.cdf((c - rho * t) / sd_cond) - stats.norm.cdf(
                (-c - rho * t) / sd_cond
            )
            p_rect = max(float(c * np.sum(gl_w * phi_t * cond_prob)), 1e-300)
            ll -= n0 * np.log(p_rect)
        return -ll

    opt = optimize.minimize_scalar(neg_ll, bounds=(-0.98, 0.98), method="bounded")
    return float(opt.x), n0


def _build_1d_var_grid(
    beta: np.ndarray,
    se: np.ndarray,
    grid_mult: float,
    max_grid_1d: int,
) -> np.ndarray:
    """Construct `ash`-style unimodal 1D variance grid `{0} U {sigma_r^2}`."""
    sigma_min = max(float(np.median(se)) / 10.0, 1e-12)
    excess_var = float(np.max(beta * beta - se * se))
    sigma_max = 2.0 * np.sqrt(max(excess_var, (4.0 * sigma_min) ** 2))
    ratio = max(sigma_max / sigma_min, 1.0)
    mult = max(float(grid_mult), ratio ** (1.0 / max(max_grid_1d - 2, 1)))
    n_pos = min(
        max_grid_1d - 1,
        max(2, int(np.ceil(np.log(ratio) / np.log(mult))) + 1),
    )
    sigmas = np.concatenate(
        ([0.0], sigma_min * (mult ** np.arange(n_pos, dtype=np.float64)))
    )
    return sigmas * sigmas


def est_cause_params(
    data: pd.DataFrame | None = None,
    variants: Sequence[str] | np.ndarray | None = None,
    *,
    beta_hat_1: Sequence[float] | np.ndarray | str | None = None,
    seb1: Sequence[float] | np.ndarray | str | None = None,
    beta_hat_2: Sequence[float] | np.ndarray | str | None = None,
    seb2: Sequence[float] | np.ndarray | str | None = None,
    snp: Sequence[str] | str | None = None,
    null_p_thresh: float = 0.05,
    null_wt: float = 10.0,
    grid_mult: float = float(np.sqrt(2.0)),
    max_grid_1d: int = 16,
    max_variants: int = 1_000_000,
    rho: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    weight_prune_thresh: float = 1e-10,
    seed: int | np.random.Generator | None = 0,
) -> CauseParams:
    """Estimate CAUSE genome-wide nuisance parameters (`rho` and bivariate `ash` mixture).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Summary statistics table (or pass 1D arrays `beta_hat_1`, `seb1`, `beta_hat_2`, `seb2`).
    variants : sequence of str, optional
        Optional subset of SNP identifiers to include.
    null_p_thresh : float, default 0.05
        Two-sided p-value threshold defining null SNPs (`p1 > null_p_thresh` and
        `p2 > null_p_thresh`) for truncated bivariate normal `rho` MLE.
    null_wt : float, default 10.0
        Dirichlet concentration prior weight on the `(0, 0)` null variance component.
    grid_mult : float, default sqrt(2)
        Multiplicative spacing between consecutive `ash` standard deviation grid points.
    max_grid_1d : int, default 16
        Maximum number of 1D variance components per trait (including `0.0`).
    max_variants : int, default 1_000_000
        Maximum number of SNPs to use; larger inputs are subsampled deterministically.
    rho : float, optional
        Optional fixed sample-overlap correlation in `(-1, 1)`. If `None`, estimated via MLE.
    seed : int or np.random.Generator or None, default 0
        Random seed used only if `N > max_variants` subsampling is triggered.
    """
    if not (0.0 < null_p_thresh < 1.0):
        raise ValueError(f"null_p_thresh must be in (0, 1) (got {null_p_thresh!r}).")
    if null_wt < 1.0:
        raise ValueError(f"null_wt must be >= 1.0 (got {null_wt!r}).")
    if grid_mult <= 1.0:
        raise ValueError(f"grid_mult must be > 1.0 (got {grid_mult!r}).")
    if max_grid_1d < 4:
        raise ValueError(f"max_grid_1d must be >= 4 (got {max_grid_1d!r}).")
    if max_variants < 1:
        raise ValueError(f"max_variants must be >= 1 (got {max_variants!r}).")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 (got {max_iter!r}).")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0 (got {tol!r}).")

    snp_ids, b1, s1, b2, s2 = _extract_cause_arrays(
        data,
        beta_hat_1=beta_hat_1,
        seb1=seb1,
        beta_hat_2=beta_hat_2,
        seb2=seb2,
        snp=snp,
    )

    if variants is not None:
        var_set = {str(v) for v in variants}
        keep = np.array([sid in var_set for sid in snp_ids], dtype=bool)
        if not np.any(keep):
            raise ValueError("No SNPs matched the provided `variants` subset.")
        b1 = b1[keep]
        s1 = s1[keep]
        b2 = b2[keep]
        s2 = s2[keep]

    n_snps = b1.size
    if n_snps > max_variants:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_snps, size=max_variants, replace=False))
        b1 = b1[idx]
        s1 = s1[idx]
        b2 = b2[idx]
        s2 = s2[idx]
        n_snps = max_variants

    b1, b2 = _canonicalize_signs(b1, b2)

    if rho is not None:
        rho_val = float(rho)
        if not np.isfinite(rho_val) or not (-1.0 < rho_val < 1.0):
            raise ValueError(f"rho must lie strictly in (-1, 1) (got {rho!r}).")
        rho_hat = rho_val
        n_null_rho = n_snps
    else:
        z1 = b1 / s1
        z2 = b2 / s2
        rho_hat, n_null_rho = _estimate_rho_trunc_mle(z1, z2, null_p_thresh)

    var1 = _build_1d_var_grid(b1, s1, grid_mult=grid_mult, max_grid_1d=max_grid_1d)
    var2 = _build_1d_var_grid(b2, s2, grid_mult=grid_mult, max_grid_1d=max_grid_1d)
    s1_grid, s2_grid = np.meshgrid(var1, var2, indexing="ij")
    s1_flat = s1_grid.ravel()
    s2_flat = s2_grid.ravel()
    n_comp = s1_flat.size

    # Cancellation-free log-likelihood matrix (N, K)
    s1_sq = (s1 * s1)[:, None]
    s2_sq = (s2 * s2)[:, None]
    b1_sq = (b1 * b1)[:, None]
    b2_sq = (b2 * b2)[:, None]
    b1_b2_s1_s2 = (s1 * s2 * b1 * b2)[:, None]

    det = (
        s1_sq * s2_sq * (1.0 - rho_hat * rho_hat)
        + s1_flat[None, :] * s2_sq
        + s2_flat[None, :] * s1_sq
        + (s1_flat * s2_flat)[None, :]
    )
    det = np.maximum(det, 1e-300)

    quad = (
        (s2_flat[None, :] + s2_sq) * b1_sq
        - 2.0 * rho_hat * b1_b2_s1_s2
        + (s1_flat[None, :] + s1_sq) * b2_sq
    ) / det

    log_l = -np.log(2.0 * np.pi) - 0.5 * np.log(det) - 0.5 * quad
    row_max = np.max(log_l, axis=1, keepdims=True)
    l_tilde = np.exp(log_l - row_max)
    row_max_1d = row_max[:, 0]

    # EM with SQUAREM acceleration and monotone fallback
    pi = np.full(n_comp, 0.5 / max(n_comp - 1, 1), dtype=np.float64)
    pi[0] = 0.5
    pi /= np.sum(pi)

    denom_const = float(n_snps) + (null_wt - 1.0)

    def _em_step(pi_cur: np.ndarray) -> np.ndarray:
        denom = np.maximum(l_tilde @ pi_cur, 1e-300)
        resp_sum = (l_tilde.T @ (1.0 / denom)) * pi_cur
        resp_sum[0] += null_wt - 1.0
        pi_new = np.maximum(resp_sum, 0.0) / denom_const
        pi_new /= np.sum(pi_new)
        return pi_new

    def _objective(pi_cur: np.ndarray) -> float:
        ll = float(np.sum(np.log(np.maximum(l_tilde @ pi_cur, 1e-300)) + row_max_1d))
        if null_wt > 1.0:
            ll += (null_wt - 1.0) * float(np.log(max(pi_cur[0], 1e-300)))
        return ll

    converged = False
    n_iter = 0
    obj_cur = _objective(pi)
    for it in range(1, max_iter + 1):
        n_iter = it
        p1 = _em_step(pi)
        if float(np.max(np.abs(p1 - pi))) < tol:
            pi = p1
            converged = True
            break
        p2 = _em_step(p1)
        obj_p2 = _objective(p2)
        r_vec = p1 - pi
        v_vec = (p2 - p1) - r_vec
        v_norm2 = float(np.dot(v_vec, v_vec))
        p_next = p2
        obj_next = obj_p2
        if v_norm2 > 1e-30:
            r_norm2 = float(np.dot(r_vec, r_vec))
            alpha = -np.sqrt(r_norm2 / v_norm2)
            alpha = min(-1.0, max(alpha, -50.0))
            p_cand = np.maximum(pi - 2.0 * alpha * r_vec + (alpha * alpha) * v_vec, 0.0)
            cand_sum = float(np.sum(p_cand))
            if cand_sum > 0.0:
                p_cand /= cand_sum
                p_cand = _em_step(p_cand)
                obj_cand = _objective(p_cand)
                if obj_cand >= obj_p2:
                    p_next = p_cand
                    obj_next = obj_cand
        if abs(obj_next - obj_cur) / float(n_snps) < tol or float(np.max(np.abs(p_next - pi))) < tol:
            pi = p_next
            converged = True
            break
        pi = p_next
        obj_cur = obj_next

    ll_final = float(np.sum(np.log(np.maximum(l_tilde @ pi, 1e-300)) + row_max_1d))

    keep_comp = pi >= weight_prune_thresh
    if pi[0] > 0.0:
        keep_comp[0] = True
    if not np.any(keep_comp):
        keep_comp[int(np.argmax(pi))] = True

    s1_active = s1_flat[keep_comp]
    s2_active = s2_flat[keep_comp]
    pi_active = pi[keep_comp]
    pi_active /= np.sum(pi_active)

    return CauseParams(
        rho=float(rho_hat),
        S1=s1_active,
        S2=s2_active,
        pi=pi_active,
        n_snps=n_snps,
        n_null_snps_rho=n_null_rho,
        log_likelihood=ll_final,
        converged=converged,
        n_iter=n_iter,
    )


def _weighted_summary(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return `(median, ci_low, ci_high, mean, sd)` for weighted discrete draws."""
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(wts))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        wts = np.full_like(vals, 1.0 / vals.size)
    else:
        wts = wts / w_sum

    mean = float(np.sum(wts * vals))
    sd = float(np.sqrt(max(float(np.sum(wts * (vals - mean) ** 2)), 0.0)))

    if np.ptp(vals) <= 1e-15:
        v0 = float(vals[0])
        return v0, v0, v0, mean, 0.0

    order = np.argsort(vals, kind="mergesort")
    v_s = vals[order]
    w_s = wts[order]

    # Collapse exact duplicate values so CDF jumps are accurately centered
    is_new = np.empty(v_s.size, dtype=bool)
    is_new[0] = True
    is_new[1:] = v_s[1:] != v_s[:-1]
    idx_starts = np.where(is_new)[0]
    v_u = v_s[idx_starts]
    w_u = np.add.reduceat(w_s, idx_starts)

    if v_u.size == 1:
        v0 = float(v_u[0])
        return v0, v0, v0, mean, sd

    cdf = np.cumsum(w_u) - 0.5 * w_u
    q_low, q_med, q_high = np.interp([0.025, 0.50, 0.975], cdf, v_u)
    return float(q_med), float(q_low), float(q_high), mean, sd


def _systematic_resample(
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `n_samples` indices via low-variance systematic resampling."""
    u0 = float(rng.uniform(0.0, 1.0 / float(n_samples)))
    u = u0 + np.arange(n_samples, dtype=np.float64) / float(n_samples)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, u, side="left")
    return np.minimum(idx, weights.size - 1)


def cause(
    data: pd.DataFrame | None = None,
    param_ests: CauseParams | None = None,
    variants: Sequence[str] | np.ndarray | None = None,
    *,
    beta_hat_1: Sequence[float] | np.ndarray | str | None = None,
    seb1: Sequence[float] | np.ndarray | str | None = None,
    beta_hat_2: Sequence[float] | np.ndarray | str | None = None,
    seb2: Sequence[float] | np.ndarray | str | None = None,
    snp: Sequence[str] | str | None = None,
    pval_thresh: float | None = None,
    qalpha: float = 1.0,
    qbeta: float = 10.0,
    max_q: float = 1.0,
    sigma_g: float | None = None,
    n_grid: int = 31,
    n_q: int = 21,
    refine_factor: int = 5,
    refine_mass_thresh: float = 1e-5,
    n_samples: int = 2000,
    seed: int | np.random.Generator | None = 0,
) -> CauseResult:
    """Fit CAUSE sharing vs causal Bayesian models and compare via PSIS-LOO.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Summary statistics table (or pass 1D arrays `beta_hat_1`, `seb1`, `beta_hat_2`, `seb2`).
    param_ests : CauseParams, optional
        Pre-estimated genome-wide nuisance parameters from `est_cause_params`.
        If `None`, `est_cause_params` is automatically run on the full input data
        prior to instrument filtering (`variants` / `pval_thresh`).
    variants : sequence of str, optional
        Instrument SNP IDs to include in posterior fitting and PSIS-LOO comparison.
    pval_thresh : float, optional
        Optional exposure p-value threshold (`p1 <= pval_thresh`) for instrument selection.
    qalpha, qbeta : float, default (1.0, 10.0)
        Beta prior shape parameters on the correlated pleiotropy proportion `q`.
    max_q : float, default 1.0
        Upper bound for quadrature integration over `q in (0, max_q]`.
    sigma_g : float, optional
        Prior SD for `gamma` and `eta`. If `None`, chosen scale-equivariantly from
        strong-instrument effect ratios.
    n_grid : int, default 31
        Coarse 1D grid points for `gamma` and `eta` (coerced to odd >= 11 so `0` is centered).
    n_q : int, default 21
        Midpoint quadrature points on `(0, max_q)`.
    refine_factor : int, default 5
        Stage-2 adaptive subdivision factor per selected coarse cell (coerced to odd >= 3).
    refine_mass_thresh : float, default 1e-5
        Coarse cell posterior mass threshold triggering adaptive subdivision.
    n_samples : int, default 2000
        Number of systematic posterior draws used to compute PSIS-LOO ELPDs.
    seed : int or np.random.Generator or None, default 0
        Random seed controlling systematic resampling offset and nuisance subsampling.

    Returns
    -------
    CauseResult
    """
    if qalpha <= 0.0 or qbeta <= 0.0:
        raise ValueError(f"qalpha and qbeta must be > 0 (got {qalpha!r}, {qbeta!r}).")
    if not (0.0 < max_q <= 1.0):
        raise ValueError(f"max_q must be in (0, 1] (got {max_q!r}).")
    if n_grid < 11:
        raise ValueError(f"n_grid must be >= 11 (got {n_grid!r}).")
    if n_q < 5:
        raise ValueError(f"n_q must be >= 5 (got {n_q!r}).")
    if refine_factor < 3:
        raise ValueError(f"refine_factor must be >= 3 (got {refine_factor!r}).")
    if n_samples < 50:
        raise ValueError(f"n_samples must be >= 50 (got {n_samples!r}).")
    if pval_thresh is not None and not (0.0 < pval_thresh <= 1.0):
        raise ValueError(f"pval_thresh must be in (0, 1] (got {pval_thresh!r}).")

    if n_grid % 2 == 0:
        n_grid += 1
    if refine_factor % 2 == 0:
        refine_factor += 1

    snp_ids_all, b1_all, s1_all, b2_all, s2_all = _extract_cause_arrays(
        data,
        beta_hat_1=beta_hat_1,
        seb1=seb1,
        beta_hat_2=beta_hat_2,
        seb2=seb2,
        snp=snp,
    )

    if param_ests is None:
        param_ests = est_cause_params(
            beta_hat_1=b1_all,
            seb1=s1_all,
            beta_hat_2=b2_all,
            seb2=s2_all,
            snp=snp_ids_all,
            seed=seed,
        )

    keep = np.ones(b1_all.size, dtype=bool)
    if variants is not None:
        var_set = {str(v) for v in variants}
        keep &= np.array([sid in var_set for sid in snp_ids_all], dtype=bool)
    if pval_thresh is not None:
        p1_all = 2.0 * stats.norm.sf(np.abs(b1_all / s1_all))
        keep &= p1_all <= pval_thresh

    idx_keep = np.where(keep)[0]
    if idx_keep.size < 2:
        raise ValueError(
            "At least 2 instruments are required for CAUSE PSIS-LOO SE estimation."
        )

    snp_ids = [snp_ids_all[i] for i in idx_keep]
    b1_orig = b1_all[idx_keep]
    s1 = s1_all[idx_keep]
    b2_orig = b2_all[idx_keep]
    s2 = s2_all[idx_keep]
    n_inst = b1_orig.size

    b1, b2 = _canonicalize_signs(b1_orig, b2_orig)

    # Scale-equivariant prior SD sigma_g and grid half-width B
    z1 = np.abs(b1 / s1)
    inf_mask = z1 >= 2.0
    if int(np.sum(inf_mask)) < 5:
        k_top = max(1, int(np.ceil(0.25 * n_inst)))
        thresh = float(np.partition(z1, n_inst - k_top)[n_inst - k_top])
        inf_mask = z1 >= thresh
    r_ratios = np.abs(b2[inf_mask]) / np.maximum(np.abs(b1[inf_mask]), s1[inf_mask])

    if sigma_g is None:
        sigma_g_val = max(
            float(np.quantile(r_ratios, 0.80)) / 0.6745,
            0.5 * float(np.median(s2 / s1)),
            1e-12,
        )
    else:
        sigma_g_val = float(sigma_g)
        if not np.isfinite(sigma_g_val) or sigma_g_val <= 0.0:
            raise ValueError(f"sigma_g must be strictly positive (got {sigma_g!r}).")

    bound = max(3.5 * sigma_g_val, 1.25 * float(np.quantile(r_ratios, 0.95)))

    # Precompute nuisance arrays for 1D slope mixture evaluation
    rho = float(param_ests.rho)
    s1_mix = np.asarray(param_ests.S1, dtype=np.float64)[None, None, :]
    s2_mix = np.asarray(param_ests.S2, dtype=np.float64)[None, None, :]
    log_pi_mix = np.log(np.maximum(np.asarray(param_ests.pi, dtype=np.float64), 1e-300))[
        None, None, :
    ]

    s1_sq = (s1 * s1)[None, :, None]
    s2_sq = (s2 * s2)[None, :, None]
    s1_s2 = (s1 * s2)[None, :, None]
    b1_sq = (b1 * b1)[None, :, None]
    b2_sq = (b2 * b2)[None, :, None]
    b1_b2 = (b1 * b2)[None, :, None]
    base_det = s1_sq * s2_sq * (1.0 - rho * rho) + s2_mix * (s1_sq + s1_mix)
    log_2pi = np.log(2.0 * np.pi)

    delta_0 = 2.0 * bound / float(n_grid - 1)
    delta_fine = delta_0 / float(refine_factor)
    half = (n_grid - 1) // 2
    idx_1d = np.arange(-half, half + 1, dtype=np.int64)
    fine_offsets = np.arange(-(refine_factor // 2), refine_factor // 2 + 1, dtype=np.int64)

    slope_cache: dict[int, np.ndarray] = {}

    def _ensure_slopes(keys: Sequence[int] | np.ndarray) -> None:
        missing = sorted({int(k) for k in keys if int(k) not in slope_cache})
        if not missing:
            return
        m_all = np.asarray(missing, dtype=np.float64) * delta_fine
        chunk_size = 64
        for start in range(0, m_all.size, chunk_size):
            m_chunk = m_all[start : start + chunk_size]
            m_col = m_chunk[:, None, None]
            det = np.maximum(
                base_det
                + s1_mix * (s2_sq + (m_col * m_col) * s1_sq - 2.0 * rho * m_col * s1_s2),
                1e-300,
            )
            quad = (
                ((m_col * m_col) * s1_mix + s2_mix + s2_sq) * b1_sq
                - 2.0 * (m_col * s1_mix + rho * s1_s2) * b1_b2
                + (s1_mix + s1_sq) * b2_sq
            ) / det
            comp_ll = log_pi_mix - log_2pi - 0.5 * np.log(det) - 0.5 * quad
            ll_chunk = special.logsumexp(comp_ll, axis=2)  # (chunk, J)
            for idx_local, key in enumerate(missing[start : start + chunk_size]):
                slope_cache[key] = ll_chunk[idx_local]

    # q quadrature grid (strictly interior midpoint rule)
    dq = max_q / float(n_q)
    q_grid = max_q * (np.arange(n_q, dtype=np.float64) + 0.5) / float(n_q)
    log_prior_q = stats.beta.logpdf(q_grid, qalpha, qbeta)
    log_1m_q = np.log1p(-q_grid)
    log_q = np.log(q_grid)

    # ------------------------------------------------------------------
    # 1. Sharing Model (gamma = 0) — 2D Adaptive Quadrature over (eta, q)
    # ------------------------------------------------------------------
    coarse_keys_1d = idx_1d * refine_factor
    _ensure_slopes(np.concatenate(([0], coarse_keys_1d)))
    ll_0 = slope_cache[0]  # (J,)

    ll_eta_coarse = np.stack([slope_cache[int(k)] for k in coarse_keys_1d], axis=0)  # (G, J)
    eta_coarse_vals = coarse_keys_1d.astype(np.float64) * delta_fine
    log_prior_eta_coarse = stats.norm.logpdf(eta_coarse_vals, loc=0.0, scale=sigma_g_val)

    # Evaluate coarse sharing posterior: (G, n_q)
    ll_mix_sh_c = np.logaddexp(
        log_1m_q[None, :, None] + ll_0[None, None, :],
        log_q[None, :, None] + ll_eta_coarse[:, None, :],
    )
    log_post_sh_c = (
        np.sum(ll_mix_sh_c, axis=2)
        + log_prior_eta_coarse[:, None]
        + log_prior_q[None, :]
        + np.log(delta_0 * dq)
    )
    w_sh_c = np.exp(log_post_sh_c - float(special.logsumexp(log_post_sh_c)))
    marg_eta_c = np.sum(w_sh_c, axis=1)

    sel_sh = marg_eta_c > (refine_mass_thresh / float(n_grid))
    top_sh = np.argsort(marg_eta_c, kind="mergesort")[-3:]
    sel_sh[top_sh] = True

    sh_keys_list: list[int] = []
    sh_widths_list: list[float] = []
    for b_idx in range(n_grid):
        ck = int(coarse_keys_1d[b_idx])
        if sel_sh[b_idx]:
            for d in fine_offsets:
                sh_keys_list.append(ck + int(d))
                sh_widths_list.append(delta_fine)
        else:
            sh_keys_list.append(ck)
            sh_widths_list.append(delta_0)

    sh_keys = np.asarray(sh_keys_list, dtype=np.int64)
    sh_widths = np.asarray(sh_widths_list, dtype=np.float64)
    _ensure_slopes(sh_keys)

    ll_eta_sh = np.stack([slope_cache[int(k)] for k in sh_keys], axis=0)  # (N_eta, J)
    eta_vals_sh = sh_keys.astype(np.float64) * delta_fine
    log_post_sh = (
        np.sum(
            np.logaddexp(
                log_1m_q[None, :, None] + ll_0[None, None, :],
                log_q[None, :, None] + ll_eta_sh[:, None, :],
            ),
            axis=2,
        )
        + stats.norm.logpdf(eta_vals_sh, loc=0.0, scale=sigma_g_val)[:, None]
        + log_prior_q[None, :]
        + np.log(sh_widths[:, None] * dq)
    )
    w_sh = np.exp(log_post_sh - float(special.logsumexp(log_post_sh)))  # (N_eta, n_q)

    # Exact summaries for sharing model
    eta_grid_sh = np.broadcast_to(eta_vals_sh[:, None], w_sh.shape).ravel()
    q_grid_sh = np.broadcast_to(q_grid[None, :], w_sh.shape).ravel()
    w_sh_flat = w_sh.ravel()

    eta_med_sh, eta_lo_sh, eta_hi_sh, eta_mean_sh, eta_sd_sh = _weighted_summary(
        eta_grid_sh, w_sh_flat
    )
    q_med_sh, q_lo_sh, q_hi_sh, q_mean_sh, q_sd_sh = _weighted_summary(
        q_grid_sh, w_sh_flat
    )

    # Per-SNP Pr(Z_j = 1 | data, sharing)
    log_num_sh = log_q[None, :, None] + ll_eta_sh[:, None, :]
    log_den_sh = np.logaddexp(
        log_1m_q[None, :, None] + ll_0[None, None, :],
        log_num_sh,
    )
    pr_z_sh = np.sum(w_sh[:, :, None] * np.exp(log_num_sh - log_den_sh), axis=(0, 1))

    # ------------------------------------------------------------------
    # 2. Causal Model (gamma, eta, q) — 3D Adaptive Quadrature
    # ------------------------------------------------------------------
    sum_keys_c = (idx_1d[:, None] + idx_1d[None, :]) * refine_factor
    _ensure_slopes(sum_keys_c.ravel())

    log_prior_1d_c = stats.norm.logpdf(eta_coarse_vals, loc=0.0, scale=sigma_g_val)
    log_area_c = np.log((delta_0 * delta_0) * dq)
    log_post_ca_c = np.empty((n_grid, n_grid, n_q), dtype=np.float64)

    for a_idx in range(n_grid):
        ll_gam = slope_cache[int(coarse_keys_1d[a_idx])]  # (J,)
        ll_sum = np.stack(
            [slope_cache[int(sum_keys_c[a_idx, b_idx])] for b_idx in range(n_grid)],
            axis=0,
        )  # (G, J)
        ll_mix_a = np.logaddexp(
            log_1m_q[None, :, None] + ll_gam[None, None, :],
            log_q[None, :, None] + ll_sum[:, None, :],
        )  # (G, n_q, J)
        log_post_ca_c[a_idx] = (
            np.sum(ll_mix_a, axis=2)
            + log_prior_1d_c[a_idx]
            + log_prior_1d_c[:, None]
            + log_prior_q[None, :]
            + log_area_c
        )

    w_ca_c = np.exp(log_post_ca_c - float(special.logsumexp(log_post_ca_c)))
    w_ca_ab = np.sum(w_ca_c, axis=2)  # (G, G)
    w_ca_ab_flat = w_ca_ab.ravel()

    sel_ca_flat = w_ca_ab_flat > (refine_mass_thresh / float(n_grid * n_grid))
    top9_idx = np.argsort(w_ca_ab_flat, kind="mergesort")[-9:]
    sel_ca_flat[top9_idx] = True

    # Cap refined coarse cells at 150 highest-mass cells so diffuse posteriors stay fast
    max_refine_cells = 150
    if int(np.sum(sel_ca_flat)) > max_refine_cells:
        top_refine = np.argsort(w_ca_ab_flat, kind="mergesort")[-max_refine_cells:]
        sel_ca_flat = np.zeros_like(sel_ca_flat)
        sel_ca_flat[top_refine] = True

    sel_ca = sel_ca_flat.reshape((n_grid, n_grid))

    ca_gam_keys_list: list[int] = []
    ca_eta_keys_list: list[int] = []
    ca_areas_list: list[float] = []
    area_fine = delta_fine * delta_fine
    area_coarse = delta_0 * delta_0

    for a_idx in range(n_grid):
        g_ck = int(coarse_keys_1d[a_idx])
        for b_idx in range(n_grid):
            e_ck = int(coarse_keys_1d[b_idx])
            if sel_ca[a_idx, b_idx]:
                for dg in fine_offsets:
                    gk = g_ck + int(dg)
                    for de in fine_offsets:
                        ca_gam_keys_list.append(gk)
                        ca_eta_keys_list.append(e_ck + int(de))
                        ca_areas_list.append(area_fine)
            elif w_ca_ab[a_idx, b_idx] > 1e-10:
                ca_gam_keys_list.append(g_ck)
                ca_eta_keys_list.append(e_ck)
                ca_areas_list.append(area_coarse)

    ca_gam_keys = np.asarray(ca_gam_keys_list, dtype=np.int64)
    ca_eta_keys = np.asarray(ca_eta_keys_list, dtype=np.int64)
    ca_sum_keys = ca_gam_keys + ca_eta_keys
    ca_areas = np.asarray(ca_areas_list, dtype=np.float64)

    _ensure_slopes(np.concatenate((ca_gam_keys, ca_sum_keys)))

    ca_gam_vals = ca_gam_keys.astype(np.float64) * delta_fine
    ca_eta_vals = ca_eta_keys.astype(np.float64) * delta_fine
    n_ca_cells = ca_gam_keys.size

    log_post_ca = np.empty((n_ca_cells, n_q), dtype=np.float64)
    log_prior_ca_cells = (
        stats.norm.logpdf(ca_gam_vals, loc=0.0, scale=sigma_g_val)
        + stats.norm.logpdf(ca_eta_vals, loc=0.0, scale=sigma_g_val)
        + np.log(ca_areas * dq)
    )

    chunk_cells = 256
    for start in range(0, n_ca_cells, chunk_cells):
        end = min(start + chunk_cells, n_ca_cells)
        ll_g_blk = np.stack([slope_cache[int(k)] for k in ca_gam_keys[start:end]], axis=0)
        ll_s_blk = np.stack([slope_cache[int(k)] for k in ca_sum_keys[start:end]], axis=0)
        ll_mix_blk = np.logaddexp(
            log_1m_q[None, :, None] + ll_g_blk[:, None, :],
            log_q[None, :, None] + ll_s_blk[:, None, :],
        )  # (blk, n_q, J)
        log_post_ca[start:end] = (
            np.sum(ll_mix_blk, axis=2)
            + log_prior_ca_cells[start:end, None]
            + log_prior_q[None, :]
        )

    w_ca = np.exp(log_post_ca - float(special.logsumexp(log_post_ca)))  # (n_ca_cells, n_q)

    # Accumulate per-SNP Pr(Z_j = 1 | data, causal) in chunks to bound memory
    pr_z_ca = np.zeros(n_inst, dtype=np.float64)
    for start in range(0, n_ca_cells, chunk_cells):
        end = min(start + chunk_cells, n_ca_cells)
        ll_g_blk = np.stack([slope_cache[int(k)] for k in ca_gam_keys[start:end]], axis=0)
        ll_s_blk = np.stack([slope_cache[int(k)] for k in ca_sum_keys[start:end]], axis=0)
        log_num_blk = log_q[None, :, None] + ll_s_blk[:, None, :]
        log_den_blk = np.logaddexp(log_1m_q[None, :, None] + ll_g_blk[:, None, :], log_num_blk)
        pr_z_ca += np.sum(
            w_ca[start:end, :, None] * np.exp(log_num_blk - log_den_blk), axis=(0, 1)
        )

    gam_grid_ca = np.broadcast_to(ca_gam_vals[:, None], w_ca.shape).ravel()
    eta_grid_ca = np.broadcast_to(ca_eta_vals[:, None], w_ca.shape).ravel()
    q_grid_ca = np.broadcast_to(q_grid[None, :], w_ca.shape).ravel()
    w_ca_flat = w_ca.ravel()

    gam_med_ca, gam_lo_ca, gam_hi_ca, gam_mean_ca, gam_sd_ca = _weighted_summary(
        gam_grid_ca, w_ca_flat
    )
    eta_med_ca, eta_lo_ca, eta_hi_ca, eta_mean_ca, eta_sd_ca = _weighted_summary(
        eta_grid_ca, w_ca_flat
    )
    q_med_ca, q_lo_ca, q_hi_ca, q_mean_ca, q_sd_ca = _weighted_summary(
        q_grid_ca, w_ca_flat
    )

    # ------------------------------------------------------------------
    # 3. Systematic Posterior Resampling & PSIS-LOO Model Comparison
    # ------------------------------------------------------------------
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    draw_idx_sh = _systematic_resample(w_sh_flat, n_samples, rng)
    sh_cell_idx, sh_q_idx = np.divmod(draw_idx_sh, n_q)
    ll_draws_sh = np.logaddexp(
        log_1m_q[sh_q_idx, None] + ll_0[None, :],
        log_q[sh_q_idx, None] + ll_eta_sh[sh_cell_idx],
    )

    draw_idx_ca = _systematic_resample(w_ca_flat, n_samples, rng)
    ca_cell_idx, ca_q_idx = np.divmod(draw_idx_ca, n_q)
    ll_g_draws = np.stack([slope_cache[int(ca_gam_keys[i])] for i in ca_cell_idx], axis=0)
    ll_s_draws = np.stack([slope_cache[int(ca_sum_keys[i])] for i in ca_cell_idx], axis=0)
    ll_draws_ca = np.logaddexp(
        log_1m_q[ca_q_idx, None] + ll_g_draws,
        log_q[ca_q_idx, None] + ll_s_draws,
    )

    loo_null = psis_loo(ll_0[None, :])
    loo_sh = psis_loo(ll_draws_sh)
    loo_ca = psis_loo(ll_draws_ca)

    elpd_table = compare_elpd({"null": loo_null, "sharing": loo_sh, "causal": loo_ca})
    sh_vs_ca = elpd_table.loc[
        (elpd_table["model1"] == "sharing") & (elpd_table["model2"] == "causal")
    ].iloc[0]

    sharing_fit = CauseModelFit(
        model_name="sharing",
        gamma_median=0.0,
        gamma_ci_low=0.0,
        gamma_ci_high=0.0,
        gamma_mean=0.0,
        gamma_sd=0.0,
        eta_median=eta_med_sh,
        eta_ci_low=eta_lo_sh,
        eta_ci_high=eta_hi_sh,
        eta_mean=eta_mean_sh,
        eta_sd=eta_sd_sh,
        q_median=q_med_sh,
        q_ci_low=q_lo_sh,
        q_ci_high=q_hi_sh,
        q_mean=q_mean_sh,
        q_sd=q_sd_sh,
        posterior_sharing_prob=pr_z_sh,
        log_lik_draws=ll_draws_sh,
        loo=loo_sh,
    )

    causal_fit = CauseModelFit(
        model_name="causal",
        gamma_median=gam_med_ca,
        gamma_ci_low=gam_lo_ca,
        gamma_ci_high=gam_hi_ca,
        gamma_mean=gam_mean_ca,
        gamma_sd=gam_sd_ca,
        eta_median=eta_med_ca,
        eta_ci_low=eta_lo_ca,
        eta_ci_high=eta_hi_ca,
        eta_mean=eta_mean_ca,
        eta_sd=eta_sd_ca,
        q_median=q_med_ca,
        q_ci_low=q_lo_ca,
        q_ci_high=q_hi_ca,
        q_mean=q_mean_ca,
        q_sd=q_sd_ca,
        posterior_sharing_prob=pr_z_ca,
        log_lik_draws=ll_draws_ca,
        loo=loo_ca,
    )

    return CauseResult(
        n_instruments=n_inst,
        snp_ids=snp_ids,
        beta_hat_1=b1_orig.copy(),
        seb1=s1.copy(),
        beta_hat_2=b2_orig.copy(),
        seb2=s2.copy(),
        params=param_ests,
        sigma_g=sigma_g_val,
        qalpha=float(qalpha),
        qbeta=float(qbeta),
        max_q=float(max_q),
        elpd_null=float(loo_null.elpd),
        elpd_sharing=float(loo_sh.elpd),
        elpd_causal=float(loo_ca.elpd),
        delta_elpd=float(sh_vs_ca["delta_elpd"]),
        se_delta_elpd=float(sh_vs_ca["se_delta_elpd"]),
        z_score=float(sh_vs_ca["z"]),
        p_value=float(sh_vs_ca["p_value"]),
        sharing=sharing_fit,
        causal=causal_fit,
        loo_null=loo_null,
        elpd_table=elpd_table,
    )
