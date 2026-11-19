"""
Core fishHook Gamma-Poisson (Negative Binomial Type-2) somatic mutation
enrichment regression (Imielinski, Guo & Meyerson, Cell 2017).

Reference
---------
Imielinski, M., Guo, W., & Meyerson, M. (2017).
Insertions and Deletions Target Lineage-Defining Genes in Human Cancers.
Cell, 168(3), 460-472.e14. https://doi.org/10.1016/j.cell.2016.12.025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .intervals import compute_territory_and_counts

_MAX_LOG_THETA = 12.5  # theta ~ 2.68e5 -> alpha ~ 3.7e-6 (Poisson regime)
_MIN_LOG_THETA = -14.0  # theta ~ 8.3e-7

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class FishHookResult:
    """Fitted fishHook Gamma-Poisson background model and region enrichment statistics."""

    region_ids: list[str]
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    eligible_bp: np.ndarray
    obs_counts: np.ndarray
    exp_counts: np.ndarray
    obs_density: np.ndarray
    exp_density: np.ndarray
    fold_enrichment: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    coef_names: list[str]
    beta: np.ndarray
    se_beta: np.ndarray
    z_beta: np.ndarray
    p_beta: np.ndarray
    alpha: float
    theta: float
    is_poisson_fallback: bool
    loglik: float
    lambda_gc: float
    qq_slope: float
    converged: bool
    n_iter: int
    p_neg: np.ndarray | None = None
    fdr_neg: np.ndarray | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return region-level enrichment (and optional depletion) statistics as a DataFrame."""
        data: dict[str, object] = {
            "region_id": self.region_ids,
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "eligible_bp": self.eligible_bp,
            "obs_counts": self.obs_counts,
            "exp_counts": self.exp_counts,
            "obs_density": self.obs_density,
            "exp_density": self.exp_density,
            "fold_enrichment": self.fold_enrichment,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }
        if self.p_neg is not None:
            data["p_neg"] = self.p_neg
        if self.fdr_neg is not None:
            data["fdr_neg"] = self.fdr_neg
        return pd.DataFrame(data)

    def coefficients_dataframe(self) -> pd.DataFrame:
        """Return fitted covariate regression coefficients and Wald tests."""
        return pd.DataFrame(
            {
                "beta": self.beta,
                "se": self.se_beta,
                "z": self.z_beta,
                "p_value": self.p_beta,
            },
            index=self.coef_names,
        )

    def significant(self, alpha: float = 0.05) -> pd.DataFrame:
        df = self.to_dataframe()
        return (
            df.loc[df["q_value"].notna() & (df["q_value"] <= alpha)]
            .sort_values("p_value")
            .reset_index(drop=True)
        )


class FishHookModel:
    """
    Object-oriented fishHook Gamma-Poisson model specification.

    Mirrors the R `Fish(hypotheses=..., events=..., eligible=..., covariates=...)`
    workflow while returning a typed `FishHookResult`.
    """

    def __init__(
        self,
        regions: pd.DataFrame,
        mutations: pd.DataFrame,
        *,
        eligible_intervals: pd.DataFrame | None = None,
        covariates: np.ndarray | pd.DataFrame | None = None,
        covariate_cols: Sequence[str] | None = None,
        max_muts_per_sample_per_tile: int | None = None,
        pos_coord_system: Literal["zero_based", "one_based"] = "zero_based",
        restrict_mutations_to_eligible: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        _validate_settings(settings)
        self.regions = regions
        self.mutations = mutations
        self.eligible_intervals = eligible_intervals
        self.covariates = covariates
        self.covariate_cols = covariate_cols
        self.max_muts_per_sample_per_tile = max_muts_per_sample_per_tile
        self.pos_coord_system = pos_coord_system
        self.restrict_mutations_to_eligible = restrict_mutations_to_eligible
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.settings: SettingsMode = settings

    def fit(self) -> FishHookResult:
        return fit_fishhook(
            regions=self.regions,
            mutations=self.mutations,
            eligible_intervals=self.eligible_intervals,
            covariates=self.covariates,
            covariate_cols=self.covariate_cols,
            max_muts_per_sample_per_tile=self.max_muts_per_sample_per_tile,
            pos_coord_system=self.pos_coord_system,
            restrict_mutations_to_eligible=self.restrict_mutations_to_eligible,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
            settings=self.settings,
        )


def _bh_fdr_nan_aware(p_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    q = np.full(p_values.shape, np.nan, dtype=np.float64)
    p_sub = p_values[valid_mask]
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


def _extract_covariates(
    regions: pd.DataFrame,
    covariates: np.ndarray | pd.DataFrame | None,
    covariate_cols: Sequence[str] | None,
    n_regions: int,
) -> tuple[np.ndarray, list[str]]:
    if covariates is not None and covariate_cols is not None:
        raise ValueError("Provide either `covariates` or `covariate_cols`, not both.")

    if covariate_cols is not None:
        if not isinstance(regions, pd.DataFrame):
            regions = pd.DataFrame(regions)
        cols = [str(c) for c in covariate_cols]
        missing = [c for c in cols if c not in regions.columns]
        if missing:
            raise ValueError(f"covariate_cols missing from regions: {missing}.")
        if len(cols) == 0:
            return np.empty((n_regions, 0), dtype=np.float64), []
        x_mat = regions[cols].to_numpy(dtype=np.float64, copy=True)
        names = cols
    elif covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            names = [str(c) for c in covariates.columns]
            x_mat = covariates.to_numpy(dtype=np.float64, copy=True)
        else:
            x_arr = np.asarray(covariates, dtype=np.float64)
            if x_arr.ndim == 1:
                x_mat = x_arr[:, None].copy()
            elif x_arr.ndim == 2:
                x_mat = x_arr.copy()
            else:
                raise ValueError(f"covariates must be 1D or 2D, got {x_arr.ndim}D.")
            names = [f"X{j}" for j in range(x_mat.shape[1])]
    else:
        return np.empty((n_regions, 0), dtype=np.float64), []

    if x_mat.shape[0] != n_regions:
        raise ValueError(
            f"covariates has {x_mat.shape[0]} rows, expected {n_regions} (matching regions)."
        )
    if not np.all(np.isfinite(x_mat)):
        raise ValueError("covariates must contain only finite numeric values (no NaN/inf).")

    return x_mat, names


def _poisson_loglik(y: np.ndarray, mu: np.ndarray, gammaln_y1: np.ndarray) -> float:
    mu_safe = np.maximum(mu, 1e-300)
    term_y = np.where(y > 0, y * np.log(mu_safe), 0.0)
    return float(np.sum(term_y - mu_safe - gammaln_y1))


def _nb2_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    theta: float,
    gammaln_y1: np.ndarray,
) -> float:
    mu_safe = np.maximum(mu, 1e-300)
    t = float(theta)
    # Exact log-PMF sum without precision loss
    y_pos = y > 0
    gamma_diff = np.where(
        y_pos,
        special.gammaln(y + t) - special.gammaln(t),
        0.0,
    )
    log_p_part = -t * np.log1p(mu_safe / t)
    log_q_part = np.where(
        y_pos,
        y * (np.log(mu_safe) - np.log(t + mu_safe)),
        0.0,
    )
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
            ll_v = _poisson_loglik(y, mu_v, gammaln_y1)
        else:
            ll_v = _nb2_loglik(y, mu_v, 1.0 / alpha, gammaln_y1)
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

        # Backtracking line search ensuring monotonic log-likelihood improvement
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
    """Profile MLE of shape theta = 1 / alpha given fixed means mu."""
    mu_safe = np.maximum(mu, 1e-300)

    def _neg_ll(log_t: float) -> float:
        return -_nb2_loglik(y, mu_safe, float(np.exp(log_t)), gammaln_y1)

    opt = optimize.minimize_scalar(
        _neg_ll,
        bounds=(_MIN_LOG_THETA, _MAX_LOG_THETA),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    log_t = float(opt.x)
    best_ll = -float(opt.fun)

    # Polish on log_theta via damped 1D Newton-Raphson when strictly interior
    y_pos = y > 0
    for _ in range(15):
        if log_t <= _MIN_LOG_THETA + 1e-5 or log_t >= _MAX_LOG_THETA - 1e-5:
            break
        t = float(np.exp(log_t))
        psi_diff = np.where(y_pos, special.digamma(y + t) - special.digamma(t), 0.0)
        d_t = float(
            np.sum(
                psi_diff
                - np.log1p(mu_safe / t)
                + (mu_safe - y) / (t + mu_safe)
            )
        )
        # Chain rule: d ell / d(log t) = t * d_t
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
                tri_diff
                + 1.0 / t
                - 2.0 / (t + mu_safe)
                + (y + t) / ((t + mu_safe) ** 2)
            )
        )
        # d^2 ell / d(log t)^2 = t^2 * d2_t + t * d_t
        h_z = (t * t) * d2_t + g_z
        if h_z >= -1e-12:
            break
        step = np.clip(g_z / h_z, -1.0, 1.0)
        cand_z = float(np.clip(log_t - step, _MIN_LOG_THETA, _MAX_LOG_THETA))
        cand_ll = _nb2_loglik(y, mu_safe, float(np.exp(cand_z)), gammaln_y1)
        if cand_ll >= best_ll:
            log_t = cand_z
            best_ll = cand_ll
        else:
            break

    return float(np.exp(log_t)), best_ll


def _compute_inflation_diagnostics(
    p_active: np.ndarray,
    *,
    settings: SettingsMode = "default",
) -> tuple[float, float]:
    """
    Compute QQ slope (-log10 observed vs uniform expected) and genomic inflation
    factor lambda_GC.
    In `settings='R_default'`, fits all-point zero-intercept OLS
    `lm(-log10(p_obs) ~ -log10(p_exp) - 1)` matching `mskilab/fishHook` `fishhook.R`.
    """
    m = p_active.size
    if m == 0:
        return 1.0, 1.0

    p_clamped = np.clip(p_active, 1e-300, 1.0)
    p_sorted = np.sort(p_clamped)
    exp_p = (np.arange(1, m + 1, dtype=np.float64) - 0.5) / m

    if settings == "R_default":
        x_q = -np.log10(exp_p)
        y_q = -np.log10(p_sorted)
        denom = float(np.dot(x_q, x_q))
        lam = float(np.dot(x_q, y_q) / denom) if denom > 0.0 else 1.0
        return lam, lam

    # QQ slope across informative interior quantiles (excluding top 1% driver tail
    # and the discrete p == 1.0 zero-mutation plateau)
    band = (p_sorted < 1.0 - 1e-12) & (exp_p >= 0.01)
    if np.sum(band) < 3:
        band = p_sorted < 1.0 - 1e-12
    if np.sum(band) >= 2:
        x_q = -np.log10(exp_p[band])
        y_q = -np.log10(p_sorted[band])
        denom = float(np.dot(x_q, x_q))
        qq_slope = float(np.dot(x_q, y_q) / denom) if denom > 0.0 else 1.0
    else:
        qq_slope = 1.0

    med_p = float(np.median(p_clamped))
    if med_p < 0.95:
        med_chi2 = float(stats.chi2.isf(med_p, df=1))
        null_med_chi2 = float(stats.chi2.ppf(0.5, df=1))
        lambda_gc = med_chi2 / null_med_chi2
    else:
        lambda_gc = qq_slope

    return float(lambda_gc), float(qq_slope)


def fit_fishhook(
    regions: pd.DataFrame,
    mutations: pd.DataFrame,
    *,
    eligible_intervals: pd.DataFrame | None = None,
    covariates: np.ndarray | pd.DataFrame | None = None,
    covariate_cols: Sequence[str] | None = None,
    max_muts_per_sample_per_tile: int | None = None,
    pos_coord_system: Literal["zero_based", "one_based"] = "zero_based",
    restrict_mutations_to_eligible: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> FishHookResult:
    """
    Fit a fishHook Gamma-Poisson (NB2) somatic mutation background model with
    callable territory offset log(L_i) and evaluate right-tail enrichment.
    """
    _validate_settings(settings)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")

    muts_df = mutations if isinstance(mutations, pd.DataFrame) else pd.DataFrame(mutations)
    eff_cap = max_muts_per_sample_per_tile
    if settings == "R_default" and eff_cap is None and "sample_id" in muts_df.columns:
        eff_cap = 1

    r_chrom, r_start, r_end, r_ids, eligible_bp, obs_counts = compute_territory_and_counts(
        regions=regions,
        mutations=muts_df,
        eligible_intervals=eligible_intervals,
        max_muts_per_sample_per_tile=eff_cap,
        pos_coord_system=pos_coord_system,
        restrict_mutations_to_eligible=restrict_mutations_to_eligible,
    )

    n_total = r_start.size
    x_raw, cov_names = _extract_covariates(regions, covariates, covariate_cols, n_total)
    n_cov = x_raw.shape[1]
    coef_names = ["(Intercept)", *cov_names]

    active_mask = eligible_bp > 0
    n_act = int(np.sum(active_mask))
    if n_act == 0:
        raise ValueError("No target regions have positive eligible territory (all L_i == 0).")
    if n_act <= n_cov:
        raise ValueError(
            f"Number of active regions ({n_act}) must exceed number of covariates ({n_cov})."
        )

    y_act = obs_counts[active_mask].astype(np.float64)
    l_act = eligible_bp[active_mask].astype(np.float64)
    offset_act = np.log(l_act)
    x_act = x_raw[active_mask]

    # Check design matrix rank on active regions and standardize covariates internally
    if n_cov > 0:
        x_means = np.mean(x_act, axis=0)
        x_stds = np.std(x_act, axis=0)
        if np.any(x_stds <= 1e-15):
            raise ValueError("Covariate matrix contains a constant (zero-variance) column on active regions.")
        x_std = (x_act - x_means) / x_stds
        z_design = np.column_stack([np.ones(n_act, dtype=np.float64), x_std])
        if np.linalg.matrix_rank(z_design) < n_cov + 1:
            raise ValueError("Covariate design matrix is rank-deficient (collinear columns).")
    else:
        x_means = np.empty(0, dtype=np.float64)
        x_stds = np.empty(0, dtype=np.float64)
        z_design = np.ones((n_act, 1), dtype=np.float64)

    # Degenerate zero-mutation cohort edge case
    if float(np.sum(y_act)) == 0.0:
        beta = np.zeros(n_cov + 1, dtype=np.float64)
        beta[0] = -np.inf
        se_beta = np.full(n_cov + 1, np.nan, dtype=np.float64)
        z_beta = np.full(n_cov + 1, np.nan, dtype=np.float64)
        p_beta = np.full(n_cov + 1, np.nan, dtype=np.float64)
        exp_counts = np.zeros(n_total, dtype=np.float64)
        obs_dens = np.where(active_mask, 0.0, np.nan)
        exp_dens = np.where(active_mask, 0.0, np.nan)
        fe = np.where(active_mask, 1.0, np.nan)
        p_neg_zero: np.ndarray | None = None
        fdr_neg_zero: np.ndarray | None = None
        if settings == "R_default":
            rng = np.random.default_rng(seed)
            u_act = rng.uniform(0.0, 1.0, size=n_act)
            p_vals = np.full(n_total, np.nan, dtype=np.float64)
            p_vals[active_mask] = u_act
            q_vals = _bh_fdr_nan_aware(p_vals, active_mask)
            p_neg_zero = np.full(n_total, np.nan, dtype=np.float64)
            p_neg_zero[active_mask] = 1.0 - u_act
            fdr_neg_zero = _bh_fdr_nan_aware(p_neg_zero, active_mask)
            lam_zero, qq_zero = _compute_inflation_diagnostics(u_act, settings="R_default")
        else:
            p_vals = np.where(active_mask, 1.0, np.nan)
            q_vals = np.where(active_mask, 1.0, np.nan)
            lam_zero, qq_zero = 1.0, 1.0
        return FishHookResult(
            region_ids=r_ids,
            chrom=r_chrom,
            start=r_start,
            end=r_end,
            eligible_bp=eligible_bp,
            obs_counts=obs_counts,
            exp_counts=exp_counts,
            obs_density=obs_dens,
            exp_density=exp_dens,
            fold_enrichment=fe,
            p_value=p_vals,
            q_value=q_vals,
            coef_names=coef_names,
            beta=beta,
            se_beta=se_beta,
            z_beta=z_beta,
            p_beta=p_beta,
            alpha=0.0,
            theta=float(np.inf),
            is_poisson_fallback=True,
            loglik=0.0,
            lambda_gc=lam_zero,
            qq_slope=qq_zero,
            converged=True,
            n_iter=1,
            p_neg=p_neg_zero,
            fdr_neg=fdr_neg_zero,
        )

    gammaln_y1 = special.gammaln(y_act + 1.0)
    gamma0 = np.zeros(n_cov + 1, dtype=np.float64)
    gamma0[0] = float(np.log(np.sum(y_act) / np.sum(l_act)))

    # Step 1: Fit exact Poisson GLM (alpha = 0)
    gamma_pois, mu_pois, ll_pois, conv_pois = _fit_irls_given_alpha(
        z_design, offset_act, y_act, 0.0, gamma0, gammaln_y1, max_inner=60, tol=tol
    )

    # Score wrt alpha at alpha -> 0+ and initial profile NB2 likelihood check
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
        n_outer = 1
    else:
        gamma_curr = gamma_pois.copy()
        theta_curr = theta_init
        prev_ll = ll_nb_init
        converged = False
        n_outer = 0

        for it in range(1, max_iter + 1):
            n_outer = it
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
                mu_next = mu_pois
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
            final_ll = _nb2_loglik(y_act, mu_hat, theta_hat, gammaln_y1)

    # Transform coefficients and Fisher covariance matrix back to original X scale
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

    # Region-level expectations, densities, fold-enrichment, exact NB2/Poisson right-tail p-values
    exp_counts = np.zeros(n_total, dtype=np.float64)
    exp_counts[active_mask] = mu_hat

    obs_density = np.full(n_total, np.nan, dtype=np.float64)
    exp_density = np.full(n_total, np.nan, dtype=np.float64)
    fold_enrichment = np.full(n_total, np.nan, dtype=np.float64)
    p_values = np.full(n_total, np.nan, dtype=np.float64)

    obs_density[active_mask] = y_act / l_act
    exp_density[active_mask] = mu_hat / l_act
    fold_enrichment[active_mask] = y_act / np.maximum(mu_hat, 1e-300)

    y_int_act = obs_counts[active_mask]
    p_neg_values: np.ndarray | None = None
    fdr_neg_values: np.ndarray | None = None

    if settings == "R_default":
        rng = np.random.default_rng(seed)
        u_act = rng.uniform(0.0, 1.0, size=n_act)

        if is_poisson:
            p_gt = stats.poisson.sf(y_int_act, mu=mu_hat)
            p_ge = np.where(
                y_int_act >= 1,
                stats.poisson.sf(np.maximum(y_int_act - 1, 0), mu=mu_hat),
                1.0,
            )
            cdf_lt = np.where(
                y_int_act >= 1,
                stats.poisson.cdf(np.maximum(y_int_act - 1, 0), mu=mu_hat),
                0.0,
            )
            cdf_le = stats.poisson.cdf(y_int_act, mu=mu_hat)
        else:
            p_nb_all = theta_hat / (theta_hat + mu_hat)
            p_gt = stats.nbinom.sf(y_int_act, n=theta_hat, p=p_nb_all)
            p_ge = np.where(
                y_int_act >= 1,
                stats.nbinom.sf(np.maximum(y_int_act - 1, 0), n=theta_hat, p=p_nb_all),
                1.0,
            )
            cdf_lt = np.where(
                y_int_act >= 1,
                stats.nbinom.cdf(np.maximum(y_int_act - 1, 0), n=theta_hat, p=p_nb_all),
                0.0,
            )
            cdf_le = stats.nbinom.cdf(y_int_act, n=theta_hat, p=p_nb_all)

        p_act = np.clip((1.0 - u_act) * p_gt + u_act * p_ge, 0.0, 1.0)
        p_neg_act = np.clip(u_act * cdf_lt + (1.0 - u_act) * cdf_le, 0.0, 1.0)

        p_values[active_mask] = p_act
        q_values = _bh_fdr_nan_aware(p_values, active_mask)

        p_neg_values = np.full(n_total, np.nan, dtype=np.float64)
        p_neg_values[active_mask] = p_neg_act
        fdr_neg_values = _bh_fdr_nan_aware(p_neg_values, active_mask)
        lambda_gc, qq_slope = _compute_inflation_diagnostics(p_act, settings="R_default")
    else:
        p_act = np.ones(n_act, dtype=np.float64)
        pos_y = y_int_act >= 1
        if np.any(pos_y):
            if is_poisson:
                p_act[pos_y] = stats.poisson.sf(y_int_act[pos_y] - 1, mu=mu_hat[pos_y])
            else:
                p_nb = theta_hat / (theta_hat + mu_hat[pos_y])
                p_act[pos_y] = stats.nbinom.sf(y_int_act[pos_y] - 1, n=theta_hat, p=p_nb)
        p_act = np.clip(p_act, 0.0, 1.0)
        p_values[active_mask] = p_act

        q_values = _bh_fdr_nan_aware(p_values, active_mask)
        lambda_gc, qq_slope = _compute_inflation_diagnostics(p_act, settings="default")

    return FishHookResult(
        region_ids=r_ids,
        chrom=r_chrom,
        start=r_start,
        end=r_end,
        eligible_bp=eligible_bp,
        obs_counts=obs_counts,
        exp_counts=exp_counts,
        obs_density=obs_density,
        exp_density=exp_density,
        fold_enrichment=fold_enrichment,
        p_value=p_values,
        q_value=q_values,
        coef_names=coef_names,
        beta=beta_orig,
        se_beta=se_orig,
        z_beta=z_orig,
        p_beta=p_orig,
        alpha=float(alpha_hat),
        theta=float(theta_hat),
        is_poisson_fallback=is_poisson,
        loglik=float(final_ll),
        lambda_gc=lambda_gc,
        qq_slope=qq_slope,
        converged=converged,
        n_iter=n_outer,
        p_neg=p_neg_values,
        fdr_neg=fdr_neg_values,
    )
