"""Covariate-adjusted RMST and RMTL regression via IPCW estimating equations.

Implements Tian, Zhao, & Wei (Biostatistics 2014, doi:10.1093/biostatistics/kxt050)
and uno1lab/survRM2::rmst2reg (v1.0-4) with exact censoring-martingale
influence-function sandwich covariance across identity, log-RMST, and log-RMTL links.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

SettingsMode = Literal["default", "R_default"]
LinkType = Literal["identity", "log_rmst", "log_rmtl"]

VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")
VALID_LINKS: tuple[str, ...] = ("identity", "log_rmst", "log_rmtl")


def validate_settings(settings: str) -> SettingsMode:
    """Validate the ``settings`` mode parameter."""
    if settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class RMST2AdjustedModelResult:
    """Covariate-adjusted IPCW regression result for a single link function."""

    link: LinkType
    tau: float
    alpha: float
    settings: SettingsMode
    coef_df: pd.DataFrame
    contrast_label: str
    contrast_est: float
    contrast_se: float
    contrast_ci: tuple[float, float]
    contrast_pval: float
    vcov: np.ndarray
    influence_matrix: np.ndarray
    ipcw_weights: np.ndarray
    converged: bool
    n_iter: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return the regression coefficient table."""
        return self.coef_df.copy()


def _coerce_covariates(
    covariates: np.ndarray | pd.DataFrame | pd.Series,
    n: int,
) -> tuple[np.ndarray, list[str]]:
    """Convert covariates into a 2D float64 array and column names."""
    if isinstance(covariates, pd.DataFrame):
        names = [str(c) for c in covariates.columns]
        mat = covariates.to_numpy(dtype=np.float64)
    elif isinstance(covariates, pd.Series):
        nm = str(covariates.name) if covariates.name is not None else "x0"
        names = [nm]
        mat = covariates.to_numpy(dtype=np.float64).reshape(-1, 1)
    else:
        arr = np.asarray(covariates, dtype=np.float64)
        if arr.ndim == 1:
            mat = arr.reshape(-1, 1)
            names = ["x0"]
        elif arr.ndim == 2:
            mat = arr
            names = [f"x{j}" for j in range(mat.shape[1])]
        else:
            raise ValueError(
                f"covariates must be 1D or 2D, got {arr.ndim}D array."
            )

    if mat.shape[0] != n:
        raise ValueError(
            f"covariates has {mat.shape[0]} rows, expected {n} to match time."
        )
    if mat.size > 0 and not np.all(np.isfinite(mat)):
        raise ValueError("covariates must contain only finite numeric values.")
    return mat, names


def _censoring_km_group(
    y_star: np.ndarray,
    d_star: np.ndarray,
    g_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute left-continuous censoring KM G(Y_i*-) and IPCW weights for a group.

    Follows the standard convention that events precede censorings at tied times,
    so subject j is at risk of censoring at s < tau iff Y_j* > s or
    (Y_j* == s and d_cens_j == 1).

    Returns
    -------
    weights : (m,) IPCW weights d_star_i / max(G(Y_i*-), g_min)
    s_times : (L,) sorted distinct censoring times strictly before tau
    d_c_vec : (L,) number of censorings at each s_l
    n_c_vec : (L,) censoring risk-set size at each s_l
    """
    d_cens = 1 - d_star
    cens_mask = d_cens == 1
    if not np.any(cens_mask):
        weights = d_star.astype(np.float64)
        empty = np.empty(0, dtype=np.float64)
        return weights, empty, empty, empty

    s_times = np.unique(y_star[cens_mask])
    n_s = s_times.shape[0]
    d_c_vec = np.empty(n_s, dtype=np.float64)
    n_c_vec = np.empty(n_s, dtype=np.float64)
    surv_steps = np.empty(n_s, dtype=np.float64)

    cur_g = 1.0
    for idx_s, s_val in enumerate(s_times):
        at_cens = (y_star == s_val) & cens_mask
        at_risk = (y_star > s_val) | at_cens
        d_c = float(np.sum(at_cens))
        n_c = float(np.sum(at_risk))
        d_c_vec[idx_s] = d_c
        n_c_vec[idx_s] = n_c
        cur_g = cur_g * (1.0 - d_c / n_c) if n_c > 0.0 else 0.0
        surv_steps[idx_s] = cur_g

    # For each subject i, G(Y_i*-) = product over s_l < Y_i*
    pos = np.searchsorted(s_times, y_star, side="left")
    g_left = np.ones_like(y_star, dtype=np.float64)
    has_prior = pos > 0
    g_left[has_prior] = surv_steps[pos[has_prior] - 1]
    if g_min > 0.0:
        g_left = np.maximum(g_left, g_min)

    weights = np.zeros_like(y_star, dtype=np.float64)
    obs_mask = d_star == 1
    if np.any(g_left[obs_mask] <= 0.0):
        raise ValueError(
            "Censoring Kaplan-Meier reached zero prior to an uncensored observation; "
            "choose a smaller truncation time tau or use settings='default'."
        )
    weights[obs_mask] = 1.0 / g_left[obs_mask]
    return weights, s_times, d_c_vec, n_c_vec


def _martingale_influence_group(
    u_group: np.ndarray,
    y_star_group: np.ndarray,
    d_star_group: np.ndarray,
    s_times: np.ndarray,
    d_c_vec: np.ndarray,
    n_c_vec: np.ndarray,
) -> np.ndarray:
    """Compute Tian et al. (2014) censoring-martingale influence functions B_i.

    With u_i = w_i * Z_i * (Y_tilde_i - mu_hat_i) and q_neg_bar(s) =
    -(1/m) sum_{j: Y_j* > s} u_j, the censoring-martingale corrected
    influence vector is:
        B_i = u_i - int_0^tau (q_neg_bar(s) / pi(s)) dM_hat_{C,i}(s)
            = u_i + sum_{s_l} (q_bar(s_l) / pi(s_l)) dM_hat_{C,i}(s_l),
    which projects out the predictable censoring nuisance tangent space.
    """
    b_mat = u_group.copy()
    if s_times.size == 0:
        return b_mat

    d_cens = 1 - d_star_group
    cens_mask = d_cens == 1

    for idx_s, s_val in enumerate(s_times):
        n_c = n_c_vec[idx_s]
        if n_c <= 0.0:
            continue
        d_c = d_c_vec[idx_s]
        # q_bar(s) / pi(s) = (sum_{j: Y_j* > s} u_j) / N_C(s)
        after_s = y_star_group > s_val
        if not np.any(after_s):
            continue
        v_s = np.sum(u_group[after_s], axis=0) / n_c  # (p_total,)

        at_cens = (y_star_group == s_val) & cens_mask
        at_risk = after_s | at_cens
        dm_c = at_cens.astype(np.float64) - at_risk.astype(np.float64) * (d_c / n_c)
        b_mat += dm_c[:, None] * v_s[None, :]

    return b_mat


def _solve_ipcw_glm(
    z_mat: np.ndarray,
    y_tilde: np.ndarray,
    weights: np.ndarray,
    link: LinkType,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """Solve sum_i w_i Z_i (y_tilde_i - g^{-1}(Z_i^T beta)) = 0."""
    n, p_dim = z_mat.shape

    if link == "identity":
        zw = z_mat * weights[:, None]
        a_mat = zw.T @ z_mat
        rhs = zw.T @ y_tilde
        try:
            beta = np.linalg.solve(a_mat, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(a_mat + 1e-10 * np.eye(p_dim), rhs, rcond=None)[0]
        mu = z_mat @ beta
        mu_deriv = np.ones(n, dtype=np.float64)
        return beta, mu, mu_deriv, True, 1

    # Log-link quasi-Poisson estimating equations ("log_rmst" or "log_rmtl")
    w_sum = float(np.sum(weights))
    wy_sum = float(np.sum(weights * y_tilde))
    if w_sum <= 0.0 or wy_sum <= 0.0:
        raise ValueError(
            f"Cannot fit link={link!r}: weighted outcome sum is non-positive "
            "(no events or zero restricted time/time-lost before tau)."
        )

    beta = np.zeros(p_dim, dtype=np.float64)
    beta[0] = np.log(wy_sum / w_sum)

    def _obj(b: np.ndarray) -> float:
        eta = np.clip(z_mat @ b, -40.0, 40.0)
        return float(np.sum(weights * (y_tilde * eta - np.exp(eta))))

    cur_obj = _obj(beta)
    converged = False
    n_iter = max_iter

    for it in range(1, max_iter + 1):
        eta = np.clip(z_mat @ beta, -40.0, 40.0)
        mu = np.exp(eta)
        grad = z_mat.T @ (weights * (y_tilde - mu))
        if float(np.max(np.abs(grad))) < tol:
            converged = True
            n_iter = it
            break

        w_hess = weights * mu
        hess = (z_mat * w_hess[:, None]).T @ z_mat
        try:
            step = np.linalg.solve(hess + 1e-11 * np.eye(p_dim), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess + 1e-8 * np.eye(p_dim), grad, rcond=None)[0]

        # Backtracking line search on strictly concave quasi-likelihood
        step_scale = 1.0
        slope = float(grad @ step)
        while step_scale > 1e-8:
            cand = beta + step_scale * step
            cand_obj = _obj(cand)
            if cand_obj >= cur_obj + 1e-4 * step_scale * max(slope, 0.0) - 1e-12:
                beta = cand
                cur_obj = cand_obj
                break
            step_scale *= 0.5
        else:
            beta = beta + step_scale * step
            cur_obj = _obj(beta)

        if float(np.max(np.abs(step_scale * step))) < tol:
            converged = True
            n_iter = it
            break

    eta = np.clip(z_mat @ beta, -40.0, 40.0)
    mu = np.exp(eta)
    return beta, mu, mu, converged, n_iter


def rmst2reg(
    time: Sequence[float] | np.ndarray,
    status: Sequence[int] | np.ndarray,
    arm: Sequence[int] | np.ndarray,
    covariates: np.ndarray | pd.DataFrame | pd.Series,
    tau: float,
    link: LinkType = "identity",
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> RMST2AdjustedModelResult:
    """Fit Tian et al. (2014) IPCW covariate-adjusted RMST / RMTL regression.

    Parameters
    ----------
    time : 1D array-like of non-negative follow-up times.
    status : 1D array-like of event indicators (1 = event, 0 = right-censored).
    arm : 1D array-like of binary treatment indicators (1 = active, 0 = control).
    covariates : 1D/2D array, Series, or DataFrame of baseline covariates.
    tau : Positive truncation horizon.
    link : ``"identity"`` (adjusted RMST difference), ``"log_rmst"`` (adjusted
        RMST ratio), or ``"log_rmtl"`` (adjusted RMTL ratio).
    alpha : Two-sided significance level in ``(0, 1)``.
    settings : ``"default"`` (pooled censoring KM with ``G_min=1e-3`` tail floor)
        or ``"R_default"`` (``survRM2::rmst2reg`` v1.0-4 exact parity with
        separate per-arm censoring KM and unfloored weights).
    """
    validate_settings(settings)
    if link not in VALID_LINKS:
        raise ValueError(f"Unknown link={link!r}. Expected one of {VALID_LINKS}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be finite and positive, got {tau}.")

    t_arr = np.asarray(time, dtype=np.float64)
    d_arr = np.asarray(status, dtype=np.int64)
    a_arr = np.asarray(arm, dtype=np.int64)
    n = t_arr.shape[0]

    cov_mat, cov_names = _coerce_covariates(covariates, n)
    term_names = ["intercept", "arm", *cov_names]
    z_mat = np.column_stack([np.ones(n, dtype=np.float64), a_arr.astype(np.float64), cov_mat])

    y_star = np.minimum(t_arr, float(tau))
    d_star = np.where(t_arr >= float(tau), 1, d_arr)

    if settings == "default":
        g_min = 1e-3
        weights, s_pool, dc_pool, nc_pool = _censoring_km_group(y_star, d_star, g_min=g_min)
    else:
        weights = np.zeros(n, dtype=np.float64)
        arm_cens_info: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for arm_val in (0, 1):
            mask_a = a_arr == arm_val
            w_a, s_a, dc_a, nc_a = _censoring_km_group(
                y_star[mask_a], d_star[mask_a], g_min=0.0
            )
            weights[mask_a] = w_a
            arm_cens_info[arm_val] = (mask_a, s_a, dc_a, nc_a)

    y_tilde = (float(tau) - y_star) if link == "log_rmtl" else y_star

    beta, mu, mu_deriv, converged, n_iter = _solve_ipcw_glm(
        z_mat=z_mat,
        y_tilde=y_tilde,
        weights=weights,
        link=link,
    )

    # Uncorrected score contributions u_i = w_i * Z_i * (y_tilde_i - mu_i)
    resid = y_tilde - mu
    u_mat = z_mat * (weights * resid)[:, None]

    # Exact censoring-martingale influence matrix B (n, p_dim)
    if settings == "default":
        b_mat = _martingale_influence_group(
            u_group=u_mat,
            y_star_group=y_star,
            d_star_group=d_star,
            s_times=s_pool,
            d_c_vec=dc_pool,
            n_c_vec=nc_pool,
        )
    else:
        b_mat = np.zeros_like(u_mat)
        for arm_val in (0, 1):
            mask_a, s_a, dc_a, nc_a = arm_cens_info[arm_val]
            b_mat[mask_a] = _martingale_influence_group(
                u_group=u_mat[mask_a],
                y_star_group=y_star[mask_a],
                d_star_group=d_star[mask_a],
                s_times=s_a,
                d_c_vec=dc_a,
                n_c_vec=nc_a,
            )

    # Bread matrix A_sum = sum_i w_i * dot{g}_i * Z_i Z_i^T and Meat = sum_i B_i B_i^T
    p_dim = z_mat.shape[1]
    bread = (z_mat * (weights * mu_deriv)[:, None]).T @ z_mat
    meat = b_mat.T @ b_mat
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread + 1e-10 * np.eye(p_dim))

    vcov = bread_inv @ meat @ bread_inv.T
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    z_crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stat = np.where(se > 0.0, beta / se, 0.0)
    pvals = 2.0 * stats.norm.sf(np.abs(z_stat))
    ci_lower = beta - z_crit * se
    ci_upper = beta + z_crit * se

    coef_data: dict[str, np.ndarray | list[str]] = {
        "term": term_names,
        "coef": beta,
        "se": se,
        "z": z_stat,
        "pval": pvals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    if link in ("log_rmst", "log_rmtl"):
        coef_data["exp_coef"] = np.exp(beta)
        coef_data["exp_ci_lower"] = np.exp(ci_lower)
        coef_data["exp_ci_upper"] = np.exp(ci_upper)

    coef_df = pd.DataFrame(coef_data)

    # Extract arm contrast (index 1 corresponds to arm == 1 vs arm == 0)
    b1 = float(beta[1])
    se1 = float(se[1])
    p1 = float(pvals[1])
    if link == "identity":
        c_label = "RMST (arm=1)-(arm=0)"
        c_est = b1
        c_se = se1
        c_ci = (float(ci_lower[1]), float(ci_upper[1]))
    elif link == "log_rmst":
        c_label = "RMST (arm=1)/(arm=0)"
        c_est = float(np.exp(b1))
        c_se = se1  # SE on log scale matching survRM2::rmst2reg
        c_ci = (float(np.exp(ci_lower[1])), float(np.exp(ci_upper[1])))
    else:
        c_label = "RMTL (arm=1)/(arm=0)"
        c_est = float(np.exp(b1))
        c_se = se1
        c_ci = (float(np.exp(ci_lower[1])), float(np.exp(ci_upper[1])))

    return RMST2AdjustedModelResult(
        link=link,
        tau=float(tau),
        alpha=alpha,
        settings=settings,
        coef_df=coef_df,
        contrast_label=c_label,
        contrast_est=c_est,
        contrast_se=c_se,
        contrast_ci=c_ci,
        contrast_pval=p1,
        vcov=vcov,
        influence_matrix=b_mat,
        ipcw_weights=weights,
        converged=converged,
        n_iter=n_iter,
    )
