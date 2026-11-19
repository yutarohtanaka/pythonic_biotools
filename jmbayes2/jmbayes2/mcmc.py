"""Bayesian Joint Longitudinal and Survival / Competing-Risks MCMC Engine.

Implements Rizopoulos (2012) and Papageorgiou et al. (Statistics in Medicine 2019,
doi:10.1002/sim.8103) with 15-node Gauss-Kronrod cumulative hazard quadrature,
Joint Laplace MAP warm-start, and preconditioned Adaptive Metropolis-within-Gibbs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .lme import LMEFit, fit_lme
from .quadrature import (
    BSplineHazardBasis,
    SettingsMode,
    gauss_kronrod_15,
    validate_settings,
)

VALID_FUNCTIONAL_FORMS: tuple[str, ...] = ("value", "slope", "area")


def _normalize_functional_forms(
    functional_forms: str | Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalize functional forms linking longitudinal trajectory to hazard."""
    if isinstance(functional_forms, str):
        forms_raw = [functional_forms]
    else:
        forms_raw = list(functional_forms)
    if not forms_raw:
        raise ValueError("functional_forms must be non-empty.")

    dedup: list[str] = []
    for f in forms_raw:
        if f not in VALID_FUNCTIONAL_FORMS:
            raise ValueError(
                f"Unknown functional form {f!r}; expected subset of {VALID_FUNCTIONAL_FORMS}."
            )
        if f not in dedup:
            dedup.append(f)
    return tuple(dedup)


def _split_rhat(draws: np.ndarray) -> float:
    """Compute Gelman-Rubin split-Rhat on a 1D MCMC chain."""
    n_half = draws.shape[0] // 2
    if n_half < 2:
        return 1.0
    chains = draws[: 2 * n_half].reshape(2, n_half)
    chain_vars = np.var(chains, axis=1, ddof=1)
    w_var = float(np.mean(chain_vars))
    if w_var <= 1e-15:
        return 1.0
    chain_means = np.mean(chains, axis=1)
    b_var = float(n_half * np.var(chain_means, ddof=1))
    var_plus = ((n_half - 1.0) / n_half) * w_var + (1.0 / n_half) * b_var
    return float(np.sqrt(max(var_plus / w_var, 1.0)))


def _mcmc_tail_pval(draws: np.ndarray, settings: SettingsMode) -> float:
    """Compute two-sided Bayesian tail probability P for posterior draws."""
    s = draws.shape[0]
    if s == 0:
        return 1.0
    if settings == "R_default":
        # Exact JMbayes2::summary tail probability: 2 * min(mean(draws > 0), mean(draws < 0))
        p_pos = float(np.mean(draws > 0.0))
        p_neg = float(np.mean(draws < 0.0))
        return float(min(1.0, 2.0 * min(p_pos, p_neg)))
    # Pythonic default: finite-sample smoothed tail probability floored at 1 / (S + 1)
    n_pos = float(np.sum(draws > 0.0))
    n_neg = float(np.sum(draws < 0.0))
    p_two = 2.0 * min((n_pos + 0.5) / (s + 1.0), (n_neg + 0.5) / (s + 1.0))
    return float(min(1.0, max(p_two, 1.0 / (s + 1.0))))


def _sample_invwishart(
    rng: np.random.Generator,
    df: float,
    scale_mat: np.ndarray,
) -> np.ndarray:
    """Exact Bartlett Cholesky sampler for Inv-Wishart(df, scale_mat)."""
    q = scale_mat.shape[0]
    sym_scale = 0.5 * (scale_mat + scale_mat.T) + 1e-10 * np.eye(q)
    inv_scale = np.linalg.inv(sym_scale)
    chol_inv = np.linalg.cholesky(0.5 * (inv_scale + inv_scale.T) + 1e-12 * np.eye(q))

    a_mat = np.zeros((q, q), dtype=np.float64)
    for j in range(q):
        a_mat[j, j] = np.sqrt(rng.chisquare(df - j))
        for k in range(j):
            a_mat[j, k] = rng.normal()

    la = chol_inv @ a_mat
    wish = la @ la.T
    d_draw = np.linalg.inv(wish + 1e-12 * np.eye(q))
    return 0.5 * (d_draw + d_draw.T)


def eval_traj_forms(
    t_eval: np.ndarray,
    beta: np.ndarray,
    b_mat: np.ndarray,
    c_long: np.ndarray,
    functional_forms: tuple[str, ...],
) -> np.ndarray:
    """Evaluate longitudinal trajectory functional forms at ``t_eval``.

    Parameters
    ----------
    t_eval : Shape ``(N,)`` or ``(N, Q)``.
    beta : Shape ``(2 + P_c,)`` with ``beta[0]`` intercept, ``beta[1]`` slope, ``beta[2:]`` covariates.
    b_mat : Shape ``(N, q_b)`` with ``q_b`` in ``{1, 2}``.
    c_long : Shape ``(N, P_c)`` baseline longitudinal covariates per subject.
    functional_forms : Tuple subset of ``("value", "slope", "area")``.

    Returns
    -------
    f_arr : Shape ``(*t_eval.shape, len(functional_forms))``.
    """
    n_subj, q_b = b_mat.shape
    cov_offset = c_long @ beta[2:] if c_long.shape[1] > 0 else np.zeros(n_subj, dtype=np.float64)
    icpt = beta[0] + b_mat[:, 0] + cov_offset
    slope = np.full(n_subj, float(beta[1]), dtype=np.float64)
    if q_b == 2:
        slope = slope + b_mat[:, 1]

    is_2d = t_eval.ndim == 2
    icpt_b = icpt[:, None] if is_2d else icpt
    slope_b = slope[:, None] if is_2d else slope

    comps: list[np.ndarray] = []
    for form in functional_forms:
        if form == "value":
            comps.append(icpt_b + slope_b * t_eval)
        elif form == "slope":
            comps.append(np.broadcast_to(slope_b, t_eval.shape).copy())
        elif form == "area":
            comps.append(icpt_b * t_eval + 0.5 * slope_b * (t_eval ** 2))
    return np.stack(comps, axis=-1)


@dataclass(frozen=True)
class JMModelSpec:
    """Internal model specification metadata used for dynamic risk prediction."""

    id_col: str
    surv_time_col: str
    event_col: str
    long_time_col: str
    y_col: str
    surv_covs: tuple[str, ...]
    long_covs: tuple[str, ...]
    random_slope: bool
    functional_forms: tuple[str, ...]
    competing_risks: bool
    causes: tuple[int, ...]
    basis: BSplineHazardBasis
    surv_cov_means: np.ndarray


@dataclass(frozen=True)
class JMResult:
    """Fitted Bayesian Joint Longitudinal-Survival Model Result."""

    assoc_df: pd.DataFrame
    surv_df_summary: pd.DataFrame
    long_df_summary: pd.DataFrame
    posterior_draws: dict[str, np.ndarray]
    dic: float
    waic: float
    rhat: dict[str, float]
    lme_fit: LMEFit
    spec: JMModelSpec
    settings: SettingsMode
    n_subjects: int
    n_obs: int
    n_events: int
    n_iter: int
    burn_in: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return combined parameter posterior summary table."""
        parts: list[pd.DataFrame] = []
        l_df = self.long_df_summary.copy()
        l_df.insert(0, "submodel", "longitudinal")
        parts.append(l_df)

        if not self.surv_df_summary.empty:
            s_df = self.surv_df_summary.copy()
            s_df.insert(0, "submodel", "survival")
            parts.append(s_df)

        a_df = self.assoc_df.copy()
        a_df["term"] = [
            f"assoc[{row['form']}]_cause{row['cause']}"
            if self.spec.competing_risks
            else f"assoc[{row['form']}]"
            for _, row in a_df.iterrows()
        ]
        a_df.insert(0, "submodel", "association")
        keep_cols = ["submodel", "term", "mean", "sd", "median", "ci_lower", "ci_upper", "pval", "rhat"]
        parts_aligned = [df.reindex(columns=keep_cols) for df in parts + [a_df]]
        return pd.concat(parts_aligned, ignore_index=True)

    def summary(self) -> str:
        """Return formatted posterior summary of joint model."""
        mode_str = "Competing-Risks" if self.spec.competing_risks else "Single-Event"
        lines = [
            f"Bayesian Joint Model ({mode_str}, N={self.n_subjects}, n_obs={self.n_obs}, events={self.n_events}, settings={self.settings!r})",
            f"  Functional forms: {self.spec.functional_forms}, GK15 quadrature, B-spline degree={self.spec.basis.degree}",
            f"  MCMC: {self.n_iter} iterations ({self.burn_in} burn-in) | DIC = {self.dic:.2f} | WAIC = {self.waic:.2f}",
            "",
            "Association Parameters (alpha):",
            self.assoc_df.to_string(index=False),
            "",
            "Longitudinal Submodel:",
            self.long_df_summary.to_string(index=False),
        ]
        if not self.surv_df_summary.empty:
            lines.extend(
                [
                    "",
                    "Baseline Survival Covariates (gamma):",
                    self.surv_df_summary.to_string(index=False),
                ]
            )
        return "\n".join(lines)


def _solve_cause_map(
    ev_mask_k: np.ndarray,
    b_event: np.ndarray,
    b_quad: np.ndarray,
    w_surv: np.ndarray,
    f_event: np.ndarray,
    f_quad: np.ndarray,
    w_quad: np.ndarray,
    pen_mat: np.ndarray,
    tau_h: float,
    psi_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact Newton-Raphson MAP and inverse Hessian covariance for cause k."""
    n_subj, n_q, p_spl = b_quad.shape
    p_gam = w_surv.shape[1]
    p_alp = f_event.shape[1]
    p_psi = p_spl + p_gam + p_alp

    # Build combined design matrices M_ev (N, P_psi) and M_quad (N, 15, P_psi)
    ev_parts = [b_event]
    quad_parts = [b_quad]
    if p_gam > 0:
        ev_parts.append(w_surv)
        quad_parts.append(np.broadcast_to(w_surv[:, None, :], (n_subj, n_q, p_gam)))
    ev_parts.append(f_event)
    quad_parts.append(f_quad)

    m_ev = np.concatenate(ev_parts, axis=1)
    m_quad = np.concatenate(quad_parts, axis=2)
    m_quad_flat = m_quad.reshape(n_subj * n_q, p_psi)
    w_quad_flat = w_quad.ravel()

    # Prior precision matrix P_k
    prior_prec = np.zeros((p_psi, p_psi), dtype=np.float64)
    prior_prec[:p_spl, :p_spl] = tau_h * pen_mat + 1e-2 * np.eye(p_spl)
    prior_prec[p_spl:, p_spl:] = (1.0 / 25.0) * np.eye(p_gam + p_alp)

    ev_sum = np.sum(m_ev[ev_mask_k], axis=0) if np.any(ev_mask_k) else np.zeros(p_psi, dtype=np.float64)

    if psi_init is None:
        psi = np.zeros(p_psi, dtype=np.float64)
        tot_time = max(float(np.sum(w_quad_flat)), 1e-6)
        n_ev_k = max(float(np.sum(ev_mask_k)), 0.5)
        psi[:p_spl] = np.log(n_ev_k / tot_time)
    else:
        psi = psi_init.copy()

    def _obj(p_vec: np.ndarray) -> float:
        eta_q = np.clip(m_quad_flat @ p_vec, -45.0, 28.0)
        ll = float(ev_sum @ p_vec) - float(np.sum(w_quad_flat * np.exp(eta_q)))
        return ll - 0.5 * float(p_vec @ (prior_prec @ p_vec))

    cur_val = _obj(psi)
    for _ in range(25):
        eta_q = np.clip(m_quad_flat @ psi, -45.0, 28.0)
        wq_exp = w_quad_flat * np.exp(eta_q)
        grad = ev_sum - (m_quad_flat.T @ wq_exp) - (prior_prec @ psi)
        if float(np.max(np.abs(grad))) < 1e-8:
            break
        hess = (m_quad_flat * wq_exp[:, None]).T @ m_quad_flat + prior_prec + 1e-8 * np.eye(p_psi)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess + 1e-5 * np.eye(p_psi), grad, rcond=None)[0]

        scale = 1.0
        while scale > 1e-6:
            cand = psi + scale * step
            cand_val = _obj(cand)
            if cand_val >= cur_val - 1e-10:
                psi = cand
                cur_val = cand_val
                break
            scale *= 0.5
        else:
            break

    eta_q = np.clip(m_quad_flat @ psi, -45.0, 28.0)
    wq_exp = w_quad_flat * np.exp(eta_q)
    hess = (m_quad_flat * wq_exp[:, None]).T @ m_quad_flat + prior_prec + 1e-6 * np.eye(p_psi)
    cov_psi = np.linalg.inv(0.5 * (hess + hess.T))
    cov_psi = 0.5 * (cov_psi + cov_psi.T)
    return psi, cov_psi


def jm(
    surv_df: pd.DataFrame,
    long_df: pd.DataFrame,
    id_col: str = "id",
    surv_time_col: str = "time",
    event_col: str = "event",
    long_time_col: str = "obstime",
    y_col: str = "y",
    surv_covs: Sequence[str] | None = None,
    long_covs: Sequence[str] | None = None,
    random_slope: bool = True,
    functional_forms: str | Sequence[str] = ("value",),
    competing_risks: bool = False,
    n_iter: int = 500,
    burn_in: int = 150,
    n_knots: int = 6,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> JMResult:
    """Fit a Bayesian joint model for longitudinal and survival / competing-risks data.

    Parameters
    ----------
    surv_df : Subject-level survival DataFrame with ``id_col``, ``surv_time_col``, ``event_col``.
    long_df : Repeated-measures longitudinal DataFrame with ``id_col``, ``long_time_col``, ``y_col``.
    id_col : Subject ID column shared by ``surv_df`` and ``long_df``.
    surv_time_col : Follow-up time column in ``surv_df``.
    event_col : Event status column (0/1 for single-event; 0, 1, ..., K for competing risks).
    long_time_col : Longitudinal measurement time column in ``long_df``.
    y_col : Continuous longitudinal outcome column in ``long_df``.
    surv_covs : Optional baseline covariate names in ``surv_df``.
    long_covs : Optional baseline covariate names in ``long_df`` (or ``surv_df``).
    random_slope : Include correlated random intercept + random time slope when ``True``.
    functional_forms : Subset of ``("value", "slope", "area")`` linking trajectory to hazard.
    competing_risks : If ``True``, fits cause-specific competing-risks submodels for all positive causes.
    n_iter : Total MCMC iterations.
    burn_in : Adaptive warmup iterations discarded prior to posterior inference.
    n_knots : Number of interior knots for B-spline log-baseline hazard.
    seed : Random number generator seed.
    settings : ``"default"`` (cubic B-spline, Laplace MAP + Adaptive Metropolis-within-Gibbs)
        or ``"R_default"`` (``JMbayes2::jm`` v0.5 parity: quadratic B-spline,
        ``Gamma(1.0, 0.005)`` spline penalty precision, exact MCMC tail probability ``P``).
    """
    validate_settings(settings)
    forms = _normalize_functional_forms(functional_forms)
    if n_iter <= burn_in or (n_iter - burn_in) < 10:
        raise ValueError(
            f"n_iter ({n_iter}) must exceed burn_in ({burn_in}) by at least 10 draws."
        )
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}.")

    if not isinstance(surv_df, pd.DataFrame) or surv_df.empty:
        raise ValueError("surv_df must be a non-empty pandas DataFrame.")
    if not isinstance(long_df, pd.DataFrame) or long_df.empty:
        raise ValueError("long_df must be a non-empty pandas DataFrame.")

    for col in (id_col, surv_time_col, event_col):
        if col not in surv_df.columns:
            raise ValueError(f"Column {col!r} not found in surv_df.")
    for col in (id_col, long_time_col, y_col):
        if col not in long_df.columns:
            raise ValueError(f"Column {col!r} not found in long_df.")

    if surv_df[id_col].duplicated().any():
        raise ValueError("surv_df must contain exactly one row per unique subject ID.")

    subj_ids = surv_df[id_col].to_numpy()
    n_subjects = subj_ids.shape[0]
    long_subj_set = set(pd.unique(long_df[id_col]))
    if set(subj_ids) != long_subj_set:
        raise ValueError("Subject IDs in surv_df and long_df must match identically.")

    surv_times = surv_df[surv_time_col].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(surv_times)) or np.any(surv_times <= 0.0):
        raise ValueError("Survival times in surv_df must be finite and strictly positive.")

    ev_raw = surv_df[event_col].to_numpy()
    if not np.all(np.isfinite(ev_raw)) or np.any(ev_raw < 0) or not np.all(ev_raw == np.floor(ev_raw)):
        raise ValueError("Event indicators must be non-negative integers.")
    ev_vec = ev_raw.astype(np.int64)

    if competing_risks:
        causes = tuple(sorted(int(c) for c in np.unique(ev_vec) if c > 0))
        if len(causes) == 0:
            raise ValueError("At least one positive event cause is required.")
    else:
        if not np.all(np.isin(ev_vec, [0, 1])):
            raise ValueError(
                "For competing_risks=False, event indicators must be in {0, 1}. "
                "Set competing_risks=True for multi-cause events."
            )
        if np.sum(ev_vec == 1) == 0:
            raise ValueError("At least one observed event (event == 1) is required.")
        causes = (1,)

    n_causes = len(causes)

    # Prepare survival covariates W_surv (centered internally? No, keep original scale for interpretable gamma!)
    s_cov_names: list[str] = []
    if surv_covs is not None:
        for col in surv_covs:
            if col not in surv_df.columns:
                raise ValueError(f"Survival covariate {col!r} not found in surv_df.")
            if col not in s_cov_names:
                s_cov_names.append(col)
    w_surv = (
        surv_df[s_cov_names].to_numpy(dtype=np.float64)
        if s_cov_names
        else np.empty((n_subjects, 0), dtype=np.float64)
    )
    if w_surv.size > 0 and not np.all(np.isfinite(w_surv)):
        raise ValueError("surv_covs in surv_df must be finite.")
    surv_cov_means = np.mean(w_surv, axis=0) if s_cov_names else np.empty(0, dtype=np.float64)

    # Prepare longitudinal covariates (merge from surv_df if missing in long_df)
    long_work = long_df.copy()
    l_cov_names: list[str] = []
    if long_covs is not None:
        for col in long_covs:
            if col == long_time_col:
                continue
            if col not in long_work.columns:
                if col in surv_df.columns:
                    long_work = long_work.merge(
                        surv_df[[id_col, col]], on=id_col, how="left"
                    )
                else:
                    raise ValueError(
                        f"Longitudinal covariate {col!r} not found in long_df or surv_df."
                    )
            if col not in l_cov_names:
                l_cov_names.append(col)

    # Reorder long_work so subject encounter order matches surv_df[id_col] exactly!
    id_order_map: dict[Any, int] = {sid: idx for idx, sid in enumerate(subj_ids)}
    long_work["_subj_order"] = [id_order_map[sid] for sid in long_work[id_col]]
    long_work = long_work.sort_values(["_subj_order", long_time_col], kind="mergesort").reset_index(drop=True)
    obs_subj_idx = long_work["_subj_order"].to_numpy(dtype=np.int64)
    long_work = long_work.drop(columns=["_subj_order"])

    # Step 1: Fit REML LME submodel
    lme_fit = fit_lme(
        long_work,
        id_col=id_col,
        time_col=long_time_col,
        y_col=y_col,
        fixed_cols=l_cov_names if l_cov_names else None,
        random_slope=random_slope,
        settings=settings,
    )

    # Extract longitudinal arrays aligned with obs_subj_idx
    t_obs = long_work[long_time_col].to_numpy(dtype=np.float64)
    y_obs = long_work[y_col].to_numpy(dtype=np.float64)
    n_obs = y_obs.shape[0]
    x_obs_cols = [np.ones(n_obs, dtype=np.float64), t_obs]
    for col in l_cov_names:
        x_obs_cols.append(long_work[col].to_numpy(dtype=np.float64))
    x_obs = np.column_stack(x_obs_cols)
    p_beta = x_obs.shape[1]

    q_b = 2 if random_slope else 1
    z_obs = (
        np.column_stack([np.ones(n_obs, dtype=np.float64), t_obs])
        if random_slope
        else np.ones((n_obs, 1), dtype=np.float64)
    )

    # Baseline longitudinal covariates per subject (first observation per subject)
    if l_cov_names:
        first_rows = long_work.groupby(id_col, sort=False)[l_cov_names].first()
        c_long = first_rows.loc[subj_ids].to_numpy(dtype=np.float64)
    else:
        c_long = np.empty((n_subjects, 0), dtype=np.float64)

    # Step 2: Precompute GK15 quadrature nodes/weights and B-spline basis
    spl_degree = 2 if settings == "R_default" else 3
    basis = BSplineHazardBasis.from_times(
        surv_times=surv_times,
        event_mask=ev_vec > 0,
        n_knots=n_knots,
        degree=spl_degree,
    )
    u_quad, w_quad = gauss_kronrod_15(np.zeros(n_subjects, dtype=np.float64), surv_times)
    b_event = basis.evaluate(surv_times)  # (N, P_spl)
    b_quad = basis.evaluate(u_quad)  # (N, 15, P_spl)
    p_spl = basis.n_basis
    p_gam = w_surv.shape[1]
    p_alp = len(forms)
    p_psi = p_spl + p_gam + p_alp

    ev_masks = [ev_vec == c_val for c_val in causes]

    # Step 3: Joint Laplace MAP warm-start
    beta_cur = lme_fit.beta.copy()
    sigma2_cur = float(lme_fit.sigma2)
    d_cur = lme_fit.D.copy()
    b_cur = lme_fit.ranef.to_numpy(dtype=np.float64).copy()

    p_reg = p_gam + p_alp
    tau_h_cur = np.full(n_causes, 5.0, dtype=np.float64)
    psi_cur = np.zeros((n_causes, p_psi), dtype=np.float64)
    chol_psi = np.zeros((n_causes, p_psi, p_psi), dtype=np.float64)
    chol_reg = np.zeros((n_causes, p_reg, p_reg), dtype=np.float64)

    f_ev_cur = eval_traj_forms(surv_times, beta_cur, b_cur, c_long, forms)
    f_q_cur = eval_traj_forms(u_quad, beta_cur, b_cur, c_long, forms)

    for k in range(n_causes):
        psi_k, cov_k = _solve_cause_map(
            ev_mask_k=ev_masks[k],
            b_event=b_event,
            b_quad=b_quad,
            w_surv=w_surv,
            f_event=f_ev_cur,
            f_quad=f_q_cur,
            w_quad=w_quad,
            pen_mat=basis.penalty_matrix,
            tau_h=tau_h_cur[k],
        )
        psi_cur[k] = psi_k
        chol_psi[k] = np.linalg.cholesky(cov_k + 1e-9 * np.eye(p_psi))
        cov_reg_k = cov_k[p_spl:, p_spl:]
        chol_reg[k] = np.linalg.cholesky(0.5 * (cov_reg_k + cov_reg_k.T) + 1e-9 * np.eye(p_reg))

    # Helper functions for vectorized log-likelihoods
    def _surv_ll_per_subj(
        beta_v: np.ndarray,
        b_v: np.ndarray,
        psi_v: np.ndarray,
    ) -> np.ndarray:
        f_ev = eval_traj_forms(surv_times, beta_v, b_v, c_long, forms)
        f_q = eval_traj_forms(u_quad, beta_v, b_v, c_long, forms)
        ll_tot = np.zeros(n_subjects, dtype=np.float64)
        for k in range(n_causes):
            pk = psi_v[k]
            xi_k = pk[:p_spl]
            gam_k = pk[p_spl : p_spl + p_gam]
            alp_k = pk[p_spl + p_gam :]
            w_lin = w_surv @ gam_k if p_gam > 0 else np.zeros(n_subjects, dtype=np.float64)
            lh_ev = (b_event @ xi_k) + w_lin + (f_ev @ alp_k)
            lh_q = (
                np.einsum("nqp,p->nq", b_quad, xi_k)
                + w_lin[:, None]
                + np.einsum("nqf,f->nq", f_q, alp_k)
            )
            lh_q = np.clip(lh_q, -48.0, 28.0)
            cum_h = np.sum(w_quad * np.exp(lh_q), axis=1)
            ll_tot += np.where(ev_masks[k], lh_ev, 0.0) - cum_h
        return ll_tot

    def _long_ss_per_subj(beta_v: np.ndarray, b_v: np.ndarray) -> np.ndarray:
        pred = (x_obs @ beta_v) + np.sum(z_obs * b_v[obs_subj_idx], axis=1)
        sq = (y_obs - pred) ** 2
        return np.bincount(obs_subj_idx, weights=sq, minlength=n_subjects)

    # Precompute Cholesky factors for proposals on b_i and beta
    chol_b = np.linalg.cholesky(lme_fit.ranef_cov + 1e-8 * np.eye(q_b)[None, :, :])
    chol_beta = np.linalg.cholesky(lme_fit.vcov_fixed + 1e-9 * np.eye(p_beta))

    scale_b = np.full(n_subjects, 2.38 / np.sqrt(q_b), dtype=np.float64)
    scale_beta = 1.6 / np.sqrt(p_beta)
    scale_psi = np.full(n_causes, 1.5 / np.sqrt(p_psi), dtype=np.float64)
    scale_reg = np.full(n_causes, 2.0 / np.sqrt(max(p_reg, 1)), dtype=np.float64)

    acc_b = np.zeros(n_subjects, dtype=np.float64)
    acc_beta = 0.0
    acc_psi = np.zeros(n_causes, dtype=np.float64)
    acc_reg = np.zeros(n_causes, dtype=np.float64)
    adapt_window = 25

    # Hyperparameters
    a_h, b_h = (1.0, 0.005) if settings == "R_default" else (1.0, 0.01)
    nu_d0 = float(q_b + 2)
    psi_d0 = lme_fit.D.copy()

    n_keep = n_iter - burn_in
    draws_beta = np.empty((n_keep, p_beta), dtype=np.float64)
    draws_sigma2 = np.empty(n_keep, dtype=np.float64)
    draws_d = np.empty((n_keep, q_b, q_b), dtype=np.float64)
    draws_xi = np.empty((n_keep, n_causes, p_spl), dtype=np.float64)
    draws_gamma = np.empty((n_keep, n_causes, p_gam), dtype=np.float64)
    draws_alpha = np.empty((n_keep, n_causes, p_alp), dtype=np.float64)
    draws_tau_h = np.empty((n_keep, n_causes), dtype=np.float64)
    draws_b = np.empty((n_keep, n_subjects, q_b), dtype=np.float64)
    draws_ll_subj = np.empty((n_keep, n_subjects), dtype=np.float64)

    n_obs_per_subj = np.bincount(obs_subj_idx, minlength=n_subjects).astype(np.float64)
    rng = np.random.default_rng(seed)

    ss_cur = _long_ss_per_subj(beta_cur, b_cur)
    ll_surv_cur = _surv_ll_per_subj(beta_cur, b_cur, psi_cur)

    # Step 4: Adaptive Metropolis-within-Gibbs loop
    for it in range(1, n_iter + 1):
        d_inv = np.linalg.inv(d_cur + 1e-10 * np.eye(q_b))

        # --- Block 1: Vectorized update of all N subject random effects b_i ---
        z_norm = rng.normal(size=(n_subjects, q_b))
        b_step = scale_b[:, None] * np.einsum("nab,nb->na", chol_b, z_norm)
        b_prop = b_cur + b_step

        ss_prop = _long_ss_per_subj(beta_cur, b_prop)
        ll_surv_prop = _surv_ll_per_subj(beta_cur, b_prop, psi_cur)
        prior_b_cur = -0.5 * np.einsum("na,ab,nb->n", b_cur, d_inv, b_cur)
        prior_b_prop = -0.5 * np.einsum("na,ab,nb->n", b_prop, d_inv, b_prop)

        log_alpha_b = (
            -0.5 * (ss_prop - ss_cur) / sigma2_cur
            + (ll_surv_prop - ll_surv_cur)
            + (prior_b_prop - prior_b_cur)
        )
        u_b = np.log(rng.uniform(size=n_subjects))
        accept_mask = u_b < log_alpha_b
        if np.any(accept_mask):
            b_cur[accept_mask] = b_prop[accept_mask]
            ss_cur[accept_mask] = ss_prop[accept_mask]
            ll_surv_cur[accept_mask] = ll_surv_prop[accept_mask]
        acc_b += accept_mask.astype(np.float64)

        # --- Block 2a: Gelfand-Sahu-Carlin hierarchically-centered exact Gibbs step on beta[:q_b] ---
        # Conditioning on theta_i = beta[:q_b] + b_i leaves both longitudinal and survival
        # likelihoods strictly invariant while eliminating random-effects centering autocorrelation!
        theta_i = beta_cur[:q_b][None, :] + b_cur
        theta_bar = np.mean(theta_i, axis=0)
        prec_hc = float(n_subjects) * d_inv + (1.0 / 100.0) * np.eye(q_b)
        cov_hc = np.linalg.inv(0.5 * (prec_hc + prec_hc.T))
        mean_hc = cov_hc @ (float(n_subjects) * (d_inv @ theta_bar))
        chol_hc = np.linalg.cholesky(0.5 * (cov_hc + cov_hc.T) + 1e-12 * np.eye(q_b))
        beta_hc_new = mean_hc + chol_hc @ rng.normal(size=q_b)
        beta_cur[:q_b] = beta_hc_new
        b_cur = theta_i - beta_hc_new[None, :]

        # --- Block 2b: Joint longitudinal + survival update of fixed effects beta ---
        beta_prop = beta_cur + scale_beta * (chol_beta @ rng.normal(size=p_beta))
        ss_beta_prop = _long_ss_per_subj(beta_prop, b_cur)
        ll_surv_beta_prop = _surv_ll_per_subj(beta_prop, b_cur, psi_cur)
        prior_beta_diff = -0.5 * (float(beta_prop @ beta_prop) - float(beta_cur @ beta_cur)) / 100.0
        log_alpha_beta = (
            -0.5 * float(np.sum(ss_beta_prop) - np.sum(ss_cur)) / sigma2_cur
            + float(np.sum(ll_surv_beta_prop) - np.sum(ll_surv_cur))
            + prior_beta_diff
        )
        if np.log(rng.uniform()) < log_alpha_beta:
            beta_cur = beta_prop
            ss_cur = ss_beta_prop
            ll_surv_cur = ll_surv_beta_prop
            acc_beta += 1.0

        # --- Block 3: Exact Conjugate Inverse-Gamma update of sigma^2 ---
        shape_sig = 0.01 + 0.5 * float(n_obs)
        rate_sig = 0.01 + 0.5 * float(np.sum(ss_cur))
        sigma2_cur = float(rate_sig / rng.gamma(shape=shape_sig, scale=1.0))

        # --- Block 4: Exact Conjugate Inverse-Wishart update of D ---
        s_b = b_cur.T @ b_cur
        d_cur = _sample_invwishart(rng, df=nu_d0 + n_subjects, scale_mat=psi_d0 + s_b)

        # --- Block 5: Laplace-preconditioned update of psi_k = (xi_k, gamma_k, alpha_k) ---
        f_ev_now = eval_traj_forms(surv_times, beta_cur, b_cur, c_long, forms)
        f_q_now = eval_traj_forms(u_quad, beta_cur, b_cur, c_long, forms)

        for k in range(n_causes):
            pk_cur = psi_cur[k]
            pk_prop = pk_cur + scale_psi[k] * (chol_psi[k] @ rng.normal(size=p_psi))

            def _cause_logpost(pk: np.ndarray, kth: int) -> tuple[float, np.ndarray]:
                xi = pk[:p_spl]
                gam = pk[p_spl : p_spl + p_gam]
                alp = pk[p_spl + p_gam :]
                w_lin = w_surv @ gam if p_gam > 0 else np.zeros(n_subjects, dtype=np.float64)
                lh_ev = (b_event @ xi) + w_lin + (f_ev_now @ alp)
                lh_q = (
                    np.einsum("nqp,p->nq", b_quad, xi)
                    + w_lin[:, None]
                    + np.einsum("nqf,f->nq", f_q_now, alp)
                )
                lh_q = np.clip(lh_q, -48.0, 28.0)
                cum_h = np.sum(w_quad * np.exp(lh_q), axis=1)
                ll_vec = np.where(ev_masks[kth], lh_ev, 0.0) - cum_h
                pen_xi = -0.5 * tau_h_cur[kth] * float(xi @ (basis.penalty_matrix @ xi)) - 0.5 * 1e-2 * float(xi @ xi)
                pen_reg = -0.5 * (float(gam @ gam) + float(alp @ alp)) / 25.0
                return float(np.sum(ll_vec)) + pen_xi + pen_reg, ll_vec

            lp_cur, _ = _cause_logpost(pk_cur, k)
            lp_prop, _ = _cause_logpost(pk_prop, k)
            if np.log(rng.uniform()) < (lp_prop - lp_cur):
                psi_cur[k] = pk_prop
                lp_cur = lp_prop
                acc_psi[k] += 1.0

            # Dedicated sub-block move on (gamma_k, alpha_k) + baseline intercept shift
            reg_step = scale_reg[k] * (chol_reg[k] @ rng.normal(size=p_reg))
            pk_reg_prop = psi_cur[k].copy()
            pk_reg_prop[p_spl:] = pk_reg_prop[p_spl:] + reg_step
            # Compensate mean predictor shift in log-baseline hazard intercept (which is in nullspace of D2!)
            mean_shift = 0.0
            if p_gam > 0:
                mean_shift += float(surv_cov_means @ reg_step[:p_gam])
            mean_shift += float(np.mean(f_q_now, axis=(0, 1)) @ reg_step[p_gam:])
            pk_reg_prop[:p_spl] = pk_reg_prop[:p_spl] - mean_shift
            lp_reg_prop, _ = _cause_logpost(pk_reg_prop, k)
            if np.log(rng.uniform()) < (lp_reg_prop - lp_cur):
                psi_cur[k] = pk_reg_prop
                acc_reg[k] += 1.0

            # --- Block 6: Exact Gamma Gibbs update of spline penalty precision tau_h_k ---
            xi_now = psi_cur[k, :p_spl]
            quad_pen = float(xi_now @ (basis.penalty_matrix @ xi_now))
            tau_h_cur[k] = float(
                rng.gamma(
                    shape=a_h + 0.5 * basis.penalty_rank,
                    scale=1.0 / (b_h + 0.5 * quad_pen),
                )
            )

        ll_surv_cur = _surv_ll_per_subj(beta_cur, b_cur, psi_cur)

        # Robbins-Monro proposal adaptation strictly during burn_in (frozen post burn_in!)
        if it <= burn_in and (it % adapt_window == 0):
            rate_b = acc_b / adapt_window
            scale_b = np.clip(scale_b * np.exp(0.6 * (rate_b - 0.38)), 0.05, 5.0)
            acc_b.fill(0.0)

            rate_beta = acc_beta / adapt_window
            scale_beta = float(np.clip(scale_beta * np.exp(0.6 * (rate_beta - 0.32)), 0.05, 5.0))
            acc_beta = 0.0

            rate_psi = acc_psi / adapt_window
            scale_psi = np.clip(scale_psi * np.exp(0.6 * (rate_psi - 0.28)), 0.05, 5.0)
            acc_psi.fill(0.0)

            rate_reg = acc_reg / adapt_window
            scale_reg = np.clip(scale_reg * np.exp(0.6 * (rate_reg - 0.35)), 0.05, 5.0)
            acc_reg.fill(0.0)

            # Refresh Laplace Hessian preconditioner midway through burn_in
            if it == max(adapt_window, (burn_in // adapt_window // 2) * adapt_window):
                for k in range(n_causes):
                    _, cov_k = _solve_cause_map(
                        ev_mask_k=ev_masks[k],
                        b_event=b_event,
                        b_quad=b_quad,
                        w_surv=w_surv,
                        f_event=f_ev_now,
                        f_quad=f_q_now,
                        w_quad=w_quad,
                        pen_mat=basis.penalty_matrix,
                        tau_h=tau_h_cur[k],
                        psi_init=psi_cur[k],
                    )
                    chol_psi[k] = np.linalg.cholesky(cov_k + 1e-9 * np.eye(p_psi))
                    cov_reg_k = cov_k[p_spl:, p_spl:]
                    chol_reg[k] = np.linalg.cholesky(0.5 * (cov_reg_k + cov_reg_k.T) + 1e-9 * np.eye(p_reg))

        if it > burn_in:
            idx_s = it - burn_in - 1
            draws_beta[idx_s] = beta_cur
            draws_sigma2[idx_s] = sigma2_cur
            draws_d[idx_s] = d_cur
            draws_xi[idx_s] = psi_cur[:, :p_spl]
            draws_gamma[idx_s] = psi_cur[:, p_spl : p_spl + p_gam]
            draws_alpha[idx_s] = psi_cur[:, p_spl + p_gam :]
            draws_tau_h[idx_s] = tau_h_cur
            draws_b[idx_s] = b_cur

            ll_long_subj = -0.5 * n_obs_per_subj * np.log(2.0 * np.pi * sigma2_cur) - 0.5 * ss_cur / sigma2_cur
            draws_ll_subj[idx_s] = ll_long_subj + ll_surv_cur

    # Step 5: Compute DIC, WAIC, split-Rhat, and posterior summary DataFrames
    dev_draws = -2.0 * np.sum(draws_ll_subj, axis=1)
    d_bar = float(np.mean(dev_draws))

    beta_mean = np.mean(draws_beta, axis=0)
    sigma2_mean = float(np.mean(draws_sigma2))
    b_mean = np.mean(draws_b, axis=0)
    psi_mean = np.concatenate(
        [
            np.mean(draws_xi, axis=0),
            np.mean(draws_gamma, axis=0),
            np.mean(draws_alpha, axis=0),
        ],
        axis=1,
    )
    ss_at_mean = _long_ss_per_subj(beta_mean, b_mean)
    ll_long_mean = -0.5 * n_obs_per_subj * np.log(2.0 * np.pi * sigma2_mean) - 0.5 * ss_at_mean / sigma2_mean
    ll_surv_mean = _surv_ll_per_subj(beta_mean, b_mean, psi_mean)
    d_hat = -2.0 * float(np.sum(ll_long_mean + ll_surv_mean))
    p_d = d_bar - d_hat
    if p_d <= 0.0:
        p_d = 0.5 * float(np.var(dev_draws, ddof=1))
    dic = d_bar + p_d

    # WAIC (Vehtari, Gelman, & Gabry 2017)
    lppd_i = logsumexp(draws_ll_subj, axis=0) - np.log(float(n_keep))
    p_waic_i = np.var(draws_ll_subj, axis=0, ddof=1)
    waic = float(-2.0 * np.sum(lppd_i - p_waic_i))

    rhat_dict: dict[str, float] = {}

    # Longitudinal summary table
    long_rows: list[dict[str, Any]] = []
    for j, t_name in enumerate(lme_fit.term_names):
        dr = draws_beta[:, j]
        rh = _split_rhat(dr)
        rhat_dict[f"long:{t_name}"] = rh
        long_rows.append(
            {
                "term": t_name,
                "mean": float(np.mean(dr)),
                "sd": float(np.std(dr, ddof=1)),
                "median": float(np.median(dr)),
                "ci_lower": float(np.quantile(dr, 0.025)),
                "ci_upper": float(np.quantile(dr, 0.975)),
                "pval": _mcmc_tail_pval(dr, settings),
                "rhat": rh,
            }
        )
    sig_dr = np.sqrt(draws_sigma2)
    rh_sig = _split_rhat(sig_dr)
    rhat_dict["long:sigma"] = rh_sig
    long_rows.append(
        {
            "term": "sigma",
            "mean": float(np.mean(sig_dr)),
            "sd": float(np.std(sig_dr, ddof=1)),
            "median": float(np.median(sig_dr)),
            "ci_lower": float(np.quantile(sig_dr, 0.025)),
            "ci_upper": float(np.quantile(sig_dr, 0.975)),
            "pval": _mcmc_tail_pval(sig_dr, settings),
            "rhat": rh_sig,
        }
    )
    for r_idx in range(q_b):
        for c_idx in range(r_idx + 1):
            d_name = f"D[{r_idx},{c_idx}]"
            d_dr = draws_d[:, r_idx, c_idx]
            rh_d = _split_rhat(d_dr)
            rhat_dict[f"long:{d_name}"] = rh_d
            long_rows.append(
                {
                    "term": d_name,
                    "mean": float(np.mean(d_dr)),
                    "sd": float(np.std(d_dr, ddof=1)),
                    "median": float(np.median(d_dr)),
                    "ci_lower": float(np.quantile(d_dr, 0.025)),
                    "ci_upper": float(np.quantile(d_dr, 0.975)),
                    "pval": _mcmc_tail_pval(d_dr, settings),
                    "rhat": rh_d,
                }
            )
    long_df_summary = pd.DataFrame(long_rows)

    # Survival baseline covariate summary table
    surv_rows: list[dict[str, Any]] = []
    for k, c_val in enumerate(causes):
        for j, s_name in enumerate(s_cov_names):
            dr = draws_gamma[:, k, j]
            rh = _split_rhat(dr)
            key = f"surv_cause{c_val}:{s_name}" if competing_risks else f"surv:{s_name}"
            rhat_dict[key] = rh
            ci_lo = float(np.quantile(dr, 0.025))
            ci_hi = float(np.quantile(dr, 0.975))
            surv_rows.append(
                {
                    "cause": c_val,
                    "term": s_name,
                    "mean": float(np.mean(dr)),
                    "sd": float(np.std(dr, ddof=1)),
                    "median": float(np.median(dr)),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "hr": float(np.mean(np.exp(np.clip(dr, -20.0, 20.0)))),
                    "hr_ci_lower": float(np.exp(ci_lo)),
                    "hr_ci_upper": float(np.exp(ci_hi)),
                    "pval": _mcmc_tail_pval(dr, settings),
                    "rhat": rh,
                }
            )
    surv_df_summary = pd.DataFrame(surv_rows)

    # Association parameter summary table
    assoc_rows: list[dict[str, Any]] = []
    for k, c_val in enumerate(causes):
        for f_idx, f_name in enumerate(forms):
            dr = draws_alpha[:, k, f_idx]
            rh = _split_rhat(dr)
            key = f"assoc_cause{c_val}:{f_name}" if competing_risks else f"assoc:{f_name}"
            rhat_dict[key] = rh
            ci_lo = float(np.quantile(dr, 0.025))
            ci_hi = float(np.quantile(dr, 0.975))
            assoc_rows.append(
                {
                    "cause": c_val,
                    "form": f_name,
                    "mean": float(np.mean(dr)),
                    "sd": float(np.std(dr, ddof=1)),
                    "median": float(np.median(dr)),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "hr": float(np.mean(np.exp(np.clip(dr, -20.0, 20.0)))),
                    "hr_ci_lower": float(np.exp(ci_lo)),
                    "hr_ci_upper": float(np.exp(ci_hi)),
                    "pval": _mcmc_tail_pval(dr, settings),
                    "rhat": rh,
                }
            )
    assoc_df = pd.DataFrame(assoc_rows)

    posterior_draws = {
        "beta": draws_beta,
        "sigma2": draws_sigma2,
        "D": draws_d,
        "xi": draws_xi,
        "gamma": draws_gamma,
        "alpha": draws_alpha,
        "tau_h": draws_tau_h,
        "b": draws_b,
        "loglik_subj": draws_ll_subj,
    }

    spec = JMModelSpec(
        id_col=id_col,
        surv_time_col=surv_time_col,
        event_col=event_col,
        long_time_col=long_time_col,
        y_col=y_col,
        surv_covs=tuple(s_cov_names),
        long_covs=tuple(l_cov_names),
        random_slope=random_slope,
        functional_forms=forms,
        competing_risks=competing_risks,
        causes=causes,
        basis=basis,
        surv_cov_means=surv_cov_means,
    )

    return JMResult(
        assoc_df=assoc_df,
        surv_df_summary=surv_df_summary,
        long_df_summary=long_df_summary,
        posterior_draws=posterior_draws,
        dic=float(dic),
        waic=waic,
        rhat=rhat_dict,
        lme_fit=lme_fit,
        spec=spec,
        settings=settings,
        n_subjects=n_subjects,
        n_obs=n_obs,
        n_events=int(np.sum(ev_vec > 0)),
        n_iter=n_iter,
        burn_in=burn_in,
    )


class JMBayes2:
    """Object-oriented estimator for Bayesian Joint Longitudinal-Survival Modeling.

    Parameters
    ----------
    random_slope : Include correlated random intercept and slope in longitudinal submodel.
    functional_forms : Subset of ``("value", "slope", "area")``.
    competing_risks : Fit cause-specific competing-risks survival submodels when ``True``.
    n_iter : Total MCMC iterations.
    burn_in : Warmup iterations discarded before inference.
    n_knots : Interior B-spline knots for log-baseline hazard.
    seed : Random seed.
    settings : ``"default"`` or ``"R_default"`` (``JMbayes2::jm`` v0.5 parity).
    """

    def __init__(
        self,
        random_slope: bool = True,
        functional_forms: str | Sequence[str] = ("value",),
        competing_risks: bool = False,
        n_iter: int = 500,
        burn_in: int = 150,
        n_knots: int = 6,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        self.functional_forms = _normalize_functional_forms(functional_forms)
        self.random_slope = random_slope
        self.competing_risks = competing_risks
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n_knots = n_knots
        self.seed = seed
        self.settings: SettingsMode = settings
        self.result_: JMResult | None = None

    def fit(
        self,
        surv_df: pd.DataFrame,
        long_df: pd.DataFrame,
        id_col: str = "id",
        surv_time_col: str = "time",
        event_col: str = "event",
        long_time_col: str = "obstime",
        y_col: str = "y",
        surv_covs: Sequence[str] | None = None,
        long_covs: Sequence[str] | None = None,
    ) -> JMBayes2:
        """Fit joint longitudinal-survival model via Adaptive Metropolis-within-Gibbs."""
        self.result_ = jm(
            surv_df=surv_df,
            long_df=long_df,
            id_col=id_col,
            surv_time_col=surv_time_col,
            event_col=event_col,
            long_time_col=long_time_col,
            y_col=y_col,
            surv_covs=surv_covs,
            long_covs=long_covs,
            random_slope=self.random_slope,
            functional_forms=self.functional_forms,
            competing_risks=self.competing_risks,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            n_knots=self.n_knots,
            seed=self.seed,
            settings=self.settings,
        )
        return self

    def predict_dynamic(
        self,
        newdata_long_df: pd.DataFrame,
        t_landmark: float,
        horizons: Sequence[float] | np.ndarray,
        surv_covs_row: Mapping[str, float] | pd.Series | pd.DataFrame | None = None,
        n_mcmc: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compute individualized dynamic conditional survival / CIF predictions."""
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .predict_dynamic().")
        from .predict import predict_dynamic as _predict_dynamic

        return _predict_dynamic(
            jm_fit=self.result_,
            newdata_long_df=newdata_long_df,
            t_landmark=t_landmark,
            horizons=horizons,
            surv_covs_row=surv_covs_row,
            n_mcmc=n_mcmc,
            seed=seed,
            settings=self.settings,
        )

    def summary(self) -> str:
        """Return posterior summary report."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet; call .fit() first.")
        return self.result_.summary()

    def to_dataframe(self) -> pd.DataFrame:
        """Return combined posterior summary DataFrame."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet; call .fit() first.")
        return self.result_.to_dataframe()
