"""Dynamic individualized survival & competing-risks prediction for JMbayes2.

Implements Rizopoulos (Biometrics 2011; 2012 Ch. 7) and Papageorgiou et al.
(Statistics in Medicine 2019) Monte Carlo conditional risk estimation given
accumulated biomarker trajectory Y_i(t_landmark) and event-free survival T_i > t_landmark.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .mcmc import JMResult, eval_traj_forms
from .quadrature import SettingsMode, gauss_kronrod_15, validate_settings


def _extract_surv_cov_vector(
    surv_covs: tuple[str, ...],
    default_means: np.ndarray,
    surv_covs_row: Mapping[str, float] | pd.Series | pd.DataFrame | None,
    newdata_subj: pd.DataFrame,
    subj_id: Any,
    id_col: str,
) -> np.ndarray:
    """Resolve baseline survival covariate vector w_* for a prediction subject."""
    p_gam = len(surv_covs)
    if p_gam == 0:
        return np.empty(0, dtype=np.float64)

    if surv_covs_row is not None:
        if isinstance(surv_covs_row, pd.DataFrame):
            if id_col in surv_covs_row.columns and subj_id in set(surv_covs_row[id_col]):
                row = surv_covs_row.loc[surv_covs_row[id_col] == subj_id].iloc[0]
            elif subj_id in surv_covs_row.index:
                row = surv_covs_row.loc[subj_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            else:
                row = surv_covs_row.iloc[0]
            return np.array([float(row[c]) for c in surv_covs], dtype=np.float64)
        return np.array([float(surv_covs_row[c]) for c in surv_covs], dtype=np.float64)

    # Check if survival covariates are already present as columns in newdata_subj
    if all(c in newdata_subj.columns for c in surv_covs) and not newdata_subj.empty:
        first_row = newdata_subj.iloc[0]
        return np.array([float(first_row[c]) for c in surv_covs], dtype=np.float64)

    return default_means.copy()


def _sample_conditional_ranef_single(
    rng: np.random.Generator,
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    z_hist: np.ndarray,
    c_long_row: np.ndarray,
    w_surv_row: np.ndarray,
    t_landmark: float,
    u_land: np.ndarray,
    w_land: np.ndarray,
    b_spl_land: np.ndarray,
    beta: np.ndarray,
    sigma2: float,
    d_mat: np.ndarray,
    xi_all: np.ndarray,
    gamma_all: np.ndarray,
    alpha_all: np.ndarray,
    forms: tuple[str, ...],
) -> np.ndarray:
    """Sample b_* ~ p(b_* | T_* > t_landmark, Y_*(t_landmark), theta^{(m)})."""
    q_b = d_mat.shape[0]
    d_inv = np.linalg.inv(d_mat + 1e-10 * np.eye(q_b))

    # Conjugate LME Gaussian posterior given Y_*(t_landmark) alone (exact initial proposal!)
    if y_hist.size > 0:
        prec_lme = d_inv + (z_hist.T @ z_hist) / sigma2
        cov_lme = np.linalg.inv(0.5 * (prec_lme + prec_lme.T))
        rhs_lme = (z_hist.T @ (y_hist - x_hist @ beta)) / sigma2
        b_cur = cov_lme @ rhs_lme
    else:
        cov_lme = d_mat.copy()
        b_cur = np.zeros(q_b, dtype=np.float64)

    chol_prop = np.linalg.cholesky(0.5 * (cov_lme + cov_lme.T) + 1e-9 * np.eye(q_b))

    if t_landmark <= 0.0:
        # At t_landmark == 0, P(T_* > 0 | b_*) = 1, so LME Gaussian posterior is exact!
        return b_cur + chol_prop @ rng.normal(size=q_b)

    n_causes = xi_all.shape[0]
    p_gam = w_surv_row.shape[0]

    def _logpost_b(b_cand: np.ndarray) -> float:
        b_2d = b_cand.reshape(1, q_b)
        lp = -0.5 * float(b_cand @ (d_inv @ b_cand))
        if y_hist.size > 0:
            resid = y_hist - (x_hist @ beta) - (z_hist @ b_cand)
            lp -= 0.5 * float(resid @ resid) / sigma2

        # Survival conditioning: -sum_{k=1}^K int_0^{t_landmark} h_k(s | b_*) ds
        f_land = eval_traj_forms(u_land.reshape(1, -1), beta, b_2d, c_long_row, forms)[0]  # (15, P_alp)
        cum_haz_tot = 0.0
        for k in range(n_causes):
            w_lin = float(w_surv_row @ gamma_all[k]) if p_gam > 0 else 0.0
            lh_q = (b_spl_land @ xi_all[k]) + w_lin + (f_land @ alpha_all[k])
            lh_q = np.clip(lh_q, -48.0, 28.0)
            cum_haz_tot += float(np.sum(w_land * np.exp(lh_q)))
        return lp - cum_haz_tot

    # Short Student-t_4 / Gaussian random-walk Metropolis chain starting at LME posterior mean
    lp_cur = _logpost_b(b_cur)
    for step_idx in range(6):
        # Heavy-tailed Student-t_4 innovation (Rizopoulos 2011)
        chi_scale = np.sqrt(4.0 / rng.chisquare(df=4.0))
        scale_rw = 1.3 if step_idx < 3 else 1.0
        b_prop = b_cur + scale_rw * chi_scale * (chol_prop @ rng.normal(size=q_b))
        lp_prop = _logpost_b(b_prop)
        if np.log(rng.uniform()) < (lp_prop - lp_cur):
            b_cur = b_prop
            lp_cur = lp_prop
    return b_cur


def predict_dynamic(
    jm_fit: JMResult,
    newdata_long_df: pd.DataFrame,
    t_landmark: float,
    horizons: Sequence[float] | np.ndarray,
    surv_covs_row: Mapping[str, float] | pd.Series | pd.DataFrame | None = None,
    n_mcmc: int = 200,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Compute individualized dynamic conditional survival or competing-risks CIF predictions.

    Evaluates:
      - Single-event: ``S_i(u | t_landmark) = P(T_i > u | T_i > t_landmark, Y_i(t_landmark))``
        with 95% credible intervals across future ``horizons`` ``u > t_landmark`` (plus
        ``event_prob = 1 - surv_prob`` when ``settings == "R_default"``).
      - Competing risks: cause-specific cumulative incidence
        ``CIF_{ik}(u | t_landmark) = P(T_i <= u, delta_i = k | T_i > t_landmark, Y_i(t_landmark))``
        and overall event-free survival ``S_i(u | t_landmark)``.

    Parameters
    ----------
    jm_fit : Fitted ``JMResult`` returned by ``jm()``.
    newdata_long_df : Longitudinal biomarker measurements for one or more subjects.
        Only observations with ``obstime <= t_landmark`` are conditioned upon.
    t_landmark : Non-negative landmark time ``t_L`` up to which the patient has survived.
    horizons : 1D sequence of future prediction times ``u > t_landmark``.
    surv_covs_row : Optional mapping, Series, or DataFrame of baseline survival covariates.
    n_mcmc : Number of Monte Carlo posterior draws for prediction uncertainty propagation.
    seed : Random seed.
    settings : ``"default"`` or ``"R_default"`` (includes ``event_prob`` matching
        ``JMbayes2::predict(..., process="event")``).
    """
    validate_settings(settings)
    if not isinstance(newdata_long_df, pd.DataFrame) or newdata_long_df.empty:
        raise ValueError("newdata_long_df must be a non-empty pandas DataFrame.")
    if not np.isfinite(t_landmark) or t_landmark < 0.0:
        raise ValueError(f"t_landmark must be finite and >= 0, got {t_landmark}.")
    if n_mcmc < 10:
        raise ValueError(f"n_mcmc must be >= 10, got {n_mcmc}.")

    horiz_arr = np.asarray(horizons, dtype=np.float64)
    if horiz_arr.ndim != 1 or horiz_arr.size == 0 or not np.all(np.isfinite(horiz_arr)):
        raise ValueError("horizons must be a non-empty 1D sequence of finite numbers.")
    if np.any(horiz_arr <= float(t_landmark)):
        raise ValueError(
            f"All prediction horizons must be strictly greater than t_landmark={t_landmark}."
        )

    spec = jm_fit.spec
    for req_col in (spec.long_time_col, spec.y_col):
        if req_col not in newdata_long_df.columns:
            raise ValueError(f"Column {req_col!r} not found in newdata_long_df.")
    for l_col in spec.long_covs:
        if l_col not in newdata_long_df.columns:
            raise ValueError(f"Longitudinal covariate {l_col!r} not found in newdata_long_df.")

    has_id = spec.id_col in newdata_long_df.columns
    work_df = newdata_long_df.copy()
    if not has_id:
        work_df[spec.id_col] = 0

    subj_ids = pd.unique(work_df[spec.id_col])
    t_l = float(t_landmark)
    n_horiz = horiz_arr.shape[0]
    n_causes = len(spec.causes)
    q_b = 2 if spec.random_slope else 1

    # Precompute GK15 quadrature grids on [0, t_L], [t_L, u_h], and nested [t_L, s_{h,q}]
    if t_l > 0.0:
        u_land, w_land = gauss_kronrod_15(0.0, t_l)
        b_spl_land = spec.basis.evaluate(u_land)  # (15, P_spl)
    else:
        u_land = np.empty(0, dtype=np.float64)
        w_land = np.empty(0, dtype=np.float64)
        b_spl_land = np.empty((0, spec.basis.n_basis), dtype=np.float64)

    u_outer, w_outer = gauss_kronrod_15(np.full(n_horiz, t_l), horiz_arr)  # (H, 15)
    b_spl_outer = spec.basis.evaluate(u_outer)  # (H, 15, P_spl)

    if spec.competing_risks:
        # Nested GK15 on [t_L, s_{h,q}] for exact CIF_k(u_h | t_L) integration
        u_inner, w_inner = gauss_kronrod_15(
            np.full(u_outer.shape, t_l), u_outer
        )  # (H, 15, 15)
        b_spl_inner = spec.basis.evaluate(u_inner)  # (H, 15, 15, P_spl)
    else:
        u_inner = np.empty((0, 0, 0), dtype=np.float64)
        w_inner = np.empty((0, 0, 0), dtype=np.float64)
        b_spl_inner = np.empty((0, 0, 0, 0), dtype=np.float64)

    draws = jm_fit.posterior_draws
    n_avail = draws["beta"].shape[0]
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, n_avail, size=n_mcmc)

    out_rows: list[dict[str, Any]] = []

    for sid in subj_ids:
        subj_all = work_df.loc[work_df[spec.id_col] == sid]
        subj_hist = subj_all.loc[subj_all[spec.long_time_col] <= t_l].sort_values(
            spec.long_time_col, kind="mergesort"
        )
        n_used = int(len(subj_hist))

        t_hist = subj_hist[spec.long_time_col].to_numpy(dtype=np.float64)
        y_hist = subj_hist[spec.y_col].to_numpy(dtype=np.float64)
        if n_used > 0:
            x_cols = [np.ones(n_used, dtype=np.float64), t_hist]
            for l_col in spec.long_covs:
                x_cols.append(subj_hist[l_col].to_numpy(dtype=np.float64))
            x_hist = np.column_stack(x_cols)
            z_hist = (
                np.column_stack([np.ones(n_used, dtype=np.float64), t_hist])
                if spec.random_slope
                else np.ones((n_used, 1), dtype=np.float64)
            )
            c_source = subj_hist.iloc[0]
        else:
            x_hist = np.empty((0, 2 + len(spec.long_covs)), dtype=np.float64)
            z_hist = np.empty((0, q_b), dtype=np.float64)
            c_source = subj_all.iloc[0]

        if spec.long_covs:
            c_long_row = np.array([[float(c_source[c]) for c in spec.long_covs]], dtype=np.float64)
        else:
            c_long_row = np.empty((1, 0), dtype=np.float64)

        w_surv_row = _extract_surv_cov_vector(
            surv_covs=spec.surv_covs,
            default_means=spec.surv_cov_means,
            surv_covs_row=surv_covs_row,
            newdata_subj=subj_all,
            subj_id=sid,
            id_col=spec.id_col,
        )
        p_gam = w_surv_row.shape[0]

        surv_mc = np.empty((n_mcmc, n_horiz), dtype=np.float64)
        cif_mc = np.empty((n_mcmc, n_horiz, n_causes), dtype=np.float64) if spec.competing_risks else np.empty((0, 0, 0), dtype=np.float64)

        for m_idx, d_idx in enumerate(draw_indices):
            beta_m = draws["beta"][d_idx]
            sigma2_m = float(draws["sigma2"][d_idx])
            d_m = draws["D"][d_idx]
            xi_m = draws["xi"][d_idx]  # (K, P_spl)
            gam_m = draws["gamma"][d_idx]  # (K, P_gam)
            alp_m = draws["alpha"][d_idx]  # (K, P_alp)

            b_star = _sample_conditional_ranef_single(
                rng=rng,
                y_hist=y_hist,
                x_hist=x_hist,
                z_hist=z_hist,
                c_long_row=c_long_row,
                w_surv_row=w_surv_row,
                t_landmark=t_l,
                u_land=u_land,
                w_land=w_land,
                b_spl_land=b_spl_land,
                beta=beta_m,
                sigma2=sigma2_m,
                d_mat=d_m,
                xi_all=xi_m,
                gamma_all=gam_m,
                alpha_all=alp_m,
                forms=spec.functional_forms,
            )
            b_2d = b_star.reshape(1, q_b)

            # Evaluate hazards on [t_L, u_h] outer GK15 nodes (H, 15)
            f_outer = eval_traj_forms(
                u_outer.reshape(1, n_horiz * 15), beta_m, b_2d, c_long_row, spec.functional_forms
            ).reshape(n_horiz, 15, -1)

            cum_haz_total = np.zeros(n_horiz, dtype=np.float64)
            haz_outer_causes = np.empty((n_causes, n_horiz, 15), dtype=np.float64)
            for k in range(n_causes):
                w_lin = float(w_surv_row @ gam_m[k]) if p_gam > 0 else 0.0
                lh_out = (
                    np.einsum("hqp,p->hq", b_spl_outer, xi_m[k])
                    + w_lin
                    + np.einsum("hqf,f->hq", f_outer, alp_m[k])
                )
                h_out = np.exp(np.clip(lh_out, -48.0, 28.0))
                haz_outer_causes[k] = h_out
                cum_haz_total += np.sum(w_outer * h_out, axis=1)

            surv_mc[m_idx] = np.exp(-cum_haz_total)

            if spec.competing_risks:
                # Evaluate inner cumulative hazards on [t_L, s_{h,q}] -> shape (H, 15, 15)
                f_inner = eval_traj_forms(
                    u_inner.reshape(1, n_horiz * 15 * 15),
                    beta_m,
                    b_2d,
                    c_long_row,
                    spec.functional_forms,
                ).reshape(n_horiz, 15, 15, -1)

                cum_inner_tot = np.zeros((n_horiz, 15), dtype=np.float64)
                for k in range(n_causes):
                    w_lin = float(w_surv_row @ gam_m[k]) if p_gam > 0 else 0.0
                    lh_in = (
                        np.einsum("hqrp,p->hqr", b_spl_inner, xi_m[k])
                        + w_lin
                        + np.einsum("hqrf,f->hqr", f_inner, alp_m[k])
                    )
                    h_in = np.exp(np.clip(lh_in, -48.0, 28.0))
                    cum_inner_tot += np.sum(w_inner * h_in, axis=2)

                surv_at_s = np.exp(-cum_inner_tot)  # (H, 15)
                for k in range(n_causes):
                    cif_mc[m_idx, :, k] = np.sum(
                        w_outer * haz_outer_causes[k] * surv_at_s, axis=1
                    )

        for h_idx, u_val in enumerate(horiz_arr):
            s_draws = surv_mc[:, h_idx]
            s_mean = float(np.mean(s_draws))
            s_sd = float(np.std(s_draws, ddof=1))
            s_lo = float(np.quantile(s_draws, 0.025))
            s_hi = float(np.quantile(s_draws, 0.975))

            if not spec.competing_risks:
                row_dict: dict[str, Any] = {
                    spec.id_col: sid,
                    "t_landmark": t_l,
                    "horizon": float(u_val),
                    "n_obs_used": n_used,
                    "surv_prob": s_mean,
                    "surv_sd": s_sd,
                    "surv_ci_lower": s_lo,
                    "surv_ci_upper": s_hi,
                }
                if settings == "R_default":
                    row_dict["event_prob"] = 1.0 - s_mean
                    row_dict["event_ci_lower"] = 1.0 - s_hi
                    row_dict["event_ci_upper"] = 1.0 - s_lo
                out_rows.append(row_dict)
            else:
                for k_idx, c_val in enumerate(spec.causes):
                    c_draws = cif_mc[:, h_idx, k_idx]
                    row_dict = {
                        spec.id_col: sid,
                        "t_landmark": t_l,
                        "horizon": float(u_val),
                        "cause": c_val,
                        "n_obs_used": n_used,
                        "cif": float(np.mean(c_draws)),
                        "cif_sd": float(np.std(c_draws, ddof=1)),
                        "cif_ci_lower": float(np.quantile(c_draws, 0.025)),
                        "cif_ci_upper": float(np.quantile(c_draws, 0.975)),
                        "surv_prob": s_mean,
                        "surv_ci_lower": s_lo,
                        "surv_ci_upper": s_hi,
                    }
                    if settings == "R_default":
                        row_dict["event_prob"] = 1.0 - s_mean
                    out_rows.append(row_dict)

    return pd.DataFrame(out_rows)
