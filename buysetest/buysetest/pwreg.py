"""Proportional Win-Fractions Regression (WR::pwreg, Mao & Wang, JASA 2020)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm

from .endpoints import Endpoint, SettingsMode, validate_settings


@dataclass
class PWRegResult:
    """Result of Mao & Wang (2020) covariate-adjusted Proportional Win-Fractions Regression."""

    coef_names: list[str]
    coef: np.ndarray
    se: np.ndarray
    win_ratio: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    z_stat: np.ndarray
    pval: np.ndarray
    vcov: np.ndarray
    wald_chisq: float
    wald_pval: float
    n_subjects: int
    n_total_pairs: int
    n_informative_pairs: int
    layer_pair_counts: list[int]
    alpha: float
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Export regression log-win-ratios, conditional Win Ratios, SEs, CIs, and p-values."""
        return pd.DataFrame(
            {
                "covariate": self.coef_names,
                "log_win_ratio": self.coef,
                "se": self.se,
                "win_ratio": self.win_ratio,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
                "z_stat": self.z_stat,
                "pval": self.pval,
            }
        )

    def summary(self) -> str:
        """Format a clinical-trial Proportional Win-Fractions regression summary table."""
        df = self.to_dataframe()
        ci_pct = int(round((1.0 - self.alpha) * 100))
        lines = [
            f"Proportional Win-Fractions Regression (WR::pwreg, Mao & Wang 2020, settings='{self.settings}')",
            f"Subjects: N={self.n_subjects}, pairs={self.n_total_pairs}, "
            f"informative pairs={self.n_informative_pairs} (by layer: {self.layer_pair_counts})",
            f"Global Wald chi2={self.wald_chisq:.4f} (df={len(self.coef_names)}, p={self.wald_pval:.4g})",
            "-" * 92,
            f"{'Covariate':<18} {'log(WR)':>10} {'SE':>10} {'WinRatio':>10} "
            f"{f'{ci_pct}% CI Lower':>14} {f'{ci_pct}% CI Upper':>14} {'p-value':>11}",
            "-" * 92,
        ]
        for _, r in df.iterrows():
            lines.append(
                f"{str(r['covariate']):<18} {r['log_win_ratio']:>10.4f} {r['se']:>10.4f} "
                f"{r['win_ratio']:>10.4f} {r['ci_lower']:>14.4f} {r['ci_upper']:>14.4f} "
                f"{r['pval']:>11.4g}"
            )
        lines.append("-" * 92)
        return "\n".join(lines)


def _km_censoring_curve(
    terminal_time: np.ndarray,
    terminal_event: np.ndarray,
    g_min: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit right- and left-continuous KM censoring survival G(t) and G(t-) on terminal follow-up."""
    t = np.asarray(terminal_time, dtype=float).ravel()
    e = np.asarray(terminal_event, dtype=int).ravel()
    # Events precede censorings at ties
    order = np.lexsort((-e, t))
    ts = t[order]
    es = e[order]

    uniq_t, grp_starts = np.unique(ts, return_index=True)
    grp_tot = np.add.reduceat(np.ones_like(ts), grp_starts)
    grp_ev = np.add.reduceat((es == 1).astype(float), grp_starts)
    grp_cens = grp_tot - grp_ev

    risk_ge = np.cumsum(grp_tot[::-1])[::-1]
    risk_cens = np.maximum(risk_ge - grp_ev, 0.0)
    haz = np.where(
        (grp_cens > 0.0) & (risk_cens > 0.0),
        np.clip(grp_cens / np.maximum(risk_cens, 1e-15), 0.0, 1.0 - 1e-12),
        0.0,
    )
    g_after = np.maximum(np.cumprod(1.0 - haz), g_min)
    g_before = np.maximum(np.concatenate([[1.0], g_after[:-1]]), g_min)
    return uniq_t, g_after, g_before


def _eval_g_at(
    query_t: np.ndarray,
    uniq_t: np.ndarray,
    g_after: np.ndarray,
    g_before: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate right-continuous G(t) and left-continuous G(t-) at query times."""
    idx_r = np.searchsorted(uniq_t, query_t, side="right") - 1
    gr = np.ones_like(query_t, dtype=float)
    valid_r = idx_r >= 0
    gr[valid_r] = g_after[idx_r[valid_r]]

    idx_l = np.searchsorted(uniq_t, query_t, side="left")
    gl = np.ones_like(query_t, dtype=float)
    in_bounds = idx_l < uniq_t.size
    gl[in_bounds] = g_before[idx_l[in_bounds]]
    if uniq_t.size > 0:
        gl[~in_bounds] = g_after[-1]
    return gr, gl


def _parse_pwreg_inputs(
    data: pd.DataFrame,
    id_col: str | None,
    time_col: str | None,
    status_col: str | None,
    treatment_col: str | None,
    covariates: Sequence[str] | str | None,
    endpoints: Sequence[Endpoint] | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, str]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Support both wide subject-level DataFrames (with Endpoint list) and long WR::pwreg format."""
    cov_list: list[str] = []
    if treatment_col is not None:
        if treatment_col not in data.columns:
            raise ValueError(f"treatment_col '{treatment_col}' not found in data.")
        cov_list.append(treatment_col)
    if covariates is not None:
        raw_covs = [covariates] if isinstance(covariates, str) else list(covariates)
        for c in raw_covs:
            if c not in data.columns:
                raise ValueError(f"Covariate column '{c}' not found in data.")
            if c not in cov_list:
                cov_list.append(c)
    if len(cov_list) == 0:
        raise ValueError("Specify at least one covariate via 'treatment_col' and/or 'covariates'.")

    # Case A: Long multi-row-per-subject format matching R WR::pwreg(ID, time, status, Z)
    # where status == 1 (highest priority / fatal), status == 2 (non-fatal priority 2), ..., 0 (censored)
    if (
        endpoints is None
        and id_col is not None
        and id_col in data.columns
        and data[id_col].duplicated().any()
    ):
        if time_col is None or status_col is None:
            raise ValueError("Long multi-record format requires both 'time_col' and 'status_col'.")
        subj_ids = pd.unique(data[id_col])
        n_subj = len(subj_ids)
        status_vals = sorted(int(s) for s in pd.unique(data[status_col]) if int(s) > 0)
        if len(status_vals) == 0:
            raise ValueError("No positive event status codes found in status_col.")

        # Extract subject-level covariates and maximum follow-up time per subject
        grouped = data.groupby(id_col, sort=False)
        z_mat = grouped[cov_list].first().loc[subj_ids].to_numpy(dtype=float)
        max_time = grouped[time_col].max().loc[subj_ids].to_numpy(dtype=float)

        layers: list[tuple[np.ndarray, np.ndarray, float, str]] = []
        term_time = max_time.copy()
        term_ev = np.zeros(n_subj, dtype=int)
        id_to_idx = {sid: idx for idx, sid in enumerate(subj_ids)}

        for rank_idx, code in enumerate(status_vals):
            t_layer = max_time.copy()
            d_layer = np.zeros(n_subj, dtype=int)
            ev_rows = data[data[status_col] == code]
            for _, row in ev_rows.iterrows():
                i = id_to_idx[row[id_col]]
                t_ev = float(row[time_col])
                if d_layer[i] == 0 or t_ev < t_layer[i]:
                    t_layer[i] = t_ev
                    d_layer[i] = 1
            if rank_idx == 0:
                term_time = t_layer.copy()
                term_ev = d_layer.copy()
            layers.append((t_layer, d_layer, 0.0, "higher"))

        return layers, term_time, term_ev, z_mat, cov_list

    # Case B: Wide one-row-per-subject format
    work = data.reset_index(drop=True)
    n_subj = len(work)
    if n_subj < 3:
        raise ValueError(f"pwreg requires at least 3 subjects, got {n_subj}.")
    z_mat = work[cov_list].to_numpy(dtype=float)
    if np.any(~np.isfinite(z_mat)):
        raise ValueError("Covariates in pwreg must be finite numeric values.")

    layers = []
    if endpoints is not None:
        ep_list = list(endpoints)
        if len(ep_list) == 0:
            raise ValueError("endpoints cannot be empty when provided.")
        for ep in ep_list:
            if ep.kind != "tte":
                raise ValueError(
                    f"Proportional Win-Fractions regression (pwreg) requires 'tte' endpoints, got '{ep.kind}'."
                )
            assert ep.status_col is not None
            t_arr = work[ep.col].to_numpy(dtype=float)
            d_arr = work[ep.status_col].to_numpy(dtype=int)
            layers.append((t_arr, d_arr, float(ep.threshold), ep.operator))
        # Terminal follow-up (typically Priority-1 OS) governs administrative/loss-to-followup censoring
        term_time = layers[0][0]
        term_ev = layers[0][1]
    else:
        if time_col is None or status_col is None:
            raise ValueError("Provide either 'endpoints' or both 'time_col' and 'status_col'.")
        t_arr = work[time_col].to_numpy(dtype=float)
        d_arr = work[status_col].to_numpy(dtype=int)
        layers.append((t_arr, d_arr, 0.0, "higher"))
        term_time = t_arr
        term_ev = d_arr

    return layers, term_time, term_ev, z_mat, cov_list


def pwreg(
    data: pd.DataFrame,
    id_col: str | None = None,
    time_col: str | None = None,
    status_col: str | None = None,
    treatment_col: str | None = None,
    covariates: Sequence[str] | str | None = None,
    endpoints: Sequence[Endpoint] | None = None,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> PWRegResult:
    """Fit covariate-adjusted Proportional Win-Fractions Regression (WR::pwreg, Mao & Wang 2020).

    Estimates conditional log Win Ratios ``beta`` across prioritized composite time-to-event
    outcomes via Inverse Probability of Censoring Weighted (IPCW) pairwise score equations
    and two-degree U-statistic Hájek sandwich covariance:

    .. math::

        \\mathbf{U}_N(\\boldsymbol{\\beta}) = \\binom{N}{2}^{-1} \\sum_{1 \\le i < j \\le N}
        w_{ij}^{\\text{IPCW}} R_{ij} \\left[ \\omega_{ij} - \\sigma\\big(\\boldsymbol{\\beta}^T (\\mathbf{Z}_i - \\mathbf{Z}_j)\\big) \\right] (\\mathbf{Z}_i - \\mathbf{Z}_j) = \\mathbf{0}

    Parameters
    ----------
    data : pd.DataFrame
        Clinical trial dataset (either one row per subject with ``endpoints`` or ``time_col``/``status_col``,
        or multi-row long format keyed by ``id_col`` matching R ``WR::pwreg``).
    id_col : str | None
        Optional subject ID column (required only for multi-row long format).
    time_col, status_col : str | None
        Event/censoring time and status columns when ``endpoints`` is not supplied.
    treatment_col : str | None
        Optional treatment indicator column (automatically prepended to ``covariates``).
    covariates : Sequence[str] | str | None
        Covariate column name(s) to include in the proportional win-fractions model.
    endpoints : Sequence[Endpoint] | None
        Optional prioritized list of ``tte`` :class:`Endpoint` layers (e.g., ``[OS, PFS]``).
    alpha : float
        Significance level for confidence intervals (default ``0.05``).
    settings : {"default", "R_default"}
        ``"default"`` uses ``G_min=1e-3`` tail stabilization; ``"R_default"`` uses unfloored
        pooled KM censoring weights and exact censoring-martingale adjusted influence functions
        matching ``WR::pwreg`` (Mao & Wang, *JASA* 2020).
    """
    validate_settings(settings)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    layers, term_time, term_ev, z_mat, coef_names = _parse_pwreg_inputs(
        data=data,
        id_col=id_col,
        time_col=time_col,
        status_col=status_col,
        treatment_col=treatment_col,
        covariates=covariates,
        endpoints=endpoints,
    )
    n_subj, p = z_mat.shape
    n_total_pairs = n_subj * (n_subj - 1) // 2

    g_min = 1e-12 if settings == "R_default" else 1e-3
    uniq_t, g_after, g_before = _km_censoring_curve(term_time, term_ev, g_min=g_min)

    # All unordered subject pairs 0 <= i < j < n_subj
    idx_i, idx_j = np.triu_indices(n_subj, k=1)
    resolved = np.zeros(n_total_pairs, dtype=bool)
    omega_ij = np.zeros(n_total_pairs, dtype=float)  # 1 if i wins, 0 if j wins
    t_star = np.zeros(n_total_pairs, dtype=float)    # event resolution time min(T_i, T_j)
    layer_counts: list[int] = []

    for t_arr, d_arr, tau, operator in layers:
        ti = t_arr[idx_i]
        tj = t_arr[idx_j]
        di = d_arr[idx_i]
        dj = d_arr[idx_j]
        tau_eff = max(tau, 1e-12)

        # Subject i beats j (longer survival) if j failed at tj and ti >= tj + tau_eff
        if tau > 0:
            i_longer = (dj == 1) & ((ti - tj) >= tau - 1e-15)
            j_longer = (di == 1) & ((tj - ti) >= tau - 1e-15)
        else:
            i_longer = (dj == 1) & ((ti - tj) >= tau_eff)
            j_longer = (di == 1) & ((tj - ti) >= tau_eff)

        if operator == "lower":
            i_wins, j_wins = j_longer, i_longer
            res_time = np.where(i_wins, ti, tj)
        else:
            i_wins, j_wins = i_longer, j_longer
            res_time = np.where(i_wins, tj, ti)

        newly_resolved = (~resolved) & (i_wins | j_wins)
        n_new = int(np.sum(newly_resolved))
        layer_counts.append(n_new)
        if n_new > 0:
            omega_ij[newly_resolved] = i_wins[newly_resolved].astype(float)
            t_star[newly_resolved] = res_time[newly_resolved]
            resolved |= newly_resolved

    inf_idx = np.flatnonzero(resolved)
    n_inf = int(inf_idx.size)
    if n_inf == 0:
        raise ValueError("No determinate win/loss pairs found across any endpoint layer.")

    ii = idx_i[inf_idx]
    jj = idx_j[inf_idx]
    y_pair = omega_ij[inf_idx]  # 1.0 if i beats j, 0.0 if j beats i
    ts_pair = t_star[inf_idx]
    dz = z_mat[ii] - z_mat[jj]  # (n_inf, p)

    # IPCW weight at resolution time t_star: 1 / (G(t_star) * G(t_star-))
    # (one subject survived strictly past t_star -> G(t_star); the failing subject was uncensored up to t_star -> G(t_star-))
    g_r, g_l = _eval_g_at(ts_pair, uniq_t, g_after, g_before)
    w_ipcw = 1.0 / np.maximum(g_r * g_l, g_min ** 2)

    # Standardize dz scales internally for Newton-Raphson conditioning
    z_scales = np.std(z_mat, axis=0)
    z_scales = np.where(z_scales > 1e-12, z_scales, 1.0)
    dz_s = dz / z_scales[None, :]

    beta_s = np.zeros(p, dtype=float)
    pair_norm = 1.0 / float(n_total_pairs)
    eye_p = np.eye(p, dtype=float) * 1e-9

    for _ in range(30):
        lin = np.clip(dz_s @ beta_s, -25.0, 25.0)
        prob = expit(lin)
        resid = w_ipcw * (y_pair - prob)
        score = pair_norm * (dz_s.T @ resid)  # (p,)
        v_wt = w_ipcw * prob * (1.0 - prob)
        jac = pair_norm * ((dz_s * v_wt[:, None]).T @ dz_s) + eye_p  # (p, p)
        step = np.linalg.solve(jac, score)
        step = np.clip(step, -5.0, 5.0)
        beta_s = beta_s + step
        if float(np.max(np.abs(step))) < 1e-9:
            break

    # Transform back to original covariate units
    beta = beta_s / z_scales
    lin_final = np.clip(dz @ beta, -25.0, 25.0)
    prob_final = expit(lin_final)

    # Bread matrix A_N = (2 / (N(N-1))) sum_{i<j} w_ij * p_ij * (1 - p_ij) * dz_ij dz_ij^T
    v_wt_f = w_ipcw * prob_final * (1.0 - prob_final)
    bread = pair_norm * ((dz * v_wt_f[:, None]).T @ dz)
    bread_inv = np.linalg.pinv(bread)

    # Symmetric pairwise score kernel psi_ij = w_ij * (y_ij - sigma(beta^T dz_ij)) * dz_ij
    psi_pairs = (w_ipcw * (y_pair - prob_final))[:, None] * dz  # (n_inf, p)

    # Subject-level Hajek projection g_i = (1 / (N - 1)) sum_{j != i} psi_ij
    g_subj = np.zeros((n_subj, p), dtype=float)
    np.add.at(g_subj, ii, psi_pairs)
    np.add.at(g_subj, jj, psi_pairs)
    g_subj /= float(n_subj - 1)
    infl = 2.0 * g_subj  # degree-2 U-statistic factor 2

    if settings == "R_default":
        # Mao & Wang (2020, Theorem 1) IPCW censoring martingale influence adjustment
        # For each censoring time u_m, H(u_m) = (2 / (N(N-1))) sum_{t_star >= u_m} 2 * psi_ij
        cens_mask = term_ev == 0
        if np.any(cens_mask):
            # Sort informative pairs by ts_pair to compute suffix sums of 2 * psi_pairs
            p_ord = np.argsort(ts_pair, kind="mergesort")
            ts_sorted = ts_pair[p_ord]
            psi_sorted = 2.0 * pair_norm * psi_pairs[p_ord]
            suffix_psi = np.vstack(
                [np.cumsum(psi_sorted[::-1], axis=0)[::-1], np.zeros((1, p))]
            )
            # Distinct censoring times
            c_times = np.unique(term_time[cens_mask])
            pos_in_pairs = np.searchsorted(ts_sorted, c_times, side="left")
            h_at_c = suffix_psi[pos_in_pairs]  # (n_c, p)

            # At each c_time u, risk set Y_C(u) = sum(term_time > u) + sum(term_time == u & cens)
            for idx_u, u_val in enumerate(c_times):
                at_risk = (term_time > u_val) | ((term_time == u_val) & cens_mask)
                y_u = float(np.sum(at_risk))
                if y_u <= 1.0:
                    continue
                is_dc = (term_time == u_val) & cens_mask
                d_nc = float(np.sum(is_dc))
                # Martingale increment dM_k^C(u) = I(X_k=u, cens) - I(X_k in risk) * (d_nc / y_u)
                dm_c = is_dc.astype(float) - at_risk.astype(float) * (d_nc / y_u)
                # Influence contribution from G(t_star) * G(t_star-)
                infl += np.outer(dm_c, h_at_c[idx_u] / (y_u / float(n_subj)))

    df_denom = float(max(1, n_subj - p))
    meat = (infl.T @ infl) / (float(n_subj) * df_denom)
    vcov = bread_inv @ meat @ bread_inv

    se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    wr = np.exp(beta)
    ci_lo = np.exp(beta - z_crit * se)
    ci_hi = np.exp(beta + z_crit * se)
    z_stat = np.where(se > 1e-15, beta / np.maximum(se, 1e-15), 0.0)
    pvals = np.where(se > 1e-15, 2.0 * norm.sf(np.abs(z_stat)), 1.0)

    wald_chisq = float(beta @ np.linalg.pinv(vcov) @ beta)
    wald_pval = float(chi2.sf(max(0.0, wald_chisq), df=p))

    return PWRegResult(
        coef_names=coef_names,
        coef=beta,
        se=se,
        win_ratio=wr,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        z_stat=z_stat,
        pval=pvals,
        vcov=vcov,
        wald_chisq=wald_chisq,
        wald_pval=wald_pval,
        n_subjects=n_subj,
        n_total_pairs=n_total_pairs,
        n_informative_pairs=n_inf,
        layer_pair_counts=layer_counts,
        alpha=alpha,
        settings=settings,
    )
