"""Generalized Pairwise Comparisons (BuyseTest) with Gehan/Peron scoring and H-decomposition inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .endpoints import Endpoint, SettingsMode, validate_settings
from .scoring import (
    fit_km_atoms,
    score_pairs_continuous_binary,
    score_pairs_tte_gehan,
    score_pairs_tte_peron,
)


@dataclass(frozen=True)
class StatisticLayerInference:
    """Point estimates, SEs, confidence intervals, and p-values across endpoint layers."""

    est: np.ndarray
    se: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pval: np.ndarray


@dataclass
class BuyseTestResult:
    """Full hierarchical Generalized Pairwise Comparisons result (Buyse 2010; Peron et al. 2016)."""

    endpoint_names: list[str]
    thresholds: np.ndarray
    delta_favorable: np.ndarray
    delta_unfavorable: np.ndarray
    favorable: np.ndarray
    unfavorable: np.ndarray
    neutral: np.ndarray
    uninf: np.ndarray
    net_benefit: StatisticLayerInference
    win_ratio: StatisticLayerInference
    win_odds: StatisticLayerInference
    n_treatment: int
    n_control: int
    n_pairs: int
    n_strata: int
    control_level: object
    treatment_level: object
    scoring_rule: str
    pool: str
    method_inference: str
    h_order: int
    alpha: float
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Export per-layer and cumulative pairwise probabilities and test statistics."""
        return pd.DataFrame(
            {
                "endpoint": self.endpoint_names,
                "threshold": self.thresholds,
                "delta_favorable": self.delta_favorable,
                "delta_unfavorable": self.delta_unfavorable,
                "favorable": self.favorable,
                "unfavorable": self.unfavorable,
                "neutral": self.neutral,
                "uninf": self.uninf,
                "net_benefit": self.net_benefit.est,
                "net_benefit_se": self.net_benefit.se,
                "net_benefit_lower": self.net_benefit.lower,
                "net_benefit_upper": self.net_benefit.upper,
                "net_benefit_pval": self.net_benefit.pval,
                "win_ratio": self.win_ratio.est,
                "win_ratio_se": self.win_ratio.se,
                "win_ratio_lower": self.win_ratio.lower,
                "win_ratio_upper": self.win_ratio.upper,
                "win_ratio_pval": self.win_ratio.pval,
                "win_odds": self.win_odds.est,
                "win_odds_se": self.win_odds.se,
                "win_odds_lower": self.win_odds.lower,
                "win_odds_upper": self.win_odds.upper,
                "win_odds_pval": self.win_odds.pval,
            }
        )

    def summary(self) -> str:
        """Format a clinical-trial hierarchical composite endpoint summary table."""
        df = self.to_dataframe()
        lines = [
            f"BuyseTest Generalized Pairwise Comparisons "
            f"(scoring='{self.scoring_rule}', pool='{self.pool}', H-order={self.h_order}, settings='{self.settings}')",
            f"Arms: treatment='{self.treatment_level}' (m={self.n_treatment}) vs "
            f"control='{self.control_level}' (n={self.n_control}), strata={self.n_strata}, pairs={self.n_pairs}",
            "-" * 104,
            f"{'Endpoint':<16} {'P_win':>8} {'P_loss':>8} {'Neutral':>8} {'Uninf':>8} "
            f"{'NetBenefit':>11} {'WinRatio':>10} {'WinOdds':>10} {'NB p-val':>11}",
            "-" * 104,
        ]
        for _, r in df.iterrows():
            lines.append(
                f"{str(r['endpoint']):<16} {r['favorable']:>8.4f} {r['unfavorable']:>8.4f} "
                f"{r['neutral']:>8.4f} {r['uninf']:>8.4f} {r['net_benefit']:>11.4f} "
                f"{r['win_ratio']:>10.4f} {r['win_odds']:>10.4f} {r['net_benefit_pval']:>11.4g}"
            )
        lines.append("-" * 104)
        return "\n".join(lines)

    def plot(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8.0, 4.0),
        *,
        cumulative: bool = True,
        title: str | None = None,
    ) -> Figure:
        """Plot hierarchical stacked pairwise proportion bars (`buysetest.plot_buyse_test`)."""
        from .plotting import plot_buyse_test

        return plot_buyse_test(
            self,
            ax=ax,
            figsize=figsize,
            cumulative=cumulative,
            title=title,
        )


def _infer_arm_levels(
    series: pd.Series,
    control_level: object | None,
    treatment_level: object | None,
) -> tuple[object, object]:
    """Identify control and active treatment arm labels cleanly."""
    uniq = list(pd.unique(series.dropna()))
    if len(uniq) != 2:
        raise ValueError(
            f"Treatment column must have exactly 2 distinct non-null levels, got {uniq}."
        )
    if control_level is not None and treatment_level is not None:
        if control_level not in uniq or treatment_level not in uniq or control_level == treatment_level:
            raise ValueError(
                f"Specified control_level={control_level} / treatment_level={treatment_level} not in {uniq}."
            )
        return control_level, treatment_level
    if control_level is not None:
        if control_level not in uniq:
            raise ValueError(f"control_level={control_level} not found in {uniq}.")
        other = uniq[1] if uniq[0] == control_level else uniq[0]
        return control_level, other
    if treatment_level is not None:
        if treatment_level not in uniq:
            raise ValueError(f"treatment_level={treatment_level} not found in {uniq}.")
        other = uniq[1] if uniq[0] == treatment_level else uniq[0]
        return other, treatment_level

    # Smart clinical convention for string labels ("control"/"placebo" vs active)
    if all(isinstance(u, str) for u in uniq):
        low = [str(u).lower() for u in uniq]
        ctrl_keywords = {"control", "ctrl", "placebo", "pbo", "standard", "soc", "0"}
        if low[0] in ctrl_keywords and low[1] not in ctrl_keywords:
            return uniq[0], uniq[1]
        if low[1] in ctrl_keywords and low[0] not in ctrl_keywords:
            return uniq[1], uniq[0]

    try:
        s_uniq = sorted(uniq)
        return s_uniq[0], s_uniq[1]
    except TypeError:
        return uniq[0], uniq[1]


def _score_stratum_hierarchy(
    df_t: pd.DataFrame,
    df_c: pd.DataFrame,
    endpoints: list[Endpoint],
    scoring_rule: str,
    correction_uninf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Evaluate prioritized endpoints within a single stratum of m treated and n control rows.

    Returns
    -------
    delta_fav : (E,)
    delta_unfav : (E,)
    cum_neut : (E,)
    cum_uninf : (E,)
    cum_pair_tensors : list of E arrays, each of shape (m, n, 3) holding (W_<=e, L_<=e, N_e)
    """
    m = len(df_t)
    n = len(df_c)
    n_ep = len(endpoints)
    if m == 0 or n == 0:
        raise ValueError("Each stratum must contain at least 1 treatment and 1 control observation.")

    residual = np.ones((m, n), dtype=float)
    cum_w = np.zeros((m, n), dtype=float)
    cum_l = np.zeros((m, n), dtype=float)

    delta_fav = np.zeros(n_ep, dtype=float)
    delta_unfav = np.zeros(n_ep, dtype=float)
    cum_neut = np.zeros(n_ep, dtype=float)
    cum_uninf = np.zeros(n_ep, dtype=float)
    tensors: list[np.ndarray] = []

    for e_idx, ep in enumerate(endpoints):
        if ep.col not in df_t.columns:
            raise ValueError(f"Endpoint column '{ep.col}' missing from data.")
        xt = df_t[ep.col].to_numpy(dtype=float)
        xc = df_c[ep.col].to_numpy(dtype=float)

        if ep.kind == "tte":
            assert ep.status_col is not None
            if ep.status_col not in df_t.columns:
                raise ValueError(f"Endpoint status_col '{ep.status_col}' missing from data.")
            dt = df_t[ep.status_col].to_numpy(dtype=float)
            dc = df_c[ep.status_col].to_numpy(dtype=float)
            if scoring_rule == "Peron":
                km_t = fit_km_atoms(xt, dt)
                km_c = fit_km_atoms(xc, dc)
                w_raw, l_raw, n_raw, u_raw = score_pairs_tte_peron(
                    xt, dt, xc, dc, km_t, km_c, threshold=ep.threshold, operator=ep.operator
                )
            else:
                w_raw, l_raw, n_raw, u_raw = score_pairs_tte_gehan(
                    xt, dt, xc, dc, threshold=ep.threshold, operator=ep.operator
                )
        else:
            w_raw, l_raw, n_raw, u_raw = score_pairs_continuous_binary(
                xt, xc, threshold=ep.threshold, operator=ep.operator
            )

        if correction_uninf == 1:
            # IPCW-style uniform scaling across informative pairs at layer e
            mean_inf = float(np.mean(residual * (w_raw + l_raw + n_raw)))
            mean_res = float(np.mean(residual))
            if mean_inf > 1e-12 and mean_res > 1e-12:
                scale = mean_res / mean_inf
                dw = residual * w_raw * scale
                dl = residual * l_raw * scale
                dn = residual * n_raw * scale
                du = np.zeros_like(u_raw)
            else:
                dw = residual * w_raw
                dl = residual * l_raw
                dn = residual * n_raw
                du = residual * u_raw
        elif correction_uninf == 2:
            # Redistribute uninf proportionally to favorable/unfavorable/neutral
            p_w = float(np.mean(residual * w_raw))
            p_l = float(np.mean(residual * l_raw))
            p_n = float(np.mean(residual * n_raw))
            p_inf = p_w + p_l + p_n
            p_res = float(np.mean(residual))
            if p_inf > 1e-12 and p_res > 1e-12:
                scale = p_res / p_inf
                dw = residual * w_raw * scale
                dl = residual * l_raw * scale
                dn = residual * n_raw * scale
                du = np.zeros_like(u_raw)
            else:
                dw = residual * w_raw
                dl = residual * l_raw
                dn = residual * n_raw
                du = residual * u_raw
        else:
            dw = residual * w_raw
            dl = residual * l_raw
            dn = residual * n_raw
            du = residual * u_raw

        cum_w = cum_w + dw
        cum_l = cum_l + dl
        # Strictly propagate neutral + uninformative residual weight to next prioritized endpoint!
        residual = dn + du

        delta_fav[e_idx] = float(np.mean(dw))
        delta_unfav[e_idx] = float(np.mean(dl))
        cum_neut[e_idx] = float(np.mean(dn))
        cum_uninf[e_idx] = float(np.mean(du))
        tensors.append(np.stack([cum_w.copy(), cum_l.copy(), dn.copy()], axis=-1))

    return delta_fav, delta_unfav, cum_neut, cum_uninf, tensors


def _ustat_covariance_3x3(tensor_mn3: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute first-order (Hajek) or second-order H-decomposition 3x3 covariance of (W, L, N).

    Exact Hoeffding decomposition under empirical product measure (Ozenne et al. 2021):
        S_10 = (1/m) sum_i h_{1,i} h_{1,i}^T
        S_01 = (1/n) sum_j h_{0,j} h_{0,j}^T
        S_11 = (1/(m*n)) sum_{i,j} h_{2,ij} h_{2,ij}^T
        V_U^{(1)} = S_10 / m + S_01 / n
        V_U^{(2)} = S_10 / m + S_01 / n + S_11 / (m * n)
    """
    m, n, _ = tensor_mn3.shape
    theta = np.mean(tensor_mn3, axis=(0, 1))  # (3,)
    row_means = np.mean(tensor_mn3, axis=1)   # (m, 3)
    col_means = np.mean(tensor_mn3, axis=0)   # (n, 3)

    h1 = row_means - theta[None, :]  # (m, 3)
    h0 = col_means - theta[None, :]  # (n, 3)

    s10 = (h1.T @ h1) / float(m)
    s01 = (h0.T @ h0) / float(n)
    v_u = s10 / float(m) + s01 / float(n)

    if order >= 2:
        h2 = (
            tensor_mn3
            - row_means[:, None, :]
            - col_means[None, :, :]
            + theta[None, None, :]
        )
        h2_flat = h2.reshape(m * n, 3)
        s11 = (h2_flat.T @ h2_flat) / float(m * n)
        v_u = v_u + s11 / float(m * n)

    return v_u


def _delta_method_inference(
    p_win: np.ndarray,
    p_loss: np.ndarray,
    p_neut: np.ndarray,
    cov_list: list[np.ndarray],
    alpha: float,
) -> tuple[StatisticLayerInference, StatisticLayerInference, StatisticLayerInference]:
    """Compute transformed CIs and p-values for Net Benefit (atanh), Win Ratio (log), Win Odds (log)."""
    n_ep = p_win.size
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    nb_est = p_win - p_loss
    nb_se = np.empty(n_ep, dtype=float)
    nb_lo = np.empty(n_ep, dtype=float)
    nb_hi = np.empty(n_ep, dtype=float)
    nb_pv = np.empty(n_ep, dtype=float)

    wr_est = np.empty(n_ep, dtype=float)
    wr_se = np.empty(n_ep, dtype=float)
    wr_lo = np.empty(n_ep, dtype=float)
    wr_hi = np.empty(n_ep, dtype=float)
    wr_pv = np.empty(n_ep, dtype=float)

    wo_est = np.empty(n_ep, dtype=float)
    wo_se = np.empty(n_ep, dtype=float)
    wo_lo = np.empty(n_ep, dtype=float)
    wo_hi = np.empty(n_ep, dtype=float)
    wo_pv = np.empty(n_ep, dtype=float)

    g_nb = np.array([1.0, -1.0, 0.0])

    for e in range(n_ep):
        v = cov_list[e]
        pw, pl, pn = float(p_win[e]), float(p_loss[e]), float(p_neut[e])

        # 1. Net Benefit with Fisher atanh transformation
        d_val = float(nb_est[e])
        var_nb = max(0.0, float(g_nb @ v @ g_nb))
        se_d = float(np.sqrt(var_nb))
        nb_se[e] = se_d
        if abs(d_val) >= 1.0 - 1e-12:
            nb_lo[e], nb_hi[e] = d_val, d_val
            nb_pv[e] = 0.0 if abs(d_val) > 0 else 1.0
        elif se_d <= 1e-15:
            nb_lo[e], nb_hi[e] = d_val, d_val
            nb_pv[e] = 1.0 if abs(d_val) <= 1e-15 else 0.0
        else:
            z_at = float(np.arctanh(d_val))
            se_at = se_d / max(1.0 - d_val ** 2, 1e-12)
            nb_lo[e] = float(np.tanh(z_at - z_crit * se_at))
            nb_hi[e] = float(np.tanh(z_at + z_crit * se_at))
            nb_pv[e] = float(2.0 * norm.sf(abs(z_at) / se_at))

        # 2. Win Ratio = P_win / P_loss with log transformation
        if pw > 1e-15 and pl > 1e-15:
            wr = pw / pl
            wr_est[e] = wr
            g_lwr = np.array([1.0 / pw, -1.0 / pl, 0.0])
            se_lwr = float(np.sqrt(max(0.0, float(g_lwr @ v @ g_lwr))))
            wr_se[e] = wr * se_lwr
            if se_lwr <= 1e-15:
                wr_lo[e], wr_hi[e] = wr, wr
                wr_pv[e] = 1.0 if abs( np.log(wr) ) <= 1e-15 else 0.0
            else:
                lwr = float(np.log(wr))
                wr_lo[e] = float(np.exp(lwr - z_crit * se_lwr))
                wr_hi[e] = float(np.exp(lwr + z_crit * se_lwr))
                wr_pv[e] = float(2.0 * norm.sf(abs(lwr) / se_lwr))
        else:
            wr_est[e] = np.inf if (pw > 1e-15 and pl <= 1e-15) else (0.0 if pl > 1e-15 else np.nan)
            wr_se[e], wr_lo[e], wr_hi[e], wr_pv[e] = np.nan, np.nan, np.nan, np.nan

        # 3. Win Odds = (P_win + 0.5*P_neut) / (P_loss + 0.5*P_neut) with log transformation
        a_wo = pw + 0.5 * pn
        b_wo = pl + 0.5 * pn
        if a_wo > 1e-15 and b_wo > 1e-15:
            wo = a_wo / b_wo
            wo_est[e] = wo
            g_lwo = np.array([1.0 / a_wo, -1.0 / b_wo, 0.5 / a_wo - 0.5 / b_wo])
            se_lwo = float(np.sqrt(max(0.0, float(g_lwo @ v @ g_lwo))))
            wo_se[e] = wo * se_lwo
            if se_lwo <= 1e-15:
                wo_lo[e], wo_hi[e] = wo, wo
                wo_pv[e] = 1.0 if abs(np.log(wo)) <= 1e-15 else 0.0
            else:
                lwo = float(np.log(wo))
                wo_lo[e] = float(np.exp(lwo - z_crit * se_lwo))
                wo_hi[e] = float(np.exp(lwo + z_crit * se_lwo))
                wo_pv[e] = float(2.0 * norm.sf(abs(lwo) / se_lwo))
        else:
            wo_est[e], wo_se[e], wo_lo[e], wo_hi[e], wo_pv[e] = np.nan, np.nan, np.nan, np.nan, np.nan

    return (
        StatisticLayerInference(nb_est, nb_se, nb_lo, nb_hi, nb_pv),
        StatisticLayerInference(wr_est, wr_se, wr_lo, wr_hi, wr_pv),
        StatisticLayerInference(wo_est, wo_se, wo_lo, wo_hi, wo_pv),
    )


def buyse_test(
    data: pd.DataFrame,
    treatment: str,
    endpoints: Sequence[Endpoint] | Endpoint,
    strata: str | Sequence[str] | None = None,
    scoring_rule: Literal["Peron", "Gehan"] | None = None,
    correction_uninf: int = 0,
    pool: Literal["Buyse", "CMH", "equal"] = "Buyse",
    method_inference: Literal["u-statistic", "bootstrap", "permutation", "none"] = "u-statistic",
    n_resampling: int = 0,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = 42,
    control_level: object | None = None,
    treatment_level: object | None = None,
    settings: SettingsMode = "default",
) -> BuyseTestResult:
    """Perform Generalized Pairwise Comparisons across prioritized composite endpoints.

    Clean-room implementation of ``BuyseTest::BuyseTest`` (Buyse, *Stat Med* 2010;
    Péron et al., *Stat Med* 2016, ``10.1002/sim.6950``; Ozenne et al., *J Stat Softw* 2021).

    Parameters
    ----------
    data : pd.DataFrame
        Subject-level clinical trial dataset.
    treatment : str
        Binary treatment arm column name.
    endpoints : Sequence[Endpoint] | Endpoint
        Ordered list of prioritized :class:`Endpoint` specifications (highest priority first).
    strata : str | Sequence[str] | None
        Optional stratification column(s); comparisons are performed strictly within strata
        and pooled according to ``pool``.
    scoring_rule : {"Peron", "Gehan"} | None
        Pairwise right-censoring rule for ``tte`` endpoints (defaults to ``"Peron"``).
    correction_uninf : {0, 1, 2}
        Handling of uninformative pairs: ``0`` propagates ``neutral + uninf`` strictly to the
        next endpoint; ``1`` applies IPCW pair scaling; ``2`` redistributes proportionally.
    pool : {"Buyse", "CMH", "equal"}
        Stratum pooling weights: ``"Buyse"`` (``m_s * n_s``), ``"CMH"`` (``m_s * n_s / (m_s + n_s)``),
        or ``"equal"`` (``1 / S``).
    method_inference : {"u-statistic", "bootstrap", "permutation", "none"}
        Variance/CI inference engine. When ``n_resampling > 0`` and ``method_inference == "u-statistic"``,
        switches automatically to stratified non-parametric bootstrap unless ``"permutation"`` is chosen.
    n_resampling : int
        Number of bootstrap or permutation resamples when ``> 0``.
    alpha : float
        Two-sided significance level (default ``0.05``).
    seed : int | np.random.Generator | None
        Random seed for resampling.
    control_level, treatment_level : object | None
        Optional explicit control and active treatment arm values.
    settings : {"default", "R_default"}
        ``"default"`` uses first-order Hájek H-projection variance (``order=1``);
        ``"R_default"`` uses exact ``BuyseTest`` v3.3.9 second-order finite-sample
        H-decomposition variance (``order=2``) with ``atanh``/``log`` transformations.
    """
    validate_settings(settings)
    if treatment not in data.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in data.")

    ep_list = [endpoints] if isinstance(endpoints, Endpoint) else list(endpoints)
    if len(ep_list) == 0:
        raise ValueError("endpoints cannot be empty.")

    eff_rule = "Peron" if scoring_rule is None else scoring_rule
    if eff_rule not in ("Peron", "Gehan"):
        raise ValueError(f"Unknown scoring_rule '{eff_rule}'. Must be 'Peron' or 'Gehan'.")
    if correction_uninf not in (0, 1, 2):
        raise ValueError(f"correction_uninf must be 0, 1, or 2, got {correction_uninf}.")
    if pool not in ("Buyse", "CMH", "equal"):
        raise ValueError(f"Unknown pool method '{pool}'. Must be 'Buyse', 'CMH', or 'equal'.")
    if method_inference not in ("u-statistic", "bootstrap", "permutation", "none"):
        raise ValueError(f"Unknown method_inference '{method_inference}'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    ctrl_val, trt_val = _infer_arm_levels(data[treatment], control_level, treatment_level)

    if strata is None:
        strata_groups = [("__all__", data)]
    else:
        s_cols = [strata] if isinstance(strata, str) else list(strata)
        for sc in s_cols:
            if sc not in data.columns:
                raise ValueError(f"Stratum column '{sc}' not found in data.")
        strata_groups = list(data.groupby(s_cols, sort=False, dropna=False))

    h_order = 2 if settings == "R_default" else 1
    n_ep = len(ep_list)

    stratum_dfs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    raw_weights: list[float] = []
    total_m = 0
    total_n = 0
    total_pairs = 0

    for _, sub_df in strata_groups:
        df_t = sub_df[sub_df[treatment] == trt_val]
        df_c = sub_df[sub_df[treatment] == ctrl_val]
        m_s, n_s = len(df_t), len(df_c)
        if m_s == 0 or n_s == 0:
            raise ValueError(
                f"Every stratum must have both treatment ({m_s}) and control ({n_s}) subjects."
            )
        stratum_dfs.append((df_t, df_c))
        total_m += m_s
        total_n += n_s
        total_pairs += m_s * n_s
        if pool == "Buyse":
            raw_weights.append(float(m_s * n_s))
        elif pool == "CMH":
            raw_weights.append(float(m_s * n_s) / float(m_s + n_s))
        else:
            raw_weights.append(1.0)

    w_strata = np.asarray(raw_weights, dtype=float)
    w_strata /= float(np.sum(w_strata))

    pooled_dfav = np.zeros(n_ep, dtype=float)
    pooled_dunfav = np.zeros(n_ep, dtype=float)
    pooled_neut = np.zeros(n_ep, dtype=float)
    pooled_uninf = np.zeros(n_ep, dtype=float)
    pooled_covs = [np.zeros((3, 3), dtype=float) for _ in range(n_ep)]

    for s_idx, (df_t, df_c) in enumerate(stratum_dfs):
        ws = float(w_strata[s_idx])
        dfav_s, dunfav_s, neut_s, uninf_s, tensors_s = _score_stratum_hierarchy(
            df_t, df_c, ep_list, eff_rule, correction_uninf
        )
        pooled_dfav += ws * dfav_s
        pooled_dunfav += ws * dunfav_s
        pooled_neut += ws * neut_s
        pooled_uninf += ws * uninf_s
        for e in range(n_ep):
            pooled_covs[e] += (ws ** 2) * _ustat_covariance_3x3(tensors_s[e], order=h_order)

    cum_fav = np.cumsum(pooled_dfav)
    cum_unfav = np.cumsum(pooled_dunfav)

    eff_method = method_inference
    if n_resampling > 0 and eff_method == "u-statistic":
        eff_method = "bootstrap"

    if eff_method == "u-statistic":
        nb_inf, wr_inf, wo_inf = _delta_method_inference(
            cum_fav, cum_unfav, pooled_neut, pooled_covs, alpha
        )
    elif eff_method in ("bootstrap", "permutation") and n_resampling > 0:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        boot_nb = np.empty((n_resampling, n_ep), dtype=float)
        boot_wr = np.empty((n_resampling, n_ep), dtype=float)
        boot_wo = np.empty((n_resampling, n_ep), dtype=float)

        for b in range(n_resampling):
            b_dfav = np.zeros(n_ep, dtype=float)
            b_dunfav = np.zeros(n_ep, dtype=float)
            b_neut = np.zeros(n_ep, dtype=float)
            for s_idx, (df_t, df_c) in enumerate(stratum_dfs):
                ws = float(w_strata[s_idx])
                m_s, n_s = len(df_t), len(df_c)
                if eff_method == "bootstrap":
                    sub_t = df_t.iloc[rng.integers(0, m_s, size=m_s)]
                    sub_c = df_c.iloc[rng.integers(0, n_s, size=n_s)]
                else:
                    comb = pd.concat([df_t, df_c], axis=0, ignore_index=True)
                    perm = rng.permutation(m_s + n_s)
                    sub_t = comb.iloc[perm[:m_s]]
                    sub_c = comb.iloc[perm[m_s:]]
                df_b, du_b, ne_b, _, _ = _score_stratum_hierarchy(
                    sub_t, sub_c, ep_list, eff_rule, correction_uninf
                )
                b_dfav += ws * df_b
                b_dunfav += ws * du_b
                b_neut += ws * ne_b
            cf_b = np.cumsum(b_dfav)
            cu_b = np.cumsum(b_dunfav)
            boot_nb[b] = cf_b - cu_b
            boot_wr[b] = np.where(cu_b > 1e-15, cf_b / np.maximum(cu_b, 1e-15), np.nan)
            a_b = cf_b + 0.5 * b_neut
            d_b = cu_b + 0.5 * b_neut
            boot_wo[b] = np.where(d_b > 1e-15, a_b / np.maximum(d_b, 1e-15), np.nan)

        nb_est = cum_fav - cum_unfav
        wr_est = np.where(cum_unfav > 1e-15, cum_fav / np.maximum(cum_unfav, 1e-15), np.nan)
        wo_est = (cum_fav + 0.5 * pooled_neut) / np.maximum(cum_unfav + 0.5 * pooled_neut, 1e-15)

        def _resamp_inf(
            est_v: np.ndarray, samp_mat: np.ndarray, null_val: float
        ) -> StatisticLayerInference:
            se_v = np.nanstd(samp_mat, axis=0, ddof=1)
            if eff_method == "bootstrap":
                lo_v = np.nanquantile(samp_mat, alpha / 2.0, axis=0)
                hi_v = np.nanquantile(samp_mat, 1.0 - alpha / 2.0, axis=0)
                pv_v = np.array(
                    [
                        float(2.0 * norm.sf(abs(est_v[k] - null_val) / max(se_v[k], 1e-15)))
                        for k in range(n_ep)
                    ]
                )
            else:
                lo_v = np.full(n_ep, np.nan)
                hi_v = np.full(n_ep, np.nan)
                pv_v = np.array(
                    [
                        float(
                            (1.0 + np.sum(np.abs(samp_mat[:, k] - null_val) >= abs(est_v[k] - null_val)))
                            / (n_resampling + 1.0)
                        )
                        for k in range(n_ep)
                    ]
                )
            return StatisticLayerInference(est_v, se_v, lo_v, hi_v, pv_v)

        nb_inf = _resamp_inf(nb_est, boot_nb, 0.0)
        wr_inf = _resamp_inf(wr_est, boot_wr, 1.0)
        wo_inf = _resamp_inf(wo_est, boot_wo, 1.0)
    else:
        nan_v = np.full(n_ep, np.nan)
        nb_inf = StatisticLayerInference(cum_fav - cum_unfav, nan_v, nan_v, nan_v, nan_v)
        wr_inf = StatisticLayerInference(
            np.where(cum_unfav > 1e-15, cum_fav / np.maximum(cum_unfav, 1e-15), np.nan),
            nan_v, nan_v, nan_v, nan_v,
        )
        wo_inf = StatisticLayerInference(
            (cum_fav + 0.5 * pooled_neut) / np.maximum(cum_unfav + 0.5 * pooled_neut, 1e-15),
            nan_v, nan_v, nan_v, nan_v,
        )

    ep_names = [ep.col for ep in ep_list]
    thresholds = np.array([ep.threshold for ep in ep_list], dtype=float)

    return BuyseTestResult(
        endpoint_names=ep_names,
        thresholds=thresholds,
        delta_favorable=pooled_dfav,
        delta_unfavorable=pooled_dunfav,
        favorable=cum_fav,
        unfavorable=cum_unfav,
        neutral=pooled_neut,
        uninf=pooled_uninf,
        net_benefit=nb_inf,
        win_ratio=wr_inf,
        win_odds=wo_inf,
        n_treatment=total_m,
        n_control=total_n,
        n_pairs=total_pairs,
        n_strata=len(stratum_dfs),
        control_level=ctrl_val,
        treatment_level=trt_val,
        scoring_rule=eff_rule,
        pool=pool,
        method_inference=eff_method,
        h_order=h_order,
        alpha=alpha,
        settings=settings,
    )


class BuyseTest:
    """Object-oriented estimator wrapper for Generalized Pairwise Comparisons."""

    def __init__(
        self,
        scoring_rule: Literal["Peron", "Gehan"] | None = None,
        correction_uninf: int = 0,
        pool: Literal["Buyse", "CMH", "equal"] = "Buyse",
        method_inference: Literal["u-statistic", "bootstrap", "permutation", "none"] = "u-statistic",
        n_resampling: int = 0,
        alpha: float = 0.05,
        seed: int | np.random.Generator | None = 42,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        self.scoring_rule = scoring_rule
        self.correction_uninf = correction_uninf
        self.pool = pool
        self.method_inference = method_inference
        self.n_resampling = n_resampling
        self.alpha = alpha
        self.seed = seed
        self.result_: BuyseTestResult | None = None

    def fit(
        self,
        data: pd.DataFrame,
        treatment: str,
        endpoints: Sequence[Endpoint] | Endpoint,
        strata: str | Sequence[str] | None = None,
        control_level: object | None = None,
        treatment_level: object | None = None,
    ) -> BuyseTestResult:
        """Fit hierarchical Generalized Pairwise Comparisons on ``data``."""
        self.result_ = buyse_test(
            data=data,
            treatment=treatment,
            endpoints=endpoints,
            strata=strata,
            scoring_rule=self.scoring_rule,
            correction_uninf=self.correction_uninf,
            pool=self.pool,
            method_inference=self.method_inference,
            n_resampling=self.n_resampling,
            alpha=self.alpha,
            seed=self.seed,
            control_level=control_level,
            treatment_level=treatment_level,
            settings=self.settings,
        )
        return self.result_

    def plot(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8.0, 4.0),
        *,
        cumulative: bool = True,
        title: str | None = None,
    ) -> Figure:
        """Plot hierarchical stacked pairwise proportion bars of fitted model."""
        if self.result_ is None:
            raise RuntimeError("Call .fit(...) before .plot().")
        return self.result_.plot(
            ax=ax,
            figsize=figsize,
            cumulative=cumulative,
            title=title,
        )
