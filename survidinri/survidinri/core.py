"""Core survival IDI, continuous/categorical NRI, and median improvement inference (Uno et al. 2013)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .cox_ipcw import (
    SettingsMode,
    SortedSurvivalIndex,
    build_survival_index,
    compute_ipcw_weights_indexed,
    fit_weighted_cox_risk_indexed,
    standardize_covariates,
    validate_settings,
    weighted_median,
)


@dataclass(frozen=True)
class MetricEstimate:
    """Point estimate, perturbation standard error, confidence interval, and two-sided p-value."""

    name: str
    est: float
    se: float
    lower: float
    upper: float
    pval: float


@dataclass
class SurvIDINRIResult:
    """Complete output of Uno et al. (2013) right-censored reclassification inference."""

    m1_idi: MetricEstimate
    m2_nri: MetricEstimate
    m3_med: MetricEstimate
    cat_nri: MetricEstimate | None
    nri_events: MetricEstimate
    nri_nonevents: MetricEstimate
    half_cNRI: MetricEstimate
    full_cNRI: MetricEstimate
    r2_idi: MetricEstimate
    t0: float
    n: int
    n_events_t0: int
    event_rate_t0: float
    censoring_surv_t0: float
    risk0: np.ndarray
    risk1: np.ndarray
    beta0: np.ndarray | None
    beta1: np.ndarray | None
    npert: int
    alpha: float
    settings: SettingsMode

    @property
    def half_cnri(self) -> MetricEstimate:
        """Lowercase alias for ``half_cNRI``."""
        return self.half_cNRI

    @property
    def full_cnri(self) -> MetricEstimate:
        """Lowercase alias for ``full_cNRI``."""
        return self.full_cNRI

    def to_dataframe(self) -> pd.DataFrame:
        """Convert estimated reclassification metrics into a tidy DataFrame."""
        metrics: list[MetricEstimate] = [
            self.m1_idi,
            self.m2_nri,
            self.m3_med,
            self.nri_events,
            self.nri_nonevents,
            self.half_cNRI,
            self.full_cNRI,
            self.r2_idi,
        ]
        if self.cat_nri is not None:
            metrics.append(self.cat_nri)
        return pd.DataFrame(
            [
                {
                    "metric": m.name,
                    "est": m.est,
                    "se": m.se,
                    "lower": m.lower,
                    "upper": m.upper,
                    "pval": m.pval,
                }
                for m in metrics
            ]
        )

    def summary(self) -> str:
        """Format a human-readable clinical-trial biostatistics summary table."""
        df = self.to_dataframe()
        ci_pct = int(round((1.0 - self.alpha) * 100))
        lines = [
            f"SurvIDINRI Summary (t0={self.t0}, n={self.n}, events<=t0={self.n_events_t0}, "
            f"KM event rate={self.event_rate_t0:.4f}, npert={self.npert}, settings='{self.settings}')",
            "-" * 86,
            f"{'Metric':<18} {'Estimate':>11} {'SE':>11} {f'{ci_pct}% CI Lower':>14} {f'{ci_pct}% CI Upper':>14} {'p-value':>12}",
            "-" * 86,
        ]
        for _, row in df.iterrows():
            lines.append(
                f"{str(row['metric']):<18} {row['est']:>11.5f} {row['se']:>11.5f} "
                f"{row['lower']:>14.5f} {row['upper']:>14.5f} {row['pval']:>12.4g}"
            )
        lines.append("-" * 86)
        return "\n".join(lines)


def _compute_reclassification_vector(
    p0: np.ndarray,
    p1: np.ndarray,
    w_case: np.ndarray,
    w_ctrl: np.ndarray,
    obs_weights: np.ndarray,
    thresholds: np.ndarray | None,
) -> np.ndarray:
    """Compute vector of reclassification measures under multiplier weights ``obs_weights``.

    Returns 8-element vector:
      [0] m1_disc_idi: (mean_1(p1 - p0) - mean_0(p1 - p0))
      [1] half_cnri  : 0.5 * nri_plus + 0.5 * nri_minus  on [-1, 1]
      [2] full_cnri  : nri_plus + nri_minus              on [-2, 2]
      [3] m3_med     : med_1(p1 - p0) - med_0(p1 - p0)
      [4] nri_plus   : P(p1 > p0 | T <= t0) - P(p1 < p0 | T <= t0)
      [5] nri_minus  : P(p1 < p0 | T > t0)  - P(p1 > p0 | T > t0)
      [6] r2_idi     : R^2_1(t0) - R^2_0(t0)
      [7] cat_nri    : categorical NRI on [-2, 2] (or NaN if thresholds is None)
    """
    wc = obs_weights * w_case
    wn = obs_weights * w_ctrl
    sum_c = float(np.sum(wc))
    sum_n = float(np.sum(wn))
    if sum_c <= 0.0 or sum_n <= 0.0:
        return np.full(8, np.nan, dtype=float)

    wc_norm = wc / sum_c
    wn_norm = wn / sum_n

    diff = p1 - p0
    idi_ev = float(np.sum(wc_norm * diff))
    idi_nev = float(np.sum(wn_norm * (-diff)))  # reduction in predicted risk among non-events
    m1_disc = idi_ev + idi_nev  # (mean_1(p1 - p0) - mean_0(p1 - p0))

    up = (diff > 1e-15).astype(float)
    down = (diff < -1e-15).astype(float)
    nri_plus = float(np.sum(wc_norm * (up - down)))
    nri_minus = float(np.sum(wn_norm * (down - up)))
    full_cnri = nri_plus + nri_minus
    half_cnri = 0.5 * full_cnri

    med_ev = weighted_median(diff, wc_norm)
    med_nev = weighted_median(diff, wn_norm)
    m3_med = med_ev - med_nev

    # Uno et al. (2013) R^2(t0) difference formulation: Var(p_m) / (pi_1 * (1 - pi_1))
    w_tot = wc + wn
    sum_tot = sum_c + sum_n
    pi_event = np.clip(sum_c / max(sum_tot, 1e-15), 1e-8, 1.0 - 1e-8)
    wt_norm = w_tot / max(sum_tot, 1e-15)
    mean_p0 = float(np.sum(wt_norm * p0))
    mean_p1 = float(np.sum(wt_norm * p1))
    var_p0 = float(np.sum(wt_norm * (p0 - mean_p0) ** 2))
    var_p1 = float(np.sum(wt_norm * (p1 - mean_p1) ** 2))
    denom_r2 = pi_event * (1.0 - pi_event)
    r2_idi = (var_p1 - var_p0) / denom_r2

    if thresholds is not None:
        cat0 = np.digitize(p0, thresholds)
        cat1 = np.digitize(p1, thresholds)
        c_up = (cat1 > cat0).astype(float)
        c_down = (cat1 < cat0).astype(float)
        cat_plus = float(np.sum(wc_norm * (c_up - c_down)))
        cat_minus = float(np.sum(wn_norm * (c_down - c_up)))
        cat_nri = cat_plus + cat_minus
    else:
        cat_nri = np.nan

    return np.array(
        [m1_disc, half_cnri, full_cnri, m3_med, nri_plus, nri_minus, r2_idi, cat_nri],
        dtype=float,
    )


def _summarize_metric(
    name: str,
    est: float,
    pert_samples: np.ndarray,
    alpha: float,
    settings: SettingsMode,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> MetricEstimate:
    """Compute perturbation SE, CI, and two-sided Wald p-value for a scalar statistic."""
    fin = pert_samples[np.isfinite(pert_samples)]
    if fin.size < 2:
        return MetricEstimate(name=name, est=float(est), se=np.nan, lower=np.nan, upper=np.nan, pval=np.nan)

    se = float(np.std(fin, ddof=1))
    if settings == "R_default":
        # Empirical quantile CIs from Exp(1) perturbations matching survIDINRI::IDI.INF
        lo = float(np.quantile(fin, alpha / 2.0))
        hi = float(np.quantile(fin, 1.0 - alpha / 2.0))
    else:
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        lo = float(est - z_crit * se)
        hi = float(est + z_crit * se)

    lo = float(np.clip(lo, bounds[0], bounds[1]))
    hi = float(np.clip(hi, bounds[0], bounds[1]))

    if se <= 1e-15:
        pval = 1.0 if abs(est) <= 1e-15 else 0.0
    else:
        pval = float(2.0 * norm.sf(abs(est) / se))

    return MetricEstimate(
        name=name,
        est=float(est),
        se=se,
        lower=lo,
        upper=hi,
        pval=pval,
    )


def _validate_thresholds(risk_thresholds: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if risk_thresholds is None:
        return None
    thr = np.asarray(risk_thresholds, dtype=float).ravel()
    if thr.size == 0:
        raise ValueError("risk_thresholds cannot be empty when provided.")
    if np.any(~np.isfinite(thr)) or np.any(thr <= 0.0) or np.any(thr >= 1.0):
        raise ValueError("All risk_thresholds must lie strictly inside (0, 1).")
    return np.sort(np.unique(thr))


def idi_inf(
    time: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray,
    covs0: np.ndarray | pd.DataFrame | None,
    covs1: np.ndarray | pd.DataFrame,
    t0: float,
    npert: int = 300,
    alpha: float = 0.05,
    risk_thresholds: Sequence[float] | np.ndarray | None = None,
    seed: int | np.random.Generator | None = 42,
    settings: SettingsMode = "default",
) -> SurvIDINRIResult:
    """Fit baseline & augmented Cox models and infer right-censored IDI, NRI, and MED at ``t0``.

    Clean-room implementation of ``survIDINRI::IDI.INF`` (Uno et al., *Stat Med* 2013,
    ``10.1002/sim.5647``), simultaneously perturbing both Cox partial likelihoods,
    baseline cumulative hazards, Kaplan-Meier censoring survival ``G(t)``, and IPCW
    empirical distributions via ``Exp(1)`` multiplier resampling.

    Parameters
    ----------
    time : Sequence[float] | np.ndarray
        Observed follow-up time ``X_i = min(T_i, C_i) > 0``.
    event : Sequence[int] | np.ndarray
        Right-censoring event indicator ``Delta_i in {0, 1}`` (``1`` = event, ``0`` = censored).
    covs0 : np.ndarray | pd.DataFrame | None
        Baseline model covariates ``(n, p0)`` (or ``None`` for null baseline model).
    covs1 : np.ndarray | pd.DataFrame
        Augmented model covariates ``(n, p1)``.
    t0 : float
        Clinical prediction time horizon ``t0 > 0``.
    npert : int
        Number of ``Exp(1)`` multiplier perturbation replicates (default ``300``).
    alpha : float
        Two-sided significance level for confidence intervals (default ``0.05``).
    risk_thresholds : Sequence[float] | np.ndarray | None
        Optional probability cutoffs in ``(0, 1)`` for categorical NRI (``cat_nri``).
    seed : int | np.random.Generator | None
        Random seed or generator for reproducible perturbation resampling.
    settings : {"default", "R_default"}
        ``"default"`` uses Efron Cox ties, ``G_min=1e-3``, and reports conventional ``[-2, 2]``
        full-scale ``cNRI`` in ``m2_nri``. ``"R_default"`` matches ``survIDINRI::IDI.INF`` v1.1-2
        with Breslow ties, ``[-1, 1]`` half-cNRI in ``m2_nri``, and empirical perturbation quantile CIs.

    Returns
    -------
    SurvIDINRIResult
        Point estimates, perturbation SEs, confidence intervals, and p-values for ``M1 (IDI)``,
        ``M2 (NRI)``, ``M3 (MED)``, ``NRI+``, ``NRI-``, ``R^2 IDI``, and optional ``cat_NRI``.
    """
    validate_settings(settings)
    if npert < 0:
        raise ValueError(f"npert must be non-negative, got {npert}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    thr = _validate_thresholds(risk_thresholds)
    surv_idx: SortedSurvivalIndex = build_survival_index(
        np.asarray(time, dtype=float), np.asarray(event, dtype=int), float(t0)
    )
    n = surv_idx.n

    z0_std, _, scales0 = standardize_covariates(
        covs0.to_numpy() if isinstance(covs0, pd.DataFrame) else covs0, n
    )
    z1_std, _, scales1 = standardize_covariates(
        covs1.to_numpy() if isinstance(covs1, pd.DataFrame) else covs1, n
    )
    z0_s = z0_std[surv_idx.order]
    z1_s = z1_std[surv_idx.order]

    ties: Literal["breslow", "efron"] = "breslow" if settings == "R_default" else "efron"
    g_min = 1e-12 if settings == "R_default" else 1e-3

    unit_w = np.ones(n, dtype=float)
    b0_std, _, risk0 = fit_weighted_cox_risk_indexed(surv_idx, z0_s, unit_w, ties=ties)
    b1_std, _, risk1 = fit_weighted_cox_risk_indexed(surv_idx, z1_s, unit_w, ties=ties)
    w_case, w_ctrl, g_t0 = compute_ipcw_weights_indexed(surv_idx, unit_w, g_min=g_min)

    pt_vec = _compute_reclassification_vector(risk0, risk1, w_case, w_ctrl, unit_w, thr)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    pert_mat = np.empty((npert, 8), dtype=float)
    for b in range(npert):
        xi = rng.exponential(scale=1.0, size=n)
        _, _, r0_b = fit_weighted_cox_risk_indexed(
            surv_idx, z0_s, xi, ties=ties, init_beta=b0_std, max_iter=15
        )
        _, _, r1_b = fit_weighted_cox_risk_indexed(
            surv_idx, z1_s, xi, ties=ties, init_beta=b1_std, max_iter=15
        )
        wc_b, wn_b, _ = compute_ipcw_weights_indexed(surv_idx, xi, g_min=g_min)
        pert_mat[b, :] = _compute_reclassification_vector(r0_b, r1_b, wc_b, wn_b, xi, thr)

    beta0_orig = b0_std / scales0 if scales0.size > 0 else np.zeros(0)
    beta1_orig = b1_std / scales1 if scales1.size > 0 else np.zeros(0)

    return _build_result(
        pt_vec=pt_vec,
        pert_mat=pert_mat,
        surv_idx=surv_idx,
        w_case=w_case,
        w_ctrl=w_ctrl,
        g_t0=g_t0,
        risk0=risk0,
        risk1=risk1,
        beta0=beta0_orig,
        beta1=beta1_orig,
        has_cat=(thr is not None),
        npert=npert,
        alpha=alpha,
        settings=settings,
    )


def idi_inf_predictions(
    time: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray,
    risk0: Sequence[float] | np.ndarray,
    risk1: Sequence[float] | np.ndarray,
    t0: float,
    npert: int = 300,
    alpha: float = 0.05,
    risk_thresholds: Sequence[float] | np.ndarray | None = None,
    seed: int | np.random.Generator | None = 42,
    settings: SettingsMode = "default",
) -> SurvIDINRIResult:
    """Infer right-censored IDI, NRI, and MED at ``t0`` from pre-computed risk predictions.

    Evaluates externally fitted or cross-validated ``t0``-horizon event risk predictions
    ``risk0`` and ``risk1`` in ``[0, 1]``, perturbing Kaplan-Meier censoring survival
    ``G*(t)`` and IPCW empirical distributions via ``Exp(1)`` multiplier resampling.
    """
    validate_settings(settings)
    if npert < 0:
        raise ValueError(f"npert must be non-negative, got {npert}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    thr = _validate_thresholds(risk_thresholds)
    surv_idx = build_survival_index(
        np.asarray(time, dtype=float), np.asarray(event, dtype=int), float(t0)
    )
    n = surv_idx.n

    p0 = np.asarray(risk0, dtype=float).ravel()
    p1 = np.asarray(risk1, dtype=float).ravel()
    if p0.size != n or p1.size != n:
        raise ValueError(f"risk0 ({p0.size}) and risk1 ({p1.size}) must match time length ({n}).")
    if np.any(~np.isfinite(p0)) or np.any(~np.isfinite(p1)):
        raise ValueError("risk0 and risk1 must be finite.")
    if np.any((p0 < 0.0) | (p0 > 1.0)) or np.any((p1 < 0.0) | (p1 > 1.0)):
        raise ValueError("Predicted event risks risk0 and risk1 must lie in [0, 1].")

    g_min = 1e-12 if settings == "R_default" else 1e-3
    unit_w = np.ones(n, dtype=float)
    w_case, w_ctrl, g_t0 = compute_ipcw_weights_indexed(surv_idx, unit_w, g_min=g_min)
    pt_vec = _compute_reclassification_vector(p0, p1, w_case, w_ctrl, unit_w, thr)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    pert_mat = np.empty((npert, 8), dtype=float)
    for b in range(npert):
        xi = rng.exponential(scale=1.0, size=n)
        wc_b, wn_b, _ = compute_ipcw_weights_indexed(surv_idx, xi, g_min=g_min)
        pert_mat[b, :] = _compute_reclassification_vector(p0, p1, wc_b, wn_b, xi, thr)

    return _build_result(
        pt_vec=pt_vec,
        pert_mat=pert_mat,
        surv_idx=surv_idx,
        w_case=w_case,
        w_ctrl=w_ctrl,
        g_t0=g_t0,
        risk0=p0,
        risk1=p1,
        beta0=None,
        beta1=None,
        has_cat=(thr is not None),
        npert=npert,
        alpha=alpha,
        settings=settings,
    )


def _build_result(
    pt_vec: np.ndarray,
    pert_mat: np.ndarray,
    surv_idx: SortedSurvivalIndex,
    w_case: np.ndarray,
    w_ctrl: np.ndarray,
    g_t0: float,
    risk0: np.ndarray,
    risk1: np.ndarray,
    beta0: np.ndarray | None,
    beta1: np.ndarray | None,
    has_cat: bool,
    npert: int,
    alpha: float,
    settings: SettingsMode,
) -> SurvIDINRIResult:
    m1_idi = _summarize_metric(
        "M1_IDI", pt_vec[0], pert_mat[:, 0] if npert > 0 else np.empty(0), alpha, settings, (-2.0, 2.0)
    )
    half_cnri = _summarize_metric(
        "half_cNRI", pt_vec[1], pert_mat[:, 1] if npert > 0 else np.empty(0), alpha, settings, (-1.0, 1.0)
    )
    full_cnri = _summarize_metric(
        "full_cNRI", pt_vec[2], pert_mat[:, 2] if npert > 0 else np.empty(0), alpha, settings, (-2.0, 2.0)
    )
    m3_med = _summarize_metric(
        "M3_MED", pt_vec[3], pert_mat[:, 3] if npert > 0 else np.empty(0), alpha, settings, (-2.0, 2.0)
    )
    nri_ev = _summarize_metric(
        "NRI_events", pt_vec[4], pert_mat[:, 4] if npert > 0 else np.empty(0), alpha, settings, (-1.0, 1.0)
    )
    nri_nev = _summarize_metric(
        "NRI_nonevents", pt_vec[5], pert_mat[:, 5] if npert > 0 else np.empty(0), alpha, settings, (-1.0, 1.0)
    )
    r2_idi = _summarize_metric(
        "R2_IDI", pt_vec[6], pert_mat[:, 6] if npert > 0 else np.empty(0), alpha, settings, (-2.0, 2.0)
    )

    if settings == "R_default":
        m2_nri = MetricEstimate(
            name="M2_NRI",
            est=half_cnri.est,
            se=half_cnri.se,
            lower=half_cnri.lower,
            upper=half_cnri.upper,
            pval=half_cnri.pval,
        )
    else:
        m2_nri = MetricEstimate(
            name="M2_NRI",
            est=full_cnri.est,
            se=full_cnri.se,
            lower=full_cnri.lower,
            upper=full_cnri.upper,
            pval=full_cnri.pval,
        )

    cat_est: MetricEstimate | None = None
    if has_cat:
        cat_est = _summarize_metric(
            "cat_NRI", pt_vec[7], pert_mat[:, 7] if npert > 0 else np.empty(0), alpha, settings, (-2.0, 2.0)
        )

    n_events_t0 = int(np.sum((surv_idx.time_s <= surv_idx.t0) & (surv_idx.event_s == 1)))
    sum_w = float(np.sum(w_case) + np.sum(w_ctrl))
    event_rate = float(np.sum(w_case) / max(sum_w, 1e-15))

    return SurvIDINRIResult(
        m1_idi=m1_idi,
        m2_nri=m2_nri,
        m3_med=m3_med,
        cat_nri=cat_est,
        nri_events=nri_ev,
        nri_nonevents=nri_nev,
        half_cNRI=half_cnri,
        full_cNRI=full_cnri,
        r2_idi=r2_idi,
        t0=surv_idx.t0,
        n=surv_idx.n,
        n_events_t0=n_events_t0,
        event_rate_t0=event_rate,
        censoring_surv_t0=g_t0,
        risk0=risk0,
        risk1=risk1,
        beta0=beta0,
        beta1=beta1,
        npert=npert,
        alpha=alpha,
        settings=settings,
    )


class SurvIDINRI:
    """Object-oriented interface for censored survival IDI, NRI, and MED evaluation."""

    def __init__(
        self,
        t0: float,
        npert: int = 300,
        alpha: float = 0.05,
        risk_thresholds: Sequence[float] | np.ndarray | None = None,
        seed: int | np.random.Generator | None = 42,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        self.t0 = float(t0)
        self.npert = int(npert)
        self.alpha = float(alpha)
        self.risk_thresholds = risk_thresholds
        self.seed = seed
        self.result_: SurvIDINRIResult | None = None

    def fit(
        self,
        time: Sequence[float] | np.ndarray,
        event: Sequence[int] | np.ndarray,
        covs0: np.ndarray | pd.DataFrame | None,
        covs1: np.ndarray | pd.DataFrame,
    ) -> SurvIDINRIResult:
        """Fit baseline and augmented Cox PH models and run perturbation inference."""
        self.result_ = idi_inf(
            time=time,
            event=event,
            covs0=covs0,
            covs1=covs1,
            t0=self.t0,
            npert=self.npert,
            alpha=self.alpha,
            risk_thresholds=self.risk_thresholds,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_

    def evaluate_predictions(
        self,
        time: Sequence[float] | np.ndarray,
        event: Sequence[int] | np.ndarray,
        risk0: Sequence[float] | np.ndarray,
        risk1: Sequence[float] | np.ndarray,
    ) -> SurvIDINRIResult:
        """Evaluate pre-computed ``t0``-horizon risk predictions."""
        self.result_ = idi_inf_predictions(
            time=time,
            event=event,
            risk0=risk0,
            risk1=risk1,
            t0=self.t0,
            npert=self.npert,
            alpha=self.alpha,
            risk_thresholds=self.risk_thresholds,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_
