"""Core Kaplan-Meier RMST/RMTL estimators and two-sample / ANCOVA inference.

Clean-room implementation of survRM2 (Uno et al., JCO 2014,
doi:10.1200/JCO.2014.55.2208; Tian et al., Biostatistics 2014,
doi:10.1093/biostatistics/kxt050).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .ancova import (
    RMST2AdjustedModelResult,
    SettingsMode,
    VALID_LINKS,
    rmst2reg,
    validate_settings,
)


@dataclass(frozen=True)
class ContrastResult:
    """Point estimate, standard error, confidence interval, and p-value for a contrast."""

    label: str
    est: float
    se: float
    ci_lower: float
    ci_upper: float
    z: float
    pval: float

    @property
    def ci(self) -> tuple[float, float]:
        """Return (ci_lower, ci_upper) tuple."""
        return (self.ci_lower, self.ci_upper)

    def __getitem__(self, key: str) -> Any:
        mapping = self.to_dict()
        if key not in mapping:
            raise KeyError(f"Unknown key {key!r}; available keys: {list(mapping)}")
        return mapping[key]

    def to_dict(self) -> dict[str, Any]:
        """Convert contrast estimate to dictionary."""
        return {
            "label": self.label,
            "est": self.est,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "pval": self.pval,
            "z": self.z,
        }


@dataclass(frozen=True)
class RMST1Result:
    """Single-arm Restricted Mean Survival Time (RMST) and RMTL result."""

    tau: float
    alpha: float
    settings: SettingsMode
    n: int
    n_events: int
    rmst: float
    var_rmst: float
    se_rmst: float
    rmst_ci: tuple[float, float]
    rmtl: float
    var_rmtl: float
    se_rmtl: float
    rmtl_ci: tuple[float, float]
    km_df: pd.DataFrame
    km_step_time: np.ndarray | None = None
    km_step_surv: np.ndarray | None = None
    km_full_df: pd.DataFrame | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return a 2-row DataFrame summarizing RMST and RMTL."""
        return pd.DataFrame(
            [
                {
                    "measure": "RMST",
                    "tau": self.tau,
                    "est": self.rmst,
                    "var": self.var_rmst,
                    "se": self.se_rmst,
                    "ci_lower": self.rmst_ci[0],
                    "ci_upper": self.rmst_ci[1],
                },
                {
                    "measure": "RMTL",
                    "tau": self.tau,
                    "est": self.rmtl,
                    "var": self.var_rmtl,
                    "se": self.se_rmtl,
                    "ci_lower": self.rmtl_ci[0],
                    "ci_upper": self.rmtl_ci[1],
                },
            ]
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        pct = int(round((1.0 - self.alpha) * 100))
        lines = [
            f"Single-Arm RMST & RMTL (tau = {self.tau:.4f}, n = {self.n}, events = {self.n_events}, settings = {self.settings!r})",
            f"  RMST: {self.rmst:.4f} (SE = {self.se_rmst:.4f}, {pct}% CI: [{self.rmst_ci[0]:.4f}, {self.rmst_ci[1]:.4f}])",
            f"  RMTL: {self.rmtl:.4f} (SE = {self.se_rmtl:.4f}, {pct}% CI: [{self.rmtl_ci[0]:.4f}, {self.rmtl_ci[1]:.4f}])",
        ]
        return "\n".join(lines)

    def plot(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (7.0, 4.5),
        *,
        title: str | None = None,
    ) -> Figure:
        """Plot single-arm Kaplan-Meier curve shading RMST vs RMTL (`survrm2.plot_rmst1`)."""
        from .plotting import plot_rmst1

        return plot_rmst1(self, ax=ax, figsize=figsize, title=title)


@dataclass(frozen=True)
class RMST2Result:
    """Two-sample unadjusted and/or covariate-adjusted RMST & RMTL inference result."""

    tau: float
    alpha: float
    settings: SettingsMode
    adjusted: bool
    arm1: RMST1Result
    arm0: RMST1Result
    rmst_diff_10: ContrastResult
    rmst_ratio_10: ContrastResult
    rmtl_ratio_10: ContrastResult
    unadjusted_df: pd.DataFrame
    adjusted_models: dict[str, RMST2AdjustedModelResult] | None = None
    adjusted_df: pd.DataFrame | None = None
    adj_rmst_diff_10: ContrastResult | None = None
    adj_rmst_ratio_10: ContrastResult | None = None
    adj_rmtl_ratio_10: ContrastResult | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return the primary between-group contrast table."""
        if self.adjusted and self.adjusted_df is not None:
            return self.adjusted_df.copy()
        return self.unadjusted_df.copy()

    def summary(self) -> str:
        """Return a formatted clinical-trial RMST/RMTL report."""
        lines = [
            f"Two-Sample RMST Analysis (tau = {self.tau:.4f}, alpha = {self.alpha}, settings = {self.settings!r})",
            f"  Arm 1 (n={self.arm1.n}): RMST = {self.arm1.rmst:.4f} (SE {self.arm1.se_rmst:.4f}), RMTL = {self.arm1.rmtl:.4f}",
            f"  Arm 0 (n={self.arm0.n}): RMST = {self.arm0.rmst:.4f} (SE {self.arm0.se_rmst:.4f}), RMTL = {self.arm0.rmtl:.4f}",
            "",
            "Unadjusted Between-Group Contrasts:",
            self.unadjusted_df.to_string(index=False),
        ]
        if self.adjusted and self.adjusted_df is not None:
            lines.extend(
                [
                    "",
                    "Covariate-Adjusted IPCW Contrasts (Tian et al. 2014):",
                    self.adjusted_df.to_string(index=False),
                ]
            )
        return "\n".join(lines)

    def plot(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (7.5, 4.8),
        *,
        arm_labels: tuple[str, str] = ("Control (Arm 0)", "Active (Arm 1)"),
        title: str | None = None,
    ) -> Figure:
        """Plot two-arm Kaplan-Meier step curves shaded over `[0, tau]` (`survrm2.plot_rmst2`)."""
        from .plotting import plot_rmst2

        return plot_rmst2(
            self,
            ax=ax,
            figsize=figsize,
            arm_labels=arm_labels,
            title=title,
        )


def _validate_time_status(
    time: Sequence[float] | np.ndarray,
    status: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate survival follow-up times and binary event indicators."""
    t_arr = np.asarray(time, dtype=np.float64)
    d_arr_raw = np.asarray(status)
    if t_arr.ndim != 1 or d_arr_raw.ndim != 1:
        raise ValueError("time and status must be 1D sequences.")
    if t_arr.shape[0] == 0:
        raise ValueError("time and status must be non-empty.")
    if t_arr.shape[0] != d_arr_raw.shape[0]:
        raise ValueError(
            f"Length mismatch: time has {t_arr.shape[0]} elements, status has {d_arr_raw.shape[0]}."
        )
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("time must contain finite non-negative numbers.")
    if not np.all(np.isfinite(d_arr_raw)):
        raise ValueError("status must contain finite values in {0, 1}.")
    if not np.all(np.isin(d_arr_raw, [0, 1])):
        raise ValueError("status must contain only binary event indicators in {0, 1}.")
    return t_arr, d_arr_raw.astype(np.int64)


def _validate_arm(arm: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """Validate binary arm indicator in {0, 1}."""
    a_raw = np.asarray(arm)
    if a_raw.ndim != 1 or a_raw.shape[0] != n:
        raise ValueError(f"arm must be a 1D array of length {n}.")
    if not np.all(np.isfinite(a_raw)) or not np.all(np.isin(a_raw, [0, 1])):
        raise ValueError("arm must contain only binary group indicators in {0, 1}.")
    a_arr = a_raw.astype(np.int64)
    if np.sum(a_arr == 0) < 2 or np.sum(a_arr == 1) < 2:
        raise ValueError("Both arm == 0 and arm == 1 must contain at least 2 observations.")
    return a_arr


def _km_tau_upper_limit(t_arr: np.ndarray, d_arr: np.ndarray) -> float:
    """Determine maximum admissible truncation horizon tau for a single arm.

    Matches survRM2: if the Kaplan-Meier curve reaches 0 at max(time) (i.e. all
    observations tied at max(time) are uncensored events), S_hat(t) = 0 is
    well-defined for all t >= max(time) so tau_upper = +inf. Otherwise (at least
    one censored observation at max(time), so S_hat(max(time)) > 0), KM is
    unidentified past max(time) and tau_upper = max(time).
    """
    t_max = float(np.max(t_arr))
    at_max_status = d_arr[t_arr == t_max]
    if np.all(at_max_status == 1):
        return float("inf")
    return t_max


def rmst1(
    time: Sequence[float] | np.ndarray,
    status: Sequence[int] | np.ndarray,
    tau: float,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> RMST1Result:
    """Compute single-arm Kaplan-Meier RMST, RMTL, and exact Greenwood tail variance.

    Parameters
    ----------
    time : 1D array-like of non-negative observed follow-up times.
    status : 1D array-like of event indicators (1 = event, 0 = right-censored).
    tau : Positive truncation time horizon.
    alpha : Two-sided significance level in ``(0, 1)``.
    settings : ``"default"`` (graceful ``d_k / Y_k^2`` boundary atom at ``Y_k == d_k``)
        or ``"R_default"`` (exact ``survRM2::rmst1`` v1.0-4 parity with ``0`` when
        ``Y_k == d_k`` and strict KM support validation).
    """
    validate_settings(settings)
    t_arr, d_arr = _validate_time_status(time, status)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be finite and positive, got {tau}.")

    tau_val = float(tau)
    tau_upper = _km_tau_upper_limit(t_arr, d_arr)
    if tau_val > tau_upper:
        raise ValueError(
            f"Truncation time tau={tau_val} exceeds the largest observed time "
            f"({tau_upper}) where Kaplan-Meier survival remains positive."
        )

    # Distinct observed times up to tau (matching survival::survfit grid in survRM2::rmst1)
    uniq_times = np.unique(t_arr[t_arr <= tau_val])
    m = uniq_times.shape[0]

    n_risk = np.empty(m, dtype=np.float64)
    n_event = np.empty(m, dtype=np.float64)
    surv = np.empty(m, dtype=np.float64)
    hazard_var = np.empty(m, dtype=np.float64)

    cur_s = 1.0
    for k, tk in enumerate(uniq_times):
        yk = float(np.sum(t_arr >= tk))
        dk = float(np.sum((t_arr == tk) & (d_arr == 1)))
        n_risk[k] = yk
        n_event[k] = dk
        if yk > 0.0:
            cur_s *= 1.0 - dk / yk
        surv[k] = cur_s

        if dk == 0.0:
            hazard_var[k] = 0.0
        elif yk == dk:
            hazard_var[k] = 0.0 if settings == "R_default" else (dk / (yk * yk))
        else:
            hazard_var[k] = dk / (yk * (yk - dk))

    # Step-function areas on [0, t_1), [t_1, t_2), ..., [t_m, tau]
    grid_edges = np.concatenate(([0.0], uniq_times, [tau_val]))
    widths = np.diff(grid_edges)  # length m + 1
    heights = np.concatenate(([1.0], surv))  # length m + 1
    areas = widths * heights

    rmst_est = float(np.sum(areas))
    # Tail integrals A_k = int_{t_k}^tau S_hat(u) du for k = 1..m
    if m > 0:
        tail_areas = np.cumsum(areas[1:][::-1])[::-1]
        var_rmst = float(np.sum((tail_areas ** 2) * hazard_var))
    else:
        tail_areas = np.empty(0, dtype=np.float64)
        var_rmst = 0.0

    se_rmst = float(np.sqrt(max(var_rmst, 0.0)))
    z_crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))
    rmst_ci = (rmst_est - z_crit * se_rmst, rmst_est + z_crit * se_rmst)

    rmtl_est = tau_val - rmst_est
    var_rmtl = var_rmst
    se_rmtl = se_rmst
    rmtl_ci = (rmtl_est - z_crit * se_rmtl, rmtl_est + z_crit * se_rmtl)

    km_df = pd.DataFrame(
        {
            "time": uniq_times,
            "n_risk": n_risk.astype(np.int64),
            "n_event": n_event.astype(np.int64),
            "surv": surv,
            "hazard_var": hazard_var,
            "cum_hazard_var": np.cumsum(hazard_var) if m > 0 else np.empty(0, dtype=np.float64),
            "tail_area": tail_areas,
        }
    )

    # Store full-horizon KM step curve and [0, tau] step arrays for zero-arg `.plot()`
    all_uniq = np.unique(t_arr)
    all_surv = np.empty(all_uniq.size, dtype=np.float64)
    cur_s_full = 1.0
    for idx_f, tk_f in enumerate(all_uniq):
        yk_f = float(np.sum(t_arr >= tk_f))
        dk_f = float(np.sum((t_arr == tk_f) & (d_arr == 1)))
        if yk_f > 0.0:
            cur_s_full *= 1.0 - dk_f / yk_f
        all_surv[idx_f] = cur_s_full
    km_full_df = pd.DataFrame({"time": all_uniq, "surv": all_surv})

    step_t_list: list[float] = [0.0]
    step_s_list: list[float] = [1.0]
    for tk_s, sk_s in zip(uniq_times, surv):
        step_t_list.append(float(tk_s))
        step_s_list.append(float(sk_s))
    if m == 0 or float(uniq_times[-1]) < tau_val:
        step_t_list.append(tau_val)
        step_s_list.append(float(surv[-1]) if m > 0 else 1.0)

    return RMST1Result(
        tau=tau_val,
        alpha=alpha,
        settings=settings,
        n=int(t_arr.shape[0]),
        n_events=int(np.sum(d_arr == 1)),
        rmst=rmst_est,
        var_rmst=var_rmst,
        se_rmst=se_rmst,
        rmst_ci=rmst_ci,
        rmtl=rmtl_est,
        var_rmtl=var_rmtl,
        se_rmtl=se_rmtl,
        rmtl_ci=rmtl_ci,
        km_df=km_df,
        km_step_time=np.asarray(step_t_list, dtype=np.float64),
        km_step_surv=np.asarray(step_s_list, dtype=np.float64),
        km_full_df=km_full_df,
    )


def _compute_unadjusted_contrasts(
    arm1: RMST1Result,
    arm0: RMST1Result,
    alpha: float,
) -> tuple[ContrastResult, ContrastResult, ContrastResult, pd.DataFrame]:
    """Compute RMST difference, RMST ratio, and RMTL ratio with exact delta-method CIs."""
    z_crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))

    # 1. RMST difference: mu_1 - mu_0
    diff_est = arm1.rmst - arm0.rmst
    diff_se = float(np.sqrt(max(arm1.var_rmst + arm0.var_rmst, 0.0)))
    diff_ci = (diff_est - z_crit * diff_se, diff_est + z_crit * diff_se)
    diff_z = diff_est / diff_se if diff_se > 0.0 else 0.0
    diff_p = float(2.0 * stats.norm.sf(abs(diff_z)))
    c_diff = ContrastResult(
        label="RMST (arm=1)-(arm=0)",
        est=diff_est,
        se=diff_se,
        ci_lower=diff_ci[0],
        ci_upper=diff_ci[1],
        z=diff_z,
        pval=diff_p,
    )

    # 2. RMST ratio: mu_1 / mu_0 (log-scale delta method)
    if arm1.rmst > 0.0 and arm0.rmst > 0.0:
        log_r = float(np.log(arm1.rmst) - np.log(arm0.rmst))
        se_log_r = float(
            np.sqrt(
                max(
                    arm1.var_rmst / (arm1.rmst ** 2) + arm0.var_rmst / (arm0.rmst ** 2),
                    0.0,
                )
            )
        )
        ratio_est = arm1.rmst / arm0.rmst
        ratio_ci = (
            float(np.exp(log_r - z_crit * se_log_r)),
            float(np.exp(log_r + z_crit * se_log_r)),
        )
        ratio_z = log_r / se_log_r if se_log_r > 0.0 else 0.0
        ratio_p = float(2.0 * stats.norm.sf(abs(ratio_z)))
    else:
        ratio_est = float("nan")
        se_log_r = float("nan")
        ratio_ci = (float("nan"), float("nan"))
        ratio_z = float("nan")
        ratio_p = float("nan")

    c_ratio = ContrastResult(
        label="RMST (arm=1)/(arm=0)",
        est=ratio_est,
        se=se_log_r,
        ci_lower=ratio_ci[0],
        ci_upper=ratio_ci[1],
        z=ratio_z,
        pval=ratio_p,
    )

    # 3. RMTL ratio: (tau - mu_1) / (tau - mu_0) (log-scale delta method)
    if arm1.rmtl > 0.0 and arm0.rmtl > 0.0:
        log_l = float(np.log(arm1.rmtl) - np.log(arm0.rmtl))
        se_log_l = float(
            np.sqrt(
                max(
                    arm1.var_rmtl / (arm1.rmtl ** 2) + arm0.var_rmtl / (arm0.rmtl ** 2),
                    0.0,
                )
            )
        )
        rmtl_ratio_est = arm1.rmtl / arm0.rmtl
        rmtl_ratio_ci = (
            float(np.exp(log_l - z_crit * se_log_l)),
            float(np.exp(log_l + z_crit * se_log_l)),
        )
        rmtl_z = log_l / se_log_l if se_log_l > 0.0 else 0.0
        rmtl_p = float(2.0 * stats.norm.sf(abs(rmtl_z)))
    else:
        rmtl_ratio_est = float("nan")
        se_log_l = float("nan")
        rmtl_ratio_ci = (float("nan"), float("nan"))
        rmtl_z = float("nan")
        rmtl_p = float("nan")

    c_rmtl = ContrastResult(
        label="RMTL (arm=1)/(arm=0)",
        est=rmtl_ratio_est,
        se=se_log_l,
        ci_lower=rmtl_ratio_ci[0],
        ci_upper=rmtl_ratio_ci[1],
        z=rmtl_z,
        pval=rmtl_p,
    )

    unadj_df = pd.DataFrame(
        [c_diff.to_dict(), c_ratio.to_dict(), c_rmtl.to_dict()]
    )
    return c_diff, c_ratio, c_rmtl, unadj_df


def rmst2(
    time: Sequence[float] | np.ndarray,
    status: Sequence[int] | np.ndarray,
    arm: Sequence[int] | np.ndarray,
    tau: float | None = None,
    covariates: np.ndarray | pd.DataFrame | pd.Series | None = None,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> RMST2Result:
    """Two-sample unadjusted and ANCOVA covariate-adjusted RMST & RMTL analysis.

    Parameters
    ----------
    time : 1D array-like of non-negative observed follow-up times.
    status : 1D array-like of event indicators (1 = event, 0 = right-censored).
    arm : 1D array-like of binary treatment indicators (1 = active, 0 = control).
    tau : Positive truncation horizon. When ``None``, defaults to
        ``min(max(time[arm == 0]), max(time[arm == 1]))``.
    covariates : Optional baseline covariate matrix / Series / DataFrame for
        Tian et al. (2014) IPCW ANCOVA across identity, log-RMST, and log-RMTL links.
    alpha : Two-sided significance level in ``(0, 1)``.
    settings : ``"default"`` or ``"R_default"`` (``uno1lab/survRM2`` v1.0-4 parity).
    """
    validate_settings(settings)
    t_arr, d_arr = _validate_time_status(time, status)
    a_arr = _validate_arm(arm, t_arr.shape[0])
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    mask0 = a_arr == 0
    mask1 = a_arr == 1
    t0, d0 = t_arr[mask0], d_arr[mask0]
    t1, d1 = t_arr[mask1], d_arr[mask1]

    default_tau = float(min(np.max(t0), np.max(t1)))
    tau_limit = min(_km_tau_upper_limit(t0, d0), _km_tau_upper_limit(t1, d1))

    if tau is None:
        tau_val = default_tau
        if tau_val <= 0.0:
            raise ValueError("Auto-selected default tau must be strictly positive.")
    else:
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be finite and positive, got {tau}.")
        tau_val = float(tau)
        if tau_val > tau_limit:
            raise ValueError(
                f"The truncation time tau={tau_val} exceeds the maximum admissible "
                f"horizon ({tau_limit}) where Kaplan-Meier survival remains positive "
                "after the last censored observation."
            )

    arm1_res = rmst1(t1, d1, tau=tau_val, alpha=alpha, settings=settings)
    arm0_res = rmst1(t0, d0, tau=tau_val, alpha=alpha, settings=settings)
    c_diff, c_ratio, c_rmtl, unadj_df = _compute_unadjusted_contrasts(
        arm1_res, arm0_res, alpha=alpha
    )

    if covariates is None:
        return RMST2Result(
            tau=tau_val,
            alpha=alpha,
            settings=settings,
            adjusted=False,
            arm1=arm1_res,
            arm0=arm0_res,
            rmst_diff_10=c_diff,
            rmst_ratio_10=c_ratio,
            rmtl_ratio_10=c_rmtl,
            unadjusted_df=unadj_df,
        )

    adj_models: dict[str, RMST2AdjustedModelResult] = {}
    for lnk in VALID_LINKS:
        adj_models[lnk] = rmst2reg(
            time=t_arr,
            status=d_arr,
            arm=a_arr,
            covariates=covariates,
            tau=tau_val,
            link=lnk,  # type: ignore[arg-type]
            alpha=alpha,
            settings=settings,
        )

    def _to_contrast(mod: RMST2AdjustedModelResult) -> ContrastResult:
        z_val = float(mod.coef_df.loc[1, "z"])
        return ContrastResult(
            label=mod.contrast_label,
            est=mod.contrast_est,
            se=mod.contrast_se,
            ci_lower=mod.contrast_ci[0],
            ci_upper=mod.contrast_ci[1],
            z=z_val,
            pval=mod.contrast_pval,
        )

    adj_diff = _to_contrast(adj_models["identity"])
    adj_ratio = _to_contrast(adj_models["log_rmst"])
    adj_rmtl = _to_contrast(adj_models["log_rmtl"])
    adj_df = pd.DataFrame(
        [adj_diff.to_dict(), adj_ratio.to_dict(), adj_rmtl.to_dict()]
    )

    return RMST2Result(
        tau=tau_val,
        alpha=alpha,
        settings=settings,
        adjusted=True,
        arm1=arm1_res,
        arm0=arm0_res,
        rmst_diff_10=c_diff,
        rmst_ratio_10=c_ratio,
        rmtl_ratio_10=c_rmtl,
        unadjusted_df=unadj_df,
        adjusted_models=adj_models,
        adjusted_df=adj_df,
        adj_rmst_diff_10=adj_diff,
        adj_rmst_ratio_10=adj_ratio,
        adj_rmtl_ratio_10=adj_rmtl,
    )


def rmst2_curve(
    time: Sequence[float] | np.ndarray,
    status: Sequence[int] | np.ndarray,
    arm: Sequence[int] | np.ndarray,
    tau_grid: Sequence[float] | np.ndarray | None = None,
    covariates: np.ndarray | pd.DataFrame | pd.Series | None = None,
    alpha: float = 0.05,
    n_grid: int = 15,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Evaluate RMST & RMTL contrasts across a grid of truncation horizons ``tau_grid``.

    Parameters
    ----------
    time, status, arm : Clinical trial survival data arrays.
    tau_grid : Optional 1D sequence of positive truncation horizons. If ``None``,
        constructs ``n_grid`` evenly spaced horizons between the earliest post-event
        support across arms and ``min(max(time[arm==0]), max(time[arm==1]))``.
    covariates : Optional baseline covariate matrix/DataFrame for ANCOVA adjustment.
    alpha : Two-sided significance level in ``(0, 1)``.
    n_grid : Number of automatic grid points when ``tau_grid`` is ``None``.
    settings : ``"default"`` or ``"R_default"``.
    """
    validate_settings(settings)
    t_arr, d_arr = _validate_time_status(time, status)
    a_arr = _validate_arm(arm, t_arr.shape[0])

    if tau_grid is None:
        if n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {n_grid}.")
        t0_ev = t_arr[(a_arr == 0) & (d_arr == 1)]
        t1_ev = t_arr[(a_arr == 1) & (d_arr == 1)]
        tau_max = float(min(np.max(t_arr[a_arr == 0]), np.max(t_arr[a_arr == 1])))
        if t0_ev.size > 0 and t1_ev.size > 0:
            tau_min = max(float(np.min(t0_ev)), float(np.min(t1_ev))) * 1.05
            if tau_min >= tau_max:
                tau_min = 0.2 * tau_max
        else:
            tau_min = 0.2 * tau_max
        taus = np.linspace(max(tau_min, 1e-4 * tau_max), tau_max, n_grid)
    else:
        taus = np.asarray(tau_grid, dtype=np.float64)
        if taus.ndim != 1 or taus.size == 0:
            raise ValueError("tau_grid must be a non-empty 1D sequence.")

    rows: list[dict[str, float]] = []
    for tau_val in taus:
        res = rmst2(
            time=t_arr,
            status=d_arr,
            arm=a_arr,
            tau=float(tau_val),
            covariates=covariates,
            alpha=alpha,
            settings=settings,
        )
        row: dict[str, float] = {
            "tau": res.tau,
            "rmst_arm1": res.arm1.rmst,
            "se_arm1": res.arm1.se_rmst,
            "rmst_arm0": res.arm0.rmst,
            "se_arm0": res.arm0.se_rmst,
            "rmst_diff_10": res.rmst_diff_10.est,
            "rmst_diff_se": res.rmst_diff_10.se,
            "rmst_diff_ci_lower": res.rmst_diff_10.ci_lower,
            "rmst_diff_ci_upper": res.rmst_diff_10.ci_upper,
            "rmst_diff_pval": res.rmst_diff_10.pval,
            "rmst_ratio_10": res.rmst_ratio_10.est,
            "rmst_ratio_ci_lower": res.rmst_ratio_10.ci_lower,
            "rmst_ratio_ci_upper": res.rmst_ratio_10.ci_upper,
            "rmst_ratio_pval": res.rmst_ratio_10.pval,
            "rmtl_ratio_10": res.rmtl_ratio_10.est,
            "rmtl_ratio_ci_lower": res.rmtl_ratio_10.ci_lower,
            "rmtl_ratio_ci_upper": res.rmtl_ratio_10.ci_upper,
            "rmtl_ratio_pval": res.rmtl_ratio_10.pval,
        }
        if res.adjusted and res.adj_rmst_diff_10 is not None and res.adj_rmst_ratio_10 is not None and res.adj_rmtl_ratio_10 is not None:
            row["adj_rmst_diff_10"] = res.adj_rmst_diff_10.est
            row["adj_rmst_diff_ci_lower"] = res.adj_rmst_diff_10.ci_lower
            row["adj_rmst_diff_ci_upper"] = res.adj_rmst_diff_10.ci_upper
            row["adj_rmst_diff_pval"] = res.adj_rmst_diff_10.pval
            row["adj_rmst_ratio_10"] = res.adj_rmst_ratio_10.est
            row["adj_rmst_ratio_pval"] = res.adj_rmst_ratio_10.pval
            row["adj_rmtl_ratio_10"] = res.adj_rmtl_ratio_10.est
            row["adj_rmtl_ratio_pval"] = res.adj_rmtl_ratio_10.pval
        rows.append(row)

    return pd.DataFrame(rows)


class SurvRM2:
    """Object-oriented estimator for Restricted Mean Survival Time (RMST) analysis.

    Parameters
    ----------
    tau : Optional truncation horizon. Auto-selected if ``None``.
    alpha : Two-sided significance level in ``(0, 1)``.
    settings : ``"default"`` or ``"R_default"`` (``uno1lab/survRM2`` v1.0-4 parity).
    """

    def __init__(
        self,
        tau: float | None = None,
        alpha: float = 0.05,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if tau is not None and (not np.isfinite(tau) or tau <= 0.0):
            raise ValueError(f"tau must be finite and positive when specified, got {tau}.")
        self.tau = tau
        self.alpha = alpha
        self.settings: SettingsMode = settings
        self.result_: RMST2Result | None = None
        self._fit_data: tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None = None

    def fit(
        self,
        time: Sequence[float] | np.ndarray,
        status: Sequence[int] | np.ndarray,
        arm: Sequence[int] | np.ndarray,
        covariates: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> SurvRM2:
        """Fit unadjusted and/or covariate-adjusted two-sample RMST & RMTL models."""
        self.result_ = rmst2(
            time=time,
            status=status,
            arm=arm,
            tau=self.tau,
            covariates=covariates,
            alpha=self.alpha,
            settings=self.settings,
        )
        self._fit_data = (
            np.asarray(time, dtype=np.float64),
            np.asarray(status, dtype=np.int64),
            np.asarray(arm, dtype=np.int64),
            covariates,
        )
        return self

    def fit_single(
        self,
        time: Sequence[float] | np.ndarray,
        status: Sequence[int] | np.ndarray,
        tau: float | None = None,
    ) -> RMST1Result:
        """Compute single-arm RMST & RMTL using estimator configuration."""
        use_tau = tau if tau is not None else self.tau
        if use_tau is None:
            raise ValueError("tau must be provided for single-arm RMST estimation.")
        return rmst1(
            time=time,
            status=status,
            tau=use_tau,
            alpha=self.alpha,
            settings=self.settings,
        )

    def curve(
        self,
        tau_grid: Sequence[float] | np.ndarray | None = None,
        n_grid: int = 15,
    ) -> pd.DataFrame:
        """Compute dynamic RMST contrast curve over ``tau_grid`` on fitted data."""
        if self._fit_data is None:
            raise RuntimeError("Call .fit(time, status, arm, ...) before .curve().")
        t_arr, d_arr, a_arr, covs = self._fit_data
        return rmst2_curve(
            time=t_arr,
            status=d_arr,
            arm=a_arr,
            tau_grid=tau_grid,
            covariates=covs,
            alpha=self.alpha,
            n_grid=n_grid,
            settings=self.settings,
        )

    def summary(self) -> str:
        """Return summary report of fitted model."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet; call .fit() first.")
        return self.result_.summary()

    def to_dataframe(self) -> pd.DataFrame:
        """Return primary contrast DataFrame of fitted model."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet; call .fit() first.")
        return self.result_.to_dataframe()

    def plot(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (7.5, 4.8),
        *,
        arm_labels: tuple[str, str] = ("Control (Arm 0)", "Active (Arm 1)"),
        title: str | None = None,
    ) -> Figure:
        """Plot two-sample RMST Kaplan-Meier curves of fitted model."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet; call .fit() first.")
        return self.result_.plot(
            ax=ax,
            figsize=figsize,
            arm_labels=arm_labels,
            title=title,
        )
