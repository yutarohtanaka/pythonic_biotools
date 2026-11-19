"""
Core algorithms for MR-PRESSO and Multivariable MR-PRESSO.

Reference:
  Verbanck, M., Chen, C.-Y., Neale, B., & Do, R. (2018).
  "Detection of widespread horizontal pleiotropy in causal relationships inferred
  from Mendelian randomization between complex traits and diseases."
  Nature Genetics, 50(5), 693-698.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class IVWFit:
    """Inverse-variance weighted regression fit (no intercept, matching R `lm(y ~ -1 + X, weights=w)`)."""

    beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    n_instruments: int
    df_resid: int
    sigma2: float


@dataclass(frozen=True)
class MRPressoResult:
    """Result container for univariable `mr_presso`."""

    n_instruments: int
    snp_ids: list[str]
    exposure_name: str
    beta_exposure: np.ndarray
    beta_outcome: np.ndarray
    sd_exposure: np.ndarray
    sd_outcome: np.ndarray
    beta_raw: float
    se_raw: float
    t_stat_raw: float
    p_value_raw: float
    beta_corrected: float | None
    se_corrected: float | None
    t_stat_corrected: float | None
    p_value_corrected: float | None
    rss_obs: float
    rss_per_snp: np.ndarray
    expected_rss_per_snp: np.ndarray
    loo_betas: np.ndarray
    p_global: float
    p_global_empirical: float
    p_global_formatted: str
    outlier_indices: list[int]
    outlier_snps: list[str]
    outlier_pvalues: np.ndarray | None
    outlier_pvalues_bonferroni: np.ndarray | None
    outlier_pvalues_chisq: np.ndarray | None
    distortion_coeff: float | None
    p_distortion: float | None
    nb_distribution: int
    signif_threshold: float
    weighted_rss: bool
    settings: str = "default"

    def to_dataframe(self) -> pd.DataFrame:
        """Return two-row summary DataFrame (`Raw` and `Outlier-corrected`) matching R `Main MR results`."""
        rows = [
            {
                "exposure": self.exposure_name,
                "mr_analysis": "Raw",
                "causal_estimate": self.beta_raw,
                "sd": self.se_raw,
                "t_stat": self.t_stat_raw,
                "p_value": self.p_value_raw,
                "n_instruments": self.n_instruments,
                "rss_obs": self.rss_obs,
                "p_global": self.p_global,
                "n_outliers": len(self.outlier_indices),
                "distortion_coeff": self.distortion_coeff if self.distortion_coeff is not None else np.nan,
                "p_distortion": self.p_distortion if self.p_distortion is not None else np.nan,
            },
            {
                "exposure": self.exposure_name,
                "mr_analysis": "Outlier-corrected",
                "causal_estimate": self.beta_corrected if self.beta_corrected is not None else np.nan,
                "sd": self.se_corrected if self.se_corrected is not None else np.nan,
                "t_stat": self.t_stat_corrected if self.t_stat_corrected is not None else np.nan,
                "p_value": self.p_value_corrected if self.p_value_corrected is not None else np.nan,
                "n_instruments": (
                    self.n_instruments - len(self.outlier_indices)
                    if self.beta_corrected is not None
                    else np.nan
                ),
                "rss_obs": self.rss_obs,
                "p_global": self.p_global,
                "n_outliers": len(self.outlier_indices),
                "distortion_coeff": self.distortion_coeff if self.distortion_coeff is not None else np.nan,
                "p_distortion": self.p_distortion if self.p_distortion is not None else np.nan,
            },
        ]
        return pd.DataFrame(rows)

    def outlier_table(self) -> pd.DataFrame:
        """Return per-instrument leave-one-out RSS and outlier test DataFrame."""
        outlier_set = set(self.outlier_indices)
        return pd.DataFrame(
            {
                "snp": self.snp_ids,
                "beta_exposure": self.beta_exposure,
                "sd_exposure": self.sd_exposure,
                "beta_outcome": self.beta_outcome,
                "sd_outcome": self.sd_outcome,
                "loo_beta": self.loo_betas,
                "rss_obs": self.rss_per_snp,
                "rss_expected": self.expected_rss_per_snp,
                "p_value": (
                    self.outlier_pvalues
                    if self.outlier_pvalues is not None
                    else np.full(self.n_instruments, np.nan)
                ),
                "p_bonferroni": (
                    self.outlier_pvalues_bonferroni
                    if self.outlier_pvalues_bonferroni is not None
                    else np.full(self.n_instruments, np.nan)
                ),
                "p_chisq": (
                    self.outlier_pvalues_chisq
                    if self.outlier_pvalues_chisq is not None
                    else np.full(self.n_instruments, np.nan)
                ),
                "is_outlier": [i in outlier_set for i in range(self.n_instruments)],
            }
        )

    def summary(self) -> str:
        """Format human-readable MR-PRESSO report."""
        lines = [
            "MR-PRESSO Summary",
            "=" * 60,
            f"Instruments: {self.n_instruments} | Simulations: {self.nb_distribution} | Alpha: {self.signif_threshold}",
            f"Global Test: RSS_obs = {self.rss_obs:.6g}, p_global = {self.p_global:.6g} ({self.p_global_formatted})",
            f"Raw IVW:     beta = {self.beta_raw:.6g}, SE = {self.se_raw:.6g}, t = {self.t_stat_raw:.4g}, p = {self.p_value_raw:.4g}",
        ]
        if self.beta_corrected is not None:
            lines.append(
                f"Corrected:   beta = {self.beta_corrected:.6g}, SE = {self.se_corrected:.6g}, "
                f"t = {self.t_stat_corrected:.4g}, p = {self.p_value_corrected:.4g} "
                f"(outliers: {', '.join(self.outlier_snps)})"
            )
        else:
            lines.append("Corrected:   None (no significant outliers removed)")
        if self.p_distortion is not None:
            lines.append(
                f"Distortion:  D = {self.distortion_coeff:.4g}%, p_distortion = {self.p_distortion:.6g}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MVMRPressoResult:
    """Result container for Multivariable MR-PRESSO (`mvmr_presso`, P >= 2 exposures)."""

    n_instruments: int
    n_exposures: int
    snp_ids: list[str]
    exposure_names: list[str]
    beta_exposure: np.ndarray
    beta_outcome: np.ndarray
    sd_exposure: np.ndarray
    sd_outcome: np.ndarray
    beta_raw: np.ndarray
    se_raw: np.ndarray
    t_stat_raw: np.ndarray
    p_value_raw: np.ndarray
    beta_corrected: np.ndarray | None
    se_corrected: np.ndarray | None
    t_stat_corrected: np.ndarray | None
    p_value_corrected: np.ndarray | None
    rss_obs: float
    rss_per_snp: np.ndarray
    expected_rss_per_snp: np.ndarray
    loo_betas: np.ndarray
    p_global: float
    p_global_empirical: float
    p_global_formatted: str
    outlier_indices: list[int]
    outlier_snps: list[str]
    outlier_pvalues: np.ndarray | None
    outlier_pvalues_bonferroni: np.ndarray | None
    outlier_pvalues_chisq: np.ndarray | None
    distortion_coeff: np.ndarray | None
    p_distortion: np.ndarray | None
    p_distortion_joint: float | None
    nb_distribution: int
    signif_threshold: float
    weighted_rss: bool
    settings: str = "default"

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-exposure summary DataFrame (`Raw` and `Outlier-corrected`)."""
        rows = []
        for p_idx, exp_name in enumerate(self.exposure_names):
            d_coef = float(self.distortion_coeff[p_idx]) if self.distortion_coeff is not None else np.nan
            p_dist = float(self.p_distortion[p_idx]) if self.p_distortion is not None else np.nan
            rows.append(
                {
                    "exposure": exp_name,
                    "mr_analysis": "Raw",
                    "causal_estimate": float(self.beta_raw[p_idx]),
                    "sd": float(self.se_raw[p_idx]),
                    "t_stat": float(self.t_stat_raw[p_idx]),
                    "p_value": float(self.p_value_raw[p_idx]),
                    "n_instruments": self.n_instruments,
                    "rss_obs": self.rss_obs,
                    "p_global": self.p_global,
                    "n_outliers": len(self.outlier_indices),
                    "distortion_coeff": d_coef,
                    "p_distortion": p_dist,
                }
            )
            rows.append(
                {
                    "exposure": exp_name,
                    "mr_analysis": "Outlier-corrected",
                    "causal_estimate": (
                        float(self.beta_corrected[p_idx]) if self.beta_corrected is not None else np.nan
                    ),
                    "sd": float(self.se_corrected[p_idx]) if self.se_corrected is not None else np.nan,
                    "t_stat": (
                        float(self.t_stat_corrected[p_idx]) if self.t_stat_corrected is not None else np.nan
                    ),
                    "p_value": (
                        float(self.p_value_corrected[p_idx]) if self.p_value_corrected is not None else np.nan
                    ),
                    "n_instruments": (
                        self.n_instruments - len(self.outlier_indices)
                        if self.beta_corrected is not None
                        else np.nan
                    ),
                    "rss_obs": self.rss_obs,
                    "p_global": self.p_global,
                    "n_outliers": len(self.outlier_indices),
                    "distortion_coeff": d_coef,
                    "p_distortion": p_dist,
                }
            )
        return pd.DataFrame(rows)

    def outlier_table(self) -> pd.DataFrame:
        """Return per-instrument leave-one-out RSS and outlier p-values."""
        outlier_set = set(self.outlier_indices)
        cols: dict[str, object] = {"snp": self.snp_ids}
        for p_idx, exp_name in enumerate(self.exposure_names):
            cols[f"beta_{exp_name}"] = self.beta_exposure[:, p_idx]
            cols[f"loo_beta_{exp_name}"] = self.loo_betas[:, p_idx]
        cols["beta_outcome"] = self.beta_outcome
        cols["sd_outcome"] = self.sd_outcome
        cols["rss_obs"] = self.rss_per_snp
        cols["rss_expected"] = self.expected_rss_per_snp
        cols["p_value"] = (
            self.outlier_pvalues
            if self.outlier_pvalues is not None
            else np.full(self.n_instruments, np.nan)
        )
        cols["p_bonferroni"] = (
            self.outlier_pvalues_bonferroni
            if self.outlier_pvalues_bonferroni is not None
            else np.full(self.n_instruments, np.nan)
        )
        cols["p_chisq"] = (
            self.outlier_pvalues_chisq
            if self.outlier_pvalues_chisq is not None
            else np.full(self.n_instruments, np.nan)
        )
        cols["is_outlier"] = [i in outlier_set for i in range(self.n_instruments)]
        return pd.DataFrame(cols)

    def summary(self) -> str:
        """Format human-readable Multivariable MR-PRESSO report."""
        lines = [
            "Multivariable MR-PRESSO Summary",
            "=" * 60,
            f"Instruments: {self.n_instruments} | Exposures: {self.n_exposures} | Simulations: {self.nb_distribution}",
            f"Global Test: RSS_obs = {self.rss_obs:.6g}, p_global = {self.p_global:.6g} ({self.p_global_formatted})",
        ]
        for p_idx, exp_name in enumerate(self.exposure_names):
            lines.append(
                f"  [{exp_name}] Raw: beta = {self.beta_raw[p_idx]:.6g} "
                f"(SE = {self.se_raw[p_idx]:.6g}, p = {self.p_value_raw[p_idx]:.4g})"
            )
            if self.beta_corrected is not None:
                lines.append(
                    f"  [{exp_name}] Corrected: beta = {self.beta_corrected[p_idx]:.6g} "
                    f"(SE = {self.se_corrected[p_idx]:.6g}, p = {self.p_value_corrected[p_idx]:.4g})"
                )
        if self.outlier_snps:
            lines.append(f"Outliers ({len(self.outlier_snps)}): {', '.join(self.outlier_snps)}")
        if self.p_distortion_joint is not None:
            lines.append(f"Joint Distortion p-value: {self.p_distortion_joint:.6g}")
        return "\n".join(lines)


def _fit_ivw(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> IVWFit:
    """Fit no-intercept weighted least squares matching R `summary(lm(y ~ -1 + X, weights = w))`."""
    n, p = X.shape
    if n <= p:
        raise ValueError(f"Need more instruments ({n}) than exposures ({p}) to fit IVW.")
    xtw = X.T * w[None, :]
    xtwx = xtw @ X
    xtwy = xtw @ y
    try:
        cov_unscaled = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Design matrix X^T W X is singular.") from exc

    beta = cov_unscaled @ xtwy
    resid = y - X @ beta
    df_resid = n - p
    sigma2 = float(np.sum(w * (resid**2)) / df_resid)
    diag_cov = np.maximum(np.diag(cov_unscaled), 0.0)
    se = np.sqrt(sigma2 * diag_cov)

    t_stat = np.empty(p, dtype=np.float64)
    p_val = np.empty(p, dtype=np.float64)
    for idx in range(p):
        if se[idx] > 0.0:
            t_stat[idx] = beta[idx] / se[idx]
            p_val[idx] = float(2.0 * stats.t.sf(abs(t_stat[idx]), df=df_resid))
        elif abs(beta[idx]) > 0.0:
            t_stat[idx] = float(np.sign(beta[idx]) * np.inf)
            p_val[idx] = 0.0
        else:
            t_stat[idx] = 0.0
            p_val[idx] = 1.0

    return IVWFit(
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_val,
        n_instruments=n,
        df_resid=df_resid,
        sigma2=sigma2,
    )


def _validate_settings(settings: str) -> Literal["default", "R_default"]:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")
    return settings  # type: ignore[return-value]


def leave_one_out_ivw(
    beta_outcome: Sequence[float] | np.ndarray,
    beta_exposure: Sequence[float] | np.ndarray,
    sd_outcome: Sequence[float] | np.ndarray,
    *,
    weighted_rss: bool | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact leave-one-out IVW slopes and squared residuals via Sherman-Morrison updates.

    For univariable MR (P = 1), computes exact O(J) leave-one-out sums.
    For multivariable MR (P >= 2), uses Sherman-Morrison rank-1 downdates:
      (X^T W X - w_j x_j x_j^T)^{-1} = A^{-1} + w_j A^{-1} x_j x_j^T A^{-1} / (1 - h_j)
      beta^{(-j)} = beta - w_j A^{-1} x_j e_j / (1 - h_j)
      r_j = y_j - x_j^T beta^{(-j)} = e_j / (1 - h_j).

    Returns
    -------
    loo_betas : np.ndarray
        Shape `(J,)` if 1D exposure was passed, else `(J, P)`.
    rss_per_snp : np.ndarray
        Shape `(J,)` per-instrument squared LOO residuals.
    rss_obs : float
        Total observed LOO residual sum of squares.
    """
    mode_settings = _validate_settings(settings)
    w_rss = (mode_settings == "R_default") if weighted_rss is None else bool(weighted_rss)
    y = np.asarray(beta_outcome, dtype=np.float64).ravel()
    sy = np.asarray(sd_outcome, dtype=np.float64).ravel()
    raw_x = np.asarray(beta_exposure, dtype=np.float64)
    is_1d = raw_x.ndim == 1
    X = raw_x[:, None] if is_1d else raw_x
    if X.ndim != 2:
        raise ValueError("beta_exposure must be 1D or 2D.")
    n, p = X.shape
    if y.size != n or sy.size != n:
        raise ValueError("beta_outcome, beta_exposure, and sd_outcome must have matching instrument count.")
    if n < p + 2:
        raise ValueError(f"At least {p + 2} instruments are required for leave-one-out IVW (got {n}).")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(sy)) and np.all(np.isfinite(X))):
        raise ValueError("Summary statistics must be finite (no NaN or Inf).")
    if np.any(sy <= 0.0):
        raise ValueError("sd_outcome must be strictly positive.")

    w = 1.0 / (sy**2)
    if p == 1:
        x = X[:, 0]
        s_xx = float(np.sum(w * (x**2)))
        s_xy = float(np.sum(w * x * y))
        denom = s_xx - w * (x**2)
        if np.any(denom <= 1e-15 * max(s_xx, 1.0)):
            raise ValueError("Leave-one-out exposure sum of squares is singular for at least one instrument.")
        loo_b = (s_xy - w * x * y) / denom
        resid_loo = y - loo_b * x
        rss_j = (w * (resid_loo**2)) if w_rss else (resid_loo**2)
        return (loo_b if is_1d else loo_b[:, None]), rss_j, float(np.sum(rss_j))

    xtw = X.T * w[None, :]
    A = xtw @ X
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Exposure matrix X^T W X is singular.") from exc

    beta_full = A_inv @ (xtw @ y)
    A_inv_Xt = A_inv @ X.T  # (P, J)
    h = w * np.sum(X.T * A_inv_Xt, axis=0)  # (J,)
    one_minus_h = 1.0 - h
    if np.any(one_minus_h <= 1e-12):
        raise ValueError("Leave-one-out design matrix X_{-j}^T W_{-j} X_{-j} is singular (leverage >= 1).")

    e_full = y - X @ beta_full  # (J,)
    # Sherman-Morrison exact LOO slopes: beta^{(-j)} = beta - w_j A^{-1} x_j e_j / (1 - h_j)
    loo_betas = beta_full[None, :] - (A_inv_Xt * (w * e_full / one_minus_h)[None, :]).T  # (J, P)
    resid_loo = e_full / one_minus_h
    rss_j = (w * (resid_loo**2)) if w_rss else (resid_loo**2)
    return loo_betas, rss_j, float(np.sum(rss_j))


def _canonicalize_signs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orient each instrument row so its first non-zero exposure effect is positive.

    Because no-intercept IVW and Gaussian perturbations N(mu, s^2) are reflection-symmetric,
    orienting instrument signs leaves the statistical model unchanged while guaranteeing
    exact bit-for-bit Monte Carlo invariance to arbitrary allele flips.
    """
    signs = np.ones(X.shape[0], dtype=np.float64)
    for j in range(X.shape[0]):
        nz = np.flatnonzero(X[j] != 0.0)
        if nz.size > 0 and X[j, nz[0]] < 0.0:
            signs[j] = -1.0
    return X * signs[:, None], y * signs


def _extract_inputs(
    beta_outcome: Sequence[float] | np.ndarray | str,
    beta_exposure: Sequence[float] | np.ndarray | str | Sequence[str],
    sd_outcome: Sequence[float] | np.ndarray | str,
    sd_exposure: Sequence[float] | np.ndarray | str | Sequence[str],
    data: pd.DataFrame | None,
    snp_ids: Sequence[str] | None,
    exposure_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], bool]:
    if data is not None:
        if not isinstance(beta_outcome, str) or not isinstance(sd_outcome, str):
            raise ValueError("When `data` is provided, beta_outcome and sd_outcome must be column names.")
        y = np.asarray(data[beta_outcome], dtype=np.float64).ravel()
        sy = np.asarray(data[sd_outcome], dtype=np.float64).ravel()
        if isinstance(beta_exposure, str):
            if not isinstance(sd_exposure, str):
                raise ValueError("sd_exposure must be a column name matching beta_exposure.")
            X = np.asarray(data[beta_exposure], dtype=np.float64).reshape(-1, 1)
            SX = np.asarray(data[sd_exposure], dtype=np.float64).reshape(-1, 1)
            exp_names = [beta_exposure] if exposure_names is None else [str(x) for x in exposure_names]
            was_1d = True
        else:
            exp_cols = list(beta_exposure)
            sd_cols = [sd_exposure] if isinstance(sd_exposure, str) else list(sd_exposure)
            if len(exp_cols) != len(sd_cols):
                raise ValueError("beta_exposure and sd_exposure column lists must have equal length.")
            X = np.asarray(data[exp_cols], dtype=np.float64)
            SX = np.asarray(data[sd_cols], dtype=np.float64)
            exp_names = [str(c) for c in exp_cols] if exposure_names is None else [str(x) for x in exposure_names]
            was_1d = len(exp_cols) == 1
        ids = (
            [str(idx) for idx in data.index]
            if snp_ids is None
            else [str(s) for s in snp_ids]
        )
    else:
        y = np.asarray(beta_outcome, dtype=np.float64).ravel()
        sy = np.asarray(sd_outcome, dtype=np.float64).ravel()
        raw_x = np.asarray(beta_exposure, dtype=np.float64)
        raw_sx = np.asarray(sd_exposure, dtype=np.float64)
        was_1d = raw_x.ndim == 1
        X = raw_x[:, None] if raw_x.ndim == 1 else raw_x
        SX = raw_sx[:, None] if raw_sx.ndim == 1 else raw_sx
        if X.ndim != 2 or SX.ndim != 2:
            raise ValueError("beta_exposure and sd_exposure must be 1D or 2D arrays.")
        p = X.shape[1]
        exp_names = (
            [f"exposure_{k + 1}" for k in range(p)]
            if exposure_names is None
            else [str(x) for x in exposure_names]
        )
        ids = [f"snp_{j + 1}" for j in range(y.size)] if snp_ids is None else [str(s) for s in snp_ids]

    n, p = X.shape
    if p == 0:
        raise ValueError("At least one exposure is required.")
    if y.size != n or sy.size != n or SX.shape != (n, p):
        raise ValueError("Mismatched shapes among beta_outcome, beta_exposure, sd_outcome, sd_exposure.")
    if len(ids) != n:
        raise ValueError("snp_ids length must match number of instruments.")
    if len(exp_names) != p:
        raise ValueError("exposure_names length must match number of exposures.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(sy)) and np.all(np.isfinite(X)) and np.all(np.isfinite(SX))):
        raise ValueError("All summary statistics and standard errors must be finite (no NaN or Inf).")
    if np.any(sy <= 0.0) or np.any(SX <= 0.0):
        raise ValueError("Standard errors sd_outcome and sd_exposure must be strictly positive.")
    if n < p + 2:
        raise ValueError(f"MR-PRESSO requires at least P + 2 = {p + 2} instruments (got {n}).")

    return y, X, sy, SX, ids, exp_names, was_1d


def _simulate_rss_univariable(
    x_obs: np.ndarray,
    sx: np.ndarray,
    pred_y_loo: np.ndarray,
    sy: np.ndarray,
    w: np.ndarray,
    r_obs_sq: np.ndarray,
    nb_distribution: int,
    weighted_rss: bool,
    rng: np.random.Generator,
    *,
    loo_betas_obs: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized O(K J) Monte Carlo simulation for univariable MR-PRESSO without J-loops."""
    n = x_obs.size
    rss_exp = np.empty(nb_distribution, dtype=np.float64)
    exceed_counts = np.zeros(n, dtype=np.int64)
    sum_r_sq = np.zeros(n, dtype=np.float64)

    batch_size = min(nb_distribution, max(250, 500_000 // max(n, 1)))
    offset = 0
    while offset < nb_distribution:
        b = min(batch_size, nb_distribution - offset)
        x_sim = x_obs[None, :] + sx[None, :] * rng.standard_normal((b, n))
        y_sim = pred_y_loo[None, :] + sy[None, :] * rng.standard_normal((b, n))

        wx2 = (x_sim**2) * w[None, :]
        wxy = (x_sim * y_sim) * w[None, :]
        s_xx = np.sum(wx2, axis=1, keepdims=True)
        s_xy = np.sum(wxy, axis=1, keepdims=True)

        denom = np.maximum(s_xx - wx2, 1e-300)
        beta_loo_sim = (s_xy - wxy) / denom
        resid_sim = y_sim - beta_loo_sim * x_sim
        r_sq = (resid_sim**2) * w[None, :] if weighted_rss else (resid_sim**2)
        rss_exp[offset : offset + b] = np.sum(r_sq, axis=1)

        if settings == "R_default" and loo_betas_obs is not None:
            resid_out = y_sim - x_sim * loo_betas_obs[None, :]
            r_sq_out = (resid_out**2) * w[None, :] if weighted_rss else (resid_out**2)
        else:
            r_sq_out = r_sq

        exceed_counts += np.sum(r_sq_out >= r_obs_sq[None, :], axis=0)
        sum_r_sq += np.sum(r_sq_out, axis=0)
        offset += b

    return rss_exp, exceed_counts / float(nb_distribution), sum_r_sq / float(nb_distribution)


def _simulate_rss_multivariable(
    X_obs: np.ndarray,
    SX: np.ndarray,
    pred_y_loo: np.ndarray,
    sy: np.ndarray,
    w: np.ndarray,
    r_obs_sq: np.ndarray,
    nb_distribution: int,
    weighted_rss: bool,
    rng: np.random.Generator,
    *,
    loo_betas_obs: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized batched Sherman-Morrison O(K J P^2) Monte Carlo simulation for MVMR-PRESSO."""
    n, p = X_obs.shape
    rss_exp = np.empty(nb_distribution, dtype=np.float64)
    exceed_counts = np.zeros(n, dtype=np.int64)
    sum_r_sq = np.zeros(n, dtype=np.float64)

    batch_size = min(nb_distribution, max(100, 200_000 // max(n * p, 1)))
    offset = 0
    while offset < nb_distribution:
        b = min(batch_size, nb_distribution - offset)
        X_sim = X_obs[None, :, :] + SX[None, :, :] * rng.standard_normal((b, n, p))
        y_sim = pred_y_loo[None, :] + sy[None, :] * rng.standard_normal((b, n))

        # A_sim = X_sim^T W X_sim -> (b, p, p); rhs = X_sim^T W y_sim -> (b, p)
        X_w = X_sim * w[None, :, None]
        A_sim = np.einsum("bjp,bjq->bpq", X_w, X_sim)
        rhs_sim = np.einsum("bjp,bj->bp", X_w, y_sim)

        # Tiny ridge guard only if a random draw is nearly singular
        reg = 1e-15 * np.eye(p, dtype=np.float64)[None, :, :]
        A_inv_Xt = np.linalg.solve(A_sim + reg, np.swapaxes(X_sim, 1, 2))  # (b, p, n)
        beta_sim = np.linalg.solve(A_sim + reg, rhs_sim[..., None])[..., 0]  # (b, p)

        h_sim = w[None, :] * np.einsum("bjp,bpj->bj", X_sim, A_inv_Xt)  # (b, n)
        e_sim = y_sim - np.einsum("bjp,bp->bj", X_sim, beta_sim)  # (b, n)
        one_minus_h = np.maximum(1.0 - h_sim, 1e-12)
        resid_sim = e_sim / one_minus_h
        r_sq = (resid_sim**2) * w[None, :] if weighted_rss else (resid_sim**2)
        rss_exp[offset : offset + b] = np.sum(r_sq, axis=1)

        if settings == "R_default" and loo_betas_obs is not None:
            resid_out = y_sim - np.sum(X_sim * loo_betas_obs[None, :, :], axis=2)
            r_sq_out = (resid_out**2) * w[None, :] if weighted_rss else (resid_out**2)
        else:
            r_sq_out = r_sq

        exceed_counts += np.sum(r_sq_out >= r_obs_sq[None, :], axis=0)
        sum_r_sq += np.sum(r_sq_out, axis=0)
        offset += b

    return rss_exp, exceed_counts / float(nb_distribution), sum_r_sq / float(nb_distribution)


def _bootstrap_distortion(
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    beta_raw: np.ndarray,
    beta_corr: np.ndarray,
    nb_distribution: int,
    rng: np.random.Generator,
    *,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bootstrap non-outlier instruments to evaluate empirical two-sided distortion p-values."""
    n_valid, p = X_valid.shape
    if p == 1:
        idx = rng.integers(0, n_valid, size=(nb_distribution, n_valid))
        xb = X_valid[idx, 0]
        yb = y_valid[idx]
        wb = w_valid[idx]
        denom = np.sum(wb * (xb**2), axis=1)
        valid_draws = denom > 1e-300
        beta_boot = np.where(valid_draws, np.sum(wb * xb * yb, axis=1) / np.maximum(denom, 1e-300), beta_corr[0])
        beta_boot_mat = beta_boot[:, None]
    else:
        idx = rng.integers(0, n_valid, size=(nb_distribution, n_valid))
        Xb = X_valid[idx]  # (K, n_valid, p)
        yb = y_valid[idx]  # (K, n_valid)
        wb = w_valid[idx]  # (K, n_valid)
        Xb_w = Xb * wb[:, :, None]
        Ab = np.einsum("kjp,kjq->kpq", Xb_w, Xb)
        rhs = np.einsum("kjp,kj->kp", Xb_w, yb)
        # Check invertibility per bootstrap replicate
        dets = np.linalg.det(Ab)
        good = np.abs(dets) > 1e-15
        beta_boot_mat = np.repeat(beta_corr[None, :], nb_distribution, axis=0)
        if np.any(good):
            beta_boot_mat[good] = np.linalg.solve(Ab[good], rhs[good, :, None])[..., 0]

    abs_corr = np.abs(beta_corr)
    valid_corr = abs_corr > 1e-15
    distortion_coeff = np.where(
        valid_corr,
        100.0 * (beta_raw - beta_corr) / np.maximum(abs_corr, 1e-300),
        np.nan,
    )

    if settings == "R_default":
        denom_corr = np.maximum(abs_corr, 1e-300)
        bias_obs = (beta_raw - beta_corr) / denom_corr
        bias_exp = (beta_boot_mat - beta_corr[None, :]) / denom_corr[None, :]
        p_dist = np.where(
            valid_corr,
            np.sum(np.abs(bias_exp) >= np.abs(bias_obs)[None, :], axis=0) / float(nb_distribution),
            np.nan,
        )
    else:
        diff_obs = np.abs(beta_raw - beta_corr)
        diff_boot = np.abs(beta_boot_mat - beta_corr[None, :])
        p_dist = (np.sum(diff_boot >= diff_obs[None, :], axis=0) + 1.0) / (nb_distribution + 1.0)

    # Joint Mahalanobis distortion test across all P exposures
    centered = beta_boot_mat - beta_corr[None, :]
    cov_boot = (centered.T @ centered) / max(nb_distribution - 1, 1)
    delta_obs = beta_raw - beta_corr
    pinv_cov = np.linalg.pinv(cov_boot, rcond=1e-10)
    q_obs = float(delta_obs @ pinv_cov @ delta_obs)
    q_boot = np.einsum("kp,pq,kq->k", centered, pinv_cov, centered)
    if settings == "R_default":
        p_joint = float(np.sum(q_boot >= q_obs) / float(nb_distribution))
    else:
        p_joint = float((np.sum(q_boot >= q_obs) + 1.0) / (nb_distribution + 1.0))

    return distortion_coeff, p_dist, p_joint


def mvmr_presso(
    beta_outcome: Sequence[float] | np.ndarray | str,
    beta_exposure: Sequence[float] | np.ndarray | Sequence[str],
    sd_outcome: Sequence[float] | np.ndarray | str,
    sd_exposure: Sequence[float] | np.ndarray | Sequence[str],
    *,
    data: pd.DataFrame | None = None,
    snp_ids: Sequence[str] | None = None,
    exposure_names: Sequence[str] | None = None,
    outlier_test: bool = True,
    distortion_test: bool = True,
    signif_threshold: float = 0.05,
    nb_distribution: int | None = None,
    outlier_method: Literal["empirical", "chisq"] = "empirical",
    weighted_rss: bool | None = None,
    seed: int | np.random.Generator | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> MVMRPressoResult:
    """Run Multivariable MR-PRESSO (P >= 2 exposures) using Sherman-Morrison LOO updates."""
    mode_settings = _validate_settings(settings)
    nb_dist = (
        (1000 if mode_settings == "R_default" else 5000)
        if nb_distribution is None
        else int(nb_distribution)
    )
    w_rss = (mode_settings == "R_default") if weighted_rss is None else bool(weighted_rss)

    if not (0.0 < signif_threshold < 1.0):
        raise ValueError("signif_threshold must lie strictly in (0, 1).")
    if nb_dist < 10:
        raise ValueError("nb_distribution must be >= 10.")
    if outlier_method not in ("empirical", "chisq"):
        raise ValueError("outlier_method must be 'empirical' or 'chisq'.")

    y_orig, X_orig, sy, SX, ids, exp_names, _ = _extract_inputs(
        beta_outcome, beta_exposure, sd_outcome, sd_exposure, data, snp_ids, exposure_names
    )
    n, p = X_orig.shape
    if p < 2:
        raise ValueError("mvmr_presso expects P >= 2 exposures; use mr_presso for single-exposure MR.")

    w = 1.0 / (sy**2)
    raw_fit = _fit_ivw(X_orig, y_orig, w)
    loo_betas_orig, rss_per_snp, rss_obs = leave_one_out_ivw(
        y_orig, X_orig, sy, weighted_rss=w_rss, settings=mode_settings
    )

    # Sign-canonicalized working copy guarantees bit-for-bit MC allele-flip invariance
    X_canon, y_canon = _canonicalize_signs(X_orig, y_orig)
    loo_betas_canon, r_obs_sq_canon, _ = leave_one_out_ivw(
        y_canon, X_canon, sy, weighted_rss=w_rss, settings=mode_settings
    )
    pred_y_loo_canon = np.sum(X_canon * loo_betas_canon, axis=1)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    rss_exp, p_emp_per_snp, expected_rss_per_snp = _simulate_rss_multivariable(
        X_canon,
        SX,
        pred_y_loo_canon,
        sy,
        w,
        r_obs_sq_canon,
        nb_dist,
        w_rss,
        rng,
        loo_betas_obs=loo_betas_canon,
        settings=mode_settings,
    )

    if mode_settings == "R_default":
        exceed_global = int(np.sum(rss_exp >= rss_obs))
        p_global_emp = float(exceed_global / nb_dist)
        p_global = p_global_emp
    else:
        exceed_global = int(np.sum(rss_exp > rss_obs))
        p_global = float((exceed_global + 1.0) / (nb_dist + 1.0))
        p_global_emp = float(exceed_global / nb_dist)
    p_global_fmt = f"<{1.0 / nb_dist:.4g}" if exceed_global == 0 else f"{p_global:.4g}"

    outlier_indices: list[int] = []
    outlier_snps: list[str] = []
    outlier_pvals: np.ndarray | None = None
    outlier_pvals_bonf: np.ndarray | None = None
    outlier_pvals_chisq: np.ndarray | None = None
    beta_corr: np.ndarray | None = None
    se_corr: np.ndarray | None = None
    t_corr: np.ndarray | None = None
    p_corr: np.ndarray | None = None
    dist_coeff: np.ndarray | None = None
    p_dist: np.ndarray | None = None
    p_dist_joint: float | None = None

    if outlier_test and p_global < signif_threshold:
        outlier_pvals_chisq = stats.chi2.sf(
            rss_per_snp / np.maximum(expected_rss_per_snp, 1e-300), df=1
        )
        outlier_pvals = p_emp_per_snp if outlier_method == "empirical" else outlier_pvals_chisq
        outlier_pvals_bonf = np.minimum(1.0, n * outlier_pvals)
        outlier_indices = [int(i) for i in np.flatnonzero(outlier_pvals_bonf < signif_threshold)]
        outlier_snps = [ids[i] for i in outlier_indices]

        if outlier_indices:
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[outlier_indices] = False
            if int(np.sum(keep_mask)) > p:
                try:
                    corr_fit = _fit_ivw(X_orig[keep_mask], y_orig[keep_mask], w[keep_mask])
                    beta_corr = corr_fit.beta
                    se_corr = corr_fit.se
                    t_corr = corr_fit.t_stat
                    p_corr = corr_fit.p_value
                    if distortion_test:
                        dist_coeff, p_dist, p_dist_joint = _bootstrap_distortion(
                            X_canon[keep_mask],
                            y_canon[keep_mask],
                            w[keep_mask],
                            raw_fit.beta,
                            beta_corr,
                            nb_dist,
                            rng,
                            settings=mode_settings,
                        )
                except ValueError:
                    pass

    return MVMRPressoResult(
        n_instruments=n,
        n_exposures=p,
        snp_ids=ids,
        exposure_names=exp_names,
        beta_exposure=X_orig.copy(),
        beta_outcome=y_orig.copy(),
        sd_exposure=SX.copy(),
        sd_outcome=sy.copy(),
        beta_raw=raw_fit.beta,
        se_raw=raw_fit.se,
        t_stat_raw=raw_fit.t_stat,
        p_value_raw=raw_fit.p_value,
        beta_corrected=beta_corr,
        se_corrected=se_corr,
        t_stat_corrected=t_corr,
        p_value_corrected=p_corr,
        rss_obs=rss_obs,
        rss_per_snp=rss_per_snp,
        expected_rss_per_snp=expected_rss_per_snp,
        loo_betas=loo_betas_orig,
        p_global=p_global,
        p_global_empirical=p_global_emp,
        p_global_formatted=p_global_fmt,
        outlier_indices=outlier_indices,
        outlier_snps=outlier_snps,
        outlier_pvalues=outlier_pvals,
        outlier_pvalues_bonferroni=outlier_pvals_bonf,
        outlier_pvalues_chisq=outlier_pvals_chisq,
        distortion_coeff=dist_coeff,
        p_distortion=p_dist,
        p_distortion_joint=p_dist_joint,
        nb_distribution=nb_dist,
        signif_threshold=signif_threshold,
        weighted_rss=w_rss,
        settings=mode_settings,
    )


def mr_presso(
    beta_outcome: Sequence[float] | np.ndarray | str,
    beta_exposure: Sequence[float] | np.ndarray | str | Sequence[str],
    sd_outcome: Sequence[float] | np.ndarray | str,
    sd_exposure: Sequence[float] | np.ndarray | str | Sequence[str],
    *,
    data: pd.DataFrame | None = None,
    snp_ids: Sequence[str] | None = None,
    exposure_names: Sequence[str] | None = None,
    outlier_test: bool = True,
    distortion_test: bool = True,
    signif_threshold: float = 0.05,
    nb_distribution: int | None = None,
    outlier_method: Literal["empirical", "chisq"] = "empirical",
    weighted_rss: bool | None = None,
    seed: int | np.random.Generator | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> MRPressoResult | MVMRPressoResult:
    """Run MR-PRESSO global, outlier, and distortion tests (Verbanck et al., Nature Genetics 2018).

    Automatically dispatches to `mvmr_presso` when `beta_exposure` contains P >= 2 exposures.
    """
    mode_settings = _validate_settings(settings)
    nb_dist = (
        (1000 if mode_settings == "R_default" else 5000)
        if nb_distribution is None
        else int(nb_distribution)
    )
    w_rss = (mode_settings == "R_default") if weighted_rss is None else bool(weighted_rss)

    if not (0.0 < signif_threshold < 1.0):
        raise ValueError("signif_threshold must lie strictly in (0, 1).")
    if nb_dist < 10:
        raise ValueError("nb_distribution must be >= 10.")
    if outlier_method not in ("empirical", "chisq"):
        raise ValueError("outlier_method must be 'empirical' or 'chisq'.")

    y_orig, X_orig, sy, SX, ids, exp_names, _ = _extract_inputs(
        beta_outcome, beta_exposure, sd_outcome, sd_exposure, data, snp_ids, exposure_names
    )
    if X_orig.shape[1] >= 2:
        return mvmr_presso(
            beta_outcome=beta_outcome,
            beta_exposure=beta_exposure,
            sd_outcome=sd_outcome,
            sd_exposure=sd_exposure,
            data=data,
            snp_ids=snp_ids,
            exposure_names=exposure_names,
            outlier_test=outlier_test,
            distortion_test=distortion_test,
            signif_threshold=signif_threshold,
            nb_distribution=nb_dist,
            outlier_method=outlier_method,
            weighted_rss=w_rss,
            seed=seed,
            settings=mode_settings,
        )

    n = y_orig.size
    x_orig = X_orig[:, 0]
    sx = SX[:, 0]
    w = 1.0 / (sy**2)

    raw_fit = _fit_ivw(X_orig, y_orig, w)
    loo_betas_orig, rss_per_snp, rss_obs = leave_one_out_ivw(
        y_orig, x_orig, sy, weighted_rss=w_rss, settings=mode_settings
    )

    X_canon, y_canon = _canonicalize_signs(X_orig, y_orig)
    x_canon = X_canon[:, 0]
    loo_betas_canon, r_obs_sq_canon, _ = leave_one_out_ivw(
        y_canon, x_canon, sy, weighted_rss=w_rss, settings=mode_settings
    )
    pred_y_loo_canon = loo_betas_canon * x_canon

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    rss_exp, p_emp_per_snp, expected_rss_per_snp = _simulate_rss_univariable(
        x_canon,
        sx,
        pred_y_loo_canon,
        sy,
        w,
        r_obs_sq_canon,
        nb_dist,
        w_rss,
        rng,
        loo_betas_obs=loo_betas_canon,
        settings=mode_settings,
    )

    if mode_settings == "R_default":
        exceed_global = int(np.sum(rss_exp >= rss_obs))
        p_global_emp = float(exceed_global / nb_dist)
        p_global = p_global_emp
    else:
        exceed_global = int(np.sum(rss_exp > rss_obs))
        p_global = float((exceed_global + 1.0) / (nb_dist + 1.0))
        p_global_emp = float(exceed_global / nb_dist)
    p_global_fmt = f"<{1.0 / nb_dist:.4g}" if exceed_global == 0 else f"{p_global:.4g}"

    outlier_indices: list[int] = []
    outlier_snps: list[str] = []
    outlier_pvals: np.ndarray | None = None
    outlier_pvals_bonf: np.ndarray | None = None
    outlier_pvals_chisq: np.ndarray | None = None
    beta_corr: float | None = None
    se_corr: float | None = None
    t_corr: float | None = None
    p_corr: float | None = None
    dist_coeff: float | None = None
    p_dist: float | None = None

    if outlier_test and p_global < signif_threshold:
        outlier_pvals_chisq = stats.chi2.sf(
            rss_per_snp / np.maximum(expected_rss_per_snp, 1e-300), df=1
        )
        outlier_pvals = p_emp_per_snp if outlier_method == "empirical" else outlier_pvals_chisq
        outlier_pvals_bonf = np.minimum(1.0, n * outlier_pvals)
        outlier_indices = [int(i) for i in np.flatnonzero(outlier_pvals_bonf < signif_threshold)]
        outlier_snps = [ids[i] for i in outlier_indices]

        if outlier_indices:
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[outlier_indices] = False
            if int(np.sum(keep_mask)) > 1:
                corr_fit = _fit_ivw(X_orig[keep_mask], y_orig[keep_mask], w[keep_mask])
                beta_corr = float(corr_fit.beta[0])
                se_corr = float(corr_fit.se[0])
                t_corr = float(corr_fit.t_stat[0])
                p_corr = float(corr_fit.p_value[0])
                if distortion_test:
                    dc_arr, pd_arr, _ = _bootstrap_distortion(
                        X_canon[keep_mask],
                        y_canon[keep_mask],
                        w[keep_mask],
                        raw_fit.beta,
                        corr_fit.beta,
                        nb_dist,
                        rng,
                        settings=mode_settings,
                    )
                    dist_coeff = float(dc_arr[0])
                    p_dist = float(pd_arr[0])

    return MRPressoResult(
        n_instruments=n,
        snp_ids=ids,
        exposure_name=exp_names[0],
        beta_exposure=x_orig.copy(),
        beta_outcome=y_orig.copy(),
        sd_exposure=sx.copy(),
        sd_outcome=sy.copy(),
        beta_raw=float(raw_fit.beta[0]),
        se_raw=float(raw_fit.se[0]),
        t_stat_raw=float(raw_fit.t_stat[0]),
        p_value_raw=float(raw_fit.p_value[0]),
        beta_corrected=beta_corr,
        se_corrected=se_corr,
        t_stat_corrected=t_corr,
        p_value_corrected=p_corr,
        rss_obs=rss_obs,
        rss_per_snp=rss_per_snp,
        expected_rss_per_snp=expected_rss_per_snp,
        loo_betas=loo_betas_orig,
        p_global=p_global,
        p_global_empirical=p_global_emp,
        p_global_formatted=p_global_fmt,
        outlier_indices=outlier_indices,
        outlier_snps=outlier_snps,
        outlier_pvalues=outlier_pvals,
        outlier_pvalues_bonferroni=outlier_pvals_bonf,
        outlier_pvalues_chisq=outlier_pvals_chisq,
        distortion_coeff=dist_coeff,
        p_distortion=p_dist,
        nb_distribution=nb_dist,
        signif_threshold=signif_threshold,
        weighted_rss=w_rss,
        settings=mode_settings,
    )
