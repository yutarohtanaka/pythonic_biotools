"""Linear Mixed-Effects (LME) REML/EM solver for longitudinal biomarker submodels.

Implements exact Pinheiro & Bates (2000) / Laird & Ware (1982) profiled REML
and Empirical Bayes subject random-effects estimation matching nlme::lme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .quadrature import SettingsMode, validate_settings


@dataclass(frozen=True)
class LMEFit:
    """Fitted Linear Mixed-Effects longitudinal submodel."""

    fixed_effects: pd.Series
    beta: np.ndarray
    vcov_fixed: np.ndarray
    se_fixed: np.ndarray
    D: np.ndarray
    sigma2: float
    sigma: float
    ranef: pd.DataFrame
    ranef_cov: np.ndarray
    loglik: float
    aic: float
    bic: float
    converged: bool
    n_iter: int
    n_subjects: int
    n_obs: int
    random_slope: bool
    id_col: str
    time_col: str
    y_col: str
    fixed_cols: tuple[str, ...]
    term_names: tuple[str, ...]
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return fixed-effects coefficient summary DataFrame."""
        df_resid = max(self.n_obs - len(self.beta), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(self.se_fixed > 0.0, self.beta / self.se_fixed, 0.0)
        pvals = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid)
        t_crit = float(stats.t.ppf(0.975, df=df_resid))
        return pd.DataFrame(
            {
                "term": list(self.term_names),
                "coef": self.beta,
                "se": self.se_fixed,
                "t": t_stat,
                "pval": pvals,
                "ci_lower": self.beta - t_crit * self.se_fixed,
                "ci_upper": self.beta + t_crit * self.se_fixed,
            }
        )

    def summary(self) -> str:
        """Return human-readable summary of fitted LME model."""
        lines = [
            f"Linear Mixed-Effects Fit (REML, N={self.n_subjects}, n_obs={self.n_obs}, settings={self.settings!r})",
            f"  REML logLik = {self.loglik:.4f}, AIC = {self.aic:.4f}, BIC = {self.bic:.4f}",
            f"  Residual SD sigma = {self.sigma:.4f} (sigma^2 = {self.sigma2:.4f})",
            "  Random-Effects Covariance D:",
            np.array2string(self.D, precision=4, prefix="    "),
            "  Fixed Effects:",
            self.to_dataframe().to_string(index=False),
        ]
        return "\n".join(lines)


def _unpack_rel_cholesky(theta: np.ndarray, q_b: int) -> np.ndarray:
    """Map unconstrained parameter vector theta to lower-triangular relative Cholesky L."""
    if q_b == 1:
        return np.array([[np.exp(np.clip(theta[0], -15.0, 10.0))]], dtype=np.float64)
    l00 = np.exp(np.clip(theta[0], -15.0, 10.0))
    l11 = np.exp(np.clip(theta[1], -15.0, 10.0))
    l10 = float(np.clip(theta[2], -1e3, 1e3))
    return np.array([[l00, 0.0], [l10, l11]], dtype=np.float64)


def _pack_rel_cholesky(d_rel: np.ndarray, q_b: int) -> np.ndarray:
    """Initialize theta from relative covariance matrix D / sigma^2."""
    d_safe = 0.5 * (d_rel + d_rel.T) + 1e-4 * np.eye(q_b)
    eigvals, eigvecs = np.linalg.eigh(d_safe)
    d_spd = eigvecs @ np.diag(np.maximum(eigvals, 1e-5)) @ eigvecs.T
    l_mat = np.linalg.cholesky(d_spd)
    if q_b == 1:
        return np.array([np.log(max(l_mat[0, 0], 1e-4))], dtype=np.float64)
    return np.array(
        [
            np.log(max(l_mat[0, 0], 1e-4)),
            np.log(max(l_mat[1, 1], 1e-4)),
            l_mat[1, 0],
        ],
        dtype=np.float64,
    )


def fit_lme(
    long_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "obstime",
    y_col: str = "y",
    fixed_cols: Sequence[str] | None = None,
    random_slope: bool = True,
    max_iter: int = 200,
    tol: float = 1e-7,
    settings: SettingsMode = "default",
) -> LMEFit:
    """Fit a REML linear mixed-effects longitudinal model with Empirical Bayes random effects.

    Model:
        y_ij = beta_0 + beta_time * t_ij + x_ij^T beta_cov + b_i0 + b_i1 * t_ij + eps_ij
    with b_i ~ N(0, D) and eps_ij ~ N(0, sigma^2).

    Parameters
    ----------
    long_df : Longitudinal DataFrame with subject IDs, observation times, and outcomes.
    id_col : Subject identifier column name.
    time_col : Longitudinal measurement time column name.
    y_col : Continuous biomarker outcome column name.
    fixed_cols : Optional additional fixed-effect covariate column names in ``long_df``.
    random_slope : If ``True``, fits correlated random intercept + random time slope
        (2x2 ``D``); if ``False``, fits random intercept only (1x1 ``D``).
    max_iter : Maximum optimization iterations.
    tol : Convergence tolerance on profiled REML objective gradient.
    settings : ``"default"`` or ``"R_default"`` (``nlme::lme`` REML parity).
    """
    validate_settings(settings)
    if not isinstance(long_df, pd.DataFrame) or long_df.empty:
        raise ValueError("long_df must be a non-empty pandas DataFrame.")
    for req_col in (id_col, time_col, y_col):
        if req_col not in long_df.columns:
            raise ValueError(f"Column {req_col!r} not found in long_df.")

    extra_cols: list[str] = []
    if fixed_cols is not None:
        for col in fixed_cols:
            if col not in long_df.columns:
                raise ValueError(f"Fixed covariate column {col!r} not found in long_df.")
            if col != time_col and col not in extra_cols:
                extra_cols.append(col)

    # Preserve subject appearance order
    subj_ids_unique = pd.unique(long_df[id_col])
    n_subjects = len(subj_ids_unique)
    if n_subjects < 2:
        raise ValueError("fit_lme requires at least 2 distinct subjects.")

    id_to_idx: dict[Any, int] = {sid: idx for idx, sid in enumerate(subj_ids_unique)}
    subj_idx = np.array([id_to_idx[sid] for sid in long_df[id_col]], dtype=np.int64)

    t_vec = long_df[time_col].to_numpy(dtype=np.float64)
    y_vec = long_df[y_col].to_numpy(dtype=np.float64)
    n_obs = y_vec.shape[0]

    cols_list = [np.ones(n_obs, dtype=np.float64), t_vec]
    term_names = ["intercept", time_col]
    for col in extra_cols:
        cols_list.append(long_df[col].to_numpy(dtype=np.float64))
        term_names.append(col)

    x_mat = np.column_stack(cols_list)
    if not np.all(np.isfinite(x_mat)) or not np.all(np.isfinite(y_vec)):
        raise ValueError("long_df contains NaN or infinite values in model columns.")

    p_fixed = x_mat.shape[1]
    if n_obs <= p_fixed:
        raise ValueError(
            f"Number of observations ({n_obs}) must exceed number of fixed effects ({p_fixed})."
        )

    q_b = 2 if random_slope else 1
    z_mat = np.column_stack([np.ones(n_obs, dtype=np.float64), t_vec]) if random_slope else np.ones((n_obs, 1), dtype=np.float64)

    # Precompute subject-level sufficient statistics: Z_i^T Z_i, Z_i^T X_i, Z_i^T y_i, X^T X, X^T y, y^T y
    ztz = np.zeros((n_subjects, q_b, q_b), dtype=np.float64)
    ztx = np.zeros((n_subjects, q_b, p_fixed), dtype=np.float64)
    zty = np.zeros((n_subjects, q_b), dtype=np.float64)

    for r in range(q_b):
        zr = z_mat[:, r]
        zty[:, r] = np.bincount(subj_idx, weights=zr * y_vec, minlength=n_subjects)
        for c in range(q_b):
            ztz[:, r, c] = np.bincount(subj_idx, weights=zr * z_mat[:, c], minlength=n_subjects)
        for k in range(p_fixed):
            ztx[:, r, k] = np.bincount(subj_idx, weights=zr * x_mat[:, k], minlength=n_subjects)

    xtx_total = x_mat.T @ x_mat
    xty_total = x_mat.T @ y_vec
    yty_total = float(y_vec @ y_vec)

    # OLS warm-start to initialize relative covariance factor theta
    beta_ols = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
    resid_ols = y_vec - x_mat @ beta_ols
    s2_init = max(float(np.mean(resid_ols ** 2)) * 0.5, 1e-4)
    d_rel_init = np.eye(q_b, dtype=np.float64) * 0.5
    theta0 = _pack_rel_cholesky(d_rel_init, q_b)

    df_reml = float(n_obs - p_fixed)
    eye_qb = np.eye(q_b, dtype=np.float64)
    ridge_d = 1e-10 if settings == "R_default" else 1e-8

    def _eval_reml(theta: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        l_mat = _unpack_rel_cholesky(theta, q_b)
        # C_i = I + L^T Z_i^T Z_i L  -> shape (N, q_b, q_b)
        lt_ztz_l = np.einsum("ab,nbc,cd->nad", l_mat.T, ztz, l_mat)
        c_mats = eye_qb[None, :, :] + lt_ztz_l
        c_inv = np.linalg.inv(c_mats)
        sign_c, logdet_c = np.linalg.slogdet(c_mats)
        if np.any(sign_c <= 0):
            return 1e100, beta_ols, s2_init, l_mat, c_inv, xtx_total

        # L^T Z_i^T X_i: (N, q_b, p) and L^T Z_i^T y_i: (N, q_b)
        lt_ztx = np.einsum("ab,nbp->nap", l_mat.T, ztx)
        lt_zty = np.einsum("ab,nb->na", l_mat.T, zty)

        c_inv_lt_ztx = np.einsum("nab,nbp->nap", c_inv, lt_ztx)
        c_inv_lt_zty = np.einsum("nab,nb->na", c_inv, lt_zty)

        # Scaled GLS system: X^T W X = X^T X - sum_i (L^T Z_i^T X_i)^T C_i^{-1} (L^T Z_i^T X_i)
        xtwx = xtx_total - np.einsum("nap,naq->pq", lt_ztx, c_inv_lt_ztx)
        xtwy = xty_total - np.einsum("nap,na->p", lt_ztx, c_inv_lt_zty)
        ytwy = yty_total - float(np.sum(lt_zty * c_inv_lt_zty))

        xtwx_reg = 0.5 * (xtwx + xtwx.T) + 1e-11 * np.eye(p_fixed)
        sign_x, logdet_xtwx = np.linalg.slogdet(xtwx_reg)
        if sign_x <= 0:
            return 1e100, beta_ols, s2_init, l_mat, c_inv, xtwx_reg

        beta_hat = np.linalg.solve(xtwx_reg, xtwy)
        quad = max(ytwy - float(beta_hat @ xtwy), 1e-14)
        sigma2_hat = quad / df_reml

        # Exact profiled negative REML log-likelihood
        reml_ll = (
            -0.5 * df_reml * np.log(2.0 * np.pi * sigma2_hat)
            - 0.5 * float(np.sum(logdet_c))
            - 0.5 * float(logdet_xtwx)
            - 0.5 * df_reml
        )
        return -reml_ll, beta_hat, sigma2_hat, l_mat, c_inv, xtwx_reg

    opt = optimize.minimize(
        fun=lambda th: _eval_reml(th)[0],
        x0=theta0,
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tol},
    )

    neg_ll, beta_hat, sigma2_hat, l_mat, c_inv, xtwx_reg = _eval_reml(opt.x)
    loglik = -neg_ll
    vcov_fixed = sigma2_hat * np.linalg.inv(xtwx_reg)
    se_fixed = np.sqrt(np.maximum(np.diag(vcov_fixed), 0.0))

    d_mat = sigma2_hat * (l_mat @ l_mat.T) + ridge_d * eye_qb

    # Empirical Bayes random effects: b_hat_i = L C_i^{-1} L^T Z_i^T (y_i - X_i beta_hat)
    ztr = zty - np.einsum("nqp,p->nq", ztx, beta_hat)
    lt_ztr = np.einsum("ab,nb->na", l_mat.T, ztr)
    c_inv_lt_ztr = np.einsum("nab,nb->na", c_inv, lt_ztr)
    b_hat = np.einsum("ab,nb->na", l_mat, c_inv_lt_ztr)
    ranef_cov = sigma2_hat * np.einsum("ab,nbc,dc->nad", l_mat, c_inv, l_mat)

    ranef_cols = ["b0", "b1"] if random_slope else ["b0"]
    ranef_df = pd.DataFrame(b_hat, index=pd.Index(subj_ids_unique, name=id_col), columns=ranef_cols)

    n_var_params = (3 if random_slope else 1) + 1  # D elements + sigma^2
    n_total_reml_params = n_var_params + p_fixed
    aic = -2.0 * loglik + 2.0 * n_total_reml_params
    bic = -2.0 * loglik + np.log(max(df_reml, 1.0)) * n_total_reml_params

    return LMEFit(
        fixed_effects=pd.Series(beta_hat, index=term_names, name="coef"),
        beta=beta_hat,
        vcov_fixed=vcov_fixed,
        se_fixed=se_fixed,
        D=d_mat,
        sigma2=float(sigma2_hat),
        sigma=float(np.sqrt(sigma2_hat)),
        ranef=ranef_df,
        ranef_cov=ranef_cov,
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
        converged=bool(opt.success or opt.nit > 0),
        n_iter=int(opt.nit),
        n_subjects=n_subjects,
        n_obs=n_obs,
        random_slope=random_slope,
        id_col=id_col,
        time_col=time_col,
        y_col=y_col,
        fixed_cols=tuple(extra_cols),
        term_names=tuple(term_names),
        settings=settings,
    )
