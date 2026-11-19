"""Heteroskedastic G-Estimation for cis-MR (MR-GENIUS: individual & vQTL summary).

References
----------
Tchetgen Tchetgen, E., Sun, B., & Walter, S. (2021). The GENIUS approach to robust
Mendelian randomization inference. Statistical Science, 36(3), 443-464.
https://doi.org/10.1214/20-STS792

Ye, T., Liu, Z., Sun, B., & Tchetgen Tchetgen, E. (2024). GENIUS-MAWII: For robust
Mendelian randomization with many weak invalid instruments. JRSS-B, 86(4), 1045-1067.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .utils import (
    SettingsType,
    as_1d_float,
    as_2d_float,
    prepare_ld_matrix,
    sym_solve,
    validate_settings,
    wald_inference,
)


@dataclass
class MRGeniusResult:
    """Result of MR-GENIUS causal estimation (individual-level or vQTL summary)."""

    estimate: float
    se: float
    pval: float
    ci_low: float
    ci_high: float
    n_instruments: int
    n_samples: int | None
    het_f_stat: float
    het_f_pval: float
    j_stat: float
    j_df: int
    j_pval: float
    mode: str
    settings: SettingsType

    @property
    def theta(self) -> float:
        return self.estimate

    @property
    def beta(self) -> float:
        return self.estimate

    def to_dict(self) -> dict[str, object]:
        return {
            "method": f"MR-GENIUS ({self.mode})",
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pval": self.pval,
            "n_instruments": self.n_instruments,
            "n_samples": self.n_samples,
            "het_f_stat": self.het_f_stat,
            "het_f_pval": self.het_f_pval,
            "j_stat": self.j_stat,
            "j_df": self.j_df,
            "j_pval": self.j_pval,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"MR-GENIUS [{self.mode}] (settings={self.settings!r}): "
            f"theta={self.estimate:.4f} (SE={self.se:.4f}, p={self.pval:.3e}), "
            f"95% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"F_het={self.het_f_stat:.2f} (p={self.het_f_pval:.3g})"
        )


def _ols_residuals(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute OLS residuals ``target - design @ coef`` via QR/SVD least squares."""
    coef, *_ = linalg.lstsq(design, target, check_finite=False)
    return target - design @ coef


def mr_genius(
    G: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    X: np.ndarray | Sequence[float],
    Y: np.ndarray | Sequence[float],
    covariates: np.ndarray | Sequence[float] | Sequence[Sequence[float]] | None = None,
    alpha: float = 0.05,
    settings: SettingsType = "default",
) -> MRGeniusResult:
    """Individual-level MR-GENIUS exploiting exposure heteroskedasticity Var(X | G).

    Identifies the causal effect of ``X`` on ``Y`` even when all cis-variants
    exert direct additive pleiotropic effects on ``Y``, provided ``Var(X | G, C)``
    varies with ``G`` (Tchetgen Tchetgen et al., Statistical Science 2021).
    """
    validate_settings(settings)
    G_mat = as_2d_float(G, "G", min_rows=10)
    n, p = G_mat.shape
    X_vec = as_1d_float(X, "X", min_len=n)
    Y_vec = as_1d_float(Y, "Y", min_len=n)
    if X_vec.shape[0] != n or Y_vec.shape[0] != n:
        raise ValueError("G, X, and Y must have the same number of samples n.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    ones = np.ones((n, 1), dtype=np.float64)
    if covariates is not None:
        C_mat = as_2d_float(covariates, "covariates", min_rows=n)
        if C_mat.shape[0] != n:
            raise ValueError("covariates must have n rows matching G.")
        Z_c = np.hstack([ones, C_mat])
    else:
        C_mat = None
        Z_c = ones

    if n <= p + Z_c.shape[1] + 2:
        raise ValueError("Sample size n is too small relative to number of instruments/covariates.")

    # Residualize genotypes on covariates: G_tilde = G - E[G | 1, C]
    G_tilde = _ols_residuals(Z_c, G_mat)
    if np.any(np.var(G_tilde, axis=0) <= 1e-14):
        raise ValueError("At least one column of G has zero residual variance given covariates.")

    # Build exposure mean design W
    w_blocks = [Z_c, G_mat]
    if settings == "default" and n > (2 * p + Z_c.shape[1] + 15):
        quad_cols = []
        for j in range(p):
            uniq = np.unique(np.round(G_mat[:, j], 8))
            if uniq.size >= 3:
                g_c = G_mat[:, j] - np.mean(G_mat[:, j])
                g_sq = _ols_residuals( np.hstack([ones, G_mat[:, [j]]]), g_c**2 )
                if np.var(g_sq) > 1e-10:
                    quad_cols.append(g_sq[:, None])
        if quad_cols:
            w_blocks.append(np.hstack(quad_cols))
    W = np.hstack(w_blocks)

    eps_x = _ols_residuals(W, X_vec)

    # Heteroskedasticity first-stage F-test: regress eps_x^2 on G_tilde controlling for Z_c
    u2 = eps_x**2
    u2_res_null = _ols_residuals(Z_c, u2)
    u2_res_full = _ols_residuals(np.hstack([Z_c, G_tilde]), u2)
    rss0 = float(np.sum(u2_res_null**2))
    rss1 = float(np.sum(u2_res_full**2))
    df2 = max(n - Z_c.shape[1] - p, 1)
    het_f_stat = max(((rss0 - rss1) / p) / max(rss1 / df2, 1e-18), 0.0)
    het_f_pval = float(stats.f.sf(het_f_stat, dfn=p, dfd=df2))

    # Outcome target: doubly-orthogonalized eps_y in default, raw Y in R_default
    if settings == "default":
        y_part = _ols_residuals(W, Y_vec)
        x_part = eps_x
    else:
        y_part = Y_vec
        x_part = X_vec

    # Precompute WtW^+ W^T for exact nuisance M-estimation influence function
    W_pinv = linalg.pinv(W)  # shape (d_W, n)

    def influence_matrix(theta_val: float) -> np.ndarray:
        u = y_part - theta_val * x_part
        prod = eps_x * u
        # Correct for E[G | Z_c] estimation
        prod_centered = _ols_residuals(Z_c, prod)
        h = G_tilde * prod_centered[:, None]  # (n, p)
        # Correct for E[X | W] (and E[Y | W] in default) nuisance plug-in
        if settings == "default":
            d_w_factor = u - theta_val * eps_x
        else:
            d_w_factor = u
        # J_gamma = -(1/n) sum_i G_tilde_i * d_w_factor_i * W_i^T -> shape (p, d_W)
        gw = (G_tilde * d_w_factor[:, None]).T @ W  # (p, d_W)
        # Sample-i contribution from nuisance OLS plug-in (W^+ eps_x):
        h = h - (W_pinv.T @ gw.T) * eps_x[:, None]
        return h

    # Moment vectors a = (1/n) G_tilde^T (eps_x * y_part), b = (1/n) G_tilde^T (eps_x * x_part)
    m_y = (G_tilde.T @ (eps_x * y_part)) / n  # (p,)
    m_x = (G_tilde.T @ (eps_x * x_part)) / n  # (p,)

    if np.linalg.norm(m_x) <= 1e-12:
        raise ValueError(
            "Exposure heteroskedasticity covariance Cov(G, Var(X|G)) is numerically zero; "
            "MR-GENIUS is not identified."
        )

    if p == 1:
        theta_hat = float(m_y[0] / m_x[0])
        h_mat = influence_matrix(theta_hat)
        omega = float(np.mean(h_mat[:, 0] ** 2))
        se_hat = float(np.sqrt(max(omega, 1e-18) / n) / abs(m_x[0]))
        j_stat = 0.0
        j_df = 0
        j_pval = np.nan
    else:
        # Step 1: initial GMM weighting with Cov(G_tilde * eps_x^2) or identity
        w_init = (G_tilde.T @ G_tilde) / n
        theta_curr = float((m_x @ sym_solve(w_init, m_y)) / (m_x @ sym_solve(w_init, m_x)))

        n_steps = 1 if settings == "R_default" else 15
        for _ in range(n_steps):
            h_curr = influence_matrix(theta_curr)
            om_curr = (h_curr.T @ h_curr) / n
            denom = float(m_x @ sym_solve(om_curr, m_x))
            theta_next = float((m_x @ sym_solve(om_curr, m_y)) / max(denom, 1e-18))
            if abs(theta_next - theta_curr) <= 1e-9 * (1.0 + abs(theta_curr)):
                theta_curr = theta_next
                break
            theta_curr = theta_next

        theta_hat = theta_curr
        h_final = influence_matrix(theta_hat)
        om_final = (h_final.T @ h_final) / n
        g_final = m_y - theta_hat * m_x
        j_df = p - 1
        j_stat = max(float(n * (g_final @ sym_solve(om_final, g_final))), 0.0)
        j_pval = float(stats.chi2.sf(j_stat, df=j_df))
        info = float(m_x @ sym_solve(om_final, m_x))
        se_base = float(1.0 / np.sqrt(max(n * info, 1e-18)))
        se_hat = (
            float(se_base * np.sqrt(max(1.0, j_stat / j_df)))
            if settings == "default"
            else se_base
        )

    pval, ci_low, ci_high = wald_inference(theta_hat, se_hat, alpha=alpha, df=None)

    return MRGeniusResult(
        estimate=theta_hat,
        se=se_hat,
        pval=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_instruments=p,
        n_samples=n,
        het_f_stat=het_f_stat,
        het_f_pval=het_f_pval,
        j_stat=j_stat,
        j_df=j_df,
        j_pval=j_pval,
        mode="individual",
        settings=settings,
    )


def mr_genius_summary(
    bx: np.ndarray | Sequence[float],
    sx: np.ndarray | Sequence[float],
    by: np.ndarray | Sequence[float],
    sy: np.ndarray | Sequence[float],
    v_gx: np.ndarray | Sequence[float],
    se_vgx: np.ndarray | Sequence[float] | None = None,
    ld_R: np.ndarray | None = None,
    alpha: float = 0.05,
    settings: SettingsType = "default",
) -> MRGeniusResult:
    """Summary-level vQTL variance-instrument MR-GENIUS estimator.

    Uses SNP-on-exposure-variance effects ``v_gx`` as heteroskedastic cis
    instruments to weight LD-whitened summary residuals ``by - theta * bx``,
    eliminating bias from pleiotropic pathways orthogonal to exposure vQTLs.
    """
    validate_settings(settings)
    bx_arr = as_1d_float(bx, "bx", min_len=1)
    p = bx_arr.shape[0]
    sx_arr = as_1d_float(sx, "sx", min_len=p)
    by_arr = as_1d_float(by, "by", min_len=p)
    sy_arr = as_1d_float(sy, "sy", min_len=p)
    vgx_arr = as_1d_float(v_gx, "v_gx", min_len=p)
    if not (sx_arr.shape[0] == by_arr.shape[0] == sy_arr.shape[0] == vgx_arr.shape[0] == p):
        raise ValueError("bx, sx, by, sy, and v_gx must all have the same length.")
    if np.any(sx_arr <= 0.0) or np.any(sy_arr <= 0.0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")
    if se_vgx is not None:
        se_v_arr = as_1d_float(se_vgx, "se_vgx", min_len=p)
        if se_v_arr.shape[0] != p or np.any(se_v_arr <= 0.0):
            raise ValueError("se_vgx must have length p with strictly positive entries.")
    else:
        se_v_arr = None
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if np.all(vgx_arr == 0.0):
        raise ValueError("v_gx cannot be all zero.")

    R = prepare_ld_matrix(ld_R, p, settings=settings) if ld_R is not None else np.eye(p)

    # Empirical-Bayes attenuation shrinkage on noisy vQTL weights in default mode
    if settings == "default" and se_v_arr is not None and p >= 2:
        tau2 = max(
            float(np.mean(vgx_arr**2 - se_v_arr**2)),
            0.1 * float(np.mean(vgx_arr**2)),
            1e-12,
        )
        v_eff = (tau2 / (tau2 + se_v_arr**2)) * vgx_arr
    else:
        v_eff = vgx_arr.copy()

    # Project onto non-degenerate eigenmodes of R for stable LD whitening
    eigvals, eigvecs = linalg.eigh(0.5 * (R + R.T))
    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]
    pos = np.maximum(eigvals, 0.0)
    cum = np.cumsum(pos) / max(float(np.sum(pos)), 1e-15)
    k_cut = int(np.searchsorted(cum, 0.999, side="left")) + 1
    k_pos = int(np.sum(eigvals > 1e-5))
    k = max(1, min(p, k_cut, max(k_pos, 1)))
    U = eigvecs[:, :k]

    Sigma_X = (R * sx_arr[:, None]) * sx_arr[None, :]
    Sigma_Y = (R * sy_arr[:, None]) * sy_arr[None, :]
    t_bx = U.T @ bx_arr
    t_by = U.T @ by_arr
    t_v = U.T @ v_eff
    t_Sx = 0.5 * (U.T @ Sigma_X @ U + (U.T @ Sigma_X @ U).T)
    t_Sy = 0.5 * (U.T @ Sigma_Y @ U + (U.T @ Sigma_Y @ U).T)

    denom0 = float(t_v @ sym_solve(t_Sy, t_bx))
    if abs(denom0) <= 1e-15:
        raise ValueError(
            "Variance instrument v_gx is orthogonal to exposure associations bx; "
            "mr_genius_summary is not identified."
        )
    theta_curr = float((t_v @ sym_solve(t_Sy, t_by)) / denom0)

    n_steps = 1 if settings == "R_default" else 25
    for _ in range(n_steps):
        om = t_Sy + (theta_curr**2) * t_Sx
        denom_t = float(t_v @ sym_solve(om, t_bx))
        if abs(denom_t) <= 1e-15:
            break
        theta_next = float((t_v @ sym_solve(om, t_by)) / denom_t)
        if abs(theta_next - theta_curr) <= 1e-9 * (1.0 + abs(theta_curr)):
            theta_curr = theta_next
            break
        theta_curr = theta_next

    theta_hat = theta_curr
    om_final = t_Sy + (theta_hat**2) * t_Sx
    denom_final = float(t_v @ sym_solve(om_final, t_bx))
    w_vec = sym_solve(om_final, t_v)  # Omega^-1 t_v
    var_1 = float(t_v @ w_vec) / (denom_final**2)

    t_resid = t_by - theta_hat * t_bx
    if se_v_arr is not None:
        if settings == "default" and p >= 2:
            shrink = tau2 / (tau2 + se_v_arr**2)
            eff_se_v = shrink * se_v_arr
        else:
            eff_se_v = se_v_arr
        Sigma_V = (R * eff_se_v[:, None]) * eff_se_v[None, :]
        t_Sv = 0.5 * (U.T @ Sigma_V @ U + (U.T @ Sigma_V @ U).T)
        om_inv_r = sym_solve(om_final, t_resid)
        var_2 = float(om_inv_r @ t_Sv @ om_inv_r) / (denom_final**2)
        z_v = vgx_arr / se_v_arr
        Sigma_V_raw = (R * se_v_arr[:, None]) * se_v_arr[None, :]
        t_Sv_raw = 0.5 * (U.T @ Sigma_V_raw @ U + (U.T @ Sigma_V_raw @ U).T)
        t_v_raw = U.T @ vgx_arr
        wald_v = max(float(t_v_raw @ sym_solve(t_Sv_raw, t_v_raw)), 0.0)
        het_f_stat = wald_v / k
        het_f_pval = float(stats.chi2.sf(wald_v, df=k))
        _ = z_v
    else:
        var_2 = 0.0
        het_f_stat = float(np.mean((vgx_arr / sx_arr) ** 2))
        het_f_pval = np.nan

    j_df = max(k - 1, 0)
    j_stat = max(float(t_resid @ sym_solve(om_final, t_resid)), 0.0) if j_df > 0 else 0.0
    j_pval = float(stats.chi2.sf(j_stat, df=j_df)) if j_df > 0 else np.nan

    se_hat = float(np.sqrt(max(var_1 + var_2, 1e-18)))
    pval, ci_low, ci_high = wald_inference(theta_hat, se_hat, alpha=alpha, df=None)

    return MRGeniusResult(
        estimate=theta_hat,
        se=se_hat,
        pval=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_instruments=p,
        n_samples=None,
        het_f_stat=het_f_stat,
        het_f_pval=het_f_pval,
        j_stat=j_stat,
        j_df=j_df,
        j_pval=j_pval,
        mode="summary_vqtl",
        settings=settings,
    )
