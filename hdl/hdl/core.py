"""Core High-Definition Likelihood estimators: hdl_h2, hdl_rg, and hdl_local.

References
----------
Ning, Z., Pawitan, Y., & Shen, X. (2020). High-definition likelihood inference of
genetic correlation across human complex traits. Nature Genetics, 52(8), 859-864.
https://doi.org/10.1038/s41588-020-0653-y

Li, Y., Pawitan, Y., & Shen, X. (2023). An enhanced framework for local genetic
correlation analysis (HDL-L). Nature Genetics / Bioinformatics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .panel import (
    LDEigenBlock,
    LDEigenPanel,
    SettingsType,
    build_ld_eigen_panel,
    ensure_panel,
    split_block_vectors,
    validate_settings,
)


@dataclass
class HDLH2Result:
    """Result of single-trait High-Definition Likelihood heritability estimation."""

    h2: float
    se_h2: float
    p_h2: float
    ci_low: float
    ci_high: float
    fisher_se_h2: float
    intercept: float
    se_intercept: float | None
    loglik: float
    n_snps: int
    n_blocks: int
    n_eigenmodes: int
    se_source: str
    intercept_output: bool
    settings: SettingsType

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "h2": self.h2,
            "se_h2": self.se_h2,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_h2": self.p_h2,
            "fisher_se_h2": self.fisher_se_h2,
            "loglik": self.loglik,
            "n_snps": self.n_snps,
            "n_blocks": self.n_blocks,
            "n_eigenmodes": self.n_eigenmodes,
            "se_source": self.se_source,
            "settings": self.settings,
        }
        if self.intercept_output or self.settings == "default":
            out["intercept"] = self.intercept
            out["se_intercept"] = self.se_intercept
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        base = (
            f"HDL.h2 (settings={self.settings!r}): h2={self.h2:.4f} "
            f"(SE={self.se_h2:.4f} [{self.se_source}], p={self.p_h2:.3e}), "
            f"95% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}]"
        )
        if self.intercept_output or self.settings == "default":
            base += f", intercept={self.intercept:.4f}"
        return base


@dataclass
class HDLRgResult:
    """Result of bivariate High-Definition Likelihood genetic correlation estimation."""

    rg: float
    se_rg: float
    p_rg: float
    rg_ci_low: float
    rg_ci_high: float
    h1_sq: float
    se_h1: float
    p_h1: float
    h2_sq: float
    se_h2: float
    p_h2: float
    rho12: float
    se_rho12: float
    p_rho12: float
    fisher_se_rg: float
    fisher_se_h1: float
    fisher_se_h2: float
    fisher_se_rho12: float
    b1: float
    b2: float
    c12: float
    loglik: float
    n_snps: int
    n_blocks: int
    n_eigenmodes: int
    se_source: str
    intercept_output: bool
    settings: SettingsType

    @property
    def h1(self) -> float:
        return self.h1_sq

    @property
    def h2(self) -> float:
        return self.h2_sq

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "rg": self.rg,
            "se_rg": self.se_rg,
            "rg_ci_low": self.rg_ci_low,
            "rg_ci_high": self.rg_ci_high,
            "p_rg": self.p_rg,
            "h1_sq": self.h1_sq,
            "se_h1": self.se_h1,
            "p_h1": self.p_h1,
            "h2_sq": self.h2_sq,
            "se_h2": self.se_h2,
            "p_h2": self.p_h2,
            "rho12": self.rho12,
            "se_rho12": self.se_rho12,
            "p_rho12": self.p_rho12,
            "fisher_se_rg": self.fisher_se_rg,
            "loglik": self.loglik,
            "n_snps": self.n_snps,
            "n_blocks": self.n_blocks,
            "n_eigenmodes": self.n_eigenmodes,
            "se_source": self.se_source,
            "settings": self.settings,
        }
        if self.intercept_output or self.settings == "default":
            out["b1"] = self.b1
            out["b2"] = self.b2
            out["c12"] = self.c12
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"HDL.rg (settings={self.settings!r}): rg={self.rg:.4f} "
            f"(SE={self.se_rg:.4f} [{self.se_source}], p={self.p_rg:.3e}), "
            f"95% CI=[{self.rg_ci_low:.4f}, {self.rg_ci_high:.4f}], "
            f"h1^2={self.h1_sq:.4f} (SE={self.se_h1:.4f}), "
            f"h2^2={self.h2_sq:.4f} (SE={self.se_h2:.4f})"
        )


@dataclass
class HDLLocalResult:
    """Result of single-locus Local High-Definition Likelihood (HDL-L)."""

    rg_local: float
    se_rg_local: float
    p_lrt: float
    lrt_stat: float
    rg_ci_low: float
    rg_ci_high: float
    h1_local: float
    se_h1_local: float
    h2_local: float
    se_h2_local: float
    rho12_local: float
    se_rho12_local: float
    b1: float
    b2: float
    c12: float
    loglik_alt: float
    loglik_null: float
    n_snps: int
    n_eigenmodes: int
    settings: SettingsType

    def to_dict(self) -> dict[str, object]:
        return {
            "rg_local": self.rg_local,
            "se_rg_local": self.se_rg_local,
            "rg_ci_low": self.rg_ci_low,
            "rg_ci_high": self.rg_ci_high,
            "lrt_stat": self.lrt_stat,
            "p_lrt": self.p_lrt,
            "h1_local": self.h1_local,
            "se_h1_local": self.se_h1_local,
            "h2_local": self.h2_local,
            "se_h2_local": self.se_h2_local,
            "rho12_local": self.rho12_local,
            "se_rho12_local": self.se_rho12_local,
            "loglik_alt": self.loglik_alt,
            "loglik_null": self.loglik_null,
            "n_snps": self.n_snps,
            "n_eigenmodes": self.n_eigenmodes,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"HDL-L (settings={self.settings!r}): rg_loc={self.rg_local:.4f} "
            f"(SE={self.se_rg_local:.4f}, LRT={self.lrt_stat:.2f}, p_LRT={self.p_lrt:.3e}), "
            f"h1_loc^2={self.h1_local:.5f}, h2_loc^2={self.h2_local:.5f}"
        )


def _project_blocks(
    z_blocks: list[np.ndarray],
    panel: LDEigenPanel,
) -> list[np.ndarray]:
    """Compute scaled eigenmode projections ``u_tilde_{b,i} = (v_{b,i}^T z_b) / sqrt(lam_{b,i})``."""
    out: list[np.ndarray] = []
    for zb, blk in zip(z_blocks, panel.blocks):
        u_b = np.sum(blk.eigvecs * zb[:, None], axis=0)
        out.append(u_b / np.sqrt(blk.eigvals))
    return out


def _ldsc_moment_init_univ(
    u_sq: np.ndarray,
    lam: np.ndarray,
    N: float,
    M: float,
    estimate_intercept: bool,
    h2_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Weighted eigen-LDSC moment initializer ``E[u_tilde^2] = (N h2 / M) lam + b``."""
    scale = N / M
    if estimate_intercept and lam.size >= 2 and np.var(lam) > 1e-12:
        # Inverse-variance weight toward smaller eigenvalues where b is best identified
        w = 1.0 / np.square(1.0 + 0.5 * lam)
        w /= np.sum(w)
        lam_m = float(np.sum(w * lam))
        u_m = float(np.sum(w * u_sq))
        denom = float(np.sum(w * np.square(lam - lam_m)))
        slope = float(np.sum(w * (lam - lam_m) * (u_sq - u_m)) / max(denom, 1e-15))
        b_init = u_m - slope * lam_m
        h2_init = slope / max(scale, 1e-15)
    else:
        b_init = 1.0
        w = 1.0 / np.square(1.0 + 0.5 * lam)
        slope = float(np.sum(w * lam * (u_sq - 1.0)) / max(float(np.sum(w * lam * lam)), 1e-15))
        h2_init = slope / max(scale, 1e-15)

    h2_clamped = float(np.clip(h2_init, max(h2_bounds[0], 1e-5), min(h2_bounds[1] - 1e-5, 0.9)))
    b_clamped = (
        float(np.clip(b_init, max(b_bounds[0] + 0.05, 0.7), min(b_bounds[1] - 0.05, 2.5)))
        if estimate_intercept
        else 1.0
    )
    return h2_clamped, b_clamped


def _fit_univariate_modes(
    u_sq: np.ndarray,
    lam: np.ndarray,
    N: float,
    M: float,
    estimate_intercept: bool,
    h2_bounds: tuple[float, float],
    b_bounds: tuple[float, float] = (0.2, 5.0),
    init: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float | None]:
    """Maximize univariate HDL Gaussian log-likelihood in O(1)-scaled ``alpha = (N/M) h2``."""
    scale = N / M
    if init is None:
        h2_0, b_0 = _ldsc_moment_init_univ(
            u_sq, lam, N, M, estimate_intercept, h2_bounds, b_bounds
        )
    else:
        h2_0 = float(np.clip(init[0], h2_bounds[0], h2_bounds[1]))
        b_0 = float(np.clip(init[1], b_bounds[0], b_bounds[1])) if estimate_intercept else 1.0

    a_bounds = (scale * h2_bounds[0], scale * h2_bounds[1])
    a_0 = scale * h2_0

    if estimate_intercept:
        x0 = np.array([a_0, b_0], dtype=np.float64)
        bounds = [a_bounds, b_bounds]

        def obj_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            a_val, b_val = float(x[0]), float(x[1])
            v = a_val * lam + b_val
            inv_v = 1.0 / np.maximum(v, 1e-9)
            ratio = u_sq * inv_v
            nll = 0.5 * float(np.sum(np.log(np.maximum(v, 1e-9)) + ratio))
            diff = (1.0 - ratio) * inv_v
            return nll, np.array([0.5 * float(np.sum(lam * diff)), 0.5 * float(np.sum(diff))])

        opt = optimize.minimize(
            obj_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 150},
        )
        h2_hat = float(np.clip(opt.x[0] / scale, h2_bounds[0], h2_bounds[1]))
        b_hat = float(opt.x[1])
    else:
        x0 = np.array([a_0], dtype=np.float64)
        bounds = [a_bounds]

        def obj_grad_1d(x: np.ndarray) -> tuple[float, np.ndarray]:
            a_val = float(x[0])
            v = a_val * lam + 1.0
            inv_v = 1.0 / np.maximum(v, 1e-9)
            ratio = u_sq * inv_v
            nll = 0.5 * float(np.sum(np.log(np.maximum(v, 1e-9)) + ratio))
            g_a = 0.5 * float(np.sum(lam * (1.0 - ratio) * inv_v))
            return nll, np.array([g_a])

        opt = optimize.minimize(
            obj_grad_1d,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 150},
        )
        h2_hat = float(np.clip(opt.x[0] / scale, h2_bounds[0], h2_bounds[1]))
        b_hat = 1.0

    v_hat = (scale * h2_hat) * lam + b_hat
    const_ll = -0.5 * float(np.sum(np.log(2.0 * np.pi * lam)))
    loglik = const_ll - 0.5 * float(np.sum(np.log(v_hat) + u_sq / v_hat))

    inv_v2 = 1.0 / (v_hat * v_hat)
    i_hh = 0.5 * (scale * scale) * float(np.sum((lam * lam) * inv_v2))
    if estimate_intercept:
        i_hb = 0.5 * scale * float(np.sum(lam * inv_v2))
        i_bb = 0.5 * float(np.sum(inv_v2))
        fim = np.array([[i_hh, i_hb], [i_hb, i_bb]], dtype=np.float64)
        cov = linalg.pinv(fim)
        se_h2 = float(np.sqrt(max(cov[0, 0], 1e-18)))
        se_b: float | None = float(np.sqrt(max(cov[1, 1], 1e-18)))
    else:
        se_h2 = float(1.0 / np.sqrt(max(i_hh, 1e-18)))
        se_b = None

    return h2_hat, b_hat, loglik, se_h2, se_b


def _smooth_biv_nll_and_partials(
    v1: np.ndarray,
    v2: np.ndarray,
    v12: np.ndarray,
    u1_sq: np.ndarray,
    u2_sq: np.ndarray,
    u1_u2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate bivariate Gaussian NLL and exact partials wrt ``(v1, v2, v12)``.

    Inside the strict PSD cone ``|r_i| <= tau`` (``tau = 0.999``), evaluates the
    exact bivariate Gaussian NLL. Beyond ``|r_i| > tau`` during trial line-search
    steps, smoothly continues via a C1 quadratic barrier on ``r_i = v12 / sqrt(v1*v2)``
    so L-BFGS-B always receives an exact restore gradient back into the PSD cone.
    """
    tau = 0.999
    v1_safe = np.maximum(v1, 1e-8)
    v2_safe = np.maximum(v2, 1e-8)
    sq12 = np.sqrt(v1_safe * v2_safe)
    r_raw = v12 / sq12
    abs_r = np.abs(r_raw)

    inside = abs_r <= tau
    if np.all(inside):
        D = v1_safe * v2_safe - v12 * v12
        Q = (v2_safe * u1_sq - 2.0 * v12 * u1_u2 + v1_safe * u2_sq) / D
        nll = 0.5 * float(np.sum(np.log(D) + Q))
        gv1 = (v2_safe * (1.0 - Q) + u2_sq) / (2.0 * D)
        gv2 = (v1_safe * (1.0 - Q) + u1_sq) / (2.0 * D)
        gv12 = (v12 * (Q - 1.0) - u1_u2) / D
        return nll, gv1, gv2, gv12

    # Clamp effective correlation strictly below 0.9995 (< 1) for out-of-cone trial points
    span = 0.0005  # tau=0.999 -> max |r_eff| = 0.9995 => 1 - r_eff^2 >= 9.99e-4
    z_arg = np.clip((abs_r - tau) / span, 0.0, 20.0)
    th = np.tanh(z_arg)
    r_eff = np.where(inside, r_raw, np.sign(r_raw) * (tau + span * th))
    dr_eff_dr = np.where(inside, 1.0, 1.0 - th * th)
    v12_eff = r_eff * sq12
    D = np.maximum(v1_safe * v2_safe - v12_eff * v12_eff, 1e-12)
    Q = (v2_safe * u1_sq - 2.0 * v12_eff * u1_u2 + v1_safe * u2_sq) / D

    excess = np.where(inside, 0.0, abs_r - tau)
    pen = 500.0 * np.sum(excess * excess)
    nll = 0.5 * float(np.sum(np.log(D) + Q)) + float(pen)

    gv1_base = (v2_safe * (1.0 - Q) + u2_sq) / (2.0 * D)
    gv2_base = (v1_safe * (1.0 - Q) + u1_sq) / (2.0 * D)
    gv12_eff = (v12_eff * (Q - 1.0) - u1_u2) / D

    dpen_dr = np.where(inside, 0.0, 1000.0 * np.sign(r_raw) * excess)
    # Chain rule through v12_eff = r_eff(r_raw) * sq12 and pen(r_raw)
    dL_dr_raw = gv12_eff * sq12 * dr_eff_dr + dpen_dr
    gv12 = dL_dr_raw / sq12
    gv1 = (
        gv1_base
        + gv12_eff * r_eff * (0.5 * v2_safe / sq12)
        - dL_dr_raw * (0.5 * r_raw / v1_safe)
    )
    gv2 = (
        gv2_base
        + gv12_eff * r_eff * (0.5 * v1_safe / sq12)
        - dL_dr_raw * (0.5 * r_raw / v2_safe)
    )
    return nll, gv1, gv2, gv12


def _biv_loglik_total(
    u1_sq: np.ndarray,
    u2_sq: np.ndarray,
    u1_u2: np.ndarray,
    lam: np.ndarray,
    N1: float,
    N2: float,
    M: float,
    h1: float,
    b1: float,
    h2: float,
    b2: float,
    rho12: float,
    c12: float,
) -> float:
    """Compute exact bivariate Gaussian log-likelihood in original ``(u1, u2)`` coordinates."""
    v1 = (N1 * h1 / M) * lam + b1
    v2 = (N2 * h2 / M) * lam + b2
    v12 = (np.sqrt(N1 * N2) * rho12 / M) * lam + c12
    D = v1 * v2 - v12 * v12
    if np.any(D <= 1e-12):
        return -np.inf
    quad = (v2 * u1_sq - 2.0 * v12 * u1_u2 + v1 * u2_sq) / D
    const_ll = -float(np.sum(np.log(2.0 * np.pi * lam)))
    return const_ll - 0.5 * float(np.sum(np.log(D) + quad))


def _fit_bivariate_modes(
    u1_sq: np.ndarray,
    u2_sq: np.ndarray,
    u1_u2: np.ndarray,
    lam: np.ndarray,
    N1: float,
    N2: float,
    M: float,
    N0: float,
    rho_overlap: float,
    estimate_univ_intercepts: bool,
    estimate_c12: bool,
    joint_polish: bool,
    rg_cap: float,
    h2_bounds: tuple[float, float],
    b_bounds: tuple[float, float] = (0.2, 5.0),
    warm_start: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit bivariate HDL model in O(1)-scaled coordinates."""
    s1 = N1 / M
    s2 = N2 / M
    s12 = np.sqrt(N1 * N2) / M
    c_prior = (N0 / np.sqrt(N1 * N2)) * rho_overlap if (N0 > 0 and N1 > 0 and N2 > 0) else 0.0

    if warm_start is not None:
        h1_init, b1_init, h2_init, b2_init, rho_init, c_init = warm_start
        h1_m, b1_m = _fit_univariate_modes(
            u1_sq, lam, N1, M, estimate_univ_intercepts, h2_bounds, b_bounds, init=(h1_init, b1_init)
        )[:2]
        h2_m, b2_m = _fit_univariate_modes(
            u2_sq, lam, N2, M, estimate_univ_intercepts, h2_bounds, b_bounds, init=(h2_init, b2_init)
        )[:2]
    else:
        h1_m, b1_m = _fit_univariate_modes(
            u1_sq, lam, N1, M, estimate_univ_intercepts, h2_bounds, b_bounds
        )[:2]
        h2_m, b2_m = _fit_univariate_modes(
            u2_sq, lam, N2, M, estimate_univ_intercepts, h2_bounds, b_bounds
        )[:2]
        # Precision-weighted moment initializer for (rho12, c12)
        v1_w0 = (s1 * h1_m) * lam + b1_m
        v2_w0 = (s2 * h2_m) * lam + b2_m
        wt = 1.0 / (v1_w0 * v2_w0)
        wt /= np.sum(wt)
        if estimate_c12 and lam.size >= 2 and np.var(lam) > 1e-12:
            lm = float(np.sum(wt * lam))
            um = float(np.sum(wt * u1_u2))
            slp = float(np.sum(wt * (lam - lm) * (u1_u2 - um)) / max(float(np.sum(wt * (lam - lm) ** 2)), 1e-15))
            c_init = float(np.clip(um - slp * lm, -0.8 * np.sqrt(b1_m * b2_m), 0.8 * np.sqrt(b1_m * b2_m)))
            rho_init = slp / max(s12, 1e-15)
        else:
            c_init = c_prior
            slp = float(np.sum(wt * lam * (u1_u2 - c_init)) / max(float(np.sum(wt * lam * lam)), 1e-15))
            rho_init = slp / max(s12, 1e-15)

    v1_0 = (s1 * h1_m) * lam + b1_m
    v2_0 = (s2 * h2_m) * lam + b2_m
    sq_h_m = np.sqrt(max(h1_m * h2_m, 1e-16))
    sq_b_m = np.sqrt(max(b1_m * b2_m, 1e-12))

    # Stage 2: Optimize dimensionless (rg, c12) given marginal (h1_m, b1_m, h2_m, b2_m)
    rg_0 = float(np.clip(rho_init / sq_h_m, -min(0.85, rg_cap), min(0.85, rg_cap)))
    c_bound = 0.98 * sq_b_m
    c_0 = float(np.clip(c_init, -0.85 * sq_b_m, 0.85 * sq_b_m)) if estimate_c12 else c_prior

    if estimate_c12:
        x0_cov = np.array([rg_0, c_0], dtype=np.float64)
        bounds_cov = [(-rg_cap, rg_cap), (-c_bound, c_bound)]

        def cov_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
            rg_v, c_v = float(x[0]), float(x[1])
            v12 = (s12 * rg_v * sq_h_m) * lam + c_v
            nll, _, _, gv12 = _smooth_biv_nll_and_partials(
                v1_0, v2_0, v12, u1_sq, u2_sq, u1_u2
            )
            return nll, np.array(
                [s12 * sq_h_m * float(np.sum(lam * gv12)), float(np.sum(gv12))]
            )

        opt_cov = optimize.minimize(
            cov_obj,
            x0_cov,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_cov,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 150},
        )
        rg_stage2, c_hat = float(opt_cov.x[0]), float(opt_cov.x[1])
    else:
        x0_cov = np.array([rg_0], dtype=np.float64)
        bounds_cov = [(-rg_cap, rg_cap)]

        def cov_obj_1d(x: np.ndarray) -> tuple[float, np.ndarray]:
            rg_v = float(x[0])
            v12 = (s12 * rg_v * sq_h_m) * lam + c_prior
            nll, _, _, gv12 = _smooth_biv_nll_and_partials(
                v1_0, v2_0, v12, u1_sq, u2_sq, u1_u2
            )
            return nll, np.array([s12 * sq_h_m * float(np.sum(lam * gv12))])

        opt_cov = optimize.minimize(
            cov_obj_1d,
            x0_cov,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_cov,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 150},
        )
        rg_stage2, c_hat = float(opt_cov.x[0]), c_prior

    h1_hat, b1_hat, h2_hat, b2_hat = h1_m, b1_m, h2_m, b2_m
    rho_hat = rg_stage2 * sq_h_m

    # Stage 3 (default mode): Joint multivariate Gaussian MLE polish in O(1)-scaled coordinates
    if joint_polish and estimate_univ_intercepts and estimate_c12:
        a1_bounds = (s1 * h2_bounds[0], s1 * h2_bounds[1])
        a2_bounds = (s2 * h2_bounds[0], s2 * h2_bounds[1])
        r0_start = float(np.clip(c_hat / np.sqrt(b1_hat * b2_hat), -0.98, 0.98))
        x0_j = np.array(
            [s1 * h1_hat, b1_hat, s2 * h2_hat, b2_hat, rg_stage2, r0_start],
            dtype=np.float64,
        )
        bounds_j = [
            a1_bounds,
            b_bounds,
            a2_bounds,
            b_bounds,
            (-rg_cap, rg_cap),
            (-0.985, 0.985),
        ]

        def joint_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
            a1, b1v, a2, b2v, rgv, r0v = [float(val) for val in x]
            sq_a = np.sqrt(max(a1 * a2, 1e-18))
            sq_b = np.sqrt(max(b1v * b2v, 1e-18))
            v1 = a1 * lam + b1v
            v2 = a2 * lam + b2v
            v12 = (rgv * sq_a) * lam + (r0v * sq_b)
            nll, gv1, gv2, gv12 = _smooth_biv_nll_and_partials(
                v1, v2, v12, u1_sq, u2_sq, u1_u2
            )
            sum_lam_gv12 = float(np.sum(lam * gv12))
            sum_gv12 = float(np.sum(gv12))
            g_a1 = float(np.sum(lam * gv1)) + sum_lam_gv12 * (0.5 * rgv * a2 / sq_a)
            g_b1 = float(np.sum(gv1)) + sum_gv12 * (0.5 * r0v * b2v / sq_b)
            g_a2 = float(np.sum(lam * gv2)) + sum_lam_gv12 * (0.5 * rgv * a1 / sq_a)
            g_b2 = float(np.sum(gv2)) + sum_gv12 * (0.5 * r0v * b1v / sq_b)
            g_rg = sq_a * sum_lam_gv12
            g_r0 = sq_b * sum_gv12
            return nll, np.array([g_a1, g_b1, g_a2, g_b2, g_rg, g_r0], dtype=np.float64)

        ll_before = _biv_loglik_total(
            u1_sq, u2_sq, u1_u2, lam, N1, N2, M, h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c_hat
        )
        opt_j = optimize.minimize(
            joint_obj,
            x0_j,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_j,
            options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 150},
        )
        a1_j, b1_j, a2_j, b2_j, rg_j, r0_j = [float(val) for val in opt_j.x]
        h1_j = float(np.clip(a1_j / s1, h2_bounds[0], h2_bounds[1]))
        h2_j = float(np.clip(a2_j / s2, h2_bounds[0], h2_bounds[1]))
        rho_j = rg_j * np.sqrt(max(h1_j * h2_j, 1e-18))
        c_j = r0_j * np.sqrt(max(b1_j * b2_j, 1e-18))
        ll_after = _biv_loglik_total(
            u1_sq, u2_sq, u1_u2, lam, N1, N2, M, h1_j, b1_j, h2_j, b2_j, rho_j, c_j
        )
        if np.isfinite(ll_after) and (not np.isfinite(ll_before) or ll_after >= ll_before - 1e-10):
            h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c_hat = (
                h1_j,
                b1_j,
                h2_j,
                b2_j,
                rho_j,
                c_j,
            )

    ll_final = _biv_loglik_total(
        u1_sq, u2_sq, u1_u2, lam, N1, N2, M, h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c_hat
    )
    return h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c_hat, ll_final


def _bivariate_fisher_ses(
    lam: np.ndarray,
    N1: float,
    N2: float,
    M: float,
    h1: float,
    b1: float,
    h2: float,
    b2: float,
    rho12: float,
    c12: float,
    estimate_univ_intercepts: bool,
    estimate_c12: bool,
) -> tuple[float, float, float, float]:
    """Compute analytical Fisher-information SEs for ``(h1^2, h2^2, rho12, rg)``."""
    s1 = N1 / M
    s2 = N2 / M
    s12 = np.sqrt(N1 * N2) / M
    v1 = (s1 * h1) * lam + b1
    v2 = (s2 * h2) * lam + b2
    v12 = (s12 * rho12) * lam + c12
    D = np.maximum(v1 * v2 - v12 * v12, 1e-12)

    p11 = v2 / D
    p22 = v1 / D
    p12 = -v12 / D

    iv11 = 0.5 * p11 * p11
    iv22 = 0.5 * p22 * p22
    iv12_cross = 0.5 * p12 * p12
    iv1_c = p11 * p12
    iv2_c = p22 * p12
    icc = p11 * p22 + p12 * p12

    lam2 = lam * lam
    F = np.zeros((6, 6), dtype=np.float64)
    F[0, 0] = s1 * s1 * float(np.sum(lam2 * iv11))
    F[0, 3] = F[3, 0] = s1 * float(np.sum(lam * iv11))
    F[3, 3] = float(np.sum(iv11))

    F[1, 1] = s2 * s2 * float(np.sum(lam2 * iv22))
    F[1, 4] = F[4, 1] = s2 * float(np.sum(lam * iv22))
    F[4, 4] = float(np.sum(iv22))

    F[2, 2] = s12 * s12 * float(np.sum(lam2 * icc))
    F[2, 5] = F[5, 2] = s12 * float(np.sum(lam * icc))
    F[5, 5] = float(np.sum(icc))

    F[0, 1] = F[1, 0] = s1 * s2 * float(np.sum(lam2 * iv12_cross))
    F[0, 4] = F[4, 0] = s1 * float(np.sum(lam * iv12_cross))
    F[3, 1] = F[1, 3] = s2 * float(np.sum(lam * iv12_cross))
    F[3, 4] = F[4, 3] = float(np.sum(iv12_cross))

    F[0, 2] = F[2, 0] = s1 * s12 * float(np.sum(lam2 * iv1_c))
    F[0, 5] = F[5, 0] = s1 * float(np.sum(lam * iv1_c))
    F[3, 2] = F[2, 3] = s12 * float(np.sum(lam * iv1_c))
    F[3, 5] = F[5, 3] = float(np.sum(iv1_c))

    F[1, 2] = F[2, 1] = s2 * s12 * float(np.sum(lam2 * iv2_c))
    F[1, 5] = F[5, 1] = s2 * float(np.sum(lam * iv2_c))
    F[4, 2] = F[2, 4] = s12 * float(np.sum(lam * iv2_c))
    F[4, 5] = F[5, 4] = float(np.sum(iv2_c))

    active = [0, 1, 2]
    if estimate_univ_intercepts:
        active.extend([3, 4])
    if estimate_c12:
        active.append(5)

    F_act = F[np.ix_(active, active)]
    cov_act = linalg.pinv(F_act)
    cov_3 = cov_act[:3, :3]

    se_h1 = float(np.sqrt(max(cov_3[0, 0], 1e-18)))
    se_h2 = float(np.sqrt(max(cov_3[1, 1], 1e-18)))
    se_rho = float(np.sqrt(max(cov_3[2, 2], 1e-18)))

    denom_rg = np.sqrt(max(h1 * h2, 1e-18))
    rg_val = rho12 / denom_rg
    grad_rg = np.array(
        [
            -0.5 * rg_val / max(h1, 1e-12),
            -0.5 * rg_val / max(h2, 1e-12),
            1.0 / denom_rg,
        ],
        dtype=np.float64,
    )
    var_rg = float(grad_rg @ cov_3 @ grad_rg)
    se_rg = float(np.sqrt(max(var_rg, 1e-18)))
    return se_h1, se_h2, se_rho, se_rg


def hdl_h2(
    z: np.ndarray | Sequence[np.ndarray] | Sequence[float],
    ld_panels: LDEigenPanel | Sequence[LDEigenBlock] | Sequence[np.ndarray] | np.ndarray,
    N: int | float,
    intercept_output: bool = False,
    jackknife: bool = True,
    settings: SettingsType = "default",
) -> HDLH2Result:
    """Single-trait High-Definition Likelihood heritability estimation (HDL)."""
    validate_settings(settings)
    if N <= 2 or not np.isfinite(N):
        raise ValueError(f"Sample size N must be finite and > 2, got {N}.")

    panel = ensure_panel(ld_panels, settings=settings)
    z_blocks = split_block_vectors(z, panel, name="z")
    u_blocks = _project_blocks(z_blocks, panel)
    u_sq_blocks = [ub * ub for ub in u_blocks]
    lam_blocks = [blk.eigvals for blk in panel.blocks]

    u_sq_all = np.concatenate(u_sq_blocks)
    lam_all = np.concatenate(lam_blocks)
    M = float(panel.total_snps)

    est_intercept = True if settings == "default" else bool(intercept_output)
    h2_bounds = (1e-8, 1.5) if settings == "default" else (1e-6, 1.0 - 1e-6)

    h2_hat, b_hat, loglik, fisher_se_h2, fisher_se_b = _fit_univariate_modes(
        u_sq_all, lam_all, float(N), M, est_intercept, h2_bounds
    )

    B = panel.n_blocks
    if jackknife and B >= 2:
        jk_h2 = np.empty(B, dtype=np.float64)
        jk_b = np.empty(B, dtype=np.float64)
        for b_idx in range(B):
            u_sq_sub = np.concatenate([u_sq_blocks[j] for j in range(B) if j != b_idx])
            lam_sub = np.concatenate([lam_blocks[j] for j in range(B) if j != b_idx])
            m_sub = M - float(panel.blocks[b_idx].n_snps)
            jk_h2[b_idx], jk_b[b_idx], *_ = _fit_univariate_modes(
                u_sq_sub,
                lam_sub,
                float(N),
                m_sub,
                est_intercept,
                h2_bounds,
                init=(h2_hat, b_hat),
            )
        factor = (B - 1.0) / float(B)
        se_h2 = float(np.sqrt(max(factor * float(np.sum((jk_h2 - np.mean(jk_h2)) ** 2)), 1e-18)))
        se_b = (
            float(np.sqrt(max(factor * float(np.sum((jk_b - np.mean(jk_b)) ** 2)), 1e-18)))
            if est_intercept
            else None
        )
        se_source = "jackknife"
    else:
        se_h2 = fisher_se_h2
        se_b = fisher_se_b
        se_source = "fisher"

    p_h2 = float(2.0 * stats.norm.sf(abs(h2_hat) / se_h2))
    crit = float(stats.norm.ppf(0.975))
    return HDLH2Result(
        h2=h2_hat,
        se_h2=se_h2,
        p_h2=p_h2,
        ci_low=float(h2_hat - crit * se_h2),
        ci_high=float(h2_hat + crit * se_h2),
        fisher_se_h2=fisher_se_h2,
        intercept=b_hat,
        se_intercept=se_b,
        loglik=loglik,
        n_snps=panel.total_snps,
        n_blocks=B,
        n_eigenmodes=panel.total_eigenmodes,
        se_source=se_source,
        intercept_output=intercept_output,
        settings=settings,
    )


def hdl_rg(
    z1: np.ndarray | Sequence[np.ndarray] | Sequence[float],
    z2: np.ndarray | Sequence[np.ndarray] | Sequence[float],
    ld_panels: LDEigenPanel | Sequence[LDEigenBlock] | Sequence[np.ndarray] | np.ndarray,
    N1: int | float,
    N2: int | float,
    N0: int | float = 0,
    rho_overlap: float = 0.0,
    intercept_output: bool = False,
    jackknife: bool = True,
    settings: SettingsType = "default",
) -> HDLRgResult:
    """Bivariate High-Definition Likelihood genetic correlation estimation (HDL.rg)."""
    validate_settings(settings)
    if N1 <= 2 or N2 <= 2 or not (np.isfinite(N1) and np.isfinite(N2)):
        raise ValueError(f"Sample sizes N1 and N2 must be finite and > 2, got {N1}, {N2}.")
    if N0 < 0:
        raise ValueError(f"Overlapping sample size N0 must be >= 0, got {N0}.")

    panel = ensure_panel(ld_panels, settings=settings)
    z1_blocks = split_block_vectors(z1, panel, name="z1")
    z2_blocks = split_block_vectors(z2, panel, name="z2")
    u1_blocks = _project_blocks(z1_blocks, panel)
    u2_blocks = _project_blocks(z2_blocks, panel)

    u1_sq_blocks = [u * u for u in u1_blocks]
    u2_sq_blocks = [u * u for u in u2_blocks]
    u12_blocks = [u1 * u2 for u1, u2 in zip(u1_blocks, u2_blocks)]
    lam_blocks = [blk.eigvals for blk in panel.blocks]

    u1_sq_all = np.concatenate(u1_sq_blocks)
    u2_sq_all = np.concatenate(u2_sq_blocks)
    u12_all = np.concatenate(u12_blocks)
    lam_all = np.concatenate(lam_blocks)
    M = float(panel.total_snps)

    est_univ_int = True if settings == "default" else bool(intercept_output)
    joint_polish = settings == "default"
    rg_cap = 1.5 if settings == "default" else 2.0
    h2_bounds = (1e-8, 1.5) if settings == "default" else (1e-6, 1.0 - 1e-6)

    h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c12_hat, loglik = _fit_bivariate_modes(
        u1_sq_all,
        u2_sq_all,
        u12_all,
        lam_all,
        float(N1),
        float(N2),
        M,
        float(N0),
        float(rho_overlap),
        estimate_univ_intercepts=est_univ_int,
        estimate_c12=True,
        joint_polish=joint_polish,
        rg_cap=rg_cap,
        h2_bounds=h2_bounds,
    )
    rg_hat = float(rho_hat / np.sqrt(max(h1_hat * h2_hat, 1e-18)))
    if settings == "default":
        rg_hat = float(np.clip(rg_hat, -1.5, 1.5))

    fisher_se_h1, fisher_se_h2, fisher_se_rho, fisher_se_rg = _bivariate_fisher_ses(
        lam_all,
        float(N1),
        float(N2),
        M,
        h1_hat,
        b1_hat,
        h2_hat,
        b2_hat,
        rho_hat,
        c12_hat,
        estimate_univ_intercepts=est_univ_int,
        estimate_c12=True,
    )

    B = panel.n_blocks
    if jackknife and B >= 2:
        jk_h1 = np.empty(B, dtype=np.float64)
        jk_h2 = np.empty(B, dtype=np.float64)
        jk_rho = np.empty(B, dtype=np.float64)
        jk_rg = np.empty(B, dtype=np.float64)
        warm = (h1_hat, b1_hat, h2_hat, b2_hat, rho_hat, c12_hat)
        for b_idx in range(B):
            u1_sub = np.concatenate([u1_sq_blocks[j] for j in range(B) if j != b_idx])
            u2_sub = np.concatenate([u2_sq_blocks[j] for j in range(B) if j != b_idx])
            u12_sub = np.concatenate([u12_blocks[j] for j in range(B) if j != b_idx])
            lam_sub = np.concatenate([lam_blocks[j] for j in range(B) if j != b_idx])
            m_sub = M - float(panel.blocks[b_idx].n_snps)

            h1_b, _, h2_b, _, rho_b, _, _ = _fit_bivariate_modes(
                u1_sub,
                u2_sub,
                u12_sub,
                lam_sub,
                float(N1),
                float(N2),
                m_sub,
                float(N0),
                float(rho_overlap),
                estimate_univ_intercepts=est_univ_int,
                estimate_c12=True,
                joint_polish=joint_polish,
                rg_cap=rg_cap,
                h2_bounds=h2_bounds,
                warm_start=warm,
            )
            jk_h1[b_idx] = h1_b
            jk_h2[b_idx] = h2_b
            jk_rho[b_idx] = rho_b
            rg_b = rho_b / np.sqrt(max(h1_b * h2_b, 1e-18))
            jk_rg[b_idx] = np.clip(rg_b, -1.5, 1.5) if settings == "default" else rg_b

        factor = (B - 1.0) / float(B)
        se_h1 = float(np.sqrt(max(factor * float(np.sum((jk_h1 - np.mean(jk_h1)) ** 2)), 1e-18)))
        se_h2 = float(np.sqrt(max(factor * float(np.sum((jk_h2 - np.mean(jk_h2)) ** 2)), 1e-18)))
        se_rho = float(np.sqrt(max(factor * float(np.sum((jk_rho - np.mean(jk_rho)) ** 2)), 1e-18)))
        se_rg = float(np.sqrt(max(factor * float(np.sum((jk_rg - np.mean(jk_rg)) ** 2)), 1e-18)))
        se_source = "jackknife"
    else:
        se_h1 = fisher_se_h1
        se_h2 = fisher_se_h2
        se_rho = fisher_se_rho
        se_rg = fisher_se_rg
        se_source = "fisher"

    p_rg = float(2.0 * stats.norm.sf(abs(rg_hat) / se_rg))
    p_h1 = float(2.0 * stats.norm.sf(abs(h1_hat) / se_h1))
    p_h2 = float(2.0 * stats.norm.sf(abs(h2_hat) / se_h2))
    p_rho = float(2.0 * stats.norm.sf(abs(rho_hat) / se_rho))
    crit = float(stats.norm.ppf(0.975))

    return HDLRgResult(
        rg=rg_hat,
        se_rg=se_rg,
        p_rg=p_rg,
        rg_ci_low=float(rg_hat - crit * se_rg),
        rg_ci_high=float(rg_hat + crit * se_rg),
        h1_sq=h1_hat,
        se_h1=se_h1,
        p_h1=p_h1,
        h2_sq=h2_hat,
        se_h2=se_h2,
        p_h2=p_h2,
        rho12=rho_hat,
        se_rho12=se_rho,
        p_rho12=p_rho,
        fisher_se_rg=fisher_se_rg,
        fisher_se_h1=fisher_se_h1,
        fisher_se_h2=fisher_se_h2,
        fisher_se_rho12=fisher_se_rho,
        b1=b1_hat,
        b2=b2_hat,
        c12=c12_hat,
        loglik=loglik,
        n_snps=panel.total_snps,
        n_blocks=B,
        n_eigenmodes=panel.total_eigenmodes,
        se_source=se_source,
        intercept_output=intercept_output,
        settings=settings,
    )


def hdl_local(
    z1: np.ndarray | Sequence[float],
    z2: np.ndarray | Sequence[float],
    R_locus: np.ndarray,
    N1: int | float,
    N2: int | float,
    eigen_cut: float = 0.99,
    N0: int | float = 0,
    rho_overlap: float = 0.0,
    settings: SettingsType = "default",
) -> HDLLocalResult:
    """Single-locus Local High-Definition Likelihood (HDL-L) heritability & correlation."""
    validate_settings(settings)
    if N1 <= 2 or N2 <= 2:
        raise ValueError(f"Sample sizes N1 and N2 must be > 2, got {N1}, {N2}.")
    R_mat = np.asarray(R_locus, dtype=np.float64)
    if R_mat.ndim != 2 or R_mat.shape[0] != R_mat.shape[1] or R_mat.shape[0] < 2:
        raise ValueError(f"R_locus must be a square 2D matrix of size >= 2, got {R_mat.shape}.")

    panel = build_ld_eigen_panel([R_mat], eigen_cut=eigen_cut, settings=settings)
    z1_blocks = split_block_vectors(z1, panel, name="z1")
    z2_blocks = split_block_vectors(z2, panel, name="z2")
    u1 = _project_blocks(z1_blocks, panel)[0]
    u2 = _project_blocks(z2_blocks, panel)[0]
    u1_sq = u1 * u1
    u2_sq = u2 * u2
    u12 = u1 * u2
    lam = panel.blocks[0].eigvals
    M_loc = float(panel.total_snps)

    est_int = settings == "default" and lam.size >= 15
    h2_bounds = (1e-9, 1.0) if settings == "default" else (1e-8, 1.0)
    b_bounds = (0.6, 1.8)
    rg_cap = 1.5 if settings == "default" else 2.0

    h1_alt, b1_alt, h2_alt, b2_alt, rho_alt, c12_alt, ll_alt = _fit_bivariate_modes(
        u1_sq,
        u2_sq,
        u12,
        lam,
        float(N1),
        float(N2),
        M_loc,
        float(N0),
        float(rho_overlap),
        estimate_univ_intercepts=est_int,
        estimate_c12=est_int,
        joint_polish=(settings == "default"),
        rg_cap=rg_cap,
        h2_bounds=h2_bounds,
        b_bounds=b_bounds,
    )

    s1 = float(N1) / M_loc
    s2 = float(N2) / M_loc
    c_fixed = (float(N0) / np.sqrt(float(N1 * N2))) * float(rho_overlap) if N0 > 0 else 0.0

    if not est_int:
        h1_0, *_ = _fit_univariate_modes(u1_sq, lam, float(N1), M_loc, False, h2_bounds, b_bounds)
        h2_0, *_ = _fit_univariate_modes(u2_sq, lam, float(N2), M_loc, False, h2_bounds, b_bounds)
        if c_fixed == 0.0:
            ll_null = _biv_loglik_total(
                u1_sq, u2_sq, u12, lam, float(N1), float(N2), M_loc, h1_0, 1.0, h2_0, 1.0, 0.0, 0.0
            )
        else:
            x0_nf = np.array([s1 * h1_0, s2 * h2_0], dtype=np.float64)

            def null_fixed_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
                a1v, a2v = float(x[0]), float(x[1])
                v1 = a1v * lam + 1.0
                v2 = a2v * lam + 1.0
                nll, gv1, gv2, _ = _smooth_biv_nll_and_partials(
                    v1, v2, np.full_like(lam, c_fixed), u1_sq, u2_sq, u12
                )
                return nll, np.array([float(np.sum(lam * gv1)), float(np.sum(lam * gv2))])

            opt_n = optimize.minimize(
                null_fixed_obj,
                x0_nf,
                method="L-BFGS-B",
                jac=True,
                bounds=[(s1 * h2_bounds[0], s1 * h2_bounds[1]), (s2 * h2_bounds[0], s2 * h2_bounds[1])],
            )
            ll_null = _biv_loglik_total(
                u1_sq,
                u2_sq,
                u12,
                lam,
                float(N1),
                float(N2),
                M_loc,
                float(opt_n.x[0]) / s1,
                1.0,
                float(opt_n.x[1]) / s2,
                1.0,
                0.0,
                c_fixed,
            )
    else:
        r0_init = float(np.clip(c12_alt / np.sqrt(b1_alt * b2_alt), -0.85, 0.85))
        x0_null = np.array([s1 * h1_alt, b1_alt, s2 * h2_alt, b2_alt, r0_init], dtype=np.float64)
        bounds_null = [
            (s1 * h2_bounds[0], s1 * h2_bounds[1]),
            b_bounds,
            (s2 * h2_bounds[0], s2 * h2_bounds[1]),
            b_bounds,
            (-0.985, 0.985),
        ]

        def null_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
            a1v, b1v, a2v, b2v, r0v = [float(val) for val in x]
            sq_b = np.sqrt(max(b1v * b2v, 1e-18))
            v1 = a1v * lam + b1v
            v2 = a2v * lam + b2v
            v12 = np.full_like(lam, r0v * sq_b)
            nll, gv1, gv2, gv12 = _smooth_biv_nll_and_partials(
                v1, v2, v12, u1_sq, u2_sq, u12
            )
            sum_gv12 = float(np.sum(gv12))
            return nll, np.array(
                [
                    float(np.sum(lam * gv1)),
                    float(np.sum(gv1)) + sum_gv12 * (0.5 * r0v * b2v / sq_b),
                    float(np.sum(lam * gv2)),
                    float(np.sum(gv2)) + sum_gv12 * (0.5 * r0v * b1v / sq_b),
                    sq_b * sum_gv12,
                ],
                dtype=np.float64,
            )

        opt_null = optimize.minimize(
            null_obj,
            x0_null,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_null,
            options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 150},
        )
        a1_n, b1_n, a2_n, b2_n, r0_n = [float(val) for val in opt_null.x]
        c_n = r0_n * np.sqrt(max(b1_n * b2_n, 1e-18))
        ll_null = _biv_loglik_total(
            u1_sq, u2_sq, u12, lam, float(N1), float(N2), M_loc, a1_n / s1, b1_n, a2_n / s2, b2_n, 0.0, c_n
        )

    if ll_alt < ll_null:
        ll_alt = ll_null
        rho_alt = 0.0

    lrt_stat = max(2.0 * (ll_alt - ll_null), 0.0)
    p_lrt = float(stats.chi2.sf(lrt_stat, df=1))

    rg_loc = float(rho_alt / np.sqrt(max(h1_alt * h2_alt, 1e-18)))
    if settings == "default":
        rg_loc = float(np.clip(rg_loc, -1.5, 1.5))

    se_h1, se_h2, se_rho, se_rg = _bivariate_fisher_ses(
        lam,
        float(N1),
        float(N2),
        M_loc,
        h1_alt,
        b1_alt,
        h2_alt,
        b2_alt,
        rho_alt,
        c12_alt,
        estimate_univ_intercepts=est_int,
        estimate_c12=est_int,
    )
    crit = float(stats.norm.ppf(0.975))

    return HDLLocalResult(
        rg_local=rg_loc,
        se_rg_local=se_rg,
        p_lrt=p_lrt,
        lrt_stat=lrt_stat,
        rg_ci_low=float(rg_loc - crit * se_rg),
        rg_ci_high=float(rg_loc + crit * se_rg),
        h1_local=h1_alt,
        se_h1_local=se_h1,
        h2_local=h2_alt,
        se_h2_local=se_h2,
        rho12_local=rho_alt,
        se_rho12_local=se_rho,
        b1=b1_alt,
        b2=b2_alt,
        c12=c12_alt,
        loglik_alt=ll_alt,
        loglik_null=ll_null,
        n_snps=panel.total_snps,
        n_eigenmodes=panel.total_eigenmodes,
        settings=settings,
    )
