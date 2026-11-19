"""Core Stage-2 optimization, SE estimation, and nested LRTs for LHC-MR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, logit
from scipy.stats import qmc

from .ldsc import (
    BivariateLDSCResult,
    SettingsMode,
    bivariate_ldsc,
    validate_settings,
)
from .likelihood import (
    LHCData,
    block_component_sensitivities,
    canonicalize_reflection,
    loglik_only,
    neg_loglik_and_grad,
    prepare_lhc_data,
    summary_jacobian,
)

PARAM_NAMES: tuple[str, ...] = (
    "alpha_xy",
    "alpha_yx",
    "t_x",
    "t_y",
    "h2_x",
    "h2_y",
    "pi_x",
    "pi_y",
    "pi_u",
)


def _softplus(x: float | np.ndarray) -> float | np.ndarray:
    return np.logaddexp(0.0, x)


def _inv_softplus(y: float) -> float:
    y_pos = max(float(y), 1e-8)
    if y_pos > 20.0:
        return y_pos
    return float(np.log(np.expm1(y_pos)))


@dataclass(frozen=True)
class LRTResult:
    """Nested Likelihood Ratio Test comparison between two LHC-MR models."""

    null_model: str
    alt_model: str
    ll_null: float
    ll_alt: float
    stat: float
    df: int
    p_value: float


@dataclass(frozen=True)
class LHCMRResult:
    """Fitted LHC-MR structural causal and heritable-confounder estimates."""

    alpha_xy: float
    alpha_xy_se: float
    alpha_xy_z: float
    alpha_xy_pval: float
    alpha_yx: float
    alpha_yx_se: float
    alpha_yx_z: float
    alpha_yx_pval: float
    t_x: float
    t_x_se: float
    t_x_z: float
    t_x_pval: float
    t_y: float
    t_y_se: float
    t_y_z: float
    t_y_pval: float
    h2_x: float
    h2_y: float
    pi_x: float
    pi_y: float
    pi_u: float
    i_x: float
    i_y: float
    i_xy: float
    params: dict[str, float]
    se: dict[str, float]
    jackknife_se: dict[str, float]
    hessian_se: dict[str, float]
    z_scores: dict[str, float]
    p_values: dict[str, float]
    log_likelihood: float
    converged: bool
    n_snps: int
    m_snps: float
    stage1_ldsc: BivariateLDSCResult
    reflection_swapped: bool
    reflection_ll_diff: float
    submodel_logliks: dict[str, float]
    lrt: dict[str, LRTResult]
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return parameter estimates, block-jackknife & Hessian SEs, Z-scores, and p-values."""
        rows = []
        for name in PARAM_NAMES:
            rows.append(
                {
                    "parameter": name,
                    "estimate": self.params[name],
                    "se": self.se[name],
                    "jackknife_se": self.jackknife_se[name],
                    "hessian_se": self.hessian_se[name],
                    "z_score": self.z_scores[name],
                    "p_value": self.p_values[name],
                }
            )
        return pd.DataFrame(rows)

    def lrt_dataframe(self) -> pd.DataFrame:
        """Return nested Likelihood Ratio Test table."""
        rows = []
        for key, item in self.lrt.items():
            rows.append(
                {
                    "comparison": key,
                    "null_model": item.null_model,
                    "alt_model": item.alt_model,
                    "ll_null": item.ll_null,
                    "ll_alt": item.ll_alt,
                    "stat": item.stat,
                    "df": item.df,
                    "p_value": item.p_value,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary of LHC-MR results."""
        lines = [
            f"LHC-MR Summary (settings={self.settings!r}, M_used={self.n_snps}, logLik={self.log_likelihood:.3f})",
            f"  X -> Y (alpha_xy): {self.alpha_xy:+.4f} (SE={self.alpha_xy_se:.4f}, Z={self.alpha_xy_z:+.2f}, p={self.alpha_xy_pval:.3e})",
            f"  Y -> X (alpha_yx): {self.alpha_yx:+.4f} (SE={self.alpha_yx_se:.4f}, Z={self.alpha_yx_z:+.2f}, p={self.alpha_yx_pval:.3e})",
            f"  U -> X (t_x)     : {self.t_x:+.4f} (SE={self.t_x_se:.4f}, p={self.t_x_pval:.3e})",
            f"  U -> Y (t_y)     : {self.t_y:+.4f} (SE={self.t_y_se:.4f}, p={self.t_y_pval:.3e})",
            f"  Direct h2        : h2_x={self.h2_x:.4f}, h2_y={self.h2_y:.4f}",
            f"  Polygenicity     : pi_x={self.pi_x:.4f}, pi_y={self.pi_y:.4f}, pi_u={self.pi_u:.4f}",
            f"  Intercepts       : i_x={self.i_x:.4f}, i_y={self.i_y:.4f}, i_xy={self.i_xy:+.4f}",
        ]
        return "\n".join(lines)


def _estimate_kurtosis_pi(
    z: np.ndarray, ld: np.ndarray, n_arr: np.ndarray, intercept: float, h2: float, m_eff: float
) -> float:
    """Rough fourth-moment starting estimate of polygenicity pi in [0.002, 0.25]."""
    c2 = z * z
    var_tot = intercept + n_arr * ld * (h2 / m_eff)
    excess_kurt = float(np.mean(c2 * c2 - 3.0 * var_tot * var_tot))
    if excess_kurt <= 1e-10:
        return 0.05
    num = 3.0 * float(np.mean(np.square(n_arr * ld * h2 / m_eff)))
    pi_est = num / (excess_kurt + num)
    return float(np.clip(pi_est, 0.003, 0.20))


def _generate_starting_points(
    ldsc_res: BivariateLDSCResult,
    zx: np.ndarray,
    zy: np.ndarray,
    ld: np.ndarray,
    nx_arr: np.ndarray,
    ny_arr: np.ndarray,
    m_eff: float,
    n_starts: int,
    seed: int,
    mode: SettingsMode,
) -> list[np.ndarray]:
    """Generate deterministic + Sobol/random starting 12-parameter vectors."""
    rng = np.random.default_rng(seed)
    h2x0 = float(np.clip(ldsc_res.h2_x, 0.01, 0.80))
    h2y0 = float(np.clip(ldsc_res.h2_y, 0.01, 0.80))
    gxy0 = float(ldsc_res.g_xy)
    ix0 = float(ldsc_res.i_x)
    iy0 = float(ldsc_res.i_y)
    ixy0 = float(ldsc_res.i_xy)

    pix0 = _estimate_kurtosis_pi(zx, ld, nx_arr, ix0, h2x0, m_eff)
    piy0 = _estimate_kurtosis_pi(zy, ld, ny_arr, iy0, h2y0, m_eff)
    piu0 = float(np.clip(0.35 * min(pix0, piy0), 0.002, 0.08))

    slope_xy = float(np.clip(gxy0 / max(h2x0, 0.02), -0.8, 0.8))
    slope_yx = float(np.clip(gxy0 / max(h2y0, 0.02), -0.8, 0.8))
    conf_mag = float(np.clip(np.sqrt(max(abs(gxy0) * 0.5, 0.005)), 0.05, 0.45))
    conf_sign = 1.0 if gxy0 >= 0.0 else -1.0

    deterministic = [
        # 1. Forward causal X -> Y + modest confounder
        np.array(
            [
                slope_xy,
                0.0,
                0.08,
                0.04 * conf_sign,
                0.85 * h2x0,
                max(h2y0 - (slope_xy**2) * 0.85 * h2x0, 0.02),
                pix0,
                piy0,
                piu0,
                ix0,
                iy0,
                ixy0,
            ],
            dtype=float,
        ),
        # 2. Reverse causal Y -> X + modest confounder
        np.array(
            [
                0.0,
                slope_yx,
                0.08,
                0.04 * conf_sign,
                max(h2x0 - (slope_yx**2) * 0.85 * h2y0, 0.02),
                0.85 * h2y0,
                pix0,
                piy0,
                piu0,
                ix0,
                iy0,
                ixy0,
            ],
            dtype=float,
        ),
        # 3. Pure heritable confounder U (alpha_xy = alpha_yx = 0)
        np.array(
            [
                0.0,
                0.0,
                conf_mag,
                conf_sign * conf_mag,
                max(h2x0 - conf_mag**2, 0.02),
                max(h2y0 - conf_mag**2, 0.02),
                pix0,
                piy0,
                piu0,
                ix0,
                iy0,
                ixy0,
            ],
            dtype=float,
        ),
        # 4. Bidirectional moderate effects
        np.array(
            [
                0.5 * slope_xy,
                0.25 * slope_yx,
                0.10,
                0.10 * conf_sign,
                0.75 * h2x0,
                0.75 * h2y0,
                pix0,
                piy0,
                piu0,
                ix0,
                iy0,
                ixy0,
            ],
            dtype=float,
        ),
    ]

    starts: list[np.ndarray] = deterministic[: max(1, min(len(deterministic), n_starts))]
    n_extra = max(0, n_starts - len(starts))
    if n_extra > 0:
        if mode == "default":
            sampler = qmc.Sobol(d=9, scramble=True, rng=rng)
            m_pow = int(np.ceil(np.log2(max(n_extra, 2))))
            unit_pts = sampler.random_base2(m_pow)[:n_extra]
        else:
            unit_pts = rng.uniform(0.0, 1.0, size=(n_extra, 9))

        for row in unit_pts:
            a_xy = float(-0.5 + 1.0 * row[0])
            a_yx = float(-0.5 + 1.0 * row[1])
            if abs(a_xy * a_yx) >= 0.80:
                a_yx *= 0.5
            tx = float(0.02 + 0.38 * row[2])
            ty = float(-0.40 + 0.80 * row[3])
            hx = float(np.clip(h2x0 * (0.4 + 0.8 * row[4]), 0.01, 0.90))
            hy = float(np.clip(h2y0 * (0.4 + 0.8 * row[5]), 0.01, 0.90))
            px = float(10.0 ** (-3.0 + 2.3 * row[6]))
            py = float(10.0 ** (-3.0 + 2.3 * row[7]))
            pu = float(10.0 ** (-3.3 + 2.0 * row[8]))
            starts.append(
                np.array(
                    [a_xy, a_yx, tx, ty, hx, hy, px, py, pu, ix0, iy0, ixy0],
                    dtype=float,
                )
            )

    return starts


def _build_codec(
    ldsc_res: BivariateLDSCResult,
    mode: SettingsMode,
    submodel: str = "bidirectional",
) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    list[tuple[float | None, float | None]],
]:
    """Construct pack/unpack functions and L-BFGS-B bounds for a model specification.

    Submodels:
    - ``"bidirectional"``: full LHC-MR
    - ``"unidirectional_xy"``: ``alpha_yx = 0``
    - ``"unidirectional_yx"``: ``alpha_xy = 0``
    - ``"confounder_only"``: ``alpha_xy = 0, alpha_yx = 0``
    - ``"no_confounder"``: ``t_x = 0, t_y = 0``
    """
    ix_0 = float(ldsc_res.i_x)
    iy_0 = float(ldsc_res.i_y)
    max_ixy = 0.95 * float(np.sqrt(ix_0 * iy_0))

    free_axy = submodel in ("bidirectional", "unidirectional_xy", "no_confounder")
    free_ayx = submodel in ("bidirectional", "unidirectional_yx", "no_confounder")
    free_conf = submodel != "no_confounder"

    if mode == "R_default":
        # Direct bounded parameters matching lhcMR L-BFGS-B with i_X, i_Y fixed at Stage-1 LDSC
        bounds: list[tuple[float | None, float | None]] = []
        if free_axy:
            bounds.append((-2.0, 2.0))
        if free_ayx:
            bounds.append((-2.0, 2.0))
        if free_conf:
            bounds.append((0.0, 2.0))   # t_x >= 0
            bounds.append((-2.0, 2.0))  # t_y
        bounds.extend(
            [
                (1e-5, 1.0),          # h2_x
                (1e-5, 1.0),          # h2_y
                (1e-5, 0.99),         # pi_x
                (1e-5, 0.99),         # pi_y
                (1e-5, 0.99),         # pi_u
                (-max_ixy, max_ixy),  # i_xy
            ]
        )

        def pack_r(p12: np.ndarray) -> np.ndarray:
            vals: list[float] = []
            if free_axy:
                vals.append(float(np.clip(p12[0], -1.99, 1.99)))
            if free_ayx:
                vals.append(float(np.clip(p12[1], -1.99, 1.99)))
            if free_conf:
                vals.append(float(np.clip(p12[2], 0.0, 1.99)))
                vals.append(float(np.clip(p12[3], -1.99, 1.99)))
            vals.extend(
                [
                    float(np.clip(p12[4], 1e-4, 0.99)),
                    float(np.clip(p12[5], 1e-4, 0.99)),
                    float(np.clip(p12[6], 1e-4, 0.98)),
                    float(np.clip(p12[7], 1e-4, 0.98)),
                    float(np.clip(p12[8], 1e-4, 0.98)),
                    float(np.clip(p12[11], -0.95 * max_ixy, 0.95 * max_ixy)),
                ]
            )
            return np.array(vals, dtype=float)

        def unpack_r(u: np.ndarray) -> np.ndarray:
            idx = 0
            axy = float(u[idx]) if free_axy else 0.0
            idx += 1 if free_axy else 0
            ayx = float(u[idx]) if free_ayx else 0.0
            idx += 1 if free_ayx else 0
            if free_conf:
                tx = float(u[idx])
                ty = float(u[idx + 1])
                idx += 2
            else:
                tx = 0.0
                ty = 0.0
                # fix unused pi_u to 0.05 when t_x=t_y=0
            h2x = float(u[idx])
            h2y = float(u[idx + 1])
            pix = float(u[idx + 2])
            piy = float(u[idx + 3])
            piu = float(u[idx + 4]) if free_conf else 0.05
            ixy = float(u[idx + 5])
            return np.array(
                [axy, ayx, tx, ty, h2x, h2y, pix, piy, piu, ix_0, iy_0, ixy],
                dtype=float,
            )

        return pack_r, unpack_r, bounds

    # settings == "default": smooth softplus/logit transforms + Stage-2 intercept refinement
    bounds_d: list[tuple[float | None, float | None]] = []
    if free_axy:
        bounds_d.append((-2.5, 2.5))
    if free_ayx:
        bounds_d.append((-2.5, 2.5))
    if free_conf:
        bounds_d.append((0.0, 2.5))    # t_x >= 0
        bounds_d.append((-2.5, 2.5))   # t_y
    bounds_d.extend(
        [
            (-11.0, 1.5),   # u_h2x -> softplus in [1.6e-5, ~1.7]
            (-11.0, 1.5),   # u_h2y
            (-11.0, 4.5),   # logit(pi_x) in [1.6e-5, 0.989]
            (-11.0, 4.5),   # logit(pi_y)
            (-11.0, 4.5),   # logit(pi_u)
            (max(0.5, 0.85 * ix_0), 1.15 * ix_0 + 0.1),  # i_x refined near LDSC
            (max(0.5, 0.85 * iy_0), 1.15 * iy_0 + 0.1),  # i_y refined near LDSC
            (-0.95, 0.95),  # rho_e = i_xy / sqrt(i_x * i_y)
        ]
    )

    def pack_d(p12: np.ndarray) -> np.ndarray:
        vals: list[float] = []
        if free_axy:
            vals.append(float(np.clip(p12[0], -2.4, 2.4)))
        if free_ayx:
            vals.append(float(np.clip(p12[1], -2.4, 2.4)))
        if free_conf:
            vals.append(float(np.clip(p12[2], 0.0, 2.4)))
            vals.append(float(np.clip(p12[3], -2.4, 2.4)))
        ix_c = float(np.clip(p12[9], max(0.5, 0.85 * ix_0), 1.15 * ix_0 + 0.1))
        iy_c = float(np.clip(p12[10], max(0.5, 0.85 * iy_0), 1.15 * iy_0 + 0.1))
        rho_e = float(np.clip(p12[11] / np.sqrt(ix_c * iy_c), -0.92, 0.92))
        vals.extend(
            [
                float(np.clip(_inv_softplus(float(p12[4])), -10.5, 1.4)),
                float(np.clip(_inv_softplus(float(p12[5])), -10.5, 1.4)),
                float(logit(np.clip(p12[6], 5e-5, 0.98))),
                float(logit(np.clip(p12[7], 5e-5, 0.98))),
                float(logit(np.clip(p12[8], 5e-5, 0.98))),
                ix_c,
                iy_c,
                rho_e,
            ]
        )
        return np.array(vals, dtype=float)

    def unpack_d(u: np.ndarray) -> np.ndarray:
        idx = 0
        axy = float(u[idx]) if free_axy else 0.0
        idx += 1 if free_axy else 0
        ayx = float(u[idx]) if free_ayx else 0.0
        idx += 1 if free_ayx else 0
        if free_conf:
            tx = float(u[idx])
            ty = float(u[idx + 1])
            idx += 2
        else:
            tx = 0.0
            ty = 0.0
        h2x = float(_softplus(float(u[idx]))) + 1e-6
        h2y = float(_softplus(float(u[idx + 1]))) + 1e-6
        pix = float(expit(float(u[idx + 2])))
        piy = float(expit(float(u[idx + 3])))
        piu = float(expit(float(u[idx + 4]))) if free_conf else 0.05
        ix = float(u[idx + 5])
        iy = float(u[idx + 6])
        rho_e = float(u[idx + 7])
        ixy = rho_e * float(np.sqrt(max(ix * iy, 1e-8)))
        return np.array(
            [axy, ayx, tx, ty, h2x, h2y, pix, piy, piu, ix, iy, ixy],
            dtype=float,
        )

    return pack_d, unpack_d, bounds_d


def _fit_single_start(
    p12_init: np.ndarray,
    pack_fn: Callable[[np.ndarray], np.ndarray],
    unpack_fn: Callable[[np.ndarray], np.ndarray],
    bounds: list[tuple[float | None, float | None]],
    data: LHCData,
    maxiter: int = 100,
) -> tuple[np.ndarray, float, bool]:
    u0 = pack_fn(p12_init)
    res = optimize.minimize(
        neg_loglik_and_grad,
        u0,
        args=(unpack_fn, data),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5},
    )
    p_hat = unpack_fn(res.x)
    ll_hat = float(-res.fun)
    return p_hat, ll_hat, bool(res.success)


def _compute_hessian_and_jackknife_se(
    p12_hat: np.ndarray,
    data: LHCData,
    mode: SettingsMode,
    ldsc_res: BivariateLDSCResult,
    n_blocks: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute observed Fisher information SEs and block-jackknife SEs."""
    free_intercepts = mode == "default"
    if free_intercepts:
        nat_vec = p12_hat.copy()

        def unpack_nat(v: np.ndarray) -> np.ndarray:
            out = v.copy()
            out[2] = max(float(out[2]), 0.0)
            out[4] = max(float(out[4]), 1e-6)
            out[5] = max(float(out[5]), 1e-6)
            out[6] = float(np.clip(out[6], 1e-6, 0.999))
            out[7] = float(np.clip(out[7], 1e-6, 0.999))
            out[8] = float(np.clip(out[8], 1e-6, 0.999))
            out[9] = max(float(out[9]), 0.2)
            out[10] = max(float(out[10]), 0.2)
            return out

    else:
        nat_vec = np.concatenate([p12_hat[:9], np.array([p12_hat[11]], dtype=float)])
        ix_f = float(ldsc_res.i_x)
        iy_f = float(ldsc_res.i_y)

        def unpack_nat(v: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    float(v[0]),
                    float(v[1]),
                    max(float(v[2]), 0.0),
                    float(v[3]),
                    max(float(v[4]), 1e-6),
                    max(float(v[5]), 1e-6),
                    float(np.clip(v[6], 1e-6, 0.999)),
                    float(np.clip(v[7], 1e-6, 0.999)),
                    float(np.clip(v[8], 1e-6, 0.999)),
                    ix_f,
                    iy_f,
                    float(v[9]),
                ],
                dtype=float,
            )

    d = nat_vec.size
    hess_nll = np.zeros((d, d), dtype=float)
    _, g0 = neg_loglik_and_grad(nat_vec, unpack_nat, data)
    for k in range(d):
        step = max(1e-5 * max(1.0, abs(float(nat_vec[k]))), 1e-5)
        vp = nat_vec.copy()
        vm = nat_vec.copy()
        vp[k] += step
        if k == 2 and nat_vec[k] - step < 1e-6:
            _, gp = neg_loglik_and_grad(vp, unpack_nat, data)
            hess_nll[:, k] = (gp - g0) / step
        else:
            vm[k] -= step
            _, gp = neg_loglik_and_grad(vp, unpack_nat, data)
            _, gm = neg_loglik_and_grad(vm, unpack_nat, data)
            hess_nll[:, k] = (gp - gm) / (2.0 * step)

    hess_sym = 0.5 * (hess_nll + hess_nll.T)
    eigvals, eigvecs = np.linalg.eigh(hess_sym)
    inv_eigs = np.where(eigvals > 1e-5, 1.0 / np.maximum(eigvals, 1e-5), 0.0)
    h_inv = (eigvecs * inv_eigs[None, :]) @ eigvecs.T

    hess_diag = np.maximum(np.diag(h_inv)[:9], 1e-12)
    hess_se_arr = np.sqrt(hess_diag)
    hess_se = {name: float(hess_se_arr[i]) for i, name in enumerate(PARAM_NAMES)}

    b_eff = min(int(n_blocks), data.bx.size // 10)
    if b_eff < 2:
        return hess_se, dict(hess_se)

    # Exact block-jackknife influence-function covariance across genomic blocks
    block_sens = block_component_sensitivities(data, unpack_nat(nat_vec), b_eff)
    jac_nat = summary_jacobian(nat_vec, unpack_nat, data.m_eff)
    block_scores = block_sens @ jac_nat  # (B, d)
    centered_scores = block_scores - np.mean(block_scores, axis=0, keepdims=True)
    meat_jk = (float(b_eff) / float(b_eff - 1)) * (centered_scores.T @ centered_scores)
    cov_jk = h_inv @ meat_jk @ h_inv
    jk_diag = np.maximum(np.diag(cov_jk)[:9], 1e-12)
    jk_se_arr = np.sqrt(jk_diag)
    jk_se = {name: float(jk_se_arr[i]) for i, name in enumerate(PARAM_NAMES)}
    return hess_se, jk_se


def lhc_mr(
    bx: ArrayLike,
    sx: ArrayLike,
    by: ArrayLike,
    sy: ArrayLike,
    ld_scores: ArrayLike,
    nx: float | ArrayLike,
    ny: float | ArrayLike,
    m_snps: float | None = None,
    n_starts: int = 5,
    n_blocks: int = 20,
    seed: int = 42,
    filter_outliers: bool | None = None,
    settings: SettingsMode = "default",
) -> LHCMRResult:
    """Estimate bidirectional causal effects under a latent heritable confounder (LHC-MR).

    Implements Darrous et al., *Nature Communications* 2021 (DOI: 10.1038/s41467-021-26970-w).
    """
    mode = validate_settings(settings)
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1.")
    if n_blocks < 0:
        raise ValueError("n_blocks must be >= 0.")

    bx_arr = np.asarray(bx, dtype=float).ravel()
    sx_arr = np.asarray(sx, dtype=float).ravel()
    by_arr = np.asarray(by, dtype=float).ravel()
    sy_arr = np.asarray(sy, dtype=float).ravel()
    ld_arr = np.asarray(ld_scores, dtype=float).ravel()

    if not (bx_arr.shape == sx_arr.shape == by_arr.shape == sy_arr.shape == ld_arr.shape):
        raise ValueError("bx, sx, by, sy, and ld_scores must have identical 1D shapes.")
    if bx_arr.size < 30:
        raise ValueError("At least 30 SNPs are required for LHC-MR.")
    if np.any(sx_arr <= 0) or np.any(sy_arr <= 0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")

    nx_full = np.broadcast_to(np.asarray(nx, dtype=float), bx_arr.shape).copy()
    ny_full = np.broadcast_to(np.asarray(ny, dtype=float), by_arr.shape).copy()
    if np.any(nx_full <= 0) or np.any(ny_full <= 0):
        raise ValueError("Sample sizes nx and ny must be strictly positive.")

    finite = (
        np.isfinite(bx_arr)
        & np.isfinite(sx_arr)
        & np.isfinite(by_arr)
        & np.isfinite(sy_arr)
        & np.isfinite(ld_arr)
        & np.isfinite(nx_full)
        & np.isfinite(ny_full)
    )
    bx_f = bx_arr[finite]
    sx_f = sx_arr[finite]
    by_f = by_arr[finite]
    sy_f = sy_arr[finite]
    ld_f = np.maximum(ld_arr[finite], 1e-3)
    nx_f = nx_full[finite]
    ny_f = ny_full[finite]

    zx_all = bx_f / sx_f
    zy_all = by_f / sy_f
    m_total = float(m_snps) if m_snps is not None else float(zx_all.size)

    do_filter = (mode == "R_default") if filter_outliers is None else bool(filter_outliers)
    ldsc_res = bivariate_ldsc(
        zx_all,
        zy_all,
        ld_f,
        nx_f,
        ny_f,
        m_snps=m_total,
        filter_outliers=do_filter,
        settings=mode,
    )

    if do_filter:
        cut_x = np.maximum(80.0, 0.001 * nx_f)
        cut_y = np.maximum(80.0, 0.001 * ny_f)
        keep = (zx_all * zx_all <= cut_x) & (zy_all * zy_all <= cut_y)
        if int(np.sum(keep)) >= 30:
            frac_kept = float(np.sum(keep)) / float(zx_all.size)
            zx_use = zx_all[keep]
            zy_use = zy_all[keep]
            ld_use = ld_f[keep]
            nx_use = nx_f[keep]
            ny_use = ny_f[keep]
            m_eff = m_total * frac_kept if m_snps is None else m_total
        else:
            zx_use, zy_use, ld_use, nx_use, ny_use = zx_all, zy_all, ld_f, nx_f, ny_f
            m_eff = m_total
    else:
        zx_use, zy_use, ld_use, nx_use, ny_use = zx_all, zy_all, ld_f, nx_f, ny_f
        m_eff = m_total

    # Standardized per-SD GWAS summary effects (Darrous et al., 2021)
    std_bx = zx_use / np.sqrt(nx_use)
    std_by = zy_use / np.sqrt(ny_use)
    data = prepare_lhc_data(std_bx, std_by, nx_use, ny_use, ld_use, m_eff=m_eff)

    starts = _generate_starting_points(
        ldsc_res,
        zx_use,
        zy_use,
        ld_use,
        nx_use,
        ny_use,
        m_eff=m_eff,
        n_starts=n_starts,
        seed=seed,
        mode=mode,
    )

    pack_bi, unpack_bi, bounds_bi = _build_codec(ldsc_res, mode, submodel="bidirectional")

    best_p12: np.ndarray | None = None
    best_ll = -np.inf
    any_converged = False
    for p_start in starts:
        p_hat, ll_hat, conv = _fit_single_start(
            p_start, pack_bi, unpack_bi, bounds_bi, data, maxiter=100
        )
        if conv:
            any_converged = True
        if ll_hat > best_ll:
            best_ll = ll_hat
            best_p12 = p_hat

    assert best_p12 is not None

    # Fit nested submodels for LRTs and ensure bidirectional global optimum >= all submodels
    submodels = (
        "unidirectional_xy",
        "unidirectional_yx",
        "confounder_only",
        "no_confounder",
    )
    sub_ll: dict[str, float] = {}
    sub_params: dict[str, np.ndarray] = {}

    for sm in submodels:
        pack_s, unpack_s, bounds_s = _build_codec(ldsc_res, mode, submodel=sm)
        cand_inits = [best_p12.copy(), starts[0].copy()]
        best_sm_ll = -np.inf
        best_sm_p = best_p12.copy()
        for p_in in cand_inits:
            ps_hat, ls_hat, _ = _fit_single_start(
                p_in, pack_s, unpack_s, bounds_s, data, maxiter=70
            )
            if ls_hat > best_sm_ll:
                best_sm_ll = ls_hat
                best_sm_p = ps_hat
        sub_ll[sm] = best_sm_ll
        sub_params[sm] = best_sm_p

        # Warm-start full bidirectional model from each submodel optimum to guarantee nestedness
        if best_sm_ll > best_ll - 0.5:
            p_warm, ll_warm, conv_w = _fit_single_start(
                best_sm_p, pack_bi, unpack_bi, bounds_bi, data, maxiter=60
            )
            if conv_w:
                any_converged = True
            if ll_warm > best_ll:
                best_ll = ll_warm
                best_p12 = p_warm

    # Darrous et al. post-fit reflection check: compare (t_X, t_Y) swap/sign reflection likelihoods
    reflection_ll_diff = 0.0
    swap_candidates = []
    p_swap_txy = best_p12.copy()
    p_swap_txy[2] = abs(float(best_p12[3]))
    p_swap_txy[3] = (1.0 if best_p12[3] >= 0 else -1.0) * float(best_p12[2])
    swap_candidates.append(p_swap_txy)

    p_flip_ty = best_p12.copy()
    p_flip_ty[3] = -float(best_p12[3])
    swap_candidates.append(p_flip_ty)

    best_swap_ll = -np.inf
    for cand in swap_candidates:
        if mode == "R_default":
            p_sw, ll_sw, _ = _fit_single_start(
                cand, pack_bi, unpack_bi, bounds_bi, data, maxiter=120
            )
        else:
            ll_sw = loglik_only(data, cand)
            p_sw = cand
        if ll_sw > best_swap_ll:
            best_swap_ll = ll_sw
        if ll_sw > best_ll + 1e-5:
            best_ll = ll_sw
            best_p12 = p_sw

    reflection_ll_diff = float(best_ll - best_swap_ll)
    best_p12, reflection_swapped = canonicalize_reflection(best_p12)
    best_ll = max(best_ll, loglik_only(data, best_p12))
    sub_ll["bidirectional"] = best_ll

    # Also ensure unidirectional nested ordering if bidirectional improved
    for sm in ("unidirectional_xy", "unidirectional_yx"):
        if sub_ll["confounder_only"] > sub_ll[sm]:
            pack_s, unpack_s, bounds_s = _build_codec(ldsc_res, mode, submodel=sm)
            ps_hat, ls_hat, _ = _fit_single_start(
                sub_params["confounder_only"], pack_s, unpack_s, bounds_s, data, maxiter=80
            )
            if ls_hat > sub_ll[sm]:
                sub_ll[sm] = ls_hat

    # Standard errors: Hessian & block-jackknife
    hess_se, jk_se = _compute_hessian_and_jackknife_se(
        best_p12, data, mode, ldsc_res, n_blocks=n_blocks
    )
    primary_se = jk_se if n_blocks >= 2 else hess_se

    params_dict = {name: float(best_p12[i]) for i, name in enumerate(PARAM_NAMES)}
    z_dict: dict[str, float] = {}
    p_dict: dict[str, float] = {}
    for name in PARAM_NAMES:
        se_val = max(primary_se[name], 1e-12)
        z_val = params_dict[name] / se_val
        z_dict[name] = float(z_val)
        p_dict[name] = float(2.0 * stats.norm.sf(abs(z_val)))

    def _make_lrt(null_name: str, alt_name: str, df: int) -> LRTResult:
        ll_0 = sub_ll[null_name]
        ll_1 = sub_ll[alt_name]
        stat = max(0.0, 2.0 * (ll_1 - ll_0))
        pval = float(stats.chi2.sf(stat, df=df))
        return LRTResult(
            null_model=null_name,
            alt_model=alt_name,
            ll_null=float(ll_0),
            ll_alt=float(ll_1),
            stat=float(stat),
            df=df,
            p_value=pval,
        )

    lrt_dict = {
        "bidirectional_vs_unidirectional_xy": _make_lrt(
            "unidirectional_xy", "bidirectional", df=1
        ),
        "bidirectional_vs_unidirectional_yx": _make_lrt(
            "unidirectional_yx", "bidirectional", df=1
        ),
        "bidirectional_vs_confounder_only": _make_lrt(
            "confounder_only", "bidirectional", df=2
        ),
        "unidirectional_xy_vs_confounder_only": _make_lrt(
            "confounder_only", "unidirectional_xy", df=1
        ),
        "unidirectional_yx_vs_confounder_only": _make_lrt(
            "confounder_only", "unidirectional_yx", df=1
        ),
        "bidirectional_vs_no_confounder": _make_lrt(
            "no_confounder", "bidirectional", df=2
        ),
    }

    return LHCMRResult(
        alpha_xy=params_dict["alpha_xy"],
        alpha_xy_se=primary_se["alpha_xy"],
        alpha_xy_z=z_dict["alpha_xy"],
        alpha_xy_pval=p_dict["alpha_xy"],
        alpha_yx=params_dict["alpha_yx"],
        alpha_yx_se=primary_se["alpha_yx"],
        alpha_yx_z=z_dict["alpha_yx"],
        alpha_yx_pval=p_dict["alpha_yx"],
        t_x=params_dict["t_x"],
        t_x_se=primary_se["t_x"],
        t_x_z=z_dict["t_x"],
        t_x_pval=p_dict["t_x"],
        t_y=params_dict["t_y"],
        t_y_se=primary_se["t_y"],
        t_y_z=z_dict["t_y"],
        t_y_pval=p_dict["t_y"],
        h2_x=params_dict["h2_x"],
        h2_y=params_dict["h2_y"],
        pi_x=params_dict["pi_x"],
        pi_y=params_dict["pi_y"],
        pi_u=params_dict["pi_u"],
        i_x=float(best_p12[9]),
        i_y=float(best_p12[10]),
        i_xy=float(best_p12[11]),
        params=params_dict,
        se=primary_se,
        jackknife_se=jk_se,
        hessian_se=hess_se,
        z_scores=z_dict,
        p_values=p_dict,
        log_likelihood=float(best_ll),
        converged=any_converged,
        n_snps=int(std_bx.size),
        m_snps=float(m_eff),
        stage1_ldsc=ldsc_res,
        reflection_swapped=reflection_swapped,
        reflection_ll_diff=reflection_ll_diff,
        submodel_logliks=sub_ll,
        lrt=lrt_dict,
        settings=mode,
    )


class LHCMR:
    """Object-oriented estimator for Latent Heritable Confounder MR (LHC-MR)."""

    def __init__(
        self,
        n_starts: int = 5,
        n_blocks: int = 20,
        seed: int = 42,
        filter_outliers: bool | None = None,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1.")
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0.")
        self.n_starts = n_starts
        self.n_blocks = n_blocks
        self.seed = seed
        self.filter_outliers = filter_outliers
        self.result_: LHCMRResult | None = None

    def fit(
        self,
        bx: ArrayLike,
        sx: ArrayLike,
        by: ArrayLike,
        sy: ArrayLike,
        ld_scores: ArrayLike,
        nx: float | ArrayLike,
        ny: float | ArrayLike,
        m_snps: float | None = None,
    ) -> LHCMRResult:
        """Fit LHC-MR on genome-wide GWAS summary statistics."""
        self.result_ = lhc_mr(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            ld_scores=ld_scores,
            nx=nx,
            ny=ny,
            m_snps=m_snps,
            n_starts=self.n_starts,
            n_blocks=self.n_blocks,
            seed=self.seed,
            filter_outliers=self.filter_outliers,
            settings=self.settings,
        )
        return self.result_
