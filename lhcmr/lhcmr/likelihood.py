"""Exact 8-component structural Gaussian mixture log-likelihood for LHC-MR."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# 8 latent activity configurations c = (c_X, c_Y, c_U) in {0, 1}^3
_C_GRID = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=float,
)
_CX = _C_GRID[:, 0]
_CY = _C_GRID[:, 1]
_CU = _C_GRID[:, 2]
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class LHCData:
    """Precomputed standardized GWAS arrays for fast likelihood evaluation."""

    bx: np.ndarray
    by: np.ndarray
    bx2: np.ndarray
    by2: np.ndarray
    bxby: np.ndarray
    inv_nx: np.ndarray
    inv_ny: np.ndarray
    inv_sqrt_nxny: np.ndarray
    ld: np.ndarray
    m_eff: float


def prepare_lhc_data(
    bx: np.ndarray,
    by: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    ld: np.ndarray,
    m_eff: float,
) -> LHCData:
    """Precompute quadratic products and inverse sample-size terms."""
    inv_nx = 1.0 / nx
    inv_ny = 1.0 / ny
    inv_sqrt_nxny = 1.0 / np.sqrt(nx * ny)
    return LHCData(
        bx=bx,
        by=by,
        bx2=bx * bx,
        by2=by * by,
        bxby=bx * by,
        inv_nx=inv_nx,
        inv_ny=inv_ny,
        inv_sqrt_nxny=inv_sqrt_nxny,
        ld=ld,
        m_eff=float(m_eff),
    )


def structural_components(
    params_12: np.ndarray, m_eff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Map 12 natural LHC-MR parameters to 8-component weights and genetic covariances."""
    (
        alpha_xy,
        alpha_yx,
        t_x,
        t_y,
        h2_x,
        h2_y,
        pi_x,
        pi_y,
        pi_u,
        i_x,
        i_y,
        i_xy,
    ) = params_12

    pi_x_c = float(np.clip(pi_x, 1e-8, 1.0 - 1e-8))
    pi_y_c = float(np.clip(pi_y, 1e-8, 1.0 - 1e-8))
    pi_u_c = float(np.clip(pi_u, 1e-8, 1.0 - 1e-8))

    log_w = (
        _CX * np.log(pi_x_c)
        + (1.0 - _CX) * np.log1p(-pi_x_c)
        + _CY * np.log(pi_y_c)
        + (1.0 - _CY) * np.log1p(-pi_y_c)
        + _CU * np.log(pi_u_c)
        + (1.0 - _CU) * np.log1p(-pi_u_c)
    )

    s2_x = max(float(h2_x), 1e-12) / (m_eff * pi_x_c)
    s2_y = max(float(h2_y), 1e-12) / (m_eff * pi_y_c)
    s2_u = 1.0 / (m_eff * pi_u_c)

    d_xx = _CX * s2_x + _CU * (t_x * t_x) * s2_u
    d_yy = _CY * s2_y + _CU * (t_y * t_y) * s2_u
    d_xy = _CU * (t_x * t_y) * s2_u

    delta = 1.0 - alpha_xy * alpha_yx
    inv_delta2 = 1.0 / max(delta * delta, 1e-8)

    g_xx = (d_xx + 2.0 * alpha_yx * d_xy + (alpha_yx * alpha_yx) * d_yy) * inv_delta2
    g_yy = ((alpha_xy * alpha_xy) * d_xx + 2.0 * alpha_xy * d_xy + d_yy) * inv_delta2
    g_xy = (
        alpha_xy * d_xx + (1.0 + alpha_xy * alpha_yx) * d_xy + alpha_yx * d_yy
    ) * inv_delta2

    return log_w, g_xx, g_xy, g_yy, float(i_x), float(i_y), float(i_xy)


def loglik_and_component_sensitivities(
    data: LHCData,
    params_12: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate total log-likelihood and exact sensitivity wrt the 35-dim summary map."""
    alpha_xy = float(params_12[0])
    alpha_yx = float(params_12[1])
    prod = abs(alpha_xy * alpha_yx)
    if prod >= 0.95:
        pen = 1e7 * (prod - 0.95 + 1e-3) ** 2
        return -pen, np.zeros(35, dtype=float)

    log_w, g_xx, g_xy, g_yy, i_x, i_y, i_xy = structural_components(
        params_12, data.m_eff
    )

    se_xx = i_x * data.inv_nx
    se_yy = i_y * data.inv_ny
    se_xy = i_xy * data.inv_sqrt_nxny

    det_e = se_xx * se_yy - se_xy * se_xy
    if np.any(se_xx <= 1e-15) or np.any(se_yy <= 1e-15) or np.any(det_e <= 1e-24):
        return -1e9, np.zeros(35, dtype=float)

    v_xx = se_xx[:, None] + data.ld[:, None] * g_xx[None, :]
    v_yy = se_yy[:, None] + data.ld[:, None] * g_yy[None, :]
    v_xy = se_xy[:, None] + data.ld[:, None] * g_xy[None, :]

    det = v_xx * v_yy - v_xy * v_xy
    if np.any(det <= 1e-30):
        return -1e9, np.zeros(35, dtype=float)

    inv_det = 1.0 / det
    u_x = (v_yy * data.bx[:, None] - v_xy * data.by[:, None]) * inv_det
    u_y = (-v_xy * data.bx[:, None] + v_xx * data.by[:, None]) * inv_det
    quad = data.bx[:, None] * u_x + data.by[:, None] * u_y

    joint = log_w[None, :] - _LOG_2PI - 0.5 * np.log(det) - 0.5 * quad
    j_max = np.max(joint, axis=1, keepdims=True)
    exp_j = np.exp(joint - j_max)
    sum_exp = np.sum(exp_j, axis=1, keepdims=True)
    ll = float(np.sum(j_max[:, 0] + np.log(sum_exp[:, 0])))

    resp = exp_j / sum_exp
    r_dvxx = resp * (0.5 * (u_x * u_x - v_yy * inv_det))
    r_dvyy = resp * (0.5 * (u_y * u_y - v_xx * inv_det))
    r_dvxy = resp * (u_x * u_y + v_xy * inv_det)

    d_log_w = np.sum(resp, axis=0)
    d_gxx = data.ld @ r_dvxx
    d_gxy = data.ld @ r_dvxy
    d_gyy = data.ld @ r_dvyy

    d_ix = float(np.dot(np.sum(r_dvxx, axis=1), data.inv_nx))
    d_iy = float(np.dot(np.sum(r_dvyy, axis=1), data.inv_ny))
    d_ixy = float(np.dot(np.sum(r_dvxy, axis=1), data.inv_sqrt_nxny))

    sens = np.empty(35, dtype=float)
    sens[0:8] = d_log_w
    sens[8:16] = d_gxx
    sens[16:24] = d_gxy
    sens[24:32] = d_gyy
    sens[32] = d_ix
    sens[33] = d_iy
    sens[34] = d_ixy
    return ll, sens


def block_component_sensitivities(
    data: LHCData,
    params_12: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Compute per-block 35-dim component sensitivities in a single vectorized pass."""
    log_w, g_xx, g_xy, g_yy, i_x, i_y, i_xy = structural_components(
        params_12, data.m_eff
    )
    se_xx = i_x * data.inv_nx
    se_yy = i_y * data.inv_ny
    se_xy = i_xy * data.inv_sqrt_nxny

    v_xx = se_xx[:, None] + data.ld[:, None] * g_xx[None, :]
    v_yy = se_yy[:, None] + data.ld[:, None] * g_yy[None, :]
    v_xy = se_xy[:, None] + data.ld[:, None] * g_xy[None, :]
    det = np.maximum(v_xx * v_yy - v_xy * v_xy, 1e-30)
    inv_det = 1.0 / det

    u_x = (v_yy * data.bx[:, None] - v_xy * data.by[:, None]) * inv_det
    u_y = (-v_xy * data.bx[:, None] + v_xx * data.by[:, None]) * inv_det
    quad = data.bx[:, None] * u_x + data.by[:, None] * u_y

    joint = log_w[None, :] - _LOG_2PI - 0.5 * np.log(det) - 0.5 * quad
    j_max = np.max(joint, axis=1, keepdims=True)
    exp_j = np.exp(joint - j_max)
    resp = exp_j / np.sum(exp_j, axis=1, keepdims=True)

    r_dvxx = resp * (0.5 * (u_x * u_x - v_yy * inv_det))
    r_dvyy = resp * (0.5 * (u_y * u_y - v_xx * inv_det))
    r_dvxy = resp * (u_x * u_y + v_xy * inv_det)

    blocks = np.array_split(np.arange(data.bx.size), n_blocks)
    out = np.zeros((n_blocks, 35), dtype=float)
    for b_idx, idx in enumerate(blocks):
        ld_b = data.ld[idx]
        rvx = r_dvxx[idx]
        rvy = r_dvyy[idx]
        rvxy = r_dvxy[idx]
        out[b_idx, 0:8] = np.sum(resp[idx], axis=0)
        out[b_idx, 8:16] = ld_b @ rvx
        out[b_idx, 16:24] = ld_b @ rvxy
        out[b_idx, 24:32] = ld_b @ rvy
        out[b_idx, 32] = float(np.dot(np.sum(rvx, axis=1), data.inv_nx[idx]))
        out[b_idx, 33] = float(np.dot(np.sum(rvy, axis=1), data.inv_ny[idx]))
        out[b_idx, 34] = float(np.dot(np.sum(rvxy, axis=1), data.inv_sqrt_nxny[idx]))
    return out


def loglik_only(data: LHCData, params_12: np.ndarray) -> float:
    """Fast forward-only log-likelihood evaluation."""
    alpha_xy = float(params_12[0])
    alpha_yx = float(params_12[1])
    prod = abs(alpha_xy * alpha_yx)
    if prod >= 0.95:
        return float(-1e7 * (prod - 0.95 + 1e-3) ** 2)

    log_w, g_xx, g_xy, g_yy, i_x, i_y, i_xy = structural_components(
        params_12, data.m_eff
    )
    se_xx = i_x * data.inv_nx
    se_yy = i_y * data.inv_ny
    se_xy = i_xy * data.inv_sqrt_nxny
    det_e = se_xx * se_yy - se_xy * se_xy
    if np.any(se_xx <= 1e-15) or np.any(se_yy <= 1e-15) or np.any(det_e <= 1e-24):
        return -1e9

    v_xx = se_xx[:, None] + data.ld[:, None] * g_xx[None, :]
    v_yy = se_yy[:, None] + data.ld[:, None] * g_yy[None, :]
    v_xy = se_xy[:, None] + data.ld[:, None] * g_xy[None, :]
    det = v_xx * v_yy - v_xy * v_xy
    if np.any(det <= 1e-30):
        return -1e9

    inv_det = 1.0 / det
    quad = (
        v_yy * data.bx2[:, None]
        - 2.0 * v_xy * data.bxby[:, None]
        + v_xx * data.by2[:, None]
    ) * inv_det
    joint = log_w[None, :] - _LOG_2PI - 0.5 * np.log(det) - 0.5 * quad
    j_max = np.max(joint, axis=1, keepdims=True)
    return float(np.sum(j_max[:, 0] + np.log(np.sum(np.exp(joint - j_max), axis=1))))


def summary_jacobian(u: np.ndarray, unpack_fn, m_eff: float) -> np.ndarray:
    """Compute (35, len(u)) Jacobian of structural summary vector wrt ``u``."""
    d = u.size
    jac = np.empty((35, d), dtype=float)
    eps = 1e-7
    for k in range(d):
        step = eps * max(1.0, abs(float(u[k])))
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[k] += step
        u_minus[k] -= step
        lw_p, gxx_p, gxy_p, gyy_p, ix_p, iy_p, ixy_p = structural_components(
            unpack_fn(u_plus), m_eff
        )
        lw_m, gxx_m, gxy_m, gyy_m, ix_m, iy_m, ixy_m = structural_components(
            unpack_fn(u_minus), m_eff
        )
        inv_2s = 0.5 / step
        jac[0:8, k] = (lw_p - lw_m) * inv_2s
        jac[8:16, k] = (gxx_p - gxx_m) * inv_2s
        jac[16:24, k] = (gxy_p - gxy_m) * inv_2s
        jac[24:32, k] = (gyy_p - gyy_m) * inv_2s
        jac[32, k] = (ix_p - ix_m) * inv_2s
        jac[33, k] = (iy_p - iy_m) * inv_2s
        jac[34, k] = (ixy_p - ixy_m) * inv_2s
    return jac


def neg_loglik_and_grad(
    u: np.ndarray,
    unpack_fn,
    data: LHCData,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and exact gradient wrt optimization vector ``u``."""
    params_12 = unpack_fn(u)
    alpha_xy = float(params_12[0])
    alpha_yx = float(params_12[1])
    prod = abs(alpha_xy * alpha_yx)
    if prod >= 0.95:
        base = 1e7 * (prod - 0.95 + 1e-3) ** 2
        grad = np.zeros_like(u)
        eps = 1e-6
        for k in range(u.size):
            up = u.copy()
            up[k] += eps
            pp = unpack_fn(up)
            pr_p = abs(float(pp[0]) * float(pp[1]))
            val_p = 1e7 * max(pr_p - 0.95 + 1e-3, 0.0) ** 2
            grad[k] = (val_p - base) / eps
        return float(base), grad

    ll, sens = loglik_and_component_sensitivities(data, params_12)
    if ll <= -5e8:
        return float(-ll), np.zeros_like(u)

    jac = summary_jacobian(u, unpack_fn, data.m_eff)
    grad_ll = sens @ jac
    return -ll, -grad_ll


def swap_factor_x_with_u(params_12: np.ndarray) -> np.ndarray | None:
    """Exact likelihood-preserving algebraic swap between genetic factors G_X and G_U."""
    p = params_12.copy()
    a_xy, a_yx, t_x, t_y, h2_x, h2_y, pi_x, pi_y, pi_u = p[:9]
    delta = 1.0 - a_xy * a_yx
    if abs(delta) < 1e-4:
        return None
    bx_1 = np.sqrt(max(h2_x, 1e-12)) / delta
    bx_2 = bx_1 * a_xy
    by_2 = np.sqrt(max(h2_y, 1e-12)) / delta
    bu_1 = (t_x + a_yx * t_y) / delta
    bu_2 = (t_y + a_xy * t_x) / delta

    if abs(bu_1) < 1e-6:
        return None
    new_a_xy = float(bu_2 / bu_1)
    new_a_yx = float(a_yx)
    if abs(new_a_xy * new_a_yx) >= 0.94:
        return None
    new_delta = 1.0 - new_a_xy * new_a_yx
    new_h2_x = float((new_delta * bu_1) ** 2)
    new_h2_y = float((new_delta * by_2) ** 2)
    if not (1e-6 < new_h2_x < 1.5 and 1e-6 < new_h2_y < 1.5):
        return None

    raw_tx = bx_1 - new_a_yx * bx_2
    raw_ty = -new_a_xy * bx_1 + bx_2
    if raw_tx < 0.0:
        raw_tx = -raw_tx
        raw_ty = -raw_ty

    out = p.copy()
    out[0] = new_a_xy
    out[1] = new_a_yx
    out[2] = float(raw_tx)
    out[3] = float(raw_ty)
    out[4] = new_h2_x
    out[5] = new_h2_y
    out[6] = float(pi_u)
    out[7] = float(pi_y)
    out[8] = float(pi_x)
    return out


def swap_factor_y_with_u(params_12: np.ndarray) -> np.ndarray | None:
    """Exact likelihood-preserving algebraic swap between genetic factors G_Y and G_U."""
    p = params_12.copy()
    a_xy, a_yx, t_x, t_y, h2_x, h2_y, pi_x, pi_y, pi_u = p[:9]
    delta = 1.0 - a_xy * a_yx
    if abs(delta) < 1e-4:
        return None
    bx_1 = np.sqrt(max(h2_x, 1e-12)) / delta
    by_2 = np.sqrt(max(h2_y, 1e-12)) / delta
    by_1 = by_2 * a_yx
    bu_1 = (t_x + a_yx * t_y) / delta
    bu_2 = (t_y + a_xy * t_x) / delta

    if abs(bu_2) < 1e-6:
        return None
    new_a_xy = float(a_xy)
    new_a_yx = float(bu_1 / bu_2)
    if abs(new_a_xy * new_a_yx) >= 0.94:
        return None
    new_delta = 1.0 - new_a_xy * new_a_yx
    new_h2_x = float((new_delta * bx_1) ** 2)
    new_h2_y = float((new_delta * bu_2) ** 2)
    if not (1e-6 < new_h2_x < 1.5 and 1e-6 < new_h2_y < 1.5):
        return None

    raw_tx = by_1 - new_a_yx * by_2
    raw_ty = -new_a_xy * by_1 + by_2
    if raw_tx < 0.0:
        raw_tx = -raw_tx
        raw_ty = -raw_ty

    out = p.copy()
    out[0] = new_a_xy
    out[1] = new_a_yx
    out[2] = float(raw_tx)
    out[3] = float(raw_ty)
    out[4] = new_h2_x
    out[5] = new_h2_y
    out[6] = float(pi_x)
    out[7] = float(pi_u)
    out[8] = float(pi_y)
    return out


def canonicalize_reflection(params_12: np.ndarray) -> tuple[np.ndarray, bool]:
    """Enforce Darrous et al. (2021) identifiability against factor-label reflection."""
    p = params_12.copy()
    swapped = False
    if p[2] < 0.0:
        p[2] = -p[2]
        p[3] = -p[3]

    for _ in range(2):
        a_xy, a_yx, t_x, t_y, h2_x, h2_y, pi_x, pi_y, pi_u = p[:9]
        delta = 1.0 - a_xy * a_yx
        bx1_sq = h2_x / max(delta * delta, 1e-8)
        by2_sq = h2_y / max(delta * delta, 1e-8)
        bu1_sq = ((t_x + a_yx * t_y) / max(delta, 1e-4)) ** 2
        bu2_sq = ((t_y + a_xy * t_x) / max(delta, 1e-4)) ** 2

        usurped_x = (bu1_sq > bx1_sq and pi_u > pi_x) or (
            h2_x < 0.02 * (bx1_sq + bu1_sq) and bu1_sq > 0.02
        )
        if usurped_x:
            cand = swap_factor_x_with_u(p)
            if cand is not None:
                p = cand
                swapped = True
                continue

        usurped_y = (bu2_sq > by2_sq and pi_u > pi_y) or (
            h2_y < 0.02 * (by2_sq + bu2_sq) and bu2_sq > 0.02
        )
        if usurped_y:
            cand = swap_factor_y_with_u(p)
            if cand is not None:
                p = cand
                swapped = True
                continue
        break

    return p, swapped
