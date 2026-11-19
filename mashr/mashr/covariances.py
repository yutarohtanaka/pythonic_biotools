"""Canonical and data-driven prior covariance matrices for `mashr`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ._linalg import (
    equilibrated_cholesky,
    gaussian_loglik_batch,
    posterior_moments_joseph,
    project_psd,
    symmetrize,
)
from .data import MashData

_VALID_CANONICAL_METHODS = {"identity", "singletons", "equal_effects", "simple_het"}


def cov_canonical(
    data: MashData | int,
    cov_methods: Sequence[str] = ("identity", "singletons", "equal_effects", "simple_het"),
    het_grids: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, np.ndarray]:
    """Construct canonical prior covariance matrices across R conditions.

    Supported methods: ``"identity"``, ``"singletons"``, ``"equal_effects"``,
    and ``"simple_het"`` (compound-symmetric correlations ``het_grids``).
    When ``R == 1``, returns ``{"identity": np.ones((1, 1))}``.
    """
    if isinstance(data, MashData):
        R = data.n_conditions
        cond_names = data.condition_names
    elif isinstance(data, (int, np.integer)) and not isinstance(data, bool):
        R = int(data)
        if R < 1:
            raise ValueError(f"Number of conditions R must be >= 1; got {R}.")
        cond_names = [f"cond_{r}" for r in range(R)]
    else:
        raise TypeError("data must be a MashData instance or an integer R >= 1.")

    methods = tuple(cov_methods)
    if len(methods) == 0:
        raise ValueError("cov_methods must not be empty.")
    for m in methods:
        if m not in _VALID_CANONICAL_METHODS:
            raise ValueError(
                f"Unknown canonical covariance method {m!r}; "
                f"expected one of {sorted(_VALID_CANONICAL_METHODS)}."
            )

    if R == 1:
        return {"identity": np.ones((1, 1), dtype=np.float64)}

    out: dict[str, np.ndarray] = {}
    for method in methods:
        if method == "identity":
            out["identity"] = np.eye(R, dtype=np.float64)
        elif method == "singletons":
            for r, cname in enumerate(cond_names):
                mat = np.zeros((R, R), dtype=np.float64)
                mat[r, r] = 1.0
                out[f"singleton_{cname}"] = mat
        elif method == "equal_effects":
            out["equal_effects"] = np.ones((R, R), dtype=np.float64)
        elif method == "simple_het":
            lower_bound = -1.0 / (R - 1)
            for rho in het_grids:
                rho_f = float(rho)
                if not np.isfinite(rho_f) or not (lower_bound < rho_f < 1.0):
                    raise ValueError(
                        f"simple_het correlation {rho_f} must satisfy "
                        f"{lower_bound:.6g} < rho < 1.0 for R={R}."
                    )
                mat = np.full((R, R), rho_f, dtype=np.float64)
                np.fill_diagonal(mat, 1.0)
                out[f"simple_het_{rho_f:g}"] = mat

    return out


def _resolve_strong_subset(
    data: MashData,
    subset: np.ndarray | Sequence[int] | Sequence[bool] | None,
    thresh: float,
) -> np.ndarray:
    J = data.n_effects
    if subset is None:
        if not np.isfinite(thresh) or thresh < 0.0:
            raise ValueError(f"thresh must be finite and >= 0; got {thresh}.")
        idx = np.flatnonzero(np.max(np.abs(data.z_scores), axis=1) > float(thresh))
        if idx.size == 0:
            raise ValueError(f"No strong signals found with max(|Z|) > {thresh}.")
        return idx

    sub_arr = np.asarray(subset)
    if sub_arr.ndim != 1:
        raise ValueError("subset must be a 1D boolean mask or integer index array.")
    if sub_arr.dtype == np.bool_:
        if sub_arr.shape[0] != J:
            raise ValueError(f"Boolean subset length {sub_arr.shape[0]} != J={J}.")
        idx = np.flatnonzero(sub_arr)
    else:
        idx = sub_arr.astype(np.intp, copy=False)
        if idx.size > 0 and (np.min(idx) < 0 or np.max(idx) >= J):
            raise ValueError(f"Integer subset indices must lie in [0, {J}).")
    if idx.size == 0:
        raise ValueError("subset resolved to 0 units.")
    return idx


def cov_pca(
    data: MashData,
    npc: int,
    subset: np.ndarray | Sequence[int] | Sequence[bool] | None = None,
    thresh: float = 2.0,
) -> dict[str, np.ndarray]:
    """Compute data-driven rank-1 PCA and rank-``npc`` truncated PCA covariance matrices.

    Runs uncentered SVD on ``data.Bhat_tilde[subset, :]`` where ``subset`` defaults
    to units with ``max_r |Z_{j,r}| > thresh``.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    R = data.n_conditions
    if not isinstance(npc, (int, np.integer)) or isinstance(npc, bool):
        raise ValueError(f"npc must be an integer; got {npc!r}.")
    npc_i = int(npc)
    if npc_i < 1 or npc_i > R:
        raise ValueError(f"npc must satisfy 1 <= npc <= R ({R}); got {npc_i}.")

    idx = _resolve_strong_subset(data, subset, thresh)
    if idx.size < npc_i:
        raise ValueError(
            f"Number of strong units ({idx.size}) is smaller than npc ({npc_i})."
        )

    mat = data.Bhat_tilde[idx, :]
    _, d, vt = np.linalg.svd(mat, full_matrices=False)

    out: dict[str, np.ndarray] = {}
    for p in range(npc_i):
        v = vt[p, :]
        out[f"PCA_{p + 1}"] = symmetrize(np.outer(v, v))

    scaled = d[:npc_i, None] * vt[:npc_i, :]
    tpca = (scaled.T @ scaled) / float(mat.shape[0])
    out["tPCA"] = symmetrize(tpca)
    return out


@dataclass(frozen=True)
class ExtremeDeconvolutionResult:
    """Result container from Bovy's Extreme Deconvolution EM (`fit_extreme_deconvolution`)."""

    U_list: dict[str, np.ndarray]
    weights: np.ndarray
    log_likelihood: float
    log_likelihood_history: np.ndarray
    n_iter: int
    converged: bool
    n_strong: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return component-level covariance summary as a tidy DataFrame."""
        rows: list[dict[str, object]] = []
        for idx, (name, cov) in enumerate(self.U_list.items()):
            cov_scale = float(np.max(np.abs(cov)))
            rank_tol = 1e-10 * cov_scale if cov_scale > 0.0 else None
            rows.append(
                {
                    "component": name,
                    "weight": float(self.weights[idx]),
                    "trace": float(np.trace(cov)),
                    "max_diag": float(np.max(np.diag(cov))),
                    "rank": int(np.linalg.matrix_rank(cov, tol=rank_tol)),
                }
            )
        return pd.DataFrame(rows)


def _validate_u_dict(
    Ulist_init: dict[str, np.ndarray] | Sequence[np.ndarray],
    R: int,
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(Ulist_init, dict):
        if len(Ulist_init) == 0:
            raise ValueError("Ulist_init must not be empty.")
        items = list(Ulist_init.items())
    else:
        seq = list(Ulist_init)
        if len(seq) == 0:
            raise ValueError("Ulist_init must not be empty.")
        items = [(f"U_{k}", mat) for k, mat in enumerate(seq)]

    names: list[str] = []
    mats: list[np.ndarray] = []
    for name, mat in items:
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (R, R):
            raise ValueError(f"Covariance {name!r} has shape {arr.shape}; expected ({R}, {R}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Covariance {name!r} contains NaN or Inf values.")
        u_scale = float(np.max(np.abs(arr)))
        max_asym = float(np.max(np.abs(arr - arr.T)))
        if u_scale > 0.0 and max_asym > 1e-8 * u_scale:
            raise ValueError(f"Covariance {name!r} is asymmetric (max diff {max_asym:.3e}).")
        sym = symmetrize(arr)
        min_eig = float(np.min(np.linalg.eigvalsh(sym)))
        if u_scale > 0.0 and min_eig < -1e-8 * u_scale:
            raise ValueError(
                f"Covariance {name!r} is not positive semi-definite (min eig {min_eig:.3e})."
            )
        names.append(str(name))
        mats.append(project_psd(sym, min_eig=0.0))

    return names, mats


def fit_extreme_deconvolution(
    data: MashData,
    Ulist_init: dict[str, np.ndarray] | Sequence[np.ndarray],
    subset: np.ndarray | Sequence[int] | Sequence[bool] | None = None,
    thresh: float = 2.0,
    weights_init: np.ndarray | Sequence[float] | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    reg: float = 0.0,
    fix_cov: bool = False,
    batch_size: int = 512,
) -> ExtremeDeconvolutionResult:
    """Refine prior covariances {U_k} via Bovy's Extreme Deconvolution EM algorithm.

    Models strong signals ``b_tilde_j ~ sum_k alpha_k N(0, U_k + V_tilde_j)``
    with heteroscedastic observation covariances ``V_tilde_j``, using Woodbury /
    Joseph-form conditional posterior updates so rank-deficient ``U_k`` are never inverted.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}.")
    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError(f"tol must be finite and >= 0; got {tol}.")
    if reg < 0.0 or not np.isfinite(reg):
        raise ValueError(f"reg must be finite and >= 0; got {reg}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")

    R = data.n_conditions
    names, u_mats = _validate_u_dict(Ulist_init, R)
    K = len(u_mats)

    idx = _resolve_strong_subset(data, subset, thresh)
    n_strong = int(idx.size)
    B_strong = data.Bhat_tilde[idx, :]
    st_strong = data.Shat_tilde[idx, :]
    V = data.V
    common_cov = data.is_common_cov

    if weights_init is None:
        alpha = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        alpha = np.asarray(weights_init, dtype=np.float64).copy()
        if alpha.shape != (K,):
            raise ValueError(f"weights_init must have shape ({K},); got {alpha.shape}.")
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0) or np.sum(alpha) <= 0.0:
            raise ValueError("weights_init must be finite, non-negative, and sum to > 0.")
        alpha /= np.sum(alpha)

    ll_history: list[float] = []
    converged = False
    prev_ll = -np.inf
    eye_r = np.eye(R, dtype=np.float64)

    V0 = data.get_cov(int(idx[0])) if common_cov else None

    for it in range(max_iter):
        log_q_tilde = np.empty((n_strong, K), dtype=np.float64)
        log_alpha = np.log(np.maximum(alpha, 1e-300))

        if common_cov:
            assert V0 is not None
            b_cache: list[np.ndarray] = []
            T_cache: list[np.ndarray] = []
            for k in range(K):
                U_k = u_mats[k]
                C_k = U_k + V0
                L_eq, d = equilibrated_cholesky(C_k)
                ll_k, y = gaussian_loglik_batch(L_eq, d, B_strong)
                log_q_tilde[:, k] = log_alpha[k] + ll_k
                b_jk, T_k = posterior_moments_joseph(
                    L_eq, d, y, U_k, V0, full_cov=True
                )
                b_cache.append(b_jk)
                T_cache.append(T_k)

            row_lse = logsumexp(log_q_tilde, axis=1)
            ll = float(np.sum(row_lse))
            ll_history.append(ll)

            if it >= 1 and abs(ll - prev_ll) < tol:
                converged = True
                break
            prev_ll = ll

            q = np.exp(log_q_tilde - row_lse[:, None])
            Q = np.sum(q, axis=0)
            alpha_new = np.where(Q > 1e-12, Q, 0.0)
            total_q = float(np.sum(alpha_new))
            alpha = alpha_new / total_q if total_q > 0.0 else np.full(K, 1.0 / K)

            if not fix_cov:
                for k in range(K):
                    if Q[k] <= 1e-12:
                        continue
                    q_k = q[:, k]
                    b_jk = b_cache[k]
                    T_k = T_cache[k]
                    second_moment = (q_k[:, None] * b_jk).T @ b_jk + Q[k] * T_k
                    if reg > 0.0:
                        U_new = (second_moment + reg * eye_r) / (Q[k] + reg)
                    else:
                        U_new = second_moment / Q[k]
                    u_mats[k] = project_psd(U_new, min_eig=0.0)
        else:
            for start in range(0, n_strong, batch_size):
                end = min(start + batch_size, n_strong)
                st_b = st_strong[start:end]
                B_b = B_strong[start:end]
                V_b = st_b[:, :, None] * V[None, :, :] * st_b[:, None, :]
                for k in range(K):
                    C_b = u_mats[k][None, :, :] + V_b
                    L_eq, d = equilibrated_cholesky(C_b)
                    ll_k, _ = gaussian_loglik_batch(L_eq, d, B_b)
                    log_q_tilde[start:end, k] = log_alpha[k] + ll_k

            row_lse = logsumexp(log_q_tilde, axis=1)
            ll = float(np.sum(row_lse))
            ll_history.append(ll)

            if it >= 1 and abs(ll - prev_ll) < tol:
                converged = True
                break
            prev_ll = ll

            q = np.exp(log_q_tilde - row_lse[:, None])
            Q = np.sum(q, axis=0)
            alpha_new = np.where(Q > 1e-12, Q, 0.0)
            total_q = float(np.sum(alpha_new))
            alpha = alpha_new / total_q if total_q > 0.0 else np.full(K, 1.0 / K)

            if not fix_cov:
                acc = [np.zeros((R, R), dtype=np.float64) for _ in range(K)]
                for start in range(0, n_strong, batch_size):
                    end = min(start + batch_size, n_strong)
                    st_b = st_strong[start:end]
                    B_b = B_strong[start:end]
                    V_b = st_b[:, :, None] * V[None, :, :] * st_b[:, None, :]
                    for k in range(K):
                        if Q[k] <= 1e-12:
                            continue
                        U_k = u_mats[k]
                        C_b = U_k[None, :, :] + V_b
                        L_eq, d = equilibrated_cholesky(C_b)
                        _, y = gaussian_loglik_batch(L_eq, d, B_b)
                        b_b, T_b = posterior_moments_joseph(
                            L_eq, d, y, U_k, V_b, full_cov=True
                        )
                        q_bk = q[start:end, k]
                        acc[k] += (q_bk[:, None] * b_b).T @ b_b + np.tensordot(
                            q_bk, T_b, axes=(0, 0)
                        )
                for k in range(K):
                    if Q[k] <= 1e-12:
                        continue
                    if reg > 0.0:
                        U_new = (acc[k] + reg * eye_r) / (Q[k] + reg)
                    else:
                        U_new = acc[k] / Q[k]
                    u_mats[k] = project_psd(U_new, min_eig=0.0)

    if not converged:
        # Evaluate log-likelihood at the final M-step parameters.
        log_q_tilde = np.empty((n_strong, K), dtype=np.float64)
        log_alpha = np.log(np.maximum(alpha, 1e-300))
        if common_cov:
            assert V0 is not None
            for k in range(K):
                L_eq, d = equilibrated_cholesky(u_mats[k] + V0)
                ll_k, _ = gaussian_loglik_batch(L_eq, d, B_strong)
                log_q_tilde[:, k] = log_alpha[k] + ll_k
        else:
            for start in range(0, n_strong, batch_size):
                end = min(start + batch_size, n_strong)
                st_b = st_strong[start:end]
                B_b = B_strong[start:end]
                V_b = st_b[:, :, None] * V[None, :, :] * st_b[:, None, :]
                for k in range(K):
                    L_eq, d = equilibrated_cholesky(u_mats[k][None, :, :] + V_b)
                    ll_k, _ = gaussian_loglik_batch(L_eq, d, B_b)
                    log_q_tilde[start:end, k] = log_alpha[k] + ll_k
        ll = float(np.sum(logsumexp(log_q_tilde, axis=1)))
        ll_history.append(ll)

    u_dict = {name: mat for name, mat in zip(names, u_mats)}
    return ExtremeDeconvolutionResult(
        U_list=u_dict,
        weights=alpha,
        log_likelihood=float(ll_history[-1]),
        log_likelihood_history=np.asarray(ll_history, dtype=np.float64),
        n_iter=len(ll_history),
        converged=converged,
        n_strong=n_strong,
    )


def cov_ed(
    data: MashData,
    Ulist_init: dict[str, np.ndarray] | Sequence[np.ndarray],
    subset: np.ndarray | Sequence[int] | Sequence[bool] | None = None,
    thresh: float = 2.0,
    weights_init: np.ndarray | Sequence[float] | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    reg: float = 0.0,
    fix_cov: bool = False,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Refine data-driven covariances via Bovy's Extreme Deconvolution EM."""
    res = fit_extreme_deconvolution(
        data=data,
        Ulist_init=Ulist_init,
        subset=subset,
        thresh=thresh,
        weights_init=weights_init,
        max_iter=max_iter,
        tol=tol,
        reg=reg,
        fix_cov=fix_cov,
        batch_size=batch_size,
    )
    return res.U_list
