"""Internal numerically stabilized linear-algebra helpers for `mashr`."""

from __future__ import annotations

import numpy as np


def symmetrize(arr: np.ndarray) -> np.ndarray:
    """Return exact symmetric average 0.5 * (A + A^T) over the last two axes."""
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def project_psd(mat: np.ndarray, min_eig: float = 0.0) -> np.ndarray:
    """Symmetrize a square matrix and clamp eigenvalues below `min_eig`."""
    sym = symmetrize(np.asarray(mat, dtype=np.float64))
    evals, evecs = np.linalg.eigh(sym)
    if np.min(evals) < min_eig:
        evals = np.maximum(evals, min_eig)
        sym = (evecs * evals[..., None, :]) @ np.swapaxes(evecs, -1, -2)
        sym = symmetrize(sym)
    return sym


def equilibrated_cholesky(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jacobi-equilibrated Cholesky factorization for `(..., R, R)` SPD matrices.

    Given symmetric matrix `T`, computes diagonal scaling `d = sqrt(max(diag(T), 1e-300))`,
    forms equilibrated `T_eq = D^{-1} T D^{-1}` with unit diagonal, and computes lower
    Cholesky factor `L_eq` such that `T approx (D @ L_eq) @ (D @ L_eq)^T`.

    Falls back to progressive diagonal jitter on `T_eq` and clamped `eigh` if needed.
    """
    T_sym = symmetrize(np.asarray(T, dtype=np.float64))
    R = T_sym.shape[-1]
    diag_T = np.diagonal(T_sym, axis1=-2, axis2=-1)
    d = np.sqrt(np.maximum(diag_T, 1e-300))
    scale = d[..., :, None] * d[..., None, :]
    T_eq = symmetrize(T_sym / scale)

    try:
        L_eq = np.linalg.cholesky(T_eq)
        return L_eq, d
    except np.linalg.LinAlgError:
        pass

    eye = np.eye(R, dtype=np.float64)
    for eps in (1e-10, 1e-8, 1e-6):
        try:
            L_eq = np.linalg.cholesky(T_eq + eps * eye)
            return L_eq, d
        except np.linalg.LinAlgError:
            continue

    evals, evecs = np.linalg.eigh(T_eq)
    evals = np.maximum(evals, 1e-12)
    T_pd = symmetrize((evecs * evals[..., None, :]) @ np.swapaxes(evecs, -1, -2))
    L_eq = np.linalg.cholesky(T_pd)
    return L_eq, d


def gaussian_loglik_batch(
    L_eq: np.ndarray,
    d: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate MVN log-density log N(b; 0, T) and whitened residual y = L^{-1} b.

    Parameters
    ----------
    L_eq : array of shape `(..., R, R)`
        Lower-triangular equilibrated Cholesky factor.
    d : array of shape `(..., R)`
        Diagonal equilibration scales.
    b : array of shape `(..., R)`
        Observation vectors.

    Returns
    -------
    loglik : array of shape `(...)`
    y : array of shape `(..., R)`
    """
    R = b.shape[-1]
    b_eq = b / d
    y = np.linalg.solve(L_eq, b_eq[..., None])[..., 0]
    diag_L = np.diagonal(L_eq, axis1=-2, axis2=-1)
    logdet = 2.0 * (np.sum(np.log(d), axis=-1) + np.sum(np.log(np.maximum(diag_L, 1e-300)), axis=-1))
    maha = np.sum(y * y, axis=-1)
    loglik = -0.5 * (R * np.log(2.0 * np.pi) + logdet + maha)
    return loglik, y


def posterior_moments_joseph(
    L_eq: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    Sigma: np.ndarray,
    V: np.ndarray,
    full_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conditional posterior mean and covariance via Woodbury/Joseph form.

    Computes for `C = Sigma + V = (D L_eq) (D L_eq)^T`:
    - `mu = Sigma C^{-1} b = W^T y`, where `W = L_eq^{-1} D^{-1} Sigma`
    - `K = Sigma C^{-1}`
    - `Psi = (I - K) Sigma (I - K)^T + K V K^T` (symmetrized, PSD-safe)

    If `full_cov=False`, returns `(mu, diag(Psi))` of shapes `(..., R)` and `(..., R)`.
    If `full_cov=True`, returns `(mu, Psi)` of shapes `(..., R)` and `(..., R, R)`.
    """
    R = Sigma.shape[-1]
    Sigma_eq = Sigma / d[..., :, None]
    W = np.linalg.solve(L_eq, Sigma_eq)
    mu = np.einsum("...ji,...j->...i", W, y)

    Z = np.linalg.solve(np.swapaxes(L_eq, -1, -2), W)
    K = np.swapaxes(Z, -1, -2) / d[..., None, :]
    I_minus_K = np.eye(R, dtype=np.float64) - K

    A = I_minus_K @ Sigma
    B = K @ V
    if full_cov:
        Psi = A @ np.swapaxes(I_minus_K, -1, -2) + B @ np.swapaxes(K, -1, -2)
        Psi = symmetrize(Psi)
        return mu, Psi

    Psi_diag = np.sum(A * I_minus_K, axis=-1) + np.sum(B * K, axis=-1)
    Psi_diag = np.maximum(Psi_diag, 0.0)
    return mu, Psi_diag
