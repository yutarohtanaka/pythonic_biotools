"""Shared validation, LD matrix regularization, and statistical helpers for cismr."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import linalg, stats

SettingsType = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsType:
    """Validate the ``settings`` mode parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


def as_1d_float(x: np.ndarray | list[float], name: str, min_len: int = 1) -> np.ndarray:
    """Convert input to a contiguous 1D float64 array and check finiteness."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    if arr.shape[0] < min_len:
        raise ValueError(
            f"{name} must have length >= {min_len}, got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def as_2d_float(
    x: np.ndarray | list[list[float]], name: str, min_rows: int = 1
) -> np.ndarray:
    """Convert input to a 2D float64 array and check finiteness."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape}.")
    if arr.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have at least {min_rows} rows, got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def prepare_ld_matrix(
    ld_R: np.ndarray | None,
    p: int,
    settings: SettingsType = "default",
    min_eig: float = 1e-6,
) -> np.ndarray:
    """Validate and optionally PSD-project an LD correlation matrix ``R``.

    In ``settings="default"``, projects ``R`` onto the space of positive
    definite correlation matrices with minimum eigenvalue ``min_eig``.
    In ``settings="R_default"``, symmetrizes ``R`` without eigenvalue clamping
    to preserve exact upstream ``MendelianRandomization::mr_pcgmm`` behavior.
    """
    validate_settings(settings)
    if ld_R is None:
        return np.eye(p, dtype=np.float64)
    R = np.asarray(ld_R, dtype=np.float64)
    if R.shape != (p, p):
        raise ValueError(f"ld_R must have shape ({p}, {p}), got {R.shape}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("ld_R contains non-finite values.")
    R = 0.5 * (R + R.T)
    if settings == "R_default":
        return R

    eigvals, eigvecs = linalg.eigh(R)
    clamped = np.maximum(eigvals, min_eig)
    R_psd = (eigvecs * clamped) @ eigvecs.T
    diag = np.diag(R_psd)
    inv_sd = 1.0 / np.sqrt(np.maximum(diag, min_eig))
    R_corr = (R_psd * inv_sd[:, None]) * inv_sd[None, :]
    R_corr = 0.5 * (R_corr + R_corr.T)
    np.fill_diagonal(R_corr, 1.0)
    # Ensure strict positive definiteness after unit-diagonal rescaling
    eigvals2, eigvecs2 = linalg.eigh(R_corr)
    if np.min(eigvals2) < 1e-8:
        R_corr = (1.0 - min_eig) * (
            (eigvecs2 * np.maximum(eigvals2, 0.0)) @ eigvecs2.T
        )
        diag2 = np.diag(R_corr)
        inv_sd2 = 1.0 / np.sqrt(np.maximum(diag2, 1e-12))
        R_corr = (1.0 - min_eig) * ((R_corr * inv_sd2[:, None]) * inv_sd2[None, :])
        np.fill_diagonal(R_corr, 1.0)
    return R_corr


def sym_solve(mat: np.ndarray, rhs: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Solve ``mat @ X = rhs`` for symmetric PSD/PD ``mat`` with safe fallback."""
    sym = 0.5 * (mat + mat.T)
    try:
        cho, lower = linalg.cho_factor(sym, lower=True, check_finite=False)
        return linalg.cho_solve((cho, lower), rhs, check_finite=False)
    except linalg.LinAlgError:
        k = sym.shape[0]
        scale = max(float(np.trace(sym)) / max(k, 1), 1e-12)
        sym_reg = sym + (ridge * scale) * np.eye(k, dtype=np.float64)
        try:
            cho, lower = linalg.cho_factor(sym_reg, lower=True, check_finite=False)
            return linalg.cho_solve((cho, lower), rhs, check_finite=False)
        except linalg.LinAlgError:
            return linalg.lstsq(sym_reg, rhs, check_finite=False)[0]


def wald_inference(
    estimate: float,
    se: float,
    alpha: float = 0.05,
    df: int | None = None,
) -> tuple[float, float, float]:
    """Compute two-sided p-value and ``(1 - alpha)`` confidence interval."""
    if not np.isfinite(se) or se <= 0.0:
        return np.nan, np.nan, np.nan
    stat = abs(estimate) / se
    if df is not None and df > 0:
        pval = float(2.0 * stats.t.sf(stat, df=df))
        crit = float(stats.t.ppf(1.0 - 0.5 * alpha, df=df))
    else:
        pval = float(2.0 * stats.norm.sf(stat))
        crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))
    return pval, float(estimate - crit * se), float(estimate + crit * se)
