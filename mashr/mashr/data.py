"""Data container, validation, alpha-scaling, and null correlation estimation for `mashr`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ._linalg import symmetrize


@dataclass(frozen=True)
class MashData:
    """Immutable container for multivariate summary statistics and residual covariance.

    Stores observed effect sizes ``Bhat`` `(J, R)`, standard errors ``Shat`` `(J, R)`,
    residual correlation/covariance ``V`` `(R, R)`, and the ``alpha`` power parameter
    controlling EE (``alpha=0.0``) vs EZ (``alpha=1.0``) modeling.
    """

    Bhat: np.ndarray
    Shat: np.ndarray
    V: np.ndarray
    alpha: float
    feature_names: list[str]
    condition_names: list[str]

    @property
    def n_effects(self) -> int:
        """Number of effects/units J."""
        return int(self.Bhat.shape[0])

    @property
    def n_conditions(self) -> int:
        """Number of conditions R."""
        return int(self.Bhat.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple (J, R)."""
        return (int(self.Bhat.shape[0]), int(self.Bhat.shape[1]))

    @property
    def z_scores(self) -> np.ndarray:
        """Elementwise Z-scores Bhat / Shat of shape (J, R)."""
        return self.Bhat / self.Shat

    @property
    def Bhat_tilde(self) -> np.ndarray:
        """Alpha-transformed effect estimates Bhat / (Shat ** alpha), shape (J, R)."""
        if self.alpha == 0.0:
            return self.Bhat
        return self.Bhat / (self.Shat ** self.alpha)

    @property
    def Shat_tilde(self) -> np.ndarray:
        """Alpha-transformed standard errors Shat ** (1 - alpha), shape (J, R)."""
        if self.alpha == 0.0:
            return self.Shat
        if self.alpha == 1.0:
            return np.ones_like(self.Shat, dtype=np.float64)
        return self.Shat ** (1.0 - self.alpha)

    @property
    def scale_factor(self) -> np.ndarray:
        """Multiplicative back-transformation factor Shat ** alpha, shape (J, R)."""
        if self.alpha == 0.0:
            return np.ones_like(self.Shat, dtype=np.float64)
        if self.alpha == 1.0:
            return self.Shat
        return self.Shat ** self.alpha

    @property
    def is_common_cov(self) -> bool:
        """True if all units share the same transformed residual covariance V_tilde_j."""
        if self.alpha == 1.0 or self.Bhat.shape[0] <= 1:
            return True
        st = self.Shat_tilde
        return bool(np.allclose(st, st[0:1, :], rtol=1e-12, atol=0.0))

    @property
    def is_diagonal_v(self) -> bool:
        """True if V is diagonal within numerical tolerance."""
        if self.V.shape[0] <= 1:
            return True
        diag_v = np.maximum(np.diag(self.V), 0.0)
        off_diag = self.V - np.diag(np.diag(self.V))
        scale_mat = np.sqrt(np.outer(diag_v, diag_v))
        return bool(np.all(np.abs(off_diag) <= 1e-12 * scale_mat))

    def get_cov(self, j: int) -> np.ndarray:
        """Return transformed residual covariance V_tilde_j = diag(s_tilde_j) V diag(s_tilde_j)."""
        if j < 0 or j >= self.n_effects:
            raise IndexError(f"Unit index {j} out of range for J={self.n_effects}.")
        st = self.Shat_tilde[j]
        cov = (st[:, None] * self.V) * st[None, :]
        return symmetrize(cov)


def _extract_matrix_and_names(
    obj: np.ndarray | pd.DataFrame,
    name: str,
) -> tuple[np.ndarray, list[str] | None, list[str] | None, bool]:
    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy(dtype=np.float64, copy=True)
        row_names = [str(x) for x in obj.index]
        col_names = [str(x) for x in obj.columns]
        return arr, row_names, col_names, True
    arr = np.asarray(obj, dtype=np.float64).copy()
    return arr, None, None, False


def mash_set_data(
    Bhat: np.ndarray | pd.DataFrame,
    Shat: np.ndarray | pd.DataFrame | float | None = None,
    alpha: float = 0.0,
    V: np.ndarray | pd.DataFrame | None = None,
    feature_names: Sequence[str] | None = None,
    condition_names: Sequence[str] | None = None,
    zero_shat_reset: float | None = None,
) -> MashData:
    """Create and validate an immutable :class:`MashData` container.

    Parameters
    ----------
    Bhat : array-like or DataFrame of shape (J, R)
        Observed effect estimates (or Z-scores when ``Shat=None``).
    Shat : array-like, DataFrame, float, or None, default=None
        Standard errors matching shape `(J, R)`. Defaults to ``1.0`` everywhere.
    alpha : float, default=0.0
        Scale transformation parameter: ``0.0`` for EE (exchangeable effects),
        ``1.0`` for EZ (exchangeable Z-scores).
    V : array-like or DataFrame of shape (R, R), or None, default=None
        Symmetric positive-definite residual correlation/covariance matrix.
        Defaults to the identity matrix ``I_R``.
    feature_names : sequence of str, optional
        Optional unit/feature labels of length ``J``.
    condition_names : sequence of str, optional
        Optional condition labels of length ``R``.
    zero_shat_reset : float, optional
        If provided and positive, entries where ``Shat == 0`` (and ``Bhat == 0``)
        are reset to this value instead of raising ``ValueError``.
    """
    b_arr, b_rows, b_cols, b_is_df = _extract_matrix_and_names(Bhat, "Bhat")
    if b_arr.ndim != 2:
        raise ValueError(f"Bhat must be a 2D array of shape (J, R); got ndim={b_arr.ndim}.")
    J, R = b_arr.shape
    if J < 1 or R < 1:
        raise ValueError(f"Bhat must be non-empty; got shape ({J}, {R}).")
    if not np.all(np.isfinite(b_arr)):
        raise ValueError("Bhat contains NaN or Inf values.")

    alpha_val = float(alpha)
    if not np.isfinite(alpha_val):
        raise ValueError(f"alpha must be finite; got {alpha}.")

    if Shat is None:
        s_arr = np.ones((J, R), dtype=np.float64)
    elif np.ndim(Shat) == 0:
        s_scalar = float(Shat)
        if not np.isfinite(s_scalar):
            raise ValueError(f"Scalar Shat must be finite; got {Shat}.")
        s_arr = np.full((J, R), s_scalar, dtype=np.float64)
    else:
        s_arr, s_rows, s_cols, s_is_df = _extract_matrix_and_names(Shat, "Shat")  # type: ignore[arg-type]
        if s_arr.shape != (J, R):
            raise ValueError(f"Shat shape {s_arr.shape} does not match Bhat shape {(J, R)}.")
        if b_is_df and s_is_df:
            if b_rows != s_rows or b_cols != s_cols:
                raise ValueError("Bhat and Shat DataFrames must have identical index and columns.")
        if not np.all(np.isfinite(s_arr)):
            raise ValueError("Shat contains NaN or Inf values.")

    if np.any(s_arr < 0.0):
        raise ValueError("Shat must not contain negative values.")

    zero_mask = s_arr == 0.0
    if np.any(zero_mask):
        if zero_shat_reset is None:
            raise ValueError("Shat must be strictly positive (> 0); found zero entries.")
        reset_val = float(zero_shat_reset)
        if not np.isfinite(reset_val) or reset_val <= 0.0:
            raise ValueError(f"zero_shat_reset must be finite and > 0; got {zero_shat_reset}.")
        if np.any(b_arr[zero_mask] != 0.0):
            raise ValueError("Found Shat == 0 with non-zero Bhat; cannot safely reset Shat.")
        s_arr[zero_mask] = reset_val

    if not np.all(s_arr > 0.0):
        raise ValueError("Shat must be strictly positive (> 0).")

    if feature_names is not None:
        feat_list = [str(x) for x in feature_names]
        if len(feat_list) != J:
            raise ValueError(f"feature_names length {len(feat_list)} does not match J={J}.")
    elif b_rows is not None:
        feat_list = b_rows
    else:
        feat_list = [f"effect_{j}" for j in range(J)]

    if condition_names is not None:
        cond_list = [str(x) for x in condition_names]
        if len(cond_list) != R:
            raise ValueError(f"condition_names length {len(cond_list)} does not match R={R}.")
    elif b_cols is not None:
        cond_list = b_cols
    else:
        cond_list = [f"cond_{r}" for r in range(R)]

    if V is None:
        v_arr = np.eye(R, dtype=np.float64)
    else:
        v_arr, v_rows, v_cols, v_is_df = _extract_matrix_and_names(V, "V")
        if v_arr.shape != (R, R):
            raise ValueError(f"V must have shape ({R}, {R}); got {v_arr.shape}.")
        if v_is_df and (b_cols is not None or condition_names is not None):
            if v_rows != cond_list or v_cols != cond_list:
                raise ValueError("V DataFrame index/columns must match condition_names.")
        if not np.all(np.isfinite(v_arr)):
            raise ValueError("V contains NaN or Inf values.")
        max_asym = float(np.max(np.abs(v_arr - v_arr.T)))
        v_scale = float(np.max(np.abs(v_arr)))
        if v_scale > 0.0 and max_asym > 1e-8 * v_scale:
            raise ValueError(
                f"V must be symmetric (max |V - V^T| = {max_asym:.3e} > 1e-8 * {v_scale:.3e})."
            )
        v_arr = symmetrize(v_arr)
        try:
            np.linalg.cholesky(v_arr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("V must be strictly positive definite.") from exc

    return MashData(
        Bhat=b_arr,
        Shat=s_arr,
        V=v_arr,
        alpha=alpha_val,
        feature_names=feat_list,
        condition_names=cond_list,
    )


def mash_update_data(
    data: MashData,
    V: np.ndarray | pd.DataFrame | None = None,
    alpha: float | None = None,
) -> MashData:
    """Return a new :class:`MashData` with updated ``V`` and/or ``alpha``."""
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    new_v = data.V if V is None else V
    new_alpha = data.alpha if alpha is None else alpha
    return mash_set_data(
        Bhat=data.Bhat,
        Shat=data.Shat,
        alpha=new_alpha,
        V=new_v,
        feature_names=data.feature_names,
        condition_names=data.condition_names,
    )


def _cov2cor(mat: np.ndarray) -> np.ndarray:
    """Convert an SPD covariance matrix to a unit-diagonal correlation matrix."""
    d = np.sqrt(np.maximum(np.diag(mat), 1e-300))
    cor = mat / np.outer(d, d)
    cor = symmetrize(cor)
    np.fill_diagonal(cor, 1.0)
    return cor


def estimate_null_correlation_simple(
    data: MashData,
    z_thresh: float = 2.0,
    est_cor: bool = True,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Estimate the R x R residual correlation (or covariance) matrix V from near-null units.

    Selects units with ``max_r |Z_{j,r}| < z_thresh``, computes the sample covariance
    (and optionally normalizes to a correlation matrix), and projects eigenvalues
    onto ``[ridge, +inf)`` to guarantee strict positive definiteness.
    """
    if not isinstance(data, MashData):
        raise TypeError("data must be a MashData instance.")
    if not np.isfinite(z_thresh) or z_thresh <= 0.0:
        raise ValueError(f"z_thresh must be finite and > 0; got {z_thresh}.")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError(f"ridge must be finite and > 0; got {ridge}.")

    z = data.z_scores
    null_mask = np.max(np.abs(z), axis=1) < float(z_thresh)
    n_null = int(np.sum(null_mask))
    if n_null < 2:
        raise ValueError(
            f"Need at least 2 null units with max(|Z|) < {z_thresh}; found {n_null}."
        )

    z_null = z[null_mask, :]
    c0 = np.atleast_2d(np.cov(z_null, rowvar=False)).astype(np.float64, copy=False)
    v_est = _cov2cor(c0) if est_cor else symmetrize(c0)

    evals, evecs = np.linalg.eigh(v_est)
    if np.min(evals) < ridge:
        target = float(ridge)
        for _ in range(3):
            evals = np.maximum(evals, target)
            v_est = symmetrize((evecs * evals) @ evecs.T)
            if est_cor:
                max_d = float(np.max(np.diag(v_est)))
                v_est = _cov2cor(v_est)
                evals, evecs = np.linalg.eigh(v_est)
                if np.min(evals) >= ridge - 1e-15:
                    break
                target = float(ridge) * max(max_d, 1.0) * 1.01
            else:
                break

    return v_est
