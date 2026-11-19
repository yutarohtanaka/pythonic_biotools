"""2D spatial spline basis construction for CSIDE non-parametric spatial DE."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate the ``settings`` mode parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """2D thin-plate spline Green's function r^2 log(r) / (8 pi)."""
    out = np.zeros_like(r, dtype=np.float64)
    mask = r > 0.0
    if np.any(mask):
        rm = r[mask]
        out[mask] = (rm * rm) * np.log(rm) / (8.0 * np.pi)
    return out


def _scale_non_intercept_columns(
    cols: np.ndarray, settings: SettingsMode
) -> np.ndarray:
    """Center and scale non-intercept basis columns to O(1) spatial span."""
    if cols.shape[1] == 0:
        return cols
    centered = cols - np.mean(cols, axis=0, keepdims=True)
    if settings == "R_default":
        # Match spacexr::get_spline_matrix column scaling (unit max-abs / SD scale)
        denom = np.max(np.abs(centered), axis=0, keepdims=True)
        denom = np.where(denom > 1e-12, denom, 1.0)
        return centered / denom
    span = np.ptp(centered, axis=0, keepdims=True)
    span = np.where(span > 1e-12, span, 1.0)
    return centered / span


def _build_thin_plate_basis(
    xy: np.ndarray, df: int, settings: SettingsMode
) -> np.ndarray:
    """Construct Wood (2003) 2D thin-plate regression spline basis of rank ``df``."""
    n = xy.shape[0]
    # Center and isotropically scale coordinates
    center = np.mean(xy, axis=0, keepdims=True)
    scale = float(np.std(xy))
    if scale < 1e-12:
        raise ValueError("Spatial coordinates have zero variance.")
    uv = (xy - center) / scale

    # Linear polynomial space: [1, u, v]
    t_full = np.column_stack([np.ones(n, dtype=np.float64), uv[:, 0], uv[:, 1]])
    q_full, r_mat = np.linalg.qr(t_full, mode="reduced")
    rank_poly = int(np.sum(np.abs(np.diag(r_mat)) > 1e-10))
    if rank_poly < 2:
        raise ValueError("Spatial coordinates are degenerate (constant).")

    # Extract linearly independent linear columns beyond intercept
    linear_cols = uv[:, :2] if rank_poly == 3 else uv[:, [0]]
    n_linear = linear_cols.shape[1]
    n_wiggly = max(0, df - 1 - n_linear)

    if n_wiggly == 0:
        keep_lin = min(df - 1, n_linear)
        scaled_lin = _scale_non_intercept_columns(linear_cols[:, :keep_lin], settings)
        return np.column_stack([np.ones(n, dtype=np.float64), scaled_lin])

    max_knots = 500
    if n <= max_knots:
        knots = uv
        e_kk = _tps_kernel(cdist(knots, knots))
        # Project out polynomial null space: (I - Q Q^T) E (I - Q Q^T)
        proj_e = e_kk - q_full @ (q_full.T @ e_kk)
        proj_e = proj_e - (proj_e @ q_full) @ q_full.T
        proj_e = 0.5 * (proj_e + proj_e.T)

        eigvals, eigvecs = np.linalg.eigh(proj_e)
        order = np.argsort(np.abs(eigvals))[::-1]
        k_keep = min(n_wiggly, len(order))
        wiggly = eigvecs[:, order[:k_keep]]
    else:
        km = KMeans(n_clusters=max_knots, random_state=0, n_init=3)
        km.fit(uv)
        knots = km.cluster_centers_
        m = knots.shape[0]
        t_knots = np.column_stack(
            [np.ones(m, dtype=np.float64), knots[:, 0], knots[:, 1]]
        )
        q_k, _ = np.linalg.qr(t_knots, mode="complete")
        z_null = q_k[:, rank_poly:]  # (m, m - rank_poly)

        e_kk = _tps_kernel(cdist(knots, knots))
        zed = z_null.T @ e_kk @ z_null
        zed = 0.5 * (zed + zed.T)
        eigvals, eigvecs = np.linalg.eigh(zed)
        order = np.argsort(np.abs(eigvals))[::-1]
        valid = [idx for idx in order if abs(eigvals[idx]) > 1e-10]
        k_keep = min(n_wiggly, len(valid))
        sel = valid[:k_keep]

        e_nk = _tps_kernel(cdist(uv, knots))
        raw_wiggly = e_nk @ (z_null @ (eigvecs[:, sel] / eigvals[sel]))
        # Orthogonalize against polynomial null space on full spots
        wiggly = raw_wiggly - q_full @ (q_full.T @ raw_wiggly)

    all_non_intercept = np.column_stack([linear_cols, wiggly])
    # Orthogonalize wiggly columns cleanly against linear columns via QR if R_default
    if settings == "R_default":
        q_ni, _ = np.linalg.qr(
            all_non_intercept - np.mean(all_non_intercept, axis=0, keepdims=True),
            mode="reduced",
        )
        all_non_intercept = q_ni[:, : df - 1]
    else:
        all_non_intercept = all_non_intercept[:, : df - 1]

    scaled = _scale_non_intercept_columns(all_non_intercept, settings)
    return np.column_stack([np.ones(n, dtype=np.float64), scaled])


def _univariate_bspline_basis(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """Construct clamped 1D B-spline basis of ``n_basis`` columns on ``x``."""
    deg = min(degree, n_basis - 1)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < 1e-12:
        return np.ones((len(x), 1), dtype=np.float64)
    # Clamped knot vector
    n_interior = max(0, n_basis - deg - 1)
    if n_interior > 0:
        quantiles = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        interior = np.quantile(x, quantiles)
        # Guard against duplicate interior knots equal to boundary
        interior = np.clip(interior, xmin + 1e-8 * (xmax - xmin), xmax - 1e-8 * (xmax - xmin))
    else:
        interior = np.array([], dtype=np.float64)

    knots = np.concatenate(
        [
            np.repeat(xmin, deg + 1),
            interior,
            np.repeat(xmax, deg + 1),
        ]
    )
    xc = np.clip(x, xmin, xmax)
    mat = BSpline.design_matrix(xc, knots, deg).toarray()
    return mat


def _build_tensor_bspline_basis(
    xy: np.ndarray, df: int, settings: SettingsMode
) -> np.ndarray:
    """Construct 2D tensor-product cubic B-spline basis reduced to ``df`` columns."""
    n = xy.shape[0]
    m_1d = max(3, int(np.ceil(np.sqrt(df + 1))))
    bx = _univariate_bspline_basis(xy[:, 0], n_basis=m_1d, degree=3)
    by = _univariate_bspline_basis(xy[:, 1], n_basis=m_1d, degree=3)

    # Row-wise Kronecker product: shape (n, bx.shape[1] * by.shape[1])
    tensor = (bx[:, :, None] * by[:, None, :]).reshape(n, -1)
    centered = tensor - np.mean(tensor, axis=0, keepdims=True)

    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    valid = s > 1e-10
    u_valid = u[:, valid]
    k_keep = min(df - 1, u_valid.shape[1])
    if k_keep == 0:
        raise ValueError("Tensor B-spline basis collapsed to rank 0; check coordinates.")

    basis_ni = _scale_non_intercept_columns(u_valid[:, :k_keep], settings)
    return np.column_stack([np.ones(n, dtype=np.float64), basis_ni])


def build_spline_design(
    coords: np.ndarray | pd.DataFrame,
    df: int = 15,
    basis: Literal["thin_plate", "bspline"] = "thin_plate",
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Build a 2D smooth spatial regression spline design matrix for CSIDE.

    Parameters
    ----------
    coords
        ``(n_spots, 2)`` array or DataFrame of spatial ``(x, y)`` coordinates.
    df
        Total degrees of freedom (including the intercept column). Must satisfy
        ``2 <= df <= n_spots``.
    basis
        Spline family: ``"thin_plate"`` (Wood 2003 2D thin-plate regression
        splines) or ``"bspline"`` (2D tensor-product cubic B-splines).
    settings
        ``"default"`` (Pythonic unit-span basis scaling) or ``"R_default"``
        (``spacexr`` QR-orthogonalized spline scaling).

    Returns
    -------
    pd.DataFrame
        ``(n_spots, df_eff)`` design matrix whose first column is ``"Intercept"``
        (all ones) followed by centered smooth spatial basis functions.
    """
    validate_settings(settings)

    if isinstance(coords, pd.DataFrame):
        index = coords.index
        xy = coords.to_numpy(dtype=np.float64)
    else:
        xy = np.asarray(coords, dtype=np.float64)
        index = pd.RangeIndex(xy.shape[0])

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (n_spots, 2); got {xy.shape}."
        )
    if not np.all(np.isfinite(xy)):
        raise ValueError("coords must contain only finite numeric values.")

    n_spots = xy.shape[0]
    if df < 2:
        raise ValueError(f"df must be >= 2 (intercept + >=1 spatial mode); got {df}.")
    if df > n_spots:
        raise ValueError(
            f"df ({df}) cannot exceed the number of spatial spots ({n_spots})."
        )

    if basis == "thin_plate":
        design = _build_thin_plate_basis(xy, df=df, settings=settings)
    elif basis == "bspline":
        design = _build_tensor_bspline_basis(xy, df=df, settings=settings)
    else:
        raise ValueError(
            f"Unknown basis={basis!r}. Expected 'thin_plate' or 'bspline'."
        )

    n_cols = design.shape[1]
    columns = ["Intercept"] + [f"spline_{i}" for i in range(1, n_cols)]
    return pd.DataFrame(design, index=index, columns=columns)
