"""15-node Gauss-Kronrod quadrature and B-spline log-baseline hazard basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

SettingsMode = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsMode:
    """Validate ``settings`` parameter."""
    if settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


# Exact 15-node Gauss-Kronrod rule on [-1, 1] (QUADPACK qk15 / JMbayes2::gaussKronrod(15)),
# exact for polynomials up to degree 22.
GK15_NODES: np.ndarray = np.array(
    [
        -0.99145537112081263921,
        -0.94910791234275852453,
        -0.86486442335976907279,
        -0.74153118559939443986,
        -0.58608723546769113029,
        -0.40584515137739716691,
        -0.20778495500789846760,
        0.0,
        0.20778495500789846760,
        0.40584515137739716691,
        0.58608723546769113029,
        0.74153118559939443986,
        0.86486442335976907279,
        0.94910791234275852453,
        0.99145537112081263921,
    ],
    dtype=np.float64,
)

GK15_WEIGHTS: np.ndarray = np.array(
    [
        0.02293532201052922496,
        0.06309209262997855329,
        0.10479001032225018384,
        0.14065325971552591875,
        0.16900472663926790283,
        0.19035057806478540991,
        0.20443294007529889241,
        0.20948214108472782801,
        0.20443294007529889241,
        0.19035057806478540991,
        0.16900472663926790283,
        0.14065325971552591875,
        0.10479001032225018384,
        0.06309209262997855329,
        0.02293532201052922496,
    ],
    dtype=np.float64,
)


def gauss_kronrod_15(
    a: float | np.ndarray,
    b: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 15-node Gauss-Kronrod abscissae and weights to interval(s) ``[a, b]``.

    Parameters
    ----------
    a, b : Scalar floats or 1D arrays of shape ``(N,)``.

    Returns
    -------
    nodes, weights : Shape ``(15,)`` for scalar endpoints or ``(N, 15)`` for 1D arrays.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.ndim == 0 and b_arr.ndim == 0:
        half = 0.5 * (float(b_arr) - float(a_arr))
        mid = 0.5 * (float(a_arr) + float(b_arr))
        return mid + half * GK15_NODES, half * GK15_WEIGHTS

    a_1d = np.broadcast_to(a_arr, np.broadcast_shapes(a_arr.shape, b_arr.shape))
    b_1d = np.broadcast_to(b_arr, a_1d.shape)
    half = 0.5 * (b_1d - a_1d)
    mid = 0.5 * (a_1d + b_1d)
    nodes = mid[..., None] + half[..., None] * GK15_NODES
    weights = half[..., None] * GK15_WEIGHTS
    return nodes, weights


@dataclass(frozen=True)
class BSplineHazardBasis:
    """B-spline basis and second-order difference penalty for log-baseline hazard."""

    knots: np.ndarray
    degree: int
    t_min: float
    t_max: float
    n_basis: int
    penalty_matrix: np.ndarray
    penalty_rank: int

    @classmethod
    def from_times(
        cls,
        surv_times: np.ndarray,
        event_mask: np.ndarray,
        n_knots: int = 6,
        degree: int = 3,
    ) -> BSplineHazardBasis:
        """Construct B-spline basis from observed survival and event times."""
        if n_knots < 1:
            raise ValueError(f"n_knots must be >= 1, got {n_knots}.")
        t_min = 0.0
        t_max = float(np.max(surv_times)) * 1.02
        if t_max <= 0.0:
            t_max = 1.0

        ev_times = surv_times[event_mask]
        ref_times = ev_times if ev_times.size >= 3 else surv_times
        ref_pos = ref_times[(ref_times > t_min) & (ref_times < t_max)]

        if ref_pos.size >= n_knots:
            probs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
            interior = np.quantile(ref_pos, probs)
            interior = np.unique(interior)
            if interior.size < n_knots:
                interior = np.linspace(t_min, t_max, n_knots + 2)[1:-1]
        else:
            interior = np.linspace(t_min, t_max, n_knots + 2)[1:-1]

        knots = np.concatenate(
            [
                np.repeat(t_min, degree + 1),
                interior,
                np.repeat(t_max, degree + 1),
            ]
        )
        n_basis = knots.shape[0] - degree - 1

        # Second-order difference penalty D2^T D2
        if n_basis >= 3:
            d2 = np.diff(np.eye(n_basis, dtype=np.float64), n=2, axis=0)
            pen = d2.T @ d2
            pen_rank = n_basis - 2
        else:
            d1 = np.diff(np.eye(n_basis, dtype=np.float64), n=1, axis=0)
            pen = d1.T @ d1
            pen_rank = max(n_basis - 1, 1)

        return cls(
            knots=knots,
            degree=degree,
            t_min=t_min,
            t_max=t_max,
            n_basis=n_basis,
            penalty_matrix=pen,
            penalty_rank=pen_rank,
        )

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate B-spline basis at arbitrary array ``t``, returning ``(*t.shape, n_basis)``."""
        t_arr = np.asarray(t, dtype=np.float64)
        orig_shape = t_arr.shape
        t_clamped = np.clip(t_arr.ravel(), self.t_min, self.t_max)
        design = BSpline.design_matrix(t_clamped, self.knots, self.degree).toarray()
        return design.reshape(*orig_shape, self.n_basis)
