"""Wakefield Approximate Bayes Factors (lnABF) for fgwas.

References
----------
Wakefield, J. (2009). Bayes factors for genome-wide association studies: comparison
with P-values. Genetic Epidemiology, 33(1), 79-86.

Pickrell, J. K. (2014). Joint analysis of functional genomic data and genome-wide
association studies of 18 human traits. AJHG, 94(4), 559-573.
https://doi.org/10.1016/j.ajhg.2014.03.004
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.special import logsumexp

SettingsType = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsType:
    """Validate the ``settings`` mode parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


def normalize_prior_variances(
    V: float | Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    """Validate and normalize Wakefield prior variance(s) ``V`` into a tuple of floats."""
    if isinstance(V, (int, float, np.floating)):
        v_list = [float(V)]
    else:
        v_list = [float(w) for w in V]
    if len(v_list) == 0:
        raise ValueError("Prior variance V must contain at least one positive value.")
    for w in v_list:
        if not np.isfinite(w) or w <= 0.0:
            raise ValueError(f"All prior variances in V must be finite and > 0, got {V!r}.")
    return tuple(v_list)


def wakefield_lnabf(
    beta: np.ndarray | Sequence[float] | float,
    se: np.ndarray | Sequence[float] | float,
    V: float | Sequence[float] = (0.01, 0.1, 0.5),
    settings: SettingsType = "default",
) -> np.ndarray:
    """Compute natural-log Wakefield Approximate Bayes Factor ``ln(ABF)`` per SNP.

    Returns ``ln(P(data | H_1) / P(data | H_0))`` so positive values indicate
    support for genetic association. When ``V`` contains multiple prior variances
    ``(W_1, ..., W_M)``, computes ``log(mean_m ABF(W_m))`` via ``logsumexp``.
    """
    validate_settings(settings)
    scalar_input = np.ndim(beta) == 0 and np.ndim(se) == 0
    beta_arr = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    se_arr = np.atleast_1d(np.asarray(se, dtype=np.float64))

    if beta_arr.ndim != 1 or se_arr.ndim != 1 or beta_arr.shape[0] != se_arr.shape[0]:
        raise ValueError(
            f"beta and se must be 1D arrays of matching length, got {beta_arr.shape} and {se_arr.shape}."
        )
    if beta_arr.shape[0] == 0:
        raise ValueError("beta and se must be non-empty.")
    if not (np.all(np.isfinite(beta_arr)) and np.all(np.isfinite(se_arr))):
        raise ValueError("beta and se must contain only finite values.")
    if np.any(se_arr <= 0.0):
        raise ValueError("Standard errors se must be strictly positive.")

    w_tuple = normalize_prior_variances(V)
    v_var = se_arr * se_arr
    z_sq = (beta_arr / se_arr) ** 2

    if len(w_tuple) == 1:
        w = w_tuple[0]
        r = w / (v_var + w)
        ln_abf = 0.5 * np.log(v_var / (v_var + w)) + 0.5 * z_sq * r
    else:
        w_vec = np.asarray(w_tuple, dtype=np.float64)[:, None]  # (M, 1)
        v_row = v_var[None, :]  # (1, N)
        r_mat = w_vec / (v_row + w_vec)  # (M, N)
        ln_stack = 0.5 * np.log(v_row / (v_row + w_vec)) + 0.5 * z_sq[None, :] * r_mat
        ln_abf = logsumexp(ln_stack, axis=0) - np.log(float(len(w_tuple)))

    return np.asarray(ln_abf[0] if scalar_input else ln_abf, dtype=np.float64)
