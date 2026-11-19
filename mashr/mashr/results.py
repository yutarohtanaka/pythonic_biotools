"""Typed result container and downstream analysis helpers for `mashr`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


def _resolve_condition_indices(
    result: MashResult,
    conditions: Sequence[int | str] | None,
) -> list[int]:
    R = len(result.condition_names)
    if conditions is None:
        return list(range(R))

    cond_list = list(conditions)
    if len(cond_list) == 0:
        raise ValueError("conditions must not be empty when specified.")

    name_to_idx = {name: idx for idx, name in enumerate(result.condition_names)}
    indices: list[int] = []
    for item in cond_list:
        if isinstance(item, bool):
            raise ValueError(f"Invalid condition selector {item!r}; expected int or str.")
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0 or idx >= R:
                raise ValueError(f"Condition index {idx} out of bounds for R={R}.")
            indices.append(idx)
        elif isinstance(item, str):
            if item not in name_to_idx:
                raise ValueError(f"Unknown condition name {item!r}.")
            indices.append(name_to_idx[item])
        else:
            raise ValueError(f"Invalid condition selector {item!r}; expected int or str.")
    return indices


@dataclass(frozen=True)
class MashResult:
    """Immutable result container returned by :func:`mash`."""

    feature_names: list[str]
    condition_names: list[str]
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    posterior_cov: np.ndarray | None
    lfsr: np.ndarray
    negative_prob: np.ndarray
    positive_prob: np.ndarray
    zero_prob: np.ndarray
    posterior_weights: np.ndarray
    pi: np.ndarray
    grid: np.ndarray
    U_list: dict[str, np.ndarray]
    loglik_matrix: np.ndarray
    loglik: float
    penalized_loglik: float
    null_loglik: float
    use_pointmass: bool
    alpha: float
    converged: bool
    n_iter: int
    kkt_gap: float
    settings: Literal["default", "R_default"] = "default"

    def to_dataframe(self, sort_by_lfsr: bool = False) -> pd.DataFrame:
        """Return tidy per-(feature, condition) posterior summary as a pandas DataFrame."""
        J = len(self.feature_names)
        R = len(self.condition_names)
        df = pd.DataFrame(
            {
                "feature": np.repeat(self.feature_names, R),
                "condition": np.tile(self.condition_names, J),
                "posterior_mean": self.posterior_mean.ravel(),
                "posterior_sd": self.posterior_sd.ravel(),
                "lfsr": self.lfsr.ravel(),
                "negative_prob": self.negative_prob.ravel(),
                "positive_prob": self.positive_prob.ravel(),
                "zero_prob": self.zero_prob.ravel(),
            }
        )
        if sort_by_lfsr:
            df = df.sort_values("lfsr", kind="mergesort", na_position="last").reset_index(
                drop=True
            )
        return df

    def significant(self, thresh: float = 0.05) -> pd.DataFrame:
        """Return tidy rows with ``lfsr < thresh`` sorted deterministically by ``lfsr``."""
        if not np.isfinite(thresh) or thresh < 0.0 or thresh > 1.0:
            raise ValueError(f"thresh must lie in [0, 1]; got {thresh}.")
        df = self.to_dataframe(sort_by_lfsr=True)
        return df[df["lfsr"] < float(thresh)].reset_index(drop=True)

    def posterior_mean_dataframe(self) -> pd.DataFrame:
        """Return `(J, R)` DataFrame of posterior means."""
        return pd.DataFrame(
            self.posterior_mean,
            index=self.feature_names,
            columns=self.condition_names,
        )

    def posterior_sd_dataframe(self) -> pd.DataFrame:
        """Return `(J, R)` DataFrame of posterior standard deviations."""
        return pd.DataFrame(
            self.posterior_sd,
            index=self.feature_names,
            columns=self.condition_names,
        )

    def lfsr_dataframe(self) -> pd.DataFrame:
        """Return `(J, R)` DataFrame of local false sign rates."""
        return pd.DataFrame(
            self.lfsr,
            index=self.feature_names,
            columns=self.condition_names,
        )

    def estimated_pi_dataframe(self) -> pd.DataFrame:
        """Return mixture weights summed across scale grid for each covariance component."""
        est = get_estimated_pi(self)
        return pd.DataFrame(
            {
                "component": list(est.keys()),
                "weight": list(est.values()),
            }
        )

    def pairwise_sharing_dataframe(
        self,
        factor: float = 0.5,
        lfsr_thresh: float = 0.05,
        conditions: Sequence[int | str] | None = None,
    ) -> pd.DataFrame:
        """Return `(R', R')` pairwise effect-sharing matrix as a labeled DataFrame."""
        cond_idx = _resolve_condition_indices(self, conditions)
        names = [self.condition_names[i] for i in cond_idx]
        mat = get_pairwise_sharing(
            self,
            factor=factor,
            lfsr_thresh=lfsr_thresh,
            conditions=conditions,
        )
        return pd.DataFrame(mat, index=names, columns=names)

    def summary(self, lfsr_thresh: float = 0.05) -> str:
        """Return a human-readable multi-line summary of the fitted MASH model."""
        if not np.isfinite(lfsr_thresh) or lfsr_thresh < 0.0 or lfsr_thresh > 1.0:
            raise ValueError(f"lfsr_thresh must lie in [0, 1]; got {lfsr_thresh}.")
        J = len(self.feature_names)
        R = len(self.condition_names)
        K = len(self.U_list)
        L = len(self.grid)
        est_pi = get_estimated_pi(self)
        null_pi = est_pi.get("null", 0.0)
        n_sig = int(get_significant_results(self, thresh=lfsr_thresh).size)
        non_null = sorted(
            ((k, v) for k, v in est_pi.items() if k != "null"),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_str = ", ".join(f"{k}: {v:.3f}" for k, v in non_null[:5])
        lines = [
            f"MashResult (J={J}, R={R}, K={K}, L={L}, alpha={self.alpha:g})",
            f"  loglik={self.loglik:.4f} | penalized_loglik={self.penalized_loglik:.4f} | null_loglik={self.null_loglik:.4f}",
            f"  converged={self.converged} (n_iter={self.n_iter}, kkt_gap={self.kkt_gap:.3e})",
            f"  null weight (pi_0)={null_pi:.4f} | significant units (lfsr < {lfsr_thresh:g}): {n_sig}/{J}",
            f"  top non-null components: {top_str if top_str else 'none'}",
        ]
        return "\n".join(lines)


def get_pm(result: MashResult) -> np.ndarray:
    """Return posterior mean matrix `(J, R)`."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    return result.posterior_mean


def get_psd(result: MashResult) -> np.ndarray:
    """Return posterior standard deviation matrix `(J, R)`."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    return result.posterior_sd


def get_lfsr(result: MashResult) -> np.ndarray:
    """Return local false sign rate matrix `(J, R)`."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    return result.lfsr


def get_estimated_pi(result: MashResult) -> dict[str, float]:
    """Return mixture weights summed over the scale grid for each covariance component."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    L = int(result.grid.shape[0])
    out: dict[str, float] = {}
    offset = 0
    if result.use_pointmass:
        out["null"] = float(result.pi[0])
        offset = 1
    for name in result.U_list:
        out[name] = float(np.sum(result.pi[offset : offset + L]))
        offset += L
    return out


def get_significant_results(
    result: MashResult,
    thresh: float = 0.05,
    conditions: Sequence[int | str] | None = None,
) -> np.ndarray:
    """Return 1D int array of unit indices with ``min_r lfsr < thresh``, sorted ascending."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    if not np.isfinite(thresh) or thresh < 0.0 or thresh > 1.0:
        raise ValueError(f"thresh must lie in [0, 1]; got {thresh}.")
    cond_idx = _resolve_condition_indices(result, conditions)
    min_lfsr = np.min(result.lfsr[:, cond_idx], axis=1)
    sig_idx = np.flatnonzero(min_lfsr < float(thresh))
    if sig_idx.size == 0:
        return np.empty(0, dtype=np.intp)
    order = np.argsort(min_lfsr[sig_idx], kind="mergesort")
    return sig_idx[order].astype(np.intp, copy=False)


def get_n_significant_conditions(
    result: MashResult,
    thresh: float = 0.05,
    conditions: Sequence[int | str] | None = None,
) -> np.ndarray:
    """Return 1D int array of length J counting conditions with ``lfsr < thresh`` per unit."""
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    if not np.isfinite(thresh) or thresh < 0.0 or thresh > 1.0:
        raise ValueError(f"thresh must lie in [0, 1]; got {thresh}.")
    cond_idx = _resolve_condition_indices(result, conditions)
    return np.sum(result.lfsr[:, cond_idx] < float(thresh), axis=1, dtype=np.intp)


def get_pairwise_sharing(
    result: MashResult,
    factor: float = 0.5,
    lfsr_thresh: float = 0.05,
    conditions: Sequence[int | str] | None = None,
) -> np.ndarray:
    """Compute R' x R' matrix of pairwise effect sharing among significant units.

    For each pair ``(r1, r2)``, considers units significant in at least one of the two
    conditions (``lfsr[:, r1] < lfsr_thresh | lfsr[:, r2] < lfsr_thresh``) and computes
    the fraction having matching sign and magnitude within ``factor``
    (``min(|pm1|, |pm2|) >= factor * max(|pm1|, |pm2|)``). Returns ``np.nan`` if no
    units are significant in either condition.
    """
    if not isinstance(result, MashResult):
        raise TypeError("result must be a MashResult instance.")
    if not np.isfinite(factor) or factor < 0.0 or factor > 1.0:
        raise ValueError(f"factor must lie in [0, 1]; got {factor}.")
    if not np.isfinite(lfsr_thresh) or lfsr_thresh < 0.0 or lfsr_thresh > 1.0:
        raise ValueError(f"lfsr_thresh must lie in [0, 1]; got {lfsr_thresh}.")

    cond_idx = _resolve_condition_indices(result, conditions)
    R_sub = len(cond_idx)
    pm = result.posterior_mean[:, cond_idx]
    lfsr = result.lfsr[:, cond_idx]
    sig = lfsr < float(lfsr_thresh)

    sharing = np.full((R_sub, R_sub), np.nan, dtype=np.float64)
    factor_f = float(factor)

    for i in range(R_sub):
        for j in range(i, R_sub):
            mask = sig[:, i] | sig[:, j]
            n_sig = int(np.sum(mask))
            if n_sig == 0:
                continue
            if i == j:
                sharing[i, i] = 1.0
                continue
            b1 = pm[mask, i]
            b2 = pm[mask, j]
            same_sign = (b1 * b2) > 0.0
            if factor_f == 0.0:
                shared = same_sign
            else:
                abs1 = np.abs(b1)
                abs2 = np.abs(b2)
                shared = same_sign & (
                    np.minimum(abs1, abs2) >= factor_f * np.maximum(abs1, abs2)
                )
            val = float(np.mean(shared))
            sharing[i, j] = val
            sharing[j, i] = val

    return sharing
