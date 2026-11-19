"""
Exact Multinomial Expectation-Maximization (EM) for mutational signature exposure
fitting and global active-signature pre-filtering.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .ccf import COSMIC_96_CONTEXTS, SettingsMode, validate_settings


class MultinomialEMResult(NamedTuple):
    """Result of multinomial EM signature exposure fitting (supports 2-tuple unpacking)."""

    exposures: np.ndarray
    log_likelihood: float


def normalize_signature_matrix(
    signature_matrix: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Validate and column-normalize a ``(C, K)`` mutational signature reference matrix.

    Returns
    -------
    phi : np.ndarray of shape (C, K)
        Column-normalized signature matrix (columns sum to 1).
    sig_names : list[str]
        Signature names of length ``K``.
    context_names : list[str]
        Mutation channel names of length ``C``.
    """
    if isinstance(signature_matrix, pd.DataFrame):
        phi = signature_matrix.to_numpy(dtype=float, copy=True)
        sig_names = [str(c) for c in signature_matrix.columns]
        context_names = [str(i) for i in signature_matrix.index]
    else:
        phi = np.asarray(signature_matrix, dtype=float).copy()
        if phi.ndim != 2:
            raise ValueError(
                f"signature_matrix must be 2D of shape (n_contexts, n_signatures), got shape {phi.shape}."
            )
        n_ctx, n_sig = phi.shape
        sig_names = [f"Sig{k + 1}" for k in range(n_sig)]
        if n_ctx == 96:
            context_names = list(COSMIC_96_CONTEXTS)
        else:
            context_names = [f"ctx_{c}" for c in range(n_ctx)]

    if phi.shape[0] < 2 or phi.shape[1] < 1:
        raise ValueError(
            f"signature_matrix must have >= 2 contexts and >= 1 signature, got {phi.shape}."
        )
    if np.any(~np.isfinite(phi)) or np.any(phi < 0.0):
        raise ValueError("signature_matrix entries must be finite and non-negative.")

    col_sums = np.sum(phi, axis=0)
    if np.any(col_sums <= 0.0):
        raise ValueError("Every signature column in signature_matrix must have positive sum.")
    phi /= col_sums[None, :]
    return phi, sig_names, context_names


def map_mutations_to_channels(
    mutations_df: pd.DataFrame,
    context_names: list[str],
) -> np.ndarray:
    """Map each row of ``mutations_df`` to an integer channel index ``0 .. C - 1``."""
    n_ctx = len(context_names)
    col_candidates = ("context", "trinucleotide", "mutation_type", "channel", "ctx")
    ctx_col = next((c for c in col_candidates if c in mutations_df.columns), None)
    if ctx_col is None:
        raise ValueError(
            f"mutations_df must contain a mutation context column (one of {col_candidates})."
        )

    vals = mutations_df[ctx_col]
    if pd.api.types.is_integer_dtype(vals):
        idx = vals.to_numpy(dtype=int)
        if np.any((idx < 0) | (idx >= n_ctx)):
            raise ValueError(
                f"Integer channel indices in {ctx_col!r} must be in [0, {n_ctx - 1}]."
            )
        return idx

    lookup = {name: i for i, name in enumerate(context_names)}
    str_vals = vals.astype(str).tolist()
    try:
        return np.array([lookup[v] for v in str_vals], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Unknown mutation context {exc.args[0]!r} in column {ctx_col!r}."
        ) from exc


def fit_multinomial_signatures_em(
    counts_96: np.ndarray | pd.Series | list[float],
    signature_matrix: pd.DataFrame | np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    settings: SettingsMode = "default",
) -> MultinomialEMResult:
    """Fit mutational signature exposures via exact Multinomial EM (with SQUAREM acceleration).

    Parameters
    ----------
    counts_96 : array-like of shape (C,) or (B, C)
        Observed mutation counts across ``C`` channels (e.g. 96 trinucleotide contexts).
        If 2D of shape ``(B, C)``, counts are pooled across bins along axis 0.
    signature_matrix : DataFrame or ndarray of shape (C, K)
        Non-negative mutational signature matrix whose columns represent ``K`` signatures.
    tol : float, default 1e-6
        L-infinity convergence tolerance on signature exposure weights ``pi``.
    max_iter : int, default 1000
        Maximum number of EM iterations.
    settings : {"default", "R_default"}, default "default"
        Execution settings mode.

    Returns
    -------
    MultinomialEMResult
        2-tuple ``(exposures, log_likelihood)`` containing the simplex weight vector
        ``exposures`` of shape ``(K,)`` and the exact multinomial log-likelihood.
    """
    validate_settings(settings)
    phi, _, _ = normalize_signature_matrix(signature_matrix)
    n_ctx, n_sig = phi.shape

    counts_arr = np.asarray(counts_96, dtype=float)
    if counts_arr.ndim == 2:
        counts_vec = np.sum(counts_arr, axis=0)
    elif counts_arr.ndim == 1:
        counts_vec = counts_arr
    else:
        raise ValueError(f"counts_96 must be 1D or 2D, got shape {counts_arr.shape}.")

    if counts_vec.shape[0] != n_ctx:
        raise ValueError(
            f"counts_96 length ({counts_vec.shape[0]}) does not match "
            f"signature_matrix rows ({n_ctx})."
        )
    if np.any(counts_vec < 0.0) or np.any(~np.isfinite(counts_vec)):
        raise ValueError("counts_96 must contain finite non-negative counts.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    total_n = float(np.sum(counts_vec))
    if total_n <= 0.0:
        return MultinomialEMResult(
            exposures=np.full(n_sig, 1.0 / n_sig, dtype=float),
            log_likelihood=0.0,
        )

    if n_sig == 1:
        pi_single = np.array([1.0], dtype=float)
        pos = counts_vec > 0.0
        q_single = np.maximum(phi[pos, 0], 1e-300)
        ll_comb = float(gammaln(total_n + 1.0) - np.sum(gammaln(counts_vec + 1.0)))
        ll = ll_comb + float(np.sum(counts_vec[pos] * np.log(q_single)))
        return MultinomialEMResult(exposures=pi_single, log_likelihood=ll)

    pos_mask = counts_vec > 0.0
    n_pos = counts_vec[pos_mask]
    phi_pos = phi[pos_mask, :]

    def _em_step(pi_vec: np.ndarray) -> np.ndarray:
        mix = np.maximum(phi_pos @ pi_vec, 1e-300)
        nxt = pi_vec * ((n_pos / mix) @ phi_pos) / total_n
        s = float(np.sum(nxt))
        return nxt / s if s > 0.0 else np.full(n_sig, 1.0 / n_sig, dtype=float)

    def _cat_ll(pi_vec: np.ndarray) -> float:
        mix = np.maximum(phi_pos @ pi_vec, 1e-300)
        return float(np.sum(n_pos * np.log(mix)))

    pi = np.full(n_sig, 1.0 / n_sig, dtype=float)
    for _ in range(max_iter):
        p1 = _em_step(pi)
        if np.max(np.abs(p1 - pi)) < tol:
            pi = p1
            break
        p2 = _em_step(p1)
        if np.max(np.abs(p2 - p1)) < tol:
            pi = p2
            break

        # SQUAREM step with strict monotonicity safeguard
        r = p1 - pi
        v = (p2 - p1) - r
        vv = float(np.dot(v, v))
        if vv > 1e-28:
            step_len = -np.sqrt(float(np.dot(r, r)) / vv)
            if step_len > -1.0:
                step_len = -1.0
            cand = pi - 2.0 * step_len * r + (step_len * step_len) * v
            cand = np.maximum(cand, 0.0)
            cand_sum = float(np.sum(cand))
            if cand_sum > 0.0:
                cand /= cand_sum
                if _cat_ll(cand) >= _cat_ll(p2) - 1e-12:
                    p_next = _em_step(cand)
                    if np.max(np.abs(p_next - cand)) < tol:
                        pi = p_next
                        break
                    pi = p_next
                    continue
        pi = p2

    # Clean tiny machine-noise boundary values and renormalize
    pi = np.where(pi < 1e-12, 0.0, pi)
    pi /= float(np.sum(pi))

    ll_mult_const = float(gammaln(total_n + 1.0) - np.sum(gammaln(counts_vec + 1.0)))
    log_lik = ll_mult_const + _cat_ll(pi)
    return MultinomialEMResult(exposures=pi, log_likelihood=log_lik)


def detect_active_signatures(
    mutations_df: pd.DataFrame,
    signature_matrix: pd.DataFrame | np.ndarray,
    active_threshold: float = 0.05,
    settings: SettingsMode = "default",
) -> list[str]:
    """Identify globally active mutational signatures prior to PELT segmentation.

    Fits global Multinomial EM across all pooled mutations in ``mutations_df``
    and retains signatures with estimated exposure ``>= active_threshold``
    (default ``0.05``, matching TrackSig's ``activeInSample`` pre-filter). If no
    signature reaches ``active_threshold``, falls back to the single highest-weight
    signature so at least one signature is always active.
    """
    validate_settings(settings)
    if not (0.0 <= float(active_threshold) < 1.0):
        raise ValueError(f"active_threshold must be in [0, 1), got {active_threshold}.")

    phi, sig_names, context_names = normalize_signature_matrix(signature_matrix)
    channels = map_mutations_to_channels(mutations_df, context_names)
    counts = np.bincount(channels, minlength=phi.shape[0]).astype(float)

    exposures, _ = fit_multinomial_signatures_em(
        counts, phi, tol=1e-7, max_iter=1000, settings=settings
    )
    active_idx = np.flatnonzero(exposures >= float(active_threshold))
    if active_idx.size == 0:
        active_idx = np.array([int(np.argmax(exposures))], dtype=int)
    return [sig_names[int(i)] for i in active_idx]
