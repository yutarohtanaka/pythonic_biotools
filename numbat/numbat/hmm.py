"""Haplotype-aware joint Beta-Binomial + Poisson-LogNormal HMM (Numba + NumPy parity)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy

from .config import COLLAPSED_STATES, CNVStateSpec, DEFAULT_STATES, NumbatConfig

try:
    from numba import njit as _numba_njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - tested via monkeypatch
    _HAS_NUMBA = False
    _numba_njit = None


def has_numba() -> bool:
    """Return True if Numba JIT acceleration is available."""
    return bool(_HAS_NUMBA)


def resolve_backend(backend: Literal["auto", "numba", "numpy"] = "auto") -> str:
    """Resolve HMM execution backend ('numba' or 'numpy')."""
    if backend == "auto":
        return "numba" if _HAS_NUMBA else "numpy"
    if backend == "numba":
        if not _HAS_NUMBA:
            raise ImportError("numba is not available; use backend='numpy' or backend='auto'.")
        return "numba"
    if backend == "numpy":
        return "numpy"
    raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'numba', or 'numpy'.")


def phase_switch_prob(
    dist_cm: np.ndarray | float,
    nu: float = 1.0,
) -> np.ndarray | float:
    """Compute Haldane phase-switch probability p_s = (1 - exp(-2 * nu * d)) / 2."""
    if nu < 0.0:
        raise ValueError(f"Phase-switch multiplier nu must be >= 0, got {nu}.")
    arr = np.asarray(dist_cm, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("Genetic distances dist_cm must be non-negative.")
    res = 0.5 * (1.0 - np.exp(-2.0 * float(nu) * arr))
    if np.ndim(dist_cm) == 0:
        return float(res)
    return res


def beta_binomial_logpmf(
    y: np.ndarray | float,
    n: np.ndarray | float,
    mu: np.ndarray | float,
    gamma: float,
) -> np.ndarray:
    """Evaluate Beta-Binomial log-PMF with n=0 uninformative guard and large-gamma asymptote."""
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"Beta-Binomial concentration gamma must be finite and > 0, got {gamma}.")

    y_arr = np.asarray(y, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    mu_raw = np.asarray(mu, dtype=np.float64)

    if not (np.all(np.isfinite(y_arr)) and np.all(np.isfinite(n_arr))):
        raise ValueError("Allele counts y (AD) and n (DP) must be finite.")
    if not np.all(np.isfinite(mu_raw)):
        raise ValueError("Allele fraction mu must be finite.")

    mu_arr = np.clip(mu_raw, 1e-6, 1.0 - 1e-6)

    if np.any(y_arr < 0.0) or np.any(n_arr < 0.0):
        raise ValueError("Allele counts y (AD) and n (DP) must be non-negative.")
    if np.any(y_arr > n_arr):
        raise ValueError("Allele counts cannot satisfy y (AD) > n (DP).")

    y_b, n_b, mu_b = np.broadcast_arrays(y_arr, n_arr, mu_arr)
    out = np.zeros(y_b.shape, dtype=np.float64)

    nz = n_b > 0.0
    if not np.any(nz):
        return out

    yn = y_b[nz]
    nn = n_b[nz]
    mn = mu_b[nz]

    use_binom = (gamma > 1e6) | ((gamma / np.maximum(nn, 1.0)) > 1e5)
    vals = np.empty_like(yn, dtype=np.float64)

    comb = gammaln(nn + 1.0) - gammaln(yn + 1.0) - gammaln(nn - yn + 1.0)

    if np.any(use_binom):
        yb = yn[use_binom]
        nb = nn[use_binom]
        mb = mn[use_binom]
        vals[use_binom] = comb[use_binom] + xlogy(yb, mb) + xlogy(nb - yb, 1.0 - mb)

    bb_mask = ~use_binom
    if np.any(bb_mask):
        ya = yn[bb_mask]
        na = nn[bb_mask]
        ma = mn[bb_mask]
        alpha = gamma * ma
        beta = gamma * (1.0 - ma)
        vals[bb_mask] = (
            comb[bb_mask]
            + gammaln(ya + alpha)
            + gammaln(na - ya + beta)
            - gammaln(na + gamma)
            - gammaln(alpha)
            - gammaln(beta)
            + gammaln(gamma)
        )

    out[nz] = vals
    return out


def _build_log_transition_matrix(
    p_s: float,
    t_cnv: float,
    subclass_id: np.ndarray,
    subclass_size: np.ndarray,
    orientation: np.ndarray,
    n_subclasses: int,
) -> np.ndarray:
    """Build K x K log transition matrix combining CNV transitions and phase switching."""
    k_states = len(subclass_id)
    mat = np.empty((k_states, k_states), dtype=np.float64)
    off_prob = t_cnv / max(1, n_subclasses - 1)
    for i in range(k_states):
        si = subclass_id[i]
        oi = orientation[i]
        for j in range(k_states):
            sj = subclass_id[j]
            oj = orientation[j]
            if si == sj:
                if oi == 0 and oj == 0:
                    p = 1.0 - t_cnv
                elif oi == oj:
                    p = (1.0 - t_cnv) * (1.0 - p_s)
                else:
                    p = (1.0 - t_cnv) * p_s
            else:
                p = off_prob / float(subclass_size[j])
            mat[i, j] = math.log(max(p, 1e-300))
    return mat


def _logsumexp_1d_numpy(vec: np.ndarray) -> float:
    m = float(np.max(vec))
    if not np.isfinite(m):
        return -math.inf
    return m + math.log(float(np.sum(np.exp(vec - m))))


def _hmm_forward_backward_numpy(
    log_emit: np.ndarray,
    chrom_offsets: np.ndarray,
    p_switch: np.ndarray,
    t_cnv: float,
    log_init_prior: np.ndarray,
    subclass_id: np.ndarray,
    subclass_size: np.ndarray,
    orientation: np.ndarray,
    n_subclasses: int,
) -> tuple[np.ndarray, float]:
    """Pure NumPy log-space Forward-Backward algorithm per chromosome."""
    n_loci, k_states = log_emit.shape
    posteriors = np.empty((n_loci, k_states), dtype=np.float64)
    total_ll = 0.0

    for c_idx in range(len(chrom_offsets) - 1):
        s = int(chrom_offsets[c_idx])
        e = int(chrom_offsets[c_idx + 1])
        if e <= s:
            continue
        length = e - s
        alpha = np.empty((length, k_states), dtype=np.float64)
        beta = np.zeros((length, k_states), dtype=np.float64)

        alpha[0] = log_init_prior + log_emit[s]

        # Precompute static log transition template when p_switch == 0
        log_a_zero = _build_log_transition_matrix(
            0.0, t_cnv, subclass_id, subclass_size, orientation, n_subclasses
        )
        log_a_cache: dict[int, np.ndarray] = {}

        for t_rel in range(1, length):
            t_abs = s + t_rel
            ps = float(p_switch[t_abs])
            if ps == 0.0:
                log_a = log_a_zero
            else:
                log_a = _build_log_transition_matrix(
                    ps, t_cnv, subclass_id, subclass_size, orientation, n_subclasses
                )
            log_a_cache[t_rel] = log_a
            # alpha[t_rel, j] = log_emit[t_abs, j] + logsumexp_i(alpha[t_rel-1, i] + log_a[i, j])
            scores = alpha[t_rel - 1][:, None] + log_a
            m = np.max(scores, axis=0)
            valid = np.isfinite(m)
            lse = np.full(k_states, -np.inf, dtype=np.float64)
            if np.any(valid):
                lse[valid] = m[valid] + np.log(np.sum(np.exp(scores[:, valid] - m[valid]), axis=0))
            alpha[t_rel] = log_emit[t_abs] + lse

        ll_chr = _logsumexp_1d_numpy(alpha[length - 1])
        total_ll += ll_chr

        for t_rel in range(length - 2, -1, -1):
            t_next = s + t_rel + 1
            log_a = log_a_cache.get(t_rel + 1, log_a_zero)
            # beta[t_rel, i] = logsumexp_j(log_a[i, j] + log_emit[t_next, j] + beta[t_rel+1, j])
            right = log_emit[t_next] + beta[t_rel + 1]
            scores = log_a + right[None, :]
            m = np.max(scores, axis=1)
            valid = np.isfinite(m)
            lse = np.full(k_states, -np.inf, dtype=np.float64)
            if np.any(valid):
                lse[valid] = m[valid] + np.log(
                    np.sum(np.exp(scores[valid, :] - m[valid, None]), axis=1)
                )
            beta[t_rel] = lse

        log_post = alpha + beta - ll_chr
        row_m = np.max(log_post, axis=1, keepdims=True)
        probs = np.exp(log_post - row_m)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        probs = np.where(row_sums > 0.0, probs / row_sums, 1.0 / k_states)
        posteriors[s:e] = probs

    return posteriors, float(total_ll)


def _hmm_viterbi_numpy(
    log_emit: np.ndarray,
    chrom_offsets: np.ndarray,
    p_switch: np.ndarray,
    t_cnv: float,
    log_init_prior: np.ndarray,
    subclass_id: np.ndarray,
    subclass_size: np.ndarray,
    orientation: np.ndarray,
    n_subclasses: int,
) -> np.ndarray:
    """Pure NumPy log-space Viterbi algorithm per chromosome."""
    n_loci, k_states = log_emit.shape
    path = np.empty(n_loci, dtype=np.int64)
    log_a_zero = _build_log_transition_matrix(
        0.0, t_cnv, subclass_id, subclass_size, orientation, n_subclasses
    )

    for c_idx in range(len(chrom_offsets) - 1):
        s = int(chrom_offsets[c_idx])
        e = int(chrom_offsets[c_idx + 1])
        if e <= s:
            continue
        length = e - s
        dp = np.empty((length, k_states), dtype=np.float64)
        bp = np.zeros((length, k_states), dtype=np.int64)
        dp[0] = log_init_prior + log_emit[s]

        for t_rel in range(1, length):
            t_abs = s + t_rel
            ps = float(p_switch[t_abs])
            log_a = (
                log_a_zero
                if ps == 0.0
                else _build_log_transition_matrix(
                    ps, t_cnv, subclass_id, subclass_size, orientation, n_subclasses
                )
            )
            scores = dp[t_rel - 1][:, None] + log_a
            best_prev = np.argmax(scores, axis=0)
            bp[t_rel] = best_prev
            dp[t_rel] = log_emit[t_abs] + scores[best_prev, np.arange(k_states)]

        cur = int(np.argmax(dp[length - 1]))
        path[s + length - 1] = cur
        for t_rel in range(length - 1, 0, -1):
            cur = int(bp[t_rel, cur])
            path[s + t_rel - 1] = cur

    return path


if _HAS_NUMBA and _numba_njit is not None:

    @_numba_njit(fastmath=False)
    def _fill_log_trans_numba(
        p_s: float,
        t_cnv: float,
        subclass_id: np.ndarray,
        subclass_size: np.ndarray,
        orientation: np.ndarray,
        n_subclasses: int,
        out_mat: np.ndarray,
    ) -> None:
        k_states = len(subclass_id)
        off_prob = t_cnv / max(1, n_subclasses - 1)
        for i in range(k_states):
            si = subclass_id[i]
            oi = orientation[i]
            for j in range(k_states):
                sj = subclass_id[j]
                oj = orientation[j]
                if si == sj:
                    if oi == 0 and oj == 0:
                        p = 1.0 - t_cnv
                    elif oi == oj:
                        p = (1.0 - t_cnv) * (1.0 - p_s)
                    else:
                        p = (1.0 - t_cnv) * p_s
                else:
                    p = off_prob / float(subclass_size[j])
                if p < 1e-300:
                    p = 1e-300
                out_mat[i, j] = math.log(p)

    @_numba_njit(fastmath=False)
    def _hmm_forward_backward_numba(
        log_emit: np.ndarray,
        chrom_offsets: np.ndarray,
        p_switch: np.ndarray,
        t_cnv: float,
        log_init_prior: np.ndarray,
        subclass_id: np.ndarray,
        subclass_size: np.ndarray,
        orientation: np.ndarray,
        n_subclasses: int,
    ) -> tuple[np.ndarray, float]:
        n_loci, k_states = log_emit.shape
        posteriors = np.empty((n_loci, k_states), dtype=np.float64)
        total_ll = 0.0

        log_a_zero = np.empty((k_states, k_states), dtype=np.float64)
        _fill_log_trans_numba(
            0.0, t_cnv, subclass_id, subclass_size, orientation, n_subclasses, log_a_zero
        )
        log_a_tmp = np.empty((k_states, k_states), dtype=np.float64)

        for c_idx in range(len(chrom_offsets) - 1):
            s = chrom_offsets[c_idx]
            e = chrom_offsets[c_idx + 1]
            if e <= s:
                continue
            length = e - s
            alpha = np.empty((length, k_states), dtype=np.float64)
            beta = np.zeros((length, k_states), dtype=np.float64)

            for k in range(k_states):
                alpha[0, k] = log_init_prior[k] + log_emit[s, k]

            for t_rel in range(1, length):
                t_abs = s + t_rel
                ps = p_switch[t_abs]
                if ps == 0.0:
                    log_a = log_a_zero
                else:
                    _fill_log_trans_numba(
                        ps, t_cnv, subclass_id, subclass_size, orientation, n_subclasses, log_a_tmp
                    )
                    log_a = log_a_tmp

                for j in range(k_states):
                    m = -math.inf
                    for i in range(k_states):
                        val = alpha[t_rel - 1, i] + log_a[i, j]
                        if val > m:
                            m = val
                    if not math.isfinite(m):
                        alpha[t_rel, j] = -math.inf
                    else:
                        acc = 0.0
                        for i in range(k_states):
                            acc += math.exp(alpha[t_rel - 1, i] + log_a[i, j] - m)
                        alpha[t_rel, j] = log_emit[t_abs, j] + m + math.log(acc)

            m_end = -math.inf
            for k in range(k_states):
                if alpha[length - 1, k] > m_end:
                    m_end = alpha[length - 1, k]
            if not math.isfinite(m_end):
                ll_chr = -math.inf
            else:
                acc_end = 0.0
                for k in range(k_states):
                    acc_end += math.exp(alpha[length - 1, k] - m_end)
                ll_chr = m_end + math.log(acc_end)
            total_ll += ll_chr

            for t_rel in range(length - 2, -1, -1):
                t_next = s + t_rel + 1
                ps = p_switch[t_next]
                if ps == 0.0:
                    log_a = log_a_zero
                else:
                    _fill_log_trans_numba(
                        ps, t_cnv, subclass_id, subclass_size, orientation, n_subclasses, log_a_tmp
                    )
                    log_a = log_a_tmp

                for i in range(k_states):
                    m = -math.inf
                    for j in range(k_states):
                        val = log_a[i, j] + log_emit[t_next, j] + beta[t_rel + 1, j]
                        if val > m:
                            m = val
                    if not math.isfinite(m):
                        beta[t_rel, i] = -math.inf
                    else:
                        acc = 0.0
                        for j in range(k_states):
                            acc += math.exp(
                                log_a[i, j] + log_emit[t_next, j] + beta[t_rel + 1, j] - m
                            )
                        beta[t_rel, i] = m + math.log(acc)

            for t_rel in range(length):
                t_abs = s + t_rel
                m_row = -math.inf
                for k in range(k_states):
                    val = alpha[t_rel, k] + beta[t_rel, k] - ll_chr
                    if val > m_row:
                        m_row = val
                row_sum = 0.0
                for k in range(k_states):
                    p = math.exp((alpha[t_rel, k] + beta[t_rel, k] - ll_chr) - m_row)
                    posteriors[t_abs, k] = p
                    row_sum += p
                if row_sum > 0.0:
                    for k in range(k_states):
                        posteriors[t_abs, k] /= row_sum
                else:
                    for k in range(k_states):
                        posteriors[t_abs, k] = 1.0 / k_states

        return posteriors, total_ll

    @_numba_njit(fastmath=False)
    def _hmm_viterbi_numba(
        log_emit: np.ndarray,
        chrom_offsets: np.ndarray,
        p_switch: np.ndarray,
        t_cnv: float,
        log_init_prior: np.ndarray,
        subclass_id: np.ndarray,
        subclass_size: np.ndarray,
        orientation: np.ndarray,
        n_subclasses: int,
    ) -> np.ndarray:
        n_loci, k_states = log_emit.shape
        path = np.empty(n_loci, dtype=np.int64)

        log_a_zero = np.empty((k_states, k_states), dtype=np.float64)
        _fill_log_trans_numba(
            0.0, t_cnv, subclass_id, subclass_size, orientation, n_subclasses, log_a_zero
        )
        log_a_tmp = np.empty((k_states, k_states), dtype=np.float64)

        for c_idx in range(len(chrom_offsets) - 1):
            s = chrom_offsets[c_idx]
            e = chrom_offsets[c_idx + 1]
            if e <= s:
                continue
            length = e - s
            dp = np.empty((length, k_states), dtype=np.float64)
            bp = np.zeros((length, k_states), dtype=np.int64)

            for k in range(k_states):
                dp[0, k] = log_init_prior[k] + log_emit[s, k]

            for t_rel in range(1, length):
                t_abs = s + t_rel
                ps = p_switch[t_abs]
                if ps == 0.0:
                    log_a = log_a_zero
                else:
                    _fill_log_trans_numba(
                        ps, t_cnv, subclass_id, subclass_size, orientation, n_subclasses, log_a_tmp
                    )
                    log_a = log_a_tmp

                for j in range(k_states):
                    best_i = 0
                    best_val = dp[t_rel - 1, 0] + log_a[0, j]
                    for i in range(1, k_states):
                        val = dp[t_rel - 1, i] + log_a[i, j]
                        if val > best_val:
                            best_val = val
                            best_i = i
                    bp[t_rel, j] = best_i
                    dp[t_rel, j] = log_emit[t_abs, j] + best_val

            cur = 0
            best_end = dp[length - 1, 0]
            for k in range(1, k_states):
                if dp[length - 1, k] > best_end:
                    best_end = dp[length - 1, k]
                    cur = k
            path[s + length - 1] = cur
            for t_rel in range(length - 1, 0, -1):
                cur = bp[t_rel, cur]
                path[s + t_rel - 1] = cur

        return path


@dataclass(frozen=True)
class HMMResult:
    """Typed output container from haplotype-aware HMM inference."""

    posteriors: np.ndarray
    collapsed_posteriors: np.ndarray
    viterbi_states: np.ndarray
    viterbi_collapsed: tuple[str, ...]
    log_likelihood: float
    backend_used: str
    collapsed_names: tuple[str, ...] = COLLAPSED_STATES
    state_names: tuple[str, ...] = ()
    locus_id: tuple[str, ...] = ()
    chrom: tuple[str, ...] = ()
    pos: np.ndarray | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-locus posterior probabilities and Viterbi calls as a DataFrame."""
        n_loci = self.posteriors.shape[0]
        data: dict[str, object] = {}
        if self.locus_id:
            data["locus_id"] = list(self.locus_id)
        else:
            data["locus_id"] = [f"locus_{i}" for i in range(n_loci)]
        if self.chrom:
            data["chrom"] = list(self.chrom)
        if self.pos is not None:
            data["pos"] = self.pos

        if self.state_names:
            data["viterbi_state"] = [self.state_names[int(i)] for i in self.viterbi_states]
        else:
            data["viterbi_state"] = self.viterbi_states.astype(int)
        data["viterbi_collapsed"] = list(self.viterbi_collapsed)

        for j, cname in enumerate(self.collapsed_names):
            data[f"p_{cname}"] = self.collapsed_posteriors[:, j]

        return pd.DataFrame(data)


def _encode_states(
    states: Sequence[CNVStateSpec],
    p_init_cnv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Encode CNVStateSpec tuple into contiguous arrays for Numba/NumPy HMM kernels."""
    k_states = len(states)
    subclass_names: list[str] = []
    subclass_lookup: dict[str, int] = {}
    for s in states:
        if s.subclass not in subclass_lookup:
            subclass_lookup[s.subclass] = len(subclass_names)
            subclass_names.append(s.subclass)

    subclass_id = np.array([subclass_lookup[s.subclass] for s in states], dtype=np.int64)
    counts_per_sub = np.bincount(subclass_id, minlength=len(subclass_names))
    subclass_size = counts_per_sub[subclass_id].astype(np.int64)
    orientation = np.array([s.orientation for s in states], dtype=np.int64)

    collapsed_lookup = {c: idx for idx, c in enumerate(COLLAPSED_STATES)}
    state_to_collapsed = np.array([collapsed_lookup[s.collapsed] for s in states], dtype=np.int64)

    n_subclasses = len(subclass_names)
    non_neu_subclasses = sum(1 for name in subclass_names if name != "neu")
    init_prior = np.empty(k_states, dtype=np.float64)
    for idx, s in enumerate(states):
        if s.subclass == "neu":
            init_prior[idx] = (1.0 - p_init_cnv) / float(subclass_size[idx])
        else:
            init_prior[idx] = p_init_cnv / (max(1, non_neu_subclasses) * float(subclass_size[idx]))
    init_prior = init_prior / np.sum(init_prior)
    log_init_prior = np.log(np.clip(init_prior, 1e-300, 1.0))

    return (
        subclass_id,
        subclass_size,
        orientation,
        n_subclasses,
        state_to_collapsed,
        log_init_prior,
    )


def run_joint_hmm(
    log_emit: np.ndarray,
    chrom_offsets: Sequence[int] | np.ndarray,
    p_switch: Sequence[float] | np.ndarray | None = None,
    t_cnv: float = 1e-5,
    p_init_cnv: float = 0.15,
    states: Sequence[CNVStateSpec] = DEFAULT_STATES,
    backend: Literal["auto", "numba", "numpy"] = "auto",
    locus_id: Sequence[str] | None = None,
    chrom: Sequence[str] | None = None,
    pos: Sequence[int] | np.ndarray | None = None,
) -> HMMResult:
    """Run haplotype-aware Forward-Backward and Viterbi HMM across chromosome slices."""
    emit = np.ascontiguousarray(np.asarray(log_emit, dtype=np.float64))
    if emit.ndim != 2 or emit.shape[0] == 0:
        raise ValueError("log_emit must be a non-empty 2D (n_loci x n_states) array.")
    n_loci, k_states = emit.shape
    if k_states != len(states):
        raise ValueError(f"log_emit has {k_states} state columns, expected {len(states)}.")
    if not np.all(np.isfinite(emit)):
        raise ValueError("log_emit contains non-finite values.")

    offsets = np.ascontiguousarray(np.asarray(chrom_offsets, dtype=np.int64))
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0 or offsets[-1] != n_loci:
        raise ValueError(
            "chrom_offsets must be 1D starting at 0 and ending at n_loci."
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError("chrom_offsets must be non-decreasing.")

    if p_switch is None:
        ps_arr = np.zeros(n_loci, dtype=np.float64)
    else:
        ps_arr = np.ascontiguousarray(np.asarray(p_switch, dtype=np.float64))
        if ps_arr.shape != (n_loci,):
            raise ValueError(f"p_switch shape {ps_arr.shape} != ({n_loci},).")
        if np.any(ps_arr < 0.0) or np.any(ps_arr > 0.5):
            raise ValueError("p_switch entries must lie in [0, 0.5].")

    if not (0.0 < t_cnv < 0.5):
        raise ValueError(f"t_cnv must lie in (0, 0.5), got {t_cnv}.")
    if not (0.0 < p_init_cnv < 1.0):
        raise ValueError(f"p_init_cnv must lie in (0, 1), got {p_init_cnv}.")

    resolved = resolve_backend(backend)
    (
        subclass_id,
        subclass_size,
        orientation,
        n_subclasses,
        state_to_collapsed,
        log_init_prior,
    ) = _encode_states(states, p_init_cnv)

    if resolved == "numba":
        posteriors, total_ll = _hmm_forward_backward_numba(
            emit,
            offsets,
            ps_arr,
            float(t_cnv),
            log_init_prior,
            subclass_id,
            subclass_size,
            orientation,
            n_subclasses,
        )
        viterbi_states = _hmm_viterbi_numba(
            emit,
            offsets,
            ps_arr,
            float(t_cnv),
            log_init_prior,
            subclass_id,
            subclass_size,
            orientation,
            n_subclasses,
        )
    else:
        posteriors, total_ll = _hmm_forward_backward_numpy(
            emit,
            offsets,
            ps_arr,
            float(t_cnv),
            log_init_prior,
            subclass_id,
            subclass_size,
            orientation,
            n_subclasses,
        )
        viterbi_states = _hmm_viterbi_numpy(
            emit,
            offsets,
            ps_arr,
            float(t_cnv),
            log_init_prior,
            subclass_id,
            subclass_size,
            orientation,
            n_subclasses,
        )

    collapsed_post = np.zeros((n_loci, len(COLLAPSED_STATES)), dtype=np.float64)
    for k_idx, c_idx in enumerate(state_to_collapsed):
        collapsed_post[:, c_idx] += posteriors[:, k_idx]

    viterbi_collapsed = tuple(COLLAPSED_STATES[state_to_collapsed[int(s)]] for s in viterbi_states)
    state_names = tuple(s.name for s in states)

    return HMMResult(
        posteriors=posteriors,
        collapsed_posteriors=collapsed_post,
        viterbi_states=viterbi_states,
        viterbi_collapsed=viterbi_collapsed,
        log_likelihood=float(total_ll),
        backend_used=resolved,
        collapsed_names=COLLAPSED_STATES,
        state_names=state_names,
        locus_id=tuple(str(x) for x in locus_id) if locus_id is not None else (),
        chrom=tuple(str(x) for x in chrom) if chrom is not None else (),
        pos=np.asarray(pos, dtype=np.int64) if pos is not None else None,
    )


def phase_hmm_loglik(
    ad: np.ndarray,
    dp: np.ndarray,
    p_switch: np.ndarray,
    theta: float,
    gamma: float,
) -> float:
    """Compute exact 2-state {0.5 + theta, 0.5 - theta} phase-switch marginal log-likelihood."""
    ad_arr = np.asarray(ad, dtype=np.float64)
    dp_arr = np.asarray(dp, dtype=np.float64)
    if ad_arr.size == 0:
        return 0.0
    if theta <= 0.0:
        return float(np.sum(beta_binomial_logpmf(ad_arr, dp_arr, 0.5, gamma)))

    mu_p = min(1.0 - 1e-6, 0.5 + float(theta))
    mu_m = max(1e-6, 0.5 - float(theta))
    emit_p = beta_binomial_logpmf(ad_arr, dp_arr, mu_p, gamma)
    emit_m = beta_binomial_logpmf(ad_arr, dp_arr, mu_m, gamma)

    a0 = math.log(0.5) + emit_p[0]
    a1 = math.log(0.5) + emit_m[0]

    for t in range(1, len(ad_arr)):
        ps = float(p_switch[t])
        log_stay = math.log(max(1.0 - ps, 1e-300))
        log_flip = math.log(max(ps, 1e-300))
        n0 = emit_p[t] + _logsumexp_pair(a0 + log_stay, a1 + log_flip)
        n1 = emit_m[t] + _logsumexp_pair(a0 + log_flip, a1 + log_stay)
        a0, a1 = n0, n1

    return float(_logsumexp_pair(a0, a1))


def _logsumexp_pair(x: float, y: float) -> float:
    if x >= y:
        if not math.isfinite(x):
            return -math.inf
        return x + math.log1p(math.exp(y - x))
    if not math.isfinite(y):
        return -math.inf
    return y + math.log1p(math.exp(x - y))


def phase_hmm_loglik_batch(
    ad_mat: np.ndarray,
    dp_mat: np.ndarray,
    p_switch: np.ndarray,
    theta: float,
    gamma: float,
) -> np.ndarray:
    """Vectorized 2-state phase-switch forward log-likelihood across (C, S) cell-by-SNP matrices."""
    n_cells, n_snps = ad_mat.shape
    if n_snps == 0:
        return np.zeros(n_cells, dtype=np.float64)
    if theta <= 0.0:
        return np.sum(beta_binomial_logpmf(ad_mat, dp_mat, 0.5, gamma), axis=1)

    mu_p = min(1.0 - 1e-6, 0.5 + float(theta))
    mu_m = max(1e-6, 0.5 - float(theta))
    emit_p = beta_binomial_logpmf(ad_mat, dp_mat, mu_p, gamma)
    emit_m = beta_binomial_logpmf(ad_mat, dp_mat, mu_m, gamma)

    a0 = math.log(0.5) + emit_p[:, 0]
    a1 = math.log(0.5) + emit_m[:, 0]

    for t in range(1, n_snps):
        ps = float(p_switch[t])
        log_stay = math.log(max(1.0 - ps, 1e-300))
        log_flip = math.log(max(ps, 1e-300))
        n0 = emit_p[:, t] + np.logaddexp(a0 + log_stay, a1 + log_flip)
        n1 = emit_m[:, t] + np.logaddexp(a0 + log_flip, a1 + log_stay)
        a0, a1 = n0, n1

    return np.logaddexp(a0, a1)


def fit_allele_hmm(
    ad: Sequence[float] | np.ndarray,
    dp: Sequence[float] | np.ndarray,
    dist_cm: Sequence[float] | np.ndarray | None = None,
    chrom: Sequence[str] | None = None,
    pos: Sequence[int] | np.ndarray | None = None,
    config: NumbatConfig | None = None,
) -> HMMResult:
    """Convenience wrapper running the haplotype-aware HMM on allele counts alone."""
    cfg = config if config is not None else NumbatConfig()
    ad_arr = np.asarray(ad, dtype=np.float64)
    dp_arr = np.asarray(dp, dtype=np.float64)
    if ad_arr.ndim != 1 or ad_arr.shape != dp_arr.shape or ad_arr.size == 0:
        raise ValueError("ad and dp must be non-empty 1D arrays of equal length.")
    n_loci = ad_arr.size

    if chrom is None:
        chrom_arr = np.array(["chr1"] * n_loci, dtype=object)
        offsets = np.array([0, n_loci], dtype=np.int64)
    else:
        if len(chrom) != n_loci:
            raise ValueError("chrom length must match ad/dp length.")
        chrom_arr = np.asarray(chrom, dtype=object)
        offsets_list = [0]
        for i in range(1, n_loci):
            if chrom_arr[i] != chrom_arr[i - 1]:
                offsets_list.append(i)
        offsets_list.append(n_loci)
        offsets = np.asarray(offsets_list, dtype=np.int64)

    if dist_cm is None:
        if pos is not None:
            pos_arr = np.asarray(pos, dtype=np.float64)
            d_arr = np.zeros(n_loci, dtype=np.float64)
            for c_idx in range(len(offsets) - 1):
                s, e = int(offsets[c_idx]), int(offsets[c_idx + 1])
                if e - s > 1:
                    d_arr[s + 1 : e] = np.maximum(0.0, np.diff(pos_arr[s:e]) * 1e-6)
        else:
            d_arr = np.zeros(n_loci, dtype=np.float64)
    else:
        d_arr = np.asarray(dist_cm, dtype=np.float64)
        if d_arr.shape != (n_loci,):
            raise ValueError("dist_cm length must match ad/dp length.")

    p_switch = np.asarray(phase_switch_prob(d_arr, nu=cfg.nu), dtype=np.float64)
    for c_idx in range(len(offsets) - 1):
        p_switch[int(offsets[c_idx])] = 0.0

    mu_vec = np.array([s.mu for s in cfg.states], dtype=np.float64)
    log_emit = beta_binomial_logpmf(ad_arr[:, None], dp_arr[:, None], mu_vec[None, :], cfg.gamma)

    return run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=offsets,
        p_switch=p_switch,
        t_cnv=cfg.t_cnv,
        p_init_cnv=cfg.p_init_cnv,
        states=cfg.states,
        backend=cfg.backend,
        chrom=chrom_arr,
        pos=pos,
    )


def fit_expression_hmm(
    counts: Sequence[float] | np.ndarray,
    ref_lambda: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray | float | None = None,
    chrom: Sequence[str] | None = None,
    pos: Sequence[int] | np.ndarray | None = None,
    config: NumbatConfig | None = None,
) -> HMMResult:
    """Convenience wrapper running the HMM on pseudobulk/cell gene expression counts alone."""
    from .expression import compute_diploid_offset, pln_logpmf

    cfg = config if config is not None else NumbatConfig()
    x_arr = np.asarray(counts, dtype=np.float64)
    lam_arr = np.asarray(ref_lambda, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.shape != lam_arr.shape or x_arr.size == 0:
        raise ValueError("counts and ref_lambda must be non-empty 1D arrays of equal length.")
    if np.any(x_arr < 0.0):
        raise ValueError("counts must be non-negative.")
    if np.any(lam_arr <= 0.0):
        raise ValueError("ref_lambda entries must be positive.")
    n_loci = x_arr.size

    if sigma is None:
        sig_arr = np.full(n_loci, cfg.default_sigma, dtype=np.float64)
    elif np.ndim(sigma) == 0:
        sig_arr = np.full(n_loci, float(sigma), dtype=np.float64)
    else:
        sig_arr = np.asarray(sigma, dtype=np.float64)
        if sig_arr.shape != (n_loci,):
            raise ValueError("sigma length must match counts length.")

    if chrom is None:
        chrom_arr = np.array(["chr1"] * n_loci, dtype=object)
        offsets = np.array([0, n_loci], dtype=np.int64)
    else:
        if len(chrom) != n_loci:
            raise ValueError("chrom length must match counts length.")
        chrom_arr = np.asarray(chrom, dtype=object)
        offsets_list = [0]
        for i in range(1, n_loci):
            if chrom_arr[i] != chrom_arr[i - 1]:
                offsets_list.append(i)
        offsets_list.append(n_loci)
        offsets = np.asarray(offsets_list, dtype=np.int64)

    lib_size = float(np.sum(x_arr))
    phi_vec = np.array([s.phi for s in cfg.states], dtype=np.float64)
    if lib_size <= 0.0:
        log_emit = np.zeros((n_loci, len(cfg.states)), dtype=np.float64)
    else:
        lam_norm = lam_arr / float(np.sum(lam_arr))
        delta = compute_diploid_offset(x_arr, lam_norm, lib_size=lib_size)
        base_mu = np.log(np.maximum(lib_size * lam_norm, 1e-12)) + delta
        log_emit = pln_logpmf(
            x_arr[:, None],
            base_mu[:, None] + np.log(phi_vec)[None, :],
            sig_arr[:, None],
            n_quadrature=cfg.n_quadrature,
        )

    return run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=offsets,
        p_switch=np.zeros(n_loci, dtype=np.float64),
        t_cnv=cfg.t_cnv,
        p_init_cnv=cfg.p_init_cnv,
        states=cfg.states,
        backend=cfg.backend,
        chrom=chrom_arr,
        pos=pos,
    )
