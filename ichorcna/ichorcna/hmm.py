"""
Student-t Hidden Markov Model, chromosome-batched scaled Forward-Backward,
L-BFGS-B Generalized M-step, and Viterbi segmentation for `ichorcna`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, special

from .io import HMMState, IchorCNAConfig

LN2 = float(np.log(2.0))


def emission_means(
    n: float,
    phi: float,
    s: float,
    copy_numbers: np.ndarray,
    is_subclonal: np.ndarray,
) -> np.ndarray:
    """
    Compute exact `ichorCNA` state emission log2 copy ratios:

        mu_k(n, phi, s_k) = log2( (2*n + (1 - n)*(2*(1 - s_k) + s_k * c_k)) / (2*n + (1 - n)*phi) )

    where `s_k = s` for subclonal states and `s_k = 1.0` for clonal states.
    """
    sk = np.where(is_subclonal, float(s), 1.0)
    c_eff = 2.0 * (1.0 - sk) + sk * copy_numbers
    num = np.maximum(2.0 * n + (1.0 - n) * c_eff, 1e-8)
    den = max(2.0 * n + (1.0 - n) * phi, 1e-8)
    return np.log2(num / den)


def student_t_logpdf(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float = 2.1,
) -> np.ndarray:
    """
    Vectorized Student-t log-density for observations `y` (shape `(T,)`) across
    `K` states (`mu`, `sigma` of shape `(K,)`), returning `(T, K)`.
    """
    y_col = y[:, None]
    mu_row = mu[None, :]
    sig_row = np.maximum(sigma[None, :], 1e-6)
    const = (
        special.gammaln(0.5 * (nu + 1.0))
        - special.gammaln(0.5 * nu)
        - 0.5 * np.log(np.pi * nu)
    )
    z2 = ((y_col - mu_row) / sig_row) ** 2
    return const - np.log(sig_row) - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)


@dataclass(frozen=True)
class ChromBatch:
    """Pre-indexed chromosome-segmented layout for fast batched HMM inference."""

    n_bins: int
    n_chroms: int
    max_len: int
    lengths: np.ndarray
    flat_idx: np.ndarray
    valid_mask: np.ndarray


def build_chrom_batch(chroms: Sequence[object]) -> ChromBatch:
    """Partition contiguous chromosome stretches into a padded 2D batch index."""
    arr = np.asarray(chroms)
    n_bins = arr.size
    if n_bins == 0:
        raise ValueError("Cannot build HMM chromosome batch from empty bins.")

    starts = [0]
    for i in range(1, n_bins):
        if arr[i] != arr[i - 1]:
            starts.append(i)
    starts.append(n_bins)

    n_chroms = len(starts) - 1
    lengths = np.array(
        [starts[c + 1] - starts[c] for c in range(n_chroms)], dtype=np.int64
    )
    max_len = int(np.max(lengths))

    flat_idx = np.zeros((n_chroms, max_len), dtype=np.int64)
    valid_mask = np.zeros((n_chroms, max_len), dtype=bool)
    for c in range(n_chroms):
        st, en = starts[c], starts[c + 1]
        ln = en - st
        flat_idx[c, :ln] = np.arange(st, en, dtype=np.int64)
        valid_mask[c, :ln] = True

    return ChromBatch(
        n_bins=n_bins,
        n_chroms=n_chroms,
        max_len=max_len,
        lengths=lengths,
        flat_idx=flat_idx,
        valid_mask=valid_mask,
    )


def build_prior_transition_matrix(
    states: list[HMMState],
    txn_e: float,
    is_subclonal: np.ndarray,
) -> np.ndarray:
    """
    Construct base transition matrix parameterized by `txn_e` on the diagonal
    and off-diagonal transition probabilities weighted modestly toward clonal states.
    """
    k = len(states)
    if k == 1:
        return np.ones((1, 1), dtype=np.float64)

    off_weights = np.where(is_subclonal, 0.35, 1.0)
    a_mat = np.zeros((k, k), dtype=np.float64)
    for j in range(k):
        a_mat[j, j] = txn_e
        w_off = off_weights.copy()
        w_off[j] = 0.0
        w_sum = float(np.sum(w_off))
        if w_sum > 0.0:
            a_mat[j, :] += (1.0 - txn_e) * (w_off / w_sum)
    return a_mat


def forward_backward_batched(
    log_emit: np.ndarray,
    pi0: np.ndarray,
    trans_mat: np.ndarray,
    batch: ChromBatch,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Exact log-shifted, scaled Forward-Backward algorithm resetting at chromosome
    boundaries.

    Returns `(loglik, gamma, xi_counts)` where `gamma` has shape `(T, K)` and
    `xi_counts` has shape `(K, K)`.
    """
    k = log_emit.shape[1]
    nc, max_l = batch.n_chroms, batch.max_len
    vmask = batch.valid_mask

    le_pad = np.zeros((nc, max_l, k), dtype=np.float64)
    le_pad[vmask] = log_emit[batch.flat_idx[vmask]]

    m_shift = np.max(le_pad, axis=2)
    lik = np.exp(le_pad - m_shift[:, :, None])
    lik[~vmask] = 1.0

    alpha = np.zeros((nc, max_l, k), dtype=np.float64)
    scale = np.ones((nc, max_l), dtype=np.float64)

    # t = 0 (all chromosomes have length >= 1)
    raw0 = pi0[None, :] * lik[:, 0, :]
    sc0 = np.maximum(np.sum(raw0, axis=1), 1e-300)
    scale[:, 0] = sc0
    alpha[:, 0, :] = raw0 / sc0[:, None]

    for t in range(1, max_l):
        act = vmask[:, t]
        if not np.any(act):
            break
        pred = alpha[act, t - 1, :] @ trans_mat
        raw = pred * lik[act, t, :]
        sc = np.maximum(np.sum(raw, axis=1), 1e-300)
        scale[act, t] = sc
        alpha[act, t, :] = raw / sc[:, None]

    loglik = float(np.sum((np.log(scale) + m_shift)[vmask]))

    beta = np.ones((nc, max_l, k), dtype=np.float64)
    xi_counts = np.zeros((k, k), dtype=np.float64)
    trans_t = trans_mat.T

    for t in range(max_l - 2, -1, -1):
        act = vmask[:, t + 1]
        if not np.any(act):
            continue
        b_next = (beta[act, t + 1, :] * lik[act, t + 1, :]) / scale[act, t + 1, None]
        beta[act, t, :] = b_next @ trans_t
        xi_counts += trans_mat * (alpha[act, t, :].T @ b_next)

    gamma_pad = alpha * beta
    g_sum = np.maximum(np.sum(gamma_pad, axis=2, keepdims=True), 1e-300)
    gamma_pad /= g_sum

    gamma = np.zeros((batch.n_bins, k), dtype=np.float64)
    gamma[batch.flat_idx[vmask]] = gamma_pad[vmask]
    return loglik, gamma, xi_counts


def viterbi_batched(
    log_emit: np.ndarray,
    pi0: np.ndarray,
    trans_mat: np.ndarray,
    batch: ChromBatch,
) -> np.ndarray:
    """Exact log-space Viterbi decoding resetting at chromosome boundaries."""
    k = log_emit.shape[1]
    nc, max_l = batch.n_chroms, batch.max_len
    vmask = batch.valid_mask

    le_pad = np.zeros((nc, max_l, k), dtype=np.float64)
    le_pad[vmask] = log_emit[batch.flat_idx[vmask]]

    log_pi0 = np.log(np.maximum(pi0, 1e-300))
    log_a = np.log(np.maximum(trans_mat, 1e-300))

    dp = np.zeros((nc, max_l, k), dtype=np.float64)
    ptr = np.zeros((nc, max_l, k), dtype=np.int64)
    dp[:, 0, :] = log_pi0[None, :] + le_pad[:, 0, :]

    for t in range(1, max_l):
        act = vmask[:, t]
        if not np.any(act):
            break
        # scores: (n_act, prev_k, next_k)
        scores = dp[act, t - 1, :, None] + log_a[None, :, :]
        best_prev = np.argmax(scores, axis=1)
        ptr[act, t, :] = best_prev
        n_act = int(np.sum(act))
        best_vals = scores[np.arange(n_act)[:, None], best_prev, np.arange(k)[None, :]]
        dp[act, t, :] = best_vals + le_pad[act, t, :]

    states_out = np.zeros(batch.n_bins, dtype=np.int64)
    for c in range(nc):
        ln = int(batch.lengths[c])
        path_c = np.zeros(ln, dtype=np.int64)
        path_c[ln - 1] = int(np.argmax(dp[c, ln - 1, :]))
        for t in range(ln - 2, -1, -1):
            path_c[t] = ptr[c, t + 1, path_c[t + 1]]
        states_out[batch.flat_idx[c, :ln]] = path_c

    return states_out


def estimate_empirical_noise(y: np.ndarray, batch: ChromBatch) -> float:
    """Robustly estimate per-bin logR noise SD from within-chromosome first differences."""
    diffs_list: list[np.ndarray] = []
    for c in range(batch.n_chroms):
        ln = int(batch.lengths[c])
        if ln >= 2:
            idx = batch.flat_idx[c, :ln]
            diffs_list.append(np.diff(y[idx]))
    if not diffs_list:
        mad = float(np.median(np.abs(y - np.median(y))))
        return max(mad / 0.67448975, 0.02)
    all_diffs = np.concatenate(diffs_list)
    mad_diff = float(np.median(np.abs(all_diffs - np.median(all_diffs))))
    sigma0 = mad_diff / (np.sqrt(2.0) * 0.67448975)
    return float(np.clip(sigma0, 0.015, 0.35))


@dataclass
class EMRunResult:
    """Single converged `(normal_init, ploidy_init)` HMM + Generalized EM trajectory."""

    normal_init: float
    ploidy_init: float
    normal_fraction: float
    tumor_fraction: float
    ploidy: float
    subclone_fraction: float
    sigmas: np.ndarray
    pi0: np.ndarray
    trans_mat: np.ndarray
    gamma: np.ndarray
    viterbi_states: np.ndarray
    loglik: float
    log_posterior: float
    n_iter: int


def fit_single_ichorcna_em(
    y: np.ndarray,
    states: list[HMMState],
    batch: ChromBatch,
    normal_init: float,
    ploidy_init: float,
    cfg: IchorCNAConfig,
    sigma0: float,
) -> EMRunResult:
    """Run Student-t HMM Generalized EM from a single `(normal_init, ploidy_init)` start."""
    k = len(states)
    copy_numbers = np.array([st.copy_number for st in states], dtype=np.float64)
    is_subclonal = np.array([st.is_subclonal for st in states], dtype=bool)
    has_subclone = bool(np.any(is_subclonal))

    # Biological copy-number state prior: diploid NEUT (c=2) baseline, moderate penalty on HLAMP
    pi0_prior = np.ones(k, dtype=np.float64)
    state_log_pen = np.zeros(k, dtype=np.float64)
    for idx_s, st in enumerate(states):
        if st.is_subclonal:
            pi0_prior[idx_s] = 0.35
            state_log_pen[idx_s] = -0.005
        elif st.copy_number == 2:
            pi0_prior[idx_s] = 3.5
        elif st.copy_number in (1, 3):
            pi0_prior[idx_s] = 1.2
        elif st.copy_number == 4:
            pi0_prior[idx_s] = 0.7
            state_log_pen[idx_s] = -0.005
        elif st.copy_number == 0:
            pi0_prior[idx_s] = 0.5
        elif st.copy_number >= 5:
            pi0_prior[idx_s] = 0.3
            state_log_pen[idx_s] = -0.025
    pi0_prior = pi0_prior / np.sum(pi0_prior)
    pi0 = pi0_prior.copy()

    a_prior = build_prior_transition_matrix(states, cfg.txn_e, is_subclonal)
    trans_mat = a_prior.copy()

    n_val = float(np.clip(normal_init, 0.01, 0.995))
    phi_val = float(np.clip(ploidy_init, 1.5, 5.5))
    s_val = 0.50
    sigmas = np.full(k, sigma0, dtype=np.float64)
    nu = cfg.student_df

    prev_obj = -np.inf
    loglik = -np.inf
    gamma = np.full((y.size, k), 1.0 / k, dtype=np.float64)
    n_iter_done = 0

    for it in range(1, cfg.max_iter + 1):
        n_iter_done = it
        mu = emission_means(n_val, phi_val, s_val, copy_numbers, is_subclonal)
        log_emit = student_t_logpdf(y, mu, sigmas, nu=nu) + state_log_pen[None, :]
        loglik, gamma, xi_counts = forward_backward_batched(log_emit, pi0, trans_mat, batch)

        # Student-t latent scale-mixture precision weights w_{t,k} = E[u_{t,k} | y_t, z_t=k]
        resid2 = (y[:, None] - mu[None, :]) ** 2
        w_lat = (nu + 1.0) / (nu + resid2 / np.maximum(sigmas[None, :] ** 2, 1e-12))
        gw = gamma * w_lat

        # Sufficient statistics per state for O(K) L-BFGS-B evaluation
        wk = np.sum(gw, axis=0)
        sk_sum = np.sum(gw * y[:, None], axis=0)
        inv_sig2 = 1.0 / np.maximum(sigmas ** 2, 1e-12)

        # Prior log-density breaking exact (c_k + 1, phi + 1) gauge symmetry toward diploid
        prior_lp = (
            0.05 * (np.log(n_val) + np.log(1.0 - n_val))
            - 2.5 * ((phi_val - 2.0) ** 2)
            + (2.0 * (np.log(s_val) + np.log(1.0 - s_val)) if has_subclone else 0.0)
        )
        curr_obj = loglik + prior_lp
        if it > 2 and abs(curr_obj - prev_obj) < cfg.em_tol:
            break
        prev_obj = curr_obj

        # M-step 1: Optimize (n, phi[, s]) via L-BFGS-B with exact analytic gradient
        if has_subclone:
            x0 = np.array([n_val, phi_val, s_val], dtype=np.float64)
            bounds = [(0.01, 0.995), (1.5, 5.5), (0.30, 0.70)]
        else:
            x0 = np.array([n_val, phi_val], dtype=np.float64)
            bounds = [(0.01, 0.995), (1.5, 5.5)]

        def neg_q_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            nv = float(params[0])
            pv = float(params[1])
            sv = float(params[2]) if has_subclone else 0.5

            sk_eff = np.where(is_subclonal, sv, 1.0)
            c_eff = 2.0 * (1.0 - sk_eff) + sk_eff * copy_numbers
            num = np.maximum(2.0 * nv + (1.0 - nv) * c_eff, 1e-8)
            den = max(2.0 * nv + (1.0 - nv) * pv, 1e-8)
            muk = np.log2(num / den)

            quad = 0.5 * np.sum(inv_sig2 * (wk * (muk ** 2) - 2.0 * sk_sum * muk))
            p_pen = (
                -0.05 * (np.log(nv) + np.log(1.0 - nv))
                + 2.5 * ((pv - 2.0) ** 2)
            )
            if has_subclone:
                p_pen -= 2.0 * (np.log(sv) + np.log(1.0 - sv))

            coeff = inv_sig2 * (wk * muk - sk_sum)
            dmu_dn = ((2.0 - c_eff) / num - (2.0 - pv) / den) / LN2
            dmu_dp = -(1.0 - nv) / (den * LN2)

            gn = float(np.sum(coeff * dmu_dn)) - 0.05 * (1.0 / nv - 1.0 / (1.0 - nv))
            gp = float(np.sum(coeff * dmu_dp)) + 5.0 * (pv - 2.0)

            if has_subclone:
                dmu_ds = np.where(
                    is_subclonal,
                    (1.0 - nv) * (copy_numbers - 2.0) / (num * LN2),
                    0.0,
                )
                gs = float(np.sum(coeff * dmu_ds)) - 2.0 * (1.0 / sv - 1.0 / (1.0 - sv))
                return float(quad + p_pen), np.array([gn, gp, gs], dtype=np.float64)

            return float(quad + p_pen), np.array([gn, gp], dtype=np.float64)

        opt = optimize.minimize(
            neg_q_and_grad,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 15, "ftol": 1e-7},
        )
        n_val = float(opt.x[0])
        phi_val = float(opt.x[1])
        if has_subclone:
            s_val = float(opt.x[2])

        # M-step 2: Closed-form MAP Student-t scale update anchored near empirical sigma0
        mu_new = emission_means(n_val, phi_val, s_val, copy_numbers, is_subclonal)
        rss_weighted = np.sum(gw * ((y[:, None] - mu_new[None, :]) ** 2), axis=0)
        gk_sum = np.sum(gamma, axis=0)
        prior_df = 50.0
        pooled_var = (prior_df * (sigma0 ** 2) + float(np.sum(rss_weighted))) / (
            prior_df + float(np.sum(gk_sum))
        )
        state_var = (25.0 * pooled_var + rss_weighted) / (25.0 + gk_sum)
        sigmas = np.clip(np.sqrt(state_var), 0.5 * sigma0, 2.2 * sigma0)

        # M-step 3: Initial distribution and Dirichlet-regularized transition matrix
        chrom_start_gamma = gamma[batch.flat_idx[:, 0], :]
        pi0 = np.sum(chrom_start_gamma, axis=0) + 5.0 * pi0_prior
        pi0 /= np.sum(pi0)

        row_counts = xi_counts + cfg.txn_strength * a_prior
        trans_mat = row_counts / np.maximum(np.sum(row_counts, axis=1, keepdims=True), 1e-300)

    # Final E-step (data log-likelihood + posterior objective) and Viterbi segmentation
    mu_final = emission_means(n_val, phi_val, s_val, copy_numbers, is_subclonal)
    raw_emit_final = student_t_logpdf(y, mu_final, sigmas, nu=nu)
    data_loglik, gamma_final, _ = forward_backward_batched(raw_emit_final, pi0, trans_mat, batch)
    pen_emit_final = raw_emit_final + state_log_pen[None, :]
    pen_ll, _, _ = forward_backward_batched(pen_emit_final, pi0, trans_mat, batch)
    vit_emit = raw_emit_final.copy()
    if has_subclone and (1.0 - n_val) * s_val < 0.025:
        # Subclonal plasma fraction < 2.5% is below ULP-WGS noise floor (Adalsteinsson et al. 2017)
        vit_emit[:, is_subclonal] -= 0.15
    vit = viterbi_batched(vit_emit, pi0, trans_mat, batch)

    # Also enforce self-consistency penalty between phi_val and Viterbi mean tumor copy number
    sk_vec = np.where(is_subclonal, s_val, 1.0)
    c_eff_vit = 2.0 * (1.0 - sk_vec[vit]) + sk_vec[vit] * copy_numbers[vit]
    mean_vit_cn = float(np.mean(c_eff_vit))
    final_prior = (
        0.05 * (np.log(n_val) + np.log(1.0 - n_val))
        - 2.5 * ((phi_val - 2.0) ** 2)
        - 4.0 * ((phi_val - mean_vit_cn) ** 2)
    )

    return EMRunResult(
        normal_init=normal_init,
        ploidy_init=ploidy_init,
        normal_fraction=n_val,
        tumor_fraction=1.0 - n_val,
        ploidy=phi_val,
        subclone_fraction=s_val if has_subclone else 0.0,
        sigmas=sigmas,
        pi0=pi0,
        trans_mat=trans_mat,
        gamma=gamma_final,
        viterbi_states=vit,
        loglik=data_loglik,
        log_posterior=pen_ll + final_prior,
        n_iter=n_iter_done,
    )
