from functools import partial
import multiprocessing

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _rdirichlet(alpha, rng=None):
    if rng is None:
        return np.random.dirichlet(alpha)
    return rng.dirichlet(alpha)


def _compute_cv_from_draws(draws: np.ndarray) -> np.ndarray:
    """
    Compute posterior coefficient of variation (SD / mean across thinned Gibbs draws)
    matching Danko-Lab/BayesPrism `ThetaPost@theta.cv`.
    `draws` has shape `(n_draws, K)` or `(N, n_draws, K)`.
    """
    mean = np.mean(draws, axis=-2)
    if draws.shape[-2] > 1:
        sd = np.std(draws, axis=-2, ddof=1)
    else:
        sd = np.zeros_like(mean)
    return np.where(mean > 0.0, sd / np.maximum(mean, 1e-300), 0.0)


def _sample_Z_theta_n(
    X_n, phi, alpha, gibbs_idx, sample_Z=True, return_draws=False, rng=None
):
    """
    Vectorized collapsed/blocked Gibbs sampler for a single bulk/spatial sample n.
    Uses C-level vectorized Generator.multinomial across all non-zero genes simultaneously.
    """
    if rng is None:
        rng = np.random.default_rng()

    K, G = phi.shape
    theta_n_i = np.ones(K, dtype=float) / K

    Z_n_sum = np.zeros((G, K), dtype=float) if sample_Z else None
    theta_n_sum = np.zeros(K, dtype=float)
    num_samples = len(gibbs_idx)
    theta_draws = (
        np.empty((num_samples, K), dtype=float) if return_draws else None
    )
    draw_pos = 0

    X_int = np.rint(X_n).astype(np.int64)
    nz_mask = X_int > 0
    X_nz = X_int[nz_mask]
    phi_nz = phi[:, nz_mask]
    has_nz = X_nz.size > 0

    max_iter = int(np.max(gibbs_idx))
    gibbs_set = set(int(idx) for idx in gibbs_idx)

    for i in range(1, max_iter + 1):
        if has_nz:
            prob_mat = phi_nz * theta_n_i[:, np.newaxis]  # (K, G_nz)
            col_sums = prob_mat.sum(axis=0, keepdims=True)
            pvals = np.where(
                col_sums > 0, prob_mat / np.maximum(col_sums, 1e-300), 1.0 / K
            ).T  # (G_nz, K)
            pvals = pvals / pvals.sum(axis=1, keepdims=True)
            Z_nz = rng.multinomial(X_nz, pvals)  # (G_nz, K)
            Z_nk_i = Z_nz.sum(axis=0)
        else:
            Z_nz = None
            Z_nk_i = np.zeros(K, dtype=float)

        theta_n_i = rng.dirichlet(Z_nk_i + alpha)

        if i in gibbs_set:
            if sample_Z and has_nz:
                Z_n_sum[nz_mask] += Z_nz
            theta_n_sum += theta_n_i
            if theta_draws is not None:
                theta_draws[draw_pos] = theta_n_i
                draw_pos += 1

    res_theta = theta_n_sum / num_samples
    res_Z = (Z_n_sum / num_samples) if sample_Z else None
    if return_draws:
        return res_Z, res_theta, theta_draws
    return res_Z, res_theta


def _worker_initial_gibbs(
    n, X, phi, alpha, gibbs_idx, sample_Z, seed, return_draws=False
):
    rng = np.random.default_rng(seed + n if seed is not None else None)
    return _sample_Z_theta_n(
        X[n],
        phi,
        alpha,
        gibbs_idx,
        sample_Z=sample_Z,
        return_draws=return_draws,
        rng=rng,
    )


def _worker_tumor_gibbs(
    n, X, psi_mal, psi_env, alpha, gibbs_idx, seed, return_draws=False
):
    rng = np.random.default_rng(seed + n if seed is not None else None)
    phi_n = np.vstack([psi_mal[n], psi_env])
    nonzero_idx = phi_n.max(axis=0) > 0

    out = _sample_Z_theta_n(
        X[n, nonzero_idx],
        phi_n[:, nonzero_idx],
        alpha,
        gibbs_idx,
        sample_Z=False,
        return_draws=return_draws,
        rng=rng,
    )
    if return_draws:
        _, res_theta, draws = out
        return res_theta, _compute_cv_from_draws(draws)
    return out[1]


def get_gibbs_idx(chain_length, burn_in, thinning):
    return np.arange(burn_in + 1, chain_length + 1, thinning)


def run_gibbs(
    X,
    phi,
    alpha,
    chain_length,
    burn_in,
    thinning,
    seed,
    n_cores,
    sample_Z,
    return_draws=False,
):
    N = X.shape[0]
    gibbs_idx = get_gibbs_idx(chain_length, burn_in, thinning)

    if n_cores > 1:
        with multiprocessing.Pool(n_cores) as pool:
            func = partial(
                _worker_initial_gibbs,
                X=X,
                phi=phi,
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                sample_Z=sample_Z,
                seed=seed,
                return_draws=return_draws,
            )
            results = pool.map(func, range(N))
    else:
        results = [
            _worker_initial_gibbs(
                n, X, phi, alpha, gibbs_idx, sample_Z, seed, return_draws=return_draws
            )
            for n in range(N)
        ]

    Z = np.array([r[0] for r in results]) if sample_Z else None
    theta = np.array([r[1] for r in results])
    if return_draws:
        draws = np.array([r[2] for r in results])
        return Z, theta, draws
    return Z, theta


def run_gibbs_tumor(
    X,
    psi_mal,
    psi_env,
    alpha,
    chain_length,
    burn_in,
    thinning,
    seed,
    n_cores,
    key_idx=0,
    return_cv=False,
):
    """
    Run final Gibbs sampling for tumor mode and map columns back to canonical
    ct_names order (placing malignant column at key_idx and non-malignant columns
    at env_idx = [i for i in range(K_ct) if i != key_idx]).
    """
    N = X.shape[0]
    gibbs_idx = get_gibbs_idx(chain_length, burn_in, thinning)

    if n_cores > 1:
        with multiprocessing.Pool(n_cores) as pool:
            func = partial(
                _worker_tumor_gibbs,
                X=X,
                psi_mal=psi_mal,
                psi_env=psi_env,
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                seed=seed,
                return_draws=return_cv,
            )
            results = pool.map(func, range(N))
    else:
        results = [
            _worker_tumor_gibbs(
                n,
                X,
                psi_mal,
                psi_env,
                alpha,
                gibbs_idx,
                seed,
                return_draws=return_cv,
            )
            for n in range(N)
        ]

    K_ct = 1 + psi_env.shape[0]
    env_idx = [i for i in range(K_ct) if i != key_idx]

    if return_cv:
        theta_raw = np.array([r[0] for r in results])
        cv_raw = np.array([r[1] for r in results])
        theta_ordered = np.zeros((N, K_ct), dtype=theta_raw.dtype)
        cv_ordered = np.zeros((N, K_ct), dtype=cv_raw.dtype)
        theta_ordered[:, key_idx] = theta_raw[:, 0]
        theta_ordered[:, env_idx] = theta_raw[:, 1:]
        cv_ordered[:, key_idx] = cv_raw[:, 0]
        cv_ordered[:, env_idx] = cv_raw[:, 1:]
        return theta_ordered, cv_ordered

    theta_raw = np.array(results)
    theta_ordered = np.zeros((N, K_ct), dtype=theta_raw.dtype)
    theta_ordered[:, key_idx] = theta_raw[:, 0]
    theta_ordered[:, env_idx] = theta_raw[:, 1:]
    return theta_ordered


# ---- Reference Optimization Functions ----


def _log_posterior_gamma(gamma_t, phi_t_log, Z_gt_t, Z_t_t, prior_num):
    x = phi_t_log + gamma_t
    psi_t_log = x - logsumexp(x)
    log_likelihood = np.sum(Z_gt_t * psi_t_log)
    log_prior = np.sum(prior_num * gamma_t**2)
    return -(log_likelihood + log_prior)


def _log_posterior_gamma_grad(gamma_t, phi_t_log, Z_gt_t, Z_t_t, prior_num):
    x = phi_t_log + gamma_t
    psi_t = np.exp(x - logsumexp(x))
    log_likelihood_grad = Z_gt_t - (Z_t_t * psi_t)
    log_prior_grad = 2.0 * prior_num * gamma_t
    return -(log_likelihood_grad + log_prior_grad)


def _optimize_psi_t(t, phi, Z_gt, prior_num, opt_control):
    phi_t = phi[t]
    phi_t_log = np.log(np.maximum(phi_t, 1e-15))
    Z_gt_t = Z_gt[:, t]
    Z_t_t = np.sum(Z_gt_t)

    init_gamma = np.zeros(len(phi_t), dtype=float)
    res = minimize(
        fun=_log_posterior_gamma,
        x0=init_gamma,
        args=(phi_t_log, Z_gt_t, Z_t_t, prior_num),
        jac=_log_posterior_gamma_grad,
        method="L-BFGS-B",
        options={"maxiter": opt_control.get("maxiter", 100000)},
    )

    opt_gamma = res.x
    if (not np.all(np.isfinite(opt_gamma))) or np.max(np.abs(opt_gamma)) > 20:
        opt_gamma = np.zeros_like(opt_gamma)

    x = phi_t_log + opt_gamma
    psi_t = np.exp(x - logsumexp(x))
    return psi_t


def optimize_psi(phi, Z_gt, prior_num, opt_control, n_cores):
    """Cell-type-specific MAP optimization of log-fold-change gamma_t per cell state/type."""
    T, _ = phi.shape
    psi = np.zeros_like(phi)

    if n_cores > 1:
        with multiprocessing.Pool(n_cores) as pool:
            func = partial(
                _optimize_psi_t,
                phi=phi,
                Z_gt=Z_gt,
                prior_num=prior_num,
                opt_control=opt_control,
            )
            results = pool.map(func, range(T))
        for t in range(T):
            psi[t] = results[t]
    else:
        for t in range(T):
            psi[t] = _optimize_psi_t(t, phi, Z_gt, prior_num, opt_control)

    return psi


def _neg_ll_one_gamma(gamma, phi_log, Z_gt_T, Z_g, Z_t):
    x = phi_log + gamma[np.newaxis, :]
    psi_log = x - logsumexp(x, axis=1, keepdims=True)
    return -np.sum(Z_gt_T * psi_log)


def _neg_ll_one_gamma_grad(gamma, phi_log, Z_gt_T, Z_g, Z_t):
    x = phi_log + gamma[np.newaxis, :]
    psi = np.exp(x - logsumexp(x, axis=1, keepdims=True))
    return -(Z_g - (Z_t @ psi))


def optimize_psi_one_gamma(phi, Z_gt, opt_control, pseudo_min=1e-8):
    """
    Optimize a single shared log-scaling vector gamma in R^G across all cell states/types
    under the multinomial log-likelihood (BayesPrism 'MLE' / spatial transcriptomics mode).
    """
    from .core import _norm_to_one

    T, G = phi.shape
    Z_g = Z_gt.sum(axis=1)  # (G,)
    Z_t = Z_gt.sum(axis=0)  # (T,)

    idx_nz = Z_g > 0
    n_nz = int(np.sum(idx_nz))
    if n_nz == 0:
        return phi.copy()
    if n_nz == 1:
        psi = np.zeros_like(phi)
        psi[:, idx_nz] = 1.0
        return _norm_to_one(psi, pseudo_min=pseudo_min)

    phi_nz = phi[:, idx_nz]
    phi_nz_log = np.log(np.maximum(phi_nz, 1e-15))
    Z_gt_nz_T = Z_gt[idx_nz, :].T  # (T, G_nz)
    Z_g_nz = Z_g[idx_nz]

    init_gamma = np.zeros(n_nz, dtype=float)
    res = minimize(
        fun=_neg_ll_one_gamma,
        x0=init_gamma,
        args=(phi_nz_log, Z_gt_nz_T, Z_g_nz, Z_t),
        jac=_neg_ll_one_gamma_grad,
        method="L-BFGS-B",
        options={"maxiter": opt_control.get("maxiter", 100000)},
    )

    opt_gamma = res.x - np.mean(res.x)
    if (not np.all(np.isfinite(opt_gamma))) or np.max(np.abs(opt_gamma)) > 50:
        opt_gamma = np.zeros_like(opt_gamma)

    x_nz = phi_nz_log + opt_gamma[np.newaxis, :]
    psi_nz = np.exp(x_nz - logsumexp(x_nz, axis=1, keepdims=True))

    psi = np.zeros_like(phi)
    psi[:, idx_nz] = psi_nz
    return _norm_to_one(psi, pseudo_min=pseudo_min)


def update_reference(
    Z_ct, phi_prime, ct_names, key, opt_control, pseudo_min, n_cores
):
    _, _, K_ct = Z_ct.shape
    optimizer = opt_control.get("optimizer", "MAP")
    sigma = opt_control.get("sigma", 2.0)
    prior_num = -1.0 / (2.0 * sigma**2)

    from .core import _norm_to_one

    if key is None:
        Z_gt = Z_ct.sum(axis=0)  # (G, K_ct)
        if optimizer == "MLE":
            return optimize_psi_one_gamma(
                phi_prime, Z_gt, opt_control, pseudo_min=pseudo_min
            )
        return optimize_psi(phi_prime, Z_gt, prior_num, opt_control, n_cores)
    else:
        key_idx = list(ct_names).index(key)
        env_idx = [i for i in range(K_ct) if i != key_idx]

        # Sample-specific MLE for malignant cells
        Z_ng_mal = Z_ct[:, :, key_idx]
        mle_psi_mal = Z_ng_mal / np.maximum(Z_ng_mal.sum(axis=1, keepdims=True), 1e-15)
        psi_mal = _norm_to_one(mle_psi_mal, pseudo_min)

        # Reference update for non-malignant microenvironment
        Z_gt_env = Z_ct[:, :, env_idx].sum(axis=0)
        phi_env = phi_prime[env_idx, :]
        if optimizer == "MLE":
            psi_env = optimize_psi_one_gamma(
                phi_env, Z_gt_env, opt_control, pseudo_min=pseudo_min
            )
        else:
            psi_env = optimize_psi(phi_env, Z_gt_env, prior_num, opt_control, n_cores)

        return {"psi_mal": psi_mal, "psi_env": psi_env}
