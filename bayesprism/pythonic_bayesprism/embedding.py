from functools import partial
import multiprocessing

import numpy as np
from sklearn.decomposition import NMF


def _sample_n_embedding(
    n, X, theta_ini, phi_hat, alpha, gibbs_idx, conditional_idx, seed
):
    rng = np.random.default_rng(seed + n if seed is not None else None)

    X_n = X[n]
    theta_ini_n = theta_ini[n]
    K_tot, G = phi_hat.shape

    theta_n_i = np.copy(theta_ini_n)
    theta_n_conditional = theta_ini_n[conditional_idx]
    tau_n = float(np.clip(1.0 - np.sum(theta_n_conditional), 0.0, 1.0))
    cond_set = set(conditional_idx)
    free_idx = [i for i in range(len(theta_ini_n)) if i not in cond_set]

    theta_n_sum = np.zeros_like(theta_ini_n)
    Z_n_sum = np.zeros((G, K_tot), dtype=float)

    X_int = np.rint(X_n).astype(np.int64)
    nz_mask = X_int > 0
    X_nz = X_int[nz_mask]
    phi_nz = phi_hat[:, nz_mask]
    has_nz = X_nz.size > 0

    max_iter = int(np.max(gibbs_idx))
    gibbs_set = set(int(idx) for idx in gibbs_idx)

    for i in range(1, max_iter + 1):
        if has_nz:
            prob_mat = phi_nz * theta_n_i[:, np.newaxis]
            col_sums = prob_mat.sum(axis=0, keepdims=True)
            pvals = np.where(
                col_sums > 0, prob_mat / np.maximum(col_sums, 1e-300), 1.0 / K_tot
            ).T
            pvals = pvals / pvals.sum(axis=1, keepdims=True)
            Z_nz = rng.multinomial(X_nz, pvals)
            Z_nk_i = Z_nz.sum(axis=0)
        else:
            Z_nz = None
            Z_nk_i = np.zeros(K_tot, dtype=float)

        theta_free = rng.dirichlet(Z_nk_i[free_idx] + alpha)
        theta_n_i[free_idx] = theta_free * tau_n
        theta_n_i[conditional_idx] = theta_n_conditional

        if i in gibbs_set:
            if has_nz:
                Z_n_sum[nz_mask] += Z_nz
            theta_n_sum += theta_n_i

    num_samples = len(gibbs_idx)
    return Z_n_sum / num_samples, theta_n_sum / num_samples


def run_gibbs_refEta(eta, psi_env, theta_env, X, gibbs_control, n_cores):
    K_tum = eta.shape[0]
    K_tot = K_tum + psi_env.shape[0]

    conditional_idx = list(range(K_tum, K_tot))
    phi_hat = np.vstack([eta, psi_env])

    tau = np.clip(1.0 - np.sum(theta_env, axis=1), 0.0, 1.0)
    omega_ini = np.repeat((tau / K_tum)[:, np.newaxis], K_tum, axis=1)
    theta_ini = np.hstack([omega_ini, theta_env])

    alpha = gibbs_control["alpha"]
    from .gibbs import get_gibbs_idx

    gibbs_idx = get_gibbs_idx(
        gibbs_control["chain_length"],
        gibbs_control["burn_in"],
        gibbs_control["thinning"],
    )
    seed = gibbs_control["seed"]

    N = X.shape[0]
    if n_cores > 1:
        with multiprocessing.Pool(n_cores) as pool:
            func = partial(
                _sample_n_embedding,
                X=X,
                theta_ini=theta_ini,
                phi_hat=phi_hat,
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                conditional_idx=conditional_idx,
                seed=seed,
            )
            results = pool.map(func, range(N))
    else:
        results = [
            _sample_n_embedding(
                n, X, theta_ini, phi_hat, alpha, gibbs_idx, conditional_idx, seed
            )
            for n in range(N)
        ]

    Z = np.array([r[0] for r in results])
    theta = np.array([r[1] for r in results])
    return Z, theta


def run_EM(
    eta_prior,
    eta_post,
    psi_env,
    theta_env,
    X,
    cycle,
    gibbs_control,
    opt_control,
    n_cores,
):
    K_tum = eta_prior.shape[0]
    sigma = opt_control.get("sigma", 2.0)
    prior_num = -1.0 / (2.0 * sigma**2)

    if eta_post is None:
        eta_post = eta_prior.copy()

    from .gibbs import optimize_psi

    theta = None
    for _ in range(1, cycle + 1):
        Z, theta = run_gibbs_refEta(
            eta_post, psi_env, theta_env, X, gibbs_control, n_cores
        )
        Z_gt = Z[:, :, :K_tum].sum(axis=0)  # (G, K_tum)
        eta_post = optimize_psi(eta_prior, Z_gt, prior_num, opt_control, n_cores)

    omega = theta[:, :K_tum]
    return {"eta_prior": eta_prior, "eta_post": eta_post, "omega": omega}


def learn_embedding_nmf(
    bp,
    K,
    cycle=50,
    n_cores=1,
    gibbs_control=None,
    opt_control=None,
    nmf_control=None,
):
    from .core import DEFAULT_GIBBS_CONTROL, DEFAULT_OPT_CONTROL_MAP, _norm_to_one

    if not hasattr(bp, "reference_update"):
        raise ValueError("Please run Prism with update_gibbs=True first.")
    if (
        not isinstance(bp.reference_update, dict)
        or "psi_mal" not in bp.reference_update
    ):
        raise ValueError(
            "Please run BayesPrism in tumor mode by specifying the key parameter."
        )

    gc = {**DEFAULT_GIBBS_CONTROL, **(gibbs_control or {})}
    oc = {**DEFAULT_OPT_CONTROL_MAP, **(opt_control or {})}
    nc = {"nrun": 10, "seed": 123, **(nmf_control or {})}

    psi_mal = bp.reference_update["psi_mal"]
    model = NMF(
        n_components=K, init="random", random_state=nc["seed"], max_iter=200
    )
    model.fit(psi_mal)
    eta_prior = _norm_to_one(model.components_, pseudo_min=bp.pseudo_min)

    key_idx = list(bp.ct_names).index(bp.key)
    env_idx = [i for i in range(len(bp.ct_names)) if i != key_idx]

    # bp.posterior_theta_f columns are always ordered identically to bp.ct_names
    theta_f = bp.posterior_theta_f
    theta_env = theta_f[:, env_idx]
    psi_env = bp.reference_update["psi_env"]

    return run_EM(
        eta_prior=eta_prior,
        eta_post=None,
        psi_env=psi_env,
        theta_env=theta_env,
        X=bp.mixture,
        cycle=cycle,
        gibbs_control=gc,
        opt_control=oc,
        n_cores=n_cores,
    )
