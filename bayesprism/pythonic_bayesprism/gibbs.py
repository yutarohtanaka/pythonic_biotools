import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import multiprocessing
from functools import partial

def _rdirichlet(alpha):
    # Draw from dirichlet
    return np.random.dirichlet(alpha)

def _sample_Z_theta_n(X_n, phi, alpha, gibbs_idx, sample_Z=True):
    K, G = phi.shape
    theta_n_i = np.ones(K) / K
    
    Z_n_sum = np.zeros((G, K))
    theta_n_sum = np.zeros(K)
    
    max_iter = np.max(gibbs_idx)
    gibbs_set = set(gibbs_idx)
    
    for i in range(1, max_iter + 1):
        prob_mat = phi * theta_n_i[:, np.newaxis] # K x G
        prob_mat = prob_mat / np.maximum(prob_mat.sum(axis=0), 1e-15)
        
        # Sample Z
        Z_n_i = np.zeros((G, K))
        for g in range(G):
            if X_n[g] > 0:
                Z_n_i[g, :] = np.random.multinomial(X_n[g], prob_mat[:, g])
                
        # Sample theta
        Z_nk_i = Z_n_i.sum(axis=0)
        theta_n_i = _rdirichlet(Z_nk_i + alpha)
        
        if i in gibbs_set:
            if sample_Z:
                Z_n_sum += Z_n_i
            theta_n_sum += theta_n_i
            
    num_samples = len(gibbs_idx)
    res_theta = theta_n_sum / num_samples
    if sample_Z:
        res_Z = Z_n_sum / num_samples
        return res_Z, res_theta
    else:
        return None, res_theta

def _worker_initial_gibbs(n, X, phi, alpha, gibbs_idx, sample_Z, seed):
    if seed is not None:
        np.random.seed(seed + n)
    return _sample_Z_theta_n(X[n], phi, alpha, gibbs_idx, sample_Z)

def _worker_tumor_gibbs(n, X, psi_mal, psi_env, alpha, gibbs_idx, seed):
    if seed is not None:
        np.random.seed(seed + n)
        
    phi_n = np.vstack([psi_mal[n], psi_env])
    nonzero_idx = phi_n.max(axis=0) > 0
    
    _, res_theta = _sample_Z_theta_n(X[n, nonzero_idx], phi_n[:, nonzero_idx], alpha, gibbs_idx, sample_Z=False)
    return res_theta

def get_gibbs_idx(chain_length, burn_in, thinning):
    return np.arange(burn_in + 1, chain_length + 1, thinning)

def run_gibbs(X, phi, alpha, chain_length, burn_in, thinning, seed, n_cores, sample_Z):
    N, G = X.shape
    K = phi.shape[0]
    gibbs_idx = get_gibbs_idx(chain_length, burn_in, thinning)
    
    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        func = partial(_worker_initial_gibbs, X=X, phi=phi, alpha=alpha, gibbs_idx=gibbs_idx, sample_Z=sample_Z, seed=seed)
        results = pool.map(func, range(N))
        pool.close()
        pool.join()
    else:
        results = [_worker_initial_gibbs(n, X, phi, alpha, gibbs_idx, sample_Z, seed) for n in range(N)]
        
    if sample_Z:
        Z = np.array([r[0] for r in results])
        theta = np.array([r[1] for r in results])
        return Z, theta
    else:
        theta = np.array([r[1] for r in results])
        return None, theta

def run_gibbs_tumor(X, psi_mal, psi_env, alpha, chain_length, burn_in, thinning, seed, n_cores):
    N, G = X.shape
    gibbs_idx = get_gibbs_idx(chain_length, burn_in, thinning)
    
    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        func = partial(_worker_tumor_gibbs, X=X, psi_mal=psi_mal, psi_env=psi_env, alpha=alpha, gibbs_idx=gibbs_idx, seed=seed)
        results = pool.map(func, range(N))
        pool.close()
        pool.join()
    else:
        results = [_worker_tumor_gibbs(n, X, psi_mal, psi_env, alpha, gibbs_idx, seed) for n in range(N)]
        
    theta = np.array(results)
    return theta

# ---- Optimization Functions ----

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
    log_prior_grad = 2 * prior_num * gamma_t
    return -(log_likelihood_grad + log_prior_grad)

def _optimize_psi_t(t, phi, Z_gt, prior_num, opt_control):
    phi_t = phi[t]
    phi_t_log = np.log(np.maximum(phi_t, 1e-15))
    Z_gt_t = Z_gt[:, t]
    Z_t_t = np.sum(Z_gt_t)
    
    init_gamma = np.zeros(len(phi_t))
    
    res = minimize(
        fun=_log_posterior_gamma,
        x0=init_gamma,
        args=(phi_t_log, Z_gt_t, Z_t_t, prior_num),
        jac=_log_posterior_gamma_grad,
        method='L-BFGS-B',
        options={'maxiter': opt_control['maxiter']}
    )
    
    opt_gamma = res.x
    # If extremities > 20, set to 0 (fallback to MLE essentially)
    if np.max(np.abs(opt_gamma)) > 20:
        opt_gamma = np.zeros_like(opt_gamma)
        
    x = phi_t_log + opt_gamma
    psi_t = np.exp(x - logsumexp(x))
    return psi_t

def optimize_psi(phi, Z_gt, prior_num, opt_control, n_cores):
    T, G = phi.shape
    psi = np.zeros_like(phi)
    
    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        func = partial(_optimize_psi_t, phi=phi, Z_gt=Z_gt, prior_num=prior_num, opt_control=opt_control)
        results = pool.map(func, range(T))
        pool.close()
        pool.join()
        for t in range(T):
            psi[t] = results[t]
    else:
        for t in range(T):
            psi[t] = _optimize_psi_t(t, phi, Z_gt, prior_num, opt_control)
            
    return psi

def update_reference(Z_ct, phi_prime, ct_names, key, opt_control, pseudo_min, n_cores):
    N, G, K_ct = Z_ct.shape
    sigma = opt_control['sigma']
    prior_num = -1 / (2 * sigma**2)
    
    from .core import _norm_to_one
    
    if key is None:
        Z_gt = Z_ct.sum(axis=0) # G x K_ct
        psi = optimize_psi(phi_prime, Z_gt, prior_num, opt_control, n_cores)
        return psi
    else:
        key_idx = list(ct_names).index(key)
        env_idx = [i for i in range(K_ct) if i != key_idx]
        
        # MLE for tumor
        Z_ng_mal = Z_ct[:, :, key_idx]
        mle_psi_mal = Z_ng_mal / np.maximum(Z_ng_mal.sum(axis=1, keepdims=True), 1e-15)
        psi_mal = _norm_to_one(mle_psi_mal, pseudo_min)
        
        # MAP for env
        Z_gt_env = Z_ct[:, :, env_idx].sum(axis=0)
        phi_env = phi_prime[env_idx, :]
        psi_env = optimize_psi(phi_env, Z_gt_env, prior_num, opt_control, n_cores)
        
        return {'psi_mal': psi_mal, 'psi_env': psi_env}

