import numpy as np
import multiprocessing
from functools import partial
from sklearn.decomposition import NMF

def _sample_n_embedding(n, X, theta_ini, phi_hat, alpha, gibbs_idx, conditional_idx, seed):
    if seed is not None:
        np.random.seed(seed + n)
        
    X_n = X[n]
    theta_ini_n = theta_ini[n]
    G = phi_hat.shape[1]
    K_tot = phi_hat.shape[0]
    
    theta_n_i = np.copy(theta_ini_n)
    theta_n_conditional = theta_ini_n[conditional_idx]
    tau_n = 1 - np.sum(theta_n_conditional)
    free_idx = [i for i in range(len(theta_ini_n)) if i not in conditional_idx]
    
    theta_n_sum = np.zeros_like(theta_ini_n)
    Z_n_sum = np.zeros((G, K_tot))
    
    max_iter = np.max(gibbs_idx)
    gibbs_set = set(gibbs_idx)
    
    from .gibbs import _rdirichlet
    
    for i in range(1, max_iter + 1):
        prob_mat = phi_hat * theta_n_i[:, np.newaxis]
        prob_mat = prob_mat / np.maximum(prob_mat.sum(axis=0), 1e-15)
        
        Z_n_i = np.zeros((G, K_tot))
        for g in range(G):
            if X_n[g] > 0:
                Z_n_i[g, :] = np.random.multinomial(X_n[g], prob_mat[:, g])
                
        Z_nk_i = Z_n_i.sum(axis=0)
        theta_free = _rdirichlet(Z_nk_i[free_idx] + alpha)
        theta_n_i[free_idx] = theta_free * tau_n
        theta_n_i[conditional_idx] = theta_n_conditional
        
        if i in gibbs_set:
            Z_n_sum += Z_n_i
            theta_n_sum += theta_n_i
            
    num_samples = len(gibbs_idx)
    return Z_n_sum / num_samples, theta_n_sum / num_samples

def run_gibbs_refEta(eta, psi_env, theta_env, X, gibbs_control, n_cores):
    K_tum = eta.shape[0]
    K_tot = K_tum + psi_env.shape[0]
    
    conditional_idx = list(range(K_tum, K_tot))
    phi_hat = np.vstack([eta, psi_env])
    
    tau = 1 - np.sum(theta_env, axis=1)
    omega_ini = np.repeat((tau / K_tum)[:, np.newaxis], K_tum, axis=1)
    theta_ini = np.hstack([omega_ini, theta_env])
    
    alpha = gibbs_control['alpha']
    from .gibbs import get_gibbs_idx
    gibbs_idx = get_gibbs_idx(gibbs_control['chain_length'], gibbs_control['burn_in'], gibbs_control['thinning'])
    seed = gibbs_control['seed']
    
    N = X.shape[0]
    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        func = partial(_sample_n_embedding, X=X, theta_ini=theta_ini, phi_hat=phi_hat, alpha=alpha, 
                       gibbs_idx=gibbs_idx, conditional_idx=conditional_idx, seed=seed)
        results = pool.map(func, range(N))
        pool.close()
        pool.join()
    else:
        results = [_sample_n_embedding(n, X, theta_ini, phi_hat, alpha, gibbs_idx, conditional_idx, seed) for n in range(N)]
        
    Z = np.array([r[0] for r in results])
    theta = np.array([r[1] for r in results])
    return Z, theta

def run_EM(eta_prior, eta_post, psi_env, theta_env, X, cycle, gibbs_control, opt_control, n_cores):
    K_tum = eta_prior.shape[0]
    sigma = opt_control['sigma']
    prior_num = -1 / (2 * sigma**2)
    
    if eta_post is None:
        eta_post = eta_prior.copy()
        
    from .gibbs import optimize_psi
    
    for cycle_idx in range(1, cycle + 1):
        print(f"Starting EM cycle # {cycle_idx}")
        # E step
        Z, theta = run_gibbs_refEta(eta_post, psi_env, theta_env, X, gibbs_control, n_cores)
        
        # M step
        Z_gt = Z[:, :, :K_tum].sum(axis=0) # G x K_tum
        eta_post = optimize_psi(eta_prior, Z_gt, prior_num, opt_control, n_cores)
        
    omega = theta[:, :K_tum]
    return {'eta_prior': eta_prior, 'eta_post': eta_post, 'omega': omega}

def learn_embedding_nmf(bp, K, cycle=50, n_cores=1, gibbs_control=None, opt_control=None, nmf_control=None):
    if not hasattr(bp, 'reference_update'):
        raise ValueError("Please update gibbs sampling first")
    if type(bp.reference_update) is not dict or 'psi_mal' not in bp.reference_update:
        raise ValueError("Please run BayesPrism using tumor mode by specifying the key parameter")
        
    if gibbs_control is None:
        gibbs_control = {'chain_length': 1000, 'burn_in': 500, 'thinning': 2, 'alpha': 1.0, 'seed': 123}
    if opt_control is None:
        opt_control = {'maxiter': 100000, 'optimizer': 'MAP', 'sigma': 2.0}
    if nmf_control is None:
        nmf_control = {'nrun': 10, 'seed': 123}
        
    psi_mal = bp.reference_update['psi_mal']
    print(f"Running NMF to initialize {K} tumor programs...")
    # NMF on X = genes x cells (in R it is t(psi_mal)). psi_mal is N x G. 
    # nmf.fit(psi_mal) will give components of shape (K, G), which is exactly eta_prior
    model = NMF(n_components=K, init='random', random_state=nmf_control['seed'], max_iter=200)
    model.fit(psi_mal)
    eta_prior = model.components_
    
    from .core import _norm_to_one
    eta_prior = _norm_to_one(eta_prior, pseudo_min=bp.pseudo_min)
    
    key_idx = list(bp.ct_names).index(bp.key)
    env_idx = [i for i in range(len(bp.ct_names)) if i != key_idx]
    
    theta_f = bp.posterior_theta_f
    theta_env = theta_f[:, env_idx]
    psi_env = bp.reference_update['psi_env']
    
    print("Running EM Embedding Learning...")
    res = run_EM(
        eta_prior=eta_prior,
        eta_post=None,
        psi_env=psi_env,
        theta_env=theta_env,
        X=bp.mixture,
        cycle=cycle,
        gibbs_control=gibbs_control,
        opt_control=opt_control,
        n_cores=n_cores
    )
    return res
