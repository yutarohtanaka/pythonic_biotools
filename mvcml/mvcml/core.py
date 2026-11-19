"""Python implementation of Multivariable Constrained Maximum Likelihood (`MVMR-cML`,
`MVMR-cML-MA`, `MVMR-cML-DP`) and conditional GWAS bias adjustment (Lin et al., AJHG 2023).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _p_from_beta_se(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    return stats.chi2.sf((np.asarray(beta, dtype=float) / np.asarray(se, dtype=float)) ** 2, 1)


def _lik_terms(betax_t, betayc, SIGinv, betay_t, b, betax_obs):
    u1 = np.column_stack([betayc - betax_t @ b - betay_t, betax_obs - betax_t])
    u2 = np.column_stack([betayc - betax_t @ b, betax_obs - betax_t])
    a1 = np.einsum("ij,ijk,ik->i", u1, SIGinv, u1)
    a2 = np.einsum("ij,ijk,ik->i", u2, SIGinv, u2)
    return a2 - a1, float(a1.sum())


def _fit_cml_path(
    N: float,
    betax: np.ndarray,
    betayc: np.ndarray,
    SIGinv: np.ndarray,
    A11: np.ndarray,
    A12: np.ndarray,
    A22: np.ndarray,
    threshold: float,
    maxit: int,
    verbose: bool = False,
):
    m, p = betax.shape
    nk = m - (p + 1) + 1
    btotal = np.zeros((p, nk))
    betayesttotal = np.zeros((m, nk))
    betaxesttotal = []
    bic = np.zeros(nk)
    valid = []
    log_n = float(np.log(N))

    for k in range(nk):
        b = np.zeros(p)
        betayest = np.zeros(m)
        betaxest = np.zeros((m, p))
        c = 1
        while True:
            c += 1
            old = b.copy()
            betayest = betayc - betaxest @ b + (1.0 / A11) * np.einsum("ij,ij->i", betax - betaxest, A12)
            if k > 0:
                go, _ = _lik_terms(betaxest, betayc, SIGinv, betayest, b, betax)
                order = np.argsort(-go, kind="stable")
                keep = np.zeros(m, bool)
                keep[order[:k]] = True
                betayest = np.where(keep, betayest, 0.0)
            else:
                betayest = np.zeros(m)

            Mi = (
                A11[:, None, None] * np.outer(b, b)[None]
                + A12[:, :, None] * b[None, None, :]
                + b[None, :, None] * A12[:, None, :]
                + A22
            )
            Q = b[None, :, None] * A12[:, None, :] + A22
            rhs = np.einsum("ijk,ik->ij", Q, betax) + (betayc - betayest)[:, None] * (A12 + A11[:, None] * b[None, :])
            betaxest = np.linalg.solve(Mi, rhs[:, :, None])[:, :, 0]

            a = np.einsum("i,ij,ik->jk", A11, betaxest, betaxest)
            rb = np.einsum("i,ij->j", A11 * (betayc - betayest) + np.einsum("ij,ij->i", A12, betax - betaxest), betaxest)
            try:
                b = np.linalg.solve(a, rb)
            except np.linalg.LinAlgError:
                b = np.linalg.pinv(a) @ rb
            if np.abs(old - b).sum() < threshold or c > maxit:
                break
        btotal[:, k] = b
        betayesttotal[:, k] = betayest
        betaxesttotal.append(betaxest)
        _, like = _lik_terms(betaxest, betayc, SIGinv, betayest, b, betax)
        bic[k] = like + log_n * k
        valid.append(np.flatnonzero(betayest == 0))
        if verbose:
            print(f"[mvcml] k={k} iterations={c} BIC={bic[k]:.4f}")

    return btotal, betayesttotal, betaxesttotal, bic, valid


def _cov_omega_at_k(
    b: np.ndarray,
    validuse: np.ndarray,
    betaxest: np.ndarray,
    betax: np.ndarray,
    betayc: np.ndarray,
    SIGinv: np.ndarray,
    need_omega: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute Cov(b) and optional influence matrix Omega in O(nv * p^3) via block-diagonal C1 Schur complement."""
    nv = len(validuse)
    p = len(b)
    siginv = SIGinv[validuse]
    a11 = siginv[:, 0, 0]
    A12v = siginv[:, 0, 1:]
    A22v = siginv[:, 1:, 1:]
    bx = betaxest[validuse]
    bx_iv = betax[validuse]
    byc_v = betayc[validuse]

    # Block-diagonal C (nv, p, p) and off-diagonal B (nv, p, p)
    C = (
        a11[:, None, None] * np.outer(b, b)[None, :, :]
        + b[None, :, None] * A12v[:, None, :]
        + b[None, None, :] * A12v[:, :, None]
        + A22v
    )
    B = (b[None, :] * a11[:, None] + A12v)[:, :, None] * bx[:, None, :]
    diag_shift = (bx @ b - byc_v) * a11 - np.einsum("ij,ij->i", A12v, bx_iv - bx)
    idx_p = np.arange(p)
    B[:, idx_p, idx_p] += diag_shift[:, None]

    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.array([np.linalg.pinv(C[i]) for i in range(nv)])

    A = np.einsum("i,ij,ik->jk", a11, bx, bx)
    Bt_Cinv_B = np.einsum("ijm,ijl,ilk->mk", B, C_inv, B)
    schur = 0.5 * ((A - Bt_Cinv_B) + (A - Bt_Cinv_B).T)
    try:
        Covb = np.linalg.inv(schur)
    except np.linalg.LinAlgError:
        Covb = np.linalg.pinv(schur)
    Covb = 0.5 * (Covb + Covb.T)
    mineig = float(np.linalg.eigvalsh(Covb).min())
    if mineig < 0.0:
        Covb = Covb + (-1.001 * mineig) * np.eye(p)

    if not need_omega:
        return Covb, None

    W = np.column_stack([b, np.eye(p)])
    W_sig = np.einsum("jk,ikl->ijl", W, siginv)
    Bt_Cinv_Wsig = np.einsum("imj,iml,ilk->ijk", B, C_inv, W_sig)
    Om2 = bx_iv[:, :, None] * siginv[:, None, 0, :]
    diff_blocks = (Bt_Cinv_Wsig - Om2).transpose(1, 0, 2).reshape(p, nv * (p + 1))
    Omega = Covb @ diff_blocks
    return Covb, Omega


def cml_ma(
    N: float,
    betax: np.ndarray,
    betayc: np.ndarray,
    SIG: np.ndarray,
    threshold: float = 1e-6,
    maxit: int = 500,
    verbose: bool = False,
) -> dict:
    """Fit MVMR-cML-MA and MVMR-cML-BIC across `k = 0 .. m - (p + 1)` invalid instruments."""
    betax, betayc = np.asarray(betax, dtype=float), np.asarray(betayc, dtype=float)
    SIG = np.asarray(SIG, dtype=float)
    m, p = betax.shape
    if m < p + 1:
        raise ValueError(f"Need at least p + 1 = {p + 1} instruments, got {m}")
    SIGinv = np.linalg.inv(SIG)
    A11 = SIGinv[:, 0, 0]
    A12 = SIGinv[:, 0, 1:]
    A22 = SIGinv[:, 1:, 1:]

    btotal, _, betaxesttotal, bic, valid = _fit_cml_path(
        N, betax, betayc, SIGinv, A11, A12, A22, threshold=threshold, maxit=maxit, verbose=verbose
    )

    kbest = int(np.argmin(bic))
    b_bic = btotal[:, kbest].copy()
    validuse = valid[kbest]
    betaxest_best = betaxesttotal[kbest]
    Covb_bic, Omega_bic = _cov_omega_at_k(b_bic, validuse, betaxest_best, betax, betayc, SIGinv, need_omega=True)
    se_bic = np.sqrt(np.maximum(np.diag(Covb_bic), 0.0))

    # Genuine BIC Model Averaging: w_k = softmax(-0.5 * bic_k)
    log_w = -0.5 * (bic - float(np.min(bic)))
    ma_weights = np.exp(log_w)
    ma_weights /= np.sum(ma_weights)
    b_ma = btotal @ ma_weights

    nk = len(bic)
    Covb_ma = np.zeros((p, p), dtype=float)
    se_ma = np.zeros(p, dtype=float)
    for k in range(nk):
        wk = float(ma_weights[k])
        if wk <= 1e-12:
            continue
        Covb_k = Covb_bic if k == kbest else _cov_omega_at_k(btotal[:, k], valid[k], betaxesttotal[k], betax, betayc, SIGinv, need_omega=False)[0]
        dk = btotal[:, k] - b_ma
        Covb_ma += wk * (Covb_k + np.outer(dk, dk))
        se_ma += wk * np.sqrt(np.maximum(np.diag(Covb_k), 0.0) + dk ** 2)

    Covb_ma = 0.5 * (Covb_ma + Covb_ma.T)
    pval_ma = 2.0 * stats.norm.sf(np.abs(b_ma) / np.maximum(se_ma, 1e-300))
    pval_bic = 2.0 * stats.norm.sf(np.abs(b_bic) / np.maximum(se_bic, 1e-300))

    return dict(
        b=b_ma,
        b_ma=b_ma,
        b_bic=b_bic,
        se_ma=se_ma,
        se_bic=se_bic,
        pval_ma=pval_ma,
        pval_bic=pval_bic,
        Covb=Covb_ma,
        Covb_ma=Covb_ma,
        Covb_bic=Covb_bic,
        ma_weights=ma_weights,
        btotal=btotal,
        betaxvalid=betaxest_best[validuse],
        betaxIVvalid=betax[validuse],
        validuse=validuse,
        Omega=Omega_bic,
        bic=bic,
        k_invalid=kbest,
    )


def mvcml_dp(
    N: float,
    betax: np.ndarray,
    betayc: np.ndarray,
    SIG: np.ndarray,
    num_pert: int = 100,
    threshold: float = 1e-6,
    maxit: int = 500,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Data-Perturbation MVMR-cML (`MVmr_cML_DP`, Lin et al., AJHG 2023).

    Resamples `num_pert` perturbed summary-statistic datasets `(betayc*, betax*) ~ N((betayc, betax), SIG)`
    to capture invalid-IV model-selection uncertainty.
    """
    if num_pert < 2:
        raise ValueError("num_pert must be >= 2.")
    betax, betayc = np.asarray(betax, dtype=float), np.asarray(betayc, dtype=float)
    SIG = np.asarray(SIG, dtype=float)
    m, p = betax.shape
    if m < p + 1:
        raise ValueError(f"Need at least p + 1 = {p + 1} instruments, got {m}")

    base = cml_ma(N, betax, betayc, SIG, threshold=threshold, maxit=maxit, verbose=verbose)
    SIGinv = np.linalg.inv(SIG)
    A11 = SIGinv[:, 0, 0]
    A12 = SIGinv[:, 0, 1:]
    A22 = SIGinv[:, 1:, 1:]

    # Batch square-root factors of SIG[i] for fast perturbation sampling
    L_sig = np.empty_like(SIG)
    for i in range(m):
        si = 0.5 * (SIG[i] + SIG[i].T)
        try:
            L_sig[i] = np.linalg.cholesky(si)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(si)
            L_sig[i] = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]

    rng = np.random.default_rng(seed)
    pert_b_ma = np.empty((num_pert, p), dtype=float)
    pert_b_bic = np.empty((num_pert, p), dtype=float)

    for t in range(num_pert):
        z = rng.standard_normal((m, p + 1, 1))
        err = (L_sig @ z)[:, :, 0]
        by_pt = betayc + err[:, 0]
        bx_pt = betax + err[:, 1:]
        bt_pt, _, _, bic_pt, _ = _fit_cml_path(
            N, bx_pt, by_pt, SIGinv, A11, A12, A22, threshold=threshold, maxit=maxit, verbose=False
        )
        kb_pt = int(np.argmin(bic_pt))
        w_pt = np.exp(-0.5 * (bic_pt - float(np.min(bic_pt))))
        w_pt /= np.sum(w_pt)
        pert_b_bic[t] = bt_pt[:, kb_pt]
        pert_b_ma[t] = bt_pt @ w_pt

    b_ma_dp = pert_b_ma.mean(axis=0)
    b_bic_dp = pert_b_bic.mean(axis=0)
    Covb_ma_dp = np.atleast_2d(np.cov(pert_b_ma, rowvar=False, ddof=1)).reshape(p, p)
    Covb_bic_dp = np.atleast_2d(np.cov(pert_b_bic, rowvar=False, ddof=1)).reshape(p, p)
    se_ma_dp = np.sqrt(np.maximum(np.diag(Covb_ma_dp), 1e-30))
    se_bic_dp = np.sqrt(np.maximum(np.diag(Covb_bic_dp), 1e-30))
    pval_ma_dp = 2.0 * stats.norm.sf(np.abs(b_ma_dp) / se_ma_dp)
    pval_bic_dp = 2.0 * stats.norm.sf(np.abs(b_bic_dp) / se_bic_dp)

    return dict(
        b=b_ma_dp,
        b_se=se_ma_dp,
        Covb=Covb_ma_dp,
        pval=pval_ma_dp,
        b_ma_dp=b_ma_dp,
        se_ma_dp=se_ma_dp,
        pval_ma_dp=pval_ma_dp,
        Covb_ma_dp=Covb_ma_dp,
        b_bic_dp=b_bic_dp,
        se_bic_dp=se_bic_dp,
        pval_bic_dp=pval_bic_dp,
        Covb_bic_dp=Covb_bic_dp,
        b_ma=base["b_ma"],
        b_bic=base["b_bic"],
        se_ma=base["se_ma"],
        se_bic=base["se_bic"],
        bic=base["bic"],
        ma_weights=base["ma_weights"],
        k_invalid=base["k_invalid"],
        validuse=base["validuse"],
        Omega=base["Omega"],
        pert_b_ma=pert_b_ma,
        pert_b_bic=pert_b_bic,
    )


def make_corr(p: int, rho_yx=None, rho_xx=None) -> np.ndarray:
    R = np.eye(p + 1)
    rho_yx_arr = np.zeros(p) if rho_yx is None else np.broadcast_to(np.asarray(rho_yx, dtype=float), (p,))
    R[0, 1:] = rho_yx_arr
    R[1:, 0] = rho_yx_arr
    if rho_xx is not None:
        R[1:, 1:] = np.asarray(rho_xx, dtype=float)
    return R


def adjusted_variances(betaGX, sdGX, betaGYC, sdGYC, fit, R, iv_snp_idx, ld=None, legacy_sigma_gzh: bool = False):
    b, Covb, Omega = fit["b"], fit["Covb"], fit["Omega"]
    S, p = betaGX.shape
    b1 = np.einsum("sj,jk,sk->s", betaGX, Covb, betaGX)
    c = (b ** 2 * sdGX ** 2).sum(1)
    d1 = (np.diag(Covb) * sdGX ** 2).sum(1)
    var_simple = sdGYC ** 2 + b1 + c + d1
    if ld is None:
        return var_simple, None

    valid = fit["validuse"]
    nv = len(valid)
    ivrow = np.asarray(iv_snp_idx)[valid]
    ldv = ld[:, valid]
    sdZY = sdGYC[ivrow]
    sdZX = sdGX[ivrow]
    D_iv = np.concatenate([sdZY[:, None], sdZX], axis=1)
    SIG_iv = D_iv[:, :, None] * R[None] * D_iv[:, None, :]
    sZ = np.zeros((nv * (p + 1), nv * (p + 1)))
    for j in range(nv):
        sZ[j * (p + 1) : (j + 1) * (p + 1), j * (p + 1) : (j + 1) * (p + 1)] = SIG_iv[j]
    M = Omega @ sZ @ Omega.T
    var_ld = np.empty(S)
    v1 = np.concatenate([[1.0], -b])
    Dg = np.concatenate([sdGYC[:, None], sdGX], axis=1)
    has_ld = np.any(ldv != 0, axis=1)
    u = Dg * v1[None]
    sub = ~has_ld
    var_ld[sub] = np.einsum("si,ij,sj->s", u[sub], R, u[sub]) + np.einsum("sj,jk,sk->s", betaGX[sub], M, betaGX[sub])
    for s in np.flatnonzero(has_ld):
        d = Dg[s]
        sG = d[:, None] * R * d[None, :]
        s12 = np.zeros((p + 1, nv, p + 1))
        s12[:, :, 0] = ldv[s][None, :] * d[:, None] * sdZY[None, :] * R[0, :][:, None]
        for a in range(1, p + 1):
            s12[a, :, 1:] = ldv[s][:, None] * d[a] * sdZX * R[a, 1:][None, :]
        if not legacy_sigma_gzh:
            s12[0, :, 1:] = ldv[s][:, None] * d[0] * sdZX * R[0, 1:][None, :]
        s12 = s12.reshape(p + 1, nv * (p + 1))
        U1 = np.zeros((2 * p + 1, 2 * p + 1))
        U1[: p + 1, : p + 1] = sG
        cross = s12 @ Omega.T
        U1[: p + 1, p + 1 :] = cross
        U1[p + 1 :, : p + 1] = cross.T
        U1[p + 1 :, p + 1 :] = M
        ev = float(np.linalg.eigvalsh((U1 + U1.T) / 2.0).min())
        if ev < 0:
            U1 = U1 + (-ev) * 1.001 * np.eye(2 * p + 1)
        v = np.concatenate([v1, -betaGX[s]])
        var_ld[s] = v @ U1 @ v
    return var_simple, var_ld


def mvcml_adjust(
    betaGX: np.ndarray,
    sdGX: np.ndarray,
    betaGYC: np.ndarray,
    sdGYC: np.ndarray,
    N: float,
    iv_mask: np.ndarray | None = None,
    iv_p_thresh: float = 5e-7,
    pvalX: np.ndarray | None = None,
    rho_yx=None,
    rho_xx=None,
    ld: np.ndarray | None = None,
    threshold: float = 1e-6,
    maxit: int = 500,
    legacy_sigma_gzh: bool = False,
    dp: bool = False,
    num_pert: int = 100,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Estimate multivariable exposure effects via MVMR-cML(-MA/-DP) and adjust conditional GWAS `betaGYC`."""
    betaGX_a = np.atleast_2d(np.asarray(betaGX, dtype=float).T).T if np.ndim(betaGX) == 1 else np.asarray(betaGX, dtype=float)
    sdGX_a = np.asarray(sdGX, dtype=float).reshape(betaGX_a.shape)
    betaGYC_a, sdGYC_a = np.asarray(betaGYC, dtype=float), np.asarray(sdGYC, dtype=float)
    S, p = betaGX_a.shape
    pvalX_a = _p_from_beta_se(betaGX_a, sdGX_a) if pvalX is None else np.asarray(pvalX, dtype=float)
    sel = (pvalX_a < iv_p_thresh).reshape(S, p).any(axis=1)
    if iv_mask is not None:
        sel &= np.asarray(iv_mask, dtype=bool)
    iv_idx = np.flatnonzero(sel)
    R = make_corr(p, rho_yx, rho_xx)
    D = np.concatenate([sdGYC_a[iv_idx, None], sdGX_a[iv_idx]], axis=1)
    SIG = D[:, :, None] * R[None] * D[:, None, :]
    if dp:
        fit = mvcml_dp(N, betaGX_a[iv_idx], betaGYC_a[iv_idx], SIG, num_pert=num_pert, threshold=threshold, maxit=maxit, seed=seed, verbose=verbose)
    else:
        fit = cml_ma(N, betaGX_a[iv_idx], betaGYC_a[iv_idx], SIG, threshold=threshold, maxit=maxit, verbose=verbose)
    covb = fit["Covb"]
    ev = float(np.linalg.eigvalsh((covb + covb.T) / 2.0).min())
    if ev < 0:
        covb = covb + (-1.001 * ev) * np.eye(p)
        fit["Covb"] = covb
    b = fit["b"]
    ybeta_adj = betaGYC_a - betaGX_a @ b
    ld_iv = None
    if ld is not None:
        ld_arr = np.asarray(ld, dtype=float)
        ld_iv = ld_arr[:, iv_idx] if ld_arr.shape == (S, S) else ld_arr
    v_simple, v_ld = adjusted_variances(
        betaGX_a, sdGX_a, betaGYC_a, sdGYC_a, fit, R, iv_idx, ld_iv, legacy_sigma_gzh
    )
    return dict(
        b=b,
        covb=covb,
        b_se=np.sqrt(np.diag(covb)),
        ybeta_adj=ybeta_adj,
        yse_adj_simple=np.sqrt(v_simple),
        yse_adj_ld=None if v_ld is None else np.sqrt(v_ld),
        iv_idx=iv_idx,
        valid_iv_idx=iv_idx[fit["validuse"]],
        fit=fit,
    )
