"""Python implementation of Multivariable Constrained Maximum Likelihood (`MVMR-cML`) and
conditional GWAS bias adjustment (Lin et al., AJHG 2023).
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


def cml_ma(
    N: float,
    betax: np.ndarray,
    betayc: np.ndarray,
    SIG: np.ndarray,
    threshold: float = 0.01,
    maxit: int = 500,
    verbose: bool = False,
) -> dict:
    """Fit MVMR-cML (`cml_MA`) across `k = 0 .. m - (p + 1)` invalid instruments selected by BIC."""
    betax, betayc = np.asarray(betax, dtype=float), np.asarray(betayc, dtype=float)
    m, p = betax.shape
    if m < p + 1:
        raise ValueError(f"Need at least p + 1 = {p + 1} instruments, got {m}")
    SIGinv = np.linalg.inv(SIG)
    A11 = SIGinv[:, 0, 0]
    A12 = SIGinv[:, 0, 1:]
    A22 = SIGinv[:, 1:, 1:]
    nk = m - (p + 1) + 1
    btotal = np.zeros((p, nk))
    betayesttotal = np.zeros((m, nk))
    betaxesttotal = []
    bic = np.zeros(nk)
    valid = []
    for k in range(nk):
        b = np.zeros(p)
        betayest = np.zeros(m)
        betaxest = np.zeros((m, p))
        c = 1
        while True:
            c += 1
            old = b.copy()
            betayest = betayc - betaxest @ b + (1.0 / A11) * np.einsum("ij,ij->i", betax - betaxest, A12)
            go, _ = _lik_terms(betaxest, betayc, SIGinv, betayest, b, betax)
            order = np.argsort(-go, kind="stable")
            keep = np.zeros(m, bool)
            keep[order[:k]] = True
            betayest = np.where(keep, betayest, 0.0)

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
            b = np.linalg.solve(a, rb)
            if np.abs(old - b).sum() < threshold or c > maxit:
                break
        btotal[:, k] = b
        betayesttotal[:, k] = betayest
        betaxesttotal.append(betaxest)
        _, like = _lik_terms(betaxest, betayc, SIGinv, betayest, b, betax)
        bic[k] = like + np.log(N) * k
        valid.append(np.flatnonzero(betayest == 0))
        if verbose:
            print(f"[mvcml] k={k} iterations={c} BIC={bic[k]:.4f}")

    kbest = int(np.argmin(bic))
    b = btotal[:, kbest]
    validuse = valid[kbest]
    betaxest = betaxesttotal[kbest]
    nv = len(validuse)
    siginv = SIGinv[validuse]
    betaxvalid = betaxest[validuse]
    betaxIVvalid = betax[validuse]
    betaGYCvalid = betayc[validuse]

    B = np.zeros((nv, p, p))
    C = np.zeros((nv, p, p))
    for i in range(nv):
        a11 = siginv[i, 0, 0]
        A12v = siginv[i, 0, 1:]
        A22v = siginv[i, 1:, 1:]
        bx = betaxvalid[i]
        for j in range(p):
            for kk in range(p):
                if j == kk:
                    B[i, j, kk] = (b[j] * bx[j] + b @ bx - betaGYCvalid[i]) * a11 - A12v @ (betaxIVvalid[i] - bx) + A12v[j] * bx[j]
                    C[i, j, kk] = b[j] ** 2 * a11 + 2.0 * b[j] * A12v[j] + A22v[j, j]
                else:
                    B[i, j, kk] = b[j] * bx[kk] * a11 + A12v[j] * bx[kk]
                    C[i, j, kk] = b[j] * b[kk] * a11 + b[j] * A12v[kk] + b[kk] * A12v[j] + A22v[j, kk]
    B1 = B.reshape(nv * p, p)
    C1 = np.zeros((nv * p, nv * p))
    for i in range(nv):
        C1[i * p : (i + 1) * p, i * p : (i + 1) * p] = C[i]
    A = np.einsum("i,ij,ik->jk", siginv[:, 0, 0], betaxvalid, betaxvalid)
    Info3 = np.block([[C1, B1], [B1.T, A]])
    Info = np.linalg.inv(Info3)
    Info_s = (Info + Info.T) / 2.0
    mineig = float(np.linalg.eigvalsh(Info_s).min())
    if mineig < 0:
        Info = Info + (-1.001) * mineig * np.eye(Info.shape[0])
    Covb = Info[-p:, -p:]
    W = np.column_stack([b, np.eye(p)])
    Omega1 = np.zeros((nv * p, nv * (p + 1)))
    Omega2 = np.zeros((p, nv * (p + 1)))
    for i in range(nv):
        Omega1[i * p : (i + 1) * p, i * (p + 1) : (i + 1) * (p + 1)] = W @ siginv[i]
        V = np.column_stack([betaxIVvalid[i], np.zeros((p, p))])
        Omega2[:, i * (p + 1) : (i + 1) * (p + 1)] = V @ siginv[i]
    Omega3 = np.vstack([Omega1, Omega2])
    Omega = -Info[-p:, :] @ Omega3
    return dict(
        b=b,
        Covb=Covb,
        betaxvalid=betaxvalid,
        betaxIVvalid=betaxIVvalid,
        validuse=validuse,
        Omega=Omega,
        bic=bic,
        k_invalid=kbest,
    )


def make_corr(p: int, rho_yx=None, rho_xx=None) -> np.ndarray:
    R = np.eye(p + 1)
    rho_yx_arr = np.zeros(p) if rho_yx is None else np.broadcast_to(np.asarray(rho_yx, dtype=float), (p,))
    R[0, 1:] = rho_yx_arr
    R[1:, 0] = rho_yx_arr
    if rho_xx is not None:
        R[1:, 1:] = np.asarray(rho_xx, dtype=float)
    return R


def adjusted_variances(betaGX, sdGX, betaGYC, sdGYC, fit, R, iv_snp_idx, ld=None, legacy_sigma_gzh: bool = True):
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
    threshold: float = 0.001,
    maxit: int = 500,
    legacy_sigma_gzh: bool = True,
    verbose: bool = False,
) -> dict:
    """Estimate multivariable exposure effects via MVMR-cML and adjust conditional GWAS `betaGYC`."""
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
    fit = cml_ma(N, betaGX_a[iv_idx], betaGYC_a[iv_idx], SIG, threshold=threshold, maxit=maxit, verbose=verbose)
    covb = fit["Covb"]
    ev = float(np.linalg.eigvalsh((covb + covb.T) / 2.0).min())
    if ev < 0:
        covb = covb + (-1.001 * ev) * np.eye(p)
        fit["Covb"] = covb
    b = fit["b"]
    ybeta_adj = betaGYC_a - betaGX_a @ b
    v_simple, v_ld = adjusted_variances(
        betaGX_a, sdGX_a, betaGYC_a, sdGYC_a, fit, R, iv_idx, None if ld is None else np.asarray(ld, dtype=float), legacy_sigma_gzh
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
