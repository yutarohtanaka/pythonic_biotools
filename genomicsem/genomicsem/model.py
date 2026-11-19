"""
Reticular Action Model (`RAM`) implied covariance, analytical Jacobian, `DWLS`/`ML`
estimators, sandwich standard errors, and model fit indices (`usermodel`,
`commonfactor`, `UserModelResult`) for `genomicsem`
(Grotzinger et al., Nature Human Behaviour 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from .ldsc import (
    GenomicCovStruct,
    SettingsMode,
    duplication_matrix,
    pinv_eig_floor,
    stabilize_psd,
    validate_settings,
    vech,
    vech_indices,
)
from .syntax import RAMSpec, parse_lavaan_syntax


@dataclass
class UserModelResult:
    """
    Fitted structural equation model output from `usermodel` / `commonfactor`.

    Attributes
    ----------
    results_df : pd.DataFrame
        Parameter table with unstandardized (`est`, `se`, `z`, `p_value`) and
        completely standardized genetic (`std_est`, `std_se`) estimates.
    modelfit : dict[str, float | int]
        Goodness-of-fit dictionary containing `chisq`, `df`, `p_chisq`, `AIC`,
        `CFI`, and `SRMR`.
    S_implied : np.ndarray
        Model-implied $k \\times k$ genetic covariance matrix $\\Sigma(\\hat{\\boldsymbol{\\theta}})$.
    S_observed : np.ndarray
        Observed $k \\times k$ LDSC genetic covariance matrix $S$.
    theta : np.ndarray
        Fitted free parameter vector $\\hat{\\boldsymbol{\\theta}}$.
    V_theta : np.ndarray
        Full $q \\times q$ sandwich covariance matrix $\\text{Cov}(\\hat{\\boldsymbol{\\theta}})$.
    J : np.ndarray
        Analytical $p^* \\times q$ Jacobian $\\partial \\text{vech}(\\Sigma) / \\partial \\boldsymbol{\\theta}$.
    ram_spec : RAMSpec
        Compiled RAM specification.
    converged : bool
        Whether numerical optimization converged cleanly.
    estimation : str
        Estimation method (`'DWLS'` or `'ML'`).
    settings : {"default", "R_default"}
        Active settings mode.
    """

    results_df: pd.DataFrame
    modelfit: dict[str, float | int]
    S_implied: np.ndarray
    S_observed: np.ndarray
    theta: np.ndarray
    V_theta: np.ndarray
    J: np.ndarray
    ram_spec: RAMSpec
    converged: bool
    estimation: str
    settings: SettingsMode = "default"

    def summary(self) -> str:
        mf = self.modelfit
        return (
            f"UserModelResult(estimation={self.estimation!r}, converged={self.converged}, "
            f"chisq={mf['chisq']:.4f}, df={mf['df']}, p={mf['p_chisq']:.4g}, "
            f"CFI={mf['CFI']:.4f}, SRMR={mf['SRMR']:.4f}, settings={self.settings!r})"
        )


def ram_implied_covariance_and_jacobian(
    theta: np.ndarray,
    ram_spec: RAMSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RAM implied observed covariance $\\Sigma(\\boldsymbol{\\theta})$, full system
    covariance $M(\\boldsymbol{\\theta})$, and exact analytical Jacobian
    $J = \\partial \\text{vech}(\\Sigma) / \\partial \\boldsymbol{\\theta} \\in \\mathbb{R}^{p^* \\times q}$.
    """
    A, S_ram = ram_spec.build_matrices(theta)
    m = A.shape[0]
    k = len(ram_spec.observed_traits)
    I_m = np.eye(m, dtype=float)

    try:
        G = np.linalg.inv(I_m - A)
    except np.linalg.LinAlgError:
        G = np.linalg.pinv(I_m - A, rcond=1e-12)

    M_full = G @ S_ram @ G.T
    M_full = 0.5 * (M_full + M_full.T)
    Sigma = M_full[:k, :k].copy()

    H = G[:k, :]  # F @ G of shape (k, m)
    B = G @ S_ram @ H.T  # shape (m, k)

    r_idx, c_idx = vech_indices(k)
    p_star = len(r_idx)
    q = ram_spec.n_free
    J = np.zeros((p_star, q), dtype=float)

    for r in range(q):
        dSigma = np.zeros((k, k), dtype=float)
        for u, v in ram_spec.A_free_entries[r]:
            # dSigma / dA_{uv} = H[:, u] B[v, :] + (H[:, u] B[v, :])^T
            outer_uv = np.outer(H[:, u], B[v, :])
            dSigma += outer_uv + outer_uv.T
        for u, v in ram_spec.S_free_entries[r]:
            if u == v:
                dSigma += np.outer(H[:, u], H[:, u])
            else:
                outer_s = np.outer(H[:, u], H[:, v])
                dSigma += outer_s + outer_s.T
        J[:, r] = dSigma[r_idx, c_idx]

    return Sigma, M_full, J


def _canonicalize_latent_signs(theta: np.ndarray, ram_spec: RAMSpec) -> np.ndarray:
    """
    Ensure each latent variable's primary free loading is positive when sign-invariant.
    """
    theta_out = theta.copy()
    k = len(ram_spec.observed_traits)

    for lat_pos, _ in enumerate(ram_spec.latent_vars):
        lat_idx = k + lat_pos
        # Check that no nonzero fixed off-diagonal entries touch lat_idx
        has_fixed_off = False
        for u in range( len(ram_spec.all_vars) ):
            if u != lat_idx:
                if ram_spec.A_fixed[u, lat_idx] != 0.0 or ram_spec.A_fixed[lat_idx, u] != 0.0:
                    has_fixed_off = True
                    break
                if ram_spec.S_fixed[u, lat_idx] != 0.0:
                    has_fixed_off = True
                    break
        if has_fixed_off:
            continue

        # Find all free parameters touching lat_idx on exactly one side
        affected_params: list[int] = []
        primary_loading_param: int | None = None
        can_flip = True
        for r in range(ram_spec.n_free):
            touches_odd = []
            for u, v in ram_spec.A_free_entries[r]:
                touches_odd.append((u == lat_idx) ^ (v == lat_idx))
                if v == lat_idx and u < k and primary_loading_param is None:
                    primary_loading_param = r
            for u, v in ram_spec.S_free_entries[r]:
                touches_odd.append((u == lat_idx) ^ (v == lat_idx))
            if any(touches_odd):
                if not all(touches_odd):
                    can_flip = False
                    break
                affected_params.append(r)

        if can_flip and primary_loading_param is not None and theta_out[primary_loading_param] < 0.0:
            for r in affected_params:
                theta_out[r] = -theta_out[r]

    return theta_out


def _build_initial_theta(S: np.ndarray, ram_spec: RAMSpec) -> np.ndarray:
    """Construct well-conditioned starting values avoiding zero-loading saddle points."""
    q = ram_spec.n_free
    theta0 = np.zeros(q, dtype=float)
    k = len(ram_spec.observed_traits)
    diag_s = np.maximum(np.diag(S), 0.02)

    # First eigenvector of S oriented positively
    eigvals, eigvecs = np.linalg.eigh(0.5 * (S + S.T))
    v1 = eigvecs[:, -1]
    if float(np.sum(v1)) < 0.0:
        v1 = -v1

    for r in range(q):
        if ram_spec.A_free_entries[r]:
            u, v = ram_spec.A_free_entries[r][0]
            if u < k:
                # Loading or regression onto observed trait u
                sgn = 1.0 if v1[u] >= -0.05 else -1.0
                # Offset slightly by v so multi-factor models break symmetry cleanly
                scale_factor = max(0.35, 0.65 - 0.12 * max(v - k, 0))
                theta0[r] = sgn * scale_factor * np.sqrt(diag_s[u])
            else:
                theta0[r] = 0.35
        elif ram_spec.S_free_entries[r]:
            u, v = ram_spec.S_free_entries[r][0]
            if u == v:
                if u < k:
                    theta0[r] = 0.45 * diag_s[u]
                else:
                    theta0[r] = float(np.mean(diag_s)) * 0.5
            else:
                if u < k and v < k:
                    theta0[r] = 0.1 * np.sqrt(diag_s[u] * diag_s[v])
                else:
                    theta0[r] = 0.25

    return theta0


def _standardized_parameters(
    theta: np.ndarray,
    V_theta: np.ndarray,
    ram_spec: RAMSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute completely standardized genetic estimates (`STD_Genotype`) and delta-method SEs
    for all terms in `ram_spec.terms`.
    """
    n_terms = len(ram_spec.terms)
    var_to_idx = {v: i for i, v in enumerate(ram_spec.all_vars)}

    def _eval_std(th: np.ndarray) -> np.ndarray:
        A, S_ram = ram_spec.build_matrices(th)
        _, M_full, _ = ram_implied_covariance_and_jacobian(th, ram_spec)
        sd_all = np.sqrt(np.maximum(np.diag(M_full), 1e-12))
        out = np.zeros(n_terms, dtype=float)
        for idx_t, term in enumerate(ram_spec.terms):
            u_lhs = var_to_idx[term.lhs]
            u_rhs = var_to_idx[term.rhs]
            if term.op == "=~":
                # Indicator u_rhs <- factor u_lhs
                raw_val = A[u_rhs, u_lhs]
                out[idx_t] = raw_val * (sd_all[u_lhs] / sd_all[u_rhs])
            elif term.op == "~":
                # Outcome u_lhs <- predictor u_rhs
                raw_val = A[u_lhs, u_rhs]
                out[idx_t] = raw_val * (sd_all[u_rhs] / sd_all[u_lhs])
            else:
                raw_val = S_ram[u_lhs, u_rhs]
                out[idx_t] = raw_val / (sd_all[u_lhs] * sd_all[u_rhs])
        return out

    std_est = _eval_std(theta)
    q = ram_spec.n_free
    if q == 0:
        return std_est, np.zeros(n_terms, dtype=float)

    eps = 1e-6
    J_std = np.zeros((n_terms, q), dtype=float)
    for r in range(q):
        step = eps * max(1.0, abs(float(theta[r])))
        t_plus = theta.copy()
        t_minus = theta.copy()
        t_plus[r] += step
        t_minus[r] -= step
        J_std[:, r] = (_eval_std(t_plus) - _eval_std(t_minus)) / (2.0 * step)

    cov_std = J_std @ V_theta @ J_std.T
    std_se = np.sqrt(np.maximum(np.diag(cov_std), 0.0))
    for idx_t, term in enumerate(ram_spec.terms):
        if not term.free and term.fixed_val == 0.0:
            std_se[idx_t] = 0.0
    return std_est, std_se


def _coerce_covstruct(
    covstruct_or_S: GenomicCovStruct | np.ndarray | pd.DataFrame | tuple[object, ...],
    V_S: np.ndarray | pd.DataFrame | None,
    settings: SettingsMode,
) -> GenomicCovStruct:
    """Normalize input into a `GenomicCovStruct`."""
    if isinstance(covstruct_or_S, GenomicCovStruct):
        return GenomicCovStruct(
            S=covstruct_or_S.S.copy(),
            V_S=covstruct_or_S.V_S.copy(),
            I_mat=None if covstruct_or_S.I_mat is None else covstruct_or_S.I_mat.copy(),
            trait_names=covstruct_or_S.trait_names,
            n_blocks=covstruct_or_S.n_blocks,
            settings=settings,
        )
    if isinstance(covstruct_or_S, tuple) and len(covstruct_or_S) in (2, 3):
        S_in = covstruct_or_S[0]
        VS_in = covstruct_or_S[1]
        I_in = covstruct_or_S[2] if len(covstruct_or_S) == 3 else None
        return GenomicCovStruct(S=S_in, V_S=VS_in, I_mat=I_in, settings=settings)  # type: ignore[arg-type]
    if V_S is None:
        raise ValueError(
            "V_S sampling covariance matrix is required when passing S matrix directly."
        )
    return GenomicCovStruct(S=covstruct_or_S, V_S=V_S, settings=settings)  # type: ignore[arg-type]


def _compute_browne_chisq(
    resid: np.ndarray,
    J: np.ndarray,
    V_S: np.ndarray,
    df: int,
    settings: SettingsMode,
) -> float:
    """
    Browne (1984) / GenomicSEM residual $\\chi^2$ test statistic:
    $\\chi^2 = (U^\\top \\mathbf{r})^\\top (U^\\top V_S U)^+ (U^\\top \\mathbf{r})$
    where $U \\in \\mathbb{R}^{p^* \\times \\text{df}}$ spans $\\text{null}(J^\\top)$.
    """
    if df <= 0:
        return 0.0
    U = null_space(J.T, rcond=1e-10)
    if U.shape[1] == 0:
        return 0.0
    if U.shape[1] > df:
        U = U[:, :df]
    u_r = U.T @ resid
    omega_u = U.T @ V_S @ U
    if settings == "R_default":
        omega_u_inv = pinv_eig_floor(omega_u, floor=1e-7)
    else:
        omega_u_inv = np.linalg.pinv(stabilize_psd(omega_u, floor=1e-12), rcond=1e-10)
    val = float(u_r @ omega_u_inv @ u_r)
    return max(val, 0.0)


def usermodel(
    covstruct_or_S: GenomicCovStruct | np.ndarray | pd.DataFrame | tuple[object, ...],
    model_str: str,
    V_S: np.ndarray | pd.DataFrame | None = None,
    estimation: Literal["DWLS", "ML"] = "DWLS",
    std_lv: bool = True,
    settings: SettingsMode = "default",
) -> UserModelResult:
    """
    Fit an arbitrary `lavaan`-syntax Genomic Structural Equation Model via `DWLS` or `ML`
    with full sandwich standard errors and Satorra-Bentler / Browne fit indices.

    - `settings='default'`: Heywood-safe lower bound (`>= 1e-6`) on free residual
      variances (fixed `0*` variances remain exact `0`), PSD-stabilized $V_S$.
    - `settings='R_default'`: unconstrained RAM residual variances (allowing negative
      Heywood estimates matching `GenomicSEM` / `OpenMx`), `CFI` against exact diagonal
      genetic independence model `diag(diag(S))`, and `1e-7` eigenvalue floor pseudo-inverse.
    """
    validate_settings(settings)
    est_upper = estimation.upper()
    if est_upper not in ("DWLS", "ML"):
        raise ValueError(f"Unsupported estimation={estimation!r}. Expected 'DWLS' or 'ML'.")

    covstruct = _coerce_covstruct(covstruct_or_S, V_S=V_S, settings=settings)
    assert covstruct.trait_names is not None
    S = covstruct.S
    VS = covstruct.V_S
    k = S.shape[0]
    p_star = k * (k + 1) // 2
    s_vec = vech(S)

    ram_spec = parse_lavaan_syntax(
        model_str=model_str,
        observed_traits=covstruct.trait_names,
        std_lv=std_lv,
        settings=settings,
    )
    q = ram_spec.n_free
    df = p_star - q
    if df < 0:
        raise ValueError(
            f"Model is under-identified: {q} free parameters exceed {p_star} covariance moments."
        )

    # Parameter bounds: Heywood-safe >= 1e-6 on free variances in 'default'; unconstrained in 'R_default'
    bounds: list[tuple[float | None, float | None]] = []
    for r in range(q):
        if settings == "default" and bool(ram_spec.is_free_variance[r]):
            bounds.append((1e-6, None))
        else:
            bounds.append((None, None))

    w_dwls_diag = 1.0 / np.maximum(np.diag(VS), 1e-12)

    def _dwls_obj_and_grad(th: np.ndarray) -> tuple[float, np.ndarray]:
        Sigma, _, J_loc = ram_implied_covariance_and_jacobian(th, ram_spec)
        diff = vech(Sigma) - s_vec
        w_diff = w_dwls_diag * diff
        f_val = float(np.dot(diff, w_diff))
        grad = 2.0 * (J_loc.T @ w_diff)
        return f_val, grad

    theta_init = _build_initial_theta(S, ram_spec)
    if settings == "default" and q > 0:
        for r in range(q):
            if ram_spec.is_free_variance[r]:
                theta_init[r] = max(float(theta_init[r]), 1e-4)

    converged = True
    if q == 0:
        theta_hat = np.zeros(0, dtype=float)
    else:
        # Multi-start DWLS optimization for global convergence
        rng_opt = np.random.default_rng(201905)
        starts = [theta_init]
        for mult in (0.7, 1.3):
            jitter = theta_init * mult + rng_opt.normal(0.0, 0.03, size=q)
            if settings == "default":
                for r in range(q):
                    if ram_spec.is_free_variance[r]:
                        jitter[r] = max(float(jitter[r]), 1e-4)
            starts.append(jitter)

        best_res = None
        for st in starts:
            res_try = minimize(
                fun=_dwls_obj_and_grad,
                x0=st,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 600},
            )
            if best_res is None or res_try.fun < best_res.fun:
                best_res = res_try
        assert best_res is not None
        theta_hat = best_res.x.copy()
        converged = bool(best_res.success) or float(best_res.fun) < 1e-8

        if est_upper == "ML":
            S_pd = stabilize_psd(S, floor=1e-6)
            sign_s, logdet_s = np.linalg.slogdet(S_pd)
            const_s = float(logdet_s) if sign_s > 0 else 0.0
            D_k = duplication_matrix(k)

            def _ml_obj_and_grad(th: np.ndarray) -> tuple[float, np.ndarray]:
                Sigma, _, J_loc = ram_implied_covariance_and_jacobian(th, ram_spec)
                eig_sig = np.linalg.eigvalsh(Sigma)
                min_eig = float(np.min(eig_sig))
                if min_eig <= 1e-9:
                    pen = 1e6 + 1e8 * (1e-9 - min_eig) ** 2
                    # Nudge toward DWLS optimum
                    return pen, 2e4 * (th - theta_hat)
                Sig_inv = np.linalg.inv(Sigma)
                _, logdet_sig = np.linalg.slogdet(Sigma)
                f_ml = float(logdet_sig + np.trace(S_pd @ Sig_inv) - const_s - k)
                # dF_ML / dSigma = Sigma^{-1} (Sigma - S_pd) Sigma^{-1}
                G_mat = Sig_inv @ (Sigma - S_pd) @ Sig_inv
                # vec(G_mat)^T D_k J_loc
                grad_vech = D_k.T @ G_mat.ravel(order="F")
                return max(f_ml, 0.0), J_loc.T @ grad_vech

            res_ml = minimize(
                fun=_ml_obj_and_grad,
                x0=theta_hat,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 600},
            )
            theta_hat = res_ml.x.copy()
            converged = bool(res_ml.success) or converged

    theta_hat = _canonicalize_latent_signs(theta_hat, ram_spec)
    Sigma_hat, _, J_hat = ram_implied_covariance_and_jacobian(theta_hat, ram_spec)

    # Construct estimation weight matrix W (p* x p*)
    if est_upper == "DWLS":
        W = np.diag(w_dwls_diag)
    else:
        Sig_pd = stabilize_psd(Sigma_hat, floor=1e-6)
        Sig_inv = np.linalg.inv(Sig_pd)
        D_k = duplication_matrix(k)
        W = 0.5 * (D_k.T @ np.kron(Sig_inv, Sig_inv) @ D_k)

    # Full sandwich covariance: (J^T W J)^+ J^T W V_S W J (J^T W J)^+
    if q > 0:
        bread_inv = J_hat.T @ W @ J_hat
        if settings == "R_default":
            bread = pinv_eig_floor(bread_inv, floor=1e-10)
        else:
            bread = np.linalg.pinv(0.5 * (bread_inv + bread_inv.T), rcond=1e-10)
        meat = J_hat.T @ W @ VS @ W @ J_hat
        V_theta = bread @ meat @ bread
        V_theta = 0.5 * (V_theta + V_theta.T)
    else:
        V_theta = np.zeros((0, 0), dtype=float)

    se_theta = np.sqrt(np.maximum(np.diag(V_theta), 0.0)) if q > 0 else np.zeros(0, dtype=float)
    std_est, std_se = _standardized_parameters(theta_hat, V_theta, ram_spec)

    # Assemble parameter DataFrame
    rows: list[dict[str, object]] = []
    for idx_t, term in enumerate(ram_spec.terms):
        if term.free:
            est_val = float(theta_hat[term.param_idx])
            se_val = float(se_theta[term.param_idx])
            z_val = est_val / se_val if se_val > 1e-15 else 0.0
            p_val = float(2.0 * norm.sf(abs(z_val))) if se_val > 1e-15 else 1.0
        else:
            est_val = float(term.fixed_val)
            se_val = 0.0
            z_val = float("nan")
            p_val = float("nan")
        rows.append(
            {
                "lhs": term.lhs,
                "op": term.op,
                "rhs": term.rhs,
                "free": term.free,
                "label": term.label,
                "est": est_val,
                "se": se_val,
                "z": z_val,
                "p_value": p_val,
                "std_est": float(std_est[idx_t]),
                "std_se": float(std_se[idx_t]),
            }
        )
    results_df = pd.DataFrame(rows)

    # Model fit indices
    resid_vec = s_vec - vech(Sigma_hat)
    chisq_val = _compute_browne_chisq(resid_vec, J_hat, VS, df=df, settings=settings)
    p_chisq = float(chi2.sf(chisq_val, df=df)) if df > 0 else 1.0
    aic_val = float(chisq_val + 2.0 * q)

    # Independence (null) model for CFI: Sigma_null = diag(diag(S)) with k diagonal variances
    r_idx, c_idx = vech_indices(k)
    off_mask = r_idx != c_idx
    df_null = int(np.sum(off_mask))
    if df_null > 0:
        r_null = s_vec[off_mask]
        VS_off = VS[np.ix_(off_mask, off_mask)]
        if settings == "R_default":
            VS_off_inv = pinv_eig_floor(VS_off, floor=1e-7)
        else:
            VS_off_inv = np.linalg.pinv(stabilize_psd(VS_off, floor=1e-12), rcond=1e-10)
        chisq_null = max(float(r_null @ VS_off_inv @ r_null), 0.0)
        num_cfi = max(chisq_val - float(df), 0.0)
        den_cfi = max(chisq_null - float(df_null), num_cfi, 1e-12)
        cfi_val = float(np.clip(1.0 - num_cfi / den_cfi, 0.0, 1.0)) if den_cfi > 1e-11 else 1.0
    else:
        cfi_val = 1.0

    # SRMR across all p* unique covariance moments standardized by observed genetic SDs
    sd_obs = np.sqrt(np.maximum(np.diag(S), 1e-12))
    scale_vech = sd_obs[r_idx] * sd_obs[c_idx]
    srmr_val = float(np.sqrt(np.mean((resid_vec / scale_vech) ** 2)))

    modelfit: dict[str, float | int] = {
        "chisq": chisq_val,
        "df": int(df),
        "p_chisq": p_chisq,
        "AIC": aic_val,
        "CFI": cfi_val,
        "SRMR": srmr_val,
    }

    return UserModelResult(
        results_df=results_df,
        modelfit=modelfit,
        S_implied=Sigma_hat,
        S_observed=S.copy(),
        theta=theta_hat,
        V_theta=V_theta,
        J=J_hat,
        ram_spec=ram_spec,
        converged=converged,
        estimation=est_upper,
        settings=settings,
    )


def commonfactor(
    covstruct_or_S: GenomicCovStruct | np.ndarray | pd.DataFrame | tuple[object, ...],
    V_S: np.ndarray | pd.DataFrame | None = None,
    estimation: Literal["DWLS", "ML"] = "DWLS",
    settings: SettingsMode = "default",
) -> UserModelResult:
    """
    Fit a single common latent genetic factor model (`F1 =~ T1 + T2 + ... + Tk`)
    with unit factor variance (`std_lv=True`).
    """
    validate_settings(settings)
    covstruct = _coerce_covstruct(covstruct_or_S, V_S=V_S, settings=settings)
    assert covstruct.trait_names is not None
    names = covstruct.trait_names
    if len(names) == 2:
        # Equal loadings for 2-indicator common factor identification
        syntax = f"F1 =~ l_eq*{names[0]} + l_eq*{names[1]}"
    else:
        syntax = "F1 =~ " + " + ".join(names)
    return usermodel(
        covstruct_or_S=covstruct,
        model_str=syntax,
        estimation=estimation,
        std_lv=True,
        settings=settings,
    )
