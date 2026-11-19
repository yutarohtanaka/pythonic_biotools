"""
Vectorized Multivariate Common-Factor SNP GWAS (`commonfactor_gwas`) and
`Q_SNP` ($\\chi^2_{k-1}$) pathway heterogeneity testing for `genomicsem`
(Grotzinger et al., Nature Human Behaviour 2019).
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .ldsc import (
    GenomicCovStruct,
    SettingsMode,
    ldsc_covstruct,
    pinv_eig_floor,
    stabilize_psd,
    validate_settings,
    vech,
)
from .model import UserModelResult, commonfactor, usermodel


def _extract_snp_matrices(
    snp_df_list: Sequence[pd.DataFrame] | pd.DataFrame,
    k: int,
    trait_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Extract aligned `(snp_ids, gamma_obs, se_obs)` arrays of shape `(M, k)` on the
    standardized-genotype genetic covariance scale.

    Supports:
    - List of $k$ per-trait DataFrames with `BETA`/`SE` (optional `MAF`/`EAF` for allele
      standardization $\\sqrt{2p(1-p)}$) or `Z`/`N`.
    - Single combined DataFrame with columns `BETA_<trait>`, `SE_<trait>` (or `Z_<trait>`, `N_<trait>`).
    """
    if isinstance(snp_df_list, pd.DataFrame):
        df = snp_df_list
        cl = {str(c).lower(): c for c in df.columns}
        snp_col = cl.get("snp", cl.get("rsid", cl.get("id")))
        snp_ids = (
            df[snp_col].astype(str).tolist()
            if snp_col is not None
            else [f"rs{i + 1}" for i in range(len(df))]
        )
        scale_maf = np.ones(len(df), dtype=float)
        maf_col = cl.get("maf", cl.get("eaf"))
        if maf_col is not None:
            p = np.clip(df[maf_col].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
            scale_maf = np.sqrt(2.0 * p * (1.0 - p))

        gamma_cols: list[np.ndarray] = []
        se_cols: list[np.ndarray] = []
        for t in trait_names:
            tl = t.lower()
            if f"beta_{tl}" in cl and f"se_{tl}" in cl:
                b = df[cl[f"beta_{tl}"]].to_numpy(dtype=float) * scale_maf
                s = df[cl[f"se_{tl}"]].to_numpy(dtype=float) * scale_maf
            elif f"z_{tl}" in cl and f"n_{tl}" in cl:
                n_arr = np.maximum(df[cl[f"n_{tl}"]].to_numpy(dtype=float), 1.0)
                s = 1.0 / np.sqrt(n_arr)
                b = df[cl[f"z_{tl}"]].to_numpy(dtype=float) * s
            else:
                raise ValueError(
                    f"Combined SNP DataFrame missing BETA_{t}/SE_{t} or Z_{t}/N_{t} columns."
                )
            gamma_cols.append(b)
            se_cols.append(np.maximum(s, 1e-12))
        return snp_ids, np.column_stack(gamma_cols), np.column_stack(se_cols)

    dfs = list(snp_df_list)
    if len(dfs) != k:
        raise ValueError(f"snp_df_list length ({len(dfs)}) != number of traits ({k}).")

    # Align on SNP column if present across all DataFrames
    first_cl = {str(c).lower(): c for c in dfs[0].columns}
    snp_key = first_cl.get("snp", first_cl.get("rsid", first_cl.get("id")))
    if snp_key is not None:
        common_snps = dfs[0][snp_key].astype(str).tolist()
        aligned_dfs: list[pd.DataFrame] = []
        for idx_d, d in enumerate(dfs):
            d_cl = {str(c).lower(): c for c in d.columns}
            d_key = d_cl.get("snp", d_cl.get("rsid", d_cl.get("id")))
            if d_key is None:
                raise ValueError(f"SNP column missing in snp_df_list[{idx_d}].")
            d_indexed = d.assign(_snp_id=d[d_key].astype(str)).set_index("_snp_id")
            aligned_dfs.append(d_indexed.loc[common_snps])
        snp_ids = common_snps
    else:
        lengths = {len(d) for d in dfs}
        if len(lengths) != 1:
            raise ValueError("All tables in snp_df_list must have identical length when 'SNP' is omitted.")
        aligned_dfs = dfs
        snp_ids = [f"rs{i + 1}" for i in range(len(dfs[0]))]

    gamma_list: list[np.ndarray] = []
    se_list: list[np.ndarray] = []
    for idx_d, d in enumerate(aligned_dfs):
        d_cl = {str(c).lower(): c for c in d.columns}
        maf_c = d_cl.get("maf", d_cl.get("eaf"))
        if maf_c is not None:
            p = np.clip(d[maf_c].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
            sd_g = np.sqrt(2.0 * p * (1.0 - p))
        else:
            sd_g = np.ones(len(d), dtype=float)

        if "beta" in d_cl and "se" in d_cl:
            b = d[d_cl["beta"]].to_numpy(dtype=float) * sd_g
            s = d[d_cl["se"]].to_numpy(dtype=float) * sd_g
        elif "z" in d_cl and "n" in d_cl:
            n_arr = np.maximum(d[d_cl["n"]].to_numpy(dtype=float), 1.0)
            s = 1.0 / np.sqrt(n_arr)
            b = d[d_cl["z"]].to_numpy(dtype=float) * s
        else:
            raise ValueError(
                f"snp_df_list[{idx_d}] must contain ('BETA', 'SE') or ('Z', 'N')."
            )
        gamma_list.append(b)
        se_list.append(np.maximum(s, 1e-12))

    return snp_ids, np.column_stack(gamma_list), np.column_stack(se_list)


def _solve_vectorized_cf_gwas(
    gamma_mat: np.ndarray,
    se_mat: np.ndarray,
    V_snp: np.ndarray,
    S: np.ndarray,
    V_S: np.ndarray,
    lam_base: np.ndarray,
    u_base: np.ndarray,
    J_lam_base: np.ndarray,
    J_u_base: np.ndarray,
    cov_lam: np.ndarray,
    settings: SettingsMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized DWLS Common-Factor SNP solver across all $M$ SNPs simultaneously.

    Returns `(beta_F, se_F, Q_snp, Q_pval)`.
    """
    M, k = gamma_mat.shape
    p_star = V_S.shape[0]
    s_vec = vech(S)

    w_snp = 1.0 / np.maximum(se_mat * se_mat, 1e-18)  # (M, k)
    w_s = 1.0 / np.maximum(np.diag(V_S), 1e-12)  # (p*,)

    # Closed-form initial DWLS beta_F at trait-only loadings lam_base
    denom0 = np.sum(w_snp * (lam_base[np.newaxis, :] ** 2), axis=1)  # (M,)
    num0 = np.sum(w_snp * lam_base[np.newaxis, :] * gamma_mat, axis=1)  # (M,)
    beta_f = num0 / np.maximum(denom0, 1e-18)

    lam_j = np.broadcast_to(lam_base[np.newaxis, :], (M, k)).copy()
    u_j = np.broadcast_to(u_base[np.newaxis, :], (M, k)).copy()

    # 2 vectorized Gauss-Newton updates of full (1 + 2k) parameter vector (beta_F, lam, u)
    from .ldsc import vech_indices

    r_idx, c_idx = vech_indices(k)
    q_full = 1 + 2 * k

    for _ in range(2):
        # Implied trait covariance vech(lam_j lam_j^T + diag(u_j)) across M SNPs
        sig_mat = np.einsum("mp,mq->mpq", lam_j, lam_j)
        for d_i in range(k):
            sig_mat[:, d_i, d_i] += u_j[:, d_i]
        sig_vech = sig_mat[:, r_idx, c_idx]  # (M, p*)

        r_snp = gamma_mat - beta_f[:, np.newaxis] * lam_j  # (M, k)
        r_trait = s_vec[np.newaxis, :] - sig_vech  # (M, p*)

        # Build batched Jacobian J_full of shape (M, k + p*, 1 + 2k)
        J_full = np.zeros((M, k + p_star, q_full), dtype=float)
        J_full[:, :k, 0] = lam_j
        for p_i in range(k):
            J_full[:, p_i, 1 + p_i] = beta_f

        # Trait block Jacobian rows for each SNP
        for pos, (ri, ci) in enumerate(zip(r_idx.tolist(), c_idx.tolist())):
            if ri == ci:
                J_full[:, k + pos, 1 + ri] = 2.0 * lam_j[:, ri]
                J_full[:, k + pos, 1 + k + ri] = 1.0
            else:
                J_full[:, k + pos, 1 + ri] = lam_j[:, ci]
                J_full[:, k + pos, 1 + ci] = lam_j[:, ri]

        w_full = np.concatenate([w_snp, np.broadcast_to(w_s[np.newaxis, :], (M, p_star))], axis=1)
        r_full = np.concatenate([r_snp, r_trait], axis=1)

        JW = J_full * w_full[:, :, np.newaxis]  # J * diag(w) works on rows: (J^T W)^T
        H_gn = np.einsum("mna,mnb->mab", JW, J_full) + 1e-10 * np.eye(q_full)[np.newaxis, :, :]
        g_gn = np.einsum("mna,mn->ma", JW, r_full)
        delta = np.linalg.solve(H_gn, g_gn[:, :, np.newaxis])[:, :, 0]

        beta_f = beta_f + delta[:, 0]
        lam_j = lam_j + delta[:, 1 : 1 + k]
        u_j = u_j + delta[:, 1 + k :]
        if settings == "default":
            u_j = np.maximum(u_j, 1e-6)

    # Final batched Jacobian & full block-diagonal sandwich SE for beta_F
    J_full = np.zeros((M, k + p_star, q_full), dtype=float)
    J_full[:, :k, 0] = lam_j
    for p_i in range(k):
        J_full[:, p_i, 1 + p_i] = beta_f
    for pos, (ri, ci) in enumerate(zip(r_idx.tolist(), c_idx.tolist())):
        if ri == ci:
            J_full[:, k + pos, 1 + ri] = 2.0 * lam_j[:, ri]
            J_full[:, k + pos, 1 + k + ri] = 1.0
        else:
            J_full[:, k + pos, 1 + ri] = lam_j[:, ci]
            J_full[:, k + pos, 1 + ci] = lam_j[:, ri]

    w_full = np.concatenate([w_snp, np.broadcast_to(w_s[np.newaxis, :], (M, p_star))], axis=1)
    JW = J_full * w_full[:, :, np.newaxis]  # shape (M, k + p*, q_full)
    bread_inv = np.einsum("mna,mnb->mab", JW, J_full) + 1e-11 * np.eye(q_full)[np.newaxis, :, :]

    # Meat = J_snp^T W_snp V_snp W_snp J_snp + J_trait^T W_s V_S W_s J_trait (block-diagonal V_full!)
    JW_snp = JW[:, :k, :]  # (M, k, q_full)
    JW_trait = JW[:, k:, :]  # (M, p*, q_full)
    meat_snp = np.einsum("mia,mij,mjb->mab", JW_snp, V_snp, JW_snp)
    meat_trait = np.einsum("mia,ij,mjb->mab", JW_trait, V_S, JW_trait)
    meat_total = meat_snp + meat_trait

    # First column of bread = bread_inv^{-1} e_0 -> var(beta_F) = e_0^T bread meat bread e_0
    e0 = np.zeros((M, q_full, 1), dtype=float)
    e0[:, 0, 0] = 1.0
    col0 = np.linalg.solve(bread_inv, e0)[:, :, 0]  # (M, q_full)
    var_beta = np.einsum("ma,mab,mb->m", col0, meat_total, col0)
    se_f = np.sqrt(np.maximum(var_beta, 1e-24))

    # Vectorized Q_SNP heterogeneity test (df = k - 1)
    if k <= 1:
        return beta_f, se_f, np.zeros(M, dtype=float), np.ones(M, dtype=float)

    r_snp_final = gamma_mat - beta_f[:, np.newaxis] * lam_j  # (M, k)
    # Orthonormal basis U of shape (M, k, k - 1) orthogonal to lam_j via batched SVD
    _, _, vh = np.linalg.svd(lam_j[:, np.newaxis, :], full_matrices=True)
    U = np.swapaxes(vh[:, 1:, :], 1, 2)  # (M, k, k - 1)

    u_r = np.einsum("mka,mk->ma", U, r_snp_final)  # (M, k - 1)
    cov_r_full = V_snp + (beta_f * beta_f)[:, np.newaxis, np.newaxis] * cov_lam[np.newaxis, :, :]
    omega_r = np.einsum("mka,mkl,mlb->mab", U, cov_r_full, U)  # (M, k - 1, k - 1)
    omega_r = 0.5 * (omega_r + np.swapaxes(omega_r, 1, 2))

    # Invert omega_r with spectral floor matching settings
    eig_val, eig_vec = np.linalg.eigh(omega_r)
    floor_q = 1e-7 if settings == "R_default" else 1e-12
    inv_eig = np.where(eig_val >= floor_q, 1.0 / np.maximum(eig_val, floor_q), 0.0)
    omega_r_inv = np.einsum("mai,mi,mbi->mab", eig_vec, inv_eig, eig_vec)

    Q_snp = np.maximum(np.einsum("ma,mab,mb->m", u_r, omega_r_inv, u_r), 0.0)
    Q_pval = chi2.sf(Q_snp, df=k - 1)
    return beta_f, se_f, Q_snp, Q_pval


def commonfactor_gwas(
    snp_df_list: Sequence[pd.DataFrame] | pd.DataFrame,
    covstruct: GenomicCovStruct,
    smooth_check: bool = True,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Run vectorized Multivariate Common-Factor SNP GWAS (`commonfactorGWAS`) estimating
    per-SNP latent factor effect $\\hat{\\beta}_{F,j}$, full sandwich SE, p-value, and
    $Q_{\\text{SNP}}$ ($\\chi^2_{k-1}$) common-pathway heterogeneity statistics.

    Parameters
    ----------
    snp_df_list : sequence of pd.DataFrame or pd.DataFrame
        Per-trait SNP GWAS summary tables containing `('BETA', 'SE')` (with optional
        `'MAF'`/`'EAF'`) or `('Z', 'N')`, or a combined wide DataFrame.
    covstruct : GenomicCovStruct
        Multivariate LDSC genetic covariance structure `(S, V_S, I_mat)`.
    smooth_check : bool, default True
        Check whether each SNP's augmented $(k+1) \\times (k+1)$ covariance matrix
        requires nearest-PSD smoothing, reporting `smoothed` and `Z_smooth`.
    settings : {"default", "R_default"}, default "default"
        `"default"` uses Heywood-safe residual variances and `1e-12` spectral stability;
        `"R_default"` allows unconstrained variances and uses `GenomicSEM` `1e-7`
        eigenvalue-floor pseudo-inverses.
    """
    validate_settings(settings)
    if not isinstance(covstruct, GenomicCovStruct):
        raise TypeError("covstruct must be a GenomicCovStruct.")
    assert covstruct.trait_names is not None
    assert covstruct.I_mat is not None

    k = covstruct.S.shape[0]
    if k < 2:
        raise ValueError(f"commonfactor_gwas requires k >= 2 traits; got k={k}.")

    # Step 1: Fit trait-only common factor model once
    cf_fit = commonfactor(covstruct, estimation="DWLS", settings=settings)
    rdf = cf_fit.results_df
    load_rows = rdf[rdf["op"] == "=~"]
    var_rows = rdf[(rdf["op"] == "~~") & (rdf["lhs"].isin(covstruct.trait_names))]

    trait_to_lam = dict(zip(load_rows["rhs"].astype(str), load_rows["est"].to_numpy(dtype=float)))
    trait_to_u = dict(zip(var_rows["lhs"].astype(str), var_rows["est"].to_numpy(dtype=float)))
    lam_base = np.array([trait_to_lam[t] for t in covstruct.trait_names], dtype=float)
    u_base = np.array([trait_to_u[t] for t in covstruct.trait_names], dtype=float)

    if k == 2:
        # 2-trait equal-loading model shares param_idx=0 across both loadings
        cov_lam = np.full((2, 2), float(cf_fit.V_theta[0, 0]), dtype=float)
        J_lam_base = np.zeros((3, 2), dtype=float)
        J_u_base = cf_fit.J[:, 1:3]
    else:
        cov_lam = cf_fit.V_theta[:k, :k]
        J_lam_base = cf_fit.J[:, :k]
        J_u_base = cf_fit.J[:, k : 2 * k]

    # Step 2: Extract aligned SNP effects and build per-SNP sampling covariance V_SNP
    snp_ids, gamma_obs, se_obs = _extract_snp_matrices(snp_df_list, k, covstruct.trait_names)
    M = len(snp_ids)
    I_mat = covstruct.I_mat
    V_snp = se_obs[:, :, np.newaxis] * I_mat[np.newaxis, :, :] * se_obs[:, np.newaxis, :]
    if settings == "default":
        eig_vs, vec_vs = np.linalg.eigh(0.5 * (V_snp + np.swapaxes(V_snp, 1, 2)))
        V_snp = np.einsum("mij,mj,mkj->mik", vec_vs, np.maximum(eig_vs, 1e-16), vec_vs)

    # Step 3: Optional PSD smoothing check on (k+1) x (k+1) SNP+traits covariance
    smoothed_flags = np.zeros(M, dtype=bool)
    gamma_use = gamma_obs.copy()

    if smooth_check:
        S_full = np.zeros((M, k + 1, k + 1), dtype=float)
        S_full[:, 0, 0] = 1.0
        S_full[:, 0, 1:] = gamma_obs
        S_full[:, 1:, 0] = gamma_obs
        S_full[:, 1:, 1:] = covstruct.S[np.newaxis, :, :]
        eig_f, vec_f = np.linalg.eigh(0.5 * (S_full + np.swapaxes(S_full, 1, 2)))
        smoothed_flags = eig_f[:, 0] < 1e-8
        if np.any(smoothed_flags):
            floor_sm = 1e-7 if settings == "R_default" else 1e-8
            eig_clamped = np.maximum(eig_f[smoothed_flags], floor_sm)
            S_sm = np.einsum(
                "mij,mj,mkj->mik",
                vec_f[smoothed_flags],
                eig_clamped,
                vec_f[smoothed_flags],
            )
            scale_g = np.sqrt(np.maximum(S_sm[:, 0, 0], 1e-12))[:, np.newaxis]
            gamma_use[smoothed_flags] = S_sm[:, 1:, 0] / scale_g

    VS_use = (
        stabilize_psd(covstruct.V_S, floor=1e-12)
        if settings == "default"
        else covstruct.V_S
    )
    beta_f, se_f, Q_snp, Q_pval = _solve_vectorized_cf_gwas(
        gamma_mat=gamma_use,
        se_mat=se_obs,
        V_snp=V_snp,
        S=covstruct.S,
        V_S=VS_use,
        lam_base=lam_base,
        u_base=u_base,
        J_lam_base=J_lam_base,
        J_u_base=J_u_base,
        cov_lam=cov_lam,
        settings=settings,
    )
    z_est = beta_f / se_f
    p_est = 2.0 * norm.sf(np.abs(z_est))

    z_smooth = np.zeros(M, dtype=float)
    if smooth_check and np.any(smoothed_flags):
        bf_raw, se_raw, _, _ = _solve_vectorized_cf_gwas(
            gamma_mat=gamma_obs[smoothed_flags],
            se_mat=se_obs[smoothed_flags],
            V_snp=V_snp[smoothed_flags],
            S=covstruct.S,
            V_S=VS_use,
            lam_base=lam_base,
            u_base=u_base,
            J_lam_base=J_lam_base,
            J_u_base=J_u_base,
            cov_lam=cov_lam,
            settings=settings,
        )
        z_smooth[smoothed_flags] = np.abs(z_est[smoothed_flags] - (bf_raw / se_raw))

    out_cols: dict[str, object] = {
        "SNP": snp_ids,
        "beta_F": beta_f,
        "se": se_f,
        "z": z_est,
        "p_value": p_est,
        "Q_SNP": Q_snp,
        "Q_df": np.full(M, k - 1, dtype=int),
        "Q_pval": Q_pval,
    }
    if smooth_check:
        out_cols["smoothed"] = smoothed_flags
        out_cols["Z_smooth"] = z_smooth

    return pd.DataFrame(out_cols)


class GenomicSEM:
    """
    Object-oriented GenomicSEM pipeline wrapping `ldsc_covstruct`, `usermodel`,
    `commonfactor`, and `commonfactor_gwas`.
    """

    def __init__(
        self,
        estimation: Literal["DWLS", "ML"] = "DWLS",
        std_lv: bool = True,
        n_blocks: int = 200,
        settings: Literal["default", "R_default"] = "default",
    ) -> None:
        validate_settings(settings)
        if estimation.upper() not in ("DWLS", "ML"):
            raise ValueError(f"Unsupported estimation={estimation!r}.")
        self.estimation: Literal["DWLS", "ML"] = estimation.upper()  # type: ignore[assignment]
        self.std_lv = std_lv
        self.n_blocks = n_blocks
        self.settings: SettingsMode = settings
        self.covstruct_: GenomicCovStruct | None = None
        self.model_result_: UserModelResult | None = None

    def ldsc(
        self,
        sumstats_list: Sequence[pd.DataFrame] | pd.DataFrame | np.ndarray | tuple[object, ...] | GenomicCovStruct,
        ld_scores: Sequence[float] | np.ndarray | pd.Series | pd.DataFrame | None = None,
        sample_sizes: Sequence[float] | Mapping[str, float] | None = None,
        m_snps: int | float | None = None,
        trait_names: Sequence[str] | None = None,
    ) -> GenomicCovStruct:
        self.covstruct_ = ldsc_covstruct(
            sumstats_list=sumstats_list,
            ld_scores=ld_scores,
            sample_sizes=sample_sizes,
            m_snps=m_snps,
            trait_names=trait_names,
            n_blocks=self.n_blocks,
            settings=self.settings,
        )
        return self.covstruct_

    def usermodel(
        self,
        model_str: str,
        covstruct: GenomicCovStruct | None = None,
    ) -> UserModelResult:
        target = covstruct if covstruct is not None else self.covstruct_
        if target is None:
            raise ValueError("Run ldsc() first or supply covstruct explicitly.")
        self.model_result_ = usermodel(
            covstruct_or_S=target,
            model_str=model_str,
            estimation=self.estimation,
            std_lv=self.std_lv,
            settings=self.settings,
        )
        return self.model_result_

    def commonfactor(
        self,
        covstruct: GenomicCovStruct | None = None,
    ) -> UserModelResult:
        target = covstruct if covstruct is not None else self.covstruct_
        if target is None:
            raise ValueError("Run ldsc() first or supply covstruct explicitly.")
        self.model_result_ = commonfactor(
            covstruct_or_S=target,
            estimation=self.estimation,
            settings=self.settings,
        )
        return self.model_result_

    def commonfactor_gwas(
        self,
        snp_df_list: Sequence[pd.DataFrame] | pd.DataFrame,
        covstruct: GenomicCovStruct | None = None,
        smooth_check: bool = True,
    ) -> pd.DataFrame:
        target = covstruct if covstruct is not None else self.covstruct_
        if target is None:
            raise ValueError("Run ldsc() first or supply covstruct explicitly.")
        return commonfactor_gwas(
            snp_df_list=snp_df_list,
            covstruct=target,
            smooth_check=smooth_check,
            settings=self.settings,
        )
