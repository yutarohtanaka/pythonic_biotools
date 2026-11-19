"""
Univariate local heritability (`run_univ`), bivariate local genetic correlation
(`run_bivar`), partial local genetic correlation (`run_pcor`), and multivariate
local genetic multiple regression (`run_multireg`) for `lava`
(Werme et al., Nature Genetics 2022).
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist

from .locus import (
    LAVALocus,
    SettingsMode,
    process_locus,
    project_psd_matrix,
    validate_settings,
)
from .simulation import (
    bivariate_null_pvalue,
    omnibus_multireg_null_pvalue,
    simulate_wishart_omega,
)


def _safe_nan_quantiles(arr: np.ndarray, q_low: float = 2.5, q_high: float = 97.5) -> tuple[float, float]:
    """Compute quantiles without emitting RuntimeWarning on all-NaN arrays."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(valid, [q_low, q_high])
    return (float(lo), float(hi))


def _invert_cov_matrix(mat: np.ndarray, settings: SettingsMode) -> np.ndarray:
    """Invert symmetric genetic covariance block safely under both settings modes."""
    sym = 0.5 * (mat + mat.T)
    if settings == "default":
        eigvals, eigvecs = np.linalg.eigh(sym)
        if float(np.min(eigvals)) > 1e-10:
            return np.linalg.inv(sym)
        inv_eig = 1.0 / np.maximum(eigvals, 1e-8)
        return (eigvecs * inv_eig[np.newaxis, :]) @ eigvecs.T
    return np.linalg.pinv(sym, rcond=1e-8)


def run_univ(
    locus: LAVALocus,
    traits: Sequence[str] | None = None,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Test univariate local genetic signal ($\\omega_{pp} > 0$) and estimate local
    SNP heritability $h^2_{\\text{loc}, p}$ per trait.

    Under $H_0: \\delta_p = \\mathbf{0}$, the mean-square ratio
    $F_p = (\\frac{1}{K}\\hat{\\delta}_p^\\top \\hat{\\delta}_p) / \\Sigma_{pp}$ follows
    an exact central $F(K, N_p - K - 1)$ distribution (with asymptotic $K F_p \\sim \\chi^2_K$).
    """
    validate_settings(settings)
    if not isinstance(locus, LAVALocus):
        raise TypeError("locus must be a LAVALocus returned by process_locus.")

    idxs = locus.trait_indices(traits)
    K = locus.K
    rows: list[dict[str, object]] = []

    for idx in idxs:
        trait = locus.traits[idx]
        n_p = locus.sample_sizes[trait]
        ss_pc = float(np.sum(locus.delta_hat[:, idx] ** 2))
        sig_pp = float(locus.sigma[idx, idx])
        omega_pp_raw = float(locus.omega_raw[idx, idx])
        h2_loc = float(max(K * omega_pp_raw, 0.0))

        if sig_pp <= 0.0:
            f_stat = 0.0
            chisq_stat = 0.0
            pval = 1.0
        else:
            f_stat = (ss_pc / float(K)) / sig_pp
            chisq_stat = float(K) * f_stat
            df_denom = n_p - float(K) - 1.0
            if df_denom > 1.0:
                pval = float(f_dist.sf(f_stat, dfn=K, dfd=df_denom))
            else:
                pval = float(chi2.sf(chisq_stat, df=K))

        rows.append(
            {
                "locus": locus.locus_id,
                "trait": trait,
                "n_snps": locus.n_snps,
                "K": K,
                "omega": omega_pp_raw,
                "h2_loc": h2_loc,
                "f_stat": float(f_stat),
                "chisq_stat": float(chisq_stat),
                "p": pval,
            }
        )

    return pd.DataFrame(rows)


def run_bivar(
    locus: LAVALocus,
    traits: Sequence[str] | None = None,
    univ_thresh: float | None = None,
    n_sim: int = 10000,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Estimate pairwise bivariate local genetic correlations $r_{g,\\text{loc}}$,
    explained genetic variance $r^2$, simulation 95% CIs, and bivariate p-values.

    - `settings='default'`: `univ_thresh=1.0` (evaluates all pairs unless overridden),
      PD floor `1e-6` on non-PSD $\\hat{\\Omega}$, `n_sim=10000`.
    - `settings='R_default'`: `univ_thresh=0.05` univariate $h^2$ screening gate,
      spectral zero-clipping `max(eig, 0)`, and adaptive `1e4 -> 1e5` Monte Carlo
      refinement when $p < 10^{-3}$.
    """
    validate_settings(settings)
    if not isinstance(locus, LAVALocus):
        raise TypeError("locus must be a LAVALocus returned by process_locus.")
    if n_sim < 100:
        raise ValueError(f"n_sim must be >= 100; got {n_sim}.")

    eff_thresh = univ_thresh if univ_thresh is not None else (0.05 if settings == "R_default" else 1.0)
    if not (0.0 < eff_thresh <= 1.0):
        raise ValueError(f"univ_thresh must be in (0, 1]; got {eff_thresh}.")

    cand_traits = list(traits) if traits is not None else list(locus.traits)
    if eff_thresh < 1.0:
        univ_df = run_univ(locus, traits=cand_traits, settings=settings)
        passed = univ_df.loc[univ_df["p"] < eff_thresh, "trait"].tolist()
        cand_traits = [t for t in cand_traits if t in passed]

    cols = [
        "locus",
        "trait1",
        "trait2",
        "K",
        "omega_xy",
        "rg",
        "ci_low",
        "ci_high",
        "r2",
        "r2_ci_low",
        "r2_ci_high",
        "p",
    ]
    if len(cand_traits) < 2:
        return pd.DataFrame(columns=cols)

    omega_full = project_psd_matrix(locus.omega_raw, settings=settings, pd_floor=1e-6)
    idxs = locus.trait_indices(cand_traits)
    rng = np.random.default_rng(seed)
    K = locus.K

    rows: list[dict[str, object]] = []
    for a_pos in range(len(idxs)):
        for b_pos in range(a_pos + 1, len(idxs)):
            i = idxs[a_pos]
            j = idxs[b_pos]
            sub_idx = np.array([i, j], dtype=int)
            om_2 = omega_full[np.ix_(sub_idx, sub_idx)]
            sig_2 = locus.sigma[np.ix_(sub_idx, sub_idx)]

            w_xx = float(om_2[0, 0])
            w_yy = float(om_2[1, 1])
            w_xy = float(om_2[0, 1])
            denom = np.sqrt(max(w_xx, 0.0) * max(w_yy, 0.0))

            if denom <= 1e-18:
                rg_est = float("nan") if settings == "R_default" else 0.0
                r2_est = float("nan") if settings == "R_default" else 0.0
            else:
                rg_est = float(np.clip(w_xy / denom, -1.0, 1.0))
                r2_est = float(rg_est * rg_est)

            # Simulation 95% CIs from non-central Wishart around om_2
            sim_om = simulate_wishart_omega(K, sig_2, om_2, n_sim, rng, settings=settings)
            sim_denom = np.sqrt(
                np.maximum(sim_om[:, 0, 0], 0.0) * np.maximum(sim_om[:, 1, 1], 0.0)
            )
            rg_sim = np.full(n_sim, np.nan, dtype=float)
            np.divide(sim_om[:, 0, 1], sim_denom, out=rg_sim, where=sim_denom > 1e-18)
            rg_sim = np.clip(rg_sim, -1.0, 1.0)
            r2_sim = rg_sim * rg_sim

            ci_low, ci_high = _safe_nan_quantiles(rg_sim)
            r2_low, r2_high = _safe_nan_quantiles(r2_sim)

            pval = bivariate_null_pvalue(
                K=K,
                omega_xx=w_xx,
                omega_yy=w_yy,
                sigma_2x2=sig_2,
                omega_xy_obs=float(locus.omega_raw[i, j]),
                n_sim=n_sim,
                rng=rng,
                settings=settings,
            )

            rows.append(
                {
                    "locus": locus.locus_id,
                    "trait1": locus.traits[i],
                    "trait2": locus.traits[j],
                    "K": K,
                    "omega_xy": w_xy,
                    "rg": rg_est,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "r2": r2_est,
                    "r2_ci_low": r2_low,
                    "r2_ci_high": r2_high,
                    "p": pval,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def run_pcor(
    locus: LAVALocus,
    pair: Sequence[str],
    cond_traits: str | Sequence[str],
    n_sim: int = 10000,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Compute partial local genetic correlation $r_g(X, Y \\mid \\mathbf{Z})$ via the
    conditional Schur complement $\\Omega_{XY \\mid Z} = \\Omega_{XY} - \\Omega_{XZ}\\Omega_{ZZ}^{-1}\\Omega_{ZY}$,
    with simulation 95% CIs and p-values.
    """
    validate_settings(settings)
    if not isinstance(locus, LAVALocus):
        raise TypeError("locus must be a LAVALocus returned by process_locus.")
    if len(pair) != 2 or str(pair[0]) == str(pair[1]):
        raise ValueError(f"pair must contain 2 distinct traits; got {pair!r}.")
    if n_sim < 100:
        raise ValueError(f"n_sim must be >= 100; got {n_sim}.")

    cond_list = [cond_traits] if isinstance(cond_traits, str) else [str(c) for c in cond_traits]
    if not cond_list:
        raise ValueError("cond_traits must contain at least 1 conditioning trait.")
    t1, t2 = str(pair[0]), str(pair[1])
    if t1 in cond_list or t2 in cond_list:
        raise ValueError(f"Conditioning traits {cond_list} must be disjoint from pair {pair!r}.")

    all_names = [t1, t2] + cond_list
    idxs = np.array(locus.trait_indices(all_names), dtype=int)
    m = len(cond_list)
    K = locus.K

    omega_full = project_psd_matrix(locus.omega_raw, settings=settings, pd_floor=1e-6)
    om_sub = omega_full[np.ix_(idxs, idxs)]
    sig_sub = locus.sigma[np.ix_(idxs, idxs)]

    om_xy = om_sub[:2, :2]
    om_xz = om_sub[:2, 2:]
    om_zz = om_sub[2:, 2:]
    om_zz_inv = _invert_cov_matrix(om_zz, settings=settings)
    weights_z = om_xz @ om_zz_inv  # (2, m)
    om_cond = om_xy - weights_z @ om_xz.T
    om_cond = 0.5 * (om_cond + om_cond.T)

    w_xx_c = max(float(om_cond[0, 0]), 0.0)
    w_yy_c = max(float(om_cond[1, 1]), 0.0)
    w_xy_c = float(om_cond[0, 1])
    denom = np.sqrt(w_xx_c * w_yy_c)
    if denom <= 1e-18:
        pcor_est = float("nan") if settings == "R_default" else 0.0
    else:
        pcor_est = float(np.clip(w_xy_c / denom, -1.0, 1.0))

    # Full (2 + m)-dimensional non-central Wishart simulation for CIs
    rng = np.random.default_rng(seed)
    sim_om = simulate_wishart_omega(K, sig_sub, om_sub, n_sim, rng, settings=settings)
    sim_xy = sim_om[:, :2, :2]
    sim_xz = sim_om[:, :2, 2:]
    sim_zz = sim_om[:, 2:, 2:]

    # Vectorized Schur complement across n_sim draws
    reg_eye = (1e-8 if settings == "default" else 1e-10) * np.eye(m)[np.newaxis, :, :]
    sim_zz_reg = sim_zz + reg_eye
    # Solve sim_zz_reg @ X = sim_xz^T -> X = sim_zz_reg^{-1} sim_zx of shape (n_sim, m, 2)
    solved = np.linalg.solve(sim_zz_reg, np.swapaxes(sim_xz, 1, 2))
    sim_cond = sim_xy - np.einsum("aim,amj->aij", sim_xz, solved)

    sim_den = np.sqrt(
        np.maximum(sim_cond[:, 0, 0], 0.0) * np.maximum(sim_cond[:, 1, 1], 0.0)
    )
    pcor_sim = np.full(n_sim, np.nan, dtype=float)
    np.divide(sim_cond[:, 0, 1], sim_den, out=pcor_sim, where=sim_den > 1e-18)
    pcor_sim = np.clip(pcor_sim, -1.0, 1.0)
    ci_low, ci_high = _safe_nan_quantiles(pcor_sim)

    # Effective residualized sampling covariance B Sigma B^T for conditional p-value
    B_mat = np.hstack([np.eye(2, dtype=float), -weights_z])
    df_eff = max(K - m, 1)
    sig_cond = (float(K) / float(df_eff)) * (B_mat @ sig_sub @ B_mat.T)
    sig_cond = 0.5 * (sig_cond + sig_cond.T)

    pval = bivariate_null_pvalue(
        K=df_eff,
        omega_xx=w_xx_c,
        omega_yy=w_yy_c,
        sigma_2x2=sig_cond,
        omega_xy_obs=w_xy_c,
        n_sim=n_sim,
        rng=rng,
        settings=settings,
    )

    return pd.DataFrame(
        [
            {
                "locus": locus.locus_id,
                "trait1": t1,
                "trait2": t2,
                "cond_traits": ",".join(cond_list),
                "K": K,
                "omega_xy_given_z": w_xy_c,
                "pcor": pcor_est,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "r2_part": float(pcor_est * pcor_est) if np.isfinite(pcor_est) else float("nan"),
                "p": pval,
            }
        ]
    )


def _standardize_cov_to_cor(om: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert covariance matrix `om` to correlation matrix and return `(cor, sd)`."""
    sd = np.sqrt(np.maximum(np.diag(om), 0.0))
    denom = np.outer(sd, sd)
    cor = np.zeros_like(om, dtype=float)
    np.divide(om, denom, out=cor, where=denom > 1e-18)
    cor = np.clip(cor, -1.0, 1.0)
    np.fill_diagonal(cor, 1.0)
    return cor, sd


def run_multireg(
    locus: LAVALocus,
    predictors: str | Sequence[str],
    outcome: str,
    n_sim: int = 10000,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Fit multivariate local genetic multiple regression of `outcome` on `predictors`:

        $\\hat{\\boldsymbol{\\gamma}} = (\\Omega^{\\*}_{XX})^{-1} \\boldsymbol{\\omega}^{\\*}_{Xy}$,
        $R^2 = (\\boldsymbol{\\omega}^{\\*}_{Xy})^\\top (\\Omega^{\\*}_{XX})^{-1} \\boldsymbol{\\omega}^{\\*}_{Xy}$,

    with simulation 95% CIs, conditional per-predictor p-values, and joint omnibus `p_joint`.
    """
    validate_settings(settings)
    if not isinstance(locus, LAVALocus):
        raise TypeError("locus must be a LAVALocus returned by process_locus.")
    if n_sim < 100:
        raise ValueError(f"n_sim must be >= 100; got {n_sim}.")

    pred_list = [predictors] if isinstance(predictors, str) else [str(p) for p in predictors]
    if not pred_list:
        raise ValueError("predictors must contain at least 1 trait.")
    if len(set(pred_list)) != len(pred_list):
        raise ValueError(f"Duplicate traits in predictors: {pred_list}")
    out_str = str(outcome)
    if out_str in pred_list:
        raise ValueError(f"outcome {out_str!r} must not appear in predictors {pred_list}.")

    all_names = pred_list + [out_str]
    idxs = np.array(locus.trait_indices(all_names), dtype=int)
    m = len(pred_list)
    K = locus.K

    omega_full = project_psd_matrix(locus.omega_raw, settings=settings, pd_floor=1e-6)
    om_sub = omega_full[np.ix_(idxs, idxs)]
    sig_sub = locus.sigma[np.ix_(idxs, idxs)]

    # Raw unstandardized genetic slopes
    om_xx = om_sub[:m, :m]
    om_xy = om_sub[:m, m]
    om_yy = float(om_sub[m, m])
    om_xx_inv = _invert_cov_matrix(om_xx, settings=settings)
    beta_raw = om_xx_inv @ om_xy

    # Standardized genetic correlation scale
    cor_sub, _ = _standardize_cov_to_cor(om_sub)
    cor_xx = cor_sub[:m, :m]
    cor_xy = cor_sub[:m, m]
    cor_xx_inv = _invert_cov_matrix(cor_xx, settings=settings)
    gamma_hat = cor_xx_inv @ cor_xy
    r2_joint = float(np.clip(float(cor_xy @ gamma_hat), 0.0, 1.0))

    # Non-central Wishart simulation for 95% CIs on gamma and joint R^2
    rng = np.random.default_rng(seed)
    sim_om = simulate_wishart_omega(K, sig_sub, om_sub, n_sim, rng, settings=settings)

    sd_sim = np.sqrt(np.maximum(np.diagonal(sim_om, axis1=1, axis2=2), 0.0))  # (n_sim, m+1)
    denom_sim = sd_sim[:, :, np.newaxis] * sd_sim[:, np.newaxis, :]
    cor_sim = np.zeros_like(sim_om, dtype=float)
    np.divide(sim_om, denom_sim, out=cor_sim, where=denom_sim > 1e-18)
    cor_sim = np.clip(cor_sim, -1.0, 1.0)
    for d_i in range(m + 1):
        cor_sim[:, d_i, d_i] = 1.0

    cor_xx_sim = cor_sim[:, :m, :m] + 1e-6 * np.eye(m)[np.newaxis, :, :]
    cor_xy_sim = cor_sim[:, :m, m]
    gamma_sim = np.linalg.solve(cor_xx_sim, cor_xy_sim[:, :, np.newaxis])[:, :, 0]
    r2_sim = np.clip(np.einsum("si,si->s", cor_xy_sim, gamma_sim), 0.0, 1.0)
    r2_low, r2_high = _safe_nan_quantiles(r2_sim)

    # Conditional p-values per predictor j: H_0: gamma_j = 0 <=> pcor(X_j, y | X_{-j}) = 0
    cond_pvals: list[float] = []
    for j_idx, pred_name in enumerate(pred_list):
        if m == 1:
            pj = bivariate_null_pvalue(
                K=K,
                omega_xx=float(om_xx[0, 0]),
                omega_yy=om_yy,
                sigma_2x2=sig_sub,
                omega_xy_obs=float(locus.omega_raw[idxs[0], idxs[1]]),
                n_sim=n_sim,
                rng=rng,
                settings=settings,
            )
        else:
            other_preds = [p for k_idx, p in enumerate(pred_list) if k_idx != j_idx]
            pcor_j = run_pcor(
                locus=locus,
                pair=(pred_name, out_str),
                cond_traits=other_preds,
                n_sim=n_sim,
                seed=int(rng.integers(0, 2**31 - 1)),
                settings=settings,
            )
            pj = float(pcor_j.loc[0, "p"])
        cond_pvals.append(pj)

    # Joint omnibus p-value H_0: omega_{Xy} = 0
    if m == 1:
        p_joint = cond_pvals[0]
    else:
        raw_xy_vec = locus.omega_raw[idxs[:m], idxs[m]]
        p_joint = omnibus_multireg_null_pvalue(
            K=K,
            omega_xx_mat=om_xx,
            omega_yy=om_yy,
            sigma_sub=sig_sub,
            omega_xy_vec=raw_xy_vec,
            n_sim=n_sim,
            rng=rng,
            settings=settings,
        )

    rows: list[dict[str, object]] = []
    for j_idx, pred_name in enumerate(pred_list):
        g_lo, g_hi = _safe_nan_quantiles(gamma_sim[:, j_idx])
        rows.append(
            {
                "locus": locus.locus_id,
                "predictor": pred_name,
                "outcome": out_str,
                "K": K,
                "gamma": float(gamma_hat[j_idx]),
                "beta_raw": float(beta_raw[j_idx]),
                "ci_low": g_lo,
                "ci_high": g_hi,
                "p": cond_pvals[j_idx],
                "r2": r2_joint,
                "r2_ci_low": r2_low,
                "r2_ci_high": r2_high,
                "p_joint": p_joint,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.attrs["r2"] = r2_joint
    out_df.attrs["p_joint"] = p_joint
    out_df.attrs["gamma"] = dict(zip(pred_list, [float(v) for v in gamma_hat]))
    return out_df


class LAVA:
    """
    Object-oriented LAVA estimator wrapping `process_locus`, `run_univ`, `run_bivar`,
    `run_pcor`, and `run_multireg`.
    """

    def __init__(
        self,
        lim: float = 0.99,
        univ_thresh: float | None = None,
        n_sim: int = 10000,
        seed: int = 42,
        settings: Literal["default", "R_default"] = "default",
    ) -> None:
        validate_settings(settings)
        if not (0.0 < lim <= 1.0):
            raise ValueError(f"lim must be in (0, 1]; got {lim}.")
        if n_sim < 100:
            raise ValueError(f"n_sim must be >= 100; got {n_sim}.")
        self.lim = lim
        self.univ_thresh = univ_thresh
        self.n_sim = n_sim
        self.seed = seed
        self.settings: SettingsMode = settings
        self.locus_: LAVALocus | None = None

    def process_locus(
        self,
        z_df: pd.DataFrame | np.ndarray,
        ld_R: pd.DataFrame | np.ndarray,
        sample_sizes: Mapping[str, float] | Sequence[float] | pd.Series,
        sample_overlap: pd.DataFrame | np.ndarray | None = None,
        locus_id: str = "locus_1",
    ) -> LAVALocus:
        """Project locus summary statistics and store `self.locus_`."""
        self.locus_ = process_locus(
            z_df=z_df,
            ld_R=ld_R,
            sample_sizes=sample_sizes,
            sample_overlap=sample_overlap,
            lim=self.lim,
            locus_id=locus_id,
            settings=self.settings,
        )
        return self.locus_

    def run_univ(
        self,
        locus: LAVALocus | None = None,
        traits: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        target = locus if locus is not None else self.locus_
        if target is None:
            raise ValueError("Call process_locus first or pass locus explicitly.")
        return run_univ(target, traits=traits, settings=self.settings)

    def run_bivar(
        self,
        locus: LAVALocus | None = None,
        traits: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        target = locus if locus is not None else self.locus_
        if target is None:
            raise ValueError("Call process_locus first or pass locus explicitly.")
        return run_bivar(
            target,
            traits=traits,
            univ_thresh=self.univ_thresh,
            n_sim=self.n_sim,
            seed=self.seed,
            settings=self.settings,
        )

    def run_pcor(
        self,
        pair: Sequence[str],
        cond_traits: str | Sequence[str],
        locus: LAVALocus | None = None,
    ) -> pd.DataFrame:
        target = locus if locus is not None else self.locus_
        if target is None:
            raise ValueError("Call process_locus first or pass locus explicitly.")
        return run_pcor(
            target,
            pair=pair,
            cond_traits=cond_traits,
            n_sim=self.n_sim,
            seed=self.seed,
            settings=self.settings,
        )

    def run_multireg(
        self,
        predictors: str | Sequence[str],
        outcome: str,
        locus: LAVALocus | None = None,
    ) -> pd.DataFrame:
        target = locus if locus is not None else self.locus_
        if target is None:
            raise ValueError("Call process_locus first or pass locus explicitly.")
        return run_multireg(
            target,
            predictors=predictors,
            outcome=outcome,
            n_sim=self.n_sim,
            seed=self.seed,
            settings=self.settings,
        )
