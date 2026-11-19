"""Sample-overlap correlation estimation and weak+strong IV selection for GRAPPLE."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike
import pandas as pd
from scipy import optimize, stats

from .loss import SettingsMode, validate_settings

_LEG_X16, _LEG_W16 = leggauss(16)


def _box_bvn_prob(c: float, rho: float, p_marginal: float) -> float:
    """Exact P(|Z_1| < c, |Z_2| < c) for standard bivariate normal via Plackett quadrature."""
    p0 = p_marginal * p_marginal
    if abs(rho) < 1e-12:
        return p0
    # Change variable u = sin(theta) from 0 to arcsin(rho): du / sqrt(1 - u^2) = dtheta!
    theta_max = float(np.arcsin(np.clip(rho, -0.9999, 0.9999)))
    half = 0.5 * theta_max
    thetas = half * (_LEG_X16 + 1.0)
    u = np.sin(thetas)
    c2 = c * c
    integrand = (np.exp(-c2 / (1.0 + u)) - np.exp(-c2 / (1.0 - u))) / np.pi
    integral = float(half * np.dot(_LEG_W16, integrand))
    return float(np.clip(p0 + integral, 1e-15, 1.0))


def _estimate_pair_corr_box_mle(za: np.ndarray, zb: np.ndarray, c: float, p_marg: float) -> float:
    """Truncated-box bivariate normal MLE of rho on {|Z_a| < c, |Z_b| < c}."""
    n = za.size
    if n < 15:
        return 0.0
    s_aa = float(np.sum(za * za))
    s_bb = float(np.sum(zb * zb))
    s_ab = float(np.sum(za * zb))

    def neg_ll(rho: float) -> float:
        one_m_r2 = 1.0 - rho * rho
        quad = (s_aa - 2.0 * rho * s_ab + s_bb) / (2.0 * one_m_r2)
        p_box = _box_bvn_prob(c, rho, p_marg)
        return float(0.5 * n * np.log(one_m_r2) + quad + n * np.log(p_box))

    opt = optimize.minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
    return float(np.clip(opt.x, -0.99, 0.99))


def _project_to_psd_correlation(mat: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix with unit diagonal onto the PSD correlation cone."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.maximum(eigvals, 1e-6)
    psd = (eigvecs * eigvals_clipped[None, :]) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(psd), 1e-12))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _extract_z_matrix(z_null_df: pd.DataFrame | ArrayLike) -> np.ndarray:
    """Extract (N, D) Z-score matrix from DataFrame or ndarray."""
    if isinstance(z_null_df, pd.DataFrame):
        cols_lower = {c.lower(): c for c in z_null_df.columns}
        # Check if bx/sx/by/sy summary stats were supplied
        if {"bx", "sx", "by", "sy"}.issubset(cols_lower.keys()):
            zx = z_null_df[cols_lower["bx"]].to_numpy(dtype=float) / z_null_df[
                cols_lower["sx"]
            ].to_numpy(dtype=float)
            zy = z_null_df[cols_lower["by"]].to_numpy(dtype=float) / z_null_df[
                cols_lower["sy"]
            ].to_numpy(dtype=float)
            return np.column_stack([zx, zy])
        z_cols = [
            c
            for c in z_null_df.columns
            if c.lower() not in ("snp", "rsid", "id", "variant_id", "chr", "pos")
            and pd.api.types.is_numeric_dtype(z_null_df[c])
        ]
        if len(z_cols) < 2:
            raise ValueError("z_null_df must contain at least 2 numeric Z-score columns.")
        return z_null_df[z_cols].to_numpy(dtype=float)

    arr = np.asarray(z_null_df, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("z_null_df must be a 2D array or DataFrame with >= 2 columns.")
    return arr


def estimate_cor_mat(
    z_null_df: pd.DataFrame | ArrayLike,
    p_thres: float = 0.5,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Estimate cross-trait GWAS error correlation matrix ``Sigma_e`` from null SNPs.

    Implements Wang et al., *PNAS* 2021 truncation-debiased null-SNP correlation
    estimation for ``p > p_thres`` (i.e. ``|Z| < Phi^{-1}(1 - p_thres / 2)``).
    """
    mode = validate_settings(settings)
    if not (0.0 < p_thres < 1.0):
        raise ValueError("p_thres must lie strictly in (0, 1).")

    z_mat = _extract_z_matrix(z_null_df)
    finite = np.all(np.isfinite(z_mat), axis=1)
    z_mat = z_mat[finite]
    n_snps, n_traits = z_mat.shape
    if n_snps < 30:
        raise ValueError("At least 30 finite SNPs are required to estimate error correlation.")

    z_cut = float(stats.norm.isf(p_thres / 2.0))
    p_marg = 1.0 - float(p_thres)  # 2 * Phi(z_cut) - 1

    cor_mat = np.eye(n_traits, dtype=float)
    if mode == "default":
        # Exact pairwise Plackett truncated-box bivariate normal MLE
        for a in range(n_traits):
            for b in range(a + 1, n_traits):
                mask = (np.abs(z_mat[:, a]) < z_cut) & (np.abs(z_mat[:, b]) < z_cut)
                rho_ab = _estimate_pair_corr_box_mle(
                    z_mat[mask, a], z_mat[mask, b], c=z_cut, p_marg=p_marg
                )
                cor_mat[a, b] = rho_ab
                cor_mat[b, a] = rho_ab
    else:
        # GRAPPLE::getInput R_default: joint null filter across all traits + truncated variance adjustment
        joint_mask = np.all(np.abs(z_mat) < z_cut, axis=1)
        if int(np.sum(joint_mask)) < 20:
            joint_mask = np.ones(n_snps, dtype=bool)
            emp_corr = np.corrcoef(z_mat[joint_mask], rowvar=False)
            return _project_to_psd_correlation(emp_corr)

        z_sub = z_mat[joint_mask]
        if n_traits == 2:
            rho_12 = _estimate_pair_corr_box_mle(
                z_sub[:, 0], z_sub[:, 1], c=z_cut, p_marg=p_marg
            )
            cor_mat[0, 1] = cor_mat[1, 0] = rho_12
        else:
            kappa = 1.0 - (2.0 * z_cut * float(stats.norm.pdf(z_cut))) / max(p_marg, 1e-12)
            raw_corr = np.corrcoef(z_sub, rowvar=False)
            for a in range(n_traits):
                for b in range(a + 1, n_traits):
                    # Adjust sample correlation by theoretical truncated normal variance attenuation kappa
                    rho_init = float(np.clip(raw_corr[a, b] / max(kappa, 1e-6), -0.95, 0.95))
                    # Refine on pairwise box to prevent >1 overshoot at high overlap
                    pair_mask = (np.abs(z_mat[:, a]) < z_cut) & (np.abs(z_mat[:, b]) < z_cut)
                    if int(np.sum(pair_mask)) >= 20:
                        rho_ab = _estimate_pair_corr_box_mle(
                            z_mat[pair_mask, a], z_mat[pair_mask, b], c=z_cut, p_marg=p_marg
                        )
                    else:
                        rho_ab = rho_init
                    cor_mat[a, b] = rho_ab
                    cor_mat[b, a] = rho_ab

    return _project_to_psd_correlation(cor_mat)


def _compute_selection_pvals(df: pd.DataFrame) -> np.ndarray:
    """Derive GRAPPLE selection p-values from explicit column or exposure summary stats."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ("selection_pvals", "pval_selection", "pval_exp", "pval"):
        if cand in cols_lower:
            p = df[cols_lower[cand]].to_numpy(dtype=float)
            return np.clip(p, 0.0, 1.0)

    # Single exposure (bx, sx) or (beta_exp, se_exp) or (gamma_exp, se_exp)
    for b_name, s_name in (
        ("bx", "sx"),
        ("beta_exp", "se_exp"),
        ("gamma_exp", "se_exp"),
        ("b_exp", "se_exp"),
        ("beta", "se"),
    ):
        if b_name in cols_lower and s_name in cols_lower:
            b = df[cols_lower[b_name]].to_numpy(dtype=float)
            s = df[cols_lower[s_name]].to_numpy(dtype=float)
            if np.any(s <= 0):
                raise ValueError("Exposure standard errors must be strictly positive.")
            return 2.0 * stats.norm.sf(np.abs(b / s))

    # Multivariable exposure columns: bx1..bxP & sx1..sxP or gamma_exp1..P & se_exp1..P
    bx_cols = sorted([c for c in df.columns if c.lower().startswith(("bx", "gamma_exp", "b_exp"))])
    sx_cols = sorted([c for c in df.columns if c.lower().startswith(("sx", "se_exp"))])
    if bx_cols and len(bx_cols) == len(sx_cols):
        b_mat = df[bx_cols].to_numpy(dtype=float)
        s_mat = df[sx_cols].to_numpy(dtype=float)
        if np.any(s_mat <= 0):
            raise ValueError("Exposure standard errors must be strictly positive.")
        p_mat = 2.0 * stats.norm.sf(np.abs(b_mat / s_mat))
        # Bonferroni minimum p-value across exposures matching GRAPPLE::getInput
        p_bonf = np.minimum(1.0, p_mat.shape[1] * np.min(p_mat, axis=1))
        return p_bonf

    raise ValueError(
        "Could not infer selection p-values from sumstats_df; provide 'selection_pvals', "
        "('bx', 'sx'), or ('bx1..P', 'sx1..P')."
    )


def select_grapple_instruments(
    sumstats_df: pd.DataFrame,
    ld_R: ArrayLike | None = None,
    p_thres: float = 1e-4,
    r2_prune: float = 0.001,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Select independent weak and strong genetic instruments for GRAPPLE.

    Filters variants to ``selection_pvals <= p_thres`` (supporting weak-instrument
    thresholds such as ``1e-4`` or ``1e-3``) and performs greedy LD pruning by
    ascending ``selection_pvals`` at ``r^2 <= r2_prune``.
    """
    validate_settings(settings)
    if not isinstance(sumstats_df, pd.DataFrame):
        raise TypeError("sumstats_df must be a pandas DataFrame.")
    if not (0.0 < p_thres <= 1.0):
        raise ValueError("p_thres must lie in (0, 1].")
    if not (0.0 <= r2_prune <= 1.0):
        raise ValueError("r2_prune must lie in [0, 1].")

    df = sumstats_df.copy()
    sel_p = _compute_selection_pvals(df)
    df["selection_pvals"] = sel_p

    ld_mat: np.ndarray | None = None
    if ld_R is not None:
        ld_mat = np.asarray(ld_R, dtype=float)
        if ld_mat.shape != (len(df), len(df)):
            raise ValueError("ld_R must be a square matrix matching len(sumstats_df).")

    cand_idx = np.where(np.isfinite(sel_p) & (sel_p <= p_thres))[0]
    if cand_idx.size == 0:
        return df.iloc[0:0].reset_index(drop=True)

    order = np.argsort(sel_p[cand_idx], kind="mergesort")
    sorted_idx = cand_idx[order]

    if ld_mat is None or sorted_idx.size <= 1:
        return df.iloc[sorted_idx].reset_index(drop=True)

    kept: list[int] = []
    alive = np.ones(sorted_idx.size, dtype=bool)
    for pos in range(sorted_idx.size):
        if not alive[pos]:
            continue
        lead = int(sorted_idx[pos])
        kept.append(lead)
        if pos + 1 < sorted_idx.size:
            rem_pos = np.where(alive[pos + 1 :])[0] + (pos + 1)
            rem_snps = sorted_idx[rem_pos]
            r2 = np.square(ld_mat[lead, rem_snps])
            alive[rem_pos[r2 > r2_prune]] = False

    return df.iloc[kept].reset_index(drop=True)
