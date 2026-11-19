"""
Locus LD eigendecomposition, orthonormal PC projection, and unbiased method-of-moments
local genetic covariance estimation (`process_locus`, `LAVALocus`) for `lava`
(Werme et al., Nature Genetics 2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` mode is `'default'` or `'R_default'`."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def project_psd_matrix(
    mat: np.ndarray,
    settings: SettingsMode = "default",
    pd_floor: float = 1e-6,
) -> np.ndarray:
    """
    Project symmetric matrix `mat` onto the PSD / PD cone according to `settings`.

    - `'default'`: when `mat` is non-PSD (`min_eig <= 1e-12`), floors eigenvalues at
      `pd_floor` (`1e-6`) to guarantee strict positive definiteness.
    - `'R_default'`: spectral zero-clipping `max(eig, 0.0)` matching `LAVA`.
    """
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if settings == "R_default":
        if float(np.min(eigvals)) >= 0.0:
            return sym
        clipped = np.maximum(eigvals, 0.0)
    else:
        if float(np.min(eigvals)) > 1e-12:
            return sym
        clipped = np.maximum(eigvals, pd_floor)
    out = (eigvecs * clipped[np.newaxis, :]) @ eigvecs.T
    return 0.5 * (out + out.T)


@dataclass
class LAVALocus:
    """
    Projected genomic locus representation for LAVA local genetic analyses.

    Attributes
    ----------
    locus_id : str
        Identifier of the genomic locus.
    traits : list[str]
        Ordered trait names ($P$ phenotypes).
    snp_ids : list[str]
        Ordered SNP identifiers ($M$ variants).
    n_snps : int
        Number of SNPs $M$ in the locus.
    K : int
        Number of retained orthonormal LD principal components explaining `lim` variance.
    eigenvalues : np.ndarray
        Retained top-$K$ LD eigenvalues $\\lambda_1 \\ge \\dots \\ge \\lambda_K$.
    eigenvectors : np.ndarray
        $M \\times K$ orthonormal LD eigenvectors $Q_K$.
    delta_hat : np.ndarray
        $K \\times P$ matrix $\\hat{\\Delta}$ of standardized marginal effects projected
        onto whitened genetic PCs ($\\Lambda_K^{-1/2} Q_K^\\top R_{\\text{marg}}$).
    sigma : np.ndarray
        $P \\times P$ per-PC sampling covariance matrix $\\Sigma$.
    omega_raw : np.ndarray
        $P \\times P$ unbiased method-of-moments local genetic covariance matrix
        $\\hat{\\Omega}_{\\text{raw}} = \\frac{1}{K}\\hat{\\Delta}^\\top\\hat{\\Delta} - \\Sigma$.
    omega : np.ndarray
        $P \\times P$ PSD/PD-projected local genetic covariance matrix $\\hat{\\Omega}$.
    sample_sizes : dict[str, float]
        Effective GWAS sample size $N_p$ per trait.
    sample_overlap : np.ndarray
        $P \\times P$ phenotypic error correlation matrix across traits.
    lim : float
        Cumulative LD variance threshold used to choose $K$.
    settings : {"default", "R_default"}
        Active settings mode.
    """

    locus_id: str
    traits: list[str]
    snp_ids: list[str]
    n_snps: int
    K: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    delta_hat: np.ndarray
    sigma: np.ndarray
    omega_raw: np.ndarray
    omega: np.ndarray
    sample_sizes: dict[str, float]
    sample_overlap: np.ndarray
    lim: float
    settings: SettingsMode = "default"

    def trait_indices(self, names: Sequence[str] | None = None) -> list[int]:
        """Resolve trait names to integer column indices."""
        if names is None:
            return list(range(len(self.traits)))
        lookup = {t: idx for idx, t in enumerate(self.traits)}
        out: list[int] = []
        for name in names:
            s = str(name)
            if s not in lookup:
                raise ValueError(
                    f"Unknown trait {s!r} in locus {self.locus_id!r}. "
                    f"Available traits: {self.traits}"
                )
            out.append(lookup[s])
        return out


def process_locus(
    z_df: pd.DataFrame | np.ndarray,
    ld_R: pd.DataFrame | np.ndarray,
    sample_sizes: Mapping[str, float] | Sequence[float] | pd.Series,
    sample_overlap: pd.DataFrame | np.ndarray | None = None,
    lim: float = 0.99,
    locus_id: str = "locus_1",
    settings: SettingsMode = "default",
) -> LAVALocus:
    """
    Project locus GWAS summary statistics onto pruned LD principal components and
    estimate the unbiased local genetic covariance matrix $\\hat{\\Omega}$.

    Implements Werme et al. (*Nature Genetics* 2022) Equations (1)-(6):
    1. Eigendecompose $R = Q \\Lambda Q^\\top$ and retain top $K$ PCs explaining
       cumulative variance proportion `lim` (`0.99`).
    2. Convert marginal Z-scores to SNP-phenotype correlations
       $r_{ip} = z_{ip} / \\sqrt{N_p - 1 + z_{ip}^2}$ and project onto whitened PCs:
       $\\hat{\\Delta} = \\Lambda_K^{-1/2} Q_K^\\top r \\in \\mathbb{R}^{K \\times P}$.
    3. Estimate per-PC sampling variance $\\hat{\\sigma}_p^2 = (1 - R^2_{\\text{obs}, p}) / (N_p - K - 1)$,
       build $\\Sigma_{pq} = \\hat{\\sigma}_p \\hat{\\sigma}_q C_{pq}$, and compute
       $\\hat{\\Omega} = \\frac{1}{K}\\hat{\\Delta}^\\top \\hat{\\Delta} - \\Sigma$.
    """
    validate_settings(settings)
    if not (0.0 < lim <= 1.0):
        raise ValueError(f"lim must be in (0, 1]; got {lim}.")

    if isinstance(z_df, pd.DataFrame):
        df = z_df.copy()
        id_cols = [c for c in df.columns if str(c).lower() in ("snp", "rsid", "variant", "id")]
        if id_cols:
            snp_ids = df[id_cols[0]].astype(str).tolist()
            df = df.drop(columns=id_cols)
        else:
            snp_ids = [str(x) for x in df.index.tolist()]
        trait_names = [str(c) for c in df.columns.tolist()]
        Z = df.to_numpy(dtype=float)
    else:
        Z = np.asarray(z_df, dtype=float)
        if Z.ndim != 2:
            raise ValueError("z_df array must be 2-dimensional (M_snps x P_traits).")
        snp_ids = [f"rs{i + 1}" for i in range(Z.shape[0])]
        if isinstance(sample_sizes, Mapping):
            trait_names = [str(k) for k in sample_sizes.keys()]
        else:
            trait_names = [f"T{i + 1}" for i in range(Z.shape[1])]

    M, P = Z.shape
    if M < 2:
        raise ValueError(f"Locus requires at least 2 SNPs; got M={M}.")
    if P < 1:
        raise ValueError("z_df must contain at least 1 trait column.")
    if np.any(~np.isfinite(Z)):
        raise ValueError("z_df contains non-finite Z-scores.")

    R_mat = ld_R.to_numpy(dtype=float) if isinstance(ld_R, pd.DataFrame) else np.asarray(ld_R, dtype=float)
    if R_mat.shape != (M, M):
        raise ValueError(f"ld_R shape {R_mat.shape} does not match (M, M)=({M}, {M}).")
    if np.any(~np.isfinite(R_mat)):
        raise ValueError("ld_R contains non-finite values.")
    R_sym = 0.5 * (R_mat + R_mat.T)

    # Resolve sample sizes N_p
    if isinstance(sample_sizes, pd.Series):
        n_dict = {t: float(sample_sizes.loc[t]) for t in trait_names}
    elif isinstance(sample_sizes, Mapping):
        missing = [t for t in trait_names if t not in sample_sizes]
        if missing:
            raise ValueError(f"Missing sample_sizes for traits: {missing}")
        n_dict = {t: float(sample_sizes[t]) for t in trait_names}
    else:
        n_seq = list(sample_sizes)
        if len(n_seq) != P:
            raise ValueError(f"sample_sizes length ({len(n_seq)}) != number of traits ({P}).")
        n_dict = {trait_names[i]: float(n_seq[i]) for i in range(P)}

    N_vec = np.array([n_dict[t] for t in trait_names], dtype=float)
    if np.any(N_vec <= 2.0):
        raise ValueError(f"All trait sample sizes must exceed 2; got {n_dict}.")

    # Resolve sample overlap correlation matrix C
    if sample_overlap is None:
        C_mat = np.eye(P, dtype=float)
    elif isinstance(sample_overlap, pd.DataFrame):
        C_mat = sample_overlap.loc[trait_names, trait_names].to_numpy(dtype=float)
    else:
        C_mat = np.asarray(sample_overlap, dtype=float)
        if C_mat.shape != (P, P):
            raise ValueError(f"sample_overlap shape {C_mat.shape} != ({P}, {P}).")
    C_mat = 0.5 * (C_mat + C_mat.T)
    # Normalize diagonal of sample overlap correlation to 1.0 if bivariate LDSC intercepts > 1 passed
    diag_c = np.diag(C_mat)
    if np.any(diag_c <= 0.0):
        raise ValueError("Diagonal of sample_overlap must be positive.")
    if not np.allclose(diag_c, 1.0, atol=1e-3):
        inv_sd = 1.0 / np.sqrt(diag_c)
        C_mat = C_mat * np.outer(inv_sd, inv_sd)
    # Ensure C_mat is valid PSD correlation matrix
    eig_c, vec_c = np.linalg.eigh(C_mat)
    if float(np.min(eig_c)) < 1e-8:
        C_mat = (vec_c * np.maximum(eig_c, 1e-8)[np.newaxis, :]) @ vec_c.T
        d_fix = 1.0 / np.sqrt(np.diag(C_mat))
        C_mat = C_mat * np.outer(d_fix, d_fix)

    # Eigendecompose R in descending eigenvalue order
    eigvals_all, eigvecs_all = np.linalg.eigh(R_sym)
    order = np.argsort(eigvals_all)[::-1]
    eigvals_desc = eigvals_all[order]
    eigvecs_desc = eigvecs_all[:, order]

    pos_mask = eigvals_desc > 1e-7
    if not np.any(pos_mask):
        raise ValueError("LD matrix ld_R has no positive eigenvalues.")
    eig_pos = eigvals_desc[pos_mask]
    vec_pos = eigvecs_desc[:, pos_mask]

    cum_var = np.cumsum(eig_pos) / float(np.sum(eig_pos))
    idx_reach = np.nonzero(cum_var >= lim - 1e-12)[0]
    K = int(idx_reach[0] + 1) if len(idx_reach) > 0 else len(eig_pos)

    # Ensure K leaves positive residual degrees of freedom N_p - K - 1 >= 1
    max_allowed_k = int(np.min(N_vec)) - 2
    if max_allowed_k < 1:
        raise ValueError("Sample sizes are too small relative to locus PCs.")
    K = max(1, min(K, max_allowed_k))

    lam_K = eig_pos[:K]
    Q_K = vec_pos[:, :K]

    # Convert marginal Z-scores to marginal correlations r_{ip} = z_{ip} / sqrt(N_p - 1 + z_{ip}^2)
    r_marg = Z / np.sqrt(N_vec[np.newaxis, :] - 1.0 + Z * Z)

    # Project onto orthonormal genetic PCs: Delta_hat = Lambda_K^{-1/2} Q_K^T r_marg
    delta_hat = (Q_K.T @ r_marg) / np.sqrt(lam_K)[:, np.newaxis]

    # Sample locus R^2 per trait and unbiased residual sampling variance per PC
    r2_obs = np.sum(delta_hat * delta_hat, axis=0)
    # Guard extreme synthetic Z-score inputs where uncalibrated r2_obs approaches/exceeds 1
    r2_clamped = np.minimum(r2_obs, 0.99)
    sigma_sq = (1.0 - r2_clamped) / (N_vec - K - 1.0)
    sigma_sd = np.sqrt(sigma_sq)
    sigma_mat = C_mat * np.outer(sigma_sd, sigma_sd)

    omega_raw = (delta_hat.T @ delta_hat) / float(K) - sigma_mat
    omega_raw = 0.5 * (omega_raw + omega_raw.T)
    omega_psd = project_psd_matrix(omega_raw, settings=settings, pd_floor=1e-6)

    return LAVALocus(
        locus_id=str(locus_id),
        traits=trait_names,
        snp_ids=snp_ids,
        n_snps=M,
        K=K,
        eigenvalues=lam_K,
        eigenvectors=Q_K,
        delta_hat=delta_hat,
        sigma=sigma_mat,
        omega_raw=omega_raw,
        omega=omega_psd,
        sample_sizes=n_dict,
        sample_overlap=C_mat,
        lim=lim,
        settings=settings,
    )
