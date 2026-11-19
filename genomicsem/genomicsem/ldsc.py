"""
Multivariate LD Score Regression (`ldsc_covstruct`) and leave-one-block-out
jackknife sampling covariance estimation (`GenomicCovStruct`) for `genomicsem`
(Grotzinger et al., Nature Human Behaviour 2019).
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


def vech_indices(k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Column-major lower-triangular indices `(rows, cols)` of a $k \\times k$ matrix,
    matching `GenomicSEM` and `lavaan` `vech` ordering:
    `(0,0), (1,0), ..., (k-1,0), (1,1), (2,1), ..., (k-1,k-1)`.
    """
    rows: list[int] = []
    cols: list[int] = []
    for c in range(k):
        for r in range(c, k):
            rows.append(r)
            cols.append(c)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def vech(mat: np.ndarray) -> np.ndarray:
    """Column-major lower-triangular half-vectorization of a $k \\times k$ matrix."""
    k = mat.shape[0]
    r_idx, c_idx = vech_indices(k)
    return np.asarray(mat[r_idx, c_idx], dtype=float)


def unvech(vec: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct symmetric $k \\times k$ matrix from column-major `vech` vector."""
    r_idx, c_idx = vech_indices(k)
    out = np.zeros((k, k), dtype=float)
    out[r_idx, c_idx] = vec
    out[c_idx, r_idx] = vec
    return out


def duplication_matrix(k: int) -> np.ndarray:
    """Magnus-Neudecker duplication matrix $D_k \\in \\mathbb{R}^{k^2 \\times p^*}$ (`vec(M) = D_k vech(M)`)."""
    p_star = k * (k + 1) // 2
    r_idx, c_idx = vech_indices(k)
    pos_map: dict[tuple[int, int], int] = {}
    for idx, (r, c) in enumerate(zip(r_idx.tolist(), c_idx.tolist())):
        pos_map[(r, c)] = idx
        pos_map[(c, r)] = idx
    D = np.zeros((k * k, p_star), dtype=float)
    # Column-major vec(M): index = col * k + row
    for col in range(k):
        for row in range(k):
            D[col * k + row, pos_map[(row, col)]] = 1.0
    return D


def stabilize_psd(mat: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Project symmetric matrix to have minimum eigenvalue >= `floor`."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if float(np.min(eigvals)) >= floor:
        return sym
    clipped = np.maximum(eigvals, floor)
    out = (eigvecs * clipped[np.newaxis, :]) @ eigvecs.T
    return 0.5 * (out + out.T)


def pinv_eig_floor(mat: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    """Symmetric spectral pseudo-inverse zeroing eigenvalues `< floor` (`GenomicSEM` parity)."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    inv_eig = np.where(eigvals >= floor, 1.0 / np.maximum(eigvals, floor), 0.0)
    out = (eigvecs * inv_eig[np.newaxis, :]) @ eigvecs.T
    return 0.5 * (out + out.T)


@dataclass
class GenomicCovStruct:
    """
    Multivariate LDSC genetic covariance structure for `GenomicSEM`.

    Attributes
    ----------
    S : np.ndarray
        $k \\times k$ genetic covariance matrix across observed traits.
    V_S : np.ndarray
        $p^* \\times p^*$ ($p^* = k(k+1)/2$) sampling covariance matrix of `vech(S)`.
    I_mat : np.ndarray
        $k \\times k$ bivariate LDSC intercept matrix capturing phenotypic sample overlap.
    trait_names : list[str]
        Ordered names of the $k$ observed traits.
    n_blocks : int
        Number of genomic jackknife blocks used to estimate `V_S`.
    settings : {"default", "R_default"}
        Active settings mode.
    """

    S: np.ndarray
    V_S: np.ndarray
    I_mat: np.ndarray | None = None
    trait_names: list[str] | None = None
    n_blocks: int = 200
    settings: SettingsMode = "default"

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if isinstance(self.S, pd.DataFrame):
            if self.trait_names is None:
                self.trait_names = [str(c) for c in self.S.columns]
            self.S = self.S.to_numpy(dtype=float)
        else:
            self.S = np.asarray(self.S, dtype=float)

        if self.S.ndim != 2 or self.S.shape[0] != self.S.shape[1]:
            raise ValueError(f"S must be a square 2D matrix; got shape {self.S.shape}.")
        k = self.S.shape[0]
        self.S = 0.5 * (self.S + self.S.T)

        if self.trait_names is None:
            self.trait_names = [f"T{i + 1}" for i in range(k)]
        else:
            self.trait_names = [str(t) for t in self.trait_names]
            if len(self.trait_names) != k:
                raise ValueError(
                    f"trait_names length ({len(self.trait_names)}) != S dimension ({k})."
                )

        p_star = k * (k + 1) // 2
        if isinstance(self.V_S, pd.DataFrame):
            self.V_S = self.V_S.to_numpy(dtype=float)
        else:
            self.V_S = np.asarray(self.V_S, dtype=float)
        if self.V_S.shape != (p_star, p_star):
            raise ValueError(
                f"V_S shape {self.V_S.shape} != expected ({p_star}, {p_star}) for k={k}."
            )
        self.V_S = 0.5 * (self.V_S + self.V_S.T)
        if self.settings == "default":
            self.V_S = stabilize_psd(self.V_S, floor=1e-12)

        if self.I_mat is None:
            self.I_mat = np.eye(k, dtype=float)
        elif isinstance(self.I_mat, pd.DataFrame):
            self.I_mat = self.I_mat.to_numpy(dtype=float)
        else:
            self.I_mat = np.asarray(self.I_mat, dtype=float)
            if self.I_mat.shape != (k, k):
                raise ValueError(f"I_mat shape {self.I_mat.shape} != ({k}, {k}).")
            self.I_mat = 0.5 * (self.I_mat + self.I_mat.T)

    def S_dataframe(self) -> pd.DataFrame:
        assert self.trait_names is not None
        return pd.DataFrame(self.S, index=self.trait_names, columns=self.trait_names)


def _wls_fit_and_jackknife(
    ell: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
) -> tuple[float, float, np.ndarray]:
    """
    Fast sufficient-statistic WLS fit `y ~ intercept + slope * ell` and exact
    leave-one-block-out jackknife slope vector of length `n_blocks`.
    """
    w = np.maximum(weights, 1e-18)
    w_ell = w * ell
    w_ell2 = w_ell * ell
    w_y = w * y
    w_elly = w_ell * y

    # Block-level sufficient sums via np.bincount
    bw = np.bincount(block_ids, weights=w, minlength=n_blocks)
    bwl = np.bincount(block_ids, weights=w_ell, minlength=n_blocks)
    bwl2 = np.bincount(block_ids, weights=w_ell2, minlength=n_blocks)
    bwy = np.bincount(block_ids, weights=w_y, minlength=n_blocks)
    bwly = np.bincount(block_ids, weights=w_elly, minlength=n_blocks)

    tw = float(np.sum(bw))
    twl = float(np.sum(bwl))
    twl2 = float(np.sum(bwl2))
    twy = float(np.sum(bwy))
    twly = float(np.sum(bwly))

    det_tot = max(tw * twl2 - twl * twl, 1e-24)
    intercept = (twl2 * twy - twl * twly) / det_tot
    slope = (tw * twly - twl * twy) / det_tot

    # Leave-one-block-out sums
    lw = tw - bw
    lwl = twl - bwl
    lwl2 = twl2 - bwl2
    lwy = twy - bwy
    lwly = twly - bwly
    det_loo = np.maximum(lw * lwl2 - lwl * lwl, 1e-24)
    slopes_loo = (lw * lwly - lwl * lwy) / det_loo
    return float(intercept), float(slope), slopes_loo


def ldsc_covstruct(
    sumstats_list: Sequence[pd.DataFrame] | pd.DataFrame | np.ndarray | tuple[object, ...] | GenomicCovStruct,
    ld_scores: Sequence[float] | np.ndarray | pd.Series | pd.DataFrame | None = None,
    sample_sizes: Sequence[float] | Mapping[str, float] | None = None,
    m_snps: int | float | None = None,
    trait_names: Sequence[str] | None = None,
    n_blocks: int = 200,
    settings: SettingsMode = "default",
) -> GenomicCovStruct:
    """
    Estimate the $k \\times k$ LDSC genetic covariance matrix $S$, bivariate intercept
    matrix $I_{\\text{mat}}$, and full $\\frac{k(k+1)}{2} \\times \\frac{k(k+1)}{2}$
    leave-one-block-out jackknife sampling covariance matrix $V_S$.

    Also accepts a precomputed `(S, V_S)` or `(S, V_S, I_mat)` tuple directly.
    """
    validate_settings(settings)
    if isinstance(sumstats_list, GenomicCovStruct):
        return GenomicCovStruct(
            S=sumstats_list.S.copy(),
            V_S=sumstats_list.V_S.copy(),
            I_mat=None if sumstats_list.I_mat is None else sumstats_list.I_mat.copy(),
            trait_names=sumstats_list.trait_names,
            n_blocks=sumstats_list.n_blocks,
            settings=settings,
        )

    # Direct precomputed (S, V_S[, I_mat]) tuple
    if isinstance(sumstats_list, tuple) and len(sumstats_list) in (2, 3) and ld_scores is None:
        S_pre = sumstats_list[0]
        VS_pre = sumstats_list[1]
        I_pre = sumstats_list[2] if len(sumstats_list) == 3 else None
        return GenomicCovStruct(
            S=S_pre,  # type: ignore[arg-type]
            V_S=VS_pre,  # type: ignore[arg-type]
            I_mat=I_pre,  # type: ignore[arg-type]
            trait_names=list(trait_names) if trait_names is not None else None,
            n_blocks=n_blocks,
            settings=settings,
        )

    if ld_scores is None:
        raise ValueError("ld_scores is required when estimating LDSC from summary statistics.")

    if isinstance(ld_scores, pd.DataFrame):
        cols_l = {str(c).lower(): c for c in ld_scores.columns}
        l_col = cols_l.get("l2", cols_l.get("ldscore", list(ld_scores.columns)[-1]))
        ell = ld_scores[l_col].to_numpy(dtype=float)
    elif isinstance(ld_scores, pd.Series):
        ell = ld_scores.to_numpy(dtype=float)
    else:
        ell = np.asarray(ld_scores, dtype=float).ravel()

    M = ell.shape[0]
    if M < 10:
        raise ValueError(f"LDSC requires at least 10 SNPs; got M={M}.")
    m_ref = float(m_snps) if m_snps is not None else float(M)

    # Extract Z-score matrix (M x k) and per-trait sample sizes (k,)
    if isinstance(sumstats_list, pd.DataFrame):
        df_z = sumstats_list.copy()
        drop_id = [c for c in df_z.columns if str(c).lower() in ("snp", "rsid", "id")]
        if drop_id:
            df_z = df_z.drop(columns=drop_id)
        inferred_names = [str(c) for c in df_z.columns]
        Z_mat = df_z.to_numpy(dtype=float)
        n_from_dfs: list[float | None] = [None] * Z_mat.shape[1]
    elif isinstance(sumstats_list, np.ndarray):
        Z_mat = np.asarray(sumstats_list, dtype=float)
        inferred_names = [f"T{i + 1}" for i in range(Z_mat.shape[1])]
        n_from_dfs = [None] * Z_mat.shape[1]
    else:
        dfs = list(sumstats_list)  # type: ignore[arg-type]
        if len(dfs) == 0:
            raise ValueError("sumstats_list cannot be empty.")
        z_cols: list[np.ndarray] = []
        n_from_dfs = []
        inferred_names = [f"T{i + 1}" for i in range(len(dfs))]
        for idx_t, df_t in enumerate(dfs):
            if not isinstance(df_t, pd.DataFrame):
                raise TypeError(f"Element {idx_t} of sumstats_list is not a DataFrame.")
            cl = {str(c).lower(): c for c in df_t.columns}
            if "z" in cl:
                z_vec = df_t[cl["z"]].to_numpy(dtype=float)
            elif "beta" in cl and "se" in cl:
                z_vec = df_t[cl["beta"]].to_numpy(dtype=float) / np.maximum(
                    df_t[cl["se"]].to_numpy(dtype=float), 1e-18
                )
            else:
                raise ValueError(f"Summary stats table {idx_t} needs 'Z' or ('BETA', 'SE').")
            z_cols.append(z_vec)
            if "n" in cl:
                n_from_dfs.append(float(np.median(df_t[cl["n"]].to_numpy(dtype=float))))
            else:
                n_from_dfs.append(None)
        Z_mat = np.column_stack(z_cols)

    if Z_mat.shape[0] != M:
        raise ValueError(f"Z-score SNP count ({Z_mat.shape[0]}) != ld_scores length ({M}).")
    k = Z_mat.shape[1]
    names = [str(t) for t in trait_names] if trait_names is not None else inferred_names
    if len(names) != k:
        raise ValueError(f"trait_names length ({len(names)}) != k ({k}).")

    # Resolve sample sizes N_p
    N_vec = np.zeros(k, dtype=float)
    if sample_sizes is None:
        for i in range(k):
            if n_from_dfs[i] is None:
                raise ValueError(
                    "sample_sizes must be provided when 'N' column is absent from sumstats."
                )
            N_vec[i] = float(n_from_dfs[i])  # type: ignore[arg-type]
    elif isinstance(sample_sizes, Mapping):
        for i, name in enumerate(names):
            N_vec[i] = float(sample_sizes[name])
    else:
        s_list = list(sample_sizes)
        if len(s_list) != k:
            raise ValueError(f"sample_sizes length ({len(s_list)}) != k ({k}).")
        N_vec = np.array(s_list, dtype=float)

    if np.any(N_vec <= 0.0):
        raise ValueError("All sample_sizes must be strictly positive.")

    eff_blocks = max(2, min(int(n_blocks), M // 2))
    # Contiguous genomic block assignment 0 .. eff_blocks-1
    block_ids = np.minimum((np.arange(M) * eff_blocks) // M, eff_blocks - 1)
    w_ld = 1.0 / np.maximum(ell, 1.0)
    mean_ell = max(float(np.mean(ell)), 1e-6)

    # Initial rough heritabilities for heteroskedasticity weights (Bulik-Sullivan et al. 2015)
    h2_init = np.zeros(k, dtype=float)
    for p in range(k):
        mean_chi2 = float(np.mean(Z_mat[:, p] ** 2))
        h2_init[p] = float(np.clip((mean_chi2 - 1.0) * m_ref / (N_vec[p] * mean_ell), 0.01, 1.0))

    S_mat = np.zeros((k, k), dtype=float)
    I_mat = np.zeros((k, k), dtype=float)
    r_idx, c_idx = vech_indices(k)
    p_star = len(r_idx)
    loo_vech = np.zeros((eff_blocks, p_star), dtype=float)

    for pos, (p, q) in enumerate(zip(r_idx.tolist(), c_idx.tolist())):
        y_pq = Z_mat[:, p] * Z_mat[:, q]
        scale_pq = np.sqrt(N_vec[p] * N_vec[q]) / m_ref

        v_p = 1.0 + (N_vec[p] * h2_init[p] / m_ref) * ell
        if p == q:
            het_var = 2.0 * v_p * v_p
        else:
            v_q = 1.0 + (N_vec[q] * h2_init[q] / m_ref) * ell
            cov_rough = float(np.mean(y_pq))
            het_var = v_p * v_q + cov_rough * cov_rough
        weights = w_ld / np.maximum(het_var, 1e-12)

        inter_pq, slope_pq, slopes_loo = _wls_fit_and_jackknife(
            ell=ell, y=y_pq, weights=weights, block_ids=block_ids, n_blocks=eff_blocks
        )
        sg_pq = slope_pq / scale_pq
        S_mat[p, q] = sg_pq
        S_mat[q, p] = sg_pq
        I_mat[p, q] = inter_pq
        I_mat[q, p] = inter_pq
        loo_vech[:, pos] = slopes_loo / scale_pq

    # Leave-one-block-out jackknife covariance V_S = ((B - 1) / B) * sum_b (s_{-b} - mean(s_{-b}))(...)^T
    centered = loo_vech - np.mean(loo_vech, axis=0, keepdims=True)
    V_S = ((eff_blocks - 1.0) / float(eff_blocks)) * (centered.T @ centered)
    V_S = 0.5 * (V_S + V_S.T)
    if settings == "default":
        V_S = stabilize_psd(V_S, floor=1e-10)

    return GenomicCovStruct(
        S=S_mat,
        V_S=V_S,
        I_mat=I_mat,
        trait_names=names,
        n_blocks=eff_blocks,
        settings=settings,
    )
