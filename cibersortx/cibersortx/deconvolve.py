"""
Core linear nu-SVR deconvolution and empirical permutation p-value testing
(Newman et al., Nature Methods 2015; Nature Biotechnology 2019).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import NuSVR

from ._io import (
    ExpressionInput,
    SettingsInput,
    align_genes,
    parse_expression_matrix,
    parse_signature_matrix,
    validate_settings,
)


@dataclass(frozen=True)
class DeconvolutionResult:
    """Typed result container for `deconvolve` / `cibersort`."""

    fractions: np.ndarray
    sample_names: list[str]
    cell_types: list[str]
    p_value: np.ndarray
    pearson_r: np.ndarray
    rmse: np.ndarray
    best_nu: np.ndarray
    n_genes_used: int
    batch_mode: str
    adjusted_mixture: np.ndarray | None = None
    adjusted_signature: np.ndarray | None = None
    gene_names_used: list[str] | None = None

    def fractions_dataframe(self) -> pd.DataFrame:
        """Return `(N, K)` cell-type fractions DataFrame (`samples x cell_types`)."""
        return pd.DataFrame(
            self.fractions,
            index=self.sample_names,
            columns=self.cell_types,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return `(N, K + 4)` deconvolution table indexed by `sample_names` with
        cell-type fractions and `[p_value, pearson_r, rmse, best_nu]`.
        """
        df = self.fractions_dataframe()
        df["p_value"] = self.p_value
        df["pearson_r"] = self.pearson_r
        df["rmse"] = self.rmse
        df["best_nu"] = self.best_nu
        return df

    def summary(self) -> str:
        """Human-readable cohort summary."""
        n_samples, n_types = self.fractions.shape
        med_r = float(np.nanmedian(self.pearson_r)) if np.any(np.isfinite(self.pearson_r)) else float("nan")
        med_rmse = float(np.nanmedian(self.rmse)) if np.any(np.isfinite(self.rmse)) else float("nan")
        sig_str = "N/A"
        finite_p = np.isfinite(self.p_value)
        if np.any(finite_p):
            frac_sig = float(np.mean(self.p_value[finite_p] <= 0.05))
            sig_str = f"{frac_sig:.1%} (p<=0.05)"
        return (
            f"DeconvolutionResult(samples={n_samples}, cell_types={n_types}, "
            f"genes={self.n_genes_used}, batch_mode='{self.batch_mode}', "
            f"median_r={med_r:.4f}, median_rmse={med_rmse:.4f}, sig={sig_str})"
        )


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom <= 1e-15:
        return 0.0
    return float(np.clip(np.sum(xc * yc) / denom, -1.0, 1.0))


def _fit_best_nusvr(
    B_std: np.ndarray,
    m_std: np.ndarray,
    nu_grid: Sequence[float],
    C: float,
    svr_tol: float,
    svr_max_iter: int,
) -> tuple[np.ndarray, float, float, float]:
    """
    Fit linear `NuSVR` over `nu_grid`, enforce non-negativity + simplex
    normalization, and return `(f_best, pearson_r, rmse, best_nu)`.
    """
    n_genes, n_types = B_std.shape
    best_rmse = np.inf
    best_f: np.ndarray | None = None
    best_nu = float(nu_grid[0])

    for nu in nu_grid:
        svr = NuSVR(
            kernel="linear",
            nu=float(nu),
            C=float(C),
            tol=float(svr_tol),
            max_iter=int(svr_max_iter),
        )
        svr.fit(B_std, m_std)
        w = np.maximum(svr.coef_.ravel(), 0.0)
        w_sum = float(np.sum(w))
        if w_sum > 1e-15:
            f = w / w_sum
            u = B_std @ f
            err = float(np.sqrt(np.mean((u - m_std) ** 2)))
            if err < best_rmse:
                best_rmse = err
                best_f = f
                best_nu = float(nu)

    if best_f is None or not np.isfinite(best_rmse):
        fallback_rmse = float(np.sqrt(np.mean(m_std * m_std)))
        return np.zeros(n_types, dtype=np.float64), 0.0, fallback_rmse, float(nu_grid[0])

    u_best = B_std @ best_f
    r_best = _safe_pearsonr(u_best, m_std)
    return best_f, r_best, best_rmse, best_nu


def _deconvolve_core(
    M_0: np.ndarray,
    B_0: np.ndarray,
    *,
    gene_names: list[str],
    sample_names: list[str],
    cell_types: list[str],
    nu_grid: Sequence[float],
    C: float,
    n_permutations: int,
    permutation_mode: Literal["per_sample", "pooled"],
    svr_tol: float,
    svr_max_iter: int,
    absolute_mode: bool,
    seed: int,
    batch_mode: str,
) -> DeconvolutionResult:
    n_genes, n_samples = M_0.shape
    n_types = B_0.shape[1]

    mu_B = float(np.mean(B_0))
    sd_B = float(np.std(B_0, ddof=1))
    if sd_B <= 1e-12:
        raise ValueError("Signature matrix has zero variance across shared genes.")
    B_std = np.ascontiguousarray((B_0 - mu_B) / sd_B, dtype=np.float64)

    fractions = np.full((n_samples, n_types), np.nan, dtype=np.float64)
    pearson_r = np.full(n_samples, np.nan, dtype=np.float64)
    rmse = np.full(n_samples, np.nan, dtype=np.float64)
    best_nu = np.full(n_samples, np.nan, dtype=np.float64)
    p_value = np.full(n_samples, np.nan, dtype=np.float64)

    m_std_list: list[np.ndarray | None] = [None] * n_samples
    valid_indices: list[int] = []

    for j in range(n_samples):
        m_col = M_0[:, j]
        sd_m = float(np.std(m_col, ddof=1))
        if sd_m <= 1e-12:
            warnings.warn(
                f"Bulk sample '{sample_names[j]}' has zero variance across shared genes; setting row to NaN.",
                UserWarning,
                stacklevel=2,
            )
            continue
        mu_m = float(np.mean(m_col))
        m_std = np.ascontiguousarray((m_col - mu_m) / sd_m, dtype=np.float64)
        f_j, r_j, err_j, nu_j = _fit_best_nusvr(
            B_std, m_std, nu_grid=nu_grid, C=C, svr_tol=svr_tol, svr_max_iter=svr_max_iter
        )
        fractions[j, :] = f_j
        pearson_r[j] = r_j
        rmse[j] = err_j
        best_nu[j] = nu_j
        m_std_list[j] = m_std
        valid_indices.append(j)

    if n_permutations > 0 and valid_indices:
        rng = np.random.default_rng(seed)
        if permutation_mode == "per_sample":
            for j in valid_indices:
                m_std = m_std_list[j]
                assert m_std is not None
                ge_count = 0
                r_obs = pearson_r[j]
                for _ in range(n_permutations):
                    m_perm = m_std[rng.permutation(n_genes)]
                    _, r_null, _, _ = _fit_best_nusvr(
                        B_std,
                        m_perm,
                        nu_grid=nu_grid,
                        C=C,
                        svr_tol=svr_tol,
                        svr_max_iter=svr_max_iter,
                    )
                    if r_null >= r_obs:
                        ge_count += 1
                p_value[j] = (1.0 + ge_count) / (n_permutations + 1.0)
        else:  # permutation_mode == "pooled"
            null_r = np.empty(n_permutations, dtype=np.float64)
            valid_arr = np.asarray(valid_indices, dtype=int)
            for b in range(n_permutations):
                j_b = int(valid_arr[rng.integers(0, len(valid_arr))])
                m_std = m_std_list[j_b]
                assert m_std is not None
                m_perm = m_std[rng.permutation(n_genes)]
                _, r_null, _, _ = _fit_best_nusvr(
                    B_std,
                    m_perm,
                    nu_grid=nu_grid,
                    C=C,
                    svr_tol=svr_tol,
                    svr_max_iter=svr_max_iter,
                )
                null_r[b] = r_null
            for j in valid_indices:
                p_value[j] = (1.0 + float(np.sum(null_r >= pearson_r[j]))) / (n_permutations + 1.0)

    if absolute_mode and valid_indices:
        global_med = float(np.median(M_0[:, valid_indices]))
        denom_med = max(global_med, 1e-12)
        for j in valid_indices:
            sample_med = float(np.median(M_0[:, j]))
            fractions[j, :] = fractions[j, :] * (sample_med / denom_med)

    return DeconvolutionResult(
        fractions=fractions,
        sample_names=sample_names,
        cell_types=cell_types,
        p_value=p_value,
        pearson_r=pearson_r,
        rmse=rmse,
        best_nu=best_nu,
        n_genes_used=n_genes,
        batch_mode=batch_mode,
        adjusted_mixture=None,
        adjusted_signature=None,
        gene_names_used=gene_names,
    )


def deconvolve(
    mixture: ExpressionInput,
    signature: Any,
    *,
    layer: str | None = None,
    samples_as_rows: bool = False,
    nu_grid: Sequence[float] = (0.25, 0.5, 0.75),
    C: float = 1.0,
    n_permutations: int = 100,
    permutation_mode: Literal["per_sample", "pooled"] = "per_sample",
    batch_mode: Literal["none", "B", "S"] = "none",
    sc_ref: ExpressionInput | None = None,
    sc_cell_type_labels: str | Sequence[Any] | np.ndarray | pd.Series | None = None,
    sc_layer: str | None = None,
    sc_cells_as_rows: bool = False,
    normalize_sc_cells: bool = True,
    n_synthetic: int = 100,
    cells_per_mixture: int = 100,
    batch_space: Literal["reference", "common"] = "reference",
    is_log2: bool = False,
    log2_offset: float = 0.0,
    svr_tol: float = 1e-3,
    svr_max_iter: int = -1,
    absolute_mode: bool = False,
    seed: int = 0,
    settings: SettingsInput = "default",
) -> DeconvolutionResult:
    """
    Deconvolve bulk expression mixtures using linear nu-SVR (CIBERSORT /
    CIBERSORTx, Newman et al. 2015, 2019), with optional `B-mode` or `S-mode`
    ComBat batch alignment.
    """
    validate_settings(settings)
    if n_permutations < 0:
        raise ValueError("`n_permutations` must be >= 0.")
    if permutation_mode not in ("per_sample", "pooled"):
        raise ValueError(
            f"Unsupported permutation_mode '{permutation_mode}'. Expected 'per_sample' or 'pooled'."
        )
    if len(nu_grid) == 0 or any(not (0.0 < float(nu) <= 1.0) for nu in nu_grid):
        raise ValueError("`nu_grid` must be non-empty with values in (0, 1].")
    if C <= 0.0:
        raise ValueError("`C` must be positive.")

    mode_norm = "none" if str(batch_mode).lower() == "none" else str(batch_mode).upper()
    if mode_norm not in ("none", "B", "S"):
        raise ValueError(f"Unsupported batch_mode '{batch_mode}'. Expected 'none', 'B', or 'S'.")

    if mode_norm == "B":
        from .batch import b_mode_adjust

        return b_mode_adjust(
            mixture=mixture,
            signature=signature,
            layer=layer,
            samples_as_rows=samples_as_rows,
            nu_grid=nu_grid,
            C=C,
            n_permutations=n_permutations,
            permutation_mode=permutation_mode,
            batch_space=batch_space,
            is_log2=is_log2,
            log2_offset=log2_offset,
            svr_tol=svr_tol,
            svr_max_iter=svr_max_iter,
            absolute_mode=absolute_mode,
            seed=seed,
            settings=settings,
        )

    if mode_norm == "S":
        if sc_ref is None or sc_cell_type_labels is None:
            raise ValueError("`batch_mode='S'` requires both `sc_ref` and `sc_cell_type_labels`.")
        from .batch import s_mode_adjust

        return s_mode_adjust(
            mixture=mixture,
            signature=signature,
            sc_ref=sc_ref,
            sc_cell_type_labels=sc_cell_type_labels,
            layer=layer,
            sc_layer=sc_layer,
            samples_as_rows=samples_as_rows,
            sc_cells_as_rows=sc_cells_as_rows,
            normalize_sc_cells=normalize_sc_cells,
            n_synthetic=n_synthetic,
            cells_per_mixture=cells_per_mixture,
            nu_grid=nu_grid,
            C=C,
            n_permutations=n_permutations,
            permutation_mode=permutation_mode,
            batch_space=batch_space,
            is_log2=is_log2,
            log2_offset=log2_offset,
            svr_tol=svr_tol,
            svr_max_iter=svr_max_iter,
            absolute_mode=absolute_mode,
            seed=seed,
            settings=settings,
        )

    B_raw, sig_genes, cell_types = parse_signature_matrix(
        signature, is_log2=is_log2, log2_offset=log2_offset
    )
    M_raw, mix_genes, sample_names = parse_expression_matrix(
        mixture,
        layer=layer,
        samples_as_rows=samples_as_rows,
        default_row_prefix="gene",
        default_col_prefix="sample",
        allow_1d=True,
        name="mixture",
        is_log2=is_log2,
        log2_offset=log2_offset,
    )

    shared_genes, idx_sig, idx_mix = align_genes(sig_genes, mix_genes)
    n_types = len(cell_types)
    if len(shared_genes) < max(n_types, 3):
        raise ValueError(
            f"Shared gene count ({len(shared_genes)}) must be >= max(K={n_types}, 3)."
        )

    B_0 = B_raw[idx_sig, :]
    M_0 = M_raw[idx_mix, :]

    return _deconvolve_core(
        M_0,
        B_0,
        gene_names=shared_genes,
        sample_names=sample_names,
        cell_types=cell_types,
        nu_grid=nu_grid,
        C=C,
        n_permutations=n_permutations,
        permutation_mode=permutation_mode,
        svr_tol=svr_tol,
        svr_max_iter=svr_max_iter,
        absolute_mode=absolute_mode,
        seed=seed,
        batch_mode="none",
    )


cibersort = deconvolve
