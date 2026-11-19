"""
Hoffman & Schadt (2016) Intra-Class Correlation (ICC) variance partitioning
across fixed-effect terms, random-effect terms, and residual variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .formula import DesignSpec, coerce_expression_and_weights
from .lmm import LMMFitResult, fit_lmm, validate_settings


@dataclass(frozen=True)
class VarPartResult:
    """Typed result container for variance partitioning (``calcVarPart``)."""

    gene_names: list[str]
    term_names: list[str]
    fixed_terms: list[str]
    random_terms: list[str]
    fractions: np.ndarray
    variances: np.ndarray
    total_variance: np.ndarray
    singular: np.ndarray
    converged: np.ndarray

    def to_dataframe(
        self,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> pd.DataFrame:
        """Return ``(G x (K + R + 1))`` DataFrame of variance fractions indexed by ``gene_names``."""
        df = pd.DataFrame(self.fractions, index=self.gene_names, columns=self.term_names)
        if sort_by is not None:
            if sort_by not in self.term_names:
                raise ValueError(
                    f"sort_by='{sort_by}' not found in term_names={self.term_names}."
                )
            df = df.sort_values(
                sort_by,
                ascending=not descending,
                kind="mergesort",
                na_position="last",
            )
        return df

    def summary(self) -> pd.DataFrame:
        """Return per-term summary statistics across valid genes."""
        rows: list[dict[str, float | int | str]] = []
        for idx, name in enumerate(self.term_names):
            col = self.fractions[:, idx]
            valid = col[np.isfinite(col)]
            if valid.size == 0:
                rows.append(
                    {
                        "term": name,
                        "n_valid": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "std": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                )
            else:
                q25, med, q75 = np.quantile(valid, [0.25, 0.50, 0.75])
                rows.append(
                    {
                        "term": name,
                        "n_valid": int(valid.size),
                        "mean": float(np.mean(valid)),
                        "median": float(med),
                        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
                        "q25": float(q25),
                        "q75": float(q75),
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                    }
                )
        return pd.DataFrame(rows).set_index("term")


def extract_var_part(
    fit: LMMFitResult,
    *,
    weighted_fixed_var: bool = False,
    settings: Literal["default", "R_default"] = "default",
    _weights: np.ndarray | None = None,
    _obs_mask: np.ndarray | None = None,
) -> VarPartResult:
    """
    Compute Hoffman & Schadt (2016) variance fractions from a fitted ``LMMFitResult``.

    When `settings="R_default"`, matches `GabrielHoffman/variancePartition` (`calcVarPart.R`):
      - Fixed-effect terms use `(N - 1)/N` population variance (`ddof=0`) and are rescaled by
        `varFixedTotal / sum(fixedVar)` where `varFixedTotal = var_pop(X @ beta)` so correlated
        fixed predictors sum exactly to total fixed-effect variance.
      - Residual variance uses unweighted `sigma_e^2` directly (`attr(VarCorr(fit), "sc")^2`).
    """
    validate_settings(settings)
    if not isinstance(fit, LMMFitResult):
        raise ValueError(f"`fit` must be an LMMFitResult instance, got {type(fit).__name__}.")

    design = fit.design
    N = design.n_samples
    G = len(fit.gene_names)
    fixed_term_names = list(design.fixed_terms.keys())
    random_term_names = list(design.random_names)
    term_names = fixed_term_names + random_term_names + ["Residuals"]

    K = len(fixed_term_names)
    R = len(random_term_names)
    n_terms = K + R + 1

    variances = np.full((G, n_terms), np.nan, dtype=np.float64)
    fractions = np.full((G, n_terms), np.nan, dtype=np.float64)
    total_variance = np.full(G, np.nan, dtype=np.float64)

    valid_mask = (
        np.isfinite(fit.sigma_e_sq)
        & np.isfinite(fit.mean_inv_weight)
        & np.all(np.isfinite(fit.betas), axis=1)
    )
    if R > 0:
        valid_mask = valid_mask & np.all(np.isfinite(fit.sigma_r_sq), axis=1)

    if np.any(valid_mask):
        X = design.X
        if settings == "R_default":
            for k_idx, term in enumerate(fixed_term_names):
                cols = design.fixed_terms[term]
                X_k = X[:, cols]
                B_k = fit.betas[valid_mask][:, cols]
                eta_k = X_k @ B_k.T
                v_k = np.var(eta_k, axis=0, ddof=0)
                variances[valid_mask, k_idx] = np.maximum(v_k, 0.0)

            if K > 0:
                eta_tot = X @ fit.betas[valid_mask].T
                var_fixed_tot = np.maximum(np.var(eta_tot, axis=0, ddof=0), 0.0)
                sum_fixed = np.sum(variances[valid_mask, :K], axis=1)
                scale = np.where(
                    sum_fixed > 0.0,
                    var_fixed_tot / np.maximum(sum_fixed, 1e-300),
                    0.0,
                )
                variances[valid_mask, :K] = variances[valid_mask, :K] * scale[:, None]
        else:
            ddof = 1 if N > 1 else 0
            # Fixed-effect term variances
            for k_idx, term in enumerate(fixed_term_names):
                cols = design.fixed_terms[term]
                X_k = X[:, cols]  # (N, p_k)
                B_k = fit.betas[valid_mask][:, cols]  # (G_v, p_k)
                eta_k = X_k @ B_k.T  # (N, G_v)
                if weighted_fixed_var and _weights is not None:
                    if _weights.ndim == 1:
                        w_sub = np.broadcast_to(_weights[:, None], eta_k.shape)
                    else:
                        w_sub = _weights[:, valid_mask]
                    w_mask = np.isfinite(w_sub) & (w_sub > 0.0)
                    if _obs_mask is not None:
                        w_mask = w_mask & _obs_mask[:, valid_mask]
                    w_clean = np.where(w_mask, w_sub, 0.0)
                    w_sum = np.sum(w_clean, axis=0, keepdims=True)
                    w_norm = np.where(w_sum > 0.0, w_clean / np.maximum(w_sum, 1e-300), 0.0)
                    w_mean = np.sum(w_norm * eta_k, axis=0, keepdims=True)
                    v_k = np.sum(w_norm * (eta_k - w_mean) ** 2, axis=0)
                    n_eff = np.sum(w_mask, axis=0).astype(np.float64)
                    bessel = np.where(n_eff > 1.0, n_eff / np.maximum(n_eff - 1.0, 1.0), 1.0)
                    v_k = np.where(n_eff > 0.0, v_k * bessel, 0.0)
                else:
                    v_k = np.var(eta_k, axis=0, ddof=ddof)
                variances[valid_mask, k_idx] = np.maximum(v_k, 0.0)

        # Random-effect term variances: sigma_r^2 * (||Z_r||_F^2 / N)
        for r_idx, Z_r in enumerate(design.random_matrices):
            z_scale = float(np.sum(Z_r * Z_r)) / float(N)
            v_r = np.maximum(fit.sigma_r_sq[valid_mask, r_idx], 0.0) * z_scale
            variances[valid_mask, K + r_idx] = v_r

        # Residual variance: unweighted sigma_e^2 in R_default vs sigma_e^2 * mean(1 / w_i) in default
        if settings == "R_default":
            v_e = np.maximum(fit.sigma_e_sq[valid_mask], 0.0)
        else:
            v_e = np.maximum(fit.sigma_e_sq[valid_mask], 0.0) * fit.mean_inv_weight[valid_mask]
        variances[valid_mask, K + R] = v_e

        tot = np.sum(variances[valid_mask], axis=1)
        zero_tol = 1e-24 * (fit.ave_expr[valid_mask] ** 2)
        is_zero = tot <= zero_tol
        total_variance[valid_mask] = np.where(is_zero, 0.0, tot)

        valid_indices = np.flatnonzero(valid_mask)
        pos_idx = valid_indices[~is_zero]
        zero_idx = valid_indices[is_zero]

        if pos_idx.size > 0:
            frac_pos = variances[pos_idx] / total_variance[pos_idx, None]
            frac_pos = np.maximum(frac_pos, 0.0)
            frac_pos = frac_pos / np.sum(frac_pos, axis=1, keepdims=True)
            fractions[pos_idx] = frac_pos

        if zero_idx.size > 0:
            variances[zero_idx] = 0.0
            fractions[zero_idx, :-1] = 0.0
            fractions[zero_idx, -1] = 1.0

    return VarPartResult(
        gene_names=list(fit.gene_names),
        term_names=term_names,
        fixed_terms=fixed_term_names,
        random_terms=random_term_names,
        fractions=fractions,
        variances=variances,
        total_variance=total_variance,
        singular=fit.singular.copy(),
        converged=fit.converged.copy(),
    )


def fit_extract_var_part_model(
    expr: Any,
    formula: str | DesignSpec | None = None,
    data: pd.DataFrame | None = None,
    *,
    design: np.ndarray | pd.DataFrame | None = None,
    random_effects: (
        Sequence[np.ndarray | pd.Series | pd.DataFrame | str]
        | Mapping[str, np.ndarray | pd.Series | pd.DataFrame]
        | None
    ) = None,
    weights: np.ndarray | pd.DataFrame | str | None = None,
    layer: str | None = None,
    genes_as_rows: bool = False,
    gene_names: Sequence[str] | None = None,
    weighted_fixed_var: bool = False,
    reml: bool | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    gamma_max: float = 1e6,
    n_jobs: int = 1,
    settings: Literal["default", "R_default"] = "default",
) -> VarPartResult:
    """
    Fit mixed-effects models and extract variance-partition fractions in one step.

    - `settings="default"`: Fits REML (`reml=True` unless overridden), sample fixed variance,
      and weight-scale-invariant marginal residual variance.
    - `settings="R_default"`: Matches `GabrielHoffman/variancePartition` (`fitExtractVarPartModel.R`,
      `calcVarPart.R`): defaults to ML (`reml=False`), `(N-1)/N` fixed variances rescaled to sum to
      `var_pop(X @ beta)`, and unweighted `sigma_e^2`.
    """
    validate_settings(settings)
    eff_reml = (False if settings == "R_default" else True) if reml is None else bool(reml)

    w_arr: np.ndarray | None = None
    obs_mask: np.ndarray | None = None
    if weighted_fixed_var and weights is not None and settings == "default":
        y_arr, w_arr, _, _, _ = coerce_expression_and_weights(
            expr,
            formula=formula,
            data=data,
            design=design,
            random_effects=random_effects,
            weights=weights,
            layer=layer,
            genes_as_rows=genes_as_rows,
            gene_names=gene_names,
        )
        obs_mask = np.isfinite(y_arr)

    fit = fit_lmm(
        expr=expr,
        formula=formula,
        data=data,
        design=design,
        random_effects=random_effects,
        weights=weights,
        layer=layer,
        genes_as_rows=genes_as_rows,
        gene_names=gene_names,
        max_iter=max_iter,
        tol=tol,
        gamma_max=gamma_max,
        reml=eff_reml,
        compute_fisher=False,
        n_jobs=n_jobs,
        settings=settings,
    )
    return extract_var_part(
        fit,
        weighted_fixed_var=weighted_fixed_var,
        settings=settings,
        _weights=w_arr,
        _obs_mask=obs_mask,
    )
