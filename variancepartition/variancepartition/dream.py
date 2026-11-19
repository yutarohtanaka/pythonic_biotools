"""
Hoffman & Roussos (2021) ``dream``: precision-weighted mixed-effects differential
expression with O(m^3) REML Expected Fisher Information Satterthwaite degrees
of freedom and Empirical Bayes variance moderation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .ebayes import _moderated_p_and_fdr, squeeze_var
from .formula import DesignSpec, coerce_expression_and_weights, make_contrasts
from .lmm import LMMFitResult, fit_lmm


def _resolve_coef_index(coef: int | str, contrast_names: Sequence[str]) -> int:
    if isinstance(coef, (int, np.integer)):
        idx = int(coef)
        if idx < 0 or idx >= len(contrast_names):
            raise IndexError(
                f"Contrast index {idx} out of bounds for {len(contrast_names)} contrasts."
            )
        return idx
    name = str(coef)
    if name not in contrast_names:
        raise KeyError(
            f"Contrast '{name}' not found in contrast_names={list(contrast_names)}."
        )
    return list(contrast_names).index(name)


@dataclass(frozen=True)
class DreamFit:
    """Unmoderated mixed-effects contrast fit with Satterthwaite degrees of freedom."""

    gene_names: list[str]
    contrast_names: list[str]
    contrasts: np.ndarray
    log_fc: np.ndarray
    unscaled_var: np.ndarray
    sigma_e_sq: np.ndarray
    df_satterthwaite: np.ndarray
    ave_expr: np.ndarray
    lmm_fit: LMMFitResult

    def to_dataframe(
        self,
        coef: int | str = 0,
        sort_by_pvalue: bool = True,
        number: int | None = None,
    ) -> pd.DataFrame:
        """Return unmoderated contrast table using Satterthwaite df."""
        idx = _resolve_coef_index(coef, self.contrast_names)
        lfc = self.log_fc[:, idx]
        uvar = self.unscaled_var[:, idx]
        df_s = self.df_satterthwaite[:, idx]

        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(np.maximum(self.sigma_e_sq * uvar, 1e-300))
            valid = (
                np.isfinite(self.sigma_e_sq)
                & (self.sigma_e_sq > 0.0)
                & np.isfinite(lfc)
                & np.isfinite(uvar)
                & (uvar > 0.0)
                & np.isfinite(df_s)
                & (df_s > 0.0)
            )
            t_stat = np.where(valid, lfc / se, np.nan)

        p_val, adj_p, z_std = _moderated_p_and_fdr(t_stat, df_s)
        df = pd.DataFrame(
            {
                "logFC": lfc,
                "AveExpr": self.ave_expr,
                "t": t_stat,
                "P.Value": p_val,
                "adj.P.Val": adj_p,
                "z.std": z_std,
                "df.Satt": df_s,
                "df.total": df_s,
                "s2.post": self.sigma_e_sq,
            },
            index=self.gene_names,
        )
        if sort_by_pvalue:
            df = df.sort_values("P.Value", kind="mergesort", na_position="last")
        if number is not None:
            df = df.iloc[:number]
        return df

    def summary(self) -> pd.DataFrame:
        """Return per-contrast summary of unmoderated Satterthwaite fit."""
        rows: list[dict[str, Any]] = []
        for idx, cname in enumerate(self.contrast_names):
            df_col = self.df_satterthwaite[:, idx]
            valid = np.isfinite(df_col) & np.isfinite(self.log_fc[:, idx])
            rows.append(
                {
                    "contrast": cname,
                    "n_valid": int(np.sum(valid)),
                    "median_df_satt": float(np.nanmedian(df_col)) if np.any(valid) else np.nan,
                    "median_sigma_e_sq": float(np.nanmedian(self.sigma_e_sq[valid])) if np.any(valid) else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("contrast")


@dataclass(frozen=True)
class DreamResult:
    """Empirical Bayes moderated mixed-effects differential expression result."""

    gene_names: list[str]
    contrast_names: list[str]
    contrasts: np.ndarray
    log_fc: np.ndarray
    ave_expr: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    adj_p_val: np.ndarray
    z_std: np.ndarray
    df_satterthwaite: np.ndarray
    df_total: np.ndarray
    var_prior: list[float | np.ndarray]
    df_prior: list[float | np.ndarray]
    var_post: np.ndarray
    sigma_e_sq: np.ndarray
    lmm_fit: LMMFitResult

    def to_dataframe(
        self,
        coef: int | str = 0,
        sort_by_pvalue: bool = True,
        number: int | None = None,
    ) -> pd.DataFrame:
        """Return ``limma``/``dream``-compatible DataFrame for a contrast."""
        idx = _resolve_coef_index(coef, self.contrast_names)
        df = pd.DataFrame(
            {
                "logFC": self.log_fc[:, idx],
                "AveExpr": self.ave_expr,
                "t": self.t_stat[:, idx],
                "P.Value": self.p_value[:, idx],
                "adj.P.Val": self.adj_p_val[:, idx],
                "z.std": self.z_std[:, idx],
                "df.Satt": self.df_satterthwaite[:, idx],
                "df.total": self.df_total[:, idx],
                "s2.post": self.var_post[:, idx],
            },
            index=self.gene_names,
        )
        if sort_by_pvalue:
            df = df.sort_values("P.Value", kind="mergesort", na_position="last")
        if number is not None:
            df = df.iloc[:number]
        return df

    def summary(self) -> pd.DataFrame:
        """Return per-contrast summary of Empirical Bayes moderated DE results."""
        rows: list[dict[str, Any]] = []
        for idx, cname in enumerate(self.contrast_names):
            p_col = self.p_value[:, idx]
            adj_col = self.adj_p_val[:, idx]
            valid = np.isfinite(p_col)
            vp = self.var_prior[idx]
            dp = self.df_prior[idx]
            rows.append(
                {
                    "contrast": cname,
                    "n_valid": int(np.sum(valid)),
                    "n_sig_fdr05": int(np.sum(adj_col[valid] < 0.05)) if np.any(valid) else 0,
                    "var_prior": float(np.nanmedian(vp)) if np.ndim(vp) > 0 else float(vp),
                    "df_prior": float(np.nanmedian(dp)) if np.ndim(dp) > 0 else float(dp),
                    "median_df_satt": float(np.nanmedian(self.df_satterthwaite[:, idx])),
                    "median_df_total": float(np.nanmedian(self.df_total[:, idx])),
                }
            )
        return pd.DataFrame(rows).set_index("contrast")


def _is_contrast_estimable(c: np.ndarray, xtx: np.ndarray, xtx_pinv: np.ndarray) -> bool:
    return bool(np.allclose(c, xtx @ (xtx_pinv @ c), atol=1e-7, rtol=1e-7))


def dream(
    expr: Any,
    formula: str | DesignSpec | None = None,
    data: pd.DataFrame | None = None,
    contrasts: (
        np.ndarray
        | Sequence[float]
        | Sequence[str]
        | pd.DataFrame
        | Mapping[str, Sequence[float] | np.ndarray | str]
        | str
        | None
    ) = None,
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
    ebayes: bool = True,
    robust: bool = False,
    trend: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
    min_df: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    gamma_max: float = 1e6,
    n_jobs: int = 1,
) -> DreamResult | DreamFit:
    """Fit precision-weighted mixed-effects models and test linear contrasts via ``dream``.

    Notes
    -----
    Satterthwaite degrees of freedom are computed from the ``O(m^3)`` REML Expected
    Fisher Information in variance-component parameterization
    ``theta = (sigma_1^2, ..., sigma_R^2, sigma_e^2)``, yielding smooth conservative
    degrees of freedom even at boundary fits ``sigma_r^2 = 0`` when ``R >= 1``, and
    reducing identically to ``N_obs - rank(X_obs)`` when ``R = 0``.
    """
    if min_df <= 0.0:
        raise ValueError(f"min_df must be > 0, got {min_df}.")

    Y, _, _, design_spec, resolved_gene_names = coerce_expression_and_weights(
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

    p = design_spec.n_fixed
    if contrasts is None:
        intercept_set = set(design_spec.intercept_cols)
        non_int = [j for j in range(p) if j not in intercept_set]
        cols_to_test = non_int if non_int else list(range(p))
        C = np.zeros((p, len(cols_to_test)), dtype=np.float64)
        contrast_names = []
        for l_idx, j in enumerate(cols_to_test):
            C[j, l_idx] = 1.0
            contrast_names.append(design_spec.fixed_names[j])
    else:
        C, contrast_names = make_contrasts(contrasts, design_spec.fixed_names)

    lmm_fit = fit_lmm(
        expr=expr,
        formula=design_spec,
        data=data,
        weights=weights,
        layer=layer,
        genes_as_rows=genes_as_rows,
        gene_names=resolved_gene_names,
        max_iter=max_iter,
        tol=tol,
        gamma_max=gamma_max,
        compute_fisher=True,
        n_jobs=n_jobs,
    )

    N, G = Y.shape
    L = C.shape[1]
    R = design_spec.n_random_terms
    X_full = design_spec.X

    block_slices: list[slice] = []
    pos = 0
    for z in design_spec.random_matrices:
        sz = z.shape[1]
        block_slices.append(slice(pos, pos + sz))
        pos += sz

    # Check estimability on full design once
    xtx_full = X_full.T @ X_full
    rank_full = int(np.linalg.matrix_rank(X_full))
    if rank_full == p:
        full_estimable = np.ones(L, dtype=bool)
    else:
        xtx_pinv_full = np.linalg.pinv(xtx_full)
        full_estimable = np.array(
            [_is_contrast_estimable(C[:, l], xtx_full, xtx_pinv_full) for l in range(L)],
            dtype=bool,
        )

    log_fc = np.full((G, L), np.nan, dtype=np.float64)
    unscaled_var = np.full((G, L), np.nan, dtype=np.float64)
    df_satt = np.full((G, L), np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Y)

    for g in range(G):
        se2 = float(lmm_fit.sigma_e_sq[g])
        df_res = float(lmm_fit.df_residual_ols[g])
        if not np.isfinite(se2) or not np.isfinite(df_res) or df_res < 1.0:
            continue

        beta_g = lmm_fit.betas[g]
        Vb_g = lmm_fit.cov_beta_unscaled[g]
        if not np.all(np.isfinite(beta_g)) or not np.all(np.isfinite(Vb_g)):
            continue

        n_g = int(lmm_fit.n_obs[g])
        if n_g == N:
            est_mask = full_estimable
        else:
            obs = finite_mask[:, g]
            X_obs = X_full[obs]
            xtx_obs = X_obs.T @ X_obs
            xtx_pinv_obs = np.linalg.pinv(xtx_obs)
            est_mask = np.array(
                [_is_contrast_estimable(C[:, l], xtx_obs, xtx_pinv_obs) for l in range(L)],
                dtype=bool,
            )

        fisher_g = lmm_fit.fisher_info[g] if lmm_fit.fisher_info is not None else None
        azx_g = lmm_fit.a_zx[g] if lmm_fit.a_zx is not None else None
        gamma_g = (
            (lmm_fit.sigma_r_sq[g] / se2)
            if (R > 0 and se2 > 0.0 and np.all(np.isfinite(lmm_fit.sigma_r_sq[g])))
            else None
        )

        for l in range(L):
            if not est_mask[l]:
                continue
            c_l = C[:, l]
            u_l = float(c_l @ Vb_g @ c_l)
            if not np.isfinite(u_l) or u_l <= 0.0:
                continue

            log_fc[g, l] = float(c_l @ beta_g)
            unscaled_var[g, l] = u_l

            if R == 0 or se2 <= 0.0 or fisher_g is None or azx_g is None or gamma_g is None:
                df_satt[g, l] = df_res
            else:
                # Compute denominator using scaled Fisher matrix for numerical stability
                a_H = Vb_g @ c_l
                z_l = azx_g @ a_H
                g_vec = np.empty(R + 1, dtype=np.float64)
                for r_idx, sl in enumerate(block_slices):
                    zr = z_l[sl]
                    g_vec[r_idx] = float(zr @ zr)
                g_vec[R] = u_l - float(gamma_g @ g_vec[:R])

                # Symmetrically Jacobi-precondition Fisher matrix by its positive diagonal
                # so Satterthwaite df is invariant to W -> c * W, Z -> c * Z, and high ICC.
                f_sym = 0.5 * (fisher_g + fisher_g.T)
                diag_f = np.diag(f_sym)
                active = np.isfinite(diag_f) & (diag_f > 0.0)
                if not np.any(active):
                    df_satt[g, l] = df_res
                else:
                    f_act = f_sym[np.ix_(active, active)]
                    g_act = g_vec[active]
                    if not (np.all(np.isfinite(f_act)) and np.all(np.isfinite(g_act))):
                        df_satt[g, l] = df_res
                    else:
                        s_diag = np.sqrt(diag_f[active])
                        f_corr = (f_act / s_diag[:, None]) / s_diag[None, :]
                        f_corr = 0.5 * (f_corr + f_corr.T)
                        g_scaled = g_act / s_diag
                        eigvals, eigvecs = np.linalg.eigh(f_corr)
                        max_eig = float(eigvals[-1])
                        if not np.isfinite(max_eig) or max_eig <= 0.0:
                            df_satt[g, l] = df_res
                        else:
                            keep = eigvals > (1e-10 * max(max_eig, 1.0))
                            if not np.any(keep):
                                df_satt[g, l] = df_res
                            else:
                                proj = eigvecs[:, keep].T @ g_scaled
                                denom = float(np.sum((proj * proj) / eigvals[keep]))
                                num = 2.0 * (se2 * u_l) ** 2
                                if (
                                    not np.isfinite(num)
                                    or num <= 0.0
                                    or not np.isfinite(denom)
                                    or denom <= 1e-30 * num
                                ):
                                    df_satt[g, l] = df_res
                                else:
                                    raw_df = num / denom
                                    df_satt[g, l] = float(np.clip(raw_df, min_df, df_res))

    dream_fit = DreamFit(
        gene_names=resolved_gene_names,
        contrast_names=contrast_names,
        contrasts=C,
        log_fc=log_fc,
        unscaled_var=unscaled_var,
        sigma_e_sq=lmm_fit.sigma_e_sq,
        df_satterthwaite=df_satt,
        ave_expr=lmm_fit.ave_expr,
        lmm_fit=lmm_fit,
    )

    if not ebayes:
        return dream_fit
    return e_bayes(
        dream_fit,
        robust=robust,
        trend=trend,
        winsor_tail_p=winsor_tail_p,
    )


def e_bayes(
    fit: DreamFit | DreamResult,
    *,
    robust: bool = False,
    trend: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> DreamResult:
    """Apply Empirical Bayes variance moderation (``squeeze_var``) to a ``DreamFit``."""
    if isinstance(fit, DreamResult):
        # Allow re-moderating an existing DreamResult using its stored unscaled_var?
        # Recompute unscaled_var from C and lmm_fit.cov_beta_unscaled
        C = fit.contrasts
        G, L = fit.log_fc.shape
        uvar = np.full((G, L), np.nan, dtype=np.float64)
        for g in range(G):
            Vb = fit.lmm_fit.cov_beta_unscaled[g]
            if np.all(np.isfinite(Vb)):
                for l in range(L):
                    if np.isfinite(fit.log_fc[g, l]):
                        uvar[g, l] = float(C[:, l] @ Vb @ C[:, l])
        base_fit = DreamFit(
            gene_names=fit.gene_names,
            contrast_names=fit.contrast_names,
            contrasts=fit.contrasts,
            log_fc=fit.log_fc,
            unscaled_var=uvar,
            sigma_e_sq=fit.sigma_e_sq,
            df_satterthwaite=fit.df_satterthwaite,
            ave_expr=fit.ave_expr,
            lmm_fit=fit.lmm_fit,
        )
    elif isinstance(fit, DreamFit):
        base_fit = fit
    else:
        raise ValueError(f"`fit` must be a DreamFit or DreamResult, got {type(fit).__name__}.")

    G, L = base_fit.log_fc.shape
    t_stat = np.full((G, L), np.nan, dtype=np.float64)
    p_value = np.full((G, L), np.nan, dtype=np.float64)
    adj_p_val = np.full((G, L), np.nan, dtype=np.float64)
    z_std = np.full((G, L), np.nan, dtype=np.float64)
    df_total = np.full((G, L), np.nan, dtype=np.float64)
    var_post = np.full((G, L), np.nan, dtype=np.float64)
    var_prior_list: list[float | np.ndarray] = []
    df_prior_list: list[float | np.ndarray] = []

    cov_arg = base_fit.ave_expr if trend else None
    R = base_fit.lmm_fit.design.n_random_terms
    cov_fixed = base_fit.lmm_fit.cov_beta_fixed

    for l in range(L):
        valid_l = (
            np.isfinite(base_fit.sigma_e_sq)
            & (base_fit.sigma_e_sq > 0.0)
            & np.isfinite(base_fit.df_satterthwaite[:, l])
            & (base_fit.df_satterthwaite[:, l] > 0.0)
            & np.isfinite(base_fit.log_fc[:, l])
            & np.isfinite(base_fit.unscaled_var[:, l])
            & (base_fit.unscaled_var[:, l] > 0.0)
        )
        if R == 0:
            v_raw = base_fit.sigma_e_sq
            u_scale = base_fit.unscaled_var[:, l]
        else:
            c_l = base_fit.contrasts[:, l]
            if cov_fixed is not None:
                u0_l = np.einsum("i,gij,j->g", c_l, cov_fixed, c_l)
            else:
                X_f = base_fit.lmm_fit.design.X
                xtx_pinv_f = np.linalg.pinv(X_f.T @ X_f)
                u0_scalar = float(c_l @ xtx_pinv_f @ c_l)
                u0_l = np.full(G, u0_scalar, dtype=np.float64)
            valid_l = valid_l & np.isfinite(u0_l) & (u0_l > 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                v_raw = base_fit.sigma_e_sq * (base_fit.unscaled_var[:, l] / u0_l)
            u_scale = u0_l

        # Mask invalid/zero-variance genes out before squeeze_var so non-estimable
        # or constant genes do not distort the variance prior for this contrast
        v_in = np.where(valid_l, v_raw, np.nan)
        df1_in = np.where(valid_l, base_fit.df_satterthwaite[:, l], np.nan)

        vp_l, s20_l, df0_l, df_tot_l = squeeze_var(
            v_in,
            df1_in,
            covariate=cov_arg,
            robust=robust,
            winsor_tail_p=winsor_tail_p,
        )
        var_prior_list.append(s20_l)
        df_prior_list.append(df0_l)
        var_post[:, l] = vp_l
        df_total[:, l] = df_tot_l

        with np.errstate(invalid="ignore", divide="ignore"):
            se_l = np.sqrt(np.maximum(vp_l * u_scale, 1e-300))
            t_l = np.where(valid_l & np.isfinite(vp_l), base_fit.log_fc[:, l] / se_l, np.nan)

        p_l, adj_l, z_l = _moderated_p_and_fdr(t_l, df_tot_l)
        t_stat[:, l] = t_l
        p_value[:, l] = p_l
        adj_p_val[:, l] = adj_l
        z_std[:, l] = z_l

    return DreamResult(
        gene_names=list(base_fit.gene_names),
        contrast_names=list(base_fit.contrast_names),
        contrasts=base_fit.contrasts,
        log_fc=base_fit.log_fc,
        ave_expr=base_fit.ave_expr,
        t_stat=t_stat,
        p_value=p_value,
        adj_p_val=adj_p_val,
        z_std=z_std,
        df_satterthwaite=base_fit.df_satterthwaite,
        df_total=df_total,
        var_prior=var_prior_list,
        df_prior=df_prior_list,
        var_post=var_post,
        sigma_e_sq=base_fit.sigma_e_sq,
        lmm_fit=base_fit.lmm_fit,
    )


def top_table(
    result: DreamResult | DreamFit,
    coef: int | str = 0,
    *,
    number: int | None = None,
    sort_by: str = "P.Value",
    p_value_cutoff: float = 1.0,
    lfc_cutoff: float = 0.0,
) -> pd.DataFrame:
    """Extract a ranked ``limma``/``dream``-compatible summary table for a contrast."""
    df = result.to_dataframe(coef=coef, sort_by_pvalue=False, number=None)

    if p_value_cutoff < 1.0:
        df = df[df["adj.P.Val"] <= p_value_cutoff]
    if lfc_cutoff > 0.0:
        df = df[df["logFC"].abs() >= lfc_cutoff]

    if sort_by in ("P.Value", "p", "P"):
        df = df.sort_values("P.Value", ascending=True, kind="mergesort", na_position="last")
    elif sort_by in ("logFC", "lfc"):
        df = df.assign(_abs_lfc=df["logFC"].abs()).sort_values(
            "_abs_lfc", ascending=False, kind="mergesort", na_position="last"
        ).drop(columns=["_abs_lfc"])
    elif sort_by in ("t", "z.std"):
        df = df.assign(_abs_stat=df[sort_by].abs()).sort_values(
            "_abs_stat", ascending=False, kind="mergesort", na_position="last"
        ).drop(columns=["_abs_stat"])
    elif sort_by == "AveExpr":
        df = df.sort_values("AveExpr", ascending=False, kind="mergesort", na_position="last")
    elif sort_by == "none":
        pass
    else:
        raise ValueError(
            f"Invalid sort_by='{sort_by}'. Expected one of "
            "['P.Value', 'logFC', 't', 'z.std', 'AveExpr', 'none']."
        )

    if number is not None:
        df = df.iloc[:number]
    return df


eBayes = e_bayes
topTable = top_table
