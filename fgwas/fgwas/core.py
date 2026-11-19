"""Hierarchical functional annotation enrichment and Bayesian GWAS fine-mapping (fgwas).

Reference
---------
Pickrell, J. K. (2014). Joint analysis of functional genomic data and genome-wide
association studies of 18 human traits. The American Journal of Human Genetics,
94(4), 559-573. https://doi.org/10.1016/j.ajhg.2014.03.004
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .abf import (
    SettingsType,
    normalize_prior_variances,
    validate_settings,
    wakefield_lnabf,
)


@dataclass
class FgwasResult:
    """Result of fitting a hierarchical fgwas enrichment / fine-mapping model."""

    gamma: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    annotations: tuple[str, ...]
    pi_region: float
    loglik: float
    unpenalized_loglik: float
    aic: float
    penalty: float
    V: tuple[float, ...]
    fine_map: bool
    n_snps: int
    n_blocks: int
    lnabf: np.ndarray
    ln_bf_region: np.ndarray
    PPA_region: np.ndarray
    PPA_snp: np.ndarray
    prior_snp: np.ndarray
    block_ids: np.ndarray
    block_idx: np.ndarray
    converged: bool
    settings: SettingsType

    @property
    def ppa_region(self) -> np.ndarray:
        return self.PPA_region

    @property
    def ppa_snp(self) -> np.ndarray:
        return self.PPA_snp

    def to_dataframe(self) -> pd.DataFrame:
        """Return parameter estimates, profile-likelihood 95% CIs, and fold enrichments."""
        return pd.DataFrame(
            {
                "annotation": list(self.annotations),
                "gamma": self.gamma,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "fold_enrichment": np.exp(np.clip(self.gamma, -50.0, 50.0)),
            }
        )

    def region_dataframe(self) -> pd.DataFrame:
        """Return per-block regional log Bayes factors and posterior probabilities."""
        return pd.DataFrame(
            {
                "block_id": self.block_ids,
                "ln_bf_region": self.ln_bf_region,
                "PPA_region": self.PPA_region,
            }
        )

    def snp_dataframe(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return per-SNP prior weights, lnABF, and re-weighted posterior probabilities."""
        out = df.copy() if df is not None else pd.DataFrame(index=np.arange(self.n_snps))
        out["block_id"] = self.block_ids[self.block_idx]
        out["lnabf"] = self.lnabf
        out["prior_snp"] = self.prior_snp
        out["PPA_snp"] = self.PPA_snp
        return out

    def summary(self) -> str:
        mode = "fine-map" if self.fine_map else f"genome-wide (pi_reg={self.pi_region:.4f})"
        header = (
            f"fgwas [{mode}] (settings={self.settings!r}, blocks={self.n_blocks}, "
            f"snps={self.n_snps}, llk={self.loglik:.2f}, AIC={self.aic:.2f}, penalty={self.penalty:.3g})"
        )
        if len(self.annotations) == 0:
            return header + "\n  (null model: 0 annotations)"
        rows = [
            f"  {name}: gamma={g:.4f} (95% CI [{lo:.4f}, {hi:.4f}], exp={np.exp(np.clip(g, -30, 30)):.2f}x)"
            for name, g, lo, hi in zip(self.annotations, self.gamma, self.ci_low, self.ci_high)
        ]
        return "\n".join([header, *rows])


def _extract_lnabf_and_annotations(
    df: pd.DataFrame,
    annotations: Sequence[str] | str,
    V: tuple[float, ...],
    settings: SettingsType,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract per-SNP ``lnabf`` and ``(N, A)`` annotation matrix from ``df``."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("df must be a non-empty pandas DataFrame.")

    if isinstance(annotations, str):
        annot_names = (annotations,)
    else:
        annot_names = tuple(str(a) for a in annotations)

    missing = [a for a in annot_names if a not in df.columns]
    if missing:
        raise ValueError(f"Annotation columns not found in df: {missing}.")

    cols_lower = {str(c).lower(): c for c in df.columns}
    if "beta" in cols_lower and "se" in cols_lower:
        beta_vals = df[cols_lower["beta"]].to_numpy(dtype=np.float64)
        se_vals = df[cols_lower["se"]].to_numpy(dtype=np.float64)
        lnabf = wakefield_lnabf(beta_vals, se_vals, V=V, settings=settings)
    elif "lnabf" in cols_lower:
        lnabf = df[cols_lower["lnabf"]].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(lnabf)):
            raise ValueError("Precomputed lnabf column contains non-finite values.")
    else:
        raise ValueError("df must contain ('beta', 'se') columns or a precomputed 'lnabf' column.")

    if len(annot_names) > 0:
        A_mat = df[list(annot_names)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(A_mat)):
            raise ValueError("Annotation columns contain non-finite values.")
    else:
        A_mat = np.empty((len(df), 0), dtype=np.float64)

    return lnabf, A_mat, annot_names


def _partition_blocks(
    df: pd.DataFrame,
    block_col: str | None,
    window_size: int,
    fine_map: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition SNPs into ``J`` blocks, returning ``(block_idx, block_ids)``."""
    n = len(df)
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")

    if block_col is not None:
        if block_col not in df.columns:
            raise ValueError(f"block_col={block_col!r} not found in df.columns.")
        codes, uniques = pd.factorize(df[block_col], sort=False)
        return np.asarray(codes, dtype=np.int64), np.asarray(uniques)

    if fine_map:
        for cand in ("SEGNUMBER", "locus", "region", "block"):
            if cand in df.columns:
                codes, uniques = pd.factorize(df[cand], sort=False)
                return np.asarray(codes, dtype=np.int64), np.asarray(uniques)
        return np.zeros(n, dtype=np.int64), np.array([0], dtype=np.int64)

    chr_col = next((c for c in ("chr", "chrom", "CHR", "CHROM") if c in df.columns), None)
    if chr_col is None:
        raw_idx = np.arange(n, dtype=np.int64) // int(window_size)
        n_b = int(raw_idx[-1]) + 1
        return raw_idx, np.arange(n_b, dtype=np.int64)

    chr_vals = df[chr_col].to_numpy()
    block_idx = np.empty(n, dtype=np.int64)
    curr_block = 0
    count_in_block = 0
    prev_chr = chr_vals[0]
    for i in range(n):
        if chr_vals[i] != prev_chr or count_in_block >= window_size:
            curr_block += 1
            count_in_block = 0
            prev_chr = chr_vals[i]
        block_idx[i] = curr_block
        count_in_block += 1
    return block_idx, np.arange(curr_block + 1, dtype=np.int64)


def _prepare_contiguous_blocks(
    lnabf: np.ndarray,
    A_mat: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Ensure SNPs are ordered contiguously by block for fast SIMD ``reduceat``."""
    is_sorted = bool(np.all(block_idx[:-1] <= block_idx[1:])) if block_idx.size > 1 else True
    if is_sorted:
        counts = np.bincount(block_idx, minlength=n_blocks)
        starts = np.zeros(n_blocks, dtype=np.int64)
        if n_blocks > 1:
            np.cumsum(counts[:-1], out=starts[1:])
        return lnabf, A_mat, block_idx, starts, None

    order = np.argsort(block_idx, kind="stable")
    b_sorted = block_idx[order]
    counts = np.bincount(b_sorted, minlength=n_blocks)
    starts = np.zeros(n_blocks, dtype=np.int64)
    if n_blocks > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    return lnabf[order], A_mat[order], b_sorted, starts, order


def _segmented_lse_and_softmax(
    x: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
    block_starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-block ``logsumexp`` and per-SNP ``softmax`` without overflow."""
    if block_starts is not None:
        max_b = np.maximum.reduceat(x, block_starts)
        exp_x = np.exp(x - max_b[block_idx])
        sum_exp = np.add.reduceat(exp_x, block_starts)
        lse = max_b + np.log(sum_exp)
        softmax = exp_x / sum_exp[block_idx]
        return lse, softmax

    max_b = np.full(n_blocks, -np.inf, dtype=np.float64)
    np.maximum.at(max_b, block_idx, x)
    exp_x = np.exp(x - max_b[block_idx])
    sum_exp = np.bincount(block_idx, weights=exp_x, minlength=n_blocks)
    lse = max_b + np.log(sum_exp)
    softmax = exp_x / sum_exp[block_idx]
    return lse, softmax


def _evaluate_fgwas_objective(
    lnabf: np.ndarray,
    A_mat: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
    fine_map: bool,
    penalty: float,
    eta: float,
    gamma: np.ndarray,
    block_starts: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact log-likelihood, analytical gradients, and posteriors."""
    n_annot = gamma.shape[0]
    if n_annot == 0:
        scores = np.zeros_like(lnabf)
    elif n_annot == 1:
        scores = A_mat[:, 0] * float(gamma[0])
    elif n_annot <= 8:
        scores = np.sum(A_mat * gamma[None, :], axis=1)
    else:
        scores = A_mat @ gamma

    lse_prior, pi_jk = _segmented_lse_and_softmax(scores, block_idx, n_blocks, block_starts)
    lse_post, w_jk = _segmented_lse_and_softmax(scores + lnabf, block_idx, n_blocks, block_starts)
    ln_bf_reg = lse_post - lse_prior

    if fine_map:
        unpen_llk = float(np.sum(ln_bf_reg))
        ppa_reg = np.ones(n_blocks, dtype=np.float64)
        ppa_snp = w_jk
        grad_eta = 0.0
        if n_annot > 0:
            diff_w = w_jk - pi_jk
            grad_gamma = (
                np.sum(A_mat * diff_w[:, None], axis=0) - 2.0 * penalty * gamma
                if n_annot <= 8
                else diff_w @ A_mat - 2.0 * penalty * gamma
            )
        else:
            grad_gamma = np.empty(0, dtype=np.float64)
    else:
        log_pi = -float(np.logaddexp(0.0, -eta))
        log_1m_pi = -float(np.logaddexp(0.0, eta))
        pi_reg = float(np.exp(log_pi))

        ll_j = np.logaddexp(log_1m_pi, log_pi + ln_bf_reg)
        unpen_llk = float(np.sum(ll_j))
        ppa_reg = np.exp(log_pi + ln_bf_reg - ll_j)
        ppa_snp = ppa_reg[block_idx] * w_jk
        grad_eta = float(np.sum(ppa_reg - pi_reg))
        if n_annot > 0:
            weights_snp = ppa_reg[block_idx] * (w_jk - pi_jk)
            grad_gamma = (
                np.sum(A_mat * weights_snp[:, None], axis=0) - 2.0 * penalty * gamma
                if n_annot <= 8
                else weights_snp @ A_mat - 2.0 * penalty * gamma
            )
        else:
            grad_gamma = np.empty(0, dtype=np.float64)

    pen_term = penalty * float(np.sum(gamma * gamma)) if n_annot > 0 else 0.0
    pen_llk = unpen_llk - pen_term
    return pen_llk, unpen_llk, grad_eta, grad_gamma, ln_bf_reg, ppa_reg, pi_jk, ppa_snp


def _optimize_fgwas_arrays(
    lnabf: np.ndarray,
    A_mat: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
    fine_map: bool,
    penalty: float,
    eta_init: float = -2.0,
    gamma_init: np.ndarray | None = None,
    block_starts: np.ndarray | None = None,
) -> tuple[float, np.ndarray, bool]:
    """Maximize penalized fgwas log-likelihood via L-BFGS-B."""
    n_annot = A_mat.shape[1]
    g0 = np.zeros(n_annot, dtype=np.float64) if gamma_init is None else gamma_init.copy()

    if fine_map and n_annot == 0:
        return 0.0, np.empty(0, dtype=np.float64), True

    if fine_map:
        bounds = [(-20.0, 20.0)] * n_annot

        def nll_fine(x: np.ndarray) -> tuple[float, np.ndarray]:
            pen_llk, _, _, grad_g, *_ = _evaluate_fgwas_objective(
                lnabf, A_mat, block_idx, n_blocks, True, penalty, 0.0, x, block_starts
            )
            return -pen_llk, -grad_g

        opt = optimize.minimize(
            nll_fine,
            g0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 200},
        )
        return 0.0, np.asarray(opt.x, dtype=np.float64), bool(opt.success)

    if gamma_init is None:
        _, _, _, _, ln_bf0, *_ = _evaluate_fgwas_objective(
            lnabf, A_mat, block_idx, n_blocks, False, penalty, eta_init, g0, block_starts
        )
        frac_sig = float(np.clip(np.mean(ln_bf0 > np.log(2.0)), 0.02, 0.85))
        eta0 = float(np.log(frac_sig / (1.0 - frac_sig)))
    else:
        eta0 = float(np.clip(eta_init, -15.0, 15.0))

    x0 = np.concatenate([[eta0], g0])
    bounds = [(-16.0, 16.0)] + [(-20.0, 20.0)] * n_annot

    def nll_gw(x: np.ndarray) -> tuple[float, np.ndarray]:
        pen_llk, _, grad_e, grad_g, *_ = _evaluate_fgwas_objective(
            lnabf, A_mat, block_idx, n_blocks, False, penalty, float(x[0]), x[1:], block_starts
        )
        return -pen_llk, -np.concatenate([[grad_e], grad_g])

    opt = optimize.minimize(
        nll_gw,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 200},
    )
    return float(opt.x[0]), np.asarray(opt.x[1:], dtype=np.float64), bool(opt.success)


def _profile_ci_gamma(
    lnabf: np.ndarray,
    A_mat: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
    fine_map: bool,
    penalty: float,
    eta_hat: float,
    gamma_hat: np.ndarray,
    max_pen_llk: float,
    llk_drop: float,
    block_starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute profile-likelihood confidence intervals for every ``gamma_a``."""
    n_annot = gamma_hat.shape[0]
    ci_low = np.empty(n_annot, dtype=np.float64)
    ci_high = np.empty(n_annot, dtype=np.float64)
    if n_annot == 0:
        return ci_low, ci_high

    target_llk = max_pen_llk - llk_drop

    for a in range(n_annot):
        other_idx = [j for j in range(n_annot) if j != a]

        def profile_llk(val_a: float) -> float:
            if fine_map and n_annot == 1:
                g_vec = np.array([val_a], dtype=np.float64)
                pllk, *_ = _evaluate_fgwas_objective(
                    lnabf, A_mat, block_idx, n_blocks, True, penalty, 0.0, g_vec, block_starts
                )
                return pllk

            if fine_map:
                x0_sub = gamma_hat[other_idx].copy()
                b_sub = [(-20.0, 20.0)] * (n_annot - 1)

                def sub_nll_fine(x_sub: np.ndarray) -> tuple[float, np.ndarray]:
                    g_full = np.empty(n_annot, dtype=np.float64)
                    g_full[a] = val_a
                    g_full[other_idx] = x_sub
                    pllk, _, _, grad_g, *_ = _evaluate_fgwas_objective(
                        lnabf, A_mat, block_idx, n_blocks, True, penalty, 0.0, g_full, block_starts
                    )
                    return -pllk, -grad_g[other_idx]

                res = optimize.minimize(
                    sub_nll_fine,
                    x0_sub,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=b_sub,
                    options={"ftol": 1e-8, "gtol": 1e-5, "maxiter": 60},
                )
                return -float(res.fun)

            x0_sub = np.concatenate([[eta_hat], gamma_hat[other_idx]])
            b_sub = [(-16.0, 16.0)] + [(-20.0, 20.0)] * (n_annot - 1)

            def sub_nll_gw(x_sub: np.ndarray) -> tuple[float, np.ndarray]:
                eta_v = float(x_sub[0])
                g_full = np.empty(n_annot, dtype=np.float64)
                g_full[a] = val_a
                g_full[other_idx] = x_sub[1:]
                pllk, _, grad_e, grad_g, *_ = _evaluate_fgwas_objective(
                    lnabf, A_mat, block_idx, n_blocks, False, penalty, eta_v, g_full, block_starts
                )
                return -pllk, -np.concatenate([[grad_e], grad_g[other_idx]])

            res = optimize.minimize(
                sub_nll_gw,
                x0_sub,
                method="L-BFGS-B",
                jac=True,
                bounds=b_sub,
                options={"ftol": 1e-8, "gtol": 1e-5, "maxiter": 60},
            )
            return -float(res.fun)

        g_center = float(gamma_hat[a])
        steps = (0.35, 0.8, 1.8, 4.0, 9.0, 20.0)

        lo_bound = -20.0
        found_lo = False
        prev_x = g_center
        for step in steps:
            cand = max(g_center - step, -20.0)
            if profile_llk(cand) < target_llk:
                lo_bound = float(
                    optimize.brentq(
                        lambda v: profile_llk(v) - target_llk,
                        cand,
                        prev_x,
                        xtol=1e-4,
                        maxiter=40,
                    )
                )
                found_lo = True
                break
            prev_x = cand
            if cand <= -20.0:
                break
        ci_low[a] = lo_bound if found_lo else -20.0

        hi_bound = 20.0
        found_hi = False
        prev_x = g_center
        for step in steps:
            cand = min(g_center + step, 20.0)
            if profile_llk(cand) < target_llk:
                hi_bound = float(
                    optimize.brentq(
                        lambda v: profile_llk(v) - target_llk,
                        prev_x,
                        cand,
                        xtol=1e-4,
                        maxiter=40,
                    )
                )
                found_hi = True
                break
            prev_x = cand
            if cand >= 20.0:
                break
        ci_high[a] = hi_bound if found_hi else 20.0

    return ci_low, ci_high


def fit_fgwas(
    df: pd.DataFrame,
    annotations: Sequence[str] | str,
    block_col: str | None = None,
    window_size: int = 5000,
    fine_map: bool = False,
    penalty: float | None = None,
    V: float | Sequence[float] | None = None,
    compute_ci: bool = True,
    settings: SettingsType = "default",
) -> FgwasResult:
    """Fit Pickrell's hierarchical fgwas enrichment & fine-mapping model."""
    validate_settings(settings)

    if V is None:
        eff_V = (0.1,) if (settings == "R_default" and fine_map) else (0.01, 0.1, 0.5)
    else:
        eff_V = normalize_prior_variances(V)

    lnabf_orig, A_orig, annot_names = _extract_lnabf_and_annotations(df, annotations, eff_V, settings)
    n_annot = len(annot_names)

    if penalty is None:
        eff_penalty = 0.1 if (settings == "default" and n_annot > 0) else 0.0
    else:
        eff_penalty = float(penalty)
    if not np.isfinite(eff_penalty) or eff_penalty < 0.0:
        raise ValueError(f"penalty must be finite and >= 0, got {penalty}.")

    block_idx_orig, block_ids = _partition_blocks(df, block_col, window_size, fine_map)
    n_blocks = int(block_ids.shape[0])

    lnabf, A_mat, block_idx, starts, order = _prepare_contiguous_blocks(
        lnabf_orig, A_orig, block_idx_orig, n_blocks
    )

    eta_hat, gamma_hat, converged = _optimize_fgwas_arrays(
        lnabf, A_mat, block_idx, n_blocks, fine_map, eff_penalty, block_starts=starts
    )
    pen_llk, unpen_llk, _, _, ln_bf_reg, ppa_reg, prior_s, ppa_s = _evaluate_fgwas_objective(
        lnabf, A_mat, block_idx, n_blocks, fine_map, eff_penalty, eta_hat, gamma_hat, starts
    )

    if order is not None:
        prior_snp = np.empty_like(prior_s)
        ppa_snp = np.empty_like(ppa_s)
        prior_snp[order] = prior_s
        ppa_snp[order] = ppa_s
    else:
        prior_snp = prior_s
        ppa_snp = ppa_s

    pi_reg = 1.0 if fine_map else float(np.exp(-np.logaddexp(0.0, -eta_hat)))
    n_params = n_annot + (0 if fine_map else 1)
    aic = 2.0 * float(n_params) - 2.0 * unpen_llk

    if compute_ci and n_annot > 0:
        llk_drop = (
            2.0 if settings == "R_default" else float(0.5 * stats.chi2.ppf(0.95, df=1))
        )
        ci_low, ci_high = _profile_ci_gamma(
            lnabf,
            A_mat,
            block_idx,
            n_blocks,
            fine_map,
            eff_penalty,
            eta_hat,
            gamma_hat,
            pen_llk,
            llk_drop,
            block_starts=starts,
        )
    else:
        ci_low = np.full(n_annot, np.nan, dtype=np.float64)
        ci_high = np.full(n_annot, np.nan, dtype=np.float64)

    return FgwasResult(
        gamma=gamma_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        annotations=annot_names,
        pi_region=pi_reg,
        loglik=pen_llk,
        unpenalized_loglik=unpen_llk,
        aic=aic,
        penalty=eff_penalty,
        V=eff_V,
        fine_map=fine_map,
        n_snps=int(lnabf_orig.shape[0]),
        n_blocks=n_blocks,
        lnabf=lnabf_orig,
        ln_bf_region=ln_bf_reg,
        PPA_region=ppa_reg,
        PPA_snp=ppa_snp,
        prior_snp=prior_snp,
        block_ids=block_ids,
        block_idx=block_idx_orig,
        converged=converged,
        settings=settings,
    )
