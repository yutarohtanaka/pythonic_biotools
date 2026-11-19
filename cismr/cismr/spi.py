"""Selecting Promoting Instruments (MR-SPI) for cis-Mendelian Randomization.

Reference
---------
Yao, M., Miller, G. W., Vardarajan, B. N., Belsky, D. W., Guo, Z., & Lin, Z. (2024).
Selecting valid instrumental variables in Mendelian randomization with cis-variants.
Biometrics, 80(1), ujad041. https://doi.org/10.1093/biomtc/ujad041
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .utils import (
    SettingsType,
    as_1d_float,
    prepare_ld_matrix,
    sym_solve,
    validate_settings,
    wald_inference,
)


@dataclass
class MRSPIResult:
    """Result of MR-SPI (Selecting Promoting Instruments) causal estimation."""

    estimate: float
    se: float
    pval: float
    ci_low: float
    ci_high: float
    n_snps: int
    n_relevant: int
    n_valid: int
    relevant_indices: np.ndarray
    selected_indices: np.ndarray
    invalid_indices: np.ndarray
    wald_ratios: np.ndarray
    wald_ses: np.ndarray
    voting_matrix: np.ndarray
    vote_threshold: float
    f_stat: float
    het_stat: float
    het_pval: float
    alpha: float
    settings: SettingsType

    @property
    def theta(self) -> float:
        return self.estimate

    @property
    def beta(self) -> float:
        return self.estimate

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "MR-SPI",
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "pval": self.pval,
            "n_snps": self.n_snps,
            "n_relevant": self.n_relevant,
            "n_valid": self.n_valid,
            "f_stat": self.f_stat,
            "het_stat": self.het_stat,
            "het_pval": self.het_pval,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"MR-SPI (settings={self.settings!r}): "
            f"theta={self.estimate:.4f} (SE={self.se:.4f}, p={self.pval:.3e}), "
            f"{int(round((1.0 - self.alpha) * 100))}% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"valid={self.n_valid}/{self.n_relevant} relevant ({self.n_snps} total)"
        )


def _find_maximum_clique(adj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Find maximum-cardinality clique via Bron-Kerbosch with Tomita pivoting.

    Ties in clique size are broken by maximum total first-stage instrument
    weight ``sum(weights[clique])``. Includes a budget cap for very dense graphs.
    """
    s = adj.shape[0]
    if s == 0:
        return np.array([], dtype=int)
    if s == 1:
        return np.array([0], dtype=int)

    neighbors = [set(np.where(adj[i] & (np.arange(s) != i))[0].tolist()) for i in range(s)]
    best_clique: list[int] = [int(np.argmax(weights))]
    best_weight: float = float(weights[best_clique[0]])
    calls = 0
    max_calls = 50_000

    def bk(r_set: list[int], p_set: set[int], x_set: set[int]) -> None:
        nonlocal best_clique, best_weight, calls
        calls += 1
        if calls > max_calls:
            return
        if len(r_set) + len(p_set) < len(best_clique):
            return
        if not p_set and not x_set:
            w_r = float(np.sum(weights[r_set])) if r_set else 0.0
            if len(r_set) > len(best_clique) or (
                len(r_set) == len(best_clique) and w_r > best_weight
            ):
                best_clique = list(r_set)
                best_weight = w_r
            return

        union_px = p_set | x_set
        pivot = max(union_px, key=lambda u: len(p_set & neighbors[u]))
        candidates = sorted(
            p_set - neighbors[pivot], key=lambda u: (-len(neighbors[u]), -weights[u])
        )
        for v in candidates:
            if calls > max_calls:
                break
            nv = neighbors[v]
            bk(r_set + [v], p_set & nv, x_set & nv)
            p_set.remove(v)
            x_set.add(v)

    bk([], set(range(s)), set())
    return np.array(sorted(best_clique), dtype=int)


def mr_spi(
    bx: np.ndarray | Sequence[float],
    sx: np.ndarray | Sequence[float],
    by: np.ndarray | Sequence[float],
    sy: np.ndarray | Sequence[float],
    ld_R: np.ndarray | None = None,
    n: int | float | None = None,
    alpha: float = 0.05,
    z_thresh: float | None = None,
    settings: SettingsType = "default",
) -> MRSPIResult:
    """Selecting Promoting Instruments (MR-SPI) for cis-MR (Yao et al., 2024).

    Parameters
    ----------
    bx, sx : array-like of shape (p,)
        Variant-exposure association estimates and standard errors.
    by, sy : array-like of shape (p,)
        Variant-outcome association estimates and standard errors.
    ld_R : ndarray of shape (p, p), optional
        Signed LD correlation matrix across cis-SNPs. Defaults to ``I_p``.
    n : int or float, optional
        Sample size (used for finite-sample strength reporting if provided).
    alpha : float, default=0.05
        Significance level for confidence intervals and compatibility testing.
    z_thresh : float, optional
        Optional override for first-stage ``|bx / sx|`` screening threshold.
        Defaults to Bonferroni ``Phi^{-1}(1 - 0.05 / (2 * p))``.
    settings : {"default", "R_default"}, default="default"
        Execution mode.
    """
    validate_settings(settings)
    bx_arr = as_1d_float(bx, "bx", min_len=1)
    p = bx_arr.shape[0]
    sx_arr = as_1d_float(sx, "sx", min_len=p)
    by_arr = as_1d_float(by, "by", min_len=p)
    sy_arr = as_1d_float(sy, "sy", min_len=p)
    if not (sx_arr.shape[0] == by_arr.shape[0] == sy_arr.shape[0] == p):
        raise ValueError("bx, sx, by, sy must all have the same length.")
    if np.any(sx_arr <= 0.0) or np.any(sy_arr <= 0.0):
        raise ValueError("Standard errors sx and sy must be strictly positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if n is not None and n <= 2:
        raise ValueError(f"n must be > 2, got {n}.")

    R_full = prepare_ld_matrix(ld_R, p, settings=settings) if ld_R is not None else np.eye(p)

    # Step 1: First-stage instrument strength screening
    z_x = bx_arr / sx_arr
    screen_thresh = (
        float(z_thresh)
        if z_thresh is not None
        else float(stats.norm.ppf(1.0 - 0.05 / (2.0 * p)))
    )
    if screen_thresh < 0.0:
        raise ValueError("z_thresh must be non-negative.")

    rel_idx = np.where(np.abs(z_x) > screen_thresh)[0]
    if rel_idx.size == 0:
        raise ValueError(
            f"No candidate SNPs passed first-stage strength screening "
            f"(max |z_x|={np.max(np.abs(z_x)):.2f} <= threshold={screen_thresh:.2f})."
        )

    s = rel_idx.size
    bx_s = bx_arr[rel_idx]
    sx_s = sx_arr[rel_idx]
    by_s = by_arr[rel_idx]
    sy_s = sy_arr[rel_idx]
    R_s = R_full[np.ix_(rel_idx, rel_idx)]

    # Step 2: LD-aware Wald ratios and pairwise compatibility voting
    gamma_s = by_s / bx_s
    term_y = sy_s / bx_s
    term_x = (gamma_s * sx_s) / bx_s
    cov_gamma = R_s * (np.outer(term_y, term_y) + np.outer(term_x, term_x))
    var_gamma = np.maximum(np.diag(cov_gamma), 1e-18)
    wald_ses = np.sqrt(var_gamma)

    if s == 1:
        voting_mat = np.ones((1, 1), dtype=bool)
        vote_thresh = 0.0
        local_valid = np.array([0], dtype=int)
    else:
        var_diff = var_gamma[:, None] + var_gamma[None, :] - 2.0 * cov_gamma
        min_floor = 1e-12 * (var_gamma[:, None] + var_gamma[None, :])
        var_diff = np.maximum(var_diff, min_floor)
        diff_mat = np.abs(gamma_s[:, None] - gamma_s[None, :])
        t_pair = diff_mat / np.sqrt(var_diff)
        np.fill_diagonal(t_pair, 0.0)

        if settings == "R_default":
            # Upstream MR.SPI Bonferroni pair voting threshold + max-degree voter rule
            vote_thresh = float(np.sqrt(2.0 * np.log(max(p, 2))))
            voting_mat = t_pair <= vote_thresh
            degrees = np.sum(voting_mat, axis=0)
            # Break ties by largest first-stage F = z_x^2
            order_scores = degrees + 1e-9 * (z_x[rel_idx] ** 2) / max(
                float(np.max(z_x[rel_idx] ** 2)), 1.0
            )
            anchor = int(np.argmax(order_scores))
            local_valid = np.where(voting_mat[:, anchor] & voting_mat[anchor, :])[0]
        else:
            # Finite-sample pairwise Bonferroni threshold + LD-aware maximum clique
            n_pairs = s * (s - 1)
            vote_thresh = float(stats.norm.ppf(1.0 - alpha / max(2, n_pairs)))
            voting_mat = t_pair <= vote_thresh
            local_valid = _find_maximum_clique(voting_mat, weights=z_x[rel_idx] ** 2)

    sel_idx = rel_idx[local_valid]
    invalid_mask = np.ones(s, dtype=bool)
    invalid_mask[local_valid] = False
    inv_idx = rel_idx[invalid_mask]

    # Step 3: Post-selection LD-aware causal estimation on selected valid set
    v = sel_idx.size
    bx_v = bx_arr[sel_idx]
    sx_v = sx_arr[sel_idx]
    by_v = by_arr[sel_idx]
    sy_v = sy_arr[sel_idx]

    if v == 1:
        theta_hat = float(by_v[0] / bx_v[0])
        se_hat = float(np.sqrt((sy_v[0] ** 2 + (theta_hat**2) * (sx_v[0] ** 2)) / (bx_v[0] ** 2)))
        f_stat = float((bx_v[0] / sx_v[0]) ** 2)
        het_stat = 0.0
        het_pval = np.nan
    elif ld_R is None:
        # Uncorrelated 2-step / iterated profile-score IVW
        w0 = 1.0 / (sy_v**2)
        theta_curr = float(np.sum(w0 * bx_v * by_v) / np.sum(w0 * (bx_v**2)))
        n_steps = 1 if settings == "R_default" else 20
        for _ in range(n_steps):
            wt = 1.0 / (sy_v**2 + (theta_curr**2) * (sx_v**2))
            theta_next = float(np.sum(wt * bx_v * by_v) / np.sum(wt * (bx_v**2)))
            if abs(theta_next - theta_curr) <= 1e-9 * (1.0 + abs(theta_curr)):
                theta_curr = theta_next
                break
            theta_curr = theta_next
        theta_hat = theta_curr
        w_final = 1.0 / (sy_v**2 + (theta_hat**2) * (sx_v**2))
        info = float(np.sum(w_final * (bx_v**2)))
        het_stat = float(np.sum(w_final * ((by_v - theta_hat * bx_v) ** 2)))
        het_df = v - 1
        het_pval = float(stats.chi2.sf(het_stat, df=het_df))
        se_base = 1.0 / np.sqrt(max(info, 1e-18))
        se_hat = (
            float(se_base * np.sqrt(max(1.0, het_stat / het_df)))
            if settings == "default"
            else se_base
        )
        f_stat = float(np.mean((bx_v / sx_v) ** 2))
    else:
        # Correlated cis-SNPs: project onto non-degenerate LD eigenmodes of R_v
        R_v = R_full[np.ix_(sel_idx, sel_idx)]
        eigvals, eigvecs = linalg.eigh(0.5 * (R_v + R_v.T))
        idx_desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_desc]
        eigvecs = eigvecs[:, idx_desc]
        pos = np.maximum(eigvals, 0.0)
        tot = float(np.sum(pos))
        cum = np.cumsum(pos) / max(tot, 1e-15)
        k_cut = int(np.searchsorted(cum, 0.999, side="left")) + 1
        k_pos = int(np.sum(eigvals > 1e-5))
        k_v = max(1, min(v, k_cut, max(k_pos, 1)))
        U_v = eigvecs[:, :k_v]

        Sigma_Xv = (R_v * sx_v[:, None]) * sx_v[None, :]
        Sigma_Yv = (R_v * sy_v[:, None]) * sy_v[None, :]
        t_bx = U_v.T @ bx_v
        t_by = U_v.T @ by_v
        t_Sx = 0.5 * (U_v.T @ Sigma_Xv @ U_v + (U_v.T @ Sigma_Xv @ U_v).T)
        t_Sy = 0.5 * (U_v.T @ Sigma_Yv @ U_v + (U_v.T @ Sigma_Yv @ U_v).T)

        denom0 = float(t_bx @ sym_solve(t_Sy, t_bx))
        theta_curr = float((t_bx @ sym_solve(t_Sy, t_by)) / max(denom0, 1e-18))
        n_steps = 1 if settings == "R_default" else 25
        for _ in range(n_steps):
            om = t_Sy + (theta_curr**2) * t_Sx
            theta_next = float(
                (t_bx @ sym_solve(om, t_by)) / max(float(t_bx @ sym_solve(om, t_bx)), 1e-18)
            )
            if abs(theta_next - theta_curr) <= 1e-9 * (1.0 + abs(theta_curr)):
                theta_curr = theta_next
                break
            theta_curr = theta_next

        theta_hat = theta_curr
        om_f = t_Sy + (theta_hat**2) * t_Sx
        resid = t_by - theta_hat * t_bx
        het_df = max(k_v - 1, 0)
        het_stat = max(float(resid @ sym_solve(om_f, resid)), 0.0) if het_df > 0 else 0.0
        het_pval = float(stats.chi2.sf(het_stat, df=het_df)) if het_df > 0 else np.nan
        info = float(t_bx @ sym_solve(om_f, t_bx))
        se_base = 1.0 / np.sqrt(max(info, 1e-18))
        se_hat = (
            float(se_base * np.sqrt(max(1.0, het_stat / het_df)))
            if (settings == "default" and het_df > 0)
            else se_base
        )
        f_stat = max(float(t_bx @ sym_solve(t_Sx, t_bx)) / max(k_v, 1), 0.0)

    pval, ci_low, ci_high = wald_inference(theta_hat, se_hat, alpha=alpha, df=None)

    return MRSPIResult(
        estimate=theta_hat,
        se=se_hat,
        pval=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_snps=p,
        n_relevant=s,
        n_valid=v,
        relevant_indices=rel_idx,
        selected_indices=sel_idx,
        invalid_indices=inv_idx,
        wald_ratios=gamma_s,
        wald_ses=wald_ses,
        voting_matrix=voting_mat,
        vote_threshold=vote_thresh,
        f_stat=f_stat,
        het_stat=het_stat,
        het_pval=het_pval,
        alpha=alpha,
        settings=settings,
    )
