"""
Core algorithms for HyPrColoc multi-trait Bayesian colocalization.

Reference:
  Foley, C. N., Staley, J. R., Breen, P. G., et al. (2021).
  "A fast and efficient colocalization algorithm for identifying shared genetic
  risk factors across multiple traits." Nature Communications, 12(1), 764.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import special


@dataclass(frozen=True)
class SubsetColocResult:
    """Regional, alignment, and SNP-level colocalization posteriors for a fixed subset of traits."""

    traits: list[str]
    trait_indices: list[int]
    snp_ids: list[str]
    regional_prob: float
    align_prob: float
    posterior_prob: float
    candidate_snp: str
    candidate_snp_index: int
    posterior_explained_by_snp: float
    snp_posteriors: np.ndarray

    def credible_set(self, coverage: float = 0.95) -> list[str]:
        """Return smallest set of SNPs whose cumulative conditional colocalization probability >= `coverage`."""
        if not (0.0 < coverage <= 1.0):
            raise ValueError("coverage must lie in (0, 1].")
        order = np.argsort(-self.snp_posteriors, kind="mergesort")
        cum = np.cumsum(self.snp_posteriors[order])
        k = int(np.searchsorted(cum, coverage, side="left")) + 1
        return [self.snp_ids[int(i)] for i in order[: min(k, len(self.snp_ids))]]

    def snp_table(self) -> pd.DataFrame:
        """Return per-SNP conditional colocalization posterior table sorted descending."""
        return (
            pd.DataFrame({"snp": self.snp_ids, "snp_posterior": self.snp_posteriors})
            .sort_values("snp_posterior", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )


@dataclass(frozen=True)
class HyPrColocCluster:
    """A colocalized cluster of >= 2 traits discovered by `hyprcoloc`."""

    cluster_id: int
    traits: list[str]
    trait_indices: list[int]
    posterior_prob: float
    regional_prob: float
    align_prob: float
    candidate_snp: str
    candidate_snp_index: int
    posterior_explained_by_snp: float
    snp_posteriors: np.ndarray
    credible_set_95: list[str]


@dataclass(frozen=True)
class HyPrColocResult:
    """Full output of `hyprcoloc` across all Q traits."""

    n_snps: int
    n_traits: int
    snp_ids: list[str]
    trait_names: list[str]
    clusters: list[HyPrColocCluster]
    uncolocalized_traits: list[str]
    lbf_matrix: np.ndarray
    prior_1: float
    prior_c: float
    reg_thresh: float
    align_thresh: float
    bb_selection: str
    uniform_priors: bool
    settings: str = "default"

    def to_dataframe(self) -> pd.DataFrame:
        """Return tidy cluster table matching R `hyprcoloc(...)$results`."""
        rows = []
        for cl in self.clusters:
            rows.append(
                {
                    "iteration": cl.cluster_id,
                    "traits": ", ".join(cl.traits),
                    "posterior_prob": cl.posterior_prob,
                    "regional_prob": cl.regional_prob,
                    "align_prob": cl.align_prob,
                    "candidate_snp": cl.candidate_snp,
                    "posterior_explained_by_snp": cl.posterior_explained_by_snp,
                    "n_traits": len(cl.traits),
                    "credible_set_95_size": len(cl.credible_set_95),
                }
            )
        if not rows or self.uncolocalized_traits:
            if self.uncolocalized_traits:
                rows.append(
                    {
                        "iteration": None,
                        "traits": ", ".join(self.uncolocalized_traits),
                        "posterior_prob": np.nan,
                        "regional_prob": np.nan,
                        "align_prob": np.nan,
                        "candidate_snp": "None",
                        "posterior_explained_by_snp": np.nan,
                        "n_traits": len(self.uncolocalized_traits),
                        "credible_set_95_size": 0,
                    }
                )
        return pd.DataFrame(rows)

    def credible_set(self, cluster_id: int = 1, coverage: float = 0.95) -> list[str]:
        """Return `coverage` credible set of SNPs for a specific 1-based `cluster_id`."""
        if not (0.0 < coverage <= 1.0):
            raise ValueError("coverage must lie in (0, 1].")
        for cl in self.clusters:
            if cl.cluster_id == cluster_id:
                order = np.argsort(-cl.snp_posteriors, kind="mergesort")
                cum = np.cumsum(cl.snp_posteriors[order])
                k = int(np.searchsorted(cum, coverage, side="left")) + 1
                return [self.snp_ids[int(i)] for i in order[: min(k, self.n_snps)]]
        raise ValueError(f"Cluster {cluster_id} not found (available: {[c.cluster_id for c in self.clusters]}).")

    def snp_scores(self, cluster_id: int = 1) -> pd.DataFrame:
        """Return per-SNP conditional posterior probabilities for `cluster_id`."""
        for cl in self.clusters:
            if cl.cluster_id == cluster_id:
                return (
                    pd.DataFrame({"snp": self.snp_ids, "snp_posterior": cl.snp_posteriors})
                    .sort_values("snp_posterior", ascending=False, kind="mergesort")
                    .reset_index(drop=True)
                )
        raise ValueError(f"Cluster {cluster_id} not found.")

    def summary(self) -> str:
        """Format human-readable HyPrColoc report."""
        lines = [
            "HyPrColoc Multi-Trait Colocalization Summary",
            "=" * 64,
            f"SNPs: {self.n_snps} | Traits: {self.n_traits} | prior_1={self.prior_1:g}, prior_c={self.prior_c:g}",
            f"Thresholds: reg_thresh={self.reg_thresh:g}, align_thresh={self.align_thresh:g} ({self.bb_selection})",
        ]
        if not self.clusters:
            lines.append("Colocalized Clusters: None")
        else:
            for cl in self.clusters:
                lines.append(
                    f"  Cluster {cl.cluster_id}: [{', '.join(cl.traits)}] | "
                    f"PP={cl.posterior_prob:.4f} (P_R={cl.regional_prob:.4f}, P_A={cl.align_prob:.4f}) | "
                    f"SNP={cl.candidate_snp} (explained={cl.posterior_explained_by_snp:.4f}, "
                    f"95% CS={len(cl.credible_set_95)} SNPs)"
                )
        if self.uncolocalized_traits:
            lines.append(f"Uncolocalized Traits: {', '.join(self.uncolocalized_traits)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SensitivityResult:
    """Prior and threshold sensitivity co-clustering stability matrix (`sensitivity_analysis`)."""

    trait_names: list[str]
    similarity_matrix: np.ndarray
    n_configurations: int
    grid_records: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return Q x Q pairwise trait co-clustering frequency DataFrame."""
        return pd.DataFrame(self.similarity_matrix, index=self.trait_names, columns=self.trait_names)

    def summary(self) -> str:
        df = self.to_dataframe().round(3)
        return (
            f"HyPrColoc Sensitivity Analysis ({self.n_configurations} parameter configurations)\n"
            + df.to_string()
        )


def _resolve_prior_w(
    n_traits: int,
    binary_outcomes: Sequence[bool | int] | np.ndarray | None,
    prior_w: float | Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if prior_w is not None:
        w_arr = np.asarray(prior_w, dtype=np.float64)
        if w_arr.ndim == 0:
            w_vec = np.full(n_traits, float(w_arr), dtype=np.float64)
        else:
            w_vec = w_arr.ravel()
            if w_vec.size != n_traits:
                raise ValueError("prior_w length must equal number of traits Q.")
        if not np.all(np.isfinite(w_vec)) or np.any(w_vec <= 0.0):
            raise ValueError("prior_w must be finite and strictly positive.")
        return w_vec

    if binary_outcomes is None:
        return np.full(n_traits, 0.15, dtype=np.float64)

    bin_arr = np.asarray(binary_outcomes).ravel()
    if bin_arr.size != n_traits:
        raise ValueError("binary_outcomes length must equal number of traits Q.")
    return np.where(bin_arr.astype(bool), 0.20, 0.15)


def _validate_settings(settings: str) -> Literal["default", "R_default"]:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")
    return settings  # type: ignore[return-value]


def _resolve_reg_steps(reg_steps: int | None, settings: Literal["default", "R_default"]) -> int:
    if reg_steps is None:
        return 1 if settings == "R_default" else 2
    rs = int(reg_steps)
    if rs not in (0, 1, 2):
        raise ValueError(f"reg_steps must be 0, 1, or 2 (got {reg_steps!r}).")
    return rs


def log_abf(
    betas: np.ndarray | pd.DataFrame,
    ses: np.ndarray | pd.DataFrame,
    *,
    binary_outcomes: Sequence[bool | int] | np.ndarray | None = None,
    prior_w: float | Sequence[float] | np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> np.ndarray:
    """Compute Wakefield log Approximate Bayes Factors (H1 vs H0) of shape `(M, Q)`.

    Uses prior standard deviation `W = 0.15` for quantitative traits (`binary_outcomes=False`)
    and `W = 0.20` for binary case-control traits (`binary_outcomes=True`), unless overridden
    via `prior_w`.
    """
    _validate_settings(settings)
    b = np.asarray(betas, dtype=np.float64)
    s = np.asarray(ses, dtype=np.float64)
    if b.ndim != 2 or s.ndim != 2 or b.shape != s.shape:
        raise ValueError("betas and ses must be 2D arrays of identical shape (M, Q).")
    m, q = b.shape
    if m < 1 or q < 1:
        raise ValueError("betas and ses must be non-empty.")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(s))):
        raise ValueError("betas and ses must contain only finite values (no NaN or Inf).")
    if np.any(s <= 0.0):
        raise ValueError("All standard errors in ses must be strictly positive.")

    w_vec = _resolve_prior_w(q, binary_outcomes, prior_w)
    w_sq = w_vec[None, :] ** 2
    v = s**2
    r = w_sq / (v + w_sq)
    z_sq = (b**2) / v
    return 0.5 * np.log(1.0 - r) + 0.5 * r * z_sq


def _log_config_priors(
    max_k: int,
    prior_1: float,
    prior_c: float,
    uniform_priors: bool,
    *,
    n_snps: int = 1,
    settings: Literal["default", "R_default"] = "default",
) -> np.ndarray:
    """Compute log(pi_k) for k = 0, 1, ..., max_k per Foley et al. (2021)."""
    log_pi = np.zeros(max_k + 1, dtype=np.float64)
    if uniform_priors:
        p1_eff = (prior_1 / float(n_snps)) if settings == "R_default" else prior_1
        log_p1_eff = np.log(p1_eff)
        for k in range(1, max_k + 1):
            log_pi[k] = log_p1_eff
    else:
        if max_k >= 1:
            log_pi[1] = np.log(prior_1)
        gamma = 1.0 - prior_c
        for k in range(2, max_k + 1):
            # Step probability for adding k-th trait given k-1 traits: 1 - gamma^(k-1)
            step = -np.expm1((k - 1) * np.log(gamma)) if gamma > 0.0 else 1.0
            log_pi[k] = log_pi[k - 1] + np.log(step)
    return log_pi


def _validate_corr_matrix(mat: np.ndarray, expected_dim: int, name: str, require_pd: bool = False) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (expected_dim, expected_dim):
        raise ValueError(f"{name} must have shape ({expected_dim}, {expected_dim}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite entries.")
    if not np.allclose(arr, arr.T, atol=1e-7):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(arr), 1.0, atol=1e-6):
        raise ValueError(f"{name} must have unit diagonal.")
    if np.any(np.abs(arr) > 1.0 + 1e-7):
        raise ValueError(f"{name} correlations must lie in [-1, 1].")
    sym = 0.5 * (arr + arr.T)
    eigs = np.linalg.eigvalsh(sym)
    if require_pd and eigs[0] <= 1e-10:
        raise ValueError(f"{name} must be strictly positive definite (min eig = {eigs[0]:.3e}).")
    if not require_pd and eigs[0] < -1e-6:
        raise ValueError(f"{name} must be positive semi-definite (min eig = {eigs[0]:.3e}).")
    return sym


def _joint_multivariate_lbf_subset(
    b_sub: np.ndarray,
    s_sub: np.ndarray,
    w_sub: np.ndarray,
    cor_sub: np.ndarray,
) -> np.ndarray:
    """Compute multivariate Wakefield log-BF across correlated traits `S` at all M SNPs.

    Reduces identically to `sum_{q in S} lbf_{jq}` when `cor_sub == I`.
    """
    m, k = b_sub.shape
    cor_inv = np.linalg.inv(cor_sub)
    # Standardize to z-scores: V_j = D_j R D_j => V_j^{-1} = D_j^{-1} R^{-1} D_j^{-1}
    # Let u_{jq} = w_q / s_{jq}; then I + V_j^{-1} W = I + D_j^{-1} R^{-1} D_j^{-1} W
    # Symmetrized similarity: A_j = R + diag(w^2 / s_j^2)^{-1}? Easier:
    # det(I + V_j^{-1} W) = det(R)^{-1} det(R + diag(w_q^2 / s_{jq}^2) ???)
    # Check carefully! V_j + W = D_j (R + U_j^2) D_j where U_j = diag(w / s_j)!
    # Let's verify: D_j (R + U_j^2) D_j = D_j R D_j + D_j diag(w^2 / s_j^2) D_j = V_j + W !!
    # Therefore:
    # 1) log det(I + V_j^{-1} W) = log det(V_j + W) - log det(V_j) = log det(R + U_j^2) - log det(R) !!
    # 2) beta_j^T (V_j^{-1} - (V_j + W)^{-1}) beta_j = z_j^T (R^{-1} - (R + U_j^2)^{-1}) z_j !!
    sign_r, logdet_r = np.linalg.slogdet(cor_sub)
    if sign_r <= 0:
        raise ValueError("Trait correlation submatrix is not positive definite.")

    z = b_sub / s_sub  # (M, k)
    u_sq = (w_sub[None, :] ** 2) / (s_sub**2)  # (M, k)
    A = np.broadcast_to(cor_sub, (m, k, k)).copy()
    idx = np.arange(k)
    A[:, idx, idx] += u_sq

    _, logdet_A = np.linalg.slogdet(A)  # (M,)
    quad_r_inv = np.einsum("mi,ij,mj->m", z, cor_inv, z)
    A_inv_z = np.linalg.solve(A, z[..., None])[..., 0]  # (M, k)
    quad_A_inv = np.sum(z * A_inv_z, axis=1)

    return -0.5 * (logdet_A - logdet_r) + 0.5 * (quad_r_inv - quad_A_inv)


def _compute_subset_posteriors(
    lbf_full: np.ndarray,
    subset_indices: Sequence[int],
    log_pi: np.ndarray,
    uniform_priors: bool,
    *,
    prior_1: float = 1e-4,
    reg_steps: int = 2,
    settings: Literal["default", "R_default"] = "default",
    betas: np.ndarray | None = None,
    ses: np.ndarray | None = None,
    w_vec: np.ndarray | None = None,
    trait_cor: np.ndarray | None = None,
    ld_matrix: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Compute exact O(M |S|) `(P_R, P_A, PP_coloc, snp_posteriors)` in log-space."""
    idx = list(subset_indices)
    m_traits = len(idx)
    if m_traits < 2:
        raise ValueError("Trait subset must have at least 2 traits.")

    lbf = lbf_full[:, idx]  # (M, m)
    n_snps = lbf.shape[0]

    has_overlap = (
        trait_cor is not None
        and betas is not None
        and ses is not None
        and w_vec is not None
        and not np.allclose(trait_cor[np.ix_(idx, idx)], np.eye(m_traits), atol=1e-12)
    )

    if not has_overlap:
        s_lbf = np.sum(lbf, axis=1)  # (M,)
        log_w_all = float(special.logsumexp(s_lbf))
        # Leave-one-trait-out log-BF sums: shape (M, m)
        lbf_minus_1 = s_lbf[:, None] - lbf
        log_w_minus_1 = special.logsumexp(lbf_minus_1, axis=0)  # (m,)
    else:
        cor_sub = trait_cor[np.ix_(idx, idx)]
        b_sub = betas[:, idx]
        s_sub = ses[:, idx]
        w_sub = w_vec[idx]
        s_lbf = _joint_multivariate_lbf_subset(b_sub, s_sub, w_sub, cor_sub)
        log_w_all = float(special.logsumexp(s_lbf))
        log_w_minus_1 = np.empty(m_traits, dtype=np.float64)
        if m_traits == 2:
            log_w_minus_1[0] = float(special.logsumexp(lbf[:, 1]))
            log_w_minus_1[1] = float(special.logsumexp(lbf[:, 0]))
        else:
            for q_pos in range(m_traits):
                rem = [r for r in range(m_traits) if r != q_pos]
                lbf_rem = _joint_multivariate_lbf_subset(
                    b_sub[:, rem], s_sub[:, rem], w_sub[rem], cor_sub[np.ix_(rem, rem)]
                )
                log_w_minus_1[q_pos] = float(special.logsumexp(lbf_rem))

    log_w_single = special.logsumexp(lbf, axis=0)  # (m,)
    log_l_all = float(log_pi[m_traits] + log_w_all)

    # Alignment alternative: (m - 1) traits share SNP j and peeled-off trait q has distinct SNP k != j
    log_d_terms = np.empty(m_traits, dtype=np.float64)
    if ld_matrix is None:
        for q_pos in range(m_traits):
            log_prod = float(log_w_minus_1[q_pos] + log_w_single[q_pos])
            if not has_overlap:
                log_diag = log_w_all
            else:
                # Diagonal j == k under independent split of (S \ {q}) and {q}
                if m_traits == 2:
                    log_diag = float(special.logsumexp(lbf[:, 0] + lbf[:, 1]))
                else:
                    rem = [r for r in range(m_traits) if r != q_pos]
                    lbf_rem = _joint_multivariate_lbf_subset(
                        betas[:, idx][:, rem],
                        ses[:, idx][:, rem],
                        w_vec[idx][rem],
                        trait_cor[np.ix_(idx, idx)][np.ix_(rem, rem)],
                    )
                    log_diag = float(special.logsumexp(lbf_rem + lbf[:, q_pos]))
            diff_ratio = max(float(np.exp(log_prod - log_diag)) - 1.0, 1e-300)
            log_d_terms[q_pos] = log_diag + np.log(diff_ratio)
    else:
        # LD-aware alignment: downweight nearly identical tag pairs (r^2 -> 1) or Weight distinct configurations (1 - r_jk^2)
        # Wait: when two distinct causal SNPs j != k exist in LD, (1 - r_jk^2 >= 0) avoids counting completely indistinguishable r^2=1 duplicates as separate causal loci while preserving distinct LD-linked peaks (0 < r^2 < 0.99)
        r2 = ld_matrix**2
        mask_distinct = 1.0 - np.eye(n_snps, dtype=np.float64)
        # Weight distinct causal pairs; exact match to standard coloc when LD is diagonal/unrestricted off-diagonal
        # If sample overlap is also present, account for shared z-score coupling across linked SNPs
        for q_pos in range(m_traits):
            if not has_overlap:
                lp_minus = s_lbf - lbf[:, q_pos]
            elif m_traits == 2:
                lp_minus = lbf[:, 1 - q_pos]
            else:
                rem = [r for r in range(m_traits) if r != q_pos]
                lp_minus = _joint_multivariate_lbf_subset(
                    betas[:, idx][:, rem],
                    ses[:, idx][:, rem],
                    w_vec[idx][rem],
                    trait_cor[np.ix_(idx, idx)][np.ix_(rem, rem)],
                )
            lp_q = lbf[:, q_pos]
            shift_a = float(np.max(lp_minus))
            shift_b = float(np.max(lp_q))
            va = np.exp(lp_minus - shift_a)
            vb = np.exp(lp_q - shift_b)
            # Exclude identical/perfectly-collinear r^2 > 0.9999 duplicates from distinct-causal-SNP hypothesis
            valid_offdiag = mask_distinct * (r2 < 0.9999).astype(np.float64)
            quad = float(va @ valid_offdiag @ vb)
            log_d_terms[q_pos] = shift_a + shift_b + np.log(max(quad, 1e-300))

    sym_factor = -np.log(2.0) if m_traits == 2 else 0.0
    if uniform_priors and settings == "R_default":
        if n_snps > 1:
            p_m_1 = prior_1 / (float(m_traits) * float(n_snps) * float(n_snps - 1))
            if m_traits == 2:
                p_m_1 *= 2.0
            log_prior_split = float(np.log(p_m_1))
        else:
            log_prior_split = float(log_pi[1])
    else:
        log_prior_split = (
            float(log_pi[1]) if uniform_priors else float(log_pi[m_traits - 1] + log_pi[1])
        )
    log_l_align_alt = log_prior_split + sym_factor + float(special.logsumexp(log_d_terms))

    log_l_reg_num = float(np.logaddexp(log_l_all, log_l_align_alt))
    p_align = float(np.exp(log_l_all - log_l_reg_num))

    # Regional non-shared hypotheses (0 traits H0, m-1 traits, and optional m-2 traits via Newton O(M m) identity)
    log_l_0 = 0.0
    log_l_m_minus_1 = float(log_pi[m_traits - 1] + special.logsumexp(log_w_minus_1))
    if settings == "R_default" and m_traits == 2:
        log_l_m_minus_1 += float(np.log(0.25))

    if reg_steps == 0:
        log_non_reg = log_l_0
    elif reg_steps == 1 or m_traits == 2:
        log_non_reg = float(np.logaddexp(log_l_0, log_l_m_minus_1))
    else:
        # Exact O(M m) 2-trait-dropped sum via Newton identity:
        # sum_{q1 < q2} exp(-l_{jq1} - l_{jq2}) = 0.5 * ((sum_q exp(-l_{jq}))^2 - sum_q exp(-2 l_{jq}))
        log_u1 = special.logsumexp(-lbf, axis=1)  # (M,)
        log_u2 = special.logsumexp(-2.0 * lbf, axis=1)  # (M,)
        ratio_sq = np.clip(np.exp(log_u2 - 2.0 * log_u1), 0.0, 1.0 - 1e-15)
        log_e2 = -np.log(2.0) + 2.0 * log_u1 + np.log(1.0 - ratio_sq)
        s_lbf_indep = np.sum(lbf, axis=1)
        log_w_minus_2 = float(special.logsumexp(s_lbf_indep + log_e2))
        log_l_m_minus_2 = float(log_pi[m_traits - 2] + log_w_minus_2)
        log_non_reg = float(special.logsumexp([log_l_0, log_l_m_minus_1, log_l_m_minus_2]))

    if settings == "R_default":
        log_reg_denom = float(np.logaddexp(log_l_all, log_non_reg))
        p_reg = float(np.exp(log_l_all - log_reg_denom))
    else:
        log_total = float(np.logaddexp(log_l_reg_num, log_non_reg))
        p_reg = float(np.exp(log_l_reg_num - log_total))
    pp_coloc = float(p_reg * p_align)

    snp_post = np.exp(s_lbf - log_w_all)
    snp_post /= np.sum(snp_post)
    return p_reg, p_align, pp_coloc, snp_post


def subset_coloc(
    betas: np.ndarray | pd.DataFrame,
    ses: np.ndarray | pd.DataFrame,
    *,
    trait_subset: Sequence[int | str] | None = None,
    trait_names: Sequence[str] | None = None,
    snp_ids: Sequence[str] | None = None,
    binary_outcomes: Sequence[bool | int] | np.ndarray | None = None,
    prior_w: float | Sequence[float] | np.ndarray | None = None,
    prior_1: float = 1e-4,
    prior_c: float = 0.02,
    prior_2: float | None = None,
    uniform_priors: bool = False,
    reg_steps: int | None = None,
    trait_cor: np.ndarray | None = None,
    ld_matrix: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> SubsetColocResult:
    """Evaluate exact O(M |S|) HyPrColoc regional, alignment, and joint posteriors for a fixed subset S."""
    mode_settings = _validate_settings(settings)
    rs = _resolve_reg_steps(reg_steps, mode_settings)
    b_arr, s_arr, t_names, s_ids = _extract_matrices(betas, ses, trait_names, snp_ids)
    m_snps, q_traits = b_arr.shape
    if q_traits < 2:
        raise ValueError("At least Q >= 2 traits are required.")

    pc = _resolve_prior_c(prior_1, prior_c, prior_2)
    w_vec = _resolve_prior_w(q_traits, binary_outcomes, prior_w)
    t_cor = _validate_corr_matrix(trait_cor, q_traits, "trait_cor", require_pd=True) if trait_cor is not None else None
    ld_mat = _validate_corr_matrix(ld_matrix, m_snps, "ld_matrix", require_pd=False) if ld_matrix is not None else None

    if trait_subset is None:
        sub_idx = list(range(q_traits))
    else:
        name_to_idx = {name: i for i, name in enumerate(t_names)}
        sub_idx = []
        for item in trait_subset:
            if isinstance(item, (int, np.integer)):
                idx_int = int(item)
                if not (0 <= idx_int < q_traits):
                    raise ValueError(f"Trait index {idx_int} out of bounds [0, {q_traits}).")
                sub_idx.append(idx_int)
            else:
                if str(item) not in name_to_idx:
                    raise ValueError(f"Unknown trait name '{item}'.")
                sub_idx.append(name_to_idx[str(item)])
        if len(set(sub_idx)) != len(sub_idx) or len(sub_idx) < 2:
            raise ValueError("trait_subset must contain >= 2 distinct traits.")

    lbf_mat = log_abf(b_arr, s_arr, prior_w=w_vec, settings=mode_settings)
    log_pi = _log_config_priors(
        len(sub_idx), prior_1, pc, uniform_priors, n_snps=m_snps, settings=mode_settings
    )
    p_reg, p_align, pp_coloc, snp_post = _compute_subset_posteriors(
        lbf_mat,
        sub_idx,
        log_pi,
        uniform_priors,
        prior_1=prior_1,
        reg_steps=rs,
        settings=mode_settings,
        betas=b_arr,
        ses=s_arr,
        w_vec=w_vec,
        trait_cor=t_cor,
        ld_matrix=ld_mat,
    )
    cand_idx = int(np.argmax(snp_post))
    return SubsetColocResult(
        traits=[t_names[i] for i in sub_idx],
        trait_indices=sub_idx,
        snp_ids=s_ids,
        regional_prob=p_reg,
        align_prob=p_align,
        posterior_prob=pp_coloc,
        candidate_snp=s_ids[cand_idx],
        candidate_snp_index=cand_idx,
        posterior_explained_by_snp=float(snp_post[cand_idx]),
        snp_posteriors=snp_post,
    )


def _extract_matrices(
    betas: np.ndarray | pd.DataFrame,
    ses: np.ndarray | pd.DataFrame,
    trait_names: Sequence[str] | None,
    snp_ids: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if isinstance(betas, pd.DataFrame):
        default_traits = [str(c) for c in betas.columns]
        default_snps = [str(i) for i in betas.index]
        b_arr = betas.to_numpy(dtype=np.float64)
    else:
        b_arr = np.asarray(betas, dtype=np.float64)
        if b_arr.ndim != 2:
            raise ValueError("betas must be a 2D array of shape (M, Q).")
        default_traits = [f"T{j + 1}" for j in range(b_arr.shape[1])]
        default_snps = [f"snp_{i + 1}" for i in range(b_arr.shape[0])]

    s_arr = ses.to_numpy(dtype=np.float64) if isinstance(ses, pd.DataFrame) else np.asarray(ses, dtype=np.float64)
    if b_arr.ndim != 2 or s_arr.ndim != 2 or b_arr.shape != s_arr.shape:
        raise ValueError("betas and ses must be 2D arrays of matching shape (M, Q).")
    m_snps, q_traits = b_arr.shape
    if m_snps < 1 or q_traits < 2:
        raise ValueError("betas and ses must have M >= 1 SNPs and Q >= 2 traits.")
    if not (np.all(np.isfinite(b_arr)) and np.all(np.isfinite(s_arr))):
        raise ValueError("betas and ses must be finite (no NaN or Inf).")
    if np.any(s_arr <= 0.0):
        raise ValueError("ses must be strictly positive.")

    t_names = default_traits if trait_names is None else [str(t) for t in trait_names]
    s_ids = default_snps if snp_ids is None else [str(s) for s in snp_ids]
    if len(t_names) != q_traits:
        raise ValueError("trait_names length must equal Q.")
    if len(s_ids) != m_snps:
        raise ValueError("snp_ids length must equal M.")
    return b_arr, s_arr, t_names, s_ids


def _resolve_prior_c(prior_1: float, prior_c: float, prior_2: float | None) -> float:
    if not (0.0 < prior_1 < 1.0):
        raise ValueError("prior_1 must lie strictly in (0, 1).")
    pc = (1.0 - prior_2) if prior_2 is not None else prior_c
    if not (0.0 < pc < 1.0):
        raise ValueError("prior_c (or 1 - prior_2) must lie strictly in (0, 1).")
    return float(pc)


def _normalize_bb_selection(bb_selection: str) -> Literal["regional", "align", "reg_only"]:
    key = bb_selection.strip().lower()
    if key in ("regional", "regional_selection", "reg"):
        return "regional"
    if key in ("align", "alignment", "align_selection"):
        return "align"
    if key in ("reg_only", "reg.only", "regional_only"):
        return "reg_only"
    raise ValueError(f"Unknown bb_selection '{bb_selection}'; choose from 'regional', 'align', 'reg_only'.")


def hyprcoloc(
    betas: np.ndarray | pd.DataFrame,
    ses: np.ndarray | pd.DataFrame,
    *,
    trait_names: Sequence[str] | None = None,
    snp_ids: Sequence[str] | None = None,
    binary_outcomes: Sequence[bool | int] | np.ndarray | None = None,
    prior_w: float | Sequence[float] | np.ndarray | None = None,
    prior_1: float = 1e-4,
    prior_c: float = 0.02,
    prior_2: float | None = None,
    uniform_priors: bool = False,
    reg_thresh: float | None = None,
    align_thresh: float | None = None,
    reg_steps: int | None = None,
    bb_alg: bool = True,
    bb_selection: str = "regional",
    trait_cor: np.ndarray | None = None,
    ld_matrix: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> HyPrColocResult:
    """Run HyPrColoc divisive branch-and-bound multi-trait colocalization (Foley et al., 2021)."""
    mode_settings = _validate_settings(settings)
    default_thresh = 0.7 if mode_settings == "R_default" else 0.5
    rt = default_thresh if reg_thresh is None else float(reg_thresh)
    at = default_thresh if align_thresh is None else float(align_thresh)
    rs = _resolve_reg_steps(reg_steps, mode_settings)
    if not (0.0 < rt < 1.0) or not (0.0 < at < 1.0):
        raise ValueError("reg_thresh and align_thresh must lie strictly in (0, 1).")
    mode = _normalize_bb_selection(bb_selection)
    pc = _resolve_prior_c(prior_1, prior_c, prior_2)

    b_arr, s_arr, t_names, s_ids = _extract_matrices(betas, ses, trait_names, snp_ids)
    m_snps, q_traits = b_arr.shape
    w_vec = _resolve_prior_w(q_traits, binary_outcomes, prior_w)
    t_cor = _validate_corr_matrix(trait_cor, q_traits, "trait_cor", require_pd=True) if trait_cor is not None else None
    ld_mat = _validate_corr_matrix(ld_matrix, m_snps, "ld_matrix", require_pd=False) if ld_matrix is not None else None

    lbf_mat = log_abf(b_arr, s_arr, prior_w=w_vec, settings=mode_settings)
    log_pi = _log_config_priors(
        q_traits, prior_1, pc, uniform_priors, n_snps=m_snps, settings=mode_settings
    )
    single_trait_evidence = special.logsumexp(lbf_mat, axis=0)

    def _eval(sub: list[int]) -> tuple[float, float, float, np.ndarray]:
        return _compute_subset_posteriors(
            lbf_mat,
            sub,
            log_pi,
            uniform_priors,
            prior_1=prior_1,
            reg_steps=rs,
            settings=mode_settings,
            betas=b_arr,
            ses=s_arr,
            w_vec=w_vec,
            trait_cor=t_cor,
            ld_matrix=ld_mat,
        )

    def _make_cluster(cid: int, sub: list[int], pr: float, pa: float, pp: float, sp: np.ndarray) -> HyPrColocCluster:
        c_idx = int(np.argmax(sp))
        order = np.argsort(-sp, kind="mergesort")
        cum = np.cumsum(sp[order])
        k95 = int(np.searchsorted(cum, 0.95, side="left")) + 1
        cs95 = [s_ids[int(i)] for i in order[: min(k95, m_snps)]]
        return HyPrColocCluster(
            cluster_id=cid,
            traits=[t_names[i] for i in sub],
            trait_indices=list(sub),
            posterior_prob=pr if mode == "reg_only" else pp,
            regional_prob=pr,
            align_prob=pa,
            candidate_snp=s_ids[c_idx],
            candidate_snp_index=c_idx,
            posterior_explained_by_snp=float(sp[c_idx]),
            snp_posteriors=sp,
            credible_set_95=cs95,
        )

    if not bb_alg:
        all_idx = list(range(q_traits))
        pr, pa, pp, sp = _eval(all_idx)
        cluster = _make_cluster(1, all_idx, pr, pa, pp, sp)
        return HyPrColocResult(
            n_snps=m_snps,
            n_traits=q_traits,
            snp_ids=s_ids,
            trait_names=t_names,
            clusters=[cluster],
            uncolocalized_traits=[],
            lbf_matrix=lbf_mat,
            prior_1=prior_1,
            prior_c=pc,
            reg_thresh=rt,
            align_thresh=at,
            bb_selection=mode,
            uniform_priors=uniform_priors,
            settings=mode_settings,
        )

    remaining = list(range(q_traits))
    clusters: list[HyPrColocCluster] = []
    uncolocalized: list[int] = []

    while len(remaining) >= 2:
        current = list(remaining)
        found_cluster = False

        while len(current) >= 2:
            pr, pa, pp, sp = _eval(current)
            passes_reg = pr > rt
            passes_align = True if mode == "reg_only" else (pa > at)

            if passes_reg and passes_align:
                clusters.append(_make_cluster(len(clusters) + 1, current, pr, pa, pp, sp))
                in_cluster = set(current)
                remaining = [t for t in remaining if t not in in_cluster]
                found_cluster = True
                break

            if len(current) == 2:
                break

            # Choose the single trait in `current` whose removal maximizes the branch criterion
            use_regional_score = (mode in ("regional", "reg_only")) and (not passes_reg)
            best_drop_pos = 0
            best_score = -np.inf
            for drop_pos in range(len(current)):
                cand_sub = current[:drop_pos] + current[drop_pos + 1 :]
                c_pr, _, c_pp, _ = _eval(cand_sub)
                score = c_pr if use_regional_score else c_pp
                if score > best_score:
                    best_score = score
                    best_drop_pos = drop_pos
            current.pop(best_drop_pos)

        if not found_cluster:
            # Final 2-trait candidate failed: drop the weaker single-trait signal from `remaining`
            t_a, t_b = current[0], current[1]
            drop_trait = t_a if single_trait_evidence[t_a] <= single_trait_evidence[t_b] else t_b
            uncolocalized.append(drop_trait)
            remaining = [t for t in remaining if t != drop_trait]

    uncolocalized.extend(remaining)
    uncolocalized_sorted = sorted(uncolocalized)

    return HyPrColocResult(
        n_snps=m_snps,
        n_traits=q_traits,
        snp_ids=s_ids,
        trait_names=t_names,
        clusters=clusters,
        uncolocalized_traits=[t_names[i] for i in uncolocalized_sorted],
        lbf_matrix=lbf_mat,
        prior_1=prior_1,
        prior_c=pc,
        reg_thresh=rt,
        align_thresh=at,
        bb_selection=mode,
        uniform_priors=uniform_priors,
        settings=mode_settings,
    )


def sensitivity_analysis(
    betas: np.ndarray | pd.DataFrame,
    ses: np.ndarray | pd.DataFrame,
    *,
    trait_names: Sequence[str] | None = None,
    snp_ids: Sequence[str] | None = None,
    binary_outcomes: Sequence[bool | int] | np.ndarray | None = None,
    prior_1: float = 1e-4,
    prior_c_grid: Sequence[float] = (0.05, 0.02, 0.01, 0.005),
    reg_thresh_grid: Sequence[float] = (0.6, 0.7, 0.8),
    align_thresh_grid: Sequence[float] = (0.6, 0.7, 0.8),
    uniform_priors: bool = False,
    reg_steps: int | None = None,
    bb_selection: str = "regional",
    trait_cor: np.ndarray | None = None,
    ld_matrix: np.ndarray | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> SensitivityResult:
    """Evaluate pairwise trait co-clustering similarity across grids of priors and thresholds."""
    mode_settings = _validate_settings(settings)
    b_arr, s_arr, t_names, s_ids = _extract_matrices(betas, ses, trait_names, snp_ids)
    q_traits = b_arr.shape[1]
    co_counts = np.zeros((q_traits, q_traits), dtype=np.float64)
    records = []
    n_configs = 0

    for pc in prior_c_grid:
        for rt in reg_thresh_grid:
            for at in align_thresh_grid:
                res = hyprcoloc(
                    b_arr,
                    s_arr,
                    trait_names=t_names,
                    snp_ids=s_ids,
                    binary_outcomes=binary_outcomes,
                    prior_1=prior_1,
                    prior_c=pc,
                    uniform_priors=uniform_priors,
                    reg_thresh=rt,
                    align_thresh=at,
                    reg_steps=reg_steps,
                    bb_selection=bb_selection,
                    trait_cor=trait_cor,
                    ld_matrix=ld_matrix,
                    settings=mode_settings,
                )
                n_configs += 1
                for cl in res.clusters:
                    idx = cl.trait_indices
                    for i in idx:
                        for j in idx:
                            if i != j:
                                co_counts[i, j] += 1.0
                records.append(
                    {
                        "prior_c": pc,
                        "reg_thresh": rt,
                        "align_thresh": at,
                        "n_clusters": len(res.clusters),
                        "clusters": "; ".join(",".join(c.traits) for c in res.clusters) or "None",
                    }
                )

    if n_configs == 0:
        raise ValueError("Sensitivity parameter grids must be non-empty.")
    sim = co_counts / float(n_configs)
    np.fill_diagonal(sim, 1.0)
    return SensitivityResult(
        trait_names=t_names,
        similarity_matrix=sim,
        n_configurations=n_configs,
        grid_records=pd.DataFrame(records),
    )
