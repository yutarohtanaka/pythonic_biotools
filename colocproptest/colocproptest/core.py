"""
End-to-end proportional colocalization pipeline (tester-pair profile-GLS + BH-FDR +
ACAT Cauchy omnibus + LD-PC omnibus test), OOP wrapper, and result container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .pairs import (
    SettingsMode,
    select_tester_pairs,
    test_pair_proportionality,
    validate_settings,
)
from .pc import proptest_pc


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment matching ``stats::p.adjust(..., method='BH')``."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=float)
    scaled = ranked * (float(n) / ranks)
    # Cumulative minimum from right to left
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def acat_combine(p_values: np.ndarray) -> float:
    """Aggregated Cauchy Association Test (Liu & Xie, JASA 2020) for correlated p-values."""
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return 1.0
    p_clip = np.clip(p, 1e-300, 1.0 - 1e-16)
    small = p_clip < 1e-15
    transformed = np.where(
        small,
        1.0 / (p_clip * np.pi),
        np.tan((0.5 - p_clip) * np.pi),
    )
    t_stat = float(np.mean(transformed))
    if t_stat > 1e15:
        return float(np.clip(1.0 / (t_stat * np.pi), 0.0, 1.0))
    return float(np.clip(0.5 - np.arctan(t_stat) / np.pi, 0.0, 1.0))


def _parse_trait_summary(
    trait: dict[str, Any] | pd.DataFrame | tuple[np.ndarray, np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Extract ``(beta, se, snp_ids)`` from flexible trait summary inputs."""
    snp_ids: list[str] | None = None
    if isinstance(trait, tuple) and len(trait) == 2:
        beta = np.asarray(trait[0], dtype=float)
        se = np.asarray(trait[1], dtype=float)
    elif isinstance(trait, pd.DataFrame):
        b_col = next((c for c in ("beta", "b", "effect") if c in trait.columns), None)
        if b_col is None:
            raise ValueError(f"{label} DataFrame must contain a 'beta' or 'b' column.")
        beta = trait[b_col].to_numpy(dtype=float)
        if "se" in trait.columns:
            se = trait["se"].to_numpy(dtype=float)
        elif "varbeta" in trait.columns:
            varb = trait["varbeta"].to_numpy(dtype=float)
            if np.any(varb <= 0.0):
                raise ValueError(f"{label} 'varbeta' must be strictly positive.")
            se = np.sqrt(varb)
        else:
            raise ValueError(f"{label} DataFrame must contain 'se' or 'varbeta'.")
        s_col = next((c for c in ("snp", "snp_id", "rsid", "id") if c in trait.columns), None)
        if s_col is not None:
            snp_ids = trait[s_col].astype(str).tolist()
    elif isinstance(trait, dict):
        b_key = next((k for k in ("beta", "b", "effect") if k in trait), None)
        if b_key is None:
            raise ValueError(f"{label} dict must contain 'beta' or 'b'.")
        beta = np.asarray(trait[b_key], dtype=float)
        if "se" in trait:
            se = np.asarray(trait["se"], dtype=float)
        elif "varbeta" in trait:
            varb = np.asarray(trait["varbeta"], dtype=float)
            if np.any(varb <= 0.0):
                raise ValueError(f"{label} 'varbeta' must be strictly positive.")
            se = np.sqrt(varb)
        else:
            raise ValueError(f"{label} dict must contain 'se' or 'varbeta'.")
        if "snp" in trait and trait["snp"] is not None:
            snp_ids = [str(s) for s in trait["snp"]]
    else:
        raise TypeError(
            f"{label} must be a dict, DataFrame, or (beta, se) tuple, got {type(trait)}."
        )

    if beta.ndim != 1 or se.ndim != 1 or len(beta) != len(se):
        raise ValueError(f"{label} beta and se must be 1D arrays of equal length.")
    if len(beta) < 2:
        raise ValueError(f"{label} must contain at least 2 SNPs.")
    if np.any(~np.isfinite(beta)) or np.any(~np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError(f"{label} beta/se must be finite with se > 0.")
    return beta, se, snp_ids


@dataclass
class ColocPropTestResult:
    """Result container for frequentist proportional colocalization testing."""

    eta_hat: float
    eta_ci: tuple[float, float]
    min_p: float
    min_fdr: float
    p_acat: float
    p_pc_omnibus: float
    reject_proportionality: bool
    evidence_for_coloc: bool
    pairs_df: pd.DataFrame
    n_pc: int
    max_abs_z1: float
    max_abs_z2: float
    alpha: float
    settings: SettingsMode

    def summary(self) -> str:
        """Return a human-readable summary of the proportional colocalization test."""
        ci_lo, ci_hi = self.eta_ci
        verdict = (
            "REJECT proportionality (distinct causal architectures / non-colocalized)"
            if self.reject_proportionality
            else (
                "COMPATIBLE with colocalization (proportionality retained & strong marginal signals)"
                if self.evidence_for_coloc
                else "INCONCLUSIVE (proportionality not rejected, insufficient marginal signal)"
            )
        )
        lines = [
            f"ColocPropTestResult(settings={self.settings!r}, alpha={self.alpha})",
            f"  Verdict: {verdict}",
            f"  Proportionality slope: eta_hat={self.eta_hat:.4f}, "
            f"95% profile CI=[{ci_lo:.4f}, {ci_hi:.4f}]",
            f"  Pairwise tests ({len(self.pairs_df)} pairs): min_p={self.min_p:.3e}, "
            f"min_fdr={self.min_fdr:.3e}, p_acat={self.p_acat:.3e}",
            f"  LD-PC omnibus (K={self.n_pc} PCs): p_pc_omnibus={self.p_pc_omnibus:.3e}",
            f"  Max |Z|: trait1={self.max_abs_z1:.2f}, trait2={self.max_abs_z2:.2f}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of ``pairs_df`` sorted by ascending p-value."""
        return self.pairs_df.copy()


def run_coloc_prop_test(
    trait1: dict[str, Any] | pd.DataFrame | tuple[np.ndarray, np.ndarray],
    trait2: dict[str, Any] | pd.DataFrame | tuple[np.ndarray, np.ndarray],
    LD: np.ndarray | pd.DataFrame,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> ColocPropTestResult:
    """Run frequentist proportional colocalization test across a genomic locus.

    Under ``H_0`` of colocalization (shared single or proportional multi-causal
    effects), ``beta_2 = eta * beta_1`` holds across all local variants regardless
    of differential per-SNP sample sizes.

    Parameters
    ----------
    trait1, trait2 : dict, DataFrame, or (beta, se) tuple
        Marginal GWAS summary statistics for Trait 1 and Trait 2. Supports ``coloc``
        dataset dictionaries containing ``beta`` and ``varbeta`` (or ``se``).
    LD : ndarray or DataFrame of shape (M, M)
        Local SNP-SNP linkage disequilibrium correlation matrix.
    alpha : float, default 0.05
        Significance threshold for rejecting ``H_0`` of proportionality.
    settings : {"default", "R_default"}, default "default"
        Pipeline settings mode:
        - ``"default"``: selects non-collinear tester pairs at ``max_r2=0.85``,
          combines exact angular 2x2 profile-GLS p-values via ACAT + BH-FDR, runs
          LD-PC omnibus at ``var_explained=0.95``, and requires ``max |z1| >= 3.0``
          and ``max |z2| >= 3.0`` for ``evidence_for_coloc=True``.
        - ``"R_default"`` (``colocPropTest::run_proptest`` CRAN v0.9.3 parity):
          selects up to ``n_top=15`` marginal Z-score variants (``r^2 < 0.99``),
          scores all pairs with BH-FDR (rejecting when ``min_fdr < alpha``), and
          evaluates the PC-projected omnibus test at ``thr=0.80`` cumulative LD variance.
    """
    validate_settings(settings)
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    b1, se1, snps1 = _parse_trait_summary(trait1, "trait1")
    b2, se2, snps2 = _parse_trait_summary(trait2, "trait2")

    if len(b1) != len(b2):
        raise ValueError(
            f"trait1 length ({len(b1)}) and trait2 length ({len(b2)}) must match."
        )
    m = len(b1)
    snp_ids = snps1 if snps1 is not None else (snps2 if snps2 is not None else [f"snp_{i}" for i in range(m)])

    if isinstance(LD, pd.DataFrame):
        r_mat = LD.to_numpy(dtype=float)
    else:
        r_mat = np.asarray(LD, dtype=float)
    if r_mat.shape != (m, m):
        raise ValueError(f"LD must have shape ({m}, {m}), got {r_mat.shape}.")
    r_mat = 0.5 * (r_mat + r_mat.T)
    np.fill_diagonal(r_mat, 1.0)

    if settings == "default":
        pairs = select_tester_pairs(
            b1, se1, b2, se2, r_mat, max_testers=15, max_r2=0.85, min_abs_z=2.0, settings=settings
        )
        pc_thr = 0.95
        min_signal_z = 3.0
    else:
        pairs = select_tester_pairs(
            b1, se1, b2, se2, r_mat, max_testers=15, max_r2=0.99, min_abs_z=0.0, settings=settings
        )
        pc_thr = 0.80
        min_signal_z = 1.96

    if not pairs:
        # Ultra-collinear fallback: test pair (0, argmin r^2 with 0)
        r2_0 = r_mat[0] ** 2
        j_best = int(np.argmin(np.where(np.arange(m) == 0, np.inf, r2_0)))
        pairs = [(0, j_best)]

    rows: list[dict[str, Any]] = []
    for i_idx, j_idx in pairs:
        idx_pair = [i_idx, j_idx]
        sub_ld = r_mat[np.ix_(idx_pair, idx_pair)]
        res_pair = test_pair_proportionality(
            b1[idx_pair],
            se1[idx_pair],
            b2[idx_pair],
            se2[idx_pair],
            sub_ld,
            settings=settings,
        )
        rows.append(
            {
                "idx_1": i_idx,
                "idx_2": j_idx,
                "snp_1": snp_ids[i_idx],
                "snp_2": snp_ids[j_idx],
                "r": float(r_mat[i_idx, j_idx]),
                "stat": res_pair.stat,
                "p_value": res_pair.p_value,
                "eta_hat": res_pair.eta_hat,
            }
        )

    pairs_df = pd.DataFrame(rows)
    p_vals = pairs_df["p_value"].to_numpy(dtype=float)
    fdr_vals = bh_fdr(p_vals)
    pairs_df["fdr"] = fdr_vals
    pairs_df = pairs_df.sort_values("p_value", ascending=True, kind="mergesort").reset_index(
        drop=True
    )

    min_p = float(np.min(p_vals))
    min_fdr = float(np.min(fdr_vals))
    p_acat = acat_combine(p_vals)

    pc_res = proptest_pc(
        b1, se1, b2, se2, r_mat, var_explained=pc_thr, alpha=alpha, settings=settings
    )

    max_abs_z1 = float(np.max(np.abs(b1 / se1)))
    max_abs_z2 = float(np.max(np.abs(b2 / se2)))

    if settings == "default":
        reject_prop = bool((min_fdr < alpha) or (p_acat < alpha))
    else:
        reject_prop = bool(min_fdr < alpha)

    evidence_coloc = bool(
        (not reject_prop) and (max_abs_z1 >= min_signal_z) and (max_abs_z2 >= min_signal_z)
    )

    return ColocPropTestResult(
        eta_hat=pc_res.eta_hat,
        eta_ci=pc_res.eta_ci,
        min_p=min_p,
        min_fdr=min_fdr,
        p_acat=p_acat,
        p_pc_omnibus=pc_res.p_value,
        reject_proportionality=reject_prop,
        evidence_for_coloc=evidence_coloc,
        pairs_df=pairs_df,
        n_pc=pc_res.n_pc,
        max_abs_z1=max_abs_z1,
        max_abs_z2=max_abs_z2,
        alpha=float(alpha),
        settings=settings,
    )


class ColocPropTest:
    """Object-oriented estimator interface for ``colocPropTest``."""

    def __init__(
        self,
        alpha: float = 0.05,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = float(alpha)
        self.settings = settings
        self.result_: ColocPropTestResult | None = None

    def fit(
        self,
        trait1: dict[str, Any] | pd.DataFrame | tuple[np.ndarray, np.ndarray],
        trait2: dict[str, Any] | pd.DataFrame | tuple[np.ndarray, np.ndarray],
        LD: np.ndarray | pd.DataFrame,
    ) -> ColocPropTestResult:
        """Run proportional colocalization test on ``(trait1, trait2, LD)``."""
        self.result_ = run_coloc_prop_test(
            trait1=trait1,
            trait2=trait2,
            LD=LD,
            alpha=self.alpha,
            settings=self.settings,
        )
        return self.result_
