"""
Log-domain Wakefield Approximate Bayes Factor genetic colocalization (`coloc::coloc.abf`).

Reference:
  Giambartolomei et al. (2014), "Bayesian Test for Colocalisation between Pairs of
  Genetic Association Studies Using Summary Statistics", PLOS Genetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import special


@dataclass(frozen=True)
class ColocABFResult:
    """Posterior summary and per-SNP H4 probabilities from `coloc_abf`."""

    locus_name: str
    trait1_name: str
    trait2_name: str
    n_snps: int
    pp_h0: float
    pp_h1: float
    pp_h2: float
    pp_h3: float
    pp_h4: float
    lead_snp: str
    snp_ids: list[str]
    lbf_trait1: np.ndarray
    lbf_trait2: np.ndarray
    snp_pp_h4: np.ndarray

    def summary_dict(self) -> dict[str, float | int | str]:
        return {
            "locus_name": self.locus_name,
            "trait1": self.trait1_name,
            "trait2": self.trait2_name,
            "n_snps": self.n_snps,
            "PP.H0": self.pp_h0,
            "PP.H1": self.pp_h1,
            "PP.H2": self.pp_h2,
            "PP.H3": self.pp_h3,
            "PP.H4": self.pp_h4,
            "lead_snp": self.lead_snp,
        }

    def snp_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "snp": self.snp_ids,
                "lABF_trait1": self.lbf_trait1,
                "lABF_trait2": self.lbf_trait2,
                "lABF_shared": self.lbf_trait1 + self.lbf_trait2,
                "SNP.PP.H4": self.snp_pp_h4,
            }
        ).sort_values("SNP.PP.H4", ascending=False, kind="mergesort").reset_index(drop=True)

    def credible_set(self, coverage: float = 0.95) -> list[str]:
        """Return the smallest set of SNPs whose cumulative `SNP.PP.H4` reaches `coverage`."""
        if not (0.0 < coverage <= 1.0):
            raise ValueError("coverage must be in (0, 1].")
        order = np.argsort(-self.snp_pp_h4, kind="mergesort")
        cum = np.cumsum(self.snp_pp_h4[order])
        cutoff = int(np.searchsorted(cum, coverage, side="left")) + 1
        return [self.snp_ids[i] for i in order[: min(cutoff, self.n_snps)]]


def coloc_abf(
    beta1: Sequence[float] | np.ndarray,
    se1: Sequence[float] | np.ndarray,
    beta2: Sequence[float] | np.ndarray,
    se2: Sequence[float] | np.ndarray,
    snp_ids: Sequence[str] | None = None,
    prior_w1: float = 0.15,
    prior_w2: float = 0.15,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
    locus_name: str = "locus",
    trait1_name: str = "trait1",
    trait2_name: str = "trait2",
) -> ColocABFResult:
    """Compute Wakefield Approximate Bayes Factor colocalization entirely in log-space.

    Default priors (`prior_w1=0.15` for quantitative traits, `0.20` recommended for case-control
    log-odds; `p1=1e-4, p2=1e-4, p12=1e-5`) match R `coloc::coloc.abf` v5 defaults.
    """
    b1 = np.asarray(beta1, dtype=np.float64).ravel()
    s1 = np.asarray(se1, dtype=np.float64).ravel()
    b2 = np.asarray(beta2, dtype=np.float64).ravel()
    s2 = np.asarray(se2, dtype=np.float64).ravel()

    m = b1.size
    if m == 0 or s1.size != m or b2.size != m or s2.size != m:
        raise ValueError("beta1, se1, beta2, se2 must have identical non-zero length.")
    if np.any(s1 <= 0.0) or np.any(s2 <= 0.0):
        raise ValueError("Standard errors se1 and se2 must be strictly positive.")
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(s1)) and np.all(np.isfinite(b2)) and np.all(np.isfinite(s2))):
        raise ValueError("Non-finite summary statistics passed to coloc_abf.")

    ids = [f"snp_{i}" for i in range(m)] if snp_ids is None else [str(x) for x in snp_ids]
    if len(ids) != m:
        raise ValueError("snp_ids length must match beta1.")

    v1 = s1**2
    v2 = s2**2
    w1_sq = prior_w1**2
    w2_sq = prior_w2**2

    r1 = w1_sq / (v1 + w1_sq)
    r2 = w2_sq / (v2 + w2_sq)
    z1_sq = (b1**2) / v1
    z2_sq = (b2**2) / v2

    lbf1 = 0.5 * np.log(1.0 - r1) + 0.5 * r1 * z1_sq
    lbf2 = 0.5 * np.log(1.0 - r2) + 0.5 * r2 * z2_sq

    log_h0 = 0.0
    log_h1 = np.log(p1) + float(special.logsumexp(lbf1))
    log_h2 = np.log(p2) + float(special.logsumexp(lbf2))

    lbf_shared = lbf1 + lbf2
    sum_lbf_shared = float(special.logsumexp(lbf_shared))
    log_h4 = np.log(p12) + sum_lbf_shared

    log_prod = float(special.logsumexp(lbf1) + special.logsumexp(lbf2))
    diff_exp = max(float(np.exp(log_prod - sum_lbf_shared)) - 1.0, 1e-300)
    log_sum_distinct = sum_lbf_shared + np.log(diff_exp)
    log_h3 = np.log(p1) + np.log(p2) + log_sum_distinct

    log_hyps = np.array([log_h0, log_h1, log_h2, log_h3, log_h4], dtype=np.float64)
    log_denom = float(special.logsumexp(log_hyps))
    posteriors = np.exp(log_hyps - log_denom)

    snp_pp_h4 = np.exp(lbf_shared - sum_lbf_shared)
    lead_idx = int(np.argmax(lbf_shared))

    return ColocABFResult(
        locus_name=locus_name,
        trait1_name=trait1_name,
        trait2_name=trait2_name,
        n_snps=m,
        pp_h0=float(posteriors[0]),
        pp_h1=float(posteriors[1]),
        pp_h2=float(posteriors[2]),
        pp_h3=float(posteriors[3]),
        pp_h4=float(posteriors[4]),
        lead_snp=ids[lead_idx],
        snp_ids=ids,
        lbf_trait1=lbf1,
        lbf_trait2=lbf2,
        snp_pp_h4=snp_pp_h4,
    )
