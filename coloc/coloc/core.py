"""
Log-domain Wakefield Approximate Bayes Factor genetic colocalization (`coloc::coloc.abf`).

Reference:
  Giambartolomei et al. (2014), "Bayesian Test for Colocalisation between Pairs of
  Genetic Association Studies Using Summary Statistics", PLOS Genetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import warnings

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
    prior_w1: float = 0.15
    prior_w2: float = 0.15
    p1: float = 1e-4
    p2: float = 1e-4
    p12: float = 1e-5
    dataset_type1: str = "quant"
    dataset_type2: str = "quant"
    settings: str = "default"

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


def _log_sum_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    """Compute log(sum_{i != j} exp(a[i] + b[j])) without float64 cancellation or overflow."""
    n = a.size
    if n <= 1:
        return -np.inf
    sum_shared = float(special.logsumexp(a + b))
    log_prod = float(special.logsumexp(a) + special.logsumexp(b))
    diff = log_prod - sum_shared
    if diff >= 40.0:
        return log_prod + float(np.log1p(-np.exp(-diff)))
    if diff >= 0.1:
        return sum_shared + float(np.log(np.expm1(diff)))

    k = int(np.argmax(a + b))
    mask = np.ones(n, dtype=bool)
    mask[k] = False
    a_rest = a[mask]
    b_rest = b[mask]
    la = float(special.logsumexp(a_rest))
    lb = float(special.logsumexp(b_rest))
    term12 = float(special.logsumexp([a[k] + lb, b[k] + la]))
    if n == 2 or (la + lb <= term12 - 45.0):
        return term12
    term3 = _log_sum_offdiag(a_rest, b_rest)
    return float(special.logsumexp([term12, term3]))


def _validate_settings(settings: str) -> Literal["default", "R_default"]:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")
    return settings  # type: ignore[return-value]


def _validate_dataset_type(dtype: str, param_name: str) -> Literal["quant", "cc"]:
    if dtype not in ("quant", "cc"):
        raise ValueError(f"Unknown {param_name} {dtype!r}; expected 'quant' or 'cc'.")
    return dtype  # type: ignore[return-value]


def _adjust_prior(p: float, n_snps: int, suffix: str) -> float:
    """Replicate `chr1swallace/coloc` `claudia.R:173-182` `adjust_prior`."""
    if n_snps * p >= 1.0:
        warnings.warn(
            f"p{suffix} * nsnps >= 1, setting p{suffix}=1/(nsnps + 1)",
            UserWarning,
            stacklevel=3,
        )
        return 1.0 / float(n_snps + 1)
    return float(p)


def coloc_abf(
    beta1: Sequence[float] | np.ndarray,
    se1: Sequence[float] | np.ndarray,
    beta2: Sequence[float] | np.ndarray,
    se2: Sequence[float] | np.ndarray,
    snp_ids: Sequence[str] | None = None,
    prior_w1: float | None = None,
    prior_w2: float | None = None,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
    sd_y1: float | None = None,
    sd_y2: float | None = None,
    locus_name: str = "locus",
    trait1_name: str = "trait1",
    trait2_name: str = "trait2",
    *,
    prior_w: float | None = None,
    dataset_type1: Literal["quant", "cc"] = "quant",
    dataset_type2: Literal["quant", "cc"] = "quant",
    settings: Literal["default", "R_default"] = "default",
) -> ColocABFResult:
    """Compute Wakefield Approximate Bayes Factor colocalization entirely in log-space.

    Default priors (`0.15` for `dataset_type="quant"`, `0.20` for `dataset_type="cc"`;
    `p1=1e-4, p2=1e-4, p12=1e-5`) match R `coloc::coloc.abf` v5 defaults.
    When `sd_y1` or `sd_y2` is provided, `prior_w1` / `prior_w2` are scaled by the trait SD.
    """
    mode_settings = _validate_settings(settings)
    dt1 = _validate_dataset_type(dataset_type1, "dataset_type1")
    dt2 = _validate_dataset_type(dataset_type2, "dataset_type2")

    if prior_w is not None and not (np.isfinite(prior_w) and prior_w > 0.0):
        raise ValueError("prior_w must be finite and strictly positive when provided.")

    resolved_w1 = (
        float(prior_w1)
        if prior_w1 is not None
        else (float(prior_w) if prior_w is not None else (0.20 if dt1 == "cc" else 0.15))
    )
    resolved_w2 = (
        float(prior_w2)
        if prior_w2 is not None
        else (float(prior_w) if prior_w is not None else (0.20 if dt2 == "cc" else 0.15))
    )

    if not (np.isfinite(resolved_w1) and resolved_w1 > 0.0 and np.isfinite(resolved_w2) and resolved_w2 > 0.0):
        raise ValueError("prior_w1 and prior_w2 must be finite and strictly positive.")
    if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(p12) and p1 > 0.0 and p2 > 0.0 and p12 > 0.0):
        raise ValueError("Prior probabilities p1, p2, p12 must be finite and strictly positive.")
    if mode_settings == "default" and (p1 + p2 + p12 >= 1.0):
        raise ValueError("Sum of prior probabilities p1 + p2 + p12 must be strictly less than 1.")
    if sd_y1 is not None and not (np.isfinite(sd_y1) and sd_y1 > 0.0):
        raise ValueError("sd_y1 must be finite and strictly positive when provided.")
    if sd_y2 is not None and not (np.isfinite(sd_y2) and sd_y2 > 0.0):
        raise ValueError("sd_y2 must be finite and strictly positive when provided.")

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

    p1_eff = float(p1)
    p2_eff = float(p2)
    p12_eff = float(p12)
    if mode_settings == "R_default":
        p1_eff = _adjust_prior(p1_eff, m, "1")
        p2_eff = _adjust_prior(p2_eff, m, "2")
        p12_eff = _adjust_prior(p12_eff, m, "12")
        if p1_eff + p2_eff + p12_eff >= 1.0:
            warnings.warn(
                "p1 + p2 + p12 >= 1; shrinking priors so null prior p0 > 0.",
                UserWarning,
                stacklevel=2,
            )
            scale = 1.0 / ((p1_eff + p2_eff + p12_eff) * float(m + 1))
            p1_eff *= scale
            p2_eff *= scale
            p12_eff *= scale

    w1 = resolved_w1 * (1.0 if sd_y1 is None else float(sd_y1))
    w2 = resolved_w2 * (1.0 if sd_y2 is None else float(sd_y2))

    v1 = s1**2
    v2 = s2**2
    w1_sq = w1**2
    w2_sq = w2**2

    r1 = w1_sq / (v1 + w1_sq)
    r2 = w2_sq / (v2 + w2_sq)
    z1_sq = (b1**2) / v1
    z2_sq = (b2**2) / v2

    lbf1 = -0.5 * np.log1p(w1_sq / v1) + 0.5 * r1 * z1_sq
    lbf2 = -0.5 * np.log1p(w2_sq / v2) + 0.5 * r2 * z2_sq

    log_h0 = 0.0
    log_h1 = np.log(p1_eff) + float(special.logsumexp(lbf1))
    log_h2 = np.log(p2_eff) + float(special.logsumexp(lbf2))

    lbf_shared = lbf1 + lbf2
    sum_lbf_shared = float(special.logsumexp(lbf_shared))
    log_h4 = np.log(p12_eff) + sum_lbf_shared

    if m == 1:
        log_h3 = -np.inf
    else:
        log_sum_distinct = _log_sum_offdiag(lbf1, lbf2)
        log_h3 = np.log(p1_eff) + np.log(p2_eff) + log_sum_distinct

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
        prior_w1=w1,
        prior_w2=w2,
        p1=p1_eff,
        p2=p2_eff,
        p12=p12_eff,
        dataset_type1=dt1,
        dataset_type2=dt2,
        settings=mode_settings,
    )
