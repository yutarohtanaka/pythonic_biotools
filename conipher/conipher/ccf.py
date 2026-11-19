"""
Multi-region PhyloCCF and standard-error estimation (`compute_phylo_ccf`)
following Grigoriadis et al., Nature Protocols 2023 (10.1038/s41596-022-00793-z).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy import special, stats

from .io import SettingsMode, normalize_purity_map, validate_settings


def compute_phylo_ccf(
    mutations_df: pd.DataFrame,
    purity_by_region: Mapping[str, float] | pd.Series | pd.DataFrame,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Compute multi-region phylogenetic Cancer Cell Fractions (`phylo_ccf`) and
    standard errors (`ccf_se`) from allele counts, local allele-specific copy
    numbers, and regional tumor purities across `R >= 2` regions.

    Parameters
    ----------
    mutations_df : pd.DataFrame
        Long-format table across `(mutation_id, region)` containing `alt_count`,
        `depth` (or `ref_count` + `alt_count`), `major_cn`, `minor_cn` (and
        optional `mutation_id` or `chrom` + `pos`).
    purity_by_region : Mapping[str, float] | pd.Series | pd.DataFrame
        Tumor purity `rho_r in (0, 1]` for each sampled tumor region (`R >= 2`).
    settings : {"default", "R_default"}, default="default"
        `"default"` uses Beta-Binomial (`rho=0.01`) multiplicity posteriors and
        Jeffreys Beta posterior standard errors; `"R_default"` uses Binomial
        maximum-likelihood integer multiplicity selection matching CONIPHER.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with `mutation_id`, `region`, `alt_count`, `depth`,
        `vaf`, `purity`, `major_cn`, `minor_cn`, `multiplicity`, `mut_cpn`,
        `phylo_ccf`, `ccf_capped`, `ccf_se`, `ccf_lower`, and `ccf_upper`.
    """
    validate_settings(settings)
    if not isinstance(mutations_df, pd.DataFrame) or mutations_df.empty:
        raise ValueError("mutations_df must be a non-empty pandas DataFrame.")

    purity_map = normalize_purity_map(purity_by_region)
    df = mutations_df.copy().reset_index(drop=True)

    if "region" not in df.columns:
        for cand in ("sample", "region_id", "sample_id"):
            if cand in df.columns:
                df = df.rename(columns={cand: "region"})
                break
    if "region" not in df.columns:
        raise ValueError("mutations_df must contain a 'region' column.")
    df["region"] = df["region"].astype(str)

    unknown_regs = set(df["region"].unique()) - set(purity_map.keys())
    if unknown_regs:
        raise ValueError(
            f"mutations_df contains regions absent from purity_by_region: {sorted(unknown_regs)}."
        )

    if "mutation_id" not in df.columns:
        if "chrom" in df.columns and "pos" in df.columns:
            if "alt" in df.columns and not pd.api.types.is_numeric_dtype(df["alt"]):
                df["mutation_id"] = (
                    df["chrom"].astype(str)
                    + ":"
                    + df["pos"].astype(str)
                    + ":"
                    + df["alt"].astype(str)
                )
            else:
                df["mutation_id"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)
        else:
            raise ValueError(
                "mutations_df must contain either 'mutation_id' or ('chrom', 'pos')."
            )
    df["mutation_id"] = df["mutation_id"].astype(str)

    if "alt_count" not in df.columns:
        for cand in ("var_counts", "t_alt_count", "alt_reads"):
            if cand in df.columns:
                df["alt_count"] = df[cand]
                break

    if "depth" not in df.columns:
        for cand in ("total_depth", "dp", "t_depth"):
            if cand in df.columns:
                df["depth"] = df[cand]
                break
        if "depth" not in df.columns and "ref_count" in df.columns and "alt_count" in df.columns:
            df["depth"] = df["alt_count"] + df["ref_count"]

    if "major_cn" not in df.columns and "nMajor" in df.columns:
        df["major_cn"] = df["nMajor"]
    if "minor_cn" not in df.columns and "nMinor" in df.columns:
        df["minor_cn"] = df["nMinor"]

    for req in ("alt_count", "depth", "major_cn", "minor_cn"):
        if req not in df.columns:
            raise ValueError(f"mutations_df is missing required column {req!r}.")

    alt = np.round(df["alt_count"].astype(np.float64).to_numpy()).astype(np.int64)
    dp = np.round(df["depth"].astype(np.float64).to_numpy()).astype(np.int64)
    maj = np.round(df["major_cn"].astype(np.float64).to_numpy()).astype(np.int64)
    mn = np.round(df["minor_cn"].astype(np.float64).to_numpy()).astype(np.int64)
    norm_cn = (
        df["normal_cn"].astype(np.float64).to_numpy()
        if "normal_cn" in df.columns
        else np.full(len(df), 2.0, dtype=np.float64)
    )

    if np.any(alt < 0) or np.any(dp <= 0) or np.any(alt > dp):
        raise ValueError("Require 0 <= alt_count <= depth and depth > 0 in mutations_df.")
    if np.any(maj < 0) or np.any(mn < 0):
        raise ValueError("major_cn and minor_cn must be non-negative.")

    swap = maj < mn
    if np.any(swap):
        maj_new = np.maximum(maj, mn)
        mn_new = np.minimum(maj, mn)
        maj, mn = maj_new, mn_new

    purities = np.array([purity_map[r] for r in df["region"]], dtype=np.float64)
    vaf = alt.astype(np.float64) / dp.astype(np.float64)

    c_tot = (maj + mn).astype(np.float64)
    denom_dna = purities * c_tot + norm_cn * (1.0 - purities)
    cn_scale = denom_dna / purities
    mut_cpn = vaf * cn_scale

    # Estimate somatic multiplicity m in {1, ..., max(1, major_cn)}
    mult_hat = np.ones(len(df), dtype=np.int64)
    for i in range(len(df)):
        m_max = int(maj[i])
        if m_max <= 1 or alt[i] == 0 or mut_cpn[i] <= 1.0:
            mult_hat[i] = 1
            continue

        m_grid = np.arange(1, m_max + 1, dtype=np.float64)
        f_grid = np.clip((purities[i] * m_grid) / denom_dna[i], 1e-6, 1.0 - 1e-6)

        if settings == "default":
            # Beta-Binomial (rho = 0.01) log-likelihood across clonal multiplicities
            conc = 99.0
            a_bb = f_grid * conc
            b_bb = (1.0 - f_grid) * conc
            ll = special.betaln(alt[i] + a_bb, (dp[i] - alt[i]) + b_bb) - special.betaln(
                a_bb, b_bb
            )
        else:
            ll = stats.binom.logpmf(alt[i], dp[i], f_grid)

        mult_hat[i] = int(m_grid[int(np.argmax(ll))])

    # Raw PhyloCCF (kept gently bounded at 1.5 so clonal clusters have unbiased mean ~ 1.0)
    phylo_ccf = np.clip(mut_cpn / mult_hat.astype(np.float64), 0.0, 1.5)
    ccf_capped = np.clip(phylo_ccf, 0.0, 1.0)

    # Jeffreys Beta(alt + 0.5, ref + 0.5) posterior SD & 95% credible bounds scaled to CCF
    a_post = alt.astype(np.float64) + 0.5
    b_post = (dp - alt).astype(np.float64) + 0.5
    p_mean = a_post / (a_post + b_post)
    vaf_sd = np.sqrt((p_mean * (1.0 - p_mean)) / (a_post + b_post + 1.0))
    ccf_scale = cn_scale / mult_hat.astype(np.float64)
    ccf_se = np.maximum(vaf_sd * ccf_scale, 0.005)

    vaf_lo = stats.beta.ppf(0.025, a_post, b_post)
    vaf_hi = stats.beta.ppf(0.975, a_post, b_post)
    ccf_lo = np.where(alt == 0, 0.0, np.clip(vaf_lo * ccf_scale, 0.0, 1.5))
    ccf_hi = np.clip(vaf_hi * ccf_scale, 0.0, 1.5)

    out = df.copy()
    out["purity"] = purities
    out["vaf"] = vaf
    out["multiplicity"] = mult_hat
    out["mut_cpn"] = mut_cpn
    out["phylo_ccf"] = phylo_ccf
    out["ccf_capped"] = ccf_capped
    out["ccf_se"] = ccf_se
    out["ccf_lower"] = ccf_lo
    out["ccf_upper"] = ccf_hi
    return out
