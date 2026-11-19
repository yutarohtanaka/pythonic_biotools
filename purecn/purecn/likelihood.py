"""
Joint 2D (purity, ploidy) copy-number and Beta-Binomial allelic likelihood solver
for PureCN (`grid_search_purity_ploidy`, `PureCNSolution`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special

from .config import PureCNConfig, SettingsMode, get_purecn_config, validate_settings
from .coverage import CoverageNormalizationResult


@dataclass(frozen=True)
class PureCNSolution:
    """Ranked (purity, ploidy) local optimum from `grid_search_purity_ploidy`."""

    purity: float
    ploidy: float
    log_likelihood: float
    cn_log_likelihood: float
    snv_log_likelihood: float
    subclonal_fraction: float
    rank: int
    segments: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        return self.segments.copy()

    def summary(self) -> str:
        return (
            f"PureCNSolution(rank={self.rank}, purity={self.purity:.3f}, "
            f"ploidy={self.ploidy:.3f}, logL={self.log_likelihood:.2f}, "
            f"subclonal={self.subclonal_fraction:.3f})"
        )


def beta_binomial_logpmf(
    k: np.ndarray | float,
    n: np.ndarray | float,
    p: np.ndarray | float,
    rho_overdisp: float = 0.005,
) -> np.ndarray:
    """
    Exact vectorized Beta-Binomial log-PMF parameterized by mean `p` in (0, 1)
    and intra-class correlation / overdispersion `rho_overdisp` in [0, 1).
    """
    k_arr = np.asarray(k, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    log_comb = (
        special.gammaln(n_arr + 1.0)
        - special.gammaln(k_arr + 1.0)
        - special.gammaln(n_arr - k_arr + 1.0)
    )
    if rho_overdisp <= 1e-8:
        return log_comb + k_arr * np.log(p_arr) + (n_arr - k_arr) * np.log(1.0 - p_arr)

    rho = min(max(float(rho_overdisp), 1e-8), 0.49)
    theta = (1.0 - rho) / rho
    alpha = p_arr * theta
    beta = (1.0 - p_arr) * theta
    return (
        log_comb
        + special.betaln(k_arr + alpha, n_arr - k_arr + beta)
        - special.betaln(alpha, beta)
    )


def expected_log2_ratio(
    cn: np.ndarray | float,
    purity: np.ndarray | float,
    ploidy: np.ndarray | float,
    *,
    min_cn_eps: float = 0.02,
) -> np.ndarray:
    """
    Theoretical PureCN segment log2 coverage ratio:
        log2((rho * max(C, eps) + 2 * (1 - rho)) / (rho * ploidy + 2 * (1 - rho))).
    """
    c_eff = np.maximum(np.asarray(cn, dtype=np.float64), min_cn_eps)
    rho = np.asarray(purity, dtype=np.float64)
    psi = np.asarray(ploidy, dtype=np.float64)
    numer = rho * c_eff + 2.0 * (1.0 - rho)
    denom = np.maximum(rho * psi + 2.0 * (1.0 - rho), 1e-6)
    return np.log2(np.maximum(numer / denom, 1e-6))


def standardize_segments_df(
    segments_input: pd.DataFrame | CoverageNormalizationResult,
) -> pd.DataFrame:
    """Validate and standardize a segment DataFrame."""
    if isinstance(segments_input, CoverageNormalizationResult):
        df = segments_input.segments.copy()
    elif isinstance(segments_input, pd.DataFrame):
        df = segments_input.copy()
    else:
        raise ValueError(
            "`segments_df` must be a pandas DataFrame or CoverageNormalizationResult."
        )

    if df.empty:
        raise ValueError("`segments_df` cannot be empty.")

    lower_map = {c.lower(): c for c in df.columns}
    renames: dict[str, str] = {}
    if "chrom" not in df.columns:
        for alias in ("chr", "chromosome", "seqnames"):
            if alias in lower_map:
                renames[lower_map[alias]] = "chrom"
                break
    if "seg_mean" not in df.columns:
        for alias in ("log2_ratio", "log2", "cnlr", "mean", "log_ratio"):
            if alias in lower_map:
                renames[lower_map[alias]] = "seg_mean"
                break
    if "num_mark" not in df.columns:
        for alias in ("n_bins", "num_bins", "probes", "n_probes"):
            if alias in lower_map:
                renames[lower_map[alias]] = "num_mark"
                break
    if renames:
        df = df.rename(columns=renames)

    for req in ("chrom", "start", "end", "seg_mean"):
        if req not in df.columns:
            raise ValueError(f"`segments_df` is missing required column {req!r}.")

    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    df["end"] = np.maximum(df["end"].astype(np.int64), df["start"])
    df["seg_mean"] = df["seg_mean"].astype(np.float64)

    if "size_bp" not in df.columns:
        df["size_bp"] = np.maximum(1, df["end"] - df["start"] + 1)
    if "num_mark" not in df.columns:
        df["num_mark"] = np.maximum(5, (df["size_bp"] // 50_000).clip(5, 500))
    if "seg_sd" not in df.columns:
        df["seg_sd"] = 0.12
    if "seg_id" not in df.columns:
        df["seg_id"] = np.arange(len(df), dtype=np.int64)

    return df.reset_index(drop=True)


def standardize_variants_df(variants_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize a variant allelic count DataFrame."""
    if not isinstance(variants_df, pd.DataFrame):
        raise ValueError("`variants_df` must be a pandas DataFrame.")
    if variants_df.empty:
        return variants_df.copy()

    df = variants_df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    renames: dict[str, str] = {}
    if "chrom" not in df.columns:
        for alias in ("chr", "chromosome", "seqnames"):
            if alias in lower_map:
                renames[lower_map[alias]] = "chrom"
                break
    if "pos" not in df.columns:
        for alias in ("start", "position"):
            if alias in lower_map:
                renames[lower_map[alias]] = "pos"
                break
    if "depth" not in df.columns:
        for alias in ("dp", "coverage", "total_depth", "t_depth", "tum_dp"):
            if alias in lower_map:
                renames[lower_map[alias]] = "depth"
                break
    if "alt_count" not in df.columns:
        for alias in ("alt", "t_alt_count", "tum_alt", "ad_alt", "var_count"):
            if alias in lower_map:
                renames[lower_map[alias]] = "alt_count"
                break
    if "af" not in df.columns:
        for alias in ("vaf", "allelic_fraction", "freq", "tumor_f"):
            if alias in lower_map:
                renames[lower_map[alias]] = "af"
                break
    if "pop_af" not in df.columns:
        for alias in ("gnomad_af", "max_pop_af", "maf", "exac_af"):
            if alias in lower_map:
                renames[lower_map[alias]] = "pop_af"
                break
    if "is_dbsnp" not in df.columns:
        for alias in ("dbsnp", "in_dbsnp"):
            if alias in lower_map:
                renames[lower_map[alias]] = "is_dbsnp"
                break
    if "cosmic_cnt" not in df.columns:
        for alias in ("cosmic_count", "cosmic"):
            if alias in lower_map:
                renames[lower_map[alias]] = "cosmic_cnt"
                break

    if renames:
        df = df.rename(columns=renames)

    for req in ("chrom", "pos"):
        if req not in df.columns:
            raise ValueError(f"`variants_df` is missing required column {req!r}.")

    # Resolve depth and alt_count
    if "depth" not in df.columns:
        if "ref_count" in df.columns and "alt_count" in df.columns:
            df["depth"] = df["ref_count"].astype(np.float64) + df["alt_count"].astype(np.float64)
        else:
            df["depth"] = 200.0

    if "alt_count" not in df.columns:
        if "af" not in df.columns:
            raise ValueError(
                "`variants_df` must provide either `alt_count` or `af`/`vaf`."
            )
        df["alt_count"] = np.round(
            np.clip(df["af"].astype(np.float64), 0.0, 1.0) * df["depth"].astype(np.float64)
        )
    elif not np.issubdtype(df["alt_count"].dtype, np.number):
        # In case 'alt' was nucleotide allele string and 'af' is present
        if "af" in df.columns:
            df["alt_allele"] = df["alt_count"]
            df["alt_count"] = np.round(
                np.clip(df["af"].astype(np.float64), 0.0, 1.0)
                * df["depth"].astype(np.float64)
            )
        else:
            raise ValueError("`alt_count` must be numeric read count.")

    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(np.int64)
    df["depth"] = np.maximum(np.round(df["depth"].astype(np.float64)), 1.0).astype(np.int64)
    df["alt_count"] = np.clip(
        np.round(df["alt_count"].astype(np.float64)).astype(np.int64),
        0,
        df["depth"],
    )
    df["af"] = df["alt_count"].astype(np.float64) / df["depth"].astype(np.float64)

    if "pop_af" not in df.columns:
        df["pop_af"] = 0.0
    else:
        df["pop_af"] = df["pop_af"].fillna(0.0).astype(np.float64)

    if "is_dbsnp" not in df.columns:
        df["is_dbsnp"] = False
    else:
        df["is_dbsnp"] = df["is_dbsnp"].fillna(False).astype(bool)

    if "cosmic_cnt" not in df.columns:
        df["cosmic_cnt"] = 0
    else:
        df["cosmic_cnt"] = df["cosmic_cnt"].fillna(0).astype(np.int64)

    return df.reset_index(drop=True)


def compute_somatic_priors(variants_df: pd.DataFrame, cfg: PureCNConfig) -> np.ndarray:
    """
    Compute per-variant somatic prior probabilities from `pop_af`, `is_dbsnp`,
    and `cosmic_cnt`, honoring explicit `prior_somatic` where provided.
    """
    n = len(variants_df)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    pop_af = variants_df["pop_af"].to_numpy(dtype=np.float64)
    is_dbsnp = variants_df["is_dbsnp"].to_numpy(dtype=bool)
    cosmic_cnt = variants_df["cosmic_cnt"].to_numpy(dtype=np.int64)

    priors = np.full(n, cfg.novel_prior_somatic, dtype=np.float64)
    if cfg.settings == "default":
        rare_mask = (pop_af >= 1e-4) & (pop_af < cfg.common_pop_af_cutoff)
        priors[rare_mask] = cfg.dbsnp_prior_somatic
        priors[is_dbsnp & (pop_af < cfg.common_pop_af_cutoff)] = cfg.dbsnp_prior_somatic
        priors[cosmic_cnt >= 2] = cfg.cosmic_hotspot_prior_somatic
        priors[pop_af >= cfg.common_pop_af_cutoff] = cfg.common_pop_prior_somatic
    else:
        # R_default (PureCN::predictSomatic / setPriorVcf parity)
        priors[is_dbsnp] = cfg.dbsnp_prior_somatic  # 0.1
        priors[cosmic_cnt >= 2] = cfg.cosmic_hotspot_prior_somatic  # 0.999
        priors[pop_af > cfg.common_pop_af_cutoff] = cfg.common_pop_prior_somatic  # 0.0005

    if "prior_somatic" in variants_df.columns:
        explicit = variants_df["prior_somatic"].to_numpy(dtype=np.float64)
        valid_exp = np.isfinite(explicit)
        priors[valid_exp] = explicit[valid_exp]

    return np.clip(priors, 1e-5, 1.0 - 1e-5)


def estimate_overdispersion(
    variants_df: pd.DataFrame,
    *,
    min_rho: float = 1e-4,
) -> float:
    """
    Empirically estimate Beta-Binomial allelic overdispersion `rho` from candidate
    germline heterozygous SNPs around expected AF = 0.5 (PureCN R_default behavior).
    """
    if variants_df.empty:
        return max(1e-3, min_rho)

    depth = variants_df["depth"].to_numpy(dtype=np.float64)
    alt = variants_df["alt_count"].to_numpy(dtype=np.float64)
    af = alt / np.maximum(depth, 1.0)
    pop_af = variants_df["pop_af"].to_numpy(dtype=np.float64)
    is_dbsnp = variants_df["is_dbsnp"].to_numpy(dtype=bool)

    germ_het = (depth >= 25) & (af >= 0.32) & (af <= 0.68) & ((pop_af > 1e-3) | is_dbsnp)
    if int(np.sum(germ_het)) < 3:
        germ_het = (depth >= 30) & (af >= 0.38) & (af <= 0.62)
    if int(np.sum(germ_het)) < 3:
        return max(1e-3, min_rho)

    k_sub = alt[germ_het]
    n_sub = depth[germ_het]

    def _neg_ll(rho_val: float) -> float:
        return -float(np.sum(beta_binomial_logpmf(k_sub, n_sub, 0.5, rho_val)))

    res = optimize.minimize_scalar(_neg_ll, bounds=(min_rho, 0.05), method="bounded")
    return float(max(res.x, min_rho))


def map_variants_to_segments(
    variants_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> np.ndarray:
    """Map each row of `variants_df` to an index in `segments_df` (-1 if unmatched chrom)."""
    n_var = len(variants_df)
    seg_idx = np.full(n_var, -1, dtype=np.int64)
    if n_var == 0 or segments_df.empty:
        return seg_idx

    chrom_to_segs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom in pd.unique(segments_df["chrom"]):
        mask = (segments_df["chrom"] == chrom).to_numpy()
        idxs = np.flatnonzero(mask)
        starts = segments_df.loc[idxs, "start"].to_numpy(dtype=np.int64)
        ends = segments_df.loc[idxs, "end"].to_numpy(dtype=np.int64)
        chrom_to_segs[str(chrom)] = (idxs, starts, ends)

    var_chroms = variants_df["chrom"].astype(str).to_numpy()
    var_pos = variants_df["pos"].to_numpy(dtype=np.int64)

    for i in range(n_var):
        ch = var_chroms[i]
        ch_alt = ch[3:] if ch.startswith("chr") else f"chr{ch}"
        entry = chrom_to_segs.get(ch) or chrom_to_segs.get(ch_alt)
        if entry is None:
            continue
        idxs, starts, ends = entry
        p = var_pos[i]
        inside = (starts <= p) & (p <= ends)
        if np.any(inside):
            seg_idx[i] = int(idxs[np.flatnonzero(inside)[0]])
        else:
            # Nearest segment on same chromosome
            dists = np.minimum(np.abs(starts - p), np.abs(ends - p))
            seg_idx[i] = int(idxs[int(np.argmin(dists))])

    return seg_idx


def _compute_segment_allelic_table(
    variants_df: pd.DataFrame,
    seg_indices: np.ndarray,
    n_segments: int,
    purity_grid: np.ndarray,
    max_cn: int,
    rho_overdisp: float,
    priors_som: np.ndarray,
    base_error: float = 0.001,
) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    """
    Precompute segment-level allelic Beta-Binomial log-likelihood `seg_var_ll[p, s, C]`
    and best minor allele `best_M[(p, s, C)]` across purity grid and C in 0..max_cn.
    """
    n_p = purity_grid.size
    seg_var_ll = np.zeros((n_p, n_segments, max_cn + 1), dtype=np.float64)
    best_m_lookup: dict[tuple[int, int, int], int] = {}

    if variants_df.empty:
        return seg_var_ll, best_m_lookup

    k_all = variants_df["alt_count"].to_numpy(dtype=np.float64)[:, np.newaxis]  # (V, 1)
    n_all = variants_df["depth"].to_numpy(dtype=np.float64)[:, np.newaxis]  # (V, 1)
    rho_row = purity_grid[np.newaxis, :]  # (1, P)

    p_som = priors_som[:, np.newaxis]  # (V, 1)
    p_germ = 1.0 - p_som

    subclonal_grid = np.array([0.25, 0.50, 0.75], dtype=np.float64)

    # Group variant indices by segment
    seg_to_vars: dict[int, np.ndarray] = {}
    for s in range(n_segments):
        v_idx = np.flatnonzero(seg_indices == s)
        if v_idx.size > 0:
            seg_to_vars[s] = v_idx

    if not seg_to_vars:
        return seg_var_ll, best_m_lookup

    for c_state in range(max_cn + 1):
        denom = rho_row * float(c_state) + 2.0 * (1.0 - rho_row)  # (1, P)

        # Somatic marginal likelihood over clonal m in 1..max(1, C) + subclonal states
        if c_state == 0:
            f_som = np.clip(rho_row * subclonal_grid[0] / denom, base_error, 1.0 - base_error)
            ll_som = beta_binomial_logpmf(k_all, n_all, f_som, rho_overdisp)
        else:
            som_terms = []
            w_clonal = 0.85 / float(c_state)
            for m_mult in range(1, c_state + 1):
                f_m = np.clip((rho_row * float(m_mult)) / denom, base_error, 1.0 - base_error)
                som_terms.append(
                    np.log(w_clonal) + beta_binomial_logpmf(k_all, n_all, f_m, rho_overdisp)
                )
            w_sub = 0.15 / float(subclonal_grid.size)
            for phi in subclonal_grid:
                f_sub = np.clip((rho_row * float(phi)) / denom, base_error, 1.0 - base_error)
                som_terms.append(
                    np.log(w_sub) + beta_binomial_logpmf(k_all, n_all, f_sub, rho_overdisp)
                )
            ll_som = special.logsumexp(np.stack(som_terms, axis=0), axis=0)  # (V, P)

        max_minor = c_state // 2
        m_candidates = list(range(max_minor + 1))
        # Evaluate germline likelihood for each minor allele M in 0..floor(C/2)
        ll_var_by_m = []
        ll_hom_alt = beta_binomial_logpmf(k_all, n_all, 1.0 - base_error, rho_overdisp)
        for m_minor in m_candidates:
            m_major = c_state - m_minor
            f_g1 = np.clip(
                (rho_row * float(m_minor) + (1.0 - rho_row)) / denom,
                base_error,
                1.0 - base_error,
            )
            f_g2 = np.clip(
                (rho_row * float(m_major) + (1.0 - rho_row)) / denom,
                base_error,
                1.0 - base_error,
            )
            ll_het = np.logaddexp(
                np.log(0.5) + beta_binomial_logpmf(k_all, n_all, f_g1, rho_overdisp),
                np.log(0.5) + beta_binomial_logpmf(k_all, n_all, f_g2, rho_overdisp),
            )
            ll_germ = np.logaddexp(np.log(0.94) + ll_het, np.log(0.06) + ll_hom_alt)
            ll_mix = np.logaddexp(np.log(p_som) + ll_som, np.log(p_germ) + ll_germ)
            ll_var_by_m.append(ll_mix)  # each is (V, P)

        # Aggregate per segment across M candidates
        n_m = len(m_candidates)
        # Mild prior favoring M >= 1 over LOH unless allelic data support LOH
        m_prior = np.full(n_m, -np.log(float(n_m)), dtype=np.float64)
        if c_state >= 2 and n_m > 1:
            m_prior[0] -= 0.25
            m_prior = m_prior - special.logsumexp(m_prior)

        for s, v_idx in seg_to_vars.items():
            seg_m_scores = np.empty((n_m, n_p), dtype=np.float64)
            for idx_m in range(n_m):
                seg_m_scores[idx_m] = m_prior[idx_m] + np.sum(ll_var_by_m[idx_m][v_idx], axis=0)
            seg_var_ll[:, s, c_state] = special.logsumexp(seg_m_scores, axis=0)
            best_m_arr = np.argmax(seg_m_scores, axis=0)
            for p_idx in range(n_p):
                best_m_lookup[(p_idx, s, c_state)] = int(m_candidates[int(best_m_arr[p_idx])])

    return seg_var_ll, best_m_lookup


def grid_search_purity_ploidy(
    segments_df: pd.DataFrame | CoverageNormalizationResult,
    variants_df: pd.DataFrame | None = None,
    purity_grid: np.ndarray | list[float] | None = None,
    ploidy_grid: np.ndarray | list[float] | None = None,
    max_cn: int = 8,
    settings: SettingsMode = "default",
) -> list[PureCNSolution]:
    """
    Joint 2D (purity, ploidy) optimization scoring segment log2 coverage ratios
    and Beta-Binomial variant allelic counts, returning ranked local-optima
    `PureCNSolution` objects (`solutions[0]` is optimal).
    """
    cfg = get_purecn_config(validate_settings(settings))
    if max_cn < 2:
        raise ValueError("`max_cn` must be >= 2.")

    segs = standardize_segments_df(segments_df)
    p_grid = (
        np.asarray(purity_grid, dtype=np.float64)
        if purity_grid is not None
        else cfg.default_purity_grid()
    )
    q_grid = (
        np.asarray(ploidy_grid, dtype=np.float64)
        if ploidy_grid is not None
        else cfg.default_ploidy_grid()
    )
    if p_grid.size == 0 or q_grid.size == 0:
        raise ValueError("`purity_grid` and `ploidy_grid` must be non-empty.")
    if np.any((p_grid <= 0.0) | (p_grid >= 1.0)):
        p_grid = np.clip(p_grid, 0.01, 0.995)
    if np.any(q_grid <= 0.2):
        raise ValueError("`ploidy_grid` values must be > 0.2.")

    vars_std = (
        standardize_variants_df(variants_df)
        if variants_df is not None and not variants_df.empty
        else pd.DataFrame()
    )

    n_seg = len(segs)
    n_p = p_grid.size
    n_q = q_grid.size
    c_states = np.arange(max_cn + 1, dtype=np.float64)

    # Precompute variant allelic table (P, S, C+1)
    if not vars_std.empty:
        rho_od = (
            cfg.rho_overdisp
            if cfg.rho_overdisp is not None
            else estimate_overdispersion(vars_std, min_rho=cfg.min_rho_overdisp)
        )
        priors_som = compute_somatic_priors(vars_std, cfg)
        seg_map = map_variants_to_segments(vars_std, segs)
        seg_var_ll, best_m_lookup = _compute_segment_allelic_table(
            vars_std,
            seg_map,
            n_seg,
            p_grid,
            max_cn,
            rho_od,
            priors_som,
        )
    else:
        seg_var_ll = np.zeros((n_p, n_seg, max_cn + 1), dtype=np.float64)
        best_m_lookup = {}

    # Segment observations and weights
    y_seg = segs["seg_mean"].to_numpy(dtype=np.float64)  # (S,)
    size_bp = segs["size_bp"].to_numpy(dtype=np.float64)
    size_frac = size_bp / np.maximum(np.sum(size_bp), 1.0)  # sums to 1
    num_mark = segs["num_mark"].to_numpy(dtype=np.float64)
    eff_marks = np.clip(num_mark, 3.0, 80.0)
    sigma_bin = np.maximum(segs["seg_sd"].to_numpy(dtype=np.float64), 0.08)

    # Expected log2 ratio grid: (P, Q, 1, C+1)
    exp_l2 = expected_log2_ratio(
        c_states[np.newaxis, np.newaxis, np.newaxis, :],
        p_grid[:, np.newaxis, np.newaxis, np.newaxis],
        q_grid[np.newaxis, :, np.newaxis, np.newaxis],
    )
    resid = y_seg[np.newaxis, np.newaxis, :, np.newaxis] - exp_l2  # (P, Q, S, C+1)
    seg_var = (sigma_bin * sigma_bin / eff_marks)[np.newaxis, np.newaxis, :, np.newaxis]
    cn_clonal_ll = -0.5 * (resid * resid) / seg_var - 0.5 * np.log(2.0 * np.pi * seg_var)

    # Robust mixture with subclonal copy-number state floor so single subclonal segments don't blow up
    subclonal_floor = -4.5 * np.ones_like(cn_clonal_ll[..., :1])
    cn_seg_ll = np.logaddexp(np.log(0.95) + cn_clonal_ll, np.log(0.05) + subclonal_floor)

    # Joint per-segment state log-likelihood (P, Q, S, C+1)
    joint_seg_state = cn_seg_ll + seg_var_ll[:, np.newaxis, :, :]
    best_c_grid = np.argmax(joint_seg_state, axis=-1)  # (P, Q, S)
    seg_marginal_ll = special.logsumexp(joint_seg_state, axis=-1)  # (P, Q, S)

    total_cn_snv_ll = np.sum(seg_marginal_ll, axis=-1)  # (P, Q)

    # Ploidy mass-balance consistency: implied mean tumor CN must match candidate ploidy
    implied_ploidy = np.sum(
        best_c_grid.astype(np.float64) * size_frac[np.newaxis, np.newaxis, :],
        axis=-1,
    )  # (P, Q)
    ploidy_mismatch = q_grid[np.newaxis, :] - implied_ploidy
    consistency_ll = -0.5 * (ploidy_mismatch / 0.14) ** 2 * min(float(n_seg), 25.0)

    # Parsimony against WGD / sub-harmonic copy-number skipping when odd states are absent
    # If all segments take even copy numbers >= 2 (e.g., {2, 4, 6}), mild parsimony favors diploid
    all_even_high = np.all((best_c_grid % 2 == 0), axis=-1) & (implied_ploidy > 2.6)
    wgd_parsimony = np.where(all_even_high, -3.5, 0.0)
    diploid_prior = -0.35 * np.abs(q_grid[np.newaxis, :] - 2.0)

    # Subclonal genome fraction at each (P, Q)
    best_exp_l2 = np.take_along_axis(
        exp_l2, best_c_grid[..., np.newaxis], axis=-1
    )[..., 0]
    best_se = np.sqrt(seg_var[..., 0])
    is_subclonal_grid = np.abs(y_seg[np.newaxis, np.newaxis, :] - best_exp_l2) > np.maximum(
        2.5 * best_se, 0.12
    )
    subclonal_frac_grid = np.sum(
        is_subclonal_grid.astype(np.float64) * size_frac[np.newaxis, np.newaxis, :],
        axis=-1,
    )

    score_grid = total_cn_snv_ll + consistency_ll + wgd_parsimony + diploid_prior
    if cfg.max_non_clonal is not None:
        excess_sub = np.maximum(0.0, subclonal_frac_grid - cfg.max_non_clonal)
        score_grid = score_grid - 150.0 * excess_sub - np.where(
            subclonal_frac_grid > cfg.max_non_clonal, 25.0, 0.0
        )

    # Identify local maxima in 3x3 neighborhood + global maximum
    candidates: list[tuple[float, int, int]] = []
    global_flat = int(np.argmax(score_grid))
    gp, gq = divmod(global_flat, n_q)
    candidates.append((float(score_grid[gp, gq]), gp, gq))

    for pi in range(n_p):
        p_lo = max(0, pi - 2)
        p_hi = min(n_p, pi + 3)
        for qi in range(n_q):
            if pi == gp and qi == gq:
                continue
            q_lo = max(0, qi - 2)
            q_hi = min(n_q, qi + 3)
            val = float(score_grid[pi, qi])
            if val >= float(np.max(score_grid[p_lo:p_hi, q_lo:q_hi])) - 1e-12:
                candidates.append((val, pi, qi))

    candidates.sort(key=lambda t: t[0], reverse=True)

    # Deduplicate nearby local maxima within (|dp| < 0.05 and |dq| < 0.25)
    kept: list[tuple[float, int, int]] = []
    for val, pi, qi in candidates:
        pur = float(p_grid[pi])
        plo = float(q_grid[qi])
        if any(
            abs(pur - float(p_grid[kp])) < 0.05 and abs(plo - float(q_grid[kq])) < 0.25
            for _, kp, kq in kept
        ):
            continue
        kept.append((val, pi, qi))
        if len(kept) >= 10:
            break

    solutions: list[PureCNSolution] = []
    for rank_idx, (tot_score, pi, qi) in enumerate(kept, start=1):
        pur = float(p_grid[pi])
        c_vec = best_c_grid[pi, qi].astype(np.int64)
        eff_ploidy = float(np.sum(c_vec.astype(np.float64) * size_frac))
        # Use grid ploidy when very close, or weighted segment ploidy when segments cover genome
        plo = float(q_grid[qi]) if abs(eff_ploidy - float(q_grid[qi])) > 0.25 else eff_ploidy

        m_vec = np.empty(n_seg, dtype=np.int64)
        for s in range(n_seg):
            c_val = int(c_vec[s])
            m_vec[s] = best_m_lookup.get((pi, s, c_val), c_val // 2)

        exp_seg = expected_log2_ratio(c_vec, pur, plo)
        sub_mask = is_subclonal_grid[pi, qi]
        loh_mask = (m_vec == 0) & (c_vec >= 0)

        seg_out = segs.copy()
        seg_out["C"] = c_vec
        seg_out["M"] = m_vec
        seg_out["expected_log2"] = exp_seg
        seg_out["is_subclonal"] = sub_mask
        seg_out["loh"] = loh_mask

        cn_ll_val = float(
            np.sum(np.take_along_axis(cn_seg_ll[pi, qi], c_vec[:, np.newaxis], axis=-1))
        )
        snv_ll_val = float(
            np.sum(np.take_along_axis(seg_var_ll[pi], c_vec[:, np.newaxis], axis=-1))
        )
        solutions.append(
            PureCNSolution(
                purity=round(pur, 4),
                ploidy=round(plo, 4),
                log_likelihood=tot_score,
                cn_log_likelihood=cn_ll_val,
                snv_log_likelihood=snv_ll_val,
                subclonal_fraction=float(subclonal_frac_grid[pi, qi]),
                rank=rank_idx,
                segments=seg_out,
            )
        )

    return solutions
