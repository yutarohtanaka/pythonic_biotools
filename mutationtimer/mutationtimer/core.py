"""
Core Beta-Binomial mixture EM, mutation multiplicity & timing classification,
and end-to-end `mutation_time` / `MutationTimeR` API for `mutationtimer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from .io import (
    MutationTimerConfig,
    SettingsMode,
    extract_subclonal_phis,
    map_variants_to_segments,
    normal_copy_number,
    normalize_gender,
    validate_and_prepare_segments,
    validate_and_prepare_variants,
    validate_settings,
)
from .timing import (
    WGDTimingResult,
    estimate_wgd_timing,
    pi_to_all_times,
    pi_to_time,
)


def expected_vaf(
    multiplicity: np.ndarray | float,
    phi: np.ndarray | float,
    purity: float,
    major_cn: int,
    minor_cn: int,
    normal_cn: int = 2,
    clonal_frequency: float = 1.0,
) -> np.ndarray:
    """
    Compute expected somatic variant allele frequency (VAF):

        f(m, phi) = (purity * m * phi) / (purity * C_tum_eff + normal_cn * (1 - purity))

    where `C_tum_eff = clonal_frequency * (major_cn + minor_cn) + (1 - clonal_frequency) * normal_cn`
    (reducing to `major_cn + minor_cn` for clonal copy-number segments `clonal_frequency == 1`).
    """
    m_arr = np.asarray(multiplicity, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    c_tum = clonal_frequency * float(major_cn + minor_cn) + (1.0 - clonal_frequency) * float(normal_cn)
    denom = purity * c_tum + float(normal_cn) * (1.0 - purity)
    if denom <= 0.0:
        return np.zeros(np.broadcast(m_arr, phi_arr).shape, dtype=np.float64)
    vaf = (purity * m_arr * phi_arr) / denom
    return np.clip(vaf, 1e-6, 1.0 - 1e-6)


def beta_binomial_logpmf(
    alt_count: np.ndarray,
    total_depth: np.ndarray,
    vaf: np.ndarray,
    rho_bb: float = 0.01,
) -> np.ndarray:
    """
    Vectorized Beta-Binomial log-PMF for observed `alt_count` (shape `(N,)` or `(N, 1)`)
    given `total_depth` and expected `vaf` (shape `(K,)` or `(1, K)`), with overdispersion
    parameter `rho_bb in (0, 1)`.
    """
    x = np.asarray(alt_count, dtype=np.float64)
    n = np.asarray(total_depth, dtype=np.float64)
    f = np.clip(np.asarray(vaf, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    conc = (1.0 - rho_bb) / rho_bb
    alpha = f * conc
    beta = (1.0 - f) * conc

    log_comb = (
        special.gammaln(n + 1.0)
        - special.gammaln(x + 1.0)
        - special.gammaln(n - x + 1.0)
    )
    return (
        log_comb
        + special.betaln(x + alpha, (n - x) + beta)
        - special.betaln(alpha, beta)
    )


def _fit_segment_em(
    log_lik: np.ndarray,
    prior_init: np.ndarray,
    n_clonal: int,
    has_default_subclone: bool,
    max_iter: int,
    tol: float,
    settings: SettingsMode,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Beta-Binomial mixture weights `pi` on precomputed `(N_s, K)` `log_lik`
    via Expectation-Maximization.

    Returns `(pi, gamma)` where `pi` has shape `(K,)` and `gamma` has shape `(N_s, K)`.
    """
    n_var, k_states = log_lik.shape
    pi = np.array(prior_init, dtype=np.float64, copy=True)
    pi = np.maximum(pi, 1e-8)
    pi /= np.sum(pi)

    if n_var == 0:
        return pi, np.empty((0, k_states), dtype=np.float64)

    w = (
        np.ones(n_var, dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )

    if settings == "default":
        # Anti-collapse Laplace-smoothed Dirichlet prior (alpha = 1.05 => +0.05 pseudocount)
        pseudo = np.full(k_states, 0.05, dtype=np.float64)
        if has_default_subclone and k_states > n_clonal:
            pseudo[n_clonal:] = 0.005
    else:
        # Weak 1/K prior (scaled by overdispersion 1e-3 so empty states approach zero without biasing counts)
        pseudo = np.full(k_states, 1e-3 / k_states, dtype=np.float64)

    gamma = np.full((n_var, k_states), 1.0 / k_states, dtype=np.float64)
    for _ in range(max_iter):
        log_pi = np.log(np.maximum(pi, 1e-300))
        log_joint = log_lik + log_pi[None, :]
        max_lj = np.max(log_joint, axis=1, keepdims=True)
        probs = np.exp(log_joint - max_lj)
        probs_sum = np.sum(probs, axis=1, keepdims=True)
        gamma = probs / np.maximum(probs_sum, 1e-300)

        nk = np.sum(gamma * w[:, None], axis=0)
        unnorm = np.maximum(nk + pseudo, 1e-10)

        pi_next = unnorm / np.sum(unnorm)
        if float(np.max(np.abs(pi_next - pi))) < tol:
            pi = pi_next
            break
        pi = pi_next

    # Final E-step at converged `pi`
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_joint = log_lik + log_pi[None, :]
    max_lj = np.max(log_joint, axis=1, keepdims=True)
    probs = np.exp(log_joint - max_lj)
    gamma = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-300)
    return pi, gamma


@dataclass
class MutationTimeResult:
    """Result container returned by `mutation_time` and `MutationTimeR`."""

    variants_df: pd.DataFrame
    segments_df: pd.DataFrame
    segment_pis: dict[int, np.ndarray]
    wgd_result: WGDTimingResult
    purity: float
    gender: str
    settings: SettingsMode

    def summary(self) -> dict[str, Any]:
        """Return high-level summary statistics of somatic mutation and CNA timing."""
        cls_counts = (
            self.variants_df["timing_class"].value_counts().to_dict()
            if "timing_class" in self.variants_df.columns and not self.variants_df.empty
            else {}
        )
        timed_mask = self.segments_df["T"].notna() if "T" in self.segments_df.columns else pd.Series(dtype=bool)
        median_gain_t = (
            float(self.segments_df.loc[timed_mask, "T"].median())
            if bool(timed_mask.any())
            else float("nan")
        )
        return {
            "n_variants": int(len(self.variants_df)),
            "n_segments": int(len(self.segments_df)),
            "n_timed_segments": int(timed_mask.sum()),
            "median_gain_time": median_gain_t,
            "timing_class_counts": {str(k): int(v) for k, v in cls_counts.items()},
            "is_wgd": self.wgd_result.is_wgd,
            "wgd_time": self.wgd_result.wgd_time,
            "is_wgd_synchronous": self.wgd_result.is_synchronous,
            "purity": self.purity,
            "gender": self.gender,
            "settings": self.settings,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return copy of timed variants DataFrame."""
        return self.variants_df.copy()


def mutation_time(
    variants_df: pd.DataFrame,
    cn_segments_df: pd.DataFrame,
    purity: float,
    clusters_df: pd.DataFrame | None = None,
    gender: str = "female",
    n_boot: int = 200,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> MutationTimeResult:
    """
    Run evolutionary timing of somatic point mutations, allele-specific copy-number
    gains, and Whole-Genome Doubling (`WGD`) following Gerstung et al. (Nature 2020).

    Parameters
    ----------
    variants_df : pd.DataFrame
        Somatic mutations with `chrom`, `pos`, `alt_count`, and either
        `total_depth` or `ref_count`.
    cn_segments_df : pd.DataFrame
        Allele-specific copy-number segments with `chrom`, `start`, `end`,
        `major_cn`, `minor_cn`, and optional `clonal_frequency`.
    purity : float
        Tumor purity `rho in (0, 1]`.
    clusters_df : pd.DataFrame | None, default=None
        Optional subclonal cluster specifications (`ccf` in `(0, 1)` or
        `cellular_prevalence` / `proportion`). When `None`, includes a default
        subclonal state at `phi = 0.5`.
    gender : str, default="female"
        Sample sex (`"female"` or `"male"`); sets germline copy number `C_norm = 1`
        on male `X` and `Y` chromosomes, and `2` elsewhere.
    n_boot : int, default=200
        Bootstrap iterations per gained segment for 95% CIs (`T_lo`, `T_hi`).
    seed : int, default=42
        Random seed for deterministic bootstrap resampling.
    settings : {"default", "R_default"}, default="default"
        `"default"` uses Laplace-smoothed Dirichlet prior (`alpha=1.05`) and
        empirical-Bayes continuous `mut_cn`; `"R_default"` uses `MutationTimeR`
        parity settings (`alpha=1/K`, raw inverted `mut_cn`, and exact 0.5
        classification thresholds).

    Returns
    -------
    MutationTimeResult
        Timed variants, timed copy-number segments, fitted per-segment mixture
        proportions, and WGD synchronicity assessment.
    """
    validate_settings(settings)
    norm_gender = normalize_gender(gender)
    cfg = MutationTimerConfig(
        purity=float(purity),
        gender=norm_gender,
        n_boot=int(n_boot),
        seed=int(seed),
        rho_bb=0.01,
        settings=settings,
    )

    segs = validate_and_prepare_segments(cn_segments_df)
    vars_clean = validate_and_prepare_variants(variants_df)
    sub_phis, sub_weights = extract_subclonal_phis(
        clusters_df, purity=cfg.purity, settings=settings
    )
    has_default_subclone = clusters_df is None

    n_var = len(vars_clean)
    n_seg = len(segs)
    rng = np.random.default_rng(cfg.seed)

    seg_map = map_variants_to_segments(vars_clean, segs)

    # Variant-level output arrays
    vaf_obs = np.full(n_var, np.nan, dtype=np.float64)
    var_seg_id = np.full(n_var, -1, dtype=np.int64)
    var_maj = np.full(n_var, -1, dtype=np.int64)
    var_min = np.full(n_var, -1, dtype=np.int64)
    mut_cn_out = np.full(n_var, np.nan, dtype=np.float64)
    mut_cn_raw = np.full(n_var, np.nan, dtype=np.float64)
    mut_cn_post = np.full(n_var, np.nan, dtype=np.float64)
    mut_cn_map = np.full(n_var, np.nan, dtype=np.float64)
    p_clonal_out = np.full(n_var, np.nan, dtype=np.float64)
    p_early_out = np.full(n_var, np.nan, dtype=np.float64)
    p_late_out = np.full(n_var, np.nan, dtype=np.float64)
    p_sub_out = np.full(n_var, np.nan, dtype=np.float64)
    timing_cls = np.full(n_var, "NA", dtype=object)

    # Segment-level output arrays
    seg_n_mut = np.zeros(n_seg, dtype=np.int64)
    seg_pi_early = np.full(n_seg, np.nan, dtype=np.float64)
    seg_pi_late = np.full(n_seg, np.nan, dtype=np.float64)
    seg_pi_sub = np.full(n_seg, np.nan, dtype=np.float64)
    seg_t = np.full(n_seg, np.nan, dtype=np.float64)
    seg_t_lo = np.full(n_seg, np.nan, dtype=np.float64)
    seg_t_hi = np.full(n_seg, np.nan, dtype=np.float64)
    seg_t_sec = np.full(n_seg, np.nan, dtype=np.float64)
    segment_pis: dict[int, np.ndarray] = {}

    if n_var > 0:
        alt_all = vars_clean["alt_count"].to_numpy(dtype=np.float64)
        dp_all = vars_clean["total_depth"].to_numpy(dtype=np.float64)
        pos_dp = dp_all > 0
        vaf_obs[pos_dp] = alt_all[pos_dp] / dp_all[pos_dp]
    else:
        alt_all = np.empty(0, dtype=np.float64)
        dp_all = np.empty(0, dtype=np.float64)

    for s_idx in range(n_seg):
        row = segs.iloc[s_idx]
        maj = int(row["major_cn"])
        mn = int(row["minor_cn"])
        cf = float(row["clonal_frequency"])
        seg_id = int(row["segment_id"])
        c_norm = normal_copy_number(str(row["chrom_canon"]), cfg.gender)

        v_idx = np.flatnonzero((seg_map == s_idx) & (dp_all > 0))
        seg_n_mut[s_idx] = int(v_idx.size)

        if v_idx.size > 0:
            var_seg_id[v_idx] = seg_id
            var_maj[v_idx] = maj
            var_min[v_idx] = mn

        if maj < 1 or v_idx.size == 0:
            continue

        # Candidate states: m = 1..maj (phi = cf) + subclonal clusters (m = 1, phi = sub_phis)
        clonal_m = np.arange(1, maj + 1, dtype=np.float64)
        clonal_phi = np.full(maj, cf, dtype=np.float64)
        state_m = np.concatenate([clonal_m, np.ones_like(sub_phis)])
        state_phi = np.concatenate([clonal_phi, sub_phis])
        eff_cn_states = state_m * state_phi

        state_vafs = expected_vaf(
            multiplicity=state_m,
            phi=state_phi,
            purity=cfg.purity,
            major_cn=maj,
            minor_cn=mn,
            normal_cn=c_norm,
            clonal_frequency=cf,
        )

        # Initial prior weights
        clonal_init = np.ones(maj, dtype=np.float64) / maj
        if sub_weights.size > 0:
            prior_init = np.concatenate([clonal_init * (1.0 - np.sum(sub_weights)), sub_weights])
        else:
            prior_init = clonal_init
        prior_init /= np.sum(prior_init)

        alt_s = alt_all[v_idx][:, None]
        dp_s = dp_all[v_idx][:, None]
        ll_mat = beta_binomial_logpmf(alt_s, dp_s, state_vafs[None, :], rho_bb=cfg.rho_bb)

        pi_hat, gamma = _fit_segment_em(
            log_lik=ll_mat,
            prior_init=prior_init,
            n_clonal=maj,
            has_default_subclone=has_default_subclone,
            max_iter=cfg.max_em_iter,
            tol=cfg.em_tol,
            settings=settings,
        )
        segment_pis[s_idx] = pi_hat

        seg_pi_late[s_idx] = float(pi_hat[0])
        seg_pi_early[s_idx] = float(np.sum(pi_hat[1:maj])) if maj >= 2 else 0.0
        seg_pi_sub[s_idx] = float(np.sum(pi_hat[maj:])) if pi_hat.size > maj else 0.0

        # Variant-level posteriors & continuous mutation copy number
        p_cl = np.sum(gamma[:, :maj], axis=1)
        p_su = np.sum(gamma[:, maj:], axis=1) if pi_hat.size > maj else np.zeros(v_idx.size)
        p_ea = np.sum(gamma[:, 1:maj], axis=1) if maj >= 2 else np.zeros(v_idx.size)
        p_la = gamma[:, 0] if maj >= 2 else np.zeros(v_idx.size)

        p_clonal_out[v_idx] = p_cl
        p_sub_out[v_idx] = p_su
        p_early_out[v_idx] = p_ea
        p_late_out[v_idx] = p_la

        c_tum_eff = cf * float(maj + mn) + (1.0 - cf) * float(c_norm)
        denom_cn = cfg.purity * c_tum_eff + float(c_norm) * (1.0 - cfg.purity)
        raw_m = vaf_obs[v_idx] * (denom_cn / cfg.purity)
        post_m = gamma @ eff_cn_states
        map_idx = np.argmax(gamma, axis=1)
        map_m = eff_cn_states[map_idx]

        mut_cn_raw[v_idx] = raw_m
        mut_cn_post[v_idx] = post_m
        mut_cn_map[v_idx] = map_m

        if settings == "default":
            shrink_w = dp_all[v_idx] / (dp_all[v_idx] + 25.0)
            mut_cn_out[v_idx] = shrink_w * raw_m + (1.0 - shrink_w) * post_m
        else:
            mut_cn_out[v_idx] = raw_m

        # Classify into the 4 PCAWG timing classes
        cls_s = np.empty(v_idx.size, dtype=object)
        is_sub = p_cl < 0.5
        cls_s[is_sub] = "subclonal"
        if maj < 2:
            cls_s[~is_sub] = "clonal [NA]"
        else:
            if settings == "R_default":
                # Exact MutationTimeR thresholds: p_clonal >= 0.5, p_early >= 0.5
                for j in np.flatnonzero(~is_sub):
                    if p_ea[j] >= 0.5:
                        cls_s[j] = "clonal [early]"
                    elif p_la[j] >= 0.5:
                        cls_s[j] = "clonal [late]"
                    else:
                        cls_s[j] = "clonal [NA]"
            else:
                cond_early = p_ea / np.maximum(p_cl, 1e-12)
                cls_s[(~is_sub) & (cond_early >= 0.5)] = "clonal [early]"
                cls_s[(~is_sub) & (cond_early < 0.5)] = "clonal [late]"

        timing_cls[v_idx] = cls_s

        # Segment gain timing + nonparametric variant bootstrap CIs
        if maj >= 2:
            all_times = pi_to_all_times(maj, mn, pi_hat, settings=settings)
            if all_times:
                seg_t[s_idx] = all_times[0]
                if len(all_times) >= 2:
                    seg_t_sec[s_idx] = all_times[1]

                if cfg.n_boot == 0:
                    seg_t_lo[s_idx] = seg_t[s_idx]
                    seg_t_hi[s_idx] = seg_t[s_idx]
                else:
                    n_s = v_idx.size
                    unif_p = np.full(n_s, 1.0 / n_s, dtype=np.float64)
                    boot_w = rng.multinomial(n_s, unif_p, size=cfg.n_boot).astype(np.float64)
                    t_boots = np.empty(cfg.n_boot, dtype=np.float64)
                    for b in range(cfg.n_boot):
                        pi_b, _ = _fit_segment_em(
                            log_lik=ll_mat,
                            prior_init=pi_hat,
                            n_clonal=maj,
                            has_default_subclone=has_default_subclone,
                            max_iter=25,
                            tol=1e-5,
                            settings=settings,
                            sample_weights=boot_w[b],
                        )
                        t_boots[b] = pi_to_time(maj, mn, pi_b, settings=settings)

                    valid_tb = t_boots[np.isfinite(t_boots)]
                    if valid_tb.size > 0:
                        lo, hi = np.percentile(valid_tb, [2.5, 97.5])
                        seg_t_lo[s_idx] = float(min(lo, seg_t[s_idx]))
                        seg_t_hi[s_idx] = float(max(hi, seg_t[s_idx]))
                    else:
                        seg_t_lo[s_idx] = seg_t[s_idx]
                        seg_t_hi[s_idx] = seg_t[s_idx]

    vars_out = vars_clean.drop(columns=["chrom_canon"]).copy()
    vars_out["vaf"] = vaf_obs
    vars_out["segment_id"] = var_seg_id
    vars_out["major_cn"] = var_maj
    vars_out["minor_cn"] = var_min
    vars_out["mut_cn"] = mut_cn_out
    vars_out["mut_cn_raw"] = mut_cn_raw
    vars_out["mut_cn_posterior"] = mut_cn_post
    vars_out["mut_cn_map"] = mut_cn_map
    vars_out["p_clonal"] = p_clonal_out
    vars_out["p_early"] = p_early_out
    vars_out["p_late"] = p_late_out
    vars_out["p_subclonal"] = p_sub_out
    vars_out["timing_class"] = timing_cls

    segs_out = segs.drop(columns=["chrom_canon"]).copy()
    segs_out["n_mutations"] = seg_n_mut
    segs_out["pi_clonal_early"] = seg_pi_early
    segs_out["pi_clonal_late"] = seg_pi_late
    segs_out["pi_subclonal"] = seg_pi_sub
    segs_out["T"] = seg_t
    segs_out["T_lo"] = seg_t_lo
    segs_out["T_hi"] = seg_t_hi
    segs_out["T_second"] = seg_t_sec

    wgd_res = estimate_wgd_timing(segs_out, settings=settings)

    return MutationTimeResult(
        variants_df=vars_out,
        segments_df=segs_out,
        segment_pis=segment_pis,
        wgd_result=wgd_res,
        purity=cfg.purity,
        gender=cfg.gender,
        settings=settings,
    )


class MutationTimeR:
    """
    Object-oriented estimator for somatic mutation multiplicity, copy-number
    gain timing, and Whole-Genome Doubling (`WGD`) analysis.
    """

    def __init__(
        self,
        purity: float | None = None,
        gender: str = "female",
        n_boot: int = 200,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings = validate_settings(settings)
        self.gender = normalize_gender(gender)
        if purity is not None and (not (0.0 < float(purity) <= 1.0) or not np.isfinite(purity)):
            raise ValueError(f"purity must be in (0, 1], got {purity!r}.")
        self.purity = float(purity) if purity is not None else None
        if n_boot < 0:
            raise ValueError(f"n_boot must be >= 0, got {n_boot!r}.")
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.result_: MutationTimeResult | None = None

    def fit(
        self,
        variants_df: pd.DataFrame,
        cn_segments_df: pd.DataFrame,
        purity: float | None = None,
        clusters_df: pd.DataFrame | None = None,
    ) -> MutationTimeResult:
        """Fit `MutationTimeR` on somatic variants and allele-specific segments."""
        eff_purity = purity if purity is not None else self.purity
        if eff_purity is None:
            raise ValueError(
                "purity must be provided either at MutationTimeR(...) initialization or in .fit(...)."
            )
        self.result_ = mutation_time(
            variants_df=variants_df,
            cn_segments_df=cn_segments_df,
            purity=eff_purity,
            clusters_df=clusters_df,
            gender=self.gender,
            n_boot=self.n_boot,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_

    def run(
        self,
        variants_df: pd.DataFrame,
        cn_segments_df: pd.DataFrame,
        purity: float | None = None,
        clusters_df: pd.DataFrame | None = None,
    ) -> MutationTimeResult:
        """Alias for `.fit(...)`."""
        return self.fit(
            variants_df=variants_df,
            cn_segments_df=cn_segments_df,
            purity=purity,
            clusters_df=clusters_df,
        )
