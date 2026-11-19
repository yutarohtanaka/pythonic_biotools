"""
Beta-Binomial somatic-vs-germline classification, allele-specific LOH,
allelic imbalance, and cancer cell fraction (CCF) estimation (`predict_somatic`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

from .config import SettingsMode, get_purecn_config, validate_settings
from .coverage import CoverageNormalizationResult
from .likelihood import (
    beta_binomial_logpmf,
    compute_somatic_priors,
    estimate_overdispersion,
    expected_log2_ratio,
    map_variants_to_segments,
    standardize_segments_df,
    standardize_variants_df,
)


def predict_somatic(
    variants_df: pd.DataFrame,
    segments_df: pd.DataFrame | CoverageNormalizationResult,
    purity: float,
    ploidy: float,
    base_error: float = 0.001,
    rho_overdisp: float | None = None,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Classify variants into somatic vs. germline states using PureCN's Beta-Binomial
    marginal likelihood over integer allele-specific copy-number states `(C, M)`,
    somatic multiplicities `m`, and subclonal cell-fraction grid.

    Returns a copy of `variants_df` augmented with:
    `prior_somatic`, `posterior_somatic`, `ml_somatic`, `ml_C`, `ml_M`,
    `ml_multiplicity`, `loh`, `allelic_imbalance`, `cell_fraction` (`CCF`).
    """
    cfg = get_purecn_config(validate_settings(settings))
    rho = float(purity)
    psi = float(ploidy)
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"`purity` must be in (0, 1], got {purity}.")
    if psi <= 0.0:
        raise ValueError(f"`ploidy` must be > 0, got {ploidy}.")
    if not (0.0 < base_error < 0.25):
        raise ValueError(f"`base_error` must be in (0, 0.25), got {base_error}.")

    if not isinstance(variants_df, pd.DataFrame):
        raise ValueError("`variants_df` must be a pandas DataFrame.")
    if variants_df.empty:
        empty = variants_df.copy()
        for col, dtype in (
            ("prior_somatic", float),
            ("posterior_somatic", float),
            ("ml_somatic", bool),
            ("ml_C", int),
            ("ml_M", int),
            ("ml_multiplicity", int),
            ("loh", bool),
            ("allelic_imbalance", float),
            ("cell_fraction", float),
        ):
            empty[col] = pd.Series(dtype=dtype)
        return empty

    segs = standardize_segments_df(segments_df)
    vars_std = standardize_variants_df(variants_df)
    n_var = len(vars_std)

    eff_rho_od = (
        float(rho_overdisp)
        if rho_overdisp is not None
        else (
            cfg.rho_overdisp
            if cfg.rho_overdisp is not None
            else estimate_overdispersion(vars_std, min_rho=cfg.min_rho_overdisp)
        )
    )
    eff_rho_od = max(eff_rho_od, cfg.min_rho_overdisp)

    priors_som = compute_somatic_priors(vars_std, cfg)
    seg_indices = map_variants_to_segments(vars_std, segs)

    max_cn = 8
    if "C" in segs.columns:
        max_cn = max(max_cn, int(np.max(segs["C"].to_numpy())))

    c_candidates = np.arange(max_cn + 1, dtype=np.int64)
    subclonal_grid = np.linspace(0.05, 0.95, 19, dtype=np.float64)

    # Step 1: Compute segment-level posterior over (C, M_minor) using segment log2 + germline SNPs on segment
    n_seg = len(segs)
    seg_cm_logpost: list[dict[tuple[int, int], float]] = []
    seg_best_cm: list[tuple[int, int]] = []

    k_arr = vars_std["alt_count"].to_numpy(dtype=np.float64)
    n_arr = vars_std["depth"].to_numpy(dtype=np.float64)
    af_arr = vars_std["af"].to_numpy(dtype=np.float64)

    for s in range(n_seg):
        y_s = float(segs.loc[s, "seg_mean"])
        n_m_bins = float(np.clip(segs.loc[s, "num_mark"], 3.0, 80.0))
        sd_s = max(float(segs.loc[s, "seg_sd"]), 0.08)
        se_s = sd_s / np.sqrt(n_m_bins)

        exp_l2 = expected_log2_ratio(c_candidates, rho, psi)
        cn_ll = -0.5 * ((y_s - exp_l2) / se_s) ** 2
        if "C" in segs.columns:
            called_c = int(segs.loc[s, "C"])
            # Anchor gently around segment's called C while permitting nearby C uncertainty
            cn_ll = cn_ll - 1.5 * np.abs(c_candidates - called_c)

        v_on_seg = np.flatnonzero(seg_indices == s)
        # Informative germline candidates on segment s (excluding high somatic prior sites)
        germ_on_seg = v_on_seg[priors_som[v_on_seg] < 0.45]

        cm_scores: dict[tuple[int, int], float] = {}
        for c_val in range(max_cn + 1):
            denom = max(rho * float(c_val) + 2.0 * (1.0 - rho), 1e-6)
            max_min = c_val // 2
            n_m_states = max_min + 1
            for m_min in range(n_m_states):
                m_maj = c_val - m_min
                base_prior = -np.log(float(n_m_states))
                if c_val >= 2 and n_m_states > 1 and m_min == 0:
                    base_prior -= 0.25

                snp_ll = 0.0
                if germ_on_seg.size > 0:
                    kg = k_arr[germ_on_seg]
                    ng = n_arr[germ_on_seg]
                    f1 = np.clip((rho * float(m_min) + (1.0 - rho)) / denom, base_error, 1.0 - base_error)
                    f2 = np.clip((rho * float(m_maj) + (1.0 - rho)) / denom, base_error, 1.0 - base_error)
                    ll_het = np.logaddexp(
                        np.log(0.5) + beta_binomial_logpmf(kg, ng, f1, eff_rho_od),
                        np.log(0.5) + beta_binomial_logpmf(kg, ng, f2, eff_rho_od),
                    )
                    ll_hom = beta_binomial_logpmf(kg, ng, 1.0 - base_error, eff_rho_od)
                    snp_ll = float(np.sum(np.logaddexp(np.log(0.94) + ll_het, np.log(0.06) + ll_hom)))

                cm_scores[(c_val, m_min)] = float(cn_ll[c_val]) + base_prior + snp_ll

        lse = float(special.logsumexp(list(cm_scores.values())))
        cm_post = {k: v - lse for k, v in cm_scores.items()}
        best_pair = max(cm_post.items(), key=lambda kv: kv[1])[0]
        seg_cm_logpost.append(cm_post)
        seg_best_cm.append(best_pair)

    # Step 2: Per-variant marginal likelihood over (C, M), somatic vs germline, and CCF
    post_som = np.zeros(n_var, dtype=np.float64)
    ml_som = np.zeros(n_var, dtype=bool)
    ml_c_out = np.zeros(n_var, dtype=np.int64)
    ml_m_out = np.zeros(n_var, dtype=np.int64)
    ml_mult_out = np.zeros(n_var, dtype=np.int64)
    loh_out = np.zeros(n_var, dtype=bool)
    imb_out = np.zeros(n_var, dtype=np.float64)
    ccf_out = np.zeros(n_var, dtype=np.float64)

    for i in range(n_var):
        ki = k_arr[i]
        ni = n_arr[i]
        afi = af_arr[i]
        pi_som = priors_som[i]
        s_idx = int(seg_indices[i])

        if s_idx >= 0:
            cm_prior_map = seg_cm_logpost[s_idx]
        else:
            dip_c = max(1, int(round(psi)))
            cm_prior_map = {(dip_c, dip_c // 2): 0.0}

        # Filter to non-negligible (C, M) states (log_post > -12) for speed
        active_cm = [(cm, lp) for cm, lp in cm_prior_map.items() if lp > -12.0]
        if not active_cm:
            best_cm = max(cm_prior_map.items(), key=lambda kv: kv[1])
            active_cm = [best_cm]

        som_terms: list[float] = []
        germ_terms: list[float] = []

        best_state_ll = -np.inf
        best_c_val = 2
        best_m_val = 1
        best_mult_val = 1
        best_ccf_val = 1.0

        for (c_val, m_min), log_p_cm in active_cm:
            m_maj = c_val - m_min
            denom = max(rho * float(c_val) + 2.0 * (1.0 - rho), 1e-6)

            # --- Somatic likelihood for (C, M_min) ---
            # Unfolded over M_allele in {m_min, m_maj}: somatic multiplicity m in 1..max(1, c_val)
            # Clonal states get weight 0.80, subclonal CCF grid gets weight 0.20
            local_som_terms: list[float] = []
            best_som_here = -np.inf
            best_som_mult = 1
            best_som_ccf = 1.0

            if c_val == 0:
                f_sub0 = np.clip((rho * 0.25) / denom, base_error, 1.0 - base_error)
                ll0 = float(beta_binomial_logpmf(ki, ni, f_sub0, eff_rho_od))
                local_som_terms.append(ll0)
                best_som_here = ll0
                best_som_mult = 0
                best_som_ccf = 0.0
            else:
                # Favor multiplicities <= m_maj (allele copy count ceiling), allow up to c_val
                max_plausible_m = max(1, m_maj)
                clonal_weights = np.array(
                    [1.0 if m <= max_plausible_m else 0.15 for m in range(1, c_val + 1)],
                    dtype=np.float64,
                )
                clonal_weights = 0.80 * (clonal_weights / np.sum(clonal_weights))
                for m_idx, m_mult in enumerate(range(1, c_val + 1)):
                    f_m = float(np.clip((rho * float(m_mult)) / denom, base_error, 1.0 - base_error))
                    ll_m = float(beta_binomial_logpmf(ki, ni, f_m, eff_rho_od))
                    term_m = float(np.log(clonal_weights[m_idx])) + ll_m
                    local_som_terms.append(term_m)
                    if term_m > best_som_here:
                        best_som_here = term_m
                        best_som_mult = m_mult
                        best_som_ccf = 1.0

                w_sub = 0.20 / float(subclonal_grid.size)
                log_w_sub = float(np.log(w_sub))
                f_subs = np.clip((rho * subclonal_grid) / denom, base_error, 1.0 - base_error)
                ll_subs = beta_binomial_logpmf(ki, ni, f_subs, eff_rho_od)
                for sub_j in range(subclonal_grid.size):
                    term_s = log_w_sub + float(ll_subs[sub_j])
                    local_som_terms.append(term_s)
                    if term_s > best_som_here:
                        best_som_here = term_s
                        best_som_mult = 1
                        best_som_ccf = float(subclonal_grid[sub_j])

            ll_som_cm = float(special.logsumexp(local_som_terms))
            som_terms.append(log_p_cm + ll_som_cm)

            # --- Germline likelihood for (C, M_min) ---
            f_g1 = float(
                np.clip((rho * float(m_min) + (1.0 - rho)) / denom, base_error, 1.0 - base_error)
            )
            f_g2 = float(
                np.clip((rho * float(m_maj) + (1.0 - rho)) / denom, base_error, 1.0 - base_error)
            )
            ll_g1 = float(beta_binomial_logpmf(ki, ni, f_g1, eff_rho_od))
            ll_g2 = float(beta_binomial_logpmf(ki, ni, f_g2, eff_rho_od))
            ll_het = float(np.logaddexp(np.log(0.5) + ll_g1, np.log(0.5) + ll_g2))
            ll_hom = float(beta_binomial_logpmf(ki, ni, 1.0 - base_error, eff_rho_od))
            ll_germ_cm = float(np.logaddexp(np.log(0.94) + ll_het, np.log(0.06) + ll_hom))
            germ_terms.append(log_p_cm + ll_germ_cm)

            # Track maximum-a-posteriori (C, M_min) joint with variant observation
            joint_var_cm = log_p_cm + float(
                np.logaddexp(
                    np.log(pi_som) + ll_som_cm,
                    np.log(1.0 - pi_som) + ll_germ_cm,
                )
            )
            if joint_var_cm > best_state_ll:
                best_state_ll = joint_var_cm
                best_c_val = c_val
                best_m_val = m_min
                if (np.log(pi_som) + ll_som_cm) >= (np.log(1.0 - pi_som) + ll_germ_cm):
                    best_mult_val = best_som_mult
                    # Continuous CCF estimate anchored to best somatic multiplicity
                    denom_best = max(rho * float(c_val) + 2.0 * (1.0 - rho), 1e-6)
                    u_copies = (afi * denom_best) / max(rho, 1e-6)
                    if best_som_mult >= 1:
                        best_ccf_val = float(np.clip(u_copies / float(best_som_mult), 0.0, 1.0))
                    else:
                        best_ccf_val = best_som_ccf
                else:
                    best_mult_val = m_maj if ll_g2 >= ll_g1 else m_min
                    best_ccf_val = 1.0

        log_num = float(np.log(pi_som)) + float(special.logsumexp(som_terms))
        log_den = float(
            np.logaddexp(
                log_num,
                float(np.log(1.0 - pi_som)) + float(special.logsumexp(germ_terms)),
            )
        )
        p_post = float(np.clip(np.exp(log_num - log_den), 0.0, 1.0))

        post_som[i] = p_post
        ml_som[i] = bool(p_post >= cfg.somatic_posterior_cutoff)
        ml_c_out[i] = best_c_val
        ml_m_out[i] = best_m_val
        ml_mult_out[i] = best_mult_val
        loh_out[i] = bool(best_m_val == 0)
        if best_c_val > 0:
            imb_out[i] = float(abs(best_c_val - 2 * best_m_val) / float(best_c_val))
        else:
            imb_out[i] = 1.0
        ccf_out[i] = best_ccf_val

    out = vars_std.copy()
    out["prior_somatic"] = priors_som
    out["posterior_somatic"] = post_som
    out["ml_somatic"] = ml_som
    out["ml_C"] = ml_c_out
    out["ml_M"] = ml_m_out
    out["ml_multiplicity"] = ml_mult_out
    out["loh"] = loh_out
    out["allelic_imbalance"] = imb_out
    out["cell_fraction"] = ccf_out
    return out
