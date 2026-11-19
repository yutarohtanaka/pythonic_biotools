"""Consensus segment calling, MLE refinement, and single-cell segment scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .config import COLLAPSED_STATES, NumbatConfig
from .expression import gaussian_logratio_logpdf, pln_logpmf
from .hmm import HMMResult, beta_binomial_logpmf, phase_hmm_loglik, phase_hmm_loglik_batch
from .io import OrderedLoci, PreparedAlleles, PreparedExpression

JOINT_POST_COLUMNS: list[str] = [
    "cell",
    "seg",
    "chrom",
    "start",
    "end",
    "cnv_state",
    "n_genes",
    "n_snps",
    "llr_x",
    "llr_y",
    "llr",
    "p_cnv_x",
    "p_cnv_y",
    "p_cnv",
]

SEGMENTS_COLUMNS: list[str] = [
    "seg",
    "chrom",
    "start",
    "end",
    "n_genes",
    "n_snps",
    "cnv_state",
    "cnv_state_post",
    "phi_mle",
    "theta_mle",
    "p_neu",
    "p_amp",
    "p_del",
    "p_loh",
    "p_bamp",
    "p_bdel",
    "entropy",
    "log_bf",
]


@dataclass(frozen=True)
class SegmentCall:
    """Consensus CNV segment call with MLEs and collapsed posterior probabilities."""

    seg_id: str
    chrom: str
    start: int
    end: int
    n_genes: int
    n_snps: int
    cnv_state: str
    cnv_state_post: str
    phi_mle: float
    theta_mle: float
    p_neu: float
    p_amp: float
    p_del: float
    p_loh: float
    p_bamp: float
    p_bdel: float
    entropy: float
    log_bf: float
    start_locus: int = 0
    end_locus: int = 0
    phi_state: float = 1.0
    theta_state: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return segment fields matching the standard consensus segment schema."""
        return {
            "seg": self.seg_id,
            "chrom": self.chrom,
            "start": int(self.start),
            "end": int(self.end),
            "n_genes": int(self.n_genes),
            "n_snps": int(self.n_snps),
            "cnv_state": self.cnv_state,
            "cnv_state_post": self.cnv_state_post,
            "phi_mle": float(self.phi_mle),
            "theta_mle": float(self.theta_mle),
            "p_neu": float(self.p_neu),
            "p_amp": float(self.p_amp),
            "p_del": float(self.p_del),
            "p_loh": float(self.p_loh),
            "p_bamp": float(self.p_bamp),
            "p_bdel": float(self.p_bdel),
            "entropy": float(self.entropy),
            "log_bf": float(self.log_bf),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return this segment call as a single-row DataFrame."""
        return pd.DataFrame([self.to_dict()], columns=SEGMENTS_COLUMNS)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = expit(arr)
    if np.ndim(x) == 0:
        return float(out)
    return out


def _canonical_state_params(
    cnv_state: str,
    seg_post_phased: np.ndarray,
    config: NumbatConfig,
) -> tuple[float, float]:
    """Select representative (phi_state, theta_state) for a collapsed CNV state."""
    matching_idx = [i for i, s in enumerate(config.states) if s.collapsed == cnv_state]
    if not matching_idx:
        return 1.0, 0.0
    best_i = max(matching_idx, key=lambda idx: float(seg_post_phased[idx]))
    spec = config.states[best_i]
    return float(spec.phi), float(abs(spec.mu - 0.5))


def extract_segments(
    hmm_result: HMMResult,
    loci: OrderedLoci,
    bulk_counts: np.ndarray,
    bulk_base_mu: np.ndarray,
    sigma_g: np.ndarray,
    bulk_ad: np.ndarray | None,
    bulk_dp: np.ndarray | None,
    p_switch: np.ndarray,
    config: NumbatConfig,
) -> tuple[SegmentCall, ...]:
    """Extract, refine (phi_mle, theta_mle), and filter non-neutral consensus CNV segments."""
    viterbi = hmm_result.viterbi_collapsed
    n_loci = loci.n_loci
    if n_loci == 0:
        return ()

    runs: list[tuple[int, int, str]] = []
    cur_state = viterbi[0]
    cur_chrom = str(loci.chrom[0])
    cur_start = 0

    for t in range(1, n_loci):
        c = str(loci.chrom[t])
        st = viterbi[t]
        if c != cur_chrom or st != cur_state:
            if cur_state != "neu":
                runs.append((cur_start, t, cur_state))
            cur_chrom = c
            cur_state = st
            cur_start = t
    if cur_state != "neu":
        runs.append((cur_start, n_loci, cur_state))

    non_neu_states = ("amp", "del", "loh", "bamp", "bdel")
    collapsed_idx = {name: i for i, name in enumerate(COLLAPSED_STATES)}
    calls: list[SegmentCall] = []
    seg_counter_per_chrom: dict[str, int] = {}

    for s_idx, e_idx, state in runs:
        chrom = str(loci.chrom[s_idx])
        gene_mask = loci.has_expr[s_idx:e_idx]
        snp_mask = loci.has_allele[s_idx:e_idx]
        n_genes = int(np.sum(gene_mask))
        n_snps = int(np.sum(snp_mask))
        if n_genes + n_snps < config.min_loci:
            continue

        mean_col = np.mean(hmm_result.collapsed_posteriors[s_idx:e_idx], axis=0)
        mean_col = mean_col / max(float(np.sum(mean_col)), 1e-12)
        p_neu = float(mean_col[collapsed_idx["neu"]])
        p_amp = float(mean_col[collapsed_idx["amp"]])
        p_del = float(mean_col[collapsed_idx["del"]])
        p_loh = float(mean_col[collapsed_idx["loh"]])
        p_bamp = float(mean_col[collapsed_idx["bamp"]])
        p_bdel = float(mean_col[collapsed_idx["bdel"]])

        if (1.0 - p_neu) < config.min_seg_prob:
            continue

        entropy = float(-np.sum(mean_col * np.log(np.maximum(mean_col, 1e-12))))
        if n_snps == 0:
            # Expression alone cannot separate equal-phi states (neu/loh or amp2/bamp)
            expr_regimes = np.array(
                [p_neu + p_loh, p_del, p_amp + p_bamp, p_bdel],
                dtype=np.float64,
            )
            eff_entropy = float(-np.sum(expr_regimes * np.log(np.maximum(expr_regimes, 1e-12))))
        else:
            eff_entropy = entropy
        if eff_entropy > config.max_entropy:
            continue

        cnv_state_post = max(
            non_neu_states,
            key=lambda st: float(mean_col[collapsed_idx[st]]),
        )

        seg_post_phased = np.mean(hmm_result.posteriors[s_idx:e_idx], axis=0)
        phi_state, theta_state = _canonical_state_params(state, seg_post_phased, config)

        # Bulk expression likelihood & phi_mle
        g_indices = loci.gene_index[s_idx:e_idx][gene_mask]
        if n_genes > 0 and np.any(np.isfinite(bulk_base_mu[g_indices])):
            xg = bulk_counts[g_indices]
            mug = bulk_base_mu[g_indices]
            sg = sigma_g[g_indices]
            if config.expr_model == "pln":
                ll_x_state = float(
                    np.sum(
                        pln_logpmf(
                            xg, mug + math.log(phi_state), sg, n_quadrature=config.n_quadrature
                        )
                    )
                )
                ll_x_neu = float(np.sum(pln_logpmf(xg, mug, sg, n_quadrature=config.n_quadrature)))

                def _neg_ll_phi(phi_val: float) -> float:
                    return -float(
                        np.sum(
                            pln_logpmf(
                                xg,
                                mug + math.log(phi_val),
                                sg,
                                n_quadrature=config.n_quadrature,
                            )
                        )
                    )

            else:
                ll_x_state = float(np.sum(gaussian_logratio_logpdf(xg, phi_state, sg)))
                ll_x_neu = float(np.sum(gaussian_logratio_logpdf(xg, 1.0, sg)))

                def _neg_ll_phi(phi_val: float) -> float:
                    return -float(np.sum(gaussian_logratio_logpdf(xg, phi_val, sg)))

            opt_phi = minimize_scalar(_neg_ll_phi, bounds=(0.1, 5.0), method="bounded")
            phi_mle = float(opt_phi.x)
            llr_x = ll_x_state - ll_x_neu
        else:
            phi_mle = phi_state
            llr_x = 0.0

        # Bulk allele likelihood & theta_mle
        if n_snps > 0 and bulk_ad is not None and bulk_dp is not None:
            snp_indices = loci.snp_index[s_idx:e_idx][snp_mask]
            ad_s = bulk_ad[snp_indices]
            dp_s = bulk_dp[snp_indices]
            ps_s = p_switch[s_idx:e_idx][snp_mask].copy()
            ps_s[0] = 0.0
            if float(np.sum(dp_s)) > 0.0:
                ll_y_state = phase_hmm_loglik(ad_s, dp_s, ps_s, theta_state, config.gamma)
                ll_y_neu = float(np.sum(beta_binomial_logpmf(ad_s, dp_s, 0.5, config.gamma)))

                def _neg_ll_theta(theta_val: float) -> float:
                    return -phase_hmm_loglik(ad_s, dp_s, ps_s, theta_val, config.gamma)

                opt_theta = minimize_scalar(_neg_ll_theta, bounds=(0.0, 0.495), method="bounded")
                theta_mle = float(opt_theta.x)
                llr_y = ll_y_state - ll_y_neu
            else:
                theta_mle = theta_state
                llr_y = 0.0
        else:
            theta_mle = theta_state
            llr_y = 0.0

        log_bf = config.w_expr * llr_x + config.w_allele * llr_y
        if log_bf < config.min_log_bf:
            continue

        seg_num = seg_counter_per_chrom.get(chrom, 0) + 1
        seg_counter_per_chrom[chrom] = seg_num
        seg_id = f"{chrom}_{seg_num}_{state}"

        calls.append(
            SegmentCall(
                seg_id=seg_id,
                chrom=chrom,
                start=int(loci.start[s_idx]),
                end=int(loci.end[e_idx - 1]),
                n_genes=n_genes,
                n_snps=n_snps,
                cnv_state=state,
                cnv_state_post=cnv_state_post,
                phi_mle=phi_mle,
                theta_mle=theta_mle,
                p_neu=p_neu,
                p_amp=p_amp,
                p_del=p_del,
                p_loh=p_loh,
                p_bamp=p_bamp,
                p_bdel=p_bdel,
                entropy=entropy,
                log_bf=float(log_bf),
                start_locus=int(s_idx),
                end_locus=int(e_idx),
                phi_state=phi_state,
                theta_state=theta_state,
            )
        )

    return tuple(calls)


def score_single_cells(
    segments: Sequence[SegmentCall],
    loci: OrderedLoci,
    expr: PreparedExpression,
    alleles: PreparedAlleles | None,
    ref_lambda: np.ndarray,
    sigma_g: np.ndarray,
    lib_sizes: np.ndarray,
    diploid_offsets: np.ndarray,
    p_switch: np.ndarray,
    config: NumbatConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell, per-segment joint posteriors (`joint_post`) and LLR matrices."""
    n_cells = len(expr.cell_ids)
    n_segs = len(segments)
    if n_segs == 0:
        empty_df = pd.DataFrame(columns=JOINT_POST_COLUMNS)
        return (
            empty_df,
            np.zeros((n_cells, 0), dtype=np.float64),
            np.zeros((n_cells, 0), dtype=np.float64),
            np.zeros(n_cells, dtype=np.float64),
        )

    prob_mat = np.empty((n_cells, n_segs), dtype=np.float64)
    llr_mat = np.empty((n_cells, n_segs), dtype=np.float64)
    weights = np.empty(n_segs, dtype=np.float64)
    logit_prior = math.log(config.seg_prior / (1.0 - config.seg_prior))

    valid_cells = lib_sizes > 0.0
    log_lib = np.zeros(n_cells, dtype=np.float64)
    log_lib[valid_cells] = np.log(lib_sizes[valid_cells]) + diploid_offsets[valid_cells]

    rows: list[pd.DataFrame] = []
    for s_col, seg in enumerate(segments):
        s_idx, e_idx = seg.start_locus, seg.end_locus
        weights[s_col] = float(max(1, seg.n_genes + seg.n_snps))

        # Expression LLR per cell
        gene_mask = loci.has_expr[s_idx:e_idx]
        g_indices = loci.gene_index[s_idx:e_idx][gene_mask]
        llr_x = np.zeros(n_cells, dtype=np.float64)
        if g_indices.size > 0 and abs(seg.phi_state - 1.0) > 1e-8:
            if expr.is_sparse:
                x_seg = expr.matrix[:, g_indices].toarray()
            else:
                x_seg = expr.matrix[:, g_indices]
            sig_seg = sigma_g[g_indices]
            if config.expr_model == "pln":
                if np.any(valid_cells):
                    base_mu = (
                        log_lib[valid_cells, None]
                        + np.log(np.maximum(ref_lambda[g_indices], 1e-12))[None, :]
                    )
                    ll_cnv = np.sum(
                        pln_logpmf(
                            x_seg[valid_cells],
                            base_mu + math.log(seg.phi_state),
                            sig_seg[None, :],
                            n_quadrature=config.n_quadrature,
                        ),
                        axis=1,
                    )
                    ll_neu = np.sum(
                        pln_logpmf(
                            x_seg[valid_cells],
                            base_mu,
                            sig_seg[None, :],
                            n_quadrature=config.n_quadrature,
                        ),
                        axis=1,
                    )
                    llr_x[valid_cells] = config.w_expr * (ll_cnv - ll_neu)
            else:
                x_centered = x_seg - ref_lambda[g_indices][None, :]
                ll_cnv = np.sum(
                    gaussian_logratio_logpdf(x_centered, seg.phi_state, sig_seg[None, :]), axis=1
                )
                ll_neu = np.sum(
                    gaussian_logratio_logpdf(x_centered, 1.0, sig_seg[None, :]), axis=1
                )
                llr_x = config.w_expr * (ll_cnv - ll_neu)

        # Allele LLR per cell
        snp_mask = loci.has_allele[s_idx:e_idx]
        snp_indices = loci.snp_index[s_idx:e_idx][snp_mask]
        llr_y = np.zeros(n_cells, dtype=np.float64)
        if snp_indices.size > 0 and alleles is not None and seg.theta_state > 1e-8:
            ad_seg = alleles.ad_mat[:, snp_indices].toarray()
            dp_seg = alleles.dp_mat[:, snp_indices].toarray()
            ps_seg = p_switch[s_idx:e_idx][snp_mask].copy()
            ps_seg[0] = 0.0
            has_reads = np.sum(dp_seg, axis=1) > 0.0
            if np.any(has_reads):
                ll_y_cnv = phase_hmm_loglik_batch(
                    ad_seg[has_reads], dp_seg[has_reads], ps_seg, seg.theta_state, config.gamma
                )
                ll_y_neu = np.sum(
                    beta_binomial_logpmf(ad_seg[has_reads], dp_seg[has_reads], 0.5, config.gamma),
                    axis=1,
                )
                llr_y[has_reads] = config.w_allele * (ll_y_cnv - ll_y_neu)

        llr_tot = llr_x + llr_y
        p_x = _sigmoid(logit_prior + llr_x)
        p_y = _sigmoid(logit_prior + llr_y)
        p_tot = _sigmoid(logit_prior + llr_tot)

        llr_mat[:, s_col] = llr_tot
        prob_mat[:, s_col] = p_tot

        rows.append(
            pd.DataFrame(
                {
                    "cell": list(expr.cell_ids),
                    "seg": seg.seg_id,
                    "chrom": seg.chrom,
                    "start": seg.start,
                    "end": seg.end,
                    "cnv_state": seg.cnv_state,
                    "n_genes": seg.n_genes,
                    "n_snps": seg.n_snps,
                    "llr_x": llr_x,
                    "llr_y": llr_y,
                    "llr": llr_tot,
                    "p_cnv_x": p_x,
                    "p_cnv_y": p_y,
                    "p_cnv": p_tot,
                },
                columns=JOINT_POST_COLUMNS,
            )
        )

    joint_post = pd.concat(rows, ignore_index=True)
    aneuploidy_score = (prob_mat @ weights) / float(np.sum(weights))
    return joint_post, prob_mat, llr_mat, aneuploidy_score
