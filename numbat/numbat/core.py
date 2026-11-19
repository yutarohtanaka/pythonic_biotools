"""High-level `run_numbat` pipeline, `NumbatResult` dataclass, and `AnnData` integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans

from .clones import (
    JOINT_POST_COLUMNS,
    SEGMENTS_COLUMNS,
    SegmentCall,
    extract_segments,
    score_single_cells,
)
from .config import NumbatConfig
from .expression import (
    compute_diploid_offset,
    compute_diploid_offsets_batch,
    compute_ref_expression,
    gaussian_logratio_logpdf,
    pln_logpmf,
)
from .hmm import (
    HMMResult,
    beta_binomial_logpmf,
    phase_switch_prob,
    resolve_backend,
    run_joint_hmm,
)
from .io import (
    OrderedLoci,
    PreparedAlleles,
    PreparedExpression,
    build_ordered_loci,
    canonical_chrom_key,
    extract_expression_input,
    parse_allele_dataframe,
)
from .phylogeny import CloneAssignment, PhylogenyTree, reconstruct_phylogeny


@dataclass(frozen=True)
class NumbatResult:
    """Complete typed result container returned by `run_numbat`."""

    segments: tuple[SegmentCall, ...]
    clones: CloneAssignment
    phylogeny: PhylogenyTree
    joint_post: pd.DataFrame
    locus_post: pd.DataFrame
    config: NumbatConfig
    backend_used: str

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-cell clone assignments and malignancy probabilities."""
        return self.clones.to_dataframe()

    def clones_dataframe(self) -> pd.DataFrame:
        """Return per-clone summary table (clone_id, compartment, n_cells, fraction, genotype)."""
        cols = ["clone_id", "compartment", "n_cells", "fraction", "mean_p_cnv", "genotype"]
        n_cells_total = max(1, len(self.clones.cell_ids))
        rows = []
        for idx, cid in enumerate(self.clones.clone_ids):
            mask = self.clones.clone_opt == cid
            n_c = int(np.sum(mask))
            mean_p = float(np.mean(self.clones.p_cnv[mask])) if n_c > 0 else 0.0
            comp = "normal" if cid == 1 else "tumor"
            geno_bits = self.clones.clone_genotypes[idx]
            active_events = [
                self.clones.event_names[m]
                for m in range(len(self.clones.event_names))
                if geno_bits[m] == 1
            ]
            rows.append(
                {
                    "clone_id": int(cid),
                    "compartment": comp,
                    "n_cells": n_c,
                    "fraction": float(n_c) / float(n_cells_total),
                    "mean_p_cnv": mean_p,
                    "genotype": ",".join(active_events) if active_events else "diploid",
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def segments_dataframe(self) -> pd.DataFrame:
        """Return consensus CNV segments as a tidy DataFrame with fixed column schema."""
        if not self.segments:
            return pd.DataFrame(columns=SEGMENTS_COLUMNS)
        return pd.DataFrame([s.to_dict() for s in self.segments], columns=SEGMENTS_COLUMNS)

    def joint_post_dataframe(self) -> pd.DataFrame:
        """Return per-cell, per-segment joint posteriors (`p_cnv_x`, `p_cnv_y`, `p_cnv`)."""
        if self.joint_post.empty:
            return pd.DataFrame(columns=JOINT_POST_COLUMNS)
        return self.joint_post.copy()

    def summary(self) -> str:
        """Return a multi-line human-readable summary of CNV calls, clones, and phylogeny."""
        n_cells = len(self.clones.cell_ids)
        n_tumor = sum(1 for c in self.clones.compartment_opt if c == "tumor")
        n_normal = n_cells - n_tumor
        seg_summary = (
            ", ".join(f"{s.seg_id}({s.cnv_state})" for s in self.segments)
            if self.segments
            else "none"
        )
        return (
            f"NumbatResult Summary\n"
            f"  Cells: {n_cells} (tumor={n_tumor}, normal={n_normal})\n"
            f"  Clones: {len(self.clones.clone_ids)}\n"
            f"  Consensus segments ({len(self.segments)}): {seg_summary}\n"
            f"  Phylogeny LL: init={self.phylogeny.initial_log_likelihood:.2f}, "
            f"opt={self.phylogeny.log_likelihood:.2f}\n"
            f"  Backend: {self.backend_used}"
        )


def _compute_pseudobulk_hmm(
    cell_indices: np.ndarray,
    expr: PreparedExpression,
    alleles: PreparedAlleles | None,
    loci: OrderedLoci,
    ref_lambda: np.ndarray,
    sigma_g: np.ndarray,
    p_switch: np.ndarray,
    config: NumbatConfig,
) -> tuple[HMMResult, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build pooled pseudobulk emissions for `cell_indices` and run joint HMM."""
    n_cells_sub = len(cell_indices)
    if expr.is_sparse:
        sub_x = expr.matrix[cell_indices]
        bulk_counts = np.asarray(sub_x.sum(axis=0), dtype=np.float64).ravel()
    else:
        bulk_counts = np.sum(expr.matrix[cell_indices], axis=0, dtype=np.float64)

    if config.expr_model == "pln":
        lib_bulk = float(np.sum(bulk_counts))
        if lib_bulk > 0.0:
            delta_bulk = compute_diploid_offset(bulk_counts, ref_lambda, lib_size=lib_bulk)
            bulk_base_mu = np.log(np.maximum(lib_bulk * ref_lambda, 1e-12)) + delta_bulk
        else:
            bulk_base_mu = np.full(len(ref_lambda), np.nan, dtype=np.float64)
    else:
        lib_bulk = 0.0
        bulk_counts = bulk_counts / float(max(1, n_cells_sub)) - ref_lambda
        bulk_base_mu = np.zeros(len(ref_lambda), dtype=np.float64)

    if alleles is not None:
        bulk_ad = np.asarray(alleles.ad_mat[cell_indices].sum(axis=0), dtype=np.float64).ravel()
        bulk_dp = np.asarray(alleles.dp_mat[cell_indices].sum(axis=0), dtype=np.float64).ravel()
    else:
        bulk_ad = None
        bulk_dp = None

    n_states = len(config.states)
    log_emit = np.zeros((loci.n_loci, n_states), dtype=np.float64)
    phi_vec = np.array([s.phi for s in config.states], dtype=np.float64)
    mu_vec = np.array([s.mu for s in config.states], dtype=np.float64)

    expr_mask = loci.has_expr
    if np.any(expr_mask) and config.w_expr > 0.0:
        g_idx = loci.gene_index[expr_mask]
        if config.expr_model == "pln":
            if lib_bulk > 0.0:
                log_emit[expr_mask] = config.w_expr * pln_logpmf(
                    bulk_counts[g_idx, None],
                    bulk_base_mu[g_idx, None] + np.log(phi_vec)[None, :],
                    sigma_g[g_idx, None],
                    n_quadrature=config.n_quadrature,
                )
        else:
            log_emit[expr_mask] = config.w_expr * gaussian_logratio_logpdf(
                bulk_counts[g_idx, None],
                phi_vec[None, :],
                sigma_g[g_idx, None],
            )

    allele_mask = loci.has_allele
    if np.any(allele_mask) and bulk_ad is not None and bulk_dp is not None and config.w_allele > 0.0:
        s_idx = loci.snp_index[allele_mask]
        log_emit[allele_mask] = config.w_allele * beta_binomial_logpmf(
            bulk_ad[s_idx, None],
            bulk_dp[s_idx, None],
            mu_vec[None, :],
            config.gamma,
        )

    hmm_res = run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=loci.chrom_offsets,
        p_switch=p_switch,
        t_cnv=config.t_cnv,
        p_init_cnv=config.p_init_cnv,
        states=config.states,
        backend=config.backend,
        locus_id=loci.locus_id,
        chrom=loci.chrom.tolist(),
        pos=loci.pos,
    )
    return hmm_res, bulk_counts, bulk_base_mu, bulk_ad, bulk_dp


def _initial_cell_clusters(
    non_ref_indices: np.ndarray,
    expr: PreparedExpression,
    alleles: PreparedAlleles | None,
    ref_lambda: np.ndarray,
    lib_sizes: np.ndarray,
    diploid_offsets: np.ndarray,
    init_k: int = 3,
    expr_model: str = "pln",
) -> list[np.ndarray]:
    """Partition non-reference cells into up to `init_k` coarse clusters for subclonal sensitivity."""
    n_sub = len(non_ref_indices)
    if n_sub < 15 or init_k <= 1:
        return [non_ref_indices]

    chroms = sorted(set(str(c) for c in expr.gene_chrom), key=canonical_chrom_key)
    feats: list[np.ndarray] = []

    snp_chrom_keys = (
        [canonical_chrom_key(str(sc)) for sc in alleles.snp_chrom]
        if alleles is not None
        else []
    )

    for c in chroms:
        c_key = canonical_chrom_key(c)
        g_idx = np.flatnonzero(expr.gene_chrom == c)
        if g_idx.size > 0:
            if expr_model == "pln":
                if expr.is_sparse:
                    x_c = np.asarray(
                        expr.matrix[non_ref_indices][:, g_idx].sum(axis=1), dtype=np.float64
                    ).ravel()
                else:
                    x_c = np.sum(
                        expr.matrix[np.ix_(non_ref_indices, g_idx)], axis=1, dtype=np.float64
                    )
                expected = np.maximum(
                    lib_sizes[non_ref_indices] * float(np.sum(ref_lambda[g_idx])),
                    1e-6,
                )
                lr = np.log((x_c + 1.0) / (expected + 1.0)) - diploid_offsets[non_ref_indices]
            else:
                if expr.is_sparse:
                    mean_c = np.asarray(
                        expr.matrix[non_ref_indices][:, g_idx].mean(axis=1), dtype=np.float64
                    ).ravel()
                else:
                    mean_c = np.mean(
                        expr.matrix[np.ix_(non_ref_indices, g_idx)], axis=1, dtype=np.float64
                    )
                lr = mean_c - float(np.mean(ref_lambda[g_idx]))
            feats.append(lr)

        if alleles is not None:
            s_idx = np.array(
                [i for i, sk in enumerate(snp_chrom_keys) if sk == c_key],
                dtype=np.int64,
            )
            if s_idx.size > 0:
                ad_c = alleles.ad_mat[non_ref_indices][:, s_idx].toarray()
                dp_c = alleles.dp_mat[non_ref_indices][:, s_idx].toarray()
                with np.errstate(divide="ignore", invalid="ignore"):
                    dev = np.where(dp_c > 0.0, np.abs(ad_c / np.maximum(dp_c, 1.0) - 0.5), np.nan)
                has_obs = np.any(dp_c > 0.0, axis=1)
                mean_dev = np.zeros(n_sub, dtype=np.float64)
                if np.any(has_obs):
                    mean_dev[has_obs] = np.nanmean(dev[has_obs], axis=1)
                feats.append(mean_dev)

    if not feats:
        return [non_ref_indices]

    feat_mat = np.column_stack(feats)
    col_std = np.std(feat_mat, axis=0)
    valid_cols = col_std > 1e-8
    if not np.any(valid_cols):
        return [non_ref_indices]

    feat_norm = (feat_mat[:, valid_cols] - np.mean(feat_mat[:, valid_cols], axis=0)) / col_std[valid_cols]
    k_eff = min(init_k, max(2, n_sub // 10))
    km = KMeans(n_clusters=k_eff, random_state=0, n_init=5)
    labels = km.fit_predict(feat_norm)
    clusters = [non_ref_indices[labels == cid] for cid in range(k_eff) if np.sum(labels == cid) >= 5]
    return clusters if clusters else [non_ref_indices]


def _merge_candidate_segments(
    candidates: Sequence[SegmentCall],
) -> tuple[SegmentCall, ...]:
    """Deduplicate overlapping segments across pseudobulk runs, retaining highest log_bf."""
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda s: (-s.log_bf, s.chrom, s.start_locus))
    kept: list[SegmentCall] = []

    for cand in ordered:
        overlaps = False
        cand_len = max(1, cand.end_locus - cand.start_locus)
        for existing in kept:
            if existing.chrom != cand.chrom:
                continue
            ov = min(cand.end_locus, existing.end_locus) - max(cand.start_locus, existing.start_locus)
            if ov <= 0:
                continue
            ex_len = max(1, existing.end_locus - existing.start_locus)
            if ov / float(min(cand_len, ex_len)) >= 0.3:
                overlaps = True
                break
        if not overlaps:
            kept.append(cand)

    # Sort deterministically by genomic locus order and assign unique sequential IDs
    kept.sort(key=lambda s: (s.start_locus, s.end_locus))
    per_chrom: dict[str, int] = {}
    final_calls: list[SegmentCall] = []
    for seg in kept:
        num = per_chrom.get(seg.chrom, 0) + 1
        per_chrom[seg.chrom] = num
        final_calls.append(
            SegmentCall(
                seg_id=f"{seg.chrom}_{num}_{seg.cnv_state}",
                chrom=seg.chrom,
                start=seg.start,
                end=seg.end,
                n_genes=seg.n_genes,
                n_snps=seg.n_snps,
                cnv_state=seg.cnv_state,
                cnv_state_post=seg.cnv_state_post,
                phi_mle=seg.phi_mle,
                theta_mle=seg.theta_mle,
                p_neu=seg.p_neu,
                p_amp=seg.p_amp,
                p_del=seg.p_del,
                p_loh=seg.p_loh,
                p_bamp=seg.p_bamp,
                p_bdel=seg.p_bdel,
                entropy=seg.entropy,
                log_bf=seg.log_bf,
                start_locus=seg.start_locus,
                end_locus=seg.end_locus,
                phi_state=seg.phi_state,
                theta_state=seg.theta_state,
            )
        )
    return tuple(final_calls)


def run_numbat(
    data: ad.AnnData | pd.DataFrame | np.ndarray | sparse.spmatrix,
    allele_df: pd.DataFrame | None = None,
    gene_info: pd.DataFrame | None = None,
    ref_cells: str | Sequence[str] | Sequence[int] | np.ndarray | pd.Series | None = None,
    ref_expression: np.ndarray | pd.Series | None = None,
    ref_key: str | None = None,
    ref_label: Any = None,
    layer: str | None = None,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    init_k: int = 3,
    config: NumbatConfig | None = None,
) -> NumbatResult:
    """Run full haplotype-aware Numbat single-cell CNV, malignancy, and phylogeny analysis."""
    cfg = config if config is not None else NumbatConfig()
    resolved_backend = resolve_backend(cfg.backend)

    if allele_df is None and isinstance(data, ad.AnnData) and "numbat_alleles" in data.uns:
        allele_df = data.uns["numbat_alleles"]

    expr = extract_expression_input(
        data=data,
        gene_info=gene_info,
        layer=layer,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        ref_cells=ref_cells,
        ref_key=ref_key,
        ref_label=ref_label,
        require_non_negative=(cfg.expr_model == "pln"),
    )
    alleles = parse_allele_dataframe(allele_df, expr.cell_ids)
    loci = build_ordered_loci(expr, alleles)

    p_switch = np.asarray(phase_switch_prob(loci.dist_cm, nu=cfg.nu), dtype=np.float64)
    for c_idx in range(len(loci.chrom_offsets) - 1):
        p_switch[int(loci.chrom_offsets[c_idx])] = 0.0

    ref_expr_vec = (
        ref_expression.to_numpy(dtype=np.float64)
        if isinstance(ref_expression, pd.Series)
        else ref_expression
    )
    ref_lambda, sigma_g = compute_ref_expression(
        counts=expr.matrix,
        ref_indices=expr.ref_indices,
        ref_expression=ref_expr_vec,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        default_sigma=cfg.default_sigma,
        expr_model=cfg.expr_model,
    )

    n_cells = len(expr.cell_ids)
    if cfg.expr_model == "pln":
        if expr.is_sparse:
            lib_sizes = np.asarray(expr.matrix.sum(axis=1), dtype=np.float64).ravel()
        else:
            lib_sizes = np.sum(expr.matrix, axis=1, dtype=np.float64)
        diploid_offsets = compute_diploid_offsets_batch(expr.matrix, ref_lambda, lib_sizes)
    else:
        lib_sizes = np.zeros(n_cells, dtype=np.float64)
        diploid_offsets = np.zeros(n_cells, dtype=np.float64)

    if expr.ref_indices is not None and len(expr.ref_indices) < n_cells:
        ref_set = set(int(i) for i in expr.ref_indices)
        non_ref_indices = np.array([i for i in range(n_cells) if i not in ref_set], dtype=np.int64)
    else:
        non_ref_indices = np.arange(n_cells, dtype=np.int64)

    # Round 1: Global pseudobulk + initial subclonal cluster pseudobulks
    global_hmm, bulk_counts, bulk_base_mu, bulk_ad, bulk_dp = _compute_pseudobulk_hmm(
        non_ref_indices, expr, alleles, loci, ref_lambda, sigma_g, p_switch, cfg
    )
    cand_segments: list[SegmentCall] = list(
        extract_segments(
            global_hmm,
            loci,
            bulk_counts,
            bulk_base_mu,
            sigma_g,
            bulk_ad,
            bulk_dp,
            p_switch,
            cfg,
        )
    )

    if cfg.max_iter >= 2 and init_k >= 2 and len(non_ref_indices) >= 15:
        clusters = _initial_cell_clusters(
            non_ref_indices,
            expr,
            alleles,
            ref_lambda,
            lib_sizes,
            diploid_offsets,
            init_k=init_k,
            expr_model=cfg.expr_model,
        )
        if len(clusters) >= 2:
            for cl_idx in clusters:
                cl_hmm, cl_counts, cl_mu, cl_ad, cl_dp = _compute_pseudobulk_hmm(
                    cl_idx, expr, alleles, loci, ref_lambda, sigma_g, p_switch, cfg
                )
                cl_segs = extract_segments(
                    cl_hmm, loci, cl_counts, cl_mu, sigma_g, cl_ad, cl_dp, p_switch, cfg
                )
                cand_segments.extend(cl_segs)

    segments = _merge_candidate_segments(cand_segments)
    joint_post, prob_mat, llr_mat, aneuploidy_score = score_single_cells(
        segments=segments,
        loci=loci,
        expr=expr,
        alleles=alleles,
        ref_lambda=ref_lambda,
        sigma_g=sigma_g,
        lib_sizes=lib_sizes,
        diploid_offsets=diploid_offsets,
        p_switch=p_switch,
        config=cfg,
    )

    # Filter any subclonal-only segment lacking meaningful single-cell support (null protection)
    eff_min_cells = (
        int(cfg.min_cells)
        if cfg.min_cells is not None
        else max(3, min(50, n_cells // 10))
    )
    if n_cells < 6:
        eff_min_cells = min(eff_min_cells, 1)

    if segments:
        keep_mask = np.array(
            [
                int(np.sum(prob_mat[:, j] >= 0.65)) >= max(2, min(eff_min_cells, n_cells))
                and float(np.sum(np.maximum(llr_mat[:, j], 0.0))) >= cfg.min_log_bf
                for j in range(len(segments))
            ],
            dtype=bool,
        )
        if not np.all(keep_mask):
            segments = tuple(seg for seg, keep in zip(segments, keep_mask) if keep)
            joint_post, prob_mat, llr_mat, aneuploidy_score = score_single_cells(
                segments=segments,
                loci=loci,
                expr=expr,
                alleles=alleles,
                ref_lambda=ref_lambda,
                sigma_g=sigma_g,
                lib_sizes=lib_sizes,
                diploid_offsets=diploid_offsets,
                p_switch=p_switch,
                config=cfg,
            )

    phylogeny, clones = reconstruct_phylogeny(
        prob_mat=prob_mat,
        llr_mat=llr_mat,
        cell_ids=expr.cell_ids,
        event_names=[s.seg_id for s in segments],
        aneuploidy_score=aneuploidy_score,
        tau=cfg.tau,
        min_cells=cfg.min_cells,
        max_tree_cells=cfg.max_tree_cells,
        max_nni_iter=cfg.max_nni_iter,
        tumor_prob_threshold=cfg.tumor_prob_threshold,
    )

    # Optional second refinement pass on discovered tumor clones
    if cfg.max_iter >= 2 and len(segments) > 0 and len(clones.clone_ids) > 1:
        tumor_mask = np.array([c == "tumor" for c in clones.compartment_opt], dtype=bool)
        if int(np.sum(tumor_mask)) >= max(5, eff_min_cells):
            refined_cands: list[SegmentCall] = []
            for cid in clones.clone_ids:
                if cid == 1:
                    continue
                c_idx = np.flatnonzero(clones.clone_opt == cid)
                if c_idx.size >= max(5, eff_min_cells):
                    cl_hmm, cl_counts, cl_mu, cl_ad, cl_dp = _compute_pseudobulk_hmm(
                        c_idx, expr, alleles, loci, ref_lambda, sigma_g, p_switch, cfg
                    )
                    refined_cands.extend(
                        extract_segments(
                            cl_hmm, loci, cl_counts, cl_mu, sigma_g, cl_ad, cl_dp, p_switch, cfg
                        )
                    )
            if refined_cands:
                segments_ref = _merge_candidate_segments(refined_cands)
                if segments_ref:
                    segments = segments_ref
                    joint_post, prob_mat, llr_mat, aneuploidy_score = score_single_cells(
                        segments=segments,
                        loci=loci,
                        expr=expr,
                        alleles=alleles,
                        ref_lambda=ref_lambda,
                        sigma_g=sigma_g,
                        lib_sizes=lib_sizes,
                        diploid_offsets=diploid_offsets,
                        p_switch=p_switch,
                        config=cfg,
                    )
                    phylogeny, clones = reconstruct_phylogeny(
                        prob_mat=prob_mat,
                        llr_mat=llr_mat,
                        cell_ids=expr.cell_ids,
                        event_names=[s.seg_id for s in segments],
                        aneuploidy_score=aneuploidy_score,
                        tau=cfg.tau,
                        min_cells=cfg.min_cells,
                        max_tree_cells=cfg.max_tree_cells,
                        max_nni_iter=cfg.max_nni_iter,
                        tumor_prob_threshold=cfg.tumor_prob_threshold,
                    )

    return NumbatResult(
        segments=segments,
        clones=clones,
        phylogeny=phylogeny,
        joint_post=joint_post,
        locus_post=global_hmm.to_dataframe(),
        config=cfg,
        backend_used=resolved_backend,
    )


def annotate_anndata(
    adata: ad.AnnData,
    result: NumbatResult,
    inplace: bool = False,
) -> ad.AnnData:
    """Attach Numbat clone assignments, malignancy scores, segments, and phylogeny to AnnData."""
    target = adata if inplace else adata.copy()
    clone_df = result.to_dataframe().set_index("cell")
    obs_names = [str(c) for c in target.obs_names]
    missing = [c for c in obs_names if c not in clone_df.index]
    if missing:
        raise ValueError(f"AnnData obs_names missing from NumbatResult: {missing[:5]}.")

    aligned = clone_df.loc[obs_names]
    target.obs["clone_opt"] = aligned["clone_opt"].to_numpy(dtype=np.int64)
    target.obs["compartment_opt"] = pd.Categorical(aligned["compartment_opt"])
    target.obs["p_cnv"] = aligned["p_cnv"].to_numpy(dtype=np.float64)
    target.obs["aneuploidy_score"] = aligned["aneuploidy_score"].to_numpy(dtype=np.float64)

    cell_pos = {cid: i for i, cid in enumerate(result.clones.cell_ids)}
    order = np.array([cell_pos[c] for c in obs_names], dtype=np.int64)
    target.obsm["X_numbat_clone_probs"] = result.clones.clone_probs[order]
    target.uns["numbat"] = {
        "segments": result.segments_dataframe(),
        "clones": result.clones_dataframe(),
        "phylogeny_newick": result.phylogeny.to_newick(),
        "phylogeny_ll": float(result.phylogeny.log_likelihood),
        "backend_used": result.backend_used,
    }
    return target
