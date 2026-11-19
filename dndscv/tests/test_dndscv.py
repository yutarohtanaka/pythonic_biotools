"""
Comprehensive deterministic test suite for `dndscv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import dndscv
from dndscv import (
    TRINUC_CONTEXTS,
    DnDScvModel,
    annotate_mutations,
    build_refcds,
    build_refcds_from_gtf,
    build_synthetic_refcds,
    dndscv as run_dndscv,
    simulate_refcds,
)
from dndscv.refcds import BASES, COMPLEMENT, _TRINUC_ALT_TO_IDX, classify_codon_impact, reverse_complement


def _gene_site_catalog(gene: dndscv.GeneRef) -> dict[tuple[int, int], list[tuple[int, str, str]]]:
    """
    Map `(trinuc_ind, cls_idx)` -> list of `(genomic_pos_1based, ref_plus, mut_plus)`
    for a `GeneRef`, used by synthetic cohort generators to emit genuine MAF rows.
    """
    catalog: dict[tuple[int, int], list[tuple[int, str, str]]] = {}

    # Coding positions
    cds_off = 0
    for s_arr, e_arr in gene.intervals_cds:
        s_i, e_i = int(s_arr), int(e_arr)
        exon_len = e_i - s_i + 1
        for offset in range(exon_len):
            cds_idx = cds_off + offset
            pos = (s_i + offset) if gene.strand == 1 else (e_i - offset)
            b0 = gene.seq_cds1up[cds_idx]
            b1 = gene.seq_cds[cds_idx]
            b2 = gene.seq_cds1down[cds_idx]
            if b0 not in BASES or b1 not in BASES or b2 not in BASES:
                continue
            codon_start = (cds_idx // 3) * 3
            ref_codon = gene.seq_cds[codon_start : codon_start + 3]
            frame = cds_idx - codon_start
            tri = b0 + b1 + b2
            ref_plus = b1 if gene.strand == 1 else COMPLEMENT[b1]
            for m in BASES:
                if m == b1:
                    continue
                mut_codon = ref_codon[:frame] + m + ref_codon[frame + 1 :]
                cls_idx = classify_codon_impact(ref_codon, mut_codon)
                if cls_idx is None:
                    continue
                t_idx = _TRINUC_ALT_TO_IDX[(tri, m)]
                mut_plus = m if gene.strand == 1 else COMPLEMENT[m]
                catalog.setdefault((t_idx, cls_idx), []).append((pos, ref_plus, mut_plus))
        cds_off += exon_len

    # Essential splice positions
    for sp_pos, tri in zip(gene.intervals_splice, gene.splice_trinucs):
        pos = int(sp_pos)
        b1 = tri[1]
        ref_plus = b1 if gene.strand == 1 else COMPLEMENT[b1]
        for m in BASES:
            if m == b1:
                continue
            t_idx = _TRINUC_ALT_TO_IDX[(tri, m)]
            mut_plus = m if gene.strand == 1 else COMPLEMENT[m]
            catalog.setdefault((t_idx, 3), []).append((pos, ref_plus, mut_plus))

    return catalog


def _simulate_maf_from_refcds(
    ref: dndscv.RefCDS,
    rng: np.random.Generator,
    r_true: np.ndarray,
    *,
    beta: np.ndarray | None = None,
    theta: float | None = 5.0,
    w_gene: np.ndarray | None = None,  # (G, 4) multipliers for (syn, mis, non, spl)
    ind_rate_scale: float = 0.0,
    w_ind_gene: np.ndarray | None = None,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """Simulate genuine genomic MAF mutations from a `RefCDS` under NB/Poisson + 192-rate model."""
    G = len(ref)
    if beta is not None and ref.covariates is not None:
        x_mat = ref.covariates.to_numpy(dtype=np.float64)
        rho = np.exp(beta[0] + x_mat @ beta[1:])
    elif beta is not None:
        rho = np.full(G, np.exp(beta[0]), dtype=np.float64)
    else:
        rho = np.ones(G, dtype=np.float64)

    if theta is not None and np.isfinite(theta):
        t_g = rng.gamma(shape=theta, scale=rho / theta)
    else:
        t_g = rho.copy()

    if w_gene is None:
        w_gene = np.ones((G, 4), dtype=np.float64)

    # Expected counts (G, 192, 4)
    lam = (
        t_g[:, None, None]
        * ref.L_all.astype(np.float64)
        * r_true[None, :, None]
        * w_gene[:, None, :]
    )
    counts = rng.poisson(lam)

    rows: list[dict[str, object]] = []
    mut_counter = 0

    for g_idx in range(G):
        nz = np.argwhere(counts[g_idx] > 0)
        if nz.size == 0 and (ind_rate_scale <= 0.0):
            continue
        gene = ref.genes[g_idx]
        if nz.size > 0:
            catalog = _gene_site_catalog(gene)
            for t_idx, cls_idx in nz:
                k = int(counts[g_idx, t_idx, cls_idx])
                sites = catalog[(int(t_idx), int(cls_idx))]
                chosen_indices = rng.integers(0, len(sites), size=k)
                for site_i in chosen_indices:
                    pos, ref_b, mut_b = sites[int(site_i)]
                    rows.append(
                        {
                            "sampleID": f"PAT_{mut_counter % n_samples:04d}",
                            "chr": gene.chrom,
                            "pos": pos,
                            "ref": ref_b,
                            "mut": mut_b,
                        }
                    )
                    mut_counter += 1

        if ind_rate_scale > 0.0:
            w_i = 1.0 if w_ind_gene is None else float(w_ind_gene[g_idx])
            lam_ind = float(t_g[g_idx] * gene.cds_length * ind_rate_scale * w_i)
            k_ind = int(rng.poisson(lam_ind))
            if k_ind > 0:
                s0, e0 = int(gene.intervals_cds[0, 0]), int(gene.intervals_cds[0, 1])
                for _ in range(k_ind):
                    pos = int(rng.integers(s0, e0 + 1))
                    rows.append(
                        {
                            "sampleID": f"PAT_{mut_counter % n_samples:04d}",
                            "chr": gene.chrom,
                            "pos": pos,
                            "ref": "A",
                            "mut": "-",
                        }
                    )
                    mut_counter += 1

    return pd.DataFrame(rows)


def test_planted_driver_recovery_missense_truncating_indel() -> None:
    """Recover planted missense oncogenes, truncating TSGs, and an indel driver."""
    rng = np.random.default_rng(42)
    ref = simulate_refcds(n_genes=300, cds_length=600, n_exons=3, n_covariates=2, seed=42)

    r_true = rng.uniform(0.015, 0.045, size=192)
    beta = np.array([0.0, 0.6, -0.5])
    w_gene = np.ones((len(ref), 4), dtype=np.float64)
    w_ind_gene = np.ones(len(ref), dtype=np.float64)

    # Plant 2 missense oncogenes, 2 truncating TSGs (non + spl), 1 indel TSG
    mis_drivers = [ref.gene_names[0], ref.gene_names[1]]
    tru_drivers = [ref.gene_names[2], ref.gene_names[3]]
    ind_driver = ref.gene_names[4]

    w_gene[0:2, 1] = 14.0
    w_gene[2:4, 2] = 18.0
    w_gene[2:4, 3] = 18.0
    w_ind_gene[4] = 18.0

    muts = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=beta,
        theta=5.0,
        w_gene=w_gene,
        ind_rate_scale=0.002,
        w_ind_gene=w_ind_gene,
        n_samples=2000,
    )

    res = run_dndscv(muts, ref)
    df = res.to_dataframe().set_index("gene_name")

    for g in mis_drivers:
        assert df.loc[g, "q_global_cv"] < 0.01
        assert df.loc[g, "w_mis_cv"] > 5.0
    for g in tru_drivers:
        assert df.loc[g, "q_global_cv"] < 0.01
        assert df.loc[g, "w_non_cv"] > 5.0
        assert df.loc[g, "w_spl_cv"] > 5.0
    assert df.loc[ind_driver, "q_global_cv"] < 0.01
    assert df.loc[ind_driver, "w_ind_cv"] > 5.0

    sig_df = res.significant(alpha=0.01)
    assert set(mis_drivers + tru_drivers + [ind_driver]).issubset(set(sig_df["gene_name"]))


def test_192_context_rate_recovery_and_strand_asymmetry() -> None:
    """Recover skewed 192 rates (CpG + strand asymmetry) and global dN/dS with CIs."""
    rng = np.random.default_rng(101)
    ref = simulate_refcds(n_genes=350, cds_length=600, n_exons=5, n_covariates=0, seed=101)

    r_true = np.full(192, 0.02, dtype=np.float64)
    for idx, ctx in enumerate(TRINUC_CONTEXTS):
        # 20x CpG C>T transitions
        if ctx in {"ACG>ATG", "CCG>CTG", "GCG>GTG", "TCG>TTG"}:
            r_true[idx] *= 20.0
        # 6x coding-strand asymmetry on TCA>TTA vs TGA>TAA
        elif ctx == "TCA>TTA":
            r_true[idx] *= 6.0
        elif ctx == "TGA>TAA":
            r_true[idx] *= 1.0

    w_gene = np.ones((len(ref), 4), dtype=np.float64)
    w_gene[:, 1] = 0.8
    w_gene[:, 2] = 0.5
    w_gene[:, 3] = 0.6

    muts = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=None,
        theta=None,
        w_gene=w_gene,
        n_samples=2000,
    )

    res = run_dndscv(muts, ref, ci_level=0.99, ci_method="wald")
    corr = float(np.corrcoef(res.r_mle, r_true)[0, 1])
    assert corr > 0.98

    # Exact normalization identity: sum_j r_mle[j] * L_syn_tot[j] == n_syn_tot
    L_syn_tot = np.sum(ref.L_all[:, :, 0], axis=0)
    n_syn_tot = float(np.sum(res.n_syn))
    assert np.isclose(float(np.sum(res.r_mle * L_syn_tot)), n_syn_tot, rtol=1e-10)

    # Strand asymmetry preserved between TCA>TTA and TGA>TAA
    idx_tca = TRINUC_CONTEXTS.index("TCA>TTA")
    idx_tga = TRINUC_CONTEXTS.index("TGA>TAA")
    assert res.r_mle[idx_tca] > 3.5 * res.r_mle[idx_tga]

    gd = res.global_dnds_dataframe().set_index("name")
    assert np.isclose(gd.loc["wmis", "mle"], 0.8, atol=0.05)
    assert np.isclose(gd.loc["wnon", "mle"], 0.5, atol=0.12)
    assert np.isclose(gd.loc["wspl", "mle"], 0.6, atol=0.12)
    assert gd.loc["wmis", "cilow"] <= 0.8 <= gd.loc["wmis", "cihigh"]
    assert gd.loc["wnon", "cilow"] <= 0.5 <= gd.loc["wnon", "cihigh"]
    assert gd.loc["wspl", "cilow"] <= 0.6 <= gd.loc["wspl", "cihigh"]
    assert 0.4 < gd.loc["wtru", "mle"] < 0.7
    assert 0.65 < gd.loc["wall", "mle"] < 0.90


def test_nb_covariate_recovery_and_confounder_calibration() -> None:
    """Recover NB slopes and dispersion theta, preventing confounder-driven false positives."""
    rng = np.random.default_rng(77)
    ref = simulate_refcds(n_genes=450, cds_length=600, n_exons=2, n_covariates=2, seed=77)

    r_true = np.full(192, 0.03, dtype=np.float64)
    beta_true = np.array([0.0, 0.85, -0.65])
    theta_true = 4.0

    muts = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=beta_true,
        theta=theta_true,
        n_samples=3000,
    )

    res_cov = run_dndscv(muts, ref)
    assert np.allclose(res_cov.beta_syn[1:], beta_true[1:], atol=0.15)
    assert 2.0 < res_cov.theta_syn < 8.0
    assert not res_cov.is_poisson_syn
    assert int(np.sum(res_cov.q_global_cv < 0.05)) == 0

    # Without covariates, dispersion is absorbed into smaller theta (higher alpha)
    res_nocov = run_dndscv(muts, ref, covariates=np.empty((len(ref), 0)))
    assert res_nocov.theta_syn < res_cov.theta_syn


def test_neutral_null_calibration_and_global_ci_coverage() -> None:
    """Under pure neutrality, null p-values are uniform and global 95% CIs cover 1.0."""
    rng = np.random.default_rng(2025)
    ref = simulate_refcds(n_genes=400, cds_length=600, n_exons=5, n_covariates=1, seed=2025)

    r_true = np.full(192, 0.20, dtype=np.float64)
    beta = np.array([0.0, 0.4])
    muts = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=beta,
        theta=8.0,
        ind_rate_scale=0.04,
        n_samples=20000,
    )

    res = run_dndscv(muts, ref)
    df = res.to_dataframe()
    powered = df["exp_syn"] > 5.0
    assert int(np.sum(powered)) > 200

    _, p_ks_all = stats.kstest(df.loc[powered, "p_all_cv"], "uniform")
    _, p_ks_ind = stats.kstest(df.loc[powered, "p_ind_cv"], "uniform")
    _, p_ks_glob = stats.kstest(df.loc[powered, "p_global_cv"], "uniform")
    assert p_ks_all > 0.01
    assert p_ks_ind > 0.01
    assert p_ks_glob > 0.01
    assert int(np.sum(df["q_global_cv"] < 0.05)) == 0

    gd = res.global_dnds_dataframe()
    assert np.all(gd["cilow"] <= 1.0)
    assert np.all(gd["cihigh"] >= 1.0)


def test_refcds_mult_exon_genomic_flanks_and_strand_orientation() -> None:
    """Verify genomic intronic flanks at exon boundaries on both + and - strands."""
    # Construct + strand 2-exon gene:
    # Exon 1: pos 10..12 = ATG, followed by intronic GT at pos 13..14
    # Exon 2: preceded by intronic AG at pos 23..24, Exon 2: pos 25..27 = TAA
    # Flanking 5' of exon 1 (pos 9) = 'C', 3' of exon 2 (pos 28) = 'G'
    # Sequence length 40 on chr1
    seq_chr1 = list("A" * 40)
    # + gene exons
    seq_chr1[8] = "C"  # pos 9 (5' flank of exon 1)
    seq_chr1[9:12] = list("ATG")  # pos 10..12
    seq_chr1[12:14] = list("GT")  # pos 13..14 (donor +1,+2)
    seq_chr1[22:24] = list("AG")  # pos 23..24 (acceptor -2,-1)
    seq_chr1[24:27] = list("TAA")  # pos 25..27
    seq_chr1[27] = "G"  # pos 28 (3' flank of exon 2)

    # - strand 2-exon gene on chr2:
    # Transcript 5'->3' = ATG | TAA
    # Genomic highest exon (5' exon): pos 30..32 -> rc("ATG") = "CAT"
    # pos 33 is 5' genomic flank on + strand -> rc('C') = 'G' -> set pos 33 = 'G' so coding 5' flank = 'C'
    # pos 28..29 are donor +1,+2 -> rc("GT") = "AC" -> pos 28='A', pos 29='C'
    # pos 16..17 are acceptor -2,-1 -> rc("AG") = "CT" -> pos 16='C', pos 17='T'
    # Genomic lowest exon (3' exon): pos 13..15 -> rc("TAA") = "TTA"
    # pos 12 is 3' genomic flank -> rc('G') = 'C' -> set pos 12 = 'C' so coding 3' flank = 'G'
    seq_chr2 = list("T" * 40)
    seq_chr2[11] = "C"  # pos 12
    seq_chr2[12:15] = list("TTA")  # pos 13..15 -> rc = TAA
    seq_chr2[15:17] = list("CT")  # pos 16..17 -> rc = AG
    seq_chr2[27:29] = list("AC")  # pos 28..29 -> rc = GT
    seq_chr2[29:32] = list("CAT")  # pos 30..32 -> rc = ATG
    seq_chr2[32] = "G"  # pos 33

    cds_table = pd.DataFrame(
        [
            {"gene_name": "PLUS_G", "chrom": "chr1", "start": 10, "end": 12, "strand": "+"},
            {"gene_name": "PLUS_G", "chrom": "chr1", "start": 25, "end": 27, "strand": "+"},
            {"gene_name": "MINUS_G", "chrom": "2", "start": 13, "end": 15, "strand": "-"},
            {"gene_name": "MINUS_G", "chrom": "2", "start": 30, "end": 32, "strand": "-"},
        ]
    )
    fasta = {"chr1": "".join(seq_chr1), "2": "".join(seq_chr2)}
    ref = build_refcds(cds_table, fasta, require_valid_cds=True)

    g_plus, g_minus = ref.genes
    assert g_plus.seq_cds == "ATGTAA"
    assert g_minus.seq_cds == "ATGTAA"

    # Last base of exon 1 (index 2, 'G') MUST use intronic 'G' (donor +1) as 3' flank, NOT exon 2 'T'!
    assert g_plus.seq_cds1down[2] == "G"
    assert g_minus.seq_cds1down[2] == "G"
    # First base of exon 2 (index 3, 'T') MUST use intronic 'G' (acceptor -1) as 5' flank, NOT exon 1 'G'!
    assert g_plus.seq_cds1up[3] == "G"
    assert g_minus.seq_cds1up[3] == "G"

    assert g_plus.cds_length == 6
    assert int(np.sum(g_plus.L[:, :3])) == 3 * 6
    assert int(np.sum(g_plus.L[:, 3])) == 12
    np.testing.assert_array_equal(g_plus.L, g_minus.L)


def test_strand_reverse_complement_invariance() -> None:
    """Mirroring a + gene onto - strand produces identical L, impacts, and dN/dS."""
    rng = np.random.default_rng(9)
    # Build a 90bp (3 exon x 30bp) + strand genomic contig with no internal stops
    intron1 = "GT" + "ACGT" * 5 + "AG"
    intron2 = "GT" + "TGCA" * 5 + "AG"
    ex1 = "ATG" + "GCTGATCGATCGATCGATCGATCGATC"
    ex2 = "GCTGCTGCTGCTGCTGCTGCTGCTGCTGCT"
    ex3 = "CGATCGATCGATCGATCGATCGATCGGTAA"
    contig_plus = ("ACGTA" * 4) + ex1 + intron1 + ex2 + intron2 + ex3 + ("TGCAT" * 4)
    L_contig = len(contig_plus)
    contig_rc = reverse_complement(contig_plus)

    # + strand exon 1-based intervals
    s1 = 21
    e1 = s1 + len(ex1) - 1
    s2 = e1 + len(intron1) + 1
    e2 = s2 + len(ex2) - 1
    s3 = e2 + len(intron2) + 1
    e3 = s3 + len(ex3) - 1

    def flip(s: int, e: int) -> tuple[int, int]:
        return (L_contig - e + 1, L_contig - s + 1)

    rs1, re1 = flip(s1, e1)
    rs2, re2 = flip(s2, e2)
    rs3, re3 = flip(s3, e3)

    ref_plus = build_refcds(
        pd.DataFrame(
            [
                {"gene_name": "G", "chrom": "chr1", "start": s1, "end": e1, "strand": 1},
                {"gene_name": "G", "chrom": "chr1", "start": s2, "end": e2, "strand": 1},
                {"gene_name": "G", "chrom": "chr1", "start": s3, "end": e3, "strand": 1},
            ]
        ),
        {"chr1": contig_plus},
    )
    ref_minus = build_refcds(
        pd.DataFrame(
            [
                {"gene_name": "G", "chrom": "chr1", "start": rs1, "end": re1, "strand": -1},
                {"gene_name": "G", "chrom": "chr1", "start": rs2, "end": re2, "strand": -1},
                {"gene_name": "G", "chrom": "chr1", "start": rs3, "end": re3, "strand": -1},
            ]
        ),
        {"chr1": contig_rc},
    )

    np.testing.assert_array_equal(ref_plus.L_all, ref_minus.L_all)
    assert ref_plus.genes[0].seq_cds == ref_minus.genes[0].seq_cds

    # Map mutations across coding + splice bases onto both references
    muts_plus_rows = []
    muts_minus_rows = []
    pos_candidates = list(range(s1, e1 + 1)) + [e1 + 1, s2 - 2] + list(range(s2, e2 + 1))
    for k, p in enumerate(pos_candidates):
        ref_b = contig_plus[p - 1]
        alts = [b for b in BASES if b != ref_b]
        alt_b = alts[int(rng.integers(0, 3))]
        sid = f"S_{k % 20}"
        muts_plus_rows.append({"sampleID": sid, "chr": "chr1", "pos": p, "ref": ref_b, "mut": alt_b})
        p_rc = L_contig - p + 1
        muts_minus_rows.append(
            {
                "sampleID": sid,
                "chr": "chr1",
                "pos": p_rc,
                "ref": COMPLEMENT[ref_b],
                "mut": COMPLEMENT[alt_b],
            }
        )

    ann_p = annotate_mutations(pd.DataFrame(muts_plus_rows), ref_plus, max_muts_per_gene_per_sample=None)
    ann_m = annotate_mutations(pd.DataFrame(muts_minus_rows), ref_minus, max_muts_per_gene_per_sample=None)

    np.testing.assert_array_equal(ann_p.n_obs, ann_m.n_obs)
    assert list(ann_p.annot_muts["impact"].sort_values()) == list(ann_m.annot_muts["impact"].sort_values())
    assert list(ann_p.annot_muts["aachange"].sort_values()) == list(ann_m.annot_muts["aachange"].sort_values())


def test_splice_boundaries_stop_codons_and_gtf_builder(tmp_path: Path) -> None:
    """Verify splice boundaries (-2/-1/+1/+2 vs -3/+3/UTR), stop-codon rules, and GTF builder."""
    # Contig: 10bp 5' flank + exon1 (ATG GCT = 6bp, pos 11..16) + 10bp intron (GT...AG, pos 17..26) + exon2 (TAA = 3bp, pos 27..29) + 10bp 3' flank
    contig = "CCCCCCCCCC" + "ATGGCT" + "GTAAAAAAAG" + "TAA" + "GGGGGGGGGG"
    fa_path = tmp_path / "mini.fa"
    fa_path.write_text(f">chr1\n{contig}\n", encoding="utf-8")

    # Include two transcripts for the same gene to test longest-transcript selection + phase trimming
    gtf_lines = [
        "# mini GTF",
        'chr1\tsrc\tCDS\t11\t16\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "TX_LONG";',
        'chr1\tsrc\tCDS\t27\t29\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "TX_LONG";',
        'chr1\tsrc\tCDS\t27\t29\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "TX_SHORT";',
    ]
    gtf_path = tmp_path / "mini.gtf"
    gtf_path.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")

    ref = build_refcds_from_gtf(gtf_path, fa_path, require_valid_cds=True)
    assert len(ref) == 1
    assert ref.genes[0].cds_length == 9
    assert list(ref.genes[0].intervals_splice) == [17, 18, 25, 26]

    muts = pd.DataFrame(
        [
            {"sampleID": "S1", "chr": "1", "pos": 17, "ref": "G", "mut": "A"},  # donor +1 -> Essential_Splice
            {"sampleID": "S1", "chr": "1", "pos": 25, "ref": "A", "mut": "G"},  # acceptor -2 -> Essential_Splice
            {"sampleID": "S1", "chr": "1", "pos": 19, "ref": "A", "mut": "C"},  # intronic +3 -> unmapped
            {"sampleID": "S1", "chr": "1", "pos": 10, "ref": "C", "mut": "T"},  # 5' UTR flank -> unmapped
            {"sampleID": "S1", "chr": "1", "pos": 30, "ref": "G", "mut": "A"},  # 3' UTR flank -> unmapped
            {"sampleID": "S1", "chr": "1", "pos": 29, "ref": "A", "mut": "G"},  # TAA->TAG stop->stop -> Synonymous
            {"sampleID": "S1", "chr": "1", "pos": 27, "ref": "T", "mut": "C"},  # TAA->CAA stop-loss -> Nonsense (matches L)
        ]
    )
    ann = annotate_mutations(muts, ref, max_muts_per_gene_per_sample=None)
    impacts = dict(zip(ann.annot_muts["pos"], ann.annot_muts["impact"]))
    assert impacts[17] == "Essential_Splice"
    assert impacts[25] == "Essential_Splice"
    assert 19 not in impacts
    assert 10 not in impacts
    assert 30 not in impacts
    assert impacts[29] == "Synonymous"
    assert impacts[27] == "Nonsense"


def test_maf_mapping_aliases_indels_zero_based_and_dedup() -> None:
    """Verify TCGA MAF columns, chrom normalization, zero_based coords, indels, and deduplication."""
    ref = build_synthetic_refcds(
        {
            "BRAF": {"seq": "ATGGCGTAA", "chrom": "chr7", "strand": 1},
            "KRAS": {"seq": "ATGGCGTAA", "chrom": "chr12", "strand": -1},
        },
        require_valid_cds=True,
    )
    braf = ref.genes[0]
    kras = ref.genes[1]
    b_s = int(braf.intervals_cds[0, 0])
    k_e = int(kras.intervals_cds[0, 1])

    # BRAF codon 2 = GCG (A -> V via C>T at b_s+4)
    # KRAS (- strand) codon 2 = GCG (on + strand, comp('C')='G' at pos k_e - 4; mut to 'A' -> coding 'T' -> GTG = V)
    maf = pd.DataFrame(
        [
            {
                "Tumor_Sample_Barcode": "TCGA-01",
                "Chromosome": "7",
                "Start_Position": b_s + 4,
                "Reference_Allele": "C",
                "Tumor_Seq_Allele2": "T",
            },
            # Exact duplicate row in same sample
            {
                "Tumor_Sample_Barcode": "TCGA-01",
                "Chromosome": "chr7",
                "Start_Position": b_s + 4,
                "Reference_Allele": "C",
                "Tumor_Seq_Allele2": "T",
            },
            # Minus-strand missense in KRAS
            {
                "Tumor_Sample_Barcode": "TCGA-02",
                "Chromosome": "12",
                "Start_Position": k_e - 4,
                "Reference_Allele": "G",
                "Tumor_Seq_Allele2": "A",
            },
            # Frameshift deletion in BRAF
            {
                "Tumor_Sample_Barcode": "TCGA-03",
                "Chromosome": "7",
                "Start_Position": b_s + 3,
                "Reference_Allele": "GC",
                "Tumor_Seq_Allele2": "-",
            },
        ]
    )

    ann = annotate_mutations(maf, ref)
    assert ann.n_raw_muts == 4
    assert ann.n_dedup_muts == 3
    assert ann.n_coding_muts == 3
    assert int(ann.n_ind[0]) == 1
    assert int(np.sum(ann.n_obs[0, :, 1])) == 1
    assert int(np.sum(ann.n_obs[1, :, 1])) == 1

    aa_map = dict(zip(ann.annot_muts["gene_name"], ann.annot_muts["aachange"]))
    assert aa_map["KRAS"] == "A2V"

    # Zero-based equivalence
    maf_0 = maf.copy()
    maf_0["Start_Position"] = maf_0["Start_Position"] - 1
    ann_0 = annotate_mutations(maf_0, ref, coord_system="zero_based")
    np.testing.assert_array_equal(ann.n_obs, ann_0.n_obs)
    np.testing.assert_array_equal(ann.n_ind, ann_0.n_ind)


def test_hypermutator_and_per_gene_sample_caps() -> None:
    """Hypermutator exclusion (>3000 coding muts) and per-gene-sample cap (3) prevent false hits."""
    rng = np.random.default_rng(555)
    ref = simulate_refcds(n_genes=100, cds_length=600, n_exons=2, n_covariates=1, seed=555)
    r_true = np.full(192, 0.005, dtype=np.float64)

    muts = _simulate_maf_from_refcds(ref, rng, r_true, beta=None, theta=None, n_samples=1000)

    # 1. Inject hypermutator sample with 3100 distinct coding mutations
    hyper_rows = []
    cnt = 0
    for g in ref.genes:
        cat = _gene_site_catalog(g)
        for (_, _), sites in cat.items():
            for pos, ref_b, mut_b in sites:
                hyper_rows.append(
                    {
                        "sampleID": "HYPER_01",
                        "chr": g.chrom,
                        "pos": pos,
                        "ref": ref_b,
                        "mut": mut_b,
                    }
                )
                cnt += 1
                if cnt >= 3100:
                    break
            if cnt >= 3100:
                break
        if cnt >= 3100:
            break

    # 2. Inject 25 missense mutations in gene 0 from ONE single sample (should be capped at 3)
    g0_cat = _gene_site_catalog(ref.genes[0])
    mis_sites_g0 = [s for (_, cls), sites in g0_cat.items() if cls == 1 for s in sites]
    single_sample_rows = [
        {
            "sampleID": "KATAEGIS_01",
            "chr": ref.genes[0].chrom,
            "pos": pos,
            "ref": ref_b,
            "mut": mut_b,
        }
        for pos, ref_b, mut_b in mis_sites_g0[:25]
    ]

    # 3. Inject genuine multi-patient driver in gene 1 (35 distinct patients)
    g1_cat = _gene_site_catalog(ref.genes[1])
    mis_sites_g1 = [s for (_, cls), sites in g1_cat.items() if cls == 1 for s in sites]
    true_driver_rows = [
        {
            "sampleID": f"DRV_PAT_{i:02d}",
            "chr": ref.genes[1].chrom,
            "pos": mis_sites_g1[i % len(mis_sites_g1)][0],
            "ref": mis_sites_g1[i % len(mis_sites_g1)][1],
            "mut": mis_sites_g1[i % len(mis_sites_g1)][2],
        }
        for i in range(35)
    ]

    all_muts = pd.concat(
        [muts, pd.DataFrame(hyper_rows), pd.DataFrame(single_sample_rows), pd.DataFrame(true_driver_rows)],
        ignore_index=True,
    )

    res = run_dndscv(all_muts, ref, max_muts_per_gene_per_sample=3, max_coding_muts_per_sample=3000)
    assert res.excluded_samples == ["HYPER_01"]

    df = res.to_dataframe().set_index("gene_name")
    assert df.loc[ref.gene_names[0], "q_global_cv"] > 0.1
    assert df.loc[ref.gene_names[1], "q_global_cv"] < 1e-4


def test_analytical_limits_and_edge_cases() -> None:
    """Verify Poisson fallback, single-exon NaN FDR hygiene, profile vs Wald CIs, empty MAF, and validation."""
    # (a) Underdispersed / equidispersed cohort triggers Poisson fallback matching use_nb=False
    rng = np.random.default_rng(321)
    seqs = {}
    for i in range(80):
        # Mix 1-exon gene at index 0 with 2-exon genes for (b)
        codons = "ATG" + ("GCTGAACGTAGCGTA" * 12) + "TAA"  # 186 bp
        seqs[f"G_{i:03d}"] = {
            "seq": codons,
            "exons": [len(codons)] if i == 0 else [93, len(codons) - 93],
            "strand": 1 if i % 2 == 0 else -1,
            "chrom": "chr1",
        }
    ref = build_synthetic_refcds(seqs, require_valid_cds=True)

    # Plant fixed deterministic counts per gene (exact equidispersion -> zero variance in syn)
    # Distributed across diverse sites so splice trinucleotide contexts have positive r_mle
    rows = []
    for i, g in enumerate(ref.genes):
        cat = _gene_site_catalog(g)
        syn_sites = [s for (_, cls), sites in cat.items() if cls == 0 for s in sites]
        mis_sites = [s for (_, cls), sites in cat.items() if cls == 1 for s in sites]
        for k in range(3):
            p, r_b, m_b = syn_sites[(i * 3 + k) % len(syn_sites)]
            rows.append({"sampleID": f"U_{k}", "chr": g.chrom, "pos": p, "ref": r_b, "mut": m_b})
        for k in range(2):
            p, r_b, m_b = mis_sites[(i * 2 + k) % len(mis_sites)]
            rows.append({"sampleID": f"M_{k}", "chr": g.chrom, "pos": p, "ref": r_b, "mut": m_b})
    muts_eq = pd.DataFrame(rows)

    res_nb = run_dndscv(muts_eq, ref, use_nb=True)
    res_pois = run_dndscv(muts_eq, ref, use_nb=False)
    assert res_nb.is_poisson_syn
    assert res_nb.alpha_syn == 0.0
    assert np.isinf(res_nb.theta_syn)
    np.testing.assert_allclose(res_nb.p_all_cv, res_pois.p_all_cv, atol=1e-10)

    # (b) Single-exon gene has NaN splice stats without inflating BH m on other genes
    assert np.isnan(res_nb.w_spl_cv[0])
    assert np.isnan(res_nb.p_spl_cv[0])
    assert np.isnan(res_nb.q_spl_cv[0])
    # Zero global splice mutations: globaldnds handles n_spl == 0 gracefully
    gd_wald = res_nb.global_dnds_dataframe().set_index("name")
    assert gd_wald.loc["wspl", "mle"] == 0.0
    assert gd_wald.loc["wspl", "cilow"] == 0.0
    assert np.isfinite(gd_wald.loc["wspl", "cihigh"])

    # (c) Profile vs Wald CIs agree closely on moderate/well-identified counts
    ref_ci = simulate_refcds(n_genes=120, cds_length=450, n_exons=2, n_covariates=0, seed=99)
    muts_ci = _simulate_maf_from_refcds(
        ref_ci, rng, np.full(192, 0.02), beta=None, theta=None, n_samples=1000
    )
    gd_ci_wald = DnDScvModel(muts_ci, ref_ci, ci_method="wald").fit().global_dnds_dataframe().set_index("name")
    gd_ci_prof = DnDScvModel(muts_ci, ref_ci, ci_method="profile").fit().global_dnds_dataframe().set_index("name")
    for wname in ["wmis", "wnon", "wspl", "wtru", "wall"]:
        assert np.isclose(gd_ci_wald.loc[wname, "cilow"], gd_ci_prof.loc[wname, "cilow"], rtol=0.06)
        assert np.isclose(gd_ci_wald.loc[wname, "cihigh"], gd_ci_prof.loc[wname, "cihigh"], rtol=0.06)

    # Panel subsetting via gene_list and RefCDS.subset
    sub_res = run_dndscv(muts_ci, ref_ci, gene_list=ref_ci.gene_names[:40])
    assert len(sub_res.gene_names) == 40
    assert len(ref_ci.subset(ref_ci.gene_names[:40])) == 40

    # (d) Empty mutations edge case + accessors
    empty_muts = pd.DataFrame(columns=["sampleID", "chr", "pos", "ref", "mut"])
    res_empty = run_dndscv(empty_muts, ref)
    assert res_empty.converged
    assert np.all(res_empty.p_global_cv == 1.0)
    assert np.all(res_empty.q_global_cv == 1.0)
    assert res_empty.rates_dataframe().shape == (192, 10)
    assert res_empty.nb_summary_dataframe().shape[0] == 1
    assert "dNdScvResult" in res_empty.summary()
    assert res_nb.to_dataframe(sort_by_pvalue=True).shape[0] == len(ref)

    # (e) Collinear / constant covariates & zero-synonymous errors
    x_col = rng.normal(size=len(ref))
    collinear_cov = np.column_stack([x_col, 2.0 * x_col])
    with pytest.raises(ValueError, match="rank-deficient"):
        run_dndscv(muts_eq, ref, covariates=collinear_cov)

    const_cov = np.ones((len(ref), 1))
    with pytest.raises(ValueError, match="constant"):
        run_dndscv(muts_eq, ref, covariates=const_cov)

    mis_only = muts_eq.loc[muts_eq["sampleID"].str.startswith("M_")].copy()
    with pytest.raises(ValueError, match="zero synonymous"):
        run_dndscv(mis_only, ref)


def test_indel_only_cohort_recovery_and_nan_snv_stats() -> None:
    """Indel-only cohorts (0 SNVs, >0 indels) skip SNV fitting, return NaN SNV stats, and run indel tests."""
    # (a) Minimal no-covariate indel-only cohort
    ref_min = simulate_refcds(n_genes=50, cds_length=300, n_exons=2, n_covariates=0, seed=1)
    ind_min = pd.DataFrame(
        [
            {
                "sampleID": f"S{k}",
                "chr": ref_min.genes[k].chrom,
                "pos": int(ref_min.genes[k].intervals_cds[0, 0]),
                "ref": "A",
                "mut": "-",
            }
            for k in range(10)
        ]
    )
    res_min = run_dndscv(ind_min, ref_min)
    assert res_min.converged
    assert int(np.sum(res_min.n_syn + res_min.n_mis + res_min.n_non + res_min.n_spl)) == 0
    assert int(np.sum(res_min.n_ind)) == 10
    assert np.all(np.isnan(res_min.w_mis_cv))
    assert np.all(np.isnan(res_min.p_all_cv))
    assert np.all(np.isfinite(res_min.p_ind_cv))
    np.testing.assert_allclose(res_min.p_global_cv, res_min.p_ind_cv)
    np.testing.assert_allclose(res_min.q_global_cv, res_min.q_ind_cv)

    # (b) Indel-only cohort with covariates and a planted indel driver
    rng = np.random.default_rng(808)
    ref = simulate_refcds(n_genes=80, cds_length=600, n_exons=2, n_covariates=1, seed=808)
    w_ind_gene = np.ones(len(ref), dtype=np.float64)
    w_ind_gene[0] = 25.0
    muts_ind = _simulate_maf_from_refcds(
        ref,
        rng,
        np.zeros(192, dtype=np.float64),
        beta=np.array([0.0, 0.5]),
        theta=6.0,
        ind_rate_scale=0.005,
        w_ind_gene=w_ind_gene,
        n_samples=2000,
    )
    res = run_dndscv(muts_ind, ref)
    assert res.converged
    assert int(np.sum(res.n_syn + res.n_mis + res.n_non + res.n_spl)) == 0
    assert int(np.sum(res.n_ind)) > 0
    assert np.all(res.r_mle == 0.0)
    assert np.all(res.exp_syn == 0.0)
    assert np.all(res.mu_syn == 0.0)

    for col in (
        "w_mis_cv",
        "w_non_cv",
        "w_spl_cv",
        "w_mis_p",
        "w_non_p",
        "w_spl_p",
        "p_mis_cv",
        "p_non_cv",
        "p_spl_cv",
        "p_tru_cv",
        "p_all_cv",
        "q_mis_cv",
        "q_non_cv",
        "q_spl_cv",
        "q_tru_cv",
        "q_all_cv",
        "p_mis_p",
        "p_non_p",
        "p_spl_p",
        "p_tru_p",
        "p_all_p",
        "q_all_p",
    ):
        assert np.all(np.isnan(getattr(res, col))), f"Expected NaN for {col}"

    assert np.all(res.exp_ind > 0.0)
    assert np.all(np.isfinite(res.w_ind_cv))
    assert np.all(np.isfinite(res.p_ind_cv))
    assert np.all(np.isfinite(res.q_ind_cv))
    np.testing.assert_allclose(res.p_global_cv, res.p_ind_cv)
    np.testing.assert_allclose(res.q_global_cv, res.q_ind_cv)

    df = res.to_dataframe().set_index("gene_name")
    assert df.loc[ref.gene_names[0], "w_ind_cv"] > 5.0
    assert df.loc[ref.gene_names[0], "q_global_cv"] < 0.01
    assert ref.gene_names[0] in set(res.significant(alpha=0.01)["gene_name"])
    assert res.global_dnds_dataframe().shape == (5, 7)
    assert res.rates_dataframe().shape == (192, 10)
    assert res.nb_summary_dataframe().shape == (2, 4)
    assert "0 coding SNVs" in res.summary()


def test_gtf_minus_strand_nonzero_phase(tmp_path: Path) -> None:
    """Verify build_refcds_from_gtf trims 5' phase=1 and phase=2 correctly on - strand."""
    # chr1 (- strand, phase=1 on 5' exon [22, 28]):
    # pos 9..11 = TTA (rc=TAA), pos 12..21 = CTTTTTTTAC (rc=GTAAAAAAAG),
    # pos 22..27 = AGCCAT (rc=ATGGCT), pos 28 = G (trimmed phase=1 -> coding 5' flank C)
    contig_chr1 = ("A" * 8) + "TTA" + "CTTTTTTTAC" + "AGCCAT" + "G" + ("T" * 12)
    # chr2 (- strand, phase=2 on 5' exon [22, 29]):
    # pos 28..29 = TG (trimmed phase=2 -> coding 5' flank at pos 28 is comp('T')='A')
    contig_chr2 = ("A" * 8) + "TTA" + "CTTTTTTTAC" + "AGCCAT" + "TG" + ("C" * 11)

    fa_path = tmp_path / "minus_phase.fa"
    fa_path.write_text(f">chr1\n{contig_chr1}\n>chr2\n{contig_chr2}\n", encoding="utf-8")

    gtf_path = tmp_path / "minus_phase.gtf"
    gtf_path.write_text(
        "\n".join(
            [
                'chr1\tsrc\tCDS\t9\t11\t.\t-\t0\tgene_id "GM1"; gene_name "NEG_P1"; transcript_id "TX1";',
                'chr1\tsrc\tCDS\t22\t28\t.\t-\t1\tgene_id "GM1"; gene_name "NEG_P1"; transcript_id "TX1";',
                'chr2\tsrc\tCDS\t9\t11\t.\t-\t0\tgene_id "GM2"; gene_name "NEG_P2"; transcript_id "TX2";',
                'chr2\tsrc\tCDS\t22\t29\t.\t-\t2\tgene_id "GM2"; gene_name "NEG_P2"; transcript_id "TX2";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ref = build_refcds_from_gtf(gtf_path, fa_path, require_valid_cds=True)
    by_name = {g.gene_name: g for g in ref.genes}

    g1 = by_name["NEG_P1"]
    assert g1.strand == -1
    assert g1.seq_cds == "ATGGCTTAA"
    assert g1.cds_length == 9
    assert g1.seq_cds1up[0] == "C"
    assert g1.seq_cds1down[5] == "G"
    assert g1.seq_cds1up[6] == "G"
    assert list(g1.intervals_splice) == [12, 13, 20, 21]

    g2 = by_name["NEG_P2"]
    assert g2.strand == -1
    assert g2.seq_cds == "ATGGCTTAA"
    assert g2.cds_length == 9
    assert g2.seq_cds1up[0] == "A"
    assert list(g2.intervals_splice) == [12, 13, 20, 21]

    # Mutation at trimmed phase base (pos 28 on chr1) must NOT map as coding;
    # mutation in codon 2 (pos 23 'G' -> rc 'C' in GCG, mut 'A' -> coding 'T' -> GTG A2V) and splice pos 21 ('C' -> 'T') must map
    muts = pd.DataFrame(
        [
            {"sampleID": "S1", "chr": "chr1", "pos": 28, "ref": "G", "mut": "A"},
            {"sampleID": "S1", "chr": "chr1", "pos": 23, "ref": "G", "mut": "A"},
            {"sampleID": "S1", "chr": "chr1", "pos": 21, "ref": "C", "mut": "T"},
        ]
    )
    ann = annotate_mutations(muts, ref, max_muts_per_gene_per_sample=None)
    mapped = dict(zip(ann.annot_muts["pos"], ann.annot_muts["impact"]))
    assert 28 not in mapped
    assert mapped[23] == "Missense"
    assert mapped[21] == "Essential_Splice"

    # Also check require_valid_cds=False trailing 3' remainder trim on - strand
    gtf_rem_path = tmp_path / "minus_rem.gtf"
    gtf_rem_path.write_text(
        "\n".join(
            [
                'chr1\tsrc\tCDS\t8\t11\t.\t-\t0\tgene_id "GM3"; gene_name "NEG_REM"; transcript_id "TX3";',
                'chr1\tsrc\tCDS\t22\t28\t.\t-\t1\tgene_id "GM3"; gene_name "NEG_REM"; transcript_id "TX3";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref_rem = build_refcds_from_gtf(gtf_rem_path, fa_path, require_valid_cds=False)
    assert ref_rem.genes[0].seq_cds == "ATGGCTTAA"


def test_sparse_zero_count_trinucleotide_contexts() -> None:
    """Handle cohorts with >150 zero-count trinucleotide contexts cleanly without NaN or 0*log(0)."""
    rng = np.random.default_rng(707)
    ref = simulate_refcds(n_genes=120, cds_length=600, n_exons=2, n_covariates=1, seed=707)

    # Select only 20 active trinucleotide contexts (leaving 172 zero-count contexts)
    # that have abundant synonymous and missense opportunities
    L_tot = np.sum(ref.L_all, axis=0)
    eligible = np.where((L_tot[:, 0] > 100) & (L_tot[:, 1] > 100))[0]
    active_idx = eligible[:20]
    r_true = np.zeros(192, dtype=np.float64)
    r_true[active_idx] = 0.25

    w_gene = np.ones((len(ref), 4), dtype=np.float64)
    w_gene[0, 1] = 12.0  # planted missense driver in GENE_0000

    muts = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=np.array([0.0, 0.3]),
        theta=6.0,
        w_gene=w_gene,
        n_samples=3000,
    )

    for rate_model in ("192r_3w", "192r_syn"):
        res = run_dndscv(muts, ref, rate_model=rate_model, ci_method="wald")
        n_zero_contexts = int(np.sum(np.sum(res.n_192, axis=1) == 0))
        assert n_zero_contexts > 150
        assert res.converged
        assert np.all(np.isfinite(res.r_mle))
        assert np.all(res.r_mle[np.sum(res.n_192, axis=1) == 0] == 0.0)
        assert np.all(np.isfinite(res.global_mle))
        assert np.all(np.isfinite(res.p_all_cv))
        assert np.all(np.isfinite(res.q_global_cv))
        df = res.to_dataframe().set_index("gene_name")
        assert df.loc[ref.gene_names[0], "q_global_cv"] < 0.01

    res_prof = run_dndscv(muts, ref, rate_model="192r_3w", ci_method="profile")
    assert np.all(np.isfinite(res_prof.global_mle))
    assert np.all(np.isfinite(res_prof.global_cilow))


def test_settings_r_default_glm_nb_trajectory_sel_cv_columns_and_theta_fallback() -> None:
    """
    Verify `settings="R_default"` vs `settings="default"` in `dndscv`:
    1. Strict `ValueError` on invalid `settings`.
    2. Alternating IRLS + Newton `theta` updates (`MASS::glm.nb`, `maxit=25`) recover
       NB dispersion `theta` and covariate slopes accurately.
    3. Exact upstream `sel_cv` DataFrame column ordering both with active indels (19 cols)
       and SNV-only cohorts (14 cols).
    4. Strict `theta` convergence fallback rules fall back cleanly to Poisson when
       outer `glm.nb` iteration fails to converge (`max_iter=1`) or `theta -> inf`.
    """
    from dndscv.model import fit_nb_covariate_model

    rng = np.random.default_rng(2026)
    ref = simulate_refcds(n_genes=220, cds_length=600, n_exons=3, n_covariates=2, seed=2026)
    r_true = rng.uniform(0.02, 0.04, size=192)
    beta_true = np.array([0.0, 0.65, -0.50])
    w_gene = np.ones((len(ref), 4), dtype=np.float64)
    w_gene[0, 1] = 16.0  # missense driver
    w_ind = np.ones(len(ref), dtype=np.float64)
    w_ind[1] = 20.0  # indel driver

    muts_with_ind = _simulate_maf_from_refcds(
        ref,
        rng,
        r_true,
        beta=beta_true,
        theta=4.5,
        w_gene=w_gene,
        ind_rate_scale=0.003,
        w_ind_gene=w_ind,
        n_samples=2500,
    )

    # 1. Invalid settings raises ValueError
    with pytest.raises(ValueError, match="settings"):
        run_dndscv(muts_with_ind, ref, settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        DnDScvModel(muts_with_ind, ref, settings="bad_mode")  # type: ignore[arg-type]

    # 2 & 3. Exact sel_cv column ordering when indels are active + MASS::glm.nb parity
    res_def = run_dndscv(muts_with_ind, ref, settings="default")
    res_r = run_dndscv(muts_with_ind, ref, settings="R_default")
    assert res_r.settings == "R_default"
    assert res_r.converged
    assert not res_r.is_poisson_syn
    np.testing.assert_allclose(res_r.theta_syn, res_def.theta_syn, rtol=1e-4)
    np.testing.assert_allclose(res_r.beta_syn, res_def.beta_syn, atol=1e-4)

    expected_indel_cols = [
        "gene_name",
        "n_syn",
        "n_mis",
        "n_non",
        "n_spl",
        "wmis_cv",
        "wnon_cv",
        "wspl_cv",
        "wind_cv",
        "pmis_cv",
        "ptrunc_cv",
        "pallsubs_cv",
        "pind_cv",
        "qmis_cv",
        "qtrunc_cv",
        "qallsubs_cv",
        "qind_cv",
        "pglobal_cv",
        "qglobal_cv",
    ]
    df_r = res_r.to_dataframe()
    assert list(df_r.columns) == expected_indel_cols
    assert list(res_r.sel_cv.columns) == expected_indel_cols
    assert "cds_length" in res_def.to_dataframe().columns
    sig_r = res_r.significant(alpha=0.05)
    assert {ref.gene_names[0], ref.gene_names[1]}.issubset(set(sig_r["gene_name"]))

    # SNV-only cohort (0 indels): exact 14-column upstream sel_cv schema
    muts_snv_only = muts_with_ind.loc[muts_with_ind["mut"] != "-"].copy()
    res_r_snv = DnDScvModel(muts_snv_only, ref, settings="R_default").fit()
    expected_snv_cols = [
        "gene_name",
        "n_syn",
        "n_mis",
        "n_non",
        "n_spl",
        "wmis_cv",
        "wnon_cv",
        "wspl_cv",
        "pmis_cv",
        "ptrunc_cv",
        "pallsubs_cv",
        "qmis_cv",
        "qtrunc_cv",
        "qallsubs_cv",
    ]
    assert list(res_r_snv.to_dataframe().columns) == expected_snv_cols

    # 4. Strict theta convergence fallback: non-convergent glm.nb (max_iter=1 on strongly
    #    overdispersed counts) falls back strictly to Poisson under R_default, whereas default
    #    retains the intermediate finite theta iterate
    nb_nonconv_r = fit_nb_covariate_model(
        res_r.n_syn.astype(float),
        res_r.exp_syn,
        ref.covariates,
        np.ones(len(ref), dtype=bool),
        use_nb=True,
        max_iter=1,
        tol=1e-14,
        settings="R_default",
    )
    nb_nonconv_def = fit_nb_covariate_model(
        res_r.n_syn.astype(float),
        res_r.exp_syn,
        ref.covariates,
        np.ones(len(ref), dtype=bool),
        use_nb=True,
        max_iter=1,
        tol=1e-14,
        settings="default",
    )
    assert nb_nonconv_r.is_poisson is True
    assert np.isinf(nb_nonconv_r.theta)
    assert nb_nonconv_def.is_poisson is False
    assert np.isfinite(nb_nonconv_def.theta)


