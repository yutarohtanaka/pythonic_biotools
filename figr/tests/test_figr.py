"""Deterministic unit and statistical validation tests for `figr`."""

from __future__ import annotations

from dataclasses import is_dataclass
import gzip
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from figr import (
    DORCCallResult,
    DORCScoresResult,
    FigRGRNResult,
    GenePeakCorrResult,
    build_knn_graph,
    call_dorcs,
    extract_gtf_tss,
    get_background_peaks,
    get_dorc_scores,
    run_figr_grn,
    run_gene_peak_corr,
    smooth_scores_knn,
)


def _make_multiome_for_cis_and_dorcs(seed: int = 123):
    """Simulate paired scRNA + scATAC with 2 planted DORC genes (7-8 cis peaks each)."""
    rng = np.random.default_rng(seed)
    n_cells = 200
    n_genes = 20
    n_peaks = 300

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    # Plant GENE_0 and GENE_1 as DORCs on chr1 and chr2
    tss_rows = []
    for i, g in enumerate(gene_names):
        chrom = "chr1" if i % 2 == 0 else "chr2"
        tss_pos = 100_000 + (i // 2) * 500_000
        tss_rows.append({"gene_name": g, "chrom": chrom, "strand": "+", "tss": tss_pos})
    tss_df = pd.DataFrame(tss_rows)

    # Create peaks: 8 cis peaks near GENE_0, 7 cis peaks near GENE_1, 1-2 cis peaks near others, rest distal
    peak_chroms: list[str] = []
    peak_starts: list[int] = []
    peak_ends: list[int] = []

    planted_links: dict[str, list[int]] = {"GENE_0": [], "GENE_1": []}
    for k in range(8):
        idx = len(peak_chroms)
        peak_chroms.append("chr1")
        st = 100_000 + (k - 4) * 5_000
        peak_starts.append(st)
        peak_ends.append(st + 500)
        planted_links["GENE_0"].append(idx)

    for k in range(7):
        idx = len(peak_chroms)
        peak_chroms.append("chr2")
        st = 100_000 + (k - 3) * 5_000
        peak_starts.append(st)
        peak_ends.append(st + 500)
        planted_links["GENE_1"].append(idx)

    # Remaining peaks spread outside or with 1-2 unlinked cis peaks
    while len(peak_chroms) < n_peaks:
        idx = len(peak_chroms)
        ch = "chr1" if idx % 2 == 0 else "chr2"
        st = 10_000_000 + idx * 2_000
        peak_chroms.append(ch)
        peak_starts.append(st)
        peak_ends.append(st + 500)

    # Give GENE_2 and GENE_3 1 noisy cis peak each
    peak_chroms[15] = "chr1"
    peak_starts[15] = 602_000
    peak_ends[15] = 602_500
    peak_chroms[16] = "chr2"
    peak_starts[16] = 602_000
    peak_ends[16] = 602_500

    peak_names = [f"{c}:{s}-{e}" for c, s, e in zip(peak_chroms, peak_starts, peak_ends)]
    peak_gc = rng.uniform(0.30, 0.70, size=n_peaks)

    # Latent cell factors driving DORC_0, DORC_1, and independent background variation
    z0 = rng.normal(0.0, 1.0, size=n_cells)
    z1 = rng.normal(0.0, 1.0, size=n_cells)

    rna_mat = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)
    rna_mat[:, 0] = rng.poisson(np.exp(1.2 + 1.1 * z0)).astype(np.float64)
    rna_mat[:, 1] = rng.poisson(np.exp(1.2 + 1.1 * z1)).astype(np.float64)

    # Background peaks span the same mean accessibility distribution as planted DORC peaks
    # using independent cell-level latent noise so matched background peaks exist cleanly.
    bg_latent = rng.normal(0.0, 1.0, size=(n_cells, 8))
    bg_assign = rng.integers(0, 8, size=n_peaks)
    atac_rates = np.exp(0.3 + 1.1 * bg_latent[:, bg_assign])
    for p_idx in planted_links["GENE_0"]:
        atac_rates[:, p_idx] = np.exp(0.3 + 1.1 * z0)
    for p_idx in planted_links["GENE_1"]:
        atac_rates[:, p_idx] = np.exp(0.3 + 1.1 * z1)

    atac_mat = rng.poisson(atac_rates).astype(np.float64)

    cell_names = [f"cell_{i}" for i in range(n_cells)]
    rna_adata = ad.AnnData(X=rna_mat, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
    atac_adata = ad.AnnData(
        X=sparse.csr_matrix(atac_mat),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame({"gc_content": peak_gc}, index=peak_names),
    )
    return rna_adata, atac_adata, tss_df, peak_names, planted_links


def test_planted_cis_links_and_dorc_calling():
    rna_adata, atac_adata, tss_df, peak_names, planted_links = _make_multiome_for_cis_and_dorcs(seed=101)

    res = run_gene_peak_corr(
        rna_adata,
        atac_adata,
        tss=tss_df,
        peak_gc="gc_content",
        window_pad=50_000,
        n_bg=80,
        seed=101,
    )
    df = res.to_dataframe()
    assert not df.empty

    sig = res.significant(p_cut=None, fdr_cut=0.05)
    recovered_pairs = set(zip(sig["gene"], sig["peak"]))
    for g, p_indices in planted_links.items():
        for p_idx in p_indices:
            assert (g, peak_names[p_idx]) in recovered_pairs

    dorcs = call_dorcs(res, dorc_k=5, p_cut=0.05, fdr_cut=0.05)
    assert set(dorcs.dorc_genes) == {"GENE_0", "GENE_1"}
    dorc_tbl = dorcs.dorc_table()
    assert list(dorc_tbl["n_peaks"]) == [8, 7]


def test_background_matching_and_null_calibration():
    rng = np.random.default_rng(202)
    n_cells = 150
    n_genes = 30
    n_peaks = 400

    # Strong GC and mean-accessibility heterogeneity across peaks
    peak_gc = rng.uniform(0.2, 0.8, size=n_peaks)
    peak_means = np.exp(rng.normal(0.0, 1.2, size=n_peaks))
    atac_mat = rng.poisson(peak_means[None, :], size=(n_cells, n_peaks)).astype(np.float64)

    # Independent RNA
    rna_mat = rng.normal(0.0, 1.0, size=(n_cells, n_genes))

    for method in ("chromvar_kernel", "knn_mahalanobis"):
        bg = get_background_peaks(atac_mat, peak_gc, n_bg=60, method=method, seed=202)
        assert bg.shape == (n_peaks, 60)
        bg_gc_mean = peak_gc[bg].mean(axis=1)
        log_acc = np.log10(np.maximum(atac_mat.mean(axis=0), 1e-8))
        bg_acc_mean = log_acc[bg].mean(axis=1)
        assert np.corrcoef(peak_gc, bg_gc_mean)[0, 1] > 0.85
        assert np.corrcoef(log_acc, bg_acc_mean)[0, 1] > 0.85

    # Place ~10 cis peaks per gene within 50kb so ~300 null cis pairs are tested
    gene_names = [f"G_{i}" for i in range(n_genes)]
    tss_df = pd.DataFrame(
        {
            "gene_name": gene_names,
            "chrom": ["chr1"] * n_genes,
            "strand": ["+"] * n_genes,
            "tss": [100_000 + i * 200_000 for i in range(n_genes)],
        }
    )
    peak_names = []
    for p in range(n_peaks):
        g_idx = p % n_genes
        st = 100_000 + g_idx * 200_000 + ((p // n_genes) - 5) * 3_000
        peak_names.append(f"chr1:{st}-{st + 400}")

    rna_df = pd.DataFrame(rna_mat, index=[f"c_{i}" for i in range(n_cells)], columns=gene_names)
    atac_df = pd.DataFrame(atac_mat, index=[f"c_{i}" for i in range(n_cells)], columns=peak_names)

    res = run_gene_peak_corr(
        rna_df,
        atac_df,
        tss=tss_df,
        peak_gc=peak_gc,
        window_pad=50_000,
        n_bg=80,
        alternative="two-sided",
        seed=202,
    )
    valid = np.isfinite(res.p_value)
    assert np.sum(valid) >= 200
    z_valid = res.z_score[valid]
    p_valid = res.p_value[valid]

    assert abs(float(np.mean(z_valid))) < 0.25
    assert 0.7 < float(np.std(z_valid, ddof=1)) < 1.3
    type1 = float(np.mean(p_valid <= 0.05))
    assert type1 <= 0.08
    ks_res = stats.kstest(p_valid, "uniform")
    assert ks_res.pvalue > 0.01


def test_sparse_spearman_tie_and_anndata_parity():
    rng = np.random.default_rng(303)
    n_cells = 50
    n_genes = 6
    n_peaks = 40

    # Zero-inflated integer ATAC counts (~80% zeros) with heavy ties
    atac_dense = rng.choice([0, 0, 0, 0, 1, 2], size=(n_cells, n_peaks)).astype(np.float64)
    # Tie-heavy RNA counts
    rna_dense = rng.choice([0, 0, 1, 2, 3], size=(n_cells, n_genes)).astype(np.float64)

    cell_names = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"G_{i}" for i in range(n_genes)]
    peak_names = [f"chr1:{10_000 + p * 1_000}-{10_400 + p * 1_000}" for p in range(n_peaks)]
    gc = rng.uniform(0.35, 0.65, size=n_peaks)

    tss_df = pd.DataFrame(
        {
            "gene_name": gene_names,
            "chrom": ["chr1"] * n_genes,
            "strand": ["+"] * n_genes,
            "tss": [12_000 + i * 5_000 for i in range(n_genes)],
        }
    )

    rna_ad = ad.AnnData(X=sparse.csr_matrix(rna_dense), obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
    atac_ad = ad.AnnData(
        X=sparse.csr_matrix(atac_dense),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame({"gc_content": gc}, index=peak_names),
    )
    rna_df = pd.DataFrame(rna_dense, index=cell_names, columns=gene_names)
    atac_df = pd.DataFrame(atac_dense, index=cell_names, columns=peak_names)

    bg_fixed = get_background_peaks(atac_ad, "gc_content", n_bg=20, seed=303)

    res_ad = run_gene_peak_corr(
        rna_ad, atac_ad, tss_df, bg_peaks=bg_fixed, normalize_atac=False, window_pad=10_000
    )
    res_df = run_gene_peak_corr(
        rna_df, atac_df, tss_df, bg_peaks=bg_fixed, normalize_atac=False, window_pad=10_000
    )
    res_arr = run_gene_peak_corr(
        rna_dense.T,
        atac_dense.T,
        tss_df,
        peaks=peak_names,
        bg_peaks=bg_fixed,
        normalize_atac=False,
        orientation="features_by_cells",
        window_pad=10_000,
    )

    # Check exact parity with scipy.stats.spearmanr
    g_map = {g: i for i, g in enumerate(gene_names)}
    p_map = {p: i for i, p in enumerate(peak_names)}
    for i, (g, p) in enumerate(zip(res_ad.gene, res_ad.peak)):
        gi = g_map[g]
        pi = p_map[p]
        if np.std(rna_dense[:, gi]) > 0 and np.std(atac_dense[:, pi]) > 0:
            expected_r = stats.spearmanr(rna_dense[:, gi], atac_dense[:, pi]).statistic
            np.testing.assert_allclose(res_ad.r_obs[i], expected_r, atol=1e-12)

    np.testing.assert_allclose(res_ad.r_obs, res_df.r_obs, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res_ad.z_score, res_df.z_score, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res_ad.p_value, res_df.p_value, atol=1e-12, equal_nan=True)
    # Array input uses default gene_0..gene_k names unless DataFrame/AnnData, so test array with matching tss names
    tss_arr = tss_df.copy()
    tss_arr["gene_name"] = [f"gene_{i}" for i in range(n_genes)]
    res_arr_named = run_gene_peak_corr(
        rna_dense.T,
        atac_dense.T,
        tss_arr,
        peaks=peak_names,
        bg_peaks=bg_fixed,
        normalize_atac=False,
        orientation="features_by_cells",
        window_pad=10_000,
    )
    np.testing.assert_allclose(res_ad.r_obs, res_arr_named.r_obs, atol=1e-12, equal_nan=True)
    assert res_arr.window_pad == 10_000


def test_gtf_tss_strand_and_cis_window_boundaries(tmp_path):
    gtf_lines = [
        "##description: synthetic test GTF",
        'chr1\tHAVANA\tgene\t1001\t5000\t.\t+\t.\tgene_id "ENSG000001.5"; gene_name "PLUS_G"; gene_type "protein_coding";',
        'chr1\tHAVANA\tgene\t8001\t12000\t.\t-\t.\tgene_id "ENSG000002.3"; gene_name "MINUS_G"; gene_type "protein_coding";',
    ]
    gtf_plain = tmp_path / "mini.gtf"
    gtf_plain.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")

    gtf_gz = tmp_path / "mini.gtf.gz"
    with gzip.open(gtf_gz, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(gtf_lines) + "\n")

    tss_plain = extract_gtf_tss(gtf_plain)
    tss_gz = extract_gtf_tss(gtf_gz)
    pd.testing.assert_frame_equal(tss_plain, tss_gz)

    assert list(tss_plain["gene_id"]) == ["ENSG000001", "ENSG000002"]
    assert list(tss_plain["tss"]) == [1000, 11999]
    assert list(tss_plain["start"]) == [1000, 8000]
    assert list(tss_plain["end"]) == [5000, 12000]

    # Test exact boundary peaks around MINUS_G (tss = 11999, window_pad = 500 -> [11499, 12499])
    peak_names = [
        "chr1:11400-11499",  # boundary touch left -> INCLUDED
        "chr1:11400-11498",  # 1 bp outside left -> EXCLUDED
        "chr1:12499-12600",  # boundary touch right -> INCLUDED
        "chr1:12500-12600",  # 1 bp outside right -> EXCLUDED
        "chr1:7900-8100",    # near 5'-wrong end (start=8000) of minus gene -> EXCLUDED
        "chr2:11800-12100",  # wrong chrom -> EXCLUDED
        "chr1:900-1100",     # overlaps PLUS_G tss=1000 -> INCLUDED for PLUS_G only
    ]
    rng = np.random.default_rng(404)
    n_cells = 40
    rna_df = pd.DataFrame(
        rng.normal(size=(n_cells, 2)),
        index=[f"c_{i}" for i in range(n_cells)],
        columns=["PLUS_G", "MINUS_G"],
    )
    atac_df = pd.DataFrame(
        rng.poisson(2.0, size=(n_cells, len(peak_names))).astype(np.float64),
        index=[f"c_{i}" for i in range(n_cells)],
        columns=peak_names,
    )
    res = run_gene_peak_corr(
        rna_df,
        atac_df,
        tss=gtf_gz,
        peak_gc=np.full(len(peak_names), 0.5),
        window_pad=500,
        n_bg=5,
        seed=404,
    )
    df = res.to_dataframe()
    minus_peaks = set(df.loc[df["gene"] == "MINUS_G", "peak"])
    plus_peaks = set(df.loc[df["gene"] == "PLUS_G", "peak"])
    assert minus_peaks == {"chr1:11400-11499", "chr1:12499-12600"}
    assert plus_peaks == {"chr1:900-1100"}


def test_dorc_scores_and_knn_smoothing_properties():
    rng = np.random.default_rng(505)
    n_cells = 60
    n_peaks = 20
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    peak_names = [f"chr1:{p * 1000}-{p * 1000 + 500}" for p in range(n_peaks)]

    # 2 distinct cell clusters in embedding space
    emb = np.vstack([
        rng.normal(loc=-5.0, scale=0.3, size=(30, 4)),
        rng.normal(loc=5.0, scale=0.3, size=(30, 4)),
    ])
    # Cluster 0 has higher signal on DORC_A peaks (0..4) with cell-level Poisson dropout noise
    atac_mat = rng.poisson(2.0, size=(n_cells, n_peaks)).astype(np.float64)
    atac_mat[:30, :5] = rng.poisson(12.0, size=(30, 5)).astype(np.float64)
    # Zero-depth cell check: cell 59 has all zeros
    atac_mat[59, :] = 0.0

    atac_ad = ad.AnnData(
        X=sparse.csr_matrix(atac_mat),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=peak_names),
    )

    links_df = pd.DataFrame(
        {
            "gene": ["DORC_A"] * 5 + ["DORC_B"] * 5,
            "peak": peak_names[:5] + peak_names[5:10],
            "p_value": [1e-4] * 10,
            "r_obs": [0.7] * 10,
            "z_score": [4.0] * 10,
        }
    )
    dorcs = call_dorcs(links_df, dorc_k=5, p_cut=0.05)

    res_sum = get_dorc_scores(atac_ad, dorcs, normalize_atac=True, agg="sum", embedding=emb, k=10)
    res_mean = get_dorc_scores(atac_ad, dorcs, normalize_atac=True, agg="mean", embedding=emb, k=10)
    assert np.all(np.isfinite(res_sum.raw_scores))
    assert np.all(res_sum.raw_scores[59] == 0.0)
    np.testing.assert_allclose(res_sum.raw_scores / 5.0, res_mean.raw_scores, atol=1e-12)

    # Constant-signal preservation: W @ 1 == 1
    ones = np.ones((n_cells, 2), dtype=np.float64)
    W_emb = build_knn_graph(embedding=emb, k=10, weights="gaussian")
    np.testing.assert_allclose(np.asarray(W_emb.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    smoothed_ones = smooth_scores_knn(ones, embedding=emb, k=10)
    np.testing.assert_allclose(smoothed_ones, ones, atol=1e-12)

    # Exact match to W @ raw
    W_unif = build_knn_graph(embedding=emb, k=10, weights="uniform")
    np.testing.assert_allclose(res_sum.smoothed_scores, W_unif @ res_sum.raw_scores, atol=1e-12)

    # Noise variance reduction within cluster 0
    raw_var = float(np.var(res_sum.raw_scores[:30, 0]))
    smooth_var = float(np.var(res_sum.smoothed_scores[:30, 0]))
    assert smooth_var < raw_var

    # Test isolated cell (zero-degree row) in sparse adjacency matrix + (N, k) index matrix
    adj = sparse.lil_matrix((n_cells, n_cells), dtype=np.float64)
    for i in range(n_cells - 1):
        adj[i, (i + 1) % (n_cells - 1)] = 1.0
    # row 59 is completely zero (isolated)
    W_iso = build_knn_graph(knn=adj.tocsr(), include_self=False)
    np.testing.assert_allclose(np.asarray(W_iso.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    assert W_iso[59, 59] == 1.0

    # (N, k) neighbor index matrix without self -> include_self adds self once
    nbr_idx = np.column_stack([(np.arange(n_cells) + 1) % n_cells, (np.arange(n_cells) + 2) % n_cells])
    smoothed_idx = smooth_scores_knn(
        pd.DataFrame(res_sum.raw_scores, index=cell_names, columns=res_sum.dorc_genes),
        knn=nbr_idx,
        include_self=True,
    )
    assert isinstance(smoothed_idx, pd.DataFrame)
    assert np.all(np.isfinite(smoothed_idx.to_numpy()))


def test_planted_activator_and_repressor_grn():
    rng = np.random.default_rng(606)
    n_cells = 180
    n_peaks = 150
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    peak_names = [f"chr1:{p * 2000}-{p * 2000 + 500}" for p in range(n_peaks)]

    # Plant 2 DORCs: DORC_ACT (peaks 0..9), DORC_REP (peaks 10..19)
    dorc_links = pd.DataFrame(
        {
            "gene": ["DORC_ACT"] * 10 + ["DORC_REP"] * 10,
            "peak": peak_names[:10] + peak_names[10:20],
        }
    )

    # 4 TFs: TF_ACT, TF_REP, TF_NO_MOTIF, TF_NO_CORR
    tfs = ["TF_ACT", "TF_REP", "TF_NO_MOTIF", "TF_NO_CORR"]
    latent_act = rng.normal(0.0, 1.0, size=n_cells)
    latent_rep = rng.normal(0.0, 1.0, size=n_cells)

    dorc_scores_df = pd.DataFrame(
        {
            "DORC_ACT": 5.0 + 2.0 * latent_act + rng.normal(0.0, 0.25, size=n_cells),
            "DORC_REP": 5.0 - 2.0 * latent_rep + rng.normal(0.0, 0.25, size=n_cells),
        },
        index=cell_names,
    )

    rna_df = pd.DataFrame(
        {
            "TF_ACT": 3.0 + 1.5 * latent_act + rng.normal(0.0, 0.25, size=n_cells),
            "TF_REP": 3.0 + 1.5 * latent_rep + rng.normal(0.0, 0.25, size=n_cells),
            "TF_NO_MOTIF": 3.0 + 1.5 * latent_act + rng.normal(0.0, 0.25, size=n_cells),
            "TF_NO_CORR": rng.normal(3.0, 1.0, size=n_cells),
        },
        index=cell_names,
    )

    # Peak-motif matrix (peaks x motifs):
    # TF_ACT motif enriched in DORC_ACT peaks (9/10 vs ~10% bg)
    # TF_REP motif enriched in DORC_REP peaks (9/10 vs ~10% bg)
    # TF_NO_MOTIF has 0 hits in DORC peaks (~10% bg)
    # TF_NO_CORR motif enriched in DORC_ACT peaks (9/10 vs ~10% bg)
    motif_mat = (rng.uniform(size=(n_peaks, 4)) < 0.08).astype(np.float64)
    motif_mat[:9, 0] = 1.0       # TF_ACT in DORC_ACT
    motif_mat[10:19, 1] = 1.0    # TF_REP in DORC_REP
    motif_mat[:20, 2] = 0.0      # TF_NO_MOTIF absent in DORCs
    motif_mat[:9, 3] = 1.0       # TF_NO_CORR in DORC_ACT
    peak_motifs_df = pd.DataFrame(motif_mat, index=peak_names, columns=tfs)

    # Homogeneous ATAC + GC so background peaks match fairly
    peak_gc = rng.uniform(0.4, 0.6, size=n_peaks)
    atac_df = pd.DataFrame(
        rng.poisson(5.0, size=(n_cells, n_peaks)).astype(np.float64),
        index=cell_names,
        columns=peak_names,
    )

    for mode in ("product", "kartha_union"):
        grn = run_figr_grn(
            dorc_scores_df,
            rna_df,
            dorc_links=dorc_links,
            peak_motifs=peak_motifs_df,
            atac=atac_df,
            peak_gc=peak_gc,
            n_bg=60,
            score_mode=mode,
            seed=606,
        )
        df = grn.to_dataframe().set_index(["tf_gene", "dorc_gene"])
        act_row = df.loc[("TF_ACT", "DORC_ACT")]
        rep_row = df.loc[("TF_REP", "DORC_REP")]
        no_motif_row = df.loc[("TF_NO_MOTIF", "DORC_ACT")]
        no_corr_row = df.loc[("TF_NO_CORR", "DORC_ACT")]

        assert act_row["dorc_reg_score"] > 5.0
        assert act_row["role"] == "activator"
        assert rep_row["dorc_reg_score"] < -5.0
        assert rep_row["role"] == "repressor"
        assert abs(no_motif_row["dorc_reg_score"]) < 1.0
        assert abs(no_corr_row["dorc_reg_score"]) < 2.0

        # Check drivers() and to_adjacency()
        top_act = grn.drivers("DORC_ACT", top_n=1)
        assert top_act["tf_gene"].iloc[0] == "TF_ACT"
        adj = grn.to_adjacency()
        assert adj.loc["TF_ACT", "DORC_ACT"] > 0
        assert adj.loc["TF_REP", "DORC_REP"] < 0


def test_negative_motif_clamping_p_underflow_and_hypergeometric():
    rng = np.random.default_rng(707)
    n_cells = 250
    n_peaks = 120
    cell_names = [f"c_{i}" for i in range(n_cells)]
    peak_names = [f"chr1:{p * 1000}-{p * 1000 + 400}" for p in range(n_peaks)]

    dorc_links = pd.DataFrame({"gene": ["D1"] * 12, "peak": peak_names[:12]})
    sig = rng.normal(0.0, 1.0, size=n_cells)
    dorc_scores_df = pd.DataFrame({"D1": sig}, index=cell_names)
    # Perfect correlation (p -> 0 underflow check) for TF_HYPER and TF_DEPLETED
    rna_df = pd.DataFrame(
        {
            "TF_DEPLETED": sig,
            "TF_EXTREME": sig,
        },
        index=cell_names,
    )

    # TF_DEPLETED: 0 hits in D1 peaks, 80% hits in background -> Z_motif < 0
    # TF_EXTREME: 12/12 hits in D1 peaks, 0% in background -> p_motif -> 0
    motif_mat = np.zeros((n_peaks, 2), dtype=np.float64)
    motif_mat[12:, 0] = 1.0   # TF_DEPLETED depleted in D1
    motif_mat[:12, 1] = 1.0   # TF_EXTREME exclusively in D1
    peak_motifs_df = pd.DataFrame(motif_mat, index=peak_names, columns=["TF_DEPLETED", "TF_EXTREME"])

    # Hypergeometric fallback (no atac / background_peaks supplied)
    grn = run_figr_grn(
        dorc_scores_df,
        rna_df,
        dorc_links=dorc_links,
        peak_motifs=peak_motifs_df,
        clamp_negative_enrichment=True,
        seed=707,
    )
    assert grn.enrichment_method == "hypergeometric"
    df = grn.to_dataframe().set_index("tf_gene")

    dep = df.loc["TF_DEPLETED"]
    assert dep["motif_enrichment_z"] < -3.0
    assert dep["dorc_reg_score"] == 0.0
    assert dep["role"] == "neutral"

    ext = df.loc["TF_EXTREME"]
    assert np.isfinite(ext["dorc_reg_score"])
    assert ext["dorc_reg_score"] > 100.0
    assert ext["role"] == "activator"


def test_nan_aware_bh_fdr_and_constant_features():
    rng = np.random.default_rng(808)
    n_cells = 60
    n_genes = 6
    n_peaks = 30
    cell_names = [f"c_{i}" for i in range(n_cells)]
    gene_names = [f"G_{i}" for i in range(n_genes)]
    peak_names = [f"chr1:{10_000 + p * 1000}-{10_400 + p * 1000}" for p in range(n_peaks)]

    rna_mat = rng.normal(2.0, 1.0, size=(n_cells, n_genes))
    atac_mat = rng.poisson(3.0, size=(n_cells, n_peaks)).astype(np.float64)
    # Inject strong signal for G_0 + peak 0, and zero-variance G_1 + peak 1
    z = rng.normal(0.0, 1.0, size=n_cells)
    rna_mat[:, 0] = z
    atac_mat[:, 0] = np.exp(1.0 + 1.5 * z)
    rna_mat[:, 1] = 0.0      # constant gene
    atac_mat[:, 1] = 0.0     # constant peak

    tss_df = pd.DataFrame(
        {
            "gene_name": gene_names,
            "chrom": ["chr1"] * n_genes,
            "strand": ["+"] * n_genes,
            "tss": [15_000] * n_genes,
        }
    )
    gc = rng.uniform(0.3, 0.7, size=n_peaks)
    bg_fixed = get_background_peaks(atac_mat, gc, n_bg=15, seed=808)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_full = run_gene_peak_corr(
            pd.DataFrame(rna_mat, index=cell_names, columns=gene_names),
            pd.DataFrame(atac_mat, index=cell_names, columns=peak_names),
            tss=tss_df,
            bg_peaks=bg_fixed,
            normalize_atac=False,
            window_pad=20_000,
        )
    # Ensure no scipy ConstantInputWarning was emitted
    assert not any("ConstantInput" in str(w.category.__name__) for w in caught)

    df_full = res_full.to_dataframe()
    const_mask = (df_full["gene"] == "G_1") | (df_full["peak"] == peak_names[1])
    assert np.all(np.isnan(df_full.loc[const_mask, "r_obs"]))
    assert np.all(np.isnan(df_full.loc[const_mask, "p_value"]))
    assert np.all(np.isnan(df_full.loc[const_mask, "fdr"]))

    # Compare FDR of valid pairs against manual BH computed strictly on finite p-values
    valid_mask = np.isfinite(res_full.p_value)
    valid_p = res_full.p_value[valid_mask]
    order = np.argsort(valid_p, kind="mergesort")
    ranked = valid_p[order] * len(valid_p) / np.arange(1, len(valid_p) + 1)
    manual_fdr = np.empty_like(valid_p)
    manual_fdr[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    np.testing.assert_allclose(res_full.fdr[valid_mask], manual_fdr, atol=1e-15)


def test_background_small_pool_self_exclusion_and_constant_gc():
    rng = np.random.default_rng(909)
    n_cells = 30
    n_peaks = 20  # Fewer peaks than n_bg=100!
    atac_mat = rng.poisson(2.0, size=(n_cells, n_peaks)).astype(np.float64)
    const_gc = np.full(n_peaks, 0.5, dtype=np.float64)

    for method in ("chromvar_kernel", "knn_mahalanobis"):
        bg = get_background_peaks(atac_mat, const_gc, n_bg=100, method=method, exclude_self=True, seed=909)
        assert bg.shape == (n_peaks, 100)
        for p_idx in range(n_peaks):
            assert p_idx not in bg[p_idx]

    with pytest.raises(ValueError, match="At least 2 peaks"):
        get_background_peaks(atac_mat[:, :1], [0.5], n_bg=10)


def test_misaligned_anndata_cell_alignment():
    rna_ad, atac_ad, tss_df, _, _ = _make_multiome_for_cis_and_dorcs(seed=111)
    rng = np.random.default_rng(112)

    # Pre-aligned subset of 180 cells
    keep_cells = rna_ad.obs_names[:180].tolist()
    perm = rng.permutation(180)
    atac_shuffled_names = [keep_cells[i] for i in perm] + ["extra_atac_1", "extra_atac_2"]
    rna_extra_names = list(rna_ad.obs_names)  # contains 20 extra RNA-only cells

    rna_sub_aligned = rna_ad[atac_shuffled_names[:180]].copy()
    atac_sub_aligned = atac_ad[atac_shuffled_names[:180]].copy()

    # Build misaligned atac_ad with shuffled shared cells + extra ATAC-only cells
    extra_X = sparse.csr_matrix(np.ones((2, atac_ad.n_vars), dtype=np.float64))
    atac_misaligned = ad.AnnData(
        X=sparse.vstack([atac_ad[atac_shuffled_names[:180]].X, extra_X]).tocsr(),
        obs=pd.DataFrame(index=atac_shuffled_names),
        var=atac_ad.var.copy(),
    )
    rna_misaligned = rna_ad[rna_extra_names].copy()

    bg = get_background_peaks(atac_sub_aligned, "gc_content", n_bg=30, seed=111)
    res_aligned = run_gene_peak_corr(rna_sub_aligned, atac_sub_aligned, tss_df, bg_peaks=bg)
    res_misaligned = run_gene_peak_corr(rna_misaligned, atac_misaligned, tss_df, bg_peaks=bg)

    assert res_misaligned.n_cells == 180
    np.testing.assert_allclose(res_aligned.r_obs, res_misaligned.r_obs, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res_aligned.z_score, res_misaligned.z_score, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res_aligned.p_value, res_misaligned.p_value, atol=1e-12, equal_nan=True)

    # Also test DORC scoring + GRN alignment with shuffled cell orders
    dorcs = call_dorcs(res_aligned, dorc_k=5, p_cut=0.05)
    scores_aligned = get_dorc_scores(atac_sub_aligned, dorcs)
    scores_mis = get_dorc_scores(atac_misaligned, dorcs)
    # Reorder rna_misaligned to reversed shared order
    rna_rev = rna_sub_aligned[list(reversed(scores_aligned.cell_names))].copy()
    motifs_df = pd.DataFrame(
        np.ones((atac_ad.n_vars, 2), dtype=np.float64),
        index=atac_ad.var_names,
        columns=["GENE_2", "GENE_3"],
    )
    motifs_df.iloc[:15, 0] = 0.0
    grn_a = run_figr_grn(scores_aligned, rna_sub_aligned, dorcs, motifs_df)
    grn_b = run_figr_grn(scores_aligned, rna_rev, dorcs, motifs_df)
    np.testing.assert_allclose(grn_a.r_tf_dorc, grn_b.r_tf_dorc, atol=1e-12, equal_nan=True)
    assert scores_mis.raw_scores.shape[0] == 182


def test_edge_cases_validation_and_dataclasses():
    rng = np.random.default_rng(1212)
    rna_small = pd.DataFrame(rng.normal(size=(2, 3)), index=["c1", "c2"], columns=["G1", "G2", "G3"])
    atac_small = pd.DataFrame(
        rng.poisson(1.0, size=(2, 3)).astype(np.float64),
        index=["c1", "c2"],
        columns=["chr1:100-200", "chr1:300-400", "chr1:500-600"],
    )
    tss_df = pd.DataFrame({"gene_name": ["G1"], "chrom": ["chr1"], "strand": ["+"], "tss": [150]})

    # < 3 shared cells
    with pytest.raises(ValueError, match="At least 3"):
        run_gene_peak_corr(rna_small, atac_small, tss_df, peak_gc=[0.5, 0.5, 0.5])

    # Empty matrix
    with pytest.raises(ValueError, match="cannot be empty"):
        run_gene_peak_corr(np.empty((5, 0)), np.ones((5, 3)), tss_df, peaks=["chr1:10-20", "chr1:30-40", "chr1:50-60"], peak_gc=[0.5, 0.5, 0.5])

    # Missing peak_gc and wrong length peak_gc
    rna_ok = pd.DataFrame(rng.normal(size=(10, 2)), index=[f"c{i}" for i in range(10)], columns=["G1", "G2"])
    atac_ok = pd.DataFrame(
        rng.poisson(1.0, size=(10, 3)).astype(np.float64),
        index=[f"c{i}" for i in range(10)],
        columns=["chr1:100-200", "chr1:300-400", "chr1:500-600"],
    )
    with pytest.raises(ValueError, match="peak_gc"):
        run_gene_peak_corr(rna_ok, atac_ok, tss_df)
    with pytest.raises(ValueError, match="peak_gc"):
        run_gene_peak_corr(rna_ok, atac_ok, tss_df, peak_gc=[0.5, 0.5])

    # Zero cis overlap returns valid empty GenePeakCorrResult
    tss_far = pd.DataFrame({"gene_name": ["G1"], "chrom": ["chr9"], "strand": ["+"], "tss": [1_000_000]})
    empty_corr = run_gene_peak_corr(rna_ok, atac_ok, tss_far, peak_gc=[0.5, 0.5, 0.5], n_bg=5)
    assert is_dataclass(empty_corr)
    assert empty_corr.to_dataframe().empty
    assert list(empty_corr.to_dataframe().columns) == [
        "gene", "peak", "chrom", "tss", "peak_start", "peak_end", "tss_distance",
        "r_obs", "bg_mean", "bg_std", "z_score", "p_value", "fdr",
    ]
    assert "n_pairs=0" in empty_corr.summary()

    # Zero DORCs when dorc_k is high
    dorcs_empty = call_dorcs(empty_corr, dorc_k=5)
    assert is_dataclass(dorcs_empty)
    assert dorcs_empty.dorc_genes == []
    assert dorcs_empty.to_dataframe().empty
    assert "n_dorcs=0" in dorcs_empty.summary()

    scores_empty = get_dorc_scores(atac_ok, dorcs_empty)
    assert is_dataclass(scores_empty)
    assert scores_empty.raw_scores.shape == (10, 0)
    assert scores_empty.to_dataframe().shape == (10, 0)
    assert "n_dorcs=0" in scores_empty.summary()

    motifs_ok = pd.DataFrame(np.ones((3, 2)), index=atac_ok.columns, columns=["G1", "G2"])
    grn_empty = run_figr_grn(scores_empty, rna_ok, dorcs_empty, motifs_ok)
    assert is_dataclass(grn_empty)
    assert grn_empty.to_dataframe().empty
    assert "n_edges=0" in grn_empty.summary()

    # Check frozen dataclass immutability
    with pytest.raises(Exception):
        empty_corr.n_cells = 999  # type: ignore[misc]
    with pytest.raises(Exception):
        dorcs_empty.dorc_k = 999  # type: ignore[misc]
    with pytest.raises(Exception):
        scores_empty.agg = "mean"  # type: ignore[misc]
    with pytest.raises(Exception):
        grn_empty.n_cells = 999  # type: ignore[misc]


def test_multi_tss_deduplication_and_dorc_k_pool():
    rng = np.random.default_rng(1313)
    n_cells = 80
    n_peaks = 40
    cell_names = [f"c_{i}" for i in range(n_cells)]
    peak_names = [f"chr1:{10_000 + p * 500}-{10_400 + p * 500}" for p in range(n_peaks)]

    # Two TSS entries for the same gene G_MULTI so peak 2 overlaps both windows
    tss_df = pd.DataFrame(
        {
            "gene_name": ["G_MULTI", "G_MULTI", "G_OTHER"],
            "chrom": ["chr1", "chr1", "chr1"],
            "strand": ["+", "+", "+"],
            "tss": [10_200, 12_000, 20_000],
        }
    )
    z = rng.normal(size=n_cells)
    rna_df = pd.DataFrame(
        {
            "G_MULTI": 2.0 + z + rng.normal(0.0, 0.2, size=n_cells),
            "G_OTHER": 2.0 + z + rng.normal(0.0, 0.2, size=n_cells),
            "TF_A": 1.0 + z + rng.normal(0.0, 0.2, size=n_cells),
            "TF_B": rng.normal(size=n_cells),
        },
        index=cell_names,
    )
    atac_mat = rng.poisson(2.0, size=(n_cells, n_peaks)).astype(np.float64)
    atac_mat[:, :6] = rng.poisson(np.exp(1.0 + 1.1 * z)[:, None], size=(n_cells, 6)).astype(np.float64)
    atac_df = pd.DataFrame(atac_mat, index=cell_names, columns=peak_names)
    gc = rng.uniform(0.35, 0.65, size=n_peaks)

    corr_res = run_gene_peak_corr(rna_df, atac_df, tss=tss_df, peak_gc=gc, window_pad=3_000, n_bg=15, seed=1313)
    corr_df = corr_res.to_dataframe()
    assert not corr_df.duplicated(subset=["gene", "peak"]).any()

    # Test motif_to_tf dict + dorc_k_pool > 0 + corr_p_method="z_across_tfs"
    dorc_links = pd.DataFrame(
        {
            "gene": ["G_MULTI"] * 4 + ["G_OTHER"] * 4,
            "peak": peak_names[:4] + peak_names[4:8],
        }
    )
    dorc_scores_df = pd.DataFrame(
        {
            "G_MULTI": z + rng.normal(0.0, 0.1, size=n_cells),
            "G_OTHER": z + rng.normal(0.0, 0.1, size=n_cells),
        },
        index=cell_names,
    )
    motif_mat = np.zeros((n_peaks, 2), dtype=np.float64)
    motif_mat[:8, 0] = 1.0  # MA0001.1 -> TF_A
    motif_mat[20:25, 1] = 1.0  # MA0002.1 -> TF_B
    peak_motifs = sparse.csr_matrix(motif_mat)

    grn = run_figr_grn(
        dorc_scores_df,
        rna_df,
        dorc_links=dorc_links,
        peak_motifs=peak_motifs,
        peak_names=peak_names,
        motif_names=["MA0001.1", "MA0002.1"],
        motif_to_tf={"MA0001.1": "TF_A", "MA0002.1": "TF_B"},
        atac=atac_df,
        peak_gc=gc,
        dorc_k_pool=2,
        n_bg=15,
        corr_p_method="z_across_tfs",
        seed=1313,
    )
    df_grn = grn.to_dataframe()
    assert set(df_grn["tf_gene"]) == {"TF_A", "TF_B"}
    # With dorc_k_pool=2, pooled DORC peak set is union of G_MULTI (4) + G_OTHER (4) = 8 peaks
    assert np.all(df_grn["n_dorc_peaks"] == 8)
    assert df_grn.loc[df_grn["tf_gene"] == "TF_A", "dorc_reg_score"].iloc[0] > 0.0


def test_grn_peak_motif_row_alignment_and_missing_peak_validation():
    rng = np.random.default_rng(1414)
    n_cells = 120
    n_peaks = 100
    cells = [f"c_{i}" for i in range(n_cells)]
    peaks = [f"chr1:{p * 1000}-{p * 1000 + 400}" for p in range(n_peaks)]

    z = rng.normal(size=n_cells)
    dorc_links = pd.DataFrame({"gene": ["D1"] * 10, "peak": peaks[:10]})
    scores = pd.DataFrame({"D1": 2.0 * z + rng.normal(0.0, 0.2, n_cells)}, index=cells)
    rna = pd.DataFrame({"TF1": 2.0 * z + rng.normal(0.0, 0.2, n_cells)}, index=cells)

    motif_arr = np.zeros((n_peaks, 1), dtype=np.float64)
    motif_arr[:10, 0] = 1.0
    motif_arr[90:, 0] = 1.0
    gc = np.full(n_peaks, 0.5)
    gc[90:] = 0.85

    atac = pd.DataFrame(rng.poisson(5.0, (n_cells, n_peaks)).astype(float), index=cells, columns=peaks)
    atac.iloc[:, 90:] = rng.poisson(50.0, (n_cells, 10)).astype(float)

    motifs_aligned = pd.DataFrame(motif_arr, index=peaks, columns=["TF1"])
    motifs_reversed = motifs_aligned.iloc[::-1].copy()
    motifs_shuffled = motifs_aligned.iloc[rng.permutation(n_peaks)].copy()
    motifs_adata_rev = ad.AnnData(
        X=sparse.csr_matrix(motifs_reversed.to_numpy()),
        obs=pd.DataFrame(index=motifs_reversed.index),
        var=pd.DataFrame(index=motifs_reversed.columns),
    )

    grn_ref = run_figr_grn(scores, rna, dorc_links, motifs_aligned, atac=atac, peak_gc=gc, n_bg=40, seed=1)
    for alt_motifs in (motifs_reversed, motifs_shuffled, motifs_adata_rev):
        grn_alt = run_figr_grn(scores, rna, dorc_links, alt_motifs, atac=atac, peak_gc=gc, n_bg=40, seed=1)
        pd.testing.assert_frame_equal(grn_ref.to_dataframe(), grn_alt.to_dataframe())

    # Superset peak_motifs (extra peaks) aligns cleanly to atac peaks
    extra_df = pd.DataFrame(
        np.ones((2, 1)), index=["chr1:999000-999400", "chr1:999500-999900"], columns=["TF1"]
    )
    motifs_superset = pd.concat([extra_df, motifs_reversed])
    grn_sup = run_figr_grn(scores, rna, dorc_links, motifs_superset, atac=atac, peak_gc=gc, n_bg=40, seed=1)
    pd.testing.assert_frame_equal(grn_ref.to_dataframe(), grn_sup.to_dataframe())

    # Missing required atac peak raises ValueError cleanly
    motifs_missing = motifs_aligned.iloc[1:].copy()
    with pytest.raises(ValueError, match="missing"):
        run_figr_grn(scores, rna, dorc_links, motifs_missing, atac=atac, peak_gc=gc, n_bg=40, seed=1)


def test_kartha_union_small_p_precision_no_cancellation():
    rng = np.random.default_rng(1515)
    n_cells = 140
    n_peaks = 150
    cells = [f"c_{i}" for i in range(n_cells)]
    peaks = [f"chr1:{p * 1000}-{p * 1000 + 400}" for p in range(n_peaks)]

    z = rng.normal(size=n_cells)
    dorc_links = pd.DataFrame({"gene": ["D1"] * 15, "peak": peaks[:15]})
    scores = pd.DataFrame({"D1": 3.0 * z + rng.normal(0.0, 0.05, n_cells)}, index=cells)
    rna = pd.DataFrame({"TF_ULTRA": 3.0 * z + rng.normal(0.0, 0.05, n_cells)}, index=cells)

    motif_mat = np.zeros((n_peaks, 1), dtype=np.float64)
    motif_mat[:15, 0] = 1.0  # all 15 DORC peaks hit, 0/135 background hits -> p_motif << 1e-16
    motifs_df = pd.DataFrame(motif_mat, index=peaks, columns=["TF_ULTRA"])

    grn = run_figr_grn(
        scores,
        rna,
        dorc_links=dorc_links,
        peak_motifs=motifs_df,
        score_mode="kartha_union",
        seed=1515,
    )
    row = grn.to_dataframe().iloc[0]
    p_m = float(row["p_motif"])
    p_c = float(row["p_corr"])
    assert p_m < 1e-16
    assert p_c < 1e-16
    expected_union = max(1e-300, min(1.0, p_m + p_c - p_m * p_c))
    expected_score = -np.log10(expected_union)
    assert row["dorc_reg_score"] < 290.0
    np.testing.assert_allclose(row["dorc_reg_score"], expected_score, rtol=1e-12, atol=1e-12)


def test_unsorted_csc_unit_rank_sparse_col_parity():
    from figr.corr import _unit_rank_sparse_col, compute_unit_rank_matrix

    rng = np.random.default_rng(1616)
    n = 15
    dense_col0 = rng.normal(size=n)
    # Introduce ties and zeros
    dense_col0[2] = dense_col0[0]
    dense_col1 = np.zeros(n, dtype=np.float64)
    dense_col1[[1, 4, 9]] = [2.5, -1.0, 2.5]
    dense_mat = np.column_stack([dense_col0, dense_col1])

    # Construct an unsorted CSC matrix with nnz == n in col 0 (and unsorted entries in col 1)
    perm0 = rng.permutation(n)
    perm1 = np.array([9, 1, 4], dtype=np.int64)
    indptr = np.array([0, n, n + 3], dtype=np.int64)
    indices = np.concatenate([perm0, perm1]).astype(np.int64)
    data = np.concatenate([dense_col0[perm0], dense_col1[perm1]]).astype(np.float64)
    unsorted_csc = sparse.csc_matrix((data, indices, indptr), shape=(n, 2))
    unsorted_csc.has_sorted_indices = False
    assert not unsorted_csc.has_canonical_format

    ok0, u0 = _unit_rank_sparse_col(unsorted_csc, 0)
    ok1, u1 = _unit_rank_sparse_col(unsorted_csc, 1)
    assert ok0 and ok1

    U_mat, valid = compute_unit_rank_matrix(unsorted_csc, [0, 1], method="spearman")
    assert np.all(valid)

    # Compare pairwise correlation and direct rank vectors against dense computation
    U_dense, valid_dense = compute_unit_rank_matrix(dense_mat, [0, 1], method="spearman")
    assert np.all(valid_dense)
    np.testing.assert_allclose(u0, U_dense[:, 0], atol=1e-12)
    np.testing.assert_allclose(u1, U_dense[:, 1], atol=1e-12)
    np.testing.assert_allclose(U_mat, U_dense, atol=1e-12)
    expected_r = stats.spearmanr(dense_col0, dense_col1).statistic
    np.testing.assert_allclose(float(u0 @ u1), expected_r, atol=1e-12)


def test_build_knn_graph_two_by_two_disambiguation():
    eye2 = np.eye(2, dtype=int)
    for inc_self in (True, False):
        W_eye = build_knn_graph(knn=eye2, include_self=inc_self).toarray()
        np.testing.assert_allclose(W_eye, np.eye(2, dtype=np.float64), atol=1e-12)

    # Explicit override to treat (2, 2) identity as neighbor indices
    W_eye_idx = build_knn_graph(knn=eye2, knn_type="indices").toarray()
    np.testing.assert_allclose(W_eye_idx, np.full((2, 2), 0.5), atol=1e-12)

    # Standard (2, 2) neighbor index matrix [[0, 1], [1, 0]]
    nbr2 = np.array([[0, 1], [1, 0]], dtype=int)
    W_nbr = build_knn_graph(knn=nbr2, include_self=True).toarray()
    np.testing.assert_allclose(W_nbr, np.full((2, 2), 0.5), atol=1e-12)

    # Explicit adjacency override for off-diagonal (2, 2) binary matrix without self-loops
    W_off_adj = build_knn_graph(knn=nbr2, include_self=False, knn_type="adjacency").toarray()
    np.testing.assert_allclose(W_off_adj, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


