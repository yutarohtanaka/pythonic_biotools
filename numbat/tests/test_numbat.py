"""Comprehensive verification suite for `numbat`."""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import is_dataclass

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import logsumexp
from scipy.stats import betabinom, binom, poisson
from sklearn.metrics import roc_auc_score

import numbat
import numbat.hmm as hmm_mod
import numbat.io as io_mod
from numbat import (
    DEFAULT_STATES,
    S15_STATES,
    CloneAssignment,
    HMMResult,
    NumbatConfig,
    NumbatResult,
    PhylogenyTree,
    SegmentCall,
    annotate_anndata,
    beta_binomial_logpmf,
    compute_ref_expression,
    fit_allele_hmm,
    fit_expression_hmm,
    has_numba,
    phase_switch_prob,
    pln_logpmf,
    reconstruct_phylogeny,
    retest_bulks,
    run_joint_hmm,
    run_numbat,
    score_single_cells,
)
from numbat.clones import JOINT_POST_COLUMNS, SEGMENTS_COLUMNS, _sigmoid


def _is_rooted_tree(tree: PhylogenyTree) -> bool:
    """Verify connectivity, acyclicity (|E| == |V| - 1), and unique root."""
    if tree.n_cells <= 1:
        return len(tree.edges) == 0
    nodes = set(tree.node_labels.keys())
    for u, v in tree.edges:
        nodes.add(u)
        nodes.add(v)
    if len(tree.edges) != len(nodes) - 1:
        return False
    in_deg = {n: 0 for n in nodes}
    adj: dict[int, list[int]] = {n: [] for n in nodes}
    for u, v in tree.edges:
        in_deg[v] = in_deg.get(v, 0) + 1
        adj[u].append(v)
    roots = [n for n, d in in_deg.items() if d == 0]
    if len(roots) != 1 or any(d > 1 for d in in_deg.values()):
        return False
    visited = set()
    stack = [roots[0]]
    while stack:
        cur = stack.pop()
        if cur in visited:
            return False
        visited.add(cur)
        stack.extend(adj[cur])
    return visited == nodes


def _simulate_cohort(
    rng: np.random.Generator,
    n_ref: int = 40,
    n_unlabeled_normal: int = 40,
    clone_specs: list[tuple[int, dict[str, tuple[str, float, float]]]] | None = None,
    n_genes_per_chrom: int = 40,
    n_snps_per_chrom: int = 35,
    chroms: tuple[str, ...] = ("chr1", "chr3", "chr7", "chr10"),
    gamma: float = 25.0,
    sigma_val: float = 0.25,
    mean_lib_size: float = 4000.0,
    mean_snp_dp: float = 12.0,
) -> tuple[ad.AnnData, pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate a multi-clone scRNA-seq + phased allele cohort with planted CNVs."""
    if clone_specs is None:
        clone_specs = [
            (
                60,
                {
                    "chr1": ("del", 0.50, 0.90),
                    "chr7": ("amp", 1.50, 2.0 / 3.0),
                    "chr3": ("loh", 1.00, 0.90),
                },
            )
        ]

    genes = []
    for c in chroms:
        for j in range(n_genes_per_chrom):
            pos = (j + 1) * 200_000
            genes.append({"gene": f"{c}_G{j}", "chrom": c, "start": pos, "end": pos + 50_000})
    var_df = pd.DataFrame(genes).set_index("gene")
    n_genes = len(var_df)

    base_weights = rng.uniform(0.6, 1.5, size=n_genes)
    base_lambda = base_weights / np.sum(base_weights)

    snps = []
    for c in chroms:
        for j in range(n_snps_per_chrom):
            pos = (j + 1) * 230_000 + 25_000
            snps.append((c, pos, (j + 1) * 0.25, f"{c}_S{j}"))

    cell_records = []
    true_compartment = []
    true_clone = []

    for i in range(n_ref):
        cell_records.append((f"ref_{i}", True, {}))
        true_compartment.append("normal")
        true_clone.append(0)
    for i in range(n_unlabeled_normal):
        cell_records.append((f"norm_{i}", False, {}))
        true_compartment.append("normal")
        true_clone.append(0)
    for cl_idx, (n_cl, cnv_map) in enumerate(clone_specs, start=1):
        for i in range(n_cl):
            cell_records.append((f"tumor_c{cl_idx}_{i}", False, cnv_map))
            true_compartment.append("tumor")
            true_clone.append(cl_idx)

    n_cells = len(cell_records)
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    gene_chroms = var_df["chrom"].to_numpy()

    for c_idx, (_cid, _is_ref, cnv_map) in enumerate(cell_records):
        lib = rng.uniform(0.8 * mean_lib_size, 1.2 * mean_lib_size)
        phi_g = np.ones(n_genes, dtype=np.float64)
        for chrom_name, (_st, phi_v, _mu_v) in cnv_map.items():
            phi_g[gene_chroms == chrom_name] = phi_v
        log_rate = np.log(lib * base_lambda * phi_g) + rng.normal(0.0, sigma_val, size=n_genes)
        counts[c_idx] = rng.poisson(np.exp(log_rate))

    # Simulate phased alleles with occasional mid-chromosome haplotype switch errors
    allele_rows = []
    for c_idx, (cid, _is_ref, cnv_map) in enumerate(cell_records):
        for s_chrom, s_pos, s_cm, s_id in snps:
            dp = int(rng.poisson(mean_snp_dp))
            if dp <= 0:
                continue
            if s_chrom in cnv_map:
                _st, _phi_v, mu_v = cnv_map[s_chrom]
                # Flip haplotype phase halfway through chr1 to test phase-switch resilience
                if s_chrom == "chr1" and s_pos > (n_snps_per_chrom // 2) * 230_000:
                    mu_eff = 1.0 - mu_v
                else:
                    mu_eff = mu_v
            else:
                mu_eff = 0.5
            p_draw = rng.beta(gamma * mu_eff, gamma * (1.0 - mu_eff))
            ad_val = int(rng.binomial(dp, p_draw))
            allele_rows.append(
                {
                    "cell": cid,
                    "chrom": s_chrom,
                    "pos": s_pos,
                    "cM": s_cm,
                    "snp_id": s_id,
                    "AD": ad_val,
                    "DP": dp,
                }
            )

    obs_df = pd.DataFrame(
        {
            "is_ref": [rec[1] for rec in cell_records],
            "true_compartment": true_compartment,
            "true_clone": true_clone,
        },
        index=[rec[0] for rec in cell_records],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs_df, var=var_df)
    allele_df = pd.DataFrame(allele_rows)
    return adata, allele_df, np.array(true_compartment), np.array(true_clone)


def test_beta_binomial_and_pln_analytical() -> None:
    """Verify Beta-Binomial & PLN normalization, SciPy parity, and asymptotic limits."""
    n = 20
    y = np.arange(n + 1, dtype=np.float64)
    mu = 2.0 / 3.0
    gamma = 20.0
    bb_lp = beta_binomial_logpmf(y, n, mu, gamma)
    scipy_lp = betabinom.logpmf(y, n, gamma * mu, gamma * (1.0 - mu))
    np.testing.assert_allclose(bb_lp, scipy_lp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.exp(logsumexp(bb_lp)), 1.0, atol=1e-12)

    # Large-gamma Binomial asymptote and n=0 / mu-boundary safety
    for large_gamma in (1e8, 1e12):
        bb_large = beta_binomial_logpmf(y, n, mu, large_gamma)
        bin_lp = binom.logpmf(y, n, mu)
        np.testing.assert_allclose(bb_large, bin_lp, rtol=1e-6, atol=1e-6)

    assert float(beta_binomial_logpmf(0.0, 0.0, 0.9, 20.0)) == 0.0
    edge_lp = beta_binomial_logpmf(np.array([0.0, 10.0]), 10.0, np.array([0.0, 1.0]), 20.0)
    assert np.all(np.isfinite(edge_lp))

    # Gauss-Hermite PLN sums to 1 and converges to Poisson as sigma -> 0+
    x = np.arange(81, dtype=np.float64)
    pln_lp = pln_logpmf(x, mu=1.5, sigma=0.4, n_quadrature=21)
    assert np.all(np.isfinite(pln_lp))
    np.testing.assert_allclose(np.exp(logsumexp(pln_lp)), 1.0, atol=1e-5)

    pln_zero_sig = pln_logpmf(x, mu=1.5, sigma=1e-8, n_quadrature=21)
    pois_lp = poisson.logpmf(x, mu=math.exp(1.5))
    np.testing.assert_allclose(pln_zero_sig, pois_lp, rtol=1e-10, atol=1e-10)


def test_hmm_log_space_long_chromosome_no_underflow() -> None:
    """Verify log-space HMM avoids float64 underflow at T=1000 and matches brute-force at T=6."""
    rng = np.random.default_rng(101)
    t_long = 1000
    k_states = len(numbat.DEFAULT_STATES)
    # Emissions ~ 1e-4 per locus -> joint likelihood ~ 1e-4000 << 1e-308
    log_emit = math.log(1e-4) + rng.normal(0.0, 0.5, size=(t_long, k_states))
    res = run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=[0, t_long],
        p_switch=np.zeros(t_long),
        t_cnv=1e-4,
        backend="numpy",
    )
    assert math.isfinite(res.log_likelihood)
    assert not np.any(np.isnan(res.posteriors))
    np.testing.assert_allclose(res.posteriors.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(res.collapsed_posteriors.sum(axis=1), 1.0, atol=1e-12)

    # Brute-force log-likelihood check over all 5^4 = 625 paths on a 5-state, T=4 slice
    states_5 = numbat.DEFAULT_STATES[:5]
    emit_short = rng.normal(-2.0, 0.8, size=(4, 5))
    ps_short = np.array([0.0, 0.1, 0.25, 0.05])
    cfg_short = NumbatConfig(states=states_5, t_cnv=0.02, p_init_cnv=0.2, backend="numpy")
    hmm_short = run_joint_hmm(
        log_emit=emit_short,
        chrom_offsets=[0, 4],
        p_switch=ps_short,
        t_cnv=cfg_short.t_cnv,
        p_init_cnv=cfg_short.p_init_cnv,
        states=states_5,
        backend="numpy",
    )
    (
        sub_id,
        sub_size,
        orient,
        n_sub,
        _s2c,
        log_pi,
    ) = hmm_mod._encode_states(states_5, cfg_short.p_init_cnv)
    path_lls = []
    for path in itertools.product(range(5), repeat=4):
        ll = float(log_pi[path[0]] + emit_short[0, path[0]])
        for t in range(1, 4):
            a_mat = hmm_mod._build_log_transition_matrix(
                float(ps_short[t]), cfg_short.t_cnv, sub_id, sub_size, orient, n_sub
            )
            ll += float(a_mat[path[t - 1], path[t]] + emit_short[t, path[t]])
        path_lls.append(ll)
    brute_ll = float(logsumexp(np.array(path_lls)))
    np.testing.assert_allclose(hmm_short.log_likelihood, brute_ll, rtol=1e-12, atol=1e-12)


def test_numba_numpy_hmm_parity_and_monkeypatch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify exact numerical parity between Numba and NumPy backends + missing-Numba fallback."""
    rng = np.random.default_rng(202)
    t_len = 180
    k_states = len(numbat.DEFAULT_STATES)
    log_emit = rng.normal(-5.0, 1.5, size=(t_len, k_states))
    offsets = np.array([0, 65, 130, 180], dtype=np.int64)
    p_switch = np.asarray(phase_switch_prob(rng.uniform(0.0, 20.0, size=t_len), nu=1.0))
    p_switch[offsets[:-1]] = 0.0

    res_np = run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=offsets,
        p_switch=p_switch,
        t_cnv=1e-5,
        backend="numpy",
    )
    assert res_np.backend_used == "numpy"

    if has_numba():
        res_nb = run_joint_hmm(
            log_emit=log_emit,
            chrom_offsets=offsets,
            p_switch=p_switch,
            t_cnv=1e-5,
            backend="numba",
        )
        assert res_nb.backend_used == "numba"
        np.testing.assert_allclose(res_nb.log_likelihood, res_np.log_likelihood, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_nb.posteriors, res_np.posteriors, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            res_nb.collapsed_posteriors, res_np.collapsed_posteriors, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(res_nb.viterbi_states, res_np.viterbi_states)

    # Simulate CI environment without numba
    monkeypatch.setattr(hmm_mod, "_HAS_NUMBA", False)
    assert not has_numba()
    res_auto = run_joint_hmm(
        log_emit=log_emit,
        chrom_offsets=offsets,
        p_switch=p_switch,
        t_cnv=1e-5,
        backend="auto",
    )
    assert res_auto.backend_used == "numpy"
    np.testing.assert_allclose(res_auto.posteriors, res_np.posteriors, rtol=1e-12, atol=1e-12)
    with pytest.raises(ImportError, match="numba"):
        run_joint_hmm(log_emit=log_emit, chrom_offsets=offsets, backend="numba")


def test_phase_switch_handling_and_haplotype_symmetry() -> None:
    """Verify phase-switch resilience inside a deletion and global m<->p symmetry."""
    rng = np.random.default_rng(303)
    n_snps = 80
    dp = np.full(n_snps, 25.0)
    dist_cm = np.full(n_snps, 0.2)
    dist_cm[0] = 0.0
    dist_cm[40] = 20.0  # Strong phase switch gap at SNP 40

    ad = np.zeros(n_snps, dtype=np.float64)
    for t in range(n_snps):
        mu_true = 0.90 if t < 40 else 0.10
        p = rng.beta(25.0 * mu_true, 25.0 * (1.0 - mu_true))
        ad[t] = rng.binomial(int(dp[t]), p)

    cfg = NumbatConfig(gamma=25.0, nu=1.0, backend="numpy")
    res = fit_allele_hmm(ad=ad, dp=dp, dist_cm=dist_cm, config=cfg)

    p_del_idx = res.collapsed_names.index("del")
    p_loh_idx = res.collapsed_names.index("loh")
    # Alleles alone identify allelic loss (del or loh) identically across the phase switch
    assert np.all(res.collapsed_posteriors[:, p_del_idx] + res.collapsed_posteriors[:, p_loh_idx] > 0.98)
    assert all(st in ("del", "loh") for st in res.viterbi_collapsed)
    # First half should prefer _p, second half _m
    v_names = [res.state_names[i] for i in res.viterbi_states]
    assert sum(name.endswith("_p") for name in v_names[:35]) >= 30
    assert sum(name.endswith("_m") for name in v_names[45:]) >= 30

    # Global haplotype flip (y -> n - y) must preserve collapsed posteriors and flip _p <-> _m
    res_flip = fit_allele_hmm(ad=dp - ad, dp=dp, dist_cm=dist_cm, config=cfg)
    np.testing.assert_allclose(
        res.collapsed_posteriors, res_flip.collapsed_posteriors, rtol=1e-12, atol=1e-12
    )
    assert res.viterbi_collapsed == res_flip.viterbi_collapsed
    for s1, s2 in zip(v_names, [res_flip.state_names[i] for i in res_flip.viterbi_states]):
        if s1.endswith("_p"):
            assert s2 == s1[:-2] + "_m"
        elif s1.endswith("_m"):
            assert s2 == s1[:-2] + "_p"

    # With deletion expression (phi=0.5), joint HMM resolves p_del > 0.95 across the phase switch
    # and extracts ONE contiguous 'del' segment rather than fragmenting at SNP 40
    mu_vec = np.array([s.mu for s in cfg.states])
    phi_vec = np.array([s.phi for s in cfg.states])
    expr_obs = rng.poisson(15.0 * 0.5, size=n_snps).astype(np.float64)
    base_mu = np.full(n_snps, math.log(15.0))
    sigma_vec = np.full(n_snps, 0.25)
    log_emit_joint = pln_logpmf(
        expr_obs[:, None], base_mu[:, None] + np.log(phi_vec)[None, :], sigma_vec[:, None]
    ) + beta_binomial_logpmf(ad[:, None], dp[:, None], mu_vec[None, :], cfg.gamma)
    p_sw = np.asarray(phase_switch_prob(dist_cm, nu=cfg.nu), dtype=np.float64)
    p_sw[0] = 0.0
    res_joint = run_joint_hmm(
        log_emit=log_emit_joint,
        chrom_offsets=[0, n_snps],
        p_switch=p_sw,
        t_cnv=cfg.t_cnv,
        backend="numpy",
    )
    assert np.all(res_joint.collapsed_posteriors[:, p_del_idx] > 0.95)
    assert all(st == "del" for st in res_joint.viterbi_collapsed)


def test_cnloh_and_balanced_cnv_recovery() -> None:
    """Verify joint HMM separates neu, loh (phi=1, BAF skewed), amp, and bamp (phi=2, BAF=0.5)."""
    rng = np.random.default_rng(404)
    chroms = ["chr1", "chr2", "chr3", "chr4"]
    specs = {
        "chr1": ("neu", 1.0, 0.50),
        "chr2": ("loh", 1.0, 0.90),
        "chr3": ("amp", 1.5, 2.0 / 3.0),
        "chr4": ("bamp", 2.0, 0.50),
    }

    n_per_chrom = 45
    n_genes = len(chroms) * n_per_chrom
    ref_lambda = np.full(n_genes, 1.0 / n_genes)
    lib_bulk = 50_000.0
    sigma = np.full(n_genes, 0.20)

    expr_counts = np.zeros(n_genes, dtype=np.float64)
    ad = np.zeros(n_genes, dtype=np.float64)
    dp = np.full(n_genes, 30.0)
    chrom_list = []

    for c_i, c in enumerate(chroms):
        _st, phi_v, mu_v = specs[c]
        for j in range(n_per_chrom):
            idx = c_i * n_per_chrom + j
            chrom_list.append(c)
            rate = math.exp(math.log(lib_bulk * ref_lambda[idx] * phi_v) + rng.normal(0.0, 0.20))
            expr_counts[idx] = rng.poisson(rate)
            p_b = rng.beta(25.0 * mu_v, 25.0 * (1.0 - mu_v))
            ad[idx] = rng.binomial(30, p_b)

    cfg = NumbatConfig(gamma=25.0, backend="numpy")
    expr_only = fit_expression_hmm(expr_counts, ref_lambda, sigma=sigma, chrom=chrom_list, config=cfg)

    # Build joint log emissions directly from PLN + Beta-Binomial
    phi_vec = np.array([s.phi for s in cfg.states])
    mu_vec = np.array([s.mu for s in cfg.states])
    base_mu = np.log(lib_bulk * ref_lambda)
    log_emit = pln_logpmf(
        expr_counts[:, None], base_mu[:, None] + np.log(phi_vec)[None, :], sigma[:, None]
    ) + beta_binomial_logpmf(ad[:, None], dp[:, None], mu_vec[None, :], cfg.gamma)
    offsets = np.array([0, 45, 90, 135, 180], dtype=np.int64)
    joint_res = run_joint_hmm(log_emit, chrom_offsets=offsets, t_cnv=cfg.t_cnv, backend="numpy")

    for c_i, c in enumerate(chroms):
        expected_st = specs[c][0]
        sl = slice(c_i * n_per_chrom, (c_i + 1) * n_per_chrom)
        calls = joint_res.viterbi_collapsed[sl]
        assert sum(st == expected_st for st in calls) >= 40
        if expected_st == "loh":
            # Expression-only cannot distinguish CN-LOH (phi=1.0) from neutral
            assert sum(st == "neu" for st in expr_only.viterbi_collapsed[sl]) >= 40


def test_chromosome_boundary_reset() -> None:
    """Verify HMM resets state prior at chromosome boundary even with tiny t_cnv=1e-12."""
    rng = np.random.default_rng(505)
    k_states = len(numbat.DEFAULT_STATES)
    n1, n2 = 50, 50
    emit = np.full((n1 + n2, k_states), -10.0)
    # chr1 strongly favors del_p (state 1); chr2 mildly favors neu (state 0)
    emit[:n1, 1] = -1.0 + rng.normal(0.0, 0.1, size=n1)
    emit[n1:, 0] = -1.0 + rng.normal(0.0, 0.1, size=n2)
    emit[n1:, 1] = -1.8

    res = run_joint_hmm(
        log_emit=emit,
        chrom_offsets=[0, n1, n1 + n2],
        t_cnv=1e-12,
        p_init_cnv=0.15,
        backend="numpy",
    )
    assert all(st == "del" for st in res.viterbi_collapsed[:n1])
    assert all(st == "neu" for st in res.viterbi_collapsed[n1:])


def test_malignant_vs_normal_classification_anndata_sparse() -> None:
    """Verify sparse vs dense AnnData parity and >0.98 ROC-AUC on malignant vs normal cells."""
    rng = np.random.default_rng(606)
    adata_sp, allele_df, true_comp, _ = _simulate_cohort(
        rng,
        n_ref=40,
        n_unlabeled_normal=50,
        clone_specs=[
            (
                60,
                {
                    "chr1": ("del", 0.50, 0.90),
                    "chr7": ("amp", 1.50, 2.0 / 3.0),
                    "chr3": ("loh", 1.00, 0.90),
                },
            )
        ],
    )
    adata_dense = adata_sp.copy()
    adata_dense.X = adata_sp.X.toarray()

    cfg = NumbatConfig(backend="auto", max_iter=2)
    res_sp = run_numbat(adata_sp, allele_df=allele_df, ref_cells="is_ref", config=cfg)
    res_dn = run_numbat(adata_dense, allele_df=allele_df, ref_cells="is_ref", config=cfg)

    np.testing.assert_allclose(res_sp.clones.p_cnv, res_dn.clones.p_cnv, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(res_sp.clones.clone_opt, res_dn.clones.clone_opt)

    called_states = {s.chrom: s.cnv_state for s in res_sp.segments}
    assert called_states.get("chr1") == "del"
    assert called_states.get("chr7") == "amp"
    assert called_states.get("chr3") == "loh"

    # Evaluate on unlabeled cells (excluding labeled ref_cells)
    unlabeled_mask = ~adata_sp.obs["is_ref"].to_numpy(dtype=bool)
    y_true = (true_comp[unlabeled_mask] == "tumor").astype(int)
    p_scores = res_sp.clones.p_cnv[unlabeled_mask]
    auc = roc_auc_score(y_true, p_scores)
    acc = float(np.mean((p_scores >= 0.5).astype(int) == y_true))
    assert auc >= 0.98
    assert acc >= 0.95
    assert float(np.mean(res_sp.clones.aneuploidy_score[true_comp == "tumor"])) > (
        float(np.mean(res_sp.clones.aneuploidy_score[true_comp == "normal"])) + 0.4
    )


def test_phylogeny_nj_nni_and_subclonal_assignment() -> None:
    """Verify NJ tie-breaking, acyclic NNI ML optimization, and 3-subclone tree recovery."""
    rng = np.random.default_rng(707)

    # 1. Direct NJ + NNI validation on a planted 3-branch tree with exact normal ties
    true_genos = np.array(
        [
            [0, 0, 0],  # Normal (duplicated rows force Q-matrix ties)
            [1, 0, 0],  # Clone A
            [1, 1, 0],  # Clone B
            [1, 0, 1],  # Clone C
        ],
        dtype=np.float64,
    )
    cell_labels = np.repeat(np.arange(4), 15)
    prob_direct = np.where(true_genos[cell_labels] == 1, 0.96, 0.04)
    # Inject small flip noise in 2 cells to verify NNI preserves/improves ML & acyclicity
    prob_direct[16, 1] = 0.45
    prob_direct[31, 2] = 0.45
    tree_dir, clones_dir = reconstruct_phylogeny(
        prob_mat=prob_direct,
        cell_ids=[f"c_{i}" for i in range(60)],
        event_names=["chr1_del", "chr7_amp", "chr3_loh"],
        min_cells=8,
        tau=0.3,
    )
    assert _is_rooted_tree(tree_dir)
    assert np.all(tree_dir.branch_lengths >= 0.0)
    assert tree_dir.log_likelihood >= tree_dir.initial_log_likelihood - 1e-12
    assert tree_dir.n_clones == 4  # normal + 3 tumor subclones
    np.testing.assert_allclose(clones_dir.clone_probs.sum(axis=1), 1.0, atol=1e-12)

    # 2. End-to-end 3-clone tumor phylogeny recovery via run_numbat
    adata, allele_df, _true_comp, true_clone = _simulate_cohort(
        rng,
        n_ref=30,
        n_unlabeled_normal=10,
        clone_specs=[
            (30, {"chr1": ("del", 0.50, 0.90)}),
            (30, {"chr1": ("del", 0.50, 0.90), "chr7": ("amp", 1.50, 2.0 / 3.0)}),
            (30, {"chr1": ("del", 0.50, 0.90), "chr3": ("loh", 1.00, 0.90)}),
        ],
        n_genes_per_chrom=40,
        n_snps_per_chrom=35,
        chroms=("chr1", "chr3", "chr7", "chr10"),
    )
    res = run_numbat(
        adata,
        allele_df=allele_df,
        ref_cells="is_ref",
        init_k=4,
        config=NumbatConfig(max_iter=2, min_cells=10, backend="auto"),
    )
    assert _is_rooted_tree(res.phylogeny)
    assert res.phylogeny.log_likelihood >= res.phylogeny.initial_log_likelihood - 1e-12
    np.testing.assert_allclose(res.clones.clone_probs.sum(axis=1), 1.0, atol=1e-12)

    # Verify >= 3 distinct tumor clones + normal clone recovered and >= 90% clone accuracy
    assert len(res.clones.clone_ids) >= 4
    # Map each predicted clone to its majority true clone and measure overall purity
    pred = res.clones.clone_opt
    correct = 0
    for cid in res.clones.clone_ids:
        mask = pred == cid
        if np.any(mask):
            vals, cnts = np.unique(true_clone[mask], return_counts=True)
            correct += int(np.max(cnts))
    assert correct / float(len(true_clone)) >= 0.90


def test_null_calibration_pure_diploid() -> None:
    """Verify purely diploid cohort yields 0 CNV segments and controlled false-positive rate."""
    rng = np.random.default_rng(808)
    adata, allele_df, _comp, _clone = _simulate_cohort(
        rng,
        n_ref=30,
        n_unlabeled_normal=50,
        clone_specs=[],
        n_genes_per_chrom=40,
        n_snps_per_chrom=30,
        chroms=("chr1", "chr2", "chr3"),
        gamma=30.0,
        sigma_val=0.25,
    )
    res = run_numbat(adata, allele_df=allele_df, ref_cells="is_ref")
    assert len(res.segments) == 0
    assert res.segments_dataframe().empty
    assert list(res.segments_dataframe().columns) == SEGMENTS_COLUMNS
    assert float(np.median(res.clones.p_cnv)) < 0.1
    fp_rate = float(np.mean(np.array(res.clones.compartment_opt) == "tumor"))
    assert fp_rate <= 0.05


def test_dataclass_api_anndata_annotation_and_zero_segments() -> None:
    """Verify frozen dataclasses, DataFrame schemas, summary(), and annotate_anndata()."""
    for cls in (NumbatResult, SegmentCall, CloneAssignment, PhylogenyTree, HMMResult):
        assert is_dataclass(cls)

    rng = np.random.default_rng(909)
    adata, allele_df, _comp, _clone = _simulate_cohort(
        rng,
        n_ref=20,
        n_unlabeled_normal=10,
        clone_specs=[(25, {"chr1": ("del", 0.50, 0.90)})],
        n_genes_per_chrom=35,
        n_snps_per_chrom=30,
        chroms=("chr1", "chr2"),
    )
    res = run_numbat(adata, allele_df=allele_df, ref_cells="is_ref")

    with pytest.raises((AttributeError, TypeError)):
        res.backend_used = "other"  # type: ignore[misc]

    df_cells = res.to_dataframe()
    df_clones = res.clones_dataframe()
    df_segs = res.segments_dataframe()
    df_jp = res.joint_post_dataframe()
    df_tree = res.phylogeny.to_dataframe()

    assert len(df_cells) == adata.n_obs
    assert {"cell", "clone_opt", "compartment_opt", "p_cnv", "aneuploidy_score"}.issubset(
        df_cells.columns
    )
    assert {"clone_id", "compartment", "n_cells", "fraction", "genotype"}.issubset(
        df_clones.columns
    )
    assert list(df_segs.columns) == SEGMENTS_COLUMNS
    assert list(df_jp.columns) == JOINT_POST_COLUMNS
    assert {"parent", "child", "branch_length", "events"}.issubset(df_tree.columns)
    assert "NumbatResult Summary" in res.summary()
    assert res.phylogeny.to_newick().endswith(";")
    assert isinstance(res.segments[0].to_dataframe(), pd.DataFrame)

    annotated = annotate_anndata(adata, res, inplace=False)
    assert "clone_opt" in annotated.obs.columns
    assert "compartment_opt" in annotated.obs.columns
    assert "p_cnv" in annotated.obs.columns
    assert "X_numbat_clone_probs" in annotated.obsm
    assert "numbat" in annotated.uns
    assert "clone_opt" not in adata.obs.columns


def test_edge_cases_and_input_validation() -> None:
    """Verify unsorted coords, expression-only mode, zero-read cells, C=2/M=0 trees, and errors."""
    rng = np.random.default_rng(1010)
    # 1. Unsorted chromosomes/positions + DataFrame input + expression-only (allele_df=None)
    genes = []
    for c in ("chr10", "chr2", "chr1"):
        for j in range(30):
            genes.append({"gene": f"{c}_g{j}", "chrom": c, "pos": (30 - j) * 100_000})
    gene_info = pd.DataFrame(genes)
    n_genes = len(gene_info)
    counts = rng.poisson(20.0, size=(30, n_genes)).astype(np.float64)
    # Zero out cell 0 completely to test L_c == 0 edge case
    counts[0, :] = 0.0
    # Plant a strong amplification on chr1 in cells 15..29
    chr1_mask = (gene_info["chrom"] == "chr1").to_numpy()
    counts[15:, chr1_mask] = rng.poisson(45.0, size=(15, int(np.sum(chr1_mask))))

    count_df = pd.DataFrame(
        counts,
        index=[f"cell_{i}" for i in range(30)],
        columns=gene_info["gene"].tolist(),
    )
    res_expr_only = run_numbat(
        count_df,
        allele_df=None,
        gene_info=gene_info,
        ref_cells=[f"cell_{i}" for i in range(1, 15)],
        config=NumbatConfig(max_iter=1),
    )
    assert not np.any(np.isnan(res_expr_only.clones.p_cnv))
    assert len(res_expr_only.segments) >= 1
    assert res_expr_only.joint_post["p_cnv_y"].eq(0.5).all()

    # 2. M=0 and C=2 phylogeny edge cases
    tree_m0, clones_m0 = reconstruct_phylogeny(np.zeros((5, 0)), cell_ids=[f"c{i}" for i in range(5)])
    assert tree_m0.n_events == 0
    assert tree_m0.n_clones == 1
    assert _is_rooted_tree(tree_m0)
    assert all(c == "normal" for c in clones_m0.compartment_opt)

    tree_c2, _ = reconstruct_phylogeny(np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert tree_c2.n_cells == 2
    assert _is_rooted_tree(tree_c2)

    # 3. Explicit ValueError validation
    with pytest.raises(ValueError, match="AD > DP"):
        bad_alleles = pd.DataFrame(
            {"cell": ["cell_0"], "chrom": ["chr1"], "pos": [100], "AD": [15], "DP": [10]}
        )
        run_numbat(count_df, allele_df=bad_alleles, gene_info=gene_info)

    with pytest.raises(ValueError, match="non-negative"):
        bad_counts = count_df.copy()
        bad_counts.iloc[0, 0] = -1.0
        run_numbat(bad_counts, gene_info=gene_info)

    with pytest.raises(ValueError, match="chrom"):
        run_numbat(count_df, gene_info=gene_info.drop(columns=["chrom"]))

    with pytest.raises(ValueError, match="gamma"):
        NumbatConfig(gamma=-1.0)


def test_gaussian_logratio_planted_cnv_recovery_no_warnings() -> None:
    """Verify expr_model='gaussian_logratio' recovers planted CNVs with zero warnings."""
    rng = np.random.default_rng(20261119)
    n_ref, n_norm, n_tumor = 20, 10, 20
    n_cells = n_ref + n_norm + n_tumor
    n_per_chrom = 25
    chroms = ("chr1", "chr2", "chr3")
    n_genes = len(chroms) * n_per_chrom

    var_df = pd.DataFrame(
        {
            "chrom": [c for c in chroms for _ in range(n_per_chrom)],
            "pos": [(j + 1) * 100_000 for _ in chroms for j in range(n_per_chrom)],
        },
        index=[f"{c}_G{j}" for c in chroms for j in range(n_per_chrom)],
    )
    obs_df = pd.DataFrame(
        {
            "grp": ["ref"] * n_ref + ["norm"] * n_norm + ["tumor"] * n_tumor,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    # Signed log-ratios with non-zero gene-specific baseline on chr2 and sigma=0.2
    lr = rng.normal(0.0, 0.2, size=(n_cells, n_genes))
    chr2_mask = (var_df["chrom"] == "chr2").to_numpy()
    chr1_mask = (var_df["chrom"] == "chr1").to_numpy()
    lr[:, chr2_mask] -= 0.35

    # Plant +log(1.5) gain on chr1 and +log(0.5) deletion on chr2 in tumor cells
    lr[n_ref + n_norm :, chr1_mask] += math.log(1.5)
    lr[n_ref + n_norm :, chr2_mask] += math.log(0.5)

    # Direct compute_ref_expression check (dense & sparse, plus <2 ref cell fallback)
    ref_idx = np.arange(n_ref, dtype=np.int64)
    mu_dn, sig_dn = compute_ref_expression(lr, ref_indices=ref_idx, expr_model="gaussian_logratio")
    mu_sp, sig_sp = compute_ref_expression(
        sparse.csr_matrix(lr), ref_indices=ref_idx, expr_model="gaussian_logratio"
    )
    np.testing.assert_allclose(mu_dn, mu_sp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sig_dn, sig_sp, rtol=1e-12, atol=1e-12)
    assert abs(float(np.mean(mu_dn[chr2_mask])) - (-0.35)) < 0.08
    assert 0.15 <= float(np.median(sig_dn)) <= 0.28

    _mu_1, sig_1 = compute_ref_expression(
        lr, ref_indices=np.array([0], dtype=np.int64), default_sigma=0.4, expr_model="gaussian_logratio"
    )
    np.testing.assert_allclose(sig_1, 0.4)

    ad_lr = ad.AnnData(X=lr, obs=obs_df, var=var_df)
    cfg = NumbatConfig(expr_model="gaussian_logratio", max_iter=2, min_cells=3, backend="numpy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_numbat(ad_lr, ref_key="grp", ref_label="ref", init_k=3, config=cfg)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0

    called = {s.chrom: s for s in res.segments}
    assert set(called.keys()) == {"chr1", "chr2"}
    assert called["chr1"].cnv_state == "amp"
    assert abs(called["chr1"].phi_mle - 1.5) < 0.2
    assert called["chr2"].cnv_state == "del"
    assert abs(called["chr2"].phi_mle - 0.5) < 0.15

    assert float(np.mean(res.clones.p_cnv[n_ref + n_norm :])) > 0.9
    assert float(np.mean(res.clones.p_cnv[n_ref : n_ref + n_norm])) < 0.1


def test_sigmoid_extreme_llr_no_overflow_warning() -> None:
    """Verify _sigmoid and score_single_cells emit zero overflow warnings for llr < -710."""
    extreme_vals = np.array([-1e6, -1000.0, -711.0, -710.0, -500.0, 0.0, 500.0, 710.0, 1e6])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sig = _sigmoid(extreme_vals)
        sig_scalar = _sigmoid(-800.0)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0
    assert np.all(np.isfinite(sig))
    assert sig[0] == 0.0 and sig[3] == 0.0
    assert sig[5] == 0.5
    assert sig[7] == 1.0 and sig[8] == 1.0
    assert isinstance(sig_scalar, float) and sig_scalar == 0.0

    # End-to-end score_single_cells check where llr_x << -710 for diploid cells
    n_genes = 40
    lr = np.full((6, n_genes), -3.0, dtype=np.float64)  # extreme mismatch against phi=2.0
    gi = pd.DataFrame({"gene": [f"G{i}" for i in range(n_genes)], "chrom": ["chr1"] * n_genes, "pos": [(i + 1) * 1000 for i in range(n_genes)]})
    expr = io_mod.extract_expression_input(
        lr,
        gene_info=gi,
        cell_ids=[f"c{i}" for i in range(6)],
        gene_ids=gi["gene"].tolist(),
        require_non_negative=False,
    )
    loci = io_mod.build_ordered_loci(expr, None)
    seg = SegmentCall(
        seg_id="chr1_1_bamp",
        chrom="chr1",
        start=1000,
        end=40000,
        n_genes=n_genes,
        n_snps=0,
        cnv_state="bamp",
        cnv_state_post="bamp",
        phi_mle=2.0,
        theta_mle=0.0,
        p_neu=0.0,
        p_amp=0.0,
        p_del=0.0,
        p_loh=0.0,
        p_bamp=1.0,
        p_bdel=0.0,
        entropy=0.0,
        log_bf=50.0,
        start_locus=0,
        end_locus=n_genes,
        phi_state=2.0,
        theta_state=0.0,
    )
    cfg = NumbatConfig(expr_model="gaussian_logratio")
    with warnings.catch_warnings(record=True) as caught_sc:
        warnings.simplefilter("always")
        _jp, prob_mat, llr_mat, _as = score_single_cells(
            segments=(seg,),
            loci=loci,
            expr=expr,
            alleles=None,
            ref_lambda=np.zeros(n_genes, dtype=np.float64),
            sigma_g=np.full(n_genes, 0.2, dtype=np.float64),
            lib_sizes=np.zeros(6, dtype=np.float64),
            diploid_offsets=np.zeros(6, dtype=np.float64),
            p_switch=np.zeros(n_genes, dtype=np.float64),
            config=cfg,
        )
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught_sc)
    assert np.all(llr_mat < -710.0)
    assert np.all(np.isfinite(prob_mat)) and np.all(prob_mat == 0.0)


def test_beta_binomial_rejects_nan_and_nonfinite() -> None:
    """Verify beta_binomial_logpmf raises ValueError on NaN or Inf inputs."""
    with pytest.raises(ValueError, match="finite"):
        beta_binomial_logpmf(np.nan, 0.0, 0.5, 20.0)
    with pytest.raises(ValueError, match="finite"):
        beta_binomial_logpmf(0.0, np.nan, 0.5, 20.0)
    with pytest.raises(ValueError, match="finite"):
        beta_binomial_logpmf(1.0, 5.0, np.nan, 20.0)
    with pytest.raises(ValueError, match="finite"):
        beta_binomial_logpmf(1.0, 5.0, 0.5, np.nan)
    with pytest.raises(ValueError, match="finite"):
        beta_binomial_logpmf(np.inf, 5.0, 0.5, 20.0)


def test_canonical_chromosome_ordering_and_mixed_naming() -> None:
    """Verify chr2 precedes chr10 and mixed 'chr2'/'2' naming does not split chromosome blocks."""
    gi = pd.DataFrame(
        {
            "gene": ["G10_a", "G2_a", "GX_a", "G1_a", "G2_b"],
            "chrom": ["chr10", "chr2", "chrX", "chr1", "chr2"],
            "pos": [100, 100, 100, 100, 300],
        }
    )
    allele_df = pd.DataFrame(
        {
            "cell": ["c0", "c0", "c0"],
            "chrom": ["10", "2", "X"],
            "pos": [200, 200, 200],
            "AD": [5, 5, 5],
            "DP": [10, 10, 10],
        }
    )
    expr = io_mod.extract_expression_input(
        np.ones((2, 5), dtype=np.float64),
        gene_info=gi,
        cell_ids=["c0", "c1"],
        gene_ids=gi["gene"].tolist(),
    )
    alleles = io_mod.parse_allele_dataframe(allele_df, expr.cell_ids)
    loci = io_mod.build_ordered_loci(expr, alleles)

    assert loci.chrom_names == ("chr1", "chr2", "chr10", "chrX")
    assert loci.chrom_names.index("chr2") < loci.chrom_names.index("chr10")
    assert len(loci.chrom_offsets) == 5

    # chr2 block (offset 1..4) must interleave G2_a (100), SNP 2:200, G2_b (300)
    s_c2, e_c2 = int(loci.chrom_offsets[1]), int(loci.chrom_offsets[2])
    assert list(loci.pos[s_c2:e_c2]) == [100, 200, 300]
    assert list(loci.chrom[s_c2:e_c2]) == ["chr2", "chr2", "chr2"]


def test_settings_validation_and_s15_state_catalog_parity() -> None:
    """
    Verify strict `settings` validation, 15-state `s15` catalog (`S15_STATES`),
    and Numba/NumPy HMM parity under `settings="R_default"`.
    """
    assert len(DEFAULT_STATES) == 11
    assert len(S15_STATES) == 15
    assert len(NumbatConfig(settings="default").states) == 11
    assert len(NumbatConfig(settings="R_default").states) == 15

    collapsed_counts = pd.Series([s.collapsed for s in S15_STATES]).value_counts().to_dict()
    assert collapsed_counts == {
        "del": 4,
        "loh": 4,
        "amp": 4,
        "neu": 1,
        "bamp": 1,
        "bdel": 1,
    }
    bdel_spec = [s for s in S15_STATES if s.collapsed == "bdel"][0]
    assert bdel_spec.phi < 1.0 and bdel_spec.mu == pytest.approx(0.50)

    with pytest.raises(ValueError, match="settings"):
        NumbatConfig(settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        run_joint_hmm(np.zeros((5, 11)), [0, 5], settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        fit_allele_hmm([5.0], [10.0], settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        fit_expression_hmm([10.0], [1.0], settings="bad")  # type: ignore[arg-type]

    rng = np.random.default_rng(1515)
    adata_tiny, allele_tiny, _, _ = _simulate_cohort(
        rng, n_ref=6, n_unlabeled_normal=4, clone_specs=[], n_genes_per_chrom=10, n_snps_per_chrom=10, chroms=("chr1",)
    )
    with pytest.raises(ValueError, match="settings"):
        run_numbat(adata_tiny, allele_df=allele_tiny, ref_cells="is_ref", settings="bad")  # type: ignore[arg-type]

    emit_15 = rng.normal(-3.0, 1.0, size=(80, 15))
    offsets = [0, 40, 80]
    ps = np.asarray(phase_switch_prob(rng.uniform(0.0, 5.0, size=80), nu=1.0))
    ps[0] = 0.0
    ps[40] = 0.0
    res_np = run_joint_hmm(emit_15, offsets, p_switch=ps, backend="numpy", settings="R_default")
    assert res_np.posteriors.shape == (80, 15)
    if has_numba():
        res_nb = run_joint_hmm(emit_15, offsets, p_switch=ps, backend="numba", settings="R_default")
        np.testing.assert_allclose(res_nb.posteriors, res_np.posteriors, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(res_nb.viterbi_states, res_np.viterbi_states)


def test_r_default_s15_continuous_mle_and_retest_bulks_end_to_end() -> None:
    """
    Verify `run_numbat(..., settings="R_default")` recovers clonal LOH, homozygous
    deletion (`bdel`), high-level amplification with continuous `phi_mle` refinement,
    subclonal deletion, and zero false positives on pure diploid data.
    """
    rng = np.random.default_rng(2026)
    adata, allele_df, true_comp, _ = _simulate_cohort(
        rng,
        n_ref=35,
        n_unlabeled_normal=25,
        clone_specs=[
            (
                30,
                {
                    "chr3": ("loh", 1.00, 0.85),   # clonal CN-neutral LOH across both clones
                    "chr7": ("amp", 3.10, 0.78),   # high-level amp (> discrete 2.25)
                    "chr10": ("bdel", 0.18, 0.50), # homozygous deletion 00
                },
            ),
            (
                30,
                {
                    "chr3": ("loh", 1.00, 0.85),   # shared clonal LOH
                    "chr7": ("amp", 3.10, 0.78),
                    "chr10": ("bdel", 0.18, 0.50),
                    "chr1": ("del", 0.50, 0.90),   # subclonal mono-allelic deletion
                },
            ),
        ],
        n_genes_per_chrom=35,
        n_snps_per_chrom=30,
        chroms=("chr1", "chr3", "chr7", "chr10"),
        mean_snp_dp=10.0,
    )

    res_r = run_numbat(adata, allele_df=allele_df, ref_cells="is_ref", init_k=3, settings="R_default")
    assert res_r.config.settings == "R_default"
    assert len(res_r.config.states) == 15
    assert callable(retest_bulks)

    by_chrom = {s.chrom: s for s in res_r.segments}
    assert by_chrom["chr3"].cnv_state == "loh"
    assert by_chrom["chr10"].cnv_state == "bdel"
    assert by_chrom["chr10"].phi_mle < 0.40
    assert by_chrom["chr7"].cnv_state == "amp"
    # Continuous MLE refinement captures >2.25 fold change on chr7
    assert by_chrom["chr7"].phi_state > 2.25
    assert by_chrom["chr1"].cnv_state == "del"

    unlabeled = ~adata.obs["is_ref"].to_numpy(dtype=bool)
    auc_r = roc_auc_score((true_comp[unlabeled] == "tumor").astype(int), res_r.clones.p_cnv[unlabeled])
    assert auc_r >= 0.98

    # Null calibration under settings="R_default"
    adata_null, allele_null, _, _ = _simulate_cohort(
        np.random.default_rng(2027),
        n_ref=25,
        n_unlabeled_normal=35,
        clone_specs=[],
        n_genes_per_chrom=30,
        n_snps_per_chrom=25,
        chroms=("chr1", "chr2"),
    )
    res_null_r = run_numbat(adata_null, allele_df=allele_null, ref_cells="is_ref", settings="R_default")
    assert len(res_null_r.segments) == 0


