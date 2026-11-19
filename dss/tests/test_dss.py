"""Unit tests for the standalone `dss` package (DSS + DMRcate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from dss import (
    BSData,
    DSSConfig,
    MultifactorFit,
    call_dml,
    call_dmr,
    dml_fit_multifactor,
    dml_test,
    dml_test_multifactor,
    dmrcate,
    make_bsseq_data,
    shrink_dispersion_bb,
    smooth_counts_tricube,
)


def _simulate_bsseq_two_group(
    n_cpgs: int = 240,
    n_reps: int = 3,
    phi_true: float = 0.04,
    seed: int = 42,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate Beta-Binomial bisulfite counts with 2 true DMRs (1 hyper, 1 hypo)."""
    rng = np.random.default_rng(seed)
    # Dense 35 bp spacing so CpG islands fall inside smoothing & DMR windows
    pos = np.arange(1, n_cpgs + 1, dtype=np.int64) * 35 + 10_000
    chrom = np.where(np.arange(n_cpgs) < (n_cpgs // 2), "chr1", "chr2")
    coords = pd.DataFrame({"chr": chrom, "pos": pos})

    mu1 = np.full(n_cpgs, 0.25, dtype=np.float64)
    mu2 = np.full(n_cpgs, 0.25, dtype=np.float64)
    # Hypermethylated DMR in group1 vs group2 on chr1: indices 25..36 (12 CpGs, 385 bp)
    mu1[25:37] = 0.80
    mu2[25:37] = 0.20
    # Hypomethylated DMR in group1 vs group2 on chr2: indices 150..160 (11 CpGs, 350 bp)
    mu1[150:161] = 0.15
    mu2[150:161] = 0.75

    m_conc = (1.0 / phi_true) - 1.0

    def _draw_group(mu_vec: np.ndarray, reps: int) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
        n_mat = rng.integers(25, 55, size=(n_cpgs, reps))
        x_mat = np.zeros_like(n_mat)
        for r in range(reps):
            p_r = rng.beta(mu_vec * m_conc, (1.0 - mu_vec) * m_conc)
            x_mat[:, r] = rng.binomial(n_mat[:, r], p_r)
        frames = [
            pd.DataFrame(
                {"chr": chrom, "pos": pos, "N": n_mat[:, r], "X": x_mat[:, r]}
            )
            for r in range(reps)
        ]
        return frames, n_mat.astype(np.float64), x_mat.astype(np.float64)

    g1_frames, n1, x1 = _draw_group(mu1, n_reps)
    g2_frames, n2, x2 = _draw_group(mu2, n_reps)
    n_all = np.column_stack([n1, n2])
    x_all = np.column_stack([x1, x2])
    return g1_frames, g2_frames, coords, n_all, x_all


def test_invalid_settings_raises() -> None:
    g1, g2, coords, n_all, x_all = _simulate_bsseq_two_group(n_cpgs=40)
    with pytest.raises(ValueError, match="Unknown settings"):
        DSSConfig(settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        dml_test(g1, g2, settings="bad")  # type: ignore[arg-type]

    dml_df = dml_test(g1, g2, settings="default")
    with pytest.raises(ValueError, match="Unknown settings"):
        call_dml(dml_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        call_dmr(dml_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        dmrcate(dml_df, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        dml_fit_multifactor(n_all, x_all, coords, design=[0, 0, 0, 1, 1, 1], settings="bad")  # type: ignore[arg-type]

    mf = dml_fit_multifactor(n_all, x_all, coords, design=[0, 0, 0, 1, 1, 1], settings="default")
    with pytest.raises(ValueError, match="Unknown settings"):
        dml_test_multifactor(mf, coef=1, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        shrink_dispersion_bb([(x_all[:, :3], n_all[:, :3], np.full(40, 0.3))], settings="bad")  # type: ignore[arg-type]


def test_dss_config_default_vs_r_default() -> None:
    cfg_def = DSSConfig(settings="default")
    assert cfg_def.smoothing is True
    assert cfg_def.p_threshold_dmr == 1e-4
    assert cfg_def.dis_merge == 100

    cfg_r = DSSConfig(settings="R_default")
    assert cfg_r.smoothing is False
    assert cfg_r.p_threshold_dmr == 1e-5
    # R_default clamps dis_merge (100) down to minlen (50) matching DSS::callDMR
    assert cfg_r.dis_merge == 50


def test_dispersion_shrinkage_pulls_toward_prior_and_clamps() -> None:
    rng = np.random.default_rng(17)
    n_cpgs = 300
    n_reps = 4
    mu = np.full(n_cpgs, 0.5)
    n_mat = np.full((n_cpgs, n_reps), 40.0)

    def _sim(phi_val: float) -> np.ndarray:
        m = (1.0 / phi_val) - 1.0
        p = rng.beta(mu[:, None] * m, (1.0 - mu[:, None]) * m, size=(n_cpgs, n_reps))
        return rng.binomial(n_mat.astype(int), p).astype(np.float64)

    x_low = _sim(0.02)
    x_high = _sim(0.25)

    phi_post_low, phi_mle_low, _ = shrink_dispersion_bb([(x_low, n_mat, mu)], settings="default")
    phi_post_high, _, _ = shrink_dispersion_bb([(x_high, n_mat, mu)], settings="default")

    assert np.all((phi_post_low >= 1e-5) & (phi_post_low <= 0.99))
    assert np.all((phi_post_high >= 1e-5) & (phi_post_high <= 0.99))
    # Empirical Bayes shrinkage reduces dispersion spread relative to raw profile MLE
    assert np.std(np.log(phi_post_low)) < np.std(np.log(phi_mle_low))
    assert np.median(phi_post_high) > np.median(phi_post_low) + 0.10


def test_tricube_smoothing_preserves_chromosome_boundaries() -> None:
    chrom = np.array(["chr1", "chr1", "chr1", "chr2", "chr2"])
    pos = np.array([100, 150, 200, 160, 210], dtype=np.float64)
    n_mat = np.full((5, 1), 50.0)
    x_mat = np.array([[45.0], [45.0], [45.0], [5.0], [5.0]])

    p_sm = smooth_counts_tricube(chrom, pos, x_mat, n_mat, smoothing_span=500.0)
    # chr1 stays at 0.90 and chr2 stays at 0.10 despite overlapping numeric coordinates!
    np.testing.assert_allclose(p_sm[:3, 0], 0.90, atol=1e-12)
    np.testing.assert_allclose(p_sm[3:, 0], 0.10, atol=1e-12)


def test_dml_test_formats_and_equal_disp() -> None:
    g1, g2, coords, n_all, x_all = _simulate_bsseq_two_group(n_cpgs=120, seed=9)
    # Drop 5 CpGs from replicate 0 of g1 to test union alignment
    g1_partial = [g1[0].iloc[5:].copy(), g1[1], g1[2]]
    res_df = dml_test(g1_partial, g2, equal_disp=False, settings="default")
    expected_cols = [
        "chr",
        "pos",
        "mu1",
        "mu2",
        "diff",
        "diff_se",
        "stat",
        "phi1",
        "phi2",
        "pval",
        "fdr",
    ]
    assert list(res_df.columns) == expected_cols
    assert len(res_df) == 120

    # Matrix-tuple input with equal_disp=True
    res_tuple = dml_test(
        (n_all[:, :3], x_all[:, :3], coords),
        (n_all[:, 3:], x_all[:, 3:], coords),
        equal_disp=True,
        settings="R_default",
    )
    np.testing.assert_allclose(res_tuple["phi1"].to_numpy(), res_tuple["phi2"].to_numpy())
    assert res_tuple.loc[30, "fdr"] < 1e-6

    # BSData input
    bs1 = make_bsseq_data(g1)
    bs2 = make_bsseq_data(g2)
    assert isinstance(bs1, BSData)
    res_bs = dml_test(bs1, bs2, settings="default")
    assert res_bs.loc[30, "fdr"] < 1e-6


def test_single_replicate_requires_smoothing() -> None:
    g1, g2, _, _, _ = _simulate_bsseq_two_group(n_cpgs=80, n_reps=1, seed=4)
    with pytest.raises(ValueError, match="smoothing=True"):
        dml_test(g1, g2, smoothing=False, settings="default")

    res_sm = dml_test(g1, g2, smoothing=True, settings="default")
    assert res_sm.loc[30, "pval"] < 1e-4


def test_call_dml_exact_composite_tail_probability() -> None:
    g1, g2, _, _, _ = _simulate_bsseq_two_group(n_cpgs=180, seed=22)
    dml_res = dml_test(g1, g2, settings="R_default")

    delta = 0.15
    called = call_dml(dml_res, delta=delta, p_threshold=1e-3, settings="R_default")
    assert len(called) >= 15
    assert "postprob.overThreshold" in called.columns

    # Verify exact R DSS::callDML composite tail formula row-by-row
    d = np.abs(called["diff"].to_numpy())
    s = called["diff_se"].to_numpy()
    expected_p = norm.cdf(-(d - delta) / s) + norm.cdf(-(d + delta) / s)
    np.testing.assert_allclose(called["pval"].to_numpy(), expected_p, rtol=1e-12)

    # Larger delta produces fewer or equal DML calls than delta=0
    called_zero = call_dml(dml_res, delta=0.0, p_threshold=1e-3, settings="R_default")
    assert len(called) <= len(called_zero)


def test_call_dmr_recovery_and_dis_merge_r_default_rule() -> None:
    g1, g2, _, _, _ = _simulate_bsseq_two_group(n_cpgs=240, seed=42)
    dml_res = dml_test(g1, g2, settings="default")
    dmrs = call_dmr(dml_res, settings="default")

    assert len(dmrs) == 2
    expected_dmr_cols = {
        "chr",
        "start",
        "end",
        "length",
        "nCG",
        "meanMethy1",
        "meanMethy2",
        "diff.Methy",
        "areaStat",
    }
    assert expected_dmr_cols.issubset(dmrs.columns)
    # Verify one positive areaStat (hyper) on chr1 and one negative areaStat (hypo) on chr2
    chr1_dmr = dmrs.loc[dmrs["chr"] == "chr1"].iloc[0]
    chr2_dmr = dmrs.loc[dmrs["chr"] == "chr2"].iloc[0]
    assert chr1_dmr["areaStat"] > 0 and chr1_dmr["diff.Methy"] > 0.35
    assert chr2_dmr["areaStat"] < 0 and chr2_dmr["diff.Methy"] < -0.35

    # Test dis_merge=100 vs R_default dis_merge -> minlen=50 clamping on 75 bp gap!
    # Bump 1: pos [100, 130, 160] (length 60 >= 50), gap 75 bp, Bump 2: pos [235, 265, 295] (length 60)
    synth_dml = pd.DataFrame(
        {
            "chr": ["chr1"] * 6,
            "pos": [100, 130, 160, 235, 265, 295],
            "mu1": [0.8] * 6,
            "mu2": [0.2] * 6,
            "diff": [0.6] * 6,
            "diff_se": [0.05] * 6,
            "stat": [12.0] * 6,
            "pval": [1e-30] * 6,
            "fdr": [1e-29] * 6,
        }
    )
    dmr_def = call_dmr(synth_dml, minlen=50, dis_merge=100, settings="default")
    dmr_r = call_dmr(synth_dml, minlen=50, dis_merge=100, settings="R_default")
    # Default merges across the 75 bp gap (<= 100) into 1 DMR; R_default clamps dis_merge=50 -> 2 DMRs!
    assert len(dmr_def) == 1
    assert len(dmr_r) == 2


def test_call_dmr_does_not_merge_opposite_sign_bumps() -> None:
    synth_dml = pd.DataFrame(
        {
            "chr": ["chr1"] * 6,
            "pos": [100, 125, 150, 175, 200, 225],
            "mu1": [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
            "mu2": [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            "diff": [0.6, 0.6, 0.6, -0.6, -0.6, -0.6],
            "diff_se": [0.05] * 6,
            "stat": [12.0, 12.0, 12.0, -12.0, -12.0, -12.0],
            "pval": [1e-30] * 6,
            "fdr": [1e-29] * 6,
        }
    )
    dmrs = call_dmr(synth_dml, minlen=50, minCG=3, dis_merge=100, settings="default")
    assert len(dmrs) == 2
    assert set(np.sign(dmrs["areaStat"])) == {-1.0, 1.0}


def test_multifactor_paired_and_continuous_designs() -> None:
    """Test Park & Wu (2016) DSS-general on paired patient + continuous covariate design."""
    rng = np.random.default_rng(123)
    n_cpgs = 120
    # 4 patients x 2 conditions (normal vs tumor) = 8 samples
    patients = np.array(["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"])
    tumor = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    age_z = np.array([-1.0, -1.0, -0.3, -0.3, 0.4, 0.4, 1.1, 1.1])
    design_df = pd.DataFrame({"patient": patients, "tumor": tumor})

    coords = pd.DataFrame(
        {"chr": ["chr1"] * n_cpgs, "pos": np.arange(n_cpgs, dtype=np.int64) * 50 + 1000}
    )
    n_mat = rng.integers(30, 60, size=(n_cpgs, 8)).astype(np.float64)

    # Strong patient baseline heterogeneity that would confound an unpaired test!
    pat_offset = np.array([-0.6, -0.6, -0.2, -0.2, 0.25, 0.25, 0.65, 0.65])
    tumor_eff = np.zeros(n_cpgs)
    tumor_eff[:10] = 0.85  # CpGs 0..9 hypermethylated in tumor

    y_true = np.clip(pat_offset[None, :] + tumor_eff[:, None] * tumor[None, :], -1.3, 1.3)
    p_true = 0.5 * (np.sin(y_true) + 1.0)
    x_mat = rng.binomial(n_mat.astype(int), p_true).astype(np.float64)

    fit = dml_fit_multifactor(n_mat, x_mat, coords, design=design_df, settings="default")
    assert isinstance(fit, MultifactorFit)
    assert "MultifactorFit" in fit.summary()
    assert "phi" in fit.to_dataframe().columns

    res_tumor = dml_test_multifactor(fit, coef="tumor", settings="default")
    assert (res_tumor.loc[:9, "fdr"] < 0.01).all()
    assert (res_tumor.loc[15:, "fdr"] < 0.01).sum() <= 1

    # Continuous covariate design
    design_cont = pd.DataFrame({"age": age_z, "tumor": tumor})
    fit_cont = dml_fit_multifactor(n_mat, x_mat, coords, design=design_cont, settings="R_default")
    res_cont = dml_test_multifactor(fit_cont, coef="tumor", settings="R_default")
    assert (res_cont.loc[:9, "fdr"] < 0.01).all()


def test_multifactor_multidf_matrix_contrast() -> None:
    rng = np.random.default_rng(77)
    n_cpgs = 80
    # 3 groups x 3 replicates = 9 samples
    groups = pd.DataFrame({"grp": ["A"] * 3 + ["B"] * 3 + ["C"] * 3})
    coords = pd.DataFrame({"chr": ["chr1"] * n_cpgs, "pos": np.arange(n_cpgs) * 40 + 500})
    n_mat = np.full((n_cpgs, 9), 45.0)
    p_mat = np.full((n_cpgs, 9), 0.30)
    p_mat[:6, 6:] = 0.85  # Group C elevated in CpGs 0..5
    x_mat = rng.binomial(n_mat.astype(int), p_mat).astype(np.float64)

    fit = dml_fit_multifactor(n_mat, x_mat, coords, design=groups, settings="default")
    # 2-df omnibus contrast testing grp_B == 0 and grp_C == 0 simultaneously
    l_mat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    res_omni = dml_test_multifactor(fit, contrast=l_mat, settings="default")
    assert (res_omni["df"] == 2).all()
    assert (res_omni.loc[:5, "fdr"] < 1e-4).all()


def test_dmrcate_satterthwaite_moments_and_dmr_calling() -> None:
    g1, g2, _, _, _ = _simulate_bsseq_two_group(n_cpgs=240, seed=42)
    dml_res = dml_test(g1, g2, smoothing=False, settings="R_default")

    dmrs = dmrcate(dml_res, lambda_bp=1000.0, C=2.0, fdr=0.05, min_cpgs=3, settings="default")
    assert len(dmrs) == 2
    cpg_stats = dmrs.attrs["cpg_stats"]
    assert {"K1", "K2", "K3", "df_satterthwaite", "smoothed_pval", "smoothed_fdr"}.issubset(
        cpg_stats.columns
    )
    # Check exact Satterthwaite df identity df == K2^2 / K3
    np.testing.assert_allclose(
        cpg_stats["df_satterthwaite"].to_numpy(),
        (cpg_stats["K2"].to_numpy() ** 2) / cpg_stats["K3"].to_numpy(),
        rtol=1e-12,
    )

    # Pure standard-normal null produces 0 false DMRs
    rng = np.random.default_rng(999)
    null_dml = pd.DataFrame(
        {
            "chr": ["chr1"] * 300,
            "pos": np.arange(300) * 40 + 1000,
            "stat": rng.standard_normal(300),
            "diff": rng.normal(0.0, 0.03, size=300),
        }
    )
    null_dmrs = dmrcate(null_dml, lambda_bp=1000.0, C=2.0, fdr=0.05, settings="default")
    assert len(null_dmrs) == 0
