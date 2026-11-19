from __future__ import annotations

import numpy as np
import pytest
from coloc import coloc_abf
from coloc.core import _log_sum_offdiag


def test_coloc_abf_shared_vs_distinct_causal_variant():
    m = 50
    se1 = np.full(m, 0.05)
    se2 = np.full(m, 0.05)
    b1 = np.zeros(m)
    b2 = np.zeros(m)
    b1[12] = 0.45
    b2[12] = -0.40
    snps = [f"rs{i}" for i in range(m)]
    res_shared = coloc_abf(b1, se1, b2, se2, snp_ids=snps, locus_name="chr1_test")
    assert res_shared.pp_h4 > 0.90
    assert res_shared.lead_snp == "rs12"
    assert res_shared.credible_set(0.95) == ["rs12"]
    np.testing.assert_allclose(
        res_shared.pp_h0 + res_shared.pp_h1 + res_shared.pp_h2 + res_shared.pp_h3 + res_shared.pp_h4,
        1.0,
        atol=1e-12,
    )
    b2_distinct = np.zeros(m)
    b2_distinct[30] = 0.45
    res_distinct = coloc_abf(b1, se1, b2_distinct, se2, snp_ids=snps)
    assert res_distinct.pp_h3 > 0.85
    assert res_distinct.pp_h4 < 0.05


def test_coloc_single_snp_m_equals_1():
    res = coloc_abf([0.4], [0.05], [0.35], [0.05], snp_ids=["rs_only"])
    assert res.pp_h3 == 0.0
    assert np.isfinite(res.pp_h4) and res.pp_h4 > 0.90
    np.testing.assert_allclose(
        res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4,
        1.0,
        atol=1e-12,
    )
    assert res.credible_set(0.95) == ["rs_only"]


def test_coloc_extreme_z80_no_cancellation_or_overflow():
    m = 40
    se1 = np.full(m, 0.05)
    se2 = np.full(m, 0.05)
    b1 = np.zeros(m)
    b2 = np.zeros(m)
    # |Z| = 80 shared causal variant
    b1[5] = 80.0 * 0.05
    b2[5] = -80.0 * 0.05
    res = coloc_abf(b1, se1, b2, se2)
    assert np.all(np.isfinite([res.pp_h0, res.pp_h1, res.pp_h2, res.pp_h3, res.pp_h4]))
    assert res.pp_h4 > 0.9999

    # Verify _log_sum_offdiag matches exact analytical log off-diagonal sum to machine precision
    lbf1 = res.lbf_trait1
    lbf2 = res.lbf_trait2
    # Off-diagonal pairs: (5, j!=5) gives (m-1)*exp(lbf1[5] + lbf2[0]),
    # (i!=5, 5) gives (m-1)*exp(lbf2[5] + lbf1[0]); remaining (m-1)*(m-2) terms are negligible (~e^-2880)
    expected_log_offdiag = lbf1[5] + lbf2[0] + np.log(2.0 * (m - 1))
    actual_log_offdiag = _log_sum_offdiag(lbf1, lbf2)
    np.testing.assert_allclose(actual_log_offdiag, expected_log_offdiag, rtol=1e-14, atol=1e-12)

    # Exact ratio PP.H3 / PP.H4 = (p1 * p2 / p12) * exp(actual_log_offdiag - (lbf1[5] + lbf2[5]))
    # Here |Z|=80 makes lbf1[5] ~ 2880 so exp(-2880) underflows to 0 cleanly; test |Z|=18 where PP.H3 ~ 1e-65 > 0
    b1_z18 = np.zeros(m)
    b2_z18 = np.zeros(m)
    b1_z18[5] = 18.0 * 0.05
    b2_z18[5] = 18.0 * 0.05
    res18 = coloc_abf(b1_z18, se1, b2_z18, se2, p1=1e-4, p2=1e-4, p12=1e-5)
    expected_ratio = (1e-4 * 1e-4 / 1e-5) * np.exp(
        res18.lbf_trait2[0] - res18.lbf_trait2[5] + np.log(2.0 * (m - 1))
    )
    np.testing.assert_allclose(res18.pp_h3 / res18.pp_h4, expected_ratio, rtol=1e-12)

    # |Z| = 80 distinct causal variants (must not overflow exp(log_prod - sum_shared))
    b2_dist = np.zeros(m)
    b2_dist[17] = 80.0 * 0.05
    res_dist = coloc_abf(b1, se1, b2_dist, se2)
    assert np.all(np.isfinite([res_dist.pp_h0, res_dist.pp_h1, res_dist.pp_h2, res_dist.pp_h3, res_dist.pp_h4]))
    np.testing.assert_allclose(res_dist.pp_h3, 1.0, atol=1e-12)


def test_coloc_sdy_scaling_and_table_credible_set_and_validation():
    rng = np.random.default_rng(99)
    m = 25
    se1 = rng.uniform(0.03, 0.07, m)
    se2 = rng.uniform(0.04, 0.08, m)
    b1 = rng.normal(0.0, se1)
    b2 = rng.normal(0.0, se2)
    se1[3] = se1[4] = 0.04
    se2[3] = se2[4] = 0.04
    b1[3] = b1[4] = 0.25
    b2[3] = b2[4] = 0.28

    r_unit = coloc_abf(b1, se1, b2, se2)
    r_scaled = coloc_abf(b1 * 4.2, se1 * 4.2, b2 * 0.35, se2 * 0.35, sd_y1=4.2, sd_y2=0.35)
    np.testing.assert_allclose(r_scaled.pp_h4, r_unit.pp_h4, rtol=1e-13)
    np.testing.assert_allclose(r_scaled.pp_h3, r_unit.pp_h3, rtol=1e-13)
    np.testing.assert_allclose(r_scaled.snp_pp_h4, r_unit.snp_pp_h4, rtol=1e-13)

    tbl = r_unit.snp_table()
    assert list(tbl.columns) == ["snp", "lABF_trait1", "lABF_trait2", "lABF_shared", "SNP.PP.H4"]
    assert len(tbl) == m
    assert np.all(np.diff(tbl["SNP.PP.H4"].to_numpy()) <= 0.0)
    np.testing.assert_allclose(tbl["SNP.PP.H4"].sum(), 1.0, atol=1e-12)

    cs = r_unit.credible_set(0.90)
    assert set(cs).issuperset({"snp_3", "snp_4"})
    summary = r_unit.summary_dict()
    assert summary["n_snps"] == m and summary["lead_snp"] in {"snp_3", "snp_4"}

    with pytest.raises(ValueError):
        coloc_abf(b1, se1, b2, se2, p1=0.0)
    with pytest.raises(ValueError):
        coloc_abf(b1, se1, b2, se2, p1=0.5, p2=0.4, p12=0.2)
    with pytest.raises(ValueError):
        coloc_abf(b1, se1, b2, se2, prior_w1=-0.1)
    with pytest.raises(ValueError):
        coloc_abf(b1, se1, b2, se2, sd_y1=0.0)
    with pytest.raises(ValueError):
        r_unit.credible_set(0.0)


def test_r_default_settings():
    m = 60
    se1 = np.full(m, 0.06)
    se2 = np.full(m, 0.06)
    b1 = np.zeros(m)
    b2 = np.zeros(m)
    b1[10] = 0.36
    b2[10] = 0.34

    # 1. Invalid settings and dataset_type validation
    with pytest.raises(ValueError, match="Unknown settings 'bad'"):
        coloc_abf(b1, se1, b2, se2, settings="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown dataset_type1"):
        coloc_abf(b1, se1, b2, se2, dataset_type1="binary")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown dataset_type2"):
        coloc_abf(b1, se1, b2, se2, dataset_type2="continuous")  # type: ignore[arg-type]

    # 2. Distinct dataset_type1 / dataset_type2 auto-set Wakefield prior_w (0.15 quant vs 0.20 cc)
    res_qc = coloc_abf(b1, se1, b2, se2, dataset_type1="quant", dataset_type2="cc")
    res_qc_explicit = coloc_abf(b1, se1, b2, se2, prior_w1=0.15, prior_w2=0.20)
    assert res_qc.prior_w1 == pytest.approx(0.15)
    assert res_qc.prior_w2 == pytest.approx(0.20)
    np.testing.assert_allclose(res_qc.lbf_trait1, res_qc_explicit.lbf_trait1, rtol=1e-14)
    np.testing.assert_allclose(res_qc.lbf_trait2, res_qc_explicit.lbf_trait2, rtol=1e-14)
    assert res_qc.pp_h4 == pytest.approx(res_qc_explicit.pp_h4, rel=1e-14)

    res_cq = coloc_abf(b1, se1, b2, se2, dataset_type1="cc", dataset_type2="quant")
    assert res_cq.prior_w1 == pytest.approx(0.20)
    assert res_cq.prior_w2 == pytest.approx(0.15)
    assert not np.isclose(res_cq.lbf_trait1[10], res_qc.lbf_trait1[10])

    # Explicit per-trait or shared prior_w overrides dataset_type defaults
    res_override = coloc_abf(b1, se1, b2, se2, prior_w1=0.12, dataset_type1="cc", dataset_type2="cc")
    assert res_override.prior_w1 == pytest.approx(0.12)
    assert res_override.prior_w2 == pytest.approx(0.20)
    res_shared_w = coloc_abf(b1, se1, b2, se2, prior_w=0.18, dataset_type1="cc", dataset_type2="quant")
    assert res_shared_w.prior_w1 == pytest.approx(0.18)
    assert res_shared_w.prior_w2 == pytest.approx(0.18)

    # 3. Upstream claudia.R:173-182 adjust_prior shrinkage + UserWarning when nsnps * p >= 1
    m_large = 12_000
    b1_lg = np.zeros(m_large)
    b2_lg = np.zeros(m_large)
    se_lg = np.full(m_large, 0.05)
    b1_lg[500] = 0.45
    b2_lg[500] = 0.42

    with pytest.warns(UserWarning, match=r"nsnps >= 1, setting p"):
        res_lg = coloc_abf(b1_lg, se_lg, b2_lg, se_lg, settings="R_default")

    expected_p = 1.0 / float(m_large + 1)
    assert res_lg.p1 == pytest.approx(expected_p, rel=1e-14)
    assert res_lg.p2 == pytest.approx(expected_p, rel=1e-14)
    assert res_lg.p12 == pytest.approx(1e-5, rel=1e-14)

    res_lg_manual = coloc_abf(
        b1_lg, se_lg, b2_lg, se_lg, p1=expected_p, p2=expected_p, p12=1e-5, settings="default"
    )
    assert res_lg.pp_h4 == pytest.approx(res_lg_manual.pp_h4, rel=1e-13)
    assert res_lg.pp_h3 == pytest.approx(res_lg_manual.pp_h3, rel=1e-13)

    # High explicit priors that raise ValueError in "default" are shrunk with warning in "R_default"
    with pytest.raises(ValueError):
        coloc_abf(b1, se1, b2, se2, p1=0.5, p2=0.4, p12=0.2, settings="default")
    with pytest.warns(UserWarning, match=r"nsnps >= 1, setting p"):
        res_shrunk = coloc_abf(b1, se1, b2, se2, p1=0.5, p2=0.4, p12=0.2, settings="R_default")
    assert res_shrunk.p1 == pytest.approx(1.0 / (m + 1))
    assert res_shrunk.p2 == pytest.approx(1.0 / (m + 1))
    assert res_shrunk.p12 == pytest.approx(1.0 / (m + 1))
    assert np.isclose(
        res_shrunk.pp_h0 + res_shrunk.pp_h1 + res_shrunk.pp_h2 + res_shrunk.pp_h3 + res_shrunk.pp_h4,
        1.0,
        atol=1e-12,
    )
