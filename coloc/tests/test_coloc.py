from __future__ import annotations
import numpy as np
from coloc import coloc_abf

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
        atol=1e-9,
    )
    b2_distinct = np.zeros(m)
    b2_distinct[30] = 0.45
    res_distinct = coloc_abf(b1, se1, b2_distinct, se2, snp_ids=snps)
    assert res_distinct.pp_h3 > 0.85
    assert res_distinct.pp_h4 < 0.05
