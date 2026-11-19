"""Frozen upstream R / C++ reference fixture benchmark tests for settings='R_default'."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("all", repo_root=REPO_ROOT)

FIXTURE_PATH = Path(__file__).resolve().parent / "golden_r_fixtures.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class _DummyRead:
    def __init__(self, nm: int, cigartuples: list[tuple[int, int]], is_paired: bool = True):
        self._nm = nm
        self.cigartuples = cigartuples
        self.is_unmapped = False
        self.is_paired = is_paired
        self.query_length = 100

    def has_tag(self, tag: str) -> bool:
        return tag == "NM"

    def get_tag(self, tag: str) -> int:
        return self._nm


def test_golden_xenofilterr_private_r_cigar_and_penalty():
    from xenofilterR import count_mismatches, score_pair

    fx = FIXTURES["xenofilterR_private_R"]
    read = _DummyRead(
        nm=fx["nm"],
        cigartuples=[(4, fx["soft_clip"]), (0, 90), (1, fx["insertion"]), (5, fx["hard_clip"])],
    )
    assert count_mismatches(read, settings="default") == fx["expected_default_edit_distance"]
    assert count_mismatches(read, settings="R_default") == fx["expected_R_default_edit_distance"]
    assert score_pair(read, None, settings="R_default") == pytest.approx(
        fx["expected_R_default_single_mapped_pair_score"], abs=1e-12
    )


def test_golden_coloc_abf_case_control_prior_and_pph4():
    from coloc import coloc_abf

    fx = FIXTURES["coloc_abf_wakefield_cc"]
    res = coloc_abf(
        fx["beta1"],
        fx["se1"],
        fx["beta2"],
        fx["se2"],
        snp_ids=["rs1", "rs2", "rs3"],
        dataset_type1=fx["dataset_type1"],
        dataset_type2=fx["dataset_type2"],
        settings="R_default",
    )
    assert res.prior_w1 == pytest.approx(fx["expected_R_default_prior_w"], abs=1e-12)
    assert res.lead_snp == fx["expected_lead_snp"]
    assert res.pp_h4 == pytest.approx(fx["expected_pp_h4"], abs=1e-9)


def test_golden_mutationtimer_gerstung_rate_equations():
    from mutationtimer import pi_to_time

    fx = FIXTURES["mutationtimer_pi_to_time"]
    t_21 = pi_to_time(2, 1, fx["cn_2_1_mid"]["pi"], settings="R_default")
    t_20 = pi_to_time(2, 0, fx["cn_2_0"]["pi"], settings="R_default")
    t_22 = pi_to_time(2, 2, fx["cn_2_2_wgd"]["pi"], settings="R_default")
    assert t_21 == pytest.approx(fx["cn_2_1_mid"]["expected_T"], abs=1e-9)
    assert t_20 == pytest.approx(fx["cn_2_0"]["expected_T"], abs=1e-9)
    assert t_22 == pytest.approx(fx["cn_2_2_wgd"]["expected_T"], abs=1e-9)


def test_golden_smr_single_exact_wald_ratio():
    from smr_heidi import smr_single

    fx = FIXTURES["smr_single_exact"]
    res = smr_single(fx["b_zy"], fx["se_zy"], fx["b_zx"], fx["se_zx"], settings="R_default")
    assert res.b_xy == pytest.approx(fx["expected_b_xy"], abs=1e-12)
    assert res.t_smr == pytest.approx(fx["expected_t_smr_chi2"], abs=1e-9)


def test_golden_survrm2_exact_kaplan_meier_integrals():
    from survrm2 import rmst2

    fx = FIXTURES["survrm2_km_exact"]
    time = np.array(fx["time_arm0"] + fx["time_arm1"])
    status = np.array(fx["status_arm0"] + fx["status_arm1"])
    arm = np.array([0] * 5 + [1] * 5)
    res = rmst2(time, status, arm, tau=fx["tau"], settings="R_default")
    assert res.arm0.rmst == pytest.approx(fx["expected_rmst0"], abs=1e-9)
    assert res.arm1.rmst == pytest.approx(fx["expected_rmst1"], abs=1e-9)
    assert res.rmst_diff_10.est == pytest.approx(fx["expected_rmst_diff"], abs=1e-9)
    assert res.rmst_ratio_10.est == pytest.approx(fx["expected_rmst_ratio"], abs=1e-9)


def test_golden_buysetest_peron_exact_pairwise_win_ratio():
    from buysetest import Endpoint, buyse_test

    fx = FIXTURES["buysetest_peron_uncensored"]
    df = pd.DataFrame(
        {
            "arm": [0, 0, 0, 1, 1, 1],
            "tte": fx["arm0_tte"] + fx["arm1_tte"],
            "ev": [1, 1, 1, 1, 1, 1],
        }
    )
    res_def = buyse_test(df, "arm", [Endpoint("tte", "tte", "ev")], settings="default")
    res_r = buyse_test(df, "arm", [Endpoint("tte", "tte", "ev")], settings="R_default")
    tbl = res_r.to_dataframe()
    assert float(tbl["win_ratio"].iloc[-1]) == pytest.approx(fx["expected_win_ratio"], abs=1e-8)
    assert float(tbl["net_benefit"].iloc[-1]) == pytest.approx(fx["expected_net_benefit"], abs=1e-8)
    assert res_r.settings == "R_default" and res_def.settings == "default"


def test_golden_fgwas_wakefield_lnabf_formula():
    from fgwas import wakefield_lnabf

    fx = FIXTURES["fgwas_wakefield_lnabf"]
    val = float(
        wakefield_lnabf([fx["beta"]], [fx["se"]], V=fx["V_fine_R_default"], settings="R_default")[0]
    )
    assert val == pytest.approx(fx["expected_lnabf"], abs=1e-9)


def test_golden_survidinri_half_vs_full_cnri_scale():
    from survidinri import idi_inf_predictions

    rng = np.random.default_rng(7)
    n = 120
    time = rng.exponential(scale=8.0, size=n)
    event = (time < 10.0).astype(int)
    risk0 = np.clip(0.35 + 0.05 * rng.standard_normal(n), 0.05, 0.95)
    risk1 = np.clip(risk0 + np.where(time < 5.0, 0.12, -0.08), 0.02, 0.98)

    res_def = idi_inf_predictions(time, event, risk0, risk1, t0=5.0, npert=80, seed=11, settings="default")
    res_r = idi_inf_predictions(time, event, risk0, risk1, t0=5.0, npert=80, seed=11, settings="R_default")
    assert res_r.m2_nri.est == pytest.approx(0.5 * res_def.m2_nri.est, rel=1e-6)


def test_golden_deqms_r_default_schema_and_count_shrinkage():
    from deqms import output_result, spectra_count_ebayes

    rng = np.random.default_rng(19)
    n_prot = 80
    counts = np.repeat([1, 2, 5, 15], n_prot // 4)
    sd_true = 0.6 / np.sqrt(counts)
    ctrl = rng.normal(0.0, sd_true[:, None], size=(n_prot, 4))
    trt = rng.normal(0.0, sd_true[:, None], size=(n_prot, 4))
    trt[0] += 1.5
    mat = pd.DataFrame(
        np.hstack([ctrl, trt]),
        index=[f"P{i}" for i in range(n_prot)],
        columns=[f"S{j}" for j in range(8)],
    )
    design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])
    fit_r = spectra_count_ebayes(mat, design=design, count=counts, settings="R_default")
    out_r = output_result(fit_r, coef=1, settings="R_default")
    expected_cols = {
        "logFC", "AveExpr", "t", "P.Value", "adj.P.Val",
        "B", "gene", "count", "sca.t", "sca.P.Value", "sca.adj.pval",
    }
    assert expected_cols.issubset(set(out_r.columns))
    assert out_r.index[0] == "P0" or out_r["gene"].iloc[0] == "P0"


def test_golden_monalisa_r_default_pseudocount_8_shrinkage():
    from monalisa import calc_binned_motif_enrichment, pfm_to_pwm

    # Rows: A, C, G, T -> motif "GATAAC"
    pfm_gata = np.array(
        [
            [2, 92, 2, 92, 92, 2],   # A
            [2, 2, 2, 2, 2, 92],     # C
            [94, 4, 2, 4, 4, 4],     # G
            [2, 2, 94, 2, 2, 2],     # T
        ],
        dtype=float,
    )
    pwm = {"GATAAC": pfm_to_pwm(pfm_gata)}
    seqs = ["TCGATAACTCGATAAC" * 4] * 25 + ["TCGCAGCGTCGCAGCG" * 4] * 25
    bins = ["up"] * 25 + ["down"] * 25

    res_def = calc_binned_motif_enrichment(seqs, bins, pwms=pwm, settings="default")
    res_r = calc_binned_motif_enrichment(seqs, bins, pwms=pwm, settings="R_default")
    df_def = res_def.to_dataframe()
    df_r = res_r.to_dataframe()
    assert float(df_def["log2enr"].abs().max()) > 1.0
    assert float(df_r["log2enr"].abs().max()) < float(df_def["log2enr"].abs().max())
