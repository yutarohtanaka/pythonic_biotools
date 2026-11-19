"""
Comprehensive unit tests for `mmsig` (Rustad et al., Communications Biology 2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmsig import (
    CANONICAL_96_CONTEXTS,
    MMSig,
    build_96_catalog,
    check_strand_bias,
    fit_simplex_ls,
    make_synthetic_cosmic_reference,
    mm_fit_signatures,
    normalize_context_label,
)


@pytest.fixture
def sig_ref() -> pd.DataFrame:
    return make_synthetic_cosmic_reference(
        ["SBS1", "SBS2", "SBS5", "SBS8", "SBS13", "SBS35", "SBS99"]
    )


def test_invalid_settings_raises(sig_ref: pd.DataFrame) -> None:
    """Every primary function and class rejects unknown settings strings."""
    cat = pd.DataFrame({"S1": sig_ref["SBS1"] * 500}, index=sig_ref.index)
    with pytest.raises(ValueError, match="Unknown settings"):
        mm_fit_signatures(cat, sig_ref, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        MMSig(settings="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown settings"):
        build_96_catalog(cat, settings="other")  # type: ignore[arg-type]
    sb = pd.DataFrame([{"sample": "S1", "signature": "SBS99", "transcribed": 20, "untranscribed": 10}])
    with pytest.raises(ValueError, match="Unknown settings"):
        check_strand_bias(sb, settings="wrong")  # type: ignore[arg-type]


def test_canonical_96_contexts_and_reverse_complement() -> None:
    """Canonical 96 channels cover all 6 pyrimidine substitutions x 16 flanking pairs."""
    assert len(CANONICAL_96_CONTEXTS) == 96
    assert len(set(CANONICAL_96_CONTEXTS)) == 96
    assert CANONICAL_96_CONTEXTS[0] == "A[C>A]A"
    assert CANONICAL_96_CONTEXTS[-1] == "T[T>G]T"
    # Purine G>A in CGT reverse-complements to A[C>T]G
    assert normalize_context_label("G", "A", "CGT") == "A[C>T]G"
    assert normalize_context_label("C", "T", "ACG") == "A[C>T]G"


def test_build_96_catalog_from_variant_table_and_sub_tri(sig_ref: pd.DataFrame) -> None:
    """Catalog builder handles both variant-level rows and R-style sub/tri metadata."""
    variants = pd.DataFrame(
        [
            {"sample": "P1", "ref": "C", "alt": "T", "trinuc": "ACG"},
            {"sample": "P1", "ref": "G", "alt": "A", "trinuc": "CGT"},  # also A[C>T]G
            {"sample": "P1", "ref": "C", "alt": "T", "trinuc": "TCA"},
            {"sample": "P2", "ref": "T", "alt": "A", "trinuc": "GTA"},
        ]
    )
    cat = build_96_catalog(variants, reference_index=sig_ref.index)
    assert cat.shape == (96, 2)
    assert list(cat.columns) == ["P1", "P2"]
    assert cat.loc["A[C>T]G", "P1"] == 2.0
    assert cat.loc["T[C>T]A", "P1"] == 1.0
    assert cat.loc["G[T>A]A", "P2"] == 1.0

    # R-style sub/tri table
    sub_list = [c[2:5] for c in CANONICAL_96_CONTEXTS]
    tri_list = [f"{c[0]}{c[2]}{c[6]}" for c in CANONICAL_96_CONTEXTS]
    r_style = pd.DataFrame({"sub": sub_list, "tri": tri_list, "S1": np.ones(96)})
    cat_r = build_96_catalog(r_style, reference_index=sig_ref.index, settings="R_default")
    assert cat_r.shape == (96, 1)
    assert float(cat_r["S1"].sum()) == 96.0


def test_simplex_ls_matches_exact_kkt_conditions(sig_ref: pd.DataFrame) -> None:
    """Simplex-constrained least squares satisfies sum(w)=1, w>=0, and KKT zero dual gap."""
    S = sig_ref[["SBS1", "SBS2", "SBS5", "SBS99"]].to_numpy()
    w_true = np.array([0.30, 0.0, 0.45, 0.25])
    y = S @ w_true
    w_hat = fit_simplex_ls(S.T @ S, S.T @ y)
    assert np.isclose(w_hat.sum(), 1.0, atol=1e-10)
    assert np.all(w_hat >= 0.0)
    np.testing.assert_allclose(w_hat, w_true, atol=1e-6)


def test_backward_cosine_pruning_eliminates_spurious_signatures(sig_ref: pd.DataFrame) -> None:
    """Backward pruning eliminates absent signatures while recovering true exposures."""
    rng = np.random.default_rng(101)
    w_true = {"SBS1": 0.30, "SBS5": 0.40, "SBS2": 0.30}
    profile = sum(w * sig_ref[k].to_numpy() for k, w in w_true.items())
    counts = rng.multinomial(3000, profile)
    cat = pd.DataFrame({"MM_APOBEC": counts}, index=sig_ref.index)

    res = mm_fit_signatures(
        cat, sig_ref, bootstrap=False, settings="default", dbg=True
    )
    exp = res.exposures.loc["MM_APOBEC"]
    assert np.isclose(exp.sum(), 1.0, atol=1e-8)
    #Absent signatures SBS13, SBS35, SBS99, SBS8 should be pruned to exact 0
    for absent in ("SBS13", "SBS35", "SBS99", "SBS8"):
        assert exp[absent] == 0.0
    for active, target in w_true.items():
        assert abs(exp[active] - target) < 0.05
    assert res.gof_df.loc[0, "cosine_similarity"] > 0.98
    assert len(res.pruning_trace["MM_APOBEC"]) >= 4


def test_force_include_preserves_clock_like_sbs1_sbs5(sig_ref: pd.DataFrame) -> None:
    """Forced clock-like signatures (SBS1, SBS5) are exempt from backward pruning."""
    # Pure SBS2 sample: SBS1 and SBS5 have near-zero weight, yet are not removed from active trace
    counts = np.round(sig_ref["SBS2"].to_numpy() * 2000).astype(float)
    cat = pd.DataFrame({"Pure_SBS2": counts}, index=sig_ref.index)

    res_forced = mm_fit_signatures(
        cat,
        sig_ref,
        force_include=("SBS1", "SBS5"),
        cos_sim_threshold=0.05,
        bootstrap=False,
        settings="R_default",
        dbg=True,
    )
    removed = {step["removed_signature"] for step in res_forced.pruning_trace["Pure_SBS2"]}
    assert "SBS1" not in removed
    assert "SBS5" not in removed

    # When force_include=(), SBS1 and SBS5 CAN be pruned
    res_unforced = mm_fit_signatures(
        cat,
        sig_ref,
        force_include=(),
        cos_sim_threshold=0.05,
        bootstrap=False,
        settings="R_default",
        dbg=True,
    )
    assert res_unforced.exposures.loc["Pure_SBS2", "SBS2"] == pytest.approx(1.0, abs=1e-6)
    assert res_unforced.exposures.loc["Pure_SBS2", "SBS1"] == 0.0
    assert res_unforced.exposures.loc["Pure_SBS2", "SBS5"] == 0.0


def test_multinomial_bootstrap_cis_and_present_prob(sig_ref: pd.DataFrame) -> None:
    """Parametric multinomial bootstrap computes well-ordered CIs and presence probabilities."""
    rng = np.random.default_rng(2024)
    p = 0.25 * sig_ref["SBS1"] + 0.45 * sig_ref["SBS5"] + 0.30 * sig_ref["SBS13"]
    cat = pd.DataFrame({"S_boot": rng.multinomial(1500, p)}, index=sig_ref.index)

    res = mm_fit_signatures(
        cat, sig_ref, bootstrap=True, n_boot=80, seed=77, settings="default"
    )
    assert res.bootstrap_df is not None
    bdf = res.bootstrap_df.set_index("signature")
    assert set(["sample", "signature", "estimate", "ci_low", "ci_high", "sd", "present_prob"]).issubset(
        res.bootstrap_df.columns
    )
    assert np.all(bdf["ci_low"] <= bdf["ci_high"] + 1e-12)
    assert bdf.loc["SBS13", "ci_low"] > 0.20
    assert bdf.loc["SBS13", "ci_high"] < 0.40
    assert bdf.loc["SBS13", "present_prob"] == 1.0
    assert bdf.loc["SBS35", "present_prob"] < 0.15


def test_r_default_sbs8_sbs99_ci_confirmation_rule(sig_ref: pd.DataFrame) -> None:
    """In R_default, borderline SBS99 with ci_low == 0 is dropped while default retains it."""
    p = 0.30 * sig_ref["SBS1"] + 0.52 * sig_ref["SBS5"] + 0.18 * sig_ref["SBS99"]
    counts = p.to_numpy() * 55.0
    cat = pd.DataFrame({"LowBurden": counts}, index=sig_ref.index)

    res_def = mm_fit_signatures(
        cat,
        sig_ref[["SBS1", "SBS5", "SBS99"]],
        cos_sim_threshold=0.002,
        bootstrap=True,
        n_boot=200,
        seed=19,
        settings="default",
    )
    res_r = mm_fit_signatures(
        cat,
        sig_ref[["SBS1", "SBS5", "SBS99"]],
        cos_sim_threshold=0.002,
        bootstrap=True,
        n_boot=200,
        seed=19,
        settings="R_default",
    )
    assert res_r.bootstrap_df is not None
    row_99 = res_r.bootstrap_df.set_index("signature").loc["SBS99"]
    assert 0.10 < row_99["present_prob"] < 0.975
    assert row_99["ci_low"] == 0.0
    assert res_r.exposures.loc["LowBurden", "SBS99"] == 0.0
    assert res_def.exposures.loc["LowBurden", "SBS99"] > 0.10
    assert np.isclose(res_r.exposures.loc["LowBurden"].sum(), 1.0)


def test_default_vs_r_default_n_boot_and_solver_parity(sig_ref: pd.DataFrame) -> None:
    """Default uses n_boot=500 (EM+NNLS) and R_default uses n_boot=1000 (simplex LS)."""
    counts = np.round((0.45 * sig_ref["SBS1"] + 0.55 * sig_ref["SBS5"]).to_numpy() * 800)
    cat = pd.DataFrame({"Clock": counts}, index=sig_ref.index)

    res_def = mm_fit_signatures(cat, sig_ref[["SBS1", "SBS5"]], seed=7, settings="default")
    res_r = mm_fit_signatures(cat, sig_ref[["SBS1", "SBS5"]], seed=7, settings="R_default")
    assert res_def.bootstrap_df is not None and res_r.bootstrap_df is not None
    assert abs(res_def.exposures.loc["Clock", "SBS1"] - 0.45) < 0.03
    assert abs(res_r.exposures.loc["Clock", "SBS1"] - 0.45) < 0.03


def test_strand_bias_test_summary_and_variant_level() -> None:
    """check_strand_bias detects melphalan/platinum strand asymmetry in both input formats."""
    summary_sb = pd.DataFrame(
        [
            {"sample": "MelphalanPos", "signature": "SBS99", "transcribed": 140, "untranscribed": 55},
            {"sample": "Control", "signature": "SBS99", "transcribed": 52, "untranscribed": 48},
        ]
    )
    out_def = check_strand_bias(summary_sb, alpha=0.05, settings="default")
    out_r = check_strand_bias(summary_sb, alpha=0.05, settings="R_default")

    assert bool(out_def.loc[0, "significant"]) is True
    assert bool(out_def.loc[1, "significant"]) is False
    assert out_def.loc[0, "p_value"] == pytest.approx(out_r.loc[0, "p_value"])
    # Default uses Haldane +0.5 smoothing; R_default uses exact ratio 140/55
    assert out_r.loc[0, "strand_ratio"] == pytest.approx(140.0 / 55.0)
    assert out_def.loc[0, "strand_ratio"] == pytest.approx(140.5 / 55.5)

    # Variant-level input
    var_rows = (
        [{"sample": "S_plat", "ref": "C", "alt": "A", "strand": "T"}] * 60
        + [{"sample": "S_plat", "ref": "C", "alt": "A", "strand": "U"}] * 20
    )
    out_var = check_strand_bias(pd.DataFrame(var_rows), settings="default")
    s35 = out_var[out_var["signature"] == "SBS35"].iloc[0]
    assert bool(s35["significant"]) is True


def test_strand_bias_confirmation_filters_unconfirmed_sbs35_sbs99(sig_ref: pd.DataFrame) -> None:
    """mm_fit_signatures drops SBS35/SBS99 when strand_bias_df shows no strand asymmetry."""
    p = 0.30 * sig_ref["SBS1"] + 0.40 * sig_ref["SBS5"] + 0.30 * sig_ref["SBS99"]
    counts = np.round(p.to_numpy() * 1500)
    cat = pd.DataFrame({"Confirmed": counts, "Unconfirmed": counts}, index=sig_ref.index)

    strand_df = pd.DataFrame(
        [
            {"sample": "Confirmed", "signature": "SBS99", "transcribed": 180, "untranscribed": 70},
            {"sample": "Unconfirmed", "signature": "SBS99", "transcribed": 124, "untranscribed": 126},
        ]
    )
    res = mm_fit_signatures(
        cat,
        sig_ref[["SBS1", "SBS5", "SBS99"]],
        bootstrap=False,
        strand_bias_df=strand_df,
        settings="R_default",
    )
    assert res.exposures.loc["Confirmed", "SBS99"] > 0.25
    assert res.exposures.loc["Unconfirmed", "SBS99"] == 0.0
    assert np.isclose(res.exposures.loc["Unconfirmed"].sum(), 1.0)


def test_mmsig_class_interface_and_per_sample_sigs_cons(sig_ref: pd.DataFrame) -> None:
    """MMSig class supports per-sample candidate signature subsets and summary export."""
    p1 = np.round((0.4 * sig_ref["SBS1"] + 0.6 * sig_ref["SBS2"]).to_numpy() * 1000)
    p2 = np.round((0.5 * sig_ref["SBS1"] + 0.5 * sig_ref["SBS35"]).to_numpy() * 1000)
    cat = pd.DataFrame({"S1": p1, "S2": p2}, index=sig_ref.index)

    model = MMSig(bootstrap=True, n_boot=40, seed=11, settings="default")
    res = model.fit(
        cat,
        sig_ref,
        sigs_cons={"S1": ["SBS1", "SBS2", "SBS5"], "S2": ["SBS1", "SBS5", "SBS35"]},
    )
    assert res.exposures.loc["S1", "SBS35"] == 0.0
    assert res.exposures.loc["S2", "SBS2"] == 0.0
    assert res.exposures.loc["S1", "SBS2"] > 0.50
    assert res.exposures.loc["S2", "SBS35"] > 0.40
    df_out = res.to_dataframe()
    assert "cosine_similarity" in df_out.columns
    assert "MMSigResult" in res.summary()


def test_zero_mutation_sample_handled_gracefully(sig_ref: pd.DataFrame) -> None:
    """All-zero mutation sample produces zero exposures and zero warnings."""
    cat = pd.DataFrame(
        {"EmptySample": np.zeros(96), "NormalSample": sig_ref["SBS1"].to_numpy() * 400},
        index=sig_ref.index,
    )
    res = mm_fit_signatures(cat, sig_ref[["SBS1", "SBS5"]], n_boot=25, settings="default")
    assert float(res.exposures.loc["EmptySample"].sum()) == 0.0
    assert float(res.exposures.loc["NormalSample"].sum()) == pytest.approx(1.0)
