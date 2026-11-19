"""
Comprehensive unit tests for `chord` covering 96 SBS + 17 Indel + 19 SV context
extraction, alignment-invariant bidirectional microhomology (`bimh`) scanning,
`pysam.FastaFile` reference lookup, pan-cancer HRD (`BRCA1`-type vs `BRCA2`-type
vs `HRP`) classification, custom Random Forest training, and `"default"` vs
`"R_default"` parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from chord import (
    ALL_CHORD_CHANNELS,
    CHORDClassifier,
    CHORDResult,
    INDEL_CHANNELS,
    SBS96_CHANNELS,
    SV_CHANNELS,
    chord_predict,
    classify_indel_sequence,
    classify_sv_channel,
    extract_sigs_chord,
    normalize_sbs96_context,
    simulate_chord_archetypes,
)


def test_sbs96_pyrimidine_normalization_and_channel_count():
    assert len(SBS96_CHANNELS) == 96
    assert len(INDEL_CHANNELS) == 17
    assert len(SV_CHANNELS) == 19
    assert len(ALL_CHORD_CHANNELS) == 132

    # Direct pyrimidine C>T in ACG -> A[C>T]G
    assert normalize_sbs96_context("C", "T", "ACG") == "A[C>T]G"
    # Purine G>A in CGT (reverse complement of ACG) -> A[C>T]G
    assert normalize_sbs96_context("G", "A", "CGT") == "A[C>T]G"
    # Purine A>T in TAC -> revcomp(TAC)=GTA, A->T, T->A -> G[T>A]A
    assert normalize_sbs96_context("A", "T", "TAC") == "G[T>A]A"
    # Ambiguous base returns None
    assert normalize_sbs96_context("C", "T", "ANG") is None


def test_indel_bimh_alignment_invariance_and_repeats():
    # 3-bp microhomology deletion: reference GG [CGAT] CGAAGG -> GG CGAAGG
    # Left-aligned representation:
    c_left = classify_indel_sequence("GCGAT", "G", left_flank="G", right_flank="CGAAGG")
    # Shifted 1 bp right: GGC [GATC] GAAGG -> GGC GAAGG (1 bp left homology 'C' + 2 bp right 'GA' = 3)
    c_mid = classify_indel_sequence("CGATC", "C", left_flank="GG", right_flank="GAAGG")
    # Shifted 3 bp right: GGCGA [TCGA] AGG -> GGCGA AGG (3 bp left homology 'CGA' + 0 bp right = 3)
    c_right = classify_indel_sequence("ATCGA", "A", left_flank="GGCG", right_flank="AGG")

    assert c_left == "del.mh.bimh.3"
    assert c_mid == "del.mh.bimh.3"
    assert c_right == "del.mh.bimh.3"

    # Hidden shifted 4-bp tandem repeat: T [TCGA][TCGA] AGG presented as T[TCGAT -> T]CGAAGG -> del.rep.len.4
    assert classify_indel_sequence("TCGAT", "T", left_flank="T", right_flank="CGAAGG") == "del.rep.len.4"

    # Homopolymer 1-bp deletion in AAAA -> del.rep.len.1
    assert classify_indel_sequence("GA", "G", left_flank="T", right_flank="AAAT") == "del.rep.len.1"
    # Dinucleotide repeat insertion CA in CACACA -> ins.rep.len.2
    assert classify_indel_sequence("T", "TCA", left_flank="G", right_flank="CACAG") == "ins.rep.len.2"
    # Non-homologous deletion and insertion
    assert classify_indel_sequence("GACT", "G", left_flank="TTT", right_flank="GGG") == "del.none"
    assert classify_indel_sequence("T", "TGG", left_flank="AAA", right_flank="CCC") == "ins.none"


def test_sv_length_and_translocation_binning():
    assert classify_sv_channel("DEL", 500) == "DEL_0_1e3"
    assert classify_sv_channel("DUP", 5_000) == "DUP_1e3_1e4"
    assert classify_sv_channel("<DUP:TANDEM>", 45_000) == "DUP_1e4_1e5"
    assert classify_sv_channel("INV", 250_000) == "INV_1e5_1e6"
    assert classify_sv_channel("DEL", 3_500_000) == "DEL_1e6_1e7"
    assert classify_sv_channel("DUP", 25_000_000) == "DUP_1e7_inf"
    assert classify_sv_channel("TRA", None) == "TRA"
    assert classify_sv_channel("BND", 0) == "TRA"


def test_extract_sigs_chord_with_pysam_fasta(tmp_path: Path):
    # Construct synthetic chr1 sequence with planted SNV, MH deletion, and repeat deletion
    # 1-based positions:
    # 1..10:   TTTTTTTTTT
    # 11..13:  ACG (pos 12 is C; C>T -> A[C>T]G)
    # 14..23:  TTTTTTTTTT
    # 24..34:  ATCGATCGTTT (pos 23='T' anchor; deleting 'ATCG' at 24..27 leaves 'ATCGTTT' -> 4bp repeat!
    #          Let's make pos 24..28 'ATCGG' followed by 'ATCCC' -> 3bp MH 'ATC'!)
    seq = (
        "TTTTTTTTTT"  # 1..10
        "ACG"  # 11..13 (pos 12 = C in ACG)
        "TTTTTTTTTT"  # 14..23 (pos 23 = T)
        "ATCGGATCCC"  # 24..33: deleting ATCGG (24..28) before ATCCC has 3-bp MH 'ATC' -> del.mh.bimh.3
        "GGGGGGGGGG"  # 34..43
        "CCCCC"  # 44..48: deleting 1 C at 44 (anchor G at 43) leaves CCCC -> del.rep.len.1
        "TTTTTTTTTT"  # 49..58
    )
    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">chr1\n{seq}\n", encoding="utf-8")
    pysam.faidx(str(fa_path))

    snv_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [12],
            "ref": ["C"],
            "alt": ["T"],
        }
    )
    indel_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [23, 43],
            "ref": ["TATCGG", "GC"],
            "alt": ["T", "G"],
        }
    )
    sv_df = pd.DataFrame(
        {
            "chrom1": ["chr1", "chr1"],
            "pos1": [100, 100],
            "chrom2": ["chr1", "chr2"],
            "pos2": [25_100, 500],
            "sv_type": ["DUP", "BND"],
        }
    )

    ctx = extract_sigs_chord(
        snv_df=snv_df,
        indel_df=indel_df,
        sv_df=sv_df,
        ref_fasta=fa_path,
        sample_id="wgs_tumor",
    )
    assert ctx.shape == (1, 132)
    assert ctx.index[0] == "wgs_tumor"
    assert int(ctx.loc["wgs_tumor", "A[C>T]G"]) == 1
    assert int(ctx.loc["wgs_tumor", "del.mh.bimh.3"]) == 1
    assert int(ctx.loc["wgs_tumor", "del.rep.len.1"]) == 1
    assert int(ctx.loc["wgs_tumor", "DUP_1e4_1e5"]) == 1
    assert int(ctx.loc["wgs_tumor", "TRA"]) == 1


def test_extract_sigs_chord_multisample_and_precomputed_indels():
    snv_df = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "ref": ["C", "G", "T"],
            "alt": ["A", "T", "G"],
            "trinuc": ["ACA", "TGT", "CTG"],
        }
    )
    indel_df = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S2"],
            "indel_type": ["del.mh", "del.rep", "ins.none"],
            "mh_len": [4, 0, 0],
            "rep_len": [0, 2, 0],
            "indel_len": [6, 2, 3],
        }
    )
    ctx = extract_sigs_chord(snv_df=snv_df, indel_df=indel_df)
    assert list(ctx.index) == ["S1", "S2"]
    assert int(ctx.loc["S1", "A[C>A]A"]) == 2
    assert int(ctx.loc["S1", "del.mh.bimh.4"]) == 1
    assert int(ctx.loc["S2", "C[T>G]G"]) == 1
    assert int(ctx.loc["S2", "del.rep.len.2"]) == 1
    assert int(ctx.loc["S2", "ins.none"]) == 1


def test_pretrained_chord_accurately_separates_brca1_brca2_and_hrp():
    # Generate independent held-out cohort with a distinct random seed
    heldout_df, truth_labels = simulate_chord_archetypes(n_per_class=25, seed=9999)
    # Ensure held-out indels >= 50 so R_default doesn't mask low-indel samples
    heldout_df[list(INDEL_CHANNELS)] = heldout_df[list(INDEL_CHANNELS)] * 3

    for mode in ("default", "R_default"):
        res = chord_predict(heldout_df, settings=mode)
        assert isinstance(res, CHORDResult)
        pred_df = res.to_dataframe()

        # Check binary HRD accuracy and BRCA1 vs BRCA2 subtype accuracy
        is_true_hrd = np.array([lbl in ("BRCA1", "BRCA2") for lbl in truth_labels])
        is_pred_hrd = (pred_df["hr_status"] == "HR_deficient").to_numpy()
        assert float(np.mean(is_true_hrd == is_pred_hrd)) >= 0.94

        b1_mask = np.array([lbl == "BRCA1" for lbl in truth_labels])
        b2_mask = np.array([lbl == "BRCA2" for lbl in truth_labels])
        assert float(np.mean(pred_df.loc[b1_mask, "hrd_type"] == "BRCA1_type")) >= 0.90
        assert float(np.mean(pred_df.loc[b2_mask, "hrd_type"] == "BRCA2_type")) >= 0.90


def test_single_sample_brca1_vs_brca2_signature_specificity():
    # Hand-crafted canonical BRCA1 vs BRCA2 vs HRP context vectors
    cols = list(ALL_CHORD_CHANNELS)
    b1 = pd.DataFrame(0, index=["brca1_case"], columns=cols)
    b1.loc[:, list(SBS96_CHANNELS)] = 25  # flat SBS3
    b1.loc[:, ["del.mh.bimh.2", "del.mh.bimh.3", "del.mh.bimh.4"]] = [30, 35, 25]
    b1.loc[:, "del.rep.len.1"] = 15
    b1.loc[:, ["DUP_1e3_1e4", "DUP_1e4_1e5", "TRA"]] = [45, 50, 10]

    b2 = b1.copy()
    b2.index = pd.Index(["brca2_case"])
    b2.loc[:, ["DUP_1e3_1e4", "DUP_1e4_1e5"]] = [1, 1]
    b2.loc[:, ["DEL_1e3_1e4", "DEL_1e4_1e5"]] = [48, 42]

    hrp = pd.DataFrame(0, index=["hrp_case"], columns=cols)
    hrp.loc[:, ["A[C>T]G", "C[C>T]G", "G[C>T]G", "T[C>T]G"]] = 400
    hrp.loc[:, ["del.rep.len.1", "ins.rep.len.1", "del.none"]] = [50, 30, 40]
    hrp.loc[:, ["INV_1e5_1e6", "TRA"]] = [15, 10]

    res_b1 = chord_predict(b1, settings="default")
    res_b2 = chord_predict(b2, settings="default")
    res_hrp = chord_predict(hrp, settings="default")

    assert res_b1.hr_status == "HR_deficient"
    assert res_b1.hrd_type == "BRCA1_type"
    assert res_b1.p_BRCA1 > 0.60
    assert np.isclose(res_b1.p_hrd, res_b1.p_BRCA1 + res_b1.p_BRCA2)

    assert res_b2.hr_status == "HR_deficient"
    assert res_b2.hrd_type == "BRCA2_type"
    assert res_b2.p_BRCA2 > 0.60

    assert res_hrp.hr_status == "HR_proficient"
    assert res_hrp.hrd_type == "none"
    assert res_hrp.p_hrp > 0.65


def test_r_default_scale_invariance_and_zero_sv_handling():
    syn_df, _ = simulate_chord_archetypes(n_per_class=2, seed=42)
    sample = syn_df.iloc[[3]].copy()
    sample[list(INDEL_CHANNELS)] += 10  # ensure >= 50 indels

    scaled = sample * 5
    r1 = chord_predict(sample, settings="R_default")
    r2 = chord_predict(scaled, settings="R_default")
    assert np.isclose(r1.p_BRCA1, r2.p_BRCA1)
    assert np.isclose(r1.p_BRCA2, r2.p_BRCA2)
    assert np.isclose(r1.p_hrd, r2.p_hrd)

    # Zero SV counts should not produce NaNs
    no_sv = sample.copy()
    no_sv[list(SV_CHANNELS)] = 0
    r_nosv = chord_predict(no_sv, settings="R_default")
    assert np.isfinite(r_nosv.p_hrd)


def test_low_indel_load_default_vs_r_default_and_msi_override():
    cols = list(ALL_CHORD_CHANNELS)
    low_ind = pd.DataFrame(0, index=["low_indel_brca1"], columns=cols)
    low_ind.loc[:, list(SBS96_CHANNELS)] = 15
    # Total indels = 24 (< 50), strongly MH-enriched
    low_ind.loc[:, ["del.mh.bimh.3", "del.mh.bimh.4", "del.rep.len.1"]] = [10, 10, 4]
    low_ind.loc[:, ["DUP_1e3_1e4", "DUP_1e4_1e5"]] = [30, 35]

    res_def = chord_predict(low_ind, settings="default")
    res_r = chord_predict(low_ind, settings="R_default")

    assert "low_indel_load" in res_def.remarks
    assert res_def.hr_status == "HR_deficient"
    assert res_def.hrd_type == "BRCA1_type"

    assert res_r.hr_status == "cannot_be_determined"
    assert res_r.hrd_type == "cannot_be_determined"
    assert res_r.remarks == "low_indel_load"

    # MSI override (> 14,000 repeat indels)
    msi_df = low_ind.copy()
    msi_df.index = pd.Index(["msi_tumor"])
    msi_df.loc[:, "del.rep.len.1"] = 10_000
    msi_df.loc[:, "ins.rep.len.1"] = 5_500
    for mode in ("default", "R_default"):
        res_msi = chord_predict(msi_df, settings=mode)
        assert res_msi.hr_status == "MSI"
        assert res_msi.hrd_type == "none"


def test_custom_classifier_fit_and_legacy_r_column_aliases():
    train_df, train_y = simulate_chord_archetypes(n_per_class=20, seed=101)
    clf = CHORDClassifier(settings="default", n_estimators=80, random_state=7)
    clf.fit(train_df, train_y)
    assert clf.is_custom_fitted_ is True

    probs = clf.predict_proba(train_df.iloc[:5])
    assert {"p_BRCA1", "p_BRCA2", "p_hrd", "p_hrp"} == set(probs.columns)

    # Test legacy R CHORD SV column names (e.g. DUP_1e03_1e04_bp)
    legacy_df = pd.DataFrame(
        {
            "A[C>A]A": [50],
            "del.mh.bimh.3": [40],
            "del.rep.len.1": [20],
            "DUP_1e03_1e04_bp": [35],
            "DUP_1e04_1e05_bp": [35],
        },
        index=["legacy_sample"],
    )
    res_leg = clf.predict(legacy_df)
    assert res_leg.hr_status == "HR_deficient"
    assert res_leg.hrd_type == "BRCA1_type"
    assert "CHORDResult" in res_leg.summary()


def test_invalid_settings_and_inputs_raise_value_error():
    sample_df = pd.DataFrame({"A[C>A]A": [10], "del.mh.bimh.3": [60]}, index=["s1"])

    for bad in ("bad", "R", "v2", ""):
        with pytest.raises(ValueError, match="Unknown settings"):
            extract_sigs_chord(settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            chord_predict(sample_df, settings=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown settings"):
            CHORDClassifier(settings=bad)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hrd_cutoff"):
        chord_predict(sample_df, hrd_cutoff=1.5)
    with pytest.raises(ValueError, match="non-empty"):
        chord_predict(pd.DataFrame())
