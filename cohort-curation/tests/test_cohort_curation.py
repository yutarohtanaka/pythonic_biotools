from __future__ import annotations

import numpy as np
import pandas as pd

from cohort_curation import (
    binarize_arm_events,
    build_genie_panel_matrix,
    cross_check_gistic_panel,
    harmonize_hgnc_symbols,
    mask_unprofiled_panel_genes,
    parse_cellosaurus,
    parse_gistic_peak_genes,
    repair_excel_mangled_genes,
)


def test_public_cohort_parsers(tmp_path):
    p1 = tmp_path / "data_gene_panel_PANEL1.txt"
    p1.write_text("cancer_study_identifier: genie\nstable_id: PANEL1\ngene_list:\tTP53\tKRAS\n")
    p2 = tmp_path / "data_gene_panel_PANEL2.txt"
    p2.write_text("cancer_study_identifier: genie\nstable_id: PANEL2\ngene_list:\tTP53\tKRAS\tLZTR1\n")

    cov = build_genie_panel_matrix(tmp_path)
    alt_df = pd.DataFrame({"TP53": [1, 0], "KRAS": [0, 1], "LZTR1": [0, 0]}, index=["S1", "S2"])
    masked = mask_unprofiled_panel_genes(alt_df, {"S1": "PANEL1", "S2": "PANEL2"}, cov)
    assert np.isnan(masked.loc["S1", "LZTR1"]) and masked.loc["S2", "LZTR1"] == 0.0

    assert repair_excel_mangled_genes(["1-Mar", "7-Sep", "1-Dec", "15-Sep"]) == ["MARCHF1", "SEPTIN7", "DELEC1", "SELENOF"]

    hgnc_df = pd.DataFrame(
        [{"symbol": "KMT2D", "status": "Approved", "prev_symbol": "MLL2", "alias_symbol": "ALR", "entrez_id": "8085"}]
    )
    assert list(harmonize_hgnc_symbols(pd.Series(["MLL2", "ALR"]), hgnc_df)) == ["KMT2D", "KMT2D"]

    cello_file = tmp_path / "cellosaurus.txt"
    cello_file.write_text("ID   HCT 116\nAC   CVCL_0291\nDR   DepMap; ACH-000971\nCC   Doubling time: 20 hours.\n//\n")
    qc_df = parse_cellosaurus(cello_file, depmap_ids={"ACH-000971"})
    assert qc_df.iloc[0]["doubling_time_hours"] == 20.0

    arm_df = pd.DataFrame({"3p_status": ["Loss", "Unchanged", np.nan]}, index=["S1", "S2", "S3"])
    bin_df = binarize_arm_events(arm_df, {"3p_loss": ("3p_status", "Loss")})
    assert bin_df.loc["S1", "3p_loss"] == 1.0 and np.isnan(bin_df.loc["S3", "3p_loss"])

    gistic_txt = tmp_path / "amp_genes.conf_99.txt"
    gistic_txt.write_text(
        "cytoband\t8q24.21\n"
        "q value\t1e-20\n"
        "residual q value\t1e-18\n"
        "wide peak boundaries\tchr8:127000000-129000000\n"
        "genes in wide peak\t[MYC]\n"
        "\tPVT1\n"
    )
    peaks = parse_gistic_peak_genes(gistic_txt)
    chk = cross_check_gistic_panel(["MYC", "TP53"], amp_peaks=peaks, del_peaks={})
    assert bool(chk.loc[chk["gene"] == "MYC", "in_significant_amp_peak"].iloc[0]) is True
