"""End-to-end verification suite for public UX features:
1. Pure-MIT wheel packaging & collision-safe `pythonic_biotools` lazy namespace
2. Turnkey bioinformatics file readers (`pythonic_biotools.io`)
3. Pipeline CLI subcommands (`pbt run <tool>`)
4. Publication `.plot()` diagnostic figures
5. Colab notebooks, MkDocs configuration, CITATION.cff, Dockerfile & environment.yml
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools as pbt
from pythonic_biotools import cli as pbt_cli
from pythonic_biotools import io as pbt_io


def test_pure_mit_wheel_contents_and_lazy_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import setuptools.build_meta as bm

    monkeypatch.chdir(REPO_ROOT)
    try:
        whl_name = bm.build_wheel(str(tmp_path))
    finally:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "pythonic_biotools.egg-info", ignore_errors=True)
    whl_path = tmp_path / whl_name
    assert whl_path.is_file()

    with zipfile.ZipFile(whl_path) as zf:
        names = set(zf.namelist())

    # Core root modules & py.typed
    assert "pythonic_biotools/__init__.py" in names
    assert "pythonic_biotools/io.py" in names
    assert "pythonic_biotools/cli.py" in names
    assert "pythonic_biotools/py.typed" in names

    # All 63 MIT packages must be bundled under pythonic_biotools.<canonical>/
    assert len(pbt.MIT_PACKAGES) == 63
    for pkg_dir in pbt.MIT_PACKAGES:
        spec = pbt.PACKAGE_SPECS[pkg_dir]
        assert f"pythonic_biotools/{spec.canonical_attr}/__init__.py" in names
        assert f"pythonic_biotools/{spec.canonical_attr}/py.typed" in names

    # Zero GPL/AGPL packages may appear inside the MIT wheel
    assert len(pbt.COPYLEFT_PACKAGES) == 7
    for gpl_dir in pbt.COPYLEFT_PACKAGES:
        spec = pbt.PACKAGE_SPECS[gpl_dir]
        assert not any(n.startswith(f"pythonic_biotools/{spec.canonical_attr}/") for n in names)

    # Lazy namespace imports & aliases
    assert hasattr(pbt.purecn, "run_purecn")
    assert hasattr(pbt.facets, "run_facets")
    assert hasattr(pbt.ctwas, "run_ctwas")
    assert hasattr(pbt.survrm2, "rmst2")
    assert hasattr(pbt.deqms, "spectra_count_ebayes")
    assert hasattr(pbt.xenofilterR, "Xenofilter")
    assert pbt.bayesprism is pbt.pythonic_bayesprism
    assert pbt.biostats is pbt.pythonic_biostats


def test_turnkey_bioinformatics_file_readers(tmp_path: Path):
    # 1. GDC/cBioPortal MAF (.maf.gz)
    maf_gz = tmp_path / "cohort.maf.gz"
    maf_text = (
        "#version 2.4\n"
        "Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele2\t"
        "Variant_Classification\tVariant_Type\tTumor_Sample_Barcode\tt_ref_count\tt_alt_count\n"
        "TP53\tchr17\t7577120\tC\tT\tMissense_Mutation\tSNP\tSAMP_1\t40\t35\n"
        "KRAS\tchr12\t25398284\tC\tA\tMissense_Mutation\tSNP\tSAMP_2\t60\t25\n"
        "TTN\tchr2\t179400000\tG\tA\tSilent\tSNP\tSAMP_1\t50\t20\n"
    )
    with gzip.open(maf_gz, "wt", encoding="utf-8") as fh:
        fh.write(maf_text)

    maf_df = pbt_io.read_maf(maf_gz, keep_silent=False)
    assert len(maf_df) == 2
    alt_mat = pbt_io.maf_to_binary_matrix(maf_df)
    assert alt_mat.loc["TP53", "SAMP_1"] == 1.0 and alt_mat.loc["TP53", "SAMP_2"] == 0.0

    # 2. VCF (.vcf.gz)
    vcf_gz = tmp_path / "sample.vcf.gz"
    vcf_text = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTUMOR\n"
        "chr1\t10500\trs1\tA\tG\t99\tPASS\tAF=0.0001;DP=100\tGT:AD:DP\t0/1:65,35:100\n"
        "chr1\t20500\t.\tC\tT\t20\tLOWQ\tAF=0.25;DP=50\tGT:AD:DP\t0/1:25,25:50\n"
    )
    with gzip.open(vcf_gz, "wt", encoding="utf-8") as fh:
        fh.write(vcf_text)
    vcf_df = pbt_io.read_vcf_variants(vcf_gz, pass_only=True)
    assert len(vcf_df) == 1
    assert float(vcf_df["vaf"].iloc[0]) == pytest.approx(0.35)

    # 3. GATK DenoisedCopyRatios with @HD/@SQ headers
    gatk_tsv = tmp_path / "sample.denoisedCR.tsv"
    gatk_tsv.write_text(
        "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:248956422\n"
        "CONTIG\tSTART\tEND\tLOG2_COPY_RATIO\n"
        "chr1\t1\t1000000\t0.02\n"
        "chr1\t1000001\t2000000\t0.58\n",
        encoding="utf-8",
    )
    cr_df = pbt_io.read_gatk_copy_ratios(gatk_tsv)
    assert len(cr_df) == 2 and "logR" in cr_df.columns

    # 4. Bismark .cov.gz -> DSS
    c1 = tmp_path / "c1.bismark.cov.gz"
    with gzip.open(c1, "wt", encoding="utf-8") as fh:
        fh.write("chr1\t1000\t1000\t20.0\t4\t16\nchr1\t1050\t1050\t80.0\t24\t6\n")
    bs = pbt_io.read_bismark_cov([c1])
    assert bs.N_mat.shape == (2, 1) and int(bs.X_mat[1, 0]) == 24

    # 5. MaxQuant proteinGroups.txt (filters '+' contaminants & Reverse)
    mq_txt = tmp_path / "proteinGroups.txt"
    mq_txt.write_text(
        "Protein IDs\tGene names\tRazor + unique peptides\tPotential contaminant\tReverse\tOnly identified by site\t"
        "LFQ intensity C1\tLFQ intensity C2\tLFQ intensity T1\tLFQ intensity T2\n"
        "P01\tEGFR\t14\t\t\t\t100000\t110000\t420000\t400000\n"
        "CON__KRT1\tKRT1\t8\t+\t\t\t500000\t500000\t500000\t500000\n"
        "REV__DEC\tDEC\t3\t\t+\t\t20000\t22000\t21000\t19000\n",
        encoding="utf-8",
    )
    abund, pep_counts = pbt_io.read_maxquant_diann(mq_txt)
    assert abund.shape == (1, 4) and int(pep_counts.iloc[0]) == 14

    # 6. GWAS sumstats (OR -> log(OR)) + PLINK LD
    gwas_tsv = tmp_path / "gwas.tsv"
    gwas_tsv.write_text(
        "MarkerName\tEffect_Allele\tOther_Allele\tOR\tSE\tP\tN\n"
        "rs10\tA\tG\t2.0\t0.10\t1e-11\t50000\n",
        encoding="utf-8",
    )
    gwas_df = pbt_io.read_gwas_sumstats(gwas_tsv)
    assert float(gwas_df["beta"].iloc[0]) == pytest.approx(np.log(2.0), abs=1e-9)

    ld_file = tmp_path / "locus.ld"
    bim_file = tmp_path / "locus.bim"
    ld_file.write_text("1.0 0.5\n0.5 1.0\n", encoding="utf-8")
    bim_file.write_text("1\trs10\t0\t1000\tA\tG\n1\trs11\t0\t2000\tC\tT\n", encoding="utf-8")
    ld_res = pbt_io.read_plink_ld(ld_file, bim_path=bim_file)
    ld_mat = ld_res[0] if isinstance(ld_res, tuple) else ld_res
    assert np.asarray(ld_mat).shape == (2, 2)


def test_pbt_cli_pipeline_runners_end_to_end(tmp_path: Path):
    # 1. pbt run scarhrd
    seg_tsv = tmp_path / "segs.tsv"
    hrd_out = tmp_path / "hrd_out.tsv"
    pd.DataFrame(
        [
            {"chrom": "chr1", "start": 1_000_000, "end": 40_000_000, "major_cn": 2, "minor_cn": 0},
            {"chrom": "chr1", "start": 40_000_000, "end": 120_000_000, "major_cn": 2, "minor_cn": 2},
        ]
    ).to_csv(seg_tsv, sep="\t", index=False)
    rc = pbt_cli.main(
        ["run", "scarhrd", "--segments", str(seg_tsv), "--genome", "GRCh38", "--out", str(hrd_out), "--settings", "R_default"]
    )
    assert rc == 0 and hrd_out.is_file()

    # 2. pbt run survrm2 & buysetest
    trial_tsv = tmp_path / "trial.tsv"
    rmst_out = tmp_path / "rmst.tsv"
    wr_out = tmp_path / "wr.tsv"
    pd.DataFrame(
        {
            "arm": [0] * 15 + [1] * 15,
            "time": np.concatenate([np.linspace(2.0, 10.0, 15), np.linspace(4.0, 14.0, 15)]),
            "status": [1, 1, 0] * 10,
        }
    ).to_csv(trial_tsv, sep="\t", index=False)
    assert pbt_cli.main(["run", "survrm2", "--input", str(trial_tsv), "--tau", "9.0", "--out", str(rmst_out)]) == 0
    assert rmst_out.is_file()
    assert (
        pbt_cli.main(
            ["run", "buysetest", "--input", str(trial_tsv), "--tte-col", "time", "--status-col", "status", "--out", str(wr_out)]
        )
        == 0
    )
    assert wr_out.is_file()

    # 3. pbt run coloc
    t1_tsv = tmp_path / "t1.tsv"
    t2_tsv = tmp_path / "t2.tsv"
    coloc_out = tmp_path / "coloc.tsv"
    pd.DataFrame({"snp": ["rs1", "rs2", "rs3"], "beta": [0.45, 0.02, -0.01], "se": [0.06, 0.07, 0.07]}).to_csv(
        t1_tsv, sep="\t", index=False
    )
    pd.DataFrame({"snp": ["rs1", "rs2", "rs3"], "beta": [0.40, -0.02, 0.03], "se": [0.06, 0.07, 0.07]}).to_csv(
        t2_tsv, sep="\t", index=False
    )
    assert pbt_cli.main(["run", "coloc", "--trait1", str(t1_tsv), "--trait2", str(t2_tsv), "--out", str(coloc_out)]) == 0
    assert coloc_out.is_file()


def test_notebooks_and_governance_files_valid():
    nb_dir = REPO_ROOT / "notebooks"
    nbs = sorted(nb_dir.glob("*.ipynb"))
    assert len(nbs) == 4
    for nb_path in nbs:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 4
        assert "colab.research.google.com" in "".join(nb["cells"][0]["source"])
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert any(len(c.get("outputs", [])) > 0 for c in code_cells)

    for required_file in (
        "CITATION.cff",
        "CONTRIBUTING.md",
        "Dockerfile",
        "environment.yml",
        "mkdocs.yml",
        ".github/workflows/docs.yml",
        "docs/R_TO_PYTHON_MIGRATION.md",
        "docs/CLI_AND_IO_GUIDE.md",
        "docs/PLOTTING_GALLERY.md",
        "docs/index.md",
    ):
        assert (REPO_ROOT / required_file).is_file(), f"Missing {required_file}"


def test_flagship_plot_helpers_via_unified_namespace():
    df_trial = pd.DataFrame(
        {
            "arm": [0] * 10 + [1] * 10,
            "time": np.concatenate([np.linspace(2.0, 8.0, 10), np.linspace(4.0, 12.0, 10)]),
            "ev": [1, 1, 0, 1, 1] * 4,
        }
    )
    rmst_res = pbt.survrm2.rmst2(df_trial["time"], df_trial["ev"], df_trial["arm"], tau=7.5)
    fig1 = rmst_res.plot()
    assert len(fig1.axes) >= 1

    bt_res = pbt.buysetest.buyse_test(
        df_trial, "arm", [pbt.buysetest.Endpoint("time", "tte", "ev")]
    )
    fig2 = bt_res.plot()
    assert len(fig2.axes) >= 1

    import matplotlib.pyplot as plt

    plt.close(fig1)
    plt.close(fig2)
