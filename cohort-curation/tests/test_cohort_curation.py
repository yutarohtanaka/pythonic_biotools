from __future__ import annotations

import gzip
import io
import tarfile

import numpy as np
import pandas as pd
import pytest
import requests

from cohort_curation import (
    binarize_arm_events,
    build_genie_panel_matrix,
    cross_check_gistic_panel,
    download_gistic_peak_files,
    fetch_clinical_attribute,
    fetch_discrete_cna,
    harmonize_hgnc_symbols,
    mask_unprofiled_panel_genes,
    parse_cellosaurus,
    parse_gistic_peak_genes,
    repair_excel_mangled_genes,
)
from cohort_curation.cbioportal import _request_with_retry
from cohort_curation.gistic_and_panels import _safe_extract_tar_member


def test_public_cohort_parsers(tmp_path):
    p1 = tmp_path / "data_gene_panel_PANEL1.txt"
    p1.write_text("cancer_study_identifier: genie\nstable_id: PANEL1\ngene_list:\tTP53\tKRAS\n")
    p2 = tmp_path / "data_gene_panel_PANEL2.txt"
    p2.write_text("cancer_study_identifier: genie\nstable_id: PANEL2\ngene_list:\tTP53\tKRAS\tLZTR1\n")

    cov = build_genie_panel_matrix(tmp_path)
    alt_df = pd.DataFrame({"TP53": [1, 0], "KRAS": [0, 1], "LZTR1": [0, 0]}, index=["S1", "S2"])
    masked = mask_unprofiled_panel_genes(alt_df, {"S1": "PANEL1", "S2": "PANEL2"}, cov)
    assert np.isnan(masked.loc["S1", "LZTR1"]) and masked.loc["S2", "LZTR1"] == 0.0

    assert repair_excel_mangled_genes(["1-Mar", "7-Sep", "1-Dec", "15-Sep"]) == [
        "MARCHF1",
        "SEPTIN7",
        "DELEC1",
        "SELENOF",
    ]

    hgnc_df = pd.DataFrame(
        [
            {
                "symbol": "KMT2D",
                "status": "Approved",
                "prev_symbol": "MLL2",
                "alias_symbol": "ALR",
                "entrez_id": "8085",
            },
            {
                "symbol": "C1orf112",
                "status": "Approved",
                "prev_symbol": np.nan,
                "alias_symbol": "FLJ10706",
                "entrez_id": "55732",
            },
        ]
    )
    assert list(harmonize_hgnc_symbols(pd.Series(["MLL2", "ALR"]), hgnc_df)) == ["KMT2D", "KMT2D"]

    # Integer/float entrez_ids, NaN/None preservation, and case-insensitive fallback
    syms = pd.Series(["UNKNOWN_PROBE", np.nan, None, "mll2", "c1orf112"], index=[0, 1, 2, 3, 4], dtype=object)
    eids = pd.Series([8085.0, np.nan, np.nan, np.nan, np.nan], index=[0, 1, 2, 3, 4])
    harm = harmonize_hgnc_symbols(syms, hgnc_df, entrez_ids=eids)
    assert harm.loc[0] == "KMT2D"
    assert pd.isna(harm.loc[1]) and harm.loc[1] != "nan"
    assert harm.loc[2] is None
    assert harm.loc[3] == "KMT2D"
    assert harm.loc[4] == "C1orf112"

    # Cellosaurus header + footer skipping, SY synonym matching, and .gz streaming
    cello_content = (
        "-----------------------------------------------------------------------\n"
        "Cellosaurus release header\n"
        "-----------------------------------------------------------------------\n"
        "ID   HCT 116\n"
        "AC   CVCL_0291\n"
        "SY   HCT-116; HCT116\n"
        "DR   DepMap; ACH-000971\n"
        "CC   Doubling time: 20 hours.\n"
        "//\n"
        "-----------------------------------------------------------------------\n"
        "Copyrighted by the SIB Swiss Institute of Bioinformatics\n"
        "-----------------------------------------------------------------------\n"
    )
    cello_gz = tmp_path / "cellosaurus.txt.gz"
    with gzip.open(cello_gz, "wt", encoding="utf-8") as gz_fh:
        gz_fh.write(cello_content)

    # Unfiltered parse must skip header and trailing copyright footer (exactly 1 cell line, no None row)
    qc_all = parse_cellosaurus(cello_gz)
    assert len(qc_all) == 1
    assert qc_all.iloc[0]["cell_line_name"] == "HCT 116"
    assert qc_all.iloc[0]["doubling_time_hours"] == 20.0

    # Match via SY synonym "HCT116" (even though primary ID is "HCT 116")
    qc_syn = parse_cellosaurus(cello_gz, cell_line_names={"hct116"})
    assert len(qc_syn) == 1 and qc_syn.iloc[0]["cvcl_accession"] == "CVCL_0291"

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


def test_mocked_cbioportal_client_diploid_zero_batching_and_retries(monkeypatch):
    class _FakeResp:
        def __init__(self, status_code: int, payload: object, headers: dict[str, str] | None = None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

        def json(self):
            return self._payload

    # 1. Test 429 Retry-After + Timeout exponential backoff in _request_with_retry
    sleeps: list[float] = []
    monkeypatch.setattr("cohort_curation.cbioportal.time.sleep", lambda s: sleeps.append(float(s)))

    seq_events = [
        requests.exceptions.Timeout("read timed out"),
        _FakeResp(429, {}, headers={"Retry-After": "1.25"}),
        _FakeResp(503, {}),
        _FakeResp(200, {"ok": True}),
    ]

    class _RetrySession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method: str, url: str, **kwargs):
            ev = seq_events.pop(0)
            if isinstance(ev, Exception):
                raise ev
            return ev

    res_retry = _request_with_retry(
        "GET", "/ping", max_retries=4, backoff_seconds=2.0, session=_RetrySession()  # type: ignore[arg-type]
    )
    assert res_retry == {"ok": True}
    # Attempt 1 Timeout -> 2.0 * 2^0 = 2.0; Attempt 2 HTTP 429 -> Retry-After 1.25; Attempt 3 HTTP 503 -> 2.0 * 2^2 = 8.0
    assert sleeps == pytest.approx([2.0, 1.25, 8.0])

    # 2. Test fetch_discrete_cna preserving diploid 0s across unaltered samples & genes + chunk_size batching
    gene_db = {"MYC": 4609, "TP53": 7157, "CDKN2A": 1029, "KRAS": 3845, "EGFR": 1956}
    calls: list[tuple[str, str, object]] = []

    class _MockCBioSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method: str, url: str, **kwargs):
            body = kwargs.get("json")
            calls.append((method, url, body))
            if url.endswith("/sample-lists/luad_cna/sample-ids"):
                return _FakeResp(200, ["S1", "S2", "S3"])  # S3 has zero CNA alterations!
            if url.endswith("/genes/fetch"):
                return _FakeResp(
                    200,
                    [{"hugoGeneSymbol": sym, "entrezGeneId": gene_db[sym]} for sym in body if sym in gene_db],
                )
            if url.endswith("/molecular-profiles/luad_gistic/molecular-data/fetch"):
                # Sparse cBioPortal response: only non-zero alterations returned; CDKN2A, EGFR, and sample S3 omitted!
                sparse_rows = [
                    {"sampleId": "S1", "patientId": "P1", "entrezGeneId": 4609, "value": 2},
                    {"sampleId": "S2", "patientId": "P2", "entrezGeneId": 7157, "value": -1},
                    {"sampleId": "S1", "patientId": "P1", "entrezGeneId": 3845, "value": 1},
                ]
                filt = [r for r in sparse_rows if r["entrezGeneId"] in set(body["entrezGeneIds"])]
                return _FakeResp(200, filt)
            if url.endswith("/studies/luad/clinical-data"):
                return _FakeResp(200, [{"patientId": "P1", "value": "Stage I"}, {"patientId": "P2", "value": "Stage III"}])
            if url.endswith("/studies/luad/samples"):
                return _FakeResp(200, [{"sampleId": "S1", "patientId": "P1"}, {"sampleId": "S2", "patientId": "P2"}])
            raise AssertionError(f"Unexpected URL: {url}")

    sess = _MockCBioSession()
    cna_df = fetch_discrete_cna(
        cna_profile_id="luad_gistic",
        sample_list_id="luad_cna",
        hugo_symbols=["MYC", "TP53", "CDKN2A", "KRAS", "EGFR"],
        chunk_size=2,
        session=sess,  # type: ignore[arg-type]
    )
    assert list(cna_df.index) == ["S1", "S2", "S3"]
    assert list(cna_df.columns) == ["MYC", "TP53", "CDKN2A", "KRAS", "EGFR"]
    assert cna_df.loc["S1", "MYC"] == 2
    assert cna_df.loc["S2", "TP53"] == -1
    # Diploid unaltered sample S3 and unaltered genes CDKN2A / EGFR must be 0, NOT NaN or dropped!
    assert (cna_df.loc["S3"] == 0).all()
    assert (cna_df["CDKN2A"] == 0).all() and (cna_df["EGFR"] == 0).all()
    # Verify chunk_size=2 produced 3 batch calls to /genes/fetch and 3 batch calls to /molecular-data/fetch
    assert sum(1 for _, u, _ in calls if u.endswith("/genes/fetch")) == 3
    assert sum(1 for _, u, _ in calls if u.endswith("/molecular-data/fetch")) == 3

    clin = fetch_clinical_attribute("luad", "STAGE", clinical_data_type="PATIENT", session=sess)  # type: ignore[arg-type]
    assert clin.loc["S1"] == "Stage I" and clin.loc["S2"] == "Stage III"


def test_download_gistic_peak_files_tumor_type_and_safe_tar(tmp_path, monkeypatch):
    archive_dir = "gdac.broadinstitute.org_SKCM-TM.CopyNumber_Gistic2.Level_4.2016012800.0.0"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in ("amp_genes.conf_99.txt", "del_genes.conf_99.txt"):
            content = b"cytoband\t9p21.3\nq\t1e-10\nrq\t1e-9\nwp\tchr9:1-2\ngenes\tCDKN2A\n"
            ti = tarfile.TarInfo(name=f"{archive_dir}/{fname}")
            ti.size = len(content)
            tar.addfile(ti, io.BytesIO(content))

    requested_urls: list[str] = []

    class _TarResp:
        content = buf.getvalue()

        def raise_for_status(self):
            return None

    def _fake_get(url: str, timeout: int = 120):
        requested_urls.append(url)
        return _TarResp()

    monkeypatch.setattr("cohort_curation.gistic_and_panels.requests.get", _fake_get)
    amp_p, del_p = download_gistic_peak_files("SKCM", cache_dir=tmp_path, tumor_type_code="TM")
    assert "SKCM-TM" in requested_urls[0]
    assert amp_p.exists() and del_p.exists()
    assert "CDKN2A" in parse_gistic_peak_genes(del_p)["9p21.3"]

    # Path traversal member must be rejected
    evil_buf = io.BytesIO()
    with tarfile.open(fileobj=evil_buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo(name="../escape.txt")
        ti.size = 4
        tar.addfile(ti, io.BytesIO(b"nope"))
    with tarfile.open(fileobj=io.BytesIO(evil_buf.getvalue()), mode="r:gz") as tar:
        with pytest.raises(ValueError, match="Unsafe path traversal"):
            _safe_extract_tar_member(tar, "../escape.txt", tmp_path)


def test_r_default_settings(tmp_path):
    from cohort_curation import EXCEL_MANGLED_GENE_MAP

    # 1. Verify expanded EXCEL_MANGLED_GENE_MAP entries
    assert EXCEL_MANGLED_GENE_MAP["2-APR"] == "PMAIP1"
    assert EXCEL_MANGLED_GENE_MAP["OCT4"] == "POU5F1"
    assert EXCEL_MANGLED_GENE_MAP["MARC1"] == "MTARC1"
    assert EXCEL_MANGLED_GENE_MAP["MARC2"] == "MTARC2"
    assert repair_excel_mangled_genes(["2-Apr", "Oct4", "MARC1", "MARC2"]) == [
        "PMAIP1",
        "POU5F1",
        "MTARC1",
        "MTARC2",
    ]

    hgnc_df = pd.DataFrame(
        [
            {
                "symbol": "KMT2D",
                "status": "Approved",
                "prev_symbol": "MLL2",
                "alias_symbol": "ALR",
                "entrez_id": "8085",
            },
            {
                "symbol": "TP53",
                "status": "Approved",
                "prev_symbol": np.nan,
                "alias_symbol": "P53",
                "entrez_id": "7157",
            },
        ]
    )

    p_entrez = tmp_path / "data_gene_panel_ENTREZ1.txt"
    p_entrez.write_text("stable_id: ENTREZ1\nentrez_gene_list:\t7157\t8085.0\n")
    p_legacy = tmp_path / "data_gene_panel_LEGACY1.txt"
    p_legacy.write_text("stable_id: LEGACY1\ngene_list: MLL2 MARC1 2-Apr OCT4\n")

    # 2. Invalid settings raises ValueError
    with pytest.raises(ValueError, match="settings"):
        harmonize_hgnc_symbols(pd.Series(["MLL2"]), hgnc_df, settings="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        build_genie_panel_matrix(tmp_path, settings="invalid")  # type: ignore[arg-type]

    # 3. harmonize_hgnc_symbols: default keeps unmapped symbols, R_default sets unmapped to NaN (HGNChelper)
    s_in = pd.Series(["MLL2", "2-Apr", "OCT4", "UNKNOWN_LOCUS"])
    h_def = harmonize_hgnc_symbols(s_in, hgnc_df, settings="default")
    h_r = harmonize_hgnc_symbols(s_in, hgnc_df, settings="R_default")
    assert list(h_def) == ["KMT2D", "PMAIP1", "POU5F1", "UNKNOWN_LOCUS"]
    assert list(h_r.iloc[:3]) == ["KMT2D", "PMAIP1", "POU5F1"]
    assert pd.isna(h_r.iloc[3])

    # 4. build_genie_panel_matrix: Entrez-ID gene lists + R_default gnomeR symbol canonicalization
    cov_raw_entrez = build_genie_panel_matrix([p_entrez], settings="default")
    assert list(cov_raw_entrez.columns) == ["7157", "8085"]

    cov_mapped = build_genie_panel_matrix(tmp_path, entrez_to_symbol=hgnc_df, settings="R_default")
    assert set(cov_mapped.columns) == {"TP53", "KMT2D", "MTARC1", "PMAIP1", "POU5F1"}
    assert cov_mapped.loc["ENTREZ1", "TP53"] == 1 and cov_mapped.loc["ENTREZ1", "KMT2D"] == 1
    assert cov_mapped.loc["LEGACY1", "MTARC1"] == 1 and cov_mapped.loc["LEGACY1", "PMAIP1"] == 1
    assert cov_mapped.loc["LEGACY1", "POU5F1"] == 1 and cov_mapped.loc["LEGACY1", "KMT2D"] == 1

