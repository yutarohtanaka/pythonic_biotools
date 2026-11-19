import subprocess
import shutil
import math
from pathlib import Path

import pysam
import pytest

from xenofilterR import (
    classify_reads,
    count_mismatches,
    run_xenofilter,
    score_pair,
)
from xenofilterR.cli import main as cli_main


BAM_CMATCH = 0
BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5


def _make_header(ref_name="chr1", ref_len=10000):
    return pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": ref_name, "LN": ref_len}]}
    )


def _make_segment(
    header,
    qname,
    seq_len=100,
    pos=100,
    cigar=None,
    nm=0,
    is_paired=True,
    is_read1=True,
    is_unmapped=False,
    is_secondary=False,
    is_supplementary=False,
    mate_unmapped=False,
):
    seg = pysam.AlignedSegment(header)
    seg.query_name = qname
    seg.query_sequence = "A" * seq_len
    seg.query_qualities = pysam.qualitystring_to_array("I" * seq_len)

    flag = 0
    if is_paired:
        flag |= 0x1
        if is_read1:
            flag |= 0x40
        else:
            flag |= 0x80
        if mate_unmapped:
            flag |= 0x8
    if is_unmapped:
        flag |= 0x4
    if is_secondary:
        flag |= 0x100
    if is_supplementary:
        flag |= 0x800

    seg.flag = flag
    if is_unmapped:
        seg.reference_id = -1
        seg.reference_start = -1
        seg.mapping_quality = 0
        seg.cigartuples = None
    else:
        seg.reference_id = 0
        seg.reference_start = pos
        seg.mapping_quality = 60
        seg.cigartuples = cigar if cigar is not None else [(BAM_CMATCH, seq_len)]
        if nm is not None:
            seg.set_tag("NM", nm, value_type="i")

    if is_paired and not mate_unmapped and not is_unmapped:
        seg.next_reference_id = 0
        seg.next_reference_start = pos + 200 if is_read1 else max(0, pos - 200)
    else:
        seg.next_reference_id = -1
        seg.next_reference_start = -1

    return seg


def _write_bam(path: Path, header, segments):
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for seg in segments:
            bam.write(seg)


def test_count_mismatches_and_cigar_clips():
    header = _make_header()

    # Unmapped / None -> inf
    assert math.isinf(count_mismatches(None))
    unmapped = _make_segment(header, "u1", is_unmapped=True)
    assert math.isinf(count_mismatches(unmapped))

    # NM + soft clip (20S80M) + hard clip (10H)
    clipped = _make_segment(
        header,
        "c1",
        seq_len=90,
        cigar=[(BAM_CHARD_CLIP, 10), (BAM_CSOFT_CLIP, 15), (BAM_CMATCH, 75)],
        nm=2,
    )
    assert count_mismatches(clipped) == 2 + 10 + 15


def test_paired_end_classification_all_scenarios(tmp_path):
    """
    Tests:
    (a) uniquely human paired-end reads (unmapped in mouse -> 'human')
    (b) uniquely mouse paired-end reads (unmapped in human -> 'mouse')
    (c) dual-mapping reads where mouse has NM=0 + 20S soft clip vs human NM=1, 0S -> 'human'
    (d) high-mismatch human reads (> cutoff) -> 'ambiguous' (rejected)
    (e) half-mapped mouse pair (1 mate mapped NM=0, mate unmapped) vs clean human pair -> 'human'
    """
    h_header = _make_header("chr1_human")
    m_header = _make_header("chr1_mouse")

    human_bam = tmp_path / "human.bam"
    mouse_bam = tmp_path / "mouse.bam"

    human_records = [
        # (a) Uniquely human pair: NM=1 on R1, NM=1 on R2 -> avg 1.0
        _make_segment(h_header, "read_human_only", nm=1, is_read1=True, pos=100),
        _make_segment(h_header, "read_human_only", nm=1, is_read1=False, pos=300),
        # Also test explicit unmapped records in human for (b) uniquely mouse read
        _make_segment(h_header, "read_mouse_only", is_unmapped=True, is_read1=True, mate_unmapped=True),
        _make_segment(h_header, "read_mouse_only", is_unmapped=True, is_read1=False, mate_unmapped=True),
        # (c) Dual-mapping: human has NM=1, 0S on both mates -> avg 1.0
        _make_segment(h_header, "read_softclip_mouse", nm=1, is_read1=True, pos=500),
        _make_segment(h_header, "read_softclip_mouse", nm=1, is_read1=False, pos=700),
        # (d) High-mismatch human read: NM=6 on both mates -> avg 6.0 > cutoff=4
        _make_segment(h_header, "read_high_mm", nm=6, is_read1=True, pos=900),
        _make_segment(h_header, "read_high_mm", nm=6, is_read1=False, pos=1100),
        # (e) Half-mapped in mouse vs fully mapped in human (NM=2 on both -> avg 2.0)
        _make_segment(h_header, "read_half_mouse", nm=2, is_read1=True, pos=1300),
        _make_segment(h_header, "read_half_mouse", nm=2, is_read1=False, pos=1500),
        # Secondary/supplementary noise in human that must be ignored
        _make_segment(h_header, "read_human_only", nm=15, is_read1=True, pos=2000, is_secondary=True),
        _make_segment(h_header, "read_human_only", nm=15, is_read1=False, pos=2200, is_supplementary=True),
    ]

    mouse_records = [
        # (a) Unmapped in mouse (explicit unmapped records in BAM)
        _make_segment(m_header, "read_human_only", is_unmapped=True, is_read1=True, mate_unmapped=True),
        _make_segment(m_header, "read_human_only", is_unmapped=True, is_read1=False, mate_unmapped=True),
        # (b) Uniquely mouse pair: NM=0 on both mates -> avg 0.0
        _make_segment(m_header, "read_mouse_only", nm=0, is_read1=True, pos=100),
        _make_segment(m_header, "read_mouse_only", nm=0, is_read1=False, pos=300),
        # (c) Dual-mapping: mouse has NM=0 but 20S soft clip (80M20S) -> edit distance 20.0
        _make_segment(
            m_header,
            "read_softclip_mouse",
            nm=0,
            cigar=[(BAM_CMATCH, 80), (BAM_CSOFT_CLIP, 20)],
            is_read1=True,
            pos=500,
        ),
        _make_segment(
            m_header,
            "read_softclip_mouse",
            nm=0,
            cigar=[(BAM_CMATCH, 80), (BAM_CSOFT_CLIP, 20)],
            is_read1=False,
            pos=700,
        ),
        # (e) Half-mapped in mouse: R1 maps with NM=0, R2 unmapped -> penalized ((0 + 100)/2 = 50.0)
        _make_segment(m_header, "read_half_mouse", nm=0, is_read1=True, pos=1300, mate_unmapped=True),
        _make_segment(m_header, "read_half_mouse", is_unmapped=True, is_read1=False),
    ]

    _write_bam(human_bam, h_header, human_records)
    _write_bam(mouse_bam, m_header, mouse_records)

    df = classify_reads(human_bam, mouse_bam, mismatch_cutoff=4).set_index("read_name")

    assert df.loc["read_human_only", "classification"] == "human"
    assert df.loc["read_human_only", "human_mm"] == pytest.approx(1.0)
    assert math.isinf(df.loc["read_human_only", "mouse_mm"])

    assert df.loc["read_mouse_only", "classification"] == "mouse"
    assert math.isinf(df.loc["read_mouse_only", "human_mm"])
    assert df.loc["read_mouse_only", "mouse_mm"] == pytest.approx(0.0)

    assert df.loc["read_softclip_mouse", "classification"] == "human"
    assert df.loc["read_softclip_mouse", "human_mm"] == pytest.approx(1.0)
    assert df.loc["read_softclip_mouse", "mouse_mm"] == pytest.approx(20.0)

    assert df.loc["read_high_mm", "classification"] == "ambiguous"
    assert df.loc["read_high_mm", "human_mm"] == pytest.approx(6.0)

    assert df.loc["read_half_mouse", "classification"] == "human"
    assert df.loc["read_half_mouse", "human_mm"] == pytest.approx(2.0)
    assert df.loc["read_half_mouse", "mouse_mm"] == pytest.approx(50.0)


def test_single_end_bam_reads(tmp_path):
    h_header = _make_header("chr1_human")
    m_header = _make_header("chr1_mouse")

    human_bam = tmp_path / "se_human.bam"
    mouse_bam = tmp_path / "se_mouse.bam"

    _write_bam(
        human_bam,
        h_header,
        [
            _make_segment(h_header, "se_human", nm=2, is_paired=False, pos=100),
            _make_segment(h_header, "se_mouse", nm=3, is_paired=False, pos=200),
            _make_segment(h_header, "se_too_many_mm", nm=5, is_paired=False, pos=300),
        ],
    )
    _write_bam(
        mouse_bam,
        m_header,
        [
            _make_segment(m_header, "se_mouse", nm=0, is_paired=False, pos=200),
        ],
    )

    df = classify_reads(human_bam, mouse_bam, mismatch_cutoff=4).set_index("read_name")
    assert df.loc["se_human", "classification"] == "human"
    assert df.loc["se_human", "human_mm"] == pytest.approx(2.0)
    assert math.isinf(df.loc["se_human", "mouse_mm"])

    assert df.loc["se_mouse", "classification"] == "mouse"
    assert df.loc["se_too_many_mm", "classification"] == "ambiguous"


def test_run_xenofilter_end_to_end_and_cli(tmp_path, monkeypatch):
    h_header = _make_header("chr1_human")
    m_header = _make_header("chr1_mouse")

    human_bam = tmp_path / "in_human.bam"
    mouse_bam = tmp_path / "in_mouse.bam"
    output_bam = tmp_path / "filtered_human.bam"
    cli_output_bam = tmp_path / "cli_filtered_human.bam"

    _write_bam(
        human_bam,
        h_header,
        [
            # Kept human pair
            _make_segment(h_header, "keep_pair", nm=1, is_read1=True, pos=100),
            _make_segment(h_header, "keep_pair", nm=2, is_read1=False, pos=300),
            # Secondary alignment of kept pair -> should be stripped from filtered BAM
            _make_segment(h_header, "keep_pair", nm=4, is_read1=True, pos=900, is_secondary=True),
            # Mouse contaminant pair -> dropped
            _make_segment(h_header, "drop_mouse", nm=3, is_read1=True, pos=500),
            _make_segment(h_header, "drop_mouse", nm=3, is_read1=False, pos=700),
            # High-mismatch pair -> dropped
            _make_segment(h_header, "drop_high_mm", nm=7, is_read1=True, pos=1100),
            _make_segment(h_header, "drop_high_mm", nm=7, is_read1=False, pos=1300),
        ],
    )
    _write_bam(
        mouse_bam,
        m_header,
        [
            _make_segment(m_header, "drop_mouse", nm=0, is_read1=True, pos=500),
            _make_segment(m_header, "drop_mouse", nm=0, is_read1=False, pos=700),
        ],
    )

    res_df = run_xenofilter(human_bam, mouse_bam, output_bam, mismatch_cutoff=4)
    assert set(res_df[res_df["classification"] == "human"]["read_name"]) == {"keep_pair"}

    with pysam.AlignmentFile(str(output_bam), "rb") as out:
        kept_records = list(out.fetch(until_eof=True))
    assert len(kept_records) == 2
    assert {r.query_name for r in kept_records} == {"keep_pair"}
    assert all(not r.is_secondary and not r.is_supplementary for r in kept_records)

    # Verify CLI entry point works cleanly with default cutoff=4
    monkeypatch.setattr(
        "sys.argv",
        ["xenofilter", str(human_bam), str(mouse_bam), str(cli_output_bam)],
    )
    cli_main()
    with pysam.AlignmentFile(str(cli_output_bam), "rb") as out:
        cli_records = list(out.fetch(until_eof=True))
    assert len(cli_records) == 2
    assert {r.query_name for r in cli_records} == {"keep_pair"}


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_wdl_awk_classification_parity(tmp_path):
    """Verify wdl-workflows/xenofilter_bam.wdl awk logic handles S/H clips and half-mapped pairs."""
    wdl_path = Path(__file__).resolve().parents[2] / "wdl-workflows" / "xenofilter_bam.wdl"
    wdl_text = wdl_path.read_text()
    cmd_body = wdl_text.split("command <<<")[1].split(">>>")[0]

    h_header = _make_header("chr1_human")
    m_header = _make_header("chr1_mouse")
    graft_bam = tmp_path / "graft.bam"
    host_bam = tmp_path / "host.bam"

    _write_bam(
        graft_bam,
        h_header,
        [
            _make_segment(h_header, "r_graft_unique", nm=1, is_read1=True, pos=100),
            _make_segment(h_header, "r_graft_unique", nm=1, is_read1=False, pos=300),
            _make_segment(h_header, "r_softclip_host", nm=1, is_read1=True, pos=500),
            _make_segment(h_header, "r_softclip_host", nm=1, is_read1=False, pos=700),
            _make_segment(h_header, "r_half_host", nm=2, is_read1=True, pos=900),
            _make_segment(h_header, "r_half_host", nm=2, is_read1=False, pos=1100),
        ],
    )
    _write_bam(
        host_bam,
        m_header,
        [
            _make_segment(
                m_header,
                "r_softclip_host",
                nm=0,
                cigar=[(BAM_CMATCH, 80), (BAM_CSOFT_CLIP, 20)],
                is_read1=True,
                pos=500,
            ),
            _make_segment(
                m_header,
                "r_softclip_host",
                nm=0,
                cigar=[(BAM_CMATCH, 80), (BAM_CSOFT_CLIP, 20)],
                is_read1=False,
                pos=700,
            ),
            _make_segment(m_header, "r_half_host", nm=0, is_read1=True, pos=900, mate_unmapped=True),
        ],
    )

    script = (
        cmd_body.replace("~{graft_bam}", str(graft_bam))
        .replace("~{host_bam}", str(host_bam))
        .replace("~{sample_name}", "sample1")
        .replace("~{mismatch_cutoff}", "4")
        .replace("~{default='' graft_reference_fasta}", "")
        .replace("~{default='' host_reference_fasta}", "")
        .replace("~{cpu}", "1")
    )
    subprocess.run(["bash", "-c", script], cwd=tmp_path, check=True)

    csv_text = (tmp_path / "sample1_xenofilter_classification.csv").read_text().strip().splitlines()
    rows = {line.split(",")[0]: line.split(",") for line in csv_text[1:]}
    assert rows["r_graft_unique"][3] == "graft"
    assert rows["r_softclip_host"][3] == "graft"
    assert float(rows["r_softclip_host"][2]) == pytest.approx(20.0)
    assert rows["r_half_host"][3] == "graft"
    assert float(rows["r_half_host"][2]) == pytest.approx(50.0)

