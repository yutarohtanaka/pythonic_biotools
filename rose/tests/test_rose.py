from __future__ import annotations
import pandas as pd
from rose import call_rose_super_enhancers

def test_rose_super_enhancer_stitching_and_threshold():
    tss_df = pd.DataFrame([{"chrom": "chr8", "tss": 127_736_000, "gene_name": "GENE_A", "gene_type": "protein_coding"}])
    peaks = [
        {"chrom": "chr8", "start": 127_735_500, "end": 127_736_500, "score": 500.0},
        {"chrom": "chr8", "start": 127_800_000, "end": 127_804_000, "score": 320.0},
        {"chrom": "chr8", "start": 127_810_000, "end": 127_815_000, "score": 410.0},
        {"chrom": "chr8", "start": 127_822_000, "end": 127_828_000, "score": 380.0},
    ] + [
        {"chrom": "chr1", "start": 1_000_000 + i * 50_000, "end": 1_000_500 + i * 50_000, "score": 10.0 + (i % 5)}
        for i in range(40)
    ]
    res = call_rose_super_enhancers(pd.DataFrame(peaks), tss_df=tss_df, stitch_bp=12_500, tss_exclusion_bp=2_500)
    assert len(res.stitched_regions) == 41
    assert res.super_enhancers.iloc[0]["n_constituent_peaks"] == 3
