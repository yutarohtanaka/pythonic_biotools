"""End-to-end cancer evolution, HRD scar scoring, and signature timing demo."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("oncology", repo_root=REPO_ROOT)

from mmsig import make_synthetic_cosmic_reference, mm_fit_signatures
from mutationtimer import estimate_wgd_timing, mutation_time
from scarhrd import calculate_hrd
from tracksig import COSMIC_96_CONTEXTS, tracksig


def main() -> None:
    rng = np.random.default_rng(42)
    cn_segments = pd.DataFrame(
        [
            {"chrom": "chr1", "start": 1_000_000, "end": 50_000_000, "major_cn": 2, "minor_cn": 0, "total_cn": 2},
            {"chrom": "chr1", "start": 50_000_000, "end": 120_000_000, "major_cn": 2, "minor_cn": 2, "total_cn": 4},
            {"chrom": "chr2", "start": 1_000_000, "end": 40_000_000, "major_cn": 2, "minor_cn": 0, "total_cn": 2},
            {"chrom": "chr2", "start": 40_000_000, "end": 100_000_000, "major_cn": 2, "minor_cn": 2, "total_cn": 4},
        ]
    )
    hrd = calculate_hrd(cn_segments, genome="GRCh38", settings="default")
    print(f"[scarHRD] HRD-LOH={hrd.hrd_loh}, TAI={hrd.tai}, LST={hrd.lst}, HRD-sum={hrd.hrd_sum}")

    sig_ref = make_synthetic_cosmic_reference(["SBS1", "SBS3", "SBS5", "SBS13"])
    is_early = np.arange(200) < 120
    variants_df = pd.DataFrame(
        {
            "chrom": np.where(np.arange(200) < 100, "chr1", "chr2"),
            "pos": rng.integers(2_000_000, 95_000_000, size=200),
            "alt_count": rng.binomial(100, np.where(is_early, 0.45, 0.21)),
            "total_depth": np.full(200, 100),
            "context": np.where(
                is_early,
                rng.choice(COSMIC_96_CONTEXTS, size=200, p=sig_ref["SBS1"].to_numpy()),
                rng.choice(COSMIC_96_CONTEXTS, size=200, p=sig_ref["SBS3"].to_numpy()),
            ),
        }
    )

    mt = mutation_time(variants_df, cn_segments, purity=0.80, n_boot=25, settings="default")
    wgd = estimate_wgd_timing(mt.segments_df, settings="default")
    print(f"[MutationTimeR] WGD={wgd.is_wgd}, T_WGD={wgd.wgd_time:.3f}")

    ts = tracksig(variants_df, sig_ref, purity=0.80, bin_size=50, settings="default")
    print(f"[TrackSig] Active signatures={ts.active_signatures}, changepoints={ts.changepoints}")

    catalog = variants_df["context"].value_counts().reindex(sig_ref.index, fill_value=0).to_frame("TUMOR_01")
    mm = mm_fit_signatures(catalog, sig_ref, n_boot=50, settings="default")
    print("[mmsig] Refitted exposures:\n", mm.exposures.round(3))


if __name__ == "__main__":
    main()
