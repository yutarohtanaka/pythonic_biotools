# xenofilterR (Python)

Python implementation of **XenofilteR** for computational deconvolution of graft (human) and host (mouse) sequencing reads in patient-derived xenograft (PDX) datasets (WES, WGS, RNA-seq, ATAC-seq, ChIP-seq).

## Citation

- Kluin, R. J. C. et al. XenofilteR: computational deconvolution of mouse and human reads in tumor xenograft sequence data. *BMC Bioinformatics* **19**, 366 (2018). https://doi.org/10.1186/s12859-018-2353-5
- Upstream R package: https://github.com/PeeperLab/XenofilteR

## Statistical & Algorithmic Overview

XenofilteR compares BAM alignments of the same library mapped independently to the graft (human) and host (mouse) reference genomes:

1. **Alignment Extraction**: Reads primary mapped alignments per query name (single-end or paired-end R1/R2 slots), discarding unmapped (`0x4`) and secondary (`0x100`) alignments.
2. **Per-Read Edit Distance**: Computes an alignment distance combining `NM` tag mismatches/indels with clipped terminal bases so local or splice-aware soft-clipped host alignments do not masquerade as zero-mismatch matches.
3. **Paired-End Scoring & Mate Penalties**: Combines R1 and R2 edit distances via the pair mean $(\text{ED}_1 + \text{ED}_2) / 2$, penalizing an unmapped mate in half-mapped pairs.
4. **Graft Classification & BAM Filtering**: Retains reads strictly closer to the human reference within the mismatch threshold and writes filtered graft alignments to an output BAM.

## Quickstart

```python
from xenofilterR import Xenofilter, classify_reads, run_xenofilter

# Object-oriented API (supports settings="default" or settings="R_default")
xf = Xenofilter(mismatch_cutoff=4, unmapped_penalty=8, settings="R_default")
summary_df = xf.run("human.bam", "mouse.bam", "filtered_human.bam")

# Functional API
df = classify_reads("human.bam", "mouse.bam", mismatch_cutoff=4, settings="default")
run_xenofilter("human.bam", "mouse.bam", "filtered_human.bam", settings="R_default")
```

Command-line usage:

```bash
xenofilter human.bam mouse.bam filtered_human.bam --cutoff 4 --settings R_default
```

## `settings="default"` vs `settings="R_default"`

- **`settings="default"` (Pythonic mode)**:
  - Computes edit distance as `NM + soft_clips("S") + hard_clips("H")` (without double-counting CIGAR insertions `"I"` already included in `NM`).
  - Excludes supplementary alignments (`0x800`) alongside secondary alignments (`0x100`).
  - Penalizes an unmapped mate in a pair by `max(read_length, 2 * mismatch_cutoff + 2)`.
  - Classifies a read/pair as `"human"` when `human_mm < mouse_mm` and `human_mm <= mismatch_cutoff`.
- **`settings="R_default"` (Exact `PeeperLab/XenofilteR` `R/private.R` parity)**:
  - Reproduces `.Count_MM`: `soft_clips("S") + insertions("I") + NM` (ignores `"H"` and adds CIGAR `"I"` op `1` on top of `NM`).
  - Uses fixed `unmapped_penalty = 8` (`Unmapped_penalty = 8` in `XenofilteR.R`).
  - Retains supplementary alignments (`0x800`), filtering only `is_unmapped` and `is_secondary` (`0x100`) per `ScanBamParam(isSecondaryAlignment = FALSE)`.
  - Uses strict `<` per-mate gating (`private.R` L184, L207, L232–238):
    - Single-end: `h_ed < mismatch_cutoff` (and `h_ed < m_ed` if mapped to mouse).
    - Paired-end graft-only (unmapped in mouse): `"human"` if `min(h_ed1, h_ed2) < mismatch_cutoff`.
    - Paired-end mapped to both human and mouse: `"human"` iff `(h_ed1 + h_ed2) / 2 < (m_ed1 + m_ed2) / 2` and every mapped human mate has `ed < mismatch_cutoff`.

## Key Outputs

- `classify_reads` / `run_xenofilter` / `Xenofilter.run`: `pandas.DataFrame` with columns `['read_name', 'human_mm', 'mouse_mm', 'classification']` (`'human'`, `'mouse'`, or `'ambiguous'`).
- Filtered graft BAM written to `output_bam`.

## Running Tests

```bash
PYTHONPATH=xenofilterR /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest xenofilterR/tests -q
```
