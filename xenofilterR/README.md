# XenoFilterR (Python)

Python implementation of the **XenofilteR** algorithm (Kluin et al., *BMC Bioinformatics* 2018) to deconvolve and remove host (mouse) reads from patient-derived xenograft (PDX) sequencing data (RNA-seq, WES, WGS, ATAC-seq, ChIP-seq).

## Algorithm Overview

1. **Primary Alignment Extraction**: Filters out secondary (`0x100`) and supplementary (`0x800`) BAM records and supports both **paired-end** and **single-end** libraries.
2. **Edit Distance Calculation**: Computes per-read edit distance as:
   $$\text{ED} = \text{NM} + \text{soft-clipped bases (S)} + \text{hard-clipped bases (H)}$$
   accounting for unaligned terminal bases clipped by local/splice-aware aligners.
3. **Paired-End & Unmapped Handling**:
   - **Single-end**: Uses the read's edit distance directly (`inf` if unmapped).
   - **Paired-end (both mates mapped)**: Uses the mate-averaged edit distance $(\text{ED}_1 + \text{ED}_2) / 2$.
   - **Paired-end (one mate unmapped)**: Penalizes the unmapped mate by read length before averaging, preventing half-mapped host alignments from outscoring fully mapped graft pairs.
   - **Completely unmapped**: Assigned edit distance `inf` (so uniquely human reads with $\text{ED}_{\text{human}} \le \text{cutoff}$ are retained as `"human"`).
4. **Classification**: A read/pair is classified as `"human"` if `human_mm < mouse_mm` and `human_mm <= mismatch_cutoff` (default `4`, matching Kluin et al. `MM_threshold = 4`).

## Installation

```bash
pip install ./xenofilterR
```

## Usage

### Command Line

```bash
xenofilter human.bam mouse.bam filtered_human.bam --cutoff 4
```

### Python API

```python
from xenofilterR import classify_reads, run_xenofilter

# Classify reads into a DataFrame
df = classify_reads("human.bam", "mouse.bam", mismatch_cutoff=4)

# Or run end-to-end filtering to produce a clean human BAM
run_xenofilter("human.bam", "mouse.bam", "filtered_human.bam", mismatch_cutoff=4)
```

## Original Package and Citation

- https://github.com/NKI-GCF/XenofilteR
- Kluin, R. J. C. et al. XenofilteR: computational deconvolution of mouse and human reads in tumor xenograft sequence data. *BMC Bioinformatics* **19**, 366 (2018).
