# `chromvar`: Python Implementation of `chromVAR`

Standalone Python implementation of **chromVAR** (Schep et al., *Nature Methods* 2017) for computing GC- and accessibility-corrected transcription factor motif deviation $Z$-scores and variability from sparse single-cell or bulk ATAC-seq / ChIP-seq fragment count matrices, alongside strand-aware GENCODE GTF TSS extraction and log-odds Position Weight Matrix (PWM) sequence scanning.

## Installation

```bash
pip install -e ./chromvar
```

## Quickstart

```python
from chromvar import compute_deviations, compute_expectations, compute_variability, get_background_peaks

exp = compute_expectations(counts_df)
bg = get_background_peaks(counts_df, gc_content=peak_gc, n_iterations=50)
res = compute_deviations(counts_df, motif_matches_df, bg, exp)
var_df = compute_variability(res.z_scores)
```

## Reference

- Schep, A. N., Wu, B., Buenrostro, J. D., & Greenleaf, W. J. chromVAR: inferring transcription-factor-associated accessibility from single-cell epigenomic data. *Nature Methods* 14, 975–978 (2017).
