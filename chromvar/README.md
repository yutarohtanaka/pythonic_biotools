# `chromvar`: Bias-Corrected Chromatin Accessibility Motif Deviations & PWM Scanning

Standalone Python implementation (`pythonic-chromvar`) of Bioconductor **`chromVAR`** and **`motifmatchr` / `TFBSTools`** for computing GC- and accessibility-matched transcription factor motif deviation $Z$-scores and across-sample variability on dense or `scipy.sparse` raw ATAC-seq / ChIP-seq fragment count matrices, paired with strand-aware GENCODE GTF TSS utilities, exact MOODS dynamic-programming PWM $p$-value thresholding, and non-coding variant motif disruption scoring.

---

## Published Citations

- **Schep, A. N., Wu, B., Buenrostro, J. D., & Greenleaf, W. J.** (2017). chromVAR: inferring transcription-factor-associated accessibility from single-cell epigenomic data. *Nature Methods*, 14(10), 975–978. https://doi.org/10.1038/nmeth.4401
- **Korhonen, J., Martinmäki, P., Pizzi, C., Rastas, P., & Ukkonen, E.** (2009). MOODS: fast search for position weight matrix matches in DNA sequences. *Bioinformatics*, 25(23), 3181–3182. https://doi.org/10.1093/bioinformatics/btp554
- **Nishida, K., Frith, M. C., & Nakai, K.** (2009). Pseudocounts for transcription factor binding sites. *Nucleic Acids Research*, 37(3), 939–944. https://doi.org/10.1093/nar/gkn1019
- **Tan, G., & Lenhard, B.** (2016). TFBSTools: an R/bioconductor package for transcription factor binding site analysis. *Bioinformatics*, 32(10), 1555–1556. https://doi.org/10.1093/bioinformatics/btw024

---

## Statistical Formulation

Given a raw non-negative integer fragment count matrix $X \in \mathbb{Z}_{\ge 0}^{P \times S}$ (`peaks x samples`) and binary/count motif match matrix $M \in \mathbb{R}_{\ge 0}^{P \times K}$ (`peaks x motifs`):

1. **Expected Peak Fractions & Sample Depths (`compute_expectations`)**:
   $$e_i = \frac{\sum_{j=1}^S X_{ij}}{\sum_{p=1}^P \sum_{j=1}^S X_{pj}}, \qquad N_j = \sum_{i=1}^P X_{ij}$$

2. **GC- & Accessibility-Matched Background Peaks (`get_background_peaks`)**:
   Transforms $(\log_{10}(\sum_j X_{ij}), \text{GC}_i)$ via lower Cholesky whitening $L^{-1} f_i$, discretizes onto a 2D `bs x bs` grid smoothed with isotropic Gaussian kernel $K_{ab} = \exp(-\|g_a - g_b\|_2^2 / (2 w^2))$, and samples $B = \text{n\_iterations}$ background peak permutations matched in smoothed Mahalanobis density.

3. **Bias-Corrected Motif Deviations & Variability (`compute_deviations`, `compute_variability`)**:
   For observed fragments $O_{mj} = (M^\top X)_{mj}$ and expected fragments $E_{mj} = (M^\top e)_m N_j$:
   $$d_{mj} = \frac{O_{mj} - E_{mj}}{E_{mj}}, \qquad \tilde{d}_{mj} = d_{mj} - \bar{d}^{\text{bg}}_{mj}, \qquad z_{mj} = \frac{d_{mj} - \bar{d}^{\text{bg}}_{mj}}{\text{SD}_b(d^{\text{bg}}_{mjb})}$$
   Across-sample variability $v_m = \text{SD}_j(z_{mj})$ is tested against $H_0: v_m = 1$ via $(S_{\text{eff}} - 1) v_m^2 \sim \chi^2_{S_{\text{eff}} - 1}$ with bootstrap 95% confidence bounds and Benjamini–Hochberg $q$-values.

4. **Log-Odds PWMs & Exact 1D DP Null Thresholding (`build_log_odds_pwm`, `pwm_threshold_from_pvalue`, `scan_sequence_pwm`)**:
   Computes the exact single-strand null score distribution of $S = \sum_{k=1}^K W_{k, B_k}$ under $B_k \stackrel{\text{iid}}{\sim} \mathbf{bg}$ by 1D convolution dynamic programming (`MOODS::threshold_from_p`) to determine the exact cutoff $T$ satisfying $P_{\text{bg}}(S \ge T) \le p_{\text{cutoff}}$.

---

## Quickstart

```python
import numpy as np
import pandas as pd
from chromvar import (
    build_log_odds_pwm,
    compute_deviations,
    compute_expectations,
    compute_variability,
    get_background_peaks,
    scan_sequence_pwm,
)

rng = np.random.default_rng(42)
counts_df = pd.DataFrame(rng.poisson(8, size=(200, 24)) + 1)
peak_gc = rng.uniform(0.35, 0.65, size=200)
motif_matches = pd.DataFrame(rng.binomial(1, 0.20, size=(200, 5)), columns=[f"TF_{k}" for k in range(5)])

exp = compute_expectations(counts_df)
bg = get_background_peaks(counts_df, gc_content=peak_gc, n_iterations=50, settings="R_default")
res = compute_deviations(counts_df, motif_matches, bg, exp, settings="R_default")
var_df = compute_variability(res.z_scores, settings="R_default")

# Raw JASPAR count matrix -> TFBSTools PWM + motifmatchr/MOODS p_cutoff=5e-5 scan
pfm_counts = rng.multinomial(100, [0.25, 0.25, 0.25, 0.25], size=10).astype(float)
pwm = build_log_odds_pwm(pfm_counts, settings="R_default")
hits = scan_sequence_pwm("ACGT" * 50, pwm, settings="R_default")
```

---

## `settings="default"` vs `settings="R_default"` Reference

`get_background_peaks`, `compute_deviations`, `compute_variability`, `build_log_odds_pwm`, and `scan_sequence_pwm` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Function | `settings="default"` | `settings="R_default"` (`GreenleafLab/chromVAR`, `TFBSTools`, `motifmatchr`) |
|---|---|---|
| **`get_background_peaks`** | Clamps zero-count peaks via `log10(max(sum, 1))`; `min(bs, max(5, P//4))` histogram bin midpoints | Raises `ValueError` if any peak has `0` fragments; exact `bs x bs` (`50 x 50`) grid on `[min, max]` with nearest-grid-point assignment |
| **`compute_deviations`** | Returns finite `0.0`-centered scores for all positive sample depths | Masks `deviations` and `z_scores` to `NaN` wherever $E_{mj} = E_{\text{motif}} \cdot N_j < 1.0$ (`threshold = 1` in `compute_deviations.R:342-352`) |
| **`compute_variability`** | Default `n_boot = 200` | Default `n_boot = 1000` |
| **`build_log_odds_pwm`** | Scale-invariant row probability normalization + `pseudocount = 0.01` | Raw-count `pseudocount = 0.8` weighted by `bg`: $(C + 0.8 \cdot \mathbf{bg}) / (N + 0.8)$ (`TFBSTools::toPWM`) |
| **`scan_sequence_pwm`** | Fixed `min_log2_odds = 6.0` cutoff with `dedup_window_bp = 8` NMS deduplication | Exact DP background score-distribution cutoff at `p_cutoff = 5e-5` (`MOODS`) with `dedup_window_bp = 0` (no NMS suppression) |

---

## Public API & Output Schema

- **`ChromVARResult`** (`compute_deviations` output):
  - `deviations`: `pd.DataFrame` (`motifs x samples`) bias-corrected deviations $\tilde{d}_{mj}$.
  - `z_scores`: `pd.DataFrame` (`motifs x samples`) deviation $Z$-scores $z_{mj}$.
- **`compute_variability` DataFrame Columns**:
  `variability`, `bootstrap_lower_bound`, `bootstrap_upper_bound`, `p_value`, `p_value_adj`.
- **`PWMMotifHit`** (`scan_sequence_pwm` item):
  `sequence_id`, `offset_5prime`, `tss_relative_coord`, `matched_kmer`, `strand`, `log2_odds_score`.
- **Additional Motif & GTF Utilities**:
  `pwm_threshold_from_pvalue`, `score_kmer_strand_max`, `compute_variant_pwm_disruption`, `extract_gtf_tss`, `signed_tss_distance`, `nearest_tss`, `reverse_complement`, `run_raw_count_pychromvar`.

---

## Testing

```bash
PYTHONPATH=chromvar /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest chromvar/tests -v
```

## License

MIT.
