# `qdnaseq`: Shallow-WGS / cfDNA Copy Number & Reference-Free `ACE` Cellularity/Ploidy

Clean-room Python implementation of **QDNAseq** (Scheinin et al., *Genome Research* 2014) bundled with **ACE** (Absolute Copy number Estimation; Poell et al., *Bioinformatics* 2019) for shallow/low-pass whole-genome sequencing (`lpWGS`) and cell-free DNA (`cfDNA`) liquid-biopsy read-depth normalization, simultaneous 2D `(GC, mappability)` bias correction, CBS segmentation, empirical-Bayes copy-number calling, and reference-free tumor cellularity/ploidy fitting.

## Citations

> Scheinin, I., Sie, D., Bengtsson, H., van de Wiel, M. A., Olshen, A. B., van Thuijl, H. F., ... & Ylstra, B. (2014). DNA copy number analysis of fresh and formalin-fixed specimens by shallow whole-genome sequencing with identification and exclusion of problematic regions in the genome assembly. *Genome Research*, 24(12), 2022–2032. https://doi.org/10.1101/gr.175141.114
>
> Poell, J. B., Mendeville, M., Sie, D., Brink, A., Brakenhoff, R. H., & Ylstra, B. (2019). ACE: absolute copy number estimation from low-coverage whole-genome sequencing data. *Bioinformatics*, 35(16), 2847–2849. https://doi.org/10.1093/bioinformatics/bty1055

## `settings` Modes (`"default"` vs `"R_default"`)

Every primary function and class (`bin_read_counts`, `apply_filters`, `estimate_correction`, `correct_counts`, `normalize_bins`, `smooth_outlier_bins`, `segment_bins`, `call_bins`, `run_ace` / `squaremodel`, and `QDNAseqPipeline`) accepts `settings: Literal["default", "R_default"] = "default"` and raises `ValueError` on unknown values:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`QDNAseq` v1.46 & `ACE` v1.28 Parity) |
| :--- | :--- | :--- |
| **BAM Binning (`bin_read_counts`)** | Primary non-duplicate alignments with `min_mapq=30`. | `QDNAseq::binReadCounts` parity: `minMapq=37`. |
| **Bin QC & 2D LOESS (`apply_filters`, `estimate_correction`)** | `mappability >= 50.0`, `|residual| <= 4.0`, blacklist & `X/Y/MT` exclusion, 3-iteration 2D `(GC, mappability)` LOESS (`cutoff=4.0`). | Identical `mappability >= 50.0`, `residual <= 4.0`, 3-iteration 2D surface fit (`maxIter=3, cutoff=4.0, span=0.65`). |
| **Segmentation (`segment_bins`)** | Anscombe variance-stabilized $\log_2(r + \frac{3}{8 c_0})$ transform (`alpha=1e-4`, `undo_sd=0.75`). | `QDNAseq::segmentBins` parity: Anscombe $\sqrt{r + \frac{3}{8 c_0}}$ transform with `undo.SD = 1.0` pruning. |
| **ACE Grid Search (`run_ace` / `squaremodel`)** | `method="RMSE"`, `penploidy=0.5`, `pencellularity=0.5`, $\rho \in [0.05, 1.00]$, $\psi \in [1.00, 6.00]$. | `ACE::squaremodel` defaults: `method="RMSE"`, `penploidy=0.0`, `pencellularity=0.0`, $\rho \in [0.05, 1.00]$, $\psi \in [1.00, 5.00]$. |

## Quickstart

```python
from qdnaseq import (
    QDNAseqPipeline,
    apply_filters,
    bin_read_counts,
    call_bins,
    correct_counts,
    estimate_correction,
    normalize_bins,
    run_ace,
    segment_bins,
    smooth_outlier_bins,
)

# Modular step-by-step lpWGS / cfDNA workflow
qdata = bin_read_counts(bins_df, counts_df=counts_df, settings="default")
qdata = apply_filters(qdata, residual=True, blacklist=True, mappability=50.0, settings="default")
qdata = estimate_correction(qdata, settings="default")
qdata = correct_counts(qdata, settings="default")
qdata = normalize_bins(qdata, settings="default")
qdata = smooth_outlier_bins(qdata, settings="default")
qdata = segment_bins(qdata, settings="default")
qdata = call_bins(qdata, settings="default")

ace_res = run_ace(qdata, settings="default")
print(ace_res.summary())
print(ace_res.to_dataframe().head())

# End-to-end pipeline in R_default parity mode
pipe_res = QDNAseqPipeline(settings="R_default").run(bins_df, counts_df=counts_df)
print(pipe_res.summary())
```

## License

MIT (clean-room implementation from published mathematical equations in Scheinin et al. 2014 and Poell et al. 2019).
