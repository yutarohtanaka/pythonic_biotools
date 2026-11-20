# `ichorcna`: Ultra-Low-Pass WGS cfDNA Tumor Fraction & Copy-Number HMM

Standalone Python implementation of **ichorCNA** (Adalsteinsson et al., *Nature Communications* 2017) for estimating circulating tumor DNA (`ctDNA`) tumor fraction (`TFx` down to 2–3%), tumor ploidy, subclonal copy-number prevalence, and discrete somatic copy-number segments from ultra-low-pass whole-genome sequencing (`ULP-WGS`, ~0.1× coverage).

## Citations

> Adalsteinsson, V. A., Ha, G., Freeman, S. S., Choudhury, A. D., Stover, D. G., Parsons, H. A., Gydush, G., Reed, S. C., Rotem, D., Rhoades, J., Loginov, D., Livitz, D., Rosebrock, D., Leshchiner, I., Kim, J., Stewart, C., Rosenberg, M., Francis, J. M., Zhang, C.-Z., ... & Meyerson, M. (2017). Scalable whole-exome sequencing of cell-free DNA reveals high concordance with metastatic tumors. *Nature Communications*, 8(1), 1324. https://doi.org/10.1038/s41467-017-00965-y

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions (`normalize_ulp_bins`, `run_ichorcna`) and `IchorCNA` / `IchorCNAConfig` accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`broadinstitute/ichorCNA` v0.3.2 Parity) |
| :--- | :--- | :--- |
| **ULP Bin Normalization** | Two-stage GC (`frac=0.25`) + mappability (`frac=0.35`) LOESS, `min_map = 0.85`, 5%-trimmed mean centering, optional PoN subtraction. | `HMMcopy::correctReadCount` parity (`frac=0.20` GC LOESS, `min_map = 0.90`, median centering, median re-centering post-PoN). |
| **Normal Initialization Grid** | Unified standard + low-ctDNA grid `normal_init = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)`. | `(0.5, 0.6, 0.7, 0.8, 0.9)` standard grid, or `(0.95, 0.99, 0.995)` when `low_tf_mode=True`. |
| **Ploidy & State Space** | `ploidy_init = (2.0, 3.0)`, `max_cn = 5`, `include_homd = False`, `sc_states = (1, 3)`. | `ploidy = (2.0, 3.0)`, `maxCN = 5`, `includeHOMD = False`, `txnE = 0.9999999`, `txnStrength = 10000`. |
| **Subclonal Filter** | Rejects degenerate subclonal aliases (`frac_cna_subclonal > 0.70` or `frac_genome_subclonal > 0.50`) when valid alternatives exist. | Exact `ichorCNA` solution filter (`frac_cna_subclonal <= 0.70` and `frac_genome_subclonal <= 0.50`). |
| **Low-TFx Diploid Guard** | Automatic diploid regularization when `tumor_fraction < 0.08` and diploid `HLAMP` fraction `< 0.01`. | MAP log-posterior selection across filtered `(normal, ploidy)` grid trajectories. |

## Quickstart

```python
from ichorcna import (
    IchorCNA,
    build_hmm_states,
    emission_means,
    normalize_ulp_bins,
    run_ichorcna,
)

# 1. Two-stage GC/mappability LOESS + Panel-of-Normals normalization
norm_bins = normalize_ulp_bins(
    raw_ulp_bins_df,
    panel_of_normals=pon_df,
    centromeres_df=centromeres_df,
    settings="default",
)

# 2. Multi-start Student-t HMM + Generalized EM inference
res = run_ichorcna(norm_bins, settings="default")
print(res.summary())
print(res.segments_df[["chrom", "start", "end", "state", "copy_number", "cellular_prevalence"]])
print(res.solutions_df[["normal_init", "ploidy_init", "tumor_fraction", "ploidy", "selected"]])

# 3. Upstream broadinstitute/ichorCNA v0.3.2 parity mode (including low-TF mode)
res_r = IchorCNA(low_tf_mode=True, settings="R_default").fit(norm_bins)
print(f"TFx={res_r.tumor_fraction:.4f}, Ploidy={res_r.ploidy:.2f}")

# 4. Publication-ready genome-wide copy-number plot (optional `pip install "ichorcna[plot]"`)
fig = res.plot(figsize=(11, 4))
fig.savefig("ichorcna_profile.png", dpi=300)
```

## License

MIT (clean-room implementation from published mathematical equations in Adalsteinsson et al., *Nature Communications* 2017).
