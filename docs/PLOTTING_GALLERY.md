# Publication-Ready `.plot()` Visualization Gallery

Eight flagship oncology, regulatory genomics, causal genetics, and clinical biostatistics packages in `pythonic_biotools` ship with built-in, publication-grade `matplotlib` visualizations accessible directly via `result.plot(...)` or standalone `plot_*()` functions.

## Design Conventions

- **Zero Headless Overhead (Lazy Import)**: `matplotlib` is imported lazily inside `.plot()` / `plot_*()` calls—never at package import time—so every package runs cleanly in minimal CLI, cluster, or WDL environments without `matplotlib` installed.
- **Optional `[plot]` Extra**: Install plotting support per package via `pip install "<package>[plot]"` (`matplotlib>=3.7`).
- **Standard Object-Oriented Return Type**: Every `.plot(...)` method and `plot_*()` helper uses `matplotlib`'s object-oriented API (`plt.subplots`), never calls `plt.show()`, accepts optional `ax=` (or 3-row `axes=` for `facets`) and `figsize=`, and consistently returns `matplotlib.figure.Figure`.

| Package | Result `.plot()` Method | Standalone Function(s) | Visual Summary |
| :--- | :--- | :--- | :--- |
| [`ichorcna`](../ichorcna/README.md) | `IchorCNAResult.plot(ax=None, figsize=(11, 4))` | `plot_ichorcna` | Genome-wide ULP-WGS `logR` bins colored by CNA state (`HOMD`/`HETD`/`NEUT`/`GAIN`/`AMP`/`HLAMP`), segment medians, and `TFx` / `Ploidy` banner. |
| [`facets`](../facets/README.md) | `FacetsResult.plot(figsize=(11, 7))` | `plot_facets` | Classic 3-panel stacked genome-wide profile: (1) `cnlr` + `dipLogR`, (2) het `\|valor\|`, and (3) integer `tcn`/`lcn` + cellular fraction (`cf`) shading. |
| [`tracksig`](../tracksig/README.md) | `TrackSigResult.plot(ax=None, figsize=(9, 4.5))` | `plot_tracksig` | Evolutionary timeline across decreasing CCF pseudotime bins with active SBS trajectories, bootstrap 95% CI bands, and dashed `PELT` changepoints. |
| [`rose`](../rose/README.md) | `ROSEResult.plot(ax=None, top_n_labels=5, figsize=(7, 5))` | `plot_rose_hockey_stick` | Ranked stitched-enhancer "hockey-stick" curve with geometric tangent cutoff separating Typical vs Super-Enhancers and top gene labels. |
| [`grapple`](../grapple/README.md) | `GrappleResult.plot(ax=None, figsize=(8, 4.5))` | `plot_grapple_profile` | Robust adjusted profile log-likelihood curve across $\beta$ with vertical mode markers and 95% CIs at every causal / pleiotropic pathway mode. |
| [`monalisa`](../monalisa/README.md) | `BinnedMotifEnrichmentResult.plot(...)` & `StabilitySelectionResult.plot(...)` | `plot_binned_motif_enrichment`, `plot_stability_selection` | Diverging `log2enr` / `-log10(padj)` motif enrichment heatmaps across genomic bins + Randomized Lasso stability selection probability paths. |
| [`survrm2`](../survrm2/README.md) | `RMST2Result.plot(ax=None, figsize=(7.5, 4.8))` | `plot_rmst2`, `plot_rmst1` | Zero-arg two-arm Kaplan–Meier step curves shaded over $[0, \tau]$ showing $\Delta\text{RMST}(\tau)$, RMST ratios, and optional ANCOVA contrasts. |
| [`buysetest`](../buysetest/README.md) | `BuyseTestResult.plot(ax=None, figsize=(8, 4))` | `plot_buyse_test` | Horizontal stacked `Favorable (Win)`, `Unfavorable (Loss)`, `Neutral`, and `Uninformative` bars across prioritized endpoints with `WR` & `NB` callouts. |

---

## 1. `ichorcna` — Genome-Wide cfDNA Copy-Number & Tumor Fraction (`TFx`) Profile

Plots normalized ultra-low-pass WGS bin $\log_2$ coverage ratios across cumulative genomic coordinates colored by Viterbi copy-number state (`HETD`, `NEUT`, `GAIN`, `AMP`, `HLAMP`, plus subclonal `1_sub` / `3_sub`), overlaid with horizontal segment median bars, chromosome boundaries, and a header banner reporting estimated circulating tumor DNA fraction (`TFx`) and tumor ploidy.

```python
from ichorcna import normalize_ulp_bins, plot_ichorcna, run_ichorcna

norm_bins = normalize_ulp_bins(raw_ulp_bins_df, panel_of_normals=pon_df, settings="default")
res = run_ichorcna(norm_bins, settings="default")

# Method syntax or functional syntax — both return matplotlib.figure.Figure
fig = res.plot(figsize=(11, 4))
fig.savefig("ichorcna_genome_profile.png", dpi=300)
```

---

## 2. `facets` — Classic 3-Panel Stacked Allele-Specific Copy-Number Profile

Renders the classic 3-panel genome-wide FACETS diagnostic view:
1. **Copy-number log-ratio (`cnlr`)** SNP scatter, segment medians, and purple dashed `dipLogR` baseline.
2. **Allelic log-odds-ratio (`|valor|`)** across germline heterozygous SNPs with segment $\sqrt{\text{mafR}}$ imbalance bars.
3. **Integer total (`tcn`) and minor (`lcn`) copy numbers** with clonal cellular fraction (`cf_em`) shading, annotated with `Purity`, `Ploidy`, and `WGD` status.

```python
from facets import plot_facets, run_facets

res = run_facets(pileup_df, genome="GRCh38", settings="default")

fig = res.plot(figsize=(11, 7))
fig.savefig("facets_three_panel.png", dpi=300)
```

---

## 3. `tracksig` — Evolutionary Mutational Signature Timeline & `PELT` Changepoints

Plots active mutational signature exposure trajectories from clonal (high CCF, left) to subclonal (low CCF, right) pseudotime bins, overlaying faint unpooled single-bin EM estimates, shaded 95% nonparametric bootstrap credible intervals (when `bootstrap_iters > 0`), and vertical dashed `PELT` changepoint boundaries annotated with boundary CCF and bootstrap support.

```python
from tracksig import plot_tracksig, tracksig

res = tracksig(
    mutations_df,
    signature_matrix=cosmic_sbs_df,
    purity=0.82,
    bin_size=100,
    score_method="SigFreq",
    bootstrap_iters=50,
    settings="default",
)

fig = res.plot(figsize=(9, 4.5))
fig.savefig("tracksig_evolution.png", dpi=300)
```

---

## 4. `rose` — Super-Enhancer Ranked Signal "Hockey-Stick" Curve

Plots stitched distal regulatory domains ranked by ascending ChIP-seq / ATAC-seq signal (`SPM` or control-subtracted `bigWig` density), drawing the geometric tangent slope and dashed inflection threshold separating Typical Enhancers from Super-Enhancers and labeling the top `top_n_labels` Super-Enhancers by nearest gene TSS. Accepts either a `ROSEResult` or a stitched-enhancer `pandas.DataFrame`.

```python
from rose import call_rose_super_enhancers, plot_rose_hockey_stick

res = call_rose_super_enhancers(
    peaks="H3K27ac_peaks.narrowPeak",
    tss_df=tss_df,
    stitch_bp=12_500,
    tss_exclusion_bp=2_500,
    settings="R_default",
)

fig = res.plot(top_n_labels=5, figsize=(7, 5))
# Or directly from a DataFrame:
fig_df = plot_rose_hockey_stick(res.stitched_regions, top_n_labels=5)
fig.savefig("rose_hockey_stick.png", dpi=300)
```

---

## 5. `grapple` — Multimodal Robust Adjusted Profile Log-Likelihood

Visualizes the orthogonalized Tukey/Huber profile log-likelihood $\ell_{\text{prof}}(\beta)$ across the $\beta$ grid, marking every prominence-filtered causal or pleiotropic local mode with vertical dashed lines, shaded 95% Stein-orthogonalized sandwich confidence intervals, and mode-level $p$-values.

```python
from grapple import grapple_robust_est, plot_grapple_profile

res = grapple_robust_est(
    ivs_df["bx"],
    ivs_df["sx"],
    ivs_df["by"],
    ivs_df["sy"],
    cor_xy=0.18,
    settings="R_default",
)

fig = res.plot(figsize=(8, 4.5))
fig.savefig("grapple_multimodal_profile.png", dpi=300)
```

---

## 6. `monalisa` — Binned Motif Enrichment Heatmaps & Randomized Lasso Stability Paths

`monalisa` provides two complementary publication plots:
1. **`BinnedMotifEnrichmentResult.plot(metric="log2enr", top_n=15)`**: Diverging `RdBu_r` heatmap of composition-corrected $\log_2$ motif enrichments (or sequential `YlOrRd` heatmap for `metric="-log10(padj)"`) across quantitative genomic change bins with FDR significance stars (`*`, `**`, `***`).
2. **`StabilitySelectionResult.plot(top_n=10)`**: Complementary-Pairs Randomized Lasso selection probability trajectories across regularization steps with the horizontal `cutoff` threshold line controlling `PFER`, distinguishing TF motifs (solid) from GC/CpG sequence-bias covariates (dashed).

```python
from monalisa import (
    bin_genomic_changes,
    calc_binned_motif_enrichment,
    pfm_to_pwm,
    rand_lasso_stab_sel,
    scan_sequences_pwm,
)

pwms = pfm_to_pwm(pfm_dict, settings="default")
bins = bin_genomic_changes(delta_accessibility, method="equalN", n_bins=5, min_abs_val=0.1)

# 1. Diverging log2 enrichment heatmap (or metric="-log10(padj)")
enr = calc_binned_motif_enrichment(sequences, bins=bins, pwms=pwms, settings="default")
fig_heatmap = enr.plot(metric="log2enr", top_n=15, figsize=(8, 5))
fig_heatmap.savefig("monalisa_enrichment_heatmap.png", dpi=300)

# 2. Randomized Lasso stability selection probability trajectories
scan = scan_sequences_pwm(sequences, pwms, min_score="80%")
stab = rand_lasso_stab_sel(scan, y_change=delta_accessibility, sequences=sequences, cutoff=0.8)
fig_stab = stab.plot(top_n=10, figsize=(8, 4.5))
fig_stab.savefig("monalisa_stability_paths.png", dpi=300)
```

---

## 7. `survrm2` — Two-Arm Kaplan–Meier Step Curves & Shaded $\Delta\text{RMST}(\tau)$

`rmst1` and `rmst2` store the exact Kaplan–Meier step survival curves directly inside `RMST1Result` and `RMST2Result`, so calling `res.plot()` with **zero extra arguments** plots both arms' step survival curves, shades the area $\Delta\text{RMST}(\tau) = \int_0^\tau [\hat{S}_1(t) - \hat{S}_0(t)]\,dt$ up to the truncation horizon $\tau$, draws dashed post-$\tau$ follow-up continuations, and displays unadjusted (plus optional Tian et al. IPCW ANCOVA-adjusted) $\Delta\text{RMST}$ estimates.

```python
from survrm2 import plot_rmst2, rmst2

res = rmst2(time, status, arm, tau=36.0, covariates=baseline_cov_df, settings="default")

# Zero extra arguments needed — KM step curves are stored inside `res`!
fig = res.plot(figsize=(7.5, 4.8))
fig.savefig("survrm2_km_rmst.png", dpi=300)
```

---

## 8. `buysetest` — Hierarchical Composite Endpoint Win / Loss / Neutral Breakdown

Plots a horizontal 100%-stacked bar chart ordered from highest-priority to lowest-priority clinical endpoint, decomposing all $m \times n$ patient pairs into `Favorable (Win)`, `Unfavorable (Loss)`, `Neutral`, and `Uninformative` percentages while annotating each hierarchical layer with its cumulative **Win Ratio (`WR`)** and **Net Benefit (`NB`)**.

```python
from buysetest import Endpoint, buyse_test, plot_buyse_test

endpoints = [
    Endpoint("os_time", kind="tte", status_col="os_event", operator="higher", threshold=30.0),
    Endpoint("pfs_time", kind="tte", status_col="pfs_event", operator="higher", threshold=15.0),
    Endpoint("grade3_tox", kind="binary", operator="lower"),
]

res = buyse_test(trial_df, treatment="arm", endpoints=endpoints, scoring_rule="Peron")

fig = res.plot(figsize=(8, 4))
fig.savefig("buysetest_hierarchical_bars.png", dpi=300)
```

---

## Composing Multi-Panel Figures (`ax=`)

Every single-panel `.plot()` method accepts an optional `matplotlib.axes.Axes` via `ax=`, enabling multi-method clinical or genomic dashboard figures without re-implementing layout logic:

```python
import matplotlib.pyplot as plt

fig, (ax_km, ax_gpc) = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
rmst_result.plot(ax=ax_km)
buyse_result.plot(ax=ax_gpc)
fig.savefig("clinical_trial_joint_summary.png", dpi=300)
plt.close(fig)
```
