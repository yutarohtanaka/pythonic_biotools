# Cross-Package Integration Suite (`tests_integration/`)

End-to-end multi-library tests verifying interoperability, data handoffs, PEP 561 `py.typed` markers, and workspace bundle imports across all **70** standalone packages in `pythonic_biotools`.

## Running the Integration Suite

```bash
pytest tests_integration -W error -q
```

## Covered Multi-Tool Workflows

1. **Workspace Bundle Manager (`pythonic_biotools_cli`)**: Verifies collision-free simultaneous imports and `py.typed` markers across all 70 packages and all 6 domain bundles (`oncology`, `statgen`, `singlecell`, `epigenomics`, `proteomics`, `survival`).
2. **Somatic Evolution, HRD & Mutational Signatures**: `scarhrd` $\to$ `mutationtimer` (SNV & WGD timing) $\to$ `tracksig` (`PELT` signature trajectories) + `mmsig` (bootstrap refitting) + `chord` (`BRCA1` vs. `BRCA2` HRD classification).
3. **Liquid Biopsy & Multi-Region Clonal Phylogenetics**: `ichorcna` + `qdnaseq` (`ACE` ULP-WGS/cfDNA) + `conipher` (multi-region PhyloCCF trees).
4. **Clinical Trial & Dynamic Survival Endpoints**: `timeroc` + `survidinri` (`IDI`/`NRI`) + `survrm2` (covariate-adjusted `RMST`) + `buysetest` (Péron Win Ratio) + `jmbayes2` (joint longitudinal-survival dynamic prediction).
5. **Spatial Cell-Type DE, Isoform DTU & Multivariate Shrinkage**: `cside` (`AnnData` spatial DE) + `saturn` (`DTU`) $\to$ `mashr` (cross-condition Empirical Bayes shrinkage).
6. **Epigenomic Regulation & Quantitative Proteomics**: `dss` (`DML`/`DMR`) + `csaw` (sliding-window DB + Simes FDR) $\to$ `monalisa` (binned PWM enrichment & Randomized Lasso stability selection) + `deqms` (`median_sweeping` $\to$ `spectra_count_ebayes`).
7. **Causal Statistical Genetics & Drug-Target `cis`-MR**: `cismr` (`PC-GMM`, `MR-SPI`) + `colocproptest` + `ctwas` (group-prior `SuSiE-RSS`) + `lava` (`run_pcor`) + `hdl` + `genomicsem` + `fgwas` + `grapple`.
