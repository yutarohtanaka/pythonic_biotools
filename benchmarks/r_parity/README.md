# Upstream `settings="R_default"` Golden Parity Benchmarks

This directory contains frozen analytical and upstream R / C++ reference fixtures (`golden_r_fixtures.json`) validating exact `settings="R_default"` numerical and behavioral replication across `pythonic_biotools`.

## Running the Suite

```bash
pytest benchmarks/r_parity -W error -q
```

## Covered Upstream Parity Invariants

- **`xenofilterR` (`PeeperLab/XenofilteR` `private.R`)**: `.Count_MM = NM + S + I` edit distance (excluding `H`) and fixed `Unmapped_penalty = 8`.
- **`coloc` (`chr1swallace/coloc` v6.0.3)**: Case-control default `W_sd = 0.20` (`W = 0.04`) Wakefield log-ABF posterior `PP.H4`.
- **`mutationtimer` (`gerstung-lab/MutationTimeR`)**: Exact closed-form clonal gain and WGD timing equations (`piToTime` on `2:1`, `2:0`, and `2:2` states).
- **`smr-heidi` (`JianYang-Lab/smr` v1.4.3)**: Exact Wald ratio $\hat{b}_{xy}$ and delta-method $T_{\text{SMR}} \sim \chi^2_1$.
- **`survrm2` (`uno1lab/survRM2` v1.0-4)**: Exact Kaplan–Meier Restricted Mean Survival Time (`RMST`) integrals, differences, and ratios.
- **`buysetest` (`bozenne/BuyseTest` v3.3.9)**: Exact Péron pairwise Win Ratio, Net Benefit, and 2nd-order H-decomposition variance toggle.
- **`survidinri` (`cran/survIDINRI` v1.1-2)**: Exact Uno et al. `[-1, 1]` half-scale continuous NRI (`M2 = 0.5 * cNRI`) vs. conventional `[-2, 2]` full cNRI.
- **`fgwas` (`joepickrell/fgwas` v0.3.6)**: Exact Wakefield `ln(ABF)` at `-fine` default `V = 0.1`.
- **`deqms` (`yafeng/DEqMS` v1.22)**: Exact 11-column Bioconductor `outputResult` schema and PSM-count-moderated Empirical Bayes ranking.
- **`monalisa` (`fmicompbio/monaLisa` v1.12)**: Exact `pseudocount.log2enr = 8.0` shrunk enrichment behavior vs. `1.0`.
