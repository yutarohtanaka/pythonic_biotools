# `mutationtimer`: Evolutionary Timing of Somatic Mutations, Copy-Number Gains & WGD

Standalone Python implementation of **MutationTimeR** (Gerstung et al., *Nature* 2020) for estimating somatic mutation multiplicities, classifying variants into the four PCAWG evolutionary timing states (`"clonal [early]"`, `"clonal [late]"`, `"clonal [NA]"`, `"subclonal"`), timing allele-specific copy-number gains in molecular time $T \in [0, 1]$, and testing synchronous Whole-Genome Doubling (`WGD`).

## Citations

> Gerstung, M., Jolly, C., Leshchiner, I., Dentro, S. C., Gonzalez, S., Rosebrock, D., Mitchell, T. J., Rubanova, Y., Anur, P., Yu, K., Tarabichi, M., Deshwar, A., Wintersinger, J., Kleinheinz, K., Vázquez-García, I., Haase, K., Jerman, L., Sengupta, S., Macintyre, G., ... & Van Loo, P. (2020). The evolutionary history of 2,658 cancers. *Nature*, 578(7793), 122–128. https://doi.org/10.1038/s41586-019-1907-7

## `settings` Modes (`"default"` vs `"R_default"`)

All primary pipeline functions (`mutation_time`, `pi_to_time`, `time_cn_gains`, `estimate_wgd_timing`) and `MutationTimeR` / `MutationTimerConfig` accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`gerstung-lab/MutationTimeR` Parity) |
| :--- | :--- | :--- |
| **Beta-Binomial Dispersion** | Overdispersion `rho_bb = 0.01` across somatic read counts. | Overdispersion `rho = 0.01` matching `MutationTimeR::mutationTime`. |
| **Segment Mixture Prior** | Anti-collapse Laplace-smoothed Dirichlet prior (`alpha = 1.05`, $+0.05$ pseudocounts) stabilizing low-mutation segments. | Weak sparse Dirichlet prior (`alpha = 1 / K`) allowing empty multiplicity components to collapse to zero. |
| **Mutation Copy Number (`mut_cn`)** | Empirical-Bayes depth-weighted shrinkage between VAF-inverted `mut_cn_raw` and posterior mean `mut_cn_posterior`. | Raw VAF-inverted mutation copy number $\hat{f} [\rho C_{\text{tot}} + C_{\text{norm}}(1-\rho)] / \rho$. |
| **Timing Classification** | `p_clonal >= 0.5` clonal gate + conditional `p_early / p_clonal >= 0.5` on gained segments (`major_cn >= 2`). | Exact `0.5` posterior gates (`p_clonal >= 0.5`, `p_early >= 0.5` early, `p_late >= 0.5` late, else `"clonal [NA]"`). |
| **Gain Timing (`pi_to_time`)** | Exact lineage-conservation rate equations across `2:1`, `2:0`, `2:2`, `3:1`, `3:0`, `4:0`, and general polyploid states. | Exact `piToTime` closed-form formulas (`2:1 -> 3*pi2/(2*pi2+pi1)`, `2:0` & `2:2 -> 2*pi2/(2*pi2+pi1)`). |
| **Subclonal `clusters_df`** | Accepts `ccf` directly in $(0, 1)$ (or `cellular_prevalence`). | Treats `proportion` column as cellularity $\rho \cdot \phi_k$ matching R `MutationTimeR` cluster tables. |

## Quickstart

```python
import pandas as pd
from mutationtimer import (
    MutationTimeR,
    mutation_time,
    pi_to_time,
    time_cn_gains,
    estimate_wgd_timing,
)

# 1. End-to-end somatic mutation & copy-number gain timing
res = mutation_time(
    variants_df,
    cn_segments_df,
    purity=0.75,
    clusters_df=pd.DataFrame({"ccf": [0.40]}),
    gender="male",
    n_boot=200,
    settings="default",
)
print(res.summary())
print(res.variants_df[["chrom", "pos", "vaf", "mut_cn", "timing_class"]].head())
print(res.segments_df[["chrom", "major_cn", "minor_cn", "T", "T_lo", "T_hi"]])

# 2. Exact Gerstung-lab MutationTimeR parity mode & standalone rate equations
res_r = MutationTimeR(purity=0.75, gender="male", settings="R_default").fit(
    variants_df, cn_segments_df
)
t_gain = pi_to_time(major_cn=2, minor_cn=1, pi_vec=[0.60, 0.40], settings="R_default")
wgd = estimate_wgd_timing(res_r.segments_df, settings="R_default")
print(f"2:1 gain T={t_gain:.3f}, WGD={wgd.is_wgd} (T_WGD={wgd.wgd_time:.3f})")
```

## License

MIT (clean-room implementation from published mathematical equations in Gerstung et al., *Nature* 2020).
