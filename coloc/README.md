# `coloc`

Log-space Bayesian two-trait genetic colocalization via Wakefield Approximate Bayes Factors (`coloc.abf`), with credible-set construction and exact upstream R parity mode.

## Citation

> Giambartolomei, C., Vukcevic, D., Schadt, E. E., Franke, L., Hingorani, A. D., Wallace, C., & Plagnol, V. (2014). Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. *PLOS Genetics*, 10(5), e1004383. https://doi.org/10.1371/journal.pgen.1004383

> Wakefield, J. (2009). Bayes factors for genome-wide association studies: comparison with $P$-values. *Genetic Epidemiology*, 33(1), 79–86. https://doi.org/10.1002/gepi.20359

## Statistical Model

Given $M$ genetic variants in a genomic locus with marginal association summary statistics $(\hat{\beta}_{j,1}, s_{j,1})$ for Trait 1 and $(\hat{\beta}_{j,2}, s_{j,2})$ for Trait 2:

1. **Wakefield Log Approximate Bayes Factors**:
   For each variant $j \in \{1, \dots, M\}$ and trait $t \in \{1, 2\}$ with prior effect standard deviation $W_t$ (`0.15` for quantitative traits `"quant"`, `0.20` for binary case-control log-odds `"cc"`, scaled by `sd_y` when provided):
   $$\log \text{ABF}_{j,t} = -\frac{1}{2}\log\!\left(1 + \frac{W_t^2}{s_{j,t}^2}\right) + \frac{1}{2} r_{j,t} z_{j,t}^2, \qquad r_{j,t} = \frac{W_t^2}{s_{j,t}^2 + W_t^2}, \qquad z_{j,t} = \frac{\hat{\beta}_{j,t}}{s_{j,t}}.$$

2. **Five Mutually Exclusive Locus Hypotheses** (single-causal-variant assumption per trait):
   - $H_0$: Neither trait has a genetic association in the locus ($\log L_0 = 0$).
   - $H_1$: Only Trait 1 has a causal variant ($\log L_1 = \log p_1 + \operatorname{logsumexp}_j(\log \text{ABF}_{j,1})$).
   - $H_2$: Only Trait 2 has a causal variant ($\log L_2 = \log p_2 + \operatorname{logsumexp}_j(\log \text{ABF}_{j,2})$).
   - $H_3$: Both traits are associated via two **distinct** causal variants $i \neq j$ ($\log L_3 = \log p_1 + \log p_2 + \log \sum_{i \neq j} \text{ABF}_{i,1}\text{ABF}_{j,2}$, evaluated in log-space without catastrophic cancellation).
   - $H_4$: Both traits share a **single** causal variant ($\log L_4 = \log p_{12} + \operatorname{logsumexp}_j(\log \text{ABF}_{j,1} + \log \text{ABF}_{j,2})$).

3. **Posterior Normalization & Credible Sets**:
   Hypothesis posteriors $\text{PP.H}k = \exp(\log L_k - \operatorname{logsumexp}_{h=0}^4 \log L_h)$ sum to $1$, and per-variant conditional colocalization probabilities $\text{SNP.PP.H4}_j \propto \text{ABF}_{j,1}\text{ABF}_{j,2}$ define the $95\%$ colocalization credible set.

## Python Quickstart

```python
import numpy as np
from coloc import coloc_abf

m = 100
se1 = np.full(m, 0.05)
se2 = np.full(m, 0.05)
beta1 = np.zeros(m)
beta2 = np.zeros(m)
beta1[42] = 0.45
beta2[42] = -0.40
snps = [f"rs{i + 1}" for i in range(m)]

# 1. Default Pythonic mode (quantitative trait1 vs case-control trait2)
res = coloc_abf(
    beta1,
    se1,
    beta2,
    se2,
    snp_ids=snps,
    dataset_type1="quant",
    dataset_type2="cc",
    settings="default",
)
print(res.summary_dict())
print(res.credible_set(0.95))

# 2. Upstream R parity mode (chr1swallace/coloc::coloc.abf)
res_r = coloc_abf(
    beta1,
    se1,
    beta2,
    se2,
    snp_ids=snps,
    dataset_type1="quant",
    dataset_type2="cc",
    settings="R_default",
)
snp_df = res_r.snp_table()
```

## `settings="default"` vs `settings="R_default"`

| Aspect | `settings="default"` | `settings="R_default"` (`chr1swallace/coloc`) |
| :--- | :--- | :--- |
| **Wakefield prior SD (`prior_w1`, `prior_w2`)** | Auto-set from `dataset_type1`/`dataset_type2` (`0.15` for `"quant"`, `0.20` for `"cc"`), overridable via `prior_w1`/`prior_w2`/`prior_w` | Identical (`0.15` for `"quant"`, `0.20` for `"cc"`, distinct `w1`/`w2` supported) |
| **Large-locus prior guard (`M * p >= 1`)** | Uses supplied `p1, p2, p12` directly; raises `ValueError` if `p1 + p2 + p12 >= 1` | Replicates `claudia.R:173-182` `adjust_prior`: when `M * p >= 1`, emits `UserWarning` and shrinks `p` to `1 / (M + 1)` |
| **Cancellation-free `H3` off-diagonal log-sum** | Exact log-space peel-off when `log_prod - sum_shared < 0.1` | Identical (avoids float64 cancellation at extreme $|Z| \ge 40$ while matching R `logdiff` at moderate $|Z|$) |

### Parity Notes

- Upstream `chr1swallace/coloc` (`R/claudia.R:173-182`) invokes `adjust_prior(p, nsnps, suffix)` independently on `p1`, `p2`, and `p12`. In dense loci (e.g., $M \ge 10{,}000$ variants with default $p_1 = 10^{-4}$), `settings="R_default"` emits a `UserWarning` (`"p1 * nsnps >= 1, setting p1=1/(nsnps + 1)"`) and rescales the offending prior to $1 / (M + 1)$ rather than failing or allowing prior odds sums to exceed unity per trait.
- Passing `dataset_type1` and `dataset_type2` (`"quant"` or `"cc"`) automatically assigns trait-specific Wakefield prior standard deviations ($W = 0.15$ vs $W = 0.20$) matching `coloc::process.dataset`.

## Output Schema

`coloc_abf(...)` returns a frozen `ColocABFResult` dataclass:

- **Scalar attributes**: `locus_name`, `trait1_name`, `trait2_name`, `n_snps`, `pp_h0`, `pp_h1`, `pp_h2`, `pp_h3`, `pp_h4`, `lead_snp`, `prior_w1`, `prior_w2`, `p1`, `p2`, `p12`, `dataset_type1`, `dataset_type2`, `settings`.
- **Array / sequence attributes**: `snp_ids` (`list[str]`), `lbf_trait1` (`np.ndarray`), `lbf_trait2` (`np.ndarray`), `snp_pp_h4` (`np.ndarray`).
- **`res.summary_dict()`**: Dictionary with keys `locus_name`, `trait1`, `trait2`, `n_snps`, `PP.H0`, `PP.H1`, `PP.H2`, `PP.H3`, `PP.H4`, `lead_snp`.
- **`res.snp_table()`**: `pd.DataFrame` sorted by descending `SNP.PP.H4` with columns `["snp", "lABF_trait1", "lABF_trait2", "lABF_shared", "SNP.PP.H4"]`.
- **`res.credible_set(coverage=0.95)`**: Smallest `list[str]` of variant IDs whose cumulative `SNP.PP.H4` meets `coverage`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/coloc /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest coloc/tests
```

## License

GPL-3.0-or-later.
