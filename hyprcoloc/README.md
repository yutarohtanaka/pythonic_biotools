# `hyprcoloc`

Deterministic $O(M Q)$ Bayesian multi-trait genetic colocalization (**HyPrColoc**: *Hypothesis Prioritisation for multi-trait Colocalization*) in Python, with sample-overlap modeling, sensitivity analysis, and exact upstream R parity mode.

## Citation

> Foley, C. N., Staley, J. R., Breen, P. G., Sun, B. B., Kirk, P. D. W., Burgess, S., & Howson, J. M. M. (2021). A fast and efficient colocalization algorithm for identifying shared genetic risk factors across multiple traits. *Nature Communications*, 12(1), 764. https://doi.org/10.1038/s41467-020-20885-8

## Statistical Model

For $M$ genetic variants across $Q \ge 2$ traits with marginal GWAS summary statistics $(\hat{\beta}_{jq}, s_{jq})$:

1. **Wakefield Log Approximate Bayes Factors (`log_abf`)**:
   $$\log \text{BF}_{jq} = \frac{1}{2}\log(1 - r_{jq}) + \frac{1}{2} r_{jq} z_{jq}^2, \qquad r_{jq} = \frac{W_q^2}{s_{jq}^2 + W_q^2}, \qquad z_{jq} = \frac{\hat{\beta}_{jq}}{s_{jq}},$$
   with $W_q = 0.15$ for quantitative traits (`binary_outcomes=False`) and $W_q = 0.20$ for binary case-control traits (`binary_outcomes=True`).

2. **Regional & Alignment Posterior Factorization**:
   With single-trait prior $p_1$ (`1e-4`) and conditional colocalization step parameter $\gamma = 1 - p_c$ (`p_c = 0.02`), configuration priors follow $\pi_k = p_1 \prod_{i=1}^{k-1}(1 - \gamma^i)$. For any candidate subset $S$ of $m = |S| \ge 2$ traits, HyPrColoc factors the joint colocalization posterior in log-space as:
   $$\text{PP}_{\text{coloc}}(S) = P_R(S) \cdot P_A(S),$$
   where $P_R(S)$ is the regional association probability and $P_A(S)$ is the single-variant alignment probability.

3. **Divisive Branch-and-Bound Clustering (`hyprcoloc`)**:
   Iteratively partitions the $Q$ traits into disjoint colocalized clusters satisfying $P_R > P_R^*$ (`reg_thresh`) and $P_A > P_A^*$ (`align_thresh`) plus uncolocalized traits, reporting the candidate causal variant, proportion of posterior explained, and $95\%$ credible set per cluster.

4. **Sample Overlap & Sensitivity Analysis**:
   Supports correlated estimation errors (`trait_cor`) via multivariate Wakefield joint Bayes factors, LD-aware alignment filtering (`ld_matrix`), and multi-grid co-clustering similarity matrices (`sensitivity_analysis`).

## Python Quickstart

```python
import numpy as np
from hyprcoloc import hyprcoloc, subset_coloc, sensitivity_analysis

rng = np.random.default_rng(0)
ses = np.full((200, 4), 0.05)
betas = rng.normal(0.0, ses)
betas[50, :3] += 0.45  # Traits 0, 1, 2 share causal variant at index 50

# 1. Default Pythonic mode (exact coloc.abf equivalence at Q=2, 2-step Newton regional sum)
res = hyprcoloc(betas, ses, trait_names=["CAD", "LDL", "APOB", "BMI"], settings="default")
print(res.summary())

# 2. Upstream R parity mode (matching jrs95/hyprcoloc rapid.hyprcoloc)
res_r = hyprcoloc(
    betas,
    ses,
    trait_names=["CAD", "LDL", "APOB", "BMI"],
    settings="R_default",
)
df = res_r.to_dataframe()
cs95 = res_r.credible_set(cluster_id=1, coverage=0.95)
```

## `settings="default"` vs `settings="R_default"`

| Aspect | `settings="default"` | `settings="R_default"` (`jrs95/hyprcoloc`) |
| :--- | :--- | :--- |
| **Regional probability $P_R$** | $(I_{\text{all}} + I_{\text{align\_alt}}) / (1 + I_{<m} + I_{\text{all}} + I_{\text{align\_alt}})$ (exact `coloc.abf` $\text{PP.H3} + \text{PP.H4}$ at $m=2$) | `rapid.hyprcoloc` formula $I_{\text{all}} / (1 + I_{<m} + I_{\text{all}})$ |
| **Default regional subset depth (`reg_steps`)** | `2`-step (`m - 1` and `m - 2` subsets via exact $O(M m)$ Newton symmetric sum) | `1`-step (`m - 1` trait subsets only, overridable via `reg_steps`) |
| **Pairwise ($m = 2$) subset scaling** | Unscaled (`1.0`, exact `coloc.abf` equivalence) | Applies `rapid.hyprcoloc`'s `0.25` factor on the $m = 2$ regional subset term |
| **Uniform priors (`uniform_priors=True`)** | Unscaled per-SNP configuration weight `prior_1` | Conditionally uniform scaling by $1 / M_{\text{snps}}$ (`prior_1 / M_snps` and $p_{m,1}$ alignment scaling) |
| **Default thresholds (`reg_thresh`, `align_thresh`)** | `0.5`, `0.5` | `0.7`, `0.7` (when not explicitly overridden) |

### Parity Notes

- In `settings="default"`, evaluating $m = 2$ traits with $p_{12} = p_1 p_c$ reproduces Wakefield/Giambartolomei `coloc.abf` (`PP.H3 + PP.H4`, `PP.H4 / (PP.H3 + PP.H4)`, and `PP.H4`) to machine precision.
- In `settings="R_default"`, `hyprcoloc` and `subset_coloc` replicate `jrs95/hyprcoloc`'s `rapid.hyprcoloc` approximation $P_R = I_{\text{all}} / (1 + I_{<m} + I_{\text{all}})$, 1-step regional subsetting, $0.25$ pairwise subset factor, $1 / M_{\text{snps}}$ uniform prior scaling, and `0.7` default regional/alignment thresholds.

## Output Schema

- **`HyPrColocResult`** (returned by `hyprcoloc`):
  - Attributes: `n_snps`, `n_traits`, `snp_ids`, `trait_names`, `clusters` (`list[HyPrColocCluster]`), `uncolocalized_traits`, `lbf_matrix`, `prior_1`, `prior_c`, `reg_thresh`, `align_thresh`, `bb_selection`, `uniform_priors`, `settings`.
  - `res.to_dataframe()`: Cluster summary `pd.DataFrame` with columns `["iteration", "traits", "posterior_prob", "regional_prob", "align_prob", "candidate_snp", "posterior_explained_by_snp", "n_traits", "credible_set_95_size"]`.
  - `res.snp_scores(cluster_id=1)`: Per-SNP posterior table `["snp", "snp_posterior"]`.
  - `res.credible_set(cluster_id=1, coverage=0.95)`: Credible set `list[str]`.
- **`SubsetColocResult`** (returned by `subset_coloc`):
  - Attributes: `traits`, `trait_indices`, `snp_ids`, `regional_prob`, `align_prob`, `posterior_prob`, `candidate_snp`, `candidate_snp_index`, `posterior_explained_by_snp`, `snp_posteriors`.
- **`SensitivityResult`** (returned by `sensitivity_analysis`):
  - Attributes: `trait_names`, `similarity_matrix` (`Q x Q` co-clustering matrix), `n_configurations`, `grid_records`.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/hyprcoloc /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest hyprcoloc/tests
```

## License

GPL-3.0-or-later (matching upstream `jrs95/hyprcoloc`).
