# hyprcoloc

Deterministic $O(M Q)$ Bayesian multi-trait genetic colocalization (**HyPrColoc**: *Hypothesis Prioritisation for multi-trait Colocalization*) in Python.

## Reference

> Foley, C. N., Staley, J. R., Breen, P. G., Sun, B. B., Kirk, P. D. W., Burgess, S., & Howson, J. M. M. (2021). A fast and efficient colocalization algorithm for identifying shared genetic risk factors across multiple traits. *Nature Communications*, 12(1), 764. https://doi.org/10.1038/s41467-020-20885-8

## Mathematical Overview

For $M$ genetic variants across $Q \ge 2$ traits with marginal summary statistics $(\hat{\beta}_{jq}, s_{jq})$:

1. **Wakefield Log-ABFs (`log_abf`)**:
   $$\log \text{BF}_{jq} = \frac{1}{2}\log(1 - r_{jq}) + \frac{1}{2} r_{jq} z_{jq}^2, \qquad r_{jq} = \frac{W_q^2}{s_{jq}^2 + W_q^2}, \qquad z_{jq} = \frac{\hat{\beta}_{jq}}{s_{jq}},$$
   with $W_q = 0.15$ for quantitative traits and $W_q = 0.20$ for binary case-control log-odds.

2. **Exact $O(M |S|)$ Factorization of Posterior Colocalization Probability**:
   With single-trait prior $p_1$ (`1e-4`) and conditional colocalization step parameter $\gamma = 1 - p_c$ (`p_c = 0.02`), configuration priors scale as $\pi_k = p_1 \prod_{i=1}^{k-1}(1 - \gamma^i)$. HyPrColoc decomposes the full colocalization posterior for any trait subset $S$ ($|S| = m \ge 2$) in log-space into:
   $$\text{PP}_{\text{coloc}}(S) = P_R(S) \cdot P_A(S),$$
   where $P_R(S)$ is the regional association probability (evaluated in $O(M m)$ via Newton's symmetric sum identity) and $P_A(S)$ is the single-variant alignment probability. At $Q = 2$ with $p_{12} = p_1 p_c$, $\text{PP}_{\text{coloc}}$ reproduces Wakefield/Giambartolomei `coloc.abf` $\text{PP.H4}$ identically.

3. **Divisive Branch-and-Bound Clustering (`bb_selection`)**:
   Iteratively partitions all $Q$ traits into colocalized clusters satisfying $P_R > P_R^*$ and $P_A > P_A^*$ plus uncolocalized traits, identifying the top candidate causal SNP, proportion of posterior explained, and 95% credible set per cluster.

4. **Sample-Overlap & Prior Sensitivity Analysis**:
   Supports correlated trait estimation errors (`trait_cor`) via multivariate Wakefield joint Bayes factors, LD-aware alignment adjustments (`ld_matrix`), and multi-grid co-clustering stability matrices (`sensitivity_analysis`).

## Quickstart

```python
import numpy as np
from hyprcoloc import hyprcoloc, sensitivity_analysis

rng = np.random.default_rng(0)
ses = np.full((200, 4), 0.05)
betas = rng.normal(0, ses)
# Traits 0, 1, 2 share causal variant at SNP 50
betas[50, :3] += 0.45

res = hyprcoloc(betas, ses, trait_names=["CAD", "LDL", "APOB", "BMI"])
print(res.summary())
df = res.to_dataframe()
cs95 = res.credible_set(cluster_id=1, coverage=0.95)
```

## License

GPL-3.0-or-later (matching upstream `jrs95/hyprcoloc`).
