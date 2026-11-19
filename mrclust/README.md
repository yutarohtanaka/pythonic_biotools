# `mrclust`: Clustered Heterogeneity in Mendelian Randomization

Standalone Python implementation of **MR-Clust** (Foley et al., *Bioinformatics* 2021) for identifying distinct causal pathway clusters, null variants, and pleiotropic outliers in two-sample Mendelian randomization.

## Citation

> Foley, C. N., Mason, A. M., Kirk, P. D. W., & Burgess, S. (2021). MR-Clust: clustering of genetic variants in Mendelian randomization with similar causal estimates. *Bioinformatics*, 37(4), 531–541. https://doi.org/10.1093/bioinformatics/btaa778

## Mathematical & Algorithmic Overview

Given $J \ge 2$ independent genetic instruments with exposure associations $(\hat{\beta}_{X,j}, \text{se}(\hat{\beta}_{X,j}))$ and outcome associations $(\hat{\beta}_{Y,j}, \text{se}(\hat{\beta}_{Y,j}))$ (or precomputed ratio estimates $(\hat{\theta}_j, \hat{\sigma}_j)$):

1. **Wald Ratio Estimates & Delta-Method Variances**:
   $$\hat{\theta}_j = \frac{\hat{\beta}_{Y,j}}{\hat{\beta}_{X,j}},$$
   with observation-specific variance under first-order (`variance_order=1`) or second-order (`variance_order=2`) delta-method approximation:
   $$\hat{\sigma}_{j,(1)}^2 = \frac{\text{se}(\hat{\beta}_{Y,j})^2}{\hat{\beta}_{X,j}^2}, \qquad \hat{\sigma}_{j,(2)}^2 = \frac{\text{se}(\hat{\beta}_{Y,j})^2}{\hat{\beta}_{X,j}^2} + \frac{\hat{\beta}_{Y,j}^2 \, \text{se}(\hat{\beta}_{X,j})^2}{\hat{\beta}_{X,j}^4}.$$

2. **$(K + 2)$-Component Mixture Model**:
   For $K \ge 0$ substantive causal clusters:
   - **Substantive clusters** ($k = 1, \dots, K$): $\hat{\theta}_j \mid (\xi_j = k) \sim \mathcal{N}(\mu_k, \hat{\sigma}_j^2)$.
   - **Null cluster**: $\hat{\theta}_j \mid (\xi_j = \text{Null}) \sim \mathcal{N}(0, \hat{\sigma}_j^2)$.
   - **Junk/outlier cluster** (default `junk_dist="t"`): generalized Student-$t$ distribution with $\nu = 4$ degrees of freedom, location $\mu_{\text{junk}} = \bar{\hat{\theta}}$, and scale $\psi_{\text{junk}} = |\hat{\theta}_{\max} - \hat{\theta}_{\min}| + 2\hat{\sigma}_{\max}$.

3. **Log-Space Expectation-Maximization & Split-Merge Optimization**:
   Responsibilities $r_{j,c}$ are computed in log-sum-exp space to avoid underflow under heteroskedastic precisions. Each M-step updates mixture weights $\pi_c$ and precision-weighted inverse-variance means:
   $$\mu_k^{(t+1)} = \frac{\sum_{j=1}^J r_{j,k}^{(t)} \hat{\theta}_j / \hat{\sigma}_j^2}{\sum_{j=1}^J r_{j,k}^{(t)} / \hat{\sigma}_j^2}.$$
   Global optimization combines empirical quantiles, 1D Ward hierarchical clustering, precision-weighted and unweighted multi-start $k$-means++ with randomized null/junk weights, forward splitting from $K-1$ (including local-density Junk/Null rescue), and duplicate-cluster merge-resplit guards.

4. **BIC Model Selection & Probability Filtering**:
   Optimal $K \in \{k_{\min}, \dots, k_{\max}\}$ minimizes $\text{BIC}(K) = -2\ell(\hat{\Theta}_K) + p_K \log J$ where $p_K = 2K + 1$ when both Null and Junk components are included. `obs_prob_thresh` and `pr_clust` separate confident cluster assignments from `"Unassigned"` boundary variants.

## Installation

```bash
pip install -e ./mrclust
```

## Quickstart

```python
import numpy as np
from mrclust import mr_clust_em, pr_clust

rng = np.random.default_rng(42)
theta_true = np.concatenate([
    np.full(20, -0.60),
    np.full(20, 0.85),
    np.zeros(12),
    np.array([-3.2, -2.5, 2.7, 3.5]),
])
bx = rng.uniform(0.05, 0.12, size=len(theta_true))
bxse = rng.uniform(0.005, 0.01, size=len(theta_true))
byse = rng.uniform(0.004, 0.009, size=len(theta_true))
by = theta_true * bx + rng.normal(0.0, byse)

res = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, seed=42)
print(res.summary())
print(res.clusters_dataframe())
filtered = pr_clust(res, obs_prob_thresh=0.8)
```

## License

MIT (clean-room implementation from published equations in Foley et al., *Bioinformatics* 2021).
