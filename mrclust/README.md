# `mrclust`: Clustered Heterogeneity in Mendelian Randomization

Standalone Python implementation of **MR-Clust** (Foley et al., *Bioinformatics* 2021) for identifying distinct causal pathway clusters, null variants, and pleiotropic outliers in two-sample Mendelian randomization.

## Citation

> Foley, C. N., Mason, A. M., Kirk, P. D. W., & Burgess, S. (2021). MR-Clust: clustering of genetic variants in Mendelian randomization with similar causal estimates. *Bioinformatics*, 37(4), 531–541. https://doi.org/10.1093/bioinformatics/btaa778

## Statistical Model

Given $J \ge 2$ independent genetic instruments with exposure associations $(\hat{\beta}_{X,j}, s_{X,j})$ and outcome associations $(\hat{\beta}_{Y,j}, s_{Y,j})$ (or precomputed ratio estimates $(\hat{\theta}_j, \hat{\sigma}_j)$):

1. **Wald Ratio Estimates & Delta-Method Variances** (`compute_ratio_estimates`):
   $$\hat{\theta}_j = \frac{\hat{\beta}_{Y,j}}{\hat{\beta}_{X,j}},$$
   with observation-specific variance under first-order (`variance_order=1`) or second-order (`variance_order=2`) delta-method approximation:
   $$\hat{\sigma}_{j,(1)}^2 = \frac{s_{Y,j}^2}{\hat{\beta}_{X,j}^2}, \qquad \hat{\sigma}_{j,(2)}^2 = \frac{s_{Y,j}^2}{\hat{\beta}_{X,j}^2} + \frac{\hat{\beta}_{Y,j}^2 \, s_{X,j}^2}{\hat{\beta}_{X,j}^4}.$$

2. **$(K + 2)$-Component Mixture Model** (`mr_clust_em`):
   For $K \ge 0$ substantive causal clusters:
   - **Substantive clusters** ($k = 1, \dots, K$): $\hat{\theta}_j \mid (\xi_j = k) \sim \mathcal{N}(\mu_k, \hat{\sigma}_j^2)$.
   - **Null cluster**: $\hat{\theta}_j \mid (\xi_j = \text{Null}) \sim \mathcal{N}(0, \hat{\sigma}_j^2)$.
   - **Junk/outlier cluster** (`junk_dist="t"`, $\nu = 4$): heavy-tailed generalized Student-$t$ component absorbing pleiotropic outliers without biasing substantive cluster means.

3. **Log-Space EM & BIC Model Selection**:
   Responsibilities $r_{j,c}$ are evaluated in log-sum-exp space, and each M-step updates mixture weights $\pi_c$ and inverse-variance weighted cluster means:
   $$\mu_k^{(t+1)} = \frac{\sum_{j=1}^J r_{j,k}^{(t)} \hat{\theta}_j / \hat{\sigma}_j^2}{\sum_{j=1}^J r_{j,k}^{(t)} / \hat{\sigma}_j^2}.$$
   Optimal $K \in \{k_{\min}, \dots, k_{\max}\}$ minimizes $\text{BIC}(K) = -2\ell(\hat{\Theta}_K) + p_K \log J$ where $p_K = 2K + 1$ when Null and Junk components are included. `obs_prob_thresh` and `pr_clust` separate confident assignments from `"Unassigned"` boundary variants.

## Python Quickstart

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

# Default Pythonic mode (weak-IV-robust Junk parameterization + multi-strategy EM search)
res = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, seed=42, settings="default")
print(res.summary())
filtered = pr_clust(res)

# Upstream R parity mode (matching cnfoley/mrclust::mr_clust_em)
res_r = mr_clust_em(bx=bx, bxse=bxse, by=by, byse=byse, seed=42, settings="R_default")
print(res_r.clusters_dataframe())
filtered_r = pr_clust(res_r)
```

## `settings="default"` vs `settings="R_default"`

All primary entrypoints (`mr_clust_em`, `pr_clust`, `compute_ratio_estimates`) accept `settings: Literal["default", "R_default"] = "default"`.

| Feature | `settings="default"` | `settings="R_default"` (`cnfoley/mrclust`) |
| :--- | :--- | :--- |
| **Junk component mean** (`junk_mean`) | Well-powered empirical center of $\hat{\theta}$ | Fixed at `0.0` (`loglik.R`) |
| **Junk component scale** (`junk_scale`) | Weak-IV-masked range $+ 2 \max(\hat{\sigma}_{\text{wp}})$ | $\text{range}(\hat{\theta}) + \hat{\sigma}_{\arg\max_j(|\hat{\theta}_j| + 2\hat{\sigma}_j)}$ |
| **Junk observation variance** | Convolved $\sqrt{\psi_{\text{junk}}^2 + \hat{\sigma}_j^2}$ (`hypot` inflation) | Fixed $\psi_{\text{junk}}$ without `hypot` inflation (`loglik.R`) |
| **Default `obs_prob_thresh`** | `0.8` | `0.5` |
| **EM initialization** | Quantiles + 1D Ward + 20 weighted/unweighted $k$-means++ + split-merge | 6 unweighted $k$-means starts (`mr_clust_em.R`) |

Explicit user overrides (e.g., `obs_prob_thresh=0.8`, `junk_mean=...`, `junk_scale=...`, `n_init=...`) take precedence in both modes.

## Output Schema

`mr_clust_em(...)` returns a frozen `MRClustResult` dataclass with structured export helpers:

- **`res.to_dataframe(include_probs=False)` / `res.snp_table()`** (per-variant assignments):
  - `observation` (`str`), `theta` (`float64`), `theta_se` (`float64`), `cluster` (`str`, `"Unassigned"` below threshold), `best_cluster` (`str`), `cluster_class` (`"Substantive" | "Null" | "Junk" | "Unassigned"`), `cluster_mean` (`float64`), `probability` (`float64`), `passes_threshold` (`bool`), optional `bx`, `bxse`, `by`, `byse`, and `prob_<component>`.
- **`res.clusters_dataframe()` / `res.cluster_summary()`** (per-component summary):
  - `cluster`, `cluster_class`, `cluster_mean`, `cluster_se`, `weight`, `effective_n`, `n_assigned`, `n_assigned_thresh`.
- **`res.bic_dataframe()` / `res.bic_table()`** (model selection across $K$):
  - `k`, `n_components`, `n_parameters`, `log_likelihood`, `bic`, `aic`, `is_best`, `converged`, `n_iter`.

## Testing

```bash
PYTHONPATH=mrclust /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest mrclust/tests
```
