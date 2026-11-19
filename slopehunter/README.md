# `slopehunter`: Model-Based Clustering Adjustment for Index-Event & Collider Bias

Clean-room Python implementation of **SlopeHunter** (Mahmoud et al., *Nature Communications* 2022), separating pure index-trait instruments ("Hunted" cluster) from horizontal-pleiotropic variants ("Pleiotropic" cluster) via a zero-mean bivariate Gaussian mixture EM algorithm to correct collider bias in subsequent-trait GWAS summary statistics.

## Citation

> Mahmoud, O., Dudbridge, F., Davey Smith, G., Munafo, M., & Tilling, K. (2022). A robust tool for collider bias correction in conditional genetic association studies. *Nature Communications*, 13, 610. https://doi.org/10.1038/s41467-022-28119-9

- Upstream R reference: `Osmahmoud/SlopeHunter` (`hunt.R`, `shclust.R`, `SHadj.R`).

## Statistical Model

1. **Two-Component Bivariate Gaussian Mixture (`shclust`, `hunt`)**:
   On independent variants passing $p_{X,j} \le p_{\text{thresh}}$, SlopeHunter models $(\hat{\beta}_{X,j}, \hat{\beta}_{Y,j})$ as a mixture of two zero-mean bivariate Gaussians:
   $$f(\hat{\beta}_{X,j}, \hat{\beta}_{Y,j}) = \pi_0 \, \mathcal{N}_2\!\left(0, \Sigma_{\text{hunted}}\right) + (1 - \pi_0) \, \mathcal{N}_2\!\left(0, \Sigma_{\text{pleio}}\right),$$
   where the "Hunted" cluster has tight correlation $0.95 \cdot \operatorname{sign}(\sigma_{XY,0})$ along the collider-bias line $\hat{\beta}_{Y} = b \hat{\beta}_{X}$ with slope $b = \operatorname{sign}(\sigma_{XY,0}) \, \sigma_{Y,0} / \sigma_{X,0}$, while the "Pleiotropic" cluster has unconstrained covariance bounded by $|\rho_1| \le 0.75$.

2. **Bootstrap Uncertainty & Genome-Wide Adjustment (`sh_adj`)**:
   `hunt` resamples candidate instruments $M$ times to estimate the bootstrap standard error `bse` of $\hat{b}$, and `sh_adj` computes adjusted summary statistics:
   $$\hat{\beta}_{Y,j}^{\text{adj}} = \hat{\beta}_{Y,j} - \hat{b} \hat{\beta}_{X,j}, \qquad s_{Y,j}^{\text{adj}} = \sqrt{s_{Y,j}^2 + \hat{b}^2 s_{X,j}^2 + s_{\hat{b}}^2 \hat{\beta}_{X,j}^2 + s_{\hat{b}}^2 s_{X,j}^2}.$$

## Python Quickstart

```python
import numpy as np
from slopehunter import hunt, sh_adj, shclust, harmonise_effects

rng = np.random.default_rng(42)
m = 300
x_true = rng.normal(0.0, 0.08, m)
direct_y = np.concatenate([np.zeros(240), rng.normal(0.0, 0.12, 60)])
xse = np.full(m, 0.01)
yse = np.full(m, 0.012)
xbeta = x_true + rng.normal(0.0, xse)
ybeta = direct_y - 0.35 * x_true + rng.normal(0.0, yse)

# Pythonic default mode (signed bootstrap SD)
res_def = hunt(xbeta, xse, ybeta, yse, xp_thresh=0.05, M=50, seed=11, settings="default")
yb_adj, ys_adj = sh_adj(res_def, xbeta, xse, ybeta, yse)

# Exact upstream R parity mode (SlopeHunter::hunt folded bootstrap SD)
res_r = hunt(xbeta, xse, ybeta, yse, xp_thresh=0.05, M=50, seed=11, settings="R_default")
print(res_def["b"], res_def["bse"], res_r["bse"])
```

## `settings="default"` vs `settings="R_default"`

`hunt`, `shclust`, and `sh_adj` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values):

| Feature | `settings="default"` | `settings="R_default"` (`Osmahmoud/SlopeHunter`) |
| :--- | :--- | :--- |
| **Bootstrap Slope SE (`bse` in `hunt`)** | Signed sample SD `np.std(bts, ddof=1)` (unbiased when bootstrap draws straddle $b = 0$) | Exact folded bootstrap SD `np.std(np.abs(bts), ddof=1)` (`hunt.R:117`: `sd(abs(bts))`) |
| **EM Log-Likelihood Stopping** | Relative log-likelihood convergence `\|Δℓ\| / (\|ℓ\| + 1e-12) < 1e-10` | Identical relative stopping rule (avoids premature exit when $\ell < 0$) |

## Output Schema

### `hunt(...) -> dict`
- `b` (`float`): Estimated collider-bias slope.
- `bse` (`float`): Bootstrap standard error of `b` (`0.0` if `bootstrapping=False`).
- `b_CI` (`tuple[float, float]`): 95% confidence interval `(b - 1.96 * bse, b + 1.96 * bse)`.
- `pi` (`float`): Estimated proportion $\hat{\pi}_0$ of pure index-trait ("Hunted") variants.
- `entropy` (`float`): Mean posterior probability among variants assigned to `"Hunted"`.
- `Fit` (`pd.DataFrame`): Per-SNP table with `xbeta`, `ybeta`, `pt`, `po`, `clusters`, `SNP`.
- `iter` (`int`), `loglkl` (`float`), `n_fit` (`int`): EM iterations, log-likelihood, and fitted SNP count.
- `Bts_est` (`np.ndarray`, when `bootstrapping=True`): Converged bootstrap slope replicates.

### `sh_adj(...) -> tuple[np.ndarray, np.ndarray]`
- Returns `(ybeta_adj, yse_adj)` across all supplied SNPs.

## Testing

```bash
PYTHONPATH=/Users/yutaro/Documents/gillanilab/pythonic_biotools/slopehunter \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest slopehunter/tests -v
```
