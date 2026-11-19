# mrpresso

Fast, vectorized Python implementation of **MR-PRESSO** (*Mendelian Randomization Pleiotropy RESidual Sum and Outlier*) and **Multivariable MR-PRESSO** for detecting and correcting horizontal pleiotropy in summary-level Mendelian randomization.

## Reference

> Verbanck, M., Chen, C.-Y., Neale, B., & Do, R. (2018). Detection of widespread horizontal pleiotropy in causal relationships inferred from Mendelian randomization between complex traits and diseases. *Nature Genetics*, 50(5), 693–698. https://doi.org/10.1038/s41588-018-0099-7

## Mathematical Overview

Given $J$ independent genetic instruments with exposure summary statistics $(\hat{\beta}_{X,j}, s_{X,j})$ and outcome summary statistics $(\hat{\beta}_{Y,j}, s_{Y,j})$ and inverse-variance weights $w_j = s_{Y,j}^{-2}$:

1. **Leave-One-Out Observed RSS ($\text{RSS}_{\text{obs}}$)**:
   Each leave-one-out IVW slope $\hat{\beta}^{(-j)}$ is computed in closed form via rank-1 updates (exact $O(J)$ sums in univariable MR; Sherman-Morrison $(X^\top W X - w_j x_j x_j^\top)^{-1}$ / weighted hat-matrix leverage $r_j = e_j / (1 - h_j)$ in multivariable MR):
   $$\text{RSS}_{\text{obs}} = \sum_{j=1}^J r_{j,\text{obs}}^2 = \sum_{j=1}^J \left(\hat{\beta}_{Y,j} - x_j^\top \hat{\beta}^{(-j)}\right)^2.$$

2. **Global Test**:
   Draws $K$ Monte Carlo null datasets completely vectorized in NumPy under zero horizontal pleiotropy:
   $$\beta_{X,j}^{(k)} \sim \mathcal{N}(\hat{\beta}_{X,j}, s_{X,j}^2), \qquad \beta_{Y,j}^{(k)} \sim \mathcal{N}(x_j^\top \hat{\beta}^{(-j)}, s_{Y,j}^2),$$
   evaluating all $K \times J$ leave-one-out residuals without Python loops over instruments.

3. **Outlier Test & Outlier-Corrected IVW**:
   When $p_{\text{global}} < \alpha$, per-instrument $p$-values compare $r_{j,\text{obs}}^2$ against the simulated null distribution (empirical tail matching `rondolab/MR-PRESSO` or calibrated $\chi^2_1$ via $r_{j,\text{obs}}^2 / \mathbb{E}[(r_j^{(k)})^2]$), apply Bonferroni correction $\min(1, J \cdot p_j)$, remove significant outliers, and re-estimate the outlier-corrected IVW causal effect.

4. **Distortion Test**:
   Quantifies the percentage bias induced by horizontal pleiotropic outliers:
   $$D = 100 \times \frac{\hat{\beta}_{\text{raw}} - \hat{\beta}_{\text{corr}}}{|\hat{\beta}_{\text{corr}}|},$$
   with a two-sided bootstrap $p$-value resampled across non-outlier instruments.

## Quickstart

```python
import numpy as np
from mrpresso import mr_presso, mvmr_presso

rng = np.random.default_rng(42)
bx = rng.uniform(0.05, 0.25, size=30)
sx = np.full(30, 0.01)
sy = np.full(30, 0.02)
by = 0.5 * bx + rng.normal(0, sy)
by[0] += 0.40  # horizontal pleiotropy outlier

res = mr_presso(by, bx, sy, sx, nb_distribution=5000, seed=42)
print(res.summary())
df_main = res.to_dataframe()
df_snps = res.outlier_table()
```

## License

GPL-3.0-or-later (matching the upstream `rondolab/MR-PRESSO` R package).
