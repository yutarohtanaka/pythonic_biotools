# `decoupler-ulm`: Univariate Linear Model Gene-Program Projection

Standalone Python implementation (`decoupler_ulm`) of Univariate Linear Model (**ULM**, `decoupleR::run_ulm`), `log2_dot`, and `top_k_zscore` gene-expression program (GEP) and transcription factor regulon activity projection onto bulk, pseudobulk, or single-cell RNA-seq expression matrices.

---

## Published Citations

- **Badia-i-Mompel, P., Vélez Santiago, J., Braunger, J., Geiss, C., Dimitrov, D., Müller-Dott, S., Taus, P., Dugourd, A., Holland, C. H., Ramirez Flores, R. O., & Saez-Rodriguez, J.** (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. *Bioinformatics Advances*, 2(1), vbac016. https://doi.org/10.1093/bioadv/vbac016
- **Kotliar, D., Veres, A., Nagy, M. A., Tabrizi, S., Hodis, E., Melton, D. A., & Sabeti, P. C.** (2019). Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq. *eLife*, 8, e43803. https://doi.org/10.7554/eLife.43803

---

## Statistical Formulation

Given a gene-by-sample expression matrix $Y \in \mathbb{R}^{G \times S}$ and a gene-by-program continuous weight matrix $W \in \mathbb{R}^{G \times P}$:

1. **Univariate Linear Model (`method="ulm_tstat"`)**:
   For each sample $s$ and program $p$, fits the exact ordinary least squares regression $y_{g,s} = \beta_0 + \beta_1 w_{g,p} + \varepsilon_{g,s}$ across $G_{\text{eff}}$ genes ($\text{df} = G_{\text{eff}} - 2$). Defining the sample Pearson correlation $r_{s,p}$ between centered $y_{\cdot,s}$ and centered $w_{\cdot,p}$, the exact OLS slope $t$-statistic and two-sided Student-$t$ $p$-value are evaluated in vectorized $O(G S P)$ matrix algebra:
   $$t_{s,p} = r_{s,p}\sqrt{\frac{G_{\text{eff}} - 2}{1 - r_{s,p}^2}}, \qquad p_{s,p} = 2 \, P\big(T_{G_{\text{eff}} - 2} \ge |t_{s,p}|\big)$$
   When `center_across_samples=False` (`decoupleR::run_ulm`), each sample is scored strictly independently (supporting $S = 1$). When `center_across_samples=True` (in `settings="default"`), each gene is first $Z$-scored across samples (`ddof=1`).

2. **NaN-Safe Positive Weighted Log2 Dot Product (`method="log2_dot"`)**:
   $$S_{s,p} = \frac{\sum_{g : y_{g,s} \text{ finite}} y_{g,s} \max(w_{g,p}, 0)}{\sum_{g : y_{g,s} \text{ finite}} \max(w_{g,p}, 0)}$$

3. **Top-$K$ Marker Mean $Z$-Score (`method="top_k_zscore"`)**:
   Computes the mean across-sample gene $Z$-score over the `top_k_markers` highest-weighted genes of each program, with optional Stouffer normal $p$-values ($\sqrt{K_{\text{obs}}} |\bar{Z}|$).

---

## Quickstart

```python
import numpy as np
import pandas as pd
from decoupler_ulm import project_gep_programs

rng = np.random.default_rng(42)
genes = [f"G{i}" for i in range(100)]
expr_df = pd.DataFrame(rng.normal(size=(100, 8)), index=genes, columns=[f"S{j}" for j in range(8)])
regulon_df = pd.DataFrame({"MYC_targets": rng.uniform(0.5, 2.0, size=15)}, index=genes[:15])
expr_df.iloc[:15, 0] += 2.5

# Exact saezlab/decoupleR::run_ulm mode (uses input scale directly, pads 85 background genes with 0.0)
scores_df, pvals_df = project_gep_programs(
    expr_df,
    regulon_df,
    method="ulm_tstat",
    return_pvalues=True,
    settings="R_default",
)
print(scores_df.head())
```

---

## `settings="default"` vs `settings="R_default"` Reference

`project_gep_programs` accepts `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Aspect | `settings="default"` | `settings="R_default"` (`saezlab/decoupleR::run_ulm`) |
|---|---|---|
| **Expression input transform** | Applies $\log_2(\max(X, 0) + 1)$ when `is_log_transformed=False` | Uses `expr_genes_by_samples` directly on its supplied scale (no auto $\log_2(X + 1)$) |
| **Feature alignment & non-targets** | Subsets both `expr` and `W` to shared genes (`len(shared) >= 10`) | Retains all genes in `expr` and pads features absent from `W` with `0.0` weights |
| **Minimum shared targets** | `>= 10` shared genes | `minsize >= 5` shared targets |
| **Across-sample centering** | Controlled by `center_across_samples` / `zscore_genes` | Forced `center_across_samples = False` (strict single-sample independence) |

---

## Public API & Output Schema

```python
project_gep_programs(
    expr_genes_by_samples: pd.DataFrame,
    program_weights_genes_by_programs: pd.DataFrame,
    is_log_transformed: bool = False,
    top_k_markers: int = 50,
    method: str = "ulm_tstat",          # {'ulm_tstat', 'log2_dot', 'top_k_zscore'}
    center_across_samples: bool = False,
    zscore_genes: bool | None = None,
    return_pvalues: bool = False,
    *,
    settings: Literal["default", "R_default"] = "default",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
```

- **Returns**:
  - `scores_df`: `pd.DataFrame` (`samples x programs`) of program activity scores ($t$-statistics for `ulm_tstat`).
  - `pvalues_df` *(when `return_pvalues=True`)*: `pd.DataFrame` (`samples x programs`) of two-sided Student-$t$ (or normal) $p$-values.

---

## Testing

```bash
PYTHONPATH=decoupler-ulm /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest decoupler-ulm/tests -v
```

## License

MIT.
