# `ctwas`: Causal Transcriptome- and Proteome-Wide Association Fine-Mapping

Standalone Python implementation of **cTWAS** (Zhao et al., *Nature Genetics* 2024) for causal Transcriptome/Proteome-Wide Association Studies, eliminating LD hitchhiking and direct non-coding SNP pleiotropy via two-stage Empirical Bayes group-prior estimation and joint `(genes + cis-SNPs)` **SuSiE-RSS** fine-mapping (Zou et al., *PLOS Genetics* 2022).

## Citations

> Zhao, S., Crouse, W., Qian, S., Luo, K., Stephens, M., & He, X. (2024). Adjusting for genetic confounders in transcriptome-wide association studies improves discovery of risk genes of complex traits. *Nature Genetics*, 56(2), 336–347. https://doi.org/10.1038/s41588-023-01648-9
>
> Zou, Y., Carbonetto, P., Wang, G., & Stephens, M. (2022). Fine-mapping from summary data with the "Sum of Single Effects" model. *PLOS Genetics*, 18(7), e1010299. https://doi.org/10.1371/journal.pgen.1010299

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`impute_expr_z`, `estimate_ctwas_priors`, `susie_rss_group_prior`, `run_ctwas`, and `CTWAS`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`xinhe-lab/ctwas` Parity) |
| :--- | :--- | :--- |
| **Stage-1 SNP Thinning (`estimate_ctwas_priors`)** | `thin=1.0` (evaluates all cis-region SNPs without random subsampling unless requested). | `thin=0.1` on background non-coding SNPs (retaining SNPs with $|z| > 1.96$ or cis-weight SNPs and scaling multiplicity by $\approx 1/\text{thin}$). |
| **Stage-1 Empirical Bayes M-Step** | Weak Empirical Bayes pseudocount stabilization so small region sets never collapse $\pi_G$ or $\pi_S$ to exact `0`. | Unregularized `L=1` Single-Effect Regression EM updates matching `ctwas::ctwas_rss`. |
| **Joint LD Regularization (`susie_rss_group_prior`)** | Positive-definite correlation projection with eigenvalue floor `1e-4` on rank-deficient `(genes + cis-SNPs)` $\tilde{R}$. | Eigenvalue floor `1e-6` on joint $\tilde{R}$. |
| **Stage-2 Fine-Mapping Gate (`run_ctwas`)** | Evaluates all regions at Stage 2 with `L=5` (`rerun_gene_pip=0.0`). | Restricts Stage-2 `L=5` IBSS to candidate regions with Stage-1 regional gene PIP sum $\ge 0.8$ (`rerun_gene_PIP=0.8`), reporting Stage-1 PIPs elsewhere. |

## Quickstart

```python
import numpy as np
import pandas as pd
from ctwas import CTWAS, estimate_ctwas_priors, impute_expr_z, run_ctwas, susie_rss_group_prior

# 1. Impute molecular trait Z-scores and exact joint (gene + cis-SNP) correlation matrix
p_snps = 40
ld_R = 0.5 ** np.abs(np.subtract.outer(np.arange(p_snps), np.arange(p_snps)))
z_snps = ld_R[:, 5] * 6.5  # Causal signal driven through eQTL of ENSG0001
gwas_df = pd.DataFrame({"snp": [f"rs{i}" for i in range(p_snps)], "z": z_snps})
weights = [
    {"gene_id": "ENSG0001", "snps": ["rs4", "rs5", "rs6"], "weights": [0.3, 0.8, 0.2]},
    {"gene_id": "ENSG0002", "snps": ["rs25", "rs26"], "weights": [0.6, -0.4]},
]
reg = impute_expr_z(gwas_df, ld_R, weights, region_id="chr1_block1", settings="default")

# 2. Run two-stage cTWAS across genomic regions
res = run_ctwas([reg], n_samples=50000, L=5, settings="default")
print(res.summary())
print(res.gene_df)

# 3. Upstream xinhe-lab/ctwas R parity mode
res_r = CTWAS(L=5, settings="R_default").fit([reg], n_samples=50000)
print(res_r.to_dataframe().head())
```

## License

MIT (clean-room implementation from the published mathematical equations in Zhao et al. 2024 and Zou et al. 2022).
