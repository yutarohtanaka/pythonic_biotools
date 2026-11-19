# `tracksig`: Evolutionary Pseudotime Changepoint Segmentation of Mutational Signatures

Standalone Python implementation of **TrackSig** (Rubanova et al., *Nature Communications* 2020) and **TrackSigFreq** (Harrigan et al., *PSB* 2020) for reconstructing evolutionary trajectories of mutational signature exposures and subclonal expansions across cancer cell fraction (CCF) pseudotime via Pruned Exact Linear Time (**PELT**) dynamic programming.

## Citations

> Rubanova, Y., Shi, R., Harrigan, C. F., Li, R., Wintersinger, J., Sahin, N., Deshwar, A. G., PCAWG Evolution and Heterogeneity Working Group, & Morris, Q. D. (2020). Reconstructing evolutionary trajectories of mutation signature activities in cancer using TrackSig. *Nature Communications*, 11(1), 731. https://doi.org/10.1038/s41467-020-14352-7
>
> Harrigan, C. F., Rubanova, Y., Morris, Q., & Selega, A. (2020). TrackSigFreq: subclonal reconstructions based on mutation signatures and allele frequencies. *Pacific Symposium on Biocomputing*, 25, 238–249. https://doi.org/10.1142/9789811215636_0021

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`compute_ccf_pseudotime`, `fit_multinomial_signatures_em`, `detect_active_signatures`, `tracksig`, and `TrackSig`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`morrislab/TrackSig` Parity) |
| :--- | :--- | :--- |
| **Pseudotime Binning (`compute_ccf_pseudotime`)** | Adaptive fallback (`min_bin_size=25` when `< 200` SNVs) so WES tumors segment cleanly out of the box. | Strict `bin_size=100` enforcement (raises `ValueError` if `n_mutations < bin_size` unless overridden). |
| **CCF Multiplicity Scoring** | Beta-Binomial likelihood over $m \in \{1, \dots, M_i\}$ with continuity-corrected tie breaking across discrete read counts. | Same Beta-Binomial somatic multiplicity inversion sorted in decreasing $\hat{\phi}_i$ pseudotime order. |
| **Active Signature Pre-filter (`detect_active_signatures`)** | Global Multinomial EM with SQUAREM acceleration and `active_threshold=0.05` (`>= 1` top-exposure fallback). | Global EM `activeInSample` filter at `active_threshold=0.05`. |
| **`Signature` PELT Penalty** | Mutation-count BIC penalty $\max(1, K_{\text{active}} - 1) \log(N_{\text{mutations}})$. | Bin-scaled PELT BIC penalty $\max(1, K_{\text{active}} - 1) \log(T_{\text{bins}})$. |
| **`SigFreq` PELT Objective & Penalty** | Joint Multinomial 96-context + piecewise-truncated CCF mixture likelihood with $K_{\text{active}} \log(N_{\text{mutations}})$ penalty. | Joint Multinomial + CCF mixture deviance with $((K_{\text{active}} - 1) + 2) \log(T_{\text{bins}})$ penalty. |

## Quickstart

```python
import numpy as np
import pandas as pd
from tracksig import (
    COSMIC_96_CONTEXTS,
    TrackSig,
    compute_ccf_pseudotime,
    detect_active_signatures,
    fit_multinomial_signatures_em,
    tracksig,
)

# 1. Construct or load a 96 x K reference mutational signature matrix
rng = np.random.default_rng(42)
sig_df = pd.DataFrame(
    rng.dirichlet(np.full(96, 0.3), size=4).T,
    index=COSMIC_96_CONTEXTS,
    columns=["SBS1", "SBS2", "SBS5", "SBS13"],
)

# 2. Simulate clonal (CCF=1.0, SBS1/SBS5) -> subclonal (CCF=0.35, SBS2/SBS13) trajectory
clonal_ctx = rng.choice(COSMIC_96_CONTEXTS, size=300, p=0.7 * sig_df["SBS1"] + 0.3 * sig_df["SBS5"])
sub_ctx = rng.choice(COSMIC_96_CONTEXTS, size=300, p=0.6 * sig_df["SBS2"] + 0.4 * sig_df["SBS13"])
depths = np.full(600, 90)
alts = np.concatenate([rng.binomial(90, 0.40, size=300), rng.binomial(90, 0.14, size=300)])
mut_df = pd.DataFrame({"context": np.concatenate([clonal_ctx, sub_ctx]), "alt_count": alts, "total_depth": depths})

# Default Pythonic TrackSigFreq pipeline with nonparametric bootstrap
res = tracksig(mut_df, sig_df, purity=0.8, bin_size=100, score_method="SigFreq", bootstrap_iters=20, settings="default")
print(res.summary())
print(res.to_dataframe().head())

# Upstream morrislab/TrackSig R parity mode
res_r = TrackSig(purity=0.8, bin_size=100, score_method="Signature", settings="R_default").fit(mut_df, sig_df)
print(res_r.segment_exposures_df)
```

## License

MIT (clean-room implementation from the published mathematical equations in Rubanova et al. 2020 and Harrigan et al. 2020).
