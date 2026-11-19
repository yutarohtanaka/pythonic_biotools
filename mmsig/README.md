# `mmsig`: Interpretable Mutational Signature Refitting & Bootstrap Pruning

Standalone Python implementation of **mmsig** (Rustad et al., *Communications Biology* 2021) for interpretable single-base substitution (SBS) mutational signature refitting via stepwise backward cosine-similarity pruning, parametric multinomial bootstrap confidence intervals, and transcriptional strand-bias confirmation for chemotherapy signatures.

## Citations

> Rustad, E. H., Nadeu, F., Angelopoulos, N., Ziccheddu, B., Bolli, N., Puente, X. S., Campo, E., Landgren, O., & Maura, F. (2021). mmsig: a flexible mutational signature fitting tool. *Communications Biology*, 4, 424. https://doi.org/10.1038/s42003-021-01938-0

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`mm_fit_signatures`, `check_strand_bias`, `build_96_catalog`, and `MMSig`) accept `settings: Literal["default", "R_default"] = "default"`:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`UM-Myeloma-Genomics/mmsig` Parity) |
| :--- | :--- | :--- |
| **Weight Solver** | Hybrid NNLS warm-start + Multinomial EM updates on the unit simplex ($\sum w_k = 1, w_k \ge 0$). | Simplex-constrained least squares ($\min \|S w - y\|_2^2$ s.t. $\sum w_k = 1, w_k \ge 0$ matching `pracma::lsqlincon`). |
| **Backward Pruning** | Iteratively deletes the non-forced signature with minimal cosine similarity loss while $\Delta\cos < 0.01$. | Stepwise backward pruning at `cos_sim_threshold = 0.01` with `sbs1_5_force = True` (`SBS1`, `SBS5`). |
| **Multinomial Bootstrap** | `n_boot = 500` resampled 96-channel catalogs ($N_s = \sum_c m_{cs}$) refitted with full backward pruning. | `n_boot = 1000` full-pruning multinomial replicates reporting `estimate`, `ci_low`, `ci_high`, `sd`, `present_prob`. |
| **Presence Confirmation** | Retains backward-pruned exposures; confirms chemo signatures (`SBS31`/`SBS35`/`SBS99`) if `strand_bias_df` supplied. | Requires lower 95% bootstrap CI `> 0` for `SBS8` / `SBS99`, plus strand bias $p < 0.05$ when `strand_bias_df` supplied. |
| **Strand Asymmetry (`check_strand_bias`)** | Exact conditional Poisson/Binomial test with Haldane $+0.5$ strand ratio and Benjamini–Hochberg `q_value`. | Exact `stats::poisson.test` conditional rate ratio $(n_T / E_T) / (n_U / E_U)$ and two-sided $p < 0.05$ gate. |

## Quickstart

```python
import numpy as np
import pandas as pd
from mmsig import MMSig, check_strand_bias, make_synthetic_cosmic_reference, mm_fit_signatures

# 1. Build or load a 96-trinucleotide x signature reference matrix
sig_ref = make_synthetic_cosmic_reference(["SBS1", "SBS2", "SBS5", "SBS8", "SBS99"])

# 2. Simulate 96-channel somatic mutation counts for two myeloma samples
rng = np.random.default_rng(42)
catalog = pd.DataFrame(
    {
        "MM_01": rng.multinomial(1200, 0.25 * sig_ref["SBS1"] + 0.45 * sig_ref["SBS5"] + 0.30 * sig_ref["SBS99"]),
        "MM_02": rng.multinomial(900, 0.35 * sig_ref["SBS1"] + 0.40 * sig_ref["SBS5"] + 0.25 * sig_ref["SBS2"]),
    },
    index=sig_ref.index,
)

# 3. Fit with Pythonic defaults (NNLS + Multinomial EM, n_boot=500)
res = mm_fit_signatures(catalog, sig_ref, settings="default")
print(res.summary())
print(res.exposures.round(3))

# 4. Fit in R parity mode with transcriptional strand-bias confirmation for SBS99
strand_df = pd.DataFrame(
    [
        {"sample": "MM_01", "signature": "SBS99", "transcribed": 145, "untranscribed": 68},
        {"sample": "MM_02", "signature": "SBS99", "transcribed": 22, "untranscribed": 24},
    ]
)
sb = check_strand_bias(strand_df, settings="R_default")
res_r = MMSig(settings="R_default", seed=42).fit(catalog, sig_ref, strand_bias_df=strand_df)
print(res_r.bootstrap_df.head())
```

## API Overview

- `mm_fit_signatures(...)` / `MMSig`: Exposure estimation, stepwise backward cosine pruning, bootstrap CIs, and presence confirmation.
- `check_strand_bias(strand_df, alpha=0.05, settings=...)`: Exact Poisson/Binomial transcriptional strand asymmetry testing.
- `build_96_catalog(muts_df, ...)`: Converts variant-level `(sample, ref, alt, trinuc)` tables or `sub`/`tri` catalogs into canonical 96-channel matrices.

## License

MIT (clean-room implementation from published mathematical methodology in Rustad et al., *Communications Biology* 2021).
