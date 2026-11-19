# `chord`: Pan-Cancer HRD Detection & `BRCA1`- vs. `BRCA2`-Type Classification

Clean-room Python implementation of **CHORD** (Classifier of HOmologous Recombination Deficiency) and **mutSigExtractor** (Nguyen et al., *Nature Communications* 2020) for pan-cancer homologous recombination deficiency (HRD) probability prediction and `BRCA1`-type vs. `BRCA2`-type subtyping from genome-wide single-base substitutions (96 SBS), bidirectional microhomology and repeat indels, and length-stratified structural variants (19 SV).

## Citation

> Nguyen, L., W. M. Martens, J., Van Hoeck, A., & Cuppen, E. (2020). Pan-cancer landscape of homologous recombination deficiency. *Nature Communications*, 11(1), 5584. https://doi.org/10.1038/s41467-020-19406-4

## `settings` Modes (`"default"` vs `"R_default"`)

All primary functions and classes (`extract_sigs_chord`, `chord_predict`, and `CHORDClassifier`) accept `settings: Literal["default", "R_default"] = "default"` and raise `ValueError` on unknown values:

| Component | `settings="default"` (Pythonic Default) | `settings="R_default"` (`UMCUGenetics/CHORD` v2.03 Parity) |
| :--- | :--- | :--- |
| **132-Channel Contexts (`extract_sigs_chord`)** | 96 pyrimidine SBS trinucleotides + 17 indels (`del.rep.len.1..5`, `ins.rep.len.1..5`, `del.mh.bimh.1..5`, `del.none`, `ins.none`) + 19 SVs. | Identical 132-channel `mutSigExtractor` extraction with alignment-invariant cyclic repeat and `bimh` scanning. |
| **Context Normalization** | Within-class Laplace-smoothed (`alpha=0.25`) relative abundance across `snv`, `indel`, `sv` (`0` when class total is `0`). | Exact within-class (`snv`, `indel`, `sv`) relative abundance `x / sum(x)` (`0` when class sum is `0`). |
| **Mutation Burden Features** | Appends $\log(1 + N_{\text{SNV}})$, $\log(1 + N_{\text{indel}})$, $\log(1 + N_{\text{SV}})$ burden features alongside relative channels. | Strictly scale-invariant relative abundance channels without total count burden features. |
| **Low Indel Load (`total_indels < 50`)** | Preserves continuous probabilities and calls `hr_status` / `hrd_type` while tagging `remarks="low_indel_load"`. | Strict QC mask: sets `hr_status="cannot_be_determined"` and `hrd_type="cannot_be_determined"`. |
| **Microsatellite Instability (`rep.indel > 14000`)** | Overrides `hr_status="MSI"`, `hrd_type="none"`, `remarks="MSI_detected"`. | Exact `CHORD::chordPredict` override: `hr_status="MSI"`, `hrd_type="none"`, `hrd.cutoff = 0.5`. |

## Quickstart

```python
from chord import CHORDClassifier, chord_predict, extract_sigs_chord, simulate_chord_archetypes

# Extract 96 SBS + 17 Indel (repeat, bimh microhomology, none) + 19 SV channels
contexts = extract_sigs_chord(
    snv_df=snv_df,
    indel_df=indel_df,
    sv_df=sv_df,
    ref_fasta="GRCh38.fa",
    sample_id="tumor_01",
    settings="default",
)

# Predict HRD status and BRCA1-type vs BRCA2-type probabilities
res = chord_predict(contexts, hrd_cutoff=0.5, settings="default")
print(res.summary())
print(res.to_dataframe())

# Upstream CHORD v2.03 parity mode or custom cohort training
clf = CHORDClassifier(settings="R_default")
res_r = clf.predict(contexts)

train_ctx, train_labels = simulate_chord_archetypes(n_per_class=50, seed=42)
custom_clf = CHORDClassifier(settings="default").fit(train_ctx, train_labels)
print(custom_clf.predict_proba(contexts))
```

## License

MIT (clean-room implementation from published mathematical and algorithmic descriptions in Nguyen et al. 2020).
