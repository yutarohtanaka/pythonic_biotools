# Multi-Package Workflow Examples (`examples/`)

Runnable end-to-end showcase scripts demonstrating how standalone libraries in `pythonic_biotools` compose across oncology, statistical genetics, clinical trials, epigenomics, and quantitative proteomics.

## Running Any Demo

```bash
python examples/oncology_evolution_demo.py
python examples/statgen_drug_target_demo.py
python examples/clinical_trial_endpoints_demo.py
python examples/epigenomics_and_proteomics_demo.py
```

- **[`oncology_evolution_demo.py`](oncology_evolution_demo.py)**: Combines `scarhrd`, `mutationtimer`, `tracksig`, `mmsig`, `ichorcna`, and `conipher`.
- **[`statgen_drug_target_demo.py`](statgen_drug_target_demo.py)**: Combines `cismr`, `colocproptest`, `ctwas`, `lava`, `genomicsem`, and `fgwas`.
- **[`clinical_trial_endpoints_demo.py`](clinical_trial_endpoints_demo.py)**: Combines `timeroc`, `survidinri`, `survrm2`, `buysetest`, and `jmbayes2`.
- **[`epigenomics_and_proteomics_demo.py`](epigenomics_and_proteomics_demo.py)**: Combines `dss`, `csaw`, `monalisa`, and `deqms`.
