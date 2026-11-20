# `pythonic_biotools` Documentation

Pure-Python, PEP 561-typed implementations of **70 published R, Bioconductor, CRAN, C++, and Fortran libraries** spanning cancer genomics, ctDNA liquid biopsy, mutational signatures, Mendelian randomization, colocalization, causal TWAS, multivariate statistical genetics, bulk & spatial deconvolution, single-cell mixed models, quantitative proteomics, epigenomics, and clinical-trial survival biostatistics.

Every library supports **both**:

- **`settings="default"`** — statistically hardened, numerically stabilized Pythonic defaults.
- **`settings="R_default"`** — strict formula-, threshold-, and convention-level replication of the original R / C++ / Fortran software (`98.1%` mean source consistency across all 70 packages).

---

## Quick Install & First Steps

```bash
pip install "pythonic-biotools[all] @ git+https://github.com/gillanilab/pythonic_biotools.git"
```

```python
from pythonic_biotools import purecn, ichorcna, ctwas, cismr, deqms, survrm2, io

# Load real-world bioinformatics formats directly
maf_df = io.read_maf("cohort.maf.gz")
abundance_df, counts = io.read_maxquant_diann("proteinGroups.txt")

# Run analyses with optional upstream R_default replication & built-in diagnostic plots
rmst_res = survrm2.rmst2(time, status, arm, tau=24.0, settings="R_default")
fig = rmst_res.plot()
```

---

## Core Documentation Guides

- **[Installation, File Readers (`io`) & Pipeline CLI (`pbt`)](CLI_AND_IO_GUIDE.md)** — One-line installation, `MAF`/`VCF`/`GATK`/`Bismark`/`MaxQuant`/`GWAS` loaders, CLI subcommands, and Docker/Conda usage.
- **[R-to-Python Rosetta Stone (`R_TO_PYTHON_MIGRATION.md`)](R_TO_PYTHON_MIGRATION.md)** — Side-by-side translation of R/Bioconductor/CRAN function calls and arguments into `pythonic_biotools`.
- **[Problem-Driven Use Cases (`USE_CASES_BY_PROBLEM.md`)](USE_CASES_BY_PROBLEM.md)** — Lookup guide matching biological & statistical questions to standalone tools and multi-tool workflows.
- **[Tool Comparison & Decision Matrix (`TOOL_COMPARISON_AND_DECISION_MATRIX.md`)](TOOL_COMPARISON_AND_DECISION_MATRIX.md)** — Side-by-side method comparison tables across all 8 domains.
- **[Diagnostic Plotting Gallery (`PLOTTING_GALLERY.md`)](PLOTTING_GALLERY.md)** — Built-in `.plot()` figures across copy-number, signature trajectory, super-enhancer, multimodal MR, motif enrichment, and clinical survival tools.
- **[70-Package Dual Parity Scorecard (`UPSTREAM_PARITY_AND_SCORECARD.md`)](UPSTREAM_PARITY_AND_SCORECARD.md)** — Complete audit of `settings="default"` vs. `settings="R_default"` behavioral toggles.
