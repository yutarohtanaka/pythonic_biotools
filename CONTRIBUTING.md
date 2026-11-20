# Contributing to `pythonic_biotools`

Thank you for contributing to `pythonic_biotools`! This monorepo maintains standalone, dependency-light, PEP 561-typed Python implementations of published R, Bioconductor, CRAN, C++, and Fortran bioinformatics, oncology, clinical trial biostatistics, and statistical genetics methodologies—usable both individually and as a unified Python toolkit without `rpy2` or external R runtimes.

Every contribution must satisfy the five engineering and statistical contracts below.

---

## 1. Zero-Python-Duplicate Screening Criterion

Before implementing a new package, verify that **no actively maintained, idiomatic, feature-complete Python equivalent already exists** on PyPI or Bioconda:

- **In scope**: Published statistical genetics, cancer genomics, epigenomics, proteomics, spatial/single-cell, or clinical biostatistics algorithms locked inside R/Bioconductor/CRAN, C++/Fortran CLIs, or archived packages with missing/broken Conda/PyPI distributions (for example, `cTWAS`, `MutationTimeR`, `CONIPHER`, `JMbayes2`, `BuyseTest`, `DEqMS`, `LAVA`, `GenomicSEM`, `CSIDE`, `monaLisa`).
- **Out of scope**: Re-implementing algorithms already well-served by native Python ecosystems such as `scanpy`, `scvi-tools`, `pydeseq2`, `lifelines`, `scikit-survival`, `statsmodels`, or `gseapy`.
- Include the original peer-reviewed citation(s) and upstream reference repository URL in the PR description and package `README.md`.

---

## 2. Clean-Room Implementation & License Rules

Respecting upstream intellectual property and open-source licensing is mandatory:

1. **Clean-room MIT packages**: Implementations derived from peer-reviewed mathematical equations, published supplements, and permissively licensed (`MIT`, `BSD`, `Apache-2.0`) specifications are released under the `MIT` license.
2. **Copyleft-compatible standalone ports**: Any package translated from or structurally derived from a copyleft (`GPL` / `AGPL`) upstream R, C++, or Fortran codebase **must** carry a compatible copyleft SPDX license in its local `<pkg>/pyproject.toml` and `README.md` (for example: `GPL-2.0-or-later` in `timeroc`; `GPL-3.0-or-later` in `cause`, `mrpresso`, `mrraps`, `mtcojo`, and `smr-heidi`; `AGPL-3.0-or-later` in `evam`). Never relicense GPL/AGPL-derived code as MIT.
3. **Restricted academic/web tools**: Algorithms whose upstream code carries custom non-commercial or registration-gated licenses (e.g., `CIBERSORTx`) must be implemented strictly **clean-room** from the published peer-reviewed mathematical formulation without copying restricted source code, proprietary signature matrices, or bundled weights.

---

## 3. Flat Package Layout & PEP 561 `py.typed` Marker

Each tool resides in its own self-contained top-level directory `<pkg_dir>/` following a flat layout:

```text
<pkg_dir>/
├── pyproject.toml            # Standalone PEP 621 build metadata & SPDX license
├── README.md                 # Citation, math overview, quickstart, settings contract
├── <import_name>/
│   ├── __init__.py           # Public API with explicit __all__
│   ├── py.typed              # Empty PEP 561 marker file (required by CI)
│   ├── core.py               # Core statistical / algorithmic implementation
│   └── ...                   # Modular helpers (io.py, utils.py, etc.)
└── tests/
    └── test_<import_name>.py # Self-contained pytest unit & analytical parity tests
```

### Layout & Dependency Rules
- **Keep dependencies minimal**: Core packages should rely on `numpy>=1.24`, `scipy>=1.10`, and `pandas>=2.0`, adding `anndata>=0.10`, `scikit-learn>=1.3`, or `pysam>=0.22` only when domain-appropriate.
- **Accept standard Python data structures**: Public functions should accept `pandas.DataFrame`, `numpy.ndarray`, and `anndata.AnnData` (for single-cell/spatial tools), returning structured `@dataclass` results with `.to_dataframe()` / `.to_dict()` helpers where appropriate.
- **Protect CI globbing (`*/tests`)**: CI discovers standalone package unit suites via `for test_dir in */tests; do ...`. **Never** create a `tests/` subdirectory inside non-package top-level folders such as `docs/` or `notebooks/`. Cross-package integration tests belong in `tests_integration/`, and golden R parity tests belong in `benchmarks/r_parity/`.
- **Workspace registration**: Register new packages in:
  - Root `pyproject.toml` (`[tool.uv.workspace].members`)
  - `pythonic_biotools_cli.py` (`PACKAGE_IMPORT_MAP` and the appropriate domain tuple(s) in `BUNDLES`)
  - `docs/UPSTREAM_PARITY_AND_SCORECARD.md`, `docs/USE_CASES_BY_PROBLEM.md`, `docs/TOOL_COMPARISON_AND_DECISION_MATRIX.md`, and `docs/R_TO_PYTHON_MIGRATION.md`.

---

## 4. Dual `settings="default"` and `settings="R_default"` Parity Contract

Every primary estimation function and model class must accept:

```python
settings: Literal["default", "R_default"] = "default"
```

and validate it immediately (raising `ValueError` on unknown modes):

- **`settings="default"` (Statistically Hardened Pythonic Mode)**:
  Applies modern numerical and statistical safeguards—such as log-space `logsumexp` Bayes factors, positive-semidefinite eigenvalue floors on out-of-sample LD matrices, coding-invariant allele orientation, unfolded bootstrap standard errors near $\beta = 0$, and outlier-robust empirical Bayes shrinkage—while preserving asymptotic equivalence to the published estimator.
- **`settings="R_default"` (Strict Upstream R / C++ / Fortran Replication)**:
  Reproduces the exact equations, convergence tolerances, degrees-of-freedom adjustments, prior defaults, boundary clipping rules, and output column schemas of the upstream reference implementation (`>= 95%` source-to-source parity).
- Document every behavioral difference between `settings="default"` and `settings="R_default"` in `<pkg_dir>/README.md` and add explicit tests exercising both modes (plus invalid `settings` rejection).

---

## 5. Running Unit, Integration, and Golden Parity Tests

Before opening a pull request, run the verification checks below from the repository root:

```bash
# 1. Verify all 70+ standalone imports and PEP 561 py.typed markers
python pythonic_biotools_cli.py check-imports --bundle all

# 2. Run a specific standalone package's unit test suite
PYTHONPATH=mutationtimer pytest mutationtimer/tests -q

# 3. Run unit tests across every standalone package (matches GitHub Actions CI)
for test_dir in */tests; do
  pkg_dir="$(dirname "$test_dir")"
  PYTHONPATH="${pkg_dir}" pytest "${test_dir}" -q
done

# 4. Run end-to-end cross-package integration tests
pytest tests_integration -q

# 5. Run frozen upstream R_default golden parity benchmarks (zero warnings allowed)
pytest benchmarks/r_parity -W error -q

# 6. Validate documentation site build in strict mode (if editing docs/ or mkdocs.yml)
mkdocs build --strict
```

### Pull Request Checklist
- [ ] Passes zero-Python-duplicate check and cites original methodology paper(s).
- [ ] Includes `<import_name>/py.typed` and passes `pythonic_biotools_cli.py check-imports --bundle all`.
- [ ] Implements and tests both `settings="default"` and `settings="R_default"`.
- [ ] Adds deterministic `pytest` unit tests using seeded (`np.random.default_rng`) synthetic or analytical fixtures with fast execution (`< 10s` per package).
- [ ] Preserves upstream license compatibility in `<pkg_dir>/pyproject.toml`.
