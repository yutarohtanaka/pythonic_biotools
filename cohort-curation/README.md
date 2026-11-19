# `cohort-curation`: Cancer Genomics Database Clients & Cohort Harmonization

Standalone Python clients and file-format parsers (`cohort_curation`) for curating multi-institutional cancer genomics cohorts across **cBioPortal**, **Broad GDAC Firehose GISTIC2.0**, **AACR Project GENIE** targeted-sequencing panels, **HGNC** gene symbol nomenclature, and **ExPASy Cellosaurus** cell-line quality control.

---

## Published Citations

- **Cerami, E., et al.** (2012). The cBio Cancer Genomics Portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discovery*, 2(5), 401–404. https://doi.org/10.1158/2159-8290.CD-12-0095
- **Mermel, C. H., Schumacher, S. E., Hill, B., Meyerson, M. L., Beroukhim, R., & Getz, G.** (2011). GISTIC2.0 facilitates sensitive and confident localization of the targets of focal somatic copy-number alteration in human cancers. *Genome Biology*, 12(4), R41. https://doi.org/10.1186/gb-2011-12-4-r41
- **AACR Project GENIE Consortium.** (2017). AACR Project GENIE: Powering Precision Medicine through an International Consortium. *Cancer Discovery*, 7(8), 818–831. https://doi.org/10.1158/2159-8290.CD-17-0151
- **Zeeberg, B. R., Riss, J., Kane, D. W., Bussey, K. J., Uchio, E., Linehan, W. M., Barrett, J. C., & Weinstein, J. N.** (2004). Mistaken Identifiers: gene name errors can be introduced inadvertently when using Excel in bioinformatics. *BMC Bioinformatics*, 5, 80. https://doi.org/10.1186/1471-2105-5-80
- **Bairoch, A.** (2018). The Cellosaurus, a cell-line knowledge resource. *Journal of Biomolecular Techniques*, 29(2), 25–38. https://doi.org/10.7171/jbt.18-2902-002

---

## Algorithmic & Data-Curation Principles

1. **Targeted-Panel Coverage Masking (`build_genie_panel_matrix`, `mask_unprofiled_panel_genes`)**:
   In multi-panel clinical sequencing cohorts (AACR GENIE, MSK-IMPACT v3–v7, DFCI-ONCOPANEL), absence of a mutation call in MAF output conflates *wild-type ($0$)* with *locus not sequenced on panel ($\text{NaN}$)*. `build_genie_panel_matrix` parses `data_gene_panel_*.txt` definitions (supporting both HUGO `gene_list:` and Entrez `entrez_gene_list:` formats) into a binary `panels x genes` indicator matrix $C$, and `mask_unprofiled_panel_genes` sets $A_{s,g} \leftarrow \text{NaN}$ wherever $C_{\text{panel}(s), g} = 0$.

2. **Deterministic HGNC Canonicalization & Spreadsheet Repair (`harmonize_hgnc_symbols`, `repair_excel_mangled_genes`)**:
   Resolves legacy microarray/RNA-seq symbols using strict precedence: (1) Entrez Gene ID (`8085`, `8085.0`, `///`-delimited), (2) missing value preservation (`NaN`/`None` never stringified to `'nan'`), (3) `EXCEL_MANGLED_GENE_MAP` repair (`1-Mar` $\to$ `MARCHF1`, `7-Sep` $\to$ `SEPTIN7`, `1-Dec` $\to$ `DELEC1`, `15-Sep` $\to$ `SELENOF`, `2-Apr` $\to$ `PMAIP1`, `OCT4`/`4-Oct` $\to$ `POU5F1`, `MARC1/2` $\to$ `MTARC1/2`), (4) approved HGNC symbol, (5) unambiguous 1-to-1 `prev_symbol`, and (6) non-colliding unambiguous `alias_symbol`.

3. **Zero-Preserving cBioPortal GISTIC Retrieval (`fetch_discrete_cna`, `fetch_arm_level_cna`, `binarize_arm_events`)**:
   cBioPortal `/molecular-data/fetch` omits diploid (`0`) sample–gene pairs; `fetch_discrete_cna` queries the full `/sample-lists/{id}/sample-ids` universe and reindexes with `fill_value=0`, with HTTP `429 Retry-After` and exponential-backoff retry resilience.

---

## Quickstart

```python
import pandas as pd
from cohort_curation import (
    build_genie_panel_matrix,
    harmonize_hgnc_symbols,
    mask_unprofiled_panel_genes,
    repair_excel_mangled_genes,
)

# 1. Repair Excel date-corrupted and legacy cancer gene symbols
clean = repair_excel_mangled_genes(["1-Mar", "2-Apr", "OCT4", "MARC1", "ENSG00000141510_TP53"])
# -> ['MARCHF1', 'PMAIP1', 'POU5F1', 'MTARC1', 'TP53']

# 2. Harmonize platform gene symbols in Bioconductor HGNChelper parity mode
hgnc_df = pd.DataFrame(
    [{"symbol": "KMT2D", "status": "Approved", "prev_symbol": "MLL2", "alias_symbol": "ALR", "entrez_id": "8085"}]
)
mapped = harmonize_hgnc_symbols(pd.Series(["MLL2", "2-Apr", "UNMAPPED_PROBE"]), hgnc_df, settings="R_default")
# -> ['KMT2D', 'PMAIP1', NaN]

# 3. Parse GENIE panels (HUGO or Entrez gene lists) & mask unassayed genes to NaN
cov = build_genie_panel_matrix("path/to/genie_panels", entrez_to_symbol=hgnc_df, settings="R_default")
masked_alt = mask_unprofiled_panel_genes(alteration_df, sample_to_panel_map, cov)
```

---

## `settings="default"` vs `settings="R_default"` Reference

`harmonize_hgnc_symbols` and `build_genie_panel_matrix` accept `settings: Literal["default", "R_default"] = "default"` (raising `ValueError` on unknown values).

| Function | `settings="default"` | `settings="R_default"` (`HGNChelper` / `MSKCC-Epi-Bio/gnomeR`) |
|---|---|---|
| **`harmonize_hgnc_symbols`** | Unrecognized symbols are preserved unchanged (`unmapped_as_na=False`) | Unrecognized symbols map to `NaN` (`HGNChelper::checkGeneSymbols(..., unmapped.as.na=TRUE)`) |
| **`build_genie_panel_matrix`** | Parses `gene_list:` and `entrez_gene_list:` (mapping Entrez IDs if `entrez_to_symbol` supplied), preserving symbol spelling verbatim | Automatically canonicalizes legacy/Excel-mangled panel symbols (`MARC1` $\to$ `MTARC1`, `OCT4` $\to$ `POU5F1`, `2-Apr` $\to$ `PMAIP1`, `MLL2` $\to$ `KMT2D` when HGNC table supplied) per `gnomeR` |

---

## Public API & Output Schema

- **HGNC & ExPASy Cellosaurus (`hgnc_and_cellosaurus.py`)**:
  - `EXCEL_MANGLED_GENE_MAP`: `dict[str, str]`
  - `repair_excel_mangled_genes(symbols, strip_ensembl_prefix=True) -> list[str]`
  - `harmonize_hgnc_symbols(symbols, hgnc_table, entrez_ids=None, *, unmapped_as_na=None, settings="default") -> pd.Series`
  - `parse_cellosaurus(cellosaurus_txt_path, depmap_ids=None, cvcl_ids=None, cell_line_names=None) -> pd.DataFrame`
    - Columns: `cell_line_name`, `cvcl_accession`, `depmap_id`, `is_problematic`, `problematic_comment`, `msi_comment`, `doubling_time_hours`.
- **GISTIC2.0 & GENIE Panels (`gistic_and_panels.py`)**:
  - `download_gistic_peak_files(study_code, run_date="20160128", cache_dir="data/firehose_cache", tumor_type_code="TP") -> tuple[Path, Path]` (traversal-safe tar extraction)
  - `parse_gistic_peak_genes(path) -> dict[str, set[str]]`
  - `cross_check_gistic_panel(panel_genes, amp_peaks, del_peaks) -> pd.DataFrame` (`gene`, `in_significant_amp_peak`, `in_significant_del_peak`, `amp_cytobands`, `del_cytobands`)
  - `build_genie_panel_matrix(panel_dir_or_files, entrez_to_symbol=None, *, settings="default") -> pd.DataFrame` (`panels x genes` `int8` indicator matrix)
  - `mask_unprofiled_panel_genes(alteration_matrix, sample_to_panel, panel_coverage_matrix) -> pd.DataFrame`
- **cBioPortal REST Client (`cbioportal.py`)**:
  - `resolve_entrez_ids`, `fetch_discrete_cna`, `fetch_arm_level_cna`, `binarize_arm_events`, `fetch_sample_clinical_data`, `fetch_clinical_attribute`

---

## Testing

```bash
PYTHONPATH=cohort-curation /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest cohort-curation/tests -v
```

## License

MIT.
