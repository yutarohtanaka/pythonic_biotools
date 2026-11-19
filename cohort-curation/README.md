# `cohort-curation`: Public Genomics Database Clients & Harmonization Utilities

Standalone Python clients and file-format parsers for publicly available genomics resources:
- **cBioPortal REST API (`cbioportal.py`)**: Resilient retrieval of discrete GISTIC CNA matrices, categorical chromosome-arm assays, and sample/patient clinical metadata (Cerami et al., *Cancer Discovery* 2012).
- **Broad GDAC Firehose GISTIC2.0 & AACR Project GENIE (`gistic_and_panels.py`)**: Parser for `amp_genes`/`del_genes.conf_99.txt` wide peaks (Mermel et al., *Genome Biology* 2011) and `NaN` coverage masking for unsequenced targeted-panel genes (`data_gene_panel_*.txt`; AACR Project GENIE Consortium, 2017).
- **HGNC & ExPASy Cellosaurus (`hgnc_and_cellosaurus.py`)**: Deterministic HGNC symbol canonicalization, spreadsheet date-mangled gene repair (`1-Mar` $\to$ `MARCHF1`; Zeeberg et al., *Genome Biology* 2004), and streaming Cellosaurus QC parsing (Bairoch, 2018).
