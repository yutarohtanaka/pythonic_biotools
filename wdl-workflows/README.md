# `wdl-workflows`: Parameterized Terra / Cromwell WDL 1.0 Genomics Workflows

Production-ready, containerized **WDL 1.0** workflows for **Terra**, **AnVIL**, **Cromwell**, and **MiniWDL** execution, wrapping paired-end sequencing quality control (`fastp_qc.wdl`) and streaming patient-derived xenograft (**PDX**) graft-versus-host BAM read deconvolution (`xenofilter_bam.wdl`).

---

## Published Citations

- **Chen, S., Zhou, Y., Chen, Y., & Gu, J.** (2018). fastp: an ultra-fast all-in-one FASTQ preprocessor. *Bioinformatics*, 34(17), i884–i890. https://doi.org/10.1093/bioinformatics/bty560
- **Kluin, R. J. C., Kemper, K., Kuilman, T., de Ruiter, J. R., Iyer, V., Forment, J. V., Cornelissen-Steijger, P., de Rink, I., ter Brugge, P., Song, J.-Y., Klarenbeek, S., McDermott, U., Jonkers, J., Velds, A., Adams, D. J., Peeper, D. S., & Krijgsman, O.** (2018). XenofilteR: computational deconvolution of mouse and human reads in tumor xenograft sequence data. *BMC Bioinformatics*, 19, 366. https://doi.org/10.1186/s12859-018-2353-5
- **Danecek, P., et al.** (2021). Twelve years of SAMtools and BCFtools. *GigaScience*, 10(2), giab008. https://doi.org/10.1093/gigascience/giab008

---

## Algorithmic Formulation

### 1. `fastp_qc.wdl` (`workflow fastp_qc`)
Executes multi-threaded `fastp` paired-end adapter autodetection, polyG tail trimming, base quality filtering, and comprehensive JSON/HTML QC report generation in a reproducible Docker runtime (`staphb/fastp:latest`).

### 2. `xenofilter_bam.wdl` (`workflow xenofilter_bam`)
Implements streaming $O(1)$-RAM BAM-level PDX read deconvolution matching the published **`XenofilteR`** edit-distance scoring model across RNA-seq, ATAC-seq, ChIP-seq, WES, and WGS libraries:

1. **Primary Alignment Edit-Distance Extraction**:
   Streams primary mapped alignments (`samtools view -F 2308`, optionally recomputing `NM` via `samtools fillmd` when `graft_reference_fasta` / `host_reference_fasta` are supplied) and computes per-read edit distance including soft (`S`) and hard (`H`) CIGAR clips:
   $$e(r) = \text{NM}(r) + L_{\text{soft-clip}}(r) + L_{\text{hard-clip}}(r)$$

2. **Mate-Averaged Scoring & Unmapped-Mate Penalty**:
   - **Both mates mapped** (`cnt >= 2`): $S(q) = \frac{1}{\text{cnt}} \sum_{m} e(r_m)$.
   - **Paired-end with one unmapped mate** (`paired == 1`, `cnt == 1`): penalizes the missing mate by $\text{pen} = \max(L_{\text{read}}, 2 \cdot \text{cutoff} + 2)$, yielding $S(q) = \frac{e(r_1) + \text{pen}}{2}$ so single-mate alignments cannot spuriously pass a strict `mismatch_cutoff`.
   - **Single-end read** (`paired == 0`): $S(q) = e(r)$.

3. **Classification Decision Rule**:
   Joining graft score $S_{\text{graft}}(q)$ and host score $S_{\text{host}}(q)$ (`inf` if unmapped to a species):
   - **`graft`**: $S_{\text{graft}}(q) < S_{\text{host}}(q)$ **and** $S_{\text{graft}}(q) \le \text{mismatch\_cutoff}$ (default `4`).
   - **`host`**: $S_{\text{host}}(q) < S_{\text{graft}}(q)$ **and** $S_{\text{host}}(q) \le \text{mismatch\_cutoff}$.
   - **`ambiguous`**: tied edit distances ($S_{\text{graft}} = S_{\text{host}}$) or both species exceed `mismatch_cutoff`.
   Only strictly `graft`-classified read templates (`samtools view -F 2304 -N`) are coordinate-sorted and indexed into `clean_graft_bam`.

---

## Quickstart (`inputs.json` & Execution)

### `fastp_qc_inputs.json`
```json
{
  "fastp_qc.sample_name": "PDX_Sample_01",
  "fastp_qc.fastq_r1": "gs://bucket/raw/PDX_Sample_01_R1.fastq.gz",
  "fastp_qc.fastq_r2": "gs://bucket/raw/PDX_Sample_01_R2.fastq.gz",
  "fastp_qc.extra_fastp_args": "--detect_adapter_for_pe --trim_poly_g",
  "fastp_qc.cpu": 8,
  "fastp_qc.memory_gb": 32,
  "fastp_qc.disk_gb": 50
}
```

### `xenofilter_bam_inputs.json`
```json
{
  "xenofilter_bam.sample_name": "PDX_Sample_01",
  "xenofilter_bam.graft_bam": "gs://bucket/aligned/PDX_Sample_01.GRCh38.bam",
  "xenofilter_bam.host_bam": "gs://bucket/aligned/PDX_Sample_01.GRCm39.bam",
  "xenofilter_bam.mismatch_cutoff": 4,
  "xenofilter_bam.cpu": 8,
  "xenofilter_bam.memory_gb": 64,
  "xenofilter_bam.disk_gb": 100
}
```

### Running Locally with Cromwell or MiniWDL
```bash
miniwdl run wdl-workflows/fastp_qc.wdl -i fastp_qc_inputs.json
miniwdl run wdl-workflows/xenofilter_bam.wdl -i xenofilter_bam_inputs.json
```

---

## Workflow Inputs & Outputs Reference

### `fastp_qc.wdl`
| Direction | Parameter | Type | Default | Description |
|---|---|---|---|---|
| **Input** | `sample_name` | `String` | *(Required)* | Output file prefix |
| **Input** | `fastq_r1`, `fastq_r2` | `File` | *(Required)* | Paired-end input FASTQ(`.gz`) files |
| **Input** | `extra_fastp_args` | `String` | `""` | Additional CLI flags forwarded to `fastp` |
| **Input** | `docker` | `String` | `"staphb/fastp:latest"` | Container image |
| **Input** | `cpu`, `memory_gb`, `disk_gb` | `Int` | `8`, `32`, `50` | Terra / Cromwell runtime resources |
| **Output** | `trimmed_fastq_r1`, `trimmed_fastq_r2` | `File` | — | Adapter- and quality-trimmed paired-end `.fastq.gz` |
| **Output** | `fastp_json`, `fastp_html` | `File` | — | Machine-readable JSON & interactive HTML QC reports |

### `xenofilter_bam.wdl`
| Direction | Parameter | Type | Default | Description |
|---|---|---|---|---|
| **Input** | `sample_name` | `String` | *(Required)* | Output prefix |
| **Input** | `graft_bam`, `host_bam` | `File` | *(Required)* | Reads aligned to graft (e.g. human) and host (e.g. mouse) |
| **Input** | `graft_reference_fasta`, `host_reference_fasta` | `File?` | `None` | Optional FASTAs to recompute `NM` tags via `samtools fillmd` |
| **Input** | `mismatch_cutoff` | `Int` | `4` | Maximum allowed mate-averaged edit distance (`XenofilteR` `MM_threshold`) |
| **Input** | `docker` | `String` | `"mgibio/samtools:1.16.1"` | Container image |
| **Input** | `cpu`, `memory_gb`, `disk_gb` | `Int` | `8`, `64`, `100` | Terra / Cromwell runtime resources |
| **Output** | `classification_csv` | `File` | — | Per-template CSV (`read_name,graft_mismatches,host_mismatches,classification`) |
| **Output** | `clean_graft_bam`, `clean_graft_bam_index` | `File` | — | Coordinate-sorted graft-specific `.bam` and `.bam.bai` |

---

## Validation

Validate WDL 1.0 syntax with `miniwdl check` or `womtool validate`:
```bash
miniwdl check wdl-workflows/fastp_qc.wdl wdl-workflows/xenofilter_bam.wdl
```
