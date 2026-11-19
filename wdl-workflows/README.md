# Terra / Cromwell WDL Workflows (`wdl-workflows`)

Parameterized WDL 1.0 workflows wrapping standard open-source command-line bioinformatics tools for Terra, AnVIL, and Cromwell execution.

| Workflow File | Tool & Citation | Purpose |
|---|---|---|
| `fastp_qc.wdl` | `fastp` (Chen et al., *Bioinformatics* 2018) | Paired-end FASTQ adapter trimming, polyG removal, and JSON/HTML QC reporting. |
| `xenofilter_bam.wdl` | `XenofilteR` (Kluin et al., *BMC Bioinformatics* 2018) | Streaming BAM-level graft vs. host read classification by mate-averaged edit distance (`NM` + soft/hard CIGAR clips, with unmapped-mate penalty). |
