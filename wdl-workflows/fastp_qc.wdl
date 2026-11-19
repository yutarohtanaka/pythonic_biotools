version 1.0

workflow fastp_qc {
    meta {
        version: "1.0"
        author: "Yutaro Tanaka"
        description: "Generalized Terra / Cromwell WDL workflow for paired-end FASTQ read QC and adapter trimming with fastp."
    }

    input {
        String sample_name
        File fastq_r1
        File fastq_r2
        String extra_fastp_args = ""
        String docker = "staphb/fastp:latest"
        Int cpu = 8
        Int memory_gb = 32
        Int disk_gb = 50
    }

    call run_fastp {
        input:
            sample_name = sample_name,
            fastq_r1 = fastq_r1,
            fastq_r2 = fastq_r2,
            extra_fastp_args = extra_fastp_args,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb
    }

    output {
        File trimmed_fastq_r1 = run_fastp.trimmed_fastq_r1
        File trimmed_fastq_r2 = run_fastp.trimmed_fastq_r2
        File fastp_json = run_fastp.fastp_json
        File fastp_html = run_fastp.fastp_html
    }
}

task run_fastp {
    input {
        String sample_name
        File fastq_r1
        File fastq_r2
        String extra_fastp_args = ""
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        fastp \
            -w ~{cpu} \
            -i "~{fastq_r1}" \
            -I "~{fastq_r2}" \
            -o "~{sample_name}_R1.trimmed.fastq.gz" \
            -O "~{sample_name}_R2.trimmed.fastq.gz" \
            --json "~{sample_name}.fastp.json" \
            --html "~{sample_name}.fastp.html" \
            ~{extra_fastp_args}
    >>>

    output {
        File trimmed_fastq_r1 = "~{sample_name}_R1.trimmed.fastq.gz"
        File trimmed_fastq_r2 = "~{sample_name}_R2.trimmed.fastq.gz"
        File fastp_json = "~{sample_name}.fastp.json"
        File fastp_html = "~{sample_name}.fastp.html"
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} HDD"
    }
}
