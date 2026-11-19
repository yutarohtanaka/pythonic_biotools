version 1.0

workflow xenofilter_bam {
    meta {
        version: "1.0"
        author: "Yutaro Tanaka"
        description: "Generalized Terra / Cromwell WDL workflow for streaming BAM-level XenofilteR graft/host read deconvolution (RNA-seq, ATAC-seq, ChIP-seq, WES/WGS)."
    }

    input {
        String sample_name
        File graft_bam
        File host_bam
        File? graft_reference_fasta
        File? host_reference_fasta
        Int mismatch_cutoff = 5
        String docker = "mgibio/samtools:1.16.1"
        Int cpu = 8
        Int memory_gb = 64
        Int disk_gb = 100
    }

    call classify_xenograft_reads {
        input:
            sample_name = sample_name,
            graft_bam = graft_bam,
            host_bam = host_bam,
            graft_reference_fasta = graft_reference_fasta,
            host_reference_fasta = host_reference_fasta,
            mismatch_cutoff = mismatch_cutoff,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb
    }

    output {
        File classification_csv = classify_xenograft_reads.classification_csv
        File clean_graft_bam = classify_xenograft_reads.clean_graft_bam
        File clean_graft_bam_index = classify_xenograft_reads.clean_graft_bam_index
    }
}

task classify_xenograft_reads {
    input {
        String sample_name
        File graft_bam
        File host_bam
        File? graft_reference_fasta
        File? host_reference_fasta
        Int mismatch_cutoff = 5
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        GRAFT_BAM="~{graft_bam}"
        HOST_BAM="~{host_bam}"
        OUTPUT_CSV="~{sample_name}_xenofilter_classification.csv"
        MISMATCH_CUTOFF="~{mismatch_cutoff}"

        TMP_DIR=$(mktemp -d)
        trap 'rm -rf -- "$TMP_DIR"' EXIT

        GRAFT_MM_SUM="$TMP_DIR/graft_mm.sum.txt"
        HOST_MM_SUM="$TMP_DIR/host_mm.sum.txt"
        COMBINED_MM="$TMP_DIR/combined.txt"

        process_bam() {
            local bam_file="$1"
            local output_file="$2"
            local ref_fasta="$3"

            if [ -n "$ref_fasta" ]; then
                samtools fillmd -b "$bam_file" "$ref_fasta" | \
                    samtools view -F 2308 - | \
                    awk '{for(i=12;i<=NF;i++) if($i ~ /^NM:i:/) {print $1"\t"substr($i,6)}}' | \
                    awk -F'\t' '{a[$1]+=$2} END{for(k in a) print k"\t"a[k]}' > "$output_file"
            else
                samtools view -F 2308 "$bam_file" | \
                    awk '{for(i=12;i<=NF;i++) if($i ~ /^NM:i:/) {print $1"\t"substr($i,6)}}' | \
                    awk -F'\t' '{a[$1]+=$2} END{for(k in a) print k"\t"a[k]}' > "$output_file"
            fi
        }

        process_bam "$GRAFT_BAM" "$GRAFT_MM_SUM" "~{default='' graft_reference_fasta}"
        process_bam "$HOST_BAM" "$HOST_MM_SUM" "~{default='' host_reference_fasta}"

        join -t $'\t' -a 1 -a 2 -e "inf" -o '0,1.2,2.2' \
            <(sort -k1,1 "$GRAFT_MM_SUM") \
            <(sort -k1,1 "$HOST_MM_SUM") > "$COMBINED_MM"

        awk -v cutoff="$MISMATCH_CUTOFF" '
        BEGIN {
            FS="\t"; OFS=",";
            print "read_name,graft_mismatches,host_mismatches,classification";
        }
        {
            read_name = $1;
            g_mm_str = $2;
            h_mm_str = $3;
            classification = "ambiguous";

            g_mm = (g_mm_str == "inf") ? 999999 : g_mm_str + 0;
            h_mm = (h_mm_str == "inf") ? 999999 : h_mm_str + 0;

            if (g_mm < h_mm && g_mm <= cutoff) {
                classification = "graft";
            } else if (h_mm < g_mm && h_mm <= cutoff) {
                classification = "host";
            }
            print read_name, g_mm_str, h_mm_str, classification;
        }' "$COMBINED_MM" > "$OUTPUT_CSV"

        GRAFT_READ_NAMES="$TMP_DIR/graft_read_names.txt"
        tail -n +2 "$OUTPUT_CSV" | awk -F',' '$4 == "graft" {print $1}' > "$GRAFT_READ_NAMES"

        CLEAN_GRAFT_BAM="~{sample_name}.graft_specific.bam"
        samtools view -@ ~{cpu} -b -N "$GRAFT_READ_NAMES" -o "$TMP_DIR/unsorted.bam" "$GRAFT_BAM"
        samtools sort -@ ~{cpu} -o "$CLEAN_GRAFT_BAM" "$TMP_DIR/unsorted.bam"
        samtools index -@ ~{cpu} "$CLEAN_GRAFT_BAM"
    >>>

    output {
        File classification_csv = "~{sample_name}_xenofilter_classification.csv"
        File clean_graft_bam = "~{sample_name}.graft_specific.bam"
        File clean_graft_bam_index = "~{sample_name}.graft_specific.bam.bai"
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} HDD"
    }
}
