from .classify import classify_reads
from .io import write_filtered_bam

def run_xenofilter(human_bam, mouse_bam, output_bam, mismatch_cutoff=5):
    results_df = classify_reads(human_bam, mouse_bam, mismatch_cutoff)
    human_reads = results_df[results_df["classification"] == "human"]["read_name"]
    write_filtered_bam(human_bam, output_bam, set(human_reads))