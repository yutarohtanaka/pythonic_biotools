import pysam
import pandas as pd
import numpy as np
import os

def load_bam(bam_path):
    return pysam.AlignmentFile(bam_path, "rb")

def count_mismatches(read):
    # Parse NM tag (edit distance)
    return read.get_tag("NM") if read.has_tag("NM") else 0

def read_pairs(bam):
    read_dict = {}
    for read in bam.fetch(until_eof=True):
        if not read.is_unmapped and read.is_paired and read.query_name:
            if read.query_name not in read_dict:
                read_dict[read.query_name] = [None, None]
            if read.is_read1:
                read_dict[read.query_name][0] = read
            elif read.is_read2:
                read_dict[read.query_name][1] = read
    return read_dict

def filter_reads(human_bam_path, mouse_bam_path, mismatch_cutoff=5):
    human_bam = load_bam(human_bam_path)
    mouse_bam = load_bam(mouse_bam_path)
    
    human_reads = read_pairs(human_bam)
    mouse_reads = read_pairs(mouse_bam)

    results = []

    for read_name, (h1, h2) in human_reads.items():
        m1, m2 = mouse_reads.get(read_name, (None, None))
        
        h_mm = (count_mismatches(h1) if h1 else 0) + (count_mismatches(h2) if h2 else 0)
        m_mm = (count_mismatches(m1) if m1 else 0) + (count_mismatches(m2) if m2 else 0)

        classification = "ambiguous"
        if h_mm < m_mm and h_mm <= mismatch_cutoff:
            classification = "human"
        elif m_mm < h_mm and m_mm <= mismatch_cutoff:
            classification = "mouse"

        results.append({
            "read_name": read_name,
            "human_mm": h_mm,
            "mouse_mm": m_mm,
            "classification": classification
        })

    return pd.DataFrame(results)

def write_filtered_bam(human_bam_path, output_bam_path, read_names_to_keep):
    input_bam = load_bam(human_bam_path)
    output_bam = pysam.AlignmentFile(output_bam_path, "wb", template=input_bam)

    for read in input_bam.fetch(until_eof=True):
        if read.query_name in read_names_to_keep:
            output_bam.write(read)

    output_bam.close()
    input_bam.close()

# Example usage:
# results_df = filter_reads("human.bam", "mouse.bam")
# human_only = results_df[results_df["classification"] == "human"]["read_name"]
# write_filtered_bam("human.bam", "filtered_human.bam", set(human_only))
