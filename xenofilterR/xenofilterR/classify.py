import pandas as pd
from .io import load_bam
from .utils import count_mismatches, read_pairs

def classify_reads(human_bam_path, mouse_bam_path, mismatch_cutoff=5):
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
