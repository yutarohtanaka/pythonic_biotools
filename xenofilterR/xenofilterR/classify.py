import pandas as pd
from .io import load_bam
from .utils import read_pairs, score_pair


def classify_reads(human_bam_path, mouse_bam_path, mismatch_cutoff=4):
    """
    Classify sequencing reads as 'human', 'mouse', or 'ambiguous' by comparing
    per-read / mate-averaged edit distance (NM + soft/hard CIGAR clips) between
    human (graft) and mouse (host) BAM alignments.

    Parameters
    ----------
    human_bam_path : str or PathLike
        Path to BAM aligned against the human (graft) reference.
    mouse_bam_path : str or PathLike
        Path to BAM aligned against the mouse (host) reference.
    mismatch_cutoff : float, default=4
        Maximum allowed per-read (single-end) or mate-averaged (paired-end) edit
        distance, matching Kluin et al. (2018) MM_threshold = 4.

    Returns
    -------
    pandas.DataFrame
        Columns: ['read_name', 'human_mm', 'mouse_mm', 'classification'].
    """
    with load_bam(human_bam_path) as human_bam:
        human_reads = read_pairs(human_bam)
    with load_bam(mouse_bam_path) as mouse_bam:
        mouse_reads = read_pairs(mouse_bam)

    all_read_names = list(
        dict.fromkeys(list(human_reads.keys()) + list(mouse_reads.keys()))
    )

    results = []
    for read_name in all_read_names:
        h1, h2 = human_reads.get(read_name, (None, None))
        m1, m2 = mouse_reads.get(read_name, (None, None))

        h_mm = score_pair(h1, h2, mismatch_cutoff=mismatch_cutoff)
        m_mm = score_pair(m1, m2, mismatch_cutoff=mismatch_cutoff)

        classification = "ambiguous"
        if h_mm < m_mm and h_mm <= mismatch_cutoff:
            classification = "human"
        elif m_mm < h_mm and m_mm <= mismatch_cutoff:
            classification = "mouse"

        results.append(
            {
                "read_name": read_name,
                "human_mm": h_mm,
                "mouse_mm": m_mm,
                "classification": classification,
            }
        )

    return pd.DataFrame(
        results, columns=["read_name", "human_mm", "mouse_mm", "classification"]
    )
