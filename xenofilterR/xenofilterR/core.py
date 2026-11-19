from .classify import classify_reads
from .io import write_filtered_bam


def run_xenofilter(human_bam, mouse_bam, output_bam, mismatch_cutoff=4):
    """
    Deconvolve xenograft reads and write human-specific primary alignments to output_bam.

    Parameters
    ----------
    human_bam : str or PathLike
        Path to human-aligned BAM file.
    mouse_bam : str or PathLike
        Path to mouse-aligned BAM file.
    output_bam : str or PathLike
        Destination path for filtered human BAM file.
    mismatch_cutoff : float, default=4
        Maximum allowed per-read / mate-averaged edit distance (Kluin et al. 2018).

    Returns
    -------
    pandas.DataFrame
        Per-read classification summary table.
    """
    results_df = classify_reads(human_bam, mouse_bam, mismatch_cutoff=mismatch_cutoff)
    human_reads = set(results_df[results_df["classification"] == "human"]["read_name"])
    write_filtered_bam(human_bam, output_bam, human_reads)
    return results_df
