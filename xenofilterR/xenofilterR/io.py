import pysam
from .utils import SettingsMode, validate_settings


def load_bam(bam_path):
    """Open a BAM file for reading (supports use as a context manager)."""
    return pysam.AlignmentFile(bam_path, "rb")


def write_filtered_bam(
    human_bam_path,
    output_bam_path,
    read_names_to_keep,
    settings: SettingsMode = "default",
):
    """
    Write primary (and mate) BAM records whose query_name is in read_names_to_keep.
    Secondary alignments (and supplementary alignments when settings="default") are
    excluded, and file handles are safely closed via context managers.
    """
    validate_settings(settings)
    keep_set = set(read_names_to_keep)
    with pysam.AlignmentFile(human_bam_path, "rb") as input_bam:
        with pysam.AlignmentFile(output_bam_path, "wb", template=input_bam) as output_bam:
            for read in input_bam.fetch(until_eof=True):
                if read.is_secondary:
                    continue
                if settings == "default" and read.is_supplementary:
                    continue
                if read.query_name in keep_set:
                    output_bam.write(read)
