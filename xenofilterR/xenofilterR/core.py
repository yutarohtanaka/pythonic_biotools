from .classify import classify_reads
from .io import write_filtered_bam
from .utils import SettingsMode, validate_settings


def run_xenofilter(
    human_bam,
    mouse_bam,
    output_bam,
    mismatch_cutoff=4,
    unmapped_penalty=8,
    settings: SettingsMode = "default",
):
    """
    Deconvolve xenograft reads and write human-specific alignments to output_bam.

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
    unmapped_penalty : float, default=8
        Fixed edit-distance penalty for an unmapped mate in ``settings="R_default"``.
    settings : {"default", "R_default"}, default="default"
        ``"default"`` preserves Pythonic behavior; ``"R_default"`` replicates
        ``PeeperLab/XenofilteR`` upstream R behavior.

    Returns
    -------
    pandas.DataFrame
        Per-read classification summary table.
    """
    validate_settings(settings)
    results_df = classify_reads(
        human_bam,
        mouse_bam,
        mismatch_cutoff=mismatch_cutoff,
        unmapped_penalty=unmapped_penalty,
        settings=settings,
    )
    human_reads = set(
        results_df[results_df["classification"] == "human"]["read_name"]
    )
    write_filtered_bam(human_bam, output_bam, human_reads, settings=settings)
    return results_df


class Xenofilter:
    """
    Configurable XenofilteR classifier and BAM deconvolver.

    Parameters
    ----------
    mismatch_cutoff : float, default=4
        Mismatch / edit-distance threshold (``MM_threshold`` in XenofilteR).
    unmapped_penalty : float, default=8
        Penalty assigned to an unmapped mate in paired-end ``settings="R_default"``
        mode (``Unmapped_penalty`` in XenofilteR).
    settings : {"default", "R_default"}, default="default"
        ``"default"`` uses the Pythonic edit-distance and read-length penalty model;
        ``"R_default"`` reproduces ``PeeperLab/XenofilteR`` ``R/private.R`` exactly.
    """

    def __init__(
        self,
        mismatch_cutoff: float = 4,
        unmapped_penalty: float = 8,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        self.mismatch_cutoff = mismatch_cutoff
        self.unmapped_penalty = unmapped_penalty
        self.settings: SettingsMode = settings

    def classify_reads(self, human_bam_path, mouse_bam_path):
        """Classify reads between ``human_bam_path`` and ``mouse_bam_path``."""
        return classify_reads(
            human_bam_path,
            mouse_bam_path,
            mismatch_cutoff=self.mismatch_cutoff,
            unmapped_penalty=self.unmapped_penalty,
            settings=self.settings,
        )

    def run(self, human_bam, mouse_bam, output_bam):
        """Deconvolve xenograft reads and write kept human records to ``output_bam``."""
        return run_xenofilter(
            human_bam,
            mouse_bam,
            output_bam,
            mismatch_cutoff=self.mismatch_cutoff,
            unmapped_penalty=self.unmapped_penalty,
            settings=self.settings,
        )

    def filter(self, human_bam, mouse_bam, output_bam):
        """Alias for :meth:`run`."""
        return self.run(human_bam, mouse_bam, output_bam)

    def __call__(self, human_bam, mouse_bam, output_bam=None):
        if output_bam is None:
            return self.classify_reads(human_bam, mouse_bam)
        return self.run(human_bam, mouse_bam, output_bam)
