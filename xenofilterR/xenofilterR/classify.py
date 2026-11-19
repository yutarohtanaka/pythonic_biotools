import pandas as pd
from .io import load_bam
from .utils import (
    SettingsMode,
    count_mismatches,
    read_pairs,
    score_pair,
    validate_settings,
)


def _mapped_mate_eds(r1, r2, settings: SettingsMode) -> list[float]:
    eds: list[float] = []
    for r in (r1, r2):
        if r is not None and not r.is_unmapped:
            eds.append(float(count_mismatches(r, settings=settings)))
    return eds


def classify_reads(
    human_bam_path,
    mouse_bam_path,
    mismatch_cutoff=4,
    unmapped_penalty=8,
    settings: SettingsMode = "default",
):
    """
    Classify sequencing reads as 'human', 'mouse', or 'ambiguous' by comparing
    per-read / mate-averaged edit distance between human (graft) and mouse (host)
    BAM alignments.

    Parameters
    ----------
    human_bam_path : str or PathLike
        Path to BAM aligned against the human (graft) reference.
    mouse_bam_path : str or PathLike
        Path to BAM aligned against the mouse (host) reference.
    mismatch_cutoff : float, default=4
        Maximum allowed per-read (single-end) or mate-averaged / per-mate
        (paired-end) edit distance, matching Kluin et al. (2018) MM_threshold = 4.
    unmapped_penalty : float, default=8
        Fixed edit-distance penalty assigned to an unmapped mate in a pair when
        ``settings="R_default"`` (matching ``XenofilteR.R`` ``Unmapped_penalty = 8``).
    settings : {"default", "R_default"}, default="default"
        ``"default"`` preserves Pythonic scoring and ``<= mismatch_cutoff`` gating;
        ``"R_default"`` replicates ``PeeperLab/XenofilteR`` ``private.R`` line-for-line.

    Returns
    -------
    pandas.DataFrame
        Columns: ['read_name', 'human_mm', 'mouse_mm', 'classification'].
    """
    validate_settings(settings)

    with load_bam(human_bam_path) as human_bam:
        human_reads = read_pairs(human_bam, settings=settings)
    with load_bam(mouse_bam_path) as mouse_bam:
        mouse_reads = read_pairs(mouse_bam, settings=settings)

    all_read_names = list(
        dict.fromkeys(list(human_reads.keys()) + list(mouse_reads.keys()))
    )

    results = []
    for read_name in all_read_names:
        h1, h2 = human_reads.get(read_name, (None, None))
        m1, m2 = mouse_reads.get(read_name, (None, None))

        h_mm = score_pair(
            h1,
            h2,
            mismatch_cutoff=mismatch_cutoff,
            unmapped_penalty=unmapped_penalty,
            settings=settings,
        )
        m_mm = score_pair(
            m1,
            m2,
            mismatch_cutoff=mismatch_cutoff,
            unmapped_penalty=unmapped_penalty,
            settings=settings,
        )

        classification = "ambiguous"
        if settings == "default":
            if h_mm < m_mm and h_mm <= mismatch_cutoff:
                classification = "human"
            elif m_mm < h_mm and m_mm <= mismatch_cutoff:
                classification = "mouse"
        else:
            # Exact PeeperLab/XenofilteR private.R gating
            any_rec = next(
                (r for r in (h1, h2, m1, m2) if r is not None and not r.is_unmapped),
                None,
            )
            is_paired = bool(any_rec.is_paired) if any_rec is not None else False
            if not is_paired:
                # Single-end: strict < mismatch_cutoff and strictly better than host
                if h_mm < m_mm and h_mm < mismatch_cutoff:
                    classification = "human"
                elif m_mm < h_mm and m_mm < mismatch_cutoff:
                    classification = "mouse"
            else:
                h_eds = _mapped_mate_eds(h1, h2, settings="R_default")
                m_eds = _mapped_mate_eds(m1, m2, settings="R_default")
                if h_eds and not m_eds:
                    # Graft-only pair: keep if at least one mapped human mate has ed < cutoff
                    if min(h_eds) < mismatch_cutoff:
                        classification = "human"
                elif m_eds and not h_eds:
                    if min(m_eds) < mismatch_cutoff:
                        classification = "mouse"
                elif h_eds and m_eds:
                    # Mapped in both graft & host: mean pair ED strictly better AND every mapped mate < cutoff
                    if h_mm < m_mm and max(h_eds) < mismatch_cutoff:
                        classification = "human"
                    elif m_mm < h_mm and max(m_eds) < mismatch_cutoff:
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
