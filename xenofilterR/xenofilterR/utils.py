from typing import Literal

BAM_CINS = 1
BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate the parity mode setting."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def count_mismatches(read, settings: SettingsMode = "default"):
    """
    Compute per-read edit distance following Kluin et al. (BMC Bioinformatics 2018):
        - settings="default": edit_distance = NM + soft_clipped_bases + hard_clipped_bases
        - settings="R_default": edit_distance = NM + soft_clipped_bases + insertions
          (matching PeeperLab/XenofilteR private.R .Count_MM L103-149)

    Returns float("inf") if the read is None or unmapped.
    """
    validate_settings(settings)
    if read is None or read.is_unmapped:
        return float("inf")

    if read.has_tag("NM"):
        nm = int(read.get_tag("NM"))
    elif read.has_tag("nM"):
        nm = int(read.get_tag("nM"))
    else:
        nm = 0

    if settings == "R_default":
        extra = sum(
            length
            for op, length in (read.cigartuples or ())
            if op in (BAM_CSOFT_CLIP, BAM_CINS)
        )
        return nm + extra

    clipped = sum(
        length
        for op, length in (read.cigartuples or ())
        if op in (BAM_CSOFT_CLIP, BAM_CHARD_CLIP)
    )
    return nm + clipped


def score_pair(
    r1,
    r2,
    mismatch_cutoff=4,
    unmapped_penalty=8,
    settings: SettingsMode = "default",
):
    """
    Compute the alignment edit-distance score for a single-end or paired-end read.

    - Unmapped in reference (both r1 and r2 are None or unmapped): float("inf").
    - Single-end read (not is_paired): per-read edit distance.
    - Paired-end read with both mates mapped: mate-averaged edit distance
      (ed(r1) + ed(r2)) / 2.0, matching Kluin et al. (2018) MM_threshold = 4.
    - Paired-end read with only one mate mapped:
      * settings="default": the unmapped mate is penalized by the read length
        (at least 2 * mismatch_cutoff + 2), then averaged:
        (ed(mapped_mate) + penalty) / 2.0.
      * settings="R_default": the unmapped mate is penalized by fixed
        unmapped_penalty (default 8, matching XenofilteR.R L18), then averaged:
        (ed(mapped_mate) + unmapped_penalty) / 2.0.
    """
    validate_settings(settings)
    if r1 is not None and r1.is_unmapped:
        r1 = None
    if r2 is not None and r2.is_unmapped:
        r2 = None

    if r1 is None and r2 is None:
        return float("inf")

    is_paired = r1.is_paired if r1 is not None else r2.is_paired
    if not is_paired:
        mapped = r1 if r1 is not None else r2
        return float(count_mismatches(mapped, settings=settings))

    if r1 is not None and r2 is not None:
        return (
            count_mismatches(r1, settings=settings)
            + count_mismatches(r2, settings=settings)
        ) / 2.0

    mapped = r1 if r1 is not None else r2
    if settings == "R_default":
        penalty = float(unmapped_penalty)
    else:
        qlen = mapped.infer_read_length() or mapped.query_length or 0
        penalty = max(float(qlen), float(2 * mismatch_cutoff + 2))
    return (count_mismatches(mapped, settings=settings) + penalty) / 2.0


def _pick_better_record(existing, candidate, settings: SettingsMode):
    """Choose which BAM record to keep when multiple alignments map to the same mate slot."""
    if existing is None:
        return candidate
    if not candidate.is_supplementary and existing.is_supplementary:
        return candidate
    if candidate.is_supplementary and not existing.is_supplementary:
        return existing
    if count_mismatches(candidate, settings=settings) < count_mismatches(
        existing, settings=settings
    ):
        return candidate
    return existing


def read_pairs(bam, settings: SettingsMode = "default"):
    """
    Extract primary mapped alignments from an open pysam.AlignmentFile.

    Filters out unmapped, secondary, and (when settings="default") supplementary records.
    When settings="R_default", supplementary alignments (0x800) are retained per
    upstream ScanBamParam(isSecondaryAlignment=FALSE).
    Supports both paired-end ([read1, read2]) and single-end ([read, None]) BAMs.
    """
    validate_settings(settings)
    read_dict = {}
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary:
            continue
        if settings == "default" and read.is_supplementary:
            continue

        qname = read.query_name
        if qname not in read_dict:
            read_dict[qname] = [None, None]

        if not read.is_paired:
            slot = 0
        elif read.is_read1:
            slot = 0
        elif read.is_read2:
            slot = 1
        else:
            slot = 0

        if settings == "R_default":
            read_dict[qname][slot] = _pick_better_record(
                read_dict[qname][slot], read, settings=settings
            )
        else:
            read_dict[qname][slot] = read

    return read_dict
