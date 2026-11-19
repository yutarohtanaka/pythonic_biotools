BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5


def count_mismatches(read):
    """
    Compute per-read edit distance following Kluin et al. (BMC Bioinformatics 2018):
        edit_distance = NM + soft_clipped_bases + hard_clipped_bases

    Returns float("inf") if the read is None or unmapped.
    """
    if read is None or read.is_unmapped:
        return float("inf")

    if read.has_tag("NM"):
        nm = int(read.get_tag("NM"))
    elif read.has_tag("nM"):
        nm = int(read.get_tag("nM"))
    else:
        nm = 0

    clipped = sum(
        length
        for op, length in (read.cigartuples or ())
        if op in (BAM_CSOFT_CLIP, BAM_CHARD_CLIP)
    )
    return nm + clipped


def score_pair(r1, r2, mismatch_cutoff=4):
    """
    Compute the alignment edit-distance score for a single-end or paired-end read.

    - Unmapped in reference (both r1 and r2 are None or unmapped): float("inf").
    - Single-end read (not is_paired): per-read edit distance.
    - Paired-end read with both mates mapped: mate-averaged edit distance
      (ed(r1) + ed(r2)) / 2.0, matching Kluin et al. (2018) MM_threshold = 4.
    - Paired-end read with only one mate mapped: the unmapped mate is penalized
      by the read length (at least 2 * mismatch_cutoff + 2), then averaged:
      (ed(mapped_mate) + unmapped_penalty) / 2.0.
    """
    if r1 is not None and r1.is_unmapped:
        r1 = None
    if r2 is not None and r2.is_unmapped:
        r2 = None

    if r1 is None and r2 is None:
        return float("inf")

    is_paired = r1.is_paired if r1 is not None else r2.is_paired
    if not is_paired:
        mapped = r1 if r1 is not None else r2
        return float(count_mismatches(mapped))

    if r1 is not None and r2 is not None:
        return (count_mismatches(r1) + count_mismatches(r2)) / 2.0

    mapped = r1 if r1 is not None else r2
    qlen = mapped.infer_read_length() or mapped.query_length or 0
    penalty = max(float(qlen), float(2 * mismatch_cutoff + 2))
    return (count_mismatches(mapped) + penalty) / 2.0


def read_pairs(bam):
    """
    Extract primary mapped alignments from an open pysam.AlignmentFile.

    Filters out unmapped, secondary, and supplementary records.
    Supports both paired-end ([read1, read2]) and single-end ([read, None]) BAMs.
    """
    read_dict = {}
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        qname = read.query_name
        if qname not in read_dict:
            read_dict[qname] = [None, None]

        if not read.is_paired:
            read_dict[qname][0] = read
        elif read.is_read1:
            read_dict[qname][0] = read
        elif read.is_read2:
            read_dict[qname][1] = read
        else:
            read_dict[qname][0] = read

    return read_dict
