"""
Mutational channel definitions and sequence-context / microhomology scanning
for CHORD & mutSigExtractor (96 SBS + 17 Indel + 19 SV channels).
"""

from __future__ import annotations

BASES: tuple[str, ...] = ("A", "C", "G", "T")
SBS_SUBS: tuple[str, ...] = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
COMPLEMENT: dict[str, str] = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
}

SBS96_CHANNELS: tuple[str, ...] = tuple(
    f"{b5}[{sub}]{b3}" for sub in SBS_SUBS for b5 in BASES for b3 in BASES
)

INDEL_REP_DEL_CHANNELS: tuple[str, ...] = tuple(f"del.rep.len.{i}" for i in range(1, 6))
INDEL_REP_INS_CHANNELS: tuple[str, ...] = tuple(f"ins.rep.len.{i}" for i in range(1, 6))
INDEL_MH_DEL_CHANNELS: tuple[str, ...] = tuple(f"del.mh.bimh.{i}" for i in range(1, 6))
INDEL_NONE_CHANNELS: tuple[str, ...] = ("del.none", "ins.none")

INDEL_CHANNELS: tuple[str, ...] = (
    INDEL_REP_DEL_CHANNELS
    + INDEL_REP_INS_CHANNELS
    + INDEL_MH_DEL_CHANNELS
    + INDEL_NONE_CHANNELS
)

SV_TYPES: tuple[str, ...] = ("DEL", "DUP", "INV")
SV_LEN_BINS: tuple[tuple[str, float, float], ...] = (
    ("0_1e3", 0.0, 1e3),
    ("1e3_1e4", 1e3, 1e4),
    ("1e4_1e5", 1e4, 1e5),
    ("1e5_1e6", 1e5, 1e6),
    ("1e6_1e7", 1e6, 1e7),
    ("1e7_inf", 1e7, float("inf")),
)

SV_CHANNELS: tuple[str, ...] = tuple(
    f"{sv_t}_{bin_lbl}" for sv_t in SV_TYPES for bin_lbl, _, _ in SV_LEN_BINS
) + ("TRA",)

ALL_CHORD_CHANNELS: tuple[str, ...] = SBS96_CHANNELS + INDEL_CHANNELS + SV_CHANNELS

# Map R CHORD / mutSigExtractor legacy column aliases if supplied in pre-extracted tables
R_SV_CHANNEL_ALIASES: dict[str, str] = {
    f"{sv_t}_{r_lbl}": f"{sv_t}_{py_lbl}"
    for sv_t in SV_TYPES
    for r_lbl, py_lbl in (
        ("0e00_1e03_bp", "0_1e3"),
        ("1e03_1e04_bp", "1e3_1e4"),
        ("1e04_1e05_bp", "1e4_1e5"),
        ("1e05_1e06_bp", "1e5_1e6"),
        ("1e06_1e07_bp", "1e6_1e7"),
        ("1e07_Inf_bp", "1e7_inf"),
    )
}


def revcomp(seq: str) -> str:
    """Return DNA reverse complement in uppercase."""
    return "".join(COMPLEMENT.get(ch, "N") for ch in reversed(seq.upper()))


def normalize_sbs96_context(ref: str, alt: str, trinuc: str) -> str | None:
    """
    Convert `(ref, alt, trinuc)` to pyrimidine-centered COSMIC 96 channel `X[R>A]Y`.
    Returns `None` if bases contain ambiguous `N` or invalid lengths.
    """
    r = ref.strip().upper()
    a = alt.strip().upper()
    ctx = trinuc.strip().upper()
    if len(r) != 1 or len(a) != 1 or r == a:
        return None
    if r not in BASES or a not in BASES:
        return None
    if len(ctx) < 3 or len(ctx) % 2 == 0:
        return None
    mid = len(ctx) // 2
    sub_ctx = ctx[mid - 1 : mid + 2]
    if any(b not in BASES for b in sub_ctx):
        return None

    # If context matches reverse complement of ref at center, handle gracefully
    if sub_ctx[1] != r:
        if COMPLEMENT.get(sub_ctx[1]) == r:
            sub_ctx = revcomp(sub_ctx)
        else:
            # Replace center base with ref while retaining 5' and 3' flanks
            sub_ctx = f"{sub_ctx[0]}{r}{sub_ctx[2]}"

    if r in ("C", "T"):
        return f"{sub_ctx[0]}[{r}>{a}]{sub_ctx[2]}"
    rc_ctx = revcomp(sub_ctx)
    rc_r = COMPLEMENT[r]
    rc_a = COMPLEMENT[a]
    return f"{rc_ctx[0]}[{rc_r}>{rc_a}]{rc_ctx[2]}"


def classify_indel_sequence(
    ref: str,
    alt: str,
    left_flank: str = "",
    right_flank: str = "",
) -> str | None:
    """
    Classify an indel into CHORD / mutSigExtractor's 17 indel channels using exact
    alignment-invariant cyclic repeat and bidirectional microhomology (`bimh`) scanning.
    """
    r = ref.strip().upper()
    a = alt.strip().upper()
    if r == a:
        return None

    lf = left_flank.strip().upper()
    rf = right_flank.strip().upper()

    # Trim shared prefix (standard VCF 1-bp anchor base) while updating left_flank
    prefix_k = 0
    max_pref = min(len(r), len(a))
    while prefix_k < max_pref and r[prefix_k] == a[prefix_k]:
        prefix_k += 1
    if prefix_k > 0:
        lf = lf + r[:prefix_k]
        r_rem = r[prefix_k:]
        a_rem = a[prefix_k:]
    else:
        r_rem = r
        a_rem = a

    # Trim shared suffix if any while updating right_flank
    suffix_k = 0
    max_suf = min(len(r_rem), len(a_rem))
    while (
        suffix_k < max_suf
        and r_rem[len(r_rem) - 1 - suffix_k] == a_rem[len(a_rem) - 1 - suffix_k]
    ):
        suffix_k += 1
    if suffix_k > 0:
        rf = r_rem[len(r_rem) - suffix_k :] + rf
        r_rem = r_rem[: len(r_rem) - suffix_k]
        a_rem = a_rem[: len(a_rem) - suffix_k]

    # Pure deletion
    if len(r_rem) > 0 and len(a_rem) == 0:
        del_seq = r_rem
        del_len = len(del_seq)
        cap_len = min(del_len, 5)

        mh_right = 0
        while mh_right < len(rf) and del_seq[mh_right % del_len] == rf[mh_right]:
            mh_right += 1

        mh_left = 0
        while (
            mh_left < len(lf)
            and del_seq[del_len - 1 - (mh_left % del_len)] == lf[len(lf) - 1 - mh_left]
        ):
            mh_left += 1

        total_hom = mh_right + mh_left
        if total_hom >= del_len:
            return f"del.rep.len.{cap_len}"
        if total_hom >= 1:
            return f"del.mh.bimh.{min(total_hom, 5)}"
        return "del.none"

    # Pure insertion
    if len(a_rem) > 0 and len(r_rem) == 0:
        ins_seq = a_rem
        ins_len = len(ins_seq)
        cap_len = min(ins_len, 5)

        hom_right = 0
        while hom_right < len(rf) and ins_seq[hom_right % ins_len] == rf[hom_right]:
            hom_right += 1

        hom_left = 0
        while (
            hom_left < len(lf)
            and ins_seq[ins_len - 1 - (hom_left % ins_len)] == lf[len(lf) - 1 - hom_left]
        ):
            hom_left += 1

        if (hom_right + hom_left) >= ins_len:
            return f"ins.rep.len.{cap_len}"
        return "ins.none"

    # Complex delins fallback
    if len(r_rem) >= len(a_rem):
        return "del.none"
    return "ins.none"


def classify_sv_channel(sv_type: str, sv_len: float | int | None = None) -> str | None:
    """Map `(sv_type, sv_len)` into one of CHORD's 19 SV type x length channels."""
    raw = str(sv_type).strip().upper().strip("<>")
    if ":" in raw:
        raw = raw.split(":", 1)[0]

    if raw in ("TRA", "BND", "TRANSLOCATION", "CTX"):
        return "TRA"
    if raw in ("DEL", "DELETION"):
        norm_type = "DEL"
    elif raw in ("DUP", "DUPLICATION", "TDUP", "TANDEM_DUP"):
        norm_type = "DUP"
    elif raw in ("INV", "INVERSION"):
        norm_type = "INV"
    else:
        return None

    length = abs(float(sv_len)) if sv_len is not None else 0.0
    for bin_lbl, lo, hi in SV_LEN_BINS:
        if lo <= length < hi:
            return f"{norm_type}_{bin_lbl}"
    return f"{norm_type}_1e7_inf"
