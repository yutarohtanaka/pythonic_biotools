"""
Built-in GRCh38 (hg38) and GRCh37 (hg19) chromosome length and centromere
boundary reference tables and chromosome normalization helpers for `facets`.
"""

from __future__ import annotations

from dataclasses import dataclass

ACROCENTRIC_CHROMS: frozenset[str] = frozenset(
    {"chr13", "chr14", "chr15", "chr21", "chr22"}
)

_CANONICAL_MAP: dict[str, str] = (
    {str(i): f"chr{i}" for i in range(1, 23)}
    | {
        "X": "chrX",
        "23": "chrX",
        "Y": "chrY",
        "24": "chrY",
        "M": "chrM",
        "MT": "chrM",
    }
)

_CHROM_RANK: dict[str, int] = (
    {f"chr{i}": i for i in range(1, 23)}
    | {"chrX": 23, "chrY": 24, "chrM": 25}
)


@dataclass(frozen=True)
class ChromArmReference:
    """Reference chromosome length and p/q-arm centromere boundaries (bp)."""

    chrom: str
    length: int
    centromere_start: int
    centromere_end: int
    is_acrocentric: bool

    @property
    def p_arm_length(self) -> int:
        return 0 if self.is_acrocentric else self.centromere_start

    @property
    def q_arm_length(self) -> int:
        return self.length - self.centromere_end

    @property
    def callable_length(self) -> int:
        return self.p_arm_length + self.q_arm_length

    def map_pos_to_arm(self, pos: int) -> str:
        """Map a 1-based coordinate on this chromosome to '<chrom>p' or '<chrom>q'."""
        midpoint = (self.centromere_start + self.centromere_end) // 2
        return f"{self.chrom}p" if pos <= midpoint else f"{self.chrom}q"


# Official UCSC hg38 (GRCh38) chromosome sizes and centromere spans
_GRCH38_RAW: list[tuple[str, int, int, int]] = [
    ("chr1", 248_956_422, 122_026_459, 124_932_724),
    ("chr2", 242_193_529, 92_188_145, 94_090_557),
    ("chr3", 198_295_559, 90_772_458, 93_655_574),
    ("chr4", 190_214_555, 49_708_101, 51_743_951),
    ("chr5", 181_538_259, 46_485_901, 50_059_807),
    ("chr6", 170_805_979, 58_553_889, 59_829_934),
    ("chr7", 159_345_973, 58_169_654, 60_828_234),
    ("chr8", 145_138_636, 44_033_745, 45_877_265),
    ("chr9", 138_394_717, 43_236_168, 45_518_558),
    ("chr10", 133_797_422, 39_686_683, 41_593_521),
    ("chr11", 135_086_622, 51_078_349, 54_425_074),
    ("chr12", 133_275_309, 34_769_408, 37_185_252),
    ("chr13", 114_364_328, 16_000_001, 18_051_248),
    ("chr14", 107_043_718, 16_000_001, 18_173_523),
    ("chr15", 101_991_189, 17_000_001, 19_725_254),
    ("chr16", 90_338_345, 36_311_159, 38_280_682),
    ("chr17", 83_257_441, 22_813_680, 26_885_980),
    ("chr18", 80_373_285, 15_460_900, 20_861_206),
    ("chr19", 58_617_616, 24_498_981, 27_190_874),
    ("chr20", 64_444_167, 26_436_233, 30_038_348),
    ("chr21", 46_709_983, 10_864_561, 12_915_808),
    ("chr22", 50_818_468, 12_954_788, 15_054_318),
    ("chrX", 156_040_895, 58_605_579, 62_412_542),
    ("chrY", 57_227_415, 10_316_945, 10_544_039),
]

# Official UCSC hg19 (GRCh37) chromosome sizes and centromere spans
_GRCH37_RAW: list[tuple[str, int, int, int]] = [
    ("chr1", 249_250_621, 121_535_434, 124_535_434),
    ("chr2", 243_199_373, 92_326_171, 95_326_171),
    ("chr3", 198_022_430, 90_504_854, 93_504_854),
    ("chr4", 191_154_276, 49_660_117, 52_660_117),
    ("chr5", 180_915_260, 46_405_641, 49_405_641),
    ("chr6", 171_115_067, 58_830_166, 61_830_166),
    ("chr7", 159_138_663, 58_054_331, 61_054_331),
    ("chr8", 146_364_022, 43_838_887, 46_838_887),
    ("chr9", 141_213_431, 47_367_679, 50_367_679),
    ("chr10", 135_534_747, 39_254_935, 42_254_935),
    ("chr11", 135_006_516, 51_644_205, 54_644_205),
    ("chr12", 133_851_895, 34_856_694, 37_856_694),
    ("chr13", 115_169_878, 16_000_000, 19_000_000),
    ("chr14", 107_349_540, 16_000_000, 19_000_000),
    ("chr15", 102_531_392, 17_000_000, 20_000_000),
    ("chr16", 90_354_753, 35_335_801, 38_335_801),
    ("chr17", 81_195_210, 22_263_006, 25_263_006),
    ("chr18", 78_077_248, 15_460_898, 18_460_898),
    ("chr19", 59_128_983, 24_681_782, 27_681_782),
    ("chr20", 63_025_520, 26_369_569, 29_369_569),
    ("chr21", 48_129_895, 11_288_129, 14_288_129),
    ("chr22", 51_304_566, 13_000_000, 16_000_000),
    ("chrX", 155_270_560, 58_632_012, 61_632_012),
    ("chrY", 59_373_566, 10_104_553, 13_104_553),
]

GRCH38_REFERENCE: dict[str, ChromArmReference] = {
    ch: ChromArmReference(ch, length, cs, ce, ch in ACROCENTRIC_CHROMS)
    for ch, length, cs, ce in _GRCH38_RAW
}

GRCH37_REFERENCE: dict[str, ChromArmReference] = {
    ch: ChromArmReference(ch, length, cs, ce, ch in ACROCENTRIC_CHROMS)
    for ch, length, cs, ce in _GRCH37_RAW
}


def canonical_chrom(chrom: str | int) -> str:
    """
    Normalize chromosome identifiers ('1'..'22', 'X', '23', 'Y', 'M'/'MT', 'chr1'..'chrX')
    to canonical 'chrN' form while preserving custom/unplaced contig names.
    """
    raw = str(chrom).strip()
    if not raw:
        raise ValueError("Chromosome name cannot be empty.")
    s = raw[3:] if raw.lower().startswith("chr") else raw
    mapped = _CANONICAL_MAP.get(s.upper())
    if mapped is not None:
        return mapped
    return raw


def chrom_sort_key(chrom: str) -> tuple[int, int, str]:
    """Deterministic genomic sort key: chr1..chr22, chrX, chrY, chrM, then custom."""
    c = canonical_chrom(chrom)
    rank = _CHROM_RANK.get(c)
    if rank is not None:
        return (0, rank, c)
    return (1, 0, c)


def get_genome_reference(
    genome: str = "GRCh38",
    *,
    include_chr_x: bool = True,
    include_chr_y: bool = False,
) -> dict[str, ChromArmReference]:
    """
    Return chromosome arm & centromere boundary table for 'GRCh38' ('hg38'),
    'GRCh37' ('hg19'), or empty dict for 'custom'.
    """
    if not isinstance(genome, str) or not genome.strip():
        raise ValueError("`genome` must be a non-empty string.")
    g = genome.strip().lower()
    if g in ("grch38", "hg38"):
        base = GRCH38_REFERENCE
    elif g in ("grch37", "hg19"):
        base = GRCH37_REFERENCE
    elif g == "custom":
        return {}
    else:
        raise ValueError(
            f"Unsupported genome '{genome}'. Expected 'GRCh38' ('hg38'), "
            "'GRCh37' ('hg19'), or 'custom'."
        )

    out: dict[str, ChromArmReference] = {}
    for k, v in base.items():
        if k == "chrX" and not include_chr_x:
            continue
        if k == "chrY" and not include_chr_y:
            continue
        out[k] = v
    return out
