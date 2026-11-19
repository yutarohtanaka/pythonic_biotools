"""
Statistical tests and reference genome chromosome tables for ShatterSeek
chromothripsis detection (Cortes-Ciriano et al., Nature Genetics 2020).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import special, stats

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]

GRCH38_CHROM_LENGTHS: dict[str, int] = {
    "chr1": 248_956_422,
    "chr2": 242_193_529,
    "chr3": 198_295_559,
    "chr4": 190_214_555,
    "chr5": 181_538_259,
    "chr6": 170_805_979,
    "chr7": 159_345_973,
    "chr8": 145_138_636,
    "chr9": 138_394_717,
    "chr10": 133_797_422,
    "chr11": 135_086_622,
    "chr12": 133_275_309,
    "chr13": 114_364_328,
    "chr14": 107_043_718,
    "chr15": 101_991_189,
    "chr16": 90_338_345,
    "chr17": 83_257_441,
    "chr18": 80_373_285,
    "chr19": 58_617_616,
    "chr20": 64_444_167,
    "chr21": 46_709_983,
    "chr22": 50_818_468,
    "chrX": 156_040_895,
    "chrY": 57_227_415,
}

GRCH37_CHROM_LENGTHS: dict[str, int] = {
    "chr1": 249_250_621,
    "chr2": 243_199_373,
    "chr3": 198_022_430,
    "chr4": 191_154_276,
    "chr5": 180_915_260,
    "chr6": 171_115_067,
    "chr7": 159_138_663,
    "chr8": 146_364_022,
    "chr9": 141_213_431,
    "chr10": 135_534_747,
    "chr11": 135_006_516,
    "chr12": 133_851_895,
    "chr13": 115_169_878,
    "chr14": 107_349_540,
    "chr15": 102_531_392,
    "chr16": 90_354_753,
    "chr17": 81_195_210,
    "chr18": 78_077_248,
    "chr19": 59_128_983,
    "chr20": 63_025_520,
    "chr21": 48_129_895,
    "chr22": 51_304_566,
    "chrX": 155_270_560,
    "chrY": 59_373_566,
}

_STD_SUFFIXES = {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}


def normalize_chrom(chrom: str) -> str:
    """Canonicalize '1' -> 'chr1', 'X' -> 'chrX' while preserving custom contigs."""
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        suf = c[3:].upper()
        if suf == "MT":
            suf = "M"
        if suf in _STD_SUFFIXES:
            return f"chr{suf}"
        return c
    up = c.upper()
    if up == "MT":
        return "chrM"
    if up in _STD_SUFFIXES:
        return f"chr{up}"
    return c


def get_reference_chrom_lengths(genome: str = "GRCh38") -> dict[str, int]:
    g = genome.strip().lower()
    if g in ("grch38", "hg38"):
        return dict(GRCH38_CHROM_LENGTHS)
    if g in ("grch37", "hg19"):
        return dict(GRCH37_CHROM_LENGTHS)
    raise ValueError(f"Unsupported reference genome '{genome}'. Expected 'GRCh38'/'hg38' or 'GRCh37'/'hg19'.")


def pval_fragment_joins(
    counts: Sequence[int] | np.ndarray,
    *,
    method: Literal["chisq", "gtest", "exact"] = "chisq",
) -> float:
    """
    Multinomial equal-probability goodness-of-fit test across the 4 SV join types
    (`DEL`, `DUP`, `h2hINV`, `t2tINV`) against null p_0 = (1/4, 1/4, 1/4, 1/4).

    In Cortes-Ciriano et al. (2020), random re-ligation of shattered DNA yields
    uniform orientations (`pval > 0.05`), whereas tandem-duplicator or BFB cycles
    skew strongly toward one orientation (`pval <= 0.05`).
    """
    obs = np.asarray(counts, dtype=np.int64).ravel()
    if obs.size != 4:
        raise ValueError(f"Fragment-join counts must have length 4 (DEL, DUP, h2hINV, t2tINV), got {obs.size}.")
    if np.any(obs < 0):
        raise ValueError("Fragment-join counts must be non-negative.")

    n = int(np.sum(obs))
    if n == 0:
        return 1.0

    if method == "chisq":
        expected = np.full(4, n / 4.0, dtype=np.float64)
        stat = float(np.sum(((obs - expected) ** 2) / expected))
        return float(np.clip(stats.chi2.sf(stat, df=3), 0.0, 1.0))

    if method == "gtest":
        expected = n / 4.0
        pos = obs > 0
        g_stat = 2.0 * float(np.sum(obs[pos] * np.log(obs[pos] / expected)))
        return float(np.clip(stats.chi2.sf(g_stat, df=3), 0.0, 1.0))

    if method == "exact":
        if n > 80:
            # Asymptotic chi-square is exact to <1e-4 for n > 80
            return pval_fragment_joins(obs, method="chisq")
        log_fact_n = float(special.gammaln(n + 1))
        n_log_quarter = -n * float(np.log(4.0))
        obs_log_p = (
            log_fact_n
            - float(np.sum(special.gammaln(obs + 1)))
            + n_log_quarter
        )
        # Enumerate all 4-compositions of n
        total_p = 0.0
        for k1 in range(n + 1):
            lf1 = float(special.gammaln(k1 + 1))
            rem1 = n - k1
            for k2 in range(rem1 + 1):
                lf12 = lf1 + float(special.gammaln(k2 + 1))
                rem2 = rem1 - k2
                k3 = np.arange(rem2 + 1, dtype=np.int64)
                k4 = rem2 - k3
                log_p = (
                    log_fact_n
                    - (lf12 + special.gammaln(k3 + 1) + special.gammaln(k4 + 1))
                    + n_log_quarter
                )
                le_mask = log_p <= obs_log_p + 1e-12
                if np.any(le_mask):
                    total_p += float(np.sum(np.exp(log_p[le_mask])))
        return float(np.clip(total_p, 0.0, 1.0))

    raise ValueError(f"Unknown method '{method}'. Expected 'chisq', 'gtest', or 'exact'.")


def pval_exp_cluster(
    breakpoints: Sequence[int | float] | np.ndarray,
    *,
    method: Literal["ks", "anderson"] = "ks",
    settings: SettingsMode = "default",
) -> float:
    """
    Goodness-of-fit test comparing consecutive intra-cluster breakpoint spacings
    against an Exponential distribution (uniform random shattering null).
    """
    validate_settings(settings)
    bp = np.sort(np.asarray(breakpoints, dtype=np.float64).ravel())
    min_bp = 6 if settings == "R_default" else 3
    if bp.size < min_bp:
        return 1.0
    spacings = np.diff(bp)
    spacings = spacings[spacings > 0.0]
    k = spacings.size
    if k < 2:
        return 1.0

    mean_d = float(np.mean(spacings))
    if mean_d <= 0.0:
        return 1.0

    if method == "ks":
        res = stats.kstest(spacings, "expon", args=(0.0, mean_d))
        return float(np.clip(res.pvalue, 0.0, 1.0))

    if method == "anderson":
        d_sorted = np.sort(spacings)
        u = np.clip(-np.expm1(-d_sorted / mean_d), 1e-15, 1.0 - 1e-15)
        idx = np.arange(1, k + 1, dtype=np.float64)
        a2 = -float(k) - float(
            np.sum((2.0 * idx - 1.0) * (np.log(u) + np.log(1.0 - u[::-1])))
        ) / float(k)
        # D'Agostino & Stephens (1986, Table 4.14) modified statistic for Exponential(mean_hat),
        # with monotone exponential tail continuation above z = 5.0 so extreme z -> 0 smoothly.
        z = max(0.0, a2 * (1.0 + 0.6 / float(k)))
        if z > 5.0:
            log_p = (0.731 - 3.009 * 5.0 + 0.15 * 25.0) - 1.509 * (z - 5.0)
            pval = float(np.exp(max(log_p, -700.0)))
        elif z >= 0.95:
            pval = float(np.exp(0.731 - 3.009 * z + 0.15 * (z ** 2)))
        elif z >= 0.51:
            pval = float(np.exp(0.9209 - 3.353 * z + 0.30 * (z ** 2)))
        elif z >= 0.26:
            pval = float(1.0 - np.exp(-6.1327 + 20.218 * z - 18.663 * (z ** 2)))
        else:
            pval = float(1.0 - np.exp(-12.2204 + 67.459 * z - 110.3 * (z ** 2)))
        return float(np.clip(pval, 0.0, 1.0))

    raise ValueError(f"Unknown method '{method}'. Expected 'ks' or 'anderson'.")


def pval_chr_breakpoint_enrichment(
    n_chr_svs: int,
    n_total_svs: int,
    chr_length: int | float,
    genome_length: int | float,
    *,
    settings: SettingsMode = "default",
) -> float:
    """
    Exact Binomial test comparing observed intrachromosomal SVs on chromosome c
    against the genome-wide rate weighted by callable chromosome length.
    Uses one-sided ('greater') in `settings='default'` and two-sided in `settings='R_default'`.
    """
    validate_settings(settings)
    if n_chr_svs < 0 or n_total_svs < n_chr_svs:
        raise ValueError(f"Invalid SV counts: n_chr_svs={n_chr_svs}, n_total_svs={n_total_svs}.")
    if chr_length <= 0 or genome_length < chr_length:
        raise ValueError(f"Invalid lengths: chr_length={chr_length}, genome_length={genome_length}.")
    if n_total_svs == 0:
        return 1.0
    if settings == "default" and n_chr_svs == 0:
        return 1.0

    p0 = float(chr_length) / float(genome_length)
    p0 = float(np.clip(p0, 0.0, 1.0))
    alt: Literal["greater", "two-sided"] = "two-sided" if settings == "R_default" else "greater"
    return float(stats.binomtest(int(n_chr_svs), int(n_total_svs), p=p0, alternative=alt).pvalue)


def max_oscillating_cn_segments(
    cn_states: Sequence[int | float] | np.ndarray,
    *,
    settings: SettingsMode = "default",
) -> tuple[int, int]:
    """
    Compute the maximum number of contiguous CN segments oscillating strictly
    between 2 copy-number states and at most 3 copy-number states.

    - `settings='default'`: Adjacent segments sharing an identical copy-number value
      are first collapsed, and 3-state oscillation finds the longest subarray with <= 3
      distinct states.
    - `settings='R_default'`: Matches `parklab/ShatterSeek` v1.1 `function_criteria.R`
      without pre-merging equal-CN segments, computing 3-state CN oscillation as the
      longest contiguous run satisfying `abs(cn[i] - cn[i-2]) <= 1` (`%in% c(0, 1)`).
    """
    validate_settings(settings)
    arr = np.asarray(cn_states, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0, 0

    if settings == "R_default":
        if arr.size < 2:
            return 0, 0
        max_2 = 0
        max_3 = 0
        cur_2 = 1
        cur_3 = 1
        for idx in range(1, arr.size):
            if arr[idx] == arr[idx - 1]:
                cur_2 = 1
                cur_3 = 1
            else:
                if cur_2 >= 2 and arr[idx] == arr[idx - 2]:
                    cur_2 += 1
                else:
                    cur_2 = 2
                if cur_2 > max_2:
                    max_2 = cur_2

                if cur_3 >= 2 and abs(float(arr[idx]) - float(arr[idx - 2])) <= 1.0 + 1e-9:
                    cur_3 += 1
                else:
                    cur_3 = 2
                if cur_3 > max_3:
                    max_3 = cur_3
        return max_2, max_3

    # Collapse adjacent identical states
    collapsed: list[float] = [float(arr[0])]
    for val in arr[1:]:
        v = float(val)
        if v != collapsed[-1]:
            collapsed.append(v)

    k = len(collapsed)
    if k < 2:
        return 0, 0

    # Exact 2-state oscillation: after collapsing adjacent duplicates, s[i] != s[i-1]
    # always holds, so a subarray alternates strictly between 2 states iff s[i] == s[i-2].
    max_2 = 2
    cur_2 = 2
    for idx in range(2, k):
        if collapsed[idx] == collapsed[idx - 2]:
            cur_2 += 1
        else:
            cur_2 = 2
        if cur_2 > max_2:
            max_2 = cur_2

    # Exact 3-state oscillation: longest contiguous collapsed subarray with <= 3 distinct states
    max_3 = 2
    counts: dict[float, int] = {}
    left = 0
    for right in range(k):
        vr = collapsed[right]
        counts[vr] = counts.get(vr, 0) + 1
        while len(counts) > 3:
            vl = collapsed[left]
            counts[vl] -= 1
            if counts[vl] == 0:
                del counts[vl]
            left += 1
        span = right - left + 1
        if span > max_3:
            max_3 = span

    return max_2, max_3
