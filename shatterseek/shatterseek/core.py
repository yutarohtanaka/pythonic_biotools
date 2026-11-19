"""
Clean-room Python implementation of ShatterSeek chromothripsis detection
(Cortes-Ciriano et al., Nature Genetics 2020).

Reference
---------
Cortes-Ciriano, I., Lee, J. J.-K., Xi, R., Jain, D., Jung, Y. L., Yang, L.,
Gordenin, D., Klimczak, L. J., Zhang, C.-Z., Pellman, D. S., & Park, P. J. (2020).
Comprehensive analysis of chromothripsis in 2,658 human cancers using
whole-genome sequencing. Nature Genetics, 52(3), 331-341.
https://doi.org/10.1038/s41588-019-0576-7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from .stats import (
    get_reference_chrom_lengths,
    max_oscillating_cn_segments,
    normalize_chrom,
    pval_chr_breakpoint_enrichment,
    pval_exp_cluster,
    pval_fragment_joins,
)

VALID_SV_TYPES = frozenset({"DEL", "DUP", "h2hINV", "t2tINV", "TRA"})
ConfidenceTier = Literal["High", "Low", "None"]


@dataclass(frozen=True)
class ChromothripsisCluster:
    """Cluster-level chromothripsis call and statistical diagnostics."""

    cluster_id: str
    chrom: str
    start: int
    end: int
    span_bp: int
    n_intrachromosomal_svs: int
    n_interleaved_svs: int
    n_del: int
    n_dup: int
    n_h2h_inv: int
    n_t2t_inv: int
    n_tra: int
    max_number_oscillating_cn_segments_2_states: int
    max_number_oscillating_cn_segments_3_states: int
    chr_max_oscillating_cn_2_states: int
    chr_max_oscillating_cn_3_states: int
    n_cn_segments: int
    n_loh_segments: int
    interspersed_loh: bool
    pval_fragment_joins: float
    pval_exp_cluster: float
    pval_chr_breakpoint_enrichment: float
    interchromosomal_chroms: tuple[str, ...]
    linked_cluster_ids: tuple[str, ...]
    confidence: ConfidenceTier

    @property
    def is_chromothripsis(self) -> bool:
        return self.confidence in ("High", "Low")

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence == "High"


@dataclass(frozen=True)
class ShatterSeekResult:
    """Genome-wide ShatterSeek chromothripsis detection summary."""

    clusters: list[ChromothripsisCluster]
    genome: str
    n_total_intrachromosomal_svs: int
    n_total_translocations: int

    @property
    def has_chromothripsis(self) -> bool:
        return any(c.is_chromothripsis for c in self.clusters)

    @property
    def has_high_confidence(self) -> bool:
        return any(c.is_high_confidence for c in self.clusters)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of all detected SV clusters and statistical criteria."""
        cols = [
            "cluster_id",
            "chrom",
            "start",
            "end",
            "span_bp",
            "n_intrachromosomal_svs",
            "n_interleaved_svs",
            "n_del",
            "n_dup",
            "n_h2h_inv",
            "n_t2t_inv",
            "n_tra",
            "max_number_oscillating_cn_segments_2_states",
            "max_number_oscillating_cn_segments_3_states",
            "chr_max_oscillating_cn_2_states",
            "chr_max_oscillating_cn_3_states",
            "n_cn_segments",
            "n_loh_segments",
            "interspersed_loh",
            "pval_fragment_joins",
            "pval_exp_cluster",
            "pval_chr_breakpoint_enrichment",
            "interchromosomal_chroms",
            "linked_cluster_ids",
            "confidence",
        ]
        if not self.clusters:
            return pd.DataFrame(columns=cols)
        rows = [
            {
                "cluster_id": c.cluster_id,
                "chrom": c.chrom,
                "start": c.start,
                "end": c.end,
                "span_bp": c.span_bp,
                "n_intrachromosomal_svs": c.n_intrachromosomal_svs,
                "n_interleaved_svs": c.n_interleaved_svs,
                "n_del": c.n_del,
                "n_dup": c.n_dup,
                "n_h2h_inv": c.n_h2h_inv,
                "n_t2t_inv": c.n_t2t_inv,
                "n_tra": c.n_tra,
                "max_number_oscillating_cn_segments_2_states": c.max_number_oscillating_cn_segments_2_states,
                "max_number_oscillating_cn_segments_3_states": c.max_number_oscillating_cn_segments_3_states,
                "chr_max_oscillating_cn_2_states": c.chr_max_oscillating_cn_2_states,
                "chr_max_oscillating_cn_3_states": c.chr_max_oscillating_cn_3_states,
                "n_cn_segments": c.n_cn_segments,
                "n_loh_segments": c.n_loh_segments,
                "interspersed_loh": c.interspersed_loh,
                "pval_fragment_joins": c.pval_fragment_joins,
                "pval_exp_cluster": c.pval_exp_cluster,
                "pval_chr_breakpoint_enrichment": c.pval_chr_breakpoint_enrichment,
                "interchromosomal_chroms": ",".join(c.interchromosomal_chroms),
                "linked_cluster_ids": ",".join(c.linked_cluster_ids),
                "confidence": c.confidence,
            }
            for c in self.clusters
        ]
        return pd.DataFrame(rows, columns=cols)


def infer_sv_type(
    chrom1: str,
    pos1: int,
    strand1: str | None,
    chrom2: str,
    pos2: int,
    strand2: str | None,
    sv_type: str | None = None,
) -> tuple[str, int, str | None, str, int, str | None, str]:
    """
    Canonicalize breakpoint ordering (`pos1 <= pos2` for intrachromosomal SVs)
    and infer or validate `sv_type` in `{"DEL", "DUP", "h2hINV", "t2tINV", "TRA"}`.

    Orientation mapping for `chrom1 == chrom2` with `pos1 < pos2`:
    - `(+, -)` -> `"DEL"`
    - `(-, +)` -> `"DUP"`
    - `(+, +)` -> `"h2hINV"`
    - `(-, -)` -> `"t2tINV"`
    """
    c1 = normalize_chrom(chrom1)
    c2 = normalize_chrom(chrom2)
    p1 = int(pos1)
    p2 = int(pos2)
    if p1 < 0 or p2 < 0:
        raise ValueError(f"SV positions must be non-negative, got ({p1}, {p2}).")

    s1 = None if (strand1 is None or pd.isna(strand1)) else str(strand1).strip()
    s2 = None if (strand2 is None or pd.isna(strand2)) else str(strand2).strip()
    if s1 is not None and s1 not in ("+", "-"):
        raise ValueError(f"Invalid strand1 '{s1}'. Expected '+' or '-'.")
    if s2 is not None and s2 not in ("+", "-"):
        raise ValueError(f"Invalid strand2 '{s2}'. Expected '+' or '-'.")

    st = None if (sv_type is None or pd.isna(sv_type) or str(sv_type).strip() == "") else str(sv_type).strip()

    if c1 != c2:
        if st is not None and st != "TRA":
            raise ValueError(
                f"Interchromosomal SV ({c1} != {c2}) must have sv_type='TRA', got '{st}'."
            )
        return c1, p1, s1, c2, p2, s2, "TRA"

    if p1 == p2:
        raise ValueError(f"Intrachromosomal SV on {c1} cannot have pos1 == pos2 == {p1}.")
    if p1 > p2:
        p1, p2 = p2, p1
        s1, s2 = s2, s1

    orientation_map = {
        ("+", "-"): "DEL",
        ("-", "+"): "DUP",
        ("+", "+"): "h2hINV",
        ("-", "-"): "t2tINV",
    }

    if st is None:
        if s1 is None or s2 is None:
            raise ValueError(
                "strand1 and strand2 are required to infer sv_type when sv_type is omitted."
            )
        inferred = orientation_map[(s1, s2)]
        return c1, p1, s1, c2, p2, s2, inferred

    if st not in VALID_SV_TYPES or st == "TRA":
        raise ValueError(
            f"Intrachromosomal sv_type must be one of ('DEL', 'DUP', 'h2hINV', 't2tINV'), got '{st}'."
        )
    return c1, p1, s1, c2, p2, s2, st


def classify_chromothripsis(
    *,
    n_interleaved_svs: int,
    n_tra: int,
    max_oscillating_cn_2_states: int,
    pval_fragment_joins: float,
    pval_chr_breakpoint_enrichment: float,
    pval_exp_cluster: float,
    alpha: float = 0.05,
) -> ConfidenceTier:
    """
    Apply official PCAWG Cortes-Ciriano et al. (2020) chromothripsis confidence criteria.

    - **High-Confidence**:
      1. Intrachromosomal rule: `n_interleaved_svs >= 6` AND `osc_2 >= 7` AND
         `pval_fragment_joins > alpha` AND
         (`pval_chr_breakpoint_enrichment <= alpha` OR `pval_exp_cluster > alpha`).
      2. Multi-chromosomal rule: `n_interleaved_svs >= 3` AND `n_tra >= 4` AND
         `osc_2 >= 7` AND `pval_fragment_joins > alpha`.
    - **Low-Confidence**:
      `n_interleaved_svs >= 6` AND `4 <= osc_2 <= 6` AND `pval_fragment_joins > alpha`
      AND (`pval_chr_breakpoint_enrichment <= alpha` OR `pval_exp_cluster > alpha`).
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie in (0, 1).")

    joins_random = pval_fragment_joins > alpha
    spatial_supported = (pval_chr_breakpoint_enrichment <= alpha) or (pval_exp_cluster > alpha)

    if joins_random and max_oscillating_cn_2_states >= 7:
        if n_interleaved_svs >= 6 and spatial_supported:
            return "High"
        if n_interleaved_svs >= 3 and n_tra >= 4:
            return "High"

    if (
        joins_random
        and spatial_supported
        and n_interleaved_svs >= 6
        and 4 <= max_oscillating_cn_2_states <= 6
    ):
        return "Low"

    return "None"


def _parse_cn_segments(cn_segments: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cn_segments, pd.DataFrame):
        cn_segments = pd.DataFrame(cn_segments)
    if len(cn_segments) == 0:
        raise ValueError("cn_segments must contain at least 1 copy-number segment.")

    req = {"chrom", "start", "end"}
    missing = req - set(cn_segments.columns)
    if missing:
        raise ValueError(f"cn_segments is missing required columns: {sorted(missing)}.")
    if cn_segments[["chrom", "start", "end"]].isna().any().any():
        raise ValueError("cn_segments contains NaN in chrom, start, or end.")

    has_total = "total_cn" in cn_segments.columns
    has_alleles = "major_cn" in cn_segments.columns and "minor_cn" in cn_segments.columns
    if not has_total and not has_alleles:
        raise ValueError(
            "cn_segments must provide 'total_cn' or both ('major_cn', 'minor_cn')."
        )

    df = pd.DataFrame(
        {
            "chrom": [normalize_chrom(c) for c in cn_segments["chrom"]],
            "start": np.asarray(cn_segments["start"], dtype=np.int64),
            "end": np.asarray(cn_segments["end"], dtype=np.int64),
        }
    )
    if np.any(df["start"] < 0) or np.any(df["end"] <= df["start"]):
        raise ValueError("cn_segments must satisfy 0 <= start < end.")

    if has_total:
        if cn_segments["total_cn"].isna().any():
            raise ValueError("cn_segments 'total_cn' contains NaN.")
        df["total_cn"] = np.asarray(cn_segments["total_cn"], dtype=np.float64)
    else:
        df["total_cn"] = np.asarray(cn_segments["major_cn"], dtype=np.float64) + np.asarray(
            cn_segments["minor_cn"], dtype=np.float64
        )

    if np.any(df["total_cn"] < 0):
        raise ValueError("total_cn must be >= 0.")

    if "minor_cn" in cn_segments.columns:
        if cn_segments["minor_cn"].isna().any():
            raise ValueError("cn_segments 'minor_cn' contains NaN.")
        minor = np.asarray(cn_segments["minor_cn"], dtype=np.float64)
        if np.any(minor < 0) or np.any(minor > df["total_cn"]):
            raise ValueError("minor_cn must satisfy 0 <= minor_cn <= total_cn.")
        df["is_loh"] = (minor == 0.0) & (df["total_cn"] >= 1.0)
    else:
        # When allele-specific CN is absent, hemizygous CN == 1 marks deletion LOH
        df["is_loh"] = df["total_cn"] == 1.0

    return df.sort_values(["chrom", "start", "end"], kind="mergesort").reset_index(drop=True)


def detect_chromothripsis(
    sv_data: pd.DataFrame,
    cn_segments: pd.DataFrame,
    *,
    genome: str = "GRCh38",
    chrom_lengths: Mapping[str, int] | None = None,
    tra_margin_bp: int = 0,
    joins_method: Literal["chisq", "gtest", "exact"] = "chisq",
    exp_cluster_method: Literal["ks", "anderson"] = "ks",
    alpha: float = 0.05,
) -> ShatterSeekResult:
    """
    Detect and classify chromothripsis regions across chromosomes from SVs and CN segments.
    """
    if tra_margin_bp < 0:
        raise ValueError("tra_margin_bp must be >= 0.")
    if not isinstance(sv_data, pd.DataFrame):
        sv_data = pd.DataFrame(sv_data)

    cn_df = _parse_cn_segments(cn_segments)

    ref_lengths = (
        {normalize_chrom(k): int(v) for k, v in chrom_lengths.items()}
        if chrom_lengths is not None
        else get_reference_chrom_lengths(genome)
    )
    # Ensure any custom contig in CN/SV data has a valid reference length
    for ch, grp in cn_df.groupby("chrom", sort=False):
        max_end = int(grp["end"].max())
        if ch not in ref_lengths or ref_lengths[ch] < max_end:
            ref_lengths[ch] = max_end

    if len(sv_data) == 0:
        return ShatterSeekResult(
            clusters=[],
            genome=genome,
            n_total_intrachromosomal_svs=0,
            n_total_translocations=0,
        )

    req_sv = {"chrom1", "pos1", "pos2"}
    missing_sv = req_sv - set(sv_data.columns)
    if missing_sv:
        raise ValueError(f"sv_data is missing required columns: {sorted(missing_sv)}.")
    if "chrom2" not in sv_data.columns:
        sv_data = sv_data.copy()
        sv_data["chrom2"] = sv_data["chrom1"]
    if sv_data[["chrom1", "pos1", "chrom2", "pos2"]].isna().any().any():
        raise ValueError("sv_data contains NaN in chrom1, pos1, chrom2, or pos2.")

    has_s1 = "strand1" in sv_data.columns
    has_s2 = "strand2" in sv_data.columns
    has_type = "sv_type" in sv_data.columns

    intra_by_chrom: dict[str, list[tuple[int, int, str]]] = {}
    tra_records: list[tuple[str, int, str, int]] = []
    n_total_intra = 0

    for idx in range(len(sv_data)):
        row = sv_data.iloc[idx]
        c1, p1, _, c2, p2, _, st = infer_sv_type(
            chrom1=str(row["chrom1"]),
            pos1=int(row["pos1"]),
            strand1=row["strand1"] if has_s1 else None,
            chrom2=str(row["chrom2"]),
            pos2=int(row["pos2"]),
            strand2=row["strand2"] if has_s2 else None,
            sv_type=row["sv_type"] if has_type else None,
        )
        for ch_p, pos_p in ((c1, p1), (c2, p2)):
            if ch_p not in ref_lengths or ref_lengths[ch_p] < pos_p:
                ref_lengths[ch_p] = max(pos_p, 1)

        if st == "TRA":
            tra_records.append((c1, p1, c2, p2))
        else:
            intra_by_chrom.setdefault(c1, []).append((p1, p2, st))
            n_total_intra += 1

    genome_length = sum(int(v) for v in ref_lengths.values() if v > 0)

    # Precompute chromosome-wide CN oscillations
    chr_osc_map: dict[str, tuple[int, int]] = {}
    cn_by_chrom: dict[str, pd.DataFrame] = {}
    for ch, grp in cn_df.groupby("chrom", sort=False):
        cn_by_chrom[str(ch)] = grp
        chr_osc_map[str(ch)] = max_oscillating_cn_segments(grp["total_cn"].to_numpy())

    # Step 1: Extract connected components of the interleaved SV graph per chromosome
    raw_clusters: list[dict[str, object]] = []
    for ch in dict.fromkeys(intra_by_chrom.keys()):
        sv_list = intra_by_chrom[ch]
        k_ch = len(sv_list)
        a = np.array([x[0] for x in sv_list], dtype=np.int64)
        b = np.array([x[1] for x in sv_list], dtype=np.int64)
        types = [x[2] for x in sv_list]

        # Strict interleaving: a_i < a_j < b_i < b_j (partial overlap without nesting)
        inter_ij = (
            (a[:, None] < a[None, :])
            & (a[None, :] < b[:, None])
            & (b[:, None] < b[None, :])
        )
        adj = inter_ij | inter_ij.T
        n_comp, labels = csgraph.connected_components(adj.astype(np.int8), directed=False)

        # Sort clusters along chromosome coordinate order
        comp_spans: list[tuple[int, int, np.ndarray]] = []
        for comp_id in range(n_comp):
            members = np.where(labels == comp_id)[0]
            c_start = int(np.min(a[members]))
            c_end = int(np.max(b[members]))
            comp_spans.append((c_start, c_end, members))
        comp_spans.sort(key=lambda t: (t[0], t[1]))

        pval_chr_enr = pval_chr_breakpoint_enrichment(
            n_chr_svs=k_ch,
            n_total_svs=n_total_intra,
            chr_length=ref_lengths[ch],
            genome_length=genome_length,
        )

        for local_idx, (c_start, c_end, members) in enumerate(comp_spans):
            cid = f"{ch}_cluster_{local_idx + 1}"
            m_types = [types[int(m)] for m in members]
            n_intra_c = int(members.size)
            n_interleaved_c = n_intra_c if n_intra_c >= 2 else 0

            n_del = sum(1 for t in m_types if t == "DEL")
            n_dup = sum(1 for t in m_types if t == "DUP")
            n_h2h = sum(1 for t in m_types if t == "h2hINV")
            n_t2t = sum(1 for t in m_types if t == "t2tINV")

            bps = np.concatenate([a[members], b[members]])
            raw_clusters.append(
                {
                    "cluster_id": cid,
                    "chrom": ch,
                    "start": c_start,
                    "end": c_end,
                    "n_intra": n_intra_c,
                    "n_interleaved": n_interleaved_c,
                    "n_del": n_del,
                    "n_dup": n_dup,
                    "n_h2h": n_h2h,
                    "n_t2t": n_t2t,
                    "bps": bps,
                    "pval_chr_enr": pval_chr_enr,
                    "n_tra": 0,
                    "partner_chroms": [],
                    "linked_clusters": [],
                }
            )

    # Step 2: Link interchromosomal translocations (TRA) to clusters across chromosomes
    for c1, p1, c2, p2 in tra_records:
        matched_1 = [
            idx
            for idx, rc in enumerate(raw_clusters)
            if rc["chrom"] == c1
            and int(rc["start"]) - tra_margin_bp <= p1 <= int(rc["end"]) + tra_margin_bp
        ]
        matched_2 = [
            idx
            for idx, rc in enumerate(raw_clusters)
            if rc["chrom"] == c2
            and int(rc["start"]) - tra_margin_bp <= p2 <= int(rc["end"]) + tra_margin_bp
        ]
        for i1 in matched_1:
            raw_clusters[i1]["n_tra"] = int(raw_clusters[i1]["n_tra"]) + 1
            raw_clusters[i1]["partner_chroms"].append(c2)  # type: ignore[attr-defined]
            for i2 in matched_2:
                raw_clusters[i1]["linked_clusters"].append(str(raw_clusters[i2]["cluster_id"]))  # type: ignore[attr-defined]
        for i2 in matched_2:
            raw_clusters[i2]["n_tra"] = int(raw_clusters[i2]["n_tra"]) + 1
            raw_clusters[i2]["partner_chroms"].append(c1)  # type: ignore[attr-defined]
            for i1 in matched_1:
                raw_clusters[i2]["linked_clusters"].append(str(raw_clusters[i1]["cluster_id"]))  # type: ignore[attr-defined]

    # Step 3: Evaluate copy-number oscillations, statistical tests, and PCAWG classification
    final_clusters: list[ChromothripsisCluster] = []
    for rc in raw_clusters:
        ch = str(rc["chrom"])
        c_start = int(rc["start"])
        c_end = int(rc["end"])
        chr_osc_2, chr_osc_3 = chr_osc_map.get(ch, (0, 0))

        if ch in cn_by_chrom:
            cn_ch = cn_by_chrom[ch]
            # Closed endpoint match includes segments bounded directly by cluster start/end breakpoints
            ov_mask = (cn_ch["end"].to_numpy() >= c_start) & (cn_ch["start"].to_numpy() <= c_end)
            cn_sub = cn_ch.loc[ov_mask]
            n_cn_seg = len(cn_sub)
            osc_2, osc_3 = max_oscillating_cn_segments(cn_sub["total_cn"].to_numpy())
            n_loh = int(np.sum(cn_sub["is_loh"].to_numpy()))
            inter_loh = bool(n_loh >= 1 and n_loh < n_cn_seg)
        else:
            n_cn_seg = 0
            osc_2, osc_3 = 0, 0
            n_loh = 0
            inter_loh = False

        n_del = int(rc["n_del"])
        n_dup = int(rc["n_dup"])
        n_h2h = int(rc["n_h2h"])
        n_t2t = int(rc["n_t2t"])
        n_interleaved = int(rc["n_interleaved"])
        n_tra = int(rc["n_tra"])

        p_joins = pval_fragment_joins([n_del, n_dup, n_h2h, n_t2t], method=joins_method)
        p_exp = pval_exp_cluster(rc["bps"], method=exp_cluster_method)  # type: ignore[arg-type]
        p_enr = float(rc["pval_chr_enr"])

        tier = classify_chromothripsis(
            n_interleaved_svs=n_interleaved,
            n_tra=n_tra,
            max_oscillating_cn_2_states=osc_2,
            pval_fragment_joins=p_joins,
            pval_chr_breakpoint_enrichment=p_enr,
            pval_exp_cluster=p_exp,
            alpha=alpha,
        )

        p_chroms = tuple(dict.fromkeys(rc["partner_chroms"]))  # type: ignore[arg-type]
        l_cids = tuple(dict.fromkeys(rc["linked_clusters"]))  # type: ignore[arg-type]

        final_clusters.append(
            ChromothripsisCluster(
                cluster_id=str(rc["cluster_id"]),
                chrom=ch,
                start=c_start,
                end=c_end,
                span_bp=c_end - c_start,
                n_intrachromosomal_svs=int(rc["n_intra"]),
                n_interleaved_svs=n_interleaved,
                n_del=n_del,
                n_dup=n_dup,
                n_h2h_inv=n_h2h,
                n_t2t_inv=n_t2t,
                n_tra=n_tra,
                max_number_oscillating_cn_segments_2_states=osc_2,
                max_number_oscillating_cn_segments_3_states=osc_3,
                chr_max_oscillating_cn_2_states=chr_osc_2,
                chr_max_oscillating_cn_3_states=chr_osc_3,
                n_cn_segments=n_cn_seg,
                n_loh_segments=n_loh,
                interspersed_loh=inter_loh,
                pval_fragment_joins=p_joins,
                pval_exp_cluster=p_exp,
                pval_chr_breakpoint_enrichment=p_enr,
                interchromosomal_chroms=p_chroms,
                linked_cluster_ids=l_cids,
                confidence=tier,
            )
        )

    return ShatterSeekResult(
        clusters=final_clusters,
        genome=genome,
        n_total_intrachromosomal_svs=n_total_intra,
        n_total_translocations=len(tra_records),
    )
