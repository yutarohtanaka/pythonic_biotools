"""
Bivariate Hotelling-T2 Circular Binary Segmentation (pre_seg + seg),
post-segmentation clustering, and diploid baseline (`dipLogR`) estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocess import PreProcResult, estimate_logr_sigma

_MAD_NORMAL_CONST: float = 0.6744897501960817


@dataclass(frozen=True)
class SegmentedResult:
    """Result container from `proc_sample`."""

    segments: pd.DataFrame
    jointseg: pd.DataFrame
    dip_logr: float
    candidate_dip_logrs: tuple[float, ...]
    cval: float
    min_nhet: int
    sigma_logr: float
    sigma_logor2: float

    def to_dataframe(self) -> pd.DataFrame:
        """Return segment-level summary DataFrame."""
        return self.segments.copy()

    def summary(self) -> str:
        """Return human-readable summary of segmentation and clustering."""
        n_segs = len(self.segments)
        n_clust = (
            int(self.segments["segclust"].nunique()) if n_segs > 0 else 0
        )
        med_snps = (
            float(self.segments["num_mark"].median()) if n_segs > 0 else 0.0
        )
        return (
            f"FACETS SegmentedResult (cval={self.cval}, min_nhet={self.min_nhet})\n"
            f"  Segments:        {n_segs}\n"
            f"  Clusters:        {n_clust}\n"
            f"  dipLogR:         {self.dip_logr:.4f}\n"
            f"  Median SNPs/seg: {med_snps:.1f}\n"
            f"  Sigma logR:      {self.sigma_logr:.4f}\n"
            f"  Sigma logOR2:    {self.sigma_logor2:.4f}"
        )


def _smooth_cna_spikes(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Olshen-Venkatraman single-point outlier smoothing: Winsorize isolated
    1-SNP spikes > 4*sigma from both neighbors when neighbors agree within 2*sigma.
    """
    out = x.copy()
    if out.size < 3 or sigma <= 0.0:
        return out
    prev = x[:-2]
    curr = x[1:-1]
    nxt = x[2:]
    d_prev = curr - prev
    d_nxt = curr - nxt
    nbr_diff = np.abs(nxt - prev)
    is_spike = (
        (np.abs(d_prev) > 4.0 * sigma)
        & (np.abs(d_nxt) > 4.0 * sigma)
        & ((d_prev * d_nxt) > 0.0)
        & (nbr_diff <= 2.0 * sigma)
    )
    if np.any(is_spike):
        med_nbr = 0.5 * (prev + nxt)
        sign = np.sign(curr - med_nbr)
        out[1:-1] = np.where(is_spike, med_nbr + sign * 2.0 * sigma, curr)
    return out


def _estimate_logor2_sigma(z2_het: np.ndarray) -> float:
    """
    Robustly estimate the null SD of Winsorized chi2_1 statistics `z2 = valor^2 / lorvar`
    using rolling block variances so isolated breakpoints and imbalanced segments
    do not inflate the baseline noise scale.
    """
    if z2_het.size < 4:
        return float(np.sqrt(2.0))
    block = 25
    if z2_het.size >= 2 * block:
        n_blocks = z2_het.size // block
        trimmed = z2_het[: n_blocks * block].reshape(n_blocks, block)
        b_means = np.mean(trimmed, axis=1)
        b_vars = np.var(trimmed, axis=1, ddof=1)
        bal_mask = b_means <= 1.5
        if np.any(bal_mask):
            bal_vars = b_vars[bal_mask]
            if bal_vars.size >= 3:
                est_var = float(np.quantile(bal_vars, 0.35)) / 0.88
            else:
                est_var = float(np.median(bal_vars))
            bal_mean = float(np.mean(trimmed[bal_mask]))
            est_var = min(est_var, 2.0 * max(bal_mean, 1.0) ** 2)
        else:
            est_var = 2.0
    else:
        d2 = 0.5 * (np.diff(z2_het) ** 2)
        hi = float(np.quantile(d2, 0.90))
        est_var = float(np.mean(d2[d2 <= hi])) if np.any(d2 <= hi) else float(np.mean(d2))

    if float(np.mean(z2_het)) > 0.2:
        return float(max(np.sqrt(2.0), np.sqrt(max(est_var, 0.0))))
    return float(max(1e-3, np.sqrt(max(est_var, 0.0))))


def _estimate_r12(
    x1: np.ndarray,
    x2: np.ndarray,
    het: np.ndarray,
) -> float:
    """Estimate local residual correlation r12 between consecutive het diffs."""
    idx = np.flatnonzero(het)
    if idx.size < 10:
        return 0.0
    adj = idx[1:] == (idx[:-1] + 1)
    if int(np.sum(adj)) >= 10:
        d1 = np.diff(x1[idx])[adj]
        d2 = np.diff(x2[idx])[adj]
    else:
        d1 = np.diff(x1[idx])
        d2 = np.diff(x2[idx])
    s1 = float(np.std(d1))
    s2 = float(np.std(d2))
    if s1 <= 1e-12 or s2 <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(d1, d2)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -0.5, 0.5))


def _two_sample_hotelling_t2(
    m1_a: float,
    n1_a: int,
    m1_b: float,
    n1_b: int,
    m2_a: float,
    n2_a: int,
    m2_b: float,
    n2_b: int,
    sigma1: float,
    sigma2: float,
    r12: float,
) -> float:
    """Two-sample bivariate Hotelling-T2 statistic between two segments/clusters."""
    if n1_a <= 0 or n1_b <= 0:
        return 0.0
    se1 = sigma1 * np.sqrt(1.0 / n1_a + 1.0 / n1_b)
    t1 = (m1_a - m1_b) / max(se1, 1e-12)
    if n2_a >= 2 and n2_b >= 2:
        se2 = sigma2 * np.sqrt(1.0 / n2_a + 1.0 / n2_b)
        t2 = (m2_a - m2_b) / max(se2, 1e-12)
    else:
        t2 = 0.0
    denom = max(1.0 - r12 * r12, 1e-6)
    return float((t1 * t1 + t2 * t2 - 2.0 * r12 * t1 * t2) / denom)


def _eval_arcs_vectorized(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    ph: np.ndarray,
    n: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    r12: float,
    min_seg_snps: int,
) -> np.ndarray:
    """Evaluate bivariate Hotelling-T2 across vectorized arrays of arcs [i, j)."""
    m1 = j_idx - i_idx
    m1_out = n - m1
    valid = (m1 >= min_seg_snps) & (m1_out >= min_seg_snps)
    t2_out = np.zeros(i_idx.shape, dtype=np.float64)
    if not np.any(valid):
        return t2_out

    iv = i_idx[valid]
    jv = j_idx[valid]
    m1v = m1[valid].astype(np.float64)
    m1ov = m1_out[valid].astype(np.float64)

    total1 = p1[-1]
    s1_in = p1[jv] - p1[iv]
    s1_out = total1 - s1_in
    mean1_in = s1_in / m1v
    mean1_out = s1_out / m1ov
    se1 = sigma1 * np.sqrt(1.0 / m1v + 1.0 / m1ov)
    t1 = (mean1_in - mean1_out) / np.maximum(se1, 1e-12)

    t2_comp = np.zeros(iv.shape, dtype=np.float64)
    if n2 >= 4:
        m2 = (ph[jv] - ph[iv]).astype(np.float64)
        m2_out = float(n2) - m2
        h_valid = (m2 >= 2.0) & (m2_out >= 2.0)
        if np.any(h_valid):
            total2 = p2[-1]
            iv_h = iv[h_valid]
            jv_h = jv[h_valid]
            m2_h = m2[h_valid]
            m2o_h = m2_out[h_valid]
            s2_in = p2[jv_h] - p2[iv_h]
            s2_out = total2 - s2_in
            se2 = sigma2 * np.sqrt(1.0 / m2_h + 1.0 / m2o_h)
            t2_comp[h_valid] = ((s2_in / m2_h) - (s2_out / m2o_h)) / np.maximum(
                se2, 1e-12
            )

    denom = max(1.0 - r12 * r12, 1e-6)
    t2_out[valid] = (t1 * t1 + t2_comp * t2_comp - 2.0 * r12 * t1 * t2_comp) / denom
    return t2_out


def _find_best_arc(
    x1: np.ndarray,
    x2: np.ndarray,
    het: np.ndarray,
    sigma1: float,
    sigma2: float,
    r12: float,
    min_seg_snps: int,
) -> tuple[float, int, int] | None:
    """Find the circular arc [i, j) maximizing bivariate Hotelling-T2."""
    n = x1.size
    if n < 2 * min_seg_snps:
        return None

    n2 = int(np.sum(het))
    p1 = np.empty(n + 1, dtype=np.float64)
    p1[0] = 0.0
    np.cumsum(x1, out=p1[1:])

    p2 = np.empty(n + 1, dtype=np.float64)
    p2[0] = 0.0
    np.cumsum(np.where(het, x2, 0.0), out=p2[1:])

    ph = np.empty(n + 1, dtype=np.int64)
    ph[0] = 0
    np.cumsum(het.astype(np.int64), out=ph[1:])

    best_t2 = -1.0
    best_i = 0
    best_j = n

    def _update(ii: np.ndarray, jj: np.ndarray) -> None:
        nonlocal best_t2, best_i, best_j
        if ii.size == 0:
            return
        vals = _eval_arcs_vectorized(
            ii, jj, p1, p2, ph, n, n2, sigma1, sigma2, r12, min_seg_snps
        )
        idx_max = int(np.argmax(vals))
        val_max = float(vals[idx_max])
        if val_max > best_t2:
            best_t2 = val_max
            best_i = int(ii[idx_max])
            best_j = int(jj[idx_max])

    if n <= 600:
        # Exact full upper-triangular scan in chunked blocks
        block = 64
        for i0 in range(0, n - min_seg_snps + 1, block):
            i1 = min(i0 + block, n - min_seg_snps + 1)
            ii_list = []
            jj_list = []
            for i in range(i0, i1):
                j_hi = n if i > 0 else n - min_seg_snps
                j_lo = i + min_seg_snps
                if j_lo <= j_hi:
                    js = np.arange(j_lo, j_hi + 1, dtype=np.int64)
                    ii_list.append(np.full(js.size, i, dtype=np.int64))
                    jj_list.append(js)
            if ii_list:
                _update(np.concatenate(ii_list), np.concatenate(jj_list))
    else:
        # 1. Exact single-breakpoint splits (i=0 or j=n)
        js0 = np.arange(min_seg_snps, n - min_seg_snps + 1, dtype=np.int64)
        _update(np.zeros(js0.size, dtype=np.int64), js0)
        is_n = np.arange(min_seg_snps, n - min_seg_snps + 1, dtype=np.int64)
        _update(is_n, np.full(is_n.size, n, dtype=np.int64))

        # 2. Coarse global grid for wide multi-breakpoint arcs
        grid = np.unique(np.linspace(0, n, 220, dtype=np.int64))
        ga, gb = np.triu_indices(grid.size, k=1)
        _update(grid[ga], grid[gb])

        # 3. Overlapping local windows of width 400 (stride 200) for focal CNVs
        w_size = 400
        stride = 200
        for w0 in range(0, n, stride):
            w1 = min(w0 + w_size, n)
            nw = w1 - w0
            if nw < min_seg_snps:
                continue
            la, lb = np.triu_indices(nw + 1, k=min_seg_snps)
            _update(w0 + la, w0 + lb)
            if w1 == n:
                break

        # 4. Exact alternating 1D coordinate refinement around best arc
        for _ in range(3):
            prev_pair = (best_i, best_j)
            j_cur = best_j
            i_cands = np.arange(0, j_cur - min_seg_snps + 1, dtype=np.int64)
            _update(i_cands, np.full(i_cands.size, j_cur, dtype=np.int64))

            i_cur = best_i
            j_cands = np.arange(i_cur + min_seg_snps, n + 1, dtype=np.int64)
            _update(np.full(j_cands.size, i_cur, dtype=np.int64), j_cands)

            if (best_i, best_j) == prev_pair:
                break

        # Local 2D grid check around refined optimum
        i_lo = max(0, best_i - 12)
        i_hi = min(n - min_seg_snps, best_i + 12)
        j_lo = max(i_lo + min_seg_snps, best_j - 12)
        j_hi = min(n, best_j + 12)
        if i_lo <= i_hi and j_lo <= j_hi:
            ii_grid, jj_grid = np.meshgrid(
                np.arange(i_lo, i_hi + 1, dtype=np.int64),
                np.arange(j_lo, j_hi + 1, dtype=np.int64),
                indexing="ij",
            )
            _update(ii_grid.ravel(), jj_grid.ravel())

    if best_t2 < 0.0:
        return None
    return best_t2, best_i, best_j


def hotelling_t2_cbs(
    cnlr: np.ndarray,
    logor2_stat: np.ndarray | None = None,
    het: np.ndarray | None = None,
    *,
    cval: float = 150.0,
    min_seg_snps: int = 5,
    sigma_logr: float | None = None,
    sigma_logor2: float | None = None,
    r12: float | None = None,
    winsor_z2: float = 25.0,
) -> list[tuple[int, int]]:
    """
    Segment a single chromosome arm via bivariate Hotelling-T2 Circular Binary
    Segmentation (CBS) over `cnlr` (all SNPs) and `logor2_stat` (het SNPs).

    Returns a sorted list of half-open index slices `[(start, end), ...]` partitioning `[0, n)`.
    """
    x1 = np.asarray(cnlr, dtype=np.float64)
    n = x1.size
    if n == 0:
        return []
    if n < 2 * min_seg_snps:
        return [(0, n)]

    if logor2_stat is None or het is None:
        het_mask = np.zeros(n, dtype=bool)
        x2 = np.zeros(n, dtype=np.float64)
    else:
        het_mask = np.asarray(het, dtype=bool)
        raw_z2 = np.asarray(logor2_stat, dtype=np.float64)
        valid_z2 = het_mask & np.isfinite(raw_z2)
        het_mask = valid_z2
        x2 = np.where(het_mask, np.minimum(np.maximum(raw_z2, 0.0), winsor_z2), 0.0)

    s1 = float(sigma_logr) if sigma_logr is not None else estimate_logr_sigma(x1)
    s1 = max(s1, 1e-4)
    if sigma_logor2 is not None:
        s2 = max(float(sigma_logor2), 1e-4)
    else:
        s2 = _estimate_logor2_sigma(x2[het_mask])
    corr12 = float(r12) if r12 is not None else _estimate_r12(x1, x2, het_mask)

    stack: list[tuple[int, int]] = [(0, n)]
    segs: list[tuple[int, int]] = []
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_seg_snps:
            segs.append((lo, hi))
            continue
        res = _find_best_arc(
            x1[lo:hi],
            x2[lo:hi],
            het_mask[lo:hi],
            s1,
            s2,
            corr12,
            min_seg_snps,
        )
        if res is None or res[0] <= cval:
            segs.append((lo, hi))
            continue
        _, rel_i, rel_j = res
        abs_i = lo + rel_i
        abs_j = lo + rel_j
        children: list[tuple[int, int]] = []
        if abs_i > lo:
            children.append((lo, abs_i))
        children.append((abs_i, abs_j))
        if abs_j < hi:
            children.append((abs_j, hi))
        if len(children) <= 1:
            segs.append((lo, hi))
        else:
            for ch in reversed(children):
                stack.append(ch)

    segs.sort(key=lambda pair: pair[0])

    # Post-CBS adjacent-segment pruning
    while len(segs) > 1:
        adj_t2: list[float] = []
        for idx in range(len(segs) - 1):
            a0, a1 = segs[idx]
            b0, b1 = segs[idx + 1]
            na1 = a1 - a0
            nb1 = b1 - b0
            if na1 < min_seg_snps or nb1 < min_seg_snps:
                adj_t2.append(-1.0)
                continue
            ha = het_mask[a0:a1]
            hb = het_mask[b0:b1]
            n2a = int(np.sum(ha))
            n2b = int(np.sum(hb))
            m1a = float(np.mean(x1[a0:a1]))
            m1b = float(np.mean(x1[b0:b1]))
            m2a = float(np.mean(x2[a0:a1][ha])) if n2a > 0 else 0.0
            m2b = float(np.mean(x2[b0:b1][hb])) if n2b > 0 else 0.0
            s2_adj = s2
            if min(n2a, n2b) < 10 and min(m2a, m2b) > 1.5:
                # Inside an already-imbalanced arm, z2 ~ non-central chi2_1(lambda)
                # has variance 2 + 4*lambda; protect tiny <10-het runs from tail spikes.
                lam = max(0.0, min(m2a, m2b) - 1.0)
                s2_adj = max(s2, float(np.sqrt(2.0 + 4.0 * lam)))
            adj_t2.append(
                _two_sample_hotelling_t2(
                    m1a, na1, m1b, nb1, m2a, n2a, m2b, n2b, s1, s2_adj, corr12
                )
            )
        min_idx = int(np.argmin(adj_t2))
        if adj_t2[min_idx] < cval:
            a0, _ = segs[min_idx]
            _, b1 = segs[min_idx + 1]
            segs[min_idx : min_idx + 2] = [(a0, b1)]
        else:
            break

    return segs


def _compute_mafr(valor: np.ndarray, lorvar: np.ndarray) -> float:
    """Variance-debiased inverse-variance-weighted squared allelic logOR."""
    valid = np.isfinite(valor) & np.isfinite(lorvar) & (lorvar > 0.0)
    if not np.any(valid):
        return 0.0
    v = valor[valid]
    lv = lorvar[valid]
    w = 1.0 / lv
    est = float(np.sum(w * (v * v - lv)) / np.sum(w))
    return max(0.0, est)


def cluster_segments(
    segments_df: pd.DataFrame,
    snps_df: pd.DataFrame,
    *,
    cval: float = 150.0,
    min_nhet: int = 15,
    delta_cn: float = 0.5,
    sigma_logr: float = 0.2,
    sigma_logor2: float = float(np.sqrt(2.0)),
    r12: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hierarchically cluster segmented genomic intervals sharing statistically
    indistinguishable (cnlr, logOR^2) profiles using bivariate Hotelling-T2.
    """
    segs = segments_df.copy()
    snps = snps_df.copy()
    n_segs = len(segs)
    if n_segs == 0:
        segs["segclust"] = np.empty(0, dtype=np.int64)
        snps["segclust"] = np.empty(len(snps), dtype=np.int64)
        return segs, snps

    seg_ids = segs["seg_idx"].to_numpy(dtype=np.int64)
    snp_seg_idx = snps["seg_idx"].to_numpy(dtype=np.int64)
    x1 = snps["cnlr"].to_numpy(dtype=np.float64)
    het = snps["het"].to_numpy(dtype=bool) & np.isfinite(
        snps["logor2_stat"].to_numpy(dtype=np.float64)
    )
    x2 = np.where(
        het,
        np.minimum(np.maximum(snps["logor2_stat"].to_numpy(dtype=np.float64), 0.0), 25.0),
        0.0,
    )

    # Precompute sufficient stats per segment
    seg_stats: dict[int, dict[str, float | int]] = {}
    for s_id in seg_ids:
        mask = snp_seg_idx == s_id
        n1 = int(np.sum(mask))
        s1 = float(np.sum(x1[mask]))
        h_mask = mask & het
        n2 = int(np.sum(h_mask))
        s2 = float(np.sum(x2[h_mask]))
        seg_stats[int(s_id)] = {"n1": n1, "s1": s1, "n2": n2, "s2": s2}

    high_het_ids = [
        int(s_id) for s_id in seg_ids if seg_stats[int(s_id)]["n2"] >= min_nhet
    ]
    low_het_ids = [
        int(s_id) for s_id in seg_ids if seg_stats[int(s_id)]["n2"] < min_nhet
    ]

    # Agglomerative clustering over allelic-informative segments
    clusters: list[dict] = [
        {
            "segs": [s_id],
            "n1": seg_stats[s_id]["n1"],
            "s1": seg_stats[s_id]["s1"],
            "n2": seg_stats[s_id]["n2"],
            "s2": seg_stats[s_id]["s2"],
        }
        for s_id in high_het_ids
    ]

    while len(clusters) > 1:
        best_dist = np.inf
        best_pair = (-1, -1)
        best_should_merge = False
        for a in range(len(clusters)):
            ca = clusters[a]
            m1a = ca["s1"] / max(ca["n1"], 1)
            m2a = ca["s2"] / max(ca["n2"], 1) if ca["n2"] > 0 else 0.0
            for b in range(a + 1, len(clusters)):
                cb = clusters[b]
                m1b = cb["s1"] / max(cb["n1"], 1)
                m2b = cb["s2"] / max(cb["n2"], 1) if cb["n2"] > 0 else 0.0

                t2_pair = _two_sample_hotelling_t2(
                    m1a,
                    ca["n1"],
                    m1b,
                    cb["n1"],
                    m2a,
                    ca["n2"],
                    m2b,
                    cb["n2"],
                    sigma_logr,
                    sigma_logor2,
                    r12,
                )
                se2 = sigma_logor2 * np.sqrt(
                    1.0 / max(ca["n2"], 1) + 1.0 / max(cb["n2"], 1)
                )
                t2_only = ((m2a - m2b) / max(se2, 1e-12)) ** 2
                tiny_logr_diff = abs(m1a - m1b) < 0.05 * delta_cn

                should_merge = (t2_pair <= cval) or (
                    tiny_logr_diff and t2_only <= cval
                )
                if should_merge and t2_pair < best_dist:
                    best_dist = t2_pair
                    best_pair = (a, b)
                    best_should_merge = True

        if not best_should_merge:
            break
        a, b = best_pair
        ca = clusters[a]
        cb = clusters[b]
        merged = {
            "segs": ca["segs"] + cb["segs"],
            "n1": ca["n1"] + cb["n1"],
            "s1": ca["s1"] + cb["s1"],
            "n2": ca["n2"] + cb["n2"],
            "s2": ca["s2"] + cb["s2"],
        }
        clusters[a] = merged
        clusters.pop(b)

    # Assign low-nhet segments by univariate cnlr distance to existing clusters
    for s_id in low_het_ids:
        st = seg_stats[s_id]
        m1_s = st["s1"] / max(st["n1"], 1)
        best_c = -1
        best_t1_sq = np.inf
        for c_idx, c in enumerate(clusters):
            m1_c = c["s1"] / max(c["n1"], 1)
            se1 = sigma_logr * np.sqrt(1.0 / max(st["n1"], 1) + 1.0 / max(c["n1"], 1))
            t1_sq = ((m1_s - m1_c) / max(se1, 1e-12)) ** 2
            if t1_sq < best_t1_sq:
                best_t1_sq = t1_sq
                best_c = c_idx
        if best_c >= 0 and (
            best_t1_sq <= cval
            or abs(m1_s - (clusters[best_c]["s1"] / max(clusters[best_c]["n1"], 1)))
            < 0.1 * delta_cn
        ):
            clusters[best_c]["segs"].append(s_id)
            clusters[best_c]["n1"] += st["n1"]
            clusters[best_c]["s1"] += st["s1"]
            clusters[best_c]["n2"] += st["n2"]
            clusters[best_c]["s2"] += st["s2"]
        else:
            clusters.append(
                {
                    "segs": [s_id],
                    "n1": st["n1"],
                    "s1": st["s1"],
                    "n2": st["n2"],
                    "s2": st["s2"],
                }
            )

    # Sort clusters deterministically by minimum seg_idx
    clusters.sort(key=lambda c: min(c["segs"]))
    seg_to_clust: dict[int, int] = {}
    for clust_id, c in enumerate(clusters):
        for s_id in c["segs"]:
            seg_to_clust[s_id] = clust_id

    segs["segclust"] = [seg_to_clust[int(s)] for s in seg_ids]
    snps["segclust"] = [seg_to_clust[int(s)] for s in snp_seg_idx]
    return segs, snps


def find_diploid_logr(
    segments_df: pd.DataFrame,
    snps_df: pd.DataFrame,
    *,
    min_nhet: int = 15,
    diploid_logr: float | None = None,
) -> tuple[float, tuple[float, ...]]:
    """
    Identify allelically balanced clusters and select the primary diploid logR
    baseline (`dipLogR`) plus ordered candidate baselines for multi-start EM.
    """
    if diploid_logr is not None:
        val = float(diploid_logr)
        return val, (val,)

    if len(segments_df) == 0 or len(snps_df) == 0:
        return 0.0, (0.0,)

    total_snps = len(snps_df)
    valor = snps_df["valor"].to_numpy(dtype=np.float64)
    lorvar = snps_df["lorvar"].to_numpy(dtype=np.float64)
    cnlr = snps_df["cnlr"].to_numpy(dtype=np.float64)
    z2 = snps_df["logor2_stat"].to_numpy(dtype=np.float64)
    het = snps_df["het"].to_numpy(dtype=bool) & np.isfinite(z2)
    clust_arr = snps_df["segclust"].to_numpy(dtype=np.int64)

    balanced_cands: list[tuple[int, float, float]] = []
    all_cands: list[tuple[float, int, float]] = []

    for cid in np.unique(clust_arr):
        mask = clust_arr == cid
        n_snps = int(np.sum(mask))
        h_mask = mask & het
        n_h = int(np.sum(h_mask))
        c_mean = float(np.mean(cnlr[mask]))
        if n_h >= min_nhet:
            mafr = _compute_mafr(valor[h_mask], lorvar[h_mask])
            s_o = float(np.sum(z2[h_mask] - 1.0) / np.sqrt(2.0 * n_h))
            all_cands.append((mafr, n_snps, c_mean))
            if s_o <= 3.0 or mafr <= 0.03:
                balanced_cands.append((n_snps, mafr, c_mean))

    if balanced_cands:
        major = [c for c in balanced_cands if c[0] >= 0.05 * total_snps]
        pool = major if major else balanced_cands
        # Primary: largest SNP span, tie-break lower mafR
        pool.sort(key=lambda item: (-item[0], item[1]))
        primary = float(pool[0][2])
        unique_cands: list[float] = []
        for _, _, val in pool:
            fval = float(val)
            if not any(abs(fval - u) < 1e-3 for u in unique_cands):
                unique_cands.append(fval)
        return primary, tuple(unique_cands)

    if all_cands:
        all_cands.sort(key=lambda item: (item[0], -item[1]))
        primary = float(all_cands[0][2])
        return primary, (primary,)

    primary = float(np.median(cnlr))
    return primary, (primary,)


def proc_sample(
    preproc: PreProcResult,
    *,
    cval: float | None = None,
    pre_cval: float | None = None,
    min_nhet: int | None = None,
    min_seg_snps: int | None = None,
    diploid_logr: float | None = None,
    delta_cn: float | None = None,
) -> SegmentedResult:
    """
    Run two-stage bivariate Hotelling-T2 CBS (`pre_seg` + `seg`), cluster segments,
    and estimate the diploid logR baseline (`dipLogR`).
    """
    cfg = preproc.config
    eff_pre_cval = float(pre_cval if pre_cval is not None else cfg.pre_cval)
    eff_cval = float(cval if cval is not None else cfg.cval)
    eff_min_nhet = int(min_nhet if min_nhet is not None else cfg.min_nhet)
    eff_min_seg = int(min_seg_snps if min_seg_snps is not None else cfg.min_seg_snps)
    eff_dip = diploid_logr if diploid_logr is not None else cfg.diploid_logr
    eff_delta_cn = float(delta_cn if delta_cn is not None else cfg.delta_cn)

    snps = preproc.snps.copy()
    n_total = len(snps)
    if n_total == 0:
        raise ValueError("Cannot segment an empty PreProcResult.")

    cnlr_all = snps["cnlr"].to_numpy(dtype=np.float64)
    het_all = snps["het"].to_numpy(dtype=bool) & np.isfinite(
        snps["logor2_stat"].to_numpy(dtype=np.float64)
    )
    z2_all = np.where(
        het_all,
        np.minimum(np.maximum(snps["logor2_stat"].to_numpy(dtype=np.float64), 0.0), 25.0),
        0.0,
    )
    valor_all = snps["valor"].to_numpy(dtype=np.float64)
    lorvar_all = snps["lorvar"].to_numpy(dtype=np.float64)
    pos_all = snps["pos"].to_numpy(dtype=np.int64)
    arm_all = snps["arm"].to_numpy()
    chrom_all = snps["chrom"].to_numpy()

    # Preserve arm order as sorted in snps
    arm_slices: list[tuple[str, int, int]] = []
    start_idx = 0
    for i in range(1, n_total + 1):
        if i == n_total or arm_all[i] != arm_all[start_idx]:
            arm_slices.append((str(arm_all[start_idx]), start_idx, i))
            start_idx = i

    sigma1_pre = preproc.sigma_logr
    sigma2_pre = _estimate_logor2_sigma(z2_all[het_all])
    r12 = _estimate_r12(cnlr_all, z2_all, het_all)

    # Smooth 1-SNP logR spikes on working array for segmentation
    cnlr_work = cnlr_all.copy()
    for _, a0, a1 in arm_slices:
        cnlr_work[a0:a1] = _smooth_cna_spikes(cnlr_work[a0:a1], sigma1_pre)

    # Stage 1: Sensitive pre-segmentation to estimate within-segment residual noise
    pre_resids1: list[np.ndarray] = []
    pre_vars2: list[float] = []
    bal_z2_chunks: list[np.ndarray] = []
    for _, a0, a1 in arm_slices:
        pre_segs = hotelling_t2_cbs(
            cnlr_work[a0:a1],
            z2_all[a0:a1],
            het_all[a0:a1],
            cval=eff_pre_cval,
            min_seg_snps=eff_min_seg,
            sigma_logr=sigma1_pre,
            sigma_logor2=sigma2_pre,
            r12=r12,
        )
        for s0, s1 in pre_segs:
            g0, g1 = a0 + s0, a0 + s1
            if g1 - g0 >= 5:
                seg_x1 = cnlr_work[g0:g1]
                pre_resids1.append(seg_x1 - float(np.median(seg_x1)))
            h_sub = het_all[g0:g1]
            if int(np.sum(h_sub)) >= 10:
                z2_sub = z2_all[g0:g1][h_sub]
                if float(np.mean(z2_sub)) <= 1.5:
                    pre_vars2.append(float(np.var(z2_sub, ddof=1)))
                    bal_z2_chunks.append(z2_sub)

    if pre_resids1:
        all_r1 = np.concatenate(pre_resids1)
        sigma1_seg = float(np.median(np.abs(all_r1))) / _MAD_NORMAL_CONST
        sigma1_seg = max(sigma1_seg, 0.5 * sigma1_pre, 1e-4)
    else:
        sigma1_seg = sigma1_pre

    has_het_signal = (
        int(np.sum(het_all)) > 0 and float(np.mean(z2_all[het_all])) > 0.2
    )
    if pre_vars2:
        # Estimate baseline logOR^2 SD only from allelically balanced pre-segments
        # (mean(z2) <= 1.5), avoiding non-central chi2_1 variance inflation.
        if len(pre_vars2) >= 3:
            est_v2 = float(np.quantile(pre_vars2, 0.35)) / 0.88
        else:
            est_v2 = float(np.median(pre_vars2))
        bal_mean = float(np.mean(np.concatenate(bal_z2_chunks)))
        est_v2 = min(est_v2, 2.0 * max(bal_mean, 1.0) ** 2)
        if has_het_signal:
            sigma2_seg = float(max(np.sqrt(2.0), np.sqrt(max(est_v2, 0.0))))
        else:
            sigma2_seg = float(max(1e-3, np.sqrt(max(est_v2, 0.0))))
    else:
        sigma2_seg = float(np.sqrt(2.0)) if has_het_signal else sigma2_pre

    # Stage 2: Primary bivariate Hotelling-T2 CBS
    seg_rows: list[dict] = []
    snp_seg_idx = np.empty(n_total, dtype=np.int64)
    seg_counter = 0

    for arm_name, a0, a1 in arm_slices:
        arm_segs = hotelling_t2_cbs(
            cnlr_work[a0:a1],
            z2_all[a0:a1],
            het_all[a0:a1],
            cval=eff_cval,
            min_seg_snps=eff_min_seg,
            sigma_logr=sigma1_seg,
            sigma_logor2=sigma2_seg,
            r12=r12,
        )
        for s0, s1 in arm_segs:
            g0, g1 = a0 + s0, a0 + s1
            snp_seg_idx[g0:g1] = seg_counter
            seg_x1 = cnlr_all[g0:g1]
            h_sub = het_all[g0:g1]
            n_het_seg = int(np.sum(h_sub))
            mafr = _compute_mafr(valor_all[g0:g1], lorvar_all[g0:g1])
            seg_rows.append(
                {
                    "chrom": str(chrom_all[g0]),
                    "arm": arm_name,
                    "seg_idx": seg_counter,
                    "start": int(pos_all[g0]),
                    "end": int(pos_all[g1 - 1]),
                    "num_mark": int(g1 - g0),
                    "nhet": n_het_seg,
                    "cnlr_mean": float(np.mean(seg_x1)),
                    "cnlr_median": float(np.median(seg_x1)),
                    "cnlr_sd": float(np.std(seg_x1, ddof=1)) if (g1 - g0) > 1 else 0.0,
                    "mafR": mafr,
                    "low_nhet": n_het_seg < eff_min_nhet,
                }
            )
            seg_counter += 1

    snps["seg_idx"] = snp_seg_idx
    segments_df = pd.DataFrame(seg_rows)

    segments_df, snps = cluster_segments(
        segments_df,
        snps,
        cval=eff_cval,
        min_nhet=eff_min_nhet,
        delta_cn=eff_delta_cn,
        sigma_logr=sigma1_seg,
        sigma_logor2=sigma2_seg,
        r12=r12,
    )

    col_order = [
        "chrom",
        "arm",
        "seg_idx",
        "segclust",
        "start",
        "end",
        "num_mark",
        "nhet",
        "cnlr_mean",
        "cnlr_median",
        "cnlr_sd",
        "mafR",
        "low_nhet",
    ]
    segments_df = segments_df[col_order]

    dip_logr, cand_dips = find_diploid_logr(
        segments_df,
        snps,
        min_nhet=eff_min_nhet,
        diploid_logr=eff_dip,
    )

    return SegmentedResult(
        segments=segments_df,
        jointseg=snps,
        dip_logr=dip_logr,
        candidate_dip_logrs=cand_dips,
        cval=eff_cval,
        min_nhet=eff_min_nhet,
        sigma_logr=sigma1_seg,
        sigma_logor2=sigma2_seg,
    )
