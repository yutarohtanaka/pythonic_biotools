"""
Variance-stabilized CBS segmentation (`segment_bins`) and empirical-Bayes 5-state
Gaussian mixture calling (`call_bins`) for `qdnaseq`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats

from .config import SettingsMode, get_qdnaseq_config, validate_settings
from .correction import normalize_bins
from .data import QDNAseqData


def _diff_noise_sd(y: np.ndarray) -> float:
    if y.size < 3:
        return max(float(np.std(y)), 0.02) if y.size > 1 else 0.05
    diffs = np.diff(y) / np.sqrt(2.0)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sd = 1.4826 * mad
    if sd <= 1e-7:
        sd = float(np.std(diffs))
    return max(sd, 1e-3)


def _cbs_segment_1d(
    y: np.ndarray,
    *,
    alpha: float = 1e-4,
    undo_sd: float = 1.0,
    min_bins: int = 3,
) -> list[tuple[int, int]]:
    """Recursive binary + ternary CBS segmentation with SD-undo post-pruning."""
    n = y.size
    if n == 0:
        return []
    if n < 2 * min_bins:
        return [(0, n)]

    noise_sd = _diff_noise_sd(y)
    prefix = np.concatenate([[0.0], np.cumsum(y)])
    z_base = float(stats.norm.isf(alpha / 2.0))
    cuts: set[int] = {0, n}

    def _split(lo: int, hi: int) -> None:
        m = hi - lo
        if m < 2 * min_bins:
            return
        ks = np.arange(lo + min_bins, hi - min_bins + 1)
        nl = (ks - lo).astype(np.float64)
        nr = (hi - ks).astype(np.float64)
        ml = (prefix[ks] - prefix[lo]) / nl
        mr = (prefix[hi] - prefix[ks]) / nr
        se = noise_sd * np.sqrt(1.0 / nl + 1.0 / nr)
        zs = np.abs(ml - mr) / np.maximum(se, 1e-9)
        best_i = int(np.argmax(zs))
        max_z = float(zs[best_i])
        best_k = int(ks[best_i])

        best_arc: tuple[int, int] | None = None
        if m >= 3 * min_bins and m <= 600:
            step = max(1, m // 80)
            grid = np.arange(lo + min_bins, hi - min_bins + 1, step)
            tot = prefix[hi] - prefix[lo]
            for gi in range(grid.size):
                p1 = int(grid[gi])
                for gj in range(gi + 1, grid.size):
                    p2 = int(grid[gj])
                    alen = p2 - p1
                    if alen < min_bins or (m - alen) < min_bins:
                        continue
                    asum = prefix[p2] - prefix[p1]
                    min_in = asum / alen
                    mout = (tot - asum) / (m - alen)
                    z_arc = abs(min_in - mout) / max(
                        noise_sd * np.sqrt(1.0 / alen + 1.0 / (m - alen)), 1e-9
                    )
                    if z_arc > max_z + 0.5:
                        max_z = z_arc
                        best_arc = (p1, p2)

        crit = max(z_base, np.sqrt(2.0 * np.log(max(m, 2))))
        if max_z >= crit:
            if best_arc is not None:
                p1, p2 = best_arc
                cuts.add(p1)
                cuts.add(p2)
                _split(lo, p1)
                _split(p1, p2)
                _split(p2, hi)
            else:
                cuts.add(best_k)
                _split(lo, best_k)
                _split(best_k, hi)

    _split(0, n)
    pts = sorted(cuts)
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    if undo_sd > 0.0 and len(segs) > 1:
        changed = True
        while changed and len(segs) > 1:
            changed = False
            gaps = [
                abs(float(np.mean(y[segs[i][0] : segs[i][1]])) - float(np.mean(y[segs[i + 1][0] : segs[i + 1][1]])))
                for i in range(len(segs) - 1)
            ]
            idx_min = int(np.argmin(gaps))
            if gaps[idx_min] < undo_sd * noise_sd:
                segs = (
                    segs[:idx_min]
                    + [(segs[idx_min][0], segs[idx_min + 1][1])]
                    + segs[idx_min + 2 :]
                )
                changed = True

    return segs


def segment_bins(
    qdata: QDNAseqData,
    alpha: float | None = None,
    undo_sd: float | None = None,
    min_bins: int = 3,
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Segment normalized copy-number bins per chromosome (`QDNAseq::segmentBins`):
    - `settings="default"`: Anscombe variance-stabilized `log2(max(r, 0) + 3/(8*c0))`
    - `settings="R_default"`: Anscombe `sqrt(max(r, 0) + 3/(8*c0))` with `undo.SD = 1.0`.
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    eff_alpha = float(alpha) if alpha is not None else cfg.seg_alpha
    eff_undo = float(undo_sd) if undo_sd is not None else cfg.seg_undo_sd

    cur = qdata if qdata.copynumber is not None else normalize_bins(qdata, settings=cfg.settings)
    assert cur.copynumber is not None

    bins = cur.bins
    use = bins["use"].to_numpy(dtype=bool)

    seg_bin_cols: dict[str, np.ndarray] = {}
    seg_tables: dict[str, pd.DataFrame] = {}

    for sname in cur.sample_names:
        r_lin = cur.copynumber[sname].to_numpy(dtype=np.float64)
        raw_cnt = cur.counts[sname].to_numpy(dtype=np.float64)
        c0 = max(float(np.median(raw_cnt[use])) if np.any(use) else 50.0, 5.0)
        anscombe_eps = 3.0 / (8.0 * c0)

        if cfg.seg_transform == "sqrt":
            y_trans = np.sqrt(np.maximum(r_lin, 0.0) + anscombe_eps)
        else:
            y_trans = np.log2(np.maximum(r_lin, 0.0) + anscombe_eps)

        seg_arr = np.full(len(bins), np.nan, dtype=np.float64)
        rows: list[dict[str, Any]] = []

        for chrom in pd.unique(bins["chrom"]):
            c_mask = (bins["chrom"].to_numpy() == chrom) & use
            c_idx = np.flatnonzero(c_mask)
            if c_idx.size == 0:
                continue
            order = np.argsort(bins.loc[c_idx, "start"].to_numpy(), kind="mergesort")
            idx_sorted = c_idx[order]

            y_sub = y_trans[idx_sorted]
            r_sub = r_lin[idx_sorted]
            spans = _cbs_segment_1d(
                y_sub, alpha=eff_alpha, undo_sd=eff_undo, min_bins=max(1, int(min_bins))
            )

            for s_lo, s_hi in spans:
                b_idxs = idx_sorted[s_lo:s_hi]
                seg_lin_vals = r_sub[s_lo:s_hi]
                seg_val = float(np.mean(seg_lin_vals))
                seg_sd = (
                    float(np.std(seg_lin_vals, ddof=1))
                    if seg_lin_vals.size > 1
                    else 0.03
                )
                seg_arr[b_idxs] = seg_val
                rows.append(
                    {
                        "chrom": str(chrom),
                        "start": int(bins.loc[b_idxs[0], "start"]),
                        "end": int(bins.loc[b_idxs[-1], "end"]),
                        "num_bins": int(b_idxs.size),
                        "segmented": seg_val,
                        "log2_ratio": float(np.log2(max(seg_val, 1e-4))),
                        "seg_sd": max(seg_sd, 0.01),
                    }
                )

        seg_bin_cols[sname] = seg_arr
        seg_tables[sname] = pd.DataFrame(rows)

    return cur.clone(
        segmented=pd.DataFrame(seg_bin_cols),
        segments=seg_tables,
        settings=cfg.settings,
    )


def call_bins(
    qdata: QDNAseqData,
    cutoffs: Sequence[float] | None = None,
    settings: SettingsMode = "default",
) -> QDNAseqData:
    """
    Call discrete copy-number states `{-2, -1, 0, +1, +2}` and empirical-Bayes /
    5-component Gaussian mixture posterior probabilities (`prob_dloss`, `prob_loss`,
    `prob_normal`, `prob_gain`, `prob_amp`) matching `QDNAseq::callBins` / `CGHcall`.
    """
    cfg = get_qdnaseq_config(validate_settings(settings))
    cur = qdata if qdata.segmented is not None else segment_bins(qdata, settings=cfg.settings)
    assert cur.segmented is not None

    use = cur.bins["use"].to_numpy(dtype=bool)
    state_values = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    prob_names = ["prob_dloss", "prob_loss", "prob_normal", "prob_gain", "prob_amp"]

    call_cols: dict[str, np.ndarray] = {}
    prob_tables: dict[str, pd.DataFrame] = {}

    for sname in cur.sample_names:
        seg_lin = cur.segmented[sname].to_numpy(dtype=np.float64)
        valid = use & np.isfinite(seg_lin)
        log2_seg = np.log2(np.maximum(np.nan_to_num(seg_lin, nan=1.0), 1e-4))

        # Initial ordered 5-state centers in log2 ratio space: dloss, loss, normal, gain, amp
        means = np.array([-1.60, -0.42, 0.0, 0.35, 1.00], dtype=np.float64)
        sds = np.array([0.35, 0.16, 0.10, 0.16, 0.35], dtype=np.float64)
        weights = np.array([0.02, 0.14, 0.66, 0.15, 0.03], dtype=np.float64)

        x_obs = log2_seg[valid]
        if x_obs.size >= 10 and cutoffs is None:
            # Constrained 8-step EM keeping component order and neutral center anchored near 0
            for _ in range(8):
                ll = (
                    np.log(weights)[np.newaxis, :]
                    - 0.5 * ((x_obs[:, np.newaxis] - means[np.newaxis, :]) / sds[np.newaxis, :]) ** 2
                    - np.log(sds)[np.newaxis, :]
                )
                log_resp = ll - special.logsumexp(ll, axis=1, keepdims=True)
                resp = np.exp(log_resp)
                nk = np.sum(resp, axis=0) + 1.0

                weights = nk / np.sum(nk)
                # Update loss/gain centers gently within biologically valid intervals
                m_loss = float(np.sum(resp[:, 1] * x_obs) / nk[1])
                m_norm = float(np.sum(resp[:, 2] * x_obs) / nk[2])
                m_gain = float(np.sum(resp[:, 3] * x_obs) / nk[3])
                means[1] = np.clip(m_loss, -0.95, -0.20)
                means[2] = np.clip(m_norm, -0.08, 0.08)
                means[3] = np.clip(m_gain, 0.18, 0.80)

        ll_all = (
            np.log(weights)[np.newaxis, :]
            - 0.5 * ((log2_seg[:, np.newaxis] - means[np.newaxis, :]) / sds[np.newaxis, :]) ** 2
            - np.log(sds)[np.newaxis, :]
        )
        probs = np.exp(ll_all - special.logsumexp(ll_all, axis=1, keepdims=True))
        probs[~valid] = np.nan

        if cutoffs is not None:
            cuts = list(cutoffs)
            if len(cuts) != 4 or sorted(cuts) != cuts:
                raise ValueError(
                    "`cutoffs` must be 4 sorted log2 thresholds: `(dloss, loss, gain, amp)`."
                )
            calls = np.zeros(len(seg_lin), dtype=np.int64)
            calls[log2_seg <= cuts[0]] = -2
            calls[(log2_seg > cuts[0]) & (log2_seg <= cuts[1])] = -1
            calls[(log2_seg >= cuts[2]) & (log2_seg < cuts[3])] = 1
            calls[log2_seg >= cuts[3]] = 2
        else:
            best_state_idx = np.argmax(ll_all, axis=1)
            calls = state_values[best_state_idx]

        calls_out = calls.astype(np.float64)
        calls_out[~valid] = np.nan
        call_cols[sname] = calls_out
        prob_tables[sname] = pd.DataFrame(probs, columns=prob_names)

    return cur.clone(
        calls=pd.DataFrame(call_cols),
        probabilities=prob_tables,
        settings=cfg.settings,
    )
