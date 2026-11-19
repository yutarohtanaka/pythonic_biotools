"""Binned motif enrichment with iterative GC/CpG/dinucleotide background weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._validation import SettingsMode, validate_settings
from .composition import DINUCLEOTIDES, calc_sequence_composition
from .motifs import MotifScanResult, scan_sequences_pwm

_R_DEFAULT_GC_BREAKS = np.array(
    [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], dtype=np.float64
)


def _assign_1d_bins(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values in ``vals`` to integer bin indices ``0 .. len(edges) - 2``."""
    n_bins = max(1, edges.size - 1)
    if n_bins == 1:
        return np.zeros(vals.size, dtype=np.int64)
    idx = np.searchsorted(edges[1:-1], vals, side="right")
    return np.clip(idx, 0, n_bins - 1)


def _compute_default_joint_weights(
    comp_fg: pd.DataFrame, comp_bg: pd.DataFrame, max_iter: int = 20
) -> np.ndarray:
    """10-bin joint ``(GC, CpG_OE)`` iterative proportional fitting weights."""
    n_fg = len(comp_fg)
    n_bg = len(comp_bg)
    if n_fg == 0 or n_bg == 0:
        return np.ones(n_bg, dtype=np.float64)

    gc_fg = comp_fg["gc_frac"].to_numpy(dtype=np.float64)
    gc_bg = comp_bg["gc_frac"].to_numpy(dtype=np.float64)
    cpg_fg = comp_fg["cpg_oe"].to_numpy(dtype=np.float64)
    cpg_bg = comp_bg["cpg_oe"].to_numpy(dtype=np.float64)

    gc_all = np.concatenate([gc_fg, gc_bg])
    cpg_all = np.concatenate([cpg_fg, cpg_bg])

    gc_min, gc_max = float(np.min(gc_all)), float(np.max(gc_all))
    gc_edges = (
        np.linspace(gc_min, gc_max, 11)
        if gc_max > gc_min
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    cpg_min, cpg_max = float(np.min(cpg_all)), float(np.max(cpg_all))
    cpg_edges = (
        np.linspace(cpg_min, cpg_max, 11)
        if cpg_max > cpg_min
        else np.array([0.0, max(1.0, cpg_max)], dtype=np.float64)
    )

    n_gc = max(1, gc_edges.size - 1)
    n_cpg = max(1, cpg_edges.size - 1)

    fg_gc_bin = _assign_1d_bins(gc_fg, gc_edges)
    bg_gc_bin = _assign_1d_bins(gc_bg, gc_edges)
    fg_cpg_bin = _assign_1d_bins(cpg_fg, cpg_edges)
    bg_cpg_bin = _assign_1d_bins(cpg_bg, cpg_edges)

    fg_gc_target = np.bincount(fg_gc_bin, minlength=n_gc).astype(np.float64) / n_fg
    fg_cpg_target = np.bincount(fg_cpg_bin, minlength=n_cpg).astype(np.float64) / n_fg

    fg_joint = fg_gc_bin * n_cpg + fg_cpg_bin
    bg_joint = bg_gc_bin * n_cpg + bg_cpg_bin
    fg_joint_target = (
        np.bincount(fg_joint, minlength=n_gc * n_cpg).astype(np.float64) / n_fg
    )

    w = np.ones(n_bg, dtype=np.float64)

    # Check if marginals overlap; if disjoint in tiny synthetic data, seed with smooth Gaussian kernel
    overlap_gc = np.isin(bg_gc_bin, np.flatnonzero(fg_gc_target > 0))
    overlap_cpg = np.isin(bg_cpg_bin, np.flatnonzero(fg_cpg_target > 0))
    if not np.any(overlap_gc & overlap_cpg):
        gc_sd = max(float(np.std(gc_all)), 0.05)
        cpg_sd = max(float(np.std(cpg_all)), 0.10)
        log_k = -0.5 * ((gc_bg - float(np.mean(gc_fg))) / gc_sd) ** 2 - 0.5 * (
            (cpg_bg - float(np.mean(cpg_fg))) / cpg_sd
        ) ** 2
        log_k -= float(np.max(log_k))
        return np.exp(log_k)

    # Initialize joint cells where both fg and bg are present, else marginal product
    bg_joint_counts = np.bincount(bg_joint, minlength=n_gc * n_cpg)
    for cell_id in range(n_gc * n_cpg):
        if bg_joint_counts[cell_id] > 0 and fg_joint_target[cell_id] > 0:
            w[bg_joint == cell_id] = fg_joint_target[cell_id] / bg_joint_counts[cell_id]
        elif bg_joint_counts[cell_id] > 0 and fg_joint_target[cell_id] == 0:
            # Keep small initial weight if marginals are positive so raking can distribute mass
            g_b = cell_id // n_cpg
            c_b = cell_id % n_cpg
            if fg_gc_target[g_b] > 0 and fg_cpg_target[c_b] > 0:
                w[bg_joint == cell_id] = 1e-3 / n_bg
            else:
                w[bg_joint == cell_id] = 0.0

    if np.sum(w) <= 0.0:
        w = np.ones(n_bg, dtype=np.float64)

    # Iterative proportional fitting (raking) on 10-bin GC and 10-bin CpG_OE
    for _ in range(max_iter):
        w_old = w.copy()
        # Rake GC marginal
        bg_gc_mass = np.bincount(bg_gc_bin, weights=w, minlength=n_gc)
        for k in range(n_gc):
            if fg_gc_target[k] == 0.0:
                w[bg_gc_bin == k] = 0.0
            elif bg_gc_mass[k] > 0.0:
                w[bg_gc_bin == k] *= fg_gc_target[k] / bg_gc_mass[k]

        # Rake CpG_OE marginal
        bg_cpg_mass = np.bincount(bg_cpg_bin, weights=w, minlength=n_cpg)
        for k in range(n_cpg):
            if fg_cpg_target[k] == 0.0:
                w[bg_cpg_bin == k] = 0.0
            elif bg_cpg_mass[k] > 0.0:
                w[bg_cpg_bin == k] *= fg_cpg_target[k] / bg_cpg_mass[k]

        s = float(np.sum(w))
        if s <= 0.0:
            w = w_old
            break
        w /= s
        if float(np.max(np.abs(w - w_old))) < 1e-6:
            break

    return w


def _compute_r_default_kmer_weights(
    comp_fg: pd.DataFrame, comp_bg: pd.DataFrame, max_iter: int = 20
) -> np.ndarray:
    """8-bin GC + 16-dinucleotide iterative k-mer composition rescaling (``monaLisa`` v1.12)."""
    n_fg = len(comp_fg)
    n_bg = len(comp_bg)
    if n_fg == 0 or n_bg == 0:
        return np.ones(n_bg, dtype=np.float64)

    gc_fg = comp_fg["gc_frac"].to_numpy(dtype=np.float64)
    gc_bg = comp_bg["gc_frac"].to_numpy(dtype=np.float64)
    di_fg = comp_fg[list(DINUCLEOTIDES)].to_numpy(dtype=np.float64)
    di_bg = comp_bg[list(DINUCLEOTIDES)].to_numpy(dtype=np.float64)

    fg_gc_bin = _assign_1d_bins(gc_fg, _R_DEFAULT_GC_BREAKS)
    bg_gc_bin = _assign_1d_bins(gc_bg, _R_DEFAULT_GC_BREAKS)
    n_gc = _R_DEFAULT_GC_BREAKS.size - 1  # 8 bins

    fg_gc_counts = np.bincount(fg_gc_bin, minlength=n_gc).astype(np.float64)
    bg_gc_counts = np.bincount(bg_gc_bin, minlength=n_gc).astype(np.float64)

    shared_bins = (fg_gc_counts > 0) & (bg_gc_counts > 0)
    if not np.any(shared_bins):
        # Fallback if synthetic foreground & background lie in completely disjoint GC deciles
        gc_sd = max(float(np.std(np.concatenate([gc_fg, gc_bg]))), 0.05)
        log_k = -0.5 * ((gc_bg - float(np.mean(gc_fg))) / gc_sd) ** 2
        log_k -= float(np.max(log_k))
        w = np.exp(log_k)
        w /= np.sum(w)
        return w

    # Remove zero-weight background bins (fg_gc_counts == 0) & normalize shared GC target
    fg_target = np.where(shared_bins, fg_gc_counts, 0.0)
    fg_target /= np.sum(fg_target)

    w = np.zeros(n_bg, dtype=np.float64)
    for k in range(n_gc):
        if shared_bins[k]:
            w[bg_gc_bin == k] = fg_target[k] / bg_gc_counts[k]

    p_di_fg = np.mean(di_fg, axis=0)
    p_di_fg_sum = float(np.sum(p_di_fg))
    if p_di_fg_sum <= 0.0:
        return w
    p_di_fg = p_di_fg / p_di_fg_sum

    active = w > 0.0
    for _ in range(max_iter):
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            break
        p_di_bg = (w @ di_bg) / w_sum
        p_di_bg_sum = float(np.sum(p_di_bg))
        if p_di_bg_sum <= 0.0:
            break
        p_di_bg = p_di_bg / p_di_bg_sum

        if float(np.max(np.abs(p_di_fg - p_di_bg))) < 1e-4:
            break

        # Sequence-level geometric dinucleotide ratio update on active background sequences
        log_ratio = np.clip(
            np.log((p_di_fg + 1e-4) / (p_di_bg + 1e-4)), -2.0, 2.0
        )
        seq_factors = np.exp(di_bg[active] @ log_ratio)
        w[active] *= seq_factors

        # Re-normalize within each populated 8-bin GC interval
        bg_gc_mass = np.bincount(bg_gc_bin, weights=w, minlength=n_gc)
        for k in range(n_gc):
            if shared_bins[k] and bg_gc_mass[k] > 0.0:
                w[bg_gc_bin == k] *= fg_target[k] / bg_gc_mass[k]
            else:
                w[bg_gc_bin == k] = 0.0

    return w


def _bh_fdr_from_log_p(log_p_mat: np.ndarray) -> np.ndarray:
    """Log-space Benjamini-Hochberg returning ``-log10(padj)`` without float underflow."""
    flat = log_p_mat.ravel()
    out_flat = np.full(flat.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flat)
    m_tot = int(np.sum(valid))
    if m_tot == 0:
        return out_flat.reshape(log_p_mat.shape)

    lp = np.minimum(flat[valid], 0.0)
    order = np.argsort(lp, kind="mergesort")  # smallest p (most negative log_p) first
    ranks = np.arange(1, m_tot + 1, dtype=np.float64)
    adj_sorted = lp[order] + np.log(m_tot / ranks)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 0.0)

    inv_order = np.empty(m_tot, dtype=np.int64)
    inv_order[order] = np.arange(m_tot)
    log_padj = adj_sorted[inv_order]
    out_flat[valid] = np.maximum(0.0, -log_padj / np.log(10.0))
    return out_flat.reshape(log_p_mat.shape)


@dataclass
class BinnedMotifEnrichmentResult:
    """Result container for ``calc_binned_motif_enrichment``."""

    log2enr: pd.DataFrame
    pearson_resid: pd.DataFrame
    neg_log10_p: pd.DataFrame
    neg_log10_padj: pd.DataFrame
    sum_fg: pd.DataFrame
    sum_bg_weighted: pd.DataFrame
    bin_weights: dict[str, np.ndarray]
    bins: pd.Categorical
    pseudocount_log2enr: float
    background: str
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame across all ``(motif, bin)`` pairs."""
        motifs = list(self.log2enr.index)
        bin_cols = list(self.log2enr.columns)
        records: list[dict[str, Any]] = []
        for m in motifs:
            for b in bin_cols:
                records.append(
                    {
                        "motif": m,
                        "bin": b,
                        "log2enr": float(self.log2enr.loc[m, b]),
                        "pearson_resid": float(self.pearson_resid.loc[m, b]),
                        "neg_log10_p": float(self.neg_log10_p.loc[m, b]),
                        "neg_log10_padj": float(self.neg_log10_padj.loc[m, b]),
                        "sum_fg": float(self.sum_fg.loc[m, b]),
                        "sum_bg_weighted": float(self.sum_bg_weighted.loc[m, b]),
                    }
                )
        return pd.DataFrame(records)

    def top_motifs(
        self,
        bin_name: str | None = None,
        n: int = 10,
        min_neg_log10_padj: float = 0.0,
        enriched_only: bool = True,
    ) -> pd.DataFrame:
        """Return top enriched motifs for ``bin_name`` (or across all bins if None)."""
        df = self.to_dataframe()
        if bin_name is not None:
            if bin_name not in self.log2enr.columns:
                raise KeyError(
                    f"bin_name={bin_name!r} not found in bins {list(self.log2enr.columns)}."
                )
            df = df[df["bin"] == bin_name]
        if enriched_only:
            df = df[df["log2enr"] > 0.0]
        if min_neg_log10_padj > 0.0:
            df = df[df["neg_log10_padj"] >= min_neg_log10_padj]
        df = df.sort_values(
            by=["neg_log10_padj", "log2enr"],
            ascending=[False, False],
            kind="mergesort",
        )
        return df.head(n).reset_index(drop=True)


def _is_sequence_collection(obj: Any) -> bool:
    """Return True if ``obj`` is a 1D collection of DNA strings."""
    if isinstance(obj, (MotifScanResult, pd.DataFrame)):
        return False
    if isinstance(obj, np.ndarray):
        return obj.ndim == 1 and obj.dtype.kind in ("U", "S", "O")
    if isinstance(obj, pd.Series):
        return not pd.api.types.is_numeric_dtype(obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return len(obj) > 0 and isinstance(obj[0], str)
    return False


def calc_binned_motif_enrichment(
    sequences_or_hits: Sequence[str] | pd.Series | pd.DataFrame | np.ndarray | MotifScanResult,
    bins: pd.Categorical | pd.Series | Sequence[Any] | np.ndarray,
    pwms: Mapping[str, np.ndarray | pd.DataFrame]
    | Sequence[np.ndarray | pd.DataFrame]
    | np.ndarray
    | pd.DataFrame
    | None = None,
    sequences: Sequence[str] | pd.Series | np.ndarray | None = None,
    background: str = "otherBins",
    pseudocount_log2enr: float | None = None,
    min_score: str | float = "80%",
    both_strands: bool = True,
    settings: SettingsMode = "default",
) -> BinnedMotifEnrichmentResult:
    """Compute sequence-composition-corrected motif enrichments across genomic bins.

    Implements ``monaLisa::calcBinnedMotifEnrR`` (Machlab et al. 2022):
    - ``settings="default"``: ``pseudocount_log2enr=1.0``, 10-bin joint
      ``(GC, CpG_OE)`` iterative raking composition matching + Kish effective
      background sample size.
    - ``settings="R_default"``: ``pseudocount_log2enr=8.0``
      (``log2((fg + 8) / (bg_weighted + 8))``), 8-bin GC + 16-dinucleotide
      iterative composition rescaling (``maxIter=20``, removing zero-weight
      background bins).
    """
    validate_settings(settings)

    valid_bgs = ("otherBins", "zeroBin", "allBins")
    if background not in valid_bgs:
        raise ValueError(
            f"Unknown background={background!r}. Expected one of {valid_bgs}."
        )

    alpha = (
        (8.0 if settings == "R_default" else 1.0)
        if pseudocount_log2enr is None
        else float(pseudocount_log2enr)
    )
    if alpha <= 0.0:
        raise ValueError(f"pseudocount_log2enr must be > 0, got {alpha}.")

    # Resolve hit count matrix + sequences
    if _is_sequence_collection(sequences_or_hits):
        if pwms is None:
            raise ValueError(
                "pwms must be provided when sequences_or_hits is a sequence collection."
            )
        seq_list: list[str] | None = [str(s) for s in sequences_or_hits]  # type: ignore[union-attr]
        scan_res = scan_sequences_pwm(
            seq_list,
            pwms=pwms,
            min_score=min_score,
            both_strands=both_strands,
            settings=settings,
        )
        hit_df = scan_res.counts
    elif isinstance(sequences_or_hits, MotifScanResult):
        hit_df = sequences_or_hits.counts
        seq_list = [str(s) for s in sequences] if sequences is not None else None
    elif isinstance(sequences_or_hits, pd.DataFrame):
        hit_df = sequences_or_hits
        seq_list = [str(s) for s in sequences] if sequences is not None else None
    else:
        arr_hits = np.asarray(sequences_or_hits, dtype=np.float64)
        if arr_hits.ndim != 2 or arr_hits.size == 0:
            raise ValueError(
                "sequences_or_hits must be a collection of DNA sequences or a 2D hit matrix."
            )
        m_names = [f"motif_{j}" for j in range(arr_hits.shape[1])]
        hit_df = pd.DataFrame(arr_hits, columns=m_names)
        seq_list = [str(s) for s in sequences] if sequences is not None else None

    hit_mat = hit_df.to_numpy(dtype=np.float64, copy=True)
    motif_names = [str(c) for c in hit_df.columns]
    n_regions, n_motifs = hit_mat.shape
    if n_regions == 0 or n_motifs == 0:
        raise ValueError("Hit matrix must have >= 1 region and >= 1 motif.")
    if seq_list is not None and len(seq_list) != n_regions:
        raise ValueError(
            f"sequences length ({len(seq_list)}) must match regions ({n_regions})."
        )

    if isinstance(bins, pd.Categorical):
        cat_bins = bins
    elif isinstance(bins, pd.Series) and isinstance(bins.dtype, pd.CategoricalDtype):
        cat_bins = bins.cat
        cat_bins = pd.Categorical(bins, categories=bins.cat.categories, ordered=bins.cat.ordered)
    else:
        raw_b = list(bins)
        seen_cats: list[str] = []
        for b_item in raw_b:
            if pd.notna(b_item):
                s_b = str(b_item)
                if s_b not in seen_cats:
                    seen_cats.append(s_b)
        str_b = [str(x) if pd.notna(x) else np.nan for x in raw_b]
        cat_bins = pd.Categorical(str_b, categories=seen_cats, ordered=True)

    if len(cat_bins) != n_regions:
        raise ValueError(
            f"bins length ({len(cat_bins)}) must match number of regions ({n_regions})."
        )

    bin_names = [
        str(c) for c in cat_bins.categories if int(np.sum(cat_bins == c)) > 0
    ]
    if len(bin_names) == 0:
        raise ValueError("No non-empty bins found.")
    if background == "zeroBin" and "zero" not in bin_names:
        raise ValueError("background='zeroBin' requires a 'zero' bin in bins.")

    comp_df = (
        calc_sequence_composition(seq_list, settings=settings)
        if seq_list is not None
        else None
    )
    seq_lengths = (
        np.array([max(1, len(s)) for s in seq_list], dtype=np.float64)
        if seq_list is not None
        else np.ones(n_regions, dtype=np.float64)
    )

    n_bins_out = len(bin_names)
    log2enr_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    pearson_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    log_p_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    neg_log10_p_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    sum_fg_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    sum_bg_w_mat = np.zeros((n_motifs, n_bins_out), dtype=np.float64)
    bin_weights: dict[str, np.ndarray] = {}

    valid_region_mask = np.asarray(pd.notna(cat_bins), dtype=bool)
    bin_str_arr = np.asarray([str(x) if pd.notna(x) else "" for x in cat_bins], dtype=object)

    max_hits_per_seq = max(1.0, float(np.max(hit_mat)))
    hit_cap_scale = 1.0 if max_hits_per_seq <= 1.0 else max(2.0 * max_hits_per_seq, 10.0)

    for b_idx, b_name in enumerate(bin_names):
        fg_mask = valid_region_mask & (bin_str_arr == b_name)
        if background == "otherBins":
            bg_mask = valid_region_mask & (bin_str_arr != b_name)
        elif background == "zeroBin":
            bg_mask = valid_region_mask & (bin_str_arr == "zero")
        else:  # allBins
            bg_mask = valid_region_mask.copy()

        fg_idx = np.flatnonzero(fg_mask)
        bg_idx = np.flatnonzero(bg_mask)
        if fg_idx.size == 0 or bg_idx.size == 0:
            bin_weights[b_name] = np.zeros(0, dtype=np.float64)
            continue

        if background == "zeroBin" and b_name == "zero":
            # Zero bin compared to itself -> exact neutral enrichment
            fg_sums = np.sum(hit_mat[fg_idx], axis=0)
            sum_fg_mat[:, b_idx] = fg_sums
            sum_bg_w_mat[:, b_idx] = fg_sums
            bin_weights[b_name] = np.ones(bg_idx.size, dtype=np.float64)
            continue

        if comp_df is not None:
            comp_fg = comp_df.iloc[fg_idx]
            comp_bg = comp_df.iloc[bg_idx]
            if settings == "R_default":
                w_raw = _compute_r_default_kmer_weights(comp_fg, comp_bg, max_iter=20)
            else:
                w_raw = _compute_default_joint_weights(comp_fg, comp_bg, max_iter=20)
        else:
            w_raw = np.ones(bg_idx.size, dtype=np.float64)

        w_sum = float(np.sum(w_raw))
        if w_sum <= 0.0:
            w_raw = np.ones(bg_idx.size, dtype=np.float64)
            w_sum = float(bg_idx.size)

        # Foreground-exposure-matched weights so sum_bg_weighted is on exact sum_fg scale
        fg_len_tot = float(np.sum(seq_lengths[fg_idx]))
        bg_len_weighted = float(np.sum(w_raw * seq_lengths[bg_idx]))
        w_fg_scale = w_raw * (fg_len_tot / max(bg_len_weighted, 1e-12))
        bin_weights[b_name] = w_fg_scale

        fg_sums = np.sum(hit_mat[fg_idx], axis=0)
        bg_sums_fg_scale = w_fg_scale @ hit_mat[bg_idx]

        sum_fg_mat[:, b_idx] = fg_sums
        sum_bg_w_mat[:, b_idx] = bg_sums_fg_scale
        log2enr_mat[:, b_idx] = np.log2((fg_sums + alpha) / (bg_sums_fg_scale + alpha))

        # Effective background sample size scaling for 2x2 Pearson test
        if settings == "R_default":
            n_bg_eff = float(np.sum(w_raw > 0.0))
        else:
            w_sq_sum = float(np.sum(w_raw**2))
            n_bg_eff = (w_sum**2) / max(w_sq_sum, 1e-30)

        w_stat = w_raw * (n_bg_eff / w_sum)
        k_fg = fg_sums
        k_bg = w_stat @ hit_mat[bg_idx]

        if seq_list is not None:
            t_fg_base = float(np.sum(seq_lengths[fg_idx]))
            t_bg_base = float(np.sum(w_stat * seq_lengths[bg_idx]))
        else:
            t_fg_base = float(fg_idx.size) * hit_cap_scale
            t_bg_base = n_bg_eff * hit_cap_scale

        for m_idx in range(n_motifs):
            kf = float(k_fg[m_idx])
            kb = float(k_bg[m_idx])
            ktot = kf + kb
            if ktot <= 0.0:
                continue

            tf = max(t_fg_base, 2.0 * kf + 1.0)
            tb = max(t_bg_base, 2.0 * kb + 1.0)
            ttot = tf + tb
            ef = ktot * (tf / ttot)
            var_f = ef * (1.0 - tf / ttot) * (1.0 - ktot / ttot)
            if var_f <= 1e-30:
                continue
            z = (kf - ef) / np.sqrt(var_f)
            pearson_mat[m_idx, b_idx] = z
            lp = min(0.0, float(stats.norm.logsf(abs(z)) + np.log(2.0)))
            log_p_mat[m_idx, b_idx] = lp
            neg_log10_p_mat[m_idx, b_idx] = max(0.0, -lp / np.log(10.0))

    neg_log10_padj_mat = _bh_fdr_from_log_p(log_p_mat)

    m_index = pd.Index(motif_names, name="motif")
    b_cols = pd.Index(bin_names, name="bin")

    return BinnedMotifEnrichmentResult(
        log2enr=pd.DataFrame(log2enr_mat, index=m_index, columns=b_cols),
        pearson_resid=pd.DataFrame(pearson_mat, index=m_index, columns=b_cols),
        neg_log10_p=pd.DataFrame(neg_log10_p_mat, index=m_index, columns=b_cols),
        neg_log10_padj=pd.DataFrame(neg_log10_padj_mat, index=m_index, columns=b_cols),
        sum_fg=pd.DataFrame(sum_fg_mat, index=m_index, columns=b_cols),
        sum_bg_weighted=pd.DataFrame(sum_bg_w_mat, index=m_index, columns=b_cols),
        bin_weights=bin_weights,
        bins=cat_bins,
        pseudocount_log2enr=alpha,
        background=background,
        settings=settings,
    )
