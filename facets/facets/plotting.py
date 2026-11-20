"""
Classic 3-panel stacked genome-wide visualization for `facets`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .reference import canonical_chrom, chrom_sort_key

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .em import FacetsResult


def _get_pyplot() -> tuple[Any, Any]:
    """Lazily import `matplotlib` with a writable sandbox-safe `MPLCONFIGDIR`."""
    import os
    import tempfile

    mpl_default = os.path.expanduser("~/.matplotlib")
    if "MPLCONFIGDIR" not in os.environ and not os.access(mpl_default, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install 'matplotlib>=3.7'`."
        ) from exc
    return matplotlib, plt


def plot_facets(
    result: FacetsResult,
    figsize: tuple[float, float] = (11.0, 7.0),
    *,
    axes: Sequence[Axes] | None = None,
    title: str | None = None,
) -> Figure:
    """
    Plot the classic 3-panel stacked FACETS genome-wide allele-specific profile:
      1. Copy-number log-ratio (`cnlr`) + segment medians + `dipLogR` baseline
      2. Heterozygous allelic log-odds-ratio (`|valor|`) + segment imbalance
      3. Integer total (`tcn`) & minor (`lcn`) copy number + clonal cellular fraction (`cf_em`) shading

    Parameters
    ----------
    result : FacetsResult
        Fitted FACETS output from `run_facets` or `em_cncf`.
    figsize : tuple[float, float], default=(11.0, 7.0)
        Figure dimensions in inches when `axes is None`.
    axes : Sequence[matplotlib.axes.Axes] | None, default=None
        Optional sequence of 3 pre-allocated `Axes`.
    title : str | None, default=None
        Optional title override. Defaults to banner with `Purity`, `Ploidy`, `dipLogR`, and `WGD`.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered 3-panel FACETS figure.
    """
    matplotlib, plt = _get_pyplot()

    if axes is None:
        fig, ax_arr = plt.subplots(
            3,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0, 1.15]},
            constrained_layout=True,
        )
        ax_cnlr, ax_valor, ax_cn = ax_arr[0], ax_arr[1], ax_arr[2]
    else:
        if len(axes) != 3:
            raise ValueError(f"`axes` must contain exactly 3 Axes, got {len(axes)}.")
        ax_cnlr, ax_valor, ax_cn = axes[0], axes[1], axes[2]
        fig = ax_cnlr.figure  # type: ignore[assignment]

    purity_str = "NA (diploid null)" if result.purity is None else f"{result.purity:.2f}"
    wgd_str = "  |  WGD" if result.is_wgd else ""
    banner = (
        title
        if title is not None
        else (
            f"FACETS Allele-Specific Profile  |  "
            f"Purity = {purity_str}  |  Ploidy = {result.ploidy:.2f}  |  "
            f"dipLogR = {result.dip_logr:.2f}{wgd_str}"
        )
    )

    segs = result.segments
    snps = result.snps

    # Determine ordered chromosomes and cumulative Mb coordinates
    chrom_candidates: list[str] = []
    if not segs.empty and "chrom" in segs.columns:
        chrom_candidates.extend(segs["chrom"].astype(str).tolist())
    if snps is not None and not snps.empty and "chrom" in snps.columns:
        chrom_candidates.extend(snps["chrom"].astype(str).tolist())

    unique_chroms = sorted(
        list(dict.fromkeys(canonical_chrom(c) for c in chrom_candidates)),
        key=chrom_sort_key,
    )

    if not unique_chroms:
        ax_cnlr.set_title(banner, fontsize=10.5, fontweight="semibold", pad=8)
        ax_cnlr.set_ylabel("Copy-Number logR")
        ax_valor.set_ylabel("Allelic |logOR|")
        ax_cn.set_ylabel("Copy Number")
        ax_cn.set_xlabel("Chromosome")
        for a in (ax_cnlr, ax_valor, ax_cn):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        return fig

    seg_canon = (
        segs["chrom"].map(canonical_chrom) if not segs.empty else None
    )
    snp_canon = (
        snps["chrom"].map(canonical_chrom)
        if (snps is not None and not snps.empty)
        else None
    )

    chrom_offsets: dict[str, float] = {}
    chrom_centers: list[float] = []
    chrom_labels: list[str] = []
    chrom_boundaries: list[float] = [0.0]

    cum_mb = 0.0
    for idx_c, ch in enumerate(unique_chroms):
        max_pos = 0.0
        if seg_canon is not None:
            s_sub = segs[seg_canon == ch]
            if not s_sub.empty:
                max_pos = max(max_pos, float(s_sub["end"].max()))
        if snp_canon is not None and snps is not None:
            p_sub = snps[snp_canon == ch]
            if not p_sub.empty:
                max_pos = max(max_pos, float(p_sub["pos"].max()))
        span_mb = max(max_pos * 1e-6, 0.5)
        chrom_offsets[ch] = cum_mb
        chrom_centers.append(cum_mb + 0.5 * span_mb)
        lbl = ch[3:] if ch.lower().startswith("chr") else ch
        chrom_labels.append(lbl)
        if idx_c % 2 == 1:
            for a in (ax_cnlr, ax_valor, ax_cn):
                a.axvspan(cum_mb, cum_mb + span_mb, color="#64748b", alpha=0.05, zorder=0)
        cum_mb += span_mb
        chrom_boundaries.append(cum_mb)

    for b_pos in chrom_boundaries[1:-1]:
        for a in (ax_cnlr, ax_valor, ax_cn):
            a.axvline(b_pos, color="#cbd5e1", linestyle="--", linewidth=0.7, zorder=1)

    # Panel 1 & 2: SNP-level cnlr and heterozygous |valor|
    if snps is not None and not snps.empty and snp_canon is not None:
        snp_off = snp_canon.map(chrom_offsets).to_numpy(dtype=np.float64)
        snp_x = snp_off + snps["pos"].to_numpy(dtype=np.float64) * 1e-6
        cnlr_vals = snps["cnlr"].to_numpy(dtype=np.float64)
        chrom_rank_idx = snp_canon.map(
            {ch: idx % 2 for idx, ch in enumerate(unique_chroms)}
        ).to_numpy(dtype=np.int64)

        raster_snps = len(snps) > 4000
        pt_s = 5.0 if len(snps) <= 3000 else 2.5

        for parity, col_pt in ((0, "#475569"), (1, "#64748b")):
            m_p = (chrom_rank_idx == parity) & np.isfinite(cnlr_vals)
            if np.any(m_p):
                ax_cnlr.scatter(
                    snp_x[m_p],
                    cnlr_vals[m_p],
                    s=pt_s,
                    c=col_pt,
                    alpha=0.45,
                    edgecolors="none",
                    zorder=2,
                    rasterized=raster_snps,
                )

        if "het" in snps.columns and "valor" in snps.columns:
            het_mask = snps["het"].to_numpy(dtype=bool)
            valor_abs = np.abs(snps["valor"].to_numpy(dtype=np.float64))
            for parity, col_pt in ((0, "#334155"), (1, "#64748b")):
                m_h = het_mask & (chrom_rank_idx == parity) & np.isfinite(valor_abs)
                if np.any(m_h):
                    ax_valor.scatter(
                        snp_x[m_h],
                        valor_abs[m_h],
                        s=pt_s + 1.0,
                        c=col_pt,
                        alpha=0.50,
                        edgecolors="none",
                        zorder=2,
                        rasterized=raster_snps,
                    )

    ax_cnlr.axhline(
        result.dip_logr,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
        label=f"dipLogR ({result.dip_logr:.2f})",
    )
    ax_cnlr.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.8, zorder=1)
    ax_valor.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.8, zorder=1)

    # Segment overlays across all 3 panels
    cmap_cf = matplotlib.colormaps["YlGnBu"]
    max_tcn_plot = 4.0
    if not segs.empty and seg_canon is not None:
        seg_med_labeled = False
        mafr_labeled = False
        tcn_labeled = False
        lcn_labeled = False
        cf_labeled = False

        for _, row in segs.iterrows():
            ch_c = canonical_chrom(row["chrom"])
            if ch_c not in chrom_offsets:
                continue
            off = chrom_offsets[ch_c]
            x0 = off + float(row["start"]) * 1e-6
            x1 = max(x0 + 0.05, off + float(row["end"]) * 1e-6)

            # Panel 1: cnlr_median
            cnlr_med = float(row["cnlr_median"])
            ax_cnlr.hlines(
                cnlr_med,
                x0,
                x1,
                colors="#dc2626",
                linewidth=2.2,
                zorder=4,
                label="Segment median" if not seg_med_labeled else None,
            )
            seg_med_labeled = True

            # Panel 2: sqrt(max(mafR, 0)) allelic log-odds shift
            mafr_val = float(row["mafR"]) if "mafR" in row and np.isfinite(row["mafR"]) else 0.0
            valor_seg = float(np.sqrt(max(mafr_val, 0.0)))
            ax_valor.hlines(
                valor_seg,
                x0,
                x1,
                colors="#dc2626",
                linewidth=2.0,
                zorder=4,
                label=r"Segment $\sqrt{\mathrm{mafR}}$" if not mafr_labeled else None,
            )
            mafr_labeled = True

            # Panel 3: Integer total (tcn), minor (lcn), and cf_em shading
            tcn_v = float(row["tcn"])
            max_tcn_plot = max(max_tcn_plot, tcn_v)
            cf_v = float(np.clip(row.get("cf_em", 1.0), 0.0, 1.0))

            ax_cn.fill_between(
                [x0, x1],
                0.0,
                tcn_v,
                color=cmap_cf(0.25 + 0.65 * cf_v),
                alpha=0.18 + 0.32 * cf_v,
                step="post",
                zorder=2,
                label="Cellular fraction (cf)" if not cf_labeled else None,
            )
            cf_labeled = True

            ax_cn.hlines(
                tcn_v,
                x0,
                x1,
                colors="#0f172a",
                linewidth=2.6,
                zorder=5,
                label="Total CN (tcn)" if not tcn_labeled else None,
            )
            tcn_labeled = True

            lcn_v = float(row["lcn"]) if np.isfinite(row["lcn"]) else np.nan
            if np.isfinite(lcn_v):
                # Slight offset when lcn == tcn == 0 so both remain visible
                y_lcn = lcn_v - 0.05 if abs(lcn_v - tcn_v) < 1e-6 else lcn_v
                ax_cn.hlines(
                    y_lcn,
                    x0,
                    x1,
                    colors="#2563eb",
                    linewidth=2.1,
                    zorder=4,
                    label="Minor CN (lcn)" if not lcn_labeled else None,
                )
                lcn_labeled = True

    ax_cn.axhline(2.0, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)
    ax_cn.set_ylim(-0.35, min(max_tcn_plot + 0.8, 12.5))
    int_ticks = list(range(0, int(min(max_tcn_plot + 1, 12)) + 1))
    if len(int_ticks) > 7:
        int_ticks = int_ticks[::2]
    ax_cn.set_yticks(int_ticks)

    ax_cnlr.set_title(banner, fontsize=10.5, fontweight="semibold", pad=8)
    ax_cnlr.set_ylabel("Copy-Number\nlogR (cnlr)", fontsize=9.0)
    ax_valor.set_ylabel("Allelic\n|logOR| (|valor|)", fontsize=9.0)
    ax_cn.set_ylabel("Integer CN\n(tcn / lcn & cf)", fontsize=9.0)
    ax_cn.set_xlabel("Chromosome", fontsize=9.5)

    ax_cn.set_xlim(0.0, max(cum_mb, 1.0))
    ax_cn.set_xticks(chrom_centers)
    ax_cn.set_xticklabels(chrom_labels, fontsize=8.5)

    for a in (ax_cnlr, ax_valor, ax_cn):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        handles, labels = a.get_legend_handles_labels()
        if handles:
            a.legend(
                handles,
                labels,
                loc="upper right",
                ncol=min(len(handles), 3),
                frameon=True,
                framealpha=0.9,
                edgecolor="#e2e8f0",
                fontsize=7.8,
            )

    return fig
