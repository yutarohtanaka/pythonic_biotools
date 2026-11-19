"""Optional lazy-matplotlib plotting utilities for MR-Clust."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import MRClustResult


def two_stage_plot(
    result: MRClustResult,
    obs_prob_thresh: float | None = None,
    ax: Any = None,
    title: str = "MR-Clust Two-Stage Plot",
) -> Any:
    """Plot Wald ratio estimates vs. precision (1 / theta_se) colored by cluster.

    Imports `matplotlib.pyplot` lazily so headless environments without `matplotlib`
    can import and run `mrclust` without error.
    """
    import matplotlib.pyplot as plt

    df = result.to_dataframe(obs_prob_thresh=obs_prob_thresh)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    precision = 1.0 / np.maximum(df["theta_se"].to_numpy(dtype=np.float64), 1e-12)
    df = df.assign(precision=precision)

    for cluster_name, sub in df.groupby("cluster", sort=False):
        ax.scatter(
            sub["theta"],
            sub["precision"],
            label=str(cluster_name),
            alpha=0.85,
            edgecolors="none",
        )

    for c in result.clusters:
        if c.cluster_class == "Substantive":
            ax.axvline(c.mean, linestyle="--", alpha=0.6)
    ax.axvline(0.0, linestyle=":", color="gray", alpha=0.6)
    ax.set_xlabel(r"Wald Ratio Estimate ($\hat{\theta}_j$)")
    ax.set_ylabel(r"Precision ($1 / \hat{\sigma}_j$)")
    ax.set_title(title)
    ax.legend(title="Cluster", frameon=False)
    return ax
