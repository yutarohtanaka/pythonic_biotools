"""
Core algorithms for MR-Clust: clustered heterogeneity in Mendelian randomization.

Reference:
  Foley, C. N., Mason, A. M., Kirk, P. D. W., & Burgess, S. (2021).
  "MR-Clust: clustering of genetic variants in Mendelian randomization with
  similar causal estimates." Bioinformatics, 37(4), 531-541.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS_PI = 1e-12


@dataclass(frozen=True)
class ClusterInfo:
    """Summary metadata for a single mixture component in the selected MR-Clust model."""

    cluster_id: str
    cluster_index: int
    cluster_class: Literal["Substantive", "Null", "Junk"]
    mean: float
    se: float
    weight: float
    effective_n: float
    n_assigned: int
    n_assigned_thresh: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame summarizing this component."""
        return pd.DataFrame(
            [
                {
                    "cluster": self.cluster_id,
                    "cluster_index": self.cluster_index,
                    "cluster_class": self.cluster_class,
                    "cluster_mean": self.mean,
                    "cluster_se": self.se,
                    "weight": self.weight,
                    "effective_n": self.effective_n,
                    "n_assigned": self.n_assigned,
                    "n_assigned_thresh": self.n_assigned_thresh,
                }
            ]
        )


@dataclass(frozen=True)
class ClusterFit:
    """Fitted MR-Clust Gaussian-mixture model for a fixed number of substantive clusters K."""

    k: int
    n_components: int
    n_parameters: int
    log_likelihood: float
    bic: float
    aic: float
    cluster_means: np.ndarray
    cluster_ses: np.ndarray
    weights: np.ndarray
    component_names: tuple[str, ...]
    responsibilities: np.ndarray
    n_iter: int
    converged: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Return a per-component summary DataFrame for this fitted K-cluster model."""
        rows: list[dict[str, object]] = []
        n_obs = self.responsibilities.shape[0]
        best_idx = np.argmax(self.responsibilities, axis=1)
        for idx, name in enumerate(self.component_names):
            if idx < self.k:
                cls = "Substantive"
                mean_val = float(self.cluster_means[idx])
                se_val = float(self.cluster_ses[idx])
            elif name == "Null":
                cls = "Null"
                mean_val = 0.0
                se_val = float("nan")
            else:
                cls = "Junk"
                mean_val = float("nan")
                se_val = float("nan")
            eff_n = float(np.sum(self.responsibilities[:, idx]))
            n_map = int(np.sum(best_idx == idx))
            rows.append(
                {
                    "cluster": name,
                    "cluster_class": cls,
                    "cluster_mean": mean_val,
                    "cluster_se": se_val,
                    "weight": float(self.weights[idx]),
                    "effective_n": eff_n,
                    "n_assigned": n_map,
                    "n_obs": n_obs,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a concise human-readable summary of this K-cluster fit."""
        means_str = (
            ", ".join(f"{m:.4g}" for m in self.cluster_means)
            if self.k > 0
            else "none"
        )
        return (
            f"ClusterFit(K={self.k}, components={self.n_components}, "
            f"p={self.n_parameters}, logLik={self.log_likelihood:.4f}, "
            f"BIC={self.bic:.4f}, AIC={self.aic:.4f}, means=[{means_str}], "
            f"converged={self.converged}, n_iter={self.n_iter})"
        )


@dataclass(frozen=True)
class MRClustResult:
    """Complete result container returned by `mr_clust_em`."""

    snp_names: tuple[str, ...]
    n_snps: int
    theta: np.ndarray
    theta_se: np.ndarray
    bx: np.ndarray | None
    bxse: np.ndarray | None
    by: np.ndarray | None
    byse: np.ndarray | None
    variance_order: int | None
    best_k: int
    best_model: ClusterFit
    all_models: tuple[ClusterFit, ...]
    clusters: tuple[ClusterInfo, ...]
    best_cluster: tuple[str, ...]
    cluster_membership: tuple[str, ...]
    assignment_probs: np.ndarray
    responsibilities: np.ndarray
    obs_prob_thresh: float
    junk_dist: str
    junk_mean: float
    junk_scale: float
    junk_df: float

    def to_dataframe(
        self,
        obs_prob_thresh: float | None = None,
        include_probs: bool = False,
        sort_by_cluster: bool = False,
    ) -> pd.DataFrame:
        """Return tidy per-variant assignment results as a pandas DataFrame."""
        thresh = self.obs_prob_thresh if obs_prob_thresh is None else float(obs_prob_thresh)
        if not np.isfinite(thresh) or thresh < 0.0 or thresh > 1.0:
            raise ValueError(f"obs_prob_thresh must be in [0, 1], got {thresh!r}.")

        info_by_id = {c.cluster_id: c for c in self.clusters}
        passes = self.assignment_probs >= thresh
        cluster_col: list[str] = []
        class_col: list[str] = []
        mean_col: list[float] = []

        for bc, ok in zip(self.best_cluster, passes):
            if ok:
                info = info_by_id[bc]
                cluster_col.append(bc)
                class_col.append(info.cluster_class)
                mean_col.append(info.mean)
            else:
                cluster_col.append("Unassigned")
                class_col.append("Unassigned")
                mean_col.append(float("nan"))

        data: dict[str, object] = {
            "observation": list(self.snp_names),
            "theta": self.theta.copy(),
            "theta_se": self.theta_se.copy(),
            "cluster": cluster_col,
            "best_cluster": list(self.best_cluster),
            "cluster_class": class_col,
            "cluster_mean": np.asarray(mean_col, dtype=np.float64),
            "probability": self.assignment_probs.copy(),
            "passes_threshold": passes.copy(),
        }

        if self.bx is not None:
            data["bx"] = self.bx.copy()
        if self.bxse is not None:
            data["bxse"] = self.bxse.copy()
        if self.by is not None:
            data["by"] = self.by.copy()
        if self.byse is not None:
            data["byse"] = self.byse.copy()

        if include_probs:
            for col_idx, name in enumerate(self.best_model.component_names):
                data[f"prob_{name}"] = self.responsibilities[:, col_idx].copy()

        df = pd.DataFrame(data)
        if sort_by_cluster:
            order_map = {name: i for i, name in enumerate(self.best_model.component_names)}
            order_map["Unassigned"] = len(order_map)
            df = df.assign(
                _sort_rank=[order_map[c] for c in df["cluster"]],
                _neg_prob=-df["probability"],
            )
            df = (
                df.sort_values(
                    ["_sort_rank", "_neg_prob"],
                    kind="mergesort",
                )
                .drop(columns=["_sort_rank", "_neg_prob"])
                .reset_index(drop=True)
            )
        return df

    def snp_table(
        self,
        obs_prob_thresh: float | None = None,
        include_probs: bool = True,
        sort_by_cluster: bool = False,
    ) -> pd.DataFrame:
        """Alias for `to_dataframe(..., include_probs=True)`."""
        return self.to_dataframe(
            obs_prob_thresh=obs_prob_thresh,
            include_probs=include_probs,
            sort_by_cluster=sort_by_cluster,
        )

    def clusters_dataframe(self) -> pd.DataFrame:
        """Return per-cluster summary statistics for the BIC-selected model."""
        cols = [
            "cluster",
            "cluster_class",
            "cluster_mean",
            "cluster_se",
            "weight",
            "effective_n",
            "n_assigned",
            "n_assigned_thresh",
        ]
        if not self.clusters:
            return pd.DataFrame(columns=cols)
        rows = [
            {
                "cluster": c.cluster_id,
                "cluster_class": c.cluster_class,
                "cluster_mean": c.mean,
                "cluster_se": c.se,
                "weight": c.weight,
                "effective_n": c.effective_n,
                "n_assigned": c.n_assigned,
                "n_assigned_thresh": c.n_assigned_thresh,
            }
            for c in self.clusters
        ]
        return pd.DataFrame(rows, columns=cols)

    def cluster_summary(self) -> pd.DataFrame:
        """Alias for `clusters_dataframe()`."""
        return self.clusters_dataframe()

    def bic_dataframe(self) -> pd.DataFrame:
        """Return model selection metrics across all evaluated values of K."""
        rows = [
            {
                "k": m.k,
                "n_components": m.n_components,
                "n_parameters": m.n_parameters,
                "log_likelihood": m.log_likelihood,
                "bic": m.bic,
                "aic": m.aic,
                "is_best": m.k == self.best_k,
                "converged": m.converged,
                "n_iter": m.n_iter,
            }
            for m in self.all_models
        ]
        return pd.DataFrame(rows)

    def bic_table(self) -> pd.DataFrame:
        """Alias for `bic_dataframe()`."""
        return self.bic_dataframe()

    def summary(self) -> str:
        """Format a human-readable MR-Clust summary report."""
        vo_label = (
            f"order={self.variance_order}"
            if self.variance_order is not None
            else "precomputed"
        )
        lines = [
            "MR-Clust Summary",
            "=" * 64,
            (
                f"Variants: {self.n_snps} | Ratio SE: {vo_label} | "
                f"Junk: {self.junk_dist} | Prob threshold: {self.obs_prob_thresh:.2f}"
            ),
            (
                f"Selected K = {self.best_k} (components={self.best_model.n_components}, "
                f"p={self.best_model.n_parameters}, logLik={self.best_model.log_likelihood:.4f}, "
                f"BIC={self.best_model.bic:.4f}, AIC={self.best_model.aic:.4f})"
            ),
            "-" * 64,
            f"{'Cluster':<10} {'Class':<13} {'Mean':>10} {'SE':>10} {'Weight':>9} {'N_map':>6} {'N_thr':>6}",
        ]
        for c in self.clusters:
            mean_s = f"{c.mean:.4f}" if np.isfinite(c.mean) else "nan"
            se_s = f"{c.se:.4f}" if np.isfinite(c.se) else "-"
            lines.append(
                f"{c.cluster_id:<10} {c.cluster_class:<13} {mean_s:>10} {se_s:>10} "
                f"{c.weight:>9.4f} {c.n_assigned:>6d} {c.n_assigned_thresh:>6d}"
            )
        n_unassigned = sum(1 for m in self.cluster_membership if m == "Unassigned")
        lines.append("-" * 64)
        lines.append(
            f"Assigned (prob >= {self.obs_prob_thresh:.2f}): "
            f"{self.n_snps - n_unassigned}/{self.n_snps} | Unassigned: {n_unassigned}"
        )
        return "\n".join(lines)


def _parse_variance_order(variance_order: Literal[1, 2, "first", "second"]) -> int:
    """Normalize `variance_order` into integer 1 or 2."""
    if isinstance(variance_order, bool):
        raise ValueError(f"variance_order must be 1, 2, 'first', or 'second'; got {variance_order!r}.")
    if variance_order in (1, "1", "first", "First", "FIRST"):
        return 1
    if variance_order in (2, "2", "second", "Second", "SECOND"):
        return 2
    raise ValueError(
        f"variance_order must be 1, 2, 'first', or 'second'; got {variance_order!r}."
    )


def _to_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert sequence/array to contiguous finite 1D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:5].tolist()
        raise ValueError(f"{name} contains non-finite values at indices {bad}.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def compute_ratio_estimates(
    bx: Sequence[float] | np.ndarray,
    by: Sequence[float] | np.ndarray,
    byse: Sequence[float] | np.ndarray,
    bxse: Sequence[float] | np.ndarray | None = None,
    variance_order: Literal[1, 2, "first", "second"] = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wald ratio estimates theta = by / bx and 1st- or 2nd-order delta-method SEs.

    Parameters
    ----------
    bx : array-like
        Variant-exposure effect estimates (must be finite and non-zero).
    by : array-like
        Variant-outcome effect estimates (must be finite).
    byse : array-like
        Standard errors of `by` (must be finite and strictly positive).
    bxse : array-like, optional
        Standard errors of `bx` (required when `variance_order=2`, must be >= 0).
    variance_order : {1, 2, "first", "second"}, default=1
        Order of the delta-method variance approximation.

    Returns
    -------
    theta : np.ndarray
        Wald ratio estimates `by / bx`.
    theta_se : np.ndarray
        Delta-method standard errors of `theta`.
    """
    order = _parse_variance_order(variance_order)
    bx_arr = _to_1d_float(bx, "bx")
    by_arr = _to_1d_float(by, "by")
    byse_arr = _to_1d_float(byse, "byse")

    n = bx_arr.size
    if by_arr.size != n or byse_arr.size != n:
        raise ValueError(
            f"Length mismatch: len(bx)={n}, len(by)={by_arr.size}, len(byse)={byse_arr.size}."
        )

    zero_bx = np.where(bx_arr == 0.0)[0]
    if zero_bx.size > 0:
        raise ValueError(
            f"bx contains exact zero(s) at indices {zero_bx[:5].tolist()}; "
            "cannot compute Wald ratio estimates."
        )

    nonpos_byse = np.where(byse_arr <= 0.0)[0]
    if nonpos_byse.size > 0:
        raise ValueError(
            f"byse must be strictly positive; found <= 0 at indices {nonpos_byse[:5].tolist()}."
        )

    theta = by_arr / bx_arr
    abs_bx = np.abs(bx_arr)
    first_term = byse_arr / abs_bx

    if order == 1:
        theta_se = first_term
    else:
        if bxse is None:
            raise ValueError("bxse is required when variance_order=2.")
        bxse_arr = _to_1d_float(bxse, "bxse")
        if bxse_arr.size != n:
            raise ValueError(
                f"Length mismatch: len(bx)={n}, len(bxse)={bxse_arr.size}."
            )
        neg_bxse = np.where(bxse_arr < 0.0)[0]
        if neg_bxse.size > 0:
            raise ValueError(
                f"bxse must be non-negative; found < 0 at indices {neg_bxse[:5].tolist()}."
            )
        with np.errstate(over="ignore"):
            second_term = theta * (bxse_arr / abs_bx)
            theta_se = np.hypot(first_term, second_term)

    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(theta_se)):
        raise ValueError(
            "Computed Wald ratios or standard errors overflowed to non-finite values."
        )
    if np.any(theta_se <= 0.0):
        raise ValueError("Computed theta_se contains non-positive values.")

    return theta, theta_se


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column name in `df` (exact or case-insensitive)."""
    cols = list(df.columns)
    col_set = set(cols)
    for cand in candidates:
        if cand in col_set:
            return cand
    lower_map = {str(c).lower(): str(c) for c in cols}
    for cand in candidates:
        match = lower_map.get(cand.lower())
        if match is not None:
            return match
    return None


def _resolve_df_arg(
    arg: str | Sequence[float] | np.ndarray | None,
    data: pd.DataFrame | None,
    candidates: tuple[str, ...],
    arg_name: str,
) -> np.ndarray | None:
    """Resolve an input argument from direct array, column name string, or DataFrame aliases."""
    if isinstance(arg, str):
        if data is None:
            raise ValueError(f"Column name {arg!r} passed for {arg_name}, but `data` is None.")
        if arg not in data.columns:
            raise ValueError(f"Column {arg!r} for {arg_name} not found in `data`.")
        return _to_1d_float(data[arg].to_numpy(), arg_name)
    if arg is not None:
        return _to_1d_float(arg, arg_name)
    if data is not None:
        col = _pick_col(data, candidates)
        if col is not None:
            return _to_1d_float(data[col].to_numpy(), col)
    return None


def _normalize_inputs(
    theta: str | Sequence[float] | np.ndarray | None,
    theta_se: str | Sequence[float] | np.ndarray | None,
    bx: str | Sequence[float] | np.ndarray | None,
    bxse: str | Sequence[float] | np.ndarray | None,
    by: str | Sequence[float] | np.ndarray | None,
    byse: str | Sequence[float] | np.ndarray | None,
    snp_names: Sequence[str] | None,
    obs_names: Sequence[str] | None,
    data: pd.DataFrame | None,
    variance_order: Literal[1, 2, "first", "second"],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int | None,
    tuple[str, ...],
]:
    """Validate and normalize multi-modal MR-Clust inputs."""
    if data is not None and not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    if snp_names is not None and obs_names is not None:
        s_list = [str(x) for x in snp_names]
        o_list = [str(x) for x in obs_names]
        if s_list != o_list:
            raise ValueError("`snp_names` and `obs_names` were both provided and differ.")
        raw_names: Sequence[str] | None = s_list
    else:
        raw_names = snp_names if snp_names is not None else obs_names

    has_direct_theta = theta is not None or theta_se is not None
    has_direct_beta = any(v is not None for v in (bx, by, byse, bxse))

    if has_direct_theta and has_direct_beta:
        raise ValueError(
            "Provide either (theta, theta_se) or (bx, by, byse[, bxse]), not both."
        )

    theta_arr: np.ndarray | None = None
    theta_se_arr: np.ndarray | None = None
    bx_arr: np.ndarray | None = None
    bxse_arr: np.ndarray | None = None
    by_arr: np.ndarray | None = None
    byse_arr: np.ndarray | None = None
    used_order: int | None = None

    if has_direct_theta:
        theta_arr = _resolve_df_arg(theta, data, ("theta",), "theta")
        theta_se_arr = _resolve_df_arg(theta_se, data, ("theta_se",), "theta_se")
        if theta_arr is None or theta_se_arr is None:
            raise ValueError("Both `theta` and `theta_se` must be provided.")
        if theta_arr.size != theta_se_arr.size:
            raise ValueError(
                f"Length mismatch: len(theta)={theta_arr.size}, len(theta_se)={theta_se_arr.size}."
            )
        bad_se = np.where(theta_se_arr <= 0.0)[0]
        if bad_se.size > 0:
            raise ValueError(
                f"theta_se must be strictly positive; found <= 0 at indices {bad_se[:5].tolist()}."
            )
    elif has_direct_beta:
        bx_arr = _resolve_df_arg(
            bx, data, ("bx", "beta_exp", "beta.exposure", "xbeta", "b_exp", "beta_exposure"), "bx"
        )
        by_arr = _resolve_df_arg(
            by, data, ("by", "beta_out", "beta.outcome", "ybeta", "b_out", "beta_outcome"), "by"
        )
        byse_arr = _resolve_df_arg(
            byse, data, ("byse", "se_out", "se.outcome", "yse", "se_outcome", "sd_outcome"), "byse"
        )
        bxse_arr = _resolve_df_arg(
            bxse, data, ("bxse", "se_exp", "se.exposure", "xse", "se_exposure", "sd_exposure"), "bxse"
        )
        if bx_arr is None or by_arr is None or byse_arr is None:
            raise ValueError("`bx`, `by`, and `byse` are all required when supplying summary statistics.")
        used_order = _parse_variance_order(variance_order)
        theta_arr, theta_se_arr = compute_ratio_estimates(
            bx=bx_arr,
            by=by_arr,
            byse=byse_arr,
            bxse=bxse_arr,
            variance_order=used_order,
        )
    elif data is not None:
        theta_arr = _resolve_df_arg(
            None, data, ("theta", "ratio", "beta_ratio"), "theta"
        )
        theta_se_arr = _resolve_df_arg(
            None, data, ("theta_se", "ratio_se", "se_theta"), "theta_se"
        )
        if theta_arr is not None and theta_se_arr is not None:
            if theta_arr.size != theta_se_arr.size:
                raise ValueError(
                    f"Length mismatch: len(theta)={theta_arr.size}, len(theta_se)={theta_se_arr.size}."
                )
            bad_se = np.where(theta_se_arr <= 0.0)[0]
            if bad_se.size > 0:
                raise ValueError(
                    f"theta_se must be strictly positive; found <= 0 at indices {bad_se[:5].tolist()}."
                )
        else:
            bx_arr = _resolve_df_arg(
                None,
                data,
                ("bx", "beta_exp", "beta.exposure", "xbeta", "b_exp", "beta_exposure", "BETA_X"),
                "bx",
            )
            by_arr = _resolve_df_arg(
                None,
                data,
                ("by", "beta_out", "beta.outcome", "ybeta", "b_out", "beta_outcome", "BETA_Y"),
                "by",
            )
            byse_arr = _resolve_df_arg(
                None,
                data,
                ("byse", "se_out", "se.outcome", "yse", "se_outcome", "sd_outcome", "SE_Y"),
                "byse",
            )
            bxse_arr = _resolve_df_arg(
                None,
                data,
                ("bxse", "se_exp", "se.exposure", "xse", "se_exposure", "sd_exposure", "SE_X"),
                "bxse",
            )
            if bx_arr is None or by_arr is None or byse_arr is None:
                raise ValueError(
                    "Could not resolve `(theta, theta_se)` or `(bx, by, byse)` from `data`."
                )
            used_order = _parse_variance_order(variance_order)
            theta_arr, theta_se_arr = compute_ratio_estimates(
                bx=bx_arr,
                by=by_arr,
                byse=byse_arr,
                bxse=bxse_arr,
                variance_order=used_order,
            )
    else:
        raise ValueError(
            "Provide either `(theta, theta_se)`, `(bx, by, byse[, bxse])`, or a `data` DataFrame."
        )

    n_snps = int(theta_arr.size)
    if n_snps < 2:
        raise ValueError("At least 2 genetic instruments are required for MR-Clust.")

    if raw_names is not None:
        names_tuple = tuple(str(x) for x in raw_names)
        if len(names_tuple) != n_snps:
            raise ValueError(
                f"Length mismatch: len(snp_names)={len(names_tuple)}, expected {n_snps}."
            )
    elif data is not None:
        id_col = _pick_col(data, ("snp", "SNP", "rsid", "observation", "obs_names", "id"))
        if id_col is not None:
            names_tuple = tuple(str(x) for x in data[id_col].tolist())
        elif not isinstance(data.index, pd.RangeIndex):
            names_tuple = tuple(str(x) for x in data.index.tolist())
        else:
            names_tuple = tuple(f"snp_{j + 1}" for j in range(n_snps))
    else:
        names_tuple = tuple(f"snp_{j + 1}" for j in range(n_snps))

    return (
        theta_arr,
        theta_se_arr,
        bx_arr,
        bxse_arr,
        by_arr,
        byse_arr,
        used_order,
        names_tuple,
    )


def _robust_junk_defaults(
    theta: np.ndarray,
    theta_se: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Compute weak-IV-robust default Junk location, scale, and well-powered mask."""
    n_obs = theta.size
    min_se = float(np.min(theta_se))
    se_anchor = float(np.quantile(theta_se, 0.15)) if n_obs >= 5 else min_se
    rel_iv = (min_se / theta_se) ** 2
    mu0 = float(_weighted_quantiles(theta, rel_iv, np.array([0.5], dtype=np.float64))[0])
    z0 = np.abs(theta - mu0) / theta_se
    wp_mask = (theta_se <= 100.0 * se_anchor) | ((z0 > 5.0) & (theta_se <= 1000.0 * se_anchor))
    if not np.any(wp_mask):
        wp_mask = np.ones(n_obs, dtype=bool)

    th_wp = theta[wp_mask]
    se_wp = theta_se[wp_mask]

    med_wp = float(np.median(th_wp))
    med_se_wp = float(np.median(se_wp))

    mu_default = med_wp + float(np.mean(th_wp - med_wp))
    psi_default = max(
        float(np.max(th_wp) - np.min(th_wp)) + 2.0 * float(np.max(se_wp)),
        8.0 * med_se_wp,
        1e-300,
    )
    return mu_default, psi_default, wp_mask


def _compute_junk_params_and_logpdf(
    theta: np.ndarray,
    theta_se: np.ndarray,
    junk_dist: Literal["t", "normal", "uniform"],
    junk_df: float,
    junk_mean: float | None,
    junk_scale: float | None,
    wp_mask: np.ndarray | None = None,
    junk_defaults: tuple[float, float, np.ndarray] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Compute fixed Junk component location, scale, and per-observation log-density."""
    if junk_dist not in ("t", "normal", "uniform"):
        raise ValueError(f"junk_dist must be 't', 'normal', or 'uniform'; got {junk_dist!r}.")
    if not np.isfinite(junk_df) or junk_df <= 0.0:
        raise ValueError(f"junk_df must be finite and > 0, got {junk_df!r}.")

    if junk_defaults is None:
        mu_default, default_psi, _ = _robust_junk_defaults(theta, theta_se)
    else:
        mu_default, default_psi, _ = junk_defaults

    mu_junk = mu_default if junk_mean is None else float(junk_mean)
    if not np.isfinite(mu_junk):
        raise ValueError(f"junk_mean must be finite, got {mu_junk!r}.")

    if junk_scale is None:
        if junk_dist == "normal":
            psi_junk = max(default_psi * 0.5, 1e-300)
        elif junk_dist == "uniform":
            psi_junk = max(default_psi * 2.0, 1e-300)
        else:
            psi_junk = default_psi
    else:
        psi_junk = float(junk_scale)
        if not np.isfinite(psi_junk) or psi_junk <= 0.0:
            raise ValueError(f"junk_scale must be finite and > 0, got {junk_scale!r}.")

    if junk_dist == "t":
        nu = float(junk_df)
        eff_scale = np.hypot(psi_junk, theta_se)
        const = (
            float(gammaln(0.5 * (nu + 1.0)))
            - float(gammaln(0.5 * nu))
            - 0.5 * float(np.log(nu * np.pi))
            - np.log(eff_scale)
        )
        z = (theta - mu_junk) / eff_scale
        abs_z = np.abs(z)
        with np.errstate(over="ignore", divide="ignore"):
            log_quad = np.where(
                abs_z < 1e100,
                np.log1p((z * z) / nu),
                2.0 * np.log(np.maximum(abs_z, 1e-300)) - np.log(nu),
            )
        log_f = const - 0.5 * (nu + 1.0) * log_quad
    elif junk_dist == "normal":
        sigma_tot = np.hypot(psi_junk, theta_se)
        z = (theta - mu_junk) / sigma_tot
        with np.errstate(over="ignore"):
            log_f = -0.5 * _LOG_2PI - np.log(sigma_tot) - 0.5 * (z * z)
    else:
        half = 0.5 * psi_junk
        width = psi_junk / theta_se
        log_f = np.empty_like(theta, dtype=np.float64)
        small_w = width < 1e-6
        if np.any(small_w):
            zm = (theta[small_w] - mu_junk) / theta_se[small_w]
            with np.errstate(over="ignore"):
                log_f[small_w] = -0.5 * _LOG_2PI - np.log(theta_se[small_w]) - 0.5 * (zm * zm)
        rest = ~small_w
        if np.any(rest):
            ar = (theta[rest] - mu_junk - half) / theta_se[rest]
            br = (theta[rest] - mu_junk + half) / theta_se[rest]
            left = br <= 0.0
            right = ar >= 0.0
            mid = ~(left | right)
            ldiff = np.empty_like(ar)
            if np.any(left):
                la = log_ndtr(ar[left])
                lb = log_ndtr(br[left])
                with np.errstate(invalid="ignore"):
                    diff_l = np.where(np.isfinite(lb), np.minimum(la - lb, 0.0), -np.inf)
                ldiff[left] = np.where(np.isfinite(lb), lb + np.log1p(-np.exp(diff_l)), -np.inf)
            if np.any(right):
                lna = log_ndtr(-ar[right])
                lnb = log_ndtr(-br[right])
                with np.errstate(invalid="ignore"):
                    diff_r = np.where(np.isfinite(lna), np.minimum(lnb - lna, 0.0), -np.inf)
                ldiff[right] = np.where(np.isfinite(lna), lna + np.log1p(-np.exp(diff_r)), -np.inf)
            if np.any(mid):
                ldiff[mid] = np.log(np.maximum(ndtr(br[mid]) - ndtr(ar[mid]), 1e-300))
            log_f[rest] = -float(np.log(psi_junk)) + ldiff

    return mu_junk, psi_junk, np.ascontiguousarray(log_f, dtype=np.float64)


def _normal_logpdf_matrix(
    theta: np.ndarray,
    theta_se: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    """Evaluate Gaussian log-densities of shape (J, M) for fixed means and per-obs SEs."""
    # theta: (J, 1), theta_se: (J, 1), means: (1, M)
    z = (theta[:, None] - means[None, :]) / theta_se[:, None]
    with np.errstate(over="ignore"):
        return -0.5 * _LOG_2PI - np.log(theta_se)[:, None] - 0.5 * (z * z)


def _run_em(
    theta: np.ndarray,
    inv_se: np.ndarray,
    log_inv_se: np.ndarray,
    rel_inv_var: np.ndarray,
    min_se: float,
    init_means: np.ndarray,
    init_weights: np.ndarray,
    log_null: np.ndarray | None,
    log_junk: np.ndarray | None,
    dirichlet_alpha: float,
    max_iter: int,
    tol: float,
    param_tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Run log-space EM with monotone SQUAREM acceleration for fixed K.

    Returns
    -------
    means : np.ndarray (K,) sorted ascending
    cluster_ses : np.ndarray (K,)
    weights : np.ndarray (C_K,)
    resp : np.ndarray (J, C_K)
    log_lik : float
    n_iter : int
    converged : bool
    """
    n_obs = theta.size
    k = int(init_means.size)
    n_comp = int(init_weights.size)
    means = np.asarray(init_means, dtype=np.float64).copy()
    weights = np.asarray(init_weights, dtype=np.float64).copy()
    weights = np.maximum(weights, _EPS_PI)
    weights /= np.sum(weights)

    log_dens = np.empty((n_obs, n_comp), dtype=np.float64)
    col_tail = k
    if log_null is not None:
        log_dens[:, col_tail] = log_null
        col_tail += 1
    if log_junk is not None:
        log_dens[:, col_tail] = log_junk

    def _e_step(cur_means: np.ndarray, cur_weights: np.ndarray) -> tuple[float, np.ndarray]:
        if k > 0:
            z = (theta[:, None] - cur_means[None, :]) * inv_se[:, None]
            with np.errstate(over="ignore"):
                log_dens[:, :k] = -0.5 * _LOG_2PI + log_inv_se[:, None] - 0.5 * (z * z)
        log_w = log_dens + np.log(np.maximum(cur_weights, 1e-300))[None, :]
        max_lw = np.max(log_w, axis=1)
        all_neg_inf = ~np.isfinite(max_lw)
        if np.any(all_neg_inf):
            safe_max = np.where(all_neg_inf, 0.0, max_lw)
            exp_w = np.exp(log_w - safe_max[:, None])
            sum_exp = np.sum(exp_w, axis=1)
            safe_sum = np.where(all_neg_inf, 1.0, sum_exp)
            resp_out = exp_w / safe_sum[:, None]
            comp_centers = (
                np.concatenate([cur_means, np.array([0.0], dtype=np.float64)])
                if log_null is not None
                else cur_means
            )
            for row in np.where(all_neg_inf)[0]:
                if comp_centers.size > 0 and log_junk is None:
                    best_c = int(np.argmin(np.abs(theta[row] - comp_centers)))
                    resp_out[row, :] = 0.0
                    resp_out[row, best_c] = 1.0
                else:
                    resp_out[row, :] = 1.0 / n_comp
            with np.errstate(divide="ignore"):
                row_ll = np.where(all_neg_inf, -np.inf, max_lw + np.log(safe_sum))
            ll = float(np.sum(row_ll))
            return ll, resp_out
        exp_w = np.exp(log_w - max_lw[:, None])
        sum_exp = np.sum(exp_w, axis=1)
        ll = float(np.sum(max_lw + np.log(sum_exp)))
        resp_out = exp_w / sum_exp[:, None]
        return ll, resp_out

    def _m_step(resp_in: np.ndarray, cur_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eff_n = np.sum(resp_in, axis=0)
        if dirichlet_alpha > 0.0:
            raw_w = (eff_n + dirichlet_alpha) / (n_obs + n_comp * dirichlet_alpha)
        else:
            raw_w = eff_n / n_obs
        w_out = np.maximum(raw_w, _EPS_PI)
        w_out /= np.sum(w_out)

        m_out = cur_means.copy()
        if k > 0:
            prec_resp = resp_in[:, :k] * rel_inv_var[:, None]
            prec_sum = np.sum(prec_resp, axis=0)
            valid = (prec_sum > 0.0) & np.isfinite(prec_sum) & (eff_n[:k] > 1e-12)
            if np.any(valid):
                m_out[valid] = (
                    np.sum(prec_resp[:, valid] * theta[:, None], axis=0) / prec_sum[valid]
                )
        return m_out, w_out

    def _em_step(
        cur_means: np.ndarray, cur_weights: np.ndarray, cur_resp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        m_next, w_next = _m_step(cur_resp, cur_means)
        ll_next, resp_next = _e_step(m_next, w_next)
        return m_next, w_next, ll_next, resp_next

    def _is_converged(
        ll_new: float,
        ll_old: float,
        m_new: np.ndarray,
        m_old: np.ndarray,
        w_new: np.ndarray,
        w_old: np.ndarray,
    ) -> bool:
        if np.isfinite(ll_new) and np.isfinite(ll_old):
            if abs(ll_new - ll_old) > tol:
                return False
            if param_tol is None:
                return True
            dm = float(np.max(np.abs(m_new - m_old))) / max(min_se, 1e-300) if k > 0 else 0.0
            dw = float(np.max(np.abs(w_new - w_old)))
            return dm <= param_tol and dw <= param_tol
        if np.isfinite(ll_new) != np.isfinite(ll_old):
            return False
        eff_ptol = 1e-9 if param_tol is None else param_tol
        dm = float(np.max(np.abs(m_new - m_old))) / max(min_se, 1e-300) if k > 0 else 0.0
        dw = float(np.max(np.abs(w_new - w_old)))
        return dm <= eff_ptol and dw <= eff_ptol

    prev_ll, resp = _e_step(means, weights)
    if n_comp == 1 and k == 0:
        return means, np.empty(0, dtype=np.float64), weights, resp, prev_ll, 1, True

    converged = False
    n_iter = 0
    inv_scale = 1.0 / max(min_se, 1e-300)

    while n_iter < max_iter:
        # EM step 1
        m1, w1, ll1, resp1 = _em_step(means, weights, resp)
        n_iter += 1
        if _is_converged(ll1, prev_ll, m1, means, w1, weights):
            means, weights, prev_ll, resp = m1, w1, ll1, resp1
            converged = True
            break
        if n_iter >= max_iter:
            means, weights, prev_ll, resp = m1, w1, ll1, resp1
            break

        # EM step 2
        m2, w2, ll2, resp2 = _em_step(m1, w1, resp1)
        n_iter += 1
        if _is_converged(ll2, ll1, m2, m1, w2, w1):
            means, weights, prev_ll, resp = m2, w2, ll2, resp2
            converged = True
            break

        # Monotone SQUAREM extrapolation in (means, centered log-weights)
        lw0 = np.log(weights)
        lw1 = np.log(w1)
        lw2 = np.log(w2)
        r_lw = lw1 - lw0
        r_lw -= np.mean(r_lw)
        v_lw = (lw2 - lw1) - (lw1 - lw0)
        v_lw -= np.mean(v_lw)

        r_norm2 = float(np.sum(r_lw * r_lw))
        v_norm2 = float(np.sum(v_lw * v_lw))
        if k > 0:
            r_m = m1 - means
            v_m = (m2 - m1) - r_m
            max_m = max(float(np.max(np.abs(r_m))), float(np.max(np.abs(v_m))))
            if max_m > 1e100 * min_se:
                r_m_u = r_m / max_m
                v_m_u = v_m / max_m
                r_norm2 = float(np.sum(r_m_u * r_m_u))
                v_norm2 = float(np.sum(v_m_u * v_m_u))
            else:
                r_m_scaled = r_m * inv_scale
                v_m_scaled = v_m * inv_scale
                r_norm2 += float(np.sum(r_m_scaled * r_m_scaled))
                v_norm2 += float(np.sum(v_m_scaled * v_m_scaled))
        else:
            r_m = m1
            v_m = m2

        accepted_sq = False
        if v_norm2 > 1e-30 and r_norm2 > 1e-30 and np.isfinite(r_norm2) and np.isfinite(v_norm2):
            alpha = -float(np.sqrt(r_norm2 / v_norm2))
            alpha = max(min(alpha, -1.0), -25.0)
            if alpha < -1.05 and n_iter < max_iter:
                lw_ext = lw0 - 2.0 * alpha * r_lw + (alpha * alpha) * v_lw
                lw_ext -= np.max(lw_ext)
                w_ext = np.maximum(np.exp(lw_ext), _EPS_PI)
                w_ext /= np.sum(w_ext)
                m_ext = (means - 2.0 * alpha * r_m + (alpha * alpha) * v_m) if k > 0 else means
                ll_ext, resp_ext = _e_step(m_ext, w_ext)
                if np.isfinite(ll_ext):
                    m3, w3, ll3, resp3 = _em_step(m_ext, w_ext, resp_ext)
                    n_iter += 1
                    if np.isfinite(ll3) and ll3 > ll2:
                        means, weights, prev_ll, resp = m3, w3, ll3, resp3
                        accepted_sq = True
                        if _is_converged(ll3, ll2, m3, m2, w3, w2):
                            converged = True
                            break

        if not accepted_sq:
            means, weights, prev_ll, resp = m2, w2, ll2, resp2

    # Canonical ascending sort of substantive clusters
    if k > 0:
        order = np.argsort(means, kind="mergesort")
        means = means[order]
        perm = list(order) + list(range(k, n_comp))
        weights = weights[perm]
        resp = resp[:, perm]
        prec_sum = np.sum(resp[:, :k] * rel_inv_var[:, None], axis=0)
        with np.errstate(divide="ignore"):
            cluster_ses = np.where(
                prec_sum > 0.0,
                min_se / np.sqrt(np.maximum(prec_sum, 1e-300)),
                np.inf,
            )
    else:
        cluster_ses = np.empty(0, dtype=np.float64)

    return means, cluster_ses, weights, resp, prev_ll, n_iter, converged


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute weighted quantiles deterministically."""
    order = np.argsort(values, kind="mergesort")
    v_sorted = values[order]
    w_sorted = weights[order]
    w_sum = float(np.sum(w_sorted))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return np.quantile(values, probs)
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= w_sum
    return np.interp(probs, cum_w, v_sorted)


def _kmeans_1d(
    theta: np.ndarray,
    k: int,
    rng: np.random.Generator,
    sample_weights: np.ndarray | None = None,
    n_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast 1D k-means++ supporting unweighted or precision-weighted clustering."""
    n = theta.size
    if k == 1:
        if sample_weights is not None and np.sum(sample_weights) > 0.0:
            center = float(np.sum(sample_weights * theta) / np.sum(sample_weights))
        else:
            center = float(np.mean(theta))
        return np.array([center], dtype=np.float64), np.array([n], dtype=np.float64)

    theta_scale = float(np.max(np.abs(theta)))
    th_s = (theta / theta_scale) if theta_scale > 0.0 else theta

    w = np.ones(n, dtype=np.float64) if sample_weights is None else np.maximum(sample_weights, 1e-300)
    w_norm = w / np.sum(w)

    centers = np.empty(k, dtype=np.float64)
    first_idx = int(rng.choice(n, p=w_norm))
    centers[0] = th_s[first_idx]

    min_sq_dist = (th_s - centers[0]) ** 2
    for c in range(1, k):
        probs = w * min_sq_dist
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            idx = int(rng.integers(0, n))
        else:
            probs /= total
            idx = int(rng.choice(n, p=probs))
        centers[c] = th_s[idx]
        min_sq_dist = np.minimum(min_sq_dist, (th_s - centers[c]) ** 2)

    labels = np.zeros(n, dtype=np.int64)
    for step in range(n_steps):
        dists = (th_s[:, None] - centers[None, :]) ** 2
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and step > 0:
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                wc = w[mask]
                wsum = float(np.sum(wc))
                centers[c] = (
                    float(np.sum(wc * th_s[mask]) / wsum)
                    if wsum > 0
                    else float(np.mean(th_s[mask]))
                )
            else:
                centers[c] = float(th_s[int(rng.integers(0, n))])

    counts = np.array([float(np.sum(labels == c)) for c in range(k)], dtype=np.float64)
    order = np.argsort(centers, kind="mergesort")
    centers_out = centers[order] * (theta_scale if theta_scale > 0.0 else 1.0)
    return centers_out, counts[order]


def _make_weights(
    subst_counts: np.ndarray,
    pi_null: float,
    pi_junk: float,
    include_null: bool,
    include_junk: bool,
) -> np.ndarray:
    """Construct normalized initial mixture weight vector from substantive counts and null/junk probs."""
    parts: list[float] = []
    extra = (pi_null if include_null else 0.0) + (pi_junk if include_junk else 0.0)
    extra = min(max(extra, 0.0), 0.95)
    rem = 1.0 - extra
    c_arr = np.maximum(np.asarray(subst_counts, dtype=np.float64), 1.0)
    c_arr = c_arr / np.sum(c_arr) * rem
    parts.extend(c_arr.tolist())
    if include_null:
        parts.append(max(pi_null, 1e-4))
    if include_junk:
        parts.append(max(pi_junk, 1e-4))
    w = np.asarray(parts, dtype=np.float64)
    w /= np.sum(w)
    return w


def _gauss_log_spread(log_gauss: np.ndarray) -> np.ndarray:
    """Compute row-wise log-density range across Gaussian components without -inf NaN warnings."""
    max_g = np.max(log_gauss, axis=1)
    min_g = np.min(log_gauss, axis=1)
    with np.errstate(invalid="ignore"):
        diff = max_g - min_g
    return np.where(
        np.isfinite(max_g) & np.isfinite(min_g),
        diff,
        np.where(np.isfinite(max_g), np.inf, 0.0),
    )


def _top_novel_center_candidates(
    theta: np.ndarray,
    inv_se: np.ndarray,
    log_inv_se: np.ndarray,
    wp_mask: np.ndarray,
    existing_means: np.ndarray,
    existing_weights: np.ndarray,
    log_null: np.ndarray | None,
    log_junk: np.ndarray | None,
    resp: np.ndarray,
    max_candidates: int = 4,
) -> list[float]:
    """Identify high-value candidate centers for a new substantive cluster via 1-step log-lik gain."""
    n_obs = theta.size
    k_exist = existing_means.size
    n_comp_exist = existing_weights.size
    log_dens_curr = np.empty((n_obs, n_comp_exist), dtype=np.float64)
    if k_exist > 0:
        z = (theta[:, None] - existing_means[None, :]) * inv_se[:, None]
        with np.errstate(over="ignore"):
            log_dens_curr[:, :k_exist] = -0.5 * _LOG_2PI + log_inv_se[:, None] - 0.5 * (z * z)
    col_tail = k_exist
    if log_null is not None:
        log_dens_curr[:, col_tail] = log_null
        col_tail += 1
    if log_junk is not None:
        log_dens_curr[:, col_tail] = log_junk

    log_w_curr = log_dens_curr + np.log(np.maximum(existing_weights, 1e-300))[None, :]
    log_marg_curr = logsumexp(log_w_curr, axis=1)

    junk_col = resp[:, -1] if log_junk is not None else np.zeros(n_obs, dtype=np.float64)
    score = (junk_col + 0.1) * inv_se
    ll_rank = np.argsort(log_marg_curr, kind="mergesort")
    score_rank = np.argsort(-score, kind="mergesort")
    pool_idx = list(dict.fromkeys(list(score_rank[:30]) + list(ll_rank[:30])))
    if n_obs <= 80:
        pool_idx = list(range(n_obs))

    th_ref = theta[wp_mask]
    iqr = float(np.quantile(th_ref, 0.75) - np.quantile(th_ref, 0.25))
    scale = max(iqr, float(1.0 / np.median(inv_se[wp_mask])), 1e-300)
    min_gap = 1e-3 * scale

    pi_new = 0.10
    log_keep = float(np.log(1.0 - pi_new))
    log_new = float(np.log(pi_new))

    gains: list[tuple[float, float]] = []
    seen_vals: list[float] = []
    for idx in pool_idx:
        cand = float(theta[idx])
        if existing_means.size > 0 and np.min(np.abs(existing_means - cand)) < min_gap:
            continue
        if any(abs(cand - s) < min_gap for s in seen_vals):
            continue
        seen_vals.append(cand)
        z = (theta - cand) * inv_se
        with np.errstate(over="ignore"):
            log_f_new = -0.5 * _LOG_2PI + log_inv_se - 0.5 * (z * z)
        new_marg = np.logaddexp(log_keep + log_marg_curr, log_new + log_f_new)
        with np.errstate(invalid="ignore"):
            gain_vec = np.where(
                np.isfinite(new_marg) & np.isfinite(log_marg_curr),
                new_marg - log_marg_curr,
                np.where(new_marg > log_marg_curr, 1e6, 0.0),
            )
        gain = float(np.sum(gain_vec))
        gains.append((gain, cand))

    gains.sort(key=lambda x: -x[0])
    chosen: list[float] = []
    sep_tol = 0.05 * scale
    for gain, cand in gains:
        if gain <= 0.0 and chosen:
            break
        if all(abs(cand - c) > sep_tol for c in chosen):
            chosen.append(cand)
        if len(chosen) >= max_candidates:
            break
    if not chosen and gains:
        chosen.append(gains[0][1])
    return chosen


def _is_degenerate(
    theta: np.ndarray,
    inv_se: np.ndarray,
    log_inv_se: np.ndarray,
    log_null: np.ndarray | None,
    wp_mask: np.ndarray,
    means: np.ndarray,
    resp: np.ndarray,
    merge_tol: float,
    min_eff_n: float = 1.5,
) -> bool:
    """Check whether a K-cluster fit has collapsed, duplicate, or singleton substantive clusters."""
    k = means.size
    if k == 0:
        return False
    if k >= 2 and np.any(np.diff(means) < merge_tol):
        return True
    n_gauss = k + (1 if log_null is not None else 0)
    if n_gauss >= 2:
        z = (theta[:, None] - means[None, :]) * inv_se[:, None]
        with np.errstate(over="ignore"):
            log_subst = -0.5 * _LOG_2PI + log_inv_se[:, None] - 0.5 * (z * z)
        log_gauss = (
            np.concatenate([log_subst, log_null[:, None]], axis=1)
            if log_null is not None
            else log_subst
        )
        for m in range(k):
            other_cols = [c for c in range(n_gauss) if c != m]
            max_other = np.max(log_gauss[:, other_cols], axis=1)
            lm = log_gauss[:, m]
            with np.errstate(invalid="ignore"):
                margin = np.where(
                    np.isfinite(lm),
                    np.where(np.isfinite(max_other), lm - max_other, np.inf),
                    -np.inf,
                )
            info_m = wp_mask | (margin >= 0.5)
            if float(np.sum(resp[info_m, m])) < min_eff_n:
                return True
        return False
    eff_n = np.sum(resp[wp_mask, :k], axis=0)
    if np.any(eff_n < min_eff_n):
        return True
    return False


def _fit_for_k(
    theta: np.ndarray,
    theta_se: np.ndarray,
    inv_se: np.ndarray,
    log_inv_se: np.ndarray,
    rel_inv_var: np.ndarray,
    min_se: float,
    wp_mask: np.ndarray,
    k: int,
    log_null: np.ndarray | None,
    log_junk: np.ndarray | None,
    include_null: bool,
    include_junk: bool,
    prev_fit: ClusterFit | None,
    n_init: int,
    dirichlet_alpha: float,
    min_cluster_eff_n: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[ClusterFit, bool]:
    """Search for the global MLE fit for a given number of substantive clusters K."""
    n_obs = theta.size
    n_comp = k + (1 if include_null else 0) + (1 if include_junk else 0)
    n_params = k + max(n_comp - 1, 0)

    comp_names = tuple(
        [str(i + 1) for i in range(k)]
        + (["Null"] if include_null else [])
        + (["Junk"] if include_junk else [])
    )

    if k == 0:
        candidates_w: list[np.ndarray] = []
        if include_null and include_junk:
            for p_null in (0.2, 0.5, 0.8, 0.95):
                candidates_w.append(np.array([p_null, 1.0 - p_null], dtype=np.float64))
        else:
            candidates_w.append(np.array([1.0], dtype=np.float64))

        best_res = None
        for w0 in candidates_w:
            res = _run_em(
                theta=theta,
                inv_se=inv_se,
                log_inv_se=log_inv_se,
                rel_inv_var=rel_inv_var,
                min_se=min_se,
                init_means=np.empty(0, dtype=np.float64),
                init_weights=w0,
                log_null=log_null,
                log_junk=log_junk,
                dirichlet_alpha=dirichlet_alpha,
                max_iter=max_iter,
                tol=tol,
            )
            if best_res is None or res[4] > best_res[4]:
                best_res = res

        assert best_res is not None
        means, ses, weights, resp, ll, n_iter, conv = best_res
        bic = -2.0 * ll + n_params * float(np.log(n_obs))
        aic = -2.0 * ll + 2.0 * n_params
        return (
            ClusterFit(
                k=0,
                n_components=n_comp,
                n_parameters=n_params,
                log_likelihood=ll,
                bic=bic,
                aic=aic,
                cluster_means=means,
                cluster_ses=ses,
                weights=weights,
                component_names=comp_names,
                responsibilities=resp,
                n_iter=n_iter,
                converged=conv,
            ),
            True,
        )

    # K >= 1: build ensemble of initializations (means0, weights0)
    inits: list[tuple[np.ndarray, np.ndarray]] = []
    probs_q = np.linspace(1.0 / (k + 1), k / (k + 1), k)

    # 1. Unweighted and precision-weighted quantile seeds
    q_unw = np.quantile(theta, probs_q)
    q_w = _weighted_quantiles(theta, rel_inv_var, probs_q)
    inits.append((q_unw, _make_weights(np.ones(k), 0.20, 0.10, include_null, include_junk)))
    inits.append((q_w, _make_weights(np.ones(k), 0.20, 0.10, include_null, include_junk)))

    # 2. 1D Ward hierarchical linkage seed (if K <= n_obs), rescaled to prevent squared-dist overflow
    if k <= n_obs:
        if k == 1:
            ward_labels = np.ones(n_obs, dtype=np.int64)
        else:
            theta_scale = float(np.max(np.abs(theta)))
            theta_ward = (theta / theta_scale) if theta_scale > 0.0 else theta
            Z = linkage(theta_ward[:, None], method="ward")
            ward_labels = fcluster(Z, t=k, criterion="maxclust")
        uniq_labs = np.unique(ward_labels)
        if uniq_labs.size == k:
            w_means_arith = np.array([float(np.mean(theta[ward_labels == lab])) for lab in uniq_labs])
            w_means_ivw = np.array(
                [
                    float(
                        np.sum(rel_inv_var[ward_labels == lab] * theta[ward_labels == lab])
                        / np.maximum(np.sum(rel_inv_var[ward_labels == lab]), 1e-300)
                    )
                    for lab in uniq_labs
                ]
            )
            w_counts = np.array([float(np.sum(ward_labels == lab)) for lab in uniq_labs])
            o1 = np.argsort(w_means_arith, kind="mergesort")
            o2 = np.argsort(w_means_ivw, kind="mergesort")
            inits.append((w_means_arith[o1], _make_weights(w_counts[o1], 0.15, 0.10, include_null, include_junk)))
            inits.append((w_means_ivw[o2], _make_weights(w_counts[o2], 0.15, 0.10, include_null, include_junk)))

    # 3. Multi-start k-means++ (both unweighted and precision-weighted) + random Null/Junk weights
    sqrt_prec = np.sqrt(rel_inv_var)
    for r in range(n_init):
        s_nj = float(rng.uniform(0.10, 0.80))
        u = float(rng.uniform(0.10, 0.90))
        if include_null and include_junk:
            p_null, p_junk = u * s_nj, (1.0 - u) * s_nj
        elif include_null:
            p_null, p_junk = s_nj, 0.0
        elif include_junk:
            p_null, p_junk = 0.0, s_nj
        else:
            p_null, p_junk = 0.0, 0.0

        sw = None if r % 3 == 0 else (rel_inv_var if r % 3 == 1 else sqrt_prec)
        centers, counts = _kmeans_1d(theta, k, rng, sample_weights=sw)
        inits.append((centers, _make_weights(counts, p_null, p_junk, include_null, include_junk)))

    # 4. Forward split + local-gain Junk/Null rescue from converged (K - 1) fit
    th_wp = theta[wp_mask]
    med_se_wp = float(np.median(theta_se[wp_mask]))
    if prev_fit is not None and prev_fit.k == k - 1:
        prev_means = prev_fit.cluster_means
        prev_w = prev_fit.weights
        prev_resp = prev_fit.responsibilities

        if prev_fit.k > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                chi2 = np.array(
                    [
                        float(
                            np.sum(
                                np.where(
                                    prev_resp[:, m] > 0.0,
                                    prev_resp[:, m] * ((theta - prev_means[m]) * inv_se) ** 2,
                                    0.0,
                                )
                            )
                        )
                        for m in range(prev_fit.k)
                    ]
                )
            split_order = np.argsort(-chi2, kind="mergesort")[:2]
            for m in split_order:
                rm = prev_resp[:, m]
                rm_sum = float(np.sum(rm))
                if rm_sum <= 1.0:
                    continue
                diff_m = theta - prev_means[m]
                diff_max = float(np.max(np.abs(diff_m)))
                if diff_max > 0.0:
                    sd_m = diff_max * float(np.sqrt(np.sum(rm * (diff_m / diff_max) ** 2) / rm_sum))
                else:
                    sd_m = 0.0
                delta_m = max(sd_m, med_se_wp)
                new_means = np.delete(prev_means, m)
                new_means = np.sort(
                    np.concatenate([new_means, [prev_means[m] - delta_m, prev_means[m] + delta_m]])
                )
                subst_w = np.delete(prev_w[: prev_fit.k], m)
                subst_w = np.concatenate([subst_w, [0.5 * prev_w[m], 0.5 * prev_w[m]]])
                tail_w = prev_w[prev_fit.k :]
                w0 = np.concatenate([subst_w, tail_w])
                w0 = np.maximum(w0, 1e-4)
                w0 /= np.sum(w0)
                inits.append((new_means, w0))

        cand_centers = _top_novel_center_candidates(
            theta=theta,
            inv_se=inv_se,
            log_inv_se=log_inv_se,
            wp_mask=wp_mask,
            existing_means=prev_means,
            existing_weights=prev_w,
            log_null=log_null,
            log_junk=log_junk,
            resp=prev_resp,
            max_candidates=4,
        )
        for cand in cand_centers:
            new_means = np.sort(np.concatenate([prev_means, [cand]]))
            new_subst_w = np.concatenate([0.85 * prev_w[: prev_fit.k], [0.15]])
            tail_w = 0.85 * prev_w[prev_fit.k :]
            w0 = np.concatenate([new_subst_w, tail_w])
            w0 = np.maximum(w0, 1e-4)
            w0 /= np.sum(w0)
            inits.append((new_means, w0))

    iqr_wp = float(np.quantile(th_wp, 0.75) - np.quantile(th_wp, 0.25))
    merge_tol = 1e-4 * max(iqr_wp, med_se_wp, 1e-300)

    best_valid = None
    best_overall = None

    for means0, w0 in inits:
        res = _run_em(
            theta=theta,
            inv_se=inv_se,
            log_inv_se=log_inv_se,
            rel_inv_var=rel_inv_var,
            min_se=min_se,
            init_means=means0,
            init_weights=w0,
            log_null=log_null,
            log_junk=log_junk,
            dirichlet_alpha=dirichlet_alpha,
            max_iter=max_iter,
            tol=tol,
        )
        if best_overall is None or res[4] > best_overall[4]:
            best_overall = res
        if not _is_degenerate(
            theta=theta,
            inv_se=inv_se,
            log_inv_se=log_inv_se,
            log_null=log_null,
            wp_mask=wp_mask,
            means=res[0],
            resp=res[3],
            merge_tol=merge_tol,
            min_eff_n=min_cluster_eff_n,
        ):
            if best_valid is None or res[4] > best_valid[4]:
                best_valid = res

    # 5. Merge-and-resplit refinement guard on best solution for K >= 2
    anchor = best_valid if best_valid is not None else best_overall
    assert anchor is not None
    if k >= 2:
        m_cur, _, w_cur, r_cur, _, _, _ = anchor
        diffs = np.diff(m_cur)
        pair_idx = int(np.argmin(diffs))
        r_pair = r_cur[:, pair_idx] + r_cur[:, pair_idx + 1]
        prec_pair = r_pair * rel_inv_var
        if np.sum(prec_pair) > 0.0:
            merged_mean = float(np.sum(prec_pair * theta) / np.sum(prec_pair))
        else:
            merged_mean = 0.5 * float(m_cur[pair_idx] + m_cur[pair_idx + 1])
        reduced_means = np.delete(m_cur, [pair_idx, pair_idx + 1])
        reduced_means = np.sort(np.append(reduced_means, merged_mean))
        reduced_subst_w = np.delete(w_cur[:k], [pair_idx, pair_idx + 1])
        reduced_subst_w = np.append(reduced_subst_w, w_cur[pair_idx] + w_cur[pair_idx + 1])
        reduced_w = np.concatenate([reduced_subst_w, w_cur[k:]])
        reduced_w = np.maximum(reduced_w, 1e-4)
        reduced_w /= np.sum(reduced_w)

        cands = _top_novel_center_candidates(
            theta=theta,
            inv_se=inv_se,
            log_inv_se=log_inv_se,
            wp_mask=wp_mask,
            existing_means=reduced_means,
            existing_weights=reduced_w,
            log_null=log_null,
            log_junk=log_junk,
            resp=r_cur,
            max_candidates=2,
        )
        for cand in cands:
            means_try = np.sort(np.append(reduced_means, cand))
            w_try = np.concatenate([0.85 * reduced_subst_w, [0.15], 0.85 * w_cur[k:]])
            w_try = np.maximum(w_try, 1e-4)
            w_try /= np.sum(w_try)
            res_try = _run_em(
                theta=theta,
                inv_se=inv_se,
                log_inv_se=log_inv_se,
                rel_inv_var=rel_inv_var,
                min_se=min_se,
                init_means=means_try,
                init_weights=w_try,
                log_null=log_null,
                log_junk=log_junk,
                dirichlet_alpha=dirichlet_alpha,
                max_iter=max_iter,
                tol=tol,
            )
            if res_try[4] > best_overall[4]:
                best_overall = res_try
            if not _is_degenerate(
                theta=theta,
                inv_se=inv_se,
                log_inv_se=log_inv_se,
                log_null=log_null,
                wp_mask=wp_mask,
                means=res_try[0],
                resp=res_try[3],
                merge_tol=merge_tol,
                min_eff_n=min_cluster_eff_n,
            ):
                if best_valid is None or res_try[4] > best_valid[4]:
                    best_valid = res_try

    is_admissible = best_valid is not None
    chosen = best_valid if best_valid is not None else best_overall
    means, ses, weights, resp, ll, n_iter, conv = chosen
    # Polish the single winning solution to tight M-step fixed-point precision
    polished = _run_em(
        theta=theta,
        inv_se=inv_se,
        log_inv_se=log_inv_se,
        rel_inv_var=rel_inv_var,
        min_se=min_se,
        init_means=means,
        init_weights=weights,
        log_null=log_null,
        log_junk=log_junk,
        dirichlet_alpha=dirichlet_alpha,
        max_iter=min(max_iter, 250),
        tol=min(tol, 1e-12),
        param_tol=1e-11,
    )
    if polished[4] >= ll:
        means, ses, weights, resp, ll, n_polish, conv_polish = polished
        n_iter = n_iter + n_polish
        conv = conv or conv_polish
    bic = -2.0 * ll + n_params * float(np.log(n_obs))
    aic = -2.0 * ll + 2.0 * n_params

    return (
        ClusterFit(
            k=k,
            n_components=n_comp,
            n_parameters=n_params,
            log_likelihood=ll,
            bic=bic,
            aic=aic,
            cluster_means=means,
            cluster_ses=ses,
            weights=weights,
            component_names=comp_names,
            responsibilities=resp,
            n_iter=n_iter,
            converged=conv,
        ),
        is_admissible,
    )


def mr_clust_em(
    theta: str | Sequence[float] | np.ndarray | None = None,
    theta_se: str | Sequence[float] | np.ndarray | None = None,
    *,
    bx: str | Sequence[float] | np.ndarray | None = None,
    bxse: str | Sequence[float] | np.ndarray | None = None,
    by: str | Sequence[float] | np.ndarray | None = None,
    byse: str | Sequence[float] | np.ndarray | None = None,
    snp_names: Sequence[str] | None = None,
    obs_names: Sequence[str] | None = None,
    data: pd.DataFrame | None = None,
    variance_order: Literal[1, 2, "first", "second"] = 1,
    k_min: int = 0,
    k_max: int | None = None,
    min_clust_search: int = 10,
    early_stop: bool = False,
    obs_prob_thresh: float = 0.8,
    include_null: bool = True,
    include_junk: bool = True,
    junk_dist: Literal["t", "normal", "uniform"] = "t",
    junk_df: float = 4.0,
    junk_mean: float | None = None,
    junk_scale: float | None = None,
    dirichlet_alpha: float = 0.0,
    min_cluster_eff_n: float = 1.5,
    n_init: int = 20,
    max_iter: int = 5000,
    tol: float = 1e-6,
    seed: int | None = 42,
) -> MRClustResult:
    """Fit the MR-Clust Gaussian-mixture EM model with BIC selection across K.

    Parameters
    ----------
    theta, theta_se : array-like or column name, optional
        Precomputed Wald ratio estimates and standard errors.
    bx, bxse, by, byse : array-like or column name, optional
        Summary association estimates and standard errors for exposure (`bx`, `bxse`)
        and outcome (`by`, `byse`). `bxse` is optional when `variance_order=1`.
    snp_names, obs_names : sequence of str, optional
        Instrument identifiers (`obs_names` is an alias for `snp_names`).
    data : pd.DataFrame, optional
        Optional summary-statistics DataFrame. Supports column-name strings in `bx`/`by`
        or automatic column detection (`theta`/`theta_se` or `bx`/`by`/`byse`/`beta_exp`...).
    variance_order : {1, 2, "first", "second"}, default=1
        Delta-method variance order when computing ratios from `(bx, by, byse, bxse)`.
    k_min : int, default=0
        Minimum number of substantive clusters to evaluate.
    k_max : int, optional
        Maximum number of substantive clusters to evaluate (default `min(10, J)`).
    min_clust_search : int, default=10
        Minimum K reached before `early_stop=True` may terminate the BIC search.
    early_stop : bool, default=False
        If True, stop increasing K once `k >= min_clust_search` and BIC increases.
    obs_prob_thresh : float, default=0.8
        Posterior probability threshold separating confident assignments from `"Unassigned"`.
    include_null : bool, default=True
        Include a fixed-zero null component N(0, theta_se^2).
    include_junk : bool, default=True
        Include a broad outlier/junk component.
    junk_dist : {"t", "normal", "uniform"}, default="t"
        Distribution family for the junk component (`"t"` matches Foley et al., 2021).
    junk_df : float, default=4.0
        Degrees of freedom when `junk_dist="t"`.
    junk_mean : float, optional
        Optional fixed mean for the junk component (default: mean of `theta`).
    junk_scale : float, optional
        Optional fixed scale for the junk component.
    dirichlet_alpha : float, default=0.0
        Optional symmetric Dirichlet smoothing for mixture weights (0.0 = exact MLE).
    min_cluster_eff_n : float, default=1.5
        Minimum effective number of variants (`sum_j r_{j,k}`) required for a substantive
        cluster to be considered non-singleton during BIC model selection.
    n_init : int, default=20
        Number of multi-start k-means++ initializations per K.
    max_iter : int, default=5000
        Maximum EM iterations per initialization.
    tol : float, default=1e-6
        Log-likelihood convergence tolerance for EM.
    seed : int or None, default=42
        Random seed for deterministic initialization (`np.random.default_rng(seed)`).

    Returns
    -------
    MRClustResult
        Frozen dataclass containing the BIC-selected model, all evaluated K models,
        cluster metadata, and per-variant responsibilities.
    """
    if not np.isfinite(obs_prob_thresh) or obs_prob_thresh < 0.0 or obs_prob_thresh > 1.0:
        raise ValueError(f"obs_prob_thresh must be in [0, 1], got {obs_prob_thresh!r}.")
    if k_min < 0:
        raise ValueError(f"k_min must be >= 0, got {k_min!r}.")
    if not include_null and not include_junk and k_min == 0:
        raise ValueError("k_min must be >= 1 when both include_null and include_junk are False.")
    if dirichlet_alpha < 0.0 or not np.isfinite(dirichlet_alpha):
        raise ValueError(f"dirichlet_alpha must be finite and >= 0, got {dirichlet_alpha!r}.")
    if min_cluster_eff_n < 0.0 or not np.isfinite(min_cluster_eff_n):
        raise ValueError(f"min_cluster_eff_n must be finite and >= 0, got {min_cluster_eff_n!r}.")
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init!r}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}.")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError(f"tol must be finite and > 0, got {tol!r}.")

    (
        theta_arr,
        theta_se_arr,
        bx_arr,
        bxse_arr,
        by_arr,
        byse_arr,
        used_order,
        names_tuple,
    ) = _normalize_inputs(
        theta=theta,
        theta_se=theta_se,
        bx=bx,
        bxse=bxse,
        by=by,
        byse=byse,
        snp_names=snp_names,
        obs_names=obs_names,
        data=data,
        variance_order=variance_order,
    )

    n_snps = int(theta_arr.size)
    k_upper = min(10, n_snps) if k_max is None else int(k_max)
    if k_upper < k_min:
        raise ValueError(f"k_max ({k_upper}) must be >= k_min ({k_min}).")

    min_se = float(np.min(theta_se_arr))
    inv_se = 1.0 / theta_se_arr
    log_inv_se = -np.log(theta_se_arr)
    rel_inv_var = (min_se * inv_se) ** 2

    log_null = (
        _normal_logpdf_matrix(theta_arr, theta_se_arr, np.array([0.0], dtype=np.float64))[:, 0]
        if include_null
        else None
    )

    junk_defaults = _robust_junk_defaults(theta_arr, theta_se_arr)
    wp_mask = junk_defaults[2]

    if include_junk:
        mu_junk, psi_junk, log_junk = _compute_junk_params_and_logpdf(
            theta=theta_arr,
            theta_se=theta_se_arr,
            junk_dist=junk_dist,
            junk_df=junk_df,
            junk_mean=junk_mean,
            junk_scale=junk_scale,
            junk_defaults=junk_defaults,
        )
    else:
        mu_junk = junk_defaults[0] if junk_mean is None else float(junk_mean)
        psi_junk = float(junk_scale) if junk_scale is not None else 0.0
        log_junk = None

    rng = np.random.default_rng(seed)
    all_models: list[ClusterFit] = []
    admissible_flags: list[bool] = []
    prev_fit: ClusterFit | None = None

    for k in range(k_min, k_upper + 1):
        fit, is_adm = _fit_for_k(
            theta=theta_arr,
            theta_se=theta_se_arr,
            inv_se=inv_se,
            log_inv_se=log_inv_se,
            rel_inv_var=rel_inv_var,
            min_se=min_se,
            wp_mask=wp_mask,
            k=k,
            log_null=log_null,
            log_junk=log_junk,
            include_null=include_null,
            include_junk=include_junk,
            prev_fit=prev_fit,
            n_init=n_init,
            dirichlet_alpha=dirichlet_alpha,
            min_cluster_eff_n=min_cluster_eff_n,
            max_iter=max_iter,
            tol=tol,
            rng=rng,
        )
        all_models.append(fit)
        admissible_flags.append(is_adm)
        prev_fit = fit
        if early_stop and k >= min_clust_search and len(all_models) >= 2:
            if all_models[-1].bic > all_models[-2].bic:
                break

    admissible_models = [m for m, adm in zip(all_models, admissible_flags) if adm]
    pool_models = admissible_models if admissible_models else all_models
    best_fit = min(pool_models, key=lambda m: (m.bic, m.k))
    resp = best_fit.responsibilities
    best_idx = np.argmax(resp, axis=1)
    assign_probs = resp[np.arange(n_snps), best_idx]

    n_gauss = best_fit.k + (1 if log_null is not None else 0)
    active_gauss = [g for g in range(n_gauss) if float(best_fit.weights[g]) > 1e-4]
    if len(active_gauss) >= 2:
        gauss_cols: list[np.ndarray] = []
        if best_fit.k > 0:
            z_g = (theta_arr[:, None] - best_fit.cluster_means[None, :]) * inv_se[:, None]
            with np.errstate(over="ignore"):
                log_subst = -0.5 * _LOG_2PI + log_inv_se[:, None] - 0.5 * (z_g * z_g)
            gauss_cols.append(log_subst)
        if log_null is not None:
            gauss_cols.append(log_null[:, None])
        log_gauss_all = np.concatenate(gauss_cols, axis=1)
        flat_gauss = _gauss_log_spread(log_gauss_all[:, active_gauss]) < 0.25
        active_centers = [
            float(best_fit.cluster_means[g]) if g < best_fit.k else 0.0
            for g in active_gauss
        ]
        active_span = float(np.ptp(active_centers))
    else:
        flat_gauss = np.ones(n_snps, dtype=bool)
        active_span = 0.0

    med_se_all = float(np.median(theta_se_arr))
    if include_junk:
        junk_idx = best_fit.n_components - 1
        junk_active = float(best_fit.weights[junk_idx]) > 1e-4
        above_junk_scale = (
            (theta_se_arr > 2.0 * psi_junk)
            if psi_junk > 0.0
            else np.ones(n_snps, dtype=bool)
        )
        if len(active_gauss) >= 2:
            weak_se = above_junk_scale & (theta_se_arr > 2.0 * active_span)
        else:
            weak_se = (
                above_junk_scale & ((theta_se_arr > 2.0 * med_se_all) | (assign_probs < 0.75))
                if junk_active
                else (theta_se_arr > max(10.0 * psi_junk, 100.0 * med_se_all))
            )
    elif include_null:
        if len(active_gauss) >= 2:
            weak_se = theta_se_arr > 2.0 * active_span
        else:
            weak_se = theta_se_arr > 100.0 * med_se_all
    else:
        weak_se = np.zeros(n_snps, dtype=bool)

    uninformative_weak = (~wp_mask) & flat_gauss & weak_se & (best_idx < best_fit.k)
    if np.any(uninformative_weak):
        if include_junk:
            junk_idx = best_fit.n_components - 1
            best_idx[uninformative_weak] = junk_idx
            assign_probs[uninformative_weak] = resp[uninformative_weak, junk_idx]
        elif include_null:
            null_idx = best_fit.k
            best_idx[uninformative_weak] = null_idx
            assign_probs[uninformative_weak] = resp[uninformative_weak, null_idx]

    best_cluster = tuple(best_fit.component_names[i] for i in best_idx)
    membership = tuple(
        bc if p >= obs_prob_thresh else "Unassigned"
        for bc, p in zip(best_cluster, assign_probs)
    )

    clusters_info: list[ClusterInfo] = []
    for c_idx, c_name in enumerate(best_fit.component_names):
        if c_idx < best_fit.k:
            c_class: Literal["Substantive", "Null", "Junk"] = "Substantive"
            c_mean = float(best_fit.cluster_means[c_idx])
            c_se = float(best_fit.cluster_ses[c_idx])
        elif c_name == "Null":
            c_class = "Null"
            c_mean = 0.0
            c_se = float("nan")
        else:
            c_class = "Junk"
            c_mean = mu_junk
            c_se = float("nan")

        mask_map = best_idx == c_idx
        n_assigned = int(np.sum(mask_map))
        n_assigned_thresh = int(np.sum(mask_map & (assign_probs >= obs_prob_thresh)))
        clusters_info.append(
            ClusterInfo(
                cluster_id=c_name,
                cluster_index=c_idx,
                cluster_class=c_class,
                mean=c_mean,
                se=c_se,
                weight=float(best_fit.weights[c_idx]),
                effective_n=float(np.sum(resp[:, c_idx])),
                n_assigned=n_assigned,
                n_assigned_thresh=n_assigned_thresh,
            )
        )

    return MRClustResult(
        snp_names=names_tuple,
        n_snps=n_snps,
        theta=theta_arr,
        theta_se=theta_se_arr,
        bx=bx_arr,
        bxse=bxse_arr,
        by=by_arr,
        byse=byse_arr,
        variance_order=used_order,
        best_k=best_fit.k,
        best_model=best_fit,
        all_models=tuple(all_models),
        clusters=tuple(clusters_info),
        best_cluster=best_cluster,
        cluster_membership=membership,
        assignment_probs=assign_probs,
        responsibilities=resp,
        obs_prob_thresh=float(obs_prob_thresh),
        junk_dist=junk_dist,
        junk_mean=mu_junk,
        junk_scale=psi_junk,
        junk_df=float(junk_df),
    )


def pr_clust(
    result: MRClustResult,
    obs_prob_thresh: float = 0.8,
    min_cluster_size: int = 1,
    include_null: bool = True,
    include_junk: bool = False,
) -> pd.DataFrame:
    """Filter variant-cluster assignments by posterior probability and minimum cluster size.

    Parameters
    ----------
    result : MRClustResult
        Output of `mr_clust_em`.
    obs_prob_thresh : float, default=0.8
        Minimum posterior assignment probability required to retain a variant.
    min_cluster_size : int, default=1
        Minimum number of threshold-passing variants required to retain a cluster.
    include_null : bool, default=True
        Whether to retain variants assigned to the `"Null"` cluster.
    include_junk : bool, default=False
        Whether to retain variants assigned to the `"Junk"` cluster.

    Returns
    -------
    pd.DataFrame
        Filtered tidy assignment table.
    """
    if not isinstance(result, MRClustResult):
        raise ValueError("`result` must be an MRClustResult instance.")
    if not np.isfinite(obs_prob_thresh) or obs_prob_thresh < 0.0 or obs_prob_thresh > 1.0:
        raise ValueError(f"obs_prob_thresh must be in [0, 1], got {obs_prob_thresh!r}.")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size!r}.")

    df = result.to_dataframe(obs_prob_thresh=obs_prob_thresh)
    df = df[df["passes_threshold"]].copy()
    if not include_null:
        df = df[df["cluster_class"] != "Null"]
    if not include_junk:
        df = df[df["cluster_class"] != "Junk"]
    if df.empty:
        return df.reset_index(drop=True)

    counts = df["cluster"].value_counts()
    keep_clusters = set(counts[counts >= min_cluster_size].index.tolist())
    df = df[df["cluster"].isin(keep_clusters)].reset_index(drop=True)
    return df
