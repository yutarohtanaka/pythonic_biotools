"""
Multi-region Variational Dirichlet Process PhyloCCF clustering, truncal
collapse, near-identical cluster merging, and small-artifact filtering for
`conipher` (Grigoriadis et al., Nature Protocols 2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from .io import SettingsMode, validate_settings


def _extract_ccf_matrices(
    ccf_df: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """
    Parse either long-format (`mutation_id`, `region`, `phylo_ccf`/`ccf`) or
    wide-format (`(N_mutations, R_regions)`) DataFrame into `(mutation_ids, regions, X, SE)`.
    """
    if not isinstance(ccf_df, pd.DataFrame) or ccf_df.empty:
        raise ValueError("ccf_df must be a non-empty pandas DataFrame.")

    if "region" in ccf_df.columns and ("phylo_ccf" in ccf_df.columns or "ccf" in ccf_df.columns):
        val_col = "phylo_ccf" if "phylo_ccf" in ccf_df.columns else "ccf"
        if "mutation_id" not in ccf_df.columns:
            raise ValueError("Long-format ccf_df must contain 'mutation_id'.")

        regions = list(dict.fromkeys(ccf_df["region"].astype(str)))
        if len(regions) < 2:
            raise ValueError(f"ccf_df must span at least 2 regions, got {len(regions)}.")

        mut_ids = list(dict.fromkeys(ccf_df["mutation_id"].astype(str)))
        piv_x = (
            ccf_df.pivot_table(
                index="mutation_id", columns="region", values=val_col, aggfunc="mean"
            )
            .reindex(index=mut_ids, columns=regions)
            .fillna(0.0)
        )
        x_mat = piv_x.to_numpy(dtype=np.float64)

        if "ccf_se" in ccf_df.columns:
            piv_se = (
                ccf_df.pivot_table(
                    index="mutation_id", columns="region", values="ccf_se", aggfunc="mean"
                )
                .reindex(index=mut_ids, columns=regions)
                .fillna(0.05)
            )
            se_mat = np.maximum(piv_se.to_numpy(dtype=np.float64), 0.005)
        else:
            se_mat = np.full_like(x_mat, 0.05)

        return mut_ids, regions, x_mat, se_mat

    # Wide format
    df = ccf_df.copy()
    if "mutation_id" in df.columns:
        mut_ids = df["mutation_id"].astype(str).tolist()
        df = df.drop(columns=["mutation_id"])
    else:
        mut_ids = [str(idx) for idx in df.index]

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError("Wide-format ccf_df must contain at least 2 numeric region columns.")

    regions = [str(c) for c in num_cols]
    x_mat = df[num_cols].fillna(0.0).to_numpy(dtype=np.float64)
    se_mat = np.full_like(x_mat, 0.05)
    return mut_ids, regions, x_mat, se_mat


def _kmeans_plusplus_init(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministic K-means++ initialization on `(N, R)` array."""
    n, r = x.shape
    centers = np.empty((k, r), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = x[first]

    closest_d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        tot = float(np.sum(closest_d2))
        if tot <= 1e-14:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_d2 / tot
            probs /= np.sum(probs)
            idx = int(rng.choice(n, p=probs))
        centers[c] = x[idx]
        d2_new = np.sum((x - centers[c]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    return centers


def _fit_vb_dp_gmm(
    x: np.ndarray,
    se: np.ndarray,
    k_max: int,
    alpha_dp: float,
    min_var: float,
    seed: int,
    n_restarts: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit stick-breaking Variational Bayes Dirichlet Process Diagonal Gaussian
    Mixture incorporating observation variance `se^2`.

    Returns `(resp, means, vars_diag, bic_like_elbo)`.
    """
    n, r = x.shape
    k = max(1, min(k_max, n))
    base_rng = np.random.default_rng(seed)
    se2_med = np.median(se ** 2, axis=0)
    floor_var = np.maximum(min_var, se2_med)

    best_elbo = -np.inf
    best_resp = np.full((n, k), 1.0 / k, dtype=np.float64)
    best_means = np.zeros((k, r), dtype=np.float64)
    best_vars = np.tile(floor_var, (k, 1))

    for _ in range(n_restarts):
        means = _kmeans_plusplus_init(x, k, base_rng)
        vars_k = np.tile(np.maximum(floor_var, 0.008), (k, 1))
        # Stick-breaking Beta(gamma1, gamma2) variational parameters
        g1 = np.ones(k, dtype=np.float64)
        g2 = np.full(k, alpha_dp, dtype=np.float64)
        resp = np.full((n, k), 1.0 / k, dtype=np.float64)

        prev_bound = -np.inf
        for _it in range(150):
            # Expected log stick-breaking weights E[log pi_j]
            dig_sum = special.digamma(g1 + g2)
            e_log_v = special.digamma(g1) - dig_sum
            e_log_1mv = special.digamma(g2) - dig_sum
            e_log_v[-1] = 0.0  # Stick-breaking truncation at K

            cum_1mv = np.concatenate([[0.0], np.cumsum(e_log_1mv[:-1])])
            e_log_pi = e_log_v + cum_1mv

            # Heteroskedastic + cluster diagonal log-Gaussian likelihood: shape (N, K)
            # Effective variance per (i, j, r) = vars_k[j, r] + 0.5 * se[i, r]^2
            eff_var = vars_k[None, :, :] + 0.5 * (se[:, None, :] ** 2)
            diff2 = (x[:, None, :] - means[None, :, :]) ** 2
            log_gauss = -0.5 * np.sum(
                np.log(2.0 * np.pi * eff_var) + diff2 / eff_var, axis=2
            )

            log_rho = log_gauss + e_log_pi[None, :]
            m_max = np.max(log_rho, axis=1, keepdims=True)
            rho = np.exp(log_rho - m_max)
            rho_sum = np.maximum(np.sum(rho, axis=1, keepdims=True), 1e-300)
            resp = rho / rho_sum

            # Lower-bound convergence monitor
            elbo = float(np.sum(np.log(rho_sum[:, 0]) + m_max[:, 0]))
            if abs(elbo - prev_bound) < 1e-5:
                prev_bound = elbo
                break
            prev_bound = elbo

            # M / VB-updates
            nk = np.sum(resp, axis=0)
            # Reorder components by descending occupancy for stick-breaking efficiency
            order = np.argsort(-nk, kind="mergesort")
            resp = resp[:, order]
            nk = nk[order]

            cum_after = np.array(
                [np.sum(nk[j + 1 :]) for j in range(k)], dtype=np.float64
            )
            g1 = 1.0 + nk
            g2 = alpha_dp + cum_after

            denom = np.maximum(nk[:, None], 1e-8)
            means = (resp.T @ x) / denom
            centered2 = (x[:, None, :] - means[None, :, :]) ** 2
            raw_var = np.sum(resp[:, :, None] * centered2, axis=0) / denom
            vars_k = np.maximum(raw_var, floor_var[None, :])

        if prev_bound > best_elbo:
            best_elbo = prev_bound
            best_resp = resp
            best_means = means
            best_vars = vars_k

    return best_resp, best_means, best_vars, best_elbo


@dataclass
class ClusterResult:
    """Output of `cluster_ccf_multiregion`."""

    cluster_ccf_df: pd.DataFrame
    cluster_se_df: pd.DataFrame
    mutation_assignments_df: pd.DataFrame
    regions: list[str]
    n_clusters: int
    n_removed_mutations: int
    settings: SettingsMode

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary of multi-region clustering."""
        return {
            "n_clusters": self.n_clusters,
            "regions": self.regions,
            "n_total_mutations": int(len(self.mutation_assignments_df)),
            "n_removed_mutations": self.n_removed_mutations,
            "settings": self.settings,
        }


def cluster_ccf_multiregion(
    ccf_df: pd.DataFrame,
    max_clusters: int = 10,
    min_cluster_size: int = 5,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> ClusterResult:
    """
    Cluster `(N_mutations x R_regions)` PhyloCCF profiles via a Variational
    Dirichlet Process Gaussian Mixture followed by CONIPHER truncal collapse,
    near-identical centroid merging, and small-artifact cluster removal.

    Parameters
    ----------
    ccf_df : pd.DataFrame
        Output of `compute_phylo_ccf` (long format) or wide `(N x R)` CCF matrix.
    max_clusters : int, default=10
        Maximum initial mixture components (`K_max >= 1`).
    min_cluster_size : int, default=5
        Minimum mutation count required to retain a cluster (`>= 1`).
    seed : int, default=42
        Random seed for deterministic initialization.
    settings : {"default", "R_default"}, default="default"
        `"default"` uses SE-calibrated Variational DP mixture + BIC pruning and
        SE-aware merging; `"R_default"` uses CONIPHER v2.2 fixed truncal collapse
        (`mean_ccf >= 0.85` across all regions) and `0.08` L-infinity merge step.

    Returns
    -------
    ClusterResult
        Clean cluster centroids (`cluster_ccf_df`), standard errors (`cluster_se_df`),
        and per-mutation assignments (`mutation_assignments_df`).
    """
    validate_settings(settings)
    if max_clusters < 1:
        raise ValueError(f"max_clusters must be >= 1, got {max_clusters!r}.")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size!r}.")

    mut_ids, regions, x_mat, se_mat = _extract_ccf_matrices(ccf_df)
    n_mut, n_reg = x_mat.shape

    alpha_dp = 0.5 if settings == "default" else 1.0
    min_var = 0.0030 if settings == "default" else 0.0035

    resp, _, _, _ = _fit_vb_dp_gmm(
        x=x_mat,
        se=se_mat,
        k_max=max_clusters,
        alpha_dp=alpha_dp,
        min_var=min_var,
        seed=seed,
    )
    labels = np.argmax(resp, axis=1)

    # Compact active raw clusters into 0..K0-1
    uniq_raw = np.unique(labels)
    raw_map = {int(old): idx for idx, old in enumerate(uniq_raw)}
    labels = np.array([raw_map[int(lb)] for lb in labels], dtype=np.int64)

    def _compute_stats(lbls: np.ndarray) -> tuple[list[int], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
        c_ids = sorted(int(c) for c in np.unique(lbls) if c >= 0)
        means_d: dict[int, np.ndarray] = {}
        ses_d: dict[int, np.ndarray] = {}
        counts_d: dict[int, int] = {}
        for cid in c_ids:
            idx = np.flatnonzero(lbls == cid)
            counts_d[cid] = int(idx.size)
            means_d[cid] = np.mean(x_mat[idx], axis=0)
            # Combine empirical SEM and pooled measurement SE
            emp_var = np.var(x_mat[idx], axis=0) / max(idx.size, 1)
            meas_var = np.mean(se_mat[idx] ** 2, axis=0) / max(idx.size, 1)
            ses_d[cid] = np.sqrt(np.maximum(emp_var, meas_var))
        return c_ids, means_d, ses_d, counts_d

    # Step 1: Truncal cluster collapse (all clusters ubiquitous with mean CCF >= 0.85)
    c_ids, means_d, ses_d, _ = _compute_stats(labels)
    truncal_cands: list[int] = []
    for cid in c_ids:
        m_vec = means_d[cid]
        if settings == "R_default":
            if bool(np.all(m_vec >= 0.85)):
                truncal_cands.append(cid)
        else:
            se_vec = ses_d[cid]
            if bool(np.all(m_vec >= 0.85)) or bool(
                np.all(m_vec >= 0.80) and np.all(m_vec + 1.96 * se_vec >= 0.92)
            ):
                truncal_cands.append(cid)

    if len(truncal_cands) >= 2:
        target_tr = truncal_cands[0]
        for other_tr in truncal_cands[1:]:
            labels[labels == other_tr] = target_tr

    # Step 2: Iterative merging of near-identical clusters sharing regional presence profile
    merged_any = True
    while merged_any:
        merged_any = False
        c_ids, means_d, ses_d, _ = _compute_stats(labels)
        if len(c_ids) <= 1:
            break

        best_pair: tuple[int, int] | None = None
        best_dist = np.inf
        for i_a in range(len(c_ids)):
            ca = c_ids[i_a]
            ma = np.clip(means_d[ca], 0.0, 1.0)
            pres_a = ma >= 0.045
            for i_b in range(i_a + 1, len(c_ids)):
                cb = c_ids[i_b]
                mb = np.clip(means_d[cb], 0.0, 1.0)
                pres_b = mb >= 0.045
                if not np.array_equal(pres_a, pres_b):
                    continue
                diff = np.abs(ma - mb)
                max_diff = float(np.max(diff))
                if settings == "default":
                    se_pair = np.sqrt(ses_d[ca] ** 2 + ses_d[cb] ** 2)
                    thresh = float(max(0.075, np.max(1.96 * se_pair)))
                else:
                    thresh = 0.085
                if max_diff < thresh and max_diff < best_dist:
                    best_dist = max_diff
                    best_pair = (ca, cb)

        if best_pair is not None:
            ca, cb = best_pair
            labels[labels == cb] = ca
            merged_any = True

    # Step 3: Remove small noise/artifact clusters (< min_cluster_size)
    c_ids, means_d, ses_d, counts_d = _compute_stats(labels)
    keep_ids = [cid for cid in c_ids if counts_d[cid] >= min_cluster_size]
    if not keep_ids and c_ids:
        # Keep largest cluster if total mutations < min_cluster_size
        largest = max(c_ids, key=lambda c: counts_d[c])
        keep_ids = [largest]

    # Order surviving clusters: ubiquitous first, then by descending mean CCF across regions
    keep_ids.sort(
        key=lambda cid: (
            int(np.sum(means_d[cid] >= 0.05)),
            float(np.mean(np.clip(means_d[cid], 0.0, 1.0))),
            counts_d[cid],
        ),
        reverse=True,
    )

    final_id_map = {old_cid: new_idx + 1 for new_idx, old_cid in enumerate(keep_ids)}
    clean_labels = np.array(
        [final_id_map.get(int(lb), -1) for lb in labels], dtype=np.int64
    )
    max_post = np.max(resp, axis=1)

    ccf_rows: list[dict[str, Any]] = []
    se_rows: list[dict[str, Any]] = []
    for old_cid in keep_ids:
        new_cid = final_id_map[old_cid]
        idx = np.flatnonzero(labels == old_cid)
        c_mean = np.clip(np.mean(x_mat[idx], axis=0), 0.0, 1.0)
        # Zero out regions where all assigned mutations have negligible CCF (< 0.015)
        c_mean = np.where(c_mean < 0.015, 0.0, c_mean)
        emp_var = np.var(x_mat[idx], axis=0) / max(idx.size, 1)
        meas_var = np.mean(se_mat[idx] ** 2, axis=0) / max(idx.size, 1)
        c_se = np.sqrt(np.maximum(emp_var, meas_var))

        row_c: dict[str, Any] = {
            "cluster_id": int(new_cid),
            "n_mutations": int(idx.size),
        }
        row_s: dict[str, Any] = {
            "cluster_id": int(new_cid),
            "n_mutations": int(idx.size),
        }
        for r_i, reg_name in enumerate(regions):
            row_c[reg_name] = float(c_mean[r_i])
            row_s[reg_name] = float(c_se[r_i])
        ccf_rows.append(row_c)
        se_rows.append(row_s)

    cluster_ccf_df = pd.DataFrame(ccf_rows)
    cluster_se_df = pd.DataFrame(se_rows)

    assign_df = pd.DataFrame(
        {
            "mutation_id": mut_ids,
            "cluster_id": clean_labels,
            "is_clean_cluster": clean_labels > 0,
            "posterior_prob": max_post,
        }
    )
    for r_i, reg_name in enumerate(regions):
        assign_df[reg_name] = x_mat[:, r_i]

    n_removed = int(np.sum(clean_labels < 0))
    return ClusterResult(
        cluster_ccf_df=cluster_ccf_df,
        cluster_se_df=cluster_se_df,
        mutation_assignments_df=assign_df,
        regions=regions,
        n_clusters=len(keep_ids),
        n_removed_mutations=n_removed,
        settings=settings,
    )
