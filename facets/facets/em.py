"""
Gaussian (logR) + Folded-Normal (logOR) Expectation-Maximization (`em_cncf`)
for joint estimation of tumor purity, ploidy, diploid baseline (`dipLogR`),
allele-specific integer copy number (`tcn`, `lcn`), LOH, and clonal cellular
fractions (`cf_em`), with WGD disambiguation and pure-diploid null guarding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .io import validate_settings
from .segment import SegmentedResult

_LN2: float = float(np.log(2.0))
_LOG2PI: float = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class FacetsResult:
    """Complete FACETS allele-specific copy-number, purity, and ploidy result."""

    purity: float | None
    ploidy: float
    dip_logr: float
    log_lik: float
    bic: float
    is_wgd: bool
    is_diploid_null: bool
    sigma_logr: float
    segments: pd.DataFrame
    clusters: pd.DataFrame
    snps: pd.DataFrame | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return segment-level allele-specific copy-number DataFrame."""
        return self.segments.copy()

    def sample_summary_dataframe(self) -> pd.DataFrame:
        """Return a 1-row DataFrame summarizing sample-level FACETS estimates."""
        segs = self.segments
        if len(segs) > 0:
            lengths = np.maximum(
                segs["end"].to_numpy(dtype=np.float64)
                - segs["start"].to_numpy(dtype=np.float64)
                + 1.0,
                1.0,
            )
            tot_len = float(np.sum(lengths))
            frac_loh = float(
                np.sum(lengths[segs["loh"].to_numpy(dtype=bool)]) / tot_len
            )
            frac_cna = float(
                np.sum(
                    lengths[
                        (segs["c_major"].to_numpy(dtype=np.int64) != 1)
                        | (segs["c_minor"].to_numpy(dtype=np.int64) != 1)
                    ]
                )
                / tot_len
            )
        else:
            frac_loh = 0.0
            frac_cna = 0.0

        return pd.DataFrame(
            [
                {
                    "purity": np.nan if self.purity is None else float(self.purity),
                    "ploidy": float(self.ploidy),
                    "dip_logr": float(self.dip_logr),
                    "is_wgd": bool(self.is_wgd),
                    "is_diploid_null": bool(self.is_diploid_null),
                    "n_segments": int(len(segs)),
                    "fraction_loh": frac_loh,
                    "fraction_cna": frac_cna,
                    "log_lik": float(self.log_lik),
                    "bic": float(self.bic),
                }
            ]
        )

    def summary(self) -> str:
        """Return human-readable multi-line report of FACETS fit."""
        purity_str = "NA (diploid null)" if self.purity is None else f"{self.purity:.4f}"
        return (
            f"FACETS FacetsResult\n"
            f"  Purity:          {purity_str}\n"
            f"  Ploidy:          {self.ploidy:.4f}\n"
            f"  dipLogR:         {self.dip_logr:.4f}\n"
            f"  WGD:             {self.is_wgd}\n"
            f"  Diploid Null:    {self.is_diploid_null}\n"
            f"  Segments:        {len(self.segments)}\n"
            f"  Log-Likelihood:  {self.log_lik:.2f}\n"
            f"  BIC:             {self.bic:.2f}"
        )

    def plot(
        self,
        figsize: tuple[float, float] = (11.0, 7.0),
        *,
        axes: Sequence[Axes] | None = None,
        title: str | None = None,
    ) -> Figure:
        """Plot classic 3-panel stacked FACETS genome-wide profile (`facets.plot_facets`)."""
        from .plotting import plot_facets

        return plot_facets(self, figsize=figsize, axes=axes, title=title)


def _logcosh(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(cosh(x)) for arbitrary real arrays."""
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - _LN2


def _build_state_space(max_cn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate ordered integer copy-number states (M, m) with M >= m >= 0, M+m <= max_cn."""
    majors: list[int] = []
    minors: list[int] = []
    for t in range(max_cn + 1):
        for m in range(t // 2 + 1):
            majors.append(t - m)
            minors.append(m)
    m_arr = np.asarray(majors, dtype=np.int64)
    n_arr = np.asarray(minors, dtype=np.int64)
    t_arr = m_arr + n_arr
    return m_arr, n_arr, t_arr


def _expected_mu_r(
    t_arr: np.ndarray | float,
    phi: np.ndarray | float,
    gamma: float,
) -> np.ndarray | float:
    """Expected logR mean under FACETS mixture model."""
    ratio = np.maximum(0.5 * (2.0 * (1.0 - phi) + phi * t_arr), 1e-4)
    return gamma + np.log2(ratio)


def _expected_mu_o(
    c_major: np.ndarray | int,
    c_minor: np.ndarray | int,
    phi: np.ndarray | float,
) -> np.ndarray | float:
    """Expected allelic |logOR| magnitude under FACETS mixture model."""
    eps = 1e-4
    num = 1.0 - phi + phi * c_major + eps
    den = 1.0 - phi + phi * c_minor + eps
    return np.log(num / den)


@dataclass
class _ClusterSuffStats:
    segclust: int
    n_r: int
    y_r_bar: float
    ss_r: float
    n_h: int
    v_het: np.ndarray
    lv_het: np.ndarray
    x2_het: np.ndarray
    inv_lv: np.ndarray
    sum_inv_lv: float
    v_over_lv: np.ndarray
    const_o: float


def _extract_cluster_stats(snps: pd.DataFrame) -> list[_ClusterSuffStats]:
    """Compute sufficient statistics per segment cluster for fast exact EM."""
    clust_ids = np.unique(snps["segclust"].to_numpy(dtype=np.int64))
    clust_arr = snps["segclust"].to_numpy(dtype=np.int64)
    cnlr = snps["cnlr"].to_numpy(dtype=np.float64)
    het = snps["het"].to_numpy(dtype=bool)
    valor = snps["valor"].to_numpy(dtype=np.float64)
    lorvar = snps["lorvar"].to_numpy(dtype=np.float64)

    out: list[_ClusterSuffStats] = []
    for cid in clust_ids:
        mask = clust_arr == cid
        y = cnlr[mask]
        n_r = int(y.size)
        y_bar = float(np.mean(y))
        ss_r = float(np.sum((y - y_bar) ** 2))

        h_mask = mask & het & np.isfinite(valor) & np.isfinite(lorvar) & (lorvar > 0.0)
        v = valor[h_mask]
        lv = lorvar[h_mask]
        n_h = int(v.size)
        if n_h > 0:
            x2_het = v * v - lv
            inv_lv = 1.0 / lv
            sum_inv_lv = float(np.sum(inv_lv))
            v_over_lv = v * inv_lv
            const_o = float(
                np.sum(-0.5 * (_LOG2PI + np.log(lv)) - 0.5 * (v * v) * inv_lv)
            )
        else:
            x2_het = np.empty(0, dtype=np.float64)
            inv_lv = np.empty(0, dtype=np.float64)
            sum_inv_lv = 0.0
            v_over_lv = np.empty(0, dtype=np.float64)
            const_o = 0.0

        out.append(
            _ClusterSuffStats(
                segclust=int(cid),
                n_r=n_r,
                y_r_bar=y_bar,
                ss_r=ss_r,
                n_h=n_h,
                v_het=v,
                lv_het=lv,
                x2_het=x2_het,
                inv_lv=inv_lv,
                sum_inv_lv=sum_inv_lv,
                v_over_lv=v_over_lv,
                const_o=const_o,
            )
        )
    return out


def _cluster_state_loglik(
    st: _ClusterSuffStats,
    c_major: np.ndarray,
    c_minor: np.ndarray,
    t_arr: np.ndarray,
    phi: float,
    gamma: float,
    sigma_r_sq: float,
    *,
    use_surrogate: bool = False,
) -> np.ndarray:
    """Evaluate joint (logR + logOR) log-likelihood across all states."""
    mu_r = _expected_mu_r(t_arr, phi, gamma)
    ll_r = -0.5 * st.n_r * (_LOG2PI + np.log(sigma_r_sq)) - (
        st.ss_r + st.n_r * (st.y_r_bar - mu_r) ** 2
    ) / (2.0 * sigma_r_sq)

    if st.n_h == 0:
        return ll_r

    mu_o = _expected_mu_o(c_major, c_minor, phi)
    if use_surrogate:
        lambda_k = mu_o * mu_o
        diff_mat = st.x2_het[:, None] - lambda_k[None, :]
        surr_w = 1.0 / (np.abs(diff_mat) + 1e-6)
        b_mat = np.sqrt(
            np.maximum(
                st.lv_het[:, None] * (st.lv_het[:, None] + 2.0 * lambda_k[None, :]),
                1e-12,
            )
        )
        ll_o = np.sum(
            -np.log(2.0 * b_mat) - (surr_w * (diff_mat * diff_mat)) / b_mat,
            axis=0,
        )
        return ll_r + ll_o

    ll_o = np.full(c_major.shape, st.const_o, dtype=np.float64)
    nz = mu_o > 1e-12
    if np.any(nz):
        mu_nz = mu_o[nz]
        ll_o[nz] = (
            st.const_o
            - 0.5 * (mu_nz * mu_nz) * st.sum_inv_lv
            + np.sum(_logcosh(st.v_over_lv[:, None] * mu_nz[None, :]), axis=0)
        )
    return ll_r + ll_o


def _optimize_alpha_1d(
    stats: list[_ClusterSuffStats],
    weights: np.ndarray,
    c_major: np.ndarray,
    c_minor: np.ndarray,
    t_arr: np.ndarray,
    is_clonal_mask: np.ndarray,
    diploid_idx: int,
    gamma: float,
    sigma_r_sq: float,
    current_alpha: float,
    *,
    use_surrogate: bool = False,
) -> float:
    """Maximize expected complete-data log-likelihood over clonal purity alpha in [0.05, 1.0]."""
    active_items: list[tuple[_ClusterSuffStats, np.ndarray, np.ndarray]] = []
    for k, st in enumerate(stats):
        if not is_clonal_mask[k]:
            continue
        w_row = weights[k].copy()
        w_row[diploid_idx] = 0.0
        nz = np.flatnonzero(w_row > 1e-6)
        if nz.size > 0 and float(np.sum(w_row[nz])) > 1e-3:
            active_items.append((st, nz, w_row[nz]))

    if not active_items:
        return current_alpha

    if use_surrogate:
        # IRLS moment update holding surrogate weights 1 / (|x2_ij - lambda_k| + 1e-6) fixed
        moment_items: list[
            tuple[_ClusterSuffStats, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for st, nz, w_nz in active_items:
            if st.n_h > 0:
                mu_cur = _expected_mu_o(c_major[nz], c_minor[nz], current_alpha)
                lam_cur = mu_cur * mu_cur
                diff_c = st.x2_het[:, None] - lam_cur[None, :]
                surr_w = 1.0 / (np.abs(diff_c) + 1e-6)
                b_cur = np.sqrt(
                    np.maximum(
                        st.lv_het[:, None] * (st.lv_het[:, None] + 2.0 * lam_cur[None, :]),
                        1e-12,
                    )
                )
                eff_w = surr_w / b_cur
                w_sum = np.sum(eff_w, axis=0)
                x2_bar = np.sum(eff_w * st.x2_het[:, None], axis=0) / np.maximum(
                    w_sum, 1e-12
                )
            else:
                w_sum = np.zeros(nz.size, dtype=np.float64)
                x2_bar = np.zeros(nz.size, dtype=np.float64)
            moment_items.append((st, nz, w_nz, w_sum, x2_bar))

        def _moment_obj(alpha_val: float) -> float:
            loss = 0.0
            for st, nz, w_nz, w_sum, x2_bar in moment_items:
                mu_r = _expected_mu_r(t_arr[nz], alpha_val, gamma)
                l_r = 0.5 * st.n_r * ((st.y_r_bar - mu_r) ** 2) / sigma_r_sq
                if st.n_h > 0:
                    mu_o = _expected_mu_o(c_major[nz], c_minor[nz], alpha_val)
                    lam_a = mu_o * mu_o
                    l_o = w_sum * ((x2_bar - lam_a) ** 2)
                else:
                    l_o = 0.0
                loss += float(np.dot(w_nz, l_r + l_o))
            return loss

        grid_m = np.linspace(0.05, 1.0, 15)
        vals_m = [_moment_obj(float(a)) for a in grid_m]
        b_idx = int(np.argmin(vals_m))
        lo_m = float(max(0.05, grid_m[max(0, b_idx - 1)]))
        hi_m = float(min(1.0, grid_m[min(len(grid_m) - 1, b_idx + 1)]))
        res_m = minimize_scalar(
            _moment_obj, bounds=(lo_m, hi_m), method="bounded", options={"xatol": 2e-4}
        )
        return float(np.clip(res_m.x, 0.05, 1.0))

    def _neg_obj(alpha_val: float) -> float:
        tot = 0.0
        for st, nz, w_nz in active_items:
            ll_vec = _cluster_state_loglik(
                st,
                c_major[nz],
                c_minor[nz],
                t_arr[nz],
                alpha_val,
                gamma,
                sigma_r_sq,
                use_surrogate=use_surrogate,
            )
            tot += float(np.dot(w_nz, ll_vec))
        return -tot

    grid = np.linspace(0.05, 1.0, 15)
    vals = [_neg_obj(float(a)) for a in grid]
    best_idx = int(np.argmin(vals))
    lo = float(max(0.05, grid[max(0, best_idx - 1)]))
    hi = float(min(1.0, grid[min(len(grid) - 1, best_idx + 1)]))
    res = minimize_scalar(
        _neg_obj, bounds=(lo, hi), method="bounded", options={"xatol": 2e-4}
    )
    return float(np.clip(res.x, 0.05, 1.0))


def _optimize_cluster_subclonal(
    st: _ClusterSuffStats,
    c_major: np.ndarray,
    c_minor: np.ndarray,
    t_arr: np.ndarray,
    diploid_idx: int,
    alpha: float,
    gamma: float,
    sigma_r_sq: float,
    *,
    use_surrogate: bool = False,
) -> tuple[int, float, float]:
    """
    Search over non-(1,1) states and phi_k in [0.15, alpha] for a cluster.
    Returns (best_state_idx, best_phi, best_ll).
    """
    if alpha <= 0.19:
        ll_c = _cluster_state_loglik(
            st,
            c_major,
            c_minor,
            t_arr,
            alpha,
            gamma,
            sigma_r_sq,
            use_surrogate=use_surrogate,
        )
        idx = int(np.argmax(ll_c))
        return idx, alpha, float(ll_c[idx])

    phi_grid = np.linspace(0.15, alpha, 15)
    state_pen = np.maximum(0.0, t_arr.astype(np.float64) - 3.0) * 1.5

    # Fast analytical screening using logR + moment-matched mafR to pick top-8 candidate states
    if st.n_h > 0 and st.sum_inv_lv > 0.0:
        obs_mafr = max(
            0.0,
            float(
                np.sum(st.inv_lv * (st.v_het * st.v_het - (1.0 / st.inv_lv)))
                / st.sum_inv_lv
            ),
        )
        obs_mu_o = float(np.sqrt(obs_mafr))
    else:
        obs_mu_o = 0.0

    # Screen across (phi_grid, all_states) with O(1) moment objective
    mu_r_all = _expected_mu_r(t_arr[None, :], phi_grid[:, None], gamma)
    mu_o_all = _expected_mu_o(c_major[None, :], c_minor[None, :], phi_grid[:, None])
    approx_ll = (
        -0.5 * st.n_r * ((st.y_r_bar - mu_r_all) ** 2) / sigma_r_sq
        - 0.5 * st.sum_inv_lv * ((obs_mu_o - mu_o_all) ** 2)
        - state_pen[None, :]
    )
    approx_ll[:, diploid_idx] = -np.inf
    best_per_state = np.max(approx_ll, axis=0)
    cand_states = np.argsort(best_per_state, kind="mergesort")[-8:]

    cM_sub = c_major[cand_states]
    cm_sub = c_minor[cand_states]
    ct_sub = t_arr[cand_states]
    pen_sub = state_pen[cand_states]

    best_score = -np.inf
    best_ll = -np.inf
    best_s = int(cand_states[-1])
    best_phi = alpha

    for phi_val in phi_grid:
        ll_vec = _cluster_state_loglik(
            st,
            cM_sub,
            cm_sub,
            ct_sub,
            float(phi_val),
            gamma,
            sigma_r_sq,
            use_surrogate=use_surrogate,
        )
        score_vec = ll_vec - pen_sub
        s_local = int(np.argmax(score_vec))
        if float(score_vec[s_local]) > best_score:
            best_score = float(score_vec[s_local])
            best_ll = float(ll_vec[s_local])
            best_s = int(cand_states[s_local])
            best_phi = float(phi_val)

    # Local 1D refinement for best_s
    cM = np.array([c_major[best_s]], dtype=np.int64)
    cm = np.array([c_minor[best_s]], dtype=np.int64)
    ct = np.array([t_arr[best_s]], dtype=np.int64)

    def _neg_phi(p: float) -> float:
        return -float(
            _cluster_state_loglik(
                st, cM, cm, ct, p, gamma, sigma_r_sq, use_surrogate=use_surrogate
            )[0]
        )

    lo = max(0.15, best_phi - 0.08)
    hi = min(alpha, best_phi + 0.08)
    if hi - lo > 1e-4:
        opt = minimize_scalar(_neg_phi, bounds=(lo, hi), method="bounded", options={"xatol": 2e-4})
        if -float(opt.fun) > best_ll:
            best_ll = -float(opt.fun)
            best_phi = float(np.clip(opt.x, 0.15, alpha))

    return best_s, best_phi, best_ll


def _cluster_mafr(st: _ClusterSuffStats) -> float:
    """Compute variance-debiased mafR for a cluster."""
    if st.n_h == 0 or st.sum_inv_lv <= 0.0:
        return 0.0
    est = float(
        np.sum(st.inv_lv * (st.v_het * st.v_het - (1.0 / st.inv_lv))) / st.sum_inv_lv
    )
    return max(0.0, est)


def _run_phase_b(
    stats: list[_ClusterSuffStats],
    c_major: np.ndarray,
    c_minor: np.ndarray,
    t_arr: np.ndarray,
    diploid_idx: int,
    tie_breaker: np.ndarray,
    alpha_start: float,
    gamma_start: float,
    sigma_r_sq_start: float,
    pi_start: np.ndarray,
    sigma_r_floor: float,
    gamma_locked: bool,
    *,
    use_surrogate: bool = False,
    max_cycles: int = 15,
) -> dict:
    """Run subclonal CCF refinement + EM updates from a starting (alpha, gamma)."""
    k_clust = len(stats)
    n_states = c_major.size
    n_r_arr = np.array([st.n_r for st in stats], dtype=np.float64)
    y_bar_arr = np.array([st.y_r_bar for st in stats], dtype=np.float64)
    ss_r_arr = np.array([st.ss_r for st in stats], dtype=np.float64)
    n_total = float(np.sum(n_r_arr))
    mafr_arr = np.array([_cluster_mafr(st) for st in stats], dtype=np.float64)

    alpha = float(alpha_start)
    gamma = float(gamma_start)
    sigma_r_sq = float(max(sigma_r_sq_start, sigma_r_floor))
    pi = pi_start.copy()
    phi_vec = np.full(k_clust, alpha, dtype=np.float64)
    is_clonal = np.ones(k_clust, dtype=bool)
    weights = np.full((k_clust, n_states), 1.0 / n_states, dtype=np.float64)

    n_cycles = max_cycles if k_clust >= 2 else 1
    for _ in range(n_cycles):
        prev_alpha_b = alpha
        prev_gamma_b = gamma
        if k_clust >= 2:
            for k, st in enumerate(stats):
                # Keep clear balanced diploid-baseline clusters anchored as clonal (1,1)
                if abs(st.y_r_bar - gamma) < 0.08 and mafr_arr[k] < 0.04:
                    phi_vec[k] = alpha
                    is_clonal[k] = True
                    continue

                ll_clonal_vec = _cluster_state_loglik(
                    st,
                    c_major,
                    c_minor,
                    t_arr,
                    alpha,
                    gamma,
                    sigma_r_sq,
                    use_surrogate=use_surrogate,
                )
                ll_clonal_best = float(np.max(ll_clonal_vec))
                s_sub, phi_sub, ll_sub = _optimize_cluster_subclonal(
                    st,
                    c_major,
                    c_minor,
                    t_arr,
                    diploid_idx,
                    alpha,
                    gamma,
                    sigma_r_sq,
                    use_surrogate=use_surrogate,
                )
                sub_thresh = max(12.0, 2.0 * np.log(st.n_r + st.n_h + 1.0))
                if (phi_sub < alpha - 0.04) and ((ll_sub - ll_clonal_best) > sub_thresh):
                    phi_vec[k] = phi_sub
                    is_clonal[k] = False
                else:
                    phi_vec[k] = alpha
                    is_clonal[k] = True

        log_pi = np.log(np.maximum(pi, 1e-12))
        for k, st in enumerate(stats):
            ll_s = _cluster_state_loglik(
                st,
                c_major,
                c_minor,
                t_arr,
                float(phi_vec[k]),
                gamma,
                sigma_r_sq,
                use_surrogate=use_surrogate,
            )
            lw = log_pi + ll_s + tie_breaker
            m_lw = float(np.max(lw))
            exp_lw = np.exp(lw - m_lw)
            weights[k] = exp_lw / float(np.sum(exp_lw))

        assigned_map = np.argmax(weights, axis=1)
        aberrant_idx = [k for k in range(k_clust) if assigned_map[k] != diploid_idx]
        if aberrant_idx and not any(is_clonal[k] for k in aberrant_idx):
            anchor_k = max(aberrant_idx, key=lambda k: (phi_vec[k], stats[k].n_r))
            alpha = float(phi_vec[anchor_k])
            is_clonal[anchor_k] = True

        phi_vec[is_clonal] = alpha
        pi = (np.mean(weights, axis=0) + 1e-3 / n_states)
        pi /= np.sum(pi)

        if not gamma_locked:
            mu_shift_mat = np.log2(
                np.maximum(
                    0.5 * (2.0 * (1.0 - phi_vec[:, None]) + phi_vec[:, None] * t_arr[None, :]),
                    1e-4,
                )
            )
            num_g = np.sum(weights * n_r_arr[:, None] * (y_bar_arr[:, None] - mu_shift_mat))
            gamma = float(num_g / max(n_total, 1.0))

        mu_r_mat = _expected_mu_r(t_arr[None, :], phi_vec[:, None], gamma)
        num_s = np.sum(
            weights
            * (ss_r_arr[:, None] + n_r_arr[:, None] * (y_bar_arr[:, None] - mu_r_mat) ** 2)
        )
        sigma_r_sq = float(max(num_s / max(n_total, 1.0), sigma_r_floor))

        alpha = _optimize_alpha_1d(
            stats,
            weights,
            c_major,
            c_minor,
            t_arr,
            is_clonal,
            diploid_idx,
            gamma,
            sigma_r_sq,
            alpha,
            use_surrogate=use_surrogate,
        )
        phi_vec[is_clonal] = alpha
        phi_vec = np.minimum(phi_vec, alpha)

        if abs(alpha - prev_alpha_b) < 1e-4 and abs(gamma - prev_gamma_b) < 1e-4:
            break

    log_pi = np.log(np.maximum(pi, 1e-12))
    map_states = np.empty(k_clust, dtype=np.int64)
    map_probs = np.empty(k_clust, dtype=np.float64)
    map_ll = 0.0
    for k, st in enumerate(stats):
        ll_s = _cluster_state_loglik(
            st,
            c_major,
            c_minor,
            t_arr,
            float(phi_vec[k]),
            gamma,
            sigma_r_sq,
            use_surrogate=use_surrogate,
        )
        lw = log_pi + ll_s + tie_breaker
        m_lw = float(np.max(lw))
        exp_lw = np.exp(lw - m_lw)
        w_k = exp_lw / float(np.sum(exp_lw))
        s_star = int(np.argmax(w_k))
        map_states[k] = s_star
        map_probs[k] = float(w_k[s_star])
        map_ll += float(ll_s[s_star])
        if s_star == diploid_idx:
            phi_vec[k] = alpha
            is_clonal[k] = True

    return {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "sigma_r_sq": float(sigma_r_sq),
        "map_states": map_states,
        "map_probs": map_probs,
        "phi_vec": phi_vec.copy(),
        "is_subclonal": ~is_clonal,
        "log_lik": float(map_ll),
    }


def _run_em_from_seed(
    stats: list[_ClusterSuffStats],
    c_major: np.ndarray,
    c_minor: np.ndarray,
    t_arr: np.ndarray,
    diploid_idx: int,
    tie_breaker: np.ndarray,
    alpha_init: float,
    gamma_init: float,
    sigma_r_sq_init: float,
    sigma_r_floor: float,
    gamma_locked: bool,
    max_iter: int,
    tol: float,
    *,
    use_surrogate: bool = False,
) -> dict:
    """Run clonal EM followed by subclonal CCF refinement from one (alpha, gamma) seed."""
    k_clust = len(stats)
    n_states = c_major.size
    n_r_arr = np.array([st.n_r for st in stats], dtype=np.float64)
    y_bar_arr = np.array([st.y_r_bar for st in stats], dtype=np.float64)
    ss_r_arr = np.array([st.ss_r for st in stats], dtype=np.float64)
    n_total = float(np.sum(n_r_arr))

    alpha = float(alpha_init)
    gamma = float(gamma_init)
    sigma_r_sq = float(max(sigma_r_sq_init, sigma_r_floor))
    pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    is_clonal = np.ones(k_clust, dtype=bool)
    weights = np.full((k_clust, n_states), 1.0 / n_states, dtype=np.float64)

    prev_ll = -np.inf

    # Phase A: Clonal EM
    for _ in range(max_iter):
        log_pi = np.log(np.maximum(pi, 1e-12))
        total_ll = 0.0
        for k, st in enumerate(stats):
            ll_s = _cluster_state_loglik(
                st,
                c_major,
                c_minor,
                t_arr,
                alpha,
                gamma,
                sigma_r_sq,
                use_surrogate=use_surrogate,
            )
            lw = log_pi + ll_s + tie_breaker
            m_lw = float(np.max(lw))
            exp_lw = np.exp(lw - m_lw)
            sum_exp = float(np.sum(exp_lw))
            weights[k] = exp_lw / sum_exp
            total_ll += m_lw + float(np.log(sum_exp))

        pi = (np.mean(weights, axis=0) + 1e-3 / n_states)
        pi /= np.sum(pi)

        if not gamma_locked:
            mu_shift = np.log2(
                np.maximum(0.5 * (2.0 * (1.0 - alpha) + alpha * t_arr), 1e-4)
            )
            num_g = np.sum(weights * n_r_arr[:, None] * (y_bar_arr[:, None] - mu_shift[None, :]))
            gamma = float(num_g / max(n_total, 1.0))

        mu_r_mat = _expected_mu_r(t_arr[None, :], alpha, gamma)
        num_s = np.sum(
            weights
            * (ss_r_arr[:, None] + n_r_arr[:, None] * (y_bar_arr[:, None] - mu_r_mat) ** 2)
        )
        sigma_r_sq = float(max(num_s / max(n_total, 1.0), sigma_r_floor))

        new_alpha = _optimize_alpha_1d(
            stats,
            weights,
            c_major,
            c_minor,
            t_arr,
            is_clonal,
            diploid_idx,
            gamma,
            sigma_r_sq,
            alpha,
            use_surrogate=use_surrogate,
        )

        if abs(total_ll - prev_ll) < tol and abs(new_alpha - alpha) < 1e-4:
            alpha = new_alpha
            break
        alpha = new_alpha
        prev_ll = total_ll

    eff_cycles = min(max_iter, 10) if use_surrogate else 15

    # Phase B from converged Phase A, and also directly from initial seed to guard against subclonal pull
    cand_a = _run_phase_b(
        stats,
        c_major,
        c_minor,
        t_arr,
        diploid_idx,
        tie_breaker,
        alpha,
        gamma,
        sigma_r_sq,
        pi,
        sigma_r_floor,
        gamma_locked,
        use_surrogate=use_surrogate,
        max_cycles=eff_cycles,
    )
    if k_clust >= 2:
        uniform_pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        cand_init = _run_phase_b(
            stats,
            c_major,
            c_minor,
            t_arr,
            diploid_idx,
            tie_breaker,
            alpha_init,
            gamma_init,
            sigma_r_sq_init,
            uniform_pi,
            sigma_r_floor,
            gamma_locked,
            use_surrogate=use_surrogate,
            max_cycles=eff_cycles,
        )
        pen_a = cand_a["log_lik"] - 5.0 * float(np.sum(cand_a["is_subclonal"]))
        pen_i = cand_init["log_lik"] - 5.0 * float(np.sum(cand_init["is_subclonal"]))
        if pen_i > pen_a:
            return cand_init
    return cand_a


def em_cncf(
    segmented: SegmentedResult,
    *,
    max_cn: int = 10,
    min_nhet: int | None = None,
    diploid_logr: float | None = None,
    keep_low_nhet_lcn: bool = False,
    max_iter: int | None = None,
    tol: float = 1e-5,
    settings: Literal["default", "R_default"] = "default",
) -> FacetsResult:
    """
    Fit mixture EM over segmented clusters to estimate tumor purity, ploidy,
    dipLogR, integer (C_major, C_minor), LOH, and clonal cellular fractions (`cf_em`).

    - `settings="default"`: Gaussian (logR) + exact Folded-Normal (logOR) EM (`max_iter=100`).
    - `settings="R_default"`: `emcncf` allele log-likelihood surrogate weight
      `1 / (abs(x2_ij - lambda_k) + 1e-6)` with `max_iter=10` moment updates.
    """
    validate_settings(settings)
    use_surrogate = settings == "R_default"
    eff_max_iter = (
        int(max_iter) if max_iter is not None else (10 if use_surrogate else 100)
    )
    eff_min_nhet = int(min_nhet if min_nhet is not None else segmented.min_nhet)
    segs = segmented.segments.copy()
    snps = segmented.jointseg.copy()
    if len(segs) == 0 or len(snps) == 0:
        raise ValueError("Cannot run EM on empty segmentation result.")

    gamma_locked = diploid_logr is not None
    locked_dip = float(diploid_logr) if diploid_logr is not None else None

    c_major, c_minor, t_arr = _build_state_space(max_cn)
    diploid_idx = int(np.flatnonzero((c_major == 1) & (c_minor == 1))[0])
    # Tiny deterministic preference for balanced / lower-CN states when nhet == 0
    tie_breaker = -1e-4 * (c_major - c_minor).astype(np.float64) - 1e-5 * t_arr.astype(
        np.float64
    )

    stats = _extract_cluster_stats(snps)
    k_clust = len(stats)
    n_total = sum(st.n_r for st in stats)

    # Initial within-cluster variance
    within_var = sum(st.ss_r for st in stats) / max(n_total, 1)
    sigma_r_floor = max(1e-4, 0.5 * (segmented.sigma_logr ** 2))
    sigma_r_init = max(within_var, segmented.sigma_logr ** 2, 1e-3)

    # Pure-diploid null model (all clusters = (1,1))
    all_cnlr = snps["cnlr"].to_numpy(dtype=np.float64)
    gamma_null = locked_dip if locked_dip is not None else float(np.mean(all_cnlr))
    sigma_null_sq = max(float(np.mean((all_cnlr - gamma_null) ** 2)), sigma_r_floor)
    ll_null = sum(
        float(
            _cluster_state_loglik(
                st,
                np.array([1], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
                0.0,
                gamma_null,
                sigma_null_sq,
                use_surrogate=use_surrogate,
            )[0]
        )
        for st in stats
    )
    bic_null = 2.0 * ll_null - 2.0 * float(np.log(max(n_total, 2)))

    # Multi-start candidate grid over (alpha, gamma)
    alpha_grid = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    if locked_dip is not None:
        seed_pairs: list[tuple[float, float, bool]] = [
            (a0, locked_dip, False) for a0 in alpha_grid
        ]
    else:
        cand_dips = list(segmented.candidate_dip_logrs)
        if segmented.dip_logr not in cand_dips:
            cand_dips.insert(0, segmented.dip_logr)
        seed_pairs = []
        for dip_c in cand_dips:
            for a0 in alpha_grid:
                seed_pairs.append((a0, float(dip_c), False))
                # WGD hypothesis: dominant balanced cluster is (2,2) rather than (1,1)
                seed_pairs.append((a0, float(dip_c - np.log2(1.0 + a0)), True))

    # Fast screening of all seed pairs
    scored_non_wgd: list[tuple[float, float, float]] = []
    scored_wgd: list[tuple[float, float, float]] = []
    for a0, g0, is_wgd_seed in seed_pairs:
        ll_seed = 0.0
        n_r_sum = 0.0
        t_weighted = 0.0
        for st in stats:
            ll_vec = _cluster_state_loglik(
                st,
                c_major,
                c_minor,
                t_arr,
                a0,
                g0,
                sigma_r_init,
                use_surrogate=use_surrogate,
            )
            best_s = int(np.argmax(ll_vec + tie_breaker))
            ll_seed += float(ll_vec[best_s])
            n_r_sum += st.n_r
            t_weighted += st.n_r * float(t_arr[best_s])
        mean_t = t_weighted / max(n_r_sum, 1.0)
        if is_wgd_seed or mean_t >= 2.8:
            scored_wgd.append((ll_seed, a0, g0))
        else:
            scored_non_wgd.append((ll_seed, a0, g0))

    scored_non_wgd.sort(key=lambda x: -x[0])
    scored_wgd.sort(key=lambda x: -x[0])

    def _pick_distinct(cands: list[tuple[float, float, float]], k_max: int) -> list[tuple[float, float]]:
        chosen: list[tuple[float, float]] = []
        for _, a0, g0 in cands:
            if not any(abs(a0 - ca) < 0.05 and abs(g0 - cg) < 0.05 for ca, cg in chosen):
                chosen.append((a0, g0))
            if len(chosen) >= k_max:
                break
        return chosen

    selected_seeds = _pick_distinct(scored_non_wgd, 3) + _pick_distinct(scored_wgd, 3)
    if not selected_seeds:
        selected_seeds = [(0.5, segmented.dip_logr)]

    # Precompute segment bp lengths mapped by segclust for ploidy evaluation
    seg_lengths = np.maximum(
        segs["end"].to_numpy(dtype=np.float64)
        - segs["start"].to_numpy(dtype=np.float64)
        + 1.0,
        1.0,
    )
    seg_clust_col = segs["segclust"].to_numpy(dtype=np.int64)
    clust_id_to_pos = {st.segclust: idx for idx, st in enumerate(stats)}
    seg_clust_pos = np.array([clust_id_to_pos[int(c)] for c in seg_clust_col], dtype=np.int64)
    total_bp = float(np.sum(seg_lengths))

    best_model: dict | None = None
    best_bic_score = -np.inf

    for a0, g0 in selected_seeds:
        fit = _run_em_from_seed(
            stats,
            c_major,
            c_minor,
            t_arr,
            diploid_idx,
            tie_breaker,
            a0,
            g0,
            sigma_r_init,
            sigma_r_floor,
            gamma_locked,
            eff_max_iter,
            tol,
            use_surrogate=use_surrogate,
        )
        seg_t = t_arr[fit["map_states"][seg_clust_pos]].astype(np.float64)
        ploidy_cand = float(np.sum(seg_lengths * seg_t) / max(total_bp, 1.0))

        n_aberr = int(np.sum(fit["map_states"] != diploid_idx))
        n_sub = int(np.sum(fit["is_subclonal"]))
        k_params = 3 + n_aberr + 3 * n_sub
        bic_raw = 2.0 * fit["log_lik"] - k_params * float(np.log(max(n_total, 2)))
        # Gentle ploidy parsimony tie-breaker: resolves clonal WGD/diploid equivalences
        # toward diploid without letting artificial subclonal states override clonal WGD.
        parsimony_pen = 3.0 * max(0.0, ploidy_cand - 2.2)
        bic_score = bic_raw - parsimony_pen

        fit["ploidy"] = ploidy_cand
        fit["bic_raw"] = bic_raw
        fit["bic_score"] = bic_score

        if bic_score > best_bic_score:
            best_bic_score = bic_score
            best_model = fit

    assert best_model is not None

    # Check pure-diploid null condition (preserve high-BIC focal amplifications < 1% genome)
    seg_states = best_model["map_states"][seg_clust_pos]
    aberr_bp = float(np.sum(seg_lengths[seg_states != diploid_idx]))
    frac_aberr = aberr_bp / max(total_bp, 1.0)
    is_diploid_null = (
        (frac_aberr == 0.0)
        or (best_model["bic_raw"] <= bic_null + 10.0)
        or (frac_aberr < 0.01 and best_model["bic_raw"] <= bic_null + 40.0)
    )

    if is_diploid_null:
        purity_out: float | None = None
        ploidy_out = 2.0
        dip_logr_out = gamma_null
        log_lik_out = ll_null
        bic_out = bic_null
        is_wgd_out = False
        sigma_out = float(np.sqrt(sigma_null_sq))

        seg_c_major = np.ones(len(segs), dtype=np.int64)
        seg_c_minor = np.ones(len(segs), dtype=np.int64)
        seg_tcn = np.full(len(segs), 2, dtype=np.int64)
        seg_cf = np.ones(len(segs), dtype=np.float64)
        seg_prob = np.ones(len(segs), dtype=np.float64)
        seg_subclonal = np.zeros(len(segs), dtype=bool)

        cl_c_major = np.ones(k_clust, dtype=np.int64)
        cl_c_minor = np.ones(k_clust, dtype=np.int64)
        cl_tcn = np.full(k_clust, 2, dtype=np.int64)
        cl_cf = np.ones(k_clust, dtype=np.float64)
        cl_prob = np.ones(k_clust, dtype=np.float64)
    else:
        purity_out = float(best_model["alpha"])
        ploidy_out = float(best_model["ploidy"])
        dip_logr_out = float(best_model["gamma"])
        log_lik_out = float(best_model["log_lik"])
        bic_out = float(best_model["bic_raw"])
        sigma_out = float(np.sqrt(best_model["sigma_r_sq"]))

        cl_states = best_model["map_states"]
        cl_c_major = c_major[cl_states]
        cl_c_minor = c_minor[cl_states]
        cl_tcn = t_arr[cl_states]
        cl_cf = np.where(cl_states == diploid_idx, 1.0, best_model["phi_vec"])
        cl_prob = best_model["map_probs"]

        seg_c_major = cl_c_major[seg_clust_pos]
        seg_c_minor = cl_c_minor[seg_clust_pos]
        seg_tcn = cl_tcn[seg_clust_pos]
        seg_cf = cl_cf[seg_clust_pos]
        seg_prob = cl_prob[seg_clust_pos]
        seg_subclonal = best_model["is_subclonal"][seg_clust_pos] & (
            seg_states != diploid_idx
        )

        bp_cminor_ge2 = float(np.sum(seg_lengths[seg_c_minor >= 2])) / max(total_bp, 1.0)
        is_wgd_out = bool((ploidy_out >= 2.8) or (bp_cminor_ge2 >= 0.45))

    low_nhet_seg = segs["nhet"].to_numpy(dtype=np.int64) < eff_min_nhet
    segs["low_nhet"] = low_nhet_seg
    segs["tcn"] = seg_tcn
    segs["lcn_raw"] = seg_c_minor
    segs["lcn"] = (
        seg_c_minor.astype(np.float64)
        if keep_low_nhet_lcn
        else np.where(low_nhet_seg, np.nan, seg_c_minor.astype(np.float64))
    )
    segs["c_major"] = seg_c_major
    segs["c_minor"] = seg_c_minor
    segs["cf_em"] = seg_cf
    segs["loh"] = (seg_c_minor == 0) & (seg_tcn >= 1)
    segs["cn_loh"] = (seg_c_minor == 0) & (seg_tcn >= 2)
    segs["posterior_prob"] = seg_prob
    segs["is_subclonal"] = seg_subclonal

    seg_col_order = [
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
        "mafR",
        "tcn",
        "lcn",
        "lcn_raw",
        "c_major",
        "c_minor",
        "cf_em",
        "loh",
        "cn_loh",
        "posterior_prob",
        "is_subclonal",
        "low_nhet",
    ]
    segs = segs[seg_col_order]

    # Build cluster summary table
    cluster_rows: list[dict] = []
    for k, st in enumerate(stats):
        c_mafr = (
            max(
                0.0,
                float(
                    np.sum(st.inv_lv * (st.v_het * st.v_het - (1.0 / st.inv_lv)))
                    / st.sum_inv_lv
                ),
            )
            if st.n_h > 0
            else 0.0
        )
        c_low_nhet = st.n_h < eff_min_nhet
        cluster_rows.append(
            {
                "segclust": st.segclust,
                "num_mark": st.n_r,
                "nhet": st.n_h,
                "cnlr_mean": st.y_r_bar,
                "mafR": c_mafr,
                "tcn": int(cl_tcn[k]),
                "lcn": (
                    float(cl_c_minor[k])
                    if (keep_low_nhet_lcn or not c_low_nhet)
                    else np.nan
                ),
                "cf_em": float(cl_cf[k]),
                "posterior_prob": float(cl_prob[k]),
            }
        )
    clusters_df = pd.DataFrame(cluster_rows)

    # Map segment calls back to SNPs
    seg_idx_to_row = {
        int(row_s): idx for idx, row_s in enumerate(segs["seg_idx"].to_numpy(dtype=np.int64))
    }
    snp_pos = np.array(
        [seg_idx_to_row[int(s)] for s in snps["seg_idx"].to_numpy(dtype=np.int64)],
        dtype=np.int64,
    )
    snps["tcn"] = seg_tcn[snp_pos]
    snps["lcn"] = segs["lcn"].to_numpy(dtype=np.float64)[snp_pos]

    return FacetsResult(
        purity=purity_out,
        ploidy=ploidy_out,
        dip_logr=dip_logr_out,
        log_lik=log_lik_out,
        bic=bic_out,
        is_wgd=is_wgd_out,
        is_diploid_null=is_diploid_null,
        sigma_logr=sigma_out,
        segments=segs,
        clusters=clusters_df,
        snps=snps,
    )
