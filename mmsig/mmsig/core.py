"""
Core mutational signature refitting, stepwise backward cosine-similarity pruning,
parametric multinomial bootstrap CIs, and R-parity confirmation rules for `mmsig`
(Rustad et al., Communications Biology 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from .reference import (
    CHEMO_STRAND_SIGNATURES,
    SettingsMode,
    build_96_catalog,
    prepare_signature_reference,
    validate_settings,
)
from .strand import check_strand_bias


def cosine_similarity_vec(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two non-negative vectors."""
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    val = float(np.dot(x, y) / (nx * ny))
    return float(np.clip(val, 0.0, 1.0))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto the unit probability simplex (Duchi et al. 2008)."""
    k = v.shape[0]
    if k == 1:
        return np.ones(1, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, k + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.full(k, 1.0 / k, dtype=float)
    return w / s


def fit_simplex_ls(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve simplex-constrained least squares matching `pracma::lsqlincon`:

        min_w  0.5 * w^T G w - b^T w    s.t. sum(w) = 1, w >= 0

    where `G = S^T S` and `b = S^T y`. Uses restarted Nesterov FISTA on the simplex
    with exact KKT optimality verification and SLSQP polish fallback.
    """
    k = b.shape[0]
    if k == 1:
        return np.ones(1, dtype=float)

    L = float(np.linalg.norm(G, ord="fro")) + 1e-12
    step = 1.0 / L

    w = _project_simplex(b / np.maximum(np.diag(G), 1e-12))
    yk = w.copy()
    t = 1.0

    for _ in range(120):
        grad = G @ yk - b
        w_next = _project_simplex(yk - step * grad)
        diff = w_next - w
        if float(np.max(np.abs(diff))) < 1e-11:
            w = w_next
            break
        # Adaptive gradient restart (O'Donoghue & Candes 2015)
        if float(np.dot(yk - w_next, w_next - w)) > 0.0:
            t_next = 1.0
            yk = w_next
        else:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            yk = w_next + ((t - 1.0) / t_next) * diff
        w = w_next
        t = t_next

    # Check exact KKT gap on the simplex: min_{z in Delta} grad^T (z - w) = min(grad) - grad^T w
    grad_final = G @ w - b
    dual_gap = float(np.dot(grad_final, w) - np.min(grad_final))
    if dual_gap > 1e-8:
        res = minimize(
            fun=lambda x: 0.5 * float(x @ G @ x) - float(b @ x),
            x0=w,
            jac=lambda x: G @ x - b,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0), "jac": lambda x: np.ones(k)},
            options={"ftol": 1e-12, "maxiter": 200},
        )
        w = _project_simplex(res.x)

    w[w < 1e-12] = 0.0
    total = float(w.sum())
    return w / total if total > 0.0 else np.full(k, 1.0 / k, dtype=float)


def fit_multinomial_em_nnls(
    S_sub: np.ndarray,
    y: np.ndarray,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Hybrid NNLS warm-start + Multinomial EM solver on the simplex (`settings='default'`).

    Initializes exposure weights from normalized NNLS (`min ||S w - y||_2, w >= 0`),
    then refines with convex blend toward Multinomial EM likelihood updates while
    preventing degradation of reconstructed cosine similarity.
    """
    k = S_sub.shape[1]
    if k == 1:
        return np.ones(1, dtype=float)

    w_nnls, _ = nnls(S_sub, y)
    s0 = float(w_nnls.sum())
    if s0 > 0.0:
        w = w_nnls / s0
    else:
        w = np.full(k, 1.0 / k, dtype=float)

    pos = y > 0.0
    if not np.any(pos):
        return w

    S_pos = S_sub[pos]
    y_pos = y[pos]
    y_pos = y_pos / float(y_pos.sum())

    w_em = w.copy()
    for _ in range(max_iter):
        recon = S_pos @ w_em
        recon = np.maximum(recon, 1e-15)
        ratios = y_pos / recon
        w_next = w_em * (ratios @ S_pos)
        tot = float(w_next.sum())
        if tot <= 0.0:
            break
        w_next /= tot
        if float(np.max(np.abs(w_next - w_em))) < tol:
            w_em = w_next
            break
        w_em = w_next

    # Hybrid convex combination balancing multinomial likelihood and L2 cosine fidelity
    w_hyb = 0.5 * w + 0.5 * w_em
    cos_nnls = cosine_similarity_vec(y, S_sub @ w)
    cos_hyb = cosine_similarity_vec(y, S_sub @ w_hyb)
    out = w_hyb if cos_hyb >= cos_nnls - 1e-4 else w
    out[out < 1e-12] = 0.0
    s_out = float(out.sum())
    return out / s_out if s_out > 0.0 else np.full(k, 1.0 / k, dtype=float)


def _fit_subset_weights(
    S_full: np.ndarray,
    G_full: np.ndarray,
    y: np.ndarray,
    b_full: np.ndarray,
    norm_y: float,
    active_idx: list[int],
    settings: SettingsMode,
    em_max_iter: int = 80,
    em_tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Fit weights and reconstructed cosine similarity on `active_idx`."""
    idx_arr = np.asarray(active_idx, dtype=int)
    if settings == "R_default":
        G_sub = G_full[np.ix_(idx_arr, idx_arr)]
        b_sub = b_full[idx_arr]
        w = fit_simplex_ls(G_sub, b_sub)
        denom_sq = float(w @ G_sub @ w)
        if norm_y <= 0.0 or denom_sq <= 0.0:
            cos = 0.0
        else:
            cos = float(np.clip(float(b_sub @ w) / (norm_y * np.sqrt(denom_sq)), 0.0, 1.0))
        return w, cos
    else:
        S_sub = S_full[:, idx_arr]
        w = fit_multinomial_em_nnls(S_sub, y, max_iter=em_max_iter, tol=em_tol)
        cos = cosine_similarity_vec(y, S_sub @ w)
        return w, cos


def prune_signatures_single(
    S_full: np.ndarray,
    G_full: np.ndarray,
    counts_or_prop: np.ndarray,
    sig_names: list[str],
    forced_mask: np.ndarray,
    cos_sim_threshold: float,
    settings: SettingsMode,
    record_trace: bool = False,
    is_bootstrap: bool = False,
) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    """
    Fit exposures and perform stepwise backward cosine-similarity pruning on one profile.
    """
    K = len(sig_names)
    total = float(counts_or_prop.sum())
    if total <= 0.0:
        return np.zeros(K, dtype=float), 0.0, []

    y = counts_or_prop / total
    norm_y = float(np.linalg.norm(y))
    b_full = S_full.T @ y

    em_iters = 45 if is_bootstrap else 80
    em_tol = 1e-6 if is_bootstrap else 1e-8

    active: list[int] = list(range(K))
    w_curr, cos_curr = _fit_subset_weights(
        S_full, G_full, y, b_full, norm_y, active, settings, em_iters, em_tol
    )

    trace: list[dict[str, object]] = []
    while len(active) > 1:
        removable_positions = [pos for pos, sig_i in enumerate(active) if not forced_mask[sig_i]]
        if not removable_positions:
            break

        best_drop = float("inf")
        best_pos = -1
        best_w: np.ndarray | None = None
        best_cos = cos_curr

        for pos in removable_positions:
            cand_active = active[:pos] + active[pos + 1 :]
            w_cand, cos_cand = _fit_subset_weights(
                S_full, G_full, y, b_full, norm_y, cand_active, settings, em_iters, em_tol
            )
            drop = cos_curr - cos_cand
            if drop < best_drop:
                best_drop = drop
                best_pos = pos
                best_w = w_cand
                best_cos = cos_cand

        if best_pos >= 0 and best_drop < cos_sim_threshold:
            removed_sig_idx = active[best_pos]
            if record_trace:
                trace.append(
                    {
                        "removed_signature": sig_names[removed_sig_idx],
                        "cos_before": cos_curr,
                        "cos_after": best_cos,
                        "cos_drop": best_drop,
                        "remaining_count": len(active) - 1,
                    }
                )
            active = active[:best_pos] + active[best_pos + 1 :]
            assert best_w is not None
            w_curr = best_w
            cos_curr = best_cos
        else:
            break

    full_w = np.zeros(K, dtype=float)
    for pos, sig_i in enumerate(active):
        full_w[sig_i] = w_curr[pos]
    s_w = float(full_w.sum())
    if s_w > 0.0:
        full_w /= s_w
    return full_w, cos_curr, trace


@dataclass
class MMSigResult:
    """Structured output from `mm_fit_signatures` / `MMSig.fit`."""

    exposures: pd.DataFrame
    exposures_counts: pd.DataFrame
    gof_df: pd.DataFrame
    bootstrap_df: pd.DataFrame | None = None
    strand_bias_df: pd.DataFrame | None = None
    pruning_trace: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    settings: SettingsMode = "default"

    def to_dataframe(self) -> pd.DataFrame:
        """Return sample x signature relative exposure DataFrame merged with cosine similarity."""
        out = self.exposures.copy()
        out["cosine_similarity"] = self.gof_df.set_index("sample").loc[
            out.index, "cosine_similarity"
        ]
        out["total_mutations"] = self.gof_df.set_index("sample").loc[
            out.index, "total_mutations"
        ]
        return out

    def summary(self) -> str:
        """Return concise text summary of fitted mutational signatures."""
        n_samp, n_sig = self.exposures.shape
        mean_cos = float(self.gof_df["cosine_similarity"].mean()) if n_samp > 0 else 0.0
        active_counts = (self.exposures > 0.0).sum(axis=1)
        mean_act = float(active_counts.mean()) if n_samp > 0 else 0.0
        return (
            f"MMSigResult(samples={n_samp}, candidate_signatures={n_sig}, "
            f"mean_active={mean_act:.2f}, mean_cos_sim={mean_cos:.4f}, "
            f"bootstrap={self.bootstrap_df is not None}, settings={self.settings!r})"
        )


def mm_fit_signatures(
    muts_df: pd.DataFrame,
    sig_ref_df: pd.DataFrame,
    sigs_cons: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    cos_sim_threshold: float = 0.01,
    force_include: Sequence[str] | None = ("SBS1", "SBS5"),
    bootstrap: bool = True,
    n_boot: int | None = None,
    strand_bias_df: pd.DataFrame | None = None,
    dbg: bool = False,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> MMSigResult:
    """
    Fit mutational signatures with stepwise backward cosine-similarity pruning,
    parametric multinomial bootstrap CIs, and transcriptional strand-bias confirmation.

    Parameters
    ----------
    muts_df : pd.DataFrame
        Either a 96-channel x sample catalog DataFrame or a variant-level mutation
        table with `sample` and `context` / (`ref`, `alt`, `trinuc`).
    sig_ref_df : pd.DataFrame
        96-channel x K signature reference DataFrame.
    sigs_cons : sequence of str or mapping of sample -> sequence of str, optional
        Subset of signatures from `sig_ref_df` to consider globally or per sample.
    cos_sim_threshold : float, default 0.01
        Backward pruning cosine similarity decrease threshold. Non-forced signatures
        whose removal decreases reconstructed cosine similarity by `< cos_sim_threshold`
        are iteratively pruned.
    force_include : sequence of str or None, default ("SBS1", "SBS5")
        Ubiquitous clock-like signatures exempt from backward pruning when present
        among candidate signatures.
    bootstrap : bool, default True
        Whether to run parametric multinomial bootstrap refitting (with full backward
        pruning on every resampled catalog).
    n_boot : int, optional
        Number of bootstrap replicates (`500` when `settings='default'`, `1000` when
        `settings='R_default'` if `None`).
    strand_bias_df : pd.DataFrame, optional
        Transcriptional strand counts or variant annotations used to confirm
        chemotherapy signatures (`SBS31`, `SBS35`, `SBS99`).
    dbg : bool, default False
        Record detailed step-by-step pruning trace per sample in `result.pruning_trace`.
    seed : int, default 42
        Random seed for multinomial bootstrap reproducibility.
    settings : {"default", "R_default"}, default "default"
        `"default"` uses NNLS + Multinomial EM hybrid refitting and `n_boot=500`.
        `"R_default"` uses simplex-constrained least squares (`pracma::lsqlincon` parity),
        `n_boot=1000`, `SBS1`/`SBS5` forcing, and `SBS8`/`SBS99` presence confirmation
        requiring lower 95% bootstrap CI > 0 (plus strand bias `p < 0.05` when
        `strand_bias_df` is supplied for `SBS35`/`SBS99`).
    """
    validate_settings(settings)
    if cos_sim_threshold < 0.0 or cos_sim_threshold >= 1.0:
        raise ValueError(f"cos_sim_threshold must be in [0, 1); got {cos_sim_threshold}.")

    effective_n_boot = n_boot if n_boot is not None else (1000 if settings == "R_default" else 500)
    if bootstrap and effective_n_boot < 2:
        raise ValueError(f"n_boot must be >= 2 when bootstrap=True; got {effective_n_boot}.")

    if force_include is None:
        forced_set = {"SBS1", "SBS5"} if settings == "R_default" else set()
    else:
        forced_set = {str(s) for s in force_include}

    ref_df = prepare_signature_reference(sig_ref_df)
    cat_df = build_96_catalog(muts_df, reference_index=ref_df.index, settings=settings)

    all_ref_sigs = list(ref_df.columns)
    sample_names = list(cat_df.columns)

    # Resolve candidate signatures per sample
    sample_sig_lists: dict[str, list[str]] = {}
    all_used_sigs_order: list[str] = []
    if sigs_cons is None:
        for s in sample_names:
            sample_sig_lists[s] = list(all_ref_sigs)
        all_used_sigs_order = list(all_ref_sigs)
    elif isinstance(sigs_cons, Mapping):
        seen: set[str] = set()
        for s in sample_names:
            cand = [str(sig) for sig in sigs_cons.get(s, all_ref_sigs)]
            missing = [sig for sig in cand if sig not in ref_df.columns]
            if missing:
                raise ValueError(f"Unknown signatures in sigs_cons for sample {s!r}: {missing}")
            if not cand:
                raise ValueError(f"Empty candidate signature list for sample {s!r}.")
            sample_sig_lists[s] = cand
            for sig in cand:
                if sig not in seen:
                    seen.add(sig)
                    all_used_sigs_order.append(sig)
    else:
        cand = [str(sig) for sig in sigs_cons]
        missing = [sig for sig in cand if sig not in ref_df.columns]
        if missing:
            raise ValueError(f"Unknown signatures in sigs_cons: {missing}")
        if not cand:
            raise ValueError("sigs_cons cannot be empty.")
        for s in sample_names:
            sample_sig_lists[s] = list(cand)
        all_used_sigs_order = list(cand)

    # Evaluate strand bias if provided
    sb_res: pd.DataFrame | None = None
    sb_lookup: dict[tuple[str, str], bool] = {}
    if strand_bias_df is not None:
        sb_res = check_strand_bias(strand_bias_df, alpha=0.05, settings=settings)
        for _, r in sb_res.iterrows():
            sb_lookup[(str(r["sample"]), str(r["signature"]))] = bool(r["significant"])

    rng = np.random.default_rng(seed)
    exposures_mat = np.zeros((len(sample_names), len(all_used_sigs_order)), dtype=float)
    counts_mat = np.zeros_like(exposures_mat)
    sig_to_global_col = {sig: idx for idx, sig in enumerate(all_used_sigs_order)}

    boot_rows: list[dict[str, object]] = []
    gof_rows: list[dict[str, object]] = []
    traces: dict[str, list[dict[str, object]]] = {}

    for s_idx, sample in enumerate(sample_names):
        cand_sigs = sample_sig_lists[sample]
        S_cand = ref_df[cand_sigs].to_numpy(dtype=float)
        G_cand = S_cand.T @ S_cand
        forced_mask = np.array([sig in forced_set for sig in cand_sigs], dtype=bool)

        obs_counts = cat_df[sample].to_numpy(dtype=float)
        total_muts = float(obs_counts.sum())

        w_est, cos_est, trace = prune_signatures_single(
            S_full=S_cand,
            G_full=G_cand,
            counts_or_prop=obs_counts,
            sig_names=cand_sigs,
            forced_mask=forced_mask,
            cos_sim_threshold=cos_sim_threshold,
            settings=settings,
            record_trace=True,
            is_bootstrap=False,
        )

        # Parametric multinomial bootstrap
        boot_weights: np.ndarray | None = None
        ci_low_arr = np.zeros(len(cand_sigs), dtype=float)
        ci_high_arr = np.zeros(len(cand_sigs), dtype=float)
        sd_arr = np.zeros(len(cand_sigs), dtype=float)
        boot_mean_arr = np.zeros(len(cand_sigs), dtype=float)
        present_prob_arr = np.zeros(len(cand_sigs), dtype=float)

        if bootstrap:
            if total_muts > 0.0:
                # If user supplied normalized proportions summing to ~1, use effective N=500
                n_draw = int(round(total_muts)) if total_muts >= 2.0 else 500
                p_emp = obs_counts / total_muts
                p_emp = p_emp / float(p_emp.sum())
                draws = rng.multinomial(n_draw, p_emp, size=effective_n_boot)
                boot_weights = np.zeros((effective_n_boot, len(cand_sigs)), dtype=float)
                for b_i in range(effective_n_boot):
                    wb, _, _ = prune_signatures_single(
                        S_full=S_cand,
                        G_full=G_cand,
                        counts_or_prop=draws[b_i].astype(float),
                        sig_names=cand_sigs,
                        forced_mask=forced_mask,
                        cos_sim_threshold=cos_sim_threshold,
                        settings=settings,
                        record_trace=False,
                        is_bootstrap=True,
                    )
                    boot_weights[b_i] = wb

                boot_mean_arr = np.mean(boot_weights, axis=0)
                ci_low_arr = np.percentile(boot_weights, 2.5, axis=0)
                ci_high_arr = np.percentile(boot_weights, 97.5, axis=0)
                sd_arr = np.std(boot_weights, axis=0, ddof=1)
                present_prob_arr = np.mean(boot_weights > 0.0, axis=0)

        # Presence confirmation rules:
        # 1) R_default: SBS8 / SBS99 / SBS-MM1 require lower 95% bootstrap CI > 0
        # 2) Strand bias confirmation when strand_bias_df supplied for SBS31/SBS35/SBS99/SBS-MM1
        active_after_confirm = [i for i, val in enumerate(w_est) if val > 0.0 or forced_mask[i]]
        dropped_confirm: list[int] = []

        if settings == "R_default" and bootstrap and total_muts > 0.0:
            for i in list(active_after_confirm):
                sig_name = cand_sigs[i]
                if (
                    sig_name in ("SBS8", "SBS99", "SBS-MM1")
                    and not forced_mask[i]
                    and ci_low_arr[i] <= 0.0
                ):
                    dropped_confirm.append(i)

        if sb_res is not None and total_muts > 0.0:
            for i in list(active_after_confirm):
                sig_name = cand_sigs[i]
                if sig_name in CHEMO_STRAND_SIGNATURES and not forced_mask[i]:
                    if (sample, sig_name) in sb_lookup and not sb_lookup[(sample, sig_name)]:
                        if i not in dropped_confirm:
                            dropped_confirm.append(i)

        if dropped_confirm and total_muts > 0.0:
            remaining = [i for i in active_after_confirm if i not in dropped_confirm]
            if not remaining:
                # Keep highest point-estimate signature if everything was dropped
                remaining = [int(np.argmax(w_est))]
            y_obs = obs_counts / total_muts
            norm_y = float(np.linalg.norm(y_obs))
            b_full = S_cand.T @ y_obs
            w_sub, cos_est = _fit_subset_weights(
                S_cand, G_cand, y_obs, b_full, norm_y, remaining, settings
            )
            w_est = np.zeros(len(cand_sigs), dtype=float)
            for pos, orig_i in enumerate(remaining):
                w_est[orig_i] = w_sub[pos]
            for drop_i in dropped_confirm:
                trace.append(
                    {
                        "removed_signature": cand_sigs[drop_i],
                        "cos_before": cos_est,
                        "cos_after": cos_est,
                        "cos_drop": 0.0,
                        "reason": "confirmation_filter",
                    }
                )

        if dbg or trace:
            traces[sample] = trace

        # Compute GOF metrics on final w_est
        if total_muts > 0.0:
            y_obs = obs_counts / total_muts
            y_hat = S_cand @ w_est
            cos_final = cosine_similarity_vec(y_obs, y_hat)
            l2_err = float(np.linalg.norm(y_obs - y_hat))
            pos = y_obs > 0.0
            kl_div = float(
                np.sum(y_obs[pos] * np.log(y_obs[pos] / np.maximum(y_hat[pos], 1e-15)))
            )
        else:
            cos_final = 0.0
            l2_err = 0.0
            kl_div = 0.0

        for local_i, sig_name in enumerate(cand_sigs):
            g_col = sig_to_global_col[sig_name]
            exposures_mat[s_idx, g_col] = w_est[local_i]
            counts_mat[s_idx, g_col] = w_est[local_i] * total_muts
            if bootstrap:
                boot_rows.append(
                    {
                        "sample": sample,
                        "signature": sig_name,
                        "estimate": float(w_est[local_i]),
                        "boot_mean": float(boot_mean_arr[local_i]),
                        "ci_low": float(ci_low_arr[local_i]),
                        "ci_high": float(ci_high_arr[local_i]),
                        "sd": float(sd_arr[local_i]),
                        "present_prob": float(present_prob_arr[local_i]),
                    }
                )

        gof_rows.append(
            {
                "sample": sample,
                "total_mutations": total_muts,
                "n_active_signatures": int(np.sum(w_est > 0.0)),
                "cosine_similarity": cos_final,
                "l2_error": l2_err,
                "kl_divergence": kl_div,
            }
        )

    exposures_df = pd.DataFrame(
        exposures_mat,
        index=pd.Index(sample_names, name="sample"),
        columns=pd.Index(all_used_sigs_order, name="signature"),
    )
    exposures_counts_df = pd.DataFrame(
        counts_mat,
        index=pd.Index(sample_names, name="sample"),
        columns=pd.Index(all_used_sigs_order, name="signature"),
    )
    gof_df = pd.DataFrame(gof_rows)
    bootstrap_df = pd.DataFrame(boot_rows) if bootstrap else None

    return MMSigResult(
        exposures=exposures_df,
        exposures_counts=exposures_counts_df,
        gof_df=gof_df,
        bootstrap_df=bootstrap_df,
        strand_bias_df=sb_res,
        pruning_trace=traces,
        settings=settings,
    )


class MMSig:
    """
    Object-oriented mutational signature refitting estimator (`mmsig`).

    Parameters
    ----------
    cos_sim_threshold : float, default 0.01
        Cosine similarity decrease threshold for stepwise backward pruning.
    force_include : sequence of str or None, default ("SBS1", "SBS5")
        Signatures forced to remain active when present in the reference.
    bootstrap : bool, default True
        Whether to estimate multinomial bootstrap CIs and presence probabilities.
    n_boot : int, optional
        Number of bootstrap replicates (`500` in `'default'`, `1000` in `'R_default'`).
    seed : int, default 42
        Random seed for bootstrap resampling.
    settings : {"default", "R_default"}, default "default"
        Estimation and confirmation parity mode.
    """

    def __init__(
        self,
        cos_sim_threshold: float = 0.01,
        force_include: Sequence[str] | None = ("SBS1", "SBS5"),
        bootstrap: bool = True,
        n_boot: int | None = None,
        seed: int = 42,
        settings: Literal["default", "R_default"] = "default",
    ) -> None:
        validate_settings(settings)
        if cos_sim_threshold < 0.0 or cos_sim_threshold >= 1.0:
            raise ValueError(f"cos_sim_threshold must be in [0, 1); got {cos_sim_threshold}.")
        self.cos_sim_threshold = cos_sim_threshold
        self.force_include = tuple(force_include) if force_include is not None else None
        self.bootstrap = bootstrap
        self.n_boot = n_boot
        self.seed = seed
        self.settings: SettingsMode = settings
        self.result_: MMSigResult | None = None

    def fit(
        self,
        muts_df: pd.DataFrame,
        sig_ref_df: pd.DataFrame,
        sigs_cons: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        strand_bias_df: pd.DataFrame | None = None,
        dbg: bool = False,
    ) -> MMSigResult:
        """Fit mutational signatures on `muts_df` and store `self.result_`."""
        self.result_ = mm_fit_signatures(
            muts_df=muts_df,
            sig_ref_df=sig_ref_df,
            sigs_cons=sigs_cons,
            cos_sim_threshold=self.cos_sim_threshold,
            force_include=self.force_include,
            bootstrap=self.bootstrap,
            n_boot=self.n_boot,
            strand_bias_df=strand_bias_df,
            dbg=dbg,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_

    def check_strand_bias(
        self,
        strand_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Evaluate transcriptional strand asymmetry using configured `self.settings`."""
        return check_strand_bias(strand_df=strand_df, alpha=alpha, settings=self.settings)
