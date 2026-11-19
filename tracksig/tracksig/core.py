"""
Main TrackSig and TrackSigFreq trajectory segmentation pipeline, nonparametric
mutation bootstrap, OOP estimator wrapper, and result container.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .ccf import SettingsMode, compute_ccf_pseudotime, validate_settings
from .em import (
    detect_active_signatures,
    fit_multinomial_signatures_em,
    map_mutations_to_channels,
    normalize_signature_matrix,
)
from .pelt import SegmentScorer, run_pelt


@dataclass
class TrackSigResult:
    """Result container for TrackSig / TrackSigFreq evolutionary trajectory inference."""

    changepoints: list[int]
    changepoint_ccfs: list[float]
    bin_exposures_df: pd.DataFrame
    segment_exposures_df: pd.DataFrame
    raw_bin_exposures_df: pd.DataFrame
    active_signatures: list[str]
    bic: float
    log_likelihood: float
    penalty_per_changepoint: float
    n_mutations: int
    n_bins: int
    purity: float
    bin_size: int
    effective_bin_size: int
    score_method: str
    settings: SettingsMode
    changepoint_support: dict[int, float] | None = None
    bin_exposure_ci_low_df: pd.DataFrame | None = None
    bin_exposure_ci_high_df: pd.DataFrame | None = None

    def summary(self) -> str:
        """Return a concise human-readable summary of the fitted evolutionary trajectory."""
        cp_str = (
            ", ".join(
                f"bin {cp} (CCF={ccf:.3f})"
                for cp, ccf in zip(self.changepoints, self.changepoint_ccfs)
            )
            if self.changepoints
            else "None (single homogeneous population)"
        )
        lines = [
            f"TrackSigResult(score_method={self.score_method!r}, settings={self.settings!r})",
            f"  Mutations: {self.n_mutations} across {self.n_bins} pseudotime bins "
            f"(effective_bin_size={self.effective_bin_size}, purity={self.purity:.2f})",
            f"  Active signatures ({len(self.active_signatures)}): {', '.join(self.active_signatures)}",
            f"  Changepoints ({len(self.changepoints)}): {cp_str}",
            f"  BIC: {self.bic:.3f} (logL={self.log_likelihood:.3f}, penalty={self.penalty_per_changepoint:.3f})",
        ]
        if self.changepoint_support is not None and self.changepoints:
            sup_str = ", ".join(
                f"bin {cp}: {self.changepoint_support.get(cp, 0.0):.2f}"
                for cp in self.changepoints
            )
            lines.append(f"  Bootstrap support: {sup_str}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the per-bin evolutionary exposure trajectory DataFrame."""
        return self.bin_exposures_df.copy()


def _fit_single_trajectory(
    binned_df: pd.DataFrame,
    phi_active: np.ndarray,
    active_sigs: list[str],
    context_names: list[str],
    score_method: str,
    settings: SettingsMode,
) -> tuple[
    list[int],
    list[float],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    float,
]:
    """Core PELT segmentation on pre-binned mutations and pre-selected active signatures."""
    n_mut = len(binned_df)
    n_bins = int(binned_df["bin_id"].max())
    n_ctx = phi_active.shape[0]
    k_active = phi_active.shape[1]

    channels = map_mutations_to_channels(binned_df, context_names)
    bin_ids = binned_df["bin_id"].to_numpy(dtype=int)
    ccf_vals = binned_df["ccf"].to_numpy(dtype=float)
    ccf_cont = binned_df["ccf_cont"].to_numpy(dtype=float)
    ccf_sds = binned_df["ccf_sd"].to_numpy(dtype=float)

    bin_counts = np.zeros((n_bins, n_ctx), dtype=float)
    np.add.at(bin_counts, (bin_ids - 1, channels), 1.0)

    bin_n = np.zeros(n_bins, dtype=float)
    bin_sum_ccf = np.zeros(n_bins, dtype=float)
    bin_sum_ccf2 = np.zeros(n_bins, dtype=float)
    bin_mean_raw_ccf = np.zeros(n_bins, dtype=float)
    bin_min_ccf = np.zeros(n_bins, dtype=float)
    bin_max_ccf = np.zeros(n_bins, dtype=float)
    bin_min_cont = np.zeros(n_bins, dtype=float)
    bin_max_cont = np.zeros(n_bins, dtype=float)

    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        c_b = ccf_vals[mask]
        cc_b = ccf_cont[mask]
        bin_n[b - 1] = float(len(c_b))
        bin_sum_ccf[b - 1] = float(np.sum(cc_b))
        bin_sum_ccf2[b - 1] = float(np.sum(cc_b * cc_b))
        bin_mean_raw_ccf[b - 1] = float(np.mean(c_b))
        bin_min_ccf[b - 1] = float(np.min(c_b))
        bin_max_ccf[b - 1] = float(np.max(c_b))
        bin_min_cont[b - 1] = float(np.min(cc_b))
        bin_max_cont[b - 1] = float(np.max(cc_b))

    # Strictly decreasing truncation boundaries c_0 = +inf > c_1 > ... > c_{T-1} > c_T = -inf
    boundaries = np.empty(n_bins + 1, dtype=float)
    boundaries[0] = np.inf
    boundaries[n_bins] = -np.inf
    for b in range(1, n_bins):
        boundaries[b] = 0.5 * (bin_min_cont[b - 1] + bin_max_cont[b])
        if boundaries[b] >= boundaries[b - 1]:
            boundaries[b] = boundaries[b - 1] - 1e-6

    global_sigma = float(np.sqrt(np.mean(ccf_sds * ccf_sds)))

    if settings == "default":
        df_seg = max(1, k_active - 1) if score_method == "Signature" else max(1, k_active)
        penalty = float(df_seg * np.log(max(2, n_mut)))
    else:
        df_seg = (
            max(1, k_active - 1)
            if score_method == "Signature"
            else max(1, (k_active - 1) + 2)
        )
        penalty = float(df_seg * np.log(max(2, n_bins)))

    scorer = SegmentScorer(
        bin_counts=bin_counts,
        phi_active=phi_active,
        score_method=score_method,
        bin_n=bin_n,
        bin_sum_ccf=bin_sum_ccf,
        bin_sum_ccf2=bin_sum_ccf2,
        boundaries=boundaries,
        global_sigma=global_sigma,
        mean_scale_factor=binned_df.attrs.get("mean_scale_factor"),
        rms_kappa=binned_df.attrs.get("rms_kappa"),
    )

    cpts, f_final = run_pelt(scorer.cost, n_bins=n_bins, penalty=penalty, use_pruning=True)

    # Changepoint CCFs at boundary between bin tau and bin tau + 1
    cp_ccfs = [
        float(0.5 * (bin_min_ccf[cp - 1] + bin_max_ccf[cp])) for cp in cpts
    ]

    # Build segment & bin exposure tables
    seg_starts = [1] + [cp + 1 for cp in cpts]
    seg_ends = cpts + [n_bins]

    seg_records: list[dict[str, float | int]] = []
    bin_seg_id = np.zeros(n_bins, dtype=int)
    bin_fitted_exp = np.zeros((n_bins, k_active), dtype=float)
    total_loglik = 0.0

    for seg_idx, (a, b) in enumerate(zip(seg_starts, seg_ends), start=1):
        seg_cost, seg_pi, fitted_ccf = scorer.evaluate(a, b)
        total_loglik += -0.5 * seg_cost
        n_seg = int(np.sum(bin_n[a - 1 : b]))
        mean_seg_ccf = float(
            np.sum(bin_mean_raw_ccf[a - 1 : b] * bin_n[a - 1 : b]) / max(n_seg, 1)
        )
        rec: dict[str, float | int] = {
            "segment_id": seg_idx,
            "start_bin": a,
            "end_bin": b,
            "n_mutations": n_seg,
            "mean_ccf": mean_seg_ccf,
            "fitted_ccf": fitted_ccf,
        }
        for s_i, s_name in enumerate(active_sigs):
            rec[s_name] = float(seg_pi[s_i])
        seg_records.append(rec)
        bin_seg_id[a - 1 : b] = seg_idx
        bin_fitted_exp[a - 1 : b, :] = seg_pi[None, :]

    segment_df = pd.DataFrame(seg_records)

    # Per-bin fitted trajectory and unpooled single-bin EM exposures
    raw_exp = np.zeros((n_bins, k_active), dtype=float)
    for b in range(n_bins):
        raw_pi, _ = fit_multinomial_signatures_em(
            bin_counts[b], phi_active, tol=1e-6, max_iter=500, settings=settings
        )
        raw_exp[b, :] = raw_pi

    base_bin_cols = {
        "bin_id": np.arange(1, n_bins + 1, dtype=int),
        "segment_id": bin_seg_id,
        "n_mutations": bin_n.astype(int),
        "mean_ccf": bin_mean_raw_ccf,
        "min_ccf": bin_min_ccf,
        "max_ccf": bin_max_ccf,
    }
    bin_df = pd.DataFrame(base_bin_cols)
    raw_bin_df = pd.DataFrame(base_bin_cols)
    for s_i, s_name in enumerate(active_sigs):
        bin_df[s_name] = bin_fitted_exp[:, s_i]
        raw_bin_df[s_name] = raw_exp[:, s_i]

    # Full BIC = sum_j (-2 logL_j) + (m + 1) * penalty = F(n_bins) + penalty
    bic = float(f_final + penalty)
    return (
        cpts,
        cp_ccfs,
        bin_df,
        segment_df,
        raw_bin_df,
        bic,
        float(total_loglik),
        penalty,
    )


def tracksig(
    mutations_df: pd.DataFrame,
    signature_matrix: pd.DataFrame | np.ndarray,
    purity: float = 1.0,
    bin_size: int = 100,
    score_method: str = "SigFreq",
    active_threshold: float = 0.05,
    bootstrap_iters: int = 0,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> TrackSigResult:
    """Reconstruct evolutionary mutational signature trajectories via TrackSig / TrackSigFreq.

    Parameters
    ----------
    mutations_df : pd.DataFrame
        SNV table with mutation context column (``context`` / ``trinucleotide`` /
        ``channel``) and read counts (``alt_count`` + ``total_depth`` / ``ref_count``)
        or ``ccf`` / ``vaf``.
    signature_matrix : DataFrame or ndarray of shape (C, K)
        Reference mutational signature definitions (e.g. COSMIC 96xK SBS matrix).
    purity : float, default 1.0
        Tumor cellularity ``rho in (0, 1]``.
    bin_size : int, default 100
        Target number of SNVs per evolutionary pseudotime bin.
    score_method : {"SigFreq", "Signature"}, default "SigFreq"
        Segment scoring objective: ``"Signature"`` (TrackSig multinomial 96-context
        log-likelihood) or ``"SigFreq"`` (TrackSigFreq joint multinomial + CCF
        mixture log-likelihood).
    active_threshold : float, default 0.05
        Global EM minimum exposure threshold for pre-filtering active signatures.
    bootstrap_iters : int, default 0
        Number of nonparametric mutation bootstrap replicates for changepoint support
        frequencies and 95% per-bin exposure credible intervals.
    seed : int, default 42
        Random seed for nonparametric bootstrap resampling.
    settings : {"default", "R_default"}, default "default"
        Pipeline settings mode.

    Returns
    -------
    TrackSigResult
        Fitted changepoints, CCF boundaries, segment and bin exposures, BIC, and
        optional bootstrap uncertainty summaries.
    """
    validate_settings(settings)
    if score_method not in ("Signature", "SigFreq"):
        raise ValueError(
            f"Unknown score_method={score_method!r}. Expected 'Signature' or 'SigFreq'."
        )
    if int(bootstrap_iters) < 0:
        raise ValueError(f"bootstrap_iters must be >= 0, got {bootstrap_iters}.")

    phi_all, sig_names, context_names = normalize_signature_matrix(signature_matrix)
    binned_df = compute_ccf_pseudotime(
        mutations_df, purity=purity, bin_size=bin_size, settings=settings
    )
    eff_bin_size = int(binned_df.attrs.get("effective_bin_size", bin_size))

    active_sigs = detect_active_signatures(
        binned_df,
        signature_matrix,
        active_threshold=active_threshold,
        settings=settings,
    )
    sig_lookup = {name: idx for idx, name in enumerate(sig_names)}
    active_indices = [sig_lookup[name] for name in active_sigs]
    phi_active = phi_all[:, active_indices]

    (
        cpts,
        cp_ccfs,
        bin_df,
        segment_df,
        raw_bin_df,
        bic,
        total_ll,
        penalty,
    ) = _fit_single_trajectory(
        binned_df=binned_df,
        phi_active=phi_active,
        active_sigs=active_sigs,
        context_names=context_names,
        score_method=score_method,
        settings=settings,
    )

    n_bins = len(bin_df)
    k_active = len(active_sigs)
    cp_support: dict[int, float] | None = None
    ci_low_df: pd.DataFrame | None = None
    ci_high_df: pd.DataFrame | None = None

    n_boot = int(bootstrap_iters)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        n_mut = len(mutations_df)
        boot_trajectories = np.zeros((n_boot, n_bins, k_active), dtype=float)
        boot_cpts_list: list[list[int]] = []

        for b_iter in range(n_boot):
            resample_idx = rng.integers(0, n_mut, size=n_mut)
            boot_mut_df = mutations_df.iloc[resample_idx].reset_index(drop=True)
            boot_binned = compute_ccf_pseudotime(
                boot_mut_df, purity=purity, bin_size=bin_size, settings=settings
            )
            b_cpts, _, b_bin_df, _, _, _, _, _ = _fit_single_trajectory(
                binned_df=boot_binned,
                phi_active=phi_active,
                active_sigs=active_sigs,
                context_names=context_names,
                score_method=score_method,
                settings=settings,
            )
            boot_cpts_list.append(b_cpts)
            b_exp = b_bin_df[active_sigs].to_numpy(dtype=float)
            if len(b_exp) == n_bins:
                boot_trajectories[b_iter] = b_exp
            else:
                # Interpolate if bootstrap bin count differs
                x_src = np.linspace(0.0, 1.0, len(b_exp))
                x_dst = np.linspace(0.0, 1.0, n_bins)
                for s_i in range(k_active):
                    boot_trajectories[b_iter, :, s_i] = np.interp(x_dst, x_src, b_exp[:, s_i])

        # Support within +/- 1 bin for every interior boundary 1 .. n_bins - 1
        cp_support = {}
        for boundary in range(1, n_bins):
            hits = sum(
                1 for bc in boot_cpts_list if any(abs(c - boundary) <= 1 for c in bc)
            )
            cp_support[boundary] = float(hits / n_boot)

        q_low = np.quantile(boot_trajectories, 0.025, axis=0)
        q_high = np.quantile(boot_trajectories, 0.975, axis=0)
        ci_low_df = pd.DataFrame({"bin_id": bin_df["bin_id"].to_numpy(copy=True)})
        ci_high_df = pd.DataFrame({"bin_id": bin_df["bin_id"].to_numpy(copy=True)})
        for s_i, s_name in enumerate(active_sigs):
            ci_low_df[s_name] = q_low[:, s_i]
            ci_high_df[s_name] = q_high[:, s_i]

    return TrackSigResult(
        changepoints=cpts,
        changepoint_ccfs=cp_ccfs,
        bin_exposures_df=bin_df,
        segment_exposures_df=segment_df,
        raw_bin_exposures_df=raw_bin_df,
        active_signatures=active_sigs,
        bic=bic,
        log_likelihood=total_ll,
        penalty_per_changepoint=penalty,
        n_mutations=len(binned_df),
        n_bins=n_bins,
        purity=float(purity),
        bin_size=int(bin_size),
        effective_bin_size=eff_bin_size,
        score_method=score_method,
        settings=settings,
        changepoint_support=cp_support,
        bin_exposure_ci_low_df=ci_low_df,
        bin_exposure_ci_high_df=ci_high_df,
    )


class TrackSig:
    """Object-oriented estimator interface for TrackSig and TrackSigFreq."""

    def __init__(
        self,
        purity: float = 1.0,
        bin_size: int = 100,
        score_method: str = "SigFreq",
        active_threshold: float = 0.05,
        bootstrap_iters: int = 0,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        if score_method not in ("Signature", "SigFreq"):
            raise ValueError(
                f"Unknown score_method={score_method!r}. Expected 'Signature' or 'SigFreq'."
            )
        self.purity = purity
        self.bin_size = bin_size
        self.score_method = score_method
        self.active_threshold = active_threshold
        self.bootstrap_iters = bootstrap_iters
        self.seed = seed
        self.settings = settings
        self.result_: TrackSigResult | None = None

    def fit(
        self,
        mutations_df: pd.DataFrame,
        signature_matrix: pd.DataFrame | np.ndarray,
    ) -> TrackSigResult:
        """Fit evolutionary signature trajectory on ``mutations_df``."""
        self.result_ = tracksig(
            mutations_df=mutations_df,
            signature_matrix=signature_matrix,
            purity=self.purity,
            bin_size=self.bin_size,
            score_method=self.score_method,
            active_threshold=self.active_threshold,
            bootstrap_iters=self.bootstrap_iters,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_
