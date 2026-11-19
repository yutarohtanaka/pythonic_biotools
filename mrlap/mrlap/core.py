"""Core MRlap analytical bias correction and formal difference testing."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from .architecture import (
    GeneticArchitectureResult,
    SettingsMode,
    estimate_bivariate_ldsc_with_jackknife,
    estimate_exposure_polygenicity,
    truncated_moments_closed_form,
    truncated_moments_gauss_hermite,
    validate_settings,
)


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find any of required columns {candidates} in DataFrame with columns {list(df.columns)}."
        )
    return None


@dataclass(frozen=True)
class MRlapResult:
    """Analytical MRlap correction output (Mounier & Kutalik, Genetic Epidemiology 2023)."""

    alpha_obs: float
    alpha_obs_se: float
    p_obs: float
    alpha_corrected: float
    alpha_corrected_se: float
    p_corrected: float
    cov_obs_corr: float
    diff_stat: float
    p_diff: float
    lambda_attenuation: float
    bias_overlap: float
    e_bx2_given_selected: float
    e_true_obs_bx_given_selected: float
    e_bx_ey_given_selected: float
    n_instruments: int
    n_snps_common: int
    instrument_snps: list[str]
    architecture: GeneticArchitectureResult
    n_sim: int
    mr_threshold: float
    mr_pruning_ld: float
    settings: SettingsMode

    def to_dataframe(self) -> pd.DataFrame:
        """Return single-row summary DataFrame of observed vs corrected MR estimates."""
        return pd.DataFrame(
            [
                {
                    "alpha_obs": self.alpha_obs,
                    "alpha_obs_se": self.alpha_obs_se,
                    "p_obs": self.p_obs,
                    "alpha_corrected": self.alpha_corrected,
                    "alpha_corrected_se": self.alpha_corrected_se,
                    "p_corrected": self.p_corrected,
                    "cov_obs_corr": self.cov_obs_corr,
                    "diff_stat": self.diff_stat,
                    "p_diff": self.p_diff,
                    "lambda_attenuation": self.lambda_attenuation,
                    "bias_overlap": self.bias_overlap,
                    "n_instruments": self.n_instruments,
                    "h2_x": self.architecture.h2_x,
                    "h2_y": self.architecture.h2_y,
                    "rg": self.architecture.rg,
                    "c_xy": self.architecture.c_xy,
                    "pi_x": self.architecture.pi_x,
                    "settings": self.settings,
                }
            ]
        )

    def architecture_dataframe(self) -> pd.DataFrame:
        """Return genetic architecture summary DataFrame."""
        a = self.architecture
        return pd.DataFrame(
            [
                {
                    "h2_x": a.h2_x,
                    "h2_x_se": a.h2_x_se,
                    "h2_y": a.h2_y,
                    "h2_y_se": a.h2_y_se,
                    "g_xy": a.g_xy,
                    "rg": a.rg,
                    "i_x": a.i_x,
                    "i_y": a.i_y,
                    "c_xy": a.c_xy,
                    "c_xy_se": a.c_xy_se,
                    "pi_x": a.pi_x,
                    "sigma2_x": a.sigma2_x,
                    "m_snps": a.m_snps,
                }
            ]
        )

    def summary(self) -> str:
        """Human-readable summary of MRlap results."""
        return "\n".join(
            [
                f"MRlap Summary (settings={self.settings!r}, IVs={self.n_instruments}, common_SNPs={self.n_snps_common})",
                f"  Observed IVW   : {self.alpha_obs:+.4f} (SE={self.alpha_obs_se:.4f}, p={self.p_obs:.3e})",
                f"  MRlap Corrected: {self.alpha_corrected:+.4f} (SE={self.alpha_corrected_se:.4f}, p={self.p_corrected:.3e})",
                f"  Bias Diff Test : Z_diff={self.diff_stat:+.3f}, p_diff={self.p_diff:.3e} (cov={self.cov_obs_corr:.6f})",
                f"  Decomposition  : lambda={self.lambda_attenuation:.4f}, overlap_bias={self.bias_overlap:+.4f}",
                f"  Architecture   : h2_x={self.architecture.h2_x:.4f}, pi_x={self.architecture.pi_x:.5f}, c_xy={self.architecture.c_xy:+.4f}",
            ]
        )


def _ld_clump_indices(
    cand_indices: np.ndarray,
    abs_z: np.ndarray,
    ld_R: np.ndarray | None,
    r2_thresh: float,
) -> np.ndarray:
    """Greedy p-value / |z| LD clumping among candidate discovery SNPs."""
    if cand_indices.size <= 1 or ld_R is None:
        order = np.argsort(-abs_z[cand_indices])
        return cand_indices[order]

    order = np.argsort(-abs_z[cand_indices])
    sorted_idx = cand_indices[order]
    kept: list[int] = []
    alive = np.ones(sorted_idx.size, dtype=bool)

    for pos in range(sorted_idx.size):
        if not alive[pos]:
            continue
        lead = int(sorted_idx[pos])
        kept.append(lead)
        if pos + 1 < sorted_idx.size:
            rem_pos = np.where(alive[pos + 1 :])[0] + (pos + 1)
            rem_snps = sorted_idx[rem_pos]
            r2 = np.square(ld_R[lead, rem_snps])
            alive[rem_pos[r2 > r2_thresh]] = False

    return np.array(kept, dtype=int)


def mrlap(
    exposure_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    ld_scores: ArrayLike | None = None,
    ld_R: ArrayLike | None = None,
    mr_threshold: float = 5e-8,
    mr_pruning_ld: float = 0.001,
    m_snps: float | None = None,
    n_sim: int | None = None,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> MRlapResult:
    """Run MRlap analytical bias correction for sample overlap, weak IVs, and Winner's curse.

    Implements Mounier & Kutalik, *Genetic Epidemiology* 2023 (DOI: 10.1002/gepi.22522).
    """
    mode = validate_settings(settings)
    if not (0.0 < mr_threshold < 1.0):
        raise ValueError("mr_threshold must lie strictly in (0, 1).")
    if not (0.0 <= mr_pruning_ld <= 1.0):
        raise ValueError("mr_pruning_ld must lie in [0, 1].")

    n_sim_eff = (500 if mode == "default" else 1000) if n_sim is None else int(n_sim)
    if n_sim_eff < 10:
        raise ValueError("n_sim must be >= 10.")

    if not isinstance(exposure_df, pd.DataFrame) or not isinstance(outcome_df, pd.DataFrame):
        raise TypeError("exposure_df and outcome_df must be pandas DataFrames.")

    exp = exposure_df.copy()
    out = outcome_df.copy()

    snp_col_x = _pick_col(exp, ("snp", "rsid", "variant_id", "id"), required=False)
    snp_col_y = _pick_col(out, ("snp", "rsid", "variant_id", "id"), required=False)
    if snp_col_x is None or snp_col_y is None:
        if len(exp) != len(out):
            raise ValueError(
                "When SNP ID column is absent, exposure_df and outcome_df must have equal length."
            )
        exp["_snp_id"] = [f"rs{i}" for i in range(len(exp))]
        out["_snp_id"] = [f"rs{i}" for i in range(len(out))]
        snp_col_x = snp_col_y = "_snp_id"

    bx_col = _pick_col(exp, ("beta", "b", "bx", "beta_exp", "effect"))
    sx_col = _pick_col(exp, ("se", "sx", "se_exp", "std_error"))
    nx_col = _pick_col(exp, ("n", "samplesize", "nx", "n_exp"), required=False)

    by_col = _pick_col(out, ("beta", "b", "by", "beta_out", "effect"))
    sy_col = _pick_col(out, ("se", "sy", "se_out", "std_error"))
    ny_col = _pick_col(out, ("n", "samplesize", "ny", "n_out"), required=False)

    assert bx_col is not None and sx_col is not None
    assert by_col is not None and sy_col is not None

    exp["_orig_idx"] = np.arange(len(exp), dtype=int)
    merged = pd.merge(
        exp,
        out,
        left_on=snp_col_x,
        right_on=snp_col_y,
        suffixes=("_x", "_y"),
        how="inner",
    )
    if len(merged) < 30:
        raise ValueError("At least 30 overlapping SNPs are required for MRlap.")

    def _resolve_merged_col(col_name: str, side: str) -> str:
        if f"{col_name}_{side}" in merged.columns:
            return f"{col_name}_{side}"
        return col_name

    snp_col_m = snp_col_x if snp_col_x in merged.columns else f"{snp_col_x}_x"
    bx_raw = merged[_resolve_merged_col(bx_col, "x")].to_numpy(dtype=float)
    sx_raw = merged[_resolve_merged_col(sx_col, "x")].to_numpy(dtype=float)
    by_raw = merged[_resolve_merged_col(by_col, "y")].to_numpy(dtype=float)
    sy_raw = merged[_resolve_merged_col(sy_col, "y")].to_numpy(dtype=float)

    # Harmonize effect alleles if present in both DataFrames
    ea_x = _pick_col(exp, ("ea", "effect_allele", "a1"), required=False)
    ea_y = _pick_col(out, ("ea", "effect_allele", "a1"), required=False)
    nea_x = _pick_col(exp, ("nea", "other_allele", "a2"), required=False)
    nea_y = _pick_col(out, ("nea", "other_allele", "a2"), required=False)
    allele_keep = np.ones(len(merged), dtype=bool)
    if ea_x and ea_y and nea_x and nea_y:
        eax_s = merged[_resolve_merged_col(ea_x, "x")].astype(str).str.upper().to_numpy()
        neax_s = merged[_resolve_merged_col(nea_x, "x")].astype(str).str.upper().to_numpy()
        eay_s = merged[_resolve_merged_col(ea_y, "y")].astype(str).str.upper().to_numpy()
        neay_s = merged[_resolve_merged_col(nea_y, "y")].astype(str).str.upper().to_numpy()
        same = (eax_s == eay_s) & (neax_s == neay_s)
        flipped = (eax_s == neay_s) & (neax_s == eay_s)
        by_raw = np.where(flipped, -by_raw, by_raw)
        allele_keep = same | flipped

    if nx_col is not None:
        nx_raw = merged[_resolve_merged_col(nx_col, "x")].to_numpy(dtype=float)
    else:
        nx_raw = 1.0 / np.maximum(sx_raw * sx_raw, 1e-12)

    if ny_col is not None:
        ny_raw = merged[_resolve_merged_col(ny_col, "y")].to_numpy(dtype=float)
    else:
        ny_raw = 1.0 / np.maximum(sy_raw * sy_raw, 1e-12)

    orig_idx = merged["_orig_idx"].to_numpy(dtype=int)
    if ld_scores is not None:
        ld_in = np.asarray(ld_scores, dtype=float).ravel()
        if ld_in.size == len(exp):
            ld_raw = ld_in[orig_idx]
        elif ld_in.size == len(merged):
            ld_raw = ld_in
        else:
            raise ValueError(
                "ld_scores length must match exposure_df or merged overlapping SNPs."
            )
    else:
        ld_col = _pick_col(merged, ("ld_score", "l2", "ld", "ld_score_x", "l2_x"), required=False)
        ld_raw = (
            merged[ld_col].to_numpy(dtype=float)
            if ld_col is not None
            else np.ones(len(merged), dtype=float)
        )

    ld_mat: np.ndarray | None = None
    if ld_R is not None:
        ld_mat_in = np.asarray(ld_R, dtype=float)
        if ld_mat_in.ndim != 2 or ld_mat_in.shape[0] != ld_mat_in.shape[1]:
            raise ValueError("ld_R must be a square 2D LD correlation matrix.")
        if ld_mat_in.shape[0] == len(exp):
            ld_mat = ld_mat_in[np.ix_(orig_idx, orig_idx)]
        elif ld_mat_in.shape[0] == len(merged):
            ld_mat = ld_mat_in
        else:
            raise ValueError("ld_R dimensions must match exposure_df or merged SNPs.")

    if np.any(sx_raw <= 0) or np.any(sy_raw <= 0):
        raise ValueError("Standard errors must be strictly positive.")
    if np.any(nx_raw <= 0) or np.any(ny_raw <= 0):
        raise ValueError("Sample sizes must be strictly positive.")

    valid = (
        allele_keep
        & np.isfinite(bx_raw)
        & np.isfinite(sx_raw)
        & np.isfinite(by_raw)
        & np.isfinite(sy_raw)
        & np.isfinite(nx_raw)
        & np.isfinite(ny_raw)
        & np.isfinite(ld_raw)
    )
    if int(np.sum(valid)) < 30:
        raise ValueError("Fewer than 30 finite harmonized SNPs remain.")

    snp_ids = merged[snp_col_m].astype(str).to_numpy()[valid]
    zx = (bx_raw / sx_raw)[valid]
    zy = (by_raw / sy_raw)[valid]
    nx_arr = nx_raw[valid]
    ny_arr = ny_raw[valid]
    ld_arr = np.maximum(ld_raw[valid], 1e-3)
    if ld_mat is not None:
        ld_mat = ld_mat[np.ix_(valid, valid)]

    # Standardized summary statistics (Mounier & Kutalik 2023)
    std_bx = zx / np.sqrt(nx_arr)
    std_sx = 1.0 / np.sqrt(nx_arr)
    std_by = zy / np.sqrt(ny_arr)
    std_sy = 1.0 / np.sqrt(ny_arr)

    n_common = int(zx.size)
    m_eff = float(m_snps) if m_snps is not None else float(n_common)

    # Stage 1: Bivariate LDSC + genomic block-jackknife covariance
    i_x, h2_x, i_y, h2_y, c_xy, g_xy, cov_ldsc = estimate_bivariate_ldsc_with_jackknife(
        zx, zy, ld_arr, nx_arr, ny_arr, m_eff=m_eff, n_blocks=20
    )
    rg = float(np.clip(g_xy / np.sqrt(max(h2_x * h2_y, 1e-12)), -1.0, 1.0))

    # Stage 2: Instrument discovery (|z_X| > T) and LD pruning
    z_thresh = float(stats.norm.isf(mr_threshold / 2.0))
    abs_zx = np.abs(zx)
    cand_idx = np.where(abs_zx > z_thresh)[0]
    if cand_idx.size < 2:
        raise ValueError(
            f"Only {cand_idx.size} SNPs passed mr_threshold={mr_threshold:g} (|z| > {z_thresh:.2f}); "
            "at least 2 instruments are required for MRlap."
        )

    iv_idx = _ld_clump_indices(cand_idx, abs_zx, ld_mat, r2_thresh=mr_pruning_ld)
    if iv_idx.size < 2:
        raise ValueError(
            f"Only {iv_idx.size} independent instrument remained after LD pruning at r^2 <= {mr_pruning_ld}."
        )

    bx_iv = std_bx[iv_idx]
    by_iv = std_by[iv_idx]
    sy_iv = std_sy[iv_idx]
    nx_iv = nx_arr[iv_idx]
    ny_iv = ny_arr[iv_idx]
    k_iv = int(iv_idx.size)

    # Standard random-effects IVW estimate on standardized instruments
    w_iv = 1.0 / (sy_iv * sy_iv)
    denom_ivw = float(np.sum(w_iv * bx_iv * bx_iv))
    alpha_obs = float(np.sum(w_iv * bx_iv * by_iv) / denom_ivw)
    resid_ivw = by_iv - alpha_obs * bx_iv
    phi_ivw = max(1.0, float(np.sum(w_iv * resid_ivw * resid_ivw) / max(k_iv - 1, 1)))
    alpha_obs_se = float(np.sqrt(phi_ivw / denom_ivw))
    p_obs = float(2.0 * stats.norm.sf(abs(alpha_obs) / max(alpha_obs_se, 1e-15)))

    # Stage 3: Truncated spike-and-slab MLE for exposure polygenicity pi_X
    pi_x, sigma2_x, se_logit_pi = estimate_exposure_polygenicity(
        bx_iv,
        nx_iv,
        h2_x=h2_x,
        m_eff=m_eff,
        z_thresh=z_thresh,
        mean_ld=float(np.mean(ld_arr)),
        settings=mode,
    )

    arch = GeneticArchitectureResult(
        h2_x=float(h2_x),
        h2_x_se=float(np.sqrt(max(cov_ldsc[0, 0], 1e-12))),
        h2_y=float(h2_y),
        h2_y_se=float(np.sqrt(max(cov_ldsc[1, 1], 1e-12))),
        g_xy=float(g_xy),
        rg=rg,
        i_x=float(i_x),
        i_y=float(i_y),
        c_xy=float(c_xy),
        c_xy_se=float(np.sqrt(max(cov_ldsc[2, 2], 1e-12))),
        cov_h2x_cxy=float(cov_ldsc[0, 2]),
        pi_x=float(pi_x),
        logit_pi_x_se=float(se_logit_pi),
        sigma2_x=float(sigma2_x),
        m_snps=float(m_eff),
    )

    # Stage 4: Exact conditional expectations given |z_hat_X| > T and analytical bias inversion
    moment_fn = (
        truncated_moments_gauss_hermite
        if mode == "default"
        else truncated_moments_closed_form
    )
    e_bx2, e_b_bhat, e_bhat_ey, lam = moment_fn(
        nx_iv, ny_iv, h2_x=h2_x, pi_x=pi_x, c_xy=c_xy, m_eff=m_eff, z_thresh=z_thresh
    )
    bias_overlap = float(e_bhat_ey / max(e_bx2, 1e-24))
    alpha_corrected = float((alpha_obs - bias_overlap) / lam)

    # Stage 5: Parametric genetic-architecture resampling for SE(alpha_corrected), Cov, and Z_diff
    rng = np.random.default_rng(seed)

    z_obs_std = rng.standard_normal(n_sim_eff)
    z_obs_std = (z_obs_std - float(np.mean(z_obs_std))) / float(np.std(z_obs_std, ddof=1))
    alpha_obs_draws = alpha_obs + alpha_obs_se * z_obs_std

    cov_2d = np.array(
        [
            [cov_ldsc[0, 0], cov_ldsc[0, 2]],
            [cov_ldsc[2, 0], cov_ldsc[2, 2]],
        ],
        dtype=float,
    )
    eig_v, eig_u = np.linalg.eigh(0.5 * (cov_2d + cov_2d.T))
    cov_2d_psd = (eig_u * np.maximum(eig_v, 1e-12)[None, :]) @ eig_u.T
    ldsc_draws = rng.multivariate_normal(
        mean=np.array([h2_x, c_xy], dtype=float), cov=cov_2d_psd, size=n_sim_eff
    )
    h2x_draws = np.clip(ldsc_draws[:, 0], 1e-4, 0.99)
    cxy_draws = np.clip(ldsc_draws[:, 1], -0.98, 0.98)

    pi_low = 1e-5 if mode == "default" else 1e-7
    pi_high = 0.50 if mode == "default" else 0.999
    u_pi_hat = float(logit(np.clip(pi_x, pi_low, pi_high)))
    pi_draws = np.clip(
        expit(rng.normal(u_pi_hat, se_logit_pi, size=n_sim_eff)), pi_low, pi_high
    )

    alpha_corr_draws = np.empty(n_sim_eff, dtype=float)
    for s_idx in range(n_sim_eff):
        eb2_s, _, ebey_s, lam_s = truncated_moments_closed_form(
            nx_iv,
            ny_iv,
            h2_x=float(h2x_draws[s_idx]),
            pi_x=float(pi_draws[s_idx]),
            c_xy=float(cxy_draws[s_idx]),
            m_eff=m_eff,
            z_thresh=z_thresh,
        )
        b_ov_s = ebey_s / max(eb2_s, 1e-24)
        alpha_corr_draws[s_idx] = (alpha_obs_draws[s_idx] - b_ov_s) / lam_s

    alpha_corrected_se = float(np.std(alpha_corr_draws, ddof=1))
    cov_obs_corr = float(np.cov(alpha_obs_draws, alpha_corr_draws, ddof=1)[0, 1])
    p_corrected = float(
        2.0 * stats.norm.sf(abs(alpha_corrected) / max(alpha_corrected_se, 1e-15))
    )

    var_diff = max(
        alpha_obs_se * alpha_obs_se
        + alpha_corrected_se * alpha_corrected_se
        - 2.0 * cov_obs_corr,
        1e-24,
    )
    diff_stat = float((alpha_obs - alpha_corrected) / np.sqrt(var_diff))
    p_diff = float(2.0 * stats.norm.sf(abs(diff_stat)))

    return MRlapResult(
        alpha_obs=alpha_obs,
        alpha_obs_se=alpha_obs_se,
        p_obs=p_obs,
        alpha_corrected=alpha_corrected,
        alpha_corrected_se=alpha_corrected_se,
        p_corrected=p_corrected,
        cov_obs_corr=cov_obs_corr,
        diff_stat=diff_stat,
        p_diff=p_diff,
        lambda_attenuation=lam,
        bias_overlap=bias_overlap,
        e_bx2_given_selected=e_bx2,
        e_true_obs_bx_given_selected=e_b_bhat,
        e_bx_ey_given_selected=e_bhat_ey,
        n_instruments=k_iv,
        n_snps_common=n_common,
        instrument_snps=[str(s) for s in snp_ids[iv_idx]],
        architecture=arch,
        n_sim=n_sim_eff,
        mr_threshold=float(mr_threshold),
        mr_pruning_ld=float(mr_pruning_ld),
        settings=mode,
    )


class MRlap:
    """Object-oriented estimator for MRlap analytical bias correction."""

    def __init__(
        self,
        mr_threshold: float = 5e-8,
        mr_pruning_ld: float = 0.001,
        n_sim: int | None = None,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        if not (0.0 < mr_threshold < 1.0):
            raise ValueError("mr_threshold must lie strictly in (0, 1).")
        if not (0.0 <= mr_pruning_ld <= 1.0):
            raise ValueError("mr_pruning_ld must lie in [0, 1].")
        if n_sim is not None and n_sim < 10:
            raise ValueError("n_sim must be >= 10.")
        self.mr_threshold = mr_threshold
        self.mr_pruning_ld = mr_pruning_ld
        self.n_sim = n_sim
        self.seed = seed
        self.result_: MRlapResult | None = None

    def fit(
        self,
        exposure_df: pd.DataFrame,
        outcome_df: pd.DataFrame,
        ld_scores: ArrayLike | None = None,
        ld_R: ArrayLike | None = None,
        m_snps: float | None = None,
    ) -> MRlapResult:
        """Fit MRlap on exposure and outcome summary-statistic DataFrames."""
        self.result_ = mrlap(
            exposure_df=exposure_df,
            outcome_df=outcome_df,
            ld_scores=ld_scores,
            ld_R=ld_R,
            mr_threshold=self.mr_threshold,
            mr_pruning_ld=self.mr_pruning_ld,
            m_snps=m_snps,
            n_sim=self.n_sim,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_
