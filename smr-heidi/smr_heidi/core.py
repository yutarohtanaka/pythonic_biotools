"""
Core algorithms for SMR, HEIDI, and SMR-multi.

References:
  - Zhu, Z., et al. (2016). "Integration of summary data from GWAS and eQTL studies
    predicts complex trait gene targets." Nature Genetics, 48(5), 481-487.
  - Wu, Y., et al. (2018). "Integrative analysis of omics summary data reveals
    putative mechanisms underlying complex traits." Nature Communications, 9(1), 918.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

QuadMethod = Literal["auto", "imhof", "saddlepoint", "satterthwaite"]
SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")

_GL16_NODES, _GL16_WEIGHTS = np.polynomial.legendre.leggauss(16)
_EULER_M = 32
_EULER_WEIGHTS = stats.binom.pmf(np.arange(_EULER_M), _EULER_M - 1, 0.5)


@dataclass(frozen=True)
class SMRSingleResult:
    """Single-instrument SMR result (Zhu et al., 2016)."""

    snp_id: str
    b_xy: float
    se_xy: float
    t_smr: float
    p_smr: float
    b_zy: float
    se_zy: float
    z_zy: float
    b_zx: float
    se_zx: float
    z_zx: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "top_snp": self.snp_id,
                    "b_GWAS": self.b_zy,
                    "se_GWAS": self.se_zy,
                    "b_eQTL": self.b_zx,
                    "se_eQTL": self.se_zx,
                    "b_SMR": self.b_xy,
                    "se_SMR": self.se_xy,
                    "T_SMR": self.t_smr,
                    "p_SMR": self.p_smr,
                }
            ]
        )

    def summary(self) -> str:
        return (
            f"SMR Single-SNP ({self.snp_id}): b_xy = {self.b_xy:.6g} "
            f"(SE = {self.se_xy:.6g}), T_SMR = {self.t_smr:.4f}, p_SMR = {self.p_smr:.4e}"
        )


@dataclass(frozen=True)
class HeidiResult:
    """HEIDI dependent-instrument heterogeneity test result (Zhu et al., 2016 Eqs. 7-11)."""

    top_snp: str
    top_idx: int
    n_snps: int
    selected_indices: list[int]
    selected_snps: list[str]
    t_heidi: float
    p_heidi: float
    b_xy_selected: np.ndarray
    d_values: np.ndarray
    z_d: np.ndarray
    corr_d: np.ndarray
    eigenvalues: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "top_snp": self.top_snp,
                    "nsnp_HEIDI": self.n_snps,
                    "T_HEIDI": self.t_heidi,
                    "p_HEIDI": self.p_heidi,
                }
            ]
        )

    def snp_table(self) -> pd.DataFrame:
        z_full = np.concatenate([[0.0], self.z_d]) if self.n_snps >= 1 else np.array([], dtype=np.float64)
        d_full = np.concatenate([[0.0], self.d_values]) if self.n_snps >= 1 else np.array([], dtype=np.float64)
        return pd.DataFrame(
            {
                "snp": self.selected_snps,
                "index": self.selected_indices,
                "is_top": [k == 0 for k in range(len(self.selected_snps))],
                "b_xy": self.b_xy_selected,
                "d_from_top": d_full[: len(self.selected_snps)],
                "z_d": z_full[: len(self.selected_snps)],
            }
        )

    def summary(self) -> str:
        return (
            f"HEIDI Test (top={self.top_snp}, m={self.n_snps} SNPs): "
            f"T_HEIDI = {self.t_heidi:.4f}, p_HEIDI = {self.p_heidi:.4e}"
        )


@dataclass(frozen=True)
class SMRMultiResult:
    """Multi-cis-SNP SMR result (Wu et al., Nature Communications 2018)."""

    n_snps: int
    selected_indices: list[int]
    selected_snps: list[str]
    t_smr_per_snp: np.ndarray
    t_smr_sum: float
    t_smr_mean: float
    p_smr_multi: float
    b_xy_gls: float
    se_xy_gls: float
    eigenvalues: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "nsnp_SMR_multi": self.n_snps,
                    "b_SMR_GLS": self.b_xy_gls,
                    "se_SMR_GLS": self.se_xy_gls,
                    "T_SMR_mean": self.t_smr_mean,
                    "T_SMR_sum": self.t_smr_sum,
                    "p_SMR_multi": self.p_smr_multi,
                }
            ]
        )

    def summary(self) -> str:
        return (
            f"SMR-multi (m={self.n_snps} SNPs): T_mean = {self.t_smr_mean:.4f}, "
            f"p_SMR_multi = {self.p_smr_multi:.4e}, GLS b_xy = {self.b_xy_gls:.6g} (SE = {self.se_xy_gls:.6g})"
        )


@dataclass(frozen=True)
class LocusSMRResult:
    """Combined end-to-end locus SMR + HEIDI + SMR-multi result."""

    probe_id: str
    n_harmonized_snps: int
    n_flipped_gwas: int
    n_flipped_ld: int
    smr: SMRSingleResult
    heidi: HeidiResult
    smr_multi: SMRMultiResult | None
    harmonized_snps: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "probe_id": self.probe_id,
                    "top_snp": self.smr.snp_id,
                    "n_harmonized_snps": self.n_harmonized_snps,
                    "b_GWAS": self.smr.b_zy,
                    "se_GWAS": self.smr.se_zy,
                    "b_eQTL": self.smr.b_zx,
                    "se_eQTL": self.smr.se_zx,
                    "b_SMR": self.smr.b_xy,
                    "se_SMR": self.smr.se_xy,
                    "T_SMR": self.smr.t_smr,
                    "p_SMR": self.smr.p_smr,
                    "p_SMR_multi": self.smr_multi.p_smr_multi if self.smr_multi is not None else np.nan,
                    "T_HEIDI": self.heidi.t_heidi,
                    "p_HEIDI": self.heidi.p_heidi,
                    "nsnp_HEIDI": self.heidi.n_snps,
                }
            ]
        )

    def summary(self) -> str:
        lines = [
            f"Locus SMR & HEIDI Summary ({self.probe_id})",
            "=" * 60,
            f"Harmonized SNPs: {self.n_harmonized_snps} "
            f"(GWAS allele flips: {self.n_flipped_gwas}, LD sign flips: {self.n_flipped_ld})",
            "  " + self.smr.summary(),
            "  " + self.heidi.summary(),
        ]
        if self.smr_multi is not None:
            lines.append("  " + self.smr_multi.summary())
        return "\n".join(lines)


def _imhof_pvalue(q: float, lam: np.ndarray) -> float:
    """Imhof (1961) / Davies (1980) exact characteristic-function inversion for sum(lam_l * chi2_1) >= q.

    Integrates over exact oscillation half-periods using vectorized 16-point Gauss-Legendre
    quadrature with 32-term Euler binomial acceleration of the alternating Fourier tail.
    """
    if q <= 0.0:
        return 1.0
    if lam.size == 1:
        return float(stats.chi2.sf(q / lam[0], df=1))

    scale = float(np.max(lam))
    lam_s = lam / scale
    x = q / scale

    # Exact half-period of sin(0.5 * x * u) is 2*pi / x; subdivide if needed so total phase shift <= pi
    half_period = (2.0 * np.pi / x) if x > 0.25 else 2.0
    sub_steps = max(1, int(np.ceil(half_period * max(float(np.sum(lam_s)), 1.0) / np.pi)))
    h = half_period / sub_steps
    n_panels = 600 * sub_steps if lam_s.size == 2 else 400 * sub_steps
    n_panels = min(n_panels, 2400)

    k = np.arange(n_panels, dtype=np.float64)[:, None]
    u = k * h + 0.5 * h * (_GL16_NODES[None, :] + 1.0)  # (K, 16), strictly > 0
    ul = u[:, :, None] * lam_s[None, None, :]  # (K, 16, r)
    theta = 0.5 * np.sum(np.arctan(ul), axis=-1) - 0.5 * x * u
    log_rho = 0.25 * np.sum(np.log1p(ul**2), axis=-1)
    integrand = np.where(log_rho < 700.0, np.sin(theta) / (u * np.exp(np.minimum(log_rho, 700.0))), 0.0)
    panel_integrals = 0.5 * h * (integrand @ _GL16_WEIGHTS)

    # Group into exact half-periods for Euler alternating-series tail acceleration
    if sub_steps > 1 and (n_panels // sub_steps) >= _EULER_M + 10:
        n_hp = n_panels // sub_steps
        hp_integrals = np.sum(panel_integrals[: n_hp * sub_steps].reshape(n_hp, sub_steps), axis=1)
        partial = np.cumsum(hp_integrals)
    else:
        partial = np.cumsum(panel_integrals)

    val = float(partial[-_EULER_M:] @ _EULER_WEIGHTS)
    return float(np.clip(0.5 + val / np.pi, 0.0, 1.0))


def _kuonen_saddlepoint_pvalue(q: float, lam: np.ndarray) -> float:
    """Kuonen (1999) Lugannani-Rice saddlepoint approximation for sum(lam_l * chi2_1) >= q."""
    if q <= 0.0:
        return 1.0
    if lam.size == 1:
        return float(stats.chi2.sf(q / lam[0], df=1))

    scale = float(np.max(lam))
    lam_s = lam / scale
    x = q / scale
    mean_x = float(np.sum(lam_s))
    sd_x = float(np.sqrt(2.0 * np.sum(lam_s**2)))

    if abs(x - mean_x) <= 1e-4 * max(sd_x, 1e-12):
        return -1.0  # Signal caller to use fallback right at the mean

    def _K(z: float) -> float:
        return float(-0.5 * np.sum(np.log1p(-2.0 * z * lam_s)))

    def _Kp(z: float) -> float:
        return float(np.sum(lam_s / (1.0 - 2.0 * z * lam_s)))

    def _Kpp(z: float) -> float:
        return float(2.0 * np.sum((lam_s / (1.0 - 2.0 * z * lam_s)) ** 2))

    inv_two_lmax = 0.5 / float(np.max(lam_s))
    if x > mean_x:
        lmin = 0.0
        eps_hi = 1e-6
        lmax = inv_two_lmax * (1.0 - eps_hi)
        while _Kp(lmax) <= x and eps_hi > 1e-15:
            eps_hi *= 0.1
            lmax = inv_two_lmax * (1.0 - eps_hi)
    else:
        lmax = 0.0
        lmin = -0.5 * len(lam_s) / max(x, 1e-300)
        while _Kp(lmin) >= x:
            lmin *= 2.0

    fa = _Kp(lmin) - x
    fb = _Kp(lmax) - x
    if fa * fb >= 0.0:
        return -1.0

    zeta = float(optimize.brentq(lambda z: _Kp(z) - x, lmin, lmax, xtol=1e-12, rtol=1e-12))
    val_term = 2.0 * (zeta * x - _K(zeta))
    if val_term <= 0.0:
        return -1.0
    w = float(np.sign(zeta) * np.sqrt(val_term))
    u = float(zeta * np.sqrt(_Kpp(zeta)))
    if abs(w) < 1e-4 or (u / w) <= 0.0:
        return -1.0
    z_star = w + np.log(u / w) / w
    p = float(stats.norm.sf(z_star))
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        return -1.0
    return p


def _satterthwaite_pvalue(q: float, lam: np.ndarray) -> float:
    """Satterthwaite two-moment scaled chi-square approximation."""
    if q <= 0.0:
        return 1.0
    s1 = float(np.sum(lam))
    s2 = float(np.sum(lam**2))
    if s1 <= 0.0 or s2 <= 0.0:
        return 1.0
    scale = s2 / s1
    df = (s1**2) / s2
    return float(stats.chi2.sf(q / scale, df=df))


def pchisqsum(
    q: float,
    weights: Sequence[float] | np.ndarray,
    *,
    method: QuadMethod | None = None,
    settings: SettingsMode = "default",
) -> float:
    """Evaluate Pr(sum_l weights_l * chi^2_1 >= q) for positive quadratic-form weights.

    Parameters
    ----------
    q : float
        Observed quadratic-form statistic.
    weights : array-like
        Eigenvalues / positive weights of independent 1-df central chi-square variables.
    method : {'auto', 'imhof', 'saddlepoint', 'satterthwaite'}, optional
        In `'default'` settings, defaults to `'auto'` (exact Imhof integration for `p > 1e-7`
        and Kuonen saddlepoint for `p <= 1e-7`, with Satterthwaite fallback).
        In `'R_default'` settings, defaults to Kuonen (1999) saddlepoint approximation with
        Satterthwaite two-moment fallback matching YangLab C++ `StatFunc::pchisqsum`.
    settings : {'default', 'R_default'}, default='default'
        Execution mode (`'default'` or `'R_default'`).
    """
    _validate_settings(settings)
    if not np.isfinite(q):
        raise ValueError("Statistic q must be finite.")
    w_arr = np.asarray(weights, dtype=np.float64).ravel()
    if w_arr.size == 0 or not np.all(np.isfinite(w_arr)):
        raise ValueError("weights must be a non-empty finite array.")
    max_w = float(np.max(w_arr))
    if max_w <= 0.0:
        return 1.0 if q <= 0.0 else 0.0
    if np.min(w_arr) < -1e-6 * max_w:
        raise ValueError("weights must be non-negative eigenvalues of a PSD matrix.")

    pos = w_arr[w_arr > 1e-12 * max_w]
    if q <= 0.0:
        return 1.0
    if pos.size == 1:
        return float(stats.chi2.sf(q / pos[0], df=1))

    if settings == "R_default" and method is None:
        p_sp = _kuonen_saddlepoint_pvalue(q, pos)
        return p_sp if p_sp >= 0.0 else _satterthwaite_pvalue(q, pos)

    eff_method: str = "auto" if method is None else method
    if eff_method == "satterthwaite":
        return _satterthwaite_pvalue(q, pos)
    if eff_method == "imhof":
        return _imhof_pvalue(q, pos)
    if eff_method == "saddlepoint":
        p_sp = _kuonen_saddlepoint_pvalue(q, pos)
        if p_sp >= 0.0:
            return p_sp
        return _satterthwaite_pvalue(q, pos) if settings == "R_default" else _imhof_pvalue(q, pos)
    if eff_method == "auto":
        # Check saddlepoint first if far out in the right tail; use exact Imhof when p > 1e-7
        p_sp = _kuonen_saddlepoint_pvalue(q, pos)
        if 0.0 <= p_sp <= 1e-7:
            return p_sp
        p_im = _imhof_pvalue(q, pos)
        if p_im > 1e-7:
            return p_im
        if p_sp >= 0.0:
            return p_sp
        return _satterthwaite_pvalue(q, pos)
    raise ValueError(f"Unknown pchisqsum method '{eff_method}'.")


def smr_single(
    b_zy: float,
    se_zy: float,
    b_zx: float,
    se_zx: float,
    *,
    snp_id: str = "top_snp",
    settings: SettingsMode = "default",
) -> SMRSingleResult:
    """Compute single-instrument SMR estimate, Delta-method SE, T_SMR, and p_SMR (Zhu et al., 2016)."""
    _validate_settings(settings)
    vals = np.array([b_zy, se_zy, b_zx, se_zx], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("b_zy, se_zy, b_zx, se_zx must all be finite.")
    if se_zy <= 0.0 or se_zx <= 0.0:
        raise ValueError("Standard errors se_zy and se_zx must be strictly positive.")
    if abs(b_zx) == 0.0:
        raise ValueError("Instrument eQTL effect b_zx cannot be zero.")

    b_xy = float(b_zy / b_zx)
    se_xy = float(np.sqrt((b_zy**2 * se_zx**2 + b_zx**2 * se_zy**2) / (b_zx**4)))
    z_zy = float(b_zy / se_zy)
    z_zx = float(b_zx / se_zx)
    denom = z_zy**2 + z_zx**2
    t_smr = float((z_zy**2 * z_zx**2) / denom) if denom > 0.0 else 0.0
    p_smr = float(stats.chi2.sf(t_smr, df=1))

    return SMRSingleResult(
        snp_id=str(snp_id),
        b_xy=b_xy,
        se_xy=se_xy,
        t_smr=t_smr,
        p_smr=p_smr,
        b_zy=float(b_zy),
        se_zy=float(se_zy),
        z_zy=z_zy,
        b_zx=float(b_zx),
        se_zx=float(se_zx),
        z_zx=z_zx,
    )


def _validate_locus_inputs(
    b_zy: Sequence[float] | np.ndarray,
    se_zy: Sequence[float] | np.ndarray,
    b_zx: Sequence[float] | np.ndarray,
    se_zx: Sequence[float] | np.ndarray,
    ld_matrix: np.ndarray,
    snp_ids: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    bzy = np.asarray(b_zy, dtype=np.float64).ravel()
    szy = np.asarray(se_zy, dtype=np.float64).ravel()
    bzx = np.asarray(b_zx, dtype=np.float64).ravel()
    szx = np.asarray(se_zx, dtype=np.float64).ravel()
    R = np.asarray(ld_matrix, dtype=np.float64)

    m = bzy.size
    if m == 0 or szy.size != m or bzx.size != m or szx.size != m:
        raise ValueError("b_zy, se_zy, b_zx, se_zx must have identical non-zero length.")
    if R.shape != (m, m):
        raise ValueError(f"ld_matrix must have shape ({m}, {m}), got {R.shape}.")
    if not (
        np.all(np.isfinite(bzy))
        and np.all(np.isfinite(szy))
        and np.all(np.isfinite(bzx))
        and np.all(np.isfinite(szx))
        and np.all(np.isfinite(R))
    ):
        raise ValueError("Summary statistics and ld_matrix must be finite (no NaN or Inf).")
    if np.any(szy <= 0.0) or np.any(szx <= 0.0):
        raise ValueError("se_zy and se_zx must be strictly positive.")
    if not np.allclose(R, R.T, atol=1e-6):
        raise ValueError("ld_matrix must be symmetric.")
    if not np.allclose(np.diag(R), 1.0, atol=1e-5):
        raise ValueError("ld_matrix must have unit diagonal.")
    if np.any(np.abs(R) > 1.0 + 1e-6):
        raise ValueError("ld_matrix entries must lie in [-1, 1].")

    ids = [f"snp_{i + 1}" for i in range(m)] if snp_ids is None else [str(s) for s in snp_ids]
    if len(ids) != m:
        raise ValueError("snp_ids length must match number of SNPs.")
    R_sym = 0.5 * (R + R.T)
    np.fill_diagonal(R_sym, 1.0)
    return bzy, szy, bzx, szx, R_sym, ids


def est_cov_bxy(
    b_zy: Sequence[float] | np.ndarray,
    se_zy: Sequence[float] | np.ndarray,
    b_zx: Sequence[float] | np.ndarray,
    se_zx: Sequence[float] | np.ndarray,
    ld_matrix: np.ndarray,
    *,
    settings: SettingsMode = "default",
) -> np.ndarray:
    """Compute m x m sampling covariance Cov(b_xy(i), b_xy(j)) across dependent cis-instruments.

    In `'default'` mode, uses the second-order expression of Zhu et al. (2016) Eq. 9.
    In `'R_default'` mode, includes the upstream YangLab `SMR_data.cpp` fourth-order
    `- b_xy_i * b_xy_j / (z_zx_i^2 * z_zx_j^2)` bias-product correction plus `1e-8` diagonal ridge.
    """
    _validate_settings(settings)
    bzy, szy, bzx, szx, R_sub, _ = _validate_locus_inputs(b_zy, se_zy, b_zx, se_zx, ld_matrix, None)
    if np.any(bzx == 0.0):
        raise ValueError("eQTL effects b_zx must be non-zero in est_cov_bxy.")
    bxy = bzy / bzx
    z_zx = bzx / szx
    ratio_gwas = szy / bzx
    if settings == "default":
        ratio_eqtl = bxy / z_zx
        return R_sub * np.outer(ratio_gwas, ratio_gwas) + R_sub * np.outer(ratio_eqtl, ratio_eqtl)

    z_zx2 = z_zx**2
    cov = R_sub * np.outer(ratio_gwas, ratio_gwas) + np.outer(bxy, bxy) * (
        R_sub / np.outer(z_zx, z_zx) - 1.0 / np.outer(z_zx2, z_zx2)
    )
    cov[np.diag_indices_from(cov)] += 1e-8
    return cov


def _cov_bxy_zhu2016(
    bzy: np.ndarray,
    szy: np.ndarray,
    bzx: np.ndarray,
    szx: np.ndarray,
    R_sub: np.ndarray,
) -> np.ndarray:
    """Compute exact m x m sampling covariance Cov(b_xy(i), b_xy(j)) per Zhu et al. (2016) Eq. 9."""
    return est_cov_bxy(bzy, szy, bzx, szx, R_sub, settings="default")


def _rm_cor_sbat(
    R_cand: np.ndarray,
    z_abs: np.ndarray,
    r2_thresh: float,
    protected: set[int] | None = None,
) -> list[int]:
    """YangLab `rm_cor_sbat` degree-based iterative LD pruning.

    Iteratively counts how many remaining SNPs each active SNP has `|r| > sqrt(r2_thresh)` with,
    removing the unprotected SNP with the highest degree (> 0) and breaking ties by removing the
    SNP with smaller `|z_zx|` (and later position on exact ties), until max degree is 0.
    """
    k = R_cand.shape[0]
    if k <= 1:
        return list(range(k))
    r_max = float(np.sqrt(r2_thresh))
    prot = set() if protected is None else set(protected)
    active = np.ones(k, dtype=bool)
    adj = np.abs(R_cand) > r_max
    np.fill_diagonal(adj, False)

    # Ensure no unprotected SNP violates r_max against a protected SNP
    for p_idx in prot:
        if 0 <= p_idx < k and active[p_idx]:
            violators = np.flatnonzero(adj[p_idx] & active)
            for v_idx in violators:
                if int(v_idx) not in prot:
                    active[v_idx] = False

    while True:
        act_idx = np.flatnonzero(active)
        if act_idx.size <= 1:
            break
        deg = np.sum(adj[np.ix_(act_idx, act_idx)], axis=1)
        if int(np.max(deg)) == 0:
            break
        unprot_mask = np.array([int(i) not in prot for i in act_idx], dtype=bool)
        unprot_pos = np.flatnonzero(unprot_mask & (deg > 0))
        if unprot_pos.size == 0:
            break
        max_deg = int(np.max(deg[unprot_pos]))
        ties = act_idx[unprot_pos[deg[unprot_pos] == max_deg]]
        rm_idx = int(min(ties, key=lambda idx: (float(z_abs[int(idx)]), -int(idx))))
        active[rm_idx] = False

    return [int(i) for i in np.flatnonzero(active)]


def heidi_test(
    b_zy: Sequence[float] | np.ndarray,
    se_zy: Sequence[float] | np.ndarray,
    b_zx: Sequence[float] | np.ndarray,
    se_zx: Sequence[float] | np.ndarray,
    ld_matrix: np.ndarray,
    *,
    snp_ids: Sequence[str] | None = None,
    top_idx: int | None = None,
    p_het: float = 1.57e-3,
    ld_top_r2: float = 0.9,
    ld_min_r2: float = 0.05,
    max_snps: int = 20,
    min_snps: int = 3,
    p_method: QuadMethod | None = None,
    settings: SettingsMode = "default",
) -> HeidiResult:
    """Run the HEIDI test for heterogeneity in dependent cis-instruments (Zhu et al., 2016 Eqs. 7-11)."""
    _validate_settings(settings)
    if not (0.0 < p_het <= 1.0):
        raise ValueError("p_het must lie in (0, 1].")
    if not (0.0 <= ld_min_r2 < ld_top_r2 <= 1.0):
        raise ValueError("Require 0 <= ld_min_r2 < ld_top_r2 <= 1.")
    if max_snps < 2 or min_snps < 2:
        raise ValueError("max_snps and min_snps must be >= 2.")

    bzy, szy, bzx, szx, R, ids = _validate_locus_inputs(b_zy, se_zy, b_zx, se_zx, ld_matrix, snp_ids)
    m_total = bzy.size
    z_zx = bzx / szx
    z_zx_sq = z_zx**2
    p_eqtl = stats.chi2.sf(z_zx_sq, df=1)

    if top_idx is None:
        t_idx = int(np.argmax(z_zx_sq))
    else:
        t_idx = int(top_idx)
        if not (0 <= t_idx < m_total):
            raise ValueError(f"top_idx {t_idx} out of bounds [0, {m_total}).")
    if abs(bzx[t_idx]) == 0.0:
        raise ValueError("Top instrument has zero eQTL effect b_zx.")

    # Candidate cis-SNPs passing p_eQTL < p_het and LD window [ld_min_r2, ld_top_r2] wrt top cis-eQTL
    r2_with_top = R[:, t_idx] ** 2
    cand_mask = (
        (np.arange(m_total) != t_idx)
        & (p_eqtl < p_het)
        & (np.abs(bzx) > 0.0)
        & (r2_with_top >= ld_min_r2)
        & (r2_with_top <= ld_top_r2)
    )
    cand_indices = np.flatnonzero(cand_mask)
    cand_ordered = cand_indices[np.argsort(-z_zx_sq[cand_indices], kind="mergesort")]

    if settings == "R_default":
        cand_all = np.concatenate([np.array([t_idx], dtype=int), cand_ordered.astype(int)])
        kept_rel = _rm_cor_sbat(R[np.ix_(cand_all, cand_all)], np.abs(z_zx[cand_all]), ld_top_r2, protected={0})
        selected = [int(cand_all[i]) for i in kept_rel[:max_snps]]
    else:
        # Greedy pairwise LD pruning (r^2 <= ld_top_r2) up to max_snps total SNPs
        selected = [t_idx]
        for c in cand_ordered:
            if len(selected) >= max_snps:
                break
            r2_to_sel = R[int(c), selected] ** 2
            if np.all(r2_to_sel <= ld_top_r2):
                selected.append(int(c))

    sel_arr = np.asarray(selected, dtype=int)
    sel_snps = [ids[i] for i in selected]
    bxy_sel = bzy[sel_arr] / bzx[sel_arr]

    if len(selected) < min_snps:
        return HeidiResult(
            top_snp=ids[t_idx],
            top_idx=t_idx,
            n_snps=len(selected),
            selected_indices=selected,
            selected_snps=sel_snps,
            t_heidi=float("nan"),
            p_heidi=float("nan"),
            b_xy_selected=bxy_sel,
            d_values=bxy_sel[1:] - bxy_sel[0],
            z_d=np.array([], dtype=np.float64),
            corr_d=np.empty((0, 0), dtype=np.float64),
            eigenvalues=np.array([], dtype=np.float64),
        )

    R_sub = R[np.ix_(sel_arr, sel_arr)]
    cov_bxy = est_cov_bxy(bzy[sel_arr], szy[sel_arr], bzx[sel_arr], szx[sel_arr], R_sub, settings=settings)

    # Deviation d_i = b_xy(i) - b_xy(top) for i = 1..k (k = m - 1)
    d_vec = bxy_sel[1:] - bxy_sel[0]
    cov_d_full = cov_bxy + cov_bxy[0, 0] - cov_bxy[:, [0]] - cov_bxy[[0], :]
    V_d = cov_d_full[1:, 1:]
    diag_vd = np.diag(V_d)
    if np.any(diag_vd <= 0.0):
        raise ValueError("Non-positive variance encountered in HEIDI deviation covariance V_d.")

    sd_d = np.sqrt(diag_vd)
    z_d = d_vec / sd_d
    psi = V_d / np.outer(sd_d, sd_d)
    psi = 0.5 * (psi + psi.T)
    np.fill_diagonal(psi, 1.0)

    t_heidi = float(np.sum(z_d**2))
    eigs = np.linalg.eigvalsh(psi)
    eigs_clamped = np.maximum(eigs, 0.0)
    p_heidi = pchisqsum(t_heidi, eigs_clamped, method=p_method, settings=settings)

    return HeidiResult(
        top_snp=ids[t_idx],
        top_idx=t_idx,
        n_snps=len(selected),
        selected_indices=selected,
        selected_snps=sel_snps,
        t_heidi=t_heidi,
        p_heidi=p_heidi,
        b_xy_selected=bxy_sel,
        d_values=d_vec,
        z_d=z_d,
        corr_d=psi,
        eigenvalues=eigs_clamped,
    )


def smr_multi(
    b_zy: Sequence[float] | np.ndarray,
    se_zy: Sequence[float] | np.ndarray,
    b_zx: Sequence[float] | np.ndarray,
    se_zx: Sequence[float] | np.ndarray,
    ld_matrix: np.ndarray,
    *,
    snp_ids: Sequence[str] | None = None,
    p_eqtl_thresh: float = 5e-8,
    ld_prune_r2: float = 0.1,
    max_snps: int | None = None,
    p_method: QuadMethod | None = None,
    settings: SettingsMode = "default",
) -> SMRMultiResult:
    """Run multi-cis-SNP SMR omnibus test across LD-pruned cis-xQTLs (Wu et al., Nat Commun 2018)."""
    _validate_settings(settings)
    if not (0.0 < p_eqtl_thresh <= 1.0):
        raise ValueError("p_eqtl_thresh must lie in (0, 1].")
    if not (0.0 < ld_prune_r2 <= 1.0):
        raise ValueError("ld_prune_r2 must lie in (0, 1].")

    bzy, szy, bzx, szx, R, ids = _validate_locus_inputs(b_zy, se_zy, b_zx, se_zx, ld_matrix, snp_ids)
    z_zx = bzx / szx
    z_zy = bzy / szy
    z_zx_sq = z_zx**2
    p_eqtl = stats.chi2.sf(z_zx_sq, df=1)

    cand = np.flatnonzero((p_eqtl < p_eqtl_thresh) & (np.abs(bzx) > 0.0))
    if cand.size == 0:
        raise ValueError(f"No cis-SNPs passed p_eqtl_thresh = {p_eqtl_thresh:g}.")

    ordered = cand[np.argsort(-z_zx_sq[cand], kind="mergesort")]
    if settings == "R_default":
        kept_rel = _rm_cor_sbat(R[np.ix_(ordered, ordered)], np.abs(z_zx[ordered]), ld_prune_r2)
        selected = [int(ordered[i]) for i in kept_rel]
        if max_snps is not None:
            selected = selected[:max_snps]
    else:
        selected = []
        for c in ordered:
            if max_snps is not None and len(selected) >= max_snps:
                break
            if not selected or np.all((R[int(c), selected] ** 2) <= ld_prune_r2):
                selected.append(int(c))

    sel_arr = np.asarray(selected, dtype=int)
    z_zx2_sel = z_zx_sq[sel_arr]
    z_zy2_sel = z_zy[sel_arr] ** 2
    t_smr_vec = (z_zy2_sel * z_zx2_sel) / (z_zy2_sel + z_zx2_sel)
    t_sum = float(np.sum(t_smr_vec))
    t_mean = float(np.mean(t_smr_vec))

    R_sub = R[np.ix_(sel_arr, sel_arr)]
    eigs = np.maximum(np.linalg.eigvalsh(0.5 * (R_sub + R_sub.T)), 0.0)
    p_multi = pchisqsum(t_sum, eigs, method=p_method, settings=settings)

    # Inverse-covariance GLS pooled b_xy across the selected cis-SNPs
    bxy_sel = bzy[sel_arr] / bzx[sel_arr]
    cov_bxy = est_cov_bxy(bzy[sel_arr], szy[sel_arr], bzx[sel_arr], szx[sel_arr], R_sub, settings=settings)
    ones = np.ones(len(selected), dtype=np.float64)
    if settings == "R_default":
        cinv_ones = np.linalg.solve(cov_bxy, ones)
    else:
        reg = 1e-10 * max(float(np.mean(np.diag(cov_bxy))), 1e-12) * np.eye(len(selected))
        cinv_ones = np.linalg.solve(cov_bxy + reg, ones)
    var_gls = float(1.0 / np.sum(cinv_ones))
    b_gls = float(var_gls * (cinv_ones @ bxy_sel))

    return SMRMultiResult(
        n_snps=len(selected),
        selected_indices=selected,
        selected_snps=[ids[i] for i in selected],
        t_smr_per_snp=t_smr_vec,
        t_smr_sum=t_sum,
        t_smr_mean=t_mean,
        p_smr_multi=p_multi,
        b_xy_gls=b_gls,
        se_xy_gls=float(np.sqrt(max(var_gls, 0.0))),
        eigenvalues=eigs,
    )


def _find_col(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"Could not find {label} column in DataFrame (tried {list(candidates)}).")


def run_locus_smr_heidi(
    gwas_df: pd.DataFrame,
    eqtl_df: pd.DataFrame,
    ld_matrix: np.ndarray,
    *,
    ld_snps: Sequence[str] | None = None,
    ld_a1: Sequence[str] | None = None,
    ld_a2: Sequence[str] | None = None,
    probe_id: str = "gene",
    p_het: float = 1.57e-3,
    ld_top_r2: float = 0.9,
    ld_min_r2: float = 0.05,
    max_heidi_snps: int = 20,
    min_heidi_snps: int = 3,
    p_smr_multi_thresh: float = 5e-8,
    ld_multi_r2: float = 0.1,
    drop_incompatible: bool = True,
    p_method: QuadMethod | None = None,
    settings: SettingsMode = "default",
) -> LocusSMRResult:
    """Harmonize GWAS + xQTL DataFrames + signed LD matrix `R` and execute SMR, HEIDI, and SMR-multi."""
    _validate_settings(settings)
    g_snp = _find_col(gwas_df, ("snp", "rsid", "id", "variant_id"), "GWAS SNP")
    g_a1 = _find_col(gwas_df, ("a1", "ea", "effect_allele", "alt"), "GWAS A1")
    g_a2 = _find_col(gwas_df, ("a2", "oa", "other_allele", "ref"), "GWAS A2")
    g_b = _find_col(gwas_df, ("beta", "b", "b_zy", "effect"), "GWAS beta")
    g_se = _find_col(gwas_df, ("se", "se_zy", "stderr"), "GWAS se")

    e_snp = _find_col(eqtl_df, ("snp", "rsid", "id", "variant_id"), "eQTL SNP")
    e_a1 = _find_col(eqtl_df, ("a1", "ea", "effect_allele", "alt"), "eQTL A1")
    e_a2 = _find_col(eqtl_df, ("a2", "oa", "other_allele", "ref"), "eQTL A2")
    e_b = _find_col(eqtl_df, ("beta", "b", "b_zx", "effect"), "eQTL beta")
    e_se = _find_col(eqtl_df, ("se", "se_zx", "stderr"), "eQTL se")

    R_raw = np.asarray(ld_matrix, dtype=np.float64)
    if R_raw.ndim != 2 or R_raw.shape[0] != R_raw.shape[1]:
        raise ValueError("ld_matrix must be a square 2D array.")

    eqtl_clean = eqtl_df[[e_snp, e_a1, e_a2, e_b, e_se]].copy()
    eqtl_clean.columns = ["snp", "a1_eqtl", "a2_eqtl", "b_zx", "se_zx"]
    eqtl_clean["snp"] = eqtl_clean["snp"].astype(str)
    eqtl_clean["a1_eqtl"] = eqtl_clean["a1_eqtl"].astype(str).str.upper()
    eqtl_clean["a2_eqtl"] = eqtl_clean["a2_eqtl"].astype(str).str.upper()

    if ld_snps is None:
        if len(eqtl_clean) != R_raw.shape[0]:
            raise ValueError("When ld_snps is omitted, ld_matrix dimension must equal len(eqtl_df).")
        ld_snp_list = list(eqtl_clean["snp"])
    else:
        ld_snp_list = [str(s) for s in ld_snps]
        if len(ld_snp_list) != R_raw.shape[0]:
            raise ValueError("ld_snps length must match ld_matrix dimension.")

    if (ld_a1 is None) != (ld_a2 is None):
        raise ValueError("Provide both ld_a1 and ld_a2 or neither.")
    if ld_a1 is not None and ld_a2 is not None:
        ld_a1_up = [str(a).upper() for a in ld_a1]
        ld_a2_up = [str(a).upper() for a in ld_a2]
        if len(ld_a1_up) != R_raw.shape[0] or len(ld_a2_up) != R_raw.shape[0]:
            raise ValueError("ld_a1 and ld_a2 lengths must match ld_matrix dimension.")
    else:
        ld_a1_up = None
        ld_a2_up = None

    ld_idx_map = {s: idx for idx, s in enumerate(ld_snp_list)}

    gwas_clean = gwas_df[[g_snp, g_a1, g_a2, g_b, g_se]].copy()
    gwas_clean.columns = ["snp", "a1_gwas", "a2_gwas", "b_zy", "se_zy"]
    gwas_clean["snp"] = gwas_clean["snp"].astype(str)
    gwas_clean["a1_gwas"] = gwas_clean["a1_gwas"].astype(str).str.upper()
    gwas_clean["a2_gwas"] = gwas_clean["a2_gwas"].astype(str).str.upper()

    merged = pd.merge(eqtl_clean, gwas_clean, on="snp", how="inner")
    merged = merged[merged["snp"].isin(ld_idx_map)].reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping SNPs found across gwas_df, eqtl_df, and ld_matrix.")

    keep_rows: list[int] = []
    gwas_signs: list[float] = []
    ld_signs: list[float] = []
    ld_indices: list[int] = []

    for row_i, row in merged.iterrows():
        ea1, ea2 = row["a1_eqtl"], row["a2_eqtl"]
        ga1, ga2 = row["a1_gwas"], row["a2_gwas"]

        if ga1 == ea1 and ga2 == ea2:
            g_sign = 1.0
        elif ga1 == ea2 and ga2 == ea1:
            g_sign = -1.0
        elif drop_incompatible:
            continue
        else:
            raise ValueError(f"Incompatible GWAS/eQTL alleles at {row['snp']}: ({ga1},{ga2}) vs ({ea1},{ea2}).")

        l_idx = ld_idx_map[row["snp"]]
        if ld_a1_up is not None and ld_a2_up is not None:
            la1, la2 = ld_a1_up[l_idx], ld_a2_up[l_idx]
            if la1 == ea1 and la2 == ea2:
                l_sign = 1.0
            elif la1 == ea2 and la2 == ea1:
                l_sign = -1.0
            elif drop_incompatible:
                continue
            else:
                raise ValueError(
                    f"Incompatible LD/eQTL alleles at {row['snp']}: ({la1},{la2}) vs ({ea1},{ea2})."
                )
        else:
            l_sign = 1.0

        keep_rows.append(int(row_i))
        gwas_signs.append(g_sign)
        ld_signs.append(l_sign)
        ld_indices.append(l_idx)

    if not keep_rows:
        raise ValueError("No allele-compatible SNPs remained after harmonization.")

    harm = merged.iloc[keep_rows].reset_index(drop=True).copy()
    g_sign_arr = np.asarray(gwas_signs, dtype=np.float64)
    l_sign_arr = np.asarray(ld_signs, dtype=np.float64)
    harm["b_zy"] = harm["b_zy"].to_numpy(dtype=np.float64) * g_sign_arr

    ld_idx_arr = np.asarray(ld_indices, dtype=int)
    R_harm = R_raw[np.ix_(ld_idx_arr, ld_idx_arr)] * np.outer(l_sign_arr, l_sign_arr)
    np.fill_diagonal(R_harm, 1.0)

    bzy = harm["b_zy"].to_numpy(dtype=np.float64)
    szy = harm["se_zy"].to_numpy(dtype=np.float64)
    bzx = harm["b_zx"].to_numpy(dtype=np.float64)
    szx = harm["se_zx"].to_numpy(dtype=np.float64)
    snps = list(harm["snp"])

    top_idx = int(np.argmax((bzx / szx) ** 2))
    smr_res = smr_single(
        bzy[top_idx], szy[top_idx], bzx[top_idx], szx[top_idx], snp_id=snps[top_idx], settings=settings
    )
    heidi_res = heidi_test(
        bzy,
        szy,
        bzx,
        szx,
        R_harm,
        snp_ids=snps,
        top_idx=top_idx,
        p_het=p_het,
        ld_top_r2=ld_top_r2,
        ld_min_r2=ld_min_r2,
        max_snps=max_heidi_snps,
        min_snps=min_heidi_snps,
        p_method=p_method,
        settings=settings,
    )

    p_eqtl_min = float(np.min(stats.chi2.sf((bzx / szx) ** 2, df=1)))
    if p_eqtl_min < p_smr_multi_thresh:
        multi_res: SMRMultiResult | None = smr_multi(
            bzy,
            szy,
            bzx,
            szx,
            R_harm,
            snp_ids=snps,
            p_eqtl_thresh=p_smr_multi_thresh,
            ld_prune_r2=ld_multi_r2,
            p_method=p_method,
            settings=settings,
        )
    else:
        multi_res = None

    harm_out = pd.DataFrame(
        {
            "snp": snps,
            "a1": harm["a1_eqtl"],
            "a2": harm["a2_eqtl"],
            "b_zy": bzy,
            "se_zy": szy,
            "b_zx": bzx,
            "se_zx": szx,
        }
    )

    return LocusSMRResult(
        probe_id=probe_id,
        n_harmonized_snps=len(snps),
        n_flipped_gwas=int(np.sum(g_sign_arr < 0.0)),
        n_flipped_ld=int(np.sum(l_sign_arr < 0.0)),
        smr=smr_res,
        heidi=heidi_res,
        smr_multi=multi_res,
        harmonized_snps=harm_out,
    )


smr_heidi_locus = run_locus_smr_heidi
