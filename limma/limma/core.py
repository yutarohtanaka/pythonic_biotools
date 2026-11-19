"""
Pythonic reimplementation of limma's empirical Bayes variance moderation
(Smyth 2004; Law et al. 2014; Phipson et al. 2016).

Implements exact moment-matching (`fitFDist` / `squeezeVar`) on a scaled
inverse-chi-squared prior via vectorized Newton-Raphson trigamma inversion,
supporting:
  - `moderated_ttest`: fast two-group differential analysis across features
  - `moderated_lm_contrast`: general linear model design + contrast moderation
  - `trend=True` (`limma_trend`): locally linear tricube LOWESS (`"default"`) or
    4-df natural cubic regression spline (`settings="R_default"`) mean-variance prior
  - `robust=True`: Phipson et al. (2016) winsorized prior & outlier `df_prior` downweighting
  - Smyth (2004) empirical Bayes `B` log-odds statistic (`b_stat` / `lods`)
  - `limma_top_table`: tidy DataFrame output mirroring R's `limma::topTable`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import betaincinv, betaln, digamma, gammaln, polygamma
from scipy.stats import chi2, f as f_dist, false_discovery_control, norm
from scipy.stats import t as t_dist

SettingsMode = Literal["default", "R_default"]

_GL128_NODES, _GL128_WEIGHTS = np.polynomial.legendre.leggauss(128)


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(f"settings must be 'default' or 'R_default', got {settings!r}.")
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class ModeratedTestResult:
    """Result container for limma-style empirical Bayes moderated hypothesis testing."""

    feature_names: list[str]
    log_fc: np.ndarray
    ave_expr: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    adj_p_val: np.ndarray
    var_prior: float | np.ndarray
    df_prior: float | np.ndarray
    var_post: np.ndarray
    b_stat: np.ndarray | None = None

    @property
    def lods(self) -> np.ndarray | None:
        """Alias for Smyth (2004) empirical Bayes `B` log-posterior-odds statistic (`limma`'s `lods`)."""
        return self.b_stat

    def to_dataframe(self, sort_by_pvalue: bool = True) -> pd.DataFrame:
        """Format results as a `limma::topTable`-compatible DataFrame."""
        cols: dict[str, np.ndarray] = {
            "logFC": self.log_fc,
            "AveExpr": self.ave_expr,
            "t": self.t_stat,
            "P.Value": self.p_value,
            "adj.P.Val": self.adj_p_val,
        }
        if self.b_stat is not None:
            cols["B"] = self.b_stat
        cols["s2_post"] = self.var_post
        df = pd.DataFrame(cols, index=self.feature_names)
        if sort_by_pvalue:
            df = df.sort_values("P.Value", kind="mergesort", na_position="last")
        return df


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Invert the trigamma function `y -> polygamma(1, y)` on `(0, inf)`.

    Returns `0.0` for `+np.inf`, `np.nan` for `np.nan` or `<= 0`, preserves input shape,
    and uses 4 vectorized Newton-Raphson iterations on convex `1 / psi_1(y)` (Smyth 2004).
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)

    pos_inf = arr == np.inf
    if np.any(pos_inf):
        out[pos_inf] = 0.0

    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return out

    xm = arr[valid]
    ym = np.empty_like(xm)
    large = xm > 1e7
    small = xm < 1e-6
    mid = ~(large | small)

    ym[large] = 1.0 / np.sqrt(xm[large])
    ym[small] = 1.0 / xm[small] + 0.5
    if np.any(mid):
        x_mid = xm[mid]
        y = np.where(x_mid > 10.0, 1.0 / np.sqrt(x_mid), 0.5 + 1.0 / x_mid)
        for _ in range(4):
            tri = polygamma(1, y)
            y = y + tri * (1.0 - tri / x_mid) / polygamma(2, y)
        ym[mid] = y

    out[valid] = ym
    return out


def _lowess_tricube_1d(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_eval: np.ndarray,
    span: float = 0.5,
    n_anchors: int = 50,
) -> np.ndarray:
    """Fast Cleveland tricube locally-linear trend evaluated on quantile anchors and interpolated."""
    n = x_obs.size
    if n < 4 or np.nanmax(x_obs) <= np.nanmin(x_obs):
        return np.full(x_eval.shape, float(np.mean(y_obs)), dtype=np.float64)

    m_anc = min(n, n_anchors)
    qs = np.linspace(0.0, 1.0, m_anc)
    anchors = np.unique(np.quantile(x_obs, qs))
    if anchors.size < 2:
        return np.full(x_eval.shape, float(np.mean(y_obs)), dtype=np.float64)

    k = max(4, min(n, int(np.ceil(span * n))))
    anchor_fits = np.empty_like(anchors, dtype=np.float64)

    for idx, a0 in enumerate(anchors):
        dx = x_obs - a0
        adx = np.abs(dx)
        h = float(np.partition(adx, k - 1)[k - 1])
        if h <= 0.0:
            h = float(np.max(adx))
        if h <= 0.0:
            anchor_fits[idx] = float(np.mean(y_obs))
            continue
        u = np.clip(adx / (h * 1.0001), 0.0, 1.0)
        w = (1.0 - u**3) ** 3
        sw = float(np.sum(w))
        sx = float(np.sum(w * dx))
        sy = float(np.sum(w * y_obs))
        sxx = float(np.sum(w * dx * dx))
        sxy = float(np.sum(w * dx * y_obs))
        denom = sw * sxx - sx * sx
        if denom > 1e-12 * sw * max(sxx, 1e-30):
            anchor_fits[idx] = (sxx * sy - sx * sxy) / denom
        else:
            anchor_fits[idx] = sy / max(sw, 1e-30)

    out = np.interp(np.where(np.isfinite(x_eval), x_eval, float(np.median(x_obs))), anchors, anchor_fits)
    return out


def _build_ns_basis(x_scaled: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Exact natural cubic spline basis spanning `splines::ns(x, df)` with intercept."""
    m_k = knots.size
    if m_k < 2:
        return np.ones((x_scaled.size, 1), dtype=np.float64)
    if m_k == 2:
        return np.column_stack([np.ones_like(x_scaled), x_scaled])

    k_last = float(knots[-1])
    k_penult = float(knots[-2])
    d_penult = (np.maximum(x_scaled - k_penult, 0.0) ** 3 - np.maximum(x_scaled - k_last, 0.0) ** 3) / max(
        k_last - k_penult, 1e-12
    )
    cols = [np.ones_like(x_scaled), x_scaled]
    for idx in range(m_k - 2):
        ki = float(knots[idx])
        d_i = (np.maximum(x_scaled - ki, 0.0) ** 3 - np.maximum(x_scaled - k_last, 0.0) ** 3) / max(
            k_last - ki, 1e-12
        )
        cols.append(d_i - d_penult)
    return np.column_stack(cols)


def _ns_regression_trend_1d(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_eval: np.ndarray,
    df: int | None = None,
) -> tuple[np.ndarray, int]:
    """Fit R `limma::fitFDist` natural cubic regression spline `ns(covariate, df=df)` (`df=min(4, max(1, G//10))`)."""
    n = x_obs.size
    df_eff = min(4, max(1, n // 10)) if df is None else int(df)
    x_min = float(np.min(x_obs))
    x_max = float(np.max(x_obs))
    if n < 3 or x_max <= x_min:
        return np.full(x_eval.shape, float(np.mean(y_obs)), dtype=np.float64), 1

    span = x_max - x_min
    x_obs_sc = (x_obs - x_min) / span
    x_med = float(np.median(x_obs))
    x_eval_clean = np.where(np.isfinite(x_eval), x_eval, x_med)
    x_eval_sc = (x_eval_clean - x_min) / span

    knots = np.unique(np.quantile(x_obs_sc, np.linspace(0.0, 1.0, df_eff + 1)))
    B_obs = _build_ns_basis(x_obs_sc, knots)
    B_eval = _build_ns_basis(x_eval_sc, knots)
    coef, _, rank, _ = np.linalg.lstsq(B_obs, y_obs, rcond=None)
    return (B_eval @ coef).reshape(x_eval.shape), int(max(rank, 1))


def _winsorized_gaussian_moments(
    r: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[float, float]:
    """Consistent mean and variance from asymmetric Winsorized residuals (Gaussian approximation)."""
    p_lo, p_hi = float(winsor_tail_p[0]), float(winsor_tail_p[1])
    if not (0.0 <= p_lo < 0.5 and 0.0 <= p_hi < 0.5 and (p_lo + p_hi) < 0.9):
        raise ValueError("Invalid winsor_tail_p; require 0 <= p_lo, p_hi < 0.5 and p_lo + p_hi < 0.9.")

    q_lo, q_hi = np.quantile(r, [p_lo, 1.0 - p_hi])
    r_win = np.clip(r, q_lo, q_hi)
    mean_win = float(np.mean(r_win))
    var_win = float(np.sum((r_win - mean_win) ** 2) / max(r.size - 1, 1))

    z_lo = float(norm.ppf(p_lo)) if p_lo > 0.0 else -np.inf
    z_hi = float(norm.ppf(1.0 - p_hi)) if p_hi > 0.0 else np.inf
    phi_lo = float(norm.pdf(z_lo)) if np.isfinite(z_lo) else 0.0
    phi_hi = float(norm.pdf(z_hi)) if np.isfinite(z_hi) else 0.0

    t1_lo = z_lo * p_lo if p_lo > 0.0 else 0.0
    t1_hi = z_hi * p_hi if p_hi > 0.0 else 0.0
    mu_w = t1_lo + t1_hi + phi_lo - phi_hi

    t2_lo = (z_lo**2) * p_lo + z_lo * phi_lo if p_lo > 0.0 else 0.0
    t2_hi = (z_hi**2) * p_hi - z_hi * phi_hi if p_hi > 0.0 else 0.0
    ez2_w = t2_lo + t2_hi + (1.0 - p_lo - p_hi)
    var_w = max(ez2_w - mu_w**2, 1e-8)

    sigma_hat = np.sqrt(var_win / var_w)
    mu_hat = mean_win - sigma_hat * mu_w
    return float(mu_hat), float(sigma_hat**2)


def _log_f_ppf(p: float | np.ndarray, d1: float, d2: float) -> float | np.ndarray:
    """Overflow-free `log(f_dist.ppf(p, d1, d2))` via complementary `betaincinv`."""
    p_arr = np.asarray(p, dtype=np.float64)
    a, b = 0.5 * d1, 0.5 * d2
    y1 = np.maximum(betaincinv(a, b, p_arr), 1e-300)
    y2 = np.maximum(betaincinv(b, a, 1.0 - p_arr), 1e-300)
    res = np.log(d2 / d1) + np.log(y1) - np.log(y2)
    return float(res) if np.ndim(p) == 0 else res


def _winsorized_log_f_moments_gl(
    d1: float,
    d2: float,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[float, float]:
    """Exact theoretical mean and variance of Winsorized `log F(d1, d2)` via 128-node Gauss-Legendre quadrature."""
    p_lo, p_hi = float(winsor_tail_p[0]), float(winsor_tail_p[1])
    if not (0.0 <= p_lo < 0.5 and 0.0 <= p_hi < 0.5 and (p_lo + p_hi) < 0.9):
        raise ValueError("Invalid winsor_tail_p; require 0 <= p_lo, p_hi < 0.5 and p_lo + p_hi < 0.9.")

    if p_lo > 0.0 and p_hi > 0.0:
        if np.isinf(d2):
            q_lo = float(np.log(chi2.ppf(p_lo, d1) / d1))
            q_hi = float(np.log(chi2.ppf(1.0 - p_hi, d1) / d1))
            mid = 0.5 * (q_hi + q_lo)
            half = 0.5 * (q_hi - q_lo)
            z = mid + half * _GL128_NODES
            log_pdf = 0.5 * d1 * np.log(0.5 * d1) - gammaln(0.5 * d1) + 0.5 * d1 * z - 0.5 * d1 * np.exp(z)
        else:
            q_lo = float(_log_f_ppf(p_lo, d1, d2))
            q_hi = float(_log_f_ppf(1.0 - p_hi, d1, d2))
            mid = 0.5 * (q_hi + q_lo)
            half = 0.5 * (q_hi - q_lo)
            z = mid + half * _GL128_NODES
            log_ratio = np.log(d1 / d2) + z
            log_pdf = 0.5 * d1 * log_ratio - betaln(0.5 * d1, 0.5 * d2) - 0.5 * (d1 + d2) * np.logaddexp(0.0, log_ratio)
        w = half * _GL128_WEIGHTS * np.exp(log_pdf)
        mu = p_lo * q_lo + p_hi * q_hi + float(np.sum(w * z))
        ez2 = p_lo * (q_lo**2) + p_hi * (q_hi**2) + float(np.sum(w * (z**2)))
        return float(mu), float(max(ez2 - mu**2, 1e-16))

    # Quantile-domain Gauss-Legendre integration when one tail has 0 Winsorization
    u_a, u_b = p_lo, 1.0 - p_hi
    u_mid = 0.5 * (u_a + u_b)
    u_half = 0.5 * (u_b - u_a)
    u_nodes = u_mid + u_half * _GL128_NODES
    w_u = u_half * _GL128_WEIGHTS
    if np.isinf(d2):
        z_nodes = np.log(chi2.ppf(u_nodes, d1) / d1)
        q_lo = float(np.log(chi2.ppf(p_lo, d1) / d1)) if p_lo > 0.0 else 0.0
        q_hi = float(np.log(chi2.ppf(1.0 - p_hi, d1) / d1)) if p_hi > 0.0 else 0.0
    else:
        z_nodes = np.asarray(_log_f_ppf(u_nodes, d1, d2), dtype=np.float64)
        q_lo = float(_log_f_ppf(p_lo, d1, d2)) if p_lo > 0.0 else 0.0
        q_hi = float(_log_f_ppf(1.0 - p_hi, d1, d2)) if p_hi > 0.0 else 0.0
    mu = (p_lo * q_lo if p_lo > 0.0 else 0.0) + (p_hi * q_hi if p_hi > 0.0 else 0.0) + float(np.sum(w_u * z_nodes))
    ez2 = (
        (p_lo * q_lo**2 if p_lo > 0.0 else 0.0)
        + (p_hi * q_hi**2 if p_hi > 0.0 else 0.0)
        + float(np.sum(w_u * (z_nodes**2)))
    )
    return float(mu), float(max(ez2 - mu**2, 1e-16))


def _solve_df2_from_upper_residual(u_max: float) -> float:
    """Solve `log(df2 / 2) - digamma(df2 / 2) == u_max` on `(0, inf)` via `brentq`."""
    if not np.isfinite(u_max) or u_max <= 1e-10:
        return float("inf")

    def _fn(d2: float) -> float:
        half = 0.5 * d2
        return float(np.log(half) - digamma(half) - u_max)

    lo = max(1e-10, min(0.1 / u_max, 1e-4))
    while _fn(lo) < 0.0 and lo > 1e-15:
        lo *= 0.1
    hi = max(10.0 / u_max, 100.0)
    while _fn(hi) > 0.0 and hi < 1e12:
        hi *= 10.0
    if _fn(lo) <= 0.0:
        return lo
    if _fn(hi) >= 0.0:
        return hi
    return float(brentq(_fn, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100))


def _fit_f_dist_r_default(
    v: np.ndarray,
    df1_arr: np.ndarray,
    cov_arr: np.ndarray | None,
    valid: np.ndarray,
    robust: bool,
    winsor_tail_p: tuple[float, float],
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Exact Bioconductor `limma::fitFDist` / `limma::fitFDistRobustly` implementation."""
    n_genes = v.size
    has_cov = cov_arr is not None
    n_val = int(np.sum(valid))

    v_raw_val = v[valid]
    med_v = float(np.median(v_raw_val))
    floor_v = 1e-5 * med_v if med_v > 0.0 else 1e-12
    v_val = np.maximum(v_raw_val, floor_v)
    d_val = df1_arr[valid]

    if not robust or n_val < 4:
        z = np.log(v_val)
        e = z - digamma(d_val / 2.0) + np.log(d_val / 2.0)
        mean_trigamma_d1 = float(np.mean(polygamma(1, d_val / 2.0)))

        if cov_arr is not None:
            e_trend_all, rank = _ns_regression_trend_1d(cov_arr[valid], e, cov_arr)
            resid_e = e - e_trend_all[valid]
            total_var_e = float(np.sum(resid_e**2) / max(n_val - rank, 1))
            evar = total_var_e - mean_trigamma_d1
            if evar > 0.0:
                df0_scalar = float(2.0 * trigamma_inverse(evar))
                shift = float(digamma(df0_scalar / 2.0) - np.log(df0_scalar / 2.0))
                s20_out: float | np.ndarray = np.exp(e_trend_all + shift)
            else:
                df0_scalar = float("inf")
                s20_out = np.exp(e_trend_all) if rank > 1 else np.full(n_genes, float(np.mean(v_val)), dtype=np.float64)
        else:
            emean = float(np.mean(e))
            total_var_e = float(np.sum((e - emean) ** 2) / max(n_val - 1, 1))
            evar = total_var_e - mean_trigamma_d1
            if evar > 0.0:
                df0_scalar = float(2.0 * trigamma_inverse(evar))
                shift = float(digamma(df0_scalar / 2.0) - np.log(df0_scalar / 2.0))
                s20_out = float(np.exp(emean + shift))
            else:
                df0_scalar = float("inf")
                s20_out = float(np.mean(v_val))

        if not robust:
            return s20_out, df0_scalar
        return s20_out, np.full(n_genes, df0_scalar, dtype=np.float64)

    # Exact `fitFDistRobustly.R`: Gauss-Legendre Winsorized log-F(d1, d2) moment matching
    p_lo, p_hi = float(winsor_tail_p[0]), float(winsor_tail_p[1])
    if not (0.0 <= p_lo < 0.5 and 0.0 <= p_hi < 0.5 and (p_lo + p_hi) < 0.9):
        raise ValueError("Invalid winsor_tail_p; require 0 <= p_lo, p_hi < 0.5 and p_lo + p_hi < 0.9.")

    d1_eff = float(np.median(d_val))
    d1_shift_eff = float(digamma(d1_eff / 2.0) - np.log(d1_eff / 2.0))
    z_aligned = np.log(v_val) - (digamma(d_val / 2.0) - np.log(d_val / 2.0)) + d1_shift_eff

    if cov_arr is not None:
        z_trend_all, rank = _ns_regression_trend_1d(cov_arr[valid], z_aligned, cov_arr)
        resid_z = z_aligned - z_trend_all[valid]
        q_lo_emp, q_hi_emp = np.quantile(resid_z, [p_lo, 1.0 - p_hi])
        r_win = np.clip(resid_z, q_lo_emp, q_hi_emp)
        mean_win = float(np.mean(r_win))
        var_win = float(np.sum((r_win - mean_win) ** 2) / max(n_val - rank, 1))
    else:
        q_lo_emp, q_hi_emp = np.quantile(z_aligned, [p_lo, 1.0 - p_hi])
        z_win = np.clip(z_aligned, q_lo_emp, q_hi_emp)
        mean_win = float(np.mean(z_win))
        var_win = float(np.sum((z_win - mean_win) ** 2) / max(n_val - 1, 1))

    mu_inf, var_inf = _winsorized_log_f_moments_gl(d1_eff, float("inf"), winsor_tail_p=winsor_tail_p)
    if var_win <= var_inf:
        df0_scalar = float("inf")
        mu_d0 = mu_inf
    else:
        # u = d2 / (1 + d2) in (0, 1); var_gl decreases strictly with u
        def _var_diff(u: float) -> float:
            d2 = u / (1.0 - u)
            _, v_gl = _winsorized_log_f_moments_gl(d1_eff, d2, winsor_tail_p=winsor_tail_p)
            return float(np.log(v_gl) - np.log(var_win))

        u_lo = 0.05
        while _var_diff(u_lo) < 0.0 and u_lo > 1e-5:
            u_lo *= 0.2
        u_hi = 1.0 - 1e-7
        while _var_diff(u_hi) > 0.0 and u_hi < 1.0 - 1e-13:
            u_hi = 1.0 - 0.1 * (1.0 - u_hi)

        if _var_diff(u_lo) <= 0.0:
            df0_scalar = u_lo / (1.0 - u_lo)
        elif _var_diff(u_hi) >= 0.0:
            df0_scalar = float("inf")
        else:
            u_root = float(brentq(_var_diff, u_lo, u_hi, xtol=1e-10, rtol=1e-10, maxiter=80))
            df0_scalar = u_root / (1.0 - u_root)
        mu_d0, _ = _winsorized_log_f_moments_gl(d1_eff, df0_scalar, winsor_tail_p=winsor_tail_p)

    if has_cov and cov_arr is not None:
        s20_out = np.exp(z_trend_all + mean_win - mu_d0)
        s20_vec = s20_out
    else:
        s20_scalar = float(np.exp(mean_win - mu_d0))
        s20_out = s20_scalar
        s20_vec = np.full(n_genes, s20_scalar, dtype=np.float64)

    # Outlier downweighting (`fitFDistRobustly.R`): prob_vals = (n - r + 0.5)/n & df2_outlier mixture
    df0_vec = np.full(n_genes, df0_scalar, dtype=np.float64)
    check_mask = np.isfinite(v) & (v >= 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0) & np.isfinite(s20_vec) & (s20_vec > 0.0)
    m_chk = int(np.sum(check_mask))
    if m_chk >= 2:
        v_chk = np.maximum(v[check_mask], floor_v)
        f_ratio = v_chk / s20_vec[check_mask]
        d_chk = df1_arr[check_mask]
        if np.isfinite(df0_scalar):
            p_upper = f_dist.sf(f_ratio, d_chk, df0_scalar)
        else:
            p_upper = chi2.sf(d_chk * f_ratio, d_chk)

        asc_order = np.argsort(f_ratio, kind="mergesort")
        r_asc = np.arange(1, m_chk + 1, dtype=np.float64)
        prob_vals = (m_chk - r_asc + 0.5) / float(m_chk)
        p_inlier_sorted = np.clip(p_upper[asc_order] / prob_vals, 0.0, 1.0)
        p_inlier_sorted = np.minimum.accumulate(p_inlier_sorted)
        p_inlier = np.empty_like(p_inlier_sorted)
        p_inlier[asc_order] = p_inlier_sorted

        u_resid = np.log(f_ratio) - digamma(d_chk / 2.0) + np.log(d_chk / 2.0)
        u_max = float(np.max(u_resid))
        if np.any(p_inlier < 1.0) and u_max > 1e-8:
            df2_outlier = min(_solve_df2_from_upper_residual(u_max), df0_scalar)
        else:
            df2_outlier = df0_scalar

        if np.isfinite(df0_scalar):
            df0_chk = p_inlier * df0_scalar + (1.0 - p_inlier) * df2_outlier
        else:
            df_pool_cap = float(np.sum(d_chk))
            df2_out_fin = df2_outlier if np.isfinite(df2_outlier) else df_pool_cap
            df0_chk = np.where(
                p_inlier >= 1.0 - 1e-12,
                np.inf,
                p_inlier * df_pool_cap + (1.0 - p_inlier) * df2_out_fin,
            )
        df0_vec[check_mask] = df0_chk

    return s20_out, df0_vec


def fit_f_dist(
    var_g: np.ndarray,
    df1: int | float | np.ndarray,
    covariate: Sequence[float] | np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
    *,
    settings: SettingsMode = "default",
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Moment-match a scaled-inverse-chi-squared prior to per-feature sample variances.

    Supports scalar or per-gene `df1`, mean-variance trend (`covariate`), and Phipson et al. (2016)
    `robust=True` winsorized estimation with per-feature outlier `df_prior` downweighting.
    """
    _validate_settings(settings)
    v = np.asarray(var_g, dtype=np.float64).ravel()
    n_genes = v.size
    df1_arr = np.broadcast_to(np.asarray(df1, dtype=np.float64), (n_genes,)).copy()

    has_cov = covariate is not None
    cov_arr = np.asarray(covariate, dtype=np.float64).ravel() if has_cov else None
    if cov_arr is not None and cov_arr.size != n_genes:
        raise ValueError(f"covariate length ({cov_arr.size}) must match var_g length ({n_genes}).")

    valid = np.isfinite(v) & (v > 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0)
    if cov_arr is not None:
        valid = valid & np.isfinite(cov_arr)

    n_val = int(np.sum(valid))
    if n_val < 2:
        s20_fallback = float(v[valid][0]) if n_val == 1 else 1.0
        s20_ret: float | np.ndarray = np.full(n_genes, s20_fallback, dtype=np.float64) if has_cov else s20_fallback
        df0_ret: float | np.ndarray = np.full(n_genes, np.inf, dtype=np.float64) if robust else float("inf")
        return s20_ret, df0_ret

    if settings == "R_default":
        return _fit_f_dist_r_default(
            v=v,
            df1_arr=df1_arr,
            cov_arr=cov_arr,
            valid=valid,
            robust=robust,
            winsor_tail_p=winsor_tail_p,
        )

    v_val = v[valid]
    d_val = df1_arr[valid]
    z = np.log(v_val)
    e = z - digamma(d_val / 2.0) + np.log(d_val / 2.0)
    mean_trigamma_d1 = float(np.mean(polygamma(1, d_val / 2.0)))

    if cov_arr is not None:
        e_trend_all = _lowess_tricube_1d(cov_arr[valid], e, cov_arr, span=0.5)
        resid_e = e - e_trend_all[valid]
        if robust and n_val >= 4:
            mu_r, var_r = _winsorized_gaussian_moments(resid_e, winsor_tail_p=winsor_tail_p)
            e_prior_all = e_trend_all + mu_r
            total_var_e = var_r * (n_val / max(n_val - 2, 1))
        else:
            e_prior_all = e_trend_all
            total_var_e = float(np.sum(resid_e**2) / max(n_val - 2, 1))
        evar = total_var_e - mean_trigamma_d1
    else:
        if robust and n_val >= 4:
            emean, total_var_e = _winsorized_gaussian_moments(e, winsor_tail_p=winsor_tail_p)
        else:
            emean = float(np.mean(e))
            total_var_e = float(np.sum((e - emean) ** 2) / (n_val - 1))
        evar = total_var_e - mean_trigamma_d1

    if evar > 0.0:
        df0_scalar = float(2.0 * trigamma_inverse(evar))
        shift = float(digamma(df0_scalar / 2.0) - np.log(df0_scalar / 2.0))
    else:
        df0_scalar = float("inf")
        shift = 0.0

    if cov_arr is not None:
        s20_out: float | np.ndarray = np.exp(e_prior_all + shift)
        s20_vec = s20_out
    else:
        s20_scalar = float(np.exp(emean + shift))
        s20_out = s20_scalar
        s20_vec = np.full(n_genes, s20_scalar, dtype=np.float64)

    if not robust:
        return s20_out, df0_scalar

    # Phipson et al. (2016): gene-specific df_prior downweighting for hypervariable outliers
    df0_vec = np.full(n_genes, df0_scalar, dtype=np.float64)
    check_mask = np.isfinite(v) & (v >= 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0) & (s20_vec > 0.0)
    m_chk = int(np.sum(check_mask))
    if m_chk >= 2:
        f_ratio = v[check_mask] / s20_vec[check_mask]
        d_chk = df1_arr[check_mask]
        if np.isfinite(df0_scalar):
            p_upper = f_dist.sf(f_ratio, d_chk, df0_scalar)
        else:
            p_upper = chi2.sf(d_chk * f_ratio, d_chk)

        asc_order = np.argsort(f_ratio, kind="mergesort")
        f_sorted = f_ratio[asc_order]
        p_sorted = p_upper[asc_order]
        # Rank from largest F_g (1 for largest, m_chk for smallest)
        rank_desc = np.arange(m_chk, 0, -1, dtype=np.float64)
        u_exp = rank_desc / float(m_chk)
        w_sorted = np.where(f_sorted > 1.0, np.clip(p_sorted / u_exp, 0.0, 1.0), 1.0)
        # Monotone non-increasing weight as F_g increases
        w_sorted = np.minimum.accumulate(w_sorted)
        weights = np.empty_like(w_sorted)
        weights[asc_order] = w_sorted

        if np.isfinite(df0_scalar):
            df0_chk = df0_scalar * weights
        else:
            # Avoid inf * 0 = nan: inliers (weight >= 1 - 1e-6) retain inf; upper-tail outliers get finite downweighted df
            df0_chk = np.where(weights >= 1.0 - 1e-6, np.inf, 100.0 * weights)
        df0_vec[check_mask] = df0_chk

    return s20_out, df0_vec


def squeeze_var(
    var_g: np.ndarray,
    df1: int | float | np.ndarray,
    covariate: Sequence[float] | np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
    *,
    settings: SettingsMode = "default",
) -> tuple[np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Shrink per-feature sample variances toward a fitted empirical Bayes prior (`limma::squeezeVar`)."""
    _validate_settings(settings)
    v = np.asarray(var_g, dtype=np.float64)
    orig_shape = v.shape
    v_flat = v.ravel()
    n_genes = v_flat.size

    scalar_df1 = np.ndim(df1) == 0
    df1_arr = np.broadcast_to(np.asarray(df1, dtype=np.float64), (n_genes,)).copy()
    valid_gene = np.isfinite(v_flat) & (v_flat >= 0.0) & np.isfinite(df1_arr) & (df1_arr > 0.0)

    s20, df0 = fit_f_dist(
        v_flat,
        df1_arr if not scalar_df1 else float(df1),
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        settings=settings,
    )
    s20_arr = np.broadcast_to(np.asarray(s20, dtype=np.float64), (n_genes,))
    df0_arr = np.broadcast_to(np.asarray(df0, dtype=np.float64), (n_genes,))

    is_inf_df0 = np.isinf(df0_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        shrunk_finite = (df0_arr * s20_arr + df1_arr * v_flat) / (df0_arr + df1_arr)
    var_post_flat = np.where(is_inf_df0, s20_arr, shrunk_finite)
    var_post_flat = np.where(valid_gene, var_post_flat, np.nan)

    df_pooled = float(np.sum(df1_arr[valid_gene])) if np.any(valid_gene) else float("inf")

    if scalar_df1 and np.ndim(df0) == 0:
        df_total: float | np.ndarray = float("inf") if np.isinf(df0) else float(df0) + float(df1)
        if settings == "R_default" and np.isfinite(df_pooled):
            df_total = min(float(df_total), df_pooled)
    else:
        df_total_arr = np.where(is_inf_df0, np.inf, df0_arr + df1_arr)
        if settings == "R_default" and np.isfinite(df_pooled):
            df_total_arr = np.minimum(df_total_arr, df_pooled)
        df_total = np.where(valid_gene, df_total_arr, np.nan)

    return var_post_flat.reshape(orig_shape), s20, df0, df_total


def _moderated_p_and_fdr(
    t_stat: np.ndarray,
    df_total: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact t / normal survival p-values (handling `df_total == inf`) and NaN-preserving BH FDR."""
    n = t_stat.size
    df_arr = np.broadcast_to(np.asarray(df_total, dtype=np.float64), (n,))
    valid = np.isfinite(t_stat) & (~np.isnan(df_arr)) & (df_arr > 0.0)

    p_val = np.full(n, np.nan, dtype=np.float64)
    if np.any(valid):
        abs_t = np.abs(t_stat[valid])
        df_v = df_arr[valid]
        inf_m = np.isinf(df_v)
        pv = np.empty_like(abs_t)
        if np.any(inf_m):
            pv[inf_m] = 2.0 * norm.sf(abs_t[inf_m])
        if np.any(~inf_m):
            pv[~inf_m] = 2.0 * t_dist.sf(abs_t[~inf_m], df_v[~inf_m])
        p_val[valid] = np.clip(pv, 0.0, 1.0)

    adj_p = np.full(n, np.nan, dtype=np.float64)
    valid_p = np.isfinite(p_val)
    if np.any(valid_p):
        adj_p[valid_p] = false_discovery_control(p_val[valid_p], method="bh")

    return p_val, adj_p


def _compute_b_lods(
    t_stat: np.ndarray,
    unscaled_var: np.ndarray,
    df_total: float | np.ndarray,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
) -> np.ndarray:
    """Compute Smyth (2004) empirical Bayes `B` log-odds (`limma::tmixture.vector` + `ebayes.R`)."""
    if not (0.0 < proportion < 1.0):
        raise ValueError(f"proportion must lie strictly in (0, 1), got {proportion!r}.")
    lo_sd, hi_sd = float(stdev_coef_lim[0]), float(stdev_coef_lim[1])
    if lo_sd <= 0.0 or hi_sd < lo_sd:
        raise ValueError(f"Invalid stdev_coef_lim={stdev_coef_lim!r}; require 0 < low <= high.")

    n = t_stat.size
    df_arr = np.broadcast_to(np.asarray(df_total, dtype=np.float64), (n,))
    u_arr = np.broadcast_to(np.asarray(unscaled_var, dtype=np.float64), (n,))
    valid = np.isfinite(t_stat) & np.isfinite(u_arr) & (u_arr > 0.0) & (~np.isnan(df_arr)) & (df_arr > 0.0)

    b_out = np.full(n, np.nan, dtype=np.float64)
    ngenes = int(np.sum(valid))
    if ngenes == 0:
        return b_out

    v0_lim = (lo_sd**2, hi_sd**2)
    t_v = t_stat[valid]
    u_v = u_arr[valid]
    df_v = df_arr[valid]

    ntarget = int(np.ceil(0.5 * proportion * ngenes))
    p_mix = max(ntarget / float(ngenes), float(proportion))
    abs_t = np.abs(t_v)
    desc_idx = np.argsort(-abs_t, kind="mergesort")[:ntarget]

    t_top = abs_t[desc_idx]
    u_top = u_v[desc_idx]
    df_top = df_v[desc_idx]

    p0 = np.empty_like(t_top)
    inf_top = np.isinf(df_top)
    if np.any(inf_top):
        p0[inf_top] = 2.0 * norm.sf(t_top[inf_top])
    if np.any(~inf_top):
        p0[~inf_top] = 2.0 * t_dist.sf(t_top[~inf_top], df_top[~inf_top])

    ptarget = (np.arange(1, ntarget + 1, dtype=np.float64) - 0.5) / float(ngenes)
    v0_vec = np.zeros(ntarget, dtype=np.float64)
    pos = ptarget > p0
    if np.any(pos):
        q_alt = 0.5 * (ptarget[pos] - (1.0 - p_mix) * p0[pos]) / p_mix
        q_alt = np.clip(q_alt, 1e-300, 0.5 - 1e-12)
        df_pos = df_top[pos]
        inf_pos = np.isinf(df_pos)
        t_os = np.empty_like(q_alt)
        if np.any(inf_pos):
            t_os[inf_pos] = norm.isf(q_alt[inf_pos])
        if np.any(~inf_pos):
            t_os[~inf_pos] = t_dist.isf(q_alt[~inf_pos], df_pos[~inf_pos])
        v0_vec[pos] = u_top[pos] * np.maximum((t_top[pos] / np.maximum(t_os, 1e-12)) ** 2 - 1.0, 0.0)

    v0_hat = float(np.mean(np.clip(v0_vec, v0_lim[0], v0_lim[1])))
    r_v = u_v / (u_v + v0_hat)
    t2 = t_v**2
    prior_lods = float(np.log(proportion / (1.0 - proportion)))

    b_val = np.empty_like(t_v)
    inf_m = np.isinf(df_v)
    if np.any(inf_m):
        b_val[inf_m] = prior_lods + 0.5 * np.log(r_v[inf_m]) + 0.5 * t2[inf_m] * (1.0 - r_v[inf_m])
    if np.any(~inf_m):
        df_f = df_v[~inf_m]
        t2_f = t2[~inf_m]
        r_f = r_v[~inf_m]
        b_val[~inf_m] = (
            prior_lods
            + 0.5 * np.log(r_f)
            + 0.5 * (df_f + 1.0) * (np.log1p(t2_f / df_f) - np.log1p(r_f * t2_f / df_f))
        )

    b_out[valid] = b_val
    return b_out


def moderated_ttest(
    X: np.ndarray | pd.DataFrame,
    group: Sequence[bool | int],
    feature_names: Sequence[str] | None = None,
    robust: bool = False,
    trend: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    *,
    settings: SettingsMode = "default",
) -> ModeratedTestResult:
    """Two-group empirical Bayes moderated t-test (`limma::eBayes`) across columns of `X`."""
    _validate_settings(settings)
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns] if feature_names is None else [str(f) for f in feature_names]
        mat = X.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(X, dtype=np.float64)
        names = [f"feature_{i}" for i in range(mat.shape[1])] if feature_names is None else [str(f) for f in feature_names]

    grp = np.asarray(group, dtype=bool)
    if mat.shape[0] != grp.size:
        raise ValueError(f"X has {mat.shape[0]} rows, but group has length {grp.size}.")

    n1_tot = int((~grp).sum())
    n2_tot = int(grp.sum())
    if n1_tot < 1 or n2_tot < 1 or (n1_tot + n2_tot - 2) < 1:
        raise ValueError("Need at least 2 total degrees of freedom and >=1 sample per group.")

    x1, x2 = mat[~grp], mat[grp]
    if np.all(np.isfinite(mat)):
        mean1 = x1.mean(axis=0)
        mean2 = x2.mean(axis=0)
        ave_expr = mat.mean(axis=0)
        df1_arg: int | np.ndarray = n1_tot + n2_tot - 2
        var_g = (((x1 - mean1) ** 2).sum(axis=0) + ((x2 - mean2) ** 2).sum(axis=0)) / df1_arg
        unscaled_var = np.full(mat.shape[1], 1.0 / n1_tot + 1.0 / n2_tot, dtype=np.float64)
        diff = mean2 - mean1
    else:
        fin1 = np.isfinite(x1)
        fin2 = np.isfinite(x2)
        n1_g = fin1.sum(axis=0)
        n2_g = fin2.sum(axis=0)
        df1_g = (n1_g + n2_g - 2).astype(np.float64)
        valid_g = (n1_g >= 1) & (n2_g >= 1) & (df1_g >= 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean1 = np.where(n1_g >= 1, np.nansum(x1, axis=0) / np.maximum(n1_g, 1), np.nan)
            mean2 = np.where(n2_g >= 1, np.nansum(x2, axis=0) / np.maximum(n2_g, 1), np.nan)
            n_all = n1_g + n2_g
            ave_expr = np.where(n_all >= 1, np.nansum(mat, axis=0) / np.maximum(n_all, 1), np.nan)
            ss1 = np.nansum((x1 - mean1[None, :]) ** 2, axis=0)
            ss2 = np.nansum((x2 - mean2[None, :]) ** 2, axis=0)
            var_g = np.where(valid_g, (ss1 + ss2) / np.maximum(df1_g, 1.0), np.nan)
            unscaled_var = np.where(valid_g, 1.0 / np.maximum(n1_g, 1) + 1.0 / np.maximum(n2_g, 1), np.nan)
            diff = np.where(valid_g, mean2 - mean1, np.nan)
        df1_arg = np.where(valid_g, df1_g, np.nan)

    var_post, s20, df0, df_total = squeeze_var(
        var_g, df1_arg, covariate=ave_expr if trend else None, robust=robust, settings=settings
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.maximum(var_post * unscaled_var, 1e-300))
        t_stat = np.where(np.isfinite(var_post) & np.isfinite(diff) & np.isfinite(unscaled_var), diff / se, np.nan)

    p_val, adj_p = _moderated_p_and_fdr(t_stat, df_total)
    b_stat = _compute_b_lods(
        t_stat, unscaled_var, df_total, proportion=proportion, stdev_coef_lim=stdev_coef_lim
    )

    return ModeratedTestResult(
        feature_names=names,
        log_fc=diff,
        ave_expr=ave_expr,
        t_stat=t_stat,
        p_value=p_val,
        adj_p_val=adj_p,
        var_prior=s20,
        df_prior=df0,
        var_post=var_post,
        b_stat=b_stat,
    )


def moderated_lm_contrast(
    Y: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
    contrast: Sequence[float],
    feature_names: Sequence[str] | None = None,
    robust: bool = False,
    trend: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    *,
    settings: SettingsMode = "default",
) -> ModeratedTestResult:
    """General linear model `limma::lmFit` + `contrasts.fit` + `eBayes` across features."""
    _validate_settings(settings)
    if isinstance(Y, pd.DataFrame):
        names = [str(c) for c in Y.columns] if feature_names is None else [str(f) for f in feature_names]
        ymat = Y.to_numpy(dtype=np.float64)
    else:
        ymat = np.asarray(Y, dtype=np.float64)
        names = [f"feature_{i}" for i in range(ymat.shape[1])] if feature_names is None else [str(f) for f in feature_names]

    X = design.to_numpy(dtype=np.float64) if isinstance(design, pd.DataFrame) else np.asarray(design, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("Design matrix must not contain NaN or Inf values.")

    c = np.asarray(contrast, dtype=np.float64).ravel()
    n, p = X.shape
    if ymat.shape[0] != n:
        raise ValueError(f"Y has {ymat.shape[0]} samples, but design has {n} rows.")
    if c.size != p:
        raise ValueError(f"Contrast length ({c.size}) must match design columns ({p}).")

    rank = int(np.linalg.matrix_rank(X))
    df1_full = n - rank
    if df1_full < 1:
        raise ValueError(f"Residual degrees of freedom must be >= 1 (got n={n}, rank={rank}).")

    xtx_inv_full = np.linalg.pinv(X.T @ X)
    unscaled_var_full = float(c @ xtx_inv_full @ c)
    if unscaled_var_full <= 0.0:
        raise ValueError("Contrast vector has zero variance under the supplied design matrix.")

    n_genes = ymat.shape[1]
    if np.all(np.isfinite(ymat)):
        betas = xtx_inv_full @ (X.T @ ymat)
        resid = ymat - X @ betas
        var_g = np.sum(resid**2, axis=0) / df1_full
        est = (c @ betas).ravel()
        ave_expr = ymat.mean(axis=0)
        unscaled_var_vec = np.full(n_genes, unscaled_var_full, dtype=np.float64)
        df1_arg: int | np.ndarray = df1_full
    else:
        var_g = np.full(n_genes, np.nan, dtype=np.float64)
        est = np.full(n_genes, np.nan, dtype=np.float64)
        ave_expr = np.full(n_genes, np.nan, dtype=np.float64)
        unscaled_var_vec = np.full(n_genes, np.nan, dtype=np.float64)
        df1_vec = np.full(n_genes, np.nan, dtype=np.float64)

        complete_cols = np.all(np.isfinite(ymat), axis=0)
        if np.any(complete_cols):
            y_comp = ymat[:, complete_cols]
            b_comp = xtx_inv_full @ (X.T @ y_comp)
            r_comp = y_comp - X @ b_comp
            var_g[complete_cols] = np.sum(r_comp**2, axis=0) / df1_full
            est[complete_cols] = (c @ b_comp).ravel()
            ave_expr[complete_cols] = y_comp.mean(axis=0)
            unscaled_var_vec[complete_cols] = unscaled_var_full
            df1_vec[complete_cols] = float(df1_full)

        for g in np.flatnonzero(~complete_cols):
            obs = np.isfinite(ymat[:, g])
            n_obs = int(np.sum(obs))
            if n_obs >= 1:
                ave_expr[g] = float(np.mean(ymat[obs, g]))
            if n_obs <= 1:
                continue
            X_obs = X[obs]
            y_obs = ymat[obs, g]
            rank_g = int(np.linalg.matrix_rank(X_obs))
            df_g = n_obs - rank_g
            if df_g < 1:
                continue
            xtx_obs = X_obs.T @ X_obs
            xtx_inv_g = np.linalg.pinv(xtx_obs)
            # Check estimability of contrast c in the row-space of X_obs
            if not np.allclose(c, c @ xtx_inv_g @ xtx_obs, atol=1e-7):
                continue
            uvar_g = float(c @ xtx_inv_g @ c)
            if uvar_g <= 0.0:
                continue
            beta_g = xtx_inv_g @ (X_obs.T @ y_obs)
            resid_g = y_obs - X_obs @ beta_g
            var_g[g] = float(np.sum(resid_g**2) / df_g)
            est[g] = float(c @ beta_g)
            unscaled_var_vec[g] = uvar_g
            df1_vec[g] = float(df_g)
        df1_arg = df1_vec

    var_post, s20, df0, df_total = squeeze_var(
        var_g, df1_arg, covariate=ave_expr if trend else None, robust=robust, settings=settings
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.maximum(var_post * unscaled_var_vec, 1e-300))
        t_stat = np.where(np.isfinite(var_post) & np.isfinite(est) & np.isfinite(unscaled_var_vec), est / se, np.nan)

    p_val, adj_p = _moderated_p_and_fdr(t_stat, df_total)
    b_stat = _compute_b_lods(
        t_stat, unscaled_var_vec, df_total, proportion=proportion, stdev_coef_lim=stdev_coef_lim
    )

    return ModeratedTestResult(
        feature_names=names,
        log_fc=est,
        ave_expr=ave_expr,
        t_stat=t_stat,
        p_value=p_val,
        adj_p_val=adj_p,
        var_prior=s20,
        df_prior=df0,
        var_post=var_post,
        b_stat=b_stat,
    )


def limma_trend(
    X: np.ndarray | pd.DataFrame,
    group: Sequence[bool | int],
    feature_names: Sequence[str] | None = None,
    robust: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    *,
    settings: SettingsMode = "default",
) -> ModeratedTestResult:
    """Two-group `limma-trend` moderated t-test fitting a mean-variance prior across `AveExpr` (Law et al. 2014)."""
    _validate_settings(settings)
    return moderated_ttest(
        X,
        group,
        feature_names=feature_names,
        robust=robust,
        trend=True,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        settings=settings,
    )


def limma_top_table(
    X: np.ndarray | pd.DataFrame,
    group: Sequence[bool | int],
    feature_names: Sequence[str] | None = None,
    sort_by_pvalue: bool = True,
    robust: bool = False,
    trend: bool = False,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    *,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Convenience wrapper returning a `limma::topTable` DataFrame for a two-group comparison."""
    _validate_settings(settings)
    return moderated_ttest(
        X,
        group,
        feature_names=feature_names,
        robust=robust,
        trend=trend,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        settings=settings,
    ).to_dataframe(sort_by_pvalue=sort_by_pvalue)
