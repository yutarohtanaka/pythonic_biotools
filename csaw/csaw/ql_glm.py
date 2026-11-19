"""Negative Binomial trended dispersion and Quasi-Likelihood Empirical Bayes F-test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import f as f_dist
from scipy.stats import norm

from .utils import SettingsMode, ave_log_cpm, benjamini_hochberg, loess_fit_1d, validate_settings
from .windows import WindowData


def _trigamma_inverse(y: float) -> float:
    """Solve polygamma(1, x) = y for x > 0 via Smyth's (2004) monotonic Newton iteration."""
    if not np.isfinite(y) or y <= 1e-7:
        return 1e7
    if y > 1e7:
        return 1.0 / np.sqrt(y)
    # Initial asymptotic approximation: psi_1(x) ~ 1/x + 1/(2 x^2)
    x = 0.5 + 1.0 / y
    for _ in range(15):
        tri = float(polygamma(1, x))
        tetra = float(polygamma(2, x))
        step = tri * (1.0 - tri / y) / tetra
        x = max(1e-8, x + step)
        if abs(step) <= 1e-10 * x:
            break
    return x


def _winsorized_var_factor(prob_left: float = 0.05, prob_right: float = 0.10) -> float:
    """Theoretical variance of a Winsorized N(0, 1) variable with asymmetric tail probs."""
    a = float(norm.ppf(prob_left))
    b = float(norm.ppf(1.0 - prob_right))
    # E[W] and E[W^2] for W = clip(Z, a, b), Z ~ N(0, 1)
    phi_a = float(norm.pdf(a))
    phi_b = float(norm.pdf(b))
    cdf_inner = (1.0 - prob_right) - prob_left
    ez = a * prob_left + b * prob_right + (phi_a - phi_b)
    ez2 = (a ** 2) * prob_left + (b ** 2) * prob_right + (cdf_inner + a * phi_a - b * phi_b)
    return max(ez2 - ez ** 2, 1e-6)


def _nb_deviance(y: np.ndarray, mu: np.ndarray, disp: np.ndarray | float) -> np.ndarray:
    """Compute row-wise Negative Binomial residual deviance without zero-log warnings."""
    mu_safe = np.maximum(mu, 1e-12)
    y_pos = np.maximum(y, 1e-12)
    term1 = np.where(y > 0.0, y * np.log(y_pos / mu_safe), 0.0)

    if np.ndim(disp) == 0:
        phi = np.full((y.shape[0], 1), float(disp), dtype=float)
    else:
        phi = np.asarray(disp, dtype=float)[:, None]

    pois_mask = phi <= 1e-10
    phi_safe = np.where(pois_mask, 1.0, phi)
    nb_unit = 2.0 * (
        term1 - (y + 1.0 / phi_safe) * np.log((1.0 + phi_safe * y) / (1.0 + phi_safe * mu_safe))
    )
    pois_unit = 2.0 * (term1 - (y - mu_safe))
    unit = np.where(pois_mask, pois_unit, nb_unit)
    return np.maximum(0.0, np.sum(unit, axis=1))


def _fit_nb_glm_irls(
    y: np.ndarray,
    design: np.ndarray,
    offsets: np.ndarray,
    disp: np.ndarray | float,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Negative Binomial GLM Fisher-scoring IRLS across all windows."""
    n_win, n_samp = y.shape
    p = design.shape[1]
    if n_win == 0:
        return np.zeros((0, p)), np.zeros((0, n_samp)), np.zeros(0)

    if p == 0:
        mu = np.exp(np.clip(offsets, -25.0, 25.0))
        dev = _nb_deviance(y, mu, disp)
        return np.zeros((n_win, 0)), mu, dev

    phi = (
        np.full((n_win, 1), float(disp), dtype=float)
        if np.ndim(disp) == 0
        else np.asarray(disp, dtype=float)[:, None]
    )

    # OLS warm start on log(y + 0.25) - offsets
    eta0 = np.log(np.maximum(y, 0.25)) - offsets
    xtx_pinv = np.linalg.pinv(design)  # (p, n_samp)
    beta = eta0 @ xtx_pinv.T  # (n_win, p)

    mu = np.exp(np.clip(offsets + beta @ design.T, -25.0, 25.0))
    dev = _nb_deviance(y, mu, disp)
    eye_p = np.eye(p, dtype=float)[None, :, :] * 1e-9

    active = np.ones(n_win, dtype=bool)
    for _ in range(max_iter):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        y_a = y[idx]
        mu_a = mu[idx]
        off_a = offsets[idx]
        phi_a = phi[idx]
        beta_a = beta[idx]
        dev_a = dev[idx]

        denom = 1.0 + phi_a * mu_a
        w_a = mu_a / denom  # (m, n_samp)
        score = ((y_a - mu_a) / denom) @ design  # (m, p)
        info = np.einsum("mn,ni,nj->mij", w_a, design, design) + eye_p
        delta = np.linalg.solve(info, score[..., None])[..., 0]
        delta = np.clip(delta, -10.0, 10.0)

        # Backtracking line search for windows where deviance increases
        step_scale = np.ones(idx.size, dtype=float)
        new_beta = beta_a + delta
        new_mu = np.exp(np.clip(off_a + new_beta @ design.T, -25.0, 25.0))
        new_dev = _nb_deviance(y_a, new_mu, phi_a[:, 0])

        for _ in range(8):
            bad = new_dev > dev_a + 1e-7 * (1.0 + dev_a)
            if not np.any(bad):
                break
            step_scale[bad] *= 0.5
            new_beta[bad] = beta_a[bad] + step_scale[bad, None] * delta[bad]
            new_mu[bad] = np.exp(
                np.clip(off_a[bad] + new_beta[bad] @ design.T, -25.0, 25.0)
            )
            new_dev[bad] = _nb_deviance(y_a[bad], new_mu[bad], phi_a[bad, 0])

        still_bad = new_dev > dev_a + 1e-7 * (1.0 + dev_a)
        if np.any(still_bad):
            new_beta[still_bad] = beta_a[still_bad]
            new_mu[still_bad] = mu_a[still_bad]
            new_dev[still_bad] = dev_a[still_bad]

        conv = (np.abs(dev_a - new_dev) / (0.1 + new_dev) < tol) | still_bad
        beta[idx] = new_beta
        mu[idx] = new_mu
        dev[idx] = new_dev
        active[idx[conv]] = False

    return beta, mu, dev


def _cox_reid_apl_subset(
    y_sub: np.ndarray,
    design: np.ndarray,
    off_sub: np.ndarray,
    phi: float,
) -> float:
    """Evaluate summed Cox-Reid adjusted profile log-likelihood at scalar NB dispersion phi."""
    _, mu, _ = _fit_nb_glm_irls(y_sub, design, off_sub, phi, max_iter=15, tol=1e-6)
    inv_phi = 1.0 / phi
    mu_safe = np.maximum(mu, 1e-12)
    ll = (
        gammaln(y_sub + inv_phi)
        - gammaln(inv_phi)
        - inv_phi * np.log(1.0 + phi * mu_safe)
        + np.where(
            y_sub > 0.0,
            y_sub * np.log((phi * mu_safe) / (1.0 + phi * mu_safe)),
            0.0,
        )
    )
    w = mu_safe / (1.0 + phi * mu_safe)
    info = np.einsum("mn,ni,nj->mij", w, design, design) + np.eye(design.shape[1])[None] * 1e-10
    _, logdet = np.linalg.slogdet(info)
    return float(np.sum(ll) - 0.5 * np.sum(logdet))


def _estimate_trended_nb_disp(
    y: np.ndarray,
    design: np.ndarray,
    offsets: np.ndarray,
    abundances: np.ndarray,
    abundance_trend: bool = True,
) -> np.ndarray:
    """Estimate common or abundance-trended Negative Binomial dispersion via Cox-Reid APL."""
    n_win = y.shape[0]
    if n_win == 0:
        return np.empty(0, dtype=float)

    grid_log_phi = np.linspace(np.log(1e-3), np.log(2.0), 13)
    grid_phi = np.exp(grid_log_phi)

    def _opt_bin(idx_bin: np.ndarray) -> float:
        # Subsample up to 250 windows per bin for fast, accurate APL maximization
        if idx_bin.size > 250:
            step = max(1, idx_bin.size // 250)
            idx_eval = idx_bin[::step][:250]
        else:
            idx_eval = idx_bin
        y_b = y[idx_eval]
        off_b = offsets[idx_eval]
        scores = np.array([_cox_reid_apl_subset(y_b, design, off_b, float(ph)) for ph in grid_phi])
        best_k = int(np.argmax(scores))
        # Local 5-point refinement around best_k
        lo_lp = grid_log_phi[max(0, best_k - 1)]
        hi_lp = grid_log_phi[min(grid_log_phi.size - 1, best_k + 1)]
        fine_lp = np.linspace(lo_lp, hi_lp, 7)
        fine_scores = [
            _cox_reid_apl_subset(y_b, design, off_b, float(np.exp(lp))) for lp in fine_lp
        ]
        return float(np.exp(fine_lp[int(np.argmax(fine_scores))]))

    if not abundance_trend or n_win < 30:
        phi_common = _opt_bin(np.arange(n_win))
        return np.full(n_win, phi_common, dtype=float)

    n_bins = min(6, max(2, n_win // 40))
    order = np.argsort(abundances, kind="mergesort")
    splits = np.array_split(order, n_bins)
    bin_centers = np.empty(len(splits), dtype=float)
    bin_log_phis = np.empty(len(splits), dtype=float)
    for b_idx, sp in enumerate(splits):
        bin_centers[b_idx] = float(np.mean(abundances[sp]))
        bin_log_phis[b_idx] = np.log(_opt_bin(sp))

    interp_log_phi = np.interp(abundances, bin_centers, bin_log_phis)
    return np.exp(interp_log_phi)


def _compute_residual_df(
    y: np.ndarray,
    mu: np.ndarray,
    design: np.ndarray,
) -> np.ndarray:
    """Compute residual df per window, accounting for exact structural all-zero groups."""
    n_win, n_samp = y.shape
    p = design.shape[1]
    base_df = max(1, n_samp - p)
    df_resid = np.full(n_win, float(base_df), dtype=float)

    zero_fitted = (y == 0.0) & (mu < 1e-4)
    has_zero = np.any(zero_fitted, axis=1)
    if np.any(has_zero):
        for w in np.flatnonzero(has_zero):
            pos_mask = ~zero_fitted[w]
            n_pos = int(np.sum(pos_mask))
            if n_pos <= 1:
                df_resid[w] = 1.0
            else:
                rank_pos = int(np.linalg.matrix_rank(design[pos_mask, :]))
                df_resid[w] = float(max(1, n_pos - rank_pos))
    return df_resid


def _squeeze_var(
    s2: np.ndarray,
    df_resid: np.ndarray,
    abundances: np.ndarray,
    abundance_trend: bool = True,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical Bayes moderation of QL dispersions (limma::squeezeVar / Lun et al. 2016)."""
    n_win = s2.shape[0]
    if n_win == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    if n_win <= 2:
        return s2.copy(), np.zeros(n_win), s2.copy()

    # Handle exact-zero residual deviances softly at lower floor for log transform
    pos_s2 = s2[s2 > 1e-10]
    floor_val = max(1e-8, float(np.min(pos_s2)) * 0.01) if pos_s2.size > 0 else 1e-6
    s2_safe = np.maximum(s2, floor_val)

    half_df = 0.5 * df_resid
    z = np.log(s2_safe) - digamma(half_df) + np.log(half_df)

    if abundance_trend and n_win >= 50:
        eff_span = 0.5 if n_win >= 200 else 0.75
        z_trend = loess_fit_1d(abundances, z, span=eff_span)
    else:
        z_trend = np.full(n_win, float(np.mean(z)), dtype=float)

    resid = z - z_trend
    if robust and n_win >= 10:
        q_lo, q_hi = np.quantile(resid, [0.05, 0.90])
        resid_win = np.clip(resid, q_lo, q_hi)
        win_factor = _winsorized_var_factor(0.05, 0.10)
        var_z = float(np.var(resid_win, ddof=1)) / win_factor
    else:
        var_z = float(np.var(resid, ddof=1))

    target_trigamma = var_z - float(np.mean(polygamma(1, half_df)))
    d0_scalar = 2.0 * _trigamma_inverse(target_trigamma)
    d0_scalar = float(np.clip(d0_scalar, 1e-4, 1e6))

    half_d0 = 0.5 * d0_scalar
    s02 = np.exp(z_trend + digamma(half_d0) - np.log(half_d0))

    d0_vec = np.full(n_win, d0_scalar, dtype=float)
    if robust and d0_scalar < 1e5 and n_win >= 10:
        # Downweight prior df on extreme upper-tail hypervariable windows (Phipson et al. 2016)
        f_ratio = s2_safe / np.maximum(s02, 1e-12)
        p_upper = f_dist.sf(f_ratio, df_resid, d0_scalar)
        adj_upper = benjamini_hochberg(p_upper)
        outlier = p_upper < 0.02
        if np.any(outlier):
            # Smoothly attenuate prior df according to outlier evidence
            weight = np.clip(adj_upper[outlier] / 0.05, 0.01, 1.0)
            d0_vec[outlier] = d0_scalar * weight

    s2_post = (d0_vec * s02 + df_resid * s2) / (d0_vec + df_resid)
    return s2_post, d0_vec, s02


def _build_contrast_and_null_design(
    design: np.ndarray,
    contrast: np.ndarray | Sequence[float] | None,
    coef: int | Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct tested contrast matrix C (p, q) and orthogonal reduced design X0 (n, p - q)."""
    n_samp, p = design.shape
    if contrast is not None and coef is not None:
        raise ValueError("Specify at most one of 'contrast' or 'coef', not both.")

    if contrast is not None:
        c_arr = np.asarray(contrast, dtype=float)
        if c_arr.ndim == 1:
            if c_arr.shape[0] != p:
                raise ValueError(
                    f"1D contrast length ({c_arr.shape[0]}) must match design columns ({p})."
                )
            c_mat = c_arr[:, None]
        elif c_arr.ndim == 2:
            if c_arr.shape[0] != p:
                raise ValueError(
                    f"2D contrast rows ({c_arr.shape[0]}) must match design columns ({p})."
                )
            c_mat = c_arr
        else:
            raise ValueError("contrast must be a 1D vector or 2D matrix.")
    else:
        if coef is None:
            cols = [p - 1]
        elif isinstance(coef, (int, np.integer)):
            cols = [int(coef)]
        else:
            cols = [int(c) for c in coef]
        for c in cols:
            if c < 0 or c >= p:
                raise ValueError(f"coef index {c} out of range for design with {p} columns.")
        c_mat = np.zeros((p, len(cols)), dtype=float)
        for idx, col in enumerate(cols):
            c_mat[col, idx] = 1.0

    # Orthogonal basis of C and its complement Q_perp in R^p via SVD
    u, s, _ = np.linalg.svd(c_mat, full_matrices=True)
    q = int(np.sum(s > 1e-10))
    if q == 0:
        raise ValueError("Specified contrast / coef has zero rank.")
    q_perp = u[:, q:]  # (p, p - q) such that C^T @ Q_perp = 0
    x0 = design @ q_perp  # (n_samp, p - q)
    return c_mat, x0, q


def glm_ql_fit_test(
    counts: WindowData | np.ndarray,
    design: np.ndarray | pd.DataFrame,
    contrast: np.ndarray | Sequence[float] | None = None,
    coef: int | Sequence[int] | None = None,
    norm_factors: np.ndarray | Sequence[float] | None = None,
    offsets: np.ndarray | None = None,
    totals: np.ndarray | None = None,
    abundance_trend: bool = True,
    robust: bool | None = None,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Fit trended NB Quasi-Likelihood GLM and perform Empirical Bayes QL F-test.

    Clean-room implementation of ``edgeR::estimateDisp`` + ``glmQLFit`` + ``glmQLFTest``
    used by ``csaw`` (Lun, Chen & Smyth 2016; Lun & Smyth, *NAR* 2016).

    Parameters
    ----------
    counts : WindowData | np.ndarray
        Window count matrix of shape ``(n_windows, n_samples)`` or ``WindowData``.
    design : np.ndarray | pd.DataFrame
        Experimental design matrix of shape ``(n_samples, n_params)``.
    contrast : np.ndarray | Sequence[float] | None
        1D contrast vector ``(n_params,)`` or 2D matrix ``(n_params, q)`` testing ``C^T beta = 0``.
    coef : int | Sequence[int] | None
        0-based column index (or indices) of ``design`` to test. Defaults to the last column
        when both ``contrast`` and ``coef`` are ``None``.
    norm_factors : np.ndarray | Sequence[float] | None
        TMM normalization factors from :func:`norm_factors`. Ignored when ``offsets`` is given.
    offsets : np.ndarray | None
        Natural-log offset matrix ``(n_windows, n_samples)`` from :func:`norm_offsets`.
    totals : np.ndarray | None
        Library sizes ``(n_samples,)`` when ``counts`` is a raw numpy array.
    abundance_trend : bool
        Fit abundance-trended NB and prior QL dispersions.
    robust : bool | None
        Use robust ``squeezeVar`` Empirical Bayes moderation. Defaults to ``True`` under
        ``settings="default"`` and ``False`` under ``settings="R_default"`` unless specified.
    settings : {"default", "R_default"}
        Parameter preset mode.

    Returns
    -------
    pd.DataFrame
        Per-window statistical results with columns ``logFC``, ``logCPM``, ``F_stat``,
        ``pval``, ``FDR``, ``ql_disp``, ``nb_disp``, and ``df_total``.
    """
    validate_settings(settings)

    if isinstance(counts, WindowData):
        y = counts.counts.astype(float)
        libs = counts.totals.astype(float) if totals is None else np.asarray(totals, dtype=float)
    else:
        y = np.asarray(counts, dtype=float)
        libs = np.sum(y, axis=0) if totals is None else np.asarray(totals, dtype=float)

    x = np.asarray(design, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_win, n_samp = y.shape
    if x.shape[0] != n_samp:
        raise ValueError(
            f"design rows ({x.shape[0]}) must match number of samples ({n_samp})."
        )
    p = x.shape[1]
    if n_samp <= p:
        raise ValueError(
            f"Quasi-likelihood F-test requires residual degrees of freedom (n_samples={n_samp} > p={p})."
        )

    libs = np.maximum(libs.ravel(), 1.0)
    if offsets is not None:
        off_mat = np.asarray(offsets, dtype=float)
        if off_mat.shape != (n_win, n_samp):
            raise ValueError(
                f"offsets shape {off_mat.shape} must match counts shape {(n_win, n_samp)}."
            )
        eff_libs = np.exp(np.mean(off_mat, axis=0))
    else:
        nf = np.ones(n_samp, dtype=float) if norm_factors is None else np.asarray(norm_factors, dtype=float).ravel()
        if nf.shape[0] != n_samp:
            raise ValueError(
                f"norm_factors length ({nf.shape[0]}) must match n_samples ({n_samp})."
            )
        eff_libs = np.maximum(libs * nf, 1.0)
        off_mat = np.broadcast_to(np.log(eff_libs)[None, :], (n_win, n_samp)).copy()

    eff_robust = (settings == "default") if robust is None else bool(robust)
    c_mat, x0, q_df = _build_contrast_and_null_design(x, contrast=contrast, coef=coef)

    if n_win == 0:
        return pd.DataFrame(
            columns=["logFC", "logCPM", "F_stat", "pval", "FDR", "ql_disp", "nb_disp", "df_total"]
        )

    log_cpm = ave_log_cpm(y, lib_sizes=eff_libs, prior_count=2.0)
    nb_disp = _estimate_trended_nb_disp(
        y, x, off_mat, log_cpm, abundance_trend=abundance_trend
    )

    beta_full, mu_full, dev_full = _fit_nb_glm_irls(y, x, off_mat, nb_disp)
    _, _, dev_null = _fit_nb_glm_irls(y, x0, off_mat, nb_disp)

    df_resid = _compute_residual_df(y, mu_full, x)
    s2_raw = dev_full / df_resid
    s2_post, d0_vec, _ = _squeeze_var(
        s2_raw,
        df_resid,
        log_cpm,
        abundance_trend=abundance_trend,
        robust=eff_robust,
    )

    lr_stat = np.maximum(0.0, dev_null - dev_full)
    f_stat = (lr_stat / float(q_df)) / np.maximum(s2_post, 1e-12)
    df_total = d0_vec + df_resid
    pvals = f_dist.sf(f_stat, float(q_df), df_total)
    pvals = np.clip(pvals, 0.0, 1.0)
    fdr = benjamini_hochberg(pvals)

    log2_fc_mat = (beta_full @ c_mat) / np.log(2.0)
    if log2_fc_mat.shape[1] == 1:
        primary_lfc = log2_fc_mat[:, 0]
    else:
        max_idx = np.argmax(np.abs(log2_fc_mat), axis=1)
        primary_lfc = log2_fc_mat[np.arange(n_win), max_idx]

    out = pd.DataFrame(
        {
            "logFC": primary_lfc,
            "logCPM": log_cpm,
            "F_stat": f_stat,
            "pval": pvals,
            "FDR": fdr,
            "ql_disp": s2_post,
            "nb_disp": nb_disp,
            "df_total": df_total,
        }
    )
    if log2_fc_mat.shape[1] > 1:
        for k in range(log2_fc_mat.shape[1]):
            out[f"logFC_{k + 1}"] = log2_fc_mat[:, k]
    return out
