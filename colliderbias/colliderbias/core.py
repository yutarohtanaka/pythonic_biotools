"""Python implementation of the R `ColliderBias` package (Cai et al., 2022; Cai & Dudbridge, 2024).

Provides:
  * `method_cb` (`cwls`): Corrected Weighted Least Squares instrumental-effect regression (`ivw` or `egger`,
    with `weighted` `'0'`, `'1'`, or `'2'`).
  * `cwbls`: Corrected Weighted Bivariate Least Squares for estimating the direct effect of an exposure (`x*`)
    and intermediate/collider trait (`d*`) on an outcome (`y*`).
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import stats


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray):
    XtW = X.T * w
    XtWX = XtW @ X
    coef = np.linalg.solve(XtWX, XtW @ y)
    resid = y - X @ coef
    df = len(y) - X.shape[1]
    sigma2 = np.sum(w * resid ** 2) / df if df > 0 else np.nan
    cov = sigma2 * np.linalg.inv(XtWX)
    return coef, np.sqrt(np.maximum(np.diag(cov), 0.0)), resid, cov


def _fit_cwls_given_weights(xb: np.ndarray, xs: np.ndarray, yb: np.ndarray, weight: np.ndarray, model_l: str):
    if model_l == "egger":
        coef, se, _, _ = _wls(yb, np.column_stack([np.ones(len(xb)), xb]), weight)
        b_raw, b_raw_se = float(coef[1]), float(se[1])
        sw = float(np.sum(weight))
        swx = float(np.sum(weight * xb))
        swx2 = float(np.sum(weight * xb ** 2))
        num_ho = sw * swx2 - swx ** 2
        den_ho = num_ho - sw * float(np.sum(weight * xs ** 2))
        ho = num_ho / den_ho if abs(den_ho) > 1e-30 else 1.0
    elif model_l == "ivw":
        coef, se, _, _ = _wls(yb, xb[:, None], weight)
        b_raw, b_raw_se = float(coef[0]), float(se[0])
        swx2 = float(np.sum(weight * xb ** 2))
        den_ho = swx2 - float(np.sum(weight * xs ** 2))
        ho = swx2 / den_ho if abs(den_ho) > 1e-30 else 1.0
    else:
        raise ValueError("model must be 'ivw' or 'egger'")
    return b_raw, b_raw_se, float(ho)


def method_cb(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    prune: np.ndarray | None = None,
    weighted: str = "1",
    model: str = "ivw",
    rho_overlap: float = 0.0,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Port of `ColliderBias::methodCB` (CWLS estimator)."""
    _validate_settings(settings)
    if not (np.isfinite(rho_overlap) and abs(rho_overlap) <= 1.0):
        raise ValueError("rho_overlap must be finite and in [-1, 1].")

    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    prune_idx = np.arange(len(xbeta)) if prune is None else np.asarray(prune, dtype=int)
    xb, xs, yb, ys = xbeta[prune_idx], xse[prune_idx], ybeta[prune_idx], yse[prune_idx]
    weighted_s, model_l = str(weighted), model.lower()
    if model_l not in ("ivw", "egger"):
        raise ValueError("model must be 'ivw' or 'egger'")

    # Orient xb >= 0 for coding-invariant Egger regression in Pythonic default mode
    if settings == "default" and model_l == "egger":
        signs = np.where(xb >= 0.0, 1.0, -1.0)
        xb = xb * signs
        yb = yb * signs

    if weighted_s == "1":
        weight = 1.0 / (ys ** 2)
        b_raw, b_raw_se, ho = _fit_cwls_given_weights(xb, xs, yb, weight, model_l)
    elif weighted_s == "2":
        if settings == "R_default":
            xb_safe = np.where(xb == 0.0, 1e-12, xb)
            weight = 1.0 / (ys ** 2 + ((yb / xb_safe) ** 2) * (xs ** 2))
            b_raw, b_raw_se, ho = _fit_cwls_given_weights(xb, xs, yb, weight, model_l)
        else:
            # Iterate slope once from first-order weights so xb == 0 never divides by zero
            w1 = 1.0 / (ys ** 2)
            b_raw_1, _, ho_1 = _fit_cwls_given_weights(xb, xs, yb, w1, model_l)
            b_step1 = b_raw_1 * ho_1
            v2 = np.maximum(ys ** 2 + (b_step1 ** 2) * (xs ** 2) - 2.0 * b_step1 * rho_overlap * xs * ys, 1e-300)
            weight = 1.0 / v2
            b_raw, b_raw_se, ho = _fit_cwls_given_weights(xb, xs, yb, weight, model_l)
    elif weighted_s == "0":
        weight = np.ones(len(prune_idx))
        b_raw, b_raw_se, ho = _fit_cwls_given_weights(xb, xs, yb, weight, model_l)
    else:
        raise ValueError("weighted must be '0', '1', or '2'")

    z975 = float(stats.norm.ppf(0.975))
    b = float(b_raw * ho)
    b_se = float(b_raw_se * abs(ho))
    b_ci = [b - z975 * b_se, b + z975 * b_se]
    b_pval = float(2.0 * stats.norm.sf(abs(b) / max(b_se, 1e-300)))

    ybeta_adj = ybeta - b * xbeta
    if settings == "R_default":
        var_adj = np.maximum(yse ** 2 + (b ** 2) * (xse ** 2), 1e-300)
    else:
        var_adj = np.maximum(
            yse ** 2
            + (b ** 2) * (xse ** 2)
            + (b_se ** 2) * (xbeta ** 2)
            + (b_se ** 2) * (xse ** 2)
            - 2.0 * b * rho_overlap * xse * yse,
            1e-300,
        )
    yse_adj = np.sqrt(var_adj)
    ychisq_adj = (ybeta_adj / yse_adj) ** 2
    return dict(
        ybeta_adj=ybeta_adj,
        yse_adj=yse_adj,
        ychisq_adj=ychisq_adj,
        yp_adj=stats.chi2.sf(ychisq_adj, 1),
        b=b,
        b_se=b_se,
        b_ci=b_ci,
        b_pval=b_pval,
        b_raw=b_raw,
        b_raw_se=b_raw_se,
        ho_factor=ho,
    )


cwls = method_cb


def _cwbls_point(xbeta, xse, dbeta, dse, ybeta, yse) -> tuple[float, float]:
    w = 1.0 / (yse ** 2)
    s_dse2 = float(np.sum(w * dse ** 2))
    s_xse2 = float(np.sum(w * xse ** 2))
    s_d2 = float(np.sum(w * dbeta ** 2))
    s_x2 = float(np.sum(w * xbeta ** 2))
    s_xy = float(np.sum(w * xbeta * ybeta))
    s_xd = float(np.sum(w * xbeta * dbeta))
    s_dy = float(np.sum(w * dbeta * ybeta))

    s_xx_corr = s_x2 - s_xse2
    s_dd_corr = s_d2 - s_dse2
    det = s_xx_corr * s_dd_corr - s_xd ** 2
    if abs(det) <= 1e-30:
        return float("nan"), float("nan")
    beta_x = float((s_dd_corr * s_xy - s_xd * s_dy) / det)
    beta_d = float((s_xx_corr * s_dy - s_xd * s_xy) / det)
    return beta_x, beta_d


def cwbls(
    xbeta: np.ndarray,
    xse: np.ndarray,
    dbeta: np.ndarray,
    dse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    n_boot: int = 100,
    seed: int | None = None,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Port of `ColliderBias::CWBLS` (`x*` = exposure, `d*` = collider/disease, `y*` = outcome)."""
    _validate_settings(settings)
    arr = [np.asarray(v, dtype=float) for v in (xbeta, xse, dbeta, dse, ybeta, yse)]
    n = len(arr[0])
    b, beta_d = _cwbls_point(*arr)
    rng = np.random.default_rng(seed)
    reps_x = np.empty(n_boot)
    reps_d = np.empty(n_boot)
    for r in range(n_boot):
        i = rng.integers(0, n, n)
        reps_x[r], reps_d[r] = _cwbls_point(*[a[i] for a in arr])

    valid_x = reps_x[np.isfinite(reps_x)]
    valid_d = reps_d[np.isfinite(reps_d)]
    b_se = float(np.std(valid_x, ddof=1)) if len(valid_x) > 1 else float("nan")
    beta_d_se = float(np.std(valid_d, ddof=1)) if len(valid_d) > 1 else float("nan")

    z975 = float(stats.norm.ppf(0.975))
    b_ci = [b - z975 * b_se, b + z975 * b_se]
    beta_d_ci = [beta_d - z975 * beta_d_se, beta_d + z975 * beta_d_se]
    b_pval = float(2.0 * stats.norm.sf(abs(b) / max(b_se, 1e-300))) if np.isfinite(b_se) else float("nan")
    beta_d_pval = float(2.0 * stats.norm.sf(abs(beta_d) / max(beta_d_se, 1e-300))) if np.isfinite(beta_d_se) else float("nan")

    return {
        "b": b,
        "b_se": b_se,
        "b_ci": b_ci,
        "pval": b_pval,
        "b_pval": b_pval,
        "beta_d": beta_d,
        "beta_d_se": beta_d_se,
        "beta_d_ci": beta_d_ci,
        "beta_d_pval": beta_d_pval,
        "boot": reps_x,
        "boot_d": reps_d,
    }
