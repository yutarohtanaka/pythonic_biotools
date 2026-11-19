"""Python implementation of the R `ColliderBias` package (Cai et al., 2022; Cai & Dudbridge, 2024).

Provides:
  * `method_cb` (`cwls`): Corrected Weighted Least Squares instrumental-effect regression (`ivw` or `egger`,
    with `weighted` `'0'`, `'1'`, or `'2'`).
  * `cwbls`: Corrected Weighted Bivariate Least Squares for estimating the direct effect of an exposure (`x*`)
    on an outcome (`y*`) conditioned on an intermediate/collider trait (`d*`).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray):
    XtW = X.T * w
    XtWX = XtW @ X
    coef = np.linalg.solve(XtWX, XtW @ y)
    resid = y - X @ coef
    df = len(y) - X.shape[1]
    sigma2 = np.sum(w * resid ** 2) / df if df > 0 else np.nan
    cov = sigma2 * np.linalg.inv(XtWX)
    return coef, np.sqrt(np.diag(cov)), resid, cov


def method_cb(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    prune: np.ndarray | None = None,
    weighted: str = "1",
    model: str = "ivw",
) -> dict:
    """Port of `ColliderBias::methodCB` (CWLS estimator)."""
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    prune_idx = np.arange(len(xbeta)) if prune is None else np.asarray(prune, dtype=int)
    xb, xs, yb, ys = xbeta[prune_idx], xse[prune_idx], ybeta[prune_idx], yse[prune_idx]
    weighted_s, model_l = str(weighted), model.lower()

    if weighted_s == "1":
        weight = 1.0 / ys ** 2
    elif weighted_s == "2":
        weight = 1.0 / (ys ** 2 + (yb / xb) ** 2 * xs ** 2)
    elif weighted_s == "0":
        weight = np.ones(len(prune_idx))
    else:
        raise ValueError("weighted must be '0', '1', or '2'")

    if model_l == "egger":
        coef, se, _, _ = _wls(yb, np.column_stack([np.ones(len(xb)), xb]), weight)
        b_raw, b_raw_se = float(coef[1]), float(se[1])
        sw = float(np.sum(weight))
        swx = float(np.sum(weight * xb))
        swx2 = float(np.sum(weight * xb ** 2))
        ho = (sw * swx2 - swx ** 2) / (sw * swx2 - swx ** 2 - sw * float(np.sum(weight * xs ** 2)))
    elif model_l == "ivw":
        coef, se, _, _ = _wls(yb, xb[:, None], weight)
        b_raw, b_raw_se = float(coef[0]), float(se[0])
        swx2 = float(np.sum(weight * xb ** 2))
        ho = swx2 / (swx2 - float(np.sum(weight * xs ** 2)))
    else:
        raise ValueError("model must be 'ivw' or 'egger'")

    z975 = float(stats.norm.ppf(0.975))
    b = float(b_raw * ho)
    b_se = float(b_raw_se * abs(ho))
    b_ci = [b - z975 * b_se, b + z975 * b_se]

    ybeta_adj = ybeta - b * xbeta
    yse_adj = np.sqrt(yse ** 2 + b ** 2 * xse ** 2)
    ychisq_adj = (ybeta_adj / yse_adj) ** 2
    return dict(
        ybeta_adj=ybeta_adj,
        yse_adj=yse_adj,
        ychisq_adj=ychisq_adj,
        yp_adj=stats.chi2.sf(ychisq_adj, 1),
        b=b,
        b_se=b_se,
        b_ci=b_ci,
        b_raw=b_raw,
        b_raw_se=b_raw_se,
        ho_factor=ho,
    )


cwls = method_cb


def _cwbls_point(xbeta, xse, dbeta, dse, ybeta, yse) -> float:
    w = 1.0 / yse ** 2
    s_dse2 = np.sum(w * dse ** 2)
    s_xse2 = np.sum(w * xse ** 2)
    s_d2 = np.sum(w * dbeta ** 2)
    s_x2 = np.sum(w * xbeta ** 2)
    s_xy = np.sum(w * xbeta * ybeta)
    s_xd = np.sum(w * xbeta * dbeta)
    s_dy = np.sum(w * dbeta * ybeta)
    n_i = s_dse2 * s_xy
    d_i = s_x2 * s_dse2 + s_xse2 * s_d2 - s_xse2 * s_dse2
    return float((s_d2 * s_xy - s_xd * s_dy - n_i) / (s_x2 * s_d2 - s_xd ** 2 - d_i))


def cwbls(
    xbeta: np.ndarray,
    xse: np.ndarray,
    dbeta: np.ndarray,
    dse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    n_boot: int = 100,
    seed: int | None = None,
) -> dict:
    """Port of `ColliderBias::CWBLS` (`x*` = exposure, `d*` = collider/disease, `y*` = outcome)."""
    arr = [np.asarray(v, dtype=float) for v in (xbeta, xse, dbeta, dse, ybeta, yse)]
    n = len(arr[0])
    b = _cwbls_point(*arr)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for r in range(n_boot):
        i = rng.integers(0, n, n)
        reps[r] = _cwbls_point(*[a[i] for a in arr])
    return {"b": b, "b_se": float(np.std(reps, ddof=1)), "boot": reps}
