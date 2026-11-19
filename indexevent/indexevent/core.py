"""Python implementation of the R `indexevent` package (Dudbridge et al., Nature Communications 2019).

Implements CWLS, Hedges-Olkin, and SIMEX (Simulation-Extrapolation with profile-likelihood CIs)
estimators for correcting regression dilution when adjusting a subsequent-trait GWAS (`ybeta`)
for selection on an index trait (`xbeta`).
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import optimize, stats

_TOL = np.finfo(float).eps ** 0.25


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
    return coef, np.sqrt(np.diag(cov)), resid, cov


def _optimise(f, lower: float, upper: float) -> tuple[float, float]:
    r = optimize.minimize_scalar(f, bounds=(lower, upper), method="bounded", options={"xatol": _TOL})
    return float(r.x), float(r.fun)


def simexllhd(pvar: float, pmean: float, simex_estimates: np.ndarray) -> float:
    """Negative log-likelihood of SIMEX slope estimates (`indexevent::simexllhd`)."""
    lam = simex_estimates[:, 0]
    mean = pmean / (1.0 + (lam + 1.0) * np.exp(np.clip(pvar, -50.0, 50.0)))
    sd = np.sqrt(np.maximum(simex_estimates[:, 2], 1e-30))
    return float(-np.sum(stats.norm.logpdf(simex_estimates[:, 1], loc=mean, scale=sd)))


def simexprofilellhd(p: float, simex_estimates: np.ndarray, variance_ratio: float) -> float:
    """Profile negative log-likelihood over the nuisance variance ratio (`indexevent::simexprofilellhd`)."""
    lam = simex_estimates[:, 0]
    lb_raw = p ** 2 * variance_ratio - 1.0
    if lb_raw > 0.0:
        lb = max(float(np.max(np.log(lb_raw / (lam + 1.0)))), -10.0)
    else:
        lb = -10.0
    ub = max(10.0, lb + 10.0)
    _, obj = _optimise(lambda v: simexllhd(v, p, simex_estimates), lb, ub)
    return obj


def _bracket_and_brentq(g, pmin: float, direction: float, bound: float) -> float:
    """Step geometrically from pmin (where g(pmin) < 0) toward bound to bracket root g(p) = 0 for brentq."""
    prev = pmin
    step = 0.005
    while True:
        cand = pmin + direction * step
        hit_bound = (cand <= bound) if direction < 0 else (cand >= bound)
        if hit_bound:
            cand = bound
        gc = g(cand)
        if gc >= 0.0:
            a, b = (cand, prev) if direction < 0 else (prev, cand)
            return float(optimize.brentq(g, a, b, xtol=_TOL))
        if hit_bound:
            break
        prev = cand
        step *= 2.0
    lo, hi = (bound, pmin) if direction < 0 else (pmin, bound)
    fallback, _ = _optimise(lambda p: g(p) ** 2, lo, hi)
    return fallback


def simexprofileCI(simex_estimates: np.ndarray, variance_ratio: float) -> np.ndarray:
    """MLE and 95% profile-likelihood confidence interval of the SIMEX slope (`indexevent::simexprofileCI`)."""
    f = lambda p: simexprofilellhd(p, simex_estimates, variance_ratio)
    b0 = float(np.clip(simex_estimates[0, 1], -95.0, 95.0))
    p_loc, obj_loc = _optimise(f, max(-100.0, b0 - 5.0), min(100.0, b0 + 5.0))
    p_glb, obj_glb = _optimise(f, -100.0, 100.0)
    pmin, pobj = (p_loc, obj_loc) if obj_loc <= obj_glb else (p_glb, obj_glb)

    crit = float(stats.chi2.ppf(0.95, 1))
    g = lambda p: 2.0 * (f(p) - pobj) - crit
    lo = _bracket_and_brentq(g, pmin, -1.0, -100.0)
    hi = _bracket_and_brentq(g, pmin, 1.0, 100.0)
    lo_s, hi_s = float(min(lo, hi)), float(max(lo, hi))
    return np.array([pmin, lo_s, hi_s])


def _resid_sandwich(weight: np.ndarray, x: np.ndarray, res: np.ndarray) -> float:
    sw = weight.sum()
    swx = (weight * x).sum()
    denom = (sw * np.sum(weight * x ** 2) - swx ** 2) ** 2
    if denom <= 0.0:
        return 1e-12
    return float(
        (
            swx ** 2 * np.sum(weight * res ** 2)
            - 2.0 * sw * swx * np.sum(weight * x * res ** 2)
            + sw ** 2 * np.sum(weight * x ** 2 * res ** 2)
        )
        / denom
    )


def indexevent(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    weighted: bool = True,
    prune: np.ndarray | None = None,
    method: str = "CWLS",
    B: int = 10,
    lambdas: np.ndarray | None = None,
    seed: int = 2018,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Adjust summary statistics for index-event bias (`indexevent::indexevent`)."""
    _validate_settings(settings)
    if not isinstance(method, str):
        raise ValueError("method must be a string: 'CWLS', 'Hedges-Olkin', or 'Simex'")
    m = method.strip().lower()
    if m not in ("cwls", "hedges-olkin", "hedges_olkin", "simex"):
        raise ValueError(f"method must be one of 'CWLS', 'Hedges-Olkin', or 'Simex' (got {method!r})")

    lambdas_arr = np.arange(0.25, 5.0 + 1e-9, 0.25) if lambdas is None else np.asarray(lambdas, dtype=float)
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    prune_idx = np.arange(len(xbeta)) if prune is None else np.asarray(prune, dtype=int)
    xb_raw, xs, yb_raw, ys = xbeta[prune_idx], xse[prune_idx], ybeta[prune_idx], yse[prune_idx]

    if settings == "R_default":
        # Upstream indexevent.R fits raw signed WLS without orienting xb >= 0
        xb = xb_raw
        yb = yb_raw
    else:
        # Orient xb >= 0 (using +1 when xb == 0) for exact allele-coding invariance
        signs = np.where(xb_raw >= 0.0, 1.0, -1.0)
        xb = xb_raw * signs
        yb = yb_raw * signs
    weight = 1.0 / ys ** 2 if weighted else np.ones(len(prune_idx))

    coef, _, res, cov = _wls(yb, np.column_stack([np.ones(len(xb)), xb]), weight)
    b_raw, b_se_raw = float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
    z975 = float(stats.norm.ppf(0.975))
    simex_estimates = None

    if m == "cwls":
        sw = float(weight.sum())
        numer = float(sw * np.sum(weight * xb * yb) - np.sum(weight * xb) * np.sum(weight * yb))
        denom_raw = float(sw * np.sum(weight * xb ** 2) - np.sum(weight * xb) ** 2)
        denom = float(denom_raw - sw * np.sum(weight * xs ** 2))
        b = float(numer / denom)
        if abs(b_raw) >= 1e-12:
            scale = abs(b / b_raw)
        else:
            scale = abs(denom_raw / denom) if abs(denom) > 1e-30 else 1.0
        half_w = z975 * b_se_raw * scale
        b_ci = np.sort(np.array([b - half_w, b + half_w], dtype=float))
    elif m in ("hedges-olkin", "hedges_olkin"):
        var_xb = float(np.var(xb, ddof=1))
        ho = float(var_xb / (var_xb - float(np.mean(xs ** 2))))
        b = float(b_raw * ho)
        half_w = z975 * b_se_raw * abs(ho)
        b_ci = np.sort(np.array([b - half_w, b + half_w], dtype=float))
    else:  # m == "simex"
        rng = np.random.default_rng(seed)
        est = [b_raw]
        var_sw = [_resid_sandwich(weight, xb, res)]
        for lam in lambdas_arr:
            coefs = np.empty(B)
            sim = xb
            r2 = res
            for it in range(B):
                sim = rng.normal(xb, xs * np.sqrt(lam))
                c2, _, r2, _ = _wls(yb, np.column_stack([np.ones(len(xb)), sim]), weight)
                coefs[it] = c2[1]
            est.append(float(coefs.mean()))
            var_sw.append(_resid_sandwich(weight, sim, r2) / B)
        simex_estimates = np.column_stack([np.concatenate([[0.0], lambdas_arr]), est, var_sw])
        var_y = max(float(np.var(yb, ddof=1)), 1e-30)
        mle = simexprofileCI(simex_estimates, float(np.var(xb, ddof=1) / var_y))
        b = float(mle[0])
        b_ci = np.sort(np.asarray(mle[1:3], dtype=float))

    b_se = float(max((b_ci[1] - b_ci[0]) / (2.0 * z975), 0.0))
    ybeta_adj = ybeta - b * xbeta
    yse_adj = np.sqrt(yse ** 2 + b ** 2 * xse ** 2 + b_se ** 2 * xbeta ** 2 + b_se ** 2 * xse ** 2)
    chisq = (ybeta_adj / yse_adj) ** 2
    return dict(
        ybeta_adj=ybeta_adj,
        yse_adj=yse_adj,
        ychisq_adj=chisq,
        yp_adj=stats.chi2.sf(chisq, 1),
        b=b,
        b_se=b_se,
        b_ci=b_ci,
        b_raw=b_raw,
        simex_estimates=simex_estimates,
    )
