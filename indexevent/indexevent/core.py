"""Python implementation of the R `indexevent` package (Dudbridge et al., Nature Communications 2019).

Implements CWLS, Hedges-Olkin, and SIMEX (Simulation-Extrapolation with profile-likelihood CIs)
estimators for correcting regression dilution when adjusting a subsequent-trait GWAS (`ybeta`)
for selection on an index trait (`xbeta`).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

_TOL = np.finfo(float).eps ** 0.25


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
    mean = pmean / (1.0 + (lam + 1.0) * np.exp(pvar))
    sd = np.sqrt(simex_estimates[:, 2])
    return float(-np.sum(stats.norm.logpdf(simex_estimates[:, 1], loc=mean, scale=sd)))


def simexprofilellhd(p: float, simex_estimates: np.ndarray, variance_ratio: float) -> float:
    """Profile negative log-likelihood over the nuisance variance ratio (`indexevent::simexprofilellhd`)."""
    lam = simex_estimates[:, 0]
    lowerbound = (p ** 2 * variance_ratio - 1.0) / (lam + 1.0)
    if np.min(lowerbound) > 0:
        lowerbound = np.array([np.max(np.log(lowerbound / (lam + 1.0)))])
    interval = np.concatenate([np.atleast_1d(lowerbound), [10.0]])
    _, obj = _optimise(lambda v: simexllhd(v, p, simex_estimates), float(interval.min()), float(interval.max()))
    return obj


def simexprofileCI(simex_estimates: np.ndarray, variance_ratio: float) -> np.ndarray:
    """MLE and 95% profile-likelihood confidence interval of the SIMEX slope (`indexevent::simexprofileCI`)."""
    f = lambda p: simexprofilellhd(p, simex_estimates, variance_ratio)
    pmin, pobj = _optimise(f, -100.0, 100.0)
    crit = stats.chi2.ppf(0.95, 1)
    obj = lambda p: (2.0 * (f(p) - pobj) - crit) ** 2
    lo, _ = _optimise(obj, -100.0, pmin)
    hi, _ = _optimise(obj, pmin, 100.0)
    return np.array([pmin, lo, hi])


def _resid_sandwich(weight: np.ndarray, x: np.ndarray, res: np.ndarray) -> float:
    sw = weight.sum()
    swx = (weight * x).sum()
    return float(
        (
            swx ** 2 * np.sum(weight * res ** 2)
            - 2.0 * sw * swx * np.sum(weight * x * res ** 2)
            + sw ** 2 * np.sum(weight * x ** 2 * res ** 2)
        )
        / (sw * np.sum(weight * x ** 2) - swx ** 2) ** 2
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
) -> dict:
    """Adjust summary statistics for index-event bias (`indexevent::indexevent`)."""
    lambdas_arr = np.arange(0.25, 5.0 + 1e-9, 0.25) if lambdas is None else np.asarray(lambdas, dtype=float)
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    prune_idx = np.arange(len(xbeta)) if prune is None else np.asarray(prune, dtype=int)
    xb, xs, yb, ys = xbeta[prune_idx], xse[prune_idx], ybeta[prune_idx], yse[prune_idx]
    weight = 1.0 / ys ** 2 if weighted else np.ones(len(prune_idx))

    coef, _, res, cov = _wls(yb, np.column_stack([np.ones(len(xb)), xb]), weight)
    b_raw, b_se_raw = float(coef[1]), float(np.sqrt(cov[1, 1]))
    m = method.lower()
    z975 = float(stats.norm.ppf(0.975))
    simex_estimates = None

    if "cwls".startswith(m):
        sw = weight.sum()
        numer = sw * np.sum(weight * xb * yb) - np.sum(weight * xb) * np.sum(weight * yb)
        denom = sw * np.sum(weight * xb ** 2) - np.sum(weight * xb) ** 2 - sw * np.sum(weight * xs ** 2)
        b = float(numer / denom)
        b_ci = np.array([b - z975 * b_se_raw * b / b_raw, b + z975 * b_se_raw * b / b_raw])
    elif "hedges-olkin".startswith(m):
        ho = float(np.var(xb, ddof=1) / (np.var(xb, ddof=1) - np.mean(xs ** 2)))
        b = float(b_raw * ho)
        b_ci = np.array([b - z975 * b_se_raw * ho, b + z975 * b_se_raw * ho])
    elif "simex".startswith(m):
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
        mle = simexprofileCI(simex_estimates, float(np.var(xb, ddof=1) / np.var(yb, ddof=1)))
        b, b_ci = float(mle[0]), mle[1:3]
    else:
        raise ValueError("method must be one of 'CWLS', 'Hedges-Olkin', or 'Simex'")

    b_se = float((b_ci[1] - b_ci[0]) / (2.0 * z975))
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
