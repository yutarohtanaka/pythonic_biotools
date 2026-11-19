"""Python implementation of the R `mr.raps` package (Zhao et al., Annals of Statistics 2020; GPL-3.0-or-later).

Implements Robust Adjusted Profile Score (RAPS) estimators for two-sample summary-data Mendelian
Randomization accounting for weak-instrument measurement error, systematic overdispersion (`tau2`),
and idiosyncratic pleiotropic outliers via L2, Huber, or Tukey biweight loss functions.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
from scipy import integrate, optimize, stats

EPS = np.finfo(float).eps
TOL = EPS ** 0.5
DEFAULT_K = {"l2": 2.0, "huber": 1.345, "tukey": 4.685}


def rho_l2(r, k: float = 2.0, deriv: int = 0):
    r = np.asarray(r, dtype=float)
    return r ** 2 / k if deriv == 0 else (r if deriv == 1 else np.ones_like(r))


def rho_huber(r, k: float = 1.345, deriv: int = 0):
    r = np.asarray(r, dtype=float)
    inside = np.abs(r) <= k
    if deriv == 0:
        return np.where(inside, r ** 2 / 2.0, k * (np.abs(r) - k / 2.0))
    if deriv == 1:
        return np.where(inside, r, k * np.sign(r))
    return np.where(inside, 1.0, 0.0)


def rho_tukey(r, k: float = 4.685, deriv: int = 0):
    r = np.asarray(r, dtype=float)
    if deriv == 0:
        return np.minimum(1.0 - (1.0 - (r / k) ** 2) ** 3, 1.0)
    if deriv == 1:
        return r * (1.0 - (r / k) ** 2) ** 2 * (np.abs(r) <= k)
    t = (r / k) ** 2
    return np.where(t < 1.0, (1.0 - t) * (1.0 - 5.0 * t), 0.0)


def make_rho(loss: str, k: float | None = None):
    kk = DEFAULT_K[loss] if k is None else k
    f = {"l2": rho_l2, "huber": rho_huber, "tukey": rho_tukey}[loss]
    return (lambda r, deriv=0: f(r, kk, deriv)), kk


def _qint(f) -> float:
    return float(
        integrate.quad(f, -40, 40, points=[-4.685, -1.345, 1.345, 4.685], limit=500, epsabs=1e-13, epsrel=1e-13)[0]
    )


def loss_constants(rho) -> tuple[float, float, float, float]:
    dn = stats.norm.pdf
    delta = _qint(lambda x: x * rho(x, 1) * dn(x))
    c1 = _qint(lambda x: rho(x, 1) ** 2 * dn(x))
    c2 = _qint(lambda x: x ** 2 * rho(x, 1) ** 2 * dn(x)) - delta ** 2
    c3 = _qint(lambda x: x ** 2 * rho(x, 2) * dn(x))
    return delta, c1, c2, c3


def _opt_max(f, lo: float, hi: float) -> float:
    return float(optimize.minimize_scalar(lambda x: -f(x), bounds=(lo, hi), method="bounded", options={"xatol": TOL}).x)


def _bound(b_out: np.ndarray, b_exp: np.ndarray, mult: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        q = float(np.nanquantile(np.abs(b_out / b_exp), 0.95))
    return q * mult


def _prune_large_se(b_exp, b_out, se_exp, se_out):
    s = se_exp > 10.0 * np.median(se_exp)
    if s.any():
        print(f"[mrraps] pruning {int(s.sum())} SNP(s) with extraordinarily large se_exp", file=sys.stderr)
    keep = ~s
    return b_exp[keep], b_out[keep], se_exp[keep], se_out[keep]


def raps_simple(b_exp, b_out, se_exp, se_out) -> dict:
    def ll(beta):
        return -0.5 * np.sum((b_out - b_exp * beta) ** 2 / (se_out ** 2 + se_exp ** 2 * beta ** 2))

    bound = _bound(b_out, b_exp, 2)
    beta = _opt_max(ll, -bound, bound)
    while abs(beta) > 0.95 * bound:
        bound *= 2
        beta = _opt_max(ll, -bound, bound)
    v = se_out ** 2 + beta ** 2 * se_exp ** 2
    score_var = np.sum(
        ((b_exp ** 2 - se_exp ** 2) * se_out ** 2 + (b_out ** 2 - se_out ** 2) * se_exp ** 2 + se_exp ** 2 * se_out ** 2)
        / v ** 2
    )
    info = np.sum(((b_exp ** 2 - se_exp ** 2) * se_out ** 2 + (b_out ** 2 - se_out ** 2) * se_exp ** 2) / v ** 2)
    se = float(np.sqrt(score_var / info ** 2))
    return dict(
        beta_hat=beta,
        beta_se=se,
        tau2_hat=0.0,
        tau2_se=0.0,
        naive_se=float(np.sqrt(1.0 / info)),
        chi_sq_test=float(np.sum((b_out - beta * b_exp) ** 2 / v)),
        beta_p_value=float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / se))),
    )


def raps_simple_robust(b_exp, b_out, se_exp, se_out, loss: str = "huber", k: float | None = None) -> dict:
    rho, _ = make_rho(loss, k)
    delta, c1, _, _ = loss_constants(rho)

    def ll(beta):
        return -np.sum(rho((b_out - b_exp * beta) / np.sqrt(se_out ** 2 + se_exp ** 2 * beta ** 2)))

    bound = _bound(b_out, b_exp, 2)
    beta = _opt_max(ll, -bound, bound)
    while abs(beta) > 0.95 * bound:
        bound *= 2
        beta = _opt_max(ll, -bound, bound)
    v = se_out ** 2 + beta ** 2 * se_exp ** 2
    score_var = c1 * np.sum(
        ((b_exp ** 2 - se_exp ** 2) * se_out ** 2 + (b_out ** 2 - se_out ** 2) * se_exp ** 2 + se_exp ** 2 * se_out ** 2)
        / v ** 2
    )
    info = delta * np.sum(((b_exp ** 2 - se_exp ** 2) * se_out ** 2 + (b_out ** 2 - se_out ** 2) * se_exp ** 2) / v ** 2)
    se = float(np.sqrt(score_var / info ** 2))
    return dict(
        beta_hat=beta,
        beta_se=se,
        tau2_hat=0.0,
        tau2_se=0.0,
        naive_se=float(np.sqrt(1.0 / info)),
        chi_sq_test=float(np.sum((b_out - beta * b_exp) ** 2 / v)),
        beta_p_value=float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / se))),
    )


def raps_overdispersed(b_exp, b_out, se_exp, se_out, pruning: bool = True, niter: int = 20, tol: float = TOL) -> dict:
    if pruning:
        b_exp, b_out, se_exp, se_out = _prune_large_se(b_exp, b_out, se_exp, se_out)

    def ll_fixbeta(beta, tau2):
        w = tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2
        return -0.5 * np.sum(se_exp ** 2 * np.log(w)) - 0.5 * np.sum(se_exp ** 2 * (b_out - b_exp * beta) ** 2 / w)

    def ll_fixtau(beta, tau2):
        return -0.5 * np.sum((b_out - b_exp * beta) ** 2 / (tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2))

    bound_beta = _bound(b_out, b_exp, 10)
    bound_tau2 = float(np.quantile(se_out ** 2, 0.95) * 100)
    beta = raps_simple(b_exp, b_out, se_exp, se_out)["beta_hat"]
    tau2 = 0.0
    for _ in range(niter):
        beta_old, tau2_old = beta, tau2
        tau2 = _opt_max(lambda t: ll_fixbeta(beta, t), 0.0, bound_tau2)
        if tau2 > bound_tau2 * 0.95:
            warnings.warn("Estimated overdispersion seems abnormally large.")
        beta = _opt_max(lambda b: ll_fixtau(b, tau2), -bound_beta, bound_beta)
        ext = 0
        while abs(beta) > 0.95 * bound_beta and ext <= niter:
            ext += 1
            bound_beta *= 2
            beta = _opt_max(lambda b: ll_fixtau(b, tau2), -bound_beta, bound_beta)
        if ext == niter:
            raise RuntimeError("Failed to find beta.")
        if abs(beta_old - beta) / abs(beta + 1e-10) + abs(tau2_old - tau2) / abs(tau2 + 1e-10) <= tol:
            break

    w = tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2
    a = (b_exp ** 2 - se_exp ** 2) * (tau2 + se_out ** 2) + (b_out ** 2 - tau2 - se_out ** 2) * se_exp ** 2
    score_var = np.diag([np.sum((a + se_exp ** 2 * (tau2 + se_out ** 2)) / w ** 2), np.sum(2.0 * se_exp ** 4 / w ** 2)])
    info = np.array([[-np.sum(a / w ** 2), -np.sum(se_exp ** 2 * beta / w ** 2)], [0.0, -np.sum(se_exp ** 2 / w ** 2)]])
    inv = np.linalg.inv(info)
    av = inv @ score_var @ inv.T
    return dict(
        beta_hat=beta,
        tau2_hat=tau2,
        beta_se=float(np.sqrt(av[0, 0])),
        tau2_se=float(np.sqrt(av[1, 1])),
        beta_p_value=float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / np.sqrt(av[0, 0])))),
    )


def raps_overdispersed_robust(
    b_exp,
    b_out,
    se_exp,
    se_out,
    loss: str = "huber",
    k: float | None = None,
    pruning: bool = True,
    niter: int = 20,
    tol: float = TOL,
) -> dict:
    rho, _ = make_rho(loss, k)
    if pruning:
        b_exp, b_out, se_exp, se_out = _prune_large_se(b_exp, b_out, se_exp, se_out)
    delta, c1, c2, c3 = loss_constants(rho)

    def ll_fixtau(beta, tau2):
        return -0.5 * np.sum(rho((b_out - b_exp * beta) / np.sqrt(np.maximum(0.0, tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2))))

    def robust_E(beta, tau2):
        w = tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2
        t = (b_out - beta * b_exp) / np.sqrt(np.maximum(0.0, w))
        return se_exp ** 2 * (t * rho(t, 1) - delta) / w

    fit = raps_overdispersed(b_exp, b_out, se_exp, se_out, pruning=False)
    beta, tau2 = fit["beta_hat"], fit["tau2_hat"]
    bound_beta = _bound(b_out, b_exp, 10)
    bound_tau2 = float(np.quantile(se_out ** 2, 0.95) * 100)

    def solve_tau(beta_val):
        f = lambda t: np.sum(robust_E(beta_val, t))
        lo, hi = 0.0, bound_tau2
        flo, fhi = f(lo), f(hi)
        n = 0
        while flo * fhi > 0 and n < 30:
            hi *= 2
            fhi = f(hi)
            n += 1
        if flo * fhi > 0:
            warnings.warn("Did not find a solution for tau2.")
            return 0.0
        return float(optimize.brentq(f, lo, hi, xtol=bound_tau2 * EPS ** 0.25))

    for _ in range(niter):
        beta_old, tau2_old = beta, tau2
        tau2 = max(solve_tau(beta), 0.0)
        if tau2 > bound_tau2 * 0.95:
            warnings.warn("Estimated overdispersion seems abnormally large.")
        beta = float(
            optimize.minimize(lambda b: -ll_fixtau(b[0], tau2), [beta], method="L-BFGS-B", bounds=[(-bound_beta, bound_beta)]).x[0]
        )
        ext = 0
        while abs(beta) > 0.95 * bound_beta and ext <= niter:
            ext += 1
            bound_beta *= 2
            beta = _opt_max(lambda b: ll_fixtau(b, tau2), -bound_beta, bound_beta)
        if ext == niter:
            raise RuntimeError("Failed to find beta.")
        if abs(beta_old - beta) / abs(beta + 1e-10) + abs(tau2_old - tau2) / abs(tau2 + 1e-10) <= tol:
            break

    w = tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2
    a = (b_exp ** 2 - se_exp ** 2) * (tau2 + se_out ** 2) + (b_out ** 2 - tau2 - se_out ** 2) * se_exp ** 2
    score_var = np.diag([c1 * np.sum((a + se_exp ** 2 * (tau2 + se_out ** 2)) / w ** 2), (c2 / 2.0) * np.sum(2.0 * se_exp ** 4 / w ** 2)])
    info = np.array([[-delta * np.sum(a / w ** 2), -delta * np.sum(se_exp ** 2 * beta / w ** 2)], [0.0, -(delta + c3) / 2.0 * np.sum(se_exp ** 2 / w ** 2)]])
    inv = np.linalg.inv(info)
    av = inv @ score_var @ inv.T
    return dict(
        beta_hat=beta,
        tau2_hat=tau2,
        beta_se=float(np.sqrt(av[0, 0])),
        tau2_se=float(np.sqrt(av[1, 1])),
        beta_p_value=float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / np.sqrt(av[0, 0])))),
    )


def raps_mle(
    b_exp: np.ndarray,
    b_out: np.ndarray,
    se_exp: np.ndarray,
    se_out: np.ndarray,
    over_dispersion: bool = False,
    loss: str = "l2",
    k: float | None = None,
    pruning: bool = True,
) -> dict:
    """Port of `mr.raps::mr.raps.mle`."""
    b_exp, b_out, se_exp, se_out = (np.asarray(v, dtype=float) for v in (b_exp, b_out, se_exp, se_out))
    if loss == "l2":
        return raps_overdispersed(b_exp, b_out, se_exp, se_out, pruning=pruning) if over_dispersion else raps_simple(b_exp, b_out, se_exp, se_out)
    kk = DEFAULT_K[loss] if k is None else k
    if over_dispersion:
        return raps_overdispersed_robust(b_exp, b_out, se_exp, se_out, loss, kk, pruning=pruning)
    return raps_simple_robust(b_exp, b_out, se_exp, se_out, loss, kk)


def raps_shrinkage(
    b_exp: np.ndarray,
    b_out: np.ndarray,
    se_exp: np.ndarray,
    se_out: np.ndarray,
    over_dispersion: bool = True,
    loss: str = "huber",
    k: float | None = None,
    num_init: int = 10,
    rng=None,
) -> dict:
    """Port of `mr.raps::mr.raps.shrinkage` (`shrinkage = FALSE` profile-score root finder)."""
    b_exp, b_out, se_exp, se_out = (np.asarray(v, dtype=float) for v in (b_exp, b_out, se_exp, se_out))
    rng = np.random.default_rng(rng)
    rho, k = make_rho(loss, k)
    delta, c1, c2, c3 = loss_constants(rho)

    if np.sum(b_exp ** 2 / se_exp ** 2) < len(b_exp) - np.sqrt(len(b_exp)):
        warnings.warn("Average instrument F-statistic is very small; causal effect may be weakly identified.")

    def get_t(beta, tau2):
        return (b_out - b_exp * beta) / np.sqrt(tau2 + se_out ** 2 + se_exp ** 2 * beta ** 2)

    def gamma_hat(beta, tau2, deriv="0"):
        denom = 1.0 / se_exp ** 2 + beta ** 2 / (se_out ** 2 + tau2)
        if deriv == "0":
            g = (b_exp / se_exp ** 2 + beta * b_out / (se_out ** 2 + tau2)) / denom
            return np.where(se_exp == 0, b_exp, g)
        if deriv == "beta":
            return (b_out / (se_out ** 2 + tau2)) / denom
        return -beta * b_out / (se_out ** 2 + tau2) ** 2 / denom

    def psi(param):
        beta = param[0]
        tau2 = param[1] if over_dispersion else 0.0
        t = get_t(beta, tau2)
        g = gamma_hat(beta, tau2)
        v = beta ** 2 * se_exp ** 2 + se_out ** 2 + tau2
        p1 = np.sum(g * rho(t, 1) / np.sqrt(v))
        if over_dispersion:
            return np.array([p1, np.sum((t * rho(t, 1) - delta) / v)])
        return np.array([p1])

    sgn = lambda n: rng.choice([-1, 1], n)
    res1 = raps_mle(b_exp, b_out, se_exp, se_out, over_dispersion, "l2")
    res2 = raps_mle(b_exp, b_out, se_exp, se_out, over_dispersion, "huber")
    pert1 = abs(res1["beta_hat"]) * rng.exponential(size=num_init - 1) * sgn(num_init - 1)
    pert2 = abs(res2["beta_hat"]) * rng.exponential(size=num_init - 1) * sgn(num_init - 1)
    if over_dispersion:
        init = np.array([[res1["beta_hat"], res1["tau2_hat"]], [res2["beta_hat"], res2["tau2_hat"]]])
        extra = np.column_stack([np.concatenate([pert1, pert2]), np.zeros(2 * (num_init - 1))])
        init = np.vstack([init, init[np.repeat([0, 1], num_init - 1)] + extra])
    else:
        init = np.array([res1["beta_hat"], res2["beta_hat"]])
        init = np.concatenate([init, init[np.repeat([0, 1], num_init - 1)] + np.concatenate([pert1, pert2])])[:, None]

    roots, betas = [], np.full(len(init), np.nan)
    for i, x0 in enumerate(init):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = optimize.root(psi, x0, method="hybr", tol=1e-9)
        roots.append(sol.x)
        ok = np.all(np.isfinite(sol.x)) and sol.success
        if over_dispersion and ok and sol.x[1] > np.median(se_out) * 10:
            ok = False
        if ok:
            betas[i] = sol.x[0]

    beta_init = raps_mle(b_exp, b_out, se_exp, se_out, over_dispersion, loss)["beta_hat"]
    if np.all(np.isnan(betas)):
        warnings.warn("Cannot find solution with finite overdispersion. Using tau2 = 0.")
        return raps_shrinkage(b_exp, b_out, se_exp, se_out, False, loss, k, num_init, rng)
    j = int(np.nanargmin(np.abs(betas - beta_init)))
    others = [
        float(b)
        for b in betas
        if np.isfinite(b) and abs(b - betas[j]) > 1e-4 and abs(b - beta_init) < 100 * max(abs(betas[j] - beta_init), abs(beta_init) / 10)
    ]
    root = roots[j]
    if over_dispersion and root[1] < 0:
        warnings.warn("Estimated overdispersion is negative. Using tau2 = 0.")
        return raps_shrinkage(b_exp, b_out, se_exp, se_out, False, loss, k, num_init, rng)

    beta = float(root[0])
    tau2 = float(root[1]) if over_dispersion else 0.0
    t = get_t(beta, tau2)
    g = gamma_hat(beta, tau2)
    v = beta ** 2 * se_exp ** 2 + se_out ** 2 + tau2
    dg_b = gamma_hat(beta, tau2, "beta")
    v22 = np.sum((dg_b * rho(t, 1) + g * delta * (-b_exp) / np.sqrt(v)) / np.sqrt(v))
    if not over_dispersion:
        beta_se = float(np.sqrt(c1 * np.sum(g ** 2 / v) / v22 ** 2))
        tau2_se = 0.0
    else:
        V1 = np.diag([c1 * np.sum(g ** 2 / v), c2 * np.sum(1.0 / v ** 2)])
        dg_t = gamma_hat(beta, tau2, "tau2")
        V2 = np.array([[v22, np.sum(dg_t * rho(t, 1) / np.sqrt(v))], [0.0, (delta + c3) / 2.0 * np.sum(1.0 / v ** 2)]])
        inv = np.linalg.inv(V2)
        V = inv @ V1 @ inv.T
        beta_se, tau2_se = float(np.sqrt(V[0, 0])), float(np.sqrt(V[1, 1]))

    pval = float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / max(beta_se, 1e-300))))
    return dict(
        beta_hat=beta,
        beta_se=beta_se,
        beta_p_value=pval,
        tau2_hat=tau2,
        tau2_se=tau2_se,
        beta_init=beta_init,
        n_roots_other=len(others),
        other_roots=others,
    )


def mr_raps(
    b_exp: np.ndarray,
    b_out: np.ndarray,
    se_exp: np.ndarray,
    se_out: np.ndarray,
    over_dispersion: bool = True,
    loss_function: str = "huber",
    estimator: str = "shrinkage",
    k: float | None = None,
    num_init: int = 10,
    seed: int | None = None,
) -> dict:
    """Fit `mr.raps` on two-sample exposure `(b_exp, se_exp)` and outcome `(b_out, se_out)` summary statistics."""
    if estimator == "shrinkage":
        return raps_shrinkage(b_exp, b_out, se_exp, se_out, over_dispersion=over_dispersion, loss=loss_function, k=k, num_init=num_init, rng=seed)
    if estimator == "mle":
        return raps_mle(b_exp, b_out, se_exp, se_out, over_dispersion=over_dispersion, loss=loss_function, k=k)
    raise ValueError("estimator must be 'shrinkage' or 'mle'")
