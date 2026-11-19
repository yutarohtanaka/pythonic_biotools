"""Python implementation of MR-Horse and MVMR-Horse (Grant & Burgess, AJHG 2024).

Samples the exact Bayesian Horseshoe Mendelian Randomization posterior for univariable (`K = 1`)
and multivariable (`K >= 1`) summary data accounting for both correlated and uncorrelated
horizontal pleiotropy without requiring external JAGS installation.
"""
from __future__ import annotations

import numpy as np
from scipy import special

GRID = 200


def _qform(bx0, mx, s, vx0, rho, alpha, phi, tau):
    K = bx0.shape[1]
    coef = rho * alpha / (phi * tau)
    z = bx0 - mx[None] - coef[:, None] * s[None]
    kappa = -rho ** 2 / (1.0 - K * rho ** 2)
    return (z ** 2 / vx0[None]).sum(1) - kappa * (z / s[None]).sum(1) ** 2


def _trunc_norm(mean: float, sd: float, lo: float, hi: float, rng: np.random.Generator) -> float:
    a, b = special.ndtr((lo - mean) / sd), special.ndtr((hi - mean) / sd)
    u = rng.uniform(a, b)
    return float(mean + sd * special.ndtri(np.clip(u, 1e-300, 1.0 - 1e-16)))


class _Chain:
    def __init__(self, by, sy, bx, sx, rng: np.random.Generator):
        self.by, self.sy, self.bx, self.sx, self.rng = by, sy, bx, sx, rng
        N, K = bx.shape
        self.N, self.K = N, K
        self.theta = rng.normal(0, 0.5, K)
        self.bx0 = bx.copy()
        self.alpha = np.zeros(N)
        self.rho = rng.uniform(-0.2, 0.2, N)
        self.phi = np.exp(rng.normal(0, 0.3, N))
        self.tau = float(np.exp(rng.normal(np.log(0.1), 0.5)))
        self.vx0 = np.maximum(bx.var(0), 1e-8) * np.exp(rng.normal(0, 0.3, K))
        self.mx = np.zeros(K)
        self.step = dict(phi=0.5, tau=0.5, vx0=np.full(K, 0.3))
        self.acc = dict(phi=0.0, tau=0.0, vx0=np.zeros(K))
        self.n_prop = 0
        rg = (np.arange(GRID) + 0.5) / GRID
        self.rho_grid = 2.0 * rg - 1.0
        self.ok_grid = np.abs(self.rho_grid) < 1.0 / np.sqrt(K)
        self.logbeta = 9.0 * np.log(rg) + 9.0 * np.log(1.0 - rg)

    def _P(self):
        K = self.K
        s = np.sqrt(self.vx0)
        u = 1.0 / s
        c = self.rho ** 2 / (1.0 - K * self.rho ** 2)
        return np.diag(1.0 / self.vx0)[None] + c[:, None, None] * np.outer(u, u)[None]

    def _gamma(self):
        return np.sqrt(self.vx0)[None] * (self.rho / (self.phi * self.tau))[:, None]

    def sweep(self, adapt: bool) -> None:
        rng, N, K = self.rng, self.N, self.K
        by, sy, bx, sx = self.by, self.sy, self.bx, self.sx
        s = np.sqrt(self.vx0)
        P = self._P()
        gam = self._gamma()

        m = self.mx[None] + gam * self.alpha[:, None]
        prec = P.copy()
        prec[:, np.arange(K), np.arange(K)] += 1.0 / sx ** 2
        prec += (self.theta[:, None] * self.theta[None, :])[None] / (sy ** 2)[:, None, None]
        rhs = np.einsum("ijk,ik->ij", P, m) + bx / sx ** 2 + self.theta[None] * ((by - self.alpha) / sy ** 2)[:, None]
        mean = np.linalg.solve(prec, rhs[:, :, None])[:, :, 0]
        Lc = np.linalg.cholesky(prec)
        self.bx0 = mean + np.linalg.solve(Lc.transpose(0, 2, 1), rng.normal(size=(N, K, 1)))[:, :, 0]

        Pg = np.einsum("ijk,ik->ij", P, gam)
        prec_a = 1.0 / sy ** 2 + 1.0 / (self.tau * self.phi) ** 2 + np.einsum("ij,ij->i", gam, Pg)
        rhs_a = (by - self.bx0 @ self.theta) / sy ** 2 + np.einsum("ij,ij->i", Pg, self.bx0 - self.mx[None])
        self.alpha = rhs_a / prec_a + rng.normal(size=N) / np.sqrt(prec_a)

        for k in range(K):
            e = by - self.alpha - self.bx0 @ self.theta + self.bx0[:, k] * self.theta[k]
            pk = float(np.sum(self.bx0[:, k] ** 2 / sy ** 2))
            mk = float(np.sum(self.bx0[:, k] * e / sy ** 2) / pk)
            self.theta[k] = _trunc_norm(mk, 1.0 / np.sqrt(pk), -10.0, 10.0, rng)

        prec_m = np.eye(K) + P.sum(0)
        rhs_m = np.einsum("ijk,ik->j", P, self.bx0 - gam * self.alpha[:, None])
        Lm = np.linalg.cholesky(prec_m)
        self.mx = np.linalg.solve(prec_m, rhs_m) + np.linalg.solve(Lm.T, rng.normal(size=K))

        G = GRID
        coef = (self.alpha / (self.phi * self.tau))[:, None] * self.rho_grid[None, :]
        z = self.bx0[:, None, :] - self.mx[None, None, :] - coef[:, :, None] * s[None, None, :]
        kappa = -self.rho_grid ** 2 / (1.0 - K * self.rho_grid ** 2 + (~self.ok_grid))
        q = (z ** 2 / self.vx0[None, None]).sum(2) - kappa[None] * (z / s[None, None]).sum(2) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = self.logbeta[None] - 0.5 * np.log(1.0 - K * self.rho_grid ** 2)[None] - 0.5 * q
        lp[:, ~self.ok_grid] = -np.inf
        lp -= lp.max(1, keepdims=True)
        pr = np.exp(lp)
        pr /= pr.sum(1, keepdims=True)
        cdf = np.cumsum(pr, 1)
        u = rng.uniform(size=N)
        idx = np.minimum((cdf < u[:, None]).sum(1), G - 1)
        r = np.clip((idx + rng.uniform(-0.5, 0.5, N) + 0.5) / G, 0.5 / G, 1.0 - 0.5 / G)
        rho_new = 2.0 * r - 1.0
        self.rho = np.where(np.abs(rho_new) < 1.0 / np.sqrt(K), rho_new, self.rho_grid[idx])

        def logt_phi(phi):
            qf = _qform(self.bx0, self.mx, s, self.vx0, self.rho, self.alpha, phi, self.tau)
            return -self.alpha ** 2 / (2.0 * self.tau ** 2 * phi ** 2) - 0.5 * qf - np.log1p(phi ** 2)

        prop = self.phi * np.exp(self.step["phi"] * rng.normal(size=N))
        acc = np.log(rng.uniform(size=N)) < logt_phi(prop) - logt_phi(self.phi)
        self.phi = np.where(acc, prop, self.phi)
        self.acc["phi"] += float(acc.mean())

        def logt_tau(tau):
            qf = _qform(self.bx0, self.mx, s, self.vx0, self.rho, self.alpha, self.phi, tau)
            return -N * np.log(tau) - np.sum(self.alpha ** 2 / self.phi ** 2) / (2.0 * tau ** 2) - 0.5 * qf.sum() - np.log1p(tau ** 2) + np.log(tau)

        prop_t = float(self.tau * np.exp(self.step["tau"] * rng.normal()))
        a = bool(np.log(rng.uniform()) < logt_tau(prop_t) - logt_tau(self.tau))
        if a:
            self.tau = prop_t
        self.acc["tau"] += float(a)

        for k in range(K):
            def logt_v(v):
                vv = self.vx0.copy()
                vv[k] = v
                qf = _qform(self.bx0, self.mx, np.sqrt(vv), vv, self.rho, self.alpha, self.phi, self.tau)
                return -0.5 * N * np.log(vv).sum() - 0.5 * qf.sum() - v ** 2 / 2.0 + np.log(v)

            prop_v = float(self.vx0[k] * np.exp(self.step["vx0"][k] * rng.normal()))
            av = bool(np.log(rng.uniform()) < logt_v(prop_v) - logt_v(self.vx0[k]))
            if av:
                self.vx0[k] = prop_v
            self.acc["vx0"][k] += float(av)

        self.n_prop += 1
        if adapt and self.n_prop % 50 == 0:
            for key in ("phi", "tau"):
                self.step[key] *= np.exp(0.5 * (self.acc[key] / 50.0 - (0.3 if key == "phi" else 0.35)))
                self.acc[key] = 0.0
            self.step["vx0"] *= np.exp(0.5 * (self.acc["vx0"] / 50.0 - 0.35))
            self.acc["vx0"][:] = 0.0


def gelman_rubin(chains: np.ndarray) -> float:
    """Potential scale reduction factor (`coda::gelman.diag` point estimate)."""
    x = np.asarray(chains, dtype=float)
    m, n0 = x.shape
    x = x[:, n0 // 2 :]
    m, n = x.shape
    s2 = x.var(1, ddof=1)
    xbar = x.mean(1)
    w = s2.mean()
    b = n * xbar.var(ddof=1)
    muhat = xbar.mean()
    var_w = s2.var(ddof=1) / m
    var_b = 2.0 * b ** 2 / (m - 1)
    cov_wb = m * (np.cov(s2, xbar ** 2)[0, 1] - 2.0 * muhat * np.cov(s2, xbar)[0, 1])
    V = (n - 1) * w / n + (1.0 + 1.0 / m) * b / n
    var_V = ((n - 1) ** 2 * var_w + (1.0 + 1.0 / m) ** 2 * var_b + 2.0 * (n - 1) * (1.0 + 1.0 / m) * cov_wb) / n ** 2
    df_V = 2.0 * V ** 2 / var_V
    df_adj = (df_V + 3.0) / (df_V + 1.0)
    r2 = (n - 1) / n + (1.0 + 1.0 / m) * (1.0 / n) * (b / w)
    return float(np.sqrt(df_adj * r2))


def horse(
    by: np.ndarray,
    sy: np.ndarray,
    bx: np.ndarray,
    sx: np.ndarray,
    n_chains: int = 3,
    n_iter: int = 10000,
    n_burnin: int = 10000,
    seed: int | None = None,
    save: tuple[str, ...] = ("theta",),
    verbose: bool = False,
) -> dict:
    """Run MR-Horse (`K = 1`) or MVMR-Horse (`K > 1`)."""
    by_a, sy_a = np.asarray(by, dtype=float), np.asarray(sy, dtype=float)
    bx_a = np.asarray(bx, dtype=float).reshape(len(by_a), -1)
    sx_a = np.asarray(sx, dtype=float).reshape(bx_a.shape)
    _, K = bx_a.shape
    thin = max(1, n_iter // 1000)
    ss = np.random.SeedSequence(seed)
    names = [f"theta[{k + 1}]" for k in range(K)]
    out: dict[str, list[np.ndarray]] = {nm: [] for nm in names}
    extra: dict[str, list[np.ndarray]] = {key: [] for key in save if key != "theta"}

    for c, child in enumerate(ss.spawn(n_chains)):
        ch = _Chain(by_a, sy_a, bx_a, sx_a, np.random.default_rng(child))
        keep: dict[str, list[float]] = {nm: [] for nm in names}
        ek: dict[str, list[np.ndarray]] = {key: [] for key in extra}
        for it in range(n_burnin + n_iter):
            ch.sweep(adapt=it < n_burnin)
            if it >= n_burnin and (it - n_burnin + 1) % thin == 0:
                for k, nm in enumerate(names):
                    keep[nm].append(float(ch.theta[k]))
                for key in extra:
                    ek[key].append(
                        {"alpha": ch.alpha, "rho": ch.rho, "phi": ch.phi, "tau": np.array(ch.tau), "vx0": ch.vx0.copy(), "mx": ch.mx.copy()}[key].copy()
                    )
        for nm in names:
            out[nm].append(np.array(keep[nm]))
        for key in extra:
            extra[key].append(np.array(ek[key]))
        if verbose:
            print(f"[mrhorse] chain {c + 1}/{n_chains} completed")

    est = []
    for nm in names:
        ch_ = np.array(out[nm])
        allv = ch_.ravel()
        est.append(
            dict(
                Parameter=nm,
                Estimate=float(allv.mean()),
                SD=float(allv.std(ddof=1)),
                q2_5=float(np.quantile(allv, 0.025)),
                q97_5=float(np.quantile(allv, 0.975)),
                Rhat=gelman_rubin(ch_) if n_chains > 1 else float("nan"),
            )
        )
    return dict(estimate=est, samples=out, extra=extra, thin=thin)


def mr_horse(betaY: np.ndarray, betaYse: np.ndarray, betaX: np.ndarray, betaXse: np.ndarray, **kw) -> dict:
    """Univariable Bayesian Horseshoe Mendelian Randomization (`mr_horse`)."""
    return horse(betaY, betaYse, np.asarray(betaX)[:, None], np.asarray(betaXse)[:, None], **kw)


def mvmr_horse(betaY: np.ndarray, betaYse: np.ndarray, betaX: np.ndarray, betaXse: np.ndarray, **kw) -> dict:
    """Multivariable Bayesian Horseshoe Mendelian Randomization (`mvmr_horse`, `betaX` shape `(N, K)`)."""
    return horse(betaY, betaYse, betaX, betaXse, **kw)
