"""Python implementation of MR-Horse and MVMR-Horse (Grant & Burgess, AJHG 2024).

Samples the exact Bayesian Horseshoe Mendelian Randomization posterior for univariable (`K = 1`)
and multivariable (`K >= 1`) summary data accounting for both correlated and uncorrelated
horizontal pleiotropy without requiring external JAGS installation.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import special, stats

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")

GRID = 200
_TP_FLOOR = 1e-60


def _qform(bx0, mx, s, vx0, rho, alpha, phi, tau):
    tp = np.maximum(phi * tau, _TP_FLOOR)
    a_vec = alpha / tp
    z = bx0 - mx[None, :] - a_vec[:, None] * rho * s[None, :]
    ztilde = z / s[None, :]
    one_minus_r2 = np.maximum(1.0 - np.sum(rho ** 2, axis=1), 1e-8)
    dot_rz = np.sum(rho * ztilde, axis=1)
    return np.sum(z ** 2 / vx0[None, :], axis=1) + (dot_rz ** 2) / one_minus_r2


def _sample_right_tail_std_norm(alpha: float, beta: float, rng: np.random.Generator) -> float:
    """Exact Robert (1995) shifted-exponential rejection sampler on [alpha, beta] for alpha > 0."""
    width = beta - alpha
    if width <= 0.0:
        return alpha
    lam = 0.5 * (alpha + np.sqrt(alpha * alpha + 4.0))
    u_max = -np.expm1(-lam * width)
    if u_max <= 0.0:
        return alpha
    for _ in range(1000):
        u = rng.uniform(0.0, u_max)
        z = alpha - np.log1p(-u) / lam
        if np.log(rng.uniform()) <= -0.5 * (z - lam) ** 2:
            return float(np.clip(z, alpha, beta))
    return float(np.clip(alpha + 1.0 / max(lam, 1.0), alpha, beta))


def _trunc_norm(mean: float, sd: float, lo: float, hi: float, rng: np.random.Generator) -> float:
    """Sample from N(mean, sd^2) truncated to [lo, hi] without extreme-tail boundary violations."""
    if lo >= hi:
        raise ValueError(f"Invalid truncation bounds lo={lo} >= hi={hi}.")
    if not (np.isfinite(sd) and sd > 0.0):
        return float(np.clip(mean, lo, hi))
    alpha = (lo - mean) / sd
    beta = (hi - mean) / sd
    if alpha > 5.0:
        z = _sample_right_tail_std_norm(alpha, beta, rng)
        return float(np.clip(mean + sd * z, lo, hi))
    if beta < -5.0:
        z = -_sample_right_tail_std_norm(-beta, -alpha, rng)
        return float(np.clip(mean + sd * z, lo, hi))
    a, b = float(special.ndtr(alpha)), float(special.ndtr(beta))
    if b - a > 1e-12:
        u = rng.uniform(a, b)
        z = float(special.ndtri(np.clip(u, 1e-300, 1.0 - 1e-16)))
        return float(np.clip(mean + sd * z, lo, hi))
    val = float(stats.truncnorm.rvs(alpha, beta, loc=mean, scale=sd, random_state=rng))
    return float(np.clip(val, lo, hi))


class _Chain:
    def __init__(self, by, sy, bx, sx, rng: np.random.Generator, shared_rho: bool = False):
        self.by, self.sy, self.bx, self.sx, self.rng = by, sy, bx, sx, rng
        self.shared_rho = shared_rho
        N, K = bx.shape
        self.N, self.K = N, K
        self.theta = rng.normal(0, 0.5, K)
        self.bx0 = bx.copy()
        self.alpha = np.zeros(N)
        init_r = 0.2 / np.sqrt(K)
        if shared_rho:
            init_r = min(init_r, 0.5 * max(1.0 / np.sqrt(K) - 1e-6, 1e-8))
            rho0 = rng.uniform(-init_r, init_r, N)
            self.rho = np.repeat(rho0[:, None], K, axis=1)
        else:
            self.rho = rng.uniform(-init_r, init_r, (N, K))
        self.phi = np.maximum(np.exp(rng.normal(0, 0.3, N)), _TP_FLOOR)
        self.tau = float(max(np.exp(rng.normal(np.log(0.1), 0.5)), _TP_FLOOR))
        self.vx0 = np.maximum(bx.var(0), 1e-8) * np.exp(rng.normal(0, 0.3, K))
        self.mx = np.zeros(K)
        self.step = dict(phi=0.5, tau=0.5, vx0=np.full(K, 0.3))
        self.acc = dict(phi=0.0, tau=0.0, vx0=np.zeros(K))
        self.n_prop = 0
        rg = (np.arange(GRID) + 0.5) / GRID
        self.rho_grid = 2.0 * rg - 1.0

    def _P(self):
        s = np.sqrt(self.vx0)
        u = self.rho / s[None, :]
        inv_one_minus_r2 = 1.0 / np.maximum(1.0 - np.sum(self.rho ** 2, axis=1), 1e-8)
        return np.diag(1.0 / self.vx0)[None, :, :] + inv_one_minus_r2[:, None, None] * (u[:, :, None] * u[:, None, :])

    def _gamma(self):
        tp = np.maximum(self.phi * self.tau, _TP_FLOOR)
        return np.sqrt(self.vx0)[None, :] * self.rho / tp[:, None]

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

        tp = np.maximum(self.tau * self.phi, _TP_FLOOR)
        Pg = np.einsum("ijk,ik->ij", P, gam)
        prec_a = 1.0 / sy ** 2 + 1.0 / (tp ** 2) + np.einsum("ij,ij->i", gam, Pg)
        rhs_a = (by - self.bx0 @ self.theta) / sy ** 2 + np.einsum("ij,ij->i", Pg, self.bx0 - self.mx[None])
        self.alpha = rhs_a / prec_a + rng.normal(size=N) / np.sqrt(prec_a)

        for k in range(K):
            e = by - self.alpha - self.bx0 @ self.theta + self.bx0[:, k] * self.theta[k]
            pk = float(np.sum(self.bx0[:, k] ** 2 / sy ** 2))
            if pk > 0.0:
                mk = float(np.sum(self.bx0[:, k] * e / sy ** 2) / pk)
                self.theta[k] = _trunc_norm(mk, 1.0 / np.sqrt(pk), -10.0, 10.0, rng)

        prec_m = np.eye(K) + P.sum(0)
        rhs_m = np.einsum("ijk,ik->j", P, self.bx0 - gam * self.alpha[:, None])
        Lm = np.linalg.cholesky(prec_m)
        self.mx = np.linalg.solve(prec_m, rhs_m) + np.linalg.solve(Lm.T, rng.normal(size=K))

        # Gibbs updates for rho: shared scalar per SNP in R_default, per-exposure vector in default
        G = GRID
        a_vec = self.alpha / tp
        d_tilde = (self.bx0 - self.mx[None, :]) / s[None, :]
        if self.shared_rho:
            r_max = 1.0 / np.sqrt(K) - 1e-6
            g = r_max * self.rho_grid
            one_minus_r2 = np.maximum(1.0 - K * (g ** 2), 1e-12)
            s_d = np.sum(d_tilde, axis=1)
            lp = (
                9.0 * np.log(np.maximum(1.0 - g[None, :] ** 2, 1e-30))
                - 0.5 * np.log(one_minus_r2[None, :])
                - 0.5 * ((a_vec[:, None] - s_d[:, None] * g[None, :]) ** 2) / one_minus_r2[None, :]
            )
            lp -= lp.max(axis=1, keepdims=True)
            pr = np.exp(lp)
            pr /= pr.sum(axis=1, keepdims=True)
            cdf = np.cumsum(pr, axis=1)
            u = rng.uniform(size=N)
            idx = np.minimum((cdf < u[:, None]).sum(axis=1), G - 1)
            u_jit = np.clip((idx + rng.uniform(-0.5, 0.5, N) + 0.5) / G, 0.5 / G, 1.0 - 0.5 / G)
            rho_s = r_max * (2.0 * u_jit - 1.0)
            self.rho[:, :] = rho_s[:, None]
        else:
            for k in range(K):
                s_other = np.sum(self.rho ** 2, axis=1) - self.rho[:, k] ** 2
                r_max_k = np.sqrt(np.maximum(1.0 - 1e-6 - s_other, 1e-8))
                g_mat = r_max_k[:, None] * self.rho_grid[None, :]

                z_cur = d_tilde - a_vec[:, None] * self.rho
                norm_z_other = np.sum(z_cur ** 2, axis=1) - z_cur[:, k] ** 2
                rz_other = np.sum(self.rho * z_cur, axis=1) - self.rho[:, k] * z_cur[:, k]

                zk_grid = d_tilde[:, k : k + 1] - a_vec[:, None] * g_mat
                norm_z2 = norm_z_other[:, None] + zk_grid ** 2
                dot_rz = rz_other[:, None] + g_mat * zk_grid
                one_minus_r2 = np.maximum((1.0 - s_other)[:, None] - g_mat ** 2, 1e-12)
                q = norm_z2 + (dot_rz ** 2) / one_minus_r2

                lp = 9.0 * np.log(np.maximum(1.0 - g_mat ** 2, 1e-30)) - 0.5 * np.log(one_minus_r2) - 0.5 * q
                lp -= lp.max(axis=1, keepdims=True)
                pr = np.exp(lp)
                pr /= pr.sum(axis=1, keepdims=True)
                cdf = np.cumsum(pr, axis=1)
                u = rng.uniform(size=N)
                idx = np.minimum((cdf < u[:, None]).sum(axis=1), G - 1)
                u_jit = np.clip((idx + rng.uniform(-0.5, 0.5, N) + 0.5) / G, 0.5 / G, 1.0 - 0.5 / G)
                self.rho[:, k] = r_max_k * (2.0 * u_jit - 1.0)

        def logt_phi(phi):
            tp_p = np.maximum(self.tau * phi, _TP_FLOOR)
            qf = _qform(self.bx0, self.mx, s, self.vx0, self.rho, self.alpha, phi, self.tau)
            return -self.alpha ** 2 / (2.0 * tp_p ** 2) - 0.5 * qf - np.log1p(np.minimum(phi, 1e150) ** 2)

        prop = np.maximum(self.phi * np.exp(self.step["phi"] * rng.normal(size=N)), _TP_FLOOR)
        acc = np.log(rng.uniform(size=N)) < logt_phi(prop) - logt_phi(self.phi)
        self.phi = np.where(acc, prop, self.phi)
        self.acc["phi"] += float(acc.mean())

        def logt_tau(tau):
            tp_t = np.maximum(tau * self.phi, _TP_FLOOR)
            qf = _qform(self.bx0, self.mx, s, self.vx0, self.rho, self.alpha, self.phi, tau)
            tau_safe = max(tau, _TP_FLOOR)
            return (
                -N * np.log(tau_safe)
                - np.sum(self.alpha ** 2 / (2.0 * tp_t ** 2))
                - 0.5 * qf.sum()
                - np.log1p(min(tau_safe, 1e150) ** 2)
                + np.log(tau_safe)
            )

        prop_t = float(max(self.tau * np.exp(self.step["tau"] * rng.normal()), _TP_FLOOR))
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

            prop_v = float(max(self.vx0[k] * np.exp(self.step["vx0"][k] * rng.normal()), 1e-30))
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


def gelman_rubin(chains: np.ndarray, split_chains: bool = False) -> float:
    """Potential scale reduction factor (`coda::gelman.diag` point estimate on post-burnin chains).

    Uses all supplied post-burnin draws without discarding half the samples. When `split_chains=True`,
    each chain is split into two equal halves (split-Rhat) while still retaining ~100% of draws.
    """
    x = np.asarray(chains, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if split_chains:
        half = x.shape[1] // 2
        if half >= 2:
            x = np.vstack([x[:, :half], x[:, half : 2 * half]])
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    s2 = x.var(1, ddof=1)
    xbar = x.mean(1)
    w = float(s2.mean())
    b = float(n * xbar.var(ddof=1))
    if w <= 1e-30:
        return 1.0 if b <= 1e-30 else float("inf")
    muhat = float(xbar.mean())
    var_w = float(s2.var(ddof=1)) / m
    var_b = 2.0 * b ** 2 / (m - 1)
    cov_wb = (n / m) * float(np.cov(s2, xbar ** 2)[0, 1] - 2.0 * muhat * np.cov(s2, xbar)[0, 1])
    V = (n - 1) * w / n + (1.0 + 1.0 / m) * b / n
    var_V = ((n - 1) ** 2 * var_w + (1.0 + 1.0 / m) ** 2 * var_b + 2.0 * (n - 1) * (1.0 + 1.0 / m) * cov_wb) / n ** 2
    if var_V <= 0.0 or not np.isfinite(var_V):
        df_adj = 1.0
    else:
        df_V = 2.0 * V ** 2 / var_V
        df_adj = (df_V + 3.0) / (df_V + 1.0)
    r2 = (n - 1) / n + (1.0 + 1.0 / m) * (1.0 / n) * (b / w)
    return float(np.sqrt(max(df_adj * r2, 0.0)))


def horse(
    by: np.ndarray,
    sy: np.ndarray,
    bx: np.ndarray,
    sx: np.ndarray,
    n_chains: int = 3,
    n_iter: int = 10000,
    n_burnin: int = 10000,
    n_thin: int | None = None,
    seed: int | None = None,
    save: tuple[str, ...] = ("theta",),
    verbose: bool = False,
    settings: SettingsMode = "default",
) -> dict:
    """Run MR-Horse (`K = 1`) or MVMR-Horse (`K > 1`)."""
    _validate_settings(settings)
    by_a, sy_a = np.asarray(by, dtype=float), np.asarray(sy, dtype=float)
    bx_a = np.asarray(bx, dtype=float).reshape(len(by_a), -1)
    sx_a = np.asarray(sx, dtype=float).reshape(bx_a.shape)
    _, K = bx_a.shape
    if n_thin is None:
        thin = 1 if settings == "R_default" else max(1, n_iter // 1000)
    else:
        if n_thin < 1:
            raise ValueError("n_thin must be >= 1.")
        thin = int(n_thin)
    shared_rho = settings == "R_default"
    ss = np.random.SeedSequence(seed)
    names = [f"theta[{k + 1}]" for k in range(K)]
    out: dict[str, list[np.ndarray]] = {nm: [] for nm in names}
    valid_save_keys = {"theta", "alpha", "rho", "phi", "tau", "vx0", "mx", "bx0"}
    unknown = set(save) - valid_save_keys
    if unknown:
        raise ValueError(f"Unsupported save key(s): {sorted(unknown)}; valid keys are {sorted(valid_save_keys)}.")
    extra: dict[str, list[np.ndarray]] = {key: [] for key in save if key != "theta"}

    for c, child in enumerate(ss.spawn(n_chains)):
        ch = _Chain(by_a, sy_a, bx_a, sx_a, np.random.default_rng(child), shared_rho=shared_rho)
        keep: dict[str, list[float]] = {nm: [] for nm in names}
        ek: dict[str, list[np.ndarray]] = {key: [] for key in extra}
        for it in range(n_burnin + n_iter):
            ch.sweep(adapt=it < n_burnin)
            if it >= n_burnin and (it - n_burnin + 1) % thin == 0:
                for k, nm in enumerate(names):
                    keep[nm].append(float(ch.theta[k]))
                if extra:
                    state_map = {
                        "alpha": ch.alpha,
                        "rho": ch.rho,
                        "phi": ch.phi,
                        "tau": np.array(ch.tau),
                        "vx0": ch.vx0,
                        "mx": ch.mx,
                        "bx0": ch.bx0,
                    }
                    for key in extra:
                        ek[key].append(state_map[key].copy())
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


def mr_horse(
    betaY: np.ndarray,
    betaYse: np.ndarray,
    betaX: np.ndarray,
    betaXse: np.ndarray,
    n_chains: int = 3,
    n_iter: int = 10000,
    n_burnin: int = 10000,
    n_thin: int | None = None,
    seed: int | None = None,
    save: tuple[str, ...] = ("theta",),
    verbose: bool = False,
    settings: SettingsMode = "default",
) -> dict:
    """Univariable Bayesian Horseshoe Mendelian Randomization (`mr_horse`)."""
    _validate_settings(settings)
    return horse(
        betaY,
        betaYse,
        np.asarray(betaX)[:, None],
        np.asarray(betaXse)[:, None],
        n_chains=n_chains,
        n_iter=n_iter,
        n_burnin=n_burnin,
        n_thin=n_thin,
        seed=seed,
        save=save,
        verbose=verbose,
        settings=settings,
    )


def mvmr_horse(
    betaY: np.ndarray,
    betaYse: np.ndarray,
    betaX: np.ndarray,
    betaXse: np.ndarray,
    n_chains: int = 3,
    n_iter: int = 10000,
    n_burnin: int = 10000,
    n_thin: int | None = None,
    seed: int | None = None,
    save: tuple[str, ...] = ("theta",),
    verbose: bool = False,
    settings: SettingsMode = "default",
) -> dict:
    """Multivariable Bayesian Horseshoe Mendelian Randomization (`mvmr_horse`, `betaX` shape `(N, K)`)."""
    _validate_settings(settings)
    return horse(
        betaY,
        betaYse,
        betaX,
        betaXse,
        n_chains=n_chains,
        n_iter=n_iter,
        n_burnin=n_burnin,
        n_thin=n_thin,
        seed=seed,
        save=save,
        verbose=verbose,
        settings=settings,
    )
