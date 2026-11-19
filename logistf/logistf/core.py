"""
Firth-penalized (Jeffreys-prior) logistic regression (`logistf` equivalent).

Eliminates first-order $O(n^{-1})$ small-sample bias and yields finite parameter
estimates, Penalized Likelihood Ratio Tests (PLRT), and asymmetric Profile Penalized
Likelihood confidence intervals under complete or quasi-complete separation
(Firth 1993; Heinze & Schemper, Statistics in Medicine 2002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

SettingsMode = Literal["default", "R_default"]


def _validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(f"settings must be 'default' or 'R_default', got {settings!r}.")
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class FirthLogisticResult:
    """Fitted Firth-penalized logistic regression summary."""

    coef_names: list[str]
    beta: np.ndarray
    se: np.ndarray
    z_stat: np.ndarray
    p_value: np.ndarray
    odds_ratio: np.ndarray
    ci_low_or: np.ndarray
    ci_high_or: np.ndarray
    penalized_loglik: float
    converged: bool
    n_iter: int
    ci_low_beta: np.ndarray | None = None
    ci_high_beta: np.ndarray | None = None
    ci_method: str = "profile"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": self.beta,
                "se": self.se,
                "z": self.z_stat,
                "p_value": self.p_value,
                "OR": self.odds_ratio,
                "OR_95CI_low": self.ci_low_or,
                "OR_95CI_high": self.ci_high_or,
            },
            index=self.coef_names,
        )


def _firth_eval(
    beta: np.ndarray,
    x: np.ndarray,
    y_bin: np.ndarray,
    ridge_lambda: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Firth penalized log-likelihood, modified score U*(beta), I(beta), I(beta)^{-1}, and augmented I_aug."""
    p = x.shape[1]
    eta = x @ beta
    pi = special.expit(eta)
    w = pi * (1.0 - pi)

    xw = x * w[:, None]
    info = x.T @ xw
    if ridge_lambda > 0.0:
        info = info + ridge_lambda * np.eye(p)

    sign_det, logdet = np.linalg.slogdet(info)
    if sign_det <= 0.0 or not np.isfinite(logdet):
        zeros_m = np.zeros((p, p), dtype=np.float64)
        return float("-inf"), np.zeros(p, dtype=np.float64), info, zeros_m, info

    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(info)

    h_diag = w * np.sum((x @ info_inv) * x, axis=1)
    info_aug = x.T @ (x * (w * (1.0 + h_diag))[:, None])
    if ridge_lambda > 0.0:
        info_aug = info_aug + ridge_lambda * np.eye(p)

    score_star = x.T @ (y_bin - pi + h_diag * (0.5 - pi))
    if ridge_lambda > 0.0:
        score_star = score_star - ridge_lambda * beta

    ll_binom = float(
        np.sum(np.where(y_bin > 0.5, -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)))
    )
    ll_pen = ll_binom + 0.5 * float(logdet)
    if ridge_lambda > 0.0:
        ll_pen -= 0.5 * ridge_lambda * float(np.sum(beta**2))

    return ll_pen, score_star, info, info_inv, info_aug


def _maximize_constrained(
    fixed_idx: int,
    fixed_val: float,
    free_init: np.ndarray,
    x: np.ndarray,
    y_bin: np.ndarray,
    ridge_lambda: float = 0.0,
    max_iter: int = 60,
    max_step_halving: int = 15,
    tol: float = 1e-6,
    use_augmented_info: bool = False,
) -> tuple[float, np.ndarray]:
    """Maximize full-matrix Firth penalized log-likelihood holding `beta[fixed_idx] = fixed_val`."""
    p = x.shape[1]
    if p == 1:
        b_full = np.array([fixed_val], dtype=np.float64)
        ll, _, _, _, _ = _firth_eval(b_full, x, y_bin, ridge_lambda=ridge_lambda)
        return ll, b_full

    free_idx = [k for k in range(p) if k != fixed_idx]

    def _assemble(b_free: np.ndarray) -> np.ndarray:
        b = np.empty(p, dtype=np.float64)
        b[fixed_idx] = fixed_val
        b[free_idx] = b_free
        return b

    b_free = free_init.copy()
    ll, score, info, _, info_aug = _firth_eval(_assemble(b_free), x, y_bin, ridge_lambda=ridge_lambda)
    if not np.isfinite(ll):
        b_free = np.zeros(len(free_idx), dtype=np.float64)
        ll, score, info, _, info_aug = _firth_eval(_assemble(b_free), x, y_bin, ridge_lambda=ridge_lambda)

    converged = False
    for _ in range(max_iter):
        score_sub = score[free_idx]
        if np.isfinite(ll) and float(np.max(np.abs(score_sub))) < tol:
            converged = True
            break
        cur_mat = info_aug if use_augmented_info else info
        info_sub = cur_mat[np.ix_(free_idx, free_idx)]
        step = np.linalg.pinv(info_sub) @ score_sub
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 5.0:
            step = step * (5.0 / step_norm)

        alpha = 1.0
        accepted = False
        for _ in range(max_step_halving):
            cand_free = b_free + alpha * step
            ll_cand, sc_cand, inf_cand, _, inf_aug_cand = _firth_eval(
                _assemble(cand_free), x, y_bin, ridge_lambda=ridge_lambda
            )
            if np.isfinite(ll_cand) and ll_cand >= ll - 1e-12:
                if float(np.max(np.abs(cand_free - b_free))) < tol and float(np.max(np.abs(sc_cand[free_idx]))) < max(
                    tol, 1e-4
                ):
                    b_free, ll, score, info, info_aug = cand_free, ll_cand, sc_cand, inf_cand, inf_aug_cand
                    converged = True
                else:
                    b_free, ll, score, info, info_aug = cand_free, ll_cand, sc_cand, inf_cand, inf_aug_cand
                accepted = True
                break
            alpha *= 0.5
        if converged or not accepted:
            break

    if not converged and not use_augmented_info and np.isfinite(ll):
        def _neg_obj(bf: np.ndarray) -> tuple[float, np.ndarray]:
            val, sc, _, _, _ = _firth_eval(_assemble(bf), x, y_bin, ridge_lambda=ridge_lambda)
            if not np.isfinite(val):
                return 1e100, np.zeros_like(bf)
            return -val, -sc[free_idx]

        opt = optimize.minimize(_neg_obj, b_free, method="BFGS", jac=True, options={"gtol": 1e-6, "maxiter": 60})
        val_opt, _, _, _, _ = _firth_eval(_assemble(opt.x), x, y_bin, ridge_lambda=ridge_lambda)
        if np.isfinite(val_opt) and val_opt > ll:
            b_free = opt.x
            ll = val_opt

    return ll, _assemble(b_free)


def _profile_ci_bound(
    j: int,
    direction: float,
    beta_hat: np.ndarray,
    se_j: float,
    info_hat: np.ndarray,
    ll_target: float,
    x: np.ndarray,
    y_bin: np.ndarray,
    ridge_lambda: float,
    alpha: float,
    use_augmented_info: bool = False,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> float:
    """Find one-sided profile penalized likelihood CI root `tilde_ell*(b) == ll_target` along `direction` (-1 or +1)."""
    p = x.shape[1]
    free_idx = [k for k in range(p) if k != j]
    if free_idx:
        tangent = -np.linalg.pinv(info_hat[np.ix_(free_idx, free_idx)]) @ info_hat[free_idx, j]
    else:
        tangent = np.array([], dtype=np.float64)

    z_crit = float(stats.norm.isf(alpha / 2.0))
    step0 = max(0.5 * float(se_j), 0.25) * z_crit

    b_in = float(beta_hat[j])
    warm_in = beta_hat[free_idx].copy() if free_idx else np.array([], dtype=np.float64)

    delta = step0
    b_out = b_in + direction * delta
    found_bracket = False

    for _ in range(16):
        b_try = float(beta_hat[j]) + direction * delta
        warm_guess = warm_in + tangent * (b_try - b_in) if free_idx else warm_in
        ll_try, full_try = _maximize_constrained(
            j,
            b_try,
            warm_guess,
            x,
            y_bin,
            ridge_lambda=ridge_lambda,
            max_iter=max_iter,
            tol=tol,
            use_augmented_info=use_augmented_info,
        )
        if not np.isfinite(ll_try) or ll_try <= ll_target:
            b_out = b_try
            found_bracket = True
            break
        b_in = b_try
        if free_idx:
            warm_in = full_try[free_idx].copy()
        delta *= 2.0
        if delta > 60.0:
            break

    if not found_bracket:
        return float(beta_hat[j] + direction * 50.0)

    warm_cache = [warm_in.copy()]

    def _root_fn(b_val: float) -> float:
        guess = warm_in + tangent * (b_val - b_in) if free_idx else warm_cache[0]
        ll_val, full_val = _maximize_constrained(
            j,
            float(b_val),
            guess,
            x,
            y_bin,
            ridge_lambda=ridge_lambda,
            max_iter=max_iter,
            tol=tol,
            use_augmented_info=use_augmented_info,
        )
        if np.isfinite(ll_val) and free_idx:
            warm_cache[0] = full_val[free_idx].copy()
        if not np.isfinite(ll_val):
            return -1e6
        return float(ll_val - ll_target)

    lo_b, hi_b = (b_out, b_in) if direction < 0.0 else (b_in, b_out)
    try:
        return float(optimize.brentq(_root_fn, lo_b, hi_b, xtol=tol, maxiter=60))
    except ValueError:
        return float(b_out)


def fit_firth_logistic(
    X: np.ndarray | pd.DataFrame,
    y: Sequence[int | float | bool],
    add_intercept: bool = True,
    coef_names: Sequence[str] | None = None,
    ridge_lambda: float = 0.0,
    ci_method: str = "profile",
    pl: bool = True,
    alpha: float = 0.05,
    max_iter: int | None = None,
    max_step_halving: int = 15,
    tol: float | None = None,
    *,
    settings: SettingsMode = "default",
) -> FirthLogisticResult:
    """Fit Firth-penalized (Jeffreys-prior) logistic regression via modified Fisher scoring with step-halving.

    Parameters
    ----------
    X : (n_samples, n_features) or 1D array / DataFrame
        Design matrix. When `add_intercept=True` (default), prepends `(Intercept)` unless `X`
        already contains a non-zero constant column.
    y : binary {0, 1} sequence
        Binary response vector.
    ridge_lambda : float, default 0.0
        Optional additional L2 penalty (0.0 = exact unperturbed Firth logistic regression).
    ci_method : {'profile', 'wald'}, default 'profile'
        `'profile'` computes exact Heinze & Schemper (2002) asymmetric profile penalized likelihood CIs;
        `'wald'` computes symmetric Wald CIs from `I(beta)^{-1}`.
    pl : bool, default True
        When `True`, computes `p_value` via the 1-df Penalized Likelihood Ratio Test (PLRT);
        when `False`, uses two-sided Wald normal p-values.
    settings : {'default', 'R_default'}, default 'default'
        When `'R_default'`, matches `georgheinze/logistf` (`src/logistf.c`): uses the augmented
        Newton-Fisher information matrix `X^T diag(w_i (1 + h_i)) X` during scoring updates with
        `max_iter=25, tol=1e-5`.
    """
    _validate_settings(settings)
    if ci_method not in ("profile", "wald"):
        raise ValueError(f"ci_method must be 'profile' or 'wald', got {ci_method!r}.")

    use_aug = settings == "R_default"
    max_iter_eff = (25 if use_aug else 80) if max_iter is None else int(max_iter)
    tol_eff = (1e-5 if use_aug else 1e-7) if tol is None else float(tol)

    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns] if coef_names is None else [str(c) for c in coef_names]
        x = X.to_numpy(dtype=np.float64)
    else:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        names = [f"x{i}" for i in range(x.shape[1])] if coef_names is None else [str(c) for c in coef_names]

    if add_intercept:
        has_const_col = (
            x.shape[0] > 1
            and x.shape[1] > 0
            and bool(np.any(np.all(np.isclose(x, x[0:1, :], atol=1e-12), axis=0) & (np.abs(x[0, :]) > 1e-12)))
        )
        if not has_const_col:
            x = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
            names = ["(Intercept)", *names]

    y_bin = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y_bin.size:
        raise ValueError(f"X has {x.shape[0]} rows, but y has length {y_bin.size}.")
    if not np.all(np.isin(y_bin, [0.0, 1.0])):
        raise ValueError("y must contain binary {0, 1} values.")

    _, p = x.shape
    beta = np.zeros(p, dtype=np.float64)
    ll, score_star, info, info_inv, info_aug = _firth_eval(beta, x, y_bin, ridge_lambda=ridge_lambda)
    if not np.isfinite(ll):
        raise ValueError("Design matrix X is rank-deficient at initialization (det(X^T W X) == 0).")

    converged = False
    n_iter = 0

    for it in range(1, max_iter_eff + 1):
        n_iter = it
        if float(np.max(np.abs(score_star))) < tol_eff:
            converged = True
            break

        if use_aug:
            try:
                step = np.linalg.solve(info_aug, score_star)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(info_aug) @ score_star
        else:
            step = info_inv @ score_star

        step_norm = float(np.max(np.abs(step)))
        if step_norm > 5.0:
            step = step * (5.0 / step_norm)

        step_scale = 1.0
        accepted = False
        for _ in range(max_step_halving):
            beta_cand = beta + step_scale * step
            ll_cand, sc_cand, inf_cand, inf_inv_cand, inf_aug_cand = _firth_eval(
                beta_cand, x, y_bin, ridge_lambda=ridge_lambda
            )
            if np.isfinite(ll_cand) and ll_cand >= ll - 1e-12:
                delta_b = float(np.max(np.abs(beta_cand - beta)))
                beta, ll, score_star, info, info_inv, info_aug = (
                    beta_cand,
                    ll_cand,
                    sc_cand,
                    inf_cand,
                    inf_inv_cand,
                    inf_aug_cand,
                )
                accepted = True
                if delta_b < tol_eff and float(np.max(np.abs(score_star))) < max(tol_eff, 1e-5):
                    converged = True
                break
            step_scale *= 0.5

        if converged or not accepted:
            break

    if not converged and not use_aug:
        def _neg_ll_and_grad(b: np.ndarray) -> tuple[float, np.ndarray]:
            val, sc, _, _, _ = _firth_eval(b, x, y_bin, ridge_lambda=ridge_lambda)
            if not np.isfinite(val):
                return 1e100, np.zeros_like(b)
            return -val, -sc

        opt = optimize.minimize(
            _neg_ll_and_grad, beta, method="BFGS", jac=True, options={"gtol": tol_eff, "maxiter": max_iter_eff}
        )
        ll_opt, sc_opt, inf_opt, inf_inv_opt, _ = _firth_eval(opt.x, x, y_bin, ridge_lambda=ridge_lambda)
        if np.isfinite(ll_opt) and ll_opt >= ll - 1e-12:
            beta, ll, score_star, info, info_inv = opt.x, ll_opt, sc_opt, inf_opt, inf_inv_opt
            n_iter += int(opt.nit)
            if float(np.max(np.abs(score_star))) < 1e-4:
                converged = True

    se = np.sqrt(np.maximum(np.diag(info_inv), 1e-12))
    z_stat = beta / se
    z_crit = float(stats.norm.isf(alpha / 2.0))

    sub_max_iter = 25 if use_aug else 60
    sub_tol = 1e-5 if use_aug else 1e-6

    if pl:
        p_val = np.empty(p, dtype=np.float64)
        for j in range(p):
            free_idx = [k for k in range(p) if k != j]
            if free_idx:
                tangent = -np.linalg.pinv(info[np.ix_(free_idx, free_idx)]) @ info[free_idx, j]
                warm_null = beta[free_idx] + tangent * (0.0 - beta[j])
            else:
                warm_null = np.array([], dtype=np.float64)
            ll_null, _ = _maximize_constrained(
                j,
                0.0,
                warm_null,
                x,
                y_bin,
                ridge_lambda=ridge_lambda,
                max_iter=sub_max_iter,
                tol=sub_tol,
                use_augmented_info=use_aug,
            )
            dev = max(0.0, 2.0 * (ll - ll_null)) if np.isfinite(ll_null) else np.inf
            p_val[j] = float(stats.chi2.sf(dev, df=1))
    else:
        p_val = 2.0 * stats.norm.sf(np.abs(z_stat))

    if ci_method == "profile":
        crit = 0.5 * float(stats.chi2.ppf(1.0 - alpha, df=1))
        ll_target = ll - crit
        ci_low_b = np.empty(p, dtype=np.float64)
        ci_high_b = np.empty(p, dtype=np.float64)
        for j in range(p):
            ci_low_b[j] = _profile_ci_bound(
                j,
                -1.0,
                beta,
                float(se[j]),
                info,
                ll_target,
                x,
                y_bin,
                ridge_lambda,
                alpha,
                use_augmented_info=use_aug,
                max_iter=sub_max_iter,
                tol=sub_tol,
            )
            ci_high_b[j] = _profile_ci_bound(
                j,
                +1.0,
                beta,
                float(se[j]),
                info,
                ll_target,
                x,
                y_bin,
                ridge_lambda,
                alpha,
                use_augmented_info=use_aug,
                max_iter=sub_max_iter,
                tol=sub_tol,
            )
    else:
        ci_low_b = beta - z_crit * se
        ci_high_b = beta + z_crit * se

    odds_ratio = np.exp(np.clip(beta, -50.0, 50.0))
    ci_low_or = np.exp(np.clip(ci_low_b, -50.0, 50.0))
    ci_high_or = np.exp(np.clip(ci_high_b, -50.0, 50.0))

    return FirthLogisticResult(
        coef_names=names,
        beta=beta,
        se=se,
        z_stat=z_stat,
        p_value=p_val,
        odds_ratio=odds_ratio,
        ci_low_or=ci_low_or,
        ci_high_or=ci_high_or,
        penalized_loglik=ll,
        converged=converged,
        n_iter=n_iter,
        ci_low_beta=ci_low_b,
        ci_high_beta=ci_high_b,
        ci_method=ci_method,
    )
