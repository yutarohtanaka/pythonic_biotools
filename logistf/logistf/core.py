"""
Firth-penalized (Jeffreys-prior) logistic regression (`logistf` equivalent).

Eliminates first-order $O(n^{-1})$ small-sample bias and yields finite parameter
estimates under complete or quasi-complete separation (Firth 1993; Heinze & Schemper 2002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import special, stats


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


def fit_firth_logistic(
    X: np.ndarray | pd.DataFrame,
    y: Sequence[int | float | bool],
    add_intercept: bool = False,
    coef_names: Sequence[str] | None = None,
    ridge_lambda: float = 1e-6,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> FirthLogisticResult:
    """Fit Firth-penalized (Jeffreys-prior) logistic regression via modified Fisher scoring."""
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns] if coef_names is None else [str(c) for c in coef_names]
        x = X.to_numpy(dtype=np.float64)
    else:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        names = [f"x{i}" for i in range(x.shape[1])] if coef_names is None else [str(c) for c in coef_names]

    if add_intercept:
        x = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
        names = ["(Intercept)", *names]

    y_bin = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y_bin.size:
        raise ValueError(f"X has {x.shape[0]} rows, but y has length {y_bin.size}.")
    if not np.all(np.isin(y_bin, [0.0, 1.0])):
        raise ValueError("y must contain binary {0, 1} values.")

    _, p = x.shape
    beta = np.zeros(p, dtype=np.float64)
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        eta = np.clip(x @ beta, -30.0, 30.0)
        pi = special.expit(eta)
        w = np.maximum(pi * (1.0 - pi), 1e-8)

        xw = x * w[:, None]
        info = x.T @ xw + ridge_lambda * np.eye(p)
        info_inv = np.linalg.pinv(info)

        h_diag = np.clip(w * np.sum((x @ info_inv) * x, axis=1), 0.0, 0.999)
        score_star = x.T @ (y_bin - pi + h_diag * (0.5 - pi))
        step = info_inv @ score_star
        step_norm = float(np.max(np.abs(step)))
        if step_norm > 2.5:
            step = step * (2.5 / step_norm)

        beta_new = beta + step
        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    eta = np.clip(x @ beta, -30.0, 30.0)
    pi = special.expit(eta)
    w = np.maximum(pi * (1.0 - pi), 1e-8)
    info = x.T @ (x * w[:, None]) + ridge_lambda * np.eye(p)
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    sign_det, logdet = np.linalg.slogdet(info)
    firth_pen = 0.5 * float(logdet) if sign_det > 0 else 0.0
    loglik = float(np.sum(y_bin * eta - np.logaddexp(0.0, eta)) + firth_pen)

    z_stat = beta / se
    p_val = 2.0 * stats.norm.sf(np.abs(z_stat))
    odds_ratio = np.exp(np.clip(beta, -50.0, 50.0))
    ci_low = np.exp(np.clip(beta - 1.959964 * se, -50.0, 50.0))
    ci_high = np.exp(np.clip(beta + 1.959964 * se, -50.0, 50.0))

    return FirthLogisticResult(
        coef_names=names,
        beta=beta,
        se=se,
        z_stat=z_stat,
        p_value=p_val,
        odds_ratio=odds_ratio,
        ci_low_or=ci_low,
        ci_high_or=ci_high,
        penalized_loglik=loglik,
        converged=converged,
        n_iter=n_iter,
    )
