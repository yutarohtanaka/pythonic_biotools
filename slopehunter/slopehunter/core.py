"""Python implementation of SlopeHunter (Mahmoud et al., Nature Communications 2022).

Fits a two-component zero-mean bivariate Gaussian mixture model by EM (`shclust`, `hunt`) to
separate pure index-trait instruments ("Hunted" cluster) from pleiotropic instruments affecting
both traits directly ("Pleiotropic" cluster), and computes bias-adjusted summary statistics (`sh_adj`).
"""
from __future__ import annotations

from typing import Literal
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def _validate_settings(settings: str) -> None:
    if settings not in ("default", "R_default"):
        raise ValueError(f"Unknown settings {settings!r}; expected 'default' or 'R_default'.")

_TINY = 1e-300
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _p_from_beta_se(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    return stats.chi2.sf((np.asarray(beta, dtype=float) / np.asarray(se, dtype=float)) ** 2, 1)


def _log_dmvnorm0(x: np.ndarray, y: np.ndarray, sx2: float, sy2: float, sxy: float) -> np.ndarray:
    """Log-density of a zero-mean bivariate normal distribution."""
    det = max(sx2 * sy2 - sxy ** 2, _TINY)
    quad = (sy2 * x ** 2 - 2.0 * sxy * x * y + sx2 * y ** 2) / det
    return -np.log(2.0 * np.pi) - 0.5 * np.log(det) - 0.5 * quad


def _dmvnorm0(x: np.ndarray, y: np.ndarray, sx2: float, sy2: float, sxy: float) -> np.ndarray:
    """Density of a zero-mean bivariate normal distribution."""
    return np.exp(_log_dmvnorm0(x, y, sx2, sy2, sxy))


def shclust(
    xbeta: np.ndarray,
    ybeta: np.ndarray,
    pi0: float = 0.6,
    sxy1: float = 1e-5,
    rng: np.random.Generator | None = None,
    max_iter: int = 50000,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Two-component bivariate Gaussian mixture EM clustering (`SlopeHunter::shclust`)."""
    _validate_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    xbeta, ybeta = np.asarray(xbeta, dtype=float), np.asarray(ybeta, dtype=float)
    n = len(xbeta)
    sx0 = sx1 = max(float(np.std(xbeta, ddof=1)), 1e-15)
    sy0 = sy1 = max(float(np.std(ybeta, ddof=1)), 1e-15)
    if sx0 <= 1e-15 or sy0 <= 1e-15:
        raise ValueError("Associations with at least one of x or y are constant.")
    cov_xy = float(np.cov(xbeta, ybeta)[0, 1])
    dir0 = float(np.sign(cov_xy)) if cov_xy != 0.0 else float(rng.choice([1.0, -1.0]))
    if abs(sxy1) > 0.75 * sx1 * sy1:
        sxy1 = float(np.sign(sxy1) * 0.75 * sx1 * sy1)

    loglkl = -np.inf
    loglkl_ck: float | None = None
    it = 0
    pt = np.full(n, 0.5)
    for it in range(1, max_iter + 1):
        sxy0 = sx0 * sy0 * dir0 * 0.95
        lf0 = _log_dmvnorm0(xbeta, ybeta, sx0 ** 2, sy0 ** 2, sxy0)
        lf1 = _log_dmvnorm0(xbeta, ybeta, sx1 ** 2, sy1 ** 2, sxy1)
        la = np.log(pi0) + lf0
        lb = np.log(1.0 - pi0) + lf1
        l_mix = np.logaddexp(la, lb)
        loglkl = float(np.sum(l_mix))
        pt = np.clip(np.exp(la - l_mix), 1e-7, 1.0 - 1e-7)

        sum_pt = float(pt.sum())
        sum_po = float(n - sum_pt)
        pi0 = float(np.clip(pt.mean(), 0.0001, 0.9999))
        sx0 = max(float(np.sqrt(np.sum(pt * xbeta ** 2) / sum_pt)), 1e-15)
        sy0 = max(float(np.sqrt(np.sum(pt * ybeta ** 2) / sum_pt)), 1e-15)
        dir0 = float(np.sign(np.sum(pt * xbeta * ybeta) / sum_pt))
        if dir0 == 0.0:
            dir0 = float(rng.choice([1.0, -1.0]))
        sx1 = max(float(np.sqrt(np.sum((1.0 - pt) * xbeta ** 2) / sum_po)), 1e-15)
        sy1 = max(float(np.sqrt(np.sum((1.0 - pt) * ybeta ** 2) / sum_po)), 1e-15)
        sxy1 = float(np.sum((1.0 - pt) * xbeta * ybeta) / sum_po)
        if abs(sxy1) > 0.75 * sx1 * sy1:
            sxy1 = float(np.sign(sxy1) * 0.75 * sx1 * sy1)

        if it % 10 == 0:
            if loglkl_ck is not None and abs(loglkl - loglkl_ck) / (abs(loglkl) + 1e-12) < 1e-10:
                break
            loglkl_ck = loglkl

    if it == max_iter:
        warnings.warn("SlopeHunter EM algorithm may not have converged.")
    hunted = pt >= 0.5
    entropy = float(pt[hunted].mean()) if hunted.any() else float("nan")
    slope = float(dir0 * sy0 / sx0)
    return dict(
        b=slope,
        bse=0.0,
        b_CI=(slope, slope),
        iter=it,
        loglkl=loglkl,
        pi0=pi0,
        entropy=entropy,
        Fit=pd.DataFrame(
            {
                "xbeta": xbeta,
                "ybeta": ybeta,
                "pt": pt,
                "po": 1.0 - pt,
                "clusters": np.where(hunted, "Hunted", "Pleiotropic"),
            }
        ),
    )


def hunt(
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    xp: np.ndarray | None = None,
    snp: np.ndarray | None = None,
    xp_thresh: float = 0.001,
    init_pi: float = 0.6,
    init_sigmaIP: float = 1e-5,
    bootstrapping: bool = True,
    M: int = 100,
    seed: int = 777,
    settings: Literal["default", "R_default"] = "default",
) -> dict:
    """Fit SlopeHunter (`SlopeHunter::hunt`) on SNPs passing `xp <= xp_thresh`."""
    _validate_settings(settings)
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    xp_arr = _p_from_beta_se(xbeta, xse) if xp is None else np.asarray(xp, dtype=float)
    snp_arr = np.array([f"snp{i + 1}" for i in range(len(xbeta))]) if snp is None else np.asarray(snp)
    keep = xp_arr <= xp_thresh
    gx, gy = xbeta[keep], ybeta[keep]
    if keep.sum() < 10:
        raise ValueError(f"Only {int(keep.sum())} SNPs pass xp_thresh={xp_thresh}; at least 10 are required.")

    model = shclust(gx, gy, init_pi, init_sigmaIP, rng=np.random.default_rng(seed), settings=settings)
    b = float(model["b"])
    out = dict(
        b=b,
        pi=model["pi0"],
        entropy=model["entropy"],
        Fit=model["Fit"].assign(SNP=snp_arr[keep]),
        iter=model["iter"],
        loglkl=model["loglkl"],
        n_fit=int(keep.sum()),
    )
    if bootstrapping:
        bts = np.empty(M)
        for i in range(1, M + 1):
            rng = np.random.default_rng(seed + i)
            ii = rng.integers(0, len(gx), len(gx))
            try:
                bts[i - 1] = shclust(gx[ii], gy[ii], init_pi, init_sigmaIP, rng=rng, settings=settings)["b"]
            except (ValueError, FloatingPointError):
                bts[i - 1] = np.nan
        if np.isnan(bts).any():
            warnings.warn(f"Only {int((~np.isnan(bts)).sum())} of {M} bootstrap resamples converged.")
            bts = bts[~np.isnan(bts)]
        if len(bts) > 1:
            bse = float(np.std(np.abs(bts), ddof=1)) if settings == "R_default" else float(np.std(bts, ddof=1))
        else:
            bse = float("nan")
        out.update(bse=bse, b_CI=(b - 1.96 * bse, b + 1.96 * bse), Bts_est=bts)
    else:
        out.update(bse=model["bse"], b_CI=model["b_CI"])
    return out


def sh_adj(
    res: dict,
    xbeta: np.ndarray,
    xse: np.ndarray,
    ybeta: np.ndarray,
    yse: np.ndarray,
    settings: Literal["default", "R_default"] = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust summary statistics `(ybeta, yse)` using a fitted SlopeHunter result (`SlopeHunter::SHadj`)."""
    _validate_settings(settings)
    xbeta, xse, ybeta, yse = (np.asarray(v, dtype=float) for v in (xbeta, xse, ybeta, yse))
    b, bse = float(res["b"]), float(res["bse"])
    yb = ybeta - b * xbeta
    ys = np.sqrt(yse ** 2 + b ** 2 * xse ** 2 + bse ** 2 * xbeta ** 2 + bse ** 2 * xse ** 2)
    return yb, ys


def harmonise_effects(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Harmonise two summary-statistic tables (`SNP`, `EA`, `OA`, `BETA`, `SE`), dropping palindromic SNPs."""
    m = x.merge(y, on="SNP", suffixes=("_x", "_y"))
    ea_x, oa_x = m["EA_x"].str.upper(), m["OA_x"].str.upper()
    ea_y, oa_y = m["EA_y"].str.upper(), m["OA_y"].str.upper()
    palin = (ea_x + oa_x).isin(["AT", "TA", "CG", "GC"])
    same = (ea_x == ea_y) & (oa_x == oa_y)
    swap = (ea_x == oa_y) & (oa_x == ea_y)
    flip = (ea_x == ea_y.map(_COMP)) & (oa_x == oa_y.map(_COMP))
    flip_swap = (ea_x == oa_y.map(_COMP)) & (oa_x == ea_y.map(_COMP))
    keep = ~palin & (same | swap | flip | flip_swap)
    m = m[keep].copy()
    sign = np.where((swap | flip_swap)[keep], -1.0, 1.0)
    return pd.DataFrame(
        {
            "SNP": m["SNP"],
            "A1": m["EA_x"],
            "A2": m["OA_x"],
            "xbeta": m["BETA_x"],
            "xse": m["SE_x"],
            "ybeta": m["BETA_y"] * sign,
            "yse": m["SE_y"],
        }
    ).reset_index(drop=True)
