"""Python implementation of GCTA-mtCOJO and GSMR (Zhu et al., Nature Communications 2018; GPL-3.0-or-later).

Provides:
  * Two-step univariate & one-step bivariate LD-score regression (`mtcojo_ldsc`, `ldsc_hsq_two_step`, `ldsc_gcov_one_step`).
  * Generalized Summary-data-based Mendelian Randomization (`gsmr_meta`) with clumping and single/global HEIDI-outlier filtering.
  * Single- and multi-covariate GWAS summary conditioning (`mtcojo`, `cond_single`, `cond_multiple`).
  * PLINK1 binary `.bed` reader (`read_bed`) and genotype correlation helper (`genotype_corr`).
"""
from __future__ import annotations

from collections import Counter
import os
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats

EPS = 1e-6


def _p_from_beta_se(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    return stats.chi2.sf((np.asarray(beta, dtype=float) / np.asarray(se, dtype=float)) ** 2, 1)


def read_bed(prefix: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Read a SNP-major PLINK1 binary fileset (`prefix.{bed,bim,fam}`)."""
    bim = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None, engine="python", names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    fam = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None, engine="python")
    n, m = len(fam), len(bim)
    nbytes = (n + 3) // 4
    raw = np.fromfile(prefix + ".bed", dtype=np.uint8)
    if raw[0] != 0x6C or raw[1] != 0x1B or raw[2] != 0x01:
        raise ValueError("Only SNP-major PLINK1 .bed files are supported.")
    raw = raw[3:].reshape(m, nbytes)
    lut = np.array([2, np.nan, 1, 0], dtype=np.float32)
    codes = np.stack([(raw >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(m, nbytes * 4)[:, :n]
    return lut[codes].T.copy(), bim


def genotype_corr(G: np.ndarray) -> np.ndarray:
    """SNP x SNP Pearson correlation matrix with mean-imputation of missing genotypes."""
    Gc = np.array(G, dtype=np.float64, copy=True)
    mu = np.nanmean(Gc, axis=0)
    bad = np.isnan(Gc)
    Gc[bad] = np.take(mu, np.where(bad)[1])
    Gc -= Gc.mean(axis=0)
    nrm = np.sqrt((Gc ** 2).sum(axis=0))
    nrm[nrm == 0] = np.inf
    Gc /= nrm
    return Gc.T @ Gc


def quantile(vals_sorted: np.ndarray, prob: float) -> float:
    v = np.asarray(vals_sorted, dtype=float)
    if len(v) == 0:
        return float("nan")
    idx = prob * (len(v) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return float(v[hi])
    return float((hi - idx) * v[lo] + (idx - lo) * v[hi])


def bh_fdr(p: list[float] | np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    n = len(p_arr)
    order = np.argsort(-p_arr, kind="stable")
    fdr = np.empty(n)
    mn = 1.0
    for i, k in enumerate(order):
        c = n / (n - i) * p_arr[k]
        mn = min(mn, c)
        fdr[k] = min(1.0, mn)
    return fdr


def _pnorm_upper(x: float) -> float:
    z = -x if x > 0 else x
    t0 = 1.0 / (1.0 + 0.2316419 * abs(z))
    z1 = np.exp(-0.5 * z * z) / 2.50662827463
    p0 = z1 * t0 * (0.31938153 + t0 * (-0.356563782 + t0 * (1.781477937 + t0 * (-1.821255978 + 1.330274429 * t0))))
    return float(p0 if x >= 0 else 1.0 - p0)


def _K(z: float, lam: np.ndarray) -> float:
    return float(-0.5 * np.log(1.0 - 2.0 * z * lam).sum())


def _Kp(z: float, lam: np.ndarray) -> float:
    return float((lam / (1.0 - 2.0 * z * lam)).sum())


def _Kpp(z: float, lam: np.ndarray) -> float:
    return float(2.0 * (lam ** 2 / (1.0 - 2.0 * z * lam) ** 2).sum())


def _brent_kp(lam: np.ndarray, x: float, lo: float, hi: float, tol: float = 1e-8) -> float:
    fa, fb = _Kp(lo, lam) - x, _Kp(hi, lam) - x
    if fa * fb >= 0:
        return lo if fa < fb else hi
    return float(optimize.brentq(lambda z: _Kp(z, lam) - x, lo, hi, xtol=tol, rtol=4.0 * np.finfo(float).eps))


def _psadd(x: float, lam: np.ndarray) -> float:
    d = float(lam.max())
    if d <= 0:
        return 2.0
    lam = lam / d
    x = x / d
    m = float(lam.min())
    if m < 0:
        lmin = 0.499995 / m
    elif x > lam.sum():
        lmin = -0.01
    else:
        lmin = -0.5 * len(lam) / x
    lmax = 0.499995 / float(lam.max())
    hz = _brent_kp(lam, x, lmin, lmax)
    if hz > lmax + 9:
        return 2.0
    sign = -1.0 if hz < 0 else 1.0
    w = sign * np.sqrt(2.0 * (hz * x - _K(hz, lam)))
    v = hz * np.sqrt(_Kpp(hz, lam))
    if abs(hz) < 1e-4:
        return 2.0
    return _pnorm_upper(w + np.log(v / w) / w)


def _psatt(x: float, lam: np.ndarray) -> float:
    s = float(lam.sum())
    if abs(s) < 1e-12:
        return 2.0
    sq = float(lam @ lam)
    a, b = sq / s, s * s / sq
    return float(stats.chi2.sf(x / a, b))


def pchisqsum(x: float, lam: np.ndarray) -> float:
    """Upper-tail p-value of a weighted sum of 1-df chi-square variables (saddle-point + Satterthwaite fallback)."""
    lam_a = np.asarray(lam, dtype=float)
    p = _psadd(x, lam_a)
    if p > 1.0:
        p = _psatt(x, lam_a)
    return p


def _w_hsq(intercept, h, M, ref_ld, w_ld, n):
    h = min(max(h, 0.0), 1.0)
    ref_ld = np.maximum(ref_ld, 1.0)
    w_ld = np.maximum(w_ld, 1.0)
    den = intercept + h / M * n * ref_ld
    return 1.0 / (2.0 * w_ld * den ** 2)


def _irls(st, x, y, wt, ref_ld, w_ld, n, M, intercept_flag, x_flag):
    wx = x[:, 0].copy()
    s = np.sqrt(wt)
    s = s / s.sum()
    xs = x * s[:, None]
    ys = y * s
    if intercept_flag:
        coef = np.linalg.solve(xs.T @ xs, xs.T @ ys)
        st["hsq"] = float(coef[0] * M / n.mean())
        st["intercept"] = float(coef[1])
    else:
        coef = float((xs[:, 0] @ ys) / (xs[:, 0] @ xs[:, 0]))
        st["hsq"] = float(coef * M / n.mean())
    return _w_hsq(st["intercept"], st["hsq"], M, wx if x_flag else ref_ld, w_ld, n)


def ldsc_hsq_two_step(chi: np.ndarray, n: np.ndarray, ref_ld: np.ndarray, w_ld: np.ndarray, M: float) -> tuple[float, float]:
    """Two-step IRLS SNP-heritability LDSC (`(intercept, h2)`)."""
    chi_a = np.asarray(chi, dtype=float).copy()
    st = {"intercept": 1.0, "hsq": float((chi_a.mean() - 1.0) * M / (ref_ld * n).mean())}
    wt = _w_hsq(1.0, st["hsq"], M, ref_ld, w_ld, n)
    x = np.column_stack([ref_ld * n / n.mean(), np.ones(len(chi_a))])
    sub = chi_a < 30.0
    swt = wt[sub]
    for _ in range(2):
        swt = _irls(st, x[sub], chi_a[sub], swt, ref_ld[sub], w_ld[sub], n[sub], M, True, True)
    st2 = {"intercept": 0.0, "hsq": st["hsq"]}
    _ = _irls(st2, x[sub], chi_a[sub], swt, ref_ld[sub], w_ld[sub], n[sub], M, True, True)
    intercept = st2["intercept"]
    chi2 = chi_a - intercept
    st3 = {"intercept": intercept, "hsq": st2["hsq"]}
    for _ in range(3):
        wt = _irls(st3, x, chi2, wt, ref_ld, w_ld, n, M, False, False)
    return float(intercept), float(st3["hsq"])


def _w_gcov(i1, h1, i2, h2, ig, g, M, ref_ld, w_ld, n1, n2, ng):
    h1, h2 = min(max(h1, 0.0), 1.0), min(max(h2, 0.0), 1.0)
    g = min(max(g, -1.0), 1.0)
    ref_ld = np.maximum(ref_ld, 1.0)
    w_ld = np.maximum(w_ld, 1.0)
    d1 = n1 * h1 * ref_ld / M + i1
    d2 = n2 * h2 * ref_ld / M + i2
    d3 = ng * g * ref_ld / M + ig
    return 1.0 / (w_ld * (d1 * d2 + d3 * d3))


def ldsc_gcov_one_step(z12, ng, ref_ld, w_ld, i1, h1, n1, i2, h2, n2, M) -> tuple[float, float]:
    """One-step IRLS cross-trait LDSC (`(cross_intercept, rg)`)."""
    gcov = float(z12.mean() * M / (ref_ld * ng).mean())
    ig = 0.0
    wt = _w_gcov(i1, h1, i2, h2, ig, gcov, M, ref_ld, w_ld, n1, n2, ng)
    x = np.column_stack([ref_ld * ng / ng.mean(), np.ones(len(z12))])
    for _ in range(3):
        s = np.sqrt(wt)
        s = s / s.sum()
        xs, ys = x * s[:, None], z12 * s
        coef = np.linalg.solve(xs.T @ xs, xs.T @ ys)
        gcov = float(coef[0] * M / ng.mean())
        ig = float(coef[1])
        wt = _w_gcov(i1, h1, i2, h2, ig, gcov, M, ref_ld, w_ld, n1, n2, ng)
    return float(ig), float(gcov / np.sqrt(h1 * h2))


def mtcojo_ldsc(z: np.ndarray, n: np.ndarray, ref_ld: np.ndarray, w_ld: np.ndarray, M: float, trait_names=None):
    """Univariate and bivariate LDSC across `T` traits (`z`, `n` of shape `(S, T)`)."""
    S, T = z.shape
    if np.any((z ** 2).sum(0) / S > 10.0):
        return None, None, False
    inter = np.zeros((T, T))
    slope = np.zeros((T, T))
    for i in range(T):
        it, h = ldsc_hsq_two_step(z[:, i] ** 2, n[:, i], ref_ld, w_ld, M)
        if not h > 0:
            raise ValueError(f"Negative SNP heritability estimate for {trait_names[i] if trait_names else i}")
        inter[i, i], slope[i, i] = it, h
    for i in range(T - 1):
        for j in range(i + 1, T):
            ig, rg = ldsc_gcov_one_step(
                z[:, i] * z[:, j],
                np.sqrt(n[:, i] * n[:, j]),
                ref_ld,
                w_ld,
                inter[i, i],
                slope[i, i],
                n[:, i],
                inter[j, j],
                slope[j, j],
                n[:, j],
                M,
            )
            inter[i, j] = inter[j, i] = ig
            slope[i, j] = slope[j, i] = rg
    return inter, slope, True


def read_ldscore_chr(prefix: str):
    scores, M = {}, 0
    for c in range(1, 23):
        f = None
        for suf in (".l2.ldscore", ".l2.ldscore.gz"):
            if os.path.exists(f"{prefix}{c}{suf}"):
                f = f"{prefix}{c}{suf}"
        if f is None:
            raise FileNotFoundError(f"{prefix}{c}.l2.ldscore[.gz]")
        d = pd.read_csv(f, sep=r"\s+", engine="python")
        col = d.columns[3] if d.shape[1] == 4 else d.columns[5]
        scores.update(zip(d.iloc[:, 1], d[col]))
        mf = f"{prefix}{c}.l2.M_5_50"
        if os.path.exists(mf):
            with open(mf) as fh:
                M += int(float(fh.read().split()[0]))
    return scores, M


def clump(chi, chrom, bp, G, chi_thresh, wind_kb, r2_thresh) -> list[int]:
    cand = np.flatnonzero(chi > chi_thresh)
    order = cand[np.lexsort((-cand, -chi[cand]))]
    flagged = np.zeros(len(chi), bool)
    wind = wind_kb * 1e3
    cand_set = np.zeros(len(chi), bool)
    cand_set[cand] = True
    Gc = G.astype(np.float64)
    mu = np.nanmean(Gc, 0)
    bad = np.isnan(Gc)
    Gc[bad] = np.take(mu, np.where(bad)[1])
    Gc = Gc - Gc.mean(0)
    index = []
    for i in order:
        if flagged[i]:
            continue
        near = np.flatnonzero(cand_set & (chrom == chrom[i]) & (np.abs(bp - bp[i]) < wind))
        near = near[near != i]
        if len(near):
            xi = Gc[:, i]
            num = Gc[:, near].T @ xi
            den = np.sqrt((xi @ xi) * (Gc[:, near] ** 2).sum(0))
            r2 = (num / den) ** 2
            flagged[near[r2 >= r2_thresh]] = True
        index.append(int(i))
    return index


def rm_cor_elements(r: np.ndarray, r2_thresh: float) -> list[int]:
    n = r.shape[0]
    thr = np.sqrt(r2_thresh)
    ii, jj = [], []
    for i in range(n):
        for j in range(i):
            if abs(r[i, j]) > thr:
                ii.append(i)
                jj.append(j)
    if not ii:
        return list(range(n))
    cnt = Counter(ii) + Counter(jj)
    for k in range(len(ii)):
        if cnt[ii[k]] < cnt[jj[k]]:
            ii[k], jj[k] = jj[k], ii[k]
    rm = set(ii)
    return [i for i in range(n) if i not in rm]


class _Inst:
    def __init__(self, bzx, bzx_se, bzy, bzy_se, bzy_p, r):
        self.bzx, self.bzx_se, self.bzy, self.bzy_se, self.bzy_p, self.r = bzx, bzx_se, bzy, bzy_se, bzy_p, r
        self.bxy = bzy / bzx
        self.z1 = bzx_se / bzx
        self.z2 = bzy_se / bzx

    def cov_parts(self, kept):
        k = np.asarray(kept)
        r = self.r[np.ix_(k, k)]
        return r * np.outer(self.z2[k], self.z2[k]), r * np.outer(self.z1[k], self.z1[k])


def _est_bxy_gsmr(inst, kept, bxy_sorted):
    k = np.asarray(kept)
    med = quantile(bxy_sorted, 0.5)
    p1, p2 = inst.cov_parts(k)
    cov = p1 + p2 * med ** 2 + EPS * np.eye(len(k))
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Variance-covariance matrix of bxy is not invertible") from exc
    inv = np.linalg.inv(cov)
    one_t_v = np.ones(len(k)) @ inv
    var = float(1.0 / one_t_v.sum())
    b = float(var * (one_t_v @ inst.bxy[k]))
    p = float(stats.chi2.sf(b * b / var, 1))
    return b, float(np.sqrt(var)), p, len(k), one_t_v


def _cov_bxy_heidi_outlier(inst, kept):
    p1, p2 = inst.cov_parts(kept)
    bx = inst.bxy[np.asarray(kept)]
    return p1 + p2 * np.outer(bx, bx)


def _topsnp_bxy(inst, kept) -> int:
    k = np.asarray(kept)
    bs = np.sort(inst.bxy[k])
    lo, hi = quantile(bs, 0.4), quantile(bs, 0.6)
    best, top = 0.0, -9
    for pos, s in enumerate(k):
        if lo <= inst.bxy[s] <= hi:
            ch = (inst.bzx[s] / inst.bzx_se[s]) ** 2
            if ch > best:
                best, top = float(ch), pos
    return top


def _topsnp_heidi_outlier(inst, kept, ci) -> int:
    k = np.asarray(kept)
    cand = np.asarray(ci)
    bs = np.sort(inst.bxy[k[cand]])
    probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lo = [quantile(bs, probs[i]) for i in range(5)]
    hi = [quantile(bs, probs[i + 1]) for i in range(5)]
    top = [0] * 5
    mx = [0.0] * 5
    sm = [0.0] * 5
    cnt = [0.0] * 5
    for c in cand:
        s = k[c]
        for j in range(5):
            if inst.bxy[s] < lo[j] or inst.bxy[s] >= hi[j]:
                continue
            ch = float((inst.bzx[s] / inst.bzx_se[s]) ** 2)
            sm[j] += ch
            cnt[j] += 1
            if mx[j] < ch:
                mx[j] = ch
                top[j] = int(c)
            break
    topsnp, best = 0, 0.0
    for j in range(5):
        if cnt[j] < 1:
            continue
        mean = sm[j] / cnt[j]
        if best < mean:
            best, topsnp = mean, top[j]
    return topsnp


def _heidi_outlier_topsnp(inst, cov, kept, top: int, multi: bool):
    k = np.asarray(kept)
    d = inst.bzy[k] / inst.bzx[k] - inst.bxy[k[top]]
    var_d = cov + cov[top, top] - cov[:, [top]] - cov[[top], :]
    var_d[top, top] = 1.0
    indi = stats.chi2.sf(d ** 2 / np.diag(var_d), 1)
    if not multi:
        return indi, 0.0
    d2 = np.delete(d, top)
    v2 = np.delete(np.delete(var_d, top, 0), top, 1)
    sd = np.sqrt(np.diag(v2))
    corr = v2 / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    chival = float(np.sum(d2 ** 2 / np.diag(v2)))
    ev = np.linalg.eigvalsh((corr + corr.T) / 2.0)
    return indi, pchisqsum(chival, ev)


def _interval(bsorted, bvals, idx, lo_p: float, hi_p: float):
    lo, hi = quantile(bsorted, lo_p), quantile(bsorted, hi_p)
    return [i for i in idx if lo <= bvals[i] <= hi]


def _indi_iter(inst, cov, kept, heidi_thresh: float):
    k = np.asarray(kept)
    pos = list(range(len(k)))
    bvals = inst.bxy[k]
    ci = _interval(np.sort(bvals), bvals, pos, 0.05, 0.95)
    if not ci:
        return [], cov, 0
    top = _topsnp_heidi_outlier(inst, kept, ci)
    indi, _ = _heidi_outlier_topsnp(inst, cov, kept, top, False)
    j = int(np.argmin(indi))
    if indi[j] < heidi_thresh:
        kept = [s for q, s in enumerate(kept) if q != j]
        cov = np.delete(np.delete(cov, j, 0), j, 1)
    return kept, cov, 1


def _global_iter(inst, cov, kept, heidi_thresh: float):
    k = np.asarray(kept)
    cand = [i for i in range(len(k)) if inst.bzy_p[k[i]] >= 5e-8]
    bvals = inst.bxy[k]
    ci = _interval(np.sort(bvals[cand]) if cand else np.array([]), bvals, cand, 0.05, 0.95)
    if not ci:
        return [], cov, -9.0
    top = _topsnp_heidi_outlier(inst, kept, ci)
    indi, gp = _heidi_outlier_topsnp(inst, cov, kept, top, True)
    if gp < heidi_thresh:
        j = int(np.argmin(indi))
        kept = [s for q, s in enumerate(kept) if q != j]
        cov = np.delete(np.delete(cov, j, 0), j, 1)
    return kept, cov, gp


def gsmr_meta(
    bzx,
    bzx_se,
    bzx_p,
    bzy,
    bzy_se,
    bzy_p,
    valid,
    chrom,
    bp,
    G,
    gwas_thresh: float = 5e-8,
    wind_kb: float = 10000,
    r2_thresh: float = 0.05,
    std_heidi_thresh: float = 0.01,
    global_heidi_thresh: float = 0.01,
    ld_fdr_thresh: float = 0.05,
    nsnp_min: int = 10,
    beta_version: int = 0,
    n_ref: int | None = None,
) -> dict:
    """Port of `gcta::gsmr_meta` with LD clumping and HEIDI-outlier filtering."""
    res = dict(bxy=np.nan, se=np.nan, p=np.nan, n_snps=np.nan, heidi_p=np.nan, instruments=[], pleio=[], err=None)
    n_ref = G.shape[0] if n_ref is None else n_ref
    chi = np.where(valid, (bzx / bzx_se) ** 2, -np.inf)
    nsnp = int((chi > stats.chi2.isf(gwas_thresh, 1)).sum())
    fail = f"Not enough SNPs to perform GSMR analysis. At least {nsnp_min} SNPs are required."
    if nsnp < nsnp_min:
        res["err"] = fail
        return res
    index = clump(chi, chrom, bp, G, stats.chi2.isf(gwas_thresh, 1), wind_kb, r2_thresh)
    if len(index) < nsnp_min:
        res["err"] = fail
        return res
    index_arr = np.asarray(index)
    r = genotype_corr(G[:, index_arr])
    kept = rm_cor_elements(r, r2_thresh)
    n = len(kept)
    if n < nsnp_min:
        res["err"] = fail
        return res

    pairs = [(stats.chi2.sf(r[kept[i], kept[j]] ** 2 * n_ref, 1), i * n + j) for i in range(n - 1) for j in range(i + 1, n)]
    pairs.sort(key=lambda t: -t[0])
    if pairs:
        fdr = bh_fdr([t[0] for t in pairs])
        for q, (_, code) in enumerate(pairs):
            if fdr[q] < ld_fdr_thresh:
                break
            a, b_ = divmod(code, n)
            r[kept[a], kept[b_]] = r[kept[b_], kept[a]] = 0.0

    inst = _Inst(bzx[index_arr], bzx_se[index_arr], bzy[index_arr], bzy_se[index_arr], bzy_p[index_arr], r)
    remain = list(kept)
    heidi_flag = (abs(global_heidi_thresh) + abs(std_heidi_thresh)) != 0
    if heidi_flag and n <= 5:
        heidi_flag = False
    cov = _cov_bxy_heidi_outlier(inst, kept)
    gp = np.nan
    if beta_version and heidi_flag:
        while True:
            before = len(kept)
            kept, cov, process = _indi_iter(inst, cov, kept, std_heidi_thresh)
            if before - len(kept) == 0:
                break
            if len(kept) < nsnp_min or process == 0:
                res["err"] = fail
                return res
        while True:
            before = len(kept)
            kept, cov, gp = _global_iter(inst, cov, kept, global_heidi_thresh)
            if before - len(kept) == 0:
                break
            if len(kept) < nsnp_min or gp < 0:
                res["err"] = fail
                return res
    elif (not beta_version) and heidi_flag:
        top = _topsnp_bxy(inst, kept)
        if top < 0:
            res["err"] = fail
            return res
        indi, _ = _heidi_outlier_topsnp(inst, cov, kept, top, False)
        kept = [s for q, s in enumerate(kept) if q == top or indi[q] >= std_heidi_thresh]

    bs = np.sort(inst.bxy[np.asarray(kept)])
    b, se, _, nk, _ = _est_bxy_gsmr(inst, kept, bs)
    res.update(
        bxy=b,
        se=se,
        p=float(stats.chi2.sf(b * b / (se * se), 1)),
        n_snps=nk,
        heidi_p=gp,
        instruments=index_arr[np.asarray(kept)].tolist(),
        pleio=index_arr[np.asarray([s for s in remain if s not in set(kept)], int)].tolist() if len(remain) > len(kept) else [],
    )
    return res


def cond_single(bzy, bzy_se, bzx, bzx_se, bxy: float, intercept_xy: float = 0.0):
    b = bzy - bzx * bxy
    se = np.sqrt(bzy_se ** 2 + bxy ** 2 * bzx_se ** 2 - 2.0 * bxy * intercept_xy * bzx_se * bzy_se)
    return b, se


def cond_multiple(bzy, bzy_se, bzx, bzx_se, bxy, ldsc_intercept, ldsc_slope, vp):
    C = bzx.shape[1]
    Dm = np.diag(np.sqrt(np.diag(ldsc_slope)[1:] * vp[1:]))
    Rm = np.eye(C)
    for i in range(C):
        for j in range(C):
            if i != j:
                Rm[i, j] = ldsc_slope[i + 1, j + 1]
    bj = np.linalg.solve(Dm, np.linalg.solve(Rm, Dm @ bxy))
    I_mat = np.eye(C)
    for i in range(C - 1):
        for j in range(i + 1, C):
            I_mat[i, j] = I_mat[j, i] = ldsc_intercept[i + 1, j + 1]
    b = bzy - bzx @ bj
    cross = (bzx_se * bzy_se[:, None] * bj[None, :] * ldsc_intercept[0, 1:][None, :]).sum(1)
    sv = bzx_se * bj[None, :]
    var = np.einsum("si,ij,sj->s", sv, I_mat, sv)
    return b, np.sqrt(bzy_se ** 2 + var - 2.0 * cross)


def mtcojo(
    df: pd.DataFrame,
    G: np.ndarray | None = None,
    ldsc: dict | None = None,
    bxy_external=None,
    ncov: int = 1,
    freq_thresh: float = 0.2,
    gwas_thresh: float = 5e-8,
    clump_wind_size: float = 10000,
    clump_r2_thresh: float = 0.05,
    std_heidi_thresh: float = 0.01,
    global_heidi_thresh: float = 0.01,
    ld_fdr_thresh: float = 0.05,
    nsnp_gsmr: int = 10,
    gsmr_beta_version: int = 0,
    intercept_xy: float | None = None,
    verbose: bool = False,
) -> dict:
    """Run GCTA-mtCOJO multi-trait conditional analysis on summary statistics `df`."""
    suffix = [""] + [str(c) for c in range(2, ncov + 1)]
    S = len(df)
    ybeta, yse = df["YBETA"].to_numpy(float), df["YSE"].to_numpy(float)
    xb = np.column_stack([df["XBETA" + s].to_numpy(float) for s in suffix])
    xs = np.column_stack([df["XSE" + s].to_numpy(float) for s in suffix])
    keep = np.all(np.isfinite(xb) & np.isfinite(xs), 1) & np.isfinite(ybeta) & np.isfinite(yse) & (yse >= EPS) & np.all(xs >= EPS, 1)
    log = print if verbose else (lambda *a, **k: None)
    freq = df["FREQ"].to_numpy(float) if "FREQ" in df.columns else None

    if G is not None:
        ref_af = np.nanmean(G, 0) / 2.0
        if freq is not None:
            diff = (np.abs(ref_af - freq) > freq_thresh) & keep
            if diff.sum() > 0.05 * keep.sum():
                raise ValueError("Too many SNPs show large allele-frequency differences against the LD reference.")
            keep &= ~diff
        keep &= ~(np.minimum(ref_af, 1.0 - ref_af) == 0)
    log(f"[mtcojo] {int(keep.sum())} of {S} SNPs retained after QC")

    vp = np.ones(ncov + 1)
    if freq is not None and "YN" in df.columns:
        h = 2.0 * freq * (1.0 - freq)
        vp[0] = float(np.nanmedian((h * ybeta ** 2 + h * df["YN"].to_numpy(float) * yse ** 2)[keep]))
        for c, s in enumerate(suffix):
            if "XN" + s in df.columns:
                vp[c + 1] = float(np.nanmedian((h * xb[:, c] ** 2 + h * df["XN" + s].to_numpy(float) * xs[:, c] ** 2)[keep]))

    inter = np.zeros((ncov + 1, ncov + 1))
    slope = np.zeros_like(inter)
    ldsc_ok = False
    ldsc_out = None
    if ldsc is not None:
        use = keep & (ldsc["ref_ld"] >= 0) & np.isfinite(ldsc["ref_ld"]) & np.isfinite(ldsc["w_ld"])
        zz = np.column_stack([ybeta / yse] + [xb[:, c] / xs[:, c] for c in range(ncov)])[use]
        nn = np.column_stack([df["YN"].to_numpy(float)] + [df["XN" + s].to_numpy(float) for s in suffix])[use]
        inter_, slope_, ok = mtcojo_ldsc(zz, nn, ldsc["ref_ld"][use], ldsc["w_ld"][use], ldsc["M"], ["target"] + [f"covar{c + 1}" for c in range(ncov)])
        if ok:
            inter, slope, ldsc_ok = inter_, slope_, True
            ldsc_out = dict(intercept=inter, slope=slope)
        else:
            if ncov > 1:
                raise ValueError("LD score regression failed (mean chi2 > 10); multi-covariate mtCOJO requires LDSC.")
            warnings.warn("Insufficient LDSC signal; assuming intercept = 0.")
    if (not ldsc_ok) and intercept_xy is not None:
        inter[0, 1] = inter[1, 0] = intercept_xy

    bxy = np.full(ncov, np.nan)
    gsmr_res = []
    if bxy_external is not None:
        bxy = np.asarray(bxy_external, dtype=float).reshape(ncov)
    else:
        if G is None:
            raise ValueError("GSMR requires an LD reference genotype matrix `G` or external `bxy_external`.")
        chrom = df["CHR"].astype(str).to_numpy()
        bp = df["BP"].to_numpy(float)
        if not all(np.all(np.diff(bp[chrom == c]) >= 0) for c in np.unique(chrom)):
            raise ValueError("Input rows must be sorted by CHR then BP.")
        for c in range(ncov):
            valid = keep & np.isfinite(xb[:, c])
            res = gsmr_meta(
                xb[:, c],
                xs[:, c],
                _p_from_beta_se(xb[:, c], xs[:, c]),
                ybeta,
                yse,
                _p_from_beta_se(ybeta, yse),
                valid,
                chrom,
                bp,
                G,
                gwas_thresh,
                clump_wind_size,
                clump_r2_thresh,
                std_heidi_thresh,
                global_heidi_thresh,
                ld_fdr_thresh,
                nsnp_gsmr,
                gsmr_beta_version,
            )
            if res["err"]:
                warnings.warn(res["err"])
                if ldsc_ok:
                    bxy[c] = slope[c + 1, 0] * np.sqrt(slope[0, 0] / slope[c + 1, c + 1])
                else:
                    raise ValueError("GSMR failed and no LDSC estimate is available.")
            else:
                bxy[c] = res["bxy"]
            gsmr_res.append(res)

    if ncov == 1:
        b_c, se_c = cond_single(ybeta, yse, xb[:, 0], xs[:, 0], float(bxy[0]), float(inter[0, 1]))
    else:
        b_c, se_c = cond_multiple(ybeta, yse, xb, xs, bxy, inter, slope, vp)
    b_c = np.where(keep, b_c, np.nan)
    se_c = np.where(keep, se_c, np.nan)
    return dict(ybeta_adj=b_c, yse_adj=se_c, bxy=bxy, gsmr=gsmr_res, ldsc=ldsc_out, vp=vp, keep=keep)
