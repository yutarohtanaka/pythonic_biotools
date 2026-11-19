"""Reference expression baseline, dispersion estimation, and Poisson-LogNormal emissions."""

from __future__ import annotations

import math

from typing import Literal

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp


def _hermgauss_logweights(n_quadrature: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Hermite nodes scaled by sqrt(2) and log normalized weights."""
    u, w = np.polynomial.hermite.hermgauss(n_quadrature)
    z_nodes = np.sqrt(2.0) * u
    log_weights = np.log(w / math.sqrt(math.pi))
    return z_nodes.astype(np.float64), log_weights.astype(np.float64)


def compute_ref_expression(
    counts: np.ndarray | sparse.spmatrix,
    ref_indices: np.ndarray | None = None,
    ref_expression: np.ndarray | None = None,
    eps_ref: float = 1e-3,
    sigma_min: float = 0.15,
    sigma_max: float = 2.5,
    default_sigma: float = 0.4,
    expr_model: Literal["pln", "gaussian_logratio"] = "pln",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate diploid reference baseline (`lambda_ref` or `mu_ref`) and dispersions `sigma_g`."""
    if expr_model not in ("pln", "gaussian_logratio"):
        raise ValueError(f"Unsupported expr_model {expr_model!r}.")
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("counts must be a non-empty 2D matrix.")
    if ref_indices is not None and len(ref_indices) == 0:
        raise ValueError("ref_indices must not be empty when provided.")
    n_cells, n_genes = counts.shape

    if expr_model == "gaussian_logratio":
        if ref_expression is not None:
            ref_vec = np.asarray(ref_expression, dtype=np.float64)
            if ref_vec.shape != (n_genes,):
                raise ValueError(
                    f"ref_expression length ({ref_vec.shape}) != number of genes ({n_genes})."
                )
            if not np.all(np.isfinite(ref_vec)):
                raise ValueError("ref_expression must be finite.")
            mu_ref = ref_vec.copy()
        elif ref_indices is not None:
            sub = counts[ref_indices]
            if sparse.issparse(sub):
                mu_ref = np.asarray(sub.mean(axis=0), dtype=np.float64).ravel()
            else:
                mu_ref = np.mean(np.asarray(sub, dtype=np.float64), axis=0)
        else:
            mu_ref = np.zeros(n_genes, dtype=np.float64)

        sub_disp = counts[ref_indices] if ref_indices is not None else counts
        if sub_disp.shape[0] >= 2:
            if sparse.issparse(sub_disp):
                mean_x = np.asarray(sub_disp.mean(axis=0), dtype=np.float64).ravel()
                mean_x2 = np.asarray(sub_disp.power(2).mean(axis=0), dtype=np.float64).ravel()
                n_sub = float(sub_disp.shape[0])
                var_x = np.maximum(0.0, mean_x2 - mean_x**2) * (n_sub / (n_sub - 1.0))
                sigma_raw = np.sqrt(var_x)
            else:
                arr_sub = np.asarray(sub_disp, dtype=np.float64)
                sigma_raw = np.std(arr_sub, axis=0, ddof=1)
            med_sigma = float(np.median(sigma_raw))
            sigma_g = np.clip(0.7 * sigma_raw + 0.3 * med_sigma, sigma_min, sigma_max)
        else:
            sigma_g = np.full(n_genes, default_sigma, dtype=np.float64)

        return np.ascontiguousarray(mu_ref, dtype=np.float64), np.ascontiguousarray(
            sigma_g, dtype=np.float64
        )

    if ref_expression is not None:
        ref_vec = np.asarray(ref_expression, dtype=np.float64)
        if ref_vec.shape != (n_genes,):
            raise ValueError(
                f"ref_expression length ({ref_vec.shape}) != number of genes ({n_genes})."
            )
        if not np.all(np.isfinite(ref_vec)) or np.any(ref_vec < 0.0):
            raise ValueError("ref_expression must be finite and non-negative.")
        total = float(ref_vec.sum())
        if total <= 0.0:
            raise ValueError("ref_expression must have a positive sum.")
        if abs(total - 1.0) <= 1e-3 and float(ref_vec.max()) <= 1.0:
            ref_lambda = np.maximum(ref_vec, 1e-8)
            ref_lambda = ref_lambda / float(ref_lambda.sum())
        else:
            ref_lambda = (ref_vec + eps_ref) / (total + n_genes * eps_ref)
    else:
        sub = counts[ref_indices] if ref_indices is not None else counts
        if sparse.issparse(sub):
            gene_sums = np.asarray(sub.sum(axis=0), dtype=np.float64).ravel()
        else:
            gene_sums = np.asarray(sub, dtype=np.float64).sum(axis=0)
        pseudocount = 0.5 if float(gene_sums.sum()) > 0.0 else eps_ref
        ref_lambda = (gene_sums + pseudocount) / (float(gene_sums.sum()) + n_genes * pseudocount)

    # Estimate gene-level log-expression dispersion from reference cells (if >= 2 valid cells)
    sub_disp = counts[ref_indices] if ref_indices is not None else counts
    if sub_disp.shape[0] >= 2:
        if sparse.issparse(sub_disp):
            lib_sizes = np.asarray(sub_disp.sum(axis=1), dtype=np.float64).ravel()
            valid = lib_sizes > 0.0
            n_valid = int(valid.sum())
            if n_valid >= 2:
                sub_valid = sub_disp[valid].tocsr()
                inv_l = sparse.diags(1.0 / lib_sizes[valid])
                prop = inv_l @ sub_valid
                mean_p = np.asarray(prop.mean(axis=0), dtype=np.float64).ravel()
                mean_p2 = np.asarray(prop.power(2).mean(axis=0), dtype=np.float64).ravel()
                var_p = np.maximum(0.0, mean_p2 - mean_p**2) * (n_valid / (n_valid - 1.0))
                mean_l = float(np.mean(lib_sizes[valid]))
            else:
                var_p = None
                mean_l = 0.0
        else:
            arr_sub = np.asarray(sub_disp, dtype=np.float64)
            lib_sizes = arr_sub.sum(axis=1)
            valid = lib_sizes > 0.0
            n_valid = int(valid.sum())
            if n_valid >= 2:
                prop = arr_sub[valid] / lib_sizes[valid, None]
                var_p = np.var(prop, axis=0, ddof=1)
                mean_l = float(np.mean(lib_sizes[valid]))
            else:
                var_p = None
                mean_l = 0.0

        if var_p is not None and mean_l > 0.0:
            excess = np.maximum(0.0, var_p - ref_lambda / mean_l)
            sigma_raw = np.sqrt(np.log1p(excess / np.maximum(ref_lambda**2, 1e-12)))
            med_sigma = float(np.median(sigma_raw))
            sigma_g = np.clip(0.7 * sigma_raw + 0.3 * med_sigma, sigma_min, sigma_max)
        else:
            sigma_g = np.full(n_genes, default_sigma, dtype=np.float64)
    else:
        sigma_g = np.full(n_genes, default_sigma, dtype=np.float64)

    return np.ascontiguousarray(ref_lambda, dtype=np.float64), np.ascontiguousarray(
        sigma_g, dtype=np.float64
    )


def compute_diploid_offset(
    counts_vec: np.ndarray,
    ref_lambda: np.ndarray,
    lib_size: float | None = None,
) -> float:
    """Compute robust median diploid log-centering offset for a cell or pseudobulk vector."""
    x = np.asarray(counts_vec, dtype=np.float64).ravel()
    total = float(x.sum()) if lib_size is None else float(lib_size)
    if total <= 0.0:
        return 0.0
    mu_raw = total * ref_lambda
    mask = mu_raw >= 1.0
    if int(mask.sum()) < 10:
        mask = mu_raw >= 0.2
    if int(mask.sum()) < 10:
        return 0.0
    log_ratios = np.log((x[mask] + 0.5) / (mu_raw[mask] + 0.5))
    delta = float(np.median(log_ratios))
    return float(np.clip(delta, -1.5, 1.5))


def compute_diploid_offsets_batch(
    counts: np.ndarray | sparse.spmatrix,
    ref_lambda: np.ndarray,
    lib_sizes: np.ndarray,
) -> np.ndarray:
    """Compute per-cell diploid log-centering offsets across a matrix."""
    n_cells = counts.shape[0]
    offsets = np.zeros(n_cells, dtype=np.float64)
    if n_cells == 0:
        return offsets
    is_sp = sparse.issparse(counts)
    for c in range(n_cells):
        total = float(lib_sizes[c])
        if total <= 0.0:
            continue
        mu_raw = total * ref_lambda
        mask = mu_raw >= 1.0
        if int(mask.sum()) < 10:
            mask = mu_raw >= 0.2
        if int(mask.sum()) < 10:
            continue
        if is_sp:
            row_vals = counts[c, mask].toarray().ravel()
        else:
            row_vals = counts[c, mask]
        log_ratios = np.log((row_vals + 0.5) / (mu_raw[mask] + 0.5))
        offsets[c] = float(np.clip(np.median(log_ratios), -1.5, 1.5))
    return offsets


def pln_logpmf(
    x: np.ndarray | float,
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
    n_quadrature: int = 21,
) -> np.ndarray:
    """Evaluate Poisson-LogNormal log-PMF via Gauss-Hermite quadrature in log-space."""
    x_arr = np.asarray(x, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    if np.any(x_arr < 0.0):
        raise ValueError("PLN count observations x must be non-negative.")
    if np.any(sigma_arr < 0.0):
        raise ValueError("PLN log-dispersion sigma must be non-negative.")

    z_nodes, log_weights = _hermgauss_logweights(n_quadrature)
    x_b, mu_b, sig_b = np.broadcast_arrays(x_arr, mu_arr, sigma_arr)

    eta = np.minimum(mu_b[..., None] + sig_b[..., None] * z_nodes, 20.0)
    log_terms = (
        log_weights
        + x_b[..., None] * eta
        - np.exp(eta)
        - gammaln(x_b[..., None] + 1.0)
    )
    out = logsumexp(log_terms, axis=-1)
    return np.asarray(out, dtype=np.float64)


def gaussian_logratio_logpdf(
    r: np.ndarray | float,
    phi: np.ndarray | float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Evaluate Gaussian log-PDF for pre-computed log expression ratios around log(phi)."""
    r_arr = np.asarray(r, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    sig_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    log_phi = np.log(np.maximum(phi_arr, 1e-12))
    return -0.5 * np.log(2.0 * math.pi * sig_arr**2) - 0.5 * ((r_arr - log_phi) / sig_arr) ** 2
