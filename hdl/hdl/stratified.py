"""Stratified High-Definition Likelihood (HDL.S) partitioned heritability & enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .panel import (
    SettingsType,
    _normalize_r_blocks,
    build_ld_eigen_panel,
    split_block_vectors,
    validate_settings,
)


@dataclass
class HDLStratifiedResult:
    """Result of Stratified High-Definition Likelihood (HDL.S) functional enrichment."""

    tau: np.ndarray
    se_tau: np.ndarray
    p_tau: np.ndarray
    cov_tau: np.ndarray
    h2_total: float
    se_h2_total: float
    h2_annotations: np.ndarray
    prop_snps: np.ndarray
    prop_h2: np.ndarray
    enrichment: np.ndarray
    se_enrichment: np.ndarray
    p_enrichment: np.ndarray
    intercept: float
    loglik: float
    n_snps: int
    n_blocks: int
    n_eigenmodes: int
    annotation_names: tuple[str, ...]
    settings: SettingsType

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "annotation": list(self.annotation_names),
                "prop_snps": self.prop_snps,
                "tau": self.tau,
                "se_tau": self.se_tau,
                "p_tau": self.p_tau,
                "h2": self.h2_annotations,
                "prop_h2": self.prop_h2,
                "enrichment": self.enrichment,
                "se_enrichment": self.se_enrichment,
                "p_enrichment": self.p_enrichment,
            }
        )

    def summary(self) -> str:
        df = self.to_dataframe()
        header = (
            f"HDL.S (settings={self.settings!r}): total_h2={self.h2_total:.4f} "
            f"(SE={self.se_h2_total:.4f}), intercept={self.intercept:.4f}"
        )
        rows = [
            f"  {row.annotation}: tau={row.tau:.3e} (p={row.p_tau:.3g}), "
            f"enrich={row.enrichment:.2f}x (SE={row.se_enrichment:.2f}, p={row.p_enrichment:.3g})"
            for row in df.itertuples()
        ]
        return "\n".join([header, *rows])


def _normalize_annotation_blocks(
    annotations_blocks: np.ndarray | pd.DataFrame | Sequence[np.ndarray | pd.DataFrame],
    block_sizes: list[int],
) -> tuple[list[np.ndarray], list[str] | None]:
    """Normalize functional annotations into a list of ``(M_b, A)`` float64 arrays."""
    total_m = sum(block_sizes)
    inferred_names: list[str] | None = None

    if isinstance(annotations_blocks, pd.DataFrame):
        inferred_names = [str(c) for c in annotations_blocks.columns]
        arr_full = annotations_blocks.to_numpy(dtype=np.float64)
        if arr_full.shape[0] != total_m:
            raise ValueError(
                f"annotations_blocks DataFrame has {arr_full.shape[0]} rows, expected {total_m}."
            )
        out: list[np.ndarray] = []
        offset = 0
        for m_b in block_sizes:
            out.append(arr_full[offset : offset + m_b])
            offset += m_b
        return out, inferred_names

    if isinstance(annotations_blocks, np.ndarray):
        arr = np.asarray(annotations_blocks, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim == 2 and arr.shape[0] == total_m:
            out = []
            offset = 0
            for m_b in block_sizes:
                out.append(arr[offset : offset + m_b])
                offset += m_b
            return out, None
        raise ValueError(
            f"annotations_blocks array has shape {arr.shape}, expected ({total_m}, A)."
        )

    seq = list(annotations_blocks)
    if len(seq) != len(block_sizes):
        raise ValueError(
            f"annotations_blocks has {len(seq)} blocks, expected {len(block_sizes)}."
        )
    out = []
    n_annot: int | None = None
    for b_idx, (ab, m_b) in enumerate(zip(seq, block_sizes)):
        if isinstance(ab, pd.DataFrame) and inferred_names is None:
            inferred_names = [str(c) for c in ab.columns]
        ab_arr = np.asarray(ab, dtype=np.float64)
        if ab_arr.ndim == 1:
            ab_arr = ab_arr[:, None]
        if ab_arr.ndim != 2 or ab_arr.shape[0] != m_b:
            raise ValueError(
                f"annotations_blocks[{b_idx}] has shape {ab_arr.shape}, expected ({m_b}, A)."
            )
        if not np.all(np.isfinite(ab_arr)):
            raise ValueError(f"annotations_blocks[{b_idx}] contains non-finite values.")
        if n_annot is None:
            n_annot = ab_arr.shape[1]
        elif ab_arr.shape[1] != n_annot:
            raise ValueError("All annotation blocks must have the same number of columns A.")
        out.append(ab_arr)
    if n_annot is None or n_annot < 1:
        raise ValueError("At least one functional annotation column is required.")
    return out, inferred_names


def _fit_stratified_blocks(
    u_tilde_blocks: list[np.ndarray],
    lam_blocks: list[np.ndarray],
    W_blocks: list[np.ndarray],  # each shape (A, k_b, k_b)
    N: float,
    est_intercept: bool,
    tau_init: np.ndarray,
    b0_init: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Optimize reduced-eigenspace stratified Gaussian log-likelihood via L-BFGS-B."""
    n_annot = tau_init.shape[0]
    B = len(u_tilde_blocks)

    # Scale tau by (N / mean_M_b) for O(1) L-BFGS-B conditioning: gamma = N * tau
    gamma_0 = float(N) * np.maximum(tau_init.copy(), 0.0)
    b_curr = b0_init if est_intercept else 1.0

    x0 = (
        np.concatenate([gamma_0, [b_curr]])
        if est_intercept
        else gamma_0.copy()
    )
    bounds = [(-20.0, 200.0)] * n_annot + ([(0.2, 5.0)] if est_intercept else [])
    eps_eig = 1e-4

    def nll_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        gamma_v = x[:n_annot]
        b_v = float(x[n_annot]) if est_intercept else 1.0
        total_nll = 0.0
        grad_gamma = np.zeros(n_annot, dtype=np.float64)
        grad_b = 0.0

        for b_idx in range(B):
            u_b = u_tilde_blocks[b_idx]
            k_b = u_b.shape[0]
            W_b = W_blocks[b_idx]  # (A, k_b, k_b)
            sig = np.sum(W_b * gamma_v[:, None, None], axis=0)
            sig = 0.5 * (sig + sig.T)
            sig.flat[:: k_b + 1] += b_v

            ev, Q = linalg.eigh(sig, check_finite=False)
            qu = np.sum(Q * u_b[:, None], axis=0)

            if np.all(ev >= eps_eig):
                inv_ev = 1.0 / ev
                total_nll += 0.5 * float(np.sum(np.log(ev) + (qu * qu) * inv_ev))
                diag_k = inv_ev - np.square(qu * inv_ev)
            else:
                inside = ev >= eps_eig
                ev_eff = np.where(
                    inside,
                    ev,
                    eps_eig * np.exp(np.clip((ev - eps_eig) / eps_eig, -30.0, 0.0)),
                )
                dev_eff = np.where(inside, 1.0, ev_eff / eps_eig)
                inv_ev = 1.0 / np.maximum(ev_eff, 1e-18)
                deficit = np.where(inside, 0.0, eps_eig - ev)
                total_nll += (
                    0.5 * float(np.sum(np.log(np.maximum(ev_eff, 1e-18)) + (qu * qu) * inv_ev))
                    + 200.0 * float(np.sum(deficit * deficit))
                )
                # d(nll)/d(ev_k) = 0.5 * (1/ev_eff - qu^2 / ev_eff^2) * dev_eff - 400 * deficit
                diag_k = (inv_ev - np.square(qu * inv_ev)) * dev_eff - 800.0 * deficit

            K_mat = (Q * diag_k[None, :]) @ Q.T
            grad_gamma += 0.5 * np.sum(W_b * K_mat[None, :, :], axis=(1, 2))
            if est_intercept:
                grad_b += 0.5 * float(np.sum(diag_k))

        grad_out = (
            np.concatenate([grad_gamma, [grad_b]]) if est_intercept else grad_gamma
        )
        return total_nll, grad_out

    opt = optimize.minimize(
        nll_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 250},
    )
    gamma_hat = np.asarray(opt.x[:n_annot], dtype=np.float64)
    b_hat = float(opt.x[n_annot]) if est_intercept else 1.0
    tau_hat = gamma_hat / float(N)

    # Exact total log-likelihood & expected Fisher Information Matrix
    dim_f = n_annot + (1 if est_intercept else 0)
    fim = np.zeros((dim_f, dim_f), dtype=np.float64)
    const_ll = -0.5 * sum(float(np.sum(np.log(2.0 * np.pi * lb))) for lb in lam_blocks)
    nll_val = 0.0

    for b_idx in range(B):
        u_b = u_tilde_blocks[b_idx]
        k_b = u_b.shape[0]
        W_b = W_blocks[b_idx]
        sig = np.sum(W_b * gamma_hat[:, None, None], axis=0)
        sig = 0.5 * (sig + sig.T)
        sig.flat[:: k_b + 1] += b_hat
        ev, Q = linalg.eigh(sig, check_finite=False)
        ev_safe = np.maximum(ev, 1e-8)
        qu = np.sum(Q * u_b[:, None], axis=0)
        nll_val += 0.5 * float(np.sum(np.log(ev_safe) + (qu * qu) / ev_safe))

        sig_inv = (Q * (1.0 / ev_safe)[None, :]) @ Q.T
        M_list = np.empty_like(W_b)
        for a in range(n_annot):
            M_list[a] = sig_inv @ W_b[a]
        for a in range(n_annot):
            for b_a in range(a, n_annot):
                val_ab = 0.5 * (float(N) ** 2) * float(np.sum(M_list[a] * M_list[b_a].T))
                fim[a, b_a] += val_ab
                if b_a != a:
                    fim[b_a, a] += val_ab
        if est_intercept:
            for a in range(n_annot):
                cb = 0.5 * float(N) * float(np.sum(M_list[a] * sig_inv))
                fim[a, n_annot] += cb
                fim[n_annot, a] += cb
            fim[n_annot, n_annot] += 0.5 * float(np.sum(sig_inv * sig_inv))

    loglik = const_ll - nll_val
    cov_full = linalg.pinv(fim)
    cov_tau = cov_full[:n_annot, :n_annot]
    return tau_hat, b_hat, loglik, cov_tau


def hdl_stratified(
    z: np.ndarray | Sequence[np.ndarray] | Sequence[float],
    R_blocks: np.ndarray | Sequence[np.ndarray],
    annotations_blocks: np.ndarray | pd.DataFrame | Sequence[np.ndarray | pd.DataFrame],
    N: int | float,
    eigen_cut: float = 0.99,
    jackknife: bool = False,
    annotation_names: Sequence[str] | None = None,
    settings: SettingsType = "default",
) -> HDLStratifiedResult:
    """Stratified High-Definition Likelihood (HDL.S) partitioned heritability & enrichment.

    Models per-block Z-score covariance as
    ``Cov(z_b) = N * sum_a tau_a R_b D_{b,a} R_b + b_0 R_b``, maximizing the
    exact multivariate Gaussian likelihood projected onto the retained LD
    eigenvectors of each block.
    """
    validate_settings(settings)
    if N <= 2 or not np.isfinite(N):
        raise ValueError(f"Sample size N must be finite and > 2, got {N}.")

    raw_R = _normalize_r_blocks(R_blocks)
    panel = build_ld_eigen_panel(raw_R, eigen_cut=eigen_cut, settings=settings)
    z_blocks = split_block_vectors(z, panel, name="z")
    block_sizes = [blk.n_snps for blk in panel.blocks]
    annot_blocks, df_names = _normalize_annotation_blocks(annotations_blocks, block_sizes)

    n_annot = annot_blocks[0].shape[1]
    if annotation_names is not None:
        if len(annotation_names) != n_annot:
            raise ValueError(
                f"annotation_names has length {len(annotation_names)}, expected {n_annot}."
            )
        names = tuple(str(c) for c in annotation_names)
    elif df_names is not None:
        names = tuple(df_names)
    else:
        names = tuple(f"annot_{a + 1}" for a in range(n_annot))

    # Precompute reduced-eigenspace projected vectors u_tilde_b and W_{b,a} = Lam^{1/2} V^T D_a V Lam^{1/2}
    u_tilde_blocks: list[np.ndarray] = []
    lam_blocks: list[np.ndarray] = []
    W_blocks: list[np.ndarray] = []
    strat_ld_scores: list[np.ndarray] = []

    for b_idx, (zb, Rb, Ab, blk) in enumerate(
        zip(z_blocks, raw_R, annot_blocks, panel.blocks)
    ):
        _ = b_idx
        V = blk.eigvecs  # (M_b, k_b)
        lam = blk.eigvals  # (k_b,)
        sqrt_lam = np.sqrt(lam)
        u_tilde = (V.T @ zb) / sqrt_lam
        u_tilde_blocks.append(u_tilde)
        lam_blocks.append(lam)

        # W_b[a] = (V * sqrt_lam)^T @ diag(Ab[:, a]) @ (V * sqrt_lam)
        V_scaled = V * sqrt_lam[None, :]  # (M_b, k_b)
        W_b = np.einsum("ma,mi,mj->aij", Ab, V_scaled, V_scaled)
        W_blocks.append(W_b)

        R_sym = 0.5 * (Rb + Rb.T)
        strat_ld_scores.append((R_sym * R_sym) @ Ab)  # (M_b, A)

    # Closed-form S-LDSC moment initializer: E[z^2] = N * L_strat @ tau + b0
    z_all = np.concatenate(z_blocks)
    L_all = np.vstack(strat_ld_scores)
    est_intercept = settings == "default"
    if est_intercept:
        X_ols = np.column_stack([float(N) * L_all, np.ones(panel.total_snps, dtype=np.float64)])
        coef_init, *_ = linalg.lstsq(X_ols, z_all**2, check_finite=False)
        tau_0 = np.clip(coef_init[:n_annot], -1e-4, 1e-1)
        b0_0 = float(np.clip(coef_init[n_annot], 0.5, 2.5))
    else:
        coef_init, *_ = linalg.lstsq(float(N) * L_all, (z_all**2) - 1.0, check_finite=False)
        tau_0 = np.clip(coef_init[:n_annot], -1e-4, 1e-1)
        b0_0 = 1.0

    tau_hat, b_hat, loglik, cov_tau_fisher = _fit_stratified_blocks(
        u_tilde_blocks, lam_blocks, W_blocks, float(N), est_intercept, tau_0, b0_0
    )

    B = panel.n_blocks
    if jackknife and B >= 2:
        jk_tau = np.empty((B, n_annot), dtype=np.float64)
        for b_idx in range(B):
            u_sub = [u_tilde_blocks[j] for j in range(B) if j != b_idx]
            l_sub = [lam_blocks[j] for j in range(B) if j != b_idx]
            w_sub = [W_blocks[j] for j in range(B) if j != b_idx]
            jk_tau[b_idx], *_ = _fit_stratified_blocks(
                u_sub, l_sub, w_sub, float(N), est_intercept, tau_hat, b_hat
            )
        centered = jk_tau - np.mean(jk_tau, axis=0, keepdims=True)
        cov_tau = ((B - 1.0) / float(B)) * (centered.T @ centered)
        # Stabilize diagonal if near-zero
        diag_min = np.maximum(np.diag(cov_tau), 1e-24)
        np.fill_diagonal(cov_tau, diag_min)
    else:
        cov_tau = cov_tau_fisher

    se_tau = np.sqrt(np.maximum(np.diag(cov_tau), 1e-24))
    p_tau = 2.0 * stats.norm.sf(np.abs(tau_hat) / se_tau)

    # Partitioned heritability and enrichment via exact multivariate delta method
    A_all = np.vstack(annot_blocks)  # (M, A)
    M_total = float(panel.total_snps)
    c_tot = np.sum(A_all, axis=0)  # (A,)
    prop_snps = c_tot / M_total
    C_mat = A_all.T @ A_all  # (A, A); column a is c_a = A_all^T A_{.,a}

    h2_total = float(c_tot @ tau_hat)
    se_h2_total = float(np.sqrt(max(float(c_tot @ cov_tau @ c_tot), 1e-24)))
    h2_annot = C_mat.T @ tau_hat  # h_a^2 = c_a^T tau

    denom_h2 = h2_total if abs(h2_total) > 1e-12 else (1e-12 if h2_total >= 0 else -1e-12)
    prop_h2 = h2_annot / denom_h2

    enrichment = np.empty(n_annot, dtype=np.float64)
    se_enrichment = np.empty(n_annot, dtype=np.float64)
    p_enrichment = np.empty(n_annot, dtype=np.float64)

    for a in range(n_annot):
        p_a = float(prop_snps[a])
        if p_a <= 1e-12:
            enrichment[a] = np.nan
            se_enrichment[a] = np.nan
            p_enrichment[a] = np.nan
            continue
        c_a = C_mat[:, a]
        enr_a = float(prop_h2[a] / p_a)
        enrichment[a] = enr_a
        grad_a = (c_a - float(prop_h2[a]) * c_tot) / (p_a * denom_h2)
        var_enr = float(grad_a @ cov_tau @ grad_a)
        se_a = float(np.sqrt(max(var_enr, 0.0)))
        se_enrichment[a] = se_a
        p_enrichment[a] = (
            float(2.0 * stats.norm.sf(abs(enr_a - 1.0) / se_a)) if se_a > 1e-15 else 1.0
        )

    return HDLStratifiedResult(
        tau=tau_hat,
        se_tau=se_tau,
        p_tau=p_tau,
        cov_tau=cov_tau,
        h2_total=h2_total,
        se_h2_total=se_h2_total,
        h2_annotations=h2_annot,
        prop_snps=prop_snps,
        prop_h2=prop_h2,
        enrichment=enrichment,
        se_enrichment=se_enrichment,
        p_enrichment=p_enrichment,
        intercept=b_hat,
        loglik=loglik,
        n_snps=panel.total_snps,
        n_blocks=B,
        n_eigenmodes=panel.total_eigenmodes,
        annotation_names=names,
        settings=settings,
    )
