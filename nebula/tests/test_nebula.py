from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, sparse
from scipy.optimize import approx_fprime
from scipy.special import gammaln
from scipy.stats import kstest

from nebula import GroupedCellData, GroupedData, NebulaResult, group_cell, nebula
from nebula.hl import _hl_nll_and_grad
from nebula.ln import _ln_nll_and_grad, _pmm_nll_and_grad


def _simulate_nbgmm(
    rng: np.random.Generator,
    *,
    m: int,
    n_per_subj: int,
    beta_true: np.ndarray,
    sigma2_true: np.ndarray,
    phi_true: np.ndarray,
    subj_cov: np.ndarray | None = None,
    cell_cov: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    gaussian_re: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate G genes x N cells under NBGMM / PMM / NBLMM."""
    n_genes = beta_true.shape[0]
    n_cells = m * n_per_subj
    subj = np.repeat(np.arange(m), n_per_subj)

    if subj_cov is None:
        subj_level = rng.integers(0, 2, size=m).astype(np.float64)
        subj_cov = subj_level[subj]
    if cell_cov is None:
        cell_cov = rng.normal(0.0, 1.0, size=n_cells)
    if offset is None:
        offset = np.ones(n_cells, dtype=np.float64)

    X = np.column_stack([np.ones(n_cells, dtype=np.float64), subj_cov, cell_cov])
    counts = np.zeros((n_genes, n_cells), dtype=np.int64)

    for g in range(n_genes):
        s2 = float(sigma2_true[g])
        ph = float(phi_true[g])
        if s2 <= 0.0:
            u_subj = np.ones(m, dtype=np.float64)
        elif gaussian_re:
            u_subj = np.exp(rng.normal(0.0, np.sqrt(s2), size=m))
        else:
            alpha_s = 1.0 / s2
            u_subj = rng.gamma(shape=alpha_s, scale=1.0 / alpha_s, size=m)

        if ph <= 0.0:
            nu_cell = np.ones(n_cells, dtype=np.float64)
        else:
            alpha_c = 1.0 / ph
            nu_cell = rng.gamma(shape=alpha_c, scale=1.0 / alpha_c, size=n_cells)

        eta = X @ beta_true[g] + np.log(offset)
        lam = np.exp(eta) * u_subj[subj] * nu_cell
        counts[g] = rng.poisson(lam)

    return counts, subj, X, offset


def test_planted_truth_recovery_ln_large_cps() -> None:
    rng = np.random.default_rng(202501)
    m, n_i = 60, 70
    n_cells = m * n_i
    subj_base = rng.normal(0.0, 1.0, size=m)
    subj_cov = subj_base[np.repeat(np.arange(m), n_i)]
    beta_true = np.array(
        [
            [-1.0, 0.8, -0.5],
            [-0.8, -0.7, 0.6],
            [-0.5, 0.0, 0.0],
            [-1.2, 0.9, 0.4],
            [-0.7, 0.0, -0.6],
            [-1.1, -0.8, 0.0],
        ],
        dtype=np.float64,
    )
    sigma2_true = np.full(6, 0.4)
    phi_true = np.full(6, 0.6)
    offset = rng.uniform(0.5, 2.0, size=n_cells)

    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=sigma2_true,
        phi_true=phi_true,
        subj_cov=subj_cov,
        offset=offset,
    )

    pred_df = pd.DataFrame({"subj_var": X[:, 1], "cell_var": X[:, 2]})
    res = nebula(
        counts,
        id=subj,
        pred=pred_df,
        offset=offset,
        method="auto",
        cpc=0.005,
        mincp=5,
    )

    assert isinstance(res, NebulaResult)
    assert res.algorithm == ["LN"] * 6
    assert np.all(res.converged)
    np.testing.assert_allclose(res.log_fc[:, 1], beta_true[:, 1], atol=0.22)
    np.testing.assert_allclose(res.log_fc[:, 2], beta_true[:, 2], atol=0.12)
    np.testing.assert_allclose(res.overdispersion_subject, sigma2_true, atol=0.20)
    np.testing.assert_allclose(res.overdispersion_cell, phi_true, atol=0.18)

    # Planted non-zero effects should be strongly significant
    assert np.all(res.q_value[[0, 1, 3, 5], 1] < 0.01)
    assert np.all(res.q_value[[0, 1, 3, 4], 2] < 0.01)


def test_small_kappa_auto_hl_prevents_bias_and_type1_inflation() -> None:
    rng = np.random.default_rng(202502)
    # Part A: parameter recovery at small kappa = 8 < 20
    m, n_i = 40, 8
    beta_true = np.tile(np.array([-0.5, 0.6, -0.5]), (6, 1))
    sigma2_true = np.full(6, 0.6)
    phi_true = np.full(6, 0.8)
    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=sigma2_true,
        phi_true=phi_true,
    )
    pred_df = pd.DataFrame({"subj_var": X[:, 1], "cell_var": X[:, 2]})

    res_auto = nebula(counts, id=subj, pred=pred_df, offset=offset, method="auto")
    res_ln = nebula(counts, id=subj, pred=pred_df, offset=offset, method="LN")

    assert res_auto.mean_cells_per_subject == pytest.approx(8.0)
    assert res_auto.algorithm == ["HL"] * 6
    assert res_ln.algorithm == ["LN"] * 6
    assert np.all(res_auto.converged)

    # HL recovers phi accurately without LN's small-kappa downward bias
    err_phi_hl = np.mean(np.abs(res_auto.overdispersion_cell - 0.8))
    err_phi_ln = np.mean(np.abs(res_ln.overdispersion_cell - 0.8))
    assert err_phi_hl < 0.25
    assert err_phi_hl <= err_phi_ln + 0.05
    np.testing.assert_allclose(np.mean(res_auto.log_fc, axis=0), [-0.5, 0.6, -0.5], atol=0.18)

    # Part B: small kappa = 5 under H_0 (subject-level confounding): HL avoids Type I inflation
    m_null, ni_null, g_null = 50, 5, 80
    beta_null = np.tile(np.array([0.5, 0.0, 0.0]), (g_null, 1))
    counts_n, subj_n, X_n, off_n = _simulate_nbgmm(
        rng,
        m=m_null,
        n_per_subj=ni_null,
        beta_true=beta_null,
        sigma2_true=np.full(g_null, 1.0),
        phi_true=np.full(g_null, 0.5),
    )
    pred_n = pd.DataFrame({"subj_var": X_n[:, 1], "cell_var": X_n[:, 2]})
    res_n_hl = nebula(counts_n, id=subj_n, pred=pred_n, offset=off_n, method="auto")
    assert res_n_hl.algorithm == ["HL"] * g_null
    fpr_subj_hl = float(np.mean(res_n_hl.p_value[:, 1] < 0.05))
    fpr_cell_hl = float(np.mean(res_n_hl.p_value[:, 2] < 0.05))
    assert fpr_subj_hl <= 0.10
    assert fpr_cell_hl <= 0.10


def test_cutoff_cell_hybrid_refit() -> None:
    rng = np.random.default_rng(202503)
    m, n_i = 35, 25  # kappa_cps = 25 >= 20
    beta_true = np.array([[0.2, 0.4, -0.3], [0.2, 0.4, -0.3]], dtype=np.float64)
    # Gene 0: phi = 0.2 -> kappa / phi = 125 >= 20 (stays LN)
    # Gene 1: phi = 5.0 -> kappa / phi = 5 < 20 (triggers HL refit)
    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.array([0.4, 0.4]),
        phi_true=np.array([0.2, 5.0]),
    )
    pred_df = pd.DataFrame({"subj_var": X[:, 1], "cell_var": X[:, 2]})
    res = nebula(counts, id=subj, pred=pred_df, offset=offset, method="auto", cutoff_cell=20.0)

    assert res.algorithm == ["LN", "HL"]
    assert np.all(res.converged)


def test_analytical_gradients_ln_pmm_hl_nblmm() -> None:
    rng = np.random.default_rng(202504)
    m, n_i, p = 8, 9, 3
    n_cells = m * n_i
    subj = np.repeat(np.arange(m), n_i)
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, p - 1))])
    log_offset = np.log(rng.uniform(0.5, 2.0, size=n_cells))
    y = rng.poisson(2.2, size=n_cells).astype(np.float64)

    cell_idx = np.flatnonzero(y)
    y_nz = y[cell_idx]
    subj_nz = subj[cell_idx]
    Y_subj = np.bincount(subj_nz, weights=y_nz, minlength=m)
    Y_nz = Y_subj[Y_subj > 0]
    const_y_fact = float(np.sum(gammaln(y_nz + 1.0)))

    for theta in [
        np.array([-0.4, 0.3, -0.2, np.log(2.5), np.log(1.8)]),
        np.array([-0.4, 0.3, -0.2, np.log(1e5), np.log(1e5)]),
    ]:
        # NEBULA-LN
        _, g_ln = _ln_nll_and_grad(
            theta, X, log_offset, subj, m, y_nz, cell_idx, subj_nz, Y_subj, Y_nz, const_y_fact
        )
        fd_ln = approx_fprime(
            theta,
            lambda t: _ln_nll_and_grad(
                t, X, log_offset, subj, m, y_nz, cell_idx, subj_nz, Y_subj, Y_nz, const_y_fact
            )[0],
            1e-7,
        )
        np.testing.assert_allclose(g_ln, fd_ln, rtol=1e-5, atol=1e-5)

        # NEBULA-HL NBGMM order 2 (Gamma-saddlepoint)
        _, g_hl2 = _hl_nll_and_grad(
            theta,
            X,
            log_offset,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            Y_nz,
            const_y_fact,
            gaussian_prior=False,
            laplace_order=2,
        )
        fd_hl2 = approx_fprime(
            theta,
            lambda t: _hl_nll_and_grad(
                t,
                X,
                log_offset,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y_fact,
                gaussian_prior=False,
                laplace_order=2,
            )[0],
            1e-7,
        )
        np.testing.assert_allclose(g_hl2, fd_hl2, rtol=1e-5, atol=1e-5)

        # NEBULA-HL NBLMM (Gaussian prior)
        _, g_nblmm = _hl_nll_and_grad(
            theta,
            X,
            log_offset,
            subj,
            m,
            y_nz,
            cell_idx,
            subj_nz,
            Y_subj,
            Y_nz,
            const_y_fact,
            gaussian_prior=True,
            laplace_order=1,
        )
        fd_nblmm = approx_fprime(
            theta,
            lambda t: _hl_nll_and_grad(
                t,
                X,
                log_offset,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y_fact,
                gaussian_prior=True,
                laplace_order=1,
            )[0],
            1e-7,
        )
        np.testing.assert_allclose(g_nblmm, fd_nblmm, rtol=1e-5, atol=1e-5)

    # PMM
    theta_pmm = np.array([-0.4, 0.3, -0.2, np.log(2.5)])
    _, g_pmm = _pmm_nll_and_grad(
        theta_pmm, X, log_offset, subj, m, y_nz, cell_idx, Y_subj, Y_nz, const_y_fact
    )
    fd_pmm = approx_fprime(
        theta_pmm,
        lambda t: _pmm_nll_and_grad(
            t, X, log_offset, subj, m, y_nz, cell_idx, Y_subj, Y_nz, const_y_fact
        )[0],
        1e-7,
    )
    np.testing.assert_allclose(g_pmm, fd_pmm, rtol=1e-5, atol=1e-5)


def test_hl_matches_exact_1d_quadrature_loglik() -> None:
    rng = np.random.default_rng(202505)
    m, n_i = 8, 12
    beta = np.array([-0.3, 0.5, -0.4])
    sigma2, phi = 0.5, 0.8
    alpha_s, alpha_c = 1.0 / sigma2, 1.0 / phi

    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta[None, :],
        sigma2_true=np.array([sigma2]),
        phi_true=np.array([phi]),
    )
    y = counts[0].astype(np.float64)
    log_offset = np.log(offset)
    eta_lin = X @ beta + log_offset
    mu = np.exp(eta_lin)

    # Exact 1D quadrature per subject over u_i in (0, inf) via omega = log(u)
    ll_quad = 0.0
    for i in range(m):
        idx = subj == i
        yi = y[idx]
        mui = mu[idx]
        etai = eta_lin[idx]
        const_i = float(
            np.sum(gammaln(yi + alpha_c) - gammaln(alpha_c) - gammaln(yi + 1.0) + yi * etai)
            + yi.size * alpha_c * np.log(alpha_c)
            + alpha_s * np.log(alpha_s)
            - gammaln(alpha_s)
        )

        def integrand(omega: float) -> float:
            u = np.exp(omega)
            log_val = (
                (float(np.sum(yi)) + alpha_s) * omega
                - alpha_s * u
                - float(np.sum((yi + alpha_c) * np.log(alpha_c + u * mui)))
            )
            return float(np.exp(log_val))

        val, _ = integrate.quad(integrand, -15.0, 15.0, epsabs=1e-12, epsrel=1e-10)
        ll_quad += const_i + np.log(val)

    cell_idx = np.flatnonzero(y)
    y_nz = y[cell_idx]
    subj_nz = subj[cell_idx]
    Y_subj = np.bincount(subj_nz, weights=y_nz, minlength=m)
    Y_nz = Y_subj[Y_subj > 0]
    const_y_fact = float(np.sum(gammaln(y_nz + 1.0)))
    theta = np.array([beta[0], beta[1], beta[2], np.log(alpha_s), np.log(alpha_c)])

    nll_hl, _ = _hl_nll_and_grad(
        theta,
        X,
        log_offset,
        subj,
        m,
        y_nz,
        cell_idx,
        subj_nz,
        Y_subj,
        Y_nz,
        const_y_fact,
        gaussian_prior=False,
        laplace_order=2,
    )
    ll_hl = -nll_hl
    assert abs(ll_hl - ll_quad) < 0.05
    assert abs(ll_hl - ll_quad) / abs(ll_quad) < 1e-3


def test_null_calibration_and_no_pseudoreplication() -> None:
    rng = np.random.default_rng(202506)
    m, n_i, n_genes = 80, 30, 80
    beta_true = np.tile(np.array([0.2, 0.0, 0.0]), (n_genes, 1))
    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(n_genes, 0.8),
        phi_true=np.full(n_genes, 0.5),
    )
    pred_df = pd.DataFrame({"subj_var": X[:, 1], "cell_var": X[:, 2]})
    res = nebula(counts, id=subj, pred=pred_df, offset=offset, method="auto")

    assert np.all(res.converged)
    p_subj = res.p_value[:, 1]
    p_cell = res.p_value[:, 2]

    # No pseudoreplication inflation on either subject-level or cell-level covariate
    assert float(np.mean(p_subj < 0.05)) <= 0.09
    assert float(np.mean(p_cell < 0.05)) <= 0.09
    assert kstest(p_subj, "uniform").pvalue > 0.01
    assert kstest(p_cell, "uniform").pvalue > 0.01


def test_group_cell_and_permutation_invariance() -> None:
    rng = np.random.default_rng(202507)
    m, n_i, n_genes = 20, 25, 4
    n_cells = m * n_i
    beta_true = np.tile(np.array([-0.4, 0.5, -0.4]), (n_genes, 1))
    counts, subj_int, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(n_genes, 0.4),
        phi_true=np.full(n_genes, 0.5),
        offset=rng.uniform(0.7, 1.5, size=n_cells),
    )
    subj_str = np.array([f"donor_{i:02d}" for i in subj_int], dtype=object)

    # Shuffle cells across subjects
    perm = rng.permutation(n_cells)
    counts_shuf = counts[:, perm]
    subj_shuf = subj_str[perm]
    X_shuf = X[perm, 1:]
    offset_shuf = offset[perm]

    assert GroupedData is GroupedCellData
    grp1 = group_cell(counts_shuf, subj_shuf, pred=X_shuf, offset=offset_shuf)
    assert grp1.was_already_grouped is False
    grp_df = grp1.to_dataframe()
    assert {"cell_index", "original_index", "subject_id", "subject_code", "offset"}.issubset(
        grp_df.columns
    )

    grp2 = group_cell(grp1.count, grp1.id, pred=grp1.pred, offset=grp1.offset)
    assert grp2.was_already_grouped is True

    res_orig = nebula(counts, id=subj_str, pred=X[:, 1:], offset=offset)
    res_shuf = nebula(counts_shuf, id=subj_shuf, pred=X_shuf, offset=offset_shuf)

    np.testing.assert_allclose(res_shuf.log_fc, res_orig.log_fc, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res_shuf.se, res_orig.se, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res_shuf.p_value, res_orig.p_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        res_shuf.overdispersion_subject, res_orig.overdispersion_subject, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        res_shuf.overdispersion_cell, res_orig.overdispersion_cell, rtol=1e-5, atol=1e-5
    )


def test_anndata_sparse_csr_csc_and_dense_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(202508)
    m, n_i, n_genes = 20, 30, 5
    n_cells = m * n_i
    beta_true = np.tile(np.array([-0.6, 0.7, -0.4]), (n_genes, 1))
    counts_gxN, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(n_genes, 0.35),
        phi_true=np.full(n_genes, 0.45),
        offset=rng.uniform(0.6, 1.8, size=n_cells),
    )

    gene_names = [f"G{g}" for g in range(n_genes)]
    obs_df = pd.DataFrame(
        {
            "donor": [f"D{i}" for i in subj],
            "cond": X[:, 1],
            "cell_cov": X[:, 2],
            "size_factor": offset,
        },
        index=[f"cell_{k}" for k in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts_gxN.T.astype(np.float64)),
        obs=obs_df,
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["counts"] = sparse.csc_matrix(counts_gxN.T.astype(np.float64))

    # Ensure sparse execution never calls .toarray() on the full count matrix
    def _forbid_toarray(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Full sparse matrix densification (.toarray) is forbidden!")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", _forbid_toarray)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", _forbid_toarray)

    res_dense = nebula(
        counts_gxN,
        id=obs_df["donor"],
        pred=obs_df[["cond", "cell_cov"]],
        offset=offset,
        covariance=True,
        output_re=True,
    )
    res_csr_gxn = nebula(
        sparse.csr_matrix(counts_gxN.astype(np.float64)),
        id=obs_df["donor"],
        pred=obs_df[["cond", "cell_cov"]],
        offset=offset,
    )
    res_adata_layer = nebula(
        adata,
        id="donor",
        pred="~ cond + cell_cov",
        offset="size_factor",
        layer="counts",
        covariance=True,
        output_re=True,
    )

    np.testing.assert_allclose(res_csr_gxn.log_fc, res_dense.log_fc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_adata_layer.log_fc, res_dense.log_fc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_adata_layer.se, res_dense.se, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_adata_layer.p_value, res_dense.p_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        res_adata_layer.overdispersion_subject, res_dense.overdispersion_subject, rtol=1e-10
    )
    np.testing.assert_allclose(
        res_adata_layer.overdispersion_cell, res_dense.overdispersion_cell, rtol=1e-10
    )

    assert res_adata_layer.covariance_matrices is not None
    assert res_adata_layer.covariance_matrices.shape == (n_genes, 3, 3)
    assert res_adata_layer.random_effects is not None
    assert res_adata_layer.random_effects.shape == (n_genes, m)

    df = res_adata_layer.to_dataframe(sort_by="p_cond")
    assert len(df) == n_genes
    assert "logFC_cond" in df.columns
    assert "overdispersion_subject" in df.columns
    od_df = res_adata_layer.overdispersion_dataframe()
    assert list(od_df.columns) == ["gene", "Subject", "Cell", "algorithm", "convergence", "converged"]
    sig_df = res_adata_layer.significant(predictor="cond", alpha=0.05)
    assert len(sig_df) >= 1
    assert "NEBULA Result" in res_adata_layer.summary()

    # Also test library_size offset on AnnData without error
    res_lib = nebula(adata, id="donor", pred=["cond", "cell_cov"], offset="library_size")
    assert np.all(res_lib.converged)


def test_poisson_boundary_sigma2_zero_and_phi_zero() -> None:
    rng = np.random.default_rng(202509)
    m, n_i = 25, 40
    beta_true = np.array(
        [
            [0.3, 0.5, -0.4],  # pure Poisson (sigma2=0, phi=0)
            [0.3, 0.5, -0.4],  # pure NB2 (sigma2=0, phi=0.5)
            [0.3, 0.5, -0.4],  # subject-only Poisson-Gamma (sigma2=0.4, phi=0)
        ],
        dtype=np.float64,
    )
    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.array([0.0, 0.0, 0.4]),
        phi_true=np.array([0.0, 0.5, 0.0]),
    )
    pred_df = pd.DataFrame({"subj_var": X[:, 1], "cell_var": X[:, 2]})
    res_ln = nebula(counts, id=subj, pred=pred_df, offset=offset, method="LN")
    res_hl = nebula(counts, id=subj, pred=pred_df, offset=offset, method="HL")

    for res in (res_ln, res_hl):
        assert np.all(np.isfinite(res.log_fc))
        assert np.all(np.isfinite(res.se))
        assert np.all(res.se > 0.0)
        assert np.all(res.converged)
        assert res.overdispersion_subject[0] < 0.05
        assert res.overdispersion_cell[0] < 0.05
        assert res.overdispersion_subject[1] < 0.05
        assert 0.25 < res.overdispersion_cell[1] < 0.85
        assert 0.15 < res.overdispersion_subject[2] < 0.80
        assert res.overdispersion_cell[2] < 0.05


def test_offset_proportionality_invariance() -> None:
    rng = np.random.default_rng(202510)
    m, n_i = 20, 30
    beta_true = np.array([[0.1, 0.5, -0.4], [-0.2, -0.6, 0.3]], dtype=np.float64)
    counts, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.array([0.3, 0.5]),
        phi_true=np.array([0.4, 0.6]),
        offset=rng.uniform(0.5, 1.5, size=m * n_i),
    )
    c = 3.5
    res1 = nebula(counts, id=subj, pred=X[:, 1:], offset=offset)
    res2 = nebula(counts, id=subj, pred=X[:, 1:], offset=c * offset)

    np.testing.assert_allclose(res2.log_fc[:, 0], res1.log_fc[:, 0] - np.log(c), atol=1e-6)
    np.testing.assert_allclose(res2.log_fc[:, 1:], res1.log_fc[:, 1:], atol=1e-6)
    np.testing.assert_allclose(res2.se, res1.se, atol=1e-6)
    np.testing.assert_allclose(res2.p_value[:, 1:], res1.p_value[:, 1:], atol=1e-6)
    np.testing.assert_allclose(res2.overdispersion_subject, res1.overdispersion_subject, atol=1e-5)
    np.testing.assert_allclose(res2.overdispersion_cell, res1.overdispersion_cell, atol=1e-5)


def test_nan_aware_fdr_cpc_filtering_and_validation_errors() -> None:
    rng = np.random.default_rng(202511)
    m, n_i = 20, 30
    n_cells = m * n_i
    counts_valid, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=np.tile(np.array([-0.2, 0.6, -0.4]), (5, 1)),
        sigma2_true=np.full(5, 0.3),
        phi_true=np.full(5, 0.4),
    )
    # Append 3 genes to be filtered: all-zero, low cpc (<0.005), and < mincp nonzero cells
    g_zero = np.zeros((1, n_cells), dtype=np.int64)
    g_low_cpc = np.zeros((1, n_cells), dtype=np.int64)
    g_low_cpc[0, :2] = 1  # cpc = 2 / 600 = 0.00333 < 0.005
    g_low_mincp = np.zeros((1, n_cells), dtype=np.int64)
    g_low_mincp[0, :3] = 5  # nonzero = 3 < 5 even though cpc = 0.025 >= 0.005

    counts_all = np.vstack([counts_valid, g_zero, g_low_cpc, g_low_mincp])

    res_valid_only = nebula(counts_valid, id=subj, pred=X[:, 1:], offset=offset)
    res_with_filt = nebula(counts_all, id=subj, pred=X[:, 1:], offset=offset, cpc=0.005, mincp=5)

    assert list(res_with_filt.kept_mask) == [True] * 5 + [False] * 3
    assert res_with_filt.algorithm[5:] == ["filtered", "filtered", "filtered"]
    assert np.all(res_with_filt.convergence[5:] == 0)
    assert np.all(np.isnan(res_with_filt.p_value[5:]))
    assert np.all(np.isnan(res_with_filt.q_value[5:]))
    np.testing.assert_allclose(res_with_filt.q_value[:5], res_valid_only.q_value, rtol=1e-12)

    # All-filtered case returns valid NebulaResult and empty DataFrame with exact columns
    res_all_filt = nebula(
        np.zeros((2, n_cells), dtype=np.int64), id=subj, pred=X[:, 1:], offset=offset
    )
    empty_df = res_all_filt.to_dataframe(include_filtered=False)
    assert empty_df.empty
    assert list(empty_df.columns) == list(res_with_filt.to_dataframe().columns)

    # Degenerate subject structure warnings
    with warnings.catch_warnings(record=True) as w_single_cell:
        warnings.simplefilter("always")
        nebula(counts_valid[:, :20], id=np.arange(20), pred=X[:20, 1:2], cpc=0.0, mincp=1)
        assert any("1 cell" in str(x.message) for x in w_single_cell)

    with warnings.catch_warnings(record=True) as w_single_subj:
        warnings.simplefilter("always")
        nebula(counts_valid[:, :30], id=np.zeros(30, dtype=int), pred=X[:30, 2:3], cpc=0.0, mincp=1)
        assert any("1 subject" in str(x.message) for x in w_single_subj)

    # Explicit ValueErrors for invalid inputs
    X_collinear = np.column_stack([X[:, 1], 2.0 * X[:, 1]])
    with pytest.raises(ValueError, match="rank-deficient"):
        nebula(counts_valid, id=subj, pred=X_collinear)
    with pytest.raises(ValueError, match="offset"):
        nebula(counts_valid, id=subj, pred=X[:, 1:], offset=-np.ones(n_cells))
    with pytest.raises(ValueError, match="negative"):
        bad = counts_valid.astype(np.float64)
        bad[0, 0] = -1.0
        nebula(bad, id=subj, pred=X[:, 1:])
    with pytest.raises(ValueError, match="integer"):
        bad2 = counts_valid.astype(np.float64)
        bad2[0, 0] = 1.5
        nebula(bad2, id=subj, pred=X[:, 1:])


def test_pmm_and_nblmm_models() -> None:
    rng = np.random.default_rng(202512)
    m, n_i = 45, 30
    subj_cov = rng.normal(0.0, 1.0, size=m)[np.repeat(np.arange(m), n_i)]
    beta_true = np.tile(np.array([-0.5, 0.6, -0.4]), (3, 1))

    counts_pmm, subj, X, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(3, 0.4),
        phi_true=np.zeros(3),
        subj_cov=subj_cov,
    )
    res_pmm = nebula(counts_pmm, id=subj, pred=X[:, 1:], offset=offset, model="PMM")
    assert res_pmm.algorithm == ["PMM"] * 3
    assert np.all(res_pmm.overdispersion_cell == 0.0)
    assert np.all(res_pmm.converged)
    np.testing.assert_allclose(res_pmm.log_fc[:, 1:], beta_true[:, 1:], atol=0.18)

    counts_nblmm, subj2, X2, off2 = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(3, 0.5),
        phi_true=np.full(3, 0.5),
        subj_cov=subj_cov,
        gaussian_re=True,
    )
    res_nblmm = nebula(counts_nblmm, id=subj2, pred=X2[:, 1:], offset=off2, model="NBLMM")
    assert res_nblmm.algorithm == ["NBLMM"] * 3
    assert np.all(res_nblmm.converged)
    np.testing.assert_allclose(res_nblmm.log_fc[:, 1:], beta_true[:, 1:], atol=0.22)
    np.testing.assert_allclose(res_nblmm.overdispersion_subject, 0.5, atol=0.25)
    np.testing.assert_allclose(res_nblmm.overdispersion_cell, 0.5, atol=0.20)


def test_multigene_multiseed_convergence_stress() -> None:
    # Part A: 30 random genes with unbalanced subjects and std=1.5 predictors
    # across NBGMM (LN & HL), PMM, and NBLMM (auto/HL & LN).
    rng = np.random.default_rng(12345)
    m = 50
    n_per_subj = rng.integers(28, 45, size=m)
    subj = np.repeat(np.arange(m), n_per_subj)
    n_cells = int(subj.size)
    subj_x = rng.normal(0.0, 1.5, size=m)[subj]
    cell_x = rng.normal(0.0, 1.5, size=n_cells)
    offset = np.exp(rng.normal(0.0, 0.35, size=n_cells))
    X = np.column_stack([np.ones(n_cells), subj_x, cell_x])

    n_genes = 30
    counts = np.zeros((n_genes, n_cells), dtype=np.int64)
    beta_true = np.zeros((n_genes, 3), dtype=np.float64)
    for g in range(n_genes):
        beta_g = np.array(
            [rng.uniform(-0.8, 0.4), rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)],
            dtype=np.float64,
        )
        beta_true[g] = beta_g
        s2 = float(rng.uniform(0.3, 0.9))
        ph = float(rng.uniform(0.3, 1.0))
        u = rng.gamma(1.0 / s2, s2, size=m)[subj]
        nu = rng.gamma(1.0 / ph, ph, size=n_cells)
        counts[g] = rng.poisson(np.exp(X @ beta_g) * offset * u * nu)

    pred_df = pd.DataFrame({"sx": subj_x, "cx": cell_x})
    for model_name, method_name in [
        ("NBGMM", "LN"),
        ("NBGMM", "HL"),
        ("PMM", "auto"),
        ("NBLMM", "auto"),
        ("NBLMM", "LN"),
    ]:
        res = nebula(
            counts,
            id=subj,
            pred=pred_df,
            offset=offset,
            model=model_name,
            method=method_name,
            cpc=0.0,
            mincp=1,
        )
        assert np.all(res.convergence == 1), f"{model_name}/{method_name}: {res.convergence}"
        assert np.all(res.converged)
        assert np.all(np.abs(res.log_fc) < 10.0)
        assert np.all(np.isfinite(res.se))
        assert np.all(res.se > 0.0)
        assert np.all(np.isfinite(res.p_value))
        assert np.all(np.isfinite(res.q_value))
        assert np.all(np.isfinite(res.loglik))
        assert np.all(res.loglik <= 0.0)

    # Part B: 20 independent per-gene seeds (including 1008, 1011, 1012, 1015)
    for seed in range(1000, 1020):
        rng_s = np.random.default_rng(seed)
        m_s, ni_s = 35, 30
        n_s = m_s * ni_s
        subj_s = np.repeat(np.arange(m_s), ni_s)
        x1 = rng_s.normal(size=m_s)[subj_s]
        x2 = rng_s.normal(size=n_s)
        X_s = np.column_stack([np.ones(n_s), x1, x2])
        beta_s = np.array([-0.3, 0.8, -0.5])
        u_s = rng_s.gamma(1.0 / 0.5, 0.5, size=m_s)[subj_s]
        nu_s = rng_s.gamma(1.0 / 0.7, 0.7, size=n_s)
        y_s = rng_s.poisson(np.exp(X_s @ beta_s) * u_s * nu_s)[None, :]

        for model_name, method_name in [
            ("NBGMM", "LN"),
            ("NBGMM", "HL"),
            ("PMM", "auto"),
            ("NBLMM", "auto"),
        ]:
            r = nebula(y_s, id=subj_s, pred=X_s[:, 1:], model=model_name, method=method_name)
            assert int(r.convergence[0]) == 1, f"seed={seed} {model_name}/{method_name}"
            assert bool(r.converged[0])
            assert np.all(np.abs(r.log_fc[0]) < 10.0)
            assert np.all(np.isfinite(r.se[0]))
            assert np.all(np.isfinite(r.p_value[0]))

    # Part C: Unstandardized multi-scale predictors + high counts
    rng_u = np.random.default_rng(98765)
    m_u, ni_u = 40, 25
    n_u = m_u * ni_u
    subj_u = np.repeat(np.arange(m_u), ni_u)
    age = rng_u.uniform(20.0, 80.0, size=m_u)[subj_u]
    x_tiny = rng_u.normal(0.0, 1e-4, size=n_u)
    x_huge = rng_u.normal(0.0, 1e4, size=m_u)[subj_u]
    X_u = np.column_stack([np.ones(n_u), age, x_tiny, x_huge])
    beta_u = np.array([5.5, 0.02, 4000.0, -5e-5])
    u_u = rng_u.gamma(2.5, 0.4, size=m_u)[subj_u]
    nu_u = rng_u.gamma(2.5, 0.4, size=n_u)
    y_u = rng_u.poisson(np.exp(X_u @ beta_u) * u_u * nu_u)[None, :]
    scales_u = np.array([1.0 / 30.0, 1e4, 1e-4])
    for model_name, method_name in [("NBGMM", "LN"), ("NBGMM", "HL"), ("PMM", "auto"), ("NBLMM", "auto")]:
        r_u = nebula(y_u, id=subj_u, pred=X_u[:, 1:], model=model_name, method=method_name)
        r_std = nebula(
            y_u, id=subj_u, pred=X_u[:, 1:] * scales_u[None, :], model=model_name, method=method_name
        )
        assert int(r_u.convergence[0]) == 1
        assert bool(r_u.converged[0])
        assert np.all(np.isfinite(r_u.se[0]))
        assert np.all(np.isfinite(r_u.p_value[0]))
        np.testing.assert_allclose(r_u.log_fc[0, 1], beta_u[1], atol=0.02)
        np.testing.assert_allclose(r_u.log_fc[0, 2], beta_u[2], rtol=0.15)
        np.testing.assert_allclose(r_u.log_fc[0, 3], beta_u[3], rtol=0.25)
        np.testing.assert_allclose(r_u.log_fc[0, 1:], r_std.log_fc[0, 1:] * scales_u, rtol=1e-4)
        np.testing.assert_allclose(r_u.se[0, 1:], r_std.se[0, 1:] * scales_u, rtol=1e-4)
        np.testing.assert_allclose(r_u.p_value[0], r_std.p_value[0], rtol=1e-4, atol=1e-6)


def test_extreme_eta_gradient_consistency() -> None:
    rng = np.random.default_rng(777)
    m, n_i = 15, 20
    n_cells = m * n_i
    subj = np.repeat(np.arange(m), n_i)
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    log_off = np.log(rng.uniform(0.5, 2.0, size=n_cells))
    y = rng.poisson(np.exp(X @ np.array([0.2, 0.5, -0.4]))).astype(np.float64)
    cell_idx = np.flatnonzero(y)
    y_nz = y[cell_idx]
    subj_nz = subj[cell_idx]
    Y_subj = np.bincount(subj_nz, weights=y_nz, minlength=m)
    Y_nz = Y_subj[Y_subj > 0]
    const_y = float(np.sum(gammaln(y_nz + 1.0)))

    def _central_fd(fn, th: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        fd = np.zeros_like(th)
        for k in range(th.size):
            t1 = th.copy()
            t2 = th.copy()
            t1[k] += eps
            t2[k] -= eps
            fd[k] = (fn(t1)[0] - fn(t2)[0]) / (2.0 * eps)
        return fd

    for th in [np.array([15.0, 15.0, 15.0, 0.0, 0.0]), np.array([-15.0, -15.0, -15.0, 0.0, 0.0])]:
        for fn in [
            lambda t: _ln_nll_and_grad(
                t, X, log_off, subj, m, y_nz, cell_idx, subj_nz, Y_subj, Y_nz, const_y
            ),
            lambda t: _hl_nll_and_grad(
                t,
                X,
                log_off,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y,
                gaussian_prior=False,
                laplace_order=2,
            ),
            lambda t: _hl_nll_and_grad(
                t,
                X,
                log_off,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y,
                gaussian_prior=False,
                laplace_order=1,
            ),
            lambda t: _hl_nll_and_grad(
                t,
                X,
                log_off,
                subj,
                m,
                y_nz,
                cell_idx,
                subj_nz,
                Y_subj,
                Y_nz,
                const_y,
                gaussian_prior=True,
                laplace_order=1,
            ),
        ]:
            nll, grad = fn(th)
            assert np.isfinite(nll) and nll > 0.0
            fd = _central_fd(fn, th)
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-4)

    for th_pmm in [np.array([15.0, 15.0, 15.0, 0.0]), np.array([-15.0, -15.0, -15.0, 0.0])]:
        fn_pmm = lambda t: _pmm_nll_and_grad(
            t, X, log_off, subj, m, y_nz, cell_idx, Y_subj, Y_nz, const_y
        )
        nll_pmm, grad_pmm = fn_pmm(th_pmm)
        assert np.isfinite(nll_pmm) and nll_pmm > 0.0
        fd_pmm = _central_fd(fn_pmm, th_pmm)
        np.testing.assert_allclose(grad_pmm, fd_pmm, rtol=1e-5, atol=1e-4)


def test_predictor_rescaling_invariance_and_null_fpr() -> None:
    rng = np.random.default_rng(202601)
    m, n_i, n_genes = 25, 30, 25
    n_cells = m * n_i
    subj = np.repeat(np.arange(m), n_i)
    z1 = rng.normal(0.0, 1.0, size=m)[subj]
    z2_null = rng.normal(0.0, 1.0, size=n_cells)
    beta_true = np.tile(np.array([-0.2, 0.4, 0.0]), (n_genes, 1))
    counts, _, _, offset = _simulate_nbgmm(
        rng,
        m=m,
        n_per_subj=n_i,
        beta_true=beta_true,
        sigma2_true=np.full(n_genes, 0.5),
        phi_true=np.full(n_genes, 0.5),
        subj_cov=z1,
        cell_cov=z2_null,
    )

    scale_pairs = [
        (1e-6, 1e6),
        (1e7, 1.0),
        (1e-4, 1e4),
    ]
    for model_name, method_name in [
        ("NBGMM", "LN"),
        ("NBGMM", "HL"),
        ("PMM", "auto"),
        ("NBLMM", "auto"),
    ]:
        res_ref = nebula(
            counts,
            id=subj,
            pred=pd.DataFrame({"z1": z1, "z2_null": z2_null}),
            offset=offset,
            model=model_name,
            method=method_name,
        )
        assert np.all(res_ref.converged)
        fpr_ref = float(np.mean(res_ref.p_value[:, 2] < 0.05))
        assert fpr_ref <= 0.20

        for s1, s2 in scale_pairs:
            res_sc = nebula(
                counts,
                id=subj,
                pred=pd.DataFrame({"z1": z1 * s1, "z2_null": z2_null * s2}),
                offset=offset,
                model=model_name,
                method=method_name,
            )
            assert res_sc.predictor_names == ["(Intercept)", "z1", "z2_null"]
            assert np.all(res_sc.converged)
            scales = np.array([1.0, s1, s2], dtype=np.float64)
            np.testing.assert_allclose(res_sc.log_fc * scales[None, :], res_ref.log_fc, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(res_sc.se * scales[None, :], res_ref.se, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(res_sc.z_stat, res_ref.z_stat, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(res_sc.p_value, res_ref.p_value, rtol=1e-5, atol=1e-10)
            assert float(np.mean(res_sc.p_value[:, 2] < 0.05)) == pytest.approx(fpr_ref)

    # Tiny 1e-9 non-constant predictors (both non-zero mean and zero-mean) must not be treated as intercept
    for z1_tiny, z1_unit in [
        ((2.0 + z1) * 1e-9, 2.0 + z1),
        ((z1 - float(np.mean(z1))) * 1e-9, z1 - float(np.mean(z1))),
    ]:
        r_ref_t = nebula(counts[:4], id=subj, pred=pd.DataFrame({"z1": z1_unit, "z2_null": z2_null}))
        r_tiny = nebula(counts[:4], id=subj, pred=pd.DataFrame({"z1": z1_tiny, "z2_null": z2_null}))
        assert r_tiny.predictor_names == ["(Intercept)", "z1", "z2_null"]
        np.testing.assert_allclose(r_tiny.log_fc[:, 1] * 1e-9, r_ref_t.log_fc[:, 1], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(r_tiny.se[:, 1] * 1e-9, r_ref_t.se[:, 1], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(r_tiny.p_value, r_ref_t.p_value, rtol=1e-5, atol=1e-10)

    # Non-unit constant intercept column (including NBLMM -0.5*s2/c_val shift)
    X_unit = np.column_stack([np.ones(n_cells), z1, z2_null])
    X_c5 = np.column_stack([np.full(n_cells, 5.0), z1, z2_null])
    for model_name, method_name in [
        ("NBGMM", "LN"),
        ("NBGMM", "HL"),
        ("NBLMM", "LN"),
        ("NBLMM", "auto"),
    ]:
        r_u = nebula(counts[:4], id=subj, pred=X_unit, model=model_name, method=method_name)
        r_5 = nebula(counts[:4], id=subj, pred=X_c5, model=model_name, method=method_name)
        np.testing.assert_allclose(r_5.log_fc[:, 0] * 5.0, r_u.log_fc[:, 0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(r_5.se[:, 0] * 5.0, r_u.se[:, 0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(r_5.p_value, r_u.p_value, rtol=1e-6, atol=1e-10)


def test_complete_separation_noncollapsed_se_and_pvalue() -> None:
    for m_s, ni_s, lam in [(20, 20, 5.0), (50, 40, 10.0)]:
        rng = np.random.default_rng(101)
        n_s = m_s * ni_s
        subj_s = np.repeat(np.arange(m_s), ni_s)
        grp = (subj_s >= m_s // 2).astype(np.float64)
        y_sep = np.zeros((1, n_s), dtype=np.int64)
        y_sep[0, subj_s < m_s // 2] = rng.poisson(lam, size=n_s // 2)

        for model_name, method_name in [
            ("NBGMM", "LN"),
            ("NBGMM", "HL"),
            ("PMM", "auto"),
            ("NBLMM", "auto"),
        ]:
            res = nebula(y_sep, id=subj_s, pred=grp, model=model_name, method=method_name)
            assert bool(res.converged[0])
            assert np.isfinite(res.se[0, 1]) and res.se[0, 1] > 10.0
            assert np.isfinite(res.p_value[0, 1]) and res.p_value[0, 1] > 0.5


def test_extreme_offsets_recovery_without_clamping() -> None:
    rng = np.random.default_rng(555)
    m, ni = 30, 30
    n_cells = m * ni
    subj = np.repeat(np.arange(m), ni)
    x1 = rng.normal(size=m)[subj]
    x2 = rng.normal(size=n_cells)
    X = np.column_stack([np.ones(n_cells), x1, x2])
    u = rng.gamma(2.0, 0.5, size=m)[subj]
    nu = rng.gamma(2.0, 0.5, size=n_cells)
    y = rng.poisson(np.exp(X @ np.array([0.0, 0.7, -0.5])) * u * nu)[None, :]

    for model_name, method_name in [
        ("NBGMM", "LN"),
        ("NBGMM", "HL"),
        ("PMM", "auto"),
        ("NBLMM", "auto"),
    ]:
        r_base = nebula(
            y, id=subj, pred=X[:, 1:], offset=np.ones(n_cells), model=model_name, method=method_name
        )
        for off_val in [1e-15, 1e15, 1e20]:
            r_ext = nebula(
                y,
                id=subj,
                pred=X[:, 1:],
                offset=np.full(n_cells, off_val),
                model=model_name,
                method=method_name,
            )
            assert int(r_ext.convergence[0]) == 1
            np.testing.assert_allclose(
                r_ext.log_fc[0, 0], r_base.log_fc[0, 0] - np.log(off_val), atol=1e-4
            )
            np.testing.assert_allclose(r_ext.log_fc[0, 1:], r_base.log_fc[0, 1:], atol=1e-4)
            np.testing.assert_allclose(r_ext.se[0], r_base.se[0], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(r_ext.p_value[0, 1:], r_base.p_value[0, 1:], rtol=1e-4, atol=1e-10)

