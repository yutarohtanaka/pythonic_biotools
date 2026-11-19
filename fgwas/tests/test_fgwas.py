"""Unit tests for fgwas (Pickrell, AJHG 2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fgwas import (
    cross_validate_fgwas,
    fit_fgwas,
    stepwise_fgwas,
    wakefield_lnabf,
)


def _simulate_fgwas_dataset(
    rng: np.random.Generator,
    n_blocks: int = 40,
    snps_per_block: int = 80,
    pi_region: float = 0.35,
    gamma_dnase: float = 2.2,
    gamma_coding: float = 1.6,
    fine_map_only: bool = False,
    prior_w: float = 0.1,
    add_ld_proxy: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate a synthetic GWAS summary DataFrame following the exact Wakefield/fgwas model."""
    rows: list[pd.DataFrame] = []
    causal_flags: list[np.ndarray] = []

    for b in range(n_blocks):
        dnase = (rng.uniform(size=snps_per_block) < 0.12).astype(float)
        coding = (rng.uniform(size=snps_per_block) < 0.08).astype(float)
        noise_ann = (rng.uniform(size=snps_per_block) < 0.25).astype(float)

        logits = gamma_dnase * dnase + gamma_coding * coding
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        se = np.full(snps_per_block, 0.04)
        is_causal_block = True if fine_map_only else (rng.uniform() < pi_region)
        causal_mask = np.zeros(snps_per_block, dtype=bool)
        beta = rng.normal(0.0, se)

        if is_causal_block:
            c_idx = int(rng.choice(snps_per_block, p=probs))
            causal_mask[c_idx] = True
            true_eff = rng.normal(0.0, np.sqrt(prior_w))
            if fine_map_only and abs(true_eff) < 0.18:
                true_eff = np.copysign(0.22, true_eff if true_eff != 0 else 1.0)
            beta[c_idx] = true_eff + rng.normal(0.0, se[c_idx])
            if add_ld_proxy:
                # Non-annotated tag SNP in high LD sharing nearly identical Z-score
                proxy_idx = (c_idx + 1) % snps_per_block
                dnase[proxy_idx] = 0.0
                coding[proxy_idx] = 0.0
                beta[proxy_idx] = beta[c_idx] * 0.995

        causal_flags.append(causal_mask)
        rows.append(
            pd.DataFrame(
                {
                    "chr": f"chr{(b % 4) + 1}",
                    "SEGNUMBER": b,
                    "beta": beta,
                    "se": se,
                    "dnase": dnase,
                    "coding": coding,
                    "noise_ann": noise_ann,
                }
            )
        )

    return pd.concat(rows, ignore_index=True), np.concatenate(causal_flags)


def test_wakefield_lnabf_formula_and_overflow_safety() -> None:
    beta = np.array([0.0, 0.25, -0.25, 4.0])
    se = np.array([0.05, 0.05, 0.05, 0.04])  # z = [0, 5, -5, 100]

    # Single prior variance exact check
    w = 0.1
    lnabf_single = wakefield_lnabf(beta, se, V=w, settings="default")
    v = se**2
    expected = 0.5 * np.log(v / (v + w)) + 0.5 * ((beta / se) ** 2) * (w / (v + w))
    assert np.allclose(lnabf_single, expected, rtol=1e-12, atol=1e-12)
    assert lnabf_single[0] < 0.0  # z = 0 favors H0 (shrinkage factor < 1)
    assert np.isclose(lnabf_single[1], lnabf_single[2])  # Symmetric in sign of beta
    assert np.isfinite(lnabf_single[3]) and lnabf_single[3] > 1000.0  # z=100 never overflows

    # Grid prior variance log-mean-exp check
    lnabf_grid = wakefield_lnabf(beta[:3], se[:3], V=(0.01, 0.1, 0.5), settings="R_default")
    abf_mean = np.mean(
        [np.exp(wakefield_lnabf(beta[:3], se[:3], V=wi)) for wi in (0.01, 0.1, 0.5)],
        axis=0,
    )
    assert np.allclose(lnabf_grid, np.log(abf_mean), atol=1e-12)

    # Scalar call
    val_scalar = wakefield_lnabf(0.25, 0.05, V=0.1)
    assert np.ndim(val_scalar) == 0 and np.isclose(float(val_scalar), lnabf_single[1])


def test_fit_fgwas_genome_wide_enrichment_and_posteriors() -> None:
    rng = np.random.default_rng(123)
    df, causal_mask = _simulate_fgwas_dataset(
        rng,
        n_blocks=120,
        snps_per_block=60,
        pi_region=0.35,
        gamma_dnase=2.3,
        gamma_coding=0.0,
        prior_w=0.1,
    )

    res = fit_fgwas(
        df,
        annotations=["dnase"],
        block_col="SEGNUMBER",
        fine_map=False,
        V=0.1,
        settings="default",
    )

    assert res.converged
    assert abs(res.pi_region - 0.35) < 0.12
    assert res.gamma[0] > 1.2
    assert res.ci_low[0] > 0.4
    assert res.ci_low[0] < res.gamma[0] < res.ci_high[0]
    assert np.all((res.PPA_region >= 0.0) & (res.PPA_region <= 1.0))
    # Sum of SNP PPAs within each block equals block's PPA_region
    block_sums = np.bincount(res.block_idx, weights=res.PPA_snp, minlength=res.n_blocks)
    assert np.allclose(block_sums, res.PPA_region, atol=1e-10)
    # Causal SNPs should have dramatically higher mean PPA than non-causal SNPs
    assert float(np.mean(res.PPA_snp[causal_mask])) > 10.0 * float(
        np.mean(res.PPA_snp[~causal_mask])
    )
    assert "fgwas" in res.summary()
    assert res.to_dataframe().shape == (1, 5)


def test_fit_fgwas_finemap_mode_and_r_default_parity() -> None:
    rng = np.random.default_rng(234)
    df, causal_mask = _simulate_fgwas_dataset(
        rng,
        n_blocks=35,
        snps_per_block=50,
        gamma_dnase=2.4,
        gamma_coding=1.7,
        fine_map_only=True,
        prior_w=0.1,
        add_ld_proxy=True,
    )

    res_def = fit_fgwas(
        df, annotations=["dnase", "coding"], block_col="SEGNUMBER", fine_map=True, settings="default"
    )
    res_r = fit_fgwas(
        df, annotations=["dnase", "coding"], fine_map=True, settings="R_default"
    )

    # Check settings defaults: default uses penalty=0.1 & V=(0.01,0.1,0.5); R_default uses penalty=0.0 & V=(0.1,)
    assert res_def.penalty == 0.1
    assert res_def.V == (0.01, 0.1, 0.5)
    assert res_r.penalty == 0.0
    assert res_r.V == (0.1,)
    assert res_def.pi_region == 1.0 and res_r.pi_region == 1.0

    # In fine-map mode, SNP PPAs within every locus sum to 1.0
    locus_sums = np.bincount(res_r.block_idx, weights=res_r.PPA_snp, minlength=res_r.n_blocks)
    assert np.allclose(locus_sums, 1.0, atol=1e-12)
    assert res_def.gamma[0] > 1.2 and res_def.gamma[1] > 0.7
    assert res_r.gamma[0] > 1.2 and res_r.gamma[1] > 0.7

    # Re-weighting boosts causal SNP PPA vs null (unannotated) fine-mapping
    res_null = fit_fgwas(df, annotations=[], fine_map=True, V=res_def.V, settings="default")
    assert float(np.mean(res_def.PPA_snp[causal_mask])) > float(
        np.mean(res_null.PPA_snp[causal_mask])
    )


def test_profile_ci_exact_drop_default_vs_r_default() -> None:
    rng = np.random.default_rng(345)
    df, _ = _simulate_fgwas_dataset(
        rng, n_blocks=20, snps_per_block=40, gamma_dnase=1.8, gamma_coding=0.0, fine_map_only=True
    )
    # Compare CIs at identical penalty=0.1 and V=(0.1,) so only CI drop (1.9207 vs 2.0) differs
    fit_d = fit_fgwas(
        df, ["dnase"], block_col="SEGNUMBER", fine_map=True, penalty=0.1, V=(0.1,), settings="default"
    )
    fit_r = fit_fgwas(
        df, ["dnase"], block_col="SEGNUMBER", fine_map=True, penalty=0.1, V=(0.1,), settings="R_default"
    )
    assert np.isclose(fit_d.gamma[0], fit_r.gamma[0], atol=1e-6)
    # R_default uses 2.0 llk drop (> 1.920729), so its CI must be slightly wider on both sides
    assert fit_r.ci_low[0] < fit_d.ci_low[0]
    assert fit_r.ci_high[0] > fit_d.ci_high[0]


def test_analytical_gradient_matches_finite_differences() -> None:
    from fgwas.core import _evaluate_fgwas_objective

    rng = np.random.default_rng(456)
    lnabf = rng.normal(-0.5, 2.5, size=120)
    A = rng.binomial(1, 0.25, size=(120, 3)).astype(float)
    block_idx = np.repeat(np.arange(6, dtype=np.int64), 20)

    eta = -0.8
    gamma = np.array([1.2, -0.7, 0.4])
    pen_llk, _, grad_eta, grad_g, *_ = _evaluate_fgwas_objective(
        lnabf, A, block_idx, 6, False, 0.15, eta, gamma
    )

    eps = 1e-6
    llk_ep, *_ = _evaluate_fgwas_objective(lnabf, A, block_idx, 6, False, 0.15, eta + eps, gamma)
    llk_em, *_ = _evaluate_fgwas_objective(lnabf, A, block_idx, 6, False, 0.15, eta - eps, gamma)
    assert np.isclose(grad_eta, (llk_ep - llk_em) / (2.0 * eps), atol=1e-6)

    for j in range(3):
        dg = np.zeros(3)
        dg[j] = eps
        lp, *_ = _evaluate_fgwas_objective(lnabf, A, block_idx, 6, False, 0.15, eta, gamma + dg)
        lm, *_ = _evaluate_fgwas_objective(lnabf, A, block_idx, 6, False, 0.15, eta, gamma - dg)
        assert np.isclose(grad_g[j], (lp - lm) / (2.0 * eps), atol=1e-6)
    _ = pen_llk


def test_window_partitioning_respects_chromosomes_and_window_size() -> None:
    rng = np.random.default_rng(567)
    # 70 SNPs on chr1 + 70 SNPs on chr2 with window_size=50 -> 2 blocks on chr1 + 2 blocks on chr2 = 4 blocks
    df = pd.DataFrame(
        {
            "chr": ["chr1"] * 70 + ["chr2"] * 70,
            "beta": rng.normal(0.0, 0.05, size=140),
            "se": np.full(140, 0.05),
            "ann": rng.binomial(1, 0.2, size=140).astype(float),
        }
    )
    res = fit_fgwas(df, ["ann"], window_size=50, compute_ci=False, settings="default")
    assert res.n_blocks == 4
    assert res.region_dataframe().shape == (4, 3)
    snp_df = res.snp_dataframe(df)
    assert "PPA_snp" in snp_df.columns and len(snp_df) == 140


def test_cross_validate_fgwas_prevents_separation_overfitting() -> None:
    rng = np.random.default_rng(678)
    # Sparse rare annotation with small block count induces separation at penalty=0.0
    df, _ = _simulate_fgwas_dataset(
        rng, n_blocks=20, snps_per_block=40, gamma_dnase=2.0, gamma_coding=0.0, fine_map_only=True
    )
    df["rare_sep"] = (rng.uniform(size=len(df)) < 0.02).astype(float)

    cv_def = cross_validate_fgwas(
        df,
        annotations=["dnase", "rare_sep"],
        penalties=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
        n_folds=5,
        block_col="SEGNUMBER",
        fine_map=True,
        settings="default",
    )
    cv_r = cross_validate_fgwas(
        df,
        annotations=["dnase", "rare_sep"],
        penalties=None,
        n_folds=5,
        block_col="SEGNUMBER",
        fine_map=True,
        settings="R_default",
    )

    assert cv_def.best_penalty in cv_def.penalties
    assert np.all(np.isfinite(cv_def.cv_logliks))
    assert cv_r.penalties[0] == 0.2  # R_default starts grid at p=0.2 when penalties=None
    assert cv_def.to_dataframe().shape == (6, 3)
    assert "fgwas CV" in cv_def.summary()


def test_stepwise_fgwas_selects_true_annotations_and_drops_noise() -> None:
    rng = np.random.default_rng(789)
    df, _ = _simulate_fgwas_dataset(
        rng, n_blocks=35, snps_per_block=50, gamma_dnase=2.5, gamma_coding=1.9, fine_map_only=True
    )

    step_def = stepwise_fgwas(
        df,
        candidate_annotations=["dnase", "coding", "noise_ann"],
        block_col="SEGNUMBER",
        fine_map=True,
        n_folds=7,
        settings="default",
    )
    step_r = stepwise_fgwas(
        df,
        candidate_annotations=["dnase", "coding", "noise_ann"],
        block_col="SEGNUMBER",
        fine_map=True,
        n_folds=7,
        settings="R_default",
    )

    assert "dnase" in step_def.selected_annotations
    assert "coding" in step_def.selected_annotations
    assert "noise_ann" not in step_def.selected_annotations
    assert "dnase" in step_r.selected_annotations
    assert step_def.history_dataframe().shape[0] >= 3
    assert "fgwas Stepwise" in step_def.summary()


def test_stepwise_fgwas_null_data_selects_empty_model() -> None:
    rng = np.random.default_rng(890)
    df, _ = _simulate_fgwas_dataset(
        rng, n_blocks=20, snps_per_block=40, pi_region=0.2, gamma_dnase=0.0, gamma_coding=0.0
    )
    step_null = stepwise_fgwas(
        df,
        candidate_annotations=["dnase", "noise_ann"],
        block_col="SEGNUMBER",
        llk_threshold=4.0,
        settings="default",
    )
    assert step_null.selected_annotations == ()
    assert step_null.final_fit.gamma.shape == (0,)


@pytest.mark.parametrize("bad_settings", ["wrong", "r_default", "DEFAULT", ""])
def test_invalid_settings_raises_value_error(bad_settings: str) -> None:
    df = pd.DataFrame(
        {
            "SEGNUMBER": [0, 0, 1, 1],
            "beta": [0.1, -0.1, 0.2, 0.0],
            "se": [0.05, 0.05, 0.05, 0.05],
            "ann": [1.0, 0.0, 1.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="settings"):
        wakefield_lnabf(df["beta"], df["se"], settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        fit_fgwas(df, ["ann"], block_col="SEGNUMBER", settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        cross_validate_fgwas(df, ["ann"], block_col="SEGNUMBER", n_folds=2, settings=bad_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        stepwise_fgwas(df, ["ann"], block_col="SEGNUMBER", n_folds=2, settings=bad_settings)  # type: ignore[arg-type]


def test_fgwas_input_validation_errors() -> None:
    df = pd.DataFrame({"beta": [0.1, 0.2], "se": [0.05, 0.05], "ann": [0.0, 1.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        wakefield_lnabf([0.1, 0.2], [0.0, 0.05])
    with pytest.raises(ValueError, match="V must be finite and > 0"):
        wakefield_lnabf([0.1], [0.05], V=-0.1)
    with pytest.raises(ValueError, match="not found in df"):
        fit_fgwas(df, ["missing_col"])
    with pytest.raises(ValueError, match="at least 2 genomic blocks"):
        cross_validate_fgwas(df, ["ann"], window_size=5000, n_folds=5)
