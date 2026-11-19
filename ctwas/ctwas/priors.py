"""
Stage-1 genome-wide Empirical Bayes Expectation-Maximization (EM) for cTWAS
prior inclusion probabilities (pi_G, pi_S) and effect-size variances (sigma_G^2, sigma_S^2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .impute import ImputedRegionData, SettingsMode, validate_settings


@dataclass
class CTWASPriors:
    """Empirical Bayes prior parameters learned across LD regions in Stage 1."""

    pi_gene: float
    pi_snp: float
    var_gene: float
    var_snp: float
    enrichment: float
    pve_gene: float
    pve_snp: float
    prop_h2_gene: float
    log_likelihood: float
    n_iter: int
    converged: bool
    n_genes: int
    n_snps: int
    n_samples: int
    thin: float
    settings: SettingsMode
    stage1_region_pips: list[np.ndarray] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary of scalar prior and heritability summary metrics."""
        return {
            "pi_gene": self.pi_gene,
            "pi_snp": self.pi_snp,
            "var_gene": self.var_gene,
            "var_snp": self.var_snp,
            "enrichment": self.enrichment,
            "pve_gene": self.pve_gene,
            "pve_snp": self.pve_snp,
            "prop_h2_gene": self.prop_h2_gene,
            "log_likelihood": self.log_likelihood,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "n_genes": self.n_genes,
            "n_snps": self.n_snps,
            "n_samples": self.n_samples,
            "thin": self.thin,
            "settings": self.settings,
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def keys(self) -> list[str]:
        return list(self.to_dict().keys())

    def items(self) -> list[tuple[str, Any]]:
        return list(self.to_dict().items())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def normalize_region_entry(entry: ImputedRegionData | dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize an ``ImputedRegionData`` or ``dict`` into canonical arrays."""
    if isinstance(entry, ImputedRegionData):
        raw = entry.to_dict()
    elif isinstance(entry, dict):
        raw = entry
    else:
        raise TypeError(
            f"Each region must be an ImputedRegionData or dict, got {type(entry)}."
        )

    if "z_joint" not in raw or "is_gene" not in raw:
        raise ValueError("Each region must provide 'z_joint' and 'is_gene'.")

    z_joint = np.asarray(raw["z_joint"], dtype=float)
    is_gene = np.asarray(raw["is_gene"], dtype=bool)
    p_tot = len(z_joint)
    if p_tot == 0 or is_gene.shape != (p_tot,):
        raise ValueError(f"Invalid z_joint / is_gene dimensions in region {idx}.")

    if "R_joint" in raw and raw["R_joint"] is not None:
        r_joint = np.asarray(raw["R_joint"], dtype=float)
        if r_joint.shape != (p_tot, p_tot):
            raise ValueError(
                f"R_joint shape {r_joint.shape} does not match z_joint ({p_tot},) in region {idx}."
            )
    else:
        r_joint = np.eye(p_tot, dtype=float)

    if "feature_ids" in raw and raw["feature_ids"] is not None:
        feature_ids = [str(f) for f in raw["feature_ids"]]
    else:
        g_counter = 0
        s_counter = 0
        feature_ids = []
        for ig in is_gene:
            if ig:
                g_counter += 1
                feature_ids.append(f"r{idx + 1}_gene{g_counter}")
            else:
                s_counter += 1
                feature_ids.append(f"r{idx + 1}_snp{s_counter}")

    n_snps_reg = int(np.sum(~is_gene))
    if "is_cis_snp" in raw and raw["is_cis_snp"] is not None:
        is_cis_snp = np.asarray(raw["is_cis_snp"], dtype=bool)
        if len(is_cis_snp) == p_tot:
            is_cis_snp = is_cis_snp[~is_gene]
        elif len(is_cis_snp) != n_snps_reg:
            raise ValueError("is_cis_snp length must match number of SNPs in region.")
    else:
        # Infer cis-SNPs as SNPs correlated (|r| > 0.05) with at least one gene in region
        if np.any(is_gene) and n_snps_reg > 0:
            r_gs = r_joint[np.ix_(is_gene, ~is_gene)]
            is_cis_snp = np.any(np.abs(r_gs) > 0.05, axis=0)
        else:
            is_cis_snp = np.zeros(n_snps_reg, dtype=bool)

    region_id = str(raw.get("region_id", f"region_{idx + 1}"))
    return {
        "region_id": region_id,
        "z_joint": z_joint,
        "R_joint": r_joint,
        "feature_ids": feature_ids,
        "is_gene": is_gene,
        "is_cis_snp": is_cis_snp,
        "n_samples": raw.get("n_samples"),
    }


def _compute_ser_region_posterior(
    z: np.ndarray,
    is_gene: np.ndarray,
    mult: np.ndarray,
    pi_gene: float,
    pi_snp: float,
    v_gene: float,
    v_snp: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-Effect Regression (L=1 with null component) E-step in a single region."""
    v_vec = np.where(is_gene, v_gene, v_snp)
    base_pi = np.where(is_gene, pi_gene, pi_snp)
    eff_pi = mult * base_pi

    sum_pi = float(np.sum(eff_pi))
    if sum_pi >= 0.99:
        scale = 0.99 / sum_pi
        eff_pi = eff_pi * scale
        pi_null = 0.01
    else:
        pi_null = max(1.0 - sum_pi, 1e-6)

    shrink = v_vec / (1.0 + v_vec)
    lbf = -0.5 * np.log1p(v_vec) + 0.5 * shrink * (z * z)

    log_u_features = np.where(eff_pi > 0.0, np.log(np.maximum(eff_pi, 1e-300)) + lbf, -np.inf)
    log_u_all = np.concatenate([[np.log(pi_null)], log_u_features])
    log_norm = float(logsumexp(log_u_all))

    alpha = np.exp(log_u_features - log_norm)
    second_moment_gamma = (shrink * shrink) * (z * z) + shrink
    return alpha, second_moment_gamma, log_norm


def estimate_ctwas_priors(
    regions: list[ImputedRegionData | dict[str, Any]],
    max_iter: int = 50,
    tol: float = 1e-4,
    thin: float | None = None,
    settings: SettingsMode = "default",
    n_samples: int | None = None,
    seed: int = 42,
) -> CTWASPriors:
    """Estimate cTWAS group prior inclusion probabilities and effect variances via Stage-1 EM.

    Parameters
    ----------
    regions : list of ImputedRegionData or dict
        Independent LD regions containing joint gene + SNP Z-scores.
    max_iter : int, default 50
        Maximum number of Empirical Bayes EM iterations.
    tol : float, default 1e-4
        Convergence tolerance on log-likelihood and relative prior parameter changes.
    thin : float or None, default None
        Fraction of background non-coding SNPs retained in Stage 1. Defaults to ``1.0``
        when ``settings="default"`` and ``0.1`` when ``settings="R_default"``
        (retaining SNPs with ``|z| > 1.96`` or cis-weight SNPs with multiplicity
        correction ``~1/thin`` on thinned background SNPs).
    settings : {"default", "R_default"}, default "default"
        Pipeline settings mode.
    n_samples : int or None, default None
        GWAS sample size ``n`` used to convert Z-score noncentrality variance
        ``V_k = n * sigma_k^2`` to per-feature PVE variance ``sigma_k^2`` (defaults
        to region ``n_samples`` or ``50000``).
    seed : int, default 42
        Random seed for deterministic SNP thinning when ``thin < 1.0``.
    """
    validate_settings(settings)
    if not regions:
        raise ValueError("regions must contain at least 1 region.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    eff_thin = (1.0 if settings == "default" else 0.1) if thin is None else float(thin)
    if not (0.0 < eff_thin <= 1.0):
        raise ValueError(f"thin must be in (0, 1], got {eff_thin}.")

    norm_regions = [normalize_region_entry(r, idx) for idx, r in enumerate(regions)]
    if n_samples is None:
        reg_ns = [r["n_samples"] for r in norm_regions if r["n_samples"] is not None]
        n_eff = int(reg_ns[0]) if reg_ns else 50000
    else:
        n_eff = int(n_samples)
    if n_eff < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_eff}.")

    rng = np.random.default_rng(seed)

    # Prepare thinned Stage-1 working views per region
    work_regions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_genes = 0
    total_snps = 0.0

    all_gene_z: list[np.ndarray] = []
    all_snp_z: list[np.ndarray] = []

    for reg in norm_regions:
        z = reg["z_joint"]
        ig = reg["is_gene"]
        is_cis = reg["is_cis_snp"]

        g_idx = np.flatnonzero(ig)
        s_idx = np.flatnonzero(~ig)
        total_genes += len(g_idx)
        total_snps += float(len(s_idx))

        if len(g_idx) > 0:
            all_gene_z.append(z[g_idx])
        if len(s_idx) > 0:
            all_snp_z.append(z[s_idx])

        if eff_thin >= 1.0 or len(s_idx) == 0:
            mult = np.ones(len(z), dtype=float)
            work_regions.append((z, ig, mult))
        else:
            z_s = z[s_idx]
            must_keep = (np.abs(z_s) > 1.96) | is_cis
            bg_local = np.flatnonzero(~must_keep)
            keep_local = np.flatnonzero(must_keep)

            if len(bg_local) == 0:
                mult = np.ones(len(z), dtype=float)
                work_regions.append((z, ig, mult))
            else:
                n_sub = max(1, int(np.rint(eff_thin * len(bg_local))))
                chosen_bg = np.sort(rng.choice(bg_local, size=n_sub, replace=False))
                bg_weight = float(len(bg_local)) / float(n_sub)

                kept_s_global = np.concatenate([s_idx[keep_local], s_idx[chosen_bg]])
                kept_s_mult = np.concatenate(
                    [
                        np.ones(len(keep_local), dtype=float),
                        np.full(len(chosen_bg), bg_weight, dtype=float),
                    ]
                )
                sel_idx = np.concatenate([g_idx, kept_s_global])
                sel_ig = np.concatenate(
                    [np.ones(len(g_idx), dtype=bool), np.zeros(len(kept_s_global), dtype=bool)]
                )
                sel_mult = np.concatenate([np.ones(len(g_idx), dtype=float), kept_s_mult])
                work_regions.append((z[sel_idx], sel_ig, sel_mult))

    n_reg = len(norm_regions)
    pi_gene = min(0.05, 0.20 / max(1.0, total_genes / n_reg)) if total_genes > 0 else 0.0
    pi_snp = min(0.01, 0.20 / max(1.0, total_snps / n_reg)) if total_snps > 0 else 0.0

    gz_cat = np.concatenate(all_gene_z) if all_gene_z else np.zeros(0)
    sz_cat = np.concatenate(all_snp_z) if all_snp_z else np.zeros(0)
    v_gene = (
        float(np.clip(np.mean(gz_cat[np.abs(gz_cat) > 2.0] ** 2 - 1.0), 1.0, 50.0))
        if np.any(np.abs(gz_cat) > 2.0)
        else 5.0
    )
    v_snp = (
        float(np.clip(np.mean(sz_cat[np.abs(sz_cat) > 2.0] ** 2 - 1.0), 1.0, 50.0))
        if np.any(np.abs(sz_cat) > 2.0)
        else 5.0
    )

    prev_ll = -np.inf
    converged = False
    n_iter_done = 0

    for it in range(1, max_iter + 1):
        n_iter_done = it
        gamma_g_sum = 0.0
        gamma_s_sum = 0.0
        m2_g_sum = 0.0
        m2_s_sum = 0.0
        total_ll = 0.0

        for z_w, ig_w, mult_w in work_regions:
            alpha, m2, reg_ll = _compute_ser_region_posterior(
                z_w, ig_w, mult_w, pi_gene, pi_snp, v_gene, v_snp
            )
            total_ll += reg_ll
            if np.any(ig_w):
                ag = alpha[ig_w]
                gamma_g_sum += float(np.sum(ag))
                m2_g_sum += float(np.sum(ag * m2[ig_w]))
            if np.any(~ig_w):
                asnp = alpha[~ig_w]
                gamma_s_sum += float(np.sum(asnp))
                m2_s_sum += float(np.sum(asnp * m2[~ig_w]))

        if settings == "default":
            # Weak Empirical Bayes stabilization prevents small-region collapse to 0
            new_pi_gene = (gamma_g_sum + 1e-3) / (total_genes + 1.0) if total_genes > 0 else 0.0
            new_pi_snp = (gamma_s_sum + 1e-3) / (total_snps + 1.0) if total_snps > 0 else 0.0
            new_v_gene = (
                float(np.clip((m2_g_sum + 0.02) / (gamma_g_sum + 0.01), 1e-3, 1e4))
                if total_genes > 0
                else v_gene
            )
            new_v_snp = (
                float(np.clip((m2_s_sum + 0.02) / (gamma_s_sum + 0.01), 1e-3, 1e4))
                if total_snps > 0
                else v_snp
            )
        else:
            new_pi_gene = gamma_g_sum / total_genes if total_genes > 0 else 0.0
            new_pi_snp = gamma_s_sum / total_snps if total_snps > 0 else 0.0
            new_v_gene = (
                float(np.clip(m2_g_sum / max(gamma_g_sum, 1e-12), 1e-4, 1e4))
                if total_genes > 0 and gamma_g_sum > 1e-12
                else v_gene
            )
            new_v_snp = (
                float(np.clip(m2_s_sum / max(gamma_s_sum, 1e-12), 1e-4, 1e4))
                if total_snps > 0 and gamma_s_sum > 1e-12
                else v_snp
            )

        new_pi_gene = float(np.clip(new_pi_gene, 0.0, 0.99))
        new_pi_snp = float(np.clip(new_pi_snp, 0.0, 0.99))

        rel_change = max(
            abs(new_pi_gene - pi_gene) / max(pi_gene, 1e-8),
            abs(new_pi_snp - pi_snp) / max(pi_snp, 1e-8),
        )
        ll_diff = abs(total_ll - prev_ll)
        pi_gene, pi_snp, v_gene, v_snp = new_pi_gene, new_pi_snp, new_v_gene, new_v_snp
        prev_ll = total_ll

        if it > 1 and ll_diff < tol and rel_change < 10.0 * tol:
            converged = True
            break

    # Compute exact unthinned Stage-1 L=1 PIPs across all features of every region
    stage1_pips: list[np.ndarray] = []
    final_ll = 0.0
    for reg in norm_regions:
        z_full = reg["z_joint"]
        ig_full = reg["is_gene"]
        mult_ones = np.ones(len(z_full), dtype=float)
        alpha_full, _, reg_ll_full = _compute_ser_region_posterior(
            z_full, ig_full, mult_ones, pi_gene, pi_snp, v_gene, v_snp
        )
        stage1_pips.append(alpha_full)
        final_ll += reg_ll_full

    var_gene = float(v_gene / n_eff)
    var_snp = float(v_snp / n_eff)
    pve_gene = float(total_genes * pi_gene * var_gene)
    pve_snp = float(total_snps * pi_snp * var_snp)
    total_pve = pve_gene + pve_snp
    prop_h2_gene = float(pve_gene / total_pve) if total_pve > 0.0 else 0.0
    enrichment = float(pi_gene / max(pi_snp, 1e-15)) if total_genes > 0 else 0.0

    return CTWASPriors(
        pi_gene=pi_gene,
        pi_snp=pi_snp,
        var_gene=var_gene,
        var_snp=var_snp,
        enrichment=enrichment,
        pve_gene=pve_gene,
        pve_snp=pve_snp,
        prop_h2_gene=prop_h2_gene,
        log_likelihood=float(final_ll),
        n_iter=n_iter_done,
        converged=converged,
        n_genes=int(total_genes),
        n_snps=int(round(total_snps)),
        n_samples=n_eff,
        thin=eff_thin,
        settings=settings,
        stage1_region_pips=stage1_pips,
    )
