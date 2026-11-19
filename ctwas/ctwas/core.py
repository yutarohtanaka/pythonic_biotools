"""
End-to-end two-stage cTWAS pipeline (Stage-1 Empirical Bayes prior estimation +
Stage-2 group-prior SuSiE-RSS fine-mapping), OOP estimator, and result container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .impute import ImputedRegionData, SettingsMode, validate_settings
from .priors import CTWASPriors, estimate_ctwas_priors, normalize_region_entry
from .susie_rss import susie_rss_group_prior


@dataclass
class CTWASResult:
    """Result container for two-stage cTWAS fine-mapping and heritability partitioning."""

    prior_params: CTWASPriors
    gene_df: pd.DataFrame
    snp_df: pd.DataFrame
    credible_sets: list[dict[str, Any]]
    pve_gene: float
    pve_snp: float
    enrichment: float
    prop_h2_gene: float
    n_samples: int
    L: int
    settings: SettingsMode

    def summary(self) -> str:
        """Return a concise multi-line summary of cTWAS priors and fine-mapped risk genes."""
        n_high_genes = int(np.sum(self.gene_df["susie_pip"].to_numpy() >= 0.8)) if len(self.gene_df) else 0
        lines = [
            f"CTWASResult(settings={self.settings!r}, n_samples={self.n_samples}, L={self.L})",
            f"  Priors: pi_gene={self.prior_params.pi_gene:.4e}, pi_snp={self.prior_params.pi_snp:.4e} "
            f"(enrichment={self.enrichment:.2f}x)",
            f"  Effect variances: var_gene={self.prior_params.var_gene:.4e}, "
            f"var_snp={self.prior_params.var_snp:.4e}",
            f"  Heritability / PVE: pve_gene={self.pve_gene:.4f}, pve_snp={self.pve_snp:.4f} "
            f"(mediated={100.0 * self.prop_h2_gene:.1f}%)",
            f"  Fine-mapping: {n_high_genes} genes with PIP >= 0.80, "
            f"{len(self.credible_sets)} credible sets across {len(self.gene_df)} genes & {len(self.snp_df)} SNPs",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return combined gene + SNP posterior table sorted by descending ``susie_pip``."""
        frames: list[pd.DataFrame] = []
        if len(self.gene_df) > 0:
            gdf = self.gene_df.rename(columns={"gene_id": "feature_id"}).copy()
            gdf["feature_type"] = "gene"
            frames.append(
                gdf[
                    [
                        "region_id",
                        "feature_id",
                        "feature_type",
                        "z",
                        "stage1_pip",
                        "susie_pip",
                        "in_cs",
                        "cs_id",
                        "rerun_stage2",
                    ]
                ]
            )
        if len(self.snp_df) > 0:
            sdf = self.snp_df.rename(columns={"snp_id": "feature_id"}).copy()
            sdf["feature_type"] = "snp"
            frames.append(
                sdf[
                    [
                        "region_id",
                        "feature_id",
                        "feature_type",
                        "z",
                        "stage1_pip",
                        "susie_pip",
                        "in_cs",
                        "cs_id",
                        "rerun_stage2",
                    ]
                ]
            )
        if not frames:
            return pd.DataFrame(
                columns=[
                    "region_id",
                    "feature_id",
                    "feature_type",
                    "z",
                    "stage1_pip",
                    "susie_pip",
                    "in_cs",
                    "cs_id",
                    "rerun_stage2",
                ]
            )
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("susie_pip", ascending=False, kind="mergesort").reset_index(
            drop=True
        )


def run_ctwas(
    regions: list[ImputedRegionData | dict[str, Any]],
    n_samples: int,
    L: int = 5,
    max_iter_eb: int = 50,
    max_iter_susie: int = 100,
    tol: float = 1e-4,
    thin: float | None = None,
    rerun_gene_pip: float | None = None,
    coverage: float = 0.95,
    seed: int = 42,
    settings: SettingsMode = "default",
) -> CTWASResult:
    """Run two-stage cTWAS (Empirical Bayes group-prior learning + SuSiE-RSS fine-mapping).

    Parameters
    ----------
    regions : list of ImputedRegionData or dict
        Genomic LD blocks containing joint molecular trait + cis-SNP Z-scores and LD.
    n_samples : int
        GWAS sample size ``n``.
    L : int, default 5
        Maximum number of causal effects per region in Stage-2 SuSiE-RSS.
    max_iter_eb : int, default 50
        Maximum iterations for Stage-1 Empirical Bayes EM.
    max_iter_susie : int, default 100
        Maximum IBSS iterations per region in Stage-2 SuSiE-RSS.
    tol : float, default 1e-4
        Convergence tolerance for Stage 1 and Stage 2.
    thin : float or None, default None
        Stage-1 non-coding SNP thinning fraction (defaults to ``1.0`` in ``"default"``
        and ``0.1`` in ``"R_default"``).
    rerun_gene_pip : float or None, default None
        Minimum Stage-1 regional gene PIP sum required to run Stage-2 ``L``-effect
        SuSiE-RSS (defaults to ``0.0`` in ``"default"`` so all regions are fine-mapped,
        and ``0.8`` in ``"R_default"`` matching ``ctwas::ctwas_rss``).
    coverage : float, default 0.95
        Target posterior coverage for credible sets.
    seed : int, default 42
        Random seed for Stage-1 thinning.
    settings : {"default", "R_default"}, default "default"
        Execution settings mode.
    """
    validate_settings(settings)
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if int(L) < 1:
        raise ValueError(f"L must be >= 1, got {L}.")

    norm_regions = [normalize_region_entry(r, idx) for idx, r in enumerate(regions)]
    priors = estimate_ctwas_priors(
        norm_regions,
        max_iter=max_iter_eb,
        tol=tol,
        thin=thin,
        settings=settings,
        n_samples=int(n_samples),
        seed=seed,
    )

    eff_rerun = (
        (0.0 if settings == "default" else 0.8)
        if rerun_gene_pip is None
        else float(rerun_gene_pip)
    )

    gene_rows: list[dict[str, Any]] = []
    snp_rows: list[dict[str, Any]] = []
    all_credible_sets: list[dict[str, Any]] = []
    global_cs_counter = 0

    for reg_idx, reg in enumerate(norm_regions):
        reg_id = reg["region_id"]
        z = reg["z_joint"]
        r_mat = reg["R_joint"]
        ig = reg["is_gene"]
        f_ids = reg["feature_ids"]
        is_cis = reg["is_cis_snp"]
        s1_pip = priors.stage1_region_pips[reg_idx]

        reg_gene_pip_sum = float(np.sum(s1_pip[ig])) if np.any(ig) else 0.0
        do_stage2 = (settings == "default" and eff_rerun <= 0.0) or (
            reg_gene_pip_sum >= eff_rerun
        )

        cs_membership = np.zeros(len(z), dtype=bool)
        cs_id_arr = np.zeros(len(z), dtype=int)

        if do_stage2:
            s2_res = susie_rss_group_prior(
                z_joint=z,
                R_joint=r_mat,
                is_gene=ig,
                pi_gene=priors.pi_gene,
                pi_snp=priors.pi_snp,
                var_gene=priors.var_gene,
                var_snp=priors.var_snp,
                n_samples=int(n_samples),
                L=int(L),
                max_iter=max_iter_susie,
                tol=tol,
                coverage=coverage,
                settings=settings,
            )
            final_pip = s2_res.susie_pip
            for cs in s2_res.credible_sets:
                global_cs_counter += 1
                idxs = cs["indices"]
                cs_membership[idxs] = True
                for i_cs in idxs:
                    if cs_id_arr[i_cs] == 0:
                        cs_id_arr[i_cs] = global_cs_counter
                all_credible_sets.append(
                    {
                        "cs_id": global_cs_counter,
                        "region_id": reg_id,
                        "effect_index": cs["effect_index"],
                        "feature_ids": [f_ids[i_cs] for i_cs in idxs],
                        "feature_types": ["gene" if ig[i_cs] else "snp" for i_cs in idxs],
                        "pips": [float(final_pip[i_cs]) for i_cs in idxs],
                        "coverage": cs["coverage"],
                        "purity": cs["purity"],
                        "contains_gene": cs["contains_gene"],
                    }
                )
        else:
            final_pip = s1_pip.copy()

        snp_cursor = 0
        for j in range(len(z)):
            if ig[j]:
                gene_rows.append(
                    {
                        "region_id": reg_id,
                        "gene_id": f_ids[j],
                        "z": float(z[j]),
                        "stage1_pip": float(s1_pip[j]),
                        "susie_pip": float(final_pip[j]),
                        "in_cs": bool(cs_membership[j]),
                        "cs_id": int(cs_id_arr[j]),
                        "rerun_stage2": bool(do_stage2),
                    }
                )
            else:
                snp_rows.append(
                    {
                        "region_id": reg_id,
                        "snp_id": f_ids[j],
                        "z": float(z[j]),
                        "stage1_pip": float(s1_pip[j]),
                        "susie_pip": float(final_pip[j]),
                        "in_cs": bool(cs_membership[j]),
                        "cs_id": int(cs_id_arr[j]),
                        "is_cis_snp": bool(is_cis[snp_cursor]),
                        "rerun_stage2": bool(do_stage2),
                    }
                )
                snp_cursor += 1

    gene_df = (
        pd.DataFrame(gene_rows)
        if gene_rows
        else pd.DataFrame(
            columns=[
                "region_id",
                "gene_id",
                "z",
                "stage1_pip",
                "susie_pip",
                "in_cs",
                "cs_id",
                "rerun_stage2",
            ]
        )
    )
    snp_df = (
        pd.DataFrame(snp_rows)
        if snp_rows
        else pd.DataFrame(
            columns=[
                "region_id",
                "snp_id",
                "z",
                "stage1_pip",
                "susie_pip",
                "in_cs",
                "cs_id",
                "is_cis_snp",
                "rerun_stage2",
            ]
        )
    )

    return CTWASResult(
        prior_params=priors,
        gene_df=gene_df,
        snp_df=snp_df,
        credible_sets=all_credible_sets,
        pve_gene=priors.pve_gene,
        pve_snp=priors.pve_snp,
        enrichment=priors.enrichment,
        prop_h2_gene=priors.prop_h2_gene,
        n_samples=int(n_samples),
        L=int(L),
        settings=settings,
    )


class CTWAS:
    """Object-oriented estimator interface for causal TWAS / PWAS (cTWAS)."""

    def __init__(
        self,
        L: int = 5,
        max_iter_eb: int = 50,
        max_iter_susie: int = 100,
        tol: float = 1e-4,
        thin: float | None = None,
        rerun_gene_pip: float | None = None,
        coverage: float = 0.95,
        seed: int = 42,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        self.L = L
        self.max_iter_eb = max_iter_eb
        self.max_iter_susie = max_iter_susie
        self.tol = tol
        self.thin = thin
        self.rerun_gene_pip = rerun_gene_pip
        self.coverage = coverage
        self.seed = seed
        self.settings = settings
        self.result_: CTWASResult | None = None

    def fit(
        self,
        regions: list[ImputedRegionData | dict[str, Any]],
        n_samples: int,
    ) -> CTWASResult:
        """Fit two-stage cTWAS model across ``regions``."""
        self.result_ = run_ctwas(
            regions=regions,
            n_samples=n_samples,
            L=self.L,
            max_iter_eb=self.max_iter_eb,
            max_iter_susie=self.max_iter_susie,
            tol=self.tol,
            thin=self.thin,
            rerun_gene_pip=self.rerun_gene_pip,
            coverage=self.coverage,
            seed=self.seed,
            settings=self.settings,
        )
        return self.result_
