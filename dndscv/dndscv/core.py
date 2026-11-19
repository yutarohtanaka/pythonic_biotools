"""
Core `dndscv` workflow, `DnDScvModel`, and `DnDScvResult` frozen dataclass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .annotate import AnnotationResult, annotate_mutations
from .model import (
    NBCovariateFit,
    SettingsMode,
    bh_fdr_nan_aware,
    compute_gene_tests_cv,
    compute_global_dnds,
    compute_indel_tests,
    fit_nb_covariate_model,
    validate_settings,
)
from .refcds import TRINUC_CONTEXTS, RefCDS, _prepare_covariates_df


@dataclass(frozen=True)
class DnDScvResult:
    """Typed result container for `dndscv` somatic selection analysis."""

    gene_names: list[str]
    cds_lengths: np.ndarray
    n_syn: np.ndarray
    n_mis: np.ndarray
    n_non: np.ndarray
    n_spl: np.ndarray
    n_ind: np.ndarray
    exp_syn: np.ndarray
    exp_mis: np.ndarray
    exp_non: np.ndarray
    exp_spl: np.ndarray
    exp_ind: np.ndarray
    mu_syn: np.ndarray
    mu_ind: np.ndarray
    w_mis_cv: np.ndarray
    w_non_cv: np.ndarray
    w_spl_cv: np.ndarray
    w_ind_cv: np.ndarray
    w_mis_p: np.ndarray
    w_non_p: np.ndarray
    w_spl_p: np.ndarray
    p_mis_cv: np.ndarray
    p_non_cv: np.ndarray
    p_spl_cv: np.ndarray
    p_tru_cv: np.ndarray
    p_all_cv: np.ndarray
    p_ind_cv: np.ndarray
    p_global_cv: np.ndarray
    q_mis_cv: np.ndarray
    q_non_cv: np.ndarray
    q_spl_cv: np.ndarray
    q_tru_cv: np.ndarray
    q_all_cv: np.ndarray
    q_ind_cv: np.ndarray
    q_global_cv: np.ndarray
    p_mis_p: np.ndarray
    p_non_p: np.ndarray
    p_spl_p: np.ndarray
    p_tru_p: np.ndarray
    p_all_p: np.ndarray
    q_all_p: np.ndarray
    global_names: list[str]
    global_mle: np.ndarray
    global_cilow: np.ndarray
    global_cihigh: np.ndarray
    global_n_obs: np.ndarray
    global_n_exp: np.ndarray
    global_p_value: np.ndarray
    r_mle: np.ndarray
    n_192: np.ndarray
    L_192: np.ndarray
    theta_syn: float
    alpha_syn: float
    is_poisson_syn: bool
    beta_syn: np.ndarray
    se_syn: np.ndarray
    p_beta_syn: np.ndarray
    coef_names: list[str]
    theta_ind: float
    alpha_ind: float
    is_poisson_ind: bool
    beta_ind: np.ndarray
    excluded_samples: list[str]
    annot_muts: pd.DataFrame
    converged: bool
    settings: SettingsMode = "default"

    def _build_r_sel_cv(self, sort_by_pvalue: bool = False) -> pd.DataFrame:
        """Construct exact upstream `im3sanger/dndscv` `sel_cv` DataFrame."""
        has_indels = int(np.sum(self.n_ind)) > 0
        if has_indels:
            df = pd.DataFrame(
                {
                    "gene_name": self.gene_names,
                    "n_syn": self.n_syn,
                    "n_mis": self.n_mis,
                    "n_non": self.n_non,
                    "n_spl": self.n_spl,
                    "wmis_cv": self.w_mis_cv,
                    "wnon_cv": self.w_non_cv,
                    "wspl_cv": self.w_spl_cv,
                    "wind_cv": self.w_ind_cv,
                    "pmis_cv": self.p_mis_cv,
                    "ptrunc_cv": self.p_tru_cv,
                    "pallsubs_cv": self.p_all_cv,
                    "pind_cv": self.p_ind_cv,
                    "qmis_cv": self.q_mis_cv,
                    "qtrunc_cv": self.q_tru_cv,
                    "qallsubs_cv": self.q_all_cv,
                    "qind_cv": self.q_ind_cv,
                    "pglobal_cv": self.p_global_cv,
                    "qglobal_cv": self.q_global_cv,
                }
            )
            if sort_by_pvalue:
                df = df.sort_values(
                    ["pglobal_cv", "pallsubs_cv", "pmis_cv", "ptrunc_cv", "gene_name"],
                    kind="mergesort",
                    na_position="last",
                ).reset_index(drop=True)
            return df

        df = pd.DataFrame(
            {
                "gene_name": self.gene_names,
                "n_syn": self.n_syn,
                "n_mis": self.n_mis,
                "n_non": self.n_non,
                "n_spl": self.n_spl,
                "wmis_cv": self.w_mis_cv,
                "wnon_cv": self.w_non_cv,
                "wspl_cv": self.w_spl_cv,
                "pmis_cv": self.p_mis_cv,
                "ptrunc_cv": self.p_tru_cv,
                "pallsubs_cv": self.p_all_cv,
                "qmis_cv": self.q_mis_cv,
                "qtrunc_cv": self.q_tru_cv,
                "qallsubs_cv": self.q_all_cv,
            }
        )
        if sort_by_pvalue:
            df = df.sort_values(
                ["pallsubs_cv", "pmis_cv", "ptrunc_cv", "gene_name"],
                kind="mergesort",
                na_position="last",
            ).reset_index(drop=True)
        return df

    @property
    def sel_cv(self) -> pd.DataFrame:
        """Upstream `im3sanger/dndscv` `sel_cv` table sorted by p-value."""
        return self._build_r_sel_cv(sort_by_pvalue=True)

    def to_dataframe(self, sort_by_pvalue: bool = False) -> pd.DataFrame:
        """Return gene-level selection ratios, p-values, and BH-adjusted q-values."""
        if self.settings == "R_default":
            return self._build_r_sel_cv(sort_by_pvalue=sort_by_pvalue)
        df = pd.DataFrame(
            {
                "gene_name": self.gene_names,
                "cds_length": self.cds_lengths,
                "n_syn": self.n_syn,
                "n_mis": self.n_mis,
                "n_non": self.n_non,
                "n_spl": self.n_spl,
                "n_ind": self.n_ind,
                "exp_syn": self.exp_syn,
                "exp_mis": self.exp_mis,
                "exp_non": self.exp_non,
                "exp_spl": self.exp_spl,
                "exp_ind": self.exp_ind,
                "mu_syn": self.mu_syn,
                "mu_ind": self.mu_ind,
                "w_mis_cv": self.w_mis_cv,
                "w_non_cv": self.w_non_cv,
                "w_spl_cv": self.w_spl_cv,
                "w_ind_cv": self.w_ind_cv,
                "p_mis_cv": self.p_mis_cv,
                "p_non_cv": self.p_non_cv,
                "p_spl_cv": self.p_spl_cv,
                "p_tru_cv": self.p_tru_cv,
                "p_all_cv": self.p_all_cv,
                "p_ind_cv": self.p_ind_cv,
                "p_global_cv": self.p_global_cv,
                "q_mis_cv": self.q_mis_cv,
                "q_non_cv": self.q_non_cv,
                "q_spl_cv": self.q_spl_cv,
                "q_tru_cv": self.q_tru_cv,
                "q_all_cv": self.q_all_cv,
                "q_ind_cv": self.q_ind_cv,
                "q_global_cv": self.q_global_cv,
                "w_mis_p": self.w_mis_p,
                "w_non_p": self.w_non_p,
                "w_spl_p": self.w_spl_p,
                "p_mis_p": self.p_mis_p,
                "p_non_p": self.p_non_p,
                "p_spl_p": self.p_spl_p,
                "p_tru_p": self.p_tru_p,
                "p_all_p": self.p_all_p,
                "q_all_p": self.q_all_p,
            }
        )
        if sort_by_pvalue:
            df = df.sort_values(
                ["p_global_cv", "p_all_cv", "gene_name"],
                kind="mergesort",
                na_position="last",
            ).reset_index(drop=True)
        return df

    def global_dnds_dataframe(self) -> pd.DataFrame:
        """Return the 5-row global selection table (`wmis`, `wnon`, `wspl`, `wtru`, `wall`)."""
        return pd.DataFrame(
            {
                "name": self.global_names,
                "mle": self.global_mle,
                "cilow": self.global_cilow,
                "cihigh": self.global_cihigh,
                "n_obs": self.global_n_obs,
                "n_exp": self.global_n_exp,
                "p_value": self.global_p_value,
            }
        )

    def significant(self, alpha: float = 0.05, by: str = "q_global_cv") -> pd.DataFrame:
        """Return genes passing FDR threshold `alpha` sorted deterministically by p-value."""
        df = self.to_dataframe(sort_by_pvalue=False)
        col = by
        if self.settings == "R_default" and col not in df.columns:
            alias_map = {
                "q_global_cv": "qglobal_cv" if "qglobal_cv" in df.columns else "qallsubs_cv",
                "q_all_cv": "qallsubs_cv",
                "q_mis_cv": "qmis_cv",
                "q_tru_cv": "qtrunc_cv",
                "q_ind_cv": "qind_cv",
            }
            col = alias_map.get(col, col)
        if col not in df.columns:
            raise ValueError(f"Column {by!r} not found in result DataFrame.")
        sub = df.loc[df[col].notna() & (df[col] <= alpha)].copy()
        if col == "q_global_cv":
            sort_col = "p_global_cv"
        elif col == "qglobal_cv":
            sort_col = "pglobal_cv"
        elif col == "qallsubs_cv":
            sort_col = "pallsubs_cv"
        else:
            sort_col = col
        return sub.sort_values(
            [sort_col, "gene_name"], kind="mergesort", na_position="last"
        ).reset_index(drop=True)

    def rates_dataframe(self) -> pd.DataFrame:
        """Return 192 coding-strand trinucleotide MLE substitution rates and pooled counts."""
        return pd.DataFrame(
            {
                "trinuc_context": list(TRINUC_CONTEXTS),
                "rate_mle": self.r_mle,
                "n_syn": self.n_192[:, 0],
                "n_mis": self.n_192[:, 1],
                "n_non": self.n_192[:, 2],
                "n_spl": self.n_192[:, 3],
                "L_syn": self.L_192[:, 0],
                "L_mis": self.L_192[:, 1],
                "L_non": self.L_192[:, 2],
                "L_spl": self.L_192[:, 3],
            }
        )

    def nb_summary_dataframe(self) -> pd.DataFrame:
        """Return synonymous Negative Binomial / Poisson regression coefficients."""
        return pd.DataFrame(
            {
                "coef": self.coef_names,
                "beta": self.beta_syn,
                "se": self.se_syn,
                "p_value": self.p_beta_syn,
            }
        )

    def summary(self) -> str:
        """Return a concise multi-line human-readable summary of the dNdScv fit."""
        n_sig_global = int(np.sum(np.isfinite(self.q_global_cv) & (self.q_global_cv <= 0.05)))
        n_sig_subs = int(np.sum(np.isfinite(self.q_all_cv) & (self.q_all_cv <= 0.05)))
        lines = [
            f"dNdScvResult: {len(self.gene_names)} genes, "
            f"{int(np.sum(self.n_syn + self.n_mis + self.n_non + self.n_spl))} coding SNVs, "
            f"{int(np.sum(self.n_ind))} coding indels",
            f"Excluded hypermutator samples: {len(self.excluded_samples)}",
            f"Synonymous NB dispersion: theta={self.theta_syn:.3g} "
            f"(alpha={self.alpha_syn:.3g}, poisson_fallback={self.is_poisson_syn})",
            f"Significant genes (q <= 0.05): global={n_sig_global}, all_subs={n_sig_subs}",
            "Global dN/dS:",
        ]
        for name, mle, lo, hi in zip(
            self.global_names, self.global_mle, self.global_cilow, self.global_cihigh
        ):
            lines.append(f"  {name}: {mle:.3f} [{lo:.3f}, {hi:.3f}]")
        return "\n".join(lines)


class DnDScvModel:
    """Object-oriented specification for fitting `dndscv` somatic selection."""

    def __init__(
        self,
        mutations: pd.DataFrame,
        refcds: RefCDS,
        *,
        covariates: pd.DataFrame | np.ndarray | None = None,
        gene_list: Sequence[str] | None = None,
        max_muts_per_gene_per_sample: int | None = 3,
        max_coding_muts_per_sample: int | None = 3000,
        known_cancergenes: Sequence[str] | None = None,
        ci_level: float = 0.95,
        ci_method: Literal["wald", "profile"] = "wald",
        rate_model: Literal["192r_3w", "192r_syn"] = "192r_3w",
        use_nb: bool = True,
        coord_system: Literal["one_based", "zero_based"] = "one_based",
        max_iter: int | None = None,
        tol: float = 1e-8,
        settings: SettingsMode = "default",
    ) -> None:
        validate_settings(settings)
        self.mutations = mutations
        self.refcds = refcds
        self.covariates = covariates
        self.gene_list = gene_list
        self.max_muts_per_gene_per_sample = max_muts_per_gene_per_sample
        self.max_coding_muts_per_sample = max_coding_muts_per_sample
        self.known_cancergenes = known_cancergenes
        self.ci_level = ci_level
        self.ci_method = ci_method
        self.rate_model = rate_model
        self.use_nb = use_nb
        self.coord_system = coord_system
        self.max_iter = max_iter
        self.tol = tol
        self.settings = settings

    def fit(self) -> DnDScvResult:
        return dndscv(
            mutations=self.mutations,
            refcds=self.refcds,
            covariates=self.covariates,
            gene_list=self.gene_list,
            max_muts_per_gene_per_sample=self.max_muts_per_gene_per_sample,
            max_coding_muts_per_sample=self.max_coding_muts_per_sample,
            known_cancergenes=self.known_cancergenes,
            ci_level=self.ci_level,
            ci_method=self.ci_method,
            rate_model=self.rate_model,
            use_nb=self.use_nb,
            coord_system=self.coord_system,
            max_iter=self.max_iter,
            tol=self.tol,
            settings=self.settings,
        )


def dndscv(
    mutations: pd.DataFrame,
    refcds: RefCDS,
    *,
    covariates: pd.DataFrame | np.ndarray | None = None,
    gene_list: Sequence[str] | None = None,
    max_muts_per_gene_per_sample: int | None = 3,
    max_coding_muts_per_sample: int | None = 3000,
    known_cancergenes: Sequence[str] | None = None,
    ci_level: float = 0.95,
    ci_method: Literal["wald", "profile"] = "wald",
    rate_model: Literal["192r_3w", "192r_syn"] = "192r_3w",
    use_nb: bool = True,
    coord_system: Literal["one_based", "zero_based"] = "one_based",
    max_iter: int | None = None,
    tol: float = 1e-8,
    settings: SettingsMode = "default",
) -> DnDScvResult:
    """
    Fit the dNdScv maximum-likelihood somatic selection model (Martincorena et al., Cell 2017).
    """
    eff_mode = validate_settings(settings)
    eff_max_iter = (25 if eff_mode == "R_default" else 100) if max_iter is None else int(max_iter)
    if not isinstance(refcds, RefCDS) or len(refcds) == 0:
        raise ValueError("refcds must be a non-empty RefCDS instance.")
    if eff_max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    # Handle gene_list subsetting and covariate alignment
    if gene_list is not None:
        sub_ref = refcds.subset(gene_list)
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                cov_df = _prepare_covariates_df(covariates, sub_ref.gene_names)
            else:
                arr = np.asarray(covariates, dtype=np.float64)
                if arr.shape[0] == len(refcds):
                    idx_map = {g: i for i, g in enumerate(refcds.gene_names)}
                    keep_idx = [idx_map[g] for g in sub_ref.gene_names]
                    cov_df = _prepare_covariates_df(arr[keep_idx], sub_ref.gene_names)
                else:
                    cov_df = _prepare_covariates_df(arr, sub_ref.gene_names)
        else:
            cov_df = sub_ref.covariates
    else:
        sub_ref = refcds
        cov_df = (
            _prepare_covariates_df(covariates, sub_ref.gene_names)
            if covariates is not None
            else sub_ref.covariates
        )

    G = len(sub_ref)
    kc_set = {str(g) for g in known_cancergenes} if known_cancergenes is not None else set()
    bg_mask = np.asarray([g not in kc_set for g in sub_ref.gene_names], dtype=bool)
    if not np.any(bg_mask):
        raise ValueError("known_cancergenes excluded all genes; no background genes remain.")

    annot: AnnotationResult = annotate_mutations(
        mutations,
        sub_ref,
        max_muts_per_gene_per_sample=max_muts_per_gene_per_sample,
        max_coding_muts_per_sample=max_coding_muts_per_sample,
        coord_system=coord_system,
    )

    n_obs = annot.n_obs  # (G, 192, 4)
    n_ind = annot.n_ind  # (G,)
    n_classes = np.sum(n_obs, axis=1).astype(np.float64)  # (G, 4)
    n_syn = n_classes[:, 0]
    n_mis = n_classes[:, 1]
    n_non = n_classes[:, 2]
    n_spl = n_classes[:, 3]
    cds_lengths = sub_ref.cds_lengths.astype(np.int64)

    n_192_bg = np.sum(n_obs[bg_mask], axis=0).astype(np.int64)
    L_192_bg = np.sum(sub_ref.L_all[bg_mask], axis=0).astype(np.int64)

    total_coding_snvs = int(np.sum(n_obs))
    total_coding_indels = int(np.sum(n_ind))

    # Empty / zero-mutation cohort edge case
    if total_coding_snvs == 0 and total_coding_indels == 0:
        cov_names = list(cov_df.columns) if cov_df is not None else []
        coef_names = ["(Intercept)", *cov_names]
        n_coef = len(coef_names)
        valid_g = cds_lengths > 0
        ones_or_nan = np.where(valid_g, 1.0, np.nan)
        zeros_or_nan = np.where(valid_g, 0.0, np.nan)
        return DnDScvResult(
            gene_names=list(sub_ref.gene_names),
            cds_lengths=cds_lengths,
            n_syn=n_syn.astype(np.int64),
            n_mis=n_mis.astype(np.int64),
            n_non=n_non.astype(np.int64),
            n_spl=n_spl.astype(np.int64),
            n_ind=n_ind.astype(np.int64),
            exp_syn=np.zeros(G, dtype=np.float64),
            exp_mis=np.zeros(G, dtype=np.float64),
            exp_non=np.zeros(G, dtype=np.float64),
            exp_spl=np.zeros(G, dtype=np.float64),
            exp_ind=np.zeros(G, dtype=np.float64),
            mu_syn=np.zeros(G, dtype=np.float64),
            mu_ind=np.zeros(G, dtype=np.float64),
            w_mis_cv=zeros_or_nan.copy(),
            w_non_cv=zeros_or_nan.copy(),
            w_spl_cv=zeros_or_nan.copy(),
            w_ind_cv=zeros_or_nan.copy(),
            w_mis_p=zeros_or_nan.copy(),
            w_non_p=zeros_or_nan.copy(),
            w_spl_p=zeros_or_nan.copy(),
            p_mis_cv=ones_or_nan.copy(),
            p_non_cv=ones_or_nan.copy(),
            p_spl_cv=ones_or_nan.copy(),
            p_tru_cv=ones_or_nan.copy(),
            p_all_cv=ones_or_nan.copy(),
            p_ind_cv=ones_or_nan.copy(),
            p_global_cv=ones_or_nan.copy(),
            q_mis_cv=ones_or_nan.copy(),
            q_non_cv=ones_or_nan.copy(),
            q_spl_cv=ones_or_nan.copy(),
            q_tru_cv=ones_or_nan.copy(),
            q_all_cv=ones_or_nan.copy(),
            q_ind_cv=ones_or_nan.copy(),
            q_global_cv=ones_or_nan.copy(),
            p_mis_p=ones_or_nan.copy(),
            p_non_p=ones_or_nan.copy(),
            p_spl_p=ones_or_nan.copy(),
            p_tru_p=ones_or_nan.copy(),
            p_all_p=ones_or_nan.copy(),
            q_all_p=ones_or_nan.copy(),
            global_names=["wmis", "wnon", "wspl", "wtru", "wall"],
            global_mle=np.ones(5, dtype=np.float64),
            global_cilow=np.zeros(5, dtype=np.float64),
            global_cihigh=np.full(5, np.inf, dtype=np.float64),
            global_n_obs=np.zeros(5, dtype=np.float64),
            global_n_exp=np.zeros(5, dtype=np.float64),
            global_p_value=np.ones(5, dtype=np.float64),
            r_mle=np.zeros(192, dtype=np.float64),
            n_192=n_192_bg,
            L_192=L_192_bg,
            theta_syn=float(np.inf),
            alpha_syn=0.0,
            is_poisson_syn=True,
            beta_syn=np.array([-np.inf] + [0.0] * (n_coef - 1), dtype=np.float64),
            se_syn=np.full(n_coef, np.nan, dtype=np.float64),
            p_beta_syn=np.full(n_coef, np.nan, dtype=np.float64),
            coef_names=coef_names,
            theta_ind=float(np.inf),
            alpha_ind=0.0,
            is_poisson_ind=True,
            beta_ind=np.array([-np.inf] + [0.0] * (n_coef - 1), dtype=np.float64),
            excluded_samples=annot.excluded_samples,
            annot_muts=annot.annot_muts,
            converged=True,
            settings=eff_mode,
        )

    if total_coding_snvs > 0:
        # Global 192-rate + selection MLE and CIs
        global_fit = compute_global_dnds(
            n_obs[bg_mask],
            sub_ref.L_all[bg_mask],
            rate_model=rate_model,
            ci_level=ci_level,
            ci_method=ci_method,
        )
        r_mle = global_fit.r_mle  # (192,)

        # Per-gene neutral expected counts E_{g, c} = sum_j L_{g, j, c} * r_j
        exp_classes = np.einsum("gjk,j->gk", sub_ref.L_all.astype(np.float64), r_mle)
        exp_syn = exp_classes[:, 0]
        exp_mis = exp_classes[:, 1]
        exp_non = exp_classes[:, 2]
        exp_spl = exp_classes[:, 3]

        # Synonymous Negative Binomial covariate regression (with Poisson fallback)
        nb_syn_fit: NBCovariateFit = fit_nb_covariate_model(
            n_syn,
            exp_syn,
            cov_df,
            bg_mask,
            use_nb=use_nb,
            max_iter=eff_max_iter,
            tol=tol,
            settings=eff_mode,
        )

        # Gene-level SNV LRTs (CV + pure Poisson)
        gene_snv = compute_gene_tests_cv(
            n_classes,
            exp_classes,
            nb_syn_fit.rho_all,
            nb_syn_fit.theta,
            nb_syn_fit.is_poisson,
        )
    else:
        # Indel-only cohort (total_coding_snvs == 0, total_coding_indels > 0):
        # skip SNV 192-rate/NB fitting and return NaN gene-level SNV stats
        global_fit = compute_global_dnds(
            n_obs[bg_mask],
            sub_ref.L_all[bg_mask],
            rate_model=rate_model,
            ci_level=ci_level,
            ci_method=ci_method,
        )
        r_mle = global_fit.r_mle  # zeros(192)
        exp_classes = np.zeros((G, 4), dtype=np.float64)
        exp_syn = exp_classes[:, 0]
        exp_mis = exp_classes[:, 1]
        exp_non = exp_classes[:, 2]
        exp_spl = exp_classes[:, 3]

        cov_names = list(cov_df.columns) if cov_df is not None else []
        coef_names = ["(Intercept)", *cov_names]
        n_coef = len(coef_names)
        beta_zero = np.zeros(n_coef, dtype=np.float64)
        beta_zero[0] = -np.inf
        nb_syn_fit = NBCovariateFit(
            coef_names=coef_names,
            beta=beta_zero,
            se=np.full(n_coef, np.nan, dtype=np.float64),
            p_beta=np.full(n_coef, np.nan, dtype=np.float64),
            theta=float(np.inf),
            alpha=0.0,
            is_poisson=True,
            rho_all=np.zeros(G, dtype=np.float64),
            mu_all=np.zeros(G, dtype=np.float64),
            loglik=0.0,
            converged=True,
        )
        gene_snv = compute_gene_tests_cv(
            n_classes,
            exp_classes,
            nb_syn_fit.rho_all,
            nb_syn_fit.theta,
            nb_syn_fit.is_poisson,
        )

    # Indel NB regression and gene-level LRT
    w_ind_cv, p_ind_cv, q_ind_cv, mu_ind, nb_ind_fit = compute_indel_tests(
        n_ind,
        cds_lengths,
        cov_df,
        bg_mask,
        use_nb=use_nb,
        max_iter=eff_max_iter,
        tol=tol,
        settings=eff_mode,
    )

    # Length-proportional baseline expected indels before covariates
    bg_len_sum = float(np.sum(cds_lengths[bg_mask]))
    if bg_len_sum > 0.0 and total_coding_indels > 0:
        bg_ind_rate = float(np.sum(n_ind[bg_mask])) / bg_len_sum
        if bg_ind_rate == 0.0:
            bg_ind_rate = float(np.sum(n_ind)) / float(np.sum(cds_lengths))
        exp_ind = cds_lengths.astype(np.float64) * bg_ind_rate
    else:
        exp_ind = np.zeros(G, dtype=np.float64)

    # Global combined p-value: Fisher's 4-df method when indels exist, else p_all_cv
    p_all_cv = gene_snv["p_all_cv"]
    p_global_cv = p_all_cv.copy()
    if total_coding_indels > 0:
        both = np.isfinite(p_all_cv) & np.isfinite(p_ind_cv)
        if np.any(both):
            p_a = np.clip(p_all_cv[both], 1e-300, 1.0)
            p_i = np.clip(p_ind_cv[both], 1e-300, 1.0)
            x2 = -2.0 * (np.log(p_a) + np.log(p_i))
            p_global_cv[both] = stats.chi2.sf(x2, df=4)
        only_ind = (~np.isfinite(p_all_cv)) & np.isfinite(p_ind_cv)
        if np.any(only_ind):
            p_global_cv[only_ind] = p_ind_cv[only_ind]
    q_global_cv = bh_fdr_nan_aware(p_global_cv)

    theta_ind = nb_ind_fit.theta if nb_ind_fit is not None else float(np.inf)
    alpha_ind = nb_ind_fit.alpha if nb_ind_fit is not None else 0.0
    is_poisson_ind = nb_ind_fit.is_poisson if nb_ind_fit is not None else True
    beta_ind = (
        nb_ind_fit.beta
        if nb_ind_fit is not None
        else np.array([-np.inf] + [0.0] * (len(nb_syn_fit.coef_names) - 1), dtype=np.float64)
    )
    converged = bool(nb_syn_fit.converged and (nb_ind_fit.converged if nb_ind_fit is not None else True))

    return DnDScvResult(
        gene_names=list(sub_ref.gene_names),
        cds_lengths=cds_lengths,
        n_syn=n_syn.astype(np.int64),
        n_mis=n_mis.astype(np.int64),
        n_non=n_non.astype(np.int64),
        n_spl=n_spl.astype(np.int64),
        n_ind=n_ind.astype(np.int64),
        exp_syn=exp_syn,
        exp_mis=exp_mis,
        exp_non=exp_non,
        exp_spl=exp_spl,
        exp_ind=exp_ind,
        mu_syn=nb_syn_fit.mu_all,
        mu_ind=mu_ind,
        w_mis_cv=gene_snv["w_mis_cv"],
        w_non_cv=gene_snv["w_non_cv"],
        w_spl_cv=gene_snv["w_spl_cv"],
        w_ind_cv=w_ind_cv,
        w_mis_p=gene_snv["w_mis_p"],
        w_non_p=gene_snv["w_non_p"],
        w_spl_p=gene_snv["w_spl_p"],
        p_mis_cv=gene_snv["p_mis_cv"],
        p_non_cv=gene_snv["p_non_cv"],
        p_spl_cv=gene_snv["p_spl_cv"],
        p_tru_cv=gene_snv["p_tru_cv"],
        p_all_cv=p_all_cv,
        p_ind_cv=p_ind_cv,
        p_global_cv=p_global_cv,
        q_mis_cv=gene_snv["q_mis_cv"],
        q_non_cv=gene_snv["q_non_cv"],
        q_spl_cv=gene_snv["q_spl_cv"],
        q_tru_cv=gene_snv["q_tru_cv"],
        q_all_cv=gene_snv["q_all_cv"],
        q_ind_cv=q_ind_cv,
        q_global_cv=q_global_cv,
        p_mis_p=gene_snv["p_mis_p"],
        p_non_p=gene_snv["p_non_p"],
        p_spl_p=gene_snv["p_spl_p"],
        p_tru_p=gene_snv["p_tru_p"],
        p_all_p=gene_snv["p_all_p"],
        q_all_p=gene_snv["q_all_p"],
        global_names=global_fit.names,
        global_mle=global_fit.mle,
        global_cilow=global_fit.cilow,
        global_cihigh=global_fit.cihigh,
        global_n_obs=global_fit.n_obs,
        global_n_exp=global_fit.n_exp,
        global_p_value=global_fit.p_value,
        r_mle=r_mle,
        n_192=n_192_bg,
        L_192=L_192_bg,
        theta_syn=nb_syn_fit.theta,
        alpha_syn=nb_syn_fit.alpha,
        is_poisson_syn=nb_syn_fit.is_poisson,
        beta_syn=nb_syn_fit.beta,
        se_syn=nb_syn_fit.se,
        p_beta_syn=nb_syn_fit.p_beta,
        coef_names=nb_syn_fit.coef_names,
        theta_ind=theta_ind,
        alpha_ind=alpha_ind,
        is_poisson_ind=is_poisson_ind,
        beta_ind=beta_ind,
        excluded_samples=annot.excluded_samples,
        annot_muts=annot.annot_muts,
        converged=converged,
        settings=eff_mode,
    )


fit_dndscv = dndscv
