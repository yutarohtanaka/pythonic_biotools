import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Invalid settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


DEFAULT_GIBBS_CONTROL = {
    "chain_length": 1000,
    "burn_in": 500,
    "thinning": 2,
    "alpha": 1.0,
    "seed": 123,
}

DEFAULT_OPT_CONTROL_MAP = {
    "maxiter": 100000,
    "optimizer": "MAP",
    "sigma": 2.0,
}

DEFAULT_OPT_CONTROL_ST = {
    "maxiter": 100000,
    "optimizer": "MLE",
    "sigma": 2.0,
}


def _norm_to_one(ref: np.ndarray, pseudo_min: float = 1e-8) -> np.ndarray:
    """
    Normalize expression matrix to sum to 1 for each row,
    with zero entries replaced by pseudo_min.
    """
    G = ref.shape[1]
    row_sums = ref.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)

    phi = (ref / row_sums) * (1.0 - pseudo_min * G) + pseudo_min

    min_val = ref.min(axis=1)
    which_row = min_val > 0
    if np.any(which_row):
        phi[which_row] = ref[which_row] / row_sums[which_row]

    return phi


def _collapse(ref: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sum up reads for each level in labels.
    Returns collapsed matrix and sorted unique labels array.
    """
    valid_idx = ~pd.isna(labels)
    if not np.all(valid_idx):
        warnings.warn("NA found in labels. Excluding these cells.")
        labels = labels[valid_idx]
        ref = ref[valid_idx]

    uniq_labels = np.unique(labels)
    ref_collapsed = np.zeros((len(uniq_labels), ref.shape[1]), dtype=float)

    for i, label in enumerate(uniq_labels):
        mask = labels == label
        if sparse.issparse(ref):
            ref_collapsed[i] = np.asarray(ref[mask].sum(axis=0)).ravel()
        else:
            ref_collapsed[i] = np.sum(ref[mask], axis=0)

    return ref_collapsed, uniq_labels


def _filter_bulk_outlier(
    mixture: np.ndarray, outlier_cut: float, outlier_fraction: float
) -> np.ndarray:
    """
    Filter genes in mixture whose expression fraction exceeds outlier_cut
    in more than outlier_fraction of bulk samples.
    """
    mixture_norm = mixture / np.maximum(mixture.sum(axis=1, keepdims=True), 1.0)
    outlier_idx = np.mean(mixture_norm > outlier_cut, axis=0) > outlier_fraction
    return ~outlier_idx


class Prism:
    def __init__(
        self,
        reference: Union[ad.AnnData, pd.DataFrame, np.ndarray],
        mixture: Union[ad.AnnData, pd.DataFrame, np.ndarray],
        cell_type_labels: Union[str, np.ndarray, List[str]],
        cell_state_labels: Optional[Union[str, np.ndarray, List[str]]] = None,
        key: Optional[str] = None,
        outlier_cut: float = 0.01,
        outlier_fraction: float = 0.1,
        pseudo_min: float = 1e-8,
        gene_names_ref: Optional[np.ndarray] = None,
        gene_names_mix: Optional[np.ndarray] = None,
        sample_names_mix: Optional[np.ndarray] = None,
        settings: SettingsMode = "default",
    ):
        """
        Construct a Prism object from single-cell reference and bulk/spatial mixture counts.
        """
        validate_settings(settings)
        self.settings: SettingsMode = settings
        self.key = key
        self.pseudo_min = pseudo_min

        # 1. Parse reference
        if isinstance(reference, ad.AnnData):
            ref_mat = reference.X
            if isinstance(cell_type_labels, str):
                ct_labels = reference.obs[cell_type_labels].values
            else:
                ct_labels = np.asarray(cell_type_labels)

            if cell_state_labels is None:
                cs_labels = ct_labels
            elif isinstance(cell_state_labels, str):
                cs_labels = reference.obs[cell_state_labels].values
            else:
                cs_labels = np.asarray(cell_state_labels)

            ref_genes = np.asarray(reference.var_names)
        elif isinstance(reference, pd.DataFrame):
            ref_mat = reference.values
            ct_labels = np.asarray(cell_type_labels)
            cs_labels = (
                np.asarray(cell_state_labels)
                if cell_state_labels is not None
                else ct_labels
            )
            ref_genes = np.asarray(reference.columns)
        else:
            ref_mat = np.asarray(reference)
            ct_labels = np.asarray(cell_type_labels)
            cs_labels = (
                np.asarray(cell_state_labels)
                if cell_state_labels is not None
                else ct_labels
            )
            ref_genes = (
                np.asarray(gene_names_ref)
                if gene_names_ref is not None
                else np.array([f"gene_{i}" for i in range(ref_mat.shape[1])])
            )

        # 2. Parse mixture
        if isinstance(mixture, ad.AnnData):
            mix_mat = mixture.X
            mix_genes = np.asarray(mixture.var_names)
            self.sample_names = np.asarray(mixture.obs_names).astype(str)
        elif isinstance(mixture, pd.DataFrame):
            mix_mat = mixture.values
            mix_genes = np.asarray(mixture.columns)
            self.sample_names = np.asarray(mixture.index).astype(str)
        else:
            mix_mat = np.asarray(mixture)
            if mix_mat.ndim == 1:
                mix_mat = mix_mat.reshape(1, -1)
            mix_genes = (
                np.asarray(gene_names_mix)
                if gene_names_mix is not None
                else np.array([f"gene_{i}" for i in range(mix_mat.shape[1])])
            )
            self.sample_names = (
                np.asarray(sample_names_mix).astype(str)
                if sample_names_mix is not None
                else np.array([f"mixture_{i}" for i in range(mix_mat.shape[0])])
            )

        if mix_mat.ndim == 1:
            mix_mat = mix_mat.reshape(1, -1)

        if sparse.issparse(ref_mat):
            ref_mat = ref_mat.toarray()
        else:
            ref_mat = np.asarray(ref_mat, dtype=float)

        if sparse.issparse(mix_mat):
            mix_mat = mix_mat.toarray()
        else:
            mix_mat = np.asarray(mix_mat, dtype=float)

        ref_mat = np.asarray(ref_mat, dtype=float)
        mix_mat = np.asarray(mix_mat, dtype=float)

        # Explicit NaN / Inf / negative input validation
        if np.isnan(ref_mat).any() or np.isnan(mix_mat).any():
            raise ValueError(
                "Input contains NaN values. Please remove or impute missing values."
            )
        if not np.isfinite(ref_mat).all() or not np.isfinite(mix_mat).all():
            raise ValueError("Input contains infinite values.")
        if np.min(ref_mat) < 0 or np.min(mix_mat) < 0:
            raise ValueError(
                "Input contains negative values. Please provide untransformed raw counts."
            )

        # Filter cells with missing labels before state-to-type validation
        valid_cells = (~pd.isna(ct_labels)) & (~pd.isna(cs_labels))
        if not np.all(valid_cells):
            warnings.warn("NA found in cell labels. Excluding these cells.")
            ref_mat = ref_mat[valid_cells]
            ct_labels = ct_labels[valid_cells]
            cs_labels = cs_labels[valid_cells]

        df_map = pd.DataFrame({"ct": ct_labels, "cs": cs_labels})
        if df_map.groupby("cs")["ct"].nunique().max() > 1:
            raise ValueError("One or more cell states belong to multiple cell types!")

        # Remove outlier genes in bulk
        keep_bulk_genes = _filter_bulk_outlier(mix_mat, outlier_cut, outlier_fraction)
        mix_mat = mix_mat[:, keep_bulk_genes]
        mix_genes = mix_genes[keep_bulk_genes]

        # Remove all-zero genes in reference
        keep_ref_genes = np.asarray(ref_mat.sum(axis=0) > 0).ravel()
        ref_mat = ref_mat[:, keep_ref_genes]
        ref_genes = ref_genes[keep_ref_genes]

        # Intersect genes
        shared_genes, ref_idx, mix_idx = np.intersect1d(
            ref_genes, mix_genes, return_indices=True
        )
        if len(shared_genes) == 0:
            raise ValueError("Gene names of reference and mixture do not match!")
        if len(shared_genes) < 100:
            warnings.warn("Very few genes from reference and mixture match!")

        ref_mat = ref_mat[:, ref_idx]
        self.mixture = mix_mat[:, mix_idx]
        self.gene_names = shared_genes.astype(str)

        # Collapse raw reference counts and normalize
        ref_cs, self.cs_names = _collapse(ref_mat, cs_labels)
        ref_ct, self.ct_names = _collapse(ref_mat, ct_labels)

        if self.key is not None and self.key not in set(self.ct_names):
            raise ValueError(
                f"Specified malignant key '{self.key}' not found in cell_type_labels: {list(self.ct_names)}"
            )

        self.ref_cs_raw = ref_cs
        self.ref_ct_raw = ref_ct
        self.phi_cellState = _norm_to_one(ref_cs, pseudo_min)
        self.phi_cellType = _norm_to_one(ref_ct, pseudo_min)

        self.map_dict: Dict[str, List[str]] = (
            df_map.drop_duplicates().groupby("ct")["cs"].apply(list).to_dict()
        )

    def _resolve_settings(self, settings: SettingsMode) -> SettingsMode:
        validate_settings(settings)
        if settings == "R_default":
            self.settings = "R_default"
            return "R_default"
        return self.settings

    def run(
        self,
        n_cores: int = 1,
        update_gibbs: bool = True,
        gibbs_control: Optional[dict] = None,
        opt_control: Optional[dict] = None,
        settings: SettingsMode = "default",
    ):
        """
        Run BayesPrism bulk deconvolution.
        """
        from .gibbs import (
            _compute_cv_from_draws,
            run_gibbs,
            run_gibbs_tumor,
            update_reference,
        )

        eff_settings = self._resolve_settings(settings)
        use_r = eff_settings == "R_default"

        gc = {**DEFAULT_GIBBS_CONTROL, **(gibbs_control or {})}
        oc = {**DEFAULT_OPT_CONTROL_MAP, **(opt_control or {})}

        if use_r:
            Z_cs, theta_cs, draws_cs = run_gibbs(
                X=self.mixture,
                phi=self.phi_cellState,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
                return_draws=True,
            )
            theta_cs_cv = _compute_cv_from_draws(draws_cs)
            self.posterior_initial_cellState = {
                "Z": Z_cs,
                "theta": theta_cs,
                "theta_cv": theta_cs_cv,
            }
        else:
            draws_cs = None
            Z_cs, theta_cs = run_gibbs(
                X=self.mixture,
                phi=self.phi_cellState,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
            )
            self.posterior_initial_cellState = {"Z": Z_cs, "theta": theta_cs}

        # Merge cell states to cell types in canonical self.ct_names order
        N, G, _ = Z_cs.shape
        K_ct = len(self.ct_names)
        Z_ct = np.zeros((N, G, K_ct), dtype=float)
        theta_ct = np.zeros((N, K_ct), dtype=float)
        draws_ct = (
            np.zeros((N, draws_cs.shape[1], K_ct), dtype=float)
            if draws_cs is not None
            else None
        )

        cs_to_idx = {name: i for i, name in enumerate(self.cs_names)}
        for k, ct in enumerate(self.ct_names):
            cs_indices = [cs_to_idx[cs] for cs in self.map_dict[ct]]
            Z_ct[:, :, k] = Z_cs[:, :, cs_indices].sum(axis=2)
            theta_ct[:, k] = theta_cs[:, cs_indices].sum(axis=1)
            if draws_ct is not None and draws_cs is not None:
                draws_ct[:, :, k] = draws_cs[:, :, cs_indices].sum(axis=2)

        if use_r and draws_ct is not None:
            theta_ct_cv = _compute_cv_from_draws(draws_ct)
            self.posterior_initial_cellType = {
                "Z": Z_ct,
                "theta": theta_ct,
                "theta_cv": theta_ct_cv,
            }
        else:
            theta_ct_cv = None
            self.posterior_initial_cellType = {"Z": Z_ct, "theta": theta_ct}

        if not update_gibbs:
            if use_r and theta_ct_cv is not None:
                self.theta_cv = theta_ct_cv
                self.posterior_theta_cv = theta_ct_cv
            return self

        psi = update_reference(
            Z_ct=Z_ct,
            phi_prime=self.phi_cellType,
            ct_names=self.ct_names,
            key=self.key,
            opt_control=oc,
            pseudo_min=self.pseudo_min,
            n_cores=n_cores,
        )
        self.reference_update = psi

        if self.key is None:
            if use_r:
                _, theta_f, draws_f = run_gibbs(
                    X=self.mixture,
                    phi=psi,
                    alpha=gc["alpha"],
                    chain_length=gc["chain_length"],
                    burn_in=gc["burn_in"],
                    thinning=gc["thinning"],
                    seed=gc["seed"],
                    n_cores=n_cores,
                    sample_Z=False,
                    return_draws=True,
                )
                theta_f_cv = _compute_cv_from_draws(draws_f)
            else:
                theta_f_cv = None
                _, theta_f = run_gibbs(
                    X=self.mixture,
                    phi=psi,
                    alpha=gc["alpha"],
                    chain_length=gc["chain_length"],
                    burn_in=gc["burn_in"],
                    thinning=gc["thinning"],
                    seed=gc["seed"],
                    n_cores=n_cores,
                    sample_Z=False,
                )
        else:
            key_idx = list(self.ct_names).index(self.key)
            if use_r:
                theta_f, theta_f_cv = run_gibbs_tumor(
                    X=self.mixture,
                    psi_mal=psi["psi_mal"],
                    psi_env=psi["psi_env"],
                    alpha=gc["alpha"],
                    chain_length=gc["chain_length"],
                    burn_in=gc["burn_in"],
                    thinning=gc["thinning"],
                    seed=gc["seed"],
                    n_cores=n_cores,
                    key_idx=key_idx,
                    return_cv=True,
                )
            else:
                theta_f_cv = None
                theta_f = run_gibbs_tumor(
                    X=self.mixture,
                    psi_mal=psi["psi_mal"],
                    psi_env=psi["psi_env"],
                    alpha=gc["alpha"],
                    chain_length=gc["chain_length"],
                    burn_in=gc["burn_in"],
                    thinning=gc["thinning"],
                    seed=gc["seed"],
                    n_cores=n_cores,
                    key_idx=key_idx,
                )

        self.posterior_theta_f = theta_f
        if use_r and theta_f_cv is not None:
            self.theta_cv = theta_f_cv
            self.posterior_theta_cv = theta_f_cv
        return self

    def get_results(self, output_type: str = "anndata"):
        """
        Return deconvolved cell-type proportions aligned with self.ct_names.
        output_type: 'anndata' or 'dataframe'.
        """
        if hasattr(self, "posterior_theta_f"):
            res_theta = self.posterior_theta_f
        else:
            res_theta = self.posterior_initial_cellType["theta"]

        df = pd.DataFrame(res_theta, index=self.sample_names, columns=self.ct_names)

        if output_type == "dataframe":
            return df
        elif output_type == "anndata":
            adata = ad.AnnData(X=self.mixture.copy(), obs=df)
            adata.var_names = self.gene_names
            return adata
        else:
            raise ValueError("output_type must be 'anndata' or 'dataframe'")

    def get_exp(self, state_or_type: str, cell_name: str) -> np.ndarray:
        """
        Extract posterior mean sample-specific gene expression profile (N x G)
        for a given cell state or cell type.
        """
        if state_or_type == "state":
            idx = list(self.cs_names).index(cell_name)
            if hasattr(self, "posterior_cellState"):
                return self.posterior_cellState["Z"][:, :, idx]
            return self.posterior_initial_cellState["Z"][:, :, idx]
        elif state_or_type == "type":
            idx = list(self.ct_names).index(cell_name)
            if hasattr(self, "posterior_cellType"):
                return self.posterior_cellType["Z"][:, :, idx]
            return self.posterior_initial_cellType["Z"][:, :, idx]
        else:
            raise ValueError("state_or_type must be 'state' or 'type'")

    def run_st(
        self,
        n_cores: int = 1,
        gibbs_control: Optional[dict] = None,
        opt_control: Optional[dict] = None,
        settings: SettingsMode = "default",
    ):
        """
        Run BayesPrism spatial transcriptomics deconvolution.
        Uses shared-gamma MLE ('optimizer': 'MLE') across cell states by default.
        """
        from .gibbs import _compute_cv_from_draws, run_gibbs, update_reference

        eff_settings = self._resolve_settings(settings)
        use_r = eff_settings == "R_default"

        gc = {**DEFAULT_GIBBS_CONTROL, **(gibbs_control or {})}
        oc = {**DEFAULT_OPT_CONTROL_ST, **(opt_control or {})}

        if use_r:
            Z_cs, theta_cs, draws_cs = run_gibbs(
                X=self.mixture,
                phi=self.phi_cellState,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
                return_draws=True,
            )
            self.posterior_initial_cellState = {
                "Z": Z_cs,
                "theta": theta_cs,
                "theta_cv": _compute_cv_from_draws(draws_cs),
            }
        else:
            Z_cs, theta_cs = run_gibbs(
                X=self.mixture,
                phi=self.phi_cellState,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
            )
            self.posterior_initial_cellState = {"Z": Z_cs, "theta": theta_cs}

        psi = update_reference(
            Z_ct=Z_cs,
            phi_prime=self.phi_cellState,
            ct_names=self.cs_names,
            key=None,
            opt_control=oc,
            pseudo_min=self.pseudo_min,
            n_cores=n_cores,
        )
        self.reference_update = psi

        if use_r:
            Z_cs_f, theta_cs_f, draws_cs_f = run_gibbs(
                X=self.mixture,
                phi=psi,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
                return_draws=True,
            )
            self.posterior_cellState = {
                "Z": Z_cs_f,
                "theta": theta_cs_f,
                "theta_cv": _compute_cv_from_draws(draws_cs_f),
            }
        else:
            draws_cs_f = None
            Z_cs_f, theta_cs_f = run_gibbs(
                X=self.mixture,
                phi=psi,
                alpha=gc["alpha"],
                chain_length=gc["chain_length"],
                burn_in=gc["burn_in"],
                thinning=gc["thinning"],
                seed=gc["seed"],
                n_cores=n_cores,
                sample_Z=True,
            )
            self.posterior_cellState = {"Z": Z_cs_f, "theta": theta_cs_f}

        N, G, _ = Z_cs_f.shape
        K_ct = len(self.ct_names)
        Z_ct = np.zeros((N, G, K_ct), dtype=float)
        theta_ct = np.zeros((N, K_ct), dtype=float)
        draws_ct_f = (
            np.zeros((N, draws_cs_f.shape[1], K_ct), dtype=float)
            if draws_cs_f is not None
            else None
        )

        cs_to_idx = {name: i for i, name in enumerate(self.cs_names)}
        for k, ct in enumerate(self.ct_names):
            cs_indices = [cs_to_idx[cs] for cs in self.map_dict[ct]]
            Z_ct[:, :, k] = Z_cs_f[:, :, cs_indices].sum(axis=2)
            theta_ct[:, k] = theta_cs_f[:, cs_indices].sum(axis=1)
            if draws_ct_f is not None and draws_cs_f is not None:
                draws_ct_f[:, :, k] = draws_cs_f[:, :, cs_indices].sum(axis=2)

        if use_r and draws_ct_f is not None:
            theta_ct_cv = _compute_cv_from_draws(draws_ct_f)
            self.posterior_cellType = {
                "Z": Z_ct,
                "theta": theta_ct,
                "theta_cv": theta_ct_cv,
            }
            self.theta_cv = theta_ct_cv
            self.posterior_theta_cv = theta_ct_cv
        else:
            self.posterior_cellType = {"Z": Z_ct, "theta": theta_ct}

        self.posterior_theta_f = theta_ct
        return self

    def learn_embedding_nmf(
        self,
        K: int,
        cycle: int = 50,
        n_cores: int = 1,
        gibbs_control: Optional[dict] = None,
        opt_control: Optional[dict] = None,
        nmf_control: Optional[dict] = None,
        settings: SettingsMode = "default",
    ):
        from .embedding import learn_embedding_nmf

        eff_settings = self._resolve_settings(settings)
        return learn_embedding_nmf(
            self,
            K,
            cycle,
            n_cores,
            gibbs_control,
            opt_control,
            nmf_control,
            settings=eff_settings,
        )
