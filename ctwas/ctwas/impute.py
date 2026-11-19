"""
S-PrediXcan / FUSION molecular trait Z-score imputation and exact joint
(genes + cis-SNPs) correlation matrix construction for cTWAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

SettingsMode = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsMode:
    """Validate mandatory ``settings`` parameter."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


@dataclass
class ImputedRegionData:
    """Imputed molecular-trait + cis-SNP regional summary data for cTWAS."""

    region_id: str
    gene_z_df: pd.DataFrame
    snp_z_df: pd.DataFrame
    z_joint: np.ndarray
    R_joint: np.ndarray
    feature_ids: list[str]
    is_gene: np.ndarray
    is_cis_snp: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation accepted by ``estimate_ctwas_priors`` and ``run_ctwas``."""
        return {
            "region_id": self.region_id,
            "z_joint": self.z_joint,
            "R_joint": self.R_joint,
            "feature_ids": list(self.feature_ids),
            "is_gene": self.is_gene,
            "is_cis_snp": self.is_cis_snp,
        }


def _extract_snp_ids_and_z(gwas_z_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    if not isinstance(gwas_z_df, pd.DataFrame):
        raise TypeError("gwas_z_df must be a pandas DataFrame.")
    if len(gwas_z_df) == 0:
        raise ValueError("gwas_z_df must contain at least 1 SNP.")

    id_col = next(
        (c for c in ("snp", "snp_id", "rsid", "id", "variant_id") if c in gwas_z_df.columns),
        None,
    )
    if id_col is not None:
        snp_ids = gwas_z_df[id_col].astype(str).tolist()
    else:
        snp_ids = [str(idx) for idx in gwas_z_df.index]

    if "z" in gwas_z_df.columns:
        z_snp = gwas_z_df["z"].to_numpy(dtype=float)
    elif "Z" in gwas_z_df.columns:
        z_snp = gwas_z_df["Z"].to_numpy(dtype=float)
    elif "beta" in gwas_z_df.columns and "se" in gwas_z_df.columns:
        se = gwas_z_df["se"].to_numpy(dtype=float)
        if np.any(se <= 0.0):
            raise ValueError("GWAS 'se' column must be strictly positive.")
        z_snp = gwas_z_df["beta"].to_numpy(dtype=float) / se
    else:
        raise ValueError(
            "gwas_z_df must contain column 'z' (or both 'beta' and 'se')."
        )

    if np.any(~np.isfinite(z_snp)):
        raise ValueError("GWAS SNP Z-scores must be finite.")
    return snp_ids, z_snp


def _parse_weights_list(
    weights_list: list[dict[str, Any]] | dict[str, Any],
    snp_ids: list[str],
) -> tuple[list[str], np.ndarray]:
    """Parse flexible cis-prediction weight formats into ``(gene_ids, W_matrix)`` of shape ``(P_snp, G)``."""
    p_snp = len(snp_ids)
    snp_to_idx = {sid: i for i, sid in enumerate(snp_ids)}

    entries: list[tuple[str, Any]] = []
    if isinstance(weights_list, dict):
        for g_id, spec in weights_list.items():
            entries.append((str(g_id), spec))
    elif isinstance(weights_list, list):
        for k, item in enumerate(weights_list):
            if not isinstance(item, dict):
                raise TypeError("Elements of weights_list must be dicts.")
            g_id = str(item.get("gene_id", item.get("gene", item.get("id", f"gene_{k + 1}"))))
            entries.append((g_id, item))
    else:
        raise TypeError("weights_list must be a list of dicts or a dict keyed by gene_id.")

    gene_ids: list[str] = []
    cols: list[np.ndarray] = []

    for g_id, spec in entries:
        w_vec = np.zeros(p_snp, dtype=float)
        if isinstance(spec, np.ndarray):
            arr = np.asarray(spec, dtype=float)
            if arr.shape != (p_snp,):
                raise ValueError(
                    f"Weight vector for gene {g_id!r} has shape {arr.shape}, expected ({p_snp},)."
                )
            w_vec = arr.copy()
        elif isinstance(spec, pd.Series):
            for sid, val in spec.items():
                if str(sid) in snp_to_idx:
                    w_vec[snp_to_idx[str(sid)]] = float(val)
        elif isinstance(spec, pd.DataFrame):
            s_col = next((c for c in ("snp", "snp_id", "rsid", "id") if c in spec.columns), None)
            w_col = next((c for c in ("weight", "weights", "w", "beta") if c in spec.columns), None)
            if w_col is None:
                raise ValueError(f"Weight DataFrame for {g_id!r} must contain a 'weight' column.")
            s_iter = spec[s_col].astype(str) if s_col is not None else spec.index.astype(str)
            for sid, val in zip(s_iter, spec[w_col].to_numpy(dtype=float)):
                if sid in snp_to_idx:
                    w_vec[snp_to_idx[sid]] = float(val)
        elif isinstance(spec, dict):
            if "snps" in spec and "weights" in spec:
                s_list = [str(s) for s in spec["snps"]]
                w_vals = np.asarray(spec["weights"], dtype=float)
                if len(s_list) != len(w_vals):
                    raise ValueError(
                        f"Gene {g_id!r} has mismatched 'snps' ({len(s_list)}) and 'weights' ({len(w_vals)})."
                    )
                for sid, val in zip(s_list, w_vals):
                    if sid in snp_to_idx:
                        w_vec[snp_to_idx[sid]] = float(val)
            elif "weights" in spec:
                raw_w = spec["weights"]
                if isinstance(raw_w, dict):
                    for sid, val in raw_w.items():
                        if str(sid) in snp_to_idx:
                            w_vec[snp_to_idx[str(sid)]] = float(val)
                else:
                    arr = np.asarray(raw_w, dtype=float)
                    if arr.shape != (p_snp,):
                        raise ValueError(
                            f"Gene {g_id!r} weight array length {arr.shape} != {p_snp}."
                        )
                    w_vec = arr.copy()
            else:
                for sid, val in spec.items():
                    if sid in ("gene_id", "gene", "id"):
                        continue
                    if str(sid) in snp_to_idx:
                        w_vec[snp_to_idx[str(sid)]] = float(val)
        else:
            raise TypeError(f"Unsupported weight specification for gene {g_id!r}: {type(spec)}.")

        if np.any(np.abs(w_vec) > 0.0):
            gene_ids.append(g_id)
            cols.append(w_vec)

    if cols:
        w_mat = np.column_stack(cols)
    else:
        w_mat = np.zeros((p_snp, 0), dtype=float)
    return gene_ids, w_mat


def impute_expr_z(
    gwas_z_df: pd.DataFrame,
    ld_R: np.ndarray | pd.DataFrame,
    weights_list: list[dict[str, Any]] | dict[str, Any],
    region_id: str = "region_1",
    settings: SettingsMode = "default",
) -> ImputedRegionData:
    """Impute molecular trait Z-scores and exact joint ``(genes + SNPs)`` LD matrix.

    Computes S-PrediXcan / FUSION imputed Z-scores
    ``z_g = (w_g^T z_S) / sqrt(w_g^T R_S w_g)`` and constructs the block correlation
    matrix ``R_tilde`` over ``G`` imputed genes followed by ``P_S`` cis-region SNPs:
    ``R_gg' = w_g^T R_S w_g' / (sigma_g sigma_g')`` and
    ``R_gs = w_g^T R_{S,.s} / sigma_g``.
    """
    validate_settings(settings)
    snp_ids, z_snp = _extract_snp_ids_and_z(gwas_z_df)
    p_snp = len(snp_ids)

    if isinstance(ld_R, pd.DataFrame):
        if set(snp_ids).issubset(set(ld_R.index.astype(str))) and set(snp_ids).issubset(
            set(ld_R.columns.astype(str))
        ):
            ld_df = ld_R.copy()
            ld_df.index = ld_df.index.astype(str)
            ld_df.columns = ld_df.columns.astype(str)
            r_snp = ld_df.loc[snp_ids, snp_ids].to_numpy(dtype=float)
        else:
            r_snp = ld_R.to_numpy(dtype=float)
    else:
        r_snp = np.asarray(ld_R, dtype=float)

    if r_snp.shape != (p_snp, p_snp):
        raise ValueError(
            f"ld_R must have shape ({p_snp}, {p_snp}), got {r_snp.shape}."
        )
    r_snp = 0.5 * (r_snp + r_snp.T)
    np.fill_diagonal(r_snp, 1.0)

    gene_ids, w_mat = _parse_weights_list(weights_list, snp_ids)
    n_genes = w_mat.shape[1]

    if n_genes > 0:
        rw = r_snp @ w_mat  # (P_S, G)
        expr_vars = np.einsum("sg,sg->g", w_mat, rw)
        valid_g = expr_vars > 1e-12
        if not np.all(valid_g):
            keep_idx = np.flatnonzero(valid_g)
            gene_ids = [gene_ids[int(i)] for i in keep_idx]
            w_mat = w_mat[:, keep_idx]
            rw = rw[:, keep_idx]
            expr_vars = expr_vars[keep_idx]
            n_genes = len(gene_ids)

    if n_genes > 0:
        expr_sd = np.sqrt(expr_vars)
        w_norm = w_mat / expr_sd[None, :]  # (P_S, G)
        z_gene = w_norm.T @ z_snp
        r_gg = (w_norm.T @ r_snp) @ w_norm
        r_gg = 0.5 * (r_gg + r_gg.T)
        np.fill_diagonal(r_gg, 1.0)
        r_gs = (r_snp @ w_norm).T  # (G, P_S)
        r_joint = np.block([[r_gg, r_gs], [r_gs.T, r_snp]])
        r_joint = np.clip(0.5 * (r_joint + r_joint.T), -1.0, 1.0)
        np.fill_diagonal(r_joint, 1.0)
        z_joint = np.concatenate([z_gene, z_snp])
        is_cis_snp = np.any(np.abs(w_mat) > 0.0, axis=1)
        n_w_snps = np.sum(np.abs(w_mat) > 0.0, axis=0).astype(int)
    else:
        expr_vars = np.zeros(0, dtype=float)
        z_gene = np.zeros(0, dtype=float)
        n_w_snps = np.zeros(0, dtype=int)
        z_joint = z_snp.copy()
        r_joint = r_snp.copy()
        is_cis_snp = np.zeros(p_snp, dtype=bool)

    gene_z_df = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "z": z_gene,
            "n_weight_snps": n_w_snps,
            "expr_var": expr_vars,
        }
    )
    snp_z_df = pd.DataFrame(
        {
            "snp_id": snp_ids,
            "z": z_snp,
            "is_cis_weight_snp": is_cis_snp,
        }
    )
    feature_ids = list(gene_ids) + list(snp_ids)
    is_gene = np.array([True] * n_genes + [False] * p_snp, dtype=bool)

    return ImputedRegionData(
        region_id=str(region_id),
        gene_z_df=gene_z_df,
        snp_z_df=snp_z_df,
        z_joint=z_joint,
        R_joint=r_joint,
        feature_ids=feature_ids,
        is_gene=is_gene,
        is_cis_snp=is_cis_snp,
    )
