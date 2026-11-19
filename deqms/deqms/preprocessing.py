"""PSM and peptide-level normalization and protein summarization for DEqMS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from ._validation import SettingsMode, validate_settings


def _safe_nanmedian_axis0(mat: np.ndarray) -> np.ndarray:
    """Column-wise nanmedian without emitting RuntimeWarning on all-NaN columns."""
    out = np.full(mat.shape[1], np.nan, dtype=np.float64)
    valid_cols = np.any(np.isfinite(mat), axis=0)
    if np.any(valid_cols):
        out[valid_cols] = np.nanmedian(mat[:, valid_cols], axis=0)
    return out


def _safe_nanmedian_axis1(mat: np.ndarray) -> np.ndarray:
    """Row-wise nanmedian without emitting RuntimeWarning on all-NaN rows."""
    out = np.full(mat.shape[0], np.nan, dtype=np.float64)
    valid_rows = np.any(np.isfinite(mat), axis=1)
    if np.any(valid_rows):
        out[valid_rows] = np.nanmedian(mat[valid_rows, :], axis=1)
    return out


def _safe_nanmedian_1d(vec: np.ndarray) -> float:
    """1D nanmedian returning 0.0 when no finite entry is present."""
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


@dataclass(frozen=True)
class ProteinSummarizationResult:
    """Protein-level abundance matrix paired with PSM/peptide counts.

    Supports both attribute access (``res.abundance``, ``res.count``) and
    2-tuple unpacking (``abundance_df, count_series = res``).
    """

    abundance: pd.DataFrame
    count: pd.Series

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        yield self.abundance
        yield self.count

    def to_dataframe(self, include_count: bool = False) -> pd.DataFrame:
        """Return protein abundance DataFrame, optionally appending ``count``."""
        df = self.abundance.copy()
        if include_count:
            df["count"] = self.count
        return df


def equal_median_normalization(
    df_or_mat: pd.DataFrame | np.ndarray | ProteinSummarizationResult,
    center_zero: bool = True,
    settings: SettingsMode = "default",
) -> pd.DataFrame | np.ndarray | ProteinSummarizationResult:
    """Equalize sample column medians across a log2 intensity/ratio matrix.

    Matches ``DEqMS::equalMedianNormalization`` by subtracting each sample
    column's median (centering each column median to ``0.0`` when
    ``center_zero=True``, or to the global median when ``center_zero=False``).
    """
    validate_settings(settings)

    if isinstance(df_or_mat, ProteinSummarizationResult):
        norm_df = equal_median_normalization(
            df_or_mat.abundance, center_zero=center_zero, settings=settings
        )
        assert isinstance(norm_df, pd.DataFrame)
        return ProteinSummarizationResult(
            abundance=norm_df, count=df_or_mat.count.copy()
        )

    is_df = isinstance(df_or_mat, pd.DataFrame)
    arr = np.asarray(df_or_mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
            "df_or_mat must be a non-empty 2D DataFrame or numpy.ndarray."
        )

    col_medians = _safe_nanmedian_axis0(arr)
    shift = col_medians.copy()
    if not center_zero:
        finite_meds = col_medians[np.isfinite(col_medians)]
        global_med = float(np.median(finite_meds)) if finite_meds.size > 0 else 0.0
        shift = col_medians - global_med

    out = arr - shift[np.newaxis, :]
    if is_df:
        assert isinstance(df_or_mat, pd.DataFrame)
        return pd.DataFrame(out, index=df_or_mat.index, columns=df_or_mat.columns)
    return out


def median_sweeping(
    psm_df: pd.DataFrame,
    group_col: str = "protein",
    sample_cols: list[str] | None = None,
    ref_cols: list[str] | str | None = None,
    log_scaled: bool = True,
    settings: SettingsMode = "default",
) -> ProteinSummarizationResult:
    """Summarize PSM/peptide intensities to protein abundances + PSM counts.

    Implements ``DEqMS::medianSweeping`` (Herbrich et al. 2013; Zhu et al. 2020):
    1. Optional ``log2`` transformation when ``log_scaled=False`` (non-positive
       intensities are treated as missing ``NaN``).
    2. Spectrum-row centering via median of ``ref_cols`` (if supplied) or row
       median across ``sample_cols``.
    3. Per-protein median summarization across mapped PSMs and exact per-protein
       quantified PSM count tracking.
    4. Sample-column median equalization to zero via ``equal_median_normalization``.
    """
    validate_settings(settings)

    if not isinstance(psm_df, pd.DataFrame) or psm_df.empty:
        raise ValueError("psm_df must be a non-empty pandas DataFrame.")
    if group_col not in psm_df.columns:
        raise ValueError(f"group_col={group_col!r} not found in psm_df.columns.")

    if isinstance(ref_cols, str):
        ref_list: list[str] | None = [ref_cols]
    elif ref_cols is not None:
        ref_list = list(ref_cols)
        if len(ref_list) == 0:
            raise ValueError("ref_cols must not be empty when specified.")
    else:
        ref_list = None

    if ref_list is not None:
        missing_ref = [c for c in ref_list if c not in psm_df.columns]
        if missing_ref:
            raise ValueError(f"ref_cols not found in psm_df: {missing_ref}")

    if sample_cols is None:
        excluded = {group_col}
        if ref_list is not None:
            excluded.update(ref_list)
        sample_list = [
            c
            for c in psm_df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(psm_df[c])
        ]
    else:
        sample_list = list(sample_cols)
        missing_samp = [c for c in sample_list if c not in psm_df.columns]
        if missing_samp:
            raise ValueError(f"sample_cols not found in psm_df: {missing_samp}")

    if len(sample_list) == 0:
        raise ValueError("No numeric sample_cols found in psm_df.")

    valid_group_mask = psm_df[group_col].notna()
    work_df = psm_df.loc[valid_group_mask]
    if work_df.empty:
        raise ValueError("No valid non-null protein identifiers in group_col.")

    sample_mat = work_df[sample_list].to_numpy(dtype=np.float64, copy=True)
    sample_mat[~np.isfinite(sample_mat)] = np.nan

    if not log_scaled:
        pos = sample_mat > 0.0
        safe = np.where(pos, sample_mat, 1.0)
        sample_mat = np.where(pos, np.log2(safe), np.nan)

    if ref_list is not None:
        ref_mat = work_df[ref_list].to_numpy(dtype=np.float64, copy=True)
        ref_mat[~np.isfinite(ref_mat)] = np.nan
        if not log_scaled:
            pos_ref = ref_mat > 0.0
            safe_ref = np.where(pos_ref, ref_mat, 1.0)
            ref_mat = np.where(pos_ref, np.log2(safe_ref), np.nan)
        row_ref = _safe_nanmedian_axis1(ref_mat)
    else:
        row_ref = _safe_nanmedian_axis1(sample_mat)

    swept_psm = sample_mat - row_ref[:, np.newaxis]
    row_has_obs = np.any(np.isfinite(swept_psm), axis=1)
    if not np.any(row_has_obs):
        raise ValueError("All PSM rows are completely missing after centering.")

    swept_psm = swept_psm[row_has_obs]
    groups = work_df[group_col].iloc[np.flatnonzero(row_has_obs)].astype(str).to_numpy()

    # Group indices preserving first-seen protein order
    unique_proteins, inverse = np.unique(groups, return_inverse=True)
    first_idx = np.zeros(len(unique_proteins), dtype=np.int64)
    seen = np.zeros(len(unique_proteins), dtype=bool)
    for pos_i, u_i in enumerate(inverse):
        if not seen[u_i]:
            first_idx[u_i] = pos_i
            seen[u_i] = True
    order = np.argsort(first_idx)
    ordered_proteins = unique_proteins[order]

    # Map each group smoothly via argsort of inverse
    remap = np.empty(len(unique_proteins), dtype=np.int64)
    remap[order] = np.arange(len(unique_proteins))
    group_ids = remap[inverse]

    sort_idx = np.argsort(group_ids, kind="mergesort")
    sorted_ids = group_ids[sort_idx]
    sorted_psm = swept_psm[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    splits = np.split(sorted_psm, boundaries, axis=0)

    n_prot = len(ordered_proteins)
    n_samp = len(sample_list)
    prot_mat = np.empty((n_prot, n_samp), dtype=np.float64)
    counts = np.empty(n_prot, dtype=np.int64)

    for g_idx, sub in enumerate(splits):
        counts[g_idx] = sub.shape[0]
        prot_mat[g_idx, :] = _safe_nanmedian_axis0(sub)

    norm_mat = equal_median_normalization(
        prot_mat, center_zero=True, settings=settings
    )
    assert isinstance(norm_mat, np.ndarray)

    prot_index = pd.Index(ordered_proteins, name=group_col)
    abundance_df = pd.DataFrame(norm_mat, index=prot_index, columns=sample_list)
    count_series = pd.Series(counts, index=prot_index, name="count", dtype=np.int64)
    return ProteinSummarizationResult(abundance=abundance_df, count=count_series)


def _tukey_median_polish(
    mat: np.ndarray, eps: float = 1e-4, max_iter: int = 50
) -> np.ndarray:
    """Two-way Tukey median polish returning ``overall + col_effects`` (length J)."""
    n_rows, n_cols = mat.shape
    col_has_obs = np.any(np.isfinite(mat), axis=0)

    if n_rows == 1:
        out = mat[0].copy()
        out[~col_has_obs] = np.nan
        return out

    resid = mat.astype(np.float64, copy=True)
    overall = 0.0
    row_eff = np.zeros(n_rows, dtype=np.float64)
    col_eff = np.zeros(n_cols, dtype=np.float64)
    old_sum = 0.0

    for _ in range(max_iter):
        # Sweep row medians
        r_delta = _safe_nanmedian_axis1(resid)
        r_delta = np.where(np.isfinite(r_delta), r_delta, 0.0)
        resid = resid - r_delta[:, np.newaxis]
        row_eff = row_eff + r_delta

        c_med = _safe_nanmedian_1d(col_eff[col_has_obs])
        col_eff = col_eff - c_med
        overall = overall + c_med

        # Sweep column medians
        c_delta = _safe_nanmedian_axis0(resid)
        c_delta = np.where(np.isfinite(c_delta), c_delta, 0.0)
        resid = resid - c_delta[np.newaxis, :]
        col_eff = col_eff + c_delta

        r_med = _safe_nanmedian_1d(row_eff)
        row_eff = row_eff - r_med
        overall = overall + r_med

        new_sum = float(np.nansum(np.abs(resid)))
        if new_sum == 0.0 or abs(new_sum - old_sum) < eps * new_sum:
            break
        old_sum = new_sum

    profile = overall + col_eff
    profile[~col_has_obs] = np.nan
    return profile


def peptide_profile_summarization(
    peptide_df: pd.DataFrame,
    group_col: str = "protein",
    sample_cols: list[str] | None = None,
    method: str = "tukey_median_polish",
    settings: SettingsMode = "default",
) -> ProteinSummarizationResult:
    """Summarize label-free / DIA log2 peptide intensities per protein.

    Implements ``DEqMS::peptideProfileSummarization`` via Tukey's two-way
    median polish (or per-sample ``"median"`` / ``"mean"``), preserving ``NaN``
    for samples where none of a protein's peptides were quantified.
    """
    validate_settings(settings)

    valid_methods = ("tukey_median_polish", "median", "mean")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method={method!r}. Expected one of {valid_methods}."
        )

    if not isinstance(peptide_df, pd.DataFrame) or peptide_df.empty:
        raise ValueError("peptide_df must be a non-empty pandas DataFrame.")
    if group_col not in peptide_df.columns:
        raise ValueError(f"group_col={group_col!r} not found in peptide_df.columns.")

    if sample_cols is None:
        sample_list = [
            c
            for c in peptide_df.columns
            if c != group_col and pd.api.types.is_numeric_dtype(peptide_df[c])
        ]
    else:
        sample_list = list(sample_cols)
        missing_samp = [c for c in sample_list if c not in peptide_df.columns]
        if missing_samp:
            raise ValueError(f"sample_cols not found in peptide_df: {missing_samp}")

    if len(sample_list) == 0:
        raise ValueError("No numeric sample_cols found in peptide_df.")

    valid_group = peptide_df[group_col].notna()
    work_df = peptide_df.loc[valid_group]
    if work_df.empty:
        raise ValueError("No valid non-null protein identifiers in group_col.")

    pep_mat = work_df[sample_list].to_numpy(dtype=np.float64, copy=True)
    pep_mat[~np.isfinite(pep_mat)] = np.nan
    row_has_obs = np.any(np.isfinite(pep_mat), axis=1)
    if not np.any(row_has_obs):
        raise ValueError("All peptide rows are completely missing.")

    pep_mat = pep_mat[row_has_obs]
    groups = work_df[group_col].iloc[np.flatnonzero(row_has_obs)].astype(str).to_numpy()

    unique_proteins, inverse = np.unique(groups, return_inverse=True)
    first_idx = np.zeros(len(unique_proteins), dtype=np.int64)
    seen = np.zeros(len(unique_proteins), dtype=bool)
    for pos_i, u_i in enumerate(inverse):
        if not seen[u_i]:
            first_idx[u_i] = pos_i
            seen[u_i] = True
    order = np.argsort(first_idx)
    ordered_proteins = unique_proteins[order]

    remap = np.empty(len(unique_proteins), dtype=np.int64)
    remap[order] = np.arange(len(unique_proteins))
    group_ids = remap[inverse]

    sort_idx = np.argsort(group_ids, kind="mergesort")
    sorted_ids = group_ids[sort_idx]
    sorted_pep = pep_mat[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    splits = np.split(sorted_pep, boundaries, axis=0)

    eps = 0.01 if settings == "R_default" else 1e-4
    max_iter = 10 if settings == "R_default" else 50

    n_prot = len(ordered_proteins)
    n_samp = len(sample_list)
    prot_mat = np.empty((n_prot, n_samp), dtype=np.float64)
    counts = np.empty(n_prot, dtype=np.int64)

    for g_idx, sub in enumerate(splits):
        counts[g_idx] = sub.shape[0]
        if method == "tukey_median_polish":
            prot_mat[g_idx, :] = _tukey_median_polish(sub, eps=eps, max_iter=max_iter)
        elif method == "median":
            prot_mat[g_idx, :] = _safe_nanmedian_axis0(sub)
        else:
            col_obs = np.any(np.isfinite(sub), axis=0)
            row_vals = np.full(n_samp, np.nan, dtype=np.float64)
            if np.any(col_obs):
                row_vals[col_obs] = np.nanmean(sub[:, col_obs], axis=0)
            prot_mat[g_idx, :] = row_vals

    prot_index = pd.Index(ordered_proteins, name=group_col)
    abundance_df = pd.DataFrame(prot_mat, index=prot_index, columns=sample_list)
    count_series = pd.Series(counts, index=prot_index, name="count", dtype=np.int64)
    return ProteinSummarizationResult(abundance=abundance_df, count=count_series)
