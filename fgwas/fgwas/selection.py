"""Cross-validated ridge penalty selection (-xv) and stepwise model building for fgwas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .abf import SettingsType, normalize_prior_variances, validate_settings
from .core import (
    FgwasResult,
    _evaluate_fgwas_objective,
    _extract_lnabf_and_annotations,
    _optimize_fgwas_arrays,
    _partition_blocks,
    _prepare_contiguous_blocks,
    fit_fgwas,
)


@dataclass
class CrossValFgwasResult:
    """Result of cross-validated ridge penalty selection (``fgwas -xv``)."""

    best_penalty: float
    best_cv_loglik: float
    penalties: tuple[float, ...]
    cv_logliks: np.ndarray
    n_folds: int
    best_fit: FgwasResult
    settings: SettingsType

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "penalty": list(self.penalties),
                "cv_loglik": self.cv_logliks,
                "is_best": [p == self.best_penalty for p in self.penalties],
            }
        )

    def summary(self) -> str:
        return (
            f"fgwas CV (settings={self.settings!r}, folds={self.n_folds}): "
            f"best_penalty={self.best_penalty:.4g} (cv_loglik={self.best_cv_loglik:.2f})\n"
            + self.best_fit.summary()
        )


@dataclass
class StepwiseFgwasResult:
    """Result of forward addition + cross-validated backward pruning in fgwas."""

    selected_annotations: tuple[str, ...]
    best_penalty: float
    best_cv_loglik: float | None
    final_fit: FgwasResult
    history: list[dict[str, object]]
    settings: SettingsType

    def to_dataframe(self) -> pd.DataFrame:
        return self.final_fit.to_dataframe()

    def history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def summary(self) -> str:
        return (
            f"fgwas Stepwise (settings={self.settings!r}): "
            f"selected={list(self.selected_annotations)}, penalty={self.best_penalty:.4g}\n"
            + self.final_fit.summary()
        )


def _make_block_folds(
    n_blocks: int,
    n_folds: int,
    seed: int,
    settings: SettingsType,
) -> np.ndarray:
    """Assign each of the ``J`` blocks to a fold in ``0 .. k_folds - 1``."""
    k_folds = min(int(n_folds), n_blocks)
    if k_folds < 2:
        raise ValueError(
            f"Cross-validation requires at least 2 genomic blocks and n_folds >= 2 "
            f"(got n_blocks={n_blocks}, n_folds={n_folds})."
        )
    if settings == "R_default":
        return np.arange(n_blocks, dtype=np.int64) % k_folds
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    fold_ids = np.empty(n_blocks, dtype=np.int64)
    fold_ids[perm] = np.arange(n_blocks, dtype=np.int64) % k_folds
    return fold_ids


def _cv_score_penalties(
    lnabf: np.ndarray,
    A_mat: np.ndarray,
    block_idx: np.ndarray,
    n_blocks: int,
    fine_map: bool,
    penalties: tuple[float, ...],
    fold_ids: np.ndarray,
) -> np.ndarray:
    """Compute total held-out unpenalized log-likelihood across folds for each penalty."""
    k_folds = int(np.max(fold_ids)) + 1
    cv_scores = np.zeros(len(penalties), dtype=np.float64)

    fold_splits = []
    for f in range(k_folds):
        te_blk_mask = fold_ids == f
        tr_blk_mask = ~te_blk_mask
        n_tr_b = int(np.sum(tr_blk_mask))
        n_te_b = int(np.sum(te_blk_mask))

        remap_tr = np.full(n_blocks, -1, dtype=np.int64)
        remap_tr[np.where(tr_blk_mask)[0]] = np.arange(n_tr_b, dtype=np.int64)
        remap_te = np.full(n_blocks, -1, dtype=np.int64)
        remap_te[np.where(te_blk_mask)[0]] = np.arange(n_te_b, dtype=np.int64)

        snp_tr = tr_blk_mask[block_idx]
        snp_te = ~snp_tr

        b_tr, a_tr, idx_tr, st_tr, _ = _prepare_contiguous_blocks(
            lnabf[snp_tr], A_mat[snp_tr], remap_tr[block_idx[snp_tr]], n_tr_b
        )
        b_te, a_te, idx_te, st_te, _ = _prepare_contiguous_blocks(
            lnabf[snp_te], A_mat[snp_te], remap_te[block_idx[snp_te]], n_te_b
        )
        fold_splits.append((b_tr, a_tr, idx_tr, n_tr_b, st_tr, b_te, a_te, idx_te, n_te_b, st_te))

    for p_i, pen in enumerate(penalties):
        tot_heldout = 0.0
        for b_tr, a_tr, idx_tr, nb_tr, st_tr, b_te, a_te, idx_te, nb_te, st_te in fold_splits:
            eta_f, gamma_f, _ = _optimize_fgwas_arrays(
                b_tr, a_tr, idx_tr, nb_tr, fine_map, pen, block_starts=st_tr
            )
            _, unpen_te, *_ = _evaluate_fgwas_objective(
                b_te, a_te, idx_te, nb_te, fine_map, 0.0, eta_f, gamma_f, st_te
            )
            tot_heldout += unpen_te
        cv_scores[p_i] = tot_heldout

    return cv_scores


def cross_validate_fgwas(
    df: pd.DataFrame,
    annotations: Sequence[str] | str,
    penalties: Sequence[float] | None = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    n_folds: int = 10,
    block_col: str | None = None,
    window_size: int = 5000,
    fine_map: bool = False,
    V: float | Sequence[float] | None = None,
    seed: int = 42,
    settings: SettingsType = "default",
) -> CrossValFgwasResult:
    """Select optimal ridge penalty ``lambda`` via block cross-validation (``fgwas -xv``)."""
    validate_settings(settings)
    if penalties is None:
        pen_tuple = (
            (0.2, 0.05, 0.1, 0.5, 1.0, 2.0)
            if settings == "R_default"
            else (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
        )
    else:
        pen_tuple = tuple(float(p) for p in penalties)
    if len(pen_tuple) == 0 or any((not np.isfinite(p) or p < 0.0) for p in pen_tuple):
        raise ValueError(
            f"penalties must be a non-empty sequence of non-negative floats, got {penalties}."
        )

    eff_V = (
        ((0.1,) if (settings == "R_default" and fine_map) else (0.01, 0.1, 0.5))
        if V is None
        else normalize_prior_variances(V)
    )
    lnabf, A_mat, annot_names = _extract_lnabf_and_annotations(df, annotations, eff_V, settings)
    block_idx, block_ids = _partition_blocks(df, block_col, window_size, fine_map)
    n_blocks = int(block_ids.shape[0])

    fold_ids = _make_block_folds(n_blocks, n_folds, seed=seed, settings=settings)
    k_folds = int(np.max(fold_ids)) + 1
    cv_scores = _cv_score_penalties(
        lnabf, A_mat, block_idx, n_blocks, fine_map, pen_tuple, fold_ids
    )

    best_idx = int(np.argmax(cv_scores))
    best_pen = pen_tuple[best_idx]
    best_cv = float(cv_scores[best_idx])

    best_fit = fit_fgwas(
        df=df,
        annotations=annot_names,
        block_col=block_col,
        window_size=window_size,
        fine_map=fine_map,
        penalty=best_pen,
        V=eff_V,
        compute_ci=True,
        settings=settings,
    )
    return CrossValFgwasResult(
        best_penalty=best_pen,
        best_cv_loglik=best_cv,
        penalties=pen_tuple,
        cv_logliks=cv_scores,
        n_folds=k_folds,
        best_fit=best_fit,
        settings=settings,
    )


def stepwise_fgwas(
    df: pd.DataFrame,
    candidate_annotations: Sequence[str],
    block_col: str | None = None,
    window_size: int = 5000,
    fine_map: bool = False,
    penalties: Sequence[float] = (0.05, 0.1, 0.2, 0.5, 1.0),
    n_folds: int = 10,
    llk_threshold: float | None = None,
    V: float | Sequence[float] | None = None,
    seed: int = 42,
    settings: SettingsType = "default",
) -> StepwiseFgwasResult:
    """Build a multi-annotation fgwas model via forward addition + CV backward pruning."""
    validate_settings(settings)
    candidates = [str(c) for c in candidate_annotations]
    if len(candidates) == 0:
        raise ValueError("candidate_annotations must contain at least one annotation column.")

    eff_V = (
        ((0.1,) if (settings == "R_default" and fine_map) else (0.01, 0.1, 0.5))
        if V is None
        else normalize_prior_variances(V)
    )
    lnabf_raw, A_raw, cand_tuple = _extract_lnabf_and_annotations(df, candidates, eff_V, settings)
    col_map = {name: i for i, name in enumerate(cand_tuple)}

    b_raw, block_ids = _partition_blocks(df, block_col, window_size, fine_map)
    n_blocks = int(block_ids.shape[0])
    lnabf, A_full, block_idx, starts, _ = _prepare_contiguous_blocks(
        lnabf_raw, A_raw, b_raw, n_blocks
    )

    thresh = (
        float(llk_threshold)
        if llk_threshold is not None
        else (2.0 if settings == "R_default" else float(0.5 * stats.chi2.ppf(0.95, df=1)))
    )
    forward_pen = 0.0 if settings == "R_default" else 0.05

    selected: list[str] = []
    remaining: list[str] = list(cand_tuple)
    empty_A = np.empty((len(df), 0), dtype=np.float64)
    eta_null, g_null, _ = _optimize_fgwas_arrays(
        lnabf, empty_A, block_idx, n_blocks, fine_map, 0.0, block_starts=starts
    )
    _, curr_unpen_llk, *_ = _evaluate_fgwas_objective(
        lnabf, empty_A, block_idx, n_blocks, fine_map, 0.0, eta_null, g_null, starts
    )
    history: list[dict[str, object]] = [
        {"phase": "init", "annotations": (), "unpen_loglik": curr_unpen_llk, "delta_llk": 0.0}
    ]

    while remaining:
        best_cand: str | None = None
        best_llk = -np.inf
        for cand in remaining:
            trial_cols = [col_map[a] for a in (*selected, cand)]
            A_trial = A_full[:, trial_cols]
            eta_t, g_t, _ = _optimize_fgwas_arrays(
                lnabf, A_trial, block_idx, n_blocks, fine_map, forward_pen, block_starts=starts
            )
            _, unpen_t, *_ = _evaluate_fgwas_objective(
                lnabf, A_trial, block_idx, n_blocks, fine_map, forward_pen, eta_t, g_t, starts
            )
            if unpen_t > best_llk:
                best_llk = unpen_t
                best_cand = cand

        delta = best_llk - curr_unpen_llk
        if best_cand is not None and delta > thresh:
            selected.append(best_cand)
            remaining.remove(best_cand)
            curr_unpen_llk = best_llk
            history.append(
                {
                    "phase": "forward_add",
                    "added": best_cand,
                    "annotations": tuple(selected),
                    "unpen_loglik": curr_unpen_llk,
                    "delta_llk": delta,
                }
            )
        else:
            break

    if len(selected) == 0:
        null_fit = fit_fgwas(
            df=df,
            annotations=[],
            block_col=block_col,
            window_size=window_size,
            fine_map=fine_map,
            penalty=0.0,
            V=eff_V,
            compute_ci=False,
            settings=settings,
        )
        return StepwiseFgwasResult(
            selected_annotations=(),
            best_penalty=0.0,
            best_cv_loglik=None,
            final_fit=null_fit,
            history=history,
            settings=settings,
        )

    pen_tuple = tuple(float(p) for p in penalties)
    if n_blocks >= 2 and n_folds >= 2 and len(pen_tuple) > 0:
        fold_ids = _make_block_folds(n_blocks, n_folds, seed=seed, settings=settings)
        sel_idx = [col_map[a] for a in selected]
        cv_vec = _cv_score_penalties(
            lnabf, A_full[:, sel_idx], block_idx, n_blocks, fine_map, pen_tuple, fold_ids
        )
        best_p_idx = int(np.argmax(cv_vec))
        best_pen = pen_tuple[best_p_idx]
        best_cv = float(cv_vec[best_p_idx])
        history.append(
            {
                "phase": "cv_tune",
                "annotations": tuple(selected),
                "best_penalty": best_pen,
                "cv_loglik": best_cv,
            }
        )

        while len(selected) > 1:
            improved_drop: str | None = None
            improved_cv = best_cv
            improved_pen = best_pen
            for drop_cand in selected:
                sub_list = [a for a in selected if a != drop_cand]
                sub_cols = [col_map[a] for a in sub_list]
                sub_scores = _cv_score_penalties(
                    lnabf, A_full[:, sub_cols], block_idx, n_blocks, fine_map, pen_tuple, fold_ids
                )
                sub_best_idx = int(np.argmax(sub_scores))
                if float(sub_scores[sub_best_idx]) > improved_cv + 1e-6:
                    improved_cv = float(sub_scores[sub_best_idx])
                    improved_pen = pen_tuple[sub_best_idx]
                    improved_drop = drop_cand

            if improved_drop is not None:
                selected.remove(improved_drop)
                best_cv = improved_cv
                best_pen = improved_pen
                history.append(
                    {
                        "phase": "backward_prune",
                        "dropped": improved_drop,
                        "annotations": tuple(selected),
                        "best_penalty": best_pen,
                        "cv_loglik": best_cv,
                    }
                )
            else:
                break
    else:
        best_pen = 0.1 if settings == "default" else 0.0
        best_cv = None

    final_fit = fit_fgwas(
        df=df,
        annotations=selected,
        block_col=block_col,
        window_size=window_size,
        fine_map=fine_map,
        penalty=best_pen,
        V=eff_V,
        compute_ci=True,
        settings=settings,
    )
    return StepwiseFgwasResult(
        selected_annotations=tuple(selected),
        best_penalty=best_pen,
        best_cv_loglik=best_cv,
        final_fit=final_fit,
        history=history,
        settings=settings,
    )
