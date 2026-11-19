"""
Core Surrogate Variable Analysis estimators: `irwsva`, `twostepsva`, `sva`, and `svaseq`.

Implements:
- `SVAResult`: frozen result dataclass with `.to_dataframe()`,
  `.feature_posteriors_dataframe()`, `.augment_design()`, and `.summary()`.
- `irwsva`: Iteratively Reweighted SVA (Leek & Storey 2007).
- `twostepsva`: Two-Step SVA (Leek & Storey 2007).
- `sva`: Unified dispatcher supporting `"irw"`, `"two-step"`, and `"supervised"`.
- `svaseq`: Moderated log-count SVA for RNA-seq count matrices (Leek 2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._io import (
    coerce_design_matrix,
    coerce_expression_matrix,
    orthonormal_basis,
    validate_nested_models,
)
from .num_sv import NumSVResult, num_sv_arrays
from .stats import edge_lfdr, f_pvalue_arrays


@dataclass(frozen=True)
class SVAResult:
    """Typed result container for Surrogate Variable Analysis (`sva`, `irwsva`, `twostepsva`, `svaseq`)."""

    sv: np.ndarray
    n_sv: int
    pprob_gam: np.ndarray
    pprob_b: np.ndarray | None
    weights: np.ndarray
    method: str
    sample_names: list[str]
    feature_names: list[str]
    sv_names: list[str]
    num_sv_result: NumSVResult | None = None
    collinear_dropped: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return sample-by-surrogate-variable matrix as a `pandas.DataFrame`."""
        return pd.DataFrame(self.sv, index=self.sample_names, columns=self.sv_names)

    def feature_posteriors_dataframe(self) -> pd.DataFrame:
        """Return per-feature posterior probabilities (`pprob_gam`, `pprob_b`) and `weights`."""
        pprob_b_col = (
            self.pprob_b
            if self.pprob_b is not None
            else np.full(len(self.feature_names), np.nan, dtype=np.float64)
        )
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "pprob_gam": self.pprob_gam,
                "pprob_b": pprob_b_col,
                "weight": self.weights,
            }
        )

    def augment_design(self, design: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Horizontally append estimated surrogate variables onto `design`."""
        n_samples = len(self.sample_names)
        if isinstance(design, pd.DataFrame):
            if len(design) != n_samples:
                raise ValueError(
                    f"Design DataFrame has {len(design)} rows, expected {n_samples} samples."
                )
            if self.n_sv == 0:
                return design.copy()
            sv_df = pd.DataFrame(self.sv, index=design.index, columns=self.sv_names)
            return pd.concat([design, sv_df], axis=1)

        arr = np.asarray(design, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != n_samples:
            raise ValueError(
                f"Design array must have shape ({n_samples}, p), got {arr.shape}."
            )
        if self.n_sv == 0:
            return arr.copy()
        return np.hstack([arr, self.sv])

    def summary(self) -> str:
        """Return a human-readable summary of the fitted SVA model."""
        mean_gam = float(np.mean(self.pprob_gam)) if self.pprob_gam.size > 0 else float("nan")
        mean_b = (
            f"{float(np.mean(self.pprob_b)):.3f}"
            if self.pprob_b is not None and self.pprob_b.size > 0
            else "N/A"
        )
        mean_w = float(np.mean(self.weights)) if self.weights.size > 0 else float("nan")
        return (
            f"SVAResult(method={self.method!r}, n_sv={self.n_sv}, "
            f"n_samples={len(self.sample_names)}, n_features={len(self.feature_names)}, "
            f"mean_pprob_gam={mean_gam:.3f}, mean_pprob_b={mean_b}, "
            f"mean_weight={mean_w:.3f}, collinear_dropped={self.collinear_dropped})"
        )


def _select_vfilter(
    Y: np.ndarray,
    feature_names: list[str],
    vfilter: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Optionally subset `Y` and `feature_names` to the top `vfilter` most variable features."""
    if vfilter is None:
        return Y, feature_names
    if vfilter < 2:
        raise ValueError(f"vfilter must be >= 2, got {vfilter}.")
    m = Y.shape[0]
    k_keep = min(int(vfilter), m)
    row_var = np.var(Y, axis=1, ddof=1)
    top_idx = np.sort(np.argsort(-row_var, kind="mergesort")[:k_keep])
    return Y[top_idx], [feature_names[i] for i in top_idx]


def _canonicalize_direction(vec: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Normalize `vec` to unit norm with a deterministic sign convention."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.copy()
    out = vec / norm
    if ref is not None:
        dot = float(np.dot(out, ref))
        if abs(dot) > 1e-12:
            return out if dot >= 0.0 else -out
    max_idx = int(np.argmax(np.abs(out)))
    if out[max_idx] < 0.0:
        out = -out
    return out


def _match_and_guard_surrogates(
    V_w: np.ndarray,
    V0: np.ndarray,
    Q1: np.ndarray,
    K: int,
    *,
    orthogonalize: bool = False,
) -> tuple[np.ndarray, int]:
    """Match weighted right-singular vectors to `V0[:, :K]` and guard collinearity with `mod`."""
    n = Q1.shape[0]
    r1 = Q1.shape[1]
    if K == 0 or V0.shape[1] == 0:
        return np.empty((n, 0), dtype=np.float64), 0

    k_eff = min(K, V0.shape[1])
    n_cand = min(V_w.shape[1], max(k_eff + r1 + 3, 2 * k_eff + 2))
    used: set[int] = set()
    candidates: list[np.ndarray] = []

    for j in range(k_eff):
        ref = V0[:, j]
        best_ell = -1
        best_score = -1.0
        for ell in range(n_cand):
            if ell in used:
                continue
            score = abs(float(np.dot(V_w[:, ell], ref)))
            if score > best_score:
                best_score = score
                best_ell = ell
        if best_ell >= 0:
            used.add(best_ell)
            candidates.append(V_w[:, best_ell].copy())
        else:
            candidates.append(ref.copy())

    Q_aug = Q1.copy()
    accepted: list[np.ndarray] = []
    dropped = 0
    collinear_tol = 1e-4

    for j, s_j in enumerate(candidates):
        ref = V0[:, j]
        r_j = s_j - Q_aug @ (Q_aug.T @ s_j)
        if float(np.linalg.norm(r_j)) < collinear_tol:
            replaced = False
            for fb_idx in [j] + [idx for idx in range(V0.shape[1]) if idx != j]:
                v_fb = V0[:, fb_idx]
                r_fb = v_fb - Q_aug @ (Q_aug.T @ v_fb)
                if float(np.linalg.norm(r_fb)) >= collinear_tol:
                    s_j = v_fb.copy()
                    r_j = r_fb
                    replaced = True
                    break
            if not replaced:
                dropped += 1
                continue

        if orthogonalize:
            s_orth = s_j - Q1 @ (Q1.T @ s_j)
            if accepted:
                Q_acc, _ = orthonormal_basis(np.column_stack(accepted))
                s_orth = s_orth - Q_acc @ (Q_acc.T @ s_orth)
            s_out = _canonicalize_direction(s_orth, ref)
        else:
            s_out = _canonicalize_direction(s_j, ref)

        accepted.append(s_out)
        r_unit = r_j / float(np.linalg.norm(r_j))
        Q_aug = np.column_stack([Q_aug, r_unit])

    if not accepted:
        return np.empty((n, 0), dtype=np.float64), dropped
    return np.column_stack(accepted), dropped


def _empty_sva_result(
    n_samples: int,
    m_eval: int,
    method: str,
    sample_names: list[str],
    feature_names: list[str],
    num_sv_res: NumSVResult | None,
    *,
    has_pprob_b: bool,
    controls: np.ndarray | None = None,
) -> SVAResult:
    """Construct an empty-SV `SVAResult` when `n_sv == 0`."""
    pprob_gam = (
        controls.copy()
        if controls is not None
        else np.zeros(m_eval, dtype=np.float64)
    )
    weights = controls.copy() if controls is not None else np.zeros(m_eval, dtype=np.float64)
    pprob_b = np.zeros(m_eval, dtype=np.float64) if has_pprob_b else None
    return SVAResult(
        sv=np.empty((n_samples, 0), dtype=np.float64),
        n_sv=0,
        pprob_gam=pprob_gam,
        pprob_b=pprob_b,
        weights=weights,
        method=method,
        sample_names=list(sample_names),
        feature_names=list(feature_names),
        sv_names=[],
        num_sv_result=num_sv_res,
        collinear_dropped=0,
    )


def _irwsva_arrays(
    Y: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray | None,
    feature_names: list[str],
    sample_names: list[str],
    *,
    n_sv: int | None = None,
    B: int = 5,
    num_sv_method: str = "be",
    vfilter: int | None = None,
    seed: int | np.random.Generator | None = None,
    orthogonalize: bool = False,
) -> SVAResult:
    """Internal Iteratively Reweighted SVA implementation on validated arrays."""
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Input matrix contains NaN or Inf values; filter or impute incomplete features before SVA."
        )
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}.")

    n_samples = Y.shape[1]
    X1_v, X0_v, Q1, r1, Q0, r0 = validate_nested_models(
        X1, X0, n_samples, require_strict_nesting=True, min_resid_df=2
    )

    Y_eval, feat_eval = _select_vfilter(Y, feature_names, vfilter)
    m_eval = Y_eval.shape[0]
    if m_eval < 2:
        raise ValueError("At least 2 features are required for SVA.")

    num_sv_res: NumSVResult | None = None
    if n_sv is None:
        num_sv_res = num_sv_arrays(Y_eval, X1_v, method=num_sv_method, seed=seed)
        k_req = num_sv_res.n_sv
    else:
        if n_sv < 0:
            raise ValueError(f"n_sv must be >= 0, got {n_sv}.")
        k_req = int(n_sv)

    if k_req == 0:
        return _empty_sva_result(
            n_samples,
            m_eval,
            "irw",
            sample_names,
            feat_eval,
            num_sv_res,
            has_pprob_b=True,
        )

    if k_req >= (n_samples - r1):
        raise ValueError(
            f"Requested n_sv={k_req} must be strictly less than residual df n - rank(mod) = {n_samples - r1}."
        )

    K = min(k_req, m_eval, n_samples - r1 - 1)
    if K < 1:
        return _empty_sva_result(
            n_samples,
            m_eval,
            "irw",
            sample_names,
            feat_eval,
            num_sv_res,
            has_pprob_b=True,
        )

    R = Y_eval - (Y_eval @ Q1) @ Q1.T
    _, _, Vt0 = np.linalg.svd(R, full_matrices=False)
    V0 = Vt0.T
    for j in range(min(K, V0.shape[1])):
        V0[:, j] = _canonicalize_direction(V0[:, j])

    Y0 = Y_eval - (Y_eval @ Q0) @ Q0.T
    S_curr = V0[:, :K].copy()
    pprob_gam = np.zeros(m_eval, dtype=np.float64)
    pprob_b = np.zeros(m_eval, dtype=np.float64)
    w = np.ones(m_eval, dtype=np.float64)
    collinear_dropped = 0

    for _ in range(B):
        X1_aug = np.hstack([X1_v, S_curr])
        X0_aug = np.hstack([X0_v, S_curr])

        f_gam = f_pvalue_arrays(Y_eval, X1_aug, X1_v, feat_eval)
        lfdr_gam = edge_lfdr(f_gam.p_value)
        pprob_gam = np.where(np.isfinite(lfdr_gam), 1.0 - lfdr_gam, 0.0)

        f_b = f_pvalue_arrays(Y_eval, X1_aug, X0_aug, feat_eval)
        lfdr_b = edge_lfdr(f_b.p_value)
        pprob_b = np.where(np.isfinite(lfdr_b), 1.0 - lfdr_b, 0.0)

        w = pprob_gam * (1.0 - pprob_b)
        if float(np.sum(w ** 2)) <= 1e-12 or int(np.sum(w > 1e-6)) < max(2, K):
            w = pprob_gam.copy()
            if float(np.sum(w ** 2)) <= 1e-12 or int(np.sum(w > 1e-6)) < max(2, K):
                w = np.ones(m_eval, dtype=np.float64)
                S_next = V0[:, :K].copy()
                collinear_dropped = 0
                S_curr = S_next
                break

        Y_w = w[:, None] * Y0
        _, _, Vt_w = np.linalg.svd(Y_w, full_matrices=False)
        V_w = Vt_w.T

        S_next, dropped = _match_and_guard_surrogates(
            V_w, V0, Q1, K, orthogonalize=orthogonalize
        )
        collinear_dropped = dropped
        if S_next.shape[1] == 0:
            S_curr = S_next
            break
        S_curr = S_next

    n_out = S_curr.shape[1]
    sv_names = [f"SV{i + 1}" for i in range(n_out)]
    return SVAResult(
        sv=S_curr,
        n_sv=n_out,
        pprob_gam=pprob_gam,
        pprob_b=pprob_b,
        weights=w,
        method="irw",
        sample_names=list(sample_names),
        feature_names=list(feat_eval),
        sv_names=sv_names,
        num_sv_result=num_sv_res,
        collinear_dropped=collinear_dropped,
    )


def _twostepsva_arrays(
    Y: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray | None,
    feature_names: list[str],
    sample_names: list[str],
    *,
    n_sv: int | None = None,
    num_sv_method: str = "be",
    vfilter: int | None = None,
    seed: int | np.random.Generator | None = None,
    lfdr_threshold: float = 0.10,
) -> SVAResult:
    """Internal Two-Step SVA implementation on validated arrays."""
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Input matrix contains NaN or Inf values; filter or impute incomplete features before SVA."
        )
    if not (0.0 < lfdr_threshold < 1.0):
        raise ValueError(f"lfdr_threshold must be in (0, 1), got {lfdr_threshold}.")

    n_samples = Y.shape[1]
    X1_v, X0_v, Q1, r1, Q0, _ = validate_nested_models(
        X1, X0, n_samples, require_strict_nesting=False, min_resid_df=2
    )

    Y_eval, feat_eval = _select_vfilter(Y, feature_names, vfilter)
    m_eval = Y_eval.shape[0]
    if m_eval < 2:
        raise ValueError("At least 2 features are required for SVA.")

    num_sv_res: NumSVResult | None = None
    if n_sv is None:
        num_sv_res = num_sv_arrays(Y_eval, X1_v, method=num_sv_method, seed=seed)
        k_req = num_sv_res.n_sv
    else:
        if n_sv < 0:
            raise ValueError(f"n_sv must be >= 0, got {n_sv}.")
        k_req = int(n_sv)

    if k_req == 0:
        return _empty_sva_result(
            n_samples,
            m_eval,
            "two-step",
            sample_names,
            feat_eval,
            num_sv_res,
            has_pprob_b=False,
        )

    if k_req >= (n_samples - r1):
        raise ValueError(
            f"Requested n_sv={k_req} must be strictly less than residual df n - rank(mod) = {n_samples - r1}."
        )

    K = min(k_req, m_eval, n_samples - r1 - 1)
    if K < 1:
        return _empty_sva_result(
            n_samples,
            m_eval,
            "two-step",
            sample_names,
            feat_eval,
            num_sv_res,
            has_pprob_b=False,
        )

    R = Y_eval - (Y_eval @ Q1) @ Q1.T
    _, _, Vt0 = np.linalg.svd(R, full_matrices=False)
    V0 = Vt0.T
    for j in range(min(K, V0.shape[1])):
        V0[:, j] = _canonicalize_direction(V0[:, j])

    S0 = V0[:, :K]
    f_gam = f_pvalue_arrays(Y_eval, np.hstack([X1_v, S0]), X1_v, feat_eval)
    lfdr = edge_lfdr(f_gam.p_value)
    pprob_gam = np.where(np.isfinite(lfdr), 1.0 - lfdr, 0.0)

    mask = np.isfinite(lfdr) & (lfdr <= lfdr_threshold)
    if int(np.sum(mask)) < max(2, K):
        k_fallback = min(m_eval, max(10, 2 * K))
        p_safe = np.where(np.isfinite(f_gam.p_value), f_gam.p_value, np.inf)
        top_idx = np.argsort(p_safe, kind="mergesort")[:k_fallback]
        mask = np.zeros(m_eval, dtype=bool)
        mask[top_idx] = True

    weights = mask.astype(np.float64)
    Y0 = Y_eval - (Y_eval @ Q0) @ Q0.T
    Y_sub = Y0[mask, :]
    _, _, Vt_sub = np.linalg.svd(Y_sub, full_matrices=False)
    V_sub = Vt_sub.T

    S_out, dropped = _match_and_guard_surrogates(V_sub, V0, Q1, K, orthogonalize=False)
    n_out = S_out.shape[1]
    sv_names = [f"SV{i + 1}" for i in range(n_out)]

    return SVAResult(
        sv=S_out,
        n_sv=n_out,
        pprob_gam=pprob_gam,
        pprob_b=None,
        weights=weights,
        method="two-step",
        sample_names=list(sample_names),
        feature_names=list(feat_eval),
        sv_names=sv_names,
        num_sv_result=num_sv_res,
        collinear_dropped=dropped,
    )


def _supervised_sva_arrays(
    Y: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray | None,
    controls: np.ndarray,
    feature_names: list[str],
    sample_names: list[str],
    *,
    n_sv: int | None = None,
    num_sv_method: str = "be",
    seed: int | np.random.Generator | None = None,
) -> SVAResult:
    """Internal supervised SVA using known negative-control feature weights."""
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Input matrix contains NaN or Inf values; filter or impute incomplete features before SVA."
        )
    m, n_samples = Y.shape
    ctrl = np.asarray(controls, dtype=np.float64)
    if ctrl.ndim != 1 or ctrl.shape[0] != m:
        raise ValueError(f"controls must be 1D of length {m}, got shape {ctrl.shape}.")
    if not np.all(np.isfinite(ctrl)) or np.any(ctrl < 0.0) or np.any(ctrl > 1.0):
        raise ValueError("controls must contain finite values in [0, 1].")
    if float(np.sum(ctrl ** 2)) <= 1e-12:
        raise ValueError("controls weights are all near zero.")

    X1_v, _, Q1, r1, Q0, _ = validate_nested_models(
        X1, X0, n_samples, require_strict_nesting=False, min_resid_df=2
    )

    num_sv_res: NumSVResult | None = None
    if n_sv is None:
        num_sv_res = num_sv_arrays(Y, X1_v, method=num_sv_method, seed=seed)
        k_req = num_sv_res.n_sv
    else:
        if n_sv < 0:
            raise ValueError(f"n_sv must be >= 0, got {n_sv}.")
        k_req = int(n_sv)

    if k_req == 0:
        return _empty_sva_result(
            n_samples,
            m,
            "supervised",
            sample_names,
            feature_names,
            num_sv_res,
            has_pprob_b=False,
            controls=ctrl,
        )

    if k_req >= (n_samples - r1):
        raise ValueError(
            f"Requested n_sv={k_req} must be strictly less than residual df n - rank(mod) = {n_samples - r1}."
        )

    K = min(k_req, m, n_samples - r1 - 1)
    R = Y - (Y @ Q1) @ Q1.T
    _, _, Vt0 = np.linalg.svd(R, full_matrices=False)
    V0 = Vt0.T
    for j in range(min(K, V0.shape[1])):
        V0[:, j] = _canonicalize_direction(V0[:, j])

    Y0 = Y - (Y @ Q0) @ Q0.T
    Y_w = ctrl[:, None] * Y0
    _, _, Vt_w = np.linalg.svd(Y_w, full_matrices=False)
    V_w = Vt_w.T

    S_out, dropped = _match_and_guard_surrogates(V_w, V0, Q1, K, orthogonalize=False)
    n_out = S_out.shape[1]
    sv_names = [f"SV{i + 1}" for i in range(n_out)]

    return SVAResult(
        sv=S_out,
        n_sv=n_out,
        pprob_gam=ctrl.copy(),
        pprob_b=None,
        weights=ctrl.copy(),
        method="supervised",
        sample_names=list(sample_names),
        feature_names=list(feature_names),
        sv_names=sv_names,
        num_sv_result=num_sv_res,
        collinear_dropped=dropped,
    )


def irwsva(
    dat: Any,
    mod: Any,
    mod0: Any | None = None,
    n_sv: int | None = None,
    B: int = 5,
    num_sv_method: str = "be",
    vfilter: int | None = None,
    seed: int | np.random.Generator | None = None,
    *,
    orthogonalize: bool = False,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> SVAResult:
    """Estimate surrogate variables via Iteratively Reweighted SVA (Leek & Storey 2007)."""
    Y, feature_names, sample_names, adata_ref = coerce_expression_matrix(
        dat, features_as_rows=features_as_rows, layer=layer
    )
    n_samples = Y.shape[1]
    X1 = coerce_design_matrix(mod, n_samples, adata_ref=adata_ref, name="mod")
    X0 = (
        coerce_design_matrix(mod0, n_samples, adata_ref=adata_ref, name="mod0")
        if mod0 is not None
        else None
    )
    return _irwsva_arrays(
        Y,
        X1,
        X0,
        feature_names,
        sample_names,
        n_sv=n_sv,
        B=B,
        num_sv_method=num_sv_method,
        vfilter=vfilter,
        seed=seed,
        orthogonalize=orthogonalize,
    )


def twostepsva(
    dat: Any,
    mod: Any,
    mod0: Any | None = None,
    n_sv: int | None = None,
    num_sv_method: str = "be",
    vfilter: int | None = None,
    seed: int | np.random.Generator | None = None,
    *,
    lfdr_threshold: float = 0.10,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> SVAResult:
    """Estimate surrogate variables via Two-Step SVA (Leek & Storey 2007)."""
    Y, feature_names, sample_names, adata_ref = coerce_expression_matrix(
        dat, features_as_rows=features_as_rows, layer=layer
    )
    n_samples = Y.shape[1]
    X1 = coerce_design_matrix(mod, n_samples, adata_ref=adata_ref, name="mod")
    X0 = (
        coerce_design_matrix(mod0, n_samples, adata_ref=adata_ref, name="mod0")
        if mod0 is not None
        else None
    )
    return _twostepsva_arrays(
        Y,
        X1,
        X0,
        feature_names,
        sample_names,
        n_sv=n_sv,
        num_sv_method=num_sv_method,
        vfilter=vfilter,
        seed=seed,
        lfdr_threshold=lfdr_threshold,
    )


def sva(
    dat: Any,
    mod: Any,
    mod0: Any | None = None,
    n_sv: int | None = None,
    controls: np.ndarray | None = None,
    method: str = "irw",
    vfilter: int | None = None,
    B: int = 5,
    num_sv_method: str = "be",
    seed: int | np.random.Generator | None = None,
    *,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> SVAResult:
    """Unified entry point for Surrogate Variable Analysis (`"irw"`, `"two-step"`, `"supervised"`)."""
    if controls is not None or method == "supervised":
        if controls is None:
            raise ValueError("controls must be provided when method='supervised'.")
        if vfilter is not None:
            raise ValueError("vfilter cannot be used with supervised controls.")
        Y, feature_names, sample_names, adata_ref = coerce_expression_matrix(
            dat, features_as_rows=features_as_rows, layer=layer
        )
        n_samples = Y.shape[1]
        X1 = coerce_design_matrix(mod, n_samples, adata_ref=adata_ref, name="mod")
        X0 = (
            coerce_design_matrix(mod0, n_samples, adata_ref=adata_ref, name="mod0")
            if mod0 is not None
            else None
        )
        return _supervised_sva_arrays(
            Y,
            X1,
            X0,
            controls,
            feature_names,
            sample_names,
            n_sv=n_sv,
            num_sv_method=num_sv_method,
            seed=seed,
        )

    if method == "irw":
        return irwsva(
            dat,
            mod,
            mod0=mod0,
            n_sv=n_sv,
            B=B,
            num_sv_method=num_sv_method,
            vfilter=vfilter,
            seed=seed,
            features_as_rows=features_as_rows,
            layer=layer,
        )
    if method == "two-step":
        return twostepsva(
            dat,
            mod,
            mod0=mod0,
            n_sv=n_sv,
            num_sv_method=num_sv_method,
            vfilter=vfilter,
            seed=seed,
            features_as_rows=features_as_rows,
            layer=layer,
        )
    raise ValueError(
        f"Unsupported sva method {method!r}; expected 'irw', 'two-step', or 'supervised'."
    )


def svaseq(
    dat: Any,
    mod: Any,
    mod0: Any | None = None,
    n_sv: int | None = None,
    controls: np.ndarray | None = None,
    method: str = "irw",
    vfilter: int | None = None,
    B: int = 5,
    num_sv_method: str = "be",
    constant: float = 1.0,
    library_size_norm: bool = False,
    seed: int | np.random.Generator | None = None,
    *,
    features_as_rows: bool = True,
    layer: str | None = None,
) -> SVAResult:
    """Moderated log-count Surrogate Variable Analysis for RNA-seq counts (Leek 2014).

    Applies `log(counts + constant)` (optionally after median library-size scaling),
    removes strictly zero-variance rows across samples, and runs `sva`.
    """
    C, feature_names, sample_names, adata_ref = coerce_expression_matrix(
        dat, features_as_rows=features_as_rows, layer=layer
    )
    if not np.all(np.isfinite(C)):
        raise ValueError("svaseq count matrix contains NaN or Inf values.")
    if np.any(C < 0.0):
        raise ValueError("svaseq requires a non-negative count matrix.")
    if constant <= 0.0 and np.any(C <= 0.0):
        raise ValueError("constant must be > 0 when counts contain zeros.")
    if np.any(C + constant <= 0.0):
        raise ValueError("counts + constant must be strictly positive for log transform.")

    n_samples = C.shape[1]
    X1 = coerce_design_matrix(mod, n_samples, adata_ref=adata_ref, name="mod")
    X0 = (
        coerce_design_matrix(mod0, n_samples, adata_ref=adata_ref, name="mod0")
        if mod0 is not None
        else None
    )

    if library_size_norm:
        lib_sizes = np.sum(C, axis=0)
        if np.any(lib_sizes <= 0.0):
            raise ValueError("Cannot library-size normalize when a sample has zero total counts.")
        med_lib = float(np.median(lib_sizes))
        C_norm = C * (med_lib / lib_sizes[None, :])
    else:
        C_norm = C

    Y_mod = np.log(C_norm + float(constant))
    row_var = np.var(Y_mod, axis=1, ddof=1)
    keep = row_var > 1e-24
    if int(np.sum(keep)) < 2:
        raise ValueError(
            "Fewer than 2 variable features remain after filtering zero-variance count rows."
        )

    Y_kept = Y_mod[keep]
    kept_features = [feature_names[i] for i in np.flatnonzero(keep)]

    if controls is not None or method == "supervised":
        if controls is None:
            raise ValueError("controls must be provided when method='supervised'.")
        if vfilter is not None:
            raise ValueError("vfilter cannot be used with supervised controls.")
        ctrl = np.asarray(controls, dtype=np.float64)
        if ctrl.ndim != 1 or ctrl.shape[0] != C.shape[0]:
            raise ValueError(
                f"controls must be 1D of length {C.shape[0]}, got shape {ctrl.shape}."
            )
        return _supervised_sva_arrays(
            Y_kept,
            X1,
            X0,
            ctrl[keep],
            kept_features,
            sample_names,
            n_sv=n_sv,
            num_sv_method=num_sv_method,
            seed=seed,
        )

    if method == "irw":
        return _irwsva_arrays(
            Y_kept,
            X1,
            X0,
            kept_features,
            sample_names,
            n_sv=n_sv,
            B=B,
            num_sv_method=num_sv_method,
            vfilter=vfilter,
            seed=seed,
        )
    if method == "two-step":
        return _twostepsva_arrays(
            Y_kept,
            X1,
            X0,
            kept_features,
            sample_names,
            n_sv=n_sv,
            num_sv_method=num_sv_method,
            vfilter=vfilter,
            seed=seed,
        )
    raise ValueError(
        f"Unsupported sva method {method!r}; expected 'irw', 'two-step', or 'supervised'."
    )
