"""LD reference eigendecomposition panels for High-Definition Likelihood (HDL).

Reference
---------
Ning, Z., Pawitan, Y., & Shen, X. (2020). High-definition likelihood inference of
genetic correlation across human complex traits. Nature Genetics, 52(8), 859-864.
https://doi.org/10.1038/s41588-020-0653-y
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import linalg

SettingsType = Literal["default", "R_default"]
VALID_SETTINGS: tuple[str, ...] = ("default", "R_default")


def validate_settings(settings: str) -> SettingsType:
    """Validate the ``settings`` mode parameter."""
    if not isinstance(settings, str) or settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


@dataclass
class LDEigenBlock:
    """Truncated eigendecomposition of a single LD reference block."""

    block_id: int
    n_snps: int
    eigvals: np.ndarray
    eigvecs: np.ndarray
    ld_scores: np.ndarray
    var_explained: float

    @property
    def n_modes(self) -> int:
        return int(self.eigvals.shape[0])


@dataclass
class LDEigenPanel:
    """Collection of regularized LD block eigendecompositions across the genome."""

    blocks: list[LDEigenBlock]
    total_snps: int
    total_eigenmodes: int
    eigen_cut: float
    lim: float
    settings: SettingsType

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)


def _normalize_r_blocks(
    R_blocks: np.ndarray | Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Convert a single 2D LD matrix or sequence of 2D LD matrices into a list."""
    if isinstance(R_blocks, np.ndarray):
        if R_blocks.ndim == 2:
            return [R_blocks]
        if R_blocks.ndim == 3:
            return [R_blocks[b] for b in range(R_blocks.shape[0])]
        raise ValueError(f"R_blocks array must be 2D or 3D, got shape {R_blocks.shape}.")
    blocks = list(R_blocks)
    if len(blocks) == 0:
        raise ValueError("R_blocks must contain at least one LD block.")
    return [np.asarray(B, dtype=np.float64) for B in blocks]


def build_ld_eigen_panel(
    R_blocks: np.ndarray | Sequence[np.ndarray],
    eigen_cut: float = 0.99,
    lim: float = float(np.exp(-18.0)),
    settings: SettingsType = "default",
) -> LDEigenPanel:
    """Compute regularized eigendecompositions ``R_b = V_b Lambda_b V_b^T`` per block.

    Retains top ``k_b`` eigenmodes explaining ``eigen_cut`` cumulative variance
    above eigenvalue floor ``lim`` (default ``exp(-18)``, matching ``zhenin/HDL``).
    """
    validate_settings(settings)
    if not (0.0 < eigen_cut <= 1.0):
        raise ValueError(f"eigen_cut must be in (0, 1], got {eigen_cut}.")
    if lim < 0.0 or not np.isfinite(lim):
        raise ValueError(f"lim must be finite and >= 0, got {lim}.")

    raw_blocks = _normalize_r_blocks(R_blocks)
    eigen_blocks: list[LDEigenBlock] = []
    total_snps = 0
    total_modes = 0

    for b_idx, R_raw in enumerate(raw_blocks):
        if R_raw.ndim != 2 or R_raw.shape[0] != R_raw.shape[1] or R_raw.shape[0] < 1:
            raise ValueError(
                f"Block {b_idx} must be a non-empty square 2D matrix, got {R_raw.shape}."
            )
        if not np.all(np.isfinite(R_raw)):
            raise ValueError(f"Block {b_idx} contains non-finite entries.")

        m_b = R_raw.shape[0]
        R_sym = 0.5 * (R_raw + R_raw.T)
        if settings == "default":
            diag = np.diag(R_sym)
            if np.all(diag > 0.0) and not np.allclose(diag, 1.0, atol=1e-6):
                inv_sd = 1.0 / np.sqrt(diag)
                R_sym = (R_sym * inv_sd[:, None]) * inv_sd[None, :]
                R_sym = 0.5 * (R_sym + R_sym.T)
            np.fill_diagonal(R_sym, 1.0)

        ld_scores = np.sum(R_sym * R_sym, axis=1)
        eigvals, eigvecs = linalg.eigh(R_sym)
        desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[desc]
        eigvecs = eigvecs[:, desc]

        pos_vals = np.maximum(eigvals, 0.0)
        sum_pos = float(np.sum(pos_vals))
        if sum_pos <= 0.0 or eigvals[0] <= max(lim, 1e-15):
            raise ValueError(f"Block {b_idx} has no positive eigenvalues above lim={lim}.")

        cum_var = np.cumsum(pos_vals) / sum_pos
        if settings == "R_default":
            above = np.where(cum_var > eigen_cut)[0]
            k_cut = int(above[0] + 1) if above.size > 0 else int(np.sum(eigvals > lim))
        else:
            above = np.where(cum_var >= eigen_cut - 1e-15)[0]
            k_cut = int(above[0] + 1) if above.size > 0 else int(np.sum(eigvals > lim))

        n_above_lim = int(np.sum(eigvals > max(lim, 1e-15)))
        k_b = max(1, min(k_cut, n_above_lim))

        sel_vals = eigvals[:k_b].copy()
        sel_vecs = eigvecs[:, :k_b].copy()
        # Deterministic eigenvector sign orientation
        for col in range(k_b):
            pivot = int(np.argmax(np.abs(sel_vecs[:, col])))
            if sel_vecs[pivot, col] < 0.0:
                sel_vecs[:, col] = -sel_vecs[:, col]

        var_exp = float(np.sum(sel_vals) / sum_pos)
        eigen_blocks.append(
            LDEigenBlock(
                block_id=b_idx,
                n_snps=m_b,
                eigvals=sel_vals,
                eigvecs=sel_vecs,
                ld_scores=ld_scores,
                var_explained=var_exp,
            )
        )
        total_snps += m_b
        total_modes += k_b

    return LDEigenPanel(
        blocks=eigen_blocks,
        total_snps=total_snps,
        total_eigenmodes=total_modes,
        eigen_cut=eigen_cut,
        lim=lim,
        settings=settings,
    )


def ensure_panel(
    ld_panels: LDEigenPanel | Sequence[LDEigenBlock] | Sequence[np.ndarray] | np.ndarray,
    eigen_cut: float = 0.99,
    lim: float = float(np.exp(-18.0)),
    settings: SettingsType = "default",
) -> LDEigenPanel:
    """Coerce an ``LDEigenPanel``, block list, or raw LD matrices into an ``LDEigenPanel``."""
    if isinstance(ld_panels, LDEigenPanel):
        return ld_panels
    if isinstance(ld_panels, Sequence) and len(ld_panels) > 0 and isinstance(
        ld_panels[0], LDEigenBlock
    ):
        blocks = list(ld_panels)  # type: ignore[arg-type]
        return LDEigenPanel(
            blocks=blocks,
            total_snps=sum(b.n_snps for b in blocks),
            total_eigenmodes=sum(b.n_modes for b in blocks),
            eigen_cut=eigen_cut,
            lim=lim,
            settings=settings,
        )
    return build_ld_eigen_panel(
        ld_panels,  # type: ignore[arg-type]
        eigen_cut=eigen_cut,
        lim=lim,
        settings=settings,
    )


def split_block_vectors(
    z: np.ndarray | Sequence[np.ndarray] | Sequence[float],
    panel: LDEigenPanel,
    name: str = "z",
) -> list[np.ndarray]:
    """Validate and split a flat 1D Z-score vector or block list to match ``panel``."""
    if isinstance(z, np.ndarray) and z.ndim == 1:
        arr = np.asarray(z, dtype=np.float64)
        if arr.shape[0] != panel.total_snps:
            raise ValueError(
                f"{name} length {arr.shape[0]} does not match panel total_snps={panel.total_snps}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        out: list[np.ndarray] = []
        offset = 0
        for blk in panel.blocks:
            out.append(arr[offset : offset + blk.n_snps])
            offset += blk.n_snps
        return out

    if isinstance(z, Sequence) and len(z) > 0 and isinstance(z[0], (np.ndarray, list, tuple)):
        if len(z) != panel.n_blocks:
            raise ValueError(
                f"{name} has {len(z)} blocks, expected {panel.n_blocks}."
            )
        out = []
        for b_idx, (zb, blk) in enumerate(zip(z, panel.blocks)):
            zb_arr = np.asarray(zb, dtype=np.float64)
            if zb_arr.ndim != 1 or zb_arr.shape[0] != blk.n_snps:
                raise ValueError(
                    f"{name}[{b_idx}] has shape {zb_arr.shape}, expected ({blk.n_snps},)."
                )
            if not np.all(np.isfinite(zb_arr)):
                raise ValueError(f"{name}[{b_idx}] contains non-finite values.")
            out.append(zb_arr)
        return out

    arr_flat = np.asarray(z, dtype=np.float64)
    if arr_flat.ndim != 1 or arr_flat.shape[0] != panel.total_snps:
        raise ValueError(
            f"{name} must be a 1D array of length {panel.total_snps} or a list of "
            f"{panel.n_blocks} block arrays."
        )
    if not np.all(np.isfinite(arr_flat)):
        raise ValueError(f"{name} contains non-finite values.")
    out = []
    offset = 0
    for blk in panel.blocks:
        out.append(arr_flat[offset : offset + blk.n_snps])
        offset += blk.n_snps
    return out
