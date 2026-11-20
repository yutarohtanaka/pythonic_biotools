"""`grapple`: Genome-wide weak-IV robust profile-likelihood MR and multimodal pathway discovery (Wang et al., PNAS 2021)."""

from .core import (
    GRAPPLE,
    GrappleMVMRResult,
    GrappleMode,
    GrappleResult,
    grapple_mvmr,
    grapple_robust_est,
)
from .loss import psi_deriv, psi_loss, resolve_loss_params, rho_loss, robust_inlier_weight
from .plotting import plot_grapple_profile
from .preprocessing import estimate_cor_mat, select_grapple_instruments

__all__ = [
    "GRAPPLE",
    "GrappleMode",
    "GrappleResult",
    "GrappleMVMRResult",
    "estimate_cor_mat",
    "select_grapple_instruments",
    "grapple_robust_est",
    "grapple_mvmr",
    "plot_grapple_profile",
    "rho_loss",
    "psi_loss",
    "psi_deriv",
    "robust_inlier_weight",
    "resolve_loss_params",
]

__version__ = "0.1.0"
