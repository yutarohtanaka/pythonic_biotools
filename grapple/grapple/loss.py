"""Robust loss functions (Tukey biweight, Huber, L2) and derivatives for GRAPPLE."""

from __future__ import annotations

from typing import Literal
import numpy as np

SettingsMode = Literal["default", "R_default"]
LossName = Literal["tukey", "huber", "l2"]

VALID_SETTINGS = ("default", "R_default")
VALID_LOSSES = ("tukey", "huber", "l2")


def validate_settings(settings: str) -> SettingsMode:
    """Validate mandatory ``settings`` mode."""
    if settings not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown settings={settings!r}. Expected one of {VALID_SETTINGS}."
        )
    return settings  # type: ignore[return-value]


def resolve_loss_params(
    loss_function: str,
    k: float | None,
    settings: str = "default",
) -> tuple[LossName, float, SettingsMode]:
    """Validate loss function and resolve tuning constant ``k``."""
    mode = validate_settings(settings)
    loss = loss_function.lower()
    if loss not in VALID_LOSSES:
        raise ValueError(
            f"Unknown loss_function={loss_function!r}. Expected one of {VALID_LOSSES}."
        )
    if k is None:
        if loss == "tukey":
            k_val = 4.685 if mode == "default" else 3.0
        elif loss == "huber":
            k_val = 1.345
        else:
            k_val = 0.0
    else:
        k_val = float(k)
        if loss in ("tukey", "huber") and k_val <= 0.0:
            raise ValueError(f"Tuning parameter k must be > 0 for loss={loss!r}.")
    return loss, k_val, mode  # type: ignore[return-value]


def rho_loss(t: np.ndarray, loss_function: LossName, k: float) -> np.ndarray:
    """Evaluate robust loss ``rho_k(t)`` elementwise."""
    if loss_function == "l2":
        return 0.5 * t * t
    if loss_function == "huber":
        abs_t = np.abs(t)
        return np.where(abs_t <= k, 0.5 * t * t, k * abs_t - 0.5 * k * k)
    # Tukey's biweight
    u2 = np.square(t / k)
    inside = 1.0 - np.power(np.maximum(1.0 - u2, 0.0), 3)
    return (k * k / 6.0) * np.where(u2 <= 1.0, inside, 1.0)


def psi_loss(t: np.ndarray, loss_function: LossName, k: float) -> np.ndarray:
    """Evaluate first derivative ``psi_k(t) = rho_k'(t)`` elementwise."""
    if loss_function == "l2":
        return t
    if loss_function == "huber":
        return np.clip(t, -k, k)
    u2 = np.square(t / k)
    w = np.square(np.maximum(1.0 - u2, 0.0))
    return np.where(u2 <= 1.0, t * w, 0.0)


def psi_deriv(t: np.ndarray, loss_function: LossName, k: float) -> np.ndarray:
    """Evaluate second derivative ``psi_k'(t) = rho_k''(t)`` elementwise."""
    if loss_function == "l2":
        return np.ones_like(t, dtype=float)
    if loss_function == "huber":
        return np.where(np.abs(t) <= k, 1.0, 0.0)
    u2 = np.square(t / k)
    return np.where(u2 <= 1.0, (1.0 - u2) * (1.0 - 5.0 * u2), 0.0)


def robust_inlier_weight(t: np.ndarray, loss_function: LossName, k: float) -> np.ndarray:
    """Return M-estimation weight ``psi_k(t) / t`` in [0, 1]."""
    if loss_function == "l2":
        return np.ones_like(t, dtype=float)
    if loss_function == "huber":
        abs_t = np.abs(t)
        return np.where(abs_t <= k, 1.0, k / np.maximum(abs_t, 1e-12))
    u2 = np.square(t / k)
    return np.where(u2 <= 1.0, np.square(1.0 - u2), 0.0)
