"""Shared modal normalization, orientation, and MAC utilities."""

from __future__ import annotations

import numpy as np


def _as_mode_matrix(modes) -> tuple[np.ndarray, bool]:
    values = np.asarray(modes, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("modes must be a non-empty vector or matrix")
    if np.any(~np.isfinite(values)):
        raise ValueError("modes contain non-finite values")
    return values.copy(), was_vector


def _validate_mass(mass, ndof: int):
    if mass.shape != (ndof, ndof):
        raise ValueError(f"mass matrix shape {mass.shape} does not match {ndof} DOFs")
    return mass


def mass_normalize_modes(modes, mass):
    """Return modes scaled so each mode satisfies phi.T @ M @ phi == 1."""
    normalized, was_vector = _as_mode_matrix(modes)
    mass = _validate_mass(mass, normalized.shape[0])
    for index in range(normalized.shape[1]):
        phi = normalized[:, index]
        modal_mass = float(phi @ (mass @ phi))
        if not np.isfinite(modal_mass) or modal_mass <= 0.0:
            raise ValueError(f"mode {index + 1} has invalid modal mass {modal_mass}")
        normalized[:, index] = phi / np.sqrt(modal_mass)
    return normalized[:, 0] if was_vector else normalized


def orient_modes_deterministic(modes):
    """Orient each real mode so its largest-magnitude DOF is nonnegative."""
    oriented, was_vector = _as_mode_matrix(modes)
    for index in range(oriented.shape[1]):
        phase_index = int(np.argmax(np.abs(oriented[:, index])))
        if oriented[phase_index, index] < 0.0:
            oriented[:, index] *= -1.0
    return oriented[:, 0] if was_vector else oriented


def normalize_and_orient_modes(modes, mass):
    """Mass-normalize modes, then apply the deterministic sign convention."""
    return orient_modes_deterministic(mass_normalize_modes(modes, mass))


def squared_mass_weighted_mac(phi, psi, mass) -> np.ndarray | float:
    """Compute squared mass-weighted MAC for every column pair."""
    left, left_vector = _as_mode_matrix(phi)
    right, right_vector = _as_mode_matrix(psi)
    if left.shape[0] != right.shape[0]:
        raise ValueError("mode sets have different numbers of DOFs")
    mass = _validate_mass(mass, left.shape[0])
    left_mass = np.sum(left * (mass @ left), axis=0)
    right_mass = np.sum(right * (mass @ right), axis=0)
    if np.any(~np.isfinite(left_mass)) or np.any(left_mass <= 0.0):
        raise ValueError("left mode set contains an invalid modal mass")
    if np.any(~np.isfinite(right_mass)) or np.any(right_mass <= 0.0):
        raise ValueError("right mode set contains an invalid modal mass")
    cross = left.T @ (mass @ right)
    mac = cross**2 / (left_mass[:, None] * right_mass[None, :])
    if np.any(~np.isfinite(mac)):
        raise ValueError("MAC calculation produced non-finite values")
    mac = np.clip(mac, 0.0, 1.0)
    if left_vector and right_vector:
        return float(mac[0, 0])
    return mac
