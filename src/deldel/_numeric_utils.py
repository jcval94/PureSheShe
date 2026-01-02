"""Shared numeric helpers used across DelDel modules."""

from __future__ import annotations

import numpy as np


def standardize_matrix(X: np.ndarray):
    """Standardize matrix columns returning Z, mean and std."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    return Z, mu, sd


def destandardize_quadratic(Qz, rz, cz, mu, sd):
    """Undo standardization for quadratic form parameters."""
    D = np.diag(1.0 / sd)
    Qx = D @ Qz @ D
    r0 = D @ rz
    rx = r0 - 2.0 * (Qx @ mu)
    cx = float(mu.T @ Qx @ mu - r0.T @ mu + cz)
    return Qx, rx, cx


def unpack_quadratic_parameters(theta, idx, d):
    """Rebuild symmetric matrix, linear and constant terms from flat params."""
    theta = np.asarray(theta)
    Qz = np.zeros((d, d))
    diag_idx = np.asarray(idx["diag"], dtype=int)
    Qz[np.arange(d), np.arange(d)] = theta[diag_idx]
    pairs = np.asarray(idx["pairs"], dtype=int)
    if pairs.size:
        off_idx = np.asarray(idx["off"], dtype=int)
        vals = 0.5 * theta[off_idx]
        Qz[pairs[:, 0], pairs[:, 1]] = vals
        Qz[pairs[:, 1], pairs[:, 0]] = vals
    lin_start = int(idx["lin"][0])
    rz = theta[lin_start : lin_start + d]
    cz = theta[int(idx["c"])]
    return Qz, rz, float(cz)
