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
    Qz = np.zeros((d, d))
    for i in range(d):
        Qz[i, i] = theta[idx["diag"][i]]
    for k, (i, j) in enumerate(idx["pairs"]):
        Qz[i, j] = Qz[j, i] = 0.5 * theta[idx["off"][k]]
    rz = theta[idx["lin"][0] : idx["lin"][0] + d]
    cz = theta[idx["c"]]
    return Qz, rz, float(cz)
