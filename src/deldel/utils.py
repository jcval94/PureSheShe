"""Utility helpers shared across deldel modules."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


def _verbosity_to_level(verbosity: int) -> int:
    """Map a verbosity integer to a logging level constant."""
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    return logging.WARNING


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features and return normalized data with mean and std."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd, mu, sd


def _destandardize(
    Qz: np.ndarray, rz: np.ndarray, cz: float, mu: np.ndarray, sd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Transform quadratic form parameters back to original scale."""
    D = np.diag(1.0 / sd)
    Qx = D @ Qz @ D
    r0 = D @ rz
    rx = r0 - 2.0 * (Qx @ mu)
    cx = float(mu.T @ Qx @ mu - r0.T @ mu + cz)
    return Qx, rx, cx


def _unpack(theta: np.ndarray, idx: dict, d: int):
    """Unpack quadratic parameters from flattened vector using index mapping."""
    Qz = np.zeros((d, d))
    for i in range(d):
        Qz[i, i] = theta[idx["diag"][i]]
    for k, (i, j) in enumerate(idx["pairs"]):
        val = 0.5 * theta[idx["off"][k]]
        Qz[i, j] = val
        Qz[j, i] = val
    rz = theta[idx["lin"][0]: idx["lin"][0] + d]
    cz = theta[idx["c"]]
    return Qz, rz, float(cz)
