"""Synthetic dataset utilities for DelDel examples and tests."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs


def make_corner_class_dataset(
    *,
    n_per_cluster: int = 150,
    std_class1: float = 0.4,
    std_other: float = 0.7,
    a: float = 3.0,
    random_state: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a 4D dataset with class 1 occupying hypercube corners.

    The dataset features three classes:

    * Class ``1`` is formed by several Gaussian blobs centred on corners of a
      4D hypercube with edge length ``2a``.
    * Classes ``0`` and ``2`` are compact clusters located away from the
      hypercube corners to provide contrasting decision boundaries.

    Parameters
    ----------
    n_per_cluster:
        Number of samples to draw for each Gaussian blob.
    std_class1:
        Standard deviation for the class ``1`` blobs.
    std_other:
        Standard deviation for the class ``0`` and ``2`` blobs.
    a:
        Controls the distance of the hypercube corners from the origin.
    random_state:
        Seed for the underlying random number generator.

    Returns
    -------
    X, y, feature_names
        ``X`` is the feature matrix with shape ``(n_clusters * n_per_cluster,
        4)``.  ``y`` stores the remapped class labels as integers ``{0, 1, 2}``.
        ``feature_names`` provides canonical feature names used across the
        DelDel codebase.
    """

    corners = np.array(
        [
            [a, a, a, a],
            [a, -a, a, -a],
            [-a, a, -a, a],
            [-a, -a, -a, a],
            [a, a, -a, -a],
            [-a, -a, a, -a],
        ],
        dtype=float,
    )

    centres_class0 = np.array([[0.0, 2.5, 0.5, 0.0]])
    centres_class2 = np.array([[-2.5, -1.5, -0.5, 1.0]])

    centres = np.vstack([centres_class0, corners, centres_class2])
    stds = np.concatenate(
        (
            np.full(len(centres_class0), std_other, dtype=float),
            np.full(len(corners), std_class1, dtype=float),
            np.full(len(centres_class2), std_other, dtype=float),
        )
    )

    X, y_raw = make_blobs(
        n_samples=[n_per_cluster] * len(centres),
        centers=centres,
        cluster_std=stds,
        n_features=4,
        random_state=random_state,
        shuffle=True,
    )

    y = np.zeros_like(y_raw)
    y[y_raw == 0] = 0
    y[np.isin(y_raw, np.arange(1, 1 + len(corners)))] = 1
    y[y_raw == len(centres) - 1] = 2

    feature_names = ["x1", "x2", "x3", "x4"]
    return X, y, feature_names


__all__ = ["make_corner_class_dataset"]
