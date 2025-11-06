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
    """Generate the canonical 4D DelDel dataset with three labelled regions.

    The layout matches the reference implementation used across tests,
    experiments and documentation examples:

    * Class ``1`` consists of several Gaussian blobs located at selected
      corners of a 4D hypercube of edge length ``2a``.
    * Classes ``0`` and ``2`` form compact clusters positioned away from the
      corners to provide contrasting regions.

    Parameters
    ----------
    n_per_cluster:
        Number of samples per Gaussian blob.
    std_class1:
        Dispersion for the class ``1`` corner blobs.
    std_other:
        Dispersion for the compact class ``0`` and ``2`` clusters.
    a:
        Hypercube scale parameter controlling the corner distance to the
        origin.
    random_state:
        Seed propagated to :func:`sklearn.datasets.make_blobs` for
        reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Feature matrix ``X`` with shape ``(n_clusters * n_per_cluster, 4)``,
        remapped labels ``y`` in ``{0, 1, 2}``, and the canonical feature names.
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

    centres_class0 = np.array([[0.0, 2.5, 0.5, 0.0]], dtype=float)
    centres_class2 = np.array([[-2.5, -1.5, -0.5, 1.0]], dtype=float)

    centres = np.vstack([centres_class0, corners, centres_class2])
    stds = [std_other] + [std_class1] * len(corners) + [std_other]

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
    y[np.isin(y_raw, range(1, len(corners) + 1))] = 1
    y[y_raw == len(centres) - 1] = 2

    feature_names = ["x1", "x2", "x3", "x4"]
    return X, y, feature_names


__all__ = ["make_corner_class_dataset"]
