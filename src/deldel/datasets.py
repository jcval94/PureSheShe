"""Synthetic dataset utilities for DelDel examples and tests."""

from __future__ import annotations

from typing import List, Tuple

import logging
from time import perf_counter

import numpy as np
from sklearn.datasets import make_blobs, make_classification


def _verbosity_to_level(verbosity: int) -> int:
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    return logging.WARNING


def make_corner_class_dataset(
    *,
    n_per_cluster: int = 150,
    std_class1: float = 0.4,
    std_other: float = 0.7,
    a: float = 3.0,
    random_state: int | None = 42,
    verbosity: int = 0,
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

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.setLevel(level)
    start = perf_counter()
    logger.log(
        level,
        "Generando corner dataset | n_per_cluster=%d std_class1=%.3f std_other=%.3f a=%.3f rs=%s",
        n_per_cluster,
        std_class1,
        std_other,
        a,
        random_state,
    )

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

    try:
        X, y_raw = make_blobs(
            n_samples=[n_per_cluster] * len(centres),
            centers=centres,
            cluster_std=stds,
            n_features=4,
            random_state=random_state,
            shuffle=True,
        )
    except Exception:
        logger.exception("Fallo make_blobs para corner dataset")
        raise

    y = np.zeros_like(y_raw)
    y[y_raw == 0] = 0
    y[np.isin(y_raw, range(1, len(corners) + 1))] = 1
    y[y_raw == len(centres) - 1] = 2

    feature_names = ["x1", "x2", "x3", "x4"]
    end = perf_counter()
    class_counts = {int(cls): int((y == cls).sum()) for cls in np.unique(y)}
    logger.log(
        level,
        "Corner dataset generado en %.4f s | muestras=%d dims=%d clases=%s",
        end - start,
        X.shape[0],
        X.shape[1],
        class_counts,
    )
    return X, y, feature_names


def make_high_dim_classification_dataset(
    *,
    n_samples: int = 30_000,
    n_features: int = 25,
    n_informative: int = 18,
    n_redundant: int = 2,
    n_repeated: int = 0,
    n_classes: int = 3,
    class_sep: float = 1.8,
    weights: Tuple[float, float, float] | None = None,
    random_state: int | None = 0,
    verbosity: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a wide synthetic dataset suitable for DelDel stress tests.

    The generator wraps :func:`sklearn.datasets.make_classification` with
    defaults aligned to the requirements of the high-cardinality experiments in
    the repository.  The default configuration produces 30,000 observations with
    25 numerical features and three balanced classes.

    Parameters
    ----------
    n_samples:
        Total number of rows to generate. Must be at least ``30_000`` for the
        requested stress tests.
    n_features:
        Number of features (columns) to synthesize.
    n_informative:
        Number of informative features driving class separation.
    n_redundant:
        Number of redundant features generated as linear combinations of the
        informative ones.
    n_repeated:
        Number of duplicated features.
    n_classes:
        Number of target classes to simulate.
    class_sep:
        Multiplicative factor controlling class separability.
    weights:
        Optional tuple of class weights. If omitted the classes are balanced.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Feature matrix ``X``, labels ``y`` in ``range(n_classes)``, and
        generated feature names ``["f1", "f2", ...]``.
    """

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.setLevel(level)
    logger.log(
        level,
        "make_high_dim_classification_dataset | n=%d features=%d inf=%d red=%d rep=%d classes=%d sep=%.2f weights=%s rs=%s",
        n_samples,
        n_features,
        n_informative,
        n_redundant,
        n_repeated,
        n_classes,
        class_sep,
        weights,
        random_state,
    )

    start = perf_counter()
    try:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=2,
            class_sep=class_sep,
            weights=weights,
            shuffle=True,
            random_state=random_state,
        )
    except Exception:
        logger.exception("Error en make_classification de dataset alto dimensional")
        raise
    end = perf_counter()
    class_counts = {int(cls): int((y == cls).sum()) for cls in np.unique(y)}
    logger.log(
        level,
        "Dataset alto dimensional generado en %.4f s | muestras=%d dims=%d clases=%s",
        end - start,
        X.shape[0],
        X.shape[1],
        class_counts,
    )

    feature_names = [f"f{i+1}" for i in range(n_features)]
    return X.astype(float), y.astype(int), feature_names


def plot_corner_class_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    show: bool = True,
):
    """Visualize the 4D corner dataset using PCA (3D) and a scatter matrix.

    Imports for plotting are intentionally delayed to avoid making ``matplotlib``
    and ``pandas`` hard dependencies of the module. Callers can disable
    ``show`` to retrieve the figures without triggering ``plt.show`` during
    automated runs.

    Parameters
    ----------
    X:
        Feature matrix returned by :func:`make_corner_class_dataset`.
    y:
        Labels in ``{0, 1, 2}``.
    feature_names:
        Names corresponding to the columns of ``X``.
    show:
        Whether to execute ``plt.show()`` after generating each figure.

    Returns
    -------
    dict
        Dictionary with the PCA and scatter-matrix figures keyed as
        ``{"pca": Figure, "scatter": Figure}``.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA

    classes = np.unique(y)
    tab10 = plt.cm.tab10(np.linspace(0, 1, 10))

    pca = PCA(n_components=3).fit(X)
    Xp = pca.transform(X)

    fig_pca = plt.figure(figsize=(8, 7))
    ax = fig_pca.add_subplot(111, projection="3d")
    for c in classes:
        mask = y == c
        ax.scatter(
            Xp[mask, 0],
            Xp[mask, 1],
            Xp[mask, 2],
            s=15,
            alpha=0.7,
            label=f"Clase {c}",
            color=tab10[int(c)],
        )
    ax.set_title("PCA 3D del dataset")
    ax.legend()
    if show:
        plt.show()

    df = pd.DataFrame(X, columns=feature_names)
    df["class"] = y
    axes = pd.plotting.scatter_matrix(
        df,
        c=[tab10[int(c)] for c in df["class"]],
        figsize=(10, 10),
        alpha=0.5,
        diagonal="hist",
    )
    fig_scatter = axes[0, 0].get_figure()
    fig_scatter.suptitle("Matriz de dispersión 4D", y=1.02)
    if show:
        plt.show()

    return {"pca": fig_pca, "scatter": fig_scatter}


__all__ = [
    "make_corner_class_dataset",
    "make_high_dim_classification_dataset",
    "plot_corner_class_dataset",
]
