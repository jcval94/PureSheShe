"""Run parameter sweeps for prune_and_orient_planes_unified_globalmaj."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification

from deldel.datasets import make_corner_class_dataset, make_high_dim_classification_dataset
from deldel.globalmaj import prune_and_orient_planes_unified_globalmaj


@dataclass
class DatasetSpec:
    name: str
    builder: callable
    kwargs: dict


def _axis_aligned_frontier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    thresholds_per_dim: int = 3,
) -> Dict[Tuple[int, int], Dict[str, dict]]:
    """Generate a lightweight frontier payload built from axis-aligned cuts.

    The helper mimics the structure expected by
    :func:`prune_and_orient_planes_unified_globalmaj` by emitting
    ``planes_by_label`` entries for every class pair. For each pair, a set of
    candidate hyperplanes is produced along individual feature axes using
    midpoints between class means and broad percentiles of the joint
    distribution.
    """

    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    classes = sorted(np.unique(y))
    res: Dict[Tuple[int, int], Dict[str, dict]] = {}

    for a, b in itertools.combinations(classes, 2):
        mask_a = y == int(a)
        mask_b = y == int(b)
        Xa, Xb = X[mask_a], X[mask_b]
        planes_by_label: Dict[int, List[dict]] = {int(a): [], int(b): []}

        for dim in range(X.shape[1]):
            va = Xa[:, dim]
            vb = Xb[:, dim]
            if va.size == 0 or vb.size == 0:
                continue

            mean_a = float(np.mean(va))
            mean_b = float(np.mean(vb))
            joint = np.concatenate([va, vb])
            # Midpoints between class means plus broad percentiles encourage
            # diversity without exploding the candidate count.
            span = np.linspace(mean_a, mean_b, thresholds_per_dim + 2)[1:-1]
            percentiles = np.percentile(joint, [25, 50, 75])
            thresholds = sorted({float(t) for t in np.concatenate([span, percentiles])})

            for thr in thresholds:
                n = np.zeros(X.shape[1], dtype=float)
                n[dim] = 1.0
                plane = dict(n=n.tolist(), b=float(-thr))
                planes_by_label[int(a)].append(dict(plane))
                planes_by_label[int(b)].append(dict(plane))

        res[(int(a), int(b))] = dict(planes_by_label=planes_by_label)

    return res


def _make_mixed_classification_dataset(
    *,
    n_samples: int,
    n_features: int,
    class_sep: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 6),
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=class_sep,
        shuffle=True,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X.astype(float), y.astype(int), feature_names


def run_sweep(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    datasets: List[DatasetSpec] = [
        DatasetSpec(
            name="corner_small",
            builder=make_corner_class_dataset,
            kwargs=dict(n_per_cluster=140, std_class1=0.45, std_other=0.8, a=2.5, random_state=0),
        ),
        DatasetSpec(
            name="corner_large",
            builder=make_corner_class_dataset,
            kwargs=dict(n_per_cluster=260, std_class1=0.55, std_other=0.9, a=3.0, random_state=1),
        ),
        DatasetSpec(
            name="highdim_mid",
            builder=make_high_dim_classification_dataset,
            kwargs=dict(n_samples=6000, n_features=18, n_informative=12, n_redundant=2, class_sep=1.6, random_state=2),
        ),
        DatasetSpec(
            name="highdim_large",
            builder=make_high_dim_classification_dataset,
            kwargs=dict(n_samples=12000, n_features=28, n_informative=20, n_redundant=3, class_sep=1.7, random_state=3),
        ),
        DatasetSpec(
            name="mixed_easy",
            builder=_make_mixed_classification_dataset,
            kwargs=dict(n_samples=4500, n_features=12, class_sep=1.4, random_state=4),
        ),
    ]

    param_grid = [
        dict(family_clustering_mode="greedy", tau_mult=0.50, min_region_size=20),
        dict(family_clustering_mode="greedy", tau_mult=0.75, min_region_size=35),
        dict(family_clustering_mode="connected", tau_mult=0.50, min_region_size=20),
        dict(family_clustering_mode="connected", tau_mult=0.75, min_region_size=35),
        dict(family_clustering_mode="dbscan", tau_mult=0.50, min_region_size=20, family_dbscan_eps=0.30),
        dict(family_clustering_mode="dbscan", tau_mult=0.75, min_region_size=35, family_dbscan_eps=0.30),
    ]

    fieldnames = [
        "dataset",
        "n_samples",
        "n_features",
        "param_family_mode",
        "tau_mult",
        "min_region_size",
        "family_dbscan_eps",
        "pair",
        "balacc",
        "precision",
        "recall",
        "f1",
        "winning_planes",
        "candidates_pair",
        "candidates_global",
        "regions_global",
    ]

    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for ds in datasets:
            X, y, feature_names = ds.builder(**ds.kwargs)
            res = _axis_aligned_frontier(X, y)

            for params in param_grid:
                selection = prune_and_orient_planes_unified_globalmaj(
                    res,
                    X,
                    y,
                    feature_names=list(feature_names),
                    min_abs_diff=0.01,
                    min_rel_lift=0.05,
                    max_k=6,
                    min_recall=0.75,
                    min_region_frac=0.05,
                    **params,
                )

                regions_global = selection.get("regions_global", {}).get("per_plane", [])
                candidates_global = selection.get("candidates_global", [])

                for pair, payload in selection.get("by_pair_augmented", {}).items():
                    metrics = payload.get("metrics_overall", {})
                    writer.writerow(
                        dict(
                            dataset=ds.name,
                            n_samples=int(X.shape[0]),
                            n_features=int(X.shape[1]),
                            param_family_mode=params["family_clustering_mode"],
                            tau_mult=params.get("tau_mult"),
                            min_region_size=params.get("min_region_size"),
                            family_dbscan_eps=params.get("family_dbscan_eps"),
                            pair=f"{int(pair[0])}-{int(pair[1])}",
                            balacc=float(metrics.get("balacc", 0.0)),
                            precision=float(metrics.get("precision", 0.0)),
                            recall=float(metrics.get("recall", 0.0)),
                            f1=float(metrics.get("f1", 0.0)),
                            winning_planes=len(payload.get("winning_planes", [])),
                            candidates_pair=int(payload.get("meta", {}).get("num_candidates", 0)),
                            candidates_global=len(candidates_global),
                            regions_global=len(regions_global),
                        )
                    )


if __name__ == "__main__":
    out_path = Path(__file__).with_name("sweep_results.csv")
    run_sweep(out_path)
    print(f"CSV written to {out_path}")
