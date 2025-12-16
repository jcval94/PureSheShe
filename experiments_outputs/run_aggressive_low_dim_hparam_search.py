"""Aggressive hyperparameter sweep for `find_low_dim_spaces` across datasets.

The script follows the EXPERIMENT_PROTOCOL pipeline: generate datasets,
train a classifier, extract frontier planes, orient/prune them, and finally
sweep `find_low_dim_spaces` hyperparameters. Results are appended to (or
created in) a CSV in this folder for further analysis. The current revision
executes the requested Grid B (324 combinaciones por dataset) on top of any
previous runs present in the CSV.
"""
from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deldel import (  # noqa: E402
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    compute_frontier_planes_all_modes,
    describe_regions_report,
    prune_and_orient_planes_unified_globalmaj,
)
from deldel.datasets import make_corner_class_dataset, make_high_dim_classification_dataset  # noqa: E402
from deldel.experiments import _build_demo_selection  # noqa: E402
from deldel.find_low_dim_spaces_fast import find_low_dim_spaces  # noqa: E402


@dataclass
class DatasetSpec:
    name: str
    builder: Callable[[], Tuple[np.ndarray, np.ndarray, List[str]]]
    description: str


def _generate_dataset_specs() -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []

    specs.append(
        DatasetSpec(
            name="corner",
            builder=lambda: make_corner_class_dataset(
                n_per_cluster=25,
                std_class1=0.38,
                std_other=0.58,
                a=2.0,
                random_state=0,
            ),
            description="Corner 4D dataset canonico",
        )
    )

    specs.append(
        DatasetSpec(
            name="medium_classification",
            builder=lambda: _make_classification_dataset(
                n_samples=200,
                n_features=6,
                n_informative=4,
                class_sep=1.2,
                random_state=1,
            ),
            description="Clasificación balanceada con 3 clases y 10 dimensiones",
        )
    )

    specs.append(
        DatasetSpec(
            name="wide_classification",
            builder=lambda: _make_classification_dataset(
                n_samples=240,
                n_features=7,
                n_informative=5,
                class_sep=1.35,
                random_state=2,
            ),
            description="Clasificación ancha inspirada en stress tests",
        )
    )

    specs.append(
        DatasetSpec(
            name="imbalanced",
            builder=lambda: _make_classification_dataset(
                n_samples=200,
                n_features=6,
                n_informative=4,
                class_sep=1.15,
                weights=(0.55, 0.3, 0.15),
                random_state=3,
            ),
            description="Clasificación desbalanceada controlada",
        )
    )

    specs.append(
        DatasetSpec(
            name="high_dim",
            builder=lambda: make_high_dim_classification_dataset(
                n_samples=260,
                n_features=8,
                n_informative=5,
                class_sep=1.25,
                random_state=4,
            ),
            description="Dataset alto dimensional reducido para barrido masivo",
        )
    )

    return specs


def _make_classification_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_informative: int,
    class_sep: float,
    random_state: int,
    weights: Tuple[float, float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if weights is not None:
        weights = list(weights)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=class_sep,
        weights=weights,
        random_state=random_state,
    )
    feature_names = [f"f{i+1}" for i in range(n_features)]
    return X.astype(float), y.astype(int), feature_names


def _build_sel(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    random_state: int,
) -> Dict[str, object]:
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X, y)

    cfg = DelDelConfig(segments_target=35, random_state=random_state)
    records = DelDel(cfg, ChangePointConfig(enabled=False)).fit(X, model).records_

    frontier = compute_frontier_planes_all_modes(
        records,
        mode="C",
        min_cluster_size=3,
        max_models_per_round=3,
        seed=random_state,
    )

    sel = prune_and_orient_planes_unified_globalmaj(
        frontier,
        X,
        y,
        feature_names=list(feature_names),
        max_k=6,
        min_improve=1e-3,
        min_region_size=8,
        min_abs_diff=0.02,
        min_rel_lift=0.05,
    )

    if not sel.get("winning_planes"):
        sel = _build_demo_selection(X, y, feature_names)
    return sel


def _param_grid(n: int, dims: int, seed: int) -> List[Dict[str, object]]:
    rng = np.random.RandomState(seed)
    grid: List[Dict[str, object]] = []
    for i in range(n):
        max_rule = int(rng.randint(1, 3))
        params = dict(
            max_planes_in_rule=max_rule,
            max_planes_per_pair=int(rng.randint(max_rule, max_rule + 2)),
            max_rules_per_dim=int(rng.randint(6, 15)),
            min_support=int(rng.randint(6, 16)),
            min_rel_gain_f1=float(rng.uniform(0.001, 0.08)),
            min_abs_gain_f1=float(rng.uniform(0.0, 0.05)),
            min_lift_prec=float(rng.uniform(1.01, 1.4)),
            min_abs_gain_prec=float(rng.uniform(0.0, 0.05)),
            consider_dims_up_to=int(rng.randint(2, min(3, dims) + 1)),
            sample_limit_per_r=int(rng.choice([500, 800, 1100, 1400, 1700, 2000])),
            enable_unions=False,
            union_max_pairs_per_bucket=int(rng.randint(50, 180)),
            rng_seed=int(rng.randint(0, 10_000)),
        )
        params["consider_dims_up_to"] = int(min(params["consider_dims_up_to"], dims))
        grid.append(params)
    return grid


def _grid_b(dims: int) -> List[Dict[str, object]]:
    """Deterministic grid requested by the user (324 combinaciones por dataset)."""
    grid: List[Dict[str, object]] = []
    for (
        consider_dims_up_to,
        max_planes_in_rule,
        max_planes_per_pair,
        min_lift_prec,
        min_rel_gain_f1,
        min_support,
        sample_limit_per_r,
    ) in product(
        [2, 5],
        [1, 4],
        [2, 6, 8],
        [1.0, 1.2, 1.6],
        [0.02, 0.07, 0.12],
        [8, 16, 32],
        [75],
    ):
        params: Dict[str, object] = {
            "consider_dims_up_to": int(min(consider_dims_up_to, dims)),
            "max_planes_in_rule": int(max_planes_in_rule),
            "max_planes_per_pair": int(max(max_planes_per_pair, max_planes_in_rule)),
            "min_lift_prec": float(min_lift_prec),
            "min_rel_gain_f1": float(min_rel_gain_f1),
            "min_support": int(min_support),
            "sample_limit_per_r": int(sample_limit_per_r),
        }
        grid.append(params)
    return grid


def _run_finder(
    X: np.ndarray,
    y: np.ndarray,
    sel: Mapping[str, object],
    feature_names: Sequence[str],
    params: Mapping[str, object],
) -> Tuple[float, Mapping[str, object]]:
    finder_kwargs = dict(params)
    finder_kwargs.setdefault("enable_logs", False)
    finder_kwargs.setdefault("compute_relations", False)
    finder_kwargs.setdefault("include_masks", False)

    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        valuable = find_low_dim_spaces(
            X,
            y,
            sel,
            feature_names=list(feature_names),
            **finder_kwargs,
        )
    runtime = time.perf_counter() - t0
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        averages = describe_regions_report(
            valuable,
            top_per_class=3,
            dataset_size=X.shape[0],
            return_average_metrics=True,
        )
    if isinstance(averages, dict):
        per_class = averages.get("per_class", {})
    else:
        per_class = {}
    mean_f1 = float(np.mean([vals.get("mean_f1", 0.0) for vals in per_class.values()])) if per_class else 0.0
    mean_lift = float(np.mean([vals.get("mean_lift_precision", 0.0) for vals in per_class.values()])) if per_class else 0.0
    total_regions = sum(len(valuable.get(dim, [])) for dim in valuable)
    return runtime, dict(
        valuable_counts={int(dim): len(valuable.get(dim, [])) for dim in valuable},
        per_class=per_class,
        mean_f1=mean_f1,
        mean_lift_precision=mean_lift,
        total_regions=total_regions,
    )


def _load_existing_progress(csv_path: Path) -> Tuple[Dict[str, int], int]:
    """Return max run_index per dataset and existing total rows (excluding header)."""
    if not csv_path.exists():
        return {}, 0

    dataset_max: Dict[str, int] = {}
    total_rows = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            ds = row.get("dataset")
            run_idx_raw = row.get("run_index")
            if ds is None or run_idx_raw is None:
                continue
            try:
                run_idx = int(run_idx_raw)
            except ValueError:
                continue
            dataset_max[ds] = max(dataset_max.get(ds, -1), run_idx)
    return dataset_max, total_rows


def main() -> None:
    specs = _generate_dataset_specs()
    csv_path = Path(__file__).resolve().parent / "aggressive_find_low_dim_sweep.csv"

    fieldnames = [
        "dataset",
        "dataset_desc",
        "dataset_n_samples",
        "dataset_n_dims",
        "run_index",
        "finder_runtime_s",
        "mean_f1",
        "mean_lift_precision",
        "total_regions",
        "per_class",
        "valuable_counts",
        "params_json",
    ]

    dataset_max_run, total_rows = _load_existing_progress(csv_path)
    mode = "a" if csv_path.exists() else "w"

    with csv_path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for ds_idx, spec in enumerate(specs):
            print(f"=== Preparando dataset {spec.name}")
            X, y, feature_names = spec.builder()
            sel = _build_sel(X, y, feature_names, random_state=ds_idx)
            start_run_idx = dataset_max_run.get(spec.name, -1) + 1

            # Solo barrido solicitado (324 combinaciones exactas)
            grid = _grid_b(X.shape[1])

            for run_idx, params in enumerate(grid):
                runtime, metrics = _run_finder(
                    X,
                    y,
                    sel,
                    feature_names,
                    params,
                )
                row = dict(
                    dataset=spec.name,
                    dataset_desc=spec.description,
                    dataset_n_samples=int(X.shape[0]),
                    dataset_n_dims=int(X.shape[1]),
                    run_index=start_run_idx + run_idx,
                    finder_runtime_s=f"{runtime:.6f}",
                    mean_f1=f"{metrics['mean_f1']:.6f}",
                    mean_lift_precision=f"{metrics['mean_lift_precision']:.6f}",
                    total_regions=metrics["total_regions"],
                    per_class=json.dumps(metrics["per_class"], sort_keys=True),
                    valuable_counts=json.dumps(metrics["valuable_counts"], sort_keys=True),
                    params_json=json.dumps(params, sort_keys=True),
                )
                writer.writerow(row)
                total_rows += 1
            print(
                f"Dataset {spec.name} completado | "
                f"filas dataset={len(grid)} | filas acumuladas={total_rows}"
            )

    print(f"Resultados exportados a {csv_path}")


if __name__ == "__main__":
    main()
