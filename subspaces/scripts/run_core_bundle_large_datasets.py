"""Ejecuta el bundle de métodos destacados sobre tres datasets grandes."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from deldel.engine import ChangePointConfig, DelDel, DelDelConfig

from subspaces.experiments.core_method_bundle import (
    CORE_METHOD_KEYS,
    run_core_method_bundle,
    run_single_method,
)


warnings.filterwarnings(
    "ignore",
    message=".*'multi_class'.*deprecated.*",
    category=FutureWarning,
)


OUTPUT_DIR = Path("subspaces/outputs/core_bundle")
SUMMARY_PATH = OUTPUT_DIR / "core_bundle_summary.csv"
DATASET_PATH = OUTPUT_DIR / "core_bundle_dataset_stats.csv"
BUNDLE_MAX_SETS = 3


@dataclass
class DatasetSpec:
    key: str
    name: str
    X: np.ndarray
    y: np.ndarray
    random_state: int


def _generate_large_dataset(
    *,
    key: str,
    name: str,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_classes: int,
    class_sep: float,
    random_state: int,
) -> DatasetSpec:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(2, n_features // 8),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return DatasetSpec(key=key, name=name, X=X_scaled, y=y, random_state=random_state)


def _fit_deldel_records(X: np.ndarray, y: np.ndarray, random_state: int) -> Iterable:
    model = RandomForestClassifier(n_estimators=160, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    cfg = DelDelConfig(log_level="ERROR", segments_target=48)
    cp_cfg = ChangePointConfig(enabled=False)
    d = DelDel(cfg, cp_cfg)
    d.fit(X, model)
    return d.records_


def _top_hits(
    explorer,
    reports,
    *,
    top_k: int,
) -> Dict[str, int]:
    sorted_reports = sorted(reports, key=lambda r: r.mean_macro_f1, reverse=True)[:top_k]
    counts: Dict[str, int] = {key: 0 for key in CORE_METHOD_KEYS}
    sources = explorer.candidate_sources_
    for report in sorted_reports:
        combo = tuple(sorted(report.features))
        origin = sources.get(combo, set())
        for method in CORE_METHOD_KEYS:
            if method in origin:
                counts[method] += 1
    return counts


def _validate_bundle(
    dataset: DatasetSpec,
    *,
    records,
    base_random_state: int,
    bundle_explorer,
    max_sets: int,
    cv_splits: int,
) -> None:
    bundle_sets = bundle_explorer.method_candidate_sets_
    for method in CORE_METHOD_KEYS:
        single = run_single_method(
            dataset.X,
            dataset.y,
            records,
            method,
            max_sets=max_sets,
            combo_sizes=bundle_explorer.combo_sizes,
            random_state=base_random_state,
            cv_splits=cv_splits,
        )
        single_sets = single.method_candidate_sets_.get(method, set())
        if bundle_sets.get(method, set()) != single_sets:
            raise RuntimeError(
                f"El método {method} no coincide entre el bundle y la ejecución individual"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets: Sequence[DatasetSpec] = [
        _generate_large_dataset(
            key="mix_large",
            name="Synthetic Mix Large",
            n_samples=1400,
            n_features=24,
            n_informative=16,
            n_classes=6,
            class_sep=1.5,
            random_state=11,
        ),
        _generate_large_dataset(
            key="wide_large",
            name="Synthetic Wide Large",
            n_samples=1300,
            n_features=30,
            n_informative=20,
            n_classes=5,
            class_sep=1.65,
            random_state=17,
        ),
        _generate_large_dataset(
            key="imbalanced_large",
            name="Synthetic Imbalanced Large",
            n_samples=1250,
            n_features=26,
            n_informative=16,
            n_classes=4,
            class_sep=1.3,
            random_state=23,
        ),
    ]

    summary_rows: List[List[object]] = [
        [
            "dataset_key",
            "dataset_name",
            "method_key",
            "method_name",
            "top50_si",
            "global_si",
            "runtime_seconds",
        ]
    ]
    dataset_rows: List[List[object]] = [
        [
            "dataset_key",
            "dataset_name",
            "n_samples",
            "n_features",
            "bundle_elapsed",
            "reports_evaluated",
        ]
    ]

    for dataset in datasets:
        records = _fit_deldel_records(dataset.X, dataset.y, random_state=dataset.random_state)
        start = perf_counter()
        result = run_core_method_bundle(
            dataset.X,
            dataset.y,
            records,
            max_sets=BUNDLE_MAX_SETS,
            combo_sizes=(2, 3),
            random_state=dataset.random_state,
            cv_splits=2,
        )
        elapsed = perf_counter() - start

        top_hits = _top_hits(result.explorer, result.reports, top_k=50)
        candidate_sets = result.explorer.method_candidate_sets_
        timings = result.explorer.method_timings_
        method_name_map = result.explorer.method_name_map_

        for method in CORE_METHOD_KEYS:
            friendly = method_name_map[method]
            summary_rows.append(
                [
                    dataset.key,
                    dataset.name,
                    method,
                    friendly,
                    top_hits.get(method, 0),
                    len(candidate_sets.get(method, set())),
                    round(float(timings.get(friendly, 0.0)), 6),
                ]
            )

        dataset_rows.append(
            [
                dataset.key,
                dataset.name,
                dataset.X.shape[0],
                dataset.X.shape[1],
                round(elapsed, 4),
                len(result.reports),
            ]
        )

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

    with DATASET_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(dataset_rows)


if __name__ == "__main__":
    main()

