"""Execute MultiClassSubspaceExplorer en varios datasets y resumir resultados.

El script recrea el análisis de combinaciones de métodos sobre múltiples
dataframes de referencia (incluyendo Iris y un conjunto sintético inspirado en
Titanic) y genera salidas CSV con:

* Tiempo empleado por cada método durante la generación de candidatos.
* Cuántos conjuntos aparecen marcados con "Si" en el top-50 y en todo el
  universo de combinaciones para cada método.

Los datasets se procesan sin depender de pandas: se construye una vista
columnar ligera compatible con ``MultiClassSubspaceExplorer`` y se emplea un
``RandomForestClassifier`` para obtener los ``DeltaRecord`` necesarios para
DelDel.
"""

from __future__ import annotations

import csv
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import io
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from deldel.engine import ChangePointConfig, DelDel, DelDelConfig
from deldel.subspace_change_detector import MultiClassSubspaceExplorer

from subspaces.experiments.run_subspace_experiments import (
    _method_sets_rows,
    _method_sort_key,
)

os.environ.setdefault(
    "PYTHONWARNINGS", "ignore::sklearn.exceptions.ConvergenceWarning"
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


NUM_METHODS = 20
OUTPUT_DIR = Path("subspaces/outputs/method_sets")
SUMMARY_PATH = OUTPUT_DIR / "multi_dataset_method_summary.csv"
DATASET_SUMMARY_PATH = OUTPUT_DIR / "multi_dataset_dataset_summary.csv"
TOP5_SUMMARY_PATH = OUTPUT_DIR / "multi_dataset_top5_summary.csv"
OVERALL_METHOD_SUMMARY_PATH = OUTPUT_DIR / "multi_dataset_overall_method_summary.csv"
OVERALL_TOP5_PATH = OUTPUT_DIR / "multi_dataset_overall_top5.csv"


class ColumnarView:
    """Vista minimalista tipo DataFrame."""

    def __init__(self, columns: Sequence[str], arrays: Sequence[np.ndarray]):
        if len(columns) != len(arrays):
            raise ValueError("El número de columnas no coincide con los datos")
        self.columns = list(columns)
        self.arrays = [np.asarray(arr).reshape(-1) for arr in arrays]
        self._mapping = {name: arr for name, arr in zip(self.columns, self.arrays)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._mapping[key]


@dataclass
class DatasetSpec:
    key: str
    name: str
    table: ColumnarView
    X_numeric: np.ndarray
    y: np.ndarray
    class_labels: np.ndarray


@contextmanager
def _suppress_stdout() -> Iterable[None]:
    original_out = sys.stdout
    original_err = sys.stderr
    buffer = io.StringIO()
    try:
        sys.stdout = buffer
        sys.stderr = buffer
        yield
    finally:
        sys.stdout = original_out
        sys.stderr = original_err


def _normalize(value: str) -> str:
    return value.strip().lower()


def _encode_categorical(values: Iterable[object]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=object)
    cleaned = []
    for item in arr:
        if item is None:
            cleaned.append("missing")
        elif isinstance(item, float) and math.isnan(item):
            cleaned.append("missing")
        else:
            cleaned.append(str(item))
    cleaned_arr = np.asarray(cleaned, dtype=object)
    uniques, inverse = np.unique(cleaned_arr, return_inverse=True)
    if uniques.size == 0:
        return np.zeros(cleaned_arr.shape[0], float)
    return inverse.astype(float)


def _encode_numeric_matrix(table: ColumnarView) -> np.ndarray:
    columns: List[np.ndarray] = []
    for arr in table.arrays:
        arr_np = np.asarray(arr)
        if arr_np.dtype.kind in {"b", "i", "u", "f"}:
            col = arr_np.astype(float)
            if np.isnan(col).any():
                mask = ~np.isnan(col)
                fill = float(np.nanmean(col[mask])) if mask.any() else 0.0
                col[~mask] = fill
            columns.append(col)
        else:
            columns.append(_encode_categorical(arr_np))
    if not columns:
        raise ValueError("No se encontraron columnas para codificar")
    return np.column_stack(columns)


def _factorize_labels(y: Sequence[object]) -> Tuple[np.ndarray, np.ndarray]:
    classes, inverse = np.unique(np.asarray(y), return_inverse=True)
    return inverse.astype(int), classes


def _build_dataset(
    *, key: str, name: str, columns: Sequence[str], arrays: Sequence[np.ndarray], y: Sequence[object]
) -> DatasetSpec:
    table = ColumnarView(columns, arrays)
    encoded_y, classes = _factorize_labels(y)
    numeric = _encode_numeric_matrix(table)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    return DatasetSpec(
        key=key,
        name=name,
        table=table,
        X_numeric=numeric_scaled,
        y=encoded_y,
        class_labels=classes,
    )


def _load_iris() -> DatasetSpec:
    from sklearn.datasets import load_iris

    iris = load_iris()
    rng = np.random.default_rng(0)
    n_samples = min(120, iris.data.shape[0])
    idx = np.sort(rng.choice(iris.data.shape[0], size=n_samples, replace=False))
    columns = list(iris.feature_names)
    arrays = [iris.data[idx, i] for i in range(iris.data.shape[1])]
    target = iris.target[idx]
    return _build_dataset(
        key="iris",
        name="Iris",
        columns=columns,
        arrays=arrays,
        y=target,
    )


def _load_wine() -> DatasetSpec:
    from sklearn.datasets import load_wine

    wine = load_wine()
    rng = np.random.default_rng(1)
    n_samples = min(150, wine.data.shape[0])
    idx = np.sort(rng.choice(wine.data.shape[0], size=n_samples, replace=False))
    columns = list(wine.feature_names)
    arrays = [wine.data[idx, i] for i in range(wine.data.shape[1])]
    target = wine.target[idx]
    return _build_dataset(
        key="wine",
        name="Wine",
        columns=columns,
        arrays=arrays,
        y=target,
    )


def _load_breast_cancer() -> DatasetSpec:
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    rng = np.random.default_rng(0)
    n_samples = min(320, cancer.data.shape[0])
    idx = np.sort(rng.choice(cancer.data.shape[0], size=n_samples, replace=False))
    feature_idx = list(range(min(18, cancer.data.shape[1])))
    columns = [cancer.feature_names[i] for i in feature_idx]
    arrays = [cancer.data[idx, i] for i in feature_idx]
    target = cancer.target[idx]
    return _build_dataset(
        key="breast_cancer",
        name="Breast cancer",
        columns=columns,
        arrays=arrays,
        y=target,
    )


def _load_titanic_synthetic(n_samples: int = 600, random_state: int = 7) -> DatasetSpec:
    rng = np.random.default_rng(random_state)
    pclass = rng.choice([1, 2, 3], size=n_samples, p=[0.24, 0.2, 0.56])
    sex = rng.choice(["male", "female"], size=n_samples, p=[0.64, 0.36])
    age = np.clip(rng.normal(loc=29.0, scale=14.0, size=n_samples), 0.5, 80.0)
    sibsp = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.68, 0.18, 0.07, 0.04, 0.02, 0.01])
    parch = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.76, 0.15, 0.05, 0.03, 0.01])
    fare = np.clip(rng.gamma(shape=2.1, scale=18.0, size=n_samples), 4.0, 300.0)
    embarked = rng.choice(["C", "Q", "S"], size=n_samples, p=[0.2, 0.1, 0.7])
    deck = rng.choice(["A", "B", "C", "D", "E", "F", "G", "Unknown"], size=n_samples, p=[0.05, 0.06, 0.19, 0.07, 0.09, 0.08, 0.04, 0.42])
    alone = (sibsp + parch == 0).astype(int)
    family_size = sibsp + parch + 1

    young_mask = (age < 14).astype(float)
    senior_mask = (age > 60).astype(float)
    logistic = (
        -0.65
        + 0.95 * (sex == "female")
        + 0.75 * (pclass == 1)
        + 0.25 * (pclass == 2)
        - 0.45 * (pclass == 3)
        - 0.015 * (age - 30.0)
        + 0.002 * (fare - 40.0)
        - 0.18 * (family_size >= 5)
        + 0.35 * (alone == 1)
        + 0.55 * young_mask
        - 0.35 * senior_mask
    )
    probs = 1.0 / (1.0 + np.exp(-logistic))
    survived = rng.binomial(1, np.clip(probs, 0.02, 0.98))

    columns = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "deck",
        "alone",
        "family_size",
    ]
    arrays = [
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked,
        deck,
        alone,
        family_size,
    ]
    return _build_dataset(
        key="titanic_synth",
        name="Titanic (sintético)",
        columns=columns,
        arrays=arrays,
        y=survived,
    )


DATASET_LOADERS: Sequence[Callable[[], DatasetSpec]] = (
    _load_iris,
    _load_wine,
    _load_breast_cancer,
    _load_titanic_synthetic,
)


def _count_si(values: Sequence[str]) -> int:
    return sum(1 for value in values if _normalize(value) == "si")


def _summarize_methods(
    method_rows: List[List[object]],
    method_keys: Sequence[str],
    friendly_names: Dict[str, str],
    method_timings: Dict[str, float],
    *,
    top_k: int = 50,
) -> Tuple[List[Dict[str, object]], Dict[int, int], Dict[int, int], int, int]:
    if not method_rows or len(method_rows[0]) < 2:
        raise ValueError("Tabla de métodos vacía")

    header = method_rows[0]
    method_columns = header[1:-1][:NUM_METHODS]

    entries: List[Dict[str, object]] = []
    for raw in method_rows[1:]:
        if not raw:
            continue
        set_name = raw[0]
        values = [str(v) for v in raw[1:-1][:NUM_METHODS]]
        try:
            score = float(raw[-1]) if str(raw[-1]).strip() else 0.0
        except ValueError:
            score = 0.0
        entries.append(
            {
                "set": set_name,
                "values": values,
                "score": score,
                "si_count": _count_si(values),
            }
        )

    entries.sort(key=lambda item: (-item["si_count"], -item["score"], item["set"]))
    top_entries = entries[: min(top_k, len(entries))]

    total_counts: Dict[int, int] = {idx: 0 for idx in range(1, NUM_METHODS + 1)}
    top_counts: Dict[int, int] = {idx: 0 for idx in range(1, NUM_METHODS + 1)}

    for entry in entries:
        for idx, value in enumerate(entry["values"], start=1):
            if idx > NUM_METHODS:
                break
            if _normalize(value) == "si":
                total_counts[idx] += 1

    for entry in top_entries:
        for idx, value in enumerate(entry["values"], start=1):
            if idx > NUM_METHODS:
                break
            if _normalize(value) == "si":
                top_counts[idx] += 1

    summary_rows: List[Dict[str, object]] = []
    for position, method_key in enumerate(method_keys[:NUM_METHODS], start=1):
        friendly = friendly_names.get(method_key, method_key)
        summary_rows.append(
            {
                "method_index": position,
                "method_internal": method_key,
                "method_label": friendly,
                "elapsed_seconds": float(method_timings.get(friendly, 0.0)),
                "top_50_si_count": top_counts.get(position, 0),
                "total_si_count": total_counts.get(position, 0),
            }
        )

    return summary_rows, top_counts, total_counts, len(top_entries), len(entries)


def _run_on_dataset(spec: DatasetSpec) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    model = RandomForestClassifier(n_estimators=160, random_state=0, n_jobs=-1)
    model.fit(spec.X_numeric, spec.y)

    deldel_cfg = DelDelConfig(log_level=40, segments_target=48)
    cp_cfg = ChangePointConfig(enabled=False)
    deldel = DelDel(deldel_cfg, cp_cfg)
    with _suppress_stdout():
        deldel.fit(spec.X_numeric, model)

    explorer = MultiClassSubspaceExplorer(
        random_state=0,
        mi_max_features=min(80, len(spec.table.columns)),
        corr_max_features=min(80, len(spec.table.columns)),
        max_sets=10,
        combo_sizes=(2, 3),
        chi2_pool=8,
        random_samples=10,
        cv_splits=2,
        max_total_candidates=120,
    )

    t0 = perf_counter()
    with _suppress_stdout():
        explorer.fit(spec.table, spec.y, deldel.records_, method_key=None)
    explorer_time = perf_counter() - t0
    reports = explorer.get_report()

    method_rows = _method_sets_rows(explorer, reports)
    method_keys = sorted(explorer.method_name_map_.keys(), key=_method_sort_key)
    summary_rows, top_counts, total_counts, top_size, total_size = _summarize_methods(
        method_rows,
        method_keys,
        explorer.method_name_map_,
        explorer.method_timings_,
    )

    dataset_info = {
        "dataset_key": spec.key,
        "dataset_name": spec.name,
        "n_samples": spec.X_numeric.shape[0],
        "n_features": len(spec.table.columns),
        "n_classes": len(spec.class_labels),
        "explorer_runtime_s": explorer_time,
        "total_sets": total_size,
        "top_50_size": top_size,
        "records_total": len(deldel.records_),
        "mean_total_si": float(np.mean(list(total_counts.values()))) if total_counts else 0.0,
        "mean_top50_si": float(np.mean(list(top_counts.values()))) if top_counts else 0.0,
    }

    return summary_rows, dataset_info, method_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, object]] = []
    dataset_records: List[Dict[str, object]] = []
    top5_records: List[Dict[str, object]] = []

    for loader in DATASET_LOADERS:
        spec = loader()
        method_summary, dataset_info, method_rows = _run_on_dataset(spec)

        dataset_records.append(dataset_info)
        for row in method_summary:
            summary_records.append({
                "dataset_key": spec.key,
                "dataset_name": spec.name,
                **row,
            })

        sorted_methods = sorted(
            method_summary,
            key=lambda row: (
                -int(row["total_si_count"]),
                -int(row["top_50_si_count"]),
                float(row["elapsed_seconds"]),
                row["method_index"],
            ),
        )

        for rank, method in enumerate(sorted_methods[:5], start=1):
            top5_records.append(
                {
                    "dataset_key": spec.key,
                    "dataset_name": spec.name,
                    "rank": rank,
                    "method_index": method["method_index"],
                    "method_internal": method["method_internal"],
                    "method_label": method["method_label"],
                    "elapsed_seconds": method["elapsed_seconds"],
                    "top_50_si_count": method["top_50_si_count"],
                    "total_si_count": method["total_si_count"],
                }
            )

        dataset_csv = OUTPUT_DIR / f"method_sets_{spec.key}.csv"
        with dataset_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(method_rows)

    aggregated_rows: List[Dict[str, object]] = []
    overall_top5_rows: List[Dict[str, object]] = []

    if summary_records:
        fieldnames = [
            "dataset_key",
            "dataset_name",
            "method_index",
            "method_internal",
            "method_label",
            "elapsed_seconds",
            "top_50_si_count",
            "total_si_count",
        ]
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_records:
                writer.writerow({
                    key: (f"{value:.6f}" if isinstance(value, float) else value)
                    for key, value in row.items()
                })

        aggregated_map: Dict[str, Dict[str, object]] = {}
        for row in summary_records:
            method_internal = row["method_internal"]
            entry = aggregated_map.setdefault(
                method_internal,
                {
                    "method_index": int(row["method_index"]),
                    "method_label": row["method_label"],
                    "dataset_coverage": 0,
                    "top_50_si_total": 0,
                    "total_si_total": 0,
                    "elapsed_total": 0.0,
                    "elapsed_max": 0.0,
                },
            )
            entry["dataset_coverage"] += 1
            entry["top_50_si_total"] += int(row["top_50_si_count"])
            entry["total_si_total"] += int(row["total_si_count"])
            elapsed = float(row["elapsed_seconds"])
            entry["elapsed_total"] += elapsed
            entry["elapsed_max"] = max(entry["elapsed_max"], elapsed)

        for method_internal, data in aggregated_map.items():
            coverage = max(1, data["dataset_coverage"])
            elapsed_mean = data["elapsed_total"] / coverage
            aggregated_rows.append(
                {
                    "method_index": data["method_index"],
                    "method_internal": method_internal,
                    "method_label": data["method_label"],
                    "dataset_coverage": coverage,
                    "top_50_si_total": data["top_50_si_total"],
                    "total_si_total": data["total_si_total"],
                    "elapsed_total": data["elapsed_total"],
                    "elapsed_mean": elapsed_mean,
                    "elapsed_max": data["elapsed_max"],
                }
            )

        aggregated_rows.sort(key=lambda row: row["method_index"])

        ranked = sorted(
            aggregated_rows,
            key=lambda row: (
                -int(row["total_si_total"]),
                -int(row["top_50_si_total"]),
                float(row["elapsed_mean"]),
                int(row["method_index"]),
            ),
        )
        for rank, row in enumerate(ranked[:5], start=1):
            overall_top5_rows.append(
                {
                    "rank": rank,
                    "method_index": row["method_index"],
                    "method_internal": row["method_internal"],
                    "method_label": row["method_label"],
                    "dataset_coverage": row["dataset_coverage"],
                    "top_50_si_total": row["top_50_si_total"],
                    "total_si_total": row["total_si_total"],
                    "elapsed_total": row["elapsed_total"],
                    "elapsed_mean": row["elapsed_mean"],
                    "elapsed_max": row["elapsed_max"],
                }
            )

    if dataset_records:
        fieldnames = [
            "dataset_key",
            "dataset_name",
            "n_samples",
            "n_features",
            "n_classes",
            "records_total",
            "explorer_runtime_s",
            "total_sets",
            "top_50_size",
            "mean_total_si",
            "mean_top50_si",
        ]
        with DATASET_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset_records:
                formatted = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted[key] = f"{value:.6f}"
                    else:
                        formatted[key] = value
                writer.writerow(formatted)

    if aggregated_rows:
        fieldnames = [
            "method_index",
            "method_internal",
            "method_label",
            "dataset_coverage",
            "top_50_si_total",
            "total_si_total",
            "elapsed_total",
            "elapsed_mean",
            "elapsed_max",
        ]
        with OVERALL_METHOD_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated_rows:
                formatted = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted[key] = f"{value:.6f}"
                    else:
                        formatted[key] = value
                writer.writerow(formatted)

    if overall_top5_rows:
        fieldnames = [
            "rank",
            "method_index",
            "method_internal",
            "method_label",
            "dataset_coverage",
            "top_50_si_total",
            "total_si_total",
            "elapsed_total",
            "elapsed_mean",
            "elapsed_max",
        ]
        with OVERALL_TOP5_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in overall_top5_rows:
                formatted = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted[key] = f"{value:.6f}"
                    else:
                        formatted[key] = value
                writer.writerow(formatted)

    if top5_records:
        fieldnames = [
            "dataset_key",
            "dataset_name",
            "rank",
            "method_index",
            "method_internal",
            "method_label",
            "elapsed_seconds",
            "top_50_si_count",
            "total_si_count",
        ]
        with TOP5_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in top5_records:
                formatted = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted[key] = f"{value:.6f}"
                    else:
                        formatted[key] = value
                writer.writerow(formatted)


if __name__ == "__main__":
    main()
