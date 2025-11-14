"""Benchmark ligero del impacto de ``maybe_adaptive_sample`` sobre matrices sintéticas."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification

from .adaptive_sampling import AdaptiveSamplingInfo, maybe_adaptive_sample


def _dataset_configs() -> Sequence[Tuple[int, int]]:
    return ((2000, 16), (8000, 32), (20000, 48))


def _simulate_processing(matrix: np.ndarray) -> float:
    start = time.perf_counter()
    flat = matrix.reshape(matrix.shape[0], -1)
    _ = np.linalg.norm(flat, axis=1).sum()
    return time.perf_counter() - start


def _benchmark_case(n_samples: int, n_features: int) -> dict[str, object]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_classes=4,
        random_state=123,
    )

    start = time.perf_counter()
    sampled_X, sampled_y, _, info = maybe_adaptive_sample(
        X,
        y,
        records=list(range(len(y))),
        adaptive_sampling=True,
        random_state=123,
    )
    sampling_overhead = time.perf_counter() - start

    time_full_processing = _simulate_processing(X)
    time_sample_processing = _simulate_processing(sampled_X)

    time_with_sampling = sampling_overhead + time_sample_processing
    time_without_sampling = time_full_processing

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "sampled_size": info.sampled_size,
        "sample_fraction": info.sample_fraction,
        "time_sampling": sampling_overhead,
        "time_processing_sample": time_sample_processing,
        "time_processing_full": time_full_processing,
        "time_with_sampling": time_with_sampling,
        "time_without_sampling": time_without_sampling,
        "estimated_speedup": time_without_sampling / time_with_sampling if time_with_sampling else float("inf"),
    }


def run_benchmarks() -> Iterable[dict[str, object]]:
    for n_samples, n_features in _dataset_configs():
        yield _benchmark_case(n_samples, n_features)


def write_csv(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "core_bundle_sampling_timings.csv"
    rows = list(run_benchmarks())
    write_csv(output_path, rows)


if __name__ == "__main__":
    main()
