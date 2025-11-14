"""Genera métricas de representatividad para el muestreo adaptativo."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .adaptive_sampling import maybe_adaptive_sample


def _flatten_features(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1)


def _dataset_configs() -> Sequence[tuple[str, tuple[int, ...]]]:
    return (
        ("matrix_500k_x_8", (500_000, 8)),
        ("matrix_200k_x_64", (200_000, 64)),
        ("tensor_60k_x_8_x_4", (60_000, 8, 4)),
        ("matrix_120k_x_16", (120_000, 16)),
    )


def _generate_dataset(shape: tuple[int, ...], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=shape).astype(np.float32)
    y = rng.integers(0, 6, size=shape[0])
    return X, y


def _representativeness_metrics(
    X: np.ndarray,
    sampled_X: np.ndarray,
) -> tuple[float, float]:
    full_matrix = _flatten_features(X)
    sampled_matrix = _flatten_features(sampled_X)
    full_mean = full_matrix.mean(axis=0)
    sampled_mean = sampled_matrix.mean(axis=0)
    full_std = full_matrix.std(axis=0)
    sampled_std = sampled_matrix.std(axis=0)
    mean_abs_error = float(np.mean(np.abs(sampled_mean - full_mean)))
    std_abs_error = float(np.mean(np.abs(sampled_std - full_std)))
    return mean_abs_error, std_abs_error


def run_experiment(random_state: int = 42) -> Iterable[dict[str, object]]:
    rng = np.random.default_rng(random_state)
    for name, shape in _dataset_configs():
        X, y = _generate_dataset(shape, rng)
        records = list(range(shape[0]))
        sampled_X, sampled_y, sampled_records, info = maybe_adaptive_sample(
            X,
            y,
            records=records,
            adaptive_sampling=True,
            random_state=random_state,
        )
        mean_abs_error, std_abs_error = _representativeness_metrics(X, sampled_X)
        yield {
            "dataset": name,
            "shape": "x".join(str(dim) for dim in shape),
            "n_samples": int(info.original_size),
            "n_features": int(info.n_features),
            "total_size": int(info.total_size),
            "sampled_size": int(info.sampled_size),
            "sample_fraction": info.sample_fraction,
            "mean_abs_error": mean_abs_error,
            "std_abs_error": std_abs_error,
            "records_sampled": len(sampled_records),
            "sampling_enabled": info.sampling_enabled,
        }


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
    output_path = output_dir / "adaptive_sampling_representativeness.csv"
    rows = list(run_experiment())
    write_csv(output_path, rows)


if __name__ == "__main__":
    main()
