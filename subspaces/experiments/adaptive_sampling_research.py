"""Herramientas para analizar la representatividad del muestreo adaptativo."""

from __future__ import annotations

import csv
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


if "deldel" not in sys.modules:
    deldel_module = types.ModuleType("deldel")
    engine_module = types.ModuleType("deldel.engine")

    class DeltaRecord:  # pragma: no cover - placeholder para importaciones
        """Stub ligero para satisfacer las dependencias de importación."""

        pass

    engine_module.DeltaRecord = DeltaRecord

    subspace_module = types.ModuleType("deldel.subspace_change_detector")

    class MultiClassSubspaceExplorer:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError(
                "Este stub de MultiClassSubspaceExplorer no está diseñado para su uso."
            )

    class SubspaceReport:  # pragma: no cover - placeholder
        pass

    subspace_module.MultiClassSubspaceExplorer = MultiClassSubspaceExplorer
    subspace_module.SubspaceReport = SubspaceReport

    sys.modules["deldel"] = deldel_module
    sys.modules["deldel.engine"] = engine_module
    sys.modules["deldel.subspace_change_detector"] = subspace_module


from .core_method_bundle import _maybe_adaptive_sample


@dataclass(frozen=True)
class Scenario:
    """Describe un escenario de dataset a evaluar."""

    label: str
    shape: Tuple[int, ...]
    replicates: int


def _summarise(errors: Sequence[float]) -> Tuple[float, float, float]:
    """Devuelve media, desviación estándar y cuantíl 95 de una métrica."""

    values = np.asarray(errors, dtype=float)
    return float(values.mean()), float(values.std(ddof=0)), float(np.quantile(values, 0.95))


def evaluate_scenario(scenario: Scenario) -> dict:
    """Ejecuta la simulación para un escenario y devuelve métricas agregadas."""

    rng = np.random.default_rng(42)
    data = rng.normal(size=scenario.shape)
    n_samples = scenario.shape[0]
    flattened_feature_volume = int(np.prod(scenario.shape[1:]) or 1)
    flat_data = data.reshape(n_samples, -1)
    population_mean = flat_data.mean(axis=0)
    population_std = flat_data.std(axis=0, ddof=1)

    mean_abs_errors: list[float] = []
    rmse_errors: list[float] = []
    max_abs_errors: list[float] = []
    sample_sizes: list[int] = []

    y = np.arange(n_samples)

    for replicate in range(scenario.replicates):
        X_sampled, _, _, sample_size = _maybe_adaptive_sample(
            data,
            y,
            records=[],
            enabled=True,
            random_state=replicate,
        )

        sample_flat = np.asarray(X_sampled).reshape(sample_size, -1)
        sample_mean = sample_flat.mean(axis=0)
        delta = sample_mean - population_mean

        mean_abs_errors.append(float(np.mean(np.abs(delta))))
        rmse_errors.append(float(np.sqrt(np.mean(delta**2))))
        max_abs_errors.append(float(np.max(np.abs(delta))))
        sample_sizes.append(int(sample_size))

    assert len(set(sample_sizes)) == 1, "El tamaño de muestra debe ser estable por escenario"
    sample_size = sample_sizes[0]

    mae_mean, mae_std, mae_p95 = _summarise(mean_abs_errors)
    rmse_mean, rmse_std, rmse_p95 = _summarise(rmse_errors)
    max_error_mean, max_error_std, max_error_p95 = _summarise(max_abs_errors)

    theoretical_se = population_std / np.sqrt(sample_size)
    theoretical_se_flat = theoretical_se.mean()
    mae_to_theoretical = mae_mean / theoretical_se_flat if theoretical_se_flat else float("nan")
    rmse_to_theoretical = rmse_mean / theoretical_se_flat if theoretical_se_flat else float("nan")

    return {
        "scenario": scenario.label,
        "shape": "x".join(str(dim) for dim in scenario.shape),
        "n_samples": n_samples,
        "feature_volume": flattened_feature_volume,
        "sample_size": sample_size,
        "sample_fraction": sample_size / n_samples,
        "replicates": scenario.replicates,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "mae_p95": mae_p95,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "rmse_p95": rmse_p95,
        "max_abs_mean": max_error_mean,
        "max_abs_std": max_error_std,
        "max_abs_p95": max_error_p95,
        "theoretical_mean_se": float(theoretical_se_flat),
        "mae_to_theoretical": mae_to_theoretical,
        "rmse_to_theoretical": rmse_to_theoretical,
    }


def run_representativeness_study(
    output_path: Path,
    *,
    scenarios: Iterable[Scenario],
) -> list[dict]:
    """Evalúa los escenarios y guarda un CSV con los resultados."""

    rows = [evaluate_scenario(scenario) for scenario in scenarios]

    fieldnames = list(rows[0].keys()) if rows else []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    """Punto de entrada CLI para lanzar la simulación predefinida."""

    study_scenarios = [
        Scenario("100x8", (100, 8), replicates=30),
        Scenario("1000x8", (1000, 8), replicates=30),
        Scenario("10000x8", (10000, 8), replicates=25),
        Scenario("100000x8", (100000, 8), replicates=20),
        Scenario("500000x8", (500000, 8), replicates=12),
        Scenario("100000x32", (100000, 32), replicates=18),
        Scenario("200000x64", (200000, 64), replicates=12),
        Scenario("20000x4x5", (20000, 4, 5), replicates=20),
    ]

    output = Path("experiments_outputs/adaptive_sampling_representativeness.csv")
    run_representativeness_study(output, scenarios=study_scenarios)


if __name__ == "__main__":
    main()
