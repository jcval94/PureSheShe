"""Utilities for comparing legacy and current stage inference behaviour.

This module exposes helpers that reproduce the historical implementation of
``_infer_stage`` and provide tooling to measure its outputs and runtime against
the optimised version that ships in :mod:`deldel.engine`.

The functions are intentionally lightweight so they can be reused both from the
test-suite (to assert compatibility) and from experiment scripts that persist
A/B metrics to CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import inspect
from time import perf_counter
from typing import Callable, Dict, Iterable, Iterator, List, Sequence
import logging

from deldel.utils import _verbosity_to_level

from . import engine

ScenarioCallable = Callable[[Callable[[], str]], str]


def infer_stage_legacy(*, verbosity: int = 0) -> str:
    """Snapshot of the pre-optimised stage inference helper."""

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.log(level, "infer_stage_legacy: inicio")

    try:
        for frame_info in inspect.stack():
            func = frame_info.function
            if func in {"scores", "_scores_raw", "_predict_labels"}:
                continue

            loc = frame_info.frame.f_locals
            if "self" in loc:
                try:
                    clsname = loc["self"].__class__.__name__
                except Exception as exc:
                    logger.log(level, "infer_stage_legacy: error obteniendo clase %s", exc)
                    clsname = None
                if clsname == "DelDel":
                    logger.log(level, "infer_stage_legacy: detectado DelDel.%s", func)
                    return f"{clsname}.{func}"

        top = inspect.stack()[1]
        module = top.frame.f_globals.get("__name__", "__main__")
        stage_val = f"{module}.{top.function}"
        logger.log(level, "infer_stage_legacy: retorno %s", stage_val)
        return stage_val
    except Exception as exc:
        logger.log(level, "infer_stage_legacy: excepción %s", exc)
        return "unknown"


def _call_in_fake_deldel(fn: Callable[[], str], *, verbosity: int = 0) -> str:
    """Invoke *fn* as if it was running inside ``DelDel.stage``."""

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.log(level, "_call_in_fake_deldel: envolviendo llamada")

    FakeDelDel = type("DelDel", (), {})

    def stage(self):
        return fn()

    FakeDelDel.stage = stage  # type: ignore[attr-defined]
    logger.log(level, "_call_in_fake_deldel: invocando stage")
    return FakeDelDel().stage()


def _call_plain(fn: Callable[[], str], *, verbosity: int = 0) -> str:
    """Invoke *fn* from a plain helper without any DelDel context."""

    logger = logging.getLogger(__name__)
    logger.log(_verbosity_to_level(verbosity), "_call_plain: ejecución directa")

    def helper():
        return fn()

    return helper()


@dataclass(frozen=True)
class Scenario:
    """Descriptor for a stage inference scenario."""

    name: str
    executor: ScenarioCallable
    metadata: Dict[str, object] = field(default_factory=dict)

    def run(self, fn: Callable[[], str], *, verbosity: int = 0) -> str:
        logger = logging.getLogger(__name__)
        level = _verbosity_to_level(verbosity)
        logger.log(level, "Scenario.run: %s inicio", self.name)
        t0 = perf_counter()
        try:
            return self.executor(fn)
        finally:
            logger.log(level, "Scenario.run: %s fin en %.6fs", self.name, perf_counter() - t0)


@contextmanager
def _override_infer_stage(fn: Callable[[], str], *, verbosity: int = 0) -> Iterator[None]:
    """Temporarily redirect :func:`engine._infer_stage` to *fn*."""

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.log(level, "_override_infer_stage: instalando override")
    original = engine._infer_stage
    engine._infer_stage = fn  # type: ignore[assignment]
    try:
        yield
    finally:
        engine._infer_stage = original  # type: ignore[assignment]
        logger.log(level, "_override_infer_stage: restaurado")


def _score_adaptor_dataframe_executor(
    *,
    values,
    feature_names: Sequence[str],
    label: str,
    verbosity: int = 0,
) -> ScenarioCallable:
    """Build an executor that exercises ``ScoreAdaptor.scores`` via pandas."""

    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.log(level, "_score_adaptor_dataframe_executor: label=%s rows=%d cols=%d", label, len(values), len(feature_names))

    import numpy as np
    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        import sys
        import types

        pd_module = types.ModuleType("pandas")

        class _StubDataFrame:
            def __init__(self, data, columns=None):
                self._array = np.asarray(data, float)
                self.columns = list(columns) if columns is not None else None

            def to_numpy(self, dtype=None, copy=False):
                array = self._array.astype(dtype) if dtype is not None else self._array
                return array.copy() if copy else array

        pd_module.DataFrame = _StubDataFrame  # type: ignore[attr-defined]
        sys.modules.setdefault("pandas", pd_module)
        pd = pd_module  # type: ignore[assignment]

    values_arr = np.asarray(values, float)

    class DummyModel:
        feature_names_in_ = list(feature_names)

        def predict_proba(self, X):
            if not isinstance(X, pd.DataFrame):
                raise AssertionError("Expected pandas.DataFrame input in scenario")
            data = X.to_numpy(dtype=float, copy=True)
            logits = data @ np.arange(1, data.shape[1] + 1)
            prob1 = 1.0 / (1.0 + np.exp(-logits))
            return np.column_stack([1.0 - prob1, prob1])

    adaptor = engine.ScoreAdaptor(DummyModel(), mode="proba", cache_enabled=True)

    def executor(fn: Callable[[], str]) -> str:
        log_entries: List[Dict[str, object]] = []
        with _override_infer_stage(fn, verbosity=max(verbosity - 1, 0)):
            stage_value = _call_in_fake_deldel(
                lambda: _run_scores_with_logging(adaptor, values_arr, log_entries, verbosity=max(verbosity - 1, 0)),
                verbosity=max(verbosity - 1, 0),
            )
        if not adaptor._cache:
            raise AssertionError(f"Scenario '{label}' expected ScoreAdaptor cache usage")
        if not log_entries:
            raise AssertionError("ScoreAdaptor scenario did not emit log entries")
        logger.log(level, "_score_adaptor_dataframe_executor: %s completado con %d logs", label, len(log_entries))
        return stage_value

    return executor


def _run_scores_with_logging(adaptor, values_arr, log_entries, *, verbosity: int = 0):
    logger = logging.getLogger(__name__)
    level = _verbosity_to_level(verbosity)
    logger.log(level, "_run_scores_with_logging: valores=%s", values_arr.shape)
    with engine._collect_calls(log_entries):
        adaptor.scores(values_arr.copy())
    logger.log(level, "_run_scores_with_logging: registros capturados=%d", len(log_entries))
    return str(log_entries[-1]["stage"]) if log_entries else "unknown"


SCENARIOS: Sequence[Scenario] = (
    Scenario(name="fake_deldel_stage", executor=_call_in_fake_deldel),
    Scenario(name="plain_helper", executor=_call_plain),
    Scenario(
        name="score_adaptor_dataframe_small",
        executor=_score_adaptor_dataframe_executor(
            values=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.5, 0.5, 0.5, 0.5]],
            feature_names=["f0", "f1", "f2", "f3"],
            label="dense_float",
        ),
        metadata={
            "dataframe_label": "dense_float",
            "dataframe_rows": 3,
            "dataframe_cols": 4,
            "cache_expected": "yes",
        },
    ),
    Scenario(
        name="score_adaptor_dataframe_duplicate_rows",
        executor=_score_adaptor_dataframe_executor(
            values=
            [
                [0.9, 0.1, 0.05, 0.8, 0.2],
                [0.9, 0.1, 0.05, 0.8, 0.2],
                [0.2, 0.8, 0.6, 0.4, 0.0],
                [0.2, 0.8, 0.6, 0.4, 0.0],
            ],
            feature_names=["alpha", "beta", "gamma", "delta", "epsilon"],
            label="duplicate_rows",
        ),
        metadata={
            "dataframe_label": "duplicate_rows",
            "dataframe_rows": 4,
            "dataframe_cols": 5,
            "duplicate_groups": 2,
            "cache_expected": "yes",
        },
    ),
)


def collect_stage_results() -> List[Dict[str, object]]:
    """Compare legacy and current outputs for all registered scenarios."""

    rows: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        legacy_value = scenario.run(infer_stage_legacy)
        current_value = scenario.run(engine._infer_stage)
        rows.append(
            dict(
                scenario=scenario.name,
                legacy_value=legacy_value,
                current_value=current_value,
                values_match="yes" if legacy_value == current_value else "no",
                **scenario.metadata,
            )
        )
    return rows


def benchmark_stage_variants(
    *,
    iterations: int | Sequence[int] = 10_000,
    repeat: int = 1,
) -> List[Dict[str, float]]:
    """Measure the runtime of the legacy and current helpers."""

    if isinstance(iterations, int):
        iteration_groups = [int(iterations)]
    else:
        iteration_groups = [int(it) for it in iterations]

    timings: List[Dict[str, float]] = []
    for iteration_count in iteration_groups:
        for run_index in range(1, int(repeat) + 1):
            legacy_duration = current_duration = 0.0
            for label, fn in (
                ("legacy", infer_stage_legacy),
                ("current", engine._infer_stage),
            ):
                start = perf_counter()
                for _ in range(iteration_count):
                    fn()
                duration = perf_counter() - start
                if label == "legacy":
                    legacy_duration = duration
                else:
                    current_duration = duration
                timings.append(
                    dict(
                        variant=label,
                        iterations=float(iteration_count),
                        run_index=float(run_index),
                        total_runtime_s=duration,
                        avg_runtime_us=(duration / iteration_count) * 1e6
                        if iteration_count
                        else 0.0,
                    )
                )

            ratio = (
                current_duration / legacy_duration
                if legacy_duration
                else float("nan")
            )
            timings.append(
                dict(
                    variant="ratio_current_over_legacy",
                    iterations=float(iteration_count),
                    run_index=float(run_index),
                    total_runtime_s=ratio,
                    avg_runtime_us=ratio,
                )
            )

    return timings


def run_infer_stage_ab(
    *,
    iterations: int | Sequence[int] = 10_000,
    repeat: int = 1,
) -> Dict[str, Iterable[Dict[str, object]]]:
    """Collect both accuracy and timing results for the stage helpers."""

    results = collect_stage_results()
    timings = benchmark_stage_variants(iterations=iterations, repeat=repeat)
    return {"results": results, "timings": timings}


def write_csv_rows(rows: Iterable[Dict[str, object]], path: str, *, field_order: Sequence[str] | None = None) -> None:
    """Serialize *rows* to *path* using CSV format."""

    import csv
    from pathlib import Path

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No rows provided for CSV serialization")

    if field_order is None:
        seen: List[str] = []
        for row in rows_list:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        field_order = seen

    with path_obj.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(field_order))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def write_infer_stage_ab_csvs(
    *,
    iterations: int | Sequence[int] = 10_000,
    repeat: int = 1,
    results_path: str,
    timings_path: str,
) -> Dict[str, Iterable[Dict[str, object]]]:
    """Run the comparison and persist both outputs as CSV files."""

    summary = run_infer_stage_ab(iterations=iterations, repeat=repeat)
    write_csv_rows(summary["results"], results_path)
    write_csv_rows(summary["timings"], timings_path)
    return summary

