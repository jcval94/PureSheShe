import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from deldel.engine import DelDel, DelDelConfig
from deldel.experiments import _build_demo_selection, make_corner_class_dataset
from deldel.frontier_planes_all_modes import compute_frontier_planes_all_modes
from deldel.find_low_dim_spaces_fast import find_low_dim_spaces
from deldel.profiling import ProfilingConfig

OUTPUT_DIR = Path("experiments_outputs")
CSV_PATH = OUTPUT_DIR / "profiling_ab_test.csv"
JSON_PATH = OUTPUT_DIR / "profiling_ab_test.json"


class DummyModel:
    def __init__(self, n_features: int, n_classes: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.W = rng.normal(size=(n_features, n_classes))
        self.b = rng.normal(size=(n_classes,))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def _timed(label: str, fn, *args, **kwargs) -> dict:
    t0 = perf_counter()
    fn(*args, **kwargs)
    elapsed = perf_counter() - t0
    return {"stage": label, "elapsed_s": elapsed}


def _run_deldel_ab() -> list[dict]:
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 4))
    model = DummyModel(n_features=4, n_classes=3)
    cfg = DelDelConfig(segments_target=20, log_level=40)

    baseline = DelDel(cfg)
    profiled = DelDel(cfg)

    prof_cfg = ProfilingConfig(enabled=True, label="ab_test", output_dir=OUTPUT_DIR)
    results = []
    results.append(_timed("DelDel.fit_baseline", baseline.fit, X, model, 0))
    results.append(_timed("DelDel.fit_profiled", profiled.fit, X, model, 0, prof_cfg))
    return results, profiled.records_


def _run_frontier_ab(records) -> list[dict]:
    prof_cfg = ProfilingConfig(enabled=True, label="ab_test", output_dir=OUTPUT_DIR)
    results = []
    results.append(
        _timed("compute_frontier_planes_all_modes_baseline", compute_frontier_planes_all_modes, records)
    )
    results.append(
        _timed(
            "compute_frontier_planes_all_modes_profiled",
            compute_frontier_planes_all_modes,
            records,
            profiling=prof_cfg,
        )
    )
    return results


def _run_find_low_dim_ab() -> list[dict]:
    X, y, feature_names = make_corner_class_dataset(n_per_cluster=60, std_class1=0.4, std_other=0.7)
    sel = _build_demo_selection(X, y, feature_names)
    prof_cfg = ProfilingConfig(enabled=True, label="ab_test", output_dir=OUTPUT_DIR)

    results = []
    results.append(
        _timed(
            "find_low_dim_spaces_baseline",
            find_low_dim_spaces,
            X,
            y,
            sel,
            feature_names=list(feature_names),
            max_planes_in_rule=2,
            max_planes_per_pair=1,
            min_support=10,
            consider_dims_up_to=2,
            verbosity=0,
        )
    )
    results.append(
        _timed(
            "find_low_dim_spaces_profiled",
            find_low_dim_spaces,
            X,
            y,
            sel,
            feature_names=list(feature_names),
            max_planes_in_rule=2,
            max_planes_per_pair=1,
            min_support=10,
            consider_dims_up_to=2,
            verbosity=0,
            profiling=prof_cfg,
        )
    )
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    deldel_results, records = _run_deldel_ab()
    results.extend(deldel_results)
    results.extend(_run_frontier_ab(records))
    results.extend(_run_find_low_dim_ab())

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
