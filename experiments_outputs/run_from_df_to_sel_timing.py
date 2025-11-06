"""Run the README pipeline while measuring stage timings.

This script reproduces the example pipeline described in the README,
starting from the construction of a pandas DataFrame and ending with the
creation of the ``sel`` selection via
``prune_and_orient_planes_unified_globalmaj``. Multiple parameter
configurations are executed to observe how timings vary when adjusting
dataset and pipeline hyper-parameters. The aggregated timings are stored
in ``from_df_to_sel_time.csv`` in the repository root.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import sys
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sklearn.ensemble import RandomForestClassifier

from deldel import (
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    compute_frontier_planes_all_modes,
    compute_frontier_planes_weighted,
    make_corner_class_dataset,
    prune_and_orient_planes_unified_globalmaj,
)


@dataclass
class StageTiming:
    """Utility to capture elapsed time for a pipeline stage."""

    name: str
    start: float = field(default_factory=time.perf_counter)
    duration: float | None = None

    def stop(self) -> float:
        if self.duration is None:
            self.duration = time.perf_counter() - self.start
        return self.duration


def run_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the pipeline for a parameter configuration."""

    records: Dict[str, Any] = {"config_id": cfg["id"]}

    # Dataset generation
    t = StageTiming("dataset_generation")
    X, y, feature_names = make_corner_class_dataset(**cfg["dataset"])
    records["dataset_time_s"] = t.stop()

    # DataFrame creation (Python list-of-dicts emulating a DataFrame)
    t = StageTiming("dataframe_creation")
    dataframe = []
    for row, target in zip(X, y):
        record = {fname: float(value) for fname, value in zip(feature_names, row)}
        record["target"] = int(target)
        dataframe.append(record)
    records["dataframe_time_s"] = t.stop()
    records["dataframe_rows"] = len(dataframe)

    # Model training
    t = StageTiming("model_training")
    model = RandomForestClassifier(**cfg["model"])
    model.fit(X, y)
    records["model_time_s"] = t.stop()

    # DelDel execution
    deldel_cfg = DelDelConfig(**cfg["deldel"])
    cp_cfg = ChangePointConfig(**cfg["change_point"])
    t = StageTiming("deldel_fit")
    deldel_instance = DelDel(deldel_cfg, cp_cfg).fit(X, model)
    records["deldel_time_s"] = t.stop()

    # Weighted planes
    t = StageTiming("compute_frontier_planes_weighted")
    records_ = deldel_instance.records_
    planes_weighted = compute_frontier_planes_weighted(
        records_, **cfg["planes_weighted"]
    )
    records["planes_weighted_time_s"] = t.stop()
    records["num_weighted_planes"] = len(planes_weighted)

    # Frontier planes (all modes)
    pairs = sorted({tuple(sorted((r.y0, r.y1))) for r in records_ if r.y0 != r.y1})
    t = StageTiming("compute_frontier_planes_all_modes")
    res_c = compute_frontier_planes_all_modes(
        records_,
        pairs=pairs,
        **cfg["planes_all_modes"],
    )
    records["planes_all_modes_time_s"] = t.stop()
    records["num_all_mode_planes"] = sum(len(v) for v in res_c.values())

    # Selection
    t = StageTiming("prune_and_orient")
    selection_params = dict(cfg["selection"])
    if selection_params.get("feature_names") is None:
        selection_params["feature_names"] = list(feature_names)

    sel = prune_and_orient_planes_unified_globalmaj(
        res_c,
        X,
        y,
        **selection_params,
    )
    records["selection_time_s"] = t.stop()
    records["selection_size"] = len(sel)

    records["total_time_s"] = sum(
        records[key]
        for key in (
            "dataset_time_s",
            "dataframe_time_s",
            "model_time_s",
            "deldel_time_s",
            "planes_weighted_time_s",
            "planes_all_modes_time_s",
            "selection_time_s",
        )
    )

    # Persist full configuration used for traceability
    records["config_json"] = json.dumps({k: v for k, v in cfg.items() if k != "id"})

    return records


def main(configs: Iterable[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        rows.append(run_configuration(cfg))

    output_path = Path("from_df_to_sel_time.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} timing rows to {output_path.resolve()}")


if __name__ == "__main__":
    CONFIGURATIONS: List[Dict[str, Any]] = [
        {
            "id": "baseline",
            "dataset": {
                "n_per_cluster": 220,
                "std_class1": 0.45,
                "std_other": 0.8,
                "a": 3.0,
                "random_state": 0,
            },
            "model": {"n_estimators": 40, "random_state": 0, "n_jobs": -1},
            "deldel": {"segments_target": 120, "random_state": 0, "log_level": logging.WARNING},
            "change_point": {"enabled": False},
            "planes_weighted": {"prefer_cp": True, "weight_map": "softmax"},
            "planes_all_modes": {
                "mode": "C",
                "min_cluster_size": 10,
                "max_models_per_round": 6,
                "seed": 0,
            },
            "selection": {
                "max_k": 10,
                "min_improve": 1e-3,
                "feature_names": None,
                "dims_for_text": (0, 1),
                "min_region_size": 25,
                "min_abs_diff": 0.02,
                "min_rel_lift": 0.05,
            },
        },
        {
            "id": "more_clusters_high_detail",
            "dataset": {
                "n_per_cluster": 320,
                "std_class1": 0.40,
                "std_other": 0.75,
                "a": 3.25,
                "random_state": 1,
            },
            "model": {"n_estimators": 60, "max_depth": 12, "random_state": 1, "n_jobs": -1},
            "deldel": {"segments_target": 180, "random_state": 1, "log_level": logging.WARNING},
            "change_point": {"enabled": True, "mode": "generic", "base_samples": 64},
            "planes_weighted": {"prefer_cp": False, "weight_map": "uniform"},
            "planes_all_modes": {
                "mode": "B",
                "min_cluster_size": 12,
                "max_models_per_round": 8,
                "seed": 4,
            },
            "selection": {
                "max_k": 12,
                "min_improve": 5e-4,
                "feature_names": None,
                "dims_for_text": (0, 2),
                "min_region_size": 35,
                "min_abs_diff": 0.015,
                "min_rel_lift": 0.04,
            },
        },
        {
            "id": "faster_setup",
            "dataset": {
                "n_per_cluster": 160,
                "std_class1": 0.55,
                "std_other": 0.9,
                "a": 2.75,
                "random_state": 7,
            },
            "model": {"n_estimators": 30, "max_depth": 10, "random_state": 7, "n_jobs": -1},
            "deldel": {"segments_target": 90, "random_state": 7, "log_level": logging.WARNING},
            "change_point": {"enabled": False},
            "planes_weighted": {"prefer_cp": True, "weight_map": "softmax"},
            "planes_all_modes": {
                "mode": "C",
                "min_cluster_size": 8,
                "max_models_per_round": 4,
                "seed": 11,
            },
            "selection": {
                "max_k": 8,
                "min_improve": 2e-3,
                "feature_names": None,
                "dims_for_text": (0, 1),
                "min_region_size": 20,
                "min_abs_diff": 0.03,
                "min_rel_lift": 0.06,
            },
        },
    ]

    main(CONFIGURATIONS)
